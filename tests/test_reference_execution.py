"""Real SQLite/planner/DispatchMixin transactions with inert worker handles."""
from copy import deepcopy
from datetime import datetime,timezone
import json
from types import SimpleNamespace

import pytest

from localml_scheduler.domain import JobStatus,TrainingJob,PackingSpec,ResourceRequirements
from localml_scheduler.hardware import HardwareProfile
from localml_scheduler.scheduler.reference_execution import ReferenceExecutionMixin
from localml_scheduler.scheduler import reference_execution as module
from localml_scheduler.scheduler.continuation_forecast import reference_continuation,forecast_continuation
from localml_scheduler.scheduler.dispatching import DispatchMixin
from localml_scheduler.scheduler.service_state import ActiveRun
from localml_scheduler.tests.test_cold_start_trials import setup


def iso(v):return datetime.fromtimestamp(v,timezone.utc).isoformat()


class Base(DispatchMixin):
    def _runnable_jobs(self):return self.store.runnable_jobs()
    def _dispatch_pending_work(self):self.online_ticks+=1
    def _known_colocation_rejection(self,plan):return False
    def _prepare_colocation_trial(self,plan,job):
        assert plan.trial_metadata['requires_live_trial'] is False
        self.trials_prepared.append(plan)
        return None
    def _cancel_prepared_colocation_trial(self,*a,**kw):raise AssertionError('not a trial')
    def _preload_job_baseline(self,job):pass
    def _log_run_group_open(self,*a,**kw):pass


class Service(ReferenceExecutionMixin,Base):
    def _v5_snapshot_context(self,profiles):
        return {'evidence_identity':json.dumps({'profiles':profiles,'hardware':self.store.hardware_key(),
            'backend':self.settings.gpu_scheduler.packing_backend},sort_keys=True),
            'membership_generation':sorted((run.group_id,run.opened_at,list(run.job_ids))for run in self._active_runs.values()),
            'snapshot_at':self.clock.now}


def fixture(tmp_path,monkeypatch):
    settings,store,planner,jobs=setup(tmp_path)
    store._hardware_profile=HardwareProfile('fixture-host','linux','NVIDIA A10',24000,'8.6',None,'fixture')
    settings.gpu_scheduler.enabled=True
    settings.gpu_scheduler.parallel_job_cap=None
    clock=SimpleNamespace(now=1_800_000_000.)
    monkeypatch.setattr(module,'time',SimpleNamespace(time=lambda:clock.now,monotonic=lambda:clock.now))
    profiles={j.job_id:{'kind':'predicted_workload','artifact_sha256':'frozen','hardware':'A10',
        'job_signature':j.job_id,'batch_size':64,'step_cost':1000.,'epoch_cost':100000.,
        'avg_vram_mib':500.,'peak_vram_mib':1000.}for j in jobs}
    for i,j in enumerate(jobs):
        actual=store.get_job(j.job_id);actual.status=JobStatus.READY;actual.submitted_at=iso(clock.now-100+i)
        actual.metadata.update(runtime_global_step=0,runtime_steps_per_epoch=100,
            runtime_planned_epochs=3,runtime_total_optimizer_updates=300,
            pilot_frozen_workload_sha256='frozen-'+j.job_id)
        store.save_job(actual)
    planner.estimator.predicted_profile=lambda job,*a,**kw:deepcopy(profiles[job.job_id])
    planner.estimator.estimate_peak_vram_mb=lambda job,*a,**kw:profiles[job.job_id]['peak_vram_mib']
    planner.estimator.safe_budget_mb=lambda:5000.
    s=Service();s.settings=settings;s.store=store;s.planner=planner;s.clock=clock;s.profiles=profiles
    s.v5_options=SimpleNamespace(continuation=True,deferral=False,evidence_ttl_seconds=30.)
    s._v5_reference_state=None;s._v5_reference_restored=False;s._v5_rate_cache={}
    s._active_runs={};s.active=[];s.exited={};s.plans=[];s.events=[];s.trials_prepared=[];s.online_ticks=0
    s._colocation_trial=None;s._bracket_bootstrap=None;s._device_samples=[];s._admission_gate=SimpleNamespace(is_open=True)
    s._next_gpu_dispatch_attempt_at=0.;s.live_memory=0.
    s._active_vram_occupancy=lambda:s.live_memory
    s.event_logger=SimpleNamespace(emit=lambda event,**kw:s.events.append((event,kw)))
    s.logger=SimpleNamespace(info=lambda *a,**kw:None,warning=lambda *a,**kw:None)
    def dispatch(selected,**kw):
        s.plans.append((tuple(j.job_id for j in selected),kw))
        s.active.extend(j.job_id for j in selected)
        return SimpleNamespace(can_run=True,group_id='actual-run-'+str(len(s.plans)))
    s.supervisor=SimpleNamespace(active_job_ids=lambda:list(s.active),dispatch=dispatch,observed_exit_code=lambda name:s.exited.get(name),
        available_backends=lambda:{'cuda_process':True},active_job_ids_by_group=lambda:
            {k:[j for j in r.job_ids if j in s.active]for k,r in s._active_runs.items()})
    return s


def begin(s,decision='accepted',trial=None):
    ids=['a','b','c'];result={'decision':decision,'error':None,
        'frozen_profiles':deepcopy(s.profiles),'remaining_updates':dict.fromkeys(ids,300),
        'halves':[{a+'_trace':{'status':'complete'}for a in ('admit','defer')}for _ in range(2)]}
    for h in result['halves']:
        for a in ('admit','defer'):h[a]={'job_completions':dict.fromkeys(ids,s.clock.now+300)}
    trial=trial or SimpleNamespace(trial_id='actual-trial',preexisting_job_ids=('a',),candidate_job_id='b')
    assert s._v5_reference_begin([s.store.get_job(j)for j in ids],result,trial)
    return result


def cache(s,group,*,age=0):
    ids=tuple(sorted(group));ctx=s._v5_reference_context()
    s._v5_rate_cache[ids]={'context':ctx,'halves':[{'rates':dict.fromkeys(ids,1.),
        'identity':ctx['evidence_identity'],'membership_generation':ctx['membership_generation'],
        'observed_at':s.clock.now-age,'max_age_seconds':30.}for _ in range(2)]}


def resident(s,name):
    s.active.append(name);s.store.update_job(name,status=JobStatus.RUNNING)
    s._active_runs['resident-'+name]=ActiveRun(group_id='resident-'+name,mode='stack_anchor',
        backend_name='cuda_process',job_ids=(name,),opened_at=iso(s.clock.now-10),hardware_key='fixture-host')


def finish(s,name,*,receipt=True):
    s.active.remove(name);s.exited[name]=0
    for key in list(s._active_runs):
        if name in s._active_runs[key].job_ids:del s._active_runs[key]
    s.store.update_job(name,status=JobStatus.COMPLETED,
        metadata_updates={'feedback_ready_at':iso(s.clock.now)if receipt else None,
            'runtime_global_step':300})


def test_real_dispatch_obeys_same_selector_known_group_and_no_trial(tmp_path,monkeypatch):
    s=fixture(tmp_path,monkeypatch);begin(s);resident(s,'a');cache(s,('a','b'));cache(s,('a','b','c'))
    ids=reference_continuation(['b','c'],('a',),memory=dict.fromkeys('abc',1000),memory_capacity=5000,
        placement_allowed=s._v5_placement_allowed)['selected']
    assert ids==('b','c')
    s._dispatch_pending_work()
    assert s.plans[0][0]==ids
    assert s.store.get_job('a').status==JobStatus.RUNNING
    assert all(not p.trial_metadata['requires_live_trial']for p in s.trials_prepared)
    assert s.store.get_job('b').metadata['placement_trial_metadata']['reference_execution']is True
    assert s.online_ticks==0


def test_unknown_group_drains_then_solo_fifo_matches_forecast(tmp_path,monkeypatch):
    s=fixture(tmp_path,monkeypatch);begin(s);resident(s,'a')
    rollout=forecast_continuation(dict.fromkeys('abc',3),('a',),dict.fromkeys('abc',1),['b','c'],
        memory=dict.fromkeys('abc',1000),memory_capacity=5000,placement_allowed=s._v5_placement_allowed)
    assert rollout['status']=='complete'
    assert [(e['job'],e['at'])for e in rollout['events']if e['event']=='start']==[('b',3),('c',6)]
    s._dispatch_pending_work();assert not s.plans
    finish(s,'a');s._dispatch_pending_work();assert s.plans[-1][0]==('b',)
    s._dispatch_pending_work();assert len(s.plans)==1
    finish(s,'b');s._dispatch_pending_work();assert s.plans[-1][0]==('c',)


@pytest.mark.parametrize('kind',['blocked','failed','retry','solo_required','exhausted'])
def test_failed_group_and_retry_guards_never_free_probe(tmp_path,monkeypatch,kind):
    s=fixture(tmp_path,monkeypatch);begin(s);resident(s,'a');cache(s,('a','b'))
    key=s._v5_reference_group_key([s.store.get_job('a'),s.store.get_job('b')])
    metadata={}
    if kind=='blocked':metadata['cold_start_blocked_groups']=[key]
    if kind=='failed':metadata['cold_start_group_attempts']={key:{'failures':1,'retry_after':s.clock.now-100}}
    if kind=='retry':metadata['cold_start_retry_after']=s.clock.now+100
    if kind=='solo_required':metadata['cold_start_solo_required']=True
    if kind=='exhausted':metadata['cold_start_failed_attempts']=s.settings.gpu_scheduler.colocation.cold_start_max_attempts
    s.store.update_job('b',metadata_updates=metadata)
    assert not s._v5_placement_allowed(('a','b'))
    s._dispatch_pending_work();assert not s.plans
    assert s._colocation_trial is None


@pytest.mark.parametrize('change',['age','half','generation','profile','hardware'])
def test_cache_actual_time_exact_profiles_and_process_generation(tmp_path,monkeypatch,change):
    s=fixture(tmp_path,monkeypatch);begin(s);resident(s,'a');cache(s,('a','b'))
    r=s._v5_rate_cache[('a','b')]
    if change=='age':r['halves'][0]['observed_at']=s.clock.now-31
    if change=='half':r['halves'][0]['rates'].pop('b')
    if change=='generation':s._active_runs['resident-a'].opened_at=iso(s.clock.now)
    if change=='profile':s._v5_reference_state['profiles']['b']['step_cost']=2.
    if change=='hardware':s.store._hardware_profile.hardware_key='other-host'
    assert not s._v5_placement_allowed(('a','b'))


def test_configured_cap_and_live_memory_floor_not_bypassed(tmp_path,monkeypatch):
    s=fixture(tmp_path,monkeypatch);begin(s);resident(s,'a');cache(s,('a','b'))
    s.settings.gpu_scheduler.parallel_job_cap=1
    s._dispatch_pending_work();assert not s.plans
    s.settings.gpu_scheduler.parallel_job_cap=None;s.live_memory=4500.
    s._dispatch_pending_work();assert not s.plans
    s.live_memory=1000.;s._dispatch_pending_work();assert s.plans[0][0]==('b',)


def test_rejected_or_unknown_candidate_waits_for_pause_cleanup(tmp_path,monkeypatch):
    s=fixture(tmp_path,monkeypatch);begin(s,'inconclusive');resident(s,'a');resident(s,'b')
    assert s._v5_reference_state['chosen_action']=='defer'and not s._v5_reference_state['prediction_complete']
    s._colocation_trial=object();s._dispatch_pending_work();assert not s.plans
    s._colocation_trial=None;s.store.update_job('b',status=JobStatus.PAUSING)
    s._dispatch_pending_work();assert not s.plans
    s.active.remove('b');s.exited['b']=0;del s._active_runs['resident-b'];s.store.update_job('b',status=JobStatus.PAUSED)
    s._dispatch_pending_work();assert not s.plans #a drains, never retry rejected pack
    finish(s,'a');s._dispatch_pending_work();assert s.plans[0][0]==('b',)


def test_persist_restart_drops_caches_retains_order_and_reservations(tmp_path,monkeypatch):
    s=fixture(tmp_path,monkeypatch);begin(s);s._v5_reference_state['reservations']=['c'];s._v5_reference_save()
    cache(s,('a','b'));s._v5_reference_restored=False;s._v5_reference_state=None
    s._v5_reference_restore()
    assert not s._v5_rate_cache
    assert s._v5_reference_state['status']=='solo_drain_restored'
    assert s._v5_reference_state['order']==['a','b','c']
    s._dispatch_pending_work();assert s.plans[0][0]==('c',)


@pytest.mark.parametrize('change',['arrival','recipe','profile','submitted'])
def test_material_changes_explicitly_return_to_online_planning(tmp_path,monkeypatch,change):
    s=fixture(tmp_path,monkeypatch);begin(s)
    if change=='arrival':
        j=s.store.get_job('c').copy(job_id='new');s.store.save_job(j)
    if change=='recipe':s.store.update_job('b',metadata_updates={'pilot_frozen_workload_sha256':'changed'})
    if change=='profile':s.profiles['b']['peak_vram_mib']=1200.
    if change=='submitted':
        j=s.store.get_job('b');j.submitted_at=iso(s.clock.now);s.store.save_job(j)
    s._dispatch_pending_work()
    assert s.online_ticks==1 and s._v5_reference_state is None and not s.plans
    assert any(e=='v5_reference_invalidated'for e,_ in s.events)


def test_delivery_is_not_completion_and_full_cohort_clear_requires_receipts(tmp_path,monkeypatch):
    s=fixture(tmp_path,monkeypatch);begin(s)
    for name in ('a','b','c'):
        resident(s,name);finish(s,name,receipt=name!='c')
    s._dispatch_pending_work();assert s._v5_reference_state is not None
    s.store.update_job('c',metadata_updates={'feedback_ready_at':iso(s.clock.now)})
    s._dispatch_pending_work();assert s._v5_reference_state is None
    assert json.loads(s._v5_reference_path().read_text())['status']=='closed'


def test_shadow_disabled_never_activates_or_changes_dispatch(tmp_path,monkeypatch):
    s=fixture(tmp_path,monkeypatch);s.v5_options.continuation=False
    assert not s._v5_reference_begin(s.store.list_jobs(),{'decision':'accepted'},None)
    s._dispatch_pending_work();assert s.online_ticks==1 and not s._v5_reference_path().exists()


def test_admission_gate_and_hold_never_bypassed(tmp_path,monkeypatch):
    s=fixture(tmp_path,monkeypatch);begin(s);s._admission_gate.is_open=False
    s._dispatch_pending_work();assert not s.plans
    s._admission_gate.is_open=True;s.store.update_job('a',hold=True)
    s._dispatch_pending_work();assert s.plans[0][0]==('b',)


def test_original_reconciliation_deadline_survives_restart_then_fails_closed(tmp_path,monkeypatch):
    s=fixture(tmp_path,monkeypatch);resident(s,'a');begin(s,'inconclusive')
    deadline=s._v5_reference_state['reconciliation_deadline_at']
    # The old worker disappeared from a newly constructed supervisor: neither
    # absence nor restored RUNNING/RECOVERABLE state proves process release.
    s.active.clear();s._active_runs.clear()
    s._v5_reference_state=None;s._v5_reference_restored=False
    s.clock.now=deadline-1;s._dispatch_pending_work()
    assert not s.plans and s.online_ticks==0
    assert s._v5_reference_state['reconciliation_deadline_at']==deadline
    s.clock.now=deadline
    with pytest.raises(TimeoutError,match='original finite deadline'):s._dispatch_pending_work()
    assert s.v5_failure['kind']=='reference_cleanup_reconciliation_timeout'
    assert s.v5_failure['pending_job_ids']==['a']


def test_clean_release_receipt_unblocks_restored_solo_without_resetting_bound(tmp_path,monkeypatch):
    s=fixture(tmp_path,monkeypatch);resident(s,'a');begin(s,'inconclusive')
    deadline=s._v5_reference_state['reconciliation_deadline_at'];s.active.clear();s._active_runs.clear()
    s._v5_reference_restored=False;s._v5_reference_state=None
    s.store.update_job('a',status=JobStatus.PAUSED)
    s._dispatch_pending_work();assert not s.plans
    s.exited['a']=0;s._dispatch_pending_work()
    assert s.plans[0][0]==('a',)
    assert s._v5_reference_state['reconciliation_deadline_at']==deadline


def test_normal_running_job_uses_own_timeout_not_cleanup_bound(tmp_path,monkeypatch):
    s=fixture(tmp_path,monkeypatch);resident(s,'a');begin(s)
    s.clock.now=s._v5_reference_state['reconciliation_deadline_at']+1000
    s._dispatch_pending_work()
    assert not s.plans and not hasattr(s,'v5_failure')


def test_original_trial_cleanup_deadline_takes_precedence(tmp_path,monkeypatch):
    s=fixture(tmp_path,monkeypatch)
    trial=SimpleNamespace(trial_id='trial',preexisting_job_ids=('a',),candidate_job_id='b',cleanup_deadline_at=s.clock.now+5)
    begin(s,'inconclusive',trial)
    assert s._v5_reference_state['reconciliation_deadline_at']==s.clock.now+5
    s._colocation_trial=trial;s.clock.now+=5
    with pytest.raises(TimeoutError):s._dispatch_pending_work()
    assert s.v5_failure['kind']=='reference_cleanup_reconciliation_timeout'


def test_protected_known_packed_placement_respects_physical_authorization(tmp_path,monkeypatch):
    s=fixture(tmp_path,monkeypatch);begin(s);resident(s,'a');cache(s,('a','b'))
    s._v5_reference_state['reservations']=['b']
    s._v5_deferral_legal_placement=lambda job,plan:False
    assert not s._v5_placement_allowed(('a','b'))
    assert s._v5_reference_last_block=='protected_service_placement'
    s._dispatch_pending_work();assert not s.plans
    finish(s,'a');s._dispatch_pending_work();assert s.plans[0][0]==('b',)


def test_repeated_authoritative_score_cannot_reset_same_trial_cleanup_bound(tmp_path,monkeypatch):
    s=fixture(tmp_path,monkeypatch);begin(s,'inconclusive')
    deadline=s._v5_reference_state['reconciliation_deadline_at'];begun=s._v5_reference_state['begun_at']
    s.clock.now+=50;begin(s,'inconclusive')
    assert s._v5_reference_state['reconciliation_deadline_at']==deadline
    assert s._v5_reference_state['begun_at']==begun


def owned_resume_with_real_supervisor(tmp_path,monkeypatch):
    """Real ownership/dispatch/reaping, with inert processes and no GPU backend."""
    from threading import Condition
    from types import MethodType
    from localml_scheduler.execution.executor import WorkerProcessHandle
    from localml_scheduler.scheduler.supervisor import WorkerSupervisor
    from localml_scheduler.scheduler.run_tracking import RunTrackingMixin
    s=fixture(tmp_path,monkeypatch)
    class Process:
        def __init__(self):self.returncode=None;self.answers=[]
        def poll(self):
            if self.answers:self.returncode=self.answers.pop(0)
            return self.returncode
    class Backend:
        def __init__(self):self.launches=[];self.processes={}
        def available(self):return True
        def launch(self,jobs):
            if self.launches:
                assert s.supervisor.observed_exit_code('a')==0,'refill requires actual current-launch exit proof'
            self.launches.append(tuple(j.job_id for j in jobs))
            handles=[]
            for job in jobs:
                process=Process();self.processes[job.job_id]=process
                handles.append(WorkerProcessHandle(job.job_id,process,tmp_path/'unused.stdout',tmp_path/'unused.stderr'))
            return handles
    backend=Backend()
    s.supervisor=WorkerSupervisor(s.settings,store=s.store,backends={'cuda_process':backend})
    s._completion_condition=Condition()
    s._exclusive_probe_job_id=None
    s.log_store=SimpleNamespace(close_run_group=lambda **kw:None)
    s._record_solo_profiles=lambda *a:None
    s._record_combination_profiles=lambda *a:None
    s._finalize_pending_pattern=lambda:None
    s._handle_worker_exit=MethodType(RunTrackingMixin._handle_worker_exit,s)
    poll=MethodType(RunTrackingMixin._poll_active_workers,s)
    s.poll_ticks=0
    def observed_poll():
        s.poll_ticks+=1
        return poll()
    s._poll_active_workers=observed_poll
    # The trial's earlier cooperative pause exited cleanly. A later normal
    # reference resume must discard that old-generation exit receipt.
    s.supervisor._record_process_exit('a',0)
    begin(s,'inconclusive')
    deadline=s._v5_reference_state['reconciliation_deadline_at']
    s.clock.now=deadline+500
    s._dispatch_pending_work()
    assert backend.launches==[('a',)]
    assert s.supervisor.observed_exit_code('a')is None
    assert not s.supervisor.has_reaped_process('a')
    return s,backend,deadline


@pytest.mark.parametrize('returncode',[0,2])
def test_resumed_generation_exit_between_tick_poll_and_active_query_is_authoritatively_reaped(tmp_path,monkeypatch,returncode):
    s,backend,deadline=owned_resume_with_real_supervisor(tmp_path,monkeypatch)
    process=backend.processes['a']
    # The normal tick observes a still-live process. Its next active-ID query
    # observes actual exit, before WorkerSupervisor.poll records that exit.
    s.store.update_job('a',status=JobStatus.COMPLETED,metadata_updates={'runtime_global_step':300})
    process.answers=[None,None,returncode]
    s._poll_active_workers()
    assert s.supervisor.observed_exit_code('a')is None
    previous_ticks=s.poll_ticks
    if returncode==0:
        s._dispatch_pending_work()
        assert backend.launches==[('a',),('b',)]
        assert not hasattr(s,'v5_failure')
    else:
        with pytest.raises(TimeoutError,match='original finite deadline'):s._dispatch_pending_work()
        assert backend.launches==[('a',)]
    assert s.poll_ticks==previous_ticks+1
    assert s.supervisor.observed_exit_code('a')==returncode
    assert s.supervisor.has_reaped_process('a')
    assert s._v5_reference_state['reconciliation_deadline_at']==deadline


def test_live_unreaped_owned_handle_cannot_turn_absence_into_success(tmp_path,monkeypatch):
    s,backend,deadline=owned_resume_with_real_supervisor(tmp_path,monkeypatch)
    worker=next(iter(s.supervisor.active_groups().values())).workers['a']
    worker.handle.monitor_via_store=True
    s.store.update_job('a',status=JobStatus.COMPLETED,metadata_updates={'runtime_global_step':300})
    assert s.supervisor.has_live_process('a')
    assert s.supervisor.active_job_ids()==[]
    with pytest.raises(TimeoutError,match='original finite deadline'):s._dispatch_pending_work()
    assert s.supervisor.observed_exit_code('a')is None
    assert not s.supervisor.has_reaped_process('a')
    assert backend.launches==[('a',)]
    assert s._v5_reference_state['reconciliation_deadline_at']==deadline


def test_old_pause_receipt_cannot_release_resumed_generation_after_supervisor_loss(tmp_path,monkeypatch):
    s,backend,deadline=owned_resume_with_real_supervisor(tmp_path,monkeypatch)
    s.supervisor._groups.clear();s._active_runs.clear()
    s._v5_reference_state=None;s._v5_reference_restored=False
    with pytest.raises(TimeoutError,match='original finite deadline'):s._dispatch_pending_work()
    assert s.supervisor.observed_exit_code('a')is None
    assert backend.launches==[('a',)]
    assert s.v5_failure['pending_job_ids']==['a']
    assert s._v5_reference_state['reconciliation_deadline_at']==deadline
