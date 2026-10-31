from types import SimpleNamespace
from datetime import datetime,timezone
import pytest
from localml_scheduler.domain import TrainingJob,JobStatus
from localml_scheduler.scheduler.v5_lifecycle import forecast_cohort,snapshot
from localml_scheduler.scheduler.continuation_forecast import compare_feedback_admission
from localml_scheduler.scheduler.feedback_objective import CompletionCosts,FeedbackRequest


def make(name,status=JobStatus.COMPLETED,step=64):
 j=TrainingJob.create('fixture:runner','fixture','/tmp/unused',job_id=name)
 j.status=status;j.submitted_at=datetime.fromtimestamp(1,timezone.utc).isoformat();j.finished_at=datetime.fromtimestamp(9,timezone.utc).isoformat()
 j.max_steps=64;j.max_epochs=2;j.config.max_epochs=2
 j.metadata.update(runtime_global_step=step,runtime_steps_per_epoch=32,runtime_total_optimizer_updates=64,runtime_planned_epochs=2,trial_runner_ready=True)
 return j

def service(jobs,active=(),code=None):
 return SimpleNamespace(store=SimpleNamespace(list_jobs=lambda:jobs),
  supervisor=SimpleNamespace(active_job_ids=lambda:list(active),observed_exit_code=lambda j:code),
  settings=SimpleNamespace(gpu_scheduler=SimpleNamespace(feedback_latency=SimpleNamespace(allow_zero_overhead=True))))

def test_all_terminal_worker_status_does_not_remove_undelivered_request():
 j=make('a');s=service([j],active=['a'])
 cohort,requests,done=forecast_cohort(s)
 assert len(cohort)==len(requests)==1 and done=={}
 phase=snapshot(s,cohort)
 assert phase['remaining']=={'a':0}and phase['phase_state']['a']['phase']=='finishing'
 assert phase['passive']==()


def test_delivering_requires_actual_successful_exit_not_finished_timestamp():
 j=make('a');s=service([j])
 with pytest.raises(ValueError,match='exit proof'):snapshot(s,[j])
 s.supervisor.observed_exit_code=lambda j:0
 result=snapshot(s,[j]);assert result['phase_state']['a']['phase']=='delivering'and result['passive']==('a',)
 assert len(forecast_cohort(s)[1])==1
 j.metadata['feedback_ready_at']=datetime.fromtimestamp(12,timezone.utc).isoformat()
 assert forecast_cohort(s)==([],(),{})


def test_failed_terminal_job_is_not_successful_feedback():
 with pytest.raises(ValueError,match='failed or was cancelled'):forecast_cohort(service([make('a',JobStatus.FAILED)]))


def test_passive_delivery_scored_in_both_alternatives_without_gpu_occupancy():
 requests=[FeedbackRequest(j,'w','individual',(j,),0)for j in('a','b','delivered-later')]
 result=compare_feedback_admission(remaining={'a':20,'b':10,'delivered-later':0},solo={'a':1,'b':1,'delivered-later':1},
  incumbents=('a',),candidate='b',queue=('b',),passive=('delivered-later',),
  pre_halves={'a':[1,1]},packed_halves={'a':[.8,.8],'b':[.8,.8]},requests=requests,completed={},now=10,
  costs={'a':CompletionCosts(),'b':CompletionCosts(),'delivered-later':CompletionCosts(delivery=7)},
  phase_state={'a':{'phase':'training'},'b':{'phase':'training'},'delivered-later':{'phase':'delivering'}},
  memory={'a':1,'b':1,'delivered-later':1000},memory_capacity=2)
 assert result['error']is None,result
 for half in result['halves']:
  for action in('admit','defer'):
   assert next(r for r in half[action]['requests'] if r['request_id']=='delivered-later')['completion_at']==17


def test_old_pre_add_windows_are_not_refreshed_at_scoring():
 requests=[FeedbackRequest(j,'w','individual',(j,),0)for j in('a','b')]
 result=compare_feedback_admission(remaining={'a':20,'b':10},solo={'a':1,'b':1},incumbents=('a',),candidate='b',queue=('b',),
  pre_halves={'a':[1,1]},packed_halves={'a':[.8,.8],'b':[.8,.8]},requests=requests,completed={},now=100,
  context={'snapshot_at':100,'pre_observed_at':50,'packed_observed_at':100,'trial_max_age_seconds':30})
 assert result['decision']=='inconclusive'and 'stale'in result['error']


def test_finish_barrier_survives_restart_and_keeps_original_trial_deadline(tmp_path,monkeypatch):
 from localml_scheduler.tests.test_bracket_bootstrap import fixture,iso
 from localml_scheduler.scheduler.experimental_v5 import V5Service,V5Options
 from localml_scheduler.scheduler import experimental_v5
 original=fixture(tmp_path,monkeypatch,count=2)
 active=['a','b'];reaped=set()
 supervisor=SimpleNamespace(active_job_ids=lambda:list(active),has_reaped_process=lambda j:j in reaped,has_live_process=lambda j:j in active)
 s=V5Service(original.settings,store=original.store,supervisor=supervisor,v5_options=V5Options(bracket=True))
 s.store.update_job('a',metadata_updates={'runtime_global_step':300})
 clock=SimpleNamespace(now=original.clock.now)
 monkeypatch.setattr(experimental_v5,'time',SimpleNamespace(time=lambda:clock.now))
 trial=SimpleNamespace(preexisting_job_ids=('a',),candidate_job_id='b',trial_id='old-trial',phase='measure_overlap',started_at=iso(clock.now-20))
 for name in ('a','b'):
  s.store.update_job(name,status=JobStatus.RUNNING,metadata_updates={'trial_step_command':{'owner':'old-trial','kind':'measure_overlap','action':'measure','token':'t','expires_at':clock.now+5}})
 s._colocation_trial=trial;s._evaluate_cold_start_trial(trial)
 deadline=clock.now-20+s.settings.gpu_scheduler.colocation.cold_start_trial_timeout_seconds
 assert s._v5_finish_transition['deadline']==deadline and s._colocation_trial is None
 restored=V5Service(original.settings,store=original.store,supervisor=supervisor,v5_options=V5Options(bracket=True))
 assert restored._v5_finish_transition['deadline']==deadline
 # Losing an active ID without observed reaping is not evidence of GPU release.
 active.remove('a');restored._dispatch_pending_work();assert restored._v5_finish_transition
 clock.now=deadline+1
 with pytest.raises(TimeoutError,match='resource release'):restored._dispatch_pending_work()
 assert restored.v5_failure['pending_ids']==['a']


def test_add_only_unknown_surviving_pair_is_bounded_and_only_unfinished_jobs_pause(tmp_path,monkeypatch):
 from localml_scheduler.tests.test_bracket_bootstrap import fixture,iso
 from localml_scheduler.scheduler.experimental_v5 import V5Service,V5Options
 from localml_scheduler.scheduler import experimental_v5
 original=fixture(tmp_path,monkeypatch,count=3)
 active=['a','b','c'];reaped=set();pauses=[]
 supervisor=SimpleNamespace(active_job_ids=lambda:list(active),has_reaped_process=lambda j:j in reaped,has_live_process=lambda j:j in active,
  request_pause=lambda j,**kw:pauses.append(j)or True)
 s=V5Service(original.settings,store=original.store,supervisor=supervisor,v5_options=V5Options(bracket=True))
 clock=SimpleNamespace(now=original.clock.now);monkeypatch.setattr(experimental_v5,'time',SimpleNamespace(time=lambda:clock.now))
 trial=SimpleNamespace(preexisting_job_ids=('a','b'),candidate_job_id='c',trial_id='old',phase='measure_overlap',started_at=iso(clock.now-20))
 for name in active:
  s.store.update_job(name,status=JobStatus.RUNNING,metadata_updates={'runtime_global_step':300 if name=='a' else 20,
   'trial_step_command':{'owner':'old','kind':'measure_overlap','action':'measure','token':'t','expires_at':clock.now+5}})
 s._colocation_trial=trial;s._evaluate_cold_start_trial(trial);deadline=s._v5_finish_transition['deadline']
 active.remove('a');reaped.add('a');s._dispatch_pending_work()
 assert s._v5_finish_transition and pauses==[]
 clock.now=deadline+.1;s._dispatch_pending_work()
 assert pauses==['b','c']and s._v5_finish_transition['phase']=='cleanup'
 assert s.store.get_job('a').metadata['runtime_global_step']==300
 assert all(s.store.get_job(n).metadata['runtime_global_step']==20 for n in('b','c'))
 clock.now=s._v5_finish_transition['cleanup_deadline']+.1
 with pytest.raises(TimeoutError,match='cleanup'):s._dispatch_pending_work()


def test_cold_phase_keeps_existing_measuring_keyword(monkeypatch):
 from localml_scheduler.scheduler.experimental_v5 import V5Service,V5Options
 from localml_scheduler.scheduler.cold_start import ColdStartTrialMixin
 calls=[]
 monkeypatch.setattr(ColdStartTrialMixin,'_cold_phase',lambda self,t,p,j,*,measuring=None:calls.append(measuring))
 s=object.__new__(V5Service);s.v5_options=V5Options()
 s._cold_phase(None,'prepare',[],measuring=['explicit'])
 assert calls==[['explicit']]


def test_incomplete_cohort_during_scoring_stays_inconclusive(monkeypatch):
 from localml_scheduler.scheduler import experimental_v5 as module
 from localml_scheduler.scheduler.experimental_v5 import V5Service,V5Options
 s=object.__new__(V5Service);s.v5_options=V5Options(continuation=True)
 s._v5_forecast_trial=lambda *a:{'decision':'inconclusive','error':'unknown cohort'}
 events=[];s.event_logger=SimpleNamespace(emit=lambda *a,**kw:events.append((a,kw)))
 monkeypatch.setattr(module,'forecast_cohort',lambda *_:(_ for _ in()).throw(ValueError('unready sealed member')))
 trial=SimpleNamespace(candidate_job_id='j',trial_id='trial')
 result=s._score_feedback_admission(trial,[],{})
 assert result['decision']=='inconclusive'and 'unready sealed member'in result['error']
 assert any(e[0][0]=='v5_reference_installation_unavailable'for e in events)


def test_accepted_score_downgrades_when_reference_install_refuses_changed_cohort(monkeypatch):
 from localml_scheduler.scheduler import experimental_v5 as module
 from localml_scheduler.scheduler.experimental_v5 import V5Service,V5Options
 s=object.__new__(V5Service);s.v5_options=V5Options(continuation=True)
 s._v5_forecast_trial=lambda *a:{'decision':'accepted','error':None}
 s._v5_reference_begin=lambda *a:False
 events=[];s.event_logger=SimpleNamespace(emit=lambda *a,**kw:events.append((a,kw)))
 monkeypatch.setattr(module,'forecast_cohort',lambda *_:([],[],{}))
 result=s._score_feedback_admission(SimpleNamespace(candidate_job_id='j',trial_id='trial'),[],{})
 assert result['decision']=='inconclusive'and 'changed before reference installation'in result['error']
 assert any(e[0][0]=='v5_reference_installation_unavailable'for e in events)


@pytest.mark.parametrize('command_present', [False, True])
def test_finisher_before_candidate_start_releases_real_cooperative_control(tmp_path, monkeypatch, command_present):
 from localml_scheduler.tests.test_bracket_bootstrap import fixture, iso
 from localml_scheduler.scheduler.experimental_v5 import V5Service, V5Options
 from localml_scheduler.domain import SafePointType
 from localml_scheduler.execution import trial_control
 original = fixture(tmp_path, monkeypatch, count=2)
 now = original.clock.now
 supervisor = SimpleNamespace(active_job_ids=lambda: ['a'])
 service = V5Service(original.settings, store=original.store, supervisor=supervisor, v5_options=V5Options(bracket=True))
 service.store.update_job('a', status=JobStatus.RUNNING, metadata_updates={'runtime_global_step': 300})
 published = {'trial_id': 'retired-prepare', 'cold_start': {'admission_policy': 'test'},
  'phase': 'prepare', 'decision': 'pending'}
 metadata = {'colocation_trial': published}
 if command_present:
  metadata['trial_step_command'] = {'owner': 'retired-prepare', 'action': 'yield', 'kind': 'prepare',
   'token': 'old-yield', 'expires_at': now+5}
 service.store.update_job('b', metadata_updates=metadata)
 trial = SimpleNamespace(preexisting_job_ids=('a',), candidate_job_id='b', trial_id='retired-prepare',
  phase='prepare', started_at=iso(now))
 service._colocation_trial = trial
 service._evaluate_cold_start_trial(trial)
 candidate = service.store.get_job('b')
 assert candidate.metadata['colocation_trial']['decision'] == 'completed_unverified'
 assert candidate.metadata['runtime_global_step'] == 0
 assert not candidate.metadata.get('trial_step_command')
 assert service._colocation_trial is None
 # Run the actual shared safe-point implementation. A stale prepare barrier
 # would sleep for 120 seconds per update; fail immediately if it tries to wait.
 clock = SimpleNamespace(now=now)
 def unexpected_sleep(_):
  pytest.fail('retired prepare metadata blocked productive optimizer progress')
 monkeypatch.setattr(trial_control, 'time', SimpleNamespace(time=lambda: clock.now,
  monotonic=lambda: clock.now, perf_counter=lambda: clock.now, sleep=unexpected_sleep))
 class Control(trial_control.StepTrialControlMixin):
  pass
 hook = Control()
 hook.job, hook.store, hook.event_logger = candidate, service.store, service.event_logger
 hook.control_plane = SimpleNamespace(settings=original.settings,
  read_command=lambda _: SimpleNamespace(action='run'))
 context = SimpleNamespace(store=service.store, control_hook=hook)
 for step in range(4):
  clock.now += 1
  assert context.control_hook._cooperative_step(SafePointType.STEP, epoch=0,
   global_step=step, steps_per_epoch=100, metrics={})
 assert context.store.get_job('b').metadata['runtime_global_step'] == 3
 assert not context.store.get_job('b').metadata.get('trial_lease_expired')
 # The same survivor must be able to supply the next admission's pre-add
 # measurements, rather than yielding under its abandoned candidate metadata.
 context.store.update_job('b', metadata_updates={'trial_step_command': {'owner': 'next-trial',
  'token': 'next-measure', 'action': 'measure', 'kind': 'pre_add', 'candidate': False,
  'expires_at': clock.now+100, 'window_min_seconds': 1}})
 for step in range(4, 15):
  clock.now += 1
  context.control_hook._cooperative_step(SafePointType.STEP, epoch=0,
   global_step=step, steps_per_epoch=100, metrics={})
 measured = context.store.get_job('b').metadata['trial_step_window']
 assert measured['owner'] == 'next-trial' and measured['kind'] == 'pre_add'
 assert measured['complete'] and measured['steps'] >= 8 and len(measured['halves']) == 2


@pytest.mark.parametrize('published_owner,command_owner', [
 ('retired', 'newer'), ('newer', 'retired'), ('newer', 'newer'), ('retired', None)])
def test_finishing_retires_only_matching_published_trial_and_command_owners(tmp_path, monkeypatch, published_owner, command_owner):
 from localml_scheduler.tests.test_bracket_bootstrap import fixture, iso
 from localml_scheduler.scheduler.experimental_v5 import V5Service, V5Options
 original = fixture(tmp_path, monkeypatch, count=2)
 now = original.clock.now
 service = V5Service(original.settings, store=original.store,
  supervisor=SimpleNamespace(active_job_ids=lambda: ['a']), v5_options=V5Options(bracket=True))
 service.store.update_job('a', status=JobStatus.RUNNING, metadata_updates={'runtime_global_step': 300})
 published = {'trial_id': published_owner, 'cold_start': True, 'phase': 'prepare', 'decision': 'pending'}
 command = {'owner': command_owner, 'token': 'owned-token', 'kind': 'prepare', 'action': 'yield', 'expires_at': now+5}
 service.store.update_job('b', metadata_updates={'colocation_trial': published,
  'trial_step_command': command if command_owner else {}})
 trial = SimpleNamespace(preexisting_job_ids=('a',), candidate_job_id='b', trial_id='retired',
  phase='prepare', started_at=iso(now))
 service._colocation_trial = trial
 service._evaluate_cold_start_trial(trial)
 metadata = service.store.get_job('b').metadata
 if published_owner == 'retired':
  assert metadata['colocation_trial']['decision'] == 'completed_unverified'
  assert metadata['colocation_trial']['phase'] == 'decide'
 else:
  assert metadata['colocation_trial'] == published
 assert metadata['trial_step_command'] == (command if command_owner == 'newer' else {})
