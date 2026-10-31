"""CPU policy/state regressions; no GPU workers or forecasting oracle."""
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
import json
import pytest
from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import JobStatus,TrainingJob
from localml_scheduler.storage.sqlite_store import SQLiteStateStore
from localml_scheduler.scheduler.planner_types import DispatchPlan
from localml_scheduler.scheduler.deferral_reservations import (
    KEY,DeferralReservationMixin,new_record,advance_record,mark_deferred,record_bypass,protected,choose_protected,
)

def iso(t):return datetime.fromtimestamp(t,timezone.utc).isoformat()
def job(store,name,submitted,status=JobStatus.READY,**metadata):
    value=TrainingJob.create('fixture:runner','fixture','/tmp/unused',job_id=name,metadata=metadata)
    value.submitted_at=iso(submitted);value.status=status;store.save_job(value);return value

def defer(store,value,now=0,reason='rejected'):
    record=mark_deferred(new_record(value.submitted_at,now),now=now,runnable=True,reason=reason,failed_group_id='failed-group')
    store.update_job(value.job_id,metadata_updates={KEY:record});return record

def plan(value,trial=False,**breakdown):return DispatchPlan(mode='stack_anchor',backend_name='cuda_process',job_ids=(value.job_id,),reason='fixture',trial_metadata={'requires_live_trial':trial},objective_breakdown=breakdown)

class Base:
    def __init__(self,store,enabled=True):
        self.store=store;self.clock=0.;self.active=[];self.dispatched=[];self._active_runs={};self.events=[];self.next_launch=0
        self.v5_options=SimpleNamespace(deferral=enabled,deferral_seconds=30.,max_younger_bypasses=2)
        self.supervisor=SimpleNamespace(active_job_ids=lambda:list(self.active))
        self.event_logger=SimpleNamespace(emit=lambda name,**kw:self.events.append((name,kw)))
    def _runnable_jobs(self):return self.store.runnable_jobs()
    def _dispatch_plan(self,proposal):
        # Represent the existing final compatibility/memory/retry gate, not a GPU.
        if proposal.objective_breakdown.get('illegal_memory_or_retry'):return False
        self.next_launch+=1;run=SimpleNamespace(group_id=str(self.next_launch),opened_at=iso(self.clock),job_ids=proposal.job_ids)
        self._active_runs[run.group_id]=run
        for name in proposal.job_ids:self.store.update_job(name,status=JobStatus.RUNNING);self.active.append(name)
        self.dispatched.append(proposal);return True
    def _finish_cold_start_trial(self,trial,decision,reason,result=None):
        self.store.update_job(trial.candidate_job_id,status=JobStatus.PAUSED,metadata_updates={'runtime_global_step':5,'cold_start_solo_required':True})
        return 'base-result'
class Service(DeferralReservationMixin,Base):
    def _v5_deferral_now(self):return self.clock
    def _v5_deferral_block_reason(self,value):return value.metadata.get('fixture_global_block')

@pytest.fixture
def store(tmp_path):return SQLiteStateStore(SchedulerSettings(runtime_root=tmp_path))


def test_206_second_ordering_drains_without_preempting_then_serves_reserved_anchor(store):
    old=job(store,'old',0);young=job(store,'child',10);resident=job(store,'resident',1,JobStatus.RUNNING)
    defer(store,old);service=Service(store);service.active=[resident.job_id];service.clock=206
    assert service._dispatch_plan(plan(young))is False
    assert service._dispatch_plan(plan(old))is False
    assert store.get_job(resident.job_id).status==JobStatus.RUNNING and not service.dispatched
    service.active=[];store.update_job(resident.job_id,status=JobStatus.COMPLETED)
    assert service._dispatch_plan(plan(old))is True
    record=store.get_job(old.job_id).metadata[KEY]
    assert record['deferred_seconds']==206 and not record['pending'] and record['served_launch_ids']
    assert store.get_job(old.job_id).submitted_at==iso(0)


def test_two_actual_younger_child_launches_reserve_next_turn_and_deduplicate(store):
    old=job(store,'old',0);child=job(store,'actual-child',1);defer(store,old);service=Service(store)
    assert service._dispatch_plan(plan(child))
    record=store.get_job(old.job_id).metadata[KEY];token=record['bypass_launch_ids'][0]
    assert record_bypass(record,launched_submitted_at=child.submitted_at,launch_id=token)==record
    service.active=[];store.update_job(child.job_id,status=JobStatus.PAUSED)
    assert service._dispatch_plan(plan(child))
    record=store.get_job(old.job_id).metadata[KEY]
    assert len(record['bypass_launch_ids'])==2 and len(set(record['bypass_launch_ids']))==2
    service.active=[];store.update_job(child.job_id,status=JobStatus.PAUSED)
    assert not service._dispatch_plan(plan(child))
    assert service._dispatch_plan(plan(old))
    assert not any(name.startswith('future')for name in record['bypass_launch_ids'])


def test_restart_sqlite_persistence_and_trial_updates_do_not_erase_debt(store):
    old=job(store,'old',0);defer(store,old);s=Service(store);s.clock=31;s._runnable_jobs()
    before=store.get_job(old.job_id).metadata[KEY]
    store.update_job(old.job_id,metadata_updates={'runtime_global_step':5})
    restored=SQLiteStateStore(store.settings);restart=Service(restored);restart.clock=206
    assert restart._runnable_jobs()[0].job_id==old.job_id
    after=restored.get_job(old.job_id).metadata[KEY]
    assert after['deferred_seconds']==206 and after['original_submitted_at']==iso(0)
    assert before['failed_group_ids']==after['failed_group_ids']==['failed-group']
    assert protected(after)


def test_hold_and_unready_intervals_do_not_accrue_or_deadlock_other_jobs(store):
    old=job(store,'old',0);young=job(store,'young',1);defer(store,old);s=Service(store);s.clock=10;s._runnable_jobs()
    store.update_job(old.job_id,hold=True);s.clock=100;s._runnable_jobs();r=store.get_job(old.job_id).metadata[KEY]
    assert r['deferred_seconds']==10 and r['blocked_reason']=='user_hold'
    assert s._dispatch_plan(plan(young))
    s.active=[];store.update_job(old.job_id,hold=False,metadata_updates={'fixture_global_block':'preparation_unready'})
    s.clock=200;s._runnable_jobs();assert store.get_job(old.job_id).metadata[KEY]['deferred_seconds']==10
    store.update_job(old.job_id,metadata_updates={'fixture_global_block':None});s._runnable_jobs();s.clock=220;s._runnable_jobs()
    assert protected(store.get_job(old.job_id).metadata[KEY])


def test_solo_infeasible_or_backend_blocked_job_is_explicit_and_does_not_freeze_queue(store):
    old=job(store,'old',0,fixture_global_block='solo_memory_infeasible');young=job(store,'young',1);defer(store,old);s=Service(store);s.clock=206
    assert s._dispatch_plan(plan(young))
    record=store.get_job(old.job_id).metadata[KEY]
    assert record['blocked_reason']=='solo_memory_infeasible' and not protected(record)
    assert store.get_job(old.job_id).status==JobStatus.READY


def test_reservation_never_forces_failed_group_new_probe_or_bypasses_existing_memory_gate(store):
    old=job(store,'old',0,cold_start_solo_required=True);defer(store,old);s=Service(store);s.clock=31
    assert not s._dispatch_plan(plan(old,trial=True))
    assert not s._dispatch_plan(plan(old,illegal_memory_or_retry=True))
    assert s._dispatch_plan(plan(old))
    assert store.get_job(old.job_id).metadata['cold_start_solo_required']is True


def test_measured_legal_placement_can_serve_without_draining_incumbent(store):
    old=job(store,'old',0);resident=job(store,'resident',1,JobStatus.RUNNING);defer(store,old);s=Service(store);s.clock=31;s.active=[resident.job_id]
    s._v5_deferral_legal_placement=lambda job,p:True
    assert s._dispatch_plan(plan(old))
    assert store.get_job(resident.job_id).status==JobStatus.RUNNING


@pytest.mark.parametrize('decision',['rejected','inconclusive'])
def test_trial_outcome_and_solo_required_are_durable_without_reset(store,decision):
    old=job(store,'old',0);defer(store,old);s=Service(store);s.clock=10;s._runnable_jobs()
    trial=SimpleNamespace(candidate_job_id=old.job_id,profile_key='failed-second')
    assert s._finish_cold_start_trial(trial,decision,'fixture_reason')=='base-result'
    r=store.get_job(old.job_id).metadata[KEY]
    assert r['deferred_seconds']==10 and r['pending'] and r['defer_count']==2
    assert 'failed-second'in r['failed_group_ids'] and decision+':fixture_reason'in r['reasons']
    s.clock=206;s._runnable_jobs();assert protected(store.get_job(old.job_id).metadata[KEY])


def test_already_solo_required_restores_pending_record_without_being_previously_ready(store):
    old=job(store,'old',0,cold_start_solo_required=True,fixture_global_block='preparation_unready');s=Service(store);s.clock=100;s._runnable_jobs()
    r=store.get_job(old.job_id).metadata[KEY];assert r['pending'] and r['deferred_seconds']==0 and not protected(r)


def test_stable_oldest_protected_order_and_original_timestamp_survives_retry(store):
    b=job(store,'b',0);a=job(store,'a',0);defer(store,b);defer(store,a);s=Service(store);s.clock=31
    rows=s._runnable_jobs();assert rows[0].job_id==a.job_id and choose_protected(rows).job_id==a.job_id
    assert store.get_job(b.job_id).submitted_at==iso(0)


def test_disabled_feature_is_exact_passthrough_and_writes_no_reservation(store):
    old=job(store,'old',0);young=job(store,'young',1);defer(store,old);s=Service(store,enabled=False);s.clock=206
    before=deepcopy(store.get_job(old.job_id).metadata)
    assert s._runnable_jobs()==store.runnable_jobs()
    assert s._dispatch_plan(plan(young))
    assert store.get_job(old.job_id).metadata==before


def test_invalid_clock_or_fake_launch_identity_rejected():
    r=mark_deferred(new_record(iso(0),10),now=10,runnable=True,reason='rejected')
    with pytest.raises(ValueError):advance_record(r,now=9,runnable=True)
    with pytest.raises(ValueError):record_bypass(r,launched_submitted_at=iso(1),launch_id='')
    assert advance_record(json.loads(json.dumps(r)),now=40,runnable=True)['deferred_seconds']==30


def test_few_productive_trial_updates_never_reset_pending_seconds_or_bypasses(store):
    old=job(store,'old',0);child=job(store,'child',1);defer(store,old);s=Service(store)
    s.clock=5;assert s._dispatch_plan(plan(child))
    s.active=[];s.clock=10;assert s._dispatch_plan(plan(old,trial=True))
    store.update_job(old.job_id,metadata_updates={'runtime_global_step':7})
    record=store.get_job(old.job_id).metadata[KEY]
    assert record['pending'] and record['deferred_seconds']==10 and len(record['bypass_launch_ids'])==1
    assert record['last_runnable']is False
    s.active=[];store.update_job(old.job_id,status=JobStatus.PAUSED);s.clock=40
    # A crash/restart before another observation must not charge trial execution.
    restarted=Service(SQLiteStateStore(store.settings));restarted.clock=40;restarted._runnable_jobs()
    assert store.get_job(old.job_id).metadata[KEY]['deferred_seconds']==10
    restarted.clock=61;restarted._runnable_jobs()
    assert protected(store.get_job(old.job_id).metadata[KEY])


def test_missing_launch_evidence_does_not_invent_a_bypass_or_serve_reservation(store):
    old=job(store,'old',0);child=job(store,'child',1);defer(store,old);s=Service(store)
    s._v5_deferral_actual_launch_id=lambda job_id:None
    assert s._dispatch_plan(plan(child))
    assert store.get_job(old.job_id).metadata[KEY]['bypass_launch_ids']==[]
    assert any(name=='v5_deferral_launch_identity_missing'for name,_ in s.events)


def test_held_or_unready_never_deferred_jobs_do_not_acquire_debt(store):
    held=job(store,'held',0);store.update_job(held.job_id,hold=True)
    unready=job(store,'unready',1,fixture_global_block='preparation_unready')
    service=Service(store);service.clock=206;service._runnable_jobs()
    assert KEY not in store.get_job(held.job_id).metadata
    assert KEY not in store.get_job(unready.job_id).metadata


def test_productive_bracket_bootstrap_is_not_normal_service_or_reserved_placement(store):
    old=job(store,'old',0);defer(store,old);s=Service(store);s.clock=10
    bootstrap=plan(old);bootstrap.trial_metadata['bracket_bootstrap']=True
    assert s._dispatch_plan(bootstrap)
    record=store.get_job(old.job_id).metadata[KEY]
    assert record['pending'] and record['deferred_seconds']==10 and record['served_launch_ids']==[]
    s.active=[];store.update_job(old.job_id,status=JobStatus.PAUSED);s.clock=11;s._runnable_jobs();s.clock=32;s._runnable_jobs()
    assert protected(store.get_job(old.job_id).metadata[KEY])
    assert not s._dispatch_plan(bootstrap)
    assert s._dispatch_plan(plan(old))


def test_real_placement_planner_cannot_resort_older_ordinary_ahead_of_reserved_job(tmp_path):
    from localml_scheduler.tests.test_cold_start_trials import setup
    settings,store,planner,jobs=setup(tmp_path)
    older,reserved,resident=jobs
    older.submitted_at=iso(0);reserved.submitted_at=iso(1);resident.submitted_at=iso(2)
    for value in jobs:store.save_job(value)
    defer(store,reserved);service=Service(store);service.clock=31
    available={'cuda_process':True,'exclusive':True}
    # Reproduce the bug: even a protected-first sorted list is reordered by the
    # real planner's queue/cold-start FIFO selection to the ordinary older job.
    sorted_only=service._v5_deferral_sync(Base._runnable_jobs(service))
    assert sorted_only[0].job_id==reserved.job_id
    assert planner.choose_plan(sorted_only,backend_available=available).job_ids==(older.job_id,)
    candidates=service._runnable_jobs()
    assert [value.job_id for value in candidates]==[reserved.job_id]
    assert {value.job_id for value in store.list_jobs()}=={value.job_id for value in jobs}
    # With incumbents, a fresh trial is not a legal reservation fulfillment.
    store.update_job(resident.job_id,status=JobStatus.RUNNING);service.active=[resident.job_id]
    trial=planner.choose_plan(candidates,active_jobs=[store.get_job(resident.job_id)],backend_available=available)
    assert trial.trial_metadata['requires_live_trial']
    assert not service._dispatch_plan(trial)
    assert store.get_job(resident.job_id).status==JobStatus.RUNNING
    service.active=[];store.update_job(resident.job_id,status=JobStatus.COMPLETED)
    anchor=planner.choose_plan(service._runnable_jobs(),backend_available=available)
    assert anchor.job_ids==(reserved.job_id,)and not anchor.trial_metadata['requires_live_trial']
    assert service._dispatch_plan(anchor)
    assert store.get_job(older.job_id).status==JobStatus.PENDING
    assert not store.get_job(reserved.job_id).metadata[KEY]['pending']


def test_globally_blocked_deferred_job_does_not_remain_a_planner_head(store):
    blocked=job(store,'blocked',0,fixture_global_block='solo_memory_infeasible');ready=job(store,'ready',1)
    defer(store,blocked);service=Service(store);service.clock=206
    assert [value.job_id for value in service._runnable_jobs()]==[ready.job_id]
    assert store.get_job(blocked.job_id).metadata[KEY]['blocked_reason']=='solo_memory_infeasible'
    store.update_job(blocked.job_id,metadata_updates={'fixture_global_block':None})
    service._runnable_jobs();service.clock=237
    assert [value.job_id for value in service._runnable_jobs()]==[blocked.job_id]


def test_existing_1800_second_guard_is_not_hidden_by_younger_new_reservation(tmp_path):
    from localml_scheduler.tests.test_cold_start_trials import setup
    settings,store,planner,jobs=setup(tmp_path)
    settings.gpu_scheduler.feedback_latency.enabled=True
    oldest,younger,blocked=jobs
    oldest.submitted_at=iso(0);younger.submitted_at=iso(1790);blocked.submitted_at=iso(-1)
    blocked.metadata['fixture_global_block']='solo_memory_infeasible'
    for value in jobs:store.save_job(value)
    defer(store,younger,now=1770)
    service=Service(store);service.settings=settings;service.planner=planner;service.clock=1801
    assert [j.job_id for j in service._runnable_jobs()]==[oldest.job_id]
    original=store.get_job(oldest.job_id).metadata[KEY]
    assert original['outer_age_protected']and original['reasons']==['outer_age_guard']and protected(original)
    debt=store.get_job(younger.job_id).metadata[KEY]
    assert debt['deferred_seconds']==31 and debt['pending']
    assert KEY not in store.get_job(blocked.job_id).metadata
    restarted=Service(SQLiteStateStore(settings));restarted.settings=settings;restarted.planner=planner;restarted.clock=1802
    assert restarted._runnable_jobs()[0].job_id==oldest.job_id
    assert store.get_job(oldest.job_id).submitted_at==iso(0)
