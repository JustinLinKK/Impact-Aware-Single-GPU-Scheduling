from datetime import datetime, timezone, timedelta

import pytest

from localml_scheduler.domain import JobStatus
from localml_scheduler.tests.test_add_only_trials import production, overlap, sample, outcome
from localml_scheduler.tests.test_cold_start_trials import setup
from localml_scheduler.tests.test_prediction_first_workflow import virtual_predictor


def enabled(tmp_path):
    result = production(tmp_path)
    settings, store, planner, service, trial = result
    settings.gpu_scheduler.feedback_latency.enabled = True
    settings.gpu_scheduler.feedback_latency.allow_zero_overhead = True
    predictor = virtual_predictor()
    original = predictor.predict_profile_options
    def profiles(job, batches):
        values = original(job, batches)
        for p in values.values(): p['step_cost'] = 1000.
        return values
    predictor.predict_profile_options = profiles
    planner.estimator.ml_predictor = predictor
    return result


def finish(store, service, trial, *, a=2, b=.01):
    sample(store, trial, 'a', a)
    sample(store, trial, 'b', b)
    service._evaluate_colocation_trial()
    return outcome(store, trial)


def test_live_controller_uses_feedback_not_fifty_percent_and_retains_incumbent(tmp_path):
    settings, store, planner, service, trial = enabled(tmp_path)
    overlap(store, service, trial)
    r = finish(store, service, trial)
    assert r['decision'] == 'accepted', r
    assert r['result']['admission_policy'] == 'derived_solo_slope_feedback_v4'
    assert r['result']['remaining_updates']['b'] == 288
    assert set(r['result']['halves'][0]['admit']['job_completions']) == {'a','b','c'}
    assert store.get_job('a').status == JobStatus.RUNNING
    assert not service.supervisor.request_pause.called


@pytest.mark.parametrize('change', ['artifact','request','arrival','phase','missing_prediction','counter','disabled','threshold'])
def test_changes_or_missing_evidence_are_inconclusive_without_v3_fallback(tmp_path, change):
    settings, store, planner, service, trial = enabled(tmp_path)
    overlap(store, service, trial)
    if change == 'artifact':
        original = planner.estimator.ml_predictor.predict_profile_options
        def changed(job,batches):
            profiles=original(job,batches)
            for p in profiles.values(): p['artifact_sha256']='changed'
            return profiles
        planner.estimator.ml_predictor.predict_profile_options=changed
    elif change == 'request':
        store.update_job('c',metadata_updates={'feedback_request':{'completion_policy':'group'}})
    elif change == 'arrival':
        store.update_job('c',status=JobStatus.COMPLETED)
    elif change == 'phase':
        settings.gpu_scheduler.feedback_latency.allow_zero_overhead=False
    elif change == 'missing_prediction':
        planner.estimator.ml_predictor.available=False
    elif change == 'disabled':
        settings.gpu_scheduler.feedback_latency.enabled=False
    elif change == 'threshold':
        settings.gpu_scheduler.feedback_latency.min_gain_seconds=1000
    else:
        store.update_job('c',metadata_updates={'runtime_global_step':301})
    r=finish(store,service,trial,a=1,b=1)
    assert r['decision']=='inconclusive',r
    assert r['result']['error']
    assert store.get_job('a').status == JobStatus.RUNNING
    assert store.get_job('b').metadata['runtime_global_step']==12
    assert all(c.args[0]=='b' for c in service.supervisor.request_pause.call_args_list)


def test_group_request_flows_through_real_trial_controller(tmp_path):
    settings,store,planner,service,trial=enabled(tmp_path)
    original=min(j.submitted_at for j in store.list_jobs())
    raw={'request_id':'round','workflow_id':'w','completion_policy':'group',
         'required_job_ids':['a','b','c'],'membership_finalized':True,'submitted_at':original}
    for j in store.list_jobs(): store.update_job(j.job_id,metadata_updates={'feedback_request':raw})
    overlap(store,service,trial)
    r=finish(store,service,trial)
    assert r['decision']=='accepted',r
    assert len(r['result']['halves'][0]['admit']['requests'])==1


def test_fifo_ignores_busy_and_overdue_cooldown_blocks_backfill_until_solo(tmp_path):
    settings,store,planner,jobs=setup(tmp_path)
    settings.gpu_scheduler.feedback_latency.enabled=True
    planner.estimator.ml_predictor=virtual_predictor()
    # Put the lightest candidate first by original submission, despite busy rank.
    jobs[1].submitted_at=(datetime.now(timezone.utc)-timedelta(hours=1)).isoformat()
    assert planner.choose_plan(jobs,backend_available={'cuda_process':True}).job_ids==('b',)
    jobs[1].metadata['cold_start_solo_required']=True
    assert planner.choose_plan(jobs[1:],active_jobs=jobs[:1],backend_available={'cuda_process':True}) is None
    assert planner.choose_plan(jobs[1:],backend_available={'cuda_process':True}).job_ids==('b',)


def test_missing_predictions_still_allow_solo_but_not_packing(tmp_path):
    settings,store,planner,jobs=setup(tmp_path)
    settings.gpu_scheduler.feedback_latency.enabled=True
    assert planner.choose_plan(jobs,backend_available={'cuda_process':True}).mode=='stack_anchor'
    assert planner.choose_plan(jobs[1:],active_jobs=jobs[:1],backend_available={'cuda_process':True}) is None


def test_idle_v4_cannot_bypass_trials_through_legacy_pair_dispatch(tmp_path):
    from unittest.mock import Mock
    from localml_scheduler.config import SCHEDULER_DECISION_MODE_BACKEND_AWARED
    settings,store,planner,jobs=setup(tmp_path)
    settings.gpu_scheduler.feedback_latency.enabled=True
    settings.gpu_scheduler.colocation.live_trial_enabled=False
    settings.gpu_scheduler.scheduler_decision_mode=SCHEDULER_DECISION_MODE_BACKEND_AWARED
    planner.backend_aware.choose_empty=Mock()
    plan=planner.choose_plan(jobs,backend_available={'cuda_process':True})
    assert len(plan.job_ids)==1
    planner.backend_aware.choose_empty.assert_not_called()


@pytest.mark.parametrize('invalid', [False, 'stale', 'hardware', 'incomplete', 'malformed'])
def test_matching_phase_cost_contract_and_invalid_evidence(tmp_path, invalid):
    from localml_scheduler.domain import BatchResolution
    settings,store,planner,service,trial=enabled(tmp_path)
    settings.gpu_scheduler.feedback_latency.allow_zero_overhead=False
    for j in store.list_jobs():
        model={'source':'measured_phase_model_v1','covers_required_phases':True,
               'job_signature':j.packing.signature or j.job_id,
               'batch_size':BatchResolution.resolved_batch_size(j),
               'hardware_key':store.hardware_key(),'backend':settings.gpu_scheduler.packing_backend,
               'observed_at':datetime.now(timezone.utc).isoformat(),
               **{name+'_seconds':1. for name in ('startup','finish','delivery','pause','resume')}}
        if invalid == 'stale': model['observed_at']='2000-01-01T00:00:00+00:00'
        if invalid == 'hardware': model['hardware_key']='different'
        if invalid == 'incomplete': model.pop('finish_seconds')
        if invalid == 'malformed': model='unknown'
        store.update_job(j.job_id,metadata_updates={'feedback_phase_model':model})
    overlap(store,service,trial)
    r=finish(store,service,trial)
    assert r['decision']==('inconclusive' if invalid else 'accepted'),r
    if not invalid:
        assert set(r['result']['phase_cost_sources'].values())=={'measured_serialized_phase_proxy'}
