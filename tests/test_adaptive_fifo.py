"""CPU-only executable contracts for nonpreemptive adaptive FIFO."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import time
from types import SimpleNamespace

import pytest

from localml_scheduler.domain import JobStatus, TrainingJob
from localml_scheduler.scheduler.adaptive_fifo import AdaptiveFIFOController, AdaptiveFIFOOptions


@pytest.fixture(autouse=True)
def synthetic_monotonic_origin(monkeypatch):
    from localml_scheduler.scheduler import adaptive_fifo
    monkeypatch.setattr(adaptive_fifo, 'time', SimpleNamespace(
        monotonic=lambda: 0., time=time.time, perf_counter=time.perf_counter))


def job(name, *, budget=1000, accepted=0., ordinal=1):
    result = TrainingJob.create('unused', '', '', job_id=name, max_steps=budget,
        max_epochs=1, runner_kwargs={'batch_size': 1}, metadata={
            'v51_accepted_at': accepted, 'runtime_steps_per_epoch': budget,
            'runtime_total_optimizer_updates': budget, 'runtime_planned_epochs': 1,
            'runtime_completion_identity': 'recipe-' + name,
            'v51_hardware_key': 'exact-host-key', 'v51_backend': 'cuda_process',
            'v51_phase_model_ttl_seconds': 36000.,
            'runtime_nonupdate_phase_counter': 0, 'trial_runner_ready': True,
            'runtime_global_step': 10, 'v51_phase': 'training'})
    result.queue_sequence = ordinal
    result.metadata['feedback_phase_model'] = {
        'source': 'measured_phase_model_v1', 'covers_required_phases': True,
        'job_signature': name, 'hardware_key': 'exact-host-key', 'backend': 'cuda_process', 'batch_size': 1,
        'observed_at': datetime.now(timezone.utc).isoformat(),
        'startup_seconds': 1., 'finish_seconds': 2., 'delivery_seconds': 1.}
    return result


def profile(j):
    return {'job_signature': j.job_id, 'batch_size': 1, 'artifact_sha256': 'student',
            'hardware': 'gpu', 'step_cost': 100., 'peak_vram_mib': 100.}


SAMPLE = {'used_mib': 300., 'total_mib': 10000.}


def setup(n=4, *, mode='individual', options=None):
    c = AdaptiveFIFOController(mode, options=options)
    jobs = [job(chr(97 + i), ordinal=i + 1) for i in range(n)]
    profiles = {j.job_id: profile(j) for j in jobs}
    c.observe(now=0., queued_jobs=jobs, running_jobs=[], profiles=profiles,
              gpu_sample=SAMPLE, completed_job_ids={})
    return c, jobs, profiles


def launch(c, jobs, n=3):
    for j in jobs[:n]:
        c.on_launch(j.job_id, 0.)
        j.status = JobStatus.RUNNING


def pair(c, jobs, profiles, *, start=10.):
    assert c._measure(jobs, profiles, start) is None
    for j in jobs:
        j.metadata['runtime_global_step'] += 10
    assert c._measure(jobs, profiles, start + 1.) is None
    for j in jobs:
        j.metadata['runtime_global_step'] += 10
    return c._measure(jobs, profiles, start + 2.)


def forecast(c, jobs, profiles, cap, halves, *, queued=None, running=None, completed=None, sample=None):
    c._jobs.update({j.job_id: deepcopy(j) for j in jobs})
    return c._forecast(cap, 0, queued if queued is not None else [j.job_id for j in jobs[3:]],
        running if running is not None else [j.job_id for j in jobs[:3]], profiles,
        completed or {}, sample or SAMPLE, halves, 12., time.perf_counter() + .1)


def test_two_modes_have_different_numerical_objectives():
    c, jobs, _ = setup(2)
    for j in jobs:
        j.metadata['feedback_request'] = {'request_id': 'r', 'workflow_id': 'w',
            'completion_policy': 'group', 'membership_finalized': True,
            'required_job_ids': ['a', 'b'], 'submitted_at': j.submitted_at}
        j.metadata['v51_request_accepted_at'] = 0.
    c._jobs.update({j.job_id: j for j in jobs})
    assert c._score({'a': 1., 'b': 11.}) == 6.
    assert c._score({'a': 4., 'b': 10.}) == 7.
    c.mode = 'group'
    assert c._score({'a': 1., 'b': 11.}) == 11.
    assert c._score({'a': 4., 'b': 10.}) == 10.


@pytest.mark.parametrize('cap', [1, 2, 3, 4])
def test_fixed_cap_forecast_keeps_all_actual_residents_and_refills_fifo(cap):
    c, jobs, profiles = setup(5)
    launch(c, jobs)
    halves = pair(c, jobs[:3], profiles)
    result = forecast(c, jobs, profiles, cap, halves)
    assert set(result['receipts']) == set('abcde')
    occupancy = 3
    launches = []
    for event in result['trace']:
        if event['kind'] == 'release':
            occupancy -= 1
        else:
            assert occupancy < cap
            occupancy += 1
            launches.append(event['job_id'])
    assert launches == ['d', 'e']
    if cap == 1:
        assert result['trace'][0]['kind'] == 'release'
        assert [v['kind'] for v in result['trace'][:3]] == ['release'] * 3


def test_fifo_head_memory_blocks_younger_and_cannot_fit_alone():
    c, jobs, profiles = setup(5)
    launch(c, jobs)
    halves = pair(c, jobs[:3], profiles)
    profiles['d']['peak_vram_mib'] = 10001.
    with pytest.raises(ValueError, match='FIFO head memory infeasible'):
        forecast(c, jobs, profiles, 4, halves)
    with pytest.raises(ValueError, match='FIFO head'):
        c.on_launch('e', 12.)


def test_known_zero_finishes_and_unknown_counter_is_not_zero():
    c, jobs, profiles = setup(3)
    launch(c, jobs)
    halves = pair(c, jobs, profiles)
    jobs[0].metadata['runtime_global_step'] = 1000
    jobs[0].metadata['v51_phase_started_at'] = 12.
    assert c._remaining(jobs[0]) == 0
    result = forecast(c, jobs, profiles, 1, halves, queued=[])
    assert result['receipts']['a'] == 15.  # pending finish=2 + delivery=1
    del jobs[0].metadata['v51_phase_started_at']
    assert forecast(c, jobs, profiles, 1, halves, queued=[])['receipts']['a'] == 15.
    del jobs[0].metadata['runtime_global_step']
    with pytest.raises(ValueError, match='unknown exact'):
        c._remaining(jobs[0])


@pytest.mark.parametrize('change', ['generation', 'membership', 'counter', 'profile', 'epoch', 'backward', 'boundary', 'zero'])
def test_window_invalidation(change):
    c, jobs, profiles = setup(3)
    launch(c, jobs)
    assert c._measure(jobs, profiles, 10.) is None
    for j in jobs:
        j.metadata['runtime_global_step'] += 10
    assert c._measure(jobs, profiles, 11.) is None
    if change == 'generation': jobs[0].metadata['runtime_completion_identity'] = 'restarted'
    if change == 'membership': jobs = jobs[:2]
    if change == 'counter': jobs[0].metadata['runtime_nonupdate_phase_counter'] += 1
    if change == 'profile': profiles['a']['step_cost'] = 101.
    if change == 'epoch': jobs[0].metadata['runtime_steps_per_epoch'] = 10
    if change == 'backward': jobs[0].metadata['runtime_global_step'] = 1
    if change == 'boundary': jobs[0].metadata['runtime_global_step'] = 1000
    if change == 'zero': jobs[0].metadata['runtime_global_step'] = 0
    assert c._measure(jobs, profiles, 12.) is None
    assert not any(e['kind'] == 'valid_windows' for e in c.events)


def test_short_success_and_stale_zero_updates_never_invent_windows():
    c, jobs, profiles = setup(1)
    launch(c, jobs, 1)
    assert c._measure(jobs, profiles, 10.) is None
    assert c._measure(jobs, profiles, 11.) is None
    jobs[0].metadata['runtime_global_step'] = 1000
    jobs[0].status = JobStatus.COMPLETED
    c.observe(now=12., queued_jobs=jobs, running_jobs=[], profiles=profiles,
              gpu_sample=SAMPLE, completed_job_ids={'a': 12.})
    assert c.admission_limit == 3
    assert not any(e['kind'] == 'valid_windows' for e in c.events)


def test_exploration_is_productive_one_head_per_count_and_dwell_bounded():
    c, jobs, profiles = setup(5)
    launch(c, jobs)
    for now, step in [(10., 10), (11., 20), (12., 30)]:
        for j in jobs[:3]: j.metadata['runtime_global_step'] = step
        c.observe(now=now, queued_jobs=jobs[3:], running_jobs=jobs[:3], profiles=profiles,
                  gpu_sample=SAMPLE, completed_job_ids={})
    assert c.admission_limit == 4
    assert c.events[-1]['reason'] == 'productive_head_exploration_no_latency_guarantee'
    assert c._explored == {4}
    c.on_launch('d', 12.)
    with pytest.raises(ValueError, match='exactly once'):
        c.on_launch('d', 12.)
    assert c._pending_exploration == 4
    json.dumps(c.events, allow_nan=False)


def test_forecast_timeout_and_unknown_phase_preserve_ordinary_refill_limit(monkeypatch):
    c, jobs, profiles = setup(4)
    launch(c, jobs)
    def timed_out(*args, **kwargs): raise TimeoutError('bounded test timeout')
    monkeypatch.setattr(c, '_forecast', timed_out)
    for now, step in [(10., 10), (11., 20), (12., 30)]:
        for j in jobs[:3]: j.metadata['runtime_global_step'] = step
        c.observe(now=now, queued_jobs=jobs[3:], running_jobs=jobs[:3], profiles=profiles,
                  gpu_sample=SAMPLE, completed_job_ids={})
    assert c.admission_limit == 3
    assert c.events[-1]['kind'] == 'forecast_hold'
    c.on_launch('d', 13.)  # caller may refill when a real worker releases


def test_restart_preserves_dispatch_and_exploration_but_invalidates_rates():
    c, jobs, profiles = setup(4)
    launch(c, jobs)
    pair(c, jobs[:3], profiles)
    c._explored.add(4)
    restored = AdaptiveFIFOController.restore_state(json.loads(json.dumps(c.export_state())))
    assert restored._counts == {} and restored._anchor is None
    assert restored._order == c._order and restored._launched == c._launched
    assert restored._explored == {4}
    with pytest.raises(ValueError): restored.on_launch('a', 20.)


def test_full_bracket_includes_completed_and_delivering_required_members():
    c, jobs, profiles = setup(4, mode='group')
    launch(c, jobs)
    halves = pair(c, jobs[:3], profiles)
    for j in jobs:
        j.metadata['feedback_request'] = {'request_id': 'all', 'completion_policy': 'group',
            'membership_finalized': True, 'required_job_ids': list('abcd')}
        j.metadata['v51_request_accepted_at'] = 0.
    jobs[0].metadata.update(runtime_global_step=1000, v51_phase='delivering', v51_resource_released_at=11.5)
    jobs[0].status = JobStatus.COMPLETED
    result = forecast(c, jobs, profiles, 3, halves, running=['b', 'c'])
    assert result['receipts']['a'] == 12.5
    assert result['score'] == max(result['receipts'].values())
    c._jobs['d'].metadata['feedback_request']['required_job_ids'].append('future-child')
    with pytest.raises(ValueError, match='missing sealed'):
        c._score(result['receipts'])


def test_bounds_and_nonpositive_observation_clock():
    with pytest.raises(ValueError): AdaptiveFIFOOptions(cpu_budget_ms=11.)
    c, jobs, profiles = setup(3)
    launch(c, jobs)
    c._measure(jobs, profiles, 10.)
    c.observe(now=0., queued_jobs=[], running_jobs=jobs, profiles=profiles,
              gpu_sample=SAMPLE, completed_job_ids={})
    assert c._anchor is None


def test_missing_acceptance_and_overlapping_requests_hold_forecast():
    c, jobs, profiles = setup(2, mode='group')
    c._accepted['a'] = None
    with pytest.raises(ValueError, match='missing accepted'):
        c._score({'a': 1., 'b': 2.})
    c._accepted['a'] = 0.
    for j in jobs:
        j.metadata['v51_request_accepted_at'] = 0.
        j.metadata['feedback_request'] = {'request_id': j.job_id,
            'completion_policy': 'group', 'membership_finalized': True,
            'required_job_ids': ['a', 'b']}
    c._jobs.update({j.job_id: j for j in jobs})
    with pytest.raises(ValueError, match='overlapping'):
        c._score({'a': 1., 'b': 2.})


def test_reboot_rejects_monotonic_state_restore():
    c, _, _ = setup()
    state = c.export_state()
    state['clock_identity'] = 'other-machine-boot'
    with pytest.raises(ValueError, match='across reboot'):
        AdaptiveFIFOController.restore_state(state)


def test_queued_real_pilot_job_has_exact_budget_without_runtime_counters(tmp_path):
    from localml_scheduler.benchmarks.a10_feedback_latency.run import build_job
    from pathlib import Path
    manifest_path = Path(__file__).parents[1] / 'benchmarks/a10_feedback_latency/workloads/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    spec = manifest['jobs'][0]
    actual, _ = build_job(spec, manifest_path=manifest_path, output=tmp_path,
        manifest=manifest, memory_budget_mib=20000.)
    assert actual.metadata.get('runtime_global_step') is None
    assert AdaptiveFIFOController._remaining(actual, queued=True) == actual.max_steps


@pytest.mark.parametrize('q', [.6, .8])
def test_solo_anchor_and_mode_specific_two_equal_job_forecast(q):
    c, jobs, profiles = setup(2)
    for j in jobs:
        j.metadata['runtime_global_step'] = 0
        j.metadata['v51_request_accepted_at'] = 0.
        j.metadata['feedback_request'] = {'request_id': 'pair', 'completion_policy': 'group',
            'membership_finalized': True, 'required_job_ids': ['a', 'b']}
        j.metadata['feedback_phase_model'].update(startup_seconds=0., finish_seconds=0., delivery_seconds=0.)
    c._jobs.update({j.job_id: deepcopy(j) for j in jobs})
    c._counts[2] = {'at': 12., 'factors': [q, q]}
    halves = [{'rates': {'a': 10. * q, 'b': 10. * q}}] * 2
    assert c._rates(['a'], 1, 0, profiles, halves, 12.)['a'] == 10.
    one = forecast(c, jobs, profiles, 1, halves, queued=['a', 'b'], running=[])
    two = forecast(c, jobs, profiles, 2, halves, queued=['a', 'b'], running=[])
    assert one['score'] == pytest.approx(162.)  # 12 + mean(100,200)
    assert (two['score'] > one['score']) == (q == .6)
    c.mode = 'group'
    assert c._score(two['receipts']) < c._score(one['receipts'])


def test_empty_busy_period_alpha_reset_starts_fresh_dwell():
    c, jobs, profiles = setup(1)
    launch(c, jobs, 1)
    c.admission_limit = 4
    jobs[0].status = JobStatus.COMPLETED
    c.observe(now=20., queued_jobs=[], running_jobs=[], profiles=profiles,
              gpu_sample=SAMPLE, completed_job_ids={'a': 20.})
    assert c.admission_limit == 3 and c._last_change == 20.
    assert c.events[-1]['kind'] == 'busy_period_empty_reset'
    newer = [job(name, accepted=20., ordinal=i + 2) for i, name in enumerate('bcde')]
    profiles.update({j.job_id: profile(j) for j in newer})
    c.observe(now=20.1, queued_jobs=newer, running_jobs=[], profiles=profiles,
              gpu_sample=SAMPLE, completed_job_ids={'a': 20.})
    for j in newer[:3]: c.on_launch(j.job_id, 20.1)
    for now, step in [(21., 10), (22., 20), (23., 30)]:
        for j in newer[:3]: j.metadata['runtime_global_step'] = step
        c.observe(now=now, queued_jobs=newer[3:], running_jobs=newer[:3], profiles=profiles,
                  gpu_sample=SAMPLE, completed_job_ids={'a': 20.})
    assert c.admission_limit == 3 and c._last_change == 20.
    assert any(e['kind'] == 'valid_windows' for e in c.events)
    assert not any(e['kind'] == 'decision' and e['at'] >= 20. for e in c.events)


@pytest.mark.parametrize('mode', ['individual', 'group'])
def test_restore_then_observe_completed_history_without_historical_job_objects(mode):
    c, jobs, profiles = setup(2, mode=mode)
    if mode == 'group':
        for j in jobs:
            j.metadata['v51_request_accepted_at'] = 0.
            j.metadata['feedback_request'] = {'request_id': 'pair', 'completion_policy': 'group',
                'membership_finalized': True, 'required_job_ids': ['a', 'b']}
    c.observe(now=.1, queued_jobs=jobs, running_jobs=[], profiles=profiles,
              gpu_sample=SAMPLE, completed_job_ids={})
    c.on_launch('a', .1)
    restored = AdaptiveFIFOController.restore_state(json.loads(json.dumps(c.export_state())))
    assert restored._jobs == {}
    restored.observe(now=10., queued_jobs=[jobs[1]], running_jobs=[], profiles={'b': profiles['b']},
                     gpu_sample=SAMPLE, completed_job_ids={'a': 9.})
    assert restored._order == ['a', 'b']
    assert restored._queue_order == ['b']
    restored.on_launch('b', 10.)
    assert restored._score({'a': 9., 'b': 15.}) == (12. if mode == 'individual' else 15.)
    state_before = restored.export_state()
    jobs[1].metadata['runtime_global_step'] += 1
    restored.observe(now=11., queued_jobs=[], running_jobs=[jobs[1]], profiles={'b': profiles['b']},
                     gpu_sample=SAMPLE, completed_job_ids={'a': 9.})
    assert restored.export_state() == state_before  # progress is not durable alpha state
