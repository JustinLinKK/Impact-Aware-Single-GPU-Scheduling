"""Sustained solo calibration has its own finite window; trials keep 64 steps."""
from types import SimpleNamespace

import pytest

from localml_scheduler.domain import SafePointType
from localml_scheduler.execution.trial_control import _window_max_steps
from localml_scheduler.tests.test_cold_start_trials import setup
from localml_scheduler.tests.test_mlevolve_runner import _build_context


@pytest.mark.parametrize('limit', [64, 128])
def test_finite_cap_ends_phase_clean_window_with_identical_productive_counts(tmp_path, monkeypatch, limit):
    settings, _, _, jobs = setup(tmp_path)
    command = dict(owner='calibration', token='measure', action='measure', kind='measure_overlap',
                   window_min_seconds=1000, expires_at=3000)
    if limit != 64:
        command['window_max_steps'] = limit
    jobs[0].metadata.update(runtime_completion_identity='source', runtime_nonupdate_phase_counter=0,
                            trial_step_command=command)
    context = _build_context(settings, jobs[0])
    clock = SimpleNamespace(now=1000.)
    monkeypatch.setattr('localml_scheduler.execution.trial_control.time', SimpleNamespace(
        time=lambda: clock.now, monotonic=lambda: clock.now, perf_counter=lambda: clock.now))
    for step in range(limit + 3):
        clock.now += .001
        context.control_hook._cooperative_step(SafePointType.STEP, epoch=0, global_step=step,
                                               steps_per_epoch=10000, metrics={})
        if step < limit + 2:
            assert not context.store.get_job('a').metadata.get('trial_step_window')
    window = context.store.get_job('a').metadata['trial_step_window']
    assert window['complete'] and window['excludes_nonupdate_phases']
    assert window['steps'] == limit and window['end_step'] - window['start_step'] == limit
    assert len(window['halves']) == 2


@pytest.mark.parametrize('invalid', [None, True, 0, 6, 9, 4097, float('inf')])
def test_invalid_unbounded_window_cannot_run(invalid):
    with pytest.raises(ValueError):
        _window_max_steps({'window_max_steps': invalid})
