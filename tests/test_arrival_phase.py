"""Portable queue lifecycle tests; no GPU or candidate execution."""
import pytest
from localml_scheduler.scheduler.arrival_phase import phase_controls
from localml_scheduler.scheduler.ready_set import Costs, choose


@pytest.mark.parametrize('count', [0, 1, 12, 19])
def test_declared_count_does_not_reveal_future_features(count):
    assert phase_controls(count, 20) == dict(finite=False, max_wait=180., phase='awaiting_arrivals_age_protected')


def test_drain_only_after_every_declared_arrival():
    assert phase_controls(20, 20) == dict(finite=True, max_wait=None, phase='all_arrived_drain')
    assert phase_controls(100)['max_wait'] == 180.


@pytest.mark.parametrize('count,total', [(-1, 20), (21, 20), (True, 20), (1.5, 20), (0, 0), (1, True)])
def test_invalid_lifecycle(count, total):
    with pytest.raises(ValueError):
        phase_controls(count, total)


@pytest.mark.parametrize('wait', [0, -1, float('nan'), float('inf'), True])
def test_invalid_age_threshold(wait):
    with pytest.raises(ValueError):
        phase_controls(0, 20, max_wait=wait)


def test_real_selector_receives_lifecycle_controls():
    reports = {k: dict(status='estimated', source_sha256=k,
        cpu_profiles={str(c): dict(runtime_seconds=100., planning_memory_mib=1000.,
                                  gpu_busy_fraction=.5, cpu_pressure_lower_bound=.4)
                      for c in (2,3,4,6)}) for k in ('a','b')}
    costs = Costs(dict(profiles=[], co_runs=[]), reports)
    for count in (2,3):
        phase = phase_controls(count, 3)
        decision = choose(['a','b'], {}, costs, accepted={'a':0.,'b':1.}, now=400.,
                          used_mib=600., capacity_mib=14670., policy='lookahead_drain',
                          finite_cohort=phase['finite'], max_wait=phase['max_wait'])
        assert decision['fairness_mode'] == ('queued_age_protection' if count==2 else 'finite_cohort_drain')
        assert decision['age_protected'] == ('a' if count==2 else None)
        assert set(decision['admit']) <= set(reports)
