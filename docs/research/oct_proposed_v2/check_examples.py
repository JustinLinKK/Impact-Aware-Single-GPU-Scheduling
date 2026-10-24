"""Check analytical proposal examples; does not simulate or run a scheduler."""
import json
from math import isclose


def examples():
    checks = {}
    checks['low_busy_can_hurt_progress'] = isclose(.55 + .15, .70) and .70 < 1
    checks['higher_busy_can_help_progress'] = isclose(.90 + .75, 1.65)
    checks['long_overlap_repays_cost'] = isclose(120 * (1.65 - 1) - 12, 66)
    checks['short_overlap_does_not'] = isclose(10 * (1.65 - 1) - 12, -5.5)
    checks['example_individual_slowdown_passes'] = max(1 / .90, 1 / .75) <= 1.5
    checks['aggregate_gain_does_not_excuse_harm'] = .50 + 1.20 > 1 and 1 / .50 > 1.5
    entitlement = 60 / 2
    checks['waiting_job_builds_debt'] = max(0, entitlement - 0) == 30
    checks['served_job_has_no_positive_debt'] = max(0, entitlement - 60) == 0
    checks['new_arrival_has_no_prior_entitlement'] = 0 < max(0, entitlement)
    # Illustrative 3% token bucket: startup credit + refill is an upper spending
    # planned budget before explicit, charged safe-boundary overrun; no job-count cap.
    checks['trial_budget_units'] = isclose(30 + .03 * 600, 48)
    checks['normalization_is_dimensionless'] = isclose(.2 / .25, .8)
    assert all(checks.values()), checks
    return {'passed': True, 'scope': 'analytical arithmetic only; no training, trace replay or GPU experiment',
            'checks': checks, 'count': len(checks),
            'example_values': {'long_overlap_gain_solo_seconds': 66,
                               'short_overlap_gain_solo_seconds': -5.5,
                               'waiting_debt_solo_seconds': 30}}


if __name__ == '__main__':
    print(json.dumps(examples(), indent=2))
