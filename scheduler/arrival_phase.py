"""Choose queue fairness from observed arrivals, never future task features."""
import math


def phase_controls(accepted_count, declared_total=None, *, max_wait=180.):
    if type(accepted_count) is not int or accepted_count < 0:
        raise ValueError('invalid accepted count')
    if type(max_wait) not in (int, float) or not math.isfinite(max_wait) or max_wait <= 0:
        raise ValueError('invalid open-queue age protection')
    if declared_total is not None:
        if type(declared_total) is not int or declared_total <= 0 or accepted_count > declared_total:
            raise ValueError('invalid declared cohort')
        if accepted_count == declared_total:
            return dict(finite=True, max_wait=None, phase='all_arrived_drain')
    return dict(finite=False, max_wait=float(max_wait), phase='awaiting_arrivals_age_protected')
