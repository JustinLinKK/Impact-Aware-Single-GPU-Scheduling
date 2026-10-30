"""Pure conservative admission proposal, separate from frozen v4 policy.

All scenarios forecast the same post-trial remaining update state. A measured
reserve is an admission margin, never deducted from completed optimizer work.
No calibration state is updated by this module or by evaluation scores.
"""
from __future__ import annotations

from math import isfinite

from ...scheduler.feedback_objective import compare_feedback_admission


POLICY_ID = "a10_feedback_worst_scenario_v1"


def trial_amortizes(*, remaining_updates, solo_updates_per_second,
                    measured_reserve_seconds, attempts, max_attempts=1):
    values = (remaining_updates, solo_updates_per_second, measured_reserve_seconds)
    if (any(isinstance(x, bool) or not isfinite(float(x)) or x <= 0 for x in values)
            or type(attempts) is not int or attempts < 0
            or type(max_attempts) is not int or max_attempts <= 0):
        raise ValueError("trial budget must have finite measured positive costs")
    return (attempts < max_attempts
            and remaining_updates / solo_updates_per_second > measured_reserve_seconds)


def conservative_admission(*, measured_reserve_seconds, min_gain_seconds=1.0,
                           solo_lower=None, solo_upper=None, **kwargs):
    """Require gain in both windows under nominal, bounded, no-recovery priors.

    Lower/upper rates are a *frozen calibration* interval, not a packed-rate solo
    measurement. The no-recovery case explicitly caps candidate recovery at its
    observed packed rate. Original v4's regret tradeoff remains untouched.
    """
    result = {"admission_policy": POLICY_ID, "decision": "inconclusive",
              "error": None, "scenarios": []}
    try:
        if (isinstance(measured_reserve_seconds, bool)
                or not isfinite(float(measured_reserve_seconds))
                or measured_reserve_seconds <= 0
                or isinstance(min_gain_seconds, bool)
                or not isfinite(float(min_gain_seconds)) or min_gain_seconds <= 0):
            raise ValueError("positive measured reserve and minimum gain required")
        kwargs = dict(kwargs)
        kwargs.pop("bracket_admission_mode", None)
        solo = dict(kwargs["solo"])
        scenarios = [("nominal", solo)]
        for name, boundary in (("calibration_lower", solo_lower), ("calibration_upper", solo_upper)):
            if boundary is not None:
                if set(boundary) != set(solo):
                    raise ValueError("calibration interval membership mismatch")
                for key, value in boundary.items():
                    if (isinstance(value, bool) or not isfinite(float(value)) or value <= 0
                            or (name.endswith("lower") and value > solo[key])
                            or (name.endswith("upper") and value < solo[key])):
                        raise ValueError("invalid frozen calibration interval")
                scenarios.append((name, dict(boundary)))
        candidate = kwargs["candidate"]
        packed = kwargs["packed_halves"][candidate]
        # Each measured half gets a whole-cohort forecast with its own cautious
        # solo hypothesis. Crossed half/scenario comparisons are conservative.
        for index, observed_rate in enumerate(packed):
            cautious = dict(solo)
            cautious[candidate] = min(cautious[candidate], observed_rate)
            scenarios.append((f"no_recovery_half_{index}", cautious))
        threshold = min_gain_seconds + measured_reserve_seconds
        scores = []
        for name, rates in scenarios:
            args = {**kwargs, "solo": rates, "min_gain_seconds": threshold,
                    "bracket_admission_mode": "nominal"}
            forecast = compare_feedback_admission(**args)
            result["scenarios"].append({"name": name, "solo_hypothesis": rates,
                                         "forecast": forecast})
            if forecast["error"] or len(forecast["halves"]) != 2:
                raise ValueError(forecast["error"] or "incomplete forecast")
            scores.extend(half["gain_seconds"] for half in forecast["halves"])
        result.update(worst_gain_seconds=min(scores), required_gain_seconds=threshold,
                      measured_reserve_seconds=measured_reserve_seconds,
                      decision="accepted" if min(scores) >= threshold else "rejected")
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, ZeroDivisionError) as exc:
        result["error"] = str(exc)
    return result
