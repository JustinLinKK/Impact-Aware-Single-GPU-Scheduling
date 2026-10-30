"""Candidate ablation: omit the historical reserve from post-trial margins.

This is prepared for evaluation, not selected or promoted. Pre-trial exploration
checks, both measurement halves and all existing stress scenarios remain common
with the previous revised arm. Future pause/resume costs stay in the forecast.
"""
from ...scheduler.feedback_trials import evaluate_feedback_trial
from ..a10_feedback_latency.async_preparation import AsyncRevisedService
from ..a10_feedback_latency.policy import conservative_admission

POLICY_ID = "a10_incremental_posttrial_margin_v1"


def incremental_admission(*, measured_reserve_seconds, **kwargs):
    result = conservative_admission(measured_reserve_seconds=measured_reserve_seconds,
        include_historical_reserve=False, **kwargs)
    result["admission_policy"] = POLICY_ID
    result["opportunity_cost_note"] = "current remaining work already credits productive trials; future defer pause/resume still charged"
    return result


class IncrementalService(AsyncRevisedService):
    def _score_feedback_admission(self, trial, jobs, windows):
        candidate = self.store.get_job(trial.candidate_job_id)
        reserve = candidate.metadata.get("feedback_trial_reserve_seconds")
        return evaluate_feedback_trial(self, trial, jobs, windows,
            compare=lambda **kwargs: incremental_admission(measured_reserve_seconds=reserve, **kwargs))
