"""Opt-in pilot service; the base service keeps the exact v4 decision rule."""
from __future__ import annotations

from ...scheduler.feedback_trials import evaluate_feedback_trial, remaining_for_forecast
from ...scheduler.feedback_objective import derived_solo_slope
from ...scheduler.service import SchedulerService
from .policy import POLICY_ID, conservative_admission, trial_amortizes


class RevisedPilotService(SchedulerService):
    def _dispatch_plan(self, plan):
        if plan.trial_metadata.get("requires_live_trial"):
            candidate = self.store.get_job(plan.job_ids[0])
            try:
                reserve = candidate.metadata["feedback_trial_reserve_seconds"]
                attempts = candidate.metadata.get("pilot_exploration_attempts", 0)
                profile = self.planner.estimator.predicted_profile(candidate)
                allowed = trial_amortizes(
                    remaining_updates=remaining_for_forecast(candidate),
                    solo_updates_per_second=derived_solo_slope(profile["step_cost"]),
                    measured_reserve_seconds=reserve, attempts=attempts)
                reason = "remaining work cannot amortize measured trial reserve or trial budget used"
            except (ValueError, KeyError, TypeError, OverflowError, ZeroDivisionError):
                allowed, reason = False, "missing valid measured trial reserve/predicted update budget"
            if not allowed:
                self.store.update_job(candidate.job_id, metadata_updates={"cold_start_solo_required": True})
                self.event_logger.emit("pilot_trial_skipped", job_id=candidate.job_id,
                    payload={"admission_policy": POLICY_ID, "reason": reason})
                return False
            self.store.update_job(candidate.job_id, metadata_updates={
                "pilot_exploration_attempts": attempts + 1})
        return super()._dispatch_plan(plan)

    def _score_feedback_admission(self, trial, jobs, windows):
        candidate = self.store.get_job(trial.candidate_job_id)
        reserve = candidate.metadata.get("feedback_trial_reserve_seconds")

        def compare(**kwargs):
            # No-recovery applies to individual/mixed/group requests alike.
            # Finite scenarios are stressed; no confidence-bound claim is made.
            return conservative_admission(measured_reserve_seconds=reserve, **kwargs)

        return evaluate_feedback_trial(self, trial, jobs, windows, compare=compare)
