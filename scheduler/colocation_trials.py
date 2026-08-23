"""Create, restart, stall, and cancel live colocation trials.

A proposed colocated job is admitted provisionally. This lifecycle layer owns
that trial's durable state and coordinates pauses when a verified placement is
unsafe; evidence collection and decision scoring live in adjacent mixins.
"""

from __future__ import annotations

import time

from ..domain import TrainingJob, build_colocation_profile_key, utc_now
from .planner_types import DispatchPlan
from .service_state import ColocationStallState, ColocationTrialState


class ColocationTrialMixin:
    """Own the durable lifecycle of a live colocation admission trial."""

    def _activate_colocation_stall(
        self,
        *,
        preexisting_job_ids: tuple[str, ...],
        candidate_job_id: str,
        profile_key: str,
        reason: str,
    ) -> None:
        """Block repeated additions to a membership that just failed validation."""
        self._colocation_stall = ColocationStallState(
            preexisting_job_ids=preexisting_job_ids,
            candidate_job_id=candidate_job_id,
            profile_key=profile_key,
            reason=reason,
        )
        self._persist_scheduler_decision_state()
        self.event_logger.emit(
            "colocation_admission_stalled",
            job_id=candidate_job_id,
            payload={
                "preexisting_job_ids": list(preexisting_job_ids),
                "profile_key": profile_key,
                "reason": reason,
            },
        )

    def _refresh_colocation_stall(self) -> None:
        """Release the rejection stall after active membership changes."""
        if self._colocation_stall is None:
            return
        active_ids = set(self._supervisor_active_job_ids())
        if all(
            job_id in active_ids
            for job_id in self._colocation_stall.preexisting_job_ids
        ):
            return
        previous = self._colocation_stall
        self._colocation_stall = None
        self._persist_scheduler_decision_state()
        self.event_logger.emit(
            "colocation_admission_resumed",
            payload={
                "previous_preexisting_job_ids": list(previous.preexisting_job_ids),
                "candidate_job_id": previous.candidate_job_id,
                "reason": "pre-trial pack membership changed",
            },
        )

    def _restart_colocation_trial(
        self, trial: ColocationTrialState, preexisting_jobs: list[TrainingJob]
    ) -> None:
        """Restart evidence collection for the current active membership."""
        candidate = self.store.get_job(trial.candidate_job_id)
        if candidate is None:
            self._colocation_trial = None
            self._persist_scheduler_decision_state()
            return
        start_epoch = int(candidate.metadata.get("last_completed_epoch", 0))
        total_epochs = (
            candidate.max_epochs or candidate.config.max_epochs or start_epoch
        )
        target_epoch = min(
            int(total_epochs),
            start_epoch + self.settings.gpu_scheduler.colocation.trial_epochs,
        )
        pretrial: dict[str, float] = {}
        for job in preexisting_jobs:
            rate = trial.pretrial_epoch_seconds.get(job.job_id)
            if rate is None:
                rate, _ = self.planner.time_objective.current_epoch_seconds(job)
            if rate is not None:
                pretrial[job.job_id] = rate
        descriptors = [
            self._member_descriptor(job, trial.backend_name)
            for job in [*preexisting_jobs, candidate]
        ]
        started_at = utc_now()
        member_start_epochs = {
            job.job_id: int(job.metadata.get("last_completed_epoch", 0))
            for job in [*preexisting_jobs, candidate]
        }
        restarted = ColocationTrialState(
            trial_id=f"trial-{candidate.job_id}-{time.monotonic_ns()}",
            candidate_job_id=candidate.job_id,
            preexisting_job_ids=tuple(job.job_id for job in preexisting_jobs),
            started_at=started_at,
            start_epoch=start_epoch,
            target_epoch=target_epoch,
            backend_name=trial.backend_name,
            profile_key=build_colocation_profile_key(
                self.store.hardware_key(), descriptors
            ),
            candidate_solo_epoch_seconds=trial.candidate_solo_epoch_seconds,
            pretrial_epoch_seconds=pretrial,
            member_start_epochs=member_start_epochs,
            evidence_deadline_at=self._trial_evidence_deadline(
                started_at,
                trial.candidate_solo_epoch_seconds,
                pretrial,
            ),
            scheduler_decision_mode=trial.scheduler_decision_mode,
            estimated_trial_cost_seconds=trial.estimated_trial_cost_seconds,
            setup_cost_seconds=trial.setup_cost_seconds,
        )
        self._colocation_trial = restarted
        self.store.update_job(
            candidate.job_id,
            metadata_updates={
                "colocation_trial": {**restarted.to_dict(), "decision": "pending"}
            },
        )
        self._persist_scheduler_decision_state()
        self.event_logger.emit(
            "colocation_trial_started",
            job_id=candidate.job_id,
            payload={
                **restarted.to_dict(),
                "reason": "pack membership changed; trial restarted",
            },
        )

    def _prepare_colocation_trial(
        self, plan: DispatchPlan, candidate: TrainingJob
    ) -> ColocationTrialState | None:
        """Persist the decision barrier needed before provisional dispatch."""
        metadata = plan.trial_metadata or plan.objective_breakdown
        if not bool(metadata.get("requires_live_trial")):
            return None
        start_epoch = int(candidate.metadata.get("last_completed_epoch", 0))
        total_epochs = candidate.max_epochs or candidate.config.max_epochs
        if total_epochs is None:
            return None
        target_epoch = min(
            int(total_epochs),
            start_epoch + self.settings.gpu_scheduler.colocation.trial_epochs,
        )
        preexisting_job_ids = tuple(
            str(job_id) for job_id in metadata.get("preexisting_job_ids", [])
        )
        pretrial_epoch_seconds = {
            str(job_id): float(seconds)
            for job_id, seconds in dict(
                metadata.get("pretrial_epoch_seconds")
                or metadata.get("active_pretrial_epoch_seconds")
                or {}
            ).items()
        }
        member_start_epochs = {candidate.job_id: start_epoch}
        for job_id in preexisting_job_ids:
            active_job = self.store.get_job(job_id)
            if active_job is not None:
                member_start_epochs[job_id] = int(
                    active_job.metadata.get("last_completed_epoch", 0)
                )
        started_at = utc_now()
        candidate_solo_epoch_seconds = float(
            metadata.get("candidate_solo_epoch_seconds") or 0.0
        )
        trial = ColocationTrialState(
            trial_id=f"trial-{candidate.job_id}-{time.monotonic_ns()}",
            candidate_job_id=candidate.job_id,
            preexisting_job_ids=preexisting_job_ids,
            started_at=started_at,
            start_epoch=start_epoch,
            target_epoch=target_epoch,
            backend_name=plan.backend_name,
            profile_key=str(
                metadata.get("profile_key")
                or metadata.get("colocation_profile_key")
                or ""
            ),
            candidate_solo_epoch_seconds=candidate_solo_epoch_seconds,
            pretrial_epoch_seconds=pretrial_epoch_seconds,
            member_start_epochs=member_start_epochs,
            evidence_deadline_at=self._trial_evidence_deadline(
                started_at,
                candidate_solo_epoch_seconds,
                pretrial_epoch_seconds,
            ),
            scheduler_decision_mode=str(
                plan.objective_breakdown.get("scheduler_decision_mode")
                or "baseline"
            ),
            estimated_trial_cost_seconds=float(
                plan.objective_breakdown.get("estimated_trial_cost_seconds")
                or 0.0
            ),
            setup_cost_seconds=float(
                self.settings.gpu_scheduler.source_trial_ranking.estimated_setup_seconds
                if plan.objective_breakdown.get("scheduler_decision_mode")
                == "backend_awared"
                else 0.0
            ),
        )
        self._colocation_trial = trial
        self.store.update_job(
            candidate.job_id,
            metadata_updates={
                "colocation_trial": {**trial.to_dict(), "decision": "pending"}
            },
        )
        self._persist_scheduler_decision_state()
        return trial

    def _cancel_prepared_colocation_trial(
        self, trial: ColocationTrialState | None, *, reason: str
    ) -> None:
        """Mark an undispatched prepared trial as cancelled and clear it."""
        if trial is None:
            return
        current = self.store.get_job(trial.candidate_job_id)
        if current is not None:
            self.store.update_job(
                current.job_id,
                metadata_updates={
                    "colocation_trial": {
                        **trial.to_dict(),
                        "decision": "cancelled",
                        "reason": reason,
                    }
                },
            )
        if (
            self._colocation_trial is not None
            and self._colocation_trial.trial_id == trial.trial_id
        ):
            self._colocation_trial = None
            self._persist_scheduler_decision_state()
