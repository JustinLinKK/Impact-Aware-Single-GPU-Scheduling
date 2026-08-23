"""Evaluate colocation trials and apply their admission decisions.

The scheduler calls this mixin once per service tick. It advances a pending
trial only when every member supplies fresh timing evidence, then accepts,
rejects, extends, or times out the candidate without mixing those decisions
with profile persistence or trial construction.
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite

from ..domain import BatchResolution, JobStatus
from .planner_types import DispatchPlan


class ColocationDecisionMixin:
    """Advance pending trials and enforce the resulting placement decision."""

    def _evaluate_colocation_trial(self) -> None:
        """Advance the persisted live-trial state machine by one service tick.

        A trial waits for fresh epochs from every current member. Complete
        evidence is scored; incomplete evidence is extended, timed out, or
        marked unverified if the newcomer finishes before the barrier.
        """
        self._refresh_colocation_stall()
        trial = self._colocation_trial
        if trial is None:
            return
        candidate = self.store.get_job(trial.candidate_job_id)
        active_ids = set(self._supervisor_active_job_ids())
        if candidate is None:
            self._colocation_trial = None
            self._persist_scheduler_decision_state()
            return
        trial_metadata = dict(candidate.metadata.get("colocation_trial") or {})
        if str(trial_metadata.get("decision")) == "timeout":
            self.store.update_job(
                candidate.job_id,
                metadata_updates={
                    "colocation_unverified_profile_key": trial.profile_key
                },
            )
            self._colocation_trial = None
            self._persist_scheduler_decision_state()
            return
        candidate_finished = self._remaining_epochs(candidate) == 0
        if candidate.job_id not in active_ids and not candidate_finished:
            self._colocation_trial = None
            self._persist_scheduler_decision_state()
            return

        current_preexisting = [
            job for job in self._active_jobs() if job.job_id != candidate.job_id
        ]
        if set(job.job_id for job in current_preexisting) != set(
            trial.preexisting_job_ids
        ):
            if candidate_finished:
                reason = "newcomer completed while colocation membership changed"
                self.store.update_job(
                    candidate.job_id,
                    metadata_updates={
                        "colocation_trial": {
                            **trial.to_dict(),
                            "decision": "completed_unverified",
                            "reason": reason,
                        },
                        "colocation_unverified_profile_key": trial.profile_key,
                    },
                )
                self.event_logger.emit(
                    "colocation_trial_completed_unverified",
                    job_id=candidate.job_id,
                    payload={"trial_id": trial.trial_id, "reason": reason},
                )
                self._colocation_trial = None
                self._persist_scheduler_decision_state()
                return
            if not current_preexisting:
                accepted = {
                    **dict(candidate.metadata.get("colocation_trial") or {}),
                    "decision": "accepted",
                    "reason": "newcomer became stack anchor",
                }
                self.store.update_job(
                    candidate.job_id, metadata_updates={"colocation_trial": accepted}
                )
                self._colocation_trial = None
                self._persist_scheduler_decision_state()
                return
            self._restart_colocation_trial(trial, current_preexisting)
            return

        expected_members = {trial.candidate_job_id, *trial.preexisting_job_ids}
        if (
            set(trial.member_start_epochs) != expected_members
            or self._parsed_timestamp(trial.evidence_deadline_at) is None
        ):
            self._restart_colocation_trial(trial, current_preexisting)
            return

        all_jobs = [*current_preexisting, candidate]
        evidence = {
            job.job_id: self._trial_epoch_evidence(job, trial) for job in all_jobs
        }
        packed_rates: dict[str, float] = {}
        packed_sources: dict[str, str] = {}
        for job in all_jobs:
            rate = evidence[job.job_id].seconds_per_epoch
            if rate is not None and isfinite(rate) and rate > 0:
                packed_rates[job.job_id] = rate
                packed_sources[job.job_id] = "fresh_trial_epoch_average"

        required_samples = self.settings.gpu_scheduler.colocation.trial_epochs
        evidence_counts = {
            job_id: item.sample_count for job_id, item in evidence.items()
        }
        evidence_samples = {
            job_id: list(item.samples) for job_id, item in evidence.items()
        }
        evidence_complete = all(
            item.sample_count >= required_samples and item.seconds_per_epoch is not None
            for item in evidence.values()
        )
        candidate_remaining = self._remaining_epochs(candidate)
        completed_epoch = int(candidate.metadata.get("last_completed_epoch", 0))
        deadline = self._parsed_timestamp(trial.evidence_deadline_at)
        deadline_expired = (
            deadline is not None and datetime.now(timezone.utc) >= deadline
        )

        if not evidence_complete:
            missing_members = sorted(
                job_id
                for job_id, item in evidence.items()
                if item.sample_count < required_samples
            )
            progress_payload = {
                "trial_id": trial.trial_id,
                "profile_key": trial.profile_key,
                "required_samples_per_member": required_samples,
                "evidence_counts": evidence_counts,
                "evidence_samples": evidence_samples,
                "missing_member_ids": missing_members,
                "evidence_deadline_at": trial.evidence_deadline_at,
            }
            if candidate_remaining == 0:
                decision = {
                    **trial.to_dict(),
                    "decision": "completed_unverified",
                    "reason": "newcomer completed before every member supplied fresh timing evidence",
                    "evidence": progress_payload,
                }
                self.store.update_job(
                    candidate.job_id,
                    metadata_updates={
                        "colocation_trial": decision,
                        "colocation_unverified_profile_key": trial.profile_key,
                    },
                )
                self.event_logger.emit(
                    "colocation_trial_completed_unverified",
                    job_id=candidate.job_id,
                    payload=progress_payload,
                )
                self._colocation_trial = None
                self._persist_scheduler_decision_state()
                return
            if deadline_expired:
                reason = "colocation trial evidence deadline expired"
                pause_requested = self.supervisor.request_pause(
                    candidate.job_id,
                    reason=reason,
                    hold=False,
                )
                decision = {
                    **trial.to_dict(),
                    "decision": "timeout",
                    "reason": reason,
                    "evidence": progress_payload,
                }
                self.store.update_job(
                    candidate.job_id,
                    status=JobStatus.PAUSING if pause_requested else candidate.status,
                    reason=reason,
                    hold=False,
                    metadata_updates={
                        "colocation_trial": decision,
                        "colocation_unverified_profile_key": trial.profile_key,
                    },
                )
                self.event_logger.emit(
                    "colocation_trial_rejected",
                    job_id=candidate.job_id,
                    payload={**progress_payload, "reason": reason},
                )
                self._colocation_trial = None
                self._persist_scheduler_decision_state()
                return
            if completed_epoch >= trial.target_epoch:
                total_epochs = (
                    candidate.max_epochs
                    or candidate.config.max_epochs
                    or completed_epoch
                )
                next_target = min(int(total_epochs), completed_epoch + 1)
                if next_target > trial.target_epoch:
                    trial.target_epoch = next_target
                    pending = {
                        **trial.to_dict(),
                        "decision": "pending",
                        "reason": "collecting fresh timing evidence from every trial member",
                        "evidence": progress_payload,
                    }
                    self.store.update_job(
                        candidate.job_id,
                        metadata_updates={"colocation_trial": pending},
                    )
                    self._persist_scheduler_decision_state()
                    self.event_logger.emit(
                        "colocation_trial_extended",
                        job_id=candidate.job_id,
                        payload={**progress_payload, "target_epoch": next_target},
                    )
            return

        result = self.planner.time_objective.estimate_gain(
            current_preexisting,
            candidate,
            backend_name=trial.backend_name,
            active_epoch_seconds=trial.pretrial_epoch_seconds,
            candidate_solo_epoch_seconds=trial.candidate_solo_epoch_seconds,
            packed_epoch_seconds=packed_rates,
            candidate_batch_size=BatchResolution.resolved_batch_size(candidate),
            active_epoch_sources={
                job_id: "pretrial_epoch_rate" for job_id in trial.pretrial_epoch_seconds
            },
            packed_epoch_sources=packed_sources,
        )
        trial_wall_seconds = 0.0
        started_at = self._parsed_timestamp(trial.started_at)
        if started_at is not None:
            trial_wall_seconds = max(
                0.0, (datetime.now(timezone.utc) - started_at).total_seconds()
            )
        charged_trial_seconds = (
            trial_wall_seconds + trial.setup_cost_seconds
            if trial.scheduler_decision_mode == "backend_awared"
            else 0.0
        )
        if result is None:
            gain = 0.0
            sequential = None
            packed = None
            reason = "colocation trial lacked complete timing evidence"
        else:
            gain = result.sequential_drain_seconds / max(
                1e-9, result.packed_drain_seconds + charged_trial_seconds
            )
            sequential = result.sequential_drain_seconds
            packed = result.packed_drain_seconds + charged_trial_seconds
            reason = (
                "colocation gain accepted"
                if gain + 1e-9 >= self.settings.gpu_scheduler.colocation.min_gain
                else "colocation gain below threshold"
            )
        payload = {
            "trial_id": trial.trial_id,
            "gain": gain,
            "gain_threshold": self.settings.gpu_scheduler.colocation.min_gain,
            "sequential_drain_seconds": sequential,
            "packed_drain_seconds": packed,
            "measured_packed_drain_seconds_before_trial_charge": (
                result.packed_drain_seconds if result is not None else None
            ),
            "trial_wall_seconds": trial_wall_seconds,
            "setup_cost_seconds": trial.setup_cost_seconds,
            "charged_trial_seconds": charged_trial_seconds,
            "sequential_drain_phases": (
                [phase.to_dict() for phase in result.sequential_phases]
                if result is not None
                else []
            ),
            "packed_drain_phases": (
                [phase.to_dict() for phase in result.packed_phases]
                if result is not None
                else []
            ),
            "packed_epoch_seconds": packed_rates,
            "packed_epoch_sources": packed_sources,
            "pretrial_epoch_seconds": trial.pretrial_epoch_seconds,
            "candidate_solo_epoch_seconds": trial.candidate_solo_epoch_seconds,
            "profile_key": trial.profile_key,
            "required_samples_per_member": required_samples,
            "evidence_counts": evidence_counts,
            "evidence_samples": evidence_samples,
            "evidence_deadline_at": trial.evidence_deadline_at,
        }
        self.event_logger.emit(
            "colocation_gain_evaluated", job_id=candidate.job_id, payload=payload
        )
        self._persist_colocation_timing_profile(
            all_jobs,
            packed_rates,
            trial,
            sources=packed_sources,
            gain=gain if result is not None else None,
            decision=(
                "accepted"
                if result is not None
                and gain + 1e-9
                >= self.settings.gpu_scheduler.colocation.min_gain
                else "rejected" if result is not None else None
            ),
        )
        newcomer_finished = self._remaining_epochs(candidate) == 0
        if (
            newcomer_finished
            or gain + 1e-9 >= self.settings.gpu_scheduler.colocation.min_gain
        ):
            if newcomer_finished:
                reason = "newcomer completed during colocation trial"
            decision = {
                **trial.to_dict(),
                "decision": "accepted",
                "reason": reason,
                "result": payload,
            }
            self.store.update_job(
                candidate.job_id, metadata_updates={"colocation_trial": decision}
            )
            self.event_logger.emit(
                "colocation_trial_accepted", job_id=candidate.job_id, payload=payload
            )
            if (
                result is not None
                and gain + 1e-9 >= self.settings.gpu_scheduler.colocation.min_gain
            ):
                self._stage_successful_pattern(
                    all_jobs, backend_name=trial.backend_name
                )
            self._colocation_trial = None
            self._persist_scheduler_decision_state()
            return

        pause_requested = self.supervisor.request_pause(
            candidate.job_id, reason=reason, hold=False
        )
        decision = {
            **trial.to_dict(),
            "decision": "rejected",
            "reason": reason,
            "result": payload,
        }
        self.store.update_job(
            candidate.job_id,
            status=JobStatus.PAUSING if pause_requested else candidate.status,
            reason=reason,
            hold=False,
            metadata_updates={
                "colocation_trial": decision,
                "colocation_unverified_profile_key": (
                    trial.profile_key if result is None else None
                ),
            },
        )
        self.event_logger.emit(
            "colocation_trial_rejected", job_id=candidate.job_id, payload=payload
        )
        if result is not None:
            self._placement_replay.pending_observation = None
            self._record_pattern_observation(
                self._build_pattern_observation(
                    current_preexisting,
                    target_width=len(current_preexisting),
                    backend_name=trial.backend_name,
                    reason="verified_addition_rejected",
                )
            )
            self._activate_colocation_stall(
                preexisting_job_ids=trial.preexisting_job_ids,
                candidate_job_id=candidate.job_id,
                profile_key=trial.profile_key,
                reason=reason,
            )
        self._colocation_trial = None
        self._persist_scheduler_decision_state()

    def _known_colocation_rejection(self, plan: DispatchPlan) -> bool:
        """Reject a plan immediately when trusted profile evidence disproves it."""
        if (
            not bool(plan.objective_breakdown.get("colocation_rejected"))
            or not plan.job_ids
        ):
            return False
        candidate_job_id = plan.job_ids[0]
        preexisting_job_ids = tuple(
            str(job_id)
            for job_id in plan.objective_breakdown.get("preexisting_job_ids", [])
        )
        profile_key = str(plan.objective_breakdown.get("colocation_profile_key") or "")
        payload = {
            "gain": plan.objective_breakdown.get("gain"),
            "gain_threshold": plan.objective_breakdown.get("gain_threshold"),
            "sequential_drain_seconds": plan.objective_breakdown.get(
                "sequential_drain_seconds"
            ),
            "packed_drain_seconds": plan.objective_breakdown.get(
                "packed_drain_seconds"
            ),
            "sequential_drain_phases": plan.objective_breakdown.get(
                "sequential_drain_phases",
                [],
            ),
            "packed_drain_phases": plan.objective_breakdown.get(
                "packed_drain_phases",
                [],
            ),
            "profile_key": profile_key,
            "preexisting_job_ids": list(preexisting_job_ids),
            "source": "exact_colocation_profile",
        }
        self.store.update_job(
            candidate_job_id,
            status=JobStatus.PAUSED,
            reason="known colocation gain is below threshold",
            hold=False,
            metadata_updates={
                "colocation_trial": {
                    "decision": "rejected",
                    "reason": "known colocation gain is below threshold",
                    "result": payload,
                }
            },
        )
        self.event_logger.emit(
            "colocation_gain_evaluated", job_id=candidate_job_id, payload=payload
        )
        self.event_logger.emit(
            "colocation_trial_rejected", job_id=candidate_job_id, payload=payload
        )
        self._activate_colocation_stall(
            preexisting_job_ids=preexisting_job_ids,
            candidate_job_id=candidate_job_id,
            profile_key=profile_key,
            reason="known colocation gain is below threshold",
        )
        return True
