"""Collect and persist the timing evidence used by colocation trials.

This mixin isolates measurement concerns from the admission state machine. It
validates fresh epoch samples, computes bounded evidence windows, and stores
the resulting timing profiles for future placement decisions.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import isfinite

from ..domain import (
    BatchResolution,
    ColocationTimingProfile,
    PairProfile,
    TrainingJob,
    build_colocation_profile_key,
    parse_timestamp,
    utc_now,
)
from .service_state import ColocationTrialState, TrialEpochEvidence


class ColocationEvidenceMixin:
    """Collect trustworthy trial samples and retain reusable timing evidence."""

    @staticmethod
    def _remaining_epochs(job: TrainingJob) -> int | None:
        """Return remaining planned epochs, or None when the plan is unknown."""
        total = job.max_epochs or job.config.max_epochs
        if total is None:
            return None
        try:
            return max(0, int(total) - int(job.metadata.get("last_completed_epoch", 0)))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parsed_timestamp(value: object) -> datetime | None:
        """Parse persisted timestamps into timezone-aware UTC-compatible values."""
        try:
            parsed = parse_timestamp(str(value) if value else None)
        except (TypeError, ValueError):
            return None
        if parsed is not None and parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed

    def _trial_epoch_evidence(
        self,
        job: TrainingJob,
        trial: ColocationTrialState,
    ) -> TrialEpochEvidence:
        """Select fresh, bounded epoch samples for one trial member."""
        trial_started = self._parsed_timestamp(trial.started_at)
        trial_deadline = self._parsed_timestamp(trial.evidence_deadline_at)
        baseline_epoch = trial.member_start_epochs.get(job.job_id)
        required = self.settings.gpu_scheduler.colocation.trial_epochs
        if trial_started is None or trial_deadline is None or baseline_epoch is None:
            return TrialEpochEvidence(None, 0)

        candidate = job.job_id == trial.candidate_job_id
        by_epoch: dict[int, tuple[datetime, float]] = {}
        for sample in list(job.metadata.get("runtime_epoch_timing_history") or []):
            if not isinstance(sample, dict):
                continue
            try:
                epoch = int(sample.get("epoch", 0))
                seconds = float(sample.get("seconds", 0.0))
            except (TypeError, ValueError):
                continue
            if epoch <= baseline_epoch or not isfinite(seconds) or seconds <= 0:
                continue
            finished_at = self._parsed_timestamp(sample.get("finished_at"))
            if (
                finished_at is None
                or finished_at < trial_started
                or finished_at > trial_deadline
            ):
                continue
            interval_started = self._parsed_timestamp(sample.get("started_at"))
            if interval_started is None:
                if not candidate or str(sample.get("source")) != "runner_step_time":
                    continue
            elif interval_started < trial_started:
                continue
            previous = by_epoch.get(epoch)
            if previous is None or finished_at > previous[0]:
                by_epoch[epoch] = (finished_at, seconds)

        ordered = [
            seconds
            for _, (_, seconds) in sorted(
                by_epoch.items(),
                key=lambda item: (item[0], item[1][0]),
            )
        ]
        selected = tuple(ordered[-required:])
        if len(selected) < required:
            return TrialEpochEvidence(None, len(selected), selected)
        return TrialEpochEvidence(
            sum(selected) / len(selected), len(selected), selected
        )

    def _trial_evidence_timeout_seconds(
        self,
        candidate_solo_epoch_seconds: float,
        pretrial_epoch_seconds: dict[str, float],
    ) -> float:
        """Size the evidence window from expected member epoch rates."""
        settings = self.settings.gpu_scheduler.colocation
        valid_rates: list[float] = []
        for value in [candidate_solo_epoch_seconds, *pretrial_epoch_seconds.values()]:
            try:
                rate = float(value)
            except (TypeError, ValueError):
                continue
            if isfinite(rate) and rate > 0:
                valid_rates.append(rate)
        estimated_window = (
            3.0 * settings.trial_epochs * max(valid_rates)
            if valid_rates
            else settings.trial_evidence_timeout_min_seconds
        )
        return min(
            settings.trial_evidence_timeout_max_seconds,
            max(settings.trial_evidence_timeout_min_seconds, estimated_window),
        )

    def _trial_evidence_deadline(
        self,
        started_at: str,
        candidate_solo_epoch_seconds: float,
        pretrial_epoch_seconds: dict[str, float],
    ) -> str:
        """Calculate the persisted deadline for complete trial evidence."""
        started = self._parsed_timestamp(started_at) or datetime.now(timezone.utc)
        timeout = self._trial_evidence_timeout_seconds(
            candidate_solo_epoch_seconds,
            pretrial_epoch_seconds,
        )
        return (started + timedelta(seconds=timeout)).isoformat()

    def _persist_colocation_timing_profile(
        self,
        jobs: list[TrainingJob],
        rates: dict[str, float],
        trial: ColocationTrialState,
        *,
        sources: dict[str, str] | None = None,
        gain: float | None = None,
        decision: str | None = None,
    ) -> None:
        """Merge verified trial rates into reusable group and pair profiles."""
        descriptors = [self._member_descriptor(job, trial.backend_name) for job in jobs]
        profile_key = build_colocation_profile_key(
            self.store.hardware_key(), descriptors
        )
        stored_profile = self.store.get_colocation_timing_profile(profile_key)
        existing = (
            stored_profile
            if stored_profile is not None
            and self.planner.time_objective.profile_is_fresh(stored_profile)
            else None
        )
        if (
            existing is not None
            and existing.metadata.get("last_trial_id") == trial.trial_id
        ):
            return
        if sources is not None and not any(
            source != "exact_colocation_profile" for source in sources.values()
        ):
            return
        old_by_key: dict[tuple[str, int, str], dict[str, object]] = {}
        if existing is not None:
            for item in existing.member_timings:
                old_by_key[
                    (
                        str(item["signature"]),
                        int(item["batch_size"]),
                        str(item["backend_name"]),
                    )
                ] = item
        timings: list[dict[str, object]] = []
        for job, descriptor in zip(jobs, descriptors, strict=True):
            rate = rates.get(job.job_id)
            if rate is None:
                continue
            key = (
                str(descriptor["signature"]),
                int(descriptor["batch_size"]),
                str(descriptor["backend_name"]),
            )
            old = old_by_key.get(key)
            source = (sources or {}).get(job.job_id, "live_training")
            if source == "exact_colocation_profile" and old is not None:
                timings.append(dict(old))
                continue
            old_count = int(old.get("observations", 0)) if old else 0
            old_rate = float(old.get("seconds_per_epoch", 0.0)) if old else 0.0
            timings.append(
                {
                    **descriptor,
                    "seconds_per_epoch": ((old_rate * old_count) + rate)
                    / (old_count + 1),
                    "observations": old_count + 1,
                    "source": source,
                }
            )
        if len(timings) != len(jobs):
            return
        metadata = dict(existing.metadata) if existing is not None else {}
        recent_outcomes = list(metadata.get("recent_trial_outcomes") or [])
        if decision in {"accepted", "rejected"} and gain is not None:
            outcome = {
                "trial_id": trial.trial_id,
                "decision": decision,
                "gain": float(gain),
                "observed_at": utc_now(),
            }
            if decision == "accepted":
                recent_outcomes = [outcome]
            else:
                recent_outcomes.append(outcome)
            recent_outcomes = recent_outcomes[-16:]
        metadata.update(
            {
                "last_trial_id": trial.trial_id,
                "job_ids": [job.job_id for job in jobs],
                "evidence_policy": "fresh_member_epochs_v1",
                "recent_trial_outcomes": recent_outcomes,
            }
        )
        profile = ColocationTimingProfile.create(
            self.store.hardware_key(),
            descriptors,
            timings,
            observations=(existing.observations + 1) if existing else 1,
            metadata=metadata,
        )
        self.store.upsert_colocation_timing_profile(profile)
        if sources is not None and any(
            source == "exact_colocation_profile" for source in sources.values()
        ):
            return
        if len(jobs) != 2 or any(not job.packing.signature for job in jobs):
            return
        per_member_slowdown: dict[str, float] = {}
        per_signature_slowdown: dict[str, float] = {}
        slowdown_sources: dict[str, str] = {}
        batch_vector: dict[str, int] = {}
        for job in jobs:
            packed_rate = rates.get(job.job_id)
            batch_size = BatchResolution.resolved_batch_size(job)
            solo_profile = self.store.get_runtime_profile(
                job.packing.signature or job.job_id,
                resolved_batch_size=batch_size,
                backend_name="exclusive",
            )
            solo_rate: float | None = None
            if solo_profile is not None:
                if (
                    solo_profile.epoch_1_seconds is not None
                    and solo_profile.epoch_1_seconds > 0
                ):
                    solo_rate = float(solo_profile.epoch_1_seconds)
                elif (
                    solo_profile.avg_step_time_ms is not None
                    and solo_profile.avg_step_time_ms > 0
                    and solo_profile.steps_per_epoch is not None
                    and solo_profile.steps_per_epoch > 0
                ):
                    solo_rate = (
                        float(solo_profile.avg_step_time_ms)
                        * int(solo_profile.steps_per_epoch)
                        / 1000.0
                    )
            if packed_rate is None or packed_rate <= 0 or solo_rate is None:
                continue
            ratio = float(packed_rate) / solo_rate
            per_member_slowdown[job.job_id] = ratio
            per_signature_slowdown[job.packing.signature or job.job_id] = ratio
            slowdown_sources[job.job_id] = "measured_epoch_against_exclusive_profile"
            batch_vector[job.job_id] = batch_size
        if len(per_member_slowdown) != 2:
            return
        left_job, right_job = jobs
        existing_pair = self.store.get_pair_profile(
            left_job.packing.signature or left_job.job_id,
            right_job.packing.signature or right_job.job_id,
            backend_name=trial.backend_name,
        )
        self.store.upsert_pair_profile(
            PairProfile.create(
                left_job.packing.signature or left_job.job_id,
                right_job.packing.signature or right_job.job_id,
                backend_name=trial.backend_name,
                hardware_key=self.store.hardware_key(),
                compatible=True,
                observations=(existing_pair.observations + 1) if existing_pair else 1,
                slowdown_ratio=max(per_member_slowdown.values()),
                metadata={
                    "backend_name": trial.backend_name,
                    "batch_vector": batch_vector,
                    "per_member_slowdown": per_member_slowdown,
                    "per_signature_slowdown": per_signature_slowdown,
                    "slowdown_sources": slowdown_sources,
                    "colocation_profile_key": profile.profile_key,
                    "trial_id": trial.trial_id,
                },
            )
        )
