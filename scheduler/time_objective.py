"""Incremental colocation-gain objective for time-aware packing."""

from __future__ import annotations

from statistics import fmean

from ..config import SchedulerSettings
from ..domain import BatchResolution, TrainingJob, build_colocation_profile_key
from .candidate_generator import CandidateGenerator
from .compatibility import CompatibilityEvaluator
from .planner_types import EvaluatedGroup
from .resource_estimator import BatchOptionEstimate, ResourceEstimator


class TimeAwareObjectiveScorer:
    """Evaluate one shortest-first addition against sequential execution."""

    def __init__(
        self,
        settings: SchedulerSettings,
        estimator: ResourceEstimator,
        compatibility: CompatibilityEvaluator,
        candidates: CandidateGenerator,
    ) -> None:
        self.settings = settings
        self.estimator = estimator
        self.compatibility = compatibility
        self.candidates = candidates

    @staticmethod
    def remaining_epochs(job: TrainingJob) -> int | None:
        total = job.max_epochs or job.config.max_epochs
        if total is None:
            return None
        try:
            return max(0, int(total) - int(job.metadata.get("last_completed_epoch", 0)))
        except (TypeError, ValueError):
            return None

    def current_epoch_seconds(self, job: TrainingJob) -> tuple[float | None, str]:
        observed = job.metadata.get("runtime_observed_epoch_seconds")
        try:
            if observed is not None and float(observed) > 0:
                return float(observed), "live_epoch"
        except (TypeError, ValueError):
            pass
        backend = str(job.metadata.get("placement_backend") or "exclusive")
        batch = BatchResolution.resolved_batch_size(job)
        options = self.estimator.estimate_batch_options(job, backend, [batch])
        if options:
            return options[0].seconds_per_epoch, options[0].source
        return None, "missing"

    @staticmethod
    def member_descriptor(job: TrainingJob, *, backend_name: str, batch_size: int) -> dict[str, object]:
        return {
            "signature": job.packing.signature or job.job_id,
            "batch_size": int(batch_size),
            "backend_name": str(backend_name),
        }

    @staticmethod
    def _profile_epoch_seconds(profile: object, descriptor: dict[str, object]) -> float | None:
        timings = getattr(profile, "member_timings", [])
        values = [
            float(item["seconds_per_epoch"])
            for item in timings
            if str(item.get("signature")) == str(descriptor["signature"])
            and int(item.get("batch_size", 0)) == int(descriptor["batch_size"])
            and str(item.get("backend_name")) == str(descriptor["backend_name"])
            and float(item.get("seconds_per_epoch", 0.0)) > 0
        ]
        return fmean(values) if values else None

    @staticmethod
    def gain(
        active_jobs: list[TrainingJob],
        candidate: TrainingJob,
        *,
        active_epoch_seconds: dict[str, float],
        candidate_solo_epoch_seconds: float,
        packed_epoch_seconds: dict[str, float],
    ) -> tuple[float, float, float] | None:
        active_remaining: dict[str, int] = {}
        for job in active_jobs:
            remaining = TimeAwareObjectiveScorer.remaining_epochs(job)
            if remaining is None or job.job_id not in active_epoch_seconds or job.job_id not in packed_epoch_seconds:
                return None
            active_remaining[job.job_id] = remaining
        candidate_remaining = TimeAwareObjectiveScorer.remaining_epochs(candidate)
        if candidate_remaining is None or candidate.job_id not in packed_epoch_seconds:
            return None
        current_drain = max(
            (active_remaining[job.job_id] * active_epoch_seconds[job.job_id] for job in active_jobs),
            default=0.0,
        )
        candidate_solo = candidate_remaining * candidate_solo_epoch_seconds
        sequential = current_drain + candidate_solo
        packed = max(
            [active_remaining[job.job_id] * packed_epoch_seconds[job.job_id] for job in active_jobs]
            + [candidate_remaining * packed_epoch_seconds[candidate.job_id]],
            default=0.0,
        )
        if packed <= 0:
            return None
        return sequential / packed, sequential, packed

    def evaluate_incremental(
        self,
        candidate: TrainingJob,
        *,
        backend_name: str,
        active_jobs: list[TrainingJob],
        active_vram_mb: float,
        mandatory_anchor: TrainingJob | None,
    ) -> EvaluatedGroup | None:
        if not active_jobs or not self.compatibility.compatible_group([*active_jobs, candidate], backend_name=backend_name):
            return None
        active_rates: dict[str, float] = {}
        active_sources: dict[str, str] = {}
        for job in active_jobs:
            rate, source = self.current_epoch_seconds(job)
            if rate is None or rate <= 0 or self.remaining_epochs(job) is None:
                return None
            active_rates[job.job_id] = rate
            active_sources[job.job_id] = source

        try:
            batch_sizes = self.candidates.candidate_batch_sizes(
                candidate,
                scheduler_mode=self.settings.gpu_scheduler.mode,
            )
        except ValueError:
            return None
        options = self.estimator.pareto_prune(
            self.estimator.estimate_batch_options(candidate, backend_name, batch_sizes)
        )
        evaluations: list[tuple[tuple[object, ...], EvaluatedGroup]] = []
        for option in options:
            if active_vram_mb + option.avg_vram_mb > self.estimator.safe_budget_mb() + 1e-9:
                continue
            solo_options = self.estimator.estimate_batch_options(candidate, "exclusive", [option.batch_size])
            if not solo_options:
                continue
            candidate_solo_option = solo_options[0]
            descriptors = [
                self.member_descriptor(
                    job,
                    backend_name=str(job.metadata.get("placement_backend") or backend_name),
                    batch_size=BatchResolution.resolved_batch_size(job),
                )
                for job in active_jobs
            ]
            candidate_descriptor = self.member_descriptor(
                candidate,
                backend_name=backend_name,
                batch_size=option.batch_size,
            )
            descriptors.append(candidate_descriptor)
            profile_key = build_colocation_profile_key(self.estimator.repository.hardware_key(), descriptors)
            if candidate.metadata.get("colocation_unverified_profile_key") == profile_key:
                continue
            profile = self.estimator.repository.get_colocation_timing_profile(profile_key)
            predicted_gain: float | None = None
            sequential: float | None = None
            packed: float | None = None
            packed_rates: dict[str, float] = {}
            if profile is not None:
                for job, descriptor in zip([*active_jobs, candidate], descriptors, strict=True):
                    rate = self._profile_epoch_seconds(profile, descriptor)
                    if rate is not None:
                        packed_rates[job.job_id] = rate
                result = self.gain(
                    active_jobs,
                    candidate,
                    active_epoch_seconds=active_rates,
                    candidate_solo_epoch_seconds=candidate_solo_option.seconds_per_epoch,
                    packed_epoch_seconds=packed_rates,
                )
                if result is not None:
                    predicted_gain, sequential, packed = result

            if predicted_gain is None and not self.settings.gpu_scheduler.colocation.live_trial_enabled:
                continue

            rejected = predicted_gain is not None and predicted_gain + 1e-9 < self.settings.gpu_scheduler.colocation.min_gain
            breakdown: dict[str, object] = {
                "score": predicted_gain if predicted_gain is not None else 1.0,
                "gain": predicted_gain,
                "gain_threshold": self.settings.gpu_scheduler.colocation.min_gain,
                "sequential_drain_seconds": sequential,
                "packed_drain_seconds": packed,
                "active_pretrial_epoch_seconds": active_rates,
                "active_timing_sources": active_sources,
                "candidate_solo_epoch_seconds": candidate_solo_option.seconds_per_epoch,
                "candidate_solo_timing_source": candidate_solo_option.source,
                "candidate_remaining_epochs": option.remaining_epochs,
                "candidate_memory_mb": option.avg_vram_mb,
                "active_vram_mb": active_vram_mb,
                "colocation_profile_key": profile_key,
                "known_profile": profile is not None,
                "colocation_rejected": rejected,
                "requires_live_trial": not rejected and self.settings.gpu_scheduler.colocation.live_trial_enabled,
                "preexisting_job_ids": [job.job_id for job in active_jobs],
                "objective_version": self.settings.gpu_scheduler.objective.objective_version,
            }
            evaluated = EvaluatedGroup(
                jobs=[candidate],
                backend_name=backend_name,
                estimated_vram_mb=option.avg_vram_mb,
                objective_score=float(predicted_gain if predicted_gain is not None else 1.0),
                batch_overrides={candidate.job_id: option.batch_size},
                fallback_order=self.candidates.fallback_order([candidate], {candidate.job_id: option.batch_size}, backend_name),
                reason=("known colocation gain is below threshold" if rejected else "two-epoch colocation trial"),
                batch_estimates={candidate.job_id: option},
                score_breakdown=breakdown,
                mandatory_anchor_job_id=mandatory_anchor.job_id if mandatory_anchor else None,
                objective_version=self.settings.gpu_scheduler.objective.objective_version,
            )
            key = (
                1 if rejected else 0,
                0 if predicted_gain is not None else 1,
                -(predicted_gain or 0.0),
                option.remaining_runtime_seconds,
                option.avg_vram_mb,
                option.batch_size,
            )
            evaluations.append((key, evaluated))
        return min(evaluations, key=lambda item: item[0])[1] if evaluations else None

    def tie_key(self, group: EvaluatedGroup) -> tuple[object, ...]:
        return (
            -group.objective_score,
            group.estimated_vram_mb,
            tuple(sorted(group.batch_overrides.items())),
            group.backend_name,
        )
