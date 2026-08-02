"""Incremental colocation-gain objective for time-aware packing."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from statistics import fmean

from ..config import SchedulerSettings
from ..domain import BatchResolution, TrainingJob, build_colocation_profile_key, parse_timestamp
from .candidate_generator import CandidateGenerator
from .compatibility import CompatibilityEvaluator
from .planner_types import EvaluatedGroup
from .resource_estimator import ResourceEstimator


@dataclass(frozen=True, slots=True)
class EpochRateSet:
    epoch_seconds: dict[str, float]
    sources: dict[str, str]


@dataclass(frozen=True, slots=True)
class DrainPhase:
    member_ids: tuple[str, ...]
    duration_seconds: float
    epoch_seconds: dict[str, float]
    timing_sources: dict[str, str]
    completed_job_ids: tuple[str, ...]
    inherited_parent_rates: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "member_ids": list(self.member_ids),
            "duration_seconds": self.duration_seconds,
            "epoch_seconds": dict(self.epoch_seconds),
            "timing_sources": dict(self.timing_sources),
            "completed_job_ids": list(self.completed_job_ids),
            "inherited_parent_rates": self.inherited_parent_rates,
        }


@dataclass(frozen=True, slots=True)
class DrainProjection:
    total_seconds: float
    phases: tuple[DrainPhase, ...]


@dataclass(frozen=True, slots=True)
class ColocationGainResult:
    gain: float
    sequential_drain_seconds: float
    packed_drain_seconds: float
    sequential_phases: tuple[DrainPhase, ...]
    packed_phases: tuple[DrainPhase, ...]

    # Preserve the former three-tuple behavior for callers outside the scheduler.
    def __iter__(self):
        yield self.gain
        yield self.sequential_drain_seconds
        yield self.packed_drain_seconds

    def __getitem__(self, index: int) -> float:
        return (
            self.gain,
            self.sequential_drain_seconds,
            self.packed_drain_seconds,
        )[index]

    def __len__(self) -> int:
        return 3


TailRateResolver = Callable[[tuple[str, ...]], EpochRateSet | None]


def project_piecewise_drain(
    remaining_epochs: Mapping[str, float],
    initial_epoch_seconds: Mapping[str, float],
    *,
    initial_sources: Mapping[str, str] | None = None,
    tail_rate_resolver: TailRateResolver | None = None,
    epsilon: float = 1e-9,
) -> DrainProjection | None:
    """Project drain time while changing rates at membership boundaries.

    If a complete timing vector for a smaller membership is unavailable, the
    previous phase's rates are inherited. This is deliberately conservative:
    missing tail evidence must not invent a post-colocation speedup.
    """

    normalized_remaining: dict[str, float] = {}
    for job_id, value in remaining_epochs.items():
        try:
            epochs = float(value)
        except (TypeError, ValueError):
            return None
        if not isfinite(epochs) or epochs < 0:
            return None
        normalized_remaining[str(job_id)] = epochs

    original_member_ids = tuple(sorted(normalized_remaining))
    remaining = {
        job_id: epochs
        for job_id, epochs in normalized_remaining.items()
        if epochs > epsilon
    }
    if not remaining:
        return DrainProjection(total_seconds=0.0, phases=())

    member_ids = tuple(sorted(remaining))
    initial_membership_reduced = member_ids != original_member_ids
    resolved = (
        tail_rate_resolver(member_ids)
        if initial_membership_reduced and tail_rate_resolver is not None
        else None
    )
    try:
        complete_resolution = resolved is not None and all(
            job_id in resolved.epoch_seconds
            and isfinite(float(resolved.epoch_seconds[job_id]))
            and float(resolved.epoch_seconds[job_id]) > 0
            for job_id in member_ids
        )
    except (TypeError, ValueError):
        complete_resolution = False
    if complete_resolution and resolved is not None:
        rates = {
            job_id: float(resolved.epoch_seconds[job_id])
            for job_id in member_ids
        }
        sources = {
            job_id: str(resolved.sources.get(job_id) or "resolved_tail_rate")
            for job_id in member_ids
        }
        inherited = False
    else:
        rates = {}
        for job_id in member_ids:
            try:
                rate = float(initial_epoch_seconds[job_id])
            except (KeyError, TypeError, ValueError):
                return None
            if not isfinite(rate) or rate <= 0:
                return None
            rates[job_id] = rate
        sources = {
            job_id: (
                "inherited_parent_rate"
                if initial_membership_reduced
                else str((initial_sources or {}).get(job_id) or "initial_rate")
            )
            for job_id in member_ids
        }
        inherited = initial_membership_reduced

    phases: list[DrainPhase] = []
    total_seconds = 0.0
    while member_ids:
        time_to_finish = {
            job_id: remaining[job_id] * rates[job_id]
            for job_id in member_ids
        }
        duration = min(time_to_finish.values(), default=0.0)
        if not isfinite(duration) or duration <= epsilon:
            return None
        for job_id in member_ids:
            remaining[job_id] = max(0.0, remaining[job_id] - duration / rates[job_id])
        completed = tuple(
            job_id for job_id in member_ids if remaining[job_id] <= epsilon
        )
        if not completed:
            return None
        phases.append(
            DrainPhase(
                member_ids=member_ids,
                duration_seconds=duration,
                epoch_seconds=dict(rates),
                timing_sources=dict(sources),
                completed_job_ids=completed,
                inherited_parent_rates=inherited,
            )
        )
        total_seconds += duration
        completed_set = set(completed)
        next_members = tuple(job_id for job_id in member_ids if job_id not in completed_set)
        if not next_members:
            break

        resolved = tail_rate_resolver(next_members) if tail_rate_resolver is not None else None
        try:
            complete_resolution = resolved is not None and all(
                job_id in resolved.epoch_seconds
                and isfinite(float(resolved.epoch_seconds[job_id]))
                and float(resolved.epoch_seconds[job_id]) > 0
                for job_id in next_members
            )
        except (TypeError, ValueError):
            complete_resolution = False
        if complete_resolution and resolved is not None:
            rates = {
                job_id: float(resolved.epoch_seconds[job_id])
                for job_id in next_members
            }
            sources = {
                job_id: str(resolved.sources.get(job_id) or "resolved_tail_rate")
                for job_id in next_members
            }
            inherited = False
        else:
            rates = {job_id: rates[job_id] for job_id in next_members}
            sources = {job_id: "inherited_parent_rate" for job_id in next_members}
            inherited = True
        member_ids = next_members

    return DrainProjection(total_seconds=total_seconds, phases=tuple(phases))


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

    def profile_is_fresh(self, profile: object, *, now: datetime | None = None) -> bool:
        try:
            updated_at = parse_timestamp(getattr(profile, "updated_at", None))
        except (TypeError, ValueError):
            return False
        if updated_at is None:
            return False
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        return 0.0 <= (current - updated_at).total_seconds() <= (
            self.settings.gpu_scheduler.colocation.profile_rejection_ttl_seconds
        )

    def profile_rates_trusted(self, profile: object | None, *, now: datetime | None = None) -> bool:
        if profile is None or not self.profile_is_fresh(profile, now=now):
            return False
        try:
            observations = int(getattr(profile, "observations", 0))
        except (TypeError, ValueError):
            return False
        metadata = getattr(profile, "metadata", {})
        return (
            observations >= self.settings.gpu_scheduler.colocation.profile_rejection_min_bad_trials
            and isinstance(metadata, dict)
            and metadata.get("evidence_policy") == "fresh_member_epochs_v1"
        )

    def profile_rejection_trusted(self, profile: object | None, *, now: datetime | None = None) -> bool:
        if not self.profile_rates_trusted(profile, now=now):
            return False
        current = now or datetime.now(timezone.utc)
        metadata = getattr(profile, "metadata", {})
        history = list(metadata.get("recent_trial_outcomes") or []) if isinstance(metadata, dict) else []
        required = self.settings.gpu_scheduler.colocation.profile_rejection_min_bad_trials
        if len(history) < required:
            return False
        trial_ids: set[str] = set()
        for sample in history[-required:]:
            if not isinstance(sample, dict) or str(sample.get("decision")) != "rejected":
                return False
            trial_id = str(sample.get("trial_id") or "")
            if not trial_id or trial_id in trial_ids:
                return False
            trial_ids.add(trial_id)
            try:
                gain = float(sample.get("gain"))
            except (TypeError, ValueError):
                return False
            if (
                not isfinite(gain)
                or gain + 1e-9 >= self.settings.gpu_scheduler.colocation.min_gain
            ):
                return False
            try:
                observed_at = parse_timestamp(sample.get("observed_at"))
            except (TypeError, ValueError):
                return False
            if observed_at is None:
                return False
            if observed_at.tzinfo is None:
                observed_at = observed_at.replace(tzinfo=timezone.utc)
            age = (current - observed_at).total_seconds()
            if age < 0 or age > self.settings.gpu_scheduler.colocation.profile_rejection_ttl_seconds:
                return False
        return True

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
        active_epoch_sources: Mapping[str, str] | None = None,
        packed_epoch_sources: Mapping[str, str] | None = None,
        active_tail_rate_resolver: TailRateResolver | None = None,
        packed_tail_rate_resolver: TailRateResolver | None = None,
    ) -> ColocationGainResult | None:
        active_remaining: dict[str, float] = {}
        for job in active_jobs:
            remaining = TimeAwareObjectiveScorer.remaining_epochs(job)
            if remaining is None:
                return None
            if remaining > 0 and (
                job.job_id not in active_epoch_seconds or job.job_id not in packed_epoch_seconds
            ):
                return None
            active_remaining[job.job_id] = float(remaining)
        candidate_remaining = TimeAwareObjectiveScorer.remaining_epochs(candidate)
        if candidate_remaining is None:
            return None
        if candidate_remaining > 0 and candidate.job_id not in packed_epoch_seconds:
            return None
        if candidate_remaining > 0:
            try:
                candidate_solo_rate = float(candidate_solo_epoch_seconds)
            except (TypeError, ValueError):
                return None
            if not isfinite(candidate_solo_rate) or candidate_solo_rate <= 0:
                return None
        else:
            candidate_solo_rate = 0.0

        active_projection = project_piecewise_drain(
            active_remaining,
            active_epoch_seconds,
            initial_sources=active_epoch_sources,
            tail_rate_resolver=active_tail_rate_resolver,
        )
        if active_projection is None:
            return None
        candidate_solo = float(candidate_remaining) * candidate_solo_rate
        sequential = active_projection.total_seconds + candidate_solo
        candidate_phase = (
            (
                DrainPhase(
                    member_ids=(candidate.job_id,),
                    duration_seconds=candidate_solo,
                    epoch_seconds={candidate.job_id: candidate_solo_rate},
                    timing_sources={candidate.job_id: "candidate_exclusive_solo"},
                    completed_job_ids=(candidate.job_id,),
                ),
            )
            if candidate_solo > 0
            else ()
        )

        all_remaining = {
            **active_remaining,
            candidate.job_id: float(candidate_remaining),
        }
        packed_projection = project_piecewise_drain(
            all_remaining,
            packed_epoch_seconds,
            initial_sources=packed_epoch_sources,
            tail_rate_resolver=packed_tail_rate_resolver,
        )
        if packed_projection is None or packed_projection.total_seconds <= 0:
            return None
        return ColocationGainResult(
            gain=sequential / packed_projection.total_seconds,
            sequential_drain_seconds=sequential,
            packed_drain_seconds=packed_projection.total_seconds,
            sequential_phases=(*active_projection.phases, *candidate_phase),
            packed_phases=packed_projection.phases,
        )

    def _tail_rate_resolver(
        self,
        jobs: list[TrainingJob],
        descriptors: dict[str, dict[str, object]],
        *,
        singleton_fallbacks: Mapping[str, tuple[float, str]] | None = None,
    ) -> TailRateResolver:
        jobs_by_id = {job.job_id: job for job in jobs}
        fallbacks = dict(singleton_fallbacks or {})

        def resolve(member_ids: tuple[str, ...]) -> EpochRateSet | None:
            if len(member_ids) == 1:
                job_id = member_ids[0]
                job = jobs_by_id[job_id]
                descriptor = descriptors[job_id]
                options = self.estimator.estimate_batch_options(
                    job,
                    str(descriptor["backend_name"]),
                    [int(descriptor["batch_size"])],
                )
                if options and options[0].seconds_per_epoch > 0:
                    return EpochRateSet(
                        epoch_seconds={job_id: float(options[0].seconds_per_epoch)},
                        sources={job_id: options[0].source},
                    )
                fallback = fallbacks.get(job_id)
                if fallback is not None and fallback[0] > 0:
                    return EpochRateSet(
                        epoch_seconds={job_id: float(fallback[0])},
                        sources={job_id: str(fallback[1])},
                    )
                return None

            member_descriptors = [descriptors[job_id] for job_id in member_ids]
            profile_key = build_colocation_profile_key(
                self.estimator.repository.hardware_key(),
                member_descriptors,
            )
            profile = self.estimator.repository.get_colocation_timing_profile(profile_key)
            if not self.profile_rates_trusted(profile):
                return None
            rates: dict[str, float] = {}
            for job_id in member_ids:
                rate = self._profile_epoch_seconds(profile, descriptors[job_id])
                if rate is None or rate <= 0:
                    return None
                rates[job_id] = rate
            return EpochRateSet(
                epoch_seconds=rates,
                sources={job_id: "exact_colocation_profile" for job_id in member_ids},
            )

        return resolve

    def estimate_gain(
        self,
        active_jobs: list[TrainingJob],
        candidate: TrainingJob,
        *,
        backend_name: str,
        active_epoch_seconds: dict[str, float],
        candidate_solo_epoch_seconds: float,
        packed_epoch_seconds: dict[str, float],
        candidate_batch_size: int | None = None,
        active_epoch_sources: Mapping[str, str] | None = None,
        packed_epoch_sources: Mapping[str, str] | None = None,
    ) -> ColocationGainResult | None:
        descriptors = {
            job.job_id: self.member_descriptor(
                job,
                backend_name=str(job.metadata.get("placement_backend") or backend_name),
                batch_size=BatchResolution.resolved_batch_size(job),
            )
            for job in active_jobs
        }
        descriptors[candidate.job_id] = self.member_descriptor(
            candidate,
            backend_name=backend_name,
            batch_size=(
                int(candidate_batch_size)
                if candidate_batch_size is not None
                else BatchResolution.resolved_batch_size(candidate)
            ),
        )
        singleton_fallbacks: dict[str, tuple[float, str]] = {
            candidate.job_id: (candidate_solo_epoch_seconds, "candidate_exclusive_solo")
        }
        if len(active_jobs) == 1:
            active_job = active_jobs[0]
            active_rate = active_epoch_seconds.get(active_job.job_id)
            if active_rate is not None:
                singleton_fallbacks[active_job.job_id] = (
                    active_rate,
                    str(
                        (active_epoch_sources or {}).get(active_job.job_id)
                        or "active_pretrial_singleton"
                    ),
                )
        resolver = self._tail_rate_resolver(
            [*active_jobs, candidate],
            descriptors,
            singleton_fallbacks=singleton_fallbacks,
        )
        return self.gain(
            active_jobs,
            candidate,
            active_epoch_seconds=active_epoch_seconds,
            candidate_solo_epoch_seconds=candidate_solo_epoch_seconds,
            packed_epoch_seconds=packed_epoch_seconds,
            active_epoch_sources=active_epoch_sources,
            packed_epoch_sources=packed_epoch_sources,
            active_tail_rate_resolver=resolver,
            packed_tail_rate_resolver=resolver,
        )

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
        profile_now = datetime.now(timezone.utc)
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
            stored_profile = self.estimator.repository.get_colocation_timing_profile(profile_key)
            profile = (
                stored_profile
                if self.profile_rates_trusted(stored_profile, now=profile_now)
                else None
            )
            predicted_gain: float | None = None
            sequential: float | None = None
            packed: float | None = None
            packed_rates: dict[str, float] = {}
            if profile is not None:
                for job, descriptor in zip([*active_jobs, candidate], descriptors, strict=True):
                    rate = self._profile_epoch_seconds(profile, descriptor)
                    if rate is not None:
                        packed_rates[job.job_id] = rate
                result = self.estimate_gain(
                    active_jobs,
                    candidate,
                    backend_name=backend_name,
                    active_epoch_seconds=active_rates,
                    candidate_solo_epoch_seconds=candidate_solo_option.seconds_per_epoch,
                    packed_epoch_seconds=packed_rates,
                    candidate_batch_size=option.batch_size,
                    active_epoch_sources=active_sources,
                    packed_epoch_sources={
                        job_id: "exact_colocation_profile"
                        for job_id in packed_rates
                    },
                )
                if result is not None:
                    predicted_gain = result.gain
                    sequential = result.sequential_drain_seconds
                    packed = result.packed_drain_seconds

            if predicted_gain is None and not self.settings.gpu_scheduler.colocation.live_trial_enabled:
                continue

            predicted_below_threshold = (
                predicted_gain is not None
                and predicted_gain + 1e-9 < self.settings.gpu_scheduler.colocation.min_gain
            )
            rejected = predicted_below_threshold and self.profile_rejection_trusted(
                profile,
                now=profile_now,
            )
            if (
                predicted_below_threshold
                and not rejected
                and not self.settings.gpu_scheduler.colocation.live_trial_enabled
            ):
                continue
            breakdown: dict[str, object] = {
                "score": predicted_gain if predicted_gain is not None else 1.0,
                "gain": predicted_gain,
                "gain_threshold": self.settings.gpu_scheduler.colocation.min_gain,
                "sequential_drain_seconds": sequential,
                "packed_drain_seconds": packed,
                "sequential_drain_phases": (
                    [phase.to_dict() for phase in result.sequential_phases]
                    if profile is not None and result is not None
                    else []
                ),
                "packed_drain_phases": (
                    [phase.to_dict() for phase in result.packed_phases]
                    if profile is not None and result is not None
                    else []
                ),
                "active_pretrial_epoch_seconds": active_rates,
                "active_timing_sources": active_sources,
                "candidate_solo_epoch_seconds": candidate_solo_option.seconds_per_epoch,
                "candidate_solo_timing_source": candidate_solo_option.source,
                "candidate_remaining_epochs": option.remaining_epochs,
                "candidate_memory_mb": option.avg_vram_mb,
                "active_vram_mb": active_vram_mb,
                "colocation_profile_key": profile_key,
                "known_profile": stored_profile is not None,
                "trusted_profile": profile is not None,
                "trusted_rejection_profile": self.profile_rejection_trusted(
                    profile,
                    now=profile_now,
                ),
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
