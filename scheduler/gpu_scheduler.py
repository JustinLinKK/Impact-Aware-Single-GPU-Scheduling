"""GPU-aware placement planning for the local scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Iterable

from ..schemas import (
    BATCH_PROBE_SEARCH_MODE_POWER_OF_TWO,
    PairProfile,
    SoloProfile,
    TrainingJob,
    build_batch_probe_key,
    build_batch_probe_shape_signature,
    build_group_signature,
    normalize_batch_probe_search_mode,
)
from ..settings import (
    SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED,
    SCHEDULER_MODE_PARALLEL_DEFAULT,
    SCHEDULER_MODE_SERIAL_BASIC,
    SCHEDULER_MODE_SERIAL_BATCH_OPTIMIZED,
    SchedulerSettings,
)
from ..storage.sqlite_store import SQLiteStateStore
from .policies import SchedulingPolicy
from .queue import RunnableJobQueue


@dataclass(slots=True)
class PlacementPlan:
    mode: str
    backend_name: str
    job_ids: tuple[str, ...]
    reason: str
    batch_overrides: dict[str, int] = field(default_factory=dict)
    fallback_order: list[str] = field(default_factory=list)


def _profile_peak_vram_mb(job: TrainingJob, profile: SoloProfile | None) -> int:
    if profile and profile.peak_vram_mb is not None:
        return int(profile.peak_vram_mb)
    if job.resource_requirements.estimated_vram_mb is not None:
        return int(job.resource_requirements.estimated_vram_mb)
    return 0


def compatibility_score(
    primary_job: TrainingJob,
    partner_job: TrainingJob,
    primary_profile: SoloProfile,
    partner_profile: SoloProfile,
    pair_profile: PairProfile | None,
    settings: SchedulerSettings,
) -> float:
    thresholds = settings.gpu_scheduler.thresholds
    primary_util = float(primary_profile.avg_gpu_utilization or 0.0)
    partner_util = float(partner_profile.avg_gpu_utilization or 0.0)
    if primary_util >= thresholds.pack_reject_sm_active_ge:
        return float("-inf")
    if partner_util >= thresholds.pack_reject_sm_active_ge:
        return float("-inf")
    if pair_profile is not None:
        if pair_profile.on_cooldown() or not pair_profile.compatible:
            return float("-inf")
        if pair_profile.slowdown_ratio is not None and pair_profile.slowdown_ratio > thresholds.pack_reject_max_slowdown:
            return float("-inf")
    util_headroom = max(0.0, 1.0 - max(primary_util, partner_util))
    priority_bonus = 0.01 * max(0, partner_job.priority)
    memory_budget_mb = settings.gpu_scheduler.memory.safe_vram_budget_gib * 1024.0
    memory_penalty = (
        _profile_peak_vram_mb(primary_job, primary_profile) + _profile_peak_vram_mb(partner_job, partner_profile)
    ) / memory_budget_mb if memory_budget_mb > 0 else 0.0
    return 1.0 + util_headroom + priority_bonus - memory_penalty


@dataclass(slots=True)
class EvaluatedGroup:
    jobs: list[TrainingJob]
    backend_name: str
    estimated_vram_mb: float
    objective_score: float
    batch_overrides: dict[str, int]
    fallback_order: list[str]
    reason: str


class GpuPlacementPlanner:
    """Decide between exclusive dispatch and packed execution."""

    def __init__(self, settings: SchedulerSettings, store: SQLiteStateStore, policy: SchedulingPolicy):
        self.settings = settings
        self.store = store
        self.policy = policy

    def _safe_budget_mb(self) -> float:
        return self.settings.gpu_scheduler.memory.safe_vram_budget_gib * 1024.0

    def _pack_eligible(self, job: TrainingJob, *, backend_name: str | None = None) -> bool:
        if not (job.packing.eligible and job.packing.signature):
            return False
        if backend_name is None:
            return True
        return job.packing.allows_backend(backend_name)

    def _job_batch_param_name(self, job: TrainingJob) -> str:
        return job.batch_probe.batch_param_name or "batch_size"

    def _resolved_batch_size(self, job: TrainingJob) -> int:
        batch_param_name = self._job_batch_param_name(job)
        if job.metadata.get("resolved_batch_size") is not None:
            try:
                return max(1, int(job.metadata["resolved_batch_size"]))
            except (TypeError, ValueError):
                pass
        raw_value = job.config.runner_kwargs.get(batch_param_name)
        try:
            return max(1, int(raw_value))
        except (TypeError, ValueError):
            return 1

    def _shape_signature(self, job: TrainingJob) -> str:
        return build_batch_probe_shape_signature(job)

    def _model_key(self, job: TrainingJob) -> str:
        return str(job.batch_probe.model_key or job.baseline_model_id)

    def _backend_candidates(self, jobs: list[TrainingJob], *, backend_available: dict[str, bool]) -> list[str]:
        if len(jobs) == 1:
            return ["exclusive"]
        candidates: list[str] = []
        for backend_name in self.settings.gpu_scheduler.backend_priority:
            if backend_name == "exclusive":
                continue
            if not backend_available.get(backend_name, False):
                continue
            if all(self._pack_eligible(job, backend_name=backend_name) for job in jobs):
                candidates.append(backend_name)
        return candidates

    def _solo_profile(self, job: TrainingJob) -> SoloProfile | None:
        if not job.packing.signature:
            return None
        return self.store.get_solo_profile(job.packing.signature)

    def _has_memory_estimate(self, job: TrainingJob, backend_name: str) -> bool:
        return self._estimate_peak_vram_mb(job, self._resolved_batch_size(job), backend_name) > 0.0

    def _estimate_peak_vram_mb(self, job: TrainingJob, batch_size: int, backend_name: str) -> float:
        hardware = self.store.hardware_profile()
        observation = self.store.get_batch_size_observation(
            model_key=self._model_key(job),
            shape_signature=self._shape_signature(job),
            hardware_key=hardware.hardware_key,
            backend_name=backend_name,
            batch_size=batch_size,
        )
        if observation and observation.peak_vram_mb is not None:
            return float(observation.peak_vram_mb)

        related = self.store.list_batch_size_observations(
            model_key=self._model_key(job),
            shape_signature=self._shape_signature(job),
            hardware_key=hardware.hardware_key,
            backend_name=backend_name,
        )
        candidates = [item for item in related if item.peak_vram_mb is not None and item.batch_size > 0]
        if candidates:
            nearest = min(candidates, key=lambda item: abs(item.batch_size - batch_size))
            return float(nearest.peak_vram_mb) * (float(batch_size) / float(max(1, nearest.batch_size)))

        device_type = hardware.gpu_name
        search_mode = normalize_batch_probe_search_mode(job.batch_probe.search_mode or self.settings.gpu_scheduler.batch_probe_search_mode)
        probe_key = build_batch_probe_key(self._model_key(job), device_type, self._shape_signature(job), search_mode=search_mode)
        batch_profile = self.store.get_batch_probe_profile(probe_key)
        if batch_profile and batch_profile.peak_vram_mb is not None:
            base_batch = max(1, int(batch_profile.resolved_batch_size))
            return float(batch_profile.peak_vram_mb) * (float(batch_size) / float(base_batch))

        solo_profile = self._solo_profile(job)
        if solo_profile and solo_profile.peak_vram_mb is not None:
            base_batch = max(1, self._resolved_batch_size(job))
            return float(solo_profile.peak_vram_mb) * (float(batch_size) / float(base_batch))

        if job.resource_requirements.estimated_vram_mb is not None:
            base_batch = max(1, self._resolved_batch_size(job))
            return float(job.resource_requirements.estimated_vram_mb) * (float(batch_size) / float(base_batch))
        return 0.0

    def _compatible_group(self, jobs: list[TrainingJob]) -> bool:
        if len(jobs) <= 1:
            return True
        thresholds = self.settings.gpu_scheduler.thresholds
        for left_job, right_job in combinations(jobs, 2):
            left_profile = self._solo_profile(left_job)
            right_profile = self._solo_profile(right_job)
            pair_profile = self.store.get_pair_profile(left_job.packing.signature or "", right_job.packing.signature or "")
            if left_profile is None or right_profile is None:
                if pair_profile is not None and (pair_profile.on_cooldown() or not pair_profile.compatible):
                    return False
                continue
            score = compatibility_score(left_job, right_job, left_profile, right_profile, pair_profile, self.settings)
            if score == float("-inf"):
                return False
            if pair_profile and pair_profile.slowdown_ratio is not None and pair_profile.slowdown_ratio > thresholds.pack_reject_max_slowdown:
                return False
        return True

    def _candidate_batch_sizes(self, job: TrainingJob) -> list[int]:
        requested = self._resolved_batch_size(job)
        min_batch = max(1, int(self.settings.gpu_scheduler.parallel_optimizer.min_batch_size))
        explicit_cap = job.config.runner_kwargs.get("probe_max_batch_size", self.settings.gpu_scheduler.batch_probe_max_batch_size)
        max_batch = int(explicit_cap) if explicit_cap is not None else max(
            requested,
            requested * max(1, int(self.settings.gpu_scheduler.parallel_optimizer.max_batch_multiplier)),
        )
        if max_batch < min_batch:
            max_batch = min_batch
        search_mode = normalize_batch_probe_search_mode(self.settings.gpu_scheduler.parallel_optimizer.batch_search_mode)
        if search_mode == BATCH_PROBE_SEARCH_MODE_POWER_OF_TWO:
            values: list[int] = []
            candidate = 1
            while candidate < min_batch:
                candidate *= 2
            while candidate <= max_batch:
                values.append(candidate)
                candidate *= 2
            if requested not in values and min_batch <= requested <= max_batch:
                values.append(requested)
            return sorted(set(values))
        return list(range(min_batch, max_batch + 1))

    def _fallback_order(self, jobs: list[TrainingJob], batch_overrides: dict[str, int], backend_name: str) -> list[str]:
        ranked = sorted(
            jobs,
            key=lambda job: (
                job.priority,
                -self._estimate_peak_vram_mb(job, batch_overrides.get(job.job_id, self._resolved_batch_size(job)), backend_name),
                job.queue_sequence,
            ),
        )
        return [job.job_id for job in ranked]

    def _evaluate_fixed_group(self, jobs: list[TrainingJob], backend_name: str) -> EvaluatedGroup | None:
        if not self._compatible_group(jobs):
            return None
        batch_overrides = {job.job_id: self._resolved_batch_size(job) for job in jobs}
        estimated_vram_mb = sum(self._estimate_peak_vram_mb(job, batch_overrides[job.job_id], backend_name) for job in jobs)
        safe_budget_mb = self._safe_budget_mb()
        if estimated_vram_mb > safe_budget_mb:
            return None
        utilization = estimated_vram_mb / safe_budget_mb if safe_budget_mb > 0 else 0.0
        age_bonus = sum(max(1, job.queue_sequence) for job in jobs) / max(1, len(jobs))
        objective = utilization + (0.001 * len(jobs)) - (0.0001 * age_bonus)
        return EvaluatedGroup(
            jobs=jobs,
            backend_name=backend_name,
            estimated_vram_mb=estimated_vram_mb,
            objective_score=objective,
            batch_overrides=batch_overrides,
            fallback_order=self._fallback_order(jobs, batch_overrides, backend_name),
            reason="fixed-batch packed group selected",
        )

    def _evaluate_optimized_group(self, jobs: list[TrainingJob], backend_name: str) -> EvaluatedGroup | None:
        if not self._compatible_group(jobs):
            return None
        hardware_key = self.store.hardware_key()
        group_signature = build_group_signature([job.packing.signature or job.job_id for job in jobs])
        cached = self.store.best_combination_profile(
            group_signature=group_signature,
            hardware_key=hardware_key,
            backend_name=backend_name,
            scheduler_mode=SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED,
        )
        if cached is not None and cached.batch_vector:
            estimated_vram_mb = float(cached.peak_vram_mb or 0)
            return EvaluatedGroup(
                jobs=jobs,
                backend_name=backend_name,
                estimated_vram_mb=estimated_vram_mb,
                objective_score=float(cached.objective_score or 0.0),
                batch_overrides=dict(cached.batch_vector),
                fallback_order=list(cached.fallback_order),
                reason="cached optimal packed group selected",
            )

        per_job_candidates = [self._candidate_batch_sizes(job) for job in jobs]
        safe_budget_mb = self._safe_budget_mb() * float(self.settings.gpu_scheduler.parallel_optimizer.target_vram_fraction)
        best: EvaluatedGroup | None = None

        if len(jobs) <= 3:
            search_space = product(*per_job_candidates)
        else:
            baseline = tuple(self._resolved_batch_size(job) for job in jobs)
            greedy_vectors = [baseline]
            for index, candidates in enumerate(per_job_candidates):
                best_batch = max(candidates)
                vector = list(baseline)
                vector[index] = best_batch
                greedy_vectors.append(tuple(vector))
            search_space = greedy_vectors

        for batch_vector in search_space:
            overrides = {job.job_id: int(batch_size) for job, batch_size in zip(jobs, batch_vector, strict=True)}
            estimated_vram_mb = sum(self._estimate_peak_vram_mb(job, overrides[job.job_id], backend_name) for job in jobs)
            if estimated_vram_mb > safe_budget_mb:
                continue
            utilization = estimated_vram_mb / safe_budget_mb if safe_budget_mb > 0 else 0.0
            slowdown_penalty = 0.0
            for left_job, right_job in combinations(jobs, 2):
                pair_profile = self.store.get_pair_profile(left_job.packing.signature or "", right_job.packing.signature or "")
                if pair_profile and pair_profile.slowdown_ratio is not None:
                    slowdown_penalty += max(0.0, pair_profile.slowdown_ratio - 1.0)
            objective = utilization - slowdown_penalty
            candidate = EvaluatedGroup(
                jobs=jobs,
                backend_name=backend_name,
                estimated_vram_mb=estimated_vram_mb,
                objective_score=objective,
                batch_overrides=overrides,
                fallback_order=self._fallback_order(jobs, overrides, backend_name),
                reason="optimized packed group selected",
            )
            if best is None or candidate.objective_score > best.objective_score:
                best = candidate
        return best

    def _candidate_groups(self, ordered: list[TrainingJob]) -> list[list[TrainingJob]]:
        max_packed = max(1, int(self.settings.gpu_scheduler.max_packed_jobs_per_gpu))
        if self.settings.gpu_scheduler.allow_three_way_packing:
            max_packed = max(max_packed, 3)
        window = ordered[: max(1, int(self.settings.gpu_scheduler.candidate_window_size))]
        groups: list[list[TrainingJob]] = [[window[0]]] if window else []
        upper = min(max_packed, len(window))
        for size in range(2, upper + 1):
            if size <= 3:
                groups.extend([list(items) for items in combinations(window, size)])
            else:
                groups.append(window[:size])
        return groups

    def choose_plan(self, jobs: Iterable[TrainingJob], *, backend_available: dict[str, bool]) -> PlacementPlan | None:
        ordered = RunnableJobQueue(policy=self.policy, jobs=list(jobs)).ordered()
        if not ordered:
            return None
        primary = ordered[0]

        if not self.settings.gpu_scheduler.enabled:
            return PlacementPlan(mode="exclusive", backend_name="exclusive", job_ids=(primary.job_id,), reason="gpu scheduler disabled")

        scheduler_mode = self.settings.gpu_scheduler.mode
        if scheduler_mode in {SCHEDULER_MODE_SERIAL_BASIC, SCHEDULER_MODE_SERIAL_BATCH_OPTIMIZED}:
            return PlacementPlan(mode="exclusive", backend_name="exclusive", job_ids=(primary.job_id,), reason=f"{scheduler_mode} selected")

        best_group: EvaluatedGroup | None = None
        packed_backend_unavailable = False
        missing_memory_estimate = False
        incompatible_group = False
        for group in self._candidate_groups(ordered):
            if len(group) == 1:
                solo = EvaluatedGroup(
                    jobs=group,
                    backend_name="exclusive",
                    estimated_vram_mb=self._estimate_peak_vram_mb(group[0], self._resolved_batch_size(group[0]), "exclusive"),
                    objective_score=0.0,
                    batch_overrides={group[0].job_id: self._resolved_batch_size(group[0])},
                    fallback_order=[],
                    reason="exclusive fallback selected",
                )
                if best_group is None:
                    best_group = solo
                continue

            configured_backends = [
                backend_name
                for backend_name in self.settings.gpu_scheduler.backend_priority
                if backend_name != "exclusive" and all(self._pack_eligible(job, backend_name=backend_name) for job in group)
            ]
            if configured_backends and not any(backend_available.get(backend_name, False) for backend_name in configured_backends):
                packed_backend_unavailable = True
                continue
            available_backends = self._backend_candidates(group, backend_available=backend_available)
            if not available_backends:
                continue
            viable_backends = [
                backend_name
                for backend_name in available_backends
                if all(self._has_memory_estimate(job, backend_name) for job in group)
            ]
            if not viable_backends:
                missing_memory_estimate = True
                continue
            if not self._compatible_group(group):
                incompatible_group = True
                continue

            for backend_name in viable_backends:
                if scheduler_mode == SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED:
                    candidate = self._evaluate_optimized_group(group, backend_name)
                else:
                    candidate = self._evaluate_fixed_group(group, backend_name)
                if candidate is None:
                    continue
                if best_group is None or candidate.objective_score > best_group.objective_score:
                    best_group = candidate

        if best_group is None:
            reason = "no compatible packed group"
            if packed_backend_unavailable:
                reason = "packed backend unavailable"
            elif missing_memory_estimate:
                reason = "solo profile or VRAM estimate unavailable"
            elif incompatible_group:
                reason = "no compatible packed group"
            return PlacementPlan(mode="exclusive", backend_name="exclusive", job_ids=(primary.job_id,), reason=reason)

        if len(best_group.jobs) == 1:
            reason = best_group.reason
            if reason == "exclusive fallback selected":
                if packed_backend_unavailable:
                    reason = "packed backend unavailable"
                elif missing_memory_estimate:
                    reason = "solo profile or VRAM estimate unavailable"
                elif incompatible_group:
                    reason = "no compatible packed group"
            return PlacementPlan(
                mode="exclusive",
                backend_name="exclusive",
                job_ids=(best_group.jobs[0].job_id,),
                reason=reason,
                batch_overrides=best_group.batch_overrides,
            )

        placement_mode = "packed_pair" if len(best_group.jobs) == 2 else "packed_group"
        return PlacementPlan(
            mode=placement_mode,
            backend_name=best_group.backend_name,
            job_ids=tuple(job.job_id for job in best_group.jobs),
            reason=best_group.reason,
            batch_overrides=best_group.batch_overrides,
            fallback_order=best_group.fallback_order,
        )
