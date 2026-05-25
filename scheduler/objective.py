"""Objective scoring for candidate groups."""

from __future__ import annotations

from itertools import combinations, product

from ..domain import CombinationProfile, TrainingJob, build_group_signature
from ..config import SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED, SchedulerSettings
from .candidate_generator import CandidateGenerator
from .compatibility import CompatibilityEvaluator
from .planning_repository import PlanningRepository
from .planner_types import EvaluatedGroup
from .resource_estimator import ResourceEstimator
from .runtime_guardrail import RuntimeGuardrail


class ObjectiveScorer:
    def __init__(
        self,
        settings: SchedulerSettings,
        repository: PlanningRepository,
        estimator: ResourceEstimator,
        compatibility: CompatibilityEvaluator,
        candidate_generator: CandidateGenerator,
        runtime_guardrail: RuntimeGuardrail,
    ):
        self.settings = settings
        self.repository = repository
        self.estimator = estimator
        self.compatibility = compatibility
        self.candidate_generator = candidate_generator
        self.runtime_guardrail = runtime_guardrail

    def evaluate_fixed_group(self, jobs: list[TrainingJob], backend_name: str) -> EvaluatedGroup | None:
        if not self.compatibility.compatible_group(jobs, backend_name=backend_name):
            return None
        batch_overrides = {job.job_id: self.estimator.resolved_batch_size(job) for job in jobs}
        estimated_vram_mb = sum(self.estimator.estimate_peak_vram_mb(job, batch_overrides[job.job_id], backend_name) for job in jobs)
        safe_budget_mb = self.estimator.safe_budget_mb()
        if estimated_vram_mb > safe_budget_mb:
            return None
        utilization = estimated_vram_mb / safe_budget_mb if safe_budget_mb > 0 else 0.0
        age_bonus = sum(max(1, job.queue_sequence) for job in jobs) / max(1, len(jobs))
        objective = utilization + (0.001 * len(jobs)) - (0.0001 * age_bonus)
        return EvaluatedGroup(
            jobs=jobs,
            backend_name=backend_name,
            estimated_vram_mb=estimated_vram_mb,
            estimated_sm_utilization=self.estimator.predicted_group_sm_utilization(jobs, backend_name=backend_name),
            objective_score=objective,
            batch_overrides=batch_overrides,
            fallback_order=self.candidate_generator.fallback_order(jobs, batch_overrides, backend_name),
            reason="fixed-batch packed group selected",
        )

    def evaluate_optimized_group(self, jobs: list[TrainingJob], backend_name: str) -> EvaluatedGroup | None:
        if not self.compatibility.compatible_group(jobs, backend_name=backend_name):
            return None
        hardware_key = self.repository.hardware_key()
        group_signature = build_group_signature([job.packing.signature or job.job_id for job in jobs])
        cached = self.repository.best_combination_profile(
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
                estimated_sm_utilization=self.estimator.predicted_group_sm_utilization(jobs, backend_name=backend_name),
                objective_score=float(cached.objective_score or 0.0),
                batch_overrides=dict(cached.batch_vector),
                fallback_order=list(cached.fallback_order),
                reason="cached optimal packed group selected",
            )

        per_job_candidates = [self.candidate_generator.candidate_batch_sizes(job) for job in jobs]
        safe_budget_mb = self.estimator.safe_budget_mb()
        best: EvaluatedGroup | None = None

        if len(jobs) <= 3:
            search_space = product(*per_job_candidates)
        else:
            baseline = tuple(self.estimator.resolved_batch_size(job) for job in jobs)
            greedy_vectors = [baseline]
            for index, candidates in enumerate(per_job_candidates):
                best_batch = max(candidates)
                vector = list(baseline)
                vector[index] = best_batch
                greedy_vectors.append(tuple(vector))
            search_space = greedy_vectors

        for batch_vector in search_space:
            overrides = {job.job_id: int(batch_size) for job, batch_size in zip(jobs, batch_vector, strict=True)}
            estimated_vram_mb = sum(self.estimator.estimate_peak_vram_mb(job, overrides[job.job_id], backend_name) for job in jobs)
            if estimated_vram_mb > safe_budget_mb:
                continue
            utilization = estimated_vram_mb / safe_budget_mb if safe_budget_mb > 0 else 0.0
            slowdown_penalty = 0.0
            for left_job, right_job in combinations(jobs, 2):
                pair_profile = self.repository.get_pair_profile(
                    left_job.packing.signature or "",
                    right_job.packing.signature or "",
                    backend_name=backend_name,
                )
                if pair_profile and pair_profile.slowdown_ratio is not None:
                    slowdown_penalty += max(0.0, pair_profile.slowdown_ratio - 1.0)
            objective = utilization - slowdown_penalty
            candidate = EvaluatedGroup(
                jobs=jobs,
                backend_name=backend_name,
                estimated_vram_mb=estimated_vram_mb,
                estimated_sm_utilization=self.estimator.predicted_group_sm_utilization(jobs, backend_name=backend_name),
                objective_score=objective,
                batch_overrides=overrides,
                fallback_order=self.candidate_generator.fallback_order(jobs, overrides, backend_name),
                reason="optimized packed group selected",
            )
            if best is None or candidate.objective_score > best.objective_score:
                best = candidate
        return best

    def evaluate_auto_pack_group(
        self,
        jobs: list[TrainingJob],
        backend_name: str,
        *,
        active_vram_mb: float,
        active_sm_utilization: float,
    ) -> EvaluatedGroup | None:
        if not self.compatibility.compatible_group(jobs, backend_name=backend_name):
            return None
        if len(jobs) > 1 and not all(self.estimator.runtime_ready(job, backend_name) for job in jobs):
            return None
        if len(jobs) == 1 and not jobs[0].runtime_probe.enabled and backend_name != "exclusive":
            return None

        batch_overrides = {job.job_id: self.estimator.resolved_batch_size(job) for job in jobs}
        estimated_vram_mb = sum(self.estimator.estimate_peak_vram_mb(job, batch_overrides[job.job_id], backend_name) for job in jobs)
        if (active_vram_mb + estimated_vram_mb) > self.estimator.safe_budget_mb():
            return None
        estimated_sm_utilization = sum(self.estimator.estimate_sm_utilization(job, batch_overrides[job.job_id], backend_name) for job in jobs)
        runtime_penalty, hard_reject = self.runtime_guardrail.runtime_penalty(jobs, backend_name=backend_name)
        if hard_reject:
            return None

        target_metric = self.settings.gpu_scheduler.auto_pack.target_metric
        target_vram_mb = self.estimator.safe_budget_mb()
        target_sm = float(self.settings.gpu_scheduler.auto_pack.target_sm_fraction)
        if target_metric == "sm":
            projected = active_sm_utilization + estimated_sm_utilization
            if projected > target_sm:
                return None
            gap = abs(target_sm - projected)
        else:
            projected = active_vram_mb + estimated_vram_mb
            if projected > target_vram_mb:
                return None
            gap = abs(target_vram_mb - projected) / max(1.0, target_vram_mb)
        objective = (1.0 - gap) + (0.01 * len(jobs)) - runtime_penalty
        if backend_name != "exclusive":
            objective += 0.02
        return EvaluatedGroup(
            jobs=jobs,
            backend_name=backend_name,
            estimated_vram_mb=estimated_vram_mb,
            estimated_sm_utilization=estimated_sm_utilization,
            objective_score=objective,
            batch_overrides=batch_overrides,
            fallback_order=self.candidate_generator.fallback_order(jobs, batch_overrides, backend_name),
            reason="auto-pack group selected",
        )
