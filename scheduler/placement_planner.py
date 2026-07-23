"""Pure-ish planning engine for GPU placement decisions."""

from __future__ import annotations

from typing import Iterable

from ..domain import TrainingJob
from ..config import (
    SCHEDULER_MODE_PARALLEL_AUTO_PACK,
    SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED,
    SCHEDULER_MODE_SERIAL_BASIC,
    SCHEDULER_MODE_SERIAL_BATCH_OPTIMIZED,
    SchedulerSettings,
)
from .candidate_generator import CandidateGenerator
from .compatibility import CompatibilityEvaluator
from .objective import ObjectiveScorer
from .planning_repository import PlanningRepository
from .planner_types import DispatchPlan, EvaluatedGroup
from .policies import SchedulingPolicy
from .queue import RunnableJobQueue
from .resource_estimator import ResourceEstimator
from .runtime_guardrail import RuntimeGuardrail


class PlacementPlanner:
    """Select the best runnable dispatch plan for the current queue state."""

    def __init__(self, settings: SchedulerSettings, repository: PlanningRepository, policy: SchedulingPolicy):
        self.settings = settings
        self.repository = repository
        self.policy = policy
        self.estimator = ResourceEstimator(settings, repository)
        self.compatibility = CompatibilityEvaluator(settings, repository, self.estimator)
        self.runtime_guardrail = RuntimeGuardrail(settings, repository)
        self.candidate_generator = CandidateGenerator(settings, self.estimator, self.compatibility)
        self.objective = ObjectiveScorer(
            settings,
            repository,
            self.estimator,
            self.compatibility,
            self.candidate_generator,
            self.runtime_guardrail,
        )

    def predicted_remaining_runtime_seconds(self, job: TrainingJob, *, backend_name: str) -> float | None:
        return self.estimator.predicted_remaining_runtime_seconds(job, backend_name=backend_name)

    def predicted_group_vram_mb(self, jobs: list[TrainingJob], *, backend_name: str) -> float:
        return self.estimator.predicted_group_vram_mb(jobs, backend_name=backend_name)

    def predicted_group_sm_utilization(self, jobs: list[TrainingJob], *, backend_name: str) -> float:
        return self.estimator.predicted_group_sm_utilization(jobs, backend_name=backend_name)

    def _shape_signature(self, job: TrainingJob) -> str:
        return self.estimator.shape_signature(job)

    def _candidate_batch_sizes(self, job: TrainingJob) -> list[int]:
        return self.candidate_generator.candidate_batch_sizes(job)

    def choose_plan(
        self,
        jobs: Iterable[TrainingJob],
        *,
        backend_available: dict[str, bool],
        active_vram_mb: float = 0.0,
        active_sm_utilization: float = 0.0,
    ) -> DispatchPlan | None:
        ordered = RunnableJobQueue(policy=self.policy, jobs=list(jobs)).ordered()
        if not ordered:
            return None
        primary = ordered[0]
        if len(ordered) == 1:
            return DispatchPlan(mode="exclusive", backend_name="exclusive", job_ids=(primary.job_id,), reason="single runnable job")

        if not self.settings.gpu_scheduler.enabled:
            return DispatchPlan(mode="exclusive", backend_name="exclusive", job_ids=(primary.job_id,), reason="gpu scheduler disabled")

        scheduler_mode = self.settings.gpu_scheduler.mode
        if scheduler_mode in {SCHEDULER_MODE_SERIAL_BASIC, SCHEDULER_MODE_SERIAL_BATCH_OPTIMIZED}:
            return DispatchPlan(mode="exclusive", backend_name="exclusive", job_ids=(primary.job_id,), reason=f"{scheduler_mode} selected")

        best_group: EvaluatedGroup | None = None
        packed_backend_unavailable = False
        missing_memory_estimate = False
        incompatible_group = False
        for group in self.candidate_generator.candidate_groups(ordered, scheduler_mode=scheduler_mode):
            configured_backends = [
                backend_name
                for backend_name in self.settings.gpu_scheduler.backend_priority
                if backend_name != "exclusive" and all(self.compatibility.pack_eligible(job, backend_name=backend_name) for job in group)
            ]
            if configured_backends and not any(backend_available.get(backend_name, False) for backend_name in configured_backends):
                packed_backend_unavailable = True
                continue
            available_backends = self.candidate_generator.backend_candidates(group, backend_available=backend_available, scheduler_mode=scheduler_mode)
            if not available_backends:
                continue
            viable_backends = [
                backend_name
                for backend_name in available_backends
                if backend_name == "exclusive" or all(self.estimator.has_memory_estimate(job, backend_name) for job in group)
            ]
            if not viable_backends:
                missing_memory_estimate = True
                continue

            for backend_name in viable_backends:
                if scheduler_mode == SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED:
                    candidate = self.objective.evaluate_optimized_group(group, backend_name)
                elif scheduler_mode == SCHEDULER_MODE_PARALLEL_AUTO_PACK:
                    candidate = self.objective.evaluate_auto_pack_group(
                        group,
                        backend_name,
                        active_vram_mb=active_vram_mb,
                        active_sm_utilization=active_sm_utilization,
                    )
                else:
                    candidate = self.objective.evaluate_fixed_group(group, backend_name)
                if candidate is None:
                    incompatible_group = True
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
            return DispatchPlan(mode="exclusive", backend_name="exclusive", job_ids=(primary.job_id,), reason=reason)

        if len(best_group.jobs) == 1:
            return DispatchPlan(
                mode="exclusive",
                backend_name=best_group.backend_name,
                job_ids=(best_group.jobs[0].job_id,),
                reason=best_group.reason,
                batch_overrides=best_group.batch_overrides,
                fallback_order=best_group.fallback_order,
            )

        placement_mode = "packed_pair" if len(best_group.jobs) == 2 else "packed_group"
        return DispatchPlan(
            mode=placement_mode,
            backend_name=best_group.backend_name,
            job_ids=tuple(job.job_id for job in best_group.jobs),
            reason=best_group.reason,
            batch_overrides=best_group.batch_overrides,
            fallback_order=best_group.fallback_order,
        )
