"""Pure-ish planning engine for GPU placement decisions."""

from __future__ import annotations

from typing import Any, Iterable

from ..domain import TrainingJob
from ..config import (
    SCHEDULER_MODE_PARALLEL_AUTO_PACK,
    SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED,
    SCHEDULER_MODE_SERIAL_BASIC,
    SCHEDULER_MODE_SERIAL_BATCH_OPTIMIZED,
    SchedulerSettings,
    effective_scheduler_mode,
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
        self.last_decision_trace: dict[str, Any] = {}

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

    def _runtime_estimates(self, jobs: list[TrainingJob], *, backend_name: str) -> dict[str, float | None]:
        return {
            job.job_id: self.estimator.predicted_remaining_runtime_seconds(job, backend_name=backend_name)
            for job in jobs
        }

    def _expected_runtime_seconds(self, estimates: dict[str, float | None]) -> float | None:
        materialized = [float(value) for value in estimates.values() if value is not None]
        return max(materialized) if materialized else None

    def _candidate_trace(
        self,
        jobs: list[TrainingJob],
        *,
        backend_name: str | None,
        status: str,
        rejection_reason: str | None = None,
        evaluated: EvaluatedGroup | None = None,
    ) -> dict[str, Any]:
        backend = backend_name or "unselected"
        runtime_estimates = self._runtime_estimates(jobs, backend_name=backend)
        payload: dict[str, Any] = {
            "job_ids": [job.job_id for job in jobs],
            "packing_signatures": [job.packing.signature for job in jobs],
            "backend_name": backend_name,
            "status": status,
            "rejection_reason": rejection_reason,
            "expected_runtime_seconds": self._expected_runtime_seconds(runtime_estimates),
            "job_expected_runtime_seconds": runtime_estimates,
        }
        if evaluated is not None:
            payload.update(
                {
                    "objective_score": evaluated.objective_score,
                    "estimated_vram_mb": evaluated.estimated_vram_mb,
                    "estimated_sm_utilization": evaluated.estimated_sm_utilization,
                    "batch_overrides": dict(evaluated.batch_overrides),
                    "fallback_order": list(evaluated.fallback_order),
                    "reason": evaluated.reason,
                }
            )
        return payload

    def _plan_trace(self, plan: DispatchPlan | None) -> dict[str, Any] | None:
        if plan is None:
            return None
        runtime_estimates: dict[str, float | None] = {}
        get_job = getattr(self.repository, "get_job", None)
        for job_id in plan.job_ids:
            job = get_job(job_id) if callable(get_job) else None
            runtime_estimates[job_id] = (
                self.predicted_remaining_runtime_seconds(job, backend_name=plan.backend_name)
                if job is not None
                else None
            )
        return {
            "mode": plan.mode,
            "backend_name": plan.backend_name,
            "job_ids": list(plan.job_ids),
            "reason": plan.reason,
            "batch_overrides": dict(plan.batch_overrides),
            "fallback_order": list(plan.fallback_order),
            "expected_runtime_seconds": self._expected_runtime_seconds(runtime_estimates),
            "job_expected_runtime_seconds": runtime_estimates,
        }

    def choose_plan(
        self,
        jobs: Iterable[TrainingJob],
        *,
        backend_available: dict[str, bool],
        active_vram_mb: float = 0.0,
        active_sm_utilization: float = 0.0,
    ) -> DispatchPlan | None:
        ordered = RunnableJobQueue(policy=self.policy, jobs=list(jobs)).ordered()
        trace: dict[str, Any] = {
            "scheduler_mode": self.settings.gpu_scheduler.mode,
            "effective_scheduler_mode": effective_scheduler_mode(self.settings.gpu_scheduler.mode),
            "backend_available": dict(backend_available),
            "ordered_job_ids": [job.job_id for job in ordered],
            "candidate_window_size": self.settings.gpu_scheduler.candidate_window_size,
            "safe_vram_budget_mb": self.estimator.safe_budget_mb(),
            "auto_pack_target_metric": self.settings.gpu_scheduler.auto_pack.target_metric,
            "auto_pack_target_vram_mb": self.estimator.safe_budget_mb(),
            "auto_pack_target_sm_utilization": self.settings.gpu_scheduler.auto_pack.target_sm_fraction,
            "active_gpu_occupancy": {
                "vram_mb": active_vram_mb,
                "sm_utilization": active_sm_utilization,
            },
            "candidates": [],
            "selected_plan": None,
        }

        def finish(plan: DispatchPlan | None) -> DispatchPlan | None:
            trace["selected_plan"] = self._plan_trace(plan)
            self.last_decision_trace = trace
            return plan

        if not ordered:
            trace["decision_reason"] = "no runnable jobs"
            return finish(None)
        primary = ordered[0]
        scheduler_mode = effective_scheduler_mode(self.settings.gpu_scheduler.mode)
        if len(ordered) == 1:
            reason = "single runnable job"
            if self.settings.gpu_scheduler.enabled and scheduler_mode == SCHEDULER_MODE_PARALLEL_AUTO_PACK:
                backends = [
                    backend_name
                    for backend_name in self.candidate_generator.backend_candidates(
                        [primary],
                        backend_available=backend_available,
                        scheduler_mode=scheduler_mode,
                    )
                    if backend_name != "exclusive"
                ]
                if backends and not any(self.estimator.has_memory_estimate(primary, backend_name) for backend_name in backends):
                    reason = "VRAM estimate unavailable; dispatching exclusive calibration probe"
            plan = DispatchPlan(mode="exclusive", backend_name="exclusive", job_ids=(primary.job_id,), reason=reason)
            trace["decision_reason"] = plan.reason
            return finish(plan)

        if not self.settings.gpu_scheduler.enabled:
            plan = DispatchPlan(mode="exclusive", backend_name="exclusive", job_ids=(primary.job_id,), reason="gpu scheduler disabled")
            trace["decision_reason"] = plan.reason
            return finish(plan)

        if scheduler_mode in {SCHEDULER_MODE_SERIAL_BASIC, SCHEDULER_MODE_SERIAL_BATCH_OPTIMIZED}:
            plan = DispatchPlan(mode="exclusive", backend_name="exclusive", job_ids=(primary.job_id,), reason=f"{scheduler_mode} selected")
            trace["decision_reason"] = plan.reason
            return finish(plan)

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
                trace["candidates"].append(
                    self._candidate_trace(
                        group,
                        backend_name=None,
                        status="rejected",
                        rejection_reason="packed backend unavailable",
                    )
                )
                continue
            available_backends = self.candidate_generator.backend_candidates(group, backend_available=backend_available, scheduler_mode=scheduler_mode)
            if not available_backends:
                trace["candidates"].append(
                    self._candidate_trace(
                        group,
                        backend_name=None,
                        status="rejected",
                        rejection_reason="no backend candidate allowed by packing policy or availability",
                    )
                )
                continue
            viable_backends = [
                backend_name
                for backend_name in available_backends
                if backend_name == "exclusive" or all(self.estimator.has_memory_estimate(job, backend_name) for job in group)
            ]
            if not viable_backends:
                missing_memory_estimate = True
                memory_rejection_reason = (
                    "VRAM estimate unavailable; exclusive calibration probe required"
                    if scheduler_mode == SCHEDULER_MODE_PARALLEL_AUTO_PACK
                    else "solo profile or VRAM estimate unavailable"
                )
                for backend_name in available_backends:
                    trace["candidates"].append(
                        self._candidate_trace(
                            group,
                            backend_name=backend_name,
                            status="rejected",
                            rejection_reason=memory_rejection_reason,
                        )
                    )
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
                    trace["candidates"].append(
                        self._candidate_trace(
                            group,
                            backend_name=backend_name,
                            status="rejected",
                            rejection_reason="incompatible group, over budget, or runtime guardrail rejected it",
                        )
                    )
                    continue
                trace["candidates"].append(self._candidate_trace(group, backend_name=backend_name, status="accepted", evaluated=candidate))
                if best_group is None or candidate.objective_score > best_group.objective_score:
                    best_group = candidate

        if best_group is None:
            reason = "no compatible packed group"
            if packed_backend_unavailable:
                reason = "packed backend unavailable"
            elif missing_memory_estimate:
                reason = (
                    "VRAM estimate unavailable; dispatching exclusive calibration probe"
                    if scheduler_mode == SCHEDULER_MODE_PARALLEL_AUTO_PACK
                    else "solo profile or VRAM estimate unavailable"
                )
            elif incompatible_group:
                reason = "no compatible packed group"
            plan = DispatchPlan(mode="exclusive", backend_name="exclusive", job_ids=(primary.job_id,), reason=reason)
            trace["decision_reason"] = reason
            return finish(plan)

        if len(best_group.jobs) == 1:
            plan_reason = best_group.reason
            if (
                scheduler_mode == SCHEDULER_MODE_PARALLEL_AUTO_PACK
                and missing_memory_estimate
                and best_group.backend_name == "exclusive"
            ):
                plan_reason = "VRAM estimate unavailable; dispatching exclusive calibration probe"
            plan = DispatchPlan(
                mode="exclusive",
                backend_name=best_group.backend_name,
                job_ids=(best_group.jobs[0].job_id,),
                reason=plan_reason,
                batch_overrides=best_group.batch_overrides,
                fallback_order=best_group.fallback_order,
            )
            trace["decision_reason"] = plan_reason
            return finish(plan)

        placement_mode = "packed_pair" if len(best_group.jobs) == 2 else "packed_group"
        plan = DispatchPlan(
            mode=placement_mode,
            backend_name=best_group.backend_name,
            job_ids=tuple(job.job_id for job in best_group.jobs),
            reason=best_group.reason,
            batch_overrides=best_group.batch_overrides,
            fallback_order=best_group.fallback_order,
        )
        trace["decision_reason"] = best_group.reason
        return finish(plan)
