"""Event-driven adaptive GPU placement planner."""

from __future__ import annotations

from typing import Any, Iterable
import gc

from ..config import PREDICTION_MODE_BRANCH_PROFILE, SchedulerSettings
from ..domain import BatchResolution, TrainingJob
from .adaptive_search import AdaptivePlacementSearch, SearchState
from .compatibility import CompatibilityEvaluator
from .planner_types import DispatchPlan
from .planning_repository import PlanningRepository
from .policies import SchedulingPolicy
from .queue import RunnableJobQueue
from .resource_estimator import ResourceEstimator

_EXCLUSIVE_PROBE_TASK_TYPES = {"mlevolve_model_family_probe", "mlevolve_startpoint_probe", "mlevolve_branch_profile_probe"}


class PlacementPlanner:
    """Plan one adaptive placement from pinned active and queued jobs."""

    def __init__(self, settings: SchedulerSettings, repository: PlanningRepository, policy: SchedulingPolicy):
        self.settings = settings
        self.repository = repository
        self.policy = policy
        self.estimator = ResourceEstimator(settings, repository)
        self.compatibility = CompatibilityEvaluator(settings, repository, self.estimator)
        self.search = AdaptivePlacementSearch(settings, self.estimator, self.compatibility)
        self.last_decision_trace: dict[str, Any] = {}

    def predicted_remaining_runtime_seconds(self, job: TrainingJob, *, backend_name: str) -> float | None:
        return self.estimator.predicted_remaining_runtime_seconds(job, backend_name=backend_name)

    def predicted_group_vram_mb(self, jobs: list[TrainingJob], *, backend_name: str) -> float:
        return self.estimator.predicted_group_vram_mb(jobs, backend_name=backend_name)

    def predicted_group_sm_utilization(self, jobs: list[TrainingJob], *, backend_name: str) -> float:
        return self.estimator.predicted_group_sm_utilization(jobs, backend_name=backend_name)

    def _candidate_batch_sizes(self, job: TrainingJob) -> list[int]:
        return self.search.candidate_batches(job, active=False)

    def profile_ready(self, job: TrainingJob) -> bool:
        if job.task_type in _EXCLUSIVE_PROBE_TASK_TYPES:
            return True
        if self.settings.prediction.mode != PREDICTION_MODE_BRANCH_PROFILE:
            return True
        if job.force_exclusive or not job.batch_probe.enabled:
            return True
        return self.estimator.compatible_batch_profile_curve(job) is not None

    def _ordered_window(
        self,
        queued: Iterable[TrainingJob],
        active: Iterable[TrainingJob],
    ) -> tuple[list[TrainingJob], set[str]]:
        active_jobs = list(active)
        active_ids = {job.job_id for job in active_jobs}
        ordered_queued = [
            job
            for job in RunnableJobQueue(policy=self.policy, jobs=list(queued)).ordered()
            if job.job_id not in active_ids
        ]
        limit = max(len(active_jobs), int(self.settings.gpu_scheduler.candidate_window_size))
        return (active_jobs + ordered_queued[: max(0, limit - len(active_jobs))], active_ids)

    def _backend_candidates(self, backend_available: dict[str, bool]) -> list[str]:
        return [
            name
            for name in self.settings.gpu_scheduler.backend_priority
            if name != "exclusive" and backend_available.get(name, False)
        ]

    def _plan_from_state(
        self,
        state: SearchState,
        *,
        backend_name: str,
        solver_kind: str,
        active_ids: set[str],
    ) -> DispatchPlan:
        jobs = list(state.jobs)
        mode = "exclusive" if len(jobs) == 1 else ("packed_pair" if len(jobs) == 2 else "packed_group")
        actual_backend = "exclusive" if len(jobs) == 1 else backend_name
        batch_overrides = {job_id: choice.batch_size for job_id, choice in state.choices.items()}
        fallback_order = [
            job.job_id
            for job in sorted(
                jobs,
                key=lambda item: (item.priority, -item.queue_sequence),
            )
        ]
        score = self.search.score(state)
        return DispatchPlan(
            mode=mode,
            backend_name=actual_backend,
            job_ids=tuple(job.job_id for job in jobs),
            reason=(
                f"adaptive {self.settings.prediction.mode} plan selected by {solver_kind}; "
                f"admitted_waiting={state.admitted_waiting}"
            ),
            batch_overrides=batch_overrides,
            fallback_order=fallback_order,
            estimated_vram_mb=state.vram_mb,
            estimated_sm_utilization=state.sm_utilization,
            predicted_throughput=state.throughput if self.settings.prediction.mode == PREDICTION_MODE_BRANCH_PROFILE else None,
            solver_kind=solver_kind,
            objective_vector=score,
            active_job_ids=tuple(job_id for job_id in active_ids if job_id in state.choices),
        )

    def choose_plan(
        self,
        jobs: Iterable[TrainingJob],
        *,
        backend_available: dict[str, bool],
        active_jobs: Iterable[TrainingJob] = (),
        active_vram_mb: float = 0.0,
        active_sm_utilization: float = 0.0,
    ) -> DispatchPlan | None:
        del active_vram_mb, active_sm_utilization
        self.estimator.begin_planning_cycle()
        self.compatibility.begin_planning_cycle()
        queued = list(jobs)
        window, active_ids = self._ordered_window(queued, active_jobs)
        missing_profiles = [job for job in window if job.job_id not in active_ids and not self.profile_ready(job)]
        candidates = [job for job in window if job.job_id in active_ids or job not in missing_profiles]
        trace: dict[str, Any] = {
            "scheduler_mode": self.settings.gpu_scheduler.mode,
            "prediction_mode": self.settings.prediction.mode,
            "ordered_job_ids": [job.job_id for job in window],
            "active_job_ids": sorted(active_ids),
            "missing_profile_job_ids": [job.job_id for job in missing_profiles],
            "safe_vram_budget_mb": self.estimator.safe_budget_mb(),
            "candidate_window_size": self.settings.gpu_scheduler.candidate_window_size,
            "selected_plan": None,
        }

        def finish(plan: DispatchPlan | None) -> DispatchPlan | None:
            trace["selected_plan"] = None if plan is None else {
                "mode": plan.mode,
                "backend_name": plan.backend_name,
                "job_ids": list(plan.job_ids),
                "batch_overrides": dict(plan.batch_overrides),
                "estimated_vram_mb": plan.estimated_vram_mb,
                "estimated_sm_utilization": plan.estimated_sm_utilization,
                "predicted_throughput": plan.predicted_throughput,
                "solver_kind": plan.solver_kind,
                "objective_vector": list(plan.objective_vector),
                "reason": plan.reason,
            }
            self.last_decision_trace = trace
            return plan

        if not candidates:
            trace["decision_reason"] = "waiting for branch profiles" if missing_profiles else "no runnable jobs"
            return finish(None)

        primary = candidates[0]
        if not self.settings.gpu_scheduler.enabled:
            return finish(
                DispatchPlan(
                    mode="exclusive",
                    backend_name="exclusive",
                    job_ids=(primary.job_id,),
                    reason="gpu scheduler disabled",
                    batch_overrides={primary.job_id: BatchResolution.resolved_batch_size(primary)},
                )
            )
        if primary.task_type in _EXCLUSIVE_PROBE_TASK_TYPES or primary.force_exclusive:
            return finish(
                DispatchPlan(
                    mode="exclusive",
                    backend_name="exclusive",
                    job_ids=(primary.job_id,),
                    reason="job requires exclusive adaptive placement",
                    batch_overrides={primary.job_id: BatchResolution.resolved_batch_size(primary)},
                )
            )

        best: tuple[tuple[float, ...], SearchState, str, str] | None = None
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
        try:
            for backend_name in self._backend_candidates(backend_available):
                state, solver_kind = self.search.solve(candidates, active_job_ids=active_ids, backend_name=backend_name)
                if state is None:
                    continue
                score = self.search.score(state)
                if best is None or score > best[0]:
                    best = (score, state, backend_name, solver_kind)
        finally:
            if gc_was_enabled:
                gc.enable()

        if best is not None:
            _, state, backend_name, solver_kind = best
            return finish(self._plan_from_state(state, backend_name=backend_name, solver_kind=solver_kind, active_ids=active_ids))

        if active_ids:
            trace["decision_reason"] = "no feasible replacement containing all pinned active jobs"
            return finish(None)
        return finish(
            DispatchPlan(
                mode="exclusive",
                backend_name="exclusive",
                job_ids=(primary.job_id,),
                reason="no safe packed plan; exclusive fallback",
                batch_overrides={primary.job_id: BatchResolution.resolved_batch_size(primary)},
            )
        )
