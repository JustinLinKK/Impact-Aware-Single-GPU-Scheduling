"""Pure-ish planning engine for GPU placement decisions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from ..domain import SchedulingClass, TrainingJob, parse_timestamp
from ..config import (
    SCHEDULER_MODE_PARALLEL_AUTO_PACK,
    SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED,
    SCHEDULER_MODE_SERIAL_BASIC,
    SCHEDULER_MODE_SERIAL_BATCH_OPTIMIZED,
    SCHEDULER_MODE_PARALLEL_TIME_AWARE,
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
from .time_objective import TimeAwareObjectiveScorer


class PlacementPlanner:
    """Select the best runnable dispatch plan for the current queue state."""

    def __init__(
        self,
        settings: SchedulerSettings,
        repository: PlanningRepository,
        policy: SchedulingPolicy,
    ):
        self.settings = settings
        self.repository = repository
        self.policy = policy
        self.estimator = ResourceEstimator(settings, repository)
        self.compatibility = CompatibilityEvaluator(settings, repository, self.estimator)
        self.runtime_guardrail = RuntimeGuardrail(settings, repository)
        self.candidate_generator = CandidateGenerator(settings, self.estimator, self.compatibility)
        self.objective = ObjectiveScorer(
            settings,
            self.estimator,
            self.compatibility,
            self.candidate_generator,
            self.runtime_guardrail,
        )
        self.time_objective = TimeAwareObjectiveScorer(
            settings,
            self.estimator,
            self.compatibility,
            self.candidate_generator,
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
        return self.candidate_generator.candidate_batch_sizes(job, scheduler_mode=self.settings.gpu_scheduler.mode)

    def _effective_priority(self, job: TrainingJob, *, now: datetime) -> int:
        if not self.settings.enable_priority_aging or self.settings.aging_interval_seconds <= 0:
            return int(job.priority)
        submitted = parse_timestamp(job.submitted_at)
        waited_seconds = max(0.0, (now - submitted).total_seconds()) if submitted is not None else 0.0
        bonus = int(waited_seconds // self.settings.aging_interval_seconds) * self.settings.aging_priority_increment
        return int(job.priority) + bonus

    def _time_aware_window(self, jobs: list[TrainingJob], *, now: datetime) -> tuple[list[TrainingJob], TrainingJob | None]:
        probe_feature_enabled = self.settings.gpu_scheduler.exclusive_probe.enabled
        normal_jobs = [
            job
            for job in jobs
            if job.scheduling_class == SchedulingClass.NORMAL
            or (job.scheduling_class == SchedulingClass.EXCLUSIVE_PROBE and not probe_feature_enabled)
        ]
        priority = sorted(normal_jobs, key=lambda job: (-self._effective_priority(job, now=now), job.queue_sequence, job.job_id))[
            : self.settings.gpu_scheduler.priority_window_size
        ]
        oldest = sorted(
            normal_jobs,
            key=lambda job: (
                parse_timestamp(job.submitted_at) or now,
                job.queue_sequence,
                job.job_id,
            ),
        )[: self.settings.gpu_scheduler.oldest_window_size]
        starving = [
            job
            for job in normal_jobs
            if (now - (parse_timestamp(job.submitted_at) or now)).total_seconds() >= self.settings.gpu_scheduler.starvation_timeout_seconds
        ]
        probes = [job for job in jobs if probe_feature_enabled and job.scheduling_class == SchedulingClass.EXCLUSIVE_PROBE]
        by_id = {job.job_id: job for job in [*priority, *oldest, *probes, *starving]}
        window = sorted(by_id.values(), key=lambda job: (-self._effective_priority(job, now=now), job.queue_sequence, job.job_id))
        mandatory = min(
            starving,
            key=lambda job: (
                parse_timestamp(job.submitted_at) or now,
                job.queue_sequence,
                job.job_id,
            ),
            default=None,
        )
        return window, mandatory

    def _exclusive_flow_cost(
        self,
        anchor: TrainingJob,
        window: list[TrainingJob],
        weights: dict[str, float],
    ) -> float | None:
        try:
            batch_sizes = self.candidate_generator.candidate_batch_sizes(
                anchor,
                scheduler_mode=SCHEDULER_MODE_PARALLEL_TIME_AWARE,
            )
        except ValueError:
            return None
        options = self.estimator.estimate_batch_options(anchor, "exclusive", batch_sizes)
        feasible = [option for option in options if option.avg_vram_mb <= self.estimator.safe_budget_mb() + 1e-9]
        if not feasible:
            fallback = self.estimator.predicted_remaining_runtime_seconds(anchor, backend_name="exclusive")
            if fallback is None or fallback <= 0:
                return None
            duration = fallback
        else:
            duration = min(option.remaining_runtime_seconds for option in feasible)
        return duration * sum(weights[job.job_id] for job in window)

    def _choose_time_aware_plan(
        self,
        jobs: list[TrainingJob],
        *,
        backend_available: dict[str, bool],
        active_vram_mb: float,
        active_jobs: list[TrainingJob],
        admission_open: bool,
        exclusive_drain_requested: bool,
        now: datetime,
    ) -> DispatchPlan | None:
        window, mandatory = self._time_aware_window(jobs, now=now)
        if not window:
            return None
        probe_feature_enabled = self.settings.gpu_scheduler.exclusive_probe.enabled
        probes = [job for job in window if probe_feature_enabled and job.scheduling_class == SchedulingClass.EXCLUSIVE_PROBE]
        if probes:
            probe = sorted(probes, key=lambda job: (-self._effective_priority(job, now=now), job.queue_sequence, job.job_id))[0]
            if active_jobs:
                return None
            return DispatchPlan(
                mode="exclusive",
                backend_name="exclusive",
                job_ids=(probe.job_id,),
                reason="reserved exclusive probe",
                objective_version=self.settings.gpu_scheduler.objective.objective_version,
            )
        anchor = mandatory or window[0]
        if active_jobs and (not admission_open or exclusive_drain_requested):
            return None

        cap = self.settings.gpu_scheduler.parallel_job_cap
        remaining_slots = None if cap is None else cap - len(active_jobs)
        if remaining_slots is not None and remaining_slots <= 0:
            return None
        weights = self._time_aware_weights(window, now=now)
        exclusive_flow_cost = self._exclusive_flow_cost(anchor, window, weights)
        if exclusive_flow_cost is None:
            if active_jobs:
                return None
            return DispatchPlan(
                mode="exclusive",
                backend_name="exclusive",
                job_ids=(anchor.job_id,),
                reason="batch-indexed runtime estimate unavailable; exclusive fallback",
                mandatory_anchor_job_id=mandatory.job_id if mandatory else None,
                objective_version=self.settings.gpu_scheduler.objective.objective_version,
            )

        groups = self.candidate_generator.time_aware_groups(
            window,
            max_new_jobs=remaining_slots,
            mandatory_anchor=mandatory,
        )
        best_group: EvaluatedGroup | None = None
        for group in groups:
            if probe_feature_enabled and any(job.scheduling_class != SchedulingClass.NORMAL for job in group):
                continue
            if len(group) == 1 and not active_jobs:
                backends = ["exclusive"]
            else:
                backends = [
                    backend_name
                    for backend_name in self.settings.gpu_scheduler.backend_priority
                    if backend_name != "exclusive"
                    and backend_available.get(backend_name, False)
                    and all(job.packing.allows_backend(backend_name) for job in group)
                ]
            for backend_name in backends:
                candidate = self.time_objective.evaluate(
                    group,
                    backend_name=backend_name,
                    planning_window=window,
                    weights=weights,
                    exclusive_flow_cost=exclusive_flow_cost,
                    active_vram_mb=active_vram_mb,
                    active_jobs=active_jobs,
                    mandatory_anchor=mandatory,
                )
                if candidate is not None and (best_group is None or self.time_objective.tie_key(candidate) < self.time_objective.tie_key(best_group)):
                    best_group = candidate

        if best_group is None:
            if active_jobs:
                return None
            return DispatchPlan(
                mode="exclusive",
                backend_name="exclusive",
                job_ids=(anchor.job_id,),
                reason="no feasible time-aware pack; exclusive anchor fallback",
                mandatory_anchor_job_id=mandatory.job_id if mandatory else None,
                objective_version=self.settings.gpu_scheduler.objective.objective_version,
            )
        placement_mode = (
            "concurrent_group"
            if active_jobs and len(best_group.jobs) == 1
            else ("exclusive" if len(best_group.jobs) == 1 else ("packed_pair" if len(best_group.jobs) == 2 else "packed_group"))
        )
        return DispatchPlan(
            mode=placement_mode,
            backend_name=best_group.backend_name,
            job_ids=tuple(job.job_id for job in best_group.jobs),
            reason=best_group.reason,
            batch_overrides=best_group.batch_overrides,
            fallback_order=best_group.fallback_order,
            objective_breakdown=best_group.score_breakdown,
            mandatory_anchor_job_id=best_group.mandatory_anchor_job_id,
            objective_version=best_group.objective_version,
        )

    def _time_aware_weights(self, window: list[TrainingJob], *, now: datetime) -> dict[str, float]:
        effective = {job.job_id: self._effective_priority(job, now=now) for job in window}
        minimum = min(effective.values(), default=0)
        eta = float(self.settings.gpu_scheduler.objective.priority_weight)
        return {job_id: 1.0 + eta * (priority - minimum) for job_id, priority in effective.items()}

    def choose_plan(
        self,
        jobs: Iterable[TrainingJob],
        *,
        backend_available: dict[str, bool],
        active_vram_mb: float = 0.0,
        active_sm_utilization: float = 0.0,
        active_jobs: Iterable[TrainingJob] = (),
        admission_open: bool = True,
        exclusive_drain_requested: bool = False,
        now: datetime | None = None,
    ) -> DispatchPlan | None:
        materialized_jobs = list(jobs)
        ordered = RunnableJobQueue(policy=self.policy, jobs=materialized_jobs).ordered()
        if not ordered:
            return None
        if self.settings.gpu_scheduler.mode == SCHEDULER_MODE_PARALLEL_TIME_AWARE:
            return self._choose_time_aware_plan(
                materialized_jobs,
                backend_available=backend_available,
                active_vram_mb=active_vram_mb,
                active_jobs=list(active_jobs),
                admission_open=admission_open,
                exclusive_drain_requested=exclusive_drain_requested,
                now=now or datetime.now(timezone.utc),
            )
        primary = ordered[0]
        if len(ordered) == 1:
            return DispatchPlan(
                mode="exclusive",
                backend_name="exclusive",
                job_ids=(primary.job_id,),
                reason="single runnable job",
            )

        if not self.settings.gpu_scheduler.enabled:
            return DispatchPlan(
                mode="exclusive",
                backend_name="exclusive",
                job_ids=(primary.job_id,),
                reason="gpu scheduler disabled",
            )

        scheduler_mode = self.settings.gpu_scheduler.mode
        if scheduler_mode in {
            SCHEDULER_MODE_SERIAL_BASIC,
            SCHEDULER_MODE_SERIAL_BATCH_OPTIMIZED,
        }:
            return DispatchPlan(
                mode="exclusive",
                backend_name="exclusive",
                job_ids=(primary.job_id,),
                reason=f"{scheduler_mode} selected",
            )

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
            available_backends = self.candidate_generator.backend_candidates(
                group,
                backend_available=backend_available,
                scheduler_mode=scheduler_mode,
            )
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
            return DispatchPlan(
                mode="exclusive",
                backend_name="exclusive",
                job_ids=(primary.job_id,),
                reason=reason,
            )

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
