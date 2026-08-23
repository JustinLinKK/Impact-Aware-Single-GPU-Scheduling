"""Incremental time-aware GPU placement planning.

VRAM is a feasibility constraint in this module. It is never an objective to
maximize; placement value comes exclusively from projected/measured drain time.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from ..domain import SchedulingClass, TrainingJob, parse_timestamp
from ..config import SCHEDULER_DECISION_MODE_BACKEND_AWARED, SchedulerSettings
from .backend_aware_planner import BackendAwarePlacementPlanner
from .candidate_generator import CandidateGenerator
from .compatibility import CompatibilityEvaluator
from .planning_repository import PlanningRepository
from .planner_types import DispatchPlan
from .policies import SchedulingPolicy
from .queue import RunnableJobQueue
from .resource_estimator import ResourceEstimator
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
        self.compatibility = CompatibilityEvaluator(repository)
        self.candidate_generator = CandidateGenerator(settings, self.estimator)
        self.time_objective = TimeAwareObjectiveScorer(
            settings,
            self.estimator,
            self.compatibility,
            self.candidate_generator,
        )
        self.backend_aware = BackendAwarePlacementPlanner(
            settings,
            repository,
            self.estimator,
            self.candidate_generator,
            self.compatibility,
            self.time_objective,
        )

    def predicted_remaining_runtime_seconds(self, job: TrainingJob, *, backend_name: str) -> float | None:
        return self.estimator.predicted_remaining_runtime_seconds(job, backend_name=backend_name)

    def predicted_group_vram_mb(self, jobs: list[TrainingJob], *, backend_name: str) -> float:
        return self.estimator.predicted_group_vram_mb(jobs, backend_name=backend_name)

    def _shape_signature(self, job: TrainingJob) -> str:
        return self.estimator.shape_signature(job)

    def _candidate_batch_sizes(self, job: TrainingJob) -> list[int]:
        return self.candidate_generator.candidate_batch_sizes(job)

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

    def _fastest_time_option(self, job: TrainingJob, backend_name: str):
        try:
            batch_sizes = self.candidate_generator.candidate_batch_sizes(job)
        except ValueError:
            return None
        options = [
            option
            for option in self.estimator.estimate_batch_options(job, backend_name, batch_sizes)
            if option.avg_vram_mb <= self.estimator.safe_budget_mb() + 1e-9
        ]
        return min(
            options,
            key=lambda option: (option.remaining_runtime_seconds, option.batch_size),
            default=None,
        )

    def _predicted_solo_remaining(self, job: TrainingJob) -> float | None:
        option = self._fastest_time_option(job, "exclusive")
        return option.remaining_runtime_seconds if option is not None else None

    def _choose_time_aware_plan(
        self,
        jobs: list[TrainingJob],
        *,
        backend_available: dict[str, bool],
        active_vram_mb: float,
        active_jobs: list[TrainingJob],
        admission_open: bool,
        exclusive_drain_requested: bool,
        packing_admission_stalled: bool,
        trial_pending: bool,
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
        if active_jobs and (packing_admission_stalled or trial_pending):
            return None

        cap = self.settings.gpu_scheduler.parallel_job_cap
        remaining_slots = None if cap is None else cap - len(active_jobs)
        if remaining_slots is not None and remaining_slots <= 0:
            return None
        normal_window = [job for job in window if job.scheduling_class == SchedulingClass.NORMAL]
        backend_aware_window = list(normal_window)
        if mandatory is not None:
            normal_window = [mandatory]

        if not active_jobs:
            if (
                self.settings.gpu_scheduler.scheduler_decision_mode
                == SCHEDULER_DECISION_MODE_BACKEND_AWARED
            ):
                pair_plan = self.backend_aware.choose_empty(
                    backend_aware_window,
                    mandatory=mandatory,
                    backend_available=backend_available,
                    effective_priority=lambda job: self._effective_priority(
                        job, now=now
                    ),
                )
                if pair_plan is not None:
                    return pair_plan
            anchor_candidates: list[tuple[tuple[object, ...], TrainingJob, str, object]] = []
            allow_stack_anchor = cap != 1
            for job in normal_window:
                backends = [
                    backend_name
                    for backend_name in self.settings.gpu_scheduler.backend_priority
                    if allow_stack_anchor
                    and backend_name != "exclusive"
                    and backend_available.get(backend_name, False)
                    and self.compatibility.pack_eligible(job, backend_name=backend_name)
                ]
                for backend_name in backends:
                    option = self._fastest_time_option(job, backend_name)
                    solo_remaining = self._predicted_solo_remaining(job)
                    if option is not None and solo_remaining is not None:
                        anchor_candidates.append(
                            ((solo_remaining, -self._effective_priority(job, now=now), job.queue_sequence, job.job_id), job, backend_name, option)
                        )
            if anchor_candidates:
                _, selected_job, backend_name, option = min(anchor_candidates, key=lambda item: item[0])
                return DispatchPlan(
                    mode="stack_anchor",
                    backend_name=backend_name,
                    job_ids=(selected_job.job_id,),
                    reason="shortest remaining-time stack anchor",
                    batch_overrides={selected_job.job_id: option.batch_size},
                    objective_breakdown={
                        "remaining_runtime_seconds": option.remaining_runtime_seconds,
                        "seconds_per_epoch": option.seconds_per_epoch,
                        "requires_live_trial": False,
                    },
                    mandatory_anchor_job_id=mandatory.job_id if mandatory else None,
                    objective_version=self.settings.gpu_scheduler.objective.objective_version,
                )
            exclusive_candidates = [
                (self._fastest_time_option(job, "exclusive"), job) for job in normal_window
            ]
            exclusive_candidates = [(option, job) for option, job in exclusive_candidates if option is not None]
            if exclusive_candidates:
                option, selected_job = min(
                    exclusive_candidates,
                    key=lambda item: (item[0].remaining_runtime_seconds, -self._effective_priority(item[1], now=now), item[1].job_id),
                )
                return DispatchPlan(
                    mode="exclusive",
                    backend_name="exclusive",
                    job_ids=(selected_job.job_id,),
                    reason="shortest remaining-time exclusive fallback",
                    batch_overrides={selected_job.job_id: option.batch_size},
                    mandatory_anchor_job_id=mandatory.job_id if mandatory else None,
                    objective_version=self.settings.gpu_scheduler.objective.objective_version,
                )
            return DispatchPlan(
                mode="exclusive",
                backend_name="exclusive",
                job_ids=(anchor.job_id,),
                reason="runtime estimate unavailable; exclusive fallback",
                mandatory_anchor_job_id=mandatory.job_id if mandatory else None,
                objective_version=self.settings.gpu_scheduler.objective.objective_version,
            )

        active_backends = {
            str(job.metadata.get("placement_backend"))
            for job in active_jobs
            if job.metadata.get("placement_backend")
        }
        if "exclusive" in active_backends or len(active_backends) > 1:
            return None
        backend_candidates = [
            backend_name
            for backend_name in self.settings.gpu_scheduler.backend_priority
            if backend_name != "exclusive"
            and backend_available.get(backend_name, False)
            and (not active_backends or backend_name in active_backends)
        ]
        if (
            self.settings.gpu_scheduler.scheduler_decision_mode
            == SCHEDULER_DECISION_MODE_BACKEND_AWARED
        ):
            return self.backend_aware.choose_active(
                normal_window,
                active_jobs=active_jobs,
                active_vram_mb=active_vram_mb,
                mandatory=mandatory,
                backend_candidates=backend_candidates,
                effective_priority=lambda job: self._effective_priority(
                    job, now=now
                ),
            )
        ordered_candidates: list[tuple[tuple[object, ...], TrainingJob]] = []
        for job in normal_window:
            solo_remaining = self._predicted_solo_remaining(job)
            has_backend_option = any(
                self._fastest_time_option(job, backend) is not None
                for backend in backend_candidates
            )
            if solo_remaining is not None and has_backend_option:
                ordered_candidates.append(
                    ((solo_remaining, -self._effective_priority(job, now=now), job.queue_sequence, job.job_id), job)
                )
        for _, candidate_job in sorted(ordered_candidates, key=lambda item: item[0]):
            evaluations = [
                self.time_objective.evaluate_incremental(
                    candidate_job,
                    backend_name=backend_name,
                    active_jobs=active_jobs,
                    active_vram_mb=active_vram_mb,
                    mandatory_anchor=mandatory,
                )
                for backend_name in backend_candidates
            ]
            viable = [evaluation for evaluation in evaluations if evaluation is not None]
            if not viable:
                continue
            best_group = min(viable, key=self.time_objective.tie_key)
            return DispatchPlan(
                mode="concurrent_group",
                backend_name=best_group.backend_name,
                job_ids=(candidate_job.job_id,),
                reason=best_group.reason,
                batch_overrides=best_group.batch_overrides,
                fallback_order=best_group.fallback_order,
                objective_breakdown=best_group.score_breakdown,
                trial_metadata={
                    "requires_live_trial": bool(best_group.score_breakdown.get("requires_live_trial")),
                    "preexisting_job_ids": list(best_group.score_breakdown.get("preexisting_job_ids", [])),
                    "profile_key": best_group.score_breakdown.get("colocation_profile_key"),
                    "candidate_solo_epoch_seconds": best_group.score_breakdown.get("candidate_solo_epoch_seconds"),
                    "pretrial_epoch_seconds": dict(best_group.score_breakdown.get("active_pretrial_epoch_seconds", {})),
                },
                mandatory_anchor_job_id=best_group.mandatory_anchor_job_id,
                objective_version=best_group.objective_version,
            )
        return None

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
        active_jobs: Iterable[TrainingJob] = (),
        admission_open: bool = True,
        exclusive_drain_requested: bool = False,
        packing_admission_stalled: bool = False,
        trial_pending: bool = False,
        now: datetime | None = None,
    ) -> DispatchPlan | None:
        materialized_jobs = list(jobs)
        ordered = RunnableJobQueue(policy=self.policy, jobs=materialized_jobs).ordered()
        if not ordered:
            return None
        primary = ordered[0]
        if not self.settings.gpu_scheduler.enabled:
            return DispatchPlan(
                mode="exclusive",
                backend_name="exclusive",
                job_ids=(primary.job_id,),
                reason="gpu scheduler disabled",
            )
        return self._choose_time_aware_plan(
            materialized_jobs,
            backend_available=backend_available,
            active_vram_mb=active_vram_mb,
            active_jobs=list(active_jobs),
            admission_open=admission_open,
            exclusive_drain_requested=exclusive_drain_requested,
            packing_admission_stalled=packing_admission_stalled,
            trial_pending=trial_pending,
            now=now or datetime.now(timezone.utc),
        )
