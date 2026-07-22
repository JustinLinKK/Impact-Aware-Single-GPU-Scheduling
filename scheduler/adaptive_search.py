"""Bounded exact and multiple-choice knapsack search for adaptive GPU packing."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, log2
from typing import Iterable

from ..config import PREDICTION_MODE_BRANCH_PROFILE, SchedulerSettings
from ..domain import TrainingJob
from .compatibility import CompatibilityEvaluator
from .resource_estimator import ResourceEstimator


@dataclass(frozen=True, slots=True)
class BatchChoice:
    batch_size: int
    vram_mb: float
    sm_utilization: float
    throughput: float


@dataclass(slots=True)
class SearchState:
    jobs: tuple[TrainingJob, ...] = ()
    choices: dict[str, BatchChoice] = field(default_factory=dict)
    vram_mb: float = 0.0
    sm_utilization: float = 0.0
    throughput: float = 0.0
    admitted_waiting: int = 0
    priority_age_score: float = 0.0
    batch_deviation: float = 0.0
    restart_cost: float = 0.0
    conservative_vram_buckets: int = 0
    objective_key: tuple[float, ...] | None = None
    blocked_job_ids: frozenset[str] = frozenset()


class AdaptivePlacementSearch:
    def __init__(
        self,
        settings: SchedulerSettings,
        estimator: ResourceEstimator,
        compatibility: CompatibilityEvaluator,
    ):
        self.settings = settings
        self.estimator = estimator
        self.compatibility = compatibility
        self._context_backend: str | None = None
        self._eligible_job_ids: set[str] = set()
        self._incompatible_by_job: dict[str, set[str]] = {}
        self._authored_batches: dict[str, int] = {}

    def candidate_batches(self, job: TrainingJob, *, active: bool) -> list[int]:
        authored = self.estimator.authored_batch_size(job)
        current = self.estimator.resolved_batch_size(job)
        cap = int(self.settings.gpu_scheduler.batch_probe_max_batch_size or 4096)
        minimum = max(1, int(job.batch_probe.minimum_batch_size or self.settings.gpu_scheduler.batch_probe_min_batch_size))
        if self.settings.prediction.mode == PREDICTION_MODE_BRANCH_PROFILE:
            curve = self.estimator.compatible_batch_profile_curve(job)
            if curve is None:
                return [current] if active else []
            feasible = sorted(
                int(point.batch_size)
                for point in curve.points
                if minimum <= int(point.batch_size) <= cap
            )
            if not feasible:
                return [current] if active else []
            below_or_equal = [value for value in feasible if value <= authored]
            center = max(below_or_equal) if below_or_equal else min(feasible)
            center_index = feasible.index(center)
            selected = feasible[max(0, center_index - 1) : center_index + 2]
            if len(selected) < 3:
                selected = feasible[max(0, min(center_index, len(feasible) - 3)) : center_index + 3]
                selected = sorted(selected, key=lambda value: (abs(log2(value / authored)), value))[:3]
            if active and current not in selected:
                if len(selected) >= 3:
                    farthest = max(range(len(selected)), key=lambda index: abs(log2(selected[index] / authored)))
                    selected[farthest] = current
                else:
                    selected.append(current)
            return sorted(set(selected))

        del active, current
        candidates = {authored, max(minimum, authored // 2), min(cap, authored * 2)}
        return sorted(value for value in candidates if minimum <= value <= cap and value > 0)

    def choices_for(self, job: TrainingJob, *, backend_name: str, active: bool) -> list[BatchChoice]:
        choices: list[BatchChoice] = []
        for batch_size in self.candidate_batches(job, active=active):
            vram = self.estimator.estimate_peak_vram_mb(job, batch_size, backend_name)
            if vram is None:
                continue
            sm = self.estimator.estimate_sm_utilization(job, batch_size, backend_name) or 0.0
            throughput = self.estimator.predicted_samples_per_second(job, batch_size) or 0.0
            choices.append(BatchChoice(batch_size=batch_size, vram_mb=float(vram), sm_utilization=float(sm), throughput=float(throughput)))
        return choices

    def solve(
        self,
        jobs: Iterable[TrainingJob],
        *,
        active_job_ids: set[str],
        backend_name: str,
    ) -> tuple[SearchState | None, str]:
        ordered = list(jobs)
        options = {
            job.job_id: self.choices_for(job, backend_name=backend_name, active=job.job_id in active_job_ids)
            for job in ordered
        }
        if any(not options[job_id] for job_id in active_job_ids):
            return None, "infeasible"
        ordered.sort(
            key=lambda job: (
                job.job_id not in active_job_ids,
                -int(job.priority),
                -max((choice.vram_mb for choice in options[job.job_id]), default=0.0),
                int(job.queue_sequence),
            )
        )
        self._context_backend = backend_name
        self._eligible_job_ids = {
            job.job_id for job in ordered if self.compatibility.pack_eligible(job, backend_name=backend_name)
        }
        self._authored_batches = {job.job_id: self.estimator.authored_batch_size(job) for job in ordered}
        self._incompatible_by_job = {job.job_id: set() for job in ordered}
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                if self.compatibility.compatible_group([left, right], backend_name=backend_name):
                    continue
                self._incompatible_by_job[left.job_id].add(right.job_id)
                self._incompatible_by_job[right.job_id].add(left.job_id)
        cutoff = self.settings.gpu_scheduler.adaptive.exact_search_max_jobs
        if len(ordered) <= cutoff:
            return self._solve_exact(ordered, options, active_job_ids, backend_name), "exact_branch_and_bound"
        return self._solve_dp(ordered, options, active_job_ids, backend_name), "bounded_multiple_choice_dp"

    def score(self, state: SearchState) -> tuple[float, ...]:
        if state.objective_key is not None:
            return state.objective_key
        return self._refresh_score(state)

    def _refresh_score(self, state: SearchState) -> tuple[float, ...]:
        if self.settings.prediction.mode == PREDICTION_MODE_BRANCH_PROFILE:
            state.objective_key = (
                float(state.admitted_waiting),
                state.priority_age_score,
                state.throughput,
                -state.restart_cost,
                -state.batch_deviation,
                state.vram_mb,
            )
            return state.objective_key
        state.objective_key = (
            float(state.admitted_waiting),
            state.priority_age_score,
            state.vram_mb,
            -state.batch_deviation,
        )
        return state.objective_key

    def _extend(
        self,
        state: SearchState,
        job: TrainingJob,
        choice: BatchChoice,
        *,
        active: bool,
        backend_name: str,
        evaluate_group_throughput: bool = True,
    ) -> SearchState | None:
        if len(state.jobs) >= self.settings.gpu_scheduler.max_packed_jobs_per_gpu:
            return None
        jobs = (*state.jobs, job)
        if self._context_backend == backend_name and job.job_id in self._incompatible_by_job:
            if len(jobs) > 1 and (
                job.job_id not in self._eligible_job_ids
                or (len(state.jobs) == 1 and state.jobs[0].job_id not in self._eligible_job_ids)
            ):
                return None
            incompatible = self._incompatible_by_job[job.job_id]
            if any(member.job_id in incompatible for member in state.jobs):
                return None
        else:
            if len(jobs) > 1 and (
                not self.compatibility.pack_eligible(job, backend_name=backend_name)
                or (len(state.jobs) == 1 and not self.compatibility.pack_eligible(state.jobs[0], backend_name=backend_name))
            ):
                return None
            if any(
                not self.compatibility.compatible_group([member, job], backend_name=backend_name)
                for member in state.jobs
            ):
                return None
        vram = state.vram_mb + choice.vram_mb
        if vram > self.estimator.safe_budget_mb():
            return None
        sm = state.sm_utilization + choice.sm_utilization
        if sm > 1.0:
            return None
        authored = max(1, self._authored_batches.get(job.job_id) or self.estimator.authored_batch_size(job))
        choices = {**state.choices, job.job_id: choice}
        if not self.estimator.combination_is_compatible(
            list(jobs),
            {job_id: selected.batch_size for job_id, selected in choices.items()},
            backend_name=backend_name,
        ):
            return None
        if evaluate_group_throughput:
            throughput = self.estimator.predicted_group_samples_per_second(
                list(jobs),
                {job_id: selected.batch_size for job_id, selected in choices.items()},
                backend_name=backend_name,
            )
        else:
            conservative_slowdown = max(1.0, float(self.settings.gpu_scheduler.thresholds.pack_reject_max_slowdown))
            throughput = state.throughput + (choice.throughput / conservative_slowdown)
        extended = SearchState(
            jobs=jobs,
            choices=choices,
            vram_mb=vram,
            sm_utilization=sm,
            throughput=throughput,
            admitted_waiting=state.admitted_waiting + (0 if active else 1),
            priority_age_score=state.priority_age_score + (0.0 if active else (float(job.priority) * 1_000_000.0) - float(job.queue_sequence)),
            batch_deviation=state.batch_deviation + abs(log2(choice.batch_size / authored)),
            restart_cost=(
                state.restart_cost
                + (
                    self._restart_cost(job)
                    if active and choice.batch_size != self.estimator.resolved_batch_size(job)
                    else 0.0
                )
            ),
            conservative_vram_buckets=(
                state.conservative_vram_buckets
                + int(ceil(choice.vram_mb / self.settings.gpu_scheduler.adaptive.vram_bucket_mb))
            ),
            blocked_job_ids=(
                state.blocked_job_ids
                | frozenset(self._incompatible_by_job.get(job.job_id, set()))
                | (
                    frozenset(self._incompatible_by_job)
                    if job.job_id not in self._eligible_job_ids
                    else frozenset()
                )
            ),
        )
        self._refresh_score(extended)
        return extended

    @staticmethod
    def _restart_cost(job: TrainingJob) -> float:
        for key in ("scheduler_checkpoint_overhead_seconds", "checkpoint_estimated_overhead_seconds"):
            try:
                value = job.metadata.get(key)
                if value is not None:
                    return max(0.0, float(value))
            except (TypeError, ValueError):
                pass
        return 1.0

    def _solve_exact(
        self,
        jobs: list[TrainingJob],
        options: dict[str, list[BatchChoice]],
        active_job_ids: set[str],
        backend_name: str,
    ) -> SearchState | None:
        best: SearchState | None = None

        def visit(index: int, state: SearchState) -> None:
            nonlocal best
            remaining_slots = self.settings.gpu_scheduler.max_packed_jobs_per_gpu - len(state.jobs)
            remaining_waiting = sum(job.job_id not in active_job_ids for job in jobs[index:])
            if best is not None and state.admitted_waiting + min(remaining_slots, remaining_waiting) < best.admitted_waiting:
                return
            if index >= len(jobs):
                if active_job_ids.issubset(state.choices) and state.jobs:
                    if best is None or self.score(state) > self.score(best):
                        best = state
                return
            job = jobs[index]
            active = job.job_id in active_job_ids
            if not active:
                visit(index + 1, state)
            for choice in options[job.job_id]:
                extended = self._extend(state, job, choice, active=active, backend_name=backend_name)
                if extended is not None:
                    visit(index + 1, extended)

        visit(0, SearchState())
        return best

    def _solve_dp(
        self,
        jobs: list[TrainingJob],
        options: dict[str, list[BatchChoice]],
        active_job_ids: set[str],
        backend_name: str,
    ) -> SearchState | None:
        bucket_mb = self.settings.gpu_scheduler.adaptive.vram_bucket_mb
        capacity = int(self.estimator.safe_budget_mb()) // bucket_mb
        width = self.settings.gpu_scheduler.adaptive.frontier_width
        frontier: dict[int, list[SearchState]] = {0: [SearchState()]}
        for job in jobs:
            active = job.job_id in active_job_ids
            next_frontier: dict[int, list[SearchState]] = {}
            for states in frontier.values():
                for state in states:
                    if not active:
                        self._retain(next_frontier, state, bucket_mb=bucket_mb, capacity=capacity, width=width)
                    for choice in options[job.job_id]:
                        extended = self._extend(
                            state,
                            job,
                            choice,
                            active=active,
                            backend_name=backend_name,
                            evaluate_group_throughput=False,
                        )
                        if extended is not None:
                            self._retain(next_frontier, extended, bucket_mb=bucket_mb, capacity=capacity, width=width)
            frontier = self._prune_frontier(next_frontier, width=width)
            if not frontier:
                return None
        finalists = [state for states in frontier.values() for state in states if active_job_ids.issubset(state.choices)]
        finalists.sort(key=self.score, reverse=True)
        finalist_limit = self.settings.gpu_scheduler.adaptive.finalist_limit
        validated: list[SearchState] = []
        for state in finalists[:finalist_limit]:
            if state.vram_mb > self.estimator.safe_budget_mb() or state.sm_utilization > 1.0:
                continue
            if not self.compatibility.compatible_group(list(state.jobs), backend_name=backend_name):
                continue
            state.throughput = self.estimator.predicted_group_samples_per_second(
                list(state.jobs),
                {job_id: choice.batch_size for job_id, choice in state.choices.items()},
                backend_name=backend_name,
            )
            state.objective_key = None
            self._refresh_score(state)
            validated.append(state)
        return max(validated, key=self.score) if validated else None

    def _retain(
        self,
        frontier: dict[int, list[SearchState]],
        state: SearchState,
        *,
        bucket_mb: int,
        capacity: int,
        width: int,
    ) -> None:
        bucket = state.conservative_vram_buckets
        if bucket > capacity:
            return
        frontier.setdefault(bucket, []).append(state)

    def _prune_frontier(
        self,
        frontier: dict[int, list[SearchState]],
        *,
        width: int,
    ) -> dict[int, list[SearchState]]:
        pruned: dict[int, list[SearchState]] = {}

        def dominates(left: SearchState, right: SearchState) -> bool:
            return (
                left.admitted_waiting >= right.admitted_waiting
                and left.priority_age_score >= right.priority_age_score
                and left.throughput >= right.throughput
                and left.batch_deviation <= right.batch_deviation
                and left.restart_cost <= right.restart_cost
                and left.sm_utilization <= right.sm_utilization
                and len(left.jobs) <= len(right.jobs)
                and left.blocked_job_ids.issubset(right.blocked_job_ids)
            )

        for bucket, states in frontier.items():
            states.sort(key=self.score, reverse=True)
            kept: list[SearchState] = []
            # Width is a hard approximation bound: only the best K states are
            # candidates for this bucket's nondominated frontier.
            for state in states[:width]:
                if any(dominates(incumbent, state) for incumbent in kept):
                    continue
                kept[:] = [incumbent for incumbent in kept if not dominates(state, incumbent)]
                kept.append(state)
                if len(kept) >= width:
                    break
            if kept:
                pruned[bucket] = kept
        return pruned
