"""Candidate group generation for placement planning."""

from __future__ import annotations

from itertools import combinations

from ..domain import (
    BATCH_PROBE_SEARCH_MODE_POWER_OF_TWO,
    TrainingJob,
    normalize_batch_probe_search_mode,
)
from ..config import (
    SCHEDULER_MODE_PARALLEL_AUTO_PACK,
    SCHEDULER_MODE_PARALLEL_TIME_AWARE,
    SchedulerSettings,
)
from .compatibility import CompatibilityEvaluator
from .resource_estimator import ResourceEstimator


class CandidateGenerator:
    def __init__(
        self,
        settings: SchedulerSettings,
        estimator: ResourceEstimator,
        compatibility: CompatibilityEvaluator,
    ):
        self.settings = settings
        self.estimator = estimator
        self.compatibility = compatibility

    def backend_candidates(
        self,
        jobs: list[TrainingJob],
        *,
        backend_available: dict[str, bool],
        scheduler_mode: str,
    ) -> list[str]:
        if len(jobs) == 1 and scheduler_mode == SCHEDULER_MODE_PARALLEL_AUTO_PACK:
            candidates: list[str] = []
            for backend_name in self.settings.gpu_scheduler.backend_priority:
                if backend_name == "exclusive":
                    continue
                if not backend_available.get(backend_name, False):
                    continue
                if not jobs[0].packing.allows_backend(backend_name):
                    continue
                candidates.append(backend_name)
            candidates.append("exclusive")
            return candidates
        if len(jobs) == 1:
            return ["exclusive"]
        candidates = []
        for backend_name in self.settings.gpu_scheduler.backend_priority:
            if backend_name == "exclusive":
                continue
            if not backend_available.get(backend_name, False):
                continue
            if all(self.compatibility.pack_eligible(job, backend_name=backend_name) for job in jobs):
                candidates.append(backend_name)
        return candidates

    def candidate_batch_sizes(self, job: TrainingJob, *, scheduler_mode: str | None = None) -> list[int]:
        if scheduler_mode == SCHEDULER_MODE_PARALLEL_TIME_AWARE:
            raw_values = self.time_aware_batch_proposals(job)
            explicit_cap = job.config.runner_kwargs.get(
                "probe_max_batch_size",
                self.settings.gpu_scheduler.batch_probe_max_batch_size,
            )
            cap = max(1, int(explicit_cap)) if explicit_cap is not None else None
            clipped = [min(value, cap) if cap is not None else value for value in raw_values]
            return list(dict.fromkeys(max(1, int(value)) for value in clipped))

        requested = self.estimator.resolved_batch_size(job)
        explicit_cap = job.config.runner_kwargs.get(
            "probe_max_batch_size",
            self.settings.gpu_scheduler.batch_probe_max_batch_size,
        )
        cap = max(1, int(explicit_cap)) if explicit_cap is not None else None
        optimizer = self.settings.gpu_scheduler.parallel_optimizer
        search_mode = normalize_batch_probe_search_mode(optimizer.batch_search_mode)
        if search_mode == BATCH_PROBE_SEARCH_MODE_POWER_OF_TWO:
            requested_exponent = max(0, requested.bit_length() - 1)
            min_exponent = max(0, requested_exponent - optimizer.power_of_two_range_down)
            max_exponent = requested_exponent + optimizer.power_of_two_range_up
            values = [2**exponent for exponent in range(min_exponent, max_exponent + 1)]
            if cap is not None:
                values = [value for value in values if value <= cap]
                if not values:
                    fallback = 2 ** max(0, cap.bit_length() - 1)
                    return [max(1, fallback)]
            return values
        min_batch = max(1, requested - optimizer.binary_range_down)
        max_batch = requested + optimizer.binary_range_up
        if cap is not None:
            max_batch = min(max_batch, cap)
        if max_batch < min_batch:
            min_batch = max(1, max_batch)
        return list(range(min_batch, max_batch + 1))

    def time_aware_batch_proposals(self, job: TrainingJob) -> list[int]:
        """Return the five pre-clipping exponent proposals for audit logs."""
        requested = int(job.requested_batch_size or self.estimator.resolved_batch_size(job))
        if self.settings.gpu_scheduler.batch_options.require_power_of_two_original and requested & (requested - 1):
            raise ValueError(f"time-aware scheduling requires a power-of-two requested batch size: {requested}")
        exponent = requested.bit_length() - 1
        return [2 ** max(0, exponent + offset) for offset in self.settings.gpu_scheduler.batch_options.exponent_offsets]

    def fallback_order(
        self,
        jobs: list[TrainingJob],
        batch_overrides: dict[str, int],
        backend_name: str,
    ) -> list[str]:
        ranked = sorted(
            jobs,
            key=lambda job: (
                job.priority,
                -(self.estimator.predicted_remaining_runtime_seconds(job, backend_name=backend_name) or 0.0),
                -self.estimator.estimate_avg_vram_mb(
                    job,
                    batch_overrides.get(job.job_id, self.estimator.resolved_batch_size(job)),
                    backend_name,
                ),
                job.queue_sequence,
            ),
        )
        return [job.job_id for job in ranked]

    def candidate_groups(self, ordered: list[TrainingJob], *, scheduler_mode: str) -> list[list[TrainingJob]]:
        if scheduler_mode == SCHEDULER_MODE_PARALLEL_TIME_AWARE:
            return self.time_aware_groups(ordered)
        if scheduler_mode == SCHEDULER_MODE_PARALLEL_AUTO_PACK:
            upper = len(ordered[: max(1, int(self.settings.gpu_scheduler.candidate_window_size))])
            window = ordered[: max(1, int(self.settings.gpu_scheduler.candidate_window_size))]
            groups: list[list[TrainingJob]] = [[job] for job in window]
            for size in range(2, upper + 1):
                if size <= 3:
                    groups.extend([list(items) for items in combinations(window, size)])
                else:
                    groups.append(window[:size])
            return groups
        max_packed = max(1, int(self.settings.gpu_scheduler.max_packed_jobs_per_gpu))
        if self.settings.gpu_scheduler.allow_three_way_packing:
            max_packed = max(max_packed, 3)
        window = ordered[: max(1, int(self.settings.gpu_scheduler.candidate_window_size))]
        upper = min(max_packed, len(window))
        groups = []
        for size in range(2, upper + 1):
            if size <= 3:
                groups.extend([list(items) for items in combinations(window, size)])
            else:
                groups.append(window[:size])
        return groups

    def time_aware_groups(
        self,
        window: list[TrainingJob],
        *,
        max_new_jobs: int | None = None,
        mandatory_anchor: TrainingJob | None = None,
    ) -> list[list[TrainingJob]]:
        """Bounded deterministic subset search; every partial state is legal."""
        if not window:
            return []
        upper = min(len(window), max_new_jobs if max_new_jobs is not None else len(window))
        if upper <= 0:
            return []
        groups: list[list[TrainingJob]] = []
        exact_limit = int(self.settings.gpu_scheduler.exact_search_max_jobs)
        if upper <= exact_limit:
            for size in range(1, upper + 1):
                groups.extend(list(items) for items in combinations(window, size))
        else:
            seed = (mandatory_anchor,) if mandatory_anchor is not None else tuple()
            states: list[tuple[TrainingJob, ...]] = [seed]
            emitted: dict[tuple[str, ...], tuple[TrainingJob, ...]] = {}
            if seed:
                emitted[tuple(member.job_id for member in seed)] = seed
            for job in (member for member in window if member != mandatory_anchor):
                expanded = states + [state + (job,) for state in states if len(state) < upper]
                unique = {tuple(member.job_id for member in state): state for state in expanded}
                states = [unique[key] for key in sorted(unique, key=lambda ids: (-len(ids), ids))[: self.settings.gpu_scheduler.beam_width]]
                for state in states:
                    if state:
                        emitted[tuple(member.job_id for member in state)] = state
            groups = [list(state) for state in emitted.values()]
        if mandatory_anchor is not None:
            groups = [group for group in groups if mandatory_anchor in group]
        groups.sort(key=lambda group: (len(group), tuple(job.job_id for job in group)))
        return groups
