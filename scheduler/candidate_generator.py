"""Candidate group generation for placement planning."""

from __future__ import annotations

from itertools import combinations

from ..domain import TrainingJob, normalize_batch_probe_search_mode
from ..config import SCHEDULER_MODE_PARALLEL_AUTO_PACK, SchedulerSettings
from .compatibility import CompatibilityEvaluator
from .group_sizing import candidate_group_sizing
from .resource_estimator import ResourceEstimator


class CandidateGenerator:
    def __init__(self, settings: SchedulerSettings, estimator: ResourceEstimator, compatibility: CompatibilityEvaluator):
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

    def candidate_batch_sizes(self, job: TrainingJob) -> list[int]:
        requested = self.estimator.resolved_batch_size(job)
        explicit_cap = job.config.runner_kwargs.get("probe_max_batch_size", self.settings.gpu_scheduler.batch_probe_max_batch_size)
        cap = max(1, int(explicit_cap)) if explicit_cap is not None else None
        optimizer = self.settings.gpu_scheduler.parallel_optimizer
        search_mode = normalize_batch_probe_search_mode(optimizer.batch_search_mode)
        del search_mode
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

    def fallback_order(self, jobs: list[TrainingJob], batch_overrides: dict[str, int], backend_name: str) -> list[str]:
        ranked = sorted(
            jobs,
            key=lambda job: (
                job.priority,
                -(self.estimator.predicted_remaining_runtime_seconds(job, backend_name=backend_name) or 0.0),
                -self.estimator.estimate_peak_vram_mb(job, batch_overrides.get(job.job_id, self.estimator.resolved_batch_size(job)), backend_name),
                job.queue_sequence,
            ),
        )
        return [job.job_id for job in ranked]

    def candidate_groups(self, ordered: list[TrainingJob], *, scheduler_mode: str) -> list[list[TrainingJob]]:
        sizing = candidate_group_sizing(self.settings, scheduler_mode=scheduler_mode, queued_job_count=len(ordered))
        window = ordered[: sizing.window_size]
        groups: list[list[TrainingJob]] = [[job] for job in window] if sizing.include_singletons else []
        upper = min(sizing.max_group_size, len(window))
        for size in range(2, upper + 1):
            if size <= 3:
                groups.extend([list(items) for items in combinations(window, size)])
            else:
                groups.append(window[:size])
        return groups
