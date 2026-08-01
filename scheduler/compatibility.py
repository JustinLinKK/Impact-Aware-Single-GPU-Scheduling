"""Compatibility checks for multi-job placement."""

from __future__ import annotations

from itertools import combinations

from ..domain import TrainingJob
from .planning_repository import PlanningRepository

class CompatibilityEvaluator:
    def __init__(
        self,
        repository: PlanningRepository,
    ):
        self.repository = repository

    def pack_eligible(self, job: TrainingJob, *, backend_name: str | None = None) -> bool:
        if not (job.packing.eligible and job.packing.signature):
            return False
        if backend_name is None:
            return True
        return job.packing.allows_backend(backend_name)

    def compatible_group(self, jobs: list[TrainingJob], *, backend_name: str) -> bool:
        if len(jobs) <= 1:
            return True
        for left_job, right_job in combinations(jobs, 2):
            if not self.pack_eligible(left_job, backend_name=backend_name):
                return False
            if not self.pack_eligible(right_job, backend_name=backend_name):
                return False
            pair_profile = self.repository.get_pair_profile(
                left_job.packing.signature or "",
                right_job.packing.signature or "",
                backend_name=backend_name,
            )
            if pair_profile is not None and (pair_profile.on_cooldown() or not pair_profile.compatible):
                return False
        return True
