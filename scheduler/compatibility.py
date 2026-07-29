"""Compatibility checks for multi-job placement."""

from __future__ import annotations

from itertools import combinations

from ..domain import PairProfile, SoloProfile, TrainingJob
from ..config import SchedulerSettings
from .planning_repository import PlanningRepository
from .resource_estimator import ResourceEstimator


def compatibility_score(
    primary_job: TrainingJob,
    partner_job: TrainingJob,
    primary_profile: SoloProfile,
    partner_profile: SoloProfile,
    pair_profile: PairProfile | None,
    settings: SchedulerSettings,
) -> float:
    thresholds = settings.gpu_scheduler.thresholds
    primary_util = float(primary_profile.avg_gpu_utilization or 0.0)
    partner_util = float(partner_profile.avg_gpu_utilization or 0.0)
    if primary_util >= thresholds.pack_reject_sm_active_ge:
        return float("-inf")
    if partner_util >= thresholds.pack_reject_sm_active_ge:
        return float("-inf")
    if pair_profile is not None:
        if pair_profile.on_cooldown() or not pair_profile.compatible:
            return float("-inf")
    util_headroom = max(0.0, 1.0 - max(primary_util, partner_util))
    priority_bonus = 0.01 * max(0, partner_job.priority)
    memory_budget_mb = settings.gpu_scheduler.memory.safe_vram_budget_gib * 1024.0
    memory_penalty = (
        (_profile_avg_vram_mb(primary_job, primary_profile) + _profile_avg_vram_mb(partner_job, partner_profile)) / memory_budget_mb
        if memory_budget_mb > 0
        else 0.0
    )
    return 1.0 + util_headroom + priority_bonus - memory_penalty


def _profile_avg_vram_mb(job: TrainingJob, profile: SoloProfile | None) -> int:
    if profile and profile.avg_vram_mb is not None:
        return int(profile.avg_vram_mb)
    if job.resource_requirements.estimated_avg_vram_mb is not None:
        return int(job.resource_requirements.estimated_avg_vram_mb)
    return 0


class CompatibilityEvaluator:
    def __init__(
        self,
        settings: SchedulerSettings,
        repository: PlanningRepository,
        estimator: ResourceEstimator,
    ):
        self.settings = settings
        self.repository = repository
        self.estimator = estimator

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
            left_profile = self.estimator.solo_profile(left_job)
            right_profile = self.estimator.solo_profile(right_job)
            pair_profile = self.repository.get_pair_profile(
                left_job.packing.signature or "",
                right_job.packing.signature or "",
                backend_name=backend_name,
            )
            if left_profile is None or right_profile is None:
                if pair_profile is not None and (pair_profile.on_cooldown() or not pair_profile.compatible):
                    return False
                continue
            score = compatibility_score(
                left_job,
                right_job,
                left_profile,
                right_profile,
                pair_profile,
                self.settings,
            )
            if score == float("-inf"):
                return False
        return True

    def compatible_time_aware_group(self, jobs: list[TrainingJob], *, backend_name: str) -> bool:
        """Compatibility for time-aware placement, intentionally independent of SM utilization."""
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
            if pair_profile is not None:
                if pair_profile.on_cooldown() or not pair_profile.compatible:
                    return False
        return True
