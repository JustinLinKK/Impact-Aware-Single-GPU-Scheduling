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
        if pair_profile.slowdown_ratio is not None and pair_profile.slowdown_ratio > thresholds.pack_reject_max_slowdown:
            return float("-inf")
    util_headroom = max(0.0, 1.0 - max(primary_util, partner_util))
    priority_bonus = 0.01 * max(0, partner_job.priority)
    memory_budget_mb = settings.gpu_scheduler.memory.budget_mb(None)
    memory_penalty = (
        _profile_peak_vram_mb(primary_job, primary_profile) + _profile_peak_vram_mb(partner_job, partner_profile)
    ) / memory_budget_mb if memory_budget_mb > 0 else 0.0
    return 1.0 + util_headroom + priority_bonus - memory_penalty


def _profile_peak_vram_mb(job: TrainingJob, profile: SoloProfile | None) -> int:
    if profile and profile.peak_vram_mb is not None:
        return int(profile.peak_vram_mb)
    if job.resource_requirements.estimated_vram_mb is not None:
        return int(job.resource_requirements.estimated_vram_mb)
    return 0


class CompatibilityEvaluator:
    def __init__(self, settings: SchedulerSettings, repository: PlanningRepository, estimator: ResourceEstimator):
        self.settings = settings
        self.repository = repository
        self.estimator = estimator
        self._pair_cache: dict[tuple[str, str, str], PairProfile | bool] = {}
        self._pairs_prefetched = False

    def begin_planning_cycle(self) -> None:
        self._pair_cache.clear()
        self._pairs_prefetched = False
        lister = getattr(self.repository, "list_pair_profiles", None)
        if not callable(lister):
            return
        self._pairs_prefetched = True
        for profile in lister(hardware_key=self.repository.hardware_key()):
            signatures = sorted((profile.left_signature, profile.right_signature))
            self._pair_cache[(signatures[0], signatures[1], profile.backend_name)] = profile

    def pack_eligible(self, job: TrainingJob, *, backend_name: str | None = None) -> bool:
        if not (job.packing.eligible and job.packing.signature):
            return False
        if backend_name is None:
            return True
        return job.packing.allows_backend(backend_name)

    def missing_runtime_profile_jobs(self, jobs: list[TrainingJob], *, backend_name: str) -> list[TrainingJob]:
        del jobs, backend_name
        return []

    def compatible_group(self, jobs: list[TrainingJob], *, backend_name: str) -> bool:
        if len(jobs) <= 1:
            return True
        thresholds = self.settings.gpu_scheduler.thresholds
        for left_job, right_job in combinations(jobs, 2):
            signatures = sorted((left_job.packing.signature or "", right_job.packing.signature or ""))
            key = (signatures[0], signatures[1], backend_name)
            if key not in self._pair_cache and not self._pairs_prefetched:
                self._pair_cache[key] = self.repository.get_pair_profile(
                    signatures[0], signatures[1], backend_name=backend_name
                ) or False
            cached = self._pair_cache.get(key, False)
            pair_profile = cached if isinstance(cached, PairProfile) else None
            if pair_profile is not None and (pair_profile.on_cooldown() or not pair_profile.compatible):
                return False
            if pair_profile and pair_profile.slowdown_ratio is not None and pair_profile.slowdown_ratio > thresholds.pack_reject_max_slowdown:
                return False
        return True
