"""GPU-aware placement planning for the local scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..schemas import PairProfile, SoloProfile, TrainingJob
from ..settings import SchedulerSettings
from ..storage.sqlite_store import SQLiteStateStore
from .policies import SchedulingPolicy
from .queue import RunnableJobQueue


@dataclass(slots=True)
class PlacementPlan:
    mode: str
    backend_name: str
    job_ids: tuple[str, ...]
    reason: str


def _profile_peak_vram_mb(job: TrainingJob, profile: SoloProfile | None) -> int:
    if profile and profile.peak_vram_mb is not None:
        return int(profile.peak_vram_mb)
    if job.resource_requirements.estimated_vram_mb is not None:
        return int(job.resource_requirements.estimated_vram_mb)
    return 0


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
    memory_budget_mb = settings.gpu_scheduler.memory.safe_vram_budget_gib * 1024.0
    memory_penalty = (
        _profile_peak_vram_mb(primary_job, primary_profile) + _profile_peak_vram_mb(partner_job, partner_profile)
    ) / memory_budget_mb if memory_budget_mb > 0 else 0.0
    return 1.0 + util_headroom + priority_bonus - memory_penalty


class GpuPlacementPlanner:
    """Decide between exclusive dispatch and pair packing."""

    def __init__(self, settings: SchedulerSettings, store: SQLiteStateStore, policy: SchedulingPolicy):
        self.settings = settings
        self.store = store
        self.policy = policy

    def _memory_gate(self, left_job: TrainingJob, right_job: TrainingJob, left_profile: SoloProfile, right_profile: SoloProfile) -> bool:
        safe_budget_mb = self.settings.gpu_scheduler.memory.safe_vram_budget_gib * 1024.0
        return (_profile_peak_vram_mb(left_job, left_profile) + _profile_peak_vram_mb(right_job, right_profile)) <= safe_budget_mb

    def _pack_eligible(self, job: TrainingJob) -> bool:
        return bool(job.packing.eligible and job.packing.signature)

    def choose_plan(self, jobs: Iterable[TrainingJob], *, backend_available: dict[str, bool]) -> PlacementPlan | None:
        ordered = RunnableJobQueue(policy=self.policy, jobs=list(jobs)).ordered()
        if not ordered:
            return None
        primary = ordered[0]

        if not self.settings.gpu_scheduler.enabled:
            return PlacementPlan(mode="exclusive", backend_name="exclusive", job_ids=(primary.job_id,), reason="gpu scheduler disabled")
        if self.settings.gpu_scheduler.max_packed_jobs_per_gpu < 2:
            return PlacementPlan(mode="exclusive", backend_name="exclusive", job_ids=(primary.job_id,), reason="pair packing disabled")
        if "mps" not in self.settings.gpu_scheduler.backend_priority or not backend_available.get("mps", False):
            return PlacementPlan(mode="exclusive", backend_name="exclusive", job_ids=(primary.job_id,), reason="mps backend unavailable")
        if not self._pack_eligible(primary):
            return PlacementPlan(mode="exclusive", backend_name="exclusive", job_ids=(primary.job_id,), reason="primary job not pack eligible")

        primary_profile = self.store.get_solo_profile(primary.packing.signature or "")
        if primary_profile is None:
            return PlacementPlan(mode="exclusive", backend_name="exclusive", job_ids=(primary.job_id,), reason="primary job needs solo profile")

        best_partner: TrainingJob | None = None
        best_score = float("-inf")
        for candidate in ordered[1:]:
            if not self._pack_eligible(candidate):
                continue
            candidate_profile = self.store.get_solo_profile(candidate.packing.signature or "")
            if candidate_profile is None:
                continue
            if not self._memory_gate(primary, candidate, primary_profile, candidate_profile):
                continue
            pair_profile = self.store.get_pair_profile(primary.packing.signature or "", candidate.packing.signature or "")
            score = compatibility_score(primary, candidate, primary_profile, candidate_profile, pair_profile, self.settings)
            if score > best_score:
                best_score = score
                best_partner = candidate

        if best_partner is None or best_score < self.settings.gpu_scheduler.thresholds.min_aggregate_gain:
            return PlacementPlan(mode="exclusive", backend_name="exclusive", job_ids=(primary.job_id,), reason="no compatible partner")

        return PlacementPlan(
            mode="packed_pair",
            backend_name="mps",
            job_ids=(primary.job_id, best_partner.job_id),
            reason="compatible pair selected",
        )
