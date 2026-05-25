"""Resource estimation helpers for the placement planner."""

from __future__ import annotations

from ..profiling.runtime_probe import runtime_profile_for_job
from ..domain import BatchResolution, SoloProfile, TrainingJob, build_batch_probe_key, build_batch_probe_shape_signature, normalize_batch_probe_search_mode
from ..config import SchedulerSettings
from .planning_repository import PlanningRepository


class ResourceEstimator:
    def __init__(self, settings: SchedulerSettings, repository: PlanningRepository):
        self.settings = settings
        self.repository = repository

    def safe_budget_mb(self) -> float:
        hardware = self.repository.hardware_profile()
        return self.settings.gpu_scheduler.memory.budget_mb(hardware.total_vram_mb)

    def resolved_batch_size(self, job: TrainingJob) -> int:
        return BatchResolution.resolved_batch_size(job)

    def shape_signature(self, job: TrainingJob) -> str:
        return build_batch_probe_shape_signature(job)

    def model_key(self, job: TrainingJob) -> str:
        return str(job.batch_probe.model_key or job.baseline_model_id)

    def runtime_ready(self, job: TrainingJob, backend_name: str) -> bool:
        if not job.runtime_probe.enabled:
            return False
        return runtime_profile_for_job(self.repository, job, backend_name=backend_name) is not None

    def predicted_remaining_runtime_seconds(self, job: TrainingJob, *, backend_name: str) -> float | None:
        if job.metadata.get("runtime_remaining_runtime_seconds") is not None:
            try:
                return max(0.0, float(job.metadata["runtime_remaining_runtime_seconds"]))
            except (TypeError, ValueError):
                pass
        profile = runtime_profile_for_job(self.repository, job, backend_name=backend_name)
        if profile is None or profile.estimated_total_runtime_seconds is None:
            return None
        return max(0.0, float(profile.estimated_total_runtime_seconds))

    def solo_profile(self, job: TrainingJob) -> SoloProfile | None:
        if not job.packing.signature:
            return None
        return self.repository.get_solo_profile(job.packing.signature)

    def has_memory_estimate(self, job: TrainingJob, backend_name: str) -> bool:
        return self.estimate_peak_vram_mb(job, self.resolved_batch_size(job), backend_name) > 0.0

    def estimate_peak_vram_mb(self, job: TrainingJob, batch_size: int, backend_name: str) -> float:
        hardware = self.repository.hardware_profile()
        observation = self.repository.get_batch_size_observation(
            model_key=self.model_key(job),
            shape_signature=self.shape_signature(job),
            hardware_key=hardware.hardware_key,
            backend_name=backend_name,
            batch_size=batch_size,
        )
        if observation and observation.peak_vram_mb is not None:
            return float(observation.peak_vram_mb)

        related = self.repository.list_batch_size_observations(
            model_key=self.model_key(job),
            shape_signature=self.shape_signature(job),
            hardware_key=hardware.hardware_key,
            backend_name=backend_name,
        )
        candidates = [item for item in related if item.peak_vram_mb is not None and item.batch_size > 0]
        if candidates:
            nearest = min(candidates, key=lambda item: abs(item.batch_size - batch_size))
            return float(nearest.peak_vram_mb) * (float(batch_size) / float(max(1, nearest.batch_size)))

        device_type = hardware.gpu_name
        search_mode = normalize_batch_probe_search_mode(job.batch_probe.search_mode or self.settings.gpu_scheduler.batch_probe_search_mode)
        probe_key = build_batch_probe_key(self.model_key(job), device_type, self.shape_signature(job), search_mode=search_mode)
        batch_profile = self.repository.get_batch_probe_profile(probe_key)
        if batch_profile and batch_profile.peak_vram_mb is not None:
            base_batch = max(1, int(batch_profile.resolved_batch_size))
            return float(batch_profile.peak_vram_mb) * (float(batch_size) / float(base_batch))

        solo_profile = self.solo_profile(job)
        if solo_profile and solo_profile.peak_vram_mb is not None:
            base_batch = max(1, self.resolved_batch_size(job))
            return float(solo_profile.peak_vram_mb) * (float(batch_size) / float(base_batch))

        if job.resource_requirements.estimated_vram_mb is not None:
            base_batch = max(1, self.resolved_batch_size(job))
            return float(job.resource_requirements.estimated_vram_mb) * (float(batch_size) / float(base_batch))
        return 0.0

    def estimate_sm_utilization(self, job: TrainingJob, batch_size: int, backend_name: str) -> float:
        hardware_key = self.repository.hardware_key()
        observation = self.repository.get_batch_size_observation(
            model_key=self.model_key(job),
            shape_signature=self.shape_signature(job),
            hardware_key=hardware_key,
            backend_name=backend_name,
            batch_size=batch_size,
        )
        if observation and observation.avg_gpu_utilization is not None:
            return max(0.0, float(observation.avg_gpu_utilization))
        related = self.repository.list_batch_size_observations(
            model_key=self.model_key(job),
            shape_signature=self.shape_signature(job),
            hardware_key=hardware_key,
            backend_name=backend_name,
        )
        util_candidates = [item for item in related if item.avg_gpu_utilization is not None]
        if util_candidates:
            nearest = min(util_candidates, key=lambda item: abs(item.batch_size - batch_size))
            return max(0.0, float(nearest.avg_gpu_utilization))
        solo_profile = self.solo_profile(job)
        if solo_profile and solo_profile.avg_gpu_utilization is not None:
            return max(0.0, float(solo_profile.avg_gpu_utilization))
        return 0.0

    def predicted_group_vram_mb(self, jobs: list[TrainingJob], *, backend_name: str) -> float:
        return sum(self.estimate_peak_vram_mb(job, self.resolved_batch_size(job), backend_name) for job in jobs)

    def predicted_group_sm_utilization(self, jobs: list[TrainingJob], *, backend_name: str) -> float:
        return sum(self.estimate_sm_utilization(job, self.resolved_batch_size(job), backend_name) for job in jobs)
