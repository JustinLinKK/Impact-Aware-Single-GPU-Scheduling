"""Resource estimation helpers for the placement planner."""

from __future__ import annotations

from dataclasses import dataclass

from ..profiling.runtime_probe import runtime_profile_for_job
from ..domain import (
    BatchResolution,
    SoloProfile,
    TrainingJob,
    build_batch_probe_key,
    build_batch_probe_shape_signature,
    normalize_batch_probe_search_mode,
)
from ..config import SCHEDULER_MODE_PARALLEL_TIME_AWARE, SchedulerSettings
from ..config import PREDICTION_MODE_ML_PREDICTOR
from ..prediction import JobPredictionError, MLVramPredictor
from .planning_repository import PlanningRepository


@dataclass(frozen=True, slots=True)
class BatchOptionEstimate:
    job_id: str
    batch_size: int
    avg_vram_mb: float
    seconds_per_epoch: float
    remaining_epochs: int
    remaining_runtime_seconds: float
    source: str
    confidence: float | None
    estimate_version: str


class ResourceEstimator:
    def __init__(self, settings: SchedulerSettings, repository: PlanningRepository):
        self.settings = settings
        self.repository = repository
        self.ml_predictor = (
            MLVramPredictor(settings.prediction, repository.hardware_profile()) if settings.prediction.mode == PREDICTION_MODE_ML_PREDICTOR else None
        )

    def safe_budget_mb(self) -> float:
        memory = self.settings.gpu_scheduler.memory
        if self.settings.gpu_scheduler.mode == SCHEDULER_MODE_PARALLEL_TIME_AWARE:
            total_mb = (
                float(memory.gpu_vram_gib) * 1024.0 if memory.gpu_vram_gib is not None else float(self.repository.hardware_profile().total_vram_mb or 0.0)
            )
            if total_mb > 0:
                return total_mb * float(memory.predicted_budget_fraction)
        if memory.safe_vram_budget_gib is not None:
            return float(memory.safe_vram_budget_gib) * 1024.0
        total_mb = float(self.repository.hardware_profile().total_vram_mb or 0.0)
        return total_mb * float(memory.predicted_budget_fraction)

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

    def estimate_batch_options(self, job: TrainingJob, backend_name: str, batch_sizes: list[int]) -> list[BatchOptionEstimate]:
        """Return batch-indexed memory and epoch-time estimates in one bundle."""
        total_epochs = job.max_epochs or job.config.max_epochs
        if total_epochs is None:
            return []
        try:
            total_epochs = max(0, int(total_epochs))
            completed_epochs = max(0, int(job.metadata.get("last_completed_epoch", 0)))
        except (TypeError, ValueError):
            return []
        remaining_epochs = max(0, total_epochs - completed_epochs)
        ml_predictions: dict[int, float] | None = None
        if self.ml_predictor is not None:
            try:
                ml_predictions = self.ml_predictor.predict_avg_vram_options(job, batch_sizes)
            except JobPredictionError:
                ml_predictions = {}
        estimates: list[BatchOptionEstimate] = []
        for batch_size in batch_sizes:
            memory_mb, memory_source = self._time_aware_memory_estimate(
                job,
                batch_size,
                backend_name,
                ml_predictions=ml_predictions,
            )
            if memory_mb <= 0:
                continue
            seconds_per_epoch, runtime_source, confidence = self._seconds_per_epoch(job, batch_size, backend_name)
            if seconds_per_epoch is None or seconds_per_epoch <= 0:
                continue
            source = runtime_source if runtime_source == memory_source else f"{memory_source}+{runtime_source}"
            estimates.append(
                BatchOptionEstimate(
                    job_id=job.job_id,
                    batch_size=int(batch_size),
                    avg_vram_mb=float(memory_mb),
                    seconds_per_epoch=float(seconds_per_epoch),
                    remaining_epochs=remaining_epochs,
                    remaining_runtime_seconds=float(seconds_per_epoch) * remaining_epochs,
                    source=source,
                    confidence=confidence,
                    estimate_version=self.settings.gpu_scheduler.objective.objective_version,
                )
            )
        return estimates

    def _time_aware_memory_estimate(
        self,
        job: TrainingJob,
        batch_size: int,
        backend_name: str,
        *,
        ml_predictions: dict[int, float] | None = None,
    ) -> tuple[float, str]:
        if ml_predictions is not None and int(batch_size) in ml_predictions:
            return float(ml_predictions[int(batch_size)]), "ml_predictor"
        if ml_predictions is None and self.ml_predictor is not None:
            try:
                return (
                    float(self.ml_predictor.predict_avg_vram_mb(job, batch_size)),
                    "ml_predictor",
                )
            except JobPredictionError:
                pass
        hardware = self.repository.hardware_profile()
        exact = self.repository.get_batch_size_observation(
            model_key=self.model_key(job),
            shape_signature=self.shape_signature(job),
            hardware_key=hardware.hardware_key,
            backend_name=backend_name,
            batch_size=int(batch_size),
        )
        if exact is None and backend_name != "exclusive":
            exact = self.repository.get_batch_size_observation(
                model_key=self.model_key(job),
                shape_signature=self.shape_signature(job),
                hardware_key=hardware.hardware_key,
                backend_name="exclusive",
                batch_size=int(batch_size),
            )
        if exact is not None and exact.avg_vram_mb is not None:
            return float(exact.avg_vram_mb), str(exact.metadata.get("estimate_source") or "branch_profile")
        observed = sorted(
            (
                item
                for item in self.repository.list_batch_size_observations(
                    model_key=self.model_key(job),
                    shape_signature=self.shape_signature(job),
                    hardware_key=hardware.hardware_key,
                    backend_name=backend_name,
                )
                if item.avg_vram_mb is not None and item.batch_size > 0
            ),
            key=lambda item: item.batch_size,
        )
        if not observed and backend_name != "exclusive":
            observed = sorted(
                (
                    item
                    for item in self.repository.list_batch_size_observations(
                        model_key=self.model_key(job),
                        shape_signature=self.shape_signature(job),
                        hardware_key=hardware.hardware_key,
                        backend_name="exclusive",
                    )
                    if item.avg_vram_mb is not None and item.batch_size > 0
                ),
                key=lambda item: item.batch_size,
            )
        lower = max(
            (item for item in observed if item.batch_size < batch_size),
            key=lambda item: item.batch_size,
            default=None,
        )
        upper = min(
            (item for item in observed if item.batch_size > batch_size),
            key=lambda item: item.batch_size,
            default=None,
        )
        if lower is not None and upper is not None:
            position = (batch_size - lower.batch_size) / (upper.batch_size - lower.batch_size)
            value = float(lower.avg_vram_mb) + position * (float(upper.avg_vram_mb) - float(lower.avg_vram_mb))
            sources = {str(item.metadata.get("estimate_source") or "branch_profile") for item in (lower, upper)}
            return value, sources.pop() if len(sources) == 1 else "branch_profile"
        if int(batch_size) == BatchResolution.requested_batch_size(job):
            profile = self.solo_profile(job)
            if profile is not None and profile.avg_vram_mb is not None:
                return float(profile.avg_vram_mb), "branch_profile"
            if job.resource_requirements.estimated_avg_vram_mb is not None:
                return (
                    float(job.resource_requirements.estimated_avg_vram_mb),
                    "branch_profile",
                )
        return 0.0, "missing"

    def _seconds_per_epoch(self, job: TrainingJob, batch_size: int, backend_name: str) -> tuple[float | None, str, float | None]:
        profile = self.repository.get_runtime_profile(
            job.packing.signature or job.job_id,
            resolved_batch_size=int(batch_size),
            backend_name=backend_name,
        )
        if profile is None and backend_name != "exclusive":
            profile = self.repository.get_runtime_profile(
                job.packing.signature or job.job_id,
                resolved_batch_size=int(batch_size),
                backend_name="exclusive",
            )
        if profile is not None:
            if profile.epoch_1_seconds is not None and profile.epoch_1_seconds > 0:
                return (
                    float(profile.epoch_1_seconds),
                    str(profile.source or "branch_profile"),
                    float(profile.confidence),
                )
            total_epochs = job.max_epochs or job.config.max_epochs
            if profile.estimated_total_runtime_seconds is not None and total_epochs:
                return (
                    float(profile.estimated_total_runtime_seconds) / max(1, int(total_epochs)),
                    str(profile.source or "branch_profile"),
                    float(profile.confidence),
                )

        hardware = self.repository.hardware_profile()
        observation = self.repository.get_batch_size_observation(
            model_key=self.model_key(job),
            shape_signature=self.shape_signature(job),
            hardware_key=hardware.hardware_key,
            backend_name=backend_name,
            batch_size=int(batch_size),
        )
        if observation is None and backend_name != "exclusive":
            observation = self.repository.get_batch_size_observation(
                model_key=self.model_key(job),
                shape_signature=self.shape_signature(job),
                hardware_key=hardware.hardware_key,
                backend_name="exclusive",
                batch_size=int(batch_size),
            )
        if observation is not None:
            raw_seconds = observation.metadata.get("seconds_per_epoch")
            if raw_seconds is None and observation.avg_step_time_ms is not None:
                steps = observation.metadata.get("steps_per_epoch") or job.metadata.get("runtime_steps_per_epoch")
                if steps is not None:
                    raw_seconds = float(observation.avg_step_time_ms) * int(steps) / 1000.0
            try:
                if raw_seconds is not None and float(raw_seconds) > 0:
                    confidence_value = observation.metadata.get("confidence")
                    confidence = float(confidence_value) if confidence_value is not None else None
                    source = str(observation.metadata.get("estimate_source") or "branch_profile")
                    return float(raw_seconds), source, confidence
            except (TypeError, ValueError):
                pass
        return None, "missing", None

    @staticmethod
    def pareto_prune(options: list[BatchOptionEstimate]) -> list[BatchOptionEstimate]:
        retained: list[BatchOptionEstimate] = []
        for option in options:
            dominated = any(
                other.batch_size != option.batch_size
                and other.avg_vram_mb <= option.avg_vram_mb
                and other.remaining_runtime_seconds <= option.remaining_runtime_seconds
                and (other.avg_vram_mb < option.avg_vram_mb or other.remaining_runtime_seconds < option.remaining_runtime_seconds)
                for other in options
            )
            if not dominated:
                retained.append(option)
        return sorted(retained, key=lambda item: item.batch_size)

    def solo_profile(self, job: TrainingJob) -> SoloProfile | None:
        if not job.packing.signature:
            return None
        return self.repository.get_solo_profile(job.packing.signature)

    def has_memory_estimate(self, job: TrainingJob, backend_name: str) -> bool:
        return self.estimate_avg_vram_mb(job, self.resolved_batch_size(job), backend_name) > 0.0

    def estimate_avg_vram_mb(self, job: TrainingJob, batch_size: int, backend_name: str) -> float:
        if self.ml_predictor is not None:
            try:
                value = self.ml_predictor.predict_avg_vram_mb(job, batch_size)
                job.metadata["vram_prediction_source"] = "ml_predictor"
                job.metadata.pop("vram_prediction_error", None)
                return value
            except JobPredictionError as exc:
                job.metadata["vram_prediction_source"] = "branch_profile"
                job.metadata["vram_prediction_error"] = str(exc)
        return self._estimate_branch_avg_vram_mb(job, batch_size, backend_name)

    def _estimate_branch_avg_vram_mb(self, job: TrainingJob, batch_size: int, backend_name: str) -> float:
        hardware = self.repository.hardware_profile()
        observation = self.repository.get_batch_size_observation(
            model_key=self.model_key(job),
            shape_signature=self.shape_signature(job),
            hardware_key=hardware.hardware_key,
            backend_name=backend_name,
            batch_size=batch_size,
        )
        if observation and observation.avg_vram_mb is not None:
            return float(observation.avg_vram_mb)

        related = self.repository.list_batch_size_observations(
            model_key=self.model_key(job),
            shape_signature=self.shape_signature(job),
            hardware_key=hardware.hardware_key,
            backend_name=backend_name,
        )
        candidates = [item for item in related if item.avg_vram_mb is not None and item.batch_size > 0]
        if candidates:
            nearest = min(candidates, key=lambda item: abs(item.batch_size - batch_size))
            return float(nearest.avg_vram_mb) * (float(batch_size) / float(max(1, nearest.batch_size)))

        device_type = hardware.gpu_name
        search_mode = normalize_batch_probe_search_mode(job.batch_probe.search_mode or self.settings.gpu_scheduler.batch_probe_search_mode)
        probe_key = build_batch_probe_key(
            self.model_key(job),
            device_type,
            self.shape_signature(job),
            search_mode=search_mode,
        )
        batch_profile = self.repository.get_batch_probe_profile(probe_key)
        if batch_profile and batch_profile.avg_vram_mb is not None:
            base_batch = max(1, int(batch_profile.resolved_batch_size))
            return float(batch_profile.avg_vram_mb) * (float(batch_size) / float(base_batch))

        solo_profile = self.solo_profile(job)
        if solo_profile and solo_profile.avg_vram_mb is not None:
            base_batch = max(1, self.resolved_batch_size(job))
            return float(solo_profile.avg_vram_mb) * (float(batch_size) / float(base_batch))

        if job.resource_requirements.estimated_avg_vram_mb is not None:
            base_batch = max(1, self.resolved_batch_size(job))
            return float(job.resource_requirements.estimated_avg_vram_mb) * (float(batch_size) / float(base_batch))
        return 0.0

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
        probe_key = build_batch_probe_key(
            self.model_key(job),
            device_type,
            self.shape_signature(job),
            search_mode=search_mode,
        )
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
        return sum(self.estimate_avg_vram_mb(job, self.resolved_batch_size(job), backend_name) for job in jobs)

    def predicted_group_sm_utilization(self, jobs: list[TrainingJob], *, backend_name: str) -> float:
        return sum(self.estimate_sm_utilization(job, self.resolved_batch_size(job), backend_name) for job in jobs)

    def prediction_metadata(self, job_id: str) -> dict[str, str | None]:
        if self.ml_predictor is None:
            return {
                "vram_prediction_source": "branch_profile",
                "vram_prediction_error": None,
            }
        return {
            "vram_prediction_source": self.ml_predictor.last_sources.get(job_id, "branch_profile"),
            "vram_prediction_error": self.ml_predictor.last_errors.get(job_id),
        }
