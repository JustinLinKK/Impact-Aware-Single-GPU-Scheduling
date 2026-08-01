"""Resource estimation helpers for the placement planner."""

from __future__ import annotations

from typing import Any

from ..domain import (
    BatchProbeProfile,
    BatchResolution,
    BatchSizeObservation,
    PredictedScalar,
    PredictionSource,
    ResourcePrediction,
    TrainingJob,
    build_batch_probe_shape_signature,
    prediction_timestamp,
)
from ..prediction import PredictionRouter, PredictionRouterResult, build_prediction_request
from ..config import SchedulerSettings
from .planning_repository import PlanningRepository


class ResourceEstimator:
    def __init__(self, settings: SchedulerSettings, repository: PlanningRepository):
        self.settings = settings
        self.repository = repository
        self.prediction_router = PredictionRouter.from_settings(settings)
        self.last_prediction_results: dict[tuple[str, str, int], dict[str, Any]] = {}

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
        return self.predicted_remaining_runtime_seconds(job, backend_name=backend_name) is not None

    def requires_successful_runtime_profile_for_packing(self, job: TrainingJob) -> bool:
        del job
        return False

    def packing_runtime_profile(self, job: TrainingJob, backend_name: str):
        del job, backend_name
        return None

    def packing_batch_size(self, job: TrainingJob, backend_name: str) -> int:
        del backend_name
        return self.resolved_batch_size(job)

    def predicted_remaining_runtime_seconds(self, job: TrainingJob, *, backend_name: str) -> float | None:
        if job.metadata.get("runtime_remaining_runtime_seconds") is not None:
            try:
                return max(0.0, float(job.metadata["runtime_remaining_runtime_seconds"]))
            except (TypeError, ValueError):
                pass
        prediction = self.resource_prediction(job, self.resolved_batch_size(job), backend_name)
        if prediction is None or prediction.remaining_runtime_seconds is None:
            return None
        return max(0.0, float(prediction.remaining_runtime_seconds.mean))

    def solo_profile(self, job: TrainingJob):
        del job
        return None

    def has_memory_estimate(self, job: TrainingJob, backend_name: str) -> bool:
        return self.estimate_peak_vram_mb(job, self.resolved_batch_size(job), backend_name) is not None

    def estimate_peak_vram_mb(self, job: TrainingJob, batch_size: int, backend_name: str) -> float | None:
        prediction = self.resource_prediction(job, batch_size, backend_name)
        if prediction is None:
            return None
        return self.reserved_vram_upper_mib(prediction)

    def estimate_sm_utilization(self, job: TrainingJob, batch_size: int, backend_name: str) -> float | None:
        prediction = self.resource_prediction(job, batch_size, backend_name)
        if prediction is None or prediction.avg_sm_util_percent is None:
            return None
        scalar = prediction.avg_sm_util_percent
        value = float(scalar.mean)
        unit = str(scalar.unit or "").lower()
        if unit in {"percent", "%"} or value > 1.0:
            value = value / 100.0
        return max(0.0, value)

    def predicted_group_vram_mb(self, jobs: list[TrainingJob], *, backend_name: str) -> float:
        total = 0.0
        for job in jobs:
            estimate = self.estimate_peak_vram_mb(job, self.packing_batch_size(job, backend_name), backend_name)
            if estimate is not None:
                total += estimate
        return total

    def predicted_group_sm_utilization(self, jobs: list[TrainingJob], *, backend_name: str) -> float:
        total = 0.0
        for job in jobs:
            estimate = self.estimate_sm_utilization(job, self.packing_batch_size(job, backend_name), backend_name)
            if estimate is not None:
                total += estimate
        return total

    def resource_prediction(self, job: TrainingJob, batch_size: int, backend_name: str) -> ResourcePrediction | None:
        live_prediction = self._live_memory_prediction(job, batch_size, backend_name)
        if live_prediction is not None:
            return live_prediction

        hardware_key = self.repository.hardware_key()
        request = build_prediction_request(job, hardware_key=hardware_key, backend=backend_name, batch_size=batch_size)
        router_result = self.prediction_router.predict(request)
        self._record_prediction_result(job, batch_size, backend_name, router_result)
        if router_result.selected is not None:
            return router_result.selected

        compatible_profile = self._compatible_batch_probe_profile(job)
        if compatible_profile is not None:
            return self._prediction_from_batch_probe_profile(
                job,
                batch_size,
                backend_name,
                compatible_profile,
                source=PredictionSource.BRANCH,
                predictor_version="branch_batch_probe_profile_v1",
                confidence=0.88,
                warnings=("branch_batch_probe_profile",),
            )

        explicit_prediction = self._explicit_prediction(job, batch_size, backend_name)
        if explicit_prediction is not None:
            return explicit_prediction

        local_prediction = self._job_local_probe_prediction(job, batch_size, backend_name)
        if local_prediction is not None:
            return local_prediction
        return None

    def reserved_vram_upper_mib(self, prediction: ResourcePrediction) -> float | None:
        scalar = prediction.peak_torch_reserved_mib or prediction.peak_vram_used_mib
        if scalar is None:
            return None
        if scalar.upper is not None:
            return max(0.0, float(scalar.upper))
        margin_settings = getattr(getattr(self.settings, "prediction", None), "safety_margin", None)
        margin_for = getattr(margin_settings, "margin_for", None)
        margin = (
            float(margin_for(prediction.source.value, warnings=prediction.warnings))
            if callable(margin_for)
            else 1.30
        )
        return max(0.0, float(scalar.mean) * margin)

    def _record_prediction_result(
        self,
        job: TrainingJob,
        batch_size: int,
        backend_name: str,
        result: PredictionRouterResult,
    ) -> None:
        self.last_prediction_results[(job.job_id, backend_name, int(batch_size))] = result.to_dict()

    def _live_memory_prediction(self, job: TrainingJob, batch_size: int, backend_name: str) -> ResourcePrediction | None:
        reserved = self._metadata_float(job, "runtime_peak_torch_reserved_mib", "runtime_peak_torch_reserved_mb")
        vram = self._metadata_float(job, "runtime_peak_vram_used_mib", "runtime_peak_vram_mb")
        if reserved is None and vram is None:
            return None
        return self._prediction_from_scalars(
            job=job,
            batch_size=batch_size,
            backend_name=backend_name,
            source=PredictionSource.JOB_LOCAL_PROBE,
            predictor_version="runtime_live_correction_v1",
            peak_vram_used_mib=vram,
            peak_torch_reserved_mib=reserved,
            confidence=1.0,
            warnings=("same_job_live_memory_override",),
        )

    def _job_local_probe_prediction(self, job: TrainingJob, batch_size: int, backend_name: str) -> ResourcePrediction | None:
        observation = self._job_local_batch_observation(job, batch_size, backend_name)
        if observation is not None:
            return self._prediction_from_batch_observation(job, batch_size, backend_name, observation)
        profile = self._job_local_batch_probe_profile(job)
        if profile is not None:
            return self._prediction_from_batch_probe_profile(job, batch_size, backend_name, profile)
        return None

    def _job_local_batch_observation(
        self,
        job: TrainingJob,
        batch_size: int,
        backend_name: str,
    ) -> BatchSizeObservation | None:
        hardware_key = self.repository.hardware_key()
        exact = self.repository.get_batch_size_observation(
            model_key=self.model_key(job),
            shape_signature=self.shape_signature(job),
            hardware_key=hardware_key,
            backend_name=backend_name,
            batch_size=batch_size,
        )
        if exact is not None and exact.last_job_id == job.job_id:
            return exact
        related = self.repository.list_batch_size_observations(
            model_key=self.model_key(job),
            shape_signature=self.shape_signature(job),
            hardware_key=hardware_key,
            backend_name=backend_name,
        )
        candidates = [
            item
            for item in related
            if item.last_job_id == job.job_id
            and item.peak_vram_mb is not None
            and int(item.batch_size or 0) > 0
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: abs(int(item.batch_size) - int(batch_size)))

    def _job_local_batch_probe_profile(self, job: TrainingJob) -> BatchProbeProfile | None:
        probe_keys = [
            job.metadata.get("batch_probe_key"),
            job.batch_probe.profile_key,
        ]
        for probe_key in probe_keys:
            if not probe_key:
                continue
            profile = self.repository.get_batch_probe_profile(str(probe_key))
            if profile is not None and profile.last_job_id == job.job_id:
                return profile
        return None

    def _compatible_batch_probe_profile(self, job: TrainingJob) -> BatchProbeProfile | None:
        getter = getattr(self.repository, "get_compatible_batch_probe_profile", None)
        if not callable(getter):
            return None
        profile_namespace = (
            job.batch_probe.profile_namespace
            or job.metadata.get("batch_probe_profile_namespace")
            or job.metadata.get("branch_profile_key")
            or job.metadata.get("model_family_profile_key")
        )
        if not profile_namespace:
            return None
        shape_signature = (
            job.batch_probe.shape_signature_override
            or job.metadata.get("branch_shape_signature")
            or job.metadata.get("model_family_shape_signature")
            or self.shape_signature(job)
        )
        search_mode = job.batch_probe.search_mode or self.settings.gpu_scheduler.batch_probe_search_mode
        contract_version = int(job.batch_probe.contract_version or job.metadata.get("batch_probe_contract_version") or 2)
        try:
            profile = getter(
                profile_namespace=str(profile_namespace),
                hardware_key=self.repository.hardware_key(),
                shape_signature=str(shape_signature),
                search_mode=str(search_mode),
                contract_version=contract_version,
            )
        except Exception:
            return None
        if profile is None:
            return None
        if profile.peak_vram_mb is None or int(profile.resolved_batch_size or 0) <= 0:
            return None
        return profile

    def _prediction_from_batch_observation(
        self,
        job: TrainingJob,
        batch_size: int,
        backend_name: str,
        observation: BatchSizeObservation,
    ) -> ResourcePrediction:
        base_batch = max(1, int(observation.batch_size or batch_size))
        ratio = float(batch_size) / float(base_batch)
        peak_vram = float(observation.peak_vram_mb) * ratio if observation.peak_vram_mb is not None else None
        avg_sm = self._util_fraction_to_percent(observation.avg_gpu_utilization)
        return self._prediction_from_scalars(
            job=job,
            batch_size=batch_size,
            backend_name=backend_name,
            source=PredictionSource.JOB_LOCAL_PROBE,
            predictor_version="job_local_batch_observation_v1",
            peak_vram_used_mib=peak_vram,
            avg_sm_util_percent=avg_sm,
            step_time_ms=observation.avg_step_time_ms,
            confidence=0.85,
            warnings=("same_job_batch_observation",),
        )

    def _prediction_from_batch_probe_profile(
        self,
        job: TrainingJob,
        batch_size: int,
        backend_name: str,
        profile: BatchProbeProfile,
        *,
        source: PredictionSource = PredictionSource.JOB_LOCAL_PROBE,
        predictor_version: str = "job_local_batch_probe_v1",
        confidence: float = 0.90,
        warnings: tuple[str, ...] = ("same_job_batch_probe",),
    ) -> ResourcePrediction:
        base_batch = max(1, int(profile.resolved_batch_size))
        ratio = float(batch_size) / float(base_batch)
        peak_vram = float(profile.peak_vram_mb) * ratio if profile.peak_vram_mb is not None else None
        avg_step = self._metadata_float_from_mapping(profile.metadata, "avg_step_time_ms")
        return self._prediction_from_scalars(
            job=job,
            batch_size=batch_size,
            backend_name=backend_name,
            source=source,
            predictor_version=predictor_version,
            peak_vram_used_mib=peak_vram,
            step_time_ms=avg_step,
            confidence=confidence,
            warnings=warnings,
        )

    def _explicit_prediction(self, job: TrainingJob, batch_size: int, backend_name: str) -> ResourcePrediction | None:
        estimated = job.resource_requirements.estimated_vram_mb
        if estimated is None:
            return None
        base_batch = max(1, self.resolved_batch_size(job))
        peak_vram = float(estimated) * (float(batch_size) / float(base_batch))
        return self._prediction_from_scalars(
            job=job,
            batch_size=batch_size,
            backend_name=backend_name,
            source=PredictionSource.EXPLICIT,
            predictor_version="explicit_job_requirements_v1",
            peak_vram_used_mib=peak_vram,
            confidence=0.50,
            warnings=("explicit_memory_uncalibrated",),
        )

    def _prediction_from_scalars(
        self,
        *,
        job: TrainingJob,
        batch_size: int,
        backend_name: str,
        source: PredictionSource,
        predictor_version: str,
        peak_vram_used_mib: float | None = None,
        peak_torch_reserved_mib: float | None = None,
        avg_sm_util_percent: float | None = None,
        step_time_ms: float | None = None,
        confidence: float,
        warnings: tuple[str, ...] = (),
    ) -> ResourcePrediction:
        remaining = self._metadata_float(job, "runtime_remaining_runtime_seconds")
        return ResourcePrediction(
            job_id=job.job_id,
            hardware_key=self.repository.hardware_key(),
            backend=backend_name,
            batch_size=max(1, int(batch_size)),
            epoch_time_ms=None,
            step_time_ms=PredictedScalar(step_time_ms, unit="ms") if step_time_ms is not None else None,
            remaining_runtime_seconds=PredictedScalar(remaining, unit="s") if remaining is not None else None,
            avg_sm_util_percent=PredictedScalar(avg_sm_util_percent, unit="percent") if avg_sm_util_percent is not None else None,
            p95_sm_util_percent=None,
            peak_vram_used_mib=PredictedScalar(peak_vram_used_mib, unit="MiB") if peak_vram_used_mib is not None else None,
            peak_torch_reserved_mib=PredictedScalar(peak_torch_reserved_mib, unit="MiB") if peak_torch_reserved_mib is not None else None,
            peak_memory_controller_util_percent=None,
            confidence=max(0.0, min(1.0, float(confidence))),
            out_of_distribution=False,
            source=source,
            predictor_version=predictor_version,
            feature_schema_version=None,
            produced_at=prediction_timestamp(),
            warnings=warnings,
        )

    def _metadata_float(self, job: TrainingJob, *keys: str) -> float | None:
        return self._metadata_float_from_mapping(job.metadata, *keys)

    def _metadata_float_from_mapping(self, mapping: dict[str, Any] | None, *keys: str) -> float | None:
        payload = mapping or {}
        for key in keys:
            if payload.get(key) is None:
                continue
            try:
                return float(payload[key])
            except (TypeError, ValueError):
                continue
        return None

    def _util_fraction_to_percent(self, value: float | None) -> float | None:
        if value is None:
            return None
        raw = float(value)
        return raw * 100.0 if raw <= 1.0 else raw
