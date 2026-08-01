"""PerfSeer ML adapter.

Wraps the external SeerNet predictor (Predictor repo) behind the
``ResourcePredictionProvider`` protocol. Fail-closed: any load or inference
failure yields ``None`` so the router falls back to other providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain import (
    PredictedScalar,
    PredictionRequest,
    PredictionSource,
    ResourcePrediction,
    prediction_timestamp,
)
from .perfseer_runtime import PerfSeerRuntime, normalize_precision


@dataclass(frozen=True, slots=True)
class PerfSeerAdapterHealth:
    enabled: bool
    healthy: bool
    reason: str | None = None


class PerfSeerMLAdapter:
    name = "ml_student"
    version = "perfseer_seernet_a10_v1"
    feature_schema_version = "teacher_pipeline_g40_v1"

    def __init__(
        self,
        *,
        enabled: bool = False,
        hardware_key: str | None = None,
        checkpoint_path: str | None = None,
        calibration_path: str | None = None,
        repo_path: str | None = None,
        device: str = "cpu",
        cache_size: int = 1024,
    ):
        self.enabled = bool(enabled)
        self.hardware_key = hardware_key
        self.checkpoint_path = checkpoint_path
        self.calibration_path = calibration_path
        self.repo_path = repo_path
        self.device = str(device or "cpu")
        self.cache_size = max(0, int(cache_size or 0))
        self._runtime: PerfSeerRuntime | None = None
        if self.enabled and self.checkpoint_path and self.repo_path:
            self._runtime = PerfSeerRuntime(
                self.repo_path,
                self.checkpoint_path,
                device=self.device,
                cache_size=self.cache_size,
            )

    def health(self) -> PerfSeerAdapterHealth:
        if not self.enabled:
            return PerfSeerAdapterHealth(enabled=False, healthy=False, reason="disabled")
        if not self.checkpoint_path:
            return PerfSeerAdapterHealth(enabled=True, healthy=False, reason="missing_checkpoint")
        if not self.repo_path:
            return PerfSeerAdapterHealth(enabled=True, healthy=False, reason="missing_repo_path")
        if self._runtime is None:
            return PerfSeerAdapterHealth(enabled=True, healthy=False, reason="runtime_not_initialized")
        error = self._runtime.load_error()
        if error:
            return PerfSeerAdapterHealth(enabled=True, healthy=False, reason=error)
        return PerfSeerAdapterHealth(enabled=True, healthy=True, reason=None)

    def available(self, hardware_key: str) -> bool:
        health = self.health()
        if not health.healthy:
            return False
        return self.hardware_key in (None, "", hardware_key)

    def predict(self, request: PredictionRequest) -> ResourcePrediction | None:
        if self._runtime is None:
            return None
        source_path = request.architecture_source
        if not source_path or not Path(str(source_path)).suffix == ".py":
            return None
        input_resolution = None
        extra = request.extra_features or {}
        for key in ("input_resolution", "resolution"):
            value = extra.get(key)
            if value:
                try:
                    input_resolution = int(value)
                except (TypeError, ValueError):
                    input_resolution = None
                break
        result = self._runtime.predict_source(
            source_path,
            batch_size=request.batch_size,
            input_resolution=input_resolution,
            input_shape=request.input_shape,
            precision=request.precision,
        )
        if result is None:
            return None

        step_time = PredictedScalar(mean=result.train_step_time_ms, unit="ms")
        epoch_time = None
        remaining_runtime = None
        if request.steps_per_epoch:
            epoch_ms = result.train_step_time_ms * float(request.steps_per_epoch)
            epoch_time = PredictedScalar(mean=epoch_ms, unit="ms")
            remaining = request.remaining_epochs
            if remaining is None and request.total_epochs is not None:
                remaining = float(request.total_epochs)
            if remaining is not None:
                remaining_runtime = PredictedScalar(mean=epoch_ms * float(remaining) / 1000.0, unit="s")

        warnings: tuple[str, ...] = ()
        if normalize_precision(request.precision) != str(request.precision or "").strip().lower():
            warnings = (f"precision '{request.precision}' mapped to '{normalize_precision(request.precision)}'",)

        return ResourcePrediction(
            job_id=request.job_id,
            hardware_key=request.hardware_key,
            backend=request.backend,
            batch_size=request.batch_size,
            epoch_time_ms=epoch_time,
            step_time_ms=step_time,
            remaining_runtime_seconds=remaining_runtime,
            avg_sm_util_percent=PredictedScalar(mean=result.train_util_percent, unit="%"),
            p95_sm_util_percent=None,
            peak_vram_used_mib=PredictedScalar(mean=result.train_mem_mib, unit="MiB"),
            peak_torch_reserved_mib=None,
            peak_memory_controller_util_percent=None,
            confidence=0.70,
            out_of_distribution=False,
            source=PredictionSource.ML_STUDENT,
            predictor_version=self.version,
            feature_schema_version=self.feature_schema_version,
            produced_at=prediction_timestamp(),
            warnings=warnings,
        )

    def to_dict(self) -> dict[str, Any]:
        health = self.health()
        return {
            "name": self.name,
            "version": self.version,
            "enabled": self.enabled,
            "healthy": health.healthy,
            "reason": health.reason,
            "hardware_key": self.hardware_key,
            "checkpoint_path": self.checkpoint_path,
            "calibration_path": self.calibration_path,
            "repo_path": self.repo_path,
            "device": self.device,
            "cache_size": self.cache_size,
        }
