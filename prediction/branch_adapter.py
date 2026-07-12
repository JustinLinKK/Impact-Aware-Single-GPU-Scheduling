"""Adapter for branch-style scheduler resource predictions."""

from __future__ import annotations

from typing import Any

from ..domain import (
    PredictedScalar,
    PredictionRequest,
    PredictionSource,
    ResourcePrediction,
    prediction_timestamp,
)


_PREDICTION_METADATA_KEYS = ("branch_prediction", "resource_prediction", "scheduler_prediction")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _scalar(payload: dict[str, Any], *keys: str, unit: str = "") -> PredictedScalar | None:
    value = None
    for key in keys:
        if key in payload:
            value = payload.get(key)
            break
    mean = _as_float(value)
    if mean is None:
        return None
    lower = _as_float(payload.get(f"{keys[0]}_lower"))
    upper = _as_float(payload.get(f"{keys[0]}_upper"))
    return PredictedScalar(mean=mean, lower=lower, upper=upper, unit=unit)


class BranchPredictionAdapter:
    """Wrap scheduler-visible branch prediction metadata behind the provider protocol.

    The repository currently has no callable structural branch predictor entry point. This
    adapter therefore accepts precomputed branch predictions attached to the job metadata
    by upstream code and otherwise returns ``None`` so the router can fail closed.
    """

    name = "branch"
    version = "branch_adapter_v1"

    def __init__(self, *, fixed_confidence_if_uncalibrated: float = 0.55):
        self.fixed_confidence_if_uncalibrated = float(fixed_confidence_if_uncalibrated)

    def available(self, hardware_key: str) -> bool:
        del hardware_key
        return True

    def predict(self, request: PredictionRequest) -> ResourcePrediction | None:
        prediction_payload = request.extra_features.get("branch_prediction")
        if not isinstance(prediction_payload, dict):
            return None
        payload = dict(prediction_payload)
        confidence = _as_float(payload.get("confidence"))
        warnings: list[str] = [str(item) for item in payload.get("warnings", ()) or ()]
        if confidence is None:
            confidence = self.fixed_confidence_if_uncalibrated
            warnings.append("branch_confidence_uncalibrated")
        epoch_time = _scalar(payload, "epoch_time_ms", "train_epoch_ms", unit="ms")
        steps_per_epoch = request.steps_per_epoch
        step_time = _scalar(payload, "step_time_ms", "train_step_ms", unit="ms")
        if step_time is None and epoch_time is not None and steps_per_epoch and steps_per_epoch > 0:
            step_time = PredictedScalar(mean=epoch_time.mean / float(steps_per_epoch), unit="ms")
        remaining_runtime = _scalar(payload, "remaining_runtime_seconds", "estimated_remaining_runtime_seconds", unit="s")
        if remaining_runtime is None and epoch_time is not None and request.remaining_epochs is not None:
            remaining_runtime = PredictedScalar(mean=(epoch_time.mean * float(request.remaining_epochs)) / 1000.0, unit="s")
        return ResourcePrediction(
            job_id=request.job_id,
            hardware_key=request.hardware_key,
            backend=request.backend,
            batch_size=request.batch_size,
            epoch_time_ms=epoch_time,
            step_time_ms=step_time,
            remaining_runtime_seconds=remaining_runtime,
            avg_sm_util_percent=_scalar(payload, "avg_sm_util_percent", "avg_gpu_utilization_percent", unit="percent"),
            p95_sm_util_percent=_scalar(payload, "p95_sm_util_percent", unit="percent"),
            peak_vram_used_mib=_scalar(payload, "peak_vram_used_mib", "peak_vram_mb", unit="MiB"),
            peak_torch_reserved_mib=_scalar(payload, "peak_torch_reserved_mib", "peak_reserved_vram_mb", unit="MiB"),
            peak_memory_controller_util_percent=_scalar(payload, "peak_memory_controller_util_percent", unit="percent"),
            confidence=max(0.0, min(1.0, float(confidence))),
            out_of_distribution=bool(payload.get("out_of_distribution", False)),
            source=PredictionSource.BRANCH,
            predictor_version=str(payload.get("predictor_version") or self.version),
            feature_schema_version=payload.get("feature_schema_version"),
            produced_at=prediction_timestamp(),
            warnings=tuple(dict.fromkeys(warnings)),
        )


def branch_prediction_metadata(job_metadata: dict[str, Any]) -> dict[str, Any] | None:
    for key in _PREDICTION_METADATA_KEYS:
        value = job_metadata.get(key)
        if isinstance(value, dict):
            return value
    return None
