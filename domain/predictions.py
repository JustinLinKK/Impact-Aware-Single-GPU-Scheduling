"""Prediction domain contracts shared by scheduler providers and planners."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .common import to_primitive


def prediction_timestamp() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class PredictionRequest:
    job_id: str
    architecture_source: str
    architecture_source_hash: str
    packing_signature: str | None
    hardware_key: str
    backend: str
    batch_size: int
    micro_batch_size: int | None
    gradient_accumulation_steps: int
    precision: str
    input_shape: tuple[int, ...] | None
    optimizer_name: str | None
    dataset_size: int | None
    steps_per_epoch: int | None
    total_epochs: int | None
    remaining_epochs: float | None
    dataloader_config: dict[str, object] = field(default_factory=dict)
    extra_features: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class PredictedScalar:
    mean: float
    lower: float | None = None
    upper: float | None = None
    unit: str = ""

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


class PredictionSource(str, Enum):
    BRANCH = "branch"
    ML_STUDENT = "ml_student"
    ML_TEACHER = "ml_teacher"
    EXPLICIT = "explicit"
    JOB_LOCAL_PROBE = "job_local_probe"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ResourcePrediction:
    job_id: str
    hardware_key: str
    backend: str
    batch_size: int

    epoch_time_ms: PredictedScalar | None
    step_time_ms: PredictedScalar | None
    remaining_runtime_seconds: PredictedScalar | None

    avg_sm_util_percent: PredictedScalar | None
    p95_sm_util_percent: PredictedScalar | None
    peak_vram_used_mib: PredictedScalar | None
    peak_torch_reserved_mib: PredictedScalar | None
    peak_memory_controller_util_percent: PredictedScalar | None

    confidence: float
    out_of_distribution: bool
    source: PredictionSource
    predictor_version: str
    feature_schema_version: str | None
    produced_at: datetime
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class GroupPrediction:
    job_ids: tuple[str, ...]
    backend: str
    batch_vector: tuple[int, ...]
    joint_reserved_vram_upper_mib: float | None
    slowdown_by_job: dict[str, float | None]
    aggregate_progress: float | None
    oom_probability: float | None
    confidence: float
    source: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class PredictionObservation:
    job_id: str
    run_group_id: str
    predictor_source: str
    predictor_version: str
    hardware_key: str
    backend: str
    batch_size: int

    predicted_epoch_time_ms: float | None
    observed_epoch_time_ms: float | None
    predicted_avg_sm_util_percent: float | None
    observed_avg_sm_util_percent: float | None
    predicted_p95_sm_util_percent: float | None
    observed_p95_sm_util_percent: float | None
    predicted_peak_vram_used_mib: float | None
    observed_peak_vram_used_mib: float | None
    predicted_peak_torch_reserved_mib: float | None
    observed_peak_torch_reserved_mib: float | None
    predicted_peak_memory_controller_util_percent: float | None
    observed_peak_memory_controller_util_percent: float | None

    packed_job_ids: tuple[str, ...]
    observed_slowdown: float | None
    aggregate_progress: float | None
    fallback_triggered: bool
    oom_occurred: bool
    phase: str
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)
