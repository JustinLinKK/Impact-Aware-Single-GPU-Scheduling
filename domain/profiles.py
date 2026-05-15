"""Profiling, command, and report domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import json

from .common import parse_timestamp, to_primitive, utc_now
from .identity import build_backend_scoped_pair_key, build_combination_key, build_runtime_profile_key, decode_batch_vector
from .jobs import CommandType, TrainingJob, normalize_runtime_probe_strategy


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    entries: int = 0
    pinned_entries: int = 0
    used_bytes: int = 0
    memory_budget_bytes: int | None = None
    entry_capacity: int | None = None
    max_ram_percent: float | None = None
    system_total_memory_bytes: int | None = None
    effective_memory_budget_bytes: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CacheStats":
        payload = payload or {}
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(slots=True)
class SoloProfile:
    signature: str
    hardware_key: str = ""
    family: str | None = None
    peak_vram_mb: int | None = None
    avg_gpu_utilization: float | None = None
    avg_memory_utilization: float | None = None
    sample_count: int = 0
    last_job_id: str | None = None
    updated_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "SoloProfile":
        metadata = json.loads(row["metadata_json"]) if row.get("metadata_json") else {}
        return cls(
            signature=row["signature"],
            hardware_key=row.get("hardware_key") or "",
            family=row["family"],
            peak_vram_mb=row["peak_vram_mb"],
            avg_gpu_utilization=row["avg_gpu_utilization"],
            avg_memory_utilization=row["avg_memory_utilization"],
            sample_count=row["sample_count"],
            last_job_id=row["last_job_id"],
            updated_at=row["updated_at"],
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(slots=True)
class BatchProbeTrialResult:
    fits: bool
    peak_vram_mb: int | None = None
    memory_total_mb: int | None = None
    avg_step_time_ms: float | None = None
    message: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BatchProbeTrialResult":
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(slots=True)
class BatchProbeProfile:
    probe_key: str
    model_key: str
    device_type: str
    shape_signature: str
    batch_param_name: str
    resolved_batch_size: int
    peak_vram_mb: int | None = None
    memory_total_mb: int | None = None
    target_budget_mb: int | None = None
    observations: int = 1
    last_job_id: str | None = None
    updated_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "BatchProbeProfile":
        metadata = json.loads(row["metadata_json"]) if row.get("metadata_json") else {}
        return cls(
            probe_key=row["probe_key"],
            model_key=row["model_key"],
            device_type=row["device_type"],
            shape_signature=row["shape_signature"],
            batch_param_name=row["batch_param_name"],
            resolved_batch_size=row["resolved_batch_size"],
            peak_vram_mb=row["peak_vram_mb"],
            memory_total_mb=row["memory_total_mb"],
            target_budget_mb=row["target_budget_mb"],
            observations=row["observations"],
            last_job_id=row["last_job_id"],
            updated_at=row["updated_at"],
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(slots=True)
class BatchSizeObservation:
    observation_key: str
    model_key: str
    shape_signature: str
    hardware_key: str
    backend_name: str
    batch_param_name: str
    batch_size: int
    peak_vram_mb: int | None = None
    memory_total_mb: int | None = None
    avg_step_time_ms: float | None = None
    avg_gpu_utilization: float | None = None
    avg_memory_utilization: float | None = None
    observations: int = 1
    last_job_id: str | None = None
    updated_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "BatchSizeObservation":
        metadata = json.loads(row["metadata_json"]) if row.get("metadata_json") else {}
        return cls(
            observation_key=row["observation_key"],
            model_key=row["model_key"],
            shape_signature=row["shape_signature"],
            hardware_key=row["hardware_key"],
            backend_name=row["backend_name"],
            batch_param_name=row["batch_param_name"],
            batch_size=row["batch_size"],
            peak_vram_mb=row["peak_vram_mb"],
            memory_total_mb=row["memory_total_mb"],
            avg_step_time_ms=row["avg_step_time_ms"],
            avg_gpu_utilization=row["avg_gpu_utilization"],
            avg_memory_utilization=row["avg_memory_utilization"],
            observations=row["observations"],
            last_job_id=row["last_job_id"],
            updated_at=row["updated_at"],
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(slots=True)
class PairProfile:
    pair_key: str
    left_signature: str
    right_signature: str
    hardware_key: str = ""
    backend_name: str = "exclusive"
    compatible: bool = True
    observations: int = 0
    peak_vram_mb: int | None = None
    avg_gpu_utilization: float | None = None
    avg_memory_utilization: float | None = None
    slowdown_ratio: float | None = None
    cooldown_until: str | None = None
    last_failure_reason: str | None = None
    updated_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        left_signature: str,
        right_signature: str,
        *,
        backend_name: str,
        **kwargs: Any,
    ) -> "PairProfile":
        return cls(
            pair_key=build_backend_scoped_pair_key(left_signature, right_signature, backend_name=backend_name),
            left_signature=left_signature,
            right_signature=right_signature,
            backend_name=backend_name,
            **kwargs,
        )

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "PairProfile":
        metadata = json.loads(row["metadata_json"]) if row.get("metadata_json") else {}
        return cls(
            pair_key=row["pair_key"],
            left_signature=row["left_signature"],
            right_signature=row["right_signature"],
            hardware_key=row.get("hardware_key") or "",
            backend_name=row.get("backend_name") or "exclusive",
            compatible=bool(row["compatible"]),
            observations=row["observations"],
            peak_vram_mb=row["peak_vram_mb"],
            avg_gpu_utilization=row["avg_gpu_utilization"],
            avg_memory_utilization=row["avg_memory_utilization"],
            slowdown_ratio=row["slowdown_ratio"],
            cooldown_until=row["cooldown_until"],
            last_failure_reason=row["last_failure_reason"],
            updated_at=row["updated_at"],
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)

    def on_cooldown(self) -> bool:
        cooldown_until = parse_timestamp(self.cooldown_until)
        return cooldown_until is not None and cooldown_until > datetime.now(timezone.utc)


@dataclass(slots=True)
class CombinationProfile:
    combination_key: str
    group_signature: str
    hardware_key: str
    backend_name: str
    scheduler_mode: str
    batch_vector: dict[str, int] = field(default_factory=dict)
    compatible: bool = True
    observations: int = 0
    peak_vram_mb: int | None = None
    memory_total_mb: int | None = None
    avg_gpu_utilization: float | None = None
    avg_memory_utilization: float | None = None
    avg_step_time_ms: float | None = None
    objective_score: float | None = None
    resolved_optimal: bool = False
    last_failure_reason: str | None = None
    fallback_order: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        group_signature: str,
        hardware_key: str,
        backend_name: str,
        scheduler_mode: str,
        batch_vector: dict[str, int],
        **kwargs: Any,
    ) -> "CombinationProfile":
        return cls(
            combination_key=build_combination_key(group_signature, hardware_key, backend_name, scheduler_mode, batch_vector),
            group_signature=group_signature,
            hardware_key=hardware_key,
            backend_name=backend_name,
            scheduler_mode=scheduler_mode,
            batch_vector={str(key): int(value) for key, value in batch_vector.items()},
            **kwargs,
        )

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "CombinationProfile":
        metadata = json.loads(row["metadata_json"]) if row.get("metadata_json") else {}
        fallback_order = json.loads(row["fallback_order_json"]) if row.get("fallback_order_json") else []
        return cls(
            combination_key=row["combination_key"],
            group_signature=row["group_signature"],
            hardware_key=row["hardware_key"],
            backend_name=row["backend_name"],
            scheduler_mode=row["scheduler_mode"],
            batch_vector=decode_batch_vector(row["batch_vector_json"]),
            compatible=bool(row["compatible"]),
            observations=row["observations"],
            peak_vram_mb=row["peak_vram_mb"],
            memory_total_mb=row["memory_total_mb"],
            avg_gpu_utilization=row["avg_gpu_utilization"],
            avg_memory_utilization=row["avg_memory_utilization"],
            avg_step_time_ms=row["avg_step_time_ms"],
            objective_score=row["objective_score"],
            resolved_optimal=bool(row["resolved_optimal"]),
            last_failure_reason=row["last_failure_reason"],
            fallback_order=[str(item) for item in fallback_order],
            updated_at=row["updated_at"],
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(slots=True)
class RuntimeProfile:
    profile_key: str
    signature: str
    hardware_key: str
    backend_name: str
    resolved_batch_size: int
    strategy: str
    startup_seconds: float | None = None
    epoch_1_seconds: float | None = None
    steps_per_epoch: int | None = None
    avg_step_time_ms: float | None = None
    estimated_total_runtime_seconds: float | None = None
    confidence: float = 0.0
    observations: int = 0
    last_job_id: str | None = None
    updated_at: str = field(default_factory=utc_now)
    source: str = "probe"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        signature: str,
        hardware_key: str,
        backend_name: str,
        resolved_batch_size: int,
        strategy: str,
        **kwargs: Any,
    ) -> "RuntimeProfile":
        normalized_strategy = normalize_runtime_probe_strategy(strategy)
        return cls(
            profile_key=build_runtime_profile_key(
                signature,
                hardware_key,
                backend_name,
                resolved_batch_size,
                normalized_strategy,
            ),
            signature=signature,
            hardware_key=hardware_key,
            backend_name=backend_name,
            resolved_batch_size=int(resolved_batch_size),
            strategy=normalized_strategy,
            **kwargs,
        )

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "RuntimeProfile":
        metadata = json.loads(row["metadata_json"]) if row.get("metadata_json") else {}
        return cls(
            profile_key=row["profile_key"],
            signature=row["signature"],
            hardware_key=row["hardware_key"],
            backend_name=row["backend_name"],
            resolved_batch_size=row["resolved_batch_size"],
            strategy=normalize_runtime_probe_strategy(row["strategy"]),
            startup_seconds=row["startup_seconds"],
            epoch_1_seconds=row["epoch_1_seconds"],
            steps_per_epoch=row["steps_per_epoch"],
            avg_step_time_ms=row["avg_step_time_ms"],
            estimated_total_runtime_seconds=row["estimated_total_runtime_seconds"],
            confidence=row["confidence"],
            observations=row["observations"],
            last_job_id=row["last_job_id"],
            updated_at=row["updated_at"],
            source=row.get("source") or "probe",
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(slots=True)
class RunProfile:
    run_profile_id: str
    run_kind: str
    status: str
    job_id: str | None = None
    probe_kind: str | None = None
    backend_name: str | None = None
    toolkit_name: str | None = None
    toolkit_version: str | None = None
    model_key: str | None = None
    signature: str | None = None
    shape_signature: str | None = None
    resolved_batch_size: int | None = None
    epochs: int | None = None
    steps_per_epoch: int | None = None
    total_steps: int | None = None
    startup_seconds: float | None = None
    epoch_time_seconds: float | None = None
    avg_step_time_ms: float | None = None
    estimated_total_runtime_seconds: float | None = None
    avg_sm_utilization: float | None = None
    avg_gpu_utilization: float | None = None
    avg_ram_utilization: float | None = None
    avg_memory_utilization: float | None = None
    peak_vram_mb: int | None = None
    memory_total_mb: int | None = None
    target_budget_mb: int | None = None
    slowdown_ratio: float | None = None
    confidence: float | None = None
    observation_count: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    updated_at: str = field(default_factory=utc_now)
    source: str = "scheduler"
    summary_text: str | None = None
    result_payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "RunProfile":
        metadata = json.loads(row["metadata_json"]) if row.get("metadata_json") else {}
        result_payload = json.loads(row["result_payload_json"]) if row.get("result_payload_json") else {}
        return cls(
            run_profile_id=row["run_profile_id"],
            run_kind=row["run_kind"],
            status=row["status"],
            job_id=row.get("job_id"),
            probe_kind=row.get("probe_kind"),
            backend_name=row.get("backend_name"),
            toolkit_name=row.get("toolkit_name"),
            toolkit_version=row.get("toolkit_version"),
            model_key=row.get("model_key"),
            signature=row.get("signature"),
            shape_signature=row.get("shape_signature"),
            resolved_batch_size=row.get("resolved_batch_size"),
            epochs=row.get("epochs"),
            steps_per_epoch=row.get("steps_per_epoch"),
            total_steps=row.get("total_steps"),
            startup_seconds=row.get("startup_seconds"),
            epoch_time_seconds=row.get("epoch_time_seconds"),
            avg_step_time_ms=row.get("avg_step_time_ms"),
            estimated_total_runtime_seconds=row.get("estimated_total_runtime_seconds"),
            avg_sm_utilization=row.get("avg_sm_utilization"),
            avg_gpu_utilization=row.get("avg_gpu_utilization"),
            avg_ram_utilization=row.get("avg_ram_utilization"),
            avg_memory_utilization=row.get("avg_memory_utilization"),
            peak_vram_mb=row.get("peak_vram_mb"),
            memory_total_mb=row.get("memory_total_mb"),
            target_budget_mb=row.get("target_budget_mb"),
            slowdown_ratio=row.get("slowdown_ratio"),
            confidence=row.get("confidence"),
            observation_count=row.get("observation_count"),
            started_at=row.get("started_at"),
            finished_at=row.get("finished_at"),
            updated_at=row.get("updated_at") or utc_now(),
            source=row.get("source") or "scheduler",
            summary_text=row.get("summary_text"),
            result_payload=result_payload,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(slots=True)
class JobCommand:
    command_id: int
    command_type: CommandType
    created_at: str
    job_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    processed_at: str | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "JobCommand":
        payload = json.loads(row["payload_json"]) if row["payload_json"] else {}
        return cls(
            command_id=row["command_id"],
            command_type=CommandType(row["command_type"]),
            created_at=row["created_at"],
            job_id=row["job_id"],
            payload=payload,
            processed_at=row["processed_at"],
        )

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(slots=True)
class PlacementDecision:
    can_run: bool
    reason: str = ""
    gpu_slot: int = 0
    group_id: str | None = None
    mode: str = "exclusive"
    backend_name: str = "exclusive"
    job_ids: list[str] = field(default_factory=list)
    batch_overrides: dict[str, int] = field(default_factory=dict)
    fallback_order: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SchedulerReport:
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    cancelled_jobs: int = 0
    average_queue_wait_seconds: float = 0.0
    average_runtime_seconds: float = 0.0
    cache_hit_rate: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_evictions: int = 0
    packed_dispatches: int = 0
    packed_fallbacks: int = 0

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


DispatchResult = PlacementDecision
