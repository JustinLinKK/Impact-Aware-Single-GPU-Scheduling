"""Core schemas for the local ML scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha1
from pathlib import Path
from typing import Any
import importlib
import json
import uuid


def utc_now() -> str:
    """Return an ISO8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def parse_timestamp(value: str | None) -> datetime | None:
    """Parse an ISO8601 timestamp if present."""
    if not value:
        return None
    return datetime.fromisoformat(value)


def stable_job_id(explicit_job_id: str | None = None) -> str:
    """Create a stable job id at submit time."""
    return explicit_job_id or str(uuid.uuid4())


def import_string(target: str) -> Any:
    """Import a callable or object from ``module:attr`` syntax."""
    if ":" not in target:
        raise ValueError(f"Import target must be in module:attr form, got: {target}")
    module_name, attr_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    attr = module
    for part in attr_name.split("."):
        attr = getattr(attr, part)
    return attr


def _to_primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {name: _to_primitive(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(key): _to_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_primitive(item) for item in value]
    return value


class JobStatus(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSING = "PAUSING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    RECOVERABLE = "RECOVERABLE"

    @property
    def is_terminal(self) -> bool:
        return self in {self.COMPLETED, self.FAILED, self.CANCELLED}


class SafePointType(str, Enum):
    MANUAL = "manual"
    STEP = "step"
    EPOCH = "epoch"
    EXPLICIT = "explicit"
    BEFORE_TRAIN = "before_train"


class CommandType(str, Enum):
    SUBMIT = "SUBMIT"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    CANCEL = "CANCEL"
    PRELOAD = "PRELOAD"


@dataclass(slots=True)
class ResourceRequirements:
    requires_gpu: bool = True
    estimated_vram_mb: int | None = None
    estimated_ram_mb: int | None = None
    gpu_slots: int = 1

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ResourceRequirements":
        payload = payload or {}
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return _to_primitive(self)


@dataclass(slots=True)
class PackingSpec:
    eligible: bool = False
    signature: str | None = None
    family: str | None = None
    max_slowdown_ratio: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PackingSpec":
        payload = payload or {}
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return _to_primitive(self)


@dataclass(slots=True)
class BatchProbeSpec:
    enabled: bool = False
    probe_target: str | None = None
    batch_param_name: str = "batch_size"
    model_key: str | None = None
    shape_hints: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "BatchProbeSpec":
        payload = payload or {}
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return _to_primitive(self)


@dataclass(slots=True)
class CheckpointPolicy:
    save_every_n_steps: int | None = None
    save_every_epoch: bool = True
    keep_last_n: int = 3
    pause_mode: SafePointType = SafePointType.STEP

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CheckpointPolicy":
        payload = dict(payload or {})
        pause_mode = payload.get("pause_mode", SafePointType.STEP.value)
        payload["pause_mode"] = SafePointType(pause_mode)
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return _to_primitive(self)


@dataclass(slots=True)
class JobConfig:
    runner_target: str
    runner_kwargs: dict[str, Any] = field(default_factory=dict)
    loader_target: str | None = None
    max_steps: int | None = None
    max_epochs: int | None = None
    seed: int | None = None
    python_executable: str | None = None
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "JobConfig":
        payload = dict(payload)
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return _to_primitive(self)


@dataclass(slots=True)
class TrainingJob:
    job_id: str
    agent_id: str | None = None
    workflow_id: str | None = None
    baseline_model_id: str = ""
    baseline_model_path: str = ""
    task_type: str = "generic"
    priority: int = 0
    status: JobStatus = JobStatus.PENDING
    submitted_at: str = field(default_factory=utc_now)
    config: JobConfig = field(default_factory=lambda: JobConfig(runner_target=""))
    resource_requirements: ResourceRequirements = field(default_factory=ResourceRequirements)
    packing: PackingSpec = field(default_factory=PackingSpec)
    batch_probe: BatchProbeSpec = field(default_factory=BatchProbeSpec)
    checkpoint_policy: CheckpointPolicy = field(default_factory=CheckpointPolicy)
    max_steps: int | None = None
    max_epochs: int | None = None
    resume_from_checkpoint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    queue_sequence: int = 0
    status_reason: str | None = None
    latest_checkpoint_path: str | None = None
    status_timestamps: dict[str, str] = field(default_factory=dict)
    last_heartbeat_at: str | None = None
    last_dispatched_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    hold: bool = False

    @classmethod
    def create(
        cls,
        runner_target: str,
        baseline_model_id: str,
        baseline_model_path: str,
        *,
        job_id: str | None = None,
        agent_id: str | None = None,
        workflow_id: str | None = None,
        task_type: str = "generic",
        priority: int = 0,
        runner_kwargs: dict[str, Any] | None = None,
        loader_target: str | None = None,
        resource_requirements: ResourceRequirements | None = None,
        packing: PackingSpec | None = None,
        batch_probe: BatchProbeSpec | None = None,
        checkpoint_policy: CheckpointPolicy | None = None,
        max_steps: int | None = None,
        max_epochs: int | None = None,
        resume_from_checkpoint: str | None = None,
        metadata: dict[str, Any] | None = None,
        seed: int | None = None,
        python_executable: str | None = None,
        env: dict[str, str] | None = None,
    ) -> "TrainingJob":
        config = JobConfig(
            runner_target=runner_target,
            runner_kwargs=runner_kwargs or {},
            loader_target=loader_target,
            max_steps=max_steps,
            max_epochs=max_epochs,
            seed=seed,
            python_executable=python_executable,
            env=env or {},
        )
        job = cls(
            job_id=stable_job_id(job_id),
            agent_id=agent_id,
            workflow_id=workflow_id,
            baseline_model_id=baseline_model_id,
            baseline_model_path=baseline_model_path,
            task_type=task_type,
            priority=priority,
            status=JobStatus.PENDING,
            config=config,
            resource_requirements=resource_requirements or ResourceRequirements(),
            packing=packing or PackingSpec(),
            batch_probe=batch_probe or BatchProbeSpec(),
            checkpoint_policy=checkpoint_policy or CheckpointPolicy(),
            max_steps=max_steps,
            max_epochs=max_epochs,
            resume_from_checkpoint=resume_from_checkpoint,
            metadata=metadata or {},
        )
        job.mark_status(JobStatus.PENDING)
        return job

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingJob":
        payload = dict(payload)
        payload["status"] = JobStatus(payload.get("status", JobStatus.PENDING.value))
        payload["config"] = JobConfig.from_dict(payload["config"])
        payload["resource_requirements"] = ResourceRequirements.from_dict(payload.get("resource_requirements"))
        payload["packing"] = PackingSpec.from_dict(payload.get("packing"))
        payload["batch_probe"] = BatchProbeSpec.from_dict(payload.get("batch_probe"))
        payload["checkpoint_policy"] = CheckpointPolicy.from_dict(payload.get("checkpoint_policy"))
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return _to_primitive(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def copy(self, **updates: Any) -> "TrainingJob":
        payload = self.to_dict()
        payload.update(_to_primitive(updates))
        return self.from_dict(payload)

    def mark_status(self, status: JobStatus, reason: str | None = None) -> None:
        now = utc_now()
        self.status = status
        self.status_reason = reason
        self.status_timestamps[status.value] = now
        if status == JobStatus.RUNNING:
            self.started_at = self.started_at or now
            self.last_dispatched_at = now
        if status.is_terminal:
            self.finished_at = now

    def is_runnable(self) -> bool:
        return (not self.hold) and self.status in {
            JobStatus.PENDING,
            JobStatus.READY,
            JobStatus.PAUSED,
            JobStatus.RECOVERABLE,
        }

    def waiting_since(self) -> str:
        if self.status == JobStatus.PAUSED:
            return self.status_timestamps.get(JobStatus.PAUSED.value, self.submitted_at)
        return self.submitted_at

    def packing_signature(self) -> str | None:
        return self.packing.signature


@dataclass(slots=True)
class ProgressSnapshot:
    job_id: str
    epoch: int = 0
    global_step: int = 0
    phase: str = "train"
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: str | None = None
    last_safe_point: str | None = None
    heartbeat_at: str = field(default_factory=utc_now)
    message: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ProgressSnapshot | None":
        if payload is None:
            return None
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return _to_primitive(self)


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    entries: int = 0
    pinned_entries: int = 0
    used_bytes: int = 0
    memory_budget_bytes: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CacheStats":
        payload = payload or {}
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return _to_primitive(self)


@dataclass(slots=True)
class SoloProfile:
    signature: str
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
        return _to_primitive(self)


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
        return _to_primitive(self)


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
        return _to_primitive(self)


def build_batch_probe_shape_signature(job: TrainingJob) -> str:
    batch_param_name = job.batch_probe.batch_param_name or "batch_size"
    runner_kwargs = dict(job.config.runner_kwargs)
    runner_kwargs.pop(batch_param_name, None)
    payload = {
        "runner_target": job.config.runner_target,
        "task_type": job.task_type,
        "loader_target": job.config.loader_target,
        "runner_kwargs": runner_kwargs,
        "shape_hints": job.batch_probe.shape_hints,
    }
    return sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_batch_probe_key(model_key: str, device_type: str, shape_signature: str) -> str:
    payload = {
        "device_type": device_type,
        "model_key": model_key,
        "shape_signature": shape_signature,
    }
    return sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def canonical_pair_key(left_signature: str, right_signature: str) -> str:
    ordered = sorted((left_signature, right_signature))
    return f"{ordered[0]}::{ordered[1]}"


@dataclass(slots=True)
class PairProfile:
    pair_key: str
    left_signature: str
    right_signature: str
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
        **kwargs: Any,
    ) -> "PairProfile":
        return cls(
            pair_key=canonical_pair_key(left_signature, right_signature),
            left_signature=left_signature,
            right_signature=right_signature,
            **kwargs,
        )

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "PairProfile":
        metadata = json.loads(row["metadata_json"]) if row.get("metadata_json") else {}
        return cls(
            pair_key=row["pair_key"],
            left_signature=row["left_signature"],
            right_signature=row["right_signature"],
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
        return _to_primitive(self)

    def on_cooldown(self) -> bool:
        cooldown_until = parse_timestamp(self.cooldown_until)
        return cooldown_until is not None and cooldown_until > datetime.now(timezone.utc)


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
        return _to_primitive(self)


@dataclass(slots=True)
class PlacementDecision:
    can_run: bool
    reason: str = ""
    gpu_slot: int = 0
    mode: str = "exclusive"
    backend_name: str = "exclusive"
    job_ids: list[str] = field(default_factory=list)


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
        return _to_primitive(self)
