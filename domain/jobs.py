"""Job-related domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import json
import re

from ..backend_mode import (
    is_retired_backend,
    normalize_backend_allowlist,
    normalize_runtime_backend,
    stream_removal_message,
)
from .common import parse_timestamp, stable_job_id, to_primitive, utc_now


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


class SchedulingClass(str, Enum):
    NORMAL = "normal"
    EXCLUSIVE_PROBE = "exclusive_probe"


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


RUNTIME_PROBE_STRATEGY_EPOCH_1 = "epoch_1"
RUNTIME_PROBE_STRATEGY_STEP_WINDOW = "step_window"


def normalize_runtime_probe_strategy(value: str | None) -> str:
    normalized = str(value or RUNTIME_PROBE_STRATEGY_EPOCH_1).strip().lower().replace("-", "_")
    if normalized in {"epoch_1", "epoch1", "epoch"}:
        return RUNTIME_PROBE_STRATEGY_EPOCH_1
    if normalized in {"step_window", "stepwindow", "steps"}:
        return RUNTIME_PROBE_STRATEGY_STEP_WINDOW
    raise ValueError(f"Unsupported runtime probe strategy: {value}")


@dataclass(slots=True)
class ResourceRequirements:
    requires_gpu: bool = True
    estimated_vram_mb: int | None = None
    estimated_avg_vram_mb: int | None = None
    estimated_ram_mb: int | None = None
    gpu_slots: int = 1

    def __post_init__(self) -> None:
        # Backward compatibility for clients that supplied the old ambiguous field.
        if self.estimated_avg_vram_mb is None and self.estimated_vram_mb is not None:
            self.estimated_avg_vram_mb = int(self.estimated_vram_mb)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ResourceRequirements":
        payload = payload or {}
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(slots=True)
class WorkloadIdentity:
    """Stable task and architecture identity used for placement replay."""

    task_key: str | None = None
    dataset_key: str | None = None
    architecture_key: str | None = None
    architecture_family: str | None = None

    def __post_init__(self) -> None:
        for name in ("task_key", "dataset_key", "architecture_key", "architecture_family"):
            value = getattr(self, name)
            if value is not None:
                normalized = re.sub(r"[^a-z0-9.+-]+", "-", str(value).strip().lower())
                normalized = re.sub(r"-+", "-", normalized).strip("-")
                setattr(self, name, normalized or None)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "WorkloadIdentity":
        return cls(**dict(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)

    @property
    def replay_eligible(self) -> bool:
        return bool(
            (self.task_key or self.dataset_key)
            and (self.architecture_key or self.architecture_family)
        )


@dataclass(slots=True)
class PackingSpec:
    eligible: bool = False
    signature: str | None = None
    family: str | None = None
    max_slowdown_ratio: float | None = None
    backend_allowlist: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.backend_allowlist = normalize_backend_allowlist(self.backend_allowlist)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PackingSpec":
        payload = dict(payload or {})
        backend_allowlist = payload.get("backend_allowlist")
        if backend_allowlist is None:
            payload["backend_allowlist"] = []
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)

    def allows_backend(self, backend_name: str) -> bool:
        if not self.backend_allowlist:
            return True
        return (
            normalize_runtime_backend(backend_name, warn_legacy=False)
            in self.backend_allowlist
        )


@dataclass(slots=True)
class BatchProbeSpec:
    enabled: bool = False
    probe_target: str | None = None
    batch_param_name: str = "batch_size"
    model_key: str | None = None
    shape_hints: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "BatchProbeSpec":
        payload = dict(payload or {})
        if "search_mode" in payload:
            raise ValueError(
                "batch_probe.search_mode was removed; time-aware exclusive probes always measure the configured five batch options"
            )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(slots=True)
class RuntimeProbeSpec:
    enabled: bool = False
    probe_target: str | None = None
    model_key: str | None = None
    strategy: str = RUNTIME_PROBE_STRATEGY_EPOCH_1

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RuntimeProbeSpec":
        payload = dict(payload or {})
        payload["strategy"] = normalize_runtime_probe_strategy(payload.get("strategy"))
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


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
        return to_primitive(self)


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
        return to_primitive(self)


@dataclass(slots=True)
class PreloadSource:
    model_id: str
    model_path: str
    loader_target: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PreloadSource | None":
        if payload is None:
            return None
        return cls(**dict(payload))

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(slots=True)
class JobSpec:
    job_id: str
    agent_id: str | None = None
    workflow_id: str | None = None
    baseline_model_id: str = ""
    baseline_model_path: str = ""
    task_type: str = "generic"
    workload_identity: WorkloadIdentity = field(default_factory=WorkloadIdentity)
    priority: int = 0
    scheduling_class: SchedulingClass = SchedulingClass.NORMAL
    requested_batch_size: int | None = None
    config: JobConfig = field(default_factory=lambda: JobConfig(runner_target=""))
    resource_requirements: ResourceRequirements = field(default_factory=ResourceRequirements)
    packing: PackingSpec = field(default_factory=PackingSpec)
    batch_probe: BatchProbeSpec = field(default_factory=BatchProbeSpec)
    runtime_probe: RuntimeProbeSpec = field(default_factory=RuntimeProbeSpec)
    checkpoint_policy: CheckpointPolicy = field(default_factory=CheckpointPolicy)
    max_steps: int | None = None
    max_epochs: int | None = None
    resume_from_checkpoint: str | None = None
    preload_source: PreloadSource | None = None

    def __post_init__(self) -> None:
        if self.workload_identity is None:
            self.workload_identity = WorkloadIdentity()
        elif isinstance(self.workload_identity, dict):
            self.workload_identity = WorkloadIdentity.from_dict(self.workload_identity)

    @classmethod
    def from_training_job(cls, job: "TrainingJob") -> "JobSpec":
        return cls(
            job_id=job.job_id,
            agent_id=job.agent_id,
            workflow_id=job.workflow_id,
            baseline_model_id=job.baseline_model_id,
            baseline_model_path=job.baseline_model_path,
            task_type=job.task_type,
            workload_identity=job.workload_identity,
            priority=job.priority,
            scheduling_class=job.scheduling_class,
            requested_batch_size=job.requested_batch_size,
            config=job.config,
            resource_requirements=job.resource_requirements,
            packing=job.packing,
            batch_probe=job.batch_probe,
            runtime_probe=job.runtime_probe,
            checkpoint_policy=job.checkpoint_policy,
            max_steps=job.max_steps,
            max_epochs=job.max_epochs,
            resume_from_checkpoint=job.resume_from_checkpoint,
            preload_source=job.preload_source,
        )

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(slots=True)
class JobRun:
    status: JobStatus = JobStatus.PENDING
    submitted_at: str = field(default_factory=utc_now)
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
    def from_training_job(cls, job: "TrainingJob") -> "JobRun":
        return cls(
            status=job.status,
            submitted_at=job.submitted_at,
            metadata=dict(job.metadata),
            queue_sequence=job.queue_sequence,
            status_reason=job.status_reason,
            latest_checkpoint_path=job.latest_checkpoint_path,
            status_timestamps=dict(job.status_timestamps),
            last_heartbeat_at=job.last_heartbeat_at,
            last_dispatched_at=job.last_dispatched_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            hold=job.hold,
        )

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(slots=True)
class TrainingJob:
    job_id: str
    agent_id: str | None = None
    workflow_id: str | None = None
    baseline_model_id: str = ""
    baseline_model_path: str = ""
    task_type: str = "generic"
    workload_identity: WorkloadIdentity = field(default_factory=WorkloadIdentity)
    priority: int = 0
    scheduling_class: SchedulingClass = SchedulingClass.NORMAL
    requested_batch_size: int | None = None
    status: JobStatus = JobStatus.PENDING
    submitted_at: str = field(default_factory=utc_now)
    config: JobConfig = field(default_factory=lambda: JobConfig(runner_target=""))
    resource_requirements: ResourceRequirements = field(default_factory=ResourceRequirements)
    packing: PackingSpec = field(default_factory=PackingSpec)
    batch_probe: BatchProbeSpec = field(default_factory=BatchProbeSpec)
    runtime_probe: RuntimeProbeSpec = field(default_factory=RuntimeProbeSpec)
    checkpoint_policy: CheckpointPolicy = field(default_factory=CheckpointPolicy)
    max_steps: int | None = None
    max_epochs: int | None = None
    resume_from_checkpoint: str | None = None
    preload_source: PreloadSource | None = None
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

    def __post_init__(self) -> None:
        if self.workload_identity is None:
            self.workload_identity = WorkloadIdentity()
        elif isinstance(self.workload_identity, dict):
            self.workload_identity = WorkloadIdentity.from_dict(self.workload_identity)

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
        workload_identity: WorkloadIdentity | None = None,
        priority: int = 0,
        scheduling_class: SchedulingClass | str = SchedulingClass.NORMAL,
        runner_kwargs: dict[str, Any] | None = None,
        loader_target: str | None = None,
        resource_requirements: ResourceRequirements | None = None,
        packing: PackingSpec | None = None,
        batch_probe: BatchProbeSpec | None = None,
        runtime_probe: RuntimeProbeSpec | None = None,
        checkpoint_policy: CheckpointPolicy | None = None,
        max_steps: int | None = None,
        max_epochs: int | None = None,
        resume_from_checkpoint: str | None = None,
        preload_source: PreloadSource | None = None,
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
        batch_param_name = (batch_probe.batch_param_name if batch_probe is not None else "batch_size") or "batch_size"
        raw_requested_batch = config.runner_kwargs.get(batch_param_name)
        try:
            requested_batch_size = 1 if raw_requested_batch is None else max(1, int(raw_requested_batch))
        except (TypeError, ValueError):
            requested_batch_size = 1
        job = cls(
            job_id=stable_job_id(job_id),
            agent_id=agent_id,
            workflow_id=workflow_id,
            baseline_model_id=baseline_model_id,
            baseline_model_path=baseline_model_path,
            task_type=task_type,
            workload_identity=workload_identity or WorkloadIdentity(),
            priority=priority,
            scheduling_class=SchedulingClass(scheduling_class),
            requested_batch_size=requested_batch_size,
            status=JobStatus.PENDING,
            config=config,
            resource_requirements=resource_requirements or ResourceRequirements(),
            packing=packing or PackingSpec(),
            batch_probe=batch_probe or BatchProbeSpec(),
            runtime_probe=runtime_probe or RuntimeProbeSpec(),
            checkpoint_policy=checkpoint_policy or CheckpointPolicy(),
            max_steps=max_steps,
            max_epochs=max_epochs,
            resume_from_checkpoint=resume_from_checkpoint,
            preload_source=preload_source,
            metadata=metadata or {},
        )
        job.mark_status(JobStatus.PENDING)
        return job

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        historical_read: bool = False,
    ) -> "TrainingJob":
        payload = dict(payload)
        raw_packing = dict(payload.get("packing") or {})
        metadata = dict(payload.get("metadata") or {})
        backend_values = [
            *(raw_packing.get("backend_allowlist") or []),
            metadata.get("placement_backend"),
            metadata.get("effective_backend"),
        ]
        retired_backends = sorted(
            {
                str(value)
                for value in backend_values
                if value and is_retired_backend(value)
            }
        )
        if retired_backends:
            if not historical_read:
                raise ValueError(stream_removal_message())
            raw_packing["eligible"] = False
            raw_packing["backend_allowlist"] = []
            payload["packing"] = raw_packing
            metadata.update(
                {
                    "historical_backend_identifiers": retired_backends,
                    "retired_backend_record": True,
                    "selectable": False,
                }
            )
            payload["metadata"] = metadata
            payload["hold"] = True
            payload["status_reason"] = (
                payload.get("status_reason")
                or "historical job uses a retired backend and is non-selectable"
            )
        payload["status"] = JobStatus(payload.get("status", JobStatus.PENDING.value))
        payload["scheduling_class"] = SchedulingClass(payload.get("scheduling_class", SchedulingClass.NORMAL.value))
        payload["config"] = JobConfig.from_dict(payload["config"])
        payload["resource_requirements"] = ResourceRequirements.from_dict(payload.get("resource_requirements"))
        payload["workload_identity"] = WorkloadIdentity.from_dict(payload.get("workload_identity"))
        payload["packing"] = PackingSpec.from_dict(payload.get("packing"))
        payload["batch_probe"] = BatchProbeSpec.from_dict(payload.get("batch_probe"))
        payload["runtime_probe"] = RuntimeProbeSpec.from_dict(payload.get("runtime_probe"))
        payload["checkpoint_policy"] = CheckpointPolicy.from_dict(payload.get("checkpoint_policy"))
        payload["preload_source"] = PreloadSource.from_dict(payload.get("preload_source"))
        if payload.get("requested_batch_size") is None:
            batch_param_name = payload["batch_probe"].batch_param_name or "batch_size"
            raw_requested = payload["config"].runner_kwargs.get(batch_param_name)
            try:
                payload["requested_batch_size"] = max(1, int(raw_requested))
            except (TypeError, ValueError):
                payload["requested_batch_size"] = 1
        return cls(**payload)

    @classmethod
    def from_parts(cls, spec: JobSpec, run: JobRun) -> "TrainingJob":
        return cls(
            job_id=spec.job_id,
            agent_id=spec.agent_id,
            workflow_id=spec.workflow_id,
            baseline_model_id=spec.baseline_model_id,
            baseline_model_path=spec.baseline_model_path,
            task_type=spec.task_type,
            workload_identity=spec.workload_identity,
            priority=spec.priority,
            scheduling_class=spec.scheduling_class,
            requested_batch_size=spec.requested_batch_size,
            status=run.status,
            submitted_at=run.submitted_at,
            config=spec.config,
            resource_requirements=spec.resource_requirements,
            packing=spec.packing,
            batch_probe=spec.batch_probe,
            runtime_probe=spec.runtime_probe,
            checkpoint_policy=spec.checkpoint_policy,
            max_steps=spec.max_steps,
            max_epochs=spec.max_epochs,
            resume_from_checkpoint=spec.resume_from_checkpoint,
            preload_source=spec.preload_source,
            metadata=run.metadata,
            queue_sequence=run.queue_sequence,
            status_reason=run.status_reason,
            latest_checkpoint_path=run.latest_checkpoint_path,
            status_timestamps=run.status_timestamps,
            last_heartbeat_at=run.last_heartbeat_at,
            last_dispatched_at=run.last_dispatched_at,
            started_at=run.started_at,
            finished_at=run.finished_at,
            hold=run.hold,
        )

    def to_job_spec(self) -> JobSpec:
        return JobSpec.from_training_job(self)

    def to_job_run(self) -> JobRun:
        return JobRun.from_training_job(self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def copy(self, **updates: Any) -> "TrainingJob":
        payload = self.to_dict()
        payload.update(to_primitive(updates))
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

    def effective_priority(self) -> int:
        return int(self.priority)

    def age_seconds(self, *, now: str | None = None) -> float:
        reference = parse_timestamp(now or utc_now())
        waiting_since = parse_timestamp(self.waiting_since())
        if reference is None or waiting_since is None:
            return 0.0
        return max(0.0, (reference - waiting_since).total_seconds())
