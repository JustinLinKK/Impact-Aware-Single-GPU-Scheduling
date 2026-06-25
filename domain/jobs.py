"""Job-related domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import json
import logging

from .common import parse_timestamp, stable_job_id, to_primitive, utc_now

logger = logging.getLogger("localml_scheduler")


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


BATCH_PROBE_SEARCH_MODE_BINARY = "binary"
BATCH_PROBE_SEARCH_MODE_POWER_OF_TWO = "power_of_two"
RUNTIME_PROBE_STRATEGY_EPOCH_1 = "epoch_1"
RUNTIME_PROBE_STRATEGY_STEP_WINDOW = "step_window"


def normalize_batch_probe_search_mode(value: str | None) -> str:
    normalized = str(value or BATCH_PROBE_SEARCH_MODE_POWER_OF_TWO).strip().lower().replace("-", "_")
    if normalized in {"binary", "default"}:
        logger.warning(
            "Batch probe search mode %r is deprecated; using power_of_two.",
            value,
        )
        return BATCH_PROBE_SEARCH_MODE_POWER_OF_TWO
    if normalized in {"power_of_two", "powers_of_two", "pow2", "2^n", "2n"}:
        return BATCH_PROBE_SEARCH_MODE_POWER_OF_TWO
    raise ValueError(f"Unsupported batch probe search mode: {value}")


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
    estimated_ram_mb: int | None = None
    gpu_slots: int = 1

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ResourceRequirements":
        payload = payload or {}
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(slots=True)
class PackingSpec:
    eligible: bool = False
    signature: str | None = None
    family: str | None = None
    max_slowdown_ratio: float | None = None
    backend_allowlist: list[str] = field(default_factory=list)

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
        return backend_name in self.backend_allowlist


@dataclass(slots=True)
class BatchProbeSpec:
    enabled: bool = False
    probe_target: str | None = None
    batch_param_name: str = "batch_size"
    model_key: str | None = None
    search_mode: str | None = None
    shape_hints: dict[str, Any] = field(default_factory=dict)
    profile_key: str | None = None
    shape_signature_override: str | None = None
    reuse_only: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "BatchProbeSpec":
        payload = dict(payload or {})
        if payload.get("search_mode") is not None:
            payload["search_mode"] = normalize_batch_probe_search_mode(payload.get("search_mode"))
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
    preemptible: bool = True

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
    priority: int = 0
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

    @classmethod
    def from_training_job(cls, job: "TrainingJob") -> "JobSpec":
        return cls(
            job_id=job.job_id,
            agent_id=job.agent_id,
            workflow_id=job.workflow_id,
            baseline_model_id=job.baseline_model_id,
            baseline_model_path=job.baseline_model_path,
            task_type=job.task_type,
            priority=job.priority,
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
    priority: int = 0
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
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingJob":
        payload = dict(payload)
        payload["status"] = JobStatus(payload.get("status", JobStatus.PENDING.value))
        payload["config"] = JobConfig.from_dict(payload["config"])
        payload["resource_requirements"] = ResourceRequirements.from_dict(payload.get("resource_requirements"))
        payload["packing"] = PackingSpec.from_dict(payload.get("packing"))
        payload["batch_probe"] = BatchProbeSpec.from_dict(payload.get("batch_probe"))
        payload["runtime_probe"] = RuntimeProbeSpec.from_dict(payload.get("runtime_probe"))
        payload["checkpoint_policy"] = CheckpointPolicy.from_dict(payload.get("checkpoint_policy"))
        payload["preload_source"] = PreloadSource.from_dict(payload.get("preload_source"))
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
            priority=spec.priority,
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
