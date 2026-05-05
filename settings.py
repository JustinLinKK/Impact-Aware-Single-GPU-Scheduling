"""Runtime configuration for the local ML scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import sys

import yaml


SCHEDULER_MODE_SERIAL_BASIC = "serial_basic"
SCHEDULER_MODE_SERIAL_BATCH_OPTIMIZED = "serial_batch_optimized"
SCHEDULER_MODE_PARALLEL_DEFAULT = "parallel_default"
SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED = "parallel_batch_optimized"


def normalize_scheduler_mode(value: str | None) -> str:
    normalized = str(value or SCHEDULER_MODE_PARALLEL_DEFAULT).strip().lower().replace("-", "_")
    allowed = {
        SCHEDULER_MODE_SERIAL_BASIC,
        SCHEDULER_MODE_SERIAL_BATCH_OPTIMIZED,
        SCHEDULER_MODE_PARALLEL_DEFAULT,
        SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED,
    }
    if normalized not in allowed:
        raise ValueError(f"Unsupported scheduler mode: {value}")
    return normalized


@dataclass(slots=True)
class GpuProfilingSettings:
    warmup_steps: int = 30
    solo_probe_steps: int = 80
    pair_probe_steps: int = 60
    reuse_profile_if_confidence_ge: float = 0.8

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GpuProfilingSettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "warmup_steps": self.warmup_steps,
            "solo_probe_steps": self.solo_probe_steps,
            "pair_probe_steps": self.pair_probe_steps,
            "reuse_profile_if_confidence_ge": self.reuse_profile_if_confidence_ge,
        }


@dataclass(slots=True)
class GpuMemorySettings:
    safe_vram_budget_gib: float = 28.0
    hard_stop_memory_fraction: float = 0.90

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GpuMemorySettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "safe_vram_budget_gib": self.safe_vram_budget_gib,
            "hard_stop_memory_fraction": self.hard_stop_memory_fraction,
        }


@dataclass(slots=True)
class GpuThresholdSettings:
    pack_prefer_sm_active_lt: float = 0.50
    pack_reject_sm_active_ge: float = 0.80
    pack_reject_max_slowdown: float = 1.30
    latency_sensitive_max_slowdown: float = 1.15
    min_aggregate_gain: float = 1.10

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GpuThresholdSettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_prefer_sm_active_lt": self.pack_prefer_sm_active_lt,
            "pack_reject_sm_active_ge": self.pack_reject_sm_active_ge,
            "pack_reject_max_slowdown": self.pack_reject_max_slowdown,
            "latency_sensitive_max_slowdown": self.latency_sensitive_max_slowdown,
            "min_aggregate_gain": self.min_aggregate_gain,
        }


@dataclass(slots=True)
class GpuTelemetrySettings:
    device_poll_ms: int = 500
    pair_recheck_every_steps: int = 20

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GpuTelemetrySettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_poll_ms": self.device_poll_ms,
            "pair_recheck_every_steps": self.pair_recheck_every_steps,
        }


@dataclass(slots=True)
class MPSSettings:
    enabled: bool = True
    compute_mode: str = "EXCLUSIVE_PROCESS"
    default_primary_active_thread_pct: int = 60
    default_secondary_active_thread_pct: int = 40
    default_omp_num_threads: int = 6
    default_mkl_num_threads: int = 6
    pipe_directory: str = "/tmp/nvidia-mps"
    log_directory: str = "/tmp/nvidia-mps-log"

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MPSSettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "compute_mode": self.compute_mode,
            "default_primary_active_thread_pct": self.default_primary_active_thread_pct,
            "default_secondary_active_thread_pct": self.default_secondary_active_thread_pct,
            "default_omp_num_threads": self.default_omp_num_threads,
            "default_mkl_num_threads": self.default_mkl_num_threads,
            "pipe_directory": self.pipe_directory,
            "log_directory": self.log_directory,
        }


@dataclass(slots=True)
class CudaProcessSettings:
    enabled: bool = True
    default_omp_num_threads: int = 6
    default_mkl_num_threads: int = 6

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CudaProcessSettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "default_omp_num_threads": self.default_omp_num_threads,
            "default_mkl_num_threads": self.default_mkl_num_threads,
        }


@dataclass(slots=True)
class StreamSettings:
    enabled: bool = False
    host_poll_interval_seconds: float = 0.1
    host_join_timeout_seconds: float = 3.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "StreamSettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "host_poll_interval_seconds": self.host_poll_interval_seconds,
            "host_join_timeout_seconds": self.host_join_timeout_seconds,
        }


@dataclass(slots=True)
class ParallelOptimizerSettings:
    batch_search_mode: str = "binary"
    target_vram_fraction: float = 0.97
    max_probe_jobs: int = 3
    max_batch_multiplier: int = 32
    min_batch_size: int = 1

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ParallelOptimizerSettings":
        instance = cls(**(payload or {}))
        instance.batch_search_mode = instance.batch_search_mode or "binary"
        return instance

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_search_mode": self.batch_search_mode,
            "target_vram_fraction": self.target_vram_fraction,
            "max_probe_jobs": self.max_probe_jobs,
            "max_batch_multiplier": self.max_batch_multiplier,
            "min_batch_size": self.min_batch_size,
        }


@dataclass(slots=True)
class SchedulerSubmissionDefaults:
    requires_gpu: bool = True
    estimated_vram_mb: int | None = None
    estimated_ram_mb: int | None = None
    packing_eligible: bool = False
    packing_family: str = "mlevolve_script"
    packing_max_slowdown_ratio: float | None = None
    backend_allowlist: list[str] = field(default_factory=lambda: ["mps", "cuda_process"])
    batch_probe_enabled: bool = True
    batch_probe_model_key: str | None = None
    batch_probe_probe_timeout_seconds: int = 45
    batch_probe_poll_interval_seconds: float = 0.5
    batch_probe_max_multiplier: int = 32
    batch_probe_search_mode: str = "binary"

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SchedulerSubmissionDefaults":
        instance = cls(**(payload or {}))
        if instance.backend_allowlist is None:
            instance.backend_allowlist = ["mps", "cuda_process"]
        else:
            instance.backend_allowlist = [str(item) for item in instance.backend_allowlist]
        instance.batch_probe_search_mode = instance.batch_probe_search_mode or "binary"
        return instance

    def to_dict(self) -> dict[str, Any]:
        return {
            "requires_gpu": self.requires_gpu,
            "estimated_vram_mb": self.estimated_vram_mb,
            "estimated_ram_mb": self.estimated_ram_mb,
            "packing_eligible": self.packing_eligible,
            "packing_family": self.packing_family,
            "packing_max_slowdown_ratio": self.packing_max_slowdown_ratio,
            "backend_allowlist": list(self.backend_allowlist),
            "batch_probe_enabled": self.batch_probe_enabled,
            "batch_probe_model_key": self.batch_probe_model_key,
            "batch_probe_probe_timeout_seconds": self.batch_probe_probe_timeout_seconds,
            "batch_probe_poll_interval_seconds": self.batch_probe_poll_interval_seconds,
            "batch_probe_max_multiplier": self.batch_probe_max_multiplier,
            "batch_probe_search_mode": self.batch_probe_search_mode,
        }


@dataclass(slots=True)
class GpuSchedulerSettings:
    enabled: bool = True
    mode: str = SCHEDULER_MODE_PARALLEL_DEFAULT
    backend_priority: list[str] = field(default_factory=lambda: ["mps", "stream", "cuda_process", "exclusive"])
    max_packed_jobs_per_gpu: int = 2
    allow_three_way_packing: bool = False
    candidate_window_size: int = 8
    device_index: int = 0
    fallback_cooldown_seconds: int = 900
    batch_probe_enabled: bool = True
    batch_probe_target_memory_fraction: float = 0.97
    batch_probe_min_batch_size: int = 1
    batch_probe_max_search_rounds: int = 12
    batch_probe_max_batch_size: int | None = None
    batch_probe_search_mode: str = "binary"
    profiling: GpuProfilingSettings = field(default_factory=GpuProfilingSettings)
    memory: GpuMemorySettings = field(default_factory=GpuMemorySettings)
    thresholds: GpuThresholdSettings = field(default_factory=GpuThresholdSettings)
    telemetry: GpuTelemetrySettings = field(default_factory=GpuTelemetrySettings)
    parallel_optimizer: ParallelOptimizerSettings = field(default_factory=ParallelOptimizerSettings)
    submission_defaults: SchedulerSubmissionDefaults = field(default_factory=SchedulerSubmissionDefaults)
    mps: MPSSettings = field(default_factory=MPSSettings)
    cuda_process: CudaProcessSettings = field(default_factory=CudaProcessSettings)
    stream: StreamSettings = field(default_factory=StreamSettings)

    def __post_init__(self) -> None:
        self.mode = normalize_scheduler_mode(self.mode)
        if self.backend_priority is None:
            self.backend_priority = ["mps", "stream", "cuda_process", "exclusive"]
        else:
            self.backend_priority = [str(item) for item in self.backend_priority]
        if self.profiling is None:
            self.profiling = GpuProfilingSettings()
        if isinstance(self.profiling, dict):
            self.profiling = GpuProfilingSettings.from_dict(self.profiling)
        if self.memory is None:
            self.memory = GpuMemorySettings()
        if isinstance(self.memory, dict):
            self.memory = GpuMemorySettings.from_dict(self.memory)
        if self.thresholds is None:
            self.thresholds = GpuThresholdSettings()
        if isinstance(self.thresholds, dict):
            self.thresholds = GpuThresholdSettings.from_dict(self.thresholds)
        if self.telemetry is None:
            self.telemetry = GpuTelemetrySettings()
        if isinstance(self.telemetry, dict):
            self.telemetry = GpuTelemetrySettings.from_dict(self.telemetry)
        if self.parallel_optimizer is None:
            self.parallel_optimizer = ParallelOptimizerSettings()
        if isinstance(self.parallel_optimizer, dict):
            self.parallel_optimizer = ParallelOptimizerSettings.from_dict(self.parallel_optimizer)
        if self.submission_defaults is None:
            self.submission_defaults = SchedulerSubmissionDefaults()
        if isinstance(self.submission_defaults, dict):
            self.submission_defaults = SchedulerSubmissionDefaults.from_dict(self.submission_defaults)
        if self.mps is None:
            self.mps = MPSSettings()
        if isinstance(self.mps, dict):
            self.mps = MPSSettings.from_dict(self.mps)
        if self.cuda_process is None:
            self.cuda_process = CudaProcessSettings()
        if isinstance(self.cuda_process, dict):
            self.cuda_process = CudaProcessSettings.from_dict(self.cuda_process)
        if self.stream is None:
            self.stream = StreamSettings()
        if isinstance(self.stream, dict):
            self.stream = StreamSettings.from_dict(self.stream)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GpuSchedulerSettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "backend_priority": list(self.backend_priority),
            "max_packed_jobs_per_gpu": self.max_packed_jobs_per_gpu,
            "allow_three_way_packing": self.allow_three_way_packing,
            "candidate_window_size": self.candidate_window_size,
            "device_index": self.device_index,
            "fallback_cooldown_seconds": self.fallback_cooldown_seconds,
            "batch_probe_enabled": self.batch_probe_enabled,
            "batch_probe_target_memory_fraction": self.batch_probe_target_memory_fraction,
            "batch_probe_min_batch_size": self.batch_probe_min_batch_size,
            "batch_probe_max_search_rounds": self.batch_probe_max_search_rounds,
            "batch_probe_max_batch_size": self.batch_probe_max_batch_size,
            "batch_probe_search_mode": self.batch_probe_search_mode,
            "profiling": self.profiling.to_dict(),
            "memory": self.memory.to_dict(),
            "thresholds": self.thresholds.to_dict(),
            "telemetry": self.telemetry.to_dict(),
            "parallel_optimizer": self.parallel_optimizer.to_dict(),
            "submission_defaults": self.submission_defaults.to_dict(),
            "mps": self.mps.to_dict(),
            "cuda_process": self.cuda_process.to_dict(),
            "stream": self.stream.to_dict(),
        }


@dataclass(slots=True)
class SchedulerSettings:
    runtime_root: Path = Path("localml_scheduler/runtime")
    scheduler_poll_interval_seconds: float = 0.5
    command_poll_limit: int = 100
    aging_interval_seconds: float = 180.0
    aging_priority_increment: int = 1
    enable_priority_aging: bool = True
    preempt_check_interval_seconds: float = 0.5
    eager_preload_top_k: int = 2
    cache_memory_budget_bytes: int = 2 * 1024 * 1024 * 1024
    cache_server_host: str = "127.0.0.1"
    cache_server_port: int = 8765
    cache_socket_name: str = "cache_server.sock"
    auto_resume_recoverable: bool = False
    gpu_scheduler: GpuSchedulerSettings = field(default_factory=GpuSchedulerSettings)
    python_executable: str = field(default_factory=lambda: sys.executable)
    sqlite_busy_timeout_ms: int = 10_000

    db_dir: Path = field(init=False)
    db_path: Path = field(init=False)
    jobs_dir: Path = field(init=False)
    checkpoints_dir: Path = field(init=False)
    cache_meta_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    events_jsonl_path: Path = field(init=False)
    scheduler_log_path: Path = field(init=False)
    cache_socket_path: Path = field(init=False)
    service_heartbeat_path: Path = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.gpu_scheduler, dict):
            self.gpu_scheduler = GpuSchedulerSettings.from_dict(self.gpu_scheduler)
        self.runtime_root = Path(self.runtime_root).resolve()
        self.db_dir = self.runtime_root / "db"
        self.db_path = self.db_dir / "scheduler.sqlite3"
        self.jobs_dir = self.runtime_root / "data" / "jobs"
        self.checkpoints_dir = self.runtime_root / "data" / "checkpoints"
        self.cache_meta_dir = self.runtime_root / "cache_meta"
        self.logs_dir = self.runtime_root / "logs"
        self.events_jsonl_path = self.logs_dir / "events.jsonl"
        self.scheduler_log_path = self.logs_dir / "scheduler.log"
        self.cache_socket_path = self.runtime_root / self.cache_socket_name
        self.service_heartbeat_path = self.runtime_root / "service_heartbeat.json"

    @classmethod
    def from_file(cls, path: str | Path | None = None, **overrides: Any) -> "SchedulerSettings":
        payload: dict[str, Any] = {}
        if path:
            with Path(path).open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
        payload.update(overrides)
        return cls(**payload)

    def ensure_runtime_layout(self) -> None:
        for directory in (
            self.runtime_root,
            self.db_dir,
            self.jobs_dir,
            self.checkpoints_dir,
            self.cache_meta_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def job_runtime_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def job_command_path(self, job_id: str) -> Path:
        return self.job_runtime_dir(job_id) / "command.json"

    def job_heartbeat_path(self, job_id: str) -> Path:
        return self.job_runtime_dir(job_id) / "heartbeat.json"

    def checkpoints_for_job(self, job_id: str) -> Path:
        return self.checkpoints_dir / job_id

    def cache_address(self) -> str | tuple[str, int]:
        if sys.platform != "win32":
            return str(self.cache_socket_path)
        return (self.cache_server_host, self.cache_server_port)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_root": str(self.runtime_root),
            "scheduler_poll_interval_seconds": self.scheduler_poll_interval_seconds,
            "command_poll_limit": self.command_poll_limit,
            "aging_interval_seconds": self.aging_interval_seconds,
            "aging_priority_increment": self.aging_priority_increment,
            "enable_priority_aging": self.enable_priority_aging,
            "preempt_check_interval_seconds": self.preempt_check_interval_seconds,
            "eager_preload_top_k": self.eager_preload_top_k,
            "cache_memory_budget_bytes": self.cache_memory_budget_bytes,
            "cache_server_host": self.cache_server_host,
            "cache_server_port": self.cache_server_port,
            "cache_socket_name": self.cache_socket_name,
            "auto_resume_recoverable": self.auto_resume_recoverable,
            "gpu_scheduler": self.gpu_scheduler.to_dict(),
            "python_executable": self.python_executable,
            "sqlite_busy_timeout_ms": self.sqlite_busy_timeout_ms,
        }
