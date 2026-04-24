"""Runtime configuration for the local ML scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import sys

import yaml


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
        }


@dataclass(slots=True)
class GpuSchedulerSettings:
    enabled: bool = True
    backend_priority: list[str] = field(default_factory=lambda: ["mps", "exclusive"])
    max_packed_jobs_per_gpu: int = 2
    allow_three_way_packing: bool = False
    device_index: int = 0
    fallback_cooldown_seconds: int = 900
    profiling: GpuProfilingSettings = field(default_factory=GpuProfilingSettings)
    memory: GpuMemorySettings = field(default_factory=GpuMemorySettings)
    thresholds: GpuThresholdSettings = field(default_factory=GpuThresholdSettings)
    telemetry: GpuTelemetrySettings = field(default_factory=GpuTelemetrySettings)
    mps: MPSSettings = field(default_factory=MPSSettings)

    def __post_init__(self) -> None:
        if isinstance(self.profiling, dict):
            self.profiling = GpuProfilingSettings.from_dict(self.profiling)
        if isinstance(self.memory, dict):
            self.memory = GpuMemorySettings.from_dict(self.memory)
        if isinstance(self.thresholds, dict):
            self.thresholds = GpuThresholdSettings.from_dict(self.thresholds)
        if isinstance(self.telemetry, dict):
            self.telemetry = GpuTelemetrySettings.from_dict(self.telemetry)
        if isinstance(self.mps, dict):
            self.mps = MPSSettings.from_dict(self.mps)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GpuSchedulerSettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "backend_priority": list(self.backend_priority),
            "max_packed_jobs_per_gpu": self.max_packed_jobs_per_gpu,
            "allow_three_way_packing": self.allow_three_way_packing,
            "device_index": self.device_index,
            "fallback_cooldown_seconds": self.fallback_cooldown_seconds,
            "profiling": self.profiling.to_dict(),
            "memory": self.memory.to_dict(),
            "thresholds": self.thresholds.to_dict(),
            "telemetry": self.telemetry.to_dict(),
            "mps": self.mps.to_dict(),
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
