"""Runtime configuration for the local ML scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import sys

import yaml


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
            "python_executable": self.python_executable,
            "sqlite_busy_timeout_ms": self.sqlite_busy_timeout_ms,
        }
