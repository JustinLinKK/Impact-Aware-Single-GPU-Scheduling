"""Standalone hardware-knowledge configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import sys

from ..redis_cache import RedisCacheSettings


@dataclass(slots=True)
class HardwareKnowledgeGraphSettings:
    enabled: bool = True
    provider: str = "neo4j"
    uri: str = "bolt://127.0.0.1:7688"
    username: str = "neo4j"
    password_env: str = "LOCALML_SCHEDULER_HARDWARE_NEO4J_PASSWORD"
    database: str = "neo4j"

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.provider = str(self.provider or "neo4j").strip().lower().replace("-", "_")
        self.uri = str(self.uri or "").strip()
        self.username = str(self.username or "").strip()
        self.password_env = str(self.password_env or "").strip()
        self.database = str(self.database or "").strip()

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "HardwareKnowledgeGraphSettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "uri": self.uri,
            "username": self.username,
            "password_env": self.password_env,
            "database": self.database,
        }


@dataclass(slots=True)
class _HardwareGpuMemorySettings:
    vram_budget_fraction: float = 0.95

    def budget_mb(self, total_vram_mb: int | float | None) -> float:
        try:
            total = float(total_vram_mb) if total_vram_mb is not None else 0.0
        except (TypeError, ValueError):
            total = 0.0
        return max(0.0, total * float(self.vram_budget_fraction))


@dataclass(slots=True)
class _HardwareGpuSettings:
    device_index: int = 0
    mode: str = "auto"
    max_packed_jobs_per_gpu: int = 0
    backend_priority: list[str] = field(default_factory=list)
    memory: _HardwareGpuMemorySettings = field(default_factory=_HardwareGpuMemorySettings)


@dataclass(slots=True)
class HardwareKnowledgeSettings:
    runtime_root: Path = Path("localml_scheduler/runtime/hardware_knowledge")
    graph: HardwareKnowledgeGraphSettings | dict[str, Any] = field(default_factory=HardwareKnowledgeGraphSettings)
    redis_cache: RedisCacheSettings | dict[str, Any] = field(default_factory=RedisCacheSettings)
    branch_profile_db_path: Path | str | None = None
    device_index: int = 0
    sqlite_busy_timeout_ms: int = 10_000
    python_executable: str = field(default_factory=lambda: sys.executable)

    db_dir: Path = field(init=False)
    db_path: Path = field(init=False)
    gpu_scheduler: _HardwareGpuSettings = field(init=False)

    def __post_init__(self) -> None:
        if self.graph is None:
            self.graph = HardwareKnowledgeGraphSettings()
        if isinstance(self.graph, dict):
            self.graph = HardwareKnowledgeGraphSettings.from_dict(self.graph)
        if self.redis_cache is None:
            self.redis_cache = RedisCacheSettings()
        if isinstance(self.redis_cache, dict):
            self.redis_cache = RedisCacheSettings.from_dict(self.redis_cache)
        self.runtime_root = Path(self.runtime_root).resolve()
        self.db_dir = self.runtime_root / "db"
        self.db_path = self.db_dir / "hardware_knowledge.sqlite3"
        if self.branch_profile_db_path is not None:
            self.branch_profile_db_path = Path(self.branch_profile_db_path).expanduser().resolve()
        else:
            self.branch_profile_db_path = self.db_dir / "branch_profile.sqlite3"
        self.gpu_scheduler = _HardwareGpuSettings(device_index=int(self.device_index or 0))

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None, **overrides: Any) -> "HardwareKnowledgeSettings":
        data = dict(payload or {})
        data.update(overrides)
        return cls(**data)

    def ensure_runtime_layout(self) -> None:
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        Path(self.branch_profile_db_path).parent.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_root": str(self.runtime_root),
            "graph": self.graph.to_dict(),
            "redis_cache": self.redis_cache.to_dict(),
            "branch_profile_db_path": str(self.branch_profile_db_path),
            "device_index": self.device_index,
            "sqlite_busy_timeout_ms": self.sqlite_busy_timeout_ms,
            "python_executable": self.python_executable,
        }
