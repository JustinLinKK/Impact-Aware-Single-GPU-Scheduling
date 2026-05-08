"""Job progress and placement models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import to_primitive, utc_now


@dataclass(slots=True)
class JobProgress:
    job_id: str
    epoch: int = 0
    global_step: int = 0
    phase: str = "train"
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: str | None = None
    last_safe_point: str | None = None
    heartbeat_at: str = field(default_factory=utc_now)
    message: str | None = None
    steps_per_epoch: int | None = None
    avg_step_time_ms: float | None = None
    estimated_total_runtime_seconds: float | None = None
    remaining_runtime_seconds: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "JobProgress | None":
        if payload is None:
            return None
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(slots=True)
class PlacementAssignment:
    mode: str = "exclusive"
    backend_name: str = "exclusive"
    role: str = "solo"
    group_id: str | None = None
    batch_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


ProgressSnapshot = JobProgress

