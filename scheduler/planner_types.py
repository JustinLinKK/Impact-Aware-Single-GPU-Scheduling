"""Planner DTOs."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..domain import TrainingJob


@dataclass(slots=True)
class DispatchPlan:
    mode: str
    backend_name: str
    job_ids: tuple[str, ...]
    reason: str
    batch_overrides: dict[str, int] = field(default_factory=dict)
    fallback_order: list[str] = field(default_factory=list)
    estimated_vram_mb: float = 0.0
    estimated_sm_utilization: float = 0.0
    predicted_throughput: float | None = None
    solver_kind: str = "exact"
    objective_vector: tuple[float, ...] = ()
    active_job_ids: tuple[str, ...] = ()

    @property
    def requires_repack(self) -> bool:
        return bool(self.active_job_ids)


@dataclass(slots=True)
class NoDispatchReason:
    reason: str


@dataclass(slots=True)
class EvaluatedGroup:
    jobs: list[TrainingJob]
    backend_name: str
    estimated_vram_mb: float
    estimated_sm_utilization: float
    objective_score: float
    batch_overrides: dict[str, int]
    fallback_order: list[str]
    reason: str
