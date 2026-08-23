"""Planner DTOs."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..domain import TrainingJob
from .resource_estimator import BatchOptionEstimate


@dataclass(slots=True)
class DispatchPlan:
    mode: str
    backend_name: str
    job_ids: tuple[str, ...]
    reason: str
    batch_overrides: dict[str, int] = field(default_factory=dict)
    fallback_order: list[str] = field(default_factory=list)
    objective_breakdown: dict[str, object] = field(default_factory=dict)
    trial_metadata: dict[str, object] = field(default_factory=dict)
    backend_config: dict[str, object] = field(default_factory=dict)
    mandatory_anchor_job_id: str | None = None
    objective_version: str | None = None


@dataclass(slots=True)
class NoDispatchReason:
    reason: str


@dataclass(slots=True)
class EvaluatedGroup:
    jobs: list[TrainingJob]
    backend_name: str
    estimated_vram_mb: float
    objective_score: float
    batch_overrides: dict[str, int]
    fallback_order: list[str]
    reason: str
    batch_estimates: dict[str, BatchOptionEstimate] = field(default_factory=dict)
    score_breakdown: dict[str, object] = field(default_factory=dict)
    mandatory_anchor_job_id: str | None = None
    objective_version: str | None = None
