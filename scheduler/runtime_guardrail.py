"""Runtime-aware packing guardrails."""

from __future__ import annotations

from ..domain import TrainingJob
from ..config import SchedulerSettings
from .planning_repository import PlanningRepository


class RuntimeGuardrail:
    def __init__(self, settings: SchedulerSettings, repository: PlanningRepository):
        self.settings = settings
        self.repository = repository

    def runtime_penalty(self, jobs: list[TrainingJob], *, backend_name: str) -> tuple[float, bool]:
        del backend_name
        estimates: list[float] = []
        missing = 0
        for job in jobs:
            estimate = job.metadata.get("runtime_remaining_runtime_seconds")
            if estimate is None:
                missing += 1
                continue
            try:
                estimates.append(max(0.0, float(estimate)))
            except (TypeError, ValueError):
                missing += 1
        if len(jobs) <= 1:
            return (0.0 if missing == 0 else 0.02 * missing, False)
        if len(estimates) < len(jobs):
            return (0.02 * missing, False)
        runtimes = [item for item in estimates if item > 0]
        if not runtimes:
            return (0.02 * max(1, len(jobs)), False)
        ratio = max(runtimes) / max(1e-9, min(runtimes))
        if ratio > float(self.settings.gpu_scheduler.auto_pack.runtime_skew_guardrail_ratio):
            return (0.0, True)
        return max(0.0, ratio - 1.0) * 0.10, False
