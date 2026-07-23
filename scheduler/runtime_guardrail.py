"""Runtime-aware packing guardrails."""

from __future__ import annotations

from ..profiling.runtime_probe import runtime_profile_for_job
from ..domain import TrainingJob
from ..config import SchedulerSettings
from .planning_repository import PlanningRepository


class RuntimeGuardrail:
    def __init__(self, settings: SchedulerSettings, repository: PlanningRepository):
        self.settings = settings
        self.repository = repository

    def runtime_penalty(self, jobs: list[TrainingJob], *, backend_name: str) -> tuple[float, bool]:
        estimates: list[tuple[float, str]] = []
        missing = 0
        for job in jobs:
            profile = runtime_profile_for_job(self.repository, job, backend_name=backend_name)
            if profile is None or profile.estimated_total_runtime_seconds is None:
                missing += 1
                continue
            estimates.append((float(profile.estimated_total_runtime_seconds), str(profile.source or "history")))
        if len(jobs) <= 1:
            return (0.0 if missing == 0 else 0.05 * missing, False)
        if len(estimates) < len(jobs):
            return (0.15 * missing, False)
        runtimes = [item[0] for item in estimates if item[0] > 0]
        if not runtimes:
            return (0.15, False)
        ratio = max(runtimes) / max(1e-9, min(runtimes))
        all_probe = all(source == "probe" for _, source in estimates)
        if all_probe and ratio > float(self.settings.gpu_scheduler.auto_pack.runtime_skew_guardrail_ratio):
            return (0.0, True)
        return max(0.0, ratio - 1.0) * (0.20 if all_probe else 0.10), False

