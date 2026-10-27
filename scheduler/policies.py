"""Scheduling policies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ..domain import TrainingJob


class SchedulingPolicy(Protocol):
    """Interface for scheduling policies."""

    def sort_key(self, job: TrainingJob, *, now: datetime | None = None) -> tuple[int, str]:
        ...

    def choose(self, jobs: list[TrainingJob], *, now: datetime | None = None) -> TrainingJob | None:
        ...

    def should_preempt(self, active_job: TrainingJob, candidate_job: TrainingJob, *, now: datetime | None = None) -> bool:
        ...


@dataclass(slots=True)
class PriorityFifoPolicy:
    """Static priority, with job identity breaking ties independently of arrival.

    The historical class name and aging arguments remain accepted for config/API
    compatibility. They no longer change admission or preemption priority.
    """

    aging_interval_seconds: float = 180.0
    aging_priority_increment: int = 1
    enable_priority_aging: bool = False

    def effective_priority(self, job: TrainingJob, *, now: datetime | None = None) -> int:
        return int(job.priority)

    def sort_key(self, job: TrainingJob, *, now: datetime | None = None) -> tuple[int, str]:
        return (-self.effective_priority(job, now=now), job.job_id)

    def choose(self, jobs: list[TrainingJob], *, now: datetime | None = None) -> TrainingJob | None:
        if not jobs:
            return None
        return sorted(jobs, key=lambda job: self.sort_key(job, now=now))[0]

    def should_preempt(self, active_job: TrainingJob, candidate_job: TrainingJob, *, now: datetime | None = None) -> bool:
        candidate_priority = self.effective_priority(candidate_job, now=now)
        active_priority = self.effective_priority(active_job, now=now)
        return candidate_priority > active_priority
