"""Scheduling policies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from ..schemas import TrainingJob, parse_timestamp


class SchedulingPolicy(Protocol):
    """Interface for scheduling policies."""

    def sort_key(self, job: TrainingJob, *, now: datetime | None = None) -> tuple[int, int]:
        ...

    def choose(self, jobs: list[TrainingJob], *, now: datetime | None = None) -> TrainingJob | None:
        ...

    def should_preempt(self, active_job: TrainingJob, candidate_job: TrainingJob, *, now: datetime | None = None) -> bool:
        ...


@dataclass(slots=True)
class PriorityFifoPolicy:
    """Priority-first, FIFO within equal priority, with optional aging."""

    aging_interval_seconds: float = 180.0
    aging_priority_increment: int = 1
    enable_priority_aging: bool = True

    def _aging_bonus(self, job: TrainingJob, *, now: datetime | None = None) -> int:
        if not self.enable_priority_aging or self.aging_interval_seconds <= 0:
            return 0
        now = now or datetime.now(timezone.utc)
        waiting_since = parse_timestamp(job.waiting_since())
        if waiting_since is None:
            return 0
        waited_seconds = max(0.0, (now - waiting_since).total_seconds())
        return int(waited_seconds // self.aging_interval_seconds) * self.aging_priority_increment

    def effective_priority(self, job: TrainingJob, *, now: datetime | None = None) -> int:
        return job.priority + self._aging_bonus(job, now=now)

    def sort_key(self, job: TrainingJob, *, now: datetime | None = None) -> tuple[int, int]:
        effective = self.effective_priority(job, now=now)
        return (-effective, job.queue_sequence)

    def choose(self, jobs: list[TrainingJob], *, now: datetime | None = None) -> TrainingJob | None:
        if not jobs:
            return None
        return sorted(jobs, key=lambda job: self.sort_key(job, now=now))[0]

    def should_preempt(self, active_job: TrainingJob, candidate_job: TrainingJob, *, now: datetime | None = None) -> bool:
        candidate_priority = self.effective_priority(candidate_job, now=now)
        active_priority = self.effective_priority(active_job, now=now)
        return candidate_priority > active_priority
