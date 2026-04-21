"""Runnable job queue abstraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..schemas import TrainingJob
from .policies import SchedulingPolicy


@dataclass
class RunnableJobQueue:
    """A policy-backed queue view over runnable jobs."""

    policy: SchedulingPolicy
    jobs: list[TrainingJob] = field(default_factory=list)

    def ordered(self) -> list[TrainingJob]:
        now = datetime.now(timezone.utc)
        return sorted(self.jobs, key=lambda job: self.policy.sort_key(job, now=now))

    def peek(self) -> TrainingJob | None:
        ordered = self.ordered()
        return ordered[0] if ordered else None

    def pop(self) -> TrainingJob | None:
        ordered = self.ordered()
        if not ordered:
            return None
        job = ordered[0]
        self.jobs = [candidate for candidate in self.jobs if candidate.job_id != job.job_id]
        return job
