"""Stable scheduler/benchmark outcome classification."""

from __future__ import annotations

from ..domain import JobStatus, TrainingJob


def classify_job_outcome(job: TrainingJob, *, externally_timed_out: bool = False) -> str:
    early_result = job.metadata.get("early_stopping_result") or {}
    if job.status == JobStatus.COMPLETED and early_result.get("early_stopped_successfully"):
        return "early_stopped_successfully"
    if job.status == JobStatus.COMPLETED:
        return "completed"
    if job.status == JobStatus.FAILED:
        return "failed"
    if job.status == JobStatus.CANCELLED:
        return "cancelled"
    if externally_timed_out:
        return "externally_timed_out"
    if job.started_at is not None or job.status == JobStatus.RUNNING:
        return "training_started"
    return "not_started"
