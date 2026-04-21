"""Scheduler restart recovery helpers."""

from __future__ import annotations

from ..observability.events import EventLogger
from ..schemas import JobStatus
from ..storage.sqlite_store import SQLiteStateStore


def reconcile_recoverable_jobs(store: SQLiteStateStore, event_logger: EventLogger, *, auto_resume: bool) -> list[str]:
    """Reconcile stale RUNNING/PAUSING jobs after a restart."""
    reconciled_ids: list[str] = []
    for job in store.reconcile_incomplete_jobs():
        if auto_resume and job.status == JobStatus.RECOVERABLE:
            store.set_job_status(job.job_id, JobStatus.READY, reason="auto-resume recoverable job", hold=False)
        event_logger.emit("job_recovered", job_id=job.job_id, payload={"status": job.status.value, "auto_resume": auto_resume})
        reconciled_ids.append(job.job_id)
    return reconciled_ids
