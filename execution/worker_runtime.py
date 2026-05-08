"""Shared worker-runtime helpers for subprocess and stream execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import traceback

from ..checkpointing.manager import CheckpointManager
from ..model_cache.cache_server import CacheClient
from ..observability.events import EventLogger
from ..observability.logging_utils import setup_scheduler_logger
from ..domain import JobStatus, import_string
from ..config import SchedulerSettings
from ..storage.sqlite_store import SQLiteStateStore
from .control import ControlPlane, TrainingControlHook
from .runner_protocol import RunnerContext


def load_runtime_settings(runtime_root: str | Path) -> SchedulerSettings:
    runtime_root_path = Path(runtime_root)
    runtime_settings_path = runtime_root_path / "scheduler_settings.json"
    return (
        SchedulerSettings.from_file(runtime_settings_path)
        if runtime_settings_path.exists()
        else SchedulerSettings(runtime_root=runtime_root_path)
    )


def create_runner_context(
    settings: SchedulerSettings,
    store: SQLiteStateStore,
    event_logger: EventLogger,
    job_id: str,
) -> tuple[RunnerContext | None, Any | None]:
    job = store.get_job(job_id)
    if job is None:
        return None, None

    control_plane = ControlPlane(settings)
    control_plane.initialize_job(job_id)
    checkpoint_manager = CheckpointManager(settings, store, event_logger)
    control_hook = TrainingControlHook(job, control_plane, checkpoint_manager, store, event_logger)

    cache_client = None
    try:
        cache_client = CacheClient(settings)
        cache_client.ping()
    except Exception:
        cache_client = None

    context = RunnerContext(
        job=store.get_job(job_id) or job,
        settings=settings,
        store=store,
        event_logger=event_logger,
        control_hook=control_hook,
        checkpoint_manager=checkpoint_manager,
        cache_client=cache_client,
    )
    return context, job


def mark_job_started(
    settings: SchedulerSettings,
    store: SQLiteStateStore,
    event_logger: EventLogger,
    job_id: str,
    *,
    backend_name: str | None = None,
) -> None:
    job = store.get_job(job_id)
    if job is None:
        return
    is_resume = bool(job.latest_checkpoint_path or job.resume_from_checkpoint)
    store.set_job_status(job_id, JobStatus.RUNNING, reason=f"{backend_name or 'worker'} started", hold=False)
    payload = {"resume": is_resume}
    if backend_name is not None:
        payload["backend_name"] = backend_name
    event_logger.emit("job_resumed" if is_resume else "job_started", job_id=job_id, payload=payload)


def mark_job_failed(
    settings: SchedulerSettings,
    store: SQLiteStateStore,
    event_logger: EventLogger,
    job_id: str,
    exc: Exception,
    *,
    backend_name: str | None = None,
) -> int:
    traceback_text = traceback.format_exc()
    logger = setup_scheduler_logger(settings.scheduler_log_path)
    store.set_job_status(job_id, JobStatus.FAILED, reason=str(exc), hold=True)
    payload = {"error": repr(exc), "traceback": traceback_text}
    if backend_name is not None:
        payload["backend_name"] = backend_name
    event_logger.emit("job_failed", job_id=job_id, payload=payload)
    logger.error("Job %s failed%s:\n%s", job_id, f" in {backend_name}" if backend_name else "", traceback_text)
    return 1


def mark_job_completed(
    settings: SchedulerSettings,
    store: SQLiteStateStore,
    event_logger: EventLogger,
    job_id: str,
    result: dict[str, Any] | None,
    *,
    backend_name: str | None = None,
) -> int:
    logger = setup_scheduler_logger(settings.scheduler_log_path)
    current = store.get_job(job_id)
    if current is None:
        return 1
    if current.status in {JobStatus.PAUSED, JobStatus.CANCELLED, JobStatus.FAILED}:
        return 0
    store.set_job_status(job_id, JobStatus.COMPLETED, reason=(result or {}).get("reason"), hold=False)
    payload = dict(result or {})
    if backend_name is not None:
        payload["backend_name"] = backend_name
    event_logger.emit("job_completed", job_id=job_id, payload=payload)
    logger.info("Job %s completed successfully%s", job_id, f" in {backend_name}" if backend_name else "")
    return 0


def resolve_runner(context: RunnerContext):
    runner = import_string(context.job.config.runner_target)
    return runner if callable(runner) else runner.run
