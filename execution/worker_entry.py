"""Worker subprocess entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import traceback

from ..model_cache.cache_server import CacheClient
from ..checkpointing.manager import CheckpointManager
from ..observability.events import EventLogger
from ..observability.logging_utils import setup_scheduler_logger
from ..schemas import JobStatus, import_string
from ..settings import SchedulerSettings
from ..storage.sqlite_store import SQLiteStateStore
from .control import CancelRequested, ControlPlane, PauseRequested, TrainingControlHook
from .runner_protocol import RunnerContext


def _run_job(runtime_root: str, job_id: str) -> int:
    runtime_settings_path = Path(runtime_root) / "scheduler_settings.json"
    settings = (
        SchedulerSettings.from_file(runtime_settings_path)
        if runtime_settings_path.exists()
        else SchedulerSettings(runtime_root=runtime_root)
    )
    store = SQLiteStateStore(settings)
    event_logger = EventLogger(store, settings.events_jsonl_path)
    logger = setup_scheduler_logger(settings.scheduler_log_path)
    job = store.get_job(job_id)
    if job is None:
        raise KeyError(f"Unknown job_id: {job_id}")

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

    is_resume = bool(job.latest_checkpoint_path or job.resume_from_checkpoint)
    store.set_job_status(job_id, JobStatus.RUNNING, reason="worker started", hold=False)
    event_logger.emit("job_resumed" if is_resume else "job_started", job_id=job_id, payload={"resume": is_resume})

    context = RunnerContext(
        job=store.get_job(job_id) or job,
        settings=settings,
        store=store,
        event_logger=event_logger,
        control_hook=control_hook,
        checkpoint_manager=checkpoint_manager,
        cache_client=cache_client,
    )
    runner = import_string(context.job.config.runner_target)

    try:
        result = runner(context) if callable(runner) else runner.run(context)
    except PauseRequested:
        logger.info("Job %s paused cleanly at a safe point", job_id)
        return 0
    except CancelRequested:
        logger.info("Job %s cancelled cleanly at a safe point", job_id)
        return 0
    except Exception as exc:
        traceback_text = traceback.format_exc()
        store.set_job_status(job_id, JobStatus.FAILED, reason=str(exc), hold=True)
        event_logger.emit("job_failed", job_id=job_id, payload={"error": repr(exc), "traceback": traceback_text})
        logger.error("Job %s failed:\n%s", job_id, traceback_text)
        return 1

    current = store.get_job(job_id)
    if current is None:
        return 1
    if current.status in {JobStatus.PAUSED, JobStatus.CANCELLED, JobStatus.FAILED}:
        return 0
    store.set_job_status(job_id, JobStatus.COMPLETED, reason=(result or {}).get("reason"), hold=False)
    event_logger.emit("job_completed", job_id=job_id, payload=result or {})
    logger.info("Job %s completed successfully", job_id)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="localml_scheduler worker entrypoint")
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    return _run_job(runtime_root=args.runtime_root, job_id=args.job_id)


if __name__ == "__main__":
    raise SystemExit(main())
