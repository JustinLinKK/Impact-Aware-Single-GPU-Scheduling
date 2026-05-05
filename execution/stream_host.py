"""In-process host for the experimental CUDA stream backend."""

from __future__ import annotations

import argparse
import threading
import traceback
from pathlib import Path

import torch

from ..checkpointing.manager import CheckpointManager
from ..model_cache.cache_server import CacheClient
from ..observability.events import EventLogger
from ..observability.logging_utils import setup_scheduler_logger
from ..schemas import JobStatus, import_string
from ..settings import SchedulerSettings
from ..storage.sqlite_store import SQLiteStateStore
from .control import CancelRequested, ControlPlane, PauseRequested, TrainingControlHook
from .runner_protocol import RunnerContext


def _run_job_in_thread(settings: SchedulerSettings, store: SQLiteStateStore, event_logger: EventLogger, job_id: str, results: dict[str, int]) -> None:
    logger = setup_scheduler_logger(settings.scheduler_log_path)
    job = store.get_job(job_id)
    if job is None:
        results[job_id] = 1
        return

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
    store.set_job_status(job_id, JobStatus.RUNNING, reason="stream host started", hold=False)
    event_logger.emit("job_resumed" if is_resume else "job_started", job_id=job_id, payload={"resume": is_resume, "backend_name": "stream"})
    context = RunnerContext(
        job=store.get_job(job_id) or job,
        settings=settings,
        store=store,
        event_logger=event_logger,
        control_hook=control_hook,
        checkpoint_manager=checkpoint_manager,
        cache_client=cache_client,
    )

    try:
        runner = import_string(context.job.config.runner_target)
        if torch.cuda.is_available():
            torch.cuda.set_device(settings.gpu_scheduler.device_index)
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                result = runner(context) if callable(runner) else runner.run(context)
            torch.cuda.synchronize()
        else:
            result = runner(context) if callable(runner) else runner.run(context)
    except PauseRequested:
        logger.info("Job %s paused cleanly in stream host", job_id)
        results[job_id] = 0
        return
    except CancelRequested:
        logger.info("Job %s cancelled cleanly in stream host", job_id)
        results[job_id] = 0
        return
    except Exception as exc:
        traceback_text = traceback.format_exc()
        store.set_job_status(job_id, JobStatus.FAILED, reason=str(exc), hold=True)
        event_logger.emit("job_failed", job_id=job_id, payload={"error": repr(exc), "traceback": traceback_text, "backend_name": "stream"})
        logger.error("Job %s failed in stream host:\n%s", job_id, traceback_text)
        results[job_id] = 1
        return

    current = store.get_job(job_id)
    if current is None:
        results[job_id] = 1
        return
    if current.status in {JobStatus.PAUSED, JobStatus.CANCELLED, JobStatus.FAILED}:
        results[job_id] = 0
        return
    store.set_job_status(job_id, JobStatus.COMPLETED, reason=(result or {}).get("reason"), hold=False)
    event_logger.emit("job_completed", job_id=job_id, payload={**(result or {}), "backend_name": "stream"})
    logger.info("Job %s completed successfully in stream host", job_id)
    results[job_id] = 0


def main() -> int:
    parser = argparse.ArgumentParser(description="localml_scheduler CUDA stream host")
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--job-id", action="append", dest="job_ids", required=True)
    args = parser.parse_args()

    runtime_settings_path = Path(args.runtime_root).resolve() / "scheduler_settings.json"
    settings = (
        SchedulerSettings.from_file(runtime_settings_path)
        if runtime_settings_path.exists()
        else SchedulerSettings(runtime_root=args.runtime_root)
    )
    store = SQLiteStateStore(settings)
    event_logger = EventLogger(store, settings.events_jsonl_path)
    results: dict[str, int] = {}
    threads = [
        threading.Thread(
            target=_run_job_in_thread,
            args=(settings, store, event_logger, job_id, results),
            name=f"stream-host-{job_id}",
            daemon=False,
        )
        for job_id in args.job_ids
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return 0 if all(results.get(job_id, 1) == 0 for job_id in args.job_ids) else 1


if __name__ == "__main__":
    raise SystemExit(main())
