"""In-process host for the experimental CUDA stream backend."""

from __future__ import annotations

import argparse
import threading

import torch

from ..observability.events import EventLogger
from ..observability.logging_utils import setup_scheduler_logger
from ..config import SchedulerSettings
from ..storage.sqlite_store import SQLiteStateStore
from .control import CancelRequested, PauseRequested
from .worker_runtime import create_runner_context, load_runtime_settings, mark_job_completed, mark_job_failed, mark_job_started, resolve_runner


def _run_job_in_thread(settings: SchedulerSettings, store: SQLiteStateStore, event_logger: EventLogger, job_id: str, results: dict[str, int]) -> None:
    logger = setup_scheduler_logger(settings.scheduler_log_path)
    context, job = create_runner_context(settings, store, event_logger, job_id)
    if context is None or job is None:
        results[job_id] = 1
        return

    mark_job_started(settings, store, event_logger, job_id, backend_name="stream")

    try:
        runner = resolve_runner(context)
        if torch.cuda.is_available():
            torch.cuda.set_device(settings.gpu_scheduler.device_index)
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                result = runner(context)
            torch.cuda.synchronize()
        else:
            result = runner(context)
    except PauseRequested:
        logger.info("Job %s paused cleanly in stream host", job_id)
        results[job_id] = 0
        return
    except CancelRequested:
        logger.info("Job %s cancelled cleanly in stream host", job_id)
        results[job_id] = 0
        return
    except Exception as exc:
        results[job_id] = mark_job_failed(settings, store, event_logger, job_id, exc, backend_name="stream")
        return

    results[job_id] = mark_job_completed(settings, store, event_logger, job_id, result, backend_name="stream")


def main() -> int:
    parser = argparse.ArgumentParser(description="localml_scheduler CUDA stream host")
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--job-id", action="append", dest="job_ids", required=True)
    args = parser.parse_args()

    settings = load_runtime_settings(args.runtime_root)
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
