"""Worker subprocess entrypoint."""

from __future__ import annotations

import argparse

from ..observability.events import EventLogger
from ..observability.logging_utils import setup_scheduler_logger
from ..profiling.batch_probe import run_batch_probe_preflight
from ..config import SchedulerSettings
from ..storage.log_store import SchedulerLogStore
from ..storage.state_store import StateStore
from .control import CancelRequested, PauseRequested
from .worker_runtime import create_runner_context, load_runtime_settings, mark_job_completed, mark_job_failed, mark_job_started, resolve_runner


def _run_job(runtime_root: str, job_id: str) -> int:
    settings = load_runtime_settings(runtime_root)
    store = StateStore(settings)
    event_logger = EventLogger(store, settings.events_jsonl_path, log_store=SchedulerLogStore(settings))
    logger = setup_scheduler_logger(settings.scheduler_log_path)
    context, job = create_runner_context(settings, store, event_logger, job_id)
    if context is None or job is None:
        raise KeyError(f"Unknown job_id: {job_id}")
    mark_job_started(settings, store, event_logger, job_id, backend_name="exclusive")

    try:
        context.job = run_batch_probe_preflight(context)
        result = resolve_runner(context)(context)
    except PauseRequested:
        logger.info("Job %s paused cleanly at a safe point", job_id)
        return 0
    except CancelRequested:
        logger.info("Job %s cancelled cleanly at a safe point", job_id)
        return 0
    except Exception as exc:
        return mark_job_failed(settings, store, event_logger, job_id, exc, backend_name="exclusive")

    return mark_job_completed(settings, store, event_logger, job_id, result, backend_name="exclusive")


def main() -> int:
    parser = argparse.ArgumentParser(description="localml_scheduler worker entrypoint")
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    return _run_job(runtime_root=args.runtime_root, job_id=args.job_id)


if __name__ == "__main__":
    raise SystemExit(main())
