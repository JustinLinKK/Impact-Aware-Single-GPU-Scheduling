"""Persistent, single-context CUDA stream execution host.

The scheduler admits jobs incrementally.  A one-shot stream host would create a
new CUDA context for every incremental admission, which is process concurrency
with a misleading name.  This module instead owns one long-lived process and
accepts jobs over a small runtime-local Unix socket.  Every admitted job runs
in a separate Python thread and on a distinct ``torch.cuda.Stream`` inside the
same CUDA context.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import threading
import time
from typing import Any

import torch

from ..observability.events import EventLogger
from ..observability.logging_utils import setup_scheduler_logger
from ..config import SchedulerSettings
from ..storage.log_store import SchedulerLogStore
from ..storage.state_store import StateStore
from .control import CancelRequested, EarlyStopRequested, PauseRequested
from .worker_runtime import (
    create_runner_context,
    load_runtime_settings,
    mark_job_completed,
    mark_job_failed,
    mark_job_started,
    resolve_runner,
)


def _stream_identifier(stream: Any) -> int | str:
    value = getattr(stream, "cuda_stream", None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value if value is not None else id(stream))


def _run_job_in_thread(
    settings: SchedulerSettings,
    store: StateStore,
    event_logger: EventLogger,
    job_id: str,
    results: dict[str, int],
    *,
    stream: Any | None = None,
    ready_event: threading.Event | None = None,
    host_pid: int | None = None,
    start_delay_seconds: float = 0.0,
) -> None:
    """Run one structured job and publish its stream identity before training."""

    logger = setup_scheduler_logger(settings.scheduler_log_path)
    try:
        context, job = create_runner_context(settings, store, event_logger, job_id)
        if context is None or job is None:
            results[job_id] = 1
            return

        if torch.cuda.is_available():
            torch.cuda.set_device(settings.gpu_scheduler.device_index)
            stream = stream or torch.cuda.Stream(
                device=settings.gpu_scheduler.device_index
            )
            stream_id: int | str | None = _stream_identifier(stream)
        else:
            stream_id = None

        process_id = int(host_pid if host_pid is not None else os.getpid())
        mark_job_started(settings, store, event_logger, job_id, backend_name="stream")
        store.update_job(
            job_id,
            metadata_updates={
                "stream_host_pid": process_id,
                "cuda_stream_id": stream_id,
                "placement_backend": "stream",
            },
        )
        # The context was constructed just before placement metadata was
        # persisted.  Keep the in-thread snapshot aligned so runner-produced
        # profiles and artifacts are correctly scoped to the stream backend.
        context.job.metadata.update(
            {
                "stream_host_pid": process_id,
                "cuda_stream_id": stream_id,
                "placement_backend": "stream",
            }
        )
        event_logger.emit(
            "cuda_stream_assigned",
            job_id=job_id,
            payload={
                "stream_host_pid": process_id,
                "cuda_stream_id": stream_id,
                "device_index": settings.gpu_scheduler.device_index,
            },
        )
        if ready_event is not None:
            ready_event.set()

        if start_delay_seconds > 0:
            time.sleep(float(start_delay_seconds))

        runner = resolve_runner(context)
        if stream is not None and torch.cuda.is_available():
            with torch.cuda.stream(stream):
                result = runner(context)
            # Synchronize only this job.  A device-wide synchronize would turn
            # independently admitted streams back into a barriered workload.
            stream.synchronize()
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
    except EarlyStopRequested as exc:
        logger.info("Job %s early-stopped successfully in stream host", job_id)
        results[job_id] = mark_job_completed(
            settings,
            store,
            event_logger,
            job_id,
            exc.result,
            backend_name="stream",
        )
        return
    except Exception as exc:
        results[job_id] = mark_job_failed(
            settings,
            store,
            event_logger,
            job_id,
            exc,
            backend_name="stream",
        )
        return
    finally:
        if ready_event is not None:
            ready_event.set()

    results[job_id] = mark_job_completed(
        settings,
        store,
        event_logger,
        job_id,
        result,
        backend_name="stream",
    )


class StreamHostServer:
    """Runtime-local JSON command server for incremental stream admission."""

    def __init__(self, settings: SchedulerSettings, socket_path: Path):
        self.settings = settings
        self.socket_path = socket_path
        self.store = StateStore(settings)
        self.event_logger = EventLogger(
            self.store,
            settings.events_jsonl_path,
            log_store=SchedulerLogStore(settings),
        )
        self.results: dict[str, int] = {}
        self.threads: dict[str, threading.Thread] = {}
        self._shutdown = threading.Event()
        self._lock = threading.Lock()

    def _clean_finished(self) -> None:
        with self._lock:
            self.threads = {
                job_id: thread
                for job_id, thread in self.threads.items()
                if thread.is_alive()
            }

    def _launch(self, job_ids: list[str]) -> dict[str, Any]:
        self._clean_finished()
        if not job_ids:
            return {"ok": False, "error": "launch requires at least one job_id"}

        duplicate = [job_id for job_id in job_ids if job_id in self.threads]
        # A cooperative pause persists PAUSED just before its runner thread
        # unwinds.  The scheduler may immediately select that resumable job;
        # bridge the tiny state/thread gap here instead of rejecting a valid
        # resume as a duplicate admission.
        for job_id in duplicate:
            self.threads[job_id].join(
                timeout=float(
                    self.settings.gpu_scheduler.stream.host_join_timeout_seconds
                )
            )
        self._clean_finished()
        duplicate = [job_id for job_id in job_ids if job_id in self.threads]
        if duplicate:
            return {
                "ok": False,
                "error": f"jobs already active in stream host: {duplicate}",
            }

        if torch.cuda.is_available():
            torch.cuda.set_device(self.settings.gpu_scheduler.device_index)

        ready: list[tuple[str, threading.Event]] = []
        stream_ids: dict[str, int | str | None] = {}
        for job_id in job_ids:
            job = self.store.get_job(job_id)
            try:
                start_delay_seconds = max(
                    0.0,
                    float(
                        job.metadata.get("placement_start_delay_seconds", 0.0)
                        if job is not None
                        else 0.0
                    ),
                )
            except (TypeError, ValueError):
                start_delay_seconds = 0.0
            stream = (
                torch.cuda.Stream(device=self.settings.gpu_scheduler.device_index)
                if torch.cuda.is_available()
                else None
            )
            stream_ids[job_id] = (
                _stream_identifier(stream) if stream is not None else None
            )
            ready_event = threading.Event()
            thread = threading.Thread(
                target=_run_job_in_thread,
                args=(
                    self.settings,
                    self.store,
                    self.event_logger,
                    job_id,
                    self.results,
                ),
                kwargs={
                    "stream": stream,
                    "ready_event": ready_event,
                    "host_pid": os.getpid(),
                    "start_delay_seconds": start_delay_seconds,
                },
                name=f"cuda-stream-job-{job_id}",
                daemon=False,
            )
            with self._lock:
                self.threads[job_id] = thread
            thread.start()
            ready.append((job_id, ready_event))

        not_ready = [job_id for job_id, event in ready if not event.wait(30.0)]
        if not_ready:
            return {
                "ok": False,
                "error": f"stream jobs did not enter the worker runtime: {not_ready}",
            }
        return {
            "ok": True,
            "host_pid": os.getpid(),
            "stream_ids": stream_ids,
        }

    def _handle(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = str(request.get("op") or "")
        if operation == "ping":
            self._clean_finished()
            return {
                "ok": True,
                "host_pid": os.getpid(),
                "active_job_ids": sorted(self.threads),
            }
        if operation == "launch":
            return self._launch([str(item) for item in request.get("job_ids") or []])
        if operation == "shutdown":
            self._shutdown.set()
            return {"ok": True, "host_pid": os.getpid()}
        return {"ok": False, "error": f"unknown operation: {operation!r}"}

    def serve(self) -> int:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            listener.listen(16)
            listener.settimeout(0.5)
            while not self._shutdown.is_set():
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    self._clean_finished()
                    continue
                with connection:
                    stream = connection.makefile("rwb")
                    try:
                        raw = stream.readline()
                        request = json.loads(raw.decode("utf-8")) if raw else {}
                        response = self._handle(request)
                    except Exception as exc:
                        response = {"ok": False, "error": repr(exc)}
                    stream.write(
                        (json.dumps(response, sort_keys=True) + "\n").encode("utf-8")
                    )
                    stream.flush()
        finally:
            listener.close()
            self.socket_path.unlink(missing_ok=True)

        # Service shutdown normally arrives after job cancellation.  Give
        # workers a brief chance to publish terminal state before the owning
        # backend escalates to process-tree termination.
        for thread in list(self.threads.values()):
            thread.join(timeout=2.0)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="persistent localml_scheduler CUDA stream host"
    )
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--socket-path", required=True)
    args = parser.parse_args()

    settings = load_runtime_settings(args.runtime_root)
    return StreamHostServer(settings, Path(args.socket_path)).serve()


if __name__ == "__main__":
    raise SystemExit(main())
