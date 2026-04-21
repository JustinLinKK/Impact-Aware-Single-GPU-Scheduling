"""Scheduler service loop."""

from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
from typing import Any
import json
import time

from ..cache.baseline_cache import BaselineModelCache, CachedModelEntry
from ..cache.cache_server import CacheClient, CacheServer
from ..cache.warming import select_models_to_warm
from ..observability.events import EventLogger
from ..observability.logging_utils import setup_scheduler_logger
from ..observability.metrics import MetricsCollector
from ..schemas import CommandType, JobStatus, TrainingJob
from ..settings import SchedulerSettings
from ..storage.sqlite_store import SQLiteStateStore
from .policies import PriorityFifoPolicy, SchedulingPolicy
from .queue import RunnableJobQueue
from .recovery import reconcile_recoverable_jobs
from .supervisor import WorkerSupervisor


class SchedulerService:
    """Single-process scheduler for a single exclusive GPU slot."""

    def __init__(
        self,
        settings: SchedulerSettings,
        *,
        store: SQLiteStateStore | None = None,
        policy: SchedulingPolicy | None = None,
    ):
        self.settings = settings
        self.settings.ensure_runtime_layout()
        self.store = store or SQLiteStateStore(settings)
        self.logger = setup_scheduler_logger(settings.scheduler_log_path)
        self.event_logger = EventLogger(self.store, settings.events_jsonl_path)
        self.metrics = MetricsCollector(self.store)
        self.policy = policy or PriorityFifoPolicy(
            aging_interval_seconds=settings.aging_interval_seconds,
            aging_priority_increment=settings.aging_priority_increment,
            enable_priority_aging=settings.enable_priority_aging,
        )
        self.supervisor = WorkerSupervisor(settings)
        self.cache = BaselineModelCache(settings.cache_memory_budget_bytes, on_update=self._on_cache_update)
        self.cache_server = CacheServer(settings, self.cache)
        self._stop_event = Event()
        self._thread: Thread | None = None

    def _persist_runtime_settings(self) -> None:
        path = self.settings.runtime_root / "scheduler_settings.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.settings.to_dict(), handle, indent=2, sort_keys=True)

    def _on_cache_update(self, event_name: str, entry: CachedModelEntry, payload: dict[str, Any] | None) -> None:
        self.store.update_cache_metadata(
            entry.model_id,
            entry.baseline_model_path,
            size_bytes=entry.size_bytes,
            pinned=entry.pinned,
            hits=entry.hits,
            misses=entry.misses,
            last_loaded_at=entry.last_loaded_at,
            last_accessed_at=entry.last_accessed_at,
            metadata=entry.metadata,
        )
        self.event_logger.emit(event_name, payload={"model_id": entry.model_id, **(payload or {}), **entry.to_stats_dict()})

    def start(self, *, background: bool = False) -> "SchedulerService":
        self._persist_runtime_settings()
        self.cache_server.start()
        reconcile_recoverable_jobs(self.store, self.event_logger, auto_resume=self.settings.auto_resume_recoverable)
        if background:
            if self._thread is not None and self._thread.is_alive():
                return self
            self._thread = Thread(target=self.run_forever, name="scheduler-service", daemon=True)
            self._thread.start()
            return self
        self.run_forever()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self.cache_server.stop()

    def run_forever(self) -> None:
        self.logger.info("Scheduler service started")
        while not self._stop_event.is_set():
            self._poll_active_worker()
            self._process_commands()
            self._warm_cache()
            self._maybe_preempt()
            self._dispatch_if_idle()
            self._stop_event.wait(self.settings.scheduler_poll_interval_seconds)
        self.logger.info("Scheduler service stopped")

    def _process_commands(self) -> None:
        commands = self.store.fetch_pending_commands(limit=self.settings.command_poll_limit)
        for command in commands:
            try:
                if command.command_type == CommandType.SUBMIT:
                    self._handle_submit(command.job_id)
                elif command.command_type == CommandType.PAUSE:
                    self._handle_pause(command.job_id)
                elif command.command_type == CommandType.RESUME:
                    self._handle_resume(command.job_id)
                elif command.command_type == CommandType.CANCEL:
                    self._handle_cancel(command.job_id)
                elif command.command_type == CommandType.PRELOAD:
                    self._handle_preload(command.payload)
            finally:
                self.store.mark_command_processed(command.command_id)

    def _handle_submit(self, job_id: str | None) -> None:
        if job_id is None:
            return
        job = self.store.get_job(job_id)
        if job is None or job.status.is_terminal:
            return
        if job.status != JobStatus.READY:
            self.store.set_job_status(job_id, JobStatus.READY, reason="job accepted by scheduler", hold=False)
        self.event_logger.emit("job_ready", job_id=job_id, payload={"priority": job.priority})

    def _handle_pause(self, job_id: str | None) -> None:
        if job_id is None:
            return
        job = self.store.get_job(job_id)
        if job is None or job.status.is_terminal:
            return
        if self.supervisor.request_pause(job_id, reason="manual pause requested", hold=True):
            self.store.set_job_status(job_id, JobStatus.PAUSING, reason="manual pause requested", hold=True)
            self.event_logger.emit("pause_requested", job_id=job_id, payload={"hold": True})
            return
        self.store.set_job_status(job_id, JobStatus.PAUSED, reason="manual pause while queued", hold=True)
        self.event_logger.emit("job_paused", job_id=job_id, payload={"hold": True, "queued": True})

    def _handle_resume(self, job_id: str | None) -> None:
        if job_id is None:
            return
        job = self.store.get_job(job_id)
        if job is None or job.status.is_terminal:
            return
        if job.status in {JobStatus.PAUSED, JobStatus.RECOVERABLE, JobStatus.PENDING, JobStatus.READY}:
            self.store.set_job_status(job_id, JobStatus.READY, reason="resume requested", hold=False)
            self.event_logger.emit("job_resume_requested", job_id=job_id, payload={})

    def _handle_cancel(self, job_id: str | None) -> None:
        if job_id is None:
            return
        job = self.store.get_job(job_id)
        if job is None or job.status.is_terminal:
            return
        if self.supervisor.request_cancel(job_id, reason="cancel requested"):
            self.store.update_job(job_id, reason="cancel requested", hold=True)
            self.event_logger.emit("cancel_requested", job_id=job_id, payload={})
            return
        self.store.set_job_status(job_id, JobStatus.CANCELLED, reason="cancelled while queued", hold=True)
        self.event_logger.emit("job_cancelled", job_id=job_id, payload={"queued": True})

    def _handle_preload(self, payload: dict[str, Any]) -> None:
        model_id = payload["baseline_model_id"]
        baseline_model_path = payload["baseline_model_path"]
        loader_target = payload.get("loader_target")
        pin = bool(payload.get("pin", False))
        ok = self.cache.preload(model_id, baseline_model_path, loader_target=loader_target, pin=pin, metadata={"source": "command"})
        self.event_logger.emit("cache_preload_requested", payload={"model_id": model_id, "ok": ok, "pin": pin})

    def _runnable_jobs(self) -> list[TrainingJob]:
        jobs = self.store.runnable_jobs()
        if not self.settings.auto_resume_recoverable:
            jobs = [job for job in jobs if job.status != JobStatus.RECOVERABLE]
        return jobs

    def _warm_cache(self) -> None:
        jobs = self._runnable_jobs()
        for model_id, baseline_model_path, loader_target in select_models_to_warm(jobs, top_k=self.settings.eager_preload_top_k):
            try:
                self.cache.preload(model_id, baseline_model_path, loader_target=loader_target, metadata={"source": "warming"})
            except Exception as exc:
                self.logger.warning("Cache warming failed for %s: %s", model_id, exc)

    def _poll_active_worker(self) -> None:
        snapshot = self.supervisor.poll()
        if snapshot is None:
            return
        if snapshot.alive:
            return
        job = self.store.get_job(snapshot.job_id)
        if job is None:
            return
        if snapshot.returncode == 0:
            if job.status in {JobStatus.COMPLETED, JobStatus.PAUSED, JobStatus.CANCELLED, JobStatus.READY}:
                return
            self.store.set_job_status(job.job_id, JobStatus.FAILED, reason="worker exited without terminal status update", hold=True)
            self.event_logger.emit("job_failed", job_id=job.job_id, payload={"reason": "worker exited cleanly without terminal status"})
            return
        if not job.status.is_terminal:
            self.store.set_job_status(job.job_id, JobStatus.FAILED, reason=f"worker exited with code {snapshot.returncode}", hold=True)
            self.event_logger.emit("job_failed", job_id=job.job_id, payload={"returncode": snapshot.returncode})

    def _next_job(self) -> TrainingJob | None:
        queue = RunnableJobQueue(policy=self.policy, jobs=self._runnable_jobs())
        return queue.peek()

    def _maybe_preempt(self) -> None:
        active_job_id = self.supervisor.active_job_id()
        if active_job_id is None:
            return
        active_job = self.store.get_job(active_job_id)
        candidate_job = self._next_job()
        if active_job is None or candidate_job is None:
            return
        if candidate_job.job_id == active_job.job_id:
            return
        if active_job.status != JobStatus.RUNNING:
            return
        if not self.policy.should_preempt(active_job, candidate_job):
            return
        reason = f"preempted by higher-priority job {candidate_job.job_id}"
        if self.supervisor.request_pause(active_job.job_id, reason=reason, hold=False):
            self.store.set_job_status(active_job.job_id, JobStatus.PAUSING, reason=reason, hold=False)
            self.event_logger.emit(
                "pause_requested",
                job_id=active_job.job_id,
                payload={"reason": reason, "preempting_job_id": candidate_job.job_id, "hold": False},
            )

    def _dispatch_if_idle(self) -> None:
        next_job = self._next_job()
        if next_job is None:
            return
        decision = self.supervisor.placement_for(next_job)
        if not decision.can_run:
            return
        try:
            self.cache.preload(
                next_job.baseline_model_id,
                next_job.baseline_model_path,
                loader_target=next_job.config.loader_target,
                metadata={"source": "dispatch", "job_id": next_job.job_id},
            )
        except Exception as exc:
            self.logger.warning("Baseline preload failed for job %s: %s", next_job.job_id, exc)
        self.store.set_job_status(next_job.job_id, JobStatus.RUNNING, reason="dispatched to worker", hold=False)
        self.event_logger.emit("job_dispatched", job_id=next_job.job_id, payload={"priority": next_job.priority})
        dispatched = self.supervisor.dispatch(self.store.get_job(next_job.job_id) or next_job)
        if not dispatched.can_run:
            self.logger.info("Skipping dispatch for %s: %s", next_job.job_id, dispatched.reason)

    def report(self) -> dict[str, Any]:
        return self.metrics.as_dict()

    def cache_stats(self) -> dict[str, Any]:
        return {
            "stats": self.cache.stats().to_dict(),
            "entries": self.cache.snapshot_entries(),
        }
