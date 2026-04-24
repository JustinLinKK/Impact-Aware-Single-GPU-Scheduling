"""Scheduler service loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Thread
from typing import Any
import json
import time

from ..model_cache.baseline_cache import BaselineModelCache, CachedModelEntry
from ..model_cache.cache_server import CacheServer
from ..model_cache.warming import select_models_to_warm
from ..observability.events import EventLogger
from ..observability.logging_utils import setup_scheduler_logger
from ..observability.metrics import MetricsCollector
from ..schemas import JobStatus, PairProfile, SoloProfile, TrainingJob
from ..settings import SchedulerSettings
from ..storage.sqlite_store import SQLiteStateStore
from .gpu_scheduler import GpuPlacementPlanner, PlacementPlan
from .policies import PriorityFifoPolicy, SchedulingPolicy
from .queue import RunnableJobQueue
from .recovery import reconcile_recoverable_jobs
from .supervisor import WorkerSnapshot, WorkerSupervisor
from .telemetry import GpuTelemetrySample, GpuTelemetrySummary, NvidiaSmiTelemetrySampler


@dataclass(slots=True)
class ActiveRun:
    mode: str
    backend_name: str
    job_ids: tuple[str, ...]
    samples: list[GpuTelemetrySample] = field(default_factory=list)
    fallback_triggered: bool = False
    fallback_reason: str | None = None


class SchedulerService:
    """Single-process scheduler with optional pairwise packed execution."""

    def __init__(
        self,
        settings: SchedulerSettings,
        *,
        store: SQLiteStateStore | None = None,
        policy: SchedulingPolicy | None = None,
        supervisor: WorkerSupervisor | None = None,
        telemetry_sampler: NvidiaSmiTelemetrySampler | None = None,
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
        self.supervisor = supervisor or WorkerSupervisor(settings)
        self.planner = GpuPlacementPlanner(settings, self.store, self.policy)
        self.telemetry_sampler = telemetry_sampler or NvidiaSmiTelemetrySampler(settings.gpu_scheduler.device_index)
        self.cache = BaselineModelCache(settings.cache_memory_budget_bytes, on_update=self._on_cache_update)
        self.cache_server = CacheServer(settings, self.cache)
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._active_run: ActiveRun | None = None
        self._last_telemetry_poll_at = 0.0

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
        self.supervisor.shutdown()
        self.cache_server.stop()

    def run_forever(self) -> None:
        self.logger.info("Scheduler service started")
        while not self._stop_event.is_set():
            self._poll_active_workers()
            self._process_commands()
            self._warm_cache()
            self._poll_telemetry()
            self._enforce_packed_safety()
            self._maybe_preempt()
            self._dispatch_if_idle()
            self._stop_event.wait(self.settings.scheduler_poll_interval_seconds)
        self.logger.info("Scheduler service stopped")

    def _process_commands(self) -> None:
        commands = self.store.fetch_pending_commands(limit=self.settings.command_poll_limit)
        for command in commands:
            try:
                if command.command_type.value == "SUBMIT":
                    self._handle_submit(command.job_id)
                elif command.command_type.value == "PAUSE":
                    self._handle_pause(command.job_id)
                elif command.command_type.value == "RESUME":
                    self._handle_resume(command.job_id)
                elif command.command_type.value == "CANCEL":
                    self._handle_cancel(command.job_id)
                elif command.command_type.value == "PRELOAD":
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

    def _summary_for_active_run(self) -> GpuTelemetrySummary:
        return GpuTelemetrySummary.from_samples(self._active_run.samples if self._active_run else [])

    def _poll_telemetry(self) -> None:
        if self._active_run is None:
            return
        interval_seconds = max(0.1, self.settings.gpu_scheduler.telemetry.device_poll_ms / 1000.0)
        now = time.monotonic()
        if (now - self._last_telemetry_poll_at) < interval_seconds:
            return
        sample = self.telemetry_sampler.sample()
        self._last_telemetry_poll_at = now
        if sample is not None:
            self._active_run.samples.append(sample)

    def _enforce_packed_safety(self) -> None:
        if self._active_run is None or self._active_run.mode != "packed_pair" or self._active_run.fallback_triggered:
            return
        if not self._active_run.samples:
            return
        latest = self._active_run.samples[-1]
        if latest.memory_total_mb <= 0:
            return
        memory_fraction = latest.memory_used_mb / latest.memory_total_mb
        if memory_fraction < self.settings.gpu_scheduler.memory.hard_stop_memory_fraction:
            return
        reason = f"packed pair exceeded hard memory threshold ({memory_fraction:.2%})"
        secondary_job_id = self.supervisor.demote_secondary(reason=reason)
        if secondary_job_id is None:
            return
        self.store.set_job_status(secondary_job_id, JobStatus.PAUSING, reason=reason, hold=False)
        self._register_packed_fallback(
            reason,
            payload={
                "secondary_job_id": secondary_job_id,
                "memory_used_mb": latest.memory_used_mb,
                "memory_total_mb": latest.memory_total_mb,
            },
        )

    def _poll_active_workers(self) -> None:
        snapshots = self.supervisor.poll()
        if not snapshots:
            return

        previous_run = self._active_run

        for snapshot in snapshots:
            self._handle_worker_exit(snapshot, run_context=previous_run)

        remaining_job_ids = self.supervisor.active_job_ids()
        if previous_run is not None and previous_run.mode == "packed_pair" and len(remaining_job_ids) < 2:
            self._record_pair_profile(previous_run)
            if remaining_job_ids:
                self._active_run = ActiveRun(
                    mode="exclusive",
                    backend_name=self.supervisor.active_group_backend() or previous_run.backend_name,
                    job_ids=tuple(remaining_job_ids),
                )
            else:
                self._active_run = None
        elif previous_run is not None and previous_run.mode == "exclusive" and not remaining_job_ids:
            self._record_solo_profiles(previous_run)
            self._active_run = None

    def _handle_worker_exit(self, snapshot: WorkerSnapshot, *, run_context: ActiveRun | None) -> None:
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
            reason = f"worker exited with code {snapshot.returncode}"
            self.store.set_job_status(job.job_id, JobStatus.FAILED, reason=reason, hold=True)
            self.event_logger.emit("job_failed", job_id=job.job_id, payload={"returncode": snapshot.returncode})
            if run_context is not None and run_context.mode == "packed_pair":
                self._register_packed_fallback(reason, payload={"failed_job_id": snapshot.job_id, "returncode": snapshot.returncode})
            return

        if run_context is not None and run_context.mode == "packed_pair":
            reason = job.status_reason or f"worker exited with code {snapshot.returncode}"
            self._register_packed_fallback(reason, payload={"failed_job_id": snapshot.job_id, "returncode": snapshot.returncode})

    def _register_packed_fallback(self, reason: str, *, payload: dict[str, Any]) -> None:
        if self._active_run is None or self._active_run.mode != "packed_pair" or self._active_run.fallback_triggered:
            return
        self._active_run.fallback_triggered = True
        self._active_run.fallback_reason = reason
        self.event_logger.emit(
            "packed_pair_fallback",
            payload={"job_ids": list(self._active_run.job_ids), "reason": reason, **payload},
        )

    def _record_solo_profiles(self, run: ActiveRun) -> None:
        summary = GpuTelemetrySummary.from_samples(run.samples)
        for job_id in run.job_ids:
            job = self.store.get_job(job_id)
            if job is None or not job.packing.signature:
                continue
            if not job.packing.eligible:
                continue
            if job.status in {JobStatus.FAILED, JobStatus.CANCELLED}:
                continue
            peak_vram_mb = summary.peak_vram_mb
            if peak_vram_mb is None:
                peak_vram_mb = job.resource_requirements.estimated_vram_mb
            self.store.upsert_solo_profile(
                SoloProfile(
                    signature=job.packing.signature,
                    family=job.packing.family,
                    peak_vram_mb=peak_vram_mb,
                    avg_gpu_utilization=summary.avg_gpu_utilization if summary.avg_gpu_utilization is not None else 0.0,
                    avg_memory_utilization=summary.avg_memory_utilization if summary.avg_memory_utilization is not None else 0.0,
                    sample_count=summary.sample_count,
                    last_job_id=job.job_id,
                    metadata={"source": "exclusive_run", "backend_name": run.backend_name},
                )
            )

    def _record_pair_profile(self, run: ActiveRun) -> None:
        if len(run.job_ids) != 2:
            return
        left_job = self.store.get_job(run.job_ids[0])
        right_job = self.store.get_job(run.job_ids[1])
        if left_job is None or right_job is None:
            return
        if not left_job.packing.signature or not right_job.packing.signature:
            return
        summary = GpuTelemetrySummary.from_samples(run.samples)
        if run.fallback_triggered or left_job.status == JobStatus.FAILED or right_job.status == JobStatus.FAILED:
            self.store.mark_pair_incompatible(
                left_job.packing.signature,
                right_job.packing.signature,
                reason=run.fallback_reason or "packed pair failed",
                cooldown_seconds=self.settings.gpu_scheduler.fallback_cooldown_seconds,
                peak_vram_mb=summary.peak_vram_mb,
                avg_gpu_utilization=summary.avg_gpu_utilization,
                avg_memory_utilization=summary.avg_memory_utilization,
                metadata={"backend_name": run.backend_name},
            )
            return
        existing = self.store.get_pair_profile(left_job.packing.signature, right_job.packing.signature)
        self.store.upsert_pair_profile(
            PairProfile.create(
                left_job.packing.signature,
                right_job.packing.signature,
                compatible=True,
                observations=(existing.observations + 1) if existing else 1,
                peak_vram_mb=summary.peak_vram_mb,
                avg_gpu_utilization=summary.avg_gpu_utilization,
                avg_memory_utilization=summary.avg_memory_utilization,
                slowdown_ratio=None,
                cooldown_until=None,
                last_failure_reason=None,
                metadata={"backend_name": run.backend_name},
            )
        )

    def _next_job(self) -> TrainingJob | None:
        queue = RunnableJobQueue(policy=self.policy, jobs=self._runnable_jobs())
        return queue.peek()

    def _maybe_preempt(self) -> None:
        if self.supervisor.active_group_mode() != "exclusive":
            return
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

    def _preload_job_baseline(self, job: TrainingJob) -> None:
        try:
            self.cache.preload(
                job.baseline_model_id,
                job.baseline_model_path,
                loader_target=job.config.loader_target,
                metadata={"source": "dispatch", "job_id": job.job_id},
            )
        except Exception as exc:
            self.logger.warning("Baseline preload failed for job %s: %s", job.job_id, exc)

    def _dispatch_if_idle(self) -> None:
        if self.supervisor.active_group() is not None:
            return
        plan = self.planner.choose_plan(self._runnable_jobs(), backend_available=self.supervisor.available_backends())
        if plan is None:
            return
        selected_jobs = []
        for job_id in plan.job_ids:
            job = self.store.get_job(job_id)
            if job is None:
                return
            selected_jobs.append(job)
        for job in selected_jobs:
            self._preload_job_baseline(job)

        try:
            dispatched = self.supervisor.dispatch(selected_jobs, mode=plan.mode, backend_name=plan.backend_name)
        except Exception as exc:
            self.logger.warning("Dispatch failed for jobs %s: %s", ",".join(plan.job_ids), exc)
            return
        if not dispatched.can_run:
            self.logger.info("Skipping dispatch for %s: %s", ",".join(plan.job_ids), dispatched.reason)
            return

        self._active_run = ActiveRun(mode=plan.mode, backend_name=plan.backend_name, job_ids=plan.job_ids)
        self._last_telemetry_poll_at = 0.0

        for index, job in enumerate(selected_jobs):
            role = "primary" if plan.mode == "packed_pair" and index == 0 else ("secondary" if plan.mode == "packed_pair" else "solo")
            self.store.update_job(
                job.job_id,
                status=JobStatus.RUNNING,
                reason="dispatched to worker",
                hold=False,
                metadata_updates={
                    "placement_mode": plan.mode,
                    "placement_backend": plan.backend_name,
                    "placement_role": role,
                },
            )
            self.event_logger.emit(
                "job_dispatched",
                job_id=job.job_id,
                payload={
                    "priority": job.priority,
                    "placement_mode": plan.mode,
                    "placement_backend": plan.backend_name,
                    "job_ids": list(plan.job_ids),
                    "reason": plan.reason,
                },
            )
        if plan.mode == "packed_pair":
            self.event_logger.emit(
                "packed_pair_dispatched",
                payload={"job_ids": list(plan.job_ids), "backend_name": plan.backend_name, "reason": plan.reason},
            )

    def report(self) -> dict[str, Any]:
        return self.metrics.as_dict()

    def cache_stats(self) -> dict[str, Any]:
        return {
            "stats": self.cache.stats().to_dict(),
            "entries": self.cache.snapshot_entries(),
        }
