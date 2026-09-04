"""Main scheduler orchestration loop.

SchedulerService owns infrastructure and defines the order of each scheduler
tick. Focused mixins implement run tracking, placement replay, live colocation
trials, and dispatch so this file remains a readable map of the overall flow.
"""

from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
from typing import Any
import json
import os
import time

from ..model_cache.baseline_cache import BaselineModelCache, CachedModelEntry
from ..model_cache.cache_server import CacheServer
from ..model_cache.warming import select_models_to_warm
from ..observability.events import EventLogger
from ..observability.logging_utils import setup_scheduler_logger
from ..observability.metrics import MetricsCollector
from ..domain import JobStatus, PreloadSource, TrainingJob, parse_timestamp, utc_now
from ..config import SchedulerSettings
from ..storage.log_store import SchedulerLogStore
from ..storage.state_store import StateStore
from .colocation_decisions import ColocationDecisionMixin
from .colocation_evidence import ColocationEvidenceMixin
from .colocation_trials import ColocationTrialMixin
from .dispatching import DispatchMixin
from .placement_replay import PlacementReplayMixin
from .placement_planner import PlacementPlanner
from .policies import PriorityFifoPolicy, SchedulingPolicy
from .recovery import reconcile_recoverable_jobs
from .run_tracking import RunTrackingMixin
from .service_state import (
    ActiveRun,
    ColocationStallState,
    ColocationTrialState,
    PlacementPatternObservation,
    PlacementProfileSnapshot,
    PlacementReplayState,
    PlacementReplayTemplate,
)
from .supervisor import WorkerSupervisor
from .telemetry import (
    GpuTelemetrySample,
    MemoryAdmissionGate,
    NvidiaSmiTelemetrySampler,
)


class SchedulerService(
    RunTrackingMixin,
    DispatchMixin,
    PlacementReplayMixin,
    ColocationEvidenceMixin,
    ColocationDecisionMixin,
    ColocationTrialMixin,
):
    """Coordinate one scheduling tick at a time.

    The coordinator owns shared infrastructure and durable decision state.
    Responsibility-specific mixins keep policy learning and execution details
    out of the loop while preserving the established SchedulerService API.
    """

    def __init__(
        self,
        settings: SchedulerSettings,
        *,
        store: StateStore | None = None,
        policy: SchedulingPolicy | None = None,
        supervisor: WorkerSupervisor | None = None,
        telemetry_sampler: NvidiaSmiTelemetrySampler | None = None,
    ):
        """Wire scheduler infrastructure and restore durable decision state."""
        self.settings = settings
        self.settings.ensure_runtime_layout()
        self.store = store or StateStore(settings)
        self.logger = setup_scheduler_logger(settings.scheduler_log_path)
        self.log_store = SchedulerLogStore(settings)
        self.event_logger = EventLogger(
            self.store, settings.events_jsonl_path, log_store=self.log_store
        )
        self.metrics = MetricsCollector(self.store)
        self.policy = policy or PriorityFifoPolicy(
            aging_interval_seconds=settings.aging_interval_seconds,
            aging_priority_increment=settings.aging_priority_increment,
            enable_priority_aging=settings.enable_priority_aging,
        )
        self.supervisor = supervisor or WorkerSupervisor(settings, store=self.store)
        self.planner = PlacementPlanner(settings, self.store, self.policy)
        self.telemetry_sampler = telemetry_sampler or NvidiaSmiTelemetrySampler(
            settings.gpu_scheduler.device_index
        )
        self.cache = BaselineModelCache(
            settings.baseline_cache.memory_budget_bytes,
            entry_capacity=settings.baseline_cache.entry_capacity,
            max_ram_percent=settings.baseline_cache.max_ram_percent,
            on_update=self._on_cache_update,
        )
        self.cache_server = CacheServer(settings, self.cache)
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._active_runs: dict[str, ActiveRun] = {}
        self._device_samples: list[GpuTelemetrySample] = []
        self._last_telemetry_poll_at = 0.0
        self._next_gpu_dispatch_attempt_at = 0.0
        memory = settings.gpu_scheduler.memory
        self._admission_gate = MemoryAdmissionGate(
            stop_fraction=memory.live_admission_stop_fraction,
            resume_fraction=memory.live_admission_resume_fraction,
            window_seconds=memory.admission_average_window_seconds,
        )
        self._exclusive_probe_job_id: str | None = None
        self._colocation_trial: ColocationTrialState | None = None
        self._colocation_stall: ColocationStallState | None = None
        self._placement_replay = PlacementReplayState()
        self._restore_scheduler_decision_state()

    @property
    def _decision_state_path(self) -> Path:
        """Return the file used to persist in-flight scheduling decisions."""
        return self.settings.runtime_root / "scheduler_decision_state.json"

    def _persist_scheduler_decision_state(self) -> None:
        """Atomically save admission, trial, stall, and replay state."""
        payload = {
            "admission_open": self._admission_gate.is_open,
            "admission_average_fraction": self._admission_gate.average_fraction,
            "admission_below_resume_since": (
                self._admission_gate.below_resume_since.isoformat()
                if self._admission_gate.below_resume_since is not None
                else None
            ),
            "admission_samples": [
                {
                    "captured_at": sample.captured_at,
                    "memory_used_mb": sample.memory_used_mb,
                    "memory_total_mb": sample.memory_total_mb,
                    "gpu_utilization": sample.gpu_utilization,
                    "memory_utilization": sample.memory_utilization,
                }
                for sample in self._admission_gate.samples
            ],
            "exclusive_probe_job_id": self._exclusive_probe_job_id,
            "colocation_trial": (
                self._colocation_trial.to_dict() if self._colocation_trial else None
            ),
            "colocation_stall": (
                self._colocation_stall.to_dict() if self._colocation_stall else None
            ),
            "placement_replay": self._placement_replay.to_dict(),
            "updated_at": utc_now(),
        }
        tmp_path = self._decision_state_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        tmp_path.replace(self._decision_state_path)

    def _restore_scheduler_decision_state(self) -> None:
        """Restore safe decision state while tolerating stale/corrupt files."""
        if not self._decision_state_path.exists():
            return
        try:
            with self._decision_state_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self._admission_gate.is_open = bool(payload.get("admission_open", True))
            average = payload.get("admission_average_fraction")
            self._admission_gate.average_fraction = (
                float(average) if average is not None else None
            )
            self._admission_gate.below_resume_since = parse_timestamp(
                payload.get("admission_below_resume_since")
            )
            self._admission_gate.samples = [
                GpuTelemetrySample(**sample)
                for sample in payload.get("admission_samples", [])
                if isinstance(sample, dict)
            ]
            reserved = payload.get("exclusive_probe_job_id")
            if self.settings.gpu_scheduler.exclusive_probe.enabled:
                self._exclusive_probe_job_id = str(reserved) if reserved else None
            else:
                # A reservation is meaningful only while exclusive-probe
                # draining is enabled.  Do not let stale restart state close
                # normal admission after the feature has been turned off.
                self._exclusive_probe_job_id = None
            trial = payload.get("colocation_trial")
            self._colocation_trial = (
                ColocationTrialState.from_dict(trial)
                if isinstance(trial, dict)
                else None
            )
            stall = payload.get("colocation_stall")
            self._colocation_stall = (
                ColocationStallState.from_dict(stall)
                if isinstance(stall, dict)
                else None
            )
            replay = payload.get("placement_replay")
            if (
                self.settings.gpu_scheduler.colocation.decision_replay.enabled
                and isinstance(replay, dict)
            ):
                self._placement_replay = PlacementReplayState.from_dict(replay)
            else:
                self._placement_replay = PlacementReplayState()
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.logger.warning("Ignoring invalid scheduler decision state: %s", exc)

    def _persist_runtime_settings(self) -> None:
        """Publish resolved settings for worker and replay processes."""
        path = self.settings.runtime_root / "scheduler_settings.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.settings.to_dict(), handle, indent=2, sort_keys=True)

    def _write_service_heartbeat(self, status: str) -> None:
        """Persist scheduler liveness and its current lifecycle status."""
        payload = {
            "pid": os.getpid(),
            "status": status,
            "updated_at": utc_now(),
            "runtime_root": str(self.settings.runtime_root),
        }
        path = self.settings.service_heartbeat_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        tmp_path.replace(path)

    def _on_cache_update(
        self, event_name: str, entry: CachedModelEntry, payload: dict[str, Any] | None
    ) -> None:
        """Mirror a model-cache change into durable state and the event log."""
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
        self.event_logger.emit(
            event_name,
            payload={
                "model_id": entry.model_id,
                **(payload or {}),
                **entry.to_stats_dict(),
            },
        )

    def start(self, *, background: bool = False) -> "SchedulerService":
        """Initialize runtime services and run the scheduler in the requested mode."""
        self._persist_runtime_settings()
        self._write_service_heartbeat("starting")
        self.log_store.start_session(
            status="starting",
            pid=os.getpid(),
            runtime_root=str(self.settings.runtime_root),
            host_identity=self.store.hardware_profile().to_dict(),
            config_json=self.settings.to_dict(),
            started_at=utc_now(),
        )
        self.cache_server.start()
        reconcile_recoverable_jobs(
            self.store,
            self.event_logger,
            auto_resume=self.settings.auto_resume_recoverable,
        )
        if background:
            if self._thread is not None and self._thread.is_alive():
                return self
            self._thread = Thread(
                target=self.run_forever, name="scheduler-service", daemon=True
            )
            self._thread.start()
            return self
        self.run_forever()
        return self

    def stop(self) -> None:
        """Stop the loop, workers, cache server, and active log session."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self.supervisor.shutdown()
        self.cache_server.stop()
        self._write_service_heartbeat("stopped")
        self.log_store.finish_session(status="stopped", stopped_at=utc_now())

    def run_forever(self) -> None:
        """Run scheduler ticks until stop is requested.

        The ordering is intentional: observe worker exits and trial evidence
        first, apply user commands next, then update resource signals before
        choosing new work. A dispatch therefore always uses the freshest
        state gathered during the current tick.
        """
        self.logger.info("Scheduler service started")
        while not self._stop_event.is_set():
            # 1. Observe the world before making decisions. Worker exits can
            # close groups, and fresh epoch evidence can resolve a live trial.
            self._write_service_heartbeat("running")
            self._poll_active_workers()
            self._evaluate_colocation_trial()

            # 2. Apply external intent and refresh cache/resource signals.
            self._process_commands()
            self._warm_cache()
            self._poll_telemetry()

            # 3. Reconsider the active placement, then fill available slots.
            self._maybe_preempt()
            self._dispatch_pending_work()
            self._stop_event.wait(self.settings.scheduler_poll_interval_seconds)
        self._write_service_heartbeat("stopped")
        self.logger.info("Scheduler service stopped")

    def _process_commands(self) -> None:
        """Drain pending user commands and route each one to its handler."""
        commands = self.store.fetch_pending_commands(
            limit=self.settings.command_poll_limit
        )
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
        """Move a submitted job into the runnable queue."""
        if job_id is None:
            return
        job = self.store.get_job(job_id)
        if job is None or job.status.is_terminal:
            return
        # Submission and scheduling run on different threads.  A newly saved
        # PENDING job can be dispatched between command polling and this
        # handler; never demote that RUNNING job back to READY when its queued
        # SUBMIT command is observed on the next tick.
        if job.status == JobStatus.PENDING:
            self.store.set_job_status(
                job_id, JobStatus.READY, reason="job accepted by scheduler", hold=False
            )
        elif job.status != JobStatus.READY:
            return
        self.event_logger.emit(
            "job_ready", job_id=job_id, payload={"priority": job.priority}
        )

    def _handle_pause(self, job_id: str | None) -> None:
        """Request a cooperative worker pause and persist the transition."""
        if job_id is None:
            return
        job = self.store.get_job(job_id)
        if job is None or job.status.is_terminal:
            return
        if self.supervisor.request_pause(
            job_id, reason="manual pause requested", hold=True
        ):
            self.store.set_job_status(
                job_id, JobStatus.PAUSING, reason="manual pause requested", hold=True
            )
            self.event_logger.emit(
                "pause_requested", job_id=job_id, payload={"hold": True}
            )
            return
        self.store.set_job_status(
            job_id, JobStatus.PAUSED, reason="manual pause while queued", hold=True
        )
        self.event_logger.emit(
            "job_paused", job_id=job_id, payload={"hold": True, "queued": True}
        )

    def _handle_resume(self, job_id: str | None) -> None:
        """Release a paused job and make it eligible for scheduling again."""
        if job_id is None:
            return
        job = self.store.get_job(job_id)
        if job is None or job.status.is_terminal:
            return
        if job.status in {
            JobStatus.PAUSED,
            JobStatus.RECOVERABLE,
            JobStatus.PENDING,
            JobStatus.READY,
        }:
            self.store.set_job_status(
                job_id, JobStatus.READY, reason="resume requested", hold=False
            )
            self.event_logger.emit("job_resume_requested", job_id=job_id, payload={})

    def _handle_cancel(self, job_id: str | None) -> None:
        """Cancel queued or active work and persist the terminal state."""
        if job_id is None:
            return
        job = self.store.get_job(job_id)
        if job is None or job.status.is_terminal:
            return
        if self.supervisor.request_cancel(job_id, reason="cancel requested"):
            self.store.update_job(job_id, reason="cancel requested", hold=True)
            self.event_logger.emit("cancel_requested", job_id=job_id, payload={})
            return
        self.store.set_job_status(
            job_id, JobStatus.CANCELLED, reason="cancelled while queued", hold=True
        )
        self.event_logger.emit("job_cancelled", job_id=job_id, payload={"queued": True})

    def _handle_preload(self, payload: dict[str, Any]) -> None:
        """Load the baseline model requested by a preload command."""
        target = PreloadSource(
            model_id=payload.get("model_id") or payload["baseline_model_id"],
            model_path=payload.get("model_path") or payload["baseline_model_path"],
            loader_target=payload.get("loader_target"),
        )
        pin = bool(payload.get("pin", False))
        ok = self.cache.preload(
            target.model_id,
            target.model_path,
            loader_target=target.loader_target,
            pin=pin,
            metadata={"source": "command"},
        )
        self.event_logger.emit(
            "cache_preload_requested",
            payload={"model_id": target.model_id, "ok": ok, "pin": pin},
        )

    def _runnable_jobs(self) -> list[TrainingJob]:
        """Return policy-ordered work that is currently eligible to run."""
        jobs = self.store.runnable_jobs()
        if not self.settings.auto_resume_recoverable:
            jobs = [job for job in jobs if job.status != JobStatus.RECOVERABLE]
        return jobs

    def _resolve_preload_target(self, job: TrainingJob) -> PreloadSource:
        """Resolve a job's model reference into a cache preload target."""
        if job.preload_source is not None:
            return job.preload_source
        return PreloadSource(
            model_id=job.baseline_model_id,
            model_path=job.baseline_model_path,
            loader_target=job.config.loader_target,
        )

    def _warm_cache(self) -> None:
        """Preload likely baseline models while respecting cache capacity."""
        jobs = self._runnable_jobs()
        cache_stats = self.cache.stats()
        cached_model_ids = {
            entry["model_id"] for entry in self.cache.snapshot_entries()
        }
        available_budget_bytes = None
        if cache_stats.effective_memory_budget_bytes is not None:
            available_budget_bytes = max(
                0,
                int(cache_stats.effective_memory_budget_bytes)
                - int(cache_stats.used_bytes),
            )
        for target in select_models_to_warm(
            jobs,
            top_k=self.settings.baseline_cache.warm_queue_top_k,
            selection_policy=self.settings.baseline_cache.warm_queue_policy,
            available_budget_bytes=available_budget_bytes,
            cached_model_ids=cached_model_ids,
            resolve_target=self._resolve_preload_target,
        ):
            try:
                self.cache.preload(
                    target.model_id,
                    target.model_path,
                    loader_target=target.loader_target,
                    metadata={"source": "warming"},
                )
            except Exception as exc:
                self.logger.warning(
                    "Cache warming failed for %s: %s", target.model_id, exc
                )

    def _poll_telemetry(self) -> None:
        """Sample GPU telemetry and update memory admission evidence."""
        if not self._active_runs:
            return
        interval_seconds = max(
            0.1, self.settings.gpu_scheduler.telemetry.device_poll_ms / 1000.0
        )
        now = time.monotonic()
        if (now - self._last_telemetry_poll_at) < interval_seconds:
            return
        sample = self.telemetry_sampler.sample()
        self._last_telemetry_poll_at = now
        if sample is None:
            return
        self._device_samples.append(sample)
        transition = self._admission_gate.update(sample)
        self._persist_scheduler_decision_state()
        if transition is not None:
            self.event_logger.emit(
                "packed_admission_gate_changed",
                payload={
                    "state": transition,
                    "average_memory_fraction": self._admission_gate.average_fraction,
                    "stop_fraction": self._admission_gate.stop_fraction,
                    "resume_fraction": self._admission_gate.resume_fraction,
                },
            )
        if len(self._active_runs) == 1:
            only_run = next(iter(self._active_runs.values()))
            only_run.samples.append(sample)
            self.log_store.record_gpu_metric_sample(
                group_id=only_run.group_id,
                created_at=sample.captured_at,
                backend_name=only_run.backend_name,
                hardware_key=only_run.hardware_key or self.store.hardware_key(),
                memory_used_mb=sample.memory_used_mb,
                memory_total_mb=sample.memory_total_mb,
                gpu_utilization=sample.gpu_utilization,
                memory_utilization=sample.memory_utilization,
                job_ids=list(only_run.job_ids),
            )
        else:
            for run in self._active_runs.values():
                run.overlapped = True
                self.log_store.record_gpu_metric_sample(
                    group_id=run.group_id,
                    created_at=sample.captured_at,
                    backend_name=run.backend_name,
                    hardware_key=run.hardware_key or self.store.hardware_key(),
                    memory_used_mb=sample.memory_used_mb,
                    memory_total_mb=sample.memory_total_mb,
                    gpu_utilization=sample.gpu_utilization,
                    memory_utilization=sample.memory_utilization,
                    job_ids=list(run.job_ids),
                )

    def report(self) -> dict[str, Any]:
        """Build the current scheduler, queue, worker, and cache status report."""
        return {
            **self.metrics.as_dict(),
            "packed_admission_open": self._admission_gate.is_open,
            "average_memory_fraction": self._admission_gate.average_fraction,
            "exclusive_drain_requested": self._exclusive_probe_job_id is not None,
            "reserved_exclusive_probe_job_id": self._exclusive_probe_job_id,
            "packing_admission_stalled": self._colocation_stall is not None,
            "colocation_stall": (
                self._colocation_stall.to_dict() if self._colocation_stall else None
            ),
            "colocation_trial": (
                self._colocation_trial.to_dict() if self._colocation_trial else None
            ),
            "placement_replay_active": self._placement_replay.template is not None,
            "placement_replay": self._placement_replay.to_dict(),
        }

    def cache_stats(self) -> dict[str, Any]:
        """Return cache statistics with scheduler-specific context."""
        return {
            "stats": self.cache.stats().to_dict(),
            "entries": self.cache.snapshot_entries(),
        }
