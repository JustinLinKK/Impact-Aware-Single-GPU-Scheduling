"""Scheduler service loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
from typing import Any
import json
import os
import time
import uuid

from ..atomic_io import atomic_json_dump
from ..model_cache.baseline_cache import BaselineModelCache, CachedModelEntry
from ..model_cache.cache_server import CacheServer
from ..model_cache.warming import select_models_to_warm
from ..observability.events import EventLogger
from ..observability.logging_utils import setup_scheduler_logger
from ..observability.metrics import MetricsCollector
from ..profiling.runtime_probe import runtime_profile_for_job, successful_runtime_profile_for_packing
from ..domain import BatchProbeSpec, BatchResolution, CombinationProfile, JobStatus, PackingSpec, PairProfile, PreloadSource, ProfileState, ResourceRequirements, RuntimeProbeSpec, SoloProfile, TrainingJob, build_group_signature, parse_timestamp, utc_now
from ..config import PREDICTION_MODE_BRANCH_PROFILE, SchedulerSettings
from ..storage.log_store import SchedulerLogStore
from ..storage.state_store import StateStore
from .placement_planner import PlacementPlanner
from .planner_types import DispatchPlan
from .policies import PriorityFifoPolicy, SchedulingPolicy
from .early_stop import EarlyStopDecision, analyze_metric_plateau
from .recovery import reconcile_recoverable_jobs
from .supervisor import WorkerSnapshot, WorkerSupervisor
from .telemetry import GpuTelemetrySample, GpuTelemetrySummary, NvidiaSmiTelemetrySampler
from .training_plot import render_training_process


RAW_MLEVOLVE_RUNNER_TARGET = "localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job"
NON_PREEMPTIBLE_PROBE_TASK_TYPES = {
    "mlevolve_model_family_probe",
    "mlevolve_startpoint_probe",
    "mlevolve_branch_profile_probe",
}
EVENT_CANDIDATE_SAMPLE_LIMIT = 8
EVENT_JOB_ID_SAMPLE_LIMIT = 12


@dataclass(slots=True)
class ActiveRun:
    group_id: str
    mode: str
    backend_name: str
    job_ids: tuple[str, ...]
    opened_at: str = field(default_factory=utc_now)
    batch_overrides: dict[str, int] = field(default_factory=dict)
    fallback_order: list[str] = field(default_factory=list)
    hardware_key: str = ""
    group_signature: str = ""
    samples: list[GpuTelemetrySample] = field(default_factory=list)
    fallback_triggered: bool = False
    fallback_reason: str | None = None
    overlapped: bool = False
    repack_transaction: PendingRepack | None = None


@dataclass(slots=True)
class PendingRepack:
    transaction_id: str
    target_plan: DispatchPlan
    rollback_plan: DispatchPlan
    active_job_ids: tuple[str, ...]
    prior_batch_sizes: dict[str, int]
    requested_at_monotonic: float
    phase: str = "preparing"


class SchedulerService:
    """Single-process scheduler with optional pairwise packed execution."""

    def __init__(
        self,
        settings: SchedulerSettings,
        *,
        store: StateStore | None = None,
        policy: SchedulingPolicy | None = None,
        supervisor: WorkerSupervisor | None = None,
        telemetry_sampler: NvidiaSmiTelemetrySampler | None = None,
    ):
        self.settings = settings
        self.settings.ensure_runtime_layout()
        self.settings.scheduler_session_id = uuid.uuid4().hex
        self.store = store or StateStore(settings)
        self.logger = setup_scheduler_logger(settings.scheduler_log_path)
        self.log_store = SchedulerLogStore(settings)
        self.event_logger = EventLogger(self.store, settings.events_jsonl_path, log_store=self.log_store)
        self.metrics = MetricsCollector(self.store)
        self.policy = policy or PriorityFifoPolicy(
            aging_interval_seconds=settings.aging_interval_seconds,
            aging_priority_increment=settings.aging_priority_increment,
            enable_priority_aging=settings.enable_priority_aging,
        )
        self.supervisor = supervisor or WorkerSupervisor(settings, store=self.store)
        self._configure_adaptive_backend_policy()
        self.planner = PlacementPlanner(settings, self.store, self.policy)
        self.telemetry_sampler = telemetry_sampler or NvidiaSmiTelemetrySampler(settings.gpu_scheduler.device_index)
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
        self._last_adaptive_replan_at = 0.0
        self._profile_drain_latched = False
        self._profile_probe_jobs: dict[str, str] = {}
        self._pending_repack: PendingRepack | None = None
        self._event_throttle_last_emitted: dict[tuple[Any, ...], float] = {}
        self.event_logger.emit(
            "scheduler_session_started",
            payload={
                "scheduler_session_id": self.settings.scheduler_session_id,
                "runtime_root": str(self.settings.runtime_root),
            },
        )

    def _configure_adaptive_backend_policy(self) -> None:
        gpu = self.settings.gpu_scheduler
        availability = self.supervisor.available_backends()
        priority: list[str] = []
        for backend_name in gpu.backend_priority:
            if backend_name != "exclusive" and backend_name not in priority and availability.get(backend_name):
                priority.append(backend_name)
        priority.append("exclusive")
        gpu.backend_priority = priority
        event_payload = {
            "configured_mode": gpu.mode,
            "effective_scheduler_mode": gpu.mode,
            "backend_availability": availability,
            "backend_priority": list(priority),
        }
        self.event_logger.emit("scheduler_adaptive_backend_probe", payload=event_payload)
        self.logger.info(
            "Adaptive scheduler backend probe resolved mode=%s priority=%s availability=%s",
            event_payload["effective_scheduler_mode"],
            event_payload["backend_priority"],
            event_payload["backend_availability"],
        )

    def _persist_runtime_settings(self) -> None:
        path = self.settings.runtime_root / "scheduler_settings.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.settings.to_dict(), handle, indent=2, sort_keys=True)

    def _write_service_heartbeat(self, status: str) -> None:
        payload = {
            "pid": os.getpid(),
            "status": status,
            "updated_at": utc_now(),
            "runtime_root": str(self.settings.runtime_root),
        }
        path = self.settings.service_heartbeat_path
        atomic_json_dump(path, payload, indent=2, sort_keys=True)

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
        self._write_service_heartbeat("stopped")
        self.log_store.finish_session(status="stopped", stopped_at=utc_now())

    def run_forever(self) -> None:
        self.logger.info("Scheduler service started")
        while not self._stop_event.is_set():
            self._write_service_heartbeat("running")
            self._poll_active_workers()
            self._process_commands()
            self._warm_cache()
            self._poll_telemetry()
            self._enforce_packed_safety()
            self._maybe_early_stop()
            self._maybe_preempt()
            self._dispatch_pending_work()
            self._stop_event.wait(self.settings.scheduler_poll_interval_seconds)
        self._write_service_heartbeat("stopped")
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
        if self.settings.prediction.mode == PREDICTION_MODE_BRANCH_PROFILE and job.batch_probe.enabled:
            if not job.batch_probe.profile_namespace:
                job.batch_probe.profile_namespace = job.packing.signature or f"branch-profile:{job.baseline_model_id}"
            if job.task_type in NON_PREEMPTIBLE_PROBE_TASK_TYPES or job.task_type == "mlevolve_branch_profile_probe":
                job.profile_state = ProfileState.PROBING
            elif self.planner.profile_ready(job):
                job.profile_state = ProfileState.READY
            else:
                job.profile_state = ProfileState.WAITING_FOR_DRAIN
                self._profile_drain_latched = True
            self.store.save_job(job)
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
        self.event_logger.emit("cache_preload_requested", payload={"model_id": target.model_id, "ok": ok, "pin": pin})

    def _runnable_jobs(self) -> list[TrainingJob]:
        jobs = self.store.runnable_jobs()
        if not self.settings.auto_resume_recoverable:
            jobs = [job for job in jobs if job.status != JobStatus.RECOVERABLE]
        return jobs

    def _resolve_preload_target(self, job: TrainingJob) -> PreloadSource:
        if job.preload_source is not None:
            return job.preload_source
        return PreloadSource(
            model_id=job.baseline_model_id,
            model_path=job.baseline_model_path,
            loader_target=job.config.loader_target,
        )

    def _warm_cache(self) -> None:
        jobs = self._runnable_jobs()
        cache_stats = self.cache.stats()
        cached_model_ids = {entry["model_id"] for entry in self.cache.snapshot_entries()}
        available_budget_bytes = None
        if cache_stats.effective_memory_budget_bytes is not None:
            available_budget_bytes = max(0, int(cache_stats.effective_memory_budget_bytes) - int(cache_stats.used_bytes))
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
                self.logger.warning("Cache warming failed for %s: %s", target.model_id, exc)

    def _poll_telemetry(self) -> None:
        if not self._active_runs:
            return
        interval_seconds = max(0.1, self.settings.gpu_scheduler.telemetry.device_poll_ms / 1000.0)
        now = time.monotonic()
        if (now - self._last_telemetry_poll_at) < interval_seconds:
            return
        sample = self.telemetry_sampler.sample()
        self._last_telemetry_poll_at = now
        if sample is None:
            return
        self._device_samples.append(sample)
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

    def _pick_fallback_candidate(self) -> tuple[str, str] | None:
        candidates: list[tuple[int, float, int, str, str]] = []
        for group_id, run in self._active_runs.items():
            if run.mode == "exclusive" or len(run.job_ids) < 2:
                continue
            for job_id in self._supervisor_active_job_ids_by_group().get(group_id, []):
                job = self.store.get_job(job_id)
                if job is None:
                    continue
                remaining_runtime = self.planner.predicted_remaining_runtime_seconds(job, backend_name=run.backend_name) or 0.0
                candidates.append((job.priority, -remaining_runtime, -job.queue_sequence, group_id, job_id))
        if not candidates:
            return None
        _, _, _, group_id, job_id = sorted(candidates)[0]
        return group_id, job_id

    def _enforce_packed_safety(self) -> None:
        if not self._active_runs or not self._device_samples:
            return
        latest = self._device_samples[-1]
        if latest.memory_total_mb <= 0:
            return
        safe_budget_mb = self.settings.gpu_scheduler.memory.budget_mb(latest.memory_total_mb)
        if safe_budget_mb <= 0:
            return
        memory_fraction = latest.memory_used_mb / latest.memory_total_mb
        if latest.memory_used_mb < safe_budget_mb:
            return
        target = self._pick_fallback_candidate()
        if target is None:
            return
        group_id, target_job_id = target
        reason = (
            f"packed groups exceeded VRAM budget "
            f"({latest.memory_used_mb:.0f} MiB/{latest.memory_total_mb:.0f} MiB, {memory_fraction:.2%})"
        )
        if not self.supervisor.request_fallback_pause(target_job_id, reason=reason):
            return
        self.store.set_job_status(target_job_id, JobStatus.PAUSING, reason=reason, hold=False)
        run = self._active_runs.get(group_id)
        if run is not None:
            self._register_packed_fallback(
                run,
                reason,
                payload={
                    "paused_job_id": target_job_id,
                    "memory_used_mb": latest.memory_used_mb,
                    "memory_total_mb": latest.memory_total_mb,
                },
            )

    def _poll_active_workers(self) -> None:
        snapshots = self.supervisor.poll()
        if not snapshots:
            return
        for snapshot in snapshots:
            run = self._active_runs.get(snapshot.group_id)
            self._handle_worker_exit(snapshot, run_context=run)

        remaining_by_group = self._supervisor_active_job_ids_by_group()
        for group_id, run in list(self._active_runs.items()):
            remaining_job_ids = remaining_by_group.get(group_id, [])
            if len(run.job_ids) > 1 and len(remaining_job_ids) < len(run.job_ids):
                self._record_combination_profiles(run)
            elif len(run.job_ids) == 1 and not remaining_job_ids:
                self._record_solo_profiles(run)

            if not remaining_job_ids:
                self.log_store.close_run_group(
                    group_id=group_id,
                    closed_at=utc_now(),
                    overlapped=run.overlapped,
                    fallback_triggered=run.fallback_triggered,
                    fallback_reason=run.fallback_reason,
                    exit_reason=run.fallback_reason or "group_complete",
                )
                self._active_runs.pop(group_id, None)
                continue
            if tuple(remaining_job_ids) != run.job_ids:
                removed_job_ids = [job_id for job_id in run.job_ids if job_id not in remaining_job_ids]
                for removed_job_id in removed_job_ids:
                    self.log_store.mark_run_group_member_left(group_id=group_id, job_id=removed_job_id, left_at=utc_now())
                if len(remaining_job_ids) == 1:
                    run.mode = "exclusive"
                run.job_ids = tuple(remaining_job_ids)
                run.fallback_order = [job_id for job_id in run.fallback_order if job_id in remaining_job_ids]
                run.group_signature = build_group_signature(
                    [
                        (self.store.get_job(job_id).packing.signature or job_id)
                        for job_id in remaining_job_ids
                        if self.store.get_job(job_id) is not None
                    ]
                )

    def _handle_worker_exit(self, snapshot: WorkerSnapshot, *, run_context: ActiveRun | None) -> None:
        job = self.store.get_job(snapshot.job_id)
        if job is None:
            return
        if (
            run_context is not None
            and run_context.repack_transaction is not None
            and self._worker_exit_indicates_oom(job, snapshot)
        ):
            self._rollback_launched_repack(run_context, failed_job_id=job.job_id, reason=job.status_reason or "OOM")
            return
        self._finalize_scheduler_preemption_on_exit(job)
        if snapshot.reported_by == "store":
            if run_context is not None and len(run_context.job_ids) > 1 and job.status == JobStatus.FAILED:
                self._register_packed_fallback(run_context, job.status_reason or "stream-backed worker failed", payload={"failed_job_id": snapshot.job_id})
            self._emit_worker_finished_event(snapshot, run_context=run_context)
            return
        if snapshot.returncode == 0:
            if job.status in {
                JobStatus.COMPLETED,
                JobStatus.PAUSED,
                JobStatus.EARLY_STOPPED,
                JobStatus.CANCELLED,
                JobStatus.READY,
                JobStatus.FAILED,
            }:
                self._emit_worker_finished_event(snapshot, run_context=run_context)
                return
            self.store.set_job_status(job.job_id, JobStatus.FAILED, reason="worker exited without terminal status update", hold=True)
            self.event_logger.emit("job_failed", job_id=job.job_id, payload={"reason": "worker exited cleanly without terminal status"})
            self._emit_worker_finished_event(snapshot, run_context=run_context)
            return

        if not job.status.is_terminal:
            reason = f"worker exited with code {snapshot.returncode}"
            self.store.set_job_status(job.job_id, JobStatus.FAILED, reason=reason, hold=True)
            self.event_logger.emit("job_failed", job_id=job.job_id, payload={"returncode": snapshot.returncode})
            if run_context is not None and len(run_context.job_ids) > 1:
                self._register_packed_fallback(run_context, reason, payload={"failed_job_id": snapshot.job_id, "returncode": snapshot.returncode})
            self._emit_worker_finished_event(snapshot, run_context=run_context)
            return

        if run_context is not None and len(run_context.job_ids) > 1 and job.status == JobStatus.FAILED:
            reason = job.status_reason or f"worker exited with code {snapshot.returncode}"
            self._register_packed_fallback(run_context, reason, payload={"failed_job_id": snapshot.job_id, "returncode": snapshot.returncode})
        self._emit_worker_finished_event(snapshot, run_context=run_context)

    def _finalize_scheduler_preemption_on_exit(self, job: TrainingJob) -> None:
        if not bool(job.metadata.get("scheduler_preemption_pending")):
            return
        if job.status == JobStatus.PAUSED:
            checkpoint_path = job.latest_checkpoint_path or self.store.latest_checkpoint(job.job_id)
            if checkpoint_path:
                completed_at = utc_now()
                self.store.update_job(
                    job.job_id,
                    metadata_updates={
                        "scheduler_preemption_pending": False,
                        "scheduler_preemption_completed_at": completed_at,
                        "scheduler_preemption_checkpoint_path": checkpoint_path,
                    },
                )
                self.event_logger.emit(
                    "scheduler_preemption_completed",
                    job_id=job.job_id,
                    payload={
                        "checkpoint_path": checkpoint_path,
                        "completed_at": completed_at,
                        "strategy": job.metadata.get("scheduler_preemption_strategy"),
                        "preempting_job_ids": list(job.metadata.get("scheduler_preemption_preempting_job_ids") or []),
                    },
                )
                return
            failed_at = utc_now()
            self.store.update_job(
                job.job_id,
                reason="scheduler preemption paused without checkpoint",
                hold=True,
                metadata_updates={
                    "scheduler_preemption_pending": False,
                    "scheduler_preemption_failed_at": failed_at,
                },
            )
            self._emit_scheduler_preemption_skipped(
                job,
                reason="preempted job paused without a checkpoint",
                payload={"failed_at": failed_at},
            )
            return
        if job.status in {JobStatus.FAILED, JobStatus.CANCELLED}:
            failed_at = utc_now()
            self.store.update_job(
                job.job_id,
                metadata_updates={
                    "scheduler_preemption_pending": False,
                    "scheduler_preemption_failed_at": failed_at,
                },
            )
            self._emit_scheduler_preemption_skipped(
                job,
                reason=f"preempted job exited with status {job.status.value}",
                payload={"failed_at": failed_at},
            )

    def _register_packed_fallback(self, run: ActiveRun, reason: str, *, payload: dict[str, Any]) -> None:
        if len(run.job_ids) < 2 or run.fallback_triggered:
            return
        run.fallback_triggered = True
        run.fallback_reason = reason
        self.event_logger.emit(
            "packed_group_fallback",
            payload={"job_ids": list(run.job_ids), "reason": reason, **payload},
        )

    def _batch_probe_profile_payload(self, job: TrainingJob) -> dict[str, Any] | None:
        probe_key = job.metadata.get("batch_probe_key")
        if not probe_key:
            return None
        profile = self.store.get_batch_probe_profile(str(probe_key))
        return profile.to_dict() if profile is not None else {"probe_key": probe_key}

    def _runtime_profile_payload(self, job: TrainingJob, *, backend_name: str) -> dict[str, Any] | None:
        try:
            profile = runtime_profile_for_job(self.store, job, backend_name=backend_name)
            if profile is None:
                profile = successful_runtime_profile_for_packing(
                    self.store,
                    job,
                    backend_name=backend_name,
                    scheduler_session_id=self.settings.scheduler_session_id,
                )
        except Exception:
            profile = None
        return profile.to_dict() if profile is not None else None

    def _artifact_paths(self, job: TrainingJob, *, stdout_path: Path | None = None, stderr_path: Path | None = None) -> dict[str, Any]:
        runner_kwargs = dict(job.config.runner_kwargs or {})
        paths: dict[str, Any] = {
            "runtime_dir": str(self.settings.job_runtime_dir(job.job_id)),
            "checkpoint_dir": str(self.settings.checkpoints_for_job(job.job_id)),
        }
        if stdout_path is not None:
            paths["stdout_path"] = str(stdout_path)
        if stderr_path is not None:
            paths["stderr_path"] = str(stderr_path)
        for key in ("script_path", "working_dir", "result_path"):
            if runner_kwargs.get(key) is not None:
                paths[key] = str(runner_kwargs[key])
        if job.latest_checkpoint_path:
            paths["latest_checkpoint_path"] = job.latest_checkpoint_path
        return paths

    def _last_event_payload(self, job_id: str, event_type: str) -> dict[str, Any] | None:
        events = self.store.list_events(job_id=job_id, event_type=event_type)
        if not events:
            return None
        return dict(events[-1].get("payload") or {})

    def _worker_handle(self, group_id: str, job_id: str):
        active_groups = getattr(self.supervisor, "active_groups", None)
        if active_groups is None:
            return None
        group = active_groups().get(group_id)
        if group is None:
            return None
        worker = group.workers.get(job_id)
        return worker.handle if worker is not None else None

    def _emit_worker_launch_events(self, *, group_id: str, run: ActiveRun, jobs: list[TrainingJob], reason: str) -> None:
        for job in jobs:
            handle = self._worker_handle(group_id, job.job_id)
            stdout_path = getattr(handle, "stdout_path", None)
            stderr_path = getattr(handle, "stderr_path", None)
            process = getattr(handle, "process", None)
            args = getattr(process, "args", []) if process is not None else []
            process_command = [str(item) for item in args] if isinstance(args, (list, tuple)) else [str(args)]
            payload = {
                "group_id": group_id,
                "scheduler_session_id": self.settings.scheduler_session_id,
                "job_ids": list(run.job_ids),
                "backend_name": run.backend_name,
                "placement_mode": run.mode,
                "placement_reason": reason,
                "placement_batch_size": run.batch_overrides.get(job.job_id),
                "batch_overrides": dict(run.batch_overrides),
                "fallback_order": list(run.fallback_order),
                "pid": getattr(process, "pid", None),
                "process_command": process_command,
                "stdout_path": str(stdout_path) if stdout_path is not None else None,
                "stderr_path": str(stderr_path) if stderr_path is not None else None,
                "artifact_paths": self._artifact_paths(job, stdout_path=stdout_path, stderr_path=stderr_path),
                "started_at": utc_now(),
                "packing_signature": job.packing.signature,
                "batch_probe_profile": self._batch_probe_profile_payload(job),
                "runtime_profile": self._runtime_profile_payload(job, backend_name=run.backend_name),
            }
            self.event_logger.emit("worker_launched", job_id=job.job_id, payload=payload)

    def _emit_worker_finished_event(self, snapshot: WorkerSnapshot, *, run_context: ActiveRun | None) -> None:
        job = self.store.get_job(snapshot.job_id)
        result_payload = self._last_event_payload(snapshot.job_id, "job_completed")
        failure_payload = self._last_event_payload(snapshot.job_id, "job_failed")
        stdout_path = snapshot.stdout_path
        stderr_path = snapshot.stderr_path
        payload = {
            "group_id": snapshot.group_id,
            "scheduler_session_id": self.settings.scheduler_session_id,
            "backend_name": run_context.backend_name if run_context is not None else None,
            "placement_mode": run_context.mode if run_context is not None else None,
            "job_ids": list(run_context.job_ids) if run_context is not None else [snapshot.job_id],
            "pid": snapshot.pid,
            "process_command": list(snapshot.process_command),
            "stdout_path": str(stdout_path) if stdout_path is not None else None,
            "stderr_path": str(stderr_path) if stderr_path is not None else None,
            "artifact_paths": self._artifact_paths(job, stdout_path=stdout_path, stderr_path=stderr_path) if job is not None else {},
            "ended_at": utc_now(),
            "exit_status": snapshot.returncode,
            "reported_by": snapshot.reported_by,
            "job_status": job.status.value if job is not None else None,
            "status_reason": job.status_reason if job is not None else None,
            "traceback": (failure_payload or {}).get("traceback"),
            "runner_result": result_payload,
            "failure": failure_payload,
        }
        self.event_logger.emit("worker_finished", job_id=snapshot.job_id, payload=payload)

    def _record_solo_profiles(self, run: ActiveRun) -> None:
        if run.overlapped:
            return
        summary = GpuTelemetrySummary.from_samples(run.samples)
        for job_id in run.job_ids:
            job = self.store.get_job(job_id)
            if job is None or not job.packing.signature:
                continue
            if not job.packing.eligible:
                continue
            if job.status in {JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.EARLY_STOPPED}:
                continue
            peak_vram_mb = summary.peak_vram_mb
            if peak_vram_mb is None:
                peak_vram_mb = job.resource_requirements.estimated_vram_mb
            self.store.upsert_solo_profile(
                SoloProfile(
                    signature=job.packing.signature,
                    hardware_key=run.hardware_key or self.store.hardware_key(),
                    family=job.packing.family,
                    peak_vram_mb=peak_vram_mb,
                    avg_gpu_utilization=summary.avg_gpu_utilization if summary.avg_gpu_utilization is not None else 0.0,
                    avg_memory_utilization=summary.avg_memory_utilization if summary.avg_memory_utilization is not None else 0.0,
                    sample_count=summary.sample_count,
                    last_job_id=job.job_id,
                    metadata={"source": "exclusive_run", "backend_name": run.backend_name},
                )
            )

    def _record_combination_profiles(self, run: ActiveRun) -> None:
        if len(run.job_ids) < 2 or run.overlapped:
            return
        jobs = [self.store.get_job(job_id) for job_id in run.job_ids]
        if any(job is None for job in jobs):
            return
        summary = GpuTelemetrySummary.from_samples(run.samples)
        materialized_jobs = [job for job in jobs if job is not None]
        group_signature = run.group_signature or build_group_signature([job.packing.signature or job.job_id for job in materialized_jobs])
        existing = self.store.best_combination_profile(
            group_signature=group_signature,
            hardware_key=run.hardware_key or self.store.hardware_key(),
            backend_name=run.backend_name,
            scheduler_mode=self.settings.gpu_scheduler.mode,
        )
        compatible = not run.fallback_triggered and all(job.status != JobStatus.FAILED for job in materialized_jobs)
        observed_throughputs = {
            job.job_id: self._metadata_float(job, "runtime_samples_per_second")
            for job in materialized_jobs
        }
        aggregate_samples_per_second = sum(value or 0.0 for value in observed_throughputs.values()) or None
        self.store.upsert_combination_profile(
            CombinationProfile.create(
                group_signature=group_signature,
                hardware_key=run.hardware_key or self.store.hardware_key(),
                backend_name=run.backend_name,
                scheduler_mode=self.settings.gpu_scheduler.mode,
                batch_vector=run.batch_overrides,
                compatible=compatible,
                observations=(existing.observations + 1) if existing else 1,
                peak_vram_mb=summary.peak_vram_mb,
                memory_total_mb=run.samples[-1].memory_total_mb if run.samples else None,
                avg_gpu_utilization=summary.avg_gpu_utilization,
                avg_memory_utilization=summary.avg_memory_utilization,
                avg_step_time_ms=None,
                objective_score=(summary.peak_vram_mb or 0)
                / max(1.0, self.settings.gpu_scheduler.memory.budget_mb(run.samples[-1].memory_total_mb if run.samples else None)),
                resolved_optimal=True,
                last_failure_reason=run.fallback_reason,
                fallback_order=run.fallback_order,
                metadata={
                    "backend_name": run.backend_name,
                    "job_ids": list(run.job_ids),
                    "aggregate_samples_per_second": aggregate_samples_per_second,
                    "per_job_samples_per_second": observed_throughputs,
                },
            )
        )
        if len(materialized_jobs) != 2:
            return
        left_job, right_job = materialized_jobs
        if not left_job.packing.signature or not right_job.packing.signature:
            return
        if not compatible:
            self.store.mark_pair_incompatible(
                left_job.packing.signature,
                right_job.packing.signature,
                backend_name=run.backend_name,
                reason=run.fallback_reason or "packed group failed",
                cooldown_seconds=self.settings.gpu_scheduler.fallback_cooldown_seconds,
                peak_vram_mb=summary.peak_vram_mb,
                avg_gpu_utilization=summary.avg_gpu_utilization,
                avg_memory_utilization=summary.avg_memory_utilization,
                metadata={"backend_name": run.backend_name},
            )
            return
        existing_pair = self.store.get_pair_profile(left_job.packing.signature, right_job.packing.signature, backend_name=run.backend_name)
        slowdown_ratios: list[float] = []
        for job in materialized_jobs:
            observed = observed_throughputs.get(job.job_id)
            standalone = self.planner.estimator.predicted_samples_per_second(
                job,
                run.batch_overrides.get(job.job_id, BatchResolution.resolved_batch_size(job)),
            )
            if observed and standalone and observed > 0:
                slowdown_ratios.append(max(1.0, float(standalone) / float(observed)))
        measured_slowdown = max(slowdown_ratios) if slowdown_ratios else None
        self.store.upsert_pair_profile(
            PairProfile.create(
                left_job.packing.signature,
                right_job.packing.signature,
                backend_name=run.backend_name,
                hardware_key=run.hardware_key or self.store.hardware_key(),
                compatible=True,
                observations=(existing_pair.observations + 1) if existing_pair else 1,
                peak_vram_mb=summary.peak_vram_mb,
                avg_gpu_utilization=summary.avg_gpu_utilization,
                avg_memory_utilization=summary.avg_memory_utilization,
                slowdown_ratio=measured_slowdown,
                cooldown_until=None,
                last_failure_reason=None,
                metadata={
                    "backend_name": run.backend_name,
                    "aggregate_samples_per_second": aggregate_samples_per_second,
                    "per_job_samples_per_second": observed_throughputs,
                },
            )
        )

    def _resolved_batch_size_for_job_id(self, job_id: str) -> int:
        job = self.store.get_job(job_id)
        if job is None:
            return 1
        return BatchResolution.resolved_batch_size(job)

    def _supervisor_active_job_ids(self) -> list[str]:
        if hasattr(self.supervisor, "active_job_ids"):
            return list(self.supervisor.active_job_ids())
        active_group = getattr(self.supervisor, "active_group", lambda: None)()
        if active_group is None:
            return []
        if hasattr(active_group, "active_job_ids"):
            return list(active_group.active_job_ids())
        workers = getattr(active_group, "workers", {}) or {}
        return list(workers.keys())

    def _supervisor_active_job_ids_by_group(self) -> dict[str, list[str]]:
        if hasattr(self.supervisor, "active_job_ids_by_group"):
            return {str(group_id): list(job_ids) for group_id, job_ids in self.supervisor.active_job_ids_by_group().items()}
        active_group = getattr(self.supervisor, "active_group", lambda: None)()
        if active_group is None:
            return {}
        group_id = str(getattr(active_group, "group_id", "legacy-active-group"))
        if hasattr(active_group, "active_job_ids"):
            return {group_id: list(active_group.active_job_ids())}
        workers = getattr(active_group, "workers", {}) or {}
        return {group_id: list(workers.keys())}

    def _apply_batch_override(self, job: TrainingJob, batch_size: int) -> TrainingJob:
        updated_job = BatchResolution.apply(job, batch_size)
        self.store.save_job(updated_job)
        return updated_job

    def _maybe_preempt(self) -> None:
        self._enforce_scheduler_preemption_timeouts()

    def _maybe_early_stop(self) -> None:
        early_stop_settings = self.settings.gpu_scheduler.early_stop
        if not bool(early_stop_settings.enabled):
            return
        for _group_id, _run, job in self._active_job_records():
            if job.status != JobStatus.RUNNING:
                continue
            if self._is_scheduler_protected_job(job):
                continue
            if bool(job.metadata.get("scheduler_early_stop_pending")) or bool(job.metadata.get("scheduler_early_stop_completed_at")):
                continue
            samples = self.store.list_job_metric_samples(job.job_id)
            if not samples:
                continue
            latest_sample = samples[-1]
            if int(latest_sample.global_step or 0) < int(early_stop_settings.min_global_step):
                continue
            runtime_seconds = self._seconds_since(job.last_dispatched_at or job.started_at)
            if runtime_seconds is not None and runtime_seconds < float(early_stop_settings.min_runtime_seconds):
                continue
            decision = analyze_metric_plateau(
                samples,
                metric_key=str(job.metadata.get("early_stop_metric_key") or early_stop_settings.metric_key or "") or None,
                direction=str(job.metadata.get("early_stop_direction") or early_stop_settings.direction),
                warmup_samples=int(early_stop_settings.warmup_samples),
                patience_samples=int(early_stop_settings.patience_samples),
                min_delta=float(early_stop_settings.min_delta),
            )
            if not decision.should_stop:
                continue
            if self._request_early_stop(job, decision=decision, samples=samples):
                return

    def _request_early_stop(
        self,
        job: TrainingJob,
        *,
        decision: EarlyStopDecision,
        samples: list[Any],
    ) -> bool:
        reason = f"early stop: {decision.reason}"
        artifact_payload: dict[str, Any] = {}
        if bool(self.settings.gpu_scheduler.early_stop.plot_enabled):
            try:
                artifact_payload = render_training_process(
                    samples,
                    self.settings.job_runtime_dir(job.job_id),
                    decision=decision,
                )
            except Exception as exc:
                artifact_payload = {"plot_error": str(exc)}
                self.logger.warning("Failed to render training process for %s: %s", job.job_id, exc)
        requested_at = utc_now()
        metadata_updates = {
            "scheduler_early_stop_pending": True,
            "scheduler_early_stop_requested_at": requested_at,
            "scheduler_early_stop_reason": reason,
            "scheduler_early_stop_decision": decision.to_dict(),
            "scheduler_early_stop_plot_path": artifact_payload.get("plot_path"),
            "scheduler_early_stop_summary_path": artifact_payload.get("summary_path"),
        }
        self.store.update_job(
            job.job_id,
            status=JobStatus.PAUSING,
            reason=reason,
            hold=True,
            metadata_updates=metadata_updates,
        )
        if not self.supervisor.request_early_stop(job.job_id, reason=reason):
            self.event_logger.emit(
                "scheduler_early_stop_skipped",
                job_id=job.job_id,
                payload={"reason": "early-stop request rejected by supervisor", "decision": decision.to_dict()},
            )
            return False
        payload = {
            "reason": reason,
            "decision": decision.to_dict(),
            "artifact_paths": {
                "plot_path": artifact_payload.get("plot_path"),
                "summary_path": artifact_payload.get("summary_path"),
            },
            "sample_count": len(samples),
            "hold": True,
        }
        self.event_logger.emit("scheduler_early_stop_requested", job_id=job.job_id, payload=payload)
        return True

    def _supports_safe_preemption(self, job: TrainingJob) -> bool:
        if not bool(getattr(job.checkpoint_policy, "preemptible", True)):
            return False
        if self._is_scheduler_protected_job(job):
            return False
        if job.config.runner_target == RAW_MLEVOLVE_RUNNER_TARGET or job.task_type == "mlevolve_script":
            return bool(job.metadata.get("elastic_contract_validated"))
        policy = job.checkpoint_policy
        return bool(job.resume_from_checkpoint or job.latest_checkpoint_path or policy.save_every_epoch or policy.save_every_n_steps)

    def _is_scheduler_protected_job(self, job: TrainingJob) -> bool:
        kind = str(job.metadata.get("kind") or "")
        return (
            job.task_type in NON_PREEMPTIBLE_PROBE_TASK_TYPES
            or job.task_type.endswith("_probe")
            or kind.endswith("_probe")
            or bool(job.metadata.get("exclusive_probe"))
        )

    def _active_job_records(self) -> list[tuple[str, ActiveRun, TrainingJob]]:
        active_by_group = self._supervisor_active_job_ids_by_group()
        records: list[tuple[str, ActiveRun, TrainingJob]] = []
        for group_id, run in self._active_runs.items():
            job_ids = active_by_group.get(group_id, list(run.job_ids))
            for job_id in job_ids:
                job = self.store.get_job(job_id)
                if job is not None:
                    records.append((group_id, run, job))
        return records

    def _has_pending_scheduler_preemption(self) -> bool:
        for _group_id, _run, job in self._active_job_records():
            if job.status == JobStatus.PAUSING or bool(job.metadata.get("scheduler_preemption_pending")):
                return True
        return False

    def _active_jobs_remain_after_excluding(self, excluded_job_ids: set[str]) -> bool:
        for _group_id, _run, job in self._active_job_records():
            if job.job_id not in excluded_job_ids:
                return True
        return False

    def _active_occupancy_excluding(self, excluded_job_ids: set[str]) -> tuple[float, float]:
        active_vram_mb = 0.0
        active_sm_utilization = 0.0
        active_by_group = self._supervisor_active_job_ids_by_group()
        for group_id, run in self._active_runs.items():
            jobs = [
                self.store.get_job(job_id)
                for job_id in active_by_group.get(group_id, list(run.job_ids))
                if job_id not in excluded_job_ids
            ]
            materialized = [job for job in jobs if job is not None]
            if not materialized:
                continue
            active_vram_mb += self.planner.predicted_group_vram_mb(materialized, backend_name=run.backend_name)
            active_sm_utilization += self.planner.predicted_group_sm_utilization(materialized, backend_name=run.backend_name)
        return active_vram_mb, active_sm_utilization

    def _seconds_since(self, timestamp: str | None) -> float | None:
        parsed = parse_timestamp(timestamp)
        if parsed is None:
            return None
        return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())

    def _metadata_int(self, job: TrainingJob, key: str, default: int = 0) -> int:
        try:
            return int(job.metadata.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    def _metadata_float(self, job: TrainingJob, key: str, default: float | None = None) -> float | None:
        try:
            value = job.metadata.get(key, default)
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return default

    def _should_emit_throttled_event(self, key: tuple[Any, ...], *, cooldown_seconds: float = 30.0) -> bool:
        now = time.monotonic()
        last = self._event_throttle_last_emitted.get(key)
        if last is not None and (now - last) < cooldown_seconds:
            return False
        self._event_throttle_last_emitted[key] = now
        return True

    def _sample_values_for_event(self, values: list[Any], *, limit: int = EVENT_JOB_ID_SAMPLE_LIMIT) -> tuple[list[Any], int]:
        sample = values[:limit]
        return sample, max(0, len(values) - len(sample))

    def _compact_candidate_for_event(self, candidate: dict[str, Any]) -> dict[str, Any]:
        job_ids = list(candidate.get("job_ids") or [])
        packing_signatures = list(candidate.get("packing_signatures") or [])
        sampled_job_ids, truncated_job_count = self._sample_values_for_event(job_ids)
        sampled_signatures, truncated_signature_count = self._sample_values_for_event(packing_signatures)
        compact: dict[str, Any] = {
            "job_ids": sampled_job_ids,
            "job_count": len(job_ids),
            "truncated_job_count": truncated_job_count,
            "backend_name": candidate.get("backend_name"),
            "status": candidate.get("status"),
            "rejection_reason": candidate.get("rejection_reason"),
            "expected_runtime_seconds": candidate.get("expected_runtime_seconds"),
            "job_expected_runtime_seconds": {
                job_id: value
                for job_id, value in list((candidate.get("job_expected_runtime_seconds") or {}).items())[
                    :EVENT_JOB_ID_SAMPLE_LIMIT
                ]
            },
        }
        if packing_signatures:
            compact["packing_signatures"] = sampled_signatures
            compact["truncated_packing_signature_count"] = truncated_signature_count
        for key in (
            "objective_score",
            "estimated_vram_mb",
            "estimated_sm_utilization",
            "batch_overrides",
            "fallback_order",
            "reason",
        ):
            if key in candidate:
                compact[key] = candidate.get(key)
        prediction_traces = candidate.get("prediction_traces") or {}
        if prediction_traces:
            trace_job_ids, truncated_trace_count = self._sample_values_for_event(list(prediction_traces), limit=EVENT_JOB_ID_SAMPLE_LIMIT)
            compact["prediction_trace_job_ids"] = trace_job_ids
            compact["truncated_prediction_trace_count"] = truncated_trace_count
        return compact

    def _candidate_summary_for_event(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        reason_counts: dict[str, int] = {}
        backend_counts: dict[str, int] = {}
        unique_job_ids: list[str] = []
        seen_job_ids: set[str] = set()
        for candidate in candidates:
            reason = str(candidate.get("rejection_reason") or candidate.get("status") or "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            backend = str(candidate.get("backend_name") or "unselected")
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
            for job_id in list(candidate.get("job_ids") or []):
                if job_id in seen_job_ids:
                    continue
                seen_job_ids.add(job_id)
                unique_job_ids.append(job_id)
        sample_candidates = [
            self._compact_candidate_for_event(candidate)
            for candidate in candidates[:EVENT_CANDIDATE_SAMPLE_LIMIT]
        ]
        sample_job_ids, truncated_job_id_count = self._sample_values_for_event(unique_job_ids)
        return {
            "candidate_count": len(candidates),
            "sample_candidates": sample_candidates,
            "truncated_candidate_count": max(0, len(candidates) - len(sample_candidates)),
            "candidate_rejection_reason_counts": reason_counts,
            "candidate_backend_counts": backend_counts,
            "candidate_job_ids": sample_job_ids,
            "candidate_unique_job_count": len(unique_job_ids),
            "truncated_candidate_job_id_count": truncated_job_id_count,
        }

    def _planner_trace_for_event(self, trace: dict[str, Any]) -> dict[str, Any]:
        event_trace = dict(trace)
        candidates = list(event_trace.pop("candidates", []) or [])
        event_trace.update(self._candidate_summary_for_event(candidates))
        return event_trace

    def _emit_scheduler_preemption_skipped(self, job: TrainingJob, *, reason: str, payload: dict[str, Any] | None = None) -> None:
        event_payload = {"reason": reason, **(payload or {})}
        key = (
            "scheduler_preemption_skipped",
            job.job_id,
            reason,
            str(event_payload.get("strategy") or ""),
        )
        if not self._should_emit_throttled_event(key):
            return
        self.event_logger.emit(
            "scheduler_preemption_skipped",
            job_id=job.job_id,
            payload=event_payload,
        )

    def _can_preempt_active_job(
        self,
        job: TrainingJob,
        run: ActiveRun,
        *,
        strategy: str,
        enforce_runtime_gate: bool,
    ) -> bool:
        if job.status != JobStatus.RUNNING:
            return False
        if not self._supports_safe_preemption(job):
            self._emit_scheduler_preemption_skipped(job, reason="job is not checkpoint-preemptible", payload={"strategy": strategy})
            return False
        max_per_job = int(self.settings.gpu_scheduler.checkpoint_preemption_max_per_job)
        count = self._metadata_int(job, "scheduler_preemption_count")
        if max_per_job > 0 and count >= max_per_job:
            self._emit_scheduler_preemption_skipped(
                job,
                reason="max scheduler preemptions reached",
                payload={"strategy": strategy, "count": count, "max_per_job": max_per_job},
            )
            return False
        cooldown_seconds = float(self.settings.gpu_scheduler.checkpoint_preemption_cooldown_seconds)
        since_last = self._seconds_since(str(job.metadata.get("scheduler_preemption_last_at") or ""))
        if since_last is not None and since_last < cooldown_seconds:
            self._emit_scheduler_preemption_skipped(
                job,
                reason="scheduler preemption cooldown active",
                payload={"strategy": strategy, "seconds_since_last": since_last, "cooldown_seconds": cooldown_seconds},
            )
            return False
        if enforce_runtime_gate:
            min_runtime = float(self.settings.gpu_scheduler.checkpoint_preemption_min_runtime_seconds)
            runtime = self._seconds_since(job.last_dispatched_at or job.started_at)
            if runtime is not None and runtime < min_runtime:
                self._emit_scheduler_preemption_skipped(
                    job,
                    reason="job has not run long enough for resource-aware preemption",
                    payload={"strategy": strategy, "runtime_seconds": runtime, "min_runtime_seconds": min_runtime},
                )
                return False
        remaining = self.planner.predicted_remaining_runtime_seconds(job, backend_name=run.backend_name)
        min_gain = float(self.settings.gpu_scheduler.checkpoint_preemption_min_estimated_gain_seconds)
        if remaining is not None and remaining <= min_gain:
            self._emit_scheduler_preemption_skipped(
                job,
                reason="job is near completion",
                payload={"strategy": strategy, "remaining_runtime_seconds": remaining, "min_estimated_gain_seconds": min_gain},
            )
            return False
        return True

    def _checkpoint_overhead_seconds(self, job: TrainingJob) -> float:
        explicit = self._metadata_float(job, "checkpoint_estimated_overhead_seconds")
        if explicit is not None:
            return max(0.0, explicit)
        explicit = self._metadata_float(job, "scheduler_checkpoint_overhead_seconds")
        if explicit is not None:
            return max(0.0, explicit)
        avg_step_ms = self._metadata_float(job, "runtime_avg_step_time_ms")
        if avg_step_ms is not None:
            return max(0.1, avg_step_ms / 1000.0)
        return 1.0

    def _preemption_resume_updates(
        self,
        job: TrainingJob,
        *,
        group_id: str,
        plan: DispatchPlan,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        if not job.metadata.get("scheduler_preemption_completed_at"):
            return {}, None
        if bool(job.metadata.get("scheduler_preemption_resume_emitted")):
            return {}, None
        checkpoint_path = job.latest_checkpoint_path or self.store.latest_checkpoint(job.job_id)
        if not checkpoint_path:
            return {}, None
        resumed_at = utc_now()
        updates = {
            "scheduler_preemption_resume_emitted": True,
            "scheduler_preemption_resumed_at": resumed_at,
        }
        payload = {
            "checkpoint_path": checkpoint_path,
            "resumed_at": resumed_at,
            "group_id": group_id,
            "placement_mode": plan.mode,
            "placement_backend": plan.backend_name,
            "preemption_reason": job.metadata.get("scheduler_preemption_reason"),
            "preemption_strategy": job.metadata.get("scheduler_preemption_strategy"),
        }
        return updates, payload

    def _enforce_scheduler_preemption_timeouts(self) -> None:
        timeout_seconds = float(self.settings.gpu_scheduler.checkpoint_preemption_pause_timeout_seconds)
        for _group_id, _run, job in self._active_job_records():
            if not bool(job.metadata.get("scheduler_preemption_pending")):
                continue
            if bool(job.metadata.get("scheduler_preemption_timeout_reported")):
                continue
            elapsed = self._seconds_since(str(job.metadata.get("scheduler_preemption_requested_at") or ""))
            if elapsed is None or elapsed <= timeout_seconds:
                continue
            self._emit_scheduler_preemption_skipped(
                job,
                reason="pause timeout before checkpointed safe point",
                payload={"elapsed_seconds": elapsed, "timeout_seconds": timeout_seconds},
            )
            self.store.update_job(job.job_id, metadata_updates={"scheduler_preemption_timeout_reported": True})

    def _preload_job_baseline(self, job: TrainingJob) -> None:
        target = self._resolve_preload_target(job)
        try:
            self.cache.preload(
                target.model_id,
                target.model_path,
                loader_target=target.loader_target,
                metadata={"source": "dispatch", "job_id": job.job_id},
            )
        except Exception as exc:
            self.logger.warning("Baseline preload failed for job %s (%s): %s", job.job_id, target.model_id, exc)

    def _active_occupancy(self) -> tuple[float, float]:
        active_vram_mb = 0.0
        active_sm_utilization = 0.0
        for group_id, run in self._active_runs.items():
            jobs = [self.store.get_job(job_id) for job_id in self._supervisor_active_job_ids_by_group().get(group_id, [])]
            materialized = [job for job in jobs if job is not None]
            if not materialized:
                continue
            active_vram_mb += self.planner.predicted_group_vram_mb(materialized, backend_name=run.backend_name)
            active_sm_utilization += self.planner.predicted_group_sm_utilization(materialized, backend_name=run.backend_name)
        return active_vram_mb, active_sm_utilization

    def _emit_planner_decision_trace(self, plan: DispatchPlan | None) -> None:
        raw_trace = getattr(self.planner, "last_decision_trace", None)
        trace = dict(raw_trace) if isinstance(raw_trace, dict) else {}
        if not trace:
            trace = {
                "scheduler_mode": self.settings.gpu_scheduler.mode,
                "selected_plan": None,
                "candidates": [],
            }
        if plan is not None and not trace.get("selected_plan"):
            trace["selected_plan"] = {
                "mode": plan.mode,
                "backend_name": plan.backend_name,
                "job_ids": list(plan.job_ids),
                "reason": plan.reason,
                "batch_overrides": dict(plan.batch_overrides),
                "fallback_order": list(plan.fallback_order),
            }
        event_trace = self._planner_trace_for_event(trace)
        selected_plan = event_trace.get("selected_plan") or {}
        key = (
            "planner_decision_trace",
            selected_plan.get("mode"),
            selected_plan.get("backend_name"),
            tuple(selected_plan.get("job_ids") or []),
            event_trace.get("decision_reason"),
            event_trace.get("candidate_count"),
        )
        if not self._should_emit_throttled_event(key, cooldown_seconds=5.0):
            return
        self.event_logger.emit("planner_decision_trace", payload=event_trace)

    def _emit_packing_probe_order_events(self, runnable: list[TrainingJob], plan: DispatchPlan | None) -> None:
        trace = getattr(self.planner, "last_decision_trace", None)
        missing = list(trace.get("missing_profile_job_ids") or []) if isinstance(trace, dict) else []
        if missing and self._should_emit_throttled_event(("adaptive_profile_wait", tuple(missing)), cooldown_seconds=5.0):
            self.event_logger.emit(
                "adaptive_profile_wait",
                payload={
                    "scheduler_session_id": self.settings.scheduler_session_id,
                    "job_ids": missing,
                    "active_job_ids": self._supervisor_active_job_ids(),
                    "drain_latched": self._profile_drain_latched,
                },
            )
        if plan is not None:
            event_key = (
                "adaptive_plan_selected",
                plan.backend_name,
                tuple(plan.job_ids),
                tuple(plan.active_job_ids),
                tuple(sorted(plan.batch_overrides.items())),
            )
            if not self._should_emit_throttled_event(event_key, cooldown_seconds=30.0):
                return
            self.event_logger.emit(
                "adaptive_plan_selected",
                payload={
                    "job_ids": list(plan.job_ids),
                    "active_job_ids": list(plan.active_job_ids),
                    "backend_name": plan.backend_name,
                    "batch_overrides": dict(plan.batch_overrides),
                    "solver_kind": plan.solver_kind,
                    "objective_vector": list(plan.objective_vector),
                    "reason": plan.reason,
                },
            )

    def _dispatch_plan(
        self,
        plan: DispatchPlan,
        *,
        allow_exclusive_fallback: bool = True,
        repack_transaction: PendingRepack | None = None,
    ) -> bool:
        selected_jobs = []
        for job_id in plan.job_ids:
            job = self.store.get_job(job_id)
            if job is None:
                return False
            selected_jobs.append(job)
        if plan.batch_overrides:
            selected_jobs = [
                self._apply_batch_override(job, plan.batch_overrides.get(job.job_id, self._resolved_batch_size_for_job_id(job.job_id)))
                for job in selected_jobs
            ]
        predispatch_jobs: list[TrainingJob] = []
        for index, job in enumerate(selected_jobs):
            if len(plan.job_ids) == 1:
                role = "solo"
            elif len(plan.job_ids) == 2:
                role = "primary" if index == 0 else "secondary"
            else:
                role = f"slot-{index}"
            predispatch_jobs.append(
                self.store.update_job(
                    job.job_id,
                    metadata_updates={
                        "placement_mode": plan.mode,
                        "placement_backend": plan.backend_name,
                        "placement_role": role,
                        "placement_batch_size": plan.batch_overrides.get(job.job_id),
                        "scheduler_session_id": self.settings.scheduler_session_id,
                    },
                )
            )
        selected_jobs = predispatch_jobs
        for job in selected_jobs:
            self._preload_job_baseline(job)

        try:
            dispatched = self.supervisor.dispatch(
                selected_jobs,
                mode=plan.mode,
                backend_name=plan.backend_name,
                batch_overrides=plan.batch_overrides,
                fallback_order=plan.fallback_order,
            )
        except Exception as exc:
            self.logger.warning("Dispatch failed for jobs %s: %s", ",".join(plan.job_ids), exc)
            if allow_exclusive_fallback and plan.backend_name != "exclusive" and selected_jobs and not self._active_runs:
                fallback_job = selected_jobs[0]
                self.logger.warning(
                    "Falling back to exclusive dispatch for %s after backend %s failed",
                    fallback_job.job_id,
                    plan.backend_name,
                )
                try:
                    fallback_job = self.store.update_job(
                        fallback_job.job_id,
                        metadata_updates={
                            "placement_mode": "exclusive",
                            "placement_backend": "exclusive",
                            "placement_role": "solo",
                            "scheduler_session_id": self.settings.scheduler_session_id,
                        },
                    )
                    fallback_decision = self.supervisor.dispatch([fallback_job], mode="exclusive", backend_name="exclusive")
                    if fallback_decision.can_run:
                        group_id = fallback_decision.group_id or f"legacy-{fallback_job.job_id}-{time.monotonic_ns()}"
                        self._active_runs[group_id] = ActiveRun(
                            group_id=group_id,
                            mode="exclusive",
                            backend_name="exclusive",
                            job_ids=(fallback_job.job_id,),
                            batch_overrides={fallback_job.job_id: self._resolved_batch_size_for_job_id(fallback_job.job_id)},
                            hardware_key=self.store.hardware_key(),
                            group_signature=build_group_signature([fallback_job.packing.signature or fallback_job.job_id]),
                        )
                        self._log_run_group_open(self._active_runs[group_id], [fallback_job], reason="backend_fallback_dispatch")
                        self._emit_worker_launch_events(
                            group_id=group_id,
                            run=self._active_runs[group_id],
                            jobs=[fallback_job],
                            reason="backend_fallback_dispatch",
                        )
                        self._last_telemetry_poll_at = 0.0
                        resume_updates, resume_payload = self._preemption_resume_updates(
                            fallback_job,
                            group_id=group_id,
                            plan=DispatchPlan(
                                mode="exclusive",
                                backend_name="exclusive",
                                job_ids=(fallback_job.job_id,),
                                reason="backend_fallback_dispatch",
                            ),
                        )
                        metadata_updates = {
                            "placement_mode": "exclusive",
                            "placement_backend": "exclusive",
                            "placement_role": "solo",
                            "scheduler_session_id": self.settings.scheduler_session_id,
                        }
                        metadata_updates.update(resume_updates)
                        self.store.update_job(
                            fallback_job.job_id,
                            status=JobStatus.RUNNING,
                            reason="dispatched to worker after backend fallback",
                            hold=False,
                            metadata_updates=metadata_updates,
                        )
                        if resume_payload is not None:
                            self.event_logger.emit("job_resumed_from_preemption", job_id=fallback_job.job_id, payload=resume_payload)
                        return True
                except Exception as fallback_exc:
                    self.logger.warning("Exclusive fallback dispatch also failed for %s: %s", fallback_job.job_id, fallback_exc)
            return False
        if not dispatched.can_run:
            if self._should_emit_throttled_event(
                ("dispatch_skipped", tuple(plan.job_ids), dispatched.reason),
                cooldown_seconds=30.0,
            ):
                self.logger.info("Skipping dispatch for %s: %s", ",".join(plan.job_ids), dispatched.reason)
            return False
        group_id = dispatched.group_id or f"legacy-{plan.job_ids[0]}-{time.monotonic_ns()}"

        signatures = [job.packing.signature or job.job_id for job in selected_jobs]
        self._active_runs[group_id] = ActiveRun(
            group_id=group_id,
            mode=plan.mode,
            backend_name=plan.backend_name,
            job_ids=plan.job_ids,
            batch_overrides=dict(plan.batch_overrides),
            fallback_order=list(plan.fallback_order),
            hardware_key=self.store.hardware_key(),
            group_signature=build_group_signature(signatures),
            repack_transaction=repack_transaction,
        )
        self._log_run_group_open(self._active_runs[group_id], selected_jobs, reason=plan.reason)
        self._emit_worker_launch_events(group_id=group_id, run=self._active_runs[group_id], jobs=selected_jobs, reason=plan.reason)
        if len(self._active_runs) > 1:
            for run in self._active_runs.values():
                run.overlapped = True
        self._last_telemetry_poll_at = 0.0

        for index, job in enumerate(selected_jobs):
            if len(plan.job_ids) == 1:
                role = "solo"
            elif len(plan.job_ids) == 2:
                role = "primary" if index == 0 else "secondary"
            else:
                role = f"slot-{index}"
            resume_updates, resume_payload = self._preemption_resume_updates(job, group_id=group_id, plan=plan)
            metadata_updates = {
                "placement_mode": plan.mode,
                "placement_backend": plan.backend_name,
                "placement_role": role,
                "placement_batch_size": plan.batch_overrides.get(job.job_id),
                "placement_group_id": group_id,
                "scheduler_session_id": self.settings.scheduler_session_id,
            }
            metadata_updates.update(resume_updates)
            self.store.update_job(
                job.job_id,
                status=JobStatus.RUNNING,
                reason="dispatched to worker",
                hold=False,
                metadata_updates=metadata_updates,
            )
            if resume_payload is not None:
                self.event_logger.emit("job_resumed_from_preemption", job_id=job.job_id, payload=resume_payload)
            self.event_logger.emit(
                "job_dispatched",
                job_id=job.job_id,
                payload={
                    "priority": job.priority,
                    "scheduler_session_id": self.settings.scheduler_session_id,
                    "placement_mode": plan.mode,
                    "placement_backend": plan.backend_name,
                    "group_id": group_id,
                    "job_ids": list(plan.job_ids),
                    "batch_overrides": dict(plan.batch_overrides),
                    "reason": plan.reason,
                    "batch_probe_profile": self._batch_probe_profile_payload(job),
                    "runtime_profile": self._runtime_profile_payload(job, backend_name=plan.backend_name),
                },
            )
        if len(plan.job_ids) == 2:
            self.event_logger.emit(
                "packed_pair_dispatched",
                payload={
                    "job_ids": list(plan.job_ids),
                    "group_id": group_id,
                    "backend_name": plan.backend_name,
                    "batch_overrides": dict(plan.batch_overrides),
                    "reason": plan.reason,
                    "members": [
                        {
                            "job_id": job.job_id,
                            "role": "primary" if index == 0 else "secondary",
                            "batch_size": plan.batch_overrides.get(job.job_id),
                            "batch_probe_profile": self._batch_probe_profile_payload(job),
                            "runtime_profile": self._runtime_profile_payload(job, backend_name=plan.backend_name),
                        }
                        for index, job in enumerate(selected_jobs)
                    ],
                },
            )
        elif len(plan.job_ids) > 2:
            self.event_logger.emit(
                "packed_group_dispatched",
                payload={
                    "job_ids": list(plan.job_ids),
                    "group_id": group_id,
                    "backend_name": plan.backend_name,
                    "batch_overrides": dict(plan.batch_overrides),
                    "reason": plan.reason,
                    "members": [
                        {
                            "job_id": job.job_id,
                            "role": f"slot-{index}",
                            "batch_size": plan.batch_overrides.get(job.job_id),
                            "batch_probe_profile": self._batch_probe_profile_payload(job),
                            "runtime_profile": self._runtime_profile_payload(job, backend_name=plan.backend_name),
                        }
                        for index, job in enumerate(selected_jobs)
                    ],
                },
            )
        return True

    def _mark_plan_incompatible(self, plan: DispatchPlan, *, reason: str) -> None:
        if len(plan.job_ids) < 2:
            return
        jobs = [self.store.get_job(job_id) for job_id in plan.job_ids]
        materialized = [job for job in jobs if job is not None]
        if len(materialized) != len(plan.job_ids):
            return
        signatures = [job.packing.signature or job.job_id for job in materialized]
        group_signature = build_group_signature(signatures)
        self.store.upsert_combination_profile(
            CombinationProfile.create(
                group_signature=group_signature,
                hardware_key=self.store.hardware_key(),
                backend_name=plan.backend_name,
                scheduler_mode=self.settings.gpu_scheduler.mode,
                batch_vector=plan.batch_overrides,
                compatible=False,
                observations=1,
                resolved_optimal=False,
                last_failure_reason=reason,
                fallback_order=plan.fallback_order,
                metadata={"job_ids": list(plan.job_ids), "adaptive_repack_failure": True},
            )
        )
        if len(materialized) == 2:
            self.store.mark_pair_incompatible(
                signatures[0],
                signatures[1],
                backend_name=plan.backend_name,
                reason=reason,
                cooldown_seconds=self.settings.gpu_scheduler.fallback_cooldown_seconds,
                metadata={"batch_vector": dict(plan.batch_overrides), "adaptive_repack_failure": True},
            )

    @staticmethod
    def _worker_exit_indicates_oom(job: TrainingJob, snapshot: WorkerSnapshot) -> bool:
        evidence = [str(job.status_reason or "")]
        if snapshot.stderr_path is not None:
            try:
                evidence.append(snapshot.stderr_path.read_text(encoding="utf-8", errors="replace")[-16_384:])
            except OSError:
                pass
        lowered = "\n".join(evidence).lower()
        return any(
            marker in lowered
            for marker in (
                "out of memory",
                "cuda oom",
                "cublas_status_alloc_failed",
                "hip out of memory",
            )
        )

    def _rollback_launched_repack(self, run: ActiveRun, *, failed_job_id: str, reason: str) -> None:
        transaction = run.repack_transaction
        if transaction is None:
            return
        failure_reason = f"adaptive repack runtime OOM: {reason}"
        self._mark_plan_incompatible(transaction.target_plan, reason=failure_reason)
        stop_group = getattr(self.supervisor, "stop_group", None)
        if callable(stop_group):
            stop_group(run.group_id)
        self._active_runs.pop(run.group_id, None)
        self.log_store.close_run_group(
            group_id=run.group_id,
            closed_at=utc_now(),
            overlapped=run.overlapped,
            fallback_triggered=True,
            fallback_reason=failure_reason,
            exit_reason="adaptive_repack_oom_rollback",
        )

        rollback_ids = set(transaction.rollback_plan.job_ids)
        for job_id in transaction.target_plan.job_ids:
            prior_batch = transaction.prior_batch_sizes.get(job_id)
            job = self.store.get_job(job_id)
            if job is None:
                continue
            if prior_batch is not None:
                job = self._apply_batch_override(job, prior_batch)
            self.store.update_job(
                job_id,
                status=JobStatus.READY,
                reason=(
                    "restored from durable pre-repack checkpoint"
                    if job_id in rollback_ids
                    else "new admission returned to queue after repack OOM"
                ),
                hold=False,
                metadata_updates={
                    "scheduler_repack_runtime_rollback": True,
                    "scheduler_repack_runtime_rollback_reason": failure_reason,
                    "scheduler_repack_transaction_id": transaction.transaction_id,
                },
            )

        rollback_launched = self._dispatch_plan(transaction.rollback_plan, allow_exclusive_fallback=False)
        self.event_logger.emit(
            "adaptive_repack_rolled_back",
            payload={
                "transaction_id": transaction.transaction_id,
                "failed_job_ids": [failed_job_id],
                "rollback_job_ids": list(transaction.rollback_plan.job_ids),
                "queued_job_ids": [
                    job_id for job_id in transaction.target_plan.job_ids if job_id not in rollback_ids
                ],
                "reason": failure_reason,
                "rollback_launched": rollback_launched,
            },
        )

    def _log_run_group_open(self, run: ActiveRun, jobs: list[TrainingJob], *, reason: str) -> None:
        self.log_store.open_run_group(
            group_id=run.group_id,
            mode=run.mode,
            backend_name=run.backend_name,
            hardware_key=run.hardware_key or self.store.hardware_key(),
            group_signature=run.group_signature,
            opened_at=run.opened_at,
            overlapped=run.overlapped,
            metadata={
                "job_ids": list(run.job_ids),
                "reason": reason,
                "scheduler_session_id": self.settings.scheduler_session_id,
            },
        )
        for index, job in enumerate(jobs):
            if len(jobs) == 1:
                role = "solo"
            elif len(jobs) == 2:
                role = "primary" if index == 0 else "secondary"
            else:
                role = f"slot-{index}"
            self.log_store.upsert_run_group_member(
                group_id=run.group_id,
                job_id=job.job_id,
                role=role,
                batch_size=run.batch_overrides.get(job.job_id),
                joined_at=run.opened_at,
                metadata={
                    "task_type": job.task_type,
                    "probe_task": bool(job.batch_probe.enabled or job.runtime_probe.enabled),
                    "placement_mode": run.mode,
                    "placement_backend": run.backend_name,
                    "placement_reason": reason,
                    "placement_group_id": run.group_id,
                    "placement_batch_size": run.batch_overrides.get(job.job_id),
                    "scheduler_session_id": self.settings.scheduler_session_id,
                    "batch_overrides": dict(run.batch_overrides),
                    "fallback_order": list(run.fallback_order),
                    "packing_signature": job.packing.signature,
                    "batch_probe_profile": self._batch_probe_profile_payload(job),
                    "runtime_profile": self._runtime_profile_payload(job, backend_name=run.backend_name),
                },
            )

    def _profile_gate_key(self, job: TrainingJob) -> str:
        namespace = job.batch_probe.profile_namespace or job.packing.signature or job.baseline_model_id
        shape_signature = self.planner.estimator.shape_signature(job)
        return f"{namespace}|{shape_signature}|{self.store.hardware_key()}|v3"

    def _set_profile_disposition(
        self,
        job: TrainingJob,
        state: ProfileState,
        *,
        force_exclusive: bool = False,
    ) -> TrainingJob:
        updated = job.copy()
        updated.profile_state = state
        updated.force_exclusive = bool(force_exclusive)
        if force_exclusive:
            updated.current_batch_size = max(
                1,
                int(updated.batch_probe.minimum_batch_size or self.settings.gpu_scheduler.batch_probe_min_batch_size),
            )
            updated.packing.eligible = False
            updated.packing.backend_allowlist = ["exclusive"]
        self.store.save_job(updated)
        return updated

    def _build_profile_probe_job(self, source: TrainingJob, gate_key: str) -> TrainingJob:
        probe_spec = BatchProbeSpec.from_dict(source.batch_probe.to_dict())
        probe_spec.contract_version = 3
        probe_spec.reuse_only = False
        runner_kwargs = dict(source.config.runner_kwargs)
        runner_kwargs[BatchResolution.param_name(source)] = source.authored_batch_size
        runner_kwargs["probe_max_batch_size"] = int(
            runner_kwargs.get("probe_max_batch_size")
            or self.settings.gpu_scheduler.batch_probe_max_batch_size
            or 4096
        )
        probe = TrainingJob.create(
            runner_target="localml_scheduler.profiling.batch_probe:run_branch_profile_probe_job",
            baseline_model_id=source.baseline_model_id,
            baseline_model_path=source.baseline_model_path,
            job_id=f"profile-{uuid.uuid4().hex[:20]}",
            workflow_id=source.workflow_id,
            task_type="mlevolve_branch_profile_probe",
            priority=max(source.priority, int(self.settings.gpu_scheduler.model_family_probe_priority)),
            runner_kwargs=runner_kwargs,
            resource_requirements=ResourceRequirements(requires_gpu=True),
            packing=PackingSpec(eligible=False, signature=gate_key, family="branch_profile_probe", backend_allowlist=["exclusive"]),
            batch_probe=probe_spec,
            runtime_probe=RuntimeProbeSpec(enabled=False),
            checkpoint_policy=source.checkpoint_policy,
            metadata={
                "kind": "mlevolve_branch_profile_probe",
                "exclusive_probe": True,
                "profile_gate_key": gate_key,
                "source_job_id": source.job_id,
            },
            python_executable=source.config.python_executable,
            loader_target=source.config.loader_target,
            env=dict(source.config.env),
        )
        probe.profile_state = ProfileState.PROBING
        return probe

    def _profile_drain_blocks_dispatch(self, runnable: list[TrainingJob]) -> bool:
        if self.settings.prediction.mode != PREDICTION_MODE_BRANCH_PROFILE:
            self._profile_drain_latched = False
            return False
        normal_jobs = [
            job
            for job in runnable
            if job.task_type not in NON_PREEMPTIBLE_PROBE_TASK_TYPES and job.task_type != "mlevolve_branch_profile_probe"
        ]
        for job in normal_jobs:
            if (
                job.batch_probe.enabled
                and not job.force_exclusive
                and self.planner.profile_ready(job)
                and job.profile_state != ProfileState.READY
            ):
                self._set_profile_disposition(job, ProfileState.READY)
        missing = [job for job in normal_jobs if job.batch_probe.enabled and not self.planner.profile_ready(job) and not job.force_exclusive]
        if missing:
            self._profile_drain_latched = True
            for job in missing:
                if job.profile_state != ProfileState.WAITING_FOR_DRAIN:
                    self._set_profile_disposition(job, ProfileState.WAITING_FOR_DRAIN)
        if not self._profile_drain_latched:
            return False
        if self._supervisor_active_job_ids():
            return True

        all_jobs = self.store.list_jobs()
        by_gate: dict[str, list[TrainingJob]] = {}
        for job in missing:
            by_gate.setdefault(self._profile_gate_key(job), []).append(job)

        for gate_key, dependents in by_gate.items():
            existing = next(
                (job for job in all_jobs if job.task_type == "mlevolve_branch_profile_probe" and job.metadata.get("profile_gate_key") == gate_key),
                None,
            )
            if existing is None:
                probe = self._build_profile_probe_job(dependents[0], gate_key)
                self.store.submit_job(probe)
                self._profile_probe_jobs[gate_key] = probe.job_id
                self.event_logger.emit(
                    "branch_profile_probe_queued",
                    job_id=probe.job_id,
                    payload={"profile_gate_key": gate_key, "dependent_job_ids": [job.job_id for job in dependents]},
                )
                return True
            self._profile_probe_jobs[gate_key] = existing.job_id
            if not existing.status.is_terminal:
                return False
            if all(self.planner.profile_ready(job) for job in dependents):
                for job in dependents:
                    self._set_profile_disposition(job, ProfileState.READY)
                continue
            if existing.status == JobStatus.FAILED:
                for job in dependents:
                    self._set_profile_disposition(job, ProfileState.UNAVAILABLE, force_exclusive=True)

        refreshed = self._runnable_jobs()
        remaining = [
            job
            for job in refreshed
            if job.task_type != "mlevolve_branch_profile_probe"
            and job.batch_probe.enabled
            and not job.force_exclusive
            and not self.planner.profile_ready(job)
        ]
        if remaining:
            return True
        self._profile_drain_latched = False
        self.event_logger.emit("branch_profile_drain_released", payload={})
        return False

    def _active_jobs(self) -> list[TrainingJob]:
        jobs = [self.store.get_job(job_id) for job_id in self._supervisor_active_job_ids()]
        return [job for job in jobs if job is not None]

    def _current_active_plan(self, active_jobs: list[TrainingJob]) -> DispatchPlan:
        run = next(iter(self._active_runs.values()))
        return DispatchPlan(
            mode=run.mode,
            backend_name=run.backend_name,
            job_ids=tuple(job.job_id for job in active_jobs),
            reason="rollback to pre-repack placement",
            batch_overrides={job.job_id: BatchResolution.resolved_batch_size(job) for job in active_jobs},
            fallback_order=list(run.fallback_order),
        )

    @staticmethod
    def _same_placement(left: DispatchPlan, right: DispatchPlan) -> bool:
        return set(left.job_ids) == set(right.job_ids) and {
            key: int(value) for key, value in left.batch_overrides.items()
        } == {key: int(value) for key, value in right.batch_overrides.items()}

    def _repack_qualifies(self, plan: DispatchPlan, active_jobs: list[TrainingJob]) -> bool:
        active_ids = {job.job_id for job in active_jobs}
        admitted = set(plan.job_ids) - active_ids
        active_run = next(iter(self._active_runs.values()))
        if any(
            not self._can_preempt_active_job(job, active_run, strategy="adaptive_repack", enforce_runtime_gate=True)
            for job in active_jobs
        ):
            return False
        if admitted:
            return True
        if self.settings.prediction.mode != PREDICTION_MODE_BRANCH_PROFILE or plan.predicted_throughput is None:
            return False
        old_throughput = sum(
            self.planner.estimator.predicted_samples_per_second(job, BatchResolution.resolved_batch_size(job)) or 0.0
            for job in active_jobs
        )
        if old_throughput <= 0:
            return False
        gain_fraction = (float(plan.predicted_throughput) - old_throughput) / old_throughput
        if gain_fraction < self.settings.gpu_scheduler.adaptive.minimum_throughput_gain_fraction:
            return False
        remaining = max(
            (
                self.planner.predicted_remaining_runtime_seconds(job, backend_name=active_run.backend_name) or 0.0
                for job in active_jobs
            ),
            default=0.0,
        )
        estimated_saved = remaining * max(0.0, 1.0 - (old_throughput / float(plan.predicted_throughput)))
        required = max(
            float(self.settings.gpu_scheduler.checkpoint_preemption_min_estimated_gain_seconds),
            sum(self._checkpoint_overhead_seconds(job) for job in active_jobs)
            * float(self.settings.gpu_scheduler.checkpoint_preemption_overhead_multiplier),
        )
        return estimated_saved >= required

    def _begin_repack(self, plan: DispatchPlan, active_jobs: list[TrainingJob]) -> bool:
        if self._pending_repack is not None or not active_jobs:
            return False
        transaction_id = uuid.uuid4().hex
        reason = f"adaptive repack to admit {','.join(sorted(set(plan.job_ids) - {job.job_id for job in active_jobs}))}"
        requested: list[str] = []
        for job in active_jobs:
            if not self.supervisor.request_repack_prepare(job.job_id, transaction_id=transaction_id, reason=reason):
                for requested_job_id in requested:
                    self.supervisor.request_repack_abort(requested_job_id, transaction_id=transaction_id)
                return False
            requested.append(job.job_id)
        requested_at = utc_now()
        for job in active_jobs:
            self.store.update_job(
                job.job_id,
                metadata_updates={
                    "scheduler_preemption_last_at": requested_at,
                    "scheduler_preemption_count": self._metadata_int(job, "scheduler_preemption_count") + 1,
                    "scheduler_preemption_strategy": "adaptive_repack",
                    "scheduler_repack_transaction_id": transaction_id,
                },
            )
        self._pending_repack = PendingRepack(
            transaction_id=transaction_id,
            target_plan=plan,
            rollback_plan=self._current_active_plan(active_jobs),
            active_job_ids=tuple(requested),
            prior_batch_sizes={
                job_id: BatchResolution.resolved_batch_size(job)
                for job_id in plan.job_ids
                if (job := self.store.get_job(job_id)) is not None
            },
            requested_at_monotonic=time.monotonic(),
        )
        self.event_logger.emit(
            "adaptive_repack_preparing",
            payload={
                "transaction_id": transaction_id,
                "active_job_ids": requested,
                "target_job_ids": list(plan.job_ids),
                "target_batch_overrides": dict(plan.batch_overrides),
            },
        )
        return True

    def _advance_pending_repack(self) -> bool:
        transaction = self._pending_repack
        if transaction is None:
            return False
        timeout = float(self.settings.gpu_scheduler.checkpoint_preemption_pause_timeout_seconds)
        elapsed = time.monotonic() - transaction.requested_at_monotonic
        if transaction.phase == "preparing":
            acknowledgements = {
                job_id: self.supervisor.repack_ack(job_id)
                for job_id in transaction.active_job_ids
            }
            ready = all(
                payload is not None
                and payload.get("transaction_id") == transaction.transaction_id
                and payload.get("checkpoint_path")
                for payload in acknowledgements.values()
            )
            if ready:
                for job_id, payload in acknowledgements.items():
                    assert payload is not None
                    self.store.update_job(job_id, latest_checkpoint_path=str(payload["checkpoint_path"]))
                    self.supervisor.request_repack_commit(job_id, transaction_id=transaction.transaction_id)
                transaction.phase = "committing"
                self.event_logger.emit(
                    "adaptive_repack_committing",
                    payload={"transaction_id": transaction.transaction_id, "job_ids": list(transaction.active_job_ids)},
                )
                return True
            if elapsed >= timeout:
                for job_id in transaction.active_job_ids:
                    self.supervisor.request_repack_abort(job_id, transaction_id=transaction.transaction_id)
                self.event_logger.emit(
                    "adaptive_repack_aborted",
                    payload={"transaction_id": transaction.transaction_id, "reason": "checkpoint barrier timeout"},
                )
                self._pending_repack = None
                return True
            return True

        if self._supervisor_active_job_ids():
            return True
        target = transaction.target_plan
        if self._dispatch_plan(
            target,
            allow_exclusive_fallback=False,
            repack_transaction=transaction,
        ):
            self.event_logger.emit(
                "adaptive_repack_committed",
                payload={"transaction_id": transaction.transaction_id, "job_ids": list(target.job_ids)},
            )
            self._pending_repack = None
            return True

        failure_reason = "adaptive repack target launch failed"
        self._mark_plan_incompatible(target, reason=failure_reason)
        for job_id, batch_size in transaction.prior_batch_sizes.items():
            job = self.store.get_job(job_id)
            if job is not None:
                self._apply_batch_override(job, batch_size)
        self._dispatch_plan(transaction.rollback_plan, allow_exclusive_fallback=False)
        self.event_logger.emit(
            "adaptive_repack_rolled_back",
            payload={
                "transaction_id": transaction.transaction_id,
                "failed_job_ids": list(target.job_ids),
                "rollback_job_ids": list(transaction.rollback_plan.job_ids),
            },
        )
        self._pending_repack = None
        return True

    def _dispatch_pending_work(self) -> None:
        if self._advance_pending_repack():
            return
        active_jobs = self._active_jobs()
        active_ids = {job.job_id for job in active_jobs}
        runnable = [job for job in self._runnable_jobs() if job.job_id not in active_ids]
        if self._profile_drain_blocks_dispatch(runnable):
            return
        if self._profile_drain_latched:
            runnable = [job for job in runnable if job.task_type == "mlevolve_branch_profile_probe"]
        if not runnable and not active_jobs:
            return

        now = time.monotonic()
        if active_jobs and (now - self._last_adaptive_replan_at) < self.settings.gpu_scheduler.adaptive.replan_debounce_seconds:
            return
        self._last_adaptive_replan_at = now
        plan = self.planner.choose_plan(
            runnable,
            active_jobs=active_jobs,
            backend_available=self.supervisor.available_backends(),
        )
        self._emit_planner_decision_trace(plan)
        self._emit_packing_probe_order_events(runnable, plan)
        if plan is None:
            return
        if active_jobs:
            current = self._current_active_plan(active_jobs)
            if self._same_placement(current, plan) or not self._repack_qualifies(plan, active_jobs):
                return
            self._begin_repack(plan, active_jobs)
            return
        self._dispatch_plan(plan)

    def _dispatch_if_idle(self) -> None:
        """Backward-compatible alias for older tests and call sites."""
        self._dispatch_pending_work()

    def report(self) -> dict[str, Any]:
        return self.metrics.as_dict()

    def cache_stats(self) -> dict[str, Any]:
        return {
            "stats": self.cache.stats().to_dict(),
            "entries": self.cache.snapshot_entries(),
        }
