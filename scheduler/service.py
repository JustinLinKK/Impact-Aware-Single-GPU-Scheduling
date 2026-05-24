"""Scheduler service loop."""

from __future__ import annotations

from dataclasses import dataclass, field
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
from ..profiling.runtime_probe import runtime_profile_for_job
from ..domain import BatchResolution, CombinationProfile, JobStatus, PairProfile, PreloadSource, SoloProfile, TrainingJob, build_group_signature, utc_now
from ..config import SCHEDULER_MODE_PARALLEL_AUTO_PACK, SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED, SchedulerSettings
from ..storage.log_store import SchedulerLogStore
from ..storage.state_store import StateStore
from .placement_planner import PlacementPlanner
from .planner_types import DispatchPlan
from .policies import PriorityFifoPolicy, SchedulingPolicy
from .queue import RunnableJobQueue
from .recovery import reconcile_recoverable_jobs
from .supervisor import WorkerSnapshot, WorkerSupervisor
from .telemetry import GpuTelemetrySample, GpuTelemetrySummary, NvidiaSmiTelemetrySampler


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
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        tmp_path.replace(path)

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
        memory_fraction = latest.memory_used_mb / latest.memory_total_mb
        if memory_fraction < self.settings.gpu_scheduler.memory.hard_stop_memory_fraction:
            return
        target = self._pick_fallback_candidate()
        if target is None:
            return
        group_id, target_job_id = target
        reason = f"packed groups exceeded hard memory threshold ({memory_fraction:.2%})"
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
        if snapshot.reported_by == "store":
            if run_context is not None and len(run_context.job_ids) > 1 and job.status == JobStatus.FAILED:
                self._register_packed_fallback(run_context, job.status_reason or "stream-backed worker failed", payload={"failed_job_id": snapshot.job_id})
            self._emit_worker_finished_event(snapshot, run_context=run_context)
            return
        if snapshot.returncode == 0:
            if job.status in {JobStatus.COMPLETED, JobStatus.PAUSED, JobStatus.CANCELLED, JobStatus.READY}:
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
            if job.status in {JobStatus.FAILED, JobStatus.CANCELLED}:
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
                objective_score=(summary.peak_vram_mb or 0) / max(1.0, self.settings.gpu_scheduler.memory.safe_vram_budget_gib * 1024.0),
                resolved_optimal=(self.settings.gpu_scheduler.mode == SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED),
                last_failure_reason=run.fallback_reason,
                fallback_order=run.fallback_order,
                metadata={"backend_name": run.backend_name, "job_ids": list(run.job_ids)},
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
                slowdown_ratio=None,
                cooldown_until=None,
                last_failure_reason=None,
                metadata={"backend_name": run.backend_name},
            )
        )

    def _next_job(self) -> TrainingJob | None:
        queue = RunnableJobQueue(policy=self.policy, jobs=self._runnable_jobs())
        return queue.peek()

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
        if len(self._active_runs) != 1:
            return
        active_run = next(iter(self._active_runs.values()))
        if active_run.mode != "exclusive":
            return
        active_job_id = active_run.job_ids[0] if active_run.job_ids else None
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
        self.event_logger.emit("planner_decision_trace", payload=trace)

    def _dispatch_plan(self, plan: DispatchPlan) -> bool:
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
            if plan.backend_name != "exclusive" and selected_jobs and not self._active_runs:
                fallback_job = selected_jobs[0]
                self.logger.warning(
                    "Falling back to exclusive dispatch for %s after backend %s failed",
                    fallback_job.job_id,
                    plan.backend_name,
                )
                try:
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
                        self.store.update_job(
                            fallback_job.job_id,
                            status=JobStatus.RUNNING,
                            reason="dispatched to worker after backend fallback",
                            hold=False,
                            metadata_updates={
                                "placement_mode": "exclusive",
                                "placement_backend": "exclusive",
                                "placement_role": "solo",
                            },
                        )
                        return True
                except Exception as fallback_exc:
                    self.logger.warning("Exclusive fallback dispatch also failed for %s: %s", fallback_job.job_id, fallback_exc)
            return False
        if not dispatched.can_run:
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
            self.store.update_job(
                job.job_id,
                status=JobStatus.RUNNING,
                reason="dispatched to worker",
                hold=False,
                metadata_updates={
                    "placement_mode": plan.mode,
                    "placement_backend": plan.backend_name,
                    "placement_role": role,
                    "placement_batch_size": plan.batch_overrides.get(job.job_id),
                    "placement_group_id": group_id,
                },
            )
            self.event_logger.emit(
                "job_dispatched",
                job_id=job.job_id,
                payload={
                    "priority": job.priority,
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

    def _log_run_group_open(self, run: ActiveRun, jobs: list[TrainingJob], *, reason: str) -> None:
        self.log_store.open_run_group(
            group_id=run.group_id,
            mode=run.mode,
            backend_name=run.backend_name,
            hardware_key=run.hardware_key or self.store.hardware_key(),
            group_signature=run.group_signature,
            opened_at=run.opened_at,
            overlapped=run.overlapped,
            metadata={"job_ids": list(run.job_ids), "reason": reason},
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
                    "batch_overrides": dict(run.batch_overrides),
                    "fallback_order": list(run.fallback_order),
                    "packing_signature": job.packing.signature,
                    "batch_probe_profile": self._batch_probe_profile_payload(job),
                    "runtime_profile": self._runtime_profile_payload(job, backend_name=run.backend_name),
                },
            )

    def _dispatch_pending_work(self) -> None:
        scheduler_mode = self.settings.gpu_scheduler.mode
        if scheduler_mode != SCHEDULER_MODE_PARALLEL_AUTO_PACK and self._active_runs:
            return

        while True:
            active_job_ids = set(self._supervisor_active_job_ids())
            runnable = [job for job in self._runnable_jobs() if job.job_id not in active_job_ids]
            if not runnable:
                return
            active_vram_mb, active_sm_utilization = self._active_occupancy()
            plan = self.planner.choose_plan(
                runnable,
                backend_available=self.supervisor.available_backends(),
                active_vram_mb=active_vram_mb,
                active_sm_utilization=active_sm_utilization,
            )
            self._emit_planner_decision_trace(plan)
            if plan is None:
                return
            if scheduler_mode == SCHEDULER_MODE_PARALLEL_AUTO_PACK and self._active_runs and plan.backend_name == "exclusive":
                return
            dispatched = self._dispatch_plan(plan)
            if not dispatched or scheduler_mode != SCHEDULER_MODE_PARALLEL_AUTO_PACK:
                return

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
