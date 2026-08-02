"""Scheduler service loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from math import isfinite
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
from ..domain import (
    BatchResolution,
    ColocationTimingProfile,
    CombinationProfile,
    JobStatus,
    PairProfile,
    PreloadSource,
    SchedulingClass,
    SoloProfile,
    TrainingJob,
    build_group_signature,
    build_colocation_profile_key,
    parse_timestamp,
    utc_now,
)
from ..config import (
    SCHEDULER_MODE_PARALLEL_AUTO_PACK,
    SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED,
    SCHEDULER_MODE_PARALLEL_TIME_AWARE,
    SchedulerSettings,
)
from ..storage.log_store import SchedulerLogStore
from ..storage.state_store import StateStore
from .placement_planner import PlacementPlanner
from .planner_types import DispatchPlan
from .policies import PriorityFifoPolicy, SchedulingPolicy
from .queue import RunnableJobQueue
from .recovery import reconcile_recoverable_jobs
from .supervisor import WorkerSnapshot, WorkerSupervisor
from .telemetry import (
    GpuTelemetrySample,
    GpuTelemetrySummary,
    MemoryAdmissionGate,
    NvidiaSmiTelemetrySampler,
)


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
    objective_breakdown: dict[str, object] = field(default_factory=dict)
    objective_version: str | None = None
    mandatory_anchor_job_id: str | None = None


@dataclass(slots=True)
class ColocationTrialState:
    trial_id: str
    candidate_job_id: str
    preexisting_job_ids: tuple[str, ...]
    started_at: str
    start_epoch: int
    target_epoch: int
    backend_name: str
    profile_key: str
    candidate_solo_epoch_seconds: float
    pretrial_epoch_seconds: dict[str, float] = field(default_factory=dict)
    member_start_epochs: dict[str, int] = field(default_factory=dict)
    evidence_deadline_at: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "trial_id": self.trial_id,
            "candidate_job_id": self.candidate_job_id,
            "preexisting_job_ids": list(self.preexisting_job_ids),
            "started_at": self.started_at,
            "start_epoch": self.start_epoch,
            "target_epoch": self.target_epoch,
            "backend_name": self.backend_name,
            "profile_key": self.profile_key,
            "candidate_solo_epoch_seconds": self.candidate_solo_epoch_seconds,
            "pretrial_epoch_seconds": dict(self.pretrial_epoch_seconds),
            "member_start_epochs": dict(self.member_start_epochs),
            "evidence_deadline_at": self.evidence_deadline_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ColocationTrialState":
        return cls(
            trial_id=str(payload["trial_id"]),
            candidate_job_id=str(payload["candidate_job_id"]),
            preexisting_job_ids=tuple(str(value) for value in payload.get("preexisting_job_ids", [])),
            started_at=str(payload["started_at"]),
            start_epoch=int(payload["start_epoch"]),
            target_epoch=int(payload["target_epoch"]),
            backend_name=str(payload["backend_name"]),
            profile_key=str(payload["profile_key"]),
            candidate_solo_epoch_seconds=float(payload["candidate_solo_epoch_seconds"]),
            pretrial_epoch_seconds={str(key): float(value) for key, value in dict(payload.get("pretrial_epoch_seconds", {})).items()},
            member_start_epochs={
                str(key): int(value)
                for key, value in dict(payload.get("member_start_epochs", {})).items()
            },
            evidence_deadline_at=str(payload.get("evidence_deadline_at") or ""),
        )


@dataclass(frozen=True, slots=True)
class TrialEpochEvidence:
    seconds_per_epoch: float | None
    sample_count: int
    samples: tuple[float, ...] = ()


@dataclass(slots=True)
class ColocationStallState:
    preexisting_job_ids: tuple[str, ...]
    candidate_job_id: str
    profile_key: str
    reason: str
    started_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, object]:
        return {
            "preexisting_job_ids": list(self.preexisting_job_ids),
            "candidate_job_id": self.candidate_job_id,
            "profile_key": self.profile_key,
            "reason": self.reason,
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ColocationStallState":
        return cls(
            preexisting_job_ids=tuple(str(value) for value in payload.get("preexisting_job_ids", [])),
            candidate_job_id=str(payload["candidate_job_id"]),
            profile_key=str(payload["profile_key"]),
            reason=str(payload["reason"]),
            started_at=str(payload.get("started_at") or utc_now()),
        )


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
        memory = settings.gpu_scheduler.memory
        self._admission_gate = MemoryAdmissionGate(
            stop_fraction=memory.live_admission_stop_fraction,
            resume_fraction=memory.live_admission_resume_fraction,
            window_seconds=memory.admission_average_window_seconds,
        )
        self._exclusive_probe_job_id: str | None = None
        self._colocation_trial: ColocationTrialState | None = None
        self._colocation_stall: ColocationStallState | None = None
        self._restore_scheduler_decision_state()

    @property
    def _decision_state_path(self) -> Path:
        return self.settings.runtime_root / "scheduler_decision_state.json"

    def _persist_scheduler_decision_state(self) -> None:
        payload = {
            "admission_open": self._admission_gate.is_open,
            "admission_average_fraction": self._admission_gate.average_fraction,
            "admission_below_resume_since": (
                self._admission_gate.below_resume_since.isoformat() if self._admission_gate.below_resume_since is not None else None
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
            "colocation_trial": self._colocation_trial.to_dict() if self._colocation_trial else None,
            "colocation_stall": self._colocation_stall.to_dict() if self._colocation_stall else None,
            "updated_at": utc_now(),
        }
        tmp_path = self._decision_state_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        tmp_path.replace(self._decision_state_path)

    def _restore_scheduler_decision_state(self) -> None:
        if not self._decision_state_path.exists():
            return
        try:
            with self._decision_state_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self._admission_gate.is_open = bool(payload.get("admission_open", True))
            average = payload.get("admission_average_fraction")
            self._admission_gate.average_fraction = float(average) if average is not None else None
            self._admission_gate.below_resume_since = parse_timestamp(payload.get("admission_below_resume_since"))
            self._admission_gate.samples = [GpuTelemetrySample(**sample) for sample in payload.get("admission_samples", []) if isinstance(sample, dict)]
            reserved = payload.get("exclusive_probe_job_id")
            if self.settings.gpu_scheduler.exclusive_probe.enabled:
                self._exclusive_probe_job_id = str(reserved) if reserved else None
            else:
                # A reservation is meaningful only while exclusive-probe
                # draining is enabled.  Do not let stale restart state close
                # normal admission after the feature has been turned off.
                self._exclusive_probe_job_id = None
            trial = payload.get("colocation_trial")
            self._colocation_trial = ColocationTrialState.from_dict(trial) if isinstance(trial, dict) else None
            stall = payload.get("colocation_stall")
            self._colocation_stall = ColocationStallState.from_dict(stall) if isinstance(stall, dict) else None
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.logger.warning("Ignoring invalid scheduler decision state: %s", exc)

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
        self.event_logger.emit(
            event_name,
            payload={
                "model_id": entry.model_id,
                **(payload or {}),
                **entry.to_stats_dict(),
            },
        )

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
        reconcile_recoverable_jobs(
            self.store,
            self.event_logger,
            auto_resume=self.settings.auto_resume_recoverable,
        )
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
            self._evaluate_colocation_trial()
            self._process_commands()
            self._warm_cache()
            self._poll_telemetry()
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
        if job.status in {
            JobStatus.PAUSED,
            JobStatus.RECOVERABLE,
            JobStatus.PENDING,
            JobStatus.READY,
        }:
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
        self.event_logger.emit(
            "cache_preload_requested",
            payload={"model_id": target.model_id, "ok": ok, "pin": pin},
        )

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
            available_budget_bytes = max(
                0,
                int(cache_stats.effective_memory_budget_bytes) - int(cache_stats.used_bytes),
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
                if self._exclusive_probe_job_id in run.job_ids:
                    completed_probe = self._exclusive_probe_job_id
                    self._exclusive_probe_job_id = None
                    self._persist_scheduler_decision_state()
                    self.event_logger.emit(
                        "exclusive_probe_drain_cleared",
                        job_id=completed_probe,
                        payload={"reason": "probe finished"},
                    )
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
                    [(self.store.get_job(job_id).packing.signature or job_id) for job_id in remaining_job_ids if self.store.get_job(job_id) is not None]
                )

    def _handle_worker_exit(self, snapshot: WorkerSnapshot, *, run_context: ActiveRun | None) -> None:
        job = self.store.get_job(snapshot.job_id)
        if job is None:
            return
        if snapshot.reported_by == "store":
            if run_context is not None and len(run_context.job_ids) > 1 and job.status == JobStatus.FAILED:
                self._register_packed_fallback(
                    run_context,
                    job.status_reason or "stream-backed worker failed",
                    payload={"failed_job_id": snapshot.job_id},
                )
            return
        if snapshot.returncode == 0:
            if job.status in {
                JobStatus.COMPLETED,
                JobStatus.PAUSED,
                JobStatus.CANCELLED,
                JobStatus.READY,
            }:
                return
            self.store.set_job_status(
                job.job_id,
                JobStatus.FAILED,
                reason="worker exited without terminal status update",
                hold=True,
            )
            self.event_logger.emit(
                "job_failed",
                job_id=job.job_id,
                payload={"reason": "worker exited cleanly without terminal status"},
            )
            return

        if not job.status.is_terminal:
            reason = f"worker exited with code {snapshot.returncode}"
            self.store.set_job_status(job.job_id, JobStatus.FAILED, reason=reason, hold=True)
            self.event_logger.emit(
                "job_failed",
                job_id=job.job_id,
                payload={"returncode": snapshot.returncode},
            )
            if run_context is not None and len(run_context.job_ids) > 1:
                self._register_packed_fallback(
                    run_context,
                    reason,
                    payload={
                        "failed_job_id": snapshot.job_id,
                        "returncode": snapshot.returncode,
                    },
                )
            return

        if run_context is not None and len(run_context.job_ids) > 1 and job.status == JobStatus.FAILED:
            reason = job.status_reason or f"worker exited with code {snapshot.returncode}"
            self._register_packed_fallback(
                run_context,
                reason,
                payload={
                    "failed_job_id": snapshot.job_id,
                    "returncode": snapshot.returncode,
                },
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
            avg_vram_mb = summary.avg_vram_mb
            if avg_vram_mb is None:
                avg_vram_mb = job.resource_requirements.estimated_avg_vram_mb
            self.store.upsert_solo_profile(
                SoloProfile(
                    signature=job.packing.signature,
                    hardware_key=run.hardware_key or self.store.hardware_key(),
                    family=job.packing.family,
                    peak_vram_mb=peak_vram_mb,
                    avg_vram_mb=avg_vram_mb,
                    avg_gpu_utilization=(summary.avg_gpu_utilization if summary.avg_gpu_utilization is not None else 0.0),
                    avg_memory_utilization=(summary.avg_memory_utilization if summary.avg_memory_utilization is not None else 0.0),
                    sample_count=summary.sample_count,
                    last_job_id=job.job_id,
                    metadata={
                        "source": "exclusive_run",
                        "backend_name": run.backend_name,
                    },
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
        objective_score = run.objective_breakdown.get("score")
        if objective_score is None and self.settings.gpu_scheduler.mode != SCHEDULER_MODE_PARALLEL_TIME_AWARE:
            objective_score = (summary.avg_vram_mb or 0) / max(1.0, self.planner.estimator.safe_budget_mb())
        numeric_objective_score = float(objective_score) if isinstance(objective_score, (int, float)) else None
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
                avg_vram_mb=summary.avg_vram_mb,
                memory_total_mb=(run.samples[-1].memory_total_mb if run.samples else None),
                avg_gpu_utilization=summary.avg_gpu_utilization,
                avg_memory_utilization=summary.avg_memory_utilization,
                avg_step_time_ms=None,
                objective_score=numeric_objective_score,
                resolved_optimal=(self.settings.gpu_scheduler.mode == SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED),
                last_failure_reason=run.fallback_reason,
                fallback_order=run.fallback_order,
                metadata={
                    "backend_name": run.backend_name,
                    "job_ids": list(run.job_ids),
                    "objective_version": run.objective_version,
                    "objective_breakdown": run.objective_breakdown,
                    "mandatory_anchor_job_id": run.mandatory_anchor_job_id,
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
                avg_vram_mb=summary.avg_vram_mb,
                avg_gpu_utilization=summary.avg_gpu_utilization,
                avg_memory_utilization=summary.avg_memory_utilization,
                metadata={"backend_name": run.backend_name},
            )
            return
        existing_pair = self.store.get_pair_profile(
            left_job.packing.signature,
            right_job.packing.signature,
            backend_name=run.backend_name,
        )
        existing_metadata = dict(existing_pair.metadata or {}) if existing_pair else {}
        existing_sources = existing_metadata.get("slowdown_sources")
        has_epoch_evidence = isinstance(existing_sources, dict) and any(
            source == "measured_epoch_against_exclusive_profile"
            for source in existing_sources.values()
        )
        # Combination completion still records compatibility and telemetry,
        # but it must never infer slowdown from whole-run elapsed time. Preserve
        # only slowdown evidence produced from clean epoch intervals.
        recorded_slowdown = existing_pair.slowdown_ratio if existing_pair and has_epoch_evidence else None
        per_member_slowdown = existing_metadata.get("per_member_slowdown", {}) if has_epoch_evidence else {}
        per_signature_slowdown = existing_metadata.get("per_signature_slowdown", {}) if has_epoch_evidence else {}
        slowdown_sources = existing_sources if has_epoch_evidence else {}
        slowdown_batch_vector = existing_metadata.get("batch_vector", {}) if has_epoch_evidence else {}
        self.store.upsert_pair_profile(
            PairProfile.create(
                left_job.packing.signature,
                right_job.packing.signature,
                backend_name=run.backend_name,
                hardware_key=run.hardware_key or self.store.hardware_key(),
                compatible=True,
                observations=(existing_pair.observations + 1) if existing_pair else 1,
                peak_vram_mb=summary.peak_vram_mb,
                avg_vram_mb=summary.avg_vram_mb,
                avg_gpu_utilization=summary.avg_gpu_utilization,
                avg_memory_utilization=summary.avg_memory_utilization,
                slowdown_ratio=recorded_slowdown,
                cooldown_until=None,
                last_failure_reason=None,
                metadata={
                    "backend_name": run.backend_name,
                    "batch_vector": slowdown_batch_vector,
                    "per_member_slowdown": per_member_slowdown,
                    "per_signature_slowdown": per_signature_slowdown,
                    "slowdown_sources": slowdown_sources,
                },
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
        group_id = str(getattr(active_group, "group_id", "untracked-active-group"))
        if hasattr(active_group, "active_job_ids"):
            return {group_id: list(active_group.active_job_ids())}
        workers = getattr(active_group, "workers", {}) or {}
        return {group_id: list(workers.keys())}

    def _apply_batch_override(self, job: TrainingJob, batch_size: int) -> TrainingJob:
        updated_job = BatchResolution.apply(job, batch_size)
        self.store.save_job(updated_job)
        return updated_job

    def _maybe_preempt(self) -> None:
        if self.settings.gpu_scheduler.mode == SCHEDULER_MODE_PARALLEL_TIME_AWARE:
            # Time-aware scheduling uses drain boundaries; exclusive probes and
            # newly urgent work do not preempt active training.
            return
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
                payload={
                    "reason": reason,
                    "preempting_job_id": candidate_job.job_id,
                    "hold": False,
                },
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
            self.logger.warning(
                "Baseline preload failed for job %s (%s): %s",
                job.job_id,
                target.model_id,
                exc,
            )

    def _active_vram_occupancy(self) -> float:
        active_vram_mb = 0.0
        for group_id, run in self._active_runs.items():
            jobs = [self.store.get_job(job_id) for job_id in self._supervisor_active_job_ids_by_group().get(group_id, [])]
            materialized = [job for job in jobs if job is not None]
            if materialized:
                active_vram_mb += self.planner.predicted_group_vram_mb(materialized, backend_name=run.backend_name)
        return active_vram_mb

    def _active_jobs(self) -> list[TrainingJob]:
        active_ids = self._supervisor_active_job_ids()
        jobs = [self.store.get_job(job_id) for job_id in active_ids]
        return [job for job in jobs if job is not None]

    @staticmethod
    def _remaining_epochs(job: TrainingJob) -> int | None:
        total = job.max_epochs or job.config.max_epochs
        if total is None:
            return None
        try:
            return max(0, int(total) - int(job.metadata.get("last_completed_epoch", 0)))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parsed_timestamp(value: object) -> datetime | None:
        try:
            parsed = parse_timestamp(str(value) if value else None)
        except (TypeError, ValueError):
            return None
        if parsed is not None and parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed

    def _trial_epoch_evidence(
        self,
        job: TrainingJob,
        trial: ColocationTrialState,
    ) -> TrialEpochEvidence:
        trial_started = self._parsed_timestamp(trial.started_at)
        trial_deadline = self._parsed_timestamp(trial.evidence_deadline_at)
        baseline_epoch = trial.member_start_epochs.get(job.job_id)
        required = self.settings.gpu_scheduler.colocation.trial_epochs
        if trial_started is None or trial_deadline is None or baseline_epoch is None:
            return TrialEpochEvidence(None, 0)

        candidate = job.job_id == trial.candidate_job_id
        by_epoch: dict[int, tuple[datetime, float]] = {}
        for sample in list(job.metadata.get("runtime_epoch_timing_history") or []):
            if not isinstance(sample, dict):
                continue
            try:
                epoch = int(sample.get("epoch", 0))
                seconds = float(sample.get("seconds", 0.0))
            except (TypeError, ValueError):
                continue
            if epoch <= baseline_epoch or not isfinite(seconds) or seconds <= 0:
                continue
            finished_at = self._parsed_timestamp(sample.get("finished_at"))
            if (
                finished_at is None
                or finished_at < trial_started
                or finished_at > trial_deadline
            ):
                continue
            interval_started = self._parsed_timestamp(sample.get("started_at"))
            if interval_started is None:
                if not candidate or str(sample.get("source")) != "runner_step_time":
                    continue
            elif interval_started < trial_started:
                continue
            previous = by_epoch.get(epoch)
            if previous is None or finished_at > previous[0]:
                by_epoch[epoch] = (finished_at, seconds)

        ordered = [
            seconds
            for _, (_, seconds) in sorted(
                by_epoch.items(),
                key=lambda item: (item[0], item[1][0]),
            )
        ]
        selected = tuple(ordered[-required:])
        if len(selected) < required:
            return TrialEpochEvidence(None, len(selected), selected)
        return TrialEpochEvidence(sum(selected) / len(selected), len(selected), selected)

    def _trial_evidence_timeout_seconds(
        self,
        candidate_solo_epoch_seconds: float,
        pretrial_epoch_seconds: dict[str, float],
    ) -> float:
        settings = self.settings.gpu_scheduler.colocation
        valid_rates: list[float] = []
        for value in [candidate_solo_epoch_seconds, *pretrial_epoch_seconds.values()]:
            try:
                rate = float(value)
            except (TypeError, ValueError):
                continue
            if isfinite(rate) and rate > 0:
                valid_rates.append(rate)
        estimated_window = (
            3.0 * settings.trial_epochs * max(valid_rates)
            if valid_rates
            else settings.trial_evidence_timeout_min_seconds
        )
        return min(
            settings.trial_evidence_timeout_max_seconds,
            max(settings.trial_evidence_timeout_min_seconds, estimated_window),
        )

    def _trial_evidence_deadline(
        self,
        started_at: str,
        candidate_solo_epoch_seconds: float,
        pretrial_epoch_seconds: dict[str, float],
    ) -> str:
        started = self._parsed_timestamp(started_at) or datetime.now(timezone.utc)
        timeout = self._trial_evidence_timeout_seconds(
            candidate_solo_epoch_seconds,
            pretrial_epoch_seconds,
        )
        return (started + timedelta(seconds=timeout)).isoformat()

    @staticmethod
    def _member_descriptor(job: TrainingJob, fallback_backend: str) -> dict[str, object]:
        return {
            "signature": job.packing.signature or job.job_id,
            "batch_size": BatchResolution.resolved_batch_size(job),
            "backend_name": str(job.metadata.get("placement_backend") or fallback_backend),
        }

    def _activate_colocation_stall(
        self,
        *,
        preexisting_job_ids: tuple[str, ...],
        candidate_job_id: str,
        profile_key: str,
        reason: str,
    ) -> None:
        self._colocation_stall = ColocationStallState(
            preexisting_job_ids=preexisting_job_ids,
            candidate_job_id=candidate_job_id,
            profile_key=profile_key,
            reason=reason,
        )
        self._persist_scheduler_decision_state()
        self.event_logger.emit(
            "colocation_admission_stalled",
            job_id=candidate_job_id,
            payload={
                "preexisting_job_ids": list(preexisting_job_ids),
                "profile_key": profile_key,
                "reason": reason,
            },
        )

    def _refresh_colocation_stall(self) -> None:
        if self._colocation_stall is None:
            return
        active_ids = set(self._supervisor_active_job_ids())
        if all(job_id in active_ids for job_id in self._colocation_stall.preexisting_job_ids):
            return
        previous = self._colocation_stall
        self._colocation_stall = None
        self._persist_scheduler_decision_state()
        self.event_logger.emit(
            "colocation_admission_resumed",
            payload={
                "previous_preexisting_job_ids": list(previous.preexisting_job_ids),
                "candidate_job_id": previous.candidate_job_id,
                "reason": "pre-trial pack membership changed",
            },
        )

    def _restart_colocation_trial(self, trial: ColocationTrialState, preexisting_jobs: list[TrainingJob]) -> None:
        candidate = self.store.get_job(trial.candidate_job_id)
        if candidate is None:
            self._colocation_trial = None
            self._persist_scheduler_decision_state()
            return
        start_epoch = int(candidate.metadata.get("last_completed_epoch", 0))
        total_epochs = candidate.max_epochs or candidate.config.max_epochs or start_epoch
        target_epoch = min(int(total_epochs), start_epoch + self.settings.gpu_scheduler.colocation.trial_epochs)
        pretrial: dict[str, float] = {}
        for job in preexisting_jobs:
            rate = trial.pretrial_epoch_seconds.get(job.job_id)
            if rate is None:
                rate, _ = self.planner.time_objective.current_epoch_seconds(job)
            if rate is not None:
                pretrial[job.job_id] = rate
        descriptors = [self._member_descriptor(job, trial.backend_name) for job in [*preexisting_jobs, candidate]]
        started_at = utc_now()
        member_start_epochs = {
            job.job_id: int(job.metadata.get("last_completed_epoch", 0))
            for job in [*preexisting_jobs, candidate]
        }
        restarted = ColocationTrialState(
            trial_id=f"trial-{candidate.job_id}-{time.monotonic_ns()}",
            candidate_job_id=candidate.job_id,
            preexisting_job_ids=tuple(job.job_id for job in preexisting_jobs),
            started_at=started_at,
            start_epoch=start_epoch,
            target_epoch=target_epoch,
            backend_name=trial.backend_name,
            profile_key=build_colocation_profile_key(self.store.hardware_key(), descriptors),
            candidate_solo_epoch_seconds=trial.candidate_solo_epoch_seconds,
            pretrial_epoch_seconds=pretrial,
            member_start_epochs=member_start_epochs,
            evidence_deadline_at=self._trial_evidence_deadline(
                started_at,
                trial.candidate_solo_epoch_seconds,
                pretrial,
            ),
        )
        self._colocation_trial = restarted
        self.store.update_job(
            candidate.job_id,
            metadata_updates={"colocation_trial": {**restarted.to_dict(), "decision": "pending"}},
        )
        self._persist_scheduler_decision_state()
        self.event_logger.emit(
            "colocation_trial_started",
            job_id=candidate.job_id,
            payload={**restarted.to_dict(), "reason": "pack membership changed; trial restarted"},
        )

    def _persist_colocation_timing_profile(
        self,
        jobs: list[TrainingJob],
        rates: dict[str, float],
        trial: ColocationTrialState,
        *,
        sources: dict[str, str] | None = None,
        gain: float | None = None,
        decision: str | None = None,
    ) -> None:
        descriptors = [self._member_descriptor(job, trial.backend_name) for job in jobs]
        profile_key = build_colocation_profile_key(self.store.hardware_key(), descriptors)
        stored_profile = self.store.get_colocation_timing_profile(profile_key)
        existing = (
            stored_profile
            if stored_profile is not None
            and self.planner.time_objective.profile_is_fresh(stored_profile)
            else None
        )
        if existing is not None and existing.metadata.get("last_trial_id") == trial.trial_id:
            return
        if sources is not None and not any(
            source != "exact_colocation_profile" for source in sources.values()
        ):
            return
        old_by_key: dict[tuple[str, int, str], dict[str, object]] = {}
        if existing is not None:
            for item in existing.member_timings:
                old_by_key[(str(item["signature"]), int(item["batch_size"]), str(item["backend_name"]))] = item
        timings: list[dict[str, object]] = []
        for job, descriptor in zip(jobs, descriptors, strict=True):
            rate = rates.get(job.job_id)
            if rate is None:
                continue
            key = (str(descriptor["signature"]), int(descriptor["batch_size"]), str(descriptor["backend_name"]))
            old = old_by_key.get(key)
            source = (sources or {}).get(job.job_id, "live_training")
            if source == "exact_colocation_profile" and old is not None:
                timings.append(dict(old))
                continue
            old_count = int(old.get("observations", 0)) if old else 0
            old_rate = float(old.get("seconds_per_epoch", 0.0)) if old else 0.0
            timings.append(
                {
                    **descriptor,
                    "seconds_per_epoch": ((old_rate * old_count) + rate) / (old_count + 1),
                    "observations": old_count + 1,
                    "source": source,
                }
            )
        if len(timings) != len(jobs):
            return
        metadata = dict(existing.metadata) if existing is not None else {}
        recent_outcomes = list(metadata.get("recent_trial_outcomes") or [])
        if decision in {"accepted", "rejected"} and gain is not None:
            outcome = {
                "trial_id": trial.trial_id,
                "decision": decision,
                "gain": float(gain),
                "observed_at": utc_now(),
            }
            if decision == "accepted":
                recent_outcomes = [outcome]
            else:
                recent_outcomes.append(outcome)
            recent_outcomes = recent_outcomes[-16:]
        metadata.update(
            {
                "last_trial_id": trial.trial_id,
                "job_ids": [job.job_id for job in jobs],
                "evidence_policy": "fresh_member_epochs_v1",
                "recent_trial_outcomes": recent_outcomes,
            }
        )
        profile = ColocationTimingProfile.create(
            self.store.hardware_key(),
            descriptors,
            timings,
            observations=(existing.observations + 1) if existing else 1,
            metadata=metadata,
        )
        self.store.upsert_colocation_timing_profile(profile)
        if sources is not None and any(
            source == "exact_colocation_profile" for source in sources.values()
        ):
            return
        if len(jobs) != 2 or any(not job.packing.signature for job in jobs):
            return
        per_member_slowdown: dict[str, float] = {}
        per_signature_slowdown: dict[str, float] = {}
        slowdown_sources: dict[str, str] = {}
        batch_vector: dict[str, int] = {}
        for job in jobs:
            packed_rate = rates.get(job.job_id)
            batch_size = BatchResolution.resolved_batch_size(job)
            solo_profile = self.store.get_runtime_profile(
                job.packing.signature or job.job_id,
                resolved_batch_size=batch_size,
                backend_name="exclusive",
            )
            solo_rate: float | None = None
            if solo_profile is not None:
                if solo_profile.epoch_1_seconds is not None and solo_profile.epoch_1_seconds > 0:
                    solo_rate = float(solo_profile.epoch_1_seconds)
                elif (
                    solo_profile.avg_step_time_ms is not None
                    and solo_profile.avg_step_time_ms > 0
                    and solo_profile.steps_per_epoch is not None
                    and solo_profile.steps_per_epoch > 0
                ):
                    solo_rate = float(solo_profile.avg_step_time_ms) * int(solo_profile.steps_per_epoch) / 1000.0
            if packed_rate is None or packed_rate <= 0 or solo_rate is None:
                continue
            ratio = float(packed_rate) / solo_rate
            per_member_slowdown[job.job_id] = ratio
            per_signature_slowdown[job.packing.signature or job.job_id] = ratio
            slowdown_sources[job.job_id] = "measured_epoch_against_exclusive_profile"
            batch_vector[job.job_id] = batch_size
        if len(per_member_slowdown) != 2:
            return
        left_job, right_job = jobs
        existing_pair = self.store.get_pair_profile(
            left_job.packing.signature or left_job.job_id,
            right_job.packing.signature or right_job.job_id,
            backend_name=trial.backend_name,
        )
        self.store.upsert_pair_profile(
            PairProfile.create(
                left_job.packing.signature or left_job.job_id,
                right_job.packing.signature or right_job.job_id,
                backend_name=trial.backend_name,
                hardware_key=self.store.hardware_key(),
                compatible=True,
                observations=(existing_pair.observations + 1) if existing_pair else 1,
                slowdown_ratio=max(per_member_slowdown.values()),
                metadata={
                    "backend_name": trial.backend_name,
                    "batch_vector": batch_vector,
                    "per_member_slowdown": per_member_slowdown,
                    "per_signature_slowdown": per_signature_slowdown,
                    "slowdown_sources": slowdown_sources,
                    "colocation_profile_key": profile.profile_key,
                    "trial_id": trial.trial_id,
                },
            )
        )

    def _evaluate_colocation_trial(self) -> None:
        self._refresh_colocation_stall()
        trial = self._colocation_trial
        if trial is None:
            return
        candidate = self.store.get_job(trial.candidate_job_id)
        active_ids = set(self._supervisor_active_job_ids())
        if candidate is None:
            self._colocation_trial = None
            self._persist_scheduler_decision_state()
            return
        trial_metadata = dict(candidate.metadata.get("colocation_trial") or {})
        if str(trial_metadata.get("decision")) == "timeout":
            self.store.update_job(
                candidate.job_id,
                metadata_updates={"colocation_unverified_profile_key": trial.profile_key},
            )
            self._colocation_trial = None
            self._persist_scheduler_decision_state()
            return
        candidate_finished = self._remaining_epochs(candidate) == 0
        if candidate.job_id not in active_ids and not candidate_finished:
            self._colocation_trial = None
            self._persist_scheduler_decision_state()
            return

        current_preexisting = [
            job for job in self._active_jobs() if job.job_id != candidate.job_id
        ]
        if set(job.job_id for job in current_preexisting) != set(trial.preexisting_job_ids):
            if candidate_finished:
                reason = "newcomer completed while colocation membership changed"
                self.store.update_job(
                    candidate.job_id,
                    metadata_updates={
                        "colocation_trial": {
                            **trial.to_dict(),
                            "decision": "completed_unverified",
                            "reason": reason,
                        },
                        "colocation_unverified_profile_key": trial.profile_key,
                    },
                )
                self.event_logger.emit(
                    "colocation_trial_completed_unverified",
                    job_id=candidate.job_id,
                    payload={"trial_id": trial.trial_id, "reason": reason},
                )
                self._colocation_trial = None
                self._persist_scheduler_decision_state()
                return
            if not current_preexisting:
                accepted = {**dict(candidate.metadata.get("colocation_trial") or {}), "decision": "accepted", "reason": "newcomer became stack anchor"}
                self.store.update_job(candidate.job_id, metadata_updates={"colocation_trial": accepted})
                self._colocation_trial = None
                self._persist_scheduler_decision_state()
                return
            self._restart_colocation_trial(trial, current_preexisting)
            return

        expected_members = {trial.candidate_job_id, *trial.preexisting_job_ids}
        if (
            set(trial.member_start_epochs) != expected_members
            or self._parsed_timestamp(trial.evidence_deadline_at) is None
        ):
            self._restart_colocation_trial(trial, current_preexisting)
            return

        all_jobs = [*current_preexisting, candidate]
        evidence = {
            job.job_id: self._trial_epoch_evidence(job, trial)
            for job in all_jobs
        }
        packed_rates: dict[str, float] = {}
        packed_sources: dict[str, str] = {}
        for job in all_jobs:
            rate = evidence[job.job_id].seconds_per_epoch
            if rate is not None and isfinite(rate) and rate > 0:
                packed_rates[job.job_id] = rate
                packed_sources[job.job_id] = "fresh_trial_epoch_average"

        required_samples = self.settings.gpu_scheduler.colocation.trial_epochs
        evidence_counts = {
            job_id: item.sample_count
            for job_id, item in evidence.items()
        }
        evidence_samples = {
            job_id: list(item.samples)
            for job_id, item in evidence.items()
        }
        evidence_complete = all(
            item.sample_count >= required_samples and item.seconds_per_epoch is not None
            for item in evidence.values()
        )
        candidate_remaining = self._remaining_epochs(candidate)
        completed_epoch = int(candidate.metadata.get("last_completed_epoch", 0))
        deadline = self._parsed_timestamp(trial.evidence_deadline_at)
        deadline_expired = deadline is not None and datetime.now(timezone.utc) >= deadline

        if not evidence_complete:
            missing_members = sorted(
                job_id
                for job_id, item in evidence.items()
                if item.sample_count < required_samples
            )
            progress_payload = {
                "trial_id": trial.trial_id,
                "profile_key": trial.profile_key,
                "required_samples_per_member": required_samples,
                "evidence_counts": evidence_counts,
                "evidence_samples": evidence_samples,
                "missing_member_ids": missing_members,
                "evidence_deadline_at": trial.evidence_deadline_at,
            }
            if candidate_remaining == 0:
                decision = {
                    **trial.to_dict(),
                    "decision": "completed_unverified",
                    "reason": "newcomer completed before every member supplied fresh timing evidence",
                    "evidence": progress_payload,
                }
                self.store.update_job(
                    candidate.job_id,
                    metadata_updates={
                        "colocation_trial": decision,
                        "colocation_unverified_profile_key": trial.profile_key,
                    },
                )
                self.event_logger.emit(
                    "colocation_trial_completed_unverified",
                    job_id=candidate.job_id,
                    payload=progress_payload,
                )
                self._colocation_trial = None
                self._persist_scheduler_decision_state()
                return
            if deadline_expired:
                reason = "colocation trial evidence deadline expired"
                pause_requested = self.supervisor.request_pause(
                    candidate.job_id,
                    reason=reason,
                    hold=False,
                )
                decision = {
                    **trial.to_dict(),
                    "decision": "timeout",
                    "reason": reason,
                    "evidence": progress_payload,
                }
                self.store.update_job(
                    candidate.job_id,
                    status=JobStatus.PAUSING if pause_requested else candidate.status,
                    reason=reason,
                    hold=False,
                    metadata_updates={
                        "colocation_trial": decision,
                        "colocation_unverified_profile_key": trial.profile_key,
                    },
                )
                self.event_logger.emit(
                    "colocation_trial_rejected",
                    job_id=candidate.job_id,
                    payload={**progress_payload, "reason": reason},
                )
                self._colocation_trial = None
                self._persist_scheduler_decision_state()
                return
            if completed_epoch >= trial.target_epoch:
                total_epochs = candidate.max_epochs or candidate.config.max_epochs or completed_epoch
                next_target = min(int(total_epochs), completed_epoch + 1)
                if next_target > trial.target_epoch:
                    trial.target_epoch = next_target
                    pending = {
                        **trial.to_dict(),
                        "decision": "pending",
                        "reason": "collecting fresh timing evidence from every trial member",
                        "evidence": progress_payload,
                    }
                    self.store.update_job(
                        candidate.job_id,
                        metadata_updates={"colocation_trial": pending},
                    )
                    self._persist_scheduler_decision_state()
                    self.event_logger.emit(
                        "colocation_trial_extended",
                        job_id=candidate.job_id,
                        payload={**progress_payload, "target_epoch": next_target},
                    )
            return

        result = self.planner.time_objective.estimate_gain(
            current_preexisting,
            candidate,
            backend_name=trial.backend_name,
            active_epoch_seconds=trial.pretrial_epoch_seconds,
            candidate_solo_epoch_seconds=trial.candidate_solo_epoch_seconds,
            packed_epoch_seconds=packed_rates,
            candidate_batch_size=BatchResolution.resolved_batch_size(candidate),
            active_epoch_sources={
                job_id: "pretrial_epoch_rate"
                for job_id in trial.pretrial_epoch_seconds
            },
            packed_epoch_sources=packed_sources,
        )
        if result is None:
            gain = 0.0
            sequential = None
            packed = None
            reason = "colocation trial lacked complete timing evidence"
        else:
            gain = result.gain
            sequential = result.sequential_drain_seconds
            packed = result.packed_drain_seconds
            reason = (
                "colocation gain accepted"
                if gain + 1e-9 >= self.settings.gpu_scheduler.colocation.min_gain
                else "colocation gain below threshold"
            )
        payload = {
            "trial_id": trial.trial_id,
            "gain": gain,
            "gain_threshold": self.settings.gpu_scheduler.colocation.min_gain,
            "sequential_drain_seconds": sequential,
            "packed_drain_seconds": packed,
            "sequential_drain_phases": (
                [phase.to_dict() for phase in result.sequential_phases]
                if result is not None
                else []
            ),
            "packed_drain_phases": (
                [phase.to_dict() for phase in result.packed_phases]
                if result is not None
                else []
            ),
            "packed_epoch_seconds": packed_rates,
            "packed_epoch_sources": packed_sources,
            "pretrial_epoch_seconds": trial.pretrial_epoch_seconds,
            "candidate_solo_epoch_seconds": trial.candidate_solo_epoch_seconds,
            "profile_key": trial.profile_key,
            "required_samples_per_member": required_samples,
            "evidence_counts": evidence_counts,
            "evidence_samples": evidence_samples,
            "evidence_deadline_at": trial.evidence_deadline_at,
        }
        self.event_logger.emit("colocation_gain_evaluated", job_id=candidate.job_id, payload=payload)
        self._persist_colocation_timing_profile(
            all_jobs,
            packed_rates,
            trial,
            sources=packed_sources,
            gain=result.gain if result is not None else None,
            decision=(
                "accepted"
                if result is not None
                and result.gain + 1e-9 >= self.settings.gpu_scheduler.colocation.min_gain
                else "rejected" if result is not None else None
            ),
        )
        newcomer_finished = self._remaining_epochs(candidate) == 0
        if newcomer_finished or gain + 1e-9 >= self.settings.gpu_scheduler.colocation.min_gain:
            if newcomer_finished:
                reason = "newcomer completed during colocation trial"
            decision = {**trial.to_dict(), "decision": "accepted", "reason": reason, "result": payload}
            self.store.update_job(candidate.job_id, metadata_updates={"colocation_trial": decision})
            self.event_logger.emit("colocation_trial_accepted", job_id=candidate.job_id, payload=payload)
            self._colocation_trial = None
            self._persist_scheduler_decision_state()
            return

        pause_requested = self.supervisor.request_pause(candidate.job_id, reason=reason, hold=False)
        decision = {**trial.to_dict(), "decision": "rejected", "reason": reason, "result": payload}
        self.store.update_job(
            candidate.job_id,
            status=JobStatus.PAUSING if pause_requested else candidate.status,
            reason=reason,
            hold=False,
            metadata_updates={
                "colocation_trial": decision,
                "colocation_unverified_profile_key": trial.profile_key if result is None else None,
            },
        )
        self.event_logger.emit("colocation_trial_rejected", job_id=candidate.job_id, payload=payload)
        if result is not None:
            self._activate_colocation_stall(
                preexisting_job_ids=trial.preexisting_job_ids,
                candidate_job_id=candidate.job_id,
                profile_key=trial.profile_key,
                reason=reason,
            )
        self._colocation_trial = None
        self._persist_scheduler_decision_state()

    def _prediction_metadata(self, job_id: str) -> dict[str, str | None]:
        estimator = getattr(self.planner, "estimator", None)
        method = getattr(estimator, "prediction_metadata", None)
        if callable(method):
            result = method(job_id)
            if isinstance(result, dict):
                return result
        return {
            "vram_prediction_source": "branch_profile",
            "vram_prediction_error": None,
        }

    def _known_colocation_rejection(self, plan: DispatchPlan) -> bool:
        if not bool(plan.objective_breakdown.get("colocation_rejected")) or not plan.job_ids:
            return False
        candidate_job_id = plan.job_ids[0]
        preexisting_job_ids = tuple(
            str(job_id) for job_id in plan.objective_breakdown.get("preexisting_job_ids", [])
        )
        profile_key = str(plan.objective_breakdown.get("colocation_profile_key") or "")
        payload = {
            "gain": plan.objective_breakdown.get("gain"),
            "gain_threshold": plan.objective_breakdown.get("gain_threshold"),
            "sequential_drain_seconds": plan.objective_breakdown.get("sequential_drain_seconds"),
            "packed_drain_seconds": plan.objective_breakdown.get("packed_drain_seconds"),
            "sequential_drain_phases": plan.objective_breakdown.get(
                "sequential_drain_phases",
                [],
            ),
            "packed_drain_phases": plan.objective_breakdown.get(
                "packed_drain_phases",
                [],
            ),
            "profile_key": profile_key,
            "preexisting_job_ids": list(preexisting_job_ids),
            "source": "exact_colocation_profile",
        }
        self.store.update_job(
            candidate_job_id,
            status=JobStatus.PAUSED,
            reason="known colocation gain is below threshold",
            hold=False,
            metadata_updates={
                "colocation_trial": {
                    "decision": "rejected",
                    "reason": "known colocation gain is below threshold",
                    "result": payload,
                }
            },
        )
        self.event_logger.emit("colocation_gain_evaluated", job_id=candidate_job_id, payload=payload)
        self.event_logger.emit("colocation_trial_rejected", job_id=candidate_job_id, payload=payload)
        self._activate_colocation_stall(
            preexisting_job_ids=preexisting_job_ids,
            candidate_job_id=candidate_job_id,
            profile_key=profile_key,
            reason="known colocation gain is below threshold",
        )
        return True

    def _prepare_colocation_trial(self, plan: DispatchPlan, candidate: TrainingJob) -> ColocationTrialState | None:
        metadata = plan.trial_metadata or plan.objective_breakdown
        if not bool(metadata.get("requires_live_trial")):
            return None
        start_epoch = int(candidate.metadata.get("last_completed_epoch", 0))
        total_epochs = candidate.max_epochs or candidate.config.max_epochs
        if total_epochs is None:
            return None
        target_epoch = min(
            int(total_epochs),
            start_epoch + self.settings.gpu_scheduler.colocation.trial_epochs,
        )
        preexisting_job_ids = tuple(
            str(job_id) for job_id in metadata.get("preexisting_job_ids", [])
        )
        pretrial_epoch_seconds = {
            str(job_id): float(seconds)
            for job_id, seconds in dict(
                metadata.get("pretrial_epoch_seconds")
                or metadata.get("active_pretrial_epoch_seconds")
                or {}
            ).items()
        }
        member_start_epochs = {candidate.job_id: start_epoch}
        for job_id in preexisting_job_ids:
            active_job = self.store.get_job(job_id)
            if active_job is not None:
                member_start_epochs[job_id] = int(
                    active_job.metadata.get("last_completed_epoch", 0)
                )
        started_at = utc_now()
        candidate_solo_epoch_seconds = float(metadata.get("candidate_solo_epoch_seconds") or 0.0)
        trial = ColocationTrialState(
            trial_id=f"trial-{candidate.job_id}-{time.monotonic_ns()}",
            candidate_job_id=candidate.job_id,
            preexisting_job_ids=preexisting_job_ids,
            started_at=started_at,
            start_epoch=start_epoch,
            target_epoch=target_epoch,
            backend_name=plan.backend_name,
            profile_key=str(metadata.get("profile_key") or metadata.get("colocation_profile_key") or ""),
            candidate_solo_epoch_seconds=candidate_solo_epoch_seconds,
            pretrial_epoch_seconds=pretrial_epoch_seconds,
            member_start_epochs=member_start_epochs,
            evidence_deadline_at=self._trial_evidence_deadline(
                started_at,
                candidate_solo_epoch_seconds,
                pretrial_epoch_seconds,
            ),
        )
        self._colocation_trial = trial
        self.store.update_job(
            candidate.job_id,
            metadata_updates={"colocation_trial": {**trial.to_dict(), "decision": "pending"}},
        )
        self._persist_scheduler_decision_state()
        return trial

    def _cancel_prepared_colocation_trial(self, trial: ColocationTrialState | None, *, reason: str) -> None:
        if trial is None:
            return
        current = self.store.get_job(trial.candidate_job_id)
        if current is not None:
            self.store.update_job(
                current.job_id,
                metadata_updates={
                    "colocation_trial": {**trial.to_dict(), "decision": "cancelled", "reason": reason}
                },
            )
        if self._colocation_trial is not None and self._colocation_trial.trial_id == trial.trial_id:
            self._colocation_trial = None
            self._persist_scheduler_decision_state()

    def _dispatch_plan(self, plan: DispatchPlan) -> bool:
        if self._known_colocation_rejection(plan):
            return False
        selected_jobs = []
        for job_id in plan.job_ids:
            job = self.store.get_job(job_id)
            if job is None:
                return False
            selected_jobs.append(job)
        if plan.batch_overrides:
            selected_jobs = [
                self._apply_batch_override(
                    job,
                    plan.batch_overrides.get(job.job_id, self._resolved_batch_size_for_job_id(job.job_id)),
                )
                for job in selected_jobs
            ]
        for job in selected_jobs:
            self._preload_job_baseline(job)

        prepared_trial = self._prepare_colocation_trial(plan, selected_jobs[0]) if selected_jobs else None

        try:
            dispatched = self.supervisor.dispatch(
                selected_jobs,
                mode=plan.mode,
                backend_name=plan.backend_name,
                batch_overrides=plan.batch_overrides,
                fallback_order=plan.fallback_order,
            )
        except Exception as exc:
            self._cancel_prepared_colocation_trial(prepared_trial, reason="trial dispatch failed")
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
                        group_id = fallback_decision.group_id or f"fallback-{fallback_job.job_id}-{time.monotonic_ns()}"
                        self._active_runs[group_id] = ActiveRun(
                            group_id=group_id,
                            mode="exclusive",
                            backend_name="exclusive",
                            job_ids=(fallback_job.job_id,),
                            batch_overrides={fallback_job.job_id: self._resolved_batch_size_for_job_id(fallback_job.job_id)},
                            hardware_key=self.store.hardware_key(),
                            group_signature=build_group_signature([fallback_job.packing.signature or fallback_job.job_id]),
                        )
                        self._log_run_group_open(
                            self._active_runs[group_id],
                            [fallback_job],
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
                                **self._prediction_metadata(fallback_job.job_id),
                            },
                        )
                        return True
                except Exception as fallback_exc:
                    self.logger.warning(
                        "Exclusive fallback dispatch also failed for %s: %s",
                        fallback_job.job_id,
                        fallback_exc,
                    )
            return False
        if not dispatched.can_run:
            self._cancel_prepared_colocation_trial(prepared_trial, reason=dispatched.reason)
            self.logger.info(
                "Skipping dispatch for %s: %s",
                ",".join(plan.job_ids),
                dispatched.reason,
            )
            return False
        group_id = dispatched.group_id or f"dispatch-{plan.job_ids[0]}-{time.monotonic_ns()}"

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
            objective_breakdown=dict(plan.objective_breakdown),
            objective_version=plan.objective_version,
            mandatory_anchor_job_id=plan.mandatory_anchor_job_id,
        )
        self._log_run_group_open(self._active_runs[group_id], selected_jobs, reason=plan.reason)
        if len(self._active_runs) > 1:
            for run in self._active_runs.values():
                run.overlapped = True
        self._last_telemetry_poll_at = 0.0

        if prepared_trial is not None:
            self.event_logger.emit(
                "colocation_trial_started",
                job_id=prepared_trial.candidate_job_id,
                payload=prepared_trial.to_dict(),
            )

        for index, job in enumerate(selected_jobs):
            if prepared_trial is not None and job.job_id == prepared_trial.candidate_job_id:
                role = "trial_newcomer"
            elif len(plan.job_ids) == 1:
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
                    "placement_objective_version": plan.objective_version,
                    "placement_objective_breakdown": plan.objective_breakdown,
                    "placement_trial_metadata": plan.trial_metadata,
                    "placement_mandatory_anchor_job_id": plan.mandatory_anchor_job_id,
                    **self._prediction_metadata(job.job_id),
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
                    "objective_version": plan.objective_version,
                    "objective_breakdown": plan.objective_breakdown,
                    "trial_metadata": plan.trial_metadata,
                    "mandatory_anchor_job_id": plan.mandatory_anchor_job_id,
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
                },
            )

    def _dispatch_pending_work(self) -> None:
        scheduler_mode = self.settings.gpu_scheduler.mode
        concurrent_mode = scheduler_mode in {
            SCHEDULER_MODE_PARALLEL_AUTO_PACK,
            SCHEDULER_MODE_PARALLEL_TIME_AWARE,
        }
        if not concurrent_mode and self._active_runs:
            return

        while True:
            active_job_ids = set(self._supervisor_active_job_ids())
            runnable = [job for job in self._runnable_jobs() if job.job_id not in active_job_ids]
            if not runnable:
                return
            if scheduler_mode == SCHEDULER_MODE_PARALLEL_TIME_AWARE and self.settings.gpu_scheduler.exclusive_probe.enabled:
                probes = [job for job in runnable if job.scheduling_class == SchedulingClass.EXCLUSIVE_PROBE]
                if self._exclusive_probe_job_id is None and probes:
                    reserved = sorted(probes, key=lambda job: self.policy.sort_key(job))[0]
                    self._exclusive_probe_job_id = reserved.job_id
                    self._persist_scheduler_decision_state()
                    self.event_logger.emit(
                        "exclusive_probe_drain_requested",
                        job_id=reserved.job_id,
                        payload={
                            "active_job_ids": sorted(active_job_ids),
                            "draining": bool(active_job_ids),
                        },
                    )
                if self._exclusive_probe_job_id is not None:
                    if active_job_ids:
                        return
                    reserved = next(
                        (job for job in runnable if job.job_id == self._exclusive_probe_job_id),
                        None,
                    )
                    if reserved is None:
                        self._exclusive_probe_job_id = None
                        self._persist_scheduler_decision_state()
                        continue
                    runnable = [reserved]
            active_vram_mb = self._active_vram_occupancy()
            active_jobs = self._active_jobs()
            plan = self.planner.choose_plan(
                runnable,
                backend_available=self.supervisor.available_backends(),
                active_vram_mb=active_vram_mb,
                active_jobs=active_jobs,
                admission_open=self._admission_gate.is_open,
                exclusive_drain_requested=bool(self._exclusive_probe_job_id and active_jobs),
                packing_admission_stalled=self._colocation_stall is not None,
                trial_pending=self._colocation_trial is not None,
            )
            if plan is None:
                return
            if concurrent_mode and self._active_runs and plan.backend_name == "exclusive":
                return
            dispatched = self._dispatch_plan(plan)
            if not dispatched or not concurrent_mode or plan.backend_name == "exclusive":
                return

    def report(self) -> dict[str, Any]:
        return {
            **self.metrics.as_dict(),
            "packed_admission_open": self._admission_gate.is_open,
            "average_memory_fraction": self._admission_gate.average_fraction,
            "exclusive_drain_requested": self._exclusive_probe_job_id is not None,
            "reserved_exclusive_probe_job_id": self._exclusive_probe_job_id,
            "packing_admission_stalled": self._colocation_stall is not None,
            "colocation_stall": self._colocation_stall.to_dict() if self._colocation_stall else None,
            "colocation_trial": self._colocation_trial.to_dict() if self._colocation_trial else None,
        }

    def cache_stats(self) -> dict[str, Any]:
        return {
            "stats": self.cache.stats().to_dict(),
            "entries": self.cache.snapshot_entries(),
        }
