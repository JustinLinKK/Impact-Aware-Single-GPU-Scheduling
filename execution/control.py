"""File-based control and heartbeat channel between scheduler and worker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import json
import time

from ..atomic_io import atomic_json_dump as _atomic_json_dump
from ..checkpointing.manager import CheckpointManager
from ..observability.events import EventLogger
from ..domain import JobStatus, ProgressSnapshot, SafePointType, TrainingJob, utc_now
from ..config import SchedulerSettings
from ..storage.state_store import StateStore


class PauseRequested(RuntimeError):
    """Raised inside a worker when the scheduler requested a safe-point pause."""


class CancelRequested(RuntimeError):
    """Raised inside a worker when the scheduler requested cancellation."""


class EarlyStopRequested(RuntimeError):
    """Raised inside a worker when the scheduler stopped an unpromising run."""


@dataclass(slots=True)
class ControlCommand:
    action: str = "none"
    requested_at: str | None = None
    reason: str | None = None
    hold: bool = False
    transaction_id: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ControlCommand":
        payload = payload or {}
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "requested_at": self.requested_at,
            "reason": self.reason,
            "hold": self.hold,
            "transaction_id": self.transaction_id,
        }

class ControlPlane:
    """Read and write job control files."""

    def __init__(self, settings: SchedulerSettings):
        self.settings = settings

    def initialize_job(self, job_id: str) -> None:
        self.settings.job_runtime_dir(job_id).mkdir(parents=True, exist_ok=True)
        if not self.settings.job_command_path(job_id).exists():
            self.clear_command(job_id)

    def read_command(self, job_id: str) -> ControlCommand:
        path = self.settings.job_command_path(job_id)
        if not path.exists():
            return ControlCommand()
        with path.open("r", encoding="utf-8") as handle:
            return ControlCommand.from_dict(json.load(handle))

    def clear_command(self, job_id: str) -> None:
        _atomic_json_dump(self.settings.job_command_path(job_id), ControlCommand().to_dict())

    def request_pause(self, job_id: str, *, reason: str, hold: bool) -> None:
        _atomic_json_dump(
            self.settings.job_command_path(job_id),
            ControlCommand(action="pause", requested_at=utc_now(), reason=reason, hold=hold).to_dict(),
        )

    def request_cancel(self, job_id: str, *, reason: str) -> None:
        _atomic_json_dump(
            self.settings.job_command_path(job_id),
            ControlCommand(action="cancel", requested_at=utc_now(), reason=reason, hold=True).to_dict(),
        )

    def request_early_stop(self, job_id: str, *, reason: str) -> None:
        _atomic_json_dump(
            self.settings.job_command_path(job_id),
            ControlCommand(action="early_stop", requested_at=utc_now(), reason=reason, hold=True).to_dict(),
        )

    def request_repack_prepare(self, job_id: str, *, transaction_id: str, reason: str) -> None:
        self.settings.job_repack_ack_path(job_id).unlink(missing_ok=True)
        _atomic_json_dump(
            self.settings.job_command_path(job_id),
            ControlCommand(
                action="prepare_repack",
                requested_at=utc_now(),
                reason=reason,
                transaction_id=transaction_id,
            ).to_dict(),
        )

    def request_repack_commit(self, job_id: str, *, transaction_id: str) -> None:
        _atomic_json_dump(
            self.settings.job_command_path(job_id),
            ControlCommand(action="commit_repack", requested_at=utc_now(), transaction_id=transaction_id).to_dict(),
        )

    def request_repack_abort(self, job_id: str, *, transaction_id: str) -> None:
        _atomic_json_dump(
            self.settings.job_command_path(job_id),
            ControlCommand(action="abort_repack", requested_at=utc_now(), transaction_id=transaction_id).to_dict(),
        )

    def read_repack_ack(self, job_id: str) -> dict[str, Any] | None:
        path = self.settings.job_repack_ack_path(job_id)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                return dict(json.load(handle))
        except (OSError, ValueError, TypeError):
            return None

    def write_heartbeat(self, snapshot: ProgressSnapshot) -> None:
        _atomic_json_dump(self.settings.job_heartbeat_path(snapshot.job_id), snapshot.to_dict())

    def read_heartbeat(self, job_id: str) -> ProgressSnapshot | None:
        path = self.settings.job_heartbeat_path(job_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return ProgressSnapshot.from_dict(json.load(handle))


class TrainingControlHook:
    """Worker-side safe-point helper for pause/resume/cancel/checkpoint handling."""

    def __init__(
        self,
        job: TrainingJob,
        control_plane: ControlPlane,
        checkpoint_manager: CheckpointManager,
        store: StateStore,
        event_logger: EventLogger,
    ):
        self.job = job
        self.control_plane = control_plane
        self.checkpoint_manager = checkpoint_manager
        self.store = store
        self.event_logger = event_logger

    def _should_checkpoint(self, safe_point_type: SafePointType, *, epoch: int, global_step: int) -> bool:
        policy = self.job.checkpoint_policy
        if safe_point_type == SafePointType.EPOCH and policy.save_every_epoch:
            return True
        if safe_point_type == SafePointType.STEP and policy.save_every_n_steps:
            return global_step > 0 and global_step % policy.save_every_n_steps == 0
        if safe_point_type == SafePointType.EXPLICIT:
            return True
        return False

    def safe_point(
        self,
        safe_point_type: SafePointType,
        *,
        epoch: int,
        global_step: int,
        metrics: dict[str, float] | None = None,
        message: str | None = None,
        state_factory: Callable[[], dict[str, Any]] | None = None,
        steps_per_epoch: int | None = None,
        avg_step_time_ms: float | None = None,
        estimated_total_runtime_seconds: float | None = None,
        remaining_runtime_seconds: float | None = None,
    ) -> None:
        command = self.control_plane.read_command(self.job.job_id)
        snapshot = ProgressSnapshot(
            job_id=self.job.job_id,
            epoch=epoch,
            global_step=global_step,
            phase="train",
            metrics=metrics or {},
            last_safe_point=safe_point_type.value,
            message=message,
            steps_per_epoch=steps_per_epoch,
            avg_step_time_ms=avg_step_time_ms,
            estimated_total_runtime_seconds=estimated_total_runtime_seconds,
            remaining_runtime_seconds=remaining_runtime_seconds,
        )
        self.control_plane.write_heartbeat(snapshot)
        metadata_updates: dict[str, Any] | None = None
        if (
            estimated_total_runtime_seconds is not None
            or remaining_runtime_seconds is not None
            or avg_step_time_ms is not None
        ):
            metadata_updates = {}
            if estimated_total_runtime_seconds is not None:
                metadata_updates["runtime_estimated_total_runtime_seconds"] = float(estimated_total_runtime_seconds)
            if remaining_runtime_seconds is not None:
                metadata_updates["runtime_remaining_runtime_seconds"] = max(0.0, float(remaining_runtime_seconds))
            if steps_per_epoch is not None:
                metadata_updates["runtime_steps_per_epoch"] = int(steps_per_epoch)
            if avg_step_time_ms is not None:
                metadata_updates["runtime_avg_step_time_ms"] = float(avg_step_time_ms)
        self.store.update_job(self.job.job_id, last_heartbeat_at=snapshot.heartbeat_at, metadata_updates=metadata_updates)
        if hasattr(self.store, "record_job_metric_sample"):
            self.store.record_job_metric_sample(
                job_id=self.job.job_id,
                created_at=snapshot.heartbeat_at,
                epoch=epoch,
                global_step=global_step,
                avg_step_time_ms=avg_step_time_ms,
                estimated_total_runtime_seconds=estimated_total_runtime_seconds,
                remaining_runtime_seconds=remaining_runtime_seconds,
                metrics=snapshot.metrics,
            )
        if getattr(self.event_logger, "log_store", None) is not None:
            self.event_logger.log_store.record_job_metric_sample(
                job_id=self.job.job_id,
                created_at=snapshot.heartbeat_at,
                epoch=epoch,
                global_step=global_step,
                avg_step_time_ms=avg_step_time_ms,
                estimated_total_runtime_seconds=estimated_total_runtime_seconds,
                remaining_runtime_seconds=remaining_runtime_seconds,
                metrics=snapshot.metrics,
            )

        pause_requested = command.action == "pause"
        repack_requested = command.action == "prepare_repack"
        cancel_requested = command.action == "cancel"
        early_stop_requested = command.action == "early_stop"
        should_checkpoint = (
            pause_requested
            or repack_requested
            or (early_stop_requested and state_factory is not None)
            or self._should_checkpoint(safe_point_type, epoch=epoch, global_step=global_step)
        )
        checkpoint_path: str | None = None

        if should_checkpoint:
            if state_factory is None:
                raise RuntimeError("A checkpoint state_factory is required at checkpoint-capable safe points")
            checkpoint_started_at = time.perf_counter()
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                self.store.get_job(self.job.job_id) or self.job,
                state=state_factory(),
                safe_point_type=safe_point_type,
                epoch=epoch,
                global_step=global_step,
                reason=command.reason or ("scheduled checkpoint" if not pause_requested else "pause requested"),
            )
            checkpoint_overhead_seconds = max(0.0, time.perf_counter() - checkpoint_started_at)
            metadata_updates = dict(metadata_updates or {})
            metadata_updates["scheduler_checkpoint_overhead_seconds"] = checkpoint_overhead_seconds
            snapshot.checkpoint_path = checkpoint_path
            self.control_plane.write_heartbeat(snapshot)
            self.store.update_job(
                self.job.job_id,
                latest_checkpoint_path=checkpoint_path,
                last_heartbeat_at=snapshot.heartbeat_at,
                metadata_updates=metadata_updates,
            )

        if pause_requested:
            self.control_plane.clear_command(self.job.job_id)
            self.store.set_job_status(
                self.job.job_id,
                JobStatus.PAUSED,
                reason=command.reason or "pause requested",
                hold=command.hold,
            )
            self.event_logger.emit(
                "job_paused",
                job_id=self.job.job_id,
                payload={"checkpoint_path": checkpoint_path, "epoch": epoch, "global_step": global_step, "hold": command.hold},
            )
            raise PauseRequested(command.reason or "pause requested")

        if repack_requested:
            if not command.transaction_id:
                raise RuntimeError("repack preparation requires a transaction id")
            _atomic_json_dump(
                self.control_plane.settings.job_repack_ack_path(self.job.job_id),
                {
                    "transaction_id": command.transaction_id,
                    "checkpoint_path": checkpoint_path,
                    "epoch": epoch,
                    "global_step": global_step,
                    "acknowledged_at": utc_now(),
                },
            )
            self.event_logger.emit(
                "repack_checkpoint_ready",
                job_id=self.job.job_id,
                payload={
                    "transaction_id": command.transaction_id,
                    "checkpoint_path": checkpoint_path,
                    "epoch": epoch,
                    "global_step": global_step,
                },
            )
            while True:
                next_command = self.control_plane.read_command(self.job.job_id)
                if next_command.transaction_id != command.transaction_id:
                    time.sleep(0.05)
                    continue
                if next_command.action == "abort_repack":
                    self.control_plane.clear_command(self.job.job_id)
                    self.control_plane.settings.job_repack_ack_path(self.job.job_id).unlink(missing_ok=True)
                    self.event_logger.emit(
                        "repack_aborted",
                        job_id=self.job.job_id,
                        payload={"transaction_id": command.transaction_id},
                    )
                    return
                if next_command.action == "commit_repack":
                    self.control_plane.clear_command(self.job.job_id)
                    self.control_plane.settings.job_repack_ack_path(self.job.job_id).unlink(missing_ok=True)
                    self.store.set_job_status(
                        self.job.job_id,
                        JobStatus.PAUSED,
                        reason="adaptive repack committed",
                        hold=False,
                    )
                    raise PauseRequested("adaptive repack committed")
                time.sleep(0.05)

        if cancel_requested:
            self.control_plane.clear_command(self.job.job_id)
            self.store.set_job_status(self.job.job_id, JobStatus.CANCELLED, reason=command.reason or "cancel requested", hold=True)
            self.event_logger.emit(
                "job_cancelled",
                job_id=self.job.job_id,
                payload={"epoch": epoch, "global_step": global_step},
            )
            raise CancelRequested(command.reason or "cancel requested")

        if early_stop_requested:
            self.control_plane.clear_command(self.job.job_id)
            stopped_at = utc_now()
            self.store.update_job(
                self.job.job_id,
                status=JobStatus.EARLY_STOPPED,
                reason=command.reason or "early stop requested",
                hold=True,
                latest_checkpoint_path=checkpoint_path,
                metadata_updates={
                    "scheduler_early_stop_pending": False,
                    "scheduler_early_stop_completed_at": stopped_at,
                    "scheduler_early_stop_checkpoint_path": checkpoint_path,
                    "scheduler_early_stop_epoch": epoch,
                    "scheduler_early_stop_global_step": global_step,
                    "scheduler_early_stop_metrics": snapshot.metrics,
                },
            )
            self.event_logger.emit(
                "job_early_stopped",
                job_id=self.job.job_id,
                payload={
                    "checkpoint_path": checkpoint_path,
                    "epoch": epoch,
                    "global_step": global_step,
                    "metrics": snapshot.metrics,
                    "reason": command.reason,
                },
            )
            raise EarlyStopRequested(command.reason or "early stop requested")
