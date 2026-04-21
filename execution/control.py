"""File-based control and heartbeat channel between scheduler and worker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import json

from ..checkpointing.manager import CheckpointManager
from ..observability.events import EventLogger
from ..schemas import JobStatus, ProgressSnapshot, SafePointType, TrainingJob, utc_now
from ..settings import SchedulerSettings
from ..storage.sqlite_store import SQLiteStateStore


class PauseRequested(RuntimeError):
    """Raised inside a worker when the scheduler requested a safe-point pause."""


class CancelRequested(RuntimeError):
    """Raised inside a worker when the scheduler requested cancellation."""


@dataclass(slots=True)
class ControlCommand:
    action: str = "none"
    requested_at: str | None = None
    reason: str | None = None
    hold: bool = False

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
        }


def _atomic_json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    temp_path.replace(path)


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
        store: SQLiteStateStore,
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
        )
        self.control_plane.write_heartbeat(snapshot)
        self.store.update_job(self.job.job_id, last_heartbeat_at=snapshot.heartbeat_at)

        pause_requested = command.action == "pause"
        cancel_requested = command.action == "cancel"
        should_checkpoint = pause_requested or self._should_checkpoint(safe_point_type, epoch=epoch, global_step=global_step)
        checkpoint_path: str | None = None

        if should_checkpoint:
            if state_factory is None:
                raise RuntimeError("A checkpoint state_factory is required at checkpoint-capable safe points")
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                self.store.get_job(self.job.job_id) or self.job,
                state=state_factory(),
                safe_point_type=safe_point_type,
                epoch=epoch,
                global_step=global_step,
                reason=command.reason or ("scheduled checkpoint" if not pause_requested else "pause requested"),
            )
            snapshot.checkpoint_path = checkpoint_path
            self.control_plane.write_heartbeat(snapshot)
            self.store.update_job(self.job.job_id, latest_checkpoint_path=checkpoint_path, last_heartbeat_at=snapshot.heartbeat_at)

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

        if cancel_requested:
            self.control_plane.clear_command(self.job.job_id)
            self.store.set_job_status(self.job.job_id, JobStatus.CANCELLED, reason=command.reason or "cancel requested", hold=True)
            self.event_logger.emit(
                "job_cancelled",
                job_id=self.job.job_id,
                payload={"epoch": epoch, "global_step": global_step},
            )
            raise CancelRequested(command.reason or "cancel requested")
