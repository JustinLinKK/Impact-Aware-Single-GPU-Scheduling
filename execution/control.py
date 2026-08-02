"""File-based control and heartbeat channel between scheduler and worker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import json
import time

from ..checkpointing.manager import CheckpointManager
from ..observability.events import EventLogger
from ..domain import JobStatus, ProgressSnapshot, SafePointType, TrainingJob, utc_now
from ..config import SchedulerSettings
from ..storage.state_store import StateStore
from ..scheduler.early_stopping import EarlyStoppingState, EarlyStoppingWatchdog


class PauseRequested(RuntimeError):
    """Raised inside a worker when the scheduler requested a safe-point pause."""


class CancelRequested(RuntimeError):
    """Raised inside a worker when the scheduler requested cancellation."""


class EarlyStopRequested(RuntimeError):
    """Normal worker control flow when validation progress has plateaued."""

    def __init__(self, result: dict[str, Any]):
        super().__init__("early_stopped_no_improvement")
        self.result = result


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
        store: StateStore,
        event_logger: EventLogger,
    ):
        self.job = job
        self.control_plane = control_plane
        self.checkpoint_manager = checkpoint_manager
        self.store = store
        self.event_logger = event_logger
        self.early_stopping = EarlyStoppingWatchdog(self.control_plane.settings.early_stopping)
        self._last_epoch_safe_point_monotonic: float | None = None
        self._last_epoch_safe_point_at: str | None = None

    def _trial_command_at_epoch(self, epoch: int) -> ControlCommand | None:
        current = self.store.get_job(self.job.job_id)
        trial = dict((current.metadata if current is not None else {}).get("colocation_trial") or {})
        if not trial or int(epoch) < int(trial.get("target_epoch") or 0):
            return None
        decision = str(trial.get("decision") or "pending")
        if decision in {"accepted", "completed_unverified"}:
            return ControlCommand()
        if decision in {"rejected", "timeout"}:
            return ControlCommand(
                action="pause",
                requested_at=utc_now(),
                reason=str(trial.get("reason") or "colocation trial rejected"),
                hold=False,
            )

        timeout = self.control_plane.settings.gpu_scheduler.colocation.trial_decision_timeout_seconds
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            command = self.control_plane.read_command(self.job.job_id)
            if command.action in {"pause", "cancel"}:
                return command
            current = self.store.get_job(self.job.job_id)
            trial = dict((current.metadata if current is not None else {}).get("colocation_trial") or {})
            if int(epoch) < int(trial.get("target_epoch") or 0):
                return ControlCommand()
            decision = str(trial.get("decision") or "pending")
            if decision in {"accepted", "completed_unverified"}:
                return ControlCommand()
            if decision in {"rejected", "timeout"}:
                return ControlCommand(
                    action="pause",
                    requested_at=utc_now(),
                    reason=str(trial.get("reason") or "colocation trial rejected"),
                    hold=False,
                )
            time.sleep(0.05)

        timeout_trial = {**trial, "decision": "timeout", "reason": "colocation trial decision timed out"}
        self.store.update_job(self.job.job_id, metadata_updates={"colocation_trial": timeout_trial})
        self.event_logger.emit(
            "colocation_trial_rejected",
            job_id=self.job.job_id,
            payload={"trial_id": timeout_trial.get("trial_id"), "reason": timeout_trial["reason"]},
        )
        return ControlCommand(
            action="pause",
            requested_at=utc_now(),
            reason=timeout_trial["reason"],
            hold=False,
        )

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
        observed_epoch_seconds: float | None = None
        epoch_interval_started_at: str | None = None
        epoch_interval_finished_at: str | None = None
        heartbeat_at = utc_now()
        if safe_point_type == SafePointType.EPOCH:
            now_monotonic = time.monotonic()
            epoch_interval_finished_at = heartbeat_at
            if self._last_epoch_safe_point_monotonic is not None:
                observed_epoch_seconds = max(0.0, now_monotonic - self._last_epoch_safe_point_monotonic)
                epoch_interval_started_at = self._last_epoch_safe_point_at
            elif avg_step_time_ms is not None and steps_per_epoch is not None:
                observed_epoch_seconds = max(0.0, float(avg_step_time_ms) * int(steps_per_epoch) / 1000.0)
            self._last_epoch_safe_point_monotonic = now_monotonic
            self._last_epoch_safe_point_at = heartbeat_at
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
            observed_epoch_seconds=observed_epoch_seconds,
            epoch_interval_started_at=epoch_interval_started_at,
            epoch_interval_finished_at=epoch_interval_finished_at,
            heartbeat_at=heartbeat_at,
        )
        self.control_plane.write_heartbeat(snapshot)
        metadata_updates: dict[str, Any] | None = None
        if any(
            value is not None
            for value in (
                estimated_total_runtime_seconds,
                remaining_runtime_seconds,
                steps_per_epoch,
                avg_step_time_ms,
            )
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
        if safe_point_type == SafePointType.EPOCH:
            metadata_updates = metadata_updates or {}
            metadata_updates["last_completed_epoch"] = int(epoch)
            if observed_epoch_seconds is not None and observed_epoch_seconds > 0:
                current = self.store.get_job(self.job.job_id)
                history = list((current.metadata if current is not None else {}).get("runtime_epoch_timing_history") or [])
                history.append(
                    {
                        "epoch": int(epoch),
                        "seconds": float(observed_epoch_seconds),
                        "started_at": epoch_interval_started_at,
                        "finished_at": epoch_interval_finished_at,
                        "source": "safe_point_interval" if epoch_interval_started_at else "runner_step_time",
                    }
                )
                metadata_updates.update(
                    {
                        "runtime_observed_epoch_seconds": float(observed_epoch_seconds),
                        "runtime_epoch_interval_started_at": epoch_interval_started_at,
                        "runtime_epoch_interval_finished_at": epoch_interval_finished_at,
                        "runtime_epoch_timing_history": history[-16:],
                    }
                )
        self.store.update_job(
            self.job.job_id,
            last_heartbeat_at=snapshot.heartbeat_at,
            metadata_updates=metadata_updates,
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

        early_decision = None
        early_state = EarlyStoppingState.from_dict(self.job.metadata.get("early_stopping_state"))
        if self.control_plane.settings.early_stopping.enabled and safe_point_type == SafePointType.EPOCH:
            early_decision = self.early_stopping.evaluate(epoch=epoch, metrics=snapshot.metrics, state=early_state)
            early_state = early_decision.state
            state_updates: dict[str, Any] = {
                "early_stopping_state": early_state.to_dict(),
                "last_completed_epoch": int(epoch),
            }
            self.job.metadata.update(state_updates)
            self.store.update_job(self.job.job_id, metadata_updates=state_updates)
            if early_decision.warning:
                self.event_logger.emit(
                    "early_stopping_metric_warning",
                    job_id=self.job.job_id,
                    payload={"epoch": epoch, "warning": early_decision.warning},
                )

        if safe_point_type == SafePointType.EPOCH:
            trial_command = self._trial_command_at_epoch(epoch)
            if trial_command is not None:
                command = trial_command

        pause_requested = command.action == "pause"
        cancel_requested = command.action == "cancel"
        save_best = bool(early_decision and early_decision.improved and self.control_plane.settings.early_stopping.save_best_checkpoint)
        should_checkpoint = pause_requested or save_best or self._should_checkpoint(safe_point_type, epoch=epoch, global_step=global_step)
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
            self.store.update_job(
                self.job.job_id,
                latest_checkpoint_path=checkpoint_path,
                last_heartbeat_at=snapshot.heartbeat_at,
                metadata_updates=metadata_updates,
            )
            if save_best:
                best_updates = {
                    "early_stopping_best_checkpoint_path": checkpoint_path,
                    "early_stopping_state": early_state.to_dict(),
                }
                self.job.metadata.update(best_updates)
                self.store.update_job(self.job.job_id, metadata_updates=best_updates)

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
                payload={
                    "checkpoint_path": checkpoint_path,
                    "epoch": epoch,
                    "global_step": global_step,
                    "hold": command.hold,
                },
            )
            raise PauseRequested(command.reason or "pause requested")

        if cancel_requested:
            self.control_plane.clear_command(self.job.job_id)
            self.store.set_job_status(
                self.job.job_id,
                JobStatus.CANCELLED,
                reason=command.reason or "cancel requested",
                hold=True,
            )
            self.event_logger.emit(
                "job_cancelled",
                job_id=self.job.job_id,
                payload={"epoch": epoch, "global_step": global_step},
            )
            raise CancelRequested(command.reason or "cancel requested")

        if early_decision is not None and early_decision.should_stop:
            total_epochs = self.job.max_epochs or self.job.config.max_epochs or epoch
            epochs_saved = max(0, int(total_epochs) - int(epoch))
            if remaining_runtime_seconds is not None:
                wall_time_saved_seconds = max(0.0, float(remaining_runtime_seconds))
            elif steps_per_epoch is not None and avg_step_time_ms is not None:
                wall_time_saved_seconds = max(0.0, epochs_saved * int(steps_per_epoch) * float(avg_step_time_ms) / 1000.0)
            else:
                wall_time_saved_seconds = 0.0
            result = {
                "reason": "early_stopped_no_improvement",
                "early_stopped_successfully": True,
                "best_metric": early_state.best_metric,
                "best_epoch": early_state.best_epoch,
                "stop_epoch": epoch,
                "patience_epochs": self.control_plane.settings.early_stopping.patience_epochs,
                "epochs_saved": epochs_saved,
                "estimated_wall_time_saved_seconds": wall_time_saved_seconds,
            }
            self.store.update_job(
                self.job.job_id,
                metadata_updates={
                    "early_stopping_result": result,
                    "last_completed_epoch": int(epoch),
                },
            )
            self.event_logger.emit("job_early_stopped", job_id=self.job.job_id, payload=result)
            raise EarlyStopRequested(result)
