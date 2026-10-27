"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math
import os
import tempfile
import time

import torch

from ..observability.events import EventLogger
from ..domain import SafePointType, TrainingJob, utc_now
from ..config import SchedulerSettings
from ..storage.state_store import StateStore


class CheckpointManager:
    """Atomically save and restore checkpoints for scheduler-managed jobs."""

    def __init__(self, settings: SchedulerSettings, store: StateStore, event_logger: EventLogger):
        self.settings = settings
        self.store = store
        self.event_logger = event_logger

    def checkpoint_dir(self, job_id: str) -> Path:
        path = self.settings.checkpoints_for_job(job_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_checkpoint(
        self,
        job: TrainingJob,
        *,
        state: dict[str, Any],
        safe_point_type: SafePointType,
        epoch: int,
        global_step: int,
        reason: str,
    ) -> str:
        started = time.perf_counter()
        directory = self.checkpoint_dir(job.job_id)
        filename = f"checkpoint_step_{global_step:08d}_epoch_{epoch:04d}.pt"
        final_path = directory / filename
        payload = {
            "job": job.to_dict(),
            "state": state,
            "saved_at": utc_now(),
            "safe_point_type": safe_point_type.value,
            "epoch": epoch,
            "global_step": global_step,
            "reason": reason,
        }
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".tmp_checkpoint_", suffix=".pt", delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            torch.save(payload, temp_path)
            os.replace(temp_path, final_path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
        self.store.record_checkpoint(
            job.job_id,
            str(final_path),
            metadata={"epoch": epoch, "global_step": global_step, "reason": reason},
        )
        self._prune(job)
        checkpoint_seconds = time.perf_counter() - started
        self._record_completion_timing(job, checkpoint_seconds, safe_point_type, epoch, global_step)
        self.event_logger.emit(
            "checkpoint_saved",
            job_id=job.job_id,
            payload={
                "checkpoint_path": str(final_path),
                "epoch": epoch,
                "global_step": global_step,
                "safe_point_type": safe_point_type.value,
                "reason": reason,
                "checkpoint_seconds": checkpoint_seconds,
            },
        )
        return str(final_path)

    def _record_completion_timing(self, job, seconds, safe_point_type, epoch, global_step):
        identity = job.metadata.get("runtime_completion_identity")
        if not isinstance(identity, str) or not identity or not math.isfinite(seconds) or seconds < 0:
            return
        current = self.store.get_job(job.job_id) or job
        if current.metadata.get("runtime_completion_identity") != identity:
            return
        history = list(current.metadata.get("runtime_phase_timing_history") or [])
        history.append({"phase": "checkpoint", "seconds": float(seconds), "identity": identity,
                        "hardware_key": self.store.hardware_key(),
                        "epoch": int(epoch), "global_step": int(global_step),
                        "finished_at": utc_now(), "source": "checkpoint_manager.save_checkpoint"})
        updates = {"runtime_phase_timing_history": history[-64:],
                   "runtime_nonupdate_phase_counter": int(current.metadata.get("runtime_nonupdate_phase_counter") or 0) + 1,
                   "runtime_checkpoint_completed_step": int(global_step)}
        if safe_point_type == SafePointType.EPOCH:
            updates["runtime_checkpoint_completed_epoch"] = int(epoch)
        if safe_point_type == SafePointType.EXPLICIT:
            updates["runtime_final_checkpoint_complete"] = True
        job.metadata.update(updates)
        self.store.update_job(job.job_id, metadata_updates=updates)

    def load_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any]:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    def latest_checkpoint(self, job: TrainingJob) -> str | None:
        return job.latest_checkpoint_path or self.store.latest_checkpoint(job.job_id)

    def _prune(self, job: TrainingJob) -> None:
        keep_last_n = max(1, job.checkpoint_policy.keep_last_n)
        directory = self.checkpoint_dir(job.job_id)
        checkpoints = sorted(directory.glob("checkpoint_step_*.pt"))
        if len(checkpoints) <= keep_last_n:
            return
        best_path = job.metadata.get("early_stopping_best_checkpoint_path")
        preserved = Path(str(best_path)).resolve() if best_path else None
        for old_path in checkpoints[:-keep_last_n]:
            if preserved is not None and old_path.resolve() == preserved:
                continue
            old_path.unlink(missing_ok=True)
