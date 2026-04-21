"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import tempfile

import torch

from ..observability.events import EventLogger
from ..schemas import SafePointType, TrainingJob, utc_now
from ..settings import SchedulerSettings
from ..storage.sqlite_store import SQLiteStateStore


class CheckpointManager:
    """Atomically save and restore checkpoints for scheduler-managed jobs."""

    def __init__(self, settings: SchedulerSettings, store: SQLiteStateStore, event_logger: EventLogger):
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
        self.event_logger.emit(
            "checkpoint_saved",
            job_id=job.job_id,
            payload={
                "checkpoint_path": str(final_path),
                "epoch": epoch,
                "global_step": global_step,
                "safe_point_type": safe_point_type.value,
                "reason": reason,
            },
        )
        self._prune(job)
        return str(final_path)

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
        for old_path in checkpoints[:-keep_last_n]:
            old_path.unlink(missing_ok=True)
