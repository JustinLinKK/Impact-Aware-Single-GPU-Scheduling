"""Runner protocol and worker context for custom training loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Protocol, runtime_checkable

import torch

from ..model_cache.baseline_cache import _materialize_payload_bytes
from ..model_cache.cache_server import CacheClient
from ..checkpointing.manager import CheckpointManager
from ..execution.control import TrainingControlHook
from ..observability.events import EventLogger
from ..schemas import BatchProbeTrialResult, TrainingJob
from ..settings import SchedulerSettings
from ..storage.sqlite_store import SQLiteStateStore


@runtime_checkable
class RunnerProtocol(Protocol):
    """Custom training integration contract."""

    def __call__(self, context: "RunnerContext") -> dict[str, Any] | None:
        ...


@runtime_checkable
class BatchProbeProtocol(Protocol):
    """Optional runner-side contract for exclusive batch-size probing."""

    def __call__(
        self,
        context: "RunnerContext",
        batch_size: int,
        warmup_steps: int,
        measure_steps: int,
    ) -> BatchProbeTrialResult:
        ...


@dataclass(slots=True)
class RunnerContext:
    """Context object handed to custom training runners."""

    job: TrainingJob
    settings: SchedulerSettings
    store: SQLiteStateStore
    event_logger: EventLogger
    control_hook: TrainingControlHook
    checkpoint_manager: CheckpointManager
    cache_client: CacheClient | None = None
    _baseline_cache: Any = field(default=None, init=False, repr=False)
    _resume_checkpoint_cache: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def load_baseline_object(self) -> Any:
        if self._baseline_cache is not None:
            return self._baseline_cache
        payload_bytes: bytes
        if self.cache_client is not None:
            try:
                payload_bytes = self.cache_client.get(
                    self.job.baseline_model_id,
                    self.job.baseline_model_path,
                    loader_target=self.job.config.loader_target,
                    metadata={"job_id": self.job.job_id},
                )
            except Exception:
                payload_bytes = _materialize_payload_bytes(
                    self.job.baseline_model_path,
                    loader_target=self.job.config.loader_target,
                )
        else:
            payload_bytes = _materialize_payload_bytes(
                self.job.baseline_model_path,
                loader_target=self.job.config.loader_target,
            )
        self._baseline_cache = torch.load(BytesIO(payload_bytes), map_location="cpu", weights_only=False)
        return self._baseline_cache

    def load_resume_checkpoint(self) -> dict[str, Any] | None:
        if self._resume_checkpoint_cache is not None:
            return self._resume_checkpoint_cache
        checkpoint_path = self.job.resume_from_checkpoint or self.job.latest_checkpoint_path or self.store.latest_checkpoint(self.job.job_id)
        if not checkpoint_path:
            return None
        self._resume_checkpoint_cache = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        return self._resume_checkpoint_cache
