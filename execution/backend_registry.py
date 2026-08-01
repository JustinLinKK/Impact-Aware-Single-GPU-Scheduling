"""Execution backend registry and capability lookup."""

from __future__ import annotations

from typing import Iterable

from ..config import SchedulerSettings
from .backends import CudaProcessBackend, ExclusiveBackend, ExecutionBackend, MPSBackend, StreamBackend
from .executor import SubprocessExecutor


class BackendRegistry:
    def __init__(
        self,
        settings: SchedulerSettings,
        executor: SubprocessExecutor,
        *,
        backends: dict[str, ExecutionBackend] | None = None,
    ):
        self.settings = settings
        self.executor = executor
        self._backends = backends or {
            "exclusive": ExclusiveBackend(settings, executor),
            "mps": MPSBackend(settings, executor),
            "cuda_process": CudaProcessBackend(settings, executor),
            "stream": StreamBackend(settings, executor),
        }

    def get(self, backend_name: str) -> ExecutionBackend | None:
        return self._backends.get(backend_name)

    def items(self) -> Iterable[tuple[str, ExecutionBackend]]:
        return self._backends.items()

    def availability(self) -> dict[str, bool]:
        return {name: backend.available() for name, backend in self._backends.items()}

