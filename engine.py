"""Runtime engine facade for the scheduler service."""

from __future__ import annotations

from typing import Any

from .client import SchedulerClient
from .config import SchedulerConfig
from .scheduler.service import SchedulerService


class SchedulerEngine:
    """Own the scheduler service lifecycle for one runtime root."""

    def __init__(self, settings: SchedulerConfig | None = None):
        self.settings = settings or SchedulerConfig()
        self.client = SchedulerClient(self.settings)
        self._service: SchedulerService | None = None

    def create_service(self, **kwargs: Any) -> SchedulerService:
        return self.client.create_service(**kwargs)

    def start(self, *, background: bool = False, **kwargs: Any) -> SchedulerService:
        self._service = self.create_service(**kwargs)
        return self._service.start(background=background)

    def stop(self) -> None:
        if self._service is not None:
            self._service.stop()
            self._service = None

    def heartbeat(self) -> dict[str, Any] | None:
        return self.client.scheduler_service_heartbeat()

    def is_active(self, *, max_staleness_seconds: float | None = None) -> bool:
        return self.client.scheduler_service_active(max_staleness_seconds=max_staleness_seconds)
