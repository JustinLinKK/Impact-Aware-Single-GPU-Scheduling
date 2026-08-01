"""Metrics helpers built on top of persisted state."""

from __future__ import annotations

from typing import Any

from ..domain import SchedulerReport
from ..storage.state_store import StateStore


class MetricsCollector:
    """Build aggregate metrics from persisted scheduler state."""

    def __init__(self, store: StateStore):
        self.store = store

    def build_report(self) -> SchedulerReport:
        return self.store.report()

    def as_dict(self) -> dict[str, Any]:
        return self.build_report().to_dict()
