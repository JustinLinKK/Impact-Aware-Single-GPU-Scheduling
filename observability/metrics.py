"""Metrics helpers built on top of persisted state."""

from __future__ import annotations

from typing import Any

from ..schemas import SchedulerReport
from ..storage.sqlite_store import SQLiteStateStore


class MetricsCollector:
    """Build aggregate metrics from persisted scheduler state."""

    def __init__(self, store: SQLiteStateStore):
        self.store = store

    def build_report(self) -> SchedulerReport:
        return self.store.report()

    def as_dict(self) -> dict[str, Any]:
        return self.build_report().to_dict()
