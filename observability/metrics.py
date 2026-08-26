"""Metrics helpers built on top of persisted state."""

from __future__ import annotations

from typing import Any
from collections import defaultdict
import threading

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


class CudaDocsMetrics:
    """Dependency-free in-process metrics for role-gated docs enrichment."""

    def __init__(self):
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._observations: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = defaultdict(list)
        self._gauges: dict[str, float] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(name: str, labels: dict[str, Any] | None) -> tuple[str, tuple[tuple[str, str], ...]]:
        return (
            str(name),
            tuple(sorted((str(key), str(value)) for key, value in (labels or {}).items())),
        )

    def increment(
        self,
        name: str,
        *,
        labels: dict[str, Any] | None = None,
        value: float = 1.0,
    ) -> None:
        with self._lock:
            self._counters[self._key(name, labels)] += float(value)

    def observe(
        self,
        name: str,
        value: float,
        *,
        labels: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._observations[self._key(name, labels)].append(float(value))

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[str(name)] = float(value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = {
                _metric_label(name, labels): value
                for (name, labels), value in self._counters.items()
            }
            observations = {
                _metric_label(name, labels): {
                    "count": len(values),
                    "sum": sum(values),
                    "max": max(values) if values else 0.0,
                }
                for (name, labels), values in self._observations.items()
            }
            return {
                "counters": counters,
                "observations": observations,
                "gauges": dict(self._gauges),
            }


def _metric_label(name: str, labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return name
    return name + "{" + ",".join(f"{key}={value}" for key, value in labels) + "}"
