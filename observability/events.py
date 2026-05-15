"""Structured event logging."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from ..domain import utc_now
from ..storage.log_store import SchedulerLogStore
from ..storage.state_store import StateStore


class EventLogger:
    """Write scheduler events to the primary store and optional sidecars."""

    def __init__(self, store: StateStore, jsonl_path: Path, log_store: SchedulerLogStore | None = None):
        self.store = store
        self.jsonl_path = jsonl_path
        self.log_store = log_store
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event_type: str, *, job_id: str | None = None, payload: dict[str, Any] | None = None) -> None:
        payload = payload or {}
        created_at = utc_now()
        self.store.log_event(event_type, job_id=job_id, payload=payload)
        if self.log_store is not None:
            self.log_store.record_event(job_id=job_id, event_type=event_type, created_at=created_at, payload=payload)
        record = {
            "created_at": created_at,
            "job_id": job_id,
            "event_type": event_type,
            "payload": payload,
        }
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
