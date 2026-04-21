"""Structured event logging."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from ..schemas import utc_now
from ..storage.sqlite_store import SQLiteStateStore


class EventLogger:
    """Write scheduler events to SQLite and a JSONL sidecar."""

    def __init__(self, store: SQLiteStateStore, jsonl_path: Path):
        self.store = store
        self.jsonl_path = jsonl_path
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event_type: str, *, job_id: str | None = None, payload: dict[str, Any] | None = None) -> None:
        payload = payload or {}
        self.store.log_event(event_type, job_id=job_id, payload=payload)
        record = {
            "created_at": utc_now(),
            "job_id": job_id,
            "event_type": event_type,
            "payload": payload,
        }
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
