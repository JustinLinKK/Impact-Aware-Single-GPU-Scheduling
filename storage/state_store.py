"""Runtime state-store facade."""

from __future__ import annotations

from typing import Any
import logging

from ..config import SchedulerSettings
from .neo4j_store import Neo4jStateStore
from .sqlite_store import SQLiteStateStore as LegacySQLiteStateStore

LOGGER = logging.getLogger(__name__)


class _MirrorStateStore:
    """Write-through SQLite store with best-effort graph mirroring."""

    def __init__(self, primary: LegacySQLiteStateStore, mirror: Neo4jStateStore):
        self._primary = primary
        self._mirror = mirror
        self._warned_methods: set[str] = set()

    @property
    def backend(self) -> LegacySQLiteStateStore:
        return self._primary

    @property
    def mirror_backend(self) -> Neo4jStateStore:
        return self._mirror

    def _warn_once(self, method_name: str, exc: Exception) -> None:
        if method_name in self._warned_methods:
            return
        self._warned_methods.add(method_name)
        LOGGER.warning("Graph mirror write failed for %s; continuing with SQLite primary only: %s", method_name, exc)

    def _mirror_call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        method = getattr(self._mirror, method_name, None)
        if not callable(method):
            return None
        try:
            return method(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - warning path
            self._warn_once(method_name, exc)
            return None

    def save_job(self, job):
        result = self._primary.save_job(job)
        self._mirror_call("record_scheduler_job_evidence", job)
        return result

    def submit_job(self, job):
        result = self._primary.submit_job(job)
        self._mirror_call("record_scheduler_job_evidence", result)
        return result

    def update_job(self, *args: Any, **kwargs: Any):
        result = self._primary.update_job(*args, **kwargs)
        self._mirror_call("record_scheduler_job_evidence", result)
        return result

    def set_job_status(self, *args: Any, **kwargs: Any):
        result = self._primary.set_job_status(*args, **kwargs)
        self._mirror_call("record_scheduler_job_evidence", result)
        return result

    def delete_job(self, job_id: str) -> None:
        self._primary.delete_job(job_id)
        self._mirror_call("delete_job", job_id)

    def log_event(self, event_type: str, *, job_id: str | None = None, payload: dict[str, Any] | None = None) -> None:
        self._primary.log_event(event_type, job_id=job_id, payload=payload)

    def record_checkpoint(self, job_id: str, checkpoint_path: str, metadata: dict[str, Any] | None = None) -> None:
        self._primary.record_checkpoint(job_id, checkpoint_path, metadata=metadata)

    def update_cache_metadata(self, *args: Any, **kwargs: Any) -> None:
        self._primary.update_cache_metadata(*args, **kwargs)

    def upsert_solo_profile(self, profile):
        result = self._primary.upsert_solo_profile(profile)
        self._mirror_call("record_solo_profile_evidence", result)
        return result

    def upsert_pair_profile(self, profile):
        result = self._primary.upsert_pair_profile(profile)
        self._mirror_call("record_pair_profile_evidence", result)
        return result

    def upsert_runtime_profile(self, profile):
        result = self._primary.upsert_runtime_profile(profile)
        self._mirror_call("record_runtime_profile_evidence", result)
        return result

    def upsert_batch_probe_profile(self, profile):
        result = self._primary.upsert_batch_probe_profile(profile)
        self._mirror_call("record_batch_probe_evidence", result)
        return result

    def upsert_batch_size_observation(self, observation):
        result = self._primary.upsert_batch_size_observation(observation)
        self._mirror_call("record_batch_size_observation_evidence", result)
        return result

    def upsert_combination_profile(self, profile):
        result = self._primary.upsert_combination_profile(profile)
        self._mirror_call("record_combination_profile_evidence", result)
        return result

    def mark_pair_incompatible(self, *args: Any, **kwargs: Any):
        result = self._primary.mark_pair_incompatible(*args, **kwargs)
        self._mirror_call("mark_pair_incompatible", *args, **kwargs)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._primary, name)


class StateStore:
    """Route runtime storage according to the configured graph mode."""

    def __init__(self, settings: SchedulerSettings):
        self.settings = settings
        self._backend = self._build_backend()

    def _build_graph_backend(self) -> Neo4jStateStore | None:
        provider = str(self.settings.graph_db.provider or "neo4j").strip().lower().replace("-", "_")
        if provider != "neo4j":
            return None
        return Neo4jStateStore(self.settings)

    def _build_backend(self):
        graph_settings = self.settings.graph_db
        mode = str(graph_settings.mode or "primary").strip().lower().replace("-", "_")
        if not graph_settings.enabled:
            mode = "off"

        primary_sqlite = LegacySQLiteStateStore(self.settings)
        if mode == "off":
            return primary_sqlite

        # SQLite remains the scheduler control-plane source of truth. When the
        # graph is enabled, Neo4j is used only as a best-effort evidence mirror
        # for measured profiles and terminal job outcomes, even if legacy
        # configs still say graph_db.mode=primary.
        try:
            mirror = self._build_graph_backend()
        except Exception as exc:
            if not graph_settings.allow_legacy_fallback:
                raise
            LOGGER.warning("Graph evidence mirror unavailable; continuing with SQLite primary only: %s", exc)
            return primary_sqlite
        if mirror is None:
            return primary_sqlite
        return _MirrorStateStore(primary_sqlite, mirror)

    @property
    def backend(self):
        if hasattr(self._backend, "backend"):
            return self._backend.backend
        return self._backend

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)
