"""Best-effort Postgres log store for scheduler activity and metrics."""

from __future__ import annotations

from typing import Any
import json
import logging
import os
import uuid

from ..config import SchedulerSettings

try:
    import psycopg
except Exception:  # pragma: no cover - optional dependency
    psycopg = None

LOGGER = logging.getLogger(__name__)


class SchedulerLogStore:
    """Append-only analytics store for activity and metric timelines."""

    def __init__(self, settings: SchedulerSettings):
        self.settings = settings
        self.enabled = bool(settings.log_db.enabled)
        self.schema = settings.log_db.schema
        self._dsn = os.getenv(settings.log_db.dsn_env, "")
        self._current_session_id: str | None = None
        self._warned = False
        self._warning_messages: set[str] = set()
        if self.enabled and not self._dsn:
            self._warn_once(f"log_db enabled but {settings.log_db.dsn_env} is unset; continuing without Postgres logging")
        if self.enabled and psycopg is None:
            self._warn_once("log_db enabled but psycopg is unavailable; continuing without Postgres logging")
        if self.enabled:
            self.initialize()

    def _available(self) -> bool:
        return bool(self.enabled and self._dsn and psycopg is not None)

    def _warn_once(self, message: str, exc: Exception | None = None) -> None:
        if message in self._warning_messages:
            return
        self._warning_messages.add(message)
        self._warned = True
        if exc is None:
            LOGGER.warning(message)
        else:
            LOGGER.warning("%s: %s", message, exc)

    def _connect(self):
        if not self._available():
            raise RuntimeError("scheduler log database is unavailable")
        return psycopg.connect(self._dsn, autocommit=True)

    def _safe(self, fn, *args, **kwargs):
        try:
            if not self._available():
                if self.enabled:
                    self._warn_once("scheduler log database is unavailable; continuing without append-only analytics")
                return None
            return fn(*args, **kwargs)
        except Exception as exc:
            self._warn_once("scheduler log database write failed; continuing without append-only analytics", exc)
            return None

    def initialize(self) -> None:
        self._safe(self._initialize_impl)

    def _initialize_impl(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.schema}.scheduler_sessions (
                        session_id TEXT PRIMARY KEY,
                        started_at TIMESTAMPTZ NOT NULL,
                        stopped_at TIMESTAMPTZ,
                        status TEXT NOT NULL,
                        pid INTEGER,
                        runtime_root TEXT,
                        host_identity JSONB,
                        config_json JSONB
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.schema}.run_groups (
                        group_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        mode TEXT,
                        backend_name TEXT,
                        hardware_key TEXT,
                        group_signature TEXT,
                        opened_at TIMESTAMPTZ NOT NULL,
                        first_sample_at TIMESTAMPTZ,
                        closed_at TIMESTAMPTZ,
                        overlapped BOOLEAN,
                        fallback_triggered BOOLEAN,
                        fallback_reason TEXT,
                        exit_reason TEXT,
                        metadata_json JSONB
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.schema}.run_group_members (
                        member_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        group_id TEXT,
                        job_id TEXT,
                        role TEXT,
                        batch_size INTEGER,
                        joined_at TIMESTAMPTZ NOT NULL,
                        left_at TIMESTAMPTZ,
                        metadata_json JSONB
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.schema}.job_activity_log (
                        activity_id BIGSERIAL PRIMARY KEY,
                        session_id TEXT,
                        job_id TEXT,
                        event_type TEXT,
                        created_at TIMESTAMPTZ NOT NULL,
                        payload_json JSONB
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.schema}.job_metric_samples (
                        sample_id BIGSERIAL PRIMARY KEY,
                        session_id TEXT,
                        job_id TEXT,
                        created_at TIMESTAMPTZ NOT NULL,
                        epoch INTEGER,
                        global_step INTEGER,
                        avg_step_time_ms DOUBLE PRECISION,
                        estimated_total_runtime_seconds DOUBLE PRECISION,
                        remaining_runtime_seconds DOUBLE PRECISION,
                        metrics_json JSONB
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.schema}.gpu_metric_samples (
                        sample_id BIGSERIAL PRIMARY KEY,
                        session_id TEXT,
                        group_id TEXT,
                        created_at TIMESTAMPTZ NOT NULL,
                        backend_name TEXT,
                        hardware_key TEXT,
                        memory_used_mb INTEGER,
                        memory_total_mb INTEGER,
                        gpu_utilization DOUBLE PRECISION,
                        memory_utilization DOUBLE PRECISION,
                        job_ids_json JSONB
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.schema}.probe_activity_log (
                        probe_id BIGSERIAL PRIMARY KEY,
                        session_id TEXT,
                        job_id TEXT,
                        event_type TEXT,
                        created_at TIMESTAMPTZ NOT NULL,
                        payload_json JSONB
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.schema}.cache_activity_log (
                        cache_id BIGSERIAL PRIMARY KEY,
                        session_id TEXT,
                        event_type TEXT,
                        model_id TEXT,
                        created_at TIMESTAMPTZ NOT NULL,
                        payload_json JSONB
                    )
                    """
                )

    @property
    def current_session_id(self) -> str | None:
        return self._current_session_id

    def start_session(self, *, status: str, pid: int, runtime_root: str, host_identity: dict[str, Any], config_json: dict[str, Any], started_at: str) -> str | None:
        session_id = str(uuid.uuid4())
        self._current_session_id = session_id
        self._safe(
            self._insert_session,
            session_id=session_id,
            status=status,
            pid=pid,
            runtime_root=runtime_root,
            host_identity=host_identity,
            config_json=config_json,
            started_at=started_at,
        )
        return session_id

    def _insert_session(self, **kwargs: Any) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.schema}.scheduler_sessions(
                        session_id, started_at, status, pid, runtime_root, host_identity, config_json
                    )
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    ON CONFLICT (session_id) DO NOTHING
                    """,
                    (
                        kwargs["session_id"],
                        kwargs["started_at"],
                        kwargs["status"],
                        kwargs["pid"],
                        kwargs["runtime_root"],
                        json.dumps(kwargs["host_identity"], sort_keys=True),
                        json.dumps(kwargs["config_json"], sort_keys=True),
                    ),
                )

    def finish_session(self, *, status: str, stopped_at: str) -> None:
        if not self._current_session_id:
            return
        self._safe(self._finish_session_impl, status=status, stopped_at=stopped_at)

    def _finish_session_impl(self, *, status: str, stopped_at: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE {self.schema}.scheduler_sessions
                    SET status = %s, stopped_at = %s
                    WHERE session_id = %s
                    """,
                    (status, stopped_at, self._current_session_id),
                )

    def record_event(self, *, job_id: str | None, event_type: str, created_at: str, payload: dict[str, Any]) -> None:
        self._safe(self._record_event_impl, job_id=job_id, event_type=event_type, created_at=created_at, payload=payload)

    def _record_event_impl(self, *, job_id: str | None, event_type: str, created_at: str, payload: dict[str, Any]) -> None:
        table = "job_activity_log"
        if event_type.startswith("batch_probe") or event_type.startswith("runtime_probe"):
            table = "probe_activity_log"
        elif event_type.startswith("cache_"):
            table = "cache_activity_log"
        with self._connect() as conn:
            with conn.cursor() as cur:
                if table == "cache_activity_log":
                    cur.execute(
                        f"""
                        INSERT INTO {self.schema}.cache_activity_log(session_id, event_type, model_id, created_at, payload_json)
                        VALUES (%s, %s, %s, %s, %s::jsonb)
                        """,
                        (
                            self._current_session_id,
                            event_type,
                            payload.get("model_id"),
                            created_at,
                            json.dumps(payload, sort_keys=True),
                        ),
                    )
                elif table == "probe_activity_log":
                    cur.execute(
                        f"""
                        INSERT INTO {self.schema}.probe_activity_log(session_id, job_id, event_type, created_at, payload_json)
                        VALUES (%s, %s, %s, %s, %s::jsonb)
                        """,
                        (
                            self._current_session_id,
                            job_id,
                            event_type,
                            created_at,
                            json.dumps(payload, sort_keys=True),
                        ),
                    )
                else:
                    cur.execute(
                        f"""
                        INSERT INTO {self.schema}.job_activity_log(session_id, job_id, event_type, created_at, payload_json)
                        VALUES (%s, %s, %s, %s, %s::jsonb)
                        """,
                        (
                            self._current_session_id,
                            job_id,
                            event_type,
                            created_at,
                            json.dumps(payload, sort_keys=True),
                        ),
                    )

    def record_job_metric_sample(self, *, job_id: str, created_at: str, epoch: int, global_step: int, avg_step_time_ms: float | None, estimated_total_runtime_seconds: float | None, remaining_runtime_seconds: float | None, metrics: dict[str, Any]) -> None:
        self._safe(
            self._record_job_metric_sample_impl,
            job_id=job_id,
            created_at=created_at,
            epoch=epoch,
            global_step=global_step,
            avg_step_time_ms=avg_step_time_ms,
            estimated_total_runtime_seconds=estimated_total_runtime_seconds,
            remaining_runtime_seconds=remaining_runtime_seconds,
            metrics=metrics,
        )

    def _record_job_metric_sample_impl(self, **kwargs: Any) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.schema}.job_metric_samples(
                        session_id, job_id, created_at, epoch, global_step,
                        avg_step_time_ms, estimated_total_runtime_seconds,
                        remaining_runtime_seconds, metrics_json
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        self._current_session_id,
                        kwargs["job_id"],
                        kwargs["created_at"],
                        kwargs["epoch"],
                        kwargs["global_step"],
                        kwargs["avg_step_time_ms"],
                        kwargs["estimated_total_runtime_seconds"],
                        kwargs["remaining_runtime_seconds"],
                        json.dumps(kwargs["metrics"], sort_keys=True),
                    ),
                )

    def open_run_group(self, *, group_id: str, mode: str, backend_name: str, hardware_key: str, group_signature: str, opened_at: str, overlapped: bool, metadata: dict[str, Any]) -> None:
        self._safe(
            self._open_run_group_impl,
            group_id=group_id,
            mode=mode,
            backend_name=backend_name,
            hardware_key=hardware_key,
            group_signature=group_signature,
            opened_at=opened_at,
            overlapped=overlapped,
            metadata=metadata,
        )

    def _open_run_group_impl(self, **kwargs: Any) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.schema}.run_groups(
                        group_id, session_id, mode, backend_name, hardware_key,
                        group_signature, opened_at, overlapped, fallback_triggered,
                        metadata_json
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, false, %s::jsonb)
                    ON CONFLICT (group_id) DO NOTHING
                    """,
                    (
                        kwargs["group_id"],
                        self._current_session_id,
                        kwargs["mode"],
                        kwargs["backend_name"],
                        kwargs["hardware_key"],
                        kwargs["group_signature"],
                        kwargs["opened_at"],
                        kwargs["overlapped"],
                        json.dumps(kwargs["metadata"], sort_keys=True),
                    ),
                )

    def upsert_run_group_member(self, *, group_id: str, job_id: str, role: str, batch_size: int | None, joined_at: str, metadata: dict[str, Any]) -> None:
        self._safe(
            self._upsert_run_group_member_impl,
            group_id=group_id,
            job_id=job_id,
            role=role,
            batch_size=batch_size,
            joined_at=joined_at,
            metadata=metadata,
        )

    def _upsert_run_group_member_impl(self, **kwargs: Any) -> None:
        member_id = f"{kwargs['group_id']}::{kwargs['job_id']}"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.schema}.run_group_members(
                        member_id, session_id, group_id, job_id, role, batch_size,
                        joined_at, metadata_json
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (member_id) DO UPDATE
                    SET role = EXCLUDED.role,
                        batch_size = EXCLUDED.batch_size,
                        metadata_json = EXCLUDED.metadata_json
                    """,
                    (
                        member_id,
                        self._current_session_id,
                        kwargs["group_id"],
                        kwargs["job_id"],
                        kwargs["role"],
                        kwargs["batch_size"],
                        kwargs["joined_at"],
                        json.dumps(kwargs["metadata"], sort_keys=True),
                    ),
                )

    def mark_run_group_member_left(self, *, group_id: str, job_id: str, left_at: str) -> None:
        self._safe(self._mark_run_group_member_left_impl, group_id=group_id, job_id=job_id, left_at=left_at)

    def _mark_run_group_member_left_impl(self, *, group_id: str, job_id: str, left_at: str) -> None:
        member_id = f"{group_id}::{job_id}"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE {self.schema}.run_group_members
                    SET left_at = COALESCE(left_at, %s)
                    WHERE member_id = %s
                    """,
                    (left_at, member_id),
                )

    def record_gpu_metric_sample(self, *, group_id: str, created_at: str, backend_name: str, hardware_key: str, memory_used_mb: int, memory_total_mb: int, gpu_utilization: float, memory_utilization: float, job_ids: list[str]) -> None:
        self._safe(
            self._record_gpu_metric_sample_impl,
            group_id=group_id,
            created_at=created_at,
            backend_name=backend_name,
            hardware_key=hardware_key,
            memory_used_mb=memory_used_mb,
            memory_total_mb=memory_total_mb,
            gpu_utilization=gpu_utilization,
            memory_utilization=memory_utilization,
            job_ids=job_ids,
        )

    def _record_gpu_metric_sample_impl(self, **kwargs: Any) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.schema}.gpu_metric_samples(
                        session_id, group_id, created_at, backend_name, hardware_key,
                        memory_used_mb, memory_total_mb, gpu_utilization,
                        memory_utilization, job_ids_json
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        self._current_session_id,
                        kwargs["group_id"],
                        kwargs["created_at"],
                        kwargs["backend_name"],
                        kwargs["hardware_key"],
                        kwargs["memory_used_mb"],
                        kwargs["memory_total_mb"],
                        kwargs["gpu_utilization"],
                        kwargs["memory_utilization"],
                        json.dumps(kwargs["job_ids"], sort_keys=True),
                    ),
                )
                cur.execute(
                    f"""
                    UPDATE {self.schema}.run_groups
                    SET first_sample_at = COALESCE(first_sample_at, %s)
                    WHERE group_id = %s
                    """,
                    (kwargs["created_at"], kwargs["group_id"]),
                )

    def close_run_group(self, *, group_id: str, closed_at: str, overlapped: bool, fallback_triggered: bool, fallback_reason: str | None, exit_reason: str | None) -> None:
        self._safe(
            self._close_run_group_impl,
            group_id=group_id,
            closed_at=closed_at,
            overlapped=overlapped,
            fallback_triggered=fallback_triggered,
            fallback_reason=fallback_reason,
            exit_reason=exit_reason,
        )

    def _close_run_group_impl(self, **kwargs: Any) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE {self.schema}.run_groups
                    SET closed_at = %s,
                        overlapped = %s,
                        fallback_triggered = %s,
                        fallback_reason = %s,
                        exit_reason = %s
                    WHERE group_id = %s
                    """,
                    (
                        kwargs["closed_at"],
                        kwargs["overlapped"],
                        kwargs["fallback_triggered"],
                        kwargs["fallback_reason"],
                        kwargs["exit_reason"],
                        kwargs["group_id"],
                    ),
                )
                cur.execute(
                    f"""
                    UPDATE {self.schema}.run_group_members
                    SET left_at = COALESCE(left_at, %s)
                    WHERE group_id = %s
                    """,
                    (kwargs["closed_at"], kwargs["group_id"]),
                )
