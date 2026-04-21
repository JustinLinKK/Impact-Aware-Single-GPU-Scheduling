"""SQLite-backed persistent state store."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json
import sqlite3

from ..schemas import CommandType, JobCommand, JobStatus, SchedulerReport, TrainingJob, parse_timestamp, utc_now
from ..settings import SchedulerSettings
from .models import SCHEMA_STATEMENTS


class SQLiteStateStore:
    """Persist jobs, commands, events, checkpoints, and cache metadata."""

    def __init__(self, settings: SchedulerSettings):
        self.settings = settings
        self.settings.ensure_runtime_layout()
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.settings.db_path, timeout=self.settings.sqlite_busy_timeout_ms / 1000.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            for statement in SCHEMA_STATEMENTS:
                connection.execute(statement)
            connection.commit()

    def next_queue_sequence(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COALESCE(MAX(queue_sequence), 0) AS value FROM jobs").fetchone()
            return int(row["value"]) + 1

    def save_job(self, job: TrainingJob) -> None:
        payload_json = job.to_json()
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs(job_id, status, priority, baseline_model_id, submitted_at, queue_sequence, payload_json, updated_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status=excluded.status,
                    priority=excluded.priority,
                    baseline_model_id=excluded.baseline_model_id,
                    submitted_at=excluded.submitted_at,
                    queue_sequence=excluded.queue_sequence,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    job.job_id,
                    job.status.value,
                    job.priority,
                    job.baseline_model_id,
                    job.submitted_at,
                    job.queue_sequence,
                    payload_json,
                    now,
                ),
            )
            connection.commit()

    def submit_job(self, job: TrainingJob) -> TrainingJob:
        if not job.queue_sequence:
            job.queue_sequence = self.next_queue_sequence()
        job.mark_status(JobStatus.PENDING, reason=job.status_reason)
        self.save_job(job)
        self.enqueue_command(CommandType.SUBMIT, job_id=job.job_id)
        self.log_event("job_submitted", job_id=job.job_id, payload={"priority": job.priority})
        return job

    def get_job(self, job_id: str) -> TrainingJob | None:
        with self._connect() as connection:
            row = connection.execute("SELECT payload_json FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if not row:
            return None
        payload = json.loads(row["payload_json"])
        return TrainingJob.from_dict(payload)

    def list_jobs(self, statuses: Iterable[JobStatus | str] | None = None) -> list[TrainingJob]:
        query = "SELECT payload_json FROM jobs"
        params: list[Any] = []
        if statuses:
            normalized = [status.value if isinstance(status, JobStatus) else status for status in statuses]
            placeholders = ",".join("?" for _ in normalized)
            query += f" WHERE status IN ({placeholders})"
            params.extend(normalized)
        query += " ORDER BY priority DESC, queue_sequence ASC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [TrainingJob.from_dict(json.loads(row["payload_json"])) for row in rows]

    def runnable_jobs(self) -> list[TrainingJob]:
        jobs = self.list_jobs(
            statuses=[
                JobStatus.PENDING,
                JobStatus.READY,
                JobStatus.PAUSED,
                JobStatus.RECOVERABLE,
            ]
        )
        return [job for job in jobs if job.is_runnable()]

    def update_job(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        reason: str | None = None,
        hold: bool | None = None,
        latest_checkpoint_path: str | None = None,
        last_heartbeat_at: str | None = None,
        last_dispatched_at: str | None = None,
        status_timestamps: dict[str, str] | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> TrainingJob:
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(f"Unknown job_id: {job_id}")
        if status is not None:
            job.mark_status(status, reason=reason)
        elif reason is not None:
            job.status_reason = reason
        if hold is not None:
            job.hold = hold
        if latest_checkpoint_path is not None:
            job.latest_checkpoint_path = latest_checkpoint_path
        if last_heartbeat_at is not None:
            job.last_heartbeat_at = last_heartbeat_at
        if last_dispatched_at is not None:
            job.last_dispatched_at = last_dispatched_at
        if status_timestamps:
            job.status_timestamps.update(status_timestamps)
        if metadata_updates:
            job.metadata.update(metadata_updates)
        self.save_job(job)
        return job

    def set_job_status(self, job_id: str, status: JobStatus, *, reason: str | None = None, hold: bool | None = None) -> TrainingJob:
        return self.update_job(job_id, status=status, reason=reason, hold=hold)

    def delete_job(self, job_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            connection.commit()

    def enqueue_command(self, command_type: CommandType, *, job_id: str | None = None, payload: dict[str, Any] | None = None) -> int:
        payload_json = json.dumps(payload or {}, sort_keys=True)
        created_at = utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO commands(job_id, command_type, payload_json, created_at, processed_at)
                VALUES(?, ?, ?, ?, NULL)
                """,
                (job_id, command_type.value, payload_json, created_at),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def fetch_pending_commands(self, limit: int = 100) -> list[JobCommand]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT command_id, job_id, command_type, payload_json, created_at, processed_at
                FROM commands
                WHERE processed_at IS NULL
                ORDER BY command_id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [JobCommand.from_row(dict(row)) for row in rows]

    def mark_command_processed(self, command_id: int) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE commands SET processed_at = ? WHERE command_id = ?", (utc_now(), command_id))
            connection.commit()

    def log_event(self, event_type: str, *, job_id: str | None = None, payload: dict[str, Any] | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO events(job_id, event_type, payload_json, created_at)
                VALUES(?, ?, ?, ?)
                """,
                (job_id, event_type, json.dumps(payload or {}, sort_keys=True), utc_now()),
            )
            connection.commit()

    def list_events(self, *, job_id: str | None = None, event_type: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT event_id, job_id, event_type, payload_json, created_at FROM events WHERE 1=1"
        params: list[Any] = []
        if job_id is not None:
            query += " AND job_id = ?"
            params.append(job_id)
        if event_type is not None:
            query += " AND event_type = ?"
            params.append(event_type)
        query += " ORDER BY event_id ASC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "job_id": row["job_id"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload_json"]) if row["payload_json"] else {},
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def record_checkpoint(self, job_id: str, checkpoint_path: str, metadata: dict[str, Any] | None = None) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE checkpoints SET is_latest = 0 WHERE job_id = ?", (job_id,))
            connection.execute(
                """
                INSERT INTO checkpoints(job_id, checkpoint_path, created_at, metadata_json, is_latest)
                VALUES(?, ?, ?, ?, 1)
                """,
                (job_id, checkpoint_path, utc_now(), json.dumps(metadata or {}, sort_keys=True)),
            )
            connection.commit()
        self.update_job(job_id, latest_checkpoint_path=checkpoint_path)

    def latest_checkpoint(self, job_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT checkpoint_path
                FROM checkpoints
                WHERE job_id = ? AND is_latest = 1
                ORDER BY checkpoint_id DESC
                LIMIT 1
                """,
                (job_id,),
            ).fetchone()
        if row:
            return str(row["checkpoint_path"])
        job = self.get_job(job_id)
        return job.latest_checkpoint_path if job else None

    def update_cache_metadata(
        self,
        model_id: str,
        baseline_model_path: str,
        *,
        size_bytes: int,
        pinned: bool,
        hits: int,
        misses: int,
        last_loaded_at: str | None,
        last_accessed_at: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO cache_entries(model_id, baseline_model_path, size_bytes, pinned, hits, misses, last_loaded_at, last_accessed_at, metadata_json)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_id) DO UPDATE SET
                    baseline_model_path=excluded.baseline_model_path,
                    size_bytes=excluded.size_bytes,
                    pinned=excluded.pinned,
                    hits=excluded.hits,
                    misses=excluded.misses,
                    last_loaded_at=excluded.last_loaded_at,
                    last_accessed_at=excluded.last_accessed_at,
                    metadata_json=excluded.metadata_json
                """,
                (
                    model_id,
                    baseline_model_path,
                    size_bytes,
                    1 if pinned else 0,
                    hits,
                    misses,
                    last_loaded_at,
                    last_accessed_at,
                    json.dumps(metadata or {}, sort_keys=True),
                ),
            )
            connection.commit()

    def cache_metadata_summary(self) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS entries,
                    COALESCE(SUM(size_bytes), 0) AS used_bytes,
                    COALESCE(SUM(CASE WHEN pinned = 1 THEN 1 ELSE 0 END), 0) AS pinned_entries,
                    COALESCE(SUM(hits), 0) AS hits,
                    COALESCE(SUM(misses), 0) AS misses
                FROM cache_entries
                """
            ).fetchone()
        return dict(row)

    def reconcile_incomplete_jobs(self) -> list[TrainingJob]:
        stale_jobs = self.list_jobs(statuses=[JobStatus.RUNNING, JobStatus.PAUSING])
        reconciled: list[TrainingJob] = []
        for job in stale_jobs:
            checkpoint_path = self.latest_checkpoint(job.job_id)
            if checkpoint_path and Path(checkpoint_path).exists():
                updated = self.update_job(
                    job.job_id,
                    status=JobStatus.RECOVERABLE,
                    reason="scheduler restarted while job was active; checkpoint available",
                    latest_checkpoint_path=checkpoint_path,
                )
            else:
                updated = self.update_job(
                    job.job_id,
                    status=JobStatus.FAILED,
                    reason="scheduler restarted while job was active; no checkpoint found",
                )
            reconciled.append(updated)
        return reconciled

    def report(self) -> SchedulerReport:
        jobs = self.list_jobs()
        wait_times: list[float] = []
        runtimes: list[float] = []
        for job in jobs:
            submitted = parse_timestamp(job.submitted_at)
            started = parse_timestamp(job.started_at)
            finished = parse_timestamp(job.finished_at)
            if submitted and started:
                wait_times.append((started - submitted).total_seconds())
            if started and finished:
                runtimes.append((finished - started).total_seconds())
        cache_summary = self.cache_metadata_summary()
        total_cache = int(cache_summary["hits"]) + int(cache_summary["misses"])
        return SchedulerReport(
            total_jobs=len(jobs),
            completed_jobs=sum(job.status == JobStatus.COMPLETED for job in jobs),
            failed_jobs=sum(job.status == JobStatus.FAILED for job in jobs),
            cancelled_jobs=sum(job.status == JobStatus.CANCELLED for job in jobs),
            average_queue_wait_seconds=sum(wait_times) / len(wait_times) if wait_times else 0.0,
            average_runtime_seconds=sum(runtimes) / len(runtimes) if runtimes else 0.0,
            cache_hit_rate=(int(cache_summary["hits"]) / total_cache) if total_cache else 0.0,
            cache_hits=int(cache_summary["hits"]),
            cache_misses=int(cache_summary["misses"]),
            cache_evictions=sum(event["event_type"] == "cache_evicted" for event in self.list_events(event_type=None)),
        )
