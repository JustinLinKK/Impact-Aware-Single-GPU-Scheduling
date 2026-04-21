"""Storage-layer constants and lightweight row helpers."""

from __future__ import annotations

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        priority INTEGER NOT NULL,
        baseline_model_id TEXT NOT NULL,
        submitted_at TEXT NOT NULL,
        queue_sequence INTEGER NOT NULL,
        payload_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS commands (
        command_id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT,
        command_type TEXT NOT NULL,
        payload_json TEXT,
        created_at TEXT NOT NULL,
        processed_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT,
        event_type TEXT NOT NULL,
        payload_json TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS checkpoints (
        checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT NOT NULL,
        checkpoint_path TEXT NOT NULL,
        created_at TEXT NOT NULL,
        metadata_json TEXT,
        is_latest INTEGER NOT NULL DEFAULT 1
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cache_entries (
        model_id TEXT PRIMARY KEY,
        baseline_model_path TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        pinned INTEGER NOT NULL DEFAULT 0,
        hits INTEGER NOT NULL DEFAULT 0,
        misses INTEGER NOT NULL DEFAULT 0,
        last_loaded_at TEXT,
        last_accessed_at TEXT,
        metadata_json TEXT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_jobs_status_priority
    ON jobs(status, priority DESC, queue_sequence ASC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_commands_processed_created
    ON commands(processed_at, created_at)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_events_created_at
    ON events(created_at)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_checkpoints_job_created
    ON checkpoints(job_id, created_at DESC)
    """,
]
