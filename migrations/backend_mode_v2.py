"""Dry-run-first migration for canonical process backend names.

Historical stream rows are retained and marked non-selectable. Legacy ``mps``
rows are normalized only when their metadata proves that the MPS runtime was
used; ambiguous rows stay untouched and are reported for operator review.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any
import json
import re
import sqlite3

from ..backend_mode import (
    RETIRED_BACKEND_MODES,
    RUNNER_CONTRACT_SUBPROCESS_V1,
    normalize_runtime_backend,
)
from ..code_knowledge.records import load_backend_guidance_seed_records
from ..domain.identity import (
    build_backend_scoped_pair_key,
    build_batch_size_observation_key,
    build_combination_key,
    build_runtime_profile_key,
    decode_batch_vector,
)


_PROFILE_TABLES = (
    "runtime_profiles",
    "pair_profiles",
    "batch_size_observations",
    "combination_profiles",
)
_ALL_LEGACY = ("mps", *sorted(RETIRED_BACKEND_MODES))


def _empty_counts() -> dict[str, int]:
    return {backend: 0 for backend in _ALL_LEGACY}


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    }


def _proves_mps_runtime(metadata: dict[str, Any]) -> bool:
    flattened = json.dumps(metadata, sort_keys=True).lower()
    explicit = str(
        metadata.get("canonical_backend")
        or metadata.get("runtime_backend")
        or metadata.get("backend_implementation")
        or ""
    ).lower()
    return explicit in {"mps_process", "mpsbackend", "mps_backend"} or any(
        marker in flattened
        for marker in (
            "cuda_mps_pipe_directory",
            "cuda_mps_active_thread_percentage",
            "nvidia-cuda-mps-control",
        )
    )


def _identifier_counts(text: str) -> dict[str, int]:
    lowered = text.lower()
    return {
        backend: len(
            re.findall(
                rf"(?<![a-z0-9_]){re.escape(backend)}(?![a-z0-9_])",
                lowered,
            )
        )
        for backend in _ALL_LEGACY
    }


def _config_counts(paths: Iterable[str | Path]) -> dict[str, int]:
    counts = _empty_counts()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            continue
        found = _identifier_counts(
            path.read_text(encoding="utf-8", errors="replace")
        )
        for backend, count in found.items():
            counts[backend] += count
    return counts


def _cache_counts(runtime_root: Path) -> dict[str, int]:
    counts = _empty_counts()
    for path in runtime_root.glob("cache_meta/**/*.json"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        found = _identifier_counts(text)
        for backend, count in found.items():
            counts[backend] += int(count > 0)
    return counts


def _knowledge_counts() -> dict[str, int]:
    counts = _empty_counts()
    for record in load_backend_guidance_seed_records():
        modes = set(record.get("backend_modes") or [])
        for backend in counts:
            counts[backend] += int(backend in modes)
    return counts


def _canonical_identity(
    table: str, row: sqlite3.Row, backend_name: str
) -> tuple[str, str]:
    backend_name = normalize_runtime_backend(backend_name, warn_legacy=False)
    if table == "runtime_profiles":
        return (
            "profile_key",
            build_runtime_profile_key(
                row["signature"],
                row["hardware_key"],
                backend_name,
                row["resolved_batch_size"],
                row["strategy"],
            ),
        )
    if table == "pair_profiles":
        return (
            "pair_key",
            build_backend_scoped_pair_key(
                row["left_signature"],
                row["right_signature"],
                backend_name=backend_name,
            ),
        )
    if table == "batch_size_observations":
        return (
            "observation_key",
            build_batch_size_observation_key(
                row["model_key"],
                row["shape_signature"],
                row["hardware_key"],
                backend_name,
                row["batch_size"],
            ),
        )
    if table == "combination_profiles":
        return (
            "combination_key",
            build_combination_key(
                row["group_signature"],
                row["hardware_key"],
                backend_name,
                row["scheduler_mode"],
                decode_batch_vector(row["batch_vector_json"]),
            ),
        )
    raise ValueError(f"Unsupported profile table: {table}")


def _identity_conflict(
    connection: sqlite3.Connection,
    table: str,
    row: sqlite3.Row,
    key_column: str,
    canonical_key: str,
) -> bool:
    query = f'SELECT rowid FROM "{table}" WHERE "{key_column}"=? AND rowid<>?'
    values: list[Any] = [canonical_key, row["rowid"]]
    if table == "pair_profiles":
        query += " AND hardware_key=?"
        values.append(row["hardware_key"])
    return connection.execute(query, values).fetchone() is not None


def migrate_backend_mode_v2(
    settings: Any,
    *,
    dry_run: bool = True,
    config_paths: Iterable[str | Path] = (),
    connection: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Report or apply the non-destructive canonical backend migration."""

    owns_connection = connection is None
    if connection is None:
        connection = sqlite3.connect(str(settings.db_path))
    connection.row_factory = sqlite3.Row
    profile_counts = _empty_counts()
    profile_counts_by_table: dict[str, dict[str, int]] = {}
    provable_mps = 0
    ambiguous_mps = 0
    conflicting_mps = 0
    schema_v2_rekeys = 0
    schema_v2_conflicts = 0
    retired_rows = 0
    retired_rows_to_change = 0
    event_counts = _empty_counts()
    try:
        for table in _PROFILE_TABLES:
            if not _table_exists(connection, table):
                continue
            columns = _columns(connection, table)
            if "backend_name" not in columns:
                continue
            metadata_column = "metadata_json" if "metadata_json" in columns else None
            rows = connection.execute(f'SELECT rowid, * FROM "{table}"').fetchall()
            table_counts = _empty_counts()
            profile_counts_by_table[table] = table_counts
            for row in rows:
                backend = str(row["backend_name"])
                if backend in profile_counts:
                    profile_counts[backend] += 1
                    table_counts[backend] += 1
                metadata = {}
                if metadata_column and row["metadata_json"]:
                    try:
                        metadata = json.loads(row["metadata_json"])
                    except (TypeError, ValueError):
                        metadata = {}
                if backend == "mps":
                    if _proves_mps_runtime(metadata):
                        key_column, canonical_key = _canonical_identity(
                            table, row, "mps_process"
                        )
                        if _identity_conflict(
                            connection,
                            table,
                            row,
                            key_column,
                            canonical_key,
                        ):
                            conflicting_mps += 1
                            continue
                        provable_mps += 1
                        if not dry_run:
                            metadata.update(
                                {
                                    "original_backend_identifier": "mps",
                                    "backend_migration": "backend_mode_v2",
                                    "backend_identity_schema_version": 2,
                                    "runner_contract": RUNNER_CONTRACT_SUBPROCESS_V1,
                                }
                            )
                            assignments = f'backend_name=?, "{key_column}"=?'
                            values: list[Any] = ["mps_process", canonical_key]
                            if metadata_column:
                                assignments += ", metadata_json=?"
                                values.append(json.dumps(metadata, sort_keys=True))
                            values.append(row["rowid"])
                            connection.execute(
                                f'UPDATE "{table}" SET {assignments} WHERE rowid=?',
                                values,
                            )
                    else:
                        ambiguous_mps += 1
                    continue
                if backend in RETIRED_BACKEND_MODES:
                    retired_rows += 1
                    already_retired = bool(
                        metadata.get("retired_backend") is True
                        and metadata.get("selectable") is False
                    )
                    if not already_retired:
                        retired_rows_to_change += 1
                    if not dry_run and metadata_column and not already_retired:
                        metadata.update(
                            {
                                "original_backend_identifier": backend,
                                "backend_migration": "backend_mode_v2",
                                "retired_backend": True,
                                "selectable": False,
                            }
                        )
                        connection.execute(
                            f'UPDATE "{table}" SET metadata_json=? WHERE rowid=?',
                            (json.dumps(metadata, sort_keys=True), row["rowid"]),
                        )
                    continue
                try:
                    key_column, canonical_key = _canonical_identity(
                        table, row, backend
                    )
                except ValueError:
                    continue
                if str(row[key_column]) == canonical_key:
                    continue
                if _identity_conflict(
                    connection,
                    table,
                    row,
                    key_column,
                    canonical_key,
                ):
                    schema_v2_conflicts += 1
                    continue
                schema_v2_rekeys += 1
                if not dry_run:
                    assignments = f'"{key_column}"=?'
                    values = [canonical_key]
                    if metadata_column:
                        metadata.update(
                            {
                                "backend_identity_schema_version": 2,
                                "runner_contract": RUNNER_CONTRACT_SUBPROCESS_V1,
                            }
                        )
                        assignments += ", metadata_json=?"
                        values.append(json.dumps(metadata, sort_keys=True))
                    values.append(row["rowid"])
                    connection.execute(
                        f'UPDATE "{table}" SET {assignments} WHERE rowid=?',
                        values,
                    )

        if _table_exists(connection, "events"):
            columns = _columns(connection, "events")
            payload_column = next(
                (name for name in ("payload_json", "payload") if name in columns),
                None,
            )
            if payload_column:
                for row in connection.execute(
                    f'SELECT "{payload_column}" FROM events'
                ).fetchall():
                    found = _identifier_counts(str(row[0] or ""))
                    for backend, count in found.items():
                        event_counts[backend] += int(count > 0)
        if not dry_run:
            connection.commit()
    finally:
        if owns_connection:
            connection.close()

    report = {
        "ok": True,
        "dry_run": bool(dry_run),
        "migration": "backend_mode_v2",
        "config_references": _config_counts(config_paths),
        "cache_entries": _cache_counts(Path(settings.runtime_root)),
        "profiles": profile_counts,
        "profiles_by_table": profile_counts_by_table,
        "events": event_counts,
        "knowledge_records": _knowledge_counts(),
        "provable_mps_rows": provable_mps,
        "ambiguous_mps_rows": ambiguous_mps,
        "conflicting_mps_rows": conflicting_mps,
        "schema_v2_rekey_rows": schema_v2_rekeys,
        "schema_v2_conflict_rows": schema_v2_conflicts,
        "retired_profile_rows": retired_rows,
        "normalized_backend": "mps_process",
        "retired_backends": sorted(RETIRED_BACKEND_MODES),
    }
    report["would_change_rows" if dry_run else "changed_rows"] = (
        provable_mps + retired_rows_to_change + schema_v2_rekeys
    )
    return report
