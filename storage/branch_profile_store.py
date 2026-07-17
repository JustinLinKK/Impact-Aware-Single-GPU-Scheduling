"""Dedicated SQLite store for empirical branch/profile evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import sqlite3

from ..config import SchedulerSettings
from .models import BRANCH_PROFILE_SCHEMA_STATEMENTS, MIGRATION_STATEMENTS, PROFILE_TABLE_NAMES
from .sqlite_store import SQLiteStateStore


class BranchProfileStore(SQLiteStateStore):
    """Persist measured scheduler profiles in ``branch_profile.sqlite3``.

    The class reuses the profile CRUD methods from ``SQLiteStateStore`` but
    connects to a separate SQLite database and initializes only profile tables.
    """

    def __init__(self, settings: SchedulerSettings, *, read_only: bool = False):
        self.settings = settings
        self.read_only = bool(read_only)
        self._hardware_profile = None
        self.settings.ensure_runtime_layout()
        if not self.read_only:
            self.initialize()
            self.import_legacy_profiles_if_empty()

    @property
    def db_path(self) -> Path:
        path = getattr(self.settings, "branch_profile_db_path", None)
        if path is None:
            path = self.settings.db_dir / "branch_profile.sqlite3"
        return Path(path)

    def _connect(self) -> sqlite3.Connection:
        path = self.db_path
        if self.read_only:
            connection = sqlite3.connect(
                f"file:{path}?mode=ro",
                timeout=self.settings.sqlite_busy_timeout_ms / 1000.0,
                uri=True,
            )
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(path, timeout=self.settings.sqlite_busy_timeout_ms / 1000.0)
            connection.execute("PRAGMA journal_mode=WAL")
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            for statement in BRANCH_PROFILE_SCHEMA_STATEMENTS:
                connection.execute(statement)
            for statement in MIGRATION_STATEMENTS:
                try:
                    connection.execute(statement)
                except sqlite3.OperationalError as exc:
                    if "duplicate column name" not in str(exc).lower():
                        raise
            connection.commit()

    def _profile_row_count(self, connection: sqlite3.Connection, *, schema: str | None = None) -> int:
        total = 0
        prefix = f"{schema}." if schema else ""
        for table in PROFILE_TABLE_NAMES:
            try:
                row = connection.execute(f"SELECT COUNT(*) AS count FROM {prefix}{table}").fetchone()
            except sqlite3.OperationalError:
                continue
            total += int(row["count"] if row is not None else 0)
        return total

    @staticmethod
    def _table_columns(connection: sqlite3.Connection, table: str, *, schema: str | None = None) -> list[str]:
        pragma = f"PRAGMA {schema}.table_info({table})" if schema else f"PRAGMA table_info({table})"
        rows = connection.execute(pragma).fetchall()
        return [str(row["name"]) for row in rows]

    @staticmethod
    def _table_exists(connection: sqlite3.Connection, table: str, *, schema: str | None = None) -> bool:
        if schema:
            query = f"SELECT 1 FROM {schema}.sqlite_master WHERE type='table' AND name=?"
        else:
            query = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?"
        return connection.execute(query, (table,)).fetchone() is not None

    @staticmethod
    def _quote_identifier(value: str) -> str:
        return '"' + str(value).replace('"', '""') + '"'

    def import_legacy_profiles_if_empty(self) -> dict[str, Any]:
        """Import legacy profile rows from ``scheduler.sqlite3`` once.

        Import is skipped unless the branch profile database has zero rows in
        all profile tables. Existing scheduler databases are left untouched.
        """

        legacy_path = Path(getattr(self.settings, "db_path", ""))
        branch_path = self.db_path
        if not legacy_path.exists() or legacy_path.resolve() == branch_path.resolve():
            return {"ok": True, "imported": {}, "skipped": True}

        imported: dict[str, int] = {}
        with self._connect() as connection:
            if self._profile_row_count(connection) > 0:
                return {"ok": True, "imported": {}, "skipped": True}
            connection.execute("ATTACH DATABASE ? AS legacy", (str(legacy_path),))
            try:
                for table in PROFILE_TABLE_NAMES:
                    if not self._table_exists(connection, table, schema="legacy"):
                        continue
                    main_columns = self._table_columns(connection, table)
                    legacy_columns = set(self._table_columns(connection, table, schema="legacy"))
                    columns = [column for column in main_columns if column in legacy_columns]
                    if not columns:
                        continue
                    quoted_columns = ", ".join(self._quote_identifier(column) for column in columns)
                    quoted_table = self._quote_identifier(table)
                    before = int(connection.execute(f"SELECT COUNT(*) AS count FROM {quoted_table}").fetchone()["count"])
                    connection.execute(
                        f"INSERT OR IGNORE INTO {quoted_table} ({quoted_columns}) "
                        f"SELECT {quoted_columns} FROM legacy.{quoted_table}"
                    )
                    after = int(connection.execute(f"SELECT COUNT(*) AS count FROM {quoted_table}").fetchone()["count"])
                    imported[table] = max(0, after - before)
                connection.commit()
            finally:
                connection.execute("DETACH DATABASE legacy")
        return {"ok": True, "imported": imported, "skipped": False}


class BranchProfileReader(BranchProfileStore):
    """Read-only profile evidence reader for hardware-knowledge clients."""

    def __init__(self, settings: SchedulerSettings):
        super().__init__(settings, read_only=True)
