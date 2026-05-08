"""SQLite-backed repository bundle facade."""

from __future__ import annotations

from dataclasses import dataclass

from .sqlite_store import SQLiteStateStore


@dataclass(slots=True)
class SQLiteRepositoryBundle:
    store: SQLiteStateStore

    @property
    def jobs(self) -> SQLiteStateStore:
        return self.store

    @property
    def commands(self) -> SQLiteStateStore:
        return self.store

    @property
    def checkpoints(self) -> SQLiteStateStore:
        return self.store

    @property
    def events(self) -> SQLiteStateStore:
        return self.store

    @property
    def cache(self) -> SQLiteStateStore:
        return self.store

    @property
    def profiles(self) -> SQLiteStateStore:
        return self.store

    @property
    def reporting(self) -> SQLiteStateStore:
        return self.store
