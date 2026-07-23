"""SQLite-backed repository bundle facade."""

from __future__ import annotations

from dataclasses import dataclass

from .state_store import StateStore


@dataclass(slots=True)
class SQLiteRepositoryBundle:
    store: StateStore

    @property
    def jobs(self) -> StateStore:
        return self.store

    @property
    def commands(self) -> StateStore:
        return self.store

    @property
    def checkpoints(self) -> StateStore:
        return self.store

    @property
    def events(self) -> StateStore:
        return self.store

    @property
    def cache(self) -> StateStore:
        return self.store

    @property
    def profiles(self) -> StateStore:
        return self.store

    @property
    def reporting(self) -> StateStore:
        return self.store
