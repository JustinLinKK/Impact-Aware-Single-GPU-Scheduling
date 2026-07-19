"""Runtime state-store facade."""

from __future__ import annotations

from typing import Any

from ..config import SchedulerSettings
from .branch_profile_store import BranchProfileStore
from .sqlite_store import SQLiteStateStore


_PROFILE_METHODS = {
    "upsert_solo_profile",
    "get_solo_profile",
    "list_solo_profiles",
    "upsert_pair_profile",
    "get_pair_profile",
    "list_pair_profiles",
    "mark_pair_incompatible",
    "upsert_runtime_profile",
    "get_runtime_profile",
    "list_runtime_profiles",
    "upsert_batch_probe_profile",
    "get_batch_probe_profile",
    "get_compatible_batch_probe_profile",
    "list_batch_probe_profiles",
    "upsert_batch_size_observation",
    "get_batch_size_observation",
    "list_batch_size_observations",
    "upsert_combination_profile",
    "best_combination_profile",
    "list_combination_profiles",
}


class StateStore:
    """Route scheduler control state and branch profile state to separate DBs."""

    def __init__(self, settings: SchedulerSettings):
        self.settings = settings
        self._backend = SQLiteStateStore(settings)
        self._profile_store = BranchProfileStore(settings)

    @property
    def backend(self) -> SQLiteStateStore:
        return self._backend

    @property
    def branch_profile_store(self) -> BranchProfileStore:
        return self._profile_store

    @property
    def _hardware_profile(self) -> Any:
        return self._backend._hardware_profile

    @_hardware_profile.setter
    def _hardware_profile(self, value: Any) -> None:
        self._backend._hardware_profile = value

    def hardware_profile(self) -> Any:
        return self._backend.hardware_profile()

    def hardware_key(self) -> str:
        return self._backend.hardware_key()

    def __getattr__(self, name: str) -> Any:
        if name in _PROFILE_METHODS:
            return getattr(self._profile_store, name)
        return getattr(self._backend, name)
