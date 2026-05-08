"""Protocols consumed by the placement planner."""

from __future__ import annotations

from typing import Protocol

from ..hardware import HardwareProfile
from ..domain import BatchProbeProfile, BatchSizeObservation, CombinationProfile, PairProfile, SoloProfile


class PlanningRepository(Protocol):
    def hardware_profile(self) -> HardwareProfile:
        ...

    def hardware_key(self) -> str:
        ...

    def get_solo_profile(self, signature: str, *, hardware_key: str | None = None) -> SoloProfile | None:
        ...

    def get_pair_profile(
        self,
        left_signature: str,
        right_signature: str,
        *,
        hardware_key: str | None = None,
        backend_name: str | None = None,
    ) -> PairProfile | None:
        ...

    def get_batch_probe_profile(self, probe_key: str) -> BatchProbeProfile | None:
        ...

    def get_batch_size_observation(
        self,
        *,
        model_key: str,
        shape_signature: str,
        hardware_key: str,
        backend_name: str,
        batch_size: int,
    ) -> BatchSizeObservation | None:
        ...

    def list_batch_size_observations(
        self,
        *,
        model_key: str | None = None,
        shape_signature: str | None = None,
        hardware_key: str | None = None,
        backend_name: str | None = None,
    ) -> list[BatchSizeObservation]:
        ...

    def best_combination_profile(
        self,
        *,
        group_signature: str,
        hardware_key: str,
        backend_name: str,
        scheduler_mode: str,
    ) -> CombinationProfile | None:
        ...

