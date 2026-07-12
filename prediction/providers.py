"""Provider protocols and typed prediction adapter errors."""

from __future__ import annotations

from typing import Protocol

from ..domain import PredictionRequest, ResourcePrediction


class PredictionProviderError(RuntimeError):
    """Raised by a provider when a request cannot be predicted safely."""


class ResourcePredictionProvider(Protocol):
    @property
    def name(self) -> str:
        ...

    def available(self, hardware_key: str) -> bool:
        ...

    def predict(self, request: PredictionRequest) -> ResourcePrediction | None:
        ...
