"""PerfSeer ML adapter socket.

The optimized PerfSeer runtime is not vendored in this repository. This socket is
intentionally fail-closed and can be replaced by a real adapter or a deterministic
test double without changing placement code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..domain import PredictionRequest, ResourcePrediction


@dataclass(frozen=True, slots=True)
class PerfSeerAdapterHealth:
    enabled: bool
    healthy: bool
    reason: str | None = None


class PerfSeerMLAdapter:
    name = "ml_student"
    version = "perfseer_socket_v1"

    def __init__(
        self,
        *,
        enabled: bool = False,
        hardware_key: str | None = None,
        checkpoint_path: str | None = None,
        calibration_path: str | None = None,
        device: str = "cpu",
        cache_size: int = 1024,
    ):
        self.enabled = bool(enabled)
        self.hardware_key = hardware_key
        self.checkpoint_path = checkpoint_path
        self.calibration_path = calibration_path
        self.device = str(device or "cpu")
        self.cache_size = max(0, int(cache_size or 0))

    def health(self) -> PerfSeerAdapterHealth:
        if not self.enabled:
            return PerfSeerAdapterHealth(enabled=False, healthy=False, reason="disabled")
        if not self.checkpoint_path:
            return PerfSeerAdapterHealth(enabled=True, healthy=False, reason="missing_checkpoint")
        return PerfSeerAdapterHealth(enabled=True, healthy=False, reason="perfseer_runtime_unavailable")

    def available(self, hardware_key: str) -> bool:
        health = self.health()
        if not health.healthy:
            return False
        return self.hardware_key in (None, "", hardware_key)

    def predict(self, request: PredictionRequest) -> ResourcePrediction | None:
        del request
        return None

    def to_dict(self) -> dict[str, Any]:
        health = self.health()
        return {
            "name": self.name,
            "version": self.version,
            "enabled": self.enabled,
            "healthy": health.healthy,
            "reason": health.reason,
            "hardware_key": self.hardware_key,
            "checkpoint_path": self.checkpoint_path,
            "calibration_path": self.calibration_path,
            "device": self.device,
            "cache_size": self.cache_size,
        }
