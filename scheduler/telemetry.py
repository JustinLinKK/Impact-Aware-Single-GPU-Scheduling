"""Lightweight GPU telemetry sampling via ``nvidia-smi``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import shutil
import subprocess

from ..schemas import utc_now


@dataclass(slots=True)
class GpuTelemetrySample:
    captured_at: str = field(default_factory=utc_now)
    memory_used_mb: int = 0
    memory_total_mb: int = 0
    gpu_utilization: float = 0.0
    memory_utilization: float = 0.0


@dataclass(slots=True)
class GpuTelemetrySummary:
    peak_vram_mb: int | None = None
    avg_gpu_utilization: float | None = None
    avg_memory_utilization: float | None = None
    sample_count: int = 0

    @classmethod
    def from_samples(cls, samples: Sequence[GpuTelemetrySample]) -> "GpuTelemetrySummary":
        if not samples:
            return cls()
        return cls(
            peak_vram_mb=max(sample.memory_used_mb for sample in samples),
            avg_gpu_utilization=sum(sample.gpu_utilization for sample in samples) / len(samples),
            avg_memory_utilization=sum(sample.memory_utilization for sample in samples) / len(samples),
            sample_count=len(samples),
        )


class NvidiaSmiTelemetrySampler:
    """Best-effort device polling for local single-GPU scheduling."""

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._binary = shutil.which("nvidia-smi")

    def available(self) -> bool:
        return self._binary is not None

    def sample(self) -> GpuTelemetrySample | None:
        if not self._binary:
            return None
        try:
            result = subprocess.run(
                [
                    self._binary,
                    f"--id={self.device_index}",
                    "--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except Exception:
            return None
        if result.returncode != 0 or not result.stdout.strip():
            return None
        try:
            raw_values = [value.strip() for value in result.stdout.strip().split(",")]
            memory_used_mb, memory_total_mb, gpu_utilization, memory_utilization = raw_values[:4]
            return GpuTelemetrySample(
                memory_used_mb=int(float(memory_used_mb)),
                memory_total_mb=int(float(memory_total_mb)),
                gpu_utilization=float(gpu_utilization) / 100.0,
                memory_utilization=float(memory_utilization) / 100.0,
            )
        except (TypeError, ValueError, IndexError):
            return None
