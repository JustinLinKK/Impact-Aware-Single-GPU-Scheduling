"""Lightweight GPU telemetry sampling via ``nvidia-smi``."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
import os
import shutil
import subprocess

from ..cuda_device_mapping import physical_cuda_device_selector
from ..domain import utc_now
from ..domain import parse_timestamp


def _resolve_nvidia_smi() -> str | None:
    configured = os.environ.get("MLEVOLVE_NVIDIA_SMI")
    if configured and Path(configured).is_file():
        return configured
    discovered = shutil.which("nvidia-smi")
    if discovered:
        return discovered
    wsl_binary = Path("/usr/lib/wsl/lib/nvidia-smi")
    return str(wsl_binary) if wsl_binary.is_file() else None


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
    avg_vram_mb: float | None = None
    avg_gpu_utilization: float | None = None
    avg_memory_utilization: float | None = None
    sample_count: int = 0

    @classmethod
    def from_samples(cls, samples: Sequence[GpuTelemetrySample]) -> "GpuTelemetrySummary":
        if not samples:
            return cls()
        return cls(
            peak_vram_mb=max(sample.memory_used_mb for sample in samples),
            avg_vram_mb=sum(sample.memory_used_mb for sample in samples) / len(samples),
            avg_gpu_utilization=sum(sample.gpu_utilization for sample in samples) / len(samples),
            avg_memory_utilization=sum(sample.memory_utilization for sample in samples) / len(samples),
            sample_count=len(samples),
        )


@dataclass(slots=True)
class MemoryAdmissionGate:
    """Rolling-average admission hysteresis; it never pauses active work."""

    stop_fraction: float = 0.90
    resume_fraction: float = 0.85
    window_seconds: float = 10.0
    is_open: bool = True
    samples: list[GpuTelemetrySample] = field(default_factory=list)
    below_resume_since: datetime | None = None
    average_fraction: float | None = None

    def update(self, sample: GpuTelemetrySample) -> str | None:
        captured = parse_timestamp(sample.captured_at) or datetime.now(timezone.utc)
        cutoff = captured.timestamp() - self.window_seconds
        self.samples.append(sample)
        self.samples = [item for item in self.samples if (parse_timestamp(item.captured_at) or captured).timestamp() >= cutoff]
        fractions = [item.memory_used_mb / item.memory_total_mb for item in self.samples if item.memory_total_mb > 0]
        if not fractions:
            self.average_fraction = None
            return None
        self.average_fraction = sum(fractions) / len(fractions)
        oldest = min(
            (parse_timestamp(item.captured_at) or captured for item in self.samples),
            default=captured,
        )
        complete_window = (captured - oldest).total_seconds() >= self.window_seconds
        if self.is_open:
            if complete_window and self.average_fraction >= self.stop_fraction:
                self.is_open = False
                self.below_resume_since = None
                return "closed"
            return None
        if self.average_fraction <= self.resume_fraction:
            self.below_resume_since = self.below_resume_since or captured
            if (captured - self.below_resume_since).total_seconds() >= self.window_seconds:
                self.is_open = True
                self.below_resume_since = None
                return "opened"
        else:
            self.below_resume_since = None
        return None


class NvidiaSmiTelemetrySampler:
    """Best-effort device polling for local single-GPU scheduling."""

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.device_selector = physical_cuda_device_selector(device_index)
        self._binary = _resolve_nvidia_smi()

    def available(self) -> bool:
        return self._binary is not None

    def sample(self) -> GpuTelemetrySample | None:
        if not self._binary:
            return None
        try:
            result = subprocess.run(
                [
                    self._binary,
                    f"--id={self.device_selector}",
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
