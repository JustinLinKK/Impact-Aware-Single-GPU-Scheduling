"""Release-compatible VRAM postprocessing and relative comparison displays.

Time costs remain physical durations in the scheduler. Ratios here describe a
named comparison set; measured labels are required to evaluate ordering accuracy.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

MIB_BYTES = 1_048_576
VRAM_BUCKET_BYTES = 100_000_000
VRAM_BUCKET_MIB = VRAM_BUCKET_BYTES / MIB_BYTES


def ceil_vram_mib(value: float) -> float:
    """Ceil to decimal 100 MB, minimum 100 MB, retaining MiB transport units."""
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("predicted VRAM must be finite and nonnegative")
    return max(1, math.ceil(value / VRAM_BUCKET_MIB)) * VRAM_BUCKET_MIB


def deployment_prediction(raw: list[float]) -> list[float]:
    """Preserve raw time/busy predictions; round both memory columns upward."""
    if len(raw) != 6:
        raise ValueError("expected six prediction columns")
    return [*raw[:4], ceil_vram_mib(raw[4]), ceil_vram_mib(raw[5])]


def relative_time_costs(durations: Mapping[str, float]) -> dict[str, float]:
    """Return each predicted duration / fastest in this explicit comparison set.

    This is a display ratio, not an accuracy score or a scheduler priority.
    Durations must share a unit and semantic meaning (step, epoch, or full job).
    """
    if not durations:
        raise ValueError("relative time comparison requires at least one duration")
    values = {name: float(value) for name, value in durations.items()}
    if any(not math.isfinite(value) or value <= 0 for value in values.values()):
        raise ValueError("predicted durations must be finite and positive")
    fastest = min(values.values())
    return {name: value / fastest for name, value in values.items()}
