"""Resolve CUDA logical device indices to physical NVIDIA selectors."""

from __future__ import annotations

import os


def physical_cuda_device_selector(
    device_index: int,
    *,
    visible_devices: str | None = None,
) -> str:
    """Return the physical selector represented by a logical CUDA index.

    PyTorch indexes the devices exposed by ``CUDA_VISIBLE_DEVICES`` from zero,
    while ``nvidia-smi`` and a newly assigned ``CUDA_VISIBLE_DEVICES`` value use
    physical indices or UUIDs.  Mapping here keeps worker placement and NVML
    telemetry on the same accelerator as the parent process.
    """
    logical_index = int(device_index)
    raw = os.environ.get("CUDA_VISIBLE_DEVICES") if visible_devices is None else visible_devices
    if raw is None:
        return str(logical_index)
    normalized = str(raw).strip()
    if not normalized or normalized.lower() in {"-1", "none", "void", "all"}:
        return str(logical_index)
    selectors = [item.strip() for item in normalized.split(",") if item.strip()]
    if 0 <= logical_index < len(selectors):
        return selectors[logical_index]
    return str(logical_index)
