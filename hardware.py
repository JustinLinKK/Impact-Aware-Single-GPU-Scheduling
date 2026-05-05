"""Hardware profile helpers for profile-cache reuse."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import platform
from typing import Any
import json

import torch


@dataclass(slots=True)
class HardwareProfile:
    hardware_key: str
    os_name: str
    gpu_name: str
    total_vram_mb: int | None
    compute_capability: str | None
    cuda_runtime: str | None
    torch_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "hardware_key": self.hardware_key,
            "os_name": self.os_name,
            "gpu_name": self.gpu_name,
            "total_vram_mb": self.total_vram_mb,
            "compute_capability": self.compute_capability,
            "cuda_runtime": self.cuda_runtime,
            "torch_version": self.torch_version,
        }


def _hardware_key_payload(*, os_name: str, gpu_name: str, total_vram_mb: int | None, compute_capability: str | None, cuda_runtime: str | None, torch_version: str) -> dict[str, Any]:
    return {
        "os_name": os_name,
        "gpu_name": gpu_name,
        "total_vram_mb": total_vram_mb,
        "compute_capability": compute_capability,
        "cuda_runtime": cuda_runtime,
        "torch_version": torch_version,
    }


def build_hardware_key(*, os_name: str, gpu_name: str, total_vram_mb: int | None, compute_capability: str | None, cuda_runtime: str | None, torch_version: str) -> str:
    payload = _hardware_key_payload(
        os_name=os_name,
        gpu_name=gpu_name,
        total_vram_mb=total_vram_mb,
        compute_capability=compute_capability,
        cuda_runtime=cuda_runtime,
        torch_version=torch_version,
    )
    return sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def detect_hardware_profile(*, device_index: int = 0) -> HardwareProfile:
    os_name = platform.system().lower()
    gpu_name = "cuda-unavailable"
    total_vram_mb: int | None = None
    compute_capability: str | None = None
    if torch.cuda.is_available():
        try:
            gpu_name = str(torch.cuda.get_device_name(device_index))
        except Exception:
            gpu_name = "cuda-visible-device"
        try:
            props = torch.cuda.get_device_properties(device_index)
            total_vram_mb = int(props.total_memory / (1024 * 1024))
            capability = getattr(props, "major", None), getattr(props, "minor", None)
            if capability[0] is not None and capability[1] is not None:
                compute_capability = f"{capability[0]}.{capability[1]}"
        except Exception:
            total_vram_mb = None
            compute_capability = None
    cuda_runtime = torch.version.cuda
    torch_version = torch.__version__
    hardware_key = build_hardware_key(
        os_name=os_name,
        gpu_name=gpu_name,
        total_vram_mb=total_vram_mb,
        compute_capability=compute_capability,
        cuda_runtime=cuda_runtime,
        torch_version=torch_version,
    )
    return HardwareProfile(
        hardware_key=hardware_key,
        os_name=os_name,
        gpu_name=gpu_name,
        total_vram_mb=total_vram_mb,
        compute_capability=compute_capability,
        cuda_runtime=cuda_runtime,
        torch_version=torch_version,
    )
