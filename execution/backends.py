"""Execution backends for exclusive and packed worker launches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import importlib
import os
import shutil
import subprocess
import sys

from ..domain import TrainingJob
from ..config import SchedulerSettings
from ..cuda_device_mapping import physical_cuda_device_selector
from .executor import SubprocessExecutor, WorkerProcessHandle


class ExecutionBackend(Protocol):
    name: str

    def available(self) -> bool: ...

    def launch(self, jobs: list[TrainingJob]) -> list[WorkerProcessHandle]: ...


def _cuda_runtime_visible(device_index: int) -> bool:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None and visible_devices.strip() in {
        "",
        "-1",
        "none",
        "None",
    }:
        return False
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return False
    try:
        if not bool(torch.cuda.is_available()):
            return False
        return int(torch.cuda.device_count()) > int(device_index)
    except Exception:
        return False


def _ensure_gpu_execution_slot_available(device_index: int) -> None:
    """Reject a launch when another process owns an exclusive GPU context.

    ``torch.cuda.is_available()`` only establishes that a device can be
    enumerated.  Under NVIDIA's ``Exclusive_Process`` compute mode it remains
    true even though a second process cannot create a CUDA context.  Checking
    that condition before launching a worker keeps a hardware-allocation
    failure from being attributed to the submitted training program.

    The check deliberately fails open when ``nvidia-smi`` is absent or cannot
    answer: scheduler deployment must remain usable on supported non-NVIDIA
    test hosts, and the worker remains the final authority in that case.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return
    selector = physical_cuda_device_selector(device_index)
    try:
        compute_mode = subprocess.run(
            [
                nvidia_smi,
                f"--id={selector}",
                "--query-gpu=compute_mode",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        if compute_mode.returncode != 0:
            return
        mode = compute_mode.stdout.strip().lower().replace(" ", "_")
        if mode not in {"exclusive_process", "exclusive_thread"}:
            return
        processes = subprocess.run(
            [
                nvidia_smi,
                f"--id={selector}",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.SubprocessError):
        return
    if processes.returncode == 0 and processes.stdout.strip():
        raise RuntimeError(
            f"CUDA device {selector} is occupied under {mode}; "
            "deferring GPU job until its exclusive owner exits"
        )


@dataclass(slots=True)
class ExclusiveBackend:
    settings: SchedulerSettings
    executor: SubprocessExecutor
    name: str = "exclusive"

    def available(self) -> bool:
        return True

    def launch(self, jobs: list[TrainingJob]) -> list[WorkerProcessHandle]:
        if len(jobs) != 1:
            raise ValueError("exclusive backend expects exactly one job")
        job = jobs[0]
        extra_env: dict[str, str] = {}
        if job.resource_requirements.requires_gpu:
            _ensure_gpu_execution_slot_available(
                self.settings.gpu_scheduler.device_index
            )
            extra_env["CUDA_VISIBLE_DEVICES"] = physical_cuda_device_selector(
                self.settings.gpu_scheduler.device_index
            )
        return [self.executor.start(job, extra_env=extra_env)]


@dataclass(slots=True)
class CudaProcessBackend:
    settings: SchedulerSettings
    executor: SubprocessExecutor
    name: str = "cuda_process"

    def available(self) -> bool:
        return bool(self.settings.gpu_scheduler.cuda_process.enabled)

    def launch(self, jobs: list[TrainingJob]) -> list[WorkerProcessHandle]:
        if not jobs:
            raise ValueError("cuda_process backend expects at least one job")
        if any(job.resource_requirements.requires_gpu for job in jobs):
            _ensure_gpu_execution_slot_available(
                self.settings.gpu_scheduler.device_index
            )
        base_env = {
            "CUDA_VISIBLE_DEVICES": physical_cuda_device_selector(
                self.settings.gpu_scheduler.device_index
            ),
            "OMP_NUM_THREADS": str(
                self.settings.gpu_scheduler.cuda_process.default_omp_num_threads
            ),
            "MKL_NUM_THREADS": str(
                self.settings.gpu_scheduler.cuda_process.default_mkl_num_threads
            ),
        }
        return [self.executor.start(job, extra_env=base_env) for job in jobs]


@dataclass(slots=True)
class MPSBackend:
    settings: SchedulerSettings
    executor: SubprocessExecutor
    mps_binary: str | None = None
    name: str = "mps_process"

    def __post_init__(self) -> None:
        if self.mps_binary is None:
            self.mps_binary = shutil.which("nvidia-cuda-mps-control")

    def available(self) -> bool:
        supported_platform = sys.platform.startswith("linux") or sys.platform == "qnx"
        return bool(
            supported_platform
            and self.settings.gpu_scheduler.mps.enabled
            and self.mps_binary
            and _cuda_runtime_visible(self.settings.gpu_scheduler.device_index)
        )

    def _daemon_env(self) -> dict[str, str]:
        mps_settings = self.settings.gpu_scheduler.mps
        return {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": physical_cuda_device_selector(
                self.settings.gpu_scheduler.device_index
            ),
            "CUDA_MPS_PIPE_DIRECTORY": mps_settings.pipe_directory,
            "CUDA_MPS_LOG_DIRECTORY": mps_settings.log_directory,
        }

    def _client_envs(self, jobs: list[TrainingJob]) -> list[dict[str, str]]:
        mps_settings = self.settings.gpu_scheduler.mps
        pipe_env = {
            "CUDA_MPS_PIPE_DIRECTORY": mps_settings.pipe_directory,
            "CUDA_MPS_LOG_DIRECTORY": mps_settings.log_directory,
            "OMP_NUM_THREADS": str(mps_settings.default_omp_num_threads),
            "MKL_NUM_THREADS": str(mps_settings.default_mkl_num_threads),
        }
        selected_config = jobs[0].metadata.get("placement_backend_config") if jobs else None
        configured_percentages = (
            selected_config.get("allocation_percentages")
            if isinstance(selected_config, dict)
            else None
        )
        if isinstance(configured_percentages, list) and len(configured_percentages) == len(jobs):
            percentages = [int(value) for value in configured_percentages]
        elif len(jobs) == 1:
            percentages = [100]
        elif len(jobs) == 2:
            percentages = [
                mps_settings.default_primary_active_thread_pct,
                mps_settings.default_secondary_active_thread_pct,
            ]
        else:
            primary = max(1, min(100, mps_settings.default_primary_active_thread_pct))
            remaining = max(1, 100 - primary)
            secondary_count = max(1, len(jobs) - 1)
            secondary_pct = max(1, remaining // secondary_count)
            percentages = [primary] + [secondary_pct] * secondary_count
        return [
            {
                **pipe_env,
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(max(1, min(100, pct))),
            }
            for pct in percentages[: len(jobs)]
        ]

    def _ensure_runtime(self) -> None:
        if not self.available() or not self.mps_binary:
            raise RuntimeError("MPS backend unavailable")
        _ensure_gpu_execution_slot_available(self.settings.gpu_scheduler.device_index)
        daemon_env = self._daemon_env()
        Path(daemon_env["CUDA_MPS_PIPE_DIRECTORY"]).mkdir(parents=True, exist_ok=True)
        Path(daemon_env["CUDA_MPS_LOG_DIRECTORY"]).mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [self.mps_binary, "-d"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
            env=daemon_env,
        )

    def launch(self, jobs: list[TrainingJob]) -> list[WorkerProcessHandle]:
        if not jobs:
            raise ValueError("mps_process backend expects at least one job")
        self._ensure_runtime()
        job_envs = self._client_envs(jobs)
        return [
            self.executor.start(job, extra_env=job_env)
            for job, job_env in zip(jobs, job_envs, strict=True)
        ]
