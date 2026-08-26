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
            extra_env["CUDA_VISIBLE_DEVICES"] = str(
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
        base_env = {
            "CUDA_VISIBLE_DEVICES": str(self.settings.gpu_scheduler.device_index),
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
            "CUDA_VISIBLE_DEVICES": str(self.settings.gpu_scheduler.device_index),
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
