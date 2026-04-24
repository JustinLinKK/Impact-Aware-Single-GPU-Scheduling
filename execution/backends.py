"""Execution backends for exclusive and pair-packed worker launches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import shutil
import subprocess

from ..schemas import TrainingJob
from ..settings import SchedulerSettings
from .executor import SubprocessExecutor, WorkerProcessHandle


class ExecutionBackend(Protocol):
    name: str

    def available(self) -> bool:
        ...

    def launch(self, jobs: list[TrainingJob]) -> list[WorkerProcessHandle]:
        ...


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
            extra_env["CUDA_VISIBLE_DEVICES"] = str(self.settings.gpu_scheduler.device_index)
        return [self.executor.start(job, extra_env=extra_env)]


@dataclass(slots=True)
class MPSBackend:
    settings: SchedulerSettings
    executor: SubprocessExecutor
    mps_binary: str | None = None
    name: str = "mps"

    def __post_init__(self) -> None:
        if self.mps_binary is None:
            self.mps_binary = shutil.which("nvidia-cuda-mps-control")

    def available(self) -> bool:
        return bool(self.settings.gpu_scheduler.mps.enabled and self.mps_binary)

    def _ensure_runtime(self) -> None:
        if not self.available() or not self.mps_binary:
            raise RuntimeError("MPS backend unavailable")
        subprocess.run([self.mps_binary, "-d"], check=False, capture_output=True, text=True, timeout=5.0)

    def launch(self, jobs: list[TrainingJob]) -> list[WorkerProcessHandle]:
        if len(jobs) != 2:
            raise ValueError("mps backend expects exactly two jobs")
        self._ensure_runtime()
        primary_pct = str(self.settings.gpu_scheduler.mps.default_primary_active_thread_pct)
        secondary_pct = str(self.settings.gpu_scheduler.mps.default_secondary_active_thread_pct)
        base_env = {
            "CUDA_VISIBLE_DEVICES": str(self.settings.gpu_scheduler.device_index),
            "OMP_NUM_THREADS": str(self.settings.gpu_scheduler.mps.default_omp_num_threads),
            "MKL_NUM_THREADS": str(self.settings.gpu_scheduler.mps.default_mkl_num_threads),
        }
        job_envs = [
            {**base_env, "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": primary_pct},
            {**base_env, "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": secondary_pct},
        ]
        return [
            self.executor.start(job, extra_env=job_env)
            for job, job_env in zip(jobs, job_envs, strict=True)
        ]
