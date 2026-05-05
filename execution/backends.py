"""Execution backends for exclusive and packed worker launches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import hashlib
import importlib
import os
import shutil
import subprocess
import sys

from ..schemas import TrainingJob
from ..settings import SchedulerSettings
from .executor import SubprocessExecutor, WorkerProcessHandle


class ExecutionBackend(Protocol):
    name: str

    def available(self) -> bool:
        ...

    def launch(self, jobs: list[TrainingJob]) -> list[WorkerProcessHandle]:
        ...


def _group_log_paths(settings: SchedulerSettings, jobs: list[TrainingJob], suffix: str) -> tuple[Path, Path]:
    group_key = hashlib.sha1(",".join(sorted(job.job_id for job in jobs)).encode("utf-8")).hexdigest()[:12]
    runtime_dir = settings.job_runtime_dir(jobs[0].job_id)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir / f"{suffix}_{group_key}.stdout.log", runtime_dir / f"{suffix}_{group_key}.stderr.log"


def _cuda_runtime_visible(device_index: int) -> bool:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None and visible_devices.strip() in {"", "-1", "none", "None"}:
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
            extra_env["CUDA_VISIBLE_DEVICES"] = str(self.settings.gpu_scheduler.device_index)
        return [self.executor.start(job, extra_env=extra_env)]


@dataclass(slots=True)
class CudaProcessBackend:
    settings: SchedulerSettings
    executor: SubprocessExecutor
    name: str = "cuda_process"

    def available(self) -> bool:
        return bool(self.settings.gpu_scheduler.cuda_process.enabled)

    def launch(self, jobs: list[TrainingJob]) -> list[WorkerProcessHandle]:
        if len(jobs) < 2:
            raise ValueError("cuda_process backend expects at least two jobs")
        base_env = {
            "CUDA_VISIBLE_DEVICES": str(self.settings.gpu_scheduler.device_index),
            "OMP_NUM_THREADS": str(self.settings.gpu_scheduler.cuda_process.default_omp_num_threads),
            "MKL_NUM_THREADS": str(self.settings.gpu_scheduler.cuda_process.default_mkl_num_threads),
        }
        return [self.executor.start(job, extra_env=base_env) for job in jobs]


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
        if len(jobs) == 2:
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
        return [{**pipe_env, "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(max(1, min(100, pct)))} for pct in percentages[: len(jobs)]]

    def _ensure_runtime(self) -> None:
        if not self.available() or not self.mps_binary:
            raise RuntimeError("MPS backend unavailable")
        daemon_env = self._daemon_env()
        Path(daemon_env["CUDA_MPS_PIPE_DIRECTORY"]).mkdir(parents=True, exist_ok=True)
        Path(daemon_env["CUDA_MPS_LOG_DIRECTORY"]).mkdir(parents=True, exist_ok=True)
        subprocess.run([self.mps_binary, "-d"], check=False, capture_output=True, text=True, timeout=5.0, env=daemon_env)

    def launch(self, jobs: list[TrainingJob]) -> list[WorkerProcessHandle]:
        if len(jobs) < 2:
            raise ValueError("mps backend expects at least two jobs")
        self._ensure_runtime()
        job_envs = self._client_envs(jobs)
        return [self.executor.start(job, extra_env=job_env) for job, job_env in zip(jobs, job_envs, strict=True)]


@dataclass(slots=True)
class StreamBackend:
    settings: SchedulerSettings
    executor: SubprocessExecutor
    name: str = "stream"

    def available(self) -> bool:
        return bool(self.settings.gpu_scheduler.stream.enabled)

    def launch(self, jobs: list[TrainingJob]) -> list[WorkerProcessHandle]:
        if len(jobs) < 2:
            raise ValueError("stream backend expects at least two jobs")
        stdout_path, stderr_path = _group_log_paths(self.settings, jobs, "stream_host")
        python_executable = jobs[0].config.python_executable or self.settings.python_executable or sys.executable
        env = os.environ.copy()
        project_root = self.executor.project_root
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(project_root) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        env["CUDA_VISIBLE_DEVICES"] = str(self.settings.gpu_scheduler.device_index)

        with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open("a", encoding="utf-8") as stderr_handle:
            process = subprocess.Popen(
                [
                    python_executable,
                    "-m",
                    "localml_scheduler.execution.stream_host",
                    "--runtime-root",
                    str(self.settings.runtime_root),
                    *[item for job in jobs for item in ("--job-id", job.job_id)],
                ],
                cwd=str(project_root),
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
        return [
            WorkerProcessHandle(
                job_id=job.job_id,
                process=process,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                monitor_via_store=True,
            )
            for job in jobs
        ]
