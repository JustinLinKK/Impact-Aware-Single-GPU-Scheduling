"""Execution backends for exclusive and packed worker launches."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
import hashlib
import importlib
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time

from ..domain import TrainingJob
from ..config import SchedulerSettings
from .executor import SubprocessExecutor, WorkerProcessHandle
from .process_utils import start_new_session_kwargs, terminate_process_tree


class ExecutionBackend(Protocol):
    name: str

    def available(self) -> bool: ...

    def launch(self, jobs: list[TrainingJob]) -> list[WorkerProcessHandle]: ...


def _group_log_paths(
    settings: SchedulerSettings, jobs: list[TrainingJob], suffix: str
) -> tuple[Path, Path]:
    group_key = hashlib.sha1(
        ",".join(sorted(job.job_id for job in jobs)).encode("utf-8")
    ).hexdigest()[:12]
    runtime_dir = settings.job_runtime_dir(jobs[0].job_id)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return (
        runtime_dir / f"{suffix}_{group_key}.stdout.log",
        runtime_dir / f"{suffix}_{group_key}.stderr.log",
    )


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
        if len(jobs) == 1:
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
            raise ValueError("mps backend expects at least one job")
        self._ensure_runtime()
        job_envs = self._client_envs(jobs)
        return [
            self.executor.start(job, extra_env=job_env)
            for job, job_env in zip(jobs, job_envs, strict=True)
        ]


@dataclass(slots=True)
class StreamBackend:
    settings: SchedulerSettings
    executor: SubprocessExecutor
    name: str = "stream"
    _process: subprocess.Popen | None = field(init=False, default=None, repr=False)
    _socket_path: Path = field(init=False, repr=False)
    _stdout_path: Path = field(init=False, repr=False)
    _stderr_path: Path = field(init=False, repr=False)
    _lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._process = None
        self._socket_path = Path(
            "/tmp"
        ) / f"localml-stream-{hashlib.sha1(str(self.settings.runtime_root).encode('utf-8')).hexdigest()[:16]}.sock"
        self._stdout_path = self.settings.runtime_root / "stream_host" / "stdout.log"
        self._stderr_path = self.settings.runtime_root / "stream_host" / "stderr.log"
        self._lock = threading.Lock()

    def available(self) -> bool:
        # Capability discovery is intentionally configuration-only.  This
        # keeps planning and CPU unit tests deterministic; the host performs
        # the authoritative CUDA check when a stream placement is launched.
        return bool(self.settings.gpu_scheduler.stream.enabled)

    def _request(self, payload: dict[str, object], *, timeout: float = 10.0) -> dict:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(timeout)
        try:
            connection.connect(str(self._socket_path))
            stream = connection.makefile("rwb")
            stream.write((json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
            stream.flush()
            raw = stream.readline()
            if not raw:
                raise RuntimeError("CUDA stream host closed the control connection")
            response = json.loads(raw.decode("utf-8"))
        finally:
            connection.close()
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "CUDA stream host rejected request"))
        return dict(response)

    def _ensure_host(self, python_executable: str) -> subprocess.Popen:
        if self._process is not None and self._process.poll() is None:
            try:
                self._request({"op": "ping"}, timeout=2.0)
                return self._process
            except (OSError, RuntimeError, ValueError):
                terminate_process_tree(self._process, timeout=2.0)
        self._socket_path.unlink(missing_ok=True)
        self._stdout_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        project_root = self.executor.project_root
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(project_root) + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
        env["CUDA_VISIBLE_DEVICES"] = str(
            self.settings.gpu_scheduler.device_index
        )
        with self._stdout_path.open("a", encoding="utf-8") as stdout_handle, self._stderr_path.open(
            "a", encoding="utf-8"
        ) as stderr_handle:
            self._process = subprocess.Popen(
                [
                    python_executable,
                    "-m",
                    "localml_scheduler.execution.stream_host",
                    "--runtime-root",
                    str(self.settings.runtime_root),
                    "--socket-path",
                    str(self._socket_path),
                ],
                cwd=str(project_root),
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                **start_new_session_kwargs(),
            )
        deadline = time.monotonic() + 30.0
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"CUDA stream host exited during startup with code {self._process.returncode}; "
                    f"see {self._stderr_path}"
                )
            try:
                self._request({"op": "ping"}, timeout=1.0)
                return self._process
            except (OSError, RuntimeError, ValueError) as exc:
                last_error = exc
                time.sleep(0.05)
        terminate_process_tree(self._process, timeout=2.0)
        raise RuntimeError(f"CUDA stream host did not become ready: {last_error}")

    def launch(self, jobs: list[TrainingJob]) -> list[WorkerProcessHandle]:
        if not jobs:
            raise ValueError("stream backend expects at least one job")
        python_executable = (
            jobs[0].config.python_executable
            or self.settings.python_executable
            or sys.executable
        )
        with self._lock:
            process = self._ensure_host(python_executable)
            self._request(
                {"op": "launch", "job_ids": [job.job_id for job in jobs]},
                timeout=35.0,
            )
        return [
            WorkerProcessHandle(
                job_id=job.job_id,
                process=process,
                stdout_path=self._stdout_path,
                stderr_path=self._stderr_path,
                monitor_via_store=True,
            )
            for job in jobs
        ]

    def shutdown(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
            if process is None:
                self._socket_path.unlink(missing_ok=True)
                return
            if process.poll() is None:
                try:
                    self._request({"op": "shutdown"}, timeout=2.0)
                    process.wait(timeout=5.0)
                except Exception:
                    terminate_process_tree(process, timeout=2.0)
            self._socket_path.unlink(missing_ok=True)
