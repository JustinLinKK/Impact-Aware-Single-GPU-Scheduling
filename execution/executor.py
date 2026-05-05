"""Subprocess launcher for worker jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import subprocess
import sys

from ..schemas import TrainingJob
from ..settings import SchedulerSettings


@dataclass(slots=True)
class WorkerProcessHandle:
    job_id: str
    process: subprocess.Popen
    stdout_path: Path
    stderr_path: Path
    monitor_via_store: bool = False


class SubprocessExecutor:
    """Launch worker_entry.py in an isolated Python subprocess."""

    def __init__(self, settings: SchedulerSettings):
        self.settings = settings
        self.project_root = Path(__file__).resolve().parents[2]

    def start(self, job: TrainingJob, *, extra_env: dict[str, str] | None = None) -> WorkerProcessHandle:
        runtime_dir = self.settings.job_runtime_dir(job.job_id)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = runtime_dir / "stdout.log"
        stderr_path = runtime_dir / "stderr.log"
        python_executable = job.config.python_executable or self.settings.python_executable or sys.executable

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(self.project_root) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        env.update(job.config.env)
        if extra_env:
            env.update(extra_env)

        stdout_handle = stdout_path.open("a", encoding="utf-8")
        stderr_handle = stderr_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            [
                python_executable,
                "-m",
                "localml_scheduler.execution.worker_entry",
                "--runtime-root",
                str(self.settings.runtime_root),
                "--job-id",
                job.job_id,
            ],
            cwd=str(self.project_root),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        stdout_handle.close()
        stderr_handle.close()
        return WorkerProcessHandle(job_id=job.job_id, process=process, stdout_path=stdout_path, stderr_path=stderr_path)
