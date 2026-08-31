from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from localml_scheduler.config import SchedulerSettings
from localml_scheduler.cuda_device_mapping import physical_cuda_device_selector
from localml_scheduler.domain import TrainingJob
from localml_scheduler.execution.backends import CudaProcessBackend, ExclusiveBackend
from localml_scheduler.execution.executor import WorkerProcessHandle
from localml_scheduler.scheduler.telemetry import NvidiaSmiTelemetrySampler


class _RecordingExecutor:
    def __init__(self, root: Path):
        self.root = root
        self.environments: list[dict[str, str]] = []

    def start(self, job: TrainingJob, *, extra_env=None):
        self.environments.append(dict(extra_env or {}))
        return WorkerProcessHandle(
            job_id=job.job_id,
            process=Mock(),
            stdout_path=self.root / f"{job.job_id}.out",
            stderr_path=self.root / f"{job.job_id}.err",
        )


def _job(job_id: str) -> TrainingJob:
    return TrainingJob.create(
        job_id=job_id,
        runner_target="builtins:dict",
        baseline_model_id=job_id,
        baseline_model_path="/tmp/none",
    )


def test_masked_logical_device_resolves_to_the_physical_selector(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")

    assert physical_cuda_device_selector(0) == "2"


def test_backends_keep_workers_on_the_parent_masked_gpu(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    settings = SchedulerSettings(
        runtime_root=tmp_path,
        gpu_scheduler={"device_index": 0, "packing_backend": "cuda_process"},
    )
    exclusive_executor = _RecordingExecutor(tmp_path)
    cuda_executor = _RecordingExecutor(tmp_path)

    ExclusiveBackend(settings, exclusive_executor).launch([_job("exclusive")])
    CudaProcessBackend(settings, cuda_executor).launch([_job("packed")])

    assert exclusive_executor.environments[0]["CUDA_VISIBLE_DEVICES"] == "2"
    assert cuda_executor.environments[0]["CUDA_VISIBLE_DEVICES"] == "2"


def test_nvidia_smi_telemetry_samples_the_masked_physical_gpu(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    sampler = NvidiaSmiTelemetrySampler(device_index=0)
    completed = SimpleNamespace(returncode=0, stdout="1024, 23028, 50, 20\n")

    with patch("localml_scheduler.scheduler.telemetry.subprocess.run", return_value=completed) as run:
        sample = sampler.sample()

    assert sample is not None
    assert "--id=2" in run.call_args.args[0]
