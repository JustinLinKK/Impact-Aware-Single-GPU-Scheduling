from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import TrainingJob
from localml_scheduler.execution.backend_registry import BackendRegistry
from localml_scheduler.execution.backends import CudaProcessBackend, MPSBackend
from localml_scheduler.execution.executor import WorkerProcessHandle
from localml_scheduler.scheduler.supervisor import WorkerSupervisor


def _job(name: str) -> TrainingJob:
    return TrainingJob.create(
        job_id=name,
        runner_target="builtins:dict",
        baseline_model_id=name,
        baseline_model_path="/tmp/none",
    )


class _RecordingExecutor:
    def __init__(self, root: Path):
        self.root = root
        self.calls: list[tuple[str, dict[str, str]]] = []

    def start(self, job: TrainingJob, *, extra_env=None):
        self.calls.append((job.job_id, dict(extra_env or {})))
        return WorkerProcessHandle(
            job_id=job.job_id,
            process=Mock(),
            stdout_path=self.root / f"{job.job_id}.out",
            stderr_path=self.root / f"{job.job_id}.err",
        )


def test_registry_contains_only_canonical_runtime_backends(tmp_path: Path) -> None:
    settings = SchedulerSettings(runtime_root=tmp_path)
    registry = BackendRegistry(settings, _RecordingExecutor(tmp_path))
    assert set(dict(registry.items())) == {
        "exclusive",
        "cuda_process",
        "mps_process",
    }


def test_cuda_process_launches_independent_jobs_without_mps_env(tmp_path: Path) -> None:
    settings = SchedulerSettings(
        runtime_root=tmp_path, gpu_scheduler={"packing_backend": "cuda_process"}
    )
    executor = _RecordingExecutor(tmp_path)
    backend = CudaProcessBackend(settings, executor)
    handles = backend.launch([_job("a"), _job("b")])
    assert [handle.job_id for handle in handles] == ["a", "b"]
    assert handles[0].process is not handles[1].process
    assert all(
        not any(key.startswith("CUDA_MPS_") for key in env)
        for _job_id, env in executor.calls
    )


def test_mps_process_launches_independent_clients_with_fixed_allocations(
    tmp_path: Path,
) -> None:
    settings = SchedulerSettings(
        runtime_root=tmp_path, gpu_scheduler={"packing_backend": "mps_process"}
    )
    executor = _RecordingExecutor(tmp_path)
    jobs = [_job("a"), _job("b")]
    for job in jobs:
        job.metadata["placement_backend_config"] = {
            "allocation_percentages": [60, 40]
        }
    backend = MPSBackend(settings, executor, mps_binary="mps")
    with patch.object(MPSBackend, "_ensure_runtime", autospec=True):
        handles = backend.launch(jobs)
    assert handles[0].process is not handles[1].process
    assert [
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] for _job_id, env in executor.calls
    ] == ["60", "40"]
    assert all("CUDA_MPS_PIPE_DIRECTORY" in env for _job_id, env in executor.calls)


def test_mps_unavailable_fail_policy_is_explicit(tmp_path: Path) -> None:
    settings = SchedulerSettings(
        runtime_root=tmp_path,
        gpu_scheduler={
            "packing_backend": "mps_process",
            "mps_unavailable_policy": "fail",
        },
    )
    unavailable = Mock()
    unavailable.available.return_value = False
    supervisor = WorkerSupervisor(
        settings,
        backends={
            "exclusive": Mock(available=Mock(return_value=True)),
            "cuda_process": Mock(available=Mock(return_value=True)),
            "mps_process": unavailable,
        },
    )
    with pytest.raises(RuntimeError, match="MPS is unavailable"):
        supervisor.available_backends()


def test_mps_unavailable_exclusive_policy_keeps_only_safe_fallback(
    tmp_path: Path,
) -> None:
    settings = SchedulerSettings(
        runtime_root=tmp_path,
        gpu_scheduler={
            "packing_backend": "mps_process",
            "mps_unavailable_policy": "exclusive",
            "exclusive_fallback_enabled": True,
        },
    )
    supervisor = WorkerSupervisor(
        settings,
        backends={
            "exclusive": Mock(available=Mock(return_value=True)),
            "cuda_process": Mock(available=Mock(return_value=True)),
            "mps_process": Mock(available=Mock(return_value=False)),
        },
    )

    availability = supervisor.available_backends()
    assert availability["exclusive"] is True
    assert availability["mps_process"] is False
    assert supervisor._overlap_allowed_for_backend("cuda_process") is False
