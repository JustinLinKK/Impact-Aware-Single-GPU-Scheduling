from __future__ import annotations

from pathlib import Path
import tempfile
import time
import unittest

from localml_scheduler.adapters.mlevolve import build_mlevolve_job
from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import (
    BatchSizeObservation,
    CheckpointPolicy,
    JobStatus,
    ResourceRequirements,
    RuntimeProfile,
    TrainingJob,
    build_batch_probe_shape_signature,
    build_batch_size_observation_key,
)
from localml_scheduler.execution.backends import CudaProcessBackend, ExclusiveBackend
from localml_scheduler.execution.executor import SubprocessExecutor
from localml_scheduler.examples.toy_pytorch_runner import create_toy_baseline_checkpoint
from localml_scheduler.scheduler.supervisor import WorkerSupervisor


def wait_for(predicate, timeout: float = 30.0, interval: float = 0.05) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise TimeoutError("condition not met in time")


def _supervisor(settings: SchedulerSettings, *, include_cuda_process: bool) -> WorkerSupervisor:
    executor = SubprocessExecutor(settings)
    backends = {"exclusive": ExclusiveBackend(settings, executor)}
    if include_cuda_process:
        backends["cuda_process"] = CudaProcessBackend(settings, executor)
    return WorkerSupervisor(settings, backends=backends)


def _job(baseline: str, *, learning_rate: float, max_steps: int = 20) -> TrainingJob:
    return build_mlevolve_job(
        workflow_id="wf-time-aware",
        baseline_model_id=f"baseline-{learning_rate}",
        baseline_model_path=baseline,
        runner_target="localml_scheduler.examples.toy_pytorch_runner:run_toy_training_job",
        runner_kwargs={
            "sleep_per_step": 0.02,
            "learning_rate": learning_rate,
            "batch_size": 8,
        },
        priority=5,
        task_type="toy_classification",
        resource_requirements=ResourceRequirements(
            requires_gpu=False,
            estimated_avg_vram_mb=512,
            estimated_ram_mb=512,
        ),
        checkpoint_policy=CheckpointPolicy(save_every_n_steps=1, save_every_epoch=True),
        packing_family="toy-mlp",
        packing_eligible=True,
        packing_backend_allowlist=["cuda_process"],
        max_steps=max_steps,
        max_epochs=2,
    )


def _seed_time_options(api: SchedulerClient, job: TrainingJob) -> None:
    hardware_key = api.store.hardware_key()
    shape_signature = build_batch_probe_shape_signature(job)
    signature = job.packing.signature or job.job_id
    for backend in ("exclusive", "cuda_process"):
        for batch_size in (2, 4, 8, 16, 32):
            api.store.upsert_batch_size_observation(
                BatchSizeObservation(
                    observation_key=build_batch_size_observation_key(
                        job.baseline_model_id,
                        shape_signature,
                        hardware_key,
                        backend,
                        batch_size,
                    ),
                    model_key=job.baseline_model_id,
                    shape_signature=shape_signature,
                    hardware_key=hardware_key,
                    backend_name=backend,
                    batch_param_name="batch_size",
                    batch_size=batch_size,
                    avg_vram_mb=256 + batch_size,
                    peak_vram_mb=320 + batch_size,
                    metadata={"estimate_source": "integration_fixture"},
                )
            )
            api.store.upsert_runtime_profile(
                RuntimeProfile.create(
                    signature=signature,
                    hardware_key=hardware_key,
                    backend_name=backend,
                    resolved_batch_size=batch_size,
                    strategy="epoch_1",
                    epoch_1_seconds=0.5 + (1.0 / batch_size),
                    estimated_total_runtime_seconds=1.2,
                    confidence=0.9,
                    observations=1,
                    source="integration_fixture",
                )
            )


class GpuSchedulerIntegrationTest(unittest.TestCase):
    def test_profiled_job_starts_as_time_aware_stack_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            settings = SchedulerSettings(
                runtime_root=runtime_root,
                scheduler_poll_interval_seconds=0.05,
                gpu_scheduler={
                    "mode": "parallel_time_aware",
                    "backend_priority": ["cuda_process", "exclusive"],
                },
                graph_db={"enabled": False},
                hardware_feature_db={"enabled": False},
            )
            api = SchedulerClient(settings)
            service = api.create_service(
                supervisor=_supervisor(settings, include_cuda_process=True)
            ).start(background=True)
            try:
                baseline = create_toy_baseline_checkpoint(
                    runtime_root / "baselines" / "anchor.pt",
                    seed=89,
                )
                job = _job(baseline, learning_rate=0.011)
                _seed_time_options(api, job)

                api.submit(job)
                wait_for(lambda: api.inspect(job.job_id).status == JobStatus.RUNNING)

                running = api.inspect(job.job_id)
                self.assertEqual(running.metadata["placement_mode"], "stack_anchor")
                self.assertEqual(running.metadata["placement_backend"], "cuda_process")
                self.assertEqual(
                    running.metadata["placement_objective_version"],
                    "time_v6_verified_piecewise_drain",
                )
                wait_for(lambda: api.inspect(job.job_id).status.is_terminal)
                self.assertEqual(api.inspect(job.job_id).status, JobStatus.COMPLETED)
            finally:
                service.stop()

    def test_exclusive_only_backend_drains_jobs_one_at_a_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            settings = SchedulerSettings(
                runtime_root=runtime_root,
                scheduler_poll_interval_seconds=0.05,
                gpu_scheduler={
                    "mode": "parallel_time_aware",
                    "backend_priority": ["exclusive"],
                },
                graph_db={"enabled": False},
                hardware_feature_db={"enabled": False},
            )
            api = SchedulerClient(settings)
            service = api.create_service(
                supervisor=_supervisor(settings, include_cuda_process=False)
            ).start(background=True)
            try:
                baseline = create_toy_baseline_checkpoint(
                    runtime_root / "baselines" / "exclusive.pt",
                    seed=90,
                )
                first = _job(baseline, learning_rate=0.021, max_steps=30)
                second = _job(baseline, learning_rate=0.022, max_steps=10)
                api.submit(first)
                api.submit(second)

                wait_for(lambda: api.inspect(first.job_id).status == JobStatus.RUNNING)
                self.assertIn(
                    api.inspect(second.job_id).status,
                    {JobStatus.PENDING, JobStatus.READY},
                )
                wait_for(
                    lambda: api.inspect(first.job_id).status.is_terminal
                    and api.inspect(second.job_id).status.is_terminal
                )
                self.assertEqual(api.inspect(first.job_id).metadata["placement_mode"], "exclusive")
                self.assertEqual(api.inspect(second.job_id).metadata["placement_mode"], "exclusive")
                self.assertEqual(api.report()["packed_dispatches"], 0)
            finally:
                service.stop()


if __name__ == "__main__":
    unittest.main()
