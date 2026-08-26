from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import time
import unittest

import torch

from localml_scheduler.adapters.mlevolve import build_mlevolve_job
from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import (
    CheckpointPolicy,
    JobStatus,
    ResourceRequirements,
    RuntimeProfile,
)
from localml_scheduler.examples.toy_pytorch_runner import create_toy_baseline_checkpoint
from localml_scheduler.execution.backends import ExclusiveBackend
from localml_scheduler.execution.executor import SubprocessExecutor
from localml_scheduler.scheduler.supervisor import WorkerSupervisor


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "PerfSeer-predictor" / "models" / "nvidia_a10" / "student_a10_cpu.torchscript.pt"
JOBLIST = ROOT / "scheduler_benchmark_test" / "fixtures" / "perfseer_ml_12_job_1_epoch" / "joblist.json"
SUPPORTED_SOURCE = Path(__file__).parent / "fixtures" / "perfseer_stress_models.py"
UNSUPPORTED_SOURCE = Path(__file__).parent / "fixtures" / "perfseer_unsupported.py"


@dataclass
class _CpuParallelBackend:
    settings: SchedulerSettings
    executor: SubprocessExecutor
    name: str = "cuda_process"

    def available(self) -> bool:
        return True

    def launch(self, jobs):
        environment = {
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "void",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        return [self.executor.start(job, extra_env=environment) for job in jobs]


def _wait_for(predicate, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise TimeoutError("12-job ML stress fixture did not finish")


class MLStressTest(unittest.TestCase):
    def test_twelve_one_epoch_jobs_use_ml_with_one_job_fallback(self) -> None:
        fixture = json.loads(JOBLIST.read_text(encoding="utf-8"))
        self.assertEqual(len(fixture["jobs"]), 12)
        with tempfile.TemporaryDirectory() as temporary:
            runtime_root = Path(temporary)
            settings = SchedulerSettings(
                runtime_root=runtime_root,
                scheduler_poll_interval_seconds=0.05,
                graph_db={"enabled": False},
                hardware_feature_db={"enabled": False},
                prediction={
                    "mode": "ml_predictor",
                    "ml": {
                        "test_override_enabled": True,
                        "test_model_path": str(ARTIFACT),
                        "source_conversion_timeout_seconds": 10,
                        "cache_size": 64,
                    },
                },
                gpu_scheduler={
                    "mode": "parallel_time_aware",
                    "packing_backend": "cuda_process",
                    "exclusive_fallback_enabled": True,
                    "parallel_job_cap": 4,
                    "priority_window_size": 4,
                    "memory": {
                        "gpu_vram_gib": 28.0,
                        "predicted_budget_fraction": 0.85,
                    },
                },
            )
            api = SchedulerClient(settings)
            executor = SubprocessExecutor(settings)
            supervisor = WorkerSupervisor(
                settings,
                backends={
                    "exclusive": ExclusiveBackend(settings, executor),
                    "cuda_process": _CpuParallelBackend(settings, executor),
                },
            )
            baseline = create_toy_baseline_checkpoint(runtime_root / "baseline" / "toy.pt")
            jobs = []
            for index, item in enumerate(fixture["jobs"]):
                unsupported = item["id"].endswith("unsupported")
                source = UNSUPPORTED_SOURCE if unsupported else SUPPORTED_SOURCE
                constructor_kwargs = {} if unsupported else {"width": item["width"]}
                job = build_mlevolve_job(
                    workflow_id="perfseer-ml-stress",
                    baseline_model_id="shared-toy-baseline",
                    baseline_model_path=baseline,
                    runner_target="localml_scheduler.examples.toy_pytorch_runner:run_toy_training_job",
                    runner_kwargs={
                        "batch_size": 2,
                        "epochs": 1,
                        "num_samples": 8,
                        "sleep_per_step": 0.005,
                        "learning_rate": 0.01 + index * 0.001,
                    },
                    priority=12 - index,
                    task_type="synthetic_classification",
                    resource_requirements=ResourceRequirements(
                        requires_gpu=False,
                        estimated_avg_vram_mb=256,
                        estimated_ram_mb=128,
                    ),
                    checkpoint_policy=CheckpointPolicy(save_every_epoch=True),
                    packing_family=f"stress-{item['entry']}",
                    packing_eligible=True,
                    max_steps=4,
                    max_epochs=1,
                    metadata={
                        "perfseer_model": {
                            "source_path": str(source),
                            "entry": item["entry"],
                            "input_shapes": [item["shape"]],
                            "input_dtypes": [item["dtype"]],
                            "precision": "fp32_ieee",
                            "constructor_kwargs": constructor_kwargs,
                        }
                    },
                )
                jobs.append(job)
                for backend_name in ("exclusive", "cuda_process"):
                    api.upsert_runtime_profile(
                        RuntimeProfile.create(
                            signature=job.packing.signature or job.job_id,
                            hardware_key=api.store.hardware_key(),
                            backend_name=backend_name,
                            resolved_batch_size=2,
                            strategy="epoch_1",
                            epoch_1_seconds=0.1,
                            avg_step_time_ms=25.0,
                            estimated_total_runtime_seconds=0.1,
                            confidence=1.0,
                            observations=1,
                            source="fixture",
                        )
                    )
                api.submit(job)

            cuda_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            service = api.create_service(supervisor=supervisor).start(background=True)
            try:
                _wait_for(lambda: all((api.inspect(job.job_id) or job).status.is_terminal for job in jobs))
            finally:
                service.stop()
            cuda_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

            completed = [api.inspect(job.job_id) for job in jobs]
            self.assertTrue(all(job is not None and job.status == JobStatus.COMPLETED for job in completed))
            sources = {job.job_id: job.metadata.get("vram_prediction_source") for job in completed if job}
            errors = {job.job_id: job.metadata.get("vram_prediction_error") for job in completed if job}
            self.assertEqual(sources[jobs[-1].job_id], "branch_profile")
            self.assertTrue(
                all(sources[job.job_id] == "ml_predictor" for job in jobs[:-1]),
                {"sources": sources, "errors": errors},
            )
            self.assertIn("unsupported", (completed[-1].metadata.get("vram_prediction_error") or "").lower())
            self.assertEqual(cuda_after, cuda_before)


if __name__ == "__main__":
    unittest.main()
