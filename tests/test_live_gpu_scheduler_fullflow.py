from __future__ import annotations

from pathlib import Path
import math
import tempfile
import time
import unittest

import torch

from localml_scheduler.adapters.mlevolve import build_mlevolve_job
from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import BatchProbeSpec, CheckpointPolicy, JobStatus, ResourceRequirements
from localml_scheduler.examples.toy_pytorch_runner import create_toy_baseline_checkpoint


LIVE_GPU_EPOCHS = 5
LIVE_GPU_NUM_SAMPLES = 128
LIVE_GPU_INITIAL_BATCH_SIZE = 32
LIVE_GPU_MAX_PROBE_BATCH_SIZE = 64
PACKED_GPU_EPOCHS = 5
PACKED_GPU_NUM_SAMPLES = 64
PACKED_GPU_BATCH_SIZE = 16


def wait_for(predicate, timeout: float = 90.0, interval: float = 0.1) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise TimeoutError("condition not met in time")


def latest_job_completed_payload(api: SchedulerClient, job_id: str) -> dict:
    wait_for(lambda: bool(api.list_events(job_id=job_id, event_type="job_completed")), timeout=10.0)
    events = api.list_events(job_id=job_id, event_type="job_completed")
    return dict(events[-1]["payload"])


class LiveGpuSchedulerFullFlowTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the live GPU scheduler full-flow test")
    def test_scheduler_runs_real_toy_model_jobs_on_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            settings = SchedulerSettings(
                runtime_root=runtime_root,
                scheduler_poll_interval_seconds=0.05,
                gpu_scheduler={
                    "mode": "adaptive",
                    "backend_priority": ["exclusive"],
                    "stream": {"enabled": False},
                    "cuda_process": {"enabled": False},
                    "mps": {"enabled": False},
                },
                prediction={"mode": "ml_predictor"},
                log_db={"enabled": False},
                redis_cache={"enabled": False},
            )
            api = SchedulerClient(settings)
            service = api.create_service().start(background=True)
            try:
                baseline = create_toy_baseline_checkpoint(
                    runtime_root / "baselines" / "live-gpu-toy.pt",
                    input_dim=1024,
                    hidden_dim=2048,
                    output_dim=16,
                    seed=31,
                )
                jobs = []
                for index, learning_rate in enumerate((0.01, 0.02), start=1):
                    jobs.append(
                        build_mlevolve_job(
                            workflow_id="live-gpu-fullflow",
                            baseline_model_id="live-gpu-toy-baseline",
                            baseline_model_path=baseline,
                            runner_target="localml_scheduler.examples.toy_pytorch_runner:run_toy_training_job",
                            runner_kwargs={
                                "input_dim": 1024,
                                "hidden_dim": 2048,
                                "output_dim": 16,
                                "num_samples": LIVE_GPU_NUM_SAMPLES,
                                "batch_size": LIVE_GPU_INITIAL_BATCH_SIZE,
                                "epochs": LIVE_GPU_EPOCHS,
                                "learning_rate": learning_rate,
                                "optimizer": "sgd",
                                "probe_max_batch_size": LIVE_GPU_MAX_PROBE_BATCH_SIZE,
                            },
                            priority=10 - index,
                            task_type="live_gpu_toy_classification",
                            resource_requirements=ResourceRequirements(
                                requires_gpu=True,
                                estimated_vram_mb=512,
                                estimated_ram_mb=1024,
                            ),
                            checkpoint_policy=CheckpointPolicy(save_every_epoch=True),
                            batch_probe=BatchProbeSpec(enabled=False),
                            packing_family="live_gpu_toy",
                            packing_signature=f"live-gpu-toy-{index}",
                            packing_eligible=False,
                            packing_backend_allowlist=["exclusive"],
                            max_epochs=LIVE_GPU_EPOCHS,
                            metadata={"live_gpu_fullflow": True, "candidate_index": index},
                        )
                    )

                submitted = api.submit_many(jobs)
                wait_for(lambda: all((api.inspect(job.job_id) or job).status.is_terminal for job in submitted))

                for submitted_job in submitted:
                    final_job = api.inspect(submitted_job.job_id)
                    self.assertIsNotNone(final_job)
                    assert final_job is not None
                    self.assertEqual(final_job.status, JobStatus.COMPLETED)
                    self.assertEqual(final_job.metadata["placement_backend"], "exclusive")
                    self.assertEqual(final_job.metadata["placement_mode"], "exclusive")
                    self.assertEqual(final_job.authored_batch_size, LIVE_GPU_INITIAL_BATCH_SIZE)
                    placed_batch = int(final_job.metadata["placement_batch_size"])
                    self.assertIn(
                        placed_batch,
                        {
                            LIVE_GPU_INITIAL_BATCH_SIZE // 2,
                            LIVE_GPU_INITIAL_BATCH_SIZE,
                            LIVE_GPU_INITIAL_BATCH_SIZE * 2,
                        },
                    )
                    expected_steps = LIVE_GPU_EPOCHS * math.ceil(
                        LIVE_GPU_NUM_SAMPLES / placed_batch
                    )

                    completion_payload = latest_job_completed_payload(api, final_job.job_id)
                    self.assertEqual(completion_payload["device"], "cuda")
                    self.assertGreater(completion_payload["peak_vram_mb"], 0)
                    self.assertIn("NVIDIA", completion_payload["cuda_device_name"])
                    self.assertEqual(completion_payload["global_step"], expected_steps)

                    metric_samples = api.list_job_metric_samples(final_job.job_id)
                    self.assertGreaterEqual(len(metric_samples), expected_steps)

                report = api.report()
                self.assertEqual(report["total_jobs"], 2)
                self.assertEqual(report["completed_jobs"], 2)
                self.assertEqual(report["failed_jobs"], 0)
                self.assertGreaterEqual(len(api.list_events(event_type="job_dispatched")), 2)
            finally:
                service.stop()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the live packed GPU scheduler test")
    def test_scheduler_packs_three_tiny_model_jobs_for_parallel_cuda_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            settings = SchedulerSettings(
                runtime_root=runtime_root,
                scheduler_poll_interval_seconds=0.05,
                gpu_scheduler={
                    "mode": "adaptive",
                    "backend_priority": ["cuda_process", "exclusive"],
                    "max_packed_jobs_per_gpu": 3,
                    "stream": {"enabled": False},
                    "mps": {"enabled": False},
                    "cuda_process": {"enabled": True},
                },
                prediction={"mode": "ml_predictor"},
                log_db={"enabled": False},
                redis_cache={"enabled": False},
            )
            api = SchedulerClient(settings)
            service = api.create_service().start(background=True)
            try:
                baseline = create_toy_baseline_checkpoint(
                    runtime_root / "baselines" / "packed-live-gpu-toy.pt",
                    input_dim=128,
                    hidden_dim=256,
                    output_dim=8,
                    seed=41,
                )
                jobs = [
                    build_mlevolve_job(
                        workflow_id="live-gpu-packed-fullflow",
                        baseline_model_id="packed-live-gpu-toy-baseline",
                        baseline_model_path=baseline,
                        runner_target="localml_scheduler.examples.toy_pytorch_runner:run_toy_training_job",
                        runner_kwargs={
                            "input_dim": 128,
                            "hidden_dim": 256,
                            "output_dim": 8,
                            "num_samples": PACKED_GPU_NUM_SAMPLES,
                            "batch_size": PACKED_GPU_BATCH_SIZE,
                            "epochs": PACKED_GPU_EPOCHS,
                            "learning_rate": 0.01 + (index * 0.005),
                            "optimizer": "sgd",
                        },
                        priority=20 - index,
                        task_type="live_gpu_packed_toy_classification",
                        resource_requirements=ResourceRequirements(
                            requires_gpu=True,
                            estimated_vram_mb=256,
                            estimated_ram_mb=512,
                        ),
                        checkpoint_policy=CheckpointPolicy(save_every_epoch=True),
                        packing_family="packed_live_gpu_toy",
                        packing_signature=f"packed-live-gpu-toy-{index}",
                        packing_eligible=True,
                        packing_backend_allowlist=["cuda_process"],
                        max_epochs=PACKED_GPU_EPOCHS,
                        metadata={"live_gpu_packed_fullflow": True, "candidate_index": index},
                    )
                    for index in range(3)
                ]

                submitted = api.submit_many(jobs)
                wait_for(lambda: all((api.inspect(job.job_id) or job).status.is_terminal for job in submitted), timeout=90.0)

                submitted_ids = {job.job_id for job in submitted}
                packed_events = api.list_events(event_type="packed_group_dispatched")
                self.assertTrue(packed_events)
                packed_payloads = [event["payload"] for event in packed_events]
                self.assertTrue(any(set(payload["job_ids"]) == submitted_ids for payload in packed_payloads))

                for submitted_job in submitted:
                    final_job = api.inspect(submitted_job.job_id)
                    self.assertIsNotNone(final_job)
                    assert final_job is not None
                    self.assertEqual(final_job.status, JobStatus.COMPLETED)
                    self.assertEqual(final_job.metadata["placement_backend"], "cuda_process")
                    self.assertEqual(final_job.metadata["placement_mode"], "packed_group")
                    self.assertIn(final_job.metadata["placement_role"], {"slot-0", "slot-1", "slot-2"})
                    self.assertEqual(final_job.authored_batch_size, PACKED_GPU_BATCH_SIZE)
                    placed_batch = int(final_job.metadata["placement_batch_size"])
                    self.assertIn(placed_batch, {PACKED_GPU_BATCH_SIZE // 2, PACKED_GPU_BATCH_SIZE, PACKED_GPU_BATCH_SIZE * 2})
                    expected_steps = PACKED_GPU_EPOCHS * math.ceil(PACKED_GPU_NUM_SAMPLES / placed_batch)

                    completion_payload = latest_job_completed_payload(api, final_job.job_id)
                    self.assertEqual(completion_payload["device"], "cuda")
                    self.assertGreater(completion_payload["peak_vram_mb"], 0)
                    self.assertIn("NVIDIA", completion_payload["cuda_device_name"])
                    self.assertEqual(completion_payload["global_step"], expected_steps)

                    metric_samples = api.list_job_metric_samples(final_job.job_id)
                    self.assertGreaterEqual(len(metric_samples), expected_steps)

                report = api.report()
                self.assertEqual(report["total_jobs"], 3)
                self.assertEqual(report["completed_jobs"], 3)
                self.assertEqual(report["failed_jobs"], 0)
                self.assertEqual(report["packed_dispatches"], 1)
            finally:
                service.stop()
