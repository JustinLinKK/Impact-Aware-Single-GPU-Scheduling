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
from localml_scheduler.domain import (
    BatchProbeSpec,
    BatchSizeObservation,
    CheckpointPolicy,
    JobStatus,
    ResourceRequirements,
    RuntimeProfile,
    SchedulingClass,
    build_batch_probe_shape_signature,
    build_batch_size_observation_key,
)
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
    wait_for(
        lambda: bool(api.list_events(job_id=job_id, event_type="job_completed")),
        timeout=10.0,
    )
    events = api.list_events(job_id=job_id, event_type="job_completed")
    return dict(events[-1]["payload"])


class LiveGpuSchedulerFullFlowTest(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is required for the live GPU scheduler full-flow test",
    )
    def test_scheduler_runs_real_toy_model_jobs_on_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            settings = SchedulerSettings(
                runtime_root=runtime_root,
                scheduler_poll_interval_seconds=0.05,
                gpu_scheduler={
                    "mode": "parallel_time_aware",
                    "backend_priority": ["exclusive"],
                    "profiling": {"warmup_steps": 1, "solo_probe_steps": 1},
                    "stream": {"enabled": False},
                    "cuda_process": {"enabled": False},
                    "mps": {"enabled": False},
                },
                log_db={"enabled": False},
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
                            batch_probe=BatchProbeSpec(
                                enabled=True,
                                probe_target="localml_scheduler.examples.toy_pytorch_runner:probe_toy_training_batch_size",
                                batch_param_name="batch_size",
                                model_key=f"live-gpu-toy-{index}",
                            ),
                            packing_family="live_gpu_toy",
                            packing_signature=f"live-gpu-toy-{index}",
                            packing_eligible=False,
                            packing_backend_allowlist=["exclusive"],
                            max_epochs=LIVE_GPU_EPOCHS,
                            metadata={
                                "live_gpu_fullflow": True,
                                "candidate_index": index,
                            },
                    )
                    )

                for job in jobs:
                    job.scheduling_class = SchedulingClass.EXCLUSIVE_PROBE

                submitted = [api.submit(job) for job in jobs]
                wait_for(
                    lambda: all(
                        (api.inspect(job.job_id) or job).status.is_terminal
                        for job in submitted
                    )
                )

                for submitted_job in submitted:
                    final_job = api.inspect(submitted_job.job_id)
                    self.assertIsNotNone(final_job)
                    assert final_job is not None
                    self.assertEqual(final_job.status, JobStatus.COMPLETED)
                    self.assertEqual(
                        final_job.metadata["placement_backend"], "exclusive"
                    )
                    self.assertEqual(final_job.metadata["placement_mode"], "exclusive")
                    self.assertEqual(
                        final_job.metadata["batch_probe_source"],
                        "exclusive_five_option_probe",
                    )
                    self.assertIn("batch_probe_key", final_job.metadata)

                    resolved_batch_size = int(final_job.metadata["resolved_batch_size"])
                    self.assertGreaterEqual(
                        resolved_batch_size, LIVE_GPU_INITIAL_BATCH_SIZE
                    )
                    expected_steps = LIVE_GPU_EPOCHS * math.ceil(
                        LIVE_GPU_NUM_SAMPLES / resolved_batch_size
                    )

                    profile = api.get_batch_probe_profile(
                        str(final_job.metadata["batch_probe_key"])
                    )
                    self.assertIsNotNone(profile)
                    assert profile is not None
                    self.assertNotEqual(profile.device_type, "cuda-unavailable")
                    self.assertGreater(profile.memory_total_mb or 0, 0)
                    self.assertGreater(profile.peak_vram_mb or 0, 0)

                    completion_payload = latest_job_completed_payload(
                        api, final_job.job_id
                    )
                    self.assertEqual(completion_payload["device"], "cuda")
                    self.assertGreater(completion_payload["peak_vram_mb"], 0)
                    self.assertIn("NVIDIA", completion_payload["cuda_device_name"])
                    self.assertEqual(completion_payload["global_step"], expected_steps)

                    self.assertEqual(
                        final_job.metadata["last_completed_epoch"], LIVE_GPU_EPOCHS
                    )

                report = api.report()
                self.assertEqual(report["total_jobs"], 2)
                self.assertEqual(report["completed_jobs"], 2)
                self.assertEqual(report["failed_jobs"], 0)
                self.assertGreaterEqual(
                    len(api.list_events(event_type="job_dispatched")), 2
                )
                self.assertGreaterEqual(
                    len(
                        api.list_events(
                            event_type="exclusive_probe_measurements_persisted"
                        )
                    ),
                    2,
                )
            finally:
                service.stop()

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is required for the live packed GPU scheduler test",
    )
    def test_scheduler_runs_profiled_time_aware_anchor_on_cuda_process(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            settings = SchedulerSettings(
                runtime_root=runtime_root,
                scheduler_poll_interval_seconds=0.05,
                gpu_scheduler={
                    "mode": "parallel_time_aware",
                    "backend_priority": ["cuda_process", "exclusive"],
                    "parallel_job_cap": 3,
                    "stream": {"enabled": False},
                    "mps": {"enabled": False},
                    "cuda_process": {"enabled": True},
                },
                log_db={"enabled": False},
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
                job = build_mlevolve_job(
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
                            "learning_rate": 0.01,
                            "optimizer": "sgd",
                        },
                        priority=20,
                        task_type="live_gpu_packed_toy_classification",
                        resource_requirements=ResourceRequirements(
                            requires_gpu=True,
                            estimated_vram_mb=256,
                            estimated_ram_mb=512,
                        ),
                        checkpoint_policy=CheckpointPolicy(save_every_epoch=True),
                        packing_family="packed_live_gpu_toy",
                        packing_signature="packed-live-gpu-toy-anchor",
                        packing_eligible=True,
                        packing_backend_allowlist=["cuda_process"],
                        max_epochs=PACKED_GPU_EPOCHS,
                        metadata={
                            "live_gpu_packed_fullflow": True,
                            "candidate_index": 0,
                        },
                    )
                hardware_key = api.store.hardware_key()
                shape_signature = build_batch_probe_shape_signature(job)
                for backend in ("exclusive", "cuda_process"):
                    for batch_size in (4, 8, 16, 32, 64):
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
                                metadata={"estimate_source": "live_gpu_fixture"},
                            )
                        )
                        api.store.upsert_runtime_profile(
                            RuntimeProfile.create(
                                signature=job.packing.signature or job.job_id,
                                hardware_key=hardware_key,
                                backend_name=backend,
                                resolved_batch_size=batch_size,
                                strategy="epoch_1",
                                epoch_1_seconds=2.0 / batch_size,
                                estimated_total_runtime_seconds=1.0,
                                confidence=0.9,
                                observations=1,
                                source="live_gpu_fixture",
                            )
                        )

                submitted = [api.submit(job)]
                wait_for(
                    lambda: all(
                        (api.inspect(job.job_id) or job).status.is_terminal
                        for job in submitted
                    ),
                    timeout=90.0,
                )

                for submitted_job in submitted:
                    final_job = api.inspect(submitted_job.job_id)
                    self.assertIsNotNone(final_job)
                    assert final_job is not None
                    self.assertEqual(final_job.status, JobStatus.COMPLETED)
                    self.assertEqual(
                        final_job.metadata["placement_backend"], "cuda_process"
                    )
                    self.assertEqual(
                        final_job.metadata["placement_mode"], "stack_anchor"
                    )
                    self.assertEqual(final_job.metadata["placement_role"], "solo")

                    resolved_batch_size = int(
                        final_job.metadata["placement_batch_size"]
                    )
                    expected_steps = PACKED_GPU_EPOCHS * math.ceil(
                        PACKED_GPU_NUM_SAMPLES / resolved_batch_size
                    )

                    completion_payload = latest_job_completed_payload(
                        api, final_job.job_id
                    )
                    self.assertEqual(completion_payload["device"], "cuda")
                    self.assertGreater(completion_payload["peak_vram_mb"], 0)
                    self.assertIn("NVIDIA", completion_payload["cuda_device_name"])
                    self.assertEqual(completion_payload["global_step"], expected_steps)

                    self.assertEqual(
                        final_job.metadata["last_completed_epoch"], PACKED_GPU_EPOCHS
                    )

                report = api.report()
                self.assertEqual(report["total_jobs"], 1)
                self.assertEqual(report["completed_jobs"], 1)
                self.assertEqual(report["failed_jobs"], 0)
            finally:
                service.stop()
