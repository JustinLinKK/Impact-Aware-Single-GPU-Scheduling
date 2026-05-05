from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import time
import unittest

from localml_scheduler.adapters.mlevolve import build_mlevolve_job
from localml_scheduler.api import LocalMLSchedulerAPI
from localml_scheduler.execution.backends import CudaProcessBackend, ExclusiveBackend, MPSBackend
from localml_scheduler.execution.executor import SubprocessExecutor
from localml_scheduler.examples.toy_pytorch_runner import create_toy_baseline_checkpoint
from localml_scheduler.execution.runner_protocol import RunnerContext
from localml_scheduler.schemas import CheckpointPolicy, JobStatus, ResourceRequirements, SoloProfile
from localml_scheduler.scheduler.supervisor import WorkerSupervisor
from localml_scheduler.settings import SchedulerSettings


def wait_for(predicate, timeout: float = 30.0, interval: float = 0.1) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise TimeoutError("condition not met in time")


def run_failing_job(context: RunnerContext) -> dict[str, object]:
    raise RuntimeError(f"forced failure for {context.job.job_id}")


def _build_supervisor(settings: SchedulerSettings, *, mps_available: bool) -> WorkerSupervisor:
    executor = SubprocessExecutor(settings)
    mps_binary = shutil.which("true") if mps_available else None
    backends = {
        "exclusive": ExclusiveBackend(settings, executor),
        "mps": MPSBackend(settings, executor, mps_binary=mps_binary),
    }
    return WorkerSupervisor(settings, backends=backends)


def _build_cuda_process_supervisor(settings: SchedulerSettings) -> WorkerSupervisor:
    executor = SubprocessExecutor(settings)
    backends = {
        "exclusive": ExclusiveBackend(settings, executor),
        "cuda_process": CudaProcessBackend(settings, executor),
    }
    return WorkerSupervisor(settings, backends=backends)


def _seed_solo_profile(api: LocalMLSchedulerAPI, job, *, avg_gpu_utilization: float = 0.2, peak_vram_mb: int = 512) -> None:
    api.upsert_solo_profile(
        SoloProfile(
            signature=job.packing.signature,
            family=job.packing.family,
            peak_vram_mb=peak_vram_mb,
            avg_gpu_utilization=avg_gpu_utilization,
            avg_memory_utilization=0.2,
            sample_count=3,
            last_job_id=job.job_id,
            metadata={"seeded": True},
        )
    )


class GpuSchedulerIntegrationTest(unittest.TestCase):
    def _build_job(self, baseline: str, *, learning_rate: float, max_steps: int = 12, priority: int = 5, runner_target: str = "localml_scheduler.examples.toy_pytorch_runner:run_toy_training_job"):
        return build_mlevolve_job(
            workflow_id="wf-1",
            baseline_model_id=f"baseline-{learning_rate}",
            baseline_model_path=baseline,
            runner_target=runner_target,
            runner_kwargs={"sleep_per_step": 0.02, "learning_rate": learning_rate, "batch_size": 8},
            priority=priority,
            task_type="toy_classification",
            resource_requirements=ResourceRequirements(requires_gpu=False, estimated_vram_mb=512, estimated_ram_mb=512),
            checkpoint_policy=CheckpointPolicy(save_every_n_steps=1, save_every_epoch=True),
            packing_family="toy-mlp",
            packing_eligible=True,
            max_steps=max_steps,
            max_epochs=2,
        )

    def test_packed_pair_dispatches_when_fake_mps_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            settings = SchedulerSettings(runtime_root=runtime_root, scheduler_poll_interval_seconds=0.05)
            api = LocalMLSchedulerAPI(settings)
            supervisor = _build_supervisor(settings, mps_available=True)
            service = api.create_scheduler_service(supervisor=supervisor).start(background=True)
            try:
                baseline = create_toy_baseline_checkpoint(runtime_root / "baselines" / "packed.pt", seed=90)
                first = self._build_job(baseline, learning_rate=0.01, priority=7)
                second = self._build_job(baseline, learning_rate=0.02, priority=6)
                _seed_solo_profile(api, first)
                _seed_solo_profile(api, second)

                api.submit_job(first)
                api.submit_job(second)

                wait_for(
                    lambda: api.get_job(first.job_id).status.is_terminal
                    and api.get_job(second.job_id).status.is_terminal
                    and api.get_pair_profile(first.packing.signature, second.packing.signature) is not None,
                    timeout=30.0,
                )
                first_state = api.get_job(first.job_id)
                second_state = api.get_job(second.job_id)
                self.assertEqual(first_state.metadata["placement_mode"], "packed_pair")
                self.assertEqual(second_state.metadata["placement_mode"], "packed_pair")
                self.assertEqual(api.report()["packed_dispatches"], 1)
                pair_profile = api.get_pair_profile(first.packing.signature, second.packing.signature)
                self.assertIsNotNone(pair_profile)
                self.assertTrue(pair_profile.compatible)
            finally:
                service.stop()

    def test_cuda_process_backend_dispatches_packed_pair_when_mps_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            settings = SchedulerSettings(
                runtime_root=runtime_root,
                scheduler_poll_interval_seconds=0.05,
                gpu_scheduler={"backend_priority": ["cuda_process", "exclusive"]},
            )
            api = LocalMLSchedulerAPI(settings)
            supervisor = _build_cuda_process_supervisor(settings)
            service = api.create_scheduler_service(supervisor=supervisor).start(background=True)
            try:
                baseline = create_toy_baseline_checkpoint(runtime_root / "baselines" / "cuda-process-pair.pt", seed=89)
                first = self._build_job(baseline, learning_rate=0.011, priority=7)
                second = self._build_job(baseline, learning_rate=0.012, priority=6)
                _seed_solo_profile(api, first)
                _seed_solo_profile(api, second)

                api.submit_job(first)
                api.submit_job(second)

                wait_for(
                    lambda: api.get_job(first.job_id).status.is_terminal and api.get_job(second.job_id).status.is_terminal,
                    timeout=30.0,
                )
                self.assertEqual(api.get_job(first.job_id).metadata["placement_mode"], "packed_pair")
                self.assertEqual(api.get_job(second.job_id).metadata["placement_mode"], "packed_pair")
                self.assertEqual(api.get_job(first.job_id).metadata["placement_backend"], "cuda_process")
                self.assertEqual(api.get_job(second.job_id).metadata["placement_backend"], "cuda_process")
                self.assertEqual(api.report()["packed_dispatches"], 1)
            finally:
                service.stop()

    def test_cuda_process_backend_dispatches_three_way_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            settings = SchedulerSettings(
                runtime_root=runtime_root,
                scheduler_poll_interval_seconds=0.05,
                gpu_scheduler={
                    "backend_priority": ["cuda_process", "exclusive"],
                    "max_packed_jobs_per_gpu": 3,
                    "allow_three_way_packing": True,
                },
            )
            api = LocalMLSchedulerAPI(settings)
            supervisor = _build_cuda_process_supervisor(settings)
            service = api.create_scheduler_service(supervisor=supervisor).start(background=True)
            try:
                baseline = create_toy_baseline_checkpoint(runtime_root / "baselines" / "cuda-process-group.pt", seed=88)
                jobs = [
                    self._build_job(baseline, learning_rate=0.021, priority=9),
                    self._build_job(baseline, learning_rate=0.022, priority=8),
                    self._build_job(baseline, learning_rate=0.023, priority=7),
                ]
                for job in jobs:
                    _seed_solo_profile(api, job, peak_vram_mb=256)
                    api.submit_job(job)

                wait_for(lambda: all(api.get_job(job.job_id).status.is_terminal for job in jobs), timeout=30.0)
                self.assertTrue(all(api.get_job(job.job_id).metadata["placement_mode"] == "packed_group" for job in jobs))
                self.assertTrue(all(api.get_job(job.job_id).metadata["placement_backend"] == "cuda_process" for job in jobs))
                self.assertEqual(api.report()["packed_dispatches"], 1)
            finally:
                service.stop()

    def test_cuda_process_failure_records_fallback_and_keeps_peer_alive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            settings = SchedulerSettings(
                runtime_root=runtime_root,
                scheduler_poll_interval_seconds=0.05,
                gpu_scheduler={"backend_priority": ["cuda_process", "exclusive"]},
            )
            api = LocalMLSchedulerAPI(settings)
            supervisor = _build_cuda_process_supervisor(settings)
            service = api.create_scheduler_service(supervisor=supervisor).start(background=True)
            try:
                baseline = create_toy_baseline_checkpoint(runtime_root / "baselines" / "cuda-process-failure.pt", seed=87)
                primary = self._build_job(baseline, learning_rate=0.031, priority=9, max_steps=30)
                failing = self._build_job(
                    baseline,
                    learning_rate=0.032,
                    priority=8,
                    max_steps=5,
                    runner_target="localml_scheduler.tests.test_gpu_scheduler_integration:run_failing_job",
                )
                _seed_solo_profile(api, primary)
                _seed_solo_profile(api, failing)

                api.submit_job(primary)
                api.submit_job(failing)

                wait_for(
                    lambda: api.get_job(primary.job_id).status.is_terminal and api.get_job(failing.job_id).status.is_terminal,
                    timeout=30.0,
                )
                self.assertEqual(api.get_job(primary.job_id).status, JobStatus.COMPLETED)
                self.assertEqual(api.get_job(failing.job_id).status, JobStatus.FAILED)
                self.assertEqual(api.report()["packed_fallbacks"], 1)
                self.assertEqual(api.get_job(primary.job_id).metadata["placement_backend"], "cuda_process")
                self.assertEqual(api.get_job(failing.job_id).metadata["placement_backend"], "cuda_process")
            finally:
                service.stop()

    def test_jobs_fall_back_to_exclusive_when_mps_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            settings = SchedulerSettings(runtime_root=runtime_root, scheduler_poll_interval_seconds=0.05)
            api = LocalMLSchedulerAPI(settings)
            supervisor = _build_supervisor(settings, mps_available=False)
            service = api.create_scheduler_service(supervisor=supervisor).start(background=True)
            try:
                baseline = create_toy_baseline_checkpoint(runtime_root / "baselines" / "exclusive.pt", seed=91)
                first = self._build_job(baseline, learning_rate=0.03, priority=7, max_steps=20)
                second = self._build_job(baseline, learning_rate=0.04, priority=6, max_steps=20)
                _seed_solo_profile(api, first)
                _seed_solo_profile(api, second)

                api.submit_job(first)
                api.submit_job(second)

                wait_for(lambda: api.get_job(first.job_id).status == JobStatus.RUNNING)
                time.sleep(0.3)
                self.assertIn(api.get_job(second.job_id).status, {JobStatus.PENDING, JobStatus.READY})

                wait_for(lambda: api.get_job(first.job_id).status.is_terminal and api.get_job(second.job_id).status.is_terminal, timeout=30.0)
                self.assertEqual(api.get_job(first.job_id).metadata["placement_mode"], "exclusive")
                self.assertEqual(api.get_job(second.job_id).metadata["placement_mode"], "exclusive")
                self.assertEqual(api.report()["packed_dispatches"], 0)
            finally:
                service.stop()

    def test_secondary_failure_marks_pair_incompatible_and_keeps_primary_alive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            settings = SchedulerSettings(runtime_root=runtime_root, scheduler_poll_interval_seconds=0.05)
            api = LocalMLSchedulerAPI(settings)
            supervisor = _build_supervisor(settings, mps_available=True)
            service = api.create_scheduler_service(supervisor=supervisor).start(background=True)
            try:
                baseline = create_toy_baseline_checkpoint(runtime_root / "baselines" / "failure.pt", seed=92)
                primary = self._build_job(baseline, learning_rate=0.05, priority=9, max_steps=30)
                failing = self._build_job(
                    baseline,
                    learning_rate=0.06,
                    priority=8,
                    max_steps=5,
                    runner_target="localml_scheduler.tests.test_gpu_scheduler_integration:run_failing_job",
                )
                _seed_solo_profile(api, primary)
                _seed_solo_profile(api, failing)

                api.submit_job(primary)
                api.submit_job(failing)

                wait_for(
                    lambda: api.get_job(primary.job_id).status.is_terminal
                    and api.get_job(failing.job_id).status.is_terminal
                    and api.get_pair_profile(primary.packing.signature, failing.packing.signature) is not None,
                    timeout=30.0,
                )
                self.assertEqual(api.get_job(primary.job_id).status, JobStatus.COMPLETED)
                self.assertEqual(api.get_job(failing.job_id).status, JobStatus.FAILED)
                pair_profile = api.get_pair_profile(primary.packing.signature, failing.packing.signature)
                self.assertIsNotNone(pair_profile)
                self.assertFalse(pair_profile.compatible)
                self.assertEqual(api.report()["packed_fallbacks"], 1)
            finally:
                service.stop()

    def test_manual_pause_and_cancel_work_per_job_in_packed_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            settings = SchedulerSettings(runtime_root=runtime_root, scheduler_poll_interval_seconds=0.05)
            api = LocalMLSchedulerAPI(settings)
            supervisor = _build_supervisor(settings, mps_available=True)
            service = api.create_scheduler_service(supervisor=supervisor).start(background=True)
            try:
                baseline = create_toy_baseline_checkpoint(runtime_root / "baselines" / "control.pt", seed=93)
                primary = self._build_job(baseline, learning_rate=0.07, priority=9, max_steps=80)
                secondary = self._build_job(baseline, learning_rate=0.08, priority=8, max_steps=80)
                _seed_solo_profile(api, primary)
                _seed_solo_profile(api, secondary)

                api.submit_job(primary)
                api.submit_job(secondary)

                wait_for(lambda: api.get_job(primary.job_id).status == JobStatus.RUNNING and api.get_job(secondary.job_id).status == JobStatus.RUNNING)
                api.pause_job(secondary.job_id)
                wait_for(lambda: api.get_job(secondary.job_id).status == JobStatus.PAUSED, timeout=20.0)

                primary_after_pause = api.get_job(primary.job_id).status
                self.assertIn(primary_after_pause, {JobStatus.RUNNING, JobStatus.COMPLETED})
                if primary_after_pause == JobStatus.RUNNING:
                    api.cancel_job(primary.job_id)
                    wait_for(lambda: api.get_job(primary.job_id).status == JobStatus.CANCELLED, timeout=20.0)
            finally:
                service.stop()


if __name__ == "__main__":
    unittest.main()
