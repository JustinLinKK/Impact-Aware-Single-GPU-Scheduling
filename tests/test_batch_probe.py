from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import time
import unittest

from localml_scheduler.adapters.mlevolve import build_mlevolve_job
from localml_scheduler.api import LocalMLSchedulerAPI
from localml_scheduler.checkpointing.manager import CheckpointManager
from localml_scheduler.execution.backends import ExclusiveBackend, MPSBackend
from localml_scheduler.execution.control import ControlPlane, TrainingControlHook
from localml_scheduler.execution.executor import SubprocessExecutor
from localml_scheduler.execution.runner_protocol import RunnerContext
from localml_scheduler.examples.toy_pytorch_runner import create_toy_baseline_checkpoint
from localml_scheduler.observability.events import EventLogger
from localml_scheduler.profiling.batch_probe import BatchProbeKeyInfo, _run_probe_controller
from localml_scheduler.schemas import (
    BatchProbeProfile,
    BatchProbeSpec,
    BatchProbeTrialResult,
    CheckpointPolicy,
    ResourceRequirements,
    SafePointType,
    SoloProfile,
    TrainingJob,
    build_batch_probe_shape_signature,
)
from localml_scheduler.scheduler.supervisor import WorkerSupervisor
from localml_scheduler.settings import SchedulerSettings
from localml_scheduler.storage.sqlite_store import SQLiteStateStore


def wait_for(predicate, timeout: float = 30.0, interval: float = 0.1) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise TimeoutError("condition not met in time")


def fake_limit_probe(context: RunnerContext, batch_size: int, warmup_steps: int, measure_steps: int) -> BatchProbeTrialResult:
    threshold = int(context.job.metadata.get("probe_threshold", 5))
    peak_vram_mb = 128 + (batch_size * 64)
    if batch_size > threshold:
        return BatchProbeTrialResult(
            fits=False,
            peak_vram_mb=peak_vram_mb,
            memory_total_mb=1024,
            avg_step_time_ms=2.0,
            message=f"batch size {batch_size} is above threshold {threshold}",
        )
    return BatchProbeTrialResult(
        fits=True,
        peak_vram_mb=peak_vram_mb,
        memory_total_mb=1024,
        avg_step_time_ms=1.0 + float(batch_size),
        message=f"batch size {batch_size} fits",
    )


def _build_context(settings: SchedulerSettings, job: TrainingJob) -> RunnerContext:
    store = SQLiteStateStore(settings)
    store.save_job(job)
    event_logger = EventLogger(store, settings.events_jsonl_path)
    checkpoint_manager = CheckpointManager(settings, store, event_logger)
    control_plane = ControlPlane(settings)
    control_plane.initialize_job(job.job_id)
    control_hook = TrainingControlHook(job, control_plane, checkpoint_manager, store, event_logger)
    return RunnerContext(
        job=job,
        settings=settings,
        store=store,
        event_logger=event_logger,
        control_hook=control_hook,
        checkpoint_manager=checkpoint_manager,
        cache_client=None,
    )


def _build_supervisor(settings: SchedulerSettings, *, mps_available: bool) -> WorkerSupervisor:
    executor = SubprocessExecutor(settings)
    mps_binary = shutil.which("true") if mps_available else None
    backends = {
        "exclusive": ExclusiveBackend(settings, executor),
        "mps": MPSBackend(settings, executor, mps_binary=mps_binary),
    }
    return WorkerSupervisor(settings, backends=backends)


def _seed_solo_profile(api: LocalMLSchedulerAPI, job: TrainingJob) -> None:
    api.upsert_solo_profile(
        SoloProfile(
            signature=job.packing.signature,
            family=job.packing.family,
            peak_vram_mb=512,
            avg_gpu_utilization=0.2,
            avg_memory_utilization=0.2,
            sample_count=3,
            last_job_id=job.job_id,
            metadata={"seeded": True},
        )
    )


class BatchProbeUnitTest(unittest.TestCase):
    def test_shape_signature_ignores_batch_size_but_changes_with_shape(self) -> None:
        job_a = TrainingJob.create(
            "pkg.runner:train",
            "baseline-a",
            "/tmp/a.pt",
            task_type="classification",
            runner_kwargs={"batch_size": 4, "precision": "bf16", "sequence_length": 128},
            batch_probe=BatchProbeSpec(enabled=True, probe_target="pkg.runner:probe"),
        )
        job_b = TrainingJob.create(
            "pkg.runner:train",
            "baseline-a",
            "/tmp/a.pt",
            task_type="classification",
            runner_kwargs={"batch_size": 8, "precision": "bf16", "sequence_length": 128},
            batch_probe=BatchProbeSpec(enabled=True, probe_target="pkg.runner:probe"),
        )
        job_c = TrainingJob.create(
            "pkg.runner:train",
            "baseline-a",
            "/tmp/a.pt",
            task_type="classification",
            runner_kwargs={"batch_size": 8, "precision": "bf16", "sequence_length": 256},
            batch_probe=BatchProbeSpec(enabled=True, probe_target="pkg.runner:probe"),
        )
        self.assertEqual(build_batch_probe_shape_signature(job_a), build_batch_probe_shape_signature(job_b))
        self.assertNotEqual(build_batch_probe_shape_signature(job_a), build_batch_probe_shape_signature(job_c))

    def test_batch_probe_store_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteStateStore(SchedulerSettings(runtime_root=tmpdir))
            profile = BatchProbeProfile(
                probe_key="probe-1",
                model_key="baseline-a",
                device_type="RTX-test",
                shape_signature="shape-1",
                batch_param_name="batch_size",
                resolved_batch_size=6,
                peak_vram_mb=1536,
                memory_total_mb=2048,
                target_budget_mb=1986,
                metadata={"source": "test"},
            )
            store.upsert_batch_probe_profile(profile)

            restored = store.get_batch_probe_profile("probe-1")
            self.assertIsNotNone(restored)
            self.assertEqual(restored.resolved_batch_size, 6)
            self.assertEqual(len(store.list_batch_probe_profiles()), 1)

    def test_probe_controller_selects_largest_safe_batch_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(runtime_root=tmpdir)
            job = TrainingJob.create(
                "pkg.runner:train",
                "baseline-a",
                "/tmp/a.pt",
                task_type="classification",
                runner_kwargs={"batch_size": 3},
                batch_probe=BatchProbeSpec(
                    enabled=True,
                    probe_target="localml_scheduler.tests.test_batch_probe:fake_limit_probe",
                ),
                metadata={"placement_backend": "exclusive", "probe_threshold": 5},
                resource_requirements=ResourceRequirements(requires_gpu=True),
                checkpoint_policy=CheckpointPolicy(save_every_n_steps=1, pause_mode=SafePointType.STEP),
            )
            context = _build_context(settings, job)
            profile = _run_probe_controller(
                context,
                key_info=BatchProbeKeyInfo(
                    probe_key="probe-1",
                    model_key="baseline-a",
                    device_type="cuda-unavailable",
                    shape_signature="shape-1",
                ),
            )
            self.assertEqual(profile.resolved_batch_size, 5)
            self.assertEqual(profile.batch_param_name, "batch_size")
            self.assertGreater(profile.target_budget_mb, 0)


class BatchProbeIntegrationTest(unittest.TestCase):
    def _build_probe_job(
        self,
        baseline: str,
        *,
        batch_size: int,
        learning_rate: float = 0.01,
        max_steps: int = 4,
        epochs: int = 1,
        sleep_per_step: float = 0.0,
        shape_hints: dict[str, int] | None = None,
        probe_max_batch_size: int | None = 6,
        packing_eligible: bool = False,
        priority: int = 5,
    ) -> TrainingJob:
        runner_kwargs = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "num_samples": 64,
            "sleep_per_step": sleep_per_step,
            "probe_memory_total_mb": 2048,
            "probe_base_memory_mb": 256,
            "probe_memory_per_sample_mb": 256,
        }
        if probe_max_batch_size is not None:
            runner_kwargs["probe_max_batch_size"] = probe_max_batch_size
        return build_mlevolve_job(
            workflow_id="wf-probe",
            baseline_model_id="toy-baseline",
            baseline_model_path=baseline,
            runner_target="localml_scheduler.examples.toy_pytorch_runner:run_toy_training_job",
            runner_kwargs=runner_kwargs,
            priority=priority,
            task_type="toy_classification",
            checkpoint_policy=CheckpointPolicy(save_every_n_steps=1, save_every_epoch=True),
            batch_probe=BatchProbeSpec(
                enabled=True,
                probe_target="localml_scheduler.examples.toy_pytorch_runner:probe_toy_training_batch_size",
                shape_hints=shape_hints or {},
            ),
            resource_requirements=ResourceRequirements(requires_gpu=True, estimated_vram_mb=1024, estimated_ram_mb=512),
            packing_family="toy-mlp",
            packing_eligible=packing_eligible,
            max_steps=max_steps,
            max_epochs=1,
        )

    def test_first_job_probes_and_second_job_reuses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(runtime_root=Path(tmpdir), scheduler_poll_interval_seconds=0.05)
            api = LocalMLSchedulerAPI(settings)
            service = api.create_scheduler_service(supervisor=_build_supervisor(settings, mps_available=False)).start(background=True)
            try:
                baseline = create_toy_baseline_checkpoint(Path(tmpdir) / "baselines" / "probe.pt", seed=100)
                first = self._build_probe_job(baseline, batch_size=3)
                second = self._build_probe_job(baseline, batch_size=1)

                api.submit_job(first)
                wait_for(lambda: api.get_job(first.job_id).status.is_terminal, timeout=30.0)
                first_state = api.get_job(first.job_id)
                self.assertEqual(first_state.config.runner_kwargs["batch_size"], 6)
                self.assertEqual(first_state.metadata["batch_probe_source"], "probe")
                self.assertEqual(len(api.store.list_batch_probe_profiles()), 1)

                api.submit_job(second)
                wait_for(lambda: api.get_job(second.job_id).status.is_terminal, timeout=30.0)
                second_state = api.get_job(second.job_id)
                self.assertEqual(second_state.config.runner_kwargs["batch_size"], 6)
                self.assertEqual(second_state.metadata["batch_probe_source"], "cache")
                self.assertEqual(len(api.store.list_events(job_id=second.job_id, event_type="batch_probe_cache_hit")), 1)
            finally:
                service.stop()

    def test_shape_change_creates_a_new_probe_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(runtime_root=Path(tmpdir), scheduler_poll_interval_seconds=0.05)
            api = LocalMLSchedulerAPI(settings)
            service = api.create_scheduler_service(supervisor=_build_supervisor(settings, mps_available=False)).start(background=True)
            try:
                baseline = create_toy_baseline_checkpoint(Path(tmpdir) / "baselines" / "shape.pt", seed=101)
                first = self._build_probe_job(baseline, batch_size=3, shape_hints={"tokens": 128})
                second = self._build_probe_job(baseline, batch_size=3, shape_hints={"tokens": 256})

                api.submit_job(first)
                wait_for(lambda: api.get_job(first.job_id).status.is_terminal, timeout=30.0)
                api.submit_job(second)
                wait_for(lambda: api.get_job(second.job_id).status.is_terminal, timeout=30.0)

                self.assertEqual(len(api.store.list_batch_probe_profiles()), 2)
                self.assertEqual(len(api.store.list_events(event_type="batch_probe_cache_miss")), 2)
            finally:
                service.stop()

    def test_resume_does_not_reprobe_once_batch_size_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(runtime_root=Path(tmpdir), scheduler_poll_interval_seconds=0.05)
            api = LocalMLSchedulerAPI(settings)
            service = api.create_scheduler_service(supervisor=_build_supervisor(settings, mps_available=False)).start(background=True)
            try:
                baseline = create_toy_baseline_checkpoint(Path(tmpdir) / "baselines" / "resume.pt", seed=102)
                job = self._build_probe_job(baseline, batch_size=3, max_steps=80, epochs=8, sleep_per_step=0.02)
                api.submit_job(job)

                wait_for(lambda: api.get_job(job.job_id).status.name == "RUNNING", timeout=30.0)
                api.pause_job(job.job_id)
                wait_for(lambda: api.get_job(job.job_id).status.name == "PAUSED", timeout=30.0)
                api.resume_job(job.job_id)
                wait_for(lambda: api.get_job(job.job_id).status.is_terminal, timeout=30.0)

                self.assertEqual(len(api.store.list_events(job_id=job.job_id, event_type="batch_probe_started")), 1)
            finally:
                service.stop()

    def test_mps_packed_jobs_skip_batch_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(runtime_root=Path(tmpdir), scheduler_poll_interval_seconds=0.05)
            api = LocalMLSchedulerAPI(settings)
            service = api.create_scheduler_service(supervisor=_build_supervisor(settings, mps_available=True)).start(background=True)
            try:
                baseline = create_toy_baseline_checkpoint(Path(tmpdir) / "baselines" / "mps.pt", seed=103)
                first = self._build_probe_job(baseline, batch_size=3, packing_eligible=True, priority=8)
                second = self._build_probe_job(baseline, batch_size=4, learning_rate=0.02, packing_eligible=True, priority=7)
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
                self.assertEqual(api.store.list_batch_probe_profiles(), [])
                self.assertEqual(len(api.store.list_events(event_type="batch_probe_started")), 0)
            finally:
                service.stop()

    def test_probe_failure_marks_job_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(runtime_root=Path(tmpdir), scheduler_poll_interval_seconds=0.05)
            api = LocalMLSchedulerAPI(settings)
            service = api.create_scheduler_service(supervisor=_build_supervisor(settings, mps_available=False)).start(background=True)
            try:
                baseline = create_toy_baseline_checkpoint(Path(tmpdir) / "baselines" / "fail.pt", seed=104)
                job = self._build_probe_job(baseline, batch_size=1, probe_max_batch_size=0)
                api.submit_job(job)
                wait_for(lambda: api.get_job(job.job_id).status.is_terminal, timeout=30.0)

                final = api.get_job(job.job_id)
                self.assertEqual(final.status.name, "FAILED")
                self.assertIn("feasible batch size", final.status_reason or "")
            finally:
                service.stop()


if __name__ == "__main__":
    unittest.main()
