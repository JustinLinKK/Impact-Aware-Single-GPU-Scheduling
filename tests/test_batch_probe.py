from __future__ import annotations

import tempfile
import unittest

from localml_scheduler.checkpointing.manager import CheckpointManager
from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import (
    BatchProbeProfile,
    BatchProbeSpec,
    BatchProbeTrialResult,
    ResourceRequirements,
    SchedulingClass,
    TrainingJob,
    build_batch_probe_shape_signature,
)
from localml_scheduler.execution.control import ControlPlane, TrainingControlHook
from localml_scheduler.execution.runner_protocol import RunnerContext
from localml_scheduler.observability.events import EventLogger
from localml_scheduler.profiling.batch_probe import run_batch_probe_preflight
from localml_scheduler.storage.sqlite_store import SQLiteStateStore


def fake_limit_probe(
    context: RunnerContext,
    batch_size: int,
    warmup_steps: int,
    measure_steps: int,
) -> BatchProbeTrialResult:
    threshold = int(context.job.metadata.get("probe_threshold", 5))
    peak_vram_mb = 128 + (batch_size * 64)
    return BatchProbeTrialResult(
        fits=batch_size <= threshold,
        peak_vram_mb=peak_vram_mb,
        avg_vram_mb=peak_vram_mb * 0.8,
        memory_total_mb=2048,
        avg_step_time_ms=1.0 + float(batch_size),
        steps_per_epoch=10,
        seconds_per_epoch=(1.0 + float(batch_size)) / 100.0,
        message=f"batch size {batch_size} {'fits' if batch_size <= threshold else 'does not fit'}",
    )


def _build_context(settings: SchedulerSettings, job: TrainingJob) -> RunnerContext:
    store = SQLiteStateStore(settings)
    store.save_job(job)
    event_logger = EventLogger(store, settings.events_jsonl_path)
    checkpoint_manager = CheckpointManager(settings, store, event_logger)
    control_plane = ControlPlane(settings)
    control_plane.initialize_job(job.job_id)
    return RunnerContext(
        job=job,
        settings=settings,
        store=store,
        event_logger=event_logger,
        control_hook=TrainingControlHook(
            job,
            control_plane,
            checkpoint_manager,
            store,
            event_logger,
        ),
        checkpoint_manager=checkpoint_manager,
        cache_client=None,
    )


class TimeAwareBatchProbeTest(unittest.TestCase):
    def test_exclusive_probe_persists_the_five_time_aware_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(
                runtime_root=tmpdir,
                gpu_scheduler={
                    "mode": "parallel_time_aware",
                    "memory": {"gpu_vram_gib": 2, "predicted_budget_fraction": 0.85},
                },
            )
            job = TrainingJob.create(
                "pkg.runner:train",
                "baseline-five",
                "/tmp/five.pt",
                runner_kwargs={"batch_size": 4, "probe_max_batch_size": 16},
                scheduling_class=SchedulingClass.EXCLUSIVE_PROBE,
                batch_probe=BatchProbeSpec(
                    enabled=True,
                    probe_target="localml_scheduler.tests.test_batch_probe:fake_limit_probe",
                ),
                metadata={"placement_backend": "exclusive", "probe_threshold": 16},
                resource_requirements=ResourceRequirements(requires_gpu=True),
            )
            context = _build_context(settings, job)

            resolved = run_batch_probe_preflight(context)

            self.assertEqual(resolved.requested_batch_size, 4)
            observations = context.store.list_batch_size_observations(
                model_key="baseline-five",
                shape_signature=build_batch_probe_shape_signature(job),
                hardware_key=context.store.hardware_key(),
                backend_name="exclusive",
            )
            self.assertEqual(
                sorted(item.batch_size for item in observations),
                [1, 2, 4, 8, 16],
            )
            self.assertTrue(all(item.metadata.get("seconds_per_epoch") for item in observations))

    def test_normal_jobs_never_run_the_exclusive_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(runtime_root=tmpdir)
            job = TrainingJob.create(
                "pkg.runner:train",
                "normal",
                "/tmp/normal.pt",
                runner_kwargs={"batch_size": 4},
                batch_probe=BatchProbeSpec(
                    enabled=True,
                    probe_target="localml_scheduler.tests.test_batch_probe:fake_limit_probe",
                ),
                metadata={"placement_backend": "exclusive"},
            )
            context = _build_context(settings, job)

            self.assertIs(run_batch_probe_preflight(context), context.job)
            self.assertEqual(context.store.list_batch_size_observations(), [])

    def test_shape_signature_ignores_batch_size_but_tracks_shape(self) -> None:
        def job(batch_size: int, sequence_length: int) -> TrainingJob:
            return TrainingJob.create(
                "pkg.runner:train",
                "baseline-a",
                "/tmp/a.pt",
                runner_kwargs={
                    "batch_size": batch_size,
                    "precision": "bf16",
                    "sequence_length": sequence_length,
                },
            )

        self.assertEqual(
            build_batch_probe_shape_signature(job(4, 128)),
            build_batch_probe_shape_signature(job(8, 128)),
        )
        self.assertNotEqual(
            build_batch_probe_shape_signature(job(4, 128)),
            build_batch_probe_shape_signature(job(4, 256)),
        )

    def test_profile_store_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteStateStore(SchedulerSettings(runtime_root=tmpdir))
            store.upsert_batch_probe_profile(
                BatchProbeProfile(
                    probe_key="probe-1",
                    model_key="baseline-a",
                    device_type="RTX-test",
                    shape_signature="shape-1",
                    batch_param_name="batch_size",
                    resolved_batch_size=4,
                    metadata={"source": "exclusive_five_option_probe"},
                )
            )
            restored = store.get_batch_probe_profile("probe-1")
            self.assertIsNotNone(restored)
            self.assertEqual(restored.resolved_batch_size, 4)

    def test_legacy_search_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "search_mode was removed"):
            BatchProbeSpec.from_dict({"enabled": True, "search_mode": "power_of_two"})


if __name__ == "__main__":
    unittest.main()
