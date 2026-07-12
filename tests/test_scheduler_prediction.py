from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import BatchProbeProfile, ResourceRequirements, TrainingJob
from localml_scheduler.hardware import HardwareProfile, build_hardware_key
from localml_scheduler.scheduler.resource_estimator import ResourceEstimator
from localml_scheduler.storage.sqlite_store import SQLiteStateStore


def _fake_hardware_profile(name: str = "prediction-gpu") -> HardwareProfile:
    return HardwareProfile(
        hardware_key=build_hardware_key(
            os_name="linux",
            gpu_name=name,
            total_vram_mb=24576,
            compute_capability="9.0",
            cuda_runtime="12.8",
            torch_version="2.8.0",
        ),
        os_name="linux",
        gpu_name=name,
        total_vram_mb=24576,
        compute_capability="9.0",
        cuda_runtime="12.8",
        torch_version="2.8.0",
    )


class _NoLegacyProfileRepo:
    def __init__(self) -> None:
        self._hardware = _fake_hardware_profile()

    def hardware_profile(self) -> HardwareProfile:
        return self._hardware

    def hardware_key(self) -> str:
        return self._hardware.hardware_key

    def get_batch_probe_profile(self, probe_key: str) -> BatchProbeProfile | None:
        raise AssertionError(f"legacy batch probe lookup should not run for explicit estimates: {probe_key}")

    def get_batch_size_observation(self, **kwargs):
        raise AssertionError(f"legacy batch observation lookup should not run for explicit estimates: {kwargs}")

    def list_batch_size_observations(self, **kwargs):
        raise AssertionError(f"legacy batch observation scan should not run for explicit estimates: {kwargs}")


class SchedulerPredictionTest(unittest.TestCase):
    def test_prediction_settings_parse_rollout_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(
                runtime_root=tmpdir,
                prediction={
                    "mode": "ml_shadow",
                    "ml": {"enabled": True, "hardware_key": "rtx5090"},
                },
            )

            self.assertEqual(settings.prediction.mode, "ml_shadow")
            self.assertTrue(settings.prediction.ml.shadow)
            self.assertEqual(settings.prediction.safety_margin.branch, 1.20)

    def test_branch_metadata_maps_to_resource_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(runtime_root=tmpdir)
            repo = _NoLegacyProfileRepo()
            estimator = ResourceEstimator(settings, repo)
            job = TrainingJob.create(
                "pkg.runner:train",
                "branch-job",
                "/tmp/branch.pt",
                runner_kwargs={"batch_size": 4, "steps_per_epoch": 10},
                metadata={
                    "branch_prediction": {
                        "epoch_time_ms": 5000.0,
                        "peak_torch_reserved_mib": 2048.0,
                        "avg_sm_util_percent": 35.0,
                        "predictor_version": "branch-test",
                    }
                },
            )

            prediction = estimator.resource_prediction(job, 4, "cuda_process")

            self.assertIsNotNone(prediction)
            self.assertEqual(prediction.source.value, "branch")
            self.assertEqual(prediction.step_time_ms.mean, 500.0)
            self.assertEqual(estimator.estimate_sm_utilization(job, 4, "cuda_process"), 0.35)
            self.assertEqual(estimator.estimate_peak_vram_mb(job, 4, "cuda_process"), 2048.0 * 1.20)

    def test_explicit_estimate_does_not_query_legacy_profile_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(runtime_root=tmpdir)
            estimator = ResourceEstimator(settings, _NoLegacyProfileRepo())
            job = TrainingJob.create(
                "pkg.runner:train",
                "explicit-job",
                "/tmp/explicit.pt",
                runner_kwargs={"batch_size": 8},
                resource_requirements=ResourceRequirements(requires_gpu=True, estimated_vram_mb=1024),
            )

            estimate = estimator.estimate_peak_vram_mb(job, 8, "cuda_process")

            self.assertEqual(estimate, 1024.0 * 1.30)

    def test_job_local_probe_prediction_requires_same_job_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(runtime_root=tmpdir)
            store = SQLiteStateStore(settings)
            store._hardware_profile = _fake_hardware_profile("job-local-probe")
            estimator = ResourceEstimator(settings, store)
            job = TrainingJob.create(
                "pkg.runner:train",
                "probe-job",
                "/tmp/probe.pt",
                runner_kwargs={"batch_size": 4},
                metadata={"batch_probe_key": "probe-1"},
            )
            store.upsert_batch_probe_profile(
                BatchProbeProfile(
                    probe_key="probe-1",
                    model_key="probe-job",
                    device_type=store.hardware_profile().gpu_name,
                    shape_signature="shape",
                    batch_param_name="batch_size",
                    resolved_batch_size=4,
                    peak_vram_mb=1024,
                    last_job_id="other-job",
                )
            )

            self.assertIsNone(estimator.estimate_peak_vram_mb(job, 4, "cuda_process"))

            store.upsert_batch_probe_profile(
                BatchProbeProfile(
                    probe_key="probe-1",
                    model_key="probe-job",
                    device_type=store.hardware_profile().gpu_name,
                    shape_signature="shape",
                    batch_param_name="batch_size",
                    resolved_batch_size=4,
                    peak_vram_mb=1024,
                    last_job_id=job.job_id,
                )
            )

            self.assertEqual(estimator.estimate_peak_vram_mb(job, 4, "cuda_process"), 1024.0 * 1.10)


if __name__ == "__main__":
    unittest.main()
