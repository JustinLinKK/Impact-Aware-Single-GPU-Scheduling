from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import (
    BatchProbeProfile,
    BatchSizeObservation,
    CombinationProfile,
    PairProfile,
    ResourceRequirements,
    SoloProfile,
    TrainingJob,
    build_batch_size_observation_key,
    build_group_signature,
)
from localml_scheduler.scheduler.resource_estimator import ResourceEstimator
from localml_scheduler.scheduler.telemetry import GpuTelemetrySample, GpuTelemetrySummary
from localml_scheduler.storage.sqlite_store import SQLiteStateStore


class AverageVramTest(unittest.TestCase):
    def test_telemetry_separates_average_used_vram_from_controller_utilization(self) -> None:
        summary = GpuTelemetrySummary.from_samples(
            [
                GpuTelemetrySample(memory_used_mb=100, gpu_utilization=0.25, memory_utilization=0.9),
                GpuTelemetrySample(memory_used_mb=300, gpu_utilization=0.75, memory_utilization=0.1),
            ]
        )
        self.assertEqual(summary.peak_vram_mb, 300)
        self.assertEqual(summary.avg_vram_mb, 200)
        self.assertEqual(summary.avg_gpu_utilization, 0.5)
        self.assertEqual(summary.avg_memory_utilization, 0.5)

    def test_sqlite_round_trips_average_vram_for_all_packing_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = SchedulerSettings(
                runtime_root=Path(temporary),
                graph_db={"enabled": False},
                hardware_feature_db={"enabled": False},
            )
            store = SQLiteStateStore(settings)
            hardware_key = store.hardware_key()
            store.upsert_solo_profile(SoloProfile("solo", hardware_key=hardware_key, avg_vram_mb=101.5))
            store.upsert_pair_profile(
                PairProfile.create("left", "right", backend_name="cuda_process", hardware_key=hardware_key, avg_vram_mb=202.5)
            )
            store.upsert_batch_probe_profile(
                BatchProbeProfile("probe", "model", "gpu", "shape", "batch_size", 2, avg_vram_mb=303.5)
            )
            observation = BatchSizeObservation(
                build_batch_size_observation_key("model", "shape", hardware_key, "cuda_process", 2),
                "model",
                "shape",
                hardware_key,
                "cuda_process",
                "batch_size",
                2,
                avg_vram_mb=404.5,
            )
            store.upsert_batch_size_observation(observation)
            combination = CombinationProfile.create(
                build_group_signature(["left", "right"]),
                hardware_key,
                "cuda_process",
                "parallel_time_aware",
                {"left": 2, "right": 2},
                avg_vram_mb=505.5,
            )
            store.upsert_combination_profile(combination)

            self.assertEqual(store.get_solo_profile("solo").avg_vram_mb, 101.5)
            self.assertEqual(
                store.get_pair_profile("left", "right", backend_name="cuda_process").avg_vram_mb,
                202.5,
            )
            self.assertEqual(store.get_batch_probe_profile("probe").avg_vram_mb, 303.5)
            self.assertEqual(
                store.get_batch_size_observation(
                    model_key="model",
                    shape_signature="shape",
                    hardware_key=hardware_key,
                    backend_name="cuda_process",
                    batch_size=2,
                ).avg_vram_mb,
                404.5,
            )
            self.assertEqual(
                store.best_combination_profile(
                    group_signature=combination.group_signature,
                    hardware_key=hardware_key,
                    backend_name="cuda_process",
                    scheduler_mode="parallel_time_aware",
                ).avg_vram_mb,
                505.5,
            )

    def test_branch_estimator_uses_average_not_peak(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = SchedulerSettings(
                runtime_root=Path(temporary),
                graph_db={"enabled": False},
                hardware_feature_db={"enabled": False},
            )
            store = SQLiteStateStore(settings)
            job = TrainingJob.create(
                "pkg.runner:train",
                "model",
                "/tmp/model.pt",
                runner_kwargs={"batch_size": 2},
                resource_requirements=ResourceRequirements(estimated_avg_vram_mb=999),
            )
            estimator = ResourceEstimator(settings, store)
            observation = BatchSizeObservation(
                build_batch_size_observation_key(
                    "model",
                    estimator.shape_signature(job),
                    store.hardware_key(),
                    "exclusive",
                    2,
                ),
                "model",
                estimator.shape_signature(job),
                store.hardware_key(),
                "exclusive",
                "batch_size",
                2,
                peak_vram_mb=9000,
                avg_vram_mb=450,
            )
            store.upsert_batch_size_observation(observation)
            self.assertEqual(estimator.estimate_avg_vram_mb(job, 2, "exclusive"), 450)
            self.assertEqual(estimator.estimate_peak_vram_mb(job, 2, "exclusive"), 9000)


if __name__ == "__main__":
    unittest.main()
