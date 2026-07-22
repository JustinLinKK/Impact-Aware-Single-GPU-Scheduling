from __future__ import annotations

from pathlib import Path

import pytest

from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import ResourceRequirements, TrainingJob
from localml_scheduler.hardware import HardwareProfile, build_hardware_key
from localml_scheduler.scheduler.resource_estimator import ResourceEstimator


def _hardware() -> HardwareProfile:
    return HardwareProfile(
        hardware_key=build_hardware_key(
            os_name="linux",
            gpu_name="prediction-gpu",
            total_vram_mb=24_576,
            compute_capability="9.0",
            cuda_runtime="12.8",
            torch_version="2.8.0",
        ),
        os_name="linux",
        gpu_name="prediction-gpu",
        total_vram_mb=24_576,
        compute_capability="9.0",
        cuda_runtime="12.8",
        torch_version="2.8.0",
    )


class MinimalRepo:
    def hardware_profile(self) -> HardwareProfile:
        return _hardware()

    def hardware_key(self) -> str:
        return _hardware().hardware_key

    def get_pair_profile(self, *args, **kwargs):
        return None

    def best_combination_profile(self, **kwargs):
        return None


def test_prediction_settings_accept_only_branch_profile_and_ml_predictor(tmp_path: Path) -> None:
    assert SchedulerSettings(runtime_root=tmp_path / "branch").prediction.mode == "branch_profile"
    settings = SchedulerSettings(
        runtime_root=tmp_path / "ml",
        prediction={"mode": "ml_predictor", "ml": {"enabled": True, "hardware_key": "rtx5090"}},
    )
    assert settings.prediction.mode == "ml_predictor"
    assert settings.prediction.ml.enabled is True
    for old in ("branch_only", "ml_shadow", "ml_primary", "confidence_first", "hybrid"):
        with pytest.raises(ValueError, match="Unsupported prediction mode"):
            SchedulerSettings(runtime_root=tmp_path / old, prediction={"mode": old})


def test_branch_metadata_maps_to_resource_prediction(tmp_path: Path) -> None:
    settings = SchedulerSettings(runtime_root=tmp_path)
    estimator = ResourceEstimator(settings, MinimalRepo())
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
    assert prediction is not None and prediction.source.value == "branch"
    assert prediction.step_time_ms.mean == 500.0
    assert estimator.estimate_sm_utilization(job, 4, "cuda_process") == 0.35
    assert estimator.estimate_peak_vram_mb(job, 4, "cuda_process") == 2048.0 * 1.20


def test_ml_mode_can_use_explicit_fallback_when_predictor_is_unavailable(tmp_path: Path) -> None:
    settings = SchedulerSettings(
        runtime_root=tmp_path,
        prediction={"mode": "ml_predictor", "fallback_to_exclusive": True},
    )
    estimator = ResourceEstimator(settings, MinimalRepo())
    job = TrainingJob.create(
        "pkg.runner:train",
        "explicit-job",
        "/tmp/explicit.pt",
        runner_kwargs={"batch_size": 8},
        resource_requirements=ResourceRequirements(requires_gpu=True, estimated_vram_mb=1024),
    )
    assert estimator.estimate_peak_vram_mb(job, 8, "cuda_process") == 1024.0 * 1.30
