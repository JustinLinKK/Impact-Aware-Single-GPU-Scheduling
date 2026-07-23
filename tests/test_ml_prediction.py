from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from localml_scheduler.config import PredictionSettings, SchedulerSettings
from localml_scheduler.domain import ResourceRequirements, TrainingJob
from localml_scheduler.hardware import HardwareProfile
from localml_scheduler.prediction import JobPredictionError, MLVramPredictor
from localml_scheduler.scheduler.resource_estimator import ResourceEstimator


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "PerfSeer-predictor" / "models" / "nvidia_a10" / "student_a10_cpu.torchscript.pt"
SUPPORTED_SOURCE = ROOT / "PerfSeer-predictor" / "tests" / "fixtures" / "tiny_conv.py"
UNSUPPORTED_SOURCE = Path(__file__).parent / "fixtures" / "perfseer_unsupported.py"
TIMEOUT_SOURCE = Path(__file__).parent / "fixtures" / "perfseer_timeout.py"


def hardware(name: str, capability: str, memory_mb: int) -> HardwareProfile:
    return HardwareProfile("test-hardware", "linux", name, memory_mb, capability, "13.0", "test")


def model_job(source: Path, *, job_id: str = "predict-job") -> TrainingJob:
    return TrainingJob.create(
        "localml_scheduler.examples.toy_pytorch_runner:run_toy_job",
        "tiny",
        str(source),
        job_id=job_id,
        resource_requirements=ResourceRequirements(estimated_avg_vram_mb=777),
        metadata={
            "perfseer_model": {
                "source_path": str(source),
                "entry": "build_model",
                "input_shapes": [["$batch", 3, 32, 32]],
                "input_dtypes": ["float32"],
                "precision": "fp32",
            }
        },
    )


class _Repository:
    def __init__(self, profile: HardwareProfile):
        self.profile = profile

    def hardware_profile(self):
        return self.profile

    def hardware_key(self):
        return self.profile.hardware_key

    def get_batch_size_observation(self, **kwargs):
        return None

    def list_batch_size_observations(self, **kwargs):
        return []

    def get_batch_probe_profile(self, key):
        return None

    def get_solo_profile(self, signature):
        return None


class _IncompatibleOverride(torch.nn.Module):
    def forward(self, x, edge_index, edge_attr, u, batch):
        return x.new_zeros((1, 1))


class MLPredictionTest(unittest.TestCase):
    def test_auto_selects_a10_and_runs_cpu_prediction(self) -> None:
        predictor = MLVramPredictor(
            PredictionSettings(mode="ml_predictor"),
            hardware("NVIDIA A10", "8.6", 23028),
        )
        self.assertTrue(predictor.available, predictor.unavailable_reason)
        value = predictor.predict_avg_vram_mb(model_job(SUPPORTED_SOURCE), 2)
        self.assertGreater(value, 0)
        self.assertEqual(predictor.last_sources["predict-job"], "ml_predictor")

    def test_rejects_non_a10_without_override(self) -> None:
        predictor = MLVramPredictor(
            PredictionSettings(mode="ml_predictor"),
            hardware("NVIDIA GeForce RTX 5090", "12.0", 32607),
        )
        self.assertFalse(predictor.available)
        with self.assertRaises(JobPredictionError):
            predictor.predict_avg_vram_mb(model_job(SUPPORTED_SOURCE), 2)

    def test_explicit_test_override_runs_on_other_hardware(self) -> None:
        predictor = MLVramPredictor(
            PredictionSettings(
                mode="ml_predictor",
                test_override_enabled=True,
                test_model_path=str(ARTIFACT),
            ),
            hardware("NVIDIA GeForce RTX 5090", "12.0", 32607),
        )
        self.assertTrue(predictor.available, predictor.unavailable_reason)
        self.assertGreater(predictor.predict_avg_vram_mb(model_job(SUPPORTED_SOURCE), 2), 0)

    def test_override_rejects_incompatible_torchscript_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "incompatible.torchscript.pt"
            example = (
                torch.zeros((2, 53)),
                torch.tensor([[0], [1]], dtype=torch.long),
                torch.zeros((1, 3)),
                torch.zeros((1, 40)),
                torch.zeros(2, dtype=torch.long),
            )
            torch.jit.trace(_IncompatibleOverride(), example).save(str(path))
            predictor = MLVramPredictor(
                PredictionSettings(
                    mode="ml_predictor",
                    test_override_enabled=True,
                    test_model_path=str(path),
                ),
                hardware("NVIDIA GeForce RTX 5090", "12.0", 32607),
            )
            self.assertFalse(predictor.available)
            self.assertIn("shape", predictor.unavailable_reason)

    def test_unsupported_job_falls_back_without_disabling_ml(self) -> None:
        settings = SchedulerSettings(
            runtime_root=Path(tempfile.gettempdir()) / "unused-predictor-test",
            prediction={
                "mode": "ml_predictor",
                "ml": {
                    "test_override_enabled": True,
                    "test_model_path": str(ARTIFACT),
                },
            },
        )
        estimator = ResourceEstimator(settings, _Repository(hardware("RTX 5090", "12.0", 32607)))
        unsupported = model_job(UNSUPPORTED_SOURCE, job_id="unsupported")
        supported = model_job(SUPPORTED_SOURCE, job_id="supported")
        self.assertEqual(estimator.estimate_avg_vram_mb(unsupported, 2, "exclusive"), 1554)
        self.assertEqual(unsupported.metadata["vram_prediction_source"], "branch_profile")
        self.assertGreater(estimator.estimate_avg_vram_mb(supported, 2, "exclusive"), 0)
        self.assertEqual(supported.metadata["vram_prediction_source"], "ml_predictor")

    def test_conversion_timeout_is_per_job_failure(self) -> None:
        predictor = MLVramPredictor(
            PredictionSettings(
                mode="ml_predictor",
                conversion_timeout_seconds=0.1,
                test_override_enabled=True,
                test_model_path=str(ARTIFACT),
            ),
            hardware("RTX 5090", "12.0", 32607),
        )
        with self.assertRaisesRegex(JobPredictionError, "exceeded"):
            predictor.predict_avg_vram_mb(model_job(TIMEOUT_SOURCE), 2)


if __name__ == "__main__":
    unittest.main()
