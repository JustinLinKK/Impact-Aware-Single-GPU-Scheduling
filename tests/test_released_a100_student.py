"""Actual published S_A100 integration; set PERFSEER_RELEASE_ROOT to run.

No trained model is mocked. The optional TRAIN-only probe checks packed feature
parity; source conversion and scheduling integration need only the public release.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import pytest
import torch

from localml_scheduler.config import PredictionSettings, SchedulerSettings
from localml_scheduler.domain import BatchSizeObservation, TrainingJob, build_batch_size_observation_key
from localml_scheduler.hardware import HardwareProfile
from localml_scheduler.prediction import JobPredictionError, MLVramPredictor
from localml_scheduler.prediction.released_student import ReleasedStudentRuntime, digest, steps_for_job
from localml_scheduler.scheduler.resource_estimator import ResourceEstimator
from localml_scheduler.storage.sqlite_store import SQLiteStateStore


def hardware(name="NVIDIA A100-SXM4-40GB"):
    return HardwareProfile("cpu-test", "linux", name, 40960, "8.0", "test", torch.__version__)


@pytest.fixture(scope="module")
def settings():
    root = os.environ.get("PERFSEER_RELEASE_ROOT")
    if not root:
        pytest.skip("set PERFSEER_RELEASE_ROOT to the extracted public release code")
    return PredictionSettings(mode="ml_predictor", student_release_path=root,
        student_model_path=os.environ.get("PERFSEER_STUDENT_MODEL_DIR"), conversion_timeout_seconds=90)


@pytest.fixture(scope="module")
def predictor(settings):
    result = MLVramPredictor(settings, hardware())
    assert result.available, result.unavailable_reason
    return result


@pytest.fixture
def job(tmp_path):
    source = tmp_path / "tiny.py"
    source.write_text("import torch\ndef build_model():\n    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))\n")
    return TrainingJob.create("example:train", "tiny", str(source), job_id="a100-smoke",
        max_epochs=3, runner_kwargs={"batch_size": 4}, metadata={"perfseer_model": {
            "source_path": str(source), "input_shapes": [["$batch", 4]], "entry": "build_model",
            "loss": "mean", "optimizer": "adamw", "dataset_size": 80}})


def test_source_conversion_actual_model_units_cache_and_steps(predictor, job):
    values = predictor._released.predict_options(job, [4, 8])
    busy = predictor.predict_busy_percent_options(job, [4, 8])
    seconds = predictor.predict_seconds_per_epoch_options(job, [4, 8])
    memory = predictor.predict_avg_vram_options(job, [4, 8])
    for batch in (4, 8):
        assert 0 <= busy[batch] <= 100 and memory[batch] > 0
        assert memory[batch] == values[batch][4]  # MiB, no decimal-MB conversion.
        assert seconds[batch] == values[batch][2] / 1000
        assert math.isclose(values[batch][2], values[batch][0] * (80 // batch), rel_tol=1e-5)
    count = len(predictor._released.cache)
    assert predictor.predict_seconds_per_epoch_options(job, [4, 8]) == seconds
    assert len(predictor._released.cache) == count
    job.metadata["perfseer_model"]["steps_per_epoch_by_batch"] = {"4": 40}
    assert math.isclose(predictor.predict_seconds_per_epoch_options(job, [4])[4], seconds[4] * 2, rel_tol=1e-4)
    assert predictor.requires_branch_memory
    assert not torch.cuda.is_initialized()


def test_estimator_uses_actual_epoch_prediction_and_measured_memory(settings, job, tmp_path):
    config = SchedulerSettings(runtime_root=tmp_path / "runtime", prediction=settings.to_dict(),
        graph_db={"enabled": False}, hardware_feature_db={"enabled": False})
    store = SQLiteStateStore(config)
    # The repository uses a declared deployment GPU; inference itself stays CPU.
    class Repository:
        def hardware_profile(self): return hardware()
        def hardware_key(self): return hardware().hardware_key
        def __getattr__(self, name): return getattr(store, name)
    estimator = ResourceEstimator(config, Repository())
    observation = BatchSizeObservation(observation_key=build_batch_size_observation_key(
        "tiny", estimator.shape_signature(job), hardware().hardware_key, "exclusive", 4),
        batch_param_name="batch_size", model_key="tiny", shape_signature=estimator.shape_signature(job),
        hardware_key=hardware().hardware_key, backend_name="exclusive", batch_size=4,
        avg_vram_mb=777, peak_vram_mb=888)
    store.upsert_batch_size_observation(observation)
    result, = estimator.estimate_batch_options(job, "exclusive", [4])
    expected = estimator.ml_predictor.predict_seconds_per_epoch_options(job, [4])[4]
    assert result.seconds_per_epoch == expected
    assert result.remaining_runtime_seconds == expected * 3
    assert result.avg_vram_mb == 777
    assert result.source == "branch_profile+ml_predictor"
    assert estimator.estimate_avg_vram_mb(job, 4, "exclusive") == 777
    assert estimator.estimate_peak_vram_mb(job, 4, "exclusive") == 888
    assert job.metadata["vram_prediction_source"] == "branch_profile"
    assert not torch.cuda.is_initialized()


def test_hardware_and_config_contract(settings):
    assert PredictionSettings.from_dict(settings.to_dict()) == settings
    with pytest.raises(ValueError, match="requires student_release_path"):
        PredictionSettings(student_model_path="/missing")
    wrong = MLVramPredictor(settings, hardware("NVIDIA V100"))
    assert not wrong.available and "trained for A100" in wrong.unavailable_reason
    assert wrong.requires_branch_memory


def test_corrupted_normalizer_rejected(settings, predictor, tmp_path):
    source = predictor._released.model_root
    for name in ("student.ts", "architecture.pt"):
        (tmp_path / name).symlink_to(source / name)
    (tmp_path / "normalization.pt").write_bytes(b"wrong normalizer")
    changed = PredictionSettings(student_release_path=settings.student_release_path, student_model_path=str(tmp_path))
    broken = MLVramPredictor(changed, hardware())
    assert not broken.available and "normalization hash mismatch" in broken.unavailable_reason


def test_unknown_steps_and_unsupported_capture_fall_back(predictor, job):
    metadata = job.metadata["perfseer_model"]
    del metadata["dataset_size"]
    with pytest.raises(JobPredictionError, match="steps_per_epoch"):
        predictor.predict_seconds_per_epoch_options(job, [4])
    metadata["steps_per_epoch"] = 7
    metadata["precision"] = "bf16"
    with pytest.raises(JobPredictionError, match="explicit training_graph_path"):
        predictor.predict_seconds_per_epoch_options(job, [4])
    assert predictor.last_sources[job.job_id] == "branch_profile"
    assert predictor.available


def test_optimizer_steps_batch_and_accumulation(job):
    job.metadata["perfseer_model"].update(dataset_size=100, gradient_accumulation_steps=2)
    assert steps_for_job(job, 4) == 13
    assert steps_for_job(job, 10) == 5
    job.metadata["perfseer_model"] = {"steps_per_epoch": 7}
    assert steps_for_job(job, 4) == 7
    with pytest.raises(ValueError, match="another physical batch"):
        steps_for_job(job, 8)


def test_real_train_probe_tensor_parity(predictor):
    probe = os.environ.get("PERFSEER_A100_PROBE")
    if not probe:
        pytest.skip("optional PERFSEER_A100_PROBE input-only TRAIN fixture is not installed")
    runtime = predictor._released
    sys.path[:0] = [str(runtime.runtime_src), str(runtime.runtime_src.parent / "aliases")]
    from perfseer_v3.features import batch_graph_features
    data = torch.load(probe, map_location="cpu", weights_only=False)
    assert data["hardware"] == "A100" and data["epoch_protocol"] == "measured_single_step"
    assert data["source_split"] == 0
    assert not {"targets", "raw_targets_v4", "labels", "splits"} & data.keys()
    normalizer = torch.load(runtime.normalization_path, map_location="cpu", weights_only=False)
    assert digest(probe) == "8877f50d1030e6541634a38806efeb164569690c49ecceb841d934e388175fe9"
    from perfseer_v3.features import apply_normalization
    data["features"] = [f if f.metadata.get("normalization_sha256") == normalizer.sha256
                        else apply_normalization(f, normalizer) for f in data["features"]]
    batch = batch_graph_features(data["features"])
    tensors = (batch.x_cont, batch.op_exact_id, batch.phase_id, batch.dtype_id, batch.node_flags,
        batch.edge_index, batch.edge_cont, batch.edge_role_id, batch.edge_flags, batch.batch,
        batch.u_cont, batch.quality, batch.precision_id, batch.optimizer_id,
        batch.hardware_cont, batch.hardware_missing_mask, torch.as_tensor(data["raw_steps_per_epoch"]))
    extra = {"student_config.json": ""}
    direct = torch.jit.load(str(runtime.artifact_path), map_location="cpu", _extra_files=extra).eval()
    with torch.inference_mode():
        expected = direct(*tensors)
    actual = runtime.predict_tensors(tensors)
    for name in ("prediction", "memory_upper"):
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    assert actual["prediction"].shape == (2, 6)
    assert runtime.contract["units"] == ("ms", "ms", "ms", "percent", "MiB", "MiB")
    assert not torch.cuda.is_initialized()
