"""CPU integration against extracted public artifacts for each supported GPU."""
from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
from dataclasses import replace

import pytest
import torch

from localml_scheduler.config import PredictionSettings
from localml_scheduler.domain import TrainingJob
from localml_scheduler.hardware import HardwareProfile
from localml_scheduler.prediction.ml_predictor import MLVramPredictor, precision_for_job, JobPredictionError
from localml_scheduler.prediction.released_artifacts import hardware_matches
from localml_scheduler.prediction.released_outputs import VRAM_BUCKET_MIB


@pytest.fixture
def source_job(tmp_path):
    source = tmp_path / "candidate.py"
    source.write_text("import torch\ndef build_model():\n    return torch.nn.Linear(4, 2)\n")
    return TrainingJob.create("example:train", "candidate", str(source), job_id="hardware-smoke",
        max_epochs=2, runner_kwargs={"batch_size": 4}, metadata={"perfseer_model": {
            "entry": "build_model", "input_shapes": [["$batch", 4]],
            "loss": "mse", "optimizer": "adamw", "dataset_size": 40}})


@pytest.fixture
def public_paths():
    root = os.environ.get("PERFSEER_RELEASE_ROOT")
    models = os.environ.get("PERFSEER_STUDENT_MODELS_ROOT")
    if not root or not models:
        pytest.skip("set PERFSEER_RELEASE_ROOT and PERFSEER_STUDENT_MODELS_ROOT")
    return Path(root), Path(models)


@pytest.mark.parametrize("device,cc,memory,model_id", [("A100", "8.0", 40960, "S_A100"),
    ("A10G", "8.6", 23028, "S_A10G"), ("A10", "8.6", 23028, "S_A10"),
    ("A10", "8.6", 23028, "TS_A10_A100")])
def test_actual_hardware_student_source_capture(public_paths, source_job, device, cc, memory, model_id):
    root, models = public_paths
    settings = PredictionSettings(mode="ml_predictor", student_release_path=str(root),
        student_model_path=str(models / model_id))
    hw = HardwareProfile("cpu-test", "linux", f"NVIDIA {device}", memory, cc, "test", torch.__version__)
    predictor = MLVramPredictor(settings, hw)
    assert predictor.available, predictor.unavailable_reason
    assert predictor.model_id == f"release/{model_id}"
    values = predictor._released.predict_options(source_job, [4, 8])
    for batch, row in values.items():
        assert len(row) == 6 and 0 <= row[3] <= 100 and min(row[4:]) > 0
        assert row[2] == pytest.approx(row[0] * (40 // batch), rel=1e-5)
        raw = predictor._released.last_capture_audits[batch]["raw_prediction"]
        assert row[:4] == raw[:4]
        for value, original in zip(row[4:], raw[4:]):
            assert value / VRAM_BUCKET_MIB == int(value / VRAM_BUCKET_MIB)
            assert original <= value < original + VRAM_BUCKET_MIB
    audits = dict(predictor._released.last_capture_audits)
    predictor._released.last_capture_audits.clear()
    assert predictor._released.predict_options(source_job, [4, 8]) == values
    assert predictor._released.last_capture_audits == audits
    assert predictor.requires_branch_memory
    assert not torch.cuda.is_initialized()


def test_hardware_aliases_are_not_interchangeable():
    assert hardware_matches("NVIDIA A10G", "A10G")
    assert not hardware_matches("NVIDIA A10G", "A10")
    assert not hardware_matches("NVIDIA A100", "A10")


def test_mlevolve_precision_hint_is_respected(source_job):
    source_job.metadata["precision_mode"] = "bf16"
    assert precision_for_job(source_job) == "bf16_amp"
    source_job.metadata.pop("precision_mode")
    source_job.metadata["uses_amp"] = True
    with pytest.raises(JobPredictionError, match="explicit precision_mode"):
        precision_for_job(source_job)


def test_manifest_tampering_rejected_before_model_load(public_paths, tmp_path):
    root, models = public_paths
    (tmp_path / "assets").mkdir()
    (tmp_path / "RELEASE.json").write_bytes((root / "RELEASE.json").read_bytes())
    (tmp_path / "assets/manifest.json").write_text(json.dumps({"assets": []}))
    settings = PredictionSettings(student_release_path=str(tmp_path), student_model_path=str(models / "S_A10G"))
    hw = HardwareProfile("cpu-test", "linux", "NVIDIA A10G", 23028, "8.6", "test", torch.__version__)
    predictor = MLVramPredictor(settings, hw)
    assert not predictor.available
    assert "manifest hash mismatch" in predictor.unavailable_reason


def test_capture_memory_budget_roundtrip():
    settings = PredictionSettings(conversion_max_rss_mb=6144)
    assert PredictionSettings.from_dict(settings.to_dict()).conversion_max_rss_mb == 6144
    with pytest.raises(ValueError, match="conversion_max_rss_mb"):
        PredictionSettings(conversion_max_rss_mb=0)


def test_generated_style_kwargs_dict_tokens_and_helper_change(public_paths, tmp_path):
    root, models = public_paths
    helper = tmp_path / "local_helper.py"
    helper.write_text("from torch import nn\ndef embedding():\n    return nn.Embedding(16, 4)\n")
    source = tmp_path / "generated.py"
    source.write_text("""import torch
from local_helper import embedding
class Candidate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = embedding()
    def forward(self, tokens, *, mask):
        hidden = self.embedding(tokens) * mask.unsqueeze(-1)
        return {'logits': hidden.mean(dim=(1, 2)).unsqueeze(-1)}
""")
    contract = {
        "version": "mlevolve.perfseer_workload.v1", "mode": "module", "entry": "Candidate",
        "inputs": {"positional": [{"shape": ["$batch", 6], "dtype": "int64", "min": 0, "max": 16}],
                   "keyword": {"mask": {"shape": ["$batch", 6], "dtype": "float32", "fill": 1.0}}},
        "loss": {"kind": "mse", "output_path": ["logits"],
                 "target": {"shape": ["$batch", 1], "dtype": "float32", "fill": 0.0}},
        "training": {"precision": "fp32_ieee", "gradient_accumulation_steps": 1,
                     "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.0,
                                   "foreach": None, "fused": None},
                     "scheduler": {"name": "none"}, "backend": "cuda_process"}}
    from localml_scheduler.prediction.job_contract import write_prediction_sidecar
    write_prediction_sidecar(source, {"dataset_size": 40, "workload_contract": contract})
    job = TrainingJob.create("agent:train", "generated", str(source), runner_kwargs={"batch_size": 4})
    settings = PredictionSettings(student_release_path=str(root), student_model_path=str(models / "S_A10G"))
    hw = HardwareProfile("cpu-test", "linux", "NVIDIA A10G", 23028, "8.6", "test", torch.__version__)
    runtime = MLVramPredictor(settings, hw)._released
    first = runtime.predict_options(job, [4])[4]
    audit = runtime.last_capture_audits[4]
    assert audit["numeric_parity"] and audit["backward_node_count"] > 0
    assert len(runtime.cache) == 1
    helper.write_text(helper.read_text().replace("16, 4", "16, 8"))
    second = runtime.predict_options(job, [4])[4]
    assert len(runtime.cache) == 2, "helper mutation must invalidate the graph cache"
    assert first != second
    assert not torch.cuda.is_initialized()


def test_precomputed_training_graph_roundtrip_uses_strict_17_tensor_schema(public_paths, tmp_path):
    root, models = public_paths
    import sys
    runtime_src = root / "portable/runtime/src"
    aliases = root / "portable/runtime/aliases"
    sys.path[:0] = [str(aliases), str(runtime_src)]
    from perfseer_v3.graph_ir_v3 import GraphIRV3
    from perfseer_v31.capture_training import capture, graph_from_design

    model = torch.nn.Linear(4, 2)
    optimizer = {
        "name": "adamw", "lr": 1e-3, "weight_decay": 0.0,
        "betas": [0.9, 0.999], "eps": 1e-8, "amsgrad": False,
    }
    training = {
        "precision": "fp32_ieee", "microbatch_size": 4, "optimizer": optimizer,
        "scheduler": {"name": "none"}, "backend": "unit-test",
        "steps_per_epoch": 1, "gradient_accumulation_steps": 1,
    }
    design, audit = capture(model, (torch.ones(4, 4),), None, None, training,
                            architecture_key="precomputed-roundtrip")
    assert audit["numerically_finite_and_close"]
    graph = graph_from_design(design)
    graph = replace(
        graph,
        metadata={**graph.metadata, "batch_size": 4, "target_hardware_id": "NVIDIA A10G"},
        training_config={**graph.training_config, "steps_per_epoch": 1,
                         "gradient_accumulation_steps": 1, "optimizer": optimizer},
    )
    graph_path = tmp_path / "strict-graph.json"
    graph.save(graph_path)
    loaded = GraphIRV3.load(graph_path)
    assert loaded.coverage.backward_capture_quality == "strict"
    assert loaded.coverage.replay_validated

    from localml_scheduler.prediction.source_worker import convert_released_source_options_worker
    request = {
        "graph_path": str(graph_path), "batch_size": 4, "raw_steps": 1,
        "precision": "fp32_ieee",
        "metadata": {"gradient_accumulation_steps": 1,
                     "graph_step_semantics": "complete_optimizer_step",
                     "graph_hardware_id": "NVIDIA A10G"},
        "hardware": {"name": "NVIDIA A10G", "memory_bytes": 23028 * 1024 * 1024,
                     "compute_capability": "8.6"},
    }
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=convert_released_source_options_worker,
        args=(sender, str(runtime_src), str(models / "S_A10G/normalization.pt"), [request], 1),
        daemon=True,
    )
    process.start()
    sender.close()
    try:
        assert receiver.poll(90), "precomputed graph conversion did not finish"
        result = receiver.recv()
    finally:
        receiver.close()
        process.join(timeout=2)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
    assert result["ok"], result
    assert len(result["tensors_by_option"][0]) == 17
    assert result["audits"][0]["mode"] == "precomputed_graph"
    assert result["audits"][0]["numeric_parity"] is True
