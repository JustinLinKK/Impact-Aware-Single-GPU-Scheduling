"""CPU contract tests; actual production feedback has no fallback path."""
from dataclasses import asdict
from hashlib import sha256
import inspect
import json
from pathlib import Path
import pytest
import torch
from localml_scheduler.adapters import shared_training
from localml_scheduler.adapters.shared_training_contract import supports_shared_training_source
from localml_scheduler.benchmarks.a10_feedback_latency.workloads import core


@pytest.fixture
def manifest():
    return json.loads((core.HERE / "manifest.json").read_text())


def test_canonical_shared_entry():
    assert supports_shared_training_source((core.HERE / "candidate.py").read_text())


def test_every_spec_signature_is_exact_shared_runner_formula(manifest):
    for spec in manifest["jobs"] + manifest["calibration"]:
        expected = sha256(json.dumps({"contract": shared_training.CONTRACT,
            "recipe": asdict(core.recipe(spec)), "identity": core.workload_identity(spec),
            "context": core.adapter_context(spec), "adapter": inspect.getsource(core.CandidateAdapter),
            "runner_sha256": core.file_hash(shared_training.__file__)}, sort_keys=True).encode()).hexdigest()
        assert expected == spec["expected_runtime_signature"] == core.expected_signature(spec)
        assert spec["parameter_count"] == spec["trainable_parameter_count"] > 0
        assert spec["epochs"] * spec["steps_per_epoch"] == 64


def test_spec_tampering_rejected(manifest):
    spec = manifest["jobs"][0]
    spec["batch"] *= 2
    with pytest.raises(ValueError, match="spec hash"):
        core.workload_identity(spec)


def test_disjoint_calibration_and_constructed_chains(manifest):
    assert {s["source_id"] for s in manifest["jobs"]} == {14, 16, 17, 18, 20, 22}
    assert {s["source_id"] for s in manifest["calibration"]} == {8, 12}
    for chain in range(3):
        rows = [s for s in manifest["jobs"] if s["chain"] == chain]
        assert {r["role"] for r in rows} == {"parent", "child"}
    split = json.loads((core.HERE / "split.json").read_text())
    assert len(split["training_ids"]) == 32768
    assert len(set(split["validation_ids"])) == 256
    assert not set(split["training_ids"]) & set(split["heldout_ids"])
    assert set(split["validation_ids"]) <= set(split["heldout_ids"])


def test_auc_direction_and_bce_separate():
    target = torch.tensor([[0.], [1.], [0.], [1.]])
    good = core.validation_metrics(torch.tensor([[-2.], [2.], [-1.], [1.]]), target)
    bad = core.validation_metrics(torch.tensor([[2.], [-2.], [1.], [-1.]]), target)
    assert good["validation_auc"] == 1
    assert good["negative_validation_auc"] == -1
    assert good["negative_validation_auc"] < bad["negative_validation_auc"]
    assert good["validation_bce"] < bad["validation_bce"]
    assert good["validation_positive_count"] == good["validation_negative_count"] == 2


@pytest.mark.parametrize("target,logit", [(torch.ones(4,1),torch.zeros(4,1)),
    (torch.tensor([[0.],[1.]]),torch.tensor([[float('nan')],[1.]]))])
def test_invalid_feedback_raises(target, logit):
    with pytest.raises(ValueError):
        core.validation_metrics(logit, target)


def test_missing_image_raises_without_fallback(tmp_path):
    spec = {"model_family": "custom_cnn", "image_size": 96}
    ds = core.CancerDataset(["missing"], {"missing": 0}, tmp_path, {"missing": {}}, spec, False)
    with pytest.raises(FileNotFoundError):
        ds[0]


def test_corrupt_image_raises_without_fallback(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "x.tif").write_bytes(b"bad")
    spec = {"model_family": "custom_cnn", "image_size": 96}
    ds = core.CancerDataset(["x"], {"x": 0}, tmp_path,
        {"x": {"size": 3, "sha256": "wrong"}}, spec, False)
    with pytest.raises(ValueError, match="image differs"):
        ds[0]


def test_custom_model_batch_optimizer_match_capture_contract(manifest):
    spec = next(s for s in manifest["jobs"] if s["source_id"] == 22)
    context = core.adapter_context(spec)
    adapter = core.CandidateAdapter()
    model = adapter.build_model(context)
    assert sum(p.numel() for p in model.parameters()) == spec["parameter_count"]
    batch = adapter.build_train_batch({"batch_size": 2}, "cpu")
    assert batch["image"].shape == (2, 3, 96, 96)
    optimizer = adapter.build_optimizer(model, context)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[0]["weight_decay"] == 1e-5
    loss = adapter.training_step(model, batch, context)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_real_data_prepare_paths_and_identities(manifest, monkeypatch, tmp_path):
    root = Path("/home/justin/MLEvolve/.cache/histopathologic-cancer-benchmark-40960")
    if not root.exists():
        pytest.skip("local real-data integration fixture unavailable")
    monkeypatch.setenv("A10_PILOT_DATA_ROOT", str(root))
    spec = next(s for s in manifest["jobs"] if s["source_id"] == 22)
    workload = core.make_workload(spec, tmp_path)
    assert workload.output_dir == tmp_path
    assert workload.selection_metric == "negative_validation_auc"
    assert len(workload.train_dataset) == 32768
    sample = workload.train_dataset[0]
    assert sample["image"].shape == (3,96,96)
    assert torch.isfinite(sample["image"]).all()


def test_prediction_contract_captures_real_recipe_and_shapes(manifest):
    from localml_scheduler.prediction.source_contract import (
        WorkloadContract, import_source_module, load_entry, build_adapter_batch,
    )
    for spec in manifest["jobs"] + manifest["calibration"]:
        metadata = core.prediction_metadata(spec)
        assert metadata == spec["metadata"]["perfseer_model"]
        contract = WorkloadContract.parse(metadata["workload_contract"],
            request={"batch_size": spec["batch"], "precision": "fp32_ieee"}, metadata=metadata)
        assert contract.raw["adapter"]["context"] == core.adapter_context(spec)
    spec = next(s for s in manifest["jobs"] if s["source_id"] == 22)
    metadata = core.prediction_metadata(spec)
    request = {"batch_size": 2, "precision": "fp32_ieee"}
    contract = WorkloadContract.parse(metadata["workload_contract"], request=request, metadata=metadata)
    module = import_source_module(core.HERE / "candidate.py", entry=contract.entry)
    model, adapter, details = load_entry(module, contract, torch=torch, request=request)
    batch, context = build_adapter_batch(adapter, model, details, torch=torch,
        max_input_numel=contract.max_input_numel, optimizer_spec=contract.optimizer)
    assert batch["image"].shape == (2,3,96,96)
    assert torch.isfinite(adapter.training_step(model,batch,context))
