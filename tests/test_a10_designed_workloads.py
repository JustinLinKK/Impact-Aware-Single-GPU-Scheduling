"""CPU verification for explicit SYNTHETIC training/resource controls."""
from dataclasses import replace
import copy
import json
import math
from pathlib import Path

import pytest
import torch

from engine.script_introspection import supports_cooperative_trial
from localml_scheduler.adapters.shared_training import run_candidate_training
from localml_scheduler.benchmarks.a10_resource_mechanisms import workloads as core
from localml_scheduler.benchmarks.a10_resource_mechanisms import candidate


def sealed_spec(**updates):
    spec = {"source": "candidate.py", "name": "cpu-control", "model_family": core.MODEL_FAMILY,
            "width": 128, "input_features": 128, "repetitions": 2, "batch": 32,
            "ballast_mib": 0, "epochs": 2, "steps_per_epoch": 3, "seed": 101,
            "pool_size": 64, "validation_size": 128,
            "train_pool_seed": 13, "validation_pool_seed": 17, **updates}
    spec.update(core.source_metadata())
    spec["parameter_count"] = core.parameter_count(spec["width"], spec["input_features"])
    spec["trainable_parameter_count"] = spec["parameter_count"]
    spec["spec_sha256"] = core.spec_digest(spec)
    spec["expected_runtime_signature"] = core.expected_signature(spec)
    return spec


def test_two_epoch_real_adamw_has_successful_updates_and_finite_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    spec = sealed_spec()
    workload = core.make_workload(spec, tmp_path / "first")
    before = core.build_model(spec).state_dict()
    result = run_candidate_training(lambda: workload)
    saved = torch.load(tmp_path / "first/training_state.pt", weights_only=False)
    assert result["signature"] == spec["expected_runtime_signature"]
    assert result["global_step"] == saved["global_step"] == 6
    assert result["epochs"] == len(result["history"]) == 2
    assert all(int(state["step"]) == 6 for state in saved["optimizer_state"]["state"].values())
    assert any(not torch.equal(before[key], saved["model_state"][key]) for key in before)
    for row in result["history"]:
        assert 0 <= row["validation_auc"] <= 1
        assert math.isfinite(row["validation_bce"])
        assert row["negative_validation_auc"] == -row["validation_auc"]
        assert row["validation_negative_count"] == row["validation_positive_count"] == 64
    resources = json.loads((tmp_path / "first/workload_resources.json").read_text())
    assert resources["trainable_parameter_bytes"] == spec["parameter_count"] * 4
    assert resources["allocated_ballast_bytes"] == 0
    again = run_candidate_training(lambda: core.make_workload(spec, tmp_path / "again"))
    assert again == result
    second = torch.load(tmp_path / "again/training_state.pt", weights_only=False)
    assert saved["batch_history"] == second["batch_history"]
    for key in saved["model_state"]:
        torch.testing.assert_close(saved["model_state"][key], second["model_state"][key], rtol=0, atol=0)


def test_data_are_reproducible_disjoint_and_virtual_for_large_budgets(tmp_path):
    spec = sealed_spec(steps_per_epoch=17, batch=32)
    first = core.make_workload(spec, tmp_path / "a")
    torch.manual_seed(999)
    second = core.make_workload(spec, tmp_path / "b")
    assert len(first.train_dataset) >= 32 * 17
    assert first.train_dataset.features.shape == (64, 128)
    torch.testing.assert_close(first.train_dataset.features, second.train_dataset.features, rtol=0, atol=0)
    torch.testing.assert_close(first.train_dataset[0]["features"], first.train_dataset[64]["features"], rtol=0, atol=0)
    validation_x, validation_y = core._pool(128, 128, 17, 33013)
    assert not torch.equal(first.train_dataset.features[0], validation_x[0])
    assert set(validation_y.flatten().tolist()) == {0.0, 1.0}
    with pytest.raises(ValueError, match="disjoint"):
        core.make_workload(sealed_spec(validation_pool_seed=13), tmp_path / "invalid")


def test_full_identity_binds_seeds_recipe_code_and_configuration_but_template_reuses(tmp_path):
    original = sealed_spec()
    key = core.template_key(original)
    for updates in ({"seed": 102}, {"train_pool_seed": 19, "validation_pool_seed": 23},
                    {"epochs": 3, "steps_per_epoch": 5}, {"name": "calibration", "role": "calibration", "chain": None}):
        changed = sealed_spec(**updates)
        assert core.template_key(changed) == key
        assert core.expected_signature(changed) != original["expected_runtime_signature"]
    for updates in ({"batch": 128}, {"repetitions": 8}, {"width": 2048},
                    {"ballast_mib": 8192}, {"validation_size": 256}, {"pool_size": 128},
                    {"learning_rate": 0.0005}, {"teacher_seed": 37}):
        assert core.template_key(sealed_spec(**updates)) != key
    altered = copy.deepcopy(original)
    altered["package_pins"]["workloads.py"] = "0" * 64
    altered["spec_sha256"] = core.spec_digest(altered)
    with pytest.raises(ValueError, match="source/runtime"):
        core.expected_signature(altered)
    altered = dict(original, batch=128)
    with pytest.raises(ValueError, match="spec hash"):
        core.make_workload(altered, tmp_path)


def test_ballast_cpu_preparation_allocates_no_ballast_and_never_checkpoints_it(tmp_path, monkeypatch):
    original_zeros = torch.zeros
    def bounded_zeros(*args, **kwargs):
        size = args[0] if args else kwargs.get("size", 0)
        if isinstance(size, int) and size >= 1024 ** 3:
            pytest.fail("CPU preparation attempted a large ballast allocation")
        return original_zeros(*args, **kwargs)
    monkeypatch.setattr(torch, "zeros", bounded_zeros)
    spec = sealed_spec(ballast_mib=8192)
    workload = core.make_workload(spec, tmp_path)
    model = workload.adapter.build_model(workload.adapter_context).float().to("cpu")
    report = model.resource_accounting()
    assert report["declared_ballast_bytes"] == 8192 * 1024 ** 2
    assert report["allocated_ballast_bytes"] == report["ballast_initializations_this_launch"] == 0
    assert model._cuda_ballast.numel() == 0
    assert "_cuda_ballast" not in model.state_dict()
    loaded = core.build_model(spec)
    loaded.load_state_dict(model.state_dict())
    assert loaded.declared_ballast_bytes == report["declared_ballast_bytes"]
    assert loaded.resource_accounting()["allocated_ballast_bytes"] == 0
    assert loaded.resource_accounting()["trainable_parameter_bytes"] == report["trainable_parameter_bytes"]


@pytest.mark.parametrize("repetitions", [2, 8])
def test_reused_compute_layer_has_finite_nonzero_gradients_and_adamw_update(repetitions):
    torch.manual_seed(3)
    model = core.ControlledMLP(input_features=8, width=16, repetitions=repetitions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = model.residual_layer.weight.detach().clone()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(torch.randn(32, 8)),
                                                                torch.arange(32).remainder(2).float().view(-1, 1))
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    assert model.residual_layer.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, model.residual_layer.weight)
    assert sum(p.numel() for p in model.parameters()) == core.parameter_count(16, 8)


def test_candidate_direct_shared_entry_and_existing_harness_environment(tmp_path, monkeypatch):
    assert supports_cooperative_trial(Path(candidate.__file__).read_text())
    spec = sealed_spec()
    monkeypatch.delenv("A10_DESIGNED_SPEC_JSON", raising=False)
    monkeypatch.delenv("A10_DESIGNED_OUTPUT", raising=False)
    monkeypatch.setenv("A10_PILOT_SPEC_JSON", json.dumps(spec))
    monkeypatch.setenv("A10_PILOT_OUTPUT", str(tmp_path))
    prepared = candidate.prepare_training()
    assert prepared.output_dir == tmp_path
    assert prepared.identity == core.workload_identity(spec)
