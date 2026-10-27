from __future__ import annotations

import pytest
import torch

from localml_scheduler.prediction.source_contract import (
    CONTRACT_VERSION,
    ContractError,
    WorkloadContract,
    build_adapter_batch,
    build_tree,
    dependency_fingerprint,
    fingerprint_source,
    import_source_module,
    load_entry,
)


def _optimizer(**overrides):
    return {
        "name": "adamw",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "foreach": None,
        "fused": None,
        **overrides,
    }


def _module_contract(**overrides):
    contract = {
        "version": CONTRACT_VERSION,
        "mode": "module",
        "entry": "build_model",
        "inputs": {
            "positional": [
                {"shape": ["$batch", 3], "dtype": "float32", "fill": 0.25},
                {"shape": ["$batch", 2], "dtype": "int64", "min": 0, "max": 5},
            ],
            "keyword": {"scale": {"literal": 2.0}},
        },
        "loss": {"kind": "mse", "target": {"shape": ["$batch", 1], "dtype": "float32", "fill": 0.0}},
        "training": {
            "precision": "fp32_ieee",
            "gradient_accumulation_steps": 1,
            "optimizer": _optimizer(),
            "scheduler": {"name": "none"},
            "backend": "unit-test",
        },
        "max_input_numel": 100,
    }
    contract.update(overrides)
    return contract


def _request(**metadata):
    return {"batch_size": 4, "precision": "fp32_ieee", "raw_steps": 3, "metadata": metadata}


def test_versioned_contract_builds_keyword_and_pytree_inputs():
    contract = WorkloadContract.parse(_module_contract(), request=_request(), metadata={})
    args = tuple(build_tree(item, batch_size=4, torch=torch, generator=torch.Generator().manual_seed(4))
                 for item in contract.raw["inputs"]["positional"])
    kwargs = {key: build_tree(value, batch_size=4, torch=torch, generator=torch.Generator().manual_seed(4))
              for key, value in contract.raw["inputs"]["keyword"].items()}
    assert [tuple(item.shape) for item in args] == [(4, 3), (4, 2)]
    assert args[1].dtype == torch.int64 and args[1].min() >= 0 and args[1].max() < 5
    assert kwargs == {"scale": 2.0}
    assert contract.optimizer["betas"] == [0.9, 0.999]
    assert contract.optimizer["eps"] == 1e-8
    assert contract.optimizer["amsgrad"] is False
    assert contract.optimizer["foreach"] is None and contract.optimizer["fused"] is None


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda c: c["training"].update(precision="bf16_amp"), "unsupported_precision"),
        (lambda c: c["training"].update(gradient_accumulation_steps=2), "unsupported_accumulation"),
        (lambda c: c["training"].pop("scheduler"), "invalid_contract"),
        (lambda c: c["loss"].update(label_smoothing=0.2), "unsupported_loss_config"),
        (lambda c: c["training"]["optimizer"].update(decoupled_weight_decay=True), "unsupported_optimizer_config"),
        (lambda c: c["inputs"]["positional"][1].pop("max"), "missing_integer_bounds"),
        (lambda c: c.update(max_input_numel=1), "input_bound_exceeded"),
    ],
)
def test_invalid_contracts_fail_closed(mutate, code):
    contract = _module_contract()
    mutate(contract)
    with pytest.raises(ContractError) as error:
        WorkloadContract.parse(contract, request=_request(), metadata={})
    assert error.value.code == code


def test_legacy_explicit_input_shapes_and_recipe_are_preserved():
    request = {
        "batch_size": 4,
        "precision": "fp32_ieee",
        "entry": "build_model",
        "input_shapes": [["$batch", 4]],
        "input_dtypes": ["float32"],
        "loss": "mean",
        "optimizer": "adamw",
        "metadata": {"learning_rate": 2e-4, "weight_decay": 0.02},
    }
    contract = WorkloadContract.parse(None, request=request, metadata=request["metadata"])
    assert contract.mode == "module" and contract.entry == "build_model"
    assert contract.raw["inputs"]["positional"][0]["shape"] == [4, 4]
    assert contract.optimizer["lr"] == pytest.approx(2e-4)
    assert contract.optimizer["weight_decay"] == pytest.approx(0.02)
    assert contract.loss["target"]["legacy_output_shape"] is True


def test_candidate_adapter_calls_declared_training_step_and_validates_batch(tmp_path):
    src = tmp_path / "candidate.py"
    src.write_text(
        "import torch\n"
        "class PetModel(torch.nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = torch.nn.Linear(4, 1)\n"
        "    def forward(self, image, tabular):\n"
        "        return self.fc(torch.cat([image[:, 0, 0, :3], tabular[:, :2]], dim=1))\n"
        "class CandidateAdapter:\n"
        "    def build_model(self, context): return PetModel()\n"
        "    def build_optimizer(self, model, context): return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)\n"
        "    def build_train_batch(self, scenario, device):\n"
        "        b = scenario['batch_size']\n"
        "        return {'image': torch.zeros(b,3,2,2), 'tabular': torch.zeros(b,12), 'target': torch.ones(b,1)}\n"
        "    def training_step(self, model, batch, context):\n"
        "        return context['criterion'](model(batch['image'], batch['tabular']), batch['target'])\n"
    )
    contract_data = {
        "version": CONTRACT_VERSION,
        "mode": "candidate_adapter",
        "entry": "CandidateAdapter",
        "adapter": {
            "context": {"criterion": {"kind": "mse"}},
            "scenario": {},
            "batch_spec": {
                "image": {"shape": ["$batch", 3, 2, 2], "dtype": "float32"},
                "tabular": {"shape": ["$batch", 12], "dtype": "float32"},
                "target": {"shape": ["$batch", 1], "dtype": "float32"},
            },
        },
        "loss": {"kind": "candidate_adapter_training_step"},
        "training": {
            "precision": "fp32_ieee", "gradient_accumulation_steps": 1,
            "optimizer": _optimizer(), "scheduler": {"name": "none"}, "backend": "unit-test",
        },
    }
    request = _request()
    contract = WorkloadContract.parse(contract_data, request=request, metadata={})
    module = import_source_module(src, entry=contract.entry)
    model, adapter, details = load_entry(module, contract, torch=torch, request=request)
    assert details["scenario"]["fixture"]["image"]["shape"] == [3, 2, 2]
    batch, context = build_adapter_batch(adapter, model, details, torch=torch,
                                         max_input_numel=contract.max_input_numel,
                                         optimizer_spec=contract.optimizer)
    loss = adapter.training_step(model, batch, context)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_candidate_optimizer_exact_class_groups_and_params_are_checked():
    model = torch.nn.Linear(2, 1)
    recipe = {
        "name": "adamw", "lr": 1e-3, "weight_decay": 1e-4, "betas": [0.9, 0.999],
        "eps": 1e-8, "amsgrad": False, "foreach": None, "fused": None,
    }
    from localml_scheduler.prediction.source_contract import _check_adapter_optimizer

    valid = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    _check_adapter_optimizer(valid, model, recipe, torch)
    with pytest.raises(ContractError, match="differs from declared optimizer"):
        _check_adapter_optimizer(torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4), model, recipe, torch)
    wrong_group = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    wrong_group.param_groups[0]["lr"] = 0.5
    with pytest.raises(ContractError, match="parameter groups override"):
        _check_adapter_optimizer(wrong_group, model, recipe, torch)


def test_source_import_roots_support_package_relative_imports_and_fingerprint(tmp_path):
    root = tmp_path / "project"
    package = root / "model_pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "blocks.py").write_text("import torch\nclass Block(torch.nn.Module):\n    def forward(self,x): return x + 1\n")
    source = package / "network.py"
    source.write_text("import torch\nfrom .blocks import Block\nclass Net(torch.nn.Module):\n    def __init__(self): super().__init__(); self.block=Block()\n    def forward(self,x): return self.block(x)\n")
    receipt = fingerprint_source(source, {"workload_contract": {"import_roots": [str(root)]}})
    assert receipt["complete"] is True
    assert any(item["path"].endswith("blocks.py") for item in receipt["files"])
    module = import_source_module(source, import_roots=[str(root)], entry="Net")
    assert torch.equal(module.Net()(torch.tensor([1.0])), torch.tensor([2.0]))


def test_source_fingerprint_marks_dynamic_imports_incomplete(tmp_path):
    source = tmp_path / "dynamic.py"
    source.write_text("import importlib\nmodule = importlib.import_module('x')\n")
    receipt = dependency_fingerprint(source)
    assert receipt["dynamic_imports"] is True
    assert receipt["complete"] is False


def test_selected_entry_download_is_rejected_but_unused_main_guard_is_ignored(tmp_path):
    source = tmp_path / "download.py"
    source.write_text(
        "def build_model():\n    return load.from_pretrained('x')\n"
        "def unused():\n    return load.from_pretrained('y')\n"
    )
    with pytest.raises(ContractError, match="selected source entry reaches"):
        import_source_module(source, entry="build_model")
    source.write_text(
        "def build_model():\n    return __import__('torch').nn.Linear(1,1)\n"
        "if __name__ == '__main__':\n    model = load.from_pretrained('x')\n"
    )
    module = import_source_module(source, entry="build_model")
    assert isinstance(module.build_model(), torch.nn.Module)


def test_selected_torch_module_instance_and_path_constants_are_supported(tmp_path):
    source = tmp_path / "instance.py"
    source.write_text(
        "from pathlib import Path\n"
        "from torch import nn\n"
        "ROOT = Path(__file__).parent\n"
        "DATA = ROOT / 'fixtures'\n"
        "MODEL = nn.Linear(4, 2)\n"
    )
    module = import_source_module(source, entry="MODEL")
    assert isinstance(module.MODEL, torch.nn.Module)
    assert module.DATA == tmp_path / "fixtures"
