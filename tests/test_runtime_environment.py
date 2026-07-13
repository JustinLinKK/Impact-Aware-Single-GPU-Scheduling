from __future__ import annotations

from pathlib import Path

from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import SchedulerConfig
from localml_scheduler.runtime_environment import repair_generated_training_code


def test_runtime_environment_reports_torch_scheduler_signature(tmp_path: Path) -> None:
    client = SchedulerClient(SchedulerConfig(runtime_root=tmp_path / "runtime"))

    env = client.get_runtime_environment(
        include_package_versions=False,
        include_precision_checks=True,
    )

    assert "python" in env
    assert "torch" in env
    assert "CosineAnnealingLR" in env["pytorch_scheduler_signatures"]
    params = env["pytorch_scheduler_signatures"]["CosineAnnealingLR"]["parameters"]
    assert "T_max" in params
    assert "eta_min" in params
    assert "T_eta_min" not in params
    assert "precision_checks" in env


def test_validate_generated_training_code_flags_known_runtime_failures(tmp_path: Path) -> None:
    client = SchedulerClient(SchedulerConfig(runtime_root=tmp_path / "runtime"))
    code = """
import torch
from torch import optim
AMP_DTYPE = torch.bfloat16
preds_np = preds.cpu().numpy()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_eta_min=1e-6, T_max=3)
scaler = torch.cuda.amp.GradScaler()
"""

    result = client.validate_generated_training_code(code, stage="code_review")

    assert result["ok"] is False
    categories = {issue["category"] for issue in result["issues"]}
    assert "bf16_numpy_conversion" in categories
    assert "invalid_torch_scheduler_argument" in categories
    assert "deprecated_cuda_amp_api" in categories
    assert result["critical_count"] == 2


def test_runtime_environment_reports_precision_export_policy(tmp_path: Path) -> None:
    client = SchedulerClient(SchedulerConfig(runtime_root=tmp_path / "runtime"))

    env = client.get_runtime_environment(
        include_package_versions=False,
        include_precision_checks=True,
    )

    checks = env["precision_checks"]
    assert "low_precision_numpy_export_policy" in checks
    assert "pytorch_float8_dtypes" in checks
    assert "torch.float32" in checks["low_precision_numpy_export_policy"]["safe_pattern"]


def test_validate_generated_training_code_flags_generic_low_precision_exports(tmp_path: Path) -> None:
    client = SchedulerClient(SchedulerConfig(runtime_root=tmp_path / "runtime"))
    code = """
import transformer_engine.pytorch as te
PRECISION = "nvfp4"
preds_np = preds.cpu().numpy()
labels_np = labels.cpu().numpy()
ids_np = ids.cpu().numpy()
safe_np = probs.detach().to(torch.float32).cpu().numpy()
"""

    result = client.validate_generated_training_code(code, stage="code_review")

    categories = [issue["category"] for issue in result["issues"]]
    assert categories == ["low_precision_numpy_export"]
    issue = result["issues"][0]
    assert issue["autofixable"] is True
    assert "preds.cpu().numpy()" in issue["evidence"]


def test_validate_generated_training_code_flags_torchao_fp8_exports(tmp_path: Path) -> None:
    client = SchedulerClient(SchedulerConfig(runtime_root=tmp_path / "runtime"))
    code = """
from torchao.float8 import convert_to_float8_training
model = convert_to_float8_training(model)
logits_np = logits.cpu().numpy()
"""

    result = client.validate_generated_training_code(code, stage="code_review")

    assert result["ok"] is False
    assert result["issues"][0]["category"] == "low_precision_numpy_export"
    assert "logits.cpu().numpy()" in result["issues"][0]["evidence"]


def test_repair_generated_training_code_only_rewrites_prediction_exports() -> None:
    code = """
import torch
AMP_DTYPE = torch.bfloat16
preds_np = preds.cpu().numpy().flatten()
labels_np = labels.cpu().numpy()
ids_np = ids.cpu().numpy()
"""

    repair = repair_generated_training_code(code, stage="code_review")

    assert repair["changed"] is True
    assert repair["replacement_count"] == 1
    assert "preds.detach().to(torch.float32).cpu().numpy().flatten()" in repair["code"]
    assert "labels.cpu().numpy()" in repair["code"]
    assert "ids.cpu().numpy()" in repair["code"]
    assert repair["validation"]["ok"] is True


def test_repair_generated_training_code_adds_torch_import_when_needed() -> None:
    code = '''"""module docstring"""
from __future__ import annotations
import transformer_engine.pytorch as te
PRECISION = "nvfp4"
preds_np = preds.cpu().numpy()
'''

    repair = repair_generated_training_code(code, stage="code_review")

    assert repair["changed"] is True
    assert '"""module docstring"""' in repair["code"]
    assert "from __future__ import annotations\nimport torch\n" in repair["code"]
    assert "preds.detach().to(torch.float32).cpu().numpy()" in repair["code"]
