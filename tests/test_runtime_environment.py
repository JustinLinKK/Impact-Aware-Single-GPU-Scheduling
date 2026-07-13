from __future__ import annotations

from pathlib import Path

from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import SchedulerConfig


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
