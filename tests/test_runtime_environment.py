from __future__ import annotations

from pathlib import Path

from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import SchedulerConfig
from localml_scheduler.runtime_environment import (
    repair_generated_training_code,
    validate_generated_training_code,
    validate_model_api_contracts,
)


def _valid_scheduler_script() -> str:
    return """
from localml_scheduler.elastic import ElasticTrainingSession
MODEL_BRANCH = "linear-v1"
batch_size = 8
epochs = 2
session = ElasticTrainingSession.from_env()
train_loader = session.make_dataloader(train_dataset, shuffle=True)
session.register_training_state(model, optimizer, lr_scheduler=lr_scheduler, scaler=scaler)
progress = session.restore_if_present()
optimizer.step()
session.optimizer_step_completed(8, progress["epoch"], 0, progress["global_step"] + 1)
"""


def test_scheduler_submission_contract_accepts_canonical_script() -> None:
    result = validate_generated_training_code(
        _valid_scheduler_script(),
        require_scheduler_submission_contract=True,
    )

    assert result["ok"] is True
    assert result["critical_count"] == 0


def test_elastic_contract_rejects_wrong_runtime_keyword_and_repairs_known_alias() -> None:
    code = _valid_scheduler_script().replace(
        "session.optimizer_step_completed(8, progress[\"epoch\"], 0, progress[\"global_step\"] + 1)",
        "session.optimizer_step_completed(samples=8, epoch=0, batch_idx=0, global_step=1)",
    )

    result = validate_generated_training_code(code, require_scheduler_submission_contract=True)
    repair = repair_generated_training_code(code, require_scheduler_submission_contract=True)

    assert result["ok"] is False
    assert "elastic_api_call_signature_invalid" in {issue.get("code") for issue in result["issues"]}
    assert "batch_index=0" in repair["code"]
    assert "batch_idx=0" not in repair["code"]
    assert repair["validation"]["ok"] is True


def test_scheduler_client_exposes_strict_generated_script_validation(tmp_path: Path) -> None:
    client = SchedulerClient(SchedulerConfig(runtime_root=tmp_path / "runtime"))

    loose = client.validate_generated_training_code("print('hello')")
    strict = client.validate_generated_training_code(
        "print('hello')",
        require_scheduler_submission_contract=True,
    )

    assert loose["ok"] is True
    assert strict["ok"] is False
    assert strict["critical_count"] >= 1


def test_scheduler_submission_contract_rejects_environment_authored_batch() -> None:
    code = _valid_scheduler_script().replace(
        "batch_size = 8",
        'batch_size = int(os.environ.get("BATCH_SIZE", "8"))',
    )

    result = validate_generated_training_code(code, require_scheduler_submission_contract=True)

    assert result["ok"] is False
    assert "scheduler_authored_batch_not_literal" in {issue.get("code") for issue in result["issues"]}


def test_elastic_contract_rejects_loader_batch_override_and_raw_training_loader() -> None:
    code = _valid_scheduler_script().replace(
        "train_loader = session.make_dataloader(train_dataset, shuffle=True)",
        "train_loader = DataLoader(train_dataset, batch_size=8)\n"
        "elastic_loader = session.make_dataloader(train_dataset, batch_size=8)",
    )

    result = validate_generated_training_code(code, require_scheduler_submission_contract=True)

    codes = {issue.get("code") for issue in result["issues"]}
    assert "elastic_loader_batch_size_override" in codes
    assert "elastic_training_loader_bypasses_session" in codes


def test_elastic_contract_method_names_in_comments_do_not_satisfy_validator() -> None:
    code = """
# ElasticTrainingSession.from_env()
# session.make_dataloader(dataset)
# session.register_training_state(model, optimizer)
# session.restore_if_present()
# session.optimizer_step_completed(1, 0, 0, 1)
MODEL_BRANCH = "comments-only"
batch_size = 4
epochs = 1
"""

    result = validate_generated_training_code(code, require_scheduler_submission_contract=True)

    assert result["ok"] is False
    assert sum(issue.get("code") == "elastic_training_contract_missing" for issue in result["issues"]) == 6


def _future_vision_contract() -> dict:
    return {
        "schema_version": 3,
        "display_name": "FutureVision",
        "model_id": "vendor/future-vision",
        "preprocessing": {"fixed_image_size": 384},
        "feature_apis": [
            {
                "method": "encode_images",
                "call": "features = model.encode_images(pixel_values=pixel_values)",
                "return_kind": "tensor",
                "invalid_result_attributes": ["pooler_output", "last_hidden_state"],
                "dimension_config_path": "vision_config.hidden_size",
            }
        ],
        "invalid_config_paths": ["hidden_size", "projection_dim"],
    }


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


def test_validate_generated_training_code_flags_diff_marker_fragments(tmp_path: Path) -> None:
    client = SchedulerClient(SchedulerConfig(runtime_root=tmp_path / "runtime"))
    open_marker = "<" * 7 + " SEARCH"
    middle_marker = "=" * 7
    close_marker = ">" * 7 + " REPLACE"
    code = "\n".join(
        [
            "def train():",
            "    return 1",
            open_marker,
            "old",
            middle_marker,
            "new",
            close_marker,
        ]
    )

    result = client.validate_generated_training_code(code, stage="debug")

    assert result["ok"] is False
    assert "diff_marker_or_conflict_fragment" in {issue["category"] for issue in result["issues"]}


def test_validate_generated_training_code_flags_engineered_feature_dim_mismatch(tmp_path: Path) -> None:
    client = SchedulerClient(SchedulerConfig(runtime_root=tmp_path / "runtime"))
    code = """
import numpy as np
from torch import nn

feature_names = [f"f{i}" for i in range(18)]

def compute_patch_features(row):
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

feature_dim = 16
norm = nn.LayerNorm(16)
"""

    result = client.validate_generated_training_code(code, stage="code_review")

    categories = {issue["category"] for issue in result["issues"]}
    assert "engineered_feature_dim_mismatch" in categories
    issue = next(item for item in result["issues"] if item["category"] == "engineered_feature_dim_mismatch")
    assert "feature_dim" in issue["evidence"]
    assert "LayerNorm" in issue["evidence"]


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


def test_validate_generated_training_code_flags_hf_repo_built_from_sanitized_branch(tmp_path: Path) -> None:
    client = SchedulerClient(SchedulerConfig(runtime_root=tmp_path / "runtime"))
    code = """
from transformers import AutoProcessor
MODEL_BRANCH = "siglip2_so400m_patch16_256"
processor = AutoProcessor.from_pretrained(f"google/{MODEL_BRANCH}")
"""

    result = client.validate_generated_training_code(code, stage="code_review")

    assert result["ok"] is False
    assert result["critical_count"] == 1
    issue = result["issues"][0]
    assert issue["category"] == "derived_huggingface_model_id_from_scheduler_branch"
    assert "PRETRAINED_MODEL_ID" in issue["repair_hint"]
    assert "google/siglip2-so400m-patch16-256" in issue["repair_hint"]


def test_repair_generated_training_code_fixes_known_sanitized_hf_branch() -> None:
    code = """
from transformers import AutoModel, AutoProcessor
MODEL_BRANCH = "siglip2_so400m_patch16_256"
processor = AutoProcessor.from_pretrained(f"google/{MODEL_BRANCH}")
model = AutoModel.from_pretrained(f"google/{MODEL_BRANCH}")
"""

    repair = repair_generated_training_code(code, stage="code_review")

    assert repair["changed"] is True
    assert repair["replacement_count"] == 2
    assert 'MODEL_BRANCH = "siglip2_so400m_patch16_256"' in repair["code"]
    assert repair["code"].count("from_pretrained('google/siglip2-so400m-patch16-256')") == 2
    assert repair["validation"]["ok"] is True


def test_repair_generated_training_code_fixes_known_direct_invalid_hf_repo() -> None:
    code = 'model = AutoModel.from_pretrained("google/siglip2_so400m_patch16_256")'

    repair = repair_generated_training_code(code, stage="code_review")

    assert repair["changed"] is True
    assert repair["replacement_count"] == 1
    assert '"google/siglip2-so400m-patch16-256"' in repair["code"]
    assert repair["validation"]["ok"] is True


def test_validate_generated_training_code_flags_zip_extractall_directory_mismatch(tmp_path: Path) -> None:
    client = SchedulerClient(SchedulerConfig(runtime_root=tmp_path / "runtime"))
    code = """
with zipfile.ZipFile(TRAIN_ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(WORKING_DIR)
with zipfile.ZipFile(TEST_ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(WORKING_DIR)
train_filepaths = list(TRAIN_DIR.glob('*.jpg'))
test_filepaths = list(TEST_DIR.glob('*.jpg'))
"""

    result = client.validate_generated_training_code(code, stage="code_review")

    assert result["ok"] is False
    assert result["critical_count"] == 1
    issue = result["issues"][0]
    assert issue["category"] == "zip_extractall_directory_mismatch"
    assert issue["autofixable"] is True


def test_repair_generated_training_code_fixes_zip_extractall_directory_mismatch() -> None:
    code = """
with zipfile.ZipFile(TRAIN_ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(WORKING_DIR)
with zipfile.ZipFile(TEST_ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(WORKING_DIR)
train_filepaths = list(TRAIN_DIR.glob('*.jpg'))
test_filepaths = list(TEST_DIR.glob('*.jpg'))
"""

    repair = repair_generated_training_code(code, stage="code_review")

    assert repair["changed"] is True
    assert repair["replacement_count"] == 2
    assert "zip_ref.extractall(TRAIN_DIR)" in repair["code"]
    assert "zip_ref.extractall(TEST_DIR)" in repair["code"]
    assert "zip_ref.extractall(WORKING_DIR)" not in repair["code"]
    assert repair["validation"]["ok"] is True


def test_repair_generated_training_code_fixes_configured_zip_extractall_directory_mismatch() -> None:
    code = """
class DataConfig:
    WORKING_PATH = Path('./working')
    TRAIN_ZIP_PATH = Path('./input/train.zip')
    TEST_ZIP_PATH = Path('./input/test.zip')
    TRAIN_DATA_PATH = WORKING_PATH / 'train'
    TEST_DATA_PATH = WORKING_PATH / 'test'

def extract_data(config):
    with zipfile.ZipFile(config.TRAIN_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(config.WORKING_PATH)
    with zipfile.ZipFile(config.TEST_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(config.WORKING_PATH)
"""

    repair = repair_generated_training_code(code, stage="code_review")

    assert repair["changed"] is True
    assert repair["replacement_count"] == 2
    assert "zip_ref.extractall(config.TRAIN_DATA_PATH)" in repair["code"]
    assert "zip_ref.extractall(config.TEST_DATA_PATH)" in repair["code"]
    assert "zip_ref.extractall(config.WORKING_PATH)" not in repair["code"]
    assert repair["validation"]["ok"] is True


def test_validate_model_api_contracts_is_model_family_agnostic() -> None:
    code = '''
MODEL_ID = "vendor/future-vision"
IMAGE_SIZE = 224
model = AutoModel.from_pretrained(MODEL_ID)
features = model.encode_images(pixel_values=pixel_values)
pooled = features.pooler_output
classifier = Linear(model.config.hidden_size, 2)
'''

    issues = validate_model_api_contracts(code, [_future_vision_contract()])

    assert {issue["category"] for issue in issues} == {
        "model_feature_return_contract_violation",
        "model_config_path_contract_violation",
        "model_input_size_contract_violation",
    }
    assert all(issue["model_id"] == "vendor/future-vision" for issue in issues)
    assert all(issue["contract_version"] == 3 for issue in issues)


def test_validate_model_api_contracts_accepts_contract_compliant_code() -> None:
    code = '''
MODEL_ID = "vendor/future-vision"
IMAGE_SIZE = 384
model = AutoModel.from_pretrained(MODEL_ID)
features = model.encode_images(pixel_values=pixel_values)
classifier = Linear(model.config.vision_config.hidden_size, 2)
'''

    assert validate_model_api_contracts(code, [_future_vision_contract()]) == []


def test_validate_model_api_contracts_catches_observed_siglip_tensor_misuse() -> None:
    contract = {
        **_future_vision_contract(),
        "display_name": "SigLIP 2",
        "model_id": "google/siglip2-so400m-patch16-256",
        "feature_apis": [
            {
                "method": "get_image_features",
                "call": "features = model.get_image_features(pixel_values=pixel_values)",
                "return_kind": "tensor",
                "invalid_result_attributes": ["pooler_output", "last_hidden_state"],
                "dimension_config_path": "vision_config.hidden_size",
            }
        ],
    }
    code = '''
MODEL_ID = "google/siglip2-so400m-patch16-256"
model = AutoModel.from_pretrained(MODEL_ID)
vision_outputs = model.get_image_features(pixel_values=pixel_values)
pooled_output = vision_outputs.pooler_output
'''

    issues = validate_model_api_contracts(code, [contract])

    assert [issue["category"] for issue in issues] == [
        "model_feature_return_contract_violation"
    ]
