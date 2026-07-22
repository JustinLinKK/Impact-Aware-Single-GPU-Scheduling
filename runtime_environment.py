"""Runtime environment and generated-code compatibility helpers."""

from __future__ import annotations

import ast
import importlib
import importlib.metadata
import inspect
import platform
import re
import sys
from typing import Any


COMMON_ML_PACKAGES = (
    "torch",
    "torchvision",
    "torchaudio",
    "timm",
    "transformers",
    "accelerate",
    "transformer_engine",
    "torchao",
    "numpy",
    "pandas",
    "sklearn",
    "scikit-learn",
    "PIL",
    "pillow",
)

_SCHEDULER_NAMES = (
    "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts",
    "StepLR",
    "MultiStepLR",
    "ReduceLROnPlateau",
    "OneCycleLR",
    "ExponentialLR",
)

_LOW_PRECISION_MARKERS = (
    "autocast",
    "amp_dtype",
    "torch.amp",
    "cuda.amp",
    "bfloat16",
    "bf16",
    "float16",
    "fp16",
    "float8",
    "fp8",
    "mxfp8",
    "mx_fp8",
    "nvfp4",
    "mxfp4",
    "mx_fp4",
    "fp4",
    "transformer_engine",
    "te.fp8_autocast",
    "te.autocast",
    "torchao.float8",
)
_BF16_MARKERS = ("bfloat16", "bf16", "torch.bfloat16")
_PREDICTION_EXPORT_TOKENS = (
    "pred",
    "prediction",
    "prob",
    "proba",
    "probability",
    "logit",
    "output",
    "score",
    "forecast",
    "submission",
)
_NON_PREDICTION_EXPORT_TOKENS = (
    "label",
    "target",
    "truth",
    "y_true",
    "id",
    "idx",
    "index",
    "indices",
    "image",
    "input",
)
_SCHEDULER_BRANCH_KEY_NAMES = {
    "MODEL_BRANCH",
    "model_branch",
    "BRANCH_NAME",
    "branch_name",
    "SCHEDULER_BRANCH_NAME",
    "scheduler_branch_name",
    "MODEL_FAMILY",
    "model_family",
    "SCHEDULER_MODEL_FAMILY",
    "scheduler_model_family",
}
_KNOWN_HF_REPO_ID_REPAIRS = {
    "google/siglip2_so400m_patch16_256": "google/siglip2-so400m-patch16-256",
}
_KNOWN_HF_BRANCH_VALUE_REPAIRS = {
    "siglip2_so400m_patch16_256": "siglip2-so400m-patch16-256",
}


def detect_runtime_environment(
    *,
    include_package_versions: bool = True,
    include_precision_checks: bool = True,
    device_index: int = 0,
) -> dict[str, Any]:
    """Return compact facts about the installed ML runtime."""
    payload: dict[str, Any] = {
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }

    torch_info = _torch_runtime_info(device_index=device_index)
    if torch_info:
        payload["torch"] = torch_info
        payload["pytorch_scheduler_signatures"] = _torch_scheduler_signatures()

    if include_package_versions:
        payload["packages"] = _package_versions(COMMON_ML_PACKAGES)

    if include_precision_checks:
        payload["precision_checks"] = _precision_checks()

    return payload


def validate_generated_training_code(
    code: str,
    *,
    stage: str = "code_review",
    model_contracts: list[dict[str, Any]] | None = None,
    require_elastic_contract: bool = False,
    require_scheduler_submission_contract: bool = False,
) -> dict[str, Any]:
    """Statically flag generated-code patterns known to fail in this runtime."""
    code_text = str(code or "")
    issues: list[dict[str, Any]] = []

    issues.extend(_detect_diff_or_conflict_fragments(code_text))
    issues.extend(_detect_invalid_torch_scheduler_kwargs(code_text))
    issues.extend(_detect_engineered_feature_dim_mismatch(code_text))
    issues.extend(_detect_low_precision_numpy_conversion(code_text))
    issues.extend(_detect_hf_model_source_identifier_issues(code_text))
    issues.extend(_detect_zip_extractall_directory_mismatch(code_text))
    issues.extend(_detect_deprecated_cuda_amp(code_text))
    issues.extend(validate_model_api_contracts(code_text, model_contracts or []))
    if require_elastic_contract or require_scheduler_submission_contract:
        issues.extend(_detect_elastic_training_contract_violations(code_text))
    if require_scheduler_submission_contract:
        issues.extend(_detect_scheduler_submission_contract_violations(code_text))

    critical_count = sum(1 for issue in issues if issue.get("severity") == "critical")
    warning_count = sum(1 for issue in issues if issue.get("severity") == "warning")
    return {
        "ok": critical_count == 0,
        "stage": stage,
        "critical_count": critical_count,
        "warning_count": warning_count,
        "issues": issues,
        "summary": _compatibility_summary(critical_count, warning_count),
    }


def validate_model_api_contracts(
    code: str,
    model_contracts: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Validate generated code against data-driven pretrained-model API contracts."""
    code_text = str(code or "")
    try:
        tree = ast.parse(code_text)
    except SyntaxError:
        return []

    issues: list[dict[str, Any]] = []
    for contract in model_contracts or []:
        if not isinstance(contract, dict):
            continue
        model_id = str(contract.get("model_id") or "").strip()
        if not model_id or model_id not in code_text:
            continue
        display_name = str(contract.get("display_name") or model_id)
        issues.extend(
            _detect_model_feature_api_contract_violations(
                code_text,
                tree,
                contract,
                model_id=model_id,
                display_name=display_name,
            )
        )
        issues.extend(
            _detect_model_config_contract_violations(
                code_text,
                tree,
                contract,
                model_id=model_id,
                display_name=display_name,
            )
        )
        issues.extend(
            _detect_model_input_size_contract_violations(
                code_text,
                tree,
                contract,
                model_id=model_id,
                display_name=display_name,
            )
        )
    return issues


def _detect_model_feature_api_contract_violations(
    code: str,
    tree: ast.AST,
    contract: dict[str, Any],
    *,
    model_id: str,
    display_name: str,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    feature_apis = contract.get("feature_apis")
    if not isinstance(feature_apis, list):
        return issues

    for feature_api in feature_apis:
        if not isinstance(feature_api, dict) or feature_api.get("return_kind") != "tensor":
            continue
        method = str(feature_api.get("method") or "").strip()
        invalid_attributes = {
            str(value).strip()
            for value in feature_api.get("invalid_result_attributes", [])
            if str(value).strip()
        }
        if not method or not invalid_attributes:
            continue

        result_names: set[str] = set()
        for node in ast.walk(tree):
            value: ast.AST | None = None
            targets: list[ast.AST] = []
            if isinstance(node, ast.Assign):
                value = node.value
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                value = node.value
                targets = [node.target]
            if not isinstance(value, ast.Call) or _call_name(value.func) != method:
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    result_names.add(target.id)

        seen: set[tuple[int, str]] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr not in invalid_attributes:
                continue
            direct_call = isinstance(node.value, ast.Call) and _call_name(node.value.func) == method
            named_result = isinstance(node.value, ast.Name) and node.value.id in result_names
            if not direct_call and not named_result:
                continue
            finding_key = (getattr(node, "lineno", 0), node.attr)
            if finding_key in seen:
                continue
            seen.add(finding_key)
            dimension_path = str(feature_api.get("dimension_config_path") or "").strip()
            call = str(feature_api.get("call") or f"features = model.{method}(...)")
            hint = f"Use the tensor returned by `{call}` directly."
            if dimension_path:
                hint += f" Read the classifier input dimension from `model.config.{dimension_path}`."
            issues.append(
                {
                    "severity": "critical",
                    "category": "model_feature_return_contract_violation",
                    "message": (
                        f"{display_name} `{method}(...)` returns a tensor, so accessing "
                        f"`.{node.attr}` on its result will fail."
                    ),
                    "evidence": _source_segment(code, node),
                    "repair_hint": hint,
                    "autofixable": False,
                    "model_id": model_id,
                    "contract_version": contract.get("schema_version"),
                }
            )
    return issues


def _detect_model_config_contract_violations(
    code: str,
    tree: ast.AST,
    contract: dict[str, Any],
    *,
    model_id: str,
    display_name: str,
) -> list[dict[str, Any]]:
    invalid_paths = {
        str(value).strip()
        for value in contract.get("invalid_config_paths", [])
        if str(value).strip()
    }
    if not invalid_paths:
        return []
    dimension_paths = [
        str(feature_api.get("dimension_config_path") or "").strip()
        for feature_api in contract.get("feature_apis", [])
        if isinstance(feature_api, dict) and feature_api.get("dimension_config_path")
    ]
    replacement_path = dimension_paths[0] if dimension_paths else ""
    issues: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        chain = _attribute_chain(node)
        if len(chain) < 2 or chain[-2] != "config" or chain[-1] not in invalid_paths:
            continue
        finding_key = (getattr(node, "lineno", 0), chain[-1])
        if finding_key in seen:
            continue
        seen.add(finding_key)
        hint = f"Do not assume `{'.'.join(chain[-2:])}` exists for `{model_id}`."
        if replacement_path:
            hint += f" Use `model.config.{replacement_path}` for this contract's feature dimension."
        issues.append(
            {
                "severity": "critical",
                "category": "model_config_path_contract_violation",
                "message": f"{display_name} does not expose the generated config path `{'.'.join(chain[-2:])}`.",
                "evidence": _source_segment(code, node),
                "repair_hint": hint,
                "autofixable": False,
                "model_id": model_id,
                "contract_version": contract.get("schema_version"),
            }
        )
    return issues


def _detect_model_input_size_contract_violations(
    code: str,
    tree: ast.AST,
    contract: dict[str, Any],
    *,
    model_id: str,
    display_name: str,
) -> list[dict[str, Any]]:
    preprocessing = contract.get("preprocessing")
    if not isinstance(preprocessing, dict):
        return []
    required_size = preprocessing.get("fixed_image_size")
    if not isinstance(required_size, int) or required_size <= 0:
        return []

    size_names = {"IMAGE_SIZE", "IMG_SIZE", "INPUT_SIZE", "INPUT_RESOLUTION"}
    issues: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = list(node.targets) if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, int):
            continue
        target = next((item for item in targets if isinstance(item, ast.Name) and item.id.upper() in size_names), None)
        if target is None or value.value == required_size:
            continue
        issues.append(
            {
                "severity": "critical",
                "category": "model_input_size_contract_violation",
                "message": (
                    f"{display_name} requires fixed `{required_size}x{required_size}` inputs, but "
                    f"`{target.id}` is set to `{value.value}`."
                ),
                "evidence": _source_segment(code, node),
                "repair_hint": (
                    f"Use `{target.id} = {required_size}` or preprocess images with the contract's configured processor."
                ),
                "autofixable": False,
                "model_id": model_id,
                "contract_version": contract.get("schema_version"),
            }
        )
    return issues


def _attribute_chain(node: ast.AST) -> list[str]:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return list(reversed(parts))


def repair_generated_training_code(
    code: str,
    *,
    stage: str = "code_review",
    require_elastic_contract: bool = False,
    require_scheduler_submission_contract: bool = False,
) -> dict[str, Any]:
    """Deterministically repair prediction-like low-precision Tensor -> NumPy exports."""
    original_code = str(code or "")
    repaired_code, model_source_repair_count = _repair_known_hf_model_source_ids(original_code)
    zip_replacements = _zip_extractall_target_replacements(repaired_code)
    for start, end, replacement in sorted(zip_replacements, key=lambda item: item[0], reverse=True):
        repaired_code = repaired_code[:start] + replacement + repaired_code[end:]
    elastic_api_replacements = (
        _elastic_api_keyword_replacements(repaired_code)
        if require_elastic_contract or require_scheduler_submission_contract
        else []
    )
    for start, end, replacement in sorted(elastic_api_replacements, key=lambda item: item[0], reverse=True):
        repaired_code = repaired_code[:start] + replacement + repaired_code[end:]
    replacements = _low_precision_numpy_replacements(repaired_code)
    for start, end, replacement in sorted(replacements, key=lambda item: item[0], reverse=True):
        repaired_code = repaired_code[:start] + replacement + repaired_code[end:]
    if replacements and "torch.float32" in repaired_code and not _has_torch_import(repaired_code):
        repaired_code = _insert_torch_import(repaired_code)

    validation = validate_generated_training_code(
        repaired_code,
        stage=stage,
        require_elastic_contract=require_elastic_contract,
        require_scheduler_submission_contract=require_scheduler_submission_contract,
    )
    return {
        "code": repaired_code,
        "changed": repaired_code != original_code,
        "replacement_count": (
            model_source_repair_count
            + len(zip_replacements)
            + len(elastic_api_replacements)
            + len(replacements)
        ),
        "stage": stage,
        "validation": validation,
    }


def _contract_issue(*, code: str, message: str, hint: str, evidence: str | None = None) -> dict[str, Any]:
    issue: dict[str, Any] = {
        "severity": "critical",
        "category": "elastic_training_contract",
        "code": code,
        "message": message,
        "repair_hint": hint,
        "autofixable": False,
    }
    if evidence:
        issue["evidence"] = evidence
    return issue


def _detect_elastic_training_contract_violations(code: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return []

    issues: list[dict[str, Any]] = []
    has_import = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "localml_scheduler.elastic"
        and any(alias.name == "ElasticTrainingSession" for alias in node.names)
        for node in tree.body
    )
    if not has_import:
        issues.append(
            _contract_issue(
                code="elastic_training_contract_missing",
                message="Mandatory elastic training contract is missing the exact `ElasticTrainingSession` import.",
                hint="Add `from localml_scheduler.elastic import ElasticTrainingSession`.",
            )
        )

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    def is_from_env_call(node: ast.AST | None) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_env"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ElasticTrainingSession"
        )

    session_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(_target_name(target) == "session" for target in _assignment_targets(node))
        and is_from_env_call(node.value)
    ]
    if not session_assignments:
        issues.append(
            _contract_issue(
                code="elastic_training_contract_missing",
                message="Mandatory elastic training contract is missing `session = ElasticTrainingSession.from_env()`.",
                hint="Assign the scheduler-owned session to the canonical `session` variable in the executable path.",
            )
        )
    else:
        for assignment in session_assignments:
            if assignment.value.args or assignment.value.keywords:
                issues.append(
                    _contract_issue(
                        code="elastic_api_call_signature_invalid",
                        message="`ElasticTrainingSession.from_env()` does not accept arguments.",
                        hint="Create the session exactly as `session = ElasticTrainingSession.from_env()`.",
                        evidence=_source_segment(code, assignment.value),
                    )
                )

    required_methods = {
        "make_dataloader": "construct the training loader through the elastic session",
        "register_training_state": "register model, optimizer, scheduler, scaler, and extra state",
        "restore_if_present": "restore checkpoint state before training",
        "optimizer_step_completed": "report each completed optimizer update as a safe point",
    }
    method_calls: dict[str, list[ast.Call]] = {name: [] for name in required_methods}
    for call in calls:
        if not isinstance(call.func, ast.Attribute):
            continue
        if (
            call.func.attr in method_calls
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "session"
        ):
            method_calls[call.func.attr].append(call)
    for method_name, instruction in required_methods.items():
        if method_calls[method_name]:
            continue
        issues.append(
            _contract_issue(
                code="elastic_training_contract_missing",
                message=f"Mandatory elastic training contract is missing `.{method_name}(...)`.",
                hint=instruction,
            )
        )

    for call in method_calls["optimizer_step_completed"]:
        issues.extend(
            _validate_elastic_method_call(
                code,
                call,
                method_name="optimizer_step_completed",
                positional_parameters=("samples", "epoch", "batch_index", "global_step"),
                keyword_only_parameters=("metrics",),
            )
        )

    for call in method_calls["register_training_state"]:
        issues.extend(
            _validate_elastic_method_call(
                code,
                call,
                method_name="register_training_state",
                positional_parameters=("model", "optimizer"),
                keyword_only_parameters=(
                    "lr_scheduler",
                    "scaler",
                    "extra_state",
                    "extra_state_loader",
                ),
            )
        )

    for call in method_calls["restore_if_present"]:
        if call.args or call.keywords:
            issues.append(
                _contract_issue(
                    code="elastic_api_call_signature_invalid",
                    message="`session.restore_if_present()` does not accept arguments.",
                    hint="Call exactly `progress = session.restore_if_present()`.",
                    evidence=_source_segment(code, call),
                )
            )

    for call in method_calls["make_dataloader"]:
        dataset_keywords = [keyword for keyword in call.keywords if keyword.arg == "dataset"]
        if len(call.args) > 1 or (not call.args and not dataset_keywords) or (call.args and dataset_keywords):
            issues.append(
                _contract_issue(
                    code="elastic_api_call_signature_invalid",
                    message="`session.make_dataloader(...)` requires exactly one training dataset.",
                    hint="Call `train_loader = session.make_dataloader(train_dataset, shuffle=True, ...)`.",
                    evidence=_source_segment(code, call),
                )
            )
        if any(keyword.arg == "batch_size" for keyword in call.keywords):
            issues.append(
                _contract_issue(
                    code="elastic_loader_batch_size_override",
                    message="`session.make_dataloader(...)` must not receive `batch_size=`; the scheduler owns the physical batch.",
                    hint="Remove the `batch_size` keyword and read the resolved value from `session.batch_size`.",
                    evidence=_source_segment(code, call),
                )
            )

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call) or _call_name(value.func) != "DataLoader":
            continue
        targets = _assignment_targets(node)
        target_names = {(_target_name(target) or "").lower() for target in targets}
        if target_names & {"loader", "train_loader", "training_loader", "train_dataloader"}:
            issues.append(
                _contract_issue(
                    code="elastic_training_loader_bypasses_session",
                    message="The training loader is constructed with raw `DataLoader(...)` instead of the elastic session.",
                    hint="Use `session.make_dataloader(train_dataset, ...)`; reserve raw DataLoader for validation/test only.",
                    evidence=_source_segment(code, node),
                )
            )

    return issues


def _validate_elastic_method_call(
    code: str,
    call: ast.Call,
    *,
    method_name: str,
    positional_parameters: tuple[str, ...],
    keyword_only_parameters: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Validate generated calls against the public ElasticTrainingSession API."""

    issues: list[dict[str, Any]] = []
    allowed_keywords = set(positional_parameters) | set(keyword_only_parameters)
    explicit_keywords = [keyword.arg for keyword in call.keywords if keyword.arg is not None]
    unknown_keywords = sorted(set(explicit_keywords) - allowed_keywords)
    starred_keywords = any(keyword.arg is None for keyword in call.keywords)
    missing_parameters = [
        name
        for index, name in enumerate(positional_parameters)
        if index >= len(call.args) and name not in explicit_keywords
    ]
    duplicate_parameters = sorted(
        name
        for name in set(explicit_keywords)
        if explicit_keywords.count(name) > 1
        or (name in positional_parameters and positional_parameters.index(name) < len(call.args))
    )
    if (
        len(call.args) > len(positional_parameters)
        or unknown_keywords
        or starred_keywords
        or duplicate_parameters
        or missing_parameters
    ):
        signature = ", ".join(positional_parameters)
        if keyword_only_parameters:
            signature += ", *, " + ", ".join(keyword_only_parameters)
        details: list[str] = []
        if len(call.args) > len(positional_parameters):
            details.append(f"too many positional arguments ({len(call.args)})")
        if unknown_keywords:
            details.append(f"unsupported keyword(s): {', '.join(unknown_keywords)}")
        if starred_keywords:
            details.append("dynamic **kwargs cannot be verified")
        if duplicate_parameters:
            details.append(f"duplicate argument(s): {', '.join(duplicate_parameters)}")
        if missing_parameters:
            details.append(f"missing required argument(s): {', '.join(missing_parameters)}")
        issues.append(
            _contract_issue(
                code="elastic_api_call_signature_invalid",
                message=f"`session.{method_name}(...)` does not match the runtime API: {'; '.join(details)}.",
                hint=f"Use only the exact parameters `{signature}`. For safe points the batch keyword is `batch_index`, not `batch_idx`.",
                evidence=_source_segment(code, call),
            )
        )
    return issues


def _elastic_api_keyword_replacements(code: str) -> list[tuple[int, int, str]]:
    """Repair unambiguous generated aliases for elastic runtime keywords."""

    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return []
    offsets = _line_offsets(code)
    replacements: list[tuple[int, int, str]] = []
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        if not (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "optimizer_step_completed"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "session"
        ):
            continue
        for keyword in call.keywords:
            if keyword.arg != "batch_idx":
                continue
            span = _node_span(keyword, offsets)
            if span is None:
                continue
            source = code[span[0] : span[1]]
            separator = source.find("=")
            if separator < 0:
                continue
            replacements.append((span[0], span[0] + separator, "batch_index"))
    return replacements


def _top_level_assignment(tree: ast.Module, name: str) -> list[ast.Assign | ast.AnnAssign]:
    matches: list[ast.Assign | ast.AnnAssign] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if any(_target_name(target) == name for target in _assignment_targets(node)):
            matches.append(node)
    return matches


def _detect_scheduler_submission_contract_violations(code: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return []

    issues: list[dict[str, Any]] = []
    branch_assignments = _top_level_assignment(tree, "MODEL_BRANCH")
    if len(branch_assignments) != 1 or not isinstance(branch_assignments[0].value, ast.Constant) or not isinstance(branch_assignments[0].value.value, str) or not branch_assignments[0].value.value.strip():
        issues.append(
            _contract_issue(
                code="scheduler_model_branch_invalid",
                message="Scheduler submission requires exactly one non-empty top-level string literal `MODEL_BRANCH`.",
                hint="Add `MODEL_BRANCH = \"canonical-architecture-name\"` near the imports and update it when the mother model changes.",
            )
        )

    batch_assignments = _top_level_assignment(tree, "batch_size")
    batch_value = None
    if len(batch_assignments) == 1:
        batch_value = _static_int_literal(batch_assignments[0].value)
    if len(batch_assignments) != 1 or batch_value is None:
        issues.append(
            _contract_issue(
                code="scheduler_authored_batch_not_literal",
                message="Scheduler submission requires exactly one top-level integer literal `batch_size`.",
                hint="Declare, for example, `batch_size = 32`; do not derive the authored batch from environment variables or overwrite it.",
            )
        )
    elif batch_value <= 0 or (batch_value & (batch_value - 1)) != 0:
        issues.append(
            _contract_issue(
                code="scheduler_authored_batch_not_power_of_two",
                message=f"Authored `batch_size = {batch_value}` is not a positive power of two.",
                hint="Choose one of 1, 2, 4, 8, 16, 32, 64, ... as the authored batch.",
                evidence=_source_segment(code, batch_assignments[0]),
            )
        )

    all_batch_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(_target_name(target) == "batch_size" for target in _assignment_targets(node))
    ]
    if len(all_batch_assignments) > len(batch_assignments):
        issues.append(
            _contract_issue(
                code="scheduler_authored_batch_mutated",
                message="The immutable authored `batch_size` is reassigned after its top-level declaration.",
                hint="Keep `batch_size` immutable and use `session.batch_size` for the scheduler-selected physical batch.",
            )
        )

    epoch_assignments = _top_level_assignment(tree, "epochs")
    epoch_value = _static_int_literal(epoch_assignments[0].value) if len(epoch_assignments) == 1 else None
    if len(epoch_assignments) != 1 or epoch_value is None or epoch_value <= 0:
        issues.append(
            _contract_issue(
                code="scheduler_epoch_count_not_literal",
                message="Scheduler submission requires one positive top-level integer literal `epochs`.",
                hint="Declare, for example, `epochs = 5`; scheduler/probe limits are applied externally.",
            )
        )

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    optimizer_steps = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "step"
        and not (
            isinstance(call.func.value, ast.Name)
            and call.func.value.id.lower() in {"scheduler", "lr_scheduler"}
        )
    ]
    if not optimizer_steps:
        issues.append(
            _contract_issue(
                code="scheduler_optimizer_step_missing",
                message="Scheduler submission has no concrete optimizer or GradScaler `step(...)` call.",
                hint="Use a checkpointable PyTorch optimizer and call its step (or `scaler.step(optimizer)`) before each elastic safe point.",
            )
        )
    return issues


_DIFF_OR_CONFLICT_MARKER_RE = re.compile(
    r"(?m)^\s*(?:<<<<<<<(?:\s+SEARCH)?|=======|>>>>>>>(?:\s+REPLACE)?|<\s*SEARCH|>\s*REPLACE)\s*$"
)


def _detect_diff_or_conflict_fragments(code: str) -> list[dict[str, Any]]:
    matches = list(_DIFF_OR_CONFLICT_MARKER_RE.finditer(code or ""))
    if not matches:
        return []
    evidence_lines = []
    lines = (code or "").splitlines()
    for match in matches[:5]:
        lineno = (code[: match.start()].count("\n") + 1)
        line_text = lines[lineno - 1].strip() if 0 <= lineno - 1 < len(lines) else match.group(0).strip()
        evidence_lines.append(f"line {lineno}: {line_text}")
    return [
        {
            "severity": "critical",
            "category": "diff_marker_or_conflict_fragment",
            "message": "Generated code still contains SEARCH/REPLACE or merge-conflict marker fragments.",
            "evidence": "\n".join(evidence_lines),
            "repair_hint": "Resolve the patch into ordinary Python code and remove all diff/conflict marker lines before execution.",
            "autofixable": False,
        }
    ]


def _static_int_literal(node: ast.AST | None) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return int(node.value)
    return None


def _literal_sequence_len(node: ast.AST | None) -> int | None:
    if isinstance(node, (ast.List, ast.Tuple)):
        return len(node.elts)
    if isinstance(node, ast.Call) and node.args:
        call_name = _call_name(node.func)
        if call_name in {"array", "tensor", "asarray"}:
            return _literal_sequence_len(node.args[0])
    return None


def _assignment_targets(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.Assign):
        return list(node.targets)
    if isinstance(node, ast.AnnAssign):
        return [node.target]
    return []


def _target_name(target: ast.AST) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _detect_engineered_feature_dim_mismatch(code: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return []

    feature_lengths: list[tuple[int, ast.AST, str]] = []
    declared_dims: list[tuple[int, ast.AST, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            for target in _assignment_targets(node):
                name = (_target_name(target) or "").lower()
                if name in {"feature_names", "feature_cols", "feature_columns"}:
                    length = _literal_sequence_len(value)
                    if length is not None:
                        feature_lengths.append((length, node, name))
                if name in {"feature_dim", "input_dim", "num_features", "n_features", "feature_size"}:
                    dim = _static_int_literal(value)
                    if dim is not None:
                        declared_dims.append((dim, node, name))
        if isinstance(node, ast.FunctionDef) and "feature" in node.name.lower():
            for inner in ast.walk(node):
                if isinstance(inner, ast.Return):
                    length = _literal_sequence_len(inner.value)
                    if length is not None:
                        feature_lengths.append((length, inner, node.name))
        if isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if call_name in {"LayerNorm", "BatchNorm1d", "Linear"} and node.args:
                dim = _static_int_literal(node.args[0])
                if dim is not None:
                    declared_dims.append((dim, node, call_name))

    expected_lengths = {length for length, _node, _source in feature_lengths if length > 0}
    if len(expected_lengths) != 1:
        return []
    expected = next(iter(expected_lengths))
    mismatches = [
        (dim, node, source)
        for dim, node, source in declared_dims
        if dim > 0 and dim != expected
    ]
    if not mismatches:
        return []
    evidence_parts = [
        f"feature vector length evidence: {source} -> {expected}"
        for _length, _node, source in feature_lengths[:2]
    ]
    for dim, node, source in mismatches[:3]:
        evidence_parts.append(f"declared {source}={dim}: {_source_segment(code, node)}")
    return [
        {
            "severity": "critical",
            "category": "engineered_feature_dim_mismatch",
            "message": (
                f"Engineered feature vector has {expected} explicit values, but model/config code declares "
                f"a different feature dimension."
            ),
            "evidence": "\n".join(evidence_parts),
            "repair_hint": (
                "Use one source of truth for engineered features: derive feature_dim from len(feature_names) "
                "or a sample feature tensor shape, then assert the data bundle and model input dimensions match."
            ),
            "autofixable": False,
        }
    ]


def _torch_runtime_info(*, device_index: int) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "import_error": str(exc)}

    cuda_available = bool(torch.cuda.is_available())
    info: dict[str, Any] = {
        "available": True,
        "version": getattr(torch, "__version__", None),
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "cuda_available": cuda_available,
        "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
    }
    if cuda_available:
        try:
            props = torch.cuda.get_device_properties(device_index)
            info["device"] = {
                "index": device_index,
                "name": torch.cuda.get_device_name(device_index),
                "total_vram_mb": int(props.total_memory / (1024 * 1024)),
                "compute_capability": f"{props.major}.{props.minor}",
            }
        except Exception as exc:
            info["device_error"] = str(exc)
    return info


def _torch_scheduler_signatures() -> dict[str, dict[str, Any]]:
    try:
        import torch
    except Exception:
        return {}

    signatures: dict[str, dict[str, Any]] = {}
    lr_scheduler = getattr(getattr(torch, "optim", None), "lr_scheduler", None)
    if lr_scheduler is None:
        return signatures
    for name in _SCHEDULER_NAMES:
        cls = getattr(lr_scheduler, name, None)
        if cls is None:
            continue
        try:
            signature = inspect.signature(cls)
        except (TypeError, ValueError):
            continue
        signatures[name] = {
            "signature": str(signature),
            "parameters": list(signature.parameters.keys()),
        }
    return signatures


def _package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
            continue
        except importlib.metadata.PackageNotFoundError:
            pass
        except Exception:
            pass
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "")) or None
        except Exception:
            versions[name] = None
    return versions


def _precision_checks() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    try:
        import torch
    except Exception as exc:
        return {"torch_import_error": str(exc)}

    try:
        torch.ones(1, dtype=torch.bfloat16).cpu().numpy()
    except Exception as exc:
        checks["bf16_cpu_numpy_supported"] = False
        checks["bf16_cpu_numpy_error"] = f"{type(exc).__name__}: {exc}"
    else:
        checks["bf16_cpu_numpy_supported"] = True

    checks["low_precision_numpy_export_policy"] = {
        "rule": "Low-precision model outputs may be used for forward/loss, but validation metrics, sklearn/NumPy/pandas, and submission exports must convert prediction/logit/probability tensors to float32 before CPU/NumPy.",
        "safe_pattern": "tensor.detach().to(torch.float32).cpu().numpy()",
        "applies_to": ["bf16", "fp16", "fp8", "mxfp8", "nvfp4", "mxfp4", "transformer_engine", "torchao.float8"],
    }
    checks["pytorch_float8_dtypes"] = _torch_float8_checks(torch)

    checks["autocast_available"] = hasattr(torch, "autocast") or hasattr(getattr(torch, "amp", None), "autocast")
    if hasattr(torch.cuda, "is_bf16_supported"):
        try:
            checks["cuda_bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        except Exception as exc:
            checks["cuda_bf16_supported_error"] = str(exc)
    return checks


def _torch_float8_checks(torch_module: Any) -> dict[str, dict[str, Any]]:
    dtype_names = ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz")
    results: dict[str, dict[str, Any]] = {}
    for name in dtype_names:
        dtype = getattr(torch_module, name, None)
        if dtype is None:
            results[name] = {"available": False}
            continue
        entry: dict[str, Any] = {"available": True}
        try:
            torch_module.empty(1, dtype=dtype).cpu().numpy()
        except Exception as exc:
            entry["cpu_numpy_supported"] = False
            entry["cpu_numpy_error"] = f"{type(exc).__name__}: {exc}"
        else:
            entry["cpu_numpy_supported"] = True
        results[name] = entry
    return results


def _detect_invalid_torch_scheduler_kwargs(code: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(code or "")
    except SyntaxError as exc:
        return [
            {
                "severity": "critical",
                "category": "syntax_error",
                "message": f"Generated code has a SyntaxError: {exc.msg}",
                "evidence": f"line {exc.lineno}: {exc.text or ''}".strip(),
                "repair_hint": "Fix Python syntax before execution.",
                "autofixable": False,
            }
        ]

    scheduler_signatures = _torch_scheduler_signatures()
    issues: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        scheduler_name = _call_name(node.func)
        if scheduler_name not in scheduler_signatures:
            continue
        allowed = set(scheduler_signatures[scheduler_name].get("parameters") or [])
        invalid = [kw.arg for kw in node.keywords if kw.arg and kw.arg not in allowed]
        if not invalid:
            continue
        hint = "Use valid parameters for torch.optim.lr_scheduler.%s: %s." % (
            scheduler_name,
            ", ".join(scheduler_signatures[scheduler_name].get("parameters") or []),
        )
        if "T_eta_min" in invalid:
            hint = "Replace T_eta_min with eta_min for CosineAnnealingLR."
        issues.append(
            {
                "severity": "critical",
                "category": "invalid_torch_scheduler_argument",
                "message": f"{scheduler_name} received unsupported keyword argument(s): {', '.join(invalid)}.",
                "evidence": _source_segment(code, node),
                "repair_hint": hint,
                "autofixable": False,
            }
        )
    return issues


def _detect_low_precision_numpy_conversion(code: str) -> list[dict[str, Any]]:
    code_text = code or ""
    lowered = code_text.lower()
    if not _uses_low_precision_export_context(code_text):
        return []

    issues: list[dict[str, Any]] = []
    bf16_context = any(token in lowered for token in _BF16_MARKERS)
    for finding in _iter_low_precision_numpy_findings(code_text):
        category = "bf16_numpy_conversion" if bf16_context else "low_precision_numpy_export"
        precision_label = "BF16" if bf16_context else "low-precision"
        repair_hint = (
            "Leave autocast/low-precision contexts before validation/export and cast "
            "predictions/logits/probabilities to float32 before CPU/NumPy, e.g. "
            "preds.detach().to(torch.float32).cpu().numpy()."
        )
        issues.append(
            {
                "severity": "critical",
                "category": category,
                "message": f"{precision_label} prediction/logit/probability tensor is converted directly with .cpu().numpy(), which can fail during validation, metric calculation, or submission export.",
                "evidence": finding["evidence"],
                "repair_hint": repair_hint,
                "autofixable": True,
            }
        )
    return issues


def _uses_low_precision_export_context(code: str) -> bool:
    lowered = (code or "").lower()
    return any(token in lowered for token in _LOW_PRECISION_MARKERS)


def _iter_low_precision_numpy_findings(code: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return []

    findings: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_numpy_call(node):
            continue
        cpu_call = node.func.value
        if not _is_zero_arg_method_call(cpu_call, "cpu"):
            continue
        tensor_expr = cpu_call.func.value
        expr_source = _source_segment(code, tensor_expr) or ""
        full_source = _source_segment(code, node) or f"{expr_source}.cpu().numpy()"
        if _has_float32_cast_before_numpy_expr(tensor_expr, expr_source):
            continue
        if not _is_prediction_like_export(expr_source):
            continue
        findings.append(
            {
                "node": node,
                "tensor_expr": tensor_expr,
                "expr_source": expr_source,
                "evidence": f"line {getattr(node, 'lineno', '?')}: {full_source.strip()}",
            }
        )
    return findings


def _low_precision_numpy_replacements(code: str) -> list[tuple[int, int, str]]:
    if not _uses_low_precision_export_context(code):
        return []
    offsets = _line_offsets(code)
    replacements: list[tuple[int, int, str]] = []
    for finding in _iter_low_precision_numpy_findings(code):
        tensor_expr = finding["tensor_expr"]
        span = _node_span(tensor_expr, offsets)
        if span is None:
            continue
        expr_source = finding["expr_source"]
        replacement = _float32_export_source(expr_source)
        if replacement and replacement != expr_source:
            replacements.append((span[0], span[1], replacement))
    return replacements


def _repair_known_hf_model_source_ids(code: str) -> tuple[str, int]:
    repaired = str(code or "")
    replacement_count = 0
    for invalid, valid in _KNOWN_HF_REPO_ID_REPAIRS.items():
        occurrences = repaired.count(invalid)
        if occurrences:
            repaired = repaired.replace(invalid, valid)
            replacement_count += occurrences

    source_replacements = _known_hf_branch_source_replacements(repaired)
    for start, end, replacement in sorted(source_replacements, key=lambda item: item[0], reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    replacement_count += len(source_replacements)

    return repaired, replacement_count


def _known_hf_branch_source_replacements(code: str) -> list[tuple[int, int, str]]:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return []

    assignments = _simple_string_assignments(tree)
    offsets = _line_offsets(code)
    replacements: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "from_pretrained" or not node.args:
            continue
        first_arg = node.args[0]
        derived = _hf_model_id_built_from_branch_key(first_arg, assignments)
        if derived is None:
            continue
        provider, _, branch_value = derived
        repaired_branch = _KNOWN_HF_BRANCH_VALUE_REPAIRS.get(branch_value)
        if not repaired_branch:
            continue
        span = _node_span(first_arg, offsets)
        if span is not None:
            replacements.append((span[0], span[1], repr(f"{provider}{repaired_branch}")))
    return replacements


def _detect_hf_model_source_identifier_issues(code: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return []

    assignments = _simple_string_assignments(tree)
    issues: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "from_pretrained" or not node.args:
            continue

        first_arg = node.args[0]
        direct_id = _constant_string(first_arg)
        if direct_id in _KNOWN_HF_REPO_ID_REPAIRS:
            issues.append(
                {
                    "severity": "critical",
                    "category": "invalid_huggingface_model_id",
                    "message": f"Hugging Face model id `{direct_id}` is a known invalid transformed id.",
                    "evidence": _source_segment(code, node),
                    "repair_hint": f"Use `{_KNOWN_HF_REPO_ID_REPAIRS[direct_id]}` exactly as the external pretrained model id.",
                    "autofixable": True,
                }
            )
            continue

        derived = _hf_model_id_built_from_branch_key(first_arg, assignments)
        if derived is None:
            continue
        provider, branch_name, branch_value = derived
        repaired_branch = _KNOWN_HF_BRANCH_VALUE_REPAIRS.get(branch_value)
        branch_looks_sanitized = "_" in branch_value and "/" not in branch_value
        if not repaired_branch and not branch_looks_sanitized:
            continue
        expected = f"{provider}{repaired_branch}" if repaired_branch else None
        hint = (
            f"Keep the exact external model id in `PRETRAINED_MODEL_ID`/`MODEL_ID` and pass that to from_pretrained; "
            f"`{branch_name}` is a scheduler/profile key, not a repo id."
        )
        if expected:
            hint = f"Use `PRETRAINED_MODEL_ID = \"{expected}\"` for from_pretrained and keep `{branch_name}` only for scheduler/profile reuse."
        issues.append(
            {
                "severity": "critical",
                "category": "derived_huggingface_model_id_from_scheduler_branch",
                "message": (
                    f"Hugging Face model id is derived from `{branch_name}` value `{branch_value}`, "
                    "which looks like a sanitized scheduler/profile key and may not be a valid repo id."
                ),
                "evidence": _source_segment(code, node),
                "repair_hint": hint,
                "autofixable": bool(repaired_branch),
            }
        )
    return issues


def _detect_zip_extractall_directory_mismatch(code: str) -> list[dict[str, Any]]:
    findings = _zip_extractall_target_replacements(code)
    if not findings:
        return []
    return [
        {
            "severity": "critical",
            "category": "zip_extractall_directory_mismatch",
            "message": (
                "Generated data preparation extracts train/test zip files into the shared working "
                "directory while later searching dedicated TRAIN_DIR/TEST_DIR image folders."
            ),
            "evidence": "zip_ref.extractall(WORKING_DIR)",
            "repair_hint": "Extract TRAIN_ZIP_PATH into TRAIN_DIR and TEST_ZIP_PATH into TEST_DIR, or search WORKING_DIR directly after extraction.",
            "autofixable": True,
        }
    ]


def _zip_extractall_target_replacements(code: str) -> list[tuple[int, int, str]]:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return []

    offsets = _line_offsets(code)
    replacements: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        expected_target_by_alias = _zipfile_expected_extract_targets(node, code)
        if not expected_target_by_alias:
            continue
        for child in ast.walk(ast.Module(body=list(node.body), type_ignores=[])):
            if not isinstance(child, ast.Call):
                continue
            if not isinstance(child.func, ast.Attribute) or child.func.attr != "extractall":
                continue
            if not isinstance(child.func.value, ast.Name):
                continue
            expected_target = expected_target_by_alias.get(child.func.value.id)
            if not expected_target or not child.args:
                continue
            first_arg = child.args[0]
            if not _is_working_extract_target(first_arg):
                continue
            span = _node_span(first_arg, offsets)
            if span is not None:
                replacements.append((span[0], span[1], expected_target))
    return replacements


def _zipfile_expected_extract_targets(node: ast.With, code: str) -> dict[str, str]:
    expected: dict[str, str] = {}
    for item in node.items:
        context = item.context_expr
        optional = item.optional_vars
        if not isinstance(optional, ast.Name):
            continue
        expected_target = _zipfile_context_expected_extract_target(context, code)
        if expected_target:
            expected[optional.id] = expected_target
    return expected


def _zipfile_context_expected_extract_target(node: ast.AST, code: str) -> str | None:
    if not isinstance(node, ast.Call) or not node.args:
        return None
    func_name = _call_name(node.func)
    if func_name != "ZipFile":
        return None
    first_arg = node.args[0]
    if isinstance(first_arg, ast.Name) and first_arg.id in {"TRAIN_ZIP_PATH", "TEST_ZIP_PATH"}:
        return _zip_extract_target_name(first_arg.id, code)
    if isinstance(first_arg, ast.Attribute) and first_arg.attr in {"TRAIN_ZIP_PATH", "TEST_ZIP_PATH"}:
        qualifier = _source_segment(code, first_arg.value)
        if not qualifier:
            return None
        target_name = _zip_extract_target_name(first_arg.attr, code, qualifier=qualifier)
        return f"{qualifier}.{target_name}" if target_name else None
    return None


def _zip_extract_target_name(archive_name: str, code: str, *, qualifier: str | None = None) -> str | None:
    prefix = f"{qualifier}." if qualifier else ""
    if archive_name == "TRAIN_ZIP_PATH":
        if f"{prefix}TRAIN_DATA_PATH" in (code or "") or (qualifier and "TRAIN_DATA_PATH" in (code or "")):
            return "TRAIN_DATA_PATH"
        return "TRAIN_DIR"
    if archive_name == "TEST_ZIP_PATH":
        if f"{prefix}TEST_DATA_PATH" in (code or "") or (qualifier and "TEST_DATA_PATH" in (code or "")):
            return "TEST_DATA_PATH"
        return "TEST_DIR"
    return None


def _is_working_extract_target(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in {"WORKING_DIR", "WORKING_PATH"}
    if isinstance(node, ast.Attribute):
        return node.attr in {"WORKING_DIR", "WORKING_PATH"}
    return False


def _simple_string_assignments(tree: ast.AST) -> dict[str, str]:
    values: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = _constant_string(node.value)
            if value is not None:
                values[node.targets[0].id] = value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            value = _constant_string(node.value)
            if value is not None:
                values[node.target.id] = value
    return values


def _hf_model_id_built_from_branch_key(node: ast.AST, assignments: dict[str, str]) -> tuple[str, str, str] | None:
    if isinstance(node, ast.JoinedStr):
        provider = ""
        branch_name = None
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                provider += value.value
            elif isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Name):
                if value.value.id in _SCHEDULER_BRANCH_KEY_NAMES:
                    branch_name = value.value.id
                    break
        if branch_name and provider.endswith("/") and assignments.get(branch_name):
            return provider, branch_name, assignments[branch_name]

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        parts = _flatten_string_add(node)
        if len(parts) == 2 and isinstance(parts[0], str) and isinstance(parts[1], ast.Name):
            branch_name = parts[1].id
            if parts[0].endswith("/") and branch_name in _SCHEDULER_BRANCH_KEY_NAMES and assignments.get(branch_name):
                return parts[0], branch_name, assignments[branch_name]
    return None


def _flatten_string_add(node: ast.AST) -> list[str | ast.Name]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _flatten_string_add(node.left) + _flatten_string_add(node.right)
    constant = _constant_string(node)
    if constant is not None:
        return [constant]
    if isinstance(node, ast.Name):
        return [node]
    return [""]


def _constant_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _line_offsets(code: str) -> list[int]:
    offsets = [0]
    total = 0
    for line in code.splitlines(keepends=True):
        total += len(line)
        offsets.append(total)
    return offsets


def _node_span(node: ast.AST, offsets: list[int]) -> tuple[int, int] | None:
    lineno = getattr(node, "lineno", None)
    col_offset = getattr(node, "col_offset", None)
    end_lineno = getattr(node, "end_lineno", None)
    end_col_offset = getattr(node, "end_col_offset", None)
    if None in (lineno, col_offset, end_lineno, end_col_offset):
        return None
    if int(lineno) - 1 >= len(offsets) or int(end_lineno) - 1 >= len(offsets):
        return None
    start = offsets[int(lineno) - 1] + int(col_offset)
    end = offsets[int(end_lineno) - 1] + int(end_col_offset)
    return start, end


def _is_numpy_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Attribute) and node.func.attr == "numpy"


def _is_zero_arg_method_call(node: ast.AST, method_name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method_name
        and not node.args
        and not node.keywords
    )


def _is_prediction_like_export(expr_source: str) -> bool:
    compact = re.sub(r"\s+", "", str(expr_source or "")).lower()
    has_prediction_token = any(token in compact for token in _PREDICTION_EXPORT_TOKENS)
    if any(token in compact for token in _NON_PREDICTION_EXPORT_TOKENS) and not has_prediction_token:
        return False
    return has_prediction_token


def _has_float32_cast_before_numpy_expr(expr: ast.AST, expr_source: str) -> bool:
    if _has_float32_cast_before_numpy(expr_source):
        return True
    for node in ast.walk(expr):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr in {"float", "double"} and not node.args and not node.keywords:
            return True
        if node.func.attr in {"to", "type"} and _call_mentions_float32(node):
            return True
    return False


def _call_mentions_float32(node: ast.Call) -> bool:
    source_bits = []
    for arg in node.args:
        source_bits.append(ast.unparse(arg).lower())
    for keyword in node.keywords:
        if keyword.value is not None:
            source_bits.append(ast.unparse(keyword.value).lower())
    combined = " ".join(source_bits)
    return "float32" in combined or "torch.float" in combined


def _float32_export_source(expr_source: str) -> str:
    stripped = str(expr_source or "").strip()
    if not stripped:
        return stripped
    compact = re.sub(r"\s+", "", stripped).lower()
    if ".detach(" in compact:
        return f"{stripped}.to(torch.float32)"
    return f"{stripped}.detach().to(torch.float32)"


def _has_torch_import(code: str) -> bool:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return "import torch" in (code or "") or "from torch" in (code or "")
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.module == "torch":
            return True
    return False


def _insert_torch_import(code: str) -> str:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return "import torch\n" + (code or "")

    insert_line = 0
    body = list(tree.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
        insert_line = int(getattr(body[0], "end_lineno", body[0].lineno))
        body = body[1:]
    for node in body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            insert_line = int(getattr(node, "end_lineno", node.lineno))
            continue
        break

    lines = (code or "").splitlines(keepends=True)
    if not lines:
        return "import torch\n"
    insert_idx = max(0, min(insert_line, len(lines)))
    lines.insert(insert_idx, "import torch\n")
    return "".join(lines)


def _detect_deprecated_cuda_amp(code: str) -> list[dict[str, Any]]:
    if "torch.cuda.amp" not in (code or ""):
        return []
    return [
        {
            "severity": "warning",
            "category": "deprecated_cuda_amp_api",
            "message": "Code uses deprecated torch.cuda.amp APIs.",
            "evidence": "torch.cuda.amp",
            "repair_hint": "Prefer torch.amp.autocast('cuda', ...) and torch.amp.GradScaler('cuda', ...) when available.",
            "autofixable": False,
        }
    ]


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _source_segment(code: str, node: ast.AST) -> str:
    try:
        return (ast.get_source_segment(code, node) or "").strip()
    except Exception:
        return ""


def _has_float32_cast_before_numpy(expr: str) -> bool:
    compact = re.sub(r"\s+", "", expr or "").lower()
    safe_tokens = (
        ".float().cpu().numpy(",
        ".to(torch.float32).cpu().numpy(",
        ".to(dtype=torch.float32).cpu().numpy(",
        ".type(torch.float32).cpu().numpy(",
    )
    return any(token in compact for token in safe_tokens)


def _compatibility_summary(critical_count: int, warning_count: int) -> str:
    if critical_count:
        return f"Found {critical_count} critical generated-code compatibility issue(s) and {warning_count} warning(s)."
    if warning_count:
        return f"Found {warning_count} generated-code compatibility warning(s)."
    return "No known generated-code compatibility issues detected."
