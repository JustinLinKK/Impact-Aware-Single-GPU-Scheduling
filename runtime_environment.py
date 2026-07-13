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


def validate_generated_training_code(code: str, *, stage: str = "code_review") -> dict[str, Any]:
    """Statically flag generated-code patterns known to fail in this runtime."""
    code_text = str(code or "")
    issues: list[dict[str, Any]] = []

    issues.extend(_detect_invalid_torch_scheduler_kwargs(code_text))
    issues.extend(_detect_low_precision_numpy_conversion(code_text))
    issues.extend(_detect_deprecated_cuda_amp(code_text))

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


def repair_generated_training_code(code: str, *, stage: str = "code_review") -> dict[str, Any]:
    """Deterministically repair prediction-like low-precision Tensor -> NumPy exports."""
    original_code = str(code or "")
    replacements = _low_precision_numpy_replacements(original_code)
    repaired_code = original_code
    for start, end, replacement in sorted(replacements, key=lambda item: item[0], reverse=True):
        repaired_code = repaired_code[:start] + replacement + repaired_code[end:]
    if replacements and "torch.float32" in repaired_code and not _has_torch_import(repaired_code):
        repaired_code = _insert_torch_import(repaired_code)

    validation = validate_generated_training_code(repaired_code, stage=stage)
    return {
        "code": repaired_code,
        "changed": repaired_code != original_code,
        "replacement_count": len(replacements),
        "stage": stage,
        "validation": validation,
    }


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
