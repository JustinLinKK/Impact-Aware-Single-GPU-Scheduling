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
    issues.extend(_detect_bf16_numpy_conversion(code_text))
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

    checks["autocast_available"] = hasattr(torch, "autocast") or hasattr(getattr(torch, "amp", None), "autocast")
    if hasattr(torch.cuda, "is_bf16_supported"):
        try:
            checks["cuda_bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        except Exception as exc:
            checks["cuda_bf16_supported_error"] = str(exc)
    return checks


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
            }
        )
    return issues


def _detect_bf16_numpy_conversion(code: str) -> list[dict[str, Any]]:
    code_text = code or ""
    lowered = code_text.lower()
    if not any(token in lowered for token in ("bfloat16", "bf16", "torch.bfloat16")):
        return []

    issues: list[dict[str, Any]] = []
    pattern = re.compile(r"(?P<expr>[^\n]{0,160}\.cpu\(\)\.numpy\()")
    for match in pattern.finditer(code_text):
        expr = match.group("expr")
        if _has_float32_cast_before_numpy(expr):
            continue
        line_no = code_text.count("\n", 0, match.start()) + 1
        issues.append(
            {
                "severity": "critical",
                "category": "bf16_numpy_conversion",
                "message": "BF16 tensor is converted directly with .cpu().numpy(), which can fail during validation or metric calculation.",
                "evidence": f"line {line_no}: {expr.strip()}",
                "repair_hint": "Leave autocast before validation/export and cast predictions/logits/probabilities to float32 before CPU/NumPy, e.g. preds.float().cpu().numpy().",
            }
        )
    return issues


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
