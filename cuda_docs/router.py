"""Deterministic role policy, CUDA incident taxonomy, redaction, and keys."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit
import hashlib
import json
import re

from ..backend_mode import normalize_packing_backend
from ..cuda_mcp_bridge import ERROR_TOPIC_PATTERNS
from .models import (
    CUDA_DOCS_CACHE_PREFIX,
    CUDA_DOCS_QUERY_TEMPLATE_VERSION,
    CUDA_DOCS_SCHEMA_VERSION,
    CudaDocsApplicability,
    CudaDocsRequest,
    RouteDecision,
    RouteOutcome,
)

_NOT_APPLICABLE = re.compile(
    r"\b(?:SyntaxError|IndentationError|ModuleNotFoundError|ImportError|FileNotFoundError|"
    r"KeyError|submission(?:\s+format)?|data\s+leakage|metric\s+mismatch|"
    r"No such file or directory|CSV|parquet)\b",
    re.I,
)
_CUDA_EXACT_QUESTION = re.compile(
    r"\b(?:cuda|cublas|cudnn|nvidia\s+mps|torch\.cuda|sm_?\d+)\b",
    re.I,
)
_CUDA_API_SYMBOL = re.compile(
    r"\b(?:torch\.cuda(?:\.[A-Za-z_][A-Za-z0-9_]*)+|"
    r"cuda[A-Z][A-Za-z0-9_]*|cublas[A-Z][A-Za-z0-9_]*|"
    r"cudnn[A-Z][A-Za-z0-9_]*)\b"
)
_MPS_FAILURE = re.compile(
    r"\b(?:MPS server|MPS client|CUDA_MPS_|nvidia-cuda-mps-control|MPS.*(?:failed|error|unavailable))\b",
    re.I,
)
_PATH = re.compile(r"(?:(?:[A-Za-z]:\\|/)(?:[^\s:'\"<>|]+[\\/])*[^\s:'\"<>|]*)")
_SECRET = re.compile(
    r"(?i)\b(api[_-]?key|authorization|bearer|cookie|password|secret|token)\b"
    r"\s*[:=]?\s*([^\s,;]+)"
)
_HOSTNAME = re.compile(
    r"\b(?=.{1,253}\b)(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+"
    r"(?:local|internal|lan|corp|cluster)\b",
    re.I,
)
_DATASET_FILE = re.compile(
    r"\b[^\s/\\]+\.(?:csv|parquet|jsonl|feather|pkl|pickle|npy|npz|pt|pth)\b",
    re.I,
)
_JOB_IDENTIFIER = re.compile(
    r"(?i)\b(job|node|run|dataset|repository|repo)[-_ ]?(?:id|name)?\s*[:=]\s*"
    r"[A-Za-z0-9_.:@/-]+"
)
_URL = re.compile(r"https?://[^\s<>\"']+", re.I)


def normalize_role(role: object) -> str:
    return str(role or "").strip().lower().replace("-", "_")


def route_request(
    *,
    role: str,
    error_text: str = "",
    question: str = "",
    topic: str | None = None,
    profile_symptoms: list[str] | tuple[str, ...] = (),
) -> RouteDecision:
    """Classify without touching any cache or network resource."""

    normalized_role = normalize_role(role)
    combined = " ".join(part for part in (error_text, question) if part).strip()
    selected_topic = str(topic or "").strip()
    signature = "explicit_topic" if selected_topic else ""
    if not selected_topic:
        for pattern, mapped_topic in ERROR_TOPIC_PATTERNS:
            if re.search(pattern, combined, re.I):
                selected_topic = mapped_topic
                signature = _signature_for_pattern(pattern)
                break
    if not selected_topic and _MPS_FAILURE.search(combined):
        selected_topic = "diagnose NVIDIA MPS client and server compatibility failures"
        signature = "mps_client_server_failure"
    if not selected_topic and combined and _NOT_APPLICABLE.search(combined):
        return RouteDecision(
            RouteOutcome.NOT_APPLICABLE,
            normalized_role,
            reason="non_cuda_failure_taxonomy",
        )
    if not selected_topic and question and _CUDA_EXACT_QUESTION.search(question):
        selected_topic = (
            "verify installed-version compatibility for the referenced CUDA API"
        )
        signature = "exact_cuda_api_question"

    symptoms = {str(item).strip().lower() for item in profile_symptoms}
    if (
        not selected_topic
        and normalized_role == "improve"
        and symptoms.intersection(
            {
                "oom",
                "out_of_memory",
                "gpu_memory_pressure",
                "unsupported_precision",
                "low_sm_utilization",
                "cuda_bottleneck",
            }
        )
    ):
        selected_topic = "reduce measured CUDA memory pressure or execution bottlenecks"
        signature = "confirmed_cuda_profile_symptom"

    if not selected_topic:
        return RouteDecision(
            RouteOutcome.NOT_APPLICABLE,
            normalized_role,
            reason="no_allowlisted_cuda_trigger",
        )
    if normalized_role not in {"debug", "draft", "improve", "code_review"}:
        return RouteDecision(
            RouteOutcome.NOT_APPLICABLE,
            normalized_role,
            reason="role_consumes_existing_local_hardware_context",
        )
    return RouteDecision(
        RouteOutcome.ELIGIBLE,
        normalized_role,
        topic=_normalize_topic(selected_topic),
        error_signature_class=signature or "cuda_incident",
        sanitized_error_excerpt=sanitize_error_excerpt(combined),
        reason="allowlisted_cuda_route",
    )


def sanitize_error_excerpt(value: str, *, max_chars: int = 400) -> str:
    """Keep only an incident/API signature, never a source-code line."""

    text = str(value or "")
    fragments: list[str] = []
    for pattern, _topic in ERROR_TOPIC_PATTERNS:
        fragments.extend(match.group(0) for match in re.finditer(pattern, text, re.I))
    if _MPS_FAILURE.search(text):
        fragments.append("NVIDIA MPS client/server failure")
    api_symbols = _CUDA_API_SYMBOL.findall(text)
    fragments.extend(api_symbols[:8])
    if fragments:
        text = "; ".join(dict.fromkeys(fragments))
    else:
        lines = text.splitlines()
        text = lines[-1] if lines else ""
    text = _SECRET.sub(lambda match: f"{match.group(1)}=<redacted>", text)
    text = _URL.sub(_redact_url, text)
    text = _PATH.sub("<path>", text)
    text = _DATASET_FILE.sub("<dataset-file>", text)
    text = _JOB_IDENTIFIER.sub(lambda match: match.group(1) + "=<redacted>", text)
    text = _HOSTNAME.sub("<hostname>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[: max(0, int(max_chars))]


def _redact_url(match: re.Match[str]) -> str:
    try:
        parsed = urlsplit(match.group(0))
        host = parsed.hostname or "<host>"
        if host not in {
            "docs.nvidia.com",
            "developer.nvidia.com",
            "github.com",
            "nvidia.github.io",
        }:
            host = "<host>"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except ValueError:
        return "<url>"


def _normalize_topic(topic: str) -> str:
    topic = sanitize_error_excerpt(topic, max_chars=240)
    return re.sub(r"\s+", " ", topic).strip().rstrip(".")


def _signature_for_pattern(pattern: str) -> str:
    lowered = pattern.lower()
    if "outofmemory" in lowered or "out of memory" in lowered:
        return "cuda_oom"
    if "cublas" in lowered:
        return "cublas_failure"
    if "cudnn" in lowered:
        return "cudnn_failure"
    if "device-side" in lowered:
        return "device_side_assert"
    if "kernel image" in lowered:
        return "kernel_architecture_mismatch"
    if "same device" in lowered:
        return "cuda_device_mismatch"
    return "cuda_incident"


def canonical_key_payload(
    *,
    topic: str,
    error_signature_class: str,
    applicability: CudaDocsApplicability,
) -> dict[str, Any]:
    backend = normalize_packing_backend(applicability.backend_mode)
    return {
        "schema_version": CUDA_DOCS_SCHEMA_VERSION,
        "query_template_version": CUDA_DOCS_QUERY_TEMPLATE_VERSION,
        "normalized_topic": _normalize_topic(topic),
        "error_signature_class": str(error_signature_class or "cuda_incident"),
        "gpu_architecture": applicability.gpu_architecture,
        "compute_capability": applicability.compute_capability,
        "driver_major_minor": applicability.driver_major_minor,
        "cuda_major_minor": applicability.cuda_major_minor,
        "framework": applicability.framework,
        "framework_major_minor": applicability.framework_major_minor,
        "backend_mode": backend,
        "backend_config_hash": applicability.backend_config_hash,
        "remote_tool_schema_hash": applicability.remote_tool_schema_hash,
    }


def canonical_cache_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return f"{CUDA_DOCS_CACHE_PREFIX}:{hashlib.sha256(encoded).hexdigest()}"


def build_request(
    decision: RouteDecision,
    applicability: CudaDocsApplicability,
) -> CudaDocsRequest:
    if decision.outcome != RouteOutcome.ELIGIBLE or not decision.topic:
        raise ValueError("cannot build a CUDA docs request from an ineligible route")
    payload = canonical_key_payload(
        topic=decision.topic,
        error_signature_class=decision.error_signature_class,
        applicability=applicability,
    )
    cache_key = canonical_cache_key(payload)
    query_parts = [decision.topic + "."]
    if applicability.gpu_architecture or applicability.compute_capability:
        query_parts.append(
            "Target GPU architecture {} with compute capability {}.".format(
                applicability.gpu_architecture or "unknown",
                applicability.compute_capability or "unknown",
            )
        )
    query_parts.append(
        "Installed NVIDIA driver {}, CUDA {}, {} {}.".format(
            applicability.driver_major_minor or "unknown",
            applicability.cuda_major_minor or "unknown",
            applicability.framework or "pytorch",
            applicability.framework_major_minor or "unknown",
        )
    )
    query_parts.append(
        "Execution backend {} with runner contract {}; independent jobs must not manage "
        "scheduler-owned services or cross-job CUDA state; return job-code guidance only.".format(
            normalize_packing_backend(applicability.backend_mode),
            applicability.runner_contract,
        )
    )
    if decision.sanitized_error_excerpt:
        query_parts.append(
            "Sanitized error signature: " + decision.sanitized_error_excerpt + "."
        )
    query_parts.append(
        "Return current first-party NVIDIA documentation context, exact source URLs, and relevant examples."
    )
    return CudaDocsRequest(
        role=decision.role,
        topic=decision.topic,
        error_signature_class=decision.error_signature_class,
        sanitized_error_excerpt=decision.sanitized_error_excerpt,
        applicability=applicability,
        canonical_key=cache_key,
        query=" ".join(query_parts),
    )


def applicability_from_facts(
    facts: Any,
    *,
    backend_mode: str,
    runner_contract: str,
    remote_tool_schema_hash: str = "unknown",
) -> CudaDocsApplicability:
    return CudaDocsApplicability(
        gpu_architecture=str(getattr(facts, "gpu_architecture", "") or "unknown"),
        compute_capability=str(getattr(facts, "capability_str", "") or "unknown"),
        driver_major_minor=_major_minor(getattr(facts, "driver_version", "")),
        cuda_major_minor=_major_minor(getattr(facts, "cuda_version", "")),
        framework="pytorch",
        framework_major_minor=_major_minor(getattr(facts, "torch_version", "")),
        backend_mode=normalize_packing_backend(backend_mode),
        backend_config_hash=str(getattr(facts, "backend_config_hash", "") or "unknown"),
        runner_contract=runner_contract,
        remote_tool_schema_hash=str(remote_tool_schema_hash or "unknown"),
    )


def _major_minor(value: object) -> str:
    match = re.search(r"(\d+)\.(\d+)", str(value or ""))
    return f"{match.group(1)}.{match.group(2)}" if match else "unknown"
