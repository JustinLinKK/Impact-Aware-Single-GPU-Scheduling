"""Code-knowledge vector record loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
from urllib.parse import urlparse
from urllib.parse import urlunparse
import re

import yaml

from localml_scheduler.hardware_features.records import HARDWARE_FEATURE_SCHEMA_VERSION, validate_feature_record
from localml_scheduler.backend_mode import (
    BACKEND_NEUTRAL,
    PACKED_BACKEND_MODES,
    RETIRED_BACKEND_MODES,
    RUNNER_CONTRACT_SUBPROCESS_V1,
)


CODE_DOC_SCHEMA_VERSION = "code_doc_chunk_v1"
OPTIMIZATION_RECIPE_SCHEMA_VERSION = "optimization_recipe_chunk_v1"
API_SYMBOL_SCHEMA_VERSION = "api_symbol_chunk_v1"
BACKEND_GUIDANCE_SCHEMA_VERSION = "backend_guidance_rule_v1"

_RECORD_ID_KEYS = {
    CODE_DOC_SCHEMA_VERSION: "chunk_id",
    OPTIMIZATION_RECIPE_SCHEMA_VERSION: "recipe_id",
    API_SYMBOL_SCHEMA_VERSION: "api_symbol_id",
    BACKEND_GUIDANCE_SCHEMA_VERSION: "rule_id",
}

_COLLECTION_BY_SCHEMA = {
    CODE_DOC_SCHEMA_VERSION: "code_doc_chunks",
    OPTIMIZATION_RECIPE_SCHEMA_VERSION: "optimization_recipe_chunks",
    API_SYMBOL_SCHEMA_VERSION: "api_symbol_chunks",
    BACKEND_GUIDANCE_SCHEMA_VERSION: "backend_guidance_rules",
}

BACKEND_GUIDANCE_MODES = frozenset(
    {BACKEND_NEUTRAL, "exclusive", *PACKED_BACKEND_MODES}
)
BACKEND_GUIDANCE_PIPELINE_STAGES = frozenset(
    {"model_design", "datatype_precision", "training_evaluation"}
)
BACKEND_GUIDANCE_RULE_TYPES = frozenset(
    {"invariant", "safety", "recommendation", "heuristic"}
)
BACKEND_GUIDANCE_OWNERS = frozenset({"scheduler", "runner", "job_code"})
BACKEND_GUIDANCE_STRENGTHS = frozenset(
    {"hard", "preferred", "informational"}
)
BACKEND_GUIDANCE_TRANSFERABILITY = frozenset(
    {"backend_neutral", "exact_backend", "exclusive_baseline"}
)
BACKEND_GUIDANCE_REVIEW_STATUSES = frozenset({"draft", "reviewed", "retired"})
CUDA_DOC_SUPPORT_STATUSES = frozenset(
    {
        "functionally_supported",
        "natively_accelerated",
        "unsupported",
        "unknown_pending_local_verification",
    }
)
CUDA_DOC_APPLICABILITY_FIELDS = frozenset(
    {
        "gpu_architecture",
        "compute_capability",
        "driver_major_minor",
        "cuda_major_minor",
        "framework",
        "framework_major_minor",
        "backend_mode",
        "backend_config_hash",
        "runner_contract",
        "remote_tool_schema_hash",
    }
)


class CodeKnowledgeRecordError(ValueError):
    """Raised when a code-knowledge vector record is invalid."""


def _as_string_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key) or []
    if not isinstance(value, list):
        raise CodeKnowledgeRecordError(f"{key} must be a list")
    return [str(item).strip() for item in value if str(item).strip()]


def _optional_float(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(payload.get(key, default))
    except (TypeError, ValueError) as exc:
        raise CodeKnowledgeRecordError(f"{key} must be a number") from exc
    if value < 0.0 or value > 1.0:
        raise CodeKnowledgeRecordError(f"{key} must be between 0.0 and 1.0")
    return value


def _record_id(payload: dict[str, Any], schema_version: str) -> str:
    key = _RECORD_ID_KEYS[schema_version]
    value = str(payload.get(key) or "").strip()
    if not value:
        raise CodeKnowledgeRecordError(f"{key} is required")
    return value


def _normalize_source_refs(payload: dict[str, Any]) -> list[dict[str, str]]:
    raw = payload.get("source_refs") or []
    if not isinstance(raw, list):
        raise CodeKnowledgeRecordError("source_refs must be a list")
    normalized: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise CodeKnowledgeRecordError("source_refs entries must be objects")
        raw_url = str(item.get("url") or "").strip()
        normalized.append(
            {
                "title": str(item.get("title") or "").strip(),
                "url": canonicalize_nvidia_source_url(raw_url),
                "path": str(item.get("path") or "").strip(),
                "source_type": str(
                    item.get("source_type") or "vendor_documentation"
                ).strip(),
                "source_version": str(
                    item.get("source_version") or item.get("version") or ""
                ).strip(),
                "retrieved_or_verified_date": str(
                    item.get("retrieved_or_verified_date")
                    or item.get("retrieved_at")
                    or item.get("last_verified")
                    or ""
                ).strip(),
            }
        )
    return normalized


def is_recognized_nvidia_source_url(url: str) -> bool:
    """Return whether a source is first-party NVIDIA documentation/code."""

    try:
        parsed = urlparse(str(url).strip())
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https":
        return False
    if host in {
        "docs.nvidia.com",
        "developer.nvidia.com",
        "nvidia.github.io",
    } or host.endswith(".docs.nvidia.com"):
        return True
    return host == "github.com" and parsed.path.lower().startswith("/nvidia")


def canonicalize_nvidia_source_url(url: str) -> str:
    """Remove credentials/query tokens while retaining a useful doc anchor."""

    value = str(url or "").strip()
    if not is_recognized_nvidia_source_url(value):
        return value
    parsed = urlparse(value)
    fragment = parsed.fragment if re.fullmatch(r"[A-Za-z0-9._:-]{1,160}", parsed.fragment or "") else ""
    return urlunparse(("https", parsed.hostname or "", parsed.path or "/", "", "", fragment))


def _base_record(payload: dict[str, Any], schema_version: str) -> dict[str, Any]:
    record_id = _record_id(payload, schema_version)
    title = str(payload.get("title") or payload.get("api_symbol") or "").strip()
    if not title:
        raise CodeKnowledgeRecordError("title or api_symbol is required")
    text = str(
        payload.get("text")
        or payload.get("solution_summary")
        or payload.get("usage_summary")
        or payload.get("detail_text")
        or payload.get("summary_text")
        or ""
    ).strip()
    if not text:
        raise CodeKnowledgeRecordError("text, solution_summary, usage_summary, detail_text, or summary_text is required")
    source_refs = _normalize_source_refs(payload)
    first_source = source_refs[0] if source_refs else {}
    source_type = str(
        payload.get("source_type")
        or first_source.get("source_type")
        or "internal_note"
    ).strip()
    source_title = str(
        payload.get("source_title") or first_source.get("title") or ""
    ).strip()
    source_url = canonicalize_nvidia_source_url(str(
        payload.get("source_url") or first_source.get("url") or ""
    ).strip())
    source_version = str(
        payload.get("source_version")
        or first_source.get("source_version")
        or ""
    ).strip()
    applicability = payload.get("applicability") or {}
    if not isinstance(applicability, dict):
        raise CodeKnowledgeRecordError("applicability must be an object")
    unknown_applicability = sorted(
        set(applicability).difference(CUDA_DOC_APPLICABILITY_FIELDS)
    )
    if unknown_applicability:
        raise CodeKnowledgeRecordError(
            "applicability contains unsupported fields: "
            + ", ".join(unknown_applicability)
        )
    normalized_applicability = {
        key: str(applicability.get(key) or "").strip()
        for key in CUDA_DOC_APPLICABILITY_FIELDS
    }
    support_status = str(
        payload.get("support_status")
        or "unknown_pending_local_verification"
    ).strip()
    if support_status not in CUDA_DOC_SUPPORT_STATUSES:
        raise CodeKnowledgeRecordError(
            "support_status must be one of: "
            + ", ".join(sorted(CUDA_DOC_SUPPORT_STATUSES))
        )
    applicability_support = payload.get("applicability_support") or {}
    if not isinstance(applicability_support, dict):
        raise CodeKnowledgeRecordError("applicability_support must be an object")
    invalid_support = sorted(
        {
            str(value)
            for value in applicability_support.values()
            if str(value) not in CUDA_DOC_SUPPORT_STATUSES
        }
    )
    if invalid_support:
        raise CodeKnowledgeRecordError(
            "applicability_support contains unsupported statuses: "
            + ", ".join(invalid_support)
        )
    if source_type == "nvidia_cuda_docs":
        if not source_refs or not any(
            is_recognized_nvidia_source_url(item.get("url", ""))
            for item in source_refs
        ):
            raise CodeKnowledgeRecordError(
                "verified NVIDIA CUDA documentation requires a recognized NVIDIA source URL"
            )
        if not is_recognized_nvidia_source_url(source_url):
            raise CodeKnowledgeRecordError(
                "primary source_url must be a recognized NVIDIA source"
            )
        retrieved_date = str(
            payload.get("retrieved_or_verified_date")
            or first_source.get("retrieved_or_verified_date")
            or ""
        ).strip()
        if not retrieved_date:
            raise CodeKnowledgeRecordError(
                "verified NVIDIA CUDA documentation requires a retrieval/verification date"
            )
        required = {
            "gpu_architecture",
            "compute_capability",
            "driver_major_minor",
            "cuda_major_minor",
            "framework_major_minor",
            "backend_mode",
            "backend_config_hash",
            "runner_contract",
            "remote_tool_schema_hash",
        }
        missing = sorted(
            key
            for key in required
            if not normalized_applicability.get(key)
            or normalized_applicability.get(key, "").lower() == "unknown"
        )
        if missing:
            raise CodeKnowledgeRecordError(
                "NVIDIA CUDA documentation applicability is missing: "
                + ", ".join(missing)
            )
    framework = str(payload.get("framework") or "pytorch").strip().lower()
    return {
        "schema_version": schema_version,
        "record_type": _COLLECTION_BY_SCHEMA[schema_version],
        _RECORD_ID_KEYS[schema_version]: record_id,
        "record_id": record_id,
        "title": title,
        "text": text,
        "source_id": str(payload.get("source_id") or "").strip(),
        "source_type": source_type,
        "source_title": source_title,
        "source_url": source_url,
        "source_version": source_version,
        "source_refs": source_refs,
        "retrieved_or_verified_date": str(
            payload.get("retrieved_or_verified_date")
            or first_source.get("retrieved_or_verified_date")
            or ""
        ).strip(),
        "framework": framework,
        "frameworks": _as_string_list(payload, "frameworks") or [framework],
        "framework_version": str(payload.get("framework_version") or "").strip(),
        "framework_versions": _as_string_list(payload, "framework_versions"),
        "toolkits": _as_string_list(payload, "toolkits"),
        "toolkit_versions": _as_string_list(payload, "toolkit_versions"),
        "driver_versions": _as_string_list(payload, "driver_versions"),
        "compute_capabilities": _as_string_list(payload, "compute_capabilities"),
        "accelerator_names": _as_string_list(payload, "accelerator_names"),
        "gpu_architectures": _as_string_list(payload, "gpu_architectures"),
        "backend_keys": _as_string_list(payload, "backend_keys"),
        "technology_keys": _as_string_list(payload, "technology_keys"),
        "hardware_keys": _as_string_list(payload, "hardware_keys"),
        "hardware_feature_keys": _as_string_list(payload, "hardware_feature_keys"),
        "model_keys": _as_string_list(payload, "model_keys"),
        "model_families": _as_string_list(payload, "model_families"),
        "workload_types": _as_string_list(payload, "workload_types"),
        "optimization_targets": _as_string_list(payload, "optimization_targets"),
        "profile_symptoms": _as_string_list(payload, "profile_symptoms"),
        "api_symbols": _as_string_list(payload, "api_symbols"),
        "precision_modes": _as_string_list(payload, "precision_modes"),
        "risk_level": str(payload.get("risk_level") or "medium").strip().lower(),
        "confidence": _optional_float(payload, "confidence", 0.5),
        "deprecated": bool(payload.get("deprecated", False)),
        "backend_modes": _as_string_list(payload, "backend_modes")
        or [BACKEND_NEUTRAL],
        "runner_contracts": _as_string_list(payload, "runner_contracts")
        or [RUNNER_CONTRACT_SUBPROCESS_V1],
        "pipeline_stages": _as_string_list(payload, "pipeline_stages")
        or list(BACKEND_GUIDANCE_PIPELINE_STAGES),
        "rule_type": str(payload.get("rule_type") or "recommendation")
        .strip()
        .lower(),
        "owner": str(payload.get("owner") or "job_code").strip().lower(),
        "strength": str(payload.get("strength") or "informational")
        .strip()
        .lower(),
        "transferability": str(
            payload.get("transferability") or BACKEND_NEUTRAL
        )
        .strip()
        .lower(),
        "applicability": normalized_applicability,
        "support_status": support_status,
        "applicability_support": {
            str(key): str(value) for key, value in applicability_support.items()
        },
        "cuda_docs_cache_key": str(
            payload.get("cuda_docs_cache_key") or ""
        ).strip(),
        "query_template_version": str(
            payload.get("query_template_version") or ""
        ).strip(),
        "remote_tool_schema_hash": str(
            payload.get("remote_tool_schema_hash") or ""
        ).strip(),
        "backend_config_hash": str(
            payload.get("backend_config_hash") or ""
        ).strip(),
        "verified_source": bool(payload.get("verified_source", False)),
    }


def _validate_enum_list(
    payload: dict[str, Any], key: str, allowed: frozenset[str]
) -> list[str]:
    values = _as_string_list(payload, key)
    if not values:
        raise CodeKnowledgeRecordError(f"{key} must contain at least one value")
    invalid = sorted(set(values).difference(allowed))
    if invalid:
        raise CodeKnowledgeRecordError(
            f"{key} contains unsupported values: {', '.join(invalid)}"
        )
    return values


def _validate_backend_scope(normalized: dict[str, Any]) -> None:
    """Reject invalid runtime applicability on every knowledge record type."""

    backend_modes = list(normalized.get("backend_modes") or [])
    invalid_modes = sorted(set(backend_modes).difference(BACKEND_GUIDANCE_MODES))
    if invalid_modes:
        raise CodeKnowledgeRecordError(
            "backend_modes contains unsupported values: "
            + ", ".join(invalid_modes)
        )
    if BACKEND_NEUTRAL in backend_modes and len(backend_modes) != 1:
        raise CodeKnowledgeRecordError(
            "backend_neutral cannot be combined with an exact backend mode"
        )

    runner_contracts = list(normalized.get("runner_contracts") or [])
    if not runner_contracts or any(
        item != RUNNER_CONTRACT_SUBPROCESS_V1 for item in runner_contracts
    ):
        raise CodeKnowledgeRecordError(
            "runner_contracts must currently contain subprocess_job_v1"
        )

    pipeline_stages = list(normalized.get("pipeline_stages") or [])
    invalid_stages = sorted(
        set(pipeline_stages).difference(BACKEND_GUIDANCE_PIPELINE_STAGES)
    )
    if not pipeline_stages or invalid_stages:
        detail = ": " + ", ".join(invalid_stages) if invalid_stages else ""
        raise CodeKnowledgeRecordError(
            "pipeline_stages contains unsupported values" + detail
        )

    transferability = str(normalized.get("transferability") or "").strip()
    if transferability not in BACKEND_GUIDANCE_TRANSFERABILITY:
        raise CodeKnowledgeRecordError(
            "transferability must be one of: "
            + ", ".join(sorted(BACKEND_GUIDANCE_TRANSFERABILITY))
        )
    if transferability == BACKEND_NEUTRAL and backend_modes != [BACKEND_NEUTRAL]:
        raise CodeKnowledgeRecordError(
            "backend_neutral transferability requires backend_modes: [backend_neutral]"
        )
    if transferability == "exact_backend" and backend_modes == [BACKEND_NEUTRAL]:
        raise CodeKnowledgeRecordError(
            "exact_backend transferability requires an exact backend mode"
        )


def _validate_backend_guidance_record(
    payload: dict[str, Any], normalized: dict[str, Any]
) -> dict[str, Any]:
    backend_modes = _validate_enum_list(
        payload, "backend_modes", BACKEND_GUIDANCE_MODES
    )
    runner_contracts = _as_string_list(payload, "runner_contracts")
    if not runner_contracts or any(
        item != RUNNER_CONTRACT_SUBPROCESS_V1 for item in runner_contracts
    ):
        raise CodeKnowledgeRecordError(
            "runner_contracts must currently contain subprocess_job_v1"
        )
    pipeline_stages = _validate_enum_list(
        payload, "pipeline_stages", BACKEND_GUIDANCE_PIPELINE_STAGES
    )
    rule_type = str(payload.get("rule_type") or "").strip().lower()
    owner = str(payload.get("owner") or "").strip().lower()
    strength = str(payload.get("strength") or "").strip().lower()
    transferability = str(payload.get("transferability") or "").strip().lower()
    review_status = str(payload.get("review_status") or "reviewed").strip().lower()
    for key, value, allowed in (
        ("rule_type", rule_type, BACKEND_GUIDANCE_RULE_TYPES),
        ("owner", owner, BACKEND_GUIDANCE_OWNERS),
        ("strength", strength, BACKEND_GUIDANCE_STRENGTHS),
        (
            "transferability",
            transferability,
            BACKEND_GUIDANCE_TRANSFERABILITY,
        ),
        ("review_status", review_status, BACKEND_GUIDANCE_REVIEW_STATUSES),
    ):
        if value not in allowed:
            raise CodeKnowledgeRecordError(
                f"{key} must be one of: {', '.join(sorted(allowed))}"
            )
    if BACKEND_NEUTRAL in backend_modes and len(backend_modes) != 1:
        raise CodeKnowledgeRecordError(
            "backend_neutral cannot be combined with an exact backend mode"
        )
    if transferability == "backend_neutral" and backend_modes != [BACKEND_NEUTRAL]:
        raise CodeKnowledgeRecordError(
            "backend_neutral transferability requires backend_modes: [backend_neutral]"
        )
    if transferability == "exact_backend" and backend_modes == [BACKEND_NEUTRAL]:
        raise CodeKnowledgeRecordError(
            "exact_backend transferability requires an exact backend mode"
        )
    constraints = payload.get("hardware_constraints") or {}
    if not isinstance(constraints, dict):
        raise CodeKnowledgeRecordError("hardware_constraints must be an object")
    source_refs = payload.get("source_refs") or []
    if not isinstance(source_refs, list) or not source_refs:
        raise CodeKnowledgeRecordError(
            "source_refs must contain at least one provenance reference"
        )
    if any(
        not isinstance(item, dict)
        or not str(item.get("title") or "").strip()
        or not str(item.get("url") or item.get("path") or "").strip()
        for item in source_refs
    ):
        raise CodeKnowledgeRecordError(
            "each source_refs entry requires title and url or path"
        )
    normalized.update(
        {
            "rule_id": normalized["record_id"],
            "backend_modes": backend_modes,
            "runner_contracts": runner_contracts,
            "pipeline_stages": pipeline_stages,
            "rule_type": rule_type,
            "owner": owner,
            "strength": strength,
            "transferability": transferability,
            "frameworks": _as_string_list(payload, "frameworks")
            or [normalized["framework"]],
            "review_status": review_status,
            "hardware_constraints": dict(constraints),
            "recommended_patterns": _as_string_list(
                payload, "recommended_patterns"
            ),
            "avoid_patterns": _as_string_list(payload, "avoid_patterns"),
            "source_refs": [dict(item) for item in source_refs],
            "last_verified": str(payload.get("last_verified") or "").strip(),
            "active": bool(payload.get("active", True)),
        }
    )
    if not normalized["frameworks"]:
        raise CodeKnowledgeRecordError("frameworks must contain at least one value")
    if not normalized["last_verified"]:
        raise CodeKnowledgeRecordError("last_verified is required")
    if normalized["review_status"] == "retired" and normalized["active"]:
        raise CodeKnowledgeRecordError(
            "a retired backend-guidance rule cannot remain active"
        )
    return normalized


def validate_code_knowledge_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one code-knowledge vector record."""
    if not isinstance(payload, dict):
        raise CodeKnowledgeRecordError("code-knowledge record must be an object")
    schema_version = str(payload.get("schema_version") or "").strip()
    if schema_version == HARDWARE_FEATURE_SCHEMA_VERSION:
        converted = convert_hardware_feature_records([validate_feature_record(payload)])
        if not converted:
            raise CodeKnowledgeRecordError("hardware feature record could not be converted")
        return converted[0]
    if schema_version not in _RECORD_ID_KEYS:
        raise CodeKnowledgeRecordError(f"unsupported schema_version: {schema_version}")
    normalized = _base_record(payload, schema_version)
    _validate_backend_scope(normalized)
    if schema_version == BACKEND_GUIDANCE_SCHEMA_VERSION:
        normalized = _validate_backend_guidance_record(payload, normalized)
    elif schema_version == OPTIMIZATION_RECIPE_SCHEMA_VERSION:
        review_status = str(payload.get("review_status") or "reviewed").strip().lower()
        if review_status not in BACKEND_GUIDANCE_REVIEW_STATUSES:
            raise CodeKnowledgeRecordError(
                "review_status must be one of: "
                + ", ".join(sorted(BACKEND_GUIDANCE_REVIEW_STATUSES))
            )
        normalized.update(
            {
                "problem_statement": str(payload.get("problem_statement") or "").strip(),
                "solution_summary": str(payload.get("solution_summary") or normalized["text"]).strip(),
                "recommended_patterns": _as_string_list(payload, "recommended_patterns"),
                "avoid_patterns": _as_string_list(payload, "avoid_patterns"),
                "source_chunk_ids": _as_string_list(payload, "source_chunk_ids"),
                "source_job_ids": _as_string_list(payload, "source_job_ids"),
                "review_status": review_status,
            }
        )
        if not normalized["optimization_targets"]:
            raise CodeKnowledgeRecordError("optimization_recipe_chunks require optimization_targets")
    elif schema_version == API_SYMBOL_SCHEMA_VERSION:
        api_symbol = str(payload.get("api_symbol") or normalized["title"]).strip()
        normalized.update(
            {
                "api_symbol": api_symbol,
                "signature": str(payload.get("signature") or "").strip(),
                "usage_summary": str(payload.get("usage_summary") or normalized["text"]).strip(),
                "parameters_json": str(payload.get("parameters_json") or "").strip(),
                "example_code": str(payload.get("example_code") or "").strip(),
            }
        )
        if api_symbol not in normalized["api_symbols"]:
            normalized["api_symbols"].append(api_symbol)
    else:
        normalized.update(
            {
                "chunk_id": normalized["record_id"],
                "text_hash": str(payload.get("text_hash") or hashlib.sha256(normalized["text"].encode("utf-8")).hexdigest()),
            }
        )
    return normalized


def validate_backend_guidance_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the public deterministic backend-guidance contract."""

    if str(payload.get("schema_version") or "") != BACKEND_GUIDANCE_SCHEMA_VERSION:
        raise CodeKnowledgeRecordError(
            f"schema_version must be {BACKEND_GUIDANCE_SCHEMA_VERSION}"
        )
    return validate_code_knowledge_record(payload)


def convert_hardware_feature_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert curated hardware capability records into code-knowledge records."""
    converted: list[dict[str, Any]] = []
    for raw_record in records:
        record = validate_feature_record(raw_record)
        feature_keys = list(record.get("features") or [])
        source_refs = list(record.get("source_refs") or [])
        first_source = source_refs[0] if source_refs else {}
        base = {
            "source_id": record["record_id"],
            "source_type": "hardware_feature_record",
            "source_title": first_source.get("title", ""),
            "source_url": first_source.get("url", ""),
            "framework": (record.get("frameworks") or ["pytorch"])[0],
            "hardware_feature_keys": feature_keys,
            "technology_keys": feature_keys,
            "workload_types": list(record.get("workload_types") or []),
            "optimization_targets": ["improve_throughput", "reduce_vram"],
            "api_symbols": [],
            "confidence": record.get("confidence", 0.5),
        }
        converted.append(
            validate_code_knowledge_record(
                {
                    **base,
                    "schema_version": CODE_DOC_SCHEMA_VERSION,
                    "chunk_id": f"hardware_feature_doc:{record['record_id']}",
                    "title": record["title"],
                    "text": "\n".join([record["summary_text"], record["detail_text"]]).strip(),
                    "tags": list(record.get("tags") or []),
                }
            )
        )
        if record.get("recommended_patterns"):
            converted.append(
                validate_code_knowledge_record(
                    {
                        **base,
                        "schema_version": OPTIMIZATION_RECIPE_SCHEMA_VERSION,
                        "recipe_id": f"hardware_feature_recipe:{record['record_id']}",
                        "title": record["title"],
                        "problem_statement": record["summary_text"],
                        "solution_summary": " ".join(record.get("recommended_patterns") or []),
                        "text": "\n".join(record.get("recommended_patterns") or []),
                        "recommended_patterns": list(record.get("recommended_patterns") or []),
                        "avoid_patterns": list(record.get("avoid_patterns") or []),
                        "profile_symptoms": ["precision_not_optimized"],
                        "risk_level": "medium",
                    }
                )
            )
    return converted


def record_to_search_text(record: dict[str, Any]) -> str:
    """Build dense-vector text for one validated code-knowledge record."""
    parts = [
        record.get("title", ""),
        record.get("text", ""),
        record.get("problem_statement", ""),
        record.get("solution_summary", ""),
        record.get("usage_summary", ""),
        "Framework: " + str(record.get("framework") or ""),
        "Framework versions: " + ", ".join(record.get("framework_versions") or []),
        "Toolkits: " + ", ".join(record.get("toolkits") or []),
        "Toolkit versions: " + ", ".join(record.get("toolkit_versions") or []),
        "Compute capabilities: " + ", ".join(record.get("compute_capabilities") or []),
        "Accelerators: " + ", ".join(record.get("accelerator_names") or []),
        "GPU architectures: " + ", ".join(record.get("gpu_architectures") or []),
        "Technologies: " + ", ".join(record.get("technology_keys") or []),
        "Hardware features: " + ", ".join(record.get("hardware_feature_keys") or []),
        "Model families: " + ", ".join(record.get("model_families") or []),
        "Workloads: " + ", ".join(record.get("workload_types") or []),
        "Symptoms: " + ", ".join(record.get("profile_symptoms") or []),
        "Targets: " + ", ".join(record.get("optimization_targets") or []),
        "APIs: " + ", ".join(record.get("api_symbols") or []),
        "Recommended patterns: " + " ".join(record.get("recommended_patterns") or []),
        "Avoid patterns: " + " ".join(record.get("avoid_patterns") or []),
        "Backend modes: " + ", ".join(record.get("backend_modes") or []),
        "Runner contracts: " + ", ".join(record.get("runner_contracts") or []),
        "Pipeline stages: " + ", ".join(record.get("pipeline_stages") or []),
        "Capability support: " + str(record.get("support_status") or ""),
    ]
    return "\n".join(part for part in parts if str(part).strip())


def load_code_knowledge_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise CodeKnowledgeRecordError("code-knowledge source must be a list or contain records: []")
    normalized: list[dict[str, Any]] = []
    for record in records:
        if isinstance(record, dict) and record.get("schema_version") == HARDWARE_FEATURE_SCHEMA_VERSION:
            normalized.extend(convert_hardware_feature_records([record]))
        else:
            normalized.append(validate_code_knowledge_record(record))
    return normalized


def load_backend_guidance_seed_records() -> list[dict[str, Any]]:
    return load_code_knowledge_records(
        Path(__file__).with_name("backend_guidance_records.yaml")
    )


def validate_backend_guidance_corpus(
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized = records or load_backend_guidance_seed_records()
    active = [record for record in normalized if record.get("active", True)]
    retired_refs = [
        record["record_id"]
        for record in active
        if set(record.get("backend_modes") or []).intersection(RETIRED_BACKEND_MODES)
    ]
    reachable = {
        mode: any(
            mode in (record.get("backend_modes") or [])
            or BACKEND_NEUTRAL in (record.get("backend_modes") or [])
            for record in active
        )
        for mode in ("exclusive", *PACKED_BACKEND_MODES)
    }
    return {
        "ok": not retired_refs and all(reachable.values()),
        "active_rule_count": len(active),
        "retired_backend_rule_ids": retired_refs,
        "reachable_backends": reachable,
        "unreachable_backends": [
            mode for mode, is_reachable in reachable.items() if not is_reachable
        ],
    }
