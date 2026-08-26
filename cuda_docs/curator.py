"""Asynchronous, JSON-only recipe curation for verified CUDA doc chunks.

This module deliberately does not interpret Markdown bullets. A recipe is
published only when the hosted result already contains a structured object and
the resulting record passes provenance, applicability, ownership, and schema
validation.
"""

from __future__ import annotations

from typing import Any, Iterable
import hashlib
import json
import re

from ..code_knowledge.records import (
    BACKEND_GUIDANCE_REVIEW_STATUSES,
    OPTIMIZATION_RECIPE_SCHEMA_VERSION,
    is_recognized_nvidia_source_url,
    validate_code_knowledge_record,
)

STRUCTURED_RECIPE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "title",
        "problem_statement",
        "solution_summary",
        "optimization_targets",
        "recommended_patterns",
    ],
    "properties": {
        "title": {"type": "string", "maxLength": 240},
        "problem_statement": {"type": "string", "maxLength": 2000},
        "solution_summary": {"type": "string", "maxLength": 4000},
        "optimization_targets": {
            "type": "array",
            "maxItems": 8,
            "items": {"type": "string", "maxLength": 120},
        },
        "recommended_patterns": {
            "type": "array",
            "maxItems": 8,
            "items": {"type": "string", "maxLength": 500},
        },
        "avoid_patterns": {
            "type": "array",
            "maxItems": 8,
            "items": {"type": "string", "maxLength": 500},
        },
    },
}

_SCHEDULER_CONTROL = re.compile(
    r"(?:nvidia-cuda-mps-control|CUDA_MPS_[A-Z_]+|active[._ ]thread[._ ]percentage|"
    r"start\s+(?:the\s+)?MPS|stop\s+(?:the\s+)?MPS|scheduler\s+(?:ranking|admission)|"
    r"cross[- ]job\s+(?:stream|CUDA context)|shared CUDA context)",
    re.I,
)


def synthesize_structured_recipe_records(
    result: Any,
    source_records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build validated recipes from structured MCP fields, never prose shape."""

    sources = [dict(item) for item in source_records]
    if not sources:
        return []
    structured = _value(result, "structuredContent", "structured_content")
    if structured is None:
        return []
    candidates = _recipe_candidates(structured)
    recipes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            normalized = _normalize_candidate(candidate)
            source = _matching_source(normalized, sources)
            recipe = _build_record(normalized, source)
            validated = validate_code_knowledge_record(recipe)
        except (TypeError, ValueError):
            continue
        record_id = str(validated.get("record_id") or "")
        if record_id and record_id not in seen:
            recipes.append(validated)
            seen.add(record_id)
    return recipes


def _recipe_candidates(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if isinstance(value.get("recommended_patterns"), list):
            found.append(value)
        for child in value.values():
            if isinstance(child, (dict, list, tuple)):
                found.extend(_recipe_candidates(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.extend(_recipe_candidates(child))
    return found


def _normalize_candidate(value: dict[str, Any]) -> dict[str, Any]:
    title = _required_text(value, "title", 240)
    problem = _required_text(value, "problem_statement", 2000)
    summary = _required_text(value, "solution_summary", 4000)
    targets = _string_list(value, "optimization_targets", limit=8, max_chars=120)
    recommended = _string_list(value, "recommended_patterns", limit=8, max_chars=500)
    avoid = _string_list(value, "avoid_patterns", limit=8, max_chars=500)
    if not targets or not recommended:
        raise ValueError("structured recipe requires targets and recommendations")
    all_text = " ".join([title, problem, summary, *recommended, *avoid])
    if _SCHEDULER_CONTROL.search(all_text):
        raise ValueError("recipe attempts to control scheduler-owned behavior")
    return {
        "title": title,
        "problem_statement": problem,
        "solution_summary": summary,
        "optimization_targets": targets,
        "recommended_patterns": recommended,
        "avoid_patterns": avoid,
        "source_url": str(value.get("source_url") or value.get("url") or "").strip(),
    }


def _matching_source(
    candidate: dict[str, Any], source_records: list[dict[str, Any]]
) -> dict[str, Any]:
    requested_url = candidate.get("source_url")
    source = next(
        (
            item
            for item in source_records
            if requested_url and item.get("source_url") == requested_url
        ),
        source_records[0],
    )
    refs = list(source.get("source_refs") or [])
    if not refs or not all(
        isinstance(ref, dict)
        and is_recognized_nvidia_source_url(str(ref.get("url") or ""))
        for ref in refs
    ):
        raise ValueError("recipe provenance is missing or unverified")
    applicability = dict(source.get("applicability") or {})
    required = {
        "compute_capability",
        "cuda_major_minor",
        "framework_major_minor",
        "backend_mode",
        "backend_config_hash",
        "runner_contract",
        "remote_tool_schema_hash",
    }
    if any(not str(applicability.get(key) or "").strip() for key in required):
        raise ValueError("recipe applicability is incomplete")
    if source.get("support_status") == "unsupported":
        raise ValueError("unsupported evidence cannot publish a recipe")
    return source


def _build_record(candidate: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    identity = json.dumps(
        {
            "candidate": candidate,
            "source_ids": [source.get("record_id")],
            "applicability": source.get("applicability"),
            "source_version": source.get("source_version"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    recipe_id = (
        "nvidia.cuda_recipe." + hashlib.sha256(identity.encode()).hexdigest()[:24]
    )
    copied_list_fields = (
        "frameworks",
        "framework_versions",
        "toolkits",
        "toolkit_versions",
        "driver_versions",
        "compute_capabilities",
        "accelerator_names",
        "gpu_architectures",
        "backend_keys",
        "backend_modes",
        "runner_contracts",
        "pipeline_stages",
    )
    record: dict[str, Any] = {
        "schema_version": OPTIMIZATION_RECIPE_SCHEMA_VERSION,
        "recipe_id": recipe_id,
        "title": candidate["title"],
        "text": candidate["solution_summary"],
        "problem_statement": candidate["problem_statement"],
        "solution_summary": candidate["solution_summary"],
        "optimization_targets": candidate["optimization_targets"],
        "recommended_patterns": candidate["recommended_patterns"],
        "avoid_patterns": candidate["avoid_patterns"],
        "source_chunk_ids": [str(source.get("record_id") or "")],
        "source_id": str(source.get("record_id") or ""),
        "source_type": "nvidia_cuda_docs",
        "source_title": source.get("source_title"),
        "source_url": source.get("source_url"),
        "source_version": source.get("source_version"),
        "source_refs": list(source.get("source_refs") or []),
        "retrieved_or_verified_date": source.get("retrieved_or_verified_date"),
        "framework": source.get("framework") or "pytorch",
        "confidence": min(float(source.get("confidence") or 0.0), 0.8),
        "risk_level": "medium",
        "owner": "job_code",
        "rule_type": "recommendation",
        "strength": "informational",
        "transferability": source.get("transferability") or "exact_backend",
        "applicability": dict(source.get("applicability") or {}),
        "support_status": source.get("support_status"),
        "applicability_support": dict(source.get("applicability_support") or {}),
        "cuda_docs_cache_key": source.get("cuda_docs_cache_key"),
        "query_template_version": source.get("query_template_version"),
        "remote_tool_schema_hash": source.get("remote_tool_schema_hash"),
        "backend_config_hash": source.get("backend_config_hash"),
        "verified_source": True,
        # Validation below is the publication gate. Curated recipes cannot set
        # their own status from remote content.
        "review_status": "reviewed",
    }
    for key in copied_list_fields:
        record[key] = list(source.get(key) or [])
    return record


def _required_text(value: dict[str, Any], key: str, limit: int) -> str:
    text = str(value.get(key) or "").strip()
    if not text or len(text) > limit:
        raise ValueError(f"{key} is required and bounded")
    return text


def _string_list(
    value: dict[str, Any], key: str, *, limit: int, max_chars: int
) -> list[str]:
    raw = value.get(key) or []
    if not isinstance(raw, list) or len(raw) > limit:
        raise ValueError(f"{key} must be a bounded list")
    result = [str(item).strip() for item in raw if str(item).strip()]
    if any(len(item) > max_chars for item in result):
        raise ValueError(f"{key} contains an overlong item")
    return list(dict.fromkeys(result))


def _value(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None
