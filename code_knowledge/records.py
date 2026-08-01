"""Code-knowledge vector record loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib

import yaml

from localml_scheduler.hardware_features.records import HARDWARE_FEATURE_SCHEMA_VERSION, validate_feature_record


CODE_DOC_SCHEMA_VERSION = "code_doc_chunk_v1"
OPTIMIZATION_RECIPE_SCHEMA_VERSION = "optimization_recipe_chunk_v1"
API_SYMBOL_SCHEMA_VERSION = "api_symbol_chunk_v1"

_RECORD_ID_KEYS = {
    CODE_DOC_SCHEMA_VERSION: "chunk_id",
    OPTIMIZATION_RECIPE_SCHEMA_VERSION: "recipe_id",
    API_SYMBOL_SCHEMA_VERSION: "api_symbol_id",
}

_COLLECTION_BY_SCHEMA = {
    CODE_DOC_SCHEMA_VERSION: "code_doc_chunks",
    OPTIMIZATION_RECIPE_SCHEMA_VERSION: "optimization_recipe_chunks",
    API_SYMBOL_SCHEMA_VERSION: "api_symbol_chunks",
}


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
    return {
        "schema_version": schema_version,
        "record_type": _COLLECTION_BY_SCHEMA[schema_version],
        _RECORD_ID_KEYS[schema_version]: record_id,
        "record_id": record_id,
        "title": title,
        "text": text,
        "source_id": str(payload.get("source_id") or "").strip(),
        "source_type": str(payload.get("source_type") or "internal_note").strip(),
        "source_title": str(payload.get("source_title") or "").strip(),
        "source_url": str(payload.get("source_url") or "").strip(),
        "source_version": str(payload.get("source_version") or "").strip(),
        "framework": str(payload.get("framework") or "pytorch").strip().lower(),
        "framework_version": str(payload.get("framework_version") or "").strip(),
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
    }


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
    if schema_version == OPTIMIZATION_RECIPE_SCHEMA_VERSION:
        normalized.update(
            {
                "problem_statement": str(payload.get("problem_statement") or "").strip(),
                "solution_summary": str(payload.get("solution_summary") or normalized["text"]).strip(),
                "recommended_patterns": _as_string_list(payload, "recommended_patterns"),
                "avoid_patterns": _as_string_list(payload, "avoid_patterns"),
                "source_chunk_ids": _as_string_list(payload, "source_chunk_ids"),
                "source_job_ids": _as_string_list(payload, "source_job_ids"),
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
        "Technologies: " + ", ".join(record.get("technology_keys") or []),
        "Hardware features: " + ", ".join(record.get("hardware_feature_keys") or []),
        "Model families: " + ", ".join(record.get("model_families") or []),
        "Workloads: " + ", ".join(record.get("workload_types") or []),
        "Symptoms: " + ", ".join(record.get("profile_symptoms") or []),
        "Targets: " + ", ".join(record.get("optimization_targets") or []),
        "APIs: " + ", ".join(record.get("api_symbols") or []),
        "Recommended patterns: " + " ".join(record.get("recommended_patterns") or []),
        "Avoid patterns: " + " ".join(record.get("avoid_patterns") or []),
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
