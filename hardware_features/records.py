"""Curated hardware feature record loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


HARDWARE_FEATURE_SCHEMA_VERSION = "hardware_feature_record_v1"

_REQUIRED_STRING_FIELDS = {
    "schema_version",
    "record_id",
    "title",
    "summary_text",
    "detail_text",
    "vendor",
}

_REQUIRED_LIST_FIELDS = {
    "architectures",
    "accelerator_names",
    "compute_capabilities",
    "toolkits",
    "frameworks",
    "workload_types",
    "features",
    "recommended_patterns",
    "avoid_patterns",
    "source_refs",
    "tags",
}


class HardwareFeatureRecordError(ValueError):
    """Raised when a curated hardware feature record is invalid."""


def _as_string_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise HardwareFeatureRecordError(f"{key} must be a list")
    result = [str(item).strip() for item in value if str(item).strip()]
    if not result and key not in {"accelerator_names", "compute_capabilities", "avoid_patterns"}:
        raise HardwareFeatureRecordError(f"{key} must contain at least one value")
    return result


def _validate_source_refs(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise HardwareFeatureRecordError("source_refs must contain at least one source")
    refs: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise HardwareFeatureRecordError(f"source_refs[{index}] must be an object")
        normalized = {
            "title": str(item.get("title") or "").strip(),
            "url": str(item.get("url") or "").strip(),
            "source_type": str(item.get("source_type") or "").strip(),
            "retrieved_or_verified_date": str(item.get("retrieved_or_verified_date") or "").strip(),
        }
        missing = [key for key, field_value in normalized.items() if not field_value]
        if missing:
            raise HardwareFeatureRecordError(f"source_refs[{index}] missing required keys: {', '.join(missing)}")
        refs.append(normalized)
    return refs


def validate_feature_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one hardware feature record."""
    if not isinstance(payload, dict):
        raise HardwareFeatureRecordError("hardware feature record must be an object")
    for field in _REQUIRED_STRING_FIELDS:
        if not str(payload.get(field) or "").strip():
            raise HardwareFeatureRecordError(f"{field} is required")
    for field in _REQUIRED_LIST_FIELDS:
        if field not in payload:
            raise HardwareFeatureRecordError(f"{field} is required")
    if payload["schema_version"] != HARDWARE_FEATURE_SCHEMA_VERSION:
        raise HardwareFeatureRecordError(f"unsupported schema_version: {payload['schema_version']}")
    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise HardwareFeatureRecordError("confidence must be a number") from exc
    if confidence < 0.0 or confidence > 1.0:
        raise HardwareFeatureRecordError("confidence must be between 0.0 and 1.0")

    normalized = {
        "schema_version": HARDWARE_FEATURE_SCHEMA_VERSION,
        "record_id": str(payload["record_id"]).strip(),
        "title": str(payload["title"]).strip(),
        "summary_text": str(payload["summary_text"]).strip(),
        "detail_text": str(payload["detail_text"]).strip(),
        "vendor": str(payload["vendor"]).strip().lower(),
        "confidence": confidence,
        "last_verified": str(payload.get("last_verified") or "").strip()
        or max((str(ref.get("retrieved_or_verified_date") or "") for ref in payload["source_refs"]), default=""),
    }
    for field in _REQUIRED_LIST_FIELDS - {"source_refs"}:
        normalized[field] = _as_string_list(payload, field)
    normalized["source_refs"] = _validate_source_refs(payload["source_refs"])
    return normalized


def record_to_search_text(record: dict[str, Any]) -> str:
    """Build the dense-vector text body for one validated feature record."""
    text_parts = [
        record["title"],
        record["summary_text"],
        record["detail_text"],
        "Vendor: " + record["vendor"],
        "Architectures: " + ", ".join(record["architectures"]),
        "Accelerators: " + ", ".join(record["accelerator_names"]),
        "Compute capabilities: " + ", ".join(record["compute_capabilities"]),
        "Toolkits: " + ", ".join(record["toolkits"]),
        "Frameworks: " + ", ".join(record["frameworks"]),
        "Workloads: " + ", ".join(record["workload_types"]),
        "Features: " + ", ".join(record["features"]),
        "Recommended patterns: " + " ".join(record["recommended_patterns"]),
        "Avoid patterns: " + " ".join(record["avoid_patterns"]),
        "Tags: " + ", ".join(record["tags"]),
    ]
    return "\n".join(part for part in text_parts if part.strip())


def load_feature_records(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate records from a YAML file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise HardwareFeatureRecordError("hardware feature source must be a list or contain records: []")
    return [validate_feature_record(record) for record in records]


def load_seed_records() -> list[dict[str, Any]]:
    """Load the repo-curated seed corpus."""
    return load_feature_records(Path(__file__).with_name("seed_records.yaml"))
