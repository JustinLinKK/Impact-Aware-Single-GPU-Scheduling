"""Common domain helpers."""

from __future__ import annotations

from dataclasses import is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
import importlib
import uuid


def utc_now() -> str:
    """Return an ISO8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def parse_timestamp(value: str | None) -> datetime | None:
    """Parse an ISO8601 timestamp if present."""
    if not value:
        return None
    return datetime.fromisoformat(value)


def stable_job_id(explicit_job_id: str | None = None) -> str:
    """Create a stable job id at submit time."""
    return explicit_job_id or str(uuid.uuid4())


def import_string(target: str) -> Any:
    """Import a callable or object from ``module:attr`` syntax."""
    if ":" not in target:
        raise ValueError(f"Import target must be in module:attr form, got: {target}")
    module_name, attr_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    attr = module
    for part in attr_name.split("."):
        attr = getattr(attr, part)
    return attr


def to_primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {name: to_primitive(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_primitive(item) for item in value]
    return value


_to_primitive = to_primitive

