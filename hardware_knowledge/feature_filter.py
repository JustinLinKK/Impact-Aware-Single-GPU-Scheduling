"""Extract hardware signals relevant for runtime prediction from a Hardware node."""

from __future__ import annotations

from typing import Any


def extract_predictor_features(node: dict[str, Any]) -> dict[str, Any]:
    """Return numeric hardware signals for runtime/memory prediction. Missing fields are None."""
    props = node.get("properties", node)

    raw_cap = _first_capability(props.get("compute_capabilities"))

    return {
        "vram_MB": _as_int(props.get("vram_MB")),
        "memory_bandwidth_GBps": _as_float(props.get("memory_bandwidth_GBps")),
        "sm_count": _as_int(props.get("sm_count")),
        "l2_cache_MB": _as_float(props.get("l2_cache_MB")),
        "compute_capability": _as_float(raw_cap),
        "compute_target": raw_cap,
    }


def _first_capability(value: Any) -> str | None:
    """Return the first entry of compute_capabilities as a string, e.g. ['8.6'] -> '8.6', ['gfx1100'] -> 'gfx1100'."""
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0] if value else None
    return str(value).strip() if value is not None else None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
