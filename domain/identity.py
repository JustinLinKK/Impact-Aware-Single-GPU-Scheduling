"""Stable identity builders for scheduling and profiling records."""

from __future__ import annotations

from hashlib import sha1
from typing import Any
import json

from .jobs import normalize_batch_probe_search_mode, normalize_runtime_probe_strategy


def build_batch_probe_key(model_key: str, device_type: str, shape_signature: str, *, search_mode: str | None = None) -> str:
    payload = {
        "device_type": device_type,
        "model_key": model_key,
        "search_mode": normalize_batch_probe_search_mode(search_mode),
        "shape_signature": shape_signature,
    }
    return sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_batch_size_observation_key(
    model_key: str,
    shape_signature: str,
    hardware_key: str,
    backend_name: str,
    batch_size: int,
) -> str:
    payload = {
        "backend_name": backend_name,
        "batch_size": int(batch_size),
        "hardware_key": hardware_key,
        "model_key": model_key,
        "shape_signature": shape_signature,
    }
    return sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def canonical_pair_key(left_signature: str, right_signature: str) -> str:
    ordered = sorted((left_signature, right_signature))
    return f"{ordered[0]}::{ordered[1]}"


def normalize_group_signatures(signatures: list[str]) -> list[str]:
    return sorted(signature for signature in signatures if signature)


def build_backend_scoped_pair_key(left_signature: str, right_signature: str, *, backend_name: str) -> str:
    return f"{str(backend_name)}::{canonical_pair_key(left_signature, right_signature)}"


def build_group_signature(signatures: list[str]) -> str:
    ordered = normalize_group_signatures(signatures)
    return "::".join(ordered)


def encode_batch_vector(items: dict[str, int]) -> str:
    normalized = {str(key): int(value) for key, value in sorted(items.items())}
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def decode_batch_vector(value: str | dict[str, Any] | None) -> dict[str, int]:
    if value is None:
        return {}
    payload = json.loads(value) if isinstance(value, str) else dict(value)
    return {str(key): int(item) for key, item in payload.items()}


def build_combination_key(
    group_signature: str,
    hardware_key: str,
    backend_name: str,
    scheduler_mode: str,
    batch_vector: dict[str, int],
) -> str:
    payload = {
        "backend_name": backend_name,
        "batch_vector": encode_batch_vector(batch_vector),
        "group_signature": group_signature,
        "hardware_key": hardware_key,
        "scheduler_mode": scheduler_mode,
    }
    return sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_runtime_profile_key(
    signature: str,
    hardware_key: str,
    backend_name: str,
    resolved_batch_size: int,
    strategy: str,
) -> str:
    payload = {
        "signature": signature,
        "hardware_key": hardware_key,
        "backend_name": backend_name,
        "resolved_batch_size": int(resolved_batch_size),
        "strategy": normalize_runtime_probe_strategy(strategy),
    }
    return sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

