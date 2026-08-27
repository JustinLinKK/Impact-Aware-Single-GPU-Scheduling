"""Stable identity builders for scheduling and profiling records."""

from __future__ import annotations

from hashlib import sha1
from typing import Any
import json

from ..backend_mode import RUNNER_CONTRACT_SUBPROCESS_V1, normalize_runtime_backend
from .jobs import normalize_runtime_probe_strategy


def build_batch_probe_key(model_key: str, device_type: str, shape_signature: str) -> str:
    payload = {
        "device_type": device_type,
        "model_key": model_key,
        "probe_policy": "time_aware_five_options",
        "shape_signature": shape_signature,
    }
    return sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_batch_size_observation_key(
    model_key: str,
    shape_signature: str,
    hardware_key: str,
    backend_name: str,
    batch_size: int,
    runner_contract: str = RUNNER_CONTRACT_SUBPROCESS_V1,
) -> str:
    payload = {
        "backend_name": normalize_runtime_backend(backend_name),
        "batch_size": int(batch_size),
        "hardware_key": hardware_key,
        "model_key": model_key,
        "runner_contract": runner_contract,
        "schema_version": 2,
        "shape_signature": shape_signature,
    }
    return sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def canonical_pair_key(left_signature: str, right_signature: str) -> str:
    ordered = sorted((left_signature, right_signature))
    return f"{ordered[0]}::{ordered[1]}"


def normalize_group_signatures(signatures: list[str]) -> list[str]:
    return sorted(signature for signature in signatures if signature)


def build_backend_scoped_pair_key(
    left_signature: str,
    right_signature: str,
    *,
    backend_name: str,
    runner_contract: str = RUNNER_CONTRACT_SUBPROCESS_V1,
) -> str:
    backend = normalize_runtime_backend(backend_name)
    return (
        f"v2::{runner_contract}::{backend}::"
        f"{canonical_pair_key(left_signature, right_signature)}"
    )


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
    runner_contract: str = RUNNER_CONTRACT_SUBPROCESS_V1,
) -> str:
    payload = {
        "backend_name": normalize_runtime_backend(backend_name),
        "batch_vector": encode_batch_vector(batch_vector),
        "group_signature": group_signature,
        "hardware_key": hardware_key,
        "scheduler_mode": scheduler_mode,
        "runner_contract": runner_contract,
        "schema_version": 2,
    }
    return sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_runtime_profile_key(
    signature: str,
    hardware_key: str,
    backend_name: str,
    resolved_batch_size: int,
    strategy: str,
    runner_contract: str = RUNNER_CONTRACT_SUBPROCESS_V1,
) -> str:
    payload = {
        "signature": signature,
        "hardware_key": hardware_key,
        "backend_name": normalize_runtime_backend(backend_name),
        "resolved_batch_size": int(resolved_batch_size),
        "strategy": normalize_runtime_probe_strategy(strategy),
        "runner_contract": runner_contract,
        "schema_version": 2,
    }
    return sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def normalize_colocation_members(members: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for member in members:
        item: dict[str, Any] = {
            "signature": str(member["signature"]),
            "batch_size": int(member["batch_size"]),
            "backend_name": normalize_runtime_backend(member["backend_name"]),
        }
        backend_config = member.get("backend_config")
        if isinstance(backend_config, dict) and backend_config:
            # A JSON round trip rejects unserializable runtime objects and
            # produces a stable topology/configuration identity.
            item["backend_config"] = json.loads(
                json.dumps(
                    backend_config,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        normalized.append(item)
    return sorted(
        normalized,
        key=lambda member: (
            member["signature"],
            member["batch_size"],
            member["backend_name"],
            json.dumps(
                member.get("backend_config", {}),
                sort_keys=True,
                separators=(",", ":"),
            ),
        ),
    )


def build_colocation_profile_key(hardware_key: str, members: list[dict[str, Any]]) -> str:
    payload = {
        "hardware_key": str(hardware_key),
        "members": normalize_colocation_members(members),
        "runner_contract": RUNNER_CONTRACT_SUBPROCESS_V1,
        "schema_version": 2,
    }
    return sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
