"""CPU-only specification planning for explicit SYNTHETIC resource controls.

This module seals caller-selected budgets. It neither observes evaluation
outcomes nor selects an experimental duration from policy results.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from . import workloads

RESOURCE_CLASSES = {
    "tiny": {"width": 128, "repetitions": 2, "batch": 32, "ballast_mib": 0},
    "resident": {"width": 128, "repetitions": 2, "batch": 32, "ballast_mib": 8192},
    "compute": {"width": 2048, "repetitions": 8, "batch": 256, "ballast_mib": 0},
    "both": {"width": 2048, "repetitions": 8, "batch": 256, "ballast_mib": 8192},
}


def resource_config(resource_class, **overrides):
    """Return an independent configuration; caller owns experiment assignment."""
    if resource_class not in RESOURCE_CLASSES:
        raise ValueError("unknown resource class: " + str(resource_class))
    return {"model_family": workloads.MODEL_FAMILY, "input_features": 128,
            "precision": "fp32_ieee", "pool_size": 4096, "validation_size": 256,
            **deepcopy(RESOURCE_CLASSES[resource_class]), **deepcopy(overrides)}


def resource_plan(resource_classes, **overrides):
    """Expand an explicitly ordered class plan without choosing its outcomes."""
    return [resource_config(name, **overrides) for name in resource_classes]


def make_spec(config, *, name, role, chain, epochs, steps_per_epoch, seed,
              train_pool_seed, validation_pool_seed, source_path):
    """Seal one immutable replica after its configuration and budget are chosen.

    ``source_path`` is the actual file to hash and is retained as the source
    reference. An absolute path is supported; a relative reference resolves from
    the process's working directory while sealing and from the manifest's parent
    at launch, so those locations must agree for a relative reference.
    """
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("name must be one owned directory component")
    if role not in {"calibration", "parent", "child"}:
        raise ValueError("role must be calibration, parent or child")
    if role in {"parent", "child"} and (type(chain) is not int or chain not in (0, 1, 2)):
        raise ValueError("evaluation chain must be 0, 1 or 2")
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    forbidden = {"target_training_seconds", "predicted_training_seconds", "clipped",
                 "selection_source", "budget_decision", "outcome", "policy_result"}
    if forbidden.intersection(config):
        raise ValueError("budget/outcome annotations belong outside workload specifications")
    spec = {**deepcopy(config), "name": name, "role": role, "chain": chain,
            "epochs": epochs, "steps_per_epoch": steps_per_epoch, "seed": seed,
            "train_pool_seed": train_pool_seed, "validation_pool_seed": validation_pool_seed,
            "source": str(source), "source_sha256": workloads.file_hash(source),
            "model_family": config.get("model_family", workloads.MODEL_FAMILY),
            "workload_kind": "synthetic_resource_control", **workloads.source_metadata()}
    # Avoid a caller's stale derived identity participating in its own hash.
    for key in ("template_key", "spec_sha256", "expected_runtime_signature"):
        spec.pop(key, None)
    count = workloads.parameter_count(spec.get("width", 128), spec.get("input_features", 128))
    spec["parameter_count"] = spec["trainable_parameter_count"] = count
    workloads.recipe(spec)
    workloads.adapter_context(spec)
    spec["template_key"] = workloads.template_key(spec)
    spec["spec_sha256"] = workloads.spec_digest(spec)
    spec["expected_runtime_signature"] = workloads.expected_signature(spec)
    return spec
