"""Thin MLEvolve-facing adapter helpers."""

from __future__ import annotations

from hashlib import sha1
from typing import Any
import json
import re

from ..client import SchedulerClient
from ..domain import (
    BATCH_PROBE_SEARCH_MODE_POWER_OF_TWO,
    BatchProbeSpec,
    CheckpointPolicy,
    PackingSpec,
    PreloadSource,
    ResourceRequirements,
    RuntimeProbeSpec,
    TrainingJob,
)


def build_packing_signature(
    *,
    runner_target: str,
    baseline_model_id: str,
    task_type: str,
    runner_kwargs: dict[str, Any] | None = None,
    max_steps: int | None = None,
    max_epochs: int | None = None,
    family: str | None = None,
) -> str:
    """Build a stable signature for structured scheduler-managed workloads."""
    payload = {
        "baseline_model_id": baseline_model_id,
        "family": family,
        "max_epochs": max_epochs,
        "max_steps": max_steps,
        "runner_kwargs": runner_kwargs or {},
        "runner_target": runner_target,
        "task_type": task_type,
    }
    digest = sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    prefix = family or task_type or "job"
    return f"{prefix}:{digest[:16]}"


def build_mlevolve_job(
    *,
    workflow_id: str,
    baseline_model_id: str,
    baseline_model_path: str,
    runner_target: str,
    runner_kwargs: dict[str, Any] | None = None,
    priority: int = 0,
    task_type: str = "mlevolve_candidate",
    loader_target: str | None = None,
    checkpoint_policy: CheckpointPolicy | None = None,
    batch_probe: BatchProbeSpec | None = None,
    resource_requirements: ResourceRequirements | None = None,
    packing_family: str | None = None,
    packing_signature: str | None = None,
    packing_eligible: bool = False,
    packing_max_slowdown_ratio: float | None = None,
    packing_backend_allowlist: list[str] | None = None,
    runtime_probe: RuntimeProbeSpec | None = None,
    max_steps: int | None = None,
    max_epochs: int | None = None,
    preload_source: PreloadSource | None = None,
    metadata: dict[str, Any] | None = None,
) -> TrainingJob:
    """Build a scheduler job from an MLEvolve candidate-training request."""
    computed_signature = packing_signature
    if packing_eligible and computed_signature is None:
        computed_signature = build_packing_signature(
            runner_target=runner_target,
            baseline_model_id=baseline_model_id,
            task_type=task_type,
            runner_kwargs=runner_kwargs,
            max_steps=max_steps,
            max_epochs=max_epochs,
            family=packing_family,
        )
    default_runtime_probe = runtime_probe
    if default_runtime_probe is None:
        default_runtime_probe = RuntimeProbeSpec(
            enabled=(runner_target != "localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job"),
            strategy="epoch_1",
        )
    return TrainingJob.create(
        runner_target=runner_target,
        baseline_model_id=baseline_model_id,
        baseline_model_path=baseline_model_path,
        workflow_id=workflow_id,
        task_type=task_type,
        priority=priority,
        runner_kwargs=runner_kwargs or {},
        loader_target=loader_target,
        checkpoint_policy=checkpoint_policy,
        resource_requirements=resource_requirements,
        preload_source=preload_source,
        packing=PackingSpec(
            eligible=packing_eligible,
            signature=computed_signature,
            family=packing_family,
            max_slowdown_ratio=packing_max_slowdown_ratio,
            backend_allowlist=list(packing_backend_allowlist or []),
        ),
        batch_probe=batch_probe or BatchProbeSpec(),
        runtime_probe=default_runtime_probe,
        max_steps=max_steps,
        max_epochs=max_epochs,
        metadata=metadata or {},
    )


def build_startpoint_profile_key(*, task_id: str, model_key: str, modality: str, shape_hints: dict[str, Any] | None = None) -> str:
    payload = {
        "modality": modality,
        "model_key": model_key,
        "shape_hints": shape_hints or {},
        "task_id": task_id,
    }
    digest = sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"mlevolve-startpoint:{digest[:24]}"


def build_startpoint_shape_signature(*, task_id: str, model_key: str, modality: str, shape_hints: dict[str, Any] | None = None) -> str:
    payload = {
        "kind": "mlevolve_startpoint_shape",
        "modality": modality,
        "model_key": model_key,
        "shape_hints": shape_hints or {},
        "task_id": task_id,
    }
    digest = sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"mlevolve-startpoint-shape:{digest[:24]}"


def normalize_model_family(value: str | None) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.:/-]+", "-", str(value or "").strip().lower())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized or "unknown-family"


def build_model_family_profile_key(*, task_id: str, model_family: str) -> str:
    payload = {
        "kind": "mlevolve_model_family",
        "model_family": normalize_model_family(model_family),
        "task_id": str(task_id or "mlevolve"),
    }
    digest = sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"mlevolve-family:{digest[:24]}"


def build_model_family_shape_signature(
    *,
    task_id: str,
    model_family: str,
    shape_hints: dict[str, Any] | None = None,
) -> str:
    payload = {
        "kind": "mlevolve_model_family_shape",
        "model_family": normalize_model_family(model_family),
        "shape_hints": shape_hints or {},
        "task_id": str(task_id or "mlevolve"),
    }
    digest = sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"mlevolve-family-shape:{digest[:24]}"


def build_model_family_probe_job(
    *,
    workflow_id: str,
    task_id: str,
    model_family: str,
    script_path: str,
    working_dir: str,
    shape_hints: dict[str, Any] | None = None,
    priority: int = 100,
    probe_timeout_seconds: int = 45,
    probe_poll_interval_seconds: float = 0.5,
    start_batch_size: int | None = None,
    probe_max_batch_size: int | None = None,
) -> TrainingJob:
    """Build an exclusive probe job for a generated MLEvolve model family."""
    normalized_family = normalize_model_family(model_family)
    hints = dict(shape_hints or {})
    hints.setdefault("model_family", normalized_family)
    profile_key = build_model_family_profile_key(task_id=task_id, model_family=normalized_family)
    shape_signature = build_model_family_shape_signature(
        task_id=task_id,
        model_family=normalized_family,
        shape_hints=hints,
    )
    runner_kwargs = {
        "script_path": script_path,
        "working_dir": working_dir,
        "probe_timeout_seconds": int(probe_timeout_seconds),
        "probe_poll_interval_seconds": float(probe_poll_interval_seconds),
        "probe_max_epochs": 1,
        "probe_max_train_batches": 3,
        "batch_size": int(start_batch_size or hints.get("start_batch_size") or 1),
        "probe_max_batch_size": int(probe_max_batch_size or hints.get("probe_max_batch_size") or 256),
    }
    return build_mlevolve_job(
        workflow_id=workflow_id,
        baseline_model_id=f"mlevolve-model-family:{normalized_family}",
        baseline_model_path=str(script_path),
        runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_model_family_probe_job",
        runner_kwargs=runner_kwargs,
        priority=priority,
        task_type="mlevolve_model_family_probe",
        loader_target="localml_scheduler.adapters.mlevolve_runner:load_raw_file",
        batch_probe=BatchProbeSpec(
            enabled=True,
            probe_target="localml_scheduler.adapters.mlevolve_runner:probe_mlevolve_script_job",
            batch_param_name="batch_size",
            model_key=normalized_family,
            search_mode=BATCH_PROBE_SEARCH_MODE_POWER_OF_TWO,
            shape_hints=hints,
            profile_key=profile_key,
            shape_signature_override=shape_signature,
            reuse_only=False,
        ),
        resource_requirements=ResourceRequirements(requires_gpu=True),
        packing_family="mlevolve_model_family_probe",
        packing_signature=profile_key,
        packing_eligible=False,
        packing_backend_allowlist=["exclusive"],
        runtime_probe=RuntimeProbeSpec(enabled=False),
        metadata={
            "kind": "mlevolve_model_family_probe",
            "task_id": str(task_id or "mlevolve"),
            "model_family": normalized_family,
            "profile_key": profile_key,
            "shape_signature": shape_signature,
            "exclusive_probe": True,
        },
    )


def build_startpoint_probe_job(
    *,
    workflow_id: str,
    startpoint: dict[str, Any],
    priority: int = 100,
) -> TrainingJob:
    """Build a scheduler job that probes one cold-start startpoint model."""
    task_id = str(startpoint.get("task_id") or workflow_id or "mlevolve")
    model_key = str(startpoint.get("model_key") or startpoint.get("display_name") or "startpoint")
    modality = str(startpoint.get("modality") or "generic")
    shape_hints = dict(startpoint.get("shape_hints") or {})
    profile_key = str(
        startpoint.get("profile_key")
        or build_startpoint_profile_key(
            task_id=task_id,
            model_key=model_key,
            modality=modality,
            shape_hints=shape_hints,
        )
    )
    shape_signature = str(
        startpoint.get("shape_signature")
        or build_startpoint_shape_signature(
            task_id=task_id,
            model_key=model_key,
            modality=modality,
            shape_hints=shape_hints,
        )
    )
    runner_kwargs = {
        "batch_size": int(shape_hints.get("start_batch_size") or 1),
        "probe_max_batch_size": int(shape_hints.get("probe_max_batch_size") or 256),
    }
    return build_mlevolve_job(
        workflow_id=workflow_id,
        baseline_model_id=f"mlevolve-startpoint:{model_key}",
        baseline_model_path=f"synthetic://{model_key}",
        runner_target="localml_scheduler.adapters.startpoint_probe:run_startpoint_probe_job",
        runner_kwargs=runner_kwargs,
        priority=priority,
        task_type="mlevolve_startpoint_probe",
        batch_probe=BatchProbeSpec(
            enabled=True,
            probe_target="localml_scheduler.adapters.startpoint_probe:probe_startpoint_batch_size",
            batch_param_name="batch_size",
            model_key=model_key,
            search_mode=BATCH_PROBE_SEARCH_MODE_POWER_OF_TWO,
            shape_hints=shape_hints,
            profile_key=profile_key,
            shape_signature_override=shape_signature,
            reuse_only=False,
        ),
        resource_requirements=ResourceRequirements(requires_gpu=True),
        packing_family="mlevolve_startpoint_probe",
        packing_signature=profile_key,
        packing_eligible=False,
        packing_backend_allowlist=["exclusive"],
        runtime_probe=RuntimeProbeSpec(enabled=False),
        metadata={
            "kind": "mlevolve_startpoint_probe",
            "task_id": task_id,
            "startpoint_model_key": model_key,
            "startpoint_display_name": startpoint.get("display_name"),
            "startpoint_rank": startpoint.get("rank"),
            "startpoint_modality": modality,
            "startpoint_profile_key": profile_key,
            "startpoint_shape_signature": shape_signature,
        },
    )


def submit_mlevolve_job(api: SchedulerClient, **kwargs: Any) -> TrainingJob:
    """Convenience wrapper for creating and submitting a job."""
    job = build_mlevolve_job(**kwargs)
    return api.submit(job)
