"""Thin MLEvolve-facing adapter helpers."""

from __future__ import annotations

from hashlib import sha1
from typing import Any
import json

from ..api import LocalMLSchedulerAPI
from ..schemas import BatchProbeSpec, CheckpointPolicy, PackingSpec, ResourceRequirements, TrainingJob


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
    max_steps: int | None = None,
    max_epochs: int | None = None,
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
        packing=PackingSpec(
            eligible=packing_eligible,
            signature=computed_signature,
            family=packing_family,
            max_slowdown_ratio=packing_max_slowdown_ratio,
        ),
        batch_probe=batch_probe or BatchProbeSpec(),
        max_steps=max_steps,
        max_epochs=max_epochs,
        metadata=metadata or {},
    )


def submit_mlevolve_job(api: LocalMLSchedulerAPI, **kwargs: Any) -> TrainingJob:
    """Convenience wrapper for creating and submitting a job."""
    job = build_mlevolve_job(**kwargs)
    return api.submit_job(job)
