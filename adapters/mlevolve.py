"""Thin MLEvolve-facing adapter helpers."""

from __future__ import annotations

from hashlib import sha1
from typing import Any
import json

from ..client import SchedulerClient
from ..domain import (
    BatchProbeSpec,
    CheckpointPolicy,
    PackingSpec,
    PreloadSource,
    ResourceRequirements,
    RuntimeProbeSpec,
    TrainingJob,
    WorkloadIdentity,
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
    workload_identity: WorkloadIdentity | dict[str, Any] | None = None,
    task_key: str | None = None,
    dataset_key: str | None = None,
    architecture_key: str | None = None,
    architecture_family: str | None = None,
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
    job_metadata = dict(metadata or {})
    supplied_identity = (
        WorkloadIdentity.from_dict(workload_identity)
        if isinstance(workload_identity, dict)
        else workload_identity or WorkloadIdentity()
    )
    identity = WorkloadIdentity(
        task_key=supplied_identity.task_key or task_key or job_metadata.get("task_key") or workflow_id,
        dataset_key=supplied_identity.dataset_key or dataset_key or job_metadata.get("dataset_key"),
        architecture_key=(
            supplied_identity.architecture_key
            or architecture_key
            or job_metadata.get("architecture_key")
            or job_metadata.get("branch_name")
            or job_metadata.get("model_name")
            or packing_family
        ),
        architecture_family=(
            supplied_identity.architecture_family
            or architecture_family
            or job_metadata.get("architecture_family")
            or job_metadata.get("architecture_type")
            or job_metadata.get("model_family")
            or packing_family
        ),
    )
    return TrainingJob.create(
        runner_target=runner_target,
        baseline_model_id=baseline_model_id,
        baseline_model_path=baseline_model_path,
        workflow_id=workflow_id,
        task_type=task_type,
        workload_identity=identity,
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
        metadata=job_metadata,
    )


def submit_mlevolve_job(api: SchedulerClient, **kwargs: Any) -> TrainingJob:
    """Convenience wrapper for creating and submitting a job."""
    job = build_mlevolve_job(**kwargs)
    return api.submit(job)
