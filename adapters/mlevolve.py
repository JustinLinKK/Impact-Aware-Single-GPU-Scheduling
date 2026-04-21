"""Thin MLEvolve-facing adapter helpers."""

from __future__ import annotations

from typing import Any

from ..api import LocalMLSchedulerAPI
from ..schemas import CheckpointPolicy, ResourceRequirements, TrainingJob


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
    resource_requirements: ResourceRequirements | None = None,
    metadata: dict[str, Any] | None = None,
) -> TrainingJob:
    """Build a scheduler job from an MLEvolve candidate-training request."""
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
        metadata=metadata or {},
    )


def submit_mlevolve_job(api: LocalMLSchedulerAPI, **kwargs: Any) -> TrainingJob:
    """Convenience wrapper for creating and submitting a job."""
    job = build_mlevolve_job(**kwargs)
    return api.submit_job(job)
