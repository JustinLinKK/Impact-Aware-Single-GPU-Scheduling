"""Batch-related helpers shared across planner, runtime, and profiling."""

from __future__ import annotations

from hashlib import sha1
import json

from .jobs import TrainingJob


class BatchResolution:
    """Resolve and persist per-job batch-size choices consistently."""

    @staticmethod
    def param_name(job: TrainingJob) -> str:
        return job.batch_probe.batch_param_name or "batch_size"

    @staticmethod
    def authored_batch_size(job: TrainingJob) -> int:
        try:
            return max(1, int(job.authored_batch_size))
        except (TypeError, ValueError):
            batch_param_name = BatchResolution.param_name(job)
            return max(1, int(job.config.runner_kwargs.get(batch_param_name, 1)))

    @staticmethod
    def resolved_batch_size(job: TrainingJob) -> int:
        try:
            return max(1, int(job.current_batch_size))
        except (TypeError, ValueError):
            return BatchResolution.authored_batch_size(job)

    @staticmethod
    def apply(job: TrainingJob, batch_size: int) -> TrainingJob:
        batch_param_name = BatchResolution.param_name(job)
        updated_job = job.copy()
        updated_job.current_batch_size = int(batch_size)
        updated_job.placement_generation += 1
        updated_job.metadata["placement_batch_param_name"] = batch_param_name
        return updated_job

    @staticmethod
    def validate_authored_batch_size(job: TrainingJob) -> None:
        batch_size = BatchResolution.authored_batch_size(job)
        if batch_size < 1 or batch_size & (batch_size - 1):
            raise ValueError(
                f"job {job.job_id} authored batch size must be a positive power of two; got {batch_size}"
            )


def build_batch_probe_shape_signature(job: TrainingJob) -> str:
    if job.batch_probe.shape_signature_override:
        return str(job.batch_probe.shape_signature_override)
    batch_param_name = BatchResolution.param_name(job)
    ignored_runner_kwargs = {
        "script_path",
        "result_path",
        "working_dir",
        "timeout",
        "probe_timeout_seconds",
        "probe_poll_interval_seconds",
    }
    runner_kwargs = {
        key: value
        for key, value in dict(job.config.runner_kwargs).items()
        if key not in ignored_runner_kwargs
    }
    runner_kwargs.pop(batch_param_name, None)
    payload = {
        "runner_target": job.config.runner_target,
        "task_type": job.task_type,
        "loader_target": job.config.loader_target,
        "runner_kwargs": runner_kwargs,
        "shape_hints": job.batch_probe.shape_hints,
    }
    return sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
