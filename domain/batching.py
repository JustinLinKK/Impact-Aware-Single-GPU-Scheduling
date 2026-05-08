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
    def resolved_batch_size(job: TrainingJob) -> int:
        batch_param_name = BatchResolution.param_name(job)
        if job.metadata.get("resolved_batch_size") is not None:
            try:
                return max(1, int(job.metadata["resolved_batch_size"]))
            except (TypeError, ValueError):
                pass
        raw_value = job.config.runner_kwargs.get(batch_param_name)
        try:
            return max(1, int(raw_value))
        except (TypeError, ValueError):
            return 1

    @staticmethod
    def apply(job: TrainingJob, batch_size: int) -> TrainingJob:
        batch_param_name = BatchResolution.param_name(job)
        updated_job = job.copy()
        updated_job.config.runner_kwargs[batch_param_name] = int(batch_size)
        updated_job.metadata.update(
            {
                "resolved_batch_size": int(batch_size),
                "placement_batch_param_name": batch_param_name,
            }
        )
        return updated_job


def build_batch_probe_shape_signature(job: TrainingJob) -> str:
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

