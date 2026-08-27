"""Batch-related helpers shared across planner, runtime, and profiling."""

from __future__ import annotations

from hashlib import sha1
import json
import math

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
            return 1 if raw_value is None else max(1, int(raw_value))
        except (TypeError, ValueError):
            return 1

    @staticmethod
    def requested_batch_size(job: TrainingJob) -> int:
        """Return the immutable batch requested when the job was submitted."""
        if job.requested_batch_size is not None:
            return max(1, int(job.requested_batch_size))
        batch_param_name = BatchResolution.param_name(job)
        raw_value = job.config.runner_kwargs.get(batch_param_name)
        try:
            return 1 if raw_value is None else max(1, int(raw_value))
        except (TypeError, ValueError):
            return 1

    @staticmethod
    def apply(job: TrainingJob, batch_size: int) -> TrainingJob:
        batch_param_name = BatchResolution.param_name(job)
        batch_size = max(1, int(batch_size))
        contract = dict(job.metadata.get("training_quality_contract") or {})
        allowed = {
            max(1, int(value))
            for value in contract.get("allowed_physical_batch_sizes") or []
        }
        if allowed and batch_size not in allowed:
            raise ValueError(
                f"batch size {batch_size} is outside the agent-approved quality-safe envelope {sorted(allowed)}"
            )
        updated_job = job.copy()
        updated_job.config.runner_kwargs[batch_param_name] = batch_size
        updated_job.metadata.update(
            {
                "resolved_batch_size": batch_size,
                "placement_batch_param_name": batch_param_name,
            }
        )
        if updated_job.requested_batch_size is None:
            updated_job.requested_batch_size = BatchResolution.requested_batch_size(job)
        return TrainingParameterResolution.apply(updated_job, batch_size)


class TrainingParameterResolution:
    """Resolve every optimizer parameter coupled to a physical-batch override."""

    @staticmethod
    def _positive_int(value: object, default: int | None = None) -> int | None:
        try:
            parsed = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default
        return max(1, parsed)

    @staticmethod
    def _positive_float(value: object) -> float | None:
        try:
            parsed = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _nonnegative_int(value: object) -> int | None:
        try:
            parsed = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return max(0, parsed)

    @staticmethod
    def apply(job: TrainingJob, physical_batch_size: int) -> TrainingJob:
        """Preserve the approved exposure/update contract after placement changes batch."""
        updated_job = job.copy()
        metadata = updated_job.metadata
        kwargs = updated_job.config.runner_kwargs
        contract = dict(metadata.get("training_quality_contract") or {})

        base_physical = TrainingParameterResolution._positive_int(
            contract.get("proposed_physical_batch_size"),
            BatchResolution.requested_batch_size(job),
        ) or 1
        base_accumulation = TrainingParameterResolution._positive_int(
            contract.get("base_gradient_accumulation_steps"),
            TrainingParameterResolution._positive_int(
                metadata.get("gradient_accumulation_steps"), 1
            ),
        ) or 1
        target_effective = TrainingParameterResolution._positive_int(
            contract.get("target_effective_batch_size"),
            base_physical * base_accumulation,
        ) or base_physical * base_accumulation

        # Smaller physical batches preserve optimizer-update semantics through
        # accumulation. Agent-approved larger batches may raise the effective
        # batch and use the explicit learning-rate scaling policy below.
        resolved_accumulation = max(
            1, int(math.ceil(float(target_effective) / float(physical_batch_size)))
        )
        resolved_effective = int(physical_batch_size) * resolved_accumulation
        effective_ratio = float(resolved_effective) / float(max(1, target_effective))

        base_learning_rate = TrainingParameterResolution._positive_float(
            contract.get("base_learning_rate", metadata.get("learning_rate"))
        )
        lr_policy = str(contract.get("learning_rate_scaling_policy") or "fixed").lower()
        resolved_learning_rate = base_learning_rate
        if base_learning_rate is not None:
            if lr_policy in {"linear", "linear_with_effective_batch"}:
                resolved_learning_rate = base_learning_rate * effective_ratio
            elif lr_policy in {"sqrt", "square_root"}:
                resolved_learning_rate = base_learning_rate * math.sqrt(effective_ratio)

        update_count_ratio = float(target_effective) / float(max(1, resolved_effective))
        base_warmup_steps = TrainingParameterResolution._nonnegative_int(
            contract.get("base_warmup_steps", metadata.get("warmup_steps"))
        )
        base_scheduler_steps = TrainingParameterResolution._positive_int(
            contract.get(
                "base_scheduler_total_steps", metadata.get("scheduler_total_steps")
            )
        )
        resolved_warmup_steps = (
            max(0, int(round(base_warmup_steps * update_count_ratio)))
            if base_warmup_steps is not None
            else None
        )
        resolved_scheduler_steps = (
            max(1, int(round(base_scheduler_steps * update_count_ratio)))
            if base_scheduler_steps is not None
            else None
        )

        resolution = {
            "physical_batch_size": int(physical_batch_size),
            "gradient_accumulation_steps": resolved_accumulation,
            "effective_batch_size": resolved_effective,
            "learning_rate": resolved_learning_rate,
            "warmup_steps": resolved_warmup_steps,
            "scheduler_total_steps": resolved_scheduler_steps,
            "learning_rate_scaling_policy": lr_policy,
        }
        metadata.update(
            {
                "resolved_gradient_accumulation_steps": resolved_accumulation,
                "resolved_effective_batch_size": resolved_effective,
                "effective_batch_size": resolved_effective,
                "resolved_learning_rate": resolved_learning_rate,
                "resolved_warmup_steps": resolved_warmup_steps,
                "resolved_scheduler_total_steps": resolved_scheduler_steps,
                "training_parameter_resolution": resolution,
            }
        )
        kwargs["gradient_accumulation_steps"] = resolved_accumulation
        kwargs["effective_batch_size"] = resolved_effective
        if resolved_learning_rate is not None:
            kwargs["learning_rate"] = resolved_learning_rate
        if resolved_warmup_steps is not None:
            kwargs["warmup_steps"] = resolved_warmup_steps
        if resolved_scheduler_steps is not None:
            kwargs["scheduler_total_steps"] = resolved_scheduler_steps
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
    runner_kwargs = {key: value for key, value in dict(job.config.runner_kwargs).items() if key not in ignored_runner_kwargs}
    runner_kwargs.pop(batch_param_name, None)
    payload = {
        "runner_target": job.config.runner_target,
        "task_type": job.task_type,
        "loader_target": job.config.loader_target,
        "runner_kwargs": runner_kwargs,
        "shape_hints": job.batch_probe.shape_hints,
    }
    return sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
