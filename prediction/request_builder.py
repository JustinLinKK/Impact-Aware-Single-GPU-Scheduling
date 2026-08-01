"""Build typed prediction requests from scheduler jobs."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from ..domain import PredictionRequest, TrainingJob
from .branch_adapter import branch_prediction_metadata


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_shape(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.replace("x", ",").replace("X", ",")
        parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        return None
    shape: list[int] = []
    for part in parts:
        coerced = _as_int(part)
        if coerced is None:
            return None
        shape.append(coerced)
    return tuple(shape) if shape else None


def _dataloader_config(job: TrainingJob) -> dict[str, object]:
    keys = (
        "num_workers",
        "pin_memory",
        "prefetch_factor",
        "persistent_workers",
        "drop_last",
        "shuffle",
    )
    config: dict[str, object] = {}
    for key in keys:
        if key in job.config.runner_kwargs:
            config[key] = job.config.runner_kwargs[key]
        elif key in job.metadata:
            config[key] = job.metadata[key]
    return config


def build_prediction_request(
    job: TrainingJob,
    *,
    hardware_key: str,
    backend: str,
    batch_size: int,
) -> PredictionRequest:
    kwargs = job.config.runner_kwargs
    metadata = job.metadata
    architecture_source = str(
        _first_present(
            metadata.get("architecture_source"),
            metadata.get("source_path"),
            job.baseline_model_path,
            job.config.runner_target,
        )
        or ""
    )
    architecture_hash = str(
        _first_present(
            metadata.get("architecture_source_hash"),
            metadata.get("source_hash"),
            job.packing.signature,
        )
        or ""
    )
    if not architecture_hash:
        architecture_hash = sha256(
            "|".join(
                [
                    architecture_source,
                    job.config.runner_target,
                    job.baseline_model_id,
                    str(job.config.runner_kwargs),
                ]
            ).encode("utf-8")
        ).hexdigest()
    input_shape = _as_shape(
        _first_present(
            metadata.get("input_shape"),
            kwargs.get("input_shape"),
            kwargs.get("shape"),
            job.batch_probe.shape_hints.get("input_shape"),
            job.batch_probe.shape_hints.get("shape"),
        )
    )
    micro_batch_size = _as_int(_first_present(kwargs.get("micro_batch_size"), metadata.get("micro_batch_size")))
    if micro_batch_size is None:
        micro_batch_size = int(batch_size)
    grad_accum = _as_int(
        _first_present(
            kwargs.get("gradient_accumulation_steps"),
            kwargs.get("grad_accumulation_steps"),
            metadata.get("gradient_accumulation_steps"),
        )
    )
    total_epochs = _as_int(_first_present(job.max_epochs, job.config.max_epochs, kwargs.get("epochs"), metadata.get("total_epochs")))
    completed_epochs = _as_float(_first_present(metadata.get("completed_epochs"), metadata.get("runtime_completed_epochs")))
    remaining_epochs = _as_float(metadata.get("remaining_epochs"))
    if remaining_epochs is None and total_epochs is not None and completed_epochs is not None:
        remaining_epochs = max(0.0, float(total_epochs) - completed_epochs)
    return PredictionRequest(
        job_id=job.job_id,
        architecture_source=architecture_source,
        architecture_source_hash=architecture_hash,
        packing_signature=job.packing.signature,
        hardware_key=hardware_key,
        backend=backend,
        batch_size=max(1, int(batch_size)),
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=max(1, grad_accum or 1),
        precision=str(_first_present(kwargs.get("precision"), kwargs.get("dtype"), metadata.get("precision"), "unknown")),
        input_shape=input_shape,
        optimizer_name=_first_present(kwargs.get("optimizer_name"), kwargs.get("optimizer"), metadata.get("optimizer_name")),
        dataset_size=_as_int(_first_present(kwargs.get("dataset_size"), metadata.get("dataset_size"))),
        steps_per_epoch=_as_int(_first_present(kwargs.get("steps_per_epoch"), metadata.get("steps_per_epoch"))),
        total_epochs=total_epochs,
        remaining_epochs=remaining_epochs,
        dataloader_config=_dataloader_config(job),
        extra_features={
            "task_type": job.task_type,
            "baseline_model_id": job.baseline_model_id,
            "runner_target": job.config.runner_target,
            "branch_prediction": branch_prediction_metadata(metadata),
            "input_resolution": _first_present(
                metadata.get("input_resolution"), kwargs.get("input_resolution")
            ),
        },
    )
