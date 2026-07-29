"""Deterministic runner fixture exercising worker-level early-stop handling."""

from __future__ import annotations

from typing import Any

from localml_scheduler.domain import SafePointType
from localml_scheduler.execution.runner_protocol import RunnerContext


def run_validation_sequence(context: RunnerContext) -> dict[str, Any]:
    values = [float(value) for value in context.job.config.runner_kwargs.get("validation_accuracy", [])]
    for epoch, accuracy in enumerate(values, start=1):
        context.control_hook.safe_point(
            SafePointType.EPOCH,
            epoch=epoch,
            global_step=epoch,
            metrics={"accuracy": accuracy},
            state_factory=lambda epoch=epoch, accuracy=accuracy: {
                "epoch": epoch,
                "accuracy": accuracy,
            },
            steps_per_epoch=1,
            avg_step_time_ms=10.0,
            estimated_total_runtime_seconds=float(len(values)) * 0.01,
            remaining_runtime_seconds=float(max(0, len(values) - epoch)) * 0.01,
        )
    return {"accuracy": values[-1] if values else None, "epochs_ran": len(values)}
