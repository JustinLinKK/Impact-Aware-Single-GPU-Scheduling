"""Runtime profile helpers for duration-aware scheduling."""

from __future__ import annotations

from typing import Any

from ..domain import (
    BatchResolution,
    ProgressSnapshot,
    RuntimeProfile,
    TrainingJob,
    normalize_runtime_probe_strategy,
)
from ..storage.sqlite_store import SQLiteStateStore


def resolved_batch_size(job: TrainingJob) -> int:
    return BatchResolution.resolved_batch_size(job)


def runtime_profile_for_job(
    store: SQLiteStateStore,
    job: TrainingJob,
    *,
    backend_name: str | None = None,
) -> RuntimeProfile | None:
    signature = job.packing.signature
    if not signature:
        return None
    return store.get_runtime_profile(
        signature,
        resolved_batch_size=resolved_batch_size(job),
        backend_name=str(backend_name or job.metadata.get("placement_backend") or "exclusive"),
    )


def runtime_ready_for_packing(store: SQLiteStateStore, job: TrainingJob, *, backend_name: str) -> bool:
    if not job.runtime_probe.enabled:
        return False
    return runtime_profile_for_job(store, job, backend_name=backend_name) is not None


def planned_total_epochs(job: TrainingJob) -> int | None:
    raw_value = job.max_epochs or job.config.max_epochs
    if raw_value is None:
        return None
    try:
        return max(0, int(raw_value))
    except (TypeError, ValueError):
        return None


def planned_total_steps(job: TrainingJob, *, steps_per_epoch: int | None = None) -> int | None:
    raw_steps = job.max_steps or job.config.max_steps
    if raw_steps is not None:
        try:
            return max(0, int(raw_steps))
        except (TypeError, ValueError):
            return None
    epochs = planned_total_epochs(job)
    if epochs is None or steps_per_epoch is None:
        return None
    return max(0, int(epochs) * max(1, int(steps_per_epoch)))


def estimate_total_runtime_from_epoch_1(
    *,
    startup_seconds: float,
    epoch_1_seconds: float,
    total_epochs: int | None,
    checkpoint_overhead_seconds: float = 0.0,
) -> float:
    if total_epochs is None or total_epochs <= 0:
        return max(0.0, float(startup_seconds + epoch_1_seconds + checkpoint_overhead_seconds))
    return max(
        0.0,
        float(startup_seconds) + (float(epoch_1_seconds) * float(total_epochs)) + float(checkpoint_overhead_seconds),
    )


def estimate_total_runtime_from_step_window(
    *,
    startup_seconds: float,
    avg_step_time_ms: float,
    total_steps: int | None,
    checkpoint_overhead_seconds: float = 0.0,
) -> float:
    if total_steps is None or total_steps <= 0:
        total_steps = 1
    return max(
        0.0,
        float(startup_seconds) + ((float(avg_step_time_ms) * float(total_steps)) / 1000.0) + float(checkpoint_overhead_seconds),
    )


def estimate_remaining_runtime_seconds(
    job: TrainingJob,
    snapshot: ProgressSnapshot | None,
    *,
    fallback_total_runtime_seconds: float | None = None,
) -> float | None:
    if snapshot is not None and snapshot.remaining_runtime_seconds is not None:
        return max(0.0, float(snapshot.remaining_runtime_seconds))
    estimated_total = fallback_total_runtime_seconds
    if snapshot is not None and snapshot.estimated_total_runtime_seconds is not None:
        estimated_total = float(snapshot.estimated_total_runtime_seconds)
    if estimated_total is None:
        estimated_total = float(job.metadata.get("runtime_estimated_total_runtime_seconds") or 0.0) or None
    if estimated_total is None:
        return None
    if snapshot is None:
        return max(0.0, estimated_total)

    total_steps = planned_total_steps(job, steps_per_epoch=snapshot.steps_per_epoch)
    if total_steps is not None and total_steps > 0:
        progress = min(1.0, max(0.0, float(snapshot.global_step) / float(total_steps)))
        return max(0.0, estimated_total * (1.0 - progress))

    total_epochs = planned_total_epochs(job)
    if total_epochs is not None and total_epochs > 0:
        progress = min(1.0, max(0.0, float(snapshot.epoch) / float(total_epochs)))
        return max(0.0, estimated_total * (1.0 - progress))
    return max(0.0, estimated_total)


def build_runtime_profile(
    job: TrainingJob,
    *,
    hardware_key: str,
    backend_name: str,
    strategy: str,
    startup_seconds: float | None,
    epoch_1_seconds: float | None,
    steps_per_epoch: int | None,
    avg_step_time_ms: float | None,
    estimated_total_runtime_seconds: float | None,
    confidence: float,
    source: str,
    observations: int,
    last_job_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> RuntimeProfile:
    return RuntimeProfile.create(
        signature=job.packing.signature or job.job_id,
        hardware_key=hardware_key,
        backend_name=backend_name,
        resolved_batch_size=resolved_batch_size(job),
        strategy=normalize_runtime_probe_strategy(strategy),
        startup_seconds=startup_seconds,
        epoch_1_seconds=epoch_1_seconds,
        steps_per_epoch=steps_per_epoch,
        avg_step_time_ms=avg_step_time_ms,
        estimated_total_runtime_seconds=estimated_total_runtime_seconds,
        confidence=confidence,
        observations=observations,
        last_job_id=last_job_id or job.job_id,
        source=source,
        metadata=metadata or {},
    )
