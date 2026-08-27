"""Exclusive-path batch-size probing and cache reuse."""

from __future__ import annotations

from dataclasses import dataclass
import gc
from typing import Any

import torch

from ..execution.runner_protocol import BatchProbeProtocol, RunnerContext
from ..domain import (
    BatchResolution,
    BatchSizeObservation,
    BatchProbeProfile,
    BatchProbeTrialResult,
    SchedulingClass,
    TrainingJob,
    build_batch_size_observation_key,
    build_batch_probe_key,
    build_batch_probe_shape_signature,
    import_string,
)


@dataclass(slots=True)
class BatchProbeKeyInfo:
    probe_key: str
    model_key: str
    device_type: str
    shape_signature: str


@dataclass(slots=True)
class ProbeAttempt:
    batch_size: int
    result: BatchProbeTrialResult
    within_budget: bool
    target_budget_mb: int


def _requires_probe(job: TrainingJob) -> bool:
    if bool(job.metadata.get("skip_active_scheduler_probes")):
        return False
    backend_name = str(job.metadata.get("placement_backend", ""))
    return job.resource_requirements.requires_gpu and job.batch_probe.enabled and backend_name == "exclusive"


def resolve_visible_device_type() -> str:
    if torch.cuda.is_available():
        try:
            return str(torch.cuda.get_device_name(torch.cuda.current_device()))
        except Exception:
            return "cuda-visible-device"
    return "cuda-unavailable"


def _visible_device_total_mb() -> int | None:
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            return max(1, int(props.total_memory / (1024 * 1024)))
        except Exception:
            return None
    return None


def _probe_key_info(job: TrainingJob) -> BatchProbeKeyInfo:
    model_key = str(job.batch_probe.model_key or job.baseline_model_id)
    device_type = resolve_visible_device_type()
    shape_signature = build_batch_probe_shape_signature(job)
    return BatchProbeKeyInfo(
        probe_key=build_batch_probe_key(model_key, device_type, shape_signature),
        model_key=model_key,
        device_type=device_type,
        shape_signature=shape_signature,
    )


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _cleanup_after_trial() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
        except Exception:
            pass


def _coerce_trial_result(
    value: BatchProbeTrialResult | dict[str, Any],
) -> BatchProbeTrialResult:
    if isinstance(value, BatchProbeTrialResult):
        return value
    return BatchProbeTrialResult.from_dict(dict(value))


def _run_trial(
    context: RunnerContext,
    probe: BatchProbeProtocol,
    batch_size: int,
    *,
    warmup_steps: int,
    measure_steps: int,
) -> ProbeAttempt:
    memory = context.settings.gpu_scheduler.memory
    if memory.gpu_vram_gib is not None:
        memory_cap_mb = int(memory.gpu_vram_gib * 1024 * memory.predicted_budget_fraction)
    else:
        memory_cap_mb = int((_visible_device_total_mb() or 1) * memory.predicted_budget_fraction)
    try:
        result = _coerce_trial_result(probe(context, batch_size, warmup_steps, measure_steps))
    except Exception as exc:
        result = BatchProbeTrialResult(
            fits=False,
            peak_vram_mb=None,
            memory_total_mb=_visible_device_total_mb(),
            message=str(exc),
        )
    device_total_mb = result.memory_total_mb or _visible_device_total_mb()
    if device_total_mb is not None:
        effective_budget_mb = int(min(device_total_mb, memory_cap_mb) * context.settings.gpu_scheduler.batch_probe_target_memory_fraction)
    else:
        effective_budget_mb = int(memory_cap_mb * context.settings.gpu_scheduler.batch_probe_target_memory_fraction)
    within_budget = result.peak_vram_mb is None or result.peak_vram_mb <= effective_budget_mb
    context.event_logger.emit(
        "batch_probe_trial",
        job_id=context.job.job_id,
        payload={
            "batch_size": batch_size,
            "fits": result.fits,
            "within_budget": within_budget,
            "peak_vram_mb": result.peak_vram_mb,
            "memory_total_mb": result.memory_total_mb,
            "target_budget_mb": effective_budget_mb,
            "message": result.message,
        },
    )
    return ProbeAttempt(
        batch_size=batch_size,
        result=result,
        within_budget=within_budget,
        target_budget_mb=effective_budget_mb,
    )


def _attempt_successful(attempt: ProbeAttempt) -> bool:
    return bool(attempt.result.fits and attempt.within_budget)


def _run_time_aware_five_option_probe(context: RunnerContext, key_info: BatchProbeKeyInfo) -> BatchProbeProfile:
    if not context.job.batch_probe.probe_target:
        raise ValueError("batch_probe.probe_target is required for an exclusive probe")
    requested = BatchResolution.requested_batch_size(context.job)
    if context.settings.gpu_scheduler.batch_options.require_power_of_two_original and not _is_power_of_two(requested):
        raise ValueError("time-aware exclusive probes require a power-of-two requested batch size")
    contract = dict(context.job.metadata.get("training_quality_contract") or {})
    approved_values = contract.get("allowed_physical_batch_sizes") or []
    if approved_values:
        proposals = sorted({max(1, int(value)) for value in approved_values})
    else:
        exponent = requested.bit_length() - 1
        proposals = [2 ** max(0, exponent + offset) for offset in context.settings.gpu_scheduler.batch_options.exponent_offsets]
    cap = context.job.config.runner_kwargs.get(
        "probe_max_batch_size",
        context.settings.gpu_scheduler.batch_probe_max_batch_size,
    )
    if approved_values and cap is not None:
        clipped = [value for value in proposals if value <= max(1, int(cap))]
    else:
        clipped = [min(value, max(1, int(cap))) if cap is not None else value for value in proposals]
    if not clipped:
        clipped = [requested]
    candidates = list(dict.fromkeys(max(1, int(value)) for value in clipped))
    probe = import_string(context.job.batch_probe.probe_target)
    warmup_steps = int(context.settings.gpu_scheduler.profiling.warmup_steps)
    measure_steps = int(context.settings.gpu_scheduler.profiling.solo_probe_steps)
    attempts: list[ProbeAttempt] = []
    hardware_key = context.store.hardware_key()
    for batch_size in candidates:
        attempt = _run_trial(
            context,
            probe,
            batch_size,
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
        )
        attempts.append(attempt)
        existing = context.store.get_batch_size_observation(
            model_key=key_info.model_key,
            shape_signature=key_info.shape_signature,
            hardware_key=hardware_key,
            backend_name="exclusive",
            batch_size=batch_size,
        )
        seconds_per_epoch = attempt.result.seconds_per_epoch
        if seconds_per_epoch is None and attempt.result.avg_step_time_ms is not None and attempt.result.steps_per_epoch:
            seconds_per_epoch = attempt.result.avg_step_time_ms * attempt.result.steps_per_epoch / 1000.0
        resolved_candidate = BatchResolution.apply(context.job, batch_size)
        context.store.upsert_batch_size_observation(
            BatchSizeObservation(
                observation_key=build_batch_size_observation_key(
                    key_info.model_key,
                    key_info.shape_signature,
                    hardware_key,
                    "exclusive",
                    batch_size,
                ),
                model_key=key_info.model_key,
                shape_signature=key_info.shape_signature,
                hardware_key=hardware_key,
                backend_name="exclusive",
                batch_param_name=BatchResolution.param_name(context.job),
                batch_size=batch_size,
                effective_batch_size=resolved_candidate.metadata.get(
                    "resolved_effective_batch_size"
                ),
                peak_vram_mb=attempt.result.peak_vram_mb,
                avg_vram_mb=attempt.result.avg_vram_mb,
                memory_total_mb=attempt.result.memory_total_mb,
                avg_step_time_ms=attempt.result.avg_step_time_ms,
                best_metric=existing.best_metric if existing else None,
                metric_name=existing.metric_name if existing else None,
                metric_maximize=existing.metric_maximize if existing else None,
                best_epoch=existing.best_epoch if existing else None,
                planned_epochs=existing.planned_epochs if existing else None,
                completed_epochs=existing.completed_epochs if existing else None,
                convergence_curve=(existing.convergence_curve if existing else []),
                seed_variance=existing.seed_variance if existing else None,
                observations=(existing.observations + 1) if existing else 1,
                last_job_id=context.job.job_id,
                metadata={
                    **(existing.metadata if existing else {}),
                    "fits": attempt.result.fits,
                    "within_budget": attempt.within_budget,
                    "message": attempt.result.message,
                    "steps_per_epoch": attempt.result.steps_per_epoch,
                    "seconds_per_epoch": seconds_per_epoch,
                    "estimate_source": "probe",
                },
            )
        )
        _cleanup_after_trial()
    successful = [attempt for attempt in attempts if _attempt_successful(attempt)]
    if not successful:
        raise RuntimeError("exclusive probe found no feasible batch option")
    resolved = min(
        successful,
        key=lambda attempt: (abs(attempt.batch_size - requested), attempt.batch_size),
    )
    existing_profile = context.store.get_batch_probe_profile(key_info.probe_key)
    return BatchProbeProfile(
        probe_key=key_info.probe_key,
        model_key=key_info.model_key,
        device_type=key_info.device_type,
        shape_signature=key_info.shape_signature,
        batch_param_name=BatchResolution.param_name(context.job),
        resolved_batch_size=resolved.batch_size,
        peak_vram_mb=resolved.result.peak_vram_mb,
        avg_vram_mb=resolved.result.avg_vram_mb,
        memory_total_mb=resolved.result.memory_total_mb,
        target_budget_mb=resolved.target_budget_mb,
        observations=(existing_profile.observations + 1) if existing_profile else 1,
        last_job_id=context.job.job_id,
        metadata={
            "source": "exclusive_five_option_probe",
            "proposed_batch_sizes": proposals,
            "clipped_batch_sizes": candidates,
            "measurements": [
                {
                    "batch_size": attempt.batch_size,
                    "fits": attempt.result.fits,
                    "within_budget": attempt.within_budget,
                    "avg_vram_mb": attempt.result.avg_vram_mb,
                    "seconds_per_epoch": attempt.result.seconds_per_epoch,
                }
                for attempt in attempts
            ],
        },
    )


def _persist_resolved_batch_size(
    context: RunnerContext,
    *,
    probe_key: str,
    device_type: str,
    batch_param_name: str,
    resolved_batch_size: int,
    source: str,
) -> TrainingJob:
    job = context.store.get_job(context.job.job_id) or context.job
    job = BatchResolution.apply(job, resolved_batch_size)
    job.metadata.update(
        {
            "batch_probe_source": source,
            "batch_probe_key": probe_key,
            "batch_probe_device_type": device_type,
        }
    )
    context.store.save_job(job)
    return job


def run_batch_probe_preflight(context: RunnerContext) -> TrainingJob:
    if (
        not context.settings.gpu_scheduler.batch_probe_enabled
        or context.job.scheduling_class != SchedulingClass.EXCLUSIVE_PROBE
        or not _requires_probe(context.job)
    ):
        return context.job
    if not context.job.batch_probe.probe_target:
        raise ValueError("batch_probe.probe_target is required when batch_probe.enabled is true")

    key_info = _probe_key_info(context.job)
    batch_param_name = BatchResolution.param_name(context.job)
    profile = _run_time_aware_five_option_probe(context, key_info)
    context.store.upsert_batch_probe_profile(profile)
    context.job = _persist_resolved_batch_size(
        context,
        probe_key=key_info.probe_key,
        device_type=key_info.device_type,
        batch_param_name=batch_param_name,
        resolved_batch_size=profile.resolved_batch_size,
        source="exclusive_five_option_probe",
    )
    context.event_logger.emit(
        "exclusive_probe_measurements_persisted",
        job_id=context.job.job_id,
        payload={
            "probe_key": key_info.probe_key,
            "proposed_batch_sizes": profile.metadata["proposed_batch_sizes"],
            "clipped_batch_sizes": profile.metadata["clipped_batch_sizes"],
        },
    )
    return context.job
