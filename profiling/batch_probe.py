"""Exclusive-path batch-size probing and cache reuse."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import logging
from typing import Any

import torch

from ..execution.runner_protocol import BatchProbeProtocol, RunnerContext
from ..schemas import (
    BatchProbeProfile,
    BatchProbeTrialResult,
    TrainingJob,
    build_batch_probe_key,
    build_batch_probe_shape_signature,
    import_string,
)

logger = logging.getLogger("localml_scheduler")


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


def _message_indicates_oom(message: str | None) -> bool:
    lowered = str(message or "").lower()
    return "out of memory" in lowered or "cuda out of memory" in lowered


def _failure_due_to_memory(attempt: ProbeAttempt | None) -> bool:
    if attempt is None:
        return False
    if not attempt.within_budget:
        return True
    return _message_indicates_oom(attempt.result.message)


def _probe_warning_details(
    resolved: ProbeAttempt,
    *,
    failure: ProbeAttempt | None,
    stop_reason: str,
    max_batch_size: int | None,
    max_search_rounds: int,
    rounds: int,
) -> tuple[str | None, str | None, bool]:
    target_budget_mb = max(1, int(resolved.target_budget_mb))
    peak_vram_mb = resolved.result.peak_vram_mb
    utilization = (float(peak_vram_mb) / float(target_budget_mb)) if peak_vram_mb is not None else None
    saturated_vram = utilization is not None and utilization >= 0.9
    if saturated_vram:
        return None, None, True

    if stop_reason == "max_batch_size_cap":
        return (
            "max_batch_size_cap",
            f"probe stopped at configured max batch size {max_batch_size} before VRAM saturation",
            False,
        )
    if stop_reason == "max_search_rounds":
        return (
            "max_search_rounds",
            f"probe exhausted max search rounds ({max_search_rounds}) before VRAM saturation",
            False,
        )
    if failure is not None and not _failure_due_to_memory(failure):
        failure_detail = (failure.result.message or "non-memory failure").strip().replace("\n", " ")
        return (
            "non_memory_failure_boundary",
            f"probe stopped due to a non-memory failure at batch size {failure.batch_size}: {failure_detail}",
            False,
        )
    if utilization is not None and utilization < 0.9:
        return (
            "underfilled_vram",
            f"probe selected batch size {resolved.batch_size} at only {utilization:.0%} of the VRAM target",
            False,
        )
    if peak_vram_mb is None and rounds >= max_search_rounds:
        return (
            "unknown_vram_max_search_rounds",
            f"probe exhausted max search rounds ({max_search_rounds}) without enough VRAM telemetry to confirm saturation",
            False,
        )
    return None, None, False


def _is_raw_script_job(job: TrainingJob) -> bool:
    return job.config.runner_target == "localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job"


def _requires_probe(job: TrainingJob) -> bool:
    backend_name = str(job.metadata.get("placement_backend", ""))
    return (
        job.resource_requirements.requires_gpu
        and job.batch_probe.enabled
        and backend_name == "exclusive"
    )


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


def _coerce_trial_result(value: BatchProbeTrialResult | dict[str, Any]) -> BatchProbeTrialResult:
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
    memory_cap_mb = int(context.settings.gpu_scheduler.memory.safe_vram_budget_gib * 1024)
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
        effective_budget_mb = int(
            min(device_total_mb, memory_cap_mb) * context.settings.gpu_scheduler.batch_probe_target_memory_fraction
        )
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
    message = (result.message or "").strip().replace("\n", " ")
    if len(message) > 180:
        message = f"{message[:177]}..."
    logger.info(
        "[batch_probe] job=%s trial batch_size=%s fits=%s within_budget=%s peak_vram_mb=%s target_budget_mb=%s detail=%s",
        context.job.job_id,
        batch_size,
        result.fits,
        within_budget,
        result.peak_vram_mb,
        effective_budget_mb,
        message or "-",
    )
    return ProbeAttempt(
        batch_size=batch_size,
        result=result,
        within_budget=within_budget,
        target_budget_mb=effective_budget_mb,
    )


def _attempt_successful(attempt: ProbeAttempt) -> bool:
    return bool(attempt.result.fits and attempt.within_budget)


def _resolve_start_batch_size(job: TrainingJob) -> int:
    batch_param_name = job.batch_probe.batch_param_name or "batch_size"
    raw_value = job.config.runner_kwargs.get(batch_param_name)
    if raw_value is None:
        return max(1, int(job.metadata.get("resolved_batch_size") or 1))
    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        return 1


def _run_probe_controller(context: RunnerContext, key_info: BatchProbeKeyInfo) -> BatchProbeProfile:
    if not context.job.batch_probe.probe_target:
        raise ValueError("batch_probe.probe_target is required when batch_probe.enabled is true")

    probe = import_string(context.job.batch_probe.probe_target)
    warmup_steps = int(context.settings.gpu_scheduler.profiling.warmup_steps)
    measure_steps = int(context.settings.gpu_scheduler.profiling.solo_probe_steps)
    min_batch_size = max(1, int(context.settings.gpu_scheduler.batch_probe_min_batch_size))
    max_search_rounds = max(1, int(context.settings.gpu_scheduler.batch_probe_max_search_rounds))
    max_batch_size = context.job.config.runner_kwargs.get("probe_max_batch_size", context.settings.gpu_scheduler.batch_probe_max_batch_size)
    if max_batch_size is not None:
        max_batch_size = max(min_batch_size, int(max_batch_size))

    start_batch_size = max(min_batch_size, _resolve_start_batch_size(context.job))
    if max_batch_size is not None:
        start_batch_size = min(start_batch_size, max_batch_size)

    context.event_logger.emit(
        "batch_probe_started",
        job_id=context.job.job_id,
        payload={
            "probe_key": key_info.probe_key,
            "model_key": key_info.model_key,
            "device_type": key_info.device_type,
            "shape_signature": key_info.shape_signature,
            "start_batch_size": start_batch_size,
        },
    )
    logger.info(
        "[batch_probe] job=%s start probe_key=%s model_key=%s device=%s start_batch_size=%s max_batch_size=%s warmup_steps=%s measure_steps=%s",
        context.job.job_id,
        key_info.probe_key,
        key_info.model_key,
        key_info.device_type,
        start_batch_size,
        max_batch_size,
        warmup_steps,
        measure_steps,
    )

    rounds = 0
    stop_reason = "resolved"

    def run_attempt(batch_size: int) -> ProbeAttempt:
        nonlocal rounds
        if rounds >= max_search_rounds:
            raise RuntimeError("batch probe exhausted max search rounds")
        rounds += 1
        attempt = _run_trial(context, probe, batch_size, warmup_steps=warmup_steps, measure_steps=measure_steps)
        _cleanup_after_trial()
        return attempt

    current_batch_size = start_batch_size
    success: ProbeAttempt | None = None
    failure: ProbeAttempt | None = None
    search_method = "initial"

    while True:
        attempt = run_attempt(current_batch_size)
        if _attempt_successful(attempt):
            success = attempt
            stop_reason = "initial_success"
            break
        failure = attempt
        if current_batch_size <= min_batch_size:
            reason = attempt.result.message or f"batch size {current_batch_size} did not fit"
            raise RuntimeError(f"batch probe could not find a feasible batch size: {reason}")
        current_batch_size = max(min_batch_size, current_batch_size // 2)
        search_method = "downshift"

    last_good = success
    low = success.batch_size
    high: int | None = None

    candidate = low * 2
    while rounds < max_search_rounds and (max_batch_size is None or candidate <= max_batch_size):
        attempt = run_attempt(candidate)
        if _attempt_successful(attempt):
            last_good = attempt
            low = attempt.batch_size
            candidate = attempt.batch_size * 2
            search_method = "expand"
            stop_reason = "expand_success"
            continue
        high = attempt.batch_size
        failure = attempt
        search_method = "binary"
        stop_reason = "failure_boundary"
        break

    if high is not None:
        while (high - low) > 1 and rounds < max_search_rounds:
            mid = low + ((high - low) // 2)
            attempt = run_attempt(mid)
            if _attempt_successful(attempt):
                last_good = attempt
                low = attempt.batch_size
                stop_reason = "binary_success"
            else:
                failure = attempt
                high = attempt.batch_size
                stop_reason = "failure_boundary"

    if high is None:
        if rounds >= max_search_rounds:
            stop_reason = "max_search_rounds"
        elif max_batch_size is not None and candidate > max_batch_size:
            stop_reason = "max_batch_size_cap"

    existing = context.store.get_batch_probe_profile(key_info.probe_key)
    resolved = last_good
    warning_reason, warning_message, saturated_vram = _probe_warning_details(
        resolved,
        failure=failure,
        stop_reason=stop_reason,
        max_batch_size=max_batch_size,
        max_search_rounds=max_search_rounds,
        rounds=rounds,
    )
    return BatchProbeProfile(
        probe_key=key_info.probe_key,
        model_key=key_info.model_key,
        device_type=key_info.device_type,
        shape_signature=key_info.shape_signature,
        batch_param_name=context.job.batch_probe.batch_param_name or "batch_size",
        resolved_batch_size=resolved.batch_size,
        peak_vram_mb=resolved.result.peak_vram_mb,
        memory_total_mb=resolved.result.memory_total_mb,
        target_budget_mb=resolved.target_budget_mb,
        observations=(existing.observations + 1) if existing else 1,
        last_job_id=context.job.job_id,
        metadata={
            "probe_target": context.job.batch_probe.probe_target,
            "search_method": search_method,
            "stop_reason": stop_reason,
            "failure_batch_size": failure.batch_size if failure else None,
            "avg_step_time_ms": resolved.result.avg_step_time_ms,
            "model_key": key_info.model_key,
            "device_type": key_info.device_type,
            "shape_signature": key_info.shape_signature,
            "warning_reason": warning_reason,
            "warning_message": warning_message,
            "saturated_vram": saturated_vram,
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
    job.config.runner_kwargs[batch_param_name] = int(resolved_batch_size)
    job.metadata.update(
        {
            "resolved_batch_size": int(resolved_batch_size),
            "batch_probe_source": source,
            "batch_probe_key": probe_key,
            "batch_probe_device_type": device_type,
        }
    )
    context.store.save_job(job)
    return job


def _job_has_resolved_batch_size(job: TrainingJob, key_info: BatchProbeKeyInfo) -> bool:
    batch_param_name = job.batch_probe.batch_param_name or "batch_size"
    if job.metadata.get("batch_probe_key") != key_info.probe_key:
        return False
    if job.metadata.get("resolved_batch_size") is None:
        return False
    return batch_param_name in job.config.runner_kwargs


def run_batch_probe_preflight(context: RunnerContext) -> TrainingJob:
    if not context.settings.gpu_scheduler.batch_probe_enabled or not _requires_probe(context.job):
        return context.job
    if not context.job.batch_probe.probe_target:
        raise ValueError("batch_probe.probe_target is required when batch_probe.enabled is true")

    key_info = _probe_key_info(context.job)
    batch_param_name = context.job.batch_probe.batch_param_name or "batch_size"

    if _job_has_resolved_batch_size(context.job, key_info):
        logger.info(
            "[batch_probe] job=%s reusing previously resolved batch size=%s for probe_key=%s",
            context.job.job_id,
            context.job.metadata.get("resolved_batch_size"),
            key_info.probe_key,
        )
        return context.job

    cached = context.store.get_batch_probe_profile(key_info.probe_key)
    if cached is not None:
        context.event_logger.emit(
            "batch_probe_cache_hit",
            job_id=context.job.job_id,
            payload={
                "probe_key": key_info.probe_key,
                "device_type": key_info.device_type,
                "resolved_batch_size": cached.resolved_batch_size,
            },
        )
        logger.info(
            "[batch_probe] job=%s cache hit probe_key=%s resolved_batch_size=%s device=%s",
            context.job.job_id,
            key_info.probe_key,
            cached.resolved_batch_size,
            key_info.device_type,
        )
        cached_warning_message = str((cached.metadata or {}).get("warning_message") or "").strip()
        if cached_warning_message:
            context.event_logger.emit(
                "batch_probe_warning",
                job_id=context.job.job_id,
                payload={
                    "probe_key": key_info.probe_key,
                    "device_type": key_info.device_type,
                    "resolved_batch_size": cached.resolved_batch_size,
                    "warning_reason": (cached.metadata or {}).get("warning_reason"),
                    "warning_message": cached_warning_message,
                    "source": "cache",
                },
            )
            logger.warning(
                "[batch_probe] job=%s warning probe_key=%s source=cache message=%s",
                context.job.job_id,
                key_info.probe_key,
                cached_warning_message,
            )
        context.job = _persist_resolved_batch_size(
            context,
            probe_key=key_info.probe_key,
            device_type=key_info.device_type,
            batch_param_name=batch_param_name,
            resolved_batch_size=cached.resolved_batch_size,
            source="cache",
        )
        return context.job

    context.event_logger.emit(
        "batch_probe_cache_miss",
        job_id=context.job.job_id,
        payload={"probe_key": key_info.probe_key, "device_type": key_info.device_type},
    )
    logger.info(
        "[batch_probe] job=%s cache miss probe_key=%s device=%s",
        context.job.job_id,
        key_info.probe_key,
        key_info.device_type,
    )

    try:
        profile = _run_probe_controller(context, key_info)
    except Exception as exc:
        context.event_logger.emit(
            "batch_probe_failed",
            job_id=context.job.job_id,
            payload={"probe_key": key_info.probe_key, "device_type": key_info.device_type, "reason": str(exc)},
        )
        logger.error(
            "[batch_probe] job=%s failed probe_key=%s device=%s reason=%s",
            context.job.job_id,
            key_info.probe_key,
            key_info.device_type,
            str(exc).replace("\n", " "),
        )
        raise

    context.store.upsert_batch_probe_profile(profile)
    context.job = _persist_resolved_batch_size(
        context,
        probe_key=key_info.probe_key,
        device_type=key_info.device_type,
        batch_param_name=batch_param_name,
        resolved_batch_size=profile.resolved_batch_size,
        source="probe",
    )
    context.event_logger.emit(
        "batch_probe_selected",
        job_id=context.job.job_id,
        payload={
            "probe_key": key_info.probe_key,
            "device_type": key_info.device_type,
            "resolved_batch_size": profile.resolved_batch_size,
            "target_budget_mb": profile.target_budget_mb,
            "warning_reason": (profile.metadata or {}).get("warning_reason"),
        },
    )
    logger.info(
        "[batch_probe] job=%s selected batch_size=%s probe_key=%s target_budget_mb=%s source=probe",
        context.job.job_id,
        profile.resolved_batch_size,
        key_info.probe_key,
        profile.target_budget_mb,
    )
    warning_message = str((profile.metadata or {}).get("warning_message") or "").strip()
    if warning_message:
        context.event_logger.emit(
            "batch_probe_warning",
            job_id=context.job.job_id,
            payload={
                "probe_key": key_info.probe_key,
                "device_type": key_info.device_type,
                "resolved_batch_size": profile.resolved_batch_size,
                "warning_reason": (profile.metadata or {}).get("warning_reason"),
                "warning_message": warning_message,
                "source": "probe",
            },
        )
        logger.warning(
            "[batch_probe] job=%s warning probe_key=%s source=probe message=%s",
            context.job.job_id,
            key_info.probe_key,
            warning_message,
        )
    refreshed = context.store.get_job(context.job.job_id)
    if refreshed is not None:
        context.job = refreshed
    return context.job
