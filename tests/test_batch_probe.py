from __future__ import annotations

from pathlib import Path

import pytest

from localml_scheduler.checkpointing.manager import CheckpointManager
from localml_scheduler.config import SCHEDULER_MODE_ADAPTIVE, SchedulerSettings
from localml_scheduler.domain import BatchProbeSpec, BatchProbeTrialResult, ResourceRequirements, TrainingJob
from localml_scheduler.execution.control import ControlPlane, TrainingControlHook
from localml_scheduler.execution.runner_protocol import RunnerContext
from localml_scheduler.observability.events import EventLogger
from localml_scheduler.profiling.batch_probe import BatchProbeExhausted, run_batch_probe_preflight
from localml_scheduler.storage import StateStore


def curve_probe(context: RunnerContext, batch_size: int, warmup_steps: int, measure_steps: int) -> BatchProbeTrialResult:
    context.event_logger.emit(
        "test_probe_call",
        job_id=context.job.job_id,
        payload={"batch_size": batch_size, "warmup_steps": warmup_steps, "measure_steps": measure_steps},
    )
    threshold = int(context.job.metadata.get("probe_threshold", 8))
    if batch_size > threshold:
        return BatchProbeTrialResult(
            fits=False,
            peak_vram_mb=None,
            memory_total_mb=24_576,
            message="CUDA out of memory",
            failure_kind="oom",
            returncode=1,
        )
    step_ms = 5.0 + batch_size
    return BatchProbeTrialResult(
        fits=True,
        peak_vram_mb=500 + 100 * batch_size,
        memory_total_mb=24_576,
        avg_step_time_ms=step_ms,
        samples_per_second=(batch_size * 1000.0) / step_ms,
        step_time_dispersion=0.05,
        probe_completed=True,
    )


def incomplete_probe(context: RunnerContext, batch_size: int, warmup_steps: int, measure_steps: int) -> BatchProbeTrialResult:
    del context, batch_size, warmup_steps, measure_steps
    return BatchProbeTrialResult(fits=True, peak_vram_mb=None, avg_step_time_ms=None, message="missing telemetry")


def _settings(tmp_path: Path) -> SchedulerSettings:
    return SchedulerSettings(
        runtime_root=tmp_path / "runtime",
        gpu_scheduler={
            "mode": SCHEDULER_MODE_ADAPTIVE,
            "batch_probe_min_batch_size": 1,
            "batch_probe_max_batch_size": 4096,
            "batch_probe_max_search_rounds": 14,
            "profiling": {"warmup_steps": 2, "solo_probe_steps": 5},
        },
    )


def _job(
    job_id: str,
    *,
    batch_size: int = 16,
    threshold: int = 8,
    namespace: str = "branch:shared",
    probe_target: str = "localml_scheduler.tests.test_batch_probe:curve_probe",
) -> TrainingJob:
    return TrainingJob.create(
        "localml_scheduler.profiling.batch_probe:run_branch_profile_probe_job",
        "baseline-a",
        "/tmp/a.py",
        job_id=job_id,
        task_type="mlevolve_branch_profile_probe",
        runner_kwargs={"batch_size": batch_size, "probe_max_batch_size": 32},
        batch_probe=BatchProbeSpec(
            enabled=True,
            probe_target=probe_target,
            model_key="model-a",
            profile_namespace=namespace,
            shape_signature_override="shape-a",
            minimum_batch_size=1,
            contract_version=3,
        ),
        metadata={"placement_backend": "exclusive", "probe_threshold": threshold, "exclusive_probe": True},
        resource_requirements=ResourceRequirements(requires_gpu=True),
    )


def _context(settings: SchedulerSettings, job: TrainingJob, *, store: StateStore | None = None) -> RunnerContext:
    store = store or StateStore(settings)
    store.save_job(job)
    event_logger = EventLogger(store, settings.events_jsonl_path)
    checkpoint_manager = CheckpointManager(settings, store, event_logger)
    control_plane = ControlPlane(settings)
    control_plane.initialize_job(job.job_id)
    return RunnerContext(
        job=job,
        settings=settings,
        store=store,
        event_logger=event_logger,
        control_hook=TrainingControlHook(job, control_plane, checkpoint_manager, store, event_logger),
        checkpoint_manager=checkpoint_manager,
        cache_client=None,
    )


def test_probe_starts_at_authored_downshifts_then_collects_complete_curve(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    job = _job("curve", batch_size=16, threshold=8)
    context = _context(settings, job)
    resolved = run_batch_probe_preflight(context)

    calls = [
        (
            event["payload"]["batch_size"],
            event["payload"]["warmup_steps"],
            event["payload"]["measure_steps"],
        )
        for event in context.store.list_events(job_id=job.job_id, event_type="test_probe_call")
    ]
    assert calls[0] == (16, 2, 5)
    assert all((warmup, measured) == (2, 5) for _, warmup, measured in calls)
    assert resolved.authored_batch_size == 16
    assert resolved.current_batch_size == 8
    assert resolved.config.runner_kwargs["batch_size"] == 16

    curves = context.store.list_batch_profile_curves()
    assert len(curves) == 1
    curve = curves[0]
    assert curve.contract_version == 3
    assert curve.maximum_feasible_batch_size == 8
    assert curve.first_oom_batch_size == 16
    assert curve.right_censored is False
    assert [point.batch_size for point in curve.points] == [1, 2, 4, 8]
    assert all(point.observations == 1 and point.samples_per_second > 0 for point in curve.points)
    assert not any(point.batch_size == 16 for point in curve.points)


def test_curve_is_right_censored_at_configured_cap_without_oom_point(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    job = _job("capped", batch_size=4, threshold=64)
    context = _context(settings, job)
    run_batch_probe_preflight(context)
    curve = context.store.list_batch_profile_curves()[0]
    assert [point.batch_size for point in curve.points] == [1, 2, 4, 8, 16, 32]
    assert curve.maximum_feasible_batch_size == 32
    assert curve.first_oom_batch_size is None
    assert curve.right_censored is True


def test_cache_reuse_keeps_each_jobs_authored_batch_immutable(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    first = _job("first", batch_size=16, threshold=8)
    first_context = _context(settings, first)
    first_result = run_batch_probe_preflight(first_context)
    second = _job("second", batch_size=16, threshold=8)
    second_context = _context(settings, second, store=first_context.store)
    second_result = run_batch_probe_preflight(second_context)

    assert first_result.current_batch_size == second_result.current_batch_size == 8
    assert first_result.authored_batch_size == second_result.authored_batch_size == 16
    assert second_result.config.runner_kwargs["batch_size"] == 16
    assert second_context.store.list_events(job_id="second", event_type="test_probe_call") == []
    events = second_context.store.list_events(job_id="second", event_type="batch_probe_cache_hit")
    assert len(events) == 1


def test_no_successful_point_raises_and_stores_no_feasible_observation(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    job = _job("unavailable", batch_size=8, threshold=0)
    context = _context(settings, job)
    with pytest.raises(BatchProbeExhausted):
        run_batch_probe_preflight(context)
    curve = context.store.list_batch_profile_curves()[0]
    assert curve.points == []
    assert context.store.list_batch_size_observations() == []
    assert context.store.list_events(job_id=job.job_id, event_type="batch_probe_failed")


def test_fit_without_end_to_end_timing_is_not_stored_as_success(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    job = _job(
        "incomplete",
        batch_size=1,
        threshold=1,
        probe_target="localml_scheduler.tests.test_batch_probe:incomplete_probe",
    )
    context = _context(settings, job)
    with pytest.raises(BatchProbeExhausted):
        run_batch_probe_preflight(context)
    assert context.store.list_batch_profile_points(context.store.list_batch_profile_curves()[0].curve_key) == []
    assert context.store.list_batch_size_observations() == []
