from __future__ import annotations

from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
import random
import tempfile
import time

import pytest

from localml_scheduler.checkpointing.manager import CheckpointManager
from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import (
    BatchSizeObservation,
    BatchResolution,
    CombinationProfile,
    JobStatus,
    PackingSpec,
    PairProfile,
    PlacementDecision,
    ResourceRequirements,
    RuntimeProfile,
    SafePointType,
    SchedulingClass,
    TrainingJob,
    build_batch_size_observation_key,
    build_group_signature,
)
from localml_scheduler.scheduler.early_stopping import (
    EarlyStoppingState,
    EarlyStoppingWatchdog,
)
from localml_scheduler.execution.control import (
    ControlPlane,
    EarlyStopRequested,
    TrainingControlHook,
)
from localml_scheduler.execution.worker_runtime import mark_job_completed
from localml_scheduler.observability.events import EventLogger
from localml_scheduler.observability.outcomes import classify_job_outcome
from localml_scheduler.scheduler.placement_planner import PlacementPlanner
from localml_scheduler.scheduler.policies import PriorityFifoPolicy
from localml_scheduler.scheduler.resource_estimator import BatchOptionEstimate
from localml_scheduler.prediction import JobPredictionError
from localml_scheduler.scheduler.telemetry import (
    GpuTelemetrySample,
    MemoryAdmissionGate,
)
from localml_scheduler.scheduler.service import ActiveRun, SchedulerService
from localml_scheduler.scheduler.supervisor import WorkerSnapshot
from localml_scheduler.scheduler.trace_simulator import (
    TraceBackendChange,
    TraceBatchOption,
    TraceJob,
    TraceMemorySample,
    TraceProblem,
    _time_aware_choice,
    benchmark_fixture,
    compare_policies,
    feasible_packs,
    markdown_table,
    simulate_policy,
)
from localml_scheduler.storage.sqlite_store import SQLiteStateStore


def _settings(tmpdir: str, **gpu_overrides: object) -> SchedulerSettings:
    gpu = {
        "mode": "parallel_time_aware",
        "backend_priority": ["cuda_process", "exclusive"],
        "parallel_job_cap": None,
        "memory": {
            "gpu_vram_gib": 10,
            "predicted_budget_fraction": 0.85,
            "live_admission_stop_fraction": 0.90,
            "live_admission_resume_fraction": 0.85,
            "admission_average_window_seconds": 10,
        },
        **gpu_overrides,
    }
    return SchedulerSettings(
        runtime_root=Path(tmpdir),
        gpu_scheduler=gpu,
        graph_db={"enabled": False},
        hardware_feature_db={"enabled": False},
    )


def _job(job_id: str, signature: str, *, priority: int = 0, submitted_at: str | None = None) -> TrainingJob:
    job = TrainingJob.create(
        "pkg.runner:train",
        signature,
        f"/tmp/{signature}.pt",
        job_id=job_id,
        priority=priority,
        runner_kwargs={"batch_size": 4},
        max_epochs=1,
        resource_requirements=ResourceRequirements(estimated_avg_vram_mb=512),
        packing=PackingSpec(
            eligible=True,
            signature=signature,
            backend_allowlist=["cuda_process"],
        ),
    )
    if submitted_at is not None:
        job.submitted_at = submitted_at
    return job


def _seed_options(store: SQLiteStateStore, planner: PlacementPlanner, job: TrainingJob) -> None:
    for backend in ("exclusive", "cuda_process"):
        for batch_size in (1, 2, 4, 8, 16):
            store.upsert_batch_size_observation(
                BatchSizeObservation(
                    observation_key=build_batch_size_observation_key(
                        job.baseline_model_id,
                        planner.estimator.shape_signature(job),
                        store.hardware_key(),
                        backend,
                        batch_size,
                    ),
                    model_key=job.baseline_model_id,
                    shape_signature=planner.estimator.shape_signature(job),
                    hardware_key=store.hardware_key(),
                    backend_name=backend,
                    batch_param_name="batch_size",
                    batch_size=batch_size,
                    avg_vram_mb=400 + batch_size,
                )
            )
            store.upsert_runtime_profile(
                RuntimeProfile.create(
                    signature=job.packing.signature or job.job_id,
                    hardware_key=store.hardware_key(),
                    backend_name=backend,
                    resolved_batch_size=batch_size,
                    strategy="epoch_1",
                    epoch_1_seconds=10.0,
                    estimated_total_runtime_seconds=10.0,
                    confidence=0.9,
                    observations=1,
                    source="branch_profile",
                )
            )


def test_five_batch_options_use_immutable_requested_batch_and_clip() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
        job = _job("job", "sig")
        assert planner._candidate_batch_sizes(job) == [1, 2, 4, 8, 16]
        changed = planner.estimator.resolved_batch_size(job.copy(metadata={"resolved_batch_size": 16}))
        assert changed == 16
        assert planner._candidate_batch_sizes(job.copy(metadata={"resolved_batch_size": 16})) == [1, 2, 4, 8, 16]
        capped = job.copy()
        capped.config.runner_kwargs["probe_max_batch_size"] = 6
        assert planner._candidate_batch_sizes(capped) == [1, 2, 4, 6]


def test_time_aware_configuration_validates_and_migrates_legacy_cap() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        migrated = SchedulerSettings(
            runtime_root=tmpdir,
            gpu_scheduler={"mode": "parallel_time_aware", "max_packed_jobs_per_gpu": 3},
        )
        assert migrated.gpu_scheduler.parallel_job_cap == 3
        restored = SchedulerSettings(
            runtime_root=tmpdir,
            gpu_scheduler=migrated.gpu_scheduler.to_dict(),
            early_stopping={"mode": "min", "patience_epochs": 2},
        )
        assert restored.gpu_scheduler.objective.objective_version == "time_v3_flow_only"
        assert restored.early_stopping.mode == "min"
        with pytest.raises(ValueError, match="no longer supports throughput scheduling controls"):
            SchedulerSettings(
                runtime_root=tmpdir,
                gpu_scheduler={
                    "mode": "parallel_time_aware",
                    "objective": {"makespan_weight": 0.8, "flow_time_weight": 0.4},
                },
            )
        with pytest.raises(ValueError, match="no longer supports throughput scheduling control"):
            SchedulerSettings(
                runtime_root=tmpdir,
                gpu_scheduler={
                    "mode": "parallel_time_aware",
                    "thresholds": {"min_aggregate_gain": 1.1},
                },
            )


def test_time_score_prefers_faster_pack_and_ignores_sm_utilization() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
        left = _job("left", "left-sig")
        right = _job("right", "right-sig")
        left.queue_sequence = 1
        right.queue_sequence = 2
        _seed_options(store, planner, left)
        _seed_options(store, planner, right)
        store.upsert_pair_profile(
            PairProfile.create(
                "left-sig",
                "right-sig",
                backend_name="cuda_process",
                hardware_key=store.hardware_key(),
                compatible=True,
                slowdown_ratio=9.0,
                avg_gpu_utilization=1.0,
                observations=1,
            )
        )
        plan = planner.choose_plan(
            [left, right],
            backend_available={"cuda_process": True, "exclusive": True},
        )
        assert plan is not None
        assert plan.mode == "packed_pair"
        assert plan.objective_version == "time_v3_flow_only"
        assert plan.objective_breakdown["score"] == pytest.approx(
            plan.objective_breakdown["normalized_flow_cost"]
        )
        assert plan.objective_breakdown["slowdown_prediction"] == "disabled"
        assert "member_slowdowns" not in plan.objective_breakdown
        assert "aggregate_gain" not in plan.objective_breakdown
        assert "makespan_proxy" not in plan.objective_breakdown
        assert "candidate_estimates" in plan.objective_breakdown


def test_starved_oldest_job_is_mandatory_exclusive_fallback() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir, starvation_timeout_seconds=60)
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        old = _job("old", "old-sig", submitted_at=(now - timedelta(seconds=61)).isoformat())
        young = _job("young", "young-sig", priority=100, submitted_at=now.isoformat())
        for job in (old, young):
            _seed_options(store, planner, job)
        plan = planner.choose_plan(
            [young, old],
            backend_available={"cuda_process": False, "exclusive": True},
            now=now,
        )
        assert plan is not None
        assert plan.job_ids == ("old",)
        assert plan.mandatory_anchor_job_id == "old"


def test_exclusive_probe_is_first_class_and_waits_for_drain() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
        probe = _job("probe", "probe-sig", priority=10)
        probe.scheduling_class = SchedulingClass.EXCLUSIVE_PROBE
        active = _job("active", "active-sig")
        assert (
            planner.choose_plan(
                [probe],
                backend_available={"exclusive": True},
                active_jobs=[active],
                exclusive_drain_requested=True,
            )
            is None
        )
        plan = planner.choose_plan([probe], backend_available={"exclusive": True})
        assert plan is not None
        assert plan.job_ids == ("probe",)
        assert plan.backend_name == "exclusive"


def _sample(second: int, fraction: float) -> GpuTelemetrySample:
    captured = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=second)
    return GpuTelemetrySample(
        captured_at=captured.isoformat(),
        memory_used_mb=int(fraction * 1000),
        memory_total_mb=1000,
    )


def test_memory_gate_uses_full_window_hysteresis_without_pausing() -> None:
    gate = MemoryAdmissionGate(window_seconds=10, stop_fraction=0.90, resume_fraction=0.85)
    assert gate.update(_sample(0, 0.99)) is None
    assert gate.is_open
    for second in range(1, 11):
        transition = gate.update(_sample(second, 0.91))
    assert transition == "closed"
    assert not gate.is_open
    for second in range(11, 26):
        assert gate.update(_sample(second, 0.80)) is None
    assert gate.update(_sample(26, 0.80)) == "opened"
    assert gate.is_open


def test_early_stopping_state_machine_supports_max_min_delta_and_restart() -> None:
    settings = SchedulerSettings(early_stopping={"enabled": True, "patience_epochs": 2, "min_delta": 0.01}).early_stopping
    watchdog = EarlyStoppingWatchdog(settings)
    state = EarlyStoppingState()
    first = watchdog.evaluate(epoch=1, metrics={"accuracy": 0.5}, state=state)
    assert first.improved
    restored = EarlyStoppingState.from_dict(first.state.to_dict())
    second = watchdog.evaluate(epoch=2, metrics={"accuracy": 0.505}, state=restored)
    assert not second.improved and not second.should_stop
    third = watchdog.evaluate(epoch=3, metrics={"accuracy": 0.50}, state=second.state)
    assert third.should_stop
    duplicate = watchdog.evaluate(epoch=3, metrics={"accuracy": 1.0}, state=third.state)
    assert not duplicate.evaluated


def test_non_power_of_two_requested_batch_falls_back_exclusive() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
        job = _job("job", "sig")
        job.requested_batch_size = 3
        plan = planner.choose_plan([job], backend_available={"exclusive": True})
        assert plan is not None
        assert plan.mode == "exclusive"
        assert "unavailable" in plan.reason


def test_active_plus_new_jobs_respect_cap_memory_and_determinism() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir, parallel_job_cap=2)
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
        active = _job("active", "active-sig")
        waiting = _job("waiting", "waiting-sig")
        for job in (active, waiting):
            _seed_options(store, planner, job)
        store.upsert_pair_profile(
            PairProfile.create(
                "active-sig",
                "waiting-sig",
                backend_name="cuda_process",
                hardware_key=store.hardware_key(),
                slowdown_ratio=1.1,
                observations=1,
            )
        )
        kwargs = {
            "backend_available": {"cuda_process": True, "exclusive": True},
            "active_jobs": [active],
            "active_vram_mb": 8_000.0,
        }
        first = planner.choose_plan([waiting], **kwargs)
        second = planner.choose_plan([waiting], **kwargs)
        assert first == second
        assert first is not None
        assert first.mode == "concurrent_group"
        assert len(first.job_ids) + 1 == 2
        assert set(first.objective_breakdown["completion_offsets_seconds"]) == {"active", "waiting"}
        assert "active" in first.objective_breakdown["active_fixed_estimates"]
        assert planner.choose_plan([waiting], **{**kwargs, "active_vram_mb": 8_500.0}) is None


def test_stale_cached_fill_objective_cannot_influence_time_score() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
        left = _job("left", "left-sig")
        right = _job("right", "right-sig")
        for job in (left, right):
            _seed_options(store, planner, job)
        store.upsert_pair_profile(
            PairProfile.create(
                "left-sig",
                "right-sig",
                backend_name="cuda_process",
                hardware_key=store.hardware_key(),
                slowdown_ratio=1.1,
                observations=1,
            )
        )
        store.upsert_combination_profile(
            CombinationProfile.create(
                build_group_signature(["left-sig", "right-sig"]),
                store.hardware_key(),
                "cuda_process",
                "parallel_time_aware",
                {"left": 16, "right": 16},
                objective_score=-9999.0,
                resolved_optimal=True,
                metadata={"objective_version": "vram_fill_v1"},
            )
        )
        plan = planner.choose_plan(
            [left, right],
            backend_available={"cuda_process": True, "exclusive": True},
        )
        assert plan is not None
        assert plan.objective_version == "time_v3_flow_only"
        assert plan.batch_overrides == {"left": 1, "right": 1}


def test_batch_optimized_ignores_slowdown_and_legacy_cached_objective() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir, mode="parallel_batch_optimized")
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
        left = _job("optimized-left", "optimized-left-sig")
        right = _job("optimized-right", "optimized-right-sig")
        for job in (left, right):
            _seed_options(store, planner, job)
        store.upsert_pair_profile(
            PairProfile.create(
                "optimized-left-sig",
                "optimized-right-sig",
                backend_name="cuda_process",
                hardware_key=store.hardware_key(),
                slowdown_ratio=9.0,
                observations=1,
            )
        )
        store.upsert_combination_profile(
            CombinationProfile.create(
                build_group_signature(["optimized-left-sig", "optimized-right-sig"]),
                store.hardware_key(),
                "cuda_process",
                "parallel_batch_optimized",
                {"optimized-left": 1, "optimized-right": 1},
                avg_vram_mb=802.0,
                objective_score=9999.0,
                resolved_optimal=True,
            )
        )
        plan = planner.choose_plan(
            [left, right],
            backend_available={"cuda_process": True, "exclusive": True},
        )
        assert plan is not None
        assert plan.mode == "packed_pair"
        expected_batch = max(planner._candidate_batch_sizes(left))
        assert plan.batch_overrides == {"optimized-left": expected_batch, "optimized-right": expected_batch}


def test_early_stop_hook_persists_state_and_completes_successfully() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = SchedulerSettings(
            runtime_root=Path(tmpdir),
            early_stopping={
                "enabled": True,
                "metric_name": "accuracy",
                "patience_epochs": 2,
                "save_best_checkpoint": True,
            },
            graph_db={"enabled": False},
            hardware_feature_db={"enabled": False},
        )
        settings.ensure_runtime_layout()
        store = SQLiteStateStore(settings)
        job = store.submit_job(
            TrainingJob.create(
                "pkg.runner:train",
                "model",
                "/tmp/model.pt",
                job_id="early-stop",
                max_epochs=10,
            )
        )
        events = EventLogger(store, settings.events_jsonl_path)
        control = ControlPlane(settings)
        control.initialize_job(job.job_id)
        hook = TrainingControlHook(job, control, CheckpointManager(settings, store, events), store, events)
        hook.safe_point(
            SafePointType.EPOCH,
            epoch=1,
            global_step=1,
            metrics={"accuracy": 0.7},
            state_factory=lambda: {"epoch": 1},
        )
        hook.safe_point(
            SafePointType.EPOCH,
            epoch=2,
            global_step=2,
            metrics={"accuracy": 0.7},
            state_factory=lambda: {"epoch": 2},
        )
        try:
            hook.safe_point(
                SafePointType.EPOCH,
                epoch=3,
                global_step=3,
                metrics={"accuracy": 0.69},
                state_factory=lambda: {"epoch": 3},
                remaining_runtime_seconds=42.0,
            )
        except EarlyStopRequested as exc:
            assert mark_job_completed(settings, store, events, job.job_id, exc.result) == 0
        else:
            raise AssertionError("expected early stop")
        completed = store.get_job(job.job_id)
        assert completed is not None
        assert completed.status.value == "COMPLETED"
        assert completed.status_reason == "early_stopped_no_improvement"
        assert completed.metadata["early_stopping_result"]["epochs_saved"] == 7
        assert completed.metadata["early_stopping_result"]["estimated_wall_time_saved_seconds"] == 42.0
        assert completed.metadata["early_stopping_best_checkpoint_path"]
        report = store.report()
        assert report.early_stopped_epochs_saved == 7
        assert report.early_stopped_wall_time_saved_seconds == 42.0


def test_trace_replay_compares_serial_fill_time_aware_and_oracle() -> None:
    results = {item.policy: item for item in compare_policies(benchmark_fixture())}
    assert set(results) == {
        "serial_fifo",
        "legacy_vram_fill",
        "parallel_time_aware",
        "small_trace_oracle",
    }
    assert results["parallel_time_aware"].makespan_seconds < results["legacy_vram_fill"].makespan_seconds
    assert results["parallel_time_aware"].mean_flow_seconds < results["serial_fifo"].mean_flow_seconds
    assert results["parallel_time_aware"].starvation_count == 0
    report = markdown_table(results.values())
    assert "Total flow (s)" in report
    assert "Median flow (s)" in report
    assert "Actual over-budget packs" in report


class _DrainSupervisor:
    def __init__(self, active_ids: list[str]) -> None:
        self.active_ids = active_ids
        self.dispatched: list[str] = []

    def active_job_ids(self) -> list[str]:
        return list(self.active_ids)

    def active_job_ids_by_group(self) -> dict[str, list[str]]:
        return {"active-group": list(self.active_ids)} if self.active_ids else {}

    def available_backends(self) -> dict[str, bool]:
        return {"exclusive": True, "cuda_process": True}

    def dispatch(self, jobs: list[TrainingJob], **kwargs: object) -> PlacementDecision:
        self.dispatched.extend(job.job_id for job in jobs)
        return PlacementDecision(
            can_run=True,
            reason="test",
            mode=str(kwargs.get("mode")),
            backend_name=str(kwargs.get("backend_name")),
            job_ids=[job.job_id for job in jobs],
            group_id="probe-group",
        )


def test_service_probe_reservation_blocks_normal_admission_until_drain() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        store = SQLiteStateStore(settings)
        active = store.submit_job(_job("active", "active-sig"))
        store.set_job_status(active.job_id, JobStatus.RUNNING, reason="test", hold=False)
        normal = store.submit_job(_job("normal", "normal-sig"))
        store.set_job_status(normal.job_id, JobStatus.READY, reason="test", hold=False)
        probe = _job("probe", "probe-sig", priority=100)
        probe.scheduling_class = SchedulingClass.EXCLUSIVE_PROBE
        probe = store.submit_job(probe)
        store.set_job_status(probe.job_id, JobStatus.READY, reason="test", hold=False)
        supervisor = _DrainSupervisor([active.job_id])
        service = SchedulerService(settings, store=store, supervisor=supervisor)
        service._active_runs["active-group"] = ActiveRun(
            group_id="active-group",
            mode="packed_pair",
            backend_name="cuda_process",
            job_ids=(active.job_id,),
        )
        service._dispatch_pending_work()
        assert service._exclusive_probe_job_id == probe.job_id
        assert supervisor.dispatched == []

        supervisor.active_ids = []
        service._active_runs.clear()
        service._dispatch_pending_work()
        assert supervisor.dispatched == [probe.job_id]
        assert normal.job_id not in supervisor.dispatched


def test_service_restores_admission_and_probe_reservation_state() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        store = SQLiteStateStore(settings)
        first = SchedulerService(settings, store=store, supervisor=_DrainSupervisor([]))
        first._admission_gate.is_open = False
        first._admission_gate.average_fraction = 0.93
        first._exclusive_probe_job_id = "reserved-probe"
        first._persist_scheduler_decision_state()

        restored = SchedulerService(settings, store=store, supervisor=_DrainSupervisor([]))
        assert not restored._admission_gate.is_open
        assert restored._admission_gate.average_fraction == 0.93
        assert restored._exclusive_probe_job_id == "reserved-probe"

        disabled_settings = _settings(
            tmpdir,
            exclusive_probe={"enabled": False, "drain_without_preemption": True},
        )
        disabled = SchedulerService(disabled_settings, store=store, supervisor=_DrainSupervisor([]))
        assert disabled._exclusive_probe_job_id is None


def test_closed_admission_blocks_only_new_work_and_keeps_active_job_running() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir, parallel_job_cap=2)
        store = SQLiteStateStore(settings)
        active = store.submit_job(_job("admission-active", "admission-active-sig"))
        waiting = store.submit_job(_job("admission-waiting", "admission-waiting-sig"))
        store.set_job_status(active.job_id, JobStatus.RUNNING, reason="test", hold=False)
        store.set_job_status(waiting.job_id, JobStatus.READY, reason="test", hold=False)
        supervisor = _DrainSupervisor([active.job_id])
        service = SchedulerService(settings, store=store, supervisor=supervisor)
        service._active_runs["active-group"] = ActiveRun(
            group_id="active-group",
            mode="packed_pair",
            backend_name="cuda_process",
            job_ids=(active.job_id,),
        )
        service._admission_gate.is_open = False
        service._dispatch_pending_work()
        assert supervisor.dispatched == []
        assert store.get_job(active.job_id).status == JobStatus.RUNNING
        assert store.get_job(waiting.job_id).status == JobStatus.READY


def test_planner_slowdown_is_not_persisted_without_same_batch_solo_profile() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        store = SQLiteStateStore(settings)
        left = store.submit_job(_job("evidence-left", "evidence-left-sig"))
        right = store.submit_job(_job("evidence-right", "evidence-right-sig"))
        for job in (left, right):
            store.set_job_status(job.job_id, JobStatus.COMPLETED, reason="test", hold=False)
        run = ActiveRun(
            group_id="evidence-group",
            mode="packed_pair",
            backend_name="cuda_process",
            job_ids=(left.job_id, right.job_id),
            batch_overrides={left.job_id: 4, right.job_id: 4},
            hardware_key=store.hardware_key(),
            group_signature=build_group_signature(["evidence-left-sig", "evidence-right-sig"]),
            objective_breakdown={
                "score": 0.5,
                "member_slowdowns": {left.job_id: 1.2, right.job_id: 1.2},
            },
            objective_version="time_v3_flow_only",
        )
        service = SchedulerService(settings, store=store, supervisor=_DrainSupervisor([]))
        service._record_combination_profiles(run)
        profile = store.get_pair_profile(
            "evidence-left-sig",
            "evidence-right-sig",
            backend_name="cuda_process",
        )
        assert profile is not None
        assert profile.slowdown_ratio is None
        assert profile.metadata["per_member_slowdown"] == {}
        assert profile.metadata["slowdown_sources"] == {}


def test_existing_measured_slowdown_survives_run_without_solo_baseline() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        store = SQLiteStateStore(settings)
        left = store.submit_job(_job("preserve-left", "preserve-left-sig"))
        right = store.submit_job(_job("preserve-right", "preserve-right-sig"))
        for job in (left, right):
            store.set_job_status(job.job_id, JobStatus.COMPLETED, reason="test", hold=False)
        store.upsert_pair_profile(
            PairProfile.create(
                "preserve-left-sig",
                "preserve-right-sig",
                backend_name="cuda_process",
                hardware_key=store.hardware_key(),
                slowdown_ratio=1.25,
                observations=1,
                metadata={
                    "batch_vector": {left.job_id: 4, right.job_id: 4},
                    "per_member_slowdown": {left.job_id: 1.25, right.job_id: 1.2},
                    "per_signature_slowdown": {"preserve-left-sig": 1.25, "preserve-right-sig": 1.2},
                    "slowdown_sources": {
                        left.job_id: "measured_against_exclusive_profile",
                        right.job_id: "measured_against_exclusive_profile",
                    },
                },
            )
        )
        run = ActiveRun(
            group_id="preserve-group",
            mode="packed_pair",
            backend_name="cuda_process",
            job_ids=(left.job_id, right.job_id),
            batch_overrides={left.job_id: 8, right.job_id: 8},
            hardware_key=store.hardware_key(),
        )
        SchedulerService(settings, store=store, supervisor=_DrainSupervisor([]))._record_combination_profiles(run)
        profile = store.get_pair_profile("preserve-left-sig", "preserve-right-sig", backend_name="cuda_process")
        assert profile is not None
        assert profile.slowdown_ratio == pytest.approx(1.25)
        assert profile.metadata["batch_vector"] == {left.job_id: 4, right.job_id: 4}
        assert set(profile.metadata["slowdown_sources"].values()) == {"measured_against_exclusive_profile"}


def test_same_batch_exclusive_profiles_support_passive_slowdown_measurement() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        store = SQLiteStateStore(settings)
        left = store.submit_job(_job("measured-left", "measured-left-sig"))
        right = store.submit_job(_job("measured-right", "measured-right-sig"))
        for job in (left, right):
            store.set_job_status(job.job_id, JobStatus.COMPLETED, reason="test", hold=False)
            store.upsert_runtime_profile(
                RuntimeProfile.create(
                    signature=job.packing.signature or job.job_id,
                    hardware_key=store.hardware_key(),
                    backend_name="exclusive",
                    resolved_batch_size=4,
                    strategy="epoch_1",
                    estimated_total_runtime_seconds=10.0,
                    observations=1,
                    source="branch_profile",
                )
            )
        run = ActiveRun(
            group_id="measured-group",
            mode="packed_pair",
            backend_name="cuda_process",
            job_ids=(left.job_id, right.job_id),
            batch_overrides={left.job_id: 4, right.job_id: 4},
            hardware_key=store.hardware_key(),
            opened_at=(datetime.now(timezone.utc) - timedelta(seconds=20)).isoformat(),
        )
        SchedulerService(settings, store=store, supervisor=_DrainSupervisor([]))._record_combination_profiles(run)
        profile = store.get_pair_profile("measured-left-sig", "measured-right-sig", backend_name="cuda_process")
        assert profile is not None
        assert profile.slowdown_ratio == pytest.approx(2.0, rel=0.05)
        assert set(profile.metadata["slowdown_sources"].values()) == {"measured_against_exclusive_profile"}


def test_batch_resolution_remains_immutable_across_dispatch_and_resume_round_trip() -> None:
    job = _job("round-trip", "round-trip-sig")
    assert job.requested_batch_size == 4
    dispatched = BatchResolution.apply(job, 16)
    dispatched.status = JobStatus.PAUSED
    restored = TrainingJob.from_dict(dispatched.to_dict())
    assert restored.requested_batch_size == 4
    assert BatchResolution.resolved_batch_size(restored) == 16
    assert restored.status == JobStatus.PAUSED


def test_batch_estimates_are_batch_specific_sourced_and_pareto_pruned() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
        job = _job("estimate", "estimate-sig")
        _seed_options(store, planner, job)
        options = planner.estimator.estimate_batch_options(job, "cuda_process", [1, 4, 16])
        assert [option.batch_size for option in options] == [1, 4, 16]
        assert [option.avg_vram_mb for option in options] == [401.0, 404.0, 416.0]
        assert all(option.source == "branch_profile" for option in options)

        synthetic = [
            BatchOptionEstimate("estimate", 1, 100.0, 10.0, 1, 10.0, "probe", 0.9, "v1"),
            BatchOptionEstimate("estimate", 2, 120.0, 12.0, 1, 12.0, "probe", 0.9, "v1"),
            BatchOptionEstimate("estimate", 4, 90.0, 15.0, 1, 15.0, "probe", 0.9, "v1"),
        ]
        assert [option.batch_size for option in planner.estimator.pareto_prune(synthetic)] == [1, 4]


class _FailingBatchPredictor:
    last_sources: dict[str, str] = {}
    last_errors: dict[str, str] = {}

    def predict_avg_vram_options(self, job: TrainingJob, batch_sizes: list[int]) -> dict[int, float]:
        raise JobPredictionError("test predictor failure")


class _SelectiveBatchPredictor:
    last_sources: dict[str, str] = {}
    last_errors: dict[str, str] = {}

    def predict_avg_vram_options(self, job: TrainingJob, batch_sizes: list[int]) -> dict[int, float]:
        if job.job_id == "predictor-failure":
            raise JobPredictionError("one job is unsupported")
        return {batch_size: 700.0 + batch_size for batch_size in batch_sizes}


def test_predictor_failure_falls_back_per_job_to_branch_profiles() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
        job = _job("fallback", "fallback-sig")
        _seed_options(store, planner, job)
        planner.estimator.ml_predictor = _FailingBatchPredictor()  # type: ignore[assignment]
        options = planner.estimator.estimate_batch_options(job, "cuda_process", [1, 2, 4, 8, 16])
        assert len(options) == 5
        assert all(option.source == "branch_profile" for option in options)


def test_predictor_failure_for_one_job_does_not_poison_other_jobs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
        failing = _job("predictor-failure", "predictor-failure-sig")
        healthy = _job("predictor-healthy", "predictor-healthy-sig")
        for job in (failing, healthy):
            _seed_options(store, planner, job)
        planner.estimator.ml_predictor = _SelectiveBatchPredictor()  # type: ignore[assignment]
        failed_options = planner.estimator.estimate_batch_options(
            failing,
            "cuda_process",
            [1, 2, 4, 8, 16],
        )
        healthy_options = planner.estimator.estimate_batch_options(
            healthy,
            "cuda_process",
            [1, 2, 4, 8, 16],
        )
        assert all(option.source == "branch_profile" for option in failed_options)
        assert [option.avg_vram_mb for option in healthy_options] == [701.0, 702.0, 704.0, 708.0, 716.0]
        assert all(option.source == "ml_predictor+branch_profile" for option in healthy_options)


def test_memory_budget_accepts_equality_and_rejects_any_excess() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir, parallel_job_cap=2)
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
        active = _job("active-equal", "active-equal-sig")
        waiting = _job("waiting-equal", "waiting-equal-sig")
        for job in (active, waiting):
            _seed_options(store, planner, job)
        store.upsert_pair_profile(
            PairProfile.create(
                "active-equal-sig",
                "waiting-equal-sig",
                backend_name="cuda_process",
                hardware_key=store.hardware_key(),
                slowdown_ratio=1.0,
                observations=1,
            )
        )
        exact_active_memory = planner.estimator.safe_budget_mb() - 401.0
        kwargs = {
            "backend_available": {"cuda_process": True, "exclusive": True},
            "active_jobs": [active],
        }
        assert planner.choose_plan([waiting], active_vram_mb=exact_active_memory, **kwargs) is not None
        assert planner.choose_plan([waiting], active_vram_mb=exact_active_memory + 1e-6, **kwargs) is None


@pytest.mark.parametrize(
    ("compatible", "slowdown", "cooldown", "expect_pair"),
    [
        (False, 1.0, False, False),
        (True, 9.0, False, True),
        (True, 1.0, True, False),
        (True, 1.10, False, True),
    ],
)
def test_time_aware_pair_rejects_incompatibility_and_cooldown_but_ignores_slowdown(
    compatible: bool,
    slowdown: float,
    cooldown: bool,
    expect_pair: bool,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
        left = _job("guard-left", "guard-left-sig")
        right = _job("guard-right", "guard-right-sig")
        for job in (left, right):
            _seed_options(store, planner, job)
        store.upsert_pair_profile(
            PairProfile.create(
                "guard-left-sig",
                "guard-right-sig",
                backend_name="cuda_process",
                hardware_key=store.hardware_key(),
                compatible=compatible,
                slowdown_ratio=slowdown,
                cooldown_until=((datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat() if cooldown else None),
                observations=1,
            )
        )
        plan = planner.choose_plan(
            [left, right],
            backend_available={"cuda_process": True, "exclusive": True},
        )
        assert plan is not None
        assert (plan.mode == "packed_pair") is expect_pair


@pytest.mark.parametrize(("cap", "expected_size"), [(1, 1), (2, 2), (3, 3), (None, 3)])
def test_parallel_cap_values_one_two_three_and_unlimited(cap: int | None, expected_size: int) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir, parallel_job_cap=cap)
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
        jobs = [_job(f"cap-{index}", f"cap-{index}-sig") for index in range(3)]
        for index, job in enumerate(jobs, start=1):
            job.queue_sequence = index
            _seed_options(store, planner, job)
        for left, right in combinations(jobs, 2):
            store.upsert_pair_profile(
                PairProfile.create(
                    left.packing.signature or left.job_id,
                    right.packing.signature or right.job_id,
                    backend_name="cuda_process",
                    hardware_key=store.hardware_key(),
                    slowdown_ratio=1.0,
                    observations=1,
                )
            )
        plan = planner.choose_plan(jobs, backend_available={"cuda_process": True, "exclusive": True})
        assert plan is not None
        assert len(plan.job_ids) == expected_size


def test_early_stopping_min_mode_min_epochs_missing_and_nan() -> None:
    settings = SchedulerSettings(
        early_stopping={
            "enabled": True,
            "mode": "min",
            "patience_epochs": 2,
            "min_delta": 0.1,
            "min_epochs": 4,
        }
    ).early_stopping
    watchdog = EarlyStoppingWatchdog(settings)
    first = watchdog.evaluate(epoch=1, metrics={"accuracy": 1.0}, state=EarlyStoppingState())
    second = watchdog.evaluate(epoch=2, metrics={"accuracy": 0.95}, state=first.state)
    missing = watchdog.evaluate(epoch=3, metrics={}, state=second.state)
    assert missing.warning and missing.state.bad_epoch_count == 1
    nan_result = watchdog.evaluate(epoch=4, metrics={"accuracy": float("nan")}, state=missing.state)
    assert nan_result.warning and not nan_result.should_stop
    fourth = watchdog.evaluate(epoch=5, metrics={"accuracy": 0.94}, state=nan_result.state)
    assert fourth.should_stop
    improved = watchdog.evaluate(epoch=6, metrics={"accuracy": 0.80}, state=fourth.state)
    assert improved.improved and improved.state.bad_epoch_count == 0


def test_exclusive_probe_configuration_is_enforced() -> None:
    with pytest.raises(ValueError, match="non-preemptive"):
        SchedulerSettings(
            gpu_scheduler={
                "mode": "parallel_time_aware",
                "exclusive_probe": {"enabled": True, "drain_without_preemption": False},
            }
        )
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir, exclusive_probe={"enabled": False, "drain_without_preemption": True})
        store = SQLiteStateStore(settings)
        probe = _job("disabled-probe", "disabled-probe-sig")
        probe.scheduling_class = SchedulingClass.EXCLUSIVE_PROBE
        probe = store.submit_job(probe)
        store.set_job_status(probe.job_id, JobStatus.READY, reason="test", hold=False)
        supervisor = _DrainSupervisor([])
        service = SchedulerService(settings, store=store, supervisor=supervisor)
        service._dispatch_pending_work()
        assert service._exclusive_probe_job_id is None
        assert supervisor.dispatched == [probe.job_id]


def test_trace_contract_models_backends_compatibility_memory_and_early_stop() -> None:
    results = compare_policies(benchmark_fixture())
    assert all(item.hard_constraint_violations == 0 for item in results)
    assert all(item.predicted_avg_vram_mb > 0 and item.actual_avg_vram_mb > 0 for item in results)
    assert all(item.early_stopped_epochs_saved == 2 for item in results)
    assert all(item.early_stopped_wall_time_saved_seconds > 0 for item in results)

    options = tuple(TraceBatchOption(2**index, 500.0 + index, 5.0) for index in range(5))
    left = TraceJob("left", 0.0, 0, options)
    right = TraceJob("right", 0.0, 0, options)
    backend_problem = TraceProblem(
        jobs=(left, right),
        memory_budget_mb=5_000,
        parallel_cap=2,
        initial_backend_availability={"exclusive": True, "cuda_process": False},
        backend_changes=(TraceBackendChange(5.0, "cuda_process", True),),
        slowdown_by_pair={("left", "right"): 1.1},
    )
    assert all(len(pack.members) == 1 for pack in feasible_packs(backend_problem, (left, right), now=0.0))
    available_packs = feasible_packs(backend_problem, (left, right), now=5.0)
    assert any(len(pack.members) == 2 for pack in available_packs)
    packed = next(pack for pack in available_packs if len(pack.members) == 2)
    assert packed.predicted_completion_offsets == pytest.approx((5.0, 5.0))
    assert packed.completion_offsets == pytest.approx((5.5, 5.5))

    incompatible = TraceProblem(
        jobs=(left, right),
        memory_budget_mb=5_000,
        parallel_cap=2,
        compatibility_by_pair={("left", "right"): False},
    )
    assert all(len(pack.members) == 1 for pack in feasible_packs(incompatible, (left, right), now=0.0))

    gated = TraceProblem(
        jobs=(left, right),
        memory_budget_mb=5_000,
        parallel_cap=2,
        slowdown_by_pair={("left", "right"): 1.1},
        live_memory_samples=tuple(TraceMemorySample(float(second), 0.91) for second in range(11)),
    )
    assert all(len(pack.members) == 1 for pack in feasible_packs(gated, (left, right), now=10.0))


def test_seeded_randomized_trace_invariants_and_determinism() -> None:
    for seed in range(20):
        rng = random.Random(seed)
        jobs: list[TraceJob] = []
        for index in range(6):
            options = tuple(
                TraceBatchOption(
                    2**exponent,
                    memory_mb=300.0 + 100.0 * exponent + rng.random(),
                    solo_seconds=5.0 + rng.random() * 8.0,
                    actual_memory_mb=320.0 + 100.0 * exponent + rng.random(),
                )
                for exponent in range(5)
            )
            jobs.append(TraceJob(f"job-{index}", float(rng.randrange(0, 5)), rng.randrange(0, 4), options))
        compatibility: dict[tuple[str, str], bool] = {}
        slowdowns: dict[tuple[str, str], float] = {}
        for left, right in combinations(jobs, 2):
            ordered_key = sorted((left.job_id, right.job_id))
            key = (ordered_key[0], ordered_key[1])
            compatibility[key] = rng.random() > 0.25
            slowdowns[key] = 1.0 + rng.random() * 0.15
        problem = TraceProblem(
            jobs=tuple(jobs),
            memory_budget_mb=4_500,
            parallel_cap=rng.choice([1, 2, 3, None]),
            compatibility_by_pair=compatibility,
            slowdown_by_pair=slowdowns,
            starvation_timeout_seconds=20,
        )
        first = simulate_policy(problem, "parallel_time_aware", _time_aware_choice)
        second = simulate_policy(problem, "parallel_time_aware", _time_aware_choice)
        assert first == second
        assert first.hard_constraint_violations == 0


def test_planning_latency_is_bounded_at_maximum_default_window() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(
            tmpdir,
            parallel_job_cap=4,
            priority_window_size=8,
            oldest_window_size=4,
            beam_width=32,
            exact_search_max_jobs=3,
        )
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
        jobs = [_job(f"latency-{index}", f"latency-{index}-sig") for index in range(12)]
        for index, job in enumerate(jobs, start=1):
            job.queue_sequence = index
            _seed_options(store, planner, job)
        for left, right in combinations(jobs, 2):
            store.upsert_pair_profile(
                PairProfile.create(
                    left.packing.signature or left.job_id,
                    right.packing.signature or right.job_id,
                    backend_name="cuda_process",
                    hardware_key=store.hardware_key(),
                    slowdown_ratio=1.02,
                    observations=1,
                )
            )
        started = time.perf_counter()
        plan = planner.choose_plan(jobs, backend_available={"cuda_process": True, "exclusive": True})
        elapsed = time.perf_counter() - started
        assert plan is not None
        assert elapsed < 5.0


def test_real_worker_validation_runner_early_stops_successfully() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = SchedulerSettings(
            runtime_root=Path(tmpdir),
            scheduler_poll_interval_seconds=0.05,
            gpu_scheduler={"mode": "serial_basic", "backend_priority": ["exclusive"]},
            early_stopping={
                "enabled": True,
                "metric_name": "accuracy",
                "mode": "max",
                "patience_epochs": 2,
                "min_delta": 0.01,
                "min_epochs": 2,
                "save_best_checkpoint": True,
            },
            graph_db={"enabled": False},
            hardware_feature_db={"enabled": False},
        )
        api = SchedulerClient(settings)
        service = api.create_service().start(background=True)
        try:
            job = api.submit(
                TrainingJob.create(
                    "localml_scheduler.tests.fixtures.early_stop_runner:run_validation_sequence",
                    "early-real-baseline",
                    str(Path(tmpdir) / "unused-baseline.pt"),
                    job_id="early-real",
                    runner_kwargs={"validation_accuracy": [0.50, 0.70, 0.705, 0.69, 0.95]},
                    max_epochs=5,
                )
            )
            deadline = time.time() + 20.0
            while time.time() < deadline:
                current = api.inspect(job.job_id)
                if current is not None and current.status.is_terminal:
                    break
                time.sleep(0.05)
            completed = api.inspect(job.job_id)
            assert completed is not None
            assert completed.status == JobStatus.COMPLETED
            assert completed.status_reason == "early_stopped_no_improvement"
            assert completed.metadata["early_stopping_result"]["stop_epoch"] == 4
            assert completed.metadata["early_stopping_result"]["epochs_saved"] == 1
            assert completed.metadata["early_stopping_best_checkpoint_path"]
            assert any(event["event_type"] == "job_early_stopped" for event in api.store.list_events())
        finally:
            service.stop()


class _PollingDrainSupervisor(_DrainSupervisor):
    def __init__(self, active_ids: list[str], snapshots: list[WorkerSnapshot]) -> None:
        super().__init__(active_ids)
        self.snapshots = snapshots

    def poll(self) -> list[WorkerSnapshot]:
        snapshots, self.snapshots = self.snapshots, []
        return snapshots


def test_pack_member_early_stop_removes_member_and_replans_replacement() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir, parallel_job_cap=2)
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
        stopped = store.submit_job(_job("stopped-member", "stopped-member-sig"))
        survivor = store.submit_job(_job("survivor", "survivor-sig"))
        replacement = store.submit_job(_job("replacement", "replacement-sig"))
        for job in (stopped, survivor, replacement):
            _seed_options(store, planner, job)
        store.set_job_status(stopped.job_id, JobStatus.COMPLETED, reason="early_stopped_no_improvement", hold=False)
        store.update_job(
            stopped.job_id,
            metadata_updates={"early_stopping_result": {"early_stopped_successfully": True}},
        )
        store.set_job_status(survivor.job_id, JobStatus.RUNNING, reason="test", hold=False)
        store.set_job_status(replacement.job_id, JobStatus.READY, reason="test", hold=False)
        store.upsert_pair_profile(
            PairProfile.create(
                "survivor-sig",
                "replacement-sig",
                backend_name="cuda_process",
                hardware_key=store.hardware_key(),
                slowdown_ratio=1.0,
                observations=1,
            )
        )
        supervisor = _PollingDrainSupervisor(
            [survivor.job_id],
            [
                    WorkerSnapshot(
                        job_id=stopped.job_id,
                        group_id="active-group",
                    alive=False,
                    returncode=0,
                    reported_by="store",
                )
            ],
        )
        service = SchedulerService(settings, store=store, supervisor=supervisor)
        service._active_runs["active-group"] = ActiveRun(
            group_id="active-group",
            mode="packed_pair",
            backend_name="cuda_process",
            job_ids=(stopped.job_id, survivor.job_id),
        )
        service._poll_active_workers()
        assert service._active_runs["active-group"].job_ids == (survivor.job_id,)
        service._dispatch_pending_work()
        assert supervisor.dispatched == [replacement.job_id]


def test_external_timeout_is_not_classified_as_training_failure() -> None:
    running = _job("timed", "timed-sig")
    running.status = JobStatus.RUNNING
    running.started_at = datetime.now(timezone.utc).isoformat()
    assert classify_job_outcome(running) == "training_started"
    assert classify_job_outcome(running, externally_timed_out=True) == "externally_timed_out"
    failed = running.copy(status=JobStatus.FAILED)
    assert classify_job_outcome(failed, externally_timed_out=True) == "failed"
    early = running.copy(
        status=JobStatus.COMPLETED,
        metadata={"early_stopping_result": {"early_stopped_successfully": True}},
    )
    assert classify_job_outcome(early) == "early_stopped_successfully"
