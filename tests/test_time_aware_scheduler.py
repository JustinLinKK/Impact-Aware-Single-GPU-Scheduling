from __future__ import annotations

from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
import random
import tempfile
import time
from unittest.mock import Mock

import pytest

from localml_scheduler.checkpointing.manager import CheckpointManager
from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import (
    BatchSizeObservation,
    BatchResolution,
    ColocationTimingProfile,
    CombinationProfile,
    JobStatus,
    PackingSpec,
    PairProfile,
    PlacementDecision,
    ResourceRequirements,
    RuntimeProfile,
    SafePointType,
    SchedulingClass,
    SoloProfile,
    TrainingJob,
    build_batch_size_observation_key,
    build_colocation_profile_key,
    build_group_signature,
)
from localml_scheduler.scheduler.early_stopping import (
    EarlyStoppingState,
    EarlyStoppingWatchdog,
)
from localml_scheduler.execution.control import (
    ControlPlane,
    EarlyStopRequested,
    PauseRequested,
    TrainingControlHook,
)
from localml_scheduler.execution.worker_runtime import mark_job_completed
from localml_scheduler.observability.events import EventLogger
from localml_scheduler.observability.outcomes import classify_job_outcome
from localml_scheduler.scheduler.placement_planner import PlacementPlanner
from localml_scheduler.scheduler.policies import PriorityFifoPolicy
from localml_scheduler.scheduler.resource_estimator import BatchOptionEstimate
from localml_scheduler.prediction import JobPredictionError, MLVramPredictor
from localml_scheduler.scheduler.telemetry import (
    GpuTelemetrySample,
    MemoryAdmissionGate,
)
from localml_scheduler.scheduler.time_objective import (
    EpochRateSet,
    TimeAwareObjectiveScorer,
    project_piecewise_drain,
)
from localml_scheduler.scheduler.service import ActiveRun, ColocationTrialState, SchedulerService
from localml_scheduler.scheduler.supervisor import WorkerSnapshot
from localml_scheduler.scheduler.trace_simulator import (
    TraceBackendChange,
    TraceBatchOption,
    TraceJob,
    TraceMemorySample,
    TraceProblem,
    _time_aware_choice,
    backend_aware_benchmark_fixture,
    benchmark_fixture,
    compare_policies,
    feasible_packs,
    markdown_table,
    simulate_recursive_time_aware,
    simulate_policy,
)
from localml_scheduler.storage.sqlite_store import SQLiteStateStore


def _settings(tmpdir: str, **gpu_overrides: object) -> SchedulerSettings:
    gpu = {
        "mode": "parallel_time_aware",
        "packing_backend": "cuda_process",
        "exclusive_fallback_enabled": True,
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


def _epoch_timing(
    epoch: int,
    seconds: float,
    *,
    started_at: datetime | None,
    finished_at: datetime,
    source: str = "safe_point_interval",
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "seconds": seconds,
        "started_at": started_at.isoformat() if started_at is not None else None,
        "finished_at": finished_at.isoformat(),
        "source": source,
    }


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


def test_time_aware_configuration_is_the_only_supported_policy() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        restored = SchedulerSettings(
            runtime_root=tmpdir,
            gpu_scheduler={"mode": "parallel_time_aware", "parallel_job_cap": 3},
            early_stopping={"mode": "min", "patience_epochs": 2},
        )
        assert restored.gpu_scheduler.parallel_job_cap == 3
        assert restored.gpu_scheduler.objective.objective_version == "time_v6_verified_piecewise_drain"
        assert restored.gpu_scheduler.colocation.min_gain == 1.0
        assert restored.gpu_scheduler.colocation.trial_epochs == 2
        assert restored.gpu_scheduler.colocation.trial_decision_timeout_seconds == 30
        assert restored.gpu_scheduler.colocation.trial_evidence_timeout_min_seconds == 300
        assert restored.gpu_scheduler.colocation.trial_evidence_timeout_max_seconds == 1800
        assert restored.gpu_scheduler.colocation.profile_rejection_min_bad_trials == 2
        assert restored.gpu_scheduler.colocation.profile_rejection_ttl_seconds == 86400
        assert restored.gpu_scheduler.colocation.live_trial_enabled
        for removed_version in (
            "time_v3_flow_only",
            "time_v4_colocation_gain",
            "time_v5_piecewise_drain",
        ):
            with pytest.raises(ValueError, match="objective_version must be"):
                SchedulerSettings(
                    runtime_root=tmpdir,
                    gpu_scheduler={
                        "mode": "parallel_time_aware",
                        "objective": {"objective_version": removed_version},
                    },
                )
        assert restored.early_stopping.mode == "min"
        serialized_gpu = restored.gpu_scheduler.to_dict()
        assert serialized_gpu["mode"] == "parallel_time_aware"
        for removed_key in (
            "max_packed_jobs_per_gpu",
            "thresholds",
            "auto_pack",
            "parallel_optimizer",
        ):
            assert removed_key not in serialized_gpu
        with pytest.raises(ValueError, match="timeout_max_seconds must be at least"):
            _settings(
                tmpdir,
                colocation={
                    "trial_evidence_timeout_min_seconds": 10,
                    "trial_evidence_timeout_max_seconds": 5,
                },
            )
        with pytest.raises(ValueError, match="min_bad_trials must be at least 1"):
            _settings(tmpdir, colocation={"profile_rejection_min_bad_trials": 0})
        for removed_mode in (
            "serial_basic",
            "serial_batch_optimized",
            "parallel_default",
            "parallel_batch_optimized",
            "parallel_auto_pack",
        ):
            with pytest.raises(ValueError, match="only supports parallel_time_aware"):
                SchedulerSettings(
                    runtime_root=tmpdir,
                    gpu_scheduler={"mode": removed_mode},
                )
        for removed_key, value in (
            ("max_packed_jobs_per_gpu", 3),
            ("thresholds", {}),
            ("auto_pack", {}),
            ("parallel_optimizer", {}),
        ):
            with pytest.raises(ValueError, match="Unsupported removed gpu_scheduler settings"):
                SchedulerSettings(runtime_root=tmpdir, gpu_scheduler={removed_key: value})
        with pytest.raises(ValueError, match="no longer supports throughput scheduling controls"):
            SchedulerSettings(
                runtime_root=tmpdir,
                gpu_scheduler={
                    "mode": "parallel_time_aware",
                    "objective": {"makespan_weight": 0.8, "flow_time_weight": 0.4},
                },
            )


def test_time_aware_placement_is_invariant_to_sm_utilization() -> None:
    def plan_for_utilization(avg_gpu_utilization: float) -> tuple[object, ...]:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _settings(
                tmpdir,
                parallel_job_cap=2,
            )
            store = SQLiteStateStore(settings)
            planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
            jobs = [_job("sm-left", "sm-left-sig"), _job("sm-right", "sm-right-sig")]
            for index, job in enumerate(jobs, start=1):
                job.queue_sequence = index
                job.runtime_probe.enabled = True
                _seed_options(store, planner, job)
                store.upsert_solo_profile(
                    SoloProfile(
                        signature=job.packing.signature or job.job_id,
                        family=job.packing.family,
                        peak_vram_mb=512,
                        avg_vram_mb=404.0,
                        avg_gpu_utilization=avg_gpu_utilization,
                        sample_count=1,
                        last_job_id=job.job_id,
                    )
                )
            plan = planner.choose_plan(
                jobs,
                backend_available={"cuda_process": True, "exclusive": True},
            )
            assert plan is not None
            assert plan.mode == "stack_anchor"
            return plan.mode, plan.backend_name, plan.job_ids, plan.batch_overrides

    baseline = plan_for_utilization(0.0)
    assert plan_for_utilization(0.9) == baseline
    assert plan_for_utilization(1.0) == baseline


def test_time_score_starts_one_anchor_and_ignores_untrusted_pair_slowdown() -> None:
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
        assert plan.mode == "stack_anchor"
        assert plan.job_ids == ("left",)
        assert plan.objective_version == "time_v6_verified_piecewise_drain"
        assert plan.objective_breakdown["requires_live_trial"] is False


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


def test_active_plus_new_jobs_ignore_legacy_parallel_cap_and_use_memory_admission() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir, parallel_job_cap=1)
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
        assert first.objective_breakdown["preexisting_job_ids"] == ["active"]
        assert first.objective_breakdown["requires_live_trial"]
        assert first.trial_metadata["preexisting_job_ids"] == ["active"]
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
        assert plan.objective_version == "time_v6_verified_piecewise_drain"
        assert plan.mode == "stack_anchor"
        assert plan.batch_overrides == {"left": 1}


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
        "backend_awared",
        "serial_fifo",
        "vram_fill_reference",
        "parallel_time_aware",
        "small_trace_oracle",
    }
    assert results["parallel_time_aware"].makespan_seconds < results["vram_fill_reference"].makespan_seconds
    assert results["parallel_time_aware"].mean_flow_seconds < results["serial_fifo"].mean_flow_seconds
    assert results["parallel_time_aware"].starvation_count == 0
    assert results["backend_awared"].hard_constraint_violations == 0
    report = markdown_table(results.values())
    assert "Total flow (s)" in report
    assert "Median flow (s)" in report
    assert "Actual over-budget packs" in report


def test_backend_aware_trace_ranking_reduces_trials_and_makespan() -> None:
    problem = backend_aware_benchmark_fixture()
    baseline = simulate_recursive_time_aware(problem)
    aware = simulate_recursive_time_aware(problem, backend_aware=True)
    assert aware.makespan_seconds < baseline.makespan_seconds
    assert aware.slowdown_rejections < baseline.slowdown_rejections
    assert aware.hard_constraint_violations == 0


class _DrainSupervisor:
    def __init__(self, active_ids: list[str]) -> None:
        self.active_ids = active_ids
        self.dispatched: list[str] = []
        self.paused: list[str] = []

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

    def request_pause(self, job_id: str, *, reason: str, hold: bool) -> bool:
        self.paused.append(job_id)
        return job_id in self.active_ids


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
                        left.job_id: "measured_epoch_against_exclusive_profile",
                        right.job_id: "measured_epoch_against_exclusive_profile",
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
        assert set(profile.metadata["slowdown_sources"].values()) == {"measured_epoch_against_exclusive_profile"}


def test_whole_run_elapsed_time_is_not_used_for_slowdown_measurement() -> None:
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
        assert profile.slowdown_ratio is None
        assert profile.metadata["slowdown_sources"] == {}


def test_batch_resolution_remains_immutable_across_dispatch_and_resume_round_trip() -> None:
    job = _job("round-trip", "round-trip-sig")
    assert job.requested_batch_size == 4
    dispatched = BatchResolution.apply(job, 16)
    dispatched.status = JobStatus.PAUSED
    restored = TrainingJob.from_dict(dispatched.to_dict())
    assert restored.requested_batch_size == 4
    assert BatchResolution.resolved_batch_size(restored) == 16
    assert restored.status == JobStatus.PAUSED


def test_batch_estimates_are_batch_specific_and_sourced() -> None:
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


def test_remaining_runtime_uses_batch_profile_epoch_time_without_runtime_profile() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
        job = _job("batch-runtime", "batch-runtime-sig")
        job.max_epochs = 4
        store.upsert_batch_size_observation(
            BatchSizeObservation(
                observation_key=build_batch_size_observation_key(
                    job.baseline_model_id,
                    planner.estimator.shape_signature(job),
                    store.hardware_key(),
                    "exclusive",
                    4,
                ),
                model_key=job.baseline_model_id,
                shape_signature=planner.estimator.shape_signature(job),
                hardware_key=store.hardware_key(),
                backend_name="exclusive",
                batch_param_name="batch_size",
                batch_size=4,
                avg_vram_mb=512.0,
                metadata={"seconds_per_epoch": 12.0},
            )
        )

        assert planner.predicted_remaining_runtime_seconds(job, backend_name="cuda_process") == 48.0


def test_runtime_estimate_reuses_completed_profile_within_same_branch() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
        shared_metadata = {
            "experiment_mode": "hardware_aware",
            "workflow_id": "petfinder-run",
            "branch_id": 2,
            "model_family": "efficientnet-b0",
        }
        observed = _job("observed", "observed-signature")
        observed.max_epochs = observed.config.max_epochs = 8
        observed.metadata.update(shared_metadata)
        store.submit_job(observed)
        store.upsert_runtime_profile(
            RuntimeProfile.create(
                signature="observed-signature",
                hardware_key=store.hardware_key(),
                backend_name="exclusive",
                resolved_batch_size=4,
                strategy="epoch_1",
                epoch_1_seconds=30.0,
                estimated_total_runtime_seconds=80.0,
                confidence=0.95,
                observations=2,
                last_job_id=observed.job_id,
                source="mlevolve_completed_wall_clock",
                metadata={"completed_epochs": 8},
            )
        )

        candidate = _job("candidate", "different-signature")
        candidate.max_epochs = candidate.config.max_epochs = 4
        candidate.metadata.update(shared_metadata)
        assert planner.estimator.predicted_remaining_runtime_seconds(
            candidate,
            backend_name="exclusive",
        ) == 40.0

        other_branch = _job("other-branch", "other-signature")
        other_branch.max_epochs = other_branch.config.max_epochs = 4
        other_branch.metadata.update(shared_metadata | {"branch_id": 3})
        assert planner.estimator.predicted_remaining_runtime_seconds(
            other_branch,
            backend_name="exclusive",
        ) is None

        other_family = _job("other-family", "other-family-signature")
        other_family.max_epochs = other_family.config.max_epochs = 4
        other_family.metadata.update(
            shared_metadata | {"model_family": "efficientnet-b3"}
        )
        assert planner.estimator.predicted_remaining_runtime_seconds(
            other_family,
            backend_name="exclusive",
        ) is None

        baseline_candidate = _job("baseline", "baseline-signature")
        baseline_candidate.max_epochs = baseline_candidate.config.max_epochs = 4
        baseline_candidate.metadata.update(
            shared_metadata | {"experiment_mode": "baseline"}
        )
        assert planner.estimator.predicted_remaining_runtime_seconds(
            baseline_candidate,
            backend_name="exclusive",
        ) is None


def test_zero_runtime_profile_falls_back_to_branch_epoch_estimate() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
        job = _job("zero-runtime", "zero-runtime-sig")
        job.max_epochs = job.config.max_epochs = 4
        job.metadata["last_completed_epoch"] = 1
        store.upsert_runtime_profile(
            RuntimeProfile.create(
                signature="zero-runtime-sig",
                hardware_key=store.hardware_key(),
                backend_name="exclusive",
                resolved_batch_size=4,
                strategy="epoch_1",
                estimated_total_runtime_seconds=0.0,
                observations=1,
                source="branch_profile",
            )
        )
        store.upsert_batch_size_observation(
            BatchSizeObservation(
                observation_key=build_batch_size_observation_key(
                    job.baseline_model_id,
                    planner.estimator.shape_signature(job),
                    store.hardware_key(),
                    "exclusive",
                    4,
                ),
                model_key=job.baseline_model_id,
                shape_signature=planner.estimator.shape_signature(job),
                hardware_key=store.hardware_key(),
                backend_name="exclusive",
                batch_param_name="batch_size",
                batch_size=4,
                avg_vram_mb=512.0,
                metadata={"seconds_per_epoch": 12.0},
            )
        )

        assert planner.predicted_remaining_runtime_seconds(job, backend_name="exclusive") == 36.0
        options = planner.estimator.estimate_batch_options(job, "exclusive", [4])
        assert options[0].seconds_per_epoch == 12.0
        assert options[0].remaining_runtime_seconds == 36.0


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


def test_memory_does_not_break_ties_between_equally_fast_feasible_options() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(
            settings,
            store,
            PriorityFifoPolicy(enable_priority_aging=False),
        )
        job = _job("memory-tie", "memory-tie-sig")
        _seed_options(store, planner, job)
        shape_signature = planner.estimator.shape_signature(job)
        for batch_size, avg_vram_mb in ((1, 7_000.0), (2, 100.0)):
            store.upsert_batch_size_observation(
                BatchSizeObservation(
                    observation_key=build_batch_size_observation_key(
                        job.baseline_model_id,
                        shape_signature,
                        store.hardware_key(),
                        "exclusive",
                        batch_size,
                    ),
                    model_key=job.baseline_model_id,
                    shape_signature=shape_signature,
                    hardware_key=store.hardware_key(),
                    backend_name="exclusive",
                    batch_param_name="batch_size",
                    batch_size=batch_size,
                    avg_vram_mb=avg_vram_mb,
                    metadata={"estimate_source": "test"},
                )
            )

        selected = planner._fastest_time_option(job, "exclusive")

        assert selected is not None
        assert selected.batch_size == 1


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
        left.metadata.update(
            {
                "placement_backend": "cuda_process",
                "runtime_observed_epoch_seconds": 10.0,
            }
        )
        plan = planner.choose_plan(
            [right],
            backend_available={"cuda_process": True, "exclusive": True},
            active_jobs=[left],
            active_vram_mb=404.0,
        )
        assert (plan is not None) is expect_pair
        if plan is not None:
            assert plan.mode == "concurrent_group"


@pytest.mark.parametrize("cap", [1, 2, 3, None])
def test_legacy_parallel_cap_does_not_restrict_stack_anchor(cap: int | None) -> None:
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
        assert len(plan.job_ids) == 1
        assert plan.mode == "stack_anchor"


@pytest.mark.parametrize(
    ("packed_rate", "expected_gain"),
    [(1.0, 2.0), (2.0, 1.0), (3.0, 2.0 / 3.0)],
)
def test_colocation_gain_above_equal_and_below_one(packed_rate: float, expected_gain: float) -> None:
    active = _job("gain-active", "gain-active-sig")
    candidate = _job("gain-candidate", "gain-candidate-sig")
    for job in (active, candidate):
        job.max_epochs = 10
        job.config.max_epochs = 10
    gain = TimeAwareObjectiveScorer.gain(
        [active],
        candidate,
        active_epoch_seconds={active.job_id: 1.0},
        candidate_solo_epoch_seconds=1.0,
        packed_epoch_seconds={active.job_id: packed_rate, candidate.job_id: packed_rate},
    )
    assert gain is not None
    assert gain[0] == pytest.approx(expected_gain)


@pytest.mark.parametrize(
    ("active_epochs", "candidate_epochs"),
    [(5, 10), (10, 5)],
)
def test_piecewise_gain_uses_mocked_singleton_speed_after_shorter_job_finishes(
    active_epochs: int,
    candidate_epochs: int,
) -> None:
    active = _job("piecewise-active", "piecewise-active-sig")
    candidate = _job("piecewise-candidate", "piecewise-candidate-sig")
    active.max_epochs = active.config.max_epochs = active_epochs
    candidate.max_epochs = candidate.config.max_epochs = candidate_epochs

    def mocked_tail_rates(member_ids: tuple[str, ...]) -> EpochRateSet:
        return EpochRateSet(
            epoch_seconds={job_id: 1.0 for job_id in member_ids},
            sources={job_id: "mock_singleton" for job_id in member_ids},
        )

    result = TimeAwareObjectiveScorer.gain(
        [active],
        candidate,
        active_epoch_seconds={active.job_id: 1.0},
        candidate_solo_epoch_seconds=1.0,
        packed_epoch_seconds={active.job_id: 1.5, candidate.job_id: 1.5},
        active_tail_rate_resolver=mocked_tail_rates,
        packed_tail_rate_resolver=mocked_tail_rates,
    )

    assert result is not None
    assert result.sequential_drain_seconds == pytest.approx(15.0)
    assert result.packed_drain_seconds == pytest.approx(12.5)
    assert result.gain == pytest.approx(1.2)
    assert [phase.duration_seconds for phase in result.packed_phases] == pytest.approx(
        [7.5, 5.0]
    )
    assert result.packed_phases[1].timing_sources == {
        (candidate.job_id if active_epochs < candidate_epochs else active.job_id): "mock_singleton"
    }


def test_estimate_gain_resolves_mocked_singleton_tail_from_estimator() -> None:
    settings = SchedulerSettings(
        gpu_scheduler={"mode": "parallel_time_aware"},
        graph_db={"enabled": False},
        hardware_feature_db={"enabled": False},
    )
    estimator = Mock()
    estimator.repository.hardware_key.return_value = "mock-gpu"
    estimator.repository.get_colocation_timing_profile.return_value = None
    estimator.estimate_batch_options.return_value = [
        BatchOptionEstimate(
            job_id="mocked",
            batch_size=4,
            avg_vram_mb=100.0,
            seconds_per_epoch=1.0,
            remaining_epochs=1,
            remaining_runtime_seconds=1.0,
            source="mock_runtime_profile",
            confidence=1.0,
            estimate_version="time_v6_verified_piecewise_drain",
        )
    ]
    scorer = TimeAwareObjectiveScorer(settings, estimator, Mock(), Mock())
    active = _job("mock-tail-active", "mock-tail-active-sig")
    candidate = _job("mock-tail-candidate", "mock-tail-candidate-sig")
    active.max_epochs = active.config.max_epochs = 5
    candidate.max_epochs = candidate.config.max_epochs = 10
    active.metadata.update({"placement_backend": "cuda_process", "resolved_batch_size": 4})
    candidate.metadata["resolved_batch_size"] = 4

    result = scorer.estimate_gain(
        [active],
        candidate,
        backend_name="cuda_process",
        active_epoch_seconds={active.job_id: 1.0},
        candidate_solo_epoch_seconds=1.0,
        packed_epoch_seconds={active.job_id: 1.5, candidate.job_id: 1.5},
        candidate_batch_size=4,
        active_epoch_sources={active.job_id: "mock_live_epoch"},
        packed_epoch_sources={
            active.job_id: "mock_trial",
            candidate.job_id: "mock_trial",
        },
    )

    assert result is not None
    assert result.packed_drain_seconds == pytest.approx(12.5)
    assert result.packed_phases[-1].timing_sources == {
        candidate.job_id: "mock_runtime_profile"
    }
    estimator.estimate_batch_options.assert_called_once_with(
        candidate,
        "cuda_process",
        [4],
    )


def test_piecewise_drain_uses_mocked_subset_rates_for_three_jobs() -> None:
    mocked_rates = {
        ("b", "c"): EpochRateSet(
            epoch_seconds={"b": 1.5, "c": 1.5},
            sources={"b": "mock_pair", "c": "mock_pair"},
        ),
        ("c",): EpochRateSet(
            epoch_seconds={"c": 1.0},
            sources={"c": "mock_singleton"},
        ),
    }
    projection = project_piecewise_drain(
        {"a": 2.0, "b": 6.0, "c": 8.0},
        {"a": 2.0, "b": 2.0, "c": 2.0},
        tail_rate_resolver=lambda member_ids: mocked_rates.get(member_ids),
    )

    assert projection is not None
    assert projection.total_seconds == pytest.approx(12.0)
    assert [phase.member_ids for phase in projection.phases] == [
        ("a", "b", "c"),
        ("b", "c"),
        ("c",),
    ]
    assert [phase.duration_seconds for phase in projection.phases] == pytest.approx(
        [4.0, 6.0, 2.0]
    )
    assert not any(phase.inherited_parent_rates for phase in projection.phases)


def test_piecewise_drain_conservatively_inherits_missing_mocked_subset_rates() -> None:
    projection = project_piecewise_drain(
        {"a": 2.0, "b": 6.0, "c": 8.0},
        {"a": 2.0, "b": 2.0, "c": 2.0},
        tail_rate_resolver=lambda _member_ids: None,
    )

    assert projection is not None
    assert projection.total_seconds == pytest.approx(16.0)
    assert [phase.duration_seconds for phase in projection.phases] == pytest.approx(
        [4.0, 8.0, 4.0]
    )
    assert [phase.inherited_parent_rates for phase in projection.phases] == [False, True, True]
    assert projection.phases[1].timing_sources == {
        "b": "inherited_parent_rate",
        "c": "inherited_parent_rate",
    }


def test_piecewise_drain_handles_simultaneous_completion_and_invalid_rates() -> None:
    empty = project_piecewise_drain({"a": 0.0}, {"a": 1.0})
    assert empty is not None
    assert empty.total_seconds == 0.0
    assert empty.phases == ()

    simultaneous = project_piecewise_drain(
        {"a": 5.0, "b": 5.0},
        {"a": 2.0, "b": 2.0},
    )
    assert simultaneous is not None
    assert simultaneous.total_seconds == pytest.approx(10.0)
    assert len(simultaneous.phases) == 1
    assert simultaneous.phases[0].completed_job_ids == ("a", "b")
    assert project_piecewise_drain({"a": 1.0}, {"a": 0.0}) is None

    malformed_tail = project_piecewise_drain(
        {"a": 1.0, "b": 2.0},
        {"a": 2.0, "b": 2.0},
        tail_rate_resolver=lambda member_ids: EpochRateSet(
            epoch_seconds={member_ids[0]: float("nan")},
            sources={member_ids[0]: "malformed_mock"},
        ),
    )
    assert malformed_tail is not None
    assert malformed_tail.total_seconds == pytest.approx(4.0)
    assert malformed_tail.phases[-1].inherited_parent_rates


def test_piecewise_drain_refreshes_rates_before_first_phase_after_zero_work_member() -> None:
    resolved_memberships: list[tuple[str, ...]] = []

    def resolve(member_ids: tuple[str, ...]) -> EpochRateSet:
        resolved_memberships.append(member_ids)
        return EpochRateSet(
            epoch_seconds={"live": 2.0},
            sources={"live": "mocked_singleton"},
        )

    projection = project_piecewise_drain(
        {"done": 0.0, "live": 5.0},
        {"live": 20.0},
        tail_rate_resolver=resolve,
    )
    assert projection is not None
    assert projection.total_seconds == pytest.approx(10.0)
    assert resolved_memberships == [("live",)]
    assert projection.phases[0].timing_sources == {"live": "mocked_singleton"}
    assert not projection.phases[0].inherited_parent_rates


def test_piecewise_drain_inherits_parent_rate_when_initial_reduced_profile_is_missing() -> None:
    projection = project_piecewise_drain(
        {"done": 0.0, "live": 5.0},
        {"live": 20.0},
        tail_rate_resolver=lambda _: None,
    )
    assert projection is not None
    assert projection.total_seconds == pytest.approx(100.0)
    assert projection.phases[0].timing_sources == {"live": "inherited_parent_rate"}
    assert projection.phases[0].inherited_parent_rates


def test_gain_does_not_require_rates_for_zero_remaining_members() -> None:
    active = _job("already-done", "already-done-sig")
    candidate = _job("still-running", "still-running-sig")
    active.max_epochs = active.config.max_epochs = 2
    active.metadata["last_completed_epoch"] = 2
    candidate.max_epochs = candidate.config.max_epochs = 5
    result = TimeAwareObjectiveScorer.gain(
        [active],
        candidate,
        active_epoch_seconds={},
        candidate_solo_epoch_seconds=1.0,
        packed_epoch_seconds={candidate.job_id: 2.0},
    )
    assert result is not None
    assert result.sequential_drain_seconds == pytest.approx(5.0)
    assert result.packed_drain_seconds == pytest.approx(10.0)

    active.metadata["last_completed_epoch"] = 0
    candidate.metadata["last_completed_epoch"] = 5
    zero_candidate = TimeAwareObjectiveScorer.gain(
        [active],
        candidate,
        active_epoch_seconds={active.job_id: 1.0},
        candidate_solo_epoch_seconds=0.0,
        packed_epoch_seconds={active.job_id: 2.0},
    )
    assert zero_candidate is not None
    assert zero_candidate.sequential_drain_seconds == pytest.approx(2.0)
    assert zero_candidate.packed_drain_seconds == pytest.approx(4.0)


def test_colocation_profile_key_isolated_by_hardware_backend_batch_duplicates_and_size() -> None:
    base = [
        {"signature": "same", "batch_size": 4, "backend_name": "cuda_process"},
        {"signature": "same", "batch_size": 4, "backend_name": "cuda_process"},
    ]
    key = build_colocation_profile_key("gpu-a", base)
    assert key == build_colocation_profile_key("gpu-a", list(reversed(base)))
    assert key != build_colocation_profile_key("gpu-b", base)
    assert key != build_colocation_profile_key(
        "gpu-a",
        [{**base[0], "batch_size": 8}, base[1]],
    )
    assert key != build_colocation_profile_key(
        "gpu-a",
        [{**base[0], "backend_name": "mps_process"}, base[1]],
    )
    assert key != build_colocation_profile_key("gpu-a", base[:1])
    assert key != build_colocation_profile_key(
        "gpu-a",
        [*base, {"signature": "same", "batch_size": 4, "backend_name": "cuda_process"}],
    )


def test_exact_colocation_profile_below_gain_is_rejected_before_dispatch_and_stalls() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir, parallel_job_cap=2)
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
        active = _job("known-active", "known-active-sig")
        candidate = _job("known-candidate", "known-candidate-sig")
        for job in (active, candidate):
            job.max_epochs = 10
            job.config.max_epochs = 10
            _seed_options(store, planner, job)
        active.metadata.update(
            {
                "placement_backend": "cuda_process",
                "runtime_observed_epoch_seconds": 1.0,
                "resolved_batch_size": 4,
            }
        )
        active = store.submit_job(active)
        candidate = store.submit_job(candidate)
        store.set_job_status(active.job_id, JobStatus.RUNNING, reason="test", hold=False)
        store.set_job_status(candidate.job_id, JobStatus.READY, reason="test", hold=False)
        members = [
            {"signature": "known-active-sig", "batch_size": 4, "backend_name": "cuda_process"},
            {"signature": "known-candidate-sig", "batch_size": 1, "backend_name": "cuda_process"},
        ]
        observed_at = datetime.now(timezone.utc).isoformat()
        store.upsert_colocation_timing_profile(
            ColocationTimingProfile.create(
                store.hardware_key(),
                members,
                [{**member, "seconds_per_epoch": 20.0, "observations": 2} for member in members],
                observations=2,
                metadata={
                    "evidence_policy": "fresh_member_epochs_v1",
                    "recent_trial_outcomes": [
                        {"trial_id": "known-bad-1", "decision": "rejected", "gain": 0.5, "observed_at": observed_at},
                        {"trial_id": "known-bad-2", "decision": "rejected", "gain": 0.5, "observed_at": observed_at},
                    ]
                },
            )
        )
        plan = planner.choose_plan(
            [store.get_job(candidate.job_id)],
            backend_available={"cuda_process": True, "exclusive": True},
            active_jobs=[store.get_job(active.job_id)],
            active_vram_mb=404.0,
        )
        assert plan is not None
        assert plan.objective_breakdown["gain"] < 1.0
        assert plan.objective_breakdown["colocation_rejected"]
        supervisor = _DrainSupervisor([active.job_id])
        service = SchedulerService(settings, store=store, supervisor=supervisor)
        assert not service._dispatch_plan(plan)
        assert supervisor.dispatched == []
        assert service._colocation_stall is not None
        rejected = store.get_job(candidate.job_id)
        assert rejected is not None and rejected.status == JobStatus.PAUSED and not rejected.hold


def test_single_bad_profile_requires_another_live_trial() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir, parallel_job_cap=2)
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
        active = _job("one-bad-active", "one-bad-active-sig")
        candidate = _job("one-bad-candidate", "one-bad-candidate-sig")
        for job in (active, candidate):
            job.max_epochs = job.config.max_epochs = 10
            _seed_options(store, planner, job)
        active.metadata.update(
            {
                "placement_backend": "cuda_process",
                "runtime_observed_epoch_seconds": 1.0,
                "resolved_batch_size": 4,
            }
        )
        members = [
            {"signature": active.packing.signature, "batch_size": 4, "backend_name": "cuda_process"},
            {"signature": candidate.packing.signature, "batch_size": 1, "backend_name": "cuda_process"},
        ]
        observed_at = datetime.now(timezone.utc).isoformat()
        store.upsert_colocation_timing_profile(
            ColocationTimingProfile.create(
                store.hardware_key(),
                members,
                [{**member, "seconds_per_epoch": 20.0, "observations": 1} for member in members],
                observations=1,
                metadata={
                    "evidence_policy": "fresh_member_epochs_v1",
                    "recent_trial_outcomes": [
                        {"trial_id": "single-bad", "decision": "rejected", "gain": 0.5, "observed_at": observed_at}
                    ]
                },
            )
        )
        plan = planner.choose_plan(
            [candidate],
            backend_available={"cuda_process": True, "exclusive": True},
            active_jobs=[active],
            active_vram_mb=404.0,
        )
        assert plan is not None
        assert plan.objective_breakdown["known_profile"]
        assert not plan.objective_breakdown["trusted_profile"]
        assert not plan.objective_breakdown["colocation_rejected"]
        assert plan.objective_breakdown["requires_live_trial"]


def test_profile_rejection_requires_two_recent_consecutive_bad_trials() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
        descriptor = {"signature": "profile-member", "batch_size": 4, "backend_name": "cuda_process"}
        now = datetime.now(timezone.utc)

        def profile(outcomes: list[dict[str, object]], *, updated_at: str | None = None) -> ColocationTimingProfile:
            return ColocationTimingProfile.create(
                store.hardware_key(),
                [descriptor],
                [{**descriptor, "seconds_per_epoch": 2.0, "observations": 2}],
                observations=2,
                updated_at=updated_at or now.isoformat(),
                metadata={
                    "evidence_policy": "fresh_member_epochs_v1",
                    "recent_trial_outcomes": outcomes,
                },
            )

        bad_1 = {"trial_id": "bad-1", "decision": "rejected", "gain": 0.8, "observed_at": now.isoformat()}
        bad_2 = {"trial_id": "bad-2", "decision": "rejected", "gain": 0.8, "observed_at": now.isoformat()}
        good = {"trial_id": "good", "decision": "accepted", "gain": 1.1, "observed_at": now.isoformat()}
        assert planner.time_objective.profile_rejection_trusted(profile([bad_1, bad_2]), now=now)
        assert not planner.time_objective.profile_rejection_trusted(profile([bad_1, good]), now=now)
        assert not planner.time_objective.profile_rejection_trusted(profile([good, bad_2]), now=now)
        stale_at = (now - timedelta(days=2)).isoformat()
        assert not planner.time_objective.profile_rejection_trusted(
            profile(
                [
                    {**bad_1, "observed_at": stale_at},
                    {**bad_2, "observed_at": stale_at},
                ],
                updated_at=stale_at,
            ),
            now=now,
        )
        assert not planner.time_objective.profile_rejection_trusted(profile([]), now=now)
        unversioned = profile([bad_1, bad_2])
        unversioned.metadata.pop("evidence_policy")
        assert not planner.time_objective.profile_rates_trusted(unversioned, now=now)


def test_expired_profile_is_replaced_instead_of_averaged() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        store = SQLiteStateStore(settings)
        jobs = [_job("expired-a", "expired-a-sig"), _job("expired-b", "expired-b-sig")]
        for job in jobs:
            job.metadata.update({"placement_backend": "cuda_process", "resolved_batch_size": 4})
        service = SchedulerService(
            settings,
            store=store,
            supervisor=_DrainSupervisor([job.job_id for job in jobs]),
        )
        descriptors = [service._member_descriptor(job, "cuda_process") for job in jobs]
        expired = ColocationTimingProfile.create(
            store.hardware_key(),
            descriptors,
            [{**descriptor, "seconds_per_epoch": 100.0, "observations": 7} for descriptor in descriptors],
            observations=7,
            metadata={"recent_trial_outcomes": [{"decision": "rejected"}]},
        )
        store.upsert_colocation_timing_profile(expired)
        with store._connect() as connection:
            connection.execute(
                "UPDATE colocation_timing_profiles SET updated_at = ? WHERE profile_key = ?",
                ((datetime.now(timezone.utc) - timedelta(days=2)).isoformat(), expired.profile_key),
            )
            connection.commit()
        started = datetime.now(timezone.utc)
        trial = ColocationTrialState(
            trial_id="expired-retry",
            candidate_job_id=jobs[1].job_id,
            preexisting_job_ids=(jobs[0].job_id,),
            started_at=started.isoformat(),
            start_epoch=0,
            target_epoch=2,
            backend_name="cuda_process",
            profile_key=expired.profile_key,
            candidate_solo_epoch_seconds=1.0,
            pretrial_epoch_seconds={jobs[0].job_id: 1.0},
        )
        service._persist_colocation_timing_profile(
            jobs,
            {jobs[0].job_id: 2.0, jobs[1].job_id: 3.0},
            trial,
            gain=0.8,
            decision="rejected",
        )
        service._persist_colocation_timing_profile(
            jobs,
            {jobs[0].job_id: 2.0, jobs[1].job_id: 3.0},
            trial,
            gain=0.8,
            decision="rejected",
        )
        refreshed = store.get_colocation_timing_profile(expired.profile_key)
        assert refreshed is not None and refreshed.observations == 1
        assert [item["seconds_per_epoch"] for item in refreshed.member_timings] == [2.0, 3.0]
        assert len(refreshed.metadata["recent_trial_outcomes"]) == 1


def test_two_real_trial_epochs_hit_decision_barrier_and_rejection_preserves_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        settings.ensure_runtime_layout()
        store = SQLiteStateStore(settings)
        job = _job("trial-barrier", "trial-barrier-sig")
        job.max_epochs = 10
        job.config.max_epochs = 10
        job.metadata["colocation_trial"] = {
            "trial_id": "trial-barrier",
            "start_epoch": 0,
            "target_epoch": 2,
            "decision": "pending",
        }
        job = store.submit_job(job)
        store.set_job_status(job.job_id, JobStatus.RUNNING, reason="test", hold=False)
        events = EventLogger(store, settings.events_jsonl_path)
        control = ControlPlane(settings)
        control.initialize_job(job.job_id)
        hook = TrainingControlHook(job, control, CheckpointManager(settings, store, events), store, events)
        hook.safe_point(
            SafePointType.EPOCH,
            epoch=1,
            global_step=1,
            state_factory=lambda: {"epoch": 1},
            steps_per_epoch=2,
            avg_step_time_ms=500.0,
        )
        store.update_job(
            job.job_id,
            metadata_updates={
                "colocation_trial": {
                    **store.get_job(job.job_id).metadata["colocation_trial"],
                    "decision": "rejected",
                    "reason": "gain below one",
                }
            },
        )
        with pytest.raises(PauseRequested):
            hook.safe_point(
                SafePointType.EPOCH,
                epoch=2,
                global_step=2,
                state_factory=lambda: {"epoch": 2},
                steps_per_epoch=2,
                avg_step_time_ms=500.0,
            )
        paused = store.get_job(job.job_id)
        assert paused is not None
        assert paused.metadata["last_completed_epoch"] == 2
        assert paused.status == JobStatus.PAUSED and not paused.hold
        assert paused.latest_checkpoint_path


def test_accepted_trial_retains_two_epochs_and_releases_third_epoch() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        settings.ensure_runtime_layout()
        store = SQLiteStateStore(settings)
        job = _job("accepted-barrier", "accepted-barrier-sig")
        job.max_epochs = 10
        job.config.max_epochs = 10
        job.metadata["colocation_trial"] = {
            "trial_id": "accepted-barrier",
            "start_epoch": 0,
            "target_epoch": 2,
            "decision": "pending",
        }
        job = store.submit_job(job)
        store.set_job_status(job.job_id, JobStatus.RUNNING, reason="test", hold=False)
        events = EventLogger(store, settings.events_jsonl_path)
        control = ControlPlane(settings)
        control.initialize_job(job.job_id)
        hook = TrainingControlHook(job, control, CheckpointManager(settings, store, events), store, events)
        hook.safe_point(SafePointType.EPOCH, epoch=1, global_step=1, state_factory=lambda: {"epoch": 1})
        store.update_job(
            job.job_id,
            metadata_updates={
                "colocation_trial": {
                    **store.get_job(job.job_id).metadata["colocation_trial"],
                    "decision": "accepted",
                }
            },
        )
        hook.safe_point(SafePointType.EPOCH, epoch=2, global_step=2, state_factory=lambda: {"epoch": 2})
        hook.safe_point(SafePointType.EPOCH, epoch=3, global_step=3, state_factory=lambda: {"epoch": 3})
        running = store.get_job(job.job_id)
        assert running is not None and running.status == JobStatus.RUNNING
        assert running.metadata["last_completed_epoch"] == 3


def test_trial_decision_timeout_checkpoints_and_pauses_as_unverified() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(
            tmpdir,
            colocation={
                "min_gain": 1.0,
                "trial_epochs": 2,
                "trial_decision_timeout_seconds": 0.05,
                "live_trial_enabled": True,
            },
        )
        settings.ensure_runtime_layout()
        store = SQLiteStateStore(settings)
        job = _job("timeout-barrier", "timeout-barrier-sig")
        job.max_epochs = 10
        job.config.max_epochs = 10
        job.metadata["colocation_trial"] = {
            "trial_id": "timeout-barrier",
            "start_epoch": 0,
            "target_epoch": 1,
            "decision": "pending",
        }
        job = store.submit_job(job)
        store.set_job_status(job.job_id, JobStatus.RUNNING, reason="test", hold=False)
        events = EventLogger(store, settings.events_jsonl_path)
        control = ControlPlane(settings)
        control.initialize_job(job.job_id)
        hook = TrainingControlHook(job, control, CheckpointManager(settings, store, events), store, events)
        with pytest.raises(PauseRequested):
            hook.safe_point(
                SafePointType.EPOCH,
                epoch=1,
                global_step=1,
                state_factory=lambda: {"epoch": 1},
            )
        paused = store.get_job(job.job_id)
        assert paused is not None and paused.status == JobStatus.PAUSED and not paused.hold
        assert paused.metadata["colocation_trial"]["decision"] == "timeout"
        assert paused.latest_checkpoint_path


def test_trial_evidence_uses_only_two_newest_fresh_member_epochs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        store = SQLiteStateStore(settings)
        service = SchedulerService(settings, store=store, supervisor=_DrainSupervisor([]))
        trial_started = datetime.now(timezone.utc)
        active = _job("evidence-active", "evidence-active-sig")
        active.metadata.update(
            {
                "runtime_avg_step_time_ms": 999999.0,
                "runtime_steps_per_epoch": 999,
                "runtime_epoch_timing_history": [
                    _epoch_timing(
                        1,
                        100.0,
                        started_at=trial_started - timedelta(seconds=5),
                        finished_at=trial_started - timedelta(seconds=1),
                    ),
                    _epoch_timing(
                        2,
                        50.0,
                        started_at=trial_started - timedelta(seconds=1),
                        finished_at=trial_started + timedelta(seconds=1),
                    ),
                    _epoch_timing(
                        3,
                        4.0,
                        started_at=trial_started + timedelta(seconds=1),
                        finished_at=trial_started + timedelta(seconds=5),
                    ),
                    _epoch_timing(
                        4,
                        6.0,
                        started_at=trial_started + timedelta(seconds=6),
                        finished_at=trial_started + timedelta(seconds=12),
                    ),
                    _epoch_timing(
                        5,
                        8.0,
                        started_at=trial_started + timedelta(seconds=13),
                        finished_at=trial_started + timedelta(seconds=21),
                    ),
                ],
            }
        )
        candidate = _job("evidence-candidate", "evidence-candidate-sig")
        candidate.metadata["runtime_epoch_timing_history"] = [
            _epoch_timing(
                1,
                2.0,
                started_at=None,
                finished_at=trial_started + timedelta(seconds=2),
                source="runner_step_time",
            ),
            _epoch_timing(
                2,
                4.0,
                started_at=trial_started + timedelta(seconds=2),
                finished_at=trial_started + timedelta(seconds=6),
            ),
        ]
        trial = ColocationTrialState(
            trial_id="fresh-evidence",
            candidate_job_id=candidate.job_id,
            preexisting_job_ids=(active.job_id,),
            started_at=trial_started.isoformat(),
            start_epoch=0,
            target_epoch=2,
            backend_name="cuda_process",
            profile_key="fresh-evidence-profile",
            candidate_solo_epoch_seconds=1.0,
            member_start_epochs={active.job_id: 0, candidate.job_id: 0},
            evidence_deadline_at=(trial_started + timedelta(minutes=5)).isoformat(),
        )
        active_evidence = service._trial_epoch_evidence(active, trial)
        candidate_evidence = service._trial_epoch_evidence(candidate, trial)
        assert active_evidence.samples == (6.0, 8.0)
        assert active_evidence.seconds_per_epoch == pytest.approx(7.0)
        assert candidate_evidence.samples == (2.0, 4.0)
        assert candidate_evidence.seconds_per_epoch == pytest.approx(3.0)


def test_missing_member_evidence_extends_candidate_target_without_profile_fallback() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        store = SQLiteStateStore(settings)
        started = datetime.now(timezone.utc) - timedelta(seconds=10)
        active = _job("extend-active", "extend-active-sig")
        candidate = _job("extend-candidate", "extend-candidate-sig")
        for job in (active, candidate):
            job.max_epochs = job.config.max_epochs = 10
            job.metadata.update({"placement_backend": "cuda_process", "resolved_batch_size": 4})
            store.submit_job(job)
            store.set_job_status(job.job_id, JobStatus.RUNNING, reason="test", hold=False)
        store.update_job(
            candidate.job_id,
            metadata_updates={
                "last_completed_epoch": 2,
                "runtime_epoch_timing_history": [
                    _epoch_timing(
                        1,
                        2.0,
                        started_at=None,
                        finished_at=started + timedelta(seconds=2),
                        source="runner_step_time",
                    ),
                    _epoch_timing(
                        2,
                        2.0,
                        started_at=started + timedelta(seconds=2),
                        finished_at=started + timedelta(seconds=4),
                    ),
                ],
            },
        )
        descriptors = [
            {"signature": active.packing.signature, "batch_size": 4, "backend_name": "cuda_process"},
            {"signature": candidate.packing.signature, "batch_size": 4, "backend_name": "cuda_process"},
        ]
        existing = ColocationTimingProfile.create(
            store.hardware_key(),
            descriptors,
            [{**descriptor, "seconds_per_epoch": 99.0, "observations": 2} for descriptor in descriptors],
            observations=2,
        )
        store.upsert_colocation_timing_profile(existing)
        supervisor = _DrainSupervisor([active.job_id, candidate.job_id])
        service = SchedulerService(settings, store=store, supervisor=supervisor)
        service._colocation_trial = ColocationTrialState(
            trial_id="extend-trial",
            candidate_job_id=candidate.job_id,
            preexisting_job_ids=(active.job_id,),
            started_at=started.isoformat(),
            start_epoch=0,
            target_epoch=2,
            backend_name="cuda_process",
            profile_key=existing.profile_key,
            candidate_solo_epoch_seconds=1.0,
            pretrial_epoch_seconds={active.job_id: 1.0},
            member_start_epochs={active.job_id: 0, candidate.job_id: 0},
            evidence_deadline_at=(started + timedelta(minutes=5)).isoformat(),
        )
        service._evaluate_colocation_trial()
        assert service._colocation_trial is not None
        assert service._colocation_trial.target_epoch == 3
        pending = store.get_job(candidate.job_id)
        assert pending is not None
        assert pending.metadata["colocation_trial"]["target_epoch"] == 3
        assert pending.metadata["colocation_trial"]["evidence"]["evidence_counts"] == {
            active.job_id: 0,
            candidate.job_id: 2,
        }
        assert store.get_colocation_timing_profile(existing.profile_key).observations == 2
        assert supervisor.paused == []


def test_evidence_deadline_pauses_unverified_without_profile_or_stall() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        store = SQLiteStateStore(settings)
        started = datetime.now(timezone.utc) - timedelta(minutes=10)
        active = _job("deadline-active", "deadline-active-sig")
        candidate = _job("deadline-candidate", "deadline-candidate-sig")
        for job in (active, candidate):
            job.max_epochs = job.config.max_epochs = 10
            job.metadata.update({"placement_backend": "cuda_process", "resolved_batch_size": 4})
            store.submit_job(job)
            store.set_job_status(job.job_id, JobStatus.RUNNING, reason="test", hold=False)
        store.update_job(candidate.job_id, metadata_updates={"last_completed_epoch": 2})
        supervisor = _DrainSupervisor([active.job_id, candidate.job_id])
        service = SchedulerService(settings, store=store, supervisor=supervisor)
        descriptors = [service._member_descriptor(job, "cuda_process") for job in (active, candidate)]
        profile_key = build_colocation_profile_key(store.hardware_key(), descriptors)
        service._colocation_trial = ColocationTrialState(
            trial_id="deadline-trial",
            candidate_job_id=candidate.job_id,
            preexisting_job_ids=(active.job_id,),
            started_at=started.isoformat(),
            start_epoch=0,
            target_epoch=2,
            backend_name="cuda_process",
            profile_key=profile_key,
            candidate_solo_epoch_seconds=1.0,
            pretrial_epoch_seconds={active.job_id: 1.0},
            member_start_epochs={active.job_id: 0, candidate.job_id: 0},
            evidence_deadline_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
        )
        service._evaluate_colocation_trial()
        assert service._colocation_trial is None
        assert service._colocation_stall is None
        assert supervisor.paused == [candidate.job_id]
        paused = store.get_job(candidate.job_id)
        assert paused is not None and paused.status == JobStatus.PAUSING
        assert paused.metadata["colocation_trial"]["decision"] == "timeout"
        assert paused.metadata["colocation_unverified_profile_key"] == profile_key
        assert store.get_colocation_timing_profile(profile_key) is None


def test_control_hook_releases_wait_when_scheduler_advances_trial_target() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        job = _job("rolling-target", "rolling-target-sig")
        waiting = job.copy(metadata={"colocation_trial": {"target_epoch": 2, "decision": "pending"}})
        extended = job.copy(metadata={"colocation_trial": {"target_epoch": 3, "decision": "pending"}})
        control = Mock()
        control.settings = settings
        control.read_command.return_value.action = "none"
        store = Mock()
        store.get_job.side_effect = [waiting, extended]
        hook = TrainingControlHook(job, control, Mock(), store, Mock())
        command = hook._trial_command_at_epoch(2)
        assert command is not None and command.action == "none"


def test_newcomer_finishing_without_full_evidence_is_completed_unverified() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir, parallel_job_cap=2)
        store = SQLiteStateStore(settings)
        active = _job("finish-active", "finish-active-sig")
        candidate = _job("finish-candidate", "finish-candidate-sig")
        active.max_epochs = active.config.max_epochs = 10
        candidate.max_epochs = candidate.config.max_epochs = 2
        started = datetime.now(timezone.utc) - timedelta(seconds=10)
        for job in (active, candidate):
            job.metadata.update(
                {
                    "placement_backend": "cuda_process",
                    "resolved_batch_size": 4,
                    "runtime_epoch_timing_history": [
                        {
                            "epoch": 2,
                            "seconds": 3.0,
                            "started_at": (started + timedelta(seconds=1)).isoformat(),
                            "finished_at": (started + timedelta(seconds=4)).isoformat(),
                        }
                    ],
                }
            )
            store.submit_job(job)
            store.set_job_status(job.job_id, JobStatus.RUNNING, reason="test", hold=False)
        store.update_job(candidate.job_id, metadata_updates={"last_completed_epoch": 2})
        supervisor = _DrainSupervisor([active.job_id, candidate.job_id])
        service = SchedulerService(settings, store=store, supervisor=supervisor)
        service._colocation_trial = ColocationTrialState(
            trial_id="finish-at-target",
            candidate_job_id=candidate.job_id,
            preexisting_job_ids=(active.job_id,),
            started_at=started.isoformat(),
            start_epoch=0,
            target_epoch=2,
            backend_name="cuda_process",
            profile_key="finish-profile",
            candidate_solo_epoch_seconds=1.0,
            pretrial_epoch_seconds={active.job_id: 1.0},
            member_start_epochs={active.job_id: 0, candidate.job_id: 0},
            evidence_deadline_at=(started + timedelta(minutes=5)).isoformat(),
        )
        service._evaluate_colocation_trial()
        assert service._colocation_trial is None
        assert service._colocation_stall is None
        assert supervisor.paused == []
        result = store.get_job(candidate.job_id)
        assert result is not None
        assert result.metadata["colocation_trial"]["decision"] == "completed_unverified"
        assert result.metadata["colocation_unverified_profile_key"] == "finish-profile"
        assert store.get_colocation_timing_profile("finish-profile") is None
        hook = TrainingControlHook(
            result,
            ControlPlane(settings),
            Mock(),
            store,
            Mock(),
        )
        command = hook._trial_command_at_epoch(2)
        assert command is not None and command.action == "none"


def test_membership_change_restarts_trial_window_and_trial_state_recovers() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir, parallel_job_cap=3)
        store = SQLiteStateStore(settings)
        jobs = [_job("restart-a", "restart-a-sig"), _job("restart-b", "restart-b-sig"), _job("restart-c", "restart-c-sig")]
        for job in jobs:
            job.max_epochs = 10
            job.config.max_epochs = 10
            job.metadata.update(
                {
                    "placement_backend": "cuda_process",
                    "runtime_observed_epoch_seconds": 1.0,
                    "resolved_batch_size": 4,
                }
            )
            store.submit_job(job)
            store.set_job_status(job.job_id, JobStatus.RUNNING, reason="test", hold=False)
        store.update_job("restart-c", metadata_updates={"last_completed_epoch": 2})
        supervisor = _DrainSupervisor(["restart-a", "restart-c"])
        service = SchedulerService(settings, store=store, supervisor=supervisor)
        service._colocation_trial = ColocationTrialState(
            trial_id="old-window",
            candidate_job_id="restart-c",
            preexisting_job_ids=("restart-a",),
            started_at=(datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat(),
            start_epoch=0,
            target_epoch=2,
            backend_name="cuda_process",
            profile_key="old-key",
            candidate_solo_epoch_seconds=1.0,
            pretrial_epoch_seconds={"restart-a": 1.0, "restart-b": 1.0},
        )
        service._persist_scheduler_decision_state()
        recovered = SchedulerService(settings, store=store, supervisor=supervisor)
        assert recovered._colocation_trial is not None
        supervisor.active_ids = ["restart-b", "restart-c"]
        recovered._evaluate_colocation_trial()
        assert recovered._colocation_trial is not None
        assert recovered._colocation_trial.trial_id != "old-window"
        assert recovered._colocation_trial.preexisting_job_ids == ("restart-b",)
        assert recovered._colocation_trial.start_epoch == 2
        assert recovered._colocation_trial.target_epoch == 4


def test_clean_epoch_trial_persists_exact_timing_and_pair_slowdown() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir)
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
        jobs = [_job("timing-a", "timing-a-sig"), _job("timing-b", "timing-b-sig")]
        for job in jobs:
            job.max_epochs = 10
            job.config.max_epochs = 10
            job.metadata.update({"placement_backend": "cuda_process", "resolved_batch_size": 4})
            _seed_options(store, planner, job)
        service = SchedulerService(settings, store=store, supervisor=_DrainSupervisor([job.job_id for job in jobs]))
        descriptors = [service._member_descriptor(job, "cuda_process") for job in jobs]
        trial = ColocationTrialState(
            trial_id="clean-timing",
            candidate_job_id="timing-b",
            preexisting_job_ids=("timing-a",),
            started_at=datetime.now(timezone.utc).isoformat(),
            start_epoch=0,
            target_epoch=2,
            backend_name="cuda_process",
            profile_key=build_colocation_profile_key(store.hardware_key(), descriptors),
            candidate_solo_epoch_seconds=10.0,
            pretrial_epoch_seconds={"timing-a": 10.0},
        )
        service._persist_colocation_timing_profile(
            jobs,
            {"timing-a": 12.0, "timing-b": 15.0},
            trial,
        )
        profile = store.get_colocation_timing_profile(trial.profile_key)
        assert profile is not None and len(profile.member_timings) == 2
        pair = store.get_pair_profile("timing-a-sig", "timing-b-sig", backend_name="cuda_process")
        assert pair is not None and pair.slowdown_ratio == pytest.approx(1.5)
        assert set(pair.metadata["slowdown_sources"].values()) == {
            "measured_epoch_against_exclusive_profile"
        }


def test_measured_slowdown_rejection_stalls_all_candidates_until_preexisting_member_leaves() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _settings(tmpdir, parallel_job_cap=4)
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
        active_jobs = [_job("stack-a", "stack-a-sig"), _job("stack-b", "stack-b-sig")]
        candidate = _job("stack-c", "stack-c-sig")
        later = _job("stack-d", "stack-d-sig")
        started = datetime.now(timezone.utc) - timedelta(seconds=10)
        for job in [*active_jobs, candidate, later]:
            job.max_epochs = 10
            job.config.max_epochs = 10
            _seed_options(store, planner, job)
            job.metadata.update({"placement_backend": "cuda_process", "resolved_batch_size": 4})
            store.submit_job(job)
        for job in active_jobs:
            store.update_job(
                job.job_id,
                status=JobStatus.RUNNING,
                hold=False,
                metadata_updates={
                    "runtime_epoch_timing_history": [
                        {
                            "epoch": 1,
                            "seconds": 3.0,
                            "started_at": (started + timedelta(seconds=1)).isoformat(),
                            "finished_at": (started + timedelta(seconds=4)).isoformat(),
                            "source": "safe_point_interval",
                        },
                        {
                            "epoch": 2,
                            "seconds": 3.0,
                            "started_at": (started + timedelta(seconds=5)).isoformat(),
                            "finished_at": (started + timedelta(seconds=8)).isoformat(),
                            "source": "safe_point_interval",
                        },
                    ]
                },
            )
        store.update_job(
            candidate.job_id,
            status=JobStatus.RUNNING,
            hold=False,
            metadata_updates={
                "last_completed_epoch": 2,
                "runtime_epoch_timing_history": [
                    {
                        "epoch": 1,
                        "seconds": 3.0,
                        "started_at": (started + timedelta(seconds=1)).isoformat(),
                        "finished_at": (started + timedelta(seconds=4)).isoformat(),
                        "source": "safe_point_interval",
                    },
                    {
                        "epoch": 2,
                        "seconds": 3.0,
                        "started_at": (started + timedelta(seconds=5)).isoformat(),
                        "finished_at": (started + timedelta(seconds=8)).isoformat(),
                        "source": "safe_point_interval",
                    },
                ],
            },
        )
        store.set_job_status(later.job_id, JobStatus.READY, reason="test", hold=False)
        supervisor = _DrainSupervisor(["stack-a", "stack-b", "stack-c"])
        service = SchedulerService(settings, store=store, supervisor=supervisor)
        trial = ColocationTrialState(
            trial_id="measured-rejection",
            candidate_job_id=candidate.job_id,
            preexisting_job_ids=("stack-a", "stack-b"),
            started_at=started.isoformat(),
            start_epoch=0,
            target_epoch=2,
            backend_name="cuda_process",
            profile_key="measured-stack-key",
            candidate_solo_epoch_seconds=1.0,
            pretrial_epoch_seconds={"stack-a": 1.0, "stack-b": 1.0},
            member_start_epochs={"stack-a": 0, "stack-b": 0, "stack-c": 0},
            evidence_deadline_at=(started + timedelta(minutes=5)).isoformat(),
        )
        service._colocation_trial = trial
        service._evaluate_colocation_trial()
        assert supervisor.paused == ["stack-c"]
        assert service._colocation_stall is not None
        assert service._colocation_stall.preexisting_job_ids == ("stack-a", "stack-b")
        service._dispatch_pending_work()
        assert supervisor.dispatched == []

        restored = SchedulerService(settings, store=store, supervisor=supervisor)
        assert restored._colocation_stall is not None
        supervisor.active_ids = ["stack-b"]
        restored._refresh_colocation_stall()
        assert restored._colocation_stall is None


def test_trace_trial_waits_for_two_epochs_from_slowest_existing_member() -> None:
    problem = TraceProblem(
        jobs=(
            TraceJob(
                "slow-active",
                0.0,
                0,
                (TraceBatchOption(4, 100.0, 300.0),),
                planned_epochs=30,
            ),
            TraceJob(
                "fast-candidate",
                0.1,
                0,
                (TraceBatchOption(4, 100.0, 30.0),),
                planned_epochs=30,
            ),
        ),
        memory_budget_mb=500.0,
        parallel_cap=2,
        default_slowdown=1.0,
        colocation_trial_epochs=2,
    )
    result = simulate_recursive_time_aware(problem)
    assert result.colocation_trial_epochs == pytest.approx(20.0)


def test_trace_simulator_preserves_two_rejected_epochs_and_models_stall() -> None:
    option = (TraceBatchOption(4, 100.0, 10.0),)
    problem = TraceProblem(
        jobs=(
            TraceJob("a", 0.0, 0, option, planned_epochs=10),
            TraceJob("b", 0.0, 0, option, planned_epochs=10),
            TraceJob("c", 0.0, 0, option, planned_epochs=10),
        ),
        memory_budget_mb=500.0,
        parallel_cap=2,
        slowdown_by_pair={("a", "b"): 3.0, ("a", "c"): 1.0, ("b", "c"): 1.0},
    )
    result = simulate_recursive_time_aware(problem)
    assert result.slowdown_rejections == 1
    assert result.admission_stalls == 1
    assert result.rejected_trial_epochs_preserved == pytest.approx(2.0)
    assert result.colocation_trial_epochs >= 2.0


def test_ml_epoch_prediction_requires_canonical_declared_target(monkeypatch: pytest.MonkeyPatch) -> None:
    unsupported = object.__new__(MLVramPredictor)
    unsupported._target_names = ("train_mem", "train_time")
    with pytest.raises(JobPredictionError, match="does not expose train_epoch_ms"):
        unsupported.predict_seconds_per_epoch_options(
            _job("unsupported-ml", "unsupported-ml-sig"), [4]
        )

    canonical = object.__new__(MLVramPredictor)
    canonical._target_names = ("train_mem", "train_epoch_ms")
    canonical.available = True
    canonical._job_epoch_seconds = {}
    canonical._prediction_key = lambda job, batch, spec: f"{job.job_id}:{batch}"  # type: ignore[method-assign]
    canonical._convert_many = lambda specs: [object() for _ in specs]  # type: ignore[method-assign]

    class _Scalar:
        def __init__(self, value: float):
            self.value = value

        def item(self) -> float:
            return self.value

    class _Runtime:
        def predict(self, encoded: object) -> list[_Scalar]:
            return [_Scalar(100.0), _Scalar(2500.0)]

    canonical._runtime = _Runtime()
    monkeypatch.setattr(
        "localml_scheduler.prediction.ml_predictor.model_specification_for_job",
        lambda job, batch_size: object(),
    )
    assert canonical.predict_seconds_per_epoch_options(_job("canonical-ml", "canonical-ml-sig"), [4]) == {4: 2.5}


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
            gpu_scheduler={
                "mode": "parallel_time_aware",
                "packing_backend": "cuda_process",
                "exclusive_fallback_enabled": True,
                "cuda_process": {"enabled": False},
            },
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
