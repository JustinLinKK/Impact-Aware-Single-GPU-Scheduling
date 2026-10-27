"""Admission remains bounded without turning startup latency into rejection."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import time
from unittest.mock import Mock

import pytest

from localml_scheduler.config.models import ColocationSettings
from localml_scheduler.domain import JobStatus, SafePointType
from localml_scheduler.execution.executor import WorkerProcessHandle
from localml_scheduler.scheduler.service import SchedulerService
from localml_scheduler.scheduler.supervisor import ManagedWorker, PlacementGroupHandle, WorkerSupervisor
from localml_scheduler.tests.test_cold_start_trials import setup
from localml_scheduler.tests.test_mlevolve_runner import _build_context
from localml_scheduler.tests.test_prediction_first_workflow import virtual_predictor


def pending_trial(tmp_path, members=("a", "b")):
    settings, store, planner, jobs = setup(tmp_path)
    planner.estimator.ml_predictor = virtual_predictor()
    for job in jobs:
        store.update_job(job.job_id, status=JobStatus.RUNNING, metadata_updates={
            "trial_runner_ready": True, "runtime_steps_per_epoch": 512,
            "runtime_global_step": 510,
        })
    supervisor = Mock()
    supervisor.active_job_ids.return_value = list(members)
    supervisor.request_pause.return_value = True
    service = SchedulerService(settings, store=store, supervisor=supervisor)
    service.planner.estimator.ml_predictor = planner.estimator.ml_predictor
    active = [store.get_job(name) for name in members[:-1]]
    candidate = store.get_job(members[-1])
    plan = planner.choose_plan([candidate], active_jobs=active, backend_available={"cuda_process": True})
    trial = service._prepare_colocation_trial(plan, candidate)
    return settings, store, planner, service, trial


def acknowledge(store, name):
    token = store.get_job(name).metadata["trial_step_command"]["token"]
    store.update_job(name, metadata_updates={"trial_yield_ack": token, "trial_runner_ready": True})


def publish_window(store, name, seconds=0.3):
    token = store.get_job(name).metadata["trial_step_command"]["token"]
    store.update_job(name, metadata_updates={"trial_step_window": {
        "token": token, "complete": True, "halves": [seconds, seconds],
    }})


@pytest.mark.parametrize("seconds_per_step", [0.3, 1.25])
def test_worker_collects_complete_slow_window_across_resumed_epoch(tmp_path, monkeypatch, seconds_per_step):
    settings, _, _, jobs = setup(tmp_path)
    clock = SimpleNamespace(now=1000.0)
    fake_time = SimpleNamespace(time=lambda: clock.now, monotonic=lambda: clock.now,
                                perf_counter=lambda: clock.now)
    monkeypatch.setattr("localml_scheduler.execution.trial_control.time", fake_time)
    job = jobs[0]
    job.metadata["trial_step_command"] = {
        "owner": "trial", "token": "fresh", "kind": "measure_overlap", "action": "measure",
        "candidate": True, "target_epoch": 1, "target_step": 1022, "expires_at": 2000,
    }
    context = _build_context(settings, job)
    for step in range(510, 521):
        clock.now += seconds_per_step
        context.control_hook._cooperative_step(SafePointType.STEP, epoch=step // 512,
                                               global_step=step, steps_per_epoch=512, metrics={})
        if step < 520:
            assert not context.store.get_job("a").metadata.get("trial_step_window")
    window = context.store.get_job("a").metadata["trial_step_window"]
    assert window["complete"] and window["steps"] == 8
    assert window["elapsed_seconds"] == pytest.approx(8 * seconds_per_step)
    assert window["halves"] == pytest.approx([seconds_per_step] * 2)
    assert context.store.get_job("a").metadata["last_completed_epoch"] == 1


def test_stale_resume_ready_and_progress_cannot_start_measurement_deadline(tmp_path):
    from localml_scheduler.tests.test_completion_trials import fixture, window
    settings, store, _, service, trial = fixture(tmp_path)
    trial.started_at = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    store.update_job("b", metadata_updates={"trial_yield_ack": "old-launch-token"})
    service._evaluate_colocation_trial()
    assert trial.phase == "prepare"  # ready=True from old worker is insufficient.
    acknowledge(store, "b")
    service._evaluate_colocation_trial()
    assert trial.phase == "pre_add"
    trial.cold_start["phase_started_at"] = time.time() - 90
    store.update_job("a", metadata_updates={"trial_measurement_progress": {
        "token": "old-window", "started_at": time.time() - 90, "seconds_per_step": 0.3,
    }})
    service._evaluate_colocation_trial()
    assert service._colocation_trial is trial
    token = store.get_job("a").metadata["trial_step_command"]["token"]
    store.update_job("a", metadata_updates={"trial_measurement_progress": {
        "token": token, "started_at": time.time() - 3, "seconds_per_step": 0.3,
    }})
    service._evaluate_colocation_trial()
    assert service._colocation_trial is trial  # More than the old two seconds.
    window(store, "a", 0.3)
    service._evaluate_colocation_trial()
    assert trial.phase == "completion_initial_barrier"
    assert service._colocation_trial is trial


def test_hung_first_step_has_absolute_bound_even_without_progress(tmp_path):
    settings, store, _, service, trial = pending_trial(tmp_path)
    acknowledge(store, "b")
    service._evaluate_colocation_trial()
    trial.started_at = (datetime.now(timezone.utc) - timedelta(
        seconds=settings.gpu_scheduler.colocation.cold_start_trial_timeout_seconds + 1)).isoformat()
    service._evaluate_colocation_trial()
    assert service._colocation_trial is None
    result = store.get_job("b").metadata["colocation_trial"]
    assert result["decision"] == "inconclusive" and "absolute" in result["reason"]
    assert all(not store.get_job(name).metadata.get("trial_step_command") for name in ("a", "b"))


def test_measurement_timeout_uses_measured_seconds_not_relative_prediction(tmp_path):
    _, store, _, service, trial = pending_trial(tmp_path)
    acknowledge(store, "b")
    service._evaluate_colocation_trial()
    token = store.get_job("a").metadata["trial_step_command"]["token"]
    trial.cold_start["predicted_work_profiles"]["a"]["step_cost"] = 1e12
    store.update_job("a", metadata_updates={"trial_measurement_progress": {
        "token": token, "started_at": time.time() - 30, "seconds_per_step": 0.3,
    }})
    service._evaluate_colocation_trial()
    assert store.get_job("b").metadata["colocation_trial"]["reason"] == "pre-add reference timed out"


def test_membership_change_reprepares_and_tracks_both_admission_keys(tmp_path):
    _, store, planner, service, trial = pending_trial(tmp_path, ("a", "b", "c"))
    original_key = trial.profile_key
    store.update_job("b", status=JobStatus.PAUSING)
    assert planner.choose_plan([store.get_job("c")], active_jobs=[store.get_job("a"), store.get_job("b")],
                               backend_available={"cuda_process": True}) is None
    service._evaluate_colocation_trial()
    assert trial.phase == "prepare"
    acknowledge(store, "c")
    service.supervisor.active_job_ids.return_value = ["a", "c"]
    service._evaluate_colocation_trial()
    assert trial.profile_key != original_key and trial.phase == "prepare"
    service._evaluate_colocation_trial()
    assert trial.phase == "prepare"  # previous group's acknowledgement is stale.
    acknowledge(store, "c")
    service._evaluate_colocation_trial()
    assert trial.phase == "pre_add"
    service._finish_cold_start_trial(trial, "rejected", "regression")
    blocked = store.get_job("c").metadata["cold_start_blocked_groups"]
    assert original_key in blocked and trial.profile_key in blocked


def test_slow_member_teardown_renews_healthy_candidate_lease(tmp_path):
    _, store, _, service, trial = pending_trial(tmp_path)
    command = store.get_job("b").metadata["trial_step_command"]
    token = command["token"]
    store.update_job("b", metadata_updates={"trial_step_command": {
        **command, "expires_at": time.time() + 0.01,
    }})
    store.update_job("a", status=JobStatus.PAUSING)
    service._evaluate_colocation_trial()
    refreshed = store.get_job("b").metadata["trial_step_command"]
    assert refreshed["token"] == token
    assert refreshed["expires_at"] > time.time() + 4
    assert service._colocation_trial is trial and trial.phase == "prepare"
    assert not store.get_job("b").metadata.get("cold_start_failed_attempts")


def test_inconclusive_retries_are_finite_across_groups_then_run_solo(tmp_path):
    settings, store, planner, service, trial = pending_trial(tmp_path)
    available = {"cuda_process": True}
    original_key = trial.profile_key
    for attempt in range(settings.gpu_scheduler.colocation.cold_start_max_attempts):
        if attempt:
            active = [store.get_job("a" if attempt == 1 else "c")]
            store.update_job("b", status=JobStatus.PAUSED, metadata_updates={"cold_start_retry_after": 0})
            history = store.get_job("b").metadata["cold_start_group_attempts"]
            for value in history.values():
                value["retry_after"] = 0
            store.update_job("b", metadata_updates={"cold_start_group_attempts": history})
            plan = planner.choose_plan([store.get_job("b")], active_jobs=active, backend_available=available)
            assert plan is not None
            trial = service._prepare_colocation_trial(plan, store.get_job("b"))
        service._finish_cold_start_trial(trial, "inconclusive", "slow-window fixture")
        current = store.get_job("b")
        assert current.metadata["cold_start_failed_attempts"] == attempt + 1
        assert planner.choose_plan([current], active_jobs=[store.get_job("a")], backend_available=available) is None
    current = store.get_job("b")
    assert original_key in current.metadata["cold_start_blocked_groups"]
    assert current.metadata["cold_start_solo_required"]
    solo = planner.choose_plan([store.get_job("a"), current], backend_available=available)
    assert solo.job_ids == ("a",) and not solo.trial_metadata["requires_live_trial"]
    solo = planner.choose_plan([current], backend_available=available)
    assert solo.job_ids == ("b",)
    current.status = JobStatus.RUNNING
    assert planner.choose_plan([store.get_job("a")], active_jobs=[current], backend_available=available).job_ids == ("a",)


@pytest.mark.parametrize("monitor_via_store", [False, True])
def test_paused_parent_is_retained_until_reaped_and_blocks_new_admission(tmp_path, monitor_via_store):
    settings, store, _, jobs = setup(tmp_path)
    supervisor = WorkerSupervisor(settings, store=store, backends={
        "cuda_process": Mock(available=Mock(return_value=True)),
    })
    process = Mock()
    process.poll.return_value = None
    handle = WorkerProcessHandle("a", process, tmp_path / "stdout", tmp_path / "stderr", monitor_via_store)
    supervisor._groups["group"] = PlacementGroupHandle(
        "group", "stack_anchor", "cuda_process", workers={"a": ManagedWorker("a", handle, "solo")})
    store.update_job("a", status=JobStatus.PAUSED)
    assert supervisor.poll() == []
    assert supervisor.has_live_process("a")
    assert "a" in supervisor.active_groups()["group"].workers
    assert not supervisor.placement_for("concurrent_group", "cuda_process", ["b"]).can_run
    assert not supervisor.placement_for("stack_anchor", "cuda_process", ["a"]).can_run
    process.poll.return_value = 0
    assert supervisor.poll()[0].alive is False
    assert not supervisor.has_live_process("a")
    assert supervisor.placement_for("stack_anchor", "cuda_process", ["a"]).can_run


def test_cold_start_bounds_round_trip_and_reject_nonfinite_values():
    settings = ColocationSettings()
    assert ColocationSettings.from_dict(settings.to_dict()) == settings
    for value in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            ColocationSettings(cold_start_trial_timeout_seconds=value)
    for value in (.10, .50):
        settings = ColocationSettings(cold_start_max_incumbent_slowdown_fraction=value)
        assert ColocationSettings.from_dict(settings.to_dict()).cold_start_max_incumbent_slowdown_fraction == value
    for value in (-1, 1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            ColocationSettings(cold_start_max_incumbent_slowdown_fraction=value)
