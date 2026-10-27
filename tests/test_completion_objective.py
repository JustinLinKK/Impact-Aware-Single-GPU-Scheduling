"""Completion projections use only measured seconds and explicit update budgets."""
from types import SimpleNamespace

import pytest

from localml_scheduler.scheduler.completion_objective import (
    compare_update_completion, project_deferred_candidate, project_update_drain, remaining_optimizer_updates,
)


def job(epochs=2, max_steps=None):
    return SimpleNamespace(max_epochs=epochs, max_steps=max_steps, config=SimpleNamespace(max_epochs=epochs))


def test_partial_epoch_budget_is_exact_and_does_not_claim_validation_finished():
    assert remaining_optimizer_updates(job(), {"global_step": 1022, "steps_per_epoch": 512}) == 2
    assert remaining_optimizer_updates(job(), {"global_step": 1024, "steps_per_epoch": 512}) == 0
    assert remaining_optimizer_updates(job(), {"global_step": 510, "steps_per_epoch": 512}) == 514


@pytest.mark.parametrize("snapshot", [
    {}, {"global_step": 1}, {"global_step": 1025, "steps_per_epoch": 512},
    {"global_step": -1, "steps_per_epoch": 512}, {"global_step": 1.5, "steps_per_epoch": 512},
    {"global_step": True, "steps_per_epoch": 512}, {"global_step": 1, "steps_per_epoch": 0},
])
def test_unknown_or_inconsistent_update_budget_is_not_zero(snapshot):
    assert remaining_optimizer_updates(job(), snapshot) is None
    assert remaining_optimizer_updates(job(max_steps=100), {"global_step": 1, "steps_per_epoch": 512}) is None


def test_short_candidate_helps_completion_despite_slowing_long_incumbent():
    work = {"a": 100, "b": 10}
    rates = {("a",): {"a": 1}, ("b",): {"b": 1}, ("a", "b"): {"a": 1.05, "b": 1}}
    deferred = project_deferred_candidate(work, "b", rates)
    packed = project_update_drain(work, rates)
    assert deferred.seconds == 110
    assert packed.seconds == pytest.approx(100.47619047619048)
    assert packed.phases[0].completed_update_budgets == ("b",)
    assert packed.phases[1].member_ids == ("a",)


def test_missing_solo_or_subgroup_vector_requests_actual_membership():
    work = {"a": 100, "b": 10, "c": 30}
    rates = {("a", "b", "c"): {"a": 1.05, "b": 1, "c": 1}}
    packed = project_update_drain(work, rates)
    assert packed.seconds is None and packed.missing_membership == ("a", "c")
    rates[("a", "c")] = {"a": 1.01, "c": 1}
    packed = project_update_drain(work, rates)
    assert packed.seconds is None and packed.missing_membership == ("a",)
    rates[("a",)] = {"a": 1}
    assert project_update_drain(work, rates).seconds is not None
    deferred = project_deferred_candidate({"a": 100, "b": 10}, "b", rates)
    assert deferred.seconds is None and deferred.missing_membership == ("b",)


def test_removing_finished_update_budgets_still_needs_reduced_group_evidence():
    result = project_update_drain({"a": 0, "b": 2}, {("a", "b"): {"a": 1, "b": 1}})
    assert result.seconds is None and result.missing_membership == ("b",)
    assert project_update_drain({"a": 0}, {}).seconds == 0  # Update drain only.


@pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan"), "unknown"])
def test_invalid_seconds_cannot_be_used_as_rate_bounds(bad):
    result = project_update_drain({"a": 10}, {("a",): {"a": bad}})
    assert result.seconds is None and result.error


def test_parallel_equal_finish_requires_no_unobserved_tail():
    result = project_update_drain({"a": 20, "b": 10}, {("a", "b"): {"a": 1, "b": 2}})
    assert result.seconds == 20
    assert len(result.phases) == 1 and result.phases[0].completed_update_budgets == ("a", "b")


def test_actual_probe_wall_time_and_credited_progress_are_counted_once():
    rates = {("a",): {"a": [1, 1]}, ("b",): {"b": [1, 1]},
             ("a", "b"): {"a": [1.05, 1.05], "b": [1, 1]}}
    initial = {"a": 100, "b": 10}
    no_progress = compare_update_completion(initial, initial, "b", rates, 12)
    for half in no_progress["halves"]:
        assert half["initial_defer_seconds"] == 110
        assert half["measured_attempt_keep_seconds"] == pytest.approx(112.47619047619048)
        assert half["current_keep_seconds"] < half["current_defer_seconds"]
    productive = compare_update_completion(initial, {"a": 90, "b": 0}, "b", rates, 12)
    for half in productive["halves"]:
        assert half["measured_attempt_keep_seconds"] == 102  # 12 elapsed +90 left.
        assert half["initial_defer_seconds"] == 110
        assert half["current_defer_seconds"] == half["current_keep_seconds"] == 90


def test_two_halves_can_require_different_tail_subgroups():
    rates = {("a",): {"a": [1, 1]}, ("b",): {"b": [1, 1]}, ("c",): {"c": [1, 1]},
             ("a", "b"): {"a": [1, 1], "b": [1, 1]},
             ("a", "b", "c"): {"a": [1, 1], "b": [1, 1], "c": [.5, 2]}}
    work = {"a": 100, "b": 20, "c": 20}
    result = compare_update_completion(work, work, "c", rates, 0)
    assert result["error"] is None and result["missing_memberships"] == [("a", "c")]
    assert result["halves"][0]["current_keep_seconds"] is not None
    assert result["halves"][1]["current_keep_seconds"] is None


@pytest.mark.parametrize("remaining,elapsed", [({"a": 101, "b": 10}, 1), ({"a": 100}, 1),
                                             ({"a": 100, "b": 10}, -1), ({"a": 100, "b": 10}, float("nan"))])
def test_backward_progress_or_invalid_elapsed_time_stays_unknown(remaining, elapsed):
    result = compare_update_completion({"a": 100, "b": 10}, remaining, "b", {}, elapsed)
    assert result["error"]


def test_yield_ack_publishes_exact_progress_even_between_throttled_updates(tmp_path, monkeypatch):
    from localml_scheduler.domain import SafePointType
    from localml_scheduler.tests.test_cold_start_trials import setup
    from localml_scheduler.tests.test_mlevolve_runner import _build_context

    settings, _, _, jobs = setup(tmp_path)
    jobs[0].metadata.update({"runtime_global_step": 1000, "trial_step_command": {
        "owner": "comparison", "token": "snapshot-1", "action": "yield",
        "kind": "completion_snapshot", "expires_at": 2000,
    }})
    context = _build_context(settings, jobs[0])
    context.control_hook._step_trial_state = {
        "poll_at": 0, "publish_at": 2000, "published_epoch": 1,
        "command": {}, "token": None, "ready": True,
    }
    def release(_seconds):
        context.store.update_job("a", metadata_updates={"trial_step_command": {}})
    clock = SimpleNamespace(time=lambda: 1000.0, monotonic=lambda: 1000.0,
                            perf_counter=lambda: 1000.0, sleep=release)
    monkeypatch.setattr("localml_scheduler.execution.trial_control.time", clock)
    context.control_hook._cooperative_step(SafePointType.STEP, epoch=1, global_step=1022,
                                           steps_per_epoch=512, metrics={})
    metadata = context.store.get_job("a").metadata
    assert metadata["runtime_global_step"] == 1022
    expected = {
        "token": "snapshot-1", "owner": "comparison", "global_step": 1022,
        "steps_per_epoch": 512, "completed_epochs": 1, "observed_at": 1000.0,
    }
    assert {key: metadata["trial_yield_snapshot"][key] for key in expected} == expected


def test_fast_step_checkpoint_invalidates_window_without_waiting_for_command_poll(tmp_path, monkeypatch):
    from localml_scheduler.domain import SafePointType
    from localml_scheduler.tests.test_cold_start_trials import setup
    from localml_scheduler.tests.test_mlevolve_runner import _build_context

    settings, _, _, jobs = setup(tmp_path)
    jobs[0].metadata.update(runtime_completion_identity="shared-recipe", runtime_nonupdate_phase_counter=0)
    jobs[0].metadata["trial_step_command"] = {
        "owner": "trial", "token": "fast", "action": "measure", "kind": "completion_measure",
        "window_min_seconds": .0001, "expires_at": 2000,
    }
    context = _build_context(settings, jobs[0])
    clock = SimpleNamespace(now=1000.0)
    fake = SimpleNamespace(time=lambda: clock.now, monotonic=lambda: clock.now, perf_counter=lambda: clock.now)
    monkeypatch.setattr("localml_scheduler.execution.trial_control.time", fake)
    for step in range(17):
        clock.now += .0001
        if step == 4:
            context.store.update_job("a", metadata_updates={"runtime_nonupdate_phase_counter": 1})
        context.control_hook._cooperative_step(SafePointType.STEP, epoch=0, global_step=step,
                                               steps_per_epoch=100, metrics={})
        if step < 16:
            assert not context.store.get_job("a").metadata.get("trial_step_window")
    window = context.store.get_job("a").metadata["trial_step_window"]
    assert window["start_step"] == 8 and window["end_step"] == 16
    assert window["nonupdate_phase_counter"] == 1 and window["excludes_nonupdate_phases"]


def test_trial_update_limit_waits_for_epoch_phases_then_stops_before_more_updates(tmp_path, monkeypatch):
    from localml_scheduler.domain import SafePointType
    from localml_scheduler.tests.test_cold_start_trials import setup
    from localml_scheduler.tests.test_mlevolve_runner import _build_context
    settings, _, _, jobs = setup(tmp_path)
    jobs[0].metadata.update({"runtime_total_optimizer_updates": 36, "trial_step_command": {
        "owner": "trial", "token": "limit", "candidate": True, "action": "measure",
        "kind": "completion_provisional", "target_step": 12, "expires_at": 2000,
    }})
    context = _build_context(settings, jobs[0])
    yielded = []
    def release(_seconds):
        yielded.append(context.store.get_job("a").metadata["trial_yield_snapshot"]["global_step"])
        context.store.update_job("a", metadata_updates={"trial_step_command": {}})
    clock = SimpleNamespace(time=lambda: 1000., monotonic=lambda: 1000., perf_counter=lambda: 1000., sleep=release)
    monkeypatch.setattr("localml_scheduler.execution.trial_control.time", clock)
    for kind, completed in [(SafePointType.STEP, 0), (SafePointType.EPOCH, 1)]:
        context.control_hook._cooperative_step(kind, epoch=completed, global_step=12, steps_per_epoch=12, metrics={})
        assert not yielded
    context.control_hook._cooperative_step(SafePointType.MANUAL, epoch=1, global_step=12, steps_per_epoch=12, metrics={})
    assert yielded == [12]


def test_snapshot_declared_recipe_budget_must_match_job_budget():
    assert remaining_optimizer_updates(job(), {"global_step": 1, "steps_per_epoch": 512,
        "total_optimizer_updates": 1025, "planned_epochs": 2}) is None
    assert remaining_optimizer_updates(job(), {"global_step": 1, "steps_per_epoch": 512,
        "total_optimizer_updates": 1024, "planned_epochs": 3}) is None
