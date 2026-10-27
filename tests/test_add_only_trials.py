"""Production admission must preserve incumbents' optimizer progress."""
from datetime import datetime, timezone, timedelta

import pytest

from localml_scheduler.domain import JobStatus
from localml_scheduler.scheduler.service import SchedulerService
from localml_scheduler.tests.test_completion_trials import fixture, ack, window


def production(tmp_path, *, members=("a", "b")):
    result = fixture(tmp_path, phases=False, members=members, service_type=SchedulerService)
    settings, _, _, service, trial = result
    # Behavioral parity tests explicitly choose a tolerance; v2/v3 may have
    # different dataclass defaults without changing this controller contract.
    settings.gpu_scheduler.colocation.cold_start_max_incumbent_slowdown_fraction = .10
    original = service._cold_command
    def guarded(trial, job, action, kind, **kwargs):
        assert action != "yield" or job.job_id == trial.candidate_job_id
        return original(trial, job, action, kind, **kwargs)
    service._cold_command = guarded
    return result


def progress(store, trial, name, *, started=None, advance=2, token=None):
    job = store.get_job(name)
    meta = job.metadata
    store.update_job(name, metadata_updates={"runtime_global_step": meta["runtime_global_step"] + advance,
        "trial_measurement_progress": {
            "token": token or meta["trial_step_command"]["token"],
            "started_at": trial.cold_start["phase_started_at"] + .001 if started is None else started,
            "global_step": meta["runtime_global_step"] + advance, "seconds_per_step": 1,
        }})


def sample(store, trial, name, seconds=1, *, start=.1, end=1.1):
    window(store, name, seconds)
    measured = store.get_job(name).metadata["trial_step_window"]
    epoch = trial.cold_start["phase_started_at"]
    measured.update(started_at=epoch + start,
                    finished_at=datetime.fromtimestamp(epoch + end, timezone.utc).isoformat())
    store.update_job(name, metadata_updates={"trial_step_window": measured})


def warmup(store, service, trial):
    ack(store, [trial.candidate_job_id])
    service._evaluate_colocation_trial()
    assert trial.phase == "pre_add"
    for name in trial.preexisting_job_ids:
        sample(store, trial, name)
    service._evaluate_colocation_trial()
    assert trial.phase == "overlap_warmup"


def overlap(store, service, trial):
    warmup(store, service, trial)
    for name in [*trial.preexisting_job_ids, trial.candidate_job_id]:
        progress(store, trial, name)
    service._evaluate_colocation_trial()
    assert trial.phase == "measure_overlap"


def outcome(store, trial):
    return store.get_job(trial.candidate_job_id).metadata["colocation_trial"]


@pytest.mark.parametrize("halves,decision", [([1, 1], "accepted"), ([1.1, 1.1], "accepted"),
    ([1.10001, 1.10001], "rejected"), ([1.09, 1.11], "inconclusive"), ([1.11, 1.09], "inconclusive")])
def test_both_halves_enforce_incremental_ten_percent_limit(tmp_path, halves, decision):
    _, store, _, service, trial = production(tmp_path)
    overlap(store, service, trial)
    sample(store, trial, "a", halves)
    sample(store, trial, "b", 100)  # Candidate solo cost is deliberately unknown.
    service._evaluate_colocation_trial()
    result = outcome(store, trial)
    assert result["decision"] == decision
    assert result["result"]["admission_policy"] == "add_only_bounded_interference_v1"
    assert result["result"]["candidate_solo_seconds"] is None
    assert result["result"]["completion_gain"] is None
    assert result["result"]["common_measurement_interval"]["finished_at"] > result["result"]["common_measurement_interval"]["started_at"]
    assert all(call.args[0] == "b" for call in service.supervisor.request_pause.call_args_list)
    assert store.get_job("a").status == JobStatus.RUNNING
    assert store.get_job("c").status == JobStatus.READY
    assert not store.get_job("c").metadata.get("trial_step_command")


def test_every_incumbent_must_pass_and_predictor_weights_do_not_buy_admission(tmp_path):
    _, store, _, service, trial = production(tmp_path, members=("a", "b", "c"))
    overlap(store, service, trial)
    trial.cold_start["predicted_work_profiles"]["a"]["step_cost"] = 1e12
    trial.cold_start["predicted_work_profiles"]["b"]["step_cost"] = 1e-12
    for name, seconds in [("a", .5), ("b", 1.2), ("c", .001)]:
        sample(store, trial, name, seconds)
    service._evaluate_colocation_trial()
    assert outcome(store, trial)["decision"] == "rejected"
    assert all(call.args[0] == "c" for call in service.supervisor.request_pause.call_args_list)


@pytest.mark.parametrize("halves,decision", [([1.5, 1.5], "accepted"),
    ([1.50001, 1.50001], "rejected"), ([1.49, 1.51], "inconclusive")])
def test_separate_fifty_percent_setting_has_exact_independent_half_boundary(tmp_path, halves, decision):
    settings, store, _, service, trial = production(tmp_path)
    assert settings.gpu_scheduler.colocation.cold_start_max_incumbent_slowdown_fraction == .10
    settings.gpu_scheduler.colocation.cold_start_max_incumbent_slowdown_fraction = .50
    overlap(store, service, trial)
    sample(store, trial, "a", halves)
    sample(store, trial, "b")
    service._evaluate_colocation_trial()
    result = outcome(store, trial)
    assert result["decision"] == decision
    assert result["result"]["max_incumbent_slowdown_fraction"] == .50


def test_slow_first_batch_requires_fresh_productive_warmup_from_every_member(tmp_path):
    _, store, _, service, trial = production(tmp_path)
    warmup(store, service, trial)
    sample(store, trial, "a")
    progress(store, trial, "a")
    progress(store, trial, "b", token="stale-token", advance=0)
    service._evaluate_colocation_trial()
    assert trial.phase == "overlap_warmup" and service._colocation_trial is trial
    progress(store, trial, "b")
    service._evaluate_colocation_trial()
    assert trial.phase == "measure_overlap"
    assert service._cold_window(store.get_job("a")) is None  # Warmup sample never accepted.
    assert not service.supervisor.request_pause.called


def test_disjoint_fast_window_is_refreshed_while_slow_member_is_still_measuring(tmp_path):
    _, store, _, service, trial = production(tmp_path)
    overlap(store, service, trial)
    old_token = store.get_job("a").metadata["trial_step_command"]["token"]
    sample(store, trial, "a", start=.1, end=.6)
    progress(store, trial, "b", started=trial.cold_start["phase_started_at"] + 2)
    service._evaluate_colocation_trial()
    assert trial.phase == "measure_overlap" and service._colocation_trial is trial
    assert store.get_job("a").metadata["trial_step_command"]["token"] != old_token
    assert service._cold_window(store.get_job("a")) is None
    sample(store, trial, "a", start=2.1, end=2.6)
    sample(store, trial, "b", start=2, end=10)
    service._evaluate_colocation_trial()
    result = outcome(store, trial)
    assert result["decision"] == "accepted"
    assert result["result"]["overlap_window_refreshes"] == 1
    assert not service.supervisor.request_pause.called


def test_all_completed_disjoint_samples_end_inconclusive_instead_of_chasing_history(tmp_path):
    _, store, _, service, trial = production(tmp_path)
    overlap(store, service, trial)
    sample(store, trial, "a", start=.1, end=.6)
    sample(store, trial, "b", start=2, end=10)
    service._evaluate_colocation_trial()
    assert outcome(store, trial)["decision"] == "inconclusive"
    assert "do not overlap" in outcome(store, trial)["reason"]


@pytest.mark.parametrize("invalid", ["missing_interval", "nonupdate_phase", "stale_interval"])
def test_invalid_or_contaminated_overlap_cannot_accept(tmp_path, invalid):
    _, store, _, service, trial = production(tmp_path)
    overlap(store, service, trial)
    sample(store, trial, "a")
    sample(store, trial, "b")
    measured = store.get_job("a").metadata["trial_step_window"]
    if invalid == "missing_interval": measured.pop("started_at")
    elif invalid == "stale_interval": measured["started_at"] = 0
    else: measured["excludes_nonupdate_phases"] = False
    store.update_job("a", metadata_updates={"trial_step_window": measured})
    service._evaluate_colocation_trial()
    assert outcome(store, trial)["decision"] != "accepted"


@pytest.mark.parametrize("phase", ["completion_decision_barrier", "reference_barrier"])
def test_persisted_research_yields_are_rejected_before_any_command_renewal(tmp_path, phase):
    _, store, _, service, trial = production(tmp_path)
    trial.phase = phase
    store.update_job("a", metadata_updates={"trial_step_command": {
        "owner": trial.trial_id, "action": "yield", "kind": phase, "token": "old"}})
    service._evaluate_colocation_trial()
    assert outcome(store, trial)["decision"] == "inconclusive"
    assert store.get_job("a").metadata["trial_step_command"] == {}
    assert store.get_job("a").status == JobStatus.RUNNING


def test_missing_incumbent_command_does_not_default_to_yield(tmp_path):
    _, store, _, service, trial = production(tmp_path)
    overlap(store, service, trial)
    store.update_job("a", metadata_updates={"trial_step_command": {}})
    service._evaluate_colocation_trial()
    assert outcome(store, trial)["decision"] == "inconclusive"
    assert "unsafe admission command" in outcome(store, trial)["reason"]


def test_persisted_prepare_cannot_renew_incumbent_yield_while_another_worker_drains(tmp_path):
    _, store, _, service, trial = production(tmp_path, members=("a", "b", "c"))
    store.update_job("a", metadata_updates={"trial_step_command": {
        "owner": trial.trial_id, "action": "yield", "kind": "completion_decision_barrier", "token": "old"}})
    store.update_job("b", status=JobStatus.PAUSING)
    service._evaluate_colocation_trial()
    assert outcome(store, trial)["decision"] == "inconclusive"
    assert store.get_job("a").metadata["trial_step_command"] == {}
    assert store.get_job("a").status == JobStatus.RUNNING


@pytest.mark.parametrize("start,steps,epochs,target", [(7, 10, 3, 27), (29, 10, 3, 30), (510, 512, 3, 1534)])
def test_measurement_budget_uses_actual_steps_and_partial_epoch(tmp_path, start, steps, epochs, target):
    _, store, _, service, trial = production(tmp_path)
    job = store.get_job("b")
    store.update_job("b", metadata_updates={"runtime_global_step": start, "runtime_steps_per_epoch": steps,
        "runtime_total_optimizer_updates": epochs * steps, "runtime_planned_epochs": epochs})
    warmup(store, service, trial)
    assert trial.cold_start["target_step"] == target
    assert store.get_job("b").metadata["trial_step_command"]["target_step"] == target


def test_candidate_budget_and_absolute_deadline_bound_alignment_attempts(tmp_path):
    _, store, _, service, trial = production(tmp_path)
    overlap(store, service, trial)
    store.update_job("b", metadata_updates={"runtime_global_step": trial.cold_start["target_step"]})
    sample(store, trial, "a")
    sample(store, trial, "b")
    service._evaluate_colocation_trial()
    assert outcome(store, trial)["decision"] == "inconclusive"
    assert "budget exhausted" in outcome(store, trial)["reason"]


def test_slow_first_progress_cannot_extend_absolute_trial_deadline(tmp_path):
    _, store, _, service, trial = production(tmp_path)
    warmup(store, service, trial)
    trial.started_at = (datetime.now(timezone.utc) - timedelta(seconds=601)).isoformat()
    service._evaluate_colocation_trial()
    assert outcome(store, trial)["decision"] == "inconclusive"
    assert "absolute cold-start trial deadline" in outcome(store, trial)["reason"]


def test_trial_deadline_does_not_clear_an_existing_user_pause_hold(tmp_path):
    _, store, _, service, trial = production(tmp_path)
    warmup(store, service, trial)
    store.update_job("b", status=JobStatus.PAUSING, hold=True)
    trial.started_at = (datetime.now(timezone.utc) - timedelta(seconds=601)).isoformat()
    service._evaluate_colocation_trial()
    assert outcome(store, trial)["decision"] == "inconclusive"
    assert store.get_job("b").hold is True
    assert not service.supervisor.request_pause.called


def test_original_submission_age_survives_failed_trial_and_retry_metadata(tmp_path):
    _, store, _, service, trial = production(tmp_path)
    original = store.get_job("b").submitted_at
    overlap(store, service, trial)
    sample(store, trial, "a", 1.5)
    sample(store, trial, "b")
    service._evaluate_colocation_trial()
    assert store.get_job("b").submitted_at == original
    assert store.get_job("b").metadata["cold_start_failed_attempts"] == 1


@pytest.mark.parametrize("field,value", [("runtime_completion_identity", None), ("runtime_completion_identity", ""),
    ("runtime_nonupdate_phase_counter", None), ("runtime_nonupdate_phase_counter", -1),
    ("runtime_nonupdate_phase_counter", True)])
def test_missing_shared_runner_proof_cannot_claim_phase_clean_admission(tmp_path, field, value):
    _, store, _, service, trial = production(tmp_path)
    ack(store, ["b"])
    service._evaluate_colocation_trial()
    sample(store, trial, "a")
    store.update_job("a", metadata_updates={field: value})
    service._evaluate_colocation_trial()
    assert outcome(store, trial)["decision"] == "inconclusive"
    assert "proof is missing" in outcome(store, trial)["reason"]
    assert store.get_job("a").status == JobStatus.RUNNING


def test_lost_shared_runner_identity_invalidates_live_overlap(tmp_path):
    _, store, _, service, trial = production(tmp_path)
    overlap(store, service, trial)
    for name in ("a", "b"):
        sample(store, trial, name)
    store.update_job("b", metadata_updates={"runtime_completion_identity": "replacement-source"})
    service._evaluate_colocation_trial()
    assert outcome(store, trial)["decision"] == "inconclusive"
    assert "identity changed" in outcome(store, trial)["reason"]


@pytest.mark.parametrize("proof", [{}, {"runtime_completion_identity": "source"},
    {"runtime_nonupdate_phase_counter": 0},
    {"runtime_completion_identity": "source", "runtime_nonupdate_phase_counter": 0}])
def test_producer_only_marks_windows_phase_clean_with_explicit_shared_runner_proof(tmp_path, monkeypatch, proof):
    from types import SimpleNamespace
    from localml_scheduler.domain import SafePointType
    from localml_scheduler.tests.test_cold_start_trials import setup
    from localml_scheduler.tests.test_mlevolve_runner import _build_context
    settings, _, _, jobs = setup(tmp_path)
    jobs[0].metadata.update(proof)
    jobs[0].metadata["trial_step_command"] = {
        "owner": "trial", "token": "measure", "action": "measure", "kind": "measure_overlap",
        "window_min_seconds": .001, "expires_at": 2000,
    }
    context = _build_context(settings, jobs[0])
    clock = SimpleNamespace(now=1000.)
    monkeypatch.setattr("localml_scheduler.execution.trial_control.time", SimpleNamespace(
        time=lambda: clock.now, monotonic=lambda: clock.now, perf_counter=lambda: clock.now))
    for step in range(11):
        clock.now += .1
        context.control_hook._cooperative_step(SafePointType.STEP, epoch=0, global_step=step,
                                               steps_per_epoch=100, metrics={})
    measured = context.store.get_job("a").metadata["trial_step_window"]
    assert measured["complete"]
    assert measured["excludes_nonupdate_phases"] == (len(proof) == 2)


def test_retiring_member_rebaselines_survivor_without_interrupting_it(tmp_path):
    _, store, _, service, trial = production(tmp_path, members=("a", "b", "c"))
    overlap(store, service, trial)
    store.update_job("a", status=JobStatus.COMPLETED)
    service.supervisor.active_job_ids.return_value = ["b", "c"]
    service._evaluate_colocation_trial()
    assert trial.phase == "prepare"
    assert trial.preexisting_job_ids == ("b",)
    assert "add_only" not in trial.cold_start
    assert store.get_job("b").metadata["trial_step_command"] == {}
    assert not service.supervisor.request_pause.called
