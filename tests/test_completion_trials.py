"""State-machine regressions for measured completion admission, without GPUs."""
from copy import deepcopy
from datetime import datetime, timezone
import time
from unittest.mock import Mock

import pytest

from localml_scheduler.domain import JobStatus
from localml_scheduler.scheduler.completion_objective import observed_phase_reserve
from localml_scheduler.scheduler.service import SchedulerService
from localml_scheduler.scheduler.completion_trials import CompletionTrialMixin
from localml_scheduler.tests.test_cold_start_trials import setup
from localml_scheduler.tests.test_prediction_first_workflow import virtual_predictor


def phase_history(store, name, *, validation=.05, checkpoint=.01):
    identity = store.get_job(name).metadata["runtime_completion_identity"]
    return [{"phase": phase, "seconds": seconds, "identity": identity,
             "hardware_key": store.hardware_key(), "epoch": 1, "global_step": 100,
             "finished_at": datetime.now(timezone.utc).isoformat(), "source": source}
            for phase, seconds, source in [("validation", validation, "shared_training.validation"),
                                           ("checkpoint", checkpoint, "checkpoint_manager.save_checkpoint")]]


class ResearchCompletionService(CompletionTrialMixin, SchedulerService):
    pass


def fixture(tmp_path, *, phases=True, members=("a", "b"), service_type=ResearchCompletionService):
    settings, store, planner, jobs = setup(tmp_path)
    planner.estimator.ml_predictor = virtual_predictor()
    for job in jobs:
        store.update_job(job.job_id, status=JobStatus.RUNNING if job.job_id in members else JobStatus.READY,
                         metadata_updates={"runtime_completion_identity": "runtime-" + job.job_id,
            "runtime_nonupdate_phase_counter": 0,
            "runtime_global_step": 0, "last_completed_epoch": 0, "runtime_steps_per_epoch": 100,
            "runtime_total_optimizer_updates": 300, "runtime_planned_epochs": 3,
            "runtime_checkpoint_completed_step": 0, "runtime_checkpoint_completed_epoch": 0,
            "runtime_final_checkpoint_complete": False, "runtime_finalization_complete": False,
            "trial_runner_ready": True})
        if phases:
            store.update_job(job.job_id, metadata_updates={"runtime_phase_timing_history": phase_history(store, job.job_id)})
    supervisor = Mock()
    supervisor.active_job_ids.return_value = list(members)
    supervisor.request_pause.return_value = True
    service = service_type(settings, store=store, supervisor=supervisor)
    service.planner = planner
    candidate = store.get_job(members[-1])
    plan = planner.choose_plan([candidate], active_jobs=[store.get_job(name) for name in members[:-1]],
                               backend_available={"cuda_process": True})
    trial = service._prepare_colocation_trial(plan, candidate)
    return settings, store, planner, service, trial


def ack(store, names):
    for name in names:
        job = store.get_job(name)
        meta = job.metadata
        command = meta["trial_step_command"]
        store.update_job(name, metadata_updates={"trial_yield_ack": command["token"], "trial_yield_snapshot": {
            "token": command["token"], "owner": command["owner"], "observed_at": time.time(),
            "global_step": meta["runtime_global_step"], "steps_per_epoch": meta["runtime_steps_per_epoch"],
            "completed_epochs": meta["last_completed_epoch"], "completion_identity": meta["runtime_completion_identity"],
            "planned_epochs": meta["runtime_planned_epochs"], "total_optimizer_updates": meta["runtime_total_optimizer_updates"],
            "checkpoint_completed_step": meta["runtime_checkpoint_completed_step"],
            "checkpoint_completed_epoch": meta["runtime_checkpoint_completed_epoch"],
            "final_checkpoint_complete": meta["runtime_final_checkpoint_complete"],
            "finalization_complete": meta["runtime_finalization_complete"],
        }})


def window(store, name, seconds=1, *, steps=10):
    meta = store.get_job(name).metadata
    start = meta["runtime_global_step"]
    store.update_job(name, metadata_updates={"runtime_global_step": start + steps,
        "trial_step_window": {"token": meta["trial_step_command"]["token"], "complete": True,
        "halves": list(seconds) if isinstance(seconds, (list, tuple)) else [seconds, seconds],
        "start_step": start + 2, "end_step": start + steps, "steps": steps - 2,
        "excludes_nonupdate_phases": True, "nonupdate_phase_counter": 0}})


def start_measurements(store, service, trial, members=("a", "b")):
    ack(store, [members[-1]])
    service._evaluate_colocation_trial()
    assert trial.phase == "pre_add"
    for name in members[:-1]:
        window(store, name)
    service._evaluate_colocation_trial()
    assert trial.phase == "completion_initial_barrier"
    ack(store, members)
    service._evaluate_colocation_trial()
    assert trial.phase == "completion_measurement_barrier"


def measure_group(store, service, trial, *, seconds=1.1):
    members = [*trial.preexisting_job_ids, trial.candidate_job_id]
    ack(store, members)
    service._evaluate_colocation_trial()
    assert trial.phase == "completion_measure"
    for name in trial.cold_start["completion"]["measuring"]:
        window(store, name, seconds)
    service._evaluate_colocation_trial()


def initial_pair_evidence(store, service, trial):
    start_measurements(store, service, trial)
    assert trial.cold_start["completion"]["measuring"] == ["b"]
    measure_group(store, service, trial, seconds=1)
    assert trial.cold_start["completion"]["measuring"] == ["a", "b"]
    measure_group(store, service, trial)
    assert trial.phase == "completion_decision_barrier"


@pytest.mark.parametrize("weights", [(1, 1), (1e-9, 1e9), (1e9, 1e-9)])
def test_predictor_cardinal_weights_cannot_change_completion_acceptance(tmp_path, weights):
    _, store, _, service, trial = fixture(tmp_path)
    trial.cold_start["predicted_work_profiles"]["a"]["step_cost"] *= weights[0]
    trial.cold_start["predicted_work_profiles"]["b"]["step_cost"] *= weights[1]
    initial_pair_evidence(store, service, trial)
    ack(store, ["a", "b"])
    service._evaluate_colocation_trial()
    outcome = store.get_job("b").metadata["colocation_trial"]
    assert outcome["decision"] == "accepted"
    result = outcome["result"]
    assert result["objective"] == "modeled_completion_proxy_v1"
    assert result["predictor_role"] == "ordering and VRAM guard only"
    assert result["unknown_finalization_job_ids"] == ["a", "b"]
    assert result["remaining_updates"] == {"a": 280, "b": 280}
    assert store.get_job("c").status == JobStatus.READY
    assert not store.get_job("c").metadata.get("trial_step_command")


def test_missing_phase_evidence_runs_bounded_productive_provisional_overlap(tmp_path):
    settings, store, _, service, trial = fixture(tmp_path, phases=False)
    initial_pair_evidence(store, service, trial)
    ack(store, ["a", "b"])
    service._evaluate_colocation_trial()
    assert trial.phase == "completion_provisional"
    assert settings.gpu_scheduler.colocation.cold_start_trial_timeout_seconds == 600
    assert trial.cold_start["target_step"] == 200
    for name in ["a", "b"]:
        store.update_job(name, metadata_updates={"runtime_phase_timing_history": phase_history(store, name),
            "runtime_global_step": 100, "last_completed_epoch": 1,
            "runtime_checkpoint_completed_epoch": 1, "runtime_checkpoint_completed_step": 100})
    service._evaluate_colocation_trial()
    assert trial.phase == "completion_decision_barrier"
    ack(store, ["a", "b"])
    service._evaluate_colocation_trial()
    result = store.get_job("b").metadata["colocation_trial"]["result"]
    assert result["remaining_updates"] == {"a": 200, "b": 200}
    assert result["initial_phase_reserves"]["b"]["pending_phase_counts"]["validation"] == 3
    assert result["current_phase_reserves"]["b"]["pending_phase_counts"]["validation"] == 2


def test_provisional_missing_phase_evidence_hits_finite_update_budget(tmp_path):
    _, store, _, service, trial = fixture(tmp_path, phases=False)
    initial_pair_evidence(store, service, trial)
    ack(store, ["a", "b"])
    service._evaluate_colocation_trial()
    store.update_job("b", metadata_updates={"runtime_global_step": trial.cold_start["target_step"]})
    service._evaluate_colocation_trial()
    result = store.get_job("b").metadata["colocation_trial"]
    assert result["decision"] == "inconclusive" and "budget exhausted" in result["reason"]


def test_fast_candidate_budget_yield_cannot_validate_slow_incumbent_window(tmp_path):
    _, store, _, service, trial = fixture(tmp_path)
    start_measurements(store, service, trial)
    measure_group(store, service, trial, seconds=1)
    ack(store, ["a", "b"])
    service._evaluate_colocation_trial()
    assert trial.phase == "completion_measure"
    window(store, "b", steps=trial.cold_start["target_step"] - store.get_job("b").metadata["runtime_global_step"])
    window(store, "a", seconds=10)
    service._evaluate_colocation_trial()
    result = store.get_job("b").metadata["colocation_trial"]
    assert result["decision"] == "inconclusive" and "budget exhausted" in result["reason"]
    assert '["a","b"]' not in trial.cold_start["completion"]["groups"]


def test_fresh_identity_matched_singleton_cache_skips_selected_solo_measurement(tmp_path):
    _, store, _, service, trial = fixture(tmp_path)
    job = store.get_job("b")
    observation = {"identity": service._completion_identity([job]), "at": time.time(),
                   "rates": {"b": [1, 1]}, "source": "phase_clean_step_windows_v1"}
    store.update_job("b", metadata_updates={"completion_group_observations": {'["b"]': observation}})
    start_measurements(store, service, trial)
    assert trial.cold_start["completion"]["measuring"] == ["a", "b"]


@pytest.mark.parametrize("change", ["hardware", "runtime_identity", "old", "source"])
def test_invalid_singleton_cache_never_supplies_candidate_solo(tmp_path, change):
    _, store, _, service, trial = fixture(tmp_path)
    observation = {"identity": service._completion_identity([store.get_job("b")]), "at": time.time(),
                   "rates": {"b": [1, 1]}, "source": "phase_clean_step_windows_v1"}
    if change == "hardware": observation["identity"]["hardware_key"] = "another-gpu"
    elif change == "runtime_identity": observation["identity"]["members"][0]["completion_identity"] = "old-source"
    elif change == "old": observation["at"] = 0
    else: observation["source"] = "model_step_cost"
    store.update_job("b", metadata_updates={"completion_group_observations": {'["b"]': observation}})
    start_measurements(store, service, trial)
    assert trial.cold_start["completion"]["measuring"] == ["b"]


def test_zero_updates_still_owes_pending_validation_and_final_checkpoint(tmp_path):
    _, store, _, service, trial = fixture(tmp_path)
    start_measurements(store, service, trial)
    snapshot = deepcopy(trial.cold_start["completion"]["initial_snapshots"]["b"])
    snapshot.update(global_step=300, completed_epochs=2, checkpoint_completed_epoch=2, checkpoint_completed_step=200)
    reserve, detail = observed_phase_reserve(store.get_job("b"), snapshot, hardware_key=store.hardware_key(),
                                            now=time.time(), ttl_seconds=86400)
    assert reserve is not None and detail["pending_phase_counts"] == {"validation": 1, "checkpoint": 2, "finalization": 1}
    snapshot.update(completed_epochs=3)
    reserve, detail = observed_phase_reserve(store.get_job("b"), snapshot, hardware_key=store.hardware_key(),
                                            now=time.time(), ttl_seconds=86400)
    assert detail["pending_phase_counts"]["validation"] == 0
    assert detail["unknown_finalization"]


def test_epoch_derived_or_wrong_hardware_history_is_not_phase_evidence(tmp_path):
    _, store, _, service, trial = fixture(tmp_path)
    start_measurements(store, service, trial)
    history = phase_history(store, "b")
    history[0]["source"] = "runner_step_time"
    history[1]["hardware_key"] = "other"
    current = store.update_job("b", metadata_updates={"runtime_phase_timing_history": history})
    reserve, detail = observed_phase_reserve(current, trial.cold_start["completion"]["initial_snapshots"]["b"],
                                            hardware_key=store.hardware_key(), now=time.time(), ttl_seconds=86400)
    assert reserve is None and set(detail["missing_required_phases"]) == {"validation", "checkpoint"}


def test_three_member_half_orders_measure_only_missing_exact_tail_vectors(tmp_path):
    _, store, _, service, trial = fixture(tmp_path, members=("a", "b", "c"))
    members = ["a", "b", "c"]
    start_measurements(store, service, trial, members)
    assert trial.cold_start["completion"]["measuring"] == ["c"]
    measure_group(store, service, trial, seconds=1)
    ack(store, members)
    service._evaluate_colocation_trial()
    for name, seconds in {"a": [1, 1], "b": [2, .5], "c": [.5, 2]}.items():
        window(store, name, seconds)
    service._evaluate_colocation_trial()
    ack(store, members)
    service._evaluate_colocation_trial()
    assert trial.phase == "completion_measurement_barrier"
    assert trial.cold_start["completion"]["measuring"] == ["a", "c"]
    assert '["a","c"]' not in trial.cold_start["completion"]["groups"]
    ack(store, members)
    service._evaluate_colocation_trial()
    assert store.get_job("b").metadata["trial_step_command"]["action"] == "yield"
    window(store, "a", 1)
    window(store, "c", 1)
    service._evaluate_colocation_trial()
    ack(store, members)
    service._evaluate_colocation_trial()
    assert trial.cold_start["completion"]["measuring"] == ["b"]
    measure_group(store, service, trial, seconds=1)
    ack(store, members)
    service._evaluate_colocation_trial()
    outcome = store.get_job("c").metadata["colocation_trial"]
    assert outcome["decision"] == "accepted"
    assert '["a","c"]' in outcome["result"]["measurement_groups"]
    assert '["b"]' in outcome["result"]["measurement_groups"]
    assert "inherited_parent" not in str(outcome["result"])


def test_membership_reduction_discards_old_comparison_snapshots(tmp_path):
    _, store, _, service, trial = fixture(tmp_path, members=("a", "b", "c"))
    start_measurements(store, service, trial, ["a", "b", "c"])
    assert trial.cold_start["completion"]["initial_snapshots"]
    store.update_job("b", status=JobStatus.COMPLETED)
    service.supervisor.active_job_ids.return_value = ["a", "c"]
    service._evaluate_colocation_trial()
    assert trial.phase == "prepare" and "completion" not in trial.cold_start
    assert trial.cold_start["admission_incumbent_job_ids"] == ["a", "b"]


def test_runtime_identity_change_cannot_splice_old_snapshot_into_new_measurements(tmp_path):
    _, store, _, service, trial = fixture(tmp_path)
    start_measurements(store, service, trial)
    store.update_job("a", metadata_updates={"runtime_completion_identity": "changed-runtime"})
    service._evaluate_colocation_trial()
    result = store.get_job("b").metadata["colocation_trial"]
    assert result["decision"] == "inconclusive" and "identity changed" in result["reason"]
