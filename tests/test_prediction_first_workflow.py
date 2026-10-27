"""Virtual profiles admit work without per-job exclusive calibration."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from localml_scheduler.domain import JobStatus
from localml_scheduler.scheduler.service import SchedulerService
from localml_scheduler.tests.test_cold_start_trials import setup


def virtual_predictor(cost_scale=1):
    def profiles(job, batches):
        return {batch: {
            "kind": "predicted_workload", "artifact_sha256": "test-shared-artifact", "hardware": "A10G",
            "batch_size": batch, "job_signature": job.packing.signature or job.job_id,
            "step_cost": cost_scale * {"a": 100, "b": 50, "c": 80}[job.job_id],
            "epoch_cost": cost_scale * 10000, "busy_indicator": {"a": 80, "b": 20, "c": 50}[job.job_id],
            "avg_vram_mib": 200, "peak_vram_mib": 300,
        } for batch in batches}
    def no_seconds(*args):
        raise AssertionError("predicted work must not become measured seconds")
    return SimpleNamespace(available=True, uses_predicted_profiles=True, requires_branch_memory=False,
        requires_measured_runtime=True, predict_profile_options=profiles,
        predict_busy_indicator_options=lambda j,b: {n:p["busy_indicator"] for n,p in profiles(j,b).items()},
        predict_epoch_cost_options=lambda j,b: {n:p["epoch_cost"] for n,p in profiles(j,b).items()},
        predict_seconds_per_epoch_options=no_seconds, last_sources={}, last_errors={})


def test_empty_profile_store_can_admit_anchor_and_newcomer_with_predicted_peak(tmp_path):
    _, _, planner, jobs = setup(tmp_path)
    planner.estimator.ml_predictor = virtual_predictor()
    available = {"cuda_process": True}
    first = planner.choose_plan(jobs[:2], backend_available=available)
    assert first.mode == "stack_anchor" and first.job_ids == ("a",)
    assert first.objective_breakdown["profile_origin"] == "predicted_workload"
    assert first.objective_breakdown["candidate_memory_mib"] == 300
    second = planner.choose_plan(jobs[1:2], active_jobs=jobs[:1], active_vram_mb=300, backend_available=available)
    assert second.trial_metadata["requires_live_trial"]
    assert set(second.trial_metadata["predicted_work_profiles"]) == {"a", "b"}
    assert planner.estimator.predicted_group_vram_mb(jobs[:2], backend_name="cuda_process") == 600
    assert planner.estimator.estimate_batch_options(jobs[0], "exclusive", [64]) == []
    assert planner.choose_plan(jobs[1:2], active_jobs=jobs[:1],
        active_vram_mb=planner.estimator.safe_budget_mb()-299, backend_available=available) is None


def test_predicted_peak_rejection_cannot_be_bypassed_by_exclusive_fallback(tmp_path):
    settings, _, planner, jobs = setup(tmp_path)
    planner.estimator.ml_predictor = virtual_predictor()
    settings.gpu_scheduler.memory.gpu_vram_gib = 200 / 1024
    assert planner.choose_plan(jobs[:1], backend_available={"cuda_process": True, "exclusive": True}) is None
    jobs[0].metadata.pop("cooperative_trial")
    assert planner.choose_plan(jobs[:1], backend_available={"exclusive": True}) is None


@pytest.mark.parametrize("scale", [.001, 1, 1000])
@pytest.mark.parametrize("overlap,expected", [(1.1, "accepted"), (3., "rejected")])
def test_online_trial_uses_selected_measured_reference_and_never_synthetic_seconds(tmp_path, scale, overlap, expected):
    from localml_scheduler.tests.test_completion_trials import fixture, start_measurements, measure_group, ack
    _, store, _, service, trial = fixture(tmp_path)
    for profile in trial.cold_start["predicted_work_profiles"].values():
        profile["step_cost"] *= scale
        profile["epoch_cost"] *= scale
    with patch.object(service, "_save_cold_reference", side_effect=AssertionError("no synthetic profile writes")), \
         patch.object(service.planner.time_objective, "estimate_gain", side_effect=AssertionError("no whole-epoch ETA")):
        start_measurements(store, service, trial)
        assert trial.cold_start["completion"]["measuring"] == ["b"]
        measure_group(store, service, trial, seconds=1)
        measure_group(store, service, trial, seconds=overlap)
        ack(store, ["a", "b"])
        service._evaluate_colocation_trial()
    result = store.get_job("b").metadata["colocation_trial"]
    assert result["decision"] == expected
    assert result["result"]["objective"] == "modeled_completion_proxy_v1"
    assert result["result"]["predictor_role"] == "ordering and VRAM guard only"
    for name in ["a", "b"]:
        assert store.get_runtime_profile(name, resolved_batch_size=64, backend_name="cuda_process") is None
    assert not store.get_job("c").metadata.get("trial_step_command")


def test_partial_or_mixed_artifact_profiles_use_existing_measurement_fallback(tmp_path):
    _, _, planner, jobs = setup(tmp_path)
    predictor = virtual_predictor()
    original = predictor.predict_profile_options
    def different_artifact(job, batches):
        values = original(job, batches)
        if job.job_id == "a":
            for value in values.values(): value["artifact_sha256"] = "different"
        return values
    predictor.predict_profile_options = different_artifact
    planner.estimator.ml_predictor = predictor
    plan = planner.choose_plan(jobs[1:2], active_jobs=jobs[:1], backend_available={"cuda_process": True})
    assert plan.trial_metadata["predicted_work_profiles"] == {}


def test_frozen_profile_rejects_batch_change_during_trial(tmp_path):
    settings, store, planner, jobs = setup(tmp_path)
    planner.estimator.ml_predictor = virtual_predictor()
    for job in jobs[:2]: store.update_job(job.job_id, status=JobStatus.RUNNING, metadata_updates={"trial_runner_ready": True})
    supervisor = Mock(); supervisor.active_job_ids.return_value = ["a", "b"]
    service = SchedulerService(settings, store=store, supervisor=supervisor)
    plan = planner.choose_plan(jobs[1:2], active_jobs=jobs[:1], backend_available={"cuda_process": True})
    trial = service._prepare_colocation_trial(plan, jobs[1])
    trial.cold_start["predicted_work_profiles"]["b"]["batch_size"] = 32
    service._evaluate_colocation_trial()
    result = store.get_job("b").metadata["colocation_trial"]
    assert result["decision"] == "inconclusive" and "identity changed" in result["reason"]
