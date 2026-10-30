"""Regression proof for invalidated pilot v1; no model fitting or GPU access."""
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from localml_scheduler.benchmarks.a10_feedback_latency.calibration import RecordingPredictor
from localml_scheduler.benchmarks.a10_feedback_latency.run import (
    accept_job, build_job, make_settings, policy_activation, require_predictor_exposure,
)
from localml_scheduler.client import SchedulerClient
from localml_scheduler.domain import JobStatus
from localml_scheduler.hardware import HardwareProfile
from localml_scheduler.scheduler.feedback_objective import requests_from_jobs
from localml_scheduler.scheduler.feedback_trials import freeze_feedback_trial, evaluate_feedback_trial
from localml_scheduler.scheduler.resource_estimator import ResourceEstimator


MANIFEST = Path(__file__).parents[1] / "benchmarks/a10_feedback_latency/workloads/manifest.json"


class FixturePredictor:
    available = True
    def __init__(self): self.last_sources, self.last_errors = {}, {}
    def predict_profile_options(self, job, batches):
        self.last_sources[job.job_id] = "ml_predictor"
        return {batch: {"kind": "predicted_workload", "artifact_sha256": "cpu-fixture-artifact",
            "hardware": "A10", "batch_size": batch, "job_signature": job.packing.signature,
            "step_cost": 100., "epoch_cost": 3200., "avg_vram_mib": 1000.,
            "peak_vram_mib": 1100., "busy_indicator": 50.} for batch in batches}


@pytest.mark.parametrize("mode", ["individual", "group"])
@pytest.mark.parametrize("slow,expected", [(False, "accepted"), (True, "rejected")])
def test_assembled_submitted_cohort_reaches_real_v4_comparator(tmp_path, mode, slow, expected):
    manifest = json.loads(MANIFEST.read_text())
    specs = manifest["jobs"][:2]
    settings = make_settings({}, tmp_path, {"total_mib": 22600.})
    client = SchedulerClient(settings)
    # Explicit synthetic protocol fixture; no local device detection or training.
    client.store.backend._hardware_profile = HardwareProfile("cpu-fixture", "linux", "NVIDIA A10",
        22600, "8.6", None, "cpu-fixture")
    estimator = ResourceEstimator(settings, client.store)
    estimator.ml_predictor = FixturePredictor()
    jobs = [build_job(spec, manifest_path=MANIFEST, output=tmp_path, manifest=manifest,
                      memory_budget_mib=20000.)[0] for spec in specs]
    accepted = time.time() - 30.
    group_time = datetime.fromtimestamp(accepted, timezone.utc).isoformat()
    for index, job in enumerate(jobs):
        job.submitted_at = "2000-01-01T00:00:00+00:00"  # The actual old-harness failure.
        stamp = datetime.fromtimestamp(accepted + index, timezone.utc).isoformat()
        job.metadata["feedback_request"] = {"request_id": job.job_id if mode == "individual" else "bracket",
            "workflow_id": job.workflow_id, "completion_policy": mode,
            "required_job_ids": [job.job_id] if mode == "individual" else [j.job_id for j in jobs],
            "membership_finalized": True, "submitted_at": stamp if mode == "individual" else group_time}
        with pytest.raises(ValueError, match="request submission cannot follow member submission"):
            requests_from_jobs([job])
        job.metadata["feedback_phase_model"] = {"source": "measured_phase_model_v1",
            "model_role": "CPU_TEST_ONLY_SYNTHETIC_PHASE_FIXTURE", "covers_required_phases": True,
            "job_signature": job.packing.signature, "batch_size": specs[index]["batch"],
            "hardware_key": "cpu-fixture", "backend": "cuda_process", "observed_at": group_time,
            **{phase + "_seconds": .01 for phase in ("startup", "finish", "delivery", "pause", "resume")}}
        submitted = accept_job(client, job, accepted + index)
        assert submitted.submitted_at == stamp
        client.store.update_job(job.job_id, status=JobStatus.RUNNING, metadata_updates={
            "runtime_global_step": 8, "runtime_total_optimizer_updates": 64,
            "runtime_steps_per_epoch": 32, "runtime_planned_epochs": 2})
    actual = client.store.list_jobs()
    requests, completed = requests_from_jobs(actual)
    assert len(requests) == (2 if mode == "individual" else 1) and not completed
    active_ids = [j.job_id for j in jobs]
    service = SimpleNamespace(settings=settings, store=client.store,
        planner=SimpleNamespace(estimator=estimator),
        supervisor=SimpleNamespace(active_job_ids=lambda: active_ids))
    trial = SimpleNamespace(candidate_job_id=jobs[1].job_id, preexisting_job_ids=(jobs[0].job_id,),
                            cold_start={"add_only": {"pre_add": {jobs[0].job_id: [.1, .11]}}})
    trial.cold_start["add_only"]["feedback"] = freeze_feedback_trial(service, trial)
    scale = 1000. if slow else 1.
    windows = {job.job_id: {"halves": [a * scale, b * scale], "steps": 8,
        "start_step": 8, "end_step": 16, "excludes_nonupdate_phases": True}
        for job, (a, b) in zip(jobs, ((.1, .11), (.1, .09)))}
    result = evaluate_feedback_trial(service, trial, actual, windows)
    assert result["error"] is None
    assert len(result["halves"]) == 2
    assert result["decision"] == expected
    assert all(value == 56 for value in result["remaining_updates"].values())


@pytest.mark.parametrize("kind", ["fallback", "exception", "missing"])
def test_any_invalid_predictor_exposure_is_fatal(kind):
    job = SimpleNamespace(job_id="a", packing=SimpleNamespace(signature="s"))
    base = FixturePredictor()
    recorder = RecordingPredictor(base, lambda *args: None)
    if kind == "fallback": base.last_sources["a"] = "branch_profile"
    elif kind == "exception": recorder.failures["a"] = {"error": "capture RSS exceeded"}
    else:
        base.last_sources["a"] = "ml_predictor"
        recorder.preparations_finished.add("a")
    with pytest.raises(RuntimeError): require_predictor_exposure(recorder, {"name": job}, {"name": {"batch": 16}})


def test_missing_numeric_profile_is_recorded_at_preparation_failure():
    class Missing(FixturePredictor):
        def predict_profile_options(self, job, batches): return {}
    recorder = RecordingPredictor(Missing(), lambda *args: None)
    job = SimpleNamespace(job_id="a", packing=SimpleNamespace(signature="s"))
    with pytest.raises(ValueError): recorder.predict_profile_options(job, [16])
    assert "a" in recorder.failures and "a" in recorder.preparations_finished


def test_policy_activation_distinguishes_actionable_windows_errors_and_revised_skips():
    events = [
        {"event_type": "cold_start_trial_decided", "payload": {"decision": "accepted", "error": None, "halves": [{}, {}]}},
        {"event_type": "cold_start_trial_decided", "payload": {"decision": "rejected", "error": None,
            "scenarios": [{"forecast": {"error": None, "halves": [{}, {}]}}]}},
        {"event_type": "cold_start_trial_decided", "payload": {"decision": "inconclusive", "error": "request clock invalid"}},
        {"event_type": "pilot_trial_skipped", "payload": {"reason": "remaining work cannot amortize reserve"}},
    ]
    counts = policy_activation(events)
    assert counts["actionable_forecasts"] == 2
    assert counts["accepted_forecasts"] == counts["rejected_forecasts"] == 1
    assert counts["inconclusive_or_invalid"] == counts["amortization_skips"] == 1
    assert counts["inconclusive_reasons"] == {"request clock invalid": 1}
