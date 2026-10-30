from copy import deepcopy
from types import SimpleNamespace

import pytest

from localml_scheduler.benchmarks.a10_feedback_latency.calibration import (
    CalibratedPredictor, fit_calibration, phase_model, trial_reserve,
)
from localml_scheduler.benchmarks.a10_feedback_latency import service as service_module


def rows():
    return [dict(name=f"dedicated{i}", role="calibration", complete=True,
        source_identity={"source_sha256": "a" * 64}, runtime_completion_identity="b" * 64,
        parameter_count=100 + i * 100, observed_at="2026-10-30T00:00:00+00:00",
        predicted_profile={"artifact_sha256": "p", "step_cost": 100., "peak_vram_mib": 1000.},
        isolated_seconds_per_update=.2, peak_reserved_mib=1100. + i * 50,
        startup_seconds=2., pause_seconds=.5, resume_seconds=1., delivery_seconds=.1,
        phase_history=[{"phase": p, "seconds": v} for p, v in
                       [("validation", 3.), ("checkpoint", 1.), ("finalization", 2.)]]) for i in range(2)]


def calibration():
    return fit_calibration(rows(), hardware_key="hardware", predictor_artifact_sha256="p")


def test_calibration_freezes_ratios_and_nonnegative_reserved_memory_proxy():
    fitted = calibration()
    assert fitted["wall_multiplier"] == 2.
    assert fitted["peak_add_mib"] == 150.
    assert fitted["observations"] == 2
    assert fitted["observed_at"] == "2026-10-30T00:00:00+00:00"
    assert trial_reserve(fitted) == 7.5
    invalid = rows()
    invalid[0]["role"] = "evaluation"
    with pytest.raises(ValueError): fit_calibration(invalid, hardware_key="hardware", predictor_artifact_sha256="p")


def test_proxy_keeps_calibration_timestamp_and_explicit_transfer_provenance():
    fitted = calibration()
    job = SimpleNamespace(max_epochs=2, packing=SimpleNamespace(signature="target"),
                          metadata={"pilot_parameter_count": 400})
    proxy = phase_model(fitted, job, hardware_key="hardware", batch_size=4)
    assert proxy["observed_at"] == fitted["observed_at"]
    assert proxy["job_signature"] == "target"
    assert proxy["model_role"] == "calibrated_cross_workload_phase_proxy_v1"
    assert proxy["observation_source"] == "dedicated_disjoint_calibration_not_this_target_job"
    assert proxy["parameter_scale"] == 2.
    assert proxy["finish_seconds"] == 22.
    assert proxy["pause_seconds"] == 1.
    with pytest.raises(ValueError): phase_model(fitted, job, hardware_key="other", batch_size=4)


def test_predictor_profile_only_calibrates_frozen_a10_prior_and_reciprocal():
    profile = dict(hardware="A10", artifact_sha256="p", step_cost=100., epoch_cost=3200.,
                   avg_vram_mib=800., peak_vram_mib=1000., derived_solo_slope={"value": 10.})
    class Predictor:
        artifact_hash = "p"
        model_id = "release/S_A10"
        def predict_profile_options(self, job, batch_sizes): return {b: profile for b in batch_sizes}
    frozen = deepcopy(profile)
    wrapper = CalibratedPredictor(Predictor(), calibration(), "hardware")
    result = wrapper.predict_profile_options(None, [4])[4]
    assert result["step_cost"] == 200. and result["epoch_cost"] == 6400.
    assert result["derived_solo_slope"]["value"] == 5.
    assert result["avg_vram_mib"] == 800. and result["peak_vram_mib"] == 1150.
    assert result["artifact_sha256"] != "p"
    assert profile == frozen
    with pytest.raises(ValueError): CalibratedPredictor(Predictor(), calibration(), "other")


def test_revised_service_wires_actual_admission_callback(monkeypatch):
    captured = {}
    def evaluate(service, trial, jobs, windows, *, compare):
        captured["result"] = compare(solo={}, candidate="missing", packed_halves={})
        return captured["result"]
    monkeypatch.setattr(service_module, "evaluate_feedback_trial", evaluate)
    service = object.__new__(service_module.RevisedPilotService)
    service.store = SimpleNamespace(get_job=lambda _: SimpleNamespace(metadata={"feedback_trial_reserve_seconds": 2.}))
    result = service._score_feedback_admission(SimpleNamespace(candidate_job_id="b"), [], {})
    assert result["admission_policy"] == "a10_feedback_worst_scenario_v1"
    assert result["error"]


def test_revised_dispatch_missing_budget_stays_solo_without_dispatching():
    candidate = SimpleNamespace(job_id="b", metadata={})
    updated, events = [], []
    service = object.__new__(service_module.RevisedPilotService)
    service.store = SimpleNamespace(get_job=lambda _: candidate,
        update_job=lambda *a, **kw: updated.append(kw))
    service.event_logger = SimpleNamespace(emit=lambda *a, **kw: events.append(kw))
    plan = SimpleNamespace(job_ids=("b",), trial_metadata={"requires_live_trial": True})
    assert not service._dispatch_plan(plan)
    assert updated[0]["metadata_updates"] == {"cold_start_solo_required": True}
    assert events[0]["payload"]["admission_policy"] == "a10_feedback_worst_scenario_v1"
