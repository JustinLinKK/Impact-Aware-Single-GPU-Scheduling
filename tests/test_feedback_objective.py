from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from localml_scheduler.config import GpuSchedulerSettings
from localml_scheduler.domain import JobStatus
from localml_scheduler.scheduler.feedback_objective import (
    CompletionCosts, FeedbackRequest, compare_feedback_admission, derived_solo_slope,
    requests_from_jobs, score_requests,
)


def example(policy="individual", **changes):
    requests = ([FeedbackRequest(j, "w", "individual", (j,), 0) for j in ("a", "b")]
                if policy == "individual" else [FeedbackRequest("round", "w", "group", ("a", "b"), 0)])
    args = dict(remaining={"a": 100., "b": 1.}, solo={"a": 1., "b": 1.},
                incumbents=("a",), candidate="b", queue=["b"],
                pre_halves={"a": [1., 1.]}, packed_halves={"a": [.5, .5], "b": [.4, .4]},
                requests=requests, completed={}, now=0.)
    args.update(changes)
    return compare_feedback_admission(**args)


def test_two_policies_share_forecasts_but_choose_different_continuations():
    individual, group = example(), example("group")
    assert individual["decision"] == "accepted"
    assert group["decision"] == "rejected"
    assert individual["halves"][0]["gain_seconds"] == 48.625
    assert group["halves"][0]["gain_seconds"] == -.25
    assert individual["halves"][0]["admit"]["job_completions"] == group["halves"][0]["admit"]["job_completions"]


def test_same_post_trial_remaining_work_is_used_for_both_actions():
    r = example(remaining={"a": 90., "b": 1.})
    assert r["halves"][0]["defer"]["job_completions"] == {"a": 90., "b": 91.}
    assert r["halves"][0]["admit"]["job_completions"] == {"a": 91.25, "b": 2.5}


def test_two_halves_disagree_and_exact_threshold_passes():
    assert example(min_gain_seconds=48.625)["decision"] == "accepted"
    r = example(packed_halves={"a": [.5, .001], "b": [.4, .001]})
    assert r["decision"] == "inconclusive" and r["error"] is None


@pytest.mark.parametrize("bad", [None, 0, -1, float('nan'), float('inf'), True])
def test_invalid_prediction_or_rate_is_not_a_zero_cost(bad):
    assert example(solo={"a": 1., "b": bad})["decision"] == "inconclusive"
    assert example(packed_halves={"a": [.5, .5], "b": [bad, .4]})["decision"] == "inconclusive"
    with pytest.raises((ValueError, TypeError, ZeroDivisionError)):
        derived_solo_slope(bad)


def test_invalid_cohort_budget_and_window_shape():
    for change in [dict(queue=["b", "b"]), dict(remaining={"a": 100, "b": 0}),
                   dict(solo={"a": 1}), dict(pre_halves={"a": [1]}),
                   dict(completed={"a": 0}), dict(requests=[]),
                   dict(costs={"a": CompletionCosts()})]:
        assert example(**change)["decision"] == "inconclusive"


def test_waiting_jobs_must_be_covered_and_impact_gain():
    requests = [FeedbackRequest(j, "w", "individual", (j,), 0) for j in ("a", "b", "c")]
    r = example(remaining={"a": 100, "b": 1, "c": 1}, solo={"a": 1, "b": 1, "c": 1},
                queue=["b", "c"], requests=requests)
    assert set(r["halves"][0]["admit"]["job_completions"]) == {"a", "b", "c"}
    assert r["halves"][0]["gain_seconds"] != example()["halves"][0]["gain_seconds"]
    requests.pop()
    assert example(requests=requests, remaining={"a": 100, "b": 1, "c": 1},
                   solo={"a": 1, "b": 1, "c": 1}, queue=["b", "c"])["error"]


def test_mixed_requests_weight_groups_once_and_preserve_partial_completion():
    reqs = [FeedbackRequest("group", "w", "group", ("a", "b"), 2, weight=2),
            FeedbackRequest("c", "w", "individual", ("c",), 5)]
    score = score_requests(reqs, {"a": 7, "b": 11, "c": 8})
    assert score['weighted_latency_seconds'] == 21
    assert score['mean_latency_seconds'] == 7
    assert score_requests([replace(r, weight=r.weight*10) for r in reqs],
                          {"a": 7, "b": 11, "c": 8})['mean_latency_seconds'] == 7
    with pytest.raises(ValueError):
        score_requests(reqs + [FeedbackRequest("again", "w", "individual", ("b",), 0)],
                       {"a": 7, "b": 11, "c": 8})
    extra = FeedbackRequest("round", "w", "group", ("a", "b", "old"), 0)
    r = example(requests=[extra], completed={"old": 2}, now=3)
    assert r["halves"][0]["admit"]["requests"][0]["completion_at"] == 104.25


def test_overhead_is_action_specific_and_delivery_does_not_hold_gpu():
    r = example(costs={"a": CompletionCosts(delivery=500), "b": CompletionCosts(pause=2, resume=3)})
    h = r["halves"][0]
    assert h["defer"]["job_completions"]["b"] == 106
    assert h["defer"]["job_completions"]["a"] == 602
    assert h["admit"]["job_completions"]["b"] == 2.5


def job(name, status=JobStatus.READY, metadata=None):
    return SimpleNamespace(job_id=name, workflow_id="w", submitted_at="2026-10-28T00:00:00+00:00",
                           status=status, finished_at="2026-10-28T00:01:00+00:00", metadata=metadata or {})


def group_metadata():
    return {"feedback_request": {"request_id":"round", "completion_policy":"group",
        "required_job_ids":["a","b"], "membership_finalized":True,
        "submitted_at":"2026-10-28T00:00:00+00:00"}}


def test_metadata_round_uses_original_submission_and_terminal_condition():
    raw = group_metadata()
    requests, completed = requests_from_jobs([job("a", JobStatus.COMPLETED, raw), job("b", metadata=raw)])
    assert len(requests) == 1 and set(completed) == {"a"}
    with pytest.raises(ValueError):
        requests_from_jobs([job("a", JobStatus.FAILED, raw), job("b", metadata=raw)])
    raw["feedback_request"]["completion_condition"] = "terminal"
    assert len(requests_from_jobs([job("a", JobStatus.FAILED, raw), job("b", metadata=raw)])[0]) == 1
    raw["feedback_request"]["submitted_at"] = "2026-10-28T00:02:00+00:00"
    with pytest.raises(ValueError):
        requests_from_jobs([job("a", metadata=raw), job("b", metadata=raw)])


@pytest.mark.parametrize("change", ["unsealed", "missing", "inconsistent", "double", "unsupported"])
def test_incomplete_or_ambiguous_request_never_silently_becomes_individual(change):
    a, b = job("a", metadata=group_metadata()), job("b", metadata=group_metadata())
    if change == "unsealed": a.metadata['feedback_request']['membership_finalized'] = False
    if change == "missing": b.job_id = 'c'
    if change == "inconsistent": b.metadata['feedback_request']['weight'] = 2
    if change == "double": b.metadata = {}
    if change == "unsupported": a.metadata['feedback_request']['completion_policy'] = 'first_acceptable'
    with pytest.raises(ValueError):
        requests_from_jobs([a, b])


def test_config_round_trip_and_reciprocal_representation():
    cfg = GpuSchedulerSettings(feedback_latency={"enabled": True, "allow_zero_overhead": True})
    assert GpuSchedulerSettings.from_dict(cfg.to_dict()).feedback_latency.to_dict() == cfg.feedback_latency.to_dict()
    assert derived_solo_slope(28.097898000851274) == pytest.approx(35.58985088385271)
    assert not GpuSchedulerSettings().feedback_latency.enabled
    with pytest.raises(ValueError):
        GpuSchedulerSettings(feedback_latency={"min_gain_seconds": float('nan')})


def test_common_request_weights_are_invariant_but_absolute_solo_bias_is_not():
    requests=[FeedbackRequest(j,'w','individual',(j,),0.,weight=7.) for j in ('a','b')]
    assert example(requests=requests)['halves'][0]['gain_seconds']==example()['halves'][0]['gain_seconds']
    original=example('group')
    biased=example('group',solo={'a':2.,'b':2.})
    assert original['decision']=='rejected' and biased['decision']=='accepted'
    assert original['halves'][0]['normalized_progress_ratio']==biased['halves'][0]['normalized_progress_ratio']


def test_unarrived_requests_and_ambiguous_member_container_are_invalid():
    requests=[FeedbackRequest(j,'w','individual',(j,),1.) for j in ('a','b')]
    assert example(requests=requests)['decision']=='inconclusive'
    with pytest.raises(ValueError):
        FeedbackRequest('round','w','group','ab',0.)


def test_profile_wrapper_adds_slope_without_mutating_six_output_model_contract():
    from localml_scheduler.prediction.ml_predictor import MLVramPredictor
    raw=[20.,18.,200.,42.,1000.,1200.]
    original=raw.copy()
    predictor=SimpleNamespace(_released=SimpleNamespace(contract={'trained_hardware':'A10'}),
        artifact_hash='artifact',_released_options=lambda job,batches:{32:raw})
    j=SimpleNamespace(job_id='job',packing=SimpleNamespace(signature='signature'))
    profile=MLVramPredictor.predict_profile_options(predictor,j,[32])[32]
    assert raw==original and len(raw)==6
    assert profile['derived_solo_slope']==dict(value=50.,unit='updates_per_second',
        source='reciprocal_wall_step_prediction',step_semantics='nominal_optimizer_update',
        artifact_sha256='artifact',calibration='uncalibrated_prior')
    assert profile['step_cost']==raw[0] and profile['epoch_cost']==raw[2]
    assert profile['kind']=='predicted_workload' and profile['requires_online_validation']
