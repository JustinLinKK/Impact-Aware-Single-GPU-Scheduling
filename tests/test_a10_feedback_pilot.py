from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from localml_scheduler.benchmarks.a10_feedback_latency.policy import (
    conservative_admission, trial_amortizes,
)
from localml_scheduler.benchmarks.a10_feedback_latency.receipts import (
    Submission, receive, request_latency,
)
from localml_scheduler.scheduler.feedback_objective import FeedbackRequest


def fixture(tmp_path):
    result = tmp_path / "result.json"
    execution = tmp_path / "execution.json"
    result.write_text(json.dumps(dict(global_step=20, epochs=2, signature="a" * 64,
        metric_name="negative_validation_auc", selected_metric=-.8,
        history=[dict(epoch=i, validation_auc=.8, negative_validation_auc=-.8,
                      validation_bce=.4, validation_negative_count=128, validation_positive_count=128)
                 for i in (1, 2)])))
    execution.write_text(json.dumps(dict(exc_type=None)))
    submission = Submission("job", 10., 100., 20, 2, str(result), str(execution), "a" * 64, "b" * 64)
    job = SimpleNamespace(job_id="job", status="COMPLETED", metadata={
        "runtime_global_step": 20, "runtime_total_optimizer_updates": 20,
        "runtime_finalization_complete": True, "runtime_completion_identity": "a" * 64})
    return submission, job


def test_receipt_waits_for_successful_exit_and_includes_parse_cost(tmp_path):
    submission, job = fixture(tmp_path)
    assert receive(submission, job, None) is None
    receipt = receive(submission, job, 0, monotonic=lambda: 15., wall_clock=lambda: 105.)
    assert receipt["jct_seconds"] == 5.
    assert receipt["feedback_ready_unix"] == 105.
    assert len(receipt["result_sha256"]) == 64
    assert receipt["returned_result"]["global_step"] == 20
    with pytest.raises(RuntimeError):
        receive(submission, job, 9)


@pytest.mark.parametrize("field,value", [("runtime_global_step", 19),
    ("runtime_total_optimizer_updates", 21), ("runtime_finalization_complete", False),
    ("runtime_completion_identity", "other-source")])
def test_completed_without_exact_success_proof_is_invalid(tmp_path, field, value):
    submission, job = fixture(tmp_path)
    job.metadata[field] = value
    with pytest.raises(ValueError):
        receive(submission, job, 0)


def test_returned_artifact_mismatch_and_nonfinite_json_fail(tmp_path):
    submission, job = fixture(tmp_path)
    for payload in ('{"global_step":19,"epochs":2,"signature":"source-data-recipe"}',
                    '{"global_step":20,"epochs":2,"signature":"source-data-recipe","score":NaN}'):
        (tmp_path / "result.json").write_text(payload)
        with pytest.raises(ValueError):
            receive(submission, job, 0)


def test_mutable_metadata_and_result_cannot_replace_frozen_expected_identity(tmp_path):
    submission, job = fixture(tmp_path)
    payload = json.loads((tmp_path / "result.json").read_text())
    payload["signature"] = "c" * 64
    job.metadata["runtime_completion_identity"] = payload["signature"]
    (tmp_path / "result.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError): receive(submission, job, 0)


@pytest.mark.parametrize("field,value", [("validation_auc", 1.01),
    ("validation_negative_count", 0), ("validation_positive_count", 127),
    ("validation_bce", -1), ("negative_validation_auc", .8)])
def test_valid_auc_and_both_classes_required_before_child_release(tmp_path, field, value):
    submission, job = fixture(tmp_path)
    payload = json.loads((tmp_path / "result.json").read_text())
    payload["history"][0][field] = value
    (tmp_path / "result.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError): receive(submission, job, 0)


def test_group_waits_for_all_receipts_and_counts_original_clock():
    receipts = {"a": {"feedback_ready_monotonic": 12.}}
    assert request_latency("group", ["a", "b"], receipts, 1.) is None
    receipts["b"] = {"feedback_ready_monotonic": 20.}
    assert request_latency("group", ["a", "b"], receipts, 1.) == 19.
    assert request_latency("individual", ["a"], receipts, 1.) == 11.
    with pytest.raises(ValueError):
        request_latency("individual", ["a", "b"], receipts, 1.)


def admission_args(mode="individual"):
    requests = ([FeedbackRequest(j, "w", "individual", (j,), 0) for j in ("a", "b")]
                if mode == "individual" else [FeedbackRequest("r", "w", "group", ("a", "b"), 0)])
    return dict(remaining={"a": 100., "b": 1.}, solo={"a": 1., "b": 1.},
                incumbents=("a",), candidate="b", queue=["b"],
                pre_halves={"a": [1., 1.]}, packed_halves={"a": [.5, .5], "b": [.4, .4]},
                requests=requests, completed={}, now=0.)


def test_conservative_policy_accounts_reserve_and_does_not_mutate_predictions():
    args = admission_args()
    frozen = deepcopy(args)
    result = conservative_admission(**args, measured_reserve_seconds=2.,
                                    solo_lower={"a": .8, "b": .8},
                                    solo_upper={"a": 1.2, "b": 1.2})
    assert result["decision"] == "accepted" and result["error"] is None
    assert result["required_gain_seconds"] == 3.
    assert args == frozen
    assert len(result["scenarios"]) == 5
    assert conservative_admission(**args, measured_reserve_seconds=100.)["decision"] == "rejected"


def test_conservative_group_does_not_hide_nominal_regret():
    result = conservative_admission(**admission_args("group"), measured_reserve_seconds=.1)
    assert result["decision"] == "rejected"
    assert result["worst_gain_seconds"] == -.25


def test_uncertain_halves_and_invalid_evidence_are_not_admitted():
    args = admission_args()
    args["packed_halves"] = {"a": [.5, .001], "b": [.4, .001]}
    assert conservative_admission(**args, measured_reserve_seconds=1.)["decision"] == "rejected"
    assert conservative_admission(**admission_args(), measured_reserve_seconds=0.)["decision"] == "inconclusive"
    assert conservative_admission(**admission_args(), measured_reserve_seconds=1.,
                                  solo_lower={"a": 2., "b": 1.})["error"]


def test_trial_budget_skips_short_work_and_repeated_exploration():
    args = dict(remaining_updates=100, solo_updates_per_second=10., measured_reserve_seconds=5.)
    assert trial_amortizes(**args, attempts=0)
    assert not trial_amortizes(**args, attempts=1)
    assert not trial_amortizes(**{**args, "remaining_updates": 20}, attempts=0)
    with pytest.raises(ValueError):
        trial_amortizes(**{**args, "measured_reserve_seconds": 0}, attempts=0)
