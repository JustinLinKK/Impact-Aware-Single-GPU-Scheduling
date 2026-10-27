"""Ranking hints cannot stand in for calibrated time/capacity measurements."""
from types import SimpleNamespace

import pytest

from localml_scheduler.prediction import JobPredictionError
from localml_scheduler.tests.test_cold_start_trials import setup


def hints(planner, busy, epoch_cost, scale=1):
    def unavailable_seconds(*args):
        raise AssertionError("nominal predictor seconds must not enter the time objective")
    planner.estimator.ml_predictor = SimpleNamespace(
        available=True, requires_branch_memory=True, requires_measured_runtime=True,
        predict_busy_indicator_options=lambda job, batches: {b: busy[job.job_id] for b in batches},
        predict_epoch_cost_options=lambda job, batches: {b: scale * epoch_cost[job.job_id] for b in batches},
        predict_seconds_per_epoch_options=unavailable_seconds,
    )


@pytest.mark.parametrize("scale", [.00001, 1, 1e8])
def test_legacy_work_helper_accounts_for_epoch_budget_but_cooperative_fallback_is_fifo(tmp_path, scale):
    _, _, planner, jobs = setup(tmp_path)
    hints(planner, {"a": 40, "b": 40, "c": 40}, {"a": 300, "b": 100, "c": 200}, scale)
    # a has fewer remaining epochs, but b still has lower total epoch cost.
    jobs[0].metadata["last_completed_epoch"] = 1
    available = {"cuda_process": True}
    assert planner.estimator.workload_rank(jobs[1], prefer_high=True) < planner.estimator.workload_rank(jobs[0], prefer_high=True)
    assert planner.choose_plan(jobs[:2], backend_available=available).job_ids == ("a",)
    jobs[0].metadata["last_completed_epoch"] = 2
    jobs[1].max_epochs = jobs[1].config.max_epochs = 5
    assert planner.estimator.workload_rank(jobs[0], prefer_high=True) < planner.estimator.workload_rank(jobs[1], prefer_high=True)
    assert planner.choose_plan(jobs[:2], backend_available=available).job_ids == ("a",)
    # No synthetic seconds or gain from either model scale.
    assert planner.estimator.estimate_batch_options(jobs[0], "exclusive", [64]) == []
    assert planner.estimator.predicted_remaining_runtime_seconds(jobs[0], backend_name="exclusive") is None


@pytest.mark.parametrize("transform", [lambda x: x, lambda x: x**2 / 100, lambda x: x / 1000])
def test_scalar_busy_helper_keeps_ordinal_order_but_unbound_cooperative_hints_use_fifo(tmp_path, transform):
    _, _, planner, jobs = setup(tmp_path)
    hints(planner, {key: transform(value) for key, value in {"a": 95, "b": 99, "c": 100}.items()},
          {"a": 1, "b": 1e6, "c": 1})
    available = {"cuda_process": True}
    assert planner.estimator.workload_rank(jobs[1], prefer_high=True) < planner.estimator.workload_rank(jobs[0], prefer_high=True)
    assert planner.choose_plan(jobs[:2], backend_available=available).job_ids == ("a",)
    addition = planner.choose_plan(jobs[:2], active_jobs=jobs[2:], backend_available=available)
    assert addition.job_ids == ("a",)
    assert addition.trial_metadata["requires_live_trial"]
    assert addition.objective_breakdown["gain"] is None


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), 0, -1])
def test_invalid_time_is_unknown_and_never_a_zero_cost_priority(tmp_path, bad):
    _, _, planner, jobs = setup(tmp_path)
    hints(planner, {"a": 30, "b": 30}, {"a": bad, "b": 100})
    assert planner.estimator.predicted_epoch_cost(jobs[0]) is None
    assert planner.estimator.workload_rank(jobs[1], prefer_high=True) < planner.estimator.workload_rank(jobs[0], prefer_high=True)
    assert planner.choose_plan(jobs[:2], backend_available={"cuda_process": True}).job_ids == ("a",)


def test_missing_epoch_budget_keeps_busy_hint_without_fabricating_runtime(tmp_path):
    _, _, planner, jobs = setup(tmp_path)
    hints(planner, {"a": 90, "b": 10}, {"a": 100, "b": 1})
    jobs[0].max_epochs = jobs[0].config.max_epochs = None
    assert planner.estimator.predicted_remaining_cost(jobs[0]) is None
    assert planner.choose_plan(jobs[:2], backend_available={"cuda_process": True}).job_ids == ("a",)


def test_unavailable_capture_falls_back_without_blocking_or_inventing_hints(tmp_path):
    _, _, planner, jobs = setup(tmp_path)
    def failure(*args):
        raise JobPredictionError("unsupported capture")
    planner.estimator.ml_predictor = SimpleNamespace(available=True,
        predict_epoch_cost_options=failure, predict_busy_indicator_options=failure)
    assert planner.estimator.workload_rank(jobs[0], prefer_high=True) == (True, 0, True, 0)
    assert planner.choose_plan(jobs[:2], backend_available={"cuda_process": True}).job_ids == ("a",)


def test_noncooperative_unmeasured_jobs_get_indicator_ranked_solo_fallback(tmp_path):
    _, _, planner, jobs = setup(tmp_path)
    hints(planner, {"a": 20, "b": 80}, {"a": 100, "b": 200})
    for job in jobs[:2]:
        job.metadata.pop("cooperative_trial")
    result = planner.choose_plan(jobs[:2], backend_available={"exclusive": True})
    assert result.mode == "exclusive" and result.job_ids == ("b",)
