"""Feasible busy bands precede original FIFO, never cardinal work costs."""
from copy import deepcopy
from itertools import permutations
from types import SimpleNamespace

import pytest

from localml_scheduler.prediction import JobPredictionError
from localml_scheduler.domain import JobStatus
from localml_scheduler.tests.test_cold_start_trials import setup


def predictor(planner, values, *, scale=1, missing=(), changed=None):
    """Values are busy, step work, epoch work, and optional peak MiB."""
    def profiles(job, batches):
        if job.job_id in missing:
            raise JobPredictionError("missing capture fixture")
        busy, step, epoch, *peak = values[job.job_id]
        result = {}
        for batch in batches:
            item = {"kind": "predicted_workload", "artifact_sha256": "one-shared-student",
                "hardware": "A10", "job_signature": job.packing.signature or job.job_id,
                "batch_size": batch, "busy_indicator": busy, "step_cost": scale * step,
                "epoch_cost": scale * epoch, "avg_vram_mib": 200,
                "peak_vram_mib": peak[0] if peak else 300}
            item.update((changed or {}).get(job.job_id, {}))
            result[batch] = item
        return result
    def no_seconds(*args):
        raise AssertionError("relative work must not become predicted seconds")
    planner.estimator.ml_predictor = SimpleNamespace(
        available=True, uses_predicted_profiles=True, requires_branch_memory=False,
        requires_measured_runtime=True, predict_profile_options=profiles,
        predict_busy_indicator_options=lambda job, batches: {batch: values[job.job_id][0] for batch in batches},
        predict_epoch_cost_options=lambda job, batches: {batch: scale * values[job.job_id][2] for batch in batches},
        predict_seconds_per_epoch_options=no_seconds, last_sources={}, last_errors={})


@pytest.mark.parametrize("scale", [1e-8, 1, 1e8])
def test_near_97_busy_uses_fifo_at_every_work_scale(tmp_path, monkeypatch, scale):
    _, _, planner, jobs = setup(tmp_path)
    heavy = deepcopy(jobs[-1])
    heavy.job_id = heavy.packing.signature = "d"
    jobs.append(heavy)
    for index, job in enumerate(jobs):
        job.submitted_at = f"2026-10-27T00:00:0{index}+00:00"
    # Same saturated-looking indicators as the source-bound PetFinder/cancer
    # predictions, but work differs by over thirty times.
    predictor(planner, {"a": (97.91, 28.53, 100), "b": (96.89, 271, 5000),
                        "c": (97.50, 419, 10000), "d": (97.56, 874, 20000)}, scale=scale)
    monkeypatch.setattr(planner, "predicted_remaining_runtime_seconds",
                        lambda *a, **k: pytest.fail("new cooperative ranking queried seconds"))
    for order in permutations(jobs):
        anchor = planner.choose_plan(list(order), backend_available={"cuda_process": True})
        assert anchor.job_ids == ("a",)
        assert anchor.objective_breakdown["candidate_ranking_policy"] == "cooperative_busy_band_fifo_v1"
        assert {rank[0] for rank in anchor.objective_breakdown["eligible_busy_ranks"].values()} == {0}
    join = planner.choose_plan(jobs[:3], active_jobs=jobs[3:], backend_available={"cuda_process": True})
    assert join.job_ids == ("a",) and join.trial_metadata["requires_live_trial"]
    assert join.objective_breakdown["gain"] is None
    # The previous relative-work experiment remains available as a research
    # helper, but its cardinal ordering is not the production policy.
    research = planner.estimator.cooperative_work_ranks(jobs, prefer_high=True)
    assert min(research, key=research.get) == "d"


@pytest.mark.parametrize("lower_busy,anchor,joining", [
    (94.99, "a", "b"), (95.0, "a", "b"), (95.01, "a", "a"),
])
def test_five_point_boundary_is_strict_and_anchored_at_group_extreme(tmp_path, lower_busy, anchor, joining):
    _, _, planner, jobs = setup(tmp_path)
    predictor(planner, {"a": (100, 10, 100), "b": (lower_busy, 1000, 10000), "c": (0, 1, 1)})
    assert planner.choose_plan(jobs[:2], backend_available={"cuda_process": True}).job_ids == (anchor,)
    assert planner.choose_plan(jobs[:2], active_jobs=jobs[2:], backend_available={"cuda_process": True}).job_ids == (joining,)


def test_busy_groups_do_not_chain_pairwise_near_values(tmp_path):
    _, _, planner, jobs = setup(tmp_path)
    predictor(planner, {"a": (100, 1, 1), "b": (97.4, 10, 10), "c": (94.9, 1000, 1000)})
    for order in permutations(jobs):
        plan = planner.choose_plan(list(order), backend_available={"cuda_process": True})
        assert plan.job_ids == ("a",)
        ranks = plan.objective_breakdown["eligible_busy_ranks"]
        assert ranks["a"][0] == ranks["b"][0] == 0 and ranks["c"][0] == 1


def test_same_group_uses_original_age_before_work_remaining_priority_and_id(tmp_path):
    _, _, planner, jobs = setup(tmp_path)
    predictor(planner, {"a": (98, 100, 1000), "b": (97, 100, 1000), "c": (96, 100, 1000)})
    jobs[1].metadata["last_completed_epoch"] = 1
    jobs[0].priority = 100
    jobs[1].submitted_at = "2020-01-01T00:00:00+00:00"
    assert planner.choose_plan(jobs, backend_available={"cuda_process": True}).job_ids == ("b",)
    jobs[1].metadata["last_completed_epoch"] = 0
    assert planner.choose_plan(jobs, backend_available={"cuda_process": True}).job_ids == ("b",)
    jobs[1].metadata.update(cold_start_solo_required=True, cold_start_failed_attempts=3)
    assert planner.choose_plan(jobs, backend_available={"cuda_process": True}).job_ids == ("b",)
    for job in jobs:
        job.submitted_at = "2020-01-01T00:00:00+00:00"
    jobs[0].priority = 0
    for order in permutations(jobs):
        assert planner.choose_plan(list(order), backend_available={"cuda_process": True}).job_ids == ("a",)


@pytest.mark.parametrize("failure", ["missing", "artifact", "hardware", "invalid_busy", "invalid_step",
                                      "invalid_artifact", "invalid_hardware"])
def test_any_candidate_invalid_or_inconsistent_profile_falls_back_to_fifo(tmp_path, failure):
    _, _, planner, jobs = setup(tmp_path)
    changed = {"artifact": {"artifact_sha256": "other"}, "hardware": {"hardware": "V100"},
               "invalid_busy": {"busy_indicator": float("nan")}, "invalid_step": {"step_cost": 0},
               "invalid_artifact": {"artifact_sha256": ["malformed"]}, "invalid_hardware": {"hardware": {"id": "A10"}}}
    predictor(planner, {"a": (97.91, 28, 100), "b": (97.5, 419, 1000), "c": (97.56, 874, 2000)},
              missing=("b",) if failure == "missing" else (),
              changed={"b": changed[failure]} if failure != "missing" else None)
    plan = planner.choose_plan(jobs, backend_available={"cuda_process": True})
    assert plan.job_ids == ("a",)
    assert plan.objective_breakdown["candidate_ranking_policy"] == "fifo_missing_consistent_busy_profiles"


@pytest.mark.parametrize("failure", ["missing", "artifact", "hardware"])
def test_active_profile_must_share_candidate_work_scale(tmp_path, failure):
    _, _, planner, jobs = setup(tmp_path)
    changed = {"artifact": {"artifact_sha256": "other"}, "hardware": {"hardware": "V100"}}
    predictor(planner, {"a": (97.91, 28, 100), "b": (97.5, 419, 1000), "c": (97.56, 874, 2000)},
              missing=("c",) if failure == "missing" else (),
              changed={"c": changed[failure]} if failure != "missing" else None)
    plan = planner.choose_plan(jobs[:2], active_jobs=jobs[2:], backend_available={"cuda_process": True})
    assert plan.job_ids == ("a",)
    assert plan.objective_breakdown["candidate_ranking_policy"] == "fifo_missing_consistent_busy_profiles"


def test_hard_memory_filter_precedes_busy_group_formation(tmp_path):
    _, _, planner, jobs = setup(tmp_path)
    budget = planner.estimator.safe_budget_mb()
    predictor(planner, {"a": (100, 1, 1, budget + 1), "b": (95.01, 10, 10), "c": (94.99, 1000, 1000)})
    plan = planner.choose_plan(jobs, backend_available={"cuda_process": True})
    assert plan.job_ids == ("b",)
    assert set(plan.objective_breakdown["eligible_busy_ranks"]) == {"b", "c"}
    assert plan.objective_breakdown["eligible_busy_ranks"]["c"][0] == 0
    assert planner.choose_plan(jobs[:1], backend_available={"cuda_process": True, "exclusive": True}) is None
    assert planner.choose_plan(jobs[1:2], active_jobs=jobs[2:], active_vram_mb=budget - 299,
                               backend_available={"cuda_process": True}) is None


def test_noncooperative_and_existing_scalar_workload_rank_remain_unchanged(tmp_path):
    _, _, planner, jobs = setup(tmp_path)
    predictor(planner, {"a": (97.91, 28, 100), "b": (97.5, 419, 1000), "c": (97.56, 874, 2000)})
    assert planner.estimator.workload_rank(jobs[0], prefer_high=True) < planner.estimator.workload_rank(jobs[1], prefer_high=True)
    for job in jobs:
        job.metadata.pop("cooperative_trial")
    plan = planner.choose_plan(jobs, backend_available={"exclusive": True})
    assert plan.mode == "exclusive" and plan.job_ids == ("a",)


def test_static_priority_window_cannot_hide_older_cooperative_job(tmp_path):
    settings, _, planner, jobs = setup(tmp_path)
    settings.gpu_scheduler.priority_window_size = 1
    predictor(planner, {"a": (97, 1, 1), "b": (97, 1000, 1000), "c": (97, 100, 100)})
    jobs[0].priority = 1000
    jobs[2].submitted_at = "2020-01-01T00:00:00+00:00"
    plan = planner.choose_plan(jobs, backend_available={"cuda_process": True})
    assert plan.job_ids == ("c",)
    jobs[2].metadata.update(cold_start_solo_required=True, cold_start_failed_attempts=3)
    assert planner.choose_plan(jobs, backend_available={"cuda_process": True}).job_ids == ("c",)


def test_oldest_paused_candidate_outside_eight_priority_slots_keeps_original_fifo_age(tmp_path):
    settings, _, planner, base = setup(tmp_path)
    settings.gpu_scheduler.priority_window_size = 8
    jobs = []
    values = {}
    for index in range(10):
        job = deepcopy(base[0])
        job.job_id = job.packing.signature = f"job-{index:02d}"
        job.priority = 100 - index
        job.submitted_at = f"2026-10-27T00:00:{index:02d}+00:00"
        values[job.job_id] = (97, 1000 - index, 1000 - index)
        jobs.append(job)
    jobs[-1].submitted_at = "2020-01-01T00:00:00+00:00"
    jobs[-1].status = JobStatus.PAUSED
    jobs[-1].hold = False
    jobs[-1].metadata.update(cold_start_failed_attempts=1, cold_start_retry_after=0)
    predictor(planner, values)
    assert planner.choose_plan(jobs, backend_available={"cuda_process": True}).job_ids == ("job-09",)
    assert planner.choose_plan(jobs[1:], active_jobs=jobs[:1], backend_available={"cuda_process": True}).job_ids == ("job-09",)


def test_memory_feasible_fourth_addition_has_no_hidden_concurrency_cap(tmp_path):
    settings, _, planner, jobs = setup(tmp_path)
    newcomer = deepcopy(jobs[-1])
    newcomer.job_id = newcomer.packing.signature = "d"
    predictor(planner, {job.job_id: (97, 100, 1000) for job in [*jobs, newcomer]})
    for job in jobs:
        job.status = JobStatus.RUNNING
    jobs[0].metadata.update(cold_start_solo_required=True, cold_start_failed_attempts=3)
    plan = planner.choose_plan([newcomer], active_jobs=jobs, backend_available={"cuda_process": True})
    assert settings.gpu_scheduler.parallel_job_cap is None
    assert plan.job_ids == ("d",) and plan.trial_metadata["requires_live_trial"]
