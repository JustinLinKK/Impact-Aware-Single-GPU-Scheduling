from __future__ import annotations

from itertools import product
from pathlib import Path
from time import perf_counter

import pytest

from localml_scheduler.config import (
    PREDICTION_MODE_BRANCH_PROFILE,
    SCHEDULER_MODE_ADAPTIVE,
    SchedulerSettings,
)
from localml_scheduler.domain import (
    BatchProfileCurve,
    BatchProfilePoint,
    BatchProbeSpec,
    JobStatus,
    PackingSpec,
    TrainingJob,
)
from localml_scheduler.scheduler.placement_planner import PlacementPlanner
from localml_scheduler.scheduler.adaptive_search import SearchState
from localml_scheduler.scheduler.policies import PriorityFifoPolicy
from localml_scheduler.storage import StateStore


def _settings(tmp_path: Path, *, exact_cutoff: int = 8) -> SchedulerSettings:
    return SchedulerSettings(
        runtime_root=tmp_path / "runtime",
        prediction={"mode": PREDICTION_MODE_BRANCH_PROFILE},
        gpu_scheduler={
            "mode": SCHEDULER_MODE_ADAPTIVE,
            "backend_priority": ["stream", "exclusive"],
            "candidate_window_size": 16,
            "max_packed_jobs_per_gpu": 8,
            "adaptive": {
                "exact_search_max_jobs": exact_cutoff,
                "vram_bucket_mb": 128,
                "frontier_width": 32,
                "finalist_limit": 64,
                "replan_debounce_seconds": 1.0,
            },
        },
    )


def _job(job_id: str, *, authored: int = 8, priority: int = 0, queue_sequence: int = 1) -> TrainingJob:
    namespace = f"branch:{job_id}"
    job = TrainingJob.create(
        runner_target="tests.fake:run",
        baseline_model_id=job_id,
        baseline_model_path=f"/{job_id}.pt",
        job_id=job_id,
        priority=priority,
        runner_kwargs={"batch_size": authored},
        packing=PackingSpec(eligible=True, signature=f"sig:{job_id}", backend_allowlist=["stream"]),
        batch_probe=BatchProbeSpec(
            enabled=True,
            model_key=job_id,
            profile_namespace=namespace,
            shape_signature_override=namespace,
            minimum_batch_size=1,
            contract_version=3,
        ),
        metadata={"elastic_contract_validated": True},
    )
    job.queue_sequence = queue_sequence
    return job


def _profile(
    store: StateStore,
    job: TrainingJob,
    points: dict[int, tuple[int, float]],
    *,
    contract_version: int = 3,
) -> None:
    curve_key = f"curve:{job.job_id}:v{contract_version}"
    curve = BatchProfileCurve(
        curve_key=curve_key,
        model_key=str(job.batch_probe.model_key),
        shape_signature=str(job.batch_probe.shape_signature_override),
        hardware_key=store.hardware_key(),
        profile_namespace=job.batch_probe.profile_namespace,
        contract_version=contract_version,
        minimum_batch_size=min(points),
        maximum_feasible_batch_size=max(points),
        right_censored=True,
    )
    store.upsert_batch_profile_curve(curve)
    for batch, (vram_mb, throughput) in points.items():
        store.upsert_batch_profile_point(
            BatchProfilePoint(
                point_key=f"{curve_key}:{batch}",
                curve_key=curve_key,
                batch_size=batch,
                peak_vram_mb=vram_mb,
                samples_per_second=throughput,
                median_step_time_ms=(batch / throughput) * 1000.0,
                observations=5,
            )
        )


def _planner(settings: SchedulerSettings) -> tuple[StateStore, PlacementPlanner]:
    store = StateStore(settings)
    planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
    return store, planner


def test_only_adaptive_scheduler_and_new_prediction_modes_are_accepted(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert settings.gpu_scheduler.mode == "adaptive"
    assert settings.gpu_scheduler.candidate_window_size == 16
    assert settings.gpu_scheduler.max_packed_jobs_per_gpu == 8
    assert settings.gpu_scheduler.adaptive.exact_search_max_jobs == 8
    with pytest.raises(ValueError, match="Unsupported scheduler mode"):
        SchedulerSettings(runtime_root=tmp_path / "old", gpu_scheduler={"mode": "parallel_auto_pack"})
    with pytest.raises(ValueError, match="Unsupported prediction mode"):
        SchedulerSettings(runtime_root=tmp_path / "old-prediction", prediction={"mode": "branch_only"})


def test_submission_snapshots_authored_batch_and_rejects_non_power_of_two(tmp_path: Path) -> None:
    store = StateStore(_settings(tmp_path))
    job = _job("power-two", authored=8)
    store.submit_job(job)
    job.current_batch_size = 16
    store.save_job(job)
    restored = store.get_job(job.job_id)
    assert restored is not None
    assert restored.authored_batch_size == 8
    assert restored.current_batch_size == 16
    assert restored.config.runner_kwargs["batch_size"] == 8

    invalid = _job("invalid", authored=3)
    with pytest.raises(ValueError, match="power of two"):
        store.submit_job(invalid)


def test_old_profile_contract_is_preserved_but_not_plannable(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store, planner = _planner(settings)
    job = _job("legacy")
    _profile(store, job, {4: (1000, 100.0), 8: (1500, 180.0)}, contract_version=2)
    assert store.list_batch_profile_curves()
    assert planner.profile_ready(job) is False


def test_branch_candidates_center_on_authored_and_preserve_active_current(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store, planner = _planner(settings)
    job = _job("candidate", authored=8)
    _profile(store, job, {1: (500, 10), 2: (700, 20), 4: (900, 40), 8: (1300, 75), 16: (2100, 120), 32: (3500, 150)})
    assert planner.search.candidate_batches(job, active=False) == [4, 8, 16]
    job.current_batch_size = 32
    assert planner.search.candidate_batches(job, active=True) == [8, 16, 32]


def test_abc_arrival_shrinks_active_batches_and_admits_all_three(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store, planner = _planner(settings)
    planner.estimator.safe_budget_mb = lambda: 10_000.0
    a, b, c = (_job("a", queue_sequence=1), _job("b", queue_sequence=2), _job("c", queue_sequence=3))
    for job in (a, b):
        _profile(store, job, {4: (2500, 90), 8: (4000, 160), 16: (6000, 240)})
        job.current_batch_size = 16
        job.status = JobStatus.RUNNING
    _profile(store, c, {4: (2000, 70), 8: (3000, 125), 16: (5000, 190)})

    plan = planner.choose_plan(
        [c],
        active_jobs=[a, b],
        backend_available={"stream": True, "exclusive": True},
    )
    assert plan is not None
    assert set(plan.job_ids) == {"a", "b", "c"}
    assert plan.batch_overrides["a"] < 16 or plan.batch_overrides["b"] < 16
    assert plan.estimated_vram_mb <= 10_000
    assert a.authored_batch_size == b.authored_batch_size == 8
    assert a.config.runner_kwargs["batch_size"] == b.config.runner_kwargs["batch_size"] == 8


def test_infeasible_waiting_job_does_not_evict_or_interrupt_active_jobs(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store, planner = _planner(settings)
    planner.estimator.safe_budget_mb = lambda: 10_000.0
    a, b, c = (_job("a"), _job("b", queue_sequence=2), _job("c", queue_sequence=3))
    for job in (a, b):
        _profile(store, job, {4: (5000, 100), 8: (6000, 150), 16: (8000, 190)})
        job.current_batch_size = 4
        job.status = JobStatus.RUNNING
    _profile(store, c, {4: (2000, 80), 8: (3000, 130), 16: (5000, 180)})

    plan = planner.choose_plan([c], active_jobs=[a, b], backend_available={"stream": True, "exclusive": True})
    assert plan is not None
    assert set(plan.job_ids) == {"a", "b"}
    assert plan.batch_overrides == {"a": 4, "b": 4}


def test_exact_search_matches_exhaustive_enumeration(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store, planner = _planner(settings)
    planner.estimator.safe_budget_mb = lambda: 9_000.0
    jobs = [_job(f"j{index}", priority=index % 2, queue_sequence=index + 1) for index in range(6)]
    for index, job in enumerate(jobs):
        _profile(
            store,
            job,
            {
                4: (800 + index * 20, 50 + index),
                8: (1300 + index * 30, 90 + index),
                16: (2200 + index * 40, 130 + index),
            },
        )
    planner.estimator.begin_planning_cycle()
    planner.compatibility.begin_planning_cycle()
    options = [planner.search.choices_for(job, backend_name="stream", active=False) for job in jobs]
    brute_best = None
    for vector in product(*[(None, *choices) for choices in options]):
        state = SearchState()
        feasible = True
        for job, choice in zip(jobs, vector):
            if choice is None:
                continue
            state = planner.search._extend(state, job, choice, active=False, backend_name="stream")
            if state is None:
                feasible = False
                break
        if feasible and state is not None and state.jobs and (brute_best is None or planner.search.score(state) > planner.search.score(brute_best)):
            brute_best = state
    exact, solver = planner.search.solve(jobs, active_job_ids=set(), backend_name="stream")
    assert solver == "exact_branch_and_bound"
    assert exact is not None and brute_best is not None
    assert planner.search.score(exact) == pytest.approx(planner.search.score(brute_best))


def test_bounded_dp_uses_conservative_buckets_and_never_exceeds_exact_budget(tmp_path: Path) -> None:
    settings = _settings(tmp_path, exact_cutoff=4)
    store, planner = _planner(settings)
    planner.estimator.safe_budget_mb = lambda: 6_000.0
    jobs = [_job(f"dp{index}", queue_sequence=index + 1) for index in range(12)]
    for job in jobs:
        _profile(store, job, {4: (900, 40), 8: (1400, 70), 16: (2300, 100)})
    plan = planner.choose_plan(jobs, backend_available={"stream": True, "exclusive": True})
    assert plan is not None
    assert plan.solver_kind == "bounded_multiple_choice_dp"
    assert plan.estimated_vram_mb <= 6_000
    assert len(plan.job_ids) <= settings.gpu_scheduler.max_packed_jobs_per_gpu


def test_sixteen_candidate_planner_p95_is_below_100_ms(tmp_path: Path) -> None:
    settings = _settings(tmp_path, exact_cutoff=8)
    store, planner = _planner(settings)
    planner.estimator.safe_budget_mb = lambda: 10_000.0
    jobs = [_job(f"bench{index}", priority=index % 3, queue_sequence=index + 1) for index in range(16)]
    for index, job in enumerate(jobs):
        _profile(
            store,
            job,
            {4: (850 + index, 40 + index), 8: (1400 + index, 70 + index), 16: (2300 + index, 100 + index)},
        )
    latencies_ms = []
    for _ in range(20):
        started = perf_counter()
        plan = planner.choose_plan(jobs, backend_available={"stream": True, "exclusive": True})
        latencies_ms.append((perf_counter() - started) * 1000.0)
        assert plan is not None and plan.solver_kind == "bounded_multiple_choice_dp"
    p95 = sorted(latencies_ms)[int(0.95 * len(latencies_ms)) - 1]
    assert p95 < 100.0, latencies_ms
