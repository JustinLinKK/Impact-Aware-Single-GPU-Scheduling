"""Arrival-invariant decisions and a single 90% device-memory budget."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from itertools import permutations
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from localml_scheduler.config import SchedulerSettings
from localml_scheduler.execution.trial_control import configure_trial_memory
from localml_scheduler.scheduler.resource_estimator import ResourceEstimator
from localml_scheduler.scheduler.telemetry import MemoryAdmissionGate, GpuTelemetrySample
from localml_scheduler.scheduler.trace_simulator import TraceBatchOption, TraceJob, TraceProblem, _time_aware_choice
from localml_scheduler.tests.test_cold_start_trials import setup
from localml_scheduler.tests.test_time_aware_scheduler import _settings, _job, _seed_options
from localml_scheduler.scheduler.placement_planner import PlacementPlanner
from localml_scheduler.scheduler.policies import PriorityFifoPolicy
from localml_scheduler.storage.sqlite_store import SQLiteStateStore


@pytest.mark.parametrize("mode", ["baseline", "backend_awared"])
def test_plan_is_invariant_to_arrival_order_wait_and_clock(tmp_path, mode):
    settings = _settings(str(tmp_path), scheduler_decision_mode=mode, starvation_timeout_seconds=0,
                         priority_window_size=2, oldest_window_size=100)
    settings.enable_priority_aging = True  # Legacy explicit settings must also be inert.
    settings.aging_interval_seconds = 1
    store = SQLiteStateStore(settings)
    planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=True))
    jobs = [_job("a", "a", priority=5), _job("b", "b", priority=5), _job("c", "c", priority=0)]
    for job in jobs:
        _seed_options(store, planner, job)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    expected = None
    for order in permutations(jobs):
        for offset in (0, 86400):
            for i, job in enumerate(order):
                job.submitted_at = (now - timedelta(days=100 * i)).isoformat()
                job.queue_sequence = len(jobs) - i
            window, mandatory = planner._time_aware_window(list(order), now=now + timedelta(seconds=offset))
            assert [j.job_id for j in window] == ["a", "b"]
            assert mandatory is None
            plan = planner.choose_plan(order, backend_available={"cuda_process": True, "exclusive": True},
                                       now=now + timedelta(seconds=offset))
            result = (plan.mode, plan.backend_name, plan.job_ids, plan.batch_overrides, plan.mandatory_anchor_job_id)
            expected = expected or result
            assert result == expected


def test_cold_start_ignores_age_and_uses_entire_remaining_budget(tmp_path):
    settings, store, planner, jobs = setup(tmp_path)
    settings.gpu_scheduler.memory.gpu_vram_gib = 80
    budget = 80 * 1024 * .9
    jobs[1].resource_requirements.estimated_vram_mb = budget - 1024
    jobs[1].submitted_at = "1900-01-01T00:00:00+00:00"
    jobs[2].submitted_at = "2099-01-01T00:00:00+00:00"
    jobs[1].queue_sequence, jobs[2].queue_sequence = 999, 0
    plan = planner.choose_plan(jobs[1:], active_jobs=jobs[:1], active_vram_mb=1024,
                              backend_available={"cuda_process": True})
    assert plan.job_ids == ("b",)
    assert plan.trial_metadata["memory_limit_mb"] == pytest.approx(budget - 1024)
    jobs[1].resource_requirements.estimated_vram_mb += 1
    assert planner.choose_plan(jobs[1:2], active_jobs=jobs[:1], active_vram_mb=1024,
                               backend_available={"cuda_process": True}) is None


def test_default_budget_is_ninety_percent_of_detected_total():
    settings = SchedulerSettings()
    repo = SimpleNamespace(hardware_profile=lambda: SimpleNamespace(total_vram_mb=80 * 1024))
    estimator = ResourceEstimator(settings, repo)
    assert estimator.safe_budget_mb() == pytest.approx(80 * 1024 * .9)
    assert settings.gpu_scheduler.parallel_job_cap is None
    assert settings.gpu_scheduler.memory.live_admission_stop_fraction == .9
    assert settings.gpu_scheduler.memory.live_admission_resume_fraction == .85


def test_allocator_rechecks_budget_after_external_allocation(tmp_path):
    _, store, _, jobs = setup(tmp_path)
    job = jobs[0]
    job.resource_requirements.requires_gpu = True
    job.metadata["trial_memory_limit_mb"] = 32 * 1024 * .9
    context = Mock(job=job, store=store)
    with patch("torch.cuda.mem_get_info", return_value=(32 * 1024**3, 32 * 1024**3)), patch("torch.cuda.set_per_process_memory_fraction") as limit:
        configure_trial_memory(context)
        assert limit.call_args.args[0] == pytest.approx(.9)
    with patch("torch.cuda.mem_get_info", return_value=(3 * 1024**3, 32 * 1024**3)), patch("torch.cuda.set_per_process_memory_fraction") as limit:
        with pytest.raises(torch.OutOfMemoryError, match="No live memory allowance"):
            configure_trial_memory(context)
        limit.assert_not_called()


def test_live_gate_has_no_eighty_five_percent_cap_while_open():
    gate = MemoryAdmissionGate()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    def sample(seconds, fraction):
        return GpuTelemetrySample(captured_at=(now + timedelta(seconds=seconds)).isoformat(),
                                  memory_used_mb=int(fraction * 1000), memory_total_mb=1000)
    for seconds in range(20):
        assert gate.update(sample(seconds, .89)) is None
        assert gate.is_open
    assert gate.update(sample(20, .90)) == "closed"
    assert not gate.is_open


def test_replay_choice_ignores_age_but_retains_explicit_priority():
    option = TraceBatchOption(batch_size=4, memory_mb=100, solo_seconds=10)
    a = TraceJob("a", -10000, 0, (option,))
    b = TraceJob("b", 0, 1, (option,))
    problem = TraceProblem((a, b), memory_budget_mb=1024, starvation_timeout_seconds=0)
    assert _time_aware_choice(problem, (a, b), 0).members[0].job_id == "b"
    swapped = (replace(a, release_seconds=0), replace(b, release_seconds=-10000))
    assert _time_aware_choice(replace(problem, jobs=swapped), swapped, 10000).members[0].job_id == "b"


def test_validated_busy_breaks_time_ties_heavy_when_idle_light_when_joining(tmp_path):
    _, _, planner, jobs = setup(tmp_path)
    class Predictor:
        available = True
        def predict_busy_percent_options(self, job, batches):
            return {batch: {"a": 20, "b": 80, "c": 50}[job.job_id] for batch in batches}
    planner.estimator.ml_predictor = Predictor()
    # Cold-start has no duration estimates here, so all time/priority terms tie.
    planner.predicted_remaining_runtime_seconds = lambda job, **kwargs: None
    for j in jobs: j.metadata["trial_allocator_supported"] = True
    idle = planner.choose_plan(jobs[:2], backend_available={"cuda_process": True})
    assert idle.job_ids == ("b",)
    joining = planner.choose_plan(jobs[:2], active_jobs=jobs[2:], backend_available={"cuda_process": True})
    assert joining.job_ids == ("a",)
    # Time remains authoritative when estimates are available.
    planner.predicted_remaining_runtime_seconds = lambda job, **kwargs: {"a": 1, "b": 10}[job.job_id]
    assert planner.choose_plan(jobs[:2], backend_available={"cuda_process": True}).job_ids == ("a",)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 101, "unknown"])
def test_invalid_busy_is_unknown_not_zero(tmp_path, value):
    _, _, planner, jobs = setup(tmp_path)
    planner.estimator.ml_predictor = SimpleNamespace(available=True,
        predict_busy_percent_options=lambda job, batches: {batch: value for batch in batches})
    assert planner.estimator.predicted_busy_percent(jobs[0]) is None
    assert planner.estimator.busy_rank(jobs[0], prefer_high=False) == (True, 0.0)


def test_trace_live_gate_matches_immediate_runtime_stop():
    from localml_scheduler.scheduler.trace_simulator import TraceMemorySample, _admission_open
    problem = TraceProblem((), memory_budget_mb=900,
        live_memory_samples=(TraceMemorySample(0, .89), TraceMemorySample(1, .90)))
    assert _admission_open(problem, 0)
    assert not _admission_open(problem, 1)
