"""CPU-only executable bracket trial lifecycle and accounting tests."""
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from localml_scheduler.domain import JobStatus
from localml_scheduler.scheduler.bracket_bootstrap import BracketBootstrapMixin, bracket_generation
from localml_scheduler.scheduler import bracket_bootstrap as module
from localml_scheduler.scheduler import v5_lifecycle
from localml_scheduler.scheduler.feedback_objective import CompletionCosts
from localml_scheduler.tests.test_cold_start_trials import setup


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


class Base:
    def _runnable_jobs(self):
        return [j for j in self.store.list_jobs() if j.job_id not in self.active and not j.status.is_terminal]

    def _supervisor_active_job_ids(self):
        return list(self.active)

    def _dispatch_plan(self, plan):
        self.plans.append(plan)
        self.active[:] = plan.job_ids
        for name in plan.job_ids:
            self.store.update_job(name, status=JobStatus.RUNNING, metadata_updates={
                "trial_runner_ready": False, "trial_step_window": {}, "trial_yield_ack": None})
        return True

    def _dispatch_pending_work(self):
        self.ordinary_dispatch_ticks += 1

    def _evaluate_colocation_trial(self):
        self.ordinary_trial_ticks += 1


class Service(BracketBootstrapMixin, Base):
    def _v5_bracket_startup_envelope(self, jobs, profiles):
        return self.envelope

    def _v5_bracket_compare(self, jobs, windows, profiles, remaining):
        self.comparisons.append(deepcopy(remaining))
        return {"decision": self.decision, "remaining_updates": deepcopy(remaining)}


def fixture(tmp_path, monkeypatch, count=3):
    settings, store, _, jobs = setup(tmp_path)
    settings.gpu_scheduler.feedback_latency.enabled = True
    clock = SimpleNamespace(now=1_800_000_000.)
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: clock.now))
    names = [j.job_id for j in jobs[:count]]
    request = {"request_id": "sealed", "workflow_id": "test", "completion_policy": "group",
        "required_job_ids": names, "membership_finalized": True, "submitted_at": iso(clock.now-10)}
    for job in jobs:
        store.update_job(job.job_id, status=JobStatus.READY if job.job_id in names else JobStatus.COMPLETED,
            metadata_updates={"feedback_request": request if job.job_id in names else None,
                "runtime_steps_per_epoch": 100, "runtime_global_step": 0,
                "runtime_total_optimizer_updates": 300, "runtime_planned_epochs": 3,
                "runtime_completion_identity": "runtime-"+job.job_id,
                "runtime_nonupdate_phase_counter": 0,
                "pilot_frozen_workload_sha256": "frozen-"+job.job_id,
                **({"feedback_ready_at": iso(clock.now-1)} if job.job_id not in names else {})})
        actual = store.get_job(job.job_id)
        actual.submitted_at = iso(clock.now-10)
        if job.job_id not in names:
            actual.finished_at = iso(clock.now-1)
        store.save_job(actual)
    profiles = {name: dict(kind="predicted_workload", artifact_sha256="frozen-artifact", hardware="A10",
        batch_size=64, job_signature=name, step_cost=1000., epoch_cost=100_000., avg_vram_mib=500.,
        peak_vram_mib=1000., busy_indicator=70.) for name in names}
    service = Service()
    service.settings, service.store, service.clock = settings, store, clock
    service.v5_options = SimpleNamespace(bracket=True, bracket_probe_seconds=30.)
    service._bracket_bootstrap = None
    service._bracket_bootstrap_restored = False
    service.active, service.plans, service.pauses, service.events, service.comparisons = [], [], [], [], []
    service.reaped = set()
    service._v5_bracket_launch_identity = lambda name: ({"group_id": "fixture-group", "job_id": name}
        if name in service.active or name in service.reaped else None)
    service._active_runs, service._colocation_trial, service._device_samples = {}, None, []
    service._admission_gate = SimpleNamespace(is_open=True)
    service.ordinary_dispatch_ticks = service.ordinary_trial_ticks = 0
    service.envelope, service.decision = {"peak_mib": 4000., "source": "measured-startup-plus-headroom"}, "accepted"
    service.event_logger = SimpleNamespace(emit=lambda event, **kw: service.events.append((event, kw)))
    def pause(name, **kw):
        service.pauses.append(name)
        return True
    service.supervisor = SimpleNamespace(available_backends=lambda: {"cuda_process": True}, request_pause=pause,
        active_job_ids=lambda: list(service.active), has_live_process=lambda name: name in service.active,
        has_reaped_process=lambda name: name in service.reaped,
        observed_exit_code=lambda name: 0 if name in service.reaped else None)
    service.planner = SimpleNamespace(estimator=SimpleNamespace(
        safe_budget_mb=lambda: 5000., predicted_profile=lambda j: deepcopy(profiles[j.job_id])),
        compatibility=SimpleNamespace(pack_eligible=lambda *a, **kw: True, compatible_group=lambda *a, **kw: True))
    service.profiles = profiles
    return service


def launch(service):
    service._dispatch_pending_work()
    assert service._bracket_bootstrap.phase == "prepare"
    assert len(service.plans) == 1
    assert not service.plans[0].trial_metadata["requires_live_trial"]
    assert not service.plans[0].trial_metadata.get("preexisting_job_ids")
    return service._bracket_bootstrap


def warmup(service):
    state = service._bracket_bootstrap
    for name in state.job_ids:
        job = service.store.get_job(name)
        service.store.update_job(name, metadata_updates={"trial_runner_ready": True,
            "trial_yield_ack": job.metadata["trial_step_command"]["token"], "runtime_global_step": 1})
    service._evaluate_colocation_trial()
    assert state.phase == "warmup"
    service.clock.now += .1
    for name in state.job_ids:
        job = service.store.get_job(name)
        service.store.update_job(name, metadata_updates={"runtime_global_step": 3,
            "trial_measurement_progress": {"token": job.metadata["trial_step_command"]["token"],
                "started_at": service.clock.now, "global_step": 3}})
    service._evaluate_colocation_trial()
    assert state.phase == "measure"


def windows(service, *, bad=None, duration=2.6):
    state = service._bracket_bootstrap
    started = service.clock.now+.01
    for name in state.job_ids:
        job = service.store.get_job(name)
        window = {"owner": state.trial_id, "token": job.metadata["trial_step_command"]["token"],
            "kind": "bracket_measure", "elapsed_seconds": duration,
            "complete": True, "steps": 8, "start_step": 5, "end_step": 13,
            "halves": [.3, .35], "excludes_nonupdate_phases": True,
            "completion_identity": "runtime-"+name, "nonupdate_phase_counter": 0,
            "completed_nonupdate_phase_counter": 0, "started_at": started, "finished_at": iso(started+duration)}
        if bad:
            bad(name, window)
        service.store.update_job(name, metadata_updates={"runtime_global_step": 13, "trial_step_window": window})
    service.clock.now = started+max(3, duration+.1)
    service._evaluate_colocation_trial()


def barrier(service, *, step=17):
    state = service._bracket_bootstrap
    for name in state.job_ids:
        token = service.store.get_job(name).metadata["trial_step_command"]["token"]
        current_step = step[name] if isinstance(step, dict) else step
        service.store.update_job(name, metadata_updates={"runtime_global_step": current_step,
            "trial_yield_ack": token, "trial_yield_snapshot": {"token": token, "owner": state.trial_id,
                "completion_identity": "runtime-"+name, "global_step": current_step, "steps_per_epoch": 100,
                "total_optimizer_updates": 300, "planned_epochs": 3}})
    service._evaluate_colocation_trial()


def test_actual_full_group_trial_two_halves_and_same_postprobe_work(tmp_path, monkeypatch):
    service = fixture(tmp_path, monkeypatch)
    state = launch(service)
    warmup(service)
    windows(service)
    assert state.phase == "barrier"
    assert service.comparisons == []
    barrier(service)
    assert service.comparisons == [{"a": 283, "b": 283, "c": 283}]
    assert service._bracket_bootstrap is None
    assert service.pauses == []
    result = service.store.get_job("a").metadata["v5_bracket_bootstrap"]["result"]
    assert result["decision"] == "accepted"
    assert result["productive_updates"] == {"a": 17, "b": 17, "c": 17}
    assert set(result["measurement_windows"]) == {"a", "b", "c"}
    assert all(not service.store.get_job(name).metadata["trial_step_command"] for name in state.job_ids)


@pytest.mark.parametrize("blocker", ["unsealed", "unready", "startup_unknown", "startup_too_small",
    "startup_oversized", "occupied", "incompatible", "profile_mixed", "reservation", "solo_required"])
def test_guarded_fast_path_falls_back_without_an_implicit_sibling_barrier(tmp_path, monkeypatch, blocker):
    service = fixture(tmp_path, monkeypatch)
    if blocker == "unsealed":
        for name in ("a", "b", "c"):
            request = service.store.get_job(name).metadata["feedback_request"]
            request["membership_finalized"] = False
            service.store.update_job(name, metadata_updates={"feedback_request": request})
    elif blocker == "unready":
        service._runnable_jobs = lambda: [service.store.get_job("a"), service.store.get_job("b")]
    elif blocker == "startup_unknown": service.envelope = None
    elif blocker == "startup_too_small": service.envelope["peak_mib"] = 2500.
    elif blocker == "startup_oversized": service.envelope["peak_mib"] = 6000.
    elif blocker == "occupied": service.active = ["resident"]
    elif blocker == "incompatible": service.planner.compatibility.compatible_group = lambda *a, **k: False
    elif blocker == "profile_mixed": service.profiles["b"]["artifact_sha256"] = "different"
    elif blocker == "reservation": service._v5_bracket_reservation_allows = lambda _: False
    elif blocker == "solo_required": service.store.update_job("a", metadata_updates={"cold_start_solo_required": True})
    service._dispatch_pending_work()
    assert service._bracket_bootstrap is None
    assert not service.plans
    assert service.ordinary_dispatch_ticks == 1


@pytest.mark.parametrize("decision", ["rejected", "inconclusive"])
def test_cleanup_preserves_progress_and_drains_before_solo_continuation(tmp_path, monkeypatch, decision):
    service = fixture(tmp_path, monkeypatch)
    service.decision = decision
    state = launch(service)
    warmup(service)
    windows(service)
    barrier(service)
    assert state.phase == "cleanup"
    assert service.pauses == ["a", "b", "c"]
    for name in state.job_ids:
        job = service.store.get_job(name)
        assert job.metadata["runtime_global_step"] == 17
        assert job.metadata["cold_start_solo_required"] is True
        assert job.hold is False
        assert job.status == JobStatus.PAUSING
    service._dispatch_pending_work()
    assert service.ordinary_dispatch_ticks == 0
    service.reaped.update(service.active)
    service.active.clear()
    service._evaluate_colocation_trial()
    assert service._bracket_bootstrap is None
    service._dispatch_pending_work()
    assert service.ordinary_dispatch_ticks == 1


@pytest.mark.parametrize("change", ["nonupdate", "identity", "recipe", "settings", "membership", "disjoint"])
def test_invalid_evidence_cannot_become_measured_admission(tmp_path, monkeypatch, change):
    service = fixture(tmp_path, monkeypatch)
    state = launch(service)
    warmup(service)
    if change == "nonupdate": service.store.update_job("b", metadata_updates={"runtime_nonupdate_phase_counter": 1})
    if change == "identity": service.store.update_job("b", metadata_updates={"runtime_completion_identity": "new"})
    if change == "recipe": service.store.update_job("b", metadata_updates={"pilot_frozen_workload_sha256": "new"})
    if change == "settings": service.settings.gpu_scheduler.feedback_latency.min_gain_seconds += 1
    if change == "membership": service.active.remove("b")
    if change == "disjoint":
        windows(service, bad=lambda name, w: w.update(started_at=w["started_at"]+10, finished_at=iso(w["started_at"]+13)) if name == "b" else None)
    else:
        service._evaluate_colocation_trial()
    assert state.phase == "cleanup"
    assert state.result["decision"] == "inconclusive"
    assert service.comparisons == []


def test_missing_or_forged_half_stays_pending_then_budget_ends_once(tmp_path, monkeypatch):
    service = fixture(tmp_path, monkeypatch)
    state = launch(service)
    warmup(service)
    windows(service, bad=lambda name, w: w.update(halves=[float("nan"), .3]) if name == "b" else None)
    assert state.phase == "measure"
    assert not service.comparisons
    service.clock.now = state.deadline_at
    service._evaluate_colocation_trial()
    assert state.phase == "cleanup"
    assert "allowance" in state.result["reason"]
    assert service.pauses == ["a", "b", "c"]
    service._evaluate_colocation_trial()
    assert service.pauses == ["a", "b", "c"]


def test_legitimate_zero_remaining_finishes_without_pause_or_failure(tmp_path, monkeypatch):
    service = fixture(tmp_path, monkeypatch, count=2)
    state = launch(service)
    warmup(service)
    service.store.update_job("a", metadata_updates={"runtime_global_step": 300})
    service._evaluate_colocation_trial()
    assert state.phase == "finishing"
    assert not service.pauses
    assert not service.store.get_job("a").metadata["trial_step_command"]
    service._dispatch_pending_work()
    assert service.ordinary_dispatch_ticks == 0  # GPU not released by R=0.
    service.active.remove("a")
    service.reaped.add("a")
    service._evaluate_colocation_trial()
    assert service._bracket_bootstrap is None
    assert not service.pauses
    assert service.store.get_job("a").metadata["runtime_global_step"] == 300
    assert state.result["decision"] == "completed_unverified"
    assert not service.store.get_job("b").metadata.get("cold_start_solo_required")


def test_changed_triple_membership_has_bounded_continuation_without_inherited_pair_rates(tmp_path, monkeypatch):
    service = fixture(tmp_path, monkeypatch)
    state = launch(service)
    warmup(service)
    service.store.update_job("a", metadata_updates={"runtime_global_step": 300})
    service._evaluate_colocation_trial()
    service.active.remove("a")
    service.reaped.add("a")
    service._evaluate_colocation_trial()
    assert not service.pauses
    service.clock.now = state.deadline_at
    service._evaluate_colocation_trial()
    assert service.pauses == ["b", "c"]
    assert service.store.get_job("a").metadata["runtime_global_step"] == 300
    assert "a" not in state.result["cleanup_job_ids"]
    assert service.comparisons == []


def test_unchanged_generation_and_combined_job_attempt_limits_are_persistent(tmp_path, monkeypatch):
    service = fixture(tmp_path, monkeypatch)
    selected = service._bracket_ready_group()
    assert selected
    generation = selected[2]
    service.store.update_job("a", metadata_updates={"v5_bracket_generations": [generation]})
    assert service._bracket_ready_group() is None
    service.store.update_job("a", metadata_updates={"v5_bracket_generations": [],
        "v5_bracket_probe_attempts": 1,
        "cold_start_failed_attempts": service.settings.gpu_scheduler.colocation.cold_start_max_attempts-1})
    assert service._bracket_ready_group() is None


def test_restart_cannot_replay_stale_full_group_measurement(tmp_path, monkeypatch):
    service = fixture(tmp_path, monkeypatch)
    state = launch(service)
    service._bracket_bootstrap = None
    service._bracket_bootstrap_restored = False
    service._bracket_restore()
    assert service._bracket_bootstrap.phase == "cleanup"
    assert service._bracket_bootstrap.cleanup_deadline_at == state.cleanup_deadline_at
    assert service.pauses == ["a", "b", "c"]
    for name in state.job_ids:
        metadata = service.store.get_job(name).metadata
        assert metadata["v5_bracket_probe_attempts"] == 1
        assert metadata["v5_bracket_bootstrap"]["phase"] == "cleanup"
        assert metadata["cold_start_solo_required"]
        assert not metadata["trial_step_command"]


def test_default_forecast_has_explicit_cleanup_costs_and_same_remaining(tmp_path, monkeypatch):
    service = fixture(tmp_path, monkeypatch, count=2)
    state = launch(service)
    warmup(service)
    windows(service)
    monkeypatch.setattr(v5_lifecycle, "phase_costs", lambda *_: (CompletionCosts(startup=2, pause=1, resume=2, finish=3, delivery=4), "test"))
    result = service._bracket_compare([service.store.get_job(n) for n in state.job_ids], state.windows,
        state.profiles, {"a": 200, "b": 200})
    assert result["decision"] == "accepted"
    assert len(result["halves"]) == 2
    assert result["error"] is None
    for half in result["halves"]:
        assert any(e["event"] == "pause_release" for e in half["defer_trace"]["events"])
        assert all(half[name+"_trace"]["status"] == "complete" for name in ("admit", "defer"))


def test_disabled_flag_leaves_original_service_methods_in_control(tmp_path, monkeypatch):
    service = fixture(tmp_path, monkeypatch)
    service.v5_options.bracket = False
    service._dispatch_pending_work()
    service._evaluate_colocation_trial()
    assert service.ordinary_dispatch_ticks == service.ordinary_trial_ticks == 1
    assert service._bracket_bootstrap is None


def test_real_dispatch_path_starts_all_workers_without_a_fabricated_add_only_trial(tmp_path, monkeypatch):
    from localml_scheduler.scheduler.experimental_v5 import V5Options, V5Service
    original = fixture(tmp_path, monkeypatch)
    active, launches = [], []
    def dispatch(jobs, **kwargs):
        active[:] = [j.job_id for j in jobs]
        launches.append((active[:], kwargs))
        return SimpleNamespace(can_run=True, group_id="actual-group")
    supervisor = SimpleNamespace(dispatch=dispatch, active_job_ids=lambda: list(active),
        available_backends=lambda: {"cuda_process": True})
    service = V5Service(original.settings, store=original.store, supervisor=supervisor,
        v5_options=V5Options(bracket=True))
    service.planner.estimator.predicted_profile = lambda j: deepcopy(original.profiles[j.job_id])
    service._v5_bracket_startup_envelope = lambda *_: original.envelope
    service._preload_job_baseline = lambda *_: None
    service._log_run_group_open = lambda *_args, **_kwargs: None
    service._dispatch_pending_work()
    assert launches == [(["a", "b", "c"], {"mode": "concurrent_group", "backend_name": "cuda_process",
        "batch_overrides": {}, "fallback_order": []})]
    assert service._colocation_trial is None
    assert service._bracket_bootstrap.phase == "prepare"
    assert service._active_runs["actual-group"].job_ids == ("a", "b", "c")
    for name in active:
        job = service.store.get_job(name)
        assert job.status == JobStatus.RUNNING
        assert job.metadata["trial_step_command"]["action"] == "yield"
        assert job.metadata["trial_step_command"]["owner"] == service._bracket_bootstrap.trial_id
        assert job.metadata["trial_runner_ready"] is False


@pytest.mark.parametrize("change", ["short", "wrong_token", "wrong_kind", "one_half"])
def test_clean_window_minimum_duration_and_command_binding_are_required(tmp_path, monkeypatch, change):
    service = fixture(tmp_path, monkeypatch)
    state = launch(service)
    warmup(service)
    def invalidate(name, window):
        if name != "b":
            return
        if change == "short": window["elapsed_seconds"] = .01
        if change == "wrong_token": window["token"] = "previous-command"
        if change == "wrong_kind": window["kind"] = "reference"
        if change == "one_half": window["halves"] = [.3]
    windows(service, bad=invalidate)
    assert state.phase == "measure"
    assert not service.comparisons
    assert not service.pauses
    service.clock.now = state.deadline_at
    service._evaluate_colocation_trial()
    assert state.phase == "cleanup"
    assert state.result["decision"] == "inconclusive"


def test_ignored_pause_has_a_persisted_cleanup_deadline_and_explicit_episode_failure(tmp_path, monkeypatch):
    service = fixture(tmp_path, monkeypatch)
    service.decision = "rejected"
    state = launch(service)
    warmup(service); windows(service); barrier(service)
    assert state.phase == "cleanup"
    deadline = state.cleanup_deadline_at
    assert service.store.get_job("a").metadata["v5_bracket_bootstrap"]["cleanup_deadline_at"] == deadline
    service.clock.now = deadline
    with pytest.raises(TimeoutError, match="no resource release is claimed"):
        service._evaluate_colocation_trial()
    assert service._bracket_bootstrap is state
    assert service.v5_failure["pending_job_ids"] == ["a", "b", "c"]
    assert not service.reaped
    service._dispatch_pending_work()
    assert service.ordinary_dispatch_ticks == 0


def test_restart_preserves_live_drain_and_original_deadline_even_if_supervisor_forgets_workers(tmp_path, monkeypatch):
    service = fixture(tmp_path, monkeypatch)
    state = launch(service)
    original_deadline = state.cleanup_deadline_at
    service.clock.now += 10
    service._bracket_bootstrap = None
    service._bracket_bootstrap_restored = False
    service._bracket_restore()
    restored = service._bracket_bootstrap
    assert restored.phase == "cleanup"
    assert restored.cleanup_deadline_at == original_deadline
    assert set(restored.live_identities) == {"a", "b", "c"}
    # Absence from a replacement supervisor is not observed process exit.
    service.active.clear()
    service._evaluate_colocation_trial()
    service._dispatch_pending_work()
    assert service._bracket_bootstrap is restored
    assert service.ordinary_dispatch_ticks == 0
    service.clock.now += 10
    service._bracket_bootstrap = None
    service._bracket_bootstrap_restored = False
    service._bracket_restore()
    assert service._bracket_bootstrap.cleanup_deadline_at == original_deadline
    service.reaped.update(("a", "b", "c"))
    service._evaluate_colocation_trial()
    assert service._bracket_bootstrap is None


def test_direct_group_launch_honors_configured_parallel_job_cap(tmp_path, monkeypatch):
    service = fixture(tmp_path, monkeypatch)
    service.settings.gpu_scheduler.parallel_job_cap = 2
    service._dispatch_pending_work()
    assert service.plans == []
    assert service._bracket_bootstrap is None
    assert service.ordinary_dispatch_ticks == 1


@pytest.mark.parametrize("status", [JobStatus.FAILED, JobStatus.CANCELLED])
def test_terminal_failure_is_not_legitimate_finishing_and_preserves_failure_reason(tmp_path, monkeypatch, status):
    service = fixture(tmp_path, monkeypatch)
    state = launch(service)
    warmup(service)
    service.store.update_job("a", status=status, reason="actual workload failure", metadata_updates={"runtime_global_step": 3})
    service._evaluate_colocation_trial()
    assert state.phase == "cleanup"
    assert state.result["decision"] == "inconclusive"
    assert state.result["failed_members"]["a"] == {"status": status.value, "reason": "actual workload failure"}
    assert service.pauses == ["b", "c"]
    assert service.store.get_job("a").status == status
    assert service.store.get_job("b").metadata["runtime_global_step"] == 3


def test_default_unequal_triple_forecast_is_explicitly_inconclusive_for_unmeasured_pair(tmp_path, monkeypatch):
    service = fixture(tmp_path, monkeypatch)
    state = launch(service)
    warmup(service); windows(service)
    monkeypatch.setattr(v5_lifecycle, "phase_costs", lambda *_: (CompletionCosts(pause=1, resume=2, finish=3, delivery=4), "test"))
    result = service._bracket_compare([service.store.get_job(n) for n in state.job_ids], state.windows,
        state.profiles, {"a": 20, "b": 200, "c": 250})
    assert result["decision"] == "inconclusive"
    assert "unknown exact training group" in result["error"]
    assert "b" in result["error"] and "c" in result["error"]
    assert result["failed_trace"]["status"] == "inconclusive"
    assert result["halves"] == []


def test_direct_launch_rechecks_cap_after_selection(tmp_path, monkeypatch):
    service = fixture(tmp_path, monkeypatch)
    selected = service._bracket_ready_group()
    assert selected is not None
    service.settings.gpu_scheduler.parallel_job_cap = 2
    assert service._bracket_launch(selected) is False
    assert not service.plans
    assert service._bracket_bootstrap is None


def test_default_forecast_retains_successful_exit_without_receipt_as_passive_delivery(tmp_path, monkeypatch):
    service = fixture(tmp_path, monkeypatch, count=2)
    state = launch(service)
    warmup(service); windows(service)
    service.store.update_job("c", metadata_updates={"feedback_ready_at": None, "runtime_global_step": 300})
    service.reaped.add("c")
    service.profiles["c"] = {**deepcopy(service.profiles["a"]), "job_signature": "c"}
    monkeypatch.setattr(v5_lifecycle, "phase_costs", lambda *_: (CompletionCosts(pause=1, resume=2, finish=3, delivery=7), "test"))
    result = service._bracket_compare([service.store.get_job(n) for n in state.job_ids], state.windows,
        state.profiles, {"a": 200, "b": 200})
    assert result["error"] is None
    for half in result["halves"]:
        for action in ("admit", "defer"):
            trace = half[action+"_trace"]
            assert trace["completions"]["c"] == 7
            assert all("c" not in phase["resource_holders"] for phase in trace["phases"])
            assert not any(e["job"] == "c" and e["event"] in {"start", "resume"} for e in trace["events"])


def integrated_bootstrap_service(tmp_path, monkeypatch, count):
    from localml_scheduler.scheduler.experimental_v5 import V5Options, V5Service
    from localml_scheduler.scheduler import experimental_v5, reference_execution
    original = fixture(tmp_path, monkeypatch, count=count)
    fake_clock = SimpleNamespace(time=lambda: original.clock.now, monotonic=lambda: original.clock.now)
    monkeypatch.setattr(reference_execution, "time", fake_clock)
    monkeypatch.setattr(experimental_v5, "time", fake_clock)
    original.settings.gpu_scheduler.enabled = True
    active, exited, launches, pauses = [], {}, [], []
    def dispatch(jobs, **kwargs):
        names = [j.job_id for j in jobs]
        for name in names:
            exited.pop(name, None)
        active.extend(names)
        launches.append((names, kwargs))
        return SimpleNamespace(can_run=True, group_id="actual-run-"+str(len(launches)))
    def pause(name, **kwargs):
        pauses.append(name)
        return True
    supervisor = SimpleNamespace(dispatch=dispatch, active_job_ids=lambda: list(active),
        available_backends=lambda: {"cuda_process": True}, request_pause=pause,
        has_live_process=lambda name: name in active, has_reaped_process=lambda name: name in exited,
        observed_exit_code=lambda name: exited.get(name), active_job_ids_by_group=lambda:
            {key: [name for name in run.job_ids if name in active] for key, run in service._active_runs.items()})
    service = V5Service(original.settings, store=original.store, supervisor=supervisor,
        v5_options=V5Options(bracket=True, continuation=False))
    service.planner.estimator.predicted_profile = lambda j, *a, **k: deepcopy(original.profiles.get(j.job_id))
    service.planner.estimator.estimate_peak_vram_mb = lambda j, *a, **k: original.profiles[j.job_id]["peak_vram_mib"]
    service.planner.estimator.safe_budget_mb = lambda: 5000.
    service._v5_bracket_startup_envelope = lambda *_: original.envelope
    service._active_vram_occupancy = lambda: 1000.*len(active)
    service._preload_job_baseline = lambda *_: None
    service._log_run_group_open = lambda *_args, **_kwargs: None
    service.event_logger = original.event_logger
    service.clock, service.profiles = original.clock, original.profiles
    service.active, service.exited, service.launches, service.pauses = active, exited, launches, pauses
    service.events = original.events
    monkeypatch.setattr(v5_lifecycle, "phase_costs", lambda *_: (CompletionCosts(startup=2, pause=1, resume=2, finish=3, delivery=4), "test"))
    return service


@pytest.mark.parametrize("outcome", ["accepted", "rejected", "inconclusive"])
def test_c_only_bootstrap_installs_and_executes_its_actual_reference_continuation(tmp_path, monkeypatch, outcome):
    count = 3 if outcome == "inconclusive" else 2
    service = integrated_bootstrap_service(tmp_path, monkeypatch, count)
    service._dispatch_pending_work()
    state = service._bracket_bootstrap
    assert service.launches[0][0] == list(state.job_ids)
    warmup(service)
    if outcome == "rejected":
        windows(service, bad=lambda _name, w: w.update(halves=[3., 4.]), duration=28.)
    else:
        windows(service)
    barrier(service, step={"a": 280, "b": 100, "c": 50} if outcome == "inconclusive" else 17)
    reference = service._v5_reference_state
    assert reference is not None
    assert reference["origin"] == "group_bootstrap"
    assert reference["group_members"] == list(state.job_ids)
    assert reference["bootstrap_generation"] == state.generation
    assert reference["score_decision"] == outcome
    assert reference["order"] == list(state.job_ids)
    assert reference["reconciliation_deadline_at"] == state.cleanup_deadline_at
    assert set(reference["profiles"]) == set(reference["cohort"]) == set(state.job_ids)
    assert "candidate_job_id" not in reference and "preexisting_job_ids" not in reference
    assert service.v5_options.continuation is False  # C never enables A scoring.
    assert not service._v5_reference_begin(service.store.list_jobs(), {"decision": "accepted"}, None)
    if outcome == "accepted":
        assert reference["chosen_action"] == "admit"
        assert reference["cleanup_members"] == []
        assert reference["prediction_complete"]
        assert service.pauses == []
        assert service._bracket_bootstrap is None
        service._dispatch_pending_work()
        assert len(service.launches) == 1
        return
    assert reference["chosen_action"] == "defer"
    assert reference["cleanup_members"] == list(state.job_ids)
    assert reference["execution_concurrency_limit"] == 1
    assert reference["prediction_complete"] is (outcome == "rejected")
    assert service.pauses == list(state.job_ids)
    service._dispatch_pending_work()
    assert len(service.launches) == 1  # Physical bootstrap still owns cleanup.
    assert any(event == "v5_reference_waiting_release" for event, _ in service.events)
    for name in state.job_ids:
        service.active.remove(name)
        service.exited[name] = 0
        job = service.store.get_job(name)
        job.latest_checkpoint_path = "/tmp/retained-"+name+".pt"
        job.status = JobStatus.PAUSED
        service.store.save_job(job)
    service._active_runs.clear()
    service._evaluate_colocation_trial()
    assert service._bracket_bootstrap is None
    service._dispatch_pending_work()
    assert service.launches[-1][0] == ["a"]
    assert service.store.get_job("a").metadata["placement_trial_metadata"]["reference_execution"]
    assert service.store.get_job("a").latest_checkpoint_path == "/tmp/retained-a.pt"
    assert service._colocation_trial is None
    service._dispatch_pending_work()
    assert len(service.launches) == 2  # No undeclared packed retry around a.
    service.active.remove("a"); service.exited["a"] = 0; service._active_runs.clear()
    service.store.update_job("a", status=JobStatus.COMPLETED, metadata_updates={
        "runtime_global_step": 300, "feedback_ready_at": iso(service.clock.now)})
    service._dispatch_pending_work()
    assert service.launches[-1][0] == ["b"]
    assert service.store.get_job("b").metadata["placement_trial_metadata"]["reference_execution"]
    assert service.store.get_job("b").latest_checkpoint_path == "/tmp/retained-b.pt"
    assert service._colocation_trial is None


def test_bootstrap_missing_full_cohort_profile_refuses_reference_installation_safely(tmp_path, monkeypatch):
    service = integrated_bootstrap_service(tmp_path, monkeypatch, 2)
    service._dispatch_pending_work()
    state = service._bracket_bootstrap
    service.profiles["b"] = None
    assert not service._v5_reference_begin_bootstrap(
        [service.store.get_job(name) for name in state.job_ids], {"decision": "inconclusive"}, state)
    assert service._v5_reference_state is None
    assert any(event == "v5_reference_not_started" and "profile" in item["payload"]["reason"]
        for event, item in service.events)
    service._bracket_close("inconclusive", "missing profile comparison", cleanup=True)
    assert service._bracket_bootstrap.phase == "cleanup"
    assert service.pauses == ["a", "b"]
