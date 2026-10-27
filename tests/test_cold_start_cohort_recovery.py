"""Fresh admission credit requires durable proof that failed incumbents retired."""
from copy import deepcopy
from unittest.mock import Mock

import pytest

from localml_scheduler.domain import JobStatus
from localml_scheduler.execution.executor import WorkerProcessHandle
from localml_scheduler.scheduler.cold_start import cold_group_blocked
from localml_scheduler.scheduler.service import SchedulerService
from localml_scheduler.scheduler.supervisor import ManagedWorker, PlacementGroupHandle, WorkerSupervisor
from localml_scheduler.storage.sqlite_store import SQLiteStateStore
from localml_scheduler.tests.test_cold_start_trials import setup
from localml_scheduler.tests.test_cold_start_liveness import pending_trial


AVAILABLE = {"cuda_process": True}


def system(tmp_path):
    settings, store, planner, jobs = setup(tmp_path)
    supervisor = Mock()
    supervisor.active_job_ids.return_value = []
    supervisor.has_live_process.return_value = False
    supervisor.has_reaped_process.return_value = True
    supervisor.request_pause.return_value = True
    service = SchedulerService(settings, store=store, supervisor=supervisor)
    service.planner = planner
    service.event_logger = Mock()
    return settings, store, planner, service, jobs


def exhausted(store, candidate="b", incumbents=("a",), **cohort_changes):
    cohort = {"generation": 0, "failed_attempts": 3, "proof_complete": True,
              "incumbent_job_ids": list(incumbents), "retired_incumbent_job_ids": []}
    cohort.update(cohort_changes)
    return store.update_job(candidate, status=JobStatus.PAUSED, hold=False, metadata_updates={
        "cold_start_failed_attempts": 3, "cold_start_solo_required": True,
        "cold_start_failure_cohort": cohort,
    })


def refresh(service, store, candidate="b", active=()):
    service.supervisor.active_job_ids.return_value = list(active)
    return service._refresh_cold_start_cohorts([store.get_job(candidate)], set(active))


@pytest.mark.parametrize("status", [JobStatus.PENDING, JobStatus.READY, JobStatus.RUNNING,
                                    JobStatus.PAUSING, JobStatus.PAUSED, JobStatus.RECOVERABLE])
def test_temporarily_absent_incumbent_cannot_renew_credit(tmp_path, status):
    _, store, _, service, _ = system(tmp_path)
    exhausted(store)
    store.update_job("a", status=status)
    for _ in range(3):
        refresh(service, store)  # Physically absent is insufficient.
    assert store.get_job("b").metadata["cold_start_failure_cohort"]["failed_attempts"] == 3
    assert store.get_job("b").metadata["cold_start_solo_required"]
    service.event_logger.emit.assert_not_called()


@pytest.mark.parametrize("terminal", [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED])
def test_every_contributor_must_be_terminal_and_physically_reaped(tmp_path, terminal):
    _, store, _, service, _ = system(tmp_path)
    exhausted(store, incumbents=("a", "c"))
    store.update_job("a", status=terminal)
    store.update_job("c", status=JobStatus.PAUSED)
    refresh(service, store)
    assert store.get_job("b").metadata["cold_start_solo_required"]
    store.update_job("c", status=terminal)
    service.supervisor.has_live_process.side_effect = lambda name: name == "c"
    refresh(service, store)  # monitor_via_store may omit a terminal but live parent.
    assert store.get_job("b").metadata["cold_start_solo_required"]
    service.supervisor.has_live_process.side_effect = None
    service.supervisor.has_live_process.return_value = False
    refresh(service, store, active=("c",))
    assert store.get_job("b").metadata["cold_start_solo_required"]
    current = refresh(service, store)[0]
    assert not current.metadata["cold_start_solo_required"]
    assert current.metadata["cold_start_failure_cohort"]["retired_incumbent_job_ids"] == ["a", "c"]


@pytest.mark.parametrize("case", ["running", "pausing", "held", "candidate_process", "active", "pending_trial"])
def test_protected_or_operator_paused_candidate_is_never_renewed(tmp_path, case):
    _, store, _, service, _ = system(tmp_path)
    exhausted(store)
    store.update_job("a", status=JobStatus.COMPLETED)
    if case in {"running", "pausing"}:
        store.update_job("b", status=JobStatus(case.upper()))
    elif case == "held":
        store.update_job("b", hold=True)
    elif case == "candidate_process":
        service.supervisor.has_live_process.side_effect = lambda name: name == "b"
    elif case == "pending_trial":
        service._colocation_trial = Mock()
    refresh(service, store, active=("b",) if case == "active" else ())
    assert store.get_job("b").metadata["cold_start_solo_required"]
    assert store.get_job("b").metadata["cold_start_failure_cohort"]["generation"] == 0


@pytest.mark.parametrize("incumbents", [(), ("missing-job",), ("a", "b")])
def test_empty_missing_or_self_contributor_proof_is_conservative(tmp_path, incumbents):
    _, store, _, service, _ = system(tmp_path)
    exhausted(store, incumbents=incumbents)
    store.update_job("a", status=JobStatus.COMPLETED)
    refresh(service, store)
    assert store.get_job("b").metadata["cold_start_solo_required"]


def test_poll_and_sqlite_restart_cannot_mint_duplicate_credit(tmp_path):
    settings, store, _, service, _ = system(tmp_path)
    stale = exhausted(store)
    store.update_job("a", status=JobStatus.COMPLETED)
    for _ in range(5):
        service._refresh_cold_start_cohorts([stale], set())
    assert service.event_logger.emit.call_count == 1
    reopened = SQLiteStateStore(settings)
    restarted = SchedulerService(settings, store=reopened, supervisor=service.supervisor)
    restarted.event_logger = Mock()
    for _ in range(3):
        restarted._refresh_cold_start_cohorts([stale], set())
    current = reopened.get_job("b")
    assert current.metadata["cold_start_failed_attempts"] == 3  # Lifetime count remains.
    assert current.metadata["cold_start_failure_cohort"] == {
        "generation": 1, "failed_attempts": 0, "proof_complete": True,
        "incumbent_job_ids": [], "retired_incumbent_job_ids": ["a"],
    }
    restarted.event_logger.emit.assert_not_called()
    # Even an exhausted later snapshot cannot reuse the same retired identity.
    exhausted(reopened, retired_incumbent_job_ids=["a"], generation=1)
    refresh(restarted, reopened)
    assert reopened.get_job("b").metadata["cold_start_failure_cohort"]["generation"] == 1


def test_mixed_consumed_and_new_contributors_need_new_terminal_proof(tmp_path):
    _, store, _, service, _ = system(tmp_path)
    exhausted(store, incumbents=("a", "c"), retired_incumbent_job_ids=["a"], generation=1)
    store.update_job("a", status=JobStatus.COMPLETED)
    store.update_job("c", status=JobStatus.RECOVERABLE)
    refresh(service, store)
    assert store.get_job("b").metadata["cold_start_failure_cohort"]["generation"] == 1
    store.update_job("c", status=JobStatus.COMPLETED)
    cohort = refresh(service, store)[0].metadata["cold_start_failure_cohort"]
    assert cohort["generation"] == 2
    assert cohort["retired_incumbent_job_ids"] == ["a", "c"]
    assert cohort["failed_attempts"] == 0


def tracked_worker(supervisor, tmp_path, name, *, returncode, monitor=True):
    process = Mock()
    process.poll.return_value = returncode
    handle = WorkerProcessHandle(name, process, tmp_path / "stdout", tmp_path / "stderr", monitor)
    supervisor._groups[name] = PlacementGroupHandle(
        name, "stack_anchor", "cuda_process", workers={name: ManagedWorker(name, handle, "solo")})
    return handle


def test_fresh_supervisor_needs_positive_candidate_and_incumbent_exit_proof(tmp_path):
    settings, store, _, service, _ = system(tmp_path)
    exhausted(store)
    store.update_job("a", status=JobStatus.COMPLETED)
    supervisor = WorkerSupervisor(settings, store=SQLiteStateStore(settings))
    service.supervisor = supervisor
    assert supervisor.active_job_ids() == [] and not supervisor.has_live_process("a")
    service._refresh_cold_start_cohorts([store.get_job("b")], set())
    assert store.get_job("b").metadata["cold_start_solo_required"]
    tracked_worker(supervisor, tmp_path, "a", returncode=0)
    supervisor.poll()
    assert supervisor.has_reaped_process("a") and not supervisor.has_reaped_process("b")
    service._refresh_cold_start_cohorts([store.get_job("b")], set())
    assert store.get_job("b").metadata["cold_start_solo_required"]  # Candidate absence is also unknown.
    tracked_worker(supervisor, tmp_path, "b", returncode=0)
    supervisor.poll()
    assert not service._refresh_cold_start_cohorts([store.get_job("b")], set())[0].metadata["cold_start_solo_required"]


@pytest.mark.parametrize("monitor", [False, True])
def test_real_supervisor_terminal_parent_must_exit_before_receipt_and_renewal(tmp_path, monitor):
    settings, store, _, service, _ = system(tmp_path)
    exhausted(store)
    store.update_job("a", status=JobStatus.COMPLETED)
    supervisor = WorkerSupervisor(settings, store=store)
    service.supervisor = supervisor
    tracked_worker(supervisor, tmp_path, "b", returncode=0)
    incumbent = tracked_worker(supervisor, tmp_path, "a", returncode=None, monitor=monitor)
    supervisor.poll()
    assert supervisor.has_reaped_process("b") and not supervisor.has_reaped_process("a")
    assert supervisor.has_live_process("a")
    service._refresh_cold_start_cohorts([store.get_job("b")], set(supervisor.active_job_ids()))
    assert store.get_job("b").metadata["cold_start_solo_required"]
    incumbent.process.poll.return_value = 0
    supervisor.poll()
    assert supervisor.has_reaped_process("a") and not supervisor.has_live_process("a")
    assert not service._refresh_cold_start_cohorts([store.get_job("b")], set())[0].metadata["cold_start_solo_required"]


def test_supervisor_exit_receipt_is_revoked_before_redispatch(tmp_path):
    settings, store, _, _, _ = system(tmp_path)
    backend = Mock()
    backend.available.return_value = True
    supervisor = WorkerSupervisor(settings, store=store, backends={"cuda_process": backend})
    store.update_job("a", status=JobStatus.PAUSED)
    tracked_worker(supervisor, tmp_path, "a", returncode=0)
    supervisor.poll()
    assert supervisor.has_reaped_process("a")
    next_process = Mock()
    next_process.poll.return_value = None
    backend.launch.return_value = [WorkerProcessHandle("a", next_process, tmp_path / "out", tmp_path / "err")]
    decision = supervisor.dispatch([store.update_job("a", status=JobStatus.READY)],
                                   mode="stack_anchor", backend_name="cuda_process")
    assert decision.can_run and supervisor.has_live_process("a")
    assert not supervisor.has_reaped_process("a")
    next_process.poll.return_value = 0
    supervisor.poll()
    assert supervisor.has_reaped_process("a")


def test_unknown_supervisor_exit_evidence_cannot_renew(tmp_path):
    _, store, _, service, _ = system(tmp_path)
    exhausted(store)
    store.update_job("a", status=JobStatus.COMPLETED)
    service.supervisor.has_reaped_process = None
    refresh(service, store)
    assert store.get_job("b").metadata["cold_start_solo_required"]


def test_legacy_exhaustion_and_partial_legacy_history_never_gain_proof(tmp_path):
    _, store, planner, service, trial = pending_trial(tmp_path)
    service.supervisor.has_live_process.return_value = False
    store.update_job("b", metadata_updates={"cold_start_failed_attempts": 2})
    service._finish_cold_start_trial(trial, "inconclusive", "new failure after legacy history")
    current = store.get_job("b")
    assert current.metadata["cold_start_failure_cohort"]["failed_attempts"] == 3
    assert current.metadata["cold_start_failure_cohort"]["proof_complete"] is False
    store.update_job("a", status=JobStatus.COMPLETED)
    store.update_job("b", status=JobStatus.PAUSED)
    refresh(service, store)
    assert store.get_job("b").metadata["cold_start_solo_required"]
    # Legacy candidate exhaustion remains, but a naturally running anchor no
    # longer vetoes fresh candidates joining it.
    store.update_job("b", status=JobStatus.RUNNING)
    assert planner.choose_plan([store.get_job("c")], active_jobs=[store.get_job("b")],
                               backend_available=AVAILABLE).job_ids == ("c",)
    store.update_job("b", metadata_updates={"cold_start_failure_cohort": {}})
    refresh(service, store)
    assert store.get_job("b").metadata["cold_start_solo_required"]


def test_membership_change_records_original_and_actual_incumbents(tmp_path):
    _, store, _, service, trial = pending_trial(tmp_path, ("a", "b", "c"))
    store.update_job("b", status=JobStatus.PAUSED)
    service.supervisor.active_job_ids.return_value = ["a", "c"]
    service._evaluate_colocation_trial()
    assert trial.preexisting_job_ids == ("a",)
    service._finish_cold_start_trial(trial, "rejected", "membership changed")
    cohort = store.get_job("c").metadata["cold_start_failure_cohort"]
    assert cohort["proof_complete"] and cohort["incumbent_job_ids"] == ["a", "b"]
    assert "c" not in cohort["incumbent_job_ids"]


def test_renewal_preserves_exact_rejected_keys_and_all_cooldowns(tmp_path):
    settings, store, planner, service, _ = system(tmp_path)
    plan = planner.choose_plan([store.get_job("b")], active_jobs=[store.get_job("a")],
                               backend_available=AVAILABLE)
    key = plan.trial_metadata["profile_key"]
    exhausted(store)
    history = {key: {"failures": 1, "retry_after": 8765432100, "decision": "rejected"}}
    store.update_job("b", metadata_updates={"cold_start_blocked_groups": [key],
                     "cold_start_group_attempts": history, "cold_start_retry_after": 8765432100})
    store.update_job("a", status=JobStatus.COMPLETED)
    current = refresh(service, store)[0]
    assert current.metadata["cold_start_blocked_groups"] == [key]
    assert current.metadata["cold_start_group_attempts"] == history
    assert current.metadata["cold_start_retry_after"] == 8765432100
    assert cold_group_blocked(current, key, settings.gpu_scheduler.colocation, 9999999999)
    assert planner.choose_plan([current], active_jobs=[store.get_job("c")], backend_available=AVAILABLE) is None


def test_dispatch_refreshes_exhausted_anchor_and_allows_fresh_addition(tmp_path):
    _, store, planner, service, _ = system(tmp_path)
    store.update_job("a", status=JobStatus.COMPLETED)
    exhausted(store, "b")
    exhausted(store, "c")
    service.supervisor.available_backends.return_value = AVAILABLE
    service._choose_placement_replay_plan = Mock(return_value=(False, None))
    service._active_vram_occupancy = Mock(return_value=0)
    service._dispatch_plan = Mock(return_value=False)
    service._dispatch_pending_work()
    anchor = service._dispatch_plan.call_args.args[0]
    assert anchor.mode == "stack_anchor" and anchor.job_ids == ("b",)
    assert not store.get_job("b").metadata["cold_start_solo_required"]
    assert not store.get_job("c").metadata["cold_start_solo_required"]
    store.update_job("b", status=JobStatus.RUNNING)
    addition = planner.choose_plan([store.get_job("c")], active_jobs=[store.get_job("b")],
                                   backend_available=AVAILABLE)
    assert addition.job_ids == ("c",) and addition.trial_metadata["requires_live_trial"]
    assert addition.objective_breakdown["gain"] is None  # Still needs measured acceptance.


def test_each_three_failure_budget_requires_new_fully_retired_incumbents(tmp_path):
    settings, store, planner, service, jobs = system(tmp_path)
    for name in ("d", "e", "f", "g"):
        job = deepcopy(jobs[0])
        job.job_id = job.packing.signature = name
        store.submit_job(job)
    consumed = set()
    for generation, pair in enumerate((("a", "c"), ("d", "e"), ("f", "g"))):
        for attempt, members in enumerate(((pair[0],), pair, (pair[1],)), 1):
            for name in members:
                store.update_job(name, status=JobStatus.RUNNING)
            candidate = store.update_job("b", status=JobStatus.PAUSED,
                                         metadata_updates={"cold_start_retry_after": 0})
            plan = planner.choose_plan([candidate], active_jobs=[store.get_job(name) for name in members],
                                       backend_available=AVAILABLE)
            assert plan is not None
            trial = service._prepare_colocation_trial(plan, candidate)
            service._finish_cold_start_trial(trial, "inconclusive", "bounded cohort fixture")
            current = store.get_job("b")
            assert current.metadata["cold_start_failure_cohort"]["failed_attempts"] == attempt
            assert current.metadata["cold_start_failed_attempts"] == 3 * generation + attempt
        store.update_job("b", status=JobStatus.PAUSED)
        for _ in range(3):
            refresh(service, store)
        assert store.get_job("b").metadata["cold_start_solo_required"]
        store.update_job(pair[0], status=JobStatus.COMPLETED)
        refresh(service, store)
        assert store.get_job("b").metadata["cold_start_solo_required"]
        store.update_job(pair[1], status=JobStatus.COMPLETED)
        current = refresh(service, store)[0]
        consumed.update(pair)
        cohort = current.metadata["cold_start_failure_cohort"]
        assert cohort["generation"] == generation + 1
        assert set(cohort["retired_incumbent_job_ids"]) == consumed
        assert cohort["failed_attempts"] == 0
    assert store.get_job("b").metadata["cold_start_failed_attempts"] == 9
    assert service.event_logger.emit.call_count == 12  # Nine decisions, three renewals.
