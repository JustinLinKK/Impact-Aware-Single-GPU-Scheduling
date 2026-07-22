from __future__ import annotations

from pathlib import Path
import time

from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import (
    BatchProbeSpec,
    JobStatus,
    PackingSpec,
    PlacementDecision,
    ProfileState,
    TrainingJob,
)
from localml_scheduler.scheduler.planner_types import DispatchPlan
from localml_scheduler.scheduler.service import ActiveRun, SchedulerService
from localml_scheduler.scheduler.supervisor import WorkerSnapshot
from localml_scheduler.storage import StateStore


class FakeSupervisor:
    def __init__(self, active_job_ids: list[str] | None = None):
        self.active_ids = list(active_job_ids or [])
        self.prepare_requests: list[tuple[str, str]] = []
        self.commit_requests: list[tuple[str, str]] = []
        self.abort_requests: list[tuple[str, str]] = []
        self.acks: dict[str, dict[str, object]] = {}
        self.dispatch_results: list[bool] = []
        self.dispatched: list[tuple[list[str], dict[str, int]]] = []
        self.stopped_groups: list[str] = []

    def available_backends(self) -> dict[str, bool]:
        return {"stream": True, "cuda_process": True, "mps": False, "stream_mps": False, "exclusive": True}

    def active_job_ids(self) -> list[str]:
        return list(self.active_ids)

    def active_job_ids_by_group(self) -> dict[str, list[str]]:
        return {"active": list(self.active_ids)} if self.active_ids else {}

    def request_repack_prepare(self, job_id: str, *, transaction_id: str, reason: str) -> bool:
        del reason
        self.prepare_requests.append((job_id, transaction_id))
        return job_id in self.active_ids

    def request_repack_commit(self, job_id: str, *, transaction_id: str) -> bool:
        self.commit_requests.append((job_id, transaction_id))
        return True

    def request_repack_abort(self, job_id: str, *, transaction_id: str) -> bool:
        self.abort_requests.append((job_id, transaction_id))
        return True

    def repack_ack(self, job_id: str):
        return self.acks.get(job_id)

    def dispatch(self, jobs, *, mode, backend_name, batch_overrides=None, fallback_order=None):
        del fallback_order
        self.dispatched.append(([job.job_id for job in jobs], dict(batch_overrides or {})))
        can_run = self.dispatch_results.pop(0) if self.dispatch_results else True
        if can_run:
            self.active_ids = [job.job_id for job in jobs]
        return PlacementDecision(
            can_run=can_run,
            reason="fake",
            group_id="replacement" if can_run else None,
            mode=mode,
            backend_name=backend_name,
            job_ids=[job.job_id for job in jobs],
            batch_overrides=dict(batch_overrides or {}),
        )

    def shutdown(self) -> None:
        return None

    def stop_group(self, group_id: str) -> None:
        self.stopped_groups.append(group_id)
        self.active_ids = []


def _settings(tmp_path: Path, *, prediction_mode: str = "branch_profile") -> SchedulerSettings:
    return SchedulerSettings(
        runtime_root=tmp_path / "runtime",
        prediction={"mode": prediction_mode, "fallback_to_exclusive": True},
        gpu_scheduler={
            "mode": "adaptive",
            "backend_priority": ["stream", "exclusive"],
            "checkpoint_preemption_min_runtime_seconds": 0,
            "checkpoint_preemption_cooldown_seconds": 0,
            "checkpoint_preemption_pause_timeout_seconds": 1,
            "adaptive": {"replan_debounce_seconds": 0},
        },
        log_db={"enabled": False},
    )


def _job(job_id: str, *, namespace: str, status: JobStatus = JobStatus.PENDING) -> TrainingJob:
    job = TrainingJob.create(
        "tests.elastic:run",
        job_id,
        f"/{job_id}.py",
        job_id=job_id,
        task_type="mlevolve_script",
        runner_kwargs={"batch_size": 8},
        loader_target="localml_scheduler.adapters.mlevolve_runner:load_raw_file",
        packing=PackingSpec(eligible=True, signature=f"sig:{job_id}", backend_allowlist=["stream"]),
        batch_probe=BatchProbeSpec(
            enabled=True,
            probe_target="localml_scheduler.tests.test_batch_probe:curve_probe",
            model_key=namespace,
            profile_namespace=namespace,
            shape_signature_override=f"shape:{namespace}",
            minimum_batch_size=1,
            contract_version=3,
        ),
        metadata={"elastic_contract_validated": True},
    )
    job.status = status
    return job


def test_missing_profile_latches_drain_and_deduplicates_arrivals(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = StateStore(settings)
    active_a = _job("a", namespace="ready-a", status=JobStatus.RUNNING)
    active_b = _job("b", namespace="ready-b", status=JobStatus.RUNNING)
    waiting_c = _job("c", namespace="shared")
    waiting_d = _job("d", namespace="shared")
    for job in (active_a, active_b, waiting_c, waiting_d):
        store.save_job(job)
    supervisor = FakeSupervisor(["a", "b"])
    service = SchedulerService(settings, store=store, supervisor=supervisor)
    service._active_runs["active"] = ActiveRun(
        group_id="active",
        mode="packed_pair",
        backend_name="stream",
        job_ids=("a", "b"),
        batch_overrides={"a": 8, "b": 8},
    )

    assert service._profile_drain_blocks_dispatch([waiting_c, waiting_d]) is True
    assert store.get_job("c").profile_state == ProfileState.WAITING_FOR_DRAIN
    assert store.get_job("d").profile_state == ProfileState.WAITING_FOR_DRAIN
    assert not [job for job in store.list_jobs() if job.task_type == "mlevolve_branch_profile_probe"]

    supervisor.active_ids = []
    service._active_runs.clear()
    refreshed = [store.get_job("c"), store.get_job("d")]
    assert service._profile_drain_blocks_dispatch([job for job in refreshed if job is not None]) is True
    probes = [job for job in store.list_jobs() if job.task_type == "mlevolve_branch_profile_probe"]
    assert len(probes) == 1
    assert probes[0].metadata["profile_gate_key"].startswith("shared|")
    assert probes[0].config.loader_target == "localml_scheduler.adapters.mlevolve_runner:load_raw_file"

    # A later job for the same branch joins the current drain cycle.
    waiting_e = _job("e", namespace="shared")
    store.save_job(waiting_e)
    service._profile_drain_blocks_dispatch([store.get_job("c"), store.get_job("d"), waiting_e])
    probes = [job for job in store.list_jobs() if job.task_type == "mlevolve_branch_profile_probe"]
    assert len(probes) == 1


def test_checkpoint_timeout_aborts_transaction_without_batch_mutation(tmp_path: Path) -> None:
    settings = _settings(tmp_path, prediction_mode="ml_predictor")
    store = StateStore(settings)
    a = _job("a", namespace="a", status=JobStatus.RUNNING)
    a.current_batch_size = 16
    store.save_job(a)
    supervisor = FakeSupervisor(["a"])
    service = SchedulerService(settings, store=store, supervisor=supervisor)
    service._active_runs["active"] = ActiveRun(
        group_id="active",
        mode="exclusive",
        backend_name="exclusive",
        job_ids=("a",),
        batch_overrides={"a": 16},
    )
    target = DispatchPlan(
        mode="exclusive",
        backend_name="exclusive",
        job_ids=("a",),
        reason="resize",
        batch_overrides={"a": 8},
    )
    assert service._begin_repack(target, [a]) is True
    assert service._pending_repack is not None
    service._pending_repack.requested_at_monotonic = time.monotonic() - 2
    assert service._advance_pending_repack() is True
    assert service._pending_repack is None
    assert supervisor.abort_requests
    assert store.get_job("a").current_batch_size == 16


def test_abc_repack_checkpoints_active_jobs_and_restarts_all_three(tmp_path: Path) -> None:
    settings = _settings(tmp_path, prediction_mode="ml_predictor")
    store = StateStore(settings)
    a = _job("a", namespace="a", status=JobStatus.RUNNING)
    b = _job("b", namespace="b", status=JobStatus.RUNNING)
    c = _job("c", namespace="c")
    a.current_batch_size = 16
    b.current_batch_size = 16
    for job in (a, b, c):
        store.save_job(job)
    supervisor = FakeSupervisor(["a", "b"])
    service = SchedulerService(settings, store=store, supervisor=supervisor)
    service._active_runs["active"] = ActiveRun(
        group_id="active",
        mode="packed_pair",
        backend_name="stream",
        job_ids=("a", "b"),
        batch_overrides={"a": 16, "b": 16},
    )
    target = DispatchPlan(
        mode="packed_group",
        backend_name="stream",
        job_ids=("a", "b", "c"),
        reason="admit c",
        batch_overrides={"a": 4, "b": 4, "c": 8},
    )

    assert service._begin_repack(target, [a, b]) is True
    transaction = service._pending_repack
    assert transaction is not None
    for job_id in ("a", "b"):
        supervisor.acks[job_id] = {
            "transaction_id": transaction.transaction_id,
            "checkpoint_path": f"/checkpoint/{job_id}.pt",
        }
    service._advance_pending_repack()
    assert transaction.phase == "committing"
    assert {job_id for job_id, _ in supervisor.commit_requests} == {"a", "b"}
    supervisor.active_ids = []
    service._active_runs.clear()
    service._advance_pending_repack()

    assert service._pending_repack is None
    assert supervisor.dispatched == [(["a", "b", "c"], {"a": 4, "b": 4, "c": 8})]
    assert [store.get_job(job_id).current_batch_size for job_id in ("a", "b", "c")] == [4, 4, 8]
    assert [store.get_job(job_id).authored_batch_size for job_id in ("a", "b", "c")] == [8, 8, 8]
    assert [store.get_job(job_id).config.runner_kwargs["batch_size"] for job_id in ("a", "b", "c")] == [8, 8, 8]


def test_target_launch_failure_marks_incompatible_and_restores_old_vector(tmp_path: Path) -> None:
    settings = _settings(tmp_path, prediction_mode="ml_predictor")
    store = StateStore(settings)
    a = _job("a", namespace="a", status=JobStatus.RUNNING)
    c = _job("c", namespace="c")
    a.current_batch_size = 16
    store.save_job(a)
    store.save_job(c)
    supervisor = FakeSupervisor(["a"])
    supervisor.dispatch_results = [False, True]
    service = SchedulerService(settings, store=store, supervisor=supervisor)
    service._active_runs["active"] = ActiveRun(
        group_id="active",
        mode="exclusive",
        backend_name="exclusive",
        job_ids=("a",),
        batch_overrides={"a": 16},
    )
    target = DispatchPlan(
        mode="packed_pair",
        backend_name="stream",
        job_ids=("a", "c"),
        reason="admit c",
        batch_overrides={"a": 4, "c": 4},
    )
    assert service._begin_repack(target, [a]) is True
    transaction = service._pending_repack
    assert transaction is not None
    supervisor.acks["a"] = {
        "transaction_id": transaction.transaction_id,
        "checkpoint_path": "/checkpoint/a.pt",
    }
    service._advance_pending_repack()
    assert transaction.phase == "committing"
    supervisor.active_ids = []
    service._active_runs.clear()
    service._advance_pending_repack()

    assert service._pending_repack is None
    assert store.get_job("a").current_batch_size == 16
    assert store.get_job("c").current_batch_size == 8
    profiles = store.list_combination_profiles(scheduler_mode="adaptive")
    assert len(profiles) == 1 and profiles[0].compatible is False
    assert [item[0] for item in supervisor.dispatched] == [["a", "c"], ["a"]]


def test_runtime_oom_rolls_back_old_pack_and_requeues_new_admission(tmp_path: Path) -> None:
    settings = _settings(tmp_path, prediction_mode="ml_predictor")
    store = StateStore(settings)
    a = _job("a", namespace="a", status=JobStatus.RUNNING)
    c = _job("c", namespace="c")
    a.current_batch_size = 16
    store.save_job(a)
    store.save_job(c)
    supervisor = FakeSupervisor(["a"])
    supervisor.dispatch_results = [True, True]
    service = SchedulerService(settings, store=store, supervisor=supervisor)
    service._active_runs["active"] = ActiveRun(
        group_id="active",
        mode="exclusive",
        backend_name="exclusive",
        job_ids=("a",),
        batch_overrides={"a": 16},
    )
    target = DispatchPlan(
        mode="packed_pair",
        backend_name="stream",
        job_ids=("a", "c"),
        reason="admit c",
        batch_overrides={"a": 4, "c": 4},
    )
    assert service._begin_repack(target, [a]) is True
    transaction = service._pending_repack
    assert transaction is not None
    supervisor.acks["a"] = {
        "transaction_id": transaction.transaction_id,
        "checkpoint_path": "/checkpoint/a.pt",
    }
    service._advance_pending_repack()
    supervisor.active_ids = []
    service._active_runs.clear()
    service._advance_pending_repack()

    run = service._active_runs["replacement"]
    assert run.repack_transaction is transaction
    store.set_job_status("c", JobStatus.FAILED, reason="CUDA out of memory", hold=True)
    service._handle_worker_exit(
        WorkerSnapshot(job_id="c", group_id="replacement", alive=False, returncode=1),
        run_context=run,
    )

    assert supervisor.stopped_groups == ["replacement"]
    assert store.get_job("a").current_batch_size == 16
    assert store.get_job("a").status == JobStatus.RUNNING
    assert store.get_job("c").current_batch_size == 8
    assert store.get_job("c").status == JobStatus.READY
    assert [item[0] for item in supervisor.dispatched] == [["a", "c"], ["a"]]
    profiles = store.list_combination_profiles(scheduler_mode="adaptive")
    assert len(profiles) == 1 and profiles[0].compatible is False
