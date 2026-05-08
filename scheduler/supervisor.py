"""Worker supervision for exclusive and packed placement groups."""

from __future__ import annotations

from dataclasses import dataclass, field
import subprocess
import uuid

from ..execution.backend_registry import BackendRegistry
from ..execution.backends import ExecutionBackend
from ..execution.control import ControlPlane
from ..execution.executor import SubprocessExecutor, WorkerProcessHandle
from ..domain import JobStatus, PlacementDecision, TrainingJob
from ..config import SchedulerSettings, SCHEDULER_MODE_PARALLEL_AUTO_PACK
from ..storage.sqlite_store import SQLiteStateStore


@dataclass(slots=True)
class WorkerSnapshot:
    job_id: str
    group_id: str
    alive: bool
    returncode: int | None = None
    reported_by: str = "process"


@dataclass(slots=True)
class ManagedWorker:
    job_id: str
    handle: WorkerProcessHandle
    role: str


@dataclass(slots=True)
class PlacementGroupHandle:
    group_id: str
    mode: str
    backend_name: str
    batch_overrides: dict[str, int] = field(default_factory=dict)
    fallback_order: list[str] = field(default_factory=list)
    workers: dict[str, ManagedWorker] = field(default_factory=dict)

    def active_job_ids(self) -> list[str]:
        return list(self.workers.keys())

    def size(self) -> int:
        return len(self.workers)


class WorkerSupervisor:
    """Manage placement groups on one physical GPU."""

    def __init__(
        self,
        settings: SchedulerSettings,
        *,
        store: SQLiteStateStore | None = None,
        backends: dict[str, ExecutionBackend] | None = None,
    ):
        self.settings = settings
        self.store = store or SQLiteStateStore(settings)
        self.control_plane = ControlPlane(settings)
        self.executor = SubprocessExecutor(settings)
        self.backend_registry = BackendRegistry(settings, self.executor, backends=backends)
        self._groups: dict[str, PlacementGroupHandle] = {}

    def _concurrency_enabled(self) -> bool:
        return (
            self.settings.gpu_scheduler.mode == SCHEDULER_MODE_PARALLEL_AUTO_PACK
            and self.settings.gpu_scheduler.concurrent_groups_enabled
        )

    def _overlap_allowed_for_backend(self, backend_name: str) -> bool:
        return backend_name in set(self.settings.gpu_scheduler.concurrent_backend_allowlist)

    def available_backends(self) -> dict[str, bool]:
        return self.backend_registry.availability()

    def active_groups(self) -> dict[str, PlacementGroupHandle]:
        return dict(self._groups)

    def active_group(self) -> PlacementGroupHandle | None:
        if not self._groups:
            return None
        first_key = next(iter(self._groups))
        return self._groups[first_key]

    def active_group_mode(self) -> str | None:
        group = self.active_group()
        return group.mode if group else None

    def active_group_backend(self) -> str | None:
        group = self.active_group()
        return group.backend_name if group else None

    def active_batch_overrides(self) -> dict[str, int]:
        group = self.active_group()
        return dict(group.batch_overrides) if group else {}

    def active_fallback_order(self) -> list[str]:
        group = self.active_group()
        return list(group.fallback_order) if group else []

    def _worker_is_alive(self, worker: ManagedWorker) -> bool:
        if worker.handle.monitor_via_store:
            job = self.store.get_job(worker.job_id)
            return bool(job is not None and job.status in {JobStatus.RUNNING, JobStatus.PAUSING})
        return worker.handle.process.poll() is None

    def active_job_ids(self) -> list[str]:
        job_ids: list[str] = []
        for group in self._groups.values():
            job_ids.extend([job_id for job_id, worker in group.workers.items() if self._worker_is_alive(worker)])
        return job_ids

    def active_job_ids_by_group(self) -> dict[str, list[str]]:
        payload: dict[str, list[str]] = {}
        for group_id, group in self._groups.items():
            payload[group_id] = [job_id for job_id, worker in group.workers.items() if self._worker_is_alive(worker)]
        return payload

    def active_job_id(self) -> str | None:
        job_ids = self.active_job_ids()
        return job_ids[0] if job_ids else None

    def placement_for(
        self,
        plan_mode: str,
        plan_backend_name: str,
        job_ids: list[str],
        *,
        batch_overrides: dict[str, int] | None = None,
        fallback_order: list[str] | None = None,
    ) -> PlacementDecision:
        backend = self.backend_registry.get(plan_backend_name)
        if backend is None:
            return PlacementDecision(
                can_run=False,
                reason=f"unknown backend {plan_backend_name}",
                gpu_slot=0,
                mode=plan_mode,
                backend_name=plan_backend_name,
                job_ids=job_ids,
                batch_overrides=batch_overrides or {},
                fallback_order=fallback_order or [],
            )
        if not backend.available():
            return PlacementDecision(
                can_run=False,
                reason=f"backend {plan_backend_name} unavailable",
                gpu_slot=0,
                mode=plan_mode,
                backend_name=plan_backend_name,
                job_ids=job_ids,
                batch_overrides=batch_overrides or {},
                fallback_order=fallback_order or [],
            )
        if not self._groups:
            return PlacementDecision(
                can_run=True,
                reason="GPU slot available",
                gpu_slot=0,
                mode=plan_mode,
                backend_name=plan_backend_name,
                job_ids=job_ids,
                batch_overrides=batch_overrides or {},
                fallback_order=fallback_order or [],
            )

        if not self._concurrency_enabled():
            active_jobs = self.active_job_ids()
            return PlacementDecision(
                can_run=False,
                reason=f"GPU busy with jobs {', '.join(active_jobs)}",
                gpu_slot=0,
                mode=plan_mode,
                backend_name=plan_backend_name,
                job_ids=active_jobs,
                batch_overrides=batch_overrides or {},
                fallback_order=fallback_order or [],
            )

        if plan_backend_name == "exclusive":
            return PlacementDecision(
                can_run=False,
                reason="exclusive groups cannot overlap with active groups",
                gpu_slot=0,
                mode=plan_mode,
                backend_name=plan_backend_name,
                job_ids=job_ids,
                batch_overrides=batch_overrides or {},
                fallback_order=fallback_order or [],
            )

        if not self._overlap_allowed_for_backend(plan_backend_name):
            return PlacementDecision(
                can_run=False,
                reason=f"backend {plan_backend_name} is not enabled for overlapping groups",
                gpu_slot=0,
                mode=plan_mode,
                backend_name=plan_backend_name,
                job_ids=job_ids,
                batch_overrides=batch_overrides or {},
                fallback_order=fallback_order or [],
            )

        for active_group in self._groups.values():
            if not self._overlap_allowed_for_backend(active_group.backend_name):
                return PlacementDecision(
                    can_run=False,
                    reason=f"active backend {active_group.backend_name} blocks concurrent dispatch",
                    gpu_slot=0,
                    mode=plan_mode,
                    backend_name=plan_backend_name,
                    job_ids=active_group.active_job_ids(),
                    batch_overrides=batch_overrides or {},
                    fallback_order=fallback_order or [],
                )

        return PlacementDecision(
            can_run=True,
            reason="concurrent group slot available",
            gpu_slot=0,
            mode=plan_mode,
            backend_name=plan_backend_name,
            job_ids=job_ids,
            batch_overrides=batch_overrides or {},
            fallback_order=fallback_order or [],
        )

    def dispatch(
        self,
        jobs: list[TrainingJob],
        *,
        mode: str,
        backend_name: str,
        batch_overrides: dict[str, int] | None = None,
        fallback_order: list[str] | None = None,
    ) -> PlacementDecision:
        decision = self.placement_for(
            mode,
            backend_name,
            [job.job_id for job in jobs],
            batch_overrides=batch_overrides,
            fallback_order=fallback_order,
        )
        if not decision.can_run:
            return decision
        group_id = uuid.uuid4().hex[:12]
        backend = self.backend_registry.get(backend_name)
        if backend is None:
            return decision
        for job in jobs:
            self.control_plane.initialize_job(job.job_id)
            self.control_plane.clear_command(job.job_id)
        handles = backend.launch(jobs)
        workers: dict[str, ManagedWorker] = {}
        for index, handle in enumerate(handles):
            if len(handles) == 1:
                role = "solo"
            elif len(handles) == 2:
                role = "primary" if index == 0 else "secondary"
            else:
                role = f"slot-{index}"
            workers[handle.job_id] = ManagedWorker(job_id=handle.job_id, handle=handle, role=role)
        self._groups[group_id] = PlacementGroupHandle(
            group_id=group_id,
            mode=mode,
            backend_name=backend_name,
            batch_overrides=dict(batch_overrides or {}),
            fallback_order=list(fallback_order or []),
            workers=workers,
        )
        decision.group_id = group_id
        return decision

    def poll(self) -> list[WorkerSnapshot]:
        snapshots: list[WorkerSnapshot] = []
        for group_id, group in list(self._groups.items()):
            for job_id, worker in list(group.workers.items()):
                if worker.handle.monitor_via_store:
                    job = self.store.get_job(job_id)
                    if job is None or job.status not in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.PAUSED}:
                        continue
                    snapshots.append(
                        WorkerSnapshot(
                            job_id=job_id,
                            group_id=group_id,
                            alive=False,
                            returncode=worker.handle.process.poll(),
                            reported_by="store",
                        )
                    )
                    del group.workers[job_id]
                    continue
                returncode = worker.handle.process.poll()
                if returncode is None:
                    continue
                snapshots.append(
                    WorkerSnapshot(
                        job_id=job_id,
                        group_id=group_id,
                        alive=False,
                        returncode=returncode,
                        reported_by="process",
                    )
                )
                del group.workers[job_id]
            self._refresh_group_shape(group_id)
        return snapshots

    def request_pause(self, job_id: str, *, reason: str, hold: bool) -> bool:
        if job_id not in self.active_job_ids():
            return False
        self.control_plane.request_pause(job_id, reason=reason, hold=hold)
        return True

    def request_cancel(self, job_id: str, *, reason: str) -> bool:
        if job_id not in self.active_job_ids():
            return False
        self.control_plane.request_cancel(job_id, reason=reason)
        return True

    def request_fallback_pause(self, job_id: str, *, reason: str) -> bool:
        for group in self._groups.values():
            if job_id not in group.workers:
                continue
            self.control_plane.request_pause(job_id, reason=reason, hold=False)
            return True
        return False

    def shutdown(self) -> None:
        seen_processes: set[int] = set()
        for group in self._groups.values():
            for worker in group.workers.values():
                process = worker.handle.process
                if process.pid in seen_processes or process.poll() is not None:
                    continue
                seen_processes.add(process.pid)
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        self._groups = {}

    def _refresh_group_shape(self, group_id: str) -> None:
        group = self._groups.get(group_id)
        if group is None:
            return
        alive_workers = {
            job_id: worker
            for job_id, worker in group.workers.items()
            if self._worker_is_alive(worker)
        }
        group.workers = alive_workers
        group.fallback_order = [job_id for job_id in group.fallback_order if job_id in alive_workers]
        if not alive_workers:
            self._groups.pop(group_id, None)
            return
        if len(alive_workers) == 1:
            sole_worker = next(iter(alive_workers.values()))
            sole_worker.role = "solo"
            group.mode = "exclusive"
