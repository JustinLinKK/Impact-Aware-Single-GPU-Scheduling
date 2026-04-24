"""Worker supervision for exclusive and pair-packed placement groups."""

from __future__ import annotations

from dataclasses import dataclass, field
import subprocess

from ..execution.backends import ExclusiveBackend, ExecutionBackend, MPSBackend
from ..execution.control import ControlPlane
from ..execution.executor import SubprocessExecutor, WorkerProcessHandle
from ..schemas import PlacementDecision, TrainingJob
from ..settings import SchedulerSettings


@dataclass(slots=True)
class WorkerSnapshot:
    job_id: str
    alive: bool
    returncode: int | None = None


@dataclass(slots=True)
class ManagedWorker:
    job_id: str
    handle: WorkerProcessHandle
    role: str


@dataclass(slots=True)
class PlacementGroupHandle:
    mode: str
    backend_name: str
    workers: dict[str, ManagedWorker] = field(default_factory=dict)

    def active_job_ids(self) -> list[str]:
        return list(self.workers.keys())

    def size(self) -> int:
        return len(self.workers)


class WorkerSupervisor:
    """Manage a single placement group on one physical GPU."""

    def __init__(self, settings: SchedulerSettings, *, backends: dict[str, ExecutionBackend] | None = None):
        self.settings = settings
        self.control_plane = ControlPlane(settings)
        self.executor = SubprocessExecutor(settings)
        self.backends = backends or {
            "exclusive": ExclusiveBackend(settings, self.executor),
            "mps": MPSBackend(settings, self.executor),
        }
        self._group: PlacementGroupHandle | None = None

    def available_backends(self) -> dict[str, bool]:
        return {name: backend.available() for name, backend in self.backends.items()}

    def active_group(self) -> PlacementGroupHandle | None:
        return self._group

    def active_group_mode(self) -> str | None:
        group = self.active_group()
        return group.mode if group else None

    def active_group_backend(self) -> str | None:
        group = self.active_group()
        return group.backend_name if group else None

    def active_job_ids(self) -> list[str]:
        group = self.active_group()
        if group is None:
            return []
        return [job_id for job_id, worker in group.workers.items() if worker.handle.process.poll() is None]

    def active_job_id(self) -> str | None:
        job_ids = self.active_job_ids()
        return job_ids[0] if job_ids else None

    def placement_for(self, plan_mode: str, plan_backend_name: str, job_ids: list[str]) -> PlacementDecision:
        group = self.active_group()
        if group is not None:
            return PlacementDecision(
                can_run=False,
                reason=f"GPU busy with jobs {', '.join(group.active_job_ids())}",
                gpu_slot=0,
                mode=group.mode,
                backend_name=group.backend_name,
                job_ids=group.active_job_ids(),
            )
        backend = self.backends.get(plan_backend_name)
        if backend is None:
            return PlacementDecision(
                can_run=False,
                reason=f"unknown backend {plan_backend_name}",
                gpu_slot=0,
                mode=plan_mode,
                backend_name=plan_backend_name,
                job_ids=job_ids,
            )
        if not backend.available():
            return PlacementDecision(
                can_run=False,
                reason=f"backend {plan_backend_name} unavailable",
                gpu_slot=0,
                mode=plan_mode,
                backend_name=plan_backend_name,
                job_ids=job_ids,
            )
        return PlacementDecision(
            can_run=True,
            reason="GPU slot available",
            gpu_slot=0,
            mode=plan_mode,
            backend_name=plan_backend_name,
            job_ids=job_ids,
        )

    def dispatch(self, jobs: list[TrainingJob], *, mode: str, backend_name: str) -> PlacementDecision:
        decision = self.placement_for(mode, backend_name, [job.job_id for job in jobs])
        if not decision.can_run:
            return decision
        backend = self.backends[backend_name]
        for job in jobs:
            self.control_plane.initialize_job(job.job_id)
            self.control_plane.clear_command(job.job_id)
        handles = backend.launch(jobs)
        workers = {}
        roles = ["primary", "secondary"] if len(handles) == 2 else ["solo"]
        for role, handle in zip(roles, handles, strict=True):
            workers[handle.job_id] = ManagedWorker(job_id=handle.job_id, handle=handle, role=role)
        self._group = PlacementGroupHandle(mode=mode, backend_name=backend_name, workers=workers)
        return decision

    def poll(self) -> list[WorkerSnapshot]:
        group = self.active_group()
        if group is None:
            return []
        snapshots: list[WorkerSnapshot] = []
        for job_id, worker in list(group.workers.items()):
            returncode = worker.handle.process.poll()
            if returncode is None:
                continue
            snapshots.append(WorkerSnapshot(job_id=job_id, alive=False, returncode=returncode))
            del group.workers[job_id]
        self._refresh_group_shape()
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

    def demote_secondary(self, *, reason: str) -> str | None:
        group = self.active_group()
        if group is None or group.mode != "packed_pair":
            return None
        for worker in group.workers.values():
            if worker.role == "secondary":
                self.control_plane.request_pause(worker.job_id, reason=reason, hold=False)
                return worker.job_id
        return None

    def shutdown(self) -> None:
        group = self._group
        if group is None:
            return
        for worker in group.workers.values():
            process = worker.handle.process
            if process.poll() is not None:
                continue
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        self._group = None

    def _refresh_group_shape(self) -> None:
        if self._group is None:
            return
        alive_workers = {
            job_id: worker
            for job_id, worker in self._group.workers.items()
            if worker.handle.process.poll() is None
        }
        self._group.workers = alive_workers
        if not alive_workers:
            self._group = None
            return
        if len(alive_workers) == 1:
            sole_worker = next(iter(alive_workers.values()))
            sole_worker.role = "solo"
            self._group.mode = "exclusive"
