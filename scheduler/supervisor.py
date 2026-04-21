"""Worker supervision for the single active GPU slot."""

from __future__ import annotations

from dataclasses import dataclass

from ..execution.control import ControlPlane
from ..execution.executor import SubprocessExecutor, WorkerProcessHandle
from ..schemas import PlacementDecision, TrainingJob
from ..settings import SchedulerSettings


@dataclass(slots=True)
class WorkerSnapshot:
    job_id: str
    alive: bool
    returncode: int | None = None


class WorkerSupervisor:
    """Manage the one active worker subprocess for V1."""

    def __init__(self, settings: SchedulerSettings):
        self.settings = settings
        self.control_plane = ControlPlane(settings)
        self.executor = SubprocessExecutor(settings)
        self._active: WorkerProcessHandle | None = None

    def placement_for(self, job: TrainingJob) -> PlacementDecision:
        active_job_id = self.active_job_id()
        if active_job_id is not None:
            return PlacementDecision(can_run=False, reason=f"GPU busy with job {active_job_id}", gpu_slot=0)
        return PlacementDecision(can_run=True, reason="GPU slot available", gpu_slot=0)

    def dispatch(self, job: TrainingJob) -> PlacementDecision:
        decision = self.placement_for(job)
        if not decision.can_run:
            return decision
        self.control_plane.initialize_job(job.job_id)
        self.control_plane.clear_command(job.job_id)
        self._active = self.executor.start(job)
        return decision

    def poll(self) -> WorkerSnapshot | None:
        if self._active is None:
            return None
        returncode = self._active.process.poll()
        if returncode is None:
            return WorkerSnapshot(job_id=self._active.job_id, alive=True)
        snapshot = WorkerSnapshot(job_id=self._active.job_id, alive=False, returncode=returncode)
        self._active = None
        return snapshot

    def active_job_id(self) -> str | None:
        if self._active is None:
            return None
        if self._active.process.poll() is not None:
            self._active = None
            return None
        return self._active.job_id

    def request_pause(self, job_id: str, *, reason: str, hold: bool) -> bool:
        if self.active_job_id() != job_id:
            return False
        self.control_plane.request_pause(job_id, reason=reason, hold=hold)
        return True

    def request_cancel(self, job_id: str, *, reason: str) -> bool:
        if self.active_job_id() != job_id:
            return False
        self.control_plane.request_cancel(job_id, reason=reason)
        return True
