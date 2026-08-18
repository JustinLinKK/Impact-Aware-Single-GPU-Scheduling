from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from localml_scheduler.config import SchedulerSettings, StreamSettings
from localml_scheduler.domain import JobStatus, TrainingJob
from localml_scheduler.execution.backends import StreamBackend
from localml_scheduler.execution.executor import SubprocessExecutor
from localml_scheduler.scheduler.service import SchedulerService
from localml_scheduler.scheduler.service_state import ActiveRun
from localml_scheduler.scheduler.supervisor import WorkerSnapshot
from localml_scheduler.storage.state_store import StateStore


def _job(name: str) -> TrainingJob:
    return TrainingJob.create(
        job_id=name,
        runner_target="builtins:dict",
        baseline_model_id=name,
        baseline_model_path="/tmp/none",
    )


def test_incremental_launches_reuse_one_host(tmp_path: Path) -> None:
    settings = SchedulerSettings(runtime_root=tmp_path)
    settings.gpu_scheduler.stream = StreamSettings(enabled=True)
    backend = StreamBackend(settings, SubprocessExecutor(settings))
    process = Mock()
    process.poll.return_value = None
    with patch.object(
        StreamBackend, "_ensure_host", return_value=process
    ) as ensure_host, patch.object(
        StreamBackend, "_request", return_value={"ok": True, "host_pid": 42}
    ) as request:
        first = backend.launch([_job("stream-a")])[0]
        second = backend.launch([_job("stream-b")])[0]

    assert first.process is second.process is process
    assert first.monitor_via_store and second.monitor_via_store
    assert ensure_host.call_count == 2
    assert request.call_args_list[0].args[0]["job_ids"] == ["stream-a"]
    assert request.call_args_list[1].args[0]["job_ids"] == ["stream-b"]


class _CrashedSharedHostSupervisor:
    def __init__(self, snapshots: list[WorkerSnapshot]):
        self.snapshots = snapshots

    def poll(self):
        snapshots, self.snapshots = self.snapshots, []
        return snapshots

    def active_job_ids_by_group(self):
        return {}

    def active_job_ids(self):
        return []

    def shutdown(self):
        return None


def test_shared_host_crash_marks_every_affected_job_failed(tmp_path: Path) -> None:
    settings = SchedulerSettings(runtime_root=tmp_path)
    store = StateStore(settings)
    jobs = [store.submit_job(_job("crash-a")), store.submit_job(_job("crash-b"))]
    for job in jobs:
        store.set_job_status(
            job.job_id, JobStatus.RUNNING, reason="test stream host", hold=False
        )
    supervisor = _CrashedSharedHostSupervisor(
        [
            WorkerSnapshot(
                job_id=job.job_id,
                group_id="shared-host",
                alive=False,
                returncode=9,
                reported_by="process",
            )
            for job in jobs
        ]
    )
    service = SchedulerService(settings, store=store, supervisor=supervisor)
    service._active_runs["shared-host"] = ActiveRun(
        group_id="shared-host",
        mode="concurrent_group",
        backend_name="stream",
        job_ids=tuple(job.job_id for job in jobs),
    )
    service._poll_active_workers()
    assert all(store.get_job(job.job_id).status.value == "FAILED" for job in jobs)


def test_late_submit_command_cannot_demote_running_stream_job(tmp_path: Path) -> None:
    settings = SchedulerSettings(runtime_root=tmp_path)
    store = StateStore(settings)
    job = store.submit_job(_job("submit-race"))
    store.set_job_status(
        job.job_id, JobStatus.RUNNING, reason="already dispatched", hold=False
    )
    service = SchedulerService(settings, store=store)
    service._handle_submit(job.job_id)
    assert store.get_job(job.job_id).status == JobStatus.RUNNING
