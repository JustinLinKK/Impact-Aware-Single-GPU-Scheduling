from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from localml_scheduler.api import LocalMLSchedulerAPI
from localml_scheduler.checkpointing.manager import CheckpointManager
from localml_scheduler.examples.toy_pytorch_runner import create_toy_baseline_checkpoint
from localml_scheduler.observability.events import EventLogger
from localml_scheduler.schemas import CheckpointPolicy, JobStatus, TrainingJob
from localml_scheduler.scheduler.service import SchedulerService
from localml_scheduler.settings import SchedulerSettings
from localml_scheduler.storage.sqlite_store import SQLiteStateStore


class RecoveryTest(unittest.TestCase):
    def test_restart_marks_jobs_recoverable_or_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            settings = SchedulerSettings(runtime_root=runtime_root, auto_resume_recoverable=False)
            store = SQLiteStateStore(settings)
            event_logger = EventLogger(store, settings.events_jsonl_path)
            checkpoint_manager = CheckpointManager(settings, store, event_logger)

            baseline = create_toy_baseline_checkpoint(runtime_root / "baselines" / "baseline.pt", seed=33)

            recoverable_job = store.submit_job(
                TrainingJob.create(
                    runner_target="localml_scheduler.examples.toy_pytorch_runner:run_toy_training_job",
                    baseline_model_id="recoverable",
                    baseline_model_path=baseline,
                    priority=2,
                    checkpoint_policy=CheckpointPolicy(save_every_n_steps=1, save_every_epoch=True),
                )
            )
            failed_job = store.submit_job(
                TrainingJob.create(
                    runner_target="localml_scheduler.examples.toy_pytorch_runner:run_toy_training_job",
                    baseline_model_id="failed",
                    baseline_model_path=baseline,
                    priority=1,
                )
            )

            store.set_job_status(recoverable_job.job_id, JobStatus.RUNNING, reason="simulated active job", hold=False)
            store.set_job_status(failed_job.job_id, JobStatus.RUNNING, reason="simulated active job", hold=False)

            checkpoint_manager.save_checkpoint(
                store.get_job(recoverable_job.job_id),
                state={"dummy": True},
                safe_point_type=CheckpointPolicy().pause_mode,
                epoch=0,
                global_step=1,
                reason="simulated checkpoint",
            )

            service = SchedulerService(settings, store=store)
            service.start(background=True)
            try:
                recoverable_state = store.get_job(recoverable_job.job_id).status
                failed_state = store.get_job(failed_job.job_id).status
                self.assertEqual(recoverable_state, JobStatus.RECOVERABLE)
                self.assertEqual(failed_state, JobStatus.FAILED)
            finally:
                service.stop()


if __name__ == "__main__":
    unittest.main()
