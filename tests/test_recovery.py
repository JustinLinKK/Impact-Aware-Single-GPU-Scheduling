from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from localml_scheduler.checkpointing.manager import CheckpointManager
from localml_scheduler.examples.toy_pytorch_runner import create_toy_baseline_checkpoint
from localml_scheduler.observability.events import EventLogger
from localml_scheduler.domain import CheckpointPolicy, JobStatus, TrainingJob
from localml_scheduler.scheduler.service import SchedulerService
from localml_scheduler.scheduler.recovery import reconcile_recoverable_jobs
from localml_scheduler.config import SchedulerSettings
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

    def test_restart_auto_resumes_job_parked_during_repack_from_atomic_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            settings = SchedulerSettings(runtime_root=runtime_root, auto_resume_recoverable=True)
            store = SQLiteStateStore(settings)
            event_logger = EventLogger(store, settings.events_jsonl_path)
            checkpoint_manager = CheckpointManager(settings, store, event_logger)
            baseline = create_toy_baseline_checkpoint(runtime_root / "baselines" / "parked.pt", seed=34)
            job = store.submit_job(
                TrainingJob.create(
                    runner_target="localml_scheduler.examples.toy_pytorch_runner:run_toy_training_job",
                    baseline_model_id="parked-repack",
                    baseline_model_path=baseline,
                    runner_kwargs={"batch_size": 8},
                    checkpoint_policy=CheckpointPolicy(save_every_n_steps=1, save_every_epoch=True),
                    metadata={
                        "scheduler_repack_transaction_id": "tx-before-restart",
                        "elastic_contract_validated": True,
                    },
                )
            )
            store.set_job_status(job.job_id, JobStatus.RUNNING, reason="parked at repack barrier", hold=False)
            checkpoint_path = checkpoint_manager.save_checkpoint(
                store.get_job(job.job_id),
                state={"elastic": {"contract_version": 1, "global_step": 4}},
                safe_point_type=CheckpointPolicy().pause_mode,
                epoch=0,
                global_step=4,
                reason="repack checkpoint ready",
            )

            reconcile_recoverable_jobs(store, event_logger, auto_resume=True)
            recovered = store.get_job(job.job_id)
            self.assertEqual(recovered.status, JobStatus.READY)
            self.assertEqual(recovered.latest_checkpoint_path, checkpoint_path)
            self.assertEqual(checkpoint_manager.load_checkpoint(checkpoint_path)["state"]["elastic"]["global_step"], 4)
            self.assertEqual(list(Path(checkpoint_path).parent.glob(".tmp_checkpoint_*")), [])


if __name__ == "__main__":
    unittest.main()
