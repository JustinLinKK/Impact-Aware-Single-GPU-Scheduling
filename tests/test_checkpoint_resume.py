from __future__ import annotations

from pathlib import Path
import tempfile
import time
import unittest

from localml_scheduler.api import LocalMLSchedulerAPI
from localml_scheduler.examples.toy_pytorch_runner import create_toy_baseline_checkpoint
from localml_scheduler.settings import SchedulerSettings
from localml_scheduler.schemas import CheckpointPolicy, TrainingJob


def wait_for(predicate, timeout: float = 20.0, interval: float = 0.1) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise TimeoutError("condition not met in time")


class CheckpointResumeIntegrationTest(unittest.TestCase):
    def test_manual_pause_and_resume_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            settings = SchedulerSettings(runtime_root=runtime_root, scheduler_poll_interval_seconds=0.05)
            api = LocalMLSchedulerAPI(settings)
            service = api.create_scheduler_service().start(background=True)
            try:
                baseline = create_toy_baseline_checkpoint(runtime_root / "baselines" / "resume.pt", seed=22)
                job = api.submit_job(
                    TrainingJob.create(
                        runner_target="localml_scheduler.examples.toy_pytorch_runner:run_toy_training_job",
                        baseline_model_id="resume-baseline",
                        baseline_model_path=baseline,
                        priority=3,
                        max_steps=25,
                        runner_kwargs={"sleep_per_step": 0.03},
                        checkpoint_policy=CheckpointPolicy(save_every_n_steps=1, save_every_epoch=True),
                    )
                )

                wait_for(lambda: api.get_job(job.job_id).status.value == "RUNNING")
                api.pause_job(job.job_id)
                wait_for(lambda: api.get_job(job.job_id).status.value == "PAUSED", timeout=20.0)
                paused_job = api.get_job(job.job_id)
                self.assertIsNotNone(paused_job.latest_checkpoint_path)
                self.assertTrue(Path(paused_job.latest_checkpoint_path).exists())

                checkpoint = service.store.latest_checkpoint(job.job_id)
                self.assertIsNotNone(checkpoint)

                api.resume_job(job.job_id)
                wait_for(lambda: api.get_job(job.job_id).status.value == "COMPLETED", timeout=30.0)
            finally:
                service.stop()


if __name__ == "__main__":
    unittest.main()
