from __future__ import annotations

from pathlib import Path
import tempfile
import time
import unittest

from localml_scheduler.api import LocalMLSchedulerAPI
from localml_scheduler.examples.toy_pytorch_runner import create_toy_baseline_checkpoint
from localml_scheduler.schemas import CheckpointPolicy, TrainingJob
from localml_scheduler.settings import SchedulerSettings


def wait_for(predicate, timeout: float = 20.0, interval: float = 0.1) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise TimeoutError("condition not met in time")


class PauseResumeIntegrationTest(unittest.TestCase):
    def test_preemption_and_auto_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            settings = SchedulerSettings(runtime_root=runtime_root, scheduler_poll_interval_seconds=0.05, eager_preload_top_k=2)
            api = LocalMLSchedulerAPI(settings)
            service = api.create_scheduler_service().start(background=True)
            try:
                baseline_dir = runtime_root / "baselines"
                baseline_a = create_toy_baseline_checkpoint(baseline_dir / "a.pt", seed=10)
                baseline_b = create_toy_baseline_checkpoint(baseline_dir / "b.pt", seed=11)

                low = api.submit_job(
                    TrainingJob.create(
                        runner_target="localml_scheduler.examples.toy_pytorch_runner:run_toy_training_job",
                        baseline_model_id="baseline-a",
                        baseline_model_path=baseline_a,
                        priority=1,
                        max_steps=40,
                        runner_kwargs={"sleep_per_step": 0.03},
                        checkpoint_policy=CheckpointPolicy(save_every_n_steps=1, save_every_epoch=True),
                    )
                )
                wait_for(lambda: api.get_job(low.job_id).status.value == "RUNNING")

                high = api.submit_job(
                    TrainingJob.create(
                        runner_target="localml_scheduler.examples.toy_pytorch_runner:run_toy_training_job",
                        baseline_model_id="baseline-b",
                        baseline_model_path=baseline_b,
                        priority=9,
                        max_steps=8,
                        runner_kwargs={"sleep_per_step": 0.02},
                        checkpoint_policy=CheckpointPolicy(save_every_n_steps=1, save_every_epoch=True),
                    )
                )

                wait_for(lambda: api.get_job(high.job_id).status.is_terminal and api.get_job(low.job_id).status.is_terminal, timeout=30.0)
                pause_events = api.store.list_events(job_id=low.job_id, event_type="job_paused")
                self.assertTrue(pause_events, "expected low-priority job to pause at least once")
                self.assertEqual(api.get_job(low.job_id).status.value, "COMPLETED")
                self.assertEqual(api.get_job(high.job_id).status.value, "COMPLETED")
            finally:
                service.stop()


if __name__ == "__main__":
    unittest.main()
