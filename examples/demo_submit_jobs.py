"""End-to-end demo for localml_scheduler."""

from __future__ import annotations

from pathlib import Path
import json
import shutil
import tempfile
import time

from ..api import LocalMLSchedulerAPI
from ..schemas import CheckpointPolicy, TrainingJob
from ..settings import SchedulerSettings
from .toy_pytorch_runner import create_toy_baseline_checkpoint


def _wait_for_terminal(api: LocalMLSchedulerAPI, job_ids: list[str], timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = [api.get_job(job_id) for job_id in job_ids]
        if all(job is not None and job.status.is_terminal for job in jobs):
            return
        time.sleep(0.2)
    raise TimeoutError("Jobs did not finish in time")


def main() -> None:
    runtime_root = Path(tempfile.mkdtemp(prefix="localml_scheduler_demo_"))
    settings = SchedulerSettings(runtime_root=runtime_root, scheduler_poll_interval_seconds=0.1, eager_preload_top_k=2)
    api = LocalMLSchedulerAPI(settings)
    service = api.create_scheduler_service().start(background=True)

    try:
        baseline_dir = runtime_root / "baselines"
        baseline_a = create_toy_baseline_checkpoint(baseline_dir / "baseline_a.pt", seed=11)
        baseline_b = create_toy_baseline_checkpoint(baseline_dir / "baseline_b.pt", seed=29)

        low_priority = api.submit_job(
            TrainingJob.create(
                runner_target="localml_scheduler.examples.toy_pytorch_runner:run_toy_training_job",
                baseline_model_id="toy-baseline-a",
                baseline_model_path=baseline_a,
                priority=1,
                max_steps=24,
                runner_kwargs={"sleep_per_step": 0.05, "learning_rate": 0.03},
                checkpoint_policy=CheckpointPolicy(save_every_n_steps=2, save_every_epoch=True),
                metadata={"demo": "low_priority"},
            )
        )
        shared_baseline = api.submit_job(
            TrainingJob.create(
                runner_target="localml_scheduler.examples.toy_pytorch_runner:run_toy_training_job",
                baseline_model_id="toy-baseline-a",
                baseline_model_path=baseline_a,
                priority=2,
                max_steps=12,
                runner_kwargs={"sleep_per_step": 0.02, "learning_rate": 0.02},
                checkpoint_policy=CheckpointPolicy(save_every_n_steps=2, save_every_epoch=True),
                metadata={"demo": "shared_baseline"},
            )
        )

        time.sleep(0.5)

        urgent = api.submit_job(
            TrainingJob.create(
                runner_target="localml_scheduler.examples.toy_pytorch_runner:run_toy_training_job",
                baseline_model_id="toy-baseline-b",
                baseline_model_path=baseline_b,
                priority=9,
                max_steps=10,
                runner_kwargs={"sleep_per_step": 0.03, "optimizer": "adam"},
                checkpoint_policy=CheckpointPolicy(save_every_n_steps=1, save_every_epoch=True),
                metadata={"demo": "urgent"},
            )
        )

        _wait_for_terminal(api, [low_priority.job_id, shared_baseline.job_id, urgent.job_id])

        print("Jobs:")
        print(api.dump_jobs_json())
        print("\nCache stats:")
        print(json.dumps(api.cache_stats(), indent=2, sort_keys=True))
        print("\nReport:")
        print(json.dumps(api.report(), indent=2, sort_keys=True))
    finally:
        service.stop()
        shutil.rmtree(runtime_root, ignore_errors=True)


if __name__ == "__main__":
    main()
