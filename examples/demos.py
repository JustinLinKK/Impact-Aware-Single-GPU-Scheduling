"""Consolidated scheduler sample entry points.

Merged from the former ``demo_submit_jobs.py``, ``demo_mlevolve_bridge.py``,
and ``mock_early_stop_sample.py`` modules.

Run with:
    python -m localml_scheduler.examples.demos submit
    python -m localml_scheduler.examples.demos bridge
    python -m localml_scheduler.examples.demos early-stop
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import shutil
import tempfile
import time

from ..adapters.mlevolve import submit_mlevolve_job
from ..client import SchedulerClient
from ..config import SchedulerSettings
from ..domain import CheckpointPolicy, JobMetricSample, ResourceRequirements, TrainingJob
from ..scheduler.early_stop import analyze_metric_plateau
from .toy_pytorch_runner import create_toy_baseline_checkpoint

TOY_RUNNER_TARGET = "localml_scheduler.examples.toy_pytorch_runner:run_toy_training_job"


def _wait_for_terminal(api: SchedulerClient, job_ids: list[str], timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = [api.inspect(job_id) for job_id in job_ids]
        if all(job is not None and job.status.is_terminal for job in jobs):
            return
        time.sleep(0.1)
    raise TimeoutError("timed out waiting for jobs to finish")


def run_submit_jobs_demo() -> None:
    """End-to-end demo: priority queue, shared baselines, cache stats."""
    runtime_root = Path(tempfile.mkdtemp(prefix="localml_scheduler_demo_"))
    settings = SchedulerSettings(
        runtime_root=runtime_root,
        scheduler_poll_interval_seconds=0.1,
        baseline_cache={"warm_queue_top_k": 2},
    )
    api = SchedulerClient(settings)
    service = api.create_service().start(background=True)

    try:
        baseline_dir = runtime_root / "baselines"
        baseline_a = create_toy_baseline_checkpoint(baseline_dir / "baseline_a.pt", seed=11)
        baseline_b = create_toy_baseline_checkpoint(baseline_dir / "baseline_b.pt", seed=29)

        low_priority = api.submit(
            TrainingJob.create(
                runner_target=TOY_RUNNER_TARGET,
                baseline_model_id="toy-baseline-a",
                baseline_model_path=baseline_a,
                priority=1,
                max_steps=24,
                runner_kwargs={"sleep_per_step": 0.05, "learning_rate": 0.03},
                checkpoint_policy=CheckpointPolicy(save_every_n_steps=2, save_every_epoch=True),
                metadata={"demo": "low_priority"},
            )
        )
        shared_baseline = api.submit(
            TrainingJob.create(
                runner_target=TOY_RUNNER_TARGET,
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

        urgent = api.submit(
            TrainingJob.create(
                runner_target=TOY_RUNNER_TARGET,
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


def run_mlevolve_bridge_demo() -> None:
    """MLEvolve-style structured job submission through the scheduler."""
    runtime_root = Path(tempfile.mkdtemp(prefix="localml_scheduler_mlevolve_bridge_"))
    settings = SchedulerSettings(runtime_root=runtime_root, scheduler_poll_interval_seconds=0.1)
    api = SchedulerClient(settings)
    service = api.create_service().start(background=True)
    try:
        baseline = create_toy_baseline_checkpoint(runtime_root / "baselines" / "bridge.pt", seed=77)
        shared_requirements = ResourceRequirements(requires_gpu=False, estimated_vram_mb=512, estimated_ram_mb=1024)
        shared_policy = CheckpointPolicy(save_every_n_steps=1, save_every_epoch=True)

        first = submit_mlevolve_job(
            api,
            workflow_id="demo-bridge",
            baseline_model_id="bridge-baseline",
            baseline_model_path=baseline,
            runner_target=TOY_RUNNER_TARGET,
            runner_kwargs={"sleep_per_step": 0.01, "batch_size": 8},
            resource_requirements=shared_requirements,
            checkpoint_policy=shared_policy,
            packing_family="toy-mlp",
            packing_eligible=True,
            max_steps=10,
            max_epochs=2,
            metadata={"source": "demo_mlevolve_bridge"},
        )
        second = submit_mlevolve_job(
            api,
            workflow_id="demo-bridge",
            baseline_model_id="bridge-baseline",
            baseline_model_path=baseline,
            runner_target=TOY_RUNNER_TARGET,
            runner_kwargs={"sleep_per_step": 0.01, "batch_size": 8, "learning_rate": 0.02},
            resource_requirements=shared_requirements,
            checkpoint_policy=shared_policy,
            packing_family="toy-mlp",
            packing_eligible=True,
            max_steps=10,
            max_epochs=2,
            metadata={"source": "demo_mlevolve_bridge"},
        )
        _wait_for_terminal(api, [first.job_id, second.job_id], timeout=30.0)
    finally:
        service.stop()


def _sample(job_id: str, step: int, loss: float) -> JobMetricSample:
    return JobMetricSample(
        job_id=job_id,
        created_at=f"2026-01-01T00:00:{step:02d}+00:00",
        epoch=max(0, step // 2),
        global_step=step,
        metrics={"val_loss": loss, "lr": 0.001},
    )


def build_plateau_samples() -> list[JobMetricSample]:
    losses = [1.0, 0.82, 0.70, 0.7002, 0.7001, 0.7003]
    return [_sample("mock-plateau-job", step, loss) for step, loss in enumerate(losses, start=1)]


def build_improving_samples() -> list[JobMetricSample]:
    losses = [1.0, 0.82, 0.70, 0.62, 0.55, 0.49]
    return [_sample("mock-improving-job", step, loss) for step, loss in enumerate(losses, start=1)]


def run_mock_early_stop_check() -> dict[str, dict[str, Any]]:
    settings = {
        "metric_key": "val_loss",
        "direction": "minimize",
        "warmup_samples": 2,
        "patience_samples": 3,
        "min_delta": 1e-3,
    }
    plateau = analyze_metric_plateau(build_plateau_samples(), **settings)
    improving = analyze_metric_plateau(build_improving_samples(), **settings)
    return {
        "plateau": plateau.to_dict(),
        "improving": improving.to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="localml_scheduler sample demos")
    parser.add_argument("demo", choices=["submit", "bridge", "early-stop"], help="Which demo to run")
    args = parser.parse_args()
    if args.demo == "submit":
        run_submit_jobs_demo()
    elif args.demo == "bridge":
        run_mlevolve_bridge_demo()
    else:
        print(json.dumps(run_mock_early_stop_check(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
