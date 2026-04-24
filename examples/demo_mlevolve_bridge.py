"""Small demo showing MLEvolve-style structured job submission through the scheduler."""

from __future__ import annotations

from pathlib import Path
import tempfile
import time

from ..adapters.mlevolve import submit_mlevolve_job
from ..api import LocalMLSchedulerAPI
from ..examples.toy_pytorch_runner import create_toy_baseline_checkpoint
from ..schemas import CheckpointPolicy, ResourceRequirements
from ..settings import SchedulerSettings


def _wait_for_terminal(api: LocalMLSchedulerAPI, job_ids: list[str], timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = [api.get_job(job_id) for job_id in job_ids]
        if all(job is not None and job.status.is_terminal for job in jobs):
            return
        time.sleep(0.1)
    raise TimeoutError("timed out waiting for jobs to finish")


def main() -> None:
    runtime_root = Path(tempfile.mkdtemp(prefix="localml_scheduler_mlevolve_bridge_"))
    settings = SchedulerSettings(runtime_root=runtime_root, scheduler_poll_interval_seconds=0.1)
    api = LocalMLSchedulerAPI(settings)
    service = api.create_scheduler_service().start(background=True)
    try:
        baseline = create_toy_baseline_checkpoint(runtime_root / "baselines" / "bridge.pt", seed=77)
        shared_requirements = ResourceRequirements(requires_gpu=False, estimated_vram_mb=512, estimated_ram_mb=1024)
        shared_policy = CheckpointPolicy(save_every_n_steps=1, save_every_epoch=True)

        first = submit_mlevolve_job(
            api,
            workflow_id="demo-bridge",
            baseline_model_id="bridge-baseline",
            baseline_model_path=baseline,
            runner_target="localml_scheduler.examples.toy_pytorch_runner:run_toy_training_job",
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
            runner_target="localml_scheduler.examples.toy_pytorch_runner:run_toy_training_job",
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


if __name__ == "__main__":
    main()
