"""Toy PyTorch runner demonstrating safe-point pause/resume and baseline reuse."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import random
import time

import torch
from torch import nn

from ..execution.runner_protocol import RunnerContext
from ..schemas import SafePointType


class ToyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def create_toy_baseline_checkpoint(path: str | Path, *, input_dim: int = 16, hidden_dim: int = 32, output_dim: int = 2, seed: int = 13) -> str:
    """Create a small baseline checkpoint used by example and tests."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    model = ToyMLP(input_dim, hidden_dim, output_dim)
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "seed": seed,
        },
        path,
    )
    return str(path)


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def _build_batches(num_samples: int, input_dim: int, output_dim: int, batch_size: int, *, dataset_seed: int, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(dataset_seed)
    features = torch.randn(num_samples, input_dim, generator=generator)
    labels = torch.randint(0, output_dim, (num_samples,), generator=generator)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, num_samples, batch_size):
        x = features[start : start + batch_size].to(device)
        y = labels[start : start + batch_size].to(device)
        batches.append((x, y))
    return batches


def probe_toy_training_batch_size(
    context: RunnerContext,
    batch_size: int,
    warmup_steps: int,
    measure_steps: int,
):
    """Probe a candidate batch size for the toy runner."""
    params = {
        "input_dim": 16,
        "hidden_dim": 32,
        "output_dim": 2,
        "learning_rate": 0.05,
        "optimizer": "sgd",
        "probe_memory_total_mb": 4096,
        "probe_base_memory_mb": 256,
        "probe_memory_per_sample_mb": 32,
        "probe_max_batch_size": None,
    }
    params.update(context.job.config.runner_kwargs)

    limit = params.get("probe_max_batch_size")
    synthetic_memory_total_mb = int(params.get("probe_memory_total_mb", 4096))
    synthetic_peak_vram_mb = int(params.get("probe_base_memory_mb", 256)) + (int(batch_size) * int(params.get("probe_memory_per_sample_mb", 32)))
    if limit is not None and int(batch_size) > int(limit):
        return {
            "fits": False,
            "peak_vram_mb": synthetic_peak_vram_mb,
            "memory_total_mb": synthetic_memory_total_mb,
            "message": f"batch size {batch_size} exceeds configured probe_max_batch_size {limit}",
        }

    if not torch.cuda.is_available():
        return {
            "fits": True,
            "peak_vram_mb": synthetic_peak_vram_mb,
            "memory_total_mb": synthetic_memory_total_mb,
            "avg_step_time_ms": 1.0,
            "message": "synthetic CPU probe result",
        }

    seed = context.job.config.seed if context.job.config.seed is not None else int(params.get("seed", 7))
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda")
    baseline = context.load_baseline_object()
    model = None
    optimizer = None
    criterion = None
    start_time = None
    measured_steps = 0
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = ToyMLP(
            int(baseline.get("input_dim", params["input_dim"])),
            int(baseline.get("hidden_dim", params["hidden_dim"])),
            int(baseline.get("output_dim", params["output_dim"])),
        ).to(device)
        model.load_state_dict(baseline["model_state"])

        optimizer_name = str(params["optimizer"]).lower()
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=float(params["learning_rate"]))
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=float(params["learning_rate"]))
        criterion = nn.CrossEntropyLoss()

        input_dim = int(baseline.get("input_dim", params["input_dim"]))
        output_dim = int(baseline.get("output_dim", params["output_dim"]))
        total_steps = max(1, int(warmup_steps) + int(measure_steps))
        for step_index in range(total_steps):
            features = torch.randn(int(batch_size), input_dim, device=device)
            labels = torch.randint(0, output_dim, (int(batch_size),), device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            if step_index == int(warmup_steps):
                torch.cuda.synchronize(device)
                start_time = time.perf_counter()
            if step_index >= int(warmup_steps):
                measured_steps += 1
        torch.cuda.synchronize(device)
        elapsed_ms = ((time.perf_counter() - start_time) * 1000.0) if start_time is not None and measured_steps > 0 else None
        return {
            "fits": True,
            "peak_vram_mb": int(torch.cuda.max_memory_allocated(device) / (1024 * 1024)),
            "memory_total_mb": int(torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)),
            "avg_step_time_ms": (elapsed_ms / measured_steps) if elapsed_ms is not None and measured_steps > 0 else None,
            "message": "cuda probe completed",
        }
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        return {
            "fits": False,
            "peak_vram_mb": int(torch.cuda.max_memory_allocated(device) / (1024 * 1024)),
            "memory_total_mb": int(torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)),
            "message": str(exc),
        }
    finally:
        del model
        del optimizer
        del criterion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_toy_training_job(context: RunnerContext) -> dict[str, Any]:
    """Run a tiny training loop that cooperates with the scheduler."""
    params = {
        "input_dim": 16,
        "hidden_dim": 32,
        "output_dim": 2,
        "num_samples": 256,
        "batch_size": 16,
        "learning_rate": 0.05,
        "optimizer": "sgd",
        "epochs": 3,
        "sleep_per_step": 0.0,
        "dataset_seed": 123,
        "scheduler_gamma": None,
    }
    params.update(context.job.config.runner_kwargs)

    seed = context.job.config.seed if context.job.config.seed is not None else int(params.get("seed", 7))
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = context.load_baseline_object()
    model = ToyMLP(
        int(baseline.get("input_dim", params["input_dim"])),
        int(baseline.get("hidden_dim", params["hidden_dim"])),
        int(baseline.get("output_dim", params["output_dim"])),
    ).to(device)
    model.load_state_dict(baseline["model_state"])

    optimizer_name = str(params["optimizer"]).lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=float(params["learning_rate"]))
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=float(params["learning_rate"]))

    scheduler = None
    if params.get("scheduler_gamma") is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=float(params["scheduler_gamma"]))

    criterion = nn.CrossEntropyLoss()
    batches = _build_batches(
        int(params["num_samples"]),
        int(params["input_dim"]),
        int(params["output_dim"]),
        int(params["batch_size"]),
        dataset_seed=int(params["dataset_seed"]),
        device=device,
    )
    total_epochs = int(context.job.max_epochs or context.job.config.max_epochs or params["epochs"])
    max_steps = context.job.max_steps or context.job.config.max_steps

    start_epoch = 0
    start_step_in_epoch = 0
    global_step = 0
    resume_payload = context.load_resume_checkpoint()
    if resume_payload is not None:
        state = resume_payload["state"]
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        if scheduler is not None and state.get("scheduler_state") is not None:
            scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state.get("epoch", 0))
        start_step_in_epoch = int(state.get("step_in_epoch", 0))
        global_step = int(state.get("global_step", 0))
        _restore_rng_state(state.get("rng_state"))

    def build_checkpoint_state(epoch: int, step_in_epoch: int, global_step_value: int) -> dict[str, Any]:
        return {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "global_step": global_step_value,
            "rng_state": _capture_rng_state(),
            "job_config": context.job.to_dict(),
        }

    context.control_hook.safe_point(
        SafePointType.BEFORE_TRAIN,
        epoch=start_epoch,
        global_step=global_step,
        metrics={},
        message="before training",
        state_factory=lambda: build_checkpoint_state(start_epoch, start_step_in_epoch, global_step),
    )

    last_loss = 0.0
    for epoch in range(start_epoch, total_epochs):
        step_offset = start_step_in_epoch if epoch == start_epoch else 0
        for step_in_epoch, (features, labels) in enumerate(batches):
            if step_in_epoch < step_offset:
                continue
            if max_steps is not None and global_step >= int(max_steps):
                break
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            last_loss = float(loss.detach().cpu().item())
            global_step += 1
            if float(params["sleep_per_step"]) > 0:
                time.sleep(float(params["sleep_per_step"]))
            context.control_hook.safe_point(
                SafePointType.STEP,
                epoch=epoch,
                global_step=global_step,
                metrics={"loss": last_loss},
                state_factory=lambda epoch=epoch, step_in_epoch=step_in_epoch + 1, global_step=global_step: build_checkpoint_state(epoch, step_in_epoch, global_step),
            )

        if scheduler is not None:
            scheduler.step()
        context.control_hook.safe_point(
            SafePointType.EPOCH,
            epoch=epoch + 1,
            global_step=global_step,
            metrics={"loss": last_loss},
            state_factory=lambda epoch=epoch + 1, global_step=global_step: build_checkpoint_state(epoch, 0, global_step),
        )
        start_step_in_epoch = 0
        if max_steps is not None and global_step >= int(max_steps):
            break

    context.control_hook.safe_point(
        SafePointType.EXPLICIT,
        epoch=min(total_epochs, start_epoch if total_epochs == 0 else total_epochs),
        global_step=global_step,
        metrics={"loss": last_loss},
        message="final checkpoint",
        state_factory=lambda: build_checkpoint_state(min(total_epochs, total_epochs), 0, global_step),
    )
    return {"final_loss": last_loss, "global_step": global_step, "device": str(device)}
