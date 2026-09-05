"""Synthetic regression with explicit input gaps; also a generated-script fixture."""

from __future__ import annotations

import json
import random
import time

import numpy as np
import torch
from torch import nn

from localml_scheduler.domain import SafePointType
from localml_scheduler.execution.script_context import script_scheduler_context


def run_training(context):
    params = context.job.config.runner_kwargs
    seed = int(params.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    device = torch.device("cuda" if context.job.resource_requirements.requires_gpu else "cpu")
    width = int(params.get("width", 256))
    batch = int(params.get("batch_size", 64))
    steps_per_epoch = int(params.get("steps_per_epoch", 256))
    epochs = int(context.job.max_epochs or params.get("epochs", 3))
    activation = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}[params.get("activation", "relu")]
    model = nn.Sequential(nn.Linear(1024, width), activation(), nn.Linear(width, 1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(8, batch, 1024, generator=generator).to(device)
    y = x[:, :, :8].sum(dim=-1, keepdim=True) / 8**0.5
    global_step = 0
    resume = context.load_resume_checkpoint()
    if resume is not None:
        saved = resume["state"]
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        lr_scheduler.load_state_dict(saved["scheduler_state"])
        global_step = int(saved["global_step"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])

    def state_factory():
        return {
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": lr_scheduler.state_dict(), "scaler_state": None,
            "global_step": global_step, "epoch": global_step // steps_per_epoch,
            "step_in_epoch": global_step % steps_per_epoch,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }

    context.control_hook.safe_point(
        SafePointType.BEFORE_TRAIN, epoch=global_step // steps_per_epoch, global_step=global_step,
        steps_per_epoch=steps_per_epoch, state_factory=state_factory,
    )
    started = time.perf_counter()
    loss_value = 0.0
    while global_step < epochs * steps_per_epoch:
        index = global_step % len(x)
        time.sleep(float(params.get("input_delay_seconds", 0.002)))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(x[index])
            loss = (prediction.float() - y[index]).square().mean()
        loss.backward()
        optimizer.step()
        global_step += 1
        if global_step % steps_per_epoch == 0:
            loss_value = float(loss.detach())
            lr_scheduler.step()
        context.control_hook.safe_point(
            SafePointType.STEP, epoch=global_step // steps_per_epoch, global_step=global_step,
            steps_per_epoch=steps_per_epoch, metrics={"mse": loss_value}, state_factory=state_factory,
        )
        if global_step % steps_per_epoch == 0:
            print('MLEVOLVE_EPOCH_METRIC ' + json.dumps({"epoch": global_step // steps_per_epoch, "metric": loss_value**0.5, "metric_name": "rmse"}), flush=True)
            context.control_hook.safe_point(
                SafePointType.EPOCH, epoch=global_step // steps_per_epoch, global_step=global_step,
                steps_per_epoch=steps_per_epoch, metrics={"rmse": loss_value**0.5}, state_factory=state_factory,
            )
    context.control_hook.safe_point(
        SafePointType.EXPLICIT, epoch=epochs, global_step=global_step,
        steps_per_epoch=steps_per_epoch, state_factory=state_factory,
    )
    print(f"Final Validation Score: {loss_value**0.5}", flush=True)
    return {"rmse": loss_value**0.5, "global_step": global_step, "epochs": epochs, "training_seconds": time.perf_counter() - started}


if __name__ == "__main__":
    context = script_scheduler_context()
    if context is None:
        raise RuntimeError("This synthetic fixture requires a scheduler context")
    run_training(context)
