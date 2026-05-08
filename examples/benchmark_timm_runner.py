"""Structured benchmark runner for the scheduler benchmark TIMM workloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import random
import time

import torch
from torch import nn

from ..execution.runner_protocol import RunnerContext
from ..profiling.runtime_probe import (
    estimate_total_runtime_from_epoch_1,
    estimate_total_runtime_from_step_window,
    planned_total_steps,
)
from ..domain import ProgressSnapshot


def _load_benchmark_modules():
    import pandas as pd
    from PIL import Image
    import timm
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    return pd, Image, timm, DataLoader, Dataset, transforms


def _load_state_dict_from_baseline(baseline: Any) -> tuple[str, int, dict[str, torch.Tensor]]:
    if isinstance(baseline, dict) and "state_dict" in baseline:
        model_name = str(baseline.get("model_name") or "")
        num_classes = int(baseline.get("num_classes") or 5)
        state_dict = baseline["state_dict"]
        return model_name, num_classes, state_dict
    if isinstance(baseline, dict):
        return "", 5, baseline
    raise TypeError(f"Unsupported baseline payload type: {type(baseline)!r}")


def _build_dataloader(
    *,
    data_root: str,
    subset_size: int,
    batch_size: int,
    seed: int,
    pin_memory: bool,
):
    pd, Image, _timm, DataLoader, Dataset, transforms = _load_benchmark_modules()

    dataframe = pd.read_csv(f"{data_root}/train.csv").sample(n=subset_size, random_state=seed).reset_index(drop=True)
    image_dir = Path(data_root) / "train_images"
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    class CassavaDataset(Dataset):
        def __len__(self) -> int:
            return len(dataframe)

        def __getitem__(self, index: int):
            row = dataframe.iloc[index]
            image = Image.open(image_dir / row["image_id"]).convert("RGB")
            return transform(image), int(row["label"])

    return DataLoader(
        CassavaDataset(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )


def _build_model(
    *,
    model_name: str,
    num_classes: int,
    baseline_state_dict: dict[str, torch.Tensor],
    device: torch.device,
):
    _pd, _Image, timm, _DataLoader, _Dataset, _transforms = _load_benchmark_modules()
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes).to(device)
    model.load_state_dict(baseline_state_dict, strict=True)
    return model


def probe_timm_benchmark_batch_size(
    context: RunnerContext,
    batch_size: int,
    warmup_steps: int,
    measure_steps: int,
):
    """Probe one candidate batch size using synthetic inputs for fast VRAM sizing."""
    params = {
        "model_name": None,
        "num_classes": 5,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "probe_max_batch_size": None,
    }
    params.update(context.job.config.runner_kwargs)

    base_batch_size = max(1, int(params.get("batch_size", 16)))
    probe_limit = params.get("probe_max_batch_size")
    if probe_limit is not None and int(batch_size) > int(probe_limit):
        estimated_vram = context.job.resource_requirements.estimated_vram_mb
        synthetic_peak = None
        if estimated_vram is not None:
            synthetic_peak = int(float(estimated_vram) * (float(batch_size) / float(base_batch_size)))
        return {
            "fits": False,
            "peak_vram_mb": synthetic_peak,
            "memory_total_mb": None,
            "message": f"batch size {batch_size} exceeds configured probe_max_batch_size {probe_limit}",
        }

    if not torch.cuda.is_available():
        estimated_vram = context.job.resource_requirements.estimated_vram_mb
        synthetic_peak = None
        if estimated_vram is not None:
            synthetic_peak = int(float(estimated_vram) * (float(batch_size) / float(base_batch_size)))
        return {
            "fits": True,
            "peak_vram_mb": synthetic_peak,
            "memory_total_mb": None,
            "avg_step_time_ms": 1.0,
            "message": "synthetic CPU probe result",
        }

    seed = int(params.get("dataset_seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda")
    baseline = context.load_baseline_object()
    baseline_model_name, baseline_num_classes, baseline_state_dict = _load_state_dict_from_baseline(baseline)
    model_name = str(params.get("model_name") or baseline_model_name)
    num_classes = int(params.get("num_classes") or baseline_num_classes)
    warmup = max(1, int(warmup_steps))
    measured = max(1, int(measure_steps))
    total_steps = warmup + measured

    model = None
    optimizer = None
    criterion = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = _build_model(
            model_name=model_name,
            num_classes=num_classes,
            baseline_state_dict=baseline_state_dict,
            device=device,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]))
        criterion = nn.CrossEntropyLoss()

        start_time = None
        for step_index in range(total_steps):
            features = torch.randn(int(batch_size), 3, 224, 224, device=device)
            labels = torch.randint(0, num_classes, (int(batch_size),), device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            if step_index == warmup:
                torch.cuda.synchronize(device)
                start_time = time.perf_counter()
        torch.cuda.synchronize(device)
        elapsed_ms = ((time.perf_counter() - start_time) * 1000.0) if start_time is not None else None
        return {
            "fits": True,
            "peak_vram_mb": int(torch.cuda.max_memory_allocated(device) / (1024 * 1024)),
            "memory_total_mb": int(torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)),
            "avg_step_time_ms": (elapsed_ms / measured) if elapsed_ms is not None else None,
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


def run_timm_benchmark_job(context: RunnerContext) -> dict[str, Any]:
    """Train one TIMM model on the cassava subset used by the benchmark harness."""
    params = {
        "data_root": "",
        "subset_size": 4000,
        "batch_size": 16,
        "epochs": 2,
        "learning_rate": 1e-3,
        "working_dir": ".",
        "dataset_seed": 42,
        "model_name": None,
        "num_classes": 5,
    }
    params.update(context.job.config.runner_kwargs)

    seed = int(params.get("dataset_seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = context.load_baseline_object()
    baseline_model_name, baseline_num_classes, baseline_state_dict = _load_state_dict_from_baseline(baseline)
    model_name = str(params.get("model_name") or baseline_model_name)
    num_classes = int(params.get("num_classes") or baseline_num_classes)
    batch_size = max(1, int(params["batch_size"]))
    subset_size = max(batch_size, int(params["subset_size"]))
    epochs = int(context.job.max_epochs or context.job.config.max_epochs or params["epochs"])
    max_steps = context.job.max_steps or context.job.config.max_steps
    working_dir = Path(str(params["working_dir"])).resolve()
    working_dir.mkdir(parents=True, exist_ok=True)
    backend_name = str(context.job.metadata.get("placement_backend") or "exclusive")

    dataloader = _build_dataloader(
        data_root=str(params["data_root"]),
        subset_size=subset_size,
        batch_size=batch_size,
        seed=seed,
        pin_memory=torch.cuda.is_available(),
    )
    model = _build_model(
        model_name=model_name,
        num_classes=num_classes,
        baseline_state_dict=baseline_state_dict,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]))
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    started_at = time.time()
    runtime_started_at = time.perf_counter()
    last_loss = 0.0
    n_samples = 0
    global_step = 0
    first_epoch_step_durations_ms: list[float] = []
    runtime_profile = context.get_runtime_profile(backend_name=backend_name)
    estimated_total_runtime_seconds = (
        float(runtime_profile.estimated_total_runtime_seconds)
        if runtime_profile is not None and runtime_profile.estimated_total_runtime_seconds is not None
        else None
    )
    profile_ready = runtime_profile is not None
    steps_per_epoch = len(dataloader)
    for epoch_index in range(epochs):
        epoch_started_at = time.perf_counter()
        for features, labels in dataloader:
            if max_steps is not None and global_step >= int(max_steps):
                break
            step_started_at = time.perf_counter()
            features = features.to(device, non_blocking=torch.cuda.is_available())
            labels = labels.to(device, non_blocking=torch.cuda.is_available())
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
            n_samples += int(features.shape[0])
            global_step += 1
            if epoch_index == 0:
                first_epoch_step_durations_ms.append((time.perf_counter() - step_started_at) * 1000.0)

        epoch_elapsed_s = time.perf_counter() - epoch_started_at
        if not profile_ready and global_step > 0:
            avg_step_time_ms = sum(first_epoch_step_durations_ms) / max(1, len(first_epoch_step_durations_ms))
            startup_seconds = max(0.0, epoch_started_at - runtime_started_at)
            if max_steps is not None:
                estimated_total_runtime_seconds = estimate_total_runtime_from_step_window(
                    startup_seconds=startup_seconds,
                    avg_step_time_ms=avg_step_time_ms,
                    total_steps=planned_total_steps(context.job, steps_per_epoch=steps_per_epoch),
                )
                strategy = "step_window"
                confidence = 0.75
            else:
                estimated_total_runtime_seconds = estimate_total_runtime_from_epoch_1(
                    startup_seconds=startup_seconds,
                    epoch_1_seconds=epoch_elapsed_s,
                    total_epochs=epochs,
                )
                strategy = "epoch_1"
                confidence = 0.90
            context.upsert_runtime_profile(
                backend_name=backend_name,
                strategy=strategy,
                startup_seconds=startup_seconds,
                epoch_1_seconds=epoch_elapsed_s,
                steps_per_epoch=steps_per_epoch,
                avg_step_time_ms=avg_step_time_ms,
                estimated_total_runtime_seconds=estimated_total_runtime_seconds,
                confidence=confidence,
                source="probe",
                observations=1,
                metadata={"epoch_index": epoch_index + 1},
            )
            context.store.update_job(
                context.job.job_id,
                metadata_updates={
                    "runtime_estimated_total_runtime_seconds": estimated_total_runtime_seconds,
                    "runtime_profile_strategy": strategy,
                    "runtime_profile_confidence": confidence,
                },
            )
            profile_ready = True
        heartbeat = ProgressSnapshot(
            job_id=context.job.job_id,
            epoch=epoch_index + 1,
            global_step=global_step,
            phase="train",
            metrics={"loss": last_loss},
            last_safe_point="epoch",
            steps_per_epoch=steps_per_epoch,
            avg_step_time_ms=(sum(first_epoch_step_durations_ms) / max(1, len(first_epoch_step_durations_ms))) if first_epoch_step_durations_ms else None,
            estimated_total_runtime_seconds=estimated_total_runtime_seconds,
            remaining_runtime_seconds=max(0.0, estimated_total_runtime_seconds - (time.perf_counter() - runtime_started_at))
            if estimated_total_runtime_seconds is not None
            else None,
        )
        context.control_hook.control_plane.write_heartbeat(heartbeat)
        context.store.update_job(
            context.job.job_id,
            last_heartbeat_at=heartbeat.heartbeat_at,
            metadata_updates={
                "runtime_estimated_total_runtime_seconds": estimated_total_runtime_seconds,
                "runtime_remaining_runtime_seconds": heartbeat.remaining_runtime_seconds,
                "runtime_steps_per_epoch": steps_per_epoch,
                "runtime_avg_step_time_ms": heartbeat.avg_step_time_ms,
            },
        )
        if max_steps is not None and global_step >= int(max_steps):
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elapsed_s = time.time() - started_at
    peak_vram_mb = None
    if torch.cuda.is_available():
        peak_vram_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1)

    metric_path = working_dir / "metric.json"
    metric_path.write_text(
        json.dumps(
            {
                "metric": last_loss,
                "elapsed_s": round(elapsed_s, 2),
                "peak_vram_mib": peak_vram_mb,
                "model": model_name,
                "bs": batch_size,
                "epochs": epochs,
                "subset": subset_size,
                "throughput_sps": round(n_samples / elapsed_s, 1) if elapsed_s > 0 else None,
            }
        ),
        encoding="utf-8",
    )
    submission_dir = working_dir / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    (submission_dir / "submission.csv").write_text("image_id,label\nfoo,0\n", encoding="utf-8")
    return {
        "reason": "benchmark timm job completed",
        "metric_path": str(metric_path),
        "elapsed_s": round(elapsed_s, 3),
        "peak_vram_mb": peak_vram_mb,
        "batch_size": batch_size,
        "model_name": model_name,
    }
