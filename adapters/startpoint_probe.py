"""Synthetic startpoint model probes for MLEvolve cold-start backbones."""

from __future__ import annotations

from typing import Any

from ..domain import BatchProbeTrialResult
from ..execution.runner_protocol import RunnerContext


def _shape_hints(context: RunnerContext) -> dict[str, Any]:
    return dict(context.job.batch_probe.shape_hints or {})


def _memory_total_mb(context: RunnerContext) -> int:
    try:
        total = int(context.store.hardware_profile().total_vram_mb)
    except Exception:
        total = 0
    return total if total > 0 else 24576


def _per_sample_mb(hints: dict[str, Any]) -> int:
    modality = str(hints.get("modality") or "generic").lower()
    if modality == "vision":
        try:
            resolution = int(hints.get("input_resolution") or 256)
        except (TypeError, ValueError):
            resolution = 256
        return max(128, int((resolution / 256.0) ** 2 * 256))
    if modality == "text":
        try:
            sequence_length = int(hints.get("sequence_length") or 512)
        except (TypeError, ValueError):
            sequence_length = 512
        return max(96, int((sequence_length / 512.0) * 192))
    if modality == "audio":
        return 192
    return 128


def _touch_synthetic_tensor(hints: dict[str, Any], batch_size: int) -> None:
    try:
        import torch
    except Exception:
        return

    modality = str(hints.get("modality") or "generic").lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = max(1, min(int(batch_size), 4))
    if modality == "vision":
        try:
            hinted_resolution = int(hints.get("input_resolution") or 256)
        except (TypeError, ValueError):
            hinted_resolution = 256
        resolution = max(32, min(hinted_resolution, 128))
        sample = torch.zeros((batch, 3, resolution, resolution), device=device)
    elif modality == "text":
        try:
            hinted_sequence_length = int(hints.get("sequence_length") or 512)
        except (TypeError, ValueError):
            hinted_sequence_length = 512
        sequence_length = max(8, min(hinted_sequence_length, 128))
        sample = torch.zeros((batch, sequence_length), dtype=torch.long, device=device)
    elif modality == "audio":
        sample = torch.zeros((batch, 1, 16000), device=device)
    else:
        sample = torch.zeros((batch, 128), device=device)
    _ = sample.float().mean().item()
    if device == "cuda":
        torch.cuda.synchronize()


def probe_startpoint_batch_size(
    context: RunnerContext,
    batch_size: int,
    warmup_steps: int,
    measure_steps: int,
) -> BatchProbeTrialResult:
    """Estimate a startpoint model batch size from synthetic modality-shaped work."""
    del warmup_steps, measure_steps
    hints = _shape_hints(context)
    _touch_synthetic_tensor(hints, batch_size)
    memory_total_mb = _memory_total_mb(context)
    base_vram_mb = int(hints.get("base_vram_mb") or 768)
    per_sample_mb = int(hints.get("vram_per_sample_mb") or _per_sample_mb(hints))
    peak_vram_mb = base_vram_mb + (max(1, int(batch_size)) * per_sample_mb)
    target_budget_mb = int(context.settings.gpu_scheduler.memory.budget_mb(memory_total_mb))
    fits = peak_vram_mb <= target_budget_mb
    return BatchProbeTrialResult(
        fits=fits,
        peak_vram_mb=peak_vram_mb,
        memory_total_mb=memory_total_mb,
        avg_step_time_ms=1.0 + (0.05 * max(1, int(batch_size))),
        message="synthetic startpoint probe" if fits else "synthetic startpoint probe exceeded memory budget",
    )


def run_startpoint_probe_job(context: RunnerContext) -> dict[str, Any]:
    """Complete after batch-probe preflight has populated the startpoint profile."""
    return {
        "kind": "mlevolve_startpoint_probe",
        "model_key": context.job.batch_probe.model_key,
        "profile_key": context.job.batch_probe.profile_key,
        "resolved_batch_size": context.job.metadata.get("resolved_batch_size"),
    }
