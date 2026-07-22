"""Mandatory elastic-training contract for scheduler-managed MLEvolve scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator
import json
import os
import random
import time

import torch
from torch.utils.data import DataLoader, Sampler

from .domain import SafePointType
from .execution.worker_runtime import create_runner_context, load_runtime_settings
from .observability.events import EventLogger
from .storage.log_store import SchedulerLogStore
from .storage.state_store import StateStore


class ResumableRandomSampler(Sampler[int]):
    """Deterministic sampler whose permutation and position survive a restart."""

    def __init__(self, data_source: Any, *, seed: int, shuffle: bool = True):
        self.data_source = data_source
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.position = 0
        self._indices: list[int] = []
        self._reset_indices()

    def _reset_indices(self) -> None:
        self._indices = list(range(len(self.data_source)))
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            self._indices = torch.randperm(len(self._indices), generator=generator).tolist()

    def __iter__(self) -> Iterator[int]:
        while self.position < len(self._indices):
            value = self._indices[self.position]
            self.position += 1
            yield value
        self.epoch += 1
        self.position = 0
        self._reset_indices()

    def __len__(self) -> int:
        return max(0, len(self._indices) - self.position)

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "shuffle": self.shuffle,
            "epoch": self.epoch,
            "position": self.position,
            "indices": list(self._indices),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.seed = int(state["seed"])
        self.shuffle = bool(state["shuffle"])
        self.epoch = int(state["epoch"])
        self.position = int(state["position"])
        self._indices = [int(item) for item in state["indices"]]


@dataclass(slots=True)
class _RegisteredState:
    model: Any
    optimizer: Any
    lr_scheduler: Any | None
    scaler: Any | None
    extra_state: dict[str, Any] | Callable[[], dict[str, Any]] | None
    extra_state_loader: Callable[[dict[str, Any]], None] | None


class ElasticTrainingSession:
    """Own batch selection, state capture, safe points, and profile reporting."""

    def __init__(self, context: Any):
        self.context = context
        override = os.environ.get("MLEVOLVE_BATCH_SIZE_OVERRIDE")
        self.batch_size = int(override) if override is not None else int(context.job.current_batch_size)
        self.authored_batch_size = int(context.job.authored_batch_size)
        self._registered: _RegisteredState | None = None
        self._sampler: Any | None = None
        self._step_started_at = time.perf_counter()
        self._training_started_at = self._step_started_at
        self._training_samples = 0
        self._probe_durations: list[float] = []
        self._probe_step_samples: list[int] = []

    @classmethod
    def from_env(cls) -> "ElasticTrainingSession":
        runtime_root = os.environ.get("LOCALML_SCHEDULER_RUNTIME_ROOT")
        job_id = os.environ.get("LOCALML_SCHEDULER_JOB_ID")
        if not runtime_root or not job_id:
            raise RuntimeError("elastic training requires LOCALML_SCHEDULER_RUNTIME_ROOT and LOCALML_SCHEDULER_JOB_ID")
        settings = load_runtime_settings(runtime_root)
        store = StateStore(settings)
        event_logger = EventLogger(store, settings.events_jsonl_path, log_store=SchedulerLogStore(settings))
        context, _ = create_runner_context(settings, store, event_logger, job_id)
        if context is None:
            raise KeyError(f"Unknown scheduler job: {job_id}")
        return cls(context)

    def make_dataloader(self, dataset: Any, **kwargs: Any) -> DataLoader:
        if "batch_size" in kwargs:
            raise ValueError("ElasticTrainingSession owns DataLoader batch_size")
        supplied_sampler = kwargs.pop("sampler", None)
        shuffle = bool(kwargs.pop("shuffle", supplied_sampler is None))
        if supplied_sampler is None:
            seed = int(self.context.job.config.seed or 0)
            supplied_sampler = ResumableRandomSampler(dataset, seed=seed, shuffle=shuffle)
        elif not hasattr(supplied_sampler, "state_dict") or not hasattr(supplied_sampler, "load_state_dict"):
            raise TypeError("custom elastic samplers must implement state_dict/load_state_dict")
        self._sampler = supplied_sampler
        return DataLoader(dataset, batch_size=self.batch_size, sampler=supplied_sampler, shuffle=False, **kwargs)

    def register_training_state(
        self,
        model: Any,
        optimizer: Any,
        *,
        lr_scheduler: Any | None = None,
        scaler: Any | None = None,
        extra_state: dict[str, Any] | Callable[[], dict[str, Any]] | None = None,
        extra_state_loader: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._registered = _RegisteredState(model, optimizer, lr_scheduler, scaler, extra_state, extra_state_loader)

    def _rng_state(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
        }
        try:
            import numpy as np

            payload["numpy"] = np.random.get_state()
        except Exception:
            pass
        if torch.cuda.is_available():
            payload["torch_cuda"] = torch.cuda.get_rng_state_all()
        return payload

    def _restore_rng_state(self, payload: dict[str, Any]) -> None:
        if payload.get("python") is not None:
            random.setstate(payload["python"])
        if payload.get("torch_cpu") is not None:
            torch.set_rng_state(payload["torch_cpu"])
        if payload.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["torch_cuda"])
        if payload.get("numpy") is not None:
            try:
                import numpy as np

                np.random.set_state(payload["numpy"])
            except Exception:
                pass

    def state_dict(self, *, epoch: int, batch_index: int, global_step: int) -> dict[str, Any]:
        if self._registered is None:
            raise RuntimeError("register_training_state must be called before the first safe point")
        registered = self._registered
        extra = registered.extra_state() if callable(registered.extra_state) else dict(registered.extra_state or {})
        return {
            "elastic": {
                "contract_version": 1,
                "authored_batch_size": self.authored_batch_size,
                "batch_size_at_checkpoint": self.batch_size,
                "epoch": int(epoch),
                "batch_index": int(batch_index),
                "global_step": int(global_step),
                "model": registered.model.state_dict(),
                "optimizer": registered.optimizer.state_dict(),
                "lr_scheduler": registered.lr_scheduler.state_dict() if registered.lr_scheduler is not None else None,
                "scaler": registered.scaler.state_dict() if registered.scaler is not None else None,
                "sampler": self._sampler.state_dict() if self._sampler is not None else None,
                "rng": self._rng_state(),
                "extra_state": extra,
            }
        }

    def restore_if_present(self) -> dict[str, int]:
        payload = self.context.load_resume_checkpoint()
        if payload is None:
            return {"epoch": 0, "batch_index": 0, "global_step": 0}
        if self._registered is None:
            raise RuntimeError("register_training_state must be called before restore_if_present")
        elastic = dict((payload.get("state") or {}).get("elastic") or {})
        if not elastic:
            raise RuntimeError("checkpoint does not contain elastic contract state")
        registered = self._registered
        registered.model.load_state_dict(elastic["model"])
        registered.optimizer.load_state_dict(elastic["optimizer"])
        if registered.lr_scheduler is not None and elastic.get("lr_scheduler") is not None:
            registered.lr_scheduler.load_state_dict(elastic["lr_scheduler"])
        if registered.scaler is not None and elastic.get("scaler") is not None:
            registered.scaler.load_state_dict(elastic["scaler"])
        if self._sampler is not None and elastic.get("sampler") is not None:
            self._sampler.load_state_dict(elastic["sampler"])
        self._restore_rng_state(dict(elastic.get("rng") or {}))
        if registered.extra_state_loader is not None:
            registered.extra_state_loader(dict(elastic.get("extra_state") or {}))
        return {
            "epoch": int(elastic.get("epoch") or 0),
            "batch_index": int(elastic.get("batch_index") or 0),
            "global_step": int(elastic.get("global_step") or 0),
        }

    def optimizer_step_completed(
        self,
        samples: int,
        epoch: int,
        batch_index: int,
        global_step: int,
        *,
        metrics: dict[str, float] | None = None,
    ) -> None:
        now = time.perf_counter()
        duration = max(1e-9, now - self._step_started_at)
        self._step_started_at = now
        self._probe_durations.append(duration)
        step_samples = max(0, int(samples))
        self._probe_step_samples.append(step_samples)
        self._training_samples += step_samples
        elapsed = max(1e-9, now - self._training_started_at)
        self.context.store.update_job(
            self.context.job.job_id,
            metadata_updates={
                "runtime_avg_step_time_ms": duration * 1000.0,
                "runtime_samples_per_second": self._training_samples / elapsed,
                "runtime_observed_samples": self._training_samples,
            },
        )
        self.context.control_hook.safe_point(
            SafePointType.STEP,
            epoch=int(epoch),
            global_step=int(global_step),
            metrics=metrics,
            avg_step_time_ms=duration * 1000.0,
            state_factory=lambda: self.state_dict(epoch=epoch, batch_index=batch_index, global_step=global_step),
        )
        self._maybe_complete_probe(global_step=global_step)

    def _maybe_complete_probe(self, *, global_step: int) -> None:
        if os.environ.get("MLEVOLVE_PROBE_MODE") != "1":
            return
        warmup = max(0, int(os.environ.get("MLEVOLVE_PROBE_WARMUP_STEPS", "2")))
        measured = max(1, int(os.environ.get("MLEVOLVE_PROBE_MEASURE_STEPS", "5")))
        if len(self._probe_durations) < warmup + measured:
            return
        durations = self._probe_durations[warmup : warmup + measured]
        total_seconds = sum(durations)
        measured_samples = sum(self._probe_step_samples[warmup : warmup + measured])
        allocated = reserved = total = None
        if torch.cuda.is_available():
            allocated = int(torch.cuda.max_memory_allocated())
            reserved = int(torch.cuda.max_memory_reserved())
            total = int(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)
        event = {
            "event": "elastic_probe_completed",
            "global_step": int(global_step),
            "batch_size": self.batch_size,
            "measured_steps": len(durations),
            "samples_per_second": measured_samples / max(total_seconds, 1e-9),
            "median_step_time_ms": sorted(durations)[len(durations) // 2] * 1000.0,
            "step_durations_ms": [item * 1000.0 for item in durations],
            "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved,
            "memory_total_bytes": total,
        }
        event_path = os.environ.get("MLEVOLVE_PROBE_EVENT_PATH")
        if event_path:
            path = Path(event_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        raise SystemExit(0)


__all__ = ["ElasticTrainingSession", "ResumableRandomSampler"]
