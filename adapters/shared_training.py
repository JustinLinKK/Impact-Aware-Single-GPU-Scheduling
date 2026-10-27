"""Versioned, cooperative training loop for agent-owned models and datasets.

The adapter owns build_model/build_optimizer/training_step; this module owns
optimizer updates, deterministic data position, and scheduler safe points.
Version 1 deliberately supports FP32, one optimizer, accumulation=1, map-style
datasets, and in-process loading. Stateful/iterable datasets require a different
explicit checkpoint contract; they must not silently opt into overlap.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import inspect
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import IterableDataset, default_collate

from ..domain import BatchResolution, SafePointType, utc_now
from ..execution.script_context import script_scheduler_context
from ..prediction.job_contract import prediction_metadata

CONTRACT = "mlevolve.shared_training.v1"


@dataclass(frozen=True)
class TrainingRecipe:
    batch_size: int
    steps_per_epoch: int
    epochs: int
    seed: int = 42
    precision: str = "fp32_ieee"
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    determinism: str = "best_effort"

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.batch_size, self.steps_per_epoch, self.epochs)):
            raise ValueError("batch size, steps and epochs must be positive integers")
        if self.precision != "fp32_ieee":
            raise ValueError("shared_training.v1 supports FP32 only")
        if self.determinism not in {"strict", "best_effort"}:
            raise ValueError("determinism must be strict or best_effort")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative and finite")

    def prediction_training(self):
        return {"precision": self.precision, "gradient_accumulation_steps": 1,
                "optimizer": {"name": "adamw", "lr": self.learning_rate,
                              "weight_decay": self.weight_decay, "betas": [0.9, 0.999],
                              "eps": 1e-8, "amsgrad": False, "foreach": None, "fused": None},
                "scheduler": {"name": "none"}}


@dataclass
class CandidateWorkload:
    adapter: Any
    adapter_context: dict
    train_dataset: Any
    recipe: TrainingRecipe
    # Bind source/dependency bytes, split membership and preprocessing version.
    identity: str
    output_dir: Path
    validate: Callable[[torch.nn.Module, torch.device], dict[str, float]]
    selection_metric: str = "rmse"
    collate: Callable = default_collate
    finalize: Callable | None = None


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_move(v, device) for v in value)
    return value


def _atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _check_runtime_recipe(workload, context):
    if context is None:
        return
    recipe, job = workload.recipe, context.job
    if BatchResolution.resolved_batch_size(job) != recipe.batch_size:
        raise ValueError("runtime batch differs from shared training recipe")
    if job.max_epochs is not None and int(job.max_epochs) != recipe.epochs:
        raise ValueError("runtime epochs differ from shared training recipe")
    if job.max_steps is not None and int(job.max_steps) != recipe.epochs * recipe.steps_per_epoch:
        raise ValueError("max_steps truncates the declared shared training recipe")
    metadata = prediction_metadata(job)
    if not metadata:
        return  # Generic cooperative runner can also use measured profiles.
    contract = metadata.get("workload_contract", {})
    expected = recipe.prediction_training()
    training = contract.get("training", {})
    for key in ("precision", "gradient_accumulation_steps", "optimizer", "scheduler"):
        if training.get(key) != expected[key]:
            raise ValueError(f"predictor/runtime {key} mismatch")
    if contract.get("adapter", {}).get("context", {}) != workload.adapter_context:
        raise ValueError("predictor/runtime adapter context mismatch")
    steps = metadata.get("steps_per_epoch_by_batch", {}).get(str(recipe.batch_size))
    if steps != recipe.steps_per_epoch or metadata.get("epochs") != recipe.epochs:
        raise ValueError("predictor/runtime training budget mismatch")


def _completion_metadata(context):
    if context is None:
        return {}
    store = getattr(context, "store", None)
    current = store.get_job(context.job.job_id) if store is not None else None
    return dict(current.metadata if current is not None else getattr(context.job, "metadata", {}))


def _publish_completion_metadata(context, updates):
    if context is None:
        return
    if not hasattr(context.job, "metadata"):
        context.job.metadata = {}
    context.job.metadata.update(updates)
    store = getattr(context, "store", None)
    if store is not None:
        store.update_job(context.job.job_id, metadata_updates=updates)


def _record_completion_phase(context, phase, seconds, *, identity, epoch, global_step):
    """Keep wall-clock evidence out of the deterministic training checkpoint."""
    if (context is None or phase not in {"validation", "finalization"}
            or not isinstance(identity, str) or not identity
            or not math.isfinite(seconds) or seconds < 0):
        return
    metadata = _completion_metadata(context)
    if metadata.get("runtime_completion_identity") != identity:
        return
    store = getattr(context, "store", None)
    history = list(metadata.get("runtime_phase_timing_history") or [])
    history.append({"phase": phase, "seconds": float(seconds), "identity": identity,
                    "hardware_key": store.hardware_key() if store is not None else None,
                    "epoch": int(epoch), "global_step": int(global_step),
                    "finished_at": utc_now(), "source": f"shared_training.{phase}"})
    updates = {"runtime_phase_timing_history": history[-64:],
               "runtime_nonupdate_phase_counter": int(metadata.get("runtime_nonupdate_phase_counter") or 0) + 1}
    if phase == "finalization":
        updates["runtime_finalization_complete"] = True
    _publish_completion_metadata(context, updates)


def _phase_clock(device):
    # Drain preceding CUDA work before starting, and phase work before stopping.
    # Scheduler safe-point waits are outside both timing boundaries.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def run_candidate_training(prepare: Callable[[], CandidateWorkload], *, context=None):
    """Call directly under __main__; prepare must not allocate CUDA at import.

    Data transforms use the saved global RNGs. Loading is synchronous: no worker
    prefetch can consume unsaved randomness. Batch order is generated privately
    per epoch, so reopening an iterator never consumes the training RNG stream.
    """
    context = context if context is not None else script_scheduler_context()
    # script_scheduler_context configures the allocator budget before CUDA use.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    workload = prepare()
    recipe = workload.recipe
    if not workload.identity or isinstance(workload.train_dataset, IterableDataset):
        raise ValueError("an identified, deterministic map-style dataset is required")
    if len(workload.train_dataset) < recipe.batch_size * recipe.steps_per_epoch:
        raise ValueError("not enough samples for the declared full optimizer batches")
    _check_runtime_recipe(workload, context)
    random.seed(recipe.seed)
    np.random.seed(recipe.seed)
    torch.manual_seed(recipe.seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    # Some valid model operators (CUDA adaptive-pool backward, for example)
    # have no deterministic kernel. Preserve the model and expose this policy
    # instead of promising bitwise GPU equality or silently replacing operators.
    torch.use_deterministic_algorithms(True, warn_only=recipe.determinism == "best_effort")
    device = torch.device("cuda" if (context.job.resource_requirements.requires_gpu if context else torch.cuda.is_available()) else "cpu")
    model = workload.adapter.build_model(workload.adapter_context).float().to(device)
    optimizer = workload.adapter.build_optimizer(model, workload.adapter_context)
    if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1:
        raise ValueError("shared_training.v1 requires one AdamW parameter group")
    trainable = [id(p) for p in model.parameters() if p.requires_grad]
    optimized = [id(p) for p in optimizer.param_groups[0]["params"]]
    if len(optimized) != len(set(optimized)) or set(optimized) != set(trainable):
        raise ValueError("optimizer must cover every trainable parameter exactly once")
    group = optimizer.param_groups[0]
    for key, value in recipe.prediction_training()["optimizer"].items():
        if key == "name":
            continue
        actual = list(group[key]) if key == "betas" else group.get(key)
        if actual != value:
            raise ValueError(f"adapter optimizer differs from recipe: {key}")
    signature = sha256(json.dumps({"contract": CONTRACT, "recipe": asdict(recipe),
        "identity": workload.identity, "context": workload.adapter_context,
        "adapter": inspect.getsource(type(workload.adapter)),
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}, sort_keys=True).encode()).hexdigest()
    global_step, completed_epochs = 0, 0
    history, batch_history = [], []
    best_metric, best_state = float("inf"), None
    resume = context.load_resume_checkpoint() if context else None
    if resume is not None:
        saved = resume["state"]
        if saved.get("contract") != CONTRACT or saved.get("signature") != signature:
            raise ValueError("incompatible checkpoint: source, data or training recipe changed")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        global_step, completed_epochs = int(saved["global_step"]), int(saved["completed_epochs"])
        history, batch_history = saved["history"], saved["batch_history"]
        best_metric, best_state = saved["best_metric"], saved["best_model_state"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])

    completion_metadata = _completion_metadata(context)
    checkpoint_epoch = 0
    if resume is not None and completion_metadata.get("runtime_completion_identity") == signature:
        # A user may restore an older checkpoint while the job metadata still
        # describes a later run. Never let that metadata skip pending work.
        checkpoint_epoch = max(0, min(completed_epochs,
            int(completion_metadata.get("runtime_checkpoint_completed_epoch") or 0)))
    completion_updates = {
        "runtime_completion_identity": signature,
        "runtime_total_optimizer_updates": recipe.epochs * recipe.steps_per_epoch,
        "runtime_planned_epochs": recipe.epochs,
        "runtime_has_finalizer": workload.finalize is not None,
        "runtime_checkpoint_completed_step": global_step,
        "runtime_checkpoint_completed_epoch": checkpoint_epoch,
        "runtime_final_checkpoint_complete": False,
        "runtime_finalization_complete": False,
        "runtime_nonupdate_phase_counter": int(completion_metadata.get("runtime_nonupdate_phase_counter") or 0),
    }
    if completion_metadata.get("runtime_completion_identity") != signature:
        completion_updates["runtime_phase_timing_history"] = []
    _publish_completion_metadata(context, completion_updates)

    def state_factory():
        return {"contract": CONTRACT, "signature": signature,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scheduler_state": None, "scaler_state": None,
                "global_step": global_step, "epoch": global_step // recipe.steps_per_epoch,
                "step_in_epoch": global_step % recipe.steps_per_epoch,
                "completed_epochs": completed_epochs, "recipe": asdict(recipe),
                "history": history, "batch_history": batch_history,
                "best_metric": best_metric, "best_model_state": best_state,
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}

    def safe_point(kind, metrics=None):
        if context:
            context.control_hook.safe_point(kind, epoch=completed_epochs, global_step=global_step,
                steps_per_epoch=recipe.steps_per_epoch, metrics=metrics, state_factory=state_factory)

    safe_point(SafePointType.BEFORE_TRAIN)
    total = recipe.epochs * recipe.steps_per_epoch
    while global_step < total or completed_epochs < recipe.epochs:
        # Validation is a separately committed phase. A pause on the final STEP
        # resumes pending validation instead of skipping/repeating an update.
        if global_step == (completed_epochs + 1) * recipe.steps_per_epoch:
            validation_started = _phase_clock(device)
            model.eval()
            with torch.no_grad():
                metrics = {k: float(v) for k, v in workload.validate(model, device).items()}
            if not metrics or not all(math.isfinite(v) for v in metrics.values()):
                raise ValueError("non-finite validation metric")
            score = metrics[workload.selection_metric]
            if score < best_metric:
                best_metric = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            completed_epochs += 1
            history.append({"epoch": completed_epochs, "global_step": global_step, **metrics})
            print("MLEVOLVE_EPOCH_METRIC " + json.dumps({"epoch": completed_epochs,
                  "metric": score, "metric_name": workload.selection_metric}), flush=True)
            _record_completion_phase(context, "validation", _phase_clock(device) - validation_started,
                identity=signature, epoch=completed_epochs, global_step=global_step)
            safe_point(SafePointType.EPOCH, metrics)
            # A scheduler budget may stop here after validation and its checkpoint,
            # without performing another optimizer update or repeating validation.
            safe_point(SafePointType.MANUAL)
            continue
        epoch, step = divmod(global_step, recipe.steps_per_epoch)
        order = torch.randperm(len(workload.train_dataset), generator=torch.Generator().manual_seed(recipe.seed + epoch))
        indices = order[step * recipe.batch_size:(step + 1) * recipe.batch_size].tolist()
        batch = _move(workload.collate([workload.train_dataset[i] for i in indices]), device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = workload.adapter.training_step(model, batch, workload.adapter_context)
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise ValueError("adapter must return a finite scalar loss")
        loss.backward()
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise ValueError("non-finite gradient")
        optimizer.step()
        global_step += 1
        # Store bounded evidence, not every sample of arbitrarily long training.
        batch_history.append(sha256(json.dumps(indices).encode()).hexdigest())
        batch_history[:] = batch_history[-256:]
        safe_point(SafePointType.STEP)
    # Final training checkpoint retains last-update state, separate from the
    # validation-selected model used for inference/submission below.
    safe_point(SafePointType.EXPLICIT)
    finalization_started = _phase_clock(device)
    workload.output_dir = Path(workload.output_dir)
    _atomic_save(state_factory(), workload.output_dir / "training_state.pt")
    _atomic_save({"model_state": best_state, "metric": best_metric, "signature": signature}, workload.output_dir / "selected_model.pt")
    model.load_state_dict(best_state)
    if workload.finalize:
        workload.finalize(model, device)
    result = {"contract": CONTRACT, "global_step": global_step, "epochs": completed_epochs,
              "selected_metric": best_metric, "metric_name": workload.selection_metric,
              "history": history, "signature": signature, "device": str(device),
              "determinism": recipe.determinism}
    (workload.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Final Validation Score: {best_metric}", flush=True)
    _record_completion_phase(context, "finalization", _phase_clock(device) - finalization_started,
        identity=signature, epoch=completed_epochs, global_step=global_step)
    return result
