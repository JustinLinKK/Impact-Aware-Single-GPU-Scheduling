"""SYNTHETIC training controls, not natural-job memory or occupancy predictions.

Width and repeated use of a shared residual layer control arithmetic. An explicit
CUDA-only allocation independently controls persistent allocator demand. It is
reported separately from trainable parameters and excluded from checkpoints.
All preparation and graph-capture helpers are CPU safe.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import inspect
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.data import default_collate

from localml_scheduler.adapters import shared_training
from localml_scheduler.adapters.shared_training import CandidateWorkload, CONTRACT, TrainingRecipe

HERE = Path(__file__).resolve().parent
VERSION = "a10-synthetic-resource-mechanisms-v1"
MODEL_FAMILY = "controlled_residual_mlp"
MIB = 1024 ** 2
TEMPLATE_EXCLUDED_FIELDS = frozenset({
    "name", "role", "chain", "seed", "train_pool_seed", "validation_pool_seed",
    "epochs", "steps_per_epoch", "max_steps", "total_updates",
    "spec_sha256", "expected_runtime_signature", "template_key",
})


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def source_metadata():
    return {"workload_version": VERSION,
            "package_pins": {name: file_hash(HERE / name) for name in
                             ("__init__.py", "workloads.py", "candidate.py")},
            "runner_sha256": file_hash(shared_training.__file__),
            "torch_version": torch.__version__}


def spec_digest(spec):
    return digest({k: v for k, v in spec.items()
                   if k not in {"spec_sha256", "expected_runtime_signature"}})


def _integer(spec, key, default=None, *, minimum=1):
    value = spec.get(key, default)
    if type(value) is not int or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return value


def recipe(spec):
    return TrainingRecipe(_integer(spec, "batch"), _integer(spec, "steps_per_epoch"),
                          _integer(spec, "epochs"), _integer(spec, "seed", 42, minimum=0),
                          precision=spec.get("precision", "fp32_ieee"),
                          learning_rate=float(spec.get("learning_rate", 0.001)),
                          weight_decay=float(spec.get("weight_decay", 0.0001)),
                          determinism=spec.get("determinism", "strict"))


def parameter_count(width, input_features=128):
    return input_features * width + width + width * width + width + width + 1


def adapter_context(spec):
    if spec.get("model_family", MODEL_FAMILY) != MODEL_FAMILY:
        raise ValueError("unknown synthetic model_family")
    width = _integer(spec, "width", 128)
    features = _integer(spec, "input_features", 128)
    repetitions = _integer(spec, "repetitions", 2)
    ballast = _integer(spec, "ballast_mib", 0, minimum=0) * MIB
    count = parameter_count(width, features)
    for key in ("parameter_count", "trainable_parameter_count"):
        if key in spec and spec[key] != count:
            raise ValueError("frozen model parameter count changed: " + key)
    training = recipe(spec)
    return {"model_family": MODEL_FAMILY, "input_features": features,
            "width": width, "repetitions": repetitions, "ballast_bytes": ballast,
            "seed": training.seed, "precision": training.precision,
            "lr": training.learning_rate, "weight_decay": training.weight_decay,
            "parameter_count": count, "trainable_parameter_count": count,
            "workload_version": VERSION}


def _data_config(spec):
    train_seed = _integer(spec, "train_pool_seed", 17011, minimum=0)
    validation_seed = _integer(spec, "validation_pool_seed", 29009, minimum=0)
    if train_seed == validation_seed:
        raise ValueError("training and validation pool seeds must be disjoint")
    validation_size = _integer(spec, "validation_size", 256)
    if validation_size not in (128, 256):
        raise ValueError("validation_size must be fixed at 128 or 256")
    return {"train_pool_seed": train_seed, "validation_pool_seed": validation_seed,
            "pool_size": _integer(spec, "pool_size", 4096, minimum=2),
            "validation_size": validation_size,
            "teacher_seed": _integer(spec, "teacher_seed", 33013, minimum=0),
            "virtual_indexing": "index_mod_pool_size",
            "label_recipe": "median_threshold_shared_linear_teacher_v1"}


def template_key(spec):
    """Same frozen configuration across evaluation seeds and update budgets.

    This explicitly permits different pool/initialization seeds for dedicated
    calibration replicas. Every seed still belongs to each workload's identity.
    These measured-template controls do not assess neural prediction accuracy.
    """
    configuration = {k: v for k, v in spec.items() if k not in TEMPLATE_EXCLUDED_FIELDS}
    # Include defaults and exact code bytes even when omitted by the caller.
    context = adapter_context(spec)
    context.pop("seed")
    data = _data_config(spec)
    for key in ("train_pool_seed", "validation_pool_seed"):
        data.pop(key)
    return digest({"spec_configuration": configuration, "adapter_context": context,
                   "data": data, "source": source_metadata(),
                   "training": recipe(spec).prediction_training(),
                   "determinism": recipe(spec).determinism})


def workload_identity(spec):
    if spec.get("spec_sha256") != spec_digest(spec):
        raise ValueError("immutable workload spec hash mismatch")
    actual = source_metadata()
    for key in ("workload_version", "package_pins", "runner_sha256", "torch_version"):
        if key in spec and spec[key] != actual[key]:
            raise ValueError("frozen workload source/runtime changed: " + key)
    return digest({"version": VERSION, "spec_sha256": spec_digest(spec),
                   "architecture": adapter_context(spec), "data": _data_config(spec),
                   "source": actual})


def expected_signature(spec):
    return sha256(json.dumps({"contract": CONTRACT, "recipe": asdict(recipe(spec)),
        "identity": workload_identity(spec), "context": adapter_context(spec),
        "adapter": inspect.getsource(CandidateAdapter),
        "runner_sha256": file_hash(shared_training.__file__)}, sort_keys=True).encode()).hexdigest()


class ControlledMLP(torch.nn.Module):
    def __init__(self, input_features=128, width=128, repetitions=2, ballast_bytes=0):
        super().__init__()
        if min(input_features, width, repetitions) < 1 or ballast_bytes < 0:
            raise ValueError("invalid controlled MLP dimensions or ballast")
        self.input_layer = torch.nn.Linear(input_features, width)
        self.residual_layer = torch.nn.Linear(width, width)
        self.output_layer = torch.nn.Linear(width, 1)
        self.repetitions = repetitions
        self.declared_ballast_bytes = ballast_bytes
        self.register_buffer("_cuda_ballast", torch.empty(0, dtype=torch.uint8), persistent=False)
        self.ballast_initializations = 0

    def _apply(self, fn, recurse=True):
        # Never copy an 8 GiB CUDA buffer to CPU or include it in dtype casts.
        previous = self._cuda_ballast
        self._buffers["_cuda_ballast"] = torch.empty(0, dtype=torch.uint8)
        try:
            result = super()._apply(fn, recurse=recurse)
        except Exception:
            self._buffers["_cuda_ballast"] = previous
            raise
        device = self.input_layer.weight.device
        if device.type == "cuda" and self.declared_ballast_bytes:
            if previous.device == device and previous.numel() == self.declared_ballast_bytes:
                self._buffers["_cuda_ballast"] = previous
            else:
                self._buffers["_cuda_ballast"] = torch.zeros(
                    self.declared_ballast_bytes, dtype=torch.uint8, device=device)
                self.ballast_initializations += 1
        else:
            self._buffers["_cuda_ballast"] = torch.empty(0, dtype=torch.uint8, device=device)
        return result

    def resource_accounting(self):
        return {"declared_ballast_bytes": self.declared_ballast_bytes,
                "allocated_ballast_bytes": self._cuda_ballast.numel() * self._cuda_ballast.element_size(),
                "trainable_parameter_bytes": sum(p.numel() * p.element_size()
                    for p in self.parameters() if p.requires_grad),
                "ballast_initializations_this_launch": self.ballast_initializations,
                "ballast_checkpoint_persistent": False,
                "control_kind": "SYNTHETIC explicit allocator ballast; separate from trainable memory"}

    def forward(self, x):
        hidden = torch.tanh(self.input_layer(x))
        # Bounded activation and step size prevent repeated shared layers from
        # producing an exploding recurrence while preserving real gradients.
        scale = 1.0 / self.repetitions
        for _ in range(self.repetitions):
            hidden = hidden + scale * torch.tanh(self.residual_layer(hidden))
        return self.output_layer(hidden)


def build_model(spec):
    context = adapter_context(spec) if "batch" in spec else spec
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(context["seed"])
        model = ControlledMLP(context["input_features"], context["width"],
                              context["repetitions"], context["ballast_bytes"])
    actual = sum(p.numel() for p in model.parameters())
    if actual != context["parameter_count"]:
        raise ValueError("frozen model parameter count changed")
    return model.float()


class CandidateAdapter:
    def build_model(self, context):
        self.context = dict(context)
        return build_model(context)

    def build_optimizer(self, model, context):
        return torch.optim.AdamW(model.parameters(), lr=context["lr"],
            weight_decay=context["weight_decay"], betas=(.9, .999), eps=1e-8,
            amsgrad=False, foreach=None, fused=None)

    def build_train_batch(self, scenario, device):
        generator = torch.Generator().manual_seed(self.context["seed"])
        batch = int(scenario["batch_size"])
        return {"features": torch.randn(batch, self.context["input_features"], generator=generator).to(device),
                "target": torch.randint(0, 2, (batch, 1), generator=generator).float().to(device)}

    def build_validation_batch(self, scenario, device):
        return self.build_train_batch(scenario, device)

    def training_step(self, model, batch, context):
        with torch.autocast(device_type=batch["features"].device.type, enabled=False):
            return torch.nn.functional.binary_cross_entropy_with_logits(
                model(batch["features"]).float(), batch["target"].float())


class VirtualPoolDataset:
    def __init__(self, features, targets, length):
        if features.device.type != "cpu" or targets.device.type != "cpu":
            raise ValueError("synthetic pools must remain on CPU")
        self.features, self.targets, self.length = features, targets, length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not 0 <= index < self.length:
            raise IndexError(index)
        row = index % len(self.features)
        return {"features": self.features[row], "target": self.targets[row]}


def _pool(size, features, seed, teacher_seed):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(size, features, generator=generator, dtype=torch.float32)
    teacher = torch.randn(features, generator=torch.Generator().manual_seed(teacher_seed))
    scores = x @ teacher / math.sqrt(features)
    y = (scores > scores.median()).float().reshape(-1, 1)
    return x, y


def validation_metrics(logits, targets):
    from sklearn.metrics import roc_auc_score
    logits, targets = logits.detach().float().cpu(), targets.detach().float().cpu()
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise ValueError("nonfinite validation values")
    y = targets.numpy().reshape(-1)
    negative, positive = int((y == 0).sum()), int((y == 1).sum())
    if not negative or not positive or negative + positive != len(y):
        raise ValueError("validation requires both binary classes")
    auc = float(roc_auc_score(y, logits.sigmoid().numpy().reshape(-1)))
    bce = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, targets))
    if not math.isfinite(auc) or not 0 <= auc <= 1 or not math.isfinite(bce):
        raise ValueError("invalid validation feedback")
    return {"negative_validation_auc": -auc, "validation_auc": auc, "validation_bce": bce,
            "validation_negative_count": negative, "validation_positive_count": positive}


def make_workload(spec, output_dir, context=None):
    identity = workload_identity(spec)
    if spec.get("expected_runtime_signature") != expected_signature(spec):
        raise ValueError("frozen runtime signature mismatch")
    training, cfg, ctx = recipe(spec), _data_config(spec), adapter_context(spec)
    train_x, train_y = _pool(cfg["pool_size"], ctx["input_features"],
                            cfg["train_pool_seed"], cfg["teacher_seed"])
    val_x, val_y = _pool(cfg["validation_size"], ctx["input_features"],
                        cfg["validation_pool_seed"], cfg["teacher_seed"])
    train = VirtualPoolDataset(train_x, train_y,
                              max(cfg["pool_size"], training.batch_size * training.steps_per_epoch))
    assert len(train) >= training.batch_size * training.steps_per_epoch
    output_dir = Path(output_dir)

    @torch.no_grad()
    def validate(model, device):
        logits = [model(val_x[start:start + training.batch_size].to(device)).float().cpu()
                  for start in range(0, len(val_x), training.batch_size)]
        return validation_metrics(torch.cat(logits), val_y)

    def finalize(model, device):
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {**model.resource_accounting(), "device": str(device),
                  "data": cfg, "virtual_training_length": len(train),
                  "precision": training.precision, "workload_version": VERSION,
                  "spec_sha256": spec_digest(spec), "signature": expected_signature(spec)}
        (output_dir / "workload_resources.json").write_text(json.dumps(report, indent=2) + "\n")

    return CandidateWorkload(CandidateAdapter(), ctx, train, training, identity, output_dir,
        validate, selection_metric="negative_validation_auc", collate=default_collate, finalize=finalize)


def read_spec_env():
    raw = os.environ.get("A10_DESIGNED_SPEC_JSON", os.environ.get("A10_PILOT_SPEC_JSON"))
    if raw is None:
        raise ValueError("A10_DESIGNED_SPEC_JSON or A10_PILOT_SPEC_JSON is required")
    spec = json.loads(raw)
    if spec.get("spec_sha256") != spec_digest(spec):
        raise ValueError("spec environment differs from frozen identity")
    return spec
