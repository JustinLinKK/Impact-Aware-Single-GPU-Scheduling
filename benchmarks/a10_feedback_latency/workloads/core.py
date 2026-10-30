"""Trace-derived real-image workloads; no synthetic training or feedback fallback."""
from __future__ import annotations
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import inspect
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import default_collate
from localml_scheduler.adapters import shared_training
from localml_scheduler.adapters.shared_training import CandidateWorkload, TrainingRecipe, CONTRACT

HERE = Path(__file__).resolve().parent


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def spec_digest(spec):
    return digest({k: v for k, v in spec.items() if k not in {"spec_sha256", "expected_runtime_signature"}})


def recipe(spec):
    return TrainingRecipe(int(spec["batch"]), int(spec["steps_per_epoch"]), int(spec["epochs"]),
                          int(spec["seed"]), learning_rate=1e-4, weight_decay=1e-5)


def adapter_context(spec):
    return {"model_family": spec["model_family"], "image_size": int(spec["image_size"]),
            "seed": int(spec["seed"]), "lr": 1e-4, "weight_decay": 1e-5,
            "channels_last": spec["model_family"] != "custom_cnn",
            "parameter_count": int(spec["parameter_count"]),
            "trainable_parameter_count": int(spec["trainable_parameter_count"])}


def prediction_metadata(spec):
    """Pure input-only source-capture contract for the frozen actual recipe."""
    from localml_scheduler.prediction.source_contract import CONTRACT_VERSION
    size = int(spec["image_size"])
    return {"entry": "CandidateAdapter", "precision": "fp32_ieee", "epochs": int(spec["epochs"]),
        "dataset_size": 32768, "steps_per_epoch_by_batch": {str(spec["batch"]): int(spec["steps_per_epoch"])},
        "workload_contract": {"version": CONTRACT_VERSION, "mode": "candidate_adapter",
            "entry": "CandidateAdapter", "max_input_numel": 16000000,
            "adapter": {"context": adapter_context(spec), "scenario": {"batch_size": int(spec["batch"])},
                "batch_spec": {"image": {"shape": ["$batch", 3, size, size], "dtype": "float32"},
                               "target": {"shape": ["$batch", 1], "dtype": "float32"}}},
            "loss": {"kind": "candidate_adapter_training_step"},
            "training": {**recipe(spec).prediction_training(), "backend": "pytorch_shared_training_v1"}}}


class CustomCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = self.block(3, 32)
        self.conv_block2 = self.block(32, 64)
        self.conv_block3 = self.block(64, 128)
        self.flatten = torch.nn.Flatten()
        self.classifier = torch.nn.Sequential(torch.nn.Linear(128 * 12 * 12, 256),
            torch.nn.ReLU(), torch.nn.Dropout(.5), torch.nn.Linear(256, 1))

    @staticmethod
    def block(a, b):
        return torch.nn.Sequential(torch.nn.Conv2d(a, b, 3, padding=1),
            torch.nn.BatchNorm2d(b), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2))

    def forward(self, x):
        return self.classifier(self.flatten(self.conv_block3(self.conv_block2(self.conv_block1(x)))))


def build_model(spec):
    context = adapter_context(spec) if "batch" in spec else spec
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(context["seed"])
        if context["model_family"] == "custom_cnn":
            model = CustomCNN()
        else:
            import timm
            model = timm.create_model(context["model_family"], pretrained=False, num_classes=1)
    model = model.float()
    if context.get("channels_last"):
        model = model.to(memory_format=torch.channels_last)
    counts = (sum(p.numel() for p in model.parameters()),
              sum(p.numel() for p in model.parameters() if p.requires_grad))
    for key, actual in zip(("parameter_count", "trainable_parameter_count"), counts):
        if context.get(key) is not None and context[key] != actual:
            raise ValueError("frozen model parameter count changed: " + key)
    return model


class CandidateAdapter:
    def build_model(self, context):
        self.context = dict(context)
        return build_model(context)

    def build_optimizer(self, model, context):
        return torch.optim.AdamW(model.parameters(), lr=context["lr"],
            weight_decay=context["weight_decay"], betas=(.9, .999), eps=1e-8,
            amsgrad=False, foreach=None, fused=None)

    def build_train_batch(self, scenario, device):
        # This synthetic batch is exclusively for CPU predictor graph capture.
        context = self.context
        n, size = int(scenario["batch_size"]), int(context["image_size"])
        generator = torch.Generator().manual_seed(context["seed"])
        x = torch.rand(n, 3, size, size, generator=generator)
        if context["channels_last"]:
            x = x.contiguous(memory_format=torch.channels_last)
        return {"image": x.to(device), "target": torch.randint(0, 2, (n, 1), generator=generator).float().to(device)}

    def build_validation_batch(self, scenario, device):
        return self.build_train_batch(scenario, device)

    def training_step(self, model, batch, context):
        with torch.autocast(device_type=batch["image"].device.type, enabled=False):
            return torch.nn.functional.binary_cross_entropy_with_logits(
                model(batch["image"]).float(), batch["target"].float())


def workload_identity(spec):
    if spec.get("spec_sha256") != spec_digest(spec):
        raise ValueError("immutable workload spec hash mismatch")
    for name, expected in spec["package_pins"].items():
        if file_hash(HERE / name) != expected:
            raise ValueError("frozen workload package changed: " + name)
    return digest({"contract": "a10-trace-derived-cancer-v1", "spec_sha256": spec_digest(spec),
                   "data_identity": spec["data_identity"], "package_pins": spec["package_pins"]})


def expected_signature(spec):
    return sha256(json.dumps({"contract": CONTRACT, "recipe": asdict(recipe(spec)),
        "identity": workload_identity(spec), "context": adapter_context(spec),
        "adapter": inspect.getsource(CandidateAdapter),
        "runner_sha256": file_hash(shared_training.__file__)}, sort_keys=True).encode()).hexdigest()


def image_transform(spec, training):
    from torchvision import transforms as T
    custom = spec["model_family"] == "custom_cnn"
    operations = [] if custom else [T.Resize((spec["image_size"], spec["image_size"]),
        interpolation=T.InterpolationMode.BILINEAR, antialias=True)]
    if training:
        operations += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(90 if custom else 20),
            T.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.05)]
    mean, std = ([.5] * 3, [.5] * 3) if custom else ([.485, .456, .406], [.229, .224, .225])
    return T.Compose(operations + [T.ToTensor(), T.Normalize(mean, std)])


class CancerDataset:
    def __init__(self, ids, labels, root, images, spec, training):
        self.ids, self.labels, self.root, self.images = ids, labels, Path(root), images
        self.transform = image_transform(spec, training)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        name = self.ids[index]
        raw = (self.root / "train" / (name + ".tif")).read_bytes()
        expected = self.images[name]
        if len(raw) != expected["size"] or sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError("image differs from frozen data manifest: " + name)
        with Image.open(BytesIO(raw)) as image:
            x = self.transform(image.convert("RGB"))
        return {"image": x, "target": torch.tensor([self.labels[name]], dtype=torch.float32)}


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
    if expected_signature(spec) != spec["expected_runtime_signature"]:
        raise ValueError("frozen runtime signature mismatch")
    root = Path(os.environ["A10_PILOT_DATA_ROOT"]).resolve()
    info = spec["data_identity"]
    if file_hash(root / "train_labels.csv") != info["labels_sha256"]:
        raise ValueError("label bytes changed")
    if file_hash(root / "cancer-data-manifest.json") != info["images_manifest_sha256"]:
        raise ValueError("image manifest bytes changed")
    split_raw = (HERE / "split.json").read_bytes()
    if sha256(split_raw).hexdigest() != info["split_sha256"]:
        raise ValueError("frozen split bytes changed")
    split = json.loads(split_raw)
    frame = pd.read_csv(root / "train_labels.csv", dtype={"id": str})
    labels = dict(zip(frame.id, frame.label))
    train, heldout, validation = split["training_ids"], split["heldout_ids"], split["validation_ids"]
    if (len(frame) != 40960 or len(labels) != 40960 or len(train) != 32768
        or len(heldout) != 8192 or len(validation) != 256 or len(set(train)) != len(train)
        or len(set(heldout)) != len(heldout) or set(train) & set(heldout)
        or set(train) | set(heldout) != set(labels) or not set(validation) <= set(heldout)
        or len(set(validation)) != 256):
        raise ValueError("invalid frozen data split")
    actual_counts = [sum(labels[x] == y for x in validation) for y in (0, 1)]
    if actual_counts != info["validation_class_counts"] or min(actual_counts) < 1:
        raise ValueError("validation class identity changed")
    manifest = json.loads((root / "cancer-data-manifest.json").read_bytes())
    if set(manifest["images"]) != set(labels):
        raise ValueError("image manifest identities differ from labels")
    train_data = CancerDataset(train, labels, root, manifest["images"], spec, True)
    val_data = CancerDataset(validation, labels, root, manifest["images"], spec, False)
    ctx = adapter_context(spec)

    def collate(rows):
        batch = default_collate(rows)
        if ctx["channels_last"]:
            batch["image"] = batch["image"].contiguous(memory_format=torch.channels_last)
        return batch

    @torch.no_grad()
    def validate(model, device):
        logits, targets = [], []
        for start in range(0, len(val_data), spec["batch"]):
            batch = collate([val_data[i] for i in range(start, min(start + spec["batch"], len(val_data)))])
            logits.append(model(batch["image"].to(device)).detach().float().cpu())
            targets.append(batch["target"])
        return validation_metrics(torch.cat(logits), torch.cat(targets))

    return CandidateWorkload(CandidateAdapter(), ctx, train_data, recipe(spec), identity,
        Path(output_dir), validate, selection_metric="negative_validation_auc", collate=collate)


def read_spec_env():
    spec = json.loads(os.environ["A10_PILOT_SPEC_JSON"])
    if spec.get("spec_sha256") != spec_digest(spec):
        raise ValueError("spec environment differs from frozen identity")
    return spec
