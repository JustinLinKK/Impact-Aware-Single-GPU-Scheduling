import json
import math
import random
import time
from collections.abc import Mapping
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

SEED = 42
IMAGE_SIZE = 256
NUM_FEATURES = 12
BATCH_SIZE = 4
EPOCHS = 3
MAX_STEPS_PER_EPOCH = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
NUM_WORKERS = 0


class PawpularityModel(nn.Module):
    def __init__(self, initial_target_mean=50.0):
        super().__init__()
        layers = []
        in_channels = 3
        for width in (12, 24, 40, 64):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        width,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(4, width),
                    nn.SiLU(),
                ]
            )
            in_channels = width

        self.image_encoder = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 48),
            nn.SiLU(),
        )
        self.tabular_encoder = nn.Sequential(
            nn.Linear(NUM_FEATURES, 24),
            nn.SiLU(),
            nn.Linear(24, 16),
            nn.SiLU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(48 + 16, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

        # Nonzero weights allow gradients into both branches immediately.
        nn.init.normal_(self.regressor[-1].weight, mean=0.0, std=0.01)
        probability = min(max(float(initial_target_mean) / 100.0, 0.01), 0.99)
        nn.init.constant_(
            self.regressor[-1].bias,
            math.log(probability / (1.0 - probability)),
        )

    def forward(self, image, tabular):
        # Explicitly protect the FP32 contract even during external capture.
        with torch.autocast(device_type=image.device.type, enabled=False):
            image_embedding = self.image_encoder(image.float())
            tabular_embedding = self.tabular_encoder(tabular.float())
            fused = torch.cat((image_embedding, tabular_embedding), dim=1)
            logits = self.regressor(fused).squeeze(-1)
            return 100.0 * torch.sigmoid(logits)


class CandidateAdapter:
    """No-argument, import-safe adapter for synthetic scheduler capture."""

    def build_model(self, context):
        if isinstance(context, Mapping):
            initial_mean = context.get("target_mean", 50.0)
        else:
            initial_mean = getattr(context, "target_mean", 50.0)
        return PawpularityModel(initial_target_mean=initial_mean).float()

    def build_optimizer(self, model, context):
        return torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

    @staticmethod
    def _build_batch(scenario, device):
        batch_size = int(scenario["batch_size"])
        if batch_size < 1:
            raise ValueError("scenario['batch_size'] must be positive.")

        fixture = scenario.get("fixture")
        if fixture is None:
            # Private CPU generator: no dataset access or global RNG mutation.
            generator = torch.Generator(device="cpu").manual_seed(SEED)
            fixture = {
                "image": (
                    torch.rand(
                        batch_size,
                        3,
                        IMAGE_SIZE,
                        IMAGE_SIZE,
                        generator=generator,
                        dtype=torch.float32,
                    )
                    * 2.0
                    - 1.0
                ),
                "tabular": (
                    torch.randint(
                        0,
                        2,
                        (batch_size, NUM_FEATURES),
                        generator=generator,
                    ).float()
                    * 2.0
                    - 1.0
                ),
                "target": torch.randint(
                    1, 101, (batch_size,), generator=generator
                ).float(),
            }

        if not isinstance(fixture, Mapping):
            raise TypeError(
                "scenario['fixture'] must map image, tabular, and target "
                "to tensors or array-like values."
            )

        shapes = {
            "image": (3, IMAGE_SIZE, IMAGE_SIZE),
            "tabular": (NUM_FEATURES,),
            "target": (),
        }
        tensors = {}
        for key, trailing_shape in shapes.items():
            if key not in fixture:
                raise KeyError(f"Supplied fixture is missing {key!r}.")
            tensor = torch.as_tensor(
                fixture[key], dtype=torch.float32, device="cpu"
            ).detach()
            if tensor.ndim == len(trailing_shape):
                tensor = tensor.unsqueeze(0)
            if (
                tensor.ndim != len(trailing_shape) + 1
                or tuple(tensor.shape[1:]) != trailing_shape
                or tensor.shape[0] < 1
            ):
                raise ValueError(
                    f"Fixture {key!r} must have shape [N, "
                    f"{', '.join(map(str, trailing_shape))}], N >= 1."
                )
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Fixture {key!r} contains non-finite values.")
            tensors[key] = tensor

        sizes = {tensor.shape[0] for tensor in tensors.values()}
        if len(sizes) != 1:
            raise ValueError("Fixture tensors must share their batch dimension.")

        # Preserve aligned supplied observations while honoring the requested B.
        source_size = tensors["target"].shape[0]
        indices = torch.arange(batch_size, dtype=torch.long) % source_size
        return {
            key: tensor.index_select(0, indices)
            .contiguous()
            .to(device=device, dtype=torch.float32)
            for key, tensor in tensors.items()
        }

    def build_train_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def training_step(self, model, batch, context):
        with torch.autocast(device_type=batch["image"].device.type, enabled=False):
            prediction = model(batch["image"], batch["tabular"]).float()
            target = batch["target"].float()
            return F.mse_loss(prediction, target, reduction="mean")

    def validation_step(self, model, batch, context):
        return self.training_step(model, batch, context)


class PetDataset(Dataset):
    def __init__(self, frame, image_directory, feature_columns, augment=False):
        self.ids = frame["Id"].astype(str).tolist()
        self.image_directory = Path(image_directory)
        self.augment = bool(augment)

        features = frame.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(features).all():
            raise ValueError("Tabular inputs contain non-finite values.")
        # Fixed transformation for binary inputs; no fitted statistics.
        self.features = np.ascontiguousarray(features * 2.0 - 1.0)
        self.targets = None
        if "Pawpularity" in frame:
            self.targets = frame["Pawpularity"].to_numpy(dtype=np.float32, copy=True)
            if not np.isfinite(self.targets).all():
                raise ValueError("Targets contain non-finite values.")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_path = self.image_directory / f"{self.ids[index]}.jpg"
        resampling = getattr(Image, "Resampling", Image)
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=resampling.BILINEAR)
            if self.augment and random.random() < 0.5:
                image = ImageOps.mirror(image)
            pixels = np.array(image, dtype=np.float32, copy=True)

        image_tensor = (
            torch.from_numpy(pixels).permute(2, 0, 1).contiguous().div_(127.5).sub_(1.0)
        )
        batch = {
            "image": image_tensor,
            "tabular": torch.from_numpy(self.features[index].copy()),
        }
        if self.targets is not None:
            batch["target"] = torch.tensor(self.targets[index], dtype=torch.float32)
        return batch


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def move_batch(batch, device):
    return {
        key: value.to(device=device, dtype=torch.float32, non_blocking=True)
        for key, value in batch.items()
    }


@torch.inference_mode()
def predict(model, loader, device):
    """Identical preprocessing, forward pass, and output handling for val/test."""
    model.eval()
    predictions = []
    targets = []

    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        prediction = model(batch["image"], batch["tabular"]).float()
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("Model produced non-finite predictions.")
        predictions.append(prediction.cpu().numpy())
        if "target" in batch:
            targets.append(batch["target"].float().cpu().numpy())

    if not predictions:
        raise ValueError("Inference requires a nonempty dataset.")

    prediction_array = np.concatenate(predictions).astype(np.float32, copy=False)
    target_array = (
        np.concatenate(targets).astype(np.float32, copy=False) if targets else None
    )
    return prediction_array, target_array


def validation_rmse(predictions, targets):
    if targets is None or predictions.shape != targets.shape:
        raise ValueError("Validation predictions and targets must align.")
    score = float(np.sqrt(mean_squared_error(targets, predictions)))
    if not math.isfinite(score):
        raise FloatingPointError("Validation RMSE is not finite.")
    return score


def capture_rng(generators, device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (torch.cuda.get_rng_state_all() if device.type == "cuda" else None),
        "loaders": {
            name: generator.get_state() for name, generator in generators.items()
        },
    }


def restore_rng(state, generators, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    for name, generator in generators.items():
        generator.set_state(state["loaders"][name].cpu())


def save_schedule_and_metrics(events, history, destination):
    """One PNG: measured phase Gantt above per-epoch metric-node graphs."""
    canvas = Image.new("RGB", (1200, 800), "#f7f9fc")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
        title_font = font

    draw.text(
        (28, 20),
        "Pawpularity: observed wall-clock phases",
        fill="#182335",
        font=title_font,
    )
    colors = {"train": "#3778bf", "validation": "#239778", "test": "#d68a27"}
    total_seconds = max(event["end_seconds"] for event in events)
    timeline_left, timeline_right = 245, 1160
    timeline_width = timeline_right - timeline_left

    for tick in range(5):
        x = timeline_left + timeline_width * tick / 4.0
        draw.line((x, 75, x, 345), fill="#dce2ec")
        draw.text(
            (x - 18, 52),
            f"{total_seconds * tick / 4.0:.1f}s",
            fill="#465267",
            font=font,
        )

    for row, event in enumerate(events):
        y = 85 + row * 31
        x0 = timeline_left + (event["start_seconds"] / total_seconds * timeline_width)
        x1 = timeline_left + (event["end_seconds"] / total_seconds * timeline_width)
        draw.text((28, y), event["label"], fill="#182335", font=font)
        draw.rectangle(
            (x0, y, max(x0 + 1, x1), y + 20),
            fill=colors[event["kind"]],
        )

    def draw_metric_panel(left, right, key, title, color):
        top, bottom = 445, 720
        values = [float(row[key]) for row in history]
        lower, upper = min(values), max(values)
        padding = max((upper - lower) * 0.15, abs(upper) * 0.02, 0.01)
        lower -= padding
        upper += padding

        draw.text((left, 390), title, fill="#182335", font=title_font)
        plot_left, plot_right = left + 75, right - 20
        draw.line(
            (plot_left, top, plot_left, bottom, plot_right, bottom),
            fill="#7f8a9b",
            width=2,
        )
        for fraction in (0.0, 0.5, 1.0):
            y = bottom - fraction * (bottom - top)
            value = lower + fraction * (upper - lower)
            draw.line((plot_left, y, plot_right, y), fill="#e0e5ee")
            draw.text((left, y - 8), f"{value:.2f}", fill="#465267", font=font)

        points = []
        for i, value in enumerate(values):
            x = plot_left + i * (plot_right - plot_left) / max(1, len(values) - 1)
            y = bottom - (value - lower) / (upper - lower) * (bottom - top)
            points.append((x, y))
            draw.text(
                (x - 3, bottom + 12),
                str(history[i]["epoch"]),
                fill="#465267",
                font=font,
            )
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        draw.text((plot_left, bottom + 40), "Epoch", fill="#465267", font=font)

    draw_metric_panel(28, 590, "train_mse", "Training MSE", "#3778bf")
    draw_metric_panel(625, 1170, "validation_rmse", "Validation RMSE", "#239778")
    canvas.save(destination)



from localml_scheduler.adapters.shared_training import run_candidate_training
from localml_scheduler.examples.petfinder_workload import prepare_petfinder


def prepare_training():
    return prepare_petfinder(globals(), variant='01')


if __name__ == "__main__":
    run_candidate_training(prepare_training)
