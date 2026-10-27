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
NUM_WORKERS = 2


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


def main():
    # Runtime-only integrations: importing CandidateAdapter never invokes them.
    from utils.training_diagnostics import TrainingDiagnostics
    from utils.precision_quality import select_validated_precision

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_directory = Path("./input")
    working_directory = Path("./working")
    submission_directory = Path("./submission")
    working_directory.mkdir(parents=True, exist_ok=True)
    submission_directory.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_directory / "train.csv", dtype={"Id": str})
    test_frame = pd.read_csv(input_directory / "test.csv", dtype={"Id": str})
    sample = pd.read_csv(input_directory / "sample_submission.csv", dtype={"Id": str})

    if list(sample.columns) != ["Id", "Pawpularity"]:
        raise ValueError("Expected sample submission columns: Id, Pawpularity.")
    if "Id" not in frame or "Pawpularity" not in frame:
        raise ValueError("train.csv requires Id and Pawpularity columns.")

    feature_columns = [
        column for column in frame.columns if column not in ("Id", "Pawpularity")
    ]
    if len(feature_columns) != NUM_FEATURES:
        raise ValueError("Exactly twelve tabular features are required.")
    if set(test_frame.columns) != {"Id", *feature_columns}:
        raise ValueError("Test feature schema does not match training.")

    for name, table in (
        ("train", frame),
        ("test", test_frame),
        ("sample submission", sample),
    ):
        if table["Id"].isna().any() or not table["Id"].is_unique:
            raise ValueError(f"{name} IDs must be nonmissing and unique.")
    if len(sample) != len(test_frame) or set(sample["Id"]) != set(test_frame["Id"]):
        raise ValueError("Sample submission must contain every test ID.")

    # Split first; the output-bias statistic sees training targets only.
    train_indices, validation_indices = train_test_split(
        np.arange(len(frame)),
        test_size=0.10,
        random_state=SEED,
        shuffle=True,
    )
    train_frame = frame.iloc[train_indices].reset_index(drop=True)
    validation_frame = frame.iloc[validation_indices].reset_index(drop=True)

    train_dataset = PetDataset(
        train_frame, input_directory / "train", feature_columns, augment=True
    )
    validation_dataset = PetDataset(
        validation_frame, input_directory / "train", feature_columns
    )
    test_dataset = PetDataset(test_frame, input_directory / "test", feature_columns)
    if len(train_dataset) < BATCH_SIZE * MAX_STEPS_PER_EPOCH:
        raise ValueError("Not enough training rows for eight full batches.")

    generators = {
        name: torch.Generator().manual_seed(SEED + offset)
        for offset, name in enumerate(("train", "validation", "test"))
    }
    common_loader_settings = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
        "worker_init_fn": seed_worker,
        "persistent_workers": False,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        generator=generators["train"],
        **common_loader_settings,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        generator=generators["validation"],
        **common_loader_settings,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        generator=generators["test"],
        **common_loader_settings,
    )

    protocol = {
        "model_family": "small_cnn_tabular",
        "architecture": "cnn_12_24_40_64_pool2_image48_tab24_16_fusion32",
        "pretrained": False,
        "dataset": str(input_directory.resolve()),
        "split_seed": SEED,
        "initialization_seed": SEED,
        "validation_fraction": 0.10,
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "features": feature_columns,
        "physical_batch_size": BATCH_SIZE,
        "effective_batch_size": BATCH_SIZE,
        "planned_epochs": EPOCHS,
        "steps_per_epoch": MAX_STEPS_PER_EPOCH,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "loss": "raw_target_mse_fp32_mean",
        "image_preprocessing": "RGB_resize_bilinear_scale_minus1_plus1",
        "training_augmentation": "horizontal_flip_probability_0.5",
        "tabular_preprocessing": "2*x-1",
    }
    precision_decision = select_validated_precision("fp32", protocol=protocol)
    chosen_precision = str(precision_decision["precision"]).lower()
    if chosen_precision not in {"fp32", "float32", "torch.float32"}:
        raise RuntimeError("This candidate requires FP32 throughout.")
    # The validated FP32 choice disables both autocast and gradient scaling.
    scaler = None

    context = {"target_mean": float(train_dataset.targets.mean(dtype=np.float64))}
    adapter = CandidateAdapter()
    model = adapter.build_model(context).to(device=device, dtype=torch.float32)
    optimizer = adapter.build_optimizer(model, context)

    settings = {
        **protocol,
        "precision": "fp32",
        "precision_reason": "Explicit task contract; no mixed precision or TF32.",
        "precision_quality": precision_decision,
    }
    checkpoint_path = working_directory / "best_pawpularity.pt"
    best_score = math.inf
    history = []
    events = []
    origin = time.perf_counter()

    def timestamp():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter() - origin

    def record_event(label, kind, start):
        events.append(
            {
                "label": label,
                "kind": kind,
                "start_seconds": start,
                "end_seconds": timestamp(),
            }
        )

    with TrainingDiagnostics(
        model, optimizer, scaler=scaler, settings=settings
    ) as diagnostics:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            loss_sum = 0.0
            sample_count = 0
            updates = 0
            phase_start = timestamp()

            # All training rows are eligible; the update budget is explicitly 8.
            for cpu_batch in islice(train_loader, MAX_STEPS_PER_EPOCH):
                batch = move_batch(cpu_batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss = adapter.training_step(model, batch, context)

                if loss.dtype != torch.float32:
                    raise RuntimeError("Training loss must be FP32.")
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        "Non-finite FP32 loss; no optimizer update attempted."
                    )
                loss.backward()

                # Validate gradients without changing finite gradients.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float("inf"),
                    error_if_nonfinite=True,
                )
                try:
                    optimizer.step()
                finally:
                    diagnostics.after_update()

                updates += 1
                count = int(batch["target"].shape[0])
                loss_sum += float(loss.detach().cpu()) * count
                sample_count += count

            if updates != MAX_STEPS_PER_EPOCH:
                raise RuntimeError("The required epoch update budget was not met.")
            record_event(f"Epoch {epoch} train", "train", phase_start)

            phase_start = timestamp()
            validation_predictions, validation_targets = predict(
                model, validation_loader, device
            )
            epoch_score = validation_rmse(validation_predictions, validation_targets)
            record_event(f"Epoch {epoch} validation", "validation", phase_start)
            train_mse = loss_sum / sample_count
            history.append(
                {
                    "epoch": epoch,
                    "train_mse": train_mse,
                    "validation_rmse": epoch_score,
                    "optimizer_updates": updates,
                }
            )

            diagnostics.report(epoch=epoch)
            print(
                f"Epoch {epoch}/{EPOCHS}: "
                f"train_mse={train_mse:.6f} "
                f"validation_rmse={epoch_score:.6f}",
                flush=True,
            )

            if math.isfinite(epoch_score) and epoch_score < best_score:
                best_score = epoch_score
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": None,
                        "diagnostics": diagnostics.state_dict(),
                        "rng": capture_rng(generators, device),
                        "data_position": {
                            "completed_epoch": epoch,
                            "completed_updates": epoch * MAX_STEPS_PER_EPOCH,
                            "next_batch": 0,
                        },
                        "validation_rmse": best_score,
                        "feature_columns": feature_columns,
                        "train_indices": train_indices,
                        "validation_indices": validation_indices,
                        "context": context,
                        "settings": settings,
                    },
                    checkpoint_path,
                )

        if not math.isfinite(best_score):
            raise RuntimeError("Training produced no finite validation checkpoint.")

        # This checkpoint was written by this invocation, never downloaded.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        diagnostics.load_state_dict(checkpoint["diagnostics"])
        restore_rng(checkpoint["rng"], generators, device)

        phase_start = timestamp()
        validation_predictions, validation_targets = predict(
            model, validation_loader, device
        )
        score = validation_rmse(validation_predictions, validation_targets)
        record_event("Best-model validation", "validation", phase_start)

        phase_start = timestamp()
        test_predictions, _ = predict(model, test_loader, device)
        record_event("Test inference", "test", phase_start)

    if len(test_predictions) != len(test_frame):
        raise RuntimeError("Inference did not cover every test sample.")

    prediction_by_id = pd.Series(test_predictions, index=test_frame["Id"].to_numpy())
    submission = sample.loc[:, ["Id"]].copy()
    submission["Pawpularity"] = (
        submission["Id"].map(prediction_by_id).astype(np.float32)
    )
    if not np.isfinite(submission["Pawpularity"].to_numpy()).all():
        raise RuntimeError("Submission contains missing or non-finite predictions.")
    submission.to_csv(submission_directory / "submission.csv", index=False)

    with (working_directory / "run_trace.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "epochs": history,
                "events": events,
                "best_epoch": checkpoint["data_position"]["completed_epoch"],
                "final_validation_rmse": score,
                "total_optimizer_updates": EPOCHS * MAX_STEPS_PER_EPOCH,
            },
            handle,
            indent=2,
        )

    save_schedule_and_metrics(
        events,
        history,
        working_directory / "pawpularity_schedule_metrics.png",
    )
    return score


if __name__ == "__main__":
    score = main()
    print(f"Final Validation Score: {score}")
