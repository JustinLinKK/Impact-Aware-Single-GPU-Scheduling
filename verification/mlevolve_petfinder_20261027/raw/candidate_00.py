import math
import os
import random
from collections.abc import Mapping
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, default_collate

SEED = 42
IMAGE_SIZE = 256
NUM_TABULAR = 12
BATCH_SIZE = 4
EPOCHS = 3
MAX_STEPS_PER_EPOCH = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
NUM_WORKERS = 2


def normalize_image(image):
    """Shared normalization for decoded images and synthetic capture inputs."""
    return (image.to(dtype=torch.float32) - 0.5) / 0.5


def collate_samples(samples):
    return {
        name: tensor.to(dtype=torch.float32)
        for name, tensor in default_collate(samples).items()
    }


def move_batch(batch, device):
    device = torch.device(device)
    return {
        name: tensor.to(
            device=device,
            dtype=torch.float32,
            non_blocking=(device.type == "cuda"),
        )
        for name, tensor in batch.items()
    }


class FP32MSE(nn.Module):
    def forward(self, prediction, target):
        prediction = prediction.float()
        target = target.float()
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction/target shape mismatch: "
                f"{prediction.shape} versus {target.shape}"
            )
        return F.mse_loss(prediction, target, reduction="mean")


class PawpularityNet(nn.Module):
    def __init__(self, initial_target_mean=50.0):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5, stride=4, padding=2, bias=False),
            nn.GroupNorm(4, 20),
            nn.SiLU(),
            nn.Conv2d(20, 36, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 36),
            nn.SiLU(),
            nn.Conv2d(36, 60, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 60),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(60 * 2 * 2, 48),
            nn.SiLU(),
        )
        self.tabular_encoder = nn.Sequential(
            nn.Linear(NUM_TABULAR, 24),
            nn.SiLU(),
            nn.Linear(24, 12),
            nn.SiLU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(48 + 12, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

        # Only the training split supplies this initialization in main().
        # Nonzero randomly initialized weights retain both input pathways.
        probability = min(max(float(initial_target_mean) / 100.0, 1e-4), 1 - 1e-4)
        with torch.no_grad():
            self.regressor[-1].bias.fill_(math.log(probability / (1.0 - probability)))

    def forward(self, image, tabular):
        image_features = self.image_encoder(image)
        tabular_features = self.tabular_encoder(tabular)
        fused = torch.cat((image_features, tabular_features), dim=1)
        return 100.0 * torch.sigmoid(self.regressor(fused).squeeze(-1))


def forward_with_loss(model, batch, criterion):
    """One real forward/loss path shared by training, validation and adapters."""
    with torch.autocast(device_type=batch["image"].device.type, enabled=False):
        prediction = model(
            image=batch["image"].float(),
            tabular=batch["tabular"].float(),
        ).float()

        loss = None
        if "target" in batch:
            target = batch["target"].float()
            if prediction.shape != target.shape:
                raise ValueError("Targets must have shape [B].")
            loss = criterion(prediction, target).float().mean()
    return prediction, loss


class CandidateAdapter:
    def __init__(self):
        # Capture batches are synthetic; this root is shared with real loading.
        self.input_dir = Path(os.environ.get("MLEVOLVE_INPUT_DIR", "./input"))

    @staticmethod
    def resolve_context(context):
        resolved = {
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "initial_target_mean": 50.0,
            "criterion": None,
        }
        if context is not None:
            if not isinstance(context, Mapping):
                raise TypeError("context must be a partial mapping.")
            resolved.update(context)
        if resolved["criterion"] is None:
            resolved["criterion"] = FP32MSE()
        return resolved

    def build_model(self, context):
        config = self.resolve_context(context)
        return PawpularityNet(initial_target_mean=config["initial_target_mean"]).float()

    def build_optimizer(self, model, context):
        config = self.resolve_context(context)
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
        )

    @staticmethod
    def _declared_shape(specification, expected, name):
        if isinstance(specification, Mapping):
            shape = specification.get("shape", specification.get("sample_shape"))
        elif isinstance(specification, (torch.Tensor, np.ndarray)):
            shape = specification.shape
        else:
            shape = specification

        if shape is None:
            raise ValueError(f"Fixture input {name!r} requires a shape.")
        if isinstance(shape, int):
            shape = (shape,)
        shape = tuple(shape)

        # Accept either sample shapes or shapes containing a batch dimension.
        # scenario['batch_size'] always determines the actual batch dimension.
        if len(shape) == len(expected) + 1:
            shape = shape[1:]
        shape = tuple(int(dimension) for dimension in shape)

        if shape != expected:
            raise ValueError(
                f"Fixture {name!r} has sample shape {shape}; "
                f"the model requires {expected}."
            )
        return shape

    def _build_batch(self, scenario, device, seed):
        batch_size = int(scenario["batch_size"])
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

        expected = {
            "image": (3, IMAGE_SIZE, IMAGE_SIZE),
            "tabular": (NUM_TABULAR,),
            "target": (),
        }
        fixture = scenario.get("fixture")
        shapes = dict(expected)

        if fixture is not None:
            if not isinstance(fixture, Mapping):
                raise TypeError("fixture must describe named input shapes.")
            if fixture:
                if "inputs" in fixture:
                    if not isinstance(fixture["inputs"], Mapping):
                        raise TypeError("fixture['inputs'] must be a mapping.")
                    named = dict(fixture["inputs"])
                    if "target" in fixture:
                        named["target"] = fixture["target"]
                else:
                    named = dict(fixture)

                unknown = set(named) - set(expected)
                missing = {"image", "tabular"} - set(named)
                if unknown or missing:
                    raise ValueError(
                        f"Invalid fixture names: missing={sorted(missing)}, "
                        f"unknown={sorted(unknown)}"
                    )
                for name, specification in named.items():
                    shapes[name] = self._declared_shape(
                        specification, expected[name], name
                    )

        generator = torch.Generator(device="cpu").manual_seed(seed)
        samples = []
        for _ in range(batch_size):
            # These values are exclusively CPU capture fixtures.
            image = normalize_image(
                torch.rand(shapes["image"], generator=generator, dtype=torch.float32)
            )
            tabular = torch.randint(
                0, 2, shapes["tabular"], generator=generator
            ).float()
            # Construct the scalar target independently, including when omitted.
            target = 1.0 + 99.0 * torch.rand(
                (), generator=generator, dtype=torch.float32
            )
            samples.append({"image": image, "tabular": tabular, "target": target})
        return move_batch(collate_samples(samples), device)

    def build_train_batch(self, scenario, device):
        return self._build_batch(scenario, device, SEED)

    def build_validation_batch(self, scenario, device):
        return self._build_batch(scenario, device, SEED + 1)

    def training_step(self, model, batch, context):
        config = self.resolve_context(context)
        model.train()
        _, loss = forward_with_loss(model, batch, config["criterion"])
        if loss is None:
            raise ValueError("Training requires a target.")
        return loss

    @torch.no_grad()
    def validation_step(self, model, batch, context):
        config = self.resolve_context(context)
        model.eval()
        _, loss = forward_with_loss(model, batch, config["criterion"])
        if loss is None:
            raise ValueError("Validation requires a target.")
        return loss


class PetDataset(Dataset):
    def __init__(self, frame, image_directory, feature_columns, has_target):
        self.ids = frame["Id"].astype(str).tolist()
        self.image_directory = Path(image_directory)

        features = frame.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=True)
        if features.shape[1] != NUM_TABULAR or not np.isfinite(features).all():
            raise ValueError("Expected 12 finite numeric tabular inputs.")
        self.features = torch.from_numpy(features)

        self.targets = None
        if has_target:
            targets = frame["Pawpularity"].to_numpy(dtype=np.float32, copy=True)
            if not np.isfinite(targets).all():
                raise ValueError("Training/validation targets must be finite.")
            self.targets = torch.from_numpy(targets)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        path = self.image_directory / f"{self.ids[index]}.jpg"
        resampling = getattr(Image, "Resampling", Image)
        with Image.open(path) as source:
            rgb = ImageOps.exif_transpose(source).convert("RGB")
            rgb = rgb.resize((IMAGE_SIZE, IMAGE_SIZE), resample=resampling.BILINEAR)
            pixels = np.array(rgb, dtype=np.float32, copy=True)
        pixels /= np.float32(255.0)
        image = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()

        sample = {
            "image": normalize_image(image),
            "tabular": self.features[index],
        }
        if self.targets is not None:
            sample["target"] = self.targets[index]
        return sample


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@torch.inference_mode()
def run_inference(model, loader, device, context):
    """Identical preprocessing, forward execution and outputs for val/test."""
    model.eval()
    criterion = CandidateAdapter.resolve_context(context)["criterion"]
    predictions = []
    targets = []

    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        prediction, loss = forward_with_loss(model, batch, criterion)

        if not torch.isfinite(prediction).all().item():
            raise FloatingPointError("Non-finite inference predictions.")
        if loss is not None and not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite validation MSE.")

        predictions.append(prediction.cpu().numpy())
        if "target" in batch:
            targets.append(batch["target"].cpu().numpy())

    if not predictions:
        raise ValueError("Inference dataset is empty.")

    predictions = np.concatenate(predictions).astype(np.float32, copy=False)
    score = None
    if targets:
        y_true = np.concatenate(targets)
        score = float(np.sqrt(mean_squared_error(y_true, predictions)))
    return predictions, score


def main():
    # Local MLEvolve utilities are required only during actual execution.
    from utils.precision_quality import select_validated_precision
    from utils.training_diagnostics import TrainingDiagnostics

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = CandidateAdapter()
    input_dir = adapter.input_dir
    working_dir = Path("./working")
    submission_dir = Path("./submission")
    working_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)

    train_frame = pd.read_csv(input_dir / "train.csv", dtype={"Id": str})
    test_frame = pd.read_csv(input_dir / "test.csv", dtype={"Id": str})
    sample_submission = pd.read_csv(
        input_dir / "sample_submission.csv", dtype={"Id": str}
    )

    if list(sample_submission.columns) != ["Id", "Pawpularity"]:
        raise ValueError("Expected sample submission columns Id, Pawpularity.")
    for frame in (train_frame, test_frame, sample_submission):
        if frame["Id"].isna().any() or not frame["Id"].is_unique:
            raise ValueError("Each input table must have unique, non-null IDs.")
    if set(test_frame["Id"]) != set(sample_submission["Id"]):
        raise ValueError("Test IDs and sample-submission IDs differ.")

    # Align before constructing the test dataset: every prediction retains its ID.
    test_frame = (
        test_frame.set_index("Id").loc[sample_submission["Id"].tolist()].reset_index()
    )

    feature_columns = [
        column for column in train_frame.columns if column not in ("Id", "Pawpularity")
    ]
    if len(feature_columns) != NUM_TABULAR:
        raise ValueError("train.csv must contain exactly 12 feature columns.")
    if not set(feature_columns).issubset(test_frame.columns):
        raise ValueError("Test data is missing required tabular features.")

    train_indices, validation_indices = train_test_split(
        np.arange(len(train_frame)),
        test_size=0.10,
        random_state=SEED,
        shuffle=True,
    )
    training_frame = train_frame.iloc[train_indices].reset_index(drop=True)
    validation_frame = train_frame.iloc[validation_indices].reset_index(drop=True)

    context = {
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "criterion": FP32MSE(),
        "initial_target_mean": float(training_frame["Pawpularity"].mean()),
    }

    training_dataset = PetDataset(
        training_frame, input_dir / "train", feature_columns, True
    )
    validation_dataset = PetDataset(
        validation_frame, input_dir / "train", feature_columns, True
    )
    test_dataset = PetDataset(test_frame, input_dir / "test", feature_columns, False)

    loader_generator = torch.Generator().manual_seed(SEED)
    loader_options = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
        "persistent_workers": True,
        "worker_init_fn": seed_worker,
        "collate_fn": collate_samples,
        "drop_last": False,
    }
    training_loader = DataLoader(
        training_dataset,
        shuffle=True,
        generator=loader_generator,
        **loader_options,
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    if len(training_dataset) < BATCH_SIZE * MAX_STEPS_PER_EPOCH:
        raise ValueError("Dataset is too small for eight full batches per epoch.")

    protocol = {
        "model_family": "small_cnn_tabular_fusion",
        "cnn_channels": [20, 36, 60],
        "image_embedding": 48,
        "spatial_pool": [2, 2],
        "tabular_widths": [24, 12],
        "fusion_width": 32,
        "image_shape": [3, IMAGE_SIZE, IMAGE_SIZE],
        "tabular_features": feature_columns,
        "preprocessing": "RGB bilinear resize; pixels scaled to [-1,1]",
        "target": "raw Pawpularity",
        "loss": "FP32 mean squared error",
        "optimizer": "AdamW",
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "physical_batch_size": BATCH_SIZE,
        "effective_batch_size": BATCH_SIZE,
        "planned_epochs": EPOCHS,
        "steps_per_epoch": MAX_STEPS_PER_EPOCH,
        "split_seed": SEED,
        "initialization_seed": SEED,
        "validation_fraction": 0.10,
        "training_rows": len(training_dataset),
        "validation_rows": len(validation_dataset),
        "initial_target_mean": context["initial_target_mean"],
    }
    precision_quality = select_validated_precision("fp32", protocol=protocol)
    selected_precision = str(precision_quality["precision"]).lower()
    if selected_precision not in {"fp32", "float32", "torch.float32"}:
        raise RuntimeError("This candidate requires FP32 execution.")

    model = adapter.build_model(context).to(device=device, dtype=torch.float32)
    optimizer = adapter.build_optimizer(model, context)

    settings = {
        **protocol,
        "precision": "fp32",
        "precision_reason": "Explicit workload contract; autocast and TF32 disabled.",
        "precision_quality": precision_quality,
    }
    checkpoint_path = working_dir / "best_pawpularity.pt"
    best_score = math.inf

    with TrainingDiagnostics(
        model, optimizer, scaler=None, settings=settings
    ) as diagnostics:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            loss_sum = 0.0
            example_count = 0
            updates = 0

            # The loader contains the complete training split. Its shuffled
            # permutation changes each epoch; only the update count is bounded.
            for cpu_batch in islice(training_loader, MAX_STEPS_PER_EPOCH):
                batch = move_batch(cpu_batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss = adapter.training_step(model, batch, context)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Non-finite FP32 training loss.")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=10.0, error_if_nonfinite=True
                )
                try:
                    optimizer.step()
                finally:
                    diagnostics.after_update()

                count = batch["target"].shape[0]
                loss_sum += float(loss.detach().item()) * count
                example_count += count
                updates += 1

            if updates != MAX_STEPS_PER_EPOCH:
                raise RuntimeError("The required update budget was not completed.")

            _, validation_score = run_inference(
                model, validation_loader, device, context
            )
            diagnostics.report(epoch=epoch)
            print(
                f"Epoch {epoch}/{EPOCHS}: "
                f"train_mse={loss_sum / example_count:.6f} "
                f"validation_rmse={validation_score:.6f}"
            )

            if math.isfinite(validation_score) and validation_score < best_score:
                best_score = validation_score
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "diagnostics": diagnostics.state_dict(),
                        "epoch": epoch,
                        "validation_rmse": validation_score,
                        "feature_columns": feature_columns,
                        "train_indices": train_indices,
                        "validation_indices": validation_indices,
                        "protocol": protocol,
                        "torch_rng_state": torch.get_rng_state(),
                        "loader_rng_state": loader_generator.get_state(),
                    },
                    checkpoint_path,
                )

        if not math.isfinite(best_score):
            raise RuntimeError("No finite validation checkpoint was produced.")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        diagnostics.load_state_dict(checkpoint["diagnostics"])

    # Final validation and all test predictions use the restored best model.
    _, score = run_inference(model, validation_loader, device, context)
    test_predictions, _ = run_inference(model, test_loader, device, context)

    if len(test_predictions) != len(sample_submission):
        raise RuntimeError("Missing test predictions.")
    if not math.isfinite(score):
        raise FloatingPointError("Final validation RMSE is not finite.")

    submission = sample_submission.copy()
    submission["Pawpularity"] = test_predictions
    submission.to_csv(submission_dir / "submission.csv", index=False)
    return score


if __name__ == "__main__":
    score = main()
    print(f"Final Validation Score: {score}")
