"""Data/validation adapter shared by the two preserved PetFinder model drafts."""
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import default_collate

from ..adapters.shared_training import CandidateWorkload, TrainingRecipe


def prepare_petfinder(namespace, *, variant, validation_limit=None):
    """Keep each draft's architecture, loss, preprocessing and training split.

    The optional validation limit is explicit smoke-test configuration, never a
    reported model accuracy. Synchronous loading supports exact augmentation
    replay and intentionally replaces the drafts' two-worker loaders.
    """
    root = Path(os.environ.get("MLEVOLVE_INPUT_DIR", "./input")).resolve()
    frame = pd.read_csv(root / "train.csv", dtype={"Id": str})
    features = [c for c in frame if c not in ("Id", "Pawpularity")]
    if len(features) != 12 or frame.Id.isna().any() or not frame.Id.is_unique:
        raise ValueError("invalid PetFinder schema or IDs")
    seed = namespace["SEED"]
    train_indices, val_indices = train_test_split(np.arange(len(frame)), test_size=0.1, random_state=seed, shuffle=True)
    if validation_limit is not None:
        if validation_limit < 1:
            raise ValueError("validation_limit must be positive")
        val_indices = val_indices[:validation_limit]
    training, validation = frame.iloc[train_indices], frame.iloc[val_indices]
    recipe = TrainingRecipe(namespace["BATCH_SIZE"], namespace["MAX_STEPS_PER_EPOCH"], namespace["EPOCHS"],
                            seed, learning_rate=namespace["LEARNING_RATE"], weight_decay=namespace["WEIGHT_DECAY"])
    dataset_type = namespace["PetDataset"]
    if variant == "00":
        train_data = dataset_type(training, root / "train", features, True)
        val_data = dataset_type(validation, root / "train", features, True)
        adapter_context = {"lr": recipe.learning_rate, "weight_decay": recipe.weight_decay,
                           "initial_target_mean": float(training.Pawpularity.mean())}
    elif variant == "01":
        train_data = dataset_type(training, root / "train", features, augment=True)
        val_data = dataset_type(validation, root / "train", features)
        adapter_context = {"target_mean": float(training.Pawpularity.mean())}
    else:
        raise ValueError("unknown PetFinder variant")

    def predict(model, device, dataset):
        values = []
        for start in range(0, len(dataset), recipe.batch_size):
            batch = default_collate([dataset[i] for i in range(start, min(start + recipe.batch_size, len(dataset)))])
            output = model(batch["image"].to(device), batch["tabular"].to(device))
            values.append(output.detach().float().cpu().reshape(-1))
        return torch.cat(values).numpy()

    def validate(model, device):
        prediction = predict(model, device, val_data)
        target = validation.Pawpularity.to_numpy(dtype=np.float32)
        return {"rmse": float(np.sqrt(np.mean(np.square(prediction - target))))}

    output = Path("working/shared_training")

    def finalize(model, device):
        test = pd.read_csv(root / "test.csv", dtype={"Id": str})
        sample = pd.read_csv(root / "sample_submission.csv", dtype={"Id": str})
        if list(sample.columns) != ["Id", "Pawpularity"] or not test.Id.is_unique or not sample.Id.is_unique or set(test.Id) != set(sample.Id):
            raise ValueError("test and sample submission IDs differ")
        test = test.set_index("Id").loc[sample.Id.tolist()].reset_index()
        data = dataset_type(test, root / "test", features, False) if variant == "00" else dataset_type(test, root / "test", features)
        with torch.no_grad():
            sample["Pawpularity"] = predict(model, device, data)
        if not np.isfinite(sample.Pawpularity.to_numpy()).all():
            raise ValueError("non-finite submission")
        destination = Path("submission")
        destination.mkdir(exist_ok=True)
        sample.to_csv(destination / "submission.csv", index=False)

    # Source model/preprocessing functions are included as well as dataset CSV
    # bytes and image identities. No validation labels enter initialization.
    identity = sha256(json.dumps({"version": 1, "variant": variant,
        "dataset": str(root), "csv": sha256((root / "train.csv").read_bytes()).hexdigest(),
        "train": train_indices.tolist(), "validation": val_indices.tolist(),
        "model_source": Path(namespace["__file__"]).read_text(),
        "data_helper": sha256(Path(__file__).read_bytes()).hexdigest(),
        "images": [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted((root / "train").glob("*.jpg"))]}, sort_keys=True).encode()).hexdigest()
    return CandidateWorkload(namespace["CandidateAdapter"](), adapter_context, train_data, recipe,
                             identity, output, validate, finalize=finalize)
