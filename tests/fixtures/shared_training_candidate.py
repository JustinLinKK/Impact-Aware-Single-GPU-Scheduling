"""Small stochastic model/dataset for real subprocess checkpoint regressions."""
import random
import time
from pathlib import Path
import numpy as np
import torch
from localml_scheduler.adapters.shared_training import CandidateWorkload, TrainingRecipe, run_candidate_training

BATCH_SIZE = 4
EPOCHS = 2


class Data:
    def __len__(self):
        return 64

    def __getitem__(self, index):
        time.sleep(0.002)
        return {"x": torch.ones(8) * (index / 64 + random.random() + np.random.random()) + torch.rand(8),
                "y": torch.tensor(float(index % 3))}


class Adapter:
    def build_model(self, context):
        return torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Dropout(0.2), torch.nn.Linear(8, 1))

    def build_optimizer(self, model, context):
        return torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

    def training_step(self, model, batch, context):
        return (model(batch["x"]).flatten() - batch["y"]).square().mean()


def prepare():
    def validate(model, device):
        # Exercises validation RNG persistence, as well as training randomness.
        return {"rmse": float(model(torch.randn(4, 8, device=device)).square().mean().sqrt())}
    return CandidateWorkload(Adapter(), {}, Data(), TrainingRecipe(BATCH_SIZE, 12, EPOCHS),
                             "stochastic-fixture-v1", Path("working/shared_training"), validate)


if __name__ == "__main__":
    run_candidate_training(prepare)
