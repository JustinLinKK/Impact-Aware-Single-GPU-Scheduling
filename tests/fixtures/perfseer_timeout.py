"""Model factory used to verify conversion subprocess timeouts."""

import time

import torch
from torch import nn


class SlowModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.relu()


def build_model() -> nn.Module:
    time.sleep(1.0)
    return SlowModel()
