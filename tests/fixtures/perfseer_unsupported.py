"""Executable model whose sine operation is intentionally outside converter coverage."""

import torch
from torch import nn


class UnsupportedModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sin(inputs)


def build_model() -> nn.Module:
    return UnsupportedModel()
