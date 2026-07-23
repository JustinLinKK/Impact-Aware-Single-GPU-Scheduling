"""Representative source models for the 12-job PerfSeer scheduler fixture."""

import torch
from torch import nn


class ConvClassifier(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 2, 4),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.body(inputs)


class ResidualClassifier(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.input = nn.Conv2d(3, width, 1)
        self.left = nn.Conv2d(width, width, 3, padding=1)
        self.right = nn.Conv2d(width, width, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(width, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input(inputs)
        hidden = torch.relu(self.left(hidden) + self.right(hidden))
        return self.head(torch.flatten(self.pool(hidden), 1))


class DepthwiseClassifier(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, width, 1),
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.SiLU(),
            nn.Conv2d(width, width * 2, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 2, 4),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.body(inputs)


class MLPClassifier(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(16, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 4),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.body(inputs)


class EmbeddingClassifier(nn.Module):
    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(128, width)
        self.head = nn.Linear(width * 8, 4)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(torch.flatten(self.embedding(tokens), 1))


def build_conv(width: int = 8) -> nn.Module:
    return ConvClassifier(width)


def build_residual(width: int = 8) -> nn.Module:
    return ResidualClassifier(width)


def build_depthwise(width: int = 8) -> nn.Module:
    return DepthwiseClassifier(width)


def build_mlp(width: int = 32) -> nn.Module:
    return MLPClassifier(width)


def build_embedding(width: int = 16) -> nn.Module:
    return EmbeddingClassifier(width)
