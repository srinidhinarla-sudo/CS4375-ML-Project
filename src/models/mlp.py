"""3-layer fully connected baseline."""
from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1, image_size: int = 28,
                 hidden: tuple[int, int] = (512, 256), dropout: float = 0.2) -> None:
        super().__init__()
        in_features = in_channels * image_size * image_size
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
