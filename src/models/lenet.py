"""LeNet-5 (LeCun 1998) adapted to work for 28x28 (MNIST) and 32x32 (CIFAR-10)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1, image_size: int = 28) -> None:
        super().__init__()
        # Original LeNet expects 32x32. For 28x28 inputs we pad to 32x32.
        self.pad = (32 - image_size) // 2 if image_size < 32 else 0
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            x = F.pad(x, [self.pad] * 4)
        x = F.avg_pool2d(F.relu(self.conv1(x)), 2)
        x = F.avg_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
