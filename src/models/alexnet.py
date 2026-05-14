"""AlexNet adapted for small (28x28 / 32x32) inputs.

The original AlexNet (Krizhevsky et al., 2012) targets 224x224 with large
strides. We keep its 5-conv + 3-FC skeleton but reduce kernel sizes/strides
so the network does not collapse small inputs to zero spatial extent.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AlexNetCifar(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1, image_size: int = 32) -> None:
        super().__init__()
        # If grayscale, replicate to 3 channels via a 1x1 conv-like adapter
        # so the rest of the network is shared across MNIST and CIFAR-10.
        self.adapter = nn.Identity()
        if in_channels != 3:
            self.adapter = nn.Conv2d(in_channels, 3, kernel_size=1)
        # Pad small inputs (e.g. MNIST 28x28) up to 32x32 so the 3 max-pools
        # plus AdaptiveAvgPool2d((2,2)) yield divisible spatial sizes on MPS.
        self._pad = (32 - image_size) // 2 if image_size < 32 else 0

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16  (28 -> 14)
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 8 (7)
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 4 (3)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 2 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        if self._pad > 0:
            x = F.pad(x, [self._pad] * 4)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)
