"""VGG-11 adapted for 28x28 / 32x32 inputs with BatchNorm.

Follows Simonyan & Zisserman (2014) configuration "A" but uses a smaller
classifier head and BatchNorm after every conv, which is the de-facto
configuration that trains reliably on CIFAR-10.
"""
from __future__ import annotations

import torch
from torch import nn

_CFG = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]


def _make_layers(cfg: list, in_channels: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    c = in_channels
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(2, 2, ceil_mode=True))
        else:
            layers += [
                nn.Conv2d(c, v, kernel_size=3, padding=1),
                nn.BatchNorm2d(v),
                nn.ReLU(inplace=True),
            ]
            c = v
    return nn.Sequential(*layers)


class VGG11(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1, image_size: int = 32) -> None:
        super().__init__()
        self.adapter = nn.Conv2d(in_channels, 3, kernel_size=1) if in_channels != 3 else nn.Identity()
        self.features = _make_layers(_CFG, in_channels=3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)
