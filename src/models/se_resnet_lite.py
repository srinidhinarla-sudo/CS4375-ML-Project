"""SE-ResNet-Lite: our novel architecture for this project.

Design rationale
----------------
We take ResNet-18 as a strong baseline for small-image classification and
make two targeted modifications motivated by the literature:

1. **Depthwise-separable stem.** The conventional CIFAR ResNet stem is a
   single 3x3 conv (3 -> 64 channels). We replace it with a *depthwise-
   separable* stem (depthwise 3x3 over the input channels, then pointwise
   1x1 expanding to 64 channels) following MobileNet (Howard et al., 2017).
   This reduces parameters and FLOPs in the stem while providing the same
   receptive field.

2. **Squeeze-and-Excitation (SE) blocks** inside every residual block,
   following Hu et al. ("Squeeze-and-Excitation Networks", CVPR 2018). The
   SE module learns a per-channel gating scalar from global average-pooled
   features and modulates the residual output, allowing the network to
   recalibrate channel responses based on the input.

The combination is intended to give ResNet-18 stronger channel-wise
representation power (via SE) while *reducing* parameter count at the
input stage (via depthwise-separable convolution). We refer to this
model as **SE-ResNet-Lite** throughout the report.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SqueezeExcite(nn.Module):
    """Channel-wise attention via global pooling + 2-layer MLP."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)
        s = self.fc(s).view(b, c, 1, 1)
        return x * s


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, reduction: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SqueezeExcite(planes, reduction=reduction)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # SE applied on the residual branch
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class DepthwiseSeparableStem(nn.Module):
    """3x3 depthwise + 1x1 pointwise stem (input_channels -> 64)."""

    def __init__(self, in_channels: int, out_channels: int = 64) -> None:
        super().__init__()
        # Ensure depthwise works: depthwise needs in_channels groups, so we
        # first expand 1->3 if the dataset is grayscale via a tiny adapter
        # so the architecture stays consistent across MNIST / CIFAR-10.
        self.adapter = (
            nn.Conv2d(in_channels, 3, kernel_size=1, bias=False) if in_channels != 3
            else nn.Identity()
        )
        self.depthwise = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3, bias=False)
        self.pointwise = nn.Conv2d(3, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return F.relu(self.bn(x), inplace=True)


class SEResNetLite(nn.Module):
    """ResNet-18 backbone with SE blocks and a depthwise-separable stem."""

    def __init__(self, num_classes: int = 10, in_channels: int = 1, image_size: int = 32) -> None:
        super().__init__()
        self.stem = DepthwiseSeparableStem(in_channels=in_channels, out_channels=64)
        self.in_planes = 64
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512 * SEBasicBlock.expansion, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: list[nn.Module] = []
        for s in strides:
            layers.append(SEBasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * SEBasicBlock.expansion
        return nn.Sequential(*layers)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.avgpool(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        return self.fc(self.dropout(f))
