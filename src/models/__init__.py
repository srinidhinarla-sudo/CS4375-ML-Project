"""Model factory."""
from __future__ import annotations

from torch import nn

from .alexnet import AlexNetCifar
from .lenet import LeNet5
from .mlp import MLP
from .resnet import ResNet18
from .se_resnet_lite import SEResNetLite
from .simple_cnn import SimpleCNN
from .vgg import VGG11


_BUILDERS = {
    "mlp": MLP,
    "simple_cnn": SimpleCNN,
    "lenet": LeNet5,
    "alexnet": AlexNetCifar,
    "vgg11": VGG11,
    "resnet18": ResNet18,
    "se_resnet_lite": SEResNetLite,
}

MODEL_NAMES = tuple(_BUILDERS)

PRETTY_NAMES = {
    "mlp": "MLP",
    "simple_cnn": "SimpleCNN",
    "lenet": "LeNet-5",
    "alexnet": "AlexNet*",
    "vgg11": "VGG-11*",
    "resnet18": "ResNet-18*",
    "se_resnet_lite": "SE-ResNet-Lite (ours)",
}


def get_model(name: str, num_classes: int, in_channels: int, image_size: int = 32) -> nn.Module:
    name = name.lower()
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model {name!r}. Available: {MODEL_NAMES}")
    return _BUILDERS[name](
        num_classes=num_classes, in_channels=in_channels, image_size=image_size
    )
