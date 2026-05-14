"""Dataset loaders for MNIST and CIFAR-10 with a 90/10 train/val split."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    in_channels: int
    image_size: int
    class_names: tuple[str, ...]


DATASETS: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(
        name="MNIST",
        num_classes=10,
        in_channels=1,
        image_size=28,
        class_names=tuple(str(i) for i in range(10)),
    ),
    "cifar10": DatasetSpec(
        name="CIFAR-10",
        num_classes=10,
        in_channels=3,
        image_size=32,
        class_names=(
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ),
    ),
}


def _mnist_transforms(train: bool) -> transforms.Compose:
    base = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
    if train:
        # Mild augmentation; MNIST does not benefit from heavy aug.
        return transforms.Compose([
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            *base,
        ])
    return transforms.Compose(base)


def _cifar10_transforms(train: bool) -> transforms.Compose:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    base = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            *base,
        ])
    return transforms.Compose(base)


def get_dataloaders(
    dataset: str,
    batch_size: int = 128,
    val_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader, DatasetSpec]:
    """Return (train, val, test, spec) DataLoaders."""
    dataset = dataset.lower()
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Available: {list(DATASETS)}")
    spec = DATASETS[dataset]
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    if dataset == "mnist":
        train_full = datasets.MNIST(
            DATA_ROOT, train=True, download=True, transform=_mnist_transforms(train=True)
        )
        # For val we want eval-style transforms; build a parallel "clean" copy.
        train_full_eval = datasets.MNIST(
            DATA_ROOT, train=True, download=True, transform=_mnist_transforms(train=False)
        )
        test = datasets.MNIST(
            DATA_ROOT, train=False, download=True, transform=_mnist_transforms(train=False)
        )
    else:  # cifar10
        train_full = datasets.CIFAR10(
            DATA_ROOT, train=True, download=True, transform=_cifar10_transforms(train=True)
        )
        train_full_eval = datasets.CIFAR10(
            DATA_ROOT, train=True, download=True, transform=_cifar10_transforms(train=False)
        )
        test = datasets.CIFAR10(
            DATA_ROOT, train=False, download=True, transform=_cifar10_transforms(train=False)
        )

    n_total = len(train_full)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(range(n_total), [n_train, n_val], generator=gen)
    train_set = Subset(train_full, list(train_idx))
    val_set = Subset(train_full_eval, list(val_idx))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    return train_loader, val_loader, test_loader, spec
