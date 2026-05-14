"""Train a single (model, dataset) pair.

Usage:
    python -m src.train --model resnet18 --dataset cifar10 --epochs 25
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data import get_dataloaders
from .evaluate import evaluate
from .models import MODEL_NAMES, get_model
from .utils import (
    count_parameters,
    ensure_dir,
    get_device,
    save_checkpoint,
    save_json,
    set_seed,
)

ROOT = Path(__file__).resolve().parent.parent


# Models with BatchNorm tolerate high-LR SGD (canonical CIFAR recipe).
# AlexNet has no BN, so SGD lr=0.1 explodes its activations -- use Adam.
_SGD_MODELS = {"vgg11", "resnet18", "se_resnet_lite"}


def make_optimizer(model: nn.Module, model_name: str, base_lr: float | None = None):
    if model_name in _SGD_MODELS:
        lr = base_lr if base_lr is not None else 0.1
        return SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    lr = base_lr if base_lr is not None else 1e-3
    return Adam(model.parameters(), lr=lr)


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return loss_sum / n, correct / n


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = get_device()
    print(f"[train] device={device}  model={args.model}  dataset={args.dataset}  epochs={args.epochs}")

    train_loader, val_loader, test_loader, spec = get_dataloaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
        val_fraction=0.1,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    model = get_model(
        args.model,
        num_classes=spec.num_classes,
        in_channels=spec.in_channels,
        image_size=spec.image_size,
    ).to(device)
    n_params = count_parameters(model)
    print(f"[train] params={n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, args.model, base_lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    log_dir = ensure_dir(ROOT / "results" / "logs" / args.dataset / args.model)
    ckpt_dir = ensure_dir(ROOT / "results" / "checkpoints" / args.dataset / args.model)

    history: list[dict] = []
    best_val_acc = -1.0
    best_epoch = -1
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        e0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val = evaluate(model, val_loader, device, spec.num_classes)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - e0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val["loss"],
            "val_acc": val["accuracy"],
            "val_f1": val["f1_macro"],
            "lr": lr_now,
            "epoch_time_s": epoch_time,
        }
        history.append(row)
        print(
            f"  epoch {epoch:02d}/{args.epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val['loss']:.4f} val_acc={val['accuracy']:.4f}  "
            f"lr={lr_now:.4f}  t={epoch_time:.1f}s",
            flush=True,
        )

        if val["accuracy"] > best_val_acc:
            best_val_acc = val["accuracy"]
            best_epoch = epoch
            save_checkpoint(
                {
                    "model_name": args.model,
                    "dataset": args.dataset,
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_metrics": val,
                    "params": n_params,
                },
                ckpt_dir / "best.pt",
            )

    total_time = time.time() - t0
    # Final test evaluation using best checkpoint
    state = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    test_metrics = evaluate(model, test_loader, device, spec.num_classes)
    train_metrics = evaluate(model, train_loader, device, spec.num_classes)
    print(f"[train] BEST val_acc={best_val_acc:.4f} at epoch {best_epoch}")
    print(f"[train] TEST  acc={test_metrics['accuracy']:.4f}  f1={test_metrics['f1_macro']:.4f}  "
          f"auroc={test_metrics['auroc_macro_ovr']:.4f}")

    result = {
        "model": args.model,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "total_train_time_s": total_time,
        "history": history,
        "train_metrics": train_metrics,
        "val_metrics": state["val_metrics"],
        "test_metrics": test_metrics,
        "class_names": list(spec.class_names),
    }
    save_json(result, log_dir / "history.json")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_NAMES))
    p.add_argument("--dataset", required=True, choices=["mnist", "cifar10"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
