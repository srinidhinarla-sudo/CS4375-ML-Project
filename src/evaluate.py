"""Evaluation: accuracy, precision/recall/F1, confusion matrix, AUROC."""
from __future__ import annotations

from typing import Any

import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    criterion: nn.Module | None = None,
) -> dict[str, Any]:
    """Run model on loader, return metrics dict + per-sample logits."""
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    loss_sum = 0.0
    n = 0
    crit_sum = criterion if criterion is not None else nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        all_targets.append(y.clone())  # capture CPU copy before any device round-trip
        x = x.to(device)
        y_dev = y.to(device)
        logits = model(x)
        loss_sum += crit_sum(logits, y_dev).item()
        n += y.size(0)
        all_logits.append(logits.detach().cpu())

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets).numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)

    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, average="macro", zero_division=0)
    rec = recall_score(targets, preds, average="macro", zero_division=0)
    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    try:
        auroc = roc_auc_score(targets, probs, multi_class="ovr", average="macro",
                              labels=list(range(num_classes)))
    except ValueError:
        auroc = float("nan")
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))

    return {
        "loss": loss_sum / max(n, 1),
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "auroc_macro_ovr": float(auroc),
        "confusion_matrix": cm.tolist(),
        "n_samples": int(n),
    }
