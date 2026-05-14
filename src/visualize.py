"""Plotting utilities: training curves, confusion matrices, t-SNE, samples."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

sns.set_context("paper")
sns.set_style("whitegrid")


def plot_training_curves(history: list[dict], title: str, out_path: str | Path) -> None:
    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss = [r["val_loss"] for r in history]
    train_acc = [r["train_acc"] for r in history]
    val_acc = [r["val_acc"] for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(epochs, train_loss, label="train", marker="o", markersize=3)
    axes[0].plot(epochs, val_loss, label="val", marker="o", markersize=3)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend()
    axes[0].set_title(f"{title}: loss")

    axes[1].plot(epochs, train_acc, label="train", marker="o", markersize=3)
    axes[1].plot(epochs, val_acc, label="val", marker="o", markersize=3)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].legend()
    axes[1].set_title(f"{title}: accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm: list[list[int]], class_names: list[str],
                          title: str, out_path: str | Path,
                          normalize: bool = True) -> None:
    arr = np.array(cm, dtype=float)
    if normalize:
        row_sums = arr.sum(axis=1, keepdims=True)
        arr = arr / np.where(row_sums == 0, 1, row_sums)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(arr, annot=True, fmt=".2f" if normalize else "d",
                cmap="Blues", xticklabels=class_names, yticklabels=class_names,
                cbar=False, square=True, annot_kws={"size": 6}, ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sample_grid(loader: DataLoader, class_names: list[str],
                     title: str, out_path: str | Path, n: int = 16) -> None:
    images, labels = next(iter(loader))
    images = images[:n]
    labels = labels[:n]
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(1.5 * cols, 1.6 * rows))
    for i, ax in enumerate(np.array(axes).flat):
        if i >= n:
            ax.axis("off"); continue
        img = images[i].numpy()
        if img.shape[0] == 1:
            ax.imshow(img[0], cmap="gray")
        else:
            # un-normalize CIFAR-10 for display
            mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
            std = np.array([0.2470, 0.2435, 0.2616]).reshape(3, 1, 1)
            disp = (img * std + mean).clip(0, 1).transpose(1, 2, 0)
            ax.imshow(disp)
        ax.set_title(class_names[int(labels[i])], fontsize=8)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def extract_features(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                     max_samples: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Extract penultimate-layer features. Falls back to logits if model has no
    .features() method."""
    model.eval()
    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    seen = 0
    for x, y in loader:
        x = x.to(device)
        if hasattr(model, "features") and callable(model.features):
            try:
                f = model.features(x)
            except TypeError:
                f = model(x)
        else:
            f = model(x)
        if f.ndim > 2:
            f = torch.flatten(f, 1)
        feats.append(f.cpu().numpy())
        labels.append(y.numpy())
        seen += y.size(0)
        if seen >= max_samples:
            break
    return np.concatenate(feats)[:max_samples], np.concatenate(labels)[:max_samples]


def plot_2d_scatter(emb: np.ndarray, labels: np.ndarray, class_names: list[str],
                    title: str, ax) -> None:
    palette = sns.color_palette("tab10", n_colors=len(class_names))
    for cls, name in enumerate(class_names):
        mask = labels == cls
        ax.scatter(emb[mask, 0], emb[mask, 1], s=5, alpha=0.65,
                   color=palette[cls], label=name)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def plot_pca_lda_tsne_triptych(features: np.ndarray, labels: np.ndarray,
                                class_names: list[str], out_path: str | Path,
                                seed: int = 42) -> None:
    """Three side-by-side 2D scatter plots (PCA / LDA / t-SNE).

    Designed for the project's presentation slide-2 layout: three small
    plots placed side by side, each point one image, colored by class.
    """
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    n = features.shape[0]
    pca = PCA(n_components=2, random_state=seed).fit_transform(features)
    try:
        lda = LinearDiscriminantAnalysis(n_components=2).fit_transform(features, labels)
    except Exception:
        lda = pca  # fall back if LDA fails (e.g. degenerate features)
    perplexity = min(30, max(5, n // 50))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                learning_rate="auto", random_state=seed).fit_transform(features)

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    plot_2d_scatter(pca, labels, class_names, "PCA", axes[0])
    plot_2d_scatter(lda, labels, class_names, "LDA", axes[1])
    plot_2d_scatter(tsne, labels, class_names, "t-SNE", axes[2])
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
                    fontsize=7, markerscale=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_tsne(features: np.ndarray, labels: np.ndarray, class_names: list[str],
              title: str, out_path: str | Path, seed: int = 42) -> None:
    n_samples = features.shape[0]
    perplexity = min(30, max(5, n_samples // 50))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                learning_rate="auto", random_state=seed)
    emb = tsne.fit_transform(features)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    palette = sns.color_palette("tab10", n_colors=len(class_names))
    for cls, name in enumerate(class_names):
        mask = labels == cls
        ax.scatter(emb[mask, 0], emb[mask, 1], s=6, alpha=0.7,
                   color=palette[cls], label=name)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, markerscale=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_params_vs_acc(metrics_per_ds: dict, out_path: str | Path) -> None:
    """metrics_per_ds: {dataset: {model: {params, accuracy, ...}}}."""
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    markers = {"mnist": "o", "cifar10": "s"}
    for ds, by_model in metrics_per_ds.items():
        xs = [by_model[m]["params"] / 1e6 for m in by_model]
        ys = [by_model[m]["accuracy"] * 100 for m in by_model]
        names = list(by_model)
        ax.scatter(xs, ys, marker=markers.get(ds, "x"), s=70, alpha=0.85,
                   label={"mnist": "MNIST", "cifar10": "CIFAR-10"}.get(ds, ds))
        for x, y, n in zip(xs, ys, names):
            ax.annotate(n, (x, y), fontsize=7, xytext=(4, 3),
                        textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters (M, log scale)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Parameter count vs. test accuracy")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_comparison_bar(metrics: dict, dataset: str, out_path: str | Path) -> None:
    """metrics: {model_name: {accuracy, f1, ...}}. Bar chart per metric."""
    models = list(metrics)
    acc = [metrics[m]["accuracy"] for m in models]
    f1 = [metrics[m]["f1_macro"] for m in models]
    x = np.arange(len(models))
    w = 0.4
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.bar(x - w/2, acc, w, label="test accuracy")
    ax.bar(x + w/2, f1, w, label="test F1 (macro)")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title(f"{dataset.upper()}: per-model test metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
