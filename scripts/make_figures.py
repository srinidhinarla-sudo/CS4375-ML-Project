"""Read every results/logs/<ds>/<model>/history.json and generate report figures.

Outputs go to results/figures/ and are then copied to report/figures/.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import DATASETS, get_dataloaders  # noqa: E402
from src.models import PRETTY_NAMES, get_model  # noqa: E402
from src.utils import get_device, load_checkpoint  # noqa: E402
from src.visualize import (  # noqa: E402
    extract_features,
    plot_confusion_matrix,
    plot_dataset_comparison_bar,
    plot_params_vs_acc,
    plot_pca_lda_tsne_triptych,
    plot_sample_grid,
    plot_training_curves,
    plot_tsne,
)

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "results" / "logs"
CKPT_DIR = ROOT / "results" / "checkpoints"
FIG_DIR = ROOT / "results" / "figures"
REPORT_FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)


def collect_results() -> dict[str, dict[str, dict]]:
    """{dataset: {model: history_json}}"""
    out: dict[str, dict[str, dict]] = {}
    for ds_dir in sorted(LOG_DIR.iterdir()):
        if not ds_dir.is_dir():
            continue
        ds = ds_dir.name
        out[ds] = {}
        for m_dir in sorted(ds_dir.iterdir()):
            hist = m_dir / "history.json"
            if hist.exists():
                with open(hist) as f:
                    out[ds][m_dir.name] = json.load(f)
    return out


def main() -> None:
    results = collect_results()
    device = get_device()
    metrics_summary: dict = {}

    # 1. sample grids per dataset
    for ds in results:
        spec = DATASETS[ds]
        _, _, te, _ = get_dataloaders(ds, batch_size=128, num_workers=0)
        plot_sample_grid(te, list(spec.class_names),
                         f"{spec.name} test samples",
                         FIG_DIR / f"samples_{ds}.png", n=16)

    # 1b. PCA / LDA / t-SNE triptych on RAW pixels per dataset (slide-2 spec)
    for ds in results:
        spec = DATASETS[ds]
        _, _, te, _ = get_dataloaders(ds, batch_size=512, num_workers=0)
        xs: list = []
        ys: list = []
        seen = 0
        for x, y in te:
            xs.append(x.flatten(1).numpy())
            ys.append(y.numpy())
            seen += y.size(0)
            if seen >= 2000:
                break
        import numpy as _np
        X = _np.concatenate(xs)[:2000]
        Y = _np.concatenate(ys)[:2000]
        plot_pca_lda_tsne_triptych(
            X, Y, list(spec.class_names),
            FIG_DIR / f"viz_raw_{ds}.png",
        )

    # 2. per-(dataset, model) figures: training curves, confusion matrix, t-SNE
    for ds, by_model in results.items():
        spec = DATASETS[ds]
        # Build one shared test loader for t-SNE
        _, _, te_loader, _ = get_dataloaders(ds, batch_size=256, num_workers=0)

        metrics_summary[ds] = {}
        for mname, hist in by_model.items():
            pretty = PRETTY_NAMES.get(mname, mname)
            # training curves
            plot_training_curves(
                hist["history"], f"{pretty} on {spec.name}",
                FIG_DIR / f"curves_{ds}_{mname}.png",
            )
            # confusion matrix on test
            plot_confusion_matrix(
                hist["test_metrics"]["confusion_matrix"], list(spec.class_names),
                f"{pretty} on {spec.name} (test)",
                FIG_DIR / f"cm_{ds}_{mname}.png",
                normalize=True,
            )
            # collect summary metrics
            tm = hist["test_metrics"]
            metrics_summary[ds][mname] = {
                "params": hist["n_params"],
                "best_epoch": hist["best_epoch"],
                "train_acc": hist["train_metrics"]["accuracy"],
                "val_acc": hist["val_metrics"]["accuracy"],
                "accuracy": tm["accuracy"],
                "precision_macro": tm["precision_macro"],
                "recall_macro": tm["recall_macro"],
                "f1_macro": tm["f1_macro"],
                "auroc_macro_ovr": tm["auroc_macro_ovr"],
                "total_train_time_s": hist["total_train_time_s"],
            }

            # t-SNE + full PCA/LDA/t-SNE triptych on penultimate features
            # of our novel model and the ResNet-18 baseline.
            if mname in ("se_resnet_lite", "resnet18"):
                ckpt_path = CKPT_DIR / ds / mname / "best.pt"
                if ckpt_path.exists():
                    model = get_model(mname, num_classes=spec.num_classes,
                                      in_channels=spec.in_channels,
                                      image_size=spec.image_size).to(device)
                    state = load_checkpoint(ckpt_path, map_location=device)
                    model.load_state_dict(state["state_dict"])
                    feats, labs = extract_features(model, te_loader, device, max_samples=1500)
                    plot_tsne(feats, labs, list(spec.class_names),
                              f"{pretty}: t-SNE of penultimate features ({spec.name})",
                              FIG_DIR / f"tsne_{ds}_{mname}.png")
                    if mname == "se_resnet_lite":
                        plot_pca_lda_tsne_triptych(
                            feats, labs, list(spec.class_names),
                            FIG_DIR / f"viz_feat_{ds}.png",
                        )

        # comparison bar chart per dataset
        if metrics_summary[ds]:
            plot_dataset_comparison_bar(
                {PRETTY_NAMES.get(k, k): v for k, v in metrics_summary[ds].items()},
                ds, FIG_DIR / f"compare_{ds}.png",
            )

    # 3. cross-dataset parameter-count vs accuracy scatter
    if metrics_summary:
        plot_params_vs_acc(metrics_summary, FIG_DIR / "params_vs_acc.png")

    # 4. write consolidated metrics.json
    with open(ROOT / "results" / "metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # 4. copy figures into report/
    for fp in FIG_DIR.glob("*.png"):
        shutil.copy(fp, REPORT_FIG_DIR / fp.name)
    print(f"[figures] wrote {len(list(FIG_DIR.glob('*.png')))} figures to {FIG_DIR}")
    print(f"[figures] copied to {REPORT_FIG_DIR}")


if __name__ == "__main__":
    main()
