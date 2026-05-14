"""Write report/tables/results.tex from results/metrics.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "results" / "metrics.json"
OUT = ROOT / "report" / "tables" / "results.tex"

PRETTY = {
    "mlp": "MLP",
    "simple_cnn": "SimpleCNN",
    "lenet": "LeNet-5",
    "alexnet": "AlexNet\\textsuperscript{*}",
    "vgg11": "VGG-11\\textsuperscript{*}",
    "resnet18": "ResNet-18\\textsuperscript{*}",
    "se_resnet_lite": "SE-ResNet-Lite (ours)",
}

ORDER = ["mlp", "simple_cnn", "lenet", "alexnet", "vgg11", "resnet18", "se_resnet_lite"]


def fmt_pct(x: float) -> str:
    if x != x:  # NaN
        return "--"
    return f"{100 * x:.2f}"


def fmt_int(x: int) -> str:
    return f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.1f}k"


def main() -> None:
    with open(METRICS) as f:
        m = json.load(f)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Test-set results for each (architecture, dataset) pair. Accuracy, "
                 r"macro-averaged Precision/Recall/F1, and macro one-vs-rest AUROC are reported, "
                 r"plus trainable parameter count. Models marked with $^{*}$ are adapted for "
                 r"32$\times$32 inputs.}")
    lines.append(r"\label{tab:main}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Model & Params & Train acc & Val acc & "
                 r"Test acc & Precision & Recall & F1 (AUROC) \\")
    lines.append(r"\midrule")

    for ds in ("mnist", "cifar10"):
        if ds not in m:
            continue
        ds_label = {"mnist": "MNIST", "cifar10": "CIFAR-10"}[ds]
        rows = []
        for mname in ORDER:
            if mname not in m[ds]:
                continue
            row = m[ds][mname]
            rows.append((mname, row))
        for i, (mname, row) in enumerate(rows):
            ds_cell = ds_label if i == 0 else ""
            line = (
                f"{ds_cell} & {PRETTY[mname]} & {fmt_int(row['params'])} & "
                f"{fmt_pct(row['train_acc'])} & {fmt_pct(row['val_acc'])} & "
                f"\\textbf{{{fmt_pct(row['accuracy'])}}} & "
                f"{fmt_pct(row['precision_macro'])} & "
                f"{fmt_pct(row['recall_macro'])} & "
                f"{fmt_pct(row['f1_macro'])} "
                f"({fmt_pct(row['auroc_macro_ovr'])}) \\\\"
            )
            lines.append(line)
        lines.append(r"\midrule")

    # Drop the trailing midrule, add bottomrule instead
    if lines[-1] == r"\midrule":
        lines.pop()
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"[tables] wrote {OUT}")


if __name__ == "__main__":
    main()
