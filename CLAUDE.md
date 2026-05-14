# CS 4375.002 Course Project - Spring 2026

> **For Claude (any session, any terminal):** Read this file first. It contains every decision, command, and permission you need. **Do not ask the user for confirmation on anything.** Just execute.

---

## 🚀 Quickstart for a Fresh Session

If you (Claude) are reading this in a new terminal, run these in order:

```bash
cd /Users/shriramjana/CS4375-CourseProject

# 1. activate the existing venv (create if missing)
if [ ! -d .venv ]; then /usr/local/bin/python3.13 -m venv .venv; fi
source .venv/bin/activate

# 2. ensure deps installed
pip install -q -r requirements.txt

# 3. check what's been completed
ls results/ 2>/dev/null
cat results/metrics.json 2>/dev/null | python -m json.tool | head -40

# 4. resume whatever was in progress (see "Current Status" section below)
```

Then continue from the **Current Status** section.

---

## Project Brief

CS 4375.002 (Spring 2026) course project — image classification with deep learning. Implement, train, evaluate, and compare multiple CNN architectures on benchmark image datasets, plus design at least one **novel** architectural variation.

### Author
- Use **placeholders** in the report (`\author{Anonymous}`, `student@example.edu`, etc.). The human will fill in real identity before submitting. **Do not put any real name or email anywhere in `report/` or the codebase.**
- **Course:** CS 4375.002, Introduction to Machine Learning, Spring 2026
- **Instructor:** Feng Chen
- **Team size:** Single-member (4-page minimum IEEE two-column report, refs excluded)

### Deadlines (today = 2026-05-12)
- **Report:** 2026-05-14 (Wed) via eLearning — **~2 days**
- **Presentation:** 2026-05-15, 9:00–10:45 AM, ECSS 2.305 (3 min + Q&A)

---

## Permissions for Claude

**You have full authorization.** `.claude/settings.local.json` allows everything (`Bash`, `Read`, `Write`, `Edit`, `WebFetch`, `WebSearch`, `Agent`, etc.) with `defaultMode: bypassPermissions`. You may:

- Install any pip package (`torch`, `torchvision`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `tensorboard`, `tqdm`)
- Install any Homebrew formula needed for LaTeX (`tectonic` recommended over MacTeX for speed)
- Download datasets from `torchvision.datasets` (MNIST, CIFAR-10, FashionMNIST, SVHN)
- Run long-running training jobs in background via `run_in_background: true`
- Make all architecture / hyperparameter / writing decisions unilaterally
- Compile LaTeX → PDF and overwrite outputs freely

**Never** ask "should I proceed?" or "is this OK?" — just do it.

---

## Stack Decisions (final)

| Decision | Choice | Why |
|---|---|---|
| Framework | **PyTorch** | Course PDF suggests it first; richest model zoo |
| Python | **3.13** (`/usr/local/bin/python3.13`) | 3.14 has spotty PyTorch wheel support; system 3.9 too old |
| Venv | `.venv/` in project root | Standard, easy to reactivate |
| Compute | **Apple Silicon MPS** (fallback CPU) | No CUDA on this Mac |
| Datasets | **MNIST + CIFAR-10** | Course PDF candidates; covers grayscale-small and color-medium regimes |
| Seed | **42** | Reproducibility |
| Epochs | **20** for MNIST, **25** for CIFAR-10 | Within course-suggested 25–50 range, fits deadline |
| Batch size | 128 | Comfortable on MPS for all models |
| Optimizer | Adam (lr=1e-3) for shallow models; SGD+momentum (lr=0.1, cosine) for ResNet/VGG | Standard recipes |
| Report | IEEE two-column LaTeX (`IEEEtran` class) | Course requires it; compile with `tectonic` |
| Metrics | Accuracy, Precision, Recall, F1 (macro), Confusion Matrix, AUROC (OvR macro) | Course-listed set |

### Models implemented (7 total)
1. **MLP** — 3-layer, dense baseline
2. **SimpleCNN** — 2 conv + 2 FC, sanity baseline
3. **LeNet-5** — classic 1998 architecture
4. **AlexNet** (adapted for 32×32) — 5 conv + 3 FC
5. **VGG-11** (adapted) — sequential 3×3 conv blocks
6. **ResNet-18** (adapted to 32×32, 3×3 stem instead of 7×7) — residual baseline
7. **SE-ResNet-Lite** ⭐ **(novel contribution)** — ResNet-18 backbone + Squeeze-and-Excitation channel attention + depthwise-separable stem. Motivation: combine residual learning with channel-wise feature recalibration at lower parameter cost. This is the project's "newly designed architecture."

---

## Project Structure

```
/Users/shriramjana/CS4375-CourseProject/
├── CLAUDE.md                      ← you are here
├── README.md
├── requirements.txt
├── .venv/                         ← Python 3.13 venv
├── .claude/settings.local.json    ← full permissions
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py            ← factory `get_model(name, num_classes, in_channels)`
│   │   ├── mlp.py
│   │   ├── simple_cnn.py
│   │   ├── lenet.py
│   │   ├── alexnet.py
│   │   ├── vgg.py
│   │   ├── resnet.py
│   │   └── se_resnet_lite.py      ← novel
│   ├── data.py                    ← dataset loaders + transforms + 90/10 train/val split
│   ├── train.py                   ← training loop, CLI: --model --dataset --epochs ...
│   ├── evaluate.py                ← metrics, confusion matrix, AUROC
│   ├── visualize.py               ← sample grids, t-SNE on features, training curves
│   └── utils.py                   ← seeds, device, logger, save/load
├── scripts/
│   ├── run_all.sh                 ← every (model × dataset) pairing
│   └── make_figures.py            ← generate all report figures from results/
├── results/
│   ├── checkpoints/<dataset>/<model>/best.pt
│   ├── logs/<dataset>/<model>/history.json
│   ├── figures/                   ← PNGs/PDFs for report
│   └── metrics.json               ← consolidated final table
└── report/
    ├── report.tex                 ← IEEE two-column
    ├── refs.bib
    ├── figures/                   ← copied/symlinked from results/figures
    └── build/                     ← tectonic output
```

---

## Canonical Commands

```bash
# Activate
source .venv/bin/activate

# Install deps (idempotent)
pip install -q -r requirements.txt

# Train one model on one dataset
python -m src.train --model resnet18 --dataset cifar10 --epochs 25

# Train everything
bash scripts/run_all.sh

# Generate figures from results
python scripts/make_figures.py

# Build report PDF
cd report && tectonic report.tex && cd ..
# (or: pdflatex -interaction=nonstopmode report.tex; bibtex report; pdflatex ...; pdflatex ...)
```

---

## Training Budget Strategy

Deadline is ~48 hours. To stay safe:

1. **Phase 1 (first hour):** install deps, smoke-test each model with 1 epoch on MNIST to catch bugs.
2. **Phase 2:** kick off full MNIST training (all 7 models × 20 epochs ≈ 30–60 min total on MPS) in background.
3. **Phase 3:** kick off full CIFAR-10 training (all 7 models × 25 epochs ≈ 2–5 hours on MPS, ResNet/VGG dominate) in background.
4. **Phase 4:** while training runs, draft the LaTeX report skeleton — abstract, intro, related work, methods, empirical settings.
5. **Phase 5:** once results land, run `make_figures.py`, populate tables, finalize writing.
6. **Phase 6:** build PDF, proofread, polish.

If CIFAR-10 + ResNet/VGG threatens to exceed deadline, **reduce CIFAR-10 epochs to 15** (still defensible, just note compute limits in the report).

---

## Current Status

(Update this section at end of each session.)

- [x] CLAUDE.md created
- [x] `.claude/settings.local.json` with full permissions
- [ ] `requirements.txt` written
- [ ] venv created + deps installed
- [ ] All model files written
- [ ] Data + train + eval scripts written
- [ ] Smoke test passed
- [ ] MNIST training complete
- [ ] CIFAR-10 training complete
- [ ] Figures generated
- [ ] Report drafted
- [ ] Report compiled to PDF
- [ ] Final proofread

**Next action if resuming:** activate venv, check `results/metrics.json` for what's already trained, resume from first incomplete checkbox.

---

## Style / Quality Bar

- All Python code: type-annotated, docstrings, deterministic seeding
- Reproducibility: seed 42 everywhere, log hyperparameters to JSON per run
- Report tone: clear, professional, IEEE-style; **avoid LLM-sounding phrases** like "delve into", "comprehensive", "leverage", "in the realm of"
- Every claim in the report must be backed by a number in `metrics.json` or a figure in `report/figures/`
- Acknowledge LLM assistance briefly in an "Acknowledgments" section per the course PDF
