# CS 4375.002 Course Project — CNN Architectures for Image Classification

PyTorch implementation and comparison of 7 neural network architectures
(MLP, SimpleCNN, LeNet-5, AlexNet, VGG-11, ResNet-18, and a novel
**SE-ResNet-Lite**) on MNIST and CIFAR-10.

## Setup

```bash
/usr/local/bin/python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train one model

```bash
python -m src.train --model resnet18 --dataset cifar10 --epochs 25
```

Valid `--model` values: `mlp simple_cnn lenet alexnet vgg11 resnet18 se_resnet_lite`
Valid `--dataset` values: `mnist cifar10`

## Train everything

```bash
bash scripts/run_all.sh                   # full schedule (20 ep MNIST, 25 ep CIFAR-10)
bash scripts/run_all.sh --smoke           # 1-epoch sanity test on MNIST only
EPOCHS_MNIST=15 EPOCHS_CIFAR=20 bash scripts/run_all.sh   # override
```

## Generate report figures + table

```bash
python scripts/make_figures.py        # generates results/figures/*.png + results/metrics.json
python scripts/make_tables.py         # writes report/tables/results.tex
```

## Build PDF

```bash
cd report
tectonic report.tex          # final report (IEEE two-column)
tectonic slides.tex          # 3-minute presentation deck (beamer)
```

Outputs: `report/report.pdf` and `report/slides.pdf`.

## Outputs

- `results/checkpoints/<dataset>/<model>/best.pt` — best-val-acc weights
- `results/logs/<dataset>/<model>/history.json` — per-epoch + final metrics
- `results/metrics.json` — consolidated test-set metrics for all runs
- `results/figures/*.png` — training curves, confusion matrices, t-SNE, sample grids
- `report/report.pdf` — the final IEEE two-column report
