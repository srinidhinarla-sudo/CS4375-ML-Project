#!/usr/bin/env bash
# Train every (model, dataset) pair. Skips runs whose history.json already exists.
#   --smoke   1-epoch sanity run, MNIST only
#   --force   re-run even if results exist
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

MODELS=(mlp simple_cnn lenet alexnet vgg11 resnet18 se_resnet_lite)
DATASETS=(mnist cifar10)
FORCE=0
SMOKE=0
for arg in "$@"; do
    case "$arg" in
        --smoke) SMOKE=1 ;;
        --force) FORCE=1 ;;
    esac
done
if [[ $SMOKE -eq 1 ]]; then
    EPOCHS_MNIST=1
    EPOCHS_CIFAR=1
    DATASETS=(mnist)
else
    EPOCHS_MNIST=${EPOCHS_MNIST:-20}
    EPOCHS_CIFAR=${EPOCHS_CIFAR:-25}
fi

for ds in "${DATASETS[@]}"; do
    if [[ "$ds" == "mnist" ]]; then E=$EPOCHS_MNIST; else E=$EPOCHS_CIFAR; fi
    for m in "${MODELS[@]}"; do
        hist="results/logs/$ds/$m/history.json"
        if [[ -f "$hist" && $FORCE -eq 0 ]]; then
            echo "===== $ds / $m — SKIP (exists at $hist) ====="
            continue
        fi
        echo "===== $ds / $m / epochs=$E ====="
        python -m src.train --model "$m" --dataset "$ds" --epochs "$E"
    done
done
echo "[run_all] DONE"
