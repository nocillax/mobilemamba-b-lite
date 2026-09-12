#!/usr/bin/env bash
set -e

VARIANT=${1:-"b4"}

# Validate that only allowed variants are passed
case "$VARIANT" in
  b1|b2|b4)
    ;;
  *)
    echo "Error: Invalid variant '$VARIANT'. Supported variants are: b1, b2, b4"
    exit 1
    ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REPO_DIR="$ROOT_DIR/MobileMamba"
ENV_DIR="$ROOT_DIR/envs/mobilemamba_env"
CUDA_DIR="$ROOT_DIR/cuda/cuda-11.8"
RESULTS_DIR="$ROOT_DIR/results"

mkdir -p "$RESULTS_DIR"

cd "$REPO_DIR"

LD_LIBRARY_PATH=/usr/lib64-nvidia:$CUDA_DIR/lib64:$LD_LIBRARY_PATH \
CUDA_HOME=$CUDA_DIR \
PATH=$CUDA_DIR/bin:$PATH \
PYTHONPATH=$REPO_DIR/model/lib_mamba/kernels/selective_scan:$REPO_DIR:$PYTHONPATH \
"$ENV_DIR/bin/python" "$ROOT_DIR/evaluation/benchmark_baseline.py" \
  --variant "$VARIANT" \
  --checkpoint "$REPO_DIR/weights/MobileMamba_${VARIANT^^}/mobilemamba_${VARIANT}.pth" \
  --save-path "$RESULTS_DIR/${VARIANT^^}_ImageNet_baseline_PROPER.json"