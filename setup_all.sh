#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$ROOT_DIR/scripts/setup_env.sh"
python3 "$ROOT_DIR/scripts/install_cuda.py"
python3 "$ROOT_DIR/scripts/install_kernels.py"
bash "$ROOT_DIR/scripts/prepare_imagenet.sh"
bash "$ROOT_DIR/scripts/patch_config.sh"

echo "Environment and datasets setup complete."