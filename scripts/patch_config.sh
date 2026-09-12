#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="$ROOT_DIR/MobileMamba/configs/mobilemamba/mobilemamba_b4.py"
DATA_DIR="$ROOT_DIR/data/imagenet"

cd "$ROOT_DIR/MobileMamba"

# Change ImageNet loader to standard ImageFolder
sed -i "s/data.type = 'ImageFolderLMDB'/data.type = 'DefaultCLS'/" "$CONFIG_FILE"

# Point B4 config to our prepared ImageNet validation dataset
sed -i "s|data.root = 'data/imagenet'|data.root = '$DATA_DIR'|" "$CONFIG_FILE"

echo "B4 ImageNet config patched successfully."