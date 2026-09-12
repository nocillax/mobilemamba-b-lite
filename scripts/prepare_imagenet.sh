#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/data/imagenet"

mkdir -p "$DATA_DIR"
wget -c "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar" -P "$DATA_DIR/"

cd "$DATA_DIR"
mkdir -p val
tar -xf ILSVRC2012_img_val.tar -C val/
rm ILSVRC2012_img_val.tar
echo "Validation archive extracted."

cd "$DATA_DIR/val"
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
echo "ImageNet validation set ready at $DATA_DIR/val"