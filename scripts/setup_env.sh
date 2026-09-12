#!/usr/bin/env bash
set -e

# Resolve repo root directory dynamically
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1. Clone repository into project root
cd "$ROOT_DIR"
rm -rf MobileMamba
git clone https://github.com/lewandofskee/MobileMamba.git

# 2. Setup isolated Python 3.10 environment inside project
mkdir -p "$ROOT_DIR/bin" "$ROOT_DIR/envs"
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "$ROOT_DIR" bin/micromamba
"$ROOT_DIR/bin/micromamba" create -y -p "$ROOT_DIR/envs/mobilemamba_env" python=3.10 -c conda-forge

ENV_PIP="$ROOT_DIR/envs/mobilemamba_env/bin/pip"

# 3. PyTorch 2.1.2 + CUDA 11.8 wheels
$ENV_PIP install \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# 4. Pin numpy < 2.0
$ENV_PIP install "numpy<2"

# 5. Model dependencies and utilities
$ENV_PIP install \
    timm==0.9.16 \
    tensorboardX \
    einops \
    torchprofile \
    fvcore==0.1.5.post20221221 \
    triton==2.1.0 \
    lmdb \
    PyWavelets \
    scikit-image \
    six \
    "opencv-python==4.8.1.78" \
    terminaltables \
    pycocotools \
    prettytable \
    xtcocotools \
    mmpretrain==1.2.0 \
    mmdet==3.3.0 \
    mmsegmentation==1.2.2 \
    ftfy \
    regex \
    "setuptools<81" \
    mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

# 6. Libcuda fix (if running with sudo permissions or Colab)
if [ -d "/usr/lib64-nvidia" ]; then
    echo "/usr/lib64-nvidia" | sudo tee /etc/ld.so.conf.d/nvidia.conf > /dev/null || true
    sudo ldconfig || true
fi