#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PYTHON="$ROOT_DIR/envs/mobilemamba_env/bin/python"

echo "Running 01_baseline_vs_fused.py..."
$ENV_PYTHON "$ROOT_DIR/smoke-tests/01_baseline_vs_fused.py"

echo "Running 02_layer_breakdown.py..."
$ENV_PYTHON "$ROOT_DIR/smoke-tests/02_layer_breakdown.py"

echo "Running 03_probe_mbwtconv2d.py..."
$ENV_PYTHON "$ROOT_DIR/smoke-tests/03_probe_mbwtconv2d.py"

echo "Running 04_bypass_wavelet.py..."
$ENV_PYTHON "$ROOT_DIR/smoke-tests/04_bypass_wavelet.py"

echo "Running 05_stage1_topology.py..."
$ENV_PYTHON "$ROOT_DIR/smoke-tests/05_stage1_topology.py"

echo "Running 06_s6_depth_exploration.py..."
$ENV_PYTHON "$ROOT_DIR/smoke-tests/06_s6_depth_exploration.py"

echo "All diagnostic tests complete."