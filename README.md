# MobileMamba-B-Lite: Zero-Shot Structural Depth Reduction

Official replication suite and benchmarks for **MobileMamba-B-Lite**, as presented in _"Zero-Shot Structural Depth Reduction of MobileMamba-B for Efficient Image Classification"_.

MobileMamba-B-Lite simplifies pretrained MobileMamba-B models without retraining or fine-tuning. By replacing the second block of Stage 1 (`blocks1[1]`) with an identity mapping, the network depth transitions from $D=(2,3,2)$ to $D=(1,3,2)$. This reduces forward latency by up to **18.4%** on high-resolution inputs with minimal accuracy loss.

---

## Key Results

### 1. Zero-Shot Candidate Screening (2,048 ImageNet-1K Samples on B4)

Screening across candidate modifications led directly to selecting Stage-1 block removal:

| Configuration                    | Depth $D$       | Top-1 (%) | ΔTop-1 (pp) | ΔLatency (%) | Decision   |
| -------------------------------- | --------------- | --------- | ------------------ | ------------------- | ---------- |
| Baseline                         | $(2, 3, 2)$     | 88.96     | 0.00               | —                   | —          |
| Conv–BatchNorm fusion            | $(2, 3, 2)$     | 88.96     | 0.00               | +1.37               | Reject     |
| Stage-2 wavelet bypass           | $(2, 3, 2)$     | 88.77     | -0.20              | +0.40               | Reject     |
| Stage-1 wavelet bypass           | $(2, 3, 2)$     | 88.72     | -0.24              | +0.19               | Reject     |
| Mixer-only removal               | $(2, 3, 2)$     | 87.01     | -1.95              | -9.70               | Reject     |
| Stage-2 block removal            | $(2, 2, 2)$     | 87.26     | -1.71              | -6.51               | Reject     |
| **Stage-1 block removal (Lite)** | **$(1, 3, 2)$** | **88.28** | **-0.68**          | **-16.54**          | **Select** |
| Aggressive reduction             | $(1, 2, 2)$     | 86.08     | -2.88              | -23.52              | Reject     |

### 2. Full 50,000-Image ImageNet-1K Benchmark

Evaluated on an NVIDIA Tesla T4 GPU (FP32, Batch Size 32):

| Model       | Resolution     | Stage Depth | Params (M) | FLOPs (G) | Top-1 (%) | Top-5 (%) | Latency (ms) | Speedup / ΔLatency      | Throughput  |
| ----------- | -------------- | ----------- | ---------- | --------- | --------- | --------- | ------------ | -------------------------- | ----------- |
| **B1**      | $256\times256$ | $(2, 3, 2)$ | 17.12      | 1.07      | 79.94     | 94.93     | 41.36        | —                          | 773.7 img/s |
| **B1-Lite** | $256\times256$ | $(1, 3, 2)$ | 16.54      | 0.93      | 77.96     | 93.79     | 38.54        | **-6.82%** (1.07x)  | 830.4 img/s |
| **B2**      | $384\times384$ | $(2, 3, 2)$ | 17.12      | 2.41      | 81.63     | 95.89     | 113.31       | —                          | 282.4 img/s |
| **B2-Lite** | $384\times384$ | $(1, 3, 2)$ | 16.54      | 2.10      | 80.25     | 95.19     | 93.49        | **-17.49%** (1.21x) | 342.3 img/s |
| **B4**      | $512\times512$ | $(2, 3, 2)$ | 17.12      | 4.29      | 82.50     | 96.26     | 195.29       | —                          | 163.9 img/s |
| **B4-Lite** | $512\times512$ | $(1, 3, 2)$ | 16.54      | 3.73      | 81.03     | 95.44     | 159.37       | **-18.40%** (1.23x) | 200.8 img/s |

---

## Quick Start: Google Colab (Recommended)

No local setup or CUDA installation is required:

1. Upload `full_experiment.ipynb` directly to [Google Colab](https://colab.research.google.com/).
2. Set the runtime: **Runtime** $\to$ **Change runtime type** $\to$ **T4 GPU**.
3. Run the notebook cells in order.

The notebook bootstraps the environment, builds the required selective scan kernels, downloads the ImageNet validation split, and executes all tests end-to-end.

---

## Local / Linux GPU Setup & Execution

### 1. Repository Layout

```text
.
├── full_experiment.ipynb       # Standalone Colab notebook
├── setup_all.sh                # Environment and dataset setup orchestrator
├── run_all_smoke-tests.sh      # Candidate screening orchestrator
├── data/                       # ImageNet validation set directory
├── evaluation/                 # Full 50k benchmark scripts
│   ├── benchmark_baseline.py
│   └── benchmark_path_a.py
├── results/                    # Output directory for evaluation metrics (.json)
├── scripts/                    # Bootstrap and evaluation helper scripts
│   ├── setup_env.sh
│   ├── install_cuda.py
│   ├── install_kernels.py
│   ├── prepare_imagenet.sh
│   ├── patch_config.sh
│   ├── eval_baseline.sh
│   └── eval_path_a.sh
└── smoke-tests/                # Fast screening tests (2,048-image subset)
    ├── 01_baseline_vs_fused.py
    ├── 02_layer_breakdown.py
    ├── 03_probe_mbwtconv2d.py
    ├── 04_bypass_wavelet.py
    ├── 05_stage1_topology.py
    └── 06_s6_depth_exploration.py

```

### 2. Setup

Run the single setup script to install dependencies, set up CUDA 11.8, compile the kernels, and download the ImageNet validation set:

```bash
chmod +x setup_all.sh scripts/*.sh run_all_smoke-tests.sh
./setup_all.sh

```

### 3. Run Diagnostic Smoke Tests (2k Subset)

To verify kernel builds and replicate the ablation screening metrics from Table II:

```bash
./run_all_smoke-tests.sh

```

You can also run any specific ablation individually:

```bash
./envs/mobilemamba_env/bin/python smoke-tests/05_stage1_topology.py

```

### 4. Run Full 50k ImageNet-1K Evaluations

Run full 50,000-sample evaluations for variants `b1`, `b2`, or `b4` (defaults to `b4`):

#### Unmodified Baseline ($D=(2,3,2)$)

```bash
bash scripts/eval_baseline.sh b1
bash scripts/eval_baseline.sh b2
bash scripts/eval_baseline.sh b4

```

#### MobileMamba-B-Lite ($D=(1,3,2)$)

```bash
bash scripts/eval_path_a.sh b1
bash scripts/eval_path_a.sh b2
bash scripts/eval_path_a.sh b4

```

Generated metric summaries and timing logs will be saved under `results/`.

---

## Author & Citation

**Author:** [Md. Asif Chowdhury]  
* Department of Computer Science and Engineering, American International University-Bangladesh (AIUB)  
* Email: [asifjarif@gmail.com]  
* GitHub: [@nocillax](https://github.com/nocillax)  
* LinkedIn: [md-asif-chowdhury-xarif](https://linkedin.com/in/md-asif-chowdhury-xarif)

Feel free to reach out if you have any questions.
