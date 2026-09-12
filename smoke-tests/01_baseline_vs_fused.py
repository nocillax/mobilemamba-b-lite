import os
import sys
import copy
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ============================================================================
# DYNAMIC ENVIRONMENT RESOLUTION
# ============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
REPO = str(ROOT_DIR / "MobileMamba")
CHECKPOINT_PATH = str(ROOT_DIR / "MobileMamba" / "weights" / "MobileMamba_B4" / "mobilemamba_b4.pth")
IMAGENET_VAL = str(ROOT_DIR / "data" / "imagenet" / "val")

sys.path.insert(0, REPO)
sys.path.insert(0, str(ROOT_DIR / "MobileMamba" / "model" / "lib_mamba" / "kernels" / "selective_scan"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 90)
print("STEP 1: RAW BASELINE vs. CONV-BN FUSION + CUDNN BENCHMARK")
print("=" * 90)
print("Device     :", device)
print("Checkpoint :", CHECKPOINT_PATH)
print("ImageNet   :", IMAGENET_VAL)

# ============================================================================
# IMPORT MODEL & FUSION UTILITY
# ============================================================================
from model.mobilemamba.mobilemamba import (
    MobileMamba,
    CFG_MobileMamba_B4,
    replace_batchnorm,
)

# ============================================================================
# MODEL LOADER
# ============================================================================
def load_model():
    cfg = copy.deepcopy(CFG_MobileMamba_B4)
    model = MobileMamba(
        **cfg,
        num_classes=1000,
        distillation=False,
        forward_type="v052d",
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    state_dict = {
        (k.replace("module.", "", 1) if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return model

# ============================================================================
# SMOKE TEST (FAIL-FAST)
# ============================================================================
print("\nRunning fail-fast smoke test on fused architecture...")
smoke_model = load_model()
replace_batchnorm(smoke_model)
smoke_model = smoke_model.to(device).eval()

dummy_smoke = torch.randn(2, 3, 512, 512, device=device)
with torch.inference_mode():
    smoke_out = smoke_model(dummy_smoke)
    if isinstance(smoke_out, (tuple, list)):
        smoke_out = smoke_out[0]

assert smoke_out.shape == (2, 1000), f"Smoke test failed! Output shape: {smoke_out.shape}"
assert not torch.isnan(smoke_out).any(), "Smoke test failed! Output contains NaN."
del smoke_model, dummy_smoke, smoke_out
if device.type == "cuda":
    torch.cuda.empty_cache()
print("Smoke test: PASS (Fusion verified safe)\n")

# ============================================================================
# IMAGE DATASET (2048 IMAGES)
# ============================================================================
transform = transforms.Compose([
    transforms.Resize(585, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(IMAGENET_VAL, transform=transform)
subset = Subset(dataset, list(range(2048)))

accuracy_loader = DataLoader(
    subset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# ============================================================================
# BENCHMARK STAGES (STAGE 1: LATENCY -> STAGE 2: ACCURACY)
# ============================================================================
@torch.inference_mode()
def measure_latency(model):
    batch_size = 32
    warmup = 30
    iterations = 100

    dummy = torch.randn(batch_size, 3, 512, 512, device=device)
    model.eval()

    for _ in range(warmup):
        _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

        for i in range(iterations):
            start_events[i].record()
            _ = model(dummy)
            end_events[i].record()

        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    else:
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(dummy)
            end = time.perf_counter()
            times.append((end - start) * 1000.0)

    times.sort()
    mean_ms = sum(times) / len(times)
    median_ms = times[len(times) // 2]
    p95_ms = times[max(0, int(0.95 * len(times)) - 1)]
    fps = batch_size / (mean_ms / 1000.0)

    del dummy
    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "fps": fps,
    }

@torch.inference_mode()
def evaluate_accuracy(model, name):
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0

    for images, labels in accuracy_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        output = model(images)
        if isinstance(output, (tuple, list)):
            output = output[0]

        _, pred = output.topk(5, dim=1, largest=True, sorted=True)
        correct = (pred == labels.unsqueeze(1))

        correct1 += correct[:, 0].sum().item()
        correct5 += correct.any(dim=1).sum().item()
        total += labels.size(0)

    top1 = 100.0 * correct1 / total
    top5 = 100.0 * correct5 / total
    print(f"{name} -> Top-1: {top1:.3f}% ({correct1}/{total}) | Top-5: {top5:.3f}% ({correct5}/{total})")
    return top1, top5

# ============================================================================
# RUN CONFIGURATIONS
# ============================================================================
experiments = [
    ("RAW BASELINE", False, False),
    ("FUSED + CUDNN BENCHMARK", True, True),
]

results = {}

for name, do_fuse, do_benchmark in experiments:
    print("-" * 90)
    print(f"Testing: {name}")
    print("-" * 90)

    torch.backends.cudnn.benchmark = do_benchmark

    model = load_model()
    if do_fuse:
        replace_batchnorm(model)

    model = model.to(device).eval()

    latency = measure_latency(model)
    print(f"Mean Latency: {latency['mean_ms']:.3f} ms | FPS: {latency['fps']:.2f}")

    top1, top5 = evaluate_accuracy(model, name)

    results[name] = {
        "top1": top1,
        "top5": top5,
        "latency": latency,
    }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

# ============================================================================
# RESULTS COMPARISON
# ============================================================================
base = results["RAW BASELINE"]
opt = results["FUSED + CUDNN BENCHMARK"]

base_lat = base["latency"]["mean_ms"]
opt_lat = opt["latency"]["mean_ms"]
lat_diff = 100.0 * (opt_lat - base_lat) / base_lat

base_fps = base["latency"]["fps"]
opt_fps = opt["latency"]["fps"]
fps_diff = 100.0 * (opt_fps - base_fps) / base_fps

print("\n" + "=" * 90)
print("STEP 1 COMPARISON RESULTS")
print("=" * 90)
print(f"{'Metric':<25} {'Raw Baseline':>18} {'Fused Baseline':>18} {'Delta':>15}")
print("-" * 90)
print(f"{'Top-1 Accuracy':<25} {base['top1']:>17.3f}% {opt['top1']:>17.3f}% {opt['top1'] - base['top1']:>+14.3f}%")
print(f"{'Top-5 Accuracy':<25} {base['top5']:>17.3f}% {opt['top5']:>17.3f}% {opt['top5'] - base['top5']:>+14.3f}%")
print(f"{'Mean Latency':<25} {base_lat:>15.3f} ms {opt_lat:>15.3f} ms {lat_diff:>+14.2f}%")
print(f"{'Throughput (FPS)':<25} {base_fps:>18.2f} {opt_fps:>18.2f} {fps_diff:>+14.2f}%")
print("=" * 90)