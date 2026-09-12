import os
import sys
import copy
import time
import types
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT_DIR = Path(__file__).resolve().parent.parent
REPO = str(ROOT_DIR / "MobileMamba")
CHECKPOINT_PATH = str(ROOT_DIR / "MobileMamba" / "weights" / "MobileMamba_B4" / "mobilemamba_b4.pth")
IMAGENET_VAL = str(ROOT_DIR / "data" / "imagenet" / "val")

sys.path.insert(0, REPO)
sys.path.insert(0, str(ROOT_DIR / "MobileMamba" / "model" / "lib_mamba" / "kernels" / "selective_scan"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 105)
print("TESTING WAVELET BRANCH BYPASS (ZERO-SHOT)")
print("=" * 105)
print("Device     :", device)
print("Checkpoint :", CHECKPOINT_PATH)
print("ImageNet   :", IMAGENET_VAL)

from model.mobilemamba.mobilemamba import MobileMamba, CFG_MobileMamba_B4

def load_model():
    cfg = copy.deepcopy(CFG_MobileMamba_B4)
    model = MobileMamba(**cfg, num_classes=1000, distillation=False, forward_type="v052d")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    state_dict = {(k.replace("module.", "", 1) if k.startswith("module.") else k): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return model

def bypassed_mbwt_forward(self, x):
    x = self.base_scale(self.global_atten(x))
    if self.do_stride is not None:
        x = self.do_stride(x)
    return x

def apply_wavelet_bypass(model, stages_to_bypass):
    count = 0
    for stage_idx in stages_to_bypass:
        blocks = getattr(model, f"blocks{stage_idx}")
        for block in blocks:
            if hasattr(block, "mixer") and hasattr(block.mixer, "m"):
                global_op = block.mixer.m.attn.global_op
                global_op.forward = types.MethodType(bypassed_mbwt_forward, global_op)
                count += 1
    return model, count

print("\nRunning fail-fast smoke test on bypassed architecture...")
test_model = load_model()
test_model, patched_cnt = apply_wavelet_bypass(test_model, [1, 2, 3])
test_model = test_model.to(device).eval()

dummy = torch.randn(2, 3, 512, 512, device=device)
with torch.inference_mode():
    out = test_model(dummy)
    if isinstance(out, (tuple, list)):
        out = out[0]

assert out.shape == (2, 1000), f"Smoke test failed! Output shape: {out.shape}"
assert not torch.isnan(out).any(), "Smoke test failed! NaN output detected."
del test_model, dummy, out
torch.cuda.empty_cache()
print(f"Smoke test: PASS ({patched_cnt} MBWTConv2d blocks successfully bypassed)\n")

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

@torch.inference_mode()
def evaluate_accuracy(model, name):
    model.eval()
    correct1, correct5, total = 0, 0, 0

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

@torch.inference_mode()
def measure_latency(model):
    batch_size = 32
    warmup = 30
    iterations = 100

    dummy = torch.randn(batch_size, 3, 512, 512, device=device)
    model.eval()

    for _ in range(warmup):
        _ = model(dummy)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    for i in range(iterations):
        start_events[i].record()
        _ = model(dummy)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    times.sort()
    mean_ms = sum(times) / len(times)
    fps = batch_size / (mean_ms / 1000.0)

    del dummy
    return {"mean_ms": mean_ms, "fps": fps}

experiments = [
    ("BASELINE", []),
    ("BYPASS WAVELET (STAGE 2)", [2]),
    ("BYPASS WAVELET (STAGE 1)", [1]),
    ("BYPASS WAVELET (ALL STAGES)", [1, 2, 3]),
]

results = {}

for name, bypass_stages in experiments:
    print("-" * 105)
    print(f"Benchmarking: {name}")
    print("-" * 105)

    model = load_model()
    if bypass_stages:
        model, count = apply_wavelet_bypass(model, bypass_stages)
        print(f"Patched {count} blocks across stages {bypass_stages}")

    model = model.to(device).eval()

    latency = measure_latency(model)
    print(f"Mean Latency: {latency['mean_ms']:.3f} ms | FPS: {latency['fps']:.2f}")

    top1, top5 = evaluate_accuracy(model, name)

    results[name] = {"top1": top1, "top5": top5, "latency": latency}

    del model
    torch.cuda.empty_cache()

base = results["BASELINE"]
base_lat = base["latency"]["mean_ms"]
base_fps = base["latency"]["fps"]
base_top1 = base["top1"]
base_top5 = base["top5"]

print("\n" + "=" * 105)
print("FINAL EXPERIMENT RESULTS: WAVELET BRANCH OPTIMIZATION")
print("=" * 105)
header = f"{'Configuration':<32} {'Top-1':>8} {'ΔTop-1':>9} {'Top-5':>8} {'ΔTop-5':>9} {'Latency':>11} {'ΔLatency':>11} {'FPS':>8} {'ΔFPS':>8}"
print(header)
print("-" * 105)

for name in results:
    r = results[name]
    lat = r["latency"]["mean_ms"]
    fps = r["latency"]["fps"]

    top1_d = r["top1"] - base_top1
    top5_d = r["top5"] - base_top5
    lat_d = 100.0 * (lat - base_lat) / base_lat
    fps_d = 100.0 * (fps - base_fps) / base_fps

    print(
        f"{name:<32} "
        f"{r['top1']:>7.3f}% {top1_d:>+8.3f}% "
        f"{r['top5']:>7.3f}% {top5_d:>+8.3f}% "
        f"{lat:>8.3f} ms {lat_d:>+10.2f}% "
        f"{fps:>8.2f} {fps_d:>+7.2f}%"
    )

print("=" * 105)