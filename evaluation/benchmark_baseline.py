import os
import sys
import copy
import time
import json
import argparse
import platform
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# ========================================================
# ENVIRONMENT & PATH SETUP (Aligned with Test 7)
# ========================================================
REPO = "/content/MobileMamba"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "model/lib_mamba/kernels/selective_scan"))

from model.mobilemamba.mobilemamba import (
    MobileMamba,
    CFG_MobileMamba_B1,
    CFG_MobileMamba_B2,
    CFG_MobileMamba_B4,
)

VARIANT_REGISTRY = {
    "b1": {"cfg": CFG_MobileMamba_B1, "res": 256, "resize": 292, "default_ckpt": "/content/MobileMamba/weights/MobileMamba_B1/mobilemamba_b1.pth"},
    "b2": {"cfg": CFG_MobileMamba_B2, "res": 384, "resize": 438, "default_ckpt": "/content/MobileMamba/weights/MobileMamba_B2/mobilemamba_b2.pth"},
    "b4": {"cfg": CFG_MobileMamba_B4, "res": 512, "resize": 585, "default_ckpt": "/content/MobileMamba/weights/MobileMamba_B4/mobilemamba_b4.pth"},
}


def log(msg):
    """Guarantees unbuffered real-time stdout output in Colab."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def percentile(values, p):
    """Linear-interpolated percentile."""
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return float(values[f])
    return float(values[f] + (values[c] - values[f]) * (k - f))


def parse_args():
    parser = argparse.ArgumentParser(description="Proper MobileMamba ImageNet Baseline Benchmark")
    parser.add_argument("--variant", type=str, default="b1", choices=["b1", "b2", "b4"], help="Model variant: b1, b2, or b4")
    parser.add_argument("--data-dir", type=str, default="/content/imagenet/val")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--warmup-batches", type=int, default=20)
    parser.add_argument("--forward-type", type=str, default="v052d")
    parser.add_argument("--amp", action="store_true", help="Enable AMP FP16. (Omit to run pure FP32 matching Test 7)")
    parser.add_argument("--save-path", type=str, default=None)
    return parser.parse_args()


def load_model(variant_key, checkpoint_path, forward_type, device):
    meta = VARIANT_REGISTRY[variant_key]
    cfg = copy.deepcopy(meta["cfg"])
    
    # Direct instantiation eliminates the get_model() TypeError
    model = MobileMamba(
        **cfg,
        num_classes=1000,
        distillation=False,
        forward_type=forward_type
    )

    log(f"Locating checkpoint at: {checkpoint_path}")
    assert os.path.exists(checkpoint_path), f"Weights missing: {checkpoint_path}"

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    state_dict = {(k.replace("module.", "", 1) if k.startswith("module.") else k): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    log(f"Checkpoint loaded successfully (Missing: {len(missing)}, Unexpected: {len(unexpected)})")
    
    return model.to(device).eval()


def main():
    args = parse_args()
    variant = args.variant.lower()
    meta = VARIANT_REGISTRY[variant]
    res, resize = meta["res"], meta["resize"]

    if args.checkpoint is None:
        args.checkpoint = meta["default_ckpt"]

    if args.save_path is None:
        args.save_path = f"/content/{variant.upper()}_ImageNet_baseline_PROPER.json"

    # ========================================================
    # 1. DEVICE / HARDWARE
    # ========================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This benchmark requires CUDA/GPU.")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_total_memory_gb = gpu_props.total_memory / (1024 ** 3)

    log(f"Target Hardware: {gpu_name}")
    log(f"GPU Memory: {gpu_total_memory_gb:.2f} GB")
    log(f"PyTorch: {torch.__version__}")
    log(f"CUDA Runtime: {torch.version.cuda}")
    log(f"Python: {platform.python_version()}")
    log(f"Precision Mode: {'AMP FP16' if args.amp else 'Pure FP32 (Test 7 Matched)'}")

    # ========================================================
    # 2 & 3. MODEL
    # ========================================================
    log(f"Instantiating MobileMamba_{variant.upper()} model (forward_type={args.forward_type})...")
    model = load_model(variant, args.checkpoint, args.forward_type, device)
    torch.cuda.synchronize()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    base_vram_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)

    log("Model successfully loaded.")
    log(f"Total Parameters: {total_params:,} ({total_params / 1e6:.3f}M)")
    log(f"Trainable Parameters: {trainable_params:,}")
    log(f"Base Model VRAM: {base_vram_mb:.2f} MB")

    # ========================================================
    # 4. FLOPs / MACs
    # ========================================================
    log("Attempting FLOPs / MACs measurement...")
    flops, macs, flops_method, flops_error = None, None, None, None

    try:
        from fvcore.nn import FlopCountAnalysis
        dummy_input = torch.randn(1, 3, res, res, device=device)
        with torch.no_grad():
            analysis = FlopCountAnalysis(model, dummy_input)
            analysis.unsupported_ops_warnings(False)
            flops = float(analysis.total())
            macs = flops / 2.0
        del dummy_input
        flops_method = "fvcore"
        log(f"FLOPs: {flops / 1e9:.4f} G")
        log(f"MACs: {macs / 1e9:.4f} G")
    except Exception as e:
        flops_method = "failed"
        flops_error = str(e)
        log("WARNING: FLOPs/MACs measurement failed.")
        log(f"Reason: {e}")
        log("Continuing benchmark...")

    # ========================================================
    # 4.5 ISOLATED FORWARD LATENCY (Aligned with Test 7)
    # ========================================================
    log("Running isolated GPU forward latency benchmark (synthetic tensor, steady state)...")
    dummy_lat = torch.randn(args.batch_size, 3, res, res, device=device)
    isolated_forward_times_ms = []

    with torch.inference_mode():
        # Warmup on synthetic data
        for _ in range(30):
            with torch.amp.autocast(device_type="cuda", enabled=args.amp):
                _ = model(dummy_lat)
        torch.cuda.synchronize()

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(100)]

        for j in range(100):
            start_events[j].record()
            with torch.amp.autocast(device_type="cuda", enabled=args.amp):
                _ = model(dummy_lat)
            end_events[j].record()

        torch.cuda.synchronize()
        isolated_forward_times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    del dummy_lat
    torch.cuda.empty_cache()

    iso_mean = float(np.mean(isolated_forward_times_ms))
    iso_fps = args.batch_size / (iso_mean / 1000.0)
    log(f"Isolated Benchmark -> Mean: {iso_mean:.3f} ms | FPS: {iso_fps:.2f}")

    # ========================================================
    # 5. DATASET
    # ========================================================
    log(f"Setting up ImageNet transforms (BICUBIC {resize} -> Crop {res})...")
    val_transform = transforms.Compose([
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(res),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    log(f"Scanning directory: {args.data_dir}")
    idx_start = time.time()
    dataset = datasets.ImageFolder(args.data_dir, transform=val_transform)
    log(f"Dataset indexed {len(dataset)} images in {time.time() - idx_start:.2f}s")

    # ========================================================
    # 6. DATALOADER
    # ========================================================
    log(f"Creating DataLoader (Batch={args.batch_size}, Workers={args.workers}, Pin_Memory=True)...")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.workers > 0),
    )
    total_batches = len(loader)
    log(f"Ready. Total batches: {total_batches}")
    print("=" * 70, flush=True)

    # ========================================================
    # 7. WARMUP
    # ========================================================
    log(f"Running {args.warmup_batches} GPU warmup batches...")
    with torch.inference_mode():
        for i, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=args.amp):
                _ = model(images)
            if i + 1 >= args.warmup_batches:
                break
    torch.cuda.synchronize()
    log("GPU warmup complete.")

    # ========================================================
    # 8. RESET MEMORY STATISTICS
    # ========================================================
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # ========================================================
    # 9. MAIN EVALUATION (Full 50K with Per-Batch Logs)
    # ========================================================
    top1_correct = 0
    top5_correct = 0
    total_samples = 0

    batch_forward_times_ms = []
    end_to_end_times_ms = []
    full_batches_measured = 0
    incomplete_batches = 0

    eval_start = time.time()
    log("Starting full 50K-image benchmark...")
    print("=" * 70, flush=True)

    with torch.inference_mode():
        for i, (images, targets) in enumerate(loader):
            step_start = time.perf_counter()
            current_batch_size = images.size(0)

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            torch.cuda.synchronize()

            # CUDA event timing per batch
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            with torch.amp.autocast(device_type="cuda", enabled=args.amp):
                outputs = model(images)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
            end_event.record()

            torch.cuda.synchronize()
            fwd_ms = start_event.elapsed_time(end_event)

            # Accuracy calculation
            _, pred = outputs.topk(5, dim=1, largest=True, sorted=True)
            correct = pred.eq(targets.view(-1, 1))

            top1_correct += correct[:, :1].sum().item()
            top5_correct += correct[:, :5].sum().item()
            total_samples += current_batch_size

            # Performance timing
            if current_batch_size == args.batch_size:
                batch_forward_times_ms.append(fwd_ms)
                step_end = time.perf_counter()
                end_to_end_times_ms.append((step_end - step_start) * 1000.0)
                full_batches_measured += 1
            else:
                incomplete_batches += 1

            top1_acc = (top1_correct / total_samples) * 100.0
            top5_acc = (top5_correct / total_samples) * 100.0

            # ETA calculation
            elapsed = time.time() - eval_start
            avg_per_batch = elapsed / (i + 1)
            remaining_batches = total_batches - (i + 1)
            eta_mins = (remaining_batches * avg_per_batch) / 60.0

            # Per-batch logging
            log(
                f"DONE Batch {i+1}/{total_batches} | "
                f"Top-1: {top1_acc:5.2f}% | "
                f"Top-5: {top5_acc:5.2f}% | "
                f"Fwd: {fwd_ms:6.2f}ms | "
                f"Batch: {current_batch_size} | "
                f"ETA: {eta_mins:4.1f}m"
            )
            print("-" * 70, flush=True)

    torch.cuda.synchronize()
    total_wall_seconds = time.time() - eval_start

    # ========================================================
    # 10. FINAL ACCURACY
    # ========================================================
    final_top1 = (top1_correct / total_samples) * 100.0
    final_top5 = (top5_correct / total_samples) * 100.0

    # ========================================================
    # 11. LATENCY STATISTICS
    # ========================================================
    def latency_stats(values):
        if not values:
            return {"count": 0, "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
        return {
            "count": len(values),
            "mean_ms": float(np.mean(values)),
            "median_ms": float(np.median(values)),
            "p95_ms": percentile(values, 95),
            "min_ms": float(np.min(values)),
            "max_ms": float(np.max(values)),
        }

    # Isolated stats represent clean GPU execution (Fair and Square with Test 7)
    isolated_stats = latency_stats(isolated_forward_times_ms)
    # Loop stats capture in-flight forward times
    loop_forward_stats = latency_stats(batch_forward_times_ms)
    e2e_stats = latency_stats(end_to_end_times_ms)

    # ========================================================
    # 12. THROUGHPUT
    # ========================================================
    forward_fps = args.batch_size / (isolated_stats["mean_ms"] / 1000.0) if isolated_stats["mean_ms"] > 0 else 0.0
    e2e_fps = args.batch_size / (e2e_stats["mean_ms"] / 1000.0) if e2e_stats["mean_ms"] > 0 else 0.0

    # ========================================================
    # 13. VRAM
    # ========================================================
    peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    final_allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    final_reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)

    # ========================================================
    # 14. COMPLETE REPORT
    # ========================================================
    report = {
        "experiment": {
            "name": f"MobileMamba_{variant.upper()}_ImageNet1K_BASELINE",
            "model": f"MobileMamba_{variant.upper()}",
            "dataset": "ImageNet-1K validation",
            "input_resolution": f"{res}x{res}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "accuracy": {
            "top1_accuracy_percent": round(final_top1, 4),
            "top5_accuracy_percent": round(final_top5, 4),
            "evaluated_images": total_samples,
            "top1_correct": int(top1_correct),
            "top5_correct": int(top5_correct),
        },
        "model_complexity": {
            "total_parameters": int(total_params),
            "total_parameters_millions": round(total_params / 1e6, 4),
            "trainable_parameters": int(trainable_params),
            "trainable_parameters_millions": round(trainable_params / 1e6, 4),
            "flops": float(flops) if flops is not None else None,
            "flops_giga": round(flops / 1e9, 4) if flops is not None else None,
            "macs": float(macs) if macs is not None else None,
            "macs_giga": round(macs / 1e9, 4) if macs is not None else None,
            "measurement_method": flops_method,
            "measurement_error": flops_error,
        },
        "performance": {
            "batch_size": args.batch_size,
            "warmup_batches": args.warmup_batches,
            "full_batches_measured": full_batches_measured,
            "incomplete_batches_excluded": incomplete_batches,
            "isolated_forward_latency": isolated_stats,
            "in_loop_forward_latency": loop_forward_stats,
            "end_to_end_batch_latency": e2e_stats,
            "forward_throughput_fps": round(forward_fps, 4),
            "end_to_end_throughput_fps": round(e2e_fps, 4),
            "total_benchmark_time_seconds": round(total_wall_seconds, 4),
        },
        "memory": {
            "gpu_total_memory_gb": round(gpu_total_memory_gb, 4),
            "base_model_vram_mb": round(base_vram_mb, 4),
            "peak_allocated_vram_mb": round(peak_allocated_mb, 4),
            "peak_reserved_vram_mb": round(peak_reserved_mb, 4),
            "final_allocated_vram_mb": round(final_allocated_mb, 4),
            "final_reserved_vram_mb": round(final_reserved_mb, 4),
        },
        "hardware": {
            "gpu": gpu_name,
            "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
            "cuda": torch.version.cuda,
            "pytorch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "dataloader": {
            "batch_size": args.batch_size,
            "workers": args.workers,
            "shuffle": False,
            "pin_memory": True,
            "drop_last": False,
            "persistent_workers": (args.workers > 0),
        },
        "benchmark_methodology": {
            "amp": args.amp,
            "amp_dtype": "FP16" if args.amp else "FP32",
            "forward_type": args.forward_type,
            "cuda_synchronized": True,
            "cuda_event_forward_timing": True,
            "warmup_excluded": True,
            "incomplete_final_batch_excluded": True,
            "accuracy_includes_final_batch": True,
            "input_preprocessing": f"Resize {resize} BICUBIC -> CenterCrop {res} -> ToTensor -> ImageNet Normalize",
        },
    }

    # ========================================================
    # 15. SAVE
    # ========================================================
    with open(args.save_path, "w") as f:
        json.dump(report, f, indent=4)

    # ========================================================
    # 16. FINAL SUMMARY (Restored Full Printout)
    # ========================================================
    print()
    print("=" * 70, flush=True)
    print(f"{variant.upper()} + ImageNet-1K PROPER BASELINE COMPLETE", flush=True)
    print("=" * 70, flush=True)

    print()
    print("ACCURACY", flush=True)
    print("-" * 70, flush=True)
    print(f"Top-1: {final_top1:.4f}%", flush=True)
    print(f"Top-5: {final_top5:.4f}%", flush=True)

    print()
    print("MODEL", flush=True)
    print("-" * 70, flush=True)
    print(f"Parameters: {total_params:,} ({total_params / 1e6:.3f}M)", flush=True)
    if flops is not None:
        print(f"FLOPs: {flops / 1e9:.4f} G", flush=True)
        print(f"MACs: {macs / 1e9:.4f} G", flush=True)

    print()
    print("ISOLATED FORWARD PERFORMANCE (Smoke Test Matched)", flush=True)
    print("-" * 70, flush=True)
    print(f"Mean: {isolated_stats['mean_ms']:.4f} ms", flush=True)
    print(f"Median: {isolated_stats['median_ms']:.4f} ms", flush=True)
    print(f"P95: {isolated_stats['p95_ms']:.4f} ms", flush=True)
    print(f"Min: {isolated_stats['min_ms']:.4f} ms", flush=True)
    print(f"Max: {isolated_stats['max_ms']:.4f} ms", flush=True)
    print(f"Forward FPS: {forward_fps:.4f}", flush=True)

    print()
    print("END-TO-END PERFORMANCE", flush=True)
    print("-" * 70, flush=True)
    print(f"Mean batch: {e2e_stats['mean_ms']:.4f} ms", flush=True)
    print(f"Median batch: {e2e_stats['median_ms']:.4f} ms", flush=True)
    print(f"P95 batch: {e2e_stats['p95_ms']:.4f} ms", flush=True)
    print(f"End-to-end FPS: {e2e_fps:.4f}", flush=True)

    print()
    print("VRAM", flush=True)
    print("-" * 70, flush=True)
    print(f"Base: {base_vram_mb:.2f} MB", flush=True)
    print(f"Peak allocated: {peak_allocated_mb:.2f} MB", flush=True)
    print(f"Peak reserved: {peak_reserved_mb:.2f} MB", flush=True)

    print()
    print("BENCHMARK DETAILS", flush=True)
    print("-" * 70, flush=True)
    print(f"Images evaluated: {total_samples:,}", flush=True)
    print(f"Full batches measured: {full_batches_measured:,}", flush=True)
    print(f"Incomplete batches excluded: {incomplete_batches}", flush=True)
    print(f"Total runtime: {total_wall_seconds / 60:.2f} minutes", flush=True)

    print()
    print(f"Results saved to: {args.save_path}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()