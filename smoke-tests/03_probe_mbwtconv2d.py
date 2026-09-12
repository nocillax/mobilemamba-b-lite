import os
import sys
import copy
import time
from pathlib import Path
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent.parent
REPO = str(ROOT_DIR / "MobileMamba")
CHECKPOINT_PATH = str(ROOT_DIR / "MobileMamba" / "weights" / "MobileMamba_B4" / "mobilemamba_b4.pth")

sys.path.insert(0, REPO)
sys.path.insert(0, str(ROOT_DIR / "MobileMamba" / "model" / "lib_mamba" / "kernels" / "selective_scan"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from model.mobilemamba.mobilemamba import MobileMamba, CFG_MobileMamba_B4

cfg = copy.deepcopy(CFG_MobileMamba_B4)
model = MobileMamba(**cfg, num_classes=1000, distillation=False, forward_type="v052d")

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
state_dict = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)
model = model.to(device).eval()

print("=" * 80)
print("PROBING MBWTConv2d: WAVELET BRANCH vs. MAMBA SCAN (512x512 INPUT)")
print("=" * 80)

@torch.inference_mode()
def time_op(func, inp, runs=100, warmup=15):
    for _ in range(warmup):
        _ = func(inp)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(runs):
        _ = func(inp)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / runs

with torch.inference_mode():
    dummy = torch.randn(32, 3, 512, 512, device=device)

    feat_s1 = model.patch_embed(dummy)
    b1_blk = model.blocks1[0]
    b1_op = b1_blk.mixer.m.attn.global_op

    x_s1_pre = b1_blk.dw0(b1_blk.ffn0(feat_s1))
    x_s1 = x_s1_pre[:, :b1_op.in_channels].contiguous()

    out_s1 = model.blocks1(feat_s1)
    down_s2 = model.blocks2[2](model.blocks2[1](model.blocks2[0](out_s1)))

    b2_blk = model.blocks2[3]
    b2_op = b2_blk.mixer.m.attn.global_op

    x_s2_pre = b2_blk.dw0(b2_blk.ffn0(down_s2))
    x_s2 = x_s2_pre[:, :b2_op.in_channels].contiguous()

    del dummy, feat_s1, out_s1, down_s2, x_s1_pre, x_s2_pre
    torch.cuda.empty_cache()

    probes = [
        (f"Stage 1 ({x_s1.shape[2]}x{x_s1.shape[3]}, C={b1_op.in_channels})", b1_op, x_s1),
        (f"Stage 2 ({x_s2.shape[2]}x{x_s2.shape[3]}, C={b2_op.in_channels})", b2_op, x_s2)
    ]

    for stage_name, op, inp in probes:
        t_total = time_op(op, inp)
        t_mamba = time_op(lambda x: op.base_scale(op.global_atten(x)), inp)

        def run_wavelet_only(x):
            curr_x_ll = x
            curr_shape = curr_x_ll.shape
            curr_x = op.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = op.wavelet_scale[0](op.wavelet_convs[0](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)
            x_ll = curr_x_tag[:, :, 0, :, :]
            x_h = curr_x_tag[:, :, 1:4, :, :]
            curr_x = torch.cat([x_ll.unsqueeze(2), x_h], dim=2)
            next_x_ll = op.iwt_function(curr_x)
            return next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        t_wavelet = time_op(run_wavelet_only, inp)

        print(f"\n--- {stage_name} ---")
        print(f"Total MBWTConv2d    : {t_total:.3f} ms")
        print(f"  Mamba SS2D Scan   : {t_mamba:.3f} ms ({100*t_mamba/t_total:.1f}%)")
        print(f"  Wavelet (DWT+IDWT): {t_wavelet:.3f} ms ({100*t_wavelet/t_total:.1f}%)")

print("=" * 80)