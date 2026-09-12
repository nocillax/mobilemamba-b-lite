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
print("MOBILEMAMBA-B4 EXECUTION TIME BREAKDOWN (BATCH SIZE = 32, 512x512)")
print("=" * 80)

dummy = torch.randn(32, 3, 512, 512, device=device)

with torch.inference_mode():
    for _ in range(20):
        _ = model(dummy)
torch.cuda.synchronize()

@torch.inference_mode()
def time_module(func, inp, runs=50, warmup=10):
    for _ in range(warmup):
        out = func(inp)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(runs):
        out = func(inp)
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) / runs), out

with torch.inference_mode():
    t_stem, x1 = time_module(model.patch_embed, dummy)
    t_b1, x2 = time_module(model.blocks1, x1)
    t_b2_total, x3 = time_module(model.blocks2, x2)

    def run_stage2_downsample(x):
        return model.blocks2[2](model.blocks2[1](model.blocks2[0](x)))

    t_b2_down, x_down = time_module(run_stage2_downsample, x2)
    b2_blk = model.blocks2[3]

    def run_pre_mixer(x):
        return b2_blk.dw0(b2_blk.ffn0(x))

    t_pre_mix, x_mid = time_module(run_pre_mixer, x_down)
    t_mixer, x_post_mix = time_module(b2_blk.mixer, x_mid)

    mixer_attn = b2_blk.mixer.m.attn
    global_in = x_mid[:, :mixer_attn.global_channels].contiguous()
    local_in = x_mid[:, mixer_attn.global_channels:mixer_attn.global_channels + mixer_attn.local_channels].contiguous()

    t_global_op, _ = time_module(mixer_attn.global_op, global_in)
    t_local_op, _ = time_module(mixer_attn.local_op, local_in)

    def run_post_mixer(x):
        return b2_blk.dw1(b2_blk.ffn1(x))

    t_post_mix, _ = time_module(run_post_mixer, x_post_mix)
    t_b3, x4 = time_module(model.blocks3, x3)
    t_head, _ = time_module(lambda x: model.head(torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)), x4)

t_total = t_stem + t_b1 + t_b2_total + t_b3 + t_head

print(f"{'Component':<40} {'Latency (ms)':>15} {'% of Total':>15}")
print("-" * 80)
print(f"{'1. Patch Embed (Stem, 128x128)':<40} {t_stem:>13.2f} ms {100*t_stem/t_total:>14.1f}%")
print(f"{'2. Stage 1 (blocks1, 128x128)':<40} {t_b1:>13.2f} ms {100*t_b1/t_total:>14.1f}%")
print(f"{'3. Stage 2 (blocks2, 64x64) Total':<40} {t_b2_total:>13.2f} ms {100*t_b2_total/t_total:>14.1f}%")
print(f"{'   - Downsampling Layer (blocks2[0..2])':<40} {t_b2_down:>13.2f} ms {100*t_b2_down/t_total:>14.1f}%")
print(f"{'   - Single Block Pre-Mixer (FFN0+DW0)':<40} {t_pre_mix:>13.2f} ms {100*t_pre_mix/t_total:>14.1f}%")
print(f"{'   - Single Block Mixer Total':<40} {t_mixer:>13.2f} ms {100*t_mixer/t_total:>14.1f}%")
print(f"{'     * Global Op (MBWTConv2d / Scan)':<40} {t_global_op:>13.2f} ms {100*t_global_op/t_total:>14.1f}%")
print(f"{'     * Local Op (DWConv)':<40} {t_local_op:>13.2f} ms {100*t_local_op/t_total:>14.1f}%")
print(f"{'   - Single Block Post-Mixer (FFN1+DW1)':<40} {t_post_mix:>13.2f} ms {100*t_post_mix/t_total:>14.1f}%")
print(f"{'4. Stage 3 (blocks3, 32x32)':<40} {t_b3:>13.2f} ms {100*t_b3/t_total:>14.1f}%")
print(f"{'5. Head & Pooling':<40} {t_head:>13.2f} ms {100*t_head/t_total:>14.1f}%")
print("-" * 80)
print(f"{'Calculated Total Sum':<40} {t_total:>13.2f} ms {'100.0%':>15}")
print("=" * 80)