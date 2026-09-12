import os
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CUDA_RUN = ROOT_DIR / "cuda_11.8.0.run"
CUDA_INSTALL_DIR = ROOT_DIR / "cuda" / "cuda-11.8"

url = "https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run"
dest = str(CUDA_RUN)

# 1. Download with real-time progress bar
if not os.path.exists(dest):
    print("Downloading CUDA 11.8 runfile (~3.0 GB)...")
    def report_hook(count, block_size, total_size):
        downloaded = count * block_size
        pct = downloaded / total_size * 100
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\rDownload Progress: [{pct:6.2f}%]  {mb_down:6.1f} MB / {mb_total:6.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=report_hook)
    print("\nDownload complete!")
else:
    print("CUDA 11.8 runfile already present. Skipping download.")

os.chmod(dest, 0o755)

# 2. Silent Toolkit Install
print(f"\nInstalling CUDA 11.8 Toolkit to {CUDA_INSTALL_DIR} (takes ~2-3 mins)...")
CUDA_INSTALL_DIR.mkdir(parents=True, exist_ok=True)

proc = subprocess.Popen(
    ["sudo", dest, "--toolkit", "--silent", f"--installpath={CUDA_INSTALL_DIR}"]
)

spinner = ["|", "/", "-", "\\"]
start_time = time.time()
idx = 0

while proc.poll() is None:
    elapsed = int(time.time() - start_time)
    mins, secs = divmod(elapsed, 60)
    print(f"\rInstalling... {spinner[idx % len(spinner)]} [Elapsed Time: {mins:02d}:{secs:02d}]", end="", flush=True)
    idx += 1
    time.sleep(0.5)

if proc.returncode == 0:
    print(f"\nCUDA 11.8 installed successfully in {int(time.time() - start_time)} seconds!")
else:
    raise RuntimeError(f"\nCUDA installation failed with exit code: {proc.returncode}")

# 3. Create driver library symlink
lib64_dir = CUDA_INSTALL_DIR / "lib64"
lib64_dir.mkdir(parents=True, exist_ok=True)
if os.path.exists("/usr/lib64-nvidia/libcuda.so"):
    subprocess.run(["sudo", "ln", "-sf", "/usr/lib64-nvidia/libcuda.so", str(lib64_dir / "libcuda.so")], check=False)