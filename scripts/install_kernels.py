import subprocess
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
env_python = str(ROOT_DIR / "envs" / "mobilemamba_env" / "bin" / "python")
root = str(ROOT_DIR / "MobileMamba" / "model" / "lib_mamba" / "kernels" / "selective_scan")

env = os.environ.copy()

result = subprocess.run(
    [
        env_python,
        "-m",
        "pip",
        "install",
        ".",
        "--no-build-isolation",
    ],
    cwd=root,
    env=env,
    text=True,
)

if result.returncode != 0:
    raise RuntimeError(
        f"Author's selective-scan installation failed with code {result.returncode}"
    )