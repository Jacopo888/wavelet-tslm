"""Probe: verify Modal GPU entitlement and environment. Costs cents.

Run:  modal run modal_probe.py
"""

import modal

app = modal.App("wavelet-probe")

image = modal.Image.debian_slim(python_version="3.11").pip_install("torch")


@app.function(gpu="L4", image=image, timeout=300)
def probe():
    import subprocess

    print(subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
        capture_output=True, text=True).stdout)
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda,
          "available", torch.cuda.is_available())
    import os
    import shutil
    print("cpu ram GB:",
          round(int(os.popen("grep MemTotal /proc/meminfo").read().split()[1]) / 1e6, 1))
    print("disk free GB:", round(shutil.disk_usage("/").free / 1e9, 1))


@app.local_entrypoint()
def main():
    probe.remote()
