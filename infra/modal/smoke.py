"""Minimal Modal smoke test: confirms auth + remote execution + GPU visibility.

Run:
    modal run infra/modal/smoke.py            # CPU-only auth/exec check
    modal run infra/modal/smoke.py::gpu_check # GPU check (costs a few cents)
"""

import modal

app = modal.App("genphys-smoke")

# Tiny CPU image — just to prove the round-trip works.
cpu_image = modal.Image.debian_slim(python_version="3.12")

# A CUDA base so torch can see the GPU. Use a generic CUDA runtime; we only probe.
gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-base-ubuntu22.04", add_python="3.10")
    .pip_install("torch==2.4.0", index_url="https://download.pytorch.org/whl/cu121")
)


@app.function(image=cpu_image)
def hello() -> str:
    import platform

    return f"hello from modal: python {platform.python_version()}"


@app.function(image=gpu_image, gpu="T4")
def gpu_check() -> str:
    import subprocess

    import torch

    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    msg = (f"torch {torch.__version__} | cuda avail={torch.cuda.is_available()} | "
           f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'} | "
           f"nvidia-smi: {smi}")
    print("GPU_CHECK:", msg)
    return msg


@app.local_entrypoint()
def main():
    print(hello.remote())
