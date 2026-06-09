"""Pick the freest CUDA device on this shared box.

The cluster GPUs are shared; hard-coding cuda:0 leads to OOM when a neighbour
grabs it. Call pick_free_gpu() early (before torch initialises a context) to set
CUDA_VISIBLE_DEVICES to the GPU with the most free memory.
"""
from __future__ import annotations

import os
import subprocess


def pick_free_gpu(min_free_mib: int = 8000, verbose: bool = True) -> int:
    """Set CUDA_VISIBLE_DEVICES to the freest GPU. Returns its physical index.

    No-op if CUDA_VISIBLE_DEVICES is already set by the caller.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        if verbose:
            print(f"[gpu] CUDA_VISIBLE_DEVICES preset to "
                  f"{os.environ['CUDA_VISIBLE_DEVICES']}")
        return int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"]
        ).decode().strip().splitlines()
        free = [(int(i), int(m)) for i, m in (l.split(",") for l in out)]
        free.sort(key=lambda x: -x[1])
        idx, mib = free[0]
        if mib < min_free_mib:
            print(f"[gpu] WARNING: freest GPU {idx} only has {mib} MiB free")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        if verbose:
            print(f"[gpu] selected physical GPU {idx} ({mib} MiB free) "
                  f"-> exposed as cuda:0")
        return idx
    except Exception as e:
        print(f"[gpu] auto-pick failed ({e}); falling back to default")
        return 0
