"""Pick the freest CUDA device on this shared box.

The cluster GPUs are shared; hard-coding cuda:0 leads to OOM when a neighbour
grabs it. Call pick_free_gpu() early (before torch initialises a context) to set
CUDA_VISIBLE_DEVICES to the GPU with the most free memory.

Shared-box quota: each user has a daily GPU-hour budget (see `gpu-policy`); once
it hits 0 the system KILLS all of the user's GPU processes mid-run. pick_free_gpu
therefore guards a hard quota floor first -- better to abort at launch than to
lose a long run (and its half-written output) to a mid-flight kill.
"""
from __future__ import annotations

import os
import re
import subprocess
from typing import Optional


def gpu_quota_seconds() -> Optional[int]:
    """Remaining personal GPU quota in seconds, parsed from `ws-status`.

    Returns None if the tool is unavailable / its quota line can't be found, so
    callers can decide whether to fail open or closed.
    """
    try:
        out = subprocess.check_output(
            ["ws-status"], stderr=subprocess.DEVNULL, timeout=15
        ).decode()
    except Exception:
        return None
    m = re.search(r"GPU quota remaining:\s*(\d+)\s*secs", out)
    return int(m.group(1)) if m else None


def assert_gpu_quota(min_hours: float = 12.0, verbose: bool = True) -> None:
    """Hard floor: abort the process if remaining GPU quota < `min_hours`.

    Defensive guard against launching a long run that the quota system will kill
    half-way (per the daily GPU-hour policy). `min_hours=0` disables the floor.
    Fails OPEN: if the quota can't be read (tool missing/slow), warns and
    proceeds rather than blocking legitimate work.
    """
    if min_hours <= 0:
        return
    secs = gpu_quota_seconds()
    if secs is None:
        if verbose:
            print(f"[gpu] WARNING: could not read GPU quota (ws-status); "
                  f"skipping the {min_hours:g}h floor check")
        return
    hours = secs / 3600.0
    if hours < min_hours:
        raise SystemExit(
            f"[gpu] ABORT: GPU quota {hours:.1f}h < {min_hours:g}h floor -- "
            f"too little to safely finish a run (it would risk a mid-flight kill).\n"
            f"       lower the floor explicitly if you really mean to: "
            f"pick_free_gpu(min_quota_hours=<h>) or min_quota_hours=0 to disable.")
    if verbose:
        print(f"[gpu] quota remaining {hours:.1f}h (>= {min_hours:g}h floor)")


def pick_free_gpu(min_free_mib: int = 8000, min_quota_hours: float = 8.0,
                  verbose: bool = True) -> int:
    """Set CUDA_VISIBLE_DEVICES to the freest GPU. Returns its physical index.

    Aborts first if the remaining GPU quota is below `min_quota_hours` (hard
    floor, set 0 to disable). No-op for device selection if CUDA_VISIBLE_DEVICES
    is already set by the caller -- but the quota floor is still enforced.
    """
    assert_gpu_quota(min_quota_hours, verbose=verbose)  # before touching CUDA
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
