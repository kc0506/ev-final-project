"""Self-contained GPU quota guard for the teacher (genphys-diff env).

Mirrors reuse_mpm/gpu.py but stdlib-only and copied here ON PURPOSE: the trainer
runs in a DIFFERENT conda env that does not have reuse_mpm importable, and the
quota logic is just subprocess parsing of `ws-status` / `nvidia-smi`.

Training is a LONG run (hours), unlike the short MPM rollouts gpu.py guards, so
besides the launch-time floor we expose `quota_seconds()` for an in-loop check
that lets the trainer checkpoint + exit GRACEFULLY before the daily-quota system
hard-kills every GPU process of the user.
"""
from __future__ import annotations

import os
import re
import subprocess
from typing import Optional


def my_gpu_count() -> int:
    """Number of DISTINCT GPUs holding a compute process owned by THIS user.

    The shared-box policy penalises a user running N>idle GPUs, and a single
    trainer should occupy exactly ONE. If this returns >1 it means my jobs have
    spread across GPUs (concurrent launches -- each pick_free_gpu grabbed a
    different free card) -> contention/penalty. Returns -1 if nvidia-smi can't be
    read (caller fails open). Does NOT pin anything (see feedback-no-pin-gpu)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=15).decode().strip()
    except Exception:
        return -1
    uid = os.getuid()
    mine = set()
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        pid, bus = parts[0], parts[1]
        try:
            if os.stat(f"/proc/{pid}").st_uid == uid:
                mine.add(bus)
        except Exception:
            pass
    return len(mine)


def status_ok(stop_secs: int, max_my_gpus: int = 1) -> "tuple[bool, str]":
    """(ok, reason) for an in-loop guard: NOT ok if my GPU footprint spread to
    >max_my_gpus cards, or remaining quota dropped below stop_secs. Fails OPEN on
    unreadable signals (returns ok=True) so a flaky tool never stalls training."""
    n = my_gpu_count()
    if n > max_my_gpus:
        return False, f"using {n} GPUs (>{max_my_gpus}) -- concurrent-job spread/penalty"
    secs = quota_seconds()
    if secs is not None and secs < stop_secs:
        return False, f"quota {secs}s < {stop_secs}s floor"
    return True, "ok"


def quota_seconds() -> Optional[int]:
    """Remaining personal GPU quota in seconds (parsed from `ws-status`), or None."""
    try:
        out = subprocess.check_output(["ws-status"], stderr=subprocess.DEVNULL,
                                      timeout=15).decode()
    except Exception:
        return None
    m = re.search(r"GPU quota remaining:\s*(\d+)\s*secs", out)
    return int(m.group(1)) if m else None


def assert_quota(min_hours: float = 4.0, verbose: bool = True) -> None:
    """Abort at launch if remaining quota < min_hours (fails OPEN if unreadable)."""
    if min_hours <= 0:
        return
    secs = quota_seconds()
    if secs is None:
        if verbose:
            print(f"[gpu] WARNING: could not read quota; skipping {min_hours:g}h floor")
        return
    h = secs / 3600.0
    if h < min_hours:
        raise SystemExit(f"[gpu] ABORT: GPU quota {h:.1f}h < {min_hours:g}h floor "
                         f"(a long train would risk a mid-flight kill).")
    if verbose:
        print(f"[gpu] quota remaining {h:.1f}h (>= {min_hours:g}h floor)")


def _my_bus_ids() -> set:
    """gpu_bus_id of every GPU this uid already has a compute proc on."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=15).decode().strip()
    except Exception:
        return set()
    uid, mine = os.getuid(), set()
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 2:
            try:
                if os.stat(f"/proc/{p[0]}").st_uid == uid:
                    mine.add(p[1])
            except Exception:
                pass
    return mine


def pick_free_gpu(min_free_mib: int = 18000, min_quota_hours: float = 4.0,
                  prefer_my_gpu: bool = True, verbose: bool = True) -> int:
    """Set CUDA_VISIBLE_DEVICES (after the quota floor). Call BEFORE torch touches
    CUDA. No-op if CUDA_VISIBLE_DEVICES preset.

    STACK-FIRST: prefer a GPU this uid is ALREADY using that has >= min_free_mib
    free -- piling onto one card keeps the uid at ONE distinct GPU (no N>idle
    penalty, my_gpu_count stays 1), instead of grabbing a fresh idle card and
    spreading. Only when no in-use card has room do we take the freest idle one.
    (Penalty counts distinct GPUs held, not whether a card is shared.)"""
    assert_quota(min_quota_hours, verbose=verbose)
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        if verbose:
            print(f"[gpu] CUDA_VISIBLE_DEVICES preset to {os.environ['CUDA_VISIBLE_DEVICES']}")
        return int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,gpu_bus_id,memory.free",
             "--format=csv,noheader,nounits"]).decode().strip().splitlines()
        rows = [(int(i), b.strip(), int(m)) for i, b, m in (l.split(",") for l in out)]
        mybus = _my_bus_ids() if prefer_my_gpu else set()
        stack = sorted([(i, m) for i, b, m in rows if b in mybus and m >= min_free_mib],
                       key=lambda x: -x[1])
        if stack:
            idx, mib = stack[0]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
            if verbose:
                print(f"[gpu] STACKING on my in-use GPU {idx} ({mib} MiB free, fits "
                      f">={min_free_mib}) -> stays 1 card (policy: procs on a card "
                      f"don't add to the N-card quota rate)")
            return idx
        idx, _, mib = sorted(rows, key=lambda r: -r[2])[0]
        if mib < min_free_mib:
            print(f"[gpu] WARNING: freest GPU {idx} only {mib} MiB free (need {min_free_mib})")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        if verbose:
            print(f"[gpu] no in-use GPU with room -> freest idle GPU {idx} ({mib} MiB) -> cuda:0")
        return idx
    except Exception as e:
        print(f"[gpu] auto-pick failed ({e}); default GPU")
        return 0
