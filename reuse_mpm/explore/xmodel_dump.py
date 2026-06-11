"""Entrypoint: dump a fresh warp-MPM forward trajectory for cross-model comparison.

Task (2026-06-11): compare warp (ours/PhysGaussian) vs gic taichi MPM with ALL
alignable parameters equal, fresh forwards on both sides, pairwise per-particle
distances (no chamfer). This dumps the warp side: per-frame positions of every
sim particle in normalized sim space, straight from the simulator (no cache
trajectory is loaded; the scene cache only pins the particle discretisation so
both sides simulate the *same* particles).

Config is LOCAL (explore convention). Output dir is auto-created under
outputs/explore/xmodel_dump/.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import tyro

from ..gpu import pick_free_gpu


@dataclass
class XModelDumpConfig:
    cache_path: str = "outputs/forward_gen/06_tele_E1e5/scene_cache.pt"  # pins particles
    logE: float = 5.0
    nu: float = 0.3
    v0: Tuple[float, float, float] = (0.0, -0.5, 0.0)
    num_frames: int = 14
    label: str = "tele_E1e5"


def run(cfg: XModelDumpConfig) -> str:
    pick_free_gpu()
    import numpy as np
    import torch

    from ..config import SceneSpec, ScenePreset, SimConfig
    from ..mpm_rollout import MpmRollout
    from ..scene_io import load_from_spec
    from ..sim_render import make_constant_v0

    t0 = time.time()
    out_dir = os.path.join("outputs", "explore", "xmodel_dump", cfg.label)
    os.makedirs(out_dir, exist_ok=True)

    spec = SceneSpec(preset=ScenePreset.telephone, cache_path=cfg.cache_path)
    sim = SimConfig()  # defaults: substep=64, delta_t=1/30, grid 32, rho 2000, jelly
    scene = load_from_spec(spec, sim)
    roll = MpmRollout(scene, sim, requires_grad=False)
    v0 = make_constant_v0(scene, cfg.v0)

    frames: List[np.ndarray] = [scene.sim_xyzs.detach().cpu().numpy()]  # frame 0 = init, (N, 3)
    for ti in range(cfg.num_frames - 1):
        pos = roll.rollout_to_frame(cfg.logE, ti, v0, grad_window=1, requires_grad=False)
        frames.append(pos.detach().cpu().numpy())
        print(f"frame {ti + 1}/{cfg.num_frames - 1} done")
    traj = np.stack(frames)  # (T, N, 3) normalized sim space

    np.save(os.path.join(out_dir, "warp_traj.npy"), traj)
    meta = {
        "side": "warp(reuse_mpm, PhysGaussian fixed-corotated 'jelly')",
        "cache_path": cfg.cache_path,
        "logE": cfg.logE, "nu": cfg.nu, "v0": list(cfg.v0),
        "rho": sim.density, "substep": sim.substep, "delta_t": sim.delta_t,
        "dt": sim.delta_t / sim.substep, "grid_size": sim.grid_size,
        "grid_lim": sim.grid_lim, "dx": sim.grid_lim / sim.grid_size,
        "gravity": 0, "num_frames": cfg.num_frames,
        "n_particles": int(traj.shape[1]),
        "freeze": "exact grid freeze BC (27-node stencil)",
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[xmodel_dump] saved {traj.shape} -> {out_dir} ({time.time() - t0:.1f}s)")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(XModelDumpConfig))
