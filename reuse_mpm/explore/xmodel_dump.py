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
from typing import List, Optional, Tuple

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
    # integer-cell translation applied at load (k * dx per axis). B-spline weights
    # depend only on the fractional grid offset, so this is bit-equivalent physics
    # at a different distance from the domain walls / position clamp.
    shift_cells: Tuple[int, int, int] = (0, 0, 0)
    # rotation about z through (0.5, 0.5), in degrees. NOT bit-equivalent (the MPM
    # grid is axis-aligned, so this re-discretizes the object); that is the point:
    # probe how object orientation vs grid/axes changes the dynamics.
    rot_z_deg: float = 0.0
    # sampling-rate override for FFT probing: e.g. delta_t=1/60 + substep=32 keeps
    # the physical sub-dt identical (substep_size is derived) while doubling the
    # frame sampling rate / Nyquist. Both None = defaults (1/30, 64).
    delta_t: Optional[float] = None
    substep: Optional[int] = None
    # streaming=True uses the O(T) no-grad forward (sim_render.simulate_positions);
    # the default MpmRollout path re-rolls from frame 0 per frame (O(T^2)) -- fine
    # for <=14 frames, ~100x slower at 128 frames. Same kernels, same physics.
    streaming: bool = False


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
    if cfg.delta_t is not None:
        sim.delta_t = cfg.delta_t
    if cfg.substep is not None:
        sim.substep = cfg.substep
    scene = load_from_spec(spec, sim)
    if any(cfg.shift_cells):
        dx = sim.grid_lim / sim.grid_size
        s = torch.tensor(cfg.shift_cells, dtype=scene.sim_xyzs.dtype,
                         device=scene.sim_xyzs.device) * dx  # [3]
        scene.sim_xyzs = scene.sim_xyzs + s
        scene.sim_aabb = scene.sim_aabb + s.to(scene.sim_aabb.device)
    if cfg.rot_z_deg:
        import math
        t = math.radians(cfg.rot_z_deg)
        c, s_ = math.cos(t), math.sin(t)
        p = scene.sim_xyzs
        x, y = p[:, 0] - 0.5, p[:, 1] - 0.5
        pr = p.clone()
        pr[:, 0] = c * x - s_ * y + 0.5
        pr[:, 1] = s_ * x + c * y + 0.5
        scene.sim_xyzs = pr
        scene.sim_aabb = torch.stack([pr.min(0).values, pr.max(0).values])
        print(f"[xmodel_dump] rotated z {cfg.rot_z_deg} deg; bbox "
              f"{pr.min(0).values.tolist()} .. {pr.max(0).values.tolist()}")
    v0 = make_constant_v0(scene, cfg.v0)

    if cfg.streaming:
        from ..sim_render import simulate_positions
        sim.num_frames = cfg.num_frames
        pos_list = simulate_positions(scene, 10.0 ** cfg.logE, v0, sim)  # world coords
        traj = np.stack([((p + scene.shift) / scene.scale).detach().cpu().numpy()
                         for p in pos_list])  # back to normalized sim space
    else:
        roll = MpmRollout(scene, sim, requires_grad=False)
        frames: List[np.ndarray] = [scene.sim_xyzs.detach().cpu().numpy()]  # frame 0, (N, 3)
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
        "shift_cells": list(cfg.shift_cells),
        "rot_z_deg": cfg.rot_z_deg,
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
