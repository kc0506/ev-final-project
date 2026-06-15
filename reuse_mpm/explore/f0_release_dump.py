"""Entrypoint: dump a warp 'pure-release' GT trajectory + its F0 field, so gic can
fit E with the deformation gradient FIXED (known) -- the F0 analog of the warp->gic
fix-v0 cross-sim E recovery.

From a snapshot (x, F): strip rotation to the left stretch V0 = sqrt(F F^T) (the
dynamics-identifiable, gic/warp-shared canonical F0), set v0=0, C=0, and roll warp
forward at the true E. Saves, in NORMALIZED sim coords (what gic --gt_traj_normalized
expects):
  warp_traj.npy  [T, n, 3]   per-frame positions (frame0 = snapshot x)
  f0_field.npy   [n, 3, 3]   V0 to inject into gic's initial F
  init_xyz.npy   [n, 3]      snapshot positions = the new t0 (gic init_vol)
  meta.json      gt_logE, nu, num_frames, maxdev

Output under outputs/explore/f0_release_dump/.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import tyro

from ..gpu import pick_free_gpu


@dataclass
class F0ReleaseDumpConfig:
    cache_path: str = ("/tmp2/b10401006/ev-project/generative-phys/outputs/"
                       "_scene_cache/telephone_ds0.1_g32_k8.pt")
    snapshot: str = ("/tmp2/b10401006/ev-project/generative-phys/outputs/"
                     "explore/f0_snapshot/tele_f0_xy/snapshot_frame08.pt")
    gt_logE: float = 5.0
    nu: float = 0.3
    num_frames: int = 9          # frame0 + K=8 future (matches landscape K)
    label: str = "tele_f0release_E1e5_f8"


def run(cfg: F0ReleaseDumpConfig) -> str:
    pick_free_gpu()
    import numpy as np
    import torch
    import warp as wp

    from ..config import SceneSpec, ScenePreset, SimConfig
    from ..scene_io import load_from_spec
    from ..sim_render import build_mpm

    t0 = time.time()
    out_dir = os.path.join("outputs", "explore", "f0_release_dump", cfg.label)
    os.makedirs(out_dir, exist_ok=True)

    spec = SceneSpec(preset=ScenePreset.telephone, cache_path=cfg.cache_path)
    sim = SimConfig(); sim.num_frames = cfg.num_frames
    scene = load_from_spec(spec, sim)
    dev = scene.device
    rest_xyz = scene.sim_xyzs
    n = rest_xyz.shape[0]
    free = (~scene.freeze_mask).to(dev)

    snap = torch.load(cfg.snapshot, map_location=dev)
    snap_x = snap["x"].to(dev).float()
    snap_F = snap["F"].to(dev).float()
    # left stretch V0 = sqrt(F F^T) (rotation stripped; right-rotation is a
    # dynamics gauge for both warp jelly and gic material 10, isotropic).
    FFt = snap_F @ snap_F.transpose(-1, -2)
    mu, Q = torch.linalg.eigh(FFt)
    mu = mu.clamp_min(1e-9)
    V0 = (Q * mu.sqrt().unsqueeze(-2)) @ Q.transpose(-1, -2)
    maxdev = float((mu[free].sqrt() - 1).abs().max())
    print(f"[dump] N={n} free={int(free.sum())} V0 maxdev {maxdev:.3f}; release v0=0")

    solver, state, model = build_mpm(scene, sim, requires_grad=False)
    dens = torch.ones_like(rest_xyz[..., 0]) * sim.density
    state.reset_density(dens.clone(), torch.ones_like(dens).int(), dev, update_mass=True)
    with torch.no_grad():
        E_t = torch.ones_like(rest_xyz[..., 0]) * float(10.0 ** cfg.gt_logE)
        nu_t = torch.ones_like(rest_xyz[..., 0]) * cfg.nu
        solver.set_E_nu_from_torch(model, E_t.clone(), nu_t.clone(), dev)
        solver.prepare_mu_lam(model, state, dev)
        v0 = torch.zeros(n, 3, device=dev); C0 = torch.zeros(n, 3, 3, device=dev)
        state.continue_from_torch(snap_x.clone(), v0, V0, C0, device=dev, requires_grad=False)
        prev = state
        traj = [wp.to_torch(prev.particle_x).clone()]
        for _ in range(cfg.num_frames - 1):
            for _ in range(sim.substep):
                nxt = prev.partial_clone(requires_grad=False)
                solver.p2g2p_differentiable(model, prev, nxt, sim.substep_size, device=dev)
                prev = nxt
            traj.append(wp.to_torch(prev.particle_x).clone())
    traj = torch.stack(traj).cpu().numpy()                       # [T,n,3] normalized
    motion = float(np.linalg.norm(traj[-1] - traj[0], axis=-1)[free.cpu().numpy()].mean())

    np.save(os.path.join(out_dir, "warp_traj.npy"), traj.astype(np.float32))
    np.save(os.path.join(out_dir, "f0_field.npy"), V0.cpu().numpy().astype(np.float32))
    np.save(os.path.join(out_dir, "init_xyz.npy"), snap_x.cpu().numpy().astype(np.float32))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"gt_logE": cfg.gt_logE, "nu": cfg.nu, "num_frames": cfg.num_frames,
                   "n": n, "v0_maxdev": maxdev, "free_mean_motion": motion,
                   "snapshot": cfg.snapshot}, f, indent=2)
    print(f"[dump] free mean motion {motion:.4f}; wrote {out_dir} ({time.time()-t0:.1f}s)")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(F0ReleaseDumpConfig))
