"""Entrypoint: dump a block scene's GT as a canonical cross-sim bundle for gic.

The forcing function for a clean warp->gic handoff: this is the PRODUCER (forward
only), gic's roundtrip is the CONSUMER (fit). One fixed bundle format decouples them
-- no forward/backward mixing. Mirrors what f0_release_dump did for telephone, but
config-driven over _block.SCENES (release/drop/squeeze/...) and emitting the bundle
gic's load_our_scene already reads.

From a scene's snapshot (x_snap, F_snap): strip rotation to the left stretch
V0 = sqrt(F F^T) (the dynamics-identifiable, gic/warp-shared canonical F0 for an
isotropic material), then roll warp forward at gt_logE (v0=0 release). Writes, in
NORMALIZED sim coords ([0,1]^3, what gic --gt_traj_normalized expects):

  scene_cache.pt  {"disc": {sim_xyzs, freeze_mask, points_vol, scale, shift}}
  traj.npy        [T,n,3]  release rollout (frame0 = snapshot t0 = init_xyz)
  f0.npy          [n,3,3]  V0 to inject into gic's initial F (--f0_npy)
  init_xyz.npy    [n,3]    snapshot positions = the new t0 (--init_xyz_npy)
  meta.json       gt_logE, nu, K, scene, n, maxdev + gic-alignment hints
                  (material, gravity, bc/floor, dx, substep, density)
  gt_panel.png / gt_3d_triplane.gif   (eyeball the GT before shipping)

Output under outputs/explore/f0_dump_gt/<label>/.
"""
from __future__ import annotations

import os
import time as _time
from dataclasses import dataclass
from typing import Tuple

import tyro

from ..gpu import pick_free_gpu


@dataclass
class DumpGTConfig:
    scene: str = "release"            # release | drop | squeeze | freefall | uniform
    # geometry (shared with f0_fit_case / f0_forward_viz)
    nx: int = 22
    ny: int = 9
    nz: int = 16
    half: Tuple[float, float, float] = (0.18, 0.08, 0.14)
    z_base: float = 0.30
    # pull (release/drop/freefall F0)
    pull_speed: float = 0.5
    release_frame: int = 5
    grip_half_x: float = 0.045
    # squeeze
    push_x: float = 0.60
    push_half_x: float = 0.07
    push_half_z: float = 0.045
    push_speed: float = 0.45
    push_frames: int = 5
    # drop
    floor_z: float = 0.25
    gravity: float = 9.8
    collider: str = "slip"
    friction: float = 0.0
    # uniform
    S_gt: Tuple[float, float, float, float, float, float] = (0.2, -0.1, -0.1, 0.05, 0.0, 0.0)
    # gradu (analytic half-sine y-bend: F0 = I + grad u, u_y = A sin(pi xi))
    gradu_A: float = 0.05
    # physics / horizon
    gt_logE: float = 4.5
    nu: float = 0.3
    K: int = 32
    material: str = "jelly"           # warp constitutive model used for the GT (FCR)
    min_quota_hours: float = 8.0
    label: str = "release_E4p5"


def run(cfg: DumpGTConfig) -> str:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        from ..gpu import assert_gpu_quota
        assert_gpu_quota(cfg.min_quota_hours)
    else:
        pick_free_gpu(min_quota_hours=cfg.min_quota_hours)
    import json
    import numpy as np
    import torch
    import warp as wp
    wp.init()

    from ._block import Scene, SCENES
    from . import _viz
    from ..config import SimConfig

    t0 = _time.time()
    assert cfg.scene in SCENES, f"unknown scene {cfg.scene!r} (have {list(SCENES)})"
    out_dir = os.path.join("outputs", "explore", "f0_dump_gt", cfg.label)
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda:0"
    sim = SimConfig()

    sc = Scene(cfg.scene, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, half=cfg.half, z_base=cfg.z_base,
               nu=cfg.nu, gt_logE=cfg.gt_logE, pull_speed=cfg.pull_speed,
               release_frame=cfg.release_frame, grip_half_x=cfg.grip_half_x,
               push_x=cfg.push_x, push_half_x=cfg.push_half_x, push_half_z=cfg.push_half_z,
               push_speed=cfg.push_speed, push_frames=cfg.push_frames, gravity=cfg.gravity,
               floor_z=cfg.floor_z, collider=cfg.collider, friction=cfg.friction,
               S_gt=cfg.S_gt, gradu_A=cfg.gradu_A, device=dev)
    n = sc.n
    X_rest = sc.X_rest
    x_snap, F_snap = sc.x_snap, sc.F_snap

    # left stretch V0 = sqrt(F F^T) (rotation stripped; right-rotation is a dynamics
    # gauge for isotropic material -- same dynamics in warp FCR and gic).
    FFt = F_snap @ F_snap.transpose(-1, -2)
    mu, Q = torch.linalg.eigh(FFt)
    mu = mu.clamp_min(1e-9)
    V0 = (Q * mu.sqrt().unsqueeze(-2)) @ Q.transpose(-1, -2)        # [n,3,3]

    # GT release rollout at gt_logE (frame0 = x_snap = init_xyz)
    traj, stretch = sc.rollout(cfg.gt_logE, cfg.K)
    traj_np = traj.cpu().numpy().astype(np.float32)                # [K+1,n,3] normalized
    motion = float(np.linalg.norm(traj_np[-1] - traj_np[0], axis=-1).mean())
    print(f"[dump] scene={cfg.scene} n={n} maxdev={sc.maxdev:.4f} free_mean_motion={motion:.4f}")

    # ---- canonical bundle for gic ----
    p_vol = float((2 * cfg.half[0] / max(cfg.nx - 1, 1)) ** 3)
    disc = {
        "sim_xyzs": X_rest.cpu(),                                  # rest layout (init_vol)
        "freeze_mask": torch.zeros(n, dtype=torch.bool),           # free body, no anchors
        "points_vol": torch.full((n,), p_vol),
        "scale": 1.0, "shift": [0.0, 0.0, 0.0],                    # already in [0,1]^3
    }
    torch.save({"disc": disc}, os.path.join(out_dir, "scene_cache.pt"))
    np.save(os.path.join(out_dir, "traj.npy"), traj_np)
    np.save(os.path.join(out_dir, "f0.npy"), V0.cpu().numpy().astype(np.float32))
    np.save(os.path.join(out_dir, "init_xyz.npy"), x_snap.cpu().numpy().astype(np.float32))

    # gic-alignment hints: gravity/bc in gic's config.json convention (z-up; warp +z up).
    # release: no gravity, no floor. drop: -z gravity + slip floor. squeeze: slip floor only.
    grav = [0.0, 0.0, -cfg.gravity] if cfg.scene in ("drop", "freefall") else [0.0, 0.0, 0.0]
    bc = ({"ground": [[0.0, 0.0, sc.floor_z], [0.0, 0.0, 1.0], cfg.collider]}
          if sc.has_floor else {})
    meta = {
        "scene": cfg.scene, "gt_logE": cfg.gt_logE, "nu": cfg.nu, "K": cfg.K, "n": n,
        "maxdev": sc.maxdev, "free_mean_motion": motion,
        # what gic must match to kill the cross-sim floor:
        "material": cfg.material, "gravity": grav, "bc": bc,
        "dx": 1.0 / sim.grid_size, "grid_size": sim.grid_size,
        "substep": sim.substep, "substep_size": sim.substep_size, "density": sim.density,
        "note": "traj is normalized [0,1]^3; frame0 == init_xyz; f0 is V0=sqrt(FF^T).",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---- eyeball the GT before shipping (reuse _viz) ----
    fz = sc.floor_z if sc.has_floor else None
    width = traj_np[:, :, 0].max(1) - traj_np[:, :, 0].min(1)
    sel = list(range(0, traj_np.shape[0], max(1, traj_np.shape[0] // 11)))[:12]
    _viz.frames_panel(os.path.join(out_dir, "gt_panel.png"), traj_np, stretch.cpu().numpy(),
                      sel=sel, floor_z=fz, width=width,
                      suptitle=f"GT {cfg.scene} release (gt_logE {cfg.gt_logE}, maxdev {sc.maxdev:.3f})")
    _viz.triplane_scalar_gif(os.path.join(out_dir, "gt_3d_triplane.gif"), traj_np,
                             stretch.cpu().numpy(), floor_z=fz, fps=6,
                             title_fn=lambda f: f"GT {cfg.scene} f{f}/{traj_np.shape[0]-1}")

    print(f"[dump] -> {out_dir}  (n={n}, T={traj_np.shape[0]}, {_time.time()-t0:.1f}s)")
    print(f"[dump] gic CLI:\n"
          f"  --scene_cache {out_dir}/scene_cache.pt --gt_traj {out_dir}/traj.npy "
          f"--gt_traj_normalized \\\n  --f0_npy {out_dir}/f0.npy --init_xyz_npy {out_dir}/init_xyz.npy "
          f"--gt_logE {cfg.gt_logE} --gt_nu {cfg.nu}")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(DumpGTConfig))
