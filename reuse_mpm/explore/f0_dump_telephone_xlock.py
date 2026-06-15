"""Telephone bend GT with POSITION-LOCK grip (telxlock) -- the stress-propagation fix.

Last-shot test of the hypothesis: the big top-60% grid-freeze of telbend ZEROED grid
velocity -> cut stress -> the upper cord stored no pre-stress (left branch had no tension).
Here the held point is the TINY original hang, and it is held by overwriting its x every
substep (NOT a velocity/grid BC) so v and F (stress) propagate THROUGH it -- the real-grip
spirit.  Tail traces the same lateral->up arc; release keeps only the hang (still x-locked).

Usage (physdreamer env):
  python -m reuse_mpm.explore.f0_dump_telephone_xlock --grip-speed 0.6 --grip-frames 12 \
      --gt-logE 5.0 --K 48 --label telephone_xlock_E5
"""
from __future__ import annotations
import json, os
from dataclasses import dataclass
import tyro
from ..gpu import pick_free_gpu
from .f0_dump_telephone import _anchor_marked_gif


@dataclass
class TelXlockConfig:
    cache: str = "/tmp2/b10401006/ev-project/generative-phys/outputs/_scene_cache/telephone_ds0.1_g32_k8.pt"
    rot_z_deg: float = -22.4
    grip_speed: float = 0.6
    grip_frames: int = 12
    grip_axis: int = 0
    tail_frac: float = 0.20
    gt_logE: float = 5.0
    nu: float = 0.3
    K: int = 48
    min_quota_hours: float = 4.0
    label: str = "telephone_xlock_E5"


def run(cfg: TelXlockConfig) -> str:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        from ..gpu import assert_gpu_quota
        assert_gpu_quota(cfg.min_quota_hours)
    else:
        pick_free_gpu(min_quota_hours=cfg.min_quota_hours)
    import math
    import numpy as np
    import torch
    import warp as wp
    wp.init()
    from ._block import Scene
    from . import _viz

    out_dir = os.path.join("outputs", "explore", "f0_dump_telephone_xlock", cfg.label)
    os.makedirs(out_dir, exist_ok=True)

    cache = torch.load(cfg.cache, map_location="cpu", weights_only=False)
    sim_xyz = cache["disc"]["sim_xyzs"].float(); ghost = (sim_xyz == 0).all(dim=1)
    X = sim_xyz[~ghost]; pvol = torch.from_numpy(cache["disc"]["points_vol"]).float()[~ghost]
    hang_mask = cache["disc"]["freeze_mask"][~ghost].bool()
    if cfg.rot_z_deg:
        th = math.radians(cfg.rot_z_deg); c, s = math.cos(th), math.sin(th)
        R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]); X = (X - 0.5) @ R.T + 0.5
    dev = "cuda:0"; X = X.to(dev); pvol = pvol.to(dev)
    z = X[:, 2]; zmin, zmax = float(z.min()), float(z.max()); L = zmax - zmin
    xc, yc = float(X[:, 0].mean()), float(X[:, 1].mean())
    xh = float(X[:, 0].max() - X[:, 0].min()) / 2 + 1e-3; yh = float(X[:, 1].max() - X[:, 1].min()) / 2 + 1e-3
    z_tail_hi = zmin + cfg.tail_frac * L
    grip_point = (xc, yc, 0.5 * (zmin + z_tail_hi)); grip_size = (xh, yh, 0.5 * (z_tail_hi - zmin) + 1e-3)
    grip_vel = [0.0, 0.0, 0.0]; grip_vel[cfg.grip_axis] = cfg.grip_speed
    print(f"[tel-xlock] N={X.shape[0]} hang(x-locked)={int(hang_mask.sum())} tail(grip)={int((z<=z_tail_hi).sum())} "
          f"grip_vel={grip_vel}")

    sc = Scene("telxlock", nu=cfg.nu, gt_logE=cfg.gt_logE, device=dev, X_rest_ext=X, p_vol_ext=pvol,
               grip_point=grip_point, grip_size=grip_size, grip_vel=grip_vel, grip_frames=cfg.grip_frames,
               release_anchor_mask=hang_mask)
    print(f"[tel-xlock] grip done: maxdev={sc.maxdev:.4f}")

    traj, stretch = sc.rollout(cfg.gt_logE, cfg.K)
    traj_np = traj.cpu().numpy().astype(np.float32)
    rel_motion = float(np.linalg.norm(traj_np[-1] - traj_np[0], axis=-1).mean())
    grip_X = torch.stack(sc.pull_X).cpu().numpy().astype(np.float32)
    grip_S = torch.stack(sc.pull_S).cpu().numpy().astype(np.float32)
    # stress-propagation check: does the UPPER cord (between hang and tail) carry stress?
    znp = X[:, 2].cpu().numpy(); upper = (znp > z_tail_hi) & (~hang_mask.cpu().numpy())
    snap_S = sc.F0_stretch
    print(f"[tel-xlock] release free_mean_motion={rel_motion:.4f}; upper-cord mean |stretch| at snapshot "
          f"{float(snap_S[upper].mean()):.4f} (telbend had ~0 here)")

    F = sc.F_snap; ev, Q = torch.linalg.eigh(F @ F.transpose(-1, -2))
    V0 = (Q * ev.clamp_min(1e-9).sqrt().unsqueeze(-2)) @ Q.transpose(-1, -2)
    keep = (~ghost).numpy(); N = ghost.shape[0]
    torch.save({"disc": {"sim_xyzs": cache["disc"]["sim_xyzs"], "freeze_mask": cache["disc"].get("freeze_mask"),
                "points_vol": cache["disc"]["points_vol"], "scale": 1.0, "shift": [0.0, 0.0, 0.0]}},
               os.path.join(out_dir, "scene_cache.pt"))
    def embed(a):
        full = np.zeros((a.shape[0], N) + a.shape[2:], np.float32); full[:, keep] = a; return full
    np.save(os.path.join(out_dir, "traj.npy"), embed(traj_np))
    x_snap_full = np.zeros((N, 3), np.float32); x_snap_full[keep] = sc.x_snap.cpu().numpy()
    np.save(os.path.join(out_dir, "init_xyz.npy"), x_snap_full)
    f0_full = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1)); f0_full[keep] = V0.cpu().numpy().astype(np.float32)
    np.save(os.path.join(out_dir, "f0.npy"), f0_full)
    json.dump({"scene": "telephone_xlock", "gt_logE": cfg.gt_logE, "nu": cfg.nu, "K": cfg.K, "n": int(X.shape[0]),
               "grip_speed": cfg.grip_speed, "grip_frames": cfg.grip_frames, "rot_z_deg": cfg.rot_z_deg,
               "gravity": [0.0, 0.0, 0.0], "bc": {}, "substep": 64, "maxdev": sc.maxdev, "n_hang": int(hang_mask.sum()),
               "note": "WARP telxlock GT: tail grip + tiny hang held by x-lock (stress propagates)."},
              open(os.path.join(out_dir, "meta.json"), "w"), indent=2)

    gsel = list(range(grip_X.shape[0]))
    _viz.frames_panel(os.path.join(out_dir, "grip_panel.png"), grip_X, grip_S, sel=gsel, floor_z=None,
                      width=grip_X[:, :, 0].max(1) - grip_X[:, :, 0].min(1), suptitle="GRIP (telxlock, hang x-locked)")
    _viz.triplane_scalar_gif(os.path.join(out_dir, "grip_triplane.gif"), grip_X, grip_S, floor_z=None, fps=4,
                             title_fn=lambda f: f"GRIP f{f}/{grip_X.shape[0]-1}")
    rsel = list(range(0, traj_np.shape[0], max(1, traj_np.shape[0] // 11)))[:12]
    _viz.frames_panel(os.path.join(out_dir, "release_panel.png"), traj_np, stretch.cpu().numpy(), sel=rsel,
                      floor_z=None, width=traj_np[:, :, 0].max(1) - traj_np[:, :, 0].min(1),
                      suptitle=f"RELEASE (telxlock, gt_logE {cfg.gt_logE}, maxdev {sc.maxdev:.3f})")
    _viz.triplane_scalar_gif(os.path.join(out_dir, "release_triplane.gif"), traj_np, stretch.cpu().numpy(),
                             floor_z=None, fps=6, title_fn=lambda f: f"RELEASE f{f}/{traj_np.shape[0]-1}")
    hb = hang_mask.cpu().numpy()
    _anchor_marked_gif(os.path.join(out_dir, "grip_anchored.gif"), grip_X, grip_S, hb, 0, "GRIP (red=hang x-lock)", fps=4)
    _anchor_marked_gif(os.path.join(out_dir, "release_anchored.gif"), traj_np, stretch.cpu().numpy(), hb, 0,
                       "RELEASE (red=hang x-lock)", fps=6)
    print(f"[tel-xlock] DONE -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(TelXlockConfig))
