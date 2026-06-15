"""Entrypoint: dump a TELEPHONE bend GT as a cross-sim bundle, WITH the grip phase.

Unlike f0_dump_gt (block, release-only dump), this loads the real telephone particles,
grips the dangling tail and bends it sideways (velocity BC, top held), then releases.
BOTH phases are dumped with per-frame stress so the deformation is fully visible, not
just the release:

  GRIP  : top anchor v=0, tail region pulled laterally for grip_frames
          -> pull_X / pull_S (stretch) / pull_F recorded, F0 snapshot
  RELEASE: continue_from (x_snap, F_snap), top still held -> traj + per-frame stretch

Outputs (outputs/explore/f0_dump_telephone/<label>/):
  scene_cache.pt / init_xyz.npy (=x_snap) / f0.npy (=V0) / traj.npy   (the recovery bundle)
  grip_panel.png  grip_triplane.gif        (the GRIP phase + stress)
  release_panel.png  release_triplane.gif  (the RELEASE + stress)
  meta.json

Usage (physdreamer env):
  python -m reuse_mpm.explore.f0_dump_telephone --cache <telephone...pt> --grip-speed 0.5 \
      --grip-frames 8 --gt-logE 5.0 --K 48 --label telephone_bend_E5
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import tyro

from ..gpu import pick_free_gpu


def _anchor_marked_gif(path, X_seq, S_seq, anchor_bool, axis, title, fps=5):
    """Side-view (bend-axis vs z) animation marking the FROZEN anchor in red.

    X_seq (T,n,3), S_seq (T,n) stress, anchor_bool (n,).  Shows the held top staying put
    while the free part bends.  Free pts coloured by stress; anchor pts big red."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np, imageio
    a = ~anchor_bool
    xlo, xhi = X_seq[:, :, axis].min(), X_seq[:, :, axis].max()
    zlo, zhi = X_seq[:, :, 2].min(), X_seq[:, :, 2].max()
    smax = float(max(S_seq.max(), 1e-6))
    frames = []
    for f in range(X_seq.shape[0]):
        fig, ax = plt.subplots(figsize=(4, 5))
        ax.scatter(X_seq[f, a, axis], X_seq[f, a, 2], c=S_seq[f, a], cmap="viridis",
                   vmin=0, vmax=smax, s=6)
        ax.scatter(X_seq[f, anchor_bool, axis], X_seq[f, anchor_bool, 2], c="red", s=14,
                   label="FROZEN anchor")
        ax.set_xlim(xlo - 0.02, xhi + 0.02); ax.set_ylim(zlo - 0.02, zhi + 0.02)
        ax.set_xlabel("xyz"[axis]); ax.set_ylabel("z"); ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=7); ax.set_title(f"{title} f{f}/{X_seq.shape[0]-1}", fontsize=9)
        fig.tight_layout(); fig.canvas.draw()
        frames.append(np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (3,)))
        plt.close(fig)
    imageio.mimsave(path, frames, fps=fps)


@dataclass
class TelDumpConfig:
    cache: str = "/tmp2/b10401006/ev-project/generative-phys/outputs/_scene_cache/telephone_ds0.1_g32_k8.pt"
    rot_z_deg: float = -22.4
    grip_speed: float = 1.2          # lateral (x) tail velocity during grip (exaggerated bend)
    grip_frames: int = 12
    grip_axis: int = 0               # lateral bend direction: 0=x (thin, compliant)
    tail_frac: float = 0.20          # bottom fraction of z gripped as the tail
    anchor_top_frac: float = 0.6     # FREEZE the top 0.6 of the length (z >= zmax - 0.6*L)
    gt_logE: float = 5.0
    nu: float = 0.3
    K: int = 48
    min_quota_hours: float = 4.0
    label: str = "telephone_bend_E5"


def run(cfg: TelDumpConfig) -> str:
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

    out_dir = os.path.join("outputs", "explore", "f0_dump_telephone", cfg.label)
    os.makedirs(out_dir, exist_ok=True)

    # ---- telephone particles (rest) from the gic cache ----
    cache = torch.load(cfg.cache, map_location="cpu", weights_only=False)
    sim_xyz = cache["disc"]["sim_xyzs"].float()
    ghost = (sim_xyz == 0).all(dim=1)
    X = sim_xyz[~ghost]
    pvol = torch.from_numpy(cache["disc"]["points_vol"]).float()[~ghost]
    hang_mask = cache["disc"]["freeze_mask"][~ghost].bool()    # original top-hang anchor (release only)
    if cfg.rot_z_deg:
        th = math.radians(cfg.rot_z_deg); c, s = math.cos(th), math.sin(th)
        R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        X = (X - 0.5) @ R.T + 0.5
    dev = "cuda:0"
    X = X.to(dev); pvol = pvol.to(dev)

    # ---- geometry -> top-anchor + tail-grip cuboids (z is the long hanging axis) ----
    z = X[:, 2]; zmin, zmax = float(z.min()), float(z.max()); L = zmax - zmin
    xc, yc = float(X[:, 0].mean()), float(X[:, 1].mean())
    xh = float(X[:, 0].max() - X[:, 0].min()) / 2 + 1e-3
    yh = float(X[:, 1].max() - X[:, 1].min()) / 2 + 1e-3
    z_anchor_lo = zmax - cfg.anchor_top_frac * L     # freeze top 0.6 of the length
    z_tail_hi = zmin + cfg.tail_frac * L
    anchor_point = (xc, yc, 0.5 * (z_anchor_lo + zmax))
    anchor_size = (xh, yh, 0.5 * (zmax - z_anchor_lo) + 1e-3)
    grip_point = (xc, yc, 0.5 * (zmin + z_tail_hi))
    grip_size = (xh, yh, 0.5 * (z_tail_hi - zmin) + 1e-3)
    grip_vel = [0.0, 0.0, 0.0]; grip_vel[cfg.grip_axis] = cfg.grip_speed
    n_anchor = int((z >= z_anchor_lo).sum()); n_tail = int((z <= z_tail_hi).sum())
    print(f"[tel-dump] N={X.shape[0]} z[{zmin:.3f},{zmax:.3f}] anchor(top)={n_anchor} tail(grip)={n_tail} "
          f"grip_vel={grip_vel}")

    # ---- build telbend scene: grip + release ----
    sc = Scene("telbend", nu=cfg.nu, gt_logE=cfg.gt_logE, device=dev,
               X_rest_ext=X, p_vol_ext=pvol,
               grip_point=grip_point, grip_size=grip_size, grip_vel=grip_vel, grip_frames=cfg.grip_frames,
               anchor_point=anchor_point, anchor_size=anchor_size, release_anchor_mask=hang_mask)
    print(f"[tel-dump] grip-hold top60%={n_anchor}; release hang anchor={int(hang_mask.sum())}")
    print(f"[tel-dump] grip done: maxdev={sc.maxdev:.4f} tail|x_snap-X|max="
          f"{float((sc.x_snap - sc.X_rest).norm(dim=1).max()):.4f}")

    traj, stretch = sc.rollout(cfg.gt_logE, cfg.K)
    traj_np = traj.cpu().numpy().astype(np.float32)          # (K+1, n, 3)
    rel_motion = float(np.linalg.norm(traj_np[-1] - traj_np[0], axis=-1).mean())
    grip_X = torch.stack(sc.pull_X).cpu().numpy().astype(np.float32)    # (grip_frames+1, n, 3)
    grip_S = torch.stack(sc.pull_S).cpu().numpy().astype(np.float32)    # (grip_frames+1, n)
    print(f"[tel-dump] release free_mean_motion={rel_motion:.4f}; "
          f"grip frames={grip_X.shape[0]} release frames={traj_np.shape[0]}")

    # ---- V0 = sqrt(F F^T) of the gripped snapshot (gauge-free GT pre-stress) ----
    F = sc.F_snap
    ev, Q = torch.linalg.eigh(F @ F.transpose(-1, -2))
    V0 = (Q * ev.clamp_min(1e-9).sqrt().unsqueeze(-2)) @ Q.transpose(-1, -2)

    # ---- recovery bundle (mirror f0_dump_gt) ----
    keep = (~ghost)
    disc = {"sim_xyzs": cache["disc"]["sim_xyzs"], "freeze_mask": cache["disc"].get("freeze_mask"),
            "points_vol": cache["disc"]["points_vol"], "scale": 1.0, "shift": [0.0, 0.0, 0.0]}
    torch.save({"disc": disc}, os.path.join(out_dir, "scene_cache.pt"))
    # re-embed live-particle arrays into the full (N,...) ghost layout for index consistency
    N = ghost.shape[0]
    def embed(a):
        full = np.zeros((a.shape[0], N) + a.shape[2:], np.float32); full[:, keep.numpy()] = a; return full
    np.save(os.path.join(out_dir, "traj.npy"), embed(traj_np))
    x_snap_full = np.zeros((N, 3), np.float32); x_snap_full[keep.numpy()] = sc.x_snap.cpu().numpy()
    np.save(os.path.join(out_dir, "init_xyz.npy"), x_snap_full)
    f0_full = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1)); f0_full[keep.numpy()] = V0.cpu().numpy().astype(np.float32)
    np.save(os.path.join(out_dir, "f0.npy"), f0_full)
    meta = {"scene": "telephone_bend", "gt_logE": cfg.gt_logE, "nu": cfg.nu, "K": cfg.K, "n": int(X.shape[0]),
            "grip_speed": cfg.grip_speed, "grip_frames": cfg.grip_frames, "grip_axis": cfg.grip_axis,
            "rot_z_deg": cfg.rot_z_deg, "gravity": [0.0, 0.0, 0.0], "bc": {}, "maxdev": sc.maxdev,
            "anchor_top_frac": cfg.anchor_top_frac, "z_anchor_lo": z_anchor_lo,
            "n_anchor": n_anchor, "n_tail": n_tail,
            "note": "WARP telbend GT: tail gripped+bent (top held), then released; grip phase dumped too."}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)

    # ---- viz: GRIP and RELEASE each with per-frame stress ----
    gsel = list(range(grip_X.shape[0]))
    _viz.frames_panel(os.path.join(out_dir, "grip_panel.png"), grip_X, grip_S, sel=gsel, floor_z=None,
                      width=grip_X[:, :, cfg.grip_axis].max(1) - grip_X[:, :, cfg.grip_axis].min(1),
                      suptitle=f"GRIP phase (telephone bend, {cfg.grip_frames}f, speed {cfg.grip_speed})")
    _viz.triplane_scalar_gif(os.path.join(out_dir, "grip_triplane.gif"), grip_X, grip_S, floor_z=None, fps=4,
                             title_fn=lambda f: f"GRIP f{f}/{grip_X.shape[0]-1}")
    rsel = list(range(0, traj_np.shape[0], max(1, traj_np.shape[0] // 11)))[:12]
    _viz.frames_panel(os.path.join(out_dir, "release_panel.png"), traj_np, stretch.cpu().numpy(), sel=rsel,
                      floor_z=None, width=traj_np[:, :, 0].max(1) - traj_np[:, :, 0].min(1),
                      suptitle=f"RELEASE (telephone bend, gt_logE {cfg.gt_logE}, maxdev {sc.maxdev:.3f})")
    _viz.triplane_scalar_gif(os.path.join(out_dir, "release_triplane.gif"), traj_np, stretch.cpu().numpy(),
                             floor_z=None, fps=6, title_fn=lambda f: f"RELEASE f{f}/{traj_np.shape[0]-1}")
    # anchor-marked side views (the held top in RED) -- so the fixed region is visible
    grip_anchor_bool = (X[:, 2] >= z_anchor_lo).cpu().numpy()        # grip: top-60% held
    hang_bool = hang_mask.cpu().numpy()                              # release: only original hang
    _anchor_marked_gif(os.path.join(out_dir, "grip_anchored.gif"), grip_X, grip_S, grip_anchor_bool,
                       cfg.grip_axis, "GRIP (red=held top60%)", fps=4)
    _anchor_marked_gif(os.path.join(out_dir, "release_anchored.gif"), traj_np, stretch.cpu().numpy(),
                       hang_bool, cfg.grip_axis, "RELEASE (red=hang anchor only)", fps=6)
    print(f"[tel-dump] DONE -> {out_dir}\n  grip_panel.png grip_triplane.gif release_panel.png "
          f"release_triplane.gif grip_anchored.gif release_anchored.gif + bundle")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(TelDumpConfig))
