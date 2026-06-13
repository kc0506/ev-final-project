"""grad-u rung, first look: hand-construct a KNOWN non-uniform displacement field
u(x) -> compatible F0 = I + grad u, set positions x0 = X_rest + u, and SEE the
release rollout. Unlike the dynamic-pull F0 (compatible but with no closed-form u to
compare against), here u is known analytically -> later a fitted u-field has a
ground truth. The uniform/global-S case is the affine special case (grad u const);
this is the non-uniform step-up.

Default field: a half-sine y-bend along x, u_y = A*sin(pi*xi), xi=(x-xmin)/Lx.
Then d(u_y)/dx = A*(pi/Lx)*cos(pi*xi) -> F0[1,0] varies with x (non-uniform), rest = I.

Forward-only. Output (outputs/explore/f0_gradu_viz/<label>/): bend_panel.png,
traj_3d_triplane.gif, deflection.png, traj.npz
"""
from __future__ import annotations

import os
import time as _time
from dataclasses import dataclass
from typing import Tuple

import tyro

from ..gpu import pick_free_gpu


@dataclass
class GradUVizConfig:
    A: float = 0.05            # bend amplitude (y displacement, half-sine along x)
    nx: int = 22
    ny: int = 9
    nz: int = 16
    half: Tuple[float, float, float] = (0.18, 0.08, 0.14)
    z_base: float = 0.30
    nu: float = 0.3
    gt_logE: float = 4.5
    K: int = 24
    min_quota_hours: float = 8.0
    label: str = "ybend_halfsine"


def run(cfg: GradUVizConfig) -> str:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        from ..gpu import assert_gpu_quota
        assert_gpu_quota(cfg.min_quota_hours)
    else:
        pick_free_gpu(min_quota_hours=cfg.min_quota_hours)
    import numpy as np
    import torch
    import warp as wp
    wp.init()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    from ._block import Scene

    t0 = _time.time()
    out_dir = os.path.join("outputs", "explore", "f0_gradu_viz", cfg.label)
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda:0"
    # Scene("uniform", S_gt=0) just gives us a g=0 / no-floor release solver + geometry.
    sc = Scene("uniform", nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, half=cfg.half, z_base=cfg.z_base,
               nu=cfg.nu, gt_logE=cfg.gt_logE, S_gt=(0.0,) * 6, device=dev)
    X = sc.X_rest; n = sc.n
    xmin = float(X[:, 0].min()); xmax = float(X[:, 0].max()); Lx = xmax - xmin
    xi = (X[:, 0] - xmin) / Lx                                   # in [0,1] along x
    u = torch.zeros(n, 3, device=dev)
    u[:, 1] = cfg.A * torch.sin(np.pi * xi)                      # y bend, half-sine
    dudx = cfg.A * (np.pi / Lx) * torch.cos(np.pi * xi)          # d(u_y)/dx, per particle
    x0 = (X + u).contiguous()
    F0 = torch.eye(3, device=dev)[None].repeat(n, 1, 1).clone()
    F0[:, 1, 0] = dudx                                           # F = I + grad u (only u_y,x nonzero)
    print(f"[gradu] y-bend A={cfg.A}  max|u_y|={float(u[:,1].abs().max()):.4f}  "
          f"F0[1,0] range [{float(dudx.min()):+.3f},{float(dudx.max()):+.3f}]")

    traj = sc.rollout_F0(x0, F0, cfg.gt_logE, cfg.K).cpu().numpy()   # [K+1,n,3]
    Xr = X.cpu().numpy()
    disp = np.linalg.norm(traj - Xr[None], axis=2)                  # [K+1,n] displacement-from-rest
    ydef = traj[:, :, 1] - Xr[None, :, 1]                           # y deflection
    maxy = np.abs(ydef).max(1)
    np.savez(os.path.join(out_dir, "traj.npz"), traj=traj, X_rest=Xr, disp=disp, ydef=ydef, A=cfg.A)
    print(f"[gradu] y-deflection over release: {' '.join(f'{v:.3f}' for v in maxy[::3])}")

    vmax = float(np.quantile(disp, 0.98)) or 1e-3
    # ---- bend panel: first 8 frames, xy view (the bend plane), colored by displacement ----
    mn = traj[:, :, :2].reshape(-1, 2).min(0); mx = traj[:, :, :2].reshape(-1, 2).max(0)
    fig, axs = plt.subplots(2, 4, figsize=(16, 7))
    for f, ax in enumerate(axs.flat):
        if f >= min(8, cfg.K + 1):
            ax.axis("off"); continue
        psc = ax.scatter(traj[f][:, 0], traj[f][:, 1], c=disp[f], s=7, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_aspect("equal")
        ax.set_title(f"frame {f}  max|Δy| {maxy[f]:.3f}"); ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(psc, ax=axs, shrink=0.6, label="|x - x_rest|")
    fig.suptitle(f"grad-u y-bend (A={cfg.A}) release, xy view (GT logE {cfg.gt_logE})", fontsize=13)
    fig.savefig(os.path.join(out_dir, "bend_panel.png"), dpi=110); plt.close(fig)

    # ---- y-deflection over time (the bend un-bending + oscillating) ----
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(maxy, "-o", ms=3)
    ax.set_xlabel("release frame"); ax.set_ylabel("max |Δy| (bend)")
    ax.set_title("bend un-bends then oscillates (release)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "deflection.png"), dpi=120); plt.close(fig)

    # ---- 3d + triplane gif ----
    mins = traj.reshape(-1, 3).min(0); maxs = traj.reshape(-1, 3).max(0)
    fig = plt.figure(figsize=(11, 9)); ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax2 = [fig.add_subplot(2, 2, k) for k in (2, 3, 4)]; proj = [(0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z")]

    def draw(f):
        ax3d.cla()
        ax3d.scatter(traj[f][:, 0], traj[f][:, 1], traj[f][:, 2], c=disp[f], s=4, cmap="viridis", vmin=0, vmax=vmax)
        ax3d.set_xlim(mins[0], maxs[0]); ax3d.set_ylim(mins[1], maxs[1]); ax3d.set_zlim(mins[2], maxs[2])
        ax3d.set_title(f"frame {f}/{cfg.K}")
        for axp, (a, b, la, lb) in zip(ax2, proj):
            axp.cla(); axp.scatter(traj[f][:, a], traj[f][:, b], c=disp[f], s=5, cmap="viridis", vmin=0, vmax=vmax)
            axp.set_xlim(mins[a], maxs[a]); axp.set_ylim(mins[b], maxs[b]); axp.set_aspect("equal")
            axp.set_xlabel(la); axp.set_ylabel(lb); axp.set_title(f"{la}{lb}")
        return ()

    draw(0); anim = FuncAnimation(fig, draw, frames=cfg.K + 1, blit=False)
    anim.save(os.path.join(out_dir, "traj_3d_triplane.gif"), writer=PillowWriter(fps=6)); plt.close(fig)
    print(f"[gradu] -> {out_dir} ({_time.time()-t0:.1f}s)")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(GradUVizConfig))
