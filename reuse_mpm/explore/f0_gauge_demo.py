"""Visual demo of the F0 (left-stretch) gauge in pre-stress sys-id.

The SAME observed config x0 can be explained by TWO different pre-stresses that
differ by a global right-rotation F0 -> F0 R0 (equivalently: a rigidly ROTATED
rest frame X' = c + R0^T (X-c)). Both release with BIT-IDENTICAL dynamics, because
the hyperelastic stress depends only on the left stretch V0 = sqrt(F0 F0^T) (the
singular values), which is invariant under right-multiplication by an orthogonal
matrix. So the trajectory constrains V0 only; rest/displacement are recoverable
just up to a global rigid motion. See reports/gauge_math.md.

Output: outputs/explore/f0_gauge_demo/<label>/gauge_demo.png
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import tyro

from ..gpu import pick_free_gpu


@dataclass
class GaugeDemoConfig:
    rot_z_deg: float = 40.0
    gradu_A: float = 0.05
    gt_logE: float = 4.5
    nu: float = 0.3
    K: int = 24
    min_quota_hours: float = 0.0
    label: str = "ybend_rot40"


def run(cfg: GaugeDemoConfig) -> str:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        from ..gpu import assert_gpu_quota
        assert_gpu_quota(cfg.min_quota_hours)
    else:
        pick_free_gpu(min_quota_hours=cfg.min_quota_hours)
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import warp as wp
    wp.init()
    from ._block import Scene

    out_dir = os.path.join("outputs", "explore", "f0_gauge_demo", cfg.label)
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda:0"

    sc = Scene("gradu", nu=cfg.nu, gt_logE=cfg.gt_logE, gradu_A=cfg.gradu_A, device=dev)
    X = sc.X_rest.clone()                       # rest A (flat)
    x0 = sc.x_snap.clone()                      # observed (shared)
    F0 = sc.F_snap.clone()                      # pre-stress A = I + grad u

    th = math.radians(cfg.rot_z_deg); c, s = math.cos(th), math.sin(th)
    R0 = torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], device=dev, dtype=F0.dtype)
    cen = X.mean(0)
    Xp = cen + (X - cen) @ R0.T                 # rest B = rigidly rotated rest (X' = c + R0^T (X-c))
    F0p = F0 @ R0                               # pre-stress B = F0 R0  (=> dx0/dX' = F0 R0)
    u = (x0 - X).cpu().numpy()                  # displacement A: flat rest -> x0
    up = (x0 - Xp).cpu().numpy()                # displacement B: rotated rest -> x0

    traj = sc.rollout_F0(x0, F0, cfg.gt_logE, cfg.K).cpu().numpy()    # (K+1,n,3)
    trajp = sc.rollout_F0(x0, F0p, cfg.gt_logE, cfg.K).cpu().numpy()
    V0, V0p = torch.linalg.svdvals(F0), torch.linalg.svdvals(F0p)
    dF = float((F0p - F0).abs().max()); dV = float((V0p - V0).abs().max()); dT = float(np.abs(trajp - traj).max())
    print(f"[gauge] rot {cfg.rot_z_deg}deg: max|F0'-F0|={dF:.3f} (diff IC)  "
          f"max|svd diff|={dV:.2e} (same V0)  max|traj'-traj|={dT:.2e} (same dynamics; "
          f"motion scale {float(np.abs(traj-traj[0:1]).max()):.4f})")

    Xn, Xpn, x0n = X.cpu().numpy(), Xp.cpu().numpy(), x0.cpu().numpy()
    sub = np.random.default_rng(0).permutation(sc.n)[:450]
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))

    # A/B: the two rests, both -> same x0
    for ax, (Xc, lab, col) in zip([axs[0, 0], axs[0, 1]],
                                  [(Xn, "rest A (flat)", "tab:gray"), (Xpn, f"rest B (rot {cfg.rot_z_deg}deg)", "tab:green")]):
        ax.scatter(Xc[sub, 0], Xc[sub, 1], s=6, c=col, alpha=0.5, label=lab)
        ax.scatter(x0n[sub, 0], x0n[sub, 1], s=6, c="tab:red", alpha=0.5, label="x0 (observed)")
        ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=8)
        ax.set_title(f"{lab}  ->  same x0")

    # C: trajectory overlay at last frame
    axc = axs[0, 2]
    axc.scatter(traj[-1, :, 0], traj[-1, :, 1], s=5, c="tab:blue", label="release from F0")
    axc.scatter(trajp[-1, :, 0], trajp[-1, :, 1], s=5, c="tab:red", alpha=0.5, label="release from F0 R0")
    axc.set_aspect("equal"); axc.set_xlabel("x"); axc.set_ylabel("y"); axc.legend(fontsize=8)
    axc.set_title(f"release frame {cfg.K}: trajectories OVERLAP")

    # D/E: the two displacement fields (different), tails at their respective rests
    for ax, (Xc, uc, lab) in zip([axs[1, 0], axs[1, 1]],
                                 [(Xn, u, "u_A = x0 - restA"), (Xpn, up, "u_B = x0 - restB")]):
        q = ax.quiver(Xc[sub, 0], Xc[sub, 1], uc[sub, 0], uc[sub, 1], np.linalg.norm(uc[sub], axis=1),
                      cmap="viridis", angles="xy", scale_units="xy", scale=1.0, width=0.004)
        ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(f"{lab} (DIFFERENT field)")

    # F: V0 identical + per-frame traj diff
    axf = axs[1, 2]
    axf.scatter(V0.flatten().cpu().numpy(), V0p.flatten().cpu().numpy(), s=5)
    lo, hi = float(V0.min()), float(V0.max())
    axf.plot([lo, hi], [lo, hi], "k--", lw=1)
    axf.set_xlabel("singular values of F0"); axf.set_ylabel("of F0 R0")
    axf.set_title(f"V0 IDENTICAL (max diff {dV:.1e})\nmax|traj'-traj|={dT:.1e} over {cfg.K} frames")

    fig.suptitle(f"F0 gauge: two rests / pre-stresses (max|F0'-F0|={dF:.2f}), SAME V0, IDENTICAL dynamics",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "gauge_demo.png"), dpi=130); plt.close(fig)
    print(f"[gauge] -> {out_dir}/gauge_demo.png")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(GaugeDemoConfig))
