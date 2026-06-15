"""(ii)-b2 core-component VISUAL sanity check (before wiring the MLP fit).

Three things, each with a figure (not just numbers):

  1. w quiver       -- the displacement field w(x0). For the transverse y-bend
                       u_y=A sin(pi xi) (u_x=u_z=0) we have x0_x=X_x, so w==u in
                       form. Arrows show the bend (|u_y| max at center xi=0.5).
  2. F0 from w+autograd -- F0 = (I - grad_{x0} w)^{-1} via the SAME vmap+jacrev+inv
                       the MLP path will use, on the ANALYTIC w. Heatmaps of the
                       recovered shear F0[1,0]=g (max at the x-ends, cos profile),
                       det(I-grad w), the pre-stress |tau| (FCR), and the residual
                       |F0_autograd - F0_gt| (must be ~0: validates the b2 formula).
  3. gauge demo     -- release from (x0, F0_gt) vs (x0, F0_gt @ R0) for a global
                       rotation R0. Different F0 (heatmap), IDENTICAL stretch V0,
                       and bit-identical trajectory overlay -> the right-rotation
                       (rest-frame) gauge is REAL; the traj only sees V0.

Output: outputs/explore/f0_b2_sanity/<label>/.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import tyro

from ..gpu import pick_free_gpu


@dataclass
class B2SanityConfig:
    gradu_A: float = 0.05
    gt_logE: float = 4.5
    nu: float = 0.3
    K: int = 24
    rot_z_deg: float = 40.0      # gauge demo rotation
    min_quota_hours: float = 8.0
    label: str = "ybend"


def run(cfg: B2SanityConfig) -> str:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        from ..gpu import assert_gpu_quota
        assert_gpu_quota(cfg.min_quota_hours)
    else:
        pick_free_gpu(min_quota_hours=cfg.min_quota_hours)
    import numpy as np
    import torch
    from torch.func import jacrev, vmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import warp as wp
    wp.init()
    from ._block import Scene

    out_dir = os.path.join("outputs", "explore", "f0_b2_sanity", cfg.label)
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda:0"

    sc = Scene("gradu", nu=cfg.nu, gt_logE=cfg.gt_logE, gradu_A=cfg.gradu_A, device=dev)
    X = sc.X_rest                              # rest
    x0 = sc.x_snap.clone()                     # observed deformed t0
    F0_gt = sc.F_snap.clone()                  # I + grad u (the truth)
    n = sc.n
    xs = x0[:, 0].cpu().numpy(); zs = x0[:, 2].cpu().numpy(); ys = x0[:, 1].cpu().numpy()
    xmin = float(X[:, 0].min()); Lx = float(X[:, 0].max() - xmin)
    A, pi = cfg.gradu_A, math.pi

    # ---- (b2) core: w on OBSERVED x0, F0 = (I - grad_{x0} w)^{-1} via autograd ----
    # analytic w (== u here): transverse y-bend, depends on x0_x only
    def w_fn(p):                               # p:(3,) -> (3,)
        xi = (p[0] - xmin) / Lx
        return torch.stack([p[0] * 0.0, A * torch.sin(pi * xi), p[2] * 0.0])
    J = vmap(jacrev(w_fn))(x0)                  # (N,3,3) = grad_{x0} w
    eye = torch.eye(3, device=dev)
    detIminus = torch.linalg.det(eye - J)       # (N,)
    F0_ag = torch.linalg.inv(eye - J)           # (N,3,3) recovered F0
    resid = (F0_ag - F0_gt).abs().amax(dim=(1, 2))   # (N,) vs truth

    # pre-stress (FCR) tau magnitude at t0 from F0_ag, for the "tension" heatmap
    E = 10.0 ** cfg.gt_logE
    mu = E / (2 * (1 + cfg.nu)); lam = E * cfg.nu / ((1 + cfg.nu) * (1 - 2 * cfg.nu))
    U, S, Vh = torch.linalg.svd(F0_ag)
    R = U @ Vh
    Jdet = torch.linalg.det(F0_ag)
    tau = 2 * mu * (F0_ag - R) @ F0_ag.transpose(-1, -2) \
        + lam * (Jdet * (Jdet - 1.0))[:, None, None] * eye
    tau_mag = torch.linalg.matrix_norm(tau).cpu().numpy()

    g_ag = J[:, 1, 0].cpu().numpy()
    g_analytic = A * (pi / Lx) * np.cos(pi * (xs - xmin) / Lx)
    print(f"[b2-sanity] F0 recovered: max|F0_ag-F0_gt|={float(resid.max()):.2e} "
          f"(w+autograd reproduces GT); det(I-gradw) in [{float(detIminus.min()):.3f},"
          f"{float(detIminus.max()):.3f}]; corr(g_ag,analytic)="
          f"{np.corrcoef(g_ag,g_analytic)[0,1]:.4f}")

    # ===== FIG 1: w quiver -- tail at REST X (flat), head at x0 (arch) =====
    # w = x0 - X is the FORWARD displacement (rest->observed); drawing it from the
    # rest config makes "flat rest -> up-bulged x0" unambiguous (arrow base = rest).
    Xn = X.cpu().numpy()
    w = (x0 - X).cpu().numpy()                  # = u, points +y (max at center)
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = np.random.default_rng(0).permutation(n)[:500]
    ax.scatter(Xn[sub, 0], Xn[sub, 1], s=5, c="tab:gray", alpha=0.4, label="rest X (flat)")
    ax.scatter(xs[sub], ys[sub], s=5, c="tab:red", alpha=0.4, label="x0 (observed arch)")
    q = ax.quiver(Xn[sub, 0], Xn[sub, 1], w[sub, 0], w[sub, 1], np.linalg.norm(w[sub], axis=1),
                  cmap="viridis", angles="xy", scale_units="xy", scale=1.0, width=0.004)
    plt.colorbar(q, ax=ax, label="|w|")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal"); ax.legend(fontsize=8)
    ax.set_title(f"w = x0 - X (rest->observed), tail at REST (A={A})\n"
                 f"flat rest pushed UP to the arch; |w| max at center")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "1_w_quiver.png"), dpi=130); plt.close(fig)

    # ===== FIG 2: F0 from w+autograd (heatmaps over x-z) =====
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    def scat(ax, c, title, cmap="coolwarm"):
        sc_ = ax.scatter(xs, zs, c=c, s=8, cmap=cmap); plt.colorbar(sc_, ax=ax)
        ax.set_xlabel("x0_x"); ax.set_ylabel("x0_z"); ax.set_title(title)
    scat(axs[0, 0], g_ag, "recovered shear F0[1,0]=g (autograd)  -- cos profile, max at x-ends")
    axs[0, 1].scatter(xs, g_ag, s=6, label="autograd g")
    axs[0, 1].scatter(xs, g_analytic, s=6, alpha=0.5, label="analytic A(pi/Lx)cos(pi xi)")
    axs[0, 1].set_xlabel("x0_x"); axs[0, 1].set_ylabel("g"); axs[0, 1].legend()
    axs[0, 1].set_title("recovered vs analytic shear")
    scat(axs[1, 0], tau_mag, "initial pre-stress |tau| (FCR) -- the tension driving release", "magma")
    scat(axs[1, 1], resid.cpu().numpy(), "residual |F0_autograd - F0_gt| (must be ~0)", "viridis")
    fig.suptitle("(b2) core: F0 = (I - grad_x0 w)^-1 via vmap+jacrev+inv on analytic w")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "2_F0_from_w.png"), dpi=130); plt.close(fig)

    # ===== FIG 3: gauge demo (rotate rest -> same dynamics) =====
    th = math.radians(cfg.rot_z_deg); c, s = math.cos(th), math.sin(th)
    R0 = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], device=dev, dtype=F0_gt.dtype)
    F0p = F0_gt @ R0[None]                       # right-multiply (global rest rotation)
    V0 = torch.linalg.svdvals(F0_gt); V0p = torch.linalg.svdvals(F0p)
    traj = sc.rollout_F0(x0, F0_gt, cfg.gt_logE, cfg.K).cpu().numpy()     # (K+1,n,3)
    trajp = sc.rollout_F0(x0, F0p, cfg.gt_logE, cfg.K).cpu().numpy()
    dF = float((F0p - F0_gt).abs().max()); dV = float((V0p - V0).abs().max())
    dT = float(np.abs(trajp - traj).max())
    print(f"[b2-sanity] GAUGE: max|F0'-F0|={dF:.3f} (DIFFERENT IC)  "
          f"max|svd(F0')-svd(F0)|={dV:.2e} (SAME stretch)  "
          f"max|traj'-traj|={dT:.2e} (SAME dynamics)")

    fig = plt.figure(figsize=(15, 8))
    # row1: traj overlay at 3 frames (xy)
    frames = [0, cfg.K // 2, cfg.K]
    for i, f in enumerate(frames):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.scatter(traj[f, :, 0], traj[f, :, 1], s=4, c="tab:blue", label="from F0")
        ax.scatter(trajp[f, :, 0], trajp[f, :, 1], s=4, c="tab:red", alpha=0.5, label="from F0@R0")
        ax.set_title(f"frame {f}  (overlap = same dynamics)"); ax.set_aspect("equal")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        if i == 0: ax.legend(fontsize=8)
    # row2: F0[1,0] vs F0'[1,0] differ; V0 identical; traj diff over time
    ax = fig.add_subplot(2, 3, 4)
    ax.scatter(xs, F0_gt[:, 1, 0].cpu().numpy(), s=5, label="F0[1,0]")
    ax.scatter(xs, F0p[:, 1, 0].cpu().numpy(), s=5, alpha=0.6, label="(F0@R0)[1,0]")
    ax.set_title(f"F0 components DIFFER (rot {cfg.rot_z_deg} deg)"); ax.set_xlabel("x0_x"); ax.legend(fontsize=8)
    ax = fig.add_subplot(2, 3, 5)
    ax.scatter(V0.flatten().cpu().numpy(), V0p.flatten().cpu().numpy(), s=4)
    ax.plot([V0.min().item(), V0.max().item()], [V0.min().item(), V0.max().item()], "k--", lw=1)
    ax.set_title(f"singular values: V0' vs V0 (max diff {dV:.1e})"); ax.set_xlabel("svd(F0)"); ax.set_ylabel("svd(F0@R0)")
    ax = fig.add_subplot(2, 3, 6)
    ax.plot(np.abs(trajp - traj).reshape(cfg.K + 1, -1).max(1), "-o", ms=3)
    ax.set_title(f"max|traj' - traj| per frame (peak {dT:.1e})"); ax.set_xlabel("frame"); ax.set_yscale("log")
    fig.suptitle(f"GAUGE: (x0, F0) vs (x0, F0@R0) -- different IC, same V0, IDENTICAL dynamics")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "3_gauge_demo.png"), dpi=130); plt.close(fig)

    print(f"[b2-sanity] -> {out_dir}  (1_w_quiver / 2_F0_from_w / 3_gauge_demo .png)")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(B2SanityConfig))
