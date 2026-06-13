"""F0 training, rung 2 (coarse field, still FD on warp -- no gic/autograd yet).

u(X) = sum_k Phi_k(X) theta[k]  with a SMALL analytic basis -> F0 = I + grad u is
analytic and AFFINE in theta (so grad u precomputes once). theta (15 coeffs) is fit by
finite difference + Adam, E fixed at GT, time_L2. GT = a single bend mode (u_y's B(xi_x)
coeff = A) so the model can represent it exactly (self-consistent, residual -> 0). Tests
coarse-field recovery + which of the 15 modes are identifiable. Only the FULL field
(MLP, hundreds of params) needs MPM autograd (gic); this rung stays cheap on warp.

Basis (xi = (X-c)/H in [-1,1]^3, B(xi)=sin(pi(xi+1)/2) = half-sine along x):
  Phi = [xi_x, xi_y, xi_z, B(xi_x), B(xi_x)*xi_y]   (5 funcs x 3 comps = 15 theta)

Output (outputs/explore/f0_train_ufield/<label>/): loss_curve.png, theta_recovery.png,
result_overlay.gif, fit_result.json
"""
from __future__ import annotations

import os
import time as _time
from dataclasses import dataclass
from typing import Tuple

import tyro

from ..gpu import pick_free_gpu

FUNCS = ["lin_x", "lin_y", "lin_z", "bend", "bend*y"]   # 5 basis functions
COMPS = ["ux", "uy", "uz"]


@dataclass
class UFieldConfig:
    A: float = 0.05            # GT bend amplitude (u_y's B(xi_x) coeff)
    nx: int = 22
    ny: int = 9
    nz: int = 16
    half: Tuple[float, float, float] = (0.18, 0.08, 0.14)
    z_base: float = 0.30
    nu: float = 0.3
    gt_logE: float = 4.5
    K: int = 16
    n_iters: int = 30
    lr: float = 0.03
    fd_eps: float = 0.01
    clip: float = 1.0
    ckpt_every: int = 5
    overlay_fps: int = 3
    min_quota_hours: float = 8.0
    label: str = "coarse_bend"


def run(cfg: UFieldConfig) -> str:
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    from ._block import Scene

    t0 = _time.time()
    out_dir = os.path.join("outputs", "explore", "f0_train_ufield", cfg.label)
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda:0"
    sc = Scene("uniform", nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, half=cfg.half, z_base=cfg.z_base,
               nu=cfg.nu, gt_logE=cfg.gt_logE, S_gt=(0.0,) * 6, device=dev)
    X = sc.X_rest; n = sc.n
    c = X.mean(0); H = torch.tensor(cfg.half, device=dev)
    xi = (X - c) / H                                       # [n,3] in ~[-1,1]
    pi = float(np.pi)
    B = torch.sin(pi * (xi[:, 0] + 1) / 2)                 # [n] half-sine along x
    dB = (pi / 2) * torch.cos(pi * (xi[:, 0] + 1) / 2)     # dB/d(xi_x)

    # Phi [n,5], and dPhi[n,5,3] = d Phi_k / d X_j  (analytic; xi_a/X_b = delta/H_a)
    Phi = torch.stack([xi[:, 0], xi[:, 1], xi[:, 2], B, B * xi[:, 1]], dim=1)   # [n,5]
    dPhi = torch.zeros(n, 5, 3, device=dev)
    dPhi[:, 0, 0] = 1.0 / H[0]                              # lin_x
    dPhi[:, 1, 1] = 1.0 / H[1]                              # lin_y
    dPhi[:, 2, 2] = 1.0 / H[2]                              # lin_z
    dPhi[:, 3, 0] = dB / H[0]                               # bend (B(xi_x))
    dPhi[:, 4, 0] = dB * xi[:, 1] / H[0]                    # bend*y : d/dx
    dPhi[:, 4, 1] = B / H[1]                                #         d/dy
    eye = torch.eye(3, device=dev)

    def theta_to_t(th):                                    # flat (15,) -> [5,3] torch
        return torch.tensor(np.asarray(th, dtype=np.float32).reshape(5, 3), device=dev)

    def x0_F0(th):
        t = theta_to_t(th)
        u = torch.einsum("nk,ki->ni", Phi, t)              # [n,3]
        F0 = eye[None] + torch.einsum("ki,nkj->nij", t, dPhi)
        return (X + u).contiguous(), F0.contiguous()

    th_gt = np.zeros((5, 3)); th_gt[3, 1] = cfg.A          # bend coeff on u_y
    th_gt = th_gt.reshape(-1)
    x0g, F0g = x0_F0(th_gt)
    gt_traj = sc.rollout_F0(x0g, F0g, cfg.gt_logE, cfg.K)
    print(f"[ufield] GT bend A={cfg.A} (theta idx {3*3+1}); maxdev {float((x0g-X).norm(dim=1).max()):.4f}")

    def L(th):
        x0, F0 = x0_F0(th)
        v = float(((sc.rollout_F0(x0, F0, cfg.gt_logE, cfg.K) - gt_traj) ** 2).sum(-1).mean())
        return v if np.isfinite(v) else 1.0

    ckpt = os.path.join(out_dir, "fit_result.json")
    if os.path.exists(ckpt):
        st = json.load(open(ckpt)); th = np.array(st["theta"]); m = np.array(st["m"]); v = np.array(st["v"])
        start = st["iter"]; thist = [np.array(x) for x in st["thist"]]; lhist = st["lhist"]
        print(f"[ufield] resume from iter {start}")
    else:
        th = np.zeros(15); m = np.zeros(15); v = np.zeros(15); start = 0
        thist = [th.copy()]; lhist = [L(th)]
    b1, b2 = 0.9, 0.999

    def save(it):
        json.dump({"A": cfg.A, "th_gt": th_gt.tolist(), "gt_logE": cfg.gt_logE, "iter": it,
                   "theta": th.tolist(), "m": m.tolist(), "v": v.tolist(),
                   "thist": [x.tolist() for x in thist], "lhist": lhist,
                   "final_err": (th - th_gt).tolist()}, open(ckpt, "w"), indent=2)

    for it in range(start, cfg.n_iters):
        g = np.zeros(15)
        for d in range(15):
            tp = th.copy(); tp[d] += cfg.fd_eps; tm = th.copy(); tm[d] -= cfg.fd_eps
            g[d] = (L(tp) - L(tm)) / (2 * cfg.fd_eps)
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** (it + 1)); vh = v / (1 - b2 ** (it + 1))
        th = np.clip(th - cfg.lr * mh / (np.sqrt(vh) + 1e-12), -cfg.clip, cfg.clip)
        thist.append(th.copy()); lhist.append(L(th))
        if (it + 1) % cfg.ckpt_every == 0 or it == cfg.n_iters - 1:
            save(it + 1)
        print(f"[ufield] it {it+1:2d} loss {lhist[-1]:.3e}  bend_uy {th[3*3+1]:+.3f} (GT {cfg.A})  max|other| {np.abs(np.delete(th-th_gt, 3*3+1)).max():.3f}")
    err = th - th_gt
    print(f"[ufield] DONE bend_uy {th[10]:+.3f} (GT {cfg.A})  worst distractor |err| {np.abs(np.delete(err,10)).max():.4f}")

    # ---- viz ----
    fig, ax = plt.subplots(figsize=(7, 4.2)); ax.plot(lhist, "-o", ms=3); ax.set_yscale("log")
    ax.set_xlabel("iter"); ax.set_ylabel("time_L2 (log)"); ax.set_title("coarse u-field recovery loss")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=120); plt.close(fig)

    labels = [f"{COMPS[i]}:{FUNCS[k]}" for k in range(5) for i in range(3)]
    fig, ax = plt.subplots(figsize=(13, 4.6)); xs = np.arange(15)
    ax.bar(xs - 0.2, th_gt, 0.4, label="GT", color="lightgray")
    ax.bar(xs + 0.2, th, 0.4, label="fit", color="steelblue")
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("theta"); ax.legend()
    ax.set_title(f"coarse field theta: GT vs fit (only uy:bend should be {cfg.A}, rest 0)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "theta_recovery.png"), dpi=120); plt.close(fig)

    # overlay GT vs fit rollout
    x0f, F0f = x0_F0(th); fit_traj = sc.rollout_F0(x0f, F0f, cfg.gt_logE, cfg.K)
    items = [("GT", gt_traj.cpu().numpy(), "black"), ("fit", fit_traj.cpu().numpy(), "tab:orange")]
    allp = np.concatenate([it[1].reshape(-1, 3) for it in items], 0); mn, mx = allp.min(0), allp.max(0)
    Lr = items[0][1].shape[0]; proj = [(0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z")]
    fig = plt.figure(figsize=(11, 9)); ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax2 = [fig.add_subplot(2, 2, k) for k in (2, 3, 4)]

    def draw(f):
        ax3d.cla()
        for lbl, Xt, col in items:
            ax3d.scatter(Xt[f][:, 0], Xt[f][:, 1], Xt[f][:, 2], c=col, s=3, alpha=0.4, label=lbl, depthshade=False)
        ax3d.set_xlim(mn[0], mx[0]); ax3d.set_ylim(mn[1], mx[1]); ax3d.set_zlim(mn[2], mx[2])
        ax3d.set_title(f"release frame {f}/{Lr-1}", fontsize=9); ax3d.legend(fontsize=8, loc="upper left")
        for axp, (a, b, la, lb) in zip(ax2, proj):
            axp.cla()
            for lbl, Xt, col in items:
                axp.scatter(Xt[f][:, a], Xt[f][:, b], c=col, s=4, alpha=0.4)
            axp.set_xlim(mn[a], mx[a]); axp.set_ylim(mn[b], mx[b]); axp.set_aspect("equal")
            axp.set_xlabel(la); axp.set_ylabel(lb); axp.set_title(f"{la}{lb}")
        return ()

    draw(0); anim = FuncAnimation(fig, draw, frames=Lr, blit=False)
    anim.save(os.path.join(out_dir, "result_overlay.gif"), writer=PillowWriter(fps=cfg.overlay_fps)); plt.close(fig)
    print(f"[ufield] -> {out_dir} ({_time.time()-t0:.1f}s)")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(UFieldConfig))
