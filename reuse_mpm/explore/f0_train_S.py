"""F0 training, rung 1: recover a GLOBAL homogeneous left-stretch V0=expm(S), S a
6-DOF symmetric matrix (log-Euclidean: always SPD, S=0 -> rest). SELF-CONSISTENT --
GT is itself a uniform F0=expm(S_gt) (hand-set, not the non-uniform dynamic-pull), so
a uniform model matches the GT and the residual can reach ~0. This validates the F0
gradient mechanism + maps which of the 6 S-DOF are identifiable from the release.

E is FIXED at GT (isolate F0 from the E.strain degeneracy). With E fixed, S changes the
deformation AMPLITUDE/shape (not frequency) -> time_L2 reads it cleanly (amplitude
channel; no beat). Gradient = finite-difference over the 6 components (12 evals/iter)
+ Adam. warp-self (no model floor); see [[warp-fd-vs-mpm-autograd]].

Output (outputs/explore/f0_train_S/<label>/): loss_curve.png, S_recovery.png,
S_error_bar.png, result_overlay.gif/png, fit_result.json
"""
from __future__ import annotations

import os
import time as _time
from dataclasses import dataclass
from typing import Tuple

import tyro

from ..gpu import pick_free_gpu

LAB = ["xx", "yy", "zz", "xy", "xz", "yz"]


@dataclass
class TrainSConfig:
    S_gt: Tuple[float, float, float, float, float, float] = (0.2, -0.1, -0.1, 0.05, 0.0, 0.0)
    nx: int = 22
    ny: int = 9
    nz: int = 16
    half: Tuple[float, float, float] = (0.18, 0.08, 0.14)
    z_base: float = 0.30
    nu: float = 0.3
    gt_logE: float = 4.5
    K: int = 20
    n_iters: int = 50
    lr: float = 0.05
    fd_eps: float = 0.02
    clip: float = 1.5
    ckpt_every: int = 5
    overlay_fps: int = 3
    min_quota_hours: float = 8.0
    label: str = "globalS"


def run(cfg: TrainSConfig) -> str:
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
    out_dir = os.path.join("outputs", "explore", "f0_train_S", cfg.label)
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda:0"
    sc = Scene("uniform", nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, half=cfg.half, z_base=cfg.z_base,
               nu=cfg.nu, gt_logE=cfg.gt_logE, S_gt=cfg.S_gt, device=dev)
    s_gt = np.array(cfg.S_gt, dtype=float)
    print(f"[trainS] S_gt={s_gt}  maxdev={sc.maxdev:.4f}  F0_stretch max {sc.F0_stretch.max():.3f}")

    gt_traj = sc.rollout(cfg.gt_logE, cfg.K)[0]   # GT release from the uniform F0(S_gt)

    def L(s6):
        x0, F0 = sc.affine_from_s6(s6)
        traj = sc.rollout_F0(x0, F0, cfg.gt_logE, cfg.K)
        v = float(((traj - gt_traj) ** 2).sum(-1).mean())
        return v if np.isfinite(v) else 1.0

    # ---- FD(6) + Adam, checkpointed ----
    ckpt = os.path.join(out_dir, "fit_result.json")
    if os.path.exists(ckpt):
        st = json.load(open(ckpt))
        s = np.array(st["s"]); m = np.array(st["m"]); v = np.array(st["v"])
        start = st["iter"]; shist = [np.array(x) for x in st["shist"]]; lhist = st["lhist"]
        print(f"[trainS] resume from iter {start}")
    else:
        s = np.zeros(6); m = np.zeros(6); v = np.zeros(6); start = 0
        shist = [s.copy()]; lhist = [L(s.tolist())]
    b1, b2 = 0.9, 0.999

    def save(it):
        json.dump({"S_gt": cfg.S_gt, "gt_logE": cfg.gt_logE, "lr": cfg.lr, "n_iters": cfg.n_iters,
                   "s": s.tolist(), "m": m.tolist(), "v": v.tolist(), "iter": it,
                   "shist": [x.tolist() for x in shist], "lhist": lhist,
                   "final_err": (s - s_gt).tolist()}, open(ckpt, "w"), indent=2)

    for it in range(start, cfg.n_iters):
        g = np.zeros(6)
        for d in range(6):
            sp = s.copy(); sp[d] += cfg.fd_eps; sm = s.copy(); sm[d] -= cfg.fd_eps
            g[d] = (L(sp.tolist()) - L(sm.tolist())) / (2 * cfg.fd_eps)
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** (it + 1)); vh = v / (1 - b2 ** (it + 1))
        s = np.clip(s - cfg.lr * mh / (np.sqrt(vh) + 1e-12), -cfg.clip, cfg.clip)
        shist.append(s.copy()); lhist.append(L(s.tolist()))
        if (it + 1) % cfg.ckpt_every == 0 or it == cfg.n_iters - 1:
            save(it + 1)
        print(f"[trainS] it {it+1:2d} loss {lhist[-1]:.3e}  S {np.array2string(s, precision=3, suppress_small=True)}")
    err = s - s_gt
    print(f"[trainS] DONE  S_gt {s_gt}  ->  {np.array2string(s, precision=3)}  |err| {np.abs(err)}")

    # ---- viz ----
    S = np.array(shist)
    fig, ax = plt.subplots(figsize=(7, 4.4)); ax.plot(lhist, "-o", ms=3); ax.set_yscale("log")
    ax.set_xlabel("iter"); ax.set_ylabel("time_L2 (log)"); ax.set_title(f"S recovery loss (GT logE {cfg.gt_logE})")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    cmap = plt.cm.tab10(np.linspace(0, 1, 6))
    for d in range(6):
        ax.plot(S[:, d], "-", color=cmap[d], label=f"{LAB[d]} ->{s[d]:+.3f} (GT {s_gt[d]:+.2f})")
        ax.axhline(s_gt[d], color=cmap[d], ls=":", lw=1)
    ax.set_xlabel("iter"); ax.set_ylabel("S component"); ax.set_title("S components vs iter (dotted = GT)")
    ax.legend(fontsize=8, ncol=2); fig.tight_layout(); fig.savefig(os.path.join(out_dir, "S_recovery.png"), dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(LAB, np.abs(err), color="indianred")
    for i, e in enumerate(np.abs(err)):
        ax.text(i, e, f"{e:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("|S - S_gt|"); ax.set_title("per-component recovery error (which DOF are identifiable)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "S_error_bar.png"), dpi=120); plt.close(fig)

    # ---- overlay: GT vs fitted-S rollout (3d + triplane) ----
    x0f, F0f = sc.affine_from_s6(s.tolist()); fit_traj = sc.rollout_F0(x0f, F0f, cfg.gt_logE, cfg.K)
    items = [("GT", gt_traj.cpu().numpy(), "black"), (f"fit S", fit_traj.cpu().numpy(), "tab:orange")]
    allp = np.concatenate([it[1].reshape(-1, 3) for it in items], 0); mn, mx = allp.min(0), allp.max(0)
    Lr = items[0][1].shape[0]; proj = [(0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z")]
    fig = plt.figure(figsize=(11, 9)); ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax2 = [fig.add_subplot(2, 2, k) for k in (2, 3, 4)]

    def draw(f):
        ax3d.cla()
        for lbl, X, c in items:
            ax3d.scatter(X[f][:, 0], X[f][:, 1], X[f][:, 2], c=c, s=3, alpha=0.4, label=lbl, depthshade=False)
        ax3d.set_xlim(mn[0], mx[0]); ax3d.set_ylim(mn[1], mx[1]); ax3d.set_zlim(mn[2], mx[2])
        ax3d.set_title(f"release frame {f}/{Lr-1}", fontsize=9); ax3d.legend(fontsize=8, loc="upper left")
        for axp, (a, b, la, lb) in zip(ax2, proj):
            axp.cla()
            for lbl, X, c in items:
                axp.scatter(X[f][:, a], X[f][:, b], c=c, s=4, alpha=0.4)
            axp.set_xlim(mn[a], mx[a]); axp.set_ylim(mn[b], mx[b]); axp.set_aspect("equal")
            axp.set_xlabel(la); axp.set_ylabel(lb); axp.set_title(f"{la}{lb}")
        return ()

    draw(0); anim = FuncAnimation(fig, draw, frames=Lr, blit=False)
    anim.save(os.path.join(out_dir, "result_overlay.gif"), writer=PillowWriter(fps=cfg.overlay_fps)); plt.close(fig)
    print(f"[trainS] -> {out_dir} ({_time.time()-t0:.1f}s)")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(TrainSConfig))
