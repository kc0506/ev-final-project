"""Probe where the F-incompatibility floor comes from: track, over the pull, how far
warp's deformation gradient F(t) drifts from the position-gradient I + grad u(t).

For each recorded pull frame t: u(t) = x(t) - X_rest, finite-difference grad u(t) on
the structured grid -> compatible F_FD(t) = I + grad u(t); compare to the tracked
F(t) = particle_F_trial(t). A growing |F(t) - F_FD(t)| over t = the grip-BC / MPM
transfer accumulating incompatibility (the part the compatible MLP can never fit).

Forward only (one pull). Output: outputs/explore/f0_F_drift_probe/<label>/.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import tyro

from ..gpu import pick_free_gpu


@dataclass
class FDriftConfig:
    scene: str = "release"          # pull-based scenes: release / drop / squeeze
    gt_logE: float = 4.5
    nu: float = 0.3
    min_quota_hours: float = 0.0
    label: str = "release"


def run(cfg: FDriftConfig) -> str:
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

    out_dir = os.path.join("outputs", "explore", "f0_F_drift_probe", cfg.label)
    os.makedirs(out_dir, exist_ok=True)

    sc = Scene(cfg.scene, nu=cfg.nu, gt_logE=cfg.gt_logE, device="cuda:0")
    X = sc.X_rest.double().cpu().numpy()
    nx = len(np.unique(np.round(X[:, 0], 6))); ny = len(np.unique(np.round(X[:, 1], 6)))
    nz = len(np.unique(np.round(X[:, 2], 6)))
    assert nx * ny * nz == X.shape[0], (nx, ny, nz, X.shape)
    Xg = X.reshape(nx, ny, nz, 3)
    gx, gy, gz = Xg[:, 0, 0, 0], Xg[0, :, 0, 1], Xg[0, 0, :, 2]

    def fd_F(xt):                                   # xt (N,3) -> F_FD = I + grad(xt - X)
        ug = (xt - X).reshape(nx, ny, nz, 3)
        gradu = np.zeros((nx, ny, nz, 3, 3))
        for i in range(3):
            d = np.gradient(ug[..., i], gx, gy, gz)
            for j in range(3):
                gradu[..., i, j] = d[j]
        return np.eye(3)[None] + gradu.reshape(-1, 3, 3)

    Xs = [x.double().cpu().numpy() for x in sc.pull_X]      # per-frame positions
    Fs = [f.double().cpu().numpy() for f in sc.pull_F]      # per-frame tracked F
    T = len(Xs)
    drift_max, drift_mean, motion = [], [], []
    print(f"[Fdrift] scene={cfg.scene} grid ({nx},{ny},{nz}) frames={T}")
    print("  frame |  motion  | max|F-F_FD| | mean|F-F_FD| | mean|F-F_FD|/|F-I|")
    for t in range(T):
        F_fd = fd_F(Xs[t]); F_tr = Fs[t]
        d = np.abs(F_tr - F_fd).reshape(-1, 9).max(1)
        mot = float(np.linalg.norm(Xs[t] - X, axis=1).mean())
        fdev = np.abs(F_tr - np.eye(3)[None]).reshape(-1, 9).max(1).mean()
        rel = float(d.mean() / max(fdev, 1e-9))
        drift_max.append(float(d.max())); drift_mean.append(float(d.mean())); motion.append(mot)
        print(f"   {t:3d}  | {mot:.5f} |  {d.max():.5f}   |   {d.mean():.5f}   |  {rel*100:5.1f}%")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(T), drift_max, "-o", label="max |F - F_FD|")
    ax.plot(range(T), drift_mean, "-s", label="mean |F - F_FD|")
    ax.set_xlabel("pull frame t"); ax.set_ylabel("F vs position-gradient drift")
    ax.set_title(f"{cfg.scene}: warp F(t) drift from compatible I+grad u(t) over the pull")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "F_drift.png"), dpi=120); plt.close(fig)
    np.savez(os.path.join(out_dir, "F_drift.npz"), drift_max=drift_max, drift_mean=drift_mean, motion=motion)
    print(f"[Fdrift] final-frame mean drift {drift_mean[-1]:.4f} (max {drift_max[-1]:.4f}) -> {out_dir}/F_drift.png")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(FDriftConfig))
