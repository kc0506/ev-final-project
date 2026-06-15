"""Forward nu-sensitivity sweep: which excitation makes Poisson ratio identifiable?

Hypothesis (see reports/gauge_math.md companion reasoning): nu lives in lambda
(volumetric/bulk); shear/bending is ~isochoric (J~1) so nu is invisible there.
Volumetric excitation (J!=1) exposes nu. Test by sweeping nu at FIXED E and FIXED
F0, measuring how much the release trajectory moves, on three same-block scenes:

  release  : natural grip-pull (F0 ~ equilibrium-for-nu, near-isochoric?) -- the "natural BC" method
  uniaxial : uniform F0 = diag(e^a,1,1) (imposed pure x-stretch, no lateral relax) -- volumetric, Poisson
  bend     : gradu I+grad u half-sine (pure shear)            -- control, expect flat

Per scene: build once, take its (x0, F0), then rollout at each nu (rollout uses
sc.nu) and compare to the nu0 trajectory. Metric = max|traj(nu)-traj(nu0)| /
motion_scale. Forward only.  Output: outputs/explore/f0_nu_sensitivity/<label>/.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import tyro

from ..gpu import pick_free_gpu


@dataclass
class NuSensConfig:
    nus: Tuple[float, ...] = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45)
    nu0: float = 0.3                  # reference nu
    gt_logE: float = 4.5
    K: int = 24
    uniaxial_a: float = 0.20          # uniform x-stretch log-stretch (V0=diag(e^a,1,1))
    gradu_A: float = 0.05
    min_quota_hours: float = 0.0
    label: str = "release_uniaxial_bend"


def run(cfg: NuSensConfig) -> str:
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

    out_dir = os.path.join("outputs", "explore", "f0_nu_sensitivity", cfg.label)
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda:0"
    nus = list(cfg.nus)
    if cfg.nu0 not in nus:
        nus = sorted(nus + [cfg.nu0])

    scenes = {
        "release":  dict(scene="release"),
        "uniaxial": dict(scene="uniform", S_gt=(cfg.uniaxial_a, 0.0, 0.0, 0.0, 0.0, 0.0)),
        "bend":     dict(scene="gradu", gradu_A=cfg.gradu_A),
    }

    results = {}
    for name, kw in scenes.items():
        sc = Scene(nu=cfg.nu0, gt_logE=cfg.gt_logE, device=dev, **kw)
        x0, F0 = sc.x_snap.clone(), sc.F_snap.clone()
        trajs = {}
        for nu in nus:
            sc.nu = nu                                   # rollout_F0 -> _setE uses self.nu
            trajs[nu] = sc.rollout_F0(x0, F0, cfg.gt_logE, cfg.K).cpu().numpy()  # (K+1,n,3)
        ref = trajs[cfg.nu0]
        motion = float(np.abs(ref - ref[0:1]).max())     # scene motion scale
        dev_abs = np.array([np.abs(trajs[nu] - ref).max() for nu in nus])
        dev_rel = dev_abs / max(motion, 1e-9)
        # the ACTUAL recovery loss: per-particle MSE vs the GT(nu0) trajectory, per nu
        mse = np.array([float(((trajs[nu] - ref) ** 2).sum(-1).mean()) for nu in nus])
        # lateral (y,z) vs axial (x) decomposition of the nu-induced change (Poisson signature)
        lat = np.array([np.abs(trajs[nu][..., 1:] - ref[..., 1:]).max() for nu in nus])
        axi = np.array([np.abs(trajs[nu][..., 0] - ref[..., 0]).max() for nu in nus])
        results[name] = dict(dev_abs=dev_abs, dev_rel=dev_rel, mse=mse, lat=lat, axi=axi, motion=motion)
        print(f"[nu-sens] {name:9s} motion={motion:.4f}  max nu-dev(rel)={dev_rel.max():.3f}  "
              f"MSE(nu0)={mse[list(nus).index(cfg.nu0)]:.2e} max-MSE={mse.max():.3e}")

    # ---- plot: the LOSS landscape L(nu) is the headline; sensitivity + Poisson alongside ----
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for name, r in results.items():
        axs[0].semilogy(nus, np.maximum(r["mse"], 1e-12), "-o", ms=4, label=f"{name}")
    axs[0].axvline(cfg.nu0, color="k", ls=":", lw=1)
    axs[0].set_xlabel("nu (rollout)"); axs[0].set_ylabel("per-particle MSE vs traj(nu0)")
    axs[0].set_title(f"LOSS landscape L(nu)  (GT nu0={cfg.nu0}, log-y)"); axs[0].legend(fontsize=9)
    for name, r in results.items():
        m = r["mse"]; mn = m / max(m.max(), 1e-30)
        axs[1].plot(nus, mn, "-o", ms=4, label=f"{name}")
    axs[1].axvline(cfg.nu0, color="k", ls=":", lw=1)
    axs[1].set_xlabel("nu (rollout)"); axs[1].set_ylabel("MSE / max (per scene)")
    axs[1].set_title("loss SHAPE normalized (bowl depth/width => recoverability)"); axs[1].legend(fontsize=9)
    for name, r in results.items():
        axs[2].plot(nus, r["lat"], "-o", ms=4, label=f"{name} lateral(y,z)")
        axs[2].plot(nus, r["axi"], "--s", ms=3, alpha=0.5, label=f"{name} axial(x)")
    axs[2].axvline(cfg.nu0, color="k", ls=":", lw=1)
    axs[2].set_xlabel("nu (rollout)"); axs[2].set_ylabel("max |traj(nu)-traj(nu0)| (abs)")
    axs[2].set_title("lateral vs axial change (Poisson signature)"); axs[2].legend(fontsize=7)
    fig.suptitle(f"nu loss landscape + sensitivity (E fixed {cfg.gt_logE}, fixed F0): "
                 f"stretch (release/uniaxial) exposes nu, bend doesn't")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "nu_sensitivity.png"), dpi=130); plt.close(fig)
    np.savez(os.path.join(out_dir, "nu_sens.npz"), nus=np.array(nus),
             **{f"{n}_{k}": v for n, r in results.items() for k, v in r.items() if k != "motion"})
    print(f"[nu-sens] -> {out_dir}/nu_sensitivity.png")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(NuSensConfig))
