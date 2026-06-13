"""Entrypoint: 2D (alpha, logE) landscape -- the E <-> F0-amplitude degeneracy probe.

Fix the F0 DIRECTION to the snapshot's true left-stretch V0 = sqrt(F F^T) (rotation
stripped: it is a dynamics-gauge for isotropic material), and scale its amplitude:
  F0(alpha) = V0^alpha = Q diag(mu_i^(alpha/2)) Q^T,  (mu,Q) = eigh(F F^T)
alpha=0 -> I (rest, no pre-stress); alpha=1 -> true snapshot deformation.

Pure release (v0=0, C=0): the ONLY thing driving motion is the pre-stress, whose
magnitude ~ E * strain(F0(alpha)). Scan (alpha, logE) and read the valley:
  - curved valley (alpha up <-> E down) => multiplicative E*strain degeneracy.
  - isolated min at (1, logE_GT)        => E and F0-amplitude jointly identifiable.

Output under outputs/explore/f0_alpha_landscape/.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Tuple

import tyro

from ..gpu import pick_free_gpu


@dataclass
class F0AlphaLandscapeConfig:
    cache_path: str = ("/tmp2/b10401006/ev-project/generative-phys/outputs/"
                       "_scene_cache/telephone_ds0.1_g32_k8.pt")
    snapshot: str = ("/tmp2/b10401006/ev-project/generative-phys/outputs/"
                     "explore/f0_snapshot/tele_f0_xy/snapshot_frame08.pt")
    gt_logE: float = 5.0
    nu: float = 0.3
    K: int = 8
    alpha_lo: float = 0.0
    alpha_hi: float = 1.4
    alpha_n: int = 15
    logE_lo: float = 4.0
    logE_hi: float = 6.0
    logE_n: int = 21
    label: str = "tele_alpha_logE_f8"


def run(cfg: F0AlphaLandscapeConfig) -> str:
    pick_free_gpu()
    import numpy as np
    import torch
    import warp as wp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..config import SceneSpec, ScenePreset, SimConfig
    from ..scene_io import load_from_spec
    from ..sim_render import build_mpm

    t0 = time.time()
    out_dir = os.path.join("outputs", "explore", "f0_alpha_landscape", cfg.label)
    os.makedirs(out_dir, exist_ok=True)

    spec = SceneSpec(preset=ScenePreset.telephone, cache_path=cfg.cache_path)
    sim = SimConfig(); sim.num_frames = cfg.K + 1
    scene = load_from_spec(spec, sim)
    dev = scene.device
    rest_xyz = scene.sim_xyzs
    n = rest_xyz.shape[0]
    free = (~scene.freeze_mask).to(dev)

    snap = torch.load(cfg.snapshot, map_location=dev)
    snap_x = snap["x"].to(dev).float()
    snap_F = snap["F"].to(dev).float()

    # left stretch V0 = sqrt(F F^T); V0^alpha via symmetric eigdecomp of F F^T.
    FFt = snap_F @ snap_F.transpose(-1, -2)               # [n,3,3] SPD
    mu, Q = torch.linalg.eigh(FFt)                        # mu>0, Q orthonormal
    mu = mu.clamp_min(1e-9)

    def F0_at(alpha):
        # V0^alpha = Q diag(mu^(alpha/2)) Q^T
        s = mu.pow(alpha / 2.0)                           # [n,3]
        return (Q * s.unsqueeze(-2)) @ Q.transpose(-1, -2)

    print(f"[a-E] N={n} free={int(free.sum())} K={cfg.K}; snapshot frame {snap.get('frame')}; "
          f"alpha 1 maxdev {(mu[free].sqrt()-1).abs().max():.3f}")

    def rollout(F0, logE):
        solver, state, model = build_mpm(scene, sim, requires_grad=False)
        dens = torch.ones_like(rest_xyz[..., 0]) * sim.density
        state.reset_density(dens.clone(), torch.ones_like(dens).int(), dev, update_mass=True)
        with torch.no_grad():
            E_t = torch.ones_like(rest_xyz[..., 0]) * float(10.0 ** logE)
            nu_t = torch.ones_like(rest_xyz[..., 0]) * cfg.nu
            solver.set_E_nu_from_torch(model, E_t.clone(), nu_t.clone(), dev)
            solver.prepare_mu_lam(model, state, dev)
            v0 = torch.zeros(n, 3, device=dev); C0 = torch.zeros(n, 3, 3, device=dev)
            state.continue_from_torch(snap_x.clone(), v0, F0, C0, device=dev, requires_grad=False)
            prev = state
            out = [wp.to_torch(prev.particle_x).clone()]
            for _ in range(cfg.K):
                for _ in range(sim.substep):
                    nxt = prev.partial_clone(requires_grad=False)
                    solver.p2g2p_differentiable(model, prev, nxt, sim.substep_size, device=dev)
                    prev = nxt
                out.append(wp.to_torch(prev.particle_x).clone())
        return torch.stack(out)

    def traj_loss(pred, gt):
        return float((((pred[1:] - gt[1:]) ** 2).sum(-1))[:, free].mean())

    gt = rollout(F0_at(1.0), cfg.gt_logE)                 # true: alpha=1, GT E
    gt_motion = (gt[-1, free] - gt[0, free]).norm(dim=-1).mean().item()

    alphas = np.linspace(cfg.alpha_lo, cfg.alpha_hi, cfg.alpha_n)
    logEs = np.linspace(cfg.logE_lo, cfg.logE_hi, cfg.logE_n)
    Lmap = np.zeros((cfg.alpha_n, cfg.logE_n))
    for i, a in enumerate(alphas):
        F0a = F0_at(float(a))
        for j, le in enumerate(logEs):
            Lmap[i, j] = traj_loss(rollout(F0a, float(le)), gt)
    ai, aj = np.unravel_index(np.argmin(Lmap), Lmap.shape)
    print(f"[a-E] GT motion {gt_motion:.4f}; argmin (alpha {alphas[ai]:.3f}, logE {logEs[aj]:.3f}) "
          f"vs GT (1.0, {cfg.gt_logE}); minloss {Lmap.min():.2e}")
    # per-logE-column min-alpha trace: reveals a degeneracy ridge if it slopes
    ridge = alphas[Lmap.argmin(axis=0)]
    print(f"[a-E] per-logE best-alpha (degeneracy ridge): "
          f"{dict(zip(np.round(logEs,2).tolist(), np.round(ridge,2).tolist()))}")

    np.savez(os.path.join(out_dir, "landscape2d.npz"), alphas=alphas, logEs=logEs, Lmap=Lmap)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    im = ax[0].imshow(np.log10(Lmap + 1e-18), origin="lower", aspect="auto",
                      extent=[logEs[0], logEs[-1], alphas[0], alphas[-1]], cmap="viridis")
    fig.colorbar(im, ax=ax[0], label="log10 trajectory loss")
    ax[0].plot(logEs, ridge, "w.-", lw=1, ms=4, label="min-alpha per logE")
    ax[0].scatter([cfg.gt_logE], [1.0], c="red", marker="*", s=160, label="GT (1, E*)", zorder=5)
    ax[0].set_xlabel("log10 E"); ax[0].set_ylabel("alpha (F0 amplitude)")
    ax[0].set_title("loss(alpha, logE)"); ax[0].legend(fontsize=8)
    # contour to see valley curvature
    cs = ax[1].contour(logEs, alphas, np.log10(Lmap + 1e-18), levels=14, cmap="viridis")
    ax[1].scatter([cfg.gt_logE], [1.0], c="red", marker="*", s=160, zorder=5)
    ax[1].set_xlabel("log10 E"); ax[1].set_ylabel("alpha"); ax[1].set_title("log-loss contours")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "alpha_logE_landscape.png"), dpi=120)
    plt.close(fig)
    print(f"[a-E] DONE -> {out_dir}  ({time.time()-t0:.1f}s)")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(F0AlphaLandscapeConfig))
