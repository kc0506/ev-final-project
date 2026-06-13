"""Entrypoint: E landscape under a FIXED (known) initial deformation F0.

The F0 analog of the fix-v0 -> E landscape (reports/.../landscape/E1d_*). Question:
if we KNOW the initial deformation, is E identifiable, and is a pre-stressed
release a BETTER-conditioned E excitation than a v0 kick?

Two excitations on one warp-self roundtrip (no cross-model floor -> the well is
pure E-observability, isolated):
  (a) v0-driven : rest state F0=I, v0 = xy kick           (reproduces E1d)
  (b) F0-release: F0 = snapshot deformation, v0=0, C=0     (pure elastic release)

For each: generate GT with true E, then scan logE and measure trajectory loss
(per-particle, we HAVE correspondence -> no chamfer null space) vs the GT future.
A deeper/narrower well = better E conditioning.

Config is LOCAL (explore convention). Output under outputs/explore/f0_e_landscape/.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import tyro

from ..gpu import pick_free_gpu


@dataclass
class F0ELandscapeConfig:
    cache_path: str = ("/tmp2/b10401006/ev-project/generative-phys/outputs/"
                       "_scene_cache/telephone_ds0.1_g32_k8.pt")
    snapshot: str = ("/tmp2/b10401006/ev-project/generative-phys/outputs/"
                     "explore/f0_snapshot/tele_f0_xy/snapshot_frame08.pt")
    gt_logE: float = 5.0
    nu: float = 0.3
    v0_kick: Tuple[float, float, float] = (0.3536, 0.3536, 0.0)  # for baseline (a)
    K: int = 8                       # frames of future to fit
    logE_lo: float = 3.5
    logE_hi: float = 6.5
    logE_n: int = 31
    label: str = "tele_f0release_vs_v0"


def run(cfg: F0ELandscapeConfig) -> str:
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
    out_dir = os.path.join("outputs", "explore", "f0_e_landscape", cfg.label)
    os.makedirs(out_dir, exist_ok=True)

    spec = SceneSpec(preset=ScenePreset.telephone, cache_path=cfg.cache_path)
    sim = SimConfig()
    sim.num_frames = cfg.K + 1
    scene = load_from_spec(spec, sim)
    dev = scene.device
    rest_xyz = scene.sim_xyzs
    n = rest_xyz.shape[0]
    free = (~scene.freeze_mask).to(dev)

    snap = torch.load(cfg.snapshot, map_location=dev)
    snap_x = snap["x"].to(dev).float()
    snap_F = snap["F"].to(dev).float()
    I_mat = torch.eye(3, device=dev)
    print(f"[f0E] N={n} free={int(free.sum())} K={cfg.K} GT logE={cfg.gt_logE}; "
          f"snapshot frame {snap.get('frame')} maxdev "
          f"{(torch.linalg.svdvals(snap_F[free])-1).abs().max():.3f}")

    # one reusable rollout: returns [K+1] lists of normalized positions (free only)
    def rollout(x0, v0, F0, C0, logE):
        solver, state, model = build_mpm(scene, sim, requires_grad=False)
        dens = torch.ones_like(rest_xyz[..., 0]) * sim.density
        state.reset_density(dens.clone(), torch.ones_like(dens).int(), dev, update_mass=True)
        with torch.no_grad():
            E_t = torch.ones_like(rest_xyz[..., 0]) * float(10.0 ** logE)
            nu_t = torch.ones_like(rest_xyz[..., 0]) * cfg.nu
            solver.set_E_nu_from_torch(model, E_t.clone(), nu_t.clone(), dev)
            solver.prepare_mu_lam(model, state, dev)
            state.continue_from_torch(x0.clone(), v0, F0, C0, device=dev, requires_grad=False)
            prev = state
            out = [wp.to_torch(prev.particle_x).clone()]
            for _ in range(cfg.K):
                for _ in range(sim.substep):
                    nxt = prev.partial_clone(requires_grad=False)
                    solver.p2g2p_differentiable(model, prev, nxt, sim.substep_size, device=dev)
                    prev = nxt
                out.append(wp.to_torch(prev.particle_x).clone())
        return torch.stack(out)  # [K+1, n, 3]

    def traj_loss(pred, gt):
        # mean over frames>0 of mean free-particle squared distance
        d = ((pred[1:] - gt[1:]) ** 2).sum(-1)        # [K, n]
        return float(d[:, free].mean())

    logEs = np.linspace(cfg.logE_lo, cfg.logE_hi, cfg.logE_n)
    results = {}
    Z = torch.zeros_like(snap_F)  # zero C for pure release
    excitations = {
        "v0_driven": dict(x0=rest_xyz, v0=torch.tensor(cfg.v0_kick, device=dev).float()[None].repeat(n, 1),
                          F0=I_mat[None].repeat(n, 1, 1), C0=torch.zeros(n, 3, 3, device=dev)),
        "F0_release": dict(x0=snap_x, v0=torch.zeros(n, 3, device=dev),
                           F0=snap_F, C0=torch.zeros(n, 3, 3, device=dev)),
    }
    for name, exc in excitations.items():
        gt = rollout(exc["x0"], exc["v0"], exc["F0"], exc["C0"], cfg.gt_logE)
        gt_motion = (gt[-1, free] - gt[0, free]).norm(dim=-1).mean().item()
        losses = []
        for le in logEs:
            pred = rollout(exc["x0"], exc["v0"], exc["F0"], exc["C0"], float(le))
            losses.append(traj_loss(pred, gt))
        losses = np.array(losses)
        results[name] = losses
        amin = logEs[int(losses.argmin())]
        # well width: logE range where loss < 2x min
        below = logEs[losses < 2 * losses.min() + 1e-12]
        width = (below.max() - below.min()) if below.size else float("nan")
        print(f"[f0E] {name}: GT motion {gt_motion:.4f}, argmin logE {amin:.3f} "
              f"(GT {cfg.gt_logE}), <2xmin width {width:.2f} dec, minloss {losses.min():.2e}")

    np.savez(os.path.join(out_dir, "landscape.npz"), logEs=logEs, **results)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, losses in results.items():
        ax[0].plot(logEs, losses, "-o", ms=3, label=name)
        nz = losses / losses.min()
        ax[1].plot(logEs, nz, "-o", ms=3, label=name)
    for a in ax:
        a.axvline(cfg.gt_logE, color="k", ls="--", lw=1, label="GT logE")
        a.set_xlabel("log10 E")
    ax[0].set_title("trajectory loss (abs)"); ax[0].set_yscale("log"); ax[0].legend()
    ax[1].set_title("loss / min (well shape)"); ax[1].set_ylim(0.8, 10); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "E_landscape.png"), dpi=120)
    plt.close(fig)
    print(f"[f0E] DONE -> {out_dir}  ({time.time()-t0:.1f}s)")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(F0ELandscapeConfig))
