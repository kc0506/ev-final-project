"""Entrypoint: E landscape for the DYNAMIC-PULL block (realistic non-uniform F0).

Pipeline (block_dynpull as the target):
  1. build block, two-end particle-grip pull for `pull_frames` -> snapshot at the
     release frame = a realistic, NON-uniform, self-consistent F0 (not hand-set).
  2. snapshot t0 = (x@release, F_trial@release, v0=0, C0=0)  -- pure elastic release.
  3. GT = forward from t0 at true E.
  4. fix F0, scan logE, trajectory loss vs GT  -> the pure-E well.

Question: does this bigger, realistic strain give a SHARPER E well than telephone's
thin cord? (compare loss/motion^2 at GT +- offsets vs f0_e_landscape.)

(alpha,E) 2D is the next step; here we do the 1D pure-E baseline first.
Output under outputs/explore/f0_block_landscape/.
"""
from __future__ import annotations

import os
import time as _time
from dataclasses import dataclass
from typing import Tuple

import tyro

from ..gpu import pick_free_gpu


@dataclass
class BlockLandscapeConfig:
    nx: int = 22
    ny: int = 9
    nz: int = 16
    half: Tuple[float, float, float] = (0.18, 0.08, 0.14)
    z_base: float = 0.30
    pull_speed: float = 0.5
    pull_frames: int = 5            # release frame = snapshot
    grip_half_x: float = 0.045
    gt_logE: float = 5.0
    nu: float = 0.3
    K: int = 12                     # landscape horizon (frames after release)
    logE_lo: float = 3.5
    logE_hi: float = 6.5
    logE_n: int = 31
    label: str = "block_dynpull_Elandscape"


def run(cfg: BlockLandscapeConfig) -> str:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        from ..gpu import assert_gpu_quota
        assert_gpu_quota(8.0)
        print(f"[gpu] preset CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        pick_free_gpu()
    import numpy as np
    import torch
    import warp as wp
    wp.init()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .._env import MPMStateStruct, MPMModelStruct, MPMWARPDiff
    from ..config import SimConfig

    t0 = _time.time()
    out_dir = os.path.join("outputs", "explore", "f0_block_landscape", cfg.label)
    os.makedirs(out_dir, exist_ok=True)
    sim = SimConfig()
    dev = "cuda:0"
    G, GL = sim.grid_size, sim.grid_lim
    hx, hy, hz = cfg.half
    cx, cy, cz = 0.5, 0.5, cfg.z_base + hz
    gx = torch.linspace(cx - hx, cx + hx, cfg.nx)
    gy = torch.linspace(cy - hy, cy + hy, cfg.ny)
    gz = torch.linspace(cz - hz, cz + hz, cfg.nz)
    X_rest = torch.stack(torch.meshgrid(gx, gy, gz, indexing="ij"), -1).reshape(-1, 3).to(dev)
    n = X_rest.shape[0]
    p_vol = torch.full((n,), float((2 * hx / max(cfg.nx - 1, 1)) ** 3), device=dev)
    eye = torch.eye(3, device=dev)

    def build():
        state = MPMStateStruct(); state.init(n, device=dev, requires_grad=False)
        state.from_torch(X_rest.clone(), p_vol, None, device=dev, requires_grad=False, n_grid=G, grid_lim=GL)
        model = MPMModelStruct(); model.init(n, device=dev, requires_grad=False)
        model.init_other_params(n_grid=G, grid_lim=GL, device=dev)
        solver = MPMWARPDiff(n, n_grid=G, grid_lim=GL, device=dev)
        solver.set_parameters_dict(model, state, {"material": sim.material, "g": [0.0, 0.0, 0.0],
                                   "density": sim.density, "grid_v_damping_scale": sim.grid_v_damping_scale})
        dens = torch.full((n,), float(sim.density), device=dev)
        state.reset_density(dens.clone(), torch.ones_like(dens).int(), dev, update_mass=True)
        return solver, state, model

    def set_E(solver, model, state, logE):
        solver.set_E_nu_from_torch(model, torch.full((n,), float(10.0 ** logE), device=dev).clone(),
                                   torch.full((n,), float(cfg.nu), device=dev).clone(), dev)
        solver.prepare_mu_lam(model, state, dev)

    # ---- phase 1: dynamic pull -> snapshot at release frame ----
    solver, state, model = build()
    set_E(solver, model, state, cfg.gt_logE)
    with torch.no_grad():
        state.continue_from_torch(X_rest.clone(), torch.zeros(n, 3, device=dev),
                                  eye[None].repeat(n, 1, 1).contiguous(),
                                  torch.zeros(n, 3, 3, device=dev), device=dev, requires_grad=False)
        end_t = cfg.pull_frames * sim.delta_t
        gsize = (cfg.grip_half_x, hy * 1.6, hz * 1.6)
        solver.enforce_particle_velocity_translation(state, point=(cx - hx, cy, cz), size=gsize,
            velocity=(-cfg.pull_speed, 0, 0), start_time=0.0, end_time=end_t, device=dev)
        solver.enforce_particle_velocity_translation(state, point=(cx + hx, cy, cz), size=gsize,
            velocity=(+cfg.pull_speed, 0, 0), start_time=0.0, end_time=end_t, device=dev)
        prev = state
        for _ in range(cfg.pull_frames):
            for _ in range(sim.substep):
                nxt = prev.partial_clone(requires_grad=False)
                solver.p2g2p_differentiable(model, prev, nxt, sim.substep_size, device=dev); prev = nxt
        x_snap = wp.to_torch(prev.particle_x).clone()
        F_snap = wp.to_torch(prev.particle_F_trial).clone()
    sig = torch.linalg.svdvals(F_snap)
    print(f"[blockL] snapshot @ release f{cfg.pull_frames}: maxdev "
          f"{(sig-1).abs().max():.3f}, mean {(sig-1).abs().max(1).values.mean():.3f}, "
          f"std/mean {(sig-1).abs().max(1).values.std()/(sig-1).abs().max(1).values.mean():.2f}")

    # ---- phase 2: pure-E landscape (fix F0=snapshot, v0=0 release) ----
    def rollout(logE):
        solver, state, model = build()
        set_E(solver, model, state, logE)
        with torch.no_grad():
            state.continue_from_torch(x_snap.clone(), torch.zeros(n, 3, device=dev),
                                      F_snap.clone(), torch.zeros(n, 3, 3, device=dev),
                                      device=dev, requires_grad=False)
            prev = state; out = [wp.to_torch(prev.particle_x).clone()]
            for _ in range(cfg.K):
                for _ in range(sim.substep):
                    nxt = prev.partial_clone(requires_grad=False)
                    solver.p2g2p_differentiable(model, prev, nxt, sim.substep_size, device=dev); prev = nxt
                out.append(wp.to_torch(prev.particle_x).clone())
        return torch.stack(out)

    def chamfer(pred, gt):  # symmetric CD over frames>0 (no correspondence used)
        tot = 0.0
        for fr in range(1, pred.shape[0]):
            D = torch.cdist(pred[fr], gt[fr])           # [n,n]
            tot += float(D.min(0).values.mean() + D.min(1).values.mean())
        return tot / (pred.shape[0] - 1)

    gt = rollout(cfg.gt_logE)
    gt_motion = (gt[-1] - gt[0]).norm(dim=-1).mean().item()
    logEs = np.linspace(cfg.logE_lo, cfg.logE_hi, cfg.logE_n)
    losses, losses_cd = [], []
    for le in logEs:
        pred = rollout(float(le))
        losses.append(float(((pred[1:] - gt[1:]) ** 2).sum(-1).mean()))   # pairwise (sq dist)
        losses_cd.append(chamfer(pred, gt))                                # chamfer
    losses = np.array(losses); losses_cd = np.array(losses_cd)
    m2 = gt_motion ** 2
    np.savez(os.path.join(out_dir, "Eland.npz"), logEs=logEs, losses=losses,
             losses_cd=losses_cd, gt_motion=gt_motion, gt_logE=cfg.gt_logE)
    amin = logEs[int(losses.argmin())]; amin_cd = logEs[int(losses_cd.argmin())]
    print(f"[blockL] argmin: pairwise logE {amin:.3f}, chamfer logE {amin_cd:.3f} (GT {cfg.gt_logE})")
    f = lambda x: np.interp(x, logEs, losses)
    print(f"[blockL] GT motion {gt_motion:.4f} (telephone f0release was 0.011); argmin logE "
          f"{amin:.3f} (GT {cfg.gt_logE}); minloss {losses.min():.2e}")
    tele_ref = {-0.5: 0.19, 0.5: 1.40, 1.0: 14.0}  # telephone f0release loss/motion^2
    print(f"[blockL] well sharpness loss/motion^2 at GT offsets (vs telephone f0release):")
    for off in (-0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 1.0):
        ref = tele_ref.get(off, "-")
        print(f"    {off:+.1f}: {f(cfg.gt_logE+off)/m2:.3f}   (telephone: {ref})")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(logEs, losses, "-o", ms=3, color="tab:blue", label="pairwise (sq dist)")
    ax[0].plot(logEs, losses_cd, "-o", ms=3, color="tab:orange", label="chamfer")
    ax[0].axvline(cfg.gt_logE, color="k", ls="--"); ax[0].set_yscale("log")
    ax[0].set_title("E landscape: pairwise vs chamfer (abs)"); ax[0].set_xlabel("log10 E"); ax[0].legend()
    # well-shape comparison: each normalized to its own min (units differ)
    ax[1].plot(logEs, losses / losses.min(), "-o", ms=3, color="tab:blue", label="pairwise")
    ax[1].plot(logEs, losses_cd / losses_cd.min(), "-o", ms=3, color="tab:orange", label="chamfer")
    ax[1].axvline(cfg.gt_logE, color="k", ls="--"); ax[1].set_yscale("log")
    ax[1].set_title("loss / own-min (well sharpness)"); ax[1].set_xlabel("log10 E"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "E_landscape.png"), dpi=120); plt.close(fig)
    print(f"[blockL] DONE -> {out_dir} ({_time.time()-t0:.1f}s)")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(BlockLandscapeConfig))
