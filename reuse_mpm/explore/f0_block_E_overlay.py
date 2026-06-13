"""Entrypoint: visualize the dynamic-pull block's release rollout at SEVERAL E
values, to SEE whether E=4.0/4.5/5.0/5.5/6.0 actually produce different motion
(they should -- especially with an initial deformation F0 != I).

Same pull -> snapshot(F0) as f0_block_landscape; then forward each E from the same
t0 (v0=0). Produces, all colored by E:
  divergence.png    -- per-frame mean particle dist vs GT(E=5.0), abs + /motion
  extent_profile.png-- per-frame block x-width / z-height (relaxation speed per E)
  overlay_frames.png-- panel of key frames, all E point clouds overlaid (xz)
  overlay.gif       -- per-frame xz overlay of all E
"""
from __future__ import annotations

import os
import time as _time
from dataclasses import dataclass
from typing import Tuple

import tyro

from ..gpu import pick_free_gpu


@dataclass
class BlockEOverlayConfig:
    nx: int = 22
    ny: int = 9
    nz: int = 16
    half: Tuple[float, float, float] = (0.18, 0.08, 0.14)
    z_base: float = 0.30
    pull_speed: float = 0.5
    pull_frames: int = 5
    grip_half_x: float = 0.045
    nu: float = 0.3
    K: int = 16
    E_list: Tuple[float, ...] = (4.0, 4.5, 5.0, 5.5, 6.0)  # logE; 5.0 = GT
    gt_logE: float = 5.0
    label: str = "block_dynpull_Eoverlay"


def run(cfg: BlockEOverlayConfig) -> str:
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
    from matplotlib.animation import FuncAnimation, PillowWriter

    from .._env import MPMStateStruct, MPMModelStruct, MPMWARPDiff
    from ..config import SimConfig

    t0 = _time.time()
    out_dir = os.path.join("outputs", "explore", "f0_block_E_overlay", cfg.label)
    os.makedirs(out_dir, exist_ok=True)
    sim = SimConfig(); dev = "cuda:0"; G, GL = sim.grid_size, sim.grid_lim
    hx, hy, hz = cfg.half
    cx, cy, cz = 0.5, 0.5, cfg.z_base + hz
    gx = torch.linspace(cx - hx, cx + hx, cfg.nx); gy = torch.linspace(cy - hy, cy + hy, cfg.ny)
    gz = torch.linspace(cz - hz, cz + hz, cfg.nz)
    X_rest = torch.stack(torch.meshgrid(gx, gy, gz, indexing="ij"), -1).reshape(-1, 3).to(dev)
    n = X_rest.shape[0]
    p_vol = torch.full((n,), float((2 * hx / max(cfg.nx - 1, 1)) ** 3), device=dev)
    eye = torch.eye(3, device=dev)

    def build():
        st = MPMStateStruct(); st.init(n, device=dev, requires_grad=False)
        st.from_torch(X_rest.clone(), p_vol, None, device=dev, requires_grad=False, n_grid=G, grid_lim=GL)
        md = MPMModelStruct(); md.init(n, device=dev, requires_grad=False)
        md.init_other_params(n_grid=G, grid_lim=GL, device=dev)
        sv = MPMWARPDiff(n, n_grid=G, grid_lim=GL, device=dev)
        sv.set_parameters_dict(md, st, {"material": sim.material, "g": [0.0, 0.0, 0.0],
                               "density": sim.density, "grid_v_damping_scale": sim.grid_v_damping_scale})
        st.reset_density(torch.full((n,), float(sim.density), device=dev).clone(),
                         torch.ones(n, device=dev).int(), dev, update_mass=True)
        return sv, st, md

    def setE(sv, md, st, logE):
        sv.set_E_nu_from_torch(md, torch.full((n,), float(10.0 ** logE), device=dev).clone(),
                               torch.full((n,), float(cfg.nu), device=dev).clone(), dev)
        sv.prepare_mu_lam(md, st, dev)

    # ---- pull -> snapshot F0 ----
    sv, st, md = build(); setE(sv, md, st, cfg.gt_logE)
    with torch.no_grad():
        st.continue_from_torch(X_rest.clone(), torch.zeros(n, 3, device=dev),
                               eye[None].repeat(n, 1, 1).contiguous(),
                               torch.zeros(n, 3, 3, device=dev), device=dev, requires_grad=False)
        et = cfg.pull_frames * sim.delta_t; gs = (cfg.grip_half_x, hy * 1.6, hz * 1.6)
        sv.enforce_particle_velocity_translation(st, point=(cx - hx, cy, cz), size=gs,
            velocity=(-cfg.pull_speed, 0, 0), start_time=0.0, end_time=et, device=dev)
        sv.enforce_particle_velocity_translation(st, point=(cx + hx, cy, cz), size=gs,
            velocity=(+cfg.pull_speed, 0, 0), start_time=0.0, end_time=et, device=dev)
        prev = st
        for _ in range(cfg.pull_frames):
            for _ in range(sim.substep):
                nx = prev.partial_clone(requires_grad=False)
                sv.p2g2p_differentiable(md, prev, nx, sim.substep_size, device=dev); prev = nx
        x_snap = wp.to_torch(prev.particle_x).clone(); F_snap = wp.to_torch(prev.particle_F_trial).clone()
    print(f"[Eov] snapshot F0 maxdev {(torch.linalg.svdvals(F_snap)-1).abs().max():.3f}; rolling {len(cfg.E_list)} E x {cfg.K}f")

    def rollout(logE):
        sv, st, md = build(); setE(sv, md, st, logE)
        with torch.no_grad():
            st.continue_from_torch(x_snap.clone(), torch.zeros(n, 3, device=dev), F_snap.clone(),
                                   torch.zeros(n, 3, 3, device=dev), device=dev, requires_grad=False)
            prev = st; out = [wp.to_torch(prev.particle_x).clone()]
            for _ in range(cfg.K):
                for _ in range(sim.substep):
                    nx = prev.partial_clone(requires_grad=False)
                    sv.p2g2p_differentiable(md, prev, nx, sim.substep_size, device=dev); prev = nx
                out.append(wp.to_torch(prev.particle_x).clone())
        return torch.stack(out).cpu().numpy()

    rolls = {le: rollout(le) for le in cfg.E_list}
    gt = rolls[cfg.gt_logE]
    gt_motion = np.linalg.norm(gt[1:] - gt[0:1], axis=-1).mean(1)            # [K]
    colors = {le: plt.cm.coolwarm((le - min(cfg.E_list)) / (max(cfg.E_list) - min(cfg.E_list))) for le in cfg.E_list}
    np.savez(os.path.join(out_dir, "rolls.npz"), **{f"E{le}": rolls[le] for le in cfg.E_list})

    # ---- divergence vs GT ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for le in cfg.E_list:
        if le == cfg.gt_logE: continue
        d = np.linalg.norm(rolls[le] - gt, axis=-1).mean(1)                  # [K+1]
        ax[0].plot(d, "-o", ms=3, color=colors[le], label=f"logE {le}")
        ax[1].plot(d[1:] / np.maximum(gt_motion, 1e-9), "-o", ms=3, color=colors[le], label=f"logE {le}")
    ax[0].set_title("mean particle dist vs GT(5.0)"); ax[0].set_xlabel("frame"); ax[0].legend()
    ax[1].axhline(0.05, color="k", ls="--", lw=0.7); ax[1].set_title("dist / GT-motion (visible if >> 0)")
    ax[1].set_xlabel("frame"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "divergence.png"), dpi=120); plt.close(fig)

    # ---- extent profile (x-width, z-height) per E ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for le in cfg.E_list:
        xw = rolls[le][:, :, 0].max(1) - rolls[le][:, :, 0].min(1)
        zh = rolls[le][:, :, 2].max(1) - rolls[le][:, :, 2].min(1)
        ax[0].plot(xw, "-o", ms=3, color=colors[le], label=f"logE {le}")
        ax[1].plot(zh, "-o", ms=3, color=colors[le], label=f"logE {le}")
    ax[0].set_title("block x-width vs frame"); ax[0].set_xlabel("frame"); ax[0].legend()
    ax[1].set_title("block z-height vs frame"); ax[1].set_xlabel("frame"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "extent_profile.png"), dpi=120); plt.close(fig)

    # ---- overlay panel: key frames, all E point clouds (xz) ----
    allx = np.concatenate([rolls[le].reshape(-1, 3) for le in cfg.E_list])
    mn, mx = allx.min(0), allx.max(0)
    kf = [f for f in (0, cfg.K // 3, 2 * cfg.K // 3, cfg.K) if f <= cfg.K]
    fig, axs = plt.subplots(1, len(kf), figsize=(4.2 * len(kf), 4.4))
    for axp, fr in zip(np.atleast_1d(axs), kf):
        for le in cfg.E_list:
            axp.scatter(rolls[le][fr][:, 0], rolls[le][fr][:, 2], s=4, color=colors[le],
                        alpha=0.5, label=f"logE {le}" if fr == kf[0] else None)
        axp.set_xlim(mn[0], mx[0]); axp.set_ylim(mn[2], mx[2]); axp.set_aspect("equal")
        axp.set_title(f"frame {fr}"); axp.set_xlabel("x"); axp.set_ylabel("z")
    np.atleast_1d(axs)[0].legend(fontsize=8)
    fig.suptitle("E rollouts overlaid (xz); blue=soft .. red=stiff", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "overlay_frames.png"), dpi=120); plt.close(fig)

    # ---- overlay gif ----
    fig, axg = plt.subplots(figsize=(6, 5))

    def draw(fr):
        axg.cla()
        for le in cfg.E_list:
            axg.scatter(rolls[le][fr][:, 0], rolls[le][fr][:, 2], s=5, color=colors[le], alpha=0.5,
                        label=f"logE {le}")
        axg.set_xlim(mn[0], mx[0]); axg.set_ylim(mn[2], mx[2]); axg.set_aspect("equal")
        axg.set_title(f"frame {fr}/{cfg.K} (xz)"); axg.legend(fontsize=7, loc="upper right")
        return ()

    draw(0)
    anim = FuncAnimation(fig, draw, frames=cfg.K + 1, blit=False)
    anim.save(os.path.join(out_dir, "overlay.gif"), writer=PillowWriter(fps=5)); plt.close(fig)

    # numeric summary
    print("[Eov] mean dist vs GT(5.0) at last frame, and as % of GT motion:")
    for le in cfg.E_list:
        if le == cfg.gt_logE: continue
        d = np.linalg.norm(rolls[le][-1] - gt[-1], axis=-1).mean()
        print(f"    logE {le}: {d:.4f}  ({d/gt_motion[-1]*100:.0f}% of motion {gt_motion[-1]:.4f})")
    print(f"[Eov] DONE -> {out_dir} ({_time.time()-t0:.1f}s)")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(BlockEOverlayConfig))
