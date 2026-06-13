"""Forward-only viz for the symmetric-pull squeeze scene (what the sweep SHOULD have
dumped). For each release frame R: run the two-end pull for R frames, snapshot F0,
then release (zero v0) for K frames at GT E. Save:

  pull_release_panel_R{R}.png  selected frames (pull + release), colored by stretch
  width_R{R}.png               width(t) over the release (the observable the fit sees)
  F0_stretch_R{R}.png          per-particle stretch |sigma-1| at the snapshot (the F0)
  traj_R{R}.npz                X[T,n,3], stretch[T,n], maxdev, release_frame

Run pinned to a chosen GPU (default: let pick_free_gpu choose). To share the GPU the
sweep already holds (avoid the 2-GPU quota penalty), launch with CUDA_VISIBLE_DEVICES
set to that physical card.
"""
from __future__ import annotations

import os
import time as _time
from dataclasses import dataclass
from typing import Tuple

import tyro

from ..gpu import pick_free_gpu


@dataclass
class FwdVizConfig:
    nx: int = 22
    ny: int = 9
    nz: int = 16
    half: Tuple[float, float, float] = (0.18, 0.08, 0.14)
    z_base: float = 0.30
    pull_speed: float = 0.5
    grip_half_x: float = 0.045
    release_frames: Tuple[int, ...] = (2, 3, 5)
    gt_logE: float = 4.5
    nu: float = 0.3
    K: int = 32
    label: str = "block_squeeze_sweep"   # share the sweep's out dir


def run(cfg: FwdVizConfig) -> str:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        from ..gpu import assert_gpu_quota
        assert_gpu_quota(8.0)
        print(f"[fwd] preset CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
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
    out_dir = os.path.join("outputs", "explore", "f0_block_squeeze_sweep", cfg.label)
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

    def stretch_of(F):  # |sigma-1| max over singular values, per particle
        return (torch.linalg.svdvals(F) - 1.0).abs().amax(1)

    sv, st, md = build()
    z3 = torch.zeros(n, 3, device=dev); z33 = torch.zeros(n, 3, 3, device=dev)

    def forward_R(R):
        # ---- pull phase: record every frame ----
        setE(sv, md, st, cfg.gt_logE); sv.time = 0.0
        Xs, Sts, phase = [], [], []
        with torch.no_grad():
            st.continue_from_torch(X_rest.clone(), z3, eye[None].repeat(n, 1, 1).contiguous(),
                                   z33, device=dev, requires_grad=False)
            et = R * sim.delta_t; gs = (cfg.grip_half_x, hy * 1.6, hz * 1.6)
            sv.enforce_particle_velocity_translation(st, point=(cx - hx, cy, cz), size=gs,
                velocity=(-cfg.pull_speed, 0, 0), start_time=0.0, end_time=et, device=dev)
            sv.enforce_particle_velocity_translation(st, point=(cx + hx, cy, cz), size=gs,
                velocity=(+cfg.pull_speed, 0, 0), start_time=0.0, end_time=et, device=dev)
            prev = st
            Xs.append(wp.to_torch(prev.particle_x).clone()); Sts.append(torch.zeros(n, device=dev)); phase.append("pull")
            for _ in range(R):
                for _ in range(sim.substep):
                    nx = prev.partial_clone(requires_grad=False)
                    sv.p2g2p_differentiable(md, prev, nx, sim.substep_size, device=dev); prev = nx
                Xs.append(wp.to_torch(prev.particle_x).clone())
                Sts.append(stretch_of(wp.to_torch(prev.particle_F_trial)).clone()); phase.append("pull")
            x_snap = wp.to_torch(prev.particle_x).clone(); F_snap = wp.to_torch(prev.particle_F_trial).clone()
            maxdev = float((x_snap - X_rest).norm(dim=1).max())
            F0_stretch = stretch_of(F_snap).clone()

            # ---- release phase: zero v0 from the snapshot ----
            setE(sv, md, st, cfg.gt_logE); sv.time = 0.0
            st.continue_from_torch(x_snap.clone(), z3, F_snap.clone(), z33, device=dev, requires_grad=False)
            prev = st
            for _ in range(cfg.K):
                for _ in range(sim.substep):
                    nx = prev.partial_clone(requires_grad=False)
                    sv.p2g2p_differentiable(md, prev, nx, sim.substep_size, device=dev); prev = nx
                Xs.append(wp.to_torch(prev.particle_x).clone())
                Sts.append(stretch_of(wp.to_torch(prev.particle_F_trial)).clone()); phase.append("release")
        X = torch.stack(Xs).cpu().numpy(); S = torch.stack(Sts).cpu().numpy()
        return X, S, np.array(phase), maxdev, F0_stretch.cpu().numpy(), R

    for R in cfg.release_frames:
        X, S, phase, maxdev, F0s, _ = forward_R(R)
        width = X[:, :, 0].max(1) - X[:, :, 0].min(1)
        rel0 = int((phase == "pull").sum())  # index where release starts (== R+1)
        print(f"[fwd] R={R} maxdev={maxdev:.4f} F0_stretch mean {F0s.mean():.3f}/max {F0s.max():.3f} "
              f"width: rest {width[0]:.3f} -> snap {width[rel0-1]:.3f}")
        np.savez(os.path.join(out_dir, f"traj_R{R}.npz"), X=X, stretch=S, phase=phase,
                 width=width, maxdev=maxdev, release_frame=R, rel_start=rel0)

        # ---- panel: pull frames + sampled release frames (xz side view), colored by stretch ----
        npull = rel0
        rel_idx = list(range(rel0, len(X), max(1, (len(X) - rel0) // 7)))[:8 - 0]
        sel = list(range(npull)) + rel_idx
        sel = sel[:12]
        vmax = float(np.quantile(S, 0.98)) or 1e-3
        mn = X[:, :, [0, 2]].reshape(-1, 2).min(0); mx = X[:, :, [0, 2]].reshape(-1, 2).max(0)
        ncol = 6; nrow = (len(sel) + ncol - 1) // ncol
        fig, axs = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow), squeeze=False)
        for ax in axs.flat:
            ax.axis("off")
        for a, f in enumerate(sel):
            ax = axs.flat[a]; ax.axis("on")
            sc = ax.scatter(X[f][:, 0], X[f][:, 2], c=S[f], s=6, cmap="inferno", vmin=0, vmax=vmax)
            ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_aspect("equal")
            tag = "PULL" if f < npull else "REL"
            ax.set_title(f"f{f} [{tag}] w={width[f]:.3f}", fontsize=9)
            ax.set_xlabel("x"); ax.set_ylabel("z")
        fig.colorbar(sc, ax=axs, shrink=0.6, label="stretch |sigma-1|")
        fig.suptitle(f"R={R}  pull->release (maxdev {maxdev:.3f}, GT logE {cfg.gt_logE})", fontsize=13)
        fig.savefig(os.path.join(out_dir, f"pull_release_panel_R{R}.png"), dpi=110); plt.close(fig)

        # ---- width(t) over the whole sequence ----
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(width, "-o", ms=3)
        ax.axvline(rel0 - 1, color="orange", ls="--", label=f"release (f{rel0-1})")
        ax.set_xlabel("frame"); ax.set_ylabel("x-extent (width)")
        ax.set_title(f"R={R}  width(t): pull then breathing release (maxdev {maxdev:.3f})")
        ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"width_R{R}.png"), dpi=120); plt.close(fig)

        # ---- F0 stretch distribution (the snapshot) ----
        fig, ax = plt.subplots(figsize=(6.5, 4))
        snap = X[rel0 - 1]
        sc = ax.scatter(snap[:, 0], snap[:, 2], c=F0s, s=10, cmap="viridis")
        ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("z")
        ax.set_title(f"R={R} F0 snapshot stretch (mean {F0s.mean():.3f}, max {F0s.max():.3f})")
        fig.colorbar(sc, ax=ax, label="|sigma-1|")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"F0_stretch_R{R}.png"), dpi=120); plt.close(fig)

    print(f"[fwd] DONE -> {out_dir} ({_time.time()-t0:.1f}s)")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(FwdVizConfig))
