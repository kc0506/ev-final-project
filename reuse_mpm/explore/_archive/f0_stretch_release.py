"""Entrypoint: EXTREME clean stretch-release forward viz (no backward yet).

A simple block of particles, base layer frozen ("glued to the ground"), given a
large horizontal pre-stretch F0 = diag(s, 1/sqrt(s), 1/sqrt(s)) with v0=0, then
RELEASED. We watch the deformation + motion: snap-back, overshoot, oscillation.
Strain concentrates near the clamp, so max deformation is usually a few frames in.

Motivation (2026-06-12): telephone's thin cord gives a weak/ambiguous E signal
(joint E,alpha near-degenerate within 8 frames). A chunky shape with a big
pre-strain release makes deformation + stress -- hence the E signal -- obvious.
This is the forward sanity check before committing to a recovery experiment.

Outputs (under outputs/explore/f0_stretch_release/<label>/):
  traj_3d_triplane.gif  -- 3D + xy/xz/yz projections, particles colored by stretch
  strain_motion.png     -- per-frame mean strain & motion, max-deformation frame
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Tuple

import tyro

from ..gpu import pick_free_gpu


@dataclass
class StretchReleaseConfig:
    nx: int = 22
    ny: int = 9
    nz: int = 16
    # rest block half-extents (normalized sim coords); centered at (0.5,0.5,zc)
    half: Tuple[float, float, float] = (0.18, 0.08, 0.16)
    z_base: float = 0.22                 # block bottom (just above ground)
    stretch: float = 1.4                 # horizontal (x) stretch factor; >1 = pulled apart
    freeze_frac: float = 0.12            # bottom z-fraction frozen ("on the ground")
    logE: float = 5.0
    nu: float = 0.3
    num_frames: int = 32
    label: str = "block_stretch1p4_release"


def run(cfg: StretchReleaseConfig) -> str:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        from ..gpu import assert_gpu_quota
        assert_gpu_quota(8.0)
        print(f"[gpu] using preset CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        pick_free_gpu()
    import numpy as np
    import torch
    import warp as wp
    wp.init()  # we bypass build_mpm (which calls _ensure_warp), so init warp ourselves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    from .._env import MPMStateStruct, MPMModelStruct, MPMWARPDiff
    from ..config import SimConfig

    t0 = time.time()
    out_dir = os.path.join("outputs", "explore", "f0_stretch_release", cfg.label)
    os.makedirs(out_dir, exist_ok=True)
    sim = SimConfig(); sim.num_frames = cfg.num_frames
    dev = "cuda:0"
    G, GL = sim.grid_size, sim.grid_lim

    # ---- simple block of particles, centered at (0.5, 0.5, z_base+half_z) ----
    hx, hy, hz = cfg.half
    cx, cy, cz = 0.5, 0.5, cfg.z_base + hz
    gx = torch.linspace(cx - hx, cx + hx, cfg.nx)
    gy = torch.linspace(cy - hy, cy + hy, cfg.ny)
    gz = torch.linspace(cz - hz, cz + hz, cfg.nz)
    X_rest = torch.stack(torch.meshgrid(gx, gy, gz, indexing="ij"), -1).reshape(-1, 3).to(dev)
    n = X_rest.shape[0]
    spacing = (2 * hx / max(cfg.nx - 1, 1))
    p_vol = torch.full((n,), float(spacing ** 3), device=dev)

    # ---- pre-stretch about the block center (volume-preserving) ----
    s = cfg.stretch
    A = torch.diag(torch.tensor([s, s ** -0.5, s ** -0.5], device=dev))   # [3,3]
    c = torch.tensor([cx, cy, cz], device=dev)
    X0 = c + (X_rest - c) @ A.T                          # stretched positions
    F0 = A[None].repeat(n, 1, 1).contiguous()            # uniform F0 = A (consistent)
    v0 = torch.zeros(n, 3, device=dev)
    C0 = torch.zeros(n, 3, 3, device=dev)
    frozen = X_rest[:, 2] < (cfg.z_base + cfg.freeze_frac * 2 * hz)       # base slab
    print(f"[stretch] n={n}, stretch x{s} (x-extent {2*hx:.3f}->{2*hx*s:.3f}), "
          f"frozen base {int(frozen.sum())}; bbox after stretch "
          f"x[{X0[:,0].min():.3f},{X0[:,0].max():.3f}] (wall-safe in [{2/G:.3f},{GL-2/G:.3f}])")

    # ---- build warp solver/state/model (mirrors sim_render.build_mpm) ----
    state = MPMStateStruct(); state.init(n, device=dev, requires_grad=False)
    state.from_torch(X0.clone(), p_vol, None, device=dev, requires_grad=False,
                     n_grid=G, grid_lim=GL)
    model = MPMModelStruct(); model.init(n, device=dev, requires_grad=False)
    model.init_other_params(n_grid=G, grid_lim=GL, device=dev)
    solver = MPMWARPDiff(n, n_grid=G, grid_lim=GL, device=dev)
    solver.set_parameters_dict(model, state, {"material": sim.material, "g": [0.0, 0.0, 0.0],
                               "density": sim.density, "grid_v_damping_scale": sim.grid_v_damping_scale})
    # freeze base via 27-node stencil grid mask (same recipe as build_mpm)
    inv_dx = G / GL
    base = (X0[frozen] * inv_dx - 0.5).to(torch.int64)
    fg = torch.zeros((G, G, G), dtype=torch.int32, device=dev)
    for di in range(3):
        for dj in range(3):
            for dk in range(3):
                fg[(base[:, 0] + di).clamp(0, G - 1), (base[:, 1] + dj).clamp(0, G - 1),
                   (base[:, 2] + dk).clamp(0, G - 1)] = 1
    solver.enforce_grid_velocity_by_mask(fg)

    # ---- forward release ----
    dens = torch.full((n,), float(sim.density), device=dev)
    state.reset_density(dens.clone(), torch.ones_like(dens).int(), dev, update_mass=True)
    with torch.no_grad():
        E_t = torch.full((n,), float(10.0 ** cfg.logE), device=dev)
        nu_t = torch.full((n,), float(cfg.nu), device=dev)
        solver.set_E_nu_from_torch(model, E_t.clone(), nu_t.clone(), dev)
        solver.prepare_mu_lam(model, state, dev)
        state.continue_from_torch(X0.clone(), v0, F0, C0, device=dev, requires_grad=False)
        prev = state
        xs = [X0.clone()]
        Fs = [F0.clone()]
        for _ in range(cfg.num_frames - 1):
            for _ in range(sim.substep):
                nxt = prev.partial_clone(requires_grad=False)
                solver.p2g2p_differentiable(model, prev, nxt, sim.substep_size, device=dev)
                prev = nxt
            xs.append(wp.to_torch(prev.particle_x).clone())
            Fs.append(wp.to_torch(prev.particle_F_trial).clone())

    X = torch.stack(xs).cpu().numpy()                    # [T,n,3]
    free = (~frozen).cpu().numpy()
    # per-particle stretch magnitude max|sigma-1| from F_trial
    strain = []
    for F in Fs:
        sig = torch.linalg.svdvals(F)
        strain.append((sig - 1.0).abs().max(dim=1).values.cpu().numpy())
    strain = np.stack(strain)                            # [T,n]
    motion = np.linalg.norm(X[1:] - X[0:1], axis=-1)     # [T-1,n] vs t0
    mean_strain = strain[:, free].mean(1)
    mean_motion = np.concatenate([[0.0], motion[:, free].mean(1)])
    max_def_frame = int(mean_strain.argmax())
    print(f"[stretch] mean strain per frame: f0 {mean_strain[0]:.3f}, "
          f"max {mean_strain.max():.3f} @ frame {max_def_frame}; "
          f"motion peak {mean_motion.max():.4f}")

    np.savez(os.path.join(out_dir, "traj.npz"), X=X, strain=strain,
             frozen=frozen.cpu().numpy(), mean_strain=mean_strain, mean_motion=mean_motion)

    # ---- first-8-frame grid panel (xz side-view, colored by stretch) ----
    vmax_p = float(np.quantile(strain[:, free], 0.98))
    mn, mx = X[:8].reshape(-1, 3).min(0), X[:8].reshape(-1, 3).max(0)
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for f, axp in enumerate(axs.flat):
        if f >= min(8, cfg.num_frames):
            axp.axis("off"); continue
        sc = axp.scatter(X[f][:, 0], X[f][:, 2], c=strain[f], s=7, cmap="inferno",
                         vmin=0, vmax=vmax_p)
        axp.set_xlim(mn[0], mx[0]); axp.set_ylim(mn[2], mx[2]); axp.set_aspect("equal")
        tag = "  MAXDEF" if f == max_def_frame else ""
        axp.set_title(f"frame {f}  strain {mean_strain[f]:.2f}  motion {mean_motion[f]:.3f}{tag}")
        axp.set_xlabel("x"); axp.set_ylabel("z")
    fig.colorbar(sc, ax=axs, shrink=0.6, label="stretch |sigma-1|")
    fig.suptitle(f"stretch x{cfg.stretch} release (xz side-view), E=1e{cfg.logE:g}", fontsize=13)
    fig.savefig(os.path.join(out_dir, "first8_panel.png"), dpi=110); plt.close(fig)

    # ---- strain & motion timeseries ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(mean_strain, "-o", ms=3); ax[0].axvline(max_def_frame, color="r", ls="--",
              label=f"max deformation @ {max_def_frame}")
    ax[0].set_title("mean stretch |sigma-1| (free)"); ax[0].set_xlabel("frame"); ax[0].legend()
    ax[1].plot(mean_motion, "-o", ms=3, color="tab:green")
    ax[1].set_title("mean motion vs t0 (free)"); ax[1].set_xlabel("frame")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "strain_motion.png"), dpi=120); plt.close(fig)

    # ---- 3D + triplane gif, colored by stretch ----
    vmax = float(np.quantile(strain[:, free], 0.98))
    mins, maxs = X.reshape(-1, 3).min(0), X.reshape(-1, 3).max(0)
    fig = plt.figure(figsize=(11, 9))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    axes2d = [fig.add_subplot(2, 2, k) for k in (2, 3, 4)]
    proj = [(0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z")]

    def draw(f):
        ax3d.cla()
        sc = ax3d.scatter(X[f][:, 0], X[f][:, 1], X[f][:, 2], c=strain[f], s=4,
                          cmap="inferno", vmin=0, vmax=vmax)
        ax3d.set_xlim(mins[0], maxs[0]); ax3d.set_ylim(mins[1], maxs[1]); ax3d.set_zlim(mins[2], maxs[2])
        tag = "  <-- MAX DEF" if f == max_def_frame else ""
        ax3d.set_title(f"frame {f}/{cfg.num_frames-1}  strain {mean_strain[f]:.2f}{tag}")
        for ax2, (a, b, la, lb) in zip(axes2d, proj):
            ax2.cla()
            ax2.scatter(X[f][:, a], X[f][:, b], c=strain[f], s=5, cmap="inferno", vmin=0, vmax=vmax)
            ax2.set_xlim(mins[a], maxs[a]); ax2.set_ylim(mins[b], maxs[b])
            ax2.set_xlabel(la); ax2.set_ylabel(lb); ax2.set_title(f"{la}{lb} plane"); ax2.set_aspect("equal")
        return ()

    draw(0)
    fig.colorbar(ax3d.collections[0], ax=ax3d, shrink=0.6, label="stretch |sigma-1|")
    anim = FuncAnimation(fig, draw, frames=cfg.num_frames, blit=False)
    gif = os.path.join(out_dir, "traj_3d_triplane.gif")
    anim.save(gif, writer=PillowWriter(fps=6)); plt.close(fig)
    print(f"[stretch] DONE -> {out_dir} ({time.time()-t0:.1f}s); max-def frame {max_def_frame}, gif {gif}")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(StretchReleaseConfig))
