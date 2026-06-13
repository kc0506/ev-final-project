"""Entrypoint: ASYMMETRIC squeeze-release forward viz.

Block resting on a slip floor. A downward grip presses a region at x~0.6 (right of
center) for `push_frames`, then releases. Expectation: the right side flattens
(compressed against the floor), the left side rises (incompressible material flows
left/up) -- an asymmetric deformation, unlike the symmetric two-end pull.

Uses the same warp particle-level grip (enforce_particle_velocity_translation) +
a slip surface collider as the floor. Forward-only; reports a left-vs-right height
asymmetry so "right flat / left high" is a number, not an eyeball.

Outputs (outputs/explore/f0_asym_squeeze/<label>/):
  traj_3d_triplane.gif, first8_panel.png, asym_height.png, traj.npz
"""
from __future__ import annotations

import os
import time as _time
from dataclasses import dataclass
from typing import Tuple

import tyro

from ..gpu import pick_free_gpu


@dataclass
class AsymSqueezeConfig:
    nx: int = 22
    ny: int = 9
    nz: int = 16
    half: Tuple[float, float, float] = (0.18, 0.08, 0.14)
    z_base: float = 0.30
    push_x: float = 0.60              # center of the downward grip (right of 0.5)
    push_half_x: float = 0.07
    push_half_z: float = 0.045        # grips the top slab of the right region
    push_speed: float = 0.45          # downward grip speed
    push_frames: int = 5
    logE: float = 5.0
    nu: float = 0.3
    num_frames: int = 32
    label: str = "block_asym_squeeze"


def run(cfg: AsymSqueezeConfig) -> str:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        from ..gpu import assert_gpu_quota
        assert_gpu_quota(8.0)
        print(f"[gpu] using preset CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
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
    out_dir = os.path.join("outputs", "explore", "f0_asym_squeeze", cfg.label)
    os.makedirs(out_dir, exist_ok=True)
    sim = SimConfig(); sim.num_frames = cfg.num_frames
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
    F0 = torch.eye(3, device=dev)[None].repeat(n, 1, 1).contiguous()
    v0 = torch.zeros(n, 3, device=dev); C0 = torch.zeros(n, 3, 3, device=dev)
    print(f"[asym] n={n}; push down @ x~{cfg.push_x} (right), {cfg.push_frames}f @ -{cfg.push_speed}; "
          f"floor(slip) z={cfg.z_base}")

    state = MPMStateStruct(); state.init(n, device=dev, requires_grad=False)
    state.from_torch(X_rest.clone(), p_vol, None, device=dev, requires_grad=False, n_grid=G, grid_lim=GL)
    model = MPMModelStruct(); model.init(n, device=dev, requires_grad=False)
    model.init_other_params(n_grid=G, grid_lim=GL, device=dev)
    solver = MPMWARPDiff(n, n_grid=G, grid_lim=GL, device=dev)
    solver.set_parameters_dict(model, state, {"material": sim.material, "g": [0.0, 0.0, 0.0],
                               "density": sim.density, "grid_v_damping_scale": sim.grid_v_damping_scale})
    # slip floor at the block base so the squeezed material can flow laterally/up
    solver.add_surface_collider(point=(0.0, 0.0, cfg.z_base), normal=(0.0, 0.0, 1.0),
                                surface="slip", friction=0.0)

    end_t = cfg.push_frames * sim.delta_t
    dens = torch.full((n,), float(sim.density), device=dev)
    state.reset_density(dens.clone(), torch.ones_like(dens).int(), dev, update_mass=True)
    with torch.no_grad():
        solver.set_E_nu_from_torch(model, torch.full((n,), float(10.0 ** cfg.logE), device=dev).clone(),
                                   torch.full((n,), float(cfg.nu), device=dev).clone(), dev)
        solver.prepare_mu_lam(model, state, dev)
        state.continue_from_torch(X_rest.clone(), v0, F0, C0, device=dev, requires_grad=False)
        # downward grip on the right-top region (particle-level), released after end_t
        solver.enforce_particle_velocity_translation(
            state, point=(cfg.push_x, cy, cz + hz), size=(cfg.push_half_x, hy * 1.6, cfg.push_half_z),
            velocity=(0.0, 0.0, -cfg.push_speed), start_time=0.0, end_time=end_t, device=dev)
        prev = state
        xs = [X_rest.clone()]; Fs = [F0.clone()]
        for _ in range(cfg.num_frames - 1):
            for _ in range(sim.substep):
                nxt = prev.partial_clone(requires_grad=False)
                solver.p2g2p_differentiable(model, prev, nxt, sim.substep_size, device=dev)
                prev = nxt
            xs.append(wp.to_torch(prev.particle_x).clone())
            Fs.append(wp.to_torch(prev.particle_F_trial).clone())

    X = torch.stack(xs).cpu().numpy()
    strain = np.stack([(torch.linalg.svdvals(F) - 1.0).abs().max(dim=1).values.cpu().numpy() for F in Fs])
    mean_strain = strain.mean(1)
    release = cfg.push_frames
    # left/right top-height asymmetry: max-z of left half vs right half per frame
    x0 = X[0][:, 0]
    L, R = x0 < 0.5, x0 > 0.5
    topz_L = X[:, L, 2].max(1); topz_R = X[:, R, 2].max(1)
    print(f"[asym] strain peak {mean_strain.max():.3f} @ f{int(mean_strain.argmax())} (release @ {release})")
    print(f"[asym] top-z  L vs R:  f0 ({topz_L[0]:.3f},{topz_R[0]:.3f})  "
          f"release f{release} ({topz_L[release]:.3f},{topz_R[release]:.3f})  "
          f"end ({topz_L[-1]:.3f},{topz_R[-1]:.3f})")
    print(f"[asym] L-R height gap: f0 {topz_L[0]-topz_R[0]:+.4f} -> release {topz_L[release]-topz_R[release]:+.4f} "
          f"(positive = left higher / right flatter)")
    np.savez(os.path.join(out_dir, "traj.npz"), X=X, strain=strain, mean_strain=mean_strain,
             topz_L=topz_L, topz_R=topz_R, release=release)

    # ---- first-8 panel (xz side-view, colored by stretch) ----
    vmax_p = float(np.quantile(strain, 0.98))
    mn, mx = X[:8].reshape(-1, 3).min(0), X[:8].reshape(-1, 3).max(0)
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for f, axp in enumerate(axs.flat):
        if f >= min(8, cfg.num_frames):
            axp.axis("off"); continue
        sc = axp.scatter(X[f][:, 0], X[f][:, 2], c=strain[f], s=7, cmap="inferno", vmin=0, vmax=vmax_p)
        axp.axvline(0.5, color="cyan", ls=":", lw=0.8)
        axp.set_xlim(mn[0], mx[0]); axp.set_ylim(mn[2], mx[2]); axp.set_aspect("equal")
        tag = "  PUSH" if f < release else ("  RELEASE" if f == release else "")
        axp.set_title(f"frame {f}  strain {mean_strain[f]:.2f}{tag}")
        axp.set_xlabel("x"); axp.set_ylabel("z")
    fig.colorbar(sc, ax=axs, shrink=0.6, label="stretch |sigma-1|")
    fig.suptitle(f"asymmetric squeeze (down @ x~{cfg.push_x}) then release", fontsize=13)
    fig.savefig(os.path.join(out_dir, "first8_panel.png"), dpi=110); plt.close(fig)

    # ---- height asymmetry timeseries ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(topz_L, "-o", ms=3, label="left half top-z"); ax.plot(topz_R, "-o", ms=3, label="right half top-z")
    ax.axvline(release, color="orange", ls="--", label=f"release @ {release}")
    ax.set_xlabel("frame"); ax.set_ylabel("max z (height)"); ax.set_title("left vs right height (asymmetric squeeze)")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(out_dir, "asym_height.png"), dpi=120); plt.close(fig)

    # ---- 3D + triplane gif ----
    mins, maxs = X.reshape(-1, 3).min(0), X.reshape(-1, 3).max(0)
    fig = plt.figure(figsize=(11, 9))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    axes2d = [fig.add_subplot(2, 2, k) for k in (2, 3, 4)]
    proj = [(0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z")]

    def draw(f):
        ax3d.cla()
        ax3d.scatter(X[f][:, 0], X[f][:, 1], X[f][:, 2], c=strain[f], s=4, cmap="inferno", vmin=0, vmax=vmax_p)
        ax3d.set_xlim(mins[0], maxs[0]); ax3d.set_ylim(mins[1], maxs[1]); ax3d.set_zlim(mins[2], maxs[2])
        ph = "PUSH" if f < release else "RELEASE"
        ax3d.set_title(f"frame {f}/{cfg.num_frames-1}  [{ph}]")
        for ax2, (a, b, la, lb) in zip(axes2d, proj):
            ax2.cla()
            ax2.scatter(X[f][:, a], X[f][:, b], c=strain[f], s=5, cmap="inferno", vmin=0, vmax=vmax_p)
            if (a, b) == (0, 2):
                ax2.axvline(0.5, color="cyan", ls=":", lw=0.8)
            ax2.set_xlim(mins[a], maxs[a]); ax2.set_ylim(mins[b], maxs[b])
            ax2.set_xlabel(la); ax2.set_ylabel(lb); ax2.set_title(f"{la}{lb}"); ax2.set_aspect("equal")
        return ()

    draw(0)
    fig.colorbar(ax3d.collections[0], ax=ax3d, shrink=0.6, label="stretch |sigma-1|")
    anim = FuncAnimation(fig, draw, frames=cfg.num_frames, blit=False)
    anim.save(os.path.join(out_dir, "traj_3d_triplane.gif"), writer=PillowWriter(fps=6)); plt.close(fig)
    print(f"[asym] DONE -> {out_dir} ({_time.time()-t0:.1f}s)")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(AsymSqueezeConfig))
