"""Entrypoint: DYNAMIC-PULL stretch-release (the principled F0, no hand-set stretch).

Instead of hand-setting a uniform F0 = diag(s, ...) (a "perfect cuboid that
uniformly expanded" -- physically unreal), we GRAB the two x-ends with prescribed-
velocity cuboid BCs and pull them apart for `pull_frames`, letting MPM grow a
self-consistent, NON-uniform deformation gradient (stress concentrates near the
grips, the middle necks). Then we remove the BC and RELEASE. The F at any frame is
whatever MPM produced -- no guessing. This is the dynamic-BC -> snapshot idea we
wanted: a realistic F0.

Block starts at rest (F0=I, v0=0). Warp solver.set_velocity_on_cuboid drives the
grips; it IS applied inside p2g2p_differentiable (grid_postprocess), gated by
time in [start, end); end_time = pull_frames * delta_t.

Outputs (outputs/explore/f0_dynamic_pull/<label>/):
  traj_3d_triplane.gif, first8_panel.png, strain_motion.png, traj.npz
"""
from __future__ import annotations

import os
import time as _time
from dataclasses import dataclass
from typing import Tuple

import tyro

from ..gpu import pick_free_gpu


@dataclass
class DynamicPullConfig:
    nx: int = 22
    ny: int = 9
    nz: int = 16
    half: Tuple[float, float, float] = (0.18, 0.08, 0.14)   # rest half-extents
    z_base: float = 0.30
    pull_speed: float = 0.5        # grip outward speed (each end), normalized units/s
    pull_frames: int = 5           # frames of pulling before release
    grip_half_x: float = 0.045     # cuboid half-width capturing each end slab
    logE: float = 5.0
    nu: float = 0.3
    num_frames: int = 32
    label: str = "block_dynpull_release"


def run(cfg: DynamicPullConfig) -> str:
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
    out_dir = os.path.join("outputs", "explore", "f0_dynamic_pull", cfg.label)
    os.makedirs(out_dir, exist_ok=True)
    sim = SimConfig(); sim.num_frames = cfg.num_frames
    dev = "cuda:0"
    G, GL = sim.grid_size, sim.grid_lim

    # ---- block of particles at REST (no pre-deformation) ----
    hx, hy, hz = cfg.half
    cx, cy, cz = 0.5, 0.5, cfg.z_base + hz
    gx = torch.linspace(cx - hx, cx + hx, cfg.nx)
    gy = torch.linspace(cy - hy, cy + hy, cfg.ny)
    gz = torch.linspace(cz - hz, cz + hz, cfg.nz)
    X_rest = torch.stack(torch.meshgrid(gx, gy, gz, indexing="ij"), -1).reshape(-1, 3).to(dev)
    n = X_rest.shape[0]
    spacing = 2 * hx / max(cfg.nx - 1, 1)
    p_vol = torch.full((n,), float(spacing ** 3), device=dev)
    F0 = torch.eye(3, device=dev)[None].repeat(n, 1, 1).contiguous()  # rest
    v0 = torch.zeros(n, 3, device=dev); C0 = torch.zeros(n, 3, 3, device=dev)
    print(f"[pull] n={n}, rest x[{cx-hx:.3f},{cx+hx:.3f}]; pull {cfg.pull_frames}f @ "
          f"+-{cfg.pull_speed} -> each end moves ~{cfg.pull_speed*cfg.pull_frames*sim.delta_t:.3f}")

    # ---- build warp solver/state/model (mirrors sim_render.build_mpm) ----
    state = MPMStateStruct(); state.init(n, device=dev, requires_grad=False)
    state.from_torch(X_rest.clone(), p_vol, None, device=dev, requires_grad=False,
                     n_grid=G, grid_lim=GL)
    model = MPMModelStruct(); model.init(n, device=dev, requires_grad=False)
    model.init_other_params(n_grid=G, grid_lim=GL, device=dev)
    solver = MPMWARPDiff(n, n_grid=G, grid_lim=GL, device=dev)
    solver.set_parameters_dict(model, state, {"material": sim.material, "g": [0.0, 0.0, 0.0],
                               "density": sim.density, "grid_v_damping_scale": sim.grid_v_damping_scale})

    # grip windows computed here; registered AFTER continue_from_torch (the particle
    # selection mask reads positions at registration time).
    end_t = cfg.pull_frames * sim.delta_t
    gsize = (cfg.grip_half_x, hy * 1.6, hz * 1.6)

    # ---- forward: pull phase + release phase ----
    dens = torch.full((n,), float(sim.density), device=dev)
    state.reset_density(dens.clone(), torch.ones_like(dens).int(), dev, update_mass=True)
    with torch.no_grad():
        E_t = torch.full((n,), float(10.0 ** cfg.logE), device=dev)
        nu_t = torch.full((n,), float(cfg.nu), device=dev)
        solver.set_E_nu_from_torch(model, E_t.clone(), nu_t.clone(), dev)
        solver.prepare_mu_lam(model, state, dev)
        state.continue_from_torch(X_rest.clone(), v0, F0, C0, device=dev, requires_grad=False)
        # PARTICLE-level rigid grip (PhysGaussian tear_bread recipe): select the two
        # x-end slabs and set their velocity to +-pull_speed during [0, end_t), then
        # release. Cleaner than grid set_velocity_on_cuboid (no 27-node stencil leak).
        solver.enforce_particle_velocity_translation(
            state, point=(cx - hx, cy, cz), size=gsize,
            velocity=(-cfg.pull_speed, 0.0, 0.0), start_time=0.0, end_time=end_t, device=dev)
        solver.enforce_particle_velocity_translation(
            state, point=(cx + hx, cy, cz), size=gsize,
            velocity=(+cfg.pull_speed, 0.0, 0.0), start_time=0.0, end_time=end_t, device=dev)
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
    strain = np.stack([(torch.linalg.svdvals(F) - 1.0).abs().max(dim=1).values.cpu().numpy()
                       for F in Fs])                                  # [T,n]
    mean_strain = strain.mean(1)
    mean_motion = np.concatenate([[0.0], np.linalg.norm(X[1:] - X[0:1], axis=-1).mean(1)])
    release = cfg.pull_frames
    max_def_frame = int(mean_strain.argmax())
    print(f"[pull] strain: peak {mean_strain.max():.3f} @ frame {max_def_frame} "
          f"(release @ {release}); strain@release {mean_strain[release]:.3f}; "
          f"motion peak {mean_motion.max():.4f}")
    np.savez(os.path.join(out_dir, "traj.npz"), X=X, strain=strain,
             mean_strain=mean_strain, mean_motion=mean_motion, release=release)

    # ---- first-8 grid panel (xz side-view, colored by stretch) ----
    vmax_p = float(np.quantile(strain, 0.98))
    mn, mx = X[:8].reshape(-1, 3).min(0), X[:8].reshape(-1, 3).max(0)
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for f, axp in enumerate(axs.flat):
        if f >= min(8, cfg.num_frames):
            axp.axis("off"); continue
        sc = axp.scatter(X[f][:, 0], X[f][:, 2], c=strain[f], s=7, cmap="inferno", vmin=0, vmax=vmax_p)
        axp.set_xlim(mn[0], mx[0]); axp.set_ylim(mn[2], mx[2]); axp.set_aspect("equal")
        tag = ("  PULL" if f < release else ("  RELEASE" if f == release else "")) \
              + ("  MAXDEF" if f == max_def_frame else "")
        axp.set_title(f"frame {f}  strain {mean_strain[f]:.2f}{tag}")
        axp.set_xlabel("x"); axp.set_ylabel("z")
    fig.colorbar(sc, ax=axs, shrink=0.6, label="stretch |sigma-1|")
    fig.suptitle(f"dynamic two-end pull ({release}f) then release, E=1e{cfg.logE:g}", fontsize=13)
    fig.savefig(os.path.join(out_dir, "first8_panel.png"), dpi=110); plt.close(fig)

    # ---- strain & motion timeseries ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a in ax:
        a.axvline(release, color="orange", ls="--", label=f"release @ {release}")
    ax[0].plot(mean_strain, "-o", ms=3); ax[0].axvline(max_def_frame, color="r", ls=":", label=f"max @ {max_def_frame}")
    ax[0].set_title("mean stretch |sigma-1|"); ax[0].set_xlabel("frame"); ax[0].legend()
    ax[1].plot(mean_motion, "-o", ms=3, color="tab:green"); ax[1].set_title("mean motion vs t0")
    ax[1].set_xlabel("frame"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "strain_motion.png"), dpi=120); plt.close(fig)

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
        ph = "PULL" if f < release else "RELEASE"
        ax3d.set_title(f"frame {f}/{cfg.num_frames-1}  [{ph}]  strain {mean_strain[f]:.2f}")
        for ax2, (a, b, la, lb) in zip(axes2d, proj):
            ax2.cla()
            ax2.scatter(X[f][:, a], X[f][:, b], c=strain[f], s=5, cmap="inferno", vmin=0, vmax=vmax_p)
            ax2.set_xlim(mins[a], maxs[a]); ax2.set_ylim(mins[b], maxs[b])
            ax2.set_xlabel(la); ax2.set_ylabel(lb); ax2.set_title(f"{la}{lb}"); ax2.set_aspect("equal")
        return ()

    draw(0)
    fig.colorbar(ax3d.collections[0], ax=ax3d, shrink=0.6, label="stretch |sigma-1|")
    anim = FuncAnimation(fig, draw, frames=cfg.num_frames, blit=False)
    anim.save(os.path.join(out_dir, "traj_3d_triplane.gif"), writer=PillowWriter(fps=6)); plt.close(fig)
    print(f"[pull] DONE -> {out_dir} ({_time.time()-t0:.1f}s)")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(DynamicPullConfig))
