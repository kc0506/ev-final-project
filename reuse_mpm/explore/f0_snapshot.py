"""Entrypoint: extract & visualize a deformation-gradient (F0) snapshot from a
warp-MPM rollout, so a mid-trajectory frame can be repurposed as a NEW t0.

Motivation (2026-06-12, F0 sys-id direction): everyone assumes initial
deformation F0 = I. There is no more reason for that than for v0 = 0. A
*physical*, self-consistent way to obtain a non-trivial F0 is to roll a normal
sim forward and slice frame t: that snapshot (x, v, F, C) is a valid MPM state
to restart from, with F(t) != I baked in by the dynamics -- no hand-tuning.

This is a LOW-GPU, no-grad probe. It:
  1. rolls telephone forward with an x/y v0 (the identifiable subspace),
  2. reads particle_F every frame (warp stores it per particle),
  3. quantifies strain: J = det F, principal stretches (SVD), max|sigma-1|,
  4. visualizes the strain field + time series + histograms,
  5. saves the chosen-frame snapshot (.pt) for reuse as t0.

--stretch builds a v0 GRADIENT along the cord's long axis (velocity grows from
the anchored base to the tip) -> pure axial stretch, no BC needed. (Warp ALSO
supports a sustained forced pull via solver.set_velocity_on_cuboid, which IS
wired into p2g2p_differentiable; this mode just needs no collider.)

Config is LOCAL (explore convention). Output auto-created under
outputs/explore/f0_snapshot/.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import tyro

from ..gpu import pick_free_gpu


@dataclass
class F0SnapshotConfig:
    cache_path: str = ("/tmp2/b10401006/ev-project/generative-phys/outputs/"
                       "_scene_cache/telephone_ds0.1_g32_k8.pt")
    logE: float = 5.0
    nu: float = 0.3
    # x+y in-plane excitation (the post-rot68 identifiable subspace); |v0|=0.5
    v0: Tuple[float, float, float] = (0.3536, 0.3536, 0.0)
    num_frames: int = 16
    # frame(s) to snapshot as candidate new-t0 (also saved as .pt)
    snapshot_frames: Tuple[int, ...] = (4, 8, 15)
    # stretch mode: replace uniform v0 with a velocity gradient along the cord's
    # long axis (PCA PC1 of free particles), slope * normalized-axis-coord.
    stretch: bool = False
    stretch_speed: float = 1.0  # tip speed along long axis (base ~0)
    label: str = "tele_f0_xy"


def _strain_stats(F):
    """F: [n,3,3] torch. Returns dict of per-particle scalars (all [n])."""
    import torch
    J = torch.linalg.det(F)                                   # volume ratio
    sig = torch.linalg.svdvals(F)                             # [n,3] principal stretches
    smax = sig.max(dim=1).values
    smin = sig.min(dim=1).values
    max_dev = (sig - 1.0).abs().max(dim=1).values             # max |stretch-1|
    I = torch.eye(3, device=F.device)
    fro = torch.linalg.matrix_norm(F - I, ord="fro")          # ||F-I||_F
    return {"J": J, "sigma_max": smax, "sigma_min": smin,
            "max_dev": max_dev, "frob": fro}


def run(cfg: F0SnapshotConfig) -> str:
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
    out_dir = os.path.join("outputs", "explore", "f0_snapshot", cfg.label)
    os.makedirs(out_dir, exist_ok=True)

    spec = SceneSpec(preset=ScenePreset.telephone, cache_path=cfg.cache_path)
    sim = SimConfig()
    sim.num_frames = cfg.num_frames
    scene = load_from_spec(spec, sim)
    device = scene.device
    xyz = scene.sim_xyzs
    n = xyz.shape[0]
    free = scene.free_mask if hasattr(scene, "free_mask") else ~scene.freeze_mask
    free = free.to(device)
    print(f"[f0] N={n}, free={int(free.sum())}, frozen={int((~free).sum())}, "
          f"frames={cfg.num_frames}, E=10^{cfg.logE}, stretch={cfg.stretch}")

    # ---- v0: uniform x+y, OR a stretch gradient along the cord long axis ----
    if cfg.stretch:
        fxyz = xyz[free]
        c = fxyz.mean(0)
        # PC1 of free particles = cord long axis
        _, _, Vh = torch.linalg.svd(fxyz - c, full_matrices=False)
        axis = Vh[0]                                       # [3] unit long axis
        anchor_c = xyz[~free].mean(0) if int((~free).sum()) else fxyz.min(0).values
        s = (xyz - anchor_c) @ axis                        # signed coord along axis [n]
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)     # 0 at base .. 1 at tip
        v0 = (s[:, None] * cfg.stretch_speed) * axis[None, :]  # [n,3] grows to tip
        v0 = v0 * free[:, None].float()                    # frozen handled by BC anyway
        print(f"[f0] stretch v0: long axis {axis.tolist()}, tip speed {cfg.stretch_speed}")
    else:
        v0 = torch.tensor(cfg.v0, device=device).float()[None, :].repeat(n, 1)

    # ---- build solver/state, set params, set initial state (F0 = I) ----
    solver, state, model = build_mpm(scene, sim, requires_grad=False)
    dens = torch.ones_like(xyz[..., 0]) * sim.density
    state.reset_density(dens.clone(), torch.ones_like(dens).int(), device, update_mass=True)
    with torch.no_grad():
        E_t = torch.ones_like(xyz[..., 0]) * float(10.0 ** cfg.logE)
        nu_t = torch.ones_like(xyz[..., 0]) * cfg.nu
        solver.set_E_nu_from_torch(model, E_t.clone(), nu_t.clone(), device)
        solver.prepare_mu_lam(model, state, device)
        I_mat = torch.eye(3, device=device)
        F = I_mat[None].repeat(n, 1, 1)
        C = torch.zeros_like(F)
        state.continue_from_torch(xyz.clone(), v0, F, C, device=device, requires_grad=False)

        # ---- roll out, capturing per-frame F (and a full snapshot at frames) ----
        sub_dt = sim.substep_size
        prev = state
        ts, J_mean, J_p95, sd_mean, sd_max = [], [], [], [], []
        snaps = {}
        free_np = free.cpu().numpy()
        for i in range(cfg.num_frames):
            if i > 0:
                for _ in range(sim.substep):
                    nxt = prev.partial_clone(requires_grad=False)
                    solver.p2g2p_differentiable(model, prev, nxt, sub_dt, device=device)
                    prev = nxt
            # particle_F_trial (NOT particle_F) is the deformation gradient the
            # differentiable rollout actually carries across substeps; continue_
            # from_torch writes the input F into F_trial. particle_F reads as 0
            # in this path. (verified 2026-06-12.)
            Ff = wp.to_torch(prev.particle_F_trial).clone()  # [n,3,3]
            st = _strain_stats(Ff)
            ts.append(i)
            J_mean.append(float(st["J"][free].mean()))
            J_p95.append(float(st["J"][free].quantile(0.95)))
            sd_mean.append(float(st["max_dev"][free].mean()))
            sd_max.append(float(st["max_dev"][free].max()))
            if i in cfg.snapshot_frames:
                snaps[i] = {
                    "x": wp.to_torch(prev.particle_x).clone().cpu(),
                    "v": wp.to_torch(prev.particle_v).clone().cpu(),
                    "F": Ff.cpu(),
                    "C": wp.to_torch(prev.particle_C).clone().cpu(),
                    "stats": {k: v.cpu() for k, v in st.items()},
                }
        print(f"[f0] rolled {cfg.num_frames} frames in {time.time()-t0:.1f}s; "
              f"max_dev(free) last-frame mean {sd_mean[-1]:.4f} max {sd_max[-1]:.4f}")

    # ---- time series ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(ts, sd_mean, "-o", label="mean")
    ax[0].plot(ts, sd_max, "-o", label="max")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_title("max|stretch-1| (free particles)"); ax[0].set_xlabel("frame"); ax[0].legend()
    ax[1].plot(ts, J_mean, "-o", label="mean J")
    ax[1].plot(ts, J_p95, "-o", label="p95 J")
    ax[1].axhline(1.0, color="k", lw=0.5, ls="--")
    ax[1].set_title("det(F) = volume ratio"); ax[1].set_xlabel("frame"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "f0_timeseries.png"), dpi=110)
    plt.close(fig)

    # ---- per-snapshot: 3D strain-colored scatter + stretch histogram ----
    import numpy as np
    for i, snap in snaps.items():
        x = snap["x"].numpy(); dev = snap["stats"]["max_dev"].numpy()
        J = snap["stats"]["J"].numpy()
        sig_hi = snap["stats"]["sigma_max"].numpy()
        sig_lo = snap["stats"]["sigma_min"].numpy()
        fig = plt.figure(figsize=(11, 4.2))
        axA = fig.add_subplot(1, 2, 1, projection="3d")
        sc = axA.scatter(x[free_np, 0], x[free_np, 1], x[free_np, 2],
                         c=dev[free_np], s=3, cmap="viridis")
        axA.scatter(x[~free_np, 0], x[~free_np, 1], x[~free_np, 2],
                    c="red", s=2, alpha=0.4)
        fig.colorbar(sc, ax=axA, shrink=0.6, label="max|stretch-1|")
        axA.set_title(f"frame {i}: strain field (red=anchor)")
        axB = fig.add_subplot(1, 2, 2)
        axB.hist(sig_hi[free_np], bins=40, alpha=0.6, label="sigma_max (tension)")
        axB.hist(sig_lo[free_np], bins=40, alpha=0.6, label="sigma_min (compression)")
        axB.axvline(1.0, color="k", ls="--", lw=1)
        axB.set_title(f"frame {i}: principal stretches (free)")
        axB.set_xlabel("stretch ratio"); axB.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"f0_strain_frame{i:02d}.png"), dpi=110)
        plt.close(fig)
        torch.save({**snap, "frame": i, "v0": v0.cpu(), "logE": cfg.logE,
                    "nu": cfg.nu, "free_mask": free.cpu()},
                   os.path.join(out_dir, f"snapshot_frame{i:02d}.pt"))
        print(f"[f0] frame {i}: J(free) [{J[free_np].min():.3f},{J[free_np].max():.3f}] "
              f"sigma_max up to {sig_hi[free_np].max():.3f}, "
              f"sigma_min down to {sig_lo[free_np].min():.3f}")

    print(f"[f0] DONE label={cfg.label} -> {out_dir}  ({time.time()-t0:.1f}s)")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(F0SnapshotConfig))
