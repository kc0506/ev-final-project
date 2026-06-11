"""Decisive test: is the visible "anchors move" bug the POROUS grid freeze BC,
and does a particle-level velocity freeze fix it?

Two MPM builds on the SAME scene/discretisation, same v0:
  A) grid     -- current path: apply_grid_bc_w_freeze_pts (zeroes ~9 grid cells)
  B) particle -- fix: solver.enforce_particle_velocity_by_mask(freeze, v=0)  (Dirichlet)

Ghosts (kmeans empty-cluster centroids snapped to normalised [0,0,0]) are EXCLUDED
from the anchor metric -- the question is whether the REAL object anchors leak.

  python -m reuse_mpm.explore.freeze_bc_fix_probe
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np


def _build(scene, cfg, mode: str):
    """Build (solver, state, model) with freeze BC = `mode` ('grid'|'particle'|'none')."""
    import torch
    import warp as wp
    from .._env import (MPMStateStruct, MPMModelStruct, MPMWARPDiff,
                        apply_grid_bc_w_freeze_pts)
    from ..sim_render import _ensure_warp
    _ensure_warp()
    dev = scene.device; sx = scene.sim_xyzs; n = sx.shape[0]
    state = MPMStateStruct(); state.init(n, device=dev, requires_grad=False)
    state.from_torch(sx.clone(), torch.from_numpy(scene.points_vol).float().to(dev), None,
                     device=dev, requires_grad=False, n_grid=cfg.grid_size, grid_lim=cfg.grid_lim)
    model = MPMModelStruct(); model.init(n, device=dev, requires_grad=False)
    model.init_other_params(n_grid=cfg.grid_size, grid_lim=cfg.grid_lim, device=dev)
    solver = MPMWARPDiff(n, n_grid=cfg.grid_size, grid_lim=cfg.grid_lim, device=dev)
    solver.set_parameters_dict(model, state, {"material": cfg.material, "g": [0.0, 0.0, 0.0],
                               "density": cfg.density, "grid_v_damping_scale": cfg.grid_v_damping_scale})
    if mode == "grid":
        apply_grid_bc_w_freeze_pts(cfg.grid_size, 1.0, sx[scene.freeze_mask, :], solver)
    elif mode == "particle":
        fm_int = scene.freeze_mask.to(torch.int32).contiguous()           # [n] int32
        solver.enforce_particle_velocity_by_mask(state, fm_int, [0.0, 0.0, 0.0], -1e9, 1e9)
    return solver, state, model


def _rollout(scene, cfg, solver, state, model, E: float, v0) -> List["np.ndarray"]:
    """No-grad MPM rollout (mirrors sim_render.simulate_positions). Returns T x [n,3] world torch."""
    import torch
    import warp as wp
    dev = scene.device; sx = scene.sim_xyzs; n = sx.shape[0]
    density = torch.ones_like(sx[..., 0]) * cfg.density
    state.reset_density(density.clone(), torch.ones_like(density).type(torch.int), dev, update_mass=True)
    init = sx.clone()
    with torch.no_grad():
        E_t = torch.ones_like(init[..., 0]) * float(E)
        nu_t = torch.ones_like(init[..., 0]) * cfg.nu
        solver.set_E_nu_from_torch(model, E_t.clone(), nu_t.clone(), dev)
        solver.prepare_mu_lam(model, state, dev)
        I = torch.eye(3, dtype=torch.float32, device=dev)
        F = I[None].repeat(n, 1, 1); C = torch.zeros_like(F)
        state.continue_from_torch(init, v0, F, C, device=dev, requires_grad=False)
        pos = [(init.clone() * scene.scale) - scene.shift]
        prev = state
        for _ in range(cfg.num_frames - 1):
            for _ in range(cfg.substep):
                nxt = prev.partial_clone(requires_grad=False)
                solver.p2g2p_differentiable(model, prev, nxt, cfg.substep_size, device=dev)
                prev = nxt
            pos.append((wp.to_torch(nxt.particle_x).clone() * scene.scale) - scene.shift)
    return pos


def run(scene_path: str = "/tmp2/b10401006/PhysDreamer/data/physics_dreamer/telephone",
        v0_vec=(0.0, -1.0, 0.0), E: float = 1e5, out: str = "outputs/_dbg_freeze_bc_fix") -> str:
    import torch, imageio
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from ..gpu import pick_free_gpu; pick_free_gpu()
    from ..config import SceneSpec, SimConfig
    from ..scene_io import load_from_spec
    from ..sim_render import make_constant_v0, render_positions, video_to_uint8

    os.makedirs(out, exist_ok=True)
    cfg = SimConfig(num_frames=14, substep=64, grid_size=32)
    scene = load_from_spec(SceneSpec(preset=None, path=scene_path, kind="pd", device="cuda:0"), cfg)
    cam = scene.camera_by_frame("frame_00001.png")
    v0 = make_constant_v0(scene, v0_vec)

    fm = scene.freeze_mask.cpu().numpy().astype(bool)                     # [n]
    ghost = scene.sim_xyzs.norm(dim=1).cpu().numpy() < 1e-4               # normalised [0,0,0]
    real_anchor = fm & ~ghost                                            # REAL object anchors
    print(f"freeze={fm.sum()}  ghost={ghost.sum()}  REAL anchors={real_anchor.sum()}  "
          f"moving={(~fm).sum()}")

    res = {}
    for mode in ["grid", "particle"]:
        solver, state, model = _build(scene, cfg, mode)
        pos = _rollout(scene, cfg, solver, state, model, E, v0)
        P = torch.stack(pos, 0).cpu().numpy()                            # [T,n,3]
        d = np.linalg.norm(P - P[0:1], axis=2)                           # [T,n]
        res[mode] = d
        imageio.mimsave(os.path.join(out, f"render_{mode}.gif"),
                        list(video_to_uint8(render_positions(scene, pos, cam))), fps=cfg.fps, loop=0)
        print(f"\n[{mode}] mean|disp| per frame:")
        print("  REAL-anchor:", " ".join(f"{d[t,real_anchor].mean():.3f}" for t in range(P.shape[0])))
        print("  moving     :", " ".join(f"{d[t,~fm].mean():.3f}" for t in range(P.shape[0])))

    # comparison plot: REAL-anchor leak, grid vs particle, vs moving
    T = res["grid"].shape[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(T), [res["grid"][t, real_anchor].mean() for t in range(T)], "o-", c="red",
            label="REAL anchors — grid BC (current)")
    ax.plot(range(T), [res["particle"][t, real_anchor].mean() for t in range(T)], "s-", c="blue",
            label="REAL anchors — particle freeze (fix)")
    ax.plot(range(T), [res["grid"][t, ~fm].mean() for t in range(T)], "^-", c="green", alpha=0.6,
            label="moving particles (grid)")
    ax.set_xlabel("frame t"); ax.set_ylabel("mean |disp| (world)")
    ax.set_title(f"REAL anchor leak: porous grid BC vs particle freeze  (v0={v0_vec})")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(out, "anchor_leak_compare.png"), dpi=120); plt.close(fig)
    print(f"\n[freeze_bc_fix] -> {out}")
    return out


if __name__ == "__main__":
    run()
