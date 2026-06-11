"""Entrypoint: WHY do frozen particles (and their KNN-driven gaussians) move?

Bug report (telephone forward_gen): particles that should be frozen visibly move,
at two levels -- (1) MPM particles in `freeze_mask` drift, and (2) 3DGS gaussians
whose entire top_k KNN are frozen still translate.

ROOT CAUSE (found by this probe)
--------------------------------
GHOST particles at the grid origin. `downsample_with_kmeans_gpu_with_chunk`
(the kmeans_gpu library) emits a spurious centroid of EXACTLY [0,0,0] for every
EMPTY cluster: its `single_batch_forward` does `torch.sum(cp * score, dim=0)` over
the assigned points `cp`, and for an empty cluster `cp` is empty so the sum is
[0,0,0] (sum over an empty axis). With n_clusters ~= 0.1*N many clusters end up
empty, so a chunk of MPM particles land at the origin. For telephone ~7% (530 of
7555) are such ghosts.

Then the cascade:
  - [0,0,0] is far from `moving_part_points`, so `find_far_points` marks every
    ghost FROZEN. For telephone ~87% of the "frozen" set is ghosts (530/612);
    only ~82 are the real anchor base.
  - [0,0,0] is OUTSIDE g2p's position clamp valid region [2*dx, grid_lim-2*dx],
    so on the FIRST substep g2p_differentiable clamps each ghost from 0 -> 2*dx,
    a fixed inward jump (~0.108 normalised at grid=32) that is INDEPENDENT of v0
    and of the freeze BC -- pure coordinate artifact. This dominates the
    "frozen particles move" symptom (level 1).
  - Any rendered gaussian whose top_k KNN reaches a ghost inherits that clamp
    jump via the KNN-averaged offset (level 2). Gaussians whose top_k are only
    REAL anchors instead leak a small, v0-proportional amount (~5% of free
    motion) from PhysDreamer's inherently porous GRID-cell freeze (only a handful
    of grid nodes get zeroed, but g2p gathers from a 27-node stencil). That
    porous leak is the same in PhysDreamer; it is masked there by small learned
    velocities (`v = velo_field*0.1`), and exposed here by a large global v0.

What this probe reports
-----------------------
  - ghost count (exact [0,0,0]) and how many are in freeze_mask
  - particles OUTSIDE the g2p clamp region, and the pure-clamp displacement
  - rollout decomposition: GHOST vs REAL-anchor vs FREE displacement, and the
    v0-(in)dependence that separates the clamp artifact from the elastic leak
  - level-2: all-frozen-knn gaussians and how many of their top_k are ghosts

Read-only diagnostic; does NOT modify the canonical pipeline. Config is LOCAL.

  python -m reuse_mpm.explore.freeze_probe --scene.preset telephone --E 1e5
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import tyro

from ..config import SceneSpec, SimConfig
from ..gpu import pick_free_gpu


@dataclass
class FreezeProbeConfig:
    """explore.freeze_probe config (local; not in the single-source config.py)."""

    scene: SceneSpec
    E: float = 1e5
    sim: SimConfig = field(default_factory=lambda: SimConfig(num_frames=14, substep=64))
    frame: str = "frame_00001.png"
    out: Optional[str] = None
    run_label: str = ""


def _rollout(scene, E, v0, cfg):
    """Mirror sim_render.simulate_positions; return T x [n,3] normalised positions."""
    import torch
    import warp as wp
    from ..sim_render import build_mpm

    device = scene.device
    sim_xyzs = scene.sim_xyzs
    n = sim_xyzs.shape[0]
    solver, state, model = build_mpm(scene, cfg, requires_grad=False)
    density = torch.ones_like(sim_xyzs[..., 0]) * cfg.density
    state.reset_density(density.clone(), torch.ones_like(density).type(torch.int),
                        device, update_mass=True)
    init_xyzs = sim_xyzs.clone()
    with torch.no_grad():
        E_t = (torch.ones_like(init_xyzs[..., 0]) * float(E)).clamp(1.0, 5e8)
        nu_t = torch.ones_like(init_xyzs[..., 0]) * cfg.nu
        solver.set_E_nu_from_torch(model, E_t.clone(), nu_t.clone(), device)
        solver.prepare_mu_lam(model, state, device)
        I_mat = torch.eye(3, dtype=torch.float32, device=device)
        F = I_mat[None, ...].repeat(n, 1, 1)
        C = torch.zeros_like(F)
        state.continue_from_torch(init_xyzs, v0, F, C, device=device, requires_grad=False)
        sub_dt = cfg.substep_size
        norm_list = [init_xyzs.clone()]
        prev = state
        for _f in range(cfg.num_frames - 1):
            for _ in range(cfg.substep):
                nxt = prev.partial_clone(requires_grad=False)
                solver.p2g2p_differentiable(model, prev, nxt, sub_dt, device=device)
                prev = nxt
            norm_list.append(wp.to_torch(nxt.particle_x).clone())
    return norm_list


def _maxdisp(norm, mask):
    import torch
    d = torch.stack([(norm[t] - norm[0]).norm(dim=-1) for t in range(len(norm))], 0)
    return float(d.max(0).values[mask].mean())


def run(cfg: FreezeProbeConfig) -> str:
    pick_free_gpu()
    import torch
    from ..sim_render import make_constant_v0
    from ..scene_io import load_from_spec
    from ..run_io import RunDir

    t0 = time.time()
    label = cfg.run_label or f"{cfg.scene.display_name}_E{cfg.E:g}"
    rd = RunDir.create(__name__, label, cfg.out)

    scene = load_from_spec(cfg.scene, cfg.sim)
    X = scene.sim_xyzs
    fm = scene.freeze_mask
    qm = scene.query_mask
    tk = scene.top_k_index
    n = X.shape[0]
    G = cfg.sim.grid_size
    a_min = 2.0 / G  # g2p clamp lower bound = 2*dx (grid_lim=1)

    # ---- ghost / clamp analysis (static) ----
    is_zero = (X.abs().sum(1) == 0)                       # exact [0,0,0] ghosts
    n_ghost = int(is_zero.sum())
    n_ghost_frozen = int(fm[is_zero].sum())
    outside = (X < a_min).any(1) | (X > (1 - a_min)).any(1)
    real_anchor = fm & ~is_zero
    pure_clamp = (X.clamp(a_min, 1 - a_min) - X).norm(dim=-1)
    print(f"[freeze_probe] {scene.name}: n={n}  frozen={int(fm.sum())}  free={int(qm.sum())}")
    print(f"[GHOST] exact [0,0,0] particles: {n_ghost} ({100*n_ghost/n:.1f}%); "
          f"in freeze_mask: {n_ghost_frozen}")
    print(f"[GHOST] real anchor (frozen & non-ghost): {int(real_anchor.sum())}")
    print(f"[CLAMP] particles outside g2p region [{a_min:.4f},{1-a_min:.4f}]: "
          f"{int(outside.sum())}  (ghosts: {int((outside & is_zero).sum())})")
    print(f"[CLAMP] pure-clamp displacement of ghosts: mean={float(pure_clamp[is_zero].mean()):.5f}"
          if n_ghost else "[CLAMP] (no ghosts)")

    # ---- rollout decomposition, and v0-(in)dependence ----
    print("\n[ROLLOUT] max-over-T displacement by group, vs v0 magnitude:")
    rows = {}
    for mag in (0.1, 1.0):
        norm = _rollout(scene, float(cfg.E), make_constant_v0(scene, (0.0, -mag, 0.0)), cfg.sim)
        g = _maxdisp(norm, is_zero) if n_ghost else 0.0
        a = _maxdisp(norm, real_anchor) if int(real_anchor.sum()) else 0.0
        f = _maxdisp(norm, qm)
        rows[mag] = (g, a, f)
        print(f"  v0=-{mag}:  ghost={g:.5f}  real_anchor={a:.5f}  free={f:.5f}")
    print("  -> ghost ~constant in v0 (clamp artifact); real_anchor scales with v0 "
          "(elastic porous-BC leak); free scales with v0 (intended motion)")

    # ---- level 2: all-frozen-knn gaussians & ghost reach ----
    all_frozen_g = fm[tk].all(dim=1)
    n_af = int(all_frozen_g.sum())
    ghost_in_tk = is_zero[tk[all_frozen_g]] if n_af else None
    af_ghost_any = int(ghost_in_tk.any(1).sum()) if n_af else 0
    af_ghost_mean = float(ghost_in_tk.float().sum(1).mean()) if n_af else 0.0
    print(f"\n[L2] gaussians whose ALL top_k are frozen: {n_af}/{tk.shape[0]}")
    print(f"[L2] of those, gaussians with >=1 GHOST in top_k: {af_ghost_any} "
          f"(mean ghosts/k = {af_ghost_mean:.2f}); the rest are driven by the "
          f"real-anchor porous-BC leak")

    rd.write_json("report.json", {
        "task": "explore.freeze_probe", "scene": scene.name, "E": float(cfg.E),
        "grid_size": G, "n_particles": n, "n_frozen": int(fm.sum()),
        "n_ghost_origin": n_ghost, "ghost_frac": round(n_ghost / n, 4),
        "n_ghost_in_freeze": n_ghost_frozen, "n_real_anchor": int(real_anchor.sum()),
        "n_outside_clamp": int(outside.sum()),
        "ghost_pure_clamp_disp": float(pure_clamp[is_zero].mean()) if n_ghost else None,
        "rollout_v0_0p1": {"ghost": rows[0.1][0], "real_anchor": rows[0.1][1], "free": rows[0.1][2]},
        "rollout_v0_1p0": {"ghost": rows[1.0][0], "real_anchor": rows[1.0][1], "free": rows[1.0][2]},
        "L2_n_allfrozen_knn_gauss": n_af,
        "L2_n_allfrozen_with_ghost_topk": af_ghost_any,
        "L2_mean_ghosts_per_k": af_ghost_mean,
        "root_cause": "kmeans_gpu empty-cluster -> [0,0,0] ghost centroids; "
                      "frozen by find_far_points; clamped inward by g2p (level1); "
                      "KNN-driven into gaussians (level2). Secondary: porous grid freeze BC.",
        "elapsed_sec": round(time.time() - t0, 2),
    })
    rd.finish()
    print(f"\n[freeze_probe] -> {rd.root}  ({time.time()-t0:.1f}s)")
    return rd.root


if __name__ == "__main__":
    run(tyro.cli(FreezeProbeConfig))
