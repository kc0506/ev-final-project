"""Free win #2: simulate_positions allocates a fresh MPMStateStruct every substep.

`reuse_mpm/sim_render.py:simulate_positions` (the no-grad forward path used by
dataset_gen / forward_gen) does, per substep:

    nxt = prev.partial_clone(requires_grad=False)   # ~15 wp.zeros + 4 wp.copy + 1 kernel
    solver.p2g2p_differentiable(model, prev, nxt, sub_dt)
    prev = nxt

So a 1024-substep rollout allocates ~1024 full states. A fresh buffer per step is
only needed to keep each step on the autograd tape -- but this path is no-grad, so
the allocation is pure waste. The step is out-of-place (only particle_x/v/C/F_trial
flow prev->next; vol/density/mass/selection are static; the grid lives on `prev`
and is zeroed at the top of every step), so TWO pre-allocated buffers ping-ponged
are sufficient and exactly equivalent.

This probe proves equivalence (max position diff must be ~0) and measures the win:

  A  clone-per-step   current behaviour
  B  ping-pong        two buffers, 1 extra alloc total

  python -m reuse_mpm.explore.mpm_state_reuse_probe
  python -m reuse_mpm.explore.mpm_state_reuse_probe --n-particles 60000
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import tyro


@dataclass
class Config:
    n_particles: int = 20000
    grid_size: int = 32
    grid_lim: float = 2.0
    num_frames: int = 16
    substep: int = 64           # 64 * 16 == 1024 substeps, one full rollout
    repeats: int = 5
    seed: int = 0
    E: float = 1e6
    nu: float = 0.3
    density: float = 1000.0
    material: str = "jelly"
    min_quota_hours: float = 4.0


def _build(cfg: Config, dev: str):
    """Fresh solver + an initial state carrying x, v(=0), F=I, C=0."""
    import torch
    from .._env import MPMStateStruct, MPMModelStruct, MPMWARPDiff
    from ..sim_render import _ensure_warp
    _ensure_warp()

    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    lo, hi = 0.35 * cfg.grid_lim, 0.65 * cfg.grid_lim
    sx = (torch.rand(cfg.n_particles, 3, generator=g) * (hi - lo) + lo).float().to(dev)
    vol = torch.full((cfg.n_particles,), (cfg.grid_lim / cfg.grid_size) ** 3,
                     dtype=torch.float32, device=dev)
    n = cfg.n_particles

    state = MPMStateStruct(); state.init(n, device=dev, requires_grad=False)
    state.from_torch(sx.clone(), vol, None, device=dev, requires_grad=False,
                     n_grid=cfg.grid_size, grid_lim=cfg.grid_lim)
    model = MPMModelStruct(); model.init(n, device=dev, requires_grad=False)
    model.init_other_params(n_grid=cfg.grid_size, grid_lim=cfg.grid_lim, device=dev)
    solver = MPMWARPDiff(n, n_grid=cfg.grid_size, grid_lim=cfg.grid_lim, device=dev)
    solver.set_parameters_dict(model, state, {
        "material": cfg.material, "g": [0.0, 0.0, -9.8],
        "density": cfg.density, "grid_v_damping_scale": 1.1})

    density = torch.full((n,), cfg.density, dtype=torch.float32, device=dev)
    state.reset_density(density.clone(), torch.ones_like(density).type(torch.int),
                        dev, update_mass=True)
    E_t = torch.full((n,), cfg.E, dtype=torch.float32, device=dev)
    nu_t = torch.full((n,), cfg.nu, dtype=torch.float32, device=dev)
    solver.set_E_nu_from_torch(model, E_t.clone(), nu_t.clone(), dev)
    solver.prepare_mu_lam(model, state, dev)

    # give the object a small initial velocity so something actually moves
    v0 = torch.zeros_like(sx); v0[:, 2] = -0.3
    I = torch.eye(3, dtype=torch.float32, device=dev)
    F = I[None].repeat(n, 1, 1); C = torch.zeros_like(F)
    state.continue_from_torch(sx.clone(), v0, F, C, device=dev, requires_grad=False)
    return solver, state, model, sx


def _rollout_clone(cfg, solver, state, model, dev):
    """Current path: partial_clone every substep."""
    import torch, warp as wp
    sub_dt = 0.05 / cfg.substep
    prev = state
    with torch.no_grad():
        for _ in range(cfg.num_frames - 1):
            for _ in range(cfg.substep):
                nxt = prev.partial_clone(requires_grad=False)
                solver.p2g2p_differentiable(model, prev, nxt, sub_dt, device=dev)
                prev = nxt
    torch.cuda.synchronize()
    return wp.to_torch(prev.particle_x).clone()


def _rollout_pingpong(cfg, solver, state, model, dev):
    """Two pre-allocated buffers, alternated."""
    import torch, warp as wp
    sub_dt = 0.05 / cfg.substep
    other = state.partial_clone(requires_grad=False)  # the ONLY extra alloc
    bufs = [state, other]
    cur = 0
    with torch.no_grad():
        for _ in range(cfg.num_frames - 1):
            for _ in range(cfg.substep):
                prev, nxt = bufs[cur], bufs[1 - cur]
                solver.p2g2p_differentiable(model, prev, nxt, sub_dt, device=dev)
                cur = 1 - cur
    torch.cuda.synchronize()
    return wp.to_torch(bufs[cur].particle_x).clone()


def _time(fn, cfg, solver_state_builder, dev, repeats):
    import torch
    samples = []
    for _ in range(repeats):
        solver, state, model, _ = solver_state_builder()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(cfg, solver, state, model, dev)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def main(cfg: Config) -> None:
    from .. import gpu
    gpu.pick_free_gpu(min_quota_hours=cfg.min_quota_hours)
    import torch
    dev = "cuda:0"
    total_sub = (cfg.num_frames - 1) * cfg.substep

    builder = lambda: _build(cfg, dev)

    # correctness: same initial state -> compare final positions
    s1 = builder(); pos_clone = _rollout_clone(cfg, *s1[:3], dev)
    s2 = builder(); pos_ping = _rollout_pingpong(cfg, *s2[:3], dev)
    max_diff = (pos_clone - pos_ping).abs().max().item()
    finite = torch.isfinite(pos_clone).all().item() and torch.isfinite(pos_ping).all().item()

    print(f"[probe] n_particles={cfg.n_particles} grid={cfg.grid_size}^3 "
          f"substeps={total_sub} repeats={cfg.repeats}")
    print(f"[probe] final-position max|clone - pingpong| = {max_diff:.3e}  "
          f"(finite={finite})  -> {'EQUIVALENT' if max_diff < 1e-5 else 'MISMATCH!'}\n")

    t_clone = _time(_rollout_clone, cfg, builder, dev, cfg.repeats)
    t_ping = _time(_rollout_pingpong, cfg, builder, dev, cfg.repeats)
    print(f"  clone-per-step   {t_clone*1e3:8.2f} ms / rollout   "
          f"{t_clone/total_sub*1e6:6.1f} us/substep")
    print(f"  ping-pong        {t_ping*1e3:8.2f} ms / rollout   "
          f"{t_ping/total_sub*1e6:6.1f} us/substep")
    print(f"\n  alloc overhead removed: {(t_clone - t_ping)*1e3:7.2f} ms/rollout  "
          f"({(t_clone/t_ping - 1)*100:5.1f}% faster)")


if __name__ == "__main__":
    main(tyro.cli(Config))
