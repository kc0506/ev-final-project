"""How much do the two free wins speed up the GRADIENT path (train_E / recover)?

A train_E iteration (recover.py:step_grads) does, per frame ti (grad_window=1):
    pos = rollout_to_frame(logE, ti, v0, grad_window)   # MPM forward, records tape
    loss = mse(render(pos), gt); loss.backward()         # render adj + MPM tape.backward

The MPM grad rollout = `ti*64` DETACHED substeps (ping-pong-able, WIN1) + `64` GRAD
substeps (kept on tape, NOT ping-pong-able). Expectation: the backward (warp adjoint
replay) goes through NO partial_clone and NO ScopedTimer, so it should NOT speed up;
only the forward-record changes (its detached prefix). This profiles that, isolating
the MPM portion (render excluded -- render is unaffected by both wins). Backward is
driven by a position-space loss; `tape.backward` replays the same grad-section
adjoint kernels regardless, so the timing is faithful.

  baseline  = MPMDifferentiableSimulation  (clone detached prefix, timer-on) == production
  optimized = OptSim                        (ping-pong detached prefix, timer-off)

  python -m reuse_mpm.explore.mpm_gradstep_profile
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import List

import tyro

from .timer_sync_bench import _patched_timer
from ..diff_sim import MPMDifferentiableSimulation
from .._env import get_float_array_product
import warp as wp


def _build_grad(cfg, dev):
    """Production build (build_mpm, requires_grad=True, freeze BC) from the real
    scene cache. Returns (solver, state, model, n)."""
    import torch
    from types import SimpleNamespace
    from ..config import SimConfig
    from ..sim_render import build_mpm

    disc = torch.load(cfg.scene_cache, map_location="cpu")["disc"]
    sx = disc["sim_xyzs"].float().to(dev)
    scene = SimpleNamespace(
        device=dev, sim_xyzs=sx, points_vol=disc["points_vol"],
        freeze_mask=disc["freeze_mask"].to(dev))
    simcfg = SimConfig(grid_size=cfg.grid_size, grid_lim=cfg.grid_lim,
                       density=cfg.density, material=cfg.material,
                       grid_v_damping_scale=cfg.grid_v_damping_scale, nu=cfg.nu)
    solver, state, model = build_mpm(scene, simcfg, requires_grad=True)
    return solver, state, model, sx.shape[0]


class OptSim(MPMDifferentiableSimulation):
    """Identical to MPMDifferentiableSimulation but the detached (no-grad) prefix
    ping-pongs two pre-allocated buffers instead of allocating one per substep, and
    allocates them requires_grad=False (the upstream `requires_grad=requires_grad`
    on the detached prefix is a bug -- those buffers never carry grad). The grad
    section + ctx + backward are untouched (inherited)."""

    @staticmethod
    def forward(ctx, mpm_solver, mpm_state, mpm_model, substep, substep_size,
                num_substeps, init_pos, init_velocity, E, nu, particle_density=None,
                density_change_mask=None, static_pos=None, device="cuda:0",
                requires_grad=True, extra_no_grad_steps=0):
        import torch
        from .._env import from_torch_safe
        from ..diff_sim import MyTape, CondTape

        num_particles = init_pos.shape[0]
        mpm_state.reset_state(init_pos.clone(), None, init_velocity,
                              device=device, requires_grad=requires_grad)
        if E.ndim == 0:
            E_inp = E.item(); ctx.aggregating_E = True
        else:
            E_inp = from_torch_safe(E, dtype=wp.float32, requires_grad=requires_grad)
            ctx.aggregating_E = False
        if nu.ndim == 0:
            nu_inp = nu.item(); ctx.aggregating_nu = True
        else:
            nu_inp = from_torch_safe(nu, dtype=wp.float32, requires_grad=requires_grad)
            ctx.aggregating_nu = False
        mpm_solver.set_E_nu(mpm_model, E_inp, nu_inp, device=device)
        mpm_state.reset_density(tensor_density=particle_density,
                                selection_mask=density_change_mask,
                                device=device, requires_grad=requires_grad)

        prev_state = mpm_state
        if extra_no_grad_steps > 0:
            with torch.no_grad():
                wp.launch(kernel=get_float_array_product, dim=num_particles,
                          inputs=[mpm_state.particle_density, mpm_state.particle_vol,
                                  mpm_state.particle_mass], device=device)
                mpm_solver.prepare_mu_lam(mpm_model, mpm_state, device=device)
                # WIN 1: two reusable buffers, mpm_state never written (only read)
                ping = [prev_state.partial_clone(requires_grad=False),
                        prev_state.partial_clone(requires_grad=False)]
                src, cur = prev_state, 0
                for _ in range(extra_no_grad_steps):
                    dst = ping[cur]
                    mpm_solver.p2g2p_differentiable(mpm_model, src, dst,
                                                    substep_size, device=device)
                    src, cur = dst, 1 - cur
                prev_state = src

        wp_tape = MyTape(); cond_tape = CondTape(wp_tape, requires_grad)
        next_state_list = []
        with cond_tape:
            wp.launch(kernel=get_float_array_product, dim=num_particles,
                      inputs=[prev_state.particle_density, prev_state.particle_vol,
                              prev_state.particle_mass], device=device)
            mpm_solver.prepare_mu_lam(mpm_model, prev_state, device=device)
            for _ in range(num_substeps):
                next_state = prev_state.partial_clone(requires_grad=requires_grad)
                mpm_solver.p2g2p_differentiable(mpm_model, prev_state, next_state,
                                                substep_size, device=device)
                next_state_list.append(next_state)
                prev_state = next_state

        ctx.mpm_solver = mpm_solver; ctx.mpm_state = mpm_state
        ctx.mpm_model = mpm_model; ctx.tape = cond_tape.tape
        ctx.device = device; ctx.num_particles = num_particles
        ctx.next_state_list = next_state_list
        ctx.save_for_backward(density_change_mask)
        return wp.to_torch(next_state_list[-1].particle_x).detach().clone()


@dataclass
class Config:
    scene_cache: str = "outputs/_scene_cache/telephone_ds0.1_g32_k8.pt"
    grid_size: int = 32
    grid_lim: float = 1.0
    substep: int = 64
    density: float = 2000.0
    material: str = "jelly"
    grid_v_damping_scale: float = 1.1
    nu: float = 0.3
    E: float = 1e5
    v0z: float = -0.5
    # (frame ti, grad_window) -> extra = (ti+1-gw)*substep detached, grad = (ti+1)*substep - extra
    frames: List[int] = field(default_factory=lambda: [2, 5, 15, 15])
    grad_windows: List[int] = field(default_factory=lambda: [1, 1, 1, 16])
    repeats: int = 5
    min_quota_hours: float = 4.0


def _run_once(cfg, SimCls, timer_mode, dev, extra, num_grad):
    """One forward(record)+backward grad rollout. Returns (t_fwd, t_bwd)."""
    import torch
    solver, state, model, n = _build_grad(cfg, dev)
    init = wp.to_torch(state.particle_x).clone()
    v0 = torch.zeros_like(init); v0[:, 1] = cfg.v0z
    E_vec = torch.full((n,), cfg.E, device=dev, requires_grad=True)   # field grad path
    nu_t = torch.tensor(float(cfg.nu), device=dev)                    # 0-dim => aggregating
    density = torch.full((n,), cfg.density, device=dev)
    density_mask = torch.ones(n, device=dev).int()                   # mirrors mpm_rollout

    with _patched_timer(timer_mode):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        pos = SimCls.apply(solver, state, model, 0, (1.0 / 30.0) / cfg.substep,
                           num_grad, init, v0, E_vec, nu_t, density, density_mask, None,
                           dev, True, extra)
        torch.cuda.synchronize(); t_fwd = time.perf_counter() - t0

        loss = (pos ** 2).sum()
        torch.cuda.synchronize(); t1 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(); t_bwd = time.perf_counter() - t1
    return t_fwd, t_bwd


def _median(cfg, SimCls, timer_mode, dev, extra, num_grad, repeats):
    fs, bs = [], []
    for _ in range(repeats):
        f, b = _run_once(cfg, SimCls, timer_mode, dev, extra, num_grad)
        fs.append(f); bs.append(b)
    return statistics.median(fs), statistics.median(bs)


def main(cfg: Config) -> None:
    from .. import gpu
    gpu.pick_free_gpu(min_quota_hours=cfg.min_quota_hours)
    import torch
    dev = "cuda:0"

    # warm JIT for both forward variants + backward
    _run_once(cfg, MPMDifferentiableSimulation, "upstream", dev, 64, 64)
    _run_once(cfg, OptSim, "no-timer", dev, 64, 64)

    print(f"[gradstep] scene={cfg.scene_cache.split('/')[-1]} grid={cfg.grid_size}^3 "
          f"substep={cfg.substep}  repeats={cfg.repeats}")
    print(f"  baseline = MPMDifferentiableSimulation (clone prefix, timer-on)")
    print(f"  optimized = OptSim (ping-pong prefix, timer-off)\n")
    hdr = (f"  {'config':22s} {'fwd_base':>9s} {'fwd_opt':>9s} {'bwd_base':>9s} "
           f"{'bwd_opt':>9s} {'tot_base':>9s} {'tot_opt':>9s} {'fwd×':>6s} {'tot×':>6s}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))

    for ti, gw in zip(cfg.frames, cfg.grad_windows):
        extra = max(0, (ti + 1 - gw)) * cfg.substep
        num_grad = cfg.substep * (ti + 1) - extra
        fb, bb = _median(cfg, MPMDifferentiableSimulation, "upstream", dev, extra, num_grad, cfg.repeats)
        fo, bo = _median(cfg, OptSim, "no-timer", dev, extra, num_grad, cfg.repeats)
        tb, to = fb + bb, fo + bo
        tag = f"ti={ti:2d} gw={gw:2d} ex={extra:4d}/g{num_grad}"
        print(f"  {tag:22s} {fb*1e3:8.1f}m {fo*1e3:8.1f}m {bb*1e3:8.1f}m "
              f"{bo*1e3:8.1f}m {tb*1e3:8.1f}m {to*1e3:8.1f}m {fb/fo:5.2f}x {tb/to:5.2f}x")

    print("\n  m = ms. fwd× = forward-record speedup, tot× = full grad-step (fwd+bwd) speedup.")
    print("  Expectation: bwd_opt ≈ bwd_base (adjoint replay has no clone/timer);")
    print("  the win is all in forward-record, scaling with the detached-prefix fraction.")


if __name__ == "__main__":
    main(tyro.cli(Config))
