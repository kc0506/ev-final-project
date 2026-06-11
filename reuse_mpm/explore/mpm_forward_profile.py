"""End-to-end profile of the two free wins on the REAL forward path.

Loads the real telephone scene cache (7555 particles, the actual dataset_gen /
forward_gen geometry & physics: grid_lim=1, gravity off, jelly, density 2000) and
times the no-grad MPM forward rollout under the full 2x2 matrix:

    rollout : clone-per-step (current)  vs  ping-pong (WIN 1)
    timer   : synchronize=True (current) vs no-op (WIN 2)

Condition (clone, timer-on) == current production. (ping-pong, timer-off) == proposed.
All four are checked bit-identical against the production baseline.

  python -m reuse_mpm.explore.mpm_forward_profile
  python -m reuse_mpm.explore.mpm_forward_profile --scene-cache outputs/_scene_cache/hat_ds0.1_g32_k8.pt
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import tyro

from .timer_sync_bench import _patched_timer
from .mpm_state_reuse_probe import _rollout_clone, _rollout_pingpong


@dataclass
class Config:
    scene_cache: str = "outputs/_scene_cache/telephone_ds0.1_g32_k8.pt"
    grid_size: int = 32
    grid_lim: float = 1.0          # production (SimConfig.grid_lim)
    num_frames: int = 16
    substep: int = 64
    density: float = 2000.0        # production
    material: str = "jelly"
    grid_v_damping_scale: float = 1.1
    nu: float = 0.3
    E: float = 1e5                 # our GT stiffness
    v0z: float = -0.5              # forward_gen example v0 = (0, -0.5, 0)
    repeats: int = 7
    min_quota_hours: float = 4.0


def _build_from_cache(cfg: Config, dev: str):
    """Build solver + initial state from the real cached sim_xyzs / points_vol,
    mirroring sim_render.simulate_positions' setup exactly."""
    import torch
    from .._env import MPMStateStruct, MPMModelStruct, MPMWARPDiff
    from ..sim_render import _ensure_warp
    _ensure_warp()

    disc = torch.load(cfg.scene_cache, map_location="cpu")["disc"]
    sx = disc["sim_xyzs"].float().to(dev)
    vol = torch.from_numpy(disc["points_vol"]).float().to(dev)
    n = sx.shape[0]

    state = MPMStateStruct(); state.init(n, device=dev, requires_grad=False)
    state.from_torch(sx.clone(), vol, None, device=dev, requires_grad=False,
                     n_grid=cfg.grid_size, grid_lim=cfg.grid_lim)
    model = MPMModelStruct(); model.init(n, device=dev, requires_grad=False)
    model.init_other_params(n_grid=cfg.grid_size, grid_lim=cfg.grid_lim, device=dev)
    solver = MPMWARPDiff(n, n_grid=cfg.grid_size, grid_lim=cfg.grid_lim, device=dev)
    solver.set_parameters_dict(model, state, {
        "material": cfg.material, "g": [0.0, 0.0, 0.0],   # gravity off (production)
        "density": cfg.density, "grid_v_damping_scale": cfg.grid_v_damping_scale})

    density = torch.full((n,), cfg.density, dtype=torch.float32, device=dev)
    state.reset_density(density.clone(), torch.ones_like(density).type(torch.int),
                        dev, update_mass=True)
    E_t = torch.full((n,), cfg.E, dtype=torch.float32, device=dev)
    nu_t = torch.full((n,), cfg.nu, dtype=torch.float32, device=dev)
    solver.set_E_nu_from_torch(model, E_t.clone(), nu_t.clone(), dev)
    solver.prepare_mu_lam(model, state, dev)

    v0 = torch.zeros_like(sx); v0[:, 1] = cfg.v0z
    I = torch.eye(3, dtype=torch.float32, device=dev)
    F = I[None].repeat(n, 1, 1); C = torch.zeros_like(F)
    state.continue_from_torch(sx.clone(), v0, F, C, device=dev, requires_grad=False)
    return solver, state, model, n


def _measure(cfg, fn, timer_mode, dev, repeats):
    """Median wall-clock of fn over `repeats` fresh states, under timer_mode."""
    import torch
    samples = []
    final = None
    for _ in range(repeats):
        solver, state, model, _ = _build_from_cache(cfg, dev)
        with _patched_timer(timer_mode):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            final = fn(cfg, solver, state, model, dev)
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), final


def main(cfg: Config) -> None:
    from .. import gpu
    gpu.pick_free_gpu(min_quota_hours=cfg.min_quota_hours)
    import torch
    dev = "cuda:0"
    total_sub = (cfg.num_frames - 1) * cfg.substep

    _, _, _, n = _build_from_cache(cfg, dev)
    # warm up kernel JIT once (untimed)
    s = _build_from_cache(cfg, dev); _rollout_clone(cfg, *s[:3], dev)
    s = _build_from_cache(cfg, dev); _rollout_pingpong(cfg, *s[:3], dev)

    grid = [
        ("clone   , timer-on  (PRODUCTION)", _rollout_clone,    "upstream"),
        ("pingpong, timer-on  (WIN1)",        _rollout_pingpong, "upstream"),
        ("clone   , timer-off (WIN2)",        _rollout_clone,    "no-timer"),
        ("pingpong, timer-off (WIN1+WIN2)",   _rollout_pingpong, "no-timer"),
    ]
    print(f"[profile] scene={cfg.scene_cache.split('/')[-1]} n={n} grid={cfg.grid_size}^3 "
          f"frames={cfg.num_frames} substep={cfg.substep} ({total_sub} substeps) "
          f"repeats={cfg.repeats}\n")

    base_t = base_final = None
    for label, fn, tmode in grid:
        t, final = _measure(cfg, fn, tmode, dev, cfg.repeats)
        if base_t is None:
            base_t, base_final = t, final
            diff = 0.0
        else:
            diff = (final - base_final).abs().max().item()
        speed = base_t / t
        print(f"  {label:34s} {t*1e3:8.2f} ms/rollout  "
              f"{t/total_sub*1e6:6.1f} us/sub   {speed:5.2f}x   "
              f"max|diff|={diff:.2e}")

    print(f"\n  baseline = clone+timer-on (current production).")
    print(f"  proposed = pingpong+timer-off (last row): overall forward speedup above.")


if __name__ == "__main__":
    main(tyro.cli(Config))
