"""How much does upstream MPMWARPDiff's `wp.ScopedTimer(synchronize=True)` cost?

Every substep of `p2g2p_differentiable` wraps ~5-6 kernel launches in
`with wp.ScopedTimer(..., synchronize=True)`. `synchronize=True` forces a device
sync after each block, which serialises the launches and kills CPU->GPU async
pipelining. With grid 32^3 the kernels are tiny, so the sync stall may dominate.

This isolates exactly that flag by monkeypatching `wp.ScopedTimer` over a pure
forward rollout (sync count is identical with/without grad; the syncs all live in
forward, so a no-grad rollout is the cheapest representative probe):

  A  upstream   ScopedTimer(synchronize=True)   -- current behaviour
  B  sync-off   ScopedTimer(synchronize=False)  -- timer kept, no host stall
  C  no-timer   ScopedTimer is a no-op CM       -- timer removed entirely

A->B isolates the sync flag itself; B->C isolates the timer's python overhead.
Each condition: warm up, then time N substeps closing with ONE device sync so the
wall-clock is fair regardless of how many syncs happened inside.

  python -m reuse_mpm.explore.timer_sync_bench
  python -m reuse_mpm.explore.timer_sync_bench --n-particles 40000 --substeps 256
"""
from __future__ import annotations

import contextlib
import statistics
import time
from dataclasses import dataclass

import tyro


@dataclass
class Config:
    n_particles: int = 20000   # representative object particle count
    grid_size: int = 32        # matches SimConfig.grid_size default
    grid_lim: float = 2.0
    substeps: int = 1024       # 64 substep * 16 frames == one full forward rollout
    warmup: int = 64
    repeats: int = 5
    seed: int = 0
    E: float = 1e6
    nu: float = 0.3
    density: float = 1000.0
    material: str = "jelly"
    min_quota_hours: float = 4.0   # bench is seconds; safe well above the 4h hard-stop


def _build(cfg: Config, dev: str):
    import torch
    from .._env import MPMStateStruct, MPMModelStruct, MPMWARPDiff
    from ..sim_render import _ensure_warp
    _ensure_warp()

    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    # particles in a centred sub-cube of the normalised [0, grid_lim] domain
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
    return solver, state, model


@contextlib.contextmanager
def _patched_timer(mode: str):
    """Override wp.ScopedTimer for the 3 conditions. Solver calls `wp.ScopedTimer`
    by attribute, so patching the module attribute is enough."""
    import warp as wp
    orig = wp.ScopedTimer
    if mode == "upstream":
        yield  # leave as-is (synchronize=True passed by the solver)
        return
    if mode == "sync-off":
        def factory(*a, **k):
            k["synchronize"] = False
            return orig(*a, **k)
        wp.ScopedTimer = factory
    elif mode == "no-timer":
        @contextlib.contextmanager
        def noop(*a, **k):
            yield
        wp.ScopedTimer = noop
    else:
        raise ValueError(mode)
    try:
        yield
    finally:
        wp.ScopedTimer = orig


def _time_substeps(cfg: Config, solver, state, model, dev: str, n_steps: int) -> float:
    """Run n_steps of p2g2p_differentiable, return total wall-clock seconds
    (single device sync at the end)."""
    import torch
    sub_dt = 0.05 / cfg.substeps  # arbitrary stable dt; only relative timing matters
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_steps):
            solver.p2g2p_differentiable(model, state, state, sub_dt, device=dev)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main(cfg: Config) -> None:
    from .. import gpu
    gpu.pick_free_gpu(min_quota_hours=cfg.min_quota_hours)
    import torch
    dev = "cuda:0"

    print(f"[bench] n_particles={cfg.n_particles} grid={cfg.grid_size}^3 "
          f"substeps={cfg.substeps} repeats={cfg.repeats} warmup={cfg.warmup}")
    print(f"[bench] one full forward rollout (16 frames x 64 substep) = 1024 substeps\n")

    results = {}
    for mode in ("upstream", "sync-off", "no-timer"):
        solver, state, model = _build(cfg, dev)
        with _patched_timer(mode):
            _time_substeps(cfg, solver, state, model, dev, cfg.warmup)  # warm
            samples = [_time_substeps(cfg, solver, state, model, dev, cfg.substeps)
                       for _ in range(cfg.repeats)]
        med = statistics.median(samples)
        per_sub_ms = med / cfg.substeps * 1e3
        results[mode] = (med, per_sub_ms)
        spread = (max(samples) - min(samples)) / med * 100
        print(f"  {mode:9s}  {med*1e3:8.2f} ms / {cfg.substeps} substeps   "
              f"{per_sub_ms*1e3:6.1f} us/substep   (spread {spread:4.1f}%)")

    up, _ = results["upstream"]
    so, _ = results["sync-off"]
    nt, _ = results["no-timer"]
    print()
    print(f"  sync flag cost (upstream -> sync-off): "
          f"{(up - so)*1e3:7.2f} ms  ({(up/so - 1)*100:5.1f}% slower)")
    print(f"  timer overhead (sync-off -> no-timer): "
          f"{(so - nt)*1e3:7.2f} ms  ({(so/nt - 1)*100:5.1f}% slower)")
    print(f"  total      (upstream -> no-timer):     "
          f"{(up - nt)*1e3:7.2f} ms  ({(up/nt - 1)*100:5.1f}% slower)")
    # project to a 1024-substep rollout
    scale = 1024 / cfg.substeps
    print(f"\n  => per full forward rollout (1024 substeps), turning sync off saves "
          f"~{(up - so)*scale*1e3:.0f} ms")


if __name__ == "__main__":
    main(tyro.cli(Config))
