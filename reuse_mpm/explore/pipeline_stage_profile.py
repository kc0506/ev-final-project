"""Where does the FULL pipeline actually spend time? (render included)

The earlier probes proved the MPM forward can be ~14x faster (ping-pong + timer)
and that gradient backward is unaffected. But MPM is only one stage. This loads the
REAL telephone scene (gaussians + cameras) and times every stage of both pipelines
so we can see what actually dominates -- not guess.

  dataset_gen per sample (no-grad): build_mpm -> sim(forward) -> render(T frames) -> encode
  recover per iter (grad): window x [ rollout(MPM fwd) -> render_disp -> backward ]

  python -m reuse_mpm.explore.pipeline_stage_profile
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import tyro


@dataclass
class Config:
    scene_path: str = "/tmp2/b10401006/PhysDreamer/data/physics_dreamer/telephone"
    frame: str = "frame_00001.png"
    num_frames: int = 14
    substep: int = 64
    E: float = 1e5
    v0z: float = -0.5
    window: int = 3       # recover loss window
    grad_window: int = 1
    repeats: int = 3
    min_quota_hours: float = 4.0


def _med(fn, repeats):
    import torch
    xs = []
    for _ in range(repeats):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(); xs.append(time.perf_counter() - t0)
    return statistics.median(xs)


def _sim_pingpong(scene, cfg, solver, state, model, v0, E):
    """No-grad forward returning pos_list (world), ping-pong buffers + timer-off.
    Mirrors simulate_positions' frame sampling."""
    import torch, warp as wp
    from .timer_sync_bench import _patched_timer
    dev = scene.device; n = scene.sim_xyzs.shape[0]
    density = torch.ones_like(scene.sim_xyzs[..., 0]) * cfg.E * 0 + 2000.0
    state.reset_density(density.clone(), torch.ones_like(density).type(torch.int),
                        dev, update_mass=True)
    init = scene.sim_xyzs.clone()
    sub_dt = (1.0 / 30.0) / cfg.substep
    with torch.no_grad(), _patched_timer("no-timer"):
        E_t = torch.ones_like(init[..., 0]) * float(E)
        nu_t = torch.ones_like(init[..., 0]) * cfg.nu if hasattr(cfg, "nu") else torch.ones_like(init[..., 0]) * 0.3
        solver.set_E_nu_from_torch(model, E_t.clone(), nu_t.clone(), dev)
        solver.prepare_mu_lam(model, state, dev)
        I = torch.eye(3, dtype=torch.float32, device=dev)
        F = I[None].repeat(n, 1, 1); C = torch.zeros_like(F)
        state.continue_from_torch(init, v0, F, C, device=dev, requires_grad=False)
        other = state.partial_clone(requires_grad=False)
        bufs = [state, other]; cur = 0
        pos_list = [(init.clone() * scene.scale) - scene.shift]
        for _ in range(cfg.num_frames - 1):
            for _ in range(cfg.substep):
                solver.p2g2p_differentiable(model, bufs[cur], bufs[1 - cur], sub_dt, device=dev)
                cur = 1 - cur
            pos = wp.to_torch(bufs[cur].particle_x).clone()
            pos_list.append((pos * scene.scale) - scene.shift)
    return pos_list


def main(cfg: Config) -> None:
    from .. import gpu
    gpu.pick_free_gpu(min_quota_hours=cfg.min_quota_hours)
    import torch
    import torch.nn.functional as F
    from ..config import SceneSpec, SimConfig
    from ..scene_io import load_from_spec
    from ..sim_render import (build_mpm, simulate_positions, render_positions,
                              render_disp_frame, video_to_uint8)
    from ..mpm_rollout import MpmRollout

    dev = "cuda:0"
    simcfg = SimConfig(num_frames=cfg.num_frames, substep=cfg.substep)
    spec = SceneSpec(path=cfg.scene_path, kind="pd", device=dev)
    t_load = time.perf_counter()
    scene = load_from_spec(spec, simcfg)
    try:
        cam = scene.camera_by_frame(cfg.frame)
    except Exception:
        cam = scene.test_camera_list[0]
    n = scene.sim_xyzs.shape[0]
    n_gauss = int(scene.sim_mask.numel())
    print(f"[pipe] scene loaded in {time.perf_counter()-t_load:.1f}s  "
          f"n_mpm={n}  n_gauss={n_gauss}  T={cfg.num_frames} substep={cfg.substep}\n")

    v0 = torch.zeros_like(scene.sim_xyzs); v0[:, 1] = cfg.v0z

    # ---------------- dataset_gen (no-grad) stage split ----------------
    t_build = _med(lambda: build_mpm(scene, simcfg, requires_grad=False), cfg.repeats)
    solver, state, model = build_mpm(scene, simcfg, requires_grad=False)

    t_sim_clone = _med(lambda: simulate_positions(scene, float(cfg.E), v0, simcfg), cfg.repeats)

    def _ping():
        s, st, m = build_mpm(scene, simcfg, requires_grad=False)
        _sim_pingpong(scene, cfg, s, st, m, v0, float(cfg.E))
    t_sim_ping = _med(_ping, cfg.repeats)

    pos_list = simulate_positions(scene, float(cfg.E), v0, simcfg)
    t_render = _med(lambda: render_positions(scene, pos_list, cam), cfg.repeats)
    vid = render_positions(scene, pos_list, cam)
    t_encode = _med(lambda: video_to_uint8(vid), cfg.repeats)

    print("=== dataset_gen per sample (no-grad forward path) ===")
    fwd_tot_clone = t_sim_clone + t_render + t_encode
    fwd_tot_ping = t_sim_ping + t_render + t_encode
    print(f"  build_mpm (incl in sim)         {t_build*1e3:8.1f} ms")
    print(f"  sim forward  (clone, current)   {t_sim_clone*1e3:8.1f} ms   "
          f"{t_sim_clone/fwd_tot_clone*100:4.1f}% of sample")
    print(f"  sim forward  (ping-pong+timer)  {t_sim_ping*1e3:8.1f} ms   "
          f"({t_sim_clone/t_sim_ping:.1f}x faster)")
    print(f"  render {cfg.num_frames} frames            {t_render*1e3:8.1f} ms   "
          f"{t_render/fwd_tot_clone*100:4.1f}% (current) / {t_render/fwd_tot_ping*100:4.1f}% (opt)")
    print(f"  video_to_uint8 (encode)         {t_encode*1e3:8.1f} ms   "
          f"{t_encode/fwd_tot_clone*100:4.1f}%")
    print(f"  ---- sample total: current={fwd_tot_clone*1e3:.0f} ms  "
          f"opt-sim={fwd_tot_ping*1e3:.0f} ms  "
          f"(end-to-end speedup {fwd_tot_clone/fwd_tot_ping:.2f}x)\n")

    # ---------------- recover (grad) stage split ----------------
    roll = MpmRollout(scene, simcfg, requires_grad=True, device=dev)
    gt = torch.rand(cfg.num_frames + 1, 1, 3,
                    render_disp_frame(scene, scene.sim_xyzs.clone(), cam).shape[-2],
                    render_disp_frame(scene, scene.sim_xyzs.clone(), cam).shape[-1],
                    device=dev)  # dummy target, right shape
    logE = torch.tensor(float(__import__("numpy").log10(cfg.E)), device=dev, requires_grad=True)

    def _grad_frame(ti, measure):
        logE2 = logE.detach().clone().requires_grad_(True)
        torch.cuda.synchronize(); a = time.perf_counter()
        pos = roll.rollout_to_frame(logE2, ti, v0, cfg.grad_window)
        torch.cuda.synchronize(); t_mpm = time.perf_counter() - a
        b = time.perf_counter()
        img = render_disp_frame(scene, pos, cam)
        l = F.mse_loss(img, gt[[ti + 1]]) / cfg.window
        torch.cuda.synchronize(); t_rfwd = time.perf_counter() - b
        c = time.perf_counter()
        l.backward()
        torch.cuda.synchronize(); t_bwd = time.perf_counter() - c
        return t_mpm, t_rfwd, t_bwd

    # warm
    _grad_frame(0, False)
    agg = {"mpm": 0.0, "rfwd": 0.0, "bwd": 0.0}
    for ti in range(cfg.window):
        ms = [_grad_frame(ti, True) for _ in range(cfg.repeats)]
        mpm = statistics.median([m[0] for m in ms])
        rfwd = statistics.median([m[1] for m in ms])
        bwd = statistics.median([m[2] for m in ms])
        agg["mpm"] += mpm; agg["rfwd"] += rfwd; agg["bwd"] += bwd
        print(f"  recover frame ti={ti}: mpm_fwd={mpm*1e3:7.1f}  render_fwd={rfwd*1e3:7.1f}  "
              f"backward={bwd*1e3:7.1f} ms")
    tot = agg["mpm"] + agg["rfwd"] + agg["bwd"]
    print(f"\n=== recover per ITER (sum over window={cfg.window}, gw={cfg.grad_window}) ===")
    print(f"  MPM forward    {agg['mpm']*1e3:8.1f} ms   {agg['mpm']/tot*100:4.1f}%")
    print(f"  render forward {agg['rfwd']*1e3:8.1f} ms   {agg['rfwd']/tot*100:4.1f}%")
    print(f"  backward       {agg['bwd']*1e3:8.1f} ms   {agg['bwd']/tot*100:4.1f}%  "
          f"(render-adj + MPM tape.backward, fused)")
    print(f"  ---- iter total {tot*1e3:.0f} ms")


if __name__ == "__main__":
    main(tyro.cli(Config))
