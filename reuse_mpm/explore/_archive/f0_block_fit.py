"""Entrypoint: actually FIT E (gradient descent) on the dynamic-pull block under
THREE losses, from cold starts -- to see which actually converges to GT.

  time_L2   : original per-particle position MSE (full trajectory)
  spectral  : ||FFT(width(t)) - FFT(width_gt)||^2
  centroid  : |spectral_centroid(width) - centroid_gt|^2

E is a single scalar, so we use FINITE-DIFFERENCE gradient (dL/dlogE via +-eps) +
Adam -- a faithful 1-D optimizer that follows the real loss surface (the warp block
forward with F0 has no autodiff path; FD sidesteps that and is exact enough for a
scalar). No assumption that centroid wins; report what each does.

Output under outputs/explore/f0_block_fit/.
"""
from __future__ import annotations

import os
import time as _time
from dataclasses import dataclass
from typing import Tuple

import tyro

from ..gpu import pick_free_gpu


@dataclass
class BlockFitConfig:
    nx: int = 22
    ny: int = 9
    nz: int = 16
    half: Tuple[float, float, float] = (0.18, 0.08, 0.14)
    z_base: float = 0.30
    pull_speed: float = 0.5
    pull_frames: int = 5
    grip_half_x: float = 0.045
    gt_logE: float = 4.5
    nu: float = 0.3
    K: int = 32
    inits: Tuple[float, ...] = (3.5, 5.5)   # cold starts (GT is 4.5)
    n_iters: int = 30
    lr: float = 0.15                        # Adam lr in logE space
    fd_eps: float = 0.02                    # finite-difference step in logE
    ckpt_every: int = 5                     # save partial trajectory every N iters (resumable)
    label: str = "block_fit_3loss"


def run(cfg: BlockFitConfig) -> str:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        from ..gpu import assert_gpu_quota
        assert_gpu_quota(8.0)
        print(f"[gpu] preset CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        pick_free_gpu()
    import numpy as np
    import torch
    import warp as wp
    wp.init()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .._env import MPMStateStruct, MPMModelStruct, MPMWARPDiff
    from ..config import SimConfig

    t0 = _time.time()
    out_dir = os.path.join("outputs", "explore", "f0_block_fit", cfg.label)
    os.makedirs(out_dir, exist_ok=True)
    sim = SimConfig(); dev = "cuda:0"; G, GL = sim.grid_size, sim.grid_lim
    hx, hy, hz = cfg.half
    cx, cy, cz = 0.5, 0.5, cfg.z_base + hz
    gx = torch.linspace(cx - hx, cx + hx, cfg.nx); gy = torch.linspace(cy - hy, cy + hy, cfg.ny)
    gz = torch.linspace(cz - hz, cz + hz, cfg.nz)
    X_rest = torch.stack(torch.meshgrid(gx, gy, gz, indexing="ij"), -1).reshape(-1, 3).to(dev)
    n = X_rest.shape[0]
    p_vol = torch.full((n,), float((2 * hx / max(cfg.nx - 1, 1)) ** 3), device=dev)
    eye = torch.eye(3, device=dev)

    def build():
        st = MPMStateStruct(); st.init(n, device=dev, requires_grad=False)
        st.from_torch(X_rest.clone(), p_vol, None, device=dev, requires_grad=False, n_grid=G, grid_lim=GL)
        md = MPMModelStruct(); md.init(n, device=dev, requires_grad=False)
        md.init_other_params(n_grid=G, grid_lim=GL, device=dev)
        sv = MPMWARPDiff(n, n_grid=G, grid_lim=GL, device=dev)
        sv.set_parameters_dict(md, st, {"material": sim.material, "g": [0.0, 0.0, 0.0],
                               "density": sim.density, "grid_v_damping_scale": sim.grid_v_damping_scale})
        st.reset_density(torch.full((n,), float(sim.density), device=dev).clone(),
                         torch.ones(n, device=dev).int(), dev, update_mass=True)
        return sv, st, md

    def setE(sv, md, st, logE):
        sv.set_E_nu_from_torch(md, torch.full((n,), float(10.0 ** logE), device=dev).clone(),
                               torch.full((n,), float(cfg.nu), device=dev).clone(), dev)
        sv.prepare_mu_lam(md, st, dev)

    sv, st, md = build(); setE(sv, md, st, cfg.gt_logE)
    with torch.no_grad():
        st.continue_from_torch(X_rest.clone(), torch.zeros(n, 3, device=dev), eye[None].repeat(n, 1, 1).contiguous(),
                               torch.zeros(n, 3, 3, device=dev), device=dev, requires_grad=False)
        et = cfg.pull_frames * sim.delta_t; gs = (cfg.grip_half_x, hy * 1.6, hz * 1.6)
        sv.enforce_particle_velocity_translation(st, point=(cx - hx, cy, cz), size=gs,
            velocity=(-cfg.pull_speed, 0, 0), start_time=0.0, end_time=et, device=dev)
        sv.enforce_particle_velocity_translation(st, point=(cx + hx, cy, cz), size=gs,
            velocity=(+cfg.pull_speed, 0, 0), start_time=0.0, end_time=et, device=dev)
        prev = st
        for _ in range(cfg.pull_frames):
            for _ in range(sim.substep):
                nx = prev.partial_clone(requires_grad=False)
                sv.p2g2p_differentiable(md, prev, nx, sim.substep_size, device=dev); prev = nx
        x_snap = wp.to_torch(prev.particle_x).clone(); F_snap = wp.to_torch(prev.particle_F_trial).clone()

    # ONE reusable release solver (no grips). Rebuilding per rollout leaks warp GPU
    # allocations -> crashes (~180 rollouts in). Reuse + reset each call instead.
    rsv, rst, rmd = build()
    z3 = torch.zeros(n, 3, device=dev); z33 = torch.zeros(n, 3, 3, device=dev)

    def rollout(logE):
        setE(rsv, rmd, rst, logE); rsv.time = 0.0
        with torch.no_grad():
            rst.continue_from_torch(x_snap.clone(), z3, F_snap.clone(), z33,
                                    device=dev, requires_grad=False)
            prev = rst; out = [wp.to_torch(prev.particle_x).clone()]
            for _ in range(cfg.K):
                for _ in range(sim.substep):
                    nx = prev.partial_clone(requires_grad=False)
                    rsv.p2g2p_differentiable(rmd, prev, nx, sim.substep_size, device=dev); prev = nx
                out.append(wp.to_torch(prev.particle_x).clone())
        res = torch.stack(out)                               # [K+1,n,3] cuda
        del prev; import gc; gc.collect()                    # free partial_clone states (leak -> crash)
        return res

    NFFT = 256; freqs = np.fft.rfftfreq(NFFT, d=1.0)           # numpy FFT on CPU: GPU
    # torch.fft on warp-sourced tensors crashed ('Interrupted system call'); np.fft is safe.

    def width_np(traj):
        w = traj[:, :, 0].amax(1) - traj[:, :, 0].amin(1)      # [K+1] cuda
        return w.cpu().numpy()

    def spec(w):
        return np.abs(np.fft.rfft(w - w.mean(), n=NFFT))

    def cen(sp):
        return float((freqs[1:] * sp[1:]).sum() / max(sp[1:].sum(), 1e-12))

    gt = rollout(cfg.gt_logE); gt_sp = spec(width_np(gt)); gt_c = cen(gt_sp)

    def _safe(v, big):
        return big if (not np.isfinite(v)) else v

    def L_time(logE):  return _safe(float(((rollout(logE) - gt) ** 2).sum(-1).mean()), 1.0)
    def L_spec(logE):  return _safe(float(((spec(width_np(rollout(logE))) - gt_sp) ** 2).mean()), 1e6)
    def L_cen(logE):   return _safe((cen(spec(width_np(rollout(logE)))) - gt_c) ** 2, 1.0)
    LOSSES = {"time_L2": L_time, "spectral": L_spec, "centroid": L_cen}

    # ---- FD-gradient + Adam per (loss, init); CHECKPOINT after each combo + RESUME ----
    import json
    ckpt = os.path.join(out_dir, "fit_result.json")
    results = json.load(open(ckpt))["results"] if os.path.exists(ckpt) else {}
    if results:
        print(f"[fit] resume: {len(results)} combos already done, skipping them")

    def save():
        with open(ckpt, "w") as f:
            json.dump({"gt_logE": cfg.gt_logE, "lr": cfg.lr, "n_iters": cfg.n_iters,
                       "results": results}, f, indent=2)

    b1, b2 = 0.9, 0.999
    for lname, Lf in LOSSES.items():
        for E0 in cfg.inits:
            key = f"{lname}|init{E0}"
            r = results.get(key)
            if r and r.get("done"):
                continue
            if r:  # partial resume: restore logE + Adam state + iter
                logE = r["logE"]; m = r["m"]; v = r["v"]; start = r["iter"]
                traj = r["traj"]; losst = r["loss"]
                print(f"[fit] resume {key} from iter {start}/{cfg.n_iters} (logE {logE:.3f})")
            else:
                logE = float(E0); m = v = 0.0; start = 0; traj = [logE]; losst = [Lf(logE)]
            for it in range(start, cfg.n_iters):
                gp = Lf(logE + cfg.fd_eps); gm = Lf(logE - cfg.fd_eps)
                g = (gp - gm) / (2 * cfg.fd_eps)
                m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
                mh = m / (1 - b1 ** (it + 1)); vh = v / (1 - b2 ** (it + 1))
                logE = float(np.clip(logE - cfg.lr * mh / (np.sqrt(vh) + 1e-12), 3.0, 6.0))
                traj.append(logE); losst.append(Lf(logE))
                if (it + 1) % cfg.ckpt_every == 0 or it == cfg.n_iters - 1:
                    results[key] = {"traj": traj, "loss": losst, "logE": logE, "m": m, "v": v,
                                    "iter": it + 1, "done": (it + 1 >= cfg.n_iters),
                                    "final": logE, "err": logE - cfg.gt_logE}
                    save()  # intra-run checkpoint: a crash keeps the partial curve
            print(f"[fit] {lname:9s} init {E0} -> final logE {logE:.3f} (GT {cfg.gt_logE}, "
                  f"err {logE-cfg.gt_logE:+.3f} dex = E x{10**(logE-cfg.gt_logE):.2f}) [done]")

    # ---- vis: logE trajectory per loss (cold starts overlaid) ----
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for axp, lname in zip(axs, LOSSES):
        for E0 in cfg.inits:
            r = results.get(f"{lname}|init{E0}")
            if r is None:
                continue
            axp.plot(r["traj"], "-o", ms=3, label=f"init {E0} -> {r['final']:.2f}")
        axp.axhline(cfg.gt_logE, color="k", ls="--", label=f"GT {cfg.gt_logE}")
        axp.set_title(f"{lname}"); axp.set_xlabel("iter"); axp.set_ylabel("log10 E"); axp.legend(fontsize=8)
    fig.suptitle("E gradient fit (FD+Adam) under 3 losses -- does it reach GT?", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fit_trajectories.png"), dpi=120); plt.close(fig)
    print(f"[fit] DONE -> {out_dir} ({_time.time()-t0:.1f}s)")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(BlockFitConfig))
