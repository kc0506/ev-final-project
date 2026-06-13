"""Why does spectral-on-pure-release learn at K=32 but stall at K=18?

Hypothesis: the gate is the INIT's oscillation period, not GT's. period ~ 1/sqrt(E),
so at the (softest) init logE=3.5 the period ~19 frames. K<19 => from the init the
width(t) completes <1 oscillation => FFT has no resolvable peak => spectral loss is
FLAT near the init => no gradient => stall. K>=~20 => even the soft init shows a peak
=> gradient appears => converges.

Efficiency: roll out each grid-E ONCE to K_max; every smaller K is just the first K
frames (release is deterministic). So one rollout per E covers all K.

Outputs (outputs/explore/f0_spectral_K_probe/<label>/):
  loss_vs_E_perK.png      spectral loss vs logE, one curve per K  (the money plot)
  width_init_vs_gt.png    width(t) at init 3.5 & GT 4.5 with period markers + K lines
  spectra_perK.png        |FFT(width)| at a few E for K=18 vs K=32
  period_vs_E.png         measured period vs logE + "K = 1 init-period" threshold
  probe.npz
"""
from __future__ import annotations

import os
import time as _time
from dataclasses import dataclass
from typing import Tuple

import tyro

from ..gpu import pick_free_gpu


@dataclass
class KProbeConfig:
    nx: int = 22
    ny: int = 9
    nz: int = 16
    half: Tuple[float, float, float] = (0.18, 0.08, 0.14)
    z_base: float = 0.30
    pull_speed: float = 0.5
    release_frame: int = 5
    grip_half_x: float = 0.045
    gt_logE: float = 4.5
    nu: float = 0.3
    K_max: int = 48
    K_list: Tuple[int, ...] = (8, 12, 16, 18, 20, 24, 32, 40, 48)
    logE_grid: Tuple[float, ...] = (3.3, 3.5, 3.7, 3.9, 4.1, 4.3, 4.5, 4.7, 4.9, 5.1)
    n_fine: int = 0                   # >0 => override logE_grid with linspace(fine_lo, fine_hi, n_fine)
    fine_lo: float = 3.4
    fine_hi: float = 4.6
    min_quota_hours: float = 8.0
    label: str = "release_Kprobe"


def run(cfg: KProbeConfig) -> str:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        from ..gpu import assert_gpu_quota
        assert_gpu_quota(cfg.min_quota_hours)
    else:
        pick_free_gpu(min_quota_hours=cfg.min_quota_hours)
    import gc
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
    out_dir = os.path.join("outputs", "explore", "f0_spectral_K_probe", cfg.label)
    os.makedirs(out_dir, exist_ok=True)
    sim = SimConfig(); dev = "cuda:0"; G, GL = sim.grid_size, sim.grid_lim
    hx, hy, hz = cfg.half
    cx, cy, cz = 0.5, 0.5, cfg.z_base + hz
    gx = torch.linspace(cx - hx, cx + hx, cfg.nx); gy = torch.linspace(cy - hy, cy + hy, cfg.ny)
    gz = torch.linspace(cz - hz, cz + hz, cfg.nz)
    X_rest = torch.stack(torch.meshgrid(gx, gy, gz, indexing="ij"), -1).reshape(-1, 3).to(dev)
    n = X_rest.shape[0]
    p_vol = torch.full((n,), float((2 * hx / max(cfg.nx - 1, 1)) ** 3), device=dev)
    eye = torch.eye(3, device=dev); z3 = torch.zeros(n, 3, device=dev); z33 = torch.zeros(n, 3, 3, device=dev)

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

    # ---- pull F0 once ----
    psv, pst, pmd = build(); setE(psv, pmd, pst, cfg.gt_logE); psv.time = 0.0
    with torch.no_grad():
        pst.continue_from_torch(X_rest.clone(), z3, eye[None].repeat(n, 1, 1).contiguous(), z33,
                                device=dev, requires_grad=False)
        et = cfg.release_frame * sim.delta_t; gs = (cfg.grip_half_x, hy * 1.6, hz * 1.6)
        psv.enforce_particle_velocity_translation(pst, point=(cx - hx, cy, cz), size=gs,
            velocity=(-cfg.pull_speed, 0, 0), start_time=0.0, end_time=et, device=dev)
        psv.enforce_particle_velocity_translation(pst, point=(cx + hx, cy, cz), size=gs,
            velocity=(+cfg.pull_speed, 0, 0), start_time=0.0, end_time=et, device=dev)
        prev = pst
        for _ in range(cfg.release_frame):
            for _ in range(sim.substep):
                nxt = prev.partial_clone(requires_grad=False)
                psv.p2g2p_differentiable(pmd, prev, nxt, sim.substep_size, device=dev); prev = nxt
        x_snap = wp.to_torch(prev.particle_x).clone(); F_snap = wp.to_torch(prev.particle_F_trial).clone()
    del prev; gc.collect()

    # ---- one full K_max release rollout per grid-E -> width(t) [K_max+1] ----
    rsv, rst, rmd = build()

    def width_full(logE):
        setE(rsv, rmd, rst, logE); rsv.time = 0.0
        with torch.no_grad():
            rst.continue_from_torch(x_snap.clone(), z3, F_snap.clone(), z33, device=dev, requires_grad=False)
            prev = rst; w = [float(wp.to_torch(prev.particle_x)[:, 0].amax() - wp.to_torch(prev.particle_x)[:, 0].amin())]
            for _ in range(cfg.K_max):
                for _ in range(sim.substep):
                    nxt = prev.partial_clone(requires_grad=False)
                    rsv.p2g2p_differentiable(rmd, prev, nxt, sim.substep_size, device=dev); prev = nxt
                x = wp.to_torch(prev.particle_x)
                w.append(float(x[:, 0].amax() - x[:, 0].amin()))
        del prev; gc.collect()
        return np.array(w)

    grid = list(np.round(np.linspace(cfg.fine_lo, cfg.fine_hi, cfg.n_fine), 4)) if cfg.n_fine > 0 else list(cfg.logE_grid)
    if cfg.gt_logE not in grid:
        grid = sorted(grid + [cfg.gt_logE])
    W = {le: width_full(le) for le in grid}                    # le -> width(t) [K_max+1]
    print(f"[Kprobe] rolled {len(grid)} E x K_max={cfg.K_max}")

    NFFT = 512; freqs = np.fft.rfftfreq(NFFT, d=1.0)

    def spec_K(w, K):
        s = w[:K + 1]
        return np.abs(np.fft.rfft(s - s.mean(), n=NFFT))

    def period_K(w, K):                                        # dominant period (frames) from peak bin
        sp = spec_K(w, K)
        f = freqs[1:][np.argmax(sp[1:])]
        return (1.0 / f) if f > 0 else np.inf

    # spectral loss vs E, per K  (vs GT spectrum at that K)
    loss_perK = {K: np.array([float(((spec_K(W[le], K) - spec_K(W[cfg.gt_logE], K)) ** 2).mean()) for le in grid])
                 for K in cfg.K_list}
    gE = np.array(grid)
    per = np.array([period_K(W[le], cfg.K_max) for le in grid])    # measured period (frames) per E
    # Nyquist: where the true period crosses 2 frames (above it, frame-sampling aliases)
    cross = np.where(per[:-1] >= 2.0)[0]
    logE_nyq = float(gE[cross[-1]]) if len(cross) and cross[-1] + 1 < len(gE) else None

    # ---- (1) money plot: spectral loss vs logE, one curve per K ----
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(cfg.K_list)))
    for K, c in zip(cfg.K_list, cmap):
        L = loss_perK[K]; Ln = L / max(L.max(), 1e-30)          # normalize per K to compare shapes
        ax.plot(gE, Ln, "-o", ms=3, color=c, label=f"K={K}")
    ax.axvline(3.5, color="red", ls=":", label="init 3.5")
    ax.axvline(cfg.gt_logE, color="k", ls="--", label=f"GT {cfg.gt_logE}")
    if logE_nyq is not None:
        ax.axvspan(logE_nyq, gE.max(), color="purple", alpha=0.10)
        ax.axvline(logE_nyq, color="purple", ls="-.", label=f"Nyquist (period=2f) @ {logE_nyq:.2f}")
    ax.set_xlabel("log10 E"); ax.set_ylabel("spectral loss (norm per K)")
    ax.set_title("spectral loss vs E per K -- is there a slope at the init (3.5)?"); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "loss_vs_E_perK.png"), dpi=120); plt.close(fig)

    # slope of spectral loss at the init (finite diff around 3.5) per K -> quantify "learnable"
    i35 = int(np.argmin(np.abs(gE - 3.5)))
    slope = {K: float((loss_perK[K][min(i35 + 1, len(gE) - 1)] - loss_perK[K][max(i35 - 1, 0)])) for K in cfg.K_list}
    print("[Kprobe] |spectral-loss slope at init 3.5| per K (bigger = more learnable):")
    for K in cfg.K_list:
        per_init = period_K(W[3.5], K) if 3.5 in W else period_K(W[grid[i35]], K)
        print(f"   K={K:3d}  slope={slope[K]:+.3e}   init-period(meas)={per_init:.1f}f   K/initperiod={K/per_init:.2f}")

    # ---- (2) width(t) at init 3.5 & GT, with K markers ----
    le_init = grid[i35]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(W[le_init], "-o", ms=3, label=f"init logE {le_init} (soft, long period)")
    ax.plot(W[cfg.gt_logE], "-s", ms=3, label=f"GT logE {cfg.gt_logE}")
    for K in (18, 32):
        if K <= cfg.K_max:
            ax.axvline(K, color="grey", ls="--"); ax.text(K, ax.get_ylim()[1], f"K={K}", fontsize=8, va="top")
    ax.set_xlabel("frame"); ax.set_ylabel("width (x-extent)")
    ax.set_title("width(t): at the soft init, <1 oscillation fits inside K=18"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "width_init_vs_gt.png"), dpi=120); plt.close(fig)

    # ---- (3) spectra at a few E for K=18 vs K=32 ----
    show_E = [le_init, float(gE[int(np.argmin(np.abs(gE - 4.1)))]), cfg.gt_logE]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    for axp, K in zip(axs, (18, 32)):
        for le in show_E:
            axp.plot(freqs[:60], spec_K(W[le], K)[:60], label=f"logE {le}")
        axp.set_title(f"|FFT(width)| at K={K}"); axp.set_xlabel("freq (1/frame)"); axp.legend(fontsize=8)
    fig.suptitle("short K = no resolvable peak at the soft init", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "spectra_perK.png"), dpi=120); plt.close(fig)

    # ---- (4) period vs E + threshold ----
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(gE, per, "-o", label="measured period (frames)")
    ax.axhline(2.0, color="purple", ls="-.", label="Nyquist (2 frames)")
    for K in (18, 32):
        ax.axhline(K, color="grey", ls="--"); ax.text(gE[0], K, f"K={K}", fontsize=8, va="bottom")
    ax.axvline(3.5, color="red", ls=":", label="init 3.5")
    ax.set_xlabel("log10 E"); ax.set_ylabel("period (frames)")
    ax.set_title("period ~ 1/sqrt(E): at init 3.5 period > K=18 -> <1 cycle"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "period_vs_E.png"), dpi=120); plt.close(fig)

    np.savez(os.path.join(out_dir, "probe.npz"), grid=gE, K_list=np.array(cfg.K_list),
             widths=np.stack([W[le] for le in grid]),
             loss_perK=np.stack([loss_perK[K] for K in cfg.K_list]), period=per)
    print(f"[Kprobe] DONE -> {out_dir} ({_time.time()-t0:.1f}s)")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(KProbeConfig))
