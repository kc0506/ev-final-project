"""Entrypoint: does a FREQUENCY/PERIOD loss broaden the E well? (block dynpull)

Time-domain L2 on an oscillatory release saturates at ~amplitude^2 once the beat
decorrelates -> a single-point well on a flat plateau (capture radius < grid step).
Here we test frequency-domain losses, and -- per request -- VISUALIZE the
intermediate products (width(t) signals, their FFT spectra, dominant period vs E),
not just the final landscape.

Teaching point baked in: naive spectral L2 ALSO saturates (non-overlapping peaks ->
constant), so we compare FOUR losses:
  time_L2      -- baseline (spike + plateau)
  spectral_L2  -- ||FFT_pred - FFT_gt||^2 (also saturates!)
  period       -- |dominant_period_pred - period_gt|^2  (monotone in E)
  centroid     -- |spectral_centroid_pred - centroid_gt|^2 (smooth, monotone)

Output under outputs/explore/f0_block_freqloss/.
"""
from __future__ import annotations

import os
import time as _time
from dataclasses import dataclass
from typing import Tuple

import tyro

from ..gpu import pick_free_gpu


@dataclass
class FreqLossConfig:
    nx: int = 22
    ny: int = 9
    nz: int = 16
    half: Tuple[float, float, float] = (0.18, 0.08, 0.14)
    z_base: float = 0.30
    pull_speed: float = 0.5
    pull_frames: int = 5
    grip_half_x: float = 0.045
    gt_logE: float = 4.5          # moved off 5.0 to stay clear of high-E aliasing
    nu: float = 0.3
    K: int = 32                   # long for clean FFT (GT period ~6 frames -> ~5 periods)
    logE_lo: float = 3.5
    logE_hi: float = 5.5
    logE_n: int = 41
    vis_E: Tuple[float, ...] = (3.5, 4.0, 4.5, 5.0, 5.5)  # spectra/signal overlay
    label: str = "block_freqloss_gt4p5"


def run(cfg: FreqLossConfig) -> str:
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
    out_dir = os.path.join("outputs", "explore", "f0_block_freqloss", cfg.label)
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

    # pull -> snapshot
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
    print(f"[freq] snapshot F0 maxdev {(torch.linalg.svdvals(F_snap)-1).abs().max():.3f}; GT logE {cfg.gt_logE}, K {cfg.K}")

    def width_signal(logE):
        sv, st, md = build(); setE(sv, md, st, logE)
        with torch.no_grad():
            st.continue_from_torch(x_snap.clone(), torch.zeros(n, 3, device=dev), F_snap.clone(),
                                   torch.zeros(n, 3, 3, device=dev), device=dev, requires_grad=False)
            prev = st; w = [float(x_snap[:, 0].max() - x_snap[:, 0].min())]
            for _ in range(cfg.K):
                for _ in range(sim.substep):
                    nx = prev.partial_clone(requires_grad=False)
                    sv.p2g2p_differentiable(md, prev, nx, sim.substep_size, device=dev); prev = nx
                xx = wp.to_torch(prev.particle_x)
                w.append(float(xx[:, 0].max() - xx[:, 0].min()))
        return np.array(w)

    NFFT = 256
    freqs = np.fft.rfftfreq(NFFT, d=1.0)               # cycles/frame

    def spectrum(sig):
        s = sig - sig.mean()
        return np.abs(np.fft.rfft(s, n=NFFT))

    def dom_period(spec):
        k = spec[1:].argmax() + 1
        return 1.0 / freqs[k]

    def centroid(spec):
        return float((freqs[1:] * spec[1:]).sum() / max(spec[1:].sum(), 1e-12))

    # GT signal + spectrum
    gt_sig = width_signal(cfg.gt_logE); gt_spec = spectrum(gt_sig)
    gt_per = dom_period(gt_spec); gt_cen = centroid(gt_spec)
    print(f"[freq] GT width period {gt_per:.2f} frames, centroid {gt_cen:.3f} cyc/frame")

    # ---- scan ----
    logEs = np.linspace(cfg.logE_lo, cfg.logE_hi, cfg.logE_n)
    sigs = {}; L_time = []; L_spec = []; L_per = []; L_cen = []; periods = []
    for le in logEs:
        s = width_signal(float(le)); sp = spectrum(s); sigs[round(float(le), 3)] = (s, sp)
        L_time.append(float(((s[1:] - gt_sig[1:]) ** 2).mean()))
        L_spec.append(float(((sp - gt_spec) ** 2).mean()))
        per = dom_period(sp); periods.append(per)
        L_per.append((per - gt_per) ** 2)
        L_cen.append((centroid(sp) - gt_cen) ** 2)
    L_time = np.array(L_time); L_spec = np.array(L_spec)
    L_per = np.array(L_per); L_cen = np.array(L_cen); periods = np.array(periods)
    np.savez(os.path.join(out_dir, "freqloss.npz"), logEs=logEs, L_time=L_time, L_spec=L_spec,
             L_per=L_per, L_cen=L_cen, periods=periods, gt_logE=cfg.gt_logE)

    # ---- VIS 1: width(t) signals (intermediate product) ----
    cmap = plt.cm.viridis
    cs = {le: cmap((le - min(cfg.vis_E)) / (max(cfg.vis_E) - min(cfg.vis_E))) for le in cfg.vis_E}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for le in cfg.vis_E:
        s = width_signal(le)
        ax[0].plot(s, "-o", ms=2, color=cs[le], label=f"logE {le}")
    ax[0].set_title("width(t) signal per E (the oscillation we FFT)"); ax[0].set_xlabel("frame")
    ax[0].set_ylabel("block x-width"); ax[0].legend(fontsize=8)
    # ---- VIS 2: FFT spectra (intermediate product) ----
    for le in cfg.vis_E:
        sp = spectrum(width_signal(le))
        ax[1].plot(freqs, sp, "-", color=cs[le], label=f"logE {le} (per {dom_period(sp):.1f})")
    ax[1].axvline(gt_cen, color="k", ls=":", lw=0.8)
    ax[1].set_title("width(t) FFT spectrum (peak shifts right as E grows)")
    ax[1].set_xlabel("frequency (cycles/frame)"); ax[1].set_ylabel("|FFT|"); ax[1].set_xlim(0, 0.5); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "signals_and_spectra.png"), dpi=120); plt.close(fig)

    # ---- VIS 3: dominant period vs E (monotone check) ----
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(logEs, periods, "-o", ms=3, label="measured dominant period")
    ref = gt_per * 10 ** (-(logEs - cfg.gt_logE) / 2)   # period ∝ 1/sqrt(E)
    ax.plot(logEs, ref, "k--", lw=1, label="1/sqrt(E) reference")
    ax.axvline(cfg.gt_logE, color="orange", ls=":"); ax.set_xlabel("log10 E"); ax.set_ylabel("period (frames)")
    ax.set_title("dominant period vs E (monotone = the E discriminator)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "period_vs_E.png"), dpi=120); plt.close(fig)

    # ---- VIS 4: four-loss landscape compare ----
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    for axp, (L, name, note) in zip(axs.flat, [
        (L_time, "time L2", "spike + flat plateau (saturates)"),
        (L_spec, "spectral L2", "ALSO saturates (peaks stop overlapping)"),
        (L_per, "period loss", "broad + monotone"),
        (L_cen, "spectral centroid loss", "smooth + monotone")]):
        axp.plot(logEs, L / max(L.max(), 1e-30), "-o", ms=3)
        axp.axvline(cfg.gt_logE, color="k", ls="--")
        axp.set_title(f"{name}\n{note}"); axp.set_xlabel("log10 E"); axp.set_ylabel("loss (norm to max)")
    fig.suptitle("E landscape under different losses (block dynpull, GT logE 4.5)", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "landscape_compare.png"), dpi=120); plt.close(fig)

    # numeric: capture-radius proxy = fraction of scan within 50% of each loss's max
    def well_frac(L):
        Ln = L / max(L.max(), 1e-30)
        return float((Ln < 0.5).mean())
    print(f"[freq] loss landscape '<50% of max' coverage (smaller=narrower well/plateau):")
    print(f"    time_L2 {well_frac(L_time):.2f}  spectral_L2 {well_frac(L_spec):.2f}  "
          f"period {well_frac(L_per):.2f}  centroid {well_frac(L_cen):.2f}")
    print(f"[freq] DONE -> {out_dir} ({_time.time()-t0:.1f}s)")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(FreqLossConfig))
