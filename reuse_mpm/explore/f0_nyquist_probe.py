"""Empirically demonstrate the Nyquist fold on block-release width(t) over logE 3.5-6.5.

The K-probe MARKED a Nyquist threshold (period<2 frames => freq>Nyquist of 1-sample/frame)
but never PROVED it: it only swept to ~5.1, and never re-sampled to show the aliasing. The
claim "1e6 is past Nyquist" thus had no empirical backing.

This probe makes it concrete. width(t) = all-particle x-extent (the spectral observable).
period ~ 1/sqrt(E), so at high E the oscillation outruns the 1-sample/frame Nyquist
(0.5 cycles/frame) and ALIASES to a slow apparent period. We roll out each E ONCE,
recording width at the 2x cadence (every substep/2), and derive the 1x trace by
sub-sampling -- IDENTICAL physics, only the recording cadence differs (the clean Nyquist
test). 2x doubles the Nyquist to 1.0 cyc/frame, resolving the 5.5-6.5 region that 1x folds.

CFL: at logE 6.5 the wave speed ~3-10x that at 1e4, so the default substep=64 (dt 5.2e-4)
is only marginally CFL-stable. We raise `substep` (smaller dt, SAME total time) so every E
in the sweep is a genuine stable oscillation -- a blow-up would masquerade as the very
"signal disappears" artifact we are testing. Per-E width range is printed as a stability gate.

Outputs (outputs/explore/f0_nyquist_probe/<label>/):
  peakfreq_vs_E.png   MONEY plot: measured peak freq (cyc/frame) vs logE for 1x & 2x +
                      theory sqrt(E) line + Nyquist lines -> 1x folds at 0.5, 2x tracks
  width_time.png      width(t) at a few E, 1x vs 2x (aliased slow period vs true fast)
  spectra.png         |FFT(width)| at those E, 1x vs 2x
  probe.npz           raw widths + freqs + measured/theory peaks
"""
from __future__ import annotations

import os
import time as _time
from dataclasses import dataclass, field
from typing import Tuple

import tyro

from ..gpu import pick_free_gpu


@dataclass
class NyquistConfig:
    # block geometry / pull (matches f0_spectral_K_probe so it IS the same release)
    nx: int = 22
    ny: int = 9
    nz: int = 16
    half: Tuple[float, float, float] = (0.18, 0.08, 0.14)
    z_base: float = 0.30
    pull_speed: float = 0.5
    release_frame: int = 5
    grip_half_x: float = 0.045
    nu: float = 0.3
    # sweep
    logE_lo: float = 3.5
    logE_hi: float = 6.5
    n_E: int = 25                     # 3.5..6.5 in 0.125 steps
    K: int = 48                       # horizon (frames); long enough to resolve soft-E period
    substep: int = 192                # CFL-safe dt (delta_t/substep); even so /2 stays integer
    show_E: Tuple[float, ...] = (4.5, 5.5, 6.0, 6.5)   # E values to draw time/spectra for
    min_quota_hours: float = 8.0
    label: str = "release_nyquist"


def run(cfg: NyquistConfig) -> str:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        from ..gpu import assert_gpu_quota
        assert_gpu_quota(cfg.min_quota_hours)
    else:
        pick_free_gpu(min_quota_hours=cfg.min_quota_hours)
    import gc
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import warp as wp
    wp.init()
    from ._block import Scene

    out_dir = os.path.join("outputs", "explore", "f0_nyquist_probe", cfg.label)
    os.makedirs(out_dir, exist_ok=True)
    t0 = _time.time()

    sc = Scene("release", nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, half=cfg.half, z_base=cfg.z_base,
               nu=cfg.nu, pull_speed=cfg.pull_speed, release_frame=cfg.release_frame,
               grip_half_x=cfg.grip_half_x)
    sc._sim.substep = cfg.substep                          # CFL-safe dt; total time = K*delta_t unchanged
    rec2 = cfg.substep // 2                                # 2x cadence: 2 samples/frame
    grid = np.linspace(cfg.logE_lo, cfg.logE_hi, cfg.n_E)
    print(f"[nyq] substep={cfg.substep} dt={sc._sim.substep_size:.2e}s  K={cfg.K}  "
          f"logE {cfg.logE_lo}..{cfg.logE_hi} x{cfg.n_E}")

    # one rollout per E at the 2x cadence; 1x = every other sample (identical physics)
    W2, W1, stab = {}, {}, {}
    for le in grid:
        w_fine, sdt, spf = sc.rollout_width(float(le), cfg.K, rec_substeps=rec2)  # spf=2 samples/frame
        w_fine = w_fine.numpy()
        W2[float(le)] = w_fine                              # 2x: 2 samples/frame
        W1[float(le)] = w_fine[::2]                         # 1x: 1 sample/frame
        rng = float(w_fine.max() - w_fine.min())
        bad = (not np.isfinite(w_fine).all()) or rng > 5.0  # blow-up gate (width is O(0.3))
        stab[float(le)] = (rng, bad)
        print(f"  logE {le:.3f}  width[min,max]=[{w_fine.min():.3f},{w_fine.max():.3f}] "
              f"range {rng:.3f}{'  <-- UNSTABLE' if bad else ''}")
    n_bad = sum(1 for _, b in stab.values() if b)
    if n_bad:
        print(f"[nyq] WARNING {n_bad}/{cfg.n_E} E values look unstable (range>5 or NaN) -- "
              f"raise --substep; high-E 'disappearance' could be blow-up not aliasing")

    # measured peak freq (cycles/frame), DC excluded, for each rate
    def peak_freq(w, samples_per_frame):
        s = w - w.mean()
        N = 1024
        sp = np.abs(np.fft.rfft(s, n=N))
        fr = np.fft.rfftfreq(N, d=1.0 / samples_per_frame)  # cycles/frame
        k = 1 + int(np.argmax(sp[1:]))
        return fr[k], fr, sp
    f1 = np.array([peak_freq(W1[float(le)], 1)[0] for le in grid])
    f2 = np.array([peak_freq(W2[float(le)], 2)[0] for le in grid])
    # theory: f ~ sqrt(E) calibrated on the low-E in-band region (2x unaliased, f<1.0)
    cal = [(le, f) for le, f in zip(grid, f2) if f < 0.9 and le <= 5.3]
    le0, f0 = (cal[len(cal) // 2] if cal else (grid[0], max(f2[0], 1e-3)))
    f_theory = f0 * 10 ** ((grid - le0) / 2.0)

    # ---- (1) money plot ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, f_theory, "k--", lw=1.3, label=r"theory $\propto\sqrt{E}$ (calib low-E)")
    ax.plot(grid, f1, "-o", ms=4, color="C3", label="measured peak, 1x (1 samp/frame)")
    ax.plot(grid, f2, "-s", ms=4, color="C0", label="measured peak, 2x (2 samp/frame)")
    ax.axhline(0.5, ls=":", color="C3", lw=1, label="Nyquist 1x = 0.5 cyc/frame")
    ax.axhline(1.0, ls=":", color="C0", lw=1, label="Nyquist 2x = 1.0 cyc/frame")
    ax.set_xlabel("log10 E"); ax.set_ylabel("oscillation freq (cycles/frame)")
    ax.set_title("Nyquist fold: 1x peak folds back below 0.5 where theory exceeds it; 2x tracks to 1.0")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "peakfreq_vs_E.png"), dpi=120); plt.close(fig)

    show = [le for le in cfg.show_E if any(abs(le - g) < 1e-9 for g in grid)] or list(cfg.show_E)
    # nearest grid E for each requested show_E
    show = [float(grid[int(np.argmin(np.abs(grid - le)))]) for le in cfg.show_E]

    # ---- (2) width(t) time domain, 1x vs 2x ----
    fig, axs = plt.subplots(1, len(show), figsize=(4 * len(show), 3.6), squeeze=False)
    for j, le in enumerate(show):
        a = axs[0][j]
        t2 = np.arange(len(W2[le])) / 2.0                  # frame units
        t1 = np.arange(len(W1[le]))
        a.plot(t2, W2[le], "-", color="C0", lw=1.2, label="2x")
        a.plot(t1, W1[le], "o-", color="C3", ms=3, lw=1, label="1x")
        a.set_title(f"logE {le:.2f}  (f1={f1[int(np.argmin(np.abs(grid-le)))]:.2f}, "
                    f"f2={f2[int(np.argmin(np.abs(grid-le)))]:.2f} c/f)")
        a.set_xlabel("frame"); a.legend(fontsize=8); a.grid(alpha=.3)
    axs[0][0].set_ylabel("width (x-extent)")
    fig.suptitle("width(t): 1x shows a SLOW aliased period where 2x shows the true fast oscillation")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "width_time.png"), dpi=120); plt.close(fig)

    # ---- (3) spectra, 1x vs 2x ----
    fig, axs = plt.subplots(1, len(show), figsize=(4 * len(show), 3.6), squeeze=False)
    for j, le in enumerate(show):
        a = axs[0][j]
        _, fr1, sp1 = peak_freq(W1[le], 1)
        _, fr2, sp2 = peak_freq(W2[le], 2)
        a.plot(fr2, sp2, color="C0", lw=1.1, label="2x")
        a.plot(fr1, sp1, color="C3", lw=1.1, label="1x")
        a.axvline(0.5, ls=":", color="C3", lw=1); a.axvline(1.0, ls=":", color="C0", lw=1)
        a.set_xlim(0, 1.05); a.set_title(f"logE {le:.2f}"); a.set_xlabel("freq (cyc/frame)")
        a.legend(fontsize=8); a.grid(alpha=.3)
    axs[0][0].set_ylabel("|FFT(width)|")
    fig.suptitle("|FFT(width)|: 1x peak sits below 0.5 (folded); 2x peak at the true freq")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "spectra.png"), dpi=120); plt.close(fig)

    np.savez(os.path.join(out_dir, "probe.npz"), grid=grid, f1=f1, f2=f2, f_theory=f_theory,
             show=np.array(show), substep=cfg.substep, K=cfg.K,
             **{f"w1_{le:.3f}": W1[le] for le in grid}, **{f"w2_{le:.3f}": W2[le] for le in grid})
    print(f"[nyq] done {_time.time()-t0:.0f}s -> {out_dir}")
    print(f"[nyq]   peakfreq_vs_E.png (money), width_time.png, spectra.png, probe.npz")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(NyquistConfig))
