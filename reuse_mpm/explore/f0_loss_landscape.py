"""time vs spectral vs COMBINED loss landscapes over E, for {release, drop} x {K}.

Splits expensive from cheap: the GPU pass rolls out each E ONCE per scene (to K_max)
and CACHES the per-E components -- L_time(E) scalar per K + width(t) [K_max+1]. Any
reweighting (combined = L_time/c_t + lam*L_spec/c_s), any spectral variant, any K
truncation is then an OFFLINE recombination of cached 1-D arrays (replot mode, no GPU).
Only changing K/scene/E-grid needs a new GPU pass.

  cache_<scene>.npz          gE, K_list, Ltime[K,E], widths[E,K_max+1], gt_width
  landscape_<scene>.png      rows=K, cols=[time / spectral / combined(lam)]
  weight_sensitivity.png     combined at one K, lam in {0.3,1,3}, per scene

Usage:
  compute+plot:  python -m reuse_mpm.explore.f0_loss_landscape
  replot only :  python -m reuse_mpm.explore.f0_loss_landscape --replot --lam 2.0
"""
from __future__ import annotations

import os
import time as _time
from dataclasses import dataclass
from typing import Tuple

import tyro

from ..gpu import pick_free_gpu


@dataclass
class LandscapeConfig:
    scenes: Tuple[str, ...] = ("release", "drop")
    K_list: Tuple[int, ...] = (12, 18, 24, 32)
    fine_lo: float = 3.4
    fine_hi: float = 6.0
    n_E: int = 27
    gt_logE: float = 4.5
    nu: float = 0.3
    # geometry / pull
    nx: int = 22
    ny: int = 9
    nz: int = 16
    half: Tuple[float, float, float] = (0.18, 0.08, 0.14)
    z_base: float = 0.30
    pull_speed: float = 0.5
    release_frame: int = 5
    grip_half_x: float = 0.045
    # drop
    floor_z: float = 0.25
    gravity: float = 9.8
    collider: str = "slip"
    # combine
    lam: float = 1.0                  # weight on (normalized) spectral in the combined loss
    ref_logE: float = 3.5             # reference E for per-component normalization (the "init")
    replot: bool = False              # skip GPU, read caches, just (re)plot
    min_quota_hours: float = 8.0
    label: str = "rel_drop"


def run(cfg: LandscapeConfig) -> str:
    import json  # noqa: F401
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.join("outputs", "explore", "f0_loss_landscape", cfg.label)
    os.makedirs(out_dir, exist_ok=True)
    K_max = max(cfg.K_list)
    NFFT = 512; freqs = np.fft.rfftfreq(NFFT, d=1.0)

    def cache_path(scene):
        return os.path.join(out_dir, f"cache_{scene}.npz")

    # ============================ EXPENSIVE: GPU compute + cache ============================
    if not cfg.replot:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            from ..gpu import assert_gpu_quota
            assert_gpu_quota(cfg.min_quota_hours)
        else:
            pick_free_gpu(min_quota_hours=cfg.min_quota_hours)
        import warp as wp
        wp.init()
        from ._block import Scene

        dev = "cuda:0"
        grid = np.round(np.linspace(cfg.fine_lo, cfg.fine_hi, cfg.n_E), 4)
        if cfg.gt_logE not in grid:
            grid = np.sort(np.append(grid, cfg.gt_logE))

        for scene in cfg.scenes:
            sc = Scene(scene, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, half=cfg.half, z_base=cfg.z_base,
                       nu=cfg.nu, gt_logE=cfg.gt_logE, pull_speed=cfg.pull_speed,
                       release_frame=cfg.release_frame, grip_half_x=cfg.grip_half_x,
                       gravity=cfg.gravity, floor_z=cfg.floor_z, collider=cfg.collider, device=dev)

            def xext(traj):
                return (traj[:, :, 0].amax(1) - traj[:, :, 0].amin(1)).cpu().numpy()

            gt_traj, _ = sc.rollout(cfg.gt_logE, K_max); gt_w = xext(gt_traj)
            Ltime = np.zeros((len(cfg.K_list), len(grid)))
            widths = np.zeros((len(grid), K_max + 1))
            for j, le in enumerate(grid):
                traj, _ = sc.rollout(float(le), K_max); widths[j] = xext(traj)
                for i, K in enumerate(cfg.K_list):
                    Ltime[i, j] = float(((traj[:K + 1] - gt_traj[:K + 1]) ** 2).sum(-1).mean())
            np.savez(cache_path(scene), gE=grid, K_list=np.array(cfg.K_list), Ltime=Ltime,
                     widths=widths, gt_width=gt_w, K_max=K_max, gt_logE=cfg.gt_logE)
            print(f"[landscape] cached {scene}: {len(grid)} E x {len(cfg.K_list)} K -> {cache_path(scene)}")

    # ============================ CHEAP: plot / reweight from cache ============================
    def spec(w):
        return np.abs(np.fft.rfft(w - w.mean(), n=NFFT))

    def spec_loss_K(widths, gt_w, K):     # spectral L2 vs GT, truncated to K, per E
        gsp = spec(gt_w[:K + 1])
        return np.array([float(((spec(widths[j, :K + 1]) - gsp) ** 2).mean()) for j in range(widths.shape[0])])

    def norm_to_max(a):
        return a / max(a.max(), 1e-30)

    # period->Nyquist marker (from GT-side period via release cache if present)
    def nyquist_logE(gE, widths, K):
        per = []
        for j in range(widths.shape[0]):
            sp = spec(widths[j, :K + 1]); f = freqs[1:][np.argmax(sp[1:])]
            per.append(1.0 / f if f > 0 else np.inf)
        per = np.array(per); cr = np.where(per >= 2.0)[0]
        return float(gE[cr[-1]]) if len(cr) and cr[-1] + 1 < len(gE) else None

    combos = {}  # scene -> (gE, K_list, Ltime, specL[K,E], combined[K,E], nyq)
    for scene in cfg.scenes:
        if not os.path.exists(cache_path(scene)):
            print(f"[landscape] no cache for {scene}, skip"); continue
        d = np.load(cache_path(scene)); gE = d["gE"]; Ks = d["K_list"]; Ltime = d["Ltime"]
        widths = d["widths"]; gt_w = d["gt_width"]
        iref = int(np.argmin(np.abs(gE - cfg.ref_logE)))
        specL = np.stack([spec_loss_K(widths, gt_w, int(K)) for K in Ks])
        combined = np.zeros_like(specL)
        for i in range(len(Ks)):
            ct = max(Ltime[i, iref], 1e-30); cs = max(specL[i, iref], 1e-30)
            combined[i] = Ltime[i] / ct + cfg.lam * specL[i] / cs
        nyq = nyquist_logE(gE, widths, int(max(Ks)))
        combos[scene] = (gE, Ks, Ltime, specL, combined, nyq)

        # ---- per-scene landscape: rows=K, cols=[time, spectral, combined] ----
        fig, axs = plt.subplots(len(Ks), 3, figsize=(15, 3.4 * len(Ks)), squeeze=False)
        for i, K in enumerate(Ks):
            for jc, (name, arr) in enumerate([("time_L2", Ltime[i]), ("spectral", specL[i]),
                                              (f"combined (lam={cfg.lam})", combined[i])]):
                ax = axs[i][jc]
                ax.plot(gE, norm_to_max(arr), "-o", ms=2.5)
                ax.axvline(cfg.gt_logE, color="k", ls="--", lw=1)
                ax.axvline(cfg.ref_logE, color="red", ls=":", lw=1)
                am = gE[int(np.argmin(arr))]
                if nyq is not None:
                    ax.axvspan(nyq, gE.max(), color="purple", alpha=0.08)
                ax.set_title(f"{scene} K={K}  {name}  (argmin {am:.2f})", fontsize=9)
                ax.set_xlabel("log10 E"); ax.set_ylabel("loss/max")
        fig.suptitle(f"{scene}: time / spectral / combined landscape (GT {cfg.gt_logE}, "
                     f"red=ref {cfg.ref_logE}, purple=Nyquist)", fontsize=13)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"landscape_{scene}.png"), dpi=115); plt.close(fig)
        print(f"[landscape] -> landscape_{scene}.png")

    # ---- weight sensitivity: combined at K=max, lam in {0.3,1,3}, per scene ----
    if combos:
        Kpick = max(cfg.K_list)
        fig, axs = plt.subplots(1, len(combos), figsize=(6.5 * len(combos), 4.4), squeeze=False)
        for ax, (scene, (gE, Ks, Ltime, specL, _, nyq)) in zip(axs[0], combos.items()):
            i = list(Ks).index(Kpick); iref = int(np.argmin(np.abs(gE - cfg.ref_logE)))
            ct = max(Ltime[i, iref], 1e-30); cs = max(specL[i, iref], 1e-30)
            for lam in (0.3, 1.0, 3.0):
                c = Ltime[i] / ct + lam * specL[i] / cs
                ax.plot(gE, norm_to_max(c), "-o", ms=2.5, label=f"lam={lam} (argmin {gE[int(np.argmin(c))]:.2f})")
            ax.axvline(cfg.gt_logE, color="k", ls="--", lw=1, label=f"GT {cfg.gt_logE}")
            if nyq is not None:
                ax.axvspan(nyq, gE.max(), color="purple", alpha=0.08)
            ax.set_title(f"{scene} K={Kpick}: combined vs lam"); ax.set_xlabel("log10 E"); ax.set_ylabel("loss/max")
            ax.legend(fontsize=8)
        fig.suptitle("weight sensitivity: does argmin stay at GT across lam? (plateau => insensitive)", fontsize=13)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "weight_sensitivity.png"), dpi=120); plt.close(fig)
        print(f"[landscape] -> weight_sensitivity.png")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(LandscapeConfig))
