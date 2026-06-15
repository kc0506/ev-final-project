"""Entrypoint: modal analysis (FFT + POD) of long forward dumps.

Goal (2026-06-12): make the vibration story QUANTITATIVE -- per-excitation
modal frequencies/amplitudes, the f ~ sqrt(E) law, and the warp-vs-gic axial
frequency delta that underlies the z-channel phase-mismatch hypothesis.

Inputs are xmodel_dump outputs (and optionally gic dumps); sampling rate is
read from each dump's meta.json (delta_t), so 30 Hz and 60 Hz dumps mix fine.

Method per dump:
  - signal: FREE-particle displacement D(t) = x(t) - x(0)
  - per-axis mean signal -> amplitude spectrum (de-mean, Hann, rfft, zero-pad x8)
  - POD: SVD of (T, 3N) displacement -> top modes; FFT of time coefficients
  - peak picking: max bin above DC + parabolic interpolation

Config is LOCAL (explore convention). Output dir auto under
outputs/explore/fft_modal/<tag>/.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import tyro

GIC_XMODEL = "/tmp2/b10401006/ev-project/gic/output/xmodel"


@dataclass
class FFTModalConfig:
    labels: List[str] = field(default_factory=lambda: [
        "fft_z_E4", "fft_z_E5", "fft_z_E6", "fft_y_E4", "fft_y_E5", "fft_y_E6"])
    gic_labels: List[str] = field(default_factory=lambda: ["fft_z_E5_gic"])
    tag: str = "fft_v1"
    n_pod: int = 4
    pad: int = 8  # zero-pad factor for spectra


def amp_spectrum(sig: np.ndarray, fs: float, pad: int) -> Tuple[np.ndarray, np.ndarray]:
    """sig: (T,) -> (freqs, amplitude); LINEAR detrend + Hann + zero-padded rfft.

    Linear detrend matters: gic heavy anchors absorb momentum and drift slowly,
    warp y-bending decays into secular drift -- both park huge energy at
    near-DC and mask the oscillation peaks if only the mean is removed.
    """
    t = np.arange(len(sig))
    s = sig - np.polyval(np.polyfit(t, sig, 1), t)
    w = np.hanning(len(s))
    n = len(s) * pad
    spec = np.abs(np.fft.rfft(s * w, n=n)) / (w.sum() / 2)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, spec


def top_peaks(freqs: np.ndarray, spec: np.ndarray, k: int = 3,
              fmin: float = 0.3) -> List[Tuple[float, float]]:
    """Top-k local maxima above fmin -> [(f, amp)], descending amplitude."""
    lo = np.searchsorted(freqs, fmin)
    s = spec[lo:]
    isloc = np.r_[False, (s[1:-1] > s[:-2]) & (s[1:-1] > s[2:]), False]
    idx = np.where(isloc)[0]
    idx = idx[np.argsort(s[idx])[::-1][:k]]
    return [(float(freqs[lo + i]), float(s[i + lo])) for i in idx]


def peak_freq(freqs: np.ndarray, spec: np.ndarray) -> Tuple[float, float]:
    """Dominant non-DC peak with parabolic interpolation -> (f_peak, amplitude)."""
    lo = np.searchsorted(freqs, 0.3)  # skip DC / drift bins
    i = lo + int(np.argmax(spec[lo:]))
    if 0 < i < len(spec) - 1:
        a, b, c = np.log(spec[i - 1] + 1e-30), np.log(spec[i] + 1e-30), np.log(spec[i + 1] + 1e-30)
        d = 0.5 * (a - c) / (a - 2 * b + c + 1e-30)
        return float(freqs[i] + d * (freqs[1] - freqs[0])), float(spec[i])
    return float(freqs[i]), float(spec[i])


def load_dump(label: str, gic: bool) -> Tuple[np.ndarray, float, dict]:
    """Return (free-particle traj (T,Nf,3), fs, meta)."""
    import torch
    if gic:
        d = f"{GIC_XMODEL}/{label}"
        traj = np.load(f"{d}/gic_traj.npy")
        meta = json.load(open(f"{d}/meta.json"))
        fs = 1.0 / (meta["dt"] * meta["substeps_per_frame"])
        cache_path = meta["cache"]
    else:
        d = f"outputs/explore/xmodel_dump/{label}"
        traj = np.load(f"{d}/warp_traj.npy")
        meta = json.load(open(f"{d}/meta.json"))
        fs = 1.0 / meta["delta_t"]
        cache_path = meta["cache_path"]
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    xyz = cache["disc"]["sim_xyzs"]
    ghost = (xyz == 0).all(dim=1).numpy()
    free = ~cache["disc"]["freeze_mask"].numpy()[~ghost]
    if traj.shape[1] == ghost.shape[0]:  # warp dumps keep ghosts; gic drops them
        traj = traj[:, ~ghost]
    return traj[:, free], fs, meta


def analyze(label: str, cfg: FFTModalConfig, gic: bool, out_dir: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traj, fs, meta = load_dump(label, gic)
    D = traj - traj[0:1]                       # (T, Nf, 3) displacement
    T = D.shape[0]
    m = D.mean(axis=1)                         # (T, 3) per-axis mean signal

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    res: Dict = {"label": label, "side": "gic" if gic else "warp", "fs": fs,
                 "logE": meta["logE"], "v0": meta["v0"], "n_frames": T,
                 "axis_peaks": {}, "pod": []}
    for k, axname in enumerate("xyz"):
        f, s = amp_spectrum(m[:, k], fs, cfg.pad)
        pf, pa = peak_freq(f, s)
        res["axis_peaks"][axname] = {"f": pf, "amp": pa, "rms": float(m[:, k].std()),
                                     "top3": top_peaks(f, s)}
        ax = axes[0, k]
        ax.plot(np.arange(T) / fs, m[:, k], lw=1)
        ax.set_title(f"{axname}(t) mean disp", fontsize=8)
        ax.set_xlabel("s")
        ax = axes[1, k]
        ax.plot(f, s, lw=1)
        ax.axvline(pf, color="r", ls=":", lw=1)
        ax.set_title(f"|{axname}|(f)  peak {pf:.2f} Hz", fontsize=8)
        ax.set_xlabel("Hz")
        ax.set_xlim(0, fs / 2)

    # POD on the full displacement field
    X = D.reshape(T, -1)
    X = X - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    energy = (S ** 2) / (S ** 2).sum()
    for r in range(min(cfg.n_pod, len(S))):
        f, s = amp_spectrum(U[:, r] * S[r], fs, cfg.pad)
        pf, pa = peak_freq(f, s)
        res["pod"].append({"mode": r, "f": pf, "energy_frac": float(energy[r])})
    fig.suptitle(f"{label} ({res['side']})  logE={meta['logE']}  v0={meta['v0']}  "
                 f"POD: " + " ".join(f"m{r['mode']}:{r['f']:.2f}Hz({r['energy_frac']:.0%})"
                                     for r in res["pod"]), fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{label}_spectra.png", dpi=120)
    plt.close(fig)
    return res


def run(cfg: FFTModalConfig) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.join("outputs", "explore", "fft_modal", cfg.tag)
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for lab in cfg.labels:
        results.append(analyze(lab, cfg, gic=False, out_dir=out_dir))
        print(f"[fft] {lab}: " + " ".join(
            f"{a}:{v['f']:.2f}Hz(rms {v['rms']:.4f})" for a, v in results[-1]["axis_peaks"].items()))
    for lab in cfg.gic_labels:
        results.append(analyze(lab, cfg, gic=True, out_dir=out_dir))
        print(f"[fft] {lab} (gic): " + " ".join(
            f"{a}:{v['f']:.2f}Hz" for a, v in results[-1]["axis_peaks"].items()))

    with open(f"{out_dir}/summary.json", "w") as fj:
        json.dump(results, fj, indent=2)

    # f vs E scaling: z-excited dumps -> z-axis peak; y-excited -> dominant POD
    fig, ax = plt.subplots(figsize=(6, 4.5))
    # rot68: y excitation stays out-of-plane (y axis); z excitation is axial (z)
    for kind, axis_key, mk in (("z", "z", "o"), ("y", "y", "s")):
        pts = [(r["logE"], r["axis_peaks"][axis_key]["f"]) for r in results
               if r["side"] == "warp" and f"fft_{kind}_" in r["label"]]
        if len(pts) >= 2:
            le = np.array([p[0] for p in pts])
            fr = np.array([p[1] for p in pts])
            slope = np.polyfit(le, np.log10(fr), 1)[0]
            ax.plot(le, fr, mk + "-", label=f"{kind}-excited ({axis_key}-axis peak), "
                                            f"dlogf/dlogE={slope:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("log10 E")
    ax.set_ylabel("peak frequency (Hz)")
    ax.set_title("modal frequency vs E (theory slope 0.5)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/f_vs_E.png", dpi=130)
    plt.close(fig)
    print(f"[fft] done -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(FFTModalConfig))
