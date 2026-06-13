"""Entrypoint: STEP 1 of the scene-generalization plan -- does the loss conclusion
(spectral-L2 converges from below; time_L2 / centroid fail) survive a CHANGE IN F0
AMPLITUDE?

Same symmetric two-end pull as f0_block_fit.py, but we RELEASE at several different
frames {2,3,5} -> three F0 magnitudes (maxdev small->0.19). For each F0 amplitude we
re-run the full FD+Adam E-fit under three losses (time_L2 / spectral / centroid) from
three cold/warm inits:

  init 3.5  -> capture-from-below (the only clean win last time)
  init 4.5  -> AT GT: gradient-fixed-point STABILITY (per the convergence-rigor rule:
               trust the fixed point, not best-score export)
  init 5.5  -> above GT (Nyquist-aliased region; expected polluted, kept for honesty)

NOTE on direction: this is OUTWARD pull (tension, J>1). fixed-corotated is tension/
compression asymmetric, so an inward squeeze is a SEPARATE excitation axis, not a
mirror -- not covered here on purpose (high inward compression also risks element
inversion). This isolates F0 *amplitude* robustness only.

Everything is visualized: maxdev-vs-R (the independent variable actually moved),
the width(t) signal + spectrum per R (the observable), per-R fit trajectories, and a
cross-R summary of final-err per (loss, init).

Output under outputs/explore/f0_block_squeeze_sweep/.
"""
from __future__ import annotations

import os
import time as _time
from dataclasses import dataclass
from typing import Tuple

import tyro

from ..gpu import pick_free_gpu


@dataclass
class SqueezeSweepConfig:
    nx: int = 22
    ny: int = 9
    nz: int = 16
    half: Tuple[float, float, float] = (0.18, 0.08, 0.14)
    z_base: float = 0.30
    pull_speed: float = 0.5
    grip_half_x: float = 0.045
    release_frames: Tuple[int, ...] = (2, 3, 5)   # -> three F0 amplitudes
    gt_logE: float = 4.5
    nu: float = 0.3
    K: int = 32
    inits: Tuple[float, ...] = (3.5, 4.5, 5.5)     # below / at-GT / above (Nyquist)
    n_iters: int = 30
    lr: float = 0.15
    fd_eps: float = 0.02
    ckpt_every: int = 5
    sig_logEs: Tuple[float, ...] = (4.0, 4.5, 5.0, 5.5)  # E's to draw the signal/spectrum for
    label: str = "block_squeeze_sweep"


def run(cfg: SqueezeSweepConfig) -> str:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        from ..gpu import assert_gpu_quota
        assert_gpu_quota(8.0)
        print(f"[gpu] preset CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        pick_free_gpu()
    import json
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
    out_dir = os.path.join("outputs", "explore", "f0_block_squeeze_sweep", cfg.label)
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

    # ---- forward the pull for R frames -> (x_snap, F_snap) = F0 of some amplitude ----
    psv, pst, pmd = build()

    def make_f0(R):
        setE(psv, pmd, pst, cfg.gt_logE); psv.time = 0.0
        with torch.no_grad():
            pst.continue_from_torch(X_rest.clone(), torch.zeros(n, 3, device=dev),
                                    eye[None].repeat(n, 1, 1).contiguous(),
                                    torch.zeros(n, 3, 3, device=dev), device=dev, requires_grad=False)
            et = R * sim.delta_t; gs = (cfg.grip_half_x, hy * 1.6, hz * 1.6)
            psv.enforce_particle_velocity_translation(pst, point=(cx - hx, cy, cz), size=gs,
                velocity=(-cfg.pull_speed, 0, 0), start_time=0.0, end_time=et, device=dev)
            psv.enforce_particle_velocity_translation(pst, point=(cx + hx, cy, cz), size=gs,
                velocity=(+cfg.pull_speed, 0, 0), start_time=0.0, end_time=et, device=dev)
            prev = pst
            for _ in range(R):
                for _ in range(sim.substep):
                    nx = prev.partial_clone(requires_grad=False)
                    psv.p2g2p_differentiable(pmd, prev, nx, sim.substep_size, device=dev); prev = nx
            x_snap = wp.to_torch(prev.particle_x).clone()
            F_snap = wp.to_torch(prev.particle_F_trial).clone()
        # maxdev = max per-particle displacement from rest (F0 amplitude proxy, same as landscape)
        maxdev = float((x_snap - X_rest).norm(dim=1).max())
        del prev; gc.collect()
        return x_snap, F_snap, maxdev

    # ---- ONE reusable release solver (rebuilding per rollout leaks warp GPU -> crash) ----
    rsv, rst, rmd = build()
    z3 = torch.zeros(n, 3, device=dev); z33 = torch.zeros(n, 3, 3, device=dev)

    def rollout(logE, x_snap, F_snap):
        setE(rsv, rmd, rst, logE); rsv.time = 0.0
        with torch.no_grad():
            rst.continue_from_torch(x_snap.clone(), z3, F_snap.clone(), z33, device=dev, requires_grad=False)
            prev = rst; out = [wp.to_torch(prev.particle_x).clone()]
            for _ in range(cfg.K):
                for _ in range(sim.substep):
                    nx = prev.partial_clone(requires_grad=False)
                    rsv.p2g2p_differentiable(rmd, prev, nx, sim.substep_size, device=dev); prev = nx
                out.append(wp.to_torch(prev.particle_x).clone())
        res = torch.stack(out)
        del prev; gc.collect()
        return res

    NFFT = 256; freqs = np.fft.rfftfreq(NFFT, d=1.0)  # numpy FFT (torch.fft on warp tensors crashed)

    def width_np(traj):
        w = traj[:, :, 0].amax(1) - traj[:, :, 0].amin(1)
        return w.cpu().numpy()

    def spec(w):
        return np.abs(np.fft.rfft(w - w.mean(), n=NFFT))

    def cen(sp):
        return float((freqs[1:] * sp[1:]).sum() / max(sp[1:].sum(), 1e-12))

    def _safe(v, big):
        return big if (not np.isfinite(v)) else v

    # ---- checkpointed fit over (R, loss, init) ----
    ckpt = os.path.join(out_dir, "fit_result.json")
    blob = json.load(open(ckpt)) if os.path.exists(ckpt) else {}
    results = blob.get("results", {}); meta = blob.get("meta", {})
    if results:
        print(f"[sweep] resume: {len(results)} combos already done")

    def save():
        with open(ckpt, "w") as f:
            json.dump({"gt_logE": cfg.gt_logE, "lr": cfg.lr, "n_iters": cfg.n_iters,
                       "results": results, "meta": meta}, f, indent=2)

    b1, b2 = 0.9, 0.999
    sig_cache = {}  # R -> (widths per sig_logE, gt arrays) for viz
    for R in cfg.release_frames:
        x_snap, F_snap, maxdev = make_f0(R)
        meta[str(R)] = {"maxdev": maxdev}
        print(f"[sweep] R={R}  maxdev={maxdev:.4f}")
        gt = rollout(cfg.gt_logE, x_snap, F_snap)
        gt_w = width_np(gt); gt_sp = spec(gt_w); gt_c = cen(gt_sp)

        # signal/spectrum viz cache (a few E's) -- the observable the losses see
        widths = {}
        for le in cfg.sig_logEs:
            widths[le] = width_np(rollout(le, x_snap, F_snap))
        sig_cache[R] = (widths, gt_w, gt_sp)

        def L_time(logE):  return _safe(float(((rollout(logE, x_snap, F_snap) - gt) ** 2).sum(-1).mean()), 1.0)
        def L_spec(logE):  return _safe(float(((spec(width_np(rollout(logE, x_snap, F_snap))) - gt_sp) ** 2).mean()), 1e6)
        def L_cen(logE):   return _safe((cen(spec(width_np(rollout(logE, x_snap, F_snap)))) - gt_c) ** 2, 1.0)
        LOSSES = {"time_L2": L_time, "spectral": L_spec, "centroid": L_cen}

        save()  # persist maxdev meta early
        for lname, Lf in LOSSES.items():
            for E0 in cfg.inits:
                key = f"R{R}|{lname}|init{E0}"
                r = results.get(key)
                if r and r.get("done"):
                    continue
                if r:
                    logE = r["logE"]; m = r["m"]; v = r["v"]; start = r["iter"]
                    traj = r["traj"]; losst = r["loss"]
                    print(f"[sweep] resume {key} from iter {start}")
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
                                        "final": logE, "err": logE - cfg.gt_logE, "maxdev": maxdev}
                        save()
                print(f"[sweep] R{R} {lname:9s} init {E0} -> final {logE:.3f} "
                      f"(err {logE-cfg.gt_logE:+.3f} dex = x{10**(logE-cfg.gt_logE):.2f}) [done]")

    # ============================ VISUALIZATION ============================
    # (1) maxdev vs R -- the independent variable actually moved
    Rs = list(cfg.release_frames)
    mds = [meta[str(R)]["maxdev"] for R in Rs]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar([str(R) for R in Rs], mds, color="steelblue")
    for x, md in zip(range(len(Rs)), mds):
        ax.text(x, md, f"{md:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("release frame R"); ax.set_ylabel("maxdev (F0 amplitude)")
    ax.set_title("F0 amplitude vs release frame")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "maxdev_vs_R.png"), dpi=120); plt.close(fig)

    # (2) width signal + spectrum per R (the observable the losses operate on)
    fig, axs = plt.subplots(len(Rs), 2, figsize=(12, 3.4 * len(Rs)), squeeze=False)
    for i, R in enumerate(Rs):
        widths, gt_w, gt_sp = sig_cache[R]
        for le in cfg.sig_logEs:
            axs[i][0].plot(widths[le], lw=1.3, label=f"logE {le}")
        axs[i][0].plot(gt_w, "k--", lw=1.6, label=f"GT {cfg.gt_logE}")
        axs[i][0].set_title(f"R={R} (maxdev {meta[str(R)]['maxdev']:.3f})  width(t)")
        axs[i][0].set_xlabel("frame"); axs[i][0].set_ylabel("x-extent"); axs[i][0].legend(fontsize=7)
        for le in cfg.sig_logEs:
            axs[i][1].plot(freqs[:40], spec(widths[le])[:40], lw=1.3, label=f"logE {le}")
        axs[i][1].plot(freqs[:40], gt_sp[:40], "k--", lw=1.6, label=f"GT")
        axs[i][1].set_title(f"R={R}  |FFT(width)|"); axs[i][1].set_xlabel("freq (1/frame)"); axs[i][1].legend(fontsize=7)
    fig.suptitle("observable: width(t) and its spectrum across E (peak shifts with E)", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "signals_and_spectra.png"), dpi=120); plt.close(fig)

    # (3) fit trajectories: rows = R, cols = loss; inits overlaid
    losses = ["time_L2", "spectral", "centroid"]
    fig, axs = plt.subplots(len(Rs), 3, figsize=(16, 4.4 * len(Rs)), squeeze=False, sharey=True)
    for i, R in enumerate(Rs):
        for j, lname in enumerate(losses):
            axp = axs[i][j]
            for E0 in cfg.inits:
                r = results.get(f"R{R}|{lname}|init{E0}")
                if r is None:
                    continue
                axp.plot(r["traj"], "-o", ms=2.5, label=f"init {E0}->{r['final']:.2f}")
            axp.axhline(cfg.gt_logE, color="k", ls="--", lw=1)
            axp.set_title(f"R={R}  {lname}"); axp.set_xlabel("iter")
            if j == 0:
                axp.set_ylabel("log10 E")
            axp.legend(fontsize=7)
    fig.suptitle("E gradient fit (FD+Adam) -- does the loss conclusion survive F0 amplitude?", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fit_trajectories.png"), dpi=120); plt.close(fig)

    # (4) cross-R summary: final |err| per (loss, init) vs R
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
    for axp, lname in zip(axs, losses):
        for E0 in cfg.inits:
            ys = []
            for R in Rs:
                r = results.get(f"R{R}|{lname}|init{E0}")
                ys.append(abs(r["err"]) if r else np.nan)
            axp.plot([str(R) for R in Rs], ys, "-o", label=f"init {E0}")
        axp.axhline(0.1, color="g", ls=":", lw=1, label="0.1 dex (~26%)")
        axp.set_title(lname); axp.set_xlabel("release frame R"); axp.set_ylabel("|final err| (dex)")
        axp.legend(fontsize=8)
    fig.suptitle("convergence vs F0 amplitude -- |log10 E error| at end of fit", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "summary_err.png"), dpi=120); plt.close(fig)

    print(f"[sweep] DONE -> {out_dir} ({_time.time()-t0:.1f}s)")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(SqueezeSweepConfig))
