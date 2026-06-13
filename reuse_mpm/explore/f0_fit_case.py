"""Single-case F0 sys-id fit: ONE scene, chosen losses+inits -> its OWN dir + viz.

This replaces the sweep-one-json pattern (hard to observe, tempts blind sweeps).
Each invocation is one self-contained, observable experiment with forward viz +
loss curves + logE trajectories + signal/spectrum, all under its own label dir.

scene="release": pull the two x-ends apart for R frames, snapshot F0, then release
                 (v0=0, g=0, NO floor) -> pure displacement-controlled oscillation.
                 Amplitude pinned by F0, E enters only via frequency (beat trap).
scene="drop":    SAME F0 snapshot, but release UNDER gravity onto a slip floor.
                 Gravity + contact = a FORCE -> deformation amplitude becomes
                 E-dependent (displacement = force/stiffness ~ 1/E). This is the
                 PAC-NeRF channel; tests whether time_L2 REVIVES here.

Outputs (outputs/explore/f0_fit_case/<label>/):
  forward_panel.png, observables.png, F0_stretch.png, forward.npz   (always)
  loss_curves.png, logE_traj.png, signals_and_spectra.png, fit_result.json  (if fit)
"""
from __future__ import annotations

import os
import time as _time
from dataclasses import dataclass, field
from typing import Tuple

import tyro

from ..gpu import pick_free_gpu


@dataclass
class FitCaseConfig:
    scene: str = "release"            # "release" | "drop"
    # geometry (shared)
    nx: int = 22
    ny: int = 9
    nz: int = 16
    half: Tuple[float, float, float] = (0.18, 0.08, 0.14)
    z_base: float = 0.30
    # pull (F0 generation)
    pull_speed: float = 0.5
    release_frame: int = 5
    grip_half_x: float = 0.045
    # drop scene
    floor_z: float = 0.25
    gravity: float = 9.8              # magnitude of -z gravity (drop only)
    collider: str = "slip"
    friction: float = 0.0
    # squeeze scene (asymmetric downward press vs floor)
    push_x: float = 0.60
    push_half_x: float = 0.07
    push_half_z: float = 0.045
    push_speed: float = 0.45
    push_frames: int = 5
    # physics / fit
    gt_logE: float = 4.5
    nu: float = 0.3
    K: int = 32
    losses: Tuple[str, ...] = ("time_L2", "spectral")   # any of: time_L2 / spectral / centroid / combined
    lam: float = 1.0                  # combined: weight on (ref-normalized) spectral
    ref_logE: float = 3.5             # combined: fixed E for per-component normalization
    inits: Tuple[float, ...] = (3.5,)
    n_iters: int = 30
    lr: float = 0.15
    fd_eps: float = 0.02
    ckpt_every: int = 5
    sig_logEs: Tuple[float, ...] = (4.0, 4.5, 5.0, 5.5)
    fit: bool = True                  # False = forward-only design pass
    overlay_results: bool = True      # FIXTURE: every fit auto-produces GT-vs-converged-E 3d/triplane overlay
    overlay_fps: int = 3              # 0.5x slow-motion
    min_quota_hours: float = 8.0      # lower for short runs when daily quota is tight
    label: str = "case"


def run(cfg: FitCaseConfig) -> str:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        from ..gpu import assert_gpu_quota
        assert_gpu_quota(cfg.min_quota_hours)
        print(f"[case] preset CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        pick_free_gpu(min_quota_hours=cfg.min_quota_hours)
    import gc
    import json
    import numpy as np
    import torch
    import warp as wp
    wp.init()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ._block import Scene, SCENES

    t0 = _time.time()
    assert cfg.scene in SCENES, f"unknown scene {cfg.scene!r} (have {list(SCENES)})"
    out_dir = os.path.join("outputs", "explore", "f0_fit_case", cfg.label)
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda:0"
    sc = Scene(cfg.scene, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, half=cfg.half, z_base=cfg.z_base,
               nu=cfg.nu, gt_logE=cfg.gt_logE, pull_speed=cfg.pull_speed,
               release_frame=cfg.release_frame, grip_half_x=cfg.grip_half_x,
               push_x=cfg.push_x, push_half_x=cfg.push_half_x, push_half_z=cfg.push_half_z,
               push_speed=cfg.push_speed, push_frames=cfg.push_frames, gravity=cfg.gravity,
               floor_z=cfg.floor_z, collider=cfg.collider, friction=cfg.friction, device=dev)
    n, X_rest = sc.n, sc.X_rest
    cx, cy, cz, hx, hy, hz = sc.cx, sc.cy, sc.cz, sc.hx, sc.hy, sc.hz
    x_snap, F_snap, maxdev, F0_stretch = sc.x_snap, sc.F_snap, sc.maxdev, sc.F0_stretch
    pull_X, pull_S = sc.pull_X, sc.pull_S
    floor_z = sc.floor_z

    def rollout(logE):
        return sc.rollout(logE, cfg.K)

    print(f"[case] scene={cfg.scene} maxdev={maxdev:.4f} F0_stretch mean {F0_stretch.mean():.3f}/max {F0_stretch.max():.3f}"
          + (f"  floor z={floor_z} ({cfg.collider})" if sc.has_floor else "")
          + (f"  g=-{cfg.gravity}" if cfg.scene in ('drop', 'freefall') else ""))

    # ---- forward viz (GT rollout): panel + observables + F0 stretch + npz ----
    gt_xyz, gt_S = rollout(cfg.gt_logE)
    relX = gt_xyz.cpu().numpy(); relS = gt_S.cpu().numpy()
    pullX = torch.stack(pull_X).cpu().numpy(); pullS = torch.stack(pull_S).cpu().numpy()
    fullX = np.concatenate([pullX, relX[1:]], 0); fullS = np.concatenate([pullS, relS[1:]], 0)
    rel0 = pullX.shape[0]                      # index where release begins
    width = fullX[:, :, 0].max(1) - fullX[:, :, 0].min(1)
    minz = fullX[:, :, 2].min(1); comz = fullX[:, :, 2].mean(1)
    np.savez(os.path.join(out_dir, "forward.npz"), X=fullX, stretch=fullS, width=width,
             minz=minz, comz=comz, rel_start=rel0, maxdev=maxdev, scene=cfg.scene,
             floor_z=floor_z if sc.has_floor else -1)

    vmax = float(np.quantile(fullS, 0.98)) or 1e-3
    mn = fullX[:, :, [0, 2]].reshape(-1, 2).min(0); mx = fullX[:, :, [0, 2]].reshape(-1, 2).max(0)
    rel_idx = list(range(rel0, len(fullX), max(1, (len(fullX) - rel0) // 7)))[:8]
    sel = (list(range(rel0)) + rel_idx)[:12]
    ncol = 6; nrow = (len(sel) + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow), squeeze=False)
    for ax in axs.flat:
        ax.axis("off")
    for a, f in enumerate(sel):
        ax = axs.flat[a]; ax.axis("on")
        psc = ax.scatter(fullX[f][:, 0], fullX[f][:, 2], c=fullS[f], s=6, cmap="inferno", vmin=0, vmax=vmax)
        if sc.has_floor:
            ax.axhline(floor_z, color="cyan", ls="-", lw=1)
        ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_aspect("equal")
        ax.set_title(f"f{f} [{'PULL' if f < rel0 else 'REL'}] w={width[f]:.3f}", fontsize=9)
    fig.colorbar(psc, ax=axs, shrink=0.6, label="stretch |sigma-1|")
    fig.suptitle(f"{cfg.scene}: pull->release (maxdev {maxdev:.3f}, GT logE {cfg.gt_logE})", fontsize=13)
    fig.savefig(os.path.join(out_dir, "forward_panel.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(width, "-o", ms=3, label="width (x-extent)")
    if cfg.scene != "release":
        ax.plot(minz, "-s", ms=3, label="min z (bottom)")
        ax.plot(comz, "-^", ms=3, label="com z")
    if sc.has_floor:
        ax.axhline(floor_z, color="cyan", ls="-", lw=1, label=f"floor {floor_z}")
    ax.axvline(rel0 - 1, color="orange", ls="--", label=f"release (f{rel0-1})")
    ax.set_xlabel("frame"); ax.set_ylabel("value")
    ax.set_title(f"{cfg.scene} observables (maxdev {maxdev:.3f})"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "observables.png"), dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    snap = fullX[rel0 - 1]
    psc = ax.scatter(snap[:, 0], snap[:, 2], c=F0_stretch, s=10, cmap="viridis")
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("z")
    ax.set_title(f"F0 snapshot stretch (mean {F0_stretch.mean():.3f}, max {F0_stretch.max():.3f})")
    fig.colorbar(psc, ax=ax, label="|sigma-1|")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "F0_stretch.png"), dpi=120); plt.close(fig)
    print(f"[case] forward viz -> forward_panel.png, observables.png, F0_stretch.png")

    # ---- result overlay (FIXTURE): GT vs each converged-E rollout (3d + triplane) ----
    # Defined here, CALLED after the fit (so fit_result.json exists) -- and in the
    # no-fit branch (reads a pre-existing json). Placing the call before the fit was
    # the bug: a fresh fit run had no json yet, so the overlay silently skipped.
    def make_overlay():
        ckpt_p = os.path.join(out_dir, "fit_result.json")
        if not (cfg.overlay_results and os.path.exists(ckpt_p)):
            return
        from matplotlib.animation import FuncAnimation, PillowWriter
        fr = json.load(open(ckpt_p))["results"]
        palette = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        items = [("GT", cfg.gt_logE, "black", gt_xyz.cpu().numpy())]
        for i, (k, r) in enumerate(sorted(fr.items())):
            items.append((f"{k}->{r['final']:.2f}", r["final"], palette[i % len(palette)],
                          rollout(r["final"])[0].cpu().numpy()))
        allp = np.concatenate([it[3].reshape(-1, 3) for it in items], 0)
        mn3, mx3 = allp.min(0), allp.max(0); Lr = items[0][3].shape[0]
        proj = [(0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z")]
        fig = plt.figure(figsize=(11, 9))
        ax3d = fig.add_subplot(2, 2, 1, projection="3d"); ax2 = [fig.add_subplot(2, 2, k) for k in (2, 3, 4)]

        def draw(f):
            ax3d.cla()
            for lbl, le, c, X in items:
                ax3d.scatter(X[f][:, 0], X[f][:, 1], X[f][:, 2], c=c, s=3, alpha=0.4, label=lbl, depthshade=False)
            ax3d.set_xlim(mn3[0], mx3[0]); ax3d.set_ylim(mn3[1], mx3[1]); ax3d.set_zlim(mn3[2], mx3[2])
            ax3d.set_title(f"{cfg.scene} release frame {f}/{Lr-1}", fontsize=9); ax3d.legend(fontsize=7, loc="upper left")
            for axp, (a, b, la, lb) in zip(ax2, proj):
                axp.cla()
                for lbl, le, c, X in items:
                    axp.scatter(X[f][:, a], X[f][:, b], c=c, s=4, alpha=0.4)
                if sc.has_floor and (a, b) == (0, 2):
                    axp.axhline(floor_z, color="cyan", lw=1)
                axp.set_xlim(mn3[a], mx3[a]); axp.set_ylim(mn3[b], mx3[b]); axp.set_aspect("equal")
                axp.set_xlabel(la); axp.set_ylabel(lb); axp.set_title(f"{la}{lb}")
            return ()

        draw(0)
        anim = FuncAnimation(fig, draw, frames=Lr, blit=False)
        anim.save(os.path.join(out_dir, "result_overlay.gif"), writer=PillowWriter(fps=cfg.overlay_fps)); plt.close(fig)
        sel = list(range(0, Lr, max(1, Lr // 8)))[:9]
        ncol = 3; nrow = (len(sel) + ncol - 1) // ncol
        fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow), squeeze=False)
        for ax in axs.flat:
            ax.axis("off")
        for a, f in enumerate(sel):
            ax = axs.flat[a]; ax.axis("on")
            for lbl, le, c, X in items:
                ax.scatter(X[f][:, 0], X[f][:, 2], c=c, s=5, alpha=0.45, label=lbl if a == 0 else None)
            if sc.has_floor:
                ax.axhline(floor_z, color="cyan", lw=1)
            ax.set_xlim(mn3[0], mx3[0]); ax.set_ylim(mn3[2], mx3[2]); ax.set_aspect("equal")
            ax.set_title(f"frame {f}", fontsize=9)
            if a == 0:
                ax.legend(fontsize=7)
        fig.suptitle(f"{cfg.scene}: GT vs converged-E rollouts (xz side view)", fontsize=13)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "result_overlay_panel.png"), dpi=110); plt.close(fig)
        print(f"[case] result overlay -> result_overlay.gif, result_overlay_panel.png  ({[it[0] for it in items]})")

    if not cfg.fit:
        make_overlay()   # uses a pre-existing fit_result.json if present
        print(f"[case] forward-only (--no-fit). DONE -> {out_dir} ({_time.time()-t0:.1f}s)")
        return out_dir

    # ---- FIT: FD+Adam per (loss, init), checkpointed ----
    NFFT = 256; freqs = np.fft.rfftfreq(NFFT, d=1.0)

    def width_np(xyz):
        w = xyz[:, :, 0].amax(1) - xyz[:, :, 0].amin(1)
        return w.cpu().numpy()

    def spec(w):
        return np.abs(np.fft.rfft(w - w.mean(), n=NFFT))

    def cen(sp):
        return float((freqs[1:] * sp[1:]).sum() / max(sp[1:].sum(), 1e-12))

    def _safe(v, big):
        return big if (not np.isfinite(v)) else v

    gt_w = width_np(gt_xyz); gt_sp = spec(gt_w); gt_c = cen(gt_sp)

    def L_time(logE):  return _safe(float(((rollout(logE)[0] - gt_xyz) ** 2).sum(-1).mean()), 1.0)
    def L_spec(logE):  return _safe(float(((spec(width_np(rollout(logE)[0])) - gt_sp) ** 2).mean()), 1e6)
    def L_cen(logE):   return _safe((cen(spec(width_np(rollout(logE)[0]))) - gt_c) ** 2, 1.0)
    ALL = {"time_L2": L_time, "spectral": L_spec, "centroid": L_cen}

    # combined = L_time/c_t + lam*L_spec/c_s, both normalized by a FIXED ref-E loss
    # (NOT the init -- if init==GT the loss->0 and the divisor blows up). One rollout/eval.
    if "combined" in cfg.losses:
        ref_traj = rollout(cfg.ref_logE)[0]
        c_t = max(float(((ref_traj - gt_xyz) ** 2).sum(-1).mean()), 1e-30)
        c_s = max(float(((spec(width_np(ref_traj)) - gt_sp) ** 2).mean()), 1e-30)
        print(f"[case] combined norm: c_t={c_t:.3e} c_s={c_s:.3e} (ref logE {cfg.ref_logE}, lam {cfg.lam})")

        def L_comb(logE):
            traj = rollout(logE)[0]
            lt = _safe(float(((traj - gt_xyz) ** 2).sum(-1).mean()), 1.0)
            ls = _safe(float(((spec(width_np(traj)) - gt_sp) ** 2).mean()), 1e6)
            return lt / c_t + cfg.lam * ls / c_s
        ALL["combined"] = L_comb
    LOSSES = {k: ALL[k] for k in cfg.losses}

    ckpt = os.path.join(out_dir, "fit_result.json")
    blob = json.load(open(ckpt)) if os.path.exists(ckpt) else {}
    results = blob.get("results", {})
    if results:
        print(f"[case] resume: {len(results)} combos done")

    def save():
        with open(ckpt, "w") as f:
            json.dump({"scene": cfg.scene, "gt_logE": cfg.gt_logE, "maxdev": maxdev,
                       "lr": cfg.lr, "n_iters": cfg.n_iters, "results": results}, f, indent=2)

    b1, b2 = 0.9, 0.999
    for lname, Lf in LOSSES.items():
        for E0 in cfg.inits:
            key = f"{lname}|init{E0}"
            r = results.get(key)
            if r and r.get("done"):
                continue
            if r:
                logE = r["logE"]; m = r["m"]; v = r["v"]; start = r["iter"]; traj = r["traj"]; losst = r["loss"]
                print(f"[case] resume {key} from iter {start}")
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
                    save()
            print(f"[case] {lname:9s} init {E0} -> final {logE:.3f} (err {logE-cfg.gt_logE:+.3f} dex "
                  f"= x{10**(logE-cfg.gt_logE):.2f}) [done]")

    # ---- fit viz: loss curves + logE trajectories ----
    losses = list(LOSSES)
    fig, axs = plt.subplots(1, len(losses), figsize=(5.4 * len(losses), 4.2), squeeze=False)
    for ax, lname in zip(axs[0], losses):
        for E0 in cfg.inits:
            r = results.get(f"{lname}|init{E0}")
            if r:
                lo = r["loss"]; rng = (max(lo) - min(lo)) / max(max(lo), 1e-30)
                ax.plot(lo, "-o", ms=3, label=f"init {E0}->{r['final']:.2f}  (range {rng:.0%})")
        ax.set_yscale("log"); ax.set_title(f"{cfg.scene}  {lname}  [log y -- watch the range%]")
        ax.set_xlabel("iter"); ax.set_ylabel("loss (log)"); ax.legend(fontsize=8)
    fig.suptitle(f"{cfg.scene}: LOSS vs iter (GT logE {cfg.gt_logE})", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=120); plt.close(fig)

    fig, axs = plt.subplots(1, len(losses), figsize=(5.4 * len(losses), 4.2), squeeze=False, sharey=True)
    for ax, lname in zip(axs[0], losses):
        for E0 in cfg.inits:
            r = results.get(f"{lname}|init{E0}")
            if r:
                ax.plot(r["traj"], "-o", ms=3, label=f"init {E0}->{r['final']:.2f}")
        ax.axhline(cfg.gt_logE, color="k", ls="--", lw=1)
        ax.set_title(f"{cfg.scene}  {lname}"); ax.set_xlabel("iter"); ax.set_ylabel("log10 E"); ax.legend(fontsize=8)
    fig.suptitle(f"{cfg.scene}: log10 E vs iter (GT {cfg.gt_logE})", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "logE_traj.png"), dpi=120); plt.close(fig)

    # ---- signal/spectrum across E (the observable) ----
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for le in cfg.sig_logEs:
        w = width_np(rollout(le)[0])
        axs[0].plot(w, lw=1.3, label=f"logE {le}")
        axs[1].plot(freqs[:40], spec(w)[:40], lw=1.3, label=f"logE {le}")
    axs[0].plot(gt_w, "k--", lw=1.6, label=f"GT {cfg.gt_logE}"); axs[0].set_title("width(t)")
    axs[0].set_xlabel("frame"); axs[0].legend(fontsize=7)
    axs[1].plot(freqs[:40], gt_sp[:40], "k--", lw=1.6); axs[1].set_title("|FFT(width)|")
    axs[1].set_xlabel("freq (1/frame)"); axs[1].legend(fontsize=7)
    fig.suptitle(f"{cfg.scene}: observable width(t) + spectrum across E", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "signals_and_spectra.png"), dpi=120); plt.close(fig)

    make_overlay()   # FIXTURE: fit_result.json now written -> overlay the converged Es
    print(f"[case] DONE -> {out_dir} ({_time.time()-t0:.1f}s)")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(FitCaseConfig))
