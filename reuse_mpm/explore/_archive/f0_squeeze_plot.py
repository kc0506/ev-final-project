"""Plot the squeeze-sweep checkpoint (loss curves FIRST, then logE trajectories).

Reads outputs/explore/f0_block_squeeze_sweep/<label>/fit_result.json and draws from
whatever combos exist (tolerates a partial / still-running sweep). Rerun anytime to
refresh as more combos finish.

  loss_curves.png       rows=R, cols=loss-type; y=loss (log), x=iter; inits overlaid
  fit_trajectories.png  rows=R, cols=loss-type; y=log10 E, x=iter; GT dashed

Usage: python -m reuse_mpm.explore.f0_squeeze_plot [--label ...]
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import tyro


@dataclass
class PlotConfig:
    label: str = "block_squeeze_sweep"
    overlay_fps: int = 3       # 0.5x of the forward gif's native fps 6 -> slow motion
    overlay_stride: int = 1    # subsample frames (1 = every frame)


def run(cfg: PlotConfig) -> str:
    import glob
    import re

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    out_dir = os.path.join("outputs", "explore", "f0_block_squeeze_sweep", cfg.label)
    ckpt = os.path.join(out_dir, "fit_result.json")
    blob = json.load(open(ckpt))
    results = blob["results"]; meta = blob.get("meta", {}); gt = blob["gt_logE"]

    # discover which R / losses / inits actually exist
    Rs, losses, inits = set(), set(), set()
    for k in results:
        rpart, lname, ipart = k.split("|")
        Rs.add(int(rpart[1:])); losses.add(lname); inits.add(float(ipart[4:]))
    Rs = sorted(Rs); losses = ["time_L2", "spectral", "centroid"]
    losses = [l for l in losses if l in {k.split("|")[1] for k in results}]
    inits = sorted(inits)
    print(f"[plot] R={Rs} losses={losses} inits={inits} ({len(results)} combos)")

    # ---- (1) LOSS CURVES -- the first thing to look at ----
    fig, axs = plt.subplots(len(Rs), len(losses), figsize=(5.2 * len(losses), 3.8 * len(Rs)),
                            squeeze=False)
    for i, R in enumerate(Rs):
        for j, lname in enumerate(losses):
            axp = axs[i][j]
            any_plot = False
            for E0 in inits:
                r = results.get(f"R{R}|{lname}|init{E0}")
                if r is None:
                    continue
                losst = r["loss"]
                axp.plot(losst, "-o", ms=2.5,
                         label=f"init {E0}->{r['final']:.2f} (it{r['iter']})")
                any_plot = True
            if any_plot:
                axp.set_yscale("log")
            md = meta.get(str(R), {}).get("maxdev")
            mdtxt = f" maxdev {md:.3f}" if md is not None else ""
            axp.set_title(f"R={R}{mdtxt}  {lname}")
            axp.set_xlabel("iter"); axp.set_ylabel("loss (log)")
            axp.legend(fontsize=7)
    fig.suptitle("LOSS vs iter -- does the loss actually go down to the GT basin?", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=120); plt.close(fig)

    # ---- (2) logE trajectories ----
    fig, axs = plt.subplots(len(Rs), len(losses), figsize=(5.2 * len(losses), 3.8 * len(Rs)),
                            squeeze=False, sharey=True)
    for i, R in enumerate(Rs):
        for j, lname in enumerate(losses):
            axp = axs[i][j]
            for E0 in inits:
                r = results.get(f"R{R}|{lname}|init{E0}")
                if r is None:
                    continue
                axp.plot(r["traj"], "-o", ms=2.5, label=f"init {E0}->{r['final']:.2f}")
            axp.axhline(gt, color="k", ls="--", lw=1)
            md = meta.get(str(R), {}).get("maxdev")
            mdtxt = f" maxdev {md:.3f}" if md is not None else ""
            axp.set_title(f"R={R}{mdtxt}  {lname}")
            axp.set_xlabel("iter")
            if j == 0:
                axp.set_ylabel("log10 E")
            axp.legend(fontsize=7)
    fig.suptitle(f"log10 E vs iter (GT {gt}) -- where does the optimizer settle?", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fit_trajectories.png"), dpi=120); plt.close(fig)

    print(f"[plot] -> {out_dir}/loss_curves.png , fit_trajectories.png")

    # ---- (3) 3-version OVERLAY gif (3d + triplane), aligned on ABSOLUTE frame ----
    # Reads the forward traj_R*.npz (no recompute). Pull is deterministic + shared BC,
    # so the versions are bit-identical until each one's snapshot frame R, then peel off.
    npz_paths = sorted(glob.glob(os.path.join(out_dir, "traj_R*.npz")),
                       key=lambda p: int(re.search(r"traj_R(\d+)", p).group(1)))
    if len(npz_paths) >= 2:
        data = []
        for p in npz_paths:
            d = np.load(p)
            R = int(re.search(r"traj_R(\d+)", p).group(1))
            data.append((R, d["X"], int(d["rel_start"])))
        L = min(d[1].shape[0] for d in data)          # common absolute-frame length
        L = (L // cfg.overlay_stride) * cfg.overlay_stride
        allX = np.concatenate([d[1][:L].reshape(-1, 3) for d in data], 0)
        mn, mx = allX.min(0), allX.max(0)
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        proj = [(0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z")]
        frames = list(range(0, L, cfg.overlay_stride))

        fig = plt.figure(figsize=(11, 9))
        ax3d = fig.add_subplot(2, 2, 1, projection="3d")
        ax2 = [fig.add_subplot(2, 2, k) for k in (2, 3, 4)]

        def draw(f):
            ax3d.cla()
            tags = []
            for (R, X, rel0), c in zip(data, colors):
                ax3d.scatter(X[f][:, 0], X[f][:, 1], X[f][:, 2], c=c, s=3, alpha=0.45,
                             label=f"R={R}", depthshade=False)
                tags.append(f"R{R}:" + (f"PULL{f}" if f < rel0 else f"REL+{f-rel0+1}"))
            ax3d.set_xlim(mn[0], mx[0]); ax3d.set_ylim(mn[1], mx[1]); ax3d.set_zlim(mn[2], mx[2])
            ax3d.set_title(f"abs frame {f}/{L-1}\n" + "  ".join(tags), fontsize=9)
            ax3d.legend(fontsize=8, loc="upper left")
            for axp, (a, b, la, lb) in zip(ax2, proj):
                axp.cla()
                for (R, X, rel0), c in zip(data, colors):
                    axp.scatter(X[f][:, a], X[f][:, b], c=c, s=4, alpha=0.45)
                axp.set_xlim(mn[a], mx[a]); axp.set_ylim(mn[b], mx[b]); axp.set_aspect("equal")
                axp.set_xlabel(la); axp.set_ylabel(lb); axp.set_title(f"{la}{lb}")
            return ()

        draw(0)
        anim = FuncAnimation(fig, draw, frames=frames, blit=False)
        gif = os.path.join(out_dir, "overlay_3versions.gif")
        anim.save(gif, writer=PillowWriter(fps=cfg.overlay_fps)); plt.close(fig)
        print(f"[plot] -> {gif}  (Rs={[d[0] for d in data]}, {len(frames)}f @ {cfg.overlay_fps}fps = 0.5x)")
    else:
        print(f"[plot] overlay gif skipped (need >=2 traj_R*.npz, found {len(npz_paths)})")

    return out_dir


if __name__ == "__main__":
    run(tyro.cli(PlotConfig))
