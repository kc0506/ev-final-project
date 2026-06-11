"""Controlled 2D sweep: which loss WINDOW does the (best-case) MPM gradient support?

Baseline (locked): global scalar E, PIXEL loss, telephone, GT uniform E, substep=64.
We vary TWO axes only: window and init_E (factor x true_E). grad_window is tied to
window (=window -> FULL BPTT, no truncation) so each window gets its BEST gradient;
a failure at window W then means "W frames of BPTT is genuinely unsupported", not a
truncation artifact (truncation cost is a separate later sweep). init is swept so a
non-convergence at (W, init) can be told apart from "this W just needs a different
init" -- a window counts as usable if ANY init converges.

  python -m reuse_mpm.explore.sweep_window --gt_run outputs/forward_gen/06_tele_E1e5 \
      --windows 1 2 3 4 --init_factors 0.1 0.3 0.6 1.5 4.0 10.0
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import tyro

from ..config import SceneSpec, SimConfig
from ..gpu import pick_free_gpu


@dataclass
class SweepWindowConfig:
    gt_run: str
    windows: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    init_factors: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.6, 1.5, 4.0, 10.0])
    # grad_window per window: "full" -> grad_window=window (full BPTT, the controlled
    # choice); or an int string ("1","2") to fix it (a different controlled setup).
    grad_window_mode: str = "full"
    iters: int = 120
    lr: float = 0.1
    out: Optional[str] = None
    run_label: str = ""


def _load_gt_frames(gt_run: str, device: str):
    import imageio.v2 as imageio
    import torch
    fs = sorted(glob.glob(os.path.join(gt_run, "frames", "frame_*.png")))
    assert fs, f"no frames in {gt_run}/frames"
    arr = np.stack([imageio.imread(f) for f in fs], 0).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device)


def run(cfg: SweepWindowConfig):
    pick_free_gpu()
    from ..scene_io import load_from_spec
    from ..sim_render import make_constant_v0
    from ..recover import recover_global_E
    from ..run_io import RunDir

    rd = RunDir.create(__name__, cfg.run_label, cfg.out, config=cfg)
    with rd.capture_output():
        g = json.load(open(os.path.join(cfg.gt_run, "config.json")))
        spec = SceneSpec(**g["scene"]); sim = SimConfig(**g["sim"])
        true_E = float(g["E"]); v0_vec = g["v0"]; frame = g["frame"]; dev = spec.device
        gt = _load_gt_frames(cfg.gt_run, dev)
        scene = load_from_spec(spec, sim)
        try:
            cam = scene.camera_by_frame(frame)
        except Exception:
            cam = scene.test_camera_list[0]
        v0 = make_constant_v0(scene, v0_vec).detach()

        # grid[i_window, j_factor] = rel_err (nan if diverged)
        nW, nF = len(cfg.windows), len(cfg.init_factors)
        rel = np.full((nW, nF), np.nan)
        points = []
        prog_path = rd.path("progress.jsonl")   # one line per finished cell (live progress)
        n_total = nW * nF; n_done = 0
        import time as _time
        for i, W in enumerate(cfg.windows):
            gw = W if cfg.grad_window_mode == "full" else int(cfg.grad_window_mode)
            for j, fac in enumerate(cfg.init_factors):
                t0 = _time.time()
                res = recover_global_E(
                    scene, gt, sim, cam, v0, init_E=fac * true_E, iters=cfg.iters,
                    lr=cfg.lr, window=W, grad_window=gw, coarse_init=False,
                    true_E=true_E, device=dev)
                rel[i, j] = res["rel_err"] if np.isfinite(res["recovered_E"]) else np.nan
                cell = {"window": W, "grad_window": gw, "init_factor": fac,
                        "recovered_E": res["recovered_E"], "rel_err": res["rel_err"],
                        "final_loss": res["final_loss"], "sec": round(_time.time() - t0, 1),
                        "loss_traj": res["loss_traj"]}  # keep the curve (research-discipline)
                points.append(cell)
                n_done += 1
                with open(prog_path, "a") as pf:  # progress stays light (drop full traj)
                    pf.write(json.dumps({k: v for k, v in cell.items() if k != "loss_traj"}) + "\n")
                tag = f"{rel[i,j]*100:.0f}%" if np.isfinite(rel[i, j]) else "NaN"
                print(f"  [{n_done}/{n_total}] window={W} gw={gw} init={fac:.2g}x -> {tag} "
                      f"(rec={res['recovered_E']:.3e}, loss={res['final_loss']:.2e}, "
                      f"{cell['sec']}s)", flush=True)

        rd.write_json("sweep.json", {
            "baseline": {"loss": "pixel", "global": True, "scene": scene.name,
                         "true_E": true_E, "substep": sim.substep,
                         "grad_window_mode": cfg.grad_window_mode},
            "windows": cfg.windows, "init_factors": cfg.init_factors,
            "rel_err_grid": rel.tolist(), "points": points})
        _heatmap(rd.path("sweep_window.png"), rel, cfg.windows, cfg.init_factors,
                 scene.name, cfg.grad_window_mode)
        _loss_grid(rd.path("sweep_window_loss.png"), points, cfg.windows,
                   cfg.init_factors, scene.name)
        rd.finish()
        print(f"[sweep_window] {nW}x{nF} grid -> {rd.path('sweep_window.png')}")
    return rd


def _heatmap(path, rel, windows, factors, scene_name, gw_mode):
    """rows=window, cols=init factor. Colour = rel_err% (log). NaN cells = grey 'NaN'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    pct = rel * 100.0
    fig, ax = plt.subplots(figsize=(1.6 * len(factors) + 2, 1.0 * len(windows) + 2))
    masked = np.ma.masked_invalid(pct)
    vmin = max(1e-2, np.nanmin(pct)) if np.isfinite(pct).any() else 1e-2
    vmax = max(vmin * 10, np.nanmax(pct)) if np.isfinite(pct).any() else 1e2
    cmap = plt.cm.viridis_r.copy(); cmap.set_bad("lightgrey")
    im = ax.imshow(masked, aspect="auto", cmap=cmap,
                   norm=LogNorm(vmin=vmin, vmax=vmax), origin="lower")
    ax.set_xticks(range(len(factors))); ax.set_xticklabels([f"{f:g}x" for f in factors])
    ax.set_yticks(range(len(windows))); ax.set_yticklabels([f"w={w}" for w in windows])
    ax.set_xlabel("init E (x true_E)"); ax.set_ylabel("loss window")
    for i in range(len(windows)):
        for j in range(len(factors)):
            v = pct[i, j]
            txt = "NaN" if not np.isfinite(v) else (f"{v:.0f}%" if v >= 1 else f"{v:.1f}%")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="red" if not np.isfinite(v) else "white")
    fig.colorbar(im, ax=ax, label="rel err % (log)")
    ax.set_title(f"{scene_name} | global+pixel+substep64 | grad_window={gw_mode} | "
                 f"window x init  (lower=better; NaN=diverged)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _loss_grid(path, points, windows, factors, scene_name):
    """One subplot per window; within it, a loss-vs-iter curve per init factor.
    Diverged cells (NaN-ending) drawn dashed. The deciding view the heatmap hides."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    by_w = {W: [] for W in windows}
    for c in points:
        by_w[c["window"]].append(c)
    n = len(windows)
    fig, ax = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
    for k, W in enumerate(windows):
        a = ax[0][k]
        for c in sorted(by_w[W], key=lambda c: c["init_factor"]):
            lt = c.get("loss_traj") or []
            diverged = not np.isfinite(c["recovered_E"])
            a.plot(lt, ("--" if diverged else "-"), lw=1,
                   label=f"{c['init_factor']:g}x" + (" NaN" if diverged else ""))
        a.set_yscale("log"); a.set_xlabel("iter"); a.set_ylabel("pixel loss")
        a.set_title(f"window={W}"); a.legend(fontsize=6, ncol=2)
    fig.suptitle(f"{scene_name} | LOSS vs iter per (window, init) | dashed=diverged")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    run(tyro.cli(SweepWindowConfig))
