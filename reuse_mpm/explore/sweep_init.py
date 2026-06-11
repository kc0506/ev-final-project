"""Controlled sweep: hold the baseline fixed, vary ONLY init_E, one figure out.

Baseline (locked, = the setup where global E recovered to 0%, run 03_scalar_w1g1):
  global scalar E, PIXEL loss, window=1, grad_window=1, telephone, GT uniform E.
We change exactly ONE variable here: init_E. Output answers, cleanly, "from how
far does global+pixel+window=1 recover E?" -- the basin, the baseline every other
axis is later compared against. No other variable moves.

Scene + sim + v0 + frame + true_E are read from the GT run (same contract as
train_global_E), so nothing is re-specified.

  python -m reuse_mpm.explore.sweep_init --gt_run outputs/forward_gen/06_tele_E1e5 \
      --init_Es 1.1e5 1.2e5 1.5e5 2e5 3e5 5e5 1e6
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
class SweepInitConfig:
    gt_run: str
    # init points as MULTIPLIERS of the GT's true_E, so they BRACKET GT on both
    # sides (earlier bug: all inits were > GT, untested below). init_E = factor*true_E.
    init_factors: List[float] = field(default_factory=lambda: [
        0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 2.0, 3.0, 5.0, 10.0])
    # baseline (do NOT change these in this sweep -- they define the controlled setup)
    window: int = 1
    grad_window: int = 1
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


def run(cfg: SweepInitConfig):
    pick_free_gpu()
    from ..scene_io import load_from_spec
    from ..sim_render import make_constant_v0
    from ..recover import recover_global_E
    from ..run_io import RunDir

    rd = RunDir.create(__name__, cfg.run_label, cfg.out, config=cfg)
    with rd.capture_output():
        g = json.load(open(os.path.join(cfg.gt_run, "config.json")))
        scene_spec = SceneSpec(**g["scene"]) if isinstance(g.get("scene"), dict) else None
        assert scene_spec is not None, "expects current ForwardConfig schema GT"
        sim = SimConfig(**g["sim"])
        true_E = float(g["E"]); v0_vec = g["v0"]; frame = g["frame"]
        dev = scene_spec.device

        gt = _load_gt_frames(cfg.gt_run, dev)               # [T,C,H,W]
        scene = load_from_spec(scene_spec, sim)             # loaded ONCE
        try:
            cam = scene.camera_by_frame(frame)
        except Exception:
            cam = scene.test_camera_list[0]
        v0 = make_constant_v0(scene, v0_vec).detach()

        init_Es = [f * true_E for f in cfg.init_factors]  # bracket GT both sides
        results = []  # (init_E, recovered_E, rel_err, final_loss, loss_traj, E_traj)
        for E0 in init_Es:
            res = recover_global_E(
                scene, gt, sim, cam, v0, init_E=float(E0), iters=cfg.iters, lr=cfg.lr,
                window=cfg.window, grad_window=cfg.grad_window,
                coarse_init=False, true_E=true_E, device=dev)
            results.append((float(E0), res["recovered_E"], res["rel_err"],
                            res["final_loss"], res["loss_traj"], res["E_traj"]))
            print(f"  init={E0:.2e} -> recovered={res['recovered_E']:.3e} "
                  f"rel_err={res['rel_err']*100:.0f}% final_loss={res['final_loss']:.2e}")

        rd.write_json("sweep.json", {
            "baseline": {"loss": "pixel", "window": cfg.window,
                         "grad_window": cfg.grad_window, "scene": scene.name,
                         "true_E": true_E, "global": True},
            "points": [{"init_E": r[0], "recovered_E": r[1], "rel_err": r[2],
                        "final_loss": r[3], "loss_traj": r[4], "E_traj": r[5]}
                       for r in results]})
        _plot(rd.path("sweep_init.png"), results, true_E, scene.name)
        rd.finish()
        print(f"[sweep_init] {len(results)} inits swept (pixel,w{cfg.window},g{cfg.grad_window}) "
              f"-> {rd.path('sweep_init.png')}")
    return rd


def _plot(path, results, true_E, scene_name):
    """Three panels sharing one log-x (init E). Diverged/NaN inits are drawn as red
    'x' at a sentinel (never silently dropped), so all swept inits appear in every
    panel consistently."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    inits = np.array([r[0] for r in results])
    rec = np.array([r[1] for r in results])
    rel = np.array([r[2] for r in results]) * 100.0  # percent
    conv = np.isfinite(rec)                            # converged mask
    xlim = (inits.min() / 1.5, inits.max() * 1.5)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    # --- A: recovered E vs init E (log-log) ---
    ax[0].plot(inits[conv], rec[conv], "o-", color="C0", label="recovered (converged)")
    ax[0].plot(inits, inits, "k:", lw=0.7, label="y=x (didn't move)")
    ax[0].axhline(true_E, color="r", ls="--", label=f"true {true_E:.1e}")
    ax[0].axvline(true_E, color="gray", ls=":", lw=0.7)
    div_y = true_E * 30
    if (~conv).any():
        ax[0].scatter(inits[~conv], [div_y] * int((~conv).sum()),
                      marker="x", color="red", s=70, label="diverged (NaN)")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlim(*xlim)
    ax[0].set_xlabel("init E (log)"); ax[0].set_ylabel("recovered E")
    ax[0].set_title("recovered vs init"); ax[0].legend(fontsize=7)
    # --- B: rel err % vs init E (log-x, log-y) ---
    ax[1].plot(inits[conv], np.clip(rel[conv], 1e-2, None), "o-", color="C0",
               label="converged")
    if (~conv).any():
        ax[1].scatter(inits[~conv], [1e3] * int((~conv).sum()),
                      marker="x", color="red", s=70, label="diverged (@1e3)")
    ax[1].axvline(true_E, color="gray", ls=":", lw=0.7)
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlim(*xlim)
    ax[1].set_xlabel("init E (log)"); ax[1].set_ylabel("rel err %")
    ax[1].set_title("recovery error vs init"); ax[1].legend(fontsize=7)
    # --- C: loss curves, labelled by init/true factor ---
    for r in results:
        ls = "--" if not np.isfinite(r[1]) else "-"
        ax[2].plot(r[4], ls, lw=0.9, label=f"{r[0]/true_E:.1f}x")
    ax[2].set_yscale("log"); ax[2].set_xlabel("iter"); ax[2].set_ylabel("pixel loss")
    ax[2].set_title("loss vs iter (label = init/true)"); ax[2].legend(fontsize=6, ncol=2)
    fig.suptitle(f"{scene_name} | baseline: global+pixel+window=1+grad_window=1 | "
                 f"vary ONLY init (x true_E={true_E:.0e})")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    run(tyro.cli(SweepInitConfig))
