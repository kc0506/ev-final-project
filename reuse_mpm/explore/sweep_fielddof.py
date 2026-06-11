"""Controlled sweep: at what FIELD DOF does convergence break? (dilution vs identifiability)

Baseline (locked, = the scalar-stable recipe from Exp2): pixel loss, window=3,
grad_window=3, telephone, init=3e5. We vary ONE axis: the voxel field resolution
res (DOF = res^3). Crucially the GT is UNIFORM E=1e5, which is identifiable by
construction -- so if the field still fails to recover the uniform value as res
grows, the bottleneck is PURE gradient dilution / parametrization, NOT
identifiability. That cleanly separates the two hypotheses conflated earlier.

res=2 (8 cells) is the lowest-DOF field, the closest thing to the known-good
scalar; if even it fails, the field machinery itself is suspect, not the DOF.

  python -m reuse_mpm.explore.sweep_fielddof --gt_run outputs/forward_gen/06_tele_E1e5 \
      --resolutions 2 4 8 16
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
class SweepFieldDofConfig:
    gt_run: str
    resolutions: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    backbone: str = "voxel"
    init_E: float = 3e5
    window: int = 3
    grad_window: int = 3
    iters: int = 100
    lr: float = 0.05
    reg_weight: float = 1e-3
    min_quota_hours: float = 8.0  # pick_free_gpu floor; set 0 to override (low quota)
    out: Optional[str] = None
    run_label: str = ""


def _load_gt_frames(gt_run: str, device: str):
    import imageio.v2 as imageio
    import torch
    fs = sorted(glob.glob(os.path.join(gt_run, "frames", "frame_*.png")))
    assert fs, f"no frames in {gt_run}/frames"
    arr = np.stack([imageio.imread(f) for f in fs], 0).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device)


def run(cfg: SweepFieldDofConfig):
    pick_free_gpu(min_quota_hours=cfg.min_quota_hours)
    import torch
    from ..scene_io import load_from_spec
    from ..sim_render import make_constant_v0
    from ..efield import EField
    from ..recover import recover_field_E
    from ..run_io import RunDir

    rd = RunDir.create(__name__, cfg.run_label, cfg.out, config=cfg)
    with rd.capture_output():
        g = json.load(open(os.path.join(cfg.gt_run, "config.json")))
        spec = SceneSpec(**g["scene"]); sim = SimConfig(**g["sim"])
        true_E = float(g["E"]); v0_vec = g["v0"]; frame = g["frame"]; dev = spec.device
        assert g.get("E_grad_axis") is None, "this sweep expects a UNIFORM-E GT"
        gt = _load_gt_frames(cfg.gt_run, dev)
        scene = load_from_spec(spec, sim)
        try:
            cam = scene.camera_by_frame(frame)
        except Exception:
            cam = scene.test_camera_list[0]
        v0 = make_constant_v0(scene, v0_vec).detach()

        prog = rd.path("progress.jsonl")
        rows = []
        for res in cfg.resolutions:
            fld = EField(scene.sim_aabb, backbone=cfg.backbone, init_E=cfg.init_E,
                         res=res).to(dev)
            npar = sum(p.numel() for p in fld.parameters())
            r = recover_field_E(
                scene, gt, sim, cam, v0, field=fld, iters=cfg.iters, lr=cfg.lr,
                window=cfg.window, grad_window=cfg.grad_window,
                reg_weight=cfg.reg_weight, true_E=true_E, device=dev)
            row = {"res": res, "n_params": npar,
                   "recovered_geomean_E": r["recovered_geomean_E"],
                   "rel_err_geomean": r.get("rel_err_geomean"),
                   "E_min": float(np.min(r["E_final"])), "E_max": float(np.max(r["E_final"])),
                   "final_loss": r["final_loss"], "min_loss": r["min_loss"],
                   "loss_traj": r["loss_traj"]}  # per-iter loss (research-discipline: keep the curve)
            rows.append(row)
            with open(prog, "a") as pf:  # progress line stays light (no full traj)
                pf.write(json.dumps({k: v for k, v in row.items() if k != "loss_traj"}) + "\n")
            print(f"  res={res} ({npar}p) -> geomean={row['recovered_geomean_E']:.3e} "
                  f"({row['rel_err_geomean']*100:.0f}%) loss={row['final_loss']:.2e} "
                  f"E:[{row['E_min']:.2e},{row['E_max']:.2e}]", flush=True)

        rd.write_json("sweep.json", {
            "baseline": {"loss": "pixel", "window": cfg.window,
                         "grad_window": cfg.grad_window, "GT": "uniform",
                         "true_E": true_E, "init_E": cfg.init_E},
            "rows": rows})
        _plot(rd.path("sweep_fielddof.png"), rows, true_E, scene.name)
        rd.finish()
        print(f"[sweep_fielddof] {len(rows)} resolutions -> {rd.path('sweep_fielddof.png')}")
    return rd


def _plot(path, rows, true_E, scene_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = [r["res"] for r in rows]
    rel = [r["rel_err_geomean"] * 100 for r in rows]
    emin = [r["E_min"] for r in rows]; emax = [r["E_max"] for r in rows]
    geo = [r["recovered_geomean_E"] for r in rows]
    # 4 panels: LOSS CURVES first (the deciding view), then the 3 endpoint summaries
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.5))
    for r in rows:  # one loss-vs-iter curve per resolution
        ax[0].plot(r["loss_traj"], label=f"res={r['res']} ({r['n_params']}p)")
    ax[0].axhline(5e-6, color="g", ls="--", lw=0.8, label="scalar basin ~5e-6")
    ax[0].set_yscale("log"); ax[0].set_xlabel("iter"); ax[0].set_ylabel("pixel loss")
    ax[0].set_title("LOSS vs iter, per res (converged? plateaued?)"); ax[0].legend(fontsize=7)
    ax[1].plot(res, rel, "o-"); ax[1].set_xlabel("voxel res (DOF=res^3)")
    ax[1].set_ylabel("geomean rel err %"); ax[1].set_title("recovery error vs field DOF")
    ax[2].plot(res, [r["final_loss"] for r in rows], "o-", label="final")
    ax[2].plot(res, [r["min_loss"] for r in rows], "s--", label="min")
    ax[2].axhline(5e-6, color="g", ls="--", label="scalar basin")
    ax[2].set_yscale("log"); ax[2].set_xlabel("voxel res"); ax[2].set_ylabel("pixel loss")
    ax[2].set_title("final/min loss vs DOF"); ax[2].legend(fontsize=8)
    ax[3].plot(res, geo, "o-", label="geomean")
    ax[3].fill_between(res, emin, emax, alpha=0.2, label="E min..max")
    ax[3].axhline(true_E, color="r", ls="--", label=f"true {true_E:.0e}")
    ax[3].set_yscale("log"); ax[3].set_xlabel("voxel res"); ax[3].set_ylabel("recovered E")
    ax[3].set_title("recovered E spread vs DOF"); ax[3].legend(fontsize=8)
    fig.suptitle(f"{scene_name} | UNIFORM GT | pixel+window=3+grad_window=3 | "
                 f"vary ONLY field res (identifiability removed -> isolates gradient dilution)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    run(tyro.cli(SweepFieldDofConfig))
