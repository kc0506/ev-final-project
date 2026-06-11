"""Explore: measure the v0 photometric-loss dynamic range + sensitivity.

train_v0 found that the loss is nearly flat (~2.8e-3..5.9e-3) while the recovered v0
wanders far from GT. Before tuning lr/window blindly, this diagnostic measures WHY:
is the loss insensitive to v0 (dominated by static background -> v0 unidentifiable),
or is there real signal that the optimiser just fails to follow?

Renders a handful of v0 settings (rest, GT, scaled-GT, wrong directions, the wandered
solutions) with no_grad, and reports per-frame photometric MSE vs GT, both over the
FULL frame and over the bounding box of the object's motion (the moving-pixel mask),
so we can see how much of the loss is background vs the actual moving object.

  python -m reuse_mpm.explore.v0_loss_landscape --gt_run outputs/forward_gen/06_tele_E1e5

Config is LOCAL (explore convention); only reads SceneSpec/SimConfig.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import tyro

from ..config import SceneSpec, SimConfig
from ..gpu import pick_free_gpu


@dataclass
class V0LossLandscapeConfig:
    gt_run: str
    n_frames: int = 4  # measure loss on the first n_frames (from t=1)
    out: Optional[str] = None
    run_label: str = ""


def _load_gt(gt_run: str, device: str):
    import imageio.v2 as imageio
    import torch
    fps = sorted(glob.glob(os.path.join(gt_run, "frames", "frame_*.png")))
    arr = np.stack([imageio.imread(p) for p in fps], 0).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device)  # [T,C,H,W]


def run(cfg: V0LossLandscapeConfig) -> str:
    pick_free_gpu()
    import torch
    from ..scene_io import load_from_spec
    from ..sim_render import make_constant_v0, simulate_and_render
    from ..run_io import RunDir

    with open(os.path.join(cfg.gt_run, "config.json")) as f:
        g = json.load(f)
    scene_spec = SceneSpec(**g["scene"]) if isinstance(g.get("scene"), dict) else None
    sim = SimConfig(**g["sim"])
    E = float(g["E"]); gt_v0 = g["v0"]; frame = g["frame"]
    device = scene_spec.device

    rd = RunDir.create(__name__, cfg.run_label or os.path.basename(os.path.normpath(cfg.gt_run)), cfg.out)
    gt = _load_gt(cfg.gt_run, device)                          # [T,C,H,W]
    scene = load_from_spec(scene_spec, sim)
    try:
        cam = scene.camera_by_frame(frame)
    except Exception:
        cam = scene.test_camera_list[0]

    gx, gy, gz = gt_v0
    probes: List[Tuple[str, Tuple[float, float, float]]] = [
        ("rest_v0=0", (0.0, 0.0, 0.0)),
        ("GT", (gx, gy, gz)),
        ("GT_x0.5", (gx * 0.5, gy * 0.5, gz * 0.5)),
        ("GT_x2", (gx * 2, gy * 2, gz * 2)),
        ("flip_y", (gx, -gy, gz)),
        ("wander_global", (0.235, -0.672, 0.665)),
        ("wander_triplane", (-1.163, 1.399, 2.067)),
    ]
    nf = min(cfg.n_frames, gt.shape[0] - 1)

    # moving-pixel mask: where GT frames differ from the rest frame (object motion).
    rest = simulate_and_render(scene, E, make_constant_v0(scene, (0, 0, 0)), sim, cam).detach()  # [T,C,H,W]
    diff = (gt[1:nf + 1] - rest[1:nf + 1]).abs().mean(1)        # [nf,H,W] over channels
    mask = (diff > 0.02).float()                               # [nf,H,W] moving pixels
    frac = float(mask.mean())
    print(f"moving-pixel fraction (|GT-rest|>0.02): {frac*100:.2f}% of frame")

    rows = []
    for label, vec in probes:
        v0 = make_constant_v0(scene, vec)
        vid = simulate_and_render(scene, E, v0, sim, cam).detach()  # [T,C,H,W]
        pred = vid[1:nf + 1]; tgt = gt[1:nf + 1]                # [nf,C,H,W]
        full = float(((pred - tgt) ** 2).mean())
        sqerr = ((pred - tgt) ** 2).mean(1)                    # [nf,H,W]
        masked = float((sqerr * mask).sum() / (mask.sum() + 1e-8))
        rows.append((label, vec, full, masked))
        print(f"  {label:18s} v0=({vec[0]:+.2f},{vec[1]:+.2f},{vec[2]:+.2f})  "
              f"full_mse={full:.4e}  masked_mse={masked:.4e}")

    gt_full = [r[2] for r in rows if r[0] == "GT"][0]
    rest_full = [r[2] for r in rows if r[0] == "rest_v0=0"][0]
    print(f"\nfull-frame dynamic range: rest={rest_full:.3e}  GT={gt_full:.3e}  "
          f"ratio rest/GT = {rest_full/ (gt_full+1e-12):.2f}x")
    rd.write_json("landscape.json", {
        "moving_frac": frac, "n_frames": nf, "E": E, "gt_v0": gt_v0,
        "probes": [{"label": r[0], "v0": list(r[1]), "full_mse": r[2],
                    "masked_mse": r[3]} for r in rows]})
    rd.finish()
    print(f"-> {rd.root}")
    return rd.root


if __name__ == "__main__":
    run(tyro.cli(V0LossLandscapeConfig))
