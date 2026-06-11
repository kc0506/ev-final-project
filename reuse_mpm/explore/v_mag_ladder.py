"""Entrypoint: a TRUE velocity magnitude ladder -- fixed direction, |v0| swept
small->large, across several E -- to judge how readable "slow vs fast" is and
which stiffness regime shows it best.

Unlike v0_sweep (a curated set of mostly DIFFERENT DIRECTIONS, magnitude barely
varied), here the direction is FIXED and only |v0| changes, tiled as a grid:
  rows  = E      (soft -> stiff, top -> bottom)
  cols  = |v0|   (slow -> fast,  left -> right)
so left->right shows speed, top->bottom shows the E effect, at a glance.

  python -m reuse_mpm.explore.v_mag_ladder --scene.preset telephone
  python -m reuse_mpm.explore.v_mag_ladder --scene.preset telephone \
      --direction 1 0 0 --mags 0.1 0.5 1 2 3 --E_list 3e4 1e5 --sim.num-frames 32

Config is LOCAL (explore convention; does not touch config.py).

Output dir (auto under outputs/explore/v_mag_ladder/):
  config.json   resolved config + provenance
  ladder.gif    the rows=E x cols=|v0| grid (the artifact to look at)
  manifest.json per-(E,|v0|) magnitude / stability
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import tyro

from ..config import SceneSpec, SimConfig
from ..gpu import pick_free_gpu


@dataclass
class VMagLadderConfig:
    """explore.v_mag_ladder config (local; not in config.py)."""

    scene: SceneSpec
    direction: Tuple[float, float, float] = (0.0, -1.0, 0.0)  # fixed unit-ish dir
    mags: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
    E_list: List[float] = field(default_factory=lambda: [3e4, 1e5, 3e5])
    sim: SimConfig = field(default_factory=lambda: SimConfig(num_frames=32, substep=64))
    frame: str = "frame_00001.png"
    jump_thresh: float = 0.5
    out: Optional[str] = None
    run_label: str = ""


def run(cfg: VMagLadderConfig) -> str:
    pick_free_gpu()
    import numpy as np
    import torch
    from ..sim_render import make_constant_v0, simulate_positions, render_positions, video_to_uint8
    from ..scene_io import load_from_spec
    from ..run_io import RunDir, save_panel_video

    t0 = time.time()
    rd = RunDir.create(__name__, cfg.run_label or cfg.scene.display_name, cfg.out)
    rd.write_config({"task": "explore.v_mag_ladder", "scene": cfg.scene.to_dict(),
                     "sim": cfg.sim.to_dict(), "direction": list(cfg.direction),
                     "mags": cfg.mags, "E_list": cfg.E_list,
                     "frame": cfg.frame, "jump_thresh": cfg.jump_thresh})

    scene = load_from_spec(cfg.scene, cfg.sim)
    try:
        cam = scene.camera_by_frame(cfg.frame)
    except Exception:
        cam = scene.test_camera_list[0]

    d = np.asarray(cfg.direction, dtype=float)
    d = d / (np.linalg.norm(d) or 1.0)  # unit direction

    clips, labels, samples = [], [], []
    n_unstable = 0
    for E in cfg.E_list:                                   # rows
        for mag in cfg.mags:                               # cols
            vec = tuple((d * mag).tolist())
            v0 = make_constant_v0(scene, vec)
            pos_list = simulate_positions(scene, float(E), v0, cfg.sim)
            qm = scene.query_mask
            norm = [(p + scene.shift) / scene.scale for p in pos_list]
            jumps = [float((norm[t] - norm[t - 1])[qm].norm(dim=-1).max())
                     for t in range(1, len(norm))]
            max_jump = max(jumps) if jumps else 0.0
            stable = bool(max_jump < cfg.jump_thresh)
            n_unstable += (not stable)
            clips.append(video_to_uint8(render_positions(scene, pos_list, cam)))
            flag = "" if stable else " !"
            labels.append(f"E{E:.0e} |v|{mag:g}{flag}")
            samples.append({"E": float(E), "mag": float(mag), "v0": list(vec),
                            "max_frame_jump": max_jump, "stable": stable})
            print(f"  E={E:.1e} |v0|={mag:<4g} max_jump={max_jump:.3f}{flag}")

    ladder = save_panel_video(
        rd.path("ladder.gif"), clips, labels, fps=cfg.sim.fps, ncols=len(cfg.mags),
        title=f"{scene.name}  dir={tuple(round(x,2) for x in d)}  rows=E cols=|v0|")
    rd.write_json("manifest.json", {
        "task": "explore.v_mag_ladder", "scene": scene.name,
        "direction": list(d), "mags": cfg.mags, "E_list": cfg.E_list,
        "num_frames": cfg.sim.num_frames, "n_unstable": n_unstable,
        "samples": samples, "ladder": os.path.relpath(ladder, rd.root),
        "elapsed_sec": round(time.time() - t0, 2)})
    rd.finish()
    print(f"[v_mag_ladder] {len(clips)} clips ({n_unstable} unstable) -> {rd.root}\n  {ladder}")
    return rd.root


if __name__ == "__main__":
    run(tyro.cli(VMagLadderConfig))
