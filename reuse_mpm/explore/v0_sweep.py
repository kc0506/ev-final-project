"""Entrypoint: visual exploration of how the INITIAL STATE (v0) changes the video.

The v1 dataset only varies E; the initial velocity v0 is a single fixed constant
(0,-1,0). This is the first step of widening the conditioning axis: hold E fixed
and sweep a handful of v0 vectors (direction + magnitude), render each, and tile
them into one panel so the effect of "changing the initial state" is visible at a
glance -- before committing GPU to a full (v0, E) dataset.

Cheap by design: one fixed E, a curated ~9-vector v0 set, no_grad forward only.

  python -m reuse_mpm.explore.v0_sweep --scene.preset telephone --E 1e5
      # out dir auto-created under outputs/explore/v0_sweep/; --no-gs_ply for a light run

The config (V0SweepConfig) is LOCAL to this explore entrypoint -- explore/ are
one-shot diagnostics and must not touch the single-source `config.py` (it only
*reads* SceneSpec/SimConfig to compose them). CLI is tyro, like the canonical
entrypoints; `out`/`run_label` follow the RunDir output-tree convention.

Output dir:
  config.json            resolved V0SweepConfig + provenance
  camera.json            render-view R/T/FoV/centre/world_view (align plys to the view)
  source_ply             symlink to the input full-scene 3DGS ply (static background)
  background_~sim_mask.png  render of the non-simulated background (incl. any "wall"); once
  manifest.json          per-v0 (vector, magnitude, paths, stability)
  v_XXX/                 per-v0 video.mp4/gif + frames/
    mpm_ply/             per-frame MPM particle ply (moving=green, anchor=red); always
    gs_ply/              per-frame full displaced 3DGS splat ply        (if gs_ply, all v0)
    gs_sim_ply/          per-frame sim_mask (foreground) gaussians only  (if gs_ply, all v0)
    gs_moving_ply/       per-frame KNN-driven moving gaussians only      (if gs_ply, all v0)
  panel.gif/.mp4         all v0 tiled side by side, labelled by vector
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import tyro

from ..config import SceneSpec, SimConfig
from ..gpu import pick_free_gpu


@dataclass
class V0SweepConfig:
    """explore.v0_sweep config (local; does not live in the single-source config.py)."""

    scene: SceneSpec
    E: float = 1e5  # fixed Young's modulus
    sim: SimConfig = field(default_factory=lambda: SimConfig(num_frames=14, substep=64))
    frame: str = "frame_00001.png"
    jump_thresh: float = 0.5  # per-frame max normalised single-step jump -> unstable flag
    # emit per-frame 3DGS splat plys (gs_ply full + gs_sim_ply + gs_moving_ply) for
    # EVERY v0. Heavy (~all-gaussians/frame, tens of GB for 9x14); set False for a
    # quick light run (videos + mpm_ply only).
    gs_ply: bool = True
    out: Optional[str] = None
    run_label: str = ""


def default_v0_set() -> List[Tuple[str, Tuple[float, float, float]]]:
    """Curated v0 vectors: 6 unit directions + a down-magnitude sweep + a diagonal.

    Returns a list of (label, (x,y,z)); kept to 9 for a clean 3x3 panel.
    """
    d = 1.0 / (2.0 ** 0.5)
    return [
        ("down 1.0  (baseline)", (0.0, -1.0, 0.0)),
        ("down 0.5", (0.0, -0.5, 0.0)),
        ("down 2.0", (0.0, -2.0, 0.0)),
        ("up 1.0", (0.0, 1.0, 0.0)),
        ("+x 1.0", (1.0, 0.0, 0.0)),
        ("-x 1.0", (-1.0, 0.0, 0.0)),
        ("+z 1.0", (0.0, 0.0, 1.0)),
        ("-z 1.0", (0.0, 0.0, -1.0)),
        ("diag +x/-y", (d, -d, 0.0)),
    ]


def run(cfg: V0SweepConfig) -> str:
    pick_free_gpu()
    import imageio
    import torch
    from ..sim_render import (make_constant_v0, simulate_positions, render_positions,
                              video_to_uint8, render_static_subset)
    from ..ply_io import camera_to_dict, mpm_particles_ply, gaussian_frame_ply
    from ..scene_io import load_from_spec
    from ..run_io import RunDir, save_panel_video

    t0 = time.time()
    label = cfg.run_label or f"{cfg.scene.display_name}_E{cfg.E:g}"
    rd = RunDir.create(__name__, label, cfg.out)

    scene = load_from_spec(cfg.scene, cfg.sim)  # resolves cfg.scene.cache_path
    try:
        cam = scene.camera_by_frame(cfg.frame)
    except Exception:
        cam = scene.test_camera_list[0]

    rd.write_config({"task": "explore.v0_sweep",
                     "scene": cfg.scene.to_dict(), "sim": cfg.sim.to_dict(),
                     "E": cfg.E, "frame": cfg.frame, "jump_thresh": cfg.jump_thresh,
                     "gs_ply": cfg.gs_ply, "scene_name": scene.name,
                     "n_mpm_particles": int(scene.sim_xyzs.shape[0])})

    # output-dir completeness: input ply symlink + camera transforms json, so the
    # per-frame plys can be aligned to the exact render view in a viewer.
    if cfg.scene.kind == "pd":
        src_ply = os.path.join(cfg.scene.path, "point_cloud.ply")
        dst = rd.path("source_ply")
        if os.path.exists(src_ply):
            if os.path.islink(dst) or os.path.exists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src_ply), dst)
    rd.write_json("camera.json", camera_to_dict(cam))

    # ~sim_mask = static background (incl. any "wall") MPM never simulates; v0- and
    # frame-independent, so render ONCE. If the wall shows here but not in the
    # sim_mask foreground, it is composited-only -> the object passes through it
    # (no MPM collision); freeze/anchor particles, by contrast, are IN the body.
    imageio.imwrite(rd.path("background_~sim_mask.png"),
                    render_static_subset(scene, cam, ~scene.sim_mask))

    # gaussian-level "moving" mask (frame-independent): a sim gaussian whose KNN
    # (top_k_index) includes >=1 query/non-frozen MPM particle. Bridges the MPM
    # moving-mask onto 3DGS (different shapes) via the existing KNN, so the moving
    # subset can be a real 3DGS splat -- not a plain point cloud.
    moving_gauss = torch.zeros(scene.sim_mask.shape[0], dtype=torch.bool,
                               device=scene.device)              # [N_gauss]
    moving_gauss[scene.sim_mask] = scene.query_mask[scene.top_k_index].any(dim=1)

    v0_set = default_v0_set()
    clips, labels, samples = [], [], []
    n_unstable = 0
    for i, (vlabel, vec) in enumerate(v0_set):
        v0 = make_constant_v0(scene, vec)                       # [n_particles, 3]
        pos_list = simulate_positions(scene, float(cfg.E), v0, cfg.sim)  # T x [n, 3] world

        # stability over MOVING particles only: frozen/anchor particles carry a
        # fixed v0-independent settling transient that otherwise floors the metric.
        qm = scene.query_mask                                   # [n] bool
        norm = [(p + scene.shift) / scene.scale for p in pos_list]  # T x [n, 3] normalised
        jumps = [float((norm[t] - norm[t - 1])[qm].norm(dim=-1).max())
                 for t in range(1, len(norm))]
        max_jump = max(jumps) if jumps else 0.0
        stable = bool(max_jump < cfg.jump_thresh)
        n_unstable += (not stable)

        vid_u8 = video_to_uint8(render_positions(scene, pos_list, cam))  # [T,H,W,C] uint8
        sub = RunDir(rd.path(f"v_{i:03d}"))
        sub.save_video(vid_u8, fps=cfg.sim.fps)

        # per-frame geometry: MPM particle ply (always; light). Checks if particles
        # are physically present vs the GS rendering dropping them.
        init_pos = pos_list[0]                                  # [n, 3] world (rest)
        # per-frame MPM particle ply (always; light): geometry/anchor reference.
        for t, p in enumerate(pos_list):
            mpm_particles_ply(scene, p, sub.path("mpm_ply", f"frame_{t:03d}.ply"))
        # per-frame 3DGS splat plys for EVERY v0 (heavy). gs_ply = whole scene;
        # gs_sim_ply = sim_mask foreground; gs_moving_ply = KNN-driven moving subset.
        if cfg.gs_ply:
            for t, p in enumerate(pos_list):
                gaussian_frame_ply(scene, p, init_pos, sub.path("gs_ply", f"frame_{t:03d}.ply"))
                gaussian_frame_ply(scene, p, init_pos, sub.path("gs_sim_ply", f"frame_{t:03d}.ply"),
                                   keep_mask=scene.sim_mask)
                gaussian_frame_ply(scene, p, init_pos, sub.path("gs_moving_ply", f"frame_{t:03d}.ply"),
                                   keep_mask=moving_gauss)
            print(f"        wrote gs_ply + gs_sim_ply + gs_moving_ply for v_{i:03d}")
        sub.finish()  # seal this v_'s video + per-frame ply dirs into its .events.txt

        clips.append(vid_u8)
        flag = "" if stable else "  [UNSTABLE]"
        labels.append(f"{vlabel}{flag}")
        samples.append({"id": i, "label": vlabel, "v0": list(vec),
                        "magnitude": float(np.linalg.norm(vec)),
                        "max_frame_jump": max_jump, "stable": stable,
                        "dir": os.path.relpath(sub.root, rd.root)})
        print(f"  [{i+1}/{len(v0_set)}] v0={vec} |v0|={np.linalg.norm(vec):.2f} "
              f"max_jump={max_jump:.3f}{flag}")

    panel = save_panel_video(
        rd.path("panel.gif"), clips, labels, fps=cfg.sim.fps, ncols=3,
        title=f"{scene.name}  E={cfg.E:.1e}  v0 sweep")
    try:
        save_panel_video(rd.path("panel.mp4"), clips, labels, fps=cfg.sim.fps,
                         ncols=3, title=f"{scene.name}  E={cfg.E:.1e}  v0 sweep")
    except Exception as e:
        print(f"[v0_sweep] mp4 panel skipped: {e}")

    rd.write_json("manifest.json", {
        "task": "explore.v0_sweep", "scene": scene.name, "E": float(cfg.E),
        "num_frames": cfg.sim.num_frames, "substep": cfg.sim.substep,
        "n": len(v0_set), "n_unstable": n_unstable,
        "jump_thresh": cfg.jump_thresh, "samples": samples,
        "panel": os.path.relpath(panel, rd.root),
        "elapsed_sec": round(time.time() - t0, 2),
    })
    rd.finish()  # seals source_ply, panel.gif/mp4, and the v_XXX/ result dirs
    print(f"[v0_sweep] {len(v0_set)} v0 ({n_unstable} unstable) -> {rd.root}  "
          f"({time.time()-t0:.1f}s)\n  panel: {panel}")
    return rd.root


if __name__ == "__main__":
    run(tyro.cli(V0SweepConfig))
