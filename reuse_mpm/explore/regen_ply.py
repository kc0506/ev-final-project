"""Entrypoint: regenerate per-frame MPM / 3DGS plys for a dataset sample ON DEMAND.

dataset_gen stores only `mpm_xyz.npy` (+ video) per sample -- the per-frame plys
are NOT stored because they are a closed-form function of that trajectory:
  MPM ply  = mpm_xyz[t] coloured by the frozen masks (scene_cache.pt)
  3DGS ply = source gaussians displaced by KNN-of-displacement (init=mpm_xyz[0],
             top_k_index/attrs from scene_cache.pt + the input ply) -- rotation
             included, derived from positions, NOT the un-saved MPM F.
So this rebuilds them only for the sample/frames you actually want to eyeball in a
GS viewer, at zero standing storage cost.

  # MPM plys for all frames of one sample
  python -m reuse_mpm.explore.regen_ply --sample_dir outputs/dataset_gen/05_a/sample_0003
  # MPM + 3DGS (full/sim/moving) for frames 0,8,15
  python -m reuse_mpm.explore.regen_ply --sample_dir .../sample_0003 --gs --frames 0,8,15

Reads the PARENT run's config.json (scene/sim spec) and pins the discretisation to
that run's frozen `scene_cache.pt`, so the rebuilt geometry matches exactly what
produced the video. Config is LOCAL (explore convention; does not touch config.py).

Output dir (auto under outputs/explore/regen_ply/):
  camera.json            render-view transforms (align plys to the view)
  source_ply             symlink to the input full-scene 3DGS ply
  meta.json              which sample/frames, paths
  mpm_ply/frame_XXX.ply  per-frame MPM particles (moving=green, anchor=red)
  gs_ply/, gs_sim_ply/, gs_moving_ply/  per-frame 3DGS splats (if --gs)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

import tyro

from ..gpu import pick_free_gpu


@dataclass
class RegenPlyConfig:
    """explore.regen_ply config (local; does not live in config.py)."""

    sample_dir: str  # path to a dataset_gen sample_XXXX dir
    frames: str = "all"  # "all" | "0,8,15" | "0-15" | "0-30:5" (comma-joinable)
    gs: bool = True  # also emit 3DGS splat plys (full + sim_mask + moving subsets)
    out: Optional[str] = None
    run_label: str = ""


def _parse_frames(spec: str, T: int) -> List[int]:
    if spec.strip() == "all":
        return list(range(T))
    out: List[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            rng, _, step = tok.partition(":")
            a, _, b = rng.partition("-")
            out += list(range(int(a), int(b) + 1, int(step) if step else 1))
        else:
            out.append(int(tok))
    return [t for t in dict.fromkeys(out) if 0 <= t < T]  # dedup, in-range, ordered


def run(cfg: RegenPlyConfig) -> str:
    pick_free_gpu()
    import numpy as np
    import torch
    from ..config import SceneSpec, SimConfig
    from ..scene_io import load_from_spec
    from ..ply_io import camera_to_dict, mpm_particles_ply, gaussian_frame_ply
    from ..run_io import RunDir

    sample_dir = os.path.abspath(cfg.sample_dir)
    run_dir = os.path.dirname(sample_dir)  # the parent dataset_gen run
    rcfg = json.load(open(os.path.join(run_dir, "config.json")))
    spec = SceneSpec(**{k: v for k, v in rcfg["scene"].items() if k != "_provenance"})
    sim = SimConfig(**rcfg["sim"])
    frame_name = rcfg.get("frame", "frame_00001.png")
    # pin discretisation to this run's frozen snapshot (matches the video exactly)
    frozen = os.path.join(run_dir, "scene_cache.pt")
    if os.path.exists(frozen):
        spec.cache_path = frozen

    mpm_xyz = np.load(os.path.join(sample_dir, "mpm_xyz.npy"))  # [T,n,3] world
    T = int(mpm_xyz.shape[0])
    frames = _parse_frames(cfg.frames, T)

    label = cfg.run_label or f"{os.path.basename(run_dir)}_{os.path.basename(sample_dir)}"
    rd = RunDir.create(__name__, label, cfg.out)

    scene = load_from_spec(spec, sim)
    try:
        cam = scene.camera_by_frame(frame_name)
    except Exception:
        cam = scene.test_camera_list[0]
    rd.write_json("camera.json", camera_to_dict(cam))
    if spec.kind == "pd":
        src = os.path.join(spec.path, "point_cloud.ply")
        dst = rd.path("source_ply")
        if os.path.exists(src) and not (os.path.islink(dst) or os.path.exists(dst)):
            os.symlink(os.path.abspath(src), dst)

    pos = torch.from_numpy(mpm_xyz).float().to(scene.device)  # [T,n,3]
    init = pos[0]  # rest pose

    # gaussian-level moving mask (KNN includes >=1 query particle); same as v0_sweep
    moving_gauss = torch.zeros(scene.sim_mask.shape[0], dtype=torch.bool, device=scene.device)
    moving_gauss[scene.sim_mask] = scene.query_mask[scene.top_k_index].any(dim=1)

    for t in frames:
        mpm_particles_ply(scene, pos[t], rd.path("mpm_ply", f"frame_{t:03d}.ply"))
        if cfg.gs:
            gaussian_frame_ply(scene, pos[t], init, rd.path("gs_ply", f"frame_{t:03d}.ply"))
            gaussian_frame_ply(scene, pos[t], init, rd.path("gs_sim_ply", f"frame_{t:03d}.ply"),
                               keep_mask=scene.sim_mask)
            gaussian_frame_ply(scene, pos[t], init, rd.path("gs_moving_ply", f"frame_{t:03d}.ply"),
                               keep_mask=moving_gauss)
        print(f"  frame {t:3d}/{T}  -> mpm_ply{' + gs (full/sim/moving)' if cfg.gs else ''}")

    rd.write_json("meta.json", {
        "task": "explore.regen_ply", "sample_dir": sample_dir, "run_dir": run_dir,
        "scene": scene.name, "n_frames_total": T, "frames": frames, "gs": cfg.gs,
    })
    rd.finish()
    print(f"[regen_ply] {len(frames)} frames (gs={cfg.gs}) -> {rd.root}")
    return rd.root


if __name__ == "__main__":
    run(tyro.cli(RegenPlyConfig))
