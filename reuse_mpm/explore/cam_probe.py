"""Entrypoint: render the STATIC scene from 4 look-at cameras placed along world
axes, to SEE (not infer) the world<->screen mapping.

Cameras sit at center + radius * dir for dir in {+y,+x,-y,-x}, optical axis aimed
at the scene center, up = world +z. No MPM, no motion -- just look at the object
from each world direction and read off the axes directly from the images.

  python -m reuse_mpm.explore.cam_probe --scene.path .../telephone --scene.kind pd
      # out dir auto-created under outputs/explore/cam_probe/

Config (CamProbeConfig) is LOCAL to this explore entrypoint (explore/ must not
touch the single-source config.py; it only reads SceneSpec/SimConfig). CLI = tyro.

Output: img_+y.png img_+x.png img_-y.png img_-x.png, panel.png, cameras.json.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from ..config import SceneSpec, SimConfig
from ..gpu import pick_free_gpu
from ..sampling import lookat_R_T  # moved here (canonical) so dataset_gen can share it


@dataclass
class CamProbeConfig:
    """explore.cam_probe config (local; not in the single-source config.py)."""

    scene: SceneSpec
    sim: SimConfig = field(default_factory=SimConfig)
    radius_k: float = 3.0  # camera distance = center + radius_k * object_bound * dir
    out: Optional[str] = None
    run_label: str = ""


def run(cfg: CamProbeConfig) -> str:
    pick_free_gpu()
    import imageio
    from ..scene_io import load_from_spec
    from ..sim_render import render_positions, video_to_uint8
    from ..ply_io import camera_to_dict
    from .._env import Camera
    from ..run_io import RunDir

    t0 = time.time()
    label = cfg.run_label or cfg.scene.display_name
    rd = RunDir.create(__name__, label, cfg.out)
    scene = load_from_spec(cfg.scene, cfg.sim)

    # reference framing from an existing capture camera (FoV + image size)
    ref = scene.test_camera_list[0]
    H, W = int(ref.image_height), int(ref.image_width)
    FoVx, FoVy = float(ref.FoVx), float(ref.FoVy)

    # static rest-pose world positions of the simulated object (target = its mean)
    pos0 = (scene.sim_xyzs * scene.scale - scene.shift).detach()   # [n, 3] world
    center = pos0.mean(0).cpu().numpy()                            # [3]
    bound = float((pos0 - pos0.mean(0)).norm(dim=-1).max().cpu())
    dist = max(bound * cfg.radius_k, 1e-3)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)               # world +z = shot "up"

    dirs = [("+y", [0, 1, 0]), ("+x", [1, 0, 0]),
            ("-y", [0, -1, 0]), ("-x", [-1, 0, 0])]
    print(f"[cam_probe] center={np.round(center,3)} bound={bound:.3f} dist={dist:.3f} "
          f"img={H}x{W} FoV=({FoVx:.3f},{FoVy:.3f})")

    imgs, labels, cams_json = [], [], {}
    for name, d in dirs:
        d = np.array(d, dtype=np.float64)
        eye = center + dist * (d / np.linalg.norm(d))              # [3]
        R, T = lookat_R_T(eye, center.astype(np.float64), up)
        cam = Camera(R=R, T=T, FoVx=FoVx, FoVy=FoVy,
                     img_path=f"cam_{name}.png", img_hw=(H, W), data_device=scene.device)
        u8 = video_to_uint8(render_positions(scene, [pos0], cam))[0]  # [H,W,C] uint8 (rest)
        imageio.imwrite(rd.path(f"img_{name}.png"), u8)
        imgs.append(u8)
        labels.append(f"cam at center+{name}  (looking at center, up=+z)")
        cams_json[name] = {"eye_world": eye.tolist(), **camera_to_dict(cam)}
        print(f"  cam {name:3s} eye={np.round(eye,3)} -> img_{name}.png")

    # 2x2 still panel (static viewpoints -> panel image, not a video)
    try:
        from PIL import Image, ImageDraw
        tile_w = 512
        tiles = []
        for u8, lab in zip(imgs, labels):
            im = Image.fromarray(u8).resize((tile_w, round(H * tile_w / W)))
            ImageDraw.Draw(im).text((6, 4), lab, fill=(230, 30, 30))
            tiles.append(np.asarray(im))
        panel = np.concatenate([np.concatenate(tiles[:2], axis=1),
                                np.concatenate(tiles[2:], axis=1)], axis=0)
        imageio.imwrite(rd.path("panel.png"), panel)
    except Exception as e:
        print(f"[cam_probe] panel skipped: {e}")

    rd.write_json("cameras.json", {"center": center.tolist(), "bound": bound,
                                   "dist": dist, "up_world": up.tolist(),
                                   "cameras": cams_json})
    print(f"[cam_probe] done -> {rd.root}  ({time.time()-t0:.1f}s)\n"
          f"  panel: {rd.path('panel.png')}")
    return rd.root


if __name__ == "__main__":
    import tyro
    run(tyro.cli(CamProbeConfig))
