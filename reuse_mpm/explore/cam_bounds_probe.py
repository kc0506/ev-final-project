"""Entrypoint: STATIC camera-bounds probe -- one photo per degree-of-freedom bound.

Before animating anything, establish HOW FAR each camera DOF can go before 3DGS
breaks. The camera DOFs are defined in the REF camera's own screen frame (its
right/up axes), so they read as the intuitive left/right/up/down/near/far -- not a
single "angle from ref" cone (which mixes directions and biased everything one
way). Each row is one DOF swept ref->max in n_steps; each cell is a STILL render
of the (rest-pose) object from that camera, labelled with its exact parameter.
You read off, per DOF, where it starts to break -> the safe bounds.

  python -m reuse_mpm.explore.cam_bounds_probe --scene.preset telephone
  python -m reuse_mpm.explore.cam_bounds_probe --scene.preset telephone \
      --azim_max 90 --elev_max 60 --dolly_frac 0.5 --n_steps 5

DOFs (rows): azimuth +/- (orbit left/right), elevation +/- (orbit up/down),
dolly in/out (pure radius change, NO rotation -- "forward/back"). Config LOCAL.

Output (auto under outputs/explore/cam_bounds_probe/):
  bounds.png   the DOF x magnitude grid of stills, each cell labelled
  bounds.json  per-cell camera eye + param + angular/radius offset from ref
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import tyro

from ..config import SceneSpec, SimConfig
from ..gpu import pick_free_gpu
from ..sampling import lookat_R_T


@dataclass
class CamBoundsProbeConfig:
    """explore.cam_bounds_probe config (local; not in config.py)."""

    scene: SceneSpec
    sim: SimConfig = field(default_factory=lambda: SimConfig(num_frames=1, substep=1))
    azim_max: float = 90.0    # orbit left/right max (deg) to test
    elev_max: float = 60.0    # orbit up/down max (deg) to test
    dolly_frac: float = 0.5   # radius in/out max as fraction of ref radius
    n_steps: int = 5          # cols: magnitudes 0..max (col 0 = ref)
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0)  # lookat image-up
    frame: str = "frame_00001.png"
    out: Optional[str] = None
    run_label: str = ""


def _rodrigues(v: np.ndarray, axis: np.ndarray, theta: float) -> np.ndarray:
    k = axis / (np.linalg.norm(axis) or 1.0)
    return (v * np.cos(theta) + np.cross(k, v) * np.sin(theta)
            + k * np.dot(k, v) * (1.0 - np.cos(theta)))


def run(cfg: CamBoundsProbeConfig) -> str:
    pick_free_gpu()
    import torch
    from PIL import Image, ImageDraw
    from ..sim_render import render_positions, video_to_uint8
    from ..scene_io import load_from_spec
    from .._env import Camera
    from ..run_io import RunDir

    t0 = time.time()
    rd = RunDir.create(__name__, cfg.run_label or cfg.scene.display_name, cfg.out)
    rd.write_config({"task": "explore.cam_bounds_probe", "scene": cfg.scene.to_dict(),
                     "azim_max": cfg.azim_max, "elev_max": cfg.elev_max,
                     "dolly_frac": cfg.dolly_frac, "n_steps": cfg.n_steps,
                     "up": list(cfg.up), "frame": cfg.frame})

    scene = load_from_spec(cfg.scene, cfg.sim)
    ref_cam = scene.camera_by_frame(cfg.frame)
    pos0 = (scene.sim_xyzs * scene.scale - scene.shift).detach()      # [n,3] world rest
    center = pos0.mean(0).cpu().numpy()
    try:
        ref_eye = ref_cam.camera_center.detach().cpu().numpy()
    except Exception:
        ref_eye = (-np.asarray(ref_cam.R) @ np.asarray(ref_cam.T))
    ref_r = float(np.linalg.norm(ref_eye - center)) or 1.0
    d0 = (ref_eye - center) / ref_r                                  # center->eye dir
    FoVx, FoVy = float(ref_cam.FoVx), float(ref_cam.FoVy)
    H, W = int(ref_cam.image_height), int(ref_cam.image_width)
    up = np.asarray(cfg.up, dtype=np.float64)

    # orbit axes = the REF camera's own world right/up (so the DOFs read as the
    # screen's left-right / up-down, not a world axis that the view is aligned with)
    Rmat = np.asarray(ref_cam.R, dtype=np.float64)                   # cols [right,down,fwd]
    right_cam = Rmat[:, 0] / (np.linalg.norm(Rmat[:, 0]) or 1.0)
    up_cam = -Rmat[:, 1] / (np.linalg.norm(Rmat[:, 1]) or 1.0)

    # training-camera envelope (the safe region) for reference
    angs, radii = [], []
    for c in getattr(scene, "test_camera_list", []) or []:
        try:
            e = c.camera_center.detach().cpu().numpy()
        except Exception:
            continue
        v = e - center; r = np.linalg.norm(v) or 1.0
        radii.append(r); angs.append(np.degrees(np.arccos(np.clip(np.dot(v / r, d0), -1, 1))))
    r_lo, r_hi = (min(radii), max(radii)) if radii else (ref_r, ref_r)
    env = (f"ang<= {max(angs):.0f}deg, radius {r_lo:.2f}..{r_hi:.2f} (n={len(angs)})"
           if angs else "unknown")
    print(f"[cam_bounds] center={np.round(center,3)} ref_r={ref_r:.3f} img={H}x{W}\n"
          f"  training envelope: {env}")

    def render_eye(eye: np.ndarray):
        R, Tcw = lookat_R_T(eye, center, up)
        cam = Camera(R=R, T=Tcw, FoVx=FoVx, FoVy=FoVy, img_path=cfg.frame,
                     img_hw=(H, W), data_device=scene.device)
        return video_to_uint8(render_positions(scene, [pos0], cam))[0]   # [H,W,3]

    def ang_from_ref(eye):
        v = (eye - center); v = v / (np.linalg.norm(v) or 1.0)
        return float(np.degrees(np.arccos(np.clip(np.dot(v, d0), -1, 1))))

    steps = np.linspace(0.0, 1.0, cfg.n_steps)                       # 0..1 magnitude

    def rot_eye(axis, deg):
        return center + ref_r * _rodrigues(d0, axis, np.deg2rad(deg))

    def mk(name, eye_fn, lab_fn):                                    # one DOF row
        return name, [eye_fn(s) for s in steps], [lab_fn(s) for s in steps]

    rows = [
        mk("azim +", lambda s: rot_eye(up_cam, +s * cfg.azim_max), lambda s: f"az +{s*cfg.azim_max:.0f}°"),
        mk("azim -", lambda s: rot_eye(up_cam, -s * cfg.azim_max), lambda s: f"az -{s*cfg.azim_max:.0f}°"),
        mk("elev +", lambda s: rot_eye(right_cam, +s * cfg.elev_max), lambda s: f"el +{s*cfg.elev_max:.0f}°"),
        mk("elev -", lambda s: rot_eye(right_cam, -s * cfg.elev_max), lambda s: f"el -{s*cfg.elev_max:.0f}°"),
        mk("dolly in", lambda s: center + ref_r * (1 - s * cfg.dolly_frac) * d0,
           lambda s: f"r {ref_r*(1-s*cfg.dolly_frac):.2f} ({-s*cfg.dolly_frac*100:.0f}%)"),
        mk("dolly out", lambda s: center + ref_r * (1 + s * cfg.dolly_frac) * d0,
           lambda s: f"r {ref_r*(1+s*cfg.dolly_frac):.2f} (+{s*cfg.dolly_frac*100:.0f}%)"),
    ]

    # render grid + record
    tile_w = 320
    tile_h = max(1, round(H * tile_w / W))
    nrows, ncols = len(rows), cfg.n_steps
    pad_l = 90                                                       # left gutter for row name
    canvas = Image.new("RGB", (pad_l + ncols * tile_w, nrows * tile_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    grid_json = {}
    for ri, (name, eyes, labs) in enumerate(rows):
        draw.text((4, ri * tile_h + tile_h // 2), name, fill=(0, 0, 0))
        cells = []
        for ci, (eye, lab) in enumerate(zip(eyes, labs)):
            img = render_eye(np.asarray(eye, dtype=np.float64))
            im = Image.fromarray(img).resize((tile_w, tile_h))
            d = ImageDraw.Draw(im)
            a = ang_from_ref(eye)
            txt = lab if ci > 0 else "ref"
            d.text((4, 2), txt, fill=(220, 30, 30))
            d.text((4, tile_h - 12), f"Δ{a:.0f}° r{np.linalg.norm(eye-center):.2f}",
                   fill=(20, 90, 220))
            canvas.paste(im, (pad_l + ci * tile_w, ri * tile_h))
            cells.append({"label": txt, "eye": list(map(float, eye)),
                          "ang_from_ref": a, "radius": float(np.linalg.norm(eye - center))})
        grid_json[name] = cells
        print(f"  row '{name}': {labs[-1]}  (max Δ{ang_from_ref(eyes[-1]):.0f}°)")

    canvas.save(rd.path("bounds.png"))
    rd._event("bounds.png", "bounds.png")
    rd.write_json("bounds.json", {"center": center.tolist(), "ref_radius": ref_r,
                                  "ref_dir": d0.tolist(), "training_envelope": env,
                                  "azim_max": cfg.azim_max, "elev_max": cfg.elev_max,
                                  "dolly_frac": cfg.dolly_frac, "grid": grid_json})
    rd.finish()
    print(f"[cam_bounds] grid {nrows}x{ncols} -> {rd.root}  ({time.time()-t0:.1f}s)\n"
          f"  image: {rd.path('bounds.png')}")
    return rd.root


if __name__ == "__main__":
    run(tyro.cli(CamBoundsProbeConfig))
