"""Entrypoint: probe RICH dynamic camera trajectories vs 3DGS validity.

The super-rich dataset wants genuinely varied camera motion -- translation (dolly),
random start, non-uniform speed, free-form path -- NOT just a fixed-radius turntable
("small aug"). The only hard limit is the 3DGS-reliable region (near training
views). This samples N random SMOOTH trajectories (Catmull-Rom through random
waypoints), each bounded inside that region (eye direction within `cap_deg` of the
ref view, radius within the training cameras' own range), renders the object
(rest by default, so only the CAMERA moves), tiles them, and prints the training
cameras' angular + radius envelope -- so you can SEE the richness and where it
breaks, then we tune `cap_deg` / radius range / speed before committing.

  python -m reuse_mpm.explore.cam_dynamic_probe --scene.preset telephone
  python -m reuse_mpm.explore.cam_dynamic_probe --scene.preset telephone \
      --cap_deg 60 --n_traj 6 --n_waypoints 4 --radius_jitter 0.3 --v0 1 0 0

Config is LOCAL (explore convention; not in config.py). The trajectory math stays
here until a safe envelope is agreed, then gets promoted to CameraDist.

Output (auto under outputs/explore/cam_dynamic_probe/):
  panel.gif   the N trajectories tiled (the artifact to look at)
  eyes.json   per-trajectory per-frame eye/radius + envelope stats
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
class CamDynamicProbeConfig:
    """explore.cam_dynamic_probe config (local; not in config.py)."""

    scene: SceneSpec
    sim: SimConfig = field(default_factory=lambda: SimConfig(num_frames=24, substep=32))
    n_traj: int = 6           # how many distinct random trajectories to show
    cap_deg: float = 60.0     # eye direction stays within this angle of the ref view
    n_waypoints: int = 4      # Catmull-Rom control points per trajectory (more = wigglier)
    radius_jitter: float = 0.3  # dolly: radius ~ ref_radius * (1 +/- this); 0 = no dolly
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    v0: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # object motion; rest -> only camera moves
    E: float = 1e5
    seed: int = 0
    frame: str = "frame_00001.png"
    out: Optional[str] = None
    run_label: str = ""


def _dir_in_cap(rng, u0: np.ndarray, cap_rad: float) -> np.ndarray:
    """Unit direction sampled (area-weighted) within `cap_rad` of u0."""
    ct = rng.uniform(np.cos(cap_rad), 1.0)
    st = float(np.sqrt(max(0.0, 1.0 - ct * ct)))
    phi = rng.uniform(0.0, 2.0 * np.pi)
    a = np.array([0.0, 0.0, 1.0]) if abs(u0[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(u0, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(u0, e1)
    return ct * u0 + st * (np.cos(phi) * e1 + np.sin(phi) * e2)


def _catmull_rom(wp: np.ndarray, T: int) -> np.ndarray:
    """Smooth path through waypoints wp [K,3] sampled at T points (end-padded)."""
    P = np.vstack([wp[0], wp, wp[-1]])                # pad ends [K+2,3]
    segs = len(wp) - 1
    out = []
    for i in range(T):
        u = i / max(1, T - 1) * segs                 # global param in [0,segs]
        s = min(int(u), segs - 1)
        t = u - s
        p0, p1, p2, p3 = P[s], P[s + 1], P[s + 2], P[s + 3]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t))
    return np.asarray(out)                            # [T,3]


def run(cfg: CamDynamicProbeConfig) -> str:
    pick_free_gpu()
    from ..sim_render import (make_constant_v0, simulate_positions,
                              render_positions_multicam, video_to_uint8)
    from ..scene_io import load_from_spec
    from .._env import Camera
    from ..run_io import RunDir, save_panel_video

    t0 = time.time()
    rd = RunDir.create(__name__, cfg.run_label or cfg.scene.display_name, cfg.out)
    rd.write_config({"task": "explore.cam_dynamic_probe", "scene": cfg.scene.to_dict(),
                     "sim": cfg.sim.to_dict(), "n_traj": cfg.n_traj, "cap_deg": cfg.cap_deg,
                     "n_waypoints": cfg.n_waypoints, "radius_jitter": cfg.radius_jitter,
                     "up": list(cfg.up), "v0": list(cfg.v0), "E": cfg.E, "seed": cfg.seed})

    scene = load_from_spec(cfg.scene, cfg.sim)
    ref_cam = scene.camera_by_frame(cfg.frame)
    pos0 = (scene.sim_xyzs * scene.scale - scene.shift).detach()      # [n,3] world rest
    center = pos0.mean(0).cpu().numpy()                              # [3] look-at target
    try:
        ref_eye = ref_cam.camera_center.detach().cpu().numpy()
    except Exception:
        ref_eye = (-np.asarray(ref_cam.R) @ np.asarray(ref_cam.T))
    ref_r = float(np.linalg.norm(ref_eye - center)) or 1.0
    u0 = (ref_eye - center) / ref_r                                  # [3] ref view dir
    FoVx, FoVy = float(ref_cam.FoVx), float(ref_cam.FoVy)
    H, W = int(ref_cam.image_height), int(ref_cam.image_width)
    up = np.asarray(cfg.up, dtype=np.float64)

    # training-camera envelope: angular spread from ref + radius range, from the
    # object centre. The "safe region" the trajectories stay inside.
    angs, radii = [], []
    for c in getattr(scene, "test_camera_list", []) or []:
        try:
            e = c.camera_center.detach().cpu().numpy()
        except Exception:
            continue
        v = e - center; r = np.linalg.norm(v) or 1.0
        radii.append(r)
        angs.append(np.degrees(np.arccos(np.clip(np.dot(v / r, u0), -1, 1))))
    r_lo, r_hi = (min(radii), max(radii)) if radii else (ref_r, ref_r)
    env = (f"ang {min(angs):.0f}..{max(angs):.0f}deg, radius {r_lo:.2f}..{r_hi:.2f} "
           f"(n={len(angs)})" if angs else "unknown")
    print(f"[cam_dynamic] center={np.round(center,3)} ref_radius={ref_r:.3f} img={H}x{W}\n"
          f"  training-cam envelope from ref: {env}\n"
          f"  trajectories: cap={cfg.cap_deg}deg, radius ref*(1+/-{cfg.radius_jitter}), "
          f"{cfg.n_waypoints} waypoints, n={cfg.n_traj}")

    T = cfg.sim.num_frames
    if any(cfg.v0):
        pos_list = simulate_positions(scene, float(cfg.E), make_constant_v0(scene, cfg.v0), cfg.sim)
    else:
        pos_list = [pos0.clone() for _ in range(T)]                  # pure camera motion

    rng = np.random.RandomState(cfg.seed)
    cap = np.deg2rad(cfg.cap_deg)
    clips, labels, eyes_json = [], [], {}
    for j in range(cfg.n_traj):
        wps = []
        for _ in range(cfg.n_waypoints):
            d = _dir_in_cap(rng, u0, cap)
            r = ref_r * (1.0 + rng.uniform(-cfg.radius_jitter, cfg.radius_jitter))
            r = float(np.clip(r, r_lo, r_hi)) if radii else r
            wps.append(center + r * d)
        eyes = _catmull_rom(np.asarray(wps), T)                      # [T,3] smooth path
        cams = []
        for e in eyes:
            R, Tcw = lookat_R_T(e, center, up)
            cams.append(Camera(R=R, T=Tcw, FoVx=FoVx, FoVy=FoVy, img_path=cfg.frame,
                               img_hw=(H, W), data_device=scene.device))
        vid = video_to_uint8(render_positions_multicam(scene, pos_list, cams))
        clips.append(vid)
        d_ang = [np.degrees(np.arccos(np.clip(np.dot((e - center) / (np.linalg.norm(e - center) or 1), u0), -1, 1))) for e in eyes]
        rr = [float(np.linalg.norm(e - center)) for e in eyes]
        labels.append(f"#{j}  ang<={max(d_ang):.0f}  r {min(rr):.2f}-{max(rr):.2f}")
        eyes_json[f"traj_{j}"] = {"eyes": eyes.tolist(),
                                  "ang_deg": [float(a) for a in d_ang], "radius": rr}
        print(f"  traj {j}: max_ang={max(d_ang):4.0f}deg  radius {min(rr):.2f}..{max(rr):.2f}")

    panel = save_panel_video(rd.path("panel.gif"), clips, labels, fps=cfg.sim.fps,
                             title=f"{scene.name}  rich dyn-cam  cap{cfg.cap_deg:g}  (env {env})")
    rd.write_json("eyes.json", {"center": center.tolist(), "ref_radius": ref_r,
                                "ref_dir": u0.tolist(), "training_envelope": env,
                                "cap_deg": cfg.cap_deg, "radius_jitter": cfg.radius_jitter,
                                "trajectories": eyes_json})
    rd.finish()
    print(f"[cam_dynamic] {cfg.n_traj} trajectories -> {rd.root}  ({time.time()-t0:.1f}s)\n"
          f"  panel: {panel}")
    return rd.root


if __name__ == "__main__":
    run(tyro.cli(CamDynamicProbeConfig))
