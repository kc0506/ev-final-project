"""Entrypoint: COMPOUND dynamic-camera probe -- azim+elev+r move together, shown
as clip + parameter-space path (no composed Euler rotations).

After the per-DOF bounds are known, the real camera motion compounds them. To keep
it legible (compound rotation is otherwise a mess) the camera is parameterised by
its EYE ON A SPHERE -- (azim, elev, r) about the object centre, always looking at
it -- so a trajectory is just a smooth PATH in (azim, elev, r), not a product of
rotations. Each sampled trajectory is rendered AND drawn as a path in the
azim-elev plane (colour = time, marker size = r), so you read the camera motion
in interpretable coords next to the video.

Stays inside the telephone SAFE BOX (one-sided: +azim to wall, +elev to no-floor,
r dolly band) -- defaults have margin; widen via flags.

  python -m reuse_mpm.explore.cam_compound_probe --scene.preset telephone
  python -m reuse_mpm.explore.cam_compound_probe --scene.preset telephone \
      --azim 0 90 --elev 0 32 --radius 0.35 1.10 --n_traj 6 --v0 1 0 0

Config LOCAL (explore). Output (auto under outputs/explore/cam_compound_probe/):
  panel.gif   the N trajectories' clips tiled (index-matched to paths.png)
  paths.png   each trajectory as a path in (azim, elev) space, colour=time, size=r
  paths.json  per-trajectory per-frame (azim, elev, r, eye)
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
class CamCompoundProbeConfig:
    """explore.cam_compound_probe config (local; not in config.py)."""

    scene: SceneSpec
    sim: SimConfig = field(default_factory=lambda: SimConfig(num_frames=24, substep=32))
    n_traj: int = 6
    azim: Tuple[float, float] = (0.0, 90.0)    # safe one-sided range (deg)
    elev: Tuple[float, float] = (0.0, 32.0)
    radius: Tuple[float, float] = (0.35, 1.10)
    n_waypoints: int = 4
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    v0: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # object motion; rest -> camera only
    E: float = 1e5
    seed: int = 0
    frame: str = "frame_00001.png"
    out: Optional[str] = None
    run_label: str = ""


def _rodrigues(v, k, th):
    k = k / (np.linalg.norm(k) or 1.0)
    return v * np.cos(th) + np.cross(k, v) * np.sin(th) + k * np.dot(k, v) * (1 - np.cos(th))


def _catmull(wp: np.ndarray, T: int) -> np.ndarray:
    """Smooth path through waypoints wp [K,D] sampled at T points (end-padded)."""
    P = np.vstack([wp[0], wp, wp[-1]])
    segs = len(wp) - 1
    out = []
    for i in range(T):
        u = i / max(1, T - 1) * segs
        s = min(int(u), segs - 1); t = u - s
        p0, p1, p2, p3 = P[s], P[s + 1], P[s + 2], P[s + 3]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.asarray(out)


def run(cfg: CamCompoundProbeConfig) -> str:
    pick_free_gpu()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..sim_render import (make_constant_v0, simulate_positions,
                              render_positions_multicam, video_to_uint8)
    from ..scene_io import load_from_spec
    from .._env import Camera
    from ..run_io import RunDir, save_panel_video

    t0 = time.time()
    rd = RunDir.create(__name__, cfg.run_label or cfg.scene.display_name, cfg.out)
    rd.write_config({"task": "explore.cam_compound_probe", "scene": cfg.scene.to_dict(),
                     "sim": cfg.sim.to_dict(), "n_traj": cfg.n_traj, "azim": list(cfg.azim),
                     "elev": list(cfg.elev), "radius": list(cfg.radius),
                     "n_waypoints": cfg.n_waypoints, "v0": list(cfg.v0), "seed": cfg.seed})

    scene = load_from_spec(cfg.scene, cfg.sim)
    ref_cam = scene.camera_by_frame(cfg.frame)
    pos0 = (scene.sim_xyzs * scene.scale - scene.shift).detach()
    center = pos0.mean(0).cpu().numpy()
    try:
        ref_eye = ref_cam.camera_center.detach().cpu().numpy()
    except Exception:
        ref_eye = (-np.asarray(ref_cam.R) @ np.asarray(ref_cam.T))
    ref_r = float(np.linalg.norm(ref_eye - center)) or 1.0
    d0 = (ref_eye - center) / ref_r
    FoVx, FoVy = float(ref_cam.FoVx), float(ref_cam.FoVy)
    Hh, Ww = int(ref_cam.image_height), int(ref_cam.image_width)
    up = np.asarray(cfg.up, dtype=np.float64)
    Rmat = np.asarray(ref_cam.R, dtype=np.float64)
    right_cam = Rmat[:, 0] / (np.linalg.norm(Rmat[:, 0]) or 1.0)
    up_cam = -Rmat[:, 1] / (np.linalg.norm(Rmat[:, 1]) or 1.0)

    def eye_of(azim_deg, elev_deg, r):
        """Eye for (azim, elev, r): yaw d0 about up_cam, then pitch about the
        yawed right axis -- a point on the sphere, no Euler-orientation compose."""
        a, e = np.deg2rad(azim_deg), np.deg2rad(elev_deg)
        d1 = _rodrigues(d0, up_cam, a)
        r1 = _rodrigues(right_cam, up_cam, a)
        d2 = _rodrigues(d1, r1, e)
        return center + r * d2

    T = cfg.sim.num_frames
    if any(cfg.v0):
        pos_list = simulate_positions(scene, float(cfg.E), make_constant_v0(scene, cfg.v0), cfg.sim)
    else:
        pos_list = [pos0.clone() for _ in range(T)]

    rng = np.random.RandomState(cfg.seed)
    clips, labels, paths_json, paths = [], [], {}, []
    for j in range(cfg.n_traj):
        wp = np.stack([
            rng.uniform(*cfg.azim, cfg.n_waypoints),
            rng.uniform(*cfg.elev, cfg.n_waypoints),
            rng.uniform(*cfg.radius, cfg.n_waypoints)], axis=1)        # [K,3] (azim,elev,r)
        path = _catmull(wp, T)                                         # [T,3]
        path[:, 0] = np.clip(path[:, 0], *cfg.azim)                    # clamp spline overshoot
        path[:, 1] = np.clip(path[:, 1], *cfg.elev)
        path[:, 2] = np.clip(path[:, 2], *cfg.radius)
        cams = []
        for (az, el, r) in path:
            R, Tcw = lookat_R_T(eye_of(az, el, r), center, up)
            cams.append(Camera(R=R, T=Tcw, FoVx=FoVx, FoVy=FoVy, img_path=cfg.frame,
                               img_hw=(Hh, Ww), data_device=scene.device))
        clips.append(video_to_uint8(render_positions_multicam(scene, pos_list, cams)))
        labels.append(f"#{j} az{path[0,0]:.0f}->{path[-1,0]:.0f} el{path[0,1]:.0f}->{path[-1,1]:.0f}")
        paths.append(path)
        paths_json[f"traj_{j}"] = {"azim": path[:, 0].tolist(), "elev": path[:, 1].tolist(),
                                   "radius": path[:, 2].tolist()}
        print(f"  traj {j}: azim {path[:,0].min():.0f}..{path[:,0].max():.0f}  "
              f"elev {path[:,1].min():.0f}..{path[:,1].max():.0f}  "
              f"r {path[:,2].min():.2f}..{path[:,2].max():.2f}")

    panel = save_panel_video(rd.path("panel.gif"), clips, labels, fps=cfg.sim.fps,
                             title=f"{scene.name}  compound dyn-cam (index-matched to paths.png)")

    # parameter-space paths: azim(x) vs elev(y), colour=time, marker size ~ r
    ncol = min(3, cfg.n_traj); nrow = int(np.ceil(cfg.n_traj / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.4 * nrow), squeeze=False)
    for j, path in enumerate(paths):
        ax = axs[j // ncol][j % ncol]
        sizes = 20 + 120 * (path[:, 2] - cfg.radius[0]) / (cfg.radius[1] - cfg.radius[0] + 1e-9)
        ax.plot(path[:, 0], path[:, 1], "-", c="0.7", lw=1, zorder=1)
        sc = ax.scatter(path[:, 0], path[:, 1], c=np.arange(T), cmap="viridis",
                        s=sizes, zorder=2)
        ax.scatter([path[0, 0]], [path[0, 1]], marker="*", s=180, c="red", zorder=3)
        ax.set_xlim(cfg.azim[0] - 5, cfg.azim[1] + 5); ax.set_ylim(cfg.elev[0] - 5, cfg.elev[1] + 5)
        ax.set_xlabel("azim (deg)"); ax.set_ylabel("elev (deg)")
        ax.set_title(f"#{j}  r {path[:,2].min():.2f}-{path[:,2].max():.2f}")
        ax.grid(alpha=0.3)
    for k in range(cfg.n_traj, nrow * ncol):
        axs[k // ncol][k % ncol].axis("off")
    fig.suptitle("compound camera paths: azim-elev (colour=time, size=r, red*=start)")
    fig.tight_layout(); fig.savefig(rd.path("paths.png"), dpi=120); plt.close(fig)
    rd._event("paths.png", "paths.png")

    rd.write_json("paths.json", {"center": center.tolist(), "ref_radius": ref_r,
                                 "azim": list(cfg.azim), "elev": list(cfg.elev),
                                 "radius": list(cfg.radius), "trajectories": paths_json})
    rd.finish()
    print(f"[cam_compound] {cfg.n_traj} trajectories -> {rd.root}  ({time.time()-t0:.1f}s)\n"
          f"  panel: {panel}\n  paths: {rd.path('paths.png')}")
    return rd.root


if __name__ == "__main__":
    run(tyro.cli(CamCompoundProbeConfig))
