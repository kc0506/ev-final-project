"""Generate an axis-aligned flow dataset: sample v0 = +-|vx| (signed, magnitude band),
roll the MPM on the CENTRED + rot_z-ALIGNED scene, and store the 3D trajectory
(mpm_xyz) -- the expensive, reusable artifact. Layout matches dataset_gen so the
EXISTING downstream tools reuse it unchanged:
  - teacher/build_flow_pack.py  -> flow_pack (projects mpm_xyz through camera.json)
  - teacher/train_flow.py       -> the flow-diffusion teacher

Separate entrypoint (does NOT touch dataset_gen). rot_z is an OBJECT transform that
aligns the cord's principal axis to coord x, so v0 along x means along the cord; it is
baked into the stored trajectory. The camera FoV is widened (cam_back) once so the large
|vx| motion stays in-frame; that widened camera is written to camera.json as the single
downstream projection source.

  python -m reuse_mpm.gen_flow_aligned --n 128
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
from dataclasses import dataclass, field

import tyro

from .config import ScenePreset, SceneSpec, SimConfig, V0Dist
from .gpu import pick_free_gpu


@dataclass
class GenFlowConfig:
    """Axis-aligned flow dataset (reads SceneSpec/SimConfig/V0Dist from config.py)."""

    scene: SceneSpec = field(default_factory=lambda: SceneSpec(preset=ScenePreset.telephone))
    sim: SimConfig = field(default_factory=lambda: SimConfig(num_frames=16, substep=64))
    v0_dist: V0Dist = field(default_factory=lambda: V0Dist(
        mode="axis", axis=0, signed=True, mag_min=2.0, mag_max=8.0))
    E: float = 1e5
    rot_z_deg: float = 67.6           # object alignment (cord principal axis -> coord x)
    density: str = "uniform"          # "uniform" U[a,b] | "linear" p(m)∝m on [a,b] (ramp)
    n: int = 128
    frames: int = 8                   # stored frames -> build_flow_pack makes frames-1 fields
    cam_back: float = 2.0             # widen FoV so large |vx| stays in-frame (camera-only)
    frame: str = "frame_00001.png"
    seed: int = 0
    out: str | None = None


def rot_z(p, deg: float):
    """Rotate (N,3) about z through (0.5,0.5) [normalised MPM frame]. Returns (N,3)."""
    import torch  # noqa: F401
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    x, y = p[:, 0] - 0.5, p[:, 1] - 0.5
    q = p.clone()
    q[:, 0] = c * x - s * y + 0.5
    q[:, 1] = s * x + c * y + 0.5
    return q


def run(cfg: GenFlowConfig):
    pick_free_gpu(min_quota_hours=0.0)

    import numpy as np
    import torch
    from .scene_io import load_from_spec
    from .sim_render import make_constant_v0, simulate_positions
    from ._env import Camera
    from .ply_io import camera_to_dict
    from .run_io import RunDir

    label = f"n{cfg.n}_axisx_mag{cfg.v0_dist.mag_min:g}-{cfg.v0_dist.mag_max:g}_rot{cfg.rot_z_deg:g}"
    rd = RunDir.create(__name__, label, cfg.out, config=cfg)
    with rd.capture_output():
        scene = load_from_spec(cfg.scene, cfg.sim)
        rd.copy_in(cfg.scene.cache_path, "scene_cache.pt")          # freeze_mask source for build_flow_pack

        # OBJECT transform: align the cord's principal axis to coord x (baked into traj).
        if cfg.rot_z_deg:
            scene.sim_xyzs = rot_z(scene.sim_xyzs, cfg.rot_z_deg)
            scene.sim_aabb = torch.stack([scene.sim_xyzs.min(0).values, scene.sim_xyzs.max(0).values])
            rd.note(f"rotated object z {cfg.rot_z_deg} deg (principal axis -> coord x)")

        # CAMERA: widen FoV (same viewpoint) so large |vx| stays in-frame; the single
        # downstream projection source. Written verbatim to camera.json.
        cam0 = scene.camera_by_frame(cfg.frame)
        fx = 2 * math.atan(math.tan(cam0.FoVx / 2) * cfg.cam_back)
        fy = 2 * math.atan(math.tan(cam0.FoVy / 2) * cfg.cam_back)
        cam = Camera(R=cam0.R, T=cam0.T, FoVx=fx, FoVy=fy, img_path=cfg.frame,
                     img_hw=(int(cam0.image_height), int(cam0.image_width)), data_device=scene.device)
        json.dump(camera_to_dict(cam), open(rd.path("camera.json"), "w"), indent=2)

        rng = np.random.RandomState(cfg.seed)
        sim_i = dataclasses.replace(cfg.sim, num_frames=cfg.frames)
        d = cfg.v0_dist

        def sample_v0() -> np.ndarray:
            """v0 along axis `d.axis`; magnitude per cfg.density; sign per d.signed. (3,)"""
            a, b = d.mag_min, d.mag_max
            if cfg.density == "linear":          # p(m) ∝ m on [a,b]  =>  m = sqrt(U[a²,b²])
                m = float(np.sqrt(rng.uniform(a * a, b * b)))
            elif cfg.density == "linear_desc":   # p(m) ∝ (b-m) on [a,b]  =>  reflect ascending about (a+b)/2
                m = float((a + b) - np.sqrt(rng.uniform(a * a, b * b)))
            else:                                # uniform U[a,b]
                m = float(rng.uniform(a, b))
            sign = -1.0 if (d.signed and rng.uniform() < 0.5) else 1.0
            vec = np.zeros(3, np.float32)
            vec[d.axis] = sign * m
            return vec

        vxs = []
        for i in range(cfg.n):
            vec = sample_v0()                                                   # (3,)
            v0 = make_constant_v0(scene, vec)                                   # [n,3]
            pos_list = simulate_positions(scene, cfg.E, v0, sim_i)             # list[F] [n,3] world
            mpm_xyz = torch.stack(pos_list, 0).cpu().numpy()                    # [F,n,3] world (ROTATED)
            sd = rd.path(f"sample_{i:04d}")
            os.makedirs(sd, exist_ok=True)
            np.save(os.path.join(sd, "mpm_xyz.npy"), mpm_xyz)                   # PRIMARY artifact
            json.dump({"id": i, "E": cfg.E, "v0": list(map(float, vec)),
                       "v0_magnitude": float(np.linalg.norm(vec)), "rot_z_deg": cfg.rot_z_deg},
                      open(os.path.join(sd, "sample.json"), "w"))
            vxs.append(float(vec[0]))
            if i % 16 == 0:
                print(f"  {i}/{cfg.n}  vx={vec[0]:+.2f}", flush=True)
        vxs = np.array(vxs)
        rd.merge_config(n=cfg.n, vx_min=float(vxs.min()), vx_max=float(vxs.max()),
                        n_pos=int((vxs > 0).sum()), n_neg=int((vxs < 0).sum()),
                        cam_back=cfg.cam_back, frames=cfg.frames)
        print(f"done: {cfg.n} samples | vx [{vxs.min():.2f},{vxs.max():.2f}] "
              f"+{int((vxs>0).sum())}/-{int((vxs<0).sum())}", flush=True)
        rd.finish()
    print(f"dataset -> {rd.path()}")
    return rd


if __name__ == "__main__":
    run(tyro.cli(GenFlowConfig))
