"""Observation tool: generate a SMALL batch of axis-aligned MPM trajectories and
render BOTH flow and rgb from the SAME stored 3D trajectory, so the dataset setup
can be eyeballed before committing GPU to a full run.

Design (settled with the user):
- rot_z is an OBJECT transform (rotate sim_xyzs about the grid centre so the cord's
  principal axis aligns with the coord axes). It is decoupled from the camera: v0 is
  expressed in MPM coords, so aligning the object is what makes "v0 along x" mean
  "along the cord". The trajectory is rolled in this rotated frame and STORED -- it is
  the expensive, reusable artifact; flow and rgb are just two projections of it.
- rgb uses pseudo-gaussians built AT the (rotated) particles (gic-style: isotropic,
  DC colour from the nearest original gaussian), so rotating the object rotates the
  render. flow projects the same particles. Both share one camera -> 1-1 by construction.

explore tool: LOCAL tyro config, reads SceneSpec/SimConfig/V0Dist; output under
outputs/explore/aligned_observe/NN/.

  python -m reuse_mpm.explore.aligned_observe --n 4 --rot_z_deg 67.6
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List, Tuple

import tyro

from ..config import ScenePreset, SceneSpec, SimConfig, V0Dist
from ..gpu import pick_free_gpu


@dataclass
class ObserveConfig:
    """Small axis-aligned observation batch (read SceneSpec/SimConfig/V0Dist)."""

    scene: SceneSpec = field(default_factory=lambda: SceneSpec(preset=ScenePreset.telephone))
    sim: SimConfig = field(default_factory=lambda: SimConfig(num_frames=16, substep=64))
    v0_dist: V0Dist = field(default_factory=lambda: V0Dist(
        mode="axis", axis=0, signed=True, mag_min=2.0, mag_max=8.0))
    E: float = 1e5
    rot_z_deg: float = 67.6          # object alignment (cord principal axis -> coord x)
    n: int = 4
    frames: int = 8                  # rgb/flow frames (<= sim.num_frames); flow fields = frames-1
    res: int = 128                   # flow/rgb raster resolution
    cam_back: float = 1.0            # >1 dollies the camera back (frame large motion); camera-only
    frame: str = "frame_00001.png"
    seed: int = 0
    out: str | None = None


def rot_z(p, deg: float):
    """Rotate (N,3) about z through (0.5,0.5) [normalised MPM frame]. Returns (N,3)."""
    import math
    import torch
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    x, y = p[:, 0] - 0.5, p[:, 1] - 0.5
    q = p.clone()
    q[:, 0] = c * x - s * y + 0.5
    q[:, 1] = s * x + c * y + 0.5
    return q


def project(X, full_proj, W: int, H: int):
    """world [n,3] torch -> (uv [n,2] pixel, w [n] clip-w). Uses the SAME
    full_proj_transform the GaussianRasterizer uses, so flow aligns with the render
    exactly (ndc2pix: pix = ((ndc+1)*size-1)*0.5)."""
    import torch
    Xh = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], 1)   # [n,4]
    clip = Xh @ full_proj                                                # [n,4] world->clip
    w = clip[:, 3:4].clamp(min=1e-6)                                      # [n,1]
    ndc = clip[:, :2] / w                                                 # [n,2]
    u = ((ndc[:, 0] + 1.0) * W - 1.0) * 0.5
    v = ((ndc[:, 1] + 1.0) * H - 1.0) * 0.5
    return torch.stack([u, v], 1), clip[:, 3]                            # [n,2], [n] (w>0 = in front)


def build_pseudo_gaussian(xyz_world, dc_color, vol, scale: float):
    """physdreamer GaussianModel AT particle world positions (isotropic, DC-only).

    xyz_world [n,3]; dc_color [n,1,3] (features_dc of nearest orig gaussian);
    vol [n] per-particle volume. Returns a GaussianModel ready for render_gaussian.
    """
    import torch
    from .._env import GaussianModel
    from physdreamer.gaussian_3d.utils.general_utils import inverse_sigmoid
    n = xyz_world.shape[0]
    g = GaussianModel(0)
    g._xyz = xyz_world.detach().clone()                                  # [n,3]
    g._features_dc = dc_color.detach().clone()                           # [n,1,3]
    g._features_rest = torch.zeros((n, 0, 3), device=xyz_world.device)
    s = (vol.clamp_min(1e-12) ** (1.0 / 3.0)) * scale                    # [n] world-ish radius
    g._scaling = torch.log(s).unsqueeze(1).repeat(1, 3)                  # [n,3]
    rot = torch.zeros((n, 4), device=xyz_world.device); rot[:, 0] = 1.0
    g._rotation = rot                                                    # [n,4] identity quat
    g._opacity = inverse_sigmoid(0.9 * torch.ones((n, 1), device=xyz_world.device))
    g.active_sh_degree = 0
    return g


def flow_to_rgb(f2):
    """packed flow [H,W,2] in [0,1] -> [H,W,3] viz (B=0.5)."""
    import numpy as np
    return np.concatenate([f2, np.full(f2.shape[:2] + (1,), 0.5, np.float32)], -1)


def run(cfg: ObserveConfig):
    pick_free_gpu(min_quota_hours=0.0)

    import math
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from types import SimpleNamespace

    from ..scene_io import load_from_spec
    from ..sim_render import make_constant_v0, simulate_positions, render_positions
    from .._env import GaussianModel
    from physdreamer.gaussian_3d.gaussian_renderer.render import render_gaussian
    from ..run_io import RunDir
    import torch.nn.functional as _F

    rd = RunDir.create(__name__, f"n{cfg.n}_rot{cfg.rot_z_deg:g}_mag{cfg.v0_dist.mag_min:g}-{cfg.v0_dist.mag_max:g}",
                       cfg.out, config=cfg)
    with rd.capture_output():
        scene = load_from_spec(cfg.scene, cfg.sim)
        dev = scene.device
        rd.copy_in(cfg.scene.cache_path, "scene_cache.pt")

        # --- DC colour per particle (nearest original gaussian), in the UNROTATED frame
        orig_norm = (scene.gaussians._xyz.detach() + scene.shift) / scene.scale   # [G,3] normalised
        idx = []
        for i in range(0, scene.sim_xyzs.shape[0], 4096):
            d = torch.cdist(scene.sim_xyzs[i:i + 4096], orig_norm)                 # [b,G]
            idx.append(d.argmin(1))
        idx = torch.cat(idx)                                                       # [n]
        dc = scene.gaussians._features_dc.detach()[idx]                            # [n,1,3]
        vol = torch.from_numpy(scene.points_vol).float().to(dev)                   # [n]

        # --- OBJECT transform: rotate the cord's principal axis onto coord x.
        # Rotate BOTH the sim particles AND the full gaussian cloud (in normalised space)
        # so the with-bg render (render_positions, real gaussians) and the no-bg render
        # (pseudo-gaussians on sim particles) sit in the same rotated frame.
        if cfg.rot_z_deg:
            scene.sim_xyzs = rot_z(scene.sim_xyzs, cfg.rot_z_deg)
            scene.sim_aabb = torch.stack([scene.sim_xyzs.min(0).values, scene.sim_xyzs.max(0).values])
            gnorm = (scene.gaussians._xyz.detach() + scene.shift) / scene.scale    # [G,3] normalised
            gnorm = rot_z(gnorm, cfg.rot_z_deg)
            scene.gaussians._xyz = (gnorm * scene.scale - scene.shift)             # back to world
            rd.note(f"rotated object z {cfg.rot_z_deg} deg (principal axis -> coord x)")

        # --- camera (shared by flow + rgb). cam_back widens FoV (same viewpoint) to
        # frame large motion: object shrinks, both project() and render() read this FoV
        # so flow/rgb stay 1-1. This is a CAMERA transform, independent of the object rot.
        cam0 = scene.camera_by_frame(cfg.frame)
        if cfg.cam_back != 1.0:
            fx = 2 * math.atan(math.tan(cam0.FoVx / 2) * cfg.cam_back)
            fy = 2 * math.atan(math.tan(cam0.FoVy / 2) * cfg.cam_back)
            from .._env import Camera
            cam = Camera(R=cam0.R, T=cam0.T, FoVx=fx, FoVy=fy, img_path=cfg.frame,
                         img_hw=(int(cam0.image_height), int(cam0.image_width)), data_device=dev)
        else:
            cam = cam0
        W, H = int(cam.image_width), int(cam.image_height)
        full_proj = cam.full_proj_transform.to(dev)                                # [4,4] world->clip
        pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
        bg = torch.ones(3, device=dev)

        # consistent panel mapping: full camera frame (W,H) -> (PH,PW), aspect preserved,
        # shared by ALL three sub-panels so the object sits identically in image space.
        PW = cfg.res
        PH = int(round(cfg.res * H / W))

        def to_panel(img_chw: "torch.Tensor") -> np.ndarray:
            """[3,H,W] in [0,1] -> [PH,PW,3] numpy (proper resize, no crop)."""
            r = _F.interpolate(img_chw.unsqueeze(0), size=(PH, PW), mode="bilinear",
                               align_corners=False)[0]
            return r.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()

        rng = np.random.RandomState(cfg.seed)
        F = cfg.frames
        sim_i = dataclasses.replace(cfg.sim, num_frames=F)
        for i in range(cfg.n):
            vec = cfg.v0_dist.sample(rng)                                          # (3,)
            v0 = make_constant_v0(scene, vec)                                      # [n,3]
            pos_list = simulate_positions(scene, cfg.E, v0, sim_i)                 # list[F] [n,3] world
            traj = torch.stack(pos_list, 0)                                        # [F,n,3] world
            np.save(rd.path(f"sample_{i:03d}_mpm_xyz.npy"), traj.cpu().numpy())    # PRIMARY artifact

            # with-bg render: full scene (real gaussians, rotated), sim part displaced
            withbg = render_positions(scene, pos_list, cam)                        # [F,3,H,W] in [0,1]

            # project every frame once into panel coords; per-clip flow scale (95p |disp|)
            uvs, zvs = [], []
            for t in range(F):
                u, z = project(traj[t], full_proj, W, H)                          # [n,2],[n]
                uvs.append(u * (PW / W)); zvs.append(z)                            # -> panel coords
            disps = [uvs[t + 1] - uvs[t] for t in range(F - 1)]                    # [n,2] panel px
            allmag = torch.cat([d.norm(2, dim=-1) for d in disps]) if disps else torch.zeros(1)
            scale_px = float(torch.quantile(allmag, 0.95).clamp(min=1e-3))
            import json
            json.dump({"v0": list(map(float, vec)), "E": cfg.E, "rot_z_deg": cfg.rot_z_deg,
                       "cam_back": cfg.cam_back, "flow_scale_px_panel": scale_px},
                      open(rd.path(f"sample_{i:03d}.json"), "w"))

            panel = []
            for t in range(F):
                p_bg = to_panel(withbg[t])                                         # [PH,PW,3] with bg
                nobg = render_gaussian(cam, build_pseudo_gaussian(traj[t], dc, vol, scene.scale),
                                       pipe, bg)["render"].clamp(0, 1)             # [3,H,W]
                p_nobg = to_panel(nobg)                                            # [PH,PW,3] no bg
                if t < F - 1:
                    packed = (disps[t] / (2 * scale_px) + 0.5).clamp(0, 1)         # [n,2]
                    fl = _splat(uvs[t], packed, zvs[t], PH, PW)                    # [PH,PW,2]
                else:
                    fl = np.full((PH, PW, 2), 0.5, np.float32)
                bar = np.ones((PH, 2, 3), np.float32)
                row = np.concatenate([p_bg, bar, p_nobg, bar, flow_to_rgb(fl)], 1)
                panel.append((row * 255).round().astype("uint8"))
            imageio.mimsave(rd.path(f"sample_{i:03d}_bg-nobg-flow.gif"), panel, fps=3)
            print(f"  sample {i}: v0={[round(x,2) for x in vec]} |v0|={np.linalg.norm(vec):.2f} "
                  f"flow_scale={scale_px:.1f}px(panel)", flush=True)
        rd.finish()
    print(f"done -> {rd.path()}")
    return rd


def _splat(uv, disp, zv, ph: int, pw: int):
    """nearest-pixel splat of per-particle disp -> [ph,pw,2] (viz only). uv in panel
    coords (x in [0,pw), y in [0,ph)). numpy out."""
    import torch
    acc = torch.zeros(ph * pw, 2, device=uv.device); cnt = torch.zeros(ph * pw, device=uv.device)
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < pw) & (uv[:, 1] >= 0) & (uv[:, 1] < ph) & (zv > 0)
    col = uv[inb, 0].round().long().clamp(0, pw - 1); row = uv[inb, 1].round().long().clamp(0, ph - 1)
    flat = row * pw + col
    acc.index_add_(0, flat, disp[inb]); cnt.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
    out = (acc / cnt.clamp(min=1).unsqueeze(1)).view(ph, pw, 2)
    out[cnt.view(ph, pw) == 0] = 0.5
    return out.cpu().numpy()


if __name__ == "__main__":
    run(tyro.cli(ObserveConfig))
