"""Minimal scene for the MPM->flow path: reconstruct exactly the fields the
differentiable rollout + camera projection need, straight from a scene_cache.pt
`disc` blob -- WITHOUT instantiating a GaussianModel / loading the PLY (those are
only needed to RGB-render gaussians, which the flow path never does).

build_mpm (reuse_mpm.sim_render) reads: device, sim_xyzs[n,3], points_vol[n],
freeze_mask[n]. MpmRollout additionally reads sim_xyzs. query_mask = ~freeze_mask.
We mirror SceneBundle's duck-type with a dataclass carrying just those.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import Tensor


@dataclass
class MinScene:
    """Duck-typed subset of reuse_mpm.scene.SceneBundle for the flow path."""

    device: str
    sim_xyzs: Tensor          # [n, 3] normalised MPM particle positions (rest)
    points_vol: np.ndarray    # [n]    per-particle volume
    freeze_mask: Tensor       # [n]    bool: frozen/anchor particles
    sim_aabb: Tensor          # [2, 3] padded aabb of sim particles
    scale: float              # world = normalised * scale - shift
    shift: Tensor             # [] or [3]

    @property
    def query_mask(self) -> Tensor:
        """[n] bool: the MOVING (non-frozen) particles."""
        return torch.logical_not(self.freeze_mask)


def load_min_scene(cache_path: str, device: str = "cuda:0") -> MinScene:
    """Load scene_cache.pt -> MinScene (no GaussianModel). Returns a MinScene whose
    tensors live on `device`."""
    blob = torch.load(cache_path, map_location=device, weights_only=False)  # dict
    d = blob["disc"]
    return MinScene(
        device=device,
        sim_xyzs=d["sim_xyzs"].to(device).float(),                 # [n,3]
        points_vol=np.asarray(d["points_vol"], dtype=np.float32),  # [n]
        freeze_mask=d["freeze_mask"].to(device).bool(),            # [n]
        sim_aabb=d["sim_aabb"].to(device).float(),                 # [2,3]
        scale=float(d["scale"]),
        shift=d["shift"].to(device).float() if torch.is_tensor(d["shift"])
        else torch.tensor(float(d["shift"]), device=device),
    )


@dataclass
class CameraParams:
    """Pinhole projection params read from camera.json (build_flow_pack convention)."""

    world_view_transform: Tensor  # [4, 4]
    fovx: float
    fovy: float
    width: int
    height: int


def apply_scene_fixes(scene: MinScene, rot_z_deg: float = 0.0,
                      recenter: bool = True) -> MinScene:
    """Apply the two gic-validated scene fixes to a MinScene (in normalised MPM space):

    - recenter: shift the particle-cloud bbox centre to 0.5 on every axis (the 'v2'
      centred normalisation: legacy shift parked the min corner ~2.5 cells from the
      g2p position clamp at 2*dx, so excitation toward that wall pinned particles
      bit-exactly -- positions AND grads invalid). Centring gives >=0.16 margin/side.
    - rot_z_deg: rotate the cloud about (0.5,0.5) in the xy-plane (gic PCA aligns the
      cord's principal axis / the anchor wall to the camera plane; default there 67.6).

    Particle identity is preserved, so freeze_mask / points_vol stay valid.
    """
    import torch as _t
    p = scene.sim_xyzs                                       # [n,3] normalised
    shift = scene.shift
    if not _t.is_tensor(shift) or shift.ndim == 0:
        shift = _t.full((3,), float(shift), device=p.device)  # scalar -> [3]
    else:
        shift = shift.clone().to(p.device)
    if recenter:
        lo3 = p.min(dim=0).values                            # [3]
        hi3 = p.max(dim=0).values                            # [3]
        delta = 0.5 - (lo3 + hi3) / 2.0                       # [3] normalised shift
        p = p + delta                                        # bbox centre -> 0.5
        shift = shift + delta * scene.scale                  # keep world = p*scale - shift fixed
    if rot_z_deg:
        import math
        c, s = math.cos(math.radians(rot_z_deg)), math.sin(math.radians(rot_z_deg))
        x, y = p[:, 0] - 0.5, p[:, 1] - 0.5                  # rotate about centre (xy-plane)
        pr = p.clone()
        pr[:, 0] = c * x - s * y + 0.5
        pr[:, 1] = s * x + c * y + 0.5
        p = pr                                               # object visibly reorients vs camera
    aabb = _t.stack([p.min(dim=0).values, p.max(dim=0).values])  # [2,3]
    return MinScene(device=scene.device, sim_xyzs=p.contiguous(), points_vol=scene.points_vol,
                    freeze_mask=scene.freeze_mask, sim_aabb=aabb, scale=scene.scale,
                    shift=shift)


def load_camera(camera_json: str, device: str = "cuda:0") -> CameraParams:
    """Load camera.json -> CameraParams on `device`. Keys mirror flow_viz.project()."""
    c = json.load(open(camera_json))
    wvt = torch.tensor(c["world_view_transform"], dtype=torch.float32, device=device)  # [4,4]
    return CameraParams(
        world_view_transform=wvt,
        fovx=float(c["FoVx"]) if "FoVx" in c else float(c["fovx"]),
        fovy=float(c["FoVy"]) if "FoVy" in c else float(c["fovy"]),
        width=int(c.get("image_width", c.get("width"))),
        height=int(c.get("image_height", c.get("height"))),
    )
