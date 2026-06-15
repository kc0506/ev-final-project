"""Load a PhysGaussian-format scene into the same SceneBundle used by the rest.

PhysGaussian scenes are pure foreground (every gaussian is the object) and ship
as: point_cloud/iteration_*/point_cloud.ply + cameras.json (3DGS format) + cfg.
There is no clean/moving segmentation ply -- movement is controlled by boundary
conditions. Per the project decision we:
  - take ALL gaussians as the simulated object (sim_mask all True)
  - define the anchor BC geometrically: freeze a thin slab at one end of the
    object's longest axis (no ply needed). Uniform v0 on the free part then bends
    it -> E-dependent deformation (a uniform v0 on an unanchored body would just
    translate rigidly and make E unidentifiable).

Coordinate / normalisation / KNN conventions are identical to scene.load_scene so
the shared simulate_and_render works unchanged.
"""
from __future__ import annotations

import glob
import json
import os
from typing import List, Optional

import numpy as np
import torch

from . import _env
from ._env import (
    GaussianModel,
    Camera,
    get_volume,
    downsample_with_kmeans_gpu_with_chunk,
)
from physdreamer.data.cameras import focal2fov
from .scene import SceneBundle


def _cameras_from_json(path: str) -> List[Camera]:
    """Build graphdeco Cameras from a 3DGS cameras.json.

    3DGS dumps position = camera centre (c2w translation), rotation = c2w rotation.
    graphdeco Camera wants R = c2w rotation, T = world-to-camera translation.
    """
    cams = []
    for c in json.load(open(path)):
        rot = np.array(c["rotation"], dtype=np.float64)   # c2w rotation
        pos = np.array(c["position"], dtype=np.float64)   # camera centre
        R = rot
        T = -rot.T @ pos                                  # w2c translation
        W, H = int(c["width"]), int(c["height"])
        cams.append(Camera(
            R=R, T=T,
            FoVx=focal2fov(c["fx"], W), FoVy=focal2fov(c["fy"], H),
            img_path=c["img_name"], img_hw=(H, W),
        ))
    return cams


def _find_ply(model_dir: str) -> str:
    cands = sorted(glob.glob(os.path.join(
        model_dir, "point_cloud", "iteration_*", "point_cloud.ply")))
    assert cands, f"no point_cloud.ply under {model_dir}/point_cloud/iteration_*"
    # prefer the highest iteration
    return cands[-1]


def default_pg_cache_path(model_dir, downsample_scale, grid_size, top_k=8):
    from .scene import _CACHE_ROOT
    name = os.path.basename(os.path.normpath(model_dir))
    return os.path.join(_CACHE_ROOT, f"PG_{name}_ds{downsample_scale}_g{grid_size}_k{top_k}.pt")


def load_physgaussian_scene(
    model_dir: str,
    name: Optional[str] = None,
    device: str = "cuda:0",
    downsample_scale: float = 0.1,
    grid_size: int = 32,
    top_k: int = 8,
    max_particles: int = 8000,
    freeze_frac: float = 0.15,
    freeze_axis: Optional[int] = None,
    cache_path: Optional[str] = None,
) -> SceneBundle:
    if name is None:
        name = os.path.basename(os.path.normpath(model_dir))

    gauss = GaussianModel(3)
    gauss.load_ply(_find_ply(model_dir))
    gauss.detach_grad()
    cameras = _cameras_from_json(os.path.join(model_dir, "cameras.json"))
    H = cameras[0].image_height
    W = cameras[0].image_width

    if cache_path is not None and os.path.exists(cache_path):
        blob = torch.load(cache_path, map_location=device)
        if blob.get("norm_version") != 2:
            print("[scene-PG] WARNING: legacy-normalization cache; see scene.py "
                  "warning (wall-clamp risk for new data with large v0).")
        d = blob["disc"]
        sim_mask = d["sim_mask"].to(device)
        sim_xyzs = d["sim_xyzs"].to(device)
        points_vol = d["points_vol"]
        top_k_index = d["top_k_index"].to(device)
        freeze_mask = d["freeze_mask"].to(device)
        sim_aabb = d["sim_aabb"].to(device)
        scale = float(d["scale"]); shift = d["shift"].to(device)
        print(f"[scene-PG] loaded cache {cache_path} ({sim_xyzs.shape[0]} particles)")
    else:
        # all gaussians are the object
        sim_mask = torch.ones(gauss._xyz.shape[0], dtype=torch.bool, device=device)
        obj = gauss._xyz[sim_mask, :].detach().clone()

        pos_max = obj.max(); pos_min = obj.min()
        scale = (pos_max - pos_min) * 1.8
        # norm v2: per-axis centering at 0.5; see scene.load_scene for rationale
        # (legacy 0.25*range shift sits ~2.5 cells from the g2p position clamp).
        lo3 = obj.min(dim=0).values  # [3]
        hi3 = obj.max(dim=0).values  # [3]
        shift = scale * 0.5 - (lo3 + hi3) / 2.0  # [3]
        obj_n = (obj + shift) / scale

        sim_aabb = torch.stack([obj_n.min(0)[0], obj_n.max(0)[0]], 0)
        sim_aabb = (sim_aabb - sim_aabb.mean(0, keepdim=True)) * 1.2 + sim_aabb.mean(0, keepdim=True)

        num_cluster = min(int(obj_n.shape[0] * downsample_scale), max_particles)
        # kmeans_gpu snaps EMPTY clusters to EXACTLY [0,0,0] (see scene.load_scene);
        # normalisation puts every real coord at >= 0.25/1.8 > 0, so exact-zero rows
        # are unambiguously ghosts. Assert, then drop them before deriving KNN/freeze.
        assert (obj_n.min(dim=0).values > 0).all(), \
            "pre-kmeans particles have a coord <= 0; exact-zero ghost filter unsafe"
        sim_xyzs = downsample_with_kmeans_gpu_with_chunk(obj_n, num_cluster)
        ghost = (sim_xyzs == 0).all(dim=1)
        if bool(ghost.any()):
            print(f"[scene-PG] dropped {int(ghost.sum())} k-means empty-cluster ghost(s) at origin")
            sim_xyzs = sim_xyzs[~ghost]

        # KNN gauss->particle, chunked to bound memory on big scenes
        gpos = (gauss._xyz[sim_mask, :].detach().clone() + shift) / scale
        idx_chunks = []
        for i in range(0, gpos.shape[0], 50000):
            cd = torch.cdist(gpos[i:i + 50000], sim_xyzs) * -1.0
            idx_chunks.append(torch.topk(cd, top_k, dim=-1)[1])
        top_k_index = torch.cat(idx_chunks, 0)

        points_vol = get_volume(sim_xyzs.detach().cpu().numpy())

        # geometric anchor: freeze the lowest `freeze_frac` FRACTION OF PARTICLES
        # along the longest axis (percentile, not spatial range -- objects like a
        # plant are bottom-sparse so a range threshold would freeze ~nothing).
        ax = freeze_axis if freeze_axis is not None else int(
            (sim_xyzs.max(0)[0] - sim_xyzs.min(0)[0]).argmax().item())
        thr = torch.quantile(sim_xyzs[:, ax], freeze_frac)
        freeze_mask = (sim_xyzs[:, ax] < thr)
        scale = float(scale)

        if cache_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            torch.save({
                "downsample_scale": downsample_scale, "grid_size": grid_size,
                "top_k": top_k, "model_dir": model_dir, "freeze_axis": ax,
                "norm_version": 2,
                "disc": {
                    "sim_mask": sim_mask.cpu(), "sim_xyzs": sim_xyzs.cpu(),
                    "points_vol": points_vol, "top_k_index": top_k_index.cpu(),
                    "freeze_mask": freeze_mask.cpu(), "sim_aabb": sim_aabb.cpu(),
                    "scale": scale, "shift": shift.cpu(),
                },
            }, cache_path)
            print(f"[scene-PG] saved cache -> {cache_path} "
                  f"({sim_xyzs.shape[0]} particles, freeze axis {ax}, "
                  f"{int(freeze_mask.sum())} frozen)")

    return SceneBundle(
        name=name, dataset_dir=model_dir, device=device,
        gaussians=gauss, test_camera_list=cameras,
        sim_mask=sim_mask, sim_xyzs=sim_xyzs, points_vol=points_vol,
        top_k_index=top_k_index, freeze_mask=freeze_mask, sim_aabb=sim_aabb,
        scale=scale, shift=shift, resolution=[H, W],
    )
