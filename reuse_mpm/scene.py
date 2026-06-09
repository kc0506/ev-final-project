"""Load a PhysDreamer-format scene into an E/v0-independent bundle.

Everything here mirrors the proven setup in
`PhysDreamer/scripts/render_trained_sim.py` (lines ~84-164) but stops *before*
any material (E/nu) or initial-velocity choice, so the same bundle can be reused
to generate many videos with different E.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from . import _env
from ._env import (
    GaussianModel,
    Camera,
    MultiviewImageDataset,
    get_volume,
    find_far_points,
    downsample_with_kmeans_gpu_with_chunk,
)


@dataclass
class SceneBundle:
    """Static, material-independent simulation context for one scene."""

    name: str
    dataset_dir: str
    device: str

    gaussians: GaussianModel          # full scene, for rendering
    test_camera_list: List[Camera]    # all cameras
    sim_mask: torch.Tensor            # [N_gauss] bool: which gaussians are simulated
    sim_xyzs: torch.Tensor            # [n, 3] normalised MPM particle positions (post-kmeans)
    points_vol: np.ndarray            # [n] per-particle volume
    top_k_index: torch.Tensor         # [n_sim_gauss, top_k] KNN: gauss -> sim particle
    freeze_mask: torch.Tensor         # [n] bool: frozen (non-moving) particles
    sim_aabb: torch.Tensor            # [2, 3] padded aabb of sim particles
    scale: float
    shift: torch.Tensor
    resolution: List[int] = field(default_factory=lambda: [576, 1024])

    @property
    def query_mask(self) -> torch.Tensor:
        """Particles that are free to move (not frozen)."""
        return torch.logical_not(self.freeze_mask)

    def camera_by_frame(self, frame_filename: str) -> Camera:
        for c in self.test_camera_list:
            if os.path.basename(c.img_path) == frame_filename:
                return c
        raise ValueError(f"camera {frame_filename} not found in {self.dataset_dir}")


_CACHE_KEYS = [
    "sim_mask", "sim_xyzs", "points_vol", "top_k_index",
    "freeze_mask", "sim_aabb", "scale", "shift",
]

_CACHE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "_scene_cache",
)


def default_cache_path(dataset_dir, downsample_scale, grid_size, top_k=8):
    """A stable cache path per (scene, discretisation params).

    Both forward-gen and inverse derive the same path -> identical particles.
    """
    scene = os.path.basename(os.path.normpath(dataset_dir))
    fname = f"{scene}_ds{downsample_scale}_g{grid_size}_k{top_k}.pt"
    return os.path.join(_CACHE_ROOT, fname)


def load_scene(
    dataset_dir: str,
    name: Optional[str] = None,
    device: str = "cuda:0",
    downsample_scale: float = 0.1,
    grid_size: int = 32,
    top_k: int = 8,
    max_particles: int = 8000,
    resolution: Optional[List[int]] = None,
    cache_path: Optional[str] = None,
) -> SceneBundle:
    """Build a SceneBundle from a PhysDreamer-format scene directory.

    Mirrors render_trained_sim.py's setup verbatim so coordinate conventions
    (scale/shift, KNN, freeze BC) match the inverse pipeline by construction.

    CRITICAL: the KNN/k-means particle downsample is non-deterministic across
    processes. For a valid roundtrip, forward-gen and inverse MUST use the SAME
    particle discretisation. Pass `cache_path`: the first call computes and saves
    the discretisation; later calls load it verbatim (gaussians/cameras are
    always re-read fresh, they are deterministic).
    """

    if name is None:
        name = os.path.basename(os.path.normpath(dataset_dir))
    if resolution is None:
        resolution = [576, 1024]

    # cameras only (no image tensors)
    dataset = MultiviewImageDataset(
        dataset_dir,
        use_white_background=False,
        resolution=resolution,
        scale_x_angle=1.0,
        load_imgs=False,
    )

    # full-scene gaussians
    gauss_path = os.path.join(dataset_dir, "point_cloud.ply")
    gauss = GaussianModel(3)
    gauss.load_ply(gauss_path)
    gauss.detach_grad()

    # ---- particle discretisation (NON-deterministic via k-means): cache it ----
    if cache_path is not None and os.path.exists(cache_path):
        blob = torch.load(cache_path, map_location=device)
        assert blob["downsample_scale"] == downsample_scale and \
            blob["grid_size"] == grid_size and blob["top_k"] == top_k, \
            f"cache {cache_path} built with different params: {blob.get('downsample_scale')}, " \
            f"{blob.get('grid_size')}, {blob.get('top_k')}"
        d = blob["disc"]
        sim_mask = d["sim_mask"].to(device)
        sim_xyzs = d["sim_xyzs"].to(device)
        points_vol = d["points_vol"]  # numpy
        top_k_index = d["top_k_index"].to(device)
        freeze_mask = d["freeze_mask"].to(device)
        sim_aabb = d["sim_aabb"].to(device)
        scale = float(d["scale"]); shift = d["shift"].to(device)
        print(f"[scene] loaded cached discretisation from {cache_path} "
              f"({sim_xyzs.shape[0]} particles)")
    else:
        import point_cloud_utils as pcu

        # reference point sets can be huge (hat: ~280k); find_far_points does an
        # [chunk x M] cdist, so cap M to keep memory bounded (the threshold test
        # only needs a dense-enough reference cloud). Seeded for reproducibility.
        def _subsample(pts: torch.Tensor, cap: int = 20000) -> torch.Tensor:
            if pts.shape[0] <= cap:
                return pts
            g = torch.Generator(device="cpu").manual_seed(0)
            idx = torch.randperm(pts.shape[0], generator=g)[:cap]
            return pts[idx]

        clean_points_path = os.path.join(dataset_dir, "clean_object_points.ply")
        clean_xyzs = (
            torch.from_numpy(pcu.load_mesh_v(clean_points_path)).float().to(device)
        )
        clean_xyzs = _subsample(clean_xyzs)
        not_sim_mask = find_far_points(gauss._xyz, clean_xyzs, thres=0.01).bool()
        sim_mask = torch.logical_not(not_sim_mask)
        sim_xyzs = gauss._xyz[sim_mask, :].detach().clone()

        pos_max = sim_xyzs.max()
        pos_min = sim_xyzs.min()
        scale = (pos_max - pos_min) * 1.8
        shift = -pos_min + (pos_max - pos_min) * 0.25

        filled = os.path.join(dataset_dir, "internal_filled_points.ply")
        if os.path.exists(filled):
            fill = torch.from_numpy(pcu.load_mesh_v(filled)).float().to(device)
            sim_xyzs = torch.cat([sim_xyzs, fill], dim=0)

        # [notes] 3dgs to mpm coords
        sim_xyzs = (sim_xyzs + shift) / scale
        sim_aabb = torch.stack([sim_xyzs.min(dim=0)[0], sim_xyzs.max(dim=0)[0]], dim=0)
        sim_aabb = (
            sim_aabb - sim_aabb.mean(dim=0, keepdim=True)
        ) * 1.2 + sim_aabb.mean(dim=0, keepdim=True)

        # MPM particle count: relative downsample, but capped so large objects
        # (e.g. alocasia ~217k gaussians) don't blow up k-means memory or sim cost.
        # chunked k-means keeps the distance matrix bounded too.

        # for tele: 7w -> 7k
        num_cluster = min(int(sim_xyzs.shape[0] * downsample_scale), max_particles)
        sim_xyzs = downsample_with_kmeans_gpu_with_chunk(sim_xyzs, num_cluster)

        sim_gauss_pos = (gauss._xyz[sim_mask, :].detach().clone() + shift) / scale
        cdist = torch.cdist(sim_gauss_pos, sim_xyzs) * -1.0
        _, top_k_index = torch.topk(cdist, top_k, dim=-1)

        # sim_xyzs:      [m, 3]
        # sim_gauss_pos: [n, 3]  (n > m = n * downscale)
        # top_k_index:   [n, k]

        points_vol = get_volume(sim_xyzs.detach().cpu().numpy())

        moving = os.path.join(dataset_dir, "moving_part_points.ply")
        mvp = torch.from_numpy(pcu.load_mesh_v(moving)).float().to(device)
        mvp = _subsample(mvp)
        mvp = (mvp + shift) / scale
        freeze_mask = find_far_points(sim_xyzs, mvp, thres=0.5 / grid_size).bool()
        scale = float(scale)

         # freeze_mask: [m,]

        if cache_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            torch.save({
                "downsample_scale": downsample_scale, "grid_size": grid_size,
                "top_k": top_k, "dataset_dir": dataset_dir,
                "disc": {
                    "sim_mask": sim_mask.cpu(), "sim_xyzs": sim_xyzs.cpu(),
                    "points_vol": points_vol, "top_k_index": top_k_index.cpu(),
                    "freeze_mask": freeze_mask.cpu(), "sim_aabb": sim_aabb.cpu(),
                    "scale": scale, "shift": shift.cpu(),
                },
            }, cache_path)
            print(f"[scene] saved discretisation cache -> {cache_path} "
                  f"({sim_xyzs.shape[0]} particles)")

    return SceneBundle(
        name=name,
        dataset_dir=dataset_dir,
        device=device,
        gaussians=gauss,
        test_camera_list=dataset.test_camera_list,
        sim_mask=sim_mask,
        sim_xyzs=sim_xyzs,
        points_vol=points_vol,
        top_k_index=top_k_index,
        freeze_mask=freeze_mask,
        sim_aabb=sim_aabb,
        scale=float(scale),
        shift=shift,
        resolution=resolution,
    )
