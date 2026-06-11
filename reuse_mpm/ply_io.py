"""Geometry / camera export IO.

Debug-oriented writers that turn simulation state into inspectable artifacts:
per-frame MPM particle plys, per-frame displaced 3DGS splat plys, and a camera
-> dict serialiser for the render view. Kept OUT of sim_render (which only
simulates+renders) and out of run_io (run-dir bookkeeping, dependency-light):
these need scene/gaussian internals + the warp/physdreamer `_env` bridge.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from ._env import Camera, interpolate_points_w_R
from .scene import SceneBundle


def camera_to_dict(cam: Camera) -> dict:
    """Serialise the intrinsics+extrinsics needed to reproduce this view in a
    ply/GS viewer: R, T (PhysDreamer/COLMAP convention), FoV, image size, the
    world-space camera centre, and the full 4x4 world->view matrix."""
    def _np(x):
        return np.asarray(x.detach().cpu().numpy() if torch.is_tensor(x) else x).tolist()
    return {
        "img_path": getattr(cam, "img_path", None),
        "image_height": int(cam.image_height),
        "image_width": int(cam.image_width),
        "FoVx": float(cam.FoVx),
        "FoVy": float(cam.FoVy),
        "R": _np(cam.R),
        "T": _np(cam.T),
        "camera_center_world": _np(cam.camera_center),
        "world_view_transform": _np(cam.world_view_transform),
        "note": "GS convention: row-vector x_view = x_world(homog) @ world_view_transform; "
                "camera_center_world = world_view_transform.inverse()[3,:3].",
    }


def write_points_ply(path: str, xyz: np.ndarray, rgb: np.ndarray = None) -> str:
    """Write a minimal point-cloud ply. Light (~12 B/pt), viewable in
    MeshLab/CloudCompare/Open3D.

    Args:
        xyz: [n, 3] float positions.
        rgb: [n, 3] uint8 colours, or None for geometry-only.
    Returns: the written path.
    """
    from plyfile import PlyData, PlyElement

    xyz = np.ascontiguousarray(xyz, dtype=np.float32)   # [n, 3]
    n = xyz.shape[0]
    if rgb is None:
        dt = [("x", "f4"), ("y", "f4"), ("z", "f4")]
        el = np.empty(n, dtype=dt)
        el["x"], el["y"], el["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    else:
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)  # [n, 3]
        dt = [("x", "f4"), ("y", "f4"), ("z", "f4"),
              ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        el = np.empty(n, dtype=dt)
        el["x"], el["y"], el["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        el["red"], el["green"], el["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    PlyData([PlyElement.describe(el, "vertex")]).write(path)
    return path


def mpm_particles_ply(
    scene: SceneBundle, pos_world: torch.Tensor, path: str,
    keep: torch.Tensor = None,
) -> str:
    """Per-frame MPM particle positions, coloured by role (moving=green, anchor=red).

    Answers 'are the particles physically there / where are the anchors',
    independent of any GS rendering.

    Args:
        pos_world: [n, 3] world-space particle positions for this frame.
        keep:      [n] bool subset to write (e.g. scene.query_mask for moving-only);
                   None -> all particles.
    Returns: the written path.
    """
    xyz = pos_world.detach().cpu().numpy()                       # [n, 3]
    qm = scene.query_mask.detach().cpu().numpy()                 # [n] bool
    rgb = np.where(qm[:, None], np.array([[60, 200, 60]], np.uint8),
                   np.array([[220, 40, 40]], np.uint8))          # [n, 3] uint8
    if keep is not None:
        k = keep.detach().cpu().numpy().astype(bool)             # [n] bool
        xyz, rgb = xyz[k], rgb[k]
    return write_points_ply(path, xyz, rgb)


_GS_ATTRS = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")


def gaussian_frame_ply(
    scene: SceneBundle, pos_world: torch.Tensor, init_pos_world: torch.Tensor, path: str,
    keep_mask: torch.Tensor = None,
) -> str:
    """Per-frame displaced 3DGS splat ply (full SH/opacity/scale/rotation), so a GS
    viewer shows exactly what the renderer draws -- including view-dependent colour.
    The moving (sim_mask) gaussians are advected by the same KNN displacement field
    as the renderer; static gaussians unchanged. Heavy; call only for chosen v0/frames.

    Args:
        pos_world:      [n_mpm, 3] this-frame world particle positions.
        init_pos_world: [n_mpm, 3] rest-pose world particle positions.
        keep_mask:      [N_gauss] bool subset of gaussians to write (e.g.
                        scene.sim_mask for foreground-only); None -> all gaussians.
    Returns: the written path.
    """
    g = scene.gaussians
    sim_mask = scene.sim_mask                                    # [N_gauss] bool
    saved = {a: getattr(g, a).clone() for a in _GS_ATTRS}        # restore-after originals
    disp = (pos_world - init_pos_world).detach()                 # [n_mpm, 3]
    new_xyz, new_rot = interpolate_points_w_R(
        saved["_xyz"][sim_mask], saved["_rotation"][sim_mask],
        init_pos_world.detach(), disp, scene.top_k_index)        # [n_sim,3], [n_sim,4]
    try:
        g._xyz[sim_mask] = new_xyz
        g._rotation[sim_mask] = new_rot
        if keep_mask is not None:                               # write only the subset
            for a in _GS_ATTRS:
                setattr(g, a, getattr(g, a)[keep_mask])
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        g.save_ply(path)
    finally:
        for a in _GS_ATTRS:
            setattr(g, a, saved[a])
    return path
