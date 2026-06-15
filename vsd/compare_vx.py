"""Side-by-side flow comparison across initial speeds vx, in ONE gif, with a COMMON
colour scale so magnitude differences are visible. A `zoom` (<1) keeps even large-vx
motion in-frame (equivalent to dollying the camera back), so the comparison reflects
the flow PATTERN/MAGNITUDE difference, not the object leaving the monocular frame.

  micromamba run -n physdreamer python -m vsd.compare_vx --vx 2 4 6 8 --zoom 0.45
"""
import vsd.bootstrap  # noqa: F401

import argparse
import os
from typing import List

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from vsd.flow_render import fill_holes, project_to_res, soft_splat
from vsd.scene_min import load_camera, load_min_scene
from vsd.traj import V0Trajectory

DATA = "outputs/dataset_gen/01_tel_axisx_rest_T16"
RES = 128


def raw_flow_px(world: torch.Tensor, cam, zoom: float) -> torch.Tensor:
    """world [8,n,3] -> per-frame screen flow in PIXELS [7,2,RES,RES] (soft-splat + fill)."""
    uv, zv = project_to_res(world, cam, RES, zoom=zoom)                  # [8,n,2],[8,n]
    fields: List[torch.Tensor] = []
    for t in range(world.shape[0] - 1):
        disp = uv[t + 1] - uv[t]                                         # [n,2] px
        inb = ((uv[t, :, 0] >= 0) & (uv[t, :, 0] < RES) &
               (uv[t, :, 1] >= 0) & (uv[t, :, 1] < RES) &
               (zv[t] > 0) & (zv[t + 1] > 0))                           # [n]
        flow, covered = soft_splat(uv[t], disp, inb, RES)               # [2,RES,RES]
        fields.append(fill_holes(flow, covered, iters=8))
    return torch.stack(fields, 0)                                       # [7,2,RES,RES]


def flow_to_rgb_packed(flow_px: np.ndarray, scale: float) -> np.ndarray:
    """flow_px [2,H,W] (pixels) -> [H,W,3] viz packed with a COMMON scale (B=0.5)."""
    packed = np.clip(flow_px / (2 * scale) + 0.5, 0, 1)                  # [2,H,W]
    packed = np.transpose(packed, (1, 2, 0))                            # [H,W,2]
    return np.concatenate([packed, np.full(packed.shape[:2] + (1,), 0.5, np.float32)], -1)


def label(img_u8: np.ndarray, text: str) -> np.ndarray:
    """Draw a small text label top-left on an [H,W,3] uint8 image."""
    im = Image.fromarray(img_u8)
    ImageDraw.Draw(im).text((3, 2), text, fill=(255, 255, 255))
    return np.asarray(im)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, nargs="+", default=[2.0, 4.0, 6.0, 8.0])
    ap.add_argument("--zoom", type=float, default=0.45, help="<1 shrinks object to stay in-frame")
    ap.add_argument("--out", default="vsd/out/mag_probe/compare_vx.gif")
    args = ap.parse_args()
    dev = "cuda:0"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    scene = load_min_scene(os.path.join(DATA, "scene_cache.pt"), device=dev)
    cam = load_camera(os.path.join(DATA, "camera.json"), device=dev)
    builder = V0Trajectory(scene, E=1e5, n_flow=7, device=dev)

    flows: List[np.ndarray] = []
    for vx in args.vx:
        with torch.no_grad():
            world = builder.world_traj(torch.tensor([vx, 0.0, 0.0], device=dev), grad_window=1)
            flows.append(raw_flow_px(world, cam, args.zoom).cpu().numpy())   # [7,2,RES,RES] px
    common = float(np.percentile(np.abs(np.stack(flows)), 99.0))             # shared colour scale
    print(f"zoom={args.zoom}  common colour scale (99p |flow|) = {common:.2f}px")

    frames = []
    for t in range(flows[0].shape[0]):
        panels = []
        for vx, fl in zip(args.vx, flows):
            rgb = (flow_to_rgb_packed(fl[t], common) * 255).round().astype("uint8")
            panels.append(label(rgb, f"vx={vx:g}"))
            panels.append(np.full((RES, 2, 3), 255, np.uint8))               # white sep
        frames.append(np.concatenate(panels[:-1], axis=1))
    imageio.mimsave(args.out, frames, fps=2)
    print(f"saved {args.out}  ({len(args.vx)} panels x {flows[0].shape[0]} frames, common scale)")


if __name__ == "__main__":
    main()
