"""Render a single-vx flow clip with the gic scene fixes (centred shift + rot_z),
to confirm the wall-clamp bug is gone and the scene is camera-aligned. Reports the
on-screen fraction and writes a flow gif. No 3DGS (screen flow only).

  micromamba run -n physdreamer python -m vsd.render_fixed --vx -4 --rot 67.6 --zoom 0.6
"""
import vsd.bootstrap  # noqa: F401

import argparse
import json
import os

import imageio.v2 as imageio
import numpy as np
import torch

from vsd.flow_render import project_to_res, render_flow
from vsd.scene_min import apply_scene_fixes, load_camera, load_min_scene
from vsd.traj import V0Trajectory

DATA = "outputs/dataset_gen/01_tel_axisx_rest_T16"
RES = 128


def flow_to_rgb(f2: np.ndarray) -> np.ndarray:
    """packed flow [2,H,W] in [0,1] -> [H,W,3] viz (B=0.5)."""
    f2 = np.transpose(f2, (1, 2, 0))
    return np.concatenate([f2, np.full(f2.shape[:2] + (1,), 0.5, np.float32)], -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, default=-4.0)
    ap.add_argument("--rot", type=float, default=67.6, help="rot_z degrees (0 = none)")
    ap.add_argument("--recenter", type=int, default=1)
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--cache", default=os.path.join(DATA, "scene_cache.pt"),
                    help="scene cache (default legacy; pass the v2 centered cache)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = "cuda:0"

    scale_px = float(json.load(open(os.path.join(DATA, "flow_pack_128_t8.npy.meta.json")))["scale_px"])
    base = load_min_scene(args.cache, device=dev)
    scene = apply_scene_fixes(base, rot_z_deg=args.rot, recenter=bool(args.recenter))
    builder = V0Trajectory(scene, E=1e5, n_flow=7, device=dev)

    cam = load_camera(os.path.join(DATA, "camera.json"), device=dev)
    with torch.no_grad():
        world = builder.world_traj(torch.tensor([args.vx, 0.0, 0.0], device=dev), grad_window=1)
        uv, zv = project_to_res(world, cam, RES, zoom=args.zoom)

    inb = ((uv[..., 0] >= 0) & (uv[..., 0] < RES) & (uv[..., 1] >= 0) & (uv[..., 1] < RES) & (zv > 0))
    onscreen = inb[1:].float().mean(dim=1).cpu().numpy()
    print(f"vx={args.vx} rot={args.rot} recenter={args.recenter} zoom={args.zoom} | wall-clamp frames="
          f"{builder.roll.wall_contact_frames} | on-screen f1..f7: {' '.join(f'{x:.2f}' for x in onscreen)}")

    # render with zoom by re-splatting (render_flow has no zoom; do it via a zoomed pack)
    from vsd.flow_render import fill_holes, soft_splat
    fields = []
    for t in range(world.shape[0] - 1):
        disp = uv[t + 1] - uv[t]
        valid = inb[t] & inb[t + 1]
        fl, cov = soft_splat(uv[t], disp, valid, RES)
        fl = fill_holes(fl, cov, iters=8)
        fields.append((fl / (2 * scale_px) + 0.5).clamp(0, 1).cpu().numpy())
    flow = np.stack(fields)

    out = args.out or f"vsd/out/fix/flow_vx{args.vx:g}_rot{args.rot:g}_z{args.zoom:g}.gif"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    frames = [(flow_to_rgb(flow[t]) * 255).round().astype("uint8") for t in range(flow.shape[0])]
    imageio.mimsave(out, frames, fps=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
