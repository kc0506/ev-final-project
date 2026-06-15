"""Probe the screen-flow QUALITY envelope vs initial speed vx, to pick the dataset
magnitude band. For each vx: roll the MPM (no 3DGS), project to screen, and report
per-frame flow magnitude (px), the fraction of moving particles still on-screen, and
the MPM wall-clamp count. Also writes a flow gif per vx for visual inspection.

Needs nothing from meow2: reuses the local telephone scene_cache + camera.

  micromamba run -n physdreamer python -m vsd.probe_magnitude --vx 2 4 6 8
"""
import vsd.bootstrap  # noqa: F401

import argparse
import json
import os
from typing import List

import imageio.v2 as imageio
import numpy as np
import torch

from vsd.flow_render import project_to_res, render_flow
from vsd.scene_min import load_camera, load_min_scene
from vsd.traj import V0Trajectory

DATA = "outputs/dataset_gen/01_tel_axisx_rest_T16"
RES = 128


def flow_to_rgb(f2: np.ndarray) -> np.ndarray:
    """packed flow [2,H,W] in [0,1] -> [H,W,3] viz (B=0.5)."""
    f2 = np.transpose(f2, (1, 2, 0))
    return np.concatenate([f2, np.full(f2.shape[:2] + (1,), 0.5, np.float32)], -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, nargs="+", default=[2.0, 4.0, 6.0, 8.0])
    ap.add_argument("--out", default="vsd/out/mag_probe")
    args = ap.parse_args()
    dev = "cuda:0"
    os.makedirs(args.out, exist_ok=True)

    scene = load_min_scene(os.path.join(DATA, "scene_cache.pt"), device=dev)
    cam = load_camera(os.path.join(DATA, "camera.json"), device=dev)
    scale_px = float(json.load(open(os.path.join(DATA, "flow_pack_128_t8.npy.meta.json")))["scale_px"])
    builder = V0Trajectory(scene, E=1e5, n_flow=7, device=dev)

    print(f"scale_px(existing pack, |vx|<=2) = {scale_px:.2f}px  | frame width = {RES}px\n")
    print("  vx | flow px/frame mean(95p)         | on-screen frac per frame (f1..f7)      | wall")
    rows: List[dict] = []
    for vx in args.vx:
        builder.roll.wall_contact_frames = 0
        builder.roll._wall_warned = False
        with torch.no_grad():
            v0 = torch.tensor([vx, 0.0, 0.0], device=dev, dtype=torch.float32)
            world = builder.world_traj(v0, grad_window=1)            # [8,n_move,3]
            uv, zv = project_to_res(world, cam, RES)                  # [8,n_move,2], [8,n_move]
            flow = render_flow(world, cam, scale_px, RES).cpu().numpy()  # [7,2,128,128]
        disp = (uv[1:] - uv[:-1]).norm(dim=-1)                        # [7,n_move] px/frame
        inb = ((uv[..., 0] >= 0) & (uv[..., 0] < RES) &
               (uv[..., 1] >= 0) & (uv[..., 1] < RES) & (zv > 0))     # [8,n_move]
        onscreen = inb[1:].float().mean(dim=1).cpu().numpy()          # [7] per-flow-frame
        rec = {"vx": vx, "flow_mean_px": float(disp.mean()),
               "flow_95p_px": float(torch.quantile(disp, 0.95)),
               "onscreen_last": float(onscreen[-1]),
               "onscreen_min": float(onscreen.min()),
               "wall_frames": int(builder.roll.wall_contact_frames)}
        rows.append(rec)
        os_str = " ".join(f"{x:.2f}" for x in onscreen)
        print(f"{vx:5.1f} | {rec['flow_mean_px']:6.1f} ({rec['flow_95p_px']:6.1f})            "
              f"| {os_str} | {rec['wall_frames']}")
        frames = [(flow_to_rgb(flow[t]) * 255).round().astype("uint8") for t in range(flow.shape[0])]
        imageio.mimsave(os.path.join(args.out, f"flow_vx{vx:g}.gif"), frames, fps=3)

    json.dump(rows, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    print(f"\nsaved flow gifs + summary.json under {args.out}/")
    print("read: flow_95p_px approaching/exceeding ~frame-width and on-screen frac dropping "
          "=> object leaving the monocular frame; pick mag_max below that.")


if __name__ == "__main__":
    main()
