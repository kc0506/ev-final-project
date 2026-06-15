"""Precompute the flow(vx) table LOCALLY (needs warp + reuse_mpm) and save it as a
plain .npy + meta json. This is the reusable bridge to Modal: the cloud ODE probe only
needs torch + video_diffusion (validated by modal_smoke), so we precompute the physics
here and upload the table -- no warp/reuse_mpm on Modal required.

  micromamba run -n physdreamer python -m vsd.build_flow_table \
      --data outputs/gen_flow_aligned/04_n128_axisx_mag0-8_rot67.6 \
      --vmin 0 --vmax 8 --grid 121 --out vsd/out/flow_grid_out04
"""
import vsd.bootstrap  # noqa: F401

import argparse
import json
import os

import numpy as np
import torch

from vsd.flow_render import render_flow
from vsd.scene_min import apply_scene_fixes, load_camera, load_min_scene
from vsd.traj import V0Trajectory

RES = 128
ROT = 67.6


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/gen_flow_aligned/04_n128_axisx_mag0-8_rot67.6")
    ap.add_argument("--vmin", type=float, default=0.0)
    ap.add_argument("--vmax", type=float, default=8.0)
    ap.add_argument("--grid", type=int, default=121, help="flow table resolution over [vmin,vmax]")
    ap.add_argument("--out", default="vsd/out/flow_grid_out04")
    args = ap.parse_args()
    dev = "cuda:0"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    scene = load_min_scene(os.path.join(args.data, "scene_cache.pt"), device=dev)
    scene = apply_scene_fixes(scene, rot_z_deg=ROT, recenter=False)
    cam = load_camera(os.path.join(args.data, "camera.json"), device=dev)
    scale_px = float(json.load(open(os.path.join(args.data, "flow_pack_128_t8.npy.meta.json")))["scale_px"])
    builder = V0Trajectory(scene, E=1e5, n_flow=7, device=dev, requires_grad=False)  # table only -> no tape

    vx_grid = np.linspace(args.vmin, args.vmax, args.grid)                # [G]
    grid = []
    for i, vx in enumerate(vx_grid):
        world = builder.world_traj(torch.tensor([float(vx), 0.0, 0.0], device=dev),
                                   grad_window=1, requires_grad=False)    # [8,n_move,3]
        grid.append(render_flow(world, cam, scale_px, RES).cpu())         # [7,2,RES,RES] in [0,1]
        if i % 20 == 0:
            print(f"  {i}/{args.grid}  vx={vx:.2f}", flush=True)
    flow_grid = torch.stack(grid, 0).permute(0, 2, 1, 3, 4).contiguous()  # [G,2,7,RES,RES] (C,F,H,W)

    np.save(args.out + ".npy", flow_grid.numpy().astype(np.float32))      # [G,2,7,128,128]
    json.dump({"vmin": args.vmin, "vmax": args.vmax, "grid": args.grid, "res": RES,
               "scale_px": scale_px, "rot": ROT, "data": args.data,
               "shape": list(flow_grid.shape)},
              open(args.out + ".meta.json", "w"), indent=2)
    print(f"saved {args.out}.npy  shape={list(flow_grid.shape)}  "
          f"({flow_grid.numel()*4/1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
