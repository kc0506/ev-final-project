"""Validate the differentiable flow renderer against the stored flow_pack the teacher
trained on. Renders the GT trajectory (stored mpm_xyz, so this isolates the RENDERER
from the rollout) and compares to flow_pack[sample]. Writes a side-by-side gif.

  micromamba run -n physdreamer python -m vsd.check_render
"""
import vsd.bootstrap  # noqa: F401

import json
import os

import imageio.v2 as imageio
import numpy as np
import torch

from vsd.flow_render import render_flow
from vsd.scene_min import load_camera, load_min_scene

DATA = "outputs/dataset_gen/01_tel_axisx_rest_T16"
SAMPLE_IDX = 0
RES = 128
NF_RGB = 8  # first 8 RGB frames -> 7 flow fields


def flow_to_rgb(f2: np.ndarray) -> np.ndarray:
    """packed flow [2,H,W] in [0,1] -> [H,W,3] viz (B=0.5), same as train_flow.py."""
    f2 = np.transpose(f2, (1, 2, 0))                                  # [H,W,2]
    return np.concatenate([f2, np.full(f2.shape[:2] + (1,), 0.5, np.float32)], -1)


def main() -> None:
    dev = "cuda:0"
    scene = load_min_scene(os.path.join(DATA, "scene_cache.pt"), device=dev)
    cam = load_camera(os.path.join(DATA, "camera.json"), device=dev)
    meta = json.load(open(os.path.join(DATA, "flow_pack_128_t8.npy.meta.json")))
    scale_px = float(meta["scale_px"])

    move = scene.query_mask                                           # [n] bool moving
    xyz = np.load(os.path.join(DATA, f"sample_{SAMPLE_IDX:04d}", "mpm_xyz.npy"))  # [16,n,3] world
    world = torch.from_numpy(xyz[:NF_RGB]).float().to(dev)[:, move, :]            # [8,n_move,3]

    with torch.no_grad():
        mine = render_flow(world, cam, scale_px, RES)                # [7,2,128,128] in [0,1]
    mine_np = mine.cpu().numpy()

    pack = np.load(os.path.join(DATA, "flow_pack_128_t8.npy"))       # [256,7,128,128,2]
    gt = np.transpose(pack[SAMPLE_IDX], (0, 3, 1, 2))               # [7,2,128,128]

    # numeric comparison in packed [0,1] units and in pixel-displacement units
    d = np.abs(mine_np - gt)
    disp_scale = 2 * scale_px
    print(f"packed L1 mean={d.mean():.5f} max={d.max():.5f}  (flow magnitude in pack: "
          f"mean|gt-0.5|={np.abs(gt-0.5).mean():.5f})")
    print(f"in px units: mean abs err={d.mean()*disp_scale:.4f}px  "
          f"(field motion ~{np.abs(gt-0.5).mean()*disp_scale:.4f}px)")

    # side-by-side animated gif (mine | stored) across the 7 flow fields
    outdir = os.path.join("vsd", "out"); os.makedirs(outdir, exist_ok=True)
    frames = []
    for t in range(gt.shape[0]):
        left = flow_to_rgb(mine_np[t]); right = flow_to_rgb(gt[t])
        bar = np.ones((RES, 2, 3), np.float32)
        row = np.concatenate([left, bar, right], axis=1)
        frames.append((row * 255).round().astype("uint8"))
    gif = os.path.join(outdir, f"render_vs_pack_s{SAMPLE_IDX:04d}.gif")
    imageio.mimsave(gif, frames, fps=3)
    print(f"saved {gif}   (LEFT = my differentiable render | RIGHT = stored flow_pack)")


if __name__ == "__main__":
    main()
