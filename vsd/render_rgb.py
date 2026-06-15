"""3DGS RGB render of an MPM rollout, for visual quality inspection (richer than flow).
Uses the canonical pipeline (load_from_spec -> simulate_positions -> render_positions)
on the v2 centred cache. Needs the real diff_gaussian_rasterization built.

  micromamba run -n physdreamer python -m vsd.render_rgb --vx -4 --frames 8
"""
import vsd.bootstrap  # noqa: F401

import argparse
import os

import imageio.v2 as imageio
import numpy as np
import torch

V2_CACHE = "vsd/out/scene_cache_v2_telephone.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, default=-4.0)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--substep", type=int, default=64)
    ap.add_argument("--frame", default="frame_00001.png")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = "cuda:0"

    from reuse_mpm.config import SceneSpec, SimConfig
    from reuse_mpm.scene_io import load_from_spec
    from reuse_mpm.sim_render import make_constant_v0, simulate_positions, render_positions

    spec = SceneSpec(path="data-pd/physics_dreamer/telephone", kind="pd", cache_path=V2_CACHE)
    sim = SimConfig(num_frames=args.frames, substep=args.substep)
    scene = load_from_spec(spec, sim)                                   # full SceneBundle (gaussians)
    cam = scene.camera_by_frame(args.frame)

    v0 = make_constant_v0(scene, torch.tensor([args.vx, 0.0, 0.0], device=scene.device))  # [n,3]
    pos_list = simulate_positions(scene, 1e5, v0, sim)                  # list[T] [n,3] world
    vid = render_positions(scene, pos_list, cam)                        # [T,C,H,W] in [0,1]
    vid_u8 = (vid.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()  # [T,H,W,C]

    out = args.out or f"vsd/out/fix/rgb_vx{args.vx:g}_v2.gif"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.mimsave(out, [vid_u8[t] for t in range(vid_u8.shape[0])], fps=3)
    print(f"saved {out}  shape={tuple(vid_u8.shape)}")


if __name__ == "__main__":
    main()
