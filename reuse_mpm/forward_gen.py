"""Entrypoint: generate ONE video from a known constant Young's modulus E.

  python -m reuse_mpm.forward_gen \
      --dataset_dir /tmp2/b10401006/PhysDreamer/data/physics_dreamer/telephone \
      --E 1e6 --v0 0 -0.5 0 --frame frame_00001.png \
      --num_frames 14 --substep 64 --grid_size 32 \
      --out outputs/fwd_telephone_E1e6

Produces a self-contained run dir (config, source ply symlink, frames, mp4/gif).
"""
from __future__ import annotations

import argparse
import os
import time

import torch

from .scene import load_scene, default_cache_path
from .sim_render import (
    SimConfig,
    make_constant_v0,
    simulate_and_render,
    video_to_uint8,
)
from .run_io import RunDir


def build_argparser():
    p = argparse.ArgumentParser(description="known-E -> video")
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--scene_type", choices=["pd", "pg"], default="pd",
                   help="pd=PhysDreamer format, pg=PhysGaussian model dir")
    p.add_argument("--name", default=None)
    p.add_argument("--E", type=float, required=True, help="constant Young's modulus")
    p.add_argument(
        "--v0", type=float, nargs=3, default=[0.0, -0.5, 0.0],
        help="constant initial velocity (normalised space) on moving particles",
    )
    p.add_argument("--frame", default="frame_00001.png", help="camera image filename")
    p.add_argument("--num_frames", type=int, default=14)
    p.add_argument("--substep", type=int, default=64)
    p.add_argument("--grid_size", type=int, default=32)
    p.add_argument("--downsample_scale", type=float, default=0.1)
    p.add_argument("--fps", type=int, default=7)
    p.add_argument("--out", required=True, help="run directory")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--scene_cache", default=None,
                   help="path to shared scene discretisation cache "
                        "(default: derived from dataset+params)")
    return p


def run(args):
    t0 = time.time()
    rd = RunDir(args.out)
    rd.link_source_ply(args.dataset_dir)

    if args.scene_type == "pg":
        from .scene_physgaussian import load_physgaussian_scene, default_pg_cache_path
        scene_cache = args.scene_cache or default_pg_cache_path(
            args.dataset_dir, args.downsample_scale, args.grid_size)
        scene = load_physgaussian_scene(
            args.dataset_dir, name=args.name, device=args.device,
            downsample_scale=args.downsample_scale, grid_size=args.grid_size,
            cache_path=scene_cache)
    else:
        scene_cache = args.scene_cache or default_cache_path(
            args.dataset_dir, args.downsample_scale, args.grid_size)
        scene = load_scene(
            args.dataset_dir, name=args.name, device=args.device,
            downsample_scale=args.downsample_scale, grid_size=args.grid_size,
            cache_path=scene_cache)
    cfg = SimConfig(
        num_frames=args.num_frames,
        substep=args.substep,
        grid_size=args.grid_size,
        fps=args.fps,
    )
    try:
        cam = scene.camera_by_frame(args.frame)
    except Exception:
        cam = scene.test_camera_list[0]  # PG cameras (r_0, ...) won't match frame_*
    v0 = make_constant_v0(scene, args.v0)

    rd.write_config({
        "task": "forward_gen",
        "scene_type": args.scene_type,
        "dataset_dir": args.dataset_dir,
        "scene": scene.name,
        "E": args.E,
        "v0": args.v0,
        "frame": args.frame,
        "sim": cfg.to_dict(),
        "downsample_scale": args.downsample_scale,
        "scene_cache": os.path.abspath(scene_cache),
        "n_mpm_particles": int(scene.sim_xyzs.shape[0]),
        "n_sim_gaussians": int(scene.sim_mask.sum().item()),
    })

    vid = simulate_and_render(scene, args.E, v0, cfg, cam, requires_grad=False)
    vid_u8 = video_to_uint8(vid)
    mp4, gif = rd.save_video(vid_u8, fps=cfg.fps)

    rd.write_json("result.json", {
        "video_shape": list(vid_u8.shape),
        "elapsed_sec": round(time.time() - t0, 2),
        "mp4": mp4,
        "gif": gif,
    })
    print(f"[forward_gen] E={args.E:g} -> {mp4}  shape={vid_u8.shape}  "
          f"({time.time()-t0:.1f}s, {scene.sim_xyzs.shape[0]} particles)")
    return rd


if __name__ == "__main__":
    run(build_argparser().parse_args())
