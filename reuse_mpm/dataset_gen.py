"""Entrypoint: sample Y ~ p*(E) and push forward to a video dataset.

Realises the "sample Y, render X" half of the goal: a known functional
distribution over the physics parameter (here Y = global scalar E,
p*(E) = log-uniform[E_min, E_max]) sampled and pushed through the SAME MPM->3DGS
pipeline to produce a paired (E, video) dataset. This depends only on render
correctness, not on any training result, so it can be produced in parallel.

  python -m reuse_mpm.dataset_gen \
      --dataset_dir .../telephone --E_min 1e4 --E_max 1e6 --n 16 \
      --v0 0 -1 0 --num_frames 8 --substep 32 --seed 0 \
      --out outputs/dataset_telephone_logU_1e4_1e6

Output dir (info-complete):
  manifest.json          p*(E) definition + per-sample (E, paths)
  source_ply             symlink to the ply used
  scene_cache            symlink to the shared particle discretisation
  p_star.png             histogram of sampled E vs the target density
  sample_XXXX/           per sample: video.mp4, video.gif, frames/, video.npy
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from .gpu import pick_free_gpu


def build_argparser():
    p = argparse.ArgumentParser(description="sample E~p* -> (E, video) dataset")
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--E_min", type=float, default=1e4)
    p.add_argument("--E_max", type=float, default=1e6)
    p.add_argument("--n", type=int, default=16, help="number of samples")
    p.add_argument("--v0", type=float, nargs=3, default=[0.0, -1.0, 0.0])
    p.add_argument("--frame", default="frame_00001.png")
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--substep", type=int, default=32)
    p.add_argument("--grid_size", type=int, default=32)
    p.add_argument("--downsample_scale", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=7)
    p.add_argument("--jump_thresh", type=float, default=0.5,
                   help="per-frame max normalised single-step jump above which a "
                        "sample is flagged numerically unstable (CFL blow-up)")
    p.add_argument("--out", required=True)
    return p


def run(args):
    pick_free_gpu()
    from .scene import load_scene, default_cache_path
    from .sim_render import (
        SimConfig, make_constant_v0, simulate_positions, render_positions,
        video_to_uint8,
    )
    from .run_io import RunDir

    device = "cuda:0"
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)

    # p*(E) = log-uniform[E_min, E_max]; sample (seeded for reproducibility)
    rng = np.random.RandomState(args.seed)
    logE = rng.uniform(np.log10(args.E_min), np.log10(args.E_max), size=args.n)
    Es = (10.0 ** logE).astype(np.float64)

    scene_cache = default_cache_path(
        args.dataset_dir, args.downsample_scale, args.grid_size)
    scene = load_scene(args.dataset_dir, device=device,
                       downsample_scale=args.downsample_scale,
                       grid_size=args.grid_size, cache_path=scene_cache)
    cfg = SimConfig(num_frames=args.num_frames, substep=args.substep,
                    grid_size=args.grid_size, fps=args.fps)
    cam = scene.camera_by_frame(args.frame)
    v0 = make_constant_v0(scene, args.v0)

    # top-level symlinks for provenance
    def _link(target, name):
        dst = os.path.join(args.out, name)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(target), dst)
    _link(os.path.join(args.dataset_dir, "point_cloud.ply"), "source_ply")
    _link(scene_cache, "scene_cache")

    samples = []
    n_unstable = 0
    for i, E in enumerate(Es):
        sd = RunDir(os.path.join(args.out, f"sample_{i:04d}"))

        # MPM rollout -> keep per-frame particle positions (the geometric state of
        # truth); KNN/top_k/init/scale/shift live once in the shared scene_cache,
        # GS positions are reconstructable from these + KNN so we don't duplicate.
        pos_list = simulate_positions(scene, float(E), v0, cfg)        # list[T] of [n,3] world
        norm = [(p + scene.shift) / scene.scale for p in pos_list]
        jumps = [float((norm[t] - norm[t - 1]).norm(dim=-1).max())
                 for t in range(1, len(norm))]
        max_jump = max(jumps) if jumps else 0.0
        stable = bool(max_jump < args.jump_thresh)
        n_unstable += (not stable)

        mpm_xyz = torch.stack(pos_list, 0).cpu().numpy()              # [T,n,3] world
        np.save(sd.path("mpm_xyz.npy"), mpm_xyz)

        vid_u8 = video_to_uint8(render_positions(scene, pos_list, cam))
        np.save(sd.path("video.npy"), vid_u8)
        sd.save_video(vid_u8, fps=cfg.fps)

        sd.write_json("sample.json", {
            "id": i, "E": float(E), "log10_E": float(np.log10(E)),
            "max_frame_jump": max_jump, "stable": stable, "frame_jumps": jumps,
        })
        samples.append({"id": i, "E": float(E),
                        "dir": os.path.relpath(sd.root, args.out),
                        "mp4": os.path.relpath(sd.path("video.mp4"), args.out),
                        "max_frame_jump": max_jump, "stable": stable})
        flag = "" if stable else "  [UNSTABLE]"
        print(f"  [{i+1}/{args.n}] E={E:.3e}  max_jump={max_jump:.3f}{flag} -> {sd.root}")

    manifest = {
        "task": "dataset_gen",
        "p_star": {"type": "log_uniform", "E_min": args.E_min, "E_max": args.E_max},
        "dataset_dir": args.dataset_dir, "scene": scene.name,
        "scene_cache": os.path.abspath(scene_cache),
        "v0": args.v0, "frame": args.frame, "seed": args.seed,
        "sim": cfg.to_dict(), "downsample_scale": args.downsample_scale,
        "n_mpm_particles": int(scene.sim_xyzs.shape[0]),
        "n": args.n, "n_unstable": n_unstable,
        "jump_thresh": args.jump_thresh, "samples": samples,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        import json
        json.dump(manifest, f, indent=2, default=str)

    # p*(E) histogram vs target
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(np.log10(Es), bins=min(args.n, 12), density=True,
                alpha=0.6, label="sampled log10 E")
        lo, hi = np.log10(args.E_min), np.log10(args.E_max)
        ax.hlines(1.0 / (hi - lo), lo, hi, color="r", ls="--",
                  label="target log-uniform density")
        ax.set_xlabel("log10 E"); ax.set_ylabel("density")
        ax.set_title(f"p*(E)=logU[{args.E_min:.0e},{args.E_max:.0e}], n={args.n}")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(args.out, "p_star.png"), dpi=120); plt.close(fig)
    except Exception as e:
        print(f"[dataset] plot failed: {e}")

    print(f"[dataset] {args.n} samples, E in [{Es.min():.2e},{Es.max():.2e}] "
          f"-> {args.out}  ({time.time()-t0:.1f}s)")
    return args.out


if __name__ == "__main__":
    run(build_argparser().parse_args())
