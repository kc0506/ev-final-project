"""Entrypoint: probe whether E is recoverable, and over what range.

Generates a GT video at a known E* (and fixed, known v0), then sweeps candidate
E and plots MSE(sim(E), GT) vs E. A clean U-shape with its minimum at E* means
gradient-based recovery will work; a flat region means E is unidentifiable there
(e.g. too stiff -> object barely moves -> all videos look the same).

  python -m reuse_mpm.probe_identifiability \
      --dataset_dir .../telephone --v0 0 -1 0 \
      --E_star 1e5 --E_min 1e3 --E_max 1e7 --n 17 \
      --out outputs/probe_v0_1
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..gpu import pick_free_gpu


def build_argparser():
    p = argparse.ArgumentParser(description="E identifiability / loss landscape")
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--scene_type", choices=["pd", "pg"], default="pd",
                   help="pd=PhysDreamer format, pg=PhysGaussian format")
    p.add_argument("--v0", type=float, nargs=3, default=[0.0, -1.0, 0.0])
    p.add_argument("--E_star", type=float, default=1e5)
    p.add_argument("--E_min", type=float, default=1e3)
    p.add_argument("--E_max", type=float, default=1e7)
    p.add_argument("--n", type=int, default=17)
    p.add_argument("--num_frames", type=int, default=14)
    p.add_argument("--substep", type=int, default=32)
    p.add_argument("--grid_size", type=int, default=32)
    p.add_argument("--downsample_scale", type=float, default=0.1)
    p.add_argument("--frame", default="frame_00001.png")
    p.add_argument("--out", default=None,
                   help="override auto outputs/explore/probe_identifiability/<run>")
    return p


def run(args):
    pick_free_gpu()
    # import after device selection so torch picks the right GPU
    from ..sim_render import (
        SimConfig, make_constant_v0, simulate_and_render, video_to_uint8,
    )
    from ..run_io import RunDir, save_panel_video

    t0 = time.time()
    rd = RunDir.create(__name__, "", args.out)

    if args.scene_type == "pg":
        from ..scene_physgaussian import load_physgaussian_scene, default_pg_cache_path
        scene = load_physgaussian_scene(
            args.dataset_dir, device="cuda:0",
            downsample_scale=args.downsample_scale, grid_size=args.grid_size,
            cache_path=default_pg_cache_path(args.dataset_dir, args.downsample_scale,
                                             args.grid_size))
    else:
        from ..scene import load_scene, default_cache_path
        rd.link_source_ply(args.dataset_dir)
        scene = load_scene(
            args.dataset_dir, device="cuda:0",
            downsample_scale=args.downsample_scale, grid_size=args.grid_size,
            cache_path=default_cache_path(args.dataset_dir, args.downsample_scale,
                                          args.grid_size))
    cfg = SimConfig(num_frames=args.num_frames, substep=args.substep,
                    grid_size=args.grid_size)
    try:
        cam = scene.camera_by_frame(args.frame)
    except Exception:
        cam = scene.test_camera_list[0]  # PG cameras (r_0, ...) won't match frame_*
    v0 = make_constant_v0(scene, args.v0)

    gt = simulate_and_render(scene, args.E_star, v0, cfg, cam).detach()
    # keep the GT video AND every candidate render -- never discard sweep intermediates
    rd.save_named_video("gt_Estar", video_to_uint8(gt), fps=cfg.fps)

    cand = np.geomspace(args.E_min, args.E_max, args.n)
    mses = []
    clips = []
    for i, E in enumerate(cand):
        v = simulate_and_render(scene, float(E), v0, cfg, cam).detach()
        mse = F.mse_loss(v, gt).item()
        mses.append(mse)
        v_u8 = video_to_uint8(v)
        clips.append(v_u8)
        rd.save_named_video(f"candidates/cand_{i:02d}_E{E:.2e}", v_u8, fps=cfg.fps)
        print(f"  E={E:9.2e}  MSE={mse:.6e}")

    # one panel gif comparing all candidate E side by side (E* tile highlighted)
    star_idx = int(np.argmin(np.abs(np.log(cand) - np.log(args.E_star))))
    save_panel_video(
        rd.path("panel.gif"), clips,
        labels=[f"E={E:.1e}" for E in cand], fps=cfg.fps,
        highlight=star_idx, title=f"{scene.name} v0={args.v0} (green=E*)",
    )
    mses = np.array(mses)
    argmin_E = float(cand[int(mses.argmin())])

    # monotonicity on each side of E*
    star_idx = int(np.argmin(np.abs(np.log(cand) - np.log(args.E_star))))
    left = mses[: star_idx + 1]
    right = mses[star_idx:]
    left_mono = bool(np.all(np.diff(left) <= 1e-9))   # decreasing toward E*
    right_mono = bool(np.all(np.diff(right) >= -1e-9))  # increasing away from E*

    rd.write_config({
        "task": "probe_identifiability",
        "dataset_dir": args.dataset_dir, "scene": scene.name,
        "v0": args.v0, "E_star": args.E_star,
        "E_range": [args.E_min, args.E_max], "n": args.n,
        "sim": cfg.to_dict(), "downsample_scale": args.downsample_scale,
        "n_mpm_particles": int(scene.sim_xyzs.shape[0]),
    })
    rd.write_json("landscape.json", {
        "E": cand.tolist(), "mse": mses.tolist(),
        "E_star": args.E_star, "argmin_E": argmin_E,
        "left_monotonic_to_star": left_mono,
        "right_monotonic_from_star": right_mono,
        "elapsed_sec": round(time.time() - t0, 2),
    })

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(cand, mses, "o-")
        ax.axvline(args.E_star, color="r", ls="--", label=f"E* = {args.E_star:.0e}")
        ax.axvline(argmin_E, color="g", ls=":", label=f"argmin = {argmin_E:.1e}")
        ax.set_xscale("log")
        ax.set_xlabel("candidate E (Young's modulus)")
        ax.set_ylabel("MSE( sim(E), GT )")
        ax.set_title(f"{scene.name}  v0={args.v0}  loss landscape")
        ax.legend()
        fig.tight_layout()
        fig.savefig(rd.path("landscape.png"), dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"[probe] plot failed: {e}")

    rd.finish()  # seals landscape.png + panel.gif (savefig/imageio bypass)
    print(f"[probe] argmin E={argmin_E:.2e} (true {args.E_star:.0e})  "
          f"left_mono={left_mono} right_mono={right_mono}  "
          f"-> {rd.path('landscape.png')}  ({time.time()-t0:.1f}s)")
    return rd


if __name__ == "__main__":
    run(build_argparser().parse_args())
