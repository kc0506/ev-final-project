"""Entrypoint: recover a single global Young's modulus E from a generated video.

Roundtrip inverse problem (v1 of the goal): Y = one global scalar E.
Consumes a forward_gen run dir (its video frames are the GT, its config has the
true E), then optimises log E by differentiable MPM + photometric loss and
reports how well E* is recovered.

  python -m reuse_mpm.train_global_E \
      --gt_run outputs/fwd_telephone_E1e5 \
      --init_E 3e5 --iters 60 --window 6 --grad_window 6 \
      --out outputs/inv_telephone_E1e5

Because the loss basin around E* is narrow (see probe_identifiability), use
--coarse_init to grid-search into the basin first, then gradient-refine.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .gpu import pick_free_gpu


def _load_gt(gt_run: str, device: str):
    cfg_path = os.path.join(gt_run, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    frame_files = sorted(glob.glob(os.path.join(gt_run, "frames", "frame_*.png")))
    assert frame_files, f"no frames in {gt_run}/frames"
    import imageio.v2 as imageio

    frames = [imageio.imread(fp) for fp in frame_files]
    arr = np.stack(frames, 0).astype(np.float32) / 255.0  # [T,H,W,C]
    gt = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device)  # [T,C,H,W]
    return gt, cfg


def build_argparser():
    p = argparse.ArgumentParser(description="recover global E from a video")
    p.add_argument("--gt_run", required=True, help="a forward_gen output dir")
    p.add_argument("--init_E", type=float, default=3e5)
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.1, help="lr on log10(E)")
    p.add_argument("--window", type=int, default=3,
                   help="number of frames (from t=1) used in the loss; keep small "
                        "-- MPM backprop is unstable past ~3 frames")
    p.add_argument("--grad_window", type=int, default=1,
                   help="how many of the latest frames keep BPTT grad (truncated "
                        "BPTT; 1 = only the last frame's substeps carry gradient)")
    p.add_argument("--coarse_init", action="store_true",
                   help="grid-search E first to land in the loss basin")
    p.add_argument("--coarse_n", type=int, default=9)
    p.add_argument("--out", required=True)
    return p


def run(args):
    pick_free_gpu()
    from .sim_render import (
        SimConfig, make_constant_v0, simulate_and_render, video_to_uint8,
    )
    from .run_io import RunDir
    from .recover import recover_global_E, plot_recovery

    device = "cuda:0"
    t0 = time.time()
    rd = RunDir(args.out)

    gt, gcfg = _load_gt(args.gt_run, device)
    dataset_dir = gcfg["dataset_dir"]
    true_E = float(gcfg["E"])
    v0_vec = gcfg["v0"]
    frame = gcfg["frame"]
    scene_type = gcfg.get("scene_type", "pd")
    scfg = gcfg["sim"]
    cfg = SimConfig(num_frames=scfg["num_frames"], substep=scfg["substep"],
                    grid_size=scfg["grid_size"], density=scfg["density"],
                    nu=scfg["nu"], fps=scfg["fps"])

    scene_cache = gcfg.get("scene_cache")
    assert scene_cache and os.path.exists(scene_cache), (
        f"GT run has no usable scene_cache ({scene_cache}); regenerate GT with "
        "the cache-aware forward_gen so particles match.")
    if scene_type == "pg":
        from .scene_physgaussian import load_physgaussian_scene
        scene = load_physgaussian_scene(
            dataset_dir, device=device, downsample_scale=gcfg["downsample_scale"],
            grid_size=cfg.grid_size, cache_path=scene_cache)
    else:
        from .scene import load_scene
        rd.link_source_ply(dataset_dir)
        scene = load_scene(
            dataset_dir, device=device, downsample_scale=gcfg["downsample_scale"],
            grid_size=cfg.grid_size, cache_path=scene_cache)
    try:
        cam = scene.camera_by_frame(frame)
    except Exception:
        cam = scene.test_camera_list[0]
    v0 = make_constant_v0(scene, v0_vec).detach()

    rd.save_named_video("gt", video_to_uint8(gt), fps=cfg.fps)

    # the single shared recovery routine (see recover.py)
    res = recover_global_E(
        scene, gt, cfg, cam, v0, init_E=args.init_E, iters=args.iters, lr=args.lr,
        window=args.window, grad_window=args.grad_window,
        coarse_init=args.coarse_init, coarse_n=args.coarse_n,
        true_E=true_E, device=device)

    # TODO: this sucks. You must have rendered videos in training
    # keep videos: init guess, recovered, GT|recovered montage
    gt_u8 = video_to_uint8(gt)
    pred_init = video_to_uint8(
        simulate_and_render(scene, float(res["init_E"]), v0, cfg, cam).detach())
    pred_final = video_to_uint8(
        simulate_and_render(scene, res["recovered_E"], v0, cfg, cam).detach())
    rd.save_named_video("pred_init", pred_init, fps=cfg.fps)
    rd.save_named_video("pred_recovered", pred_final, fps=cfg.fps)
    T = min(gt_u8.shape[0], pred_final.shape[0])
    rd.save_named_video("gt_vs_recovered",
                        np.concatenate([gt_u8[:T], pred_final[:T]], axis=2), fps=cfg.fps)

    rd.write_config({
        "task": "train_global_E", "gt_run": args.gt_run, "scene_type": scene_type,
        "dataset_dir": dataset_dir, "scene": scene.name,
        "true_E": true_E, "init_E": args.init_E, "v0": v0_vec,
        "iters": args.iters, "lr": args.lr, "window": args.window,
        "grad_window": args.grad_window, "coarse_init": args.coarse_init,
        "sim": cfg.to_dict(), "n_mpm_particles": int(scene.sim_xyzs.shape[0]),
    })
    rd.write_json("metrics.json", {
        "true_E": true_E, "recovered_E": res["recovered_E"],
        "final_iter_E": res["final_iter_E"], "rel_err": res["rel_err"],
        "log10_err": res["log10_err"], "final_loss": res["final_loss"],
        "coarse": res["coarse"], "elapsed_sec": round(time.time() - t0, 2),
    })
    rd.write_json("trace.json", {"E": res["E_traj"], "loss": res["loss_traj"]})
    plot_recovery(rd.path("recovery.png"), res, true_E,
                  title=f"{scene_type}:{scene.name}")

    print(f"[train] {scene_type}:{scene.name} true={true_E:.3e} "
          f"recovered={res['recovered_E']:.3e} rel_err={res['rel_err']*100:.0f}% "
          f"final_loss={res['final_loss']:.2e} ({time.time()-t0:.0f}s) -> {rd.root}")
    return rd


if __name__ == "__main__":
    run(build_argparser().parse_args())
