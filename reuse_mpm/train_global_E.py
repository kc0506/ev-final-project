"""Entrypoint: recover a single global Young's modulus E from a generated video.

Roundtrip inverse problem (v1 of the goal): Y = one global scalar E. Consumes a
forward_gen run dir -- its frames are the GT, its config.json carries the resolved
ForwardConfig (scene + sim + v0 + frame + true E) -- then optimises log E by
differentiable MPM + photometric loss and reports how well E* is recovered. The
inverse therefore uses the EXACT setup that produced the GT (no re-specification).

  python -m reuse_mpm.train_global_E \
      --gt_run outputs/fwd_telephone_E1e5 \
      --init_E 3e5 --iters 60 --window 6 --grad_window 6 \
      --out outputs/inv_telephone_E1e5

The loss basin around E* is narrow (see probe_identifiability): use --coarse_init
to grid-search into the basin first, then gradient-refine.
"""
from __future__ import annotations

import glob
import json
import os
import time

import numpy as np
import tyro

from .config import RecoverConfig, SceneSpec, SimConfig
from .gpu import pick_free_gpu


def _load_gt_frames(gt_run: str, device: str):
    frame_files = sorted(glob.glob(os.path.join(gt_run, "frames", "frame_*.png")))
    assert frame_files, f"no frames in {gt_run}/frames"
    import imageio.v2 as imageio
    import torch

    frames = [imageio.imread(fp) for fp in frame_files]
    arr = np.stack(frames, 0).astype(np.float32) / 255.0  # [T,H,W,C]
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device)  # [T,C,H,W]


def run(cfg: RecoverConfig):
    pick_free_gpu()
    from .scene_io import load_from_spec
    from .sim_render import make_constant_v0, simulate_and_render, video_to_uint8
    from .run_io import RecoverRun
    from .recover import recover_global_E, plot_recovery

    t0 = time.time()
    label = cfg.run_label or (
        f"from-{os.path.basename(os.path.normpath(cfg.gt_run))}_init{cfg.init_E:g}")
    rd = RecoverRun.create(__name__, label, cfg.out, config=cfg)  # auto-saves RecoverConfig

    with open(os.path.join(cfg.gt_run, "config.json")) as f:
        g = json.load(f)
    if isinstance(g.get("scene"), dict):  # current schema: resolved ForwardConfig
        scene_spec = SceneSpec(**g["scene"])
    else:  # legacy schema: top-level dataset_dir / scene_type / scene_cache
        scene_spec = SceneSpec(
            path=g["dataset_dir"], kind=g.get("scene_type", "pd"),
            downsample_scale=g.get("downsample_scale", 0.1),
            cache_path=g.get("scene_cache"))
    sim = SimConfig(**g["sim"])  # grid_lim absent in legacy -> defaults
    true_E = float(g["E"])
    v0_vec = g["v0"]
    frame = g["frame"]
    device = scene_spec.device

    # The discretisation cache path is deterministic from (scene, downsample,
    # grid, top_k), all read from the GT config -- so load_from_spec re-derives
    # the SAME path the GT run created and loads the identical particles. (The
    # auto-saved GT config may carry cache_path=None when it was the default.)
    gt = _load_gt_frames(cfg.gt_run, device)
    scene = load_from_spec(scene_spec, sim)
    if scene_spec.kind == "pd":
        rd.link_source_ply(scene_spec.path)
    try:
        cam = scene.camera_by_frame(frame)
    except Exception:
        cam = scene.test_camera_list[0]
    v0 = make_constant_v0(scene, v0_vec).detach()

    rd.gt_video(video_to_uint8(gt), fps=sim.fps)

    # the single shared recovery routine (see recover.py)
    res = recover_global_E(
        scene, gt, sim, cam, v0, init_E=cfg.init_E, iters=cfg.iters, lr=cfg.lr,
        window=cfg.window, grad_window=cfg.grad_window,
        coarse_init=cfg.coarse_init, coarse_n=cfg.coarse_n,
        true_E=true_E, device=device)

    # keep videos: init guess, recovered, GT|recovered montage
    gt_u8 = video_to_uint8(gt)
    pred_init = video_to_uint8(
        simulate_and_render(scene, float(res["init_E"]), v0, sim, cam).detach())
    pred_final = video_to_uint8(
        simulate_and_render(scene, res["recovered_E"], v0, sim, cam).detach())
    rd.pred_videos(pred_init, pred_final, gt_u8, fps=sim.fps)

    # the scene/sim were RECONSTRUCTED from the GT run (not in RecoverConfig), so
    # merge them into the auto-saved config as the "special" extras.
    rd.merge_config(
        scene=scene_spec.to_dict(), sim=sim.to_dict(),
        scene_name=scene.name, true_E=true_E, v0=v0_vec,
        n_mpm_particles=int(scene.sim_xyzs.shape[0]),
    )
    rd.metrics(
        true_E=true_E, recovered_E=res["recovered_E"],
        final_iter_E=res["final_iter_E"], rel_err=res["rel_err"],
        log10_err=res["log10_err"], final_loss=res["final_loss"],
        coarse=res["coarse"], elapsed_sec=round(time.time() - t0, 2),
    )
    rd.trace(res["E_traj"], res["loss_traj"])
    plot_recovery(rd.path("recovery.png"), res, true_E,
                  title=f"{scene_spec.kind}:{scene.name}")
    rd.finish()  # seals recovery.png (written via savefig, not a RunDir method)

    print(f"[train] {scene_spec.kind}:{scene.name} true={true_E:.3e} "
          f"recovered={res['recovered_E']:.3e} rel_err={res['rel_err']*100:.0f}% "
          f"final_loss={res['final_loss']:.2e} ({time.time()-t0:.0f}s) -> {rd.root}")
    return rd


if __name__ == "__main__":
    run(tyro.cli(RecoverConfig))
