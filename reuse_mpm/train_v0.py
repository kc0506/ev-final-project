"""Entrypoint: recover an initial-velocity FIELD v0 from a video, with E KNOWN.

The dual of train_global_E / train_field_E. Those assumed v0 was a known constant
and fit the stiffness E; here E is read from the GT run and held fixed, and a
`V0Field` (global | voxel | triplane) is optimised instead. Same roundtrip contract
-- consumes a forward_gen run dir (its frames = GT, its config.json carries
scene+sim+E+v0+frame) and uses the EXACT setup that produced the GT.

The MPM here has gravity off (motion is driven entirely by v0), so v0 is highly
identifiable and the field is zero-init (== start at rest). v0 only gets a gradient
on FULL-BPTT frames, so recover_v0 rolls a short `window` of frames to t=0.

  python -m reuse_mpm.train_v0 \
      --gt_run outputs/forward_gen/06_tele_E1e5 \
      --kind triplane --iters 120 --window 2 --lr 0.05

When the GT v0 was a constant (phase A), the "right" field is ~uniform == that
vector: the metric is whether the field's moving-particle mean v0 recovers it AND
the photometric loss converges.
"""
from __future__ import annotations

import glob
import json
import os
import time

import numpy as np
import tyro

from .config import RecoverV0Config, SceneSpec, SimConfig
from .gpu import pick_free_gpu


def _load_gt_frames(gt_run: str, device: str):
    """gt_run/frames/*.png -> [T,C,H,W] float in [0,1] on `device`."""
    frame_files = sorted(glob.glob(os.path.join(gt_run, "frames", "frame_*.png")))
    assert frame_files, f"no frames in {gt_run}/frames"
    import imageio.v2 as imageio
    import torch

    frames = [imageio.imread(fp) for fp in frame_files]
    arr = np.stack(frames, 0).astype(np.float32) / 255.0  # [T,H,W,C]
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device)  # [T,C,H,W]


def run(cfg: RecoverV0Config):
    pick_free_gpu()
    import torch
    from .scene_io import load_from_spec
    from .sim_render import make_constant_v0, simulate_and_render, video_to_uint8
    from .run_io import RecoverRun
    from .v0field import V0Field
    from .recover_v0 import (
        recover_v0, plot_v0_recovery, plot_v0_quiver,
        v0_field_vs_gt_metrics, plot_v0_field_vs_gt)

    t0 = time.time()
    label = cfg.run_label or (
        f"{cfg.kind}_from-{os.path.basename(os.path.normpath(cfg.gt_run))}")
    rd = RecoverRun.create(__name__, label, cfg.out, config=cfg)  # auto-saves config
    with rd.capture_output():  # tee stdout+stderr into the run dir
        with open(os.path.join(cfg.gt_run, "config.json")) as f:
            g = json.load(f)
        if isinstance(g.get("scene"), dict):  # current schema: resolved ForwardConfig
            scene_spec = SceneSpec(**g["scene"])
        else:  # legacy schema
            scene_spec = SceneSpec(
                path=g["dataset_dir"], kind=g.get("scene_type", "pd"),
                downsample_scale=g.get("downsample_scale", 0.1),
                cache_path=g.get("scene_cache"))
        sim = SimConfig(**g["sim"])
        E = float(g["E"])           # known global E, held fixed
        gt_v0 = g["v0"]             # GT constant v0 [3] (phase A), for error reporting
        frame = g["frame"]
        device = scene_spec.device

        gt = _load_gt_frames(cfg.gt_run, device)
        scene = load_from_spec(scene_spec, sim)
        rd.copy_in(scene_spec.cache_path, "scene_cache.pt")  # freeze discretisation
        if scene_spec.kind == "pd":
            rd.link_source_ply(scene_spec.path)
        try:
            cam = scene.camera_by_frame(frame)
        except Exception:
            cam = scene.test_camera_list[0]

        rd.gt_video(video_to_uint8(gt), fps=sim.fps)

        # two-stage (good-init): a robust global stage solves the mean v0, then the
        # field starts AT that solution and refines (the pixel gradient only behaves
        # near the basin). For kind="global" the field IS the global stage -> skip.
        init_v0 = None
        if cfg.kind != "global" and cfg.two_stage:
            g_field = V0Field(aabb=scene.sim_aabb, kind="global", v_clamp=cfg.v_clamp).to(device)
            g_res = recover_v0(
                scene, gt, sim, cam, E, field=g_field, iters=cfg.stage1_iters,
                lr=cfg.stage1_lr, window=cfg.window, reg_weight=0.0,
                grad_clip=cfg.grad_clip, gt_v0=gt_v0, device=device)
            init_v0 = tuple(g_res["recovered_mean_v0"])
            print(f"[train_v0] stage-1 global init_v0={init_v0} "
                  f"(loss {g_res['final_loss']:.2e})")
            rd.write_json("stage1_trace.json", {
                "loss": g_res["loss_traj"], "v0_mean": g_res["v0_mean_traj"]})

        field = V0Field(
            aabb=scene.sim_aabb, kind=cfg.kind, res=cfg.res, feat_dim=cfg.feat_dim,
            mlp_hidden=cfg.mlp_hidden, v_clamp=cfg.v_clamp, init_v0=init_v0,
            out_scale=cfg.vel_scale,
        ).to(device)
        n_params = sum(p.numel() for p in field.parameters())

        res = recover_v0(
            scene, gt, sim, cam, E, field=field, iters=cfg.iters, lr=cfg.lr,
            window=cfg.window, window_start=(cfg.window_start or None),
            reg_weight=cfg.reg_weight, grad_clip=cfg.grad_clip,
            weight_decay=cfg.weight_decay, gt_v0=gt_v0, device=device)

        # videos: init (v0=0 -> rest), recovered v0, GT|recovered montage.
        v0_final = torch.from_numpy(res["v0_final"]).float().to(device)  # [n,3]
        v0_zero = torch.zeros_like(v0_final)
        gt_u8 = video_to_uint8(gt)
        pred_final = video_to_uint8(
            simulate_and_render(scene, E, v0_final, sim, cam).detach())
        pred_init = video_to_uint8(
            simulate_and_render(scene, E, v0_zero, sim, cam).detach())
        rd.pred_videos(pred_init, pred_final, gt_u8, fps=sim.fps)

        rd.merge_config(
            scene=scene_spec.to_dict(), sim=sim.to_dict(), scene_name=scene.name,
            E=E, gt_v0=gt_v0, kind=cfg.kind,
            n_mpm_particles=int(scene.sim_xyzs.shape[0]), n_field_params=n_params,
        )
        # per-particle scoring against a known GT v0 field (phase B: gradient GT)
        gt_field_path = os.path.join(cfg.gt_run, "v0_field.npy")
        v0_field_metrics = {}
        if os.path.exists(gt_field_path):
            v0_gt = np.load(gt_field_path)                       # [n,3]
            qm = scene.query_mask.detach().cpu().numpy()
            v0_field_metrics = v0_field_vs_gt_metrics(res["v0_final"], v0_gt, qm)
            plot_v0_field_vs_gt(
                rd.path("v0_field_vs_gt.png"), scene.sim_xyzs.detach().cpu().numpy(),
                res["v0_final"], v0_gt, qm,
                title=f"{scene.name} [{cfg.kind}] v0 field vs GT")

        rd.metrics(
            kind=cfg.kind, n_field_params=n_params, E=E, gt_v0=gt_v0,
            recovered_mean_v0=res["recovered_mean_v0"],
            v0_moving_std=res["v0_moving_std"],
            v0_l2_err=res.get("v0_l2_err"), v0_rel_err=res.get("v0_rel_err"),
            v0_mag_err=res.get("v0_mag_err"), v0_angle_deg=res.get("v0_angle_deg"),
            final_loss=res["final_loss"], min_loss=res["min_loss"],
            reg_weight=cfg.reg_weight, elapsed_sec=round(time.time() - t0, 2),
            **v0_field_metrics,
        )
        rd.write_json("trace.json", {
            "loss": res["loss_traj"], "reg": res["reg_traj"],
            "v0_mean": res["v0_mean_traj"], "v0_mag": res["v0_mag_traj"]})
        np.save(rd.path("v0_final.npy"), res["v0_final"])
        plot_v0_recovery(rd.path("recovery.png"), res,
                         title=f"{scene_spec.kind}:{scene.name} [{cfg.kind}] E={E:.1e}")
        plot_v0_quiver(rd.path("v0_quiver.png"),
                       scene.sim_xyzs.detach().cpu().numpy(), res["v0_final"],
                       title=f"{scene.name} recovered v0 [{cfg.kind}]")
        rd.finish()

        rec = res["recovered_mean_v0"]
        print(f"[train_v0] {scene_spec.kind}:{scene.name} [{cfg.kind}] "
              f"gt_v0={gt_v0} recovered=[{rec[0]:.3f},{rec[1]:.3f},{rec[2]:.3f}] "
              f"l2_err={res.get('v0_l2_err', float('nan')):.3f} "
              f"angle={res.get('v0_angle_deg', float('nan')):.1f}deg "
              f"final_loss={res['final_loss']:.2e} ({time.time()-t0:.0f}s) -> {rd.root}")
    return rd


if __name__ == "__main__":
    run(tyro.cli(RecoverV0Config))
