"""Entrypoint: recover a spatially-varying Young's-modulus FIELD from a video.

The field variant of train_global_E. Same roundtrip contract -- consumes a
forward_gen run dir (its frames = GT, its config.json carries scene+sim+v0+frame+E)
and uses the EXACT setup that produced the GT -- but optimises an EField (voxel |
triplane) instead of a single global scalar E, testing whether the
over-parameterised landscape is easier to fit than the narrow 1-D scalar basin.

Predicts ABSOLUTE log10(E); --init_E only sets the (uniform) field initialisation.

  python -m reuse_mpm.train_field_E \
      --gt_run outputs/forward_gen/06_tele_E1e5 \
      --backbone voxel --init_E 3e5 --iters 80 --reg_weight 1e-3 \
      --window 6 --grad_window 2

When the GT was a global scalar E (phase A), the "right" field is ~uniform = E*:
the metric is whether the field's geomean E recovers E* AND the photometric loss
converges as well as / better than the scalar baseline.
"""
from __future__ import annotations

import glob
import json
import os
import time

import numpy as np
import tyro

from .config import RecoverFieldConfig, SceneSpec, SimConfig
from .gpu import pick_free_gpu


def _load_gt_frames(gt_run: str, device: str):
    frame_files = sorted(glob.glob(os.path.join(gt_run, "frames", "frame_*.png")))
    assert frame_files, f"no frames in {gt_run}/frames"
    import imageio.v2 as imageio
    import torch

    frames = [imageio.imread(fp) for fp in frame_files]
    arr = np.stack(frames, 0).astype(np.float32) / 255.0  # [T,H,W,C]
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device)  # [T,C,H,W]


def run(cfg: RecoverFieldConfig):
    pick_free_gpu()
    import torch
    from .scene_io import load_from_spec
    from .sim_render import make_constant_v0, simulate_and_render, video_to_uint8
    from .run_io import RecoverRun
    from .efield import EField
    from .recover import (
        recover_field_E, plot_field_recovery, plot_field_scatter,
        field_vs_gt_metrics, plot_field_vs_gt)

    t0 = time.time()
    label = cfg.run_label or (
        f"{cfg.backbone}_from-{os.path.basename(os.path.normpath(cfg.gt_run))}")
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
        true_E = float(g["E"])  # GT global scalar (phase A); field should recover ~uniform
        v0_vec = g["v0"]
        frame = g["frame"]
        device = scene_spec.device

        gt = _load_gt_frames(cfg.gt_run, device)
        scene = load_from_spec(scene_spec, sim)
        rd.copy_in(scene_spec.cache_path, "scene_cache.pt")  # freeze this run's discretisation
        if scene_spec.kind == "pd":
            rd.link_source_ply(scene_spec.path)
        try:
            cam = scene.camera_by_frame(frame)
        except Exception:
            cam = scene.test_camera_list[0]
        v0 = make_constant_v0(scene, v0_vec).detach()

        rd.gt_video(video_to_uint8(gt), fps=sim.fps)

        field = EField(
            aabb=scene.sim_aabb, backbone=cfg.backbone, init_E=cfg.init_E,
            res=cfg.res, feat_dim=cfg.feat_dim, mlp_hidden=cfg.mlp_hidden,
        ).to(device)
        n_params = sum(p.numel() for p in field.parameters())

        res = recover_field_E(
            scene, gt, sim, cam, v0, field=field, iters=cfg.iters, lr=cfg.lr,
            window=cfg.window, grad_window=cfg.grad_window, reg_weight=cfg.reg_weight,
            true_E=true_E, device=device)

        # videos: init field render, recovered field render, GT|recovered montage
        E_final = torch.from_numpy(res["E_final"]).float().to(device)  # [n]
        gt_u8 = video_to_uint8(gt)
        pred_final = video_to_uint8(
            simulate_and_render(scene, E_final, v0, sim, cam).detach())
        pred_init = video_to_uint8(
            simulate_and_render(scene, float(cfg.init_E), v0, sim, cam).detach())
        rd.pred_videos(pred_init, pred_final, gt_u8, fps=sim.fps)

        rd.merge_config(
            scene=scene_spec.to_dict(), sim=sim.to_dict(),
            scene_name=scene.name, true_E=true_E, v0=v0_vec,
            n_mpm_particles=int(scene.sim_xyzs.shape[0]), n_field_params=n_params,
        )
        # per-particle scoring against a known GT field (phase B: gradient GT)
        gt_field_path = os.path.join(cfg.gt_run, "E_field.npy")
        gt_metrics = {}
        if os.path.exists(gt_field_path):
            E_gt = np.load(gt_field_path)
            gt_metrics = field_vs_gt_metrics(res["E_final"], E_gt)
            plot_field_vs_gt(rd.path("field_vs_gt.png"),
                             scene.sim_xyzs.detach().cpu().numpy(), res["E_final"], E_gt,
                             title=f"{scene.name} [{cfg.backbone}] field vs GT")

        rd.metrics(
            backbone=cfg.backbone, n_field_params=n_params, true_E=true_E,
            recovered_geomean_E=res["recovered_geomean_E"],
            rel_err_geomean=res.get("rel_err_geomean"),
            log10_err_geomean=res.get("log10_err_geomean"),
            E_min=float(np.min(res["E_final"])), E_max=float(np.max(res["E_final"])),
            final_loss=res["final_loss"], min_loss=res["min_loss"],
            reg_weight=cfg.reg_weight,
            elapsed_sec=round(time.time() - t0, 2),
            **gt_metrics,
        )
        rd.write_json("trace.json", {
            "loss": res["loss_traj"], "reg": res["reg_traj"],
            "E_geomean": res["E_geomean_traj"],
            "E_min": res["E_min_traj"], "E_max": res["E_max_traj"]})
        np.save(rd.path("E_final.npy"), res["E_final"])
        plot_field_recovery(rd.path("recovery.png"), res, true_E,
                            title=f"{scene_spec.kind}:{scene.name} [{cfg.backbone}]")
        plot_field_scatter(rd.path("field_scatter.png"),
                           scene.sim_xyzs.detach().cpu().numpy(), res["E_final"],
                           title=f"{scene.name} recovered E [{cfg.backbone}]")
        rd.finish()

        print(f"[train_field] {scene_spec.kind}:{scene.name} [{cfg.backbone}] "
              f"true={true_E:.3e} geomean={res['recovered_geomean_E']:.3e} "
              f"rel_err={res.get('rel_err_geomean', float('nan'))*100:.0f}% "
              f"E:[{np.min(res['E_final']):.2e},{np.max(res['E_final']):.2e}] "
              f"final_loss={res['final_loss']:.2e} ({time.time()-t0:.0f}s) -> {rd.root}")
    return rd


if __name__ == "__main__":
    run(tyro.cli(RecoverFieldConfig))
