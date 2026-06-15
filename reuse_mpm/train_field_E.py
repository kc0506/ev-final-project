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

Shares `recover_context` with train_global_E / train_v0: the GT load, scene rebuild,
cache-freeze and context.json provenance are identical and live there, NOT here.
"""
from __future__ import annotations

import os
import time

import numpy as np
import tyro

from .config import RecoverFieldConfig
from .gt_context import RecoverContext, recover_context
from .run_io import RecoverRun, entrypoint


@entrypoint(
    RecoverRun,
    pick_gpu=True,
    context=recover_context,
    label=lambda c: c.run_label
    or f"{c.backbone}_from-{os.path.basename(os.path.normpath(c.gt_run))}",
)
def run(cfg: RecoverFieldConfig, ctx: RecoverContext) -> None:
    import torch

    from .efield import EField
    from .gt_context import load_run_field
    from .recover import (
        field_vs_gt_metrics, plot_field_recovery, plot_field_scatter,
        plot_field_vs_gt, recover_field_E)
    from .sim_render import simulate_and_render

    rd, scene = ctx.rd, ctx.scene
    field = EField(
        aabb=scene.sim_aabb, backbone=cfg.backbone, init_E=cfg.init_E,
        res=cfg.res, feat_dim=cfg.feat_dim, mlp_hidden=cfg.mlp_hidden,
    ).to(ctx.device)
    n_params = sum(p.numel() for p in field.parameters())

    res = recover_field_E(
        scene, ctx.gt, ctx.sim, ctx.cam, ctx.v0, field=field, iters=cfg.iters,
        lr=cfg.lr, window=cfg.window, grad_window=cfg.grad_window,
        reg_weight=cfg.reg_weight, true_E=ctx.true_E, device=ctx.device)

    # videos: init field render, recovered field render, GT|recovered montage. Raw
    # float renders -> pred_videos owns the uint8 encoding (IO boundary).
    E_final = torch.from_numpy(res["E_final"]).float().to(ctx.device)  # [n]
    rd.pred_videos(
        simulate_and_render(scene, float(cfg.init_E), ctx.v0, ctx.sim, ctx.cam),
        simulate_and_render(scene, E_final, ctx.v0, ctx.sim, ctx.cam),
        ctx.gt, fps=ctx.sim.fps)

    # per-particle scoring against a known GT field (phase B: gradient GT)
    xyz = scene.sim_xyzs.detach().cpu().numpy()                  # [n,3]
    E_gt = load_run_field(ctx.gt_run, "E_field.npy")             # [n] or None
    gt_metrics = {}
    if E_gt is not None:
        gt_metrics = field_vs_gt_metrics(res["E_final"], E_gt)
        rd.savefig("field_vs_gt.png", plot_field_vs_gt(
            xyz, res["E_final"], E_gt,
            title=f"{scene.name} [{cfg.backbone}] field vs GT"))

    # context.json (scene/sim/gt_E/gt_v0) is written by ctx.seal(); the body writes
    # only RESULTS. n_field_params is a result -> it lives in metrics.json.
    rd.metrics(
        backbone=cfg.backbone, n_field_params=n_params, true_E=ctx.true_E,
        recovered_geomean_E=res["recovered_geomean_E"],
        rel_err_geomean=res.get("rel_err_geomean"),
        log10_err_geomean=res.get("log10_err_geomean"),
        E_min=float(np.min(res["E_final"])), E_max=float(np.max(res["E_final"])),
        final_loss=res["final_loss"], min_loss=res["min_loss"],
        reg_weight=cfg.reg_weight, elapsed_sec=round(time.time() - ctx.t0, 2),
        **gt_metrics,
    )
    rd.write_json("trace.json", {
        "loss": res["loss_traj"], "reg": res["reg_traj"],
        "E_geomean": res["E_geomean_traj"],
        "E_min": res["E_min_traj"], "E_max": res["E_max_traj"]})
    np.save(rd.path("E_final.npy"), res["E_final"])  # data artifact (sealed by finish)
    rd.recovery_plot(plot_field_recovery(
        res, ctx.true_E, title=f"{ctx.scene_spec.kind}:{scene.name} [{cfg.backbone}]"))
    rd.savefig("field_scatter.png", plot_field_scatter(
        xyz, res["E_final"], title=f"{scene.name} recovered E [{cfg.backbone}]"))

    print(f"[train_field] {ctx.scene_spec.kind}:{scene.name} [{cfg.backbone}] "
          f"true={ctx.true_E:.3e} geomean={res['recovered_geomean_E']:.3e} "
          f"rel_err={res.get('rel_err_geomean', float('nan'))*100:.0f}% "
          f"E:[{np.min(res['E_final']):.2e},{np.max(res['E_final']):.2e}] "
          f"final_loss={res['final_loss']:.2e} ({time.time()-ctx.t0:.0f}s) -> {rd.root}")


if __name__ == "__main__":
    run(tyro.cli(RecoverFieldConfig))
