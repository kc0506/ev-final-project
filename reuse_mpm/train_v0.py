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

Shares `recover_context` with train_global_E / train_field_E. One asymmetry: this
task RECOVERS v0, so it does not feed the GT's constant v0 (ctx.v0) into the sim --
it reads E (ctx.true_E) as the known constant and ctx.gt_v0 as the target to score.
"""
from __future__ import annotations

import os
import time

import numpy as np
import tyro

from .config import RecoverV0Config
from .gt_context import RecoverContext, recover_context
from .run_io import RecoverRun, entrypoint


@entrypoint(
    RecoverRun,
    pick_gpu=True,
    context=recover_context,
    label=lambda c: c.run_label
    or f"{c.kind}_from-{os.path.basename(os.path.normpath(c.gt_run))}",
)
def run(cfg: RecoverV0Config, ctx: RecoverContext) -> None:
    import torch

    from .gt_context import load_run_field
    from .recover_v0 import (
        plot_v0_field_vs_gt, plot_v0_quiver, plot_v0_recovery, recover_v0,
        v0_field_vs_gt_metrics)
    from .sim_render import simulate_and_render
    from .v0field import V0Field

    rd, scene = ctx.rd, ctx.scene
    E, gt_v0, device = ctx.true_E, ctx.gt_v0, ctx.device  # E known/fixed; gt_v0 = target

    # two-stage (good-init): a robust global stage solves the mean v0, then the field
    # starts AT that solution and refines (the pixel gradient only behaves near the
    # basin). For kind="global" the field IS the global stage -> skip.
    init_v0 = None
    if cfg.kind != "global" and cfg.two_stage:
        g_field = V0Field(aabb=scene.sim_aabb, kind="global", v_clamp=cfg.v_clamp).to(device)
        g_res = recover_v0(
            scene, ctx.gt, ctx.sim, ctx.cam, E, field=g_field, iters=cfg.stage1_iters,
            lr=cfg.stage1_lr, window=cfg.window, reg_weight=0.0,
            grad_clip=cfg.grad_clip, gt_v0=gt_v0, device=device)
        init_v0 = tuple(g_res["recovered_mean_v0"])
        print(f"[train_v0] stage-1 global init_v0={init_v0} (loss {g_res['final_loss']:.2e})")
        rd.write_json("stage1_trace.json", {
            "loss": g_res["loss_traj"], "v0_mean": g_res["v0_mean_traj"]})

    field = V0Field(
        aabb=scene.sim_aabb, kind=cfg.kind, res=cfg.res, feat_dim=cfg.feat_dim,
        mlp_hidden=cfg.mlp_hidden, v_clamp=cfg.v_clamp, init_v0=init_v0,
        out_scale=cfg.vel_scale,
    ).to(device)
    n_params = sum(p.numel() for p in field.parameters())

    res = recover_v0(
        scene, ctx.gt, ctx.sim, ctx.cam, E, field=field, iters=cfg.iters, lr=cfg.lr,
        window=cfg.window, window_start=(cfg.window_start or None),
        reg_weight=cfg.reg_weight, grad_clip=cfg.grad_clip,
        weight_decay=cfg.weight_decay, gt_v0=gt_v0, device=device)

    # videos: init (v0=0 -> rest), recovered v0, GT|recovered montage. Raw float
    # renders -> pred_videos owns the uint8 encoding (IO boundary).
    v0_final = torch.from_numpy(res["v0_final"]).float().to(device)  # [n,3]
    rd.pred_videos(
        simulate_and_render(scene, E, torch.zeros_like(v0_final), ctx.sim, ctx.cam),
        simulate_and_render(scene, E, v0_final, ctx.sim, ctx.cam),
        ctx.gt, fps=ctx.sim.fps)

    # per-particle scoring against a known GT v0 field (phase B: gradient GT)
    xyz = scene.sim_xyzs.detach().cpu().numpy()                  # [n,3]
    v0_gt = load_run_field(ctx.gt_run, "v0_field.npy")           # [n,3] or None
    v0_field_metrics = {}
    if v0_gt is not None:
        qm = scene.query_mask.detach().cpu().numpy()
        v0_field_metrics = v0_field_vs_gt_metrics(res["v0_final"], v0_gt, qm)
        rd.savefig("v0_field_vs_gt.png", plot_v0_field_vs_gt(
            xyz, res["v0_final"], v0_gt, qm,
            title=f"{scene.name} [{cfg.kind}] v0 field vs GT"))

    # context.json (scene/sim/gt_E/gt_v0) written by ctx.seal(); body writes RESULTS.
    rd.metrics(
        kind=cfg.kind, n_field_params=n_params, E=E, gt_v0=gt_v0,
        recovered_mean_v0=res["recovered_mean_v0"],
        v0_moving_std=res["v0_moving_std"],
        v0_l2_err=res.get("v0_l2_err"), v0_rel_err=res.get("v0_rel_err"),
        v0_mag_err=res.get("v0_mag_err"), v0_angle_deg=res.get("v0_angle_deg"),
        final_loss=res["final_loss"], min_loss=res["min_loss"],
        reg_weight=cfg.reg_weight, elapsed_sec=round(time.time() - ctx.t0, 2),
        **v0_field_metrics,
    )
    rd.write_json("trace.json", {
        "loss": res["loss_traj"], "reg": res["reg_traj"],
        "v0_mean": res["v0_mean_traj"], "v0_mag": res["v0_mag_traj"]})
    np.save(rd.path("v0_final.npy"), res["v0_final"])  # data artifact (sealed by finish)
    rd.recovery_plot(plot_v0_recovery(
        res, title=f"{ctx.scene_spec.kind}:{scene.name} [{cfg.kind}] E={E:.1e}"))
    rd.savefig("v0_quiver.png", plot_v0_quiver(
        xyz, res["v0_final"], title=f"{scene.name} recovered v0 [{cfg.kind}]"))

    rec = res["recovered_mean_v0"]
    print(f"[train_v0] {ctx.scene_spec.kind}:{scene.name} [{cfg.kind}] "
          f"gt_v0={gt_v0} recovered=[{rec[0]:.3f},{rec[1]:.3f},{rec[2]:.3f}] "
          f"l2_err={res.get('v0_l2_err', float('nan')):.3f} "
          f"angle={res.get('v0_angle_deg', float('nan')):.1f}deg "
          f"final_loss={res['final_loss']:.2e} ({time.time()-ctx.t0:.0f}s) -> {rd.root}")


if __name__ == "__main__":
    run(tyro.cli(RecoverV0Config))
