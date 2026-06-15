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

The GT run is read, the scene rebuilt, and this run's discretisation cache frozen
by `recover_context` (the shared materialiser); the body below just optimises and
records results -- it receives a ready RecoverContext, not a bare run dir.
"""
from __future__ import annotations

import os
import time

import tyro

from .config import RecoverConfig
from .gt_context import RecoverContext, recover_context
from .run_io import RecoverRun, entrypoint


@entrypoint(
    RecoverRun,
    pick_gpu=True,
    context=recover_context,
    label=lambda c: c.run_label
    or f"from-{os.path.basename(os.path.normpath(c.gt_run))}_init{c.init_E:g}",
)
def run(cfg: RecoverConfig, ctx: RecoverContext) -> None:
    from .recover import plot_recovery, recover_global_E
    from .sim_render import simulate_and_render

    rd = ctx.rd
    # the single shared recovery routine (see recover.py)
    res = recover_global_E(
        ctx.scene, ctx.gt, ctx.sim, ctx.cam, ctx.v0,
        init_E=cfg.init_E, iters=cfg.iters, lr=cfg.lr,
        window=cfg.window, grad_window=cfg.grad_window,
        coarse_init=cfg.coarse_init, coarse_n=cfg.coarse_n,
        true_E=ctx.true_E, device=ctx.device)

    # keep videos: init guess, recovered, GT|recovered montage. Pass the raw float
    # renders -- pred_videos owns the uint8 encoding (IO boundary), not this body.
    rd.pred_videos(
        simulate_and_render(ctx.scene, float(res["init_E"]), ctx.v0, ctx.sim, ctx.cam),
        simulate_and_render(ctx.scene, res["recovered_E"], ctx.v0, ctx.sim, ctx.cam),
        ctx.gt, fps=ctx.sim.fps)

    # NOTE: scene/sim/gt_E/gt_v0 are recorded to context.json by ctx.seal() at
    # lifetime end -- the body writes only RESULTS (metrics/trace/plot) here.
    rd.metrics(
        true_E=ctx.true_E, recovered_E=res["recovered_E"],
        final_iter_E=res["final_iter_E"], rel_err=res["rel_err"],
        log10_err=res["log10_err"], final_loss=res["final_loss"],
        coarse=res["coarse"], elapsed_sec=round(time.time() - ctx.t0, 2),
    )
    rd.trace(res["E_traj"], res["loss_traj"])
    rd.recovery_plot(plot_recovery(res, ctx.true_E,
                                   title=f"{ctx.scene_spec.kind}:{ctx.scene.name}"))

    print(f"[train] {ctx.scene_spec.kind}:{ctx.scene.name} true={ctx.true_E:.3e} "
          f"recovered={res['recovered_E']:.3e} rel_err={res['rel_err']*100:.0f}% "
          f"final_loss={res['final_loss']:.2e} ({time.time()-ctx.t0:.0f}s) -> {rd.root}")


if __name__ == "__main__":
    run(tyro.cli(RecoverConfig))
