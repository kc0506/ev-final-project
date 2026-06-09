"""Run the forward+backward pipeline across multiple scenes (PD and PhysGaussian).

For each scene: forward-generate a GT video at a known true E, then recover E
from a few inits using the faithful gradient method (substep 96, full BPTT,
per-particle E path, no coarse/cheating). Saves per-scene GT + recovered videos,
recovery curves, and a cross-scene summary. This is a generalisation smoke test,
not the full GT x init sweep.

Scene specs: "pd:/path/to/physdreamer_scene" or "pg:/path/to/physgaussian_model".

  python -m reuse_mpm.multiscene_fwdbwd \
      --scenes pd:.../telephone pd:.../alocasia pg:.../ficus_whitebg-trained \
      --true_E 1e5 --init_Es 3e4 3e5 --substep 96 --window 3 --iters 30 \
      --out outputs/multiscene
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..gpu import pick_free_gpu


def build_argparser():
    p = argparse.ArgumentParser(description="multi-scene forward+backward smoke")
    p.add_argument("--scenes", nargs="+", required=True,
                   help='scene specs "pd:<dir>" or "pg:<model_dir>"')
    p.add_argument("--true_E", type=float, default=1e5)
    p.add_argument("--init_Es", type=float, nargs="+", default=[3e4, 3e5])
    p.add_argument("--substep", type=int, default=96)
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--grad_window", type=int, default=14)  # >=window => full BPTT
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--v0", type=float, nargs=3, default=[0.0, -1.0, 0.0])
    p.add_argument("--grid_size", type=int, default=32)
    p.add_argument("--out", default=None,
                   help="override auto outputs/explore/multiscene_fwdbwd/<run>")
    return p


def _load(spec, cfg):
    """Load 'pd:<dir>' or 'pg:<model_dir>' via the unified SceneSpec dispatch."""
    from ..config import SceneSpec
    from ..scene_io import load_from_spec
    kind, path = spec.split(":", 1)
    return load_from_spec(SceneSpec(path=path, kind=kind, device="cuda:0"), cfg), kind


def run(args):
    pick_free_gpu()
    from ..sim_render import (
        SimConfig, make_constant_v0, simulate_and_render, video_to_uint8,
    )
    from ..recover import recover_global_E
    from ..run_io import RunDir

    device = "cuda:0"
    rd = RunDir.create(__name__, "", args.out)
    out = rd.root
    cfg = SimConfig(num_frames=args.num_frames, substep=args.substep, grid_size=args.grid_size)
    summary = []

    for spec in args.scenes:
        t0 = time.time()
        scene, kind = _load(spec, cfg)
        sd = RunDir(os.path.join(out, f"{kind}_{scene.name}"))
        cam = scene.test_camera_list[0]
        v0 = make_constant_v0(scene, args.v0).detach()
        n = scene.sim_xyzs.shape[0]

        gt = simulate_and_render(scene, float(args.true_E), v0, cfg, cam).detach()
        gt_motion = (gt - gt[0:1]).abs().mean().item()
        sd.save_named_video("gt", video_to_uint8(gt), fps=cfg.fps)

        def recover(init_E):
            res = recover_global_E(
                scene, gt, cfg, cam, v0, init_E=float(init_E), iters=args.iters,
                lr=args.lr, window=args.window, grad_window=args.grad_window,
                coarse_init=False, true_E=args.true_E, cosine=False, device=device)
            return res["E_traj"], res["loss_traj"]

        scene_res = {"scene": scene.name, "kind": kind, "spec": spec,
                     "true_E": args.true_E, "n_mpm": n,
                     "n_frozen": int(scene.freeze_mask.sum()),
                     "gt_motion": gt_motion, "inits": {}}
        for iE in args.init_Es:
            Es, losses = recover(iE)
            final = float(np.mean(Es[-5:]))
            scene_res["inits"][f"{iE:.0e}"] = {
                "init_E": iE, "E_traj": Es, "loss_traj": losses, "final_E": final,
                "final_loss": losses[-1], "rel_err": abs(final - args.true_E) / args.true_E}
            pred = simulate_and_render(scene, final, v0, cfg, cam).detach()
            sd.save_named_video(f"pred_init{iE:.0e}", video_to_uint8(pred), fps=cfg.fps)
            print(f"  [{kind}:{scene.name}] init={iE:.0e} -> final={final:.3e} "
                  f"rel_err={scene_res['inits'][f'{iE:.0e}']['rel_err']*100:.0f}%")
        scene_res["elapsed_sec"] = round(time.time() - t0, 1)
        sd.write_json("recovery.json", scene_res)
        _scene_plot(sd, scene_res, args.true_E)
        sd.finish()  # seals recovery.png + the per-init pred videos
        summary.append(scene_res)
        print(f"  [{kind}:{scene.name}] done in {scene_res['elapsed_sec']}s "
              f"(gt_motion={gt_motion:.4f}, frozen={scene_res['n_frozen']}/{n})")

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump({"args": vars(args), "scenes": summary}, f, indent=2, default=str)
    _summary_plot(out, summary, args.true_E)
    rd.finish()
    print(f"[multiscene] {len(summary)} scenes -> {out}")
    return out


def _scene_plot(sd, res, true_E):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    for k, r in res["inits"].items():
        ax.plot(r["E_traj"], label=f"init {r['init_E']:.0e} -> {r['final_E']:.1e}")
    ax.axhline(true_E, color="r", ls="--", label=f"true {true_E:.0e}")
    ax.set_yscale("log"); ax.set_xlabel("iter"); ax.set_ylabel("E")
    ax.set_title(f"{res['kind']}:{res['scene']} (motion={res['gt_motion']:.4f})")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(sd.path("recovery.png"), dpi=120); plt.close(fig)


def _summary_plot(out, summary, true_E):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{s['kind']}:{s['scene']}" for s in summary]
    x = np.arange(len(labels))
    for j, iE_key in enumerate(summary[0]["inits"].keys()):
        finals = [s["inits"][iE_key]["final_E"] for s in summary]
        ax.scatter(x, finals, label=f"init {iE_key}", zorder=3)
    ax.axhline(true_E, color="r", ls="--", label=f"true {true_E:.0e}")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("recovered E"); ax.set_title("E recovery across scenes")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(out, "summary.png"), dpi=120); plt.close(fig)


if __name__ == "__main__":
    run(build_argparser().parse_args())
