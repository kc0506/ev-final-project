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

from .gpu import pick_free_gpu


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
    p.add_argument("--out", required=True)
    return p


def _load(spec, grid_size):
    kind, path = spec.split(":", 1)
    if kind == "pd":
        from .scene import load_scene, default_cache_path
        return load_scene(path, device="cuda:0", grid_size=grid_size,
                          cache_path=default_cache_path(path, 0.1, grid_size)), "pd"
    elif kind == "pg":
        from .scene_physgaussian import load_physgaussian_scene, default_pg_cache_path
        return load_physgaussian_scene(path, device="cuda:0", grid_size=grid_size,
                                       cache_path=default_pg_cache_path(path, 0.1, grid_size)), "pg"
    raise ValueError(f"bad scene spec {spec}")


def run(args):
    pick_free_gpu()
    from .sim_render import (
        SimConfig, make_constant_v0, build_mpm, render_disp_frame,
        simulate_and_render, video_to_uint8,
    )
    from .diff_sim import MPMDifferentiableSimulation
    from .run_io import RunDir

    device = "cuda:0"
    os.makedirs(args.out, exist_ok=True)
    cfg = SimConfig(num_frames=args.num_frames, substep=args.substep, grid_size=args.grid_size)
    substep_size = cfg.substep_size
    summary = []

    for spec in args.scenes:
        t0 = time.time()
        scene, kind = _load(spec, args.grid_size)
        sd = RunDir(os.path.join(args.out, f"{kind}_{scene.name}"))
        cam = scene.test_camera_list[0]
        v0 = make_constant_v0(scene, args.v0).detach()
        n = scene.sim_xyzs.shape[0]
        init_xyzs = scene.sim_xyzs.clone()
        density = torch.ones_like(init_xyzs[..., 0]) * cfg.density
        dmask = torch.ones_like(density).int()
        onev = torch.ones(n, device=device)
        nu_t = torch.tensor(float(cfg.nu), device=device)
        window = min(args.window, cfg.num_frames - 1)
        solver, state, model = build_mpm(scene, cfg, requires_grad=True)

        gt = simulate_and_render(scene, float(args.true_E), v0, cfg, cam).detach()
        gt_motion = (gt - gt[0:1]).abs().mean().item()
        sd.save_named_video("gt", video_to_uint8(gt), fps=cfg.fps)

        def recover(init_E):
            logE = torch.tensor(float(np.log10(init_E)), device=device, requires_grad=True)
            opt = torch.optim.Adam([logE], lr=args.lr)
            Es, losses = [], []
            for _ in range(args.iters):
                opt.zero_grad()
                ltot = 0.0
                for ti in range(window):
                    extra = max(0, (ti + 1 - args.grad_window) * cfg.substep)
                    ng = cfg.substep * (ti + 1) - extra
                    pos = MPMDifferentiableSimulation.apply(
                        solver, state, model, 0, substep_size, ng,
                        init_xyzs, v0, (10.0 ** logE) * onev, nu_t, density, dmask,
                        None, device, True, extra)
                    l = F.mse_loss(render_disp_frame(scene, pos, cam), gt[[ti + 1]]) / window
                    l.backward()
                    ltot += float(l.item())
                opt.step()
                Es.append(float(10.0 ** logE.item()))
                losses.append(ltot)
            return Es, losses

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
        summary.append(scene_res)
        print(f"  [{kind}:{scene.name}] done in {scene_res['elapsed_sec']}s "
              f"(gt_motion={gt_motion:.4f}, frozen={scene_res['n_frozen']}/{n})")

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"args": vars(args), "scenes": summary}, f, indent=2, default=str)
    _summary_plot(args.out, summary, args.true_E)
    print(f"[multiscene] {len(summary)} scenes -> {args.out}")
    return args.out


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
