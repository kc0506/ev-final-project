"""Entrypoint: honest E-recovery sweep, faithful to PhysDreamer's gradient method.

Faithful to PhysDreamer (no improvised tricks):
  - substep = 96, density 2000, nu 0.3, jelly, gravity 0  (reference physics)
  - grad_window large => FULL BPTT through the whole rollout (extra_no_grad=0),
    exactly like PhysDreamer's default grad_window=14 for short clips
  - per-particle E gradient path (the path PhysDreamer actually uses), torch
    reduces it onto a single log E
  - per-frame forward+backward (state is mutated between frames)
NO coarse grid, NO pre-init cheating. Init is swept openly and we just plot the
curves -- if a far init cannot reach the minimum, the curve will show it.

Produces, for the cross product {true_E} x {init_E}:
  - per-(GT,init) E-vs-iter trajectory and loss curve
  - per-GT plot: E-vs-iter coloured by init (true E as dashed line)
  - summary plot: final recovered E vs init E, one line per GT
  - results.json with everything

  python -m reuse_mpm.recovery_sweep \
      --dataset_dir .../telephone \
      --true_Es 3e4 1e5 3e5 --init_Es 1e4 3e4 1e5 3e5 1e6 \
      --substep 96 --window 4 --iters 40 --lr 0.05 --out outputs/recsweep
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .gpu import pick_free_gpu


def build_argparser():
    p = argparse.ArgumentParser(description="honest E-recovery sweep (faithful PhysDreamer)")
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--true_Es", type=float, nargs="+", default=[3e4, 1e5, 3e5])
    p.add_argument("--init_Es", type=float, nargs="+",
                   default=[1e4, 3e4, 1e5, 3e5, 1e6])
    p.add_argument("--substep", type=int, default=96)
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--window", type=int, default=4, help="frames (from t=1) in the loss")
    p.add_argument("--grad_window", type=int, default=14,
                   help="frames keeping BPTT grad; >=window => full BPTT (PhysDreamer default 14)")
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.05, help="lr on log10 E")
    p.add_argument("--grid_size", type=int, default=32)
    p.add_argument("--downsample_scale", type=float, default=0.1)
    p.add_argument("--v0", type=float, nargs=3, default=[0.0, -1.0, 0.0])
    p.add_argument("--frame", default="frame_00001.png")
    p.add_argument("--out", required=True)
    return p


def run(args):
    pick_free_gpu()
    from .scene import load_scene, default_cache_path
    from .sim_render import (
        SimConfig, make_constant_v0, build_mpm, render_disp_frame,
        simulate_and_render,
    )
    from .diff_sim import MPMDifferentiableSimulation

    device = "cuda:0"
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)

    scene = load_scene(args.dataset_dir, device=device,
                       downsample_scale=args.downsample_scale, grid_size=args.grid_size,
                       cache_path=default_cache_path(args.dataset_dir,
                                                     args.downsample_scale, args.grid_size))
    cfg = SimConfig(num_frames=args.num_frames, substep=args.substep, grid_size=args.grid_size)
    cam = scene.camera_by_frame(args.frame)
    v0 = make_constant_v0(scene, args.v0).detach()

    n = scene.sim_xyzs.shape[0]
    init_xyzs = scene.sim_xyzs.clone()
    density = torch.ones_like(init_xyzs[..., 0]) * cfg.density
    density_mask = torch.ones_like(density).int()
    onev = torch.ones(n, device=device)
    nu_t = torch.tensor(float(cfg.nu), device=device)
    substep_size = cfg.substep_size
    window = min(args.window, cfg.num_frames - 1)
    solver, state, model = build_mpm(scene, cfg, requires_grad=True)

    def step_grads(logE, gt):
        total = 0.0
        for ti in range(window):
            extra = max(0, (ti + 1 - args.grad_window) * cfg.substep)  # 0 => full BPTT
            num_grad = cfg.substep * (ti + 1) - extra
            E_vec = (10.0 ** logE) * onev
            pos = MPMDifferentiableSimulation.apply(
                solver, state, model, 0, substep_size, num_grad,
                init_xyzs, v0, E_vec, nu_t, density, density_mask, None,
                device, True, extra,
            )
            l = F.mse_loss(render_disp_frame(scene, pos, cam), gt[[ti + 1]]) / window
            l.backward()
            total += float(l.item())
        return total

    def recover(true_E, init_E):
        gt = simulate_and_render(scene, float(true_E), v0, cfg, cam).detach()
        logE = torch.tensor(float(np.log10(init_E)), device=device, requires_grad=True)
        opt = torch.optim.Adam([logE], lr=args.lr)
        Es, losses = [], []
        for _ in range(args.iters):
            opt.zero_grad()
            loss = step_grads(logE, gt)
            opt.step()
            Es.append(float(10.0 ** logE.item()))
            losses.append(loss)
        final = float(np.mean(Es[-5:]))   # last-5 mean (robust to oscillation)
        return {"E_traj": Es, "loss_traj": losses,
                "final_E": final, "final_iter_E": Es[-1],
                "rel_err": abs(final - true_E) / true_E,
                "log10_err": abs(np.log10(final) - np.log10(true_E))}

    results = []
    for tE in args.true_Es:
        for iE in args.init_Es:
            r = recover(tE, iE)
            r["true_E"] = float(tE); r["init_E"] = float(iE)
            results.append(r)
            print(f"  true={tE:.1e} init={iE:.1e} -> final={r['final_E']:.3e} "
                  f"rel_err={r['rel_err']*100:5.1f}%  (init/true={iE/tE:.2f}x)")

    import json
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"args": vars(args), "results": results,
                   "elapsed_sec": round(time.time() - t0, 2)}, f, indent=2, default=str)

    _plots(args, results)
    print(f"[recsweep] {len(results)} runs in {time.time()-t0:.0f}s -> {args.out}")
    return args.out


def _plots(args, results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[recsweep] plotting skipped: {e}")
        return
    import numpy as np

    true_Es = sorted(set(r["true_E"] for r in results))
    init_Es = sorted(set(r["init_E"] for r in results))
    cmap = plt.get_cmap("viridis")

    # per-GT: E vs iter, coloured by init
    fig, axes = plt.subplots(1, len(true_Es), figsize=(5 * len(true_Es), 4), squeeze=False)
    for k, tE in enumerate(true_Es):
        ax = axes[0][k]
        for r in [r for r in results if r["true_E"] == tE]:
            c = cmap((np.log10(r["init_E"]) - np.log10(min(init_Es))) /
                     max(1e-9, np.log10(max(init_Es)) - np.log10(min(init_Es))))
            ax.plot(r["E_traj"], color=c, label=f"init {r['init_E']:.0e}")
        ax.axhline(tE, color="r", ls="--", lw=2)
        ax.set_yscale("log"); ax.set_xlabel("iter"); ax.set_ylabel("E")
        ax.set_title(f"true E = {tE:.0e}"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "E_vs_iter_by_GT.png"), dpi=120)
    plt.close(fig)

    # summary: final E vs init E, one line per GT
    fig, ax = plt.subplots(figsize=(6, 5))
    for tE in true_Es:
        rs = sorted([r for r in results if r["true_E"] == tE], key=lambda r: r["init_E"])
        ax.plot([r["init_E"] for r in rs], [r["final_E"] for r in rs],
                "o-", label=f"true {tE:.0e}")
        ax.axhline(tE, color="gray", ls=":", lw=0.8)
    ax.plot(init_Es, init_Es, "k--", lw=0.6, label="final=init (no move)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("init E"); ax.set_ylabel("final recovered E (last-5 mean)")
    ax.set_title("recovery vs init (flat-to-true = converged regardless of init)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(args.out, "final_vs_init.png"), dpi=120); plt.close(fig)


if __name__ == "__main__":
    run(build_argparser().parse_args())
