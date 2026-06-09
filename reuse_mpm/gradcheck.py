"""Localise the recovery gradient problem: trajectory-only (MPM) vs pixel (MPM+3DGS).

At chosen E points it compares, against finite-difference, the analytic dL/dlogE of:
  - trajectory L2 loss   = MSE(MPM particle positions, GT positions)   [MPM grad only]
  - pixel MSE loss       = MSE(3DGS-rendered frame, GT frame)          [MPM + render grad]
A FLIP (analytic vs numeric opposite sign) localises where the gradient breaks:
  traj OK + pixel FLIP  => the 3DGS render gradient is the culprit
  traj FLIP             => the MPM gradient itself is wrong (long-horizon / high-E)

Config matches the recovery (substep/window/grad_window). Point-to-point L2 is used
(not Chamfer): MPM particles are in exact correspondence across E.

  python -m reuse_mpm.gradcheck --scenes pd:telephone pd:carnations \
      --true_E 1e5 --points 3e4 3e5 --substep 32 --window 3 --grad_window 1 \
      --out outputs/gradcheck_traj_vs_pixel
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .gpu import pick_free_gpu

PD_ROOT = os.environ.get("PHYSDREAMER_ROOT", "/tmp2/b10401006/PhysDreamer")
PD_DATA = os.path.join(PD_ROOT, "data", "physics_dreamer")
PG_ROOT = "/tmp2/b10401006/PhysGaussian/model"


def build_argparser():
    p = argparse.ArgumentParser(description="trajectory vs pixel gradient gradcheck")
    p.add_argument("--scenes", nargs="+", default=["pd:telephone", "pd:carnations"],
                   help='scene specs "pd:<name>" or "pg:<model_dir_name>"')
    p.add_argument("--true_E", type=float, default=1e5)
    p.add_argument("--points", type=float, nargs="+", default=[3e4, 3e5])
    p.add_argument("--substep", type=int, default=32)
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--grad_window", type=int, default=1)
    p.add_argument("--fd", type=float, default=0.02, help="finite-diff step on log10 E")
    p.add_argument("--v0", type=float, nargs=3, default=[0.0, -1.0, 0.0])
    p.add_argument("--out", required=True)
    return p


def _load(spec, grid_size=32):
    kind, name = spec.split(":", 1)
    if kind == "pd":
        from .scene import load_scene, default_cache_path
        ds = os.path.join(PD_DATA, name)
        return load_scene(ds, device="cuda:0",
                          cache_path=default_cache_path(ds, 0.1, grid_size)), name
    from .scene_physgaussian import load_physgaussian_scene, default_pg_cache_path
    md = os.path.join(PG_ROOT, name)
    return load_physgaussian_scene(md, device="cuda:0",
                                   cache_path=default_pg_cache_path(md, 0.1, grid_size)), name


def run(args):
    pick_free_gpu()
    from .sim_render import SimConfig, make_constant_v0, render_disp_frame
    from .mpm_rollout import MpmRollout

    device = "cuda:0"
    cfg = SimConfig(num_frames=args.num_frames, substep=args.substep, grid_size=32)
    W = args.window
    d = args.fd
    os.makedirs(args.out, exist_ok=True)
    results = []

    for spec in args.scenes:
        scene, name = _load(spec)
        cam = scene.test_camera_list[0]
        try:
            cam = scene.camera_by_frame("frame_00001.png")
        except Exception:
            pass
        v0 = make_constant_v0(scene, args.v0).detach()
        roll = MpmRollout(scene, cfg, requires_grad=True, device=device)

        def rollout(logE, ti, grad):
            return roll.rollout_to_frame(logE, ti, v0, args.grad_window, requires_grad=grad)

        # GT trajectory (detached positions) + GT video (detached) at true E
        le_true = torch.tensor(float(np.log10(args.true_E)), device=device)
        gtpos, gtvid = [], []
        for ti in range(W):
            p = rollout(le_true, ti, False)
            gtpos.append(p.detach())
            gtvid.append(render_disp_frame(scene, p, cam).detach())

        def grads_at(Ept):
            le = torch.tensor(float(np.log10(Ept)), device=device, requires_grad=True)
            for ti in range(W):
                (F.mse_loss(rollout(le, ti, True), gtpos[ti]) / W).backward()
            traj_an = le.grad.item()
            le2 = torch.tensor(float(np.log10(Ept)), device=device, requires_grad=True)
            for ti in range(W):
                (F.mse_loss(render_disp_frame(scene, rollout(le2, ti, True), cam),
                            gtvid[ti]) / W).backward()
            pix_an = le2.grad.item()

            def Ltraj(lv):
                return float(sum((F.mse_loss(rollout(torch.tensor(lv, device=device), ti, False),
                                  gtpos[ti]) / W).item() for ti in range(W)))

            def Lpix(lv):
                return float(sum((F.mse_loss(render_disp_frame(
                    scene, rollout(torch.tensor(lv, device=device), ti, False), cam),
                    gtvid[ti]) / W).item() for ti in range(W)))

            l0 = float(np.log10(Ept))
            return (traj_an, (Ltraj(l0 + d) - Ltraj(l0 - d)) / (2 * d),
                    pix_an, (Lpix(l0 + d) - Lpix(l0 - d)) / (2 * d))

        for Ept in args.points:
            ta, tn, pa, pn = grads_at(Ept)
            row = {"scene": name, "spec": spec, "E_point": float(Ept),
                   "traj_analytic": ta, "traj_numeric": tn,
                   "traj_ok": bool(ta * tn > 0),
                   "pixel_analytic": pa, "pixel_numeric": pn,
                   "pixel_ok": bool(pa * pn > 0)}
            results.append(row)
            print(f"{name:12} E={Ept:.0e}: "
                  f"traj  an={ta:+.2e} num={tn:+.2e} [{'OK' if row['traj_ok'] else 'FLIP'}] | "
                  f"pixel an={pa:+.2e} num={pn:+.2e} [{'OK' if row['pixel_ok'] else 'FLIP'}]")

    with open(os.path.join(args.out, "gradcheck.json"), "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"[gradcheck] -> {os.path.join(args.out, 'gradcheck.json')}")
    return results


if __name__ == "__main__":
    run(build_argparser().parse_args())
