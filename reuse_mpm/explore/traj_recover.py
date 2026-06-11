"""Decisive probe: does MULTI-FRAME TRAJECTORY supervision recover E (scalar OR field)?

All gradchecks + the backward audit converge on: the pixel gradient is the fragile
projection; the MPM 1-frame gradient is directionally sound (~2x scalar attenuation,
Adam-harmless); and the truncated (g=1) trajectory gradient stays usable multi-frame.
GIC/PAC-NeRF/NeuMA all avoid pure-pixel supervision (geometry / particle losses). So
the robust scheme is: supervise PARTICLE POSITIONS over a long window, truncated g=1.

This optimises either a global log10(E) or an EField (voxel/triplane) from a far
init, using window-frame trajectory MSE vs GT positions (GT = rollout at the true
material, no grad). GT material is uniform true_E, or a gradient field (Phase B).

  # scalar, uniform GT
  python -m reuse_mpm.explore.traj_recover --scene.preset telephone --mode scalar \
      --true_E 1e5 --init_E 3e5 --window 8
  # voxel field, gradient GT (the real test: recover a spatially-varying field)
  python -m reuse_mpm.explore.traj_recover --scene.preset telephone --mode field \
      --backbone voxel --true_E 1e5 --init_E 3e5 --window 8 \
      --grad_axis 0 --grad_decades 1.0
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import tyro

from ..config import SceneSpec, SimConfig
from ..gpu import pick_free_gpu


@dataclass
class TrajRecoverConfig:
    scene: SceneSpec
    mode: str = "scalar"            # "scalar" | "field"
    backbone: str = "voxel"        # field backbone
    true_E: float = 1e5
    init_E: float = 3e5
    # Phase-B gradient GT: if grad_axis is set, the GT material is a smooth E
    # gradient (geomean true_E) instead of uniform -> tests spatial field recovery.
    grad_axis: Optional[int] = None
    grad_decades: float = 1.0
    v0: tuple = (0.0, -0.5, 0.0)
    sim: SimConfig = field(default_factory=lambda: SimConfig(num_frames=14, substep=64))
    window: int = 8
    grad_window: int = 1
    iters: int = 120
    lr: float = 0.1               # scalar: lr on log10E; field: Adam lr on params
    res: int = 16
    reg_weight: float = 1e-3
    out: Optional[str] = None
    run_label: str = ""


def run(cfg: TrajRecoverConfig):
    pick_free_gpu()
    from ..scene_io import load_from_spec
    from ..sim_render import make_constant_v0, make_gradient_E
    from ..mpm_rollout import MpmRollout
    from ..efield import EField
    from ..run_io import RunDir

    rd = RunDir.create(__name__, cfg.run_label, cfg.out, config=cfg)
    with rd.capture_output():
        scene = load_from_spec(cfg.scene, cfg.sim)
        dev = scene.device
        v0 = make_constant_v0(scene, cfg.v0).detach()
        roll = MpmRollout(scene, cfg.sim, requires_grad=True, device=dev)
        W = min(cfg.window, cfg.sim.num_frames - 1)
        rest = scene.sim_xyzs.detach()                         # [n,3]
        log_true = math.log10(cfg.true_E)

        # GT material field (for scoring) + GT particle trajectory
        if cfg.grad_axis is None:
            E_gt = torch.full((rest.shape[0],), cfg.true_E, device=dev)  # [n] uniform
        else:
            E_gt = make_gradient_E(scene, cfg.true_E, cfg.grad_axis, cfg.grad_decades)
        with torch.no_grad():
            gt_pos = [roll.rollout_Evec(E_gt, ti, v0, cfg.grad_window,
                                        requires_grad=False).detach() for ti in range(W)]

        # ---- optimise ----
        if cfg.mode == "scalar":
            logE = torch.tensor(math.log10(cfg.init_E), device=dev, requires_grad=True)
            params = [logE]
            efield = None
        else:
            efield = EField(scene.sim_aabb, backbone=cfg.backbone, init_E=cfg.init_E,
                            res=cfg.res).to(dev)
            params = list(efield.parameters())
        opt = torch.optim.Adam(params, lr=cfg.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.iters,
                                                           eta_min=cfg.lr * 0.05)

        def cur_Evec():
            if efield is None:
                return (10.0 ** logE) * torch.ones(rest.shape[0], device=dev)
            return efield.E_vec(rest)

        losses, geos = [], []
        for it in range(cfg.iters):
            opt.zero_grad()
            tot = 0.0
            for ti in range(W):
                E_vec = cur_Evec()
                pos = roll.rollout_Evec(E_vec, ti, v0, cfg.grad_window)
                l = F.mse_loss(pos, gt_pos[ti]) / W
                l.backward()
                tot += float(l.item())
            if efield is not None and cfg.reg_weight > 0:
                rl = cfg.reg_weight * efield.regularization(); rl.backward()
            opt.step(); sched.step()
            with torch.no_grad():
                E = cur_Evec()
                geos.append(float(torch.exp(torch.log(E).mean())))
            losses.append(tot)
            if it % 10 == 0 or it == cfg.iters - 1:
                print(f"  iter {it:3d}  E_geomean={geos[-1]:.3e}  traj_loss={tot:.3e}")

        with torch.no_grad():
            E_final = cur_Evec().detach().cpu().numpy()         # [n]
            E_gt_np = E_gt.detach().cpu().numpy()
        geo = float(np.exp(np.log(E_final).mean()))
        rel_geo = abs(geo - cfg.true_E) / cfg.true_E
        # per-particle field score (log10 space)
        lf, lg = np.log10(E_final), np.log10(E_gt_np)
        rmse = float(np.sqrt(np.mean((lf - lg) ** 2)))
        corr = float(np.corrcoef(lf, lg)[0, 1]) if lf.std() > 1e-9 and lg.std() > 1e-9 else float("nan")
        np.save(rd.path("E_final.npy"), E_final)
        rd.write_json("trace.json", {"E_geomean": geos, "loss": losses})
        _plot(rd.path("recovery.png"), losses, geos, E_final, E_gt_np,
              rest.detach().cpu().numpy(), cfg.true_E, corr,
              f"{cfg.mode} {'['+cfg.backbone+']' if efield else ''} "
              f"win={W} init={cfg.init_E:.1e}")
        rd.write_json("result.json", {
            "mode": cfg.mode, "backbone": cfg.backbone if efield else None,
            "true_E": cfg.true_E, "init_E": cfg.init_E,
            "grad_axis": cfg.grad_axis, "grad_decades": cfg.grad_decades,
            "recovered_geomean_E": geo, "rel_err_geomean": rel_geo,
            "E_min": float(E_final.min()), "E_max": float(E_final.max()),
            "log10_rmse_vs_gt": rmse, "log10_corr_vs_gt": corr,
            "window": W, "grad_window": cfg.grad_window,
            "final_loss": losses[-1], "min_loss": float(np.min(losses))})
        rd.finish()
        print(f"[traj_recover] mode={cfg.mode} {'['+cfg.backbone+']' if efield else ''} "
              f"window={W} g={cfg.grad_window} true={cfg.true_E:.2e} init={cfg.init_E:.2e} "
              f"-> geomean={geo:.3e} ({rel_geo*100:.0f}%) "
              f"E:[{E_final.min():.2e},{E_final.max():.2e}] "
              f"log10rmse_vs_gt={rmse:.3f} corr={corr:.3f} loss={losses[-1]:.2e}")
    return rd


def _plot(path, losses, geos, E_final, E_gt, xyz, true_E, corr, title):
    """LOSS curve first (the deciding artifact), then E_geomean vs iter, then a
    per-particle GT-vs-recovered scatter (field spatial fidelity). Always emitted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    # 1) LOSS vs iter -- converged? plateaued? diverged?
    ax[0].plot(losses, "-")
    ax[0].set_yscale("log"); ax[0].set_xlabel("iter"); ax[0].set_ylabel("traj loss")
    ax[0].set_title(f"LOSS vs iter (min {min(losses):.2e})")
    # 2) E geomean vs iter
    ax[1].plot(geos, "-"); ax[1].axhline(true_E, color="r", ls="--", label=f"true {true_E:.0e}")
    ax[1].set_yscale("log"); ax[1].set_xlabel("iter"); ax[1].set_ylabel("E geomean")
    ax[1].set_title("E geomean vs iter"); ax[1].legend(fontsize=8)
    # 3) per-particle GT vs recovered (only meaningful if GT varies spatially)
    lf, lg = np.log10(E_final), np.log10(E_gt)
    if lg.std() > 1e-9:
        ax[2].scatter(lg, lf, s=3, alpha=0.4)
        lim = [min(lf.min(), lg.min()), max(lf.max(), lg.max())]
        ax[2].plot(lim, lim, "r--", lw=1)
        ax[2].set_xlabel("GT log10 E"); ax[2].set_ylabel("recovered log10 E")
        ax[2].set_title(f"per-particle (corr={corr:.2f}, ideal=diagonal)")
    else:
        ax[2].hist(lf, bins=40); ax[2].axvline(np.log10(true_E), color="r", ls="--")
        ax[2].set_xlabel("recovered log10 E"); ax[2].set_title("field distribution (uniform GT)")
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    run(tyro.cli(TrajRecoverConfig))
