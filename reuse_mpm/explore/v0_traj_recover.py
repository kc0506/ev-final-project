"""Decisive probe: does TRAJECTORY supervision recover a v0 FIELD (vs pixel)?

train_v0 (pixel loss) recovers the GLOBAL mean v0 perfectly (window=1, l2_err 0.006)
but a v0 FIELD's spatial variation either diverges (triplane) or undershoots (voxel),
even from a good two-stage init. Hypothesis (same as the E-field finding,
mpm-grad-stability): the single-frame pixel image is an AGGREGATE that under-
determines the per-particle v0 ramp -> weak/noisy per-particle gradient. TRAJECTORY
loss gives a PER-PARTICLE position target at frame 1, which should directly identify
each particle's v0 and sculpt the ramp.

This optimises a V0Field (global|voxel|triplane) against frame-position MSE vs GT
positions (GT = rollout at the GT v0 field, no grad). v0 needs FULL BPTT to t=0, so
each windowed frame rolls with grad_window=ti+1 (gravity off -> early frames pure-v0).

  # uniform GT (phase A) -- field should match the global recovery
  python -m reuse_mpm.explore.v0_traj_recover --scene.preset telephone --kind voxel
  # gradient GT (phase B) -- the real test: recover the spatial v0 ramp
  python -m reuse_mpm.explore.v0_traj_recover --scene.preset telephone --kind triplane \
      --v0_grad_axis 0 --v0_grad_slope 1.0

Config is LOCAL (explore convention); reads SceneSpec/SimConfig + V0Field only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import tyro

from ..config import SceneSpec, SimConfig
from ..gpu import pick_free_gpu


@dataclass
class V0TrajRecoverConfig:
    scene: SceneSpec
    kind: str = "voxel"             # global | voxel | triplane
    v0: tuple = (0.0, -0.5, 0.0)    # GT base v0
    # Phase-B gradient GT: if v0_grad_axis set, GT v0 magnitude ramps along that axis.
    v0_grad_axis: Optional[int] = None
    v0_grad_slope: float = 1.0
    sim: SimConfig = field(default_factory=lambda: SimConfig(num_frames=14, substep=64))
    window: int = 1                 # full-BPTT frames whose particle positions supervise
    iters: int = 150
    lr: float = 0.02
    res: int = 16
    reg_weight: float = 0.0
    v_clamp: float = 5.0
    out: Optional[str] = None
    run_label: str = ""


def run(cfg: V0TrajRecoverConfig):
    pick_free_gpu()
    import torch
    import torch.nn.functional as F
    from ..scene_io import load_from_spec
    from ..sim_render import make_constant_v0, make_gradient_v0
    from ..mpm_rollout import MpmRollout
    from ..v0field import V0Field
    from ..run_io import RunDir

    rd = RunDir.create(__name__, cfg.run_label, cfg.out, config=cfg)
    with rd.capture_output():
        scene = load_from_spec(cfg.scene, cfg.sim)
        dev = scene.device
        roll = MpmRollout(scene, cfg.sim, requires_grad=True, device=dev)
        W = min(cfg.window, cfg.sim.num_frames - 1)
        rest = scene.sim_xyzs.detach()                          # [n,3]
        n = rest.shape[0]
        qm = scene.query_mask                                   # [n] bool
        E_vec = torch.full((n,), 1e5, device=dev)               # known E (uniform)

        # GT v0 field (uniform or spatial ramp) + GT particle trajectory.
        if cfg.v0_grad_axis is None:
            v0_gt = make_constant_v0(scene, cfg.v0).detach()    # [n,3]
        else:
            v0_gt = make_gradient_v0(scene, cfg.v0, cfg.v0_grad_axis,
                                     cfg.v0_grad_slope).detach()
        with torch.no_grad():
            gt_pos = [roll.rollout_Evec(E_vec, ti, v0_gt, ti + 1,
                                        requires_grad=False).detach() for ti in range(W)]

        field = V0Field(scene.sim_aabb, kind=cfg.kind, res=cfg.res,
                        v_clamp=cfg.v_clamp).to(dev)
        opt = torch.optim.Adam(field.parameters(), lr=cfg.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.iters,
                                                           eta_min=cfg.lr * 0.05)
        qm_np = qm.detach().cpu().numpy()
        v0_gt_np = v0_gt.detach().cpu().numpy()

        losses, rmses = [], []
        for it in range(cfg.iters):
            opt.zero_grad()
            tot = 0.0
            for ti in range(W):
                v0 = field.v0_vec(rest, qm)                     # [n,3]
                pos = roll.rollout_Evec(E_vec, ti, v0, ti + 1)  # full BPTT
                l = F.mse_loss(pos, gt_pos[ti]) / W
                l.backward()
                tot += float(l.item())
            if cfg.reg_weight > 0:
                rl = cfg.reg_weight * field.regularization(); rl.backward()
            opt.step(); sched.step()
            with torch.no_grad():
                vf = field.v0_vec(rest, qm).detach().cpu().numpy()  # [n,3]
                rmse = float(np.sqrt(((vf[qm_np] - v0_gt_np[qm_np]) ** 2).sum(-1).mean()))
            losses.append(tot); rmses.append(rmse)
            if it % 15 == 0 or it == cfg.iters - 1:
                print(f"  iter {it:3d}  traj_loss={tot:.3e}  v0_rmse_vs_gt={rmse:.4f}")

        with torch.no_grad():
            v0_final = field.v0_vec(rest, qm).detach().cpu().numpy()  # [n,3]
        f, g = v0_final[qm_np], v0_gt_np[qm_np]
        corr = []
        for c in range(3):
            corr.append(float(np.corrcoef(f[:, c], g[:, c])[0, 1])
                        if f[:, c].std() > 1e-9 and g[:, c].std() > 1e-9 else float("nan"))
        np.save(rd.path("v0_final.npy"), v0_final)
        rd.write_json("trace.json", {"loss": losses, "v0_rmse": rmses})
        rd.write_json("result.json", {
            "kind": cfg.kind, "v0_grad_axis": cfg.v0_grad_axis,
            "v0_grad_slope": cfg.v0_grad_slope, "window": W,
            "recovered_mean_v0": f.mean(0).tolist(), "gt_mean_v0": g.mean(0).tolist(),
            "v0_per_particle_rmse": rmses[-1], "min_rmse": float(np.min(rmses)),
            "v0_per_comp_corr": corr, "final_loss": losses[-1],
            "min_loss": float(np.min(losses))})
        rd.finish()
        print(f"[v0_traj_recover] kind={cfg.kind} grad_axis={cfg.v0_grad_axis} "
              f"window={W} -> rmse={rmses[-1]:.4f} (min {np.min(rmses):.4f}) "
              f"corr={corr} loss={losses[-1]:.2e}")
    return rd


if __name__ == "__main__":
    run(tyro.cli(V0TrajRecoverConfig))
