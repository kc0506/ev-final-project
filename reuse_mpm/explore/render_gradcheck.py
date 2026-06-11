"""Localise the pixel-gradient sign-flip to a LAYER: rasterizer vs +topk vs +MPM.

The E-level gradcheck (explore.gradcheck) lumps topk-interp + 3DGS-rasterizer + MPM
into one "pixel" gradient. This probe removes MPM entirely and tests the
render-only gradient w.r.t. a DISPLACEMENT SCALE theta, sweeping the motion
magnitude from sub-pixel to large -- directly answering "does small-motion 3DGS
flip?" (the 4DGS question).

Setup: a fixed per-particle direction field d_dir [n,3]; displaced positions =
rest + theta * d_dir. target image = render at theta0 (the "GT" motion). Then
  loss(theta) = MSE( render(rest + theta*d_dir), target )
and we compare analytic dL/dtheta (autograd through render_disp_frame, i.e.
topk + rasterizer, NO MPM) against central finite-difference, at theta_eval =
1.3*theta0 (an off-target guess). FLIP = signs disagree.

Direction modes:
  - "uniform": same unit vector on all moving particles -> topk mean of identical
    values is identity, so this is ~PURE rasterizer.
  - "mpm": the real frame-1 MPM displacement direction at true E -> rasterizer+topk
    with a realistic heterogeneous pattern.

  python -m reuse_mpm.explore.render_gradcheck --scene.preset telephone \
      --true_E 1e5 --scales 1.0 0.3 0.1 0.03 0.01 --fd 1e-3
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
import tyro

from ..config import SceneSpec, SimConfig
from ..gpu import pick_free_gpu


@dataclass
class RenderGradcheckConfig:
    scene: SceneSpec
    true_E: float = 1e5
    v0: tuple = (0.0, -0.5, 0.0)
    sim: SimConfig = field(default_factory=lambda: SimConfig(num_frames=14, substep=64))
    scales: List[float] = field(default_factory=lambda: [1.0, 0.3, 0.1, 0.03, 0.01])
    eval_mul: float = 1.3   # evaluate gradient at theta_eval = eval_mul * theta0
    fd: float = 1e-3        # central finite-diff step on theta
    out: Optional[str] = None
    run_label: str = ""


def run(cfg: RenderGradcheckConfig):
    pick_free_gpu()
    from ..scene_io import load_from_spec
    from ..sim_render import make_constant_v0, render_disp_frame
    from ..mpm_rollout import MpmRollout
    from ..run_io import RunDir

    rd = RunDir.create(__name__, cfg.run_label, cfg.out, config=cfg)
    with rd.capture_output():
        scene = load_from_spec(cfg.scene, cfg.sim)
        dev = scene.device
        try:
            cam = scene.camera_by_frame("frame_00001.png")
        except Exception:
            cam = scene.test_camera_list[0]
        rest = scene.sim_xyzs.detach()                     # [n,3] normalised
        qm = scene.query_mask                              # [n] moving

        # real frame-1 MPM displacement direction at true E (heterogeneous)
        v0 = make_constant_v0(scene, cfg.v0).detach()
        roll = MpmRollout(scene, cfg.sim, requires_grad=False, device=dev)
        with torch.no_grad():
            pos1 = roll.rollout_to_frame(float(torch.log10(torch.tensor(cfg.true_E))),
                                         0, v0, 1, requires_grad=False)  # [n,3]
        d_mpm = (pos1 - rest).detach()                     # [n,3]

        # uniform direction (moving only): topk-mean of identical -> ~pure rasterizer
        d_uni = torch.zeros_like(rest)
        d_uni[qm] = torch.tensor(cfg.v0, device=dev, dtype=rest.dtype)
        d_uni = d_uni / (d_uni.norm(dim=-1).mean() + 1e-12)  # ~unit typical norm

        def loss_at(theta: float, d_dir, target):
            disp = rest + theta * d_dir
            img = render_disp_frame(scene, disp, cam)       # [1,C,H,W]
            return F.mse_loss(img, target)

        modes = {"uniform(~raster)": d_uni, "mpm(raster+topk)": d_mpm}
        print(f"# {scene.name}  px-motion proxy: |d_mpm|.mean={float(d_mpm.norm(dim=-1).mean()):.2e} (norm units)")
        print(f"# scale -> theta0; eval at {cfg.eval_mul}*theta0; FLIP = analytic vs finite-diff disagree\n")
        for name, d_dir in modes.items():
            for s in cfg.scales:
                theta0 = float(s)
                with torch.no_grad():
                    target = render_disp_frame(scene, rest + theta0 * d_dir, cam).detach()
                theta_eval = cfg.eval_mul * theta0
                # analytic
                th = torch.tensor(theta_eval, device=dev, requires_grad=True)
                disp = rest + th * d_dir
                img = render_disp_frame(scene, disp, cam)
                l = F.mse_loss(img, target)
                l.backward()
                an = float(th.grad)
                # central finite-diff
                lp = float(loss_at(theta_eval + cfg.fd, d_dir, target))
                lm = float(loss_at(theta_eval - cfg.fd, d_dir, target))
                num = (lp - lm) / (2 * cfg.fd)
                flip = "FLIP" if (an * num < 0) else "OK"
                print(f"{name:18s} scale={s:<5g} theta0={theta0:<6g} "
                      f"an={an:+.3e} num={num:+.3e} [{flip}]")
        rd.finish()
    return rd


if __name__ == "__main__":
    run(tyro.cli(RenderGradcheckConfig))
