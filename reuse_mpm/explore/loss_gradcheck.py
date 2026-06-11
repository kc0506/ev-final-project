"""Does a different PHOTOMETRIC LOSS give a sign-reliable dL/dE through the chain?

render_gradcheck showed the render gradient (raster+topk) is sign-correct; the
window>=3 flip comes from the truncated MPM Jacobian dpos/dE, REVEALED through the
pixel-loss projection (MSE pulls along a photometric direction that is sensitive to
that distortion). A different loss = a different dL_pix/dpos direction. Maybe one
(L1, D-SSIM, L1+D-SSIM, blurred-MSE) projects more like the trajectory/position-
error direction and stays sign-correct even at window>=3.

This compares, through the FULL MPM+render chain, analytic dL/dlogE vs central
finite-difference, for several losses, at chosen E points and window.

  python -m reuse_mpm.explore.loss_gradcheck --scene.preset telephone \
      --true_E 1e5 --points 2e5 3e5 --window 3 --grad_window 1 --fd 0.02
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
class LossGradcheckConfig:
    scene: SceneSpec
    true_E: float = 1e5
    v0: tuple = (0.0, -0.5, 0.0)
    sim: SimConfig = field(default_factory=lambda: SimConfig(num_frames=14, substep=64))
    points: List[float] = field(default_factory=lambda: [2e5, 3e5])
    window: int = 3
    grad_window: int = 1
    fd: float = 0.02       # central finite-diff step on log10(E)
    blur_sigma: float = 2.0
    out: Optional[str] = None
    run_label: str = ""


def _gauss_kernel(sigma: float, device) -> torch.Tensor:
    r = max(1, int(3 * sigma))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    return k  # [2r+1]


def _blur(img: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Separable gaussian blur. img [1,C,H,W]."""
    C = img.shape[1]
    r = (k.numel() - 1) // 2
    kx = k.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    ky = k.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    img = F.conv2d(img, kx, padding=(0, r), groups=C)
    img = F.conv2d(img, ky, padding=(r, 0), groups=C)
    return img


def run(cfg: LossGradcheckConfig):
    pick_free_gpu()
    from ..scene_io import load_from_spec
    from ..sim_render import make_constant_v0, render_disp_frame
    from ..mpm_rollout import MpmRollout
    from ..run_io import RunDir
    from physdreamer.gaussian_3d.utils.loss_utils import ssim

    rd = RunDir.create(__name__, cfg.run_label, cfg.out, config=cfg)
    with rd.capture_output():
        scene = load_from_spec(cfg.scene, cfg.sim)
        dev = scene.device
        try:
            cam = scene.camera_by_frame("frame_00001.png")
        except Exception:
            cam = scene.test_camera_list[0]
        v0 = make_constant_v0(scene, cfg.v0).detach()
        roll = MpmRollout(scene, cfg.sim, requires_grad=True, device=dev)
        W = min(cfg.window, cfg.sim.num_frames - 1)
        log_true = float(math.log10(cfg.true_E))
        gk = _gauss_kernel(cfg.blur_sigma, dev)

        # GT target frames = render at true E (== the GT video by construction)
        gt_imgs = []
        with torch.no_grad():
            for ti in range(W):
                pos = roll.rollout_to_frame(log_true, ti, v0, cfg.grad_window,
                                            requires_grad=False)
                gt_imgs.append(render_disp_frame(scene, pos, cam).detach())  # [1,C,H,W]

        def dssim(a, b):
            return 1.0 - ssim(a, b)

        losses = {
            "mse": lambda a, b: F.mse_loss(a, b),
            "l1": lambda a, b: F.l1_loss(a, b),
            "dssim": dssim,
            "l1+dssim": lambda a, b: 0.8 * F.l1_loss(a, b) + 0.2 * dssim(a, b),
            "blur_mse": lambda a, b: F.mse_loss(_blur(a, gk), _blur(b, gk)),
        }

        def total_loss(logE, lf, requires_grad):
            tot = logE.new_zeros(()) if requires_grad else 0.0
            for ti in range(W):
                pos = roll.rollout_to_frame(logE, ti, v0, cfg.grad_window,
                                            requires_grad=requires_grad)
                img = render_disp_frame(scene, pos, cam)
                l = lf(img, gt_imgs[ti]) / W
                if requires_grad:
                    tot = tot + l
                else:
                    tot += float(l)
            return tot

        print(f"# {scene.name} window={W} grad_window={cfg.grad_window} "
              f"fd={cfg.fd} (log10E)  FLIP = analytic vs finite-diff disagree\n")
        for E in cfg.points:
            logE0 = math.log10(E)
            print(f"E={E:.2e}:")
            for name, lf in losses.items():
                # analytic
                logE = torch.tensor(logE0, device=dev, requires_grad=True)
                tot = total_loss(logE, lf, requires_grad=True)
                tot.backward()
                an = float(logE.grad)
                # central finite-diff (forward only)
                with torch.no_grad():
                    lp = total_loss(torch.tensor(logE0 + cfg.fd, device=dev), lf, False)
                    lm = total_loss(torch.tensor(logE0 - cfg.fd, device=dev), lf, False)
                num = (lp - lm) / (2 * cfg.fd)
                flip = "FLIP" if (an * num < 0) else "OK"
                print(f"   {name:10s} an={an:+.3e} num={num:+.3e} [{flip}]")
        rd.finish()
    return rd


if __name__ == "__main__":
    run(tyro.cli(LossGradcheckConfig))
