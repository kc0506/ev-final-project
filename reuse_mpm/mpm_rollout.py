"""One owner for the differentiable MPM rollout.

The bare `MPMDifferentiableSimulation.apply(...)` is a 16-positional-arg call into
the vendored PhysDreamer autograd Function; it used to be copy-pasted (with the
same extra/num_grad truncated-BPTT arithmetic) in recover, recovery_sweep,
multiscene, and gradcheck. `MpmRollout` builds (solver, state, model) once for a
(scene, cfg) and exposes a single `rollout_to_frame`, so that call -- and the
truncated-BPTT window math -- lives in exactly one place.

This is only the DIFFERENTIABLE path. Forward generation (every per-frame
position, no grad) uses the streaming O(T) loop in sim_render.simulate_positions;
the differentiable path re-rolls from the initial state per frame (O(T^2)) because
truncated BPTT needs to choose, per frame, how many trailing substeps carry grad.
The two are intentionally separate; only the render path is shared.
"""
from __future__ import annotations

from typing import Union

import torch
from torch import Tensor

from .config import SimConfig
from .diff_sim import MPMDifferentiableSimulation
from .scene import SceneBundle
from .sim_render import build_mpm


class MpmRollout:
    """Differentiable MPM rollout for one (scene, cfg), reusing one solver/state/model."""

    def __init__(
        self,
        scene: SceneBundle,
        cfg: SimConfig,
        requires_grad: bool = True,
        device: str = "cuda:0",
    ) -> None:
        self.scene = scene
        self.cfg = cfg
        self.device = device
        self.solver, self.state, self.model = build_mpm(
            scene, cfg, requires_grad=requires_grad
        )
        n = scene.sim_xyzs.shape[0]
        self.init_xyzs = scene.sim_xyzs.clone()
        self.density = torch.ones_like(self.init_xyzs[..., 0]) * cfg.density
        self.density_mask = torch.ones_like(self.density).int()
        self.nu_t = torch.tensor(float(cfg.nu), device=device)
        self._onev = torch.ones(n, device=device)

    def rollout_to_frame(
        self,
        logE: Union[float, Tensor],
        ti: int,
        v0: Tensor,
        grad_window: int,
        requires_grad: bool = True,
    ) -> Tensor:
        """Differentiable rollout from the initial state to frame `ti+1`, GLOBAL E.

        `ti` is 0-indexed over the loss window (frame ti+1 is the target).
        `logE` is a scalar (python float or 0-dim tensor); it is broadcast to a [n]
        per-particle vector and handed to `rollout_Evec` (the per-particle grad path
        is the validated one -- the scalar/aggregating path is buggy, README gotcha
        #2). Used by recover_global_E.
        """
        return self.rollout_Evec((10.0 ** logE) * self._onev, ti, v0, grad_window,
                                 requires_grad=requires_grad)

    def rollout_Evec(
        self,
        E_vec: Tensor,
        ti: int,
        v0: Tensor,
        grad_window: int,
        requires_grad: bool = True,
    ) -> Tensor:
        """Differentiable rollout from the initial state to frame `ti+1`, FIELD E.

        `E_vec` is a [n] per-particle ABSOLUTE Young's modulus (already 10**log10E),
        e.g. queried from an `EField` at the rest positions; gradient flows back
        through it to the field parameters. This is the shared rollout body --
        `rollout_to_frame` is just the broadcast-a-scalar special case. Truncated
        BPTT: only the latest `grad_window` frames' substeps carry gradient; earlier
        substeps run detached (`extra_no_grad_steps`).
        """
        cfg = self.cfg
        extra = max(0, (ti + 1 - grad_window) * cfg.substep)  # 0 => full BPTT
        num_grad = cfg.substep * (ti + 1) - extra  # num_grad + extra == total substeps
        return MPMDifferentiableSimulation.apply(
            self.solver, self.state, self.model, 0, cfg.substep_size, num_grad,
            self.init_xyzs, v0, E_vec, self.nu_t, self.density, self.density_mask,
            None, self.device, requires_grad, extra,
        )
