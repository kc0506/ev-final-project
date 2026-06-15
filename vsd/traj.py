"""Shared differentiable trajectory builder: a global v0 vector -> the 8-frame world
trajectory of the MOVING particles (frame 0 = rest, frames 1..7 = MPM rollout), which
render_flow turns into the 7-field flow the teacher consumes.

Kept separate so the SDS loop and the final visualiser build the trajectory identically.
"""
from __future__ import annotations

import torch
from torch import Tensor

from reuse_mpm.config import SimConfig
from reuse_mpm.mpm_rollout import MpmRollout
from vsd.scene_min import MinScene


class V0Trajectory:
    """Builds world trajectories for a (scene, cfg, E); one MpmRollout reused across steps."""

    def __init__(self, scene: MinScene, E: float, n_flow: int = 7,
                 device: str = "cuda:0", requires_grad: bool = True) -> None:
        self.scene = scene
        self.device = device
        self.n_flow = n_flow                                  # number of flow fields (=> n_flow+1 frames)
        self.cfg = SimConfig(num_frames=16, substep=64)       # dataset_gen config for this pack
        # requires_grad=False for table precompute (no warp tape -> faster, no leak)
        self.roll = MpmRollout(scene, self.cfg, requires_grad=requires_grad, device=device)
        n = scene.sim_xyzs.shape[0]
        self.E_vec = torch.full((n,), float(E), device=device)        # [n]
        self.qm = scene.query_mask                                    # [n] bool moving
        # rest world position of moving particles (constant, no grad)
        rest_world = scene.sim_xyzs * scene.scale - scene.shift        # [n,3]
        self.rest_move = rest_world[self.qm].detach()                  # [n_move,3]

    def world_traj(self, v0_vec: Tensor, grad_window: int | None = None,
                   requires_grad: bool = True) -> Tensor:
        """v0_vec [3] -> world trajectory of moving particles [n_flow+1, n_move, 3].

        grad_window: trailing frames carrying BPTT grad per rollout; None = full BPTT.
        requires_grad=False skips the warp tape (table precompute -> faster, no leak).
        """
        n = self.scene.sim_xyzs.shape[0]
        v0 = torch.zeros((n, 3), device=self.device, dtype=torch.float32)  # [n,3]
        v0 = v0.index_copy(0, self.qm.nonzero(as_tuple=True)[0],
                           v0_vec[None, :].expand(int(self.qm.sum()), 3))   # moving <- v0_vec
        frames = [self.rest_move]                                          # frame 0 = rest
        for ti in range(self.n_flow):
            gw = (ti + 1) if grad_window is None else min(grad_window, ti + 1)
            pos_norm = self.roll.rollout_Evec(self.E_vec, ti, v0, grad_window=gw,
                                              requires_grad=requires_grad)  # [n,3] norm
            world = pos_norm * self.scene.scale - self.scene.shift                  # [n,3]
            frames.append(world[self.qm])                                          # [n_move,3]
        return torch.stack(frames, dim=0)                                          # [n_flow+1,n_move,3]
