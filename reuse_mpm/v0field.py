"""Learnable initial-velocity field v0(pos) -> R^3.

The dual of `EField`. Where E-recovery (train_global_E / train_field_E) assumed the
initial velocity was KNOWN and only fit the stiffness, this module makes v0 the
optimised parameter (E held known, at least for the de-risk milestones). The MPM in
this pipeline has GRAVITY OFF (config.py:70) -- motion is driven *entirely* by v0 --
so v0 is highly identifiable from the early frames, and a zero-init field means
"start from rest" (an unbiased init, the PhysDreamer recipe).

Design choices (mirroring EField + PhysDreamer's TriplaneFields velocity field):
  - Three swappable variants behind one `kind` flag (global | voxel | triplane), so
    the param-count / landscape A/B is one CLI flag, exactly like EField's backbone.
  - ZERO-INIT everywhere: the field outputs v0=0 at iter 0 (PhysDreamer zero-inits
    the decoder's last layer). With gravity off, v0=0 == fully at rest, so the photo
    loss starts maximal and its gradient pulls v0 towards the true motion.
  - Output is the PHYSICAL normalised-space velocity directly (NO 0.1 scaling), to
    match `make_constant_v0`'s convention so forward-gen and inverse use the same v0.
  - A magnitude clamp (`v_clamp`) is the CFL/blow-up guard (EField clamps log10 E).

The per-particle v0[n,3] is queried at rest positions (`scene.sim_xyzs`) and then
masked to the moving (`query_mask`) particles by the caller, so frozen/anchor
particles keep v0=0 (again matching make_constant_v0). Gradient flows render -> MPM
-> init_velocity -> field params (the MPM autograd Function already returns the
init_velocity grad, diff_sim.py:172).
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

V0Kind = Literal["global", "voxel", "triplane"]


class V0Field(nn.Module):
    """pos[n,3] (normalised MPM coords) -> v0[n,3], learnable, zero-init.

    Args:
        aabb:       [2,3] padded aabb of the sim particles ([min; max] rows), used to
                    map query positions into [-1,1] for grid sampling (field variants).
        kind:       "global" (single shared [3] vector), "voxel" (3D grid of v0,
                    trilinear) or "triplane" (3 feature planes + tiny MLP).
        res:        grid/plane resolution (voxel|triplane).
        feat_dim:   per-plane feature channels (triplane only).
        mlp_hidden: hidden width of the triplane decoder.
        v_clamp:    hard clamp on per-component |v0| (a CFL/blow-up guard); None off.
    """

    def __init__(
        self,
        aabb: Tensor,
        kind: V0Kind = "triplane",
        res: int = 16,
        feat_dim: int = 16,
        mlp_hidden: int = 64,
        v_clamp: float | None = 5.0,
        init_v0: "tuple | None" = None,
        out_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.res = res
        self.register_buffer("aabb", aabb.detach().clone())  # [2,3]
        self.v_clamp = None if v_clamp is None else float(v_clamp)
        # out_scale: the field's raw output is multiplied by this before clamping
        # (PhysDreamer uses 0.1 -> the net outputs ~5 to mean v0~0.5, a conditioning
        # choice). init_v0 below is in PHYSICAL units, so the stored init is /out_scale
        # so that raw*out_scale == init_v0 at iter 0.
        self.out_scale = float(out_scale)
        # init_v0 biases the field's INITIAL output to a constant vector (else 0 =
        # rest). Used by the two-stage recovery: a robust global stage solves the
        # mean v0, then the field is initialised AT that solution so it starts inside
        # the loss basin (the pixel grad only behaves near the basin) and merely
        # refines the spatial variation -- the good-init path. See recover_v0.
        iv = torch.zeros(3) if init_v0 is None else torch.as_tensor(
            init_v0, dtype=torch.float32) / self.out_scale     # [3] stored pre-scale

        if kind == "global":
            self.vec = nn.Parameter(iv.clone())                # [3]
        elif kind == "voxel":
            # [1,3,R,R,R] grid holding (vx,vy,vz); init constant -> uniform v0=init_v0.
            grid = iv.view(1, 3, 1, 1, 1).expand(1, 3, res, res, res).clone()
            self.grid = nn.Parameter(grid)
        elif kind == "triplane":
            # 3 planes (xy, yz, xz), each [1, feat_dim, R, R]; decoder last layer
            # zero-WEIGHT init + bias=init_v0 so the field outputs init_v0 uniformly
            # at init (the spatial variation is learned on top via the planes).
            self.planes = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, feat_dim, res, res)) for _ in range(3)]
            )
            self.decoder = nn.Sequential(
                nn.Linear(3 * feat_dim, mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden, 3),
            )
            nn.init.zeros_(self.decoder[-1].weight)
            with torch.no_grad():
                self.decoder[-1].bias.copy_(iv)
        else:
            raise ValueError(f"unknown kind {kind!r} (global|voxel|triplane)")

    # ------------------------------------------------------------------ #
    def _normalise(self, pos: Tensor) -> Tensor:
        """[n,3] MPM coords -> [n,3] in [-1,1] (grid_sample convention)."""
        lo, hi = self.aabb[0], self.aabb[1]                     # [3], [3]
        p = 2.0 * (pos - lo) / (hi - lo + 1e-8) - 1.0
        return p.clamp(-1.0, 1.0)

    def forward(self, pos: Tensor) -> Tensor:
        """pos[n,3] -> v0[n,3] (physical normalised-space velocity, clamped)."""
        n = pos.shape[0]
        if self.kind == "global":
            v = self.vec[None, :].expand(n, 3)                  # [n,3]
        else:
            p = self._normalise(pos)                            # [n,3] in [-1,1]
            if self.kind == "voxel":
                # grid_sample 5D: grid coords last dim = (x,y,z) -> (W,H,D).
                g = p.view(1, n, 1, 1, 3)                       # [1,n,1,1,3]
                out = F.grid_sample(self.grid, g, mode="bilinear",
                                    align_corners=True)         # [1,3,n,1,1]
                v = out.reshape(3, n).t()                       # [n,3]
            else:  # triplane
                idx = ((0, 1), (1, 2), (0, 2))                  # xy, yz, xz
                feats = []
                for plane, (a, b) in zip(self.planes, idx):
                    uv = torch.stack([p[:, a], p[:, b]], dim=-1)  # [n,2]
                    uv = uv.view(1, n, 1, 2)                       # [1,n,1,2]
                    f = F.grid_sample(plane, uv, mode="bilinear",
                                      align_corners=True)          # [1,C,n,1]
                    feats.append(f.reshape(plane.shape[1], n).t())  # [n,C]
                v = self.decoder(torch.cat(feats, dim=-1))        # [n,3]
        v = v * self.out_scale
        if self.v_clamp is not None:
            v = v.clamp(-self.v_clamp, self.v_clamp)
        return v

    def v0_vec(self, pos: Tensor, query_mask: Tensor) -> Tensor:
        """pos[n,3], query_mask[n] bool -> v0[n,3] with 0 on non-query particles.

        Matches make_constant_v0: only the moving (query) particles carry v0; frozen
        / anchor particles stay at rest. The mask multiply keeps gradient flowing only
        through the moving particles (the only ones whose v0 is identifiable).
        """
        v = self.forward(pos)                                   # [n,3]
        return v * query_mask[:, None].to(v.dtype)              # [n,3]

    def regularization(self) -> Tensor:
        """Smoothness penalty (total variation) on the field params. Scalar.

        Global has no spatial structure -> 0. Voxel/triplane: mean |diff| along each
        spatial axis (same TV form as EField.regularization).
        """
        if self.kind == "global":
            return self.vec.new_zeros(())
        if self.kind == "voxel":
            g = self.grid                                       # [1,3,R,R,R]
            return (g.diff(dim=2).abs().mean()
                    + g.diff(dim=3).abs().mean()
                    + g.diff(dim=4).abs().mean())
        tv = self.planes[0].new_zeros(())
        for plane in self.planes:                               # each [1,C,R,R]
            tv = tv + plane.diff(dim=2).abs().mean() + plane.diff(dim=3).abs().mean()
        return tv
