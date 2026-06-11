"""Learnable spatial Young's-modulus field E(pos).

The v1 recovery (train_global_E) optimises a SINGLE global scalar log10(E); its
loss landscape is a narrow 1-D basin that is hard to fit. This module replaces the
scalar by a learnable FIELD with many parameters, on the hypothesis that a
better-conditioned, over-parameterised landscape is easier to optimise (and is the
substrate a later z-conditioned generative G will reuse).

Design choices (pinned against the project discussion):
  - Predicts ABSOLUTE log10(E) directly -- NOT a fixed-baseline + delta scheme.
    Every parameter is learnable; the only role of `init_E` is the INITIALISATION
    (the field starts as a ~uniform log10(init_E) so it matches the scalar start).
  - Output is log10(E) (positivity is then free, and it matches the rest of the
    pipeline which optimises in log10 space).
  - Backbone is swappable (voxel grid | triplane), selected by a string, so the
    voxel-vs-triplane A/B is one flag. The physics layer is untouched: the field
    just produces an E_vec[n] fed into the SAME validated per-particle grad path.

Positions in: normalised MPM coordinates (`scene.sim_xyzs`, [n,3]). The field
normalises them to [-1,1] via the scene's padded aabb before querying its grid.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

Backbone = Literal["voxel", "triplane"]


class EField(nn.Module):
    """pos[n,3] (normalised MPM coords) -> log10(E)[n], learnable.

    Args:
        aabb:        [2,3] padded aabb of the sim particles ([min; max] rows), used
                     to map query positions into [-1,1] for grid sampling.
        backbone:    "voxel" (3D grid of log10E, trilinear) or "triplane"
                     (3 feature planes + tiny MLP).
        init_E:      field initialises to ~uniform log10(init_E) everywhere.
        res:         grid/plane resolution.
        feat_dim:    per-plane feature channels (triplane only).
        mlp_hidden:  hidden width of the triplane decoder.
        log10_E_clamp: (lo, hi) hard clamp on output log10(E), a CFL/blow-up guard.
    """

    def __init__(
        self,
        aabb: Tensor,
        backbone: Backbone = "voxel",
        init_E: float = 1e5,
        res: int = 16,
        feat_dim: int = 16,
        mlp_hidden: int = 64,
        # CFL/identifiability-safe clamp: [1e4, ~1.4e6]. The old [1e3,1e8] let a
        # voxel cell drift to E=1e8 >> CFL limit -> blow-up/NaN during field opt.
        log10_E_clamp: tuple = (4.0, 6.15),
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.res = res
        self.register_buffer("aabb", aabb.detach().clone())  # [2,3]
        self.log10_lo, self.log10_hi = float(log10_E_clamp[0]), float(log10_E_clamp[1])
        init_log10 = float(torch.log10(torch.tensor(float(init_E))))

        if backbone == "voxel":
            # [1,1,R,R,R] grid holding log10(E); init constant -> uniform field.
            grid = torch.full((1, 1, res, res, res), init_log10, dtype=torch.float32)
            self.grid = nn.Parameter(grid)
        elif backbone == "triplane":
            # 3 planes (xy, yz, xz), each [1, feat_dim, R, R]; init 0 features so
            # the field output == the decoder's init bias == uniform log10(init_E).
            self.planes = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, feat_dim, res, res)) for _ in range(3)]
            )
            self.decoder = nn.Sequential(
                nn.Linear(3 * feat_dim, mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden, 1),
            )
            nn.init.zeros_(self.decoder[-1].weight)
            nn.init.constant_(self.decoder[-1].bias, init_log10)
        else:
            raise ValueError(f"unknown backbone {backbone!r} (voxel|triplane)")

    # ------------------------------------------------------------------ #
    def _normalise(self, pos: Tensor) -> Tensor:
        """[n,3] MPM coords -> [n,3] in [-1,1] (grid_sample convention)."""
        lo, hi = self.aabb[0], self.aabb[1]                     # [3], [3]
        p = 2.0 * (pos - lo) / (hi - lo + 1e-8) - 1.0
        return p.clamp(-1.0, 1.0)

    def forward(self, pos: Tensor) -> Tensor:
        """pos[n,3] -> log10(E)[n] (clamped to the CFL guard range)."""
        n = pos.shape[0]
        p = self._normalise(pos)                               # [n,3] in [-1,1]
        if self.backbone == "voxel":
            # grid_sample 5D: grid coords last dim = (x,y,z) -> (W,H,D).
            g = p.view(1, n, 1, 1, 3)                          # [1,n,1,1,3]
            out = F.grid_sample(self.grid, g, mode="bilinear",
                                align_corners=True)            # [1,1,n,1,1]
            logE = out.reshape(n)                              # [n]
        else:  # triplane
            # project to the 3 axis planes: xy, yz, xz
            idx = ((0, 1), (1, 2), (0, 2))
            feats = []
            for plane, (a, b) in zip(self.planes, idx):
                uv = torch.stack([p[:, a], p[:, b]], dim=-1)    # [n,2]
                uv = uv.view(1, n, 1, 2)                         # [1,n,1,2]
                f = F.grid_sample(plane, uv, mode="bilinear",
                                  align_corners=True)           # [1,C,n,1]
                feats.append(f.reshape(plane.shape[1], n).t())  # [n,C]
            logE = self.decoder(torch.cat(feats, dim=-1)).reshape(n)  # [n]
        return logE.clamp(self.log10_lo, self.log10_hi)

    def E_vec(self, pos: Tensor) -> Tensor:
        """pos[n,3] -> E[n] (absolute Young's modulus, = 10**log10E)."""
        return torch.pow(10.0, self.forward(pos))

    def regularization(self) -> Tensor:
        """Smoothness penalty (total variation) on the field parameters. Scalar."""
        if self.backbone == "voxel":
            g = self.grid                                       # [1,1,R,R,R]
            tv = (g.diff(dim=2).abs().mean()
                  + g.diff(dim=3).abs().mean()
                  + g.diff(dim=4).abs().mean())
            return tv
        tv = self.planes[0].new_zeros(())
        for plane in self.planes:                               # each [1,C,R,R]
            tv = tv + plane.diff(dim=2).abs().mean() + plane.diff(dim=3).abs().mean()
        return tv
