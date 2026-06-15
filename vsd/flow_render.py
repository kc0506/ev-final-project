"""Differentiable screen-flow renderer: MPM world trajectory -> packed [0,1] flow,
matching build_flow_pack.py's convention so the output feeds the flow teacher exactly.

build_flow_pack is non-differentiable (round-to-pixel scatter + iterative hole fill).
Here both the splat LOCATION and VALUE are differentiable: bilinear soft-splat instead
of round, and a conv-based hole fill instead of the python neighbour loop. The pinhole
projection and the [0,1] packing (disp_px = (x-0.5)*2*scale_px) are identical.

No 3DGS / gaussian rasterizer involved -- screen flow is just particle projection.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from vsd.scene_min import CameraParams


def project_to_res(X_world: Tensor, cam: CameraParams, res: int,
                   zoom: float = 1.0) -> Tuple[Tensor, Tensor]:
    """world [T,n,3] -> (uv [T,n,2] in [0,res) pixel coords at the res grid, zv [T,n]).
    GS/OpenCV pinhole (mirrors flow_viz.project), then scaled W,H -> res.

    `zoom` < 1 shrinks the object about the frame centre (equivalent to dollying the
    camera back / a smaller effective focal length) so large motion stays in-frame; the
    on-screen flow magnitude scales by `zoom` too. zoom=1 = the dataset's real camera.
    """
    T, n, _ = X_world.shape
    ones = torch.ones(T, n, 1, device=X_world.device, dtype=X_world.dtype)  # [T,n,1]
    Xh = torch.cat([X_world, ones], dim=-1)                                  # [T,n,4]
    view = Xh @ cam.world_view_transform                                     # [T,n,4]
    xv, yv, zv = view[..., 0], view[..., 1], view[..., 2]                    # [T,n] each
    W, H = cam.width, cam.height
    fx = W / (2 * torch.tan(torch.tensor(cam.fovx / 2)))
    fy = H / (2 * torch.tan(torch.tensor(cam.fovy / 2)))
    u = (fx * xv / zv + W / 2.0) * (res / W)                                 # [T,n] -> res grid
    v = (fy * yv / zv + H / 2.0) * (res / H)                                 # [T,n]
    if zoom != 1.0:                                                          # shrink about centre
        u = (u - res / 2.0) * zoom + res / 2.0
        v = (v - res / 2.0) * zoom + res / 2.0
    return torch.stack([u, v], dim=-1), zv                                   # [T,n,2], [T,n]


def soft_splat(uv_src: Tensor, disp: Tensor, valid: Tensor, res: int) -> Tuple[Tensor, Tensor]:
    """Bilinear-splat per-particle displacement onto a dense grid (differentiable).

    uv_src [n,2] source-pixel coords (res grid); disp [n,2] displacement to splat;
    valid [n] bool. Returns (flow [2,res,res], covered [res,res] in [0,1] weight mass).
    """
    n = uv_src.shape[0]
    acc = uv_src.new_zeros(res * res, 2)        # [res*res, 2] weighted displacement
    cnt = uv_src.new_zeros(res * res)           # [res*res]    weight mass
    u, v = uv_src[:, 0], uv_src[:, 1]           # [n], [n]
    x0 = torch.floor(u); y0 = torch.floor(v)    # [n]
    wx = u - x0; wy = v - y0                     # [n] bilinear fracs
    vmask = valid.float()                        # [n]
    for cx, wxx in ((0, 1 - wx), (1, wx)):
        for cy, wyy in ((0, 1 - wy), (1, wy)):
            ix = (x0 + cx).long(); iy = (y0 + cy).long()                      # [n]
            inb = (ix >= 0) & (ix < res) & (iy >= 0) & (iy < res)             # [n]
            w = (wxx * wyy) * vmask * inb.float()                             # [n] corner weight
            ixc = ix.clamp(0, res - 1); iyc = iy.clamp(0, res - 1)
            flat = iyc * res + ixc                                            # [n] row-major
            acc.index_add_(0, flat, disp * w[:, None])
            cnt.index_add_(0, flat, w)
    flow = acc / cnt.clamp(min=1e-8)[:, None]                                 # [res*res,2]
    flow = flow.view(res, res, 2).permute(2, 0, 1).contiguous()              # [2,res,res]
    covered = (cnt.view(res, res) > 1e-6).float()                            # [res,res]
    return flow, covered


def fill_holes(flow: Tensor, covered: Tensor, iters: int = 8) -> Tensor:
    """Differentiable hole fill: unset pixels <- mean of set 8-neighbours, repeated.
    Background (never reached) stays 0. flow [2,res,res], covered [res,res]."""
    k = flow.new_ones(1, 1, 3, 3)                                            # 8-neighbour + self box
    f = flow.unsqueeze(0)                                                    # [1,2,res,res]
    m = covered.view(1, 1, *covered.shape)                                   # [1,1,res,res]
    for _ in range(iters):
        fsum = F.conv2d(f * m, k.expand(2, 1, 3, 3), padding=1, groups=2)    # [1,2,res,res]
        msum = F.conv2d(m, k, padding=1)                                     # [1,1,res,res]
        newly = ((m < 0.5) & (msum > 0)).float()                            # [1,1,res,res]
        filled = fsum / msum.clamp(min=1e-8)                                 # [1,2,res,res]
        f = f * m + filled * newly                                          # keep set, add newly
        m = (m + newly).clamp(max=1.0)
    return f.squeeze(0)                                                      # [2,res,res]


def render_flow(world_traj: Tensor, cam: CameraParams, scale_px: float, res: int,
                fill_iters: int = 8) -> Tensor:
    """world_traj [T,n,3] (MOVING particles, world coords) -> packed flow [T-1,2,res,res]
    in [0,1], matching the teacher's pack. Encoding: x = disp_px/(2*scale_px) + 0.5."""
    uv, zv = project_to_res(world_traj, cam, res)                            # [T,n,2], [T,n]
    T = world_traj.shape[0]
    fields = []
    for t in range(T - 1):
        disp = uv[t + 1] - uv[t]                                             # [n,2] px at res scale
        inb = ((uv[t, :, 0] >= 0) & (uv[t, :, 0] < res) &
               (uv[t, :, 1] >= 0) & (uv[t, :, 1] < res) &
               (zv[t] > 0) & (zv[t + 1] > 0))                               # [n]
        flow, covered = soft_splat(uv[t], disp, inb, res)                    # [2,res,res],[res,res]
        flow = fill_holes(flow, covered, iters=fill_iters)                   # [2,res,res]
        packed = (flow / (2 * scale_px) + 0.5).clamp(0.0, 1.0)               # [2,res,res] in [0,1]
        fields.append(packed)
    return torch.stack(fields, dim=0)                                        # [T-1,2,res,res]
