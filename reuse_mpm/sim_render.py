"""The single shared MPM-simulate-then-3DGS-render code path.

Both forward generation (this file, no_grad) and inverse training will call into
the *same* simulation + render routine, so pixel conventions match by
construction (this is what removes the "two render paths might disagree" risk).

Physics regime (mirrors render_trained_sim.py):
  - material = jelly, gravity = 0, density = 2000, grid damping = 1.1
  - nu (Poisson) fixed at 0.3
  - motion is driven entirely by an initial velocity v0 on the moving particles
    (no gravity), and its stiffness/response is governed by E.
So with v0 fixed and known, the video dynamics are a function of E alone --
exactly the identifiability setup we want for recovering p*(E).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
import warp as wp

from . import _env
from ._env import (
    Camera,
    GaussianModel,
    MPMModelStruct,
    MPMStateStruct,
    MPMWARPDiff,
    apply_grid_bc_w_freeze_pts,
    render_gaussian_seq_w_mask_with_disp,
)
from .scene import SceneBundle
from .config import SimConfig  # re-exported: callers may `from .sim_render import SimConfig`

_WARP_INITED = False


def _ensure_warp() -> None:
    global _WARP_INITED
    if not _WARP_INITED:
        wp.init()
        _WARP_INITED = True


class RenderPipe:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False


class _RenderParams:
    def __init__(
        self,
        gaussians: GaussianModel,
        camera_list: List[Camera],
        bg_color: torch.Tensor,
    ) -> None:
        self.render_pipe = RenderPipe()
        self.bg_color = bg_color
        self.gaussians = gaussians
        self.camera_list = camera_list


def make_constant_v0(
    scene: SceneBundle, v0_vec: Union[Sequence[float], torch.Tensor]
) -> torch.Tensor:
    """Constant initial velocity applied to moving (query) particles, 0 elsewhere.

    NOTE: matches render_trained_sim's `v = velo_field * 0.1` convention is NOT
    applied here -- v0_vec is the *physical* normalised-space velocity directly.
    Keep v0 identical between forward-gen and inverse so only E differs.
    """
    device = scene.device
    n = scene.sim_xyzs.shape[0]
    v = torch.zeros((n, 3), dtype=torch.float32, device=device)
    qm = scene.query_mask
    v0_vec = torch.as_tensor(v0_vec, dtype=torch.float32, device=device)
    v[qm] = v0_vec[None, :]
    return v


def build_mpm(
    scene: SceneBundle, cfg: SimConfig, requires_grad: bool = False
) -> Tuple[MPMWARPDiff, MPMStateStruct, MPMModelStruct]:
    """Construct (solver, state, model) for a scene: material params, volume and
    freeze BC set, but E / v0 / rollout NOT applied.

    Shared by forward-gen (no_grad) and the differentiable training path, so the
    physics setup is identical in both.
    """
    _ensure_warp()
    device = scene.device
    sim_xyzs = scene.sim_xyzs
    n = sim_xyzs.shape[0]

    state = MPMStateStruct()
    state.init(n, device=device, requires_grad=requires_grad)
    state.from_torch(
        sim_xyzs.clone(),
        torch.from_numpy(scene.points_vol).float().to(device),
        None,
        device=device,
        requires_grad=requires_grad,
        n_grid=cfg.grid_size,
        grid_lim=cfg.grid_lim,
    )
    model = MPMModelStruct()
    model.init(n, device=device, requires_grad=requires_grad)
    model.init_other_params(n_grid=cfg.grid_size, grid_lim=cfg.grid_lim, device=device)
    solver = MPMWARPDiff(n, n_grid=cfg.grid_size, grid_lim=cfg.grid_lim, device=device)
    solver.set_parameters_dict(
        model,
        state,
        {
            "material": cfg.material,
            "g": [0.0, 0.0, 0.0],
            "density": cfg.density,
            "grid_v_damping_scale": cfg.grid_v_damping_scale,
        },
    )
    freeze_pts = sim_xyzs[scene.freeze_mask, :]
    apply_grid_bc_w_freeze_pts(cfg.grid_size, 1.0, freeze_pts, solver)
    return solver, state, model


def simulate_positions(
    scene: SceneBundle,
    E: Union[float, torch.Tensor],
    v0: torch.Tensor,
    cfg: SimConfig,
    requires_grad: bool = False,
) -> List[torch.Tensor]:
    """Run MPM forward; return list of per-frame particle positions (un-normalised).

    E may be a python float (constant) or a [n] tensor (per-particle field).
    With requires_grad=False this is the forward-gen path (mirrors
    render_trained_sim). The differentiable path is added later for training.

    Returns:
        pos_list: [(M, 3), ...] (T times)
    """
    assert not requires_grad, "differentiable path not implemented yet (Task #4)"
    device = scene.device
    sim_xyzs = scene.sim_xyzs
    n = sim_xyzs.shape[0]

    solver, state, model = build_mpm(scene, cfg, requires_grad=False)

    density = torch.ones_like(sim_xyzs[..., 0]) * cfg.density
    state.reset_density(
        density.clone(),
        torch.ones_like(density).type(torch.int),
        device,
        update_mass=True,
    )

    init_xyzs = sim_xyzs.clone()

    with torch.no_grad():
        if isinstance(E, (int, float)):
            E_t = torch.ones_like(init_xyzs[..., 0]) * float(E)
        else:
            E_t = E.to(device).reshape(-1)
        E_t = E_t.clamp(1.0, 5e8)
        nu_t = torch.ones_like(init_xyzs[..., 0]) * cfg.nu

        solver.set_E_nu_from_torch(model, E_t.clone(), nu_t.clone(), device)
        solver.prepare_mu_lam(model, state, device)

        I_mat = torch.eye(3, dtype=torch.float32, device=device)
        F = I_mat[None, ...].repeat(n, 1, 1)
        C = torch.zeros_like(F)
        state.continue_from_torch(
            init_xyzs, v0, F, C, device=device, requires_grad=False
        )

        sub_dt = cfg.substep_size
        pos_list = [(init_xyzs.clone() * scene.scale) - scene.shift]
        prev = state
        for i in range(cfg.num_frames - 1):
            for _ in range(cfg.substep):
                nxt: MPMStateStruct = prev.partial_clone(requires_grad=False)
                solver.p2g2p_differentiable(model, prev, nxt, sub_dt, device=device)
                prev = nxt
            pos = wp.to_torch(nxt.particle_x).clone()
            pos = (pos * scene.scale) - scene.shift  # [notes] mpm to 3dgs coords
            pos_list.append(pos)

    return pos_list


def render_positions(
    scene: SceneBundle,
    pos_list: List[torch.Tensor],
    cam: Camera,
) -> torch.Tensor:
    """Render a sequence of particle positions from one (fixed) camera.

    Returns video tensor [T, C, H, W] in [0, 1].
    """
    device = scene.device
    init_pos = pos_list[0]
    pos_diff_list = [p - init_pos for p in pos_list]
    bg = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    rp = _RenderParams(scene.gaussians, scene.test_camera_list, bg)
    cams = [cam] * len(pos_list)
    vid = render_gaussian_seq_w_mask_with_disp(
        cams, rp, init_pos, scene.top_k_index, pos_diff_list, scene.sim_mask
    )
    return vid


def render_disp_frame(
    scene: SceneBundle,
    particle_pos_normalised: torch.Tensor,
    cam: Camera,
) -> torch.Tensor:
    """Differentiable render of ONE frame from normalised particle positions.

    Mirrors fast_train_velocity's render block: convert to world space, form a
    displacement vs the undeformed gaussians, render through the KNN disp render.
    Gradient flows through `particle_pos_normalised`.
    Returns [1, C, H, W].
    """
    device = scene.device
    world_pos = particle_pos_normalised * scene.scale - scene.shift
    undeformed = (scene.sim_xyzs * scene.scale - scene.shift).detach()
    disp = world_pos - undeformed
    bg = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    rp = _RenderParams(scene.gaussians, scene.test_camera_list, bg)
    vid = render_gaussian_seq_w_mask_with_disp(
        cam, rp, undeformed, scene.top_k_index, [disp], scene.sim_mask
    )
    return vid


def simulate_and_render(
    scene: SceneBundle,
    E: Union[float, torch.Tensor],
    v0: torch.Tensor,  # (m, 3)
    cfg: SimConfig,
    cam: Camera,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Forward: (E, v0) -> video [T, C, H, W] in [0,1]. The shared code path.
    
    Returns:
        pos_list: [(M, 3), ...] (T times)
    """
    pos_list = simulate_positions(scene, E, v0, cfg, requires_grad=requires_grad)
    return render_positions(scene, pos_list, cam)


def video_to_uint8(vid: torch.Tensor) -> np.ndarray:
    """[T,C,H,W] in [0,1] -> [T,H,W,C] uint8."""
    v = (vid.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return np.transpose(v, [0, 2, 3, 1])
