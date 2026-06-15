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
    """Attribute bag for PhysDreamer's render fns: they read only gaussians,
    render_pipe and bg_color (camera comes in as a separate positional arg, so no
    camera_list here -- verified against local_utils.render_gaussian_seq_*)."""

    def __init__(
        self,
        gaussians: GaussianModel,
        bg_color: torch.Tensor,
    ) -> None:
        self.render_pipe = RenderPipe()
        self.bg_color = bg_color
        self.gaussians = gaussians


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


def make_gradient_E(
    scene: SceneBundle, E_base: float, axis: int, decades: float
) -> torch.Tensor:
    """A smooth monotonic per-particle E field for spatially-varying GT (phase B).

    E[i] = E_base * 10**(decades * (frac_i - 0.5)), where frac_i in [0,1] is the
    particle's position along `axis` over the sim aabb. So the geometric mean stays
    ~E_base and the field spans `decades` decades of log10(E) end-to-end -- a known,
    serialisable ground truth the field-recovery (train_field_E) must reproduce.

    Returns:
        E_vec: [n] per-particle Young's modulus.
    """
    pos = scene.sim_xyzs                                        # [n,3]
    lo, hi = scene.sim_aabb[0], scene.sim_aabb[1]              # [3], [3]
    frac = ((pos[:, axis] - lo[axis]) / (hi[axis] - lo[axis] + 1e-8)).clamp(0, 1)  # [n]
    return E_base * torch.pow(10.0, decades * (frac - 0.5))     # [n]


def make_gradient_v0(
    scene: SceneBundle, base_vec: Union[Sequence[float], torch.Tensor],
    axis: int, slope: float
) -> torch.Tensor:
    """A smooth per-particle v0 field for spatially-varying GT (phase B, v0 dual).

    v0[i] = base_vec * (1 + slope * (frac_i - 0.5)) on moving (query) particles, 0
    elsewhere, where frac_i in [0,1] is the particle's position along `axis` over the
    sim aabb. Direction is fixed (base_vec); only the magnitude ramps linearly along
    `axis`, spanning base*(1-slope/2)..base*(1+slope/2) end-to-end. The mean over the
    axis stays ~base_vec, so a UNIFORM v0 recovers the mean but misses the spatial
    ramp -- a field must reproduce it. A known, serialisable ground truth.

    Returns:
        v0: [n,3] per-particle initial velocity (0 on non-query particles).
    """
    device = scene.device
    n = scene.sim_xyzs.shape[0]
    base = torch.as_tensor(base_vec, dtype=torch.float32, device=device)  # [3]
    pos = scene.sim_xyzs                                        # [n,3]
    lo, hi = scene.sim_aabb[0], scene.sim_aabb[1]              # [3], [3]
    frac = ((pos[:, axis] - lo[axis]) / (hi[axis] - lo[axis] + 1e-8)).clamp(0, 1)  # [n]
    scale = (1.0 + slope * (frac - 0.5))                       # [n]
    v = scale[:, None] * base[None, :]                         # [n,3]
    v = v * scene.query_mask[:, None].to(v.dtype)             # zero non-query
    return v


def build_mpm(
    scene: SceneBundle, cfg: SimConfig, requires_grad: bool = False
) -> Tuple[MPMWARPDiff, MPMStateStruct, MPMModelStruct]:
    """Construct (solver, state, model) for a scene: material params, volume and
    particle-level freeze set, but E / v0 / rollout NOT applied.

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
    # Freeze anchor particles via a grid BC that zeroes each frozen particle's FULL
    # 27-node G2P stencil. PhysDreamer's apply_grid_bc_w_freeze_pts marks only the
    # single floor node floor(pos*inv_dx) per frozen pt, but g2p gathers velocity
    # from a 27-node quadratic B-spline stencil (base = int(pos*inv_dx - 0.5), i,j,k
    # in 0..2); the un-marked neighbour nodes carry the moving material's velocity,
    # so anchor particles leaked ~30% of free motion (telephone). Marking every
    # stencil node a frozen particle reads makes its gathered velocity exactly 0 ->
    # it stays put. (particle_selection is unusable here: the partial_clone rollout
    # does not carry a skipped particle's x to the next state.) See explore/freeze_probe.
    G = cfg.grid_size
    inv_dx = G / cfg.grid_lim
    base = (sim_xyzs[scene.freeze_mask, :] * inv_dx - 0.5).to(torch.int64)  # [F,3]
    freeze_grid = torch.zeros((G, G, G), dtype=torch.int32, device=device)
    for di in range(3):
        for dj in range(3):
            for dk in range(3):
                ix = (base[:, 0] + di).clamp(0, G - 1)
                iy = (base[:, 1] + dj).clamp(0, G - 1)
                iz = (base[:, 2] + dk).clamp(0, G - 1)
                freeze_grid[ix, iy, iz] = 1
    solver.enforce_grid_velocity_by_mask(freeze_grid)
    return solver, state, model


def wall_contact_count(pos_normalized: torch.Tensor, cfg: SimConfig) -> int:
    """Count particles touching the g2p position clamp (warp-wall-clamp).

    pos_normalized: [n, 3] particle positions in NORMALIZED sim coords.

    The vendored g2p hard-clamps positions per component to
    [2*dx, grid_lim - 2*dx] without zeroing velocity (PhysDreamer
    warp_mpm/mpm_utils.py:554-565), so a contact pins the coordinate
    bit-exactly at the bound: positions AND gradients of such particles are
    invalid from that substep on. `<=`/`>=` is therefore an exact signature,
    not a tolerance.

    Returns the number of particles with any clamped component.
    """
    dx = cfg.grid_lim / cfg.grid_size
    lo, hi = 2.0 * dx, cfg.grid_lim - 2.0 * dx
    with torch.no_grad():
        return int(((pos_normalized <= lo) | (pos_normalized >= hi)).any(dim=-1).sum().item())


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
        wall_frames: List[tuple] = []  # (frame, n_pinned)
        for i in range(cfg.num_frames - 1):
            for _ in range(cfg.substep):
                nxt: MPMStateStruct = prev.partial_clone(requires_grad=False)
                solver.p2g2p_differentiable(model, prev, nxt, sub_dt, device=device)
                prev = nxt
            pos = wp.to_torch(nxt.particle_x).clone()
            n_wall = wall_contact_count(pos, cfg)
            if n_wall:
                wall_frames.append((i + 1, n_wall))
            pos = (pos * scene.scale) - scene.shift  # [notes] mpm to 3dgs coords
            pos_list.append(pos)
        if wall_frames:
            print(f"[sim] WARNING: wall contact in {len(wall_frames)} frame(s) -- particles "
                  f"pinned at the g2p position clamp, (frame, count): {wall_frames[:6]}"
                  f"{'...' if len(wall_frames) > 6 else ''}. This run's dynamics are "
                  "invalid near the wall (warp-wall-clamp); reject or re-center.")

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
    rp = _RenderParams(scene.gaussians, bg)
    cams = [cam] * len(pos_list)
    vid = render_gaussian_seq_w_mask_with_disp(
        cams, rp, init_pos, scene.top_k_index, pos_diff_list, scene.sim_mask
    )
    return vid


def render_positions_multicam(
    scene: SceneBundle,
    pos_list: List[torch.Tensor],
    cams: List[Camera],
) -> torch.Tensor:
    """Render a position sequence with a PER-FRAME camera (dynamic / moving cam).

    Same as render_positions but the viewpoint changes each frame: cams[t] renders
    pos_list[t]. Enables a moving camera within one clip (orbit/dolly), which the
    underlying render already supports (it takes a per-frame cam list). The
    displacement field is still computed vs the rest pose pos_list[0].

    Args:
        pos_list: T x [n, 3] world particle positions.
        cams:     T cameras, one per frame (len must equal len(pos_list)).
    Returns:
        video tensor [T, C, H, W] in [0, 1].
    """
    assert len(cams) == len(pos_list), f"{len(cams)} cams != {len(pos_list)} frames"
    device = scene.device
    init_pos = pos_list[0]
    pos_diff_list = [p - init_pos for p in pos_list]
    bg = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    rp = _RenderParams(scene.gaussians, bg)
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
    rp = _RenderParams(scene.gaussians, bg)
    # [note] this render all gs in `render_params.gaussians`, including non MPM particles.
    # Recap: four types of particles:
    # a. ~clean = ~simulated = background. 
    #    Do not participate MPM (both rollout & interpolate)
    # b. clean. 
    #    Do not participate MPM rollout, but displaced by interpolation.
    # c. knn centers & freeze_mask.
    #    Participate MPM rollout, but fixed by BC.
    # d. knn centers & ~freeze_mask.
    #    The only particle that can be moved in MPM
    # 
    # To be clear, `a+b` = all 3dgs. c, d are "virtual" particles participating MPM, but themselves do not have appearance.

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


def render_static_subset(
    scene: SceneBundle, cam: Camera, keep_gauss_mask: torch.Tensor
) -> np.ndarray:
    """Render the REST-POSE scene using only `keep_gauss_mask` gaussians (the rest
    made transparent), to see a subset in isolation -- e.g. ~sim_mask = the static
    background (incl. any "wall") that MPM never simulates.

    Args:
        keep_gauss_mask: [N_gauss] bool -- gaussians to keep visible.
    Returns:
        [H, W, C] uint8 single frame.
    """
    g = scene.gaussians
    old_op = g._opacity.clone()                                  # [N_gauss, 1]
    try:
        with torch.no_grad():
            g._opacity[~keep_gauss_mask] = -1e4                  # sigmoid(-1e4) ~ 0 -> invisible
            pos0 = (scene.sim_xyzs * scene.scale - scene.shift).detach()  # [n, 3] world
            vid = render_positions(scene, [pos0], cam)           # [1, C, H, W]
        return video_to_uint8(vid)[0]                            # [H, W, C] uint8
    finally:
        g._opacity = old_op


def render_positions_recolor(
    scene: SceneBundle, pos_list: List[torch.Tensor], cam: Camera,
    red_gauss_mask: torch.Tensor,
) -> torch.Tensor:
    """render_positions, but with `red_gauss_mask` gaussians forced to opaque,
    view-independent RED -- overlays a gaussian-level mask onto the moving render
    (e.g. 'all-KNN-freeze' gaussians, to SEE the fully-anchored region in the gif).

    Args:
        pos_list:       T x [n, 3] world particle positions (the motion to render).
        red_gauss_mask: [N_gauss] bool -- gaussians to paint red.
    Returns:
        [T, C, H, W] in [0,1].
    """
    g = scene.gaussians
    sh_c0 = 0.28209479177387814                                  # SH band-0 constant
    old_dc = g._features_dc.clone()                              # [N_gauss, 1, 3]
    old_rest = g._features_rest.clone()                          # [N_gauss, R, 3]
    try:
        with torch.no_grad():
            dev, dt = g._features_dc.device, g._features_dc.dtype
            # rendered colour ~ sh_c0 * dc + 0.5; solve for rgb=(1,0,0)
            red_dc = torch.tensor([0.5 / sh_c0, -0.5 / sh_c0, -0.5 / sh_c0],
                                  device=dev, dtype=dt)           # [3]
            g._features_dc[red_gauss_mask] = red_dc.view(1, 1, 3)
            g._features_rest[red_gauss_mask] = 0.0                # kill view-dependence
            vid = render_positions(scene, pos_list, cam)         # [T, C, H, W]
        return vid
    finally:
        g._features_dc = old_dc
        g._features_rest = old_rest


def video_to_uint8(vid: torch.Tensor) -> np.ndarray:
    """[T,C,H,W] in [0,1] -> [T,H,W,C] uint8."""
    v = (vid.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return np.transpose(v, [0, 2, 3, 1])
