"""Shared block-scene machinery for the F0 sys-id explore tools.

A `Scene` bundles everything scene-specific: geometry, solver build, F0-snapshot
generation (a symmetric x-pull OR an asymmetric downward squeeze vs a floor), and a
reusable release rollout with scene-dependent gravity/floor. Both f0_fit_case (fit)
and f0_loss_landscape (analysis) build a Scene by name and only ever call
`sc.rollout(logE, K)` -- so ADDING A SCENE = one entry in SCENES + (if new) its f0
method here, nothing in the consumers.

  SCENES[name] = (f0_method, release_gravity_on, release_floor_on)
    release  : pull,    g=0,  no floor   -- displacement-controlled, pure frequency channel
    drop     : pull,    g!=0, floor      -- falls + collides; contact force => amplitude channel
    freefall : pull,    g!=0, no floor   -- gravity but inertial (no contact) => no E signal
    squeeze  : squeeze, g=0,  floor      -- asym downward press vs floor; force => amplitude channel

Exposed on a Scene: X_rest, n, cx/cy/cz/hx/hy/hz, floor_z, has_floor, scene_name,
x_snap, F_snap (torch), maxdev (float), F0_stretch (np), pull_X/pull_S (lists),
and rollout(logE, K) -> (traj[K+1,n,3], stretch[K+1,n]) (both torch on device).

Caller must have run `wp.init()` before constructing a Scene (warp is imported lazily).
"""
from __future__ import annotations

SCENES = {
    #            f0 method   release gravity?  release floor?
    "release":  ("pull",     False,            False),
    "drop":     ("pull",     True,             True),
    "freefall": ("pull",     True,             False),
    "squeeze":  ("squeeze",  False,            True),
    "uniform":  ("uniform",  False,            False),   # hand-set homogeneous V0=expm(S), pure release
}


def sym3_from_6(torch, s6):
    """Symmetric 3x3 from 6-vector [xx, yy, zz, xy, xz, yz]."""
    s = [float(v) for v in s6]
    return torch.tensor([[s[0], s[3], s[4]], [s[3], s[1], s[5]], [s[4], s[5], s[2]]],
                        dtype=torch.float32)


def v0_from_s6(torch, s6, device="cpu"):
    """Left-stretch V0 = expm(S), S symmetric (log-Euclidean). Always SPD; s6=0 -> I."""
    return torch.linalg.matrix_exp(sym3_from_6(torch, s6).to(device))


class Scene:
    def __init__(self, scene, *, nx=22, ny=9, nz=16, half=(0.18, 0.08, 0.14), z_base=0.30,
                 nu=0.3, gt_logE=4.5,
                 pull_speed=0.5, release_frame=5, grip_half_x=0.045,
                 push_x=0.60, push_half_x=0.07, push_half_z=0.045, push_speed=0.45, push_frames=5,
                 S_gt=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 gravity=9.8, floor_z=0.25, collider="slip", friction=0.0, device="cuda:0"):
        import gc
        import torch
        import warp as wp
        from .._env import MPMStateStruct, MPMModelStruct, MPMWARPDiff
        from ..config import SimConfig
        assert scene in SCENES, f"unknown scene {scene!r} (have {list(SCENES)})"
        f0_method, rel_grav, rel_floor = SCENES[scene]
        self._torch, self._wp, self._gc = torch, wp, gc
        self._E = (MPMStateStruct, MPMModelStruct, MPMWARPDiff)
        self.scene_name = scene
        sim = SimConfig(); self._sim = sim; self._G, self._GL = sim.grid_size, sim.grid_lim
        self.nu, self.gt_logE, self.device = nu, gt_logE, device
        self.gravity, self.collider, self.friction = gravity, collider, friction
        self.has_floor = rel_floor
        # squeeze presses against the block's own base; drop's floor is a gap below it
        self.floor_z = z_base if scene == "squeeze" else floor_z

        hx, hy, hz = half; cx, cy, cz = 0.5, 0.5, z_base + hz
        self.hx, self.hy, self.hz, self.cx, self.cy, self.cz, self.z_base = hx, hy, hz, cx, cy, cz, z_base
        gx = torch.linspace(cx - hx, cx + hx, nx); gy = torch.linspace(cy - hy, cy + hy, ny)
        gz = torch.linspace(cz - hz, cz + hz, nz)
        X_rest = torch.stack(torch.meshgrid(gx, gy, gz, indexing="ij"), -1).reshape(-1, 3).to(device)
        self.X_rest = X_rest; self.n = n = X_rest.shape[0]
        self._p_vol = torch.full((n,), float((2 * hx / max(nx - 1, 1)) ** 3), device=device)
        self._eye = torch.eye(3, device=device)
        self._z3 = torch.zeros(n, 3, device=device); self._z33 = torch.zeros(n, 3, 3, device=device)

        if f0_method == "pull":
            self._make_pull_f0(pull_speed, release_frame, grip_half_x)
        elif f0_method == "squeeze":
            self._make_squeeze_f0(push_x, push_half_x, push_half_z, push_speed, push_frames)
        else:
            self._make_uniform_f0(S_gt)

        g_vec = (0.0, 0.0, -gravity) if rel_grav else (0.0, 0.0, 0.0)
        self._rsv, self._rst, self._rmd = self._build(g=g_vec, floor=rel_floor)

    # ---- low-level ----
    def _build(self, g=(0.0, 0.0, 0.0), floor=False):
        torch, wp, dev = self._torch, self._wp, self.device
        MPMStateStruct, MPMModelStruct, MPMWARPDiff = self._E
        n, G, GL, sim = self.n, self._G, self._GL, self._sim
        st = MPMStateStruct(); st.init(n, device=dev, requires_grad=False)
        st.from_torch(self.X_rest.clone(), self._p_vol, None, device=dev, requires_grad=False, n_grid=G, grid_lim=GL)
        md = MPMModelStruct(); md.init(n, device=dev, requires_grad=False)
        md.init_other_params(n_grid=G, grid_lim=GL, device=dev)
        sv = MPMWARPDiff(n, n_grid=G, grid_lim=GL, device=dev)
        sv.set_parameters_dict(md, st, {"material": sim.material, "g": list(g),
                               "density": sim.density, "grid_v_damping_scale": sim.grid_v_damping_scale})
        if floor:
            sv.add_surface_collider(point=(0.0, 0.0, self.floor_z), normal=(0.0, 0.0, 1.0),
                                    surface=self.collider, friction=self.friction)
        st.reset_density(torch.full((n,), float(sim.density), device=dev).clone(),
                         torch.ones(n, device=dev).int(), dev, update_mass=True)
        return sv, st, md

    def _setE(self, sv, md, st, logE):
        torch, dev, n = self._torch, self.device, self.n
        sv.set_E_nu_from_torch(md, torch.full((n,), float(10.0 ** logE), device=dev).clone(),
                               torch.full((n,), float(self.nu), device=dev).clone(), dev)
        sv.prepare_mu_lam(md, st, dev)

    def _stretch(self, F):
        return (self._torch.linalg.svdvals(F) - 1.0).abs().amax(1)

    def _record_snapshot(self, sv, st, md, n_frames):
        """Step the (already grip-configured) solver n_frames, recording x/stretch; snapshot the last F."""
        torch, wp, sim, gc = self._torch, self._wp, self._sim, self._gc
        dev, n = self.device, self.n
        prev = st; xs = [wp.to_torch(prev.particle_x).clone()]; ss = [torch.zeros(n, device=dev)]
        for _ in range(n_frames):
            for _ in range(sim.substep):
                nx = prev.partial_clone(requires_grad=False)
                sv.p2g2p_differentiable(md, prev, nx, sim.substep_size, device=dev); prev = nx
            xs.append(wp.to_torch(prev.particle_x).clone())
            ss.append(self._stretch(wp.to_torch(prev.particle_F_trial)).clone())
        self.x_snap = wp.to_torch(prev.particle_x).clone()
        self.F_snap = wp.to_torch(prev.particle_F_trial).clone()
        self.maxdev = float((self.x_snap - self.X_rest).norm(dim=1).max())
        self.F0_stretch = self._stretch(self.F_snap).cpu().numpy()
        self.pull_X, self.pull_S = xs, ss
        del prev; gc.collect()

    # ---- F0 generators ----
    def _make_pull_f0(self, pull_speed, release_frame, grip_half_x):
        torch, dev, sim = self._torch, self.device, self._sim
        cx, cy, cz, hy, hz = self.cx, self.cy, self.cz, self.hy, self.hz
        sv, st, md = self._build()
        self._setE(sv, md, st, self.gt_logE); sv.time = 0.0
        with torch.no_grad():
            st.continue_from_torch(self.X_rest.clone(), self._z3, self._eye[None].repeat(self.n, 1, 1).contiguous(),
                                   self._z33, device=dev, requires_grad=False)
            et = release_frame * sim.delta_t; gs = (grip_half_x, hy * 1.6, hz * 1.6)
            sv.enforce_particle_velocity_translation(st, point=(cx - self.hx, cy, cz), size=gs,
                velocity=(-pull_speed, 0, 0), start_time=0.0, end_time=et, device=dev)
            sv.enforce_particle_velocity_translation(st, point=(cx + self.hx, cy, cz), size=gs,
                velocity=(+pull_speed, 0, 0), start_time=0.0, end_time=et, device=dev)
            self._record_snapshot(sv, st, md, release_frame)

    def _make_squeeze_f0(self, push_x, push_half_x, push_half_z, push_speed, push_frames):
        torch, dev, sim = self._torch, self.device, self._sim
        cy, cz, hy, hz = self.cy, self.cz, self.hy, self.hz
        sv, st, md = self._build(floor=True)   # floor present so the press has something to react against
        self._setE(sv, md, st, self.gt_logE); sv.time = 0.0
        with torch.no_grad():
            st.continue_from_torch(self.X_rest.clone(), self._z3, self._eye[None].repeat(self.n, 1, 1).contiguous(),
                                   self._z33, device=dev, requires_grad=False)
            et = push_frames * sim.delta_t
            sv.enforce_particle_velocity_translation(st, point=(push_x, cy, cz + hz),
                size=(push_half_x, hy * 1.6, push_half_z), velocity=(0.0, 0.0, -push_speed),
                start_time=0.0, end_time=et, device=dev)
            self._record_snapshot(sv, st, md, push_frames)

    def _make_uniform_f0(self, S_gt):
        # homogeneous deformation: F0 = V0 = expm(S) everywhere, positions = affine map of rest
        # about the centroid (COMPATIBLE, i.e. a real uniform deformation -- not eigenstrain).
        torch = self._torch
        V0 = v0_from_s6(torch, S_gt, device=self.device)
        c = self.X_rest.mean(0)
        self.x_snap = (c + (self.X_rest - c) @ V0.T).contiguous()
        self.F_snap = V0[None].repeat(self.n, 1, 1).contiguous()
        self.maxdev = float((self.x_snap - self.X_rest).norm(dim=1).max())
        self.F0_stretch = self._stretch(self.F_snap).cpu().numpy()
        self.pull_X = [self.X_rest.clone(), self.x_snap.clone()]
        self.pull_S = [torch.zeros(self.n, device=self.device), self._stretch(self.F_snap).clone()]

    def rollout_F0(self, x0, F0, logE, K):
        """Release from an ARBITRARY (positions x0, deformation F0) at fixed E -- for F0 fitting."""
        torch, wp, sim, gc = self._torch, self._wp, self._sim, self._gc
        dev = self.device; sv, st, md = self._rsv, self._rst, self._rmd
        self._setE(sv, md, st, logE); sv.time = 0.0
        with torch.no_grad():
            st.continue_from_torch(x0.clone(), self._z3, F0.clone(), self._z33, device=dev, requires_grad=False)
            prev = st; out = [wp.to_torch(prev.particle_x).clone()]
            for _ in range(K):
                for _ in range(sim.substep):
                    nx = prev.partial_clone(requires_grad=False)
                    sv.p2g2p_differentiable(md, prev, nx, sim.substep_size, device=dev); prev = nx
                out.append(wp.to_torch(prev.particle_x).clone())
        res = torch.stack(out); del prev; gc.collect()
        return res

    def affine_from_s6(self, s6):
        """(x0, F0) for a homogeneous V0=expm(S(s6)): compatible affine positions + constant F."""
        torch = self._torch
        V0 = v0_from_s6(torch, s6, device=self.device)
        c = self.X_rest.mean(0)
        x0 = (c + (self.X_rest - c) @ V0.T).contiguous()
        F0 = V0[None].repeat(self.n, 1, 1).contiguous()
        return x0, F0

    # ---- release rollout (reuses ONE solver; rebuilding per call leaks warp GPU) ----
    def rollout(self, logE, K):
        torch, wp, sim, gc = self._torch, self._wp, self._sim, self._gc
        dev = self.device; sv, st, md = self._rsv, self._rst, self._rmd
        self._setE(sv, md, st, logE); sv.time = 0.0
        with torch.no_grad():
            st.continue_from_torch(self.x_snap.clone(), self._z3, self.F_snap.clone(), self._z33,
                                   device=dev, requires_grad=False)
            prev = st; out = [wp.to_torch(prev.particle_x).clone()]; ss = [self._stretch(self.F_snap).clone()]
            for _ in range(K):
                for _ in range(sim.substep):
                    nx = prev.partial_clone(requires_grad=False)
                    sv.p2g2p_differentiable(md, prev, nx, sim.substep_size, device=dev); prev = nx
                out.append(wp.to_torch(prev.particle_x).clone())
                ss.append(self._stretch(wp.to_torch(prev.particle_F_trial)).clone())
        res = torch.stack(out); sres = torch.stack(ss)
        del prev; gc.collect()
        return res, sres
