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
    gradu    : gradu,   g=0,  no floor   -- KNOWN analytic u(x): F0=I+grad u, x0=X_rest+u (u-field truth)

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
    "gradu":    ("gradu",    False,            False),   # analytic non-uniform F0=I+grad u, pure release
    "telbend":  ("telbend",  False,            False),   # external particles: grip a region + curl, then release
    "telsag":   ("telsag",   False,            False),   # swing free cord horizontal, pull midpoint down, release
    "telxlock": ("telxlock", False,            False),   # tail grip + tiny hang held by x-lock (stress propagates)
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
                 S_gt=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), gradu_A=0.05,
                 gravity=9.8, floor_z=0.25, collider="slip", friction=0.0, device="cuda:0",
                 X_rest_ext=None, p_vol_ext=None,
                 grip_point=None, grip_size=None, grip_vel=None, grip_frames=8,
                 anchor_point=None, anchor_size=None, release_anchor_mask=None,
                 swing_frames=10, pull_frames=10, sag_pull_speed=0.5, swing_dir=1.0,
                 tail_frac_sag=0.12, mid_frac_sag=0.08):
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
        if X_rest_ext is not None:                       # external particles (e.g. telephone cache)
            X_rest = X_rest_ext.to(device)
            self._p_vol = (p_vol_ext.to(device) if p_vol_ext is not None
                           else torch.full((X_rest.shape[0],), float((2 * hx / max(nx - 1, 1)) ** 3), device=device))
        else:
            gx = torch.linspace(cx - hx, cx + hx, nx); gy = torch.linspace(cy - hy, cy + hy, ny)
            gz = torch.linspace(cz - hz, cz + hz, nz)
            X_rest = torch.stack(torch.meshgrid(gx, gy, gz, indexing="ij"), -1).reshape(-1, 3).to(device)
            self._p_vol = torch.full((X_rest.shape[0],), float((2 * hx / max(nx - 1, 1)) ** 3), device=device)
        self.X_rest = X_rest; self.n = n = X_rest.shape[0]
        self.grip_point, self.grip_size, self.grip_vel, self.grip_frames = grip_point, grip_size, grip_vel, grip_frames
        self.anchor_point, self.anchor_size = anchor_point, anchor_size
        # release keeps ONLY the original hanging anchor; the grip-time top hold is let go
        self._release_anchor_mask = (release_anchor_mask.to(device) if release_anchor_mask is not None else None)
        self._freeze_anchor_mask = None
        self._xlock_mask = None; self._xlock_pos = None
        self.swing_frames, self.pull_frames, self.sag_pull_speed, self.swing_dir = swing_frames, pull_frames, sag_pull_speed, swing_dir
        self.tail_frac_sag, self.mid_frac_sag = tail_frac_sag, mid_frac_sag
        self._eye = torch.eye(3, device=device)
        self._z3 = torch.zeros(n, 3, device=device); self._z33 = torch.zeros(n, 3, 3, device=device)

        if f0_method == "pull":
            self._make_pull_f0(pull_speed, release_frame, grip_half_x)
        elif f0_method == "squeeze":
            self._make_squeeze_f0(push_x, push_half_x, push_half_z, push_speed, push_frames)
        elif f0_method == "telbend":
            self._make_telbend_f0()
        elif f0_method == "telsag":
            self._make_telsag_f0()
        elif f0_method == "telxlock":
            self._make_telxlock_f0()
        elif f0_method == "gradu":
            self._make_gradu_f0(gradu_A)
        else:
            self._make_uniform_f0(S_gt)

        g_vec = (0.0, 0.0, -gravity) if rel_grav else (0.0, 0.0, 0.0)
        if scene in ("telbend", "telsag"):           # release: drop grip-time holds, keep only the hang
            self._freeze_anchor_mask = self._release_anchor_mask
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
        if getattr(self, "_freeze_anchor_mask", None) is not None:   # telbend: TRUE freeze via grid BC
            self._apply_grid_freeze(sv)
        return sv, st, md

    def _apply_grid_freeze(self, sv):
        """Verified no-leak freeze (sim_render build_mpm): zero grid_v at the FULL 27-node
        G2P stencil of every frozen anchor particle (not just the base node)."""
        torch = self._torch
        G, GL = self._G, self._GL; inv_dx = G / GL
        pos = self.X_rest[self._freeze_anchor_mask]                  # frozen particles don't move
        base = (pos * inv_dx - 0.5).to(torch.int64)                  # [F,3]
        fg = torch.zeros((G, G, G), dtype=torch.int32, device=self.device)
        for di in range(3):
            for dj in range(3):
                for dk in range(3):
                    fg[(base[:, 0] + di).clamp(0, G - 1), (base[:, 1] + dj).clamp(0, G - 1),
                       (base[:, 2] + dk).clamp(0, G - 1)] = 1
        sv.enforce_grid_velocity_by_mask(fg)

    def _xlock(self, state):
        """Position-lock BC (real-grip spirit): overwrite the held particles' x in place
        every substep, WITHOUT touching v or F -> stress/deformation propagate THROUGH the
        held points (unlike velocity-BC / grid-freeze, which zero velocity and cut stress)."""
        if getattr(self, "_xlock_mask", None) is None:
            return
        xt = self._wp.to_torch(state.particle_x)        # zero-copy view -> in-place write
        xt[self._xlock_mask] = self._xlock_pos

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
        Fs = [wp.to_torch(prev.particle_F_trial).clone()]            # full F(t) per frame (F-drift probe)
        for _ in range(n_frames):
            for _ in range(sim.substep):
                nx = prev.partial_clone(requires_grad=False)
                sv.p2g2p_differentiable(md, prev, nx, sim.substep_size, device=dev); prev = nx
                self._xlock(prev)                        # position-lock held points (stress still flows)
            xs.append(wp.to_torch(prev.particle_x).clone())
            ss.append(self._stretch(wp.to_torch(prev.particle_F_trial)).clone())
            Fs.append(wp.to_torch(prev.particle_F_trial).clone())
        self.x_snap = wp.to_torch(prev.particle_x).clone()
        self.F_snap = wp.to_torch(prev.particle_F_trial).clone()
        self.maxdev = float((self.x_snap - self.X_rest).norm(dim=1).max())
        self.F0_stretch = self._stretch(self.F_snap).cpu().numpy()
        self.pull_X, self.pull_S, self.pull_F = xs, ss, Fs
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

    def _make_telbend_f0(self):
        """External particles (e.g. telephone): the top is TRULY FROZEN (heavy anchors,
        via self._heavy_mask in _build) AND held at v=0; the gripped tail traces an ARC --
        lateral (grip_axis) for the first half of grip_frames, then UP (+z) for the second
        half -- so it curls round rather than translating straight.  Records the GRIP phase
        (pull_X/pull_S/pull_F) + snapshots F0.  g=0, no floor; freeze re-applied at release."""
        torch, dev, sim = self._torch, self.device, self._sim
        assert self.grip_point and self.grip_size and self.grip_vel, "telbend needs grip_point/size/vel"
        # TRUE freeze of the top: grid-velocity-by-mask over the 27-node stencil (set BEFORE
        # _build so both the grip solver and the release solver _rsv get it).
        ap, asz = torch.tensor(self.anchor_point, device=dev), torch.tensor(self.anchor_size, device=dev)
        self._freeze_anchor_mask = ((self.X_rest - ap).abs() <= asz).all(dim=1)
        sv, st, md = self._build()                                     # applies grid freeze (see _build)
        self._setE(sv, md, st, self.gt_logE); sv.time = 0.0
        speed = float(max(abs(v) for v in self.grip_vel))
        with torch.no_grad():
            st.continue_from_torch(self.X_rest.clone(), self._z3, self._eye[None].repeat(self.n, 1, 1).contiguous(),
                                   self._z33, device=dev, requires_grad=False)
            et = self.grip_frames * sim.delta_t; et_half = (self.grip_frames // 2) * sim.delta_t
            sv.enforce_particle_velocity_translation(st, point=tuple(self.grip_point), size=tuple(self.grip_size),
                velocity=tuple(self.grip_vel), start_time=0.0, end_time=et_half, device=dev)   # phase 1: lateral
            sv.enforce_particle_velocity_translation(st, point=tuple(self.grip_point), size=tuple(self.grip_size),
                velocity=(0.0, 0.0, speed), start_time=et_half, end_time=et, device=dev)        # phase 2: up (arc)
            self._record_snapshot(sv, st, md, self.grip_frames)

    def _make_telsag_f0(self):
        """Clean symmetric bend with stress throughout (small grips, not a big freeze):
          phase A: swing the free cord about the hang to ~horizontal (rotation BC), so the
                   subsequent pull is a transverse (not axial) bend;
          phase B: hold the hang (left) and the swung tail-end (right) at v=0 by MASK, pull
                   the MIDPOINT down by mask -> a symmetric sag.
        Records pull_X/pull_S/pull_F + snapshots F0.  Release keeps only the hang (see __init__)."""
        import math
        torch, dev, sim = self._torch, self.device, self._sim
        X = self.X_rest; z = X[:, 2]; zmin, zmax = float(z.min()), float(z.max()); Lz = zmax - zmin
        hang = self._release_anchor_mask
        assert hang is not None, "telsag needs release_anchor_mask (the original hang)"
        tail = z <= zmin + self.tail_frac_sag * Lz                       # bottom end (becomes right after swing)
        mid = (z - 0.5 * (zmin + zmax)).abs() <= self.mid_frac_sag * Lz   # midpoint band
        self._freeze_anchor_mask = hang                                  # grip-time: only hang grid-frozen (small)
        sv, st, md = self._build()
        self._setE(sv, md, st, self.gt_logE); sv.time = 0.0
        sf, pf = self.swing_frames, self.pull_frames
        tA, tB = sf * sim.delta_t, (sf + pf) * sim.delta_t
        omega = (math.pi / 2) / max(tA, 1e-9) * self.swing_dir            # ~90 deg over phase A
        hang_c = X[hang].mean(0).tolist()
        with torch.no_grad():
            st.continue_from_torch(X.clone(), self._z3, self._eye[None].repeat(self.n, 1, 1).contiguous(),
                                   self._z33, device=dev, requires_grad=False)
            # phase A: rotate the free cord about the hang in the x-z plane (normal = y)
            sv.enforce_particle_velocity_rotation(st, point=tuple(hang_c), normal=(0.0, 1.0, 0.0),
                half_height_and_radius=(0.12, Lz * 1.3), rotation_scale=omega, translation_scale=0.0,
                start_time=0.0, end_time=tA, device=dev)
            # phase B: hold hang + tail-end (v=0), pull midpoint DOWN (-z) -> symmetric sag
            sv.enforce_particle_velocity_by_mask(st, hang.int().contiguous(), [0.0, 0.0, 0.0], tA, tB)
            sv.enforce_particle_velocity_by_mask(st, tail.int().contiguous(), [0.0, 0.0, 0.0], tA, tB)
            sv.enforce_particle_velocity_by_mask(st, mid.int().contiguous(), [0.0, 0.0, -self.sag_pull_speed], tA, tB)
            self._record_snapshot(sv, st, md, sf + pf)

    def _make_telxlock_f0(self):
        """Tail-grip bend with the TINY hang held by POSITION-LOCK (not grid-freeze):
        x-overwrite each substep keeps the hang fixed while v/F (stress) propagate through it,
        so the upper cord actually stores pre-stress (the big grid-freeze of telbend zeroed
        grid velocity -> cut stress -> left branch had none).  Tail traces the same arc."""
        torch, dev, sim = self._torch, self.device, self._sim
        assert self.grip_point and self.grip_size and self.grip_vel, "telxlock needs grip_point/size/vel"
        hang = self._release_anchor_mask
        assert hang is not None, "telxlock needs release_anchor_mask (the hang)"
        self._freeze_anchor_mask = None                       # NO grid-freeze (it cuts stress)
        self._xlock_mask = hang                               # tiny hang held by position-lock
        self._xlock_pos = self.X_rest[hang].clone()
        sv, st, md = self._build()
        self._setE(sv, md, st, self.gt_logE); sv.time = 0.0
        speed = float(max(abs(v) for v in self.grip_vel))
        with torch.no_grad():
            st.continue_from_torch(self.X_rest.clone(), self._z3, self._eye[None].repeat(self.n, 1, 1).contiguous(),
                                   self._z33, device=dev, requires_grad=False)
            et = self.grip_frames * sim.delta_t; et_half = (self.grip_frames // 2) * sim.delta_t
            sv.enforce_particle_velocity_translation(st, point=tuple(self.grip_point), size=tuple(self.grip_size),
                velocity=tuple(self.grip_vel), start_time=0.0, end_time=et_half, device=dev)
            sv.enforce_particle_velocity_translation(st, point=tuple(self.grip_point), size=tuple(self.grip_size),
                velocity=(0.0, 0.0, speed), start_time=et_half, end_time=et, device=dev)
            self._record_snapshot(sv, st, md, self.grip_frames)

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

    def _make_gradu_f0(self, A):
        # KNOWN non-uniform displacement u(x): half-sine y-bend along x. F0 = I + grad u
        # is COMPATIBLE (it IS the gradient of the applied u), positions x0 = X_rest + u.
        # Unlike pull/uniform, u has a closed form -> a fitted u-field has a ground truth.
        import math
        torch = self._torch
        X = self.X_rest
        xmin = X[:, 0].min(); Lx = (X[:, 0].max() - xmin).clamp_min(1e-9)
        xi = (X[:, 0] - xmin) / Lx                                  # in [0,1] along x
        u = torch.zeros_like(X)
        u[:, 1] = A * torch.sin(math.pi * xi)                       # u_y half-sine
        dudx = A * (math.pi / Lx) * torch.cos(math.pi * xi)         # d(u_y)/dx, per particle
        self.x_snap = (X + u).contiguous()
        F0 = self._eye[None].repeat(self.n, 1, 1).clone()
        F0[:, 1, 0] = dudx                                          # F = I + grad u (only u_y,x nonzero)
        self.F_snap = F0.contiguous()
        self.maxdev = float(u.norm(dim=1).max())
        self.F0_stretch = self._stretch(self.F_snap).cpu().numpy()
        self.pull_X = [X.clone(), self.x_snap.clone()]
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
                    self._xlock(prev)
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
            # top freeze: _rsv's _build already applied enforce_grid_velocity_by_mask (telbend)
            prev = st; out = [wp.to_torch(prev.particle_x).clone()]; ss = [self._stretch(self.F_snap).clone()]
            for _ in range(K):
                for _ in range(sim.substep):
                    nx = prev.partial_clone(requires_grad=False)
                    sv.p2g2p_differentiable(md, prev, nx, sim.substep_size, device=dev); prev = nx
                    self._xlock(prev)
                out.append(wp.to_torch(prev.particle_x).clone())
                ss.append(self._stretch(wp.to_torch(prev.particle_F_trial)).clone())
        res = torch.stack(out); sres = torch.stack(ss)
        del prev; gc.collect()
        return res, sres

    def rollout_width(self, logE, K, rec_substeps=None):
        """Release rollout recording the x-WIDTH (all-particle x-extent, the spectral
        observable) every `rec_substeps` substeps -- SUB-FRAME sampling for the Nyquist
        probe.  Physics (dt = substep_size, total time = K*delta_t) is IDENTICAL to
        rollout(); only the RECORDING cadence changes, so 1x vs 2x sampling tests aliasing
        without touching the dynamics.

        rec_substeps: substeps between width samples (default = sim.substep = once/frame =
        the original 1x rate; sim.substep//2 = 2x rate).  Returns (width (T,) cpu tensor,
        sample_dt seconds, samples_per_frame).  T = K*sim.substep//rec_substeps + 1.
        """
        torch, wp, sim, gc = self._torch, self._wp, self._sim, self._gc
        dev = self.device; sv, st, md = self._rsv, self._rst, self._rmd
        rec = sim.substep if rec_substeps is None else int(rec_substeps)
        assert (sim.substep % rec) == 0, (sim.substep, rec)   # keep frame boundaries aligned
        self._setE(sv, md, st, logE); sv.time = 0.0
        def _w(state):
            x0 = wp.to_torch(state.particle_x)[:, 0]
            return float(x0.amax() - x0.amin())
        with torch.no_grad():
            st.continue_from_torch(self.x_snap.clone(), self._z3, self.F_snap.clone(), self._z33,
                                   device=dev, requires_grad=False)
            prev = st; w = [_w(prev)]; step = 0
            for _ in range(K):
                for _ in range(sim.substep):
                    nx = prev.partial_clone(requires_grad=False)
                    sv.p2g2p_differentiable(md, prev, nx, sim.substep_size, device=dev); prev = nx
                    self._xlock(prev)
                    step += 1
                    if step % rec == 0:
                        w.append(_w(prev))
        del prev; gc.collect()
        return torch.tensor(w), float(sim.substep_size * rec), sim.substep // rec
