"""Pure-CPU samplers for the conditioning distribution p*(Y) of dataset_gen.

Y = (E, v0, T). Each axis is an INDEPENDENT 1-D spec drawn per sample with a
seeded numpy RandomState, so the realised dataset's marginals ARE exactly these
specs (recorded + plotted in the manifest). No torch / no GPU here -- these emit
plain python scalars / tuples that sim_render then lifts onto the device.

Why uniform (not log) for velocity: v0 is linear, signed, and has a meaningful
zero (rest). log can't represent rest, and the whole point of conditioning on v0
is an INTERPRETABLE recovered marginal -- uniform magnitude -> flat, readable.
E keeps log-uniform (it spans orders of magnitude).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

try:  # py3.8+: Literal in typing
    from typing import Literal
except ImportError:  # pragma: no cover
    from typing_extensions import Literal


def _rodrigues(v: np.ndarray, axis: np.ndarray, theta: float) -> np.ndarray:
    """Rotate vector v about unit `axis` by `theta` rad (Rodrigues)."""
    k = axis / (np.linalg.norm(axis) or 1.0)
    return (v * np.cos(theta) + np.cross(k, v) * np.sin(theta)
            + k * np.dot(k, v) * (1.0 - np.cos(theta)))


def _catmull(wp: np.ndarray, T: int) -> np.ndarray:
    """Smooth path through waypoints wp [K,D] sampled at T points (end-padded)."""
    P = np.vstack([wp[0], wp, wp[-1]])
    segs = len(wp) - 1
    out = []
    for i in range(T):
        u = i / max(1, T - 1) * segs
        s = min(int(u), segs - 1); t = u - s
        p0, p1, p2, p3 = P[s], P[s + 1], P[s + 2], P[s + 3]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.asarray(out)


def lookat_R_T(eye: np.ndarray, target: np.ndarray, up_world: np.ndarray):
    """OpenCV look-at -> (R, T) in PhysDreamer/getWorld2View2 convention.

    (Lives here, NOT in explore/cam_probe, so canonical code -- dataset_gen's
    camera sweep -- can use it without depending on an explore script.)

    Args:
        eye:      [3] camera position in world.
        target:   [3] point the optical axis aims at (world).
        up_world: [3] desired image-up direction (world).
    Returns:
        R: [3,3] camera-to-world rotation, columns = [right(+x), down(+y), forward(+z)].
        T: [3] world-to-camera translation s.t. getWorld2View2 recovers cam_center=eye.
    """
    forward = target - eye
    forward = forward / np.linalg.norm(forward)        # [3] camera +z
    right = np.cross(forward, up_world)
    right = right / np.linalg.norm(right)              # [3] camera +x
    down = np.cross(forward, right)                    # [3] camera +y
    R = np.stack([right, down, forward], axis=1)       # [3,3] columns = cam axes in world
    T = -R.T @ eye                                     # [3]
    return R.astype(np.float32), T.astype(np.float32)


@dataclass
class EDist:
    """p*(E). fixed -> constant `E`; loguniform -> 10^U[log10 E_min, log10 E_max]."""

    mode: Literal["fixed", "loguniform"] = "loguniform"
    E: float = 1e5  # used when mode="fixed"
    E_min: float = 1e4
    E_max: float = 1e6

    @property
    def varies(self) -> bool:
        return self.mode != "fixed"

    def sample(self, rng: "np.random.RandomState") -> float:
        if self.mode == "fixed":
            return float(self.E)
        return float(10.0 ** rng.uniform(np.log10(self.E_min), np.log10(self.E_max)))

    def to_dict(self) -> dict:
        return {"mode": self.mode, "E": self.E, "E_min": self.E_min, "E_max": self.E_max}


@dataclass
class V0Dist:
    """p*(v0) over a constant initial-velocity vector (normalised-space).

    mode:
      fixed  -- always `vec` (back-compat: the old single v0 tuple).
      axis   -- direction fixed to `axis` (0=x,1=y,2=z); sign +/- if `signed`,
                else + only. magnitude ~ U[mag_min, mag_max].
      sphere -- direction uniform on the FULL S^2; magnitude ~ U[mag_min, mag_max].

    `include_rest` is expressed at the config layer as mag_min=0 (a/c) vs
    mag_min>0 (b/d) -- no separate flag here.
    """

    mode: Literal["fixed", "axis", "sphere"] = "fixed"
    vec: Tuple[float, float, float] = (0.0, -1.0, 0.0)
    axis: int = 0
    signed: bool = True
    mag_min: float = 0.0
    mag_max: float = 2.0

    @property
    def varies(self) -> bool:
        return self.mode != "fixed"

    def sample(self, rng: "np.random.RandomState") -> Tuple[float, float, float]:
        if self.mode == "fixed":
            return (float(self.vec[0]), float(self.vec[1]), float(self.vec[2]))
        mag = float(rng.uniform(self.mag_min, self.mag_max))
        if self.mode == "axis":
            d = np.zeros(3)
            d[self.axis] = -1.0 if (self.signed and rng.uniform() < 0.5) else 1.0
        else:  # sphere: uniform direction = normalised isotropic gaussian
            g = rng.normal(size=3)
            nrm = float(np.linalg.norm(g))
            d = g / nrm if nrm > 1e-8 else np.array([0.0, -1.0, 0.0])
        v = d * mag
        return (float(v[0]), float(v[1]), float(v[2]))

    def to_dict(self) -> dict:
        return {"mode": self.mode, "vec": list(self.vec), "axis": self.axis,
                "signed": self.signed, "mag_min": self.mag_min, "mag_max": self.mag_max}


@dataclass
class TDist:
    """p*(T) over the rendered frame count. fixed -> sim.num_frames; uniform_int
    -> U{T_min..T_max} per sample (the T-agnostic / variable-horizon dataset)."""

    mode: Literal["fixed", "uniform_int"] = "fixed"
    T_min: int = 8
    T_max: int = 64

    @property
    def varies(self) -> bool:
        return self.mode != "fixed"

    def sample(self, rng: "np.random.RandomState", default_T: int) -> int:
        if self.mode == "fixed":
            return int(default_T)
        return int(rng.randint(self.T_min, self.T_max + 1))

    def to_dict(self) -> dict:
        return {"mode": self.mode, "T_min": self.T_min, "T_max": self.T_max}


@dataclass
class CameraDist:
    """p*(camera) over the RENDER viewpoint.

    fixed     -- always the dataset's reference camera (cfg.frame); v1 behaviour.
    orbit_cap -- per-sample STATIC: one eye within `cap_deg` of the ref view.
    compound  -- per-FRAME DYNAMIC: the camera MOVES within the clip along a smooth
                 path in (azim, elev, r) about the object centre (eye-on-sphere,
                 always looking at it -- NO composed Euler rotations). Bounds are
                 the scene's SAFE BOX (one-sided here: +azim to wall, +elev to
                 no-floor). azim/elev in deg off the ref view; radius as a FRACTION
                 of the ref radius (scale-portable). Defaults = telephone safe box.
    """

    mode: Literal["fixed", "orbit_cap", "compound"] = "fixed"
    cap_deg: float = 40.0                              # orbit_cap only
    azim: Tuple[float, float] = (0.0, 90.0)            # compound: off-ref deg range
    elev: Tuple[float, float] = (0.0, 32.0)
    r_frac: Tuple[float, float] = (0.4, 1.35)          # radius = ref_r * U[r_frac]
    n_waypoints: int = 4
    # per-clip MOTION EXTENT: span ~ U[span] = fraction of each axis range the clip
    # uses (at a random sub-window). span~0 -> camera ~static at a random viewpoint;
    # span~1 -> traverses the full box. Gives a spread of motion amount/speed, not
    # always-large. (0,1) = anywhere from near-static to full.
    span: Tuple[float, float] = (0.0, 1.0)
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0)

    @property
    def varies(self) -> bool:
        return self.mode != "fixed"

    @property
    def dynamic(self) -> bool:
        """Per-frame moving camera (needs a per-frame render path)."""
        return self.mode == "compound"

    def sample_RT(self, rng: "np.random.RandomState", center, ref_eye):
        """Sample (R[3,3], T[3], eye[3]) (np.float32) for one clip's camera.

        center:   [3] world point to look at (object centroid).
        ref_eye:  [3] reference camera position in world (cap is centred on its dir).
        """
        center = np.asarray(center, dtype=np.float64)
        ref_eye = np.asarray(ref_eye, dtype=np.float64)
        radius = float(np.linalg.norm(ref_eye - center)) or 1.0
        u0 = (ref_eye - center) / radius                       # [3] ref view dir
        cap = np.deg2rad(self.cap_deg)
        ct = rng.uniform(np.cos(cap), 1.0)                     # area-weighted polar
        st = float(np.sqrt(max(0.0, 1.0 - ct * ct)))
        phi = rng.uniform(0.0, 2.0 * np.pi)
        a = np.array([0.0, 0.0, 1.0]) if abs(u0[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(u0, a); e1 /= np.linalg.norm(e1)         # [3] basis around u0
        e2 = np.cross(u0, e1)                                  # [3]
        dirn = ct * u0 + st * (np.cos(phi) * e1 + np.sin(phi) * e2)
        eye = center + radius * dirn                           # [3]
        R, T = lookat_R_T(eye, center, np.asarray(self.up, dtype=np.float64))
        return R, T, eye.astype(np.float32)

    def sample_traj(self, rng, center, ref_eye, ref_R, T):
        """One clip's DYNAMIC camera path (mode='compound'). Returns a list of T
        dicts {R,T,eye,azim,elev,r}: smooth (azim,elev,r) path through n_waypoints
        random points in the safe box, eye-on-sphere about `center`, looking at it.

        center/ref_eye: [3] world. ref_R: [3,3] ref camera (cols [right,down,fwd]).
        """
        center = np.asarray(center, dtype=np.float64)
        ref_eye = np.asarray(ref_eye, dtype=np.float64)
        ref_r = float(np.linalg.norm(ref_eye - center)) or 1.0
        d0 = (ref_eye - center) / ref_r
        Rm = np.asarray(ref_R, dtype=np.float64)
        right_cam = Rm[:, 0] / (np.linalg.norm(Rm[:, 0]) or 1.0)       # ref screen-right (world)
        up_cam = -Rm[:, 1] / (np.linalg.norm(Rm[:, 1]) or 1.0)         # ref screen-up (world)
        up = np.asarray(self.up, dtype=np.float64)

        # per-clip motion extent: a random sub-window of width span*range per axis
        sp = float(rng.uniform(*self.span))

        def _sub(lo, hi):
            w = (hi - lo) * sp
            c = rng.uniform(lo + w / 2, hi - w / 2) if w < (hi - lo) else (lo + hi) / 2
            return c - w / 2, c + w / 2

        azr, elr, rfr = _sub(*self.azim), _sub(*self.elev), _sub(*self.r_frac)
        wp = np.stack([rng.uniform(*azr, self.n_waypoints),
                       rng.uniform(*elr, self.n_waypoints),
                       rng.uniform(*rfr, self.n_waypoints)], axis=1)  # [K,3]
        path = _catmull(wp, T)                                          # [T,3]
        path[:, 0] = np.clip(path[:, 0], *self.azim)                    # clamp spline overshoot
        path[:, 1] = np.clip(path[:, 1], *self.elev)
        path[:, 2] = np.clip(path[:, 2], *self.r_frac)

        out = []
        for az, el, rf in path:
            a, e = np.deg2rad(az), np.deg2rad(el)
            d1 = _rodrigues(d0, up_cam, a)                              # yaw about ref up
            r1 = _rodrigues(right_cam, up_cam, a)
            d2 = _rodrigues(d1, r1, e)                                  # then pitch
            eye = center + (rf * ref_r) * d2
            R, Tcw = lookat_R_T(eye, center, up)
            out.append({"R": R, "T": Tcw, "eye": eye.astype(np.float32),
                        "azim": float(az), "elev": float(el), "r": float(rf * ref_r)})
        return out

    def to_dict(self) -> dict:
        return {"mode": self.mode, "cap_deg": self.cap_deg,
                "azim": list(self.azim), "elev": list(self.elev),
                "r_frac": list(self.r_frac), "n_waypoints": self.n_waypoints,
                "span": list(self.span), "up": list(self.up)}
