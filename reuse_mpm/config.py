"""Single source of truth for run configuration.

Every entrypoint builds one of these dataclasses (via `tyro.cli`) and serialises
it verbatim into the run dir's `config.json`, so the *resolved* config and the
*recorded* config are the same object -- no hand-built per-task dicts that drift.

Physics constants live ONLY here (never hardcoded deep inside `build_mpm`), and
derived quantities (`substep_size`) are properties, not values recomputed at each
call site.

Composition:
  SimConfig    -- physics + rollout (shared by every task)
  SceneSpec    -- which scene + how it is discretised (shared by every task)
  <Task>Config -- one per entrypoint, composing SimConfig + SceneSpec + task args
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

try:  # py3.8+: Literal in typing
    from typing import Literal
except ImportError:  # pragma: no cover
    from typing_extensions import Literal

from .sampling import CameraDist, EDist, TDist, V0Dist


# --------------------------------------------------------------------------- #
# Scene presets: name -> (kind, path), resolved against env-overridable roots.
# A preset is the "I don't want to type a path" door; it bakes in the correct
# kind, so the "looks like pd but I said pg" footgun cannot happen.
# --------------------------------------------------------------------------- #
_PD_DATA = os.path.join(
    os.environ.get("PHYSDREAMER_ROOT", "/tmp2/b10401006/PhysDreamer"),
    "data", "physics_dreamer",
)
_PG_ROOT = os.environ.get("PG_ROOT", "/tmp2/b10401006/PhysGaussian/model")


class ScenePreset(Enum):
    telephone = "telephone"
    alocasia = "alocasia"
    carnations = "carnations"
    hat = "hat"
    ficus = "ficus"
    bread = "bread"


# (kind, path) for each preset
PRESETS: dict = {
    ScenePreset.telephone: ("pd", os.path.join(_PD_DATA, "telephone")),
    ScenePreset.alocasia: ("pd", os.path.join(_PD_DATA, "alocasia")),
    ScenePreset.carnations: ("pd", os.path.join(_PD_DATA, "carnations")),
    ScenePreset.hat: ("pd", os.path.join(_PD_DATA, "hat")),
    ScenePreset.ficus: ("pg", os.path.join(_PG_ROOT, "ficus_whitebg-trained")),
    ScenePreset.bread: ("pg", os.path.join(_PG_ROOT, "bread-trained")),
}


# --------------------------------------------------------------------------- #
# Shared building blocks
# --------------------------------------------------------------------------- #
@dataclass
class SimConfig:
    """Physics regime + rollout discretisation (mirrors render_trained_sim.py).

    Material = jelly, gravity off, motion driven entirely by an initial velocity
    on the moving particles; stiffness/response governed by E. With v0 fixed and
    known, the video dynamics are a function of E alone.
    """

    # rollout
    num_frames: int = 14
    substep: int = 64
    fps: int = 7
    delta_t: float = 1.0 / 30.0  # physical time per rendered frame

    # MPM grid
    grid_size: int = 32
    grid_lim: float = 1.0

    # material
    density: float = 2000.0
    material: str = "jelly"
    grid_v_damping_scale: float = 1.1
    nu: float = 0.3  # Poisson ratio (held fixed in v1; a future Y axis)

    @property
    def substep_size(self) -> float:
        """Physical dt per MPM substep (delta_t spread over `substep` substeps)."""
        return self.delta_t / self.substep

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SceneSpec:
    """Which scene to load and how to discretise it into MPM particles.

    Two DISJOINT ways to name the scene (exactly one, enforced at construction):
      - `preset`: an enum that bakes in the correct (kind, path). One word, no
        path, no chance of a kind/path mismatch.
      - `path` + `kind`: spell out every field yourself. `kind` is REQUIRED here
        and never inferred -- if you take the explicit door you state pd/pg.

    `preset` is consumed in __post_init__: it resolves into `kind`/`path` (and
    `name`), so every consumer afterwards just reads `.kind` / `.path` -- no
    preset-vs-path branching anywhere downstream. The serialized form is always
    the resolved (kind, path), so config.json round-trips cleanly.

    `grid_size` is intentionally NOT here: it must equal SimConfig.grid_size, so
    the loader reads it from the SimConfig at load time (single source of truth).
    """

    preset: Optional[ScenePreset] = None
    path: Optional[str] = None  # dataset_dir (pd) or model_dir (pg)
    kind: Optional[Literal["pd", "pg"]] = None
    name: Optional[str] = None
    downsample_scale: float = 0.1
    top_k: int = 8
    max_particles: int = 8000
    device: str = "cuda:0"
    # explicit cache path; default = derived from (path, downsample_scale, grid_size, top_k)
    cache_path: Optional[str] = None
    # anchor BC. pd default "moving" (far-from-moving_part_points); "slab" = robust
    # geometric bottom-slab (pg always uses slab). freeze_frac/freeze_axis param the
    # slab (axis None => longest axis).
    freeze_mode: str = "moving"
    freeze_frac: float = 0.15
    freeze_axis: Optional[int] = None

    def __post_init__(self):
        if self.preset is not None:
            if self.path is not None:
                raise ValueError(
                    "scene: choose EITHER --scene.preset OR --scene.path, not both")
            self.kind, self.path = PRESETS[self.preset]
            if self.name is None:
                self.name = self.preset.value
            self.preset = None  # consumed -> serialized form is the resolved (kind, path)
        else:
            if self.path is None:
                raise ValueError(
                    "scene: give --scene.preset <name>, or "
                    "--scene.path <dir> --scene.kind {pd,pg}")
            if self.kind is None:
                raise ValueError(
                    "scene: --scene.path requires --scene.kind {pd,pg} "
                    "(kind is never inferred, by design)")

    @property
    def display_name(self) -> str:
        """Human label: explicit name, else the leaf dir of the resolved path."""
        return self.name or os.path.basename(os.path.normpath(self.path))

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Per-task configs (one per entrypoint)
# --------------------------------------------------------------------------- #
# `out` / `run_label` are the output-tree convention (see run_io): out=None ->
# auto outputs/<task>/<NNNN>_<ts>[_<run_label>]; out=<path> overrides.
@dataclass
class ForwardConfig:
    """forward_gen: known constant E -> one video."""

    scene: SceneSpec
    E: float  # constant Young's modulus, OR the base/geomean E when a gradient is set
    sim: SimConfig = field(default_factory=SimConfig)
    v0: Tuple[float, float, float] = (0.0, -0.5, 0.0)
    frame: str = "frame_00001.png"  # camera image filename
    # Optional spatially-varying GT (phase B): when E_grad_axis is set, E becomes a
    # smooth gradient along that axis spanning E_grad_decades decades of log10(E)
    # (geomean still == E). The per-particle GT field is saved to E_field.npy so the
    # field-recovery can score per-particle reconstruction. None => uniform E.
    E_grad_axis: Optional[int] = None
    E_grad_decades: float = 1.0
    # Optional spatially-varying GT v0 (phase B, v0 dual): when v0_grad_axis is set,
    # v0 magnitude ramps linearly along that axis (slope v0_grad_slope), direction
    # fixed to `v0`; the mean stays `v0`. The per-particle GT field is saved to
    # v0_field.npy so train_v0 can score per-particle reconstruction. None => uniform.
    v0_grad_axis: Optional[int] = None
    v0_grad_slope: float = 1.0
    out: Optional[str] = None
    run_label: str = ""


@dataclass
class DatasetConfig:
    """dataset_gen: sample Y=(E,v0,T)~p*(Y) -> (Y, video) dataset.

    Each conditioning axis is an INDEPENDENT 1-D distribution (sampling.EDist /
    V0Dist / TDist); the realised dataset's marginals ARE these specs (recorded +
    plotted in the manifest). Defaults reproduce the v1 behaviour -- E
    log-uniform[1e4,1e6], v0 fixed (0,-1,0), T = sim.num_frames -- so the old
    E-only sweep is just e_dist=loguniform with v0_dist/t_dist=fixed.

    The four uniform-v0 datasets (E fixed, v0 varies, T fixed):
      a = v0_dist{axis,  mag_min 0}    b = v0_dist{axis,  mag_min>0}
      c = v0_dist{sphere,mag_min 0}    d = v0_dist{sphere,mag_min>0}
    """

    scene: SceneSpec
    sim: SimConfig = field(default_factory=lambda: SimConfig(num_frames=16, substep=64))
    e_dist: EDist = field(default_factory=EDist)
    v0_dist: V0Dist = field(default_factory=V0Dist)
    t_dist: TDist = field(default_factory=TDist)
    cam_dist: CameraDist = field(default_factory=CameraDist)  # default fixed = v1 view
    n: int = 16
    frame: str = "frame_00001.png"
    seed: int = 0
    jump_thresh: float = 0.5  # per-frame max normalised single-step jump -> unstable flag
    # per-sample IO: light_io True => skip per-frame pngs (redundant with mp4);
    # mp4 + gif still written. Per-frame plys are reconstructable from
    # mpm_xyz.npy on demand (regen_ply); the dataset "glance" is panel.gif.
    light_io: bool = True
    panel_max: int = 16  # max clips tiled into panel.gif (evenly spaced if n>this)
    # one-line human note on what this dataset is for (-> manifest + README.md).
    # If empty, an auto summary of the (E,v0,T) specs is used.
    description: str = ""
    out: Optional[str] = None
    run_label: str = ""


@dataclass
class RecoverConfig:
    """train_global_E: recover one global E from a forward_gen run.

    scene + sim + v0 + frame are read from the GT run's config.json, NOT set here,
    so the inverse uses the exact same setup that produced the GT.
    """

    gt_run: str
    init_E: float = 3e5
    iters: int = 60
    lr: float = 0.1  # lr on log10(E)
    window: int = 3  # frames (from t=1) used in the loss
    grad_window: int = 1  # frames keeping BPTT grad (truncated BPTT)
    coarse_init: bool = False
    coarse_n: int = 9
    out: Optional[str] = None
    run_label: str = ""


@dataclass
class RecoverFieldConfig:
    """train_field_E: recover a spatially-varying E FIELD from a forward_gen run.

    Same GT contract as RecoverConfig (scene+sim+v0+frame read from the GT run),
    but optimises an EField (voxel|triplane) instead of a global scalar -- the
    over-parameterised-landscape variant. Predicts ABSOLUTE log10(E); `init_E` is
    only the field initialisation (uniform start matching the scalar baseline).
    """

    gt_run: str
    backbone: Literal["voxel", "triplane"] = "voxel"
    init_E: float = 3e5
    res: int = 16          # grid / plane resolution
    feat_dim: int = 16     # triplane per-plane feature channels
    mlp_hidden: int = 64   # triplane decoder width
    reg_weight: float = 1e-3  # smoothness (TV) weight; 0 disables
    iters: int = 80
    lr: float = 0.05
    # window=1 is the VALIDATED default: with truncated BPTT, window>1 rolls deep
    # target frames from a detached prefix whose render gradient sign-flips and
    # overwhelms the correct frame-1 term (see mpm-diff-gotchas #4 / explore.gradcheck).
    window: int = 1
    grad_window: int = 1
    out: Optional[str] = None
    run_label: str = ""


@dataclass
class RecoverV0Config:
    """train_v0: recover an initial-velocity FIELD v0 from a forward_gen run, with E
    held KNOWN (the dual of RecoverConfig, which assumed v0 known and fit E).

    Same GT contract: scene + sim + E + v0(GT) + frame are read from the GT run's
    config.json. `kind` selects the v0 parametrisation (global 3-vector | voxel |
    triplane), like EField's backbone. Gravity is off, so v0 drives all motion and a
    SHORT window of full-BPTT frames carries the v0 gradient (see recover_v0).
    """

    gt_run: str
    kind: Literal["global", "voxel", "triplane"] = "triplane"
    res: int = 16          # grid / plane resolution (voxel|triplane)
    feat_dim: int = 16     # triplane per-plane feature channels
    mlp_hidden: int = 64   # triplane decoder width
    v_clamp: float = 5.0   # per-component |v0| clamp (CFL/blow-up guard)
    vel_scale: float = 1.0    # field output ×this (PhysDreamer uses 0.1)
    reg_weight: float = 0.0   # TV smoothness weight; 0 disables (no-op for "global")
    grad_clip: float = 10.0   # max grad-norm on field params (PhysDreamer-style loose)
    weight_decay: float = 0.0  # AdamW weight decay (PhysDreamer uses 1e-4)
    # window_start: if >0, the loss window GROWS window_start->window over training
    # (PhysDreamer curriculum: the spatial field is identified from accumulated multi-
    # frame motion). 0 => fixed `window`.
    window_start: int = 0
    # two_stage (non-global only): first solve the MEAN v0 with a robust global
    # stage, then init the field AT that solution and refine. The pixel gradient only
    # behaves near the basin (mpm-grad-stability), and a from-scratch field either
    # blows up (triplane) or undershoots (voxel); good-init keeps it in the basin.
    two_stage: bool = True
    stage1_iters: int = 60
    stage1_lr: float = 0.05
    iters: int = 120
    lr: float = 0.05
    # window=1 is the VALIDATED default: each frame is rolled with FULL BPTT
    # (grad_window=ti+1) so v0 gets gradient. window>=2 pulls in the frame-2+ long-
    # horizon MPM gradient, which is biased/unstable (drifts v0 to a wrong basin at
    # HIGHER loss); window=1 (frame-1, 64 substeps) descends cleanly to GT
    # (06_tele: l2_err 0.006, angle 0.4deg). With gravity off, frame 1 is pure-v0.
    window: int = 1
    out: Optional[str] = None
    run_label: str = ""
