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
    # pg-only geometric anchor BC
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
    E: float  # constant Young's modulus (required)
    sim: SimConfig = field(default_factory=SimConfig)
    v0: Tuple[float, float, float] = (0.0, -0.5, 0.0)
    frame: str = "frame_00001.png"  # camera image filename
    out: Optional[str] = None
    run_label: str = ""


@dataclass
class DatasetConfig:
    """dataset_gen: sample E~p*(E)=logU[E_min,E_max] -> (E, video) dataset."""

    scene: SceneSpec
    sim: SimConfig = field(default_factory=lambda: SimConfig(num_frames=8, substep=32))
    E_min: float = 1e4
    E_max: float = 1e6
    n: int = 16
    v0: Tuple[float, float, float] = (0.0, -1.0, 0.0)
    frame: str = "frame_00001.png"
    seed: int = 0
    jump_thresh: float = 0.5  # per-frame max normalised single-step jump -> unstable flag
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
