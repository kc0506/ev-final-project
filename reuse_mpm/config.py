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

from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple


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

    `kind` selects the loader: "pd" = PhysDreamer dataset dir (foreground is
    segmented from a full scene via clean/moving plys), "pg" = PhysGaussian model
    dir (all gaussians are the object; anchor BC is a geometric bottom slab).

    `grid_size` is intentionally NOT here: it must equal SimConfig.grid_size, so
    the loader reads it from the SimConfig at load time (single source of truth).
    """

    path: str  # dataset_dir (pd) or model_dir (pg)
    kind: str = "pd"  # "pd" | "pg"
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
