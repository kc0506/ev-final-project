"""Materialise a recover entrypoint's context from an upstream forward_gen run.

The recover family (train_global_E / train_field_E / train_v0) all start the same
way: point at a forward_gen run dir, read its config.json back into the typed
ForwardConfig (scene + sim + v0 + frame + true E), load its frames as the GT, and
rebuild the EXACT scene that produced them. That block was copy-pasted verbatim
across all three (and several explore scripts), which meant the provenance
side-effects -- freezing this run's non-deterministic discretisation cache
(`copy_in`) and symlinking the source ply -- were a step each entrypoint had to
REMEMBER. Forgetting `copy_in` silently breaks reproducibility: the shared k-means
cache is non-deterministic and gets rebuilt, so only the per-run frozen copy
survives.

`recover_context` does that block ONCE and returns a `RecoverContext`. You cannot
obtain `ctx.scene` without the provenance writes having happened -- the side
effects are wired into materialisation, not left to the caller to remember.

The GT run's config is held as a typed `ForwardConfig` (via
`ForwardConfig.from_run_json`), never a raw dict: no stringly-typed `g["E"]`
indexing that drifts or typos.
"""

from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol, Tuple

import numpy as np

from .config import ForwardConfig, SceneSpec, SimConfig

if TYPE_CHECKING:  # types only -- `from __future__ import annotations` keeps these
    import torch  # out of the runtime import graph, so loading this module pulls
    from ._env import Camera  # neither torch nor scene (preserves GPU-pick-before-
    from .run_io import RecoverRun  # CUDA: the heavy imports stay lazy in run()).
    from .scene import SceneBundle


class HasGtRun(Protocol):
    """The only field recover_context needs off a task config: the GT run dir.
    All three recover configs (RecoverConfig / RecoverFieldConfig / RecoverV0Config)
    satisfy this structurally, so the materialiser stays generic over them."""

    gt_run: str


def load_gt_frames(gt_run: str, device: str) -> torch.Tensor:
    """gt_run/frames/*.png -> [T, C, H, W] float in [0, 1] on `device`."""
    frame_files = sorted(glob.glob(os.path.join(gt_run, "frames", "frame_*.png")))
    assert frame_files, f"no frames in {gt_run}/frames"
    import imageio.v2 as imageio
    import torch

    frames = [imageio.imread(fp) for fp in frame_files]  # list[H,W,C] uint8
    arr = np.stack(frames, 0).astype(np.float32) / 255.0  # [T,H,W,C]
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device)  # [T,C,H,W]


def load_run_field(run_dir: str, name: str) -> Optional[np.ndarray]:
    """Read an optional per-particle field artifact (E_field.npy / v0_field.npy) from
    a run dir; None if this GT carried no spatial field (phase A uniform GT). The one
    place that knows these are optional, so consumers don't re-spell the path or the
    `os.path.exists` guard."""
    p = os.path.join(run_dir, name)
    return np.load(p) if os.path.exists(p) else None


@dataclass
class RecoverContext:
    """Everything a recover entrypoint needs, materialised from a GT run dir.

    Built by `recover_context` and injected into the entrypoint body in place of a
    bare RunDir. Holds the rebuilt scene/cam plus the loaded GT, so the body goes
    straight to optimisation. The provenance side-effects (frozen cache copy,
    source-ply link, GT video) already ran during construction, and the derived
    facts get sealed to context.json at lifetime end -- neither is the body's job.
    """

    rd: RecoverRun  # the open run dir for THIS recover run
    gt_run: str  # the upstream forward_gen run dir this was rebuilt from
    gt_config: ForwardConfig  # GT run's config.json, typed (scene/sim/E/v0/frame)
    scene: SceneBundle  # scene rebuilt from the GT spec
    cam: Camera  # camera matching the GT frame (pd) or first test cam (pg)
    v0: torch.Tensor  # [n, 3] GT constant v0 on device, detached
    gt: torch.Tensor  # [T, C, H, W] GT frames in [0, 1] on device
    device: str
    t0: float  # materialisation start time (for elapsed_sec)

    # Typed views onto the GT config -- the body reads these, never a raw dict.
    @property
    def scene_spec(self) -> SceneSpec:
        return self.gt_config.scene

    @property
    def sim(self) -> SimConfig:
        return self.gt_config.sim

    @property
    def true_E(self) -> float:
        """The GT global scalar E (phase A); for field tasks, the geomean target."""
        return float(self.gt_config.E)

    @property
    def gt_v0(self) -> Tuple[float, float, float]:
        """The GT constant v0 vector [3] (for error reporting / known-v0 tasks)."""
        return self.gt_config.v0

    @property
    def frame(self) -> str:
        return self.gt_config.frame

    def to_dict(self) -> dict:
        """The reconstructed/derived facts this context rebuilt from the GT run.

        Kept SEPARATE from config.json (which is the *input* RecoverConfig): this is
        the reproducibility record of what the inverse actually ran against -- which
        scene, which discretisation (frozen alongside as scene_cache.pt), which sim
        regime, and the GT's own E / v0. Serialised by `seal()` at lifetime end.
        """
        return {
            "gt_run": self.gt_run,
            "scene": self.scene_spec.to_dict(),
            "sim": self.sim.to_dict(),
            "scene_name": self.scene.name,
            "n_mpm_particles": int(self.scene.sim_xyzs.shape[0]),
            "gt_E": self.true_E,
            "gt_v0": self.gt_v0,
            "frame": self.frame,
        }

    def seal(self) -> None:
        """Persist the derived facts to context.json. Called by the `entrypoint`
        decorator at the end of the run lifetime (before finish seals the dir) -- the
        context records its OWN reconstructed state, so no entrypoint body ever
        hand-writes this provenance or can forget to."""
        self.rd.write_json("context.json", self.to_dict())


def recover_context(cfg: HasGtRun, rd: RecoverRun) -> RecoverContext:
    """Read `cfg.gt_run`, rebuild scene+cam, load the GT, and record provenance.

    Side-effects (all guaranteed, none optional) -- this is the whole point, the
    consumer cannot forget them:
      - `copy_in` the resolved discretisation cache (frozen snapshot, reproducible)
      - symlink the source ply (pd scenes only; pg has none at the dir root)
      - write the GT video deliverable (gt/)

    The reconstructed sub-config (scene/sim/n_particles/gt_E/gt_v0) is NOT written
    here -- the returned context serialises it itself to context.json at lifetime
    end (see RecoverContext.to_dict / seal), keeping derived facts out of the input
    config.json.

    The discretisation cache path is deterministic from (scene, downsample, grid,
    top_k), all read from the GT config -- so `load_from_spec` re-derives the SAME
    path the GT run created and loads the identical particles.

    Returns the RecoverContext the entrypoint body consumes.
    """
    from .scene_io import load_from_spec
    from .sim_render import make_constant_v0

    t0 = time.time()
    with open(os.path.join(cfg.gt_run, "config.json")) as f:
        gt_config = ForwardConfig.from_run_json(json.load(f))
    scene_spec, sim = gt_config.scene, gt_config.sim
    device = scene_spec.device

    gt = load_gt_frames(cfg.gt_run, device)  # [T,C,H,W]
    scene = load_from_spec(scene_spec, sim)  # resolves cache_path
    rd.copy_in(scene_spec.cache_path, "scene_cache.pt")  # freeze discretisation
    if scene_spec.kind == "pd":
        rd.link_source_ply(scene_spec.path)
    try:
        cam = scene.camera_by_frame(gt_config.frame)
    except Exception:
        cam = scene.test_camera_list[0]  # PG cameras (r_0, ...) won't match frame_*
    v0 = make_constant_v0(scene, gt_config.v0).detach()  # [n,3]

    rd.gt_video(gt, fps=sim.fps)
    return RecoverContext(
        rd=rd,
        gt_run=cfg.gt_run,
        gt_config=gt_config,
        scene=scene,
        cam=cam,
        v0=v0,
        gt=gt,
        device=device,
        t0=t0,
    )
