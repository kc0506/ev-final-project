"""Reproducible run-directory management.

Goal contract: one output dir == everything about that run.
  - config.json        (full resolved config, incl. git-ish provenance)
  - source_ply         (symlink to the point_cloud.ply actually used)
  - frames/            (every rendered frame as png)
  - video.mp4, video.gif
  - (training adds) curves.png, metrics.json, ...

This applies to *deliverables*. Throwaway debug artifacts are exempt.

Output-tree convention
----------------------
Runs are auto-placed at  outputs/<task>/<NN>[_<label>]/  where <task> is derived
from the entrypoint's module path (so it never drifts from a rename/move):

    reuse_mpm.forward_gen        -> outputs/forward_gen/01_.../
    reuse_mpm.explore.gradcheck  -> outputs/explore/gradcheck/01/

`NN` auto-increments within each <task> dir so `ls` shows run order at a glance
(no timestamp in the name -- the wall-clock time is written to a `started_at.txt`
inside each run dir instead). The entrypoint passes its own `__name__` to
`RunDir.create(__name__, ...)` (explicit, greppable -- no stack introspection);
pass `out=` to override the auto-placement entirely (escape hatch for scratch).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from typing import Optional

import numpy as np

_RUN_PREFIX_RE = re.compile(r"^(\d+)(?:_|$)")  # matches "07" and "07_label"


def task_subpath_from_module(module: str, pkg: str = "reuse_mpm") -> str:
    """'reuse_mpm.forward_gen' -> 'forward_gen';
    'reuse_mpm.explore.gradcheck' -> 'explore/gradcheck'.

    Under `python -m pkg.mod` the entrypoint's `__name__` is "__main__"; recover
    the real dotted path from the __main__ module's import spec.
    """
    if module == "__main__":
        spec = getattr(sys.modules.get("__main__"), "__spec__", None)
        if spec is not None:
            module = spec.name
    prefix = pkg + "."
    name = module[len(prefix):] if module.startswith(prefix) else module
    return name.replace(".", "/")


def next_run_dir(task_subpath: str, label: str = "", *, root: str = "outputs") -> str:
    """Auto-incrementing run dir under outputs/<task_subpath>/.

    Returns outputs/<task_subpath>/<NN>[_<label>] (not yet created -- RunDir makes
    it). NN = 1 + max existing prefix in the task dir, so listing sorts by run
    order. No timestamp in the name (RunDir.create writes started_at.txt instead).
    Single-user box: the count scan is not locked (a simultaneous double-launch
    could collide).
    """
    base = os.path.join(root, task_subpath)
    os.makedirs(base, exist_ok=True)
    nmax = 0
    for d in os.listdir(base):
        m = _RUN_PREFIX_RE.match(d)
        if m and os.path.isdir(os.path.join(base, d)):
            nmax = max(nmax, int(m.group(1)))
    label = re.sub(r"[^0-9A-Za-z._-]+", "-", label).strip("-")
    name = f"{nmax + 1:02d}" + (f"_{label}" if label else "")
    return os.path.join(base, name)


def _git_describe(path: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "-C", path, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def save_panel_video(
    out_path: str,
    clips,                 # List[np.ndarray], each [T,H,W,C] uint8
    labels,                # List[str]
    fps: int,
    ncols: Optional[int] = None,
    tile_w: int = 256,
    highlight: Optional[int] = None,
    title: Optional[str] = None,
) -> str:
    """Tile several clips into one grid gif/mp4 so they can be compared at a glance.

    Each tile is downscaled to `tile_w` and labelled; `highlight` draws a green
    border (e.g. the true E*). Clips may differ in length (clipped to the min).
    """
    import math
    import numpy as np
    import imageio
    from PIL import Image, ImageDraw

    n = len(clips)
    ncols = ncols or int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))
    T = min(c.shape[0] for c in clips)
    H, W = clips[0].shape[1:3]
    tile_h = max(1, round(H * tile_w / W))

    panel_frames = []
    for t in range(T):
        tiles = []
        for i in range(nrows * ncols):
            if i < n:
                im = Image.fromarray(clips[i][t]).resize((tile_w, tile_h))
                d = ImageDraw.Draw(im)
                col = (0, 170, 0) if highlight == i else (220, 30, 30)
                d.text((4, 2), labels[i], fill=col)
                if highlight == i:
                    d.rectangle([0, 0, tile_w - 1, tile_h - 1], outline=(0, 170, 0), width=3)
                tiles.append(np.asarray(im))
            else:
                tiles.append(np.full((tile_h, tile_w, 3), 255, np.uint8))
        rows = [np.concatenate(tiles[r * ncols:(r + 1) * ncols], axis=1)
                for r in range(nrows)]
        panel = np.concatenate(rows, axis=0)
        if title:
            pim = Image.fromarray(panel)
            ImageDraw.Draw(pim).text((4, tile_h * nrows - 12), title, fill=(0, 0, 0))
            panel = np.asarray(pim)
        panel_frames.append(panel)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    imageio.mimsave(out_path, panel_frames, fps=fps, loop=0)
    return out_path


@dataclass
class RunDir:
    root: str

    def __post_init__(self):
        os.makedirs(self.root, exist_ok=True)

    @classmethod
    def create(cls, module: str, label: str = "", out: Optional[str] = None) -> "RunDir":
        """Build a run dir following the output-tree convention (see module docstring).

        `module` is the entrypoint's `__name__`; the task subpath is derived from
        it. `out`, if given, is used verbatim (bypasses the auto convention).
        Writes `started_at.txt` (wall-clock time, kept out of the dir name).
        Returns an instance of `cls` (so subclasses keep their schema methods).
        """
        root = out or next_run_dir(task_subpath_from_module(module), label)
        rd = cls(root)
        with open(rd.path("started_at.txt"), "w") as f:
            f.write(datetime.now().isoformat(timespec="seconds") + "\n")
        return rd

    @property
    def frames_dir(self):
        # created lazily by save_video; non-video tasks won't leave an empty dir
        d = os.path.join(self.root, "frames")
        os.makedirs(d, exist_ok=True)
        return d

    def path(self, *parts):
        return os.path.join(self.root, *parts)

    def link_source_ply(self, dataset_dir: str):
        src = os.path.abspath(os.path.join(dataset_dir, "point_cloud.ply"))
        dst = self.path("source_ply")
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)

    def write_config(self, cfg: dict):
        cfg = dict(cfg)
        # repo root = parent of the reuse_mpm package dir (where .git lives)
        _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg["_provenance"] = {
            "reuse_mpm_git": _git_describe(_repo_root),
            "physdreamer_git": _git_describe(
                os.environ.get("PHYSDREAMER_ROOT", "/tmp2/b10401006/PhysDreamer")
            ),
        }
        with open(self.path("config.json"), "w") as f:
            json.dump(cfg, f, indent=2, default=str)

    def write_json(self, name: str, obj: dict):
        with open(self.path(name), "w") as f:
            json.dump(obj, f, indent=2, default=str)

    def save_named_video(self, subdir: str, vid_uint8: np.ndarray, fps: int):
        """Save a video (mp4+gif+frames) into <root>/<subdir>/. Returns that dir.

        Used to persist sweep intermediates (e.g. every candidate-E render) so a
        landscape/grid result keeps the actual videos behind each data point,
        not just the scalar metric.
        """
        sub = RunDir(os.path.join(self.root, subdir))
        sub.save_video(vid_uint8, fps=fps)
        return sub.root

    def save_video(self, vid_uint8: np.ndarray, fps: int, stem: str = "video"):
        """vid_uint8: [T,H,W,C]. Writes mp4 + gif + per-frame pngs."""
        import mediapy

        mp4 = self.path(f"{stem}.mp4")
        gif = self.path(f"{stem}.gif")
        mediapy.write_video(mp4, vid_uint8, fps=fps)
        try:
            mediapy.write_image(gif, vid_uint8[0])  # placeholder if gif unsupported
        except Exception:
            pass
        # robust gif via imageio
        try:
            import imageio

            imageio.mimsave(gif, list(vid_uint8), fps=fps, loop=0)
        except Exception:
            pass
        # per-frame pngs
        import imageio

        for t, fr in enumerate(vid_uint8):
            imageio.imwrite(os.path.join(self.frames_dir, f"frame_{t:03d}.png"), fr)
        return mp4, gif


# --------------------------------------------------------------------------- #
# Declarative per-task run dirs
#
# Each subclass DECLARES (in its docstring + named methods) the exact set of
# artifacts a task produces, so the output schema lives in one place instead of
# being scattered as free-form `write_json`/`save_video` calls in the entrypoint.
# Writes are still incremental (no buffering of tensors until a final flush);
# `config()` serialises the resolved config DATACLASS verbatim, so config.json's
# schema is identical across tasks (== the dataclass) and never hand-built.
# --------------------------------------------------------------------------- #
def _config_payload(cfg, task: str, **derived) -> dict:
    d = asdict(cfg) if is_dataclass(cfg) else dict(cfg)
    return {"task": task, **d, **derived}


class ForwardRun(RunDir):
    """forward_gen deliverables: config.json, source_ply, frames/, video.{mp4,gif}, result.json."""

    def config(self, cfg, **derived) -> None:
        self.write_config(_config_payload(cfg, "forward_gen", **derived))

    def video(self, vid_u8: np.ndarray, fps: int):
        return self.save_video(vid_u8, fps=fps)

    def result(self, **obj) -> None:
        self.write_json("result.json", obj)


class RecoverRun(RunDir):
    """train_global_E deliverables: config.json, source_ply, gt/ pred_init/
    pred_recovered/ gt_vs_recovered/ videos, metrics.json, trace.json, recovery.png."""

    def config(self, cfg, **derived) -> None:
        self.write_config(_config_payload(cfg, "train_global_E", **derived))

    def gt_video(self, gt_u8: np.ndarray, fps: int) -> None:
        self.save_named_video("gt", gt_u8, fps)

    def pred_videos(self, init_u8, recovered_u8, gt_u8, fps: int) -> None:
        self.save_named_video("pred_init", init_u8, fps)
        self.save_named_video("pred_recovered", recovered_u8, fps)
        T = min(gt_u8.shape[0], recovered_u8.shape[0])
        self.save_named_video(
            "gt_vs_recovered",
            np.concatenate([gt_u8[:T], recovered_u8[:T]], axis=2), fps)

    def metrics(self, **obj) -> None:
        self.write_json("metrics.json", obj)

    def trace(self, E_traj, loss_traj) -> None:
        self.write_json("trace.json", {"E": E_traj, "loss": loss_traj})


class DatasetRun(RunDir):
    """dataset_gen deliverables (top-level dir): config.json, manifest.json,
    source_ply, scene_cache (symlink), p_star.png, sample_XXXX/ subdirs."""

    def config(self, cfg, **derived) -> None:
        self.write_config(_config_payload(cfg, "dataset_gen", **derived))

    def link(self, target: str, name: str) -> None:
        dst = self.path(name)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(target), dst)

    def sample_dir(self, i: int) -> "RunDir":
        return RunDir(self.path(f"sample_{i:04d}"))

    def manifest(self, obj: dict) -> None:
        self.write_json("manifest.json", obj)
