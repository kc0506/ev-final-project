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
(no timestamp in the name). The entrypoint passes its own `__name__` to
`RunDir.create(__name__, ...)` (explicit, greppable -- no stack introspection);
pass `out=` to override the auto-placement entirely (escape hatch for scratch).

Event log
---------
Each run dir keeps a `.events.txt` timeline. Every RunDir write method appends a
timestamped *semantic* event (created / config.json / video.mp4 / metrics.json
...) as it happens -- no filesystem watcher, because RunDir is the single write
choke-point. `finish()` then seals the dir: any top-level file written *outside* a
RunDir method (e.g. `np.save` / `plt.savefig` via `.path()`) is appended in
file-mtime order, so the log is complete. `note(msg)` logs a custom line.
"""

from __future__ import annotations

import functools
import json
import os
import re
import subprocess
import sys
import threading
import traceback
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:  # annotations only (kept off the runtime import graph by
    import torch  # `from __future__ import annotations`); torch/matplotlib stay
    from matplotlib.figure import Figure  # lazy inside the methods that use them.

_RUN_PREFIX_RE = re.compile(r"^(\d+)(?:_|$)")  # matches "07" and "07_label"


class _FdTee:
    """FD-level tee of stdout+stderr to a log file.

    Redirects the PROCESS file descriptors 1 and 2 (via os.dup2), not just the
    Python `sys.stdout` object -- so output from C extensions / taichi /
    subprocesses lands in the log too, not only Python-level `print`. A pump
    thread copies the byte stream to BOTH the original terminal and the log
    file, so it is a real tee (output stays live) rather than a swallow.

    Used as a context manager around an entrypoint's run body; the captured log
    therefore covers the whole run including any uncaught traceback.
    """

    def __init__(self, log_path: str):
        self.log_path = log_path

    def __enter__(self) -> "_FdTee":
        self._log = open(self.log_path, "ab", buffering=0)  # unbuffered: crash-safe
        sys.stdout.flush()
        sys.stderr.flush()
        self._saved_out = os.dup(1)  # keep the real terminal fds to tee back to
        self._saved_err = os.dup(2)
        r, w = os.pipe()
        os.dup2(w, 1)  # fd 1 and 2 now both feed the pipe...
        os.dup2(w, 2)
        os.close(w)  # ...so only fd 1/2 hold the write end (EOF when both restored)
        self._pump = threading.Thread(target=self._drain, args=(r,), daemon=True)
        self._pump.start()
        return self

    def _drain(self, r: int) -> None:
        with os.fdopen(r, "rb", buffering=0) as pipe:
            for chunk in iter(lambda: pipe.read(65536), b""):
                self._log.write(chunk)  # persisted log (priority)
                try:
                    os.write(self._saved_out, chunk)  # live terminal (best-effort)
                except OSError:
                    pass

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:  # capture the crash INTO the log (still piped)
            traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self._saved_out, 1)  # restore real fds; drops the pipe write ends...
        os.dup2(self._saved_err, 2)
        self._pump.join()  # ...so pipe hits EOF; pump drains the tail to saved_out
        os.close(self._saved_out)  # close only AFTER the pump is done using it
        os.close(self._saved_err)
        self._log.close()
        return False  # don't suppress


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
    name = module[len(prefix) :] if module.startswith(prefix) else module
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
    clips,  # List[np.ndarray], each [T,H,W,C] uint8
    labels,  # List[str]
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

    import imageio
    import numpy as np
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
                    d.rectangle(
                        [0, 0, tile_w - 1, tile_h - 1], outline=(0, 170, 0), width=3
                    )
                tiles.append(np.asarray(im))
            else:
                tiles.append(np.full((tile_h, tile_w, 3), 255, np.uint8))
        rows = [
            np.concatenate(tiles[r * ncols : (r + 1) * ncols], axis=1)
            for r in range(nrows)
        ]
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
        self._logged: set = set()  # basenames already in .events.txt (for finish())

    @classmethod
    def create(
        cls, module: str, label: str = "", out: Optional[str] = None, config=None
    ) -> "RunDir":
        """Build a run dir following the output-tree convention (see module docstring).

        `module` is the entrypoint's `__name__`; the task subpath is derived from
        it. `out`, if given, is used verbatim (bypasses the auto convention).
        Opens the `.events.txt` timeline with a `created` event.

        If `config` (a dataclass, e.g. the tyro-parsed task config) is passed, its
        resolved values are AUTO-SAVED to config.json right here -- the entrypoint
        no longer hand-writes the base config. Run-specific extras (derived facts,
        reconstructed sub-config) are added afterwards via `merge_config(**extra)`.
        Returns an instance of `cls`.
        """
        root = out or next_run_dir(task_subpath_from_module(module), label)
        rd = cls(root)
        rd._event("created")

        def _config_payload(cfg, task: str, **derived) -> dict:
            d = asdict(cfg) if is_dataclass(cfg) else dict(cfg)
            return {"task": task, **d, **derived}

        if config is not None:
            rd.write_config(_config_payload(config, task_subpath_from_module(module)))
        return rd

    # ---- event log (.events.txt) ------------------------------------------- #
    def _append(self, line: str) -> None:
        with open(self.path(".events.txt"), "a") as f:
            f.write(line + "\n")

    def _event(self, msg: str, *files: str) -> None:
        """Append a timestamped semantic event; register `files` as already-logged
        so finish()'s mtime seal does not double-count them."""
        self._logged.update(files)
        self._append(f"{datetime.now().isoformat(timespec='seconds')}  {msg}")

    def note(self, msg: str) -> None:
        """Log a custom event line (e.g. progress) to .events.txt."""
        self._event(msg)

    def finish(self) -> None:
        """Seal the timeline: append anything written OUTSIDE a RunDir method that
        is not yet logged -- top-level files (np.save / plt.savefig via .path()) by
        name, and immediate sub-DIRS as a one-line summary (so result dirs like
        v_000/ show up without spamming a line per nested frame). Then 'finished'.
        """
        seal = []  # (mtime, register_name, display)
        with os.scandir(self.root) as it:
            for e in it:
                if e.name == ".events.txt" or e.name in self._logged:
                    continue
                mt = e.stat(
                    follow_symlinks=False
                ).st_mtime  # link's own mtime, not target's
                if e.is_file():  # follows symlinks: a symlink->file counts as a file
                    seal.append((mt, e.name, e.name))
                elif e.is_dir():  # one summary per subdir, NOT per nested file
                    try:
                        n = sum(1 for _ in os.scandir(e.path))
                    except OSError:
                        n = 0
                    seal.append((mt, e.name, f"{e.name}/ ({n} items)"))
        for mt, name, display in sorted(seal):
            self._logged.add(name)
            self._append(
                f"{datetime.fromtimestamp(mt).isoformat(timespec='seconds')}  (seal) {display}"
            )
        self._event("finished")
        # sort the whole timeline chronologically: seal lines carry real file
        # mtimes that can predate later live events (ISO ts prefix sorts by time;
        # stable sort keeps same-second insertion order, 'finished' stays last).
        p = self.path(".events.txt")
        lines = sorted(
            (l for l in open(p).read().splitlines() if l), key=lambda l: l[:19]
        )
        with open(p, "w") as f:
            f.write("\n".join(lines) + "\n")

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
        self._event("source_ply", "source_ply")

    def copy_in(self, src: str, name: str) -> None:
        """COPY an external file into the run dir (immutable snapshot). Unlike a
        symlink, this survives the source being later rebuilt/deleted -- used to
        freeze the exact scene-discretisation cache each run actually used, since
        the shared cache is non-deterministic and gets rebuilt."""
        import shutil

        if src and os.path.exists(src):
            shutil.copy2(src, self.path(name))
            self._event(name, name)

    def capture_output(self, name: str = "console.log") -> _FdTee:
        """Tee this process's stdout+stderr (FD level) into <root>/<name> for the
        duration of the `with` block. Use it to wrap an entrypoint's run body so the
        run dir keeps the full console transcript (taichi/C-ext output included)."""
        self._event(name, name)
        return _FdTee(self.path(name))

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
        self._event("config.json", "config.json")

    def merge_config(self, **extra) -> None:
        """Merge run-specific extras into the already-written config.json (derived
        facts / reconstructed sub-config not present in the auto-saved dataclass)."""
        p = self.path("config.json")
        d = json.load(open(p)) if os.path.exists(p) else {}
        d.update(extra)
        with open(p, "w") as f:
            json.dump(d, f, indent=2, default=str)
        self._event("config.json (+extra)", "config.json")

    def write_json(self, name: str, obj: dict):
        with open(self.path(name), "w") as f:
            json.dump(obj, f, indent=2, default=str)
        self._event(name, name)

    def savefig(self, name: str, fig: Optional[Figure], dpi: int = 120) -> None:
        """Persist a matplotlib Figure to <root>/<name> and close it. The plot helper
        builds the Figure (pure, no IO); the run dir owns where it lands -- business
        code never spells out a path. No-op if `fig` is None (helper skipped, e.g. no
        matplotlib), so callers don't guard."""
        if fig is None:
            return
        import matplotlib.pyplot as plt

        fig.savefig(self.path(name), dpi=dpi)
        plt.close(fig)
        self._event(name, name)

    def save_named_video(self, subdir: str, vid_uint8: np.ndarray, fps: int):
        """Save a video (mp4+gif+frames) into <root>/<subdir>/. Returns that dir.

        Used to persist sweep intermediates (e.g. every candidate-E render) so a
        landscape/grid result keeps the actual videos behind each data point,
        not just the scalar metric.
        """
        sub = RunDir(os.path.join(self.root, subdir))
        sub.save_video(vid_uint8, fps=fps)
        self._event(f"{subdir}/ (video, {len(vid_uint8)} frames)", subdir)
        return sub.root

    def save_video(
        self,
        vid_uint8: np.ndarray,
        fps: int,
        stem: str = "video",
        frames: bool = True,
        gif: bool = True,
    ):
        """vid_uint8: [T,H,W,C]. Writes mp4 (+ gif + per-frame pngs unless disabled).

        frames/gif default True (back-compat). Set both False for the light
        per-sample IO of large datasets: just the mp4. video.npy + mpm_xyz.npy
        carry the data; frames/plys are reconstructable on demand."""
        import imageio
        import mediapy

        mp4 = self.path(f"{stem}.mp4")
        gif_path = self.path(f"{stem}.gif")
        mediapy.write_video(mp4, vid_uint8, fps=fps)
        if gif:
            try:
                imageio.mimsave(gif_path, list(vid_uint8), fps=fps, loop=0)
            except Exception:
                pass
        if frames:
            for t, fr in enumerate(vid_uint8):
                imageio.imwrite(os.path.join(self.frames_dir, f"frame_{t:03d}.png"), fr)
        self._event(
            f"{stem}.mp4 ({len(vid_uint8)} frames)",
            f"{stem}.mp4",
            *([f"{stem}.gif"] if gif else []),
        )
        return mp4, gif_path


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


def entrypoint(
    run_cls,
    *,
    label=None,
    pick_gpu: bool = False,
    capture: str = "console.log",
    context=None,
):
    """Wrap an entrypoint body with the full run-dir lifecycle, so the
    three identical-everywhere boilerplate steps (create / capture_output / finish)
    live in one place instead of being re-typed per entrypoint:

      - (optional) pick a free GPU *before* any CUDA context is built
      - create the task RunDir (auto-saving config.json) under the convention tree
      - tee stdout+stderr into the run dir for the whole body (FD-level)
      - (optional) materialise a task CONTEXT and inject it instead of the bare rd
      - seal the dir (finish) on the way out, EVEN on exception

    Keeps the external `run(cfg) -> RunDir` signature. The decorated body takes
    `(cfg, rd)` by default; if `context` is given it is called as
    `context(cfg, rd)` (inside capture_output, AFTER pick_gpu so its lazy torch/
    scene imports happen post-GPU-pick) and its return value is injected instead --
    so the body is `fn(cfg, ctx)` and reaches `rd` via `ctx.rd`. This is the DI
    seam: the context factory owns shared preprocessing + provenance side-effects
    (e.g. recover_context freezes the discretisation cache), so the body cannot
    forget them. `label` is a str or a callable cfg->str (defaults to
    cfg.run_label). The task subpath is derived from `fn.__module__` (== `__name__`
    of the entrypoint module; under `python -m` that is "__main__", which
    task_subpath_from_module resolves via __spec__ -- no stack introspection)."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(cfg) -> RunDir:
            if pick_gpu:
                from .gpu import pick_free_gpu

                pick_free_gpu()
            lbl = (
                label(cfg)
                if callable(label)
                else label or getattr(cfg, "run_label", "") or ""
            )
            rd = run_cls.create(
                fn.__module__, lbl, getattr(cfg, "out", None), config=cfg
            )
            with rd.capture_output(capture):
                if context is None:
                    try:
                        fn(cfg, rd)
                    finally:
                        rd.finish()  # seal even on crash (traceback still tees)
                else:
                    ctx = context(cfg, rd)  # the DI payload (a RunContext)
                    try:
                        fn(cfg, ctx)
                    finally:
                        # lifetime end: the context persists its OWN derived state
                        # (context.json) before the dir is sealed -- explicit hook,
                        # part of the context contract. Runs even on crash, so a
                        # partial run still records what it ran against.
                        ctx.seal()
                        rd.finish()
            return rd

        return wrapper

    return deco


class ForwardRun(RunDir):
    """forward_gen deliverables: config.json, source_ply, frames/, video.{mp4,gif}, result.json."""

    def video(self, vid_u8: np.ndarray, fps: int):
        return self.save_video(vid_u8, fps=fps)

    def result(self, **obj) -> None:
        self.write_json("result.json", obj)


class RecoverRun(RunDir):
    """train_global_E deliverables: config.json, source_ply, gt/ pred_init/
    pred_recovered/ gt_vs_recovered/ videos, metrics.json, trace.json, recovery.png."""

    # gt_video / pred_videos take the RAW render tensors ([T,C,H,W] float in [0,1],
    # as simulate_and_render returns) -- the uint8 encoding is an IO detail owned by
    # this write boundary, NOT marshalled by the recovery body. (video_to_uint8 also
    # detaches, so the body need not.)
    def gt_video(self, gt: torch.Tensor, fps: int) -> None:
        from .sim_render import video_to_uint8

        self.save_named_video("gt", video_to_uint8(gt), fps)

    def pred_videos(
        self,
        init: torch.Tensor,
        recovered: torch.Tensor,
        gt: torch.Tensor,
        fps: int,
    ) -> None:
        """Save the init-guess and recovered renders, plus a GT|recovered montage."""
        from .sim_render import video_to_uint8

        init_u8, rec_u8, gt_u8 = (video_to_uint8(v) for v in (init, recovered, gt))
        self.save_named_video("pred_init", init_u8, fps)
        self.save_named_video("pred_recovered", rec_u8, fps)
        T = min(gt_u8.shape[0], rec_u8.shape[0])
        self.save_named_video(
            "gt_vs_recovered",
            np.concatenate([gt_u8[:T], rec_u8[:T]], axis=2),
            fps,
        )

    def recovery_plot(self, fig: Optional[Figure]) -> None:
        """The per-run recovery diagnostic -> recovery.png. The body hands over the
        Figure its task-specific plot helper built; this declares the artifact NAME
        (the body never spells out a path). No-op if the helper skipped (None)."""
        self.savefig("recovery.png", fig)

    def metrics(self, **obj) -> None:
        self.write_json("metrics.json", obj)

    def trace(self, E_traj, loss_traj) -> None:
        self.write_json("trace.json", {"E": E_traj, "loss": loss_traj})


class DatasetRun(RunDir):
    """dataset_gen deliverables (top-level dir): config.json, manifest.json,
    source_ply, scene_cache (symlink), p_star.png, sample_XXXX/ subdirs."""

    def link(self, target: str, name: str) -> None:
        dst = self.path(name)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(target), dst)
        self._event(name, name)

    def sample_dir(self, i: int) -> "RunDir":
        return RunDir(self.path(f"sample_{i:04d}"))

    def manifest(self, obj: dict) -> None:
        self.write_json("manifest.json", obj)
