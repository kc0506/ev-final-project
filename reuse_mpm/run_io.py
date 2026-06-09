"""Reproducible run-directory management.

Goal contract: one output dir == everything about that run.
  - config.json        (full resolved config, incl. git-ish provenance)
  - source_ply         (symlink to the point_cloud.ply actually used)
  - frames/            (every rendered frame as png)
  - video.mp4, video.gif
  - (training adds) curves.png, metrics.json, ...

This applies to *deliverables*. Throwaway debug artifacts are exempt.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Optional

import numpy as np


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
