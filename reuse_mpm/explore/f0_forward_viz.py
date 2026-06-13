"""Entrypoint: FORWARD-only viz of a block excitation (the "see a scene" task).

One thin entrypoint for every block forward look: pick a scene from _block.SCENES
(release / drop / freefall / squeeze / uniform), roll it forward, and draw the
standard set via the shared _viz helpers. Replaces the per-scene forward scripts
(dynamic_pull = release, asym_squeeze = squeeze, squeeze_forward_viz = release multi-R,
block_E_overlay = release multi-E) -- they all had the SAME upstream (Scene + rollout),
differing only in which viz. Forward ONLY: no fit here (that is f0_fit_case).

Modes (compose freely):
  default              one Scene at gt_logE -> panel + observables + F0 stretch + gif
  --release-frames a b multi-R overlay: one Scene per release frame, GT rollouts overlaid
  --e-list a b c       multi-E overlay: release the snapshot at several E + divergence

Outputs (outputs/explore/f0_forward_viz/<label>/):
  forward_panel.png, observables.png, F0_stretch.png, forward_3d_triplane.gif, forward.npz
  [multi-R]  overlay_R_3d.gif, overlay_R_panel.png
  [multi-E]  overlay_E_3d.gif, overlay_E_panel.png, divergence.png
"""
from __future__ import annotations

import os
import time as _time
from dataclasses import dataclass
from typing import Tuple

import tyro

from ..gpu import pick_free_gpu

_PALETTE = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


@dataclass
class ForwardVizConfig:
    scene: str = "release"            # release | drop | freefall | squeeze | uniform
    # geometry (shared with f0_fit_case)
    nx: int = 22
    ny: int = 9
    nz: int = 16
    half: Tuple[float, float, float] = (0.18, 0.08, 0.14)
    z_base: float = 0.30
    # pull (release/drop/freefall F0)
    pull_speed: float = 0.5
    grip_half_x: float = 0.045
    release_frames: Tuple[int, ...] = (5,)   # >1 entry => multi-R overlay
    # squeeze
    push_x: float = 0.60
    push_half_x: float = 0.07
    push_half_z: float = 0.045
    push_speed: float = 0.45
    push_frames: int = 5
    # drop
    floor_z: float = 0.25
    gravity: float = 9.8
    collider: str = "slip"
    friction: float = 0.0
    # uniform
    S_gt: Tuple[float, float, float, float, float, float] = (0.2, -0.1, -0.1, 0.05, 0.0, 0.0)
    # physics / horizon
    gt_logE: float = 4.5
    nu: float = 0.3
    K: int = 32
    e_list: Tuple[float, ...] = ()    # non-empty => multi-E overlay from the snapshot
    gif_fps: int = 6
    overlay_fps: int = 3
    min_quota_hours: float = 8.0
    label: str = "fwd"


def run(cfg: ForwardVizConfig) -> str:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        from ..gpu import assert_gpu_quota
        assert_gpu_quota(cfg.min_quota_hours)
        print(f"[fwd] preset CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        pick_free_gpu(min_quota_hours=cfg.min_quota_hours)
    import numpy as np
    import torch
    import warp as wp
    wp.init()

    from ._block import Scene, SCENES
    from . import _viz

    t0 = _time.time()
    assert cfg.scene in SCENES, f"unknown scene {cfg.scene!r} (have {list(SCENES)})"
    out_dir = os.path.join("outputs", "explore", "f0_forward_viz", cfg.label)
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda:0"

    def make_scene(release_frame):
        return Scene(cfg.scene, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, half=cfg.half, z_base=cfg.z_base,
                     nu=cfg.nu, gt_logE=cfg.gt_logE, pull_speed=cfg.pull_speed,
                     release_frame=release_frame, grip_half_x=cfg.grip_half_x,
                     push_x=cfg.push_x, push_half_x=cfg.push_half_x, push_half_z=cfg.push_half_z,
                     push_speed=cfg.push_speed, push_frames=cfg.push_frames, gravity=cfg.gravity,
                     floor_z=cfg.floor_z, collider=cfg.collider, friction=cfg.friction,
                     S_gt=cfg.S_gt, device=dev)

    def full_seq(sc):
        """pull + release positions/stretch (mirrors f0_fit_case)."""
        relX, relS = sc.rollout(cfg.gt_logE, cfg.K)
        relX = relX.cpu().numpy(); relS = relS.cpu().numpy()
        pullX = torch.stack(sc.pull_X).cpu().numpy(); pullS = torch.stack(sc.pull_S).cpu().numpy()
        X = np.concatenate([pullX, relX[1:]], 0); S = np.concatenate([pullS, relS[1:]], 0)
        return X, S, pullX.shape[0]

    fz = cfg.floor_z if cfg.scene == "drop" else (cfg.z_base if cfg.scene == "squeeze" else None)

    # ---- primary scene (first release frame): the standard single-forward set ----
    R0 = cfg.release_frames[0]
    sc0 = make_scene(R0)
    X0, S0, rel0 = full_seq(sc0)
    width = X0[:, :, 0].max(1) - X0[:, :, 0].min(1)
    minz = X0[:, :, 2].min(1); comz = X0[:, :, 2].mean(1)
    print(f"[fwd] scene={cfg.scene} R0={R0} maxdev={sc0.maxdev:.4f} "
          f"F0_stretch mean {sc0.F0_stretch.mean():.3f}/max {sc0.F0_stretch.max():.3f}")
    np.savez(os.path.join(out_dir, "forward.npz"), X=X0, stretch=S0, width=width, minz=minz,
             comz=comz, rel_start=rel0, maxdev=sc0.maxdev, scene=cfg.scene,
             floor_z=fz if fz is not None else -1)

    rel_idx = list(range(rel0, len(X0), max(1, (len(X0) - rel0) // 7)))[:8]
    sel = (list(range(rel0)) + rel_idx)[:12]
    _viz.frames_panel(os.path.join(out_dir, "forward_panel.png"), X0, S0, sel=sel, rel_start=rel0,
                      floor_z=fz, width=width,
                      suptitle=f"{cfg.scene}: pull->release (maxdev {sc0.maxdev:.3f}, GT logE {cfg.gt_logE})")
    obs = {"width (x-extent)": width}
    if cfg.scene != "release":
        obs["min z"] = minz; obs["com z"] = comz
    _viz.observables_plot(os.path.join(out_dir, "observables.png"), obs, rel_start=rel0, floor_z=fz,
                          suptitle=f"{cfg.scene} observables (maxdev {sc0.maxdev:.3f})")
    _viz.scalar_scatter(os.path.join(out_dir, "F0_stretch.png"), X0[rel0 - 1], sc0.F0_stretch,
                        title=f"F0 snapshot stretch (mean {sc0.F0_stretch.mean():.3f}, max {sc0.F0_stretch.max():.3f})")
    _viz.triplane_scalar_gif(os.path.join(out_dir, "forward_3d_triplane.gif"), X0, S0, floor_z=fz,
                             fps=cfg.gif_fps,
                             title_fn=lambda f: f"{cfg.scene} f{f}/{len(X0)-1} [{'PULL' if f < rel0 else 'REL'}]")
    print(f"[fwd] single-forward viz -> forward_panel/observables/F0_stretch/forward_3d_triplane")

    # ---- multi-R overlay (replaces squeeze_forward_viz): one Scene per release frame ----
    if len(cfg.release_frames) > 1:
        seqs = [(R0, X0)] + [(R, full_seq(make_scene(R))[0]) for R in cfg.release_frames[1:]]
        L = min(X.shape[0] for _, X in seqs)
        items = [(f"R={R}", _PALETTE[i % len(_PALETTE)], X[:L]) for i, (R, X) in enumerate(seqs)]
        _viz.triplane_overlay_gif(os.path.join(out_dir, "overlay_R_3d.gif"), items, floor_z=fz,
                                  fps=cfg.overlay_fps, title=f"{cfg.scene} R-overlay")
        _viz.overlay_panel(os.path.join(out_dir, "overlay_R_panel.png"), items, floor_z=fz,
                           suptitle=f"{cfg.scene}: release-frame overlay (xz)")
        print(f"[fwd] multi-R overlay -> overlay_R_3d.gif, overlay_R_panel.png (R={list(cfg.release_frames)})")

    # ---- multi-E overlay (replaces block_E_overlay): release the snapshot at several E ----
    if cfg.e_list:
        rolls = {le: sc0.rollout(float(le), cfg.K)[0].cpu().numpy() for le in cfg.e_list}
        items = [(f"logE {le}", _PALETTE[i % len(_PALETTE)], rolls[le]) for i, le in enumerate(cfg.e_list)]
        _viz.triplane_overlay_gif(os.path.join(out_dir, "overlay_E_3d.gif"), items, floor_z=fz,
                                  fps=cfg.overlay_fps, title=f"{cfg.scene} E-overlay")
        _viz.overlay_panel(os.path.join(out_dir, "overlay_E_panel.png"), items, floor_z=fz,
                           suptitle=f"{cfg.scene}: release at several E (xz)")
        # divergence vs gt_logE (if present in the list, else vs the first)
        ref_le = cfg.gt_logE if cfg.gt_logE in cfg.e_list else cfg.e_list[0]
        ref = rolls[ref_le]
        ref_motion = np.linalg.norm(ref[1:] - ref[0:1], axis=-1).mean(1)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
        for i, le in enumerate(cfg.e_list):
            if le == ref_le:
                continue
            d = np.linalg.norm(rolls[le] - ref, axis=-1).mean(1)
            ax[0].plot(d, "-o", ms=3, color=_PALETTE[i % len(_PALETTE)], label=f"logE {le}")
            ax[1].plot(d[1:] / np.maximum(ref_motion, 1e-9), "-o", ms=3,
                       color=_PALETTE[i % len(_PALETTE)], label=f"logE {le}")
        ax[0].set_title(f"mean dist vs logE {ref_le}"); ax[0].set_xlabel("frame"); ax[0].legend()
        ax[1].axhline(0.05, color="k", ls="--", lw=0.7)
        ax[1].set_title("dist / ref-motion (visible if >> 0)"); ax[1].set_xlabel("frame"); ax[1].legend()
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "divergence.png"), dpi=120); plt.close(fig)
        print(f"[fwd] multi-E overlay -> overlay_E_3d.gif, overlay_E_panel.png, divergence.png (E={list(cfg.e_list)})")

    print(f"[fwd] DONE -> {out_dir} ({_time.time()-t0:.1f}s)")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(ForwardVizConfig))
