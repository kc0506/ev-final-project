"""Shared matplotlib helpers for the f0 explore tools.

Every f0 script re-implemented the same xz panel + 3D/triplane FuncAnimation. These
are the extracted, single source of truth so the forward-viz entrypoint, the fit
tools (f0_fit_case, f0_train_S, f0_train_ufield) and any future forward/backward
split all draw the SAME way. PURE plotting: takes numpy arrays, writes PNG/GIF --
no warp, no torch, no Scene (so it stays trivially splittable).

Conventions: X is [T, n, 3] (a rollout), scalar is [T, n] (e.g. per-particle
stretch |sigma-1|). `items` (overlays) is a list of (label, color, X) tuples.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

PROJ = [(0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z")]


def _bounds(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = X.reshape(-1, 3)
    return p.min(0), p.max(0)


def frames_panel(path: str, X: np.ndarray, scalar: np.ndarray, *,
                 sel: Optional[Sequence[int]] = None, rel_start: Optional[int] = None,
                 floor_z: Optional[float] = None, vmax: Optional[float] = None,
                 width: Optional[np.ndarray] = None, suptitle: str = "",
                 cmap: str = "inferno", ncol: int = 6) -> str:
    """Grid of xz scatter frames colored by `scalar`. rel_start tags PULL/REL;
    floor_z draws the collider line; width annotates per-frame x-extent."""
    T = X.shape[0]
    if sel is None:
        sel = list(range(min(T, 12)))
    sel = list(sel)
    if vmax is None:
        vmax = float(np.quantile(scalar, 0.98)) or 1e-3
    mn = X[:, :, [0, 2]].reshape(-1, 2).min(0)
    mx = X[:, :, [0, 2]].reshape(-1, 2).max(0)
    nrow = (len(sel) + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow), squeeze=False)
    for ax in axs.flat:
        ax.axis("off")
    psc = None
    for a, f in enumerate(sel):
        ax = axs.flat[a]; ax.axis("on")
        psc = ax.scatter(X[f][:, 0], X[f][:, 2], c=scalar[f], s=6, cmap=cmap, vmin=0, vmax=vmax)
        if floor_z is not None:
            ax.axhline(floor_z, color="cyan", ls="-", lw=1)
        ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_aspect("equal")
        tag = "" if rel_start is None else (" [PULL]" if f < rel_start else " [REL]")
        wtxt = "" if width is None else f" w={width[f]:.3f}"
        ax.set_title(f"f{f}{tag}{wtxt}", fontsize=9); ax.set_xlabel("x"); ax.set_ylabel("z")
    if psc is not None:
        fig.colorbar(psc, ax=axs, shrink=0.6, label="stretch |sigma-1|")
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.savefig(path, dpi=110); plt.close(fig)
    return path


def observables_plot(path: str, series: Dict[str, np.ndarray], *,
                     rel_start: Optional[int] = None, floor_z: Optional[float] = None,
                     suptitle: str = "") -> str:
    """Line plot of named per-frame scalars (width / min-z / com-z ...)."""
    markers = ["-o", "-s", "-^", "-v", "-d"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for (name, arr), mk in zip(series.items(), markers):
        ax.plot(arr, mk, ms=3, label=name)
    if floor_z is not None:
        ax.axhline(floor_z, color="cyan", ls="-", lw=1, label=f"floor {floor_z}")
    if rel_start is not None:
        ax.axvline(rel_start - 1, color="orange", ls="--", label=f"release (f{rel_start-1})")
    ax.set_xlabel("frame"); ax.set_ylabel("value"); ax.legend(fontsize=8)
    if suptitle:
        ax.set_title(suptitle)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def scalar_scatter(path: str, X2: np.ndarray, scalar: np.ndarray, *,
                   title: str = "", cbar_label: str = "|sigma-1|", cmap: str = "viridis") -> str:
    """Single-frame xz scatter colored by a per-particle scalar (e.g. the F0 snapshot)."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    psc = ax.scatter(X2[:, 0], X2[:, 2], c=scalar, s=10, cmap=cmap)
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("z")
    if title:
        ax.set_title(title)
    fig.colorbar(psc, ax=ax, label=cbar_label)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def triplane_scalar_gif(path: str, X: np.ndarray, scalar: np.ndarray, *,
                        cmap: str = "inferno", vmax: Optional[float] = None,
                        floor_z: Optional[float] = None, fps: int = 6,
                        title_fn=None) -> str:
    """3D + xy/xz/yz triplane animation of ONE rollout, colored by per-particle scalar."""
    if vmax is None:
        vmax = float(np.quantile(scalar, 0.98)) or 1e-3
    mins, maxs = _bounds(X); T = X.shape[0]
    fig = plt.figure(figsize=(11, 9))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax2 = [fig.add_subplot(2, 2, k) for k in (2, 3, 4)]

    def draw(f):
        ax3d.cla()
        ax3d.scatter(X[f][:, 0], X[f][:, 1], X[f][:, 2], c=scalar[f], s=4, cmap=cmap, vmin=0, vmax=vmax)
        ax3d.set_xlim(mins[0], maxs[0]); ax3d.set_ylim(mins[1], maxs[1]); ax3d.set_zlim(mins[2], maxs[2])
        ax3d.set_title(title_fn(f) if title_fn else f"frame {f}/{T-1}", fontsize=9)
        for axp, (a, b, la, lb) in zip(ax2, PROJ):
            axp.cla()
            axp.scatter(X[f][:, a], X[f][:, b], c=scalar[f], s=5, cmap=cmap, vmin=0, vmax=vmax)
            if floor_z is not None and (a, b) == (0, 2):
                axp.axhline(floor_z, color="cyan", lw=1)
            axp.set_xlim(mins[a], maxs[a]); axp.set_ylim(mins[b], maxs[b]); axp.set_aspect("equal")
            axp.set_xlabel(la); axp.set_ylabel(lb); axp.set_title(f"{la}{lb}")
        return ()

    draw(0)
    FuncAnimation(fig, draw, frames=T, blit=False).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def triplane_overlay_gif(path: str, items: List[Tuple[str, str, np.ndarray]], *,
                         floor_z: Optional[float] = None, fps: int = 3,
                         title: str = "") -> str:
    """3D + triplane animation overlaying several rollouts (each its own color).

    items: list of (label, color, X[T,n,3]). Frame count = first item's length."""
    allp = np.concatenate([X.reshape(-1, 3) for _, _, X in items], 0)
    mn, mx = allp.min(0), allp.max(0); T = items[0][2].shape[0]
    fig = plt.figure(figsize=(11, 9))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax2 = [fig.add_subplot(2, 2, k) for k in (2, 3, 4)]

    def draw(f):
        ax3d.cla()
        for lbl, c, X in items:
            ax3d.scatter(X[f][:, 0], X[f][:, 1], X[f][:, 2], c=c, s=3, alpha=0.4,
                         label=lbl, depthshade=False)
        ax3d.set_xlim(mn[0], mx[0]); ax3d.set_ylim(mn[1], mx[1]); ax3d.set_zlim(mn[2], mx[2])
        ax3d.set_title(f"{title} frame {f}/{T-1}".strip(), fontsize=9)
        ax3d.legend(fontsize=7, loc="upper left")
        for axp, (a, b, la, lb) in zip(ax2, PROJ):
            axp.cla()
            for lbl, c, X in items:
                axp.scatter(X[f][:, a], X[f][:, b], c=c, s=4, alpha=0.4)
            if floor_z is not None and (a, b) == (0, 2):
                axp.axhline(floor_z, color="cyan", lw=1)
            axp.set_xlim(mn[a], mx[a]); axp.set_ylim(mn[b], mx[b]); axp.set_aspect("equal")
            axp.set_xlabel(la); axp.set_ylabel(lb); axp.set_title(f"{la}{lb}")
        return ()

    draw(0)
    FuncAnimation(fig, draw, frames=T, blit=False).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def overlay_panel(path: str, items: List[Tuple[str, str, np.ndarray]], *,
                  floor_z: Optional[float] = None, suptitle: str = "", ncol: int = 3) -> str:
    """Static grid of xz frames overlaying several rollouts (each its own color)."""
    allp = np.concatenate([X.reshape(-1, 3) for _, _, X in items], 0)
    mn, mx = allp.min(0), allp.max(0); T = items[0][2].shape[0]
    sel = list(range(0, T, max(1, T // 8)))[:9]
    nrow = (len(sel) + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow), squeeze=False)
    for ax in axs.flat:
        ax.axis("off")
    for a, f in enumerate(sel):
        ax = axs.flat[a]; ax.axis("on")
        for lbl, c, X in items:
            ax.scatter(X[f][:, 0], X[f][:, 2], c=c, s=5, alpha=0.45, label=lbl if a == 0 else None)
        if floor_z is not None:
            ax.axhline(floor_z, color="cyan", lw=1)
        ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[2], mx[2]); ax.set_aspect("equal")
        ax.set_title(f"frame {f}", fontsize=9)
        if a == 0:
            ax.legend(fontsize=7)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return path
