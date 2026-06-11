"""One-shot diagnostic: is the MPM-side motion the same as the 3DGS render?

Renders the MPM particles with an INDEPENDENT pipeline (matplotlib 3D scatter,
freeze=red / moving=green) alongside the 3DGS render, and quantifies whether the
particles/gaussians that should be anchored actually move:

  - freeze MPM particles: count + |disp|-over-time distribution (frame x bucket heatmap)
  - "all-top_k-freeze" gaussians (KNN driven entirely by freeze particles, i.e. the
    red ones in freeze_red.gif): count + |disp|-over-time distribution

Self-contained single run so the scatter / render / histograms all share ONE
discretisation (the archived v_002 only saved mpm_ply, not the gaussian top_k, and
its cache was overwritten -- so its gaussian disp can't be recomputed).

  python -m reuse_mpm.explore.mpm_motion_check
"""
from __future__ import annotations

import os

import numpy as np


def run(scene_path: str = "/tmp2/b10401006/PhysDreamer/data/physics_dreamer/telephone",
        v0_vec=(0.0, -2.0, 0.0), E: float = 1e5,
        out: str = "outputs/_dbg_mpm_motion_v2") -> str:
    import imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d proj)

    from ..gpu import pick_free_gpu
    pick_free_gpu()
    from ..config import SceneSpec, SimConfig
    from ..scene_io import load_from_spec
    from ..sim_render import (make_constant_v0, simulate_positions, render_positions,
                              video_to_uint8)

    os.makedirs(out, exist_ok=True)
    cfg = SimConfig(num_frames=14, substep=64, grid_size=32)
    s = load_from_spec(SceneSpec(preset=None, path=scene_path, kind="pd",
                                 device="cuda:0"), cfg)
    cam = s.camera_by_frame("frame_00001.png")
    import torch
    v0 = make_constant_v0(s, v0_vec)
    pos = simulate_positions(s, float(E), v0, cfg)              # T x [n,3] world
    P = torch.stack(pos, 0).cpu().numpy()                      # [T, n, 3]
    T, n, _ = P.shape
    fm = s.freeze_mask.cpu().numpy().astype(bool)              # [n]
    qm = ~fm
    tk = s.top_k_index.cpu().numpy()                           # [n_sim_gauss, k] -> particle idx

    dvec = P - P[0:1]                                          # [T, n, 3]
    disp = np.linalg.norm(dvec, axis=2)                        # [T, n] particle |disp|
    redg = fm[tk].all(axis=1)                                  # [n_sim_gauss] all-knn-freeze
    gdisp = np.linalg.norm(dvec[:, tk, :].mean(axis=2), axis=2)  # [T, n_sim_gauss] gauss |disp|

    n_freeze = int(fm.sum()); n_redg = int(redg.sum())
    print(f"freeze MPM particles: {n_freeze} / {n}   "
          f"all-top_k-freeze gaussians: {n_redg} / {tk.shape[0]} sim")
    print(" frame |  freeze-particle mean|disp | red-gaussian mean|disp | query mean")
    for t in range(T):
        print(f"  {t:3d}  |  {disp[t, fm].mean():.4f}  |  "
              f"{gdisp[t, redg].mean() if n_redg else float('nan'):.4f}  |  {disp[t, qm].mean():.4f}")

    # ---- matplotlib 3D scatter per frame (freeze=red, moving=green) -> gif ----
    lims = [(P[..., k].min(), P[..., k].max()) for k in range(3)]
    png_paths = []
    for t in range(T):
        fig = plt.figure(figsize=(5, 6)); ax = fig.add_subplot(111, projection="3d")
        ax.scatter(P[t, qm, 0], P[t, qm, 1], P[t, qm, 2], s=1, c="green", alpha=0.25)
        ax.scatter(P[t, fm, 0], P[t, fm, 1], P[t, fm, 2], s=6, c="red", alpha=0.9)
        ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
        ax.set_xlabel("world x"); ax.set_ylabel("world y"); ax.set_zlabel("world z")
        ax.set_title(f"MPM particles  frame {t}\nred=freeze  green=moving")
        ax.view_init(elev=12, azim=-60)
        p = os.path.join(out, f"mpm_f{t:02d}.png"); fig.savefig(p, dpi=90); plt.close(fig)
        png_paths.append(p)
    imageio.mimsave(os.path.join(out, "mpm_scatter.gif"),
                    [imageio.imread(p) for p in png_paths], fps=7, loop=0)

    # ---- 3DGS render for side-by-side ----
    imageio.mimsave(os.path.join(out, "render.gif"),
                    list(video_to_uint8(render_positions(s, pos, cam))), fps=7, loop=0)

    # ---- |disp|-over-time distribution heatmaps (frame x bucket) ----
    def heatmap(data: np.ndarray, title: str, path: str) -> None:
        vmax = max(float(data.max()), 1e-6)
        bins = np.linspace(0.0, vmax * 1.01, 31)
        Hc = np.stack([np.histogram(data[t], bins=bins)[0] for t in range(T)], 0)  # [T,30]
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(Hc.T, origin="lower", aspect="auto",
                       extent=[0, T - 1, 0.0, bins[-1]], cmap="viridis")
        ax.set_xlabel("frame t"); ax.set_ylabel("|disp| (world units)"); ax.set_title(title)
        fig.colorbar(im, label="particle/gaussian count"); fig.tight_layout()
        fig.savefig(path, dpi=110); plt.close(fig)

    heatmap(disp[:, fm], f"FREEZE MPM particle |disp| over t (n={n_freeze})",
            os.path.join(out, "hist_freeze_particles.png"))
    if n_redg:
        heatmap(gdisp[:, redg], f"all-top_k-freeze GAUSSIAN |disp| over t (n={n_redg})",
                os.path.join(out, "hist_freeze_gaussians.png"))

    # ---- clean summary line plot: mean |disp| vs t ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(T), [disp[t, fm].mean() for t in range(T)], "o-", c="red", label=f"freeze particles ({n_freeze})")
    if n_redg:
        ax.plot(range(T), [gdisp[t, redg].mean() for t in range(T)], "s-", c="darkorange", label=f"all-knn-freeze gaussians ({n_redg})")
    ax.plot(range(T), [disp[t, qm].mean() for t in range(T)], "^-", c="green", label="moving particles")
    ax.set_xlabel("frame t"); ax.set_ylabel("mean |disp| (world)")
    ax.set_title("mean displacement vs t (anchored things SHOULD stay ~0)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(out, "mean_disp_vs_t.png"), dpi=110); plt.close(fig)

    print(f"[mpm_motion_check] -> {out}")
    return out


if __name__ == "__main__":
    run()
