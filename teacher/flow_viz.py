"""CPU-only: visualise screen-space flow of the moving MPM particles.

Projects the moving particles (~freeze_mask) through the render camera each frame
(world_view_transform from camera.json + FoV pinhole), computes per-frame screen
displacement, and overlays it on the rendered video: quiver arrows + motion
tracks. This is the GT for the screen-space-flow teacher POC. No GPU.

  python flow_viz.py --sample ../outputs/dataset_gen/01_tel_axisx_rest_T16/sample_0004 \
      --camera ../outputs/explore/v0_sweep/04_freezefix_check/camera.json \
      --cache ../outputs/dataset_gen/01_tel_axisx_rest_T16/scene_cache.pt
"""
import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def project(X, wvt, fovx, fovy, W, H):
    """X [n,3] world -> (uv [n,2] pixel, z_view [n]). GS convention: x_view =
    [X,1] @ wvt; OpenCV view (+x right,+y down,+z fwd) -> pinhole pixel."""
    Xh = np.concatenate([X, np.ones((len(X), 1))], 1)          # [n,4]
    view = Xh @ wvt                                            # [n,4]
    xv, yv, zv = view[:, 0], view[:, 1], view[:, 2]
    fx = W / (2 * np.tan(fovx / 2)); fy = H / (2 * np.tan(fovy / 2))
    u = fx * xv / zv + W / 2.0
    v = fy * yv / zv + H / 2.0
    return np.stack([u, v], 1), zv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_arrows", type=int, default=300)
    ap.add_argument("--arrow_scale", type=float, default=3.0)
    args = ap.parse_args()
    out = args.out or os.path.join("flow_viz_out", os.path.basename(os.path.dirname(args.sample)) + "_" + os.path.basename(args.sample))
    os.makedirs(out, exist_ok=True)

    cam = json.load(open(args.camera))
    wvt = np.asarray(cam["world_view_transform"], dtype=np.float64)  # [4,4]
    W, H = int(cam["image_width"]), int(cam["image_height"])
    fovx, fovy = float(cam["FoVx"]), float(cam["FoVy"])

    disc = torch.load(args.cache, map_location="cpu")["disc"]
    freeze = disc["freeze_mask"].numpy().astype(bool)               # [n] True=anchor/frozen
    move = ~freeze                                                  # moving (query) particles
    xyz = np.load(os.path.join(args.sample, "mpm_xyz.npy"))          # [T,n,3] world
    vid = np.load(os.path.join(args.sample, "video.npy"))           # [T,H,W,3] uint8
    T = xyz.shape[0]
    Xm = xyz[:, move, :]                                            # [T,nmove,3]

    # project every frame
    uv = np.zeros((T, Xm.shape[1], 2)); zv = np.zeros((T, Xm.shape[1]))
    for t in range(T):
        uv[t], zv[t] = project(Xm[t], wvt, fovx, fovy, W, H)
    inb = (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H) & (zv > 0)
    f0 = inb[0]
    print(f"[sanity] moving particles={Xm.shape[1]}  in-bounds@frame0={f0.mean()*100:.0f}%  "
          f"proj bbox px x[{uv[0,f0,0].min():.0f},{uv[0,f0,0].max():.0f}] "
          f"y[{uv[0,f0,1].min():.0f},{uv[0,f0,1].max():.0f}]  (img {W}x{H})")

    # subsample for arrows (use particles in-bounds the whole clip)
    keep = inb.all(0)
    idx = np.where(keep)[0]
    if len(idx) > args.n_arrows:
        idx = idx[np.linspace(0, len(idx) - 1, args.n_arrows).astype(int)]

    # (1) per-frame quiver overlay gif (arrow = screen velocity to next frame)
    frames = []
    for t in range(T):
        fig, ax = plt.subplots(figsize=(W / 120, H / 120), dpi=120)
        ax.imshow(vid[t]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
        if t < T - 1:
            p = uv[t, idx]; d = (uv[t + 1, idx] - uv[t, idx]) * args.arrow_scale
            # data axes are pixel coords (y-down via ylim(H,0)); with angles/scale_units
            # ='xy' the arrow runs (u,v)->(u+U,v+V) in DATA coords, so pass dv as-is.
            # (the old -dv mirrored the vertical component -> arrows looked reversed.)
            ax.quiver(p[:, 0], p[:, 1], d[:, 0], d[:, 1], color="lime", angles="xy",
                      scale_units="xy", scale=1, width=0.003, headwidth=3)
        ax.set_title(f"frame {t}->{t+1} screen flow (x{args.arrow_scale:g})", fontsize=8, color="r")
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(os.path.join(out, "flow_quiver.gif"), frames, fps=4)

    # (2) motion tracks over the whole clip on frame 0
    fig, ax = plt.subplots(figsize=(W / 120, H / 120), dpi=120)
    ax.imshow(vid[0]); ax.axis("off")
    for j in idx:
        ax.plot(uv[:, j, 0], uv[:, j, 1], "-", lw=0.5, alpha=0.6)
    ax.scatter(uv[0, idx, 0], uv[0, idx, 1], s=4, c="lime")
    ax.set_title("moving-particle screen tracks over clip", fontsize=8, color="r")
    fig.tight_layout(pad=0); fig.savefig(os.path.join(out, "flow_tracks.png"), dpi=120); plt.close(fig)

    np.save(os.path.join(out, "screen_uv.npy"), uv)  # [T,nmove,2] raw screen positions
    print(f"saved -> {out}/  (flow_quiver.gif, flow_tracks.png, screen_uv.npy)")


if __name__ == "__main__":
    main()
