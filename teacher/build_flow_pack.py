"""Build a dense screen-space-FLOW pack from the MPM sim, the GT-motion teacher
target (no RAFT). For each sample: project moving particles through the render
camera each frame (reusing flow_viz's pinhole math), take per-frame screen
displacement, rasterise the sparse per-particle vectors into a dense [H,W,2]
field (splat to pixel + fill pinholes), and pack the first `--frames` window so
flow gap t lives between RGB frames t and t+1 (t0 preserved, like the RGB pack).

Flow is signed and small; we map it to [0,1] with a single global scale (stored
in the .meta.json) so it feeds GaussianDiffusion exactly like an RGB image
(channels=2). Invert at sample time: disp_px = (x-0.5)*2*scale.

  python build_flow_pack.py \
    --data_dir ../outputs/dataset_gen/01_tel_axisx_rest_T16 \
    --camera   ../outputs/dataset_gen/01_tel_axisx_rest_T16/camera.json \
    --frames 8 --res 128 --out ../outputs/dataset_gen/01_tel_axisx_rest_T16/flow_pack_128_t8.npy
"""
import argparse, glob, json, os
import numpy as np


def project(X, wvt, fovx, fovy, W, H):
    """world [n,3] -> (uv pixel [n,2], z_view [n]); GS/OpenCV pinhole (from flow_viz.py)."""
    Xh = np.concatenate([X, np.ones((len(X), 1))], 1)
    view = Xh @ wvt
    xv, yv, zv = view[:, 0], view[:, 1], view[:, 2]
    fx = W / (2 * np.tan(fovx / 2)); fy = H / (2 * np.tan(fovy / 2))
    u = fx * xv / zv + W / 2.0
    v = fy * yv / zv + H / 2.0
    return np.stack([u, v], 1), zv


def fill_holes(flow, mask, iters=8):
    """Fill unset pixels (inside/around the splat) with the mean of set 4/8-neighbours,
    a few iterations. Keeps background (far from any particle) at zero."""
    H, W, _ = flow.shape
    f = flow.copy(); m = mask.copy()
    for _ in range(iters):
        if m.all():
            break
        # 8-neighbour sum of values and of mask
        s = np.zeros_like(f); c = np.zeros((H, W))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ys, xs = slice(max(0, dy), H + min(0, dy)), slice(max(0, dx), W + min(0, dx))
                yd, xd = slice(max(0, -dy), H + min(0, -dy)), slice(max(0, -dx), W + min(0, -dx))
                s[yd, xd] += f[ys, xs] * m[ys, xs, None]
                c[yd, xd] += m[ys, xs]
        new = (~m) & (c > 0)
        f[new] = s[new] / c[new, None]
        m[new] = True
    return f, m


def rasterise(pos, disp, valid, res):
    """Scatter per-particle screen displacement to a dense [res,res,2] field, then
    fill pinholes. pos/uv are in pixel coords already scaled to `res`."""
    acc = np.zeros((res, res, 2)); cnt = np.zeros((res, res))
    p = pos[valid]; d = disp[valid]
    col = np.clip(np.round(p[:, 0]).astype(int), 0, res - 1)
    row = np.clip(np.round(p[:, 1]).astype(int), 0, res - 1)
    np.add.at(acc, (row, col), d)
    np.add.at(cnt, (row, col), 1.0)
    m = cnt > 0
    flow = np.zeros_like(acc); flow[m] = acc[m] / cnt[m, None]
    flow, _ = fill_holes(flow, m)
    return flow  # [res,res,2] in pixel-displacement units (at `res` scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--cache", default=None, help="scene_cache.pt (default: <data_dir>/scene_cache.pt)")
    ap.add_argument("--frames", type=int, default=8, help="RGB frames -> frames-1 flow fields")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pct", type=float, default=99.5, help="percentile of |disp| for the [0,1] scale")
    ap.add_argument("--scale", type=float, default=None,
                    help="force scale_px instead of computing p<pct> (use to MATCH another pack's encoding)")
    args = ap.parse_args()
    import torch

    cam = json.load(open(args.camera))
    wvt = np.asarray(cam["world_view_transform"], dtype=np.float64)
    W, H = int(cam["image_width"]), int(cam["image_height"])
    fovx, fovy = float(cam["FoVx"]), float(cam["FoVy"])
    sx, sy = args.res / W, args.res / H   # scale projected px -> res grid

    cache = args.cache or os.path.join(args.data_dir, "scene_cache.pt")
    freeze = torch.load(cache, map_location="cpu", weights_only=False)["disc"]["freeze_mask"].numpy().astype(bool)
    move = ~freeze
    samples = sorted(glob.glob(os.path.join(args.data_dir, "sample_*")))
    samples = [s for s in samples if os.path.isfile(os.path.join(s, "mpm_xyz.npy"))]
    F = args.frames
    print(f"{len(samples)} samples | moving particles={move.sum()} | cam {W}x{H} -> res {args.res} | {F-1} flow fields")

    raw = np.zeros((len(samples), F - 1, args.res, args.res, 2), dtype=np.float32)
    for si, s in enumerate(samples):
        xyz = np.load(os.path.join(s, "mpm_xyz.npy"))[:F]      # [F,n,3]
        Xm = xyz[:, move, :]
        uv = np.zeros((F, Xm.shape[1], 2)); zv = np.zeros((F, Xm.shape[1]))
        for t in range(F):
            uv[t], zv[t] = project(Xm[t], wvt, fovx, fovy, W, H)
        uv[..., 0] *= sx; uv[..., 1] *= sy                      # -> res grid
        for t in range(F - 1):
            inb = ((uv[t, :, 0] >= 0) & (uv[t, :, 0] < args.res) &
                   (uv[t, :, 1] >= 0) & (uv[t, :, 1] < args.res) & (zv[t] > 0) & (zv[t + 1] > 0))
            disp = uv[t + 1] - uv[t]                            # [n,2] px displacement at res scale
            raw[si, t] = rasterise(uv[t], disp, inb, args.res)
        if si % 32 == 0:
            print(f"  {si}/{len(samples)}", flush=True)

    # global scale -> map signed flow to [0,1]; invert later: disp=(x-0.5)*2*scale
    scale = float(args.scale) if args.scale else float(np.percentile(np.abs(raw), args.pct))
    scale = max(scale, 1e-6)
    packed = np.clip(raw / (2 * scale) + 0.5, 0.0, 1.0).astype(np.float32)
    np.save(args.out, packed)
    meta = {"kind": "screen_flow", "n": len(samples), "frames_rgb": F, "flow_fields": F - 1,
            "res": args.res, "scale_px": scale, "scale_pct": args.pct,
            "encode": "x in [0,1]; disp_px = (x-0.5)*2*scale_px", "data_dir": os.path.abspath(args.data_dir)}
    json.dump(meta, open(args.out + ".meta.json", "w"), indent=2)
    print(f"\nsaved {args.out}  shape={packed.shape}  scale_px(p{args.pct})={scale:.3f}")
    print(f"raw |disp| px: mean={np.abs(raw).mean():.3f} max={np.abs(raw).max():.3f}")


if __name__ == "__main__":
    main()
