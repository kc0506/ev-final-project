"""Sample K clips from a trained checkpoint and emit an ANIMATED grid gif so we
can judge (a) whether the lack of motion is systematic or just one unlucky
sample, and (b) sample-to-sample variance. Also prints per-sample motion vs the
training data.

  python sample_grid_gif.py --ckpt out_.../checkpoints/diff_final.pt --k 8 --cols 4
"""
import argparse, os, numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None, help="output dir (default: <ckpt dir>/../diag)")
    ap.add_argument("--k", type=int, default=8, help="number of samples")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=4, help="samples per sampling call (mem)")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--dim_mults", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--pack", default="../outputs/dataset_gen/01_tel_axisx_rest_T16/video_pack_128.npy")
    args = ap.parse_args()
    import torch, imageio.v2 as imageio
    from video_diffusion_pytorch import Unet3D, GaussianDiffusion
    dev = "cuda"
    outdir = args.out or os.path.join(os.path.dirname(args.ckpt), "..", "diag")
    os.makedirs(outdir, exist_ok=True)

    unet = Unet3D(dim=args.dim, dim_mults=tuple(args.dim_mults))
    diff = GaussianDiffusion(unet, image_size=args.res, num_frames=args.frames,
                             timesteps=1000, loss_type="l2").to(dev)
    diff.load_state_dict(torch.load(args.ckpt, map_location=dev)["diffusion"])
    diff.eval()

    vids = []
    with torch.no_grad():
        got = 0
        while got < args.k:
            n = min(args.chunk, args.k - got)
            v = diff.sample(batch_size=n)            # (n,C,T,H,W) [0,1]
            vids.append(v.cpu()); got += n
            print(f"sampled {got}/{args.k}", flush=True)
    vid = torch.cat(vids)[: args.k].clamp(0, 1).numpy()   # (K,C,T,H,W)
    K, C, T, H, W = vid.shape
    vid = vid.transpose(0, 2, 3, 4, 1)                     # (K,T,H,W,3)

    # per-sample motion (frame-to-frame |Δ|, x1000), vs data reference
    f2f = np.abs(np.diff(vid, axis=1)).mean(axis=(2, 3, 4)) * 1000   # (K,T-1)
    full = np.load(args.pack).astype(np.float32) / 255.0
    dref = np.abs(np.diff(full[:, :T], axis=1)).mean(axis=(0, 2, 3, 4)) * 1000
    print("\nper-sample mean f2f |Δ| x1000:", np.round(f2f.mean(1), 2))
    print("ALL samples mean:", round(float(f2f.mean()), 2),
          "| DATA mean:", round(float(dref.mean()), 2))

    # animated grid gif: each gif-frame = grid of the K samples at time t
    cols = args.cols; rows = (K + cols - 1) // cols
    pad = 2
    canvas_frames = []
    for t in range(T):
        cv = np.ones((rows * H + (rows - 1) * pad, cols * W + (cols - 1) * pad, 3))
        for k in range(K):
            r, c = divmod(k, cols)
            cv[r*(H+pad):r*(H+pad)+H, c*(W+pad):c*(W+pad)+W] = vid[k, t]
        canvas_frames.append((cv * 255).astype(np.uint8))
    gpath = os.path.join(outdir, "grid_samples.gif")
    imageio.mimsave(gpath, canvas_frames, fps=7)
    # also a delta grid gif (amplified) to see motion structure across samples
    dframes = []
    k_amp = 6
    for t in range(T - 1):
        cv = np.zeros((rows * H + (rows - 1) * pad, cols * W + (cols - 1) * pad, 3))
        for k in range(K):
            r, c = divmod(k, cols)
            d = np.clip(np.abs(vid[k, t+1] - vid[k, t]).mean(-1)[..., None] * k_amp, 0, 1)
            cv[r*(H+pad):r*(H+pad)+H, c*(W+pad):c*(W+pad)+W] = d
        dframes.append((cv * 255).astype(np.uint8))
    dpath = os.path.join(outdir, "grid_delta.gif")
    imageio.mimsave(dpath, dframes, fps=7)
    print(f"\nwrote {gpath}\nwrote {dpath}")


if __name__ == "__main__":
    main()
