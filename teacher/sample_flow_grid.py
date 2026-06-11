"""Sample N clips from a FLOW checkpoint, write an ANIMATED grid gif (all N
samples moving together -- not a static PNG, not a single sample) and report
the two smoke red-flags on real data: localisation (%moving px, want ~train
11.5%, not smeared) and direction coverage (signed flow_x: does it produce BOTH
v0 signs or collapse to one?).

  python sample_flow_grid.py --ckpt out_01_flow_T8_local/checkpoints/diff_final.pt --n 24 --cols 6
"""
import argparse, json, os, numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=6)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--frames", type=int, default=7)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--dim_mults", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--pack", default="../outputs/dataset_gen/01_tel_axisx_rest_T16/flow_pack_128_t8.npy")
    args = ap.parse_args()
    import torch, imageio.v2 as imageio
    from video_diffusion_pytorch import Unet3D, GaussianDiffusion
    outdir = args.out or os.path.join(os.path.dirname(args.ckpt), "..", "diag")
    os.makedirs(outdir, exist_ok=True)
    scale = json.load(open(args.pack + ".meta.json")).get("scale_px", 1.0)
    dec = lambda x: (x - 0.5) * 2 * scale

    unet = Unet3D(dim=args.dim, dim_mults=tuple(args.dim_mults), channels=2)
    diff = GaussianDiffusion(unet, image_size=args.res, num_frames=args.frames, channels=2,
                             timesteps=1000, loss_type="l2").cuda()
    diff.load_state_dict(torch.load(args.ckpt, map_location="cuda")["diffusion"])
    diff.eval()
    outs = []
    with torch.no_grad():
        got = 0
        while got < args.n:
            k = min(args.chunk, args.n - got)
            outs.append(diff.sample(batch_size=k).cpu().numpy()); got += k
            print(f"sampled {got}/{args.n}", flush=True)
    vid = np.concatenate(outs)[:args.n].transpose(0, 2, 3, 4, 1)   # (N,F,H,W,2) in [0,1]
    N, F, H, W, _ = vid.shape
    cols = args.cols; rows = (N + cols - 1) // cols; pad = 2

    def to_rgb(f2):                                                # (H,W,2)[0,1]->(H,W,3)
        return np.concatenate([f2, np.full((H, W, 1), 0.5, np.float32)], -1)

    frames = []
    for t in range(F):
        cv = np.ones((rows * H + (rows - 1) * pad, cols * W + (cols - 1) * pad, 3))
        for k in range(N):
            r, c = divmod(k, cols)
            cv[r*(H+pad):r*(H+pad)+H, c*(W+pad):c*(W+pad)+W] = to_rgb(vid[k, t])
        frames.append((cv * 255).astype(np.uint8))
    gpath = os.path.join(outdir, "flow_grid_samples.gif")
    imageio.mimsave(gpath, frames, fps=5)

    # stats on decoded px flow
    disp = dec(vid)                                                # (N,F,H,W,2) px
    mag = np.abs(disp).sum(-1)                                     # (N,F,H,W)
    pct_moving = (mag > 1.0).mean(axis=(1, 2, 3)) * 100            # per sample
    fx = np.array([disp[i, ..., 0][mag[i] > 1.0].mean() if (mag[i] > 1.0).any() else 0.0
                   for i in range(N)])
    train = np.load(args.pack)
    tmag = np.abs(dec(train)).sum(-1)
    tpct = (tmag > 1.0).mean() * 100
    print(json.dumps({
        "ckpt": args.ckpt, "n": N,
        "pct_moving_mean": round(float(pct_moving.mean()), 1),
        "pct_moving_train_ref": round(float(tpct), 1),
        "pct_moving_per_sample": [round(float(x), 1) for x in pct_moving],
        "signed_flow_x_per_sample": [round(float(x), 2) for x in fx],
        "n_pos_dir": int((fx > 0.2).sum()), "n_neg_dir": int((fx < -0.2).sum()),
        "gif": gpath}))


if __name__ == "__main__":
    main()
