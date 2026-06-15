"""Sample N clips from a FLOW checkpoint and write, PER SAMPLE, both an animated gif
(the 7 flow fields moving) and a static frame-grid png (the 7 fields laid out in a row,
for pausing on detail). Also reports the two red-flags on the learned distribution:
localisation (%moving px vs train) and signed flow_x direction coverage (does it produce
BOTH +-x signs, i.e. did it learn the +-[2,8] band?).

  python sample_flow_each.py --ckpt out_02_flow_aligned_mag2-8/checkpoints/diff_final.pt \
      --pack ../outputs/gen_flow_aligned/02_n128_axisx_mag2-8_rot67.6/flow_pack_128_t8.npy --n 8
"""
import argparse
import json
import os
from typing import List

import numpy as np


def to_rgb(f2: np.ndarray) -> np.ndarray:
    """flow field [H,W,2] in [0,1] -> [H,W,3] viz (B=0.5 grey background)."""
    return np.concatenate([f2, np.full(f2.shape[:2] + (1,), 0.5, np.float32)], -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pack", required=True, help="for scale_px (decode) + train ref stats")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--frames", type=int, default=7)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--dim_mults", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import torch
    import imageio.v2 as imageio
    from PIL import Image
    from video_diffusion_pytorch import Unet3D, GaussianDiffusion

    outdir = args.out or os.path.join(os.path.dirname(args.ckpt), "..", "samples_each")
    os.makedirs(outdir, exist_ok=True)
    scale = float(json.load(open(args.pack + ".meta.json")).get("scale_px", 1.0))

    torch.manual_seed(args.seed)
    unet = Unet3D(dim=args.dim, dim_mults=tuple(args.dim_mults), channels=2)
    diff = GaussianDiffusion(unet, image_size=args.res, num_frames=args.frames, channels=2,
                             timesteps=1000, loss_type="l2").cuda()
    diff.load_state_dict(torch.load(args.ckpt, map_location="cuda")["diffusion"])
    diff.eval()

    outs: List[np.ndarray] = []
    with torch.no_grad():
        got = 0
        while got < args.n:
            k = min(args.chunk, args.n - got)
            outs.append(diff.sample(batch_size=k).cpu().numpy())          # (k,2,F,H,W) in [0,1]
            got += k
            print(f"sampled {got}/{args.n}", flush=True)
    vid = np.concatenate(outs)[:args.n].transpose(0, 2, 3, 4, 1)          # (N,F,H,W,2) in [0,1]
    N, F, H, W, _ = vid.shape

    for i in range(N):
        # animated gif (F frames)
        frames = [(to_rgb(vid[i, t]) * 255).round().astype("uint8") for t in range(F)]
        imageio.mimsave(os.path.join(outdir, f"sample_{i:02d}.gif"), frames, fps=3)
        # static frame grid (F in a row, white separators)
        bar = np.ones((H, 2, 3), np.float32)
        row = []
        for t in range(F):
            row.append(to_rgb(vid[i, t]))
            if t < F - 1:
                row.append(bar)
        grid = (np.concatenate(row, 1) * 255).round().astype("uint8")
        Image.fromarray(grid).save(os.path.join(outdir, f"sample_{i:02d}_grid.png"))

    # distribution red-flags (decoded to pixel flow)
    dec = lambda x: (x - 0.5) * 2 * scale
    disp = dec(vid)                                                       # (N,F,H,W,2) px
    mag = np.abs(disp).sum(-1)                                            # (N,F,H,W)
    pct_moving = (mag > 1.0).mean(axis=(1, 2, 3)) * 100                   # per sample
    fx = np.array([disp[i, ..., 0][mag[i] > 1.0].mean() if (mag[i] > 1.0).any() else 0.0
                   for i in range(N)])                                    # signed mean flow_x
    train = np.load(args.pack)
    tpct = (np.abs(dec(train)).sum(-1) > 1.0).mean() * 100
    print(json.dumps({
        "ckpt": args.ckpt, "n": int(N), "outdir": outdir,
        "pct_moving_mean": round(float(pct_moving.mean()), 1),
        "pct_moving_train_ref": round(float(tpct), 1),
        "signed_flow_x_per_sample": [round(float(x), 2) for x in fx],
        "n_pos_dir": int((fx > 0.2).sum()), "n_neg_dir": int((fx < -0.2).sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
