"""Sample clips from a trained video-diffusion checkpoint (no training).

Reads a train run's config.json (for the exact arch) + a checkpoint, samples N
clips, writes a grid png + gifs. Used to inspect a model whose training exited via
the quota guard before its scheduled sample.

  python sample_ckpt.py --out out_01_tel_axisx_rest_T16 --n 6
  python sample_ckpt.py --out out_01_... --ckpt checkpoints/diff_e0050.pt --n 6
"""
import argparse
import json
import os

import gpu_guard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="a train run dir (has config.json + checkpoints/)")
    ap.add_argument("--ckpt", default="checkpoints/latest.pt", help="ckpt rel to --out")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--tag", default="latest")
    args = ap.parse_args()
    gpu_guard.pick_free_gpu(min_quota_hours=0)  # quick job; don't abort on low quota

    import numpy as np
    import torch
    import imageio.v2 as imageio
    from PIL import Image
    from video_diffusion_pytorch import Unet3D, GaussianDiffusion

    cfg = json.load(open(os.path.join(args.out, "config.json")))
    unet = Unet3D(dim=cfg["dim"], dim_mults=tuple(cfg["dim_mults"]))
    diff = GaussianDiffusion(unet, image_size=cfg["res"], num_frames=cfg["frames"],
                             timesteps=cfg["timesteps"], loss_type=cfg["loss_type"]).cuda()
    ck = torch.load(os.path.join(args.out, args.ckpt), map_location="cuda")
    diff.load_state_dict(ck["diffusion"])
    diff.eval()
    print(f"loaded {args.ckpt} (epoch {ck.get('epoch')}) ; sampling {args.n} clips...")

    with torch.no_grad():
        vid = diff.sample(batch_size=args.n)  # (n,C,T,H,W) in [0,1]
    v = (vid.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    n, C, T, H, W = v.shape
    sdir = os.path.join(args.out, "samples")
    os.makedirs(sdir, exist_ok=True)
    # grid: rows=clips, cols=frames
    canvas = np.zeros((n * H, T * W, 3), np.uint8)
    for i in range(n):
        for t in range(T):
            canvas[i*H:(i+1)*H, t*W:(t+1)*W] = v[i, :, t].transpose(1, 2, 0)
    grid = os.path.join(sdir, f"sample_{args.tag}_grid.png")
    Image.fromarray(canvas).save(grid)
    for i in range(min(n, 3)):
        imageio.mimsave(os.path.join(sdir, f"sample_{args.tag}_c{i}.gif"),
                        [v[i, :, t].transpose(1, 2, 0) for t in range(T)], fps=7)
    print(f"saved {grid} (+ {min(n,3)} gifs)")


if __name__ == "__main__":
    main()
