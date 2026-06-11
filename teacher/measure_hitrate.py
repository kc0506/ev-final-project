"""Sample N clips from an RGB video-diffusion checkpoint and report the
'dynamic hit-rate': fraction of samples whose mean frame-to-frame |Δ| (x1000)
>= --thresh. Quantifies how OFTEN the model samples the moving mode (vs the
near-static collapse), so we can compare a checkpoint before/after more epochs.

  python measure_hitrate.py --ckpt out_.../checkpoints/diff_final.pt --n 24 --label RGB200
"""
import argparse, json, numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--thresh", type=float, default=5.0, help="mean f2f |Δ|*1000 to count as 'dynamic'")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--dim_mults", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    import torch
    from video_diffusion_pytorch import Unet3D, GaussianDiffusion
    unet = Unet3D(dim=args.dim, dim_mults=tuple(args.dim_mults))
    diff = GaussianDiffusion(unet, image_size=args.res, num_frames=args.frames,
                             timesteps=1000, loss_type="l2").cuda()
    diff.load_state_dict(torch.load(args.ckpt, map_location="cuda")["diffusion"])
    diff.eval()
    vids = []
    with torch.no_grad():
        got = 0
        while got < args.n:
            k = min(args.chunk, args.n - got)
            vids.append(diff.sample(batch_size=k).cpu().numpy()); got += k
    vid = np.concatenate(vids)[:args.n].transpose(0, 2, 3, 4, 1)   # (N,T,H,W,3)
    f2f = np.abs(np.diff(vid, axis=1)).mean(axis=(2, 3, 4)) * 1000  # (N,T-1)
    per = f2f.mean(1)
    hit = (per >= args.thresh).mean() * 100
    print(json.dumps({"label": args.label, "ckpt": args.ckpt, "n": args.n,
                      "thresh": args.thresh, "hit_rate_pct": round(float(hit), 1),
                      "per_sample_f2f": [round(float(x), 2) for x in per],
                      "mean_f2f": round(float(per.mean()), 2)}))


if __name__ == "__main__":
    main()
