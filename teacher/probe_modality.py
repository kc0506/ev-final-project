"""Modality-agnostic denoiser probes for an RGB (channels=3) or FLOW (channels=2)
diffusion ckpt. Queries the denoiser/score on controlled inputs (no sampling,
no distillation). Three signals, all using physically-consistent transforms:

  signdist  : denoising residual of each REAL clip vs its true latent v0_x.
              symmetric-about-0 -> both +-x signs learned ; asymmetric -> collapse.
  mirror    : residual of the LEFT-RIGHT-FLIPPED clip (a consistent opposite-sign
              version) vs the real one. ratio ~1 -> +-x symmetry learned.
  treverse  : residual of the TIME-REVERSED clip (anti-damping / anti-physical)
              vs the real one. ratio >>1 -> temporal arrow (damping dir) learned.

Flow transforms also flip the relevant vector channels (hflip negates fx;
time-reverse negates the displacement). RGB transforms are plain flips.
"""
import argparse
import glob
import json
import os
from typing import Tuple

import numpy as np


def load_pack(path: str) -> np.ndarray:
    """Load a clip pack -> (N, F, H, W, C) float32 in [0,1]. RGB packs are uint8."""
    arr = np.load(path)                                  # (N,F,H,W,C)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


def motion_mask(clip: np.ndarray, top_frac: float = 0.15) -> np.ndarray:
    """clip (F,H,W,C) -> (H,W) bool mask of the top-`top_frac` temporally-varying pixels."""
    g = clip.mean(-1)                                    # (F,H,W) content over time
    s = g.std(0)                                         # (H,W) per-pixel temporal std
    thr = np.quantile(s, 1.0 - top_frac)
    return s >= thr                                      # (H,W)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--channels", type=int, required=True, choices=[2, 3])
    ap.add_argument("--data_dir", default="../outputs/dataset_gen/01_tel_axisx_rest_T16")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--dim_mults", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--ts", type=int, nargs="+", default=[300, 500])
    ap.add_argument("--nnoise", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=3)
    args = ap.parse_args()
    import torch
    from torch import Tensor
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from video_diffusion_pytorch import Unet3D, GaussianDiffusion

    dev = "cuda"
    outdir: str = args.out or os.path.join(os.path.dirname(args.ckpt), "..", "diag")
    os.makedirs(outdir, exist_ok=True)

    pack = load_pack(args.pack)                          # (N,F,H,W,C)
    N, F, H, W, C = pack.shape
    assert C == args.channels, f"pack C={C} != --channels {args.channels}"
    samples = sorted(glob.glob(os.path.join(args.data_dir, "sample_*")))
    samples = [s for s in samples if os.path.isfile(os.path.join(s, "sample.json"))][:N]
    v0x = np.array([json.load(open(os.path.join(s, "sample.json")))["v0"][0] for s in samples])  # (N,)
    mask = np.stack([motion_mask(pack[i]) for i in range(N)]).astype(np.float32)  # (N,H,W)

    unet = Unet3D(dim=args.dim, dim_mults=tuple(args.dim_mults), channels=C)
    diff = GaussianDiffusion(unet, image_size=args.res, num_frames=F, channels=C,
                             timesteps=1000, loss_type="l2").to(dev)
    diff.load_state_dict(torch.load(args.ckpt, map_location=dev)["diffusion"])
    diff.eval()

    def to_model(x_bfhwc: np.ndarray) -> Tensor:
        """(B,F,H,W,C) [0,1] -> (B,C,F,H,W) normalized to [-1,1] on device."""
        t = torch.from_numpy(x_bfhwc).float().permute(0, 4, 1, 2, 3).to(dev)
        return t * 2 - 1

    def hflip(x_bfhwc: np.ndarray) -> np.ndarray:
        """left-right mirror; for flow also negate fx (channel 0). (B,F,H,W,C)->same."""
        y = x_bfhwc[:, :, :, ::-1, :].copy()
        if C == 2:
            y[..., 0] = 1.0 - y[..., 0]                  # packed flow: 0.5-centered -> negate fx
        return y

    def treverse(x_bfhwc: np.ndarray) -> np.ndarray:
        """time reversal; for flow also negate displacement. (B,F,H,W,C)->same."""
        y = x_bfhwc[:, ::-1].copy()
        if C == 2:
            y = 1.0 - y                                  # negate both packed flow channels
        return y

    @torch.no_grad()
    def err(x0n: Tensor, m: Tensor) -> np.ndarray:
        """x0n (B,C,F,H,W) in [-1,1]; m (B,H,W) -> per-clip residual on mask, (B,)."""
        out = torch.zeros(x0n.shape[0], device=dev)
        for t in args.ts:
            tt = torch.full((x0n.shape[0],), t, device=dev, dtype=torch.long)
            for _ in range(args.nnoise):
                noise = torch.randn_like(x0n)
                xt = diff.q_sample(x0n, tt, noise)
                pred = diff.denoise_fn(xt, tt, cond=None)
                x0p = diff.predict_start_from_noise(xt, tt, pred)
                e = ((x0p - x0n) ** 2).mean(1)            # (B,F,H,W) over channels
                mm = m[:, None]                           # (B,1,H,W) broadcast over F
                out += (e * mm).sum((1, 2, 3)) / (mm.expand_as(e).sum((1, 2, 3)).clamp(min=1))
        return (out / (len(args.ts) * args.nnoise)).cpu().numpy()

    e_real = np.zeros(N); e_mir = np.zeros(N); e_trev = np.zeros(N)
    for i in range(0, N, args.chunk):
        j = min(N, i + args.chunk)
        m = torch.from_numpy(mask[i:j]).to(dev)
        e_real[i:j] = err(to_model(pack[i:j]), m)
        e_mir[i:j] = err(to_model(hflip(pack[i:j])), torch.flip(m, dims=[-1]))
        e_trev[i:j] = err(to_model(treverse(pack[i:j])), m)

    pos, neg = v0x > 0, v0x < 0
    res = {
        "label": args.label, "channels": C, "n": int(N),
        "signdist": {
            "err_POS": round(float(e_real[pos].mean()), 5),
            "err_NEG": round(float(e_real[neg].mean()), 5),
            "ratio_NEG_over_POS": round(float(e_real[neg].mean() / max(e_real[pos].mean(), 1e-9)), 3),
            "corr_err_vs_v0x": round(float(np.corrcoef(v0x, e_real)[0, 1]), 3),
            "corr_err_vs_absv0x": round(float(np.corrcoef(np.abs(v0x), e_real)[0, 1]), 3)},
        "mirror_ratio_flip_over_real": round(float(e_mir.mean() / max(e_real.mean(), 1e-9)), 3),
        "treverse_ratio_rev_over_real": round(float(e_trev.mean() / max(e_real.mean(), 1e-9)), 3),
        "err_real_mean": round(float(e_real.mean()), 5),
    }
    print(json.dumps(res))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.scatter(v0x, e_real, s=14, alpha=.6)
    bins = np.linspace(v0x.min(), v0x.max(), 11); bc = .5 * (bins[1:] + bins[:-1])
    bm = [e_real[(v0x >= bins[k]) & (v0x < bins[k + 1])].mean()
          if ((v0x >= bins[k]) & (v0x < bins[k + 1])).any() else np.nan for k in range(10)]
    ax.plot(bc, bm, "r-o", lw=2, label="binned mean")
    ax.axvline(0, color="k", ls=":")
    ax.set_xlabel("true v0_x"); ax.set_ylabel("denoising residual (moving region)")
    ax.set_title(f"signdist [{args.label}] — err vs v0_x")
    ax.legend()
    out_png = os.path.join(outdir, f"signdist_{args.label}.png")
    plt.tight_layout(); plt.savefig(out_png, dpi=120); plt.close()
    print("saved", out_png)


if __name__ == "__main__":
    main()
