"""Clean test (no rotation, real clips only): does the flow ckpt find ALL real
in-distribution clips equally easy to denoise, across the v0_x range and BOTH
signs? Plots denoising residual vs each clip's true v0_x. Flat -> learned the
full +-x distribution. Low on one sign / high on the other -> sign-collapsed
(matches the all-one-sign sampling result).
"""
import argparse, glob, json, os, numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="../outputs/dataset_gen/01_tel_axisx_rest_T16")
    ap.add_argument("--pack", default="../outputs/dataset_gen/01_tel_axisx_rest_T16/flow_pack_128_t8.npy")
    ap.add_argument("--out", default=None)
    ap.add_argument("--res", type=int, default=128); ap.add_argument("--frames", type=int, default=7)
    ap.add_argument("--dim", type=int, default=64); ap.add_argument("--dim_mults", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--ts", type=int, nargs="+", default=[300, 500]); ap.add_argument("--nnoise", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=16)
    args = ap.parse_args()
    import torch
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from video_diffusion_pytorch import Unet3D, GaussianDiffusion
    dev = "cuda"
    outdir = args.out or os.path.join(os.path.dirname(args.ckpt), "..", "diag")
    os.makedirs(outdir, exist_ok=True)
    scale = json.load(open(args.pack + ".meta.json"))["scale_px"]

    unet = Unet3D(dim=args.dim, dim_mults=tuple(args.dim_mults), channels=2)
    diff = GaussianDiffusion(unet, image_size=args.res, num_frames=args.frames, channels=2,
                             timesteps=1000, loss_type="l2").to(dev)
    diff.load_state_dict(torch.load(args.ckpt, map_location=dev)["diffusion"]); diff.eval()

    pack = np.load(args.pack)                                  # (N,F,H,W,2) [0,1]
    N = pack.shape[0]
    samples = sorted(glob.glob(os.path.join(args.data_dir, "sample_*")))
    samples = [s for s in samples if os.path.isfile(os.path.join(s, "mpm_xyz.npy"))][:N]
    v0x = np.array([json.load(open(os.path.join(s, "sample.json")))["v0"][0] for s in samples])

    disp = (pack - 0.5) * 2 * scale
    mask = (np.abs(disp).sum(-1) > 0.5).astype(np.float32)     # (N,F,H,W)
    x0n_all = torch.from_numpy(pack).float().permute(0, 4, 1, 2, 3) * 2 - 1   # (N,2,F,H,W) [-1,1]

    @torch.no_grad()
    def err_chunk(x0n, m):
        out = torch.zeros(x0n.shape[0], device=dev)
        for t in args.ts:
            tt = torch.full((x0n.shape[0],), t, device=dev, dtype=torch.long)
            for _ in range(args.nnoise):
                noise = torch.randn_like(x0n)
                xt = diff.q_sample(x0n, tt, noise)
                pred = diff.denoise_fn(xt, tt, cond=None)
                x0p = diff.predict_start_from_noise(xt, tt, pred)
                e = ((x0p - x0n) ** 2).mean(1)                 # (B,F,H,W)
                out += (e * m).sum((1, 2, 3)) / m.sum((1, 2, 3)).clamp(min=1)
        return (out / (len(args.ts) * args.nnoise)).cpu().numpy()

    errs = np.zeros(N)
    for i in range(0, N, args.chunk):
        j = min(N, i + args.chunk)
        x0n = x0n_all[i:j].to(dev); m = torch.from_numpy(mask[i:j]).to(dev)
        errs[i:j] = err_chunk(x0n, m)

    pos, neg = v0x > 0, v0x < 0
    # correlation of err with |v0x| (magnitude effect) and with sign
    print(json.dumps({
        "n": int(N),
        "err_mean_v0x_POS": round(float(errs[pos].mean()), 5),
        "err_mean_v0x_NEG": round(float(errs[neg].mean()), 5),
        "ratio_NEG_over_POS": round(float(errs[neg].mean() / max(errs[pos].mean(), 1e-9)), 3),
        "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
        "corr_err_vs_v0x": round(float(np.corrcoef(v0x, errs)[0, 1]), 3),
        "corr_err_vs_absv0x": round(float(np.corrcoef(np.abs(v0x), errs)[0, 1]), 3)}))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.scatter(v0x, errs, s=14, alpha=.6)
    # binned means
    bins = np.linspace(v0x.min(), v0x.max(), 11); bc = .5 * (bins[1:] + bins[:-1])
    bm = [errs[(v0x >= bins[k]) & (v0x < bins[k+1])].mean() if ((v0x >= bins[k]) & (v0x < bins[k+1])).any() else np.nan for k in range(10)]
    ax.plot(bc, bm, "r-o", lw=2, label="binned mean")
    ax.axvline(0, color="k", ls=":")
    ax.set_xlabel("true v0_x (the latent)"); ax.set_ylabel("denoising residual (in-dist clip)")
    ax.set_title("sign/magnitude distribution probe: err vs v0_x  (flat=learned full ±x dist)")
    ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "signdist_probe.png"), dpi=120); plt.close()
    print("saved", os.path.join(outdir, "signdist_probe.png"))


if __name__ == "__main__":
    main()
