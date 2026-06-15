"""Directly probe what a FLOW diffusion ckpt learned about velocity DIRECTION --
without sampling or distillation. A diffusion model is a score/denoiser field;
we query it on controlled inputs.

Construction: take real flow clips (motion ~ +-x) and ROTATE the velocity
vectors by angle phi (phi=0 -> original +-x ; phi=90deg -> same localisation but
pointing +y, which is OFF-manifold). Noise at a few sigma, run ONE denoiser
step, and measure the denoising residual on the moving region as a function of
phi. If the ckpt learned "velocity lives on the +-x axis":
  err(phi) has VALLEYS at phi=0,180 (x) and PEAKS at phi=90,270 (y).
Anisotropy A = err(90)/err(0) > 1 quantifies it. We also feed a +y clip and
read the x-energy-fraction of the denoiser OUTPUT -- if the model rotates it
back toward x, that fraction rises from ~0 toward 1.

  python probe_direction.py --ckpt out_01_flow_T8_local/checkpoints/diff_final.pt
"""
import argparse, json, os, numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pack", default="../outputs/dataset_gen/01_tel_axisx_rest_T16/flow_pack_128_t8.npy")
    ap.add_argument("--out", default=None)
    ap.add_argument("--k", type=int, default=6, help="real clips used as templates")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--frames", type=int, default=7)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--dim_mults", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--ts", type=int, nargs="+", default=[100, 300, 500, 700])
    ap.add_argument("--nnoise", type=int, default=4)
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
    diff.load_state_dict(torch.load(args.ckpt, map_location=dev)["diffusion"])
    diff.eval()

    # pick K highest-motion real clips as templates
    pack = np.load(args.pack)                                   # (N,F,H,W,2) [0,1]
    disp0 = (pack - 0.5) * 2 * scale                            # px
    mot = np.abs(disp0).sum(-1).mean(axis=(1, 2, 3))
    idx = np.argsort(-mot)[:args.k]
    base = disp0[idx]                                           # (K,F,H,W,2) px
    mask = (np.abs(base).sum(-1) > 0.5)                         # (K,F,H,W) moving region
    base_t = torch.from_numpy(base).float().to(dev)
    mask_t = torch.from_numpy(mask).float().to(dev)

    def rotate(d, phi):                                         # d (K,F,H,W,2) px -> rotated vectors
        c, s = np.cos(phi), np.sin(phi)
        fx, fy = d[..., 0], d[..., 1]
        return torch.stack([c * fx - s * fy, s * fx + c * fy], -1)

    def pack01(d):                                             # px -> [0,1] -> (K,2,F,H,W) normalized [-1,1]
        x01 = torch.clamp(d / (2 * scale) + 0.5, 0, 1).permute(0, 4, 1, 2, 3)
        return x01 * 2 - 1

    @torch.no_grad()
    def denoise_err(x0n, t):
        # average denoising residual on the moving region over nnoise draws
        errs = []
        for _ in range(args.nnoise):
            noise = torch.randn_like(x0n)
            xt = diff.q_sample(x0n, t, noise)
            pred = diff.denoise_fn(xt, t, cond=None)
            x0p = diff.predict_start_from_noise(xt, t, pred)
            e = ((x0p - x0n) ** 2).mean(1)                      # (K,F,H,W) over channels
            m = mask_t
            errs.append((e * m).sum((1, 2, 3)) / m.sum((1, 2, 3)).clamp(min=1))
        return torch.stack(errs).mean(0)                        # (K,)

    phis = np.arange(0, 360, 30)
    K = len(idx)
    curve = {t: [] for t in args.ts}
    for phi in phis:
        x0n = pack01(rotate(base_t, np.deg2rad(phi)))
        for t in args.ts:
            tt = torch.full((K,), t, device=dev, dtype=torch.long)
            curve[t].append(denoise_err(x0n, tt).mean().item())
    # anisotropy A = err(90)/err(0) per t
    i0, i90 = list(phis).index(0), list(phis).index(90)
    aniso = {t: curve[t][i90] / max(curve[t][i0], 1e-9) for t in args.ts}

    # rotate-back: feed +y (phi=90), read x-energy-fraction of denoiser output
    @torch.no_grad()
    def xfrac_after_denoise(phi, t):
        x0n = pack01(rotate(base_t, np.deg2rad(phi)))
        tt = torch.full((K,), t, device=dev, dtype=torch.long)
        noise = torch.randn_like(x0n)
        xt = diff.q_sample(x0n, t=tt, noise=noise)
        pred = diff.denoise_fn(xt, tt, cond=None)
        x0p = diff.predict_start_from_noise(xt, tt, pred)
        d = ((x0p + 1) * 0.5 - 0.5) * 2 * scale                 # unnorm -> [0,1] -> px, (K,2,F,H,W)
        fx2 = (d[:, 0] ** 2); fy2 = (d[:, 1] ** 2); m = mask_t
        xf = (fx2 * m).sum((1, 2, 3)) / ((fx2 + fy2) * m + 1e-9).sum((1, 2, 3))
        return xf.mean().item()
    t_mid = args.ts[len(args.ts) // 2]
    xfrac_in_y = xfrac_after_denoise(90, t_mid)      # input is +y (x-frac ~0); output x-frac = pulled-back-to-x?
    xfrac_in_x = xfrac_after_denoise(0, t_mid)       # sanity: x input stays x

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for t in args.ts:
        ax.plot(phis, curve[t], "-o", ms=3, label=f"t={t}  (A={aniso[t]:.2f})")
    for x in (0, 180, 360):
        ax.axvline(x, color="g", ls=":", alpha=.5)
    for x in (90, 270):
        ax.axvline(x, color="r", ls=":", alpha=.5)
    ax.set_xlabel("velocity-vector rotation phi (deg)   [green=x-axis, red=y-axis]")
    ax.set_ylabel("denoising residual on moving region")
    ax.set_title("direction probe: did the ckpt learn velocity ∈ ±x?")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "direction_probe.png"), dpi=120); plt.close()

    print(json.dumps({
        "anisotropy_err90_over_err0": {str(t): round(aniso[t], 3) for t in args.ts},
        "rotate_back_test": {"t": t_mid,
                             "x_energy_frac_after_denoise__input_+y": round(xfrac_in_y, 3),
                             "x_energy_frac_after_denoise__input_+x": round(xfrac_in_x, 3),
                             "note": "input +y has x-frac~0; if model pulls toward x-axis this rises toward 1"},
        "err_curve": {str(t): [round(v, 5) for v in curve[t]] for t in args.ts},
        "phis_deg": [int(p) for p in phis]}))


if __name__ == "__main__":
    main()
