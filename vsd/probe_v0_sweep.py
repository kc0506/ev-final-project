"""OOD gap probe: sweep v0_x across the WHOLE axis (gap + bands + beyond) and measure
the teacher's denoising residual on a FIXED mask. The dataset is v0_x in +-[2,8] with a
deliberate GAP at (-2,2). If the teacher learned the distribution's SUPPORT, residual is
low inside the bands and HIGH in the gap (clips that are physically valid but never in
training) and beyond +-8 -- a valley shape that proves the gap was learned (so a recovered
density near 0 is unambiguously OOD, not 'maybe in-dist').

Fixed mask (the band's moving region) keeps residuals comparable across v0_x, removing the
|v0_x|-scales-the-signal confound that contaminates a per-clip mask.

  python -m vsd.probe_v0_sweep --n 37 --vmax 9
"""
import vsd.bootstrap  # noqa: F401

import argparse
import json
import os
from typing import List

import numpy as np
import torch
from torch import Tensor

from video_diffusion_pytorch import GaussianDiffusion, Unet3D

from vsd.flow_render import render_flow
from vsd.scene_min import apply_scene_fixes, load_camera, load_min_scene
from vsd.traj import V0Trajectory

DATA = "outputs/gen_flow_aligned/02_n128_axisx_mag2-8_rot67.6"
CKPT = "teacher/out_02_flow_aligned_mag2-8/checkpoints/diff_final.pt"
ROT = 67.6
RES = 128


def load_teacher(dev: str) -> GaussianDiffusion:
    unet = Unet3D(dim=64, dim_mults=(1, 2, 4, 8), channels=2)
    diff = GaussianDiffusion(unet, image_size=RES, num_frames=7, channels=2,
                             timesteps=1000, loss_type="l2").to(dev)
    diff.load_state_dict(torch.load(CKPT, map_location=dev)["diffusion"])
    diff.eval()
    for p in diff.parameters():
        p.requires_grad_(False)
    return diff


@torch.no_grad()
def residual(diff: GaussianDiffusion, x0: Tensor, mask: Tensor,
             ts: List[int], nnoise: int, dev: str) -> float:
    """x0 [1,2,7,128,128] in [-1,1]; mask [128,128] in {0,1} -> mean denoising residual."""
    out = 0.0
    for t in ts:
        tt = torch.full((1,), t, device=dev, dtype=torch.long)
        for _ in range(nnoise):
            noise = torch.randn_like(x0)
            xt = diff.q_sample(x0, tt, noise)
            pred = diff.denoise_fn(xt, tt, cond=None)
            x0p = diff.predict_start_from_noise(xt, tt, pred)
            e = ((x0p - x0) ** 2).mean(1)[0]                 # [7,128,128] over channels
            m = mask[None]                                   # [1,128,128]
            out += float((e * m).sum() / m.expand_as(e).sum().clamp(min=1))
    return out / (len(ts) * nnoise)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=37, help="sweep points over [-vmax, vmax]")
    ap.add_argument("--vmax", type=float, default=9.0)
    ap.add_argument("--ts", type=int, nargs="+", default=[300, 500])
    ap.add_argument("--nnoise", type=int, default=3)
    ap.add_argument("--mask_ref", type=float, default=5.0, help="|v0_x| whose moving region is the fixed mask")
    ap.add_argument("--out", default="vsd/out/probe_v0_sweep")
    args = ap.parse_args()
    dev = "cuda:0"
    os.makedirs(args.out, exist_ok=True)

    scene = load_min_scene(os.path.join(DATA, "scene_cache.pt"), device=dev)
    scene = apply_scene_fixes(scene, rot_z_deg=ROT, recenter=False)
    cam = load_camera(os.path.join(DATA, "camera.json"), device=dev)
    scale_px = float(json.load(open(os.path.join(DATA, "flow_pack_128_t8.npy.meta.json")))["scale_px"])
    diff = load_teacher(dev)
    builder = V0Trajectory(scene, E=1e5, n_flow=7, device=dev)

    def flow_x0(vx: float) -> Tensor:
        world = builder.world_traj(torch.tensor([vx, 0.0, 0.0], device=dev), grad_window=1)  # [8,nm,3]
        flow = render_flow(world, cam, scale_px, RES)                                        # [7,2,128,128]
        return (flow.permute(1, 0, 2, 3) * 2 - 1).unsqueeze(0)                               # [1,2,7,128,128]

    # FIXED mask = moving region of a reference band clip (top 15% temporal std)
    with torch.no_grad():
        ref = flow_x0(args.mask_ref)[0]                      # [2,7,128,128]
        s = ref.mean(0).std(0)                               # [128,128] temporal std of content
        mask = (s >= torch.quantile(s, 0.85)).float()        # [128,128]

    vxs = np.linspace(-args.vmax, args.vmax, args.n)
    errs = []
    for vx in vxs:
        with torch.no_grad():
            x0 = flow_x0(float(vx))
        errs.append(residual(diff, x0, mask, args.ts, args.nnoise, dev))
        print(f"  vx={vx:+.2f}  err={errs[-1]:.5f}", flush=True)
    errs = np.array(errs)

    res = {"vxs": [round(float(v), 2) for v in vxs], "errs": [round(float(e), 6) for e in errs],
           "band": [2, 8], "gap": [-2, 2], "mask_ref": args.mask_ref,
           "gap_mean": round(float(errs[np.abs(vxs) < 2].mean()), 5),
           "band_mean": round(float(errs[(np.abs(vxs) >= 2) & (np.abs(vxs) <= 8)].mean()), 5)}
    json.dump(res, open(os.path.join(args.out, "sweep.json"), "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("gap_mean", "band_mean")}))
    print(f"gap/band residual ratio = {res['gap_mean']/max(res['band_mean'],1e-9):.2f} "
          f"(>1 = teacher learned the gap)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9, 4.5))
    plt.plot(vxs, errs, "-o", ms=4)
    plt.axvspan(2, 8, color="g", alpha=.12, label="train band +x")
    plt.axvspan(-8, -2, color="g", alpha=.12, label="train band -x")
    plt.axvspan(-2, 2, color="r", alpha=.12, label="GAP (OOD)")
    plt.xlabel("v0_x (swept)"); plt.ylabel("teacher denoising residual (fixed mask)")
    plt.title("OOD gap probe: residual vs v0_x  (valley in bands, high in gap = gap learned)")
    plt.legend(fontsize=8); plt.tight_layout()
    p = os.path.join(args.out, "v0_sweep.png")
    plt.savefig(p, dpi=120); plt.close()
    print(f"saved {p}  and  {args.out}/sweep.json")


if __name__ == "__main__":
    main()
