"""SDS: distill the flow teacher's score into a single global v0 vector.

The pipeline is fully differentiable: v0 -> MPM rollout -> screen-flow render -> the
teacher's denoiser score. SDS gradient (eps-parameterisation, unconditional teacher):

    grad_x0 = w(t) * (eps_pred(x_t, t) - eps),   x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) eps

is backpropagated through render+rollout to v0. With a teacher that learned the +-x
flow distribution, SDS should pull v0 onto the x axis. This is the single-point
(mode-seeking) precursor to VSD; it tests whether the score-> v0 gradient path works
and which mode it falls into.

  micromamba run -n physdreamer python -m vsd.sds_v0 --iters 80 --init 0.3 0.2 0.1
"""
import vsd.bootstrap  # noqa: F401

import argparse
import json
import os
from typing import List

import imageio.v2 as imageio
import numpy as np
import torch
from torch import Tensor

from video_diffusion_pytorch import GaussianDiffusion, Unet3D

from vsd.flow_render import render_flow
from vsd.scene_min import apply_scene_fixes, load_camera, load_min_scene
from vsd.traj import V0Trajectory

DATA = "outputs/gen_flow_aligned/02_n128_axisx_mag2-8_rot67.6"
CKPT = "teacher/out_02_flow_aligned_mag2-8/checkpoints/diff_final.pt"
ROT_Z_DEG = 67.6  # object alignment baked into the dataset; SDS scene must match
RES = 128


def load_teacher(device: str) -> GaussianDiffusion:
    """Build the flow-teacher arch and load the trained weights (eval)."""
    unet = Unet3D(dim=64, dim_mults=(1, 2, 4, 8), channels=2)
    diff = GaussianDiffusion(unet, image_size=RES, num_frames=7, channels=2,
                             timesteps=1000, loss_type="l2").to(device)
    diff.load_state_dict(torch.load(CKPT, map_location=device)["diffusion"])
    diff.eval()
    for p in diff.parameters():
        p.requires_grad_(False)
    return diff


def flow_to_rgb(f2: np.ndarray) -> np.ndarray:
    """packed flow [2,H,W] in [0,1] -> [H,W,3] viz (B=0.5)."""
    f2 = np.transpose(f2, (1, 2, 0))
    return np.concatenate([f2, np.full(f2.shape[:2] + (1,), 0.5, np.float32)], -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--init", type=str, default="0.3,0.2,0.1",
                    help="initial v0 vector 'vx,vy,vz' (comma-sep so negatives work)")
    ap.add_argument("--t_min", type=int, default=20)
    ap.add_argument("--t_max", type=int, default=500)
    ap.add_argument("--n_noise", type=int, default=2, help="noise samples averaged per step")
    ap.add_argument("--grad_window", type=int, default=2, help="trailing BPTT frames per rollout")
    ap.add_argument("--w_pow", type=float, default=0.0, help="weight w(t)=(1-abar_t)**w_pow")
    ap.add_argument("--out", default="vsd/out/sds_v0")
    args = ap.parse_args()
    dev = "cuda:0"
    os.makedirs(args.out, exist_ok=True)

    scene = load_min_scene(os.path.join(DATA, "scene_cache.pt"), device=dev)
    scene = apply_scene_fixes(scene, rot_z_deg=ROT_Z_DEG, recenter=False)  # match dataset's rotated frame
    cam = load_camera(os.path.join(DATA, "camera.json"), device=dev)
    scale_px = float(json.load(open(os.path.join(DATA, "flow_pack_128_t8.npy.meta.json")))["scale_px"])
    diff = load_teacher(dev)
    abar = diff.alphas_cumprod.to(dev)                                   # [1000]
    builder = V0Trajectory(scene, E=1e5, n_flow=7, device=dev)

    init = [float(x) for x in args.init.split(",")]                                    # [3]
    v0 = torch.tensor(init, dtype=torch.float32, device=dev, requires_grad=True)        # [3]
    opt = torch.optim.Adam([v0], lr=args.lr)

    hist: List[dict] = []
    print(f"init v0={args.init}  iters={args.iters} lr={args.lr} ts=[{args.t_min},{args.t_max}] "
          f"grad_window={args.grad_window}")
    for it in range(args.iters):
        opt.zero_grad(set_to_none=True)
        world = builder.world_traj(v0, grad_window=args.grad_window)     # [8,n_move,3]
        flow = render_flow(world, cam, scale_px, RES)                    # [7,2,128,128] (F,C,H,W) in [0,1]
        x0 = (flow.permute(1, 0, 2, 3) * 2 - 1).unsqueeze(0)            # [1,2,7,128,128] (B,C,F,H,W) in [-1,1]

        # SDS gradient target: computed with the teacher in NO_GRAD (the score is a
        # fixed target -- standard SDS stops grad through the UNet; this is also what
        # keeps memory bounded, no UNet activations retained for backward).
        sds_grad = torch.zeros_like(x0)
        loss_mag = 0.0
        with torch.no_grad():
            x0d = x0.detach()
            for _ in range(args.n_noise):
                t = torch.randint(args.t_min, args.t_max, (1,), device=dev)  # [1]
                noise = torch.randn_like(x0d)
                xt = diff.q_sample(x0d, t, noise)
                eps_pred = diff.denoise_fn(xt, t, cond=None)
                w = (1 - abar[t]).view(-1, 1, 1, 1, 1) ** args.w_pow         # weight
                sds_grad = sds_grad + w * (eps_pred - noise) / args.n_noise
                loss_mag += float((eps_pred - noise).pow(2).mean()) / args.n_noise
        # surrogate loss whose grad wrt x0 is sds_grad (constant) -> backprop to v0
        loss = (x0 * sds_grad).sum()
        loss.backward()
        gnorm = float(v0.grad.norm())
        opt.step()

        vv = v0.detach().cpu().numpy()
        rec = {"it": it, "v0x": float(vv[0]), "v0y": float(vv[1]), "v0z": float(vv[2]),
               "v0_norm": float(np.linalg.norm(vv)), "eps_mse": loss_mag, "gnorm": gnorm}
        hist.append(rec)
        if it % 5 == 0 or it == args.iters - 1:
            print(f"  it {it:3d}  v0=({vv[0]:+.3f},{vv[1]:+.3f},{vv[2]:+.3f}) "
                  f"|v0|={rec['v0_norm']:.3f}  eps_mse={loss_mag:.4f}  gnorm={gnorm:.2e}", flush=True)

    # save history + curves + final flow gif
    json.dump(hist, open(os.path.join(args.out, "history.json"), "w"), indent=2)
    _plot(hist, os.path.join(args.out, "sds_curves.png"))
    with torch.no_grad():
        world = builder.world_traj(v0, grad_window=1)
        flow = render_flow(world, cam, scale_px, RES).cpu().numpy()      # [7,2,128,128]
    frames = [(flow_to_rgb(flow[t]) * 255).round().astype("uint8") for t in range(flow.shape[0])]
    gif = os.path.join(args.out, "final_flow.gif")
    imageio.mimsave(gif, frames, fps=3)
    vv = v0.detach().cpu().numpy()
    print(f"\nFINAL v0=({vv[0]:+.4f},{vv[1]:+.4f},{vv[2]:+.4f})  |v0|={np.linalg.norm(vv):.4f}")
    print(f"  x-axis alignment |v0x|/|v0| = {abs(vv[0])/max(np.linalg.norm(vv),1e-9):.3f}  "
          f"(1.0 = pure x-axis, as the dataset)")
    print(f"saved curves {args.out}/sds_curves.png  and  final flow gif {gif}")


def _plot(hist: List[dict], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    it = [h["it"] for h in hist]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for key, lab in (("v0x", "v0_x"), ("v0y", "v0_y"), ("v0z", "v0_z")):
        ax[0].plot(it, [h[key] for h in hist], label=lab)
    ax[0].axhline(0, color="k", ls=":"); ax[0].set_title("v0 components"); ax[0].legend()
    ax[0].set_xlabel("iter")
    ax[1].plot(it, [h["eps_mse"] for h in hist]); ax[1].set_title("eps MSE (||eps_pred-eps||^2)")
    ax[1].set_xlabel("iter")
    ax[2].plot(it, [h["v0_norm"] for h in hist]); ax[2].set_title("|v0|"); ax[2].set_xlabel("iter")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


if __name__ == "__main__":
    main()
