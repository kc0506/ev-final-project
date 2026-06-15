"""VSD over a 1-D scalar vx (vy=vz=0 LOCKED): a particle ensemble whose stationary
distribution should match the teacher's learned p(vx). Unlike SDS (single point,
mode-seeking) this maintains K particles + a trainable auxiliary score and uses the
VSD gradient w(t)(eps_teacher - eps_aux), so the particles SPREAD to cover the density
instead of collapsing -> the final particle histogram IS the recovered marginal.

1-D speedup: vx -> flow is a deterministic 1-D map, so we tabulate flow(vx) on a grid
ONCE and read each particle's (differentiable) flow by linear interpolation -- no MPM
rollout inside the VSD loop.

  python -m vsd.vsd_1d --k 48 --iters 400 \
      --ckpt teacher/out_02_flow_aligned_mag2-8/checkpoints/diff_final.pt \
      --data outputs/gen_flow_aligned/02_n128_axisx_mag2-8_rot67.6 --vmin 2 --vmax 8

NOTE: written while the GPU was down; smoke-test before trusting numbers.
"""
import vsd.bootstrap  # noqa: F401

import argparse
import json
import os
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from video_diffusion_pytorch import GaussianDiffusion, Unet3D

from vsd.flow_render import render_flow
from vsd.scene_min import apply_scene_fixes, load_camera, load_min_scene
from vsd.traj import V0Trajectory

RES = 128
ROT = 67.6


def build_diff(dev: str, train: bool) -> GaussianDiffusion:
    """A flow GaussianDiffusion (teacher arch). train=False -> frozen eval."""
    unet = Unet3D(dim=64, dim_mults=(1, 2, 4, 8), channels=2)
    diff = GaussianDiffusion(unet, image_size=RES, num_frames=7, channels=2,
                             timesteps=1000, loss_type="l2").to(dev)
    if not train:
        diff.eval()
        for p in diff.parameters():
            p.requires_grad_(False)
    return diff


@torch.no_grad()
def precompute_flow_grid(builder: V0Trajectory, cam, scale_px: float,
                         vx_grid: np.ndarray, dev: str) -> Tensor:
    """Render flow(vx) for each grid vx -> [G,2,7,RES,RES] in [0,1] (constant table)."""
    grid = []
    for vx in vx_grid:
        world = builder.world_traj(torch.tensor([float(vx), 0.0, 0.0], device=dev),
                                   grad_window=1, requires_grad=False)     # no tape -> fast
        grid.append(render_flow(world, cam, scale_px, RES))               # [7,2,RES,RES] (F,C,H,W)
    return torch.stack(grid, 0).permute(0, 2, 1, 3, 4).contiguous()        # [G,2,7,RES,RES] (C,F,H,W)


def interp_flow(flow_grid: Tensor, vx: Tensor, vmin: float, dvx: float) -> Tensor:
    """Differentiable linear interpolation of the flow table at scalar particle vx.
    flow_grid [G,2,7,H,W]; vx [K] -> [K,2,7,H,W]; grad flows to vx (lerp frac)."""
    G = flow_grid.shape[0]
    pos = (vx - vmin) / dvx                                                # [K]
    g0 = pos.floor().clamp(0, G - 2).long()                               # [K]
    f = (pos - g0.float()).clamp(0, 1).view(-1, 1, 1, 1, 1)               # [K,1,1,1,1]
    a = flow_grid[g0]                                                      # [K,2,7,H,W]
    b = flow_grid[g0 + 1]
    return a * (1 - f) + b * f                                            # [K,2,7,H,W]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="teacher/out_02_flow_aligned_mag2-8/checkpoints/diff_final.pt")
    ap.add_argument("--data", default="outputs/gen_flow_aligned/02_n128_axisx_mag2-8_rot67.6")
    ap.add_argument("--vmin", type=float, default=2.0)
    ap.add_argument("--vmax", type=float, default=8.0)
    ap.add_argument("--grid", type=int, default=121, help="flow table resolution over [vmin,vmax]")
    ap.add_argument("--k", type=int, default=48, help="particles")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--aux_lr", type=float, default=1e-4)
    ap.add_argument("--aux_steps", type=int, default=1, help="aux updates per particle update")
    ap.add_argument("--chunk", type=int, default=8, help="particle sub-batch for UNet eval")
    ap.add_argument("--t_min", type=int, default=20)
    ap.add_argument("--t_max", type=int, default=800)
    ap.add_argument("--w_pow", type=float, default=0.0)
    ap.add_argument("--init_pad", type=float, default=0.5, help="init particles inside [vmin+pad, vmax-pad]")
    ap.add_argument("--out", default="vsd/out/vsd_1d")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt_every", type=int, default=50, help="save full resumable ckpt every N iters (<=0 off)")
    ap.add_argument("--resume", default="no", help="ckpt path | 'auto' (out/ckpt/latest.pt) | 'no'")
    args = ap.parse_args()
    dev = "cuda:0"
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    # scene/camera/teacher (must match the dataset the teacher trained on)
    scene = load_min_scene(os.path.join(args.data, "scene_cache.pt"), device=dev)
    scene = apply_scene_fixes(scene, rot_z_deg=ROT, recenter=False)
    cam = load_camera(os.path.join(args.data, "camera.json"), device=dev)
    scale_px = float(json.load(open(os.path.join(args.data, "flow_pack_128_t8.npy.meta.json")))["scale_px"])
    builder = V0Trajectory(scene, E=1e5, n_flow=7, device=dev, requires_grad=False)  # table only

    teacher = build_diff(dev, train=False)
    teacher.load_state_dict(torch.load(args.ckpt, map_location=dev)["diffusion"])
    aux = build_diff(dev, train=True)
    aux.load_state_dict(torch.load(args.ckpt, map_location=dev)["diffusion"])  # init aux = teacher
    abar = teacher.alphas_cumprod.to(dev)                                  # [1000]
    opt_aux = torch.optim.AdamW(aux.parameters(), lr=args.aux_lr)

    # flow table over [vmin,vmax]
    vx_grid = np.linspace(args.vmin, args.vmax, args.grid)
    dvx = float(vx_grid[1] - vx_grid[0])
    print(f"tabulating flow over {args.grid} vx in [{args.vmin},{args.vmax}] ...", flush=True)
    flow_grid = precompute_flow_grid(builder, cam, scale_px, vx_grid, dev)  # [G,2,7,H,W]
    print("flow table done", flush=True)

    # particles spread across the support
    vx = torch.tensor(rng.uniform(args.vmin + args.init_pad, args.vmax - args.init_pad, args.k),
                      dtype=torch.float32, device=dev, requires_grad=True)  # [K]
    opt_p = torch.optim.Adam([vx], lr=args.lr)

    def eps_both(xt: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """teacher (no grad) + aux (no grad) eps on xt [K,2,7,H,W]; chunked over K."""
        et, ea = [], []
        for i in range(0, xt.shape[0], args.chunk):
            sl = slice(i, i + args.chunk)
            tt = t[sl]
            et.append(teacher.denoise_fn(xt[sl], tt, cond=None))
            ea.append(aux.denoise_fn(xt[sl], tt, cond=None))
        return torch.cat(et), torch.cat(ea)

    ckpt_dir = os.path.join(args.out, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    def save_ckpt(name: str, it: int, hist: list) -> None:
        """Full resumable VSD state: particles + their optimizer + aux net + aux optimizer.
        (vx alone can't resume -- the aux score net spent the whole run adapting to q.)"""
        path = os.path.join(ckpt_dir, name)
        tmp = path + ".tmp"
        torch.save({"vx": vx.detach().cpu(), "opt_p": opt_p.state_dict(),
                    "aux": aux.state_dict(), "opt_aux": opt_aux.state_dict(),
                    "it": it, "history": hist, "args": vars(args)}, tmp)
        os.replace(tmp, path)

    hist: list = []
    start_it = 0
    rp = (os.path.join(ckpt_dir, "latest.pt") if args.resume == "auto"
          else (args.resume if args.resume != "no" else None))
    if rp and os.path.exists(rp):
        ck = torch.load(rp, map_location=dev)
        with torch.no_grad():
            vx.copy_(ck["vx"].to(dev))                                    # in-place keeps the leaf + opt_p ref
        opt_p.load_state_dict(ck["opt_p"])
        aux.load_state_dict(ck["aux"])
        opt_aux.load_state_dict(ck["opt_aux"])
        hist = ck.get("history", [])
        start_it = int(ck.get("it", 0)) + 1
        print(f"[resume] from {rp} -> it {start_it}/{args.iters}", flush=True)

    for it in range(start_it, args.iters):
        flow_k = interp_flow(flow_grid, vx, args.vmin, dvx)                # [K,2,7,H,W] diff wrt vx
        x0 = flow_k * 2 - 1                                                # [-1,1]

        # --- aux score update: learn the score of the CURRENT particle renders (DSM loss)
        la_sum, la_n = 0.0, 0
        for _ in range(args.aux_steps):
            opt_aux.zero_grad(set_to_none=True)
            x0d = x0.detach()
            t = torch.randint(args.t_min, args.t_max, (x0d.shape[0],), device=dev)
            noise = torch.randn_like(x0d)
            xt = aux.q_sample(x0d, t, noise)
            for i in range(0, x0d.shape[0], args.chunk):
                sl = slice(i, i + args.chunk)
                pred = aux.denoise_fn(xt[sl], t[sl], cond=None)
                loss = torch.nn.functional.mse_loss(pred, noise[sl])
                loss.backward()
                la_sum += float(loss); la_n += 1
            opt_aux.step()
        aux_loss = la_sum / max(la_n, 1)                                  # mean DSM loss this iter

        # --- VSD gradient on particles: w(t)(eps_teacher - eps_aux), backprop to vx
        t = torch.randint(args.t_min, args.t_max, (x0.shape[0],), device=dev)
        noise = torch.randn_like(x0)
        with torch.no_grad():
            xt = teacher.q_sample(x0, t, noise)
            eps_t, eps_a = eps_both(xt, t)
            w = (1 - abar[t]).view(-1, 1, 1, 1, 1) ** args.w_pow
            vsd_grad = w * (eps_t - eps_a)                                 # [K,2,7,H,W]
            eps_diff = float((eps_t - eps_a).pow(2).mean().sqrt())        # KL-gradient scale (->0 = q≈teacher)
        opt_p.zero_grad(set_to_none=True)
        (x0 * vsd_grad).sum().backward()                                  # grad flows x0->flow->vx
        grad_norm = float(vx.grad.norm())                                 # particle-grad norm (->0 = converged)
        opt_p.step()
        with torch.no_grad():
            vx.clamp_(args.vmin, args.vmax)                               # keep on support

        v = vx.detach().cpu().numpy()
        rec = {"it": it, "mean": float(v.mean()), "std": float(v.std()),
               "aux_loss": round(aux_loss, 6), "eps_diff": round(eps_diff, 6),
               "grad_norm": round(grad_norm, 6)}
        if it % 10 == 0 or it == args.iters - 1:
            rec["vx"] = [round(float(x), 3) for x in v]
            print(f"  it {it:4d}  vx mean={v.mean():.2f} std={v.std():.2f}  "
                  f"aux_loss={aux_loss:.4f} eps_diff={eps_diff:.4f} grad={grad_norm:.4f}", flush=True)
        hist.append(rec)
        if args.ckpt_every > 0 and (it + 1) % args.ckpt_every == 0:
            save_ckpt("latest.pt", it, hist)
            save_ckpt(f"vsd_it{it + 1:04d}.pt", it, hist)

    save_ckpt("latest.pt", args.iters - 1, hist)
    save_ckpt("final.pt", args.iters - 1, hist)
    v = vx.detach().cpu().numpy()
    json.dump({"args": vars(args), "vx_final": [float(x) for x in v], "history": hist},
              open(os.path.join(args.out, "result.json"), "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    ax[0].hist(v, bins=24, range=(args.vmin, args.vmax), density=True, alpha=.8)
    xs = np.linspace(args.vmin, args.vmax, 100)
    ramp = 2 * xs / (args.vmax ** 2 - args.vmin ** 2)                      # p(m)∝m normalised on [vmin,vmax]
    ax[0].plot(xs, ramp, "r--", label="linear source p(vx)∝vx")
    ax[0].set_title(f"recovered q(vx)  (K={args.k})"); ax[0].set_xlabel("vx"); ax[0].legend()
    its = [h["it"] for h in hist]
    ax[1].plot(its, [h["mean"] for h in hist], label="mean")
    ax[1].fill_between(its, [h["mean"] - h["std"] for h in hist], [h["mean"] + h["std"] for h in hist], alpha=.2)
    ax[1].set_title("particle vx mean±std"); ax[1].set_xlabel("iter"); ax[1].legend()
    ax[2].plot(its, [h["aux_loss"] for h in hist], label="aux DSM loss", color="C1")
    ax[2].plot(its, [h["eps_diff"] for h in hist], label="||eps_t-eps_a|| (KL-grad scale)", color="C2", alpha=.8)
    ax[2].plot(its, [h["grad_norm"] for h in hist], label="particle grad norm", color="C3", alpha=.8)
    ax[2].set_yscale("log"); ax[2].set_title("VSD losses"); ax[2].set_xlabel("iter"); ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "vsd_marginal.png"), dpi=120); plt.close()
    print(f"\nFINAL q(vx): mean={v.mean():.2f} std={v.std():.2f}")
    print(f"saved {args.out}/vsd_marginal.png  and  result.json + ckpt/")


if __name__ == "__main__":
    main()
