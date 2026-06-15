"""VSD-over-vx on Modal: the same particle-ensemble recovery as vsd/vsd_1d.py, but the
flow table is loaded from the Volume (precomputed locally) so NO warp/reuse_mpm is needed
on the cloud -- only torch + video_diffusion (validated by modal_smoke). L40S has 48GB so
we use a big particle chunk (the local 16GB laptop forced chunk=1).

Fanned out over SEEDS: independent VSD chains run in parallel -> tests whether the recovered
q(vx) (and its high-vx bias) is seed-robust, and gives across-seed variance. Uses the
epoch-299 teacher (train04_ckpts/diff_final.pt) on the volume.

  modal run vsd/vsd_1d_modal.py --n-seeds 3 --iters 300
"""
import json

import modal

TORCH_CU118 = "https://download.pytorch.org/whl/cu118"
image = (  # identical image -> cache hit
    modal.Image.from_registry("nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04", add_python="3.10")
    .pip_install("torch==2.0.0+cu118", "torchvision==0.15.1+cu118", extra_index_url=TORCH_CU118)
    .pip_install("numpy<2", "warp-lang==0.10.1", "video-diffusion-pytorch==0.7.0",
                 "einops==0.8.2", "einops-exts==0.0.4", "rotary-embedding-torch==0.8.9")
)
vol = modal.Volume.from_name("physgen-logp", create_if_missing=True)
app = modal.App("physgen-vsd", image=image)

TABLE = "flow_grid_out04.npy"          # precomputed flow table on the volume
CKPT = "train04_ckpts/diff_final.pt"   # epoch-299 teacher on the volume


@app.function(gpu="L40S", volumes={"/data": vol}, timeout=5400)
def vsd_chain(spec_json: str) -> str:
    """Run one VSD chain (one seed) and return {seed, mean, std, vx_final, history}."""
    import os

    import numpy as np
    import torch
    from video_diffusion_pytorch import GaussianDiffusion, Unet3D

    s = json.loads(spec_json)
    dev = "cuda"
    res, vmin, vmax = int(s["res"]), float(s["vmin"]), float(s["vmax"])
    k, iters, chunk = int(s["k"]), int(s["iters"]), int(s["chunk"])
    aux_steps, t_min, t_max = int(s["aux_steps"]), int(s["t_min"]), int(s["t_max"])
    lr, aux_lr, w_pow, pad = float(s["lr"]), float(s["aux_lr"]), float(s["w_pow"]), float(s["init_pad"])
    seed = int(s["seed"])
    ckpt_every, resume = int(s.get("ckpt_every", 0)), bool(s.get("resume", False))
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    table = torch.from_numpy(np.load(f"/data/{TABLE}").astype(np.float32)).to(dev)  # [G,2,7,H,W] [0,1]
    G = int(table.shape[0])
    dvx = (vmax - vmin) / (G - 1)

    def build(train: bool) -> GaussianDiffusion:
        unet = Unet3D(dim=64, dim_mults=(1, 2, 4, 8), channels=2)
        d = GaussianDiffusion(unet, image_size=res, num_frames=7, channels=2,
                              timesteps=1000, loss_type="l2").to(dev)
        if not train:
            d.eval()
            for p in d.parameters():
                p.requires_grad_(False)
        return d

    sd = torch.load(f"/data/{CKPT}", map_location=dev)["diffusion"]
    teacher = build(False); teacher.load_state_dict(sd)
    aux = build(True); aux.load_state_dict(sd)                              # aux init = teacher
    abar = teacher.alphas_cumprod.to(dev)                                  # [1000]
    opt_aux = torch.optim.AdamW(aux.parameters(), lr=aux_lr)

    vx = torch.tensor(rng.uniform(vmin + pad, vmax - pad, k), dtype=torch.float32,
                      device=dev, requires_grad=True)                      # [K]
    opt_p = torch.optim.Adam([vx], lr=lr)

    def interp(vxv: torch.Tensor) -> torch.Tensor:
        """differentiable lerp of the flow table at particle vx [K] -> [K,2,7,H,W]."""
        pos = (vxv - vmin) / dvx
        g0 = pos.floor().clamp(0, G - 2).long()
        f = (pos - g0.float()).clamp(0, 1).view(-1, 1, 1, 1, 1)
        return table[g0] * (1 - f) + table[g0 + 1] * f

    def eps_both(xt: torch.Tensor, t: torch.Tensor):
        et, ea = [], []
        for i in range(0, xt.shape[0], chunk):
            sl = slice(i, i + chunk)
            et.append(teacher.denoise_fn(xt[sl], t[sl], cond=None))
            ea.append(aux.denoise_fn(xt[sl], t[sl], cond=None))
        return torch.cat(et), torch.cat(ea)

    # per-seed checkpoint dir on the volume (so a dying container loses nothing + can resume)
    cdir = f"/data/vsd_ckpts/seed{seed}"
    os.makedirs(cdir, exist_ok=True)

    def save_ckpt(name: str, it: int, hist: list) -> None:
        """Full resumable VSD state -> volume (vx + opt_p + aux net + aux opt + history)."""
        tmp = f"{cdir}/{name}.tmp"
        torch.save({"vx": vx.detach().cpu(), "opt_p": opt_p.state_dict(),
                    "aux": aux.state_dict(), "opt_aux": opt_aux.state_dict(),
                    "it": it, "history": hist}, tmp)
        os.replace(tmp, f"{cdir}/{name}")
        vol.commit()                                                       # persist for download/resume

    hist: list = []
    start_it = 0
    if resume and os.path.exists(f"{cdir}/latest.pt"):
        ck = torch.load(f"{cdir}/latest.pt", map_location=dev)
        with torch.no_grad():
            vx.copy_(ck["vx"].to(dev))
        opt_p.load_state_dict(ck["opt_p"]); aux.load_state_dict(ck["aux"]); opt_aux.load_state_dict(ck["opt_aux"])
        hist = ck.get("history", []); start_it = int(ck.get("it", 0)) + 1
        print(f"[resume] seed{seed} from it {start_it}/{iters}", flush=True)

    for it in range(start_it, iters):
        x0 = interp(vx) * 2 - 1                                            # [K,2,7,H,W] in [-1,1]
        la_sum, la_n = 0.0, 0
        for _ in range(aux_steps):
            opt_aux.zero_grad(set_to_none=True)
            x0d = x0.detach()
            t = torch.randint(t_min, t_max, (x0d.shape[0],), device=dev)
            noise = torch.randn_like(x0d)
            xt = aux.q_sample(x0d, t, noise)
            for i in range(0, x0d.shape[0], chunk):
                sl = slice(i, i + chunk)
                loss = torch.nn.functional.mse_loss(aux.denoise_fn(xt[sl], t[sl], cond=None), noise[sl])
                loss.backward()
                la_sum += float(loss); la_n += 1
            opt_aux.step()
        aux_loss = la_sum / max(la_n, 1)

        t = torch.randint(t_min, t_max, (x0.shape[0],), device=dev)
        noise = torch.randn_like(x0)
        with torch.no_grad():
            xt = teacher.q_sample(x0, t, noise)
            eps_t, eps_a = eps_both(xt, t)
            w = (1 - abar[t]).view(-1, 1, 1, 1, 1) ** w_pow
            vsd_grad = w * (eps_t - eps_a)                                 # [K,2,7,H,W]
            eps_diff = float((eps_t - eps_a).pow(2).mean().sqrt())
        opt_p.zero_grad(set_to_none=True)
        (x0 * vsd_grad).sum().backward()
        grad_norm = float(vx.grad.norm())
        opt_p.step()
        with torch.no_grad():
            vx.clamp_(vmin, vmax)

        v = vx.detach().cpu().numpy()
        rec = {"it": it, "mean": float(v.mean()), "std": float(v.std()),
               "aux_loss": round(aux_loss, 6), "eps_diff": round(eps_diff, 6),
               "grad_norm": round(grad_norm, 6)}
        if it % 10 == 0 or it == iters - 1:
            rec["vx"] = [round(float(x), 3) for x in v]
            print(f"  seed{seed} it {it:4d}  mean={v.mean():.2f} std={v.std():.2f} "
                  f"aux={aux_loss:.4f} eps_diff={eps_diff:.4f} grad={grad_norm:.4f}", flush=True)
        hist.append(rec)
        if ckpt_every > 0 and (it + 1) % ckpt_every == 0:
            save_ckpt("latest.pt", it, hist)
            save_ckpt(f"it{it + 1:04d}.pt", it, hist)

    if ckpt_every > 0:
        save_ckpt("final.pt", iters - 1, hist)
    v = vx.detach().cpu().numpy()
    return json.dumps({"seed": seed, "mean": float(v.mean()), "std": float(v.std()),
                       "vx_final": [round(float(x), 4) for x in v], "history": hist})


@app.local_entrypoint()
def main(n_seeds: int = 3, k: int = 48, iters: int = 300, chunk: int = 4,
         aux_steps: int = 1, vmin: float = 0.0, vmax: float = 8.0,
         lr: float = 0.05, aux_lr: float = 1e-4, w_pow: float = 0.0, init_pad: float = 0.5,
         t_min: int = 20, t_max: int = 800, res: int = 128, ckpt_every: int = 50, resume: bool = False,
         out: str = "vsd/out/vsd_modal_ep299.json") -> None:
    """Fan out N VSD chains (seeds) in parallel, collect, save combined json. Each chain
    checkpoints to /data/vsd_ckpts/seed{N}/ every ckpt_every iters (resumable)."""
    import os

    specs = [json.dumps({"seed": 100 + i, "k": k, "iters": iters, "chunk": chunk,
                         "aux_steps": aux_steps, "vmin": vmin, "vmax": vmax, "lr": lr,
                         "aux_lr": aux_lr, "w_pow": w_pow, "init_pad": init_pad,
                         "t_min": t_min, "t_max": t_max, "res": res,
                         "ckpt_every": ckpt_every, "resume": resume}) for i in range(n_seeds)]
    print(f"running {n_seeds} VSD chains (ep299, aux_steps={aux_steps}, ckpt_every={ckpt_every}) on L40S ...",
          flush=True)
    chains = [json.loads(r) for r in vsd_chain.map(specs)]

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump({"ckpt": CKPT, "n_seeds": n_seeds, "iters": iters, "aux_steps": aux_steps,
               "chains": chains}, open(out, "w"), indent=2)
    means = [c["mean"] for c in chains]
    print(f"\nsaved {out}")
    for c in chains:
        print(f"  seed {c['seed']}: mean={c['mean']:.2f} std={c['std']:.2f}  ({len(c['history'])} iters logged)")
    print(f"across-seed: mean={sum(means)/len(means):.2f}  range=[{min(means):.2f},{max(means):.2f}]")
    print(f"(baseline ep249 aux1 = 5.70 ; ramp E[vx] = 5.33)")
    # ckpts committed by the remote fn to /data/vsd_ckpts/seed{N}/ (vol.reload() is NOT callable
    # locally). Inspect/fetch with:  modal volume ls/get physgen-logp vsd_ckpts/seed100
