"""Prob-flow ODE log-likelihood of the flow teacher, fanned out over vx on Modal.

WHY ODE (vs the denoising residual): the residual ||x0_pred-x0||^2 is magnitude-confounded
(bigger motion -> bigger absolute residual), so it can't tell "low density" from "small
motion". The probability-flow ODE integrates the divergence (log-volume change) along the
deterministic ODE, which NORMALISES volume -> a true (smoothed) density. We then convert
the ambient density at x0(vx) to the teacher's implied marginal over vx:

    log q(vx) = log p_teacher(x0(vx)) + log |dx0/dvx|      (1-D pushforward back to vx)

and compare its SHAPE to the source ramp p(vx) ∝ vx. If q tracks the ramp, the teacher
learned the density (not just the support); if q piles at high vx, the density is biased.

DIM/VARIANCE: x is 2*7*128*128 = 229k dims dominated by the static background (identical
for every vx). Hutchinson divergence variance scales with full dim, so absolute log p is
noisy -- but we use the SAME seeded Hutchinson noise + SAME ODE time grid for every vx, so
the background's (large, deterministic) contribution CANCELS in relative log p across vx.
Only the small moving region differs. => compare q(vx) up to an additive constant.

Pipeline (3 steps; the table is built locally because Modal has no warp/reuse_mpm):
  1. micromamba run -n physdreamer python -m vsd.build_flow_table --out vsd/out/flow_grid_out04
  2. modal run vsd/logp_ode_modal.py            # uploads table+ckpt, fans out, writes json
  3. micromamba run -n physdreamer python -m vsd.plot_logp vsd/out/logp_out04.json
"""
import json

import modal

TORCH_CU118 = "https://download.pytorch.org/whl/cu118"
# IDENTICAL to vsd/modal_smoke.py so the already-built image is reused from cache.
image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04", add_python="3.10")
    .pip_install("torch==2.0.0+cu118", "torchvision==0.15.1+cu118", extra_index_url=TORCH_CU118)
    .pip_install(
        "numpy<2",
        "warp-lang==0.10.1",
        "video-diffusion-pytorch==0.7.0",
        "einops==0.8.2",
        "einops-exts==0.0.4",
        "rotary-embedding-torch==0.8.9",
    )
)

vol = modal.Volume.from_name("physgen-logp", create_if_missing=True)
app = modal.App("physgen-logp", image=image)


@app.function(gpu="L40S", volumes={"/data": vol}, timeout=2400)
def logp_chunk(spec_json: str) -> str:
    """Compute prob-flow ODE log p + log|dx0/dvx| for a chunk of vx values.

    spec_json carries: vx_list, ckpt, table, vmin, vmax, n_ode, n_hutch, t_eps, seed, res.
    Returns a json list of {vx, logp, logjac, x1_sq_over_D} (x1_sq_over_D ~= 1 if the ODE
    integrated to ~N(0,I): a built-in correctness gauge).
    """
    import numpy as np
    import torch
    from video_diffusion_pytorch import GaussianDiffusion, Unet3D

    s = json.loads(spec_json)
    dev = "cuda:0"
    res: int = s["res"]
    vmin, vmax = float(s["vmin"]), float(s["vmax"])
    n_ode, n_hutch = int(s["n_ode"]), int(s["n_hutch"])
    t_eps, seed = float(s["t_eps"]), int(s["seed"])

    table = torch.from_numpy(np.load(f"/data/{s['table']}")).to(dev)      # [G,2,7,H,W] in [0,1]
    G = int(table.shape[0])
    dvx = (vmax - vmin) / (G - 1)

    unet = Unet3D(dim=64, dim_mults=(1, 2, 4, 8), channels=2)
    diff = GaussianDiffusion(unet, image_size=res, num_frames=7, channels=2,
                             timesteps=1000, loss_type="l2").to(dev)
    diff.load_state_dict(torch.load(f"/data/{s['ckpt']}", map_location=dev)["diffusion"])
    diff.eval()
    for p in diff.parameters():
        p.requires_grad_(False)
    abar = diff.alphas_cumprod.to(dev)                                    # [N]
    betas = diff.betas.to(dev)                                            # [N]
    N = int(abar.shape[0])
    D = 2 * 7 * res * res                                                 # ambient dim

    # shared Rademacher Hutchinson probes (seeded -> identical every container)
    g = torch.Generator(device=dev).manual_seed(seed)
    eps_h = torch.randint(0, 2, (n_hutch, 2, 7, res, res), generator=g, device=dev).float() * 2 - 1  # [H,2,7,H,W]

    def interp(vx: float):
        """vx -> (x0 [1,2,7,H,W] in [-1,1], djac [2,7,H,W] = dx0/dvx)."""
        pos = (vx - vmin) / dvx
        g0 = int(min(max(int(pos), 0), G - 2))
        f = pos - g0
        flow = table[g0] * (1 - f) + table[g0 + 1] * f                    # [2,7,H,W]
        x0 = (flow * 2 - 1).unsqueeze(0)                                  # [1,2,7,H,W]
        djac = (table[g0 + 1] - table[g0]) / dvx * 2                      # [2,7,H,W]
        return x0, djac

    def drift(x: torch.Tensor, idx: int) -> torch.Tensor:
        """Prob-flow ODE drift f = -0.5 beta_c (x + score), score = -eps/sqrt(1-abar). [1,2,7,H,W]."""
        t = torch.full((1,), idx, device=dev, dtype=torch.long)
        eps = diff.denoise_fn(x, t, cond=None)                            # [1,2,7,H,W]
        sigma = (1 - abar[idx]).clamp(min=1e-8).sqrt()
        score = -eps / sigma
        beta_c = N * betas[idx]                                           # continuous beta(t) = N * beta_discrete
        return -0.5 * beta_c * (x + score)

    def logp_one(x0: torch.Tensor):
        """Euler-integrate x0->x1 over [t_eps,1], accumulate Hutchinson divergence. -> (logp, x1_sq/D)."""
        x = x0.clone()
        logdet = 0.0
        ts = torch.linspace(t_eps, 1.0, n_ode + 1, device=dev)            # [n_ode+1]
        for k in range(n_ode):
            t0 = ts[k]
            dt = float(ts[k + 1] - ts[k])
            idx = int(round(float(t0) * (N - 1)))
            with torch.enable_grad():
                xr = x.detach().requires_grad_(True)
                f = drift(xr, idx)                                        # [1,2,7,H,W]
                div = x.new_zeros(())
                for j in range(n_hutch):
                    e = eps_h[j:j + 1]                                    # [1,2,7,H,W]
                    gj = torch.autograd.grad((f * e).sum(), xr, retain_graph=(j < n_hutch - 1))[0]
                    div = div + (gj * e).sum()
                div = div / n_hutch
            x = x + dt * f.detach()
            logdet = logdet + dt * float(div.detach())
        x1_sq = float((x ** 2).sum())
        logN = -0.5 * D * float(np.log(2 * np.pi)) - 0.5 * x1_sq
        return logN + logdet, x1_sq / D

    out = []
    for vx in s["vx_list"]:
        x0, djac = interp(float(vx))
        lp, x1norm = logp_one(x0)
        logjac = 0.5 * float(torch.log((djac ** 2).sum().clamp(min=1e-30)))
        out.append({"vx": round(float(vx), 4), "logp": lp, "logjac": logjac,
                    "x1_sq_over_D": round(x1norm, 4)})
        print(f"  vx={float(vx):.3f}  logp={lp:.1f}  logjac={logjac:.2f}  ||x1||^2/D={x1norm:.3f}", flush=True)
    return json.dumps(out)


@app.local_entrypoint()
def main(
    ckpt_local: str = "teacher/out_04_flow_linear_mag0-8/checkpoints/diff_final.pt",
    table_local: str = "vsd/out/flow_grid_out04.npy",
    vmin: float = 0.0, vmax: float = 8.0,
    n_vx: int = 21, per_chunk: int = 3,
    n_ode: int = 128, n_hutch: int = 4, t_eps: float = 1e-3, seed: int = 1234, res: int = 128,
    out: str = "vsd/out/logp_out04.json",
) -> None:
    """Upload ckpt+table to the volume (once), fan out vx over GPUs, save the merged json."""
    import os

    ckpt_name = os.path.basename(ckpt_local)
    table_name = os.path.basename(table_local)

    existing = set()
    try:
        for e in vol.listdir("/"):
            existing.add(os.path.basename(e.path))
    except Exception:
        pass
    to_upload = [(ckpt_local, ckpt_name), (table_local, table_name)]
    to_upload = [(lp, rn) for lp, rn in to_upload if rn not in existing]
    if to_upload:
        print(f"uploading {[rn for _, rn in to_upload]} to volume ...", flush=True)
        with vol.batch_upload() as batch:
            for lp, rn in to_upload:
                batch.put_file(lp, rn)
        print("upload done", flush=True)
    else:
        print(f"volume already has {ckpt_name} + {table_name}", flush=True)

    vxs = [round(vmin + (vmax - vmin) * i / (n_vx - 1), 4) for i in range(n_vx)]
    chunks = [vxs[i:i + per_chunk] for i in range(0, len(vxs), per_chunk)]
    specs = [json.dumps({"vx_list": c, "ckpt": ckpt_name, "table": table_name,
                         "vmin": vmin, "vmax": vmax, "n_ode": n_ode, "n_hutch": n_hutch,
                         "t_eps": t_eps, "seed": seed, "res": res}) for c in chunks]
    print(f"fanning out {len(vxs)} vx over {len(chunks)} containers "
          f"(n_ode={n_ode}, n_hutch={n_hutch}) ...", flush=True)

    merged = []
    for r in logp_chunk.map(specs):
        merged.extend(json.loads(r))
    merged.sort(key=lambda d: d["vx"])

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump({"meta": {"ckpt": ckpt_name, "table": table_name, "vmin": vmin, "vmax": vmax,
                        "n_ode": n_ode, "n_hutch": n_hutch, "t_eps": t_eps, "seed": seed},
               "points": merged}, open(out, "w"), indent=2)
    bad = [p for p in merged if not (0.5 < p["x1_sq_over_D"] < 2.0)]
    print(f"\nsaved {out}  ({len(merged)} vx)")
    print(f"calibration ||x1||^2/D in [{min(p['x1_sq_over_D'] for p in merged):.2f}, "
          f"{max(p['x1_sq_over_D'] for p in merged):.2f}]  (want ~1.0; {len(bad)} out of [0.5,2])")
