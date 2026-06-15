"""DIRECT learned-marginal probe: ancestral-sample the out_04 flow teacher on Modal
(fan-out over containers to beat the 1000-step serial cost) and measure the flow-magnitude
distribution of the GENERATED clips. This is the cleanest readout of what the teacher
actually learned -- free of the ODE (ambient-density) and VSD (aux-score) method artifacts
that pull in opposite directions. If the samples' magnitude is ramp-like (mass toward high
|flow|) and positive-x, the teacher learned the ramp; if they pile high/collapse, the
high-vx bias lives in the teacher itself.

Loads the epoch-299 ckpt already on the volume (/data/train04_ckpts/diff_final.pt).

  modal run vsd/sample_probe_modal.py --n-containers 6 --batch 4
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
app = modal.App("physgen-sample", image=image)

CKPT = "train04_ckpts/diff_final.pt"   # epoch 299 (on the volume from the train run)


@app.function(gpu="L40S", volumes={"/data": vol}, timeout=2400)
def sample_chunk(spec_json: str) -> str:
    """Ancestral-sample `batch` clips and return per-clip flow stats (centred [0,1] units)."""
    import numpy as np
    import torch
    from video_diffusion_pytorch import GaussianDiffusion, Unet3D

    s = json.loads(spec_json)
    dev = "cuda"
    res, batch, thr = int(s["res"]), int(s["batch"]), float(s["thr"])
    torch.manual_seed(int(s["seed"]))

    unet = Unet3D(dim=64, dim_mults=(1, 2, 4, 8), channels=2)
    diff = GaussianDiffusion(unet, image_size=res, num_frames=7, channels=2,
                             timesteps=1000, loss_type="l2").to(dev)
    diff.load_state_dict(torch.load(f"/data/{s['ckpt']}", map_location=dev)["diffusion"])
    diff.eval()

    with torch.no_grad():
        vid = diff.sample(batch_size=batch)                  # [B,2,7,H,W] in [0,1]
    v = vid.clamp(0, 1).cpu().numpy().astype(np.float32)     # [B,2,7,H,W]

    # DECODE vx by nearest-neighbour against the physics flow table (the clean vx readout:
    # scalar flow stats don't encode vx because optical flow has mixed-sign spatial structure).
    table = np.load("/data/flow_grid_out04.npy").astype(np.float32)  # [G,2,7,H,W] in [0,1]
    G = table.shape[0]
    vx_grid = np.linspace(s["vmin"], s["vmax"], G)            # [G]
    tflat = table.reshape(G, -1)                             # [G, D]

    out = []
    for b in range(batch):
        d = ((tflat - v[b].reshape(-1)[None]) ** 2).mean(1)  # [G] L2 to each table vx
        gi = int(d.argmin())
        last = np.sqrt((v[b, 0, -1] - 0.5) ** 2 + (v[b, 1, -1] - 0.5) ** 2)  # [H,W] mag
        out.append({
            "decoded_vx": round(float(vx_grid[gi]), 4),      # nearest table vx = readout
            "fit_resid": round(float(d[gi]), 6),             # how well it matched the manifold
            "mag_mean": round(float(last[last > thr].mean()) if (last > thr).any() else 0.0, 4),
            "pct_moving": round(float((last > thr).mean()), 4),
        })
    print(f"sampled {batch}: " + ", ".join(f"vx={o['decoded_vx']:.2f}(r={o['fit_resid']:.4f})"
                                           for o in out), flush=True)
    return json.dumps(out)


@app.local_entrypoint()
def main(ckpt: str = CKPT, n_containers: int = 8, batch: int = 4, res: int = 128, thr: float = 0.02,
         vmin: float = 0.0, vmax: float = 8.0,
         out: str = "vsd/out/sample_probe_out04.json") -> None:
    """Fan out sampling of `ckpt` (a path on the volume), decode each clip's vx via the table."""
    import os

    specs = [json.dumps({"ckpt": ckpt, "batch": batch, "res": res, "thr": thr,
                         "vmin": vmin, "vmax": vmax, "seed": 1000 + i}) for i in range(n_containers)]
    print(f"sampling {n_containers * batch} clips over {n_containers} L40S containers ...", flush=True)
    clips = []
    for r in sample_chunk.map(specs):
        clips.extend(json.loads(r))

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump({"ckpt": ckpt, "n": len(clips), "vmin": vmin, "vmax": vmax, "clips": clips},
              open(out, "w"), indent=2)
    vx = sorted(c["decoded_vx"] for c in clips)
    mean = sum(vx) / len(vx)
    print(f"\nsaved {out}  ({len(clips)} clips)")
    print(f"decoded_vx: min={vx[0]:.2f} max={vx[-1]:.2f} mean={mean:.2f}  (ramp E[vx]=5.33)")
    print(f"  quartiles: {vx[len(vx)//4]:.2f} / {vx[len(vx)//2]:.2f} / {vx[3*len(vx)//4]:.2f}")
