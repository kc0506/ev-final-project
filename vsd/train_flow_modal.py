"""Train a flow-diffusion teacher on Modal (fresh OR resume). Same Unet3D+GaussianDiffusion
+ {diffusion,optimizer,epoch,step} ckpt as teacher/train_flow.py. Sampling is OFF (slow,
irrelevant), so the slim torch+video_diffusion image suffices. Checkpoints commit to the
volume each ckpt_every epochs (resumable; a dying/timed-out container loses nothing).

  # measure s/epoch first (user discipline: estimate timeout before the long run)
  modal run vsd/train_flow_modal.py --pack-local <pack> --pack-remote desc_pack.npy \
      --out-remote train_desc --measure-epochs 5
  # then the full run
  modal run vsd/train_flow_modal.py --pack-local <pack> --pack-remote desc_pack.npy \
      --out-remote train_desc --epochs 250 --download-dir teacher/out_05_desc_modal
  # fetch:  modal volume get physgen-logp train_desc <download_dir> --force
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
app = modal.App("physgen-train", image=image)

RESUME_REMOTE = "latest_in.pt"          # only used when resume=True


@app.function(gpu="L40S", volumes={"/data": vol}, timeout=18000)
def train(pack_name: str, out_remote: str, epochs: int, resume: bool,
          lr: float, grad_accum: int, dim: int, res: int,
          timesteps: int, snapshot_every: int, ckpt_every: int, seed: int) -> str:
    """Train fresh (resume=False) or resume from /data/RESUME_REMOTE; ckpts -> /data/{out_remote}."""
    import csv
    import os
    import time

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from video_diffusion_pytorch import GaussianDiffusion, Unet3D

    dev = "cuda"
    torch.manual_seed(seed)
    outdir = f"/data/{out_remote}"
    os.makedirs(outdir, exist_ok=True)

    class FlowCache(Dataset):
        def __init__(self, path: str) -> None:
            self.arr = np.load(path)                              # (N,F,H,W,2) float32 [0,1]
            assert self.arr.ndim == 5 and self.arr.shape[-1] == 2, self.arr.shape

        def __len__(self) -> int:
            return self.arr.shape[0]

        def __getitem__(self, i: int) -> torch.Tensor:
            v = torch.from_numpy(self.arr[i].copy()).float()     # (F,H,W,2)
            return v.permute(3, 0, 1, 2)                          # (2,F,H,W)

    ds = FlowCache(f"/data/{pack_name}")
    frames = int(ds.arr.shape[1])
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
    print(f"flow dataset: {len(ds)} clips of {frames}x{res}x{res}x2", flush=True)

    unet = Unet3D(dim=dim, dim_mults=(1, 2, 4, 8), channels=2)
    diff = GaussianDiffusion(unet, image_size=res, num_frames=frames, channels=2,
                             timesteps=timesteps, loss_type="l2").to(dev)
    opt = torch.optim.AdamW(diff.parameters(), lr=lr)

    start_epoch, step = 0, 0
    if resume:
        ck = torch.load(f"/data/{out_remote}/latest.pt", map_location=dev)  # continue THIS run
        diff.load_state_dict(ck["diffusion"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        step = int(ck.get("step", 0))
        print(f"[resume] epoch {start_epoch}, step {step} -> {epochs}", flush=True)
    else:
        print(f"[fresh] training 0 -> {epochs}", flush=True)

    metrics_path = os.path.join(outdir, "metrics_modal.csv")
    new_csv = not os.path.exists(metrics_path)
    mf = open(metrics_path, "a", newline="", buffering=1)
    writer = csv.writer(mf)
    if new_csv:
        writer.writerow(["step", "epoch", "loss", "peak_mem_gb"])

    def save_ckpt(name: str, epoch: int) -> None:
        path = os.path.join(outdir, name)
        tmp = path + ".tmp"
        torch.save({"diffusion": diff.state_dict(), "optimizer": opt.state_dict(),
                    "epoch": epoch, "step": step}, tmp)
        os.replace(tmp, path)
        vol.commit()                                             # persist for download/resume

    last_loss, peak, t0, n_done = float("nan"), 0.0, time.time(), 0
    for epoch in range(start_epoch, epochs):
        opt.zero_grad(set_to_none=True)
        for i, x in enumerate(dl):
            x = x.to(dev, non_blocking=True)                     # (1,2,F,H,W)
            loss = diff(x) / grad_accum
            loss.backward()
            if (i + 1) % grad_accum == 0:
                opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            last_loss = loss.item() * grad_accum
            peak = torch.cuda.max_memory_allocated() / 1e9
            writer.writerow([step, epoch, f"{last_loss:.6f}", f"{peak:.3f}"])
        n_done += 1
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d}/{epochs}  loss {last_loss:.4f}  peakmem {peak:.2f}GB "
                  f"({(time.time()-t0)/max(n_done,1):.1f}s/ep)", flush=True)
        if epoch % ckpt_every == 0:
            save_ckpt("latest.pt", epoch)
        if snapshot_every > 0 and epoch % snapshot_every == 0 and epoch > 0:
            save_ckpt(f"diff_e{epoch:04d}.pt", epoch)

    save_ckpt("latest.pt", epochs - 1)
    save_ckpt("diff_final.pt", epochs - 1)
    mf.close()
    vol.commit()
    return json.dumps({"start_epoch": start_epoch, "end_epoch": epochs - 1,
                       "final_loss": round(last_loss, 6), "step": step,
                       "s_per_epoch": round((time.time() - t0) / max(n_done, 1), 1)})


@app.local_entrypoint()
def main(
    pack_local: str = "outputs/gen_flow_aligned/05_desc_n128_mag0-8/flow_pack_128_t8.npy",
    pack_remote: str = "desc_pack.npy", out_remote: str = "train_desc",
    download_dir: str = "teacher/out_05_desc_modal",
    epochs: int = 250, resume: bool = False, resume_local: str = "",
    measure_epochs: int = 0,
    lr: float = 1e-4, grad_accum: int = 4, dim: int = 64, res: int = 128,
    timesteps: int = 1000, snapshot_every: int = 25, ckpt_every: int = 10, seed: int = 0,
) -> None:
    """Upload the pack (once), optionally measure s/epoch, then train; ckpts land on the volume."""
    import os

    existing = set()
    try:
        for e in vol.listdir("/"):
            existing.add(os.path.basename(e.path))
    except Exception:
        pass
    up = [(pack_local, pack_remote)]
    if resume and resume_local:
        up.append((resume_local, RESUME_REMOTE))
    up = [(lp, rn) for lp, rn in up if rn not in existing]
    if up:
        print(f"uploading {[rn for _, rn in up]} ...", flush=True)
        with vol.batch_upload() as batch:
            for lp, rn in up:
                batch.put_file(lp, rn)
        print("upload done", flush=True)

    if measure_epochs > 0:
        print(f"MEASURE: {measure_epochs} epochs to estimate s/epoch ...", flush=True)
        m = json.loads(train.remote(pack_remote, out_remote + "_measure", measure_epochs, resume,
                                    lr, grad_accum, dim, res, timesteps, 0, 9999, seed))
        spe = m["s_per_epoch"]
        print(f"\n=> {spe:.1f} s/epoch ; {epochs} epochs ~= {spe*epochs/60:.0f} min "
              f"(timeout cap is 18000s={18000/60:.0f}min)")
        return

    print(f"training {epochs} epochs (resume={resume}) -> /data/{out_remote} ...", flush=True)
    summary = train.remote(pack_remote, out_remote, epochs, resume, lr, grad_accum, dim, res,
                           timesteps, snapshot_every, ckpt_every, seed)
    print("TRAIN SUMMARY:", summary, flush=True)
    print(f"fetch:  modal volume get physgen-logp {out_remote} {download_dir} --force")
