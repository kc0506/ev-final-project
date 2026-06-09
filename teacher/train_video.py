"""
Phase 2: Video-diffusion OVERFIT test on the telephone dataset.

Unconditional video diffusion (no E) over the 256 rendered 8-frame clips.
Goal: confirm a small 3D-UNet can fit/overfit the telephone dynamics, i.e.
that the rendered video distribution is learnable as a generative model.
This is the object that would later become the DMD teacher.

Uses lucidrains' video-diffusion-pytorch (Unet3D + GaussianDiffusion) for the
vetted 3D-UNet + diffusion internals, wrapped in our own training loop so the
record-keeping matches phase 1.

ALL artifacts persisted under OUT_DIR:
    out_video/
      config.json
      metrics.csv                 step,epoch,loss,peak_mem_gb
      loss_curve.png
      samples/sample_eXXXX_grid.png   (rows=clips, cols=frames)
      samples/sample_eXXXX_v0.gif     animated first clip
      checkpoints/diff_eXXXX.pt
      train.log

Run:
    CUDA_VISIBLE_DEVICES=1 python train_video.py
"""
import os
import json
import csv
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import imageio.v2 as imageio

from video_diffusion_pytorch import Unet3D, GaussianDiffusion

# ----------------------------- config -----------------------------
CACHE = "/tmp2/b10401006/ev-project/generative-phys/teacher/cache/video_128.npy"
OUT_DIR = "/tmp2/b10401006/ev-project/generative-phys/teacher/out_video"
RES = 128
FRAMES = 8
DIM = 64
DIM_MULTS = (1, 2, 4, 8)
TIMESTEPS = 1000
LOSS_TYPE = "l2"
BATCH = 1               # b1 fp32 = 8.29GB peak (<12GB); accumulate for eff. batch
GRAD_ACCUM = 4          # effective batch = BATCH * GRAD_ACCUM = 4
LR = 1e-4
EPOCHS = 150            # overfit on 256 clips; ~52s/epoch -> ~2.2h train
SAMPLE_EVERY = 50       # epochs (sampling is ~235s/round, keep sparse)
N_SAMPLE = 4
SEED = 0
DEVICE = "cuda"


class VideoCache(Dataset):
    def __init__(self, path):
        self.arr = np.load(path)  # (N,T,H,W,3) uint8
        assert self.arr.ndim == 5
    def __len__(self):
        return self.arr.shape[0]
    def __getitem__(self, i):
        v = torch.from_numpy(self.arr[i].copy()).float() / 255.0  # (T,H,W,3) [0,1]
        return v.permute(3, 0, 1, 2)  # (C,T,H,W)  -> lib expects (B,C,F,H,W)


def build():
    unet = Unet3D(dim=DIM, dim_mults=DIM_MULTS)
    diff = GaussianDiffusion(unet, image_size=RES, num_frames=FRAMES,
                             timesteps=TIMESTEPS, loss_type=LOSS_TYPE)
    return diff


def save_grid(video, path):
    # video: (n,C,T,H,W) float [0,1] -> rows=clips, cols=frames
    v = (video.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    n, C, T, H, W = v.shape
    canvas = np.zeros((n * H, T * W, 3), dtype=np.uint8)
    for i in range(n):
        for t in range(T):
            canvas[i * H:(i + 1) * H, t * W:(t + 1) * W] = v[i, :, t].transpose(1, 2, 0)
    Image.fromarray(canvas).save(path)


def save_gif(video0, path, fps=7):
    v = (video0.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()  # (C,T,H,W)
    frames = [v[:, t].transpose(1, 2, 0) for t in range(v.shape[1])]
    imageio.mimsave(path, frames, fps=fps)


def plot_curve(metrics_csv, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] no matplotlib: {e}")
        return
    steps, losses = [], []
    with open(metrics_csv) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"])); losses.append(float(row["loss"]))
    if not steps:
        return
    k = max(1, len(losses) // 200)
    ma = np.convolve(losses, np.ones(k) / k, mode="valid")
    plt.figure(figsize=(8, 4))
    plt.plot(steps, losses, alpha=0.25, label="raw")
    plt.plot(steps[len(steps) - len(ma):], ma, label=f"MA({k})")
    plt.xlabel("step"); plt.ylabel(f"{LOSS_TYPE} loss"); plt.yscale("log")
    plt.title("Phase 2 video diffusion overfit"); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=120); plt.close()
    print(f"saved curve -> {out_png}")


@torch.no_grad()
def sample(diff, n):
    diff.eval()
    vid = diff.sample(batch_size=n)  # (n,C,T,H,W) in [0,1]
    diff.train()
    return vid


def main():
    torch.manual_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "samples"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "checkpoints"), exist_ok=True)

    ds = VideoCache(CACHE)
    print(f"dataset: {len(ds)} clips of {FRAMES}x{RES}x{RES}")
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=4,
                    drop_last=True, pin_memory=True)

    diff = build().to(DEVICE)
    n_params = sum(p.numel() for p in diff.parameters())
    print(f"3D-UNet+diffusion params: {n_params/1e6:.1f}M")
    opt = torch.optim.AdamW(diff.parameters(), lr=LR)

    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump({"cache": CACHE, "res": RES, "frames": FRAMES, "dim": DIM,
                   "dim_mults": list(DIM_MULTS), "timesteps": TIMESTEPS,
                   "loss_type": LOSS_TYPE, "batch": BATCH, "grad_accum": GRAD_ACCUM,
                   "eff_batch": BATCH * GRAD_ACCUM, "lr": LR,
                   "epochs": EPOCHS, "n_clips": len(ds),
                   "params_M": round(n_params / 1e6, 2), "seed": SEED}, f, indent=2)

    metrics_csv = os.path.join(OUT_DIR, "metrics.csv")
    mf = open(metrics_csv, "w", newline="")
    writer = csv.writer(mf); writer.writerow(["step", "epoch", "loss", "peak_mem_gb"])

    step = 0
    last_loss = float("nan")
    for epoch in range(EPOCHS):
        opt.zero_grad(set_to_none=True)
        for i, x in enumerate(dl):
            x = x.to(DEVICE, non_blocking=True)  # (B,C,T,H,W) in [0,1]
            loss = diff(x) / GRAD_ACCUM
            loss.backward()
            if (i + 1) % GRAD_ACCUM == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
            step += 1
            last_loss = loss.item() * GRAD_ACCUM  # report unscaled
            peak = torch.cuda.max_memory_allocated() / 1e9
            writer.writerow([step, epoch, f"{last_loss:.6f}", f"{peak:.3f}"])
        if epoch % 5 == 0:
            mf.flush()
            print(f"epoch {epoch:4d}  loss {last_loss:.4f}  peakmem {peak:.2f}GB", flush=True)
        if epoch % SAMPLE_EVERY == 0 and epoch > 0:
            vid = sample(diff, N_SAMPLE)
            save_grid(vid, os.path.join(OUT_DIR, "samples", f"sample_e{epoch:04d}_grid.png"))
            save_gif(vid[0], os.path.join(OUT_DIR, "samples", f"sample_e{epoch:04d}_v0.gif"))
            torch.save({"diffusion": diff.state_dict(), "epoch": epoch},
                       os.path.join(OUT_DIR, "checkpoints", f"diff_e{epoch:04d}.pt"))

    vid = sample(diff, N_SAMPLE)
    save_grid(vid, os.path.join(OUT_DIR, "samples", "sample_final_grid.png"))
    save_gif(vid[0], os.path.join(OUT_DIR, "samples", "sample_final_v0.gif"))
    torch.save({"diffusion": diff.state_dict(), "epoch": EPOCHS},
               os.path.join(OUT_DIR, "checkpoints", "diff_final.pt"))
    mf.close()
    plot_curve(metrics_csv, os.path.join(OUT_DIR, "loss_curve.png"))
    print("done")


if __name__ == "__main__":
    main()
