"""
Phase 1: Image-diffusion teacher-machinery run on the telephone dataset.

Unconditional pixel-space DDPM over all rendered frames. No E conditioning
(single frames barely depend on E; this phase validates the env + data +
training loop and gives us a first generative model of the frame manifold).

ALL artifacts are persisted under OUT_DIR for later inspection:
    out_image/
      config.json                 hyperparameters of this run
      metrics.csv                 per-step: step,epoch,loss,lr,peak_mem_gb
      loss_curve.png              training curve (plotted at the end)
      samples/sample_eXXXX.png    inference grid every SAMPLE_EVERY epochs
      checkpoints/unet_eXXXX/     periodic checkpoints (kept, not overwritten)
      checkpoints/unet_final/

Run:
    CUDA_VISIBLE_DEVICES=1 python train_image.py
"""
import glob
import json
import os
import math
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

from diffusers import UNet2DModel, DDPMScheduler

# ----------------------------- config -----------------------------
DATA_GLOB = "/tmp2/b10401006/ev-project/generative-phys/outputs/dataset_telephone_256/sample_*/frames/frame_*.png"
OUT_DIR = "/tmp2/b10401006/ev-project/generative-phys/teacher/out_image"
RES = 128
BATCH = 32
LR = 1e-4
EPOCHS = 200
NUM_TRAIN_TIMESTEPS = 1000
SAMPLE_EVERY = 20       # epochs: save inference grid + checkpoint
N_SAMPLE = 9
DEVICE = "cuda"         # pinned to GPU1 via CUDA_VISIBLE_DEVICES
SEED = 0


class FrameDataset(Dataset):
    def __init__(self, glob_pat, res):
        self.paths = sorted(glob.glob(glob_pat))
        assert len(self.paths) > 0, f"no frames matched: {glob_pat}"
        self.res = res

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB").resize((self.res, self.res), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img)).float() / 127.5 - 1.0  # [-1,1]
        return x.permute(2, 0, 1)  # C,H,W


def build_model(res):
    return UNet2DModel(
        sample_size=res,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )


@torch.no_grad()
def sample(model, scheduler, n, res, device):
    model.eval()
    x = torch.randn(n, 3, res, res, device=device)
    scheduler.set_timesteps(NUM_TRAIN_TIMESTEPS)
    for t in scheduler.timesteps:
        eps = model(x, t).sample
        x = scheduler.step(eps, t, x).prev_sample
    model.train()
    x = ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).cpu().numpy()


def save_grid(arr, path):
    n = arr.shape[0]
    g = int(math.ceil(math.sqrt(n)))
    h, w = arr.shape[1:3]
    canvas = np.zeros((g * h, g * w, 3), dtype=np.uint8)
    for i in range(n):
        r, c = divmod(i, g)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = arr[i]
    Image.fromarray(canvas).save(path)


def plot_curve(metrics_csv, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib unavailable, skipping curve plot: {e}")
        return
    steps, losses = [], []
    with open(metrics_csv) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
    if not steps:
        return
    # light moving average for readability, raw kept in csv
    k = max(1, len(losses) // 200)
    ma = np.convolve(losses, np.ones(k) / k, mode="valid")
    plt.figure(figsize=(8, 4))
    plt.plot(steps, losses, alpha=0.25, label="raw")
    plt.plot(steps[len(steps) - len(ma):], ma, label=f"MA({k})")
    plt.xlabel("step"); plt.ylabel("MSE loss"); plt.yscale("log")
    plt.title("Phase 1 image DDPM training"); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=120); plt.close()
    print(f"saved curve -> {out_png}")


def main():
    torch.manual_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "samples"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "checkpoints"), exist_ok=True)

    ds = FrameDataset(DATA_GLOB, RES)
    print(f"dataset: {len(ds)} frames")
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    model = build_model(RES).to(DEVICE)
    model.enable_gradient_checkpointing()   # keeps peak mem ~6GB at batch 32 (<12GB budget)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet params: {n_params/1e6:.1f}M")
    scheduler = DDPMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda")

    # record config
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump({
            "data_glob": DATA_GLOB, "res": RES, "batch": BATCH, "lr": LR,
            "epochs": EPOCHS, "num_train_timesteps": NUM_TRAIN_TIMESTEPS,
            "n_frames": len(ds), "unet_params_M": round(n_params / 1e6, 2),
            "block_out_channels": [128, 128, 256, 256, 512], "seed": SEED,
        }, f, indent=2)

    metrics_csv = os.path.join(OUT_DIR, "metrics.csv")
    mf = open(metrics_csv, "w", newline="")
    writer = csv.writer(mf)
    writer.writerow(["step", "epoch", "loss", "lr", "peak_mem_gb"])

    step = 0
    for epoch in range(EPOCHS):
        for x in dl:
            x = x.to(DEVICE, non_blocking=True)
            noise = torch.randn_like(x)
            t = torch.randint(0, NUM_TRAIN_TIMESTEPS, (x.size(0),), device=DEVICE)
            xt = scheduler.add_noise(x, noise, t)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(xt, t).sample
                loss = F.mse_loss(pred, noise)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            peak = torch.cuda.max_memory_allocated() / 1e9
            writer.writerow([step, epoch, f"{loss.item():.6f}", LR, f"{peak:.3f}"])
        if epoch % 10 == 0:
            mf.flush()
            print(f"epoch {epoch:4d}  loss {loss.item():.4f}  peakmem {peak:.2f}GB")
        if epoch % SAMPLE_EVERY == 0 and epoch > 0:
            arr = sample(model, scheduler, N_SAMPLE, RES, DEVICE)
            save_grid(arr, os.path.join(OUT_DIR, "samples", f"sample_e{epoch:04d}.png"))
            model.save_pretrained(os.path.join(OUT_DIR, "checkpoints", f"unet_e{epoch:04d}"))

    arr = sample(model, scheduler, N_SAMPLE, RES, DEVICE)
    save_grid(arr, os.path.join(OUT_DIR, "samples", "sample_final.png"))
    model.save_pretrained(os.path.join(OUT_DIR, "checkpoints", "unet_final"))
    mf.close()
    plot_curve(metrics_csv, os.path.join(OUT_DIR, "loss_curve.png"))
    print("done")


if __name__ == "__main__":
    main()
