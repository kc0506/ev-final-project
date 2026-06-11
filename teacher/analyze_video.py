"""
Quantitative motion check for the phase-2 video model.

Generates clips from diff_final.pt and compares their temporal motion against
real cache clips:
  - mean |frame_{t+1}-frame_t|  (per-step motion energy)
  - |frame_7 - frame_0|         (total drift over the clip)
Also dumps side-by-side grids (rows=clips, cols=8 frames) for gen vs real,
and saves the raw generated tensor for the record.

Output under out_video/analysis/.
"""
import os
import json
import numpy as np
import torch
from PIL import Image

from train_video import build, CACHE, RES, FRAMES

CKPT = "/tmp2/b10401006/ev-project/generative-phys/teacher/out_video/checkpoints/diff_final.pt"
OUT = "/tmp2/b10401006/ev-project/generative-phys/teacher/out_video/analysis"
N = 4
SEED = 1


def grid(v, path):  # v: (n,C,T,H,W) float[0,1]
    a = (np.clip(v, 0, 1) * 255).round().astype(np.uint8)
    n, C, T, H, W = a.shape
    canvas = np.zeros((n * H, T * W, 3), np.uint8)
    for i in range(n):
        for t in range(T):
            canvas[i*H:(i+1)*H, t*W:(t+1)*W] = a[i, :, t].transpose(1, 2, 0)
    Image.fromarray(canvas).save(path)


def motion_stats(v):  # v: (n,C,T,H,W) float[0,1]
    d = np.abs(np.diff(v, axis=2)).mean(axis=(1, 3, 4))     # (n, T-1) per-step
    drift = np.abs(v[:, :, -1] - v[:, :, 0]).mean(axis=(1, 2, 3))  # (n,)
    return d.mean(), d.mean(0), drift.mean()


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(SEED)
    diff = build().cuda()
    sd = torch.load(CKPT, map_location="cuda")
    diff.load_state_dict(sd["diffusion"])
    diff.eval()
    with torch.no_grad():
        gen = diff.sample(batch_size=N).cpu().numpy()  # (N,C,T,H,W) [0,1]
    np.save(os.path.join(OUT, "gen_final.npy"), gen.astype(np.float32))

    cache = np.load(CACHE).astype(np.float32) / 255.0  # (Nall,T,H,W,3)
    real = cache[:N].transpose(0, 4, 1, 2, 3)  # (N,C,T,H,W)

    grid(gen, os.path.join(OUT, "gen_grid.png"))
    grid(real, os.path.join(OUT, "real_grid.png"))

    gm, gper, gd = motion_stats(gen)
    rm, rper, rd = motion_stats(real)
    rep = {
        "gen_per_step_motion_mean": float(gm),
        "real_per_step_motion_mean": float(rm),
        "gen_total_drift_mean": float(gd),
        "real_total_drift_mean": float(rd),
        "gen_per_step_curve": [round(float(x), 5) for x in gper],
        "real_per_step_curve": [round(float(x), 5) for x in rper],
        "note": "motion in [0,1] pixel units; per-step = mean|f_{t+1}-f_t|",
    }
    with open(os.path.join(OUT, "motion_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
