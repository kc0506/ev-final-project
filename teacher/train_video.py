"""Video-diffusion training on a dataset_gen video cache (genphys-diff env).

Unconditional 3D-UNet + GaussianDiffusion (lucidrains video-diffusion-pytorch):
confirm the rendered video distribution is learnable as a generative model -- the
object that later becomes the DMD teacher. Parametrised (cache/out/res/frames/...)
so it trains ANY dataset_gen cache, not just the hardcoded telephone one.

LONG-RUN SAFETY (shared-box daily GPU quota kills all your GPU procs on exhaustion):
  - launch floor: abort if quota < --quota_floor_hours (resume makes this lenient).
  - in-loop guard: every --ckpt_every epochs, if quota < --quota_stop_secs, save a
    checkpoint and exit GRACEFULLY (0) instead of being hard-killed mid-epoch.
  - high-freq checkpoint + --resume auto: re-launch continues from latest.pt
    (model+optimizer+epoch), so a stop/kill costs at most --ckpt_every epochs.

  CUDA_VISIBLE_DEVICES auto-picked (freest GPU) unless preset.

  python train_video.py --pack ../outputs/dataset_gen/01_tel_axisx_rest_T16/video_pack_128.npy \
      --out out_tel_axisx_T16 --res 128 --frames 16 --epochs 150 --resume auto

Artifacts under --out: config.json (incl. `dataset` provenance block), dataset.json
  (which dataset_gen run this pack came from), metrics.csv, loss_curve.png,
  samples/sample_e*.{png,gif}, checkpoints/{latest.pt,diff_e*.pt}, train.log
"""
import argparse
import csv
import json
import os
import signal

import gpu_guard  # self-contained quota guard (same dir)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True,
                    help="dataset video pack .npy (N,T,H,W,3) uint8; see pack_dataset.py")
    ap.add_argument("--out", required=True, help="output run dir")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--frames", type=int, default=16, help="must match cache T")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--dim_mults", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--loss_type", default="l2")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--sample_every", type=int, default=50, help="epochs between sample grids")
    ap.add_argument("--ckpt_every", type=int, default=10, help="epochs between latest.pt saves + quota checks")
    ap.add_argument("--resume", default="auto", help="'auto' (latest.pt in --out), a path, or 'no'")
    ap.add_argument("--quota_floor_hours", type=float, default=1.0, help="abort at launch below this")
    ap.add_argument("--quota_stop_secs", type=int, default=30000,
                    help="checkpoint+exit when quota below this; keep ABOVE the 8h "
                         "(28800s) watchdog floor so the trainer self-stops gracefully "
                         "BEFORE the watchdog hard-kills it")
    ap.add_argument("--max_my_gpus", type=int, default=1,
                    help="checkpoint+exit if I occupy more than this many GPUs (concurrent-job spread)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    # quota floor + freest-GPU pick BEFORE torch touches CUDA
    gpu_guard.pick_free_gpu(min_quota_hours=args.quota_floor_hours)

    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import imageio.v2 as imageio
    from video_diffusion_pytorch import Unet3D, GaussianDiffusion
    from tqdm import tqdm

    DEVICE = "cuda"
    torch.manual_seed(args.seed)
    os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "checkpoints"), exist_ok=True)

    class VideoCache(Dataset):
        def __init__(self, path):
            self.arr = np.load(path)  # (N,T,H,W,3) uint8
            assert self.arr.ndim == 5, self.arr.shape
        def __len__(self):
            return self.arr.shape[0]
        def __getitem__(self, i):
            v = torch.from_numpy(self.arr[i].copy()).float() / 255.0  # (T,H,W,3)
            return v.permute(3, 0, 1, 2)  # (C,T,H,W)

    ds = VideoCache(args.pack)
    assert ds.arr.shape[1] == args.frames, \
        f"--frames {args.frames} != pack T {ds.arr.shape[1]}"
    print(f"dataset: {len(ds)} clips of {args.frames}x{args.res}x{args.res}")

    # provenance: WHICH dataset_gen run produced this pack (from <pack>.meta.json
    # -> its data_dir -> that run's manifest/config). Written to dataset.json and
    # embedded in config.json, so a train run self-documents its source dataset.
    prov = {"pack": os.path.abspath(args.pack)}
    meta_p = args.pack + ".meta.json"
    if os.path.exists(meta_p):
        meta = json.load(open(meta_p))
        prov.update({k: meta.get(k) for k in ("data_dir", "n", "t", "res")})
        dd = meta.get("data_dir")
        man_p = os.path.join(dd, "manifest.json") if dd else None
        if man_p and os.path.exists(man_p):
            man = json.load(open(man_p))
            prov["dataset_description"] = man.get("description")
            prov["dataset_summary"] = man.get("summary")
            prov["dataset_p_star"] = man.get("p_star")
        cfg_p = os.path.join(dd, "config.json") if dd else None
        if cfg_p and os.path.exists(cfg_p):
            prov["dataset_provenance"] = json.load(open(cfg_p)).get("_provenance")
    else:
        print(f"[provenance] WARNING: no {meta_p}; dataset source unknown")
    with open(os.path.join(args.out, "dataset.json"), "w") as f:
        json.dump(prov, f, indent=2)
    print(f"[provenance] trained on: {prov.get('dataset_description') or prov.get('data_dir')}")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4,
                    drop_last=True, pin_memory=True)

    unet = Unet3D(dim=args.dim, dim_mults=tuple(args.dim_mults))
    diff = GaussianDiffusion(unet, image_size=args.res, num_frames=args.frames,
                             timesteps=args.timesteps, loss_type=args.loss_type).to(DEVICE)
    n_params = sum(p.numel() for p in diff.parameters())
    print(f"3D-UNet+diffusion params: {n_params/1e6:.1f}M")
    opt = torch.optim.AdamW(diff.parameters(), lr=args.lr)

    # ---- resume ----
    start_epoch, step = 0, 0
    latest = os.path.join(args.out, "checkpoints", "latest.pt")
    resume_path = (latest if args.resume == "auto" and os.path.exists(latest)
                   else (args.resume if args.resume not in ("auto", "no") else None))
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=DEVICE)
        diff.load_state_dict(ckpt["diffusion"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        step = int(ckpt.get("step", 0))
        print(f"[resume] from {resume_path} -> epoch {start_epoch}, step {step}")

    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({**vars(args), "n_clips": len(ds),
                   "params_M": round(n_params/1e6, 2), "dataset": prov}, f, indent=2)

    metrics_csv = os.path.join(args.out, "metrics.csv")
    new_csv = not os.path.exists(metrics_csv)
    mf = open(metrics_csv, "a", newline="", buffering=1)  # line-buffered -> tail -f is live
    writer = csv.writer(mf)
    if new_csv:
        writer.writerow(["step", "epoch", "loss", "peak_mem_gb"])

    def save_ckpt(name, epoch):
        # atomic: write a temp file then os.replace, so a kill mid-write can never
        # leave a truncated/corrupt checkpoint (latest.pt is overwritten often).
        path = os.path.join(args.out, "checkpoints", name)
        tmp = path + ".tmp"
        torch.save({"diffusion": diff.state_dict(), "optimizer": opt.state_dict(),
                    "epoch": epoch, "step": step}, tmp)
        os.replace(tmp, path)

    @torch.no_grad()
    def sample(n):
        diff.eval()
        vid = diff.sample(batch_size=n)  # (n,C,T,H,W) [0,1]
        diff.train()
        return vid

    def save_grid(video, path):
        v = (video.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        n, C, T, H, W = v.shape
        canvas = np.zeros((n * H, T * W, 3), dtype=np.uint8)
        for i in range(n):
            for t in range(T):
                canvas[i*H:(i+1)*H, t*W:(t+1)*W] = v[i, :, t].transpose(1, 2, 0)
        Image.fromarray(canvas).save(path)

    def save_gif(video0, path, fps=7):
        v = (video0.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()  # (C,T,H,W)
        imageio.mimsave(path, [v[:, t].transpose(1, 2, 0) for t in range(v.shape[1])], fps=fps)

    def save_grid_gif(video, path, cols=2, fps=7):   # (n,C,T,H,W) -> animated grid of ALL samples
        v = (video.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        n, C, T, H, W = v.shape
        rows = (n + cols - 1) // cols
        frames = []
        for t in range(T):
            cv = np.zeros((rows * H, cols * W, 3), np.uint8)
            for k in range(n):
                r, c = divmod(k, cols)
                cv[r*H:(r+1)*H, c*W:(c+1)*W] = v[k, :, t].transpose(1, 2, 0)
            frames.append(cv)
        imageio.mimsave(path, frames, fps=fps)

    # Route SIGTERM (`kill`) through the same path as Ctrl-C (SIGINT) so ANY stop
    # checkpoints the current epoch instead of dying mid-flight. SIGKILL stays
    # uncatchable -- that's what the atomic save_ckpt + periodic ckpt protect.
    def _graceful_stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _graceful_stop)

    last_loss, peak = float("nan"), 0.0
    epoch = start_epoch
    interrupted = False
    try:
        for epoch in range(start_epoch, args.epochs):
            opt.zero_grad(set_to_none=True)
            # disable=None -> tqdm renders the bar on a real TTY but stays silent
            # when stdout/stderr is redirected to a file (no \r spam in logs).
            pbar = tqdm(dl, desc=f"epoch {epoch}/{args.epochs}", leave=False,
                        dynamic_ncols=True, disable=None)
            for i, x in enumerate(pbar):
                x = x.to(DEVICE, non_blocking=True)  # (B,C,T,H,W)
                loss = diff(x) / args.grad_accum
                loss.backward()
                if (i + 1) % args.grad_accum == 0:
                    opt.step(); opt.zero_grad(set_to_none=True)
                step += 1
                last_loss = loss.item() * args.grad_accum
                peak = torch.cuda.max_memory_allocated() / 1e9
                writer.writerow([step, epoch, f"{last_loss:.6f}", f"{peak:.3f}"])
                pbar.set_postfix(loss=f"{last_loss:.4f}", mem=f"{peak:.1f}G")
            # one clean summary line per epoch -- this is what lands in the log file
            print(f"epoch {epoch:4d}/{args.epochs}  loss {last_loss:.4f}  "
                  f"peakmem {peak:.2f}GB", flush=True)

            # meow2 shared-box guard (no-op locally: no ws-status, single GPU):
            # spread/quota breach -> route into the graceful-stop path.
            ok, why = gpu_guard.status_ok(args.quota_stop_secs, max_my_gpus=args.max_my_gpus)
            if not ok:
                print(f"[gpu] STOP ({why}) -> graceful exit.", flush=True)
                raise KeyboardInterrupt
            if epoch % args.ckpt_every == 0:
                save_ckpt("latest.pt", epoch)

            if epoch % args.sample_every == 0 and epoch > 0:
                vid = sample(min(4, args.batch * 4) or 4)
                save_grid(vid, os.path.join(args.out, "samples", f"sample_e{epoch:04d}_grid.png"))
                save_grid_gif(vid, os.path.join(args.out, "samples", f"sample_e{epoch:04d}_grid.gif"))
                save_ckpt(f"diff_e{epoch:04d}.pt", epoch)

        save_ckpt("latest.pt", args.epochs - 1)
        save_ckpt("diff_final.pt", args.epochs - 1)
        vid = sample(4)
        save_grid(vid, os.path.join(args.out, "samples", "sample_final_grid.png"))
        save_grid_gif(vid, os.path.join(args.out, "samples", "sample_final_grid.gif"))
    except KeyboardInterrupt:
        interrupted = True
        save_ckpt("latest.pt", epoch)
        print(f"\n[stop] interrupted @ epoch {epoch} -> saved latest.pt "
              f"(resume with --resume auto).", flush=True)
    finally:
        mf.close()
    if interrupted:
        return

    # loss curve
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        steps, losses = [], []
        for row in csv.DictReader(open(metrics_csv)):
            steps.append(int(row["step"])); losses.append(float(row["loss"]))
        if steps:
            k = max(1, len(losses) // 200)
            ma = np.convolve(losses, np.ones(k)/k, mode="valid")
            plt.figure(figsize=(8, 4))
            plt.plot(steps, losses, alpha=0.25, label="raw")
            plt.plot(steps[len(steps)-len(ma):], ma, label=f"MA({k})")
            plt.xlabel("step"); plt.ylabel(f"{args.loss_type} loss"); plt.yscale("log")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(args.out, "loss_curve.png"), dpi=120); plt.close()
    except Exception as e:
        print(f"[warn] loss curve: {e}")
    print("done")


if __name__ == "__main__":
    main()
