"""Train a SCREEN-FLOW video-diffusion teacher (2-channel) on a flow pack built
by build_flow_pack.py. Same Unet3D+GaussianDiffusion as train_video.py but the
modelled quantity is the dense GT screen flow (motion), not RGB -- so motion IS
the target and the pixel-L2 objective can no longer ignore it. The flow encodes
the latent v0 directly, which is what the downstream MPM distillation needs.

Mirrors train_video.py's canonical machinery: tqdm, atomic checkpoints, graceful
SIGTERM/Ctrl-C stop, line-buffered metrics, --resume auto.

  python train_flow.py --pack ../outputs/dataset_gen/01_tel_axisx_rest_T16/flow_pack_128_t8.npy \
      --out out_01_flow_T8_local --res 128 --epochs 200 --resume auto
"""
import argparse, csv, json, os, signal
import gpu_guard


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="flow pack .npy (N,F,H,W,2) in [0,1]; see build_flow_pack.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--frames", type=int, default=None, help="flow fields; default = pack dim1")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--dim_mults", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--loss_type", default="l2")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--sample_every", type=int, default=50)
    ap.add_argument("--ckpt_every", type=int, default=10)
    ap.add_argument("--resume", default="auto")
    ap.add_argument("--quota_floor_hours", type=float, default=0.0)
    ap.add_argument("--quota_stop_secs", type=int, default=0)
    ap.add_argument("--max_my_gpus", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
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

    scale = 1.0
    mp = args.pack + ".meta.json"
    if os.path.exists(mp):
        scale = float(json.load(open(mp)).get("scale_px", 1.0))

    class FlowCache(Dataset):
        def __init__(self, path):
            self.arr = np.load(path)            # (N,F,H,W,2) float32 in [0,1]
            assert self.arr.ndim == 5 and self.arr.shape[-1] == 2, self.arr.shape
        def __len__(self):
            return self.arr.shape[0]
        def __getitem__(self, i):
            v = torch.from_numpy(self.arr[i].copy()).float()   # (F,H,W,2)
            return v.permute(3, 0, 1, 2)                        # (2,F,H,W)

    ds = FlowCache(args.pack)
    frames = args.frames or ds.arr.shape[1]
    assert ds.arr.shape[1] == frames, f"--frames {frames} != pack F {ds.arr.shape[1]}"
    print(f"flow dataset: {len(ds)} clips of {frames}x{args.res}x{args.res}x2  scale_px={scale:.3f}")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4,
                    drop_last=True, pin_memory=True)

    unet = Unet3D(dim=args.dim, dim_mults=tuple(args.dim_mults), channels=2)
    diff = GaussianDiffusion(unet, image_size=args.res, num_frames=frames, channels=2,
                             timesteps=args.timesteps, loss_type=args.loss_type).to(DEVICE)
    n_params = sum(p.numel() for p in diff.parameters())
    print(f"3D-UNet+diffusion (2ch) params: {n_params/1e6:.1f}M")
    opt = torch.optim.AdamW(diff.parameters(), lr=args.lr)

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
        json.dump({**vars(args), "n_clips": len(ds), "frames": frames,
                   "params_M": round(n_params/1e6, 2), "scale_px": scale, "kind": "flow"}, f, indent=2)

    metrics_csv = os.path.join(args.out, "metrics.csv")
    new_csv = not os.path.exists(metrics_csv)
    mf = open(metrics_csv, "a", newline="", buffering=1)
    writer = csv.writer(mf)
    if new_csv:
        writer.writerow(["step", "epoch", "loss", "peak_mem_gb"])

    def save_ckpt(name, epoch):
        path = os.path.join(args.out, "checkpoints", name)
        tmp = path + ".tmp"
        torch.save({"diffusion": diff.state_dict(), "optimizer": opt.state_dict(),
                    "epoch": epoch, "step": step}, tmp)
        os.replace(tmp, path)

    @torch.no_grad()
    def sample(n):
        diff.eval(); v = diff.sample(batch_size=n); diff.train()
        return v   # (n,2,F,H,W) in [0,1]

    def flow_to_rgb(f2):                 # (2,H,W) [0,1] -> (H,W,3) viz (B=0.5)
        f2 = np.transpose(f2, (1, 2, 0))                              # -> (H,W,2)
        return np.concatenate([f2, np.full(f2.shape[:2] + (1,), 0.5, np.float32)], -1)

    def save_grid(video, path):          # (n,2,F,H,W)
        v = video.clamp(0, 1).cpu().numpy()
        n, C, T, H, W = v.shape
        canvas = np.zeros((n * H, T * W, 3), np.float32)
        for i in range(n):
            for t in range(T):
                canvas[i*H:(i+1)*H, t*W:(t+1)*W] = flow_to_rgb(v[i, :, t])
        Image.fromarray((canvas * 255).round().astype("uint8")).save(path)

    def save_grid_gif(video, path, cols=2, fps=7):   # (n,2,F,H,W) -> animated grid of ALL samples
        v = video.clamp(0, 1).cpu().numpy()
        n, C, F, H, W = v.shape
        rows = (n + cols - 1) // cols
        frames = []
        for t in range(F):
            cv = np.ones((rows * H, cols * W, 3), np.float32)
            for k in range(n):
                r, c = divmod(k, cols)
                cv[r*H:(r+1)*H, c*W:(c+1)*W] = flow_to_rgb(v[k, :, t])
            frames.append((cv * 255).round().astype("uint8"))
        imageio.mimsave(path, frames, fps=fps)

    def _graceful_stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _graceful_stop)

    last_loss, peak = float("nan"), 0.0
    epoch = start_epoch
    interrupted = False
    try:
        for epoch in range(start_epoch, args.epochs):
            opt.zero_grad(set_to_none=True)
            pbar = tqdm(dl, desc=f"epoch {epoch}/{args.epochs}", leave=False,
                        dynamic_ncols=True, disable=None)
            for i, x in enumerate(pbar):
                x = x.to(DEVICE, non_blocking=True)            # (B,2,F,H,W)
                loss = diff(x) / args.grad_accum
                loss.backward()
                if (i + 1) % args.grad_accum == 0:
                    opt.step(); opt.zero_grad(set_to_none=True)
                step += 1
                last_loss = loss.item() * args.grad_accum
                peak = torch.cuda.max_memory_allocated() / 1e9
                writer.writerow([step, epoch, f"{last_loss:.6f}", f"{peak:.3f}"])
                pbar.set_postfix(loss=f"{last_loss:.4f}", mem=f"{peak:.1f}G")
            print(f"epoch {epoch:4d}/{args.epochs}  loss {last_loss:.4f}  peakmem {peak:.2f}GB", flush=True)

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
        print(f"\n[stop] interrupted @ epoch {epoch} -> saved latest.pt (resume with --resume auto).", flush=True)
    finally:
        mf.close()
    if interrupted:
        return
    print("done")


if __name__ == "__main__":
    main()
