"""Measure peak GPU memory of the teacher video-diffusion model for one config.

Mirrors train_video.py's model build + a real train step (forward diff(x) ->
backward -> AdamW step) on RANDOM data of the right shape, then reports peak
allocated memory. Run ONE config per process so CUDA memory starts clean.

  python probe_fit.py --frames 16 --res 128 --dim 64 --dim_mults 1 2 4 8 --batch 1

Prints a single JSON line: {config, params_M, peak_gb, status}.
status: "ok" | "oom" | "error".
"""
import argparse, json, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--dim_mults", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast around forward")
    a = ap.parse_args()
    cfg = {"frames": a.frames, "res": a.res, "dim": a.dim,
           "dim_mults": a.dim_mults, "batch": a.batch, "amp": a.amp}
    out = {"config": cfg, "params_M": None, "peak_gb": None, "status": "error"}
    try:
        import torch
        from video_diffusion_pytorch import Unet3D, GaussianDiffusion
        dev = "cuda"
        torch.cuda.reset_peak_memory_stats()
        unet = Unet3D(dim=a.dim, dim_mults=tuple(a.dim_mults))
        diff = GaussianDiffusion(unet, image_size=a.res, num_frames=a.frames,
                                 timesteps=1000, loss_type="l2").to(dev)
        out["params_M"] = round(sum(p.numel() for p in diff.parameters()) / 1e6, 2)
        opt = torch.optim.AdamW(diff.parameters(), lr=1e-4)
        for _ in range(a.steps):
            x = torch.rand(a.batch, 3, a.frames, a.res, a.res, device=dev)
            if a.amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = diff(x)
            else:
                loss = diff(x)
            loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        out["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        out["status"] = "ok"
    except Exception as e:  # noqa
        msg = f"{type(e).__name__}: {e}"
        out["status"] = "oom" if "out of memory" in str(e).lower() else "error"
        out["error"] = msg[:300]
        try:
            import torch
            out["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        except Exception:
            pass
    print(json.dumps(out))
    sys.exit(0)


if __name__ == "__main__":
    main()
