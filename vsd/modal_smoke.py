"""Modal smoke test: reproduce the sim+render+teacher env on a cloud GPU and run a
SHORT real task to prove the full stack imports + executes. Two pillars of the flow
pipeline are exercised:
  (1) warp 0.10.1  -> the MPM physics backend (a tiny GPU kernel)
  (2) video_diffusion_pytorch 0.7.0 Unet3D -> the teacher denoiser (one GPU forward)

If both pass, the ODE/VSD probes (which only add the flow table + ckpt) can be ported
to Modal fan-out. The locally-built 3DGS CUDA exts (simple_knn / diff_gaussian_raster)
are NOT included -- the flow path never renders 3DGS (acknowledged env-debt).

Run (after `modal setup`):
    modal run vsd/modal_smoke.py                 # default L40S (Ada, ~= local 4090)
    modal run vsd/modal_smoke.py --gpu H100      # override GPU type
"""
import modal

# Exact versions mirrored from the local `physdreamer` micromamba env so cloud == local.
TORCH_CU118 = "https://download.pytorch.org/whl/cu118"
# Full CUDA 11.8 devel base (provides libnvrtc / cudnn that debian_slim lacks -> torch
# conv ops + warp kernel compile need them). py3.10 (Modal builder min; ckpt/tensors are
# version-agnostic vs the local 3.9 env). numpy<2 because torch 2.0.0 was built on numpy 1.x.
image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04", add_python="3.10")
    # +cu118 local tag pins the CUDA build; extra_index_url keeps PyPI primary so
    # triton's build deps (wheel/setuptools/lit) resolve (index_url alone hid them).
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

app = modal.App("physgen-smoke", image=image)


@app.function(gpu="L40S", timeout=900)
def smoke() -> str:
    """Run both pillars on the GPU; return a JSON STRING (not a dict): the modal CLI
    runs in a torch-less venv, so any torch type in the pickled return value fails to
    deserialize locally. A json string needs nothing to deserialize."""
    import json
    import time

    import torch

    report: dict = {}
    report["torch"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    report["device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    dev = "cuda:0"

    # --- pillar 1: warp GPU kernel (MPM backend) ----------------------------------
    import warp as wp

    wp.init()
    report["warp"] = wp.config.version

    @wp.kernel
    def add_one(a: wp.array(dtype=float)):  # noqa: ANN001
        i = wp.tid()
        a[i] = a[i] + 1.0

    t0 = time.time()
    a = wp.zeros(1024, dtype=float, device="cuda")          # [1024] on GPU
    wp.launch(add_one, dim=1024, inputs=[a], device="cuda")  # first launch JIT-compiles
    wp.synchronize()
    arr = a.numpy()                                          # [1024] -> host
    report["warp_kernel_ok"] = bool((arr == 1.0).all())
    report["warp_first_launch_s"] = round(time.time() - t0, 3)

    # --- pillar 2: Unet3D forward (teacher denoiser) -------------------------------
    from video_diffusion_pytorch import GaussianDiffusion, Unet3D

    RES, FRAMES, CH = 128, 7, 2
    unet = Unet3D(dim=64, dim_mults=(1, 2, 4, 8), channels=CH).to(dev)
    diff = GaussianDiffusion(  # noqa: F841  (constructs the buffers the probes use)
        unet, image_size=RES, num_frames=FRAMES, channels=CH, timesteps=1000, loss_type="l2"
    ).to(dev)
    unet.eval()
    x = torch.randn(1, CH, FRAMES, RES, RES, device=dev)     # [1,2,7,128,128]
    t = torch.randint(0, 1000, (1,), device=dev)             # [1]
    torch.cuda.synchronize()
    t1 = time.time()
    with torch.no_grad():
        out = unet(x, t, cond=None)                          # [1,2,7,128,128]
    torch.cuda.synchronize()
    report["unet_out_shape"] = [int(s) for s in out.shape]              # plain ints (no torch type)
    report["unet_forward_s"] = round(time.time() - t1, 3)
    report["unet_n_params_M"] = round(sum(int(p.numel()) for p in unet.parameters()) / 1e6, 2)
    report["peak_mem_GB"] = round(int(torch.cuda.max_memory_allocated()) / 1e9, 2)
    out_json = json.dumps(report, indent=2)
    print("REMOTE REPORT:\n" + out_json, flush=True)                    # also visible in modal logs
    return out_json


@app.local_entrypoint()
def main(gpu: str = "L40S") -> None:
    """Run the smoke function on the requested GPU type and print its JSON report string."""
    # rebind the GPU type if overridden (re-create the function spec)
    fn = smoke if gpu == "L40S" else smoke.with_options(gpu=gpu)
    print(fn.remote())  # already a json string
