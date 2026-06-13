"""Modal image + smoke for the `gic` env (../gic), reproduced faithfully from the
live conda env's `pip freeze` (NOT environment.yml — the real env dropped
torch_scatter/mmcv and swapped diff_gaussian_rasterization -> diff_gauss).

Modal's in-container runtime needs Python >=3.10, but gic is a Python 3.9 stack.
So we keep Modal's default base python and install the *exact* 3.9 gic env into a
SEPARATE micromamba env at /opt/conda/envs/gic; @app.functions run on the base
python and shell out to /opt/conda/envs/gic/bin/python (the standard Modal idiom
for legacy-python code).

Layers (ordered cheap->risky so Modal caches the stable bottom):
  1. micromamba create -n gic: python 3.9 + torch 2.4.0/cu121 + cu121 toolchain
  2. pip into gic: 88 pinned pure-pip deps (infra/modal/gic_requirements.txt)
  3. build 3 git CUDA extensions into gic: diff-gauss, simple-knn, pytorch3d@0.7.8

The gic repo code (../gic) is mounted at /root/gic at runtime (not baked).

Volumes:
  /data  (gic-data) -> read scene caches, e.g. telephone_ds0.1_g32_k8.pt
  /out   (gic-out)  -> write run dirs (proves volume read+write)

Run:
  modal run infra/modal/gic.py::versions          # import smoke (GPU)
  modal run infra/modal/gic.py::fit_telephone     # real entrypoint, tiny iters
"""

import os

import modal

app = modal.App("genphys-gic")

# Smoke = cheapest (just imports). Real runs default to A10 (24GB) for headroom over
# the >10GB peaks; override at launch with GENPHYS_GPU=A100-40GB / H100 / L4 / ...
# (gic is torch2.4/cu121 -> up to H100; B200 sm_100 needs cu12.8 and won't work.)
GPU_SMOKE = "T4"
GPU_RUN = os.environ.get("GENPHYS_GPU", "A10")

GIC_LOCAL = "/tmp2/b10401006/ev-project/gic"
GIC_PY = "/opt/conda/envs/gic/bin/python"
GIC_PIP = "/opt/conda/envs/gic/bin/pip"
ENV_PREFIX = "/opt/conda/envs/gic"
ARCH = "7.0 7.5 8.0 8.6 8.9 9.0"  # T4=7.5 A10=8.6 A100=8.0 L4=8.9 H100=9.0

CHANNELS = "-c pytorch -c nvidia/label/cuda-12.1.1 -c nvidia -c conda-forge"
CONDA_PKGS = (
    "python=3.9 pip setuptools wheel "
    "pytorch=2.4.0 torchvision=0.19.0 torchaudio=2.4.0 pytorch-cuda=12.1 "
    "cuda-nvcc=12.1 cuda-cudart-dev=12.1 cuda-libraries-dev=12.1 cuda-version=12.1 "
    "gxx_linux-64=11 gcc_linux-64=11 ninja git"
)

# diff-gauss (jukgei fork, module name `diff_gauss`), simple-knn, pytorch3d@0.7.8
DIFF_GAUSS = "git+https://github.com/jukgei/diff-gaussian-rasterization.git@b1e1cb83e27923579983a9ed19640c6031112b94"
SIMPLE_KNN = "git+https://gitlab.inria.fr/bkerbl/simple-knn.git@86710c2d4b46680c02301765dd79e465819c8f19"
PYTORCH3D = "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"

gic_image = (
    modal.Image.micromamba()  # base env keeps a Modal-compatible python (>=3.10)
    .run_commands(f"micromamba create -y -n gic {CHANNELS} {CONDA_PKGS}")
    .add_local_file("infra/modal/gic_requirements.txt", "/gic_requirements.txt", copy=True)
    .env({"TORCH_CUDA_ARCH_LIST": ARCH, "FORCE_CUDA": "1", "CUDA_HOME": ENV_PREFIX})
    .run_commands(f"{GIC_PIP} install -r /gic_requirements.txt")
    # 3 CUDA extensions into the gic env: need torch at build time -> --no-build-isolation.
    # Use `micromamba run -n gic` so the env is fully ACTIVATED (PATH gets git + the conda
    # compiler, and CC/CXX point at x86_64-conda-linux-gnu-g++ which torch's cpp_extension
    # needs). Plain `pip` by abs-path skips activation -> torch's `which g++` fails.
    .run_commands(f'micromamba run -n gic pip install --no-build-isolation "{DIFF_GAUSS}"')
    .run_commands(f'micromamba run -n gic pip install --no-build-isolation "{SIMPLE_KNN}"')
    .run_commands(f'micromamba run -n gic pip install --no-build-isolation "{PYTORCH3D}"')
    # open3d needs system libGL/libglib (not provided by conda) -> add late so the heavy
    # conda+ext layers above stay cached.
    .apt_install("libgl1", "libglib2.0-0")
    .add_local_dir(
        GIC_LOCAL, "/root/gic",
        ignore=["data", "output", "archive", "figures_out", "debug_nan",
                "**/__pycache__", ".git", "**/*.pyc", "matting"],
    )
)

data_vol = modal.Volume.from_name("gic-data", create_if_missing=True)
out_vol = modal.Volume.from_name("gic-out", create_if_missing=True)
# fit_image_* read point_cloud.ply from the scene cache's HARDCODED dataset_dir;
# mount pd-data at that absolute path so the hardcode resolves (dataset_dir caveat).
pd_data_vol = modal.Volume.from_name("pd-data", create_if_missing=True)
PD_DATASET_ROOT = "/tmp2/b10401006/PhysDreamer/data/physics_dreamer"

_IMPORT_PROBE = r"""
import os, sys, subprocess
os.chdir("/root/gic"); sys.path.insert(0, "/root/gic")
import torch
print("python", sys.version.split()[0], "| torch", torch.__version__,
      "| cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
for m in ("diff_gauss","simple_knn._C","pytorch3d","taichi","open3d","lpips"):
    try:
        mod=__import__(m); print(f"  {m:16s} OK", getattr(mod,"__version__",""))
    except Exception as e: print(f"  {m:16s} FAIL {type(e).__name__}: {e}")
for m in ("simulator","gaussian_renderer","ours.scene","ours.config"):
    try: __import__(m); print(f"  {m:16s} OK")
    except Exception as e: print(f"  {m:16s} FAIL {type(e).__name__}: {e}")
"""


@app.function(image=gic_image, gpu=GPU_SMOKE)
def versions() -> str:
    """Import smoke: confirm CUDA exts + gic packages import on a GPU (via the gic env)."""
    import subprocess
    r = subprocess.run([GIC_PY, "-c", _IMPORT_PROBE], capture_output=True, text=True)
    out = r.stdout + ("\n[stderr]\n" + r.stderr if r.returncode else "")
    print(out)
    return out


def _gpu_mem_peak_mib(stop) -> "list[int]":
    """Sample whole-card memory.used (MiB) until stop is set; returns [peak]. Captures
    taichi's allocations too (torch.cuda.* would miss them -> ti grabs 30% by default)."""
    import subprocess
    import time

    peak = [0]
    while not stop.is_set():
        try:
            used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                   "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.strip().splitlines()
            peak[0] = max(peak[0], max(int(x) for x in used if x.strip()))
        except Exception:
            pass
        time.sleep(0.3)
    return peak


@app.function(image=gic_image, gpu=GPU_RUN, volumes={"/data": data_vol, "/out": out_vol}, timeout=1800)
def fit_telephone(iters: int = 3) -> str:
    """Real entrypoint: fit_traj_Escalar on the telephone scene cache (tiny iters).
    Reports whole-card peak memory (incl. taichi) sampled via nvidia-smi."""
    import subprocess
    import threading

    cache = "/data/telephone_ds0.1_g32_k8.pt"
    assert os.path.exists(cache), f"cache missing in volume: {cache}"
    out_dir = "/out/telephone_smoke"
    cmd = [
        GIC_PY, "fit_traj_Escalar.py",
        "--scene.cache", cache,
        "--scene.config", "config/ours/telephone.json",
        "--gt.logE", "5.0", "--init_logE", "4.0", "--fix_v0",
        "--train.iter_cnt", str(iters), "--train.min_iters", "1", "--train.patience", "99",
        "--out", out_dir,
    ]
    print("RUN:", " ".join(cmd), flush=True)
    stop = threading.Event()
    peak = [0]
    t = threading.Thread(target=lambda: peak.__setitem__(0, _gpu_mem_peak_mib(stop)[0]))
    t.start()
    r = subprocess.run(cmd, cwd="/root/gic", capture_output=True, text=True)
    stop.set()
    t.join()
    print("STDOUT tail:\n" + "\n".join(r.stdout.splitlines()[-40:]))
    print("STDERR tail:\n" + "\n".join(r.stderr.splitlines()[-40:]))
    out_vol.commit()
    result_path = os.path.join(out_dir, "result.json")
    ok = os.path.exists(result_path)
    msg = (f"gpu={GPU_RUN} exit={r.returncode} whole_card_peak={peak[0]} MiB "
           f"result.json={'OK ' + result_path if ok else 'MISSING'}")
    print(msg)
    return msg


@app.function(image=gic_image, gpu=GPU_RUN, timeout=2400,
              volumes={"/data": data_vol, "/out": out_vol, PD_DATASET_ROOT: pd_data_vol})
def fit_image(iters: int = 5) -> str:
    """fit_image_Escalar (full-res with-bg render) on telephone, tiny iters.

    Needs point_cloud.ply from the cache's hardcoded dataset_dir -> pd-data mounted
    at PD_DATASET_ROOT. Defaults (rot -22.4 / fov 0.14 / gaussians=fullres) come
    from RenderCfg/SceneCfg. Reports whole-card peak mem (incl. taichi)."""
    import subprocess
    import threading

    cache = "/data/telephone_ds0.1_g32_k8.pt"
    assert os.path.exists(cache), f"cache missing in volume: {cache}"
    ply = f"{PD_DATASET_ROOT}/telephone/point_cloud.ply"
    assert os.path.exists(ply), f"point_cloud.ply missing (dataset_dir): {ply}"
    out_dir = "/out/img_smoke"
    cmd = [
        GIC_PY, "fit_image_Escalar.py",
        "--scene.cache", cache,
        "--scene.config", "config/ours/telephone.json",
        "--scene.mpm-iter-cnt", "64",
        "--gt.logE", "5.0", "--init-logE", "4.0", "--frames.n-frames", "8",
        "--train.iter-cnt", str(iters), "--train.min-iters", "1", "--train.patience", "99",
        "--out", out_dir,
    ]
    print("RUN:", " ".join(cmd), flush=True)
    stop = threading.Event()
    peak = [0]
    t = threading.Thread(target=lambda: peak.__setitem__(0, _gpu_mem_peak_mib(stop)[0]))
    t.start()
    r = subprocess.run(cmd, cwd="/root/gic", capture_output=True, text=True)
    stop.set()
    t.join()
    print("STDOUT tail:\n" + "\n".join(r.stdout.splitlines()[-40:]))
    print("STDERR tail:\n" + "\n".join(r.stderr.splitlines()[-40:]))
    out_vol.commit()
    result_path = os.path.join(out_dir, "result.json")
    ok = os.path.exists(result_path)
    msg = (f"gpu={GPU_RUN} exit={r.returncode} whole_card_peak={peak[0]} MiB "
           f"result.json={'OK ' + result_path if ok else 'MISSING'}")
    print(msg)
    return msg


@app.local_entrypoint()
def main():
    print(versions.remote())
