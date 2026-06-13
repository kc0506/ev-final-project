"""Modal image + smoke for the `physdreamer` env, reproduced from the live conda
env's `pip freeze` (NOT environment.yml). reuse_mpm pipeline lives in THIS repo and
binds to the PhysDreamer checkout via reuse_mpm/_env.py (sys.path, PHYSDREAMER_ROOT).

Same idiom as infra/modal/gic.py: Modal's in-container runtime needs Python >=3.10,
so the py3.9 physdreamer stack goes into a SEPARATE env /opt/conda/envs/pd and
@app.functions shell out to /opt/conda/envs/pd/bin/python.

Non-PyPI pieces (from freeze):
  - torch 2.0.0+cu118 / torchvision 0.15.1+cu118  (pip, cu118 index)
  - warp-lang 0.10.1                               (pip, bundles its own CUDA)
  - diff_gaussian_rasterization @ graphdeco-inria@59f5f77  (ORIGINAL, not diff_gauss)
  - simple_knn @ bkerbl@44f7642
  - physdreamer + local_utils                      (PhysDreamer checkout, sys.path only)

Volumes:
  /tmp2/b10401006/PhysDreamer/data/physics_dreamer (pd-data) -> dataset dirs at the
      cache's hardcoded absolute path (telephone/point_cloud.ply, transforms, ...)
  /out (pd-out) -> run dirs + writable scene cache

Run:
  modal run infra/modal/physdreamer.py::versions          # import smoke (GPU)
  modal run infra/modal/physdreamer.py::forward_telephone  # forward_gen, 1 video
"""

import os

import modal

app = modal.App("genphys-physdreamer")

GENPHYS_LOCAL = "/tmp2/b10401006/ev-project/generative-phys"
PD_CHECKOUT = "/tmp2/b10401006/PhysDreamer"
PD_PY = "/opt/conda/envs/pd/bin/python"
PD_PIP = "/opt/conda/envs/pd/bin/pip"
ENV_PREFIX = "/opt/conda/envs/pd"
PHYSDREAMER_ROOT = "/root/PhysDreamer"

GPU_SMOKE = "T4"
# physdreamer is torch2.0+cu118 -> cap at Ampere (A100). cu118/torch2.0 don't support
# L4(8.9)/Hopper(9.0)/Blackwell well. Default A100-40GB for headroom + speed.
GPU_RUN = os.environ.get("GENPHYS_GPU", "A100-40GB")
ARCH = "7.0 7.5 8.0 8.6"  # T4=7.5 A100=8.0 A10=8.6 (no 8.9/9.0 for cu118)

CHANNELS = "-c nvidia/label/cuda-11.8.0 -c conda-forge"
CONDA_PKGS = "python=3.9 pip setuptools wheel cuda-toolkit=11.8 gxx_linux-64=11.4 ninja git"

TORCH = ("torch==2.0.0+cu118 torchvision==0.15.1+cu118 "
         "--extra-index-url https://download.pytorch.org/whl/cu118")
DGR = "https://github.com/graphdeco-inria/diff-gaussian-rasterization"
DGR_SHA = "59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d"
SIMPLE_KNN = "git+https://gitlab.inria.fr/bkerbl/simple-knn.git@44f764299fa305faf6ec5ebd99939e0508331503"

pd_image = (
    modal.Image.micromamba()
    .run_commands(f"micromamba create -y -n pd {CHANNELS} {CONDA_PKGS}")
    .add_local_file("infra/modal/pd_requirements.txt", "/pd_requirements.txt", copy=True)
    .env({"TORCH_CUDA_ARCH_LIST": ARCH, "FORCE_CUDA": "1", "CUDA_HOME": ENV_PREFIX,
          "PHYSDREAMER_ROOT": PHYSDREAMER_ROOT})
    .run_commands(f"micromamba run -n pd pip install {TORCH}")
    .run_commands(f"micromamba run -n pd pip install warp-lang==0.10.1")
    .run_commands(f"{PD_PIP} install -r /pd_requirements.txt")
    # diff_gaussian_rasterization has a glm submodule -> clone --recursive (pip git+ won't).
    # Whole thing inside `micromamba run -n pd bash -c` so git + the conda compiler/nvcc are
    # on PATH (base env has no git; raw `git clone` would hit `command not found`).
    .run_commands(
        f'micromamba run -n pd bash -c "git clone --recursive {DGR} /tmp/dgr && '
        f'cd /tmp/dgr && git checkout {DGR_SHA} && git submodule update --init --recursive && '
        f'pip install --no-build-isolation ."'
    )
    .run_commands(f'micromamba run -n pd pip install --no-build-isolation "{SIMPLE_KNN}"')
    # physdreamer package runtime deps NOT in environment.yml's pip list (found by grepping
    # the modules reuse_mpm/_env.py imports). After the ext layers so those stay cached.
    # jaxtyping hard-pins typeguard==2.13.3 which conflicts with tyro's 4.x -> --no-deps
    # (the live env has 0.2.28 + typeguard 4.5.2 coexisting; jaxtyping runs fine at runtime).
    .run_commands(
        # exact versions from the live env -> all numpy<2 compatible (latest opencv pulls
        # numpy>=2, which breaks torch2.0/warp; environment.yml pins numpy==1.24.1).
        f'{PD_PIP} install "numpy==1.24.1" "opencv-python==4.8.1.78" "decord==0.6.0" '
        f'"omegaconf==2.1.1" "scikit-learn==1.3.2" "trimesh==4.12.2" "pymeshlab==2023.12.post1" '
        f'"kmeans-gpu==0.0.5" && '  # kmeans_gpu: lazy import in local_utils, only on cache build
        f"{PD_PIP} install --no-deps jaxtyping==0.2.28"
    )
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg")  # ffmpeg: mediapy.write_video
    # PhysDreamer checkout (code only) -> PHYSDREAMER_ROOT; reuse_mpm/_env.py binds to it.
    .add_local_dir(
        PD_CHECKOUT, PHYSDREAMER_ROOT,
        ignore=["data", "output", "models", "figures", ".git", "**/__pycache__",
                "**/*.pyc", "physgaia_pipeline", "scripts"],
    )
    # this repo (reuse_mpm pipeline) -> /root/genphys
    .add_local_dir(
        GENPHYS_LOCAL, "/root/genphys",
        ignore=["outputs", "PhysDreamer", "data-pd", ".git", "vsd", "teacher", "poster",
                "reports", "_archive", "**/__pycache__", "**/*.pyc", "*.pdf"],
    )
)

data_vol = modal.Volume.from_name("pd-data", create_if_missing=True)
out_vol = modal.Volume.from_name("pd-out", create_if_missing=True)

_IMPORT_PROBE = r"""
import os, sys
os.chdir("/root/genphys"); sys.path.insert(0, "/root/genphys")
import torch
print("python", sys.version.split()[0], "| torch", torch.__version__,
      "| cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
for m in ("warp","diff_gaussian_rasterization","simple_knn._C","numpy"):
    try:
        mod=__import__(m); print(f"  {m:30s} OK", getattr(mod,"__version__",""))
    except Exception as e: print(f"  {m:30s} FAIL {type(e).__name__}: {e}")
# the big one: pulls physdreamer.* + local_utils via _env.py (PHYSDREAMER_ROOT)
try:
    import reuse_mpm._env
    print("  reuse_mpm._env                 OK (physdreamer + local_utils wired)")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  reuse_mpm._env                 FAIL {type(e).__name__}: {e}")
"""


# The pd-env python (3.9) must NOT inherit Modal runtime's PYTHONPATH=/pkg, or it
# auto-imports Modal's package and hits its py>=3.10 guard. Override PYTHONPATH.
def _pd_env() -> dict:
    return {**os.environ, "PYTHONPATH": "/root/genphys", "PHYSDREAMER_ROOT": PHYSDREAMER_ROOT}


@app.function(image=pd_image, gpu=GPU_SMOKE)
def versions() -> str:
    import subprocess
    r = subprocess.run([PD_PY, "-c", _IMPORT_PROBE], capture_output=True, text=True,
                       env=_pd_env())
    out = r.stdout + ("\n[stderr]\n" + r.stderr if r.returncode else "")
    print(out)
    return out


def _gpu_mem_peak_mib(stop) -> "list[int]":
    """Sample whole-card memory.used (MiB) until stop is set; returns [peak]."""
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


# telephone dataset dir at the cache's hardcoded absolute path (pd-data mounts here).
PD_DATASET_ROOT = "/tmp2/b10401006/PhysDreamer/data/physics_dreamer"
TELEPHONE_DIR = f"{PD_DATASET_ROOT}/telephone"


@app.function(image=pd_image, gpu=GPU_RUN,
              volumes={PD_DATASET_ROOT: data_vol, "/out": out_vol}, timeout=2400)
def forward_telephone() -> str:
    """Real entrypoint: reuse_mpm.forward_gen -> 1 telephone video at constant E=1e6.
    Builds the scene cache into /out (cwd is read-only). Reports whole-card peak mem."""
    import subprocess
    import threading

    assert os.path.exists(f"{TELEPHONE_DIR}/point_cloud.ply"), "point_cloud.ply missing in pd-data"
    cmd = [
        PD_PY, "-m", "reuse_mpm.forward_gen",
        "--scene.path", TELEPHONE_DIR,
        "--scene.kind", "pd",
        "--scene.cache-path", "/out/telephone_scene_cache.pt",
        "--E", "1e6", "--v0", "0", "-0.5", "0",
        "--frame", "frame_00001.png",
        "--sim.num-frames", "14", "--sim.substep", "64",
        "--out", "/out/fwd_telephone",
    ]
    print("RUN:", " ".join(cmd), flush=True)
    env = _pd_env()
    stop = threading.Event()
    peak = [0]
    t = threading.Thread(target=lambda: peak.__setitem__(0, _gpu_mem_peak_mib(stop)[0]))
    t.start()
    r = subprocess.run(cmd, cwd="/root/genphys", capture_output=True, text=True, env=env)
    stop.set()
    t.join()
    print("STDOUT tail:\n" + "\n".join(r.stdout.splitlines()[-50:]))
    print("STDERR tail:\n" + "\n".join(r.stderr.splitlines()[-30:]))
    out_vol.commit()
    import glob
    mp4s = glob.glob("/out/fwd_telephone/**/*.mp4", recursive=True)
    msg = (f"gpu={GPU_RUN} exit={r.returncode} whole_card_peak={peak[0]} MiB "
           f"mp4={'OK ' + mp4s[0] if mp4s else 'MISSING'}")
    print(msg)
    return msg


@app.local_entrypoint()
def main():
    print(versions.remote())
