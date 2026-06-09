"""Entrypoint: sample Y ~ p*(E) and push forward to a video dataset.

Realises the "sample Y, render X" half of the goal: a known functional
distribution over the physics parameter (here Y = global scalar E,
p*(E) = log-uniform[E_min, E_max]) sampled and pushed through the SAME MPM->3DGS
pipeline to produce a paired (E, video) dataset. Depends only on render
correctness, not on any training result.

  python -m reuse_mpm.dataset_gen \
      --scene.path .../telephone --E_min 1e4 --E_max 1e6 --n 16 \
      --v0 0 -1 0 --sim.num-frames 8 --sim.substep 32 --seed 0 \
      --out outputs/dataset_telephone_logU_1e4_1e6

Output dir (info-complete):
  config.json            resolved DatasetConfig + provenance
  manifest.json          p*(E) definition + per-sample (E, paths, stability)
  source_ply             symlink to the ply used
  scene_cache            symlink to the shared particle discretisation
  p_star.png             histogram of sampled E vs the target density
  sample_XXXX/           per sample: video.mp4/gif, frames/, video.npy, mpm_xyz.npy
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import tyro

from .config import DatasetConfig
from .gpu import pick_free_gpu


def run(cfg: DatasetConfig):
    pick_free_gpu()
    from .scene_io import load_from_spec
    from .sim_render import make_constant_v0, simulate_positions, render_positions, video_to_uint8
    from .run_io import DatasetRun

    t0 = time.time()
    rd = DatasetRun(cfg.out)

    # p*(E) = log-uniform[E_min, E_max]; sample (seeded for reproducibility)
    rng = np.random.RandomState(cfg.seed)
    logE = rng.uniform(np.log10(cfg.E_min), np.log10(cfg.E_max), size=cfg.n)
    Es = (10.0 ** logE).astype(np.float64)

    scene = load_from_spec(cfg.scene, cfg.sim)  # resolves cfg.scene.cache_path
    cam = scene.camera_by_frame(cfg.frame)
    v0 = make_constant_v0(scene, cfg.v0)

    # provenance + top-level symlinks
    rd.config(cfg, scene_name=scene.name, n_mpm_particles=int(scene.sim_xyzs.shape[0]))
    if cfg.scene.kind == "pd":
        rd.link(os.path.join(cfg.scene.path, "point_cloud.ply"), "source_ply")
    rd.link(cfg.scene.cache_path, "scene_cache")

    samples = []
    n_unstable = 0
    for i, E in enumerate(Es):
        sd = rd.sample_dir(i)

        # MPM rollout -> keep per-frame particle positions (the geometric state of
        # truth); KNN/top_k/init/scale/shift live once in the shared scene_cache,
        # GS positions are reconstructable from these + KNN so we don't duplicate.
        pos_list = simulate_positions(scene, float(E), v0, cfg.sim)  # list[T] of [n,3] world
        norm = [(p + scene.shift) / scene.scale for p in pos_list]
        jumps = [float((norm[t] - norm[t - 1]).norm(dim=-1).max())
                 for t in range(1, len(norm))]
        max_jump = max(jumps) if jumps else 0.0
        stable = bool(max_jump < cfg.jump_thresh)
        n_unstable += (not stable)

        mpm_xyz = torch.stack(pos_list, 0).cpu().numpy()  # [T,n,3] world
        np.save(sd.path("mpm_xyz.npy"), mpm_xyz)

        vid_u8 = video_to_uint8(render_positions(scene, pos_list, cam))
        np.save(sd.path("video.npy"), vid_u8)
        sd.save_video(vid_u8, fps=cfg.sim.fps)

        sd.write_json("sample.json", {
            "id": i, "E": float(E), "log10_E": float(np.log10(E)),
            "max_frame_jump": max_jump, "stable": stable, "frame_jumps": jumps,
        })
        samples.append({"id": i, "E": float(E),
                        "dir": os.path.relpath(sd.root, rd.root),
                        "mp4": os.path.relpath(sd.path("video.mp4"), rd.root),
                        "max_frame_jump": max_jump, "stable": stable})
        flag = "" if stable else "  [UNSTABLE]"
        print(f"  [{i+1}/{cfg.n}] E={E:.3e}  max_jump={max_jump:.3f}{flag} -> {sd.root}")

    rd.manifest({
        "task": "dataset_gen",
        "p_star": {"type": "log_uniform", "E_min": cfg.E_min, "E_max": cfg.E_max},
        "scene": scene.name, "seed": cfg.seed,
        "n_mpm_particles": int(scene.sim_xyzs.shape[0]),
        "n": cfg.n, "n_unstable": n_unstable,
        "jump_thresh": cfg.jump_thresh, "samples": samples,
        "elapsed_sec": round(time.time() - t0, 2),
    })

    # p*(E) histogram vs target
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(np.log10(Es), bins=min(cfg.n, 12), density=True,
                alpha=0.6, label="sampled log10 E")
        lo, hi = np.log10(cfg.E_min), np.log10(cfg.E_max)
        ax.hlines(1.0 / (hi - lo), lo, hi, color="r", ls="--",
                  label="target log-uniform density")
        ax.set_xlabel("log10 E"); ax.set_ylabel("density")
        ax.set_title(f"p*(E)=logU[{cfg.E_min:.0e},{cfg.E_max:.0e}], n={cfg.n}")
        ax.legend(); fig.tight_layout()
        fig.savefig(rd.path("p_star.png"), dpi=120); plt.close(fig)
    except Exception as e:
        print(f"[dataset] plot failed: {e}")

    print(f"[dataset] {cfg.n} samples, E in [{Es.min():.2e},{Es.max():.2e}] "
          f"-> {rd.root}  ({time.time()-t0:.1f}s)")
    return rd.root


if __name__ == "__main__":
    run(tyro.cli(DatasetConfig))
