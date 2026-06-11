"""Entrypoint: sample Y=(E, v0, T) ~ p*(Y) and push forward to a video dataset.

Realises the "sample Y, render X" half of the goal: a known functional
distribution over the conditioning Y -- now FACTORISED into three independent
1-D axes (sampling.EDist / V0Dist / TDist) -- sampled and pushed through the
SAME MPM->3DGS pipeline to produce a paired (Y, video) dataset. Depends only on
render correctness, not on any training result.

  # v1 back-compat: E log-uniform, v0/T fixed
  python -m reuse_mpm.dataset_gen --scene.preset telephone \
      --e_dist.mode loguniform --e_dist.E-min 1e4 --e_dist.E-max 1e6 --n 16

  # dataset a: E fixed, v0 along +/-x with magnitude U[0,2] (incl. rest), T=16
  python -m reuse_mpm.dataset_gen --scene.preset telephone \
      --e_dist.mode fixed --e_dist.E 1e5 \
      --v0_dist.mode axis --v0_dist.axis 0 --v0_dist.mag-min 0 --v0_dist.mag-max 2 \
      --n 64 --run_label a_axisx_rest

Output dir (info-complete):
  config.json            resolved DatasetConfig + provenance
  console.log            full stdout/stderr transcript (capture_output)
  manifest.json          p*(Y) factor specs + per-sample (E, v0, T, paths, stability)
  source_ply             symlink to the ply used
  scene_cache.pt         frozen copy of this run's particle discretisation
  p_star_*.png           per-axis marginal (only for axes that actually vary)
  sample_XXXX/           per sample: video.mp4/gif, frames/, video.npy, mpm_xyz.npy, sample.json
"""
from __future__ import annotations

import dataclasses
import os
import time

import numpy as np
import torch
import tyro

from .config import DatasetConfig
from .gpu import pick_free_gpu


def run(cfg: DatasetConfig):
    pick_free_gpu()
    from .run_io import DatasetRun

    t0 = time.time()
    label = cfg.run_label or (
        f"{cfg.scene.kind}-{cfg.scene.display_name}"
        f"_E-{cfg.e_dist.mode}_v0-{cfg.v0_dist.mode}_T-{cfg.t_dist.mode}_n{cfg.n}")
    rd = DatasetRun.create(__name__, label, cfg.out, config=cfg)  # auto-saves config.json
    with rd.capture_output():  # tee stdout+stderr into the run dir
        _run(cfg, rd, t0)
    return rd.root


def _auto_summary(cfg: DatasetConfig, scene_name: str) -> str:
    """One-line description of p*(Y) when the user gives none."""
    e = cfg.e_dist
    e_s = f"E={e.E:g}" if not e.varies else f"E~logU[{e.E_min:g},{e.E_max:g}]"
    v = cfg.v0_dist
    if not v.varies:
        v_s = f"v0={tuple(v.vec)}"
    else:
        rest = "incl.rest" if v.mag_min <= 0 else "always-moving"
        dirn = (f"axis-{'xyz'[v.axis]}{'±' if v.signed else '+'}"
                if v.mode == "axis" else "dir~S^2")
        v_s = f"v0 {dirn}, |v0|~U[{v.mag_min:g},{v.mag_max:g}] ({rest})"
    t = cfg.t_dist
    t_s = f"T={cfg.sim.num_frames}" if not t.varies else f"T~U{{{t.T_min}..{t.T_max}}}"
    c = cfg.cam_dist
    c_s = "" if not c.varies else f" | cam orbit±{c.cap_deg:g}°"
    return f"{scene_name}: {e_s} | {v_s} | {t_s}{c_s} | n={cfg.n}"


def _run(cfg: DatasetConfig, rd, t0: float):
    from .scene_io import load_from_spec
    from .sim_render import (make_constant_v0, simulate_positions, render_positions,
                             render_positions_multicam, video_to_uint8)
    from .run_io import save_panel_video
    from ._env import Camera
    from .ply_io import camera_to_dict

    scene = load_from_spec(cfg.scene, cfg.sim)  # resolves cfg.scene.cache_path
    ref_cam = scene.camera_by_frame(cfg.frame)  # reference view (used as-is if cam fixed)

    # camera-sweep framing: object centroid (look-at target) + reference eye/intrinsics.
    # Only consumed when cfg.cam_dist.varies -- so a fixed-camera run never draws from
    # rng for the camera and reproduces the v1 (E,v0,T) sequence exactly.
    pos0 = (scene.sim_xyzs * scene.scale - scene.shift).detach()  # [n,3] world rest
    cam_center = pos0.mean(0).cpu().numpy()                       # [3] look-at target
    try:
        ref_eye = ref_cam.camera_center.detach().cpu().numpy()    # [3]
    except Exception:
        import numpy as _np
        ref_eye = (-_np.asarray(ref_cam.R) @ _np.asarray(ref_cam.T))  # [3] fallback
    FoVx, FoVy = float(ref_cam.FoVx), float(ref_cam.FoVy)
    Hc, Wc = int(ref_cam.image_height), int(ref_cam.image_width)

    # top-level provenance: input ply symlink + frozen discretisation snapshot
    if cfg.scene.kind == "pd":
        rd.link(os.path.join(cfg.scene.path, "point_cloud.ply"), "source_ply")
    rd.copy_in(cfg.scene.cache_path, "scene_cache.pt")  # freeze this run's discretisation

    rng = np.random.RandomState(cfg.seed)
    samples = []
    Es, mags, vecs, Ts = [], [], [], []
    # panel.gif "glance": tile <=panel_max evenly-spaced clips, labelled by the
    # conditioning axes that vary (like v0_sweep's panel).
    panel_idx = set(np.linspace(0, cfg.n - 1, min(cfg.n, cfg.panel_max))
                    .round().astype(int).tolist())
    panel_clips, panel_labels = [], []
    n_unstable = 0
    for i in range(cfg.n):
        # draw the three conditioning axes (seeded, sequential -> reproducible)
        E = cfg.e_dist.sample(rng)
        vec = cfg.v0_dist.sample(rng)
        T = cfg.t_dist.sample(rng, cfg.sim.num_frames)
        sim_i = dataclasses.replace(cfg.sim, num_frames=T)  # per-sample horizon
        # camera: draw ONLY when it varies (keeps fixed-cam rng sequence == v1).
        # dynamic (compound) -> a PER-FRAME trajectory; static -> one viewpoint.
        cam_i, cams_i, cam_meta = ref_cam, None, None
        if cfg.cam_dist.dynamic:
            traj = cfg.cam_dist.sample_traj(rng, cam_center, ref_eye, ref_cam.R, T)
            cams_i = [Camera(R=s["R"], T=s["T"], FoVx=FoVx, FoVy=FoVy, img_path=cfg.frame,
                             img_hw=(Hc, Wc), data_device=scene.device) for s in traj]
            cam_meta = {"mode": "compound",
                        "azim": [s["azim"] for s in traj], "elev": [s["elev"] for s in traj],
                        "r": [s["r"] for s in traj],
                        "eye": [list(map(float, s["eye"])) for s in traj]}
        elif cfg.cam_dist.varies:
            R, Tcw, eye = cfg.cam_dist.sample_RT(rng, cam_center, ref_eye)
            cam_i = Camera(R=R, T=Tcw, FoVx=FoVx, FoVy=FoVy, img_path=cfg.frame,
                           img_hw=(Hc, Wc), data_device=scene.device)
            cam_meta = {"eye": eye.tolist(), **camera_to_dict(cam_i)}

        sd = rd.sample_dir(i)
        v0 = make_constant_v0(scene, vec)
        # MPM rollout -> per-frame particle positions (the geometric truth);
        # KNN/top_k/init/scale/shift live once in scene_cache, GS positions are
        # reconstructable from these + KNN so we don't duplicate.
        pos_list = simulate_positions(scene, float(E), v0, sim_i)  # list[T] [n,3] world

        # jump metric over MOVING particles only: frozen/anchor particles carry a
        # fixed v0-independent settling transient that otherwise floors the metric
        # (~0.108 for telephone) and hides genuine moving-body blow-ups.
        qm = scene.query_mask
        norm = [(p + scene.shift) / scene.scale for p in pos_list]
        jumps = [float((norm[t] - norm[t - 1])[qm].norm(dim=-1).max())
                 for t in range(1, len(norm))]
        max_jump = max(jumps) if jumps else 0.0
        stable = bool(max_jump < cfg.jump_thresh)
        n_unstable += (not stable)

        mpm_xyz = torch.stack(pos_list, 0).cpu().numpy()  # [T,n,3] world
        np.save(sd.path("mpm_xyz.npy"), mpm_xyz)
        if cams_i is not None:  # dynamic per-frame camera
            vid_u8 = video_to_uint8(render_positions_multicam(scene, pos_list, cams_i))
        else:
            vid_u8 = video_to_uint8(render_positions(scene, pos_list, cam_i))
        np.save(sd.path("video.npy"), vid_u8)
        # light_io drops only the per-frame pngs (redundant with mp4); per-sample
        # gif stays (loads faster than mp4 for quick scrubbing).
        sd.save_video(vid_u8, fps=cfg.sim.fps, frames=not cfg.light_io, gif=True)

        mag = float(np.linalg.norm(vec))
        if i in panel_idx:
            lab = []
            if cfg.e_dist.varies:
                lab.append(f"E{E:.0e}")
            if cfg.v0_dist.varies:
                lab.append(f"vx{vec[0]:+.1f}" if cfg.v0_dist.mode == "axis"
                           else f"|v|{mag:.1f}")
            if cfg.t_dist.varies:
                lab.append(f"T{T}")
            panel_clips.append(vid_u8)
            panel_labels.append(" ".join(lab) or f"#{i}")
        sd.write_json("sample.json", {
            "id": i, "E": float(E), "log10_E": float(np.log10(E)),
            "v0": list(vec), "v0_magnitude": mag, "T": int(T),
            "camera": cam_meta,  # None if cam fixed; else eye + view transforms
            "max_frame_jump": max_jump, "stable": stable, "frame_jumps": jumps,
        })
        samples.append({"id": i, "E": float(E), "v0": list(vec),
                        "v0_magnitude": mag, "T": int(T),
                        "dir": os.path.relpath(sd.root, rd.root),
                        "mp4": os.path.relpath(sd.path("video.mp4"), rd.root),
                        "max_frame_jump": max_jump, "stable": stable})
        Es.append(E); mags.append(mag); vecs.append(vec); Ts.append(T)
        flag = "" if stable else "  [UNSTABLE]"
        print(f"  [{i+1}/{cfg.n}] E={E:.2e} "
              f"v0=({vec[0]:+.2f},{vec[1]:+.2f},{vec[2]:+.2f}) T={T} "
              f"max_jump={max_jump:.3f}{flag} -> {sd.root}")
        sd.finish()  # seals this sample's mpm_xyz.npy + video.npy (np.save bypass)

    summary = _auto_summary(cfg, scene.name)
    description = cfg.description or summary
    rd.manifest({
        "task": "dataset_gen",
        "description": description,
        "summary": summary,
        "p_star": {"E": cfg.e_dist.to_dict(), "v0": cfg.v0_dist.to_dict(),
                   "T": cfg.t_dist.to_dict(), "camera": cfg.cam_dist.to_dict()},
        "scene": scene.name, "seed": cfg.seed,
        "n_mpm_particles": int(scene.sim_xyzs.shape[0]),
        "n": cfg.n, "n_unstable": n_unstable,
        "jump_thresh": cfg.jump_thresh, "samples": samples,
        "elapsed_sec": round(time.time() - t0, 2),
    })
    # README.md: glanceable description of what this dataset is + how to read it.
    readme = (
        f"# {description}\n\n"
        f"`dataset_gen` paired (Y, video) dataset. Y=(E, v0, T) ~ p*(Y), each axis "
        f"an independent 1-D spec (see manifest.json `p_star`).\n\n"
        f"- **{cfg.n}** samples, **{n_unstable}** unstable (|jump|>{cfg.jump_thresh}).\n"
        f"- **{summary}**\n\n"
        f"## Layout\n"
        f"- `panel.gif` — all samples tiled, glance at the dataset's spread.\n"
        f"- `p_star_*.png` — sampled marginal of each varying axis vs target.\n"
        f"- `sample_XXXX/` — `video.npy` (X, [T,H,W,3] uint8), `video.mp4/gif`, "
        f"`mpm_xyz.npy` ([T,n,3] world particle traj), `sample.json` (its E,v0,T).\n"
        f"- `scene_cache.pt` / `source_ply` — frozen discretisation + input 3DGS; "
        f"with `mpm_xyz.npy` they reconstruct every per-frame MPM/3DGS ply on demand.\n")
    with open(rd.path("README.md"), "w") as f:
        f.write(readme)
    rd._event("README.md", "README.md")

    _plot_marginals(rd, cfg, Es, mags, vecs, Ts)
    if panel_clips:
        save_panel_video(
            rd.path("panel.gif"), panel_clips, panel_labels, fps=cfg.sim.fps,
            title=f"{scene.name}  E:{cfg.e_dist.mode} v0:{cfg.v0_dist.mode} "
                  f"T:{cfg.t_dist.mode}  n={cfg.n}")

    rd.finish()  # seals manifest, p_star_*.png, panel.gif, console.log
    print(f"[dataset] {cfg.n} samples "
          f"(E:{cfg.e_dist.mode} v0:{cfg.v0_dist.mode} T:{cfg.t_dist.mode}, "
          f"{n_unstable} unstable) -> {rd.root}  ({time.time()-t0:.1f}s)")


def _plot_marginals(rd, cfg, Es, mags, vecs, Ts):
    """Per-axis marginal plots -- only for the conditioning axes that vary."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[dataset] plot skipped (no matplotlib): {e}")
        return

    if cfg.e_dist.varies:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(np.log10(Es), bins=min(len(Es), 12), density=True, alpha=0.6,
                label="sampled log10 E")
        lo, hi = np.log10(cfg.e_dist.E_min), np.log10(cfg.e_dist.E_max)
        ax.hlines(1.0 / (hi - lo), lo, hi, color="r", ls="--", label="target log-uniform")
        ax.set_xlabel("log10 E"); ax.set_ylabel("density")
        ax.set_title(f"p*(E)=logU[{cfg.e_dist.E_min:.0e},{cfg.e_dist.E_max:.0e}], n={len(Es)}")
        ax.legend(); fig.tight_layout()
        fig.savefig(rd.path("p_star_E.png"), dpi=120); plt.close(fig)

    if cfg.v0_dist.varies:
        V = np.asarray(vecs)  # [n,3]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(mags, bins=min(len(mags), 12), density=True, alpha=0.6, color="C2")
        ax.axvline(cfg.v0_dist.mag_min, color="r", ls="--")
        ax.axvline(cfg.v0_dist.mag_max, color="r", ls="--", label="target U range")
        ax.set_xlabel("|v0|"); ax.set_ylabel("density")
        ax.set_title(f"p*(|v0|)=U[{cfg.v0_dist.mag_min},{cfg.v0_dist.mag_max}], {cfg.v0_dist.mode}")
        ax.legend(); fig.tight_layout()
        fig.savefig(rd.path("p_star_v0mag.png"), dpi=120); plt.close(fig)

        # per-component hist: the interpretability check (axis dataset -> vy,vz==0)
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
        for k, name in enumerate("xyz"):
            axs[k].hist(V[:, k], bins=min(len(mags), 12), alpha=0.7, color=f"C{k}")
            axs[k].set_title(f"v0_{name}"); axs[k].axvline(0, color="k", lw=0.6)
        axs[0].set_ylabel("count")
        fig.suptitle(f"v0 components ({cfg.v0_dist.mode})"); fig.tight_layout()
        fig.savefig(rd.path("p_star_v0dir.png"), dpi=120); plt.close(fig)

    if cfg.t_dist.varies:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(Ts, bins=range(cfg.t_dist.T_min, cfg.t_dist.T_max + 2), density=True,
                alpha=0.6, color="C3", align="left")
        ax.set_xlabel("T (frames)"); ax.set_ylabel("density")
        ax.set_title(f"p*(T)=U{{{cfg.t_dist.T_min}..{cfg.t_dist.T_max}}}, n={len(Ts)}")
        fig.tight_layout()
        fig.savefig(rd.path("p_star_T.png"), dpi=120); plt.close(fig)


if __name__ == "__main__":
    run(tyro.cli(DatasetConfig))
