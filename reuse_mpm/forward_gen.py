"""Entrypoint: generate ONE video from a known constant Young's modulus E.

  python -m reuse_mpm.forward_gen \
      --scene.path /tmp2/b10401006/PhysDreamer/data/physics_dreamer/telephone \
      --E 1e6 --out outputs/fwd_telephone_E1e6 \
      --v0 0 -0.5 0 --frame frame_00001.png --sim.num-frames 14 --sim.substep 64

Produces a self-contained run dir (config, source ply symlink, frames, mp4/gif).
"""
from __future__ import annotations

import os
import time

import tyro

from .config import ForwardConfig


def run(cfg: ForwardConfig):
    # NOTE: forward_gen does not auto-pick a GPU (uses cfg.scene.device as-is),
    # matching the original behaviour; the GPU-contended entrypoints call
    # pick_free_gpu themselves.
    import numpy as np
    from .scene_io import load_from_spec
    from .sim_render import (
        make_constant_v0, make_gradient_E, make_gradient_v0, simulate_and_render,
        video_to_uint8)
    from .run_io import ForwardRun

    t0 = time.time()
    label = cfg.run_label or f"{cfg.scene.kind}-{cfg.scene.display_name}_E{cfg.E:g}"
    rd = ForwardRun.create(__name__, label, cfg.out, config=cfg)  # auto-saves config.json
    with rd.capture_output():  # tee stdout+stderr into the run dir
        if cfg.scene.kind == "pd":  # pg has no point_cloud.ply at the dir root
            rd.link_source_ply(cfg.scene.path)

        scene = load_from_spec(cfg.scene, cfg.sim)  # resolves cfg.scene.cache_path
        rd.copy_in(cfg.scene.cache_path, "scene_cache.pt")  # freeze this run's discretisation
        try:
            cam = scene.camera_by_frame(cfg.frame)
        except Exception:
            cam = scene.test_camera_list[0]  # PG cameras (r_0, ...) won't match frame_*
        # uniform v0 (constant) unless a spatial gradient is requested (phase B),
        # in which case build the per-particle GT v0 field and persist it for scoring.
        if cfg.v0_grad_axis is None:
            v0 = make_constant_v0(scene, cfg.v0)
        else:
            v0 = make_gradient_v0(scene, cfg.v0, cfg.v0_grad_axis, cfg.v0_grad_slope)
            np.save(rd.path("v0_field.npy"), v0.detach().cpu().numpy())

        # uniform E (float) unless a spatial gradient is requested (phase B), in
        # which case build the per-particle GT field and persist it for scoring.
        if cfg.E_grad_axis is None:
            E_in = cfg.E
        else:
            E_in = make_gradient_E(scene, cfg.E, cfg.E_grad_axis, cfg.E_grad_decades)
            np.save(rd.path("E_field.npy"), E_in.detach().cpu().numpy())

        vid = simulate_and_render(scene, E_in, v0, cfg.sim, cam, requires_grad=False)
        vid_u8 = video_to_uint8(vid)
        mp4, gif = rd.video(vid_u8, fps=cfg.sim.fps)

        rd.result(  # derived run facts live with the result, not the input config
            video_shape=list(vid_u8.shape),
            elapsed_sec=round(time.time() - t0, 2),
            mp4=mp4, gif=gif,
            scene_cache=cfg.scene.cache_path,  # resolved by load_from_spec
            n_mpm_particles=int(scene.sim_xyzs.shape[0]),
            n_sim_gaussians=int(scene.sim_mask.sum().item()),
            E_grad_axis=cfg.E_grad_axis, E_grad_decades=cfg.E_grad_decades,
            v0_grad_axis=cfg.v0_grad_axis, v0_grad_slope=cfg.v0_grad_slope,
        )
        rd.finish()
        print(f"[forward_gen] E={cfg.E:g} -> {mp4}  shape={vid_u8.shape}  "
              f"({time.time()-t0:.1f}s, {scene.sim_xyzs.shape[0]} particles)")
    return rd


if __name__ == "__main__":
    run(tyro.cli(ForwardConfig))
