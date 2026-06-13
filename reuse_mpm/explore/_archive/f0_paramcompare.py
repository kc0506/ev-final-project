"""Entrypoint: forward BOTH the joint-recovered (E, alpha) and the GT (E*, 1) from
the same release t0, roll LONG (beyond the fit window), and compare per-frame
distance. Decides why the joint fit settled off-truth:

  - large distance already within the fit window (frame<=K)  => bad optimization
  - same within K, diverges only at long horizon             => K-frame info can't
                                                                 separate them (longer
                                                                 observation needed)
  - same even at long horizon                                => genuinely non-identifiable
                                                                 (the two param sets are
                                                                 the same trajectory)

In-model (warp) on purpose: isolates identifiability from the cross-model floor.

Output under outputs/explore/f0_paramcompare/.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import tyro

from ..gpu import pick_free_gpu


@dataclass
class F0ParamCompareConfig:
    cache_path: str = ("/tmp2/b10401006/ev-project/generative-phys/outputs/"
                       "_scene_cache/telephone_ds0.1_g32_k8.pt")
    snapshot: str = ("/tmp2/b10401006/ev-project/generative-phys/outputs/"
                     "explore/f0_snapshot/tele_f0_xy/snapshot_frame08.pt")
    nu: float = 0.3
    num_frames: int = 48          # LONG; fit used 8
    fit_window: int = 8           # marker
    # (label, logE, alpha) -- first is the GT reference; others compared to it
    gt: Tuple[float, float] = (5.0, 1.0)             # (logE, alpha)
    rec: Tuple[float, float] = (4.9493, 1.093)        # log10(88990), joint-recovered
    label: str = "tele_gt_vs_jointrec_long"


def run(cfg: F0ParamCompareConfig) -> str:
    # respect a preset CUDA_VISIBLE_DEVICES (co-location to keep idle-GPU count up,
    # per the quota-penalty policy); else auto-pick the freest.
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        from ..gpu import assert_gpu_quota
        assert_gpu_quota(8.0)
        print(f"[gpu] using preset CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        pick_free_gpu()
    import numpy as np
    import torch
    import warp as wp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..config import SceneSpec, ScenePreset, SimConfig
    from ..scene_io import load_from_spec
    from ..sim_render import build_mpm

    t0 = time.time()
    out_dir = os.path.join("outputs", "explore", "f0_paramcompare", cfg.label)
    os.makedirs(out_dir, exist_ok=True)

    spec = SceneSpec(preset=ScenePreset.telephone, cache_path=cfg.cache_path)
    sim = SimConfig(); sim.num_frames = cfg.num_frames
    scene = load_from_spec(spec, sim)
    dev = scene.device
    rest_xyz = scene.sim_xyzs
    n = rest_xyz.shape[0]
    free = (~scene.freeze_mask).to(dev)

    snap = torch.load(cfg.snapshot, map_location=dev)
    snap_x = snap["x"].to(dev).float()
    V0 = snap["F"].to(dev).float()
    FFt = V0 @ V0.transpose(-1, -2)
    mu, Q = torch.linalg.eigh(FFt); mu = mu.clamp_min(1e-9)

    def F0_at(alpha):
        return (Q * mu.pow(alpha / 2.0).unsqueeze(-2)) @ Q.transpose(-1, -2)

    def rollout(logE, alpha):
        solver, state, model = build_mpm(scene, sim, requires_grad=False)
        dens = torch.ones_like(rest_xyz[..., 0]) * sim.density
        state.reset_density(dens.clone(), torch.ones_like(dens).int(), dev, update_mass=True)
        with torch.no_grad():
            E_t = torch.ones_like(rest_xyz[..., 0]) * float(10.0 ** logE)
            nu_t = torch.ones_like(rest_xyz[..., 0]) * cfg.nu
            solver.set_E_nu_from_torch(model, E_t.clone(), nu_t.clone(), dev)
            solver.prepare_mu_lam(model, state, dev)
            v0 = torch.zeros(n, 3, device=dev); C0 = torch.zeros(n, 3, 3, device=dev)
            state.continue_from_torch(snap_x.clone(), v0, F0_at(alpha), C0,
                                      device=dev, requires_grad=False)
            prev = state
            out = [wp.to_torch(prev.particle_x).clone()]
            for _ in range(cfg.num_frames - 1):
                for _ in range(sim.substep):
                    nxt = prev.partial_clone(requires_grad=False)
                    solver.p2g2p_differentiable(model, prev, nxt, sim.substep_size, device=dev)
                    prev = nxt
                out.append(wp.to_torch(prev.particle_x).clone())
        return torch.stack(out)  # [T,n,3]

    gt = rollout(cfg.gt[0], cfg.gt[1])
    rec = rollout(cfg.rec[0], cfg.rec[1])
    fr = free
    # per-frame: distance between the two rollouts, and each one's own motion vs t0
    d = (gt - rec).norm(dim=-1)                       # [T,n]
    d_mean = d[:, fr].mean(1).cpu().numpy()
    d_p95 = d[:, fr].quantile(0.95, dim=1).cpu().numpy()
    gt_motion = (gt - gt[0]).norm(dim=-1)[:, fr].mean(1).cpu().numpy()
    rec_motion = (rec - rec[0]).norm(dim=-1)[:, fr].mean(1).cpu().numpy()
    ratio = d_mean / np.maximum(gt_motion, 1e-9)

    np.savez(os.path.join(out_dir, "compare.npz"), d_mean=d_mean, d_p95=d_p95,
             gt_motion=gt_motion, rec_motion=rec_motion, ratio=ratio)

    K = cfg.fit_window
    print(f"[cmp] GT (logE {cfg.gt[0]}, a {cfg.gt[1]}) vs REC (logE {cfg.rec[0]}, a {cfg.rec[1]})")
    print(f"[cmp] frame {K} (fit window end): d_mean {d_mean[K]:.5f}  motion {gt_motion[K]:.5f}  "
          f"d/motion {ratio[K]:.3f}")
    print(f"[cmp] frame {cfg.num_frames-1} (long):      d_mean {d_mean[-1]:.5f}  motion {gt_motion[-1]:.5f}  "
          f"d/motion {ratio[-1]:.3f}")
    print(f"[cmp] d_mean growth K->long: {d_mean[K]:.5f} -> {d_mean[-1]:.5f} ({d_mean[-1]/max(d_mean[K],1e-9):.1f}x)")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    fr_idx = range(cfg.num_frames)
    ax[0].plot(fr_idx, d_mean, "-o", ms=3, label="mean |GT - rec|")
    ax[0].plot(fr_idx, d_p95, "-o", ms=3, label="p95")
    ax[0].plot(fr_idx, gt_motion, "--", color="gray", label="GT motion (ref)")
    ax[0].axvline(K, color="red", ls=":", label=f"fit window (frame {K})")
    ax[0].set_title("per-frame distance between the two param sets")
    ax[0].set_xlabel("frame"); ax[0].set_ylabel("normalized sim dist"); ax[0].legend(fontsize=8)
    ax[1].plot(fr_idx, ratio, "-o", ms=3, color="tab:purple")
    ax[1].axvline(K, color="red", ls=":")
    ax[1].axhline(0.05, color="k", ls="--", lw=0.7, label="5% of motion")
    ax[1].set_title("distance / GT-motion  (separability)")
    ax[1].set_xlabel("frame"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "param_distance.png"), dpi=120); plt.close(fig)
    print(f"[cmp] DONE -> {out_dir} ({time.time()-t0:.1f}s)")
    return out_dir


if __name__ == "__main__":
    run(tyro.cli(F0ParamCompareConfig))
