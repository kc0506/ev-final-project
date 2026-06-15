"""Validate the no-3DGS differentiable MPM rollout, end to end, against the stored
ground-truth trajectory (mpm_xyz.npy) -- and confirm gradients reach v0.

This is the de-risking milestone for the flow VSD pipeline: it proves we can drive
the vendored differentiable MPM solver locally with NO CUDA-ext build (simple_knn
stubbed, gaussians never instantiated), reproduce the dataset's physics, and get a
clean gradient d(position)/d(v0). No teacher, no rendering yet.

  micromamba run -n physdreamer python -m vsd.check_rollout
"""
import vsd.bootstrap  # noqa: F401  (side-effecting: sys.path + PHYSDREAMER_ROOT + stub)

import json
import os

import numpy as np
import torch

from reuse_mpm.config import SimConfig
from reuse_mpm.mpm_rollout import MpmRollout
from reuse_mpm.sim_render import make_constant_v0
from vsd.scene_min import load_min_scene

DATA = "outputs/dataset_gen/01_tel_axisx_rest_T16"
SAMPLE = "sample_0000"
NFLOW = 7  # flow fields the teacher models => compare frames 1..7


def main() -> None:
    dev = "cuda:0"
    scene = load_min_scene(os.path.join(DATA, "scene_cache.pt"), device=dev)
    meta = json.load(open(os.path.join(DATA, SAMPLE, "sample.json")))
    v0_vec = torch.tensor(meta["v0"], dtype=torch.float32, device=dev)  # [3]
    E = float(meta["E"])
    print(f"scene: {scene.sim_xyzs.shape[0]} particles, "
          f"{int(scene.query_mask.sum())} moving | v0={meta['v0']} E={E:g} scale={scene.scale:.4f}")

    cfg = SimConfig(num_frames=16, substep=64)  # the dataset_gen config for this pack
    roll = MpmRollout(scene, cfg, requires_grad=True, device=dev)

    # v0 as an optimisable leaf on the MOVING particles (frozen stay 0).
    v0 = make_constant_v0(scene, v0_vec).clone().requires_grad_(True)  # [n,3]
    E_vec = torch.full((scene.sim_xyzs.shape[0],), E, device=dev)       # [n]

    gt = np.load(os.path.join(DATA, SAMPLE, "mpm_xyz.npy"))            # [16,n,3] WORLD
    shift = scene.shift
    qm = scene.query_mask.cpu().numpy()

    print("\nframe |  rolled-vs-GT world L2 (moving)  | max")
    errs = []
    last_pos = None
    for ti in range(NFLOW):
        # full-BPTT frame ti+1; rollout returns NORMALISED coords -> to world.
        pos_norm = roll.rollout_Evec(E_vec, ti, v0, grad_window=ti + 1)  # [n,3] normalised
        last_pos = pos_norm
        world = (pos_norm * scene.scale - shift).detach().cpu().numpy()  # [n,3]
        gt_w = gt[ti + 1]                                                # [n,3]
        d = np.linalg.norm(world[qm] - gt_w[qm], axis=-1)               # [n_moving]
        errs.append(d.mean())
        print(f"  {ti + 1:2d}  |  mean {d.mean():.6e}              | {d.max():.6e}")

    motion = np.linalg.norm(gt[NFLOW][qm] - gt[0][qm], axis=-1).mean()
    print(f"\nGT motion scale @f{NFLOW} (mean |disp| world) = {motion:.6e}")
    print(f"rollout error / motion = {np.mean(errs) / max(motion, 1e-9):.3%}")

    # gradient check: does d(last position)/d(v0) flow back to the moving leaf?
    loss = (last_pos ** 2).sum()
    loss.backward()
    g = v0.grad                                                         # [n,3]
    qm_t = scene.query_mask
    print(f"\ngrad d(pos^2)/d(v0): moving |grad| mean={g[qm_t].norm(dim=-1).mean():.4e} "
          f"max={g[qm_t].norm(dim=-1).max():.4e} | frozen |grad| max={g[~qm_t].norm(dim=-1).max():.4e}")
    print("OK: differentiable no-3DGS rollout works" if g[qm_t].abs().sum() > 0
          else "FAIL: no gradient to v0")


if __name__ == "__main__":
    main()
