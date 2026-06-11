"""Recover an initial-velocity field v0 from a video, with E held KNOWN.

The dual of recover.recover_field_E: there the v0 was known and E optimised; here E
is the known constant and a `V0Field` (global | voxel | triplane) is optimised. The
loss / per-frame-backward / render machinery is identical, with one CRITICAL change:

  v0 only receives gradient when a frame is rolled out with FULL BPTT to t=0.
  reset_state sets init_velocity at t=0, then the truncated-BPTT detached prefix
  (`extra_no_grad_steps`) runs; any detached prefix severs v0's grad. So unlike
  E-recovery (which gets a valid grad with grad_window=1 because E enters every
  taped substep), v0-recovery rolls every windowed frame with grad_window=ti+1
  (extra=0). With gravity off, the early frames are pure-v0, so a short window is
  both cheap (few taped substeps) and the most informative -- the PhysDreamer
  curriculum insight ("optimise velocity on a short window first").
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .config import SimConfig
from .mpm_rollout import MpmRollout
from .scene import SceneBundle
from .sim_render import render_disp_frame
from .v0field import V0Field


def recover_v0(
    scene: SceneBundle,
    gt: torch.Tensor,
    cfg: SimConfig,
    cam,
    E: float,
    *,
    field: V0Field,
    iters: int = 120,
    lr: float = 0.05,
    window: int = 2,
    window_start: Optional[int] = None,
    reg_weight: float = 0.0,
    grad_clip: Optional[float] = 10.0,
    weight_decay: float = 0.0,
    gt_v0: Optional[Sequence[float]] = None,
    cosine: bool = True,
    device: str = "cuda:0",
) -> dict:
    """Recover a v0 FIELD (E known) by photometric matching with full-BPTT frames.

    Args:
        gt:        [T,C,H,W] ground-truth frames in [0,1].
        E:         known global Young's modulus (held fixed, broadcast to E_vec[n]).
        field:     V0Field producing v0[n,3]; optimised in place.
        window:    max number of frames (from t=1) summed in the loss; each is rolled
                   out with FULL BPTT (grad_window=ti+1) so v0 gets gradient.
        window_start: if set, the loss window GROWS linearly window_start -> window
                   over training (PhysDreamer curriculum: identify the spatial field
                   from accumulated multi-frame motion). None => fixed `window`.
        weight_decay: AdamW weight decay (PhysDreamer uses 1e-4 on the velocity field).
        reg_weight: TV smoothness weight on the field (0 disables; off for "global").
        grad_clip: max grad-norm on the field params (PhysDreamer clips velocity
                   loosely); None disables.
        gt_v0:     known GT constant v0 [3] for error reporting (phase-A analog).
    Returns dict with loss_traj, per-iter recovered-v0 mean/components, final v0[n,3].
    """
    window = min(window, cfg.num_frames - 1)
    roll = MpmRollout(scene, cfg, requires_grad=True, device=device)
    rest_pos = scene.sim_xyzs.detach()                          # [n,3]
    n = rest_pos.shape[0]
    qmask = scene.query_mask                                    # [n] bool
    E_vec = torch.full((n,), float(E), device=device)           # [n] known, no grad
    field = field.to(device)

    def step_grads(w_cur: int) -> tuple:
        """Returns (photometric_loss, reg_loss); tracked separately so the photo term
        is comparable across reg settings (mirrors recover_field_E.step_grads).
        `w_cur` is the current (possibly growing) loss window."""
        photo = 0.0
        for ti in range(w_cur):
            v0 = field.v0_vec(rest_pos, qmask)                  # [n,3], fresh graph
            # full BPTT to t=0 (extra=0) -> v0 carries gradient from this frame.
            pos = roll.rollout_Evec(E_vec, ti, v0, grad_window=ti + 1)
            l = F.mse_loss(render_disp_frame(scene, pos, cam), gt[[ti + 1]]) / w_cur
            l.backward()
            photo += float(l.item())
        reg = 0.0
        if reg_weight > 0:
            rl = reg_weight * field.regularization()
            if rl.requires_grad:            # "global" has no spatial TV -> const 0
                rl.backward()
            reg = float(rl.item())
        return photo, reg

    opt = torch.optim.AdamW(field.parameters(), lr=lr, weight_decay=weight_decay)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, iters),
             eta_min=lr * 0.05) if cosine else None)

    def cur_window(it: int) -> int:
        """Growing-window schedule: window_start -> window linearly over iters."""
        if window_start is None:
            return window
        frac = it / max(1, iters - 1)
        return int(round(window_start + frac * (window - window_start)))

    gt_vec = None if gt_v0 is None else np.asarray(gt_v0, dtype=np.float64)  # [3]
    losses, regs, means, mags = [], [], [], []
    for _it in range(iters):
        opt.zero_grad()
        photo, reg = step_grads(cur_window(_it))
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(field.parameters(), grad_clip,
                                           error_if_nonfinite=False)
        opt.step()
        if sched is not None:
            sched.step()
        with torch.no_grad():
            v = field.v0_vec(rest_pos, qmask)[qmask]            # [m,3] moving only
            mean = v.mean(0).detach().cpu().numpy()             # [3]
            means.append(mean.tolist())
            mags.append(float(v.norm(dim=-1).mean()))
        losses.append(photo)
        regs.append(reg)

    with torch.no_grad():
        v0_final = field.v0_vec(rest_pos, qmask).detach().cpu().numpy()  # [n,3]
    v_moving = v0_final[qmask.detach().cpu().numpy()]           # [m,3]
    recovered_mean = v_moving.mean(0)                           # [3]
    out = {
        "loss_traj": losses, "reg_traj": regs,
        "v0_mean_traj": means, "v0_mag_traj": mags,
        "v0_final": v0_final, "recovered_mean_v0": recovered_mean.tolist(),
        "v0_moving_std": v_moving.std(0).tolist(),
        "final_loss": losses[-1], "min_loss": float(np.min(losses)),
        "window": window, "kind": field.kind, "reg_weight": reg_weight, "E": float(E),
    }
    if gt_vec is not None:
        err = recovered_mean - gt_vec                           # [3]
        l2 = float(np.linalg.norm(err))
        gt_norm = float(np.linalg.norm(gt_vec))
        cos = (float(np.dot(recovered_mean, gt_vec) / (np.linalg.norm(recovered_mean)
               * gt_norm + 1e-12)) if gt_norm > 1e-9 else float("nan"))
        out.update({
            "gt_v0": gt_vec.tolist(),
            "v0_abs_err": np.abs(err).tolist(),
            "v0_l2_err": l2,
            "v0_rel_err": l2 / (gt_norm + 1e-12),
            "v0_mag_err": abs(float(np.linalg.norm(recovered_mean)) - gt_norm),
            "v0_cos": cos,
            "v0_angle_deg": float(np.degrees(np.arccos(np.clip(cos, -1, 1)))),
        })
    return out


def plot_v0_recovery(path: str, result: dict, title: str = "") -> None:
    """Three panels: loss vs iter (THE objective) + recovered v0 components vs iter
    (with GT dashed lines if known) + final per-particle |v0| histogram."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[recover_v0] plot skipped: {e}")
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(result["loss_traj"], "o-", ms=3)
    ax[0].set_yscale("log"); ax[0].set_xlabel("iter"); ax[0].set_ylabel("photometric loss")
    ax[0].set_title("loss (objective)")

    means = np.asarray(result["v0_mean_traj"])                  # [iters,3]
    cols = ["tab:red", "tab:green", "tab:blue"]
    for c in range(3):
        ax[1].plot(means[:, c], color=cols[c], label=f"v{'xyz'[c]}")
    gt = result.get("gt_v0")
    if gt is not None:
        for c in range(3):
            ax[1].axhline(gt[c], color=cols[c], ls="--", lw=1)
    ax[1].set_xlabel("iter"); ax[1].set_ylabel("recovered v0 (moving mean)")
    rec = result["recovered_mean_v0"]
    ax[1].set_title(f"v0=[{rec[0]:.2f},{rec[1]:.2f},{rec[2]:.2f}]"
                    + (f"  err={result.get('v0_l2_err', float('nan')):.3f}"
                       if gt is not None else ""))
    ax[1].legend(fontsize=8)

    mag = np.linalg.norm(np.asarray(result["v0_final"]), axis=-1)  # [n]
    ax[2].hist(mag[mag > 1e-6], bins=40)
    ax[2].set_xlabel("|v0| per particle (moving)"); ax[2].set_ylabel("count")
    ax[2].set_title("final v0 magnitude distribution")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def v0_field_vs_gt_metrics(v0_final: np.ndarray, v0_gt: np.ndarray,
                           query_mask: np.ndarray) -> dict:
    """Per-particle recovery score of a v0 field vs a known GT field (phase B).

    Both [n,3]; scored over the MOVING (query) particles only. Reports the
    per-particle RMSE of v0, the per-component RMSE, and the per-component Pearson
    correlation of the spatial pattern -- i.e. did the field recover the SHAPE of the
    spatial ramp (not just the mean, which a uniform v0 already gets).
    """
    m = query_mask.astype(bool)
    f, g = v0_final[m], v0_gt[m]                                # [k,3]
    rmse = float(np.sqrt(((f - g) ** 2).sum(-1).mean()))
    per_comp_rmse = np.sqrt(((f - g) ** 2).mean(0)).tolist()
    corrs = []
    for c in range(3):
        if f[:, c].std() > 1e-9 and g[:, c].std() > 1e-9:
            corrs.append(float(np.corrcoef(f[:, c], g[:, c])[0, 1]))
        else:
            corrs.append(float("nan"))
    return {"v0_per_particle_rmse": rmse, "v0_per_comp_rmse": per_comp_rmse,
            "v0_per_comp_corr": corrs}


def plot_v0_field_vs_gt(path: str, sim_xyzs: np.ndarray, v0_final: np.ndarray,
                        v0_gt: np.ndarray, query_mask: np.ndarray,
                        title: str = "") -> None:
    """GT vs recovered v0 field: per-component scatter (recovered vs GT, ideal=diag)
    plus the dominant component coloured spatially side-by-side (shared scale)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[recover_v0] field-vs-gt plot skipped: {e}")
        return
    m = query_mask.astype(bool)
    f, g = v0_final[m], v0_gt[m]                                # [k,3]
    xyz = sim_xyzs[m]
    dom = int(np.argmax(g.std(0)))                             # most-varying component
    fig = plt.figure(figsize=(15, 4))
    ax0 = fig.add_subplot(1, 3, 1)
    cols = ["tab:red", "tab:green", "tab:blue"]
    for c in range(3):
        ax0.scatter(g[:, c], f[:, c], s=2, alpha=0.3, color=cols[c], label=f"v{'xyz'[c]}")
    lim = [min(g.min(), f.min()), max(g.max(), f.max())]
    ax0.plot(lim, lim, "k--", lw=1)
    ax0.set_xlabel("GT v0"); ax0.set_ylabel("recovered v0")
    ax0.set_title("per-particle v0 (ideal=diag)"); ax0.legend(fontsize=8)
    vmin = float(min(g[:, dom].min(), f[:, dom].min()))
    vmax = float(max(g[:, dom].max(), f[:, dom].max()))
    for i, (vals, name) in enumerate([(g[:, dom], "GT"), (f[:, dom], "recovered")]):
        ax = fig.add_subplot(1, 3, 2 + i, projection="3d")
        p = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=vals, s=3,
                       cmap="coolwarm", vmin=vmin, vmax=vmax)
        fig.colorbar(p, ax=ax, shrink=0.6)
        ax.set_title(f"{name} v{'xyz'[dom]}")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_v0_quiver(path: str, sim_xyzs: np.ndarray, v0_final: np.ndarray,
                   title: str = "") -> None:
    """3D quiver of the recovered per-particle v0 (subsampled for legibility)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[recover_v0] quiver skipped: {e}")
        return
    mag = np.linalg.norm(v0_final, axis=-1)                     # [n]
    moving = mag > 1e-6
    xyz = sim_xyzs[moving]; v = v0_final[moving]
    if xyz.shape[0] > 2000:                                     # subsample for speed
        sel = np.random.RandomState(0).choice(xyz.shape[0], 2000, replace=False)
        xyz, v = xyz[sel], v[sel]
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.quiver(xyz[:, 0], xyz[:, 1], xyz[:, 2], v[:, 0], v[:, 1], v[:, 2],
              length=0.3, normalize=False, color="tab:blue", linewidth=0.5)
    ax.set_title(title or "recovered v0 field")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
