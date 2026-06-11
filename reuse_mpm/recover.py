"""Single shared global-E recovery routine + plot.

Both `train_global_E` and `multiscene_fwdbwd` call this, so there is ONE recovery
implementation (no duplicated loop, no feature drift like the dropped loss curve).

Method (the chunked one that actually moves): per-particle constant E (validated
grad path), truncated BPTT (`grad_window` frames carry gradient), per-frame
forward+backward (state is mutated between frames), Adam on log10(E) with optional
cosine decay. Optional coarse grid pre-init.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .config import SimConfig
from .efield import EField
from .sim_render import render_disp_frame, simulate_and_render
from .mpm_rollout import MpmRollout
from .scene import SceneBundle


def recover_global_E(
    scene: SceneBundle,
    gt: torch.Tensor,
    cfg: SimConfig,
    cam,
    v0: torch.Tensor,
    *,
    init_E: float,
    iters: int = 64,
    lr: float = 0.05,
    window: int = 3,
    grad_window: int = 1,
    coarse_init: bool = False,
    coarse_n: int = 9,
    true_E: Optional[float] = None,
    cosine: bool = True,
    device: str = "cuda:0",
) -> dict:
    window = min(window, cfg.num_frames - 1)
    roll = MpmRollout(scene, cfg, requires_grad=True, device=device)

    # `window` = number of frames (from t=1) whose photometric loss we sum each
    # iter; each target frame is rolled out independently from the initial state
    # (per-frame BPTT), with truncated grad controlled by `grad_window`.
    def step_grads(logE: torch.Tensor) -> float:
        tot = 0.0
        for ti in range(window):
            pos = roll.rollout_to_frame(logE, ti, v0, grad_window)
            l = F.mse_loss(render_disp_frame(scene, pos, cam), gt[[ti + 1]]) / window
            l.backward()
            tot += float(l.item())
        return tot

    coarse = None
    if coarse_init and true_E is not None:
        cand = np.geomspace(true_E / 100, true_E * 100, coarse_n)
        best = (1e18, init_E)
        coarse = []
        for E in cand:
            v = simulate_and_render(scene, float(E), v0, cfg, cam).detach()
            m = F.mse_loss(v[1:window + 1], gt[1:window + 1]).item()
            coarse.append((float(E), m))
            if m < best[0]:
                best = (m, float(E))
        init_E = best[1]

    logE = torch.tensor(float(np.log10(init_E)), device=device, requires_grad=True)
    opt = torch.optim.Adam([logE], lr=lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, iters),
             eta_min=lr * 0.05) if cosine else None)

    Es, losses = [], []
    for _ in range(iters):
        opt.zero_grad()
        loss = step_grads(logE)
        opt.step()
        if sched is not None:
            sched.step()
        Es.append(float(10.0 ** logE.item()))
        losses.append(loss)

    recovered = float(np.mean(Es[-5:]))   # last-5 mean, robust to oscillation
    out = {"E_traj": Es, "loss_traj": losses, "recovered_E": recovered,
           "final_iter_E": Es[-1], "final_loss": losses[-1],
           "init_E": float(init_E), "window": window, "grad_window": grad_window,
           "coarse": coarse}
    if true_E is not None:
        out["true_E"] = float(true_E)
        out["rel_err"] = abs(recovered - true_E) / true_E
        out["log10_err"] = abs(np.log10(recovered) - np.log10(true_E))
    return out


def recover_field_E(
    scene: SceneBundle,
    gt: torch.Tensor,
    cfg: SimConfig,
    cam,
    v0: torch.Tensor,
    *,
    field: EField,
    iters: int = 80,
    lr: float = 0.05,
    window: int = 3,
    grad_window: int = 1,
    reg_weight: float = 0.0,
    true_E: Optional[float] = None,
    cosine: bool = True,
    device: str = "cuda:0",
) -> dict:
    """Recover a spatially-varying E FIELD (vs the global scalar in recover_global_E).

    Same loss / window / truncated-BPTT machinery, but the optimised parameter is
    `field` (an EField); the per-particle E_vec[n] is queried from it at the rest
    positions (`scene.sim_xyzs`) each frame, so gradient flows render -> MPM ->
    E_vec -> field params. An optional smoothness penalty (`reg_weight` * field TV)
    is added. `true_E` (if the GT was a global scalar) is only used to report how
    close the recovered field's geometric-mean E lands.

    Args:
        field: EField producing log10(E); optimised in place.
    Returns dict with loss_traj, per-iter E geomean/min/max, final per-particle E.
    """
    window = min(window, cfg.num_frames - 1)
    roll = MpmRollout(scene, cfg, requires_grad=True, device=device)
    rest_pos = scene.sim_xyzs.detach()                          # [n,3]
    field = field.to(device)

    def step_grads() -> tuple:
        """Returns (photometric_loss, reg_loss) -- tracked SEPARATELY so the
        photometric term is directly comparable to the scalar baseline (which has
        no reg); the TV reg must not contaminate the reported fit quality."""
        photo = 0.0
        for ti in range(window):
            # rebuild E_vec per frame (each frame does its own backward, so the
            # field's graph must be fresh -- mirrors rollout_to_frame recomputing
            # 10**logE per call).
            E_vec = field.E_vec(rest_pos)                       # [n]
            pos = roll.rollout_Evec(E_vec, ti, v0, grad_window)
            l = F.mse_loss(render_disp_frame(scene, pos, cam), gt[[ti + 1]]) / window
            l.backward()
            photo += float(l.item())
        reg = 0.0
        if reg_weight > 0:
            rl = reg_weight * field.regularization()
            rl.backward()
            reg = float(rl.item())
        return photo, reg

    opt = torch.optim.Adam(field.parameters(), lr=lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, iters),
             eta_min=lr * 0.05) if cosine else None)

    losses, reg_losses, geomeans, mins, maxs = [], [], [], [], []
    for _ in range(iters):
        opt.zero_grad()
        photo, reg = step_grads()
        opt.step()
        if sched is not None:
            sched.step()
        with torch.no_grad():
            E = field.E_vec(rest_pos)                           # [n]
            geomeans.append(float(torch.exp(torch.log(E).mean())))
            mins.append(float(E.min())); maxs.append(float(E.max()))
        losses.append(photo)         # photometric only (comparable to scalar)
        reg_losses.append(reg)

    with torch.no_grad():
        E_final = field.E_vec(rest_pos).detach().cpu().numpy()  # [n]
    recovered_geomean = float(np.exp(np.log(E_final).mean()))
    out = {"loss_traj": losses, "reg_traj": reg_losses, "E_geomean_traj": geomeans,
           "E_min_traj": mins, "E_max_traj": maxs,
           "E_final": E_final, "recovered_geomean_E": recovered_geomean,
           "final_loss": losses[-1], "min_loss": float(np.min(losses)),
           "window": window, "grad_window": grad_window,
           "backbone": field.backbone, "reg_weight": reg_weight}
    if true_E is not None:
        out["true_E"] = float(true_E)
        out["rel_err_geomean"] = abs(recovered_geomean - true_E) / true_E
        out["log10_err_geomean"] = abs(np.log10(recovered_geomean) - np.log10(true_E))
    return out


def plot_field_recovery(path: str, result: dict, true_E: Optional[float],
                        title: str = "") -> None:
    """Three panels: loss vs iter (objective) + E geomean/range vs iter + final
    per-particle E histogram."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[recover] field plot skipped: {e}")
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(result["loss_traj"], "o-", ms=3)
    ax[0].set_yscale("log"); ax[0].set_xlabel("iter"); ax[0].set_ylabel("photometric loss")
    ax[0].set_title("loss (objective)")

    ax[1].plot(result["E_geomean_traj"], "o-", ms=3, label="geomean E")
    ax[1].fill_between(range(len(result["E_min_traj"])), result["E_min_traj"],
                       result["E_max_traj"], alpha=0.2, label="min..max")
    if true_E is not None:
        ax[1].axhline(true_E, color="r", ls="--", label=f"true {true_E:.1e}")
    ax[1].set_yscale("log"); ax[1].set_xlabel("iter"); ax[1].set_ylabel("E")
    ax[1].set_title(f"E field (geomean {result['recovered_geomean_E']:.2e})")
    ax[1].legend(fontsize=8)

    ax[2].hist(np.log10(result["E_final"]), bins=40)
    if true_E is not None:
        ax[2].axvline(np.log10(true_E), color="r", ls="--")
    ax[2].set_xlabel("log10(E) per particle"); ax[2].set_ylabel("count")
    ax[2].set_title("final field distribution")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_field_scatter(path: str, sim_xyzs: np.ndarray, E_final: np.ndarray,
                       title: str = "") -> None:
    """3D scatter of particles coloured by recovered log10(E)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[recover] field scatter skipped: {e}")
        return
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    c = np.log10(E_final)
    p = ax.scatter(sim_xyzs[:, 0], sim_xyzs[:, 1], sim_xyzs[:, 2],
                   c=c, s=3, cmap="viridis")
    fig.colorbar(p, ax=ax, shrink=0.6, label="log10(E)")
    ax.set_title(title or "recovered E field")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def field_vs_gt_metrics(E_final: np.ndarray, E_gt: np.ndarray) -> dict:
    """Per-particle recovery score of a field against a known GT field (phase B).

    Both [n]. Compares in log10 space (E spans decades): RMSE of log10(E) and the
    Pearson correlation of the per-particle log10(E) -- i.e. did the field recover
    the SHAPE of the spatial variation, not just the mean.
    """
    lf, lg = np.log10(E_final), np.log10(E_gt)
    rmse = float(np.sqrt(np.mean((lf - lg) ** 2)))
    if lf.std() > 1e-9 and lg.std() > 1e-9:
        corr = float(np.corrcoef(lf, lg)[0, 1])
    else:
        corr = float("nan")
    return {"log10_rmse_vs_gt": rmse, "log10_corr_vs_gt": corr}


def plot_field_vs_gt(path: str, sim_xyzs: np.ndarray, E_final: np.ndarray,
                     E_gt: np.ndarray, title: str = "") -> None:
    """GT vs recovered field: scatter of per-particle log10(E), plus side-by-side
    spatial scatters coloured by log10(E) on a shared colour scale."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[recover] field-vs-gt plot skipped: {e}")
        return
    lf, lg = np.log10(E_final), np.log10(E_gt)
    vmin, vmax = float(min(lf.min(), lg.min())), float(max(lf.max(), lg.max()))
    fig = plt.figure(figsize=(15, 4))
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.scatter(lg, lf, s=3, alpha=0.4)
    lim = [vmin, vmax]
    ax0.plot(lim, lim, "r--", lw=1)
    ax0.set_xlabel("GT log10(E)"); ax0.set_ylabel("recovered log10(E)")
    ax0.set_title("per-particle E (ideal=diagonal)")
    for i, (vals, name) in enumerate([(lg, "GT"), (lf, "recovered")]):
        ax = fig.add_subplot(1, 3, 2 + i, projection="3d")
        p = ax.scatter(sim_xyzs[:, 0], sim_xyzs[:, 1], sim_xyzs[:, 2],
                       c=vals, s=3, cmap="viridis", vmin=vmin, vmax=vmax)
        fig.colorbar(p, ax=ax, shrink=0.6)
        ax.set_title(f"{name} log10(E)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_recovery(path: str, result: dict, true_E: float, title: str = "") -> None:
    """Two panels: loss vs iter (THE objective) + E vs iter."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[recover] plot skipped: {e}")
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(result["loss_traj"], "o-", ms=3)
    ax[0].set_yscale("log"); ax[0].set_xlabel("iter"); ax[0].set_ylabel("photometric loss")
    ax[0].set_title("loss (objective)")
    ax[1].plot(result["E_traj"], "o-", ms=3, label="recovered E")
    ax[1].axhline(true_E, color="r", ls="--", label=f"true {true_E:.1e}")
    ax[1].set_yscale("log"); ax[1].set_xlabel("iter"); ax[1].set_ylabel("E")
    ax[1].set_title(f"E: {result['recovered_E']:.2e} "
                    f"(rel_err {result.get('rel_err', float('nan')) * 100:.0f}%)")
    ax[1].legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
