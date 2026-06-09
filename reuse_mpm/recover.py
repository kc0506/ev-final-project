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
