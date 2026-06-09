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

from .sim_render import (
    SimConfig, build_mpm, render_disp_frame, simulate_and_render,
)
from .diff_sim import MPMDifferentiableSimulation
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
    n = scene.sim_xyzs.shape[0]
    init_xyzs = scene.sim_xyzs.clone()
    density = torch.ones_like(init_xyzs[..., 0]) * cfg.density
    dmask = torch.ones_like(density).int()
    onev = torch.ones(n, device=device)
    nu_t = torch.tensor(float(cfg.nu), device=device)
    ss = cfg.substep_size
    window = min(window, cfg.num_frames - 1)
    solver, state, model = build_mpm(scene, cfg, requires_grad=True)

    # [question] So window is not actually "chunk" - you only use first `window` frame?

    def step_grads(logE: torch.Tensor) -> float:
        tot = 0.0
        for ti in range(window):
            extra = max(0, (ti + 1 - grad_window) * cfg.substep)  # 0 => full BPTT
            num_grad = cfg.substep * (ti + 1) - extra  # [note] this is actually correct. `num_substeps + extra == total subusteps`
            E_vec = (10.0 ** logE) * onev
            # [note] ss = sub dt
            pos = MPMDifferentiableSimulation.apply(
                solver, state, model, 0, ss, num_grad,
                init_xyzs, v0, E_vec, nu_t, density, dmask, None,
                device, True, extra)
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
