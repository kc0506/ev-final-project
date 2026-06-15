"""Plot the teacher's implied marginal q(vx) from the prob-flow ODE log p, vs the source
ramp p(vx) ∝ vx. log q(vx) = log p_teacher(x0(vx)) + log|dx0/dvx| (up to an additive
constant -- only the SHAPE is meaningful, since absolute log p is background-confounded).

  micromamba run -n physdreamer python -m vsd.plot_logp vsd/out/logp_out04.json
"""
import argparse
import json
import os

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json", help="output of logp_ode_modal.py")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = json.load(open(args.json))
    pts = d["points"]
    vmin, vmax = d["meta"]["vmin"], d["meta"]["vmax"]
    vx = np.array([p["vx"] for p in pts])                       # [n]
    logp = np.array([p["logp"] for p in pts])                   # [n] ambient log density
    logjac = np.array([p["logjac"] for p in pts])               # [n] log|dx0/dvx|
    logq = logp + logjac                                        # [n] implied log marginal over vx

    # exp + normalise on the vx grid (shape only; absolute offset is meaningless)
    logq_rel = logq - logq.max()
    q = np.exp(logq_rel)                                        # [n]
    q = q / np.trapz(q, vx)                                     # normalise to integrate to 1
    ramp = 2 * vx / (vmax ** 2 - vmin ** 2)                     # source p(vx) ∝ vx on [vmin,vmax]

    out = args.out or os.path.splitext(args.json)[0] + ".png"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(vx, logp, "-o", ms=3, label="log p_teacher (ambient)")
    ax[0].plot(vx, logq, "-s", ms=3, label="log q = log p + log|dx0/dvx|")
    ax[0].set_title("raw log-likelihood (relative)"); ax[0].set_xlabel("vx"); ax[0].legend(fontsize=8)
    ax[1].plot(vx, q, "-o", ms=4, label="teacher q(vx) (normalised)")
    ax[1].plot(vx, ramp, "r--", label="source ramp p(vx)∝vx")
    ax[1].set_title("implied marginal vs source ramp"); ax[1].set_xlabel("vx"); ax[1].legend(fontsize=8)
    x1 = np.array([p["x1_sq_over_D"] for p in pts])
    ax[2].plot(vx, x1, "-o", ms=3); ax[2].axhline(1.0, color="g", ls="--", label="ideal ~1.0")
    ax[2].set_title("calibration  ||x1||²/D  (ODE -> N(0,I)?)"); ax[2].set_xlabel("vx"); ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    print(f"saved {out}")
    print(f"calibration ||x1||²/D: [{x1.min():.2f}, {x1.max():.2f}] (want ~1.0)")


if __name__ == "__main__":
    main()
