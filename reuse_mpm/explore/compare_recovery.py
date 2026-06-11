"""Entrypoint: overlay the loss curves of several recovery runs on one plot.

The A/B for "is a multi-parameter E FIELD easier to fit than a single global
scalar?" comes down to comparing loss-vs-iter across runs that share a GT and an
init. This reads each run dir's trace.json (loss; field runs also carry E_geomean)
and metrics.json (label/backbone/final), and tiles them onto one figure -- the
loss curve is the deciding artifact (research-discipline: loss first).

No GPU. CPU-only matplotlib.

  python -m reuse_mpm.explore.compare_recovery \
      --runs outputs/train_global_E/NN outputs/train_field_E/MM outputs/train_field_E/KK

Output dir (auto, outputs/explore/compare_recovery/NN/):
  config.json   resolved config
  compare.png   loss-vs-iter (log y) for every run, + geomean-E-vs-iter panel
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import tyro


@dataclass
class CompareConfig:
    """explore.compare_recovery config (local; not in the single-source config.py)."""

    runs: List[str] = field(default_factory=list)  # run dirs to overlay
    out: Optional[str] = None
    run_label: str = ""


def _label(run_dir: str) -> str:
    """A short legend label from a run's metrics.json (backbone/params) + dir name."""
    base = os.path.basename(os.path.normpath(run_dir))
    m = os.path.join(run_dir, "metrics.json")
    if os.path.exists(m):
        d = json.load(open(m))
        if "backbone" in d:
            return f"{base} [{d['backbone']} {d.get('n_field_params','?')}p]"
        return f"{base} [scalar]"
    return base


def run(cfg: CompareConfig):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..run_io import RunDir

    assert cfg.runs, "pass --runs <dir> [<dir> ...]"
    rd = RunDir.create(__name__, cfg.run_label, cfg.out, config=cfg)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for run_dir in cfg.runs:
        tp = os.path.join(run_dir, "trace.json")
        if not os.path.exists(tp):
            print(f"[compare] skip {run_dir}: no trace.json")
            continue
        tr = json.load(open(tp))
        lab = _label(run_dir)
        ax[0].plot(tr["loss"], "-", lw=1.5, label=lab)
        # field runs: E_geomean; scalar run: E (the recovered scalar per iter)
        e = tr.get("E_geomean") or tr.get("E")
        if e is not None:
            ax[1].plot(e, "-", lw=1.5, label=lab)

    ax[0].set_yscale("log"); ax[0].set_xlabel("iter"); ax[0].set_ylabel("photometric loss")
    ax[0].set_title("loss vs iter (the A/B objective)"); ax[0].legend(fontsize=8)
    ax[1].set_yscale("log"); ax[1].set_xlabel("iter"); ax[1].set_ylabel("E (geomean)")
    ax[1].set_title("recovered E vs iter"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(rd.path("compare.png"), dpi=120)
    plt.close(fig)
    rd.finish()
    print(f"[compare] {len(cfg.runs)} runs -> {rd.path('compare.png')}")
    return rd


if __name__ == "__main__":
    run(tyro.cli(CompareConfig))
