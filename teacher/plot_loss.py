"""Plot a training loss curve from a metrics.csv (step,epoch,loss,peak_mem_gb).
Static PNG is appropriate here (a curve, not motion)."""
import argparse
import csv
import os
from typing import List

import numpy as np


def load_metrics(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read metrics.csv -> (steps [N], losses [N]) as 1-D float arrays."""
    steps: List[int] = []
    losses: List[float] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
    return np.asarray(steps, dtype=np.int64), np.asarray(losses, dtype=np.float64)


def plot(metrics_csv: str, out_png: str, title: str) -> str:
    """Render loss-vs-step (log-y, raw + moving average) to out_png; return path."""
    steps, losses = load_metrics(metrics_csv)          # steps [N], losses [N]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k: int = max(1, len(losses) // 200)
    ma: np.ndarray = np.convolve(losses, np.ones(k) / k, mode="valid")  # [N-k+1]
    plt.figure(figsize=(9, 4.5))
    plt.plot(steps, losses, alpha=0.25, label="raw")
    plt.plot(steps[len(steps) - len(ma):], ma, label=f"MA({k})", lw=2)
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("l2 loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()
    return out_png


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="loss")
    a = ap.parse_args()
    p = plot(a.metrics, a.out, a.title)
    s, l = load_metrics(a.metrics)
    print(f"{p}  | steps={s[-1]} final_loss={l[-1]:.5f} min_loss={l.min():.5f} n={len(l)}")


if __name__ == "__main__":
    main()
