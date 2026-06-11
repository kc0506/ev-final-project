"""TEMPORARY: smoke-test the `entrypoint` decorator (no GPU, no torch).

    python -m reuse_mpm._entrypoint_demo --n 3
    python -m reuse_mpm._entrypoint_demo --crash   # exercise the finally-seal path

Verifies the decorator handles create / capture_output / finish so the body is
just `def run(cfg, rd): ...`. Delete once the pattern is rolled out for real.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tyro

from .run_io import RunDir, entrypoint


@dataclass
class DemoConfig:
    n: int = 3                       # how many fake "frames" to write
    crash: bool = False              # raise mid-body to test the finally-seal
    run_label: str = ""              # picked up by the decorator's default label
    out: Optional[str] = None        # escape hatch (decorator passes to create)


class DemoRun(RunDir):
    """Demo deliverables: config.json, console.log, data.npy, result.json."""

    def result(self, **obj) -> None:
        self.write_json("result.json", obj)


@entrypoint(DemoRun, label=lambda c: c.run_label or f"n{c.n}")
def run(cfg: DemoConfig, rd: DemoRun) -> None:
    # plain python print -> should land in console.log via the FD tee
    print(f"[demo] starting, n={cfg.n}")
    # subprocess / C-ext-style output -> proves the tee is FD-level, not just print
    os.system('echo "[demo] hello from a subprocess (fd-level tee check)"')

    arr = np.arange(cfg.n, dtype=np.float32)
    np.save(rd.path("data.npy"), arr)  # top-level file written OUTSIDE a RunDir method
    rd.note(f"wrote data.npy with {cfg.n} values")

    if cfg.crash:
        raise RuntimeError("intentional crash to test finally-seal")

    rd.result(n=cfg.n, sum=float(arr.sum()))
    print(f"[demo] done -> {rd.root}")


if __name__ == "__main__":
    run(tyro.cli(DemoConfig))
