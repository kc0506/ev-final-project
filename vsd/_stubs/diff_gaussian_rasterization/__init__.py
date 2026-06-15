"""Stub for diff_gaussian_rasterization. PhysDreamer's render.py / flow_depth_render.py
import GaussianRasterizationSettings + GaussianRasterizer at module load; the MPM->flow
VSD path imports those modules transitively (via local_utils) but NEVER renders gaussians,
so neither class is constructed. Instantiating raises loudly. Build the real ext (Task:
env debt) for any RGB / 3DGS-render path."""
from typing import Any

_MSG = ("diff_gaussian_rasterization stub: the real CUDA ext is not built. The flow "
        "VSD path must not render gaussians. Build the ext for RGB/3DGS rendering.")


class GaussianRasterizationSettings:  # noqa: D101
    def __init__(self, *a: Any, **k: Any) -> None:
        raise NotImplementedError(_MSG)


class GaussianRasterizer:  # noqa: D101
    def __init__(self, *a: Any, **k: Any) -> None:
        raise NotImplementedError(_MSG)
