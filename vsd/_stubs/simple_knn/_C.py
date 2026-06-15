"""Stub for simple_knn._C. Only `distCUDA2` is imported by PhysDreamer; it is never
called on the flow path. Raise loudly if anything actually invokes it, so a silent
wrong-result can never happen."""
from typing import Any


def distCUDA2(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "simple_knn stub: distCUDA2 is not built. The MPM->flow VSD path must never "
        "call it (it only renders particle screen-flow, not gaussians). If you hit "
        "this, you're on an RGB / scene-from-PLY path -- build the real simple_knn ext."
    )
