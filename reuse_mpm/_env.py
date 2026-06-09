"""Centralised PhysDreamer dependency wiring.

We treat the (messy but working) PhysDreamer checkout as a third-party library and
bind to a SINGLE source of truth:
  - `physdreamer.*`            (properly pip-installed package)
  - `projects/inference/local_utils`  (the render / field helpers)

We deliberately do NOT touch PhysDreamer internals and do NOT import the
`motionrep` namespace (it only resolves via cwd/sys.path hacks under
`projects/uncleaned_train`). Importing this module makes the whole dependency
surface available with one `import reuse_mpm._env`.
"""
import os
import sys

PHYSDREAMER_ROOT = os.environ.get(
    "PHYSDREAMER_ROOT", "/tmp2/b10401006/PhysDreamer"
)
_INFERENCE_DIR = os.path.join(PHYSDREAMER_ROOT, "projects", "inference")

for _p in (PHYSDREAMER_ROOT, _INFERENCE_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

# Re-export the exact symbols our pipeline depends on, so callers never have to
# know where they live.
from physdreamer.gaussian_3d.scene import GaussianModel  # noqa: E402
from physdreamer.data.cameras import Camera  # noqa: E402
from physdreamer.data.datasets.multiview_dataset import (  # noqa: E402
    MultiviewImageDataset,
)
from physdreamer.warp_mpm.mpm_data_structure import (  # noqa: E402
    MPMStateStruct,
    MPMModelStruct,
)
from physdreamer.warp_mpm.mpm_solver_diff import MPMWARPDiff  # noqa: E402
from physdreamer.warp_mpm.gaussian_sim_utils import get_volume  # noqa: E402

from local_utils import (  # noqa: E402
    find_far_points,
    apply_grid_bc_w_freeze_pts,
    downsample_with_kmeans_gpu,
    downsample_with_kmeans_gpu_with_chunk,
    render_gaussian_seq_w_mask_with_disp,
)

__all__ = [
    "PHYSDREAMER_ROOT",
    "GaussianModel",
    "Camera",
    "MultiviewImageDataset",
    "MPMStateStruct",
    "MPMModelStruct",
    "MPMWARPDiff",
    "get_volume",
    "find_far_points",
    "apply_grid_bc_w_freeze_pts",
    "downsample_with_kmeans_gpu",
    "downsample_with_kmeans_gpu_with_chunk",
    "render_gaussian_seq_w_mask_with_disp",
]
