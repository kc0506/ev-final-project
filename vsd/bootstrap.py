"""Import-order guard for the VSD pipeline. Import this FIRST (before any
`reuse_mpm` / `physdreamer` import) so the dependency surface resolves locally:

  - prepend the simple_knn stub to sys.path (no CUDA ext build needed for flow)
  - prepend the repo root so `import reuse_mpm` works from anywhere
  - point PHYSDREAMER_ROOT at the local checkout (the _env.py default is a meow2 path)

Usage:
    import vsd.bootstrap  # noqa: F401  (side-effecting)
    from reuse_mpm.config import SimConfig
"""
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS)
_STUBS = os.path.join(_THIS, "_stubs")
_PHYSDREAMER = os.environ.get(
    "PHYSDREAMER_ROOT", "/home/kc0506/main/meow2/PhysDreamer"
)

os.environ["PHYSDREAMER_ROOT"] = _PHYSDREAMER

if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Expose a stub ONLY for a 3DGS CUDA ext that is not really installed, so a real
# build (for RGB render) is never shadowed by the flow-path stubs.
import importlib.util as _u  # noqa: E402


def _real(mod: str) -> bool:
    try:
        return _u.find_spec(mod) is not None
    except Exception:
        return False


for _stub in ("simple_knn", "diff_gaussian_rasterization"):
    _stub_pkg = os.path.join(_STUBS, _stub)
    if not _real(_stub) and os.path.isdir(_stub_pkg):
        if _STUBS not in sys.path:
            sys.path.insert(0, _STUBS)


def physdreamer_root() -> str:
    """Return the resolved local PhysDreamer checkout path."""
    return _PHYSDREAMER
