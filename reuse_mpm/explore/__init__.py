"""One-shot exploration / diagnostic scripts (kept for the record, re-runnable).

These answered a specific question once and are NOT part of the canonical
pipeline; they import the canonical routines (recover_global_E, MpmRollout,
load_from_spec) rather than re-implementing them:

  probe_identifiability  -- is E recoverable, over what range? (loss landscape)
  recovery_sweep         -- does a far init converge? (GT x init cross product)
  multiscene_fwdbwd      -- does recovery generalise across scenes? (smoke test)
  gradcheck              -- where does the gradient break? (trajectory vs pixel)

Run as e.g.  python -m reuse_mpm.explore.probe_identifiability ...
They keep their argparse CLIs (one-shot diagnostics; not part of the
config-dataclass handshake that the canonical entrypoints share).
"""
