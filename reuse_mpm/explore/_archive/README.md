# explore/_archive

Superseded one-shot f0 explore scripts, kept for reference + git history rather than
deleted. Each was verified (by reading) to be fully covered by a surviving canonical
entrypoint, or to be a probe whose conclusion is already recorded (memory / report).

**Import caveat:** these moved one package level deeper, so their relative imports
(`from ..gpu`, `from .._env`, `from ..config`, `from ._block`) now point one level
short. They are *not runnable in place* — to rerun, `git mv` the file back to
`explore/`, or bump each `..`→`...` and `.`→`..`. Frozen references, not maintained.

## Canonical f0 entrypoints (what survives)

Block synthetic lab (`_block.Scene` + `_viz`):
- `f0_forward_viz` — see a block scene (release/drop/squeeze/uniform); single-E / multi-R / multi-E
- `f0_fit_case` — recover scalar E (scene × loss × init)
- `f0_loss_landscape` — block E loss-landscape (time/spectral/combined × K)
- `f0_train_S` — recover global F0 = expm(S) (6-DOF)
- `f0_train_ufield` — recover coarse F0 field I+∇u (15-DOF)

Telephone / gic bridge (production `load_from_spec` SceneBundle):
- `f0_snapshot` — produce a telephone F0 snapshot .pt (consumed by the landscapes/dump)
- `f0_e_landscape` — telephone fix-F0 E identifiability (v0-driven vs F0-release)
- `f0_alpha_landscape` — telephone (α, logE) degeneracy 2D
- `f0_release_dump` — export warp GT trajectory + F0 bundle for gic

Libs (not entrypoints): `_block.py` (Scene + SCENES), `_viz.py` (panels + triplane gifs).

(`f0_gradu_viz` kept in `explore/`: the only ∇u-field forward viz, likely wanted for the
fine-field MLP rung. Its viz could later fold into `f0_train_ufield --no-fit`.)

## Archived

| archived | superseded by | why |
|----------|---------------|-----|
| `f0_block_fit.py` | `f0_fit_case --scene release` | strict subset: same FD+Adam scalar-E fit, same 3 losses, same checkpoint format. |
| `f0_block_landscape.py` | `f0_loss_landscape` | early "is block's E-well sharper than telephone?" probe (recorded). Unique chamfer + loss/motion² doesn't fit loss_landscape's cached-1D design. |
| `f0_spectral_K_probe.py` | `f0_loss_landscape` (K rows + Nyquist) | tested the "init period gates K" hypothesis — **retracted** (real effect = local-bump position). K-sweep already in loss_landscape. |
| `f0_dynamic_pull.py` | `f0_forward_viz --scene release` | release forward viz; same upstream (Scene pull→snapshot), now drawn via shared `_viz`. |
| `f0_asym_squeeze.py` | `f0_forward_viz --scene squeeze` | asymmetric-squeeze forward viz; = the `squeeze` scene. |
| `f0_squeeze_forward_viz.py` | `f0_forward_viz --release-frames ...` | release forward at several R (multi-R overlay mode). |
| `f0_block_E_overlay.py` | `f0_forward_viz --e-list ...` | release at several E + divergence (multi-E overlay mode). |
| `f0_block_freqloss.py` | `f0_loss_landscape` / `f0_fit_case` | 4-loss (time/spec/period/centroid) landscape probe; conclusion recorded (naive spectral L2 saturates; period/centroid monotone-but-broad; spectral chosen). |
| `f0_block_squeeze_sweep.py` | `f0_fit_case` (per R) | E-fit swept over F0 amplitude R∈{2,3,5}; finding "loss conclusion survives F0 amplitude" recorded. The per-case fit is fit_case; sweeping R is a shell loop. |
| `f0_squeeze_plot.py` | `f0_fit_case` inline plots | standalone plotter for f0_block_squeeze_sweep's json — archived with it. |
| `f0_paramcompare.py` | — (one-shot) | long-horizon separability of a specific joint-recovered (E,α) vs GT; hardcoded recovered value. Conclusion in `v0-E-coupling-illconditioned` memory (K=8 can't separate the ridge). |
| `f0_stretch_release.py` | — (no `_block` scene) | early "chunky pre-stretch + frozen base" forward sanity. Its clamped-base diag-stretch upstream isn't a `_block` SCENE; superseded by the actual release/drop work. Restore (or add a `clamped` scene) if needed. |
