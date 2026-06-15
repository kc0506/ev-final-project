# HANDOFF — project state, provenance, how to resume

End-of-poster-session handoff (2026-06-15). Read this first if you are coming
back to review or extend the work.

- **Snapshot tag:** `snapshot/poster-2026-06-15` (complete state at session end)
- **Project page:** https://kc0506.github.io/ev-final-project/ (branch `gh-pages`)
- **Code (this repo):** https://github.com/kc0506/ev-final-project (`main`)
- **Inverse pipeline (image loss):** https://github.com/kc0506/gic/tree/feat/f0-img-loss
  (fork of [GIC, NeurIPS 2024](https://github.com/Jukgei/gic))

> All run outputs and `*.npy/*.npz/*.pt` data are **gitignored** (reproducible,
> many GB). The paths below are **local** under `ev-project/`, not in the repos.

## Headline results + provenance

| Target | Run dir (local) | Metric |
| --- | --- | --- |
| `F₀` bend | `gic_val/output/prestress_field/bend_inverse_ufield` | rest-state RMSE 1.2e-3, shear-profile corr 0.99 |
| `F₀` spring | `gic_val/output/prestress_field/spring_F0` | rest-state RMSE 9.1e-5 |
| `E` field | `gic/output/ours_efield/efcirc_uniformv0` | 2.8% rel err (logE err 0.012) |
| `v₀` field | `gic/output/ours_telephone/field_a16_bend_s6_lr075` | profile corr 0.97 (rel-L2 ~0.20; amplitude smoothed) |

Note: `v₀` rel-L2 (~0.20) and profile-corr (0.97) tell different stories — the
shape is recovered, the amplitude is smoothed by TV. Use profile corr for the claim.

## Figure regeneration map

Figures live in `poster/assets/` (and mirrored into `poster/site/static/images/`).
Generators are in the **gic** repo and write to `generative-phys/poster/assets`:

| Figure | Generator (gic) | Data source |
| --- | --- | --- |
| `overlay_telephone_diag45_x.png` | `poster_overlay.py` | telephone scene + rollout |
| `traj_cloud_telephone_x*.png` | `poster_traj_cloud.py` | telephone rollout |
| `ybend_gradu_f{0,4,8}.png` | `poster_gradu_orig.py` | `generative-phys/outputs/explore/f0_gradu_viz/ybend_halfsine/traj.npz` |
| `ybend_cloud_*` (magma/viridis) | `poster_npz_cloud.py` | same ybend traj.npz |
| `efcirc_uniformv0_cloud.png` | `poster_efield_cloud.py` | efcirc E-field rollout |
| `field_a16_bend_s6_cloud_{gt,pred}.png` | `poster_v0field_cloud.py` | `field_a16_bend_s6_lr075` ckpt |
| `spring_F0_*` frames | gic_val `spring_F0/overlay.gif` split | `gic_val/output/prestress_field/spring_F0` |

GIFs on the project page (`spring_F0_cloud.gif`, `spring_F0_render.gif`) were built
from the frame sequences with ImageMagick (white bg, ping-pong loop):

```
magick -delay 11 -loop 0 spring_F0_cloud_f{1..16}.png spring_F0_cloud_f{15..2}.png \
  -background white -alpha remove -alpha off -layers Optimize spring_F0_cloud.gif
```

## Repo layout (this repo)

```
reuse_mpm/   warp-based differentiable MPM pipeline (forward / recover / dataset)
  explore/   f0 sys-id + loss-design diagnostics (_block, _viz libs, f0_* probes)
teacher/     optical-flow distillation subsystem (+ probes, HANDOFF)
infra/modal/ Modal cloud GPU offload (gic + PhysDreamer envs)
poster/      LaTeX poster + project page (site/) + figure assets
reports/     landscape / loss-design writeups (npz data gitignored)
vsd/         score-distillation subproject (source only; out/ = 120MB gitignored)
```

## Cross-repo / cross-tree notes

- **`gic_val/`** (sibling dir, not a tracked repo here): the heavy-anchor / F0
  validation tree. The `prestress_field` F0 results live here, not in `gic`.
- **`gic` `ours/`**: the image-loss inverse pipeline (geom / fields / imgloss /
  viz / f0field) + `fit_traj_*` / `fit_image_*` entrypoints + `poster_*.py`.
- File-contract gap: `forward_gen` does not emit `traj.npy`; some gic tools expect
  it. `_block.Scene` is a third scene abstraction that does not produce the
  SceneBundle cache (integration debt).

## Refactor status — `refactor/pipeline-quality` (INCOMPLETE)

See the isolated `wip(refactor/pipeline-quality)` commit (just before the README
on `main`). To get a pre-refactor-finish tree, checkout the commit before it.

- Done: lib package + fields/geom; scene/estimator/train/viz extraction + GPU-pick
  contract; legacy entrypoints reconnected to the shared lib.
- Left:
  - **#13** re-split entrypoints by training mode (mapping table unconfirmed)
  - **#14** archive retired scripts + GPU landing verification
  - **#15** `fit_image_*` three entrypoints + bg flag + with-bg loss experiment

## Reproducing the environment

`environment.yml` (this repo) + per-env notes in the `modal-runs` skill /
`infra/modal/`. GPU jobs offload to Modal; never pin `CUDA_VISIBLE_DEVICES`.
