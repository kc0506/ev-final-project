# reuse_mpm — reuse PhysDreamer's MPM→3DGS pipeline as a controlled roundtrip

A clean layer on top of PhysDreamer (treated as a third-party lib). One shared
`simulate_and_render` code path drives forward generation, dataset generation,
and inverse training, so pixel/coordinate conventions match **by construction**.

Scope (v1 of the project goal): **Y = a single global Young's modulus E**,
`p*(E) = log-uniform[E_min, E_max]`. Forward: known E → MPM → 3DGS → video.
Inverse: video → recover E. Dataset: sample E~p* → paired (E, video).

## Environment
```
conda env: physdreamer  (python 3.9, torch 2.0.0+cu118, warp 0.10.1, tyro)
PHYSDREAMER_ROOT=/tmp2/b10401006/PhysDreamer   (override via env var)
ENV=/tmp2/b10401006/.symlinks/miniforge3/envs/physdreamer
```
GPU is shared & quota-limited — always `ws-status` / `nvidia-smi` before a run.
`reuse_mpm.gpu.pick_free_gpu()` auto-selects the freest GPU when
`CUDA_VISIBLE_DEVICES` is unset.

## Module layout
Config is a single source of truth: `config.py` holds `SimConfig` (physics +
rollout, incl. `grid_lim`, `substep_size`), `SceneSpec` (which scene + how it is
discretised), and one dataclass per task. `tyro.cli(<Config>)` builds the CLI from
the dataclass, and the *resolved dataclass is what gets serialised* into the run
dir's `config.json` (so the recorded config can't drift from the run).

- **Canonical pipeline** (tyro CLIs, `run(cfg)`):
  `forward_gen`, `dataset_gen`, `train_global_E`, plus the shared building blocks
  `scene` / `scene_physgaussian` / `scene_io` (one `load_from_spec` pd/pg
  dispatch), `sim_render`, `mpm_rollout` (the one owner of the differentiable
  rollout), `diff_sim`, `recover` (the one `recover_global_E` loop), `run_io`,
  `gpu`, `_env` (single PhysDreamer boundary).
- **Exploration** (`reuse_mpm/explore/`, one-shot diagnostics, argparse, import
  the canonical routines): `probe_identifiability`, `recovery_sweep`,
  `multiscene_fwdbwd`, `gradcheck`.

## Canonical entrypoints (tyro; each writes a self-contained, reproducible run dir)
Nested fields are addressed with dots, e.g. `--scene.path`, `--sim.substep`.
```bash
# 1. forward: known E -> video
$ENV/bin/python -m reuse_mpm.forward_gen \
    --scene.path $PHYSDREAMER_ROOT/data/physics_dreamer/telephone \
    --E 1e5 --v0 0 -1 0 --sim.num-frames 8 --sim.substep 32 \
    --out outputs/fwd_telephone_E1e5
#   PhysGaussian scene: add --scene.kind pg --scene.path <model_dir>

# 2. inverse: recover global E from a forward_gen run (reads its config.json)
$ENV/bin/python -m reuse_mpm.train_global_E \
    --gt_run outputs/fwd_telephone_E1e5 --init_E 3e5 --coarse-init \
    --iters 24 --lr 0.04 --window 3 --grad_window 1 --out outputs/inv_headline

# 3. dataset: sample E~p*(E)=logU[E_min,E_max] -> (E, video) dataset
#    saves per-frame mpm_xyz.npy + per-sample stability flag
$ENV/bin/python -m reuse_mpm.dataset_gen \
    --scene.path .../telephone --E_min 1e4 --E_max 1e6 --n 256 \
    --sim.substep 96 --seed 0 --out outputs/dataset_telephone_256
```

## Exploration scripts (argparse; under reuse_mpm.explore)
```bash
# identifiability / loss landscape (is E recoverable, over what range?)
$ENV/bin/python -m reuse_mpm.explore.probe_identifiability \
    --dataset_dir .../telephone --v0 0 -1 0 --E_star 1e5 --E_min 1e3 --E_max 1e7 \
    --n 15 --out outputs/probe_v0_1

# honest recovery sweep (faithful PhysDreamer: substep 96, full BPTT, no cheat)
$ENV/bin/python -m reuse_mpm.explore.recovery_sweep \
    --dataset_dir .../telephone --true_Es 3e4 1e5 3e5 --init_Es 1e4 3e4 1e5 3e5 1e6 \
    --substep 96 --window 4 --iters 40 --out outputs/recsweep_faithful

# multi-scene forward+backward (PhysDreamer "pd:" + PhysGaussian "pg:")
$ENV/bin/python -m reuse_mpm.explore.multiscene_fwdbwd \
    --scenes pd:.../telephone pg:.../ficus_whitebg-trained \
    --true_E 1e5 --init_Es 3e4 3e5 --substep 96 --out outputs/multiscene_7

# gradient localisation: trajectory (MPM) vs pixel (MPM+3DGS)
$ENV/bin/python -m reuse_mpm.explore.gradcheck --scenes pd:telephone \
    --true_E 1e5 --points 3e4 3e5 --substep 32 --out outputs/gradcheck
```

## Scenes (7)
- PhysDreamer (`scene.load_scene`, native format): telephone, alocasia, carnations, hat
  (extracted from `claude-downloads/physdreamer/physics_dreamer.zip`).
- PhysGaussian (`scene_physgaussian.load_physgaussian_scene`): ficus, wolf, vasedeck.
  Pure foreground → all gaussians simulated; cameras from `cameras.json`; anchor =
  geometric bottom-slab (lowest `freeze_frac` of particles along the longest axis),
  since a uniform v0 on an unanchored body just translates (E unidentifiable).
- Scene discretisation capped at `max_particles=8000`; reference clouds subsampled
  to 20k before `find_far_points`; chunked k-means — all to bound memory on large
  objects (alocasia ~217k, vasedeck ~730k gaussians).

## Results (telephone scene, 7585 MPM particles)
- Forward roundtrip is self-consistent: at the true E the inverse photometric
  loss floor is ~5e-6 (uint8 quantisation).
- **E recovered to 0.3–1.7% rel. error** (log10_err ≤ 0.007) from a far init via
  coarse-grid → gradient refine.

## Non-obvious gotchas (learned the hard way — keep these)
1. **Scene discretisation is non-deterministic** (k-means particle downsample).
   Forward-gen and inverse MUST share the SAME particles or the roundtrip is
   invalid. Fixed by `scene.load_scene(cache_path=...)`; the cache lives in
   `outputs/_scene_cache/<scene>_ds<>_g<>_k<>.pt` and is auto-derived/shared.
2. **Long-horizon MPM backprop is unstable** — the gradient w.r.t. E is correct
   for ~1 frame (32 substeps), flips sign by ~2 frames, explodes by ~3+. Fixes:
   (a) use the **per-particle E grad path** (pass E as a [n] constant tensor, let
   torch reduce to logE) — the scalar/aggregating path is buggy; (b) **truncated
   BPTT**: `grad_window=1` (only the last frame's substeps carry gradient); (c)
   keep the loss `window` small (≤3 frames).
3. **Narrow loss basin**: pixel-MSE vs E is a sharp well at E* on a flat plateau.
   Pure gradient from far fails (no signal) → use `--coarse_init` (grid search
   into the basin) then gradient refine with cosine-decayed lr.
4. Motion is driven by initial velocity `v0` (gravity off); E is identifiable in
   roughly [1e4, 1e6] for telephone — above ~1e6 the object barely moves (plateau).
5. Don't bind to `motionrep` / `thirdparty_code.warp_mpm` (cwd/path-hack copies);
   bind only to installed `physdreamer.*` + `projects/inference/local_utils`.

## Run-dir contract (deliverables)
Each canonical task has a declarative run-dir class in `run_io.py`
(`ForwardRun` / `RecoverRun` / `DatasetRun`) whose named methods ARE the output
schema; writes are incremental. `config.json` is the resolved config dataclass
(`asdict`) + a `task` tag + an `_provenance` block stamping the `reuse_mpm` and
PhysDreamer git SHAs, so a run records exactly the code + config that produced it.
Artifacts: `config.json`, `source_ply` (symlink), `frames/`, `video.mp4`/`.gif`,
plus task extras (`recovery.png`+`metrics.json`+`trace.json`, dataset
`manifest.json`+`p_star.png`+`sample_XXXX/`). Exploration diagnostics
(`landscape.png`, sweep `results.json`, …) are exempt.
```
