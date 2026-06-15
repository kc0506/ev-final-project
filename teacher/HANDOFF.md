# Teacher diffusion — handoff

Scope: building/verifying the **video-diffusion teacher** for `01_tel_axisx_rest_T16`,
to later distill into an MPM physics-parameter generator. Worked on a **local RTX
4090 Laptop (16 GB)**; code synced to `meow2:~/tmp2` via mutagen.

## TL;DR (decision: continue with FLOW)

- The dataset's only latent is **`v0_x` (initial velocity, ±x axis); `E` fixed = 1e5**.
  Motion is **small, front-loaded, damped-settle** (net ~1% pixel change).
- An **RGB** pixel-L2 teacher fits to loss ~6e-4 but its samples are mostly static /
  sign-collapsed — because L2 is dominated by the static background, so **low loss ≠
  learned motion**.
- A **FLOW** teacher (model the dense GT screen-flow instead of pixels) makes motion
  the explicit target. GT flow is **free + exact** from the sim (project MPM particles),
  no RAFT.
- **Key result (score probes): the learned SCORE is good; the bad samples are a
  SAMPLING / mode-coverage artifact, not a failure of what the ckpt learned.** Both RGB
  and flow scores encode the full ±x distribution. This matters: VSD/DMD distillation
  uses the **score**, not ancestral samples.
- Both RGB and flow teachers are trained to **epoch 399** (`diff_final.pt`). Going
  forward: **flow**, and the real lever is **improving sampling**, not the score.

## Environment / how to run

- venv: `teacher/.venv` (`uv`-managed). torch **2.6.0+cu124** (cu124 wheels needed —
  the default cu128 wheel is too new for the local driver). `video_diffusion_pytorch`
  (lucidrains), imageio, matplotlib, tqdm.
- GPU fit (measured, full 35.71M params, fp32 unless noted):

  | frames × res | peak GB | 16 GB laptop |
  |---|---|---|
  | 16×128 (lucidrains default) | 16.13 | ❌ OOM |
  | **8×128** (what we train) | **8.28** | ✅ |
  | 16×128 **bf16 autocast** | 12.27 | ✅ (if you want full 16 frames) |
  | 12×128 | 12.32 | ✅ |

  → We train **res=128, frames=8**. Probes with batch>~4 at res128 OOM (3D attention);
  use `--chunk 3..4`.
- **Frame window**: take the **first 8** frames of the T16 clip (`build_flow_pack` /
  `video_pack_128_t8`). NOT an arbitrary window — `v0` lives at t0, so the clip must
  start at t0 or the learned v0 distribution drifts.

## Files (all in `teacher/`)

Training (canonical infra: tqdm, **atomic** ckpt via tmp+os.replace, **graceful
SIGTERM/Ctrl-C** stop→checkpoint, line-buffered metrics, `--resume auto`, loss_curve.png,
**animated grid gif** of all samples):
- `train_video.py` — RGB 3-ch teacher.
- `train_flow.py` — FLOW 2-ch teacher.
- `train_local.sh` — launches RGB; `EPOCHS=N ./train_local.sh` to resume to N. Slices
  the T16 pack → first-8 idempotently.
- `build_flow_pack.py` — MPM particles → dense screen-flow pack. Reuses `flow_viz.py`'s
  camera projection. Output `flow_pack_128_t8.npy (256,7,128,128,2)` in [0,1];
  decode px: `disp = (x-0.5)*2*scale_px` (`scale_px≈15.6`, in the `.meta.json`).

Verification / analysis (denoiser-score probes — query the model on controlled inputs,
no sampling, no distillation):
- `probe_modality.py` — **the main probe**, RGB or flow (`--channels 2|3`). Runs 3
  signals: `signdist` (err vs true v0_x), `mirror` (hflip), `treverse` (time reversal).
- `probe_signdist.py`, `probe_direction.py` — earlier flow-only versions (signdist;
  velocity-vector rotation anisotropy).
- `measure_hitrate.py` — RGB sampling "dynamic hit-rate" (fraction of samples with mean
  f2f |Δ| ≥ thresh).
- `sample_flow_grid.py` — sample N flow clips → animated grid gif + localisation/dir stats.
- `plot_loss.py` — loss curve PNG from any `metrics.csv`.

Data needed for flow work (in `outputs/`, mutagen-ignored, local only): per-sample
`mpm_xyz.npy [T,n,3]` + `sample.json` (has `v0`, `frame_jumps`) + `scene_cache.pt`
(`freeze_mask`) + a `camera.json`. Pull from meow2 if missing (see `scripts/`).

Checkpoints (`out_*/`, mutagen-ignored, local only):
- RGB: `out_01_tel_axisx_T8_local/` — epoch 399.
- FLOW: `out_01_flow_T8_local/` — epoch 399.
- Plots/gifs/probe figures under each `out_*/diag/`.

## Findings (objective numbers)

Sampling quality:
- RGB dynamic hit-rate (24 samples, f2f≥5) = **54.2%** (the eyeballed 2/8≈25% was
  small-sample noise — true coverage ~half).
- Flow ancestral samples are **diffuse**: %moving 74–97% vs train 12%; FFT radial
  spectral centroid **0.80 (gen) vs 3.24 (data)**, 97% vs 89% power in k≤5 → generated
  flow is a smooth low-freq blob, missing the data's localized high-freq motion
  structure. (gen stats from a few lossy PNG-extracted clips — re-run cleanly on the
  400 ckpt if needed.)

Score probes (`probe_modality.py`, epoch-199 ckpts; ts=[300,500]):

| signal | RGB | FLOW | read |
|---|---|---|---|
| signdist `ratio NEG/POS` | 1.04 | 0.82 | ≈1 ⇒ both ±x signs in-distribution |
| signdist `corr(err, v0_x)` | −0.005 | 0.15 | ≈0 ⇒ no sign bias |
| signdist `corr(err, \|v0_x\|)` | 0.63 | 0.77 | >0 ⇒ magnitude effect (expected, not a defect) |
| `treverse` rev/real | 8.1 | 35.0 | ≫1 ⇒ **temporal arrow / damping direction learned** |
| direction anisotropy err(90°)/err(0°) | — | 30–46× | ≫1 ⇒ strongly direction-sensitive (flow) |
| `mirror` flip/real | 38.5 | 10.6 | **CONFOUNDED — ignore for sign** (see below) |

→ Both scores learned the **±x sign distribution** (signdist symmetric) and the
**temporal arrow** (treverse). Combined with the poor samples ⇒ **score good, sampling
is the bottleneck.**

## Methodology learnings (how to verify a ckpt without VSD)

- A diffusion ckpt **is a score/denoiser field**. Verify by **querying the denoiser on
  controlled inputs** (add noise at chosen σ → one denoise step → residual / output).
  This isolates the teacher: no sampling stochasticity, no student, no optimizer — so
  when something's off you know it's the teacher. (VSD bundles all three; "if it breaks
  you don't know which part" — exactly why we avoided it for verification.)
- **Decompose the claim**: axis (direction anisotropy) vs sign distribution (signdist on
  real clips) vs temporal structure (treverse). Each is a separate falsifiable test.
- **Watch confounds**:
  - vector-rotation probe (`probe_direction`) makes spatially-**inconsistent** fields
    (displacement says one thing, flow vectors say another) → proves "direction-sensitive"
    but can't cleanly separate ±x. Use real-clip signdist for sign.
  - `mirror` (hflip) is dominated by **appearance asymmetry** (object shape/lighting/
    position mirror), not motion sign — high ratio ≠ "didn't learn −x". Discard for sign.
- **Don't trust sample-based metrics alone**: the score probes *contradicted* the
  sampling story (samples collapse, but the score covers both signs) — and the score is
  what distillation actually uses.
- Loss ~0 with bad samples = the loss isn't penalizing what you care about (here: motion
  is a tiny fraction of pixel-L2). Always look at the **loss curve** AND a
  task-relevant metric.

## Open / next (flow-first)

1. **Sampling is the lever, not the score.** Try: more sampler steps / DDIM / guidance;
   or fix the smearing at training (mask-weighted loss so background denoises to clean 0;
   or condition on `v0`). Goal: raise flow %moving toward ~12% and recover sign/dir
   coverage in *samples*.
2. Re-run `probe_modality.py` + `sample_flow_grid.py` + the FFT compare on the **epoch-399
   flow** ckpt to confirm whether +200 epochs changed score/sampling (not yet done).
3. **Density calibration** unverified: signdist shows ±x both covered, but not that the
   learned density matches the true ~uniform `v0_x`. Needs probability-flow ODE `log p`
   (heavier) if you want that.
4. For distillation: since the score is healthy, the flow teacher may be usable for
   VSD/DMD as-is — the ancestral-sample quality is not the blocker.

## Gotchas

- torch 2.6 `torch.load` defaults `weights_only=True` → load `scene_cache.pt` with
  `weights_only=False`.
- mutagen ignores `out_*/`, `outputs/`, `**/.venv/` → checkpoints/data/venv stay local,
  only code syncs to meow2. (`scripts/code-{up,down,status}.sh`.)
- RGB-400 once hit a transient `c10::Error` (CUDA) around epoch 304; plain resume
  recovered, did not recur. Watch for recurrence = real instability.
- Probes at res128 OOM above batch ~4 (3D attention) — keep `--chunk` small.
