# Beyond Rest-State, Homogeneous System Identification

Complex initial conditions, non-uniform material fields, and a spectral loss, via
**Differentiable MPM + 3D Gaussian Splatting**.

> Embodied Vision (CSIE5421) final project &middot; National Taiwan University &middot; Casey Hong (b10401006)

**🔗 Project page (figures, results, references): https://kc0506.github.io/ev-final-project/**

---

## What this is

Classical physical system identification makes two convenient assumptions: the object starts
**at rest**, and it is made of **one homogeneous material**. Real objects break both. This project
poses a harder inverse problem: from observed dynamics, recover a **complex initial condition** (an
initial velocity field `v₀` and a pre-stressed initial deformation `F₀`) on a **non-uniform material
field** `E(x)`, by back-propagating through a differentiable Material-Point-Method simulator coupled
to 3D Gaussian Splatting. We also study *when* each quantity is identifiable, and analyze a
**spectral loss** for oscillatory motion where a time-domain L2 loss bounds frequency mismatch.

Key results (details on the project page):

| Target | Setup | Error |
| --- | --- | --- |
| `F₀` (pre-stress) | released bend | rest-state RMSE 1.2×10⁻³, shear-profile corr 0.99 |
| `F₀` (pre-stress) | spring, uniform | rest-state RMSE 9.1×10⁻⁵ |
| `v₀` field | bend velocity field | profile corr 0.97 |
| `E` field | circular excitation | 2.8% relative error (logE error 0.012) |

## Repository layout

```
reuse_mpm/        warp-based differentiable MPM pipeline (this repo's core)
  scene.py          build an anchored sim scene from a PhysGaussian / 3DGS cache
  forward_gen.py    generate ground-truth trajectories
  recover.py        recover scalar / field Young's modulus E
  recover_v0.py     recover initial velocity
  train_global_E.py / train_field_E.py / train_v0.py   inverse-fit entrypoints
  dataset_gen.py    factorized (E, v0, T) dataset generation
  sim_render.py     render rollouts
teacher/          optical-flow distillation subsystem
infra/modal/      Modal cloud GPU offload (gic + PhysDreamer environments)
poster/           LaTeX poster + the project-page source (poster/site/)
```

## Companion repository: the image-loss inverse pipeline

The single-/multi-view **image-loss** inverse pipeline (recovering `v₀` / `F₀` / `E` directly from
rendered video through a differentiable 3DGS renderer) lives in a fork of
[GIC (Gaussian-Informed Continuum, NeurIPS 2024)](https://github.com/Jukgei/gic):

**→ https://github.com/kc0506/gic/tree/feat/f0-img-loss** (`ours/` + `fit_*.py`)

## Running

Entrypoints are [tyro](https://github.com/brentyi/tyro)-based dataclass CLIs; each writes to an
auto-created run directory. GPU jobs can be offloaded to [Modal](https://modal.com) via
`infra/modal/`. See the project page and per-module docstrings for the exact invocations.

## References

This work builds directly on **GIC** ([Cai, Yang, Yuan et al., NeurIPS 2024](https://arxiv.org/abs/2406.14927)),
and relates to 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023), PhysGaussian (Xie et al., CVPR 2024),
PhysDreamer (Zhang et al., ECCV 2024), PAC-NeRF (Li et al., ICLR 2023), Spring-Gaus (Zhong et al., ECCV 2024),
and MLS-MPM (Hu et al., SIGGRAPH 2018). Full list on the
[project page](https://kc0506.github.io/ev-final-project/#references).
