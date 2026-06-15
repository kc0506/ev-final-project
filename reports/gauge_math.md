# The F0 left-stretch gauge in pre-stress system identification

**TL;DR** A release trajectory cannot distinguish a pre-stress `F0` from `F0·R0`
for any orthogonal `R0`. The only observable is the left stretch
`V0 = √(F0 F0ᵀ)`. So when we fit a pre-stress field, the *stretch* is identifiable
but the *rest configuration / displacement* is recoverable only up to a (global,
if we enforce compatibility) rigid rotation. Validation must be gauge-aware.

Companion figure: `outputs/explore/f0_gauge_demo/ybend_rot40/gauge_demo.png`
(produced by `reuse_mpm/explore/f0_gauge_demo.py`).

---

## Setup

We observe a deformed configuration `x0` (the release frame 0) and the subsequent
release trajectory. The unknown is the **pre-stress**
`F0 = ∂x0/∂X`, the deformation gradient mapping a stress-free **rest** `X` to the
observed `x0`. Material is isotropic hyperelastic — here fixed-corotated (FCR, the
warp PhysDreamer "jelly"):

```
τ(F) = 2μ (F − R) Fᵀ + λ J (J−1) I ,   F = U Σ Vᵀ ,  R = U Vᵀ (polar rotation),  J = det F
```

## Claim

The position trajectory is **invariant** under `F0 → F0·R0` for any orthogonal
`R0`. Therefore the trajectory only constrains `V0 = √(F0 F0ᵀ) = U Σ Uᵀ`; the
right rotation is a **gauge** (unobservable).

## Proof — step 1: stress is invariant under right rotation

Let `F' = F·R0`, `R0ᵀR0 = I`. Then:

- SVD: `F' = U Σ (Vᵀ R0) = U Σ (R0ᵀV)ᵀ` → **singular values `Σ` unchanged**, right factor `V' = R0ᵀV`.
- `J' = det(F R0) = det F = J`.
- polar rotation `R' = U V'ᵀ = U Vᵀ R0 = R·R0`.
- therefore
```
τ(F') = 2μ (F R0 − R R0)(F R0)ᵀ + λ J (J−1) I
      = 2μ (F − R) R0 R0ᵀ Fᵀ + λ J (J−1) I
      = 2μ (F − R) Fᵀ + λ J (J−1) I  =  τ(F).
```
So the Kirchhoff stress — hence the grid force — is unchanged. (This is the general
fact that an isotropic energy `Ψ(F)=Ψ̂(Σ)` depends only on singular values; FCR is
one instance.)

## Proof — step 2: the whole trajectory is invariant (induction)

One MPM substep: positions → stress → grid forces → velocity gradient `∇v` → update
`F ← (I + Δt ∇v) F`.

- **Base:** `x'(0) = x0 = x(0)`, `F'(0) = F0 R0 = F(0) R0`.
- **Step:** assume `x'(t) = x(t)` and `F'(t) = F(t) R0`. Same positions + step-1 ⇒
  same stress ⇒ same `∇v` ⇒ `x'(t+Δt) = x(t+Δt)`, and
  `F'(t+Δt) = (I+Δt∇v) F(t) R0 = F(t+Δt) R0`.

By induction `x'(t) = x(t)` for all `t`. The position trajectory is **bit-identical**. ∎

## How big is the gauge?

- The stress is **per-particle local**, so `R0` may be a *different* orthogonal
  matrix per particle: `F0(x) → F0(x) R0(x)` preserves `V0(x)` pointwise → identical
  dynamics. Among **all** `F0` fields, the observable is exactly the per-particle
  left-stretch field `V0(x)`; the right-rotation field (3 DOF/particle) is pure gauge.
- If we **enforce compatibility** (`F0 = ∂x0/∂X` for an actual map, i.e. curl-free),
  a per-particle `R0(x)` generally breaks it. The compatible right-rotations are the
  **global rigid** ones (`R0` constant). So among compatible fields the residual
  gauge is a single global rigid motion of the rest (rotation + translation).

## Consequence 1: storing `V0` is lossless

The cross-sim bundle stores `V0 = √(F0 F0ᵀ)`, not the raw `F0 = I+∇u`. Since
`F0` and `V0` differ by a right rotation (`F0 = V0 R_polar`), they give identical
dynamics — no information lost. Verified: `max|traj_F0 − traj_V0| ~ 1e-6`.

## Consequence 2: (ii)-b2 fit — what is and isn't recoverable

The inverse-map parametrization: MLP `x0 ↦ w`, rest `X = x0 − w`,
`F0 = (I − ∇_{x0} w)⁻¹`.

- **Identifiable:** the stretch field `V0(x)` — exactly, wherever the body moves
  enough to excite it (weak near low-motion nodes).
- **NOT identifiable:** the right-rotation gauge → rest/displacement only up to a
  global rigid motion.
- **Validation must be gauge-aware:**
  1. compare `V0` (or the symmetric strain) to truth directly — the hard, gauge-free metric;
  2. compare recovered displacement/rest to the true `u` **only after Procrustes-aligning** out a global rigid transform;
  3. "does the MLP look like `sin`?" → judge on the strain / aligned `u`, never raw `w`.

## Numerical demonstration

`f0_gauge_demo.py`, `R0 = rot_z(40°)`, on the half-sine bundle:

| quantity | value | meaning |
|---|---|---|
| `max\|F0' − F0\|` | **0.745** | the two pre-stresses are very different ICs |
| `max\|svd(F0') − svd(F0)\|` | **6.0e-7** | identical left stretch `V0` |
| `max\|traj' − traj\|` | **9.5e-7** | identical dynamics (motion scale ≈ 0.05) |

The figure shows: rest A (flat) and rest B (rotated 40°) both mapping to the *same*
`x0`; their two *different* displacement fields `u_A`, `u_B`; and the *overlapping*
release trajectories.
