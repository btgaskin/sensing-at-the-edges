# How symmetry is broken in S4 (mechanistic note)

> **Historical snapshot (2026-06-14).** A preserved point-in-time record from the S4 validity
> audit, published as part of the audit trail rather than as current documentation. The trace
> package it analyses (`paper_data/…`) is an internal working record not included in this
> repository (full run datasets available on request), and the questions in the closing section
> were open at the time of writing. Findings and numbers are unedited.

**Date:** 2026-06-14. Evidence: `paper_data/preprint/s4_cross_hardware_state_space_20260614/`
trace `.npz` (per-step `plant_positions`, `ctrnn_carry`, `nca_state`, `action_outputs`,
`phase_h_drive`, `phase_deformation_inputs`), all on disk for A100-40/80, H100, L4, T4, and local
CPU (JAX 0.9.0.1 + pyarrow), both `intact` and `deformation_off`.

## The puzzle
The bilateral cliff task is **mirror-symmetric**: for each distance pair, the left-goal and
right-goal trials are exact mirror images (goal/cliff positions swapped about a fixed axis;
identical spawn `COMx≈1.55`). Yet the body commits to **one absolute initial direction regardless
of which side the goal is on** — wrong-side-goal trials go that way first, then reverse. So the
symmetry is broken by the **agent**, not the environment.

## What breaks it (verified)
1. **Explicit, not spontaneous.** The environment is symmetric except for a 1-cell
   `bilateral_spawn_x_offset: 1` (config). With `hidden_init_scale=0` (symmetric initial state)
   and `noise=off`, the remaining asymmetry source is the **evolved CTRNN weights/biases**, which
   carry no symmetry constraint. The body therefore has an intrinsic **handedness**.
2. **Intrinsic handedness = LEFT.** In `deformation_off` — where the only terrain-derived input is
   removed, so behaviour is driven by the controller's own weights — the body commits **left on
   every runtime except T4** (A100-40, A100-80, H100, L4, CPU → left; T4 → right). Left is the
   default the controller falls into when sensory drive is stripped.
3. **The deformation signal pushes RIGHT — but marginally (poised).** Turning deformation on
   (`intact`) biases trials toward right. The push is just strong enough to win **under GPU float
   precision but not CPU**: GPU intact is right-leaning (native-core 12/12 right on A100/L4; mixed
   ~8L/12R across all surfaces), while **CPU intact stays all-left (20/20)**. The *same* deformation
   signal, under CPU float32 vs GPU TF32, yields **opposite** commits. Net effect: CPU shows ~no
   flip (1/20), GPUs show a large flip (12/20).
4. **The break is slow and near-balanced.** The body dithers near-symmetric for hundreds–thousands
   of steps before committing: `|ΔCOMx| < 0.06` cell at step 100, and the 0.2-cell commit
   threshold isn't crossed until step ~190 (A100 intact) to ~3500 (CPU def-off). Left/right
   actuator drive differs by only ~2–5%. Intact vs deformation-off trajectories diverge as early as
   step ~40 but amplify slowly. This is the signature of a system sitting near a symmetric unstable
   manifold (pitchfork/saddle-like), where a tiny bias selects the branch.
5. **T4 is a precision outlier** — its `deformation_off` commits *right*, opposite all other
   runtimes — confirming branch selection is governed by low-order floating-point differences, not
   a robust mechanism.

## One-line mechanism
**Symmetry is broken explicitly by the asymmetric evolved controller (+ a 1-cell spawn offset),
and the resulting left/right competition is poised so finely that the deformation signal — and
even GPU-vs-CPU rounding — decides which way the body goes.** This is exactly the knife-edge
reading: safety is invariant (the body never falls; see `CROSS_HARDWARE_READ.md`), but the
*direction* of the autonomous commit sits on a razor's edge.

## What this does NOT yet pin down (open)
- **Spawn-offset vs weight-asymmetry** as the source of the intrinsic-left bias — needs a
  centered-spawn (offset 0) or mirrored-weights control to separate.
- **The precise motor route**: net per-side drive sums (~few %) do not linearly predict direction
  (locomotion is gait-*phase* mediated), so the controller→COM causal path is not yet traced.
- **Is the intact flip specifically TF32?** Re-running GPU intact with `matmul_precision=highest`
  would prove it: if GPU-intact then commits left (matching CPU), the flip is TF32. (Cheap add.)
- **Controller-internal handedness signal**: where in `ctrnn_carry`/`nca_state` the chosen
  direction first appears — decodable from the existing trace npz (no new runs).
