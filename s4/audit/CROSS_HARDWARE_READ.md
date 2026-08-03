# S4 Cross-Hardware — Initial Read (D3, partial)

> **Historical snapshot (2026-06-14).** A preserved point-in-time record from the S4 validity
> audit, published as part of the audit trail rather than as current documentation. The trace
> package it reads (`paper_data/…`) and the audit runbook it refers to are internal working
> records not included in this repository (full run datasets available on request); the items
> marked pending were open at the time of writing. Numbers, tables, and verdicts are unedited.

**Date:** 2026-06-14. Source: `paper_data/preprint/s4_cross_hardware_state_space_20260614/`
(today's GPU package). **Local JAX-0.9.2 run still pending** — this is the GPU + CPU-0.9.0.1 read.
Same genotype everywhere (`genotype.npz` sha256 `dc13ba50…cf02`), same `config_snapshot.yaml`,
seed 137. 2 trials × 10 distance-pairs × 2 conditions per runtime.

## The decisive number: safety is invariant

**`fall_count = 0` across all 110 rows** — every hardware (A100-40/80, H100, L4, T4, local CPU),
both `intact` and `deformation_off`, every distance pair. The body never falls off the cliff
anywhere. **Safety does not depend on hardware, precision, or the deformation signal.**

## Intact is robust; deformation-off is the poised/sensitive readout

| Runtime (JAX) | intact success | **def-off success** | def-off endpoint mix (of 20) |
|---|---|---|---|
| modal A100-40GB | 20/20 | **15/20** | 7 direct-goal · 8 reverse-&-goal · 2 cliff-stop · 3 stall · **0 fall** |
| modal A100-80GB | 20/20 | **15/20** | 7 · 8 · 2 · 3 · 0 — *bit-identical to A100-40* |
| modal L4 | 20/20 | **15/20** | 7 · 8 · 2 · 3 · 0 — *bit-identical to A100* |
| modal H100 | 18/20 | **5/20** | 3 direct · 2 reverse-&-goal · 4 returned-no-goal · 11 stall · **0 fall** |
| modal T4 | 20/20 | **10/20** | 10 direct · 0 reverse-&-goal · 10 returned-no-goal · **0 fall** |
| local CPU (0.9.0.1) | 18/20 | *pending* | (intact only so far) |

Native-core deformation-off specifically: **A100-40, A100-80, L4 = 12/12 (100%, all 6 pairs)** —
reproducing the archive headline. H100 = 4/12 (stalls before the edge on (3,2)–(6,3)); T4 = 6/12.

## Verdict: STRUCTURED sensitivity, not artifact

The audit runbook's key question (3.1) — *do safety and reversal dissociate, safety robust everywhere,
reversal the sensitive part?* — answers **yes, emphatically**:

- **Safety (no fall): perfectly robust** across all hardware and JAX 0.9.0.1↔0.9.2. 0/110.
- **Intact goal-completion: robust** (90–100% everywhere).
- **Deformation-off goal-completion: the poised, hardware-sensitive readout** — and crucially,
  every degradation mode is **safe**: `stall_no_edge` (H100) or `returned_no_goal` (T4), i.e. the
  body either stops short or reverses-from-the-edge-without-finishing. It **never** converts to a
  fall, a tunnelling success, or a fall-near-goal. This is exactly the signature that *supports*
  the knife-edge reading and *rules out* a scoring/physics artifact.

Within the Modal set, jax/config/seed/genotype are held constant (one image), so the GPU-to-GPU
spread (A100/L4 cluster vs H100 vs T4) is a **pure hardware numeric-precision effect** (TF32 /
reduction-order on a poised system) — not a code or version difference. That A100-40, A100-80, and
L4 land **bit-identically** shows the result is reproducible across a GPU cluster, not a
single-device fluke; H100/T4 are the documented sensitivity tail.

## Initial direction: intact vs deformation-off (directional-flip / claim B7)

Derived per trial from the trial matrix (`first_left/right_cliff_reach_step`, else net COM sign);
**validated 12/12 against true COM first-commit** (first >0.2-cell displacement) on the local-CPU
traces. Key structure: the body commits to a **single global initial direction per (runtime,
condition), independent of which side the goal is on** — wrong-side-goal trials go that way first,
then reverse. So "initial direction" is a property of the controller+plant dynamics, not the task.

Initial commit direction, native_core (all 12 trials per cell unless noted):

| Runtime | intact | deformation-off | flip (of 12 matched) |
|---|---|---|---|
| A100-40GB | **right** (12/12) | **left** (12/12) | **12/12** |
| A100-80GB | **right** (12/12) | **left** (12/12) | **12/12** |
| L4 | **right** (12/12) | **left** (12/12) | **12/12** |
| H100 | mixed (8 left / 4 right) | left (12/12) | 4/12 |
| T4 | right (12/12) | right (12/12) | 0/12 |
| local CPU (0.9.0.1) | **left** (12/12) | *pending* | — |
| **Total matched** | | | **40 / 60 flip** |

**Read:** on the canonical A100 regime (and A100-80, L4) the deformation signal sets the body's
initial commit — removing it **flips the direction in every native trial (right → left, 12/12)**.
That is the symmetry/directional-flip evidence for B7, and it's clean. *But the absolute direction
is itself poised/hardware-sensitive:* T4's deformation-off stays right (no flip), H100 intact is
mixed, and **local CPU intact commits left — opposite the GPUs' intact-right**. So the *flip* is
real on the canonical regime; *which way* each condition points is part of the knife-edge
sensitivity, not an invariant. The CPU↔GPU intact disagreement is the same mismatch under
investigation — the pending CPU deformation-off + JAX-0.9.2 runs will show whether CPU flips too.

## Caveats / open items
- **Modal GPU JAX version not separately recorded.** The embedded per-assay `manifest.json`
  carries the *specimen's origin* stamp (A100/JAX 0.9.2); the *execution* runtime version is only
  recorded for the local runs (`local_*_runtime.json`). Confirm the Modal image's resolved
  `jax[cuda12]` version so the GPU column's JAX is documented (currently unpinned — see
  `RELEASE_CLEANUP.md` B2/B5).
- **n = 2 per pair** — success *rates* are coarse (multiples of 0.05). The qualitative
  dissociation is solid; don't over-read 0.75 vs 0.50.
- **Pending:** local CPU JAX-0.9.0.1 deformation-off + the JAX-0.9.2 local run. When they land,
  run `regime_classifier.py` to confirm the CPU side preserves the safety-robust / completion-
  sensitive split and to attribute any CPU↔GPU gap to version vs backend.
