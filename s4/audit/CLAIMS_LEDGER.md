# S4 Claims Ledger (D2)

> **Historical snapshot (2026-06-14).** A preserved point-in-time record from the S4 validity
> audit, published as part of the audit trail rather than as current documentation. Paths, line
> numbers, and pending items describe the pre-release working tree. Several sources cited in the
> evidence columns — the dated working notes (`notes/16-04-2026.md`), `KEY_STATS.md`, and the
> cross-hardware trace package — are internal working records not included in this repository;
> the full run datasets are available on request (see the repository README). Claims, values,
> and verdicts are unedited.

**Date:** 2026-06-14. Maps every empirical S4 claim → the script + command that regenerates it →
the value produced. **Provenance column is essential:** most values were produced on the archived
regime (A100 / JAX 0.9.2 / `config_snapshot.yaml`) and are **not yet reproduced under current
code** (see `AUDIT_REPORT.md` config-drift finding and `notes/16-04-2026.md`). Rows marked
**REGIME-PENDING** must be regenerated under the chosen canonical regime before print.

Specimen: S4 = `bin_03_rank_00_cell_6-0-3_flat_0423`. Genotype:
`s4/results/bin_03_rank_00_cell_6-0-3_flat_0423/genotype.npz`.

---

## A. By-construction claims (verified in code, regime-independent)

| # | Claim | Verified at | Value / verdict |
|---|---|---|---|
| A1 | Controller is blind: under `deformation_off`, perception has no terrain term | by construction `nca.py:205-242,663-732`; **empirically** cross-terrain `max|Δctrnn|=0` for full active horizon, all hardware (`SYMMETRY_BREAKING.md`) | **PASS (construction + empirical, exact)** — controller is a closed autonomous oscillator, terrain-invariant |
| A2 | `deformation_off` zeroes exactly the 4 deformation channels, nothing else | `nca.py:702-703`; `env.py:2545-2570` | **PASS** |
| A3 | Perception filters are lattice-topological (cannot encode geometry) | `nca.py:43-46,437-444` | **PASS** |
| A4 | Eval is deterministic given a seed (no eval-time RNG) | `config_snapshot.yaml:8,11` (`p_update=1.0`, `noise=off`); `post_run.py:5296-5298` | **PASS** |
| A5 | Success = reach-goal-without-falling | `batch_eval.py:211-218`; `fitness.py:113-120` | **PASS** (reporting wire-up to confirm at eval) |

## B. Empirical S4 behavioural claims

| # | Claim | Regenerating script + command | Archived value | Provenance / status |
|---|---|---|---|---|
| B1 | `deformation_off` increases peak cliff overhang ~1.256× vs intact (matched (4,2)) | `s4/scripts/` overhang probe → `KEY_STATS.md:15-19` | intact 0.3073 → def-off 0.3859 world units (ratio 1.256×) | A100/JAX0.9.2. **REGIME-PENDING** under current code |
| B2 | `deformation_off` deepens cliff commitment before recovery; both reach frontier & recover | `KEY_STATS.md:21-31` | intact final signed edge −0.793 vs def-off −0.688 cells; both recover | **REGIME-PENDING** |
| B3 | Mean cliff protrusion decreases as terrain lengthens (intact series) | native_core assay → `KEY_STATS.md:33-45` | (2,1) 0.104 → (10,5) 0.061 | **REGIME-PENDING** |
| B4 | Controller trace identical while plant diverges, divergence only *after* edge event (native (2,1) vs (2,9)) | `notes/16-04-2026.md:79-100` | `controller_max_abs_diff=0.0`; edge env-step 3513; `plant_first_diff_phase=10733` | **REGIME-PENDING** (within current code path) |
| B5 | Same plant divergence reproduced inside native bilateral family ((2,1) vs (6,3)) | `notes/16-04-2026.md:478-501` | `controller_max_abs_diff=0.0`; `plant_first_diff_phase=10733` | strongest result; **REGIME-PENDING** |
| B6 | Double-cliff: controller identical across widths (10 vs 3, 6000 steps) | `notes/16-04-2026.md:503-518` | `controller_max_abs_diff=0.0`; narrow edge phase 11005; `plant_first_diff_phase=11083` | **REGIME-PENDING** |
| B7 | Directional flip: S4 starts opposite direction in `deformation_off` vs intact | cross-hardware `hardware_trial_matrix.csv`, derived initial commit (validated 12/12 vs true COM) → `CROSS_HARDWARE_READ.md` | **12/12 flip (right→left) on A100-40/80 + L4**; 4/12 H100; 0/12 T4; 40/60 overall. Initial commit is a single global direction independent of goal side | **CONFIRMED (canonical regime)** — but absolute direction is hardware-poised (T4 no-flip, CPU intact=left vs GPU intact=right). CPU def-off pending |
| B8 | "deformation-off 100% success" (headline, native-core) | `s4_cross_hardware_state_space_20260614` state_space_trace_deformation_off, surface=native_core | **native-core def-off = 12/12 (100%)** on A100-40GB, A100-80GB, L4 | **CONFIRMED (A100/L4)** — reproduces archive; H100=4/12, T4=6/12 are the sensitivity tail. NB: 100% is *native-core*; with long-distance included def-off=15/20 |
| B9 | Cross-hardware: safety robust everywhere, deformation-off completion is the poised part | same package, `tables/hardware_endpoint_summary.csv` → `CROSS_HARDWARE_READ.md` | **fall_count=0 across all 110 rows**; def-off success A100/L4 0.75, T4 0.50, H100 0.25; degradations are safe (stall / return-no-goal) | **CONFIRMED (GPU + CPU-0.9.0.1 intact)** — STRUCTURED, not artifact. CPU def-off + JAX-0.9.2 local pending |

## C. Reproducibility / cross-regime claims

| # | Claim | Evidence | Status |
|---|---|---|---|
| C1 | Current-code `deformation_off` native-core differs from archived paper outputs | `notes/16-04-2026.md:568-623`; `preprint/README.md:226-229` (pre-release version; since rewritten for release) | **CONFIRMED mismatch** — e.g. archived (4,2) def-off `final_com_x=2.043, goal_touched=True` vs current `1.075, goal_touched=False` |
| C2 | Mismatch is NOT the controller artifact (genotype/CTRNN bit-identical) | `notes/16-04-2026.md:627-641` | **CONFIRMED** — max abs diff 0.0 on all CTRNN fields |
| C3 | Mismatch is backend/version/config drift (CPU vs A100, JAX 0.9.0.1 vs 0.9.2, config-hash) | `notes/16-04-2026.md:649-710`; this audit's config-drift table | **CONFIRMED** drift sources; **attribution PENDING** the author's reruns + Workstream B |

---

## Gaps to close before camera-ready (priority order)

1. **C3 attribution (CPU side)** — local CPU deformation-off + the JAX-0.9.2 local run (pending)
   confirm whether the safety-robust / completion-sensitive split survives the CPU backend and
   isolate version-vs-backend. Run `regime_classifier.py` when they land.
2. **B1-B6** — regenerate under the canonical regime (now tagged in `S4_REPRODUCIBILITY.md`:
   A100-40GB / JAX 0.9.2 / `config_snapshot.yaml` / seed 137) so printed numbers are reproducible.
3. **B7** — run the directional-flip isolation (Workstream C/4.4).
4. **Doc the headline precisely**: "deformation-off 100%" is *native-core* (12/12 on A100/L4); state
   the long-distance figure (15/20) and the H100/T4 sensitivity tail as a caveat, not a footnote.

**Resolved this pass:** B8 (headline reproduced on A100/L4), B9 (cross-hardware structured
sensitivity), and the genotype round-trip integrity (`genotype.npz` sha256 `dc13ba50…cf02`
identical across the package and committed bundle).

> Honesty note: every B-row value is archived (A100/JAX0.9.2) and, per C1, does **not** currently
> reproduce on CPU/JAX-0.9.0.1. Do not print these as reproducible-from-clean-checkout until the
> canonical regime is fixed and the numbers are regenerated there.
