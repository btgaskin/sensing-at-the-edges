# S4 Validity Audit — Report (D1)

> **Historical snapshot (2026-06-14).** A preserved point-in-time record from the S4 validity
> audit, published as part of the audit trail rather than as current documentation. Paths, line
> numbers, and open items describe the pre-release working tree, and some sources cited here —
> the audit plan and runbook, and the dated working notes (`notes/16-04-2026.md`) — are internal
> records not included in this repository. Findings, numbers, and verdicts are unedited.

**Date:** 2026-06-14
**Specimen:** S4 = `bin_03_rank_00_cell_6-0-3_flat_0423`
**Scope of this pass:** static / code-interpretation audit (Workstreams A + E). No evaluations
run. Cross-hardware / precision / perturbation checks are *specified but deferred* (the author is
running the regime comparison separately and will report back).

---

## One-line verdict

**The S4 blindness claim is sound *by construction* — under `deformation_off` the controller's
perception contains no terrain/geometry/COM/position term, so "no cliff signal reaches the
controller" holds structurally (PASS on the CLAIM-CRITICAL by-construction checks).** The
remaining risk is *not* the claim's logic but its **reproducibility**: the archived S4 was
produced under a config (`max_steps=4000`, `n_emitters=12`, archive `[7,7,10]`, A100/TF32, JAX
0.9.2) that has **drifted** from the current repo preset and cannot be reproduced as-is. Strength
of claim is gated on the author's local-current vs updated-local reruns resolving the
structured-vs-artifact question (Workstream B), not on any code defect found here.

No STOP-condition was triggered: no geometry term reaches the controller; no physics-artifact
inspection was in scope this pass (deferred to Phase 5).

---

## Phase 1 — Pipeline correctness

### 1.1 Scoring logic — **PASS (with one item to confirm at eval)**
- Outcome gate `batch_eval.py:211-218` `_gait_bonus_gate`: success ≡ `(goal_hit_count > 0) &
  (no falls across episode)`. Cannot be satisfied by a fall (any `fell` ⇒ excluded).
- Safe-edge arrival gate `fitness.py:113-120` `_safe_edge_gate_array`:
  `(cliff_vicinity_steps > 0) & ~touched_death_plane & ~went_airborne & ~com_x_went_negative`.
  A fall (`touched_death_plane`/`went_airborne`), an arena exit (`com_x_went_negative`), and a
  body that never neared the edge are all excluded.
- **To confirm at eval (1.1 / 1.6):** that the *reported* "deformation-off success" column in
  `post_run` aggregation uses exactly this `goal_hit & no-fall` predicate (not a timeout- or
  proximity-based proxy). The gate code is correct; the reporting wire-up should be verified
  against three rendered S4 episodes when runs are available.

### 1.2 Reversal definition — **PASS (independent re-derivation pending)**
- `env.py:680-689` `reversal_progress = clip((peak_incorrect_progress −
  final_incorrect_displacement) / incorrect_route, 0, 1)` — a real COM turnaround toward the
  wrong-facing edge and back, derived from COM displacement, not from a heading proxy.
- `pca_analogue.py:340-347` labels a "reversal" stage from `goalward_x` range fractions (post-hoc
  visualization only; not a scoring input).
- **Workstream B** will independently re-derive reversal from the raw COM-x trajectory and assert
  agreement on ≥3 episodes. Not yet run (no committed trace for the chosen episodes in scope).

### 1.3 Deformation-off integrity — **PASS**
- Two equivalent zeroing routes, both touching *only* the 4 deformation channels:
  - Controller-flag route `nca.py:702-703`: `nca_state.at[:, :, :N_DEFORM_CHANNELS].set(0.0)`
    (`N_DEFORM_CHANNELS == 4`).
  - Env-variant route `env.py:2545-2570` `_apply_deformation_variant`: for `deformation_off`/
    `zero` returns `jnp.ones_like(deformation)` ⇒ `signed_delta = ratio − 1 = 0` ⇒ channels 0-3
    become 0; all other slices untouched.
- The 4 channels are `(abs_h, abs_v, comp_h, comp_v)` deltas (`deformation_probe.py:9-14,22-25`),
  written at `nca.py:698-701`. Action channels (4,5) and grid-hidden channels (6-11) are written
  elsewhere (CTRNN outputs) and are unaffected by the ablation. **No leakage** into
  action-feedback or signalling.

### 1.4 Controller blindness, by construction — **PASS (CLAIM-CRITICAL)**
For the loaded S4 config (`config_snapshot.yaml`: `noise_type=off` ⇒ `ou_enabled=False`,
`ctrnn_grid_hidden=6`, `gasnet=off`, growth off), the perception layout
(`nca.py:162-164,205-242`, `perception_dim` `nca.py:245-251`) is:

| Perception channels | Source channels | Provenance | Terrain-derived? |
|---|---|---|---|
| Private identity 0-3 (deformation) | NCA state 0-3 | physics deformation `inject_deformation_channels` | **YES — and these are exactly what `deformation_off` zeroes** |
| Private identity 4-5 (action) | NCA state 4-5 | CTRNN action outputs (efference) | No |
| Comm filter bank ×4 over 6-11 | NCA state 6-11 (grid hidden) | CTRNN recurrent hidden state | No |

`base_count = 4 deform + 2 action + 0 noise = 6`; private = [0..5], comm = [6..11];
`perception_dim = 6 + 6×4 = `**`30`**. Under `deformation_off`, channels 0-3 ≡ 0, so the
perception reduces to a pure function of **(action feedback, CTRNN grid-hidden)** — both
controller-internal. **No term derived from terrain, global position, COM, or voxel geometry
reaches the controller.** The only terrain sensory route into the NCA state is channels 0-3, and
the ablation zeroes exactly those. STOP-condition **not** triggered.

> **Runbook correction:** the audit runbook's "30-d perception vector" is correct *for the paper config
> specifically*. An exploration pass assuming the NCA default (`noise=ou_fixed`) would count 31-d
> (an extra OU channel). Ground truth is the per-specimen `config_snapshot.yaml`, which has
> `noise=off` ⇒ 30-d. Always read the snapshot, not repo defaults.

### 1.5 Filters are lattice-topological — **PASS**
- Kernels `nca.py:43-46` (Sobel-X/Y, Laplacian) + identity.
- Applied via `jax.lax.conv_general_dilated(..., padding="SAME", feature_group_count=c)`
  (`nca.py:437-444`): a fixed 3×3 Moore-neighbourhood convolution over **cell-lattice indices**,
  per-channel. Operates on topological grid neighbours, **not** physical voxel positions, so the
  filters cannot encode geometry or deformation. PASS.

### 1.6 Episode termination — **PASS (last-frame artifact characterised)**
- Outcome flags live on `EpisodeResult` (`env.py` ~330-379); termination via goal-touch or fall.
- Confirmed a **post-termination frozen-frame artifact** in the traces: after `done`, plant/phase
  traces keep emitting a frozen final frame while controller traces drop inactive microsteps
  (`notes/16-04-2026.md:183-202`). Empirically, cross-terrain controller diffs are exactly 0 for
  the whole active horizon and first differ only at the final recorded index (T4 666/667, H100
  1933/1934). Analyses must trim to the active horizon — no scoring/off-by-one impact, but
  trace-diff tools must exclude the last frame.

---

## Phase 2 — Precision & determinism (static portion)

### 2.1 TF32 / matmul precision — **CONCERN + WIRING BUG**
- `matmul_precision` is a config knob (`jax_cache.py:16,35,48`) but **`setup_jax_cache` is called
  only in `train.py:4873` — NOT in the post-run/eval path.** So eval/post-run always runs at JAX
  default precision (**TF32 on Ampere+ GPUs**) regardless of config. The knob is *train-only*; the
  cross-hardware GPU results are therefore all at TF32. **To run the TF32 attribution test, wire
  `setup_jax_cache(cfg)` (or `jax.config.update("jax_default_matmul_precision", ...)`) into the
  post-run modal entry first** — otherwise a `matmul_precision=highest` override is a silent no-op.
  This wiring gap is itself a reproducibility finding (advertised knob, dead in eval).
- The archived S4 snapshot has `jax.matmul_precision: default` on an **A100** ⇒ TF32 in play for
  the paper numbers — consistent with the GPU-vs-CPU directional split (`SYMMETRY_BREAKING.md`).
  The clean attribution (does GPU-intact commit left at `highest`?) is **pending the one-line wire-up**.

### 2.3 Determinism — **PASS (static)**
- Loaded S4 has `p_update=1.0` (`config_snapshot.yaml:8`) ⇒ Bernoulli update mask is all-ones
  (no stochastic masking) and `noise_type=off` ⇒ OU channel absent (no `jax.random.normal`
  draw). With these, eval is **deterministic given a seed** (single-trial determinism also
  enforced in `post_run.py:5296-5298`). `ou_noise_seed: 12345` exists in config but is inert
  with noise off. No other eval-time RNG source identified.
- **float64/x64 is NOT a config knob** — only hard-coded in
  `preprint/scripts/s4_cross_hardware_state_space.py:310`. A config-driven `jax_enable_x64` knob
  must be added (mirroring `setup_jax_cache`) before the float32-vs-float64 cross-hardware check
  (2.2/3.1) can run through the core path.

---

## CRITICAL FINDING — config drift: the repo cannot configure the run that made S4

The archived `s4/results/bin_03_rank_00_cell_6-0-3_flat_0423/config_snapshot.yaml` (ground truth
for the paper specimen) differs materially from the repo preset
`sensing_edges/config/preset/paper_run.yaml`. This is the concrete
form of the "logical_config_sha256 mismatch / code-config drift" flagged in
`notes/16-04-2026.md:693-710`:

| Param | Archived (made S4) | Current repo preset | Impact |
|---|---|---|---|
| `train.max_steps` | **4000** | 3000 | Different episode horizon → different outcomes |
| `qd.n_emitters` | **12** (p300) | 8 (p200) | Different population → different evolution |
| `qd.archive.dims` | **[7,7,10]** | [7,7,20] | Different archive resolution / selection grid |
| `deformation_conditions` | `…deformation_reversed` (5) | `…noise_only, shuffled` (6) | Different ablation suite |
| `sim.airborne_kill_height_margin_cells` | **1.5** | 1.0 | Different fall predicate threshold |

**Consequence:** to reproduce S4 you must drive eval from the archived `config_snapshot.yaml`,
**not** the repo preset. Any "reproduction" using the current preset is a *different experiment*.
This belongs in the canonical-regime decision (the author's reruns) and in release packaging
(D0/Workstream D). Note the README's "6 emitters × 25 = 150" is a *third* population figure — all
three (150 / 200 / 300) appear in different sources; the archived run was **300** (`p300` in the
output-dir name `…p300…seed137_e4fcc2fb`).

---

## Phase 7 — Release hygiene (static; full inventory in `RELEASE_CLEANUP.md`)

- **7.1 Secrets/dead content — CONCERN:** hardcoded absolute paths confirmed
  (`s4/scripts/compare_probe_to_archived_summaries.py` `ROOT=/Users/<user>/…`;
  `preprint/scripts/s4_cross_hardware_state_space.py`). No inline credentials found yet (wandb is
  a Modal secret reference). Full grep inventory in D0.
- **7.3 License/provenance — FAIL (must fix):** no LICENSE; JAX-EvoGym / Evolution Gym lineage +
  physics-param provenance not stated.
- **7.4 Selection/leakage — PASS (static):** selection rule `elite_import.py:224-235`
  (`generation_asc, candidate_index_asc, archive_flat_index_asc, specimen_id_asc`);
  `_derived_specimen_id` `post_run.py:283-293`. S4 = `bin_03_rank_00` = top of completion-bin 3,
  cell `6-0-3`, archive flat 0423. Matches "high-performing archive corner." No eval/
  generalization-terrain leakage into selection observed in code. Reconfirm against the paper's
  stated rule when finalizing.

---

## Phase 4-5 results from the cross-hardware traces (2026-06-14)

### 4.2 Controller blindness, empirically — **PASS (exact, universal)**
In `deformation_off`, the controller state (`ctrnn_carry`) and hidden channels are **bit-identical
across maximally different terrains** (left-goal (2,1) vs right-goal (6,3)) for the **entire active
rollout**, on **every** runtime (A100-40/80, L4, T4, H100, CPU); the only nonzero diff is the final
post-termination frozen frame (1.6). The controller carries **exactly zero** terrain information —
a closed autonomous oscillator, not "near chance." Intact, by contrast, diverges by 222–400 (the
deformation input differs). This is the empirical core of the blindness claim and it holds even on
T4/H100 where the *behaviour* differs — that difference is purely plant-mediated. Source:
`SYMMETRY_BREAKING.md`, `CROSS_HARDWARE_READ.md`.

### 4.4 Directional flip — **PASS (canonical regime), mechanism characterised**
See `CROSS_HARDWARE_READ.md` + `SYMMETRY_BREAKING.md`: deformation-off commit is the intrinsic
LEFT handedness (all runtimes except T4); the deformation signal pushes right but only wins under
GPU precision (CPU ~no flip, GPU large flip). Symmetry is broken explicitly (asymmetric evolved
controller + 1-cell spawn offset), poised so finely that precision/TF32 selects the branch.

### 5.1 / 5.2 Physics / energy sanity — **PASS**
Across deformation-off traces (A100, CPU): velocities bounded (`max|v|≈9`), per-step velocity
change bounded (`max|Δv|≈9–13`), **no NaN/inf**, no blow-ups. The reversal does not exploit a
physics instability. (Frame-by-frame leading-edge render 5.1 still recommended for the figure, but
no instability signature in the numerics.)

## Deferred checks (specified; pending the author's reruns / go-ahead)

Phase 2.2 (float32 vs float64 — needs the x64 knob), Phase 3 (cross-hardware matrix, integrator
resolution sweep), Phase 4 (perturbation robustness 4.1, empirical blindness 4.2, no-input-route
counterfactual 4.3, directional flip 4.4), Phase 5 (edge/contact + energy/force sanity), Phase 6
(env pin, checkpoint round-trip, one-command repro, smoke test). See plan Workstreams C/D.

## Stop-condition status
- 1.4 geometry-term-in-controller: **not triggered** (blindness holds by construction).
- 5.1/5.2 physics artifact: **not assessed this pass** (deferred).
- 4.1 fragile-fluke: **not assessed** (needs perturbation runs).
- 2.x/3.1 incoherent-across-hardware: **open** — this is precisely what the author's reruns +
  Workstream B will decide.
