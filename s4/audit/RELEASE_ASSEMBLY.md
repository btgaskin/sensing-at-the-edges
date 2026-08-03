# Release Assembly Blueprint — `sensing-at-the-edges`

> **Historical snapshot (2026-06-14).** This blueprint from the S4 audit described how to
> assemble this public repository. It was carried out in its essentials — the split from
> JAX-EvoGym into a pinned library dependency, the curated specimen bundles, the rewritten
> READMEs, the license and lineage notes, and shipping this audit layer. Eight specimens are
> exported. There is still no one-command local reproduction path (§8). The repository and
> Modal image pin JAX 0.9.0.1, while the archived run used 0.9.2; the Modal image installs
> `jax-evogym` from the pinned `v0.1.0` tag (§3). The config drift between the archived
> snapshot and the repo preset remains (§7), and the realized repository layout differs from
> the §1 target in places. Preserved as part of the audit trail, not as pending work.

**Purpose.** Guidance for assembling a clean, self-contained experiment-release repo for the
"Sensing at the Edges" paper, to be committed clean and published **after** JAX-EvoGym is
released as a library. This is the blueprint — what goes in, what gets cleaned, the core
results, versions, genotypes, and the curated assay/analysis scripts.

**Relationship to JAX-EvoGym.** JAX-EvoGym ships separately as a library (the physics engine +
core API). This repo is the *experiment layer*: it **depends on a pinned JAX-EvoGym version** and
bundles everything needed to reproduce the paper — the `sensing_edges` experiment package, the
canonical config, the committed genotypes, the reproducibility tag, and the extended-assay /
analysis scripts. A reader should be able to `pip install`/`uv sync` against the pinned stack and
regenerate the core S4 results without Modal (CPU path) or with Modal (GPU cross-hardware path).

> Validation backing for everything below lives in `s4/audit/`:
> `AUDIT_REPORT.md` (per-check verdicts), `CLAIMS_LEDGER.md` (claim→command→value),
> `S4_REPRODUCIBILITY.md` (the definitive tag), `CROSS_HARDWARE_READ.md`,
> `SYMMETRY_BREAKING.md`, `RELEASE_CLEANUP.md` (the cleanup list).

---

## 1. Target repo layout

```text
sensing-at-the-edges/
  README.md                     # install, reproduce-each-figure, hardware caveats (TF32)
  LICENSE                       # + JAX-EvoGym / Evolution Gym lineage note
  pyproject.toml                # pins jax-evogym==<ver>, jax==<ver>, all deps
  uv.lock                       # single canonical lockfile
  REPRODUCIBILITY.md            # = S4_REPRODUCIBILITY.md (the tag), generalized to all specimens
  sensing_edges/                # the cleaned experiment package (from sensing_edges/sensing_edges/)
  config/                       # hydra config tree (the canonical paper_run + assays)
  schemas/  terrain_suites/     # validation schemas + the seed_137 terrain suite used by S4
  specimens/                    # committed genotypes + provenance + hashes (see §5)
  scripts/
    pipeline/                   # train / eval / post_run / release entrypoints (Modal + local)
    assays/                     # extended-assay launchers (cross-hardware, ablation, state-space)
    analysis/                   # figure/table/decoding analyses (curated, see §6)
  results/                      # the shipped core numbers + figures (paper payload)
  docs/                         # method docs that map 1:1 to paper results
  audit/                        # this validation layer (optional but recommended to ship)
```

## 2. Core results to ship (the scientific payload)

Lead the README/results with these, all S4 (`bin_03_rank_00_cell_6-0-3_flat_0423`, seed 137):

1. **Controller blindness — exact, not statistical.** In `deformation_off` the controller state
   is **bit-identical across all terrains** for the whole active rollout, on every GPU and CPU
   (`max|Δctrnn|=0`). The controller is a closed autonomous oscillator carrying zero terrain
   information; intact differs by 222–400. *(empirical core of the claim)*
2. **Safety is invariant; completion is poised.** Across A100-40/80, H100, L4, T4, CPU and JAX
   0.9.0.1↔0.9.2, the body **never falls** (`fall_count=0` over 110 conditions). Deformation-off
   native-core success = **100% on A100-40/80 + L4** (reproducing the archive); degradations on
   H100/T4 are *safe* modes (stall / reverse-without-goal), never falls → **structured
   sensitivity, not artifact**.
3. **Directional flip + symmetry breaking.** Task is mirror-symmetric; the agent has intrinsic
   LEFT handedness (deformation-off → left on all but T4). The deformation signal pushes right but
   only wins under GPU precision (CPU ~no flip, GPU large flip). Symmetry is broken explicitly
   (asymmetric evolved controller + 1-cell spawn offset), poised so finely that TF32-vs-float32
   selects the branch. *(state this hardware caveat plainly in the README)*
4. **Physics is clean.** Bounded velocities/forces, no NaN/inf, no instability exploited at the
   edge.

## 3. Canonical versions & pins (from `S4_REPRODUCIBILITY.md`)

- **Regime:** A100-40GB · JAX/jaxlib **0.9.2** · Python 3.11.12 · NumPy 2.4.4 · evosax 0.2.0 ·
  `matmul_precision=default` (TF32 active). Reproduces bit-identically on A100-80GB + L4.
- **Resolve the 3-way JAX mismatch** before release: lockfiles pin 0.9.0.1, archive ran 0.9.2,
  Modal image is unpinned. Pin **one** version everywhere (pyproject + uv.lock + Modal image).
- **Pin JAX-EvoGym** to the exact released library version (currently an editable local path).
- Physics: `rk4`, 15 substeps, dt 1e-4, 3 phases × 5 NCA/substeps, spring 0.8, max_steps 4000.

## 4. What to INCLUDE (carry over, cleaned)

| Category | From | Notes |
|---|---|---|
| Experiment package | `sensing_edges/sensing_edges/*.py` | the engine of record; clean dead branches (§7) |
| Config tree | `sensing_edges/config/` | ship the **canonical paper config** = the archived `config_snapshot.yaml`, not the drifted preset (see `AUDIT_REPORT.md` config-drift) |
| Terrain suite | `terrain_suites/seed_137_…_g150_ep6_val6_test6` | the exact suite S4 used; drop the other seeds/variants unless a result needs them |
| Schemas | `sensing_edges/schemas/` | manifest/config validation |
| Pipeline entrypoints | `scripts/{modal_train,modal_eval,modal_post_run,modal_release,paper_protocol,make_figures,make_tables,_modal_common}.py` | the canonical pipeline |
| Specimens | §5 | committed genotypes + provenance + hashes |
| Repro tag | `audit/S4_REPRODUCIBILITY.md` | promote to top-level `REPRODUCIBILITY.md` |
| Validation layer | `s4/audit/*` | recommend shipping as `audit/` — it *is* the camera-ready validity record |

## 5. Genotypes / specimens to commit

- **S4 is the only genotype currently committed locally** (`genotype.npz` sha256
  `dc13ba50…cf02`, `controller_params.npz` `7b55de00…`). It is present and verified bit-identical
  across the cross-hardware package and `source_specimen`.
- **Export the rest from the Modal `se-outputs` volume** (the 5 "preprint" / 8-followup set) and
  commit each with `genotype.npz`, `config_snapshot.yaml`, `selection_summary.json`, and a
  recorded `genotype_sha256`. Confirm save/load round-trips bit-identically (6.2).
- Store under `specimens/<specimen_id>/` with a top-level `specimens/manifest.csv` (id, label,
  hash, seed, archive cell, fitness).

## 6. Extended-assay & analysis scripts (curate — keepers vs sprawl)

**Keep (extended assays):**
- `scripts/modal_post_run.py` — the fixed assay battery (native_core, long_distance, double_cliff,
  deformation + signaling ablation, state_space_trace[/_deformation_off], body transfer).
- `preprint/scripts/s4_cross_hardware_state_space.py` — the cross-hardware harness
  that produced the GPU package (parameterize its hardcoded paths first, §7).
- `preprint/scripts/native_core_ablation.py` — ablation matrix.

**Keep (analyses — promote the clean ones):**
- `committed_direction_decoding.py`, `state_space_pca_overlay.py`,
  `leading_edge_reversal_analysis.py`, `dynamical_regime_analysis.py`,
  `voxel_reversal_heatmap_analysis.py`, `signal_susceptibility_analysis.py`,
  `intercell_tension_network_analysis.py` — these are the substantive figure/analysis generators.
- `s4/audit/regime_classifier.py` — the regime/structured-vs-artifact classifier (new).
- `make_figures.py` / `make_tables.py` / `publication.py` — figure/table generation.

**Curate down, don't ship wholesale:** `s4/scripts/` (~21 ad-hoc probes) and the
`make_sensing_edges_*_packet.py` family are exploratory. Pick the ones a paper figure depends on;
move the rest out of the release (note them as artifacts, §7).

## 7. Artifacts & code to CLEAN before release (see `RELEASE_CLEANUP.md` for the full list)

- **Config drift (B1):** make the released config reproduce the archived S4
  `logical_config_sha256`, or re-baseline and update numbers. The repo preset currently can't
  configure the run that made S4.
- **`matmul_precision` dead in eval (H7b):** `setup_jax_cache` is train-only; wire it into the
  post-run entry so the precision knob works at eval (also unblocks the TF32 attribution test).
- **Hardcoded absolute paths (H1):** `s4/scripts/compare_probe_to_archived_summaries.py`,
  `preprint/scripts/s4_cross_hardware_state_space.py`, etc. — grep `/Users/<user>`, parameterize.
- **Unpinned Modal image deps (B5)** and **3-way JAX mismatch (B2)** — pin.
- **Aspirational `preprint/README.md` (H4):** describes a package that doesn't exist — rewrite or
  build before anyone reads it.
- **Probe sprawl:** `s4/` (probe_runs, analysis_runs, many one-off scripts) and the
  exploratory packet generators are working artifacts, not release material. Keep the genotype +
  provenance + the figures a paper uses; drop the rest.
- **License + lineage (B?/H2):** add LICENSE + JAX-EvoGym/Evolution Gym provenance note.
- **Secrets sweep (H3):** confirm no inline tokens.

## 8. One-command reproduction (target)

Provide `scripts/reproduce_s4.sh` (or a `paper_protocol.py reproduce` subcommand) that, from a
clean checkout against the pinned stack:
1. loads S4 from `specimens/` (verify hash),
2. runs the native-core assay (CPU local path — no Modal needed) under the canonical config,
3. asserts the headline (no falls; deformation-off native-core success; controller-blindness
   `max|Δctrnn|=0`), and emits the core table + figures.
The Modal/GPU cross-hardware path is a documented optional extension (records GPU type + JAX ver).
Today `paper_protocol.py figures` is Modal-only — **closing the local path is required** (B4).

## 9. Open items & sequencing before release

1. **Pending author runs:** local JAX-0.9.2 + CPU deformation-off → completes the version×backend
   cell (CPU-0.9.0.1 doesn't flip; 0.9.2-CPU says backend-vs-version). Run `regime_classifier.py`.
2. **TF32 attribution:** wire the precision knob (§7), then one GPU run at `matmul_precision=highest`
   — if GPU-intact then commits left (= CPU), the flip is TF32. *(held until wired — would be a
   no-op now)*.
3. **Export + commit the remaining specimens** from `se-outputs` (§5).
4. **Resolve config drift + JAX pin** (§3, §7).
5. **Close the local reproduction path** + smoke test (§8).
6. **Curate scripts** into `scripts/{pipeline,assays,analysis}` and strip artifacts (§6, §7).
7. **Optional extra assays** (cheap on Modal now): perturbation robustness (4.1), no-input-route
   counterfactual (4.3), integrator-resolution sweep (3.2) — add if a reviewer would expect them.

> Release gate at the time of writing: push only once JAX-EvoGym is public and pinned. That
> condition is now met — this repository depends on the released `jax-evogym` library at a
> pinned tag.
