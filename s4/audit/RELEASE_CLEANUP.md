# Release Cleanup Notes (D0) — for publishing `sensing_edges` so it can be replicated

> **Historical snapshot (2026-06-14).** This is an audit-trail record of issues found during
> the S4 validity audit, preserved as part of the audit deliverables. Several items (license,
> hardcoded paths, repo layout) were addressed when this repository was assembled for public
> release; the config-drift finding (B1) remains an open reproducibility caveat. Read it as a
> record of what the audit found, not as a live to-do list.

**Date:** 2026-06-14. Audience: the release prep + library packaging pass.
Concrete issues found during the S4 audit that **break clean-checkout replication** or are
release-embarrassing. Grouped by severity. Each item names the evidence and the fix.

---

## P0 Blockers — a public clone cannot reproduce the paper without these

### B1. Config drift: the repo can't even configure the run that produced S4 *(new, verified)*
The archived ground-truth `s4/results/bin_03_rank_00_cell_6-0-3_flat_0423/config_snapshot.yaml`
differs from the current `sensing_edges/config/preset/paper_run.yaml`:

| Param | Archived (made S4) | Current preset |
|---|---|---|
| `train.max_steps` | **4000** | 3000 |
| `qd.n_emitters` | **12** (→ pop 300) | 8 (→ pop 200) |
| `qd.archive.dims` | **[7,7,10]** | [7,7,20] |
| `deformation_conditions` | `…deformation_reversed` (5) | `…noise_only, shuffled` (6) |
| `sim.airborne_kill_height_margin_cells` | **1.5** | 1.0 |

The README states a *third* population figure ("6 emitters × 25 = 150"); the archived run was
**300** (`p300` in `output_dir: …p300…seed137_e4fcc2fb`). **Fix:** decide the canonical config,
make the released preset reproduce the archived `logical_config_sha256` (or explicitly re-baseline
and update the paper's numbers), and reconcile the README population figure. Ship the archived
`config_snapshot.yaml` alongside the specimen as the authoritative config.

### B2. Three-way JAX version mismatch
Lockfiles pin `jax/jaxlib 0.9.0.1` (root + `sensing_edges/` + `s4/` `uv.lock`,
`pyproject.toml:7-8`); the archived paper run used **0.9.2** (S4 `manifest.json`); the Modal image
installs **unpinned** `jax[cuda12]` (`scripts/_modal_common.py:65`). Three versions. **Fix:** pick
one, pin in `pyproject.toml`, every `uv.lock`, and the Modal image. This is entangled with the
GPU/CPU numeric mismatch — the single biggest replication risk.

### B3. Only S4's genotype is committed
Only `bin_03_…_0423` has a committed genotype + provenance (plus one duplicate under
`stored_run/`). The other reported specimens live **only** in the Modal `se-outputs` volume.
**Fix:** export and commit (or archive with a documented fetch step) every specimen the paper
reports, each with `genotype_sha256` + `config_snapshot.yaml`.

### B4. Figure/number regeneration is Modal-only
`paper_protocol.py` `figures/core/eval/bundle` all shell `modal run …`;
`make_figures.py`/`make_tables.py` only post-process an already-materialized release dir. A
clone without Modal credentials can regenerate nothing. **Fix:** provide a local (CPU) path for
at least the core S4 numbers/figures, or document Modal + credentials as a hard requirement with
setup steps.

### B5. Unpinned Modal image dependencies
`_modal_common.py:61-78` installs `numpy, hydra-core, omegaconf, evosax>=0.2.0, matplotlib,
pandas, pyarrow, wandb, jsonschema, scipy` mostly unpinned ⇒ a fresh image build floats. **Fix:**
pin to the lockfile.

### B6. Dirty tree / no commit anchor
Many core files are edited-but-uncommitted (`env.py`, `post_run.py`, `fitness.py`,
`runtime_config.py`, configs). The archived run recorded `git.commit "unknown", dirty true`, so
no past run is anchorable. **Fix:** land a clean commit matching the released numbers and record
its hash in provenance going forward.

---

## P1 Hygiene — embarrassing if a reviewer/reader finds it post-release

### H1. Hardcoded absolute paths
e.g. `s4/scripts/compare_probe_to_archived_summaries.py` `ROOT=/Users/<user>/…`;
`preprint/scripts/s4_cross_hardware_state_space.py`. **Fix:** grep the repo for `/Users/<user>`
and parameterize (env var / CLI arg / repo-relative).

### H2. Missing license + lineage
No LICENSE. **Fix:** add one; note JAX-EvoGym / Evolution Gym lineage and physics-parameter
provenance.

### H3. Secrets sweep
wandb is referenced as a Modal secret (`wandb-key`) — fine. **Fix:** confirm no inline API
keys/tokens in code/configs/notebooks before publishing.

### H4. Docs that don't match reality
`preprint/README.md` prescribes a `preprint/` package + 5 named scripts that **do not
exist** (only differently-named ad-hoc scripts do). **Fix:** build the package or rewrite the
README so a reader isn't misled.

### H5. Uncommitted doc deletions
`research_dossier.md`, `plausibility.md`, `system_report.md`, `jax-evogym.md`,
`paper/intro_draft1.md`, and the research packet are deleted-but-unstaged. **Fix:** decide the
release doc set; keep only what produces paper results, drop infra/dead-config docs.

### H6. Inconsistent archived provenance
S4 `manifest.json` records GPU/jax-0.9.2 hardware but paths under
`/outputs/rerun_local_analyses/…`. **Fix:** tighten provenance recording so released manifests
are internally consistent.

### H7b. `matmul_precision` knob is dead in eval *(new, verified)*
`setup_jax_cache` (which applies `jax.matmul_precision`) is called only in `train.py:4873`, not in
the post-run/eval path — so eval always runs at JAX default precision (TF32 on Ampere+ GPUs) and
the config knob is silently ignored at eval. **Fix:** call `setup_jax_cache(cfg)` in the post-run
modal entry (`scripts/modal_post_run.py` / `sensing_edges/post_run.py`) so the documented knob
actually controls eval precision; needed for the TF32 reproducibility test and to honor the config.

### H7. Multiple lockfiles
Root, `sensing_edges/`, and `s4/` each have a `uv.lock`. **Fix:** declare the canonical one for
the released package and ensure consistency.

---

## Quick verification commands (for the release pass)

```bash
# H1/H3 — paths & secrets sweep
grep -rn "/Users/<user>" . --include=*.py
grep -rniE "api[_-]?key|secret|token|password" . --include=*.py --include=*.yaml

# B2/B7 — jax pin consistency
grep -rn "jax" . --include=uv.lock --include=pyproject.toml --include=_modal_common.py | grep -i "0.9"

# B1 — config drift vs the archived snapshot
diff <(yq -P s4/results/bin_03_rank_00_cell_6-0-3_flat_0423/config_snapshot.yaml) \
     <(uv run python -c "import sensing_edges.runtime_config as r; ...compose paper_run...")
```
