# bin_04_rank_00_cell_6-0-4_flat_0424

An intact cliff-task cohort specimen from the seed-42 `iter_run` with a 2000-step
episode horizon.

## Provenance

- **Preset:** `iter_run` (cliff, fixed 7x2 rectangle body, terrain variant
  `bilateral_goal_cliff_asymmetric_pairs_8x2_30`, sep-CMA-ES, 100 configured
  generations, population 150 from 10 emitters x 15 candidates, 15 physics substeps,
  2000 train / 2000 eval steps)
- **Seed:** 42
- **Run:** `iter_run_cliff_rect_7x2_sep_cmaes_g100_p150_15sub_t2000_e2000_ref003_seed42_d4611d8e`
- **Lineage:** branch-mode continuation from a generation-49 parent checkpoint; the
  parent checkpoint and its configuration are not included
- **Selection:** rank 0 in archive cell (6,0,4), flat index 424, taken at generation
  99 (the run's last; `stop_after_generations: 100`)
- **Deformation sensing:** on (intact condition)

The bundled `config_snapshot.yaml` is the authoritative configuration for the final
branch run. It supports specimen replay, but the public bundle cannot regenerate the
specimen's evolution from scratch. The live `sensing_edges/config/preset/iter_run.yaml`
has since changed (now 50 generations and 2500 steps) and no longer reproduces this run
as shipped. The manifest records the producing Git commit as `unknown`, so exact code
provenance is not pinned.

## Files

- `genotype.npz` — the specimen genome (NCA seed genotype)
- `controller_params.npz` — evolved NCA-CTRNN controller parameters
- `config_snapshot.yaml` — full resolved training configuration
- `manifest.json` — specimen manifest (identity, condition, selection metadata)
- `selection_summary.json` — archive selection summary for this specimen
- `renders/`
  - `pair_4_2_left.gif` / `pair_4_2_right.gif` — intact-controller replays of the
    matched (4,2) distance pair in left and right mirror orientations

This bundle carries the specimen's identity and replay inputs only: genome, evolved
controller, resolved config, and provenance. Full assay outputs, trace bundles, derived
analyses, and the stored-run replay root are available on request — open an issue.
