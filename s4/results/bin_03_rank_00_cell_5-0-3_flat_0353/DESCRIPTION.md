# bin_03_rank_00_cell_5-0-3_flat_0353

An intact cliff-task cohort specimen from the seed-1632 `paper_run`.

## Provenance

- **Preset:** `paper_run` (cliff, fixed 7x2 rectangle body, terrain variant
  `bilateral_goal_cliff_asymmetric_pairs_7x2_grid30`, sep-CMA-ES, 150 configured
  generations, population 150 from 10 emitters x 15 candidates, 15 physics substeps,
  3000 train / 3000 eval steps)
- **Seed:** 1632
- **Run:** `paper_run_cliff_rect_7x2_sep_cmaes_g150_p150_15sub_t3000_e3000_ref003_seed1632_27f7e19b`
- **Selection:** rank 0 in archive cell (5,0,3), flat index 353, taken at generation
  99 (the run's last; `stop_after_generations: 100`)
- **Deformation sensing:** on (intact condition)

The bundled `config_snapshot.yaml` is the authoritative provenance for these numbers; the
live `sensing_edges/config/preset/paper_run.yaml` has since changed (now population 200
and 3000 steps) and no longer reproduces this run as shipped.

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
