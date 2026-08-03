# bin_06_rank_01_cell_5-0-6_flat_0356

A comparison specimen from an earlier iteration run, used as the counterpart in the
deterministic deformation-suite comparison against the canonical paper specimen
`bin_03_rank_00_cell_6-0-3_flat_0423` (**S4**).

## Provenance

- **Preset:** `iter_run` (cliff, rect 7x2 body, sep-CMA-ES, 100 generations, population 150, 15 substeps, 2000 train / 2000 eval steps)
- **Seed:** 42
- **Run:** `iter_run_cliff_rect_7x2_sep_cmaes_g100_p150_15sub_t2000_e2000_ref003_seed42_d4611d8e`
- **Lineage:** branch-mode continuation from a generation-49 parent checkpoint; the
  parent checkpoint and its configuration are not included
- **Selection:** rank 1 in archive cell (5,0,6)
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
- `renders/rank_0_ep_0..2.gif` — evaluation episode replays

This bundle carries the specimen's identity: genome, evolved controller, resolved
config, and provenance. Full assay outputs and derived analyses are part of the
complete dataset, available on request.
