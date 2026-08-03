# bin_03_rank_00_cell_6-0-3_flat_0423

The canonical high-performing cliff specimen from the paper runs. The audit documents
under `s4/audit/` refer to it as **S4**.

## Provenance

- **Preset:** `paper_run` (cliff, rect 7x2 body, sep-CMA-ES, 150 generations, population 300, 15 substeps, 4000 train / 4000 eval steps)
- **Seed:** 137
- **Run:** `paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed137_e4fcc2fb`
- **Selection:** rank 0 in archive cell (6,0,3), taken at generation 99 (the run's last;
  `stop_after_generations: 100`)
- **Deformation sensing:** on (intact condition)

The bundled `config_snapshot.yaml` is the authoritative provenance for these numbers; the
live `sensing_edges/config/preset/paper_run.yaml` has since changed (now population 200
and 3000 steps) and no longer reproduces this run as shipped.

## Files

- `genotype.npz` — the specimen genome (NCA seed genotype)
- `controller_params.npz` — evolved NCA-CTRNN controller parameters
- `config_snapshot.yaml` — full resolved training configuration
- `manifest.json` — specimen manifest (identity, condition, selection metadata)
- `renders/`
  - `rank_0_ep_0..2.gif` — evaluation episode replays
  - `pair_4_2_left.gif` / `pair_4_2_right.gif` — the matched (4,2) cliff-first pair (intact vs deformation-off) analyzed in the paper
  - `width_6.gif` — double-cliff generalization probe at width 6

## Headline results

From the matched (4,2) cliff-first pair:

- Intact peak overhang: 0.307 world units (0.44 body lengths); deformation-off: 0.386 world units (0.55 body lengths) — a 1.26x deeper cliff commitment when deformation sensing is ablated.
- Both conditions reached the cliff frontier and recovered behind it; the ablated controller committed further over the edge before recovering.
- Intact terrain series shows mean max cliff protrusion declining with gap width: 0.104 at (2,1) down to 0.065 at (8,4).

This bundle carries the specimen's identity: genome, evolved controller, resolved config, and provenance. Full assay outputs, trace bundles, derived analyses, and the stored-run replay root used by `s4/probe.py` are part of the complete dataset, available on request.
