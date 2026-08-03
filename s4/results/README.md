# Results

Curated specimen bundles. Each candidate directory carries the minimum needed to
identify, replay, and inspect one robot:

- `DESCRIPTION.md` — provenance and file inventory, with verified results only where
  they are available
- `genotype.npz` + `controller_params.npz` — the genome and evolved controller
- `config_snapshot.yaml` + `manifest.json` — resolved configuration and specimen manifest
- `selection_summary.json` — archive selection detail when included (seven bundles)
- `renders/` — representative episode or matched-pair GIFs

Array contents, identical across all eight bundles:

- `genotype.npz` holds one array, `genotype`: float32, shape `(624,)` — the evolved
  parameter vector as CMA-ES searched it.
- `controller_params.npz` holds the same 624 numbers unpacked into the CTRNN:
  `w_input` `(30, 12)`, `w_rec` `(12, 12)`, `w_output` `(12, 8)`, `bias` `(12,)`,
  `log_tau` `(12,)`, all float32 — 30 perception inputs, 12 hidden neurons, 8 outputs.
- The pack order is `w_input`, `w_rec`, `w_output`, `bias`, `log_tau`
  (`sensing_edges.nca.ctrnn_flat_to_params`), so either file reconstructs the controller.

## Cohort

| Specimen | Preset | Seed | Run | Lineage | Population (emitters x candidates) | Episode horizon (train / rerank / test) |
| --- | --- | ---: | --- | --- | ---: | ---: |
| `bin_03_rank_00_cell_5-0-3_flat_0353` | `paper_run` | 1632 | `27f7e19b` | No recorded branch | 150 (10 x 15) | 3000 / 3000 / 3000 |
| `bin_03_rank_00_cell_6-0-3_flat_0423` (**S4**) | `paper_run` | 137 | `e4fcc2fb` | No recorded branch | 300 (12 x 25) | 4000 / 4000 / 4000 |
| `bin_04_rank_00_cell_5-0-4_flat_0354` | `paper_run` | 1564 | `a792f078` | No recorded branch | 300 (12 x 25) | 4000 / 4000 / 4000 |
| `bin_04_rank_00_cell_6-0-4_flat_0424` | `iter_run` | 42 | `d4611d8e` | Branch from generation 49 | 150 (10 x 15) | 2000 / 2000 / 2000 |
| `bin_04_rank_01_cell_6-0-4_flat_0424` | `paper_run` | 137 | `e4fcc2fb` | No recorded branch | 300 (12 x 25) | 4000 / 4000 / 4000 |
| `bin_05_rank_02_cell_5-0-5_flat_0355` | `paper_run` | 137 | `e4fcc2fb` | No recorded branch | 300 (12 x 25) | 4000 / 4000 / 4000 |
| `bin_06_rank_00_cell_4-0-6_flat_0286` | `iter_run` | 42 | `be529c33` | Branch from generation 49 | 150 (10 x 15) | 3000 / 3000 / 3000 |
| `bin_06_rank_01_cell_5-0-6_flat_0356` | `iter_run` | 42 | `d4611d8e` | Branch from generation 49 | 150 (10 x 15) | 2000 / 2000 / 2000 |

The `Run` column is the logical-config hash that suffixes each source run directory;
specimens sharing a hash came from the same evolutionary run. Seed 42 contributed two
distinct runs, so the hash rather than the seed identifies a run. Each bundle's
`DESCRIPTION.md` gives the full run directory name.

The three branch-mode specimens resumed from generation-49 parent checkpoints.
The parent checkpoints and their configurations are not included in this public
release.

The five source runs do not share one configuration. Four specimens use population 300
with 4000-step training and evaluation horizons; two use population 150 with 3000-step
horizons; and two use population 150 with 2000-step horizons. Each bundle's own
`config_snapshot.yaml` is the authoritative record of the final run's configuration.
For the three branch-mode specimens, the bundle supports specimen replay but not
evolutionary regeneration from scratch. Every manifest records the producing Git
commit as `unknown`, so exact code provenance is not pinned. Cohort-level spread should
be read with this training-budget variation in mind.

The audit documents under `s4/audit/` use **S4** for the canonical paper specimen,
`bin_03_rank_00_cell_6-0-3_flat_0423`.

The live presets under `sensing_edges/config/preset/` have since changed and no longer
reproduce these runs as shipped. The manifests, configuration snapshots, and seven
included selection summaries are frozen source artifacts. Absolute `/outputs` paths in
these files point at the machine that produced them, not at this repository.

The shipped manifests are the original provenance records with operational failure logs
and provider run identifiers removed; no other fields were changed.

These bundles carry identity and replay inputs only. The assay trees, trace bundles,
derived analyses, and stored-run replay roots behind the paper figures are not stored in
this repository; they are available on request — open an issue.

That split decides what runs on a clean clone. The `s4/scripts/` probes that read a
bundle directly work here; the `s4/probe.py` presets and the analyses that read archived
assay output do not. `s4/scripts/README.md` and `s4/presets/README.md` say which is which.
