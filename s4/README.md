# S4

`s4` is the specimen audit layer: small, specimen-focused studies built on top of `sensing_edges`. Its audit documents call the canonical paper specimen **S4**, short for `bin_03_rank_00_cell_6-0-3_flat_0423`.

Current contents:

- `results/`: curated specimen bundles — genome, evolved controller parameters, resolved config, manifest, and representative renders for the eight-specimen cohort. Identity and replay inputs only; no assay summaries or derived analyses ship here.
- `audit/`: the validity-audit deliverables — reproducibility tag, claims ledger, audit report, cross-hardware read, and the symmetry-breaking note.
- `presets/`: small local probe presets for stored-specimen replay.
- `probe.py`: preset-driven local probe runner built on the `sensing_edges` stored-specimen post-run path.
- `scripts/`: offline analysis helpers that read a specimen bundle directly.

This directory is intentionally lightweight:

- no training code
- no local evaluation sweeps
- only specimen artifacts, audit documents, and small local study support

Training and evaluation live in `sensing_edges/`. The full run outputs these studies were originally computed from — assay trees, trace bundles, and derived analyses — are not stored in this repository; see the root README's Data section. `results/README.md` and `scripts/README.md` say which commands run against the shipped bundles and which need the full dataset.
