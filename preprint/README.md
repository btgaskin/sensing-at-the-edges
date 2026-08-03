# Preprint Analysis

Analysis and figure scripts for the preprint: the consolidation layer between
raw `sensing_edges` run outputs and the manuscript's tables and figures.

These scripts operate on post-run outputs (assay summaries, trace bundles,
specimen payloads) produced by the `sensing_edges` pipeline. Every one of them
resolves its inputs under a local `paper_data/` or `paper_workspace/` root — 23
of the 30 name it directly, and the figure builders inherit it from
`make_sensing_edges_revision_materials.py`. Those trees are not stored in this
repository (see the root README's Data section); they are available on request —
open an issue. Pointed at a clean clone the scripts raise `FileNotFoundError`.
The curated specimen bundles under `s4/results/` cover the identity/replay side
only; `s4/scripts/README.md` lists what runs against them.

## Script groups

- `make_sensing_edges_*.py` — manuscript figure and packet builders
  (bar figures, simple figures, tau histogram, clean packet, revision
  materials)
- `s4_*.py` — analyses centered on the canonical specimen, called **S4**
  throughout the code and the audit documents
  (`bin_03_rank_00_cell_6-0-3_flat_0423`): state-space PCA, propagation,
  fixed-width distance sweeps and their housekeeping, proximity and
  old-decoder decoding, mechanism packet, path-ensemble analysis,
  support-preserved control, and cross-hardware comparison
- `signaling_*.py` / `signal_susceptibility_analysis.py` — signaling-channel
  ablation and susceptibility analyses
- `native_core_ablation.py` — the native-core deformation-ablation suite
  driver (local and Modal execution)
- `dynamical_regime_analysis.py`, `state_space_pca_overlay.py`,
  `intercell_tension_network_analysis.py` — dynamical-systems analyses of
  controller and plant state
- `committed_direction_decoding.py`, `leading_edge_reversal_analysis.py`,
  `voxel_reversal_heatmap_analysis.py` — direction-commitment and
  edge-reversal analyses on deformation-off traces
- `build_statistics_requirements_packet.py` — assembles the camera-ready
  statistics requirements packet from the specimen episode manifests
- `update_eight_specimen_followups.py` — merges targeted follow-up assays into
  the eight-specimen preprint package and rewrites its episode manifest
- `modal_*.py` — Modal-side launchers for the heavier analyses (require your
  own Modal account; see the root README)

Scripts resolve the repository root relative to their own location and expect
the sibling `sensing_edges/` package to be importable (installed via
`uv sync` in `sensing_edges/`).
