# Scripts

Analysis utilities for the specimen bundles under `s4/results/`.

## What runs on a clean clone

The bundles that ship carry a specimen's identity — genome, controller parameters,
resolved config, manifest, renders. They do not carry the archived assay output the
original studies were computed from.

So the scripts split three ways, and each entry below is tagged:

- **[bundle]** — reads a shipped bundle directly; runs on a clean clone.
- **[derived]** — reads the output of a **[bundle]** script; run that one first.
- **[full data]** — reads archived assay trees or stored-run replay roots that are not in
  this repository. These are available on request — open an issue. Pointed at the shipped
  bundles they raise `FileNotFoundError`, by design.

## Specimen inspection

- `controller_weight_map.py` **[bundle]**
  Load `controller_params.npz`, map the full CTRNN parameter set, and render simple heatmaps plus summary stats.

## Deformation-ablation suite

- `deterministic_deformation_suite.py` **[bundle]**
  Run the current-code deterministic deformation-ablation suite for the canonical specimen across native core, long-distance generalization, and double-cliff surfaces, with one GIF per deterministic episode/group.
- `deformation_suite_clean_charts.py` **[derived]**
  Compact summary charts for one suite run. Takes `--suite-root` pointing at a `deterministic_deformation_suite.py` output directory.
- `compare_deformation_suite_charts.py` **[derived]**
  Direct comparison charts across several suite runs. Takes repeated `--suite LABEL=/path/to/suite_root`.

## Terrain and task probes

- `terrain_width_compare_probe.py` **[bundle]**
  Replay the specimen across cliff terrains of varying plateau and ledge width, with shared-PCA state plots and stacked robot renders. Also the shared render/trace toolkit the other compare probes import.
- `double_cliff_compare_probe.py` **[bundle]**
  Native double-cliff width comparison, intact against `deformation_off`, with the same shared PCA and render outputs.
- `native_core_task_compare_probe.py` **[bundle]**
  Compare the mirrored native-core (2,1) tasks against a broadened bilateral terrain.
- `right_native_pair_compare_probe.py` **[bundle]**
  Compare two right-native bilateral terrains in block-off and block-touchable variants.
- `right_native_goalblock_doublecliff_probe.py` **[bundle]**
  Compare right-native (2,1) goal-block variants against the native double-cliff at width 10.
- `rerender_double_cliff_compare_bundle.py` **[derived]**
  Rerender an existing `double_cliff_compare_probe.py` output with the current stacked-robot and 3D PCA views. Takes `--input-dir`.

## Free-space branch workflow

- `free_space_branch_capture.py` **[bundle]**
  Capture a single free-space `deformation_off` branched rollout with a plant-only support-release perturbation, and save a reusable `branch_trace.npz`.
- `free_space_branch_render.py` **[derived]**
  Consume a saved branch trace and render stacked natural robot videos/GIFs in `tight` and `world` camera modes, with a full-frame flash at perturbation time.
- `free_space_branch_state_pca.py` **[derived]**
  Consume a saved branch trace and render a separate PCA video/GIF with plant state built from raw absolute per-cell H/V extension ratios and controller state kept separate.
- `free_space_branch_composite.py` **[derived]**
  Compose the plant PCA, robot render, and controller PCA from the three steps above into one synchronized video.
- `free_space_branch_common.py` — helper module, not an entrypoint.
  Bundle loading, environment construction, rollout capture, PCA, and frame encoding shared by the free-space and compare scripts.
- `free_space_branch_pca_render.py` **[bundle]**
  Legacy mixed composite script. Superseded by the split capture/render/PCA workflow above.

## Dynamics and spectra

- `deformation_off_wave_probe.py` **[bundle]**
  Capture a lightweight `deformation_off` controller trace on the widest double-cliff width, then measure recurrence and perturbation convergence in action and internal-state space.
- `state_pca_spectral_purity.py` **[derived]** / **[full data]**
  Spectral-purity analysis of state-PCA trajectories. The branch half (`--skip-matched`) reads a `free_space_branch_state_pca.py` output; the matched intact vs `deformation_off` half needs an archived `pca_bundle`.
- `deformation_absolute_pca.py` **[full data]**
  Fit a shared PCA on the saved absolute deformation channels (`abs_h_delta`, `abs_v_delta`) across the intact `pca_bundle` and the deformation-off cohort.
- `pca_frequency_analysis.py` **[full data]**
  Fit PCA on the intact selected trace, project intact and deformation-off into the same PC space, and compare their FFT spectra.
- `compare_probe_to_archived_summaries.py` **[full data]**
  Cross-check local probe outputs against the archived post-run summary CSVs for the same terrain pairs.

## Running them

From the repo root with the `sensing_edges` environment. Every command here is
**[bundle]** or **[derived]**, so all of it runs on a clean clone:

```bash
cd /path/to/sensing-at-the-edges
sensing_edges/.venv/bin/python s4/scripts/controller_weight_map.py
sensing_edges/.venv/bin/python s4/scripts/deterministic_deformation_suite.py --count-only
sensing_edges/.venv/bin/python s4/scripts/deformation_off_wave_probe.py --max-steps 1500
sensing_edges/.venv/bin/python s4/scripts/free_space_branch_capture.py --max-steps 360
sensing_edges/.venv/bin/python s4/scripts/free_space_branch_render.py --max-steps 360
sensing_edges/.venv/bin/python s4/scripts/free_space_branch_state_pca.py --max-steps 360
sensing_edges/.venv/bin/python s4/scripts/free_space_branch_composite.py --max-steps 360
```

`deformation_absolute_pca.py`, `pca_frequency_analysis.py`, and
`compare_probe_to_archived_summaries.py` are **[full data]** and are deliberately absent
from that list: they read archived `pca_bundle` trees that do not ship.

Every script defaults to the canonical bundle under
`s4/results/bin_03_rank_00_cell_6-0-3_flat_0423` and writes under that bundle's
`analysis_runs/`; the free-space workflow uses
`analysis_runs/free_space_branch_steps_<max_steps>`. Scripts that expose `--bundle-root`
can be pointed at another bundled specimen; the rest take explicit
`--config-snapshot-path`, `--controller-params-path`, and `--genotype-path` overrides.
