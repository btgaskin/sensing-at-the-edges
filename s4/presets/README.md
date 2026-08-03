# Presets

Local probe presets for `s4/probe.py`, which replays a stored specimen through the
`sensing_edges` post-run path.

## What runs on a clean clone

Only preset listing. Everything else needs the stored-run replay root
(`s4/results/<specimen_id>/stored_run/`), which holds the original run's
`resolved_config.yaml` and specimen tree. That root is not stored in this repository; it
is available on request — open an issue. `--dry-run` is not an exception: it composes the
config from the stored run before it reports, so it fails the same way.

Run presets with:

```bash
cd /path/to/sensing-at-the-edges

# works on a clean clone
sensing_edges/.venv/bin/python s4/probe.py --list-presets
sensing_edges/.venv/bin/python s4/probe.py presets

# needs the stored-run replay root
sensing_edges/.venv/bin/python s4/probe.py pair42_trace_compare --dry-run
sensing_edges/.venv/bin/python s4/probe.py pair42_trace_compare
```

Multiple presets can be queued in one invocation (up to five), with the same stored-run
requirement:

```bash
sensing_edges/.venv/bin/python s4/probe.py pair42_trace_compare pair63_native_deformation_compare
```

Design constraints:

- presets are for small, local, deterministic probes only
- each preset keeps outputs under `s4/results/<specimen_id>/<preset_name>/`
- render-enabled presets write simple GIFs into the corresponding assay `renders/` directories under that preset root
- heavy experiment sweeps still belong on Modal

Each preset is a YAML file with:

- `description`
- `seed` (optional; defaults to 137)
- `jax_platform`
- `assays` (required, at least one)
- `config_patch`

A preset can also override `specimen_id`, `bundle_root`, `run_dir`, `specimen_root`,
`analysis_root`, and `pointer_root`. The built-in presets stay short because `probe.py`
already defaults those to the canonical specimen — including `run_dir` and
`specimen_root`, which point into the unshipped `stored_run/` root described above.

Terrain shorthand:

- `native1`, `native2`, `native3` map to `(2,1)`, `(4,2)`, `(6,3)`
- `long4`, `long5` map to `(8,4)`, `(10,5)`
- suffix `n` means the normal left-side orientation
- suffix `m` means the mirrored right-side orientation
