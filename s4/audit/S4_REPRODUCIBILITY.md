# S4 — Definitive Reproducibility Tag

> **Historical snapshot (2026-06-14).** A preserved point-in-time record from the S4 validity
> audit, published as part of the audit trail. The pins, hashes, physics schedule, and
> controller/body spec below are verifiable against the specimen bundle shipped in
> `s4/results/`; paths into `paper_data/…` and the Modal `/outputs/…` volume, and the items
> marked pending, describe the pre-release working environment and are not part of this
> repository (full run datasets available on request). Values are unedited.

**Purpose:** the single authoritative spec for reproducing specimen **S4**. Everything here is
pinned to the exact artifacts that produced the paper's S4 results. Treat this file as the source
of truth; if the repo preset disagrees, the repo preset is wrong (see `RELEASE_CLEANUP.md` B1).

Confirmed cross-hardware on 2026-06-14: this setup reproduces the deformation-off native-core
headline **bit-identically on A100-40GB, A100-80GB, and L4**; safety (no fall) holds on every
device tested. See `CROSS_HARDWARE_READ.md`.

---

## Identity & integrity

| Field | Value |
|---|---|
| Specimen ID | `bin_03_rank_00_cell_6-0-3_flat_0423` |
| Seed | **137** |
| Generation | `gen_0099` (final; `stop_after_generations=100`) |
| Archive cell | `6-0-3` (goal_hit=6, fall=0, completion-bin 3), flat index 0423 |
| `genotype.npz` SHA-256 | `dc13ba50718e6e27a21546e0a4a7f832acb3233fcbd57aafc11ceaa44409cf02` |
| `controller_params.npz` SHA-256 | `7b55de00efca315036c27c560f1321b1e44d097c8c50916b2d01700b563e8576` |
| Run dir name | `paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed137_e4fcc2fb` |
| `logical_config_sha256` | `e4fcc2fb…` (suffix of run dir; full in `s4/results/bin_03_rank_00_cell_6-0-3_flat_0423/manifest.json`) |
| `resolved_config_sha256` | `646ba6e6…` |
| Canonical config file | `s4/results/bin_03_rank_00_cell_6-0-3_flat_0423/config_snapshot.yaml` |
| Canonical specimen bundle | `paper_data/preprint/s4_cross_hardware_state_space_20260614/specimen/stages/gen_0099/specimen_bin_03_rank_00_cell_6-0-3_flat_0423/` (internal; the shipped equivalent is `s4/results/bin_03_rank_00_cell_6-0-3_flat_0423/`) |

## Canonical execution regime (the reported numbers)

| Field | Value |
|---|---|
| Hardware | **NVIDIA A100-40GB** (GPU, `cuda`) |
| JAX / jaxlib | **0.9.2 / 0.9.2** |
| Python | 3.11.12 · NumPy 2.4.4 · evosax 0.2.0 |
| Platform | Linux x86_64 (glibc 2.36) |
| `matmul_precision` | `default` (**TF32 active on A100**). NB: the config knob is **not wired into eval** (`setup_jax_cache` is train-only, `train.py:4873`), so eval always runs JAX-default precision regardless of config — see `RELEASE_CLEANUP.md` H7b. To control eval precision the knob must be wired into the post-run entry. |
| Reproduces bit-identically on | A100-80GB, L4 |
| Documented sensitivity tail | H100 (stalls before edge), T4 (reverses without completing) — *safe modes, never falls* |

## Reproducibility guarantee (verified 2026-06-14)

The canonical regime **reproduces the archived paper numbers to float32 epsilon**. Deformation-off
native-core (4,2), new A100-40GB vs archived `pca_bundle/cohorts/deformation_off_native_core`:

| Trial | archived `final_com_x` / steps | new A100-40 | Δ |
|---|---|---|---|
| `pair_4_2_left_000` | 0.953816 / 1759 | 0.953816 / 1759 | −3.6e-07 / 0 |
| `pair_4_2_right_000` | 2.042928 / 2633 | 2.042928 / 2633 | −4.8e-07 / 0 |

- **A100-40, A100-80, L4 are bit-identical to each other** (GPU determinism within a fixed env;
  no atomic-nondeterminism issue) and all match the archive to ~1 ULP. CPU does **not** reproduce
  (0.954 → 1.386) — reproducibility is guaranteed on the **GPU canonical regime, not CPU**.
- The ~3e-7 new-vs-archived residual (while the new runs are mutually bit-identical) is attributable
  to **CUDA/cuDNN/XLA/driver drift** between the archived and new A100 runs — not nondeterminism.

## Known gaps for *exact* (bit-for-bit) reproduction

1. **Full GPU stack not captured/pinned.** Provenance records jax/jaxlib/python/numpy but **not
   CUDA, cuDNN, XLA, or driver versions** — the likely source of the ~3e-7 residual. Capture and
   pin these (or accept float32-epsilon tolerance as the reproduction bar).
2. **No artifact checksum manifest.** Ship a `SHA256SUMS` over all genotypes, configs, and the
   terrain suite (only `genotype.npz` is currently hashed).
3. **Terrain suite not content-hashed.** `terrain_suites/seed_137_…` is a reproduction input;
   record its hash.
4. **XLA determinism flags undocumented.** Empirically deterministic (A100-40=A100-80=L4 exact),
   but `--xla_gpu_deterministic_ops` state isn't recorded; document it.
5. **No CPU-canonical baseline.** If CPU reproduction is desired, a CPU regime must be frozen as
   its own reference (it will not match GPU).

## Physics schedule (frozen)

`integrator=rk4`, `physics_substeps=15`, `dt=1e-4`, `sense_act_phases_per_env_step=3`,
`nca_steps_per_phase=5` (⇒ per env step: 3×[5 NCA + 5 substeps]), `spring_stiffness_scale=0.8`,
`actuator_convergence=0.003`, `floor_static_friction_const=0.4`,
`airborne_kill_height_margin_cells=1.5`, `train.max_steps=4000`.

## Controller & body (SBT, 624 params)

`nca=sbt`: `ctrnn_hidden=12`, `ctrnn_grid_hidden=6`, `ctrnn_tau_min/max=1.0/10.0`,
**`p_update=1.0`** (deterministic update), **`noise=off`** (⇒ **30-d perception**, no OU channel),
`active_comm_filters=[sobel_x, sobel_y, laplacian, identity]`, `ablate_deformation=false`,
`deformation_gain_absolute=10.0`, `deformation_gain_compliance=85.0` (`auto_scale=true`),
`mace=off`, `gasnet=off`, `state_clip_value=3.0`.
Body: **`rect_7x2` fixed** (14 contractile cells, `hidden_init_scale=0`).

## Terrain

`bilateral_goal_cliff_asymmetric_pairs_7x2_grid30`, `task_mode=bilateral_goal_cliff`, grid 10×30,
`platform_y=3`, `center_plateau_width=12`, spawn `(11,4)`, `cliff_x=20–25`,
native distance pairs `(2,1) (4,2) (6,3)`, `bilateral_spawn_x_offset=1`.

## Reproduction command (canonical assays)

Post-run state-space assays (intact + deformation-off) on the stored specimen, driven by the
**archived config snapshot** (NOT the repo preset):

```bash
modal run --detach sensing_edges/scripts/modal_post_run.py \
  --run-dir   /outputs/rerun_local_runs/paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed137_e4fcc2fb \
  --specimen-id bin_03_rank_00_cell_6-0-3_flat_0423 \
  --config-path .../config_snapshot.yaml \
  --assays state_space_trace,state_space_trace_deformation_off
```

(Full launch argv per GPU profile is in
`paper_data/preprint/s4_cross_hardware_state_space_20260614/manifests/modal_launch_manifest.json`.)

## Machine-readable tag

```yaml
s4_reproducibility:
  specimen_id: bin_03_rank_00_cell_6-0-3_flat_0423
  seed: 137
  generation: 99
  genotype_sha256: dc13ba50718e6e27a21546e0a4a7f832acb3233fcbd57aafc11ceaa44409cf02
  controller_params_sha256: 7b55de00efca315036c27c560f1321b1e44d097c8c50916b2d01700b563e8576
  run_dir_name: paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed137_e4fcc2fb
  logical_config_sha256: e4fcc2fb
  resolved_config_sha256: 646ba6e6
  regime:
    hardware: a100-40gb
    jax: 0.9.2
    jaxlib: 0.9.2
    python: 3.11.12
    numpy: 2.4.4
    matmul_precision: default   # TF32 active
    reproduces_identically_on: [a100-80gb, l4]
    sensitivity_tail: [h100, t4] # safe modes only; never falls
  physics: {integrator: rk4, physics_substeps: 15, dt: 1.0e-4, phases: 3, nca_steps_per_phase: 5,
            spring_stiffness_scale: 0.8, max_steps: 4000}
  controller: {nca: sbt, ctrnn_hidden: 12, ctrnn_grid_hidden: 6, p_update: 1.0, noise: off,
               perception_dim: 30, params: 624}
  body: {shape: rect_7x2, contractile_cells: 14}
  terrain: {variant: bilateral_goal_cliff_asymmetric_pairs_7x2_grid30, native_pairs: [[2,1],[4,2],[6,3]]}
```

> **Why A100-40GB is canonical:** it is the archive regime *and* it reproduces the deformation-off
> native-core headline (12/12), confirmed identical on A100-80GB and L4. The cross-hardware spread
> (H100/T4) is a documented numeric-sensitivity caveat to state plainly in the README, not a
> defect — the body's *safety* is invariant; only deformation-off goal-completion is poised.
>
> **Pending:** the local JAX-0.9.2 run will confirm whether the CPU backend (any version)
> preserves the safety-robust / completion-sensitive split; until then, do not claim CPU
> reproduction of the exact GPU numbers.
