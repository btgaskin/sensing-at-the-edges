# Reproduction Guide

The training and evaluation protocol, with the commands used to produce the paper results.

> Note: reproducing the full paper results requires GPU compute (via Modal) and the
> run datasets that are not bundled in this repository (see the root README's Data
> section). This guide documents the protocol and entrypoints; some environment-specific
> values (e.g. wall-clock runtimes) are left unbenchmarked.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- [Modal](https://modal.com/) account with GPU access
- wandb API key (optional, for dashboard)

```bash
# Install
cd sensing_edges
uv sync

# Configure Modal secrets
modal secret create wandb-key WANDB_API_KEY=your_key_here
```

## Step 1: Preflight

```bash
uv run python scripts/paper_protocol.py preflight
```

Validates the active SBT run surfaces, schemas, and git tree is clean.

## Step 2: Generate Terrain Suites

Generate the frozen terrain schedule for each seed. All runs with the same seed
share the same frozen terrain sequences. The paper run uses a single bilateral
variant for training, rerank, and test.

```bash
for seed in 42 99 7 13 21; do
    uv run python scripts/generate_terrain_suite.py \
        --variant bilateral_goal_cliff_asymmetric_pairs_7x2_grid30 \
        --seed $seed --n-gens 150 --n-episodes 6 --n-val 6 --n-test 6
done
```

## Step 3: Smoke Test

```bash
modal run scripts/modal_smoke.py
```

Expected: ~5 minutes on L4 GPU. Verifies the full pipeline runs without errors.

## Step 4: Core Body Matrix

10 runs by default: 1 body (rect_7x2) x 5 seeds x 2 deformation conditions (on/off). Each run is launched as a detached Modal job.

```bash
uv run python scripts/paper_protocol.py core --seeds 42,99,7,13,21
```

Or manually:
```bash
for seed in 42 99 7 13 21; do
    modal run --detach scripts/modal_train.py \
        --run paper_run \
        --body rect_7x2 \
        --seed $seed \
        --schedule terrain_suites/seed_${seed}_bilateral_goal_cliff_asymmetric_pairs_7x2_grid30_g150_ep6_val6_test6
done
```

The schedule name must match the bundle Step 2 wrote, suffix included; `paper_run`
resolves it automatically if `--schedule` is omitted, and fails loudly if the
bundle is missing.

These commands perform training only. Preset `eval.rerank_variant` and `eval.test_variant`
values are preserved for later evaluation tooling, but inline post-train rerank/eval runs
only when both `--rerank-schedule` and `--eval-schedule` are supplied.

Monitor: `modal container list`

Retrieve outputs:
```bash
modal volume ls se-outputs /outputs/
modal volume get se-outputs /outputs/ ./local_outputs/ --force
```

## Step 5: Held-Out Evaluation

After training completes, evaluate each run's latest durable checkpoint and materialize the canonical publication release tree on the Modal volume:

```bash
uv run python scripts/paper_protocol.py eval \
    --outputs-dir local_outputs \
    --schedules-dir terrain_suites
```

Results are written to `se-outputs:/outputs/publication/paper_main/`.

## Step 6: Figures and Tables

```bash
uv run python scripts/paper_protocol.py figures
```

Outputs:
- `/outputs/publication/paper_main/figures/*.pdf` — body-wise fitness/coverage curves and deformation-ablation summary plots
- `/outputs/publication/paper_main/stats/deformation_ablation.json` — matched-seed ON/OFF ablation summaries

## Step 7: Release Bundle

```bash
uv run python scripts/paper_protocol.py bundle
```

Output: `/outputs/publication/paper_main/bundles/paper_main.tar.gz` with SHA256 checksum.

## Expected Runtimes

| Stage | GPU | Time per run | Total (10 runs) |
|-------|-----|-------------|-------|
| Smoke test | L4 | 5 min | 5 min |
| Pilot (50 gens) | L4 | ~30 min | ~30 min |
| Paper run (150 gens, stop after 100) | A100 40GB | not benchmarked | not benchmarked |
| Held-out eval | L4 | ~15 min | ~2.5 hours |
| Figures/tables | CPU | <1 min | <1 min |

## Output Structure

```
outputs/
  {preset}_{body_variant}_g{G}_p{P}_{substeps}sub[_nodeform]_seed{N}/
    manifest.json           # Run provenance
    resolved_config.yaml    # Full Hydra config
    metrics.jsonl           # Per-generation metrics
    checkpoints/
      gen_0049.json         # Checkpoint metadata
      gen_0049.npz          # Checkpoint arrays
      gen_0049.done         # Durable completion marker
    final_archive.json
    final_archive.npz
    final_archive.done

  publication/paper_main/
    release_manifest.json
    manifests/runs.jsonl
    summaries/{run_id}.json
    stats/deformation_ablation.json
    figures/*.pdf
    bundles/paper_main.tar.gz
    schedules/*             # Copied schedule bundles used by the release
```

## Determinism

- Terrain schedules are frozen and SHA256-hashed
- Same master seed → same RNG chain → same terrain sequence
- JAX compilation cache shared across runs (se-jax-cache volume)
- `pyproject.toml` pins `jax`/`jaxlib` to 0.9.0.1 and the Modal image installs the
  same pin, so local and GPU runs share one JAX version. The archived canonical
  specimen predates that pin: its manifest records JAX 0.9.2 on A100-40GB. Read
  the archived numbers against 0.9.2, not against the current pin
- `jax.matmul_precision` is a train-path knob only (`jax_cache.py`, called from
  `train.py`); eval and post-run always run at the JAX default, which is TF32 on
  Ampere and later. The archived specimen's numbers are TF32 numbers
- OU noise uses a fixed eval seed across all conditions; the paper run sets
  `noise_type: off`, so `ou_noise_seed` is inert there
- Cross-platform numerics may differ (GPU vs CPU); the canonical regime is Modal
  A100-40GB for training and L4 for smoke and held-out eval
