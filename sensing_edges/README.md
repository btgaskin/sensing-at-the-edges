# Sensing Edges

A publication-oriented SBT-grounded locomotion experiment for distributed soft-body control. Built on [jax-evogym](https://github.com/btgaskin/jax-evogym), a JAX reimplementation of the EvoGym soft-body simulator.

## Motivation

The **Skin Brain Thesis** (Keijzer, 2015) proposes that early nervous systems evolved to coordinate contractile tissue through local chemical signalling and mechanosensation, rather than through centralized input-output processing. In `sensing_edges`, each voxel is treated as an excitable contractile cell in a shared myoepithelium: it senses local deformation, communicates through grid-visible chemical channels, and updates an intracellular CTRNN state.

The paper run uses bilateral goal-cliff terrains with the SBT controller (`sbt`: `ctrnn_hidden=12`, `ctrnn_grid_hidden=6`, `ctrnn_tau_min=1.0`, `gasnet=off`, `mace=off`, `noise=off`, `p_update=1.0`, 624 params) on asymmetric distance-pair schedules. Body: fixed `rect_7x2` (14 CONTRACTILE cells, `spring_stiffness_scale=0.8`).

## The Task

### Bilateral Goal-Cliff

```
W=30, H=10

      cliff side          center plateau (12 cells)          goal side
  ┌──────────────────┬──────────────────────────────┬──────────────────┐
  │     (empty)      │     FIXED platform row       │   FIXED + GOAL   │
  │                  │     robot spawns here         │   (raised block) │
  └──────────────────┴──────────────────────────────┴──────────────────┘
```

- **Robot**: 7 wide x 2 high, all CONTRACTILE cells. Fixed morphology.
- **Platform**: a single FIXED row at `y=3`. A centered 12-cell start plateau extends outward by `goal_distance` on the goal side and `cliff_distance` on the cliff side. Spawn position: `(x=11, y=4)` with `bilateral_spawn_x_offset=1` (1 cell right of plateau center).
- **Goal block**: a single raised FIXED cell at `platform_y + 1` on the goal-side platform edge. Goal side alternates left/right across episodes.
- **Asymmetric distance pairs**: `(goal_distance, cliff_distance)` in {(2,1), (4,2), (6,3)}. The goal is always farther than the cliff, forcing the controller to choose the harder direction.
- **Probe gap variant**: a 1-cell gap + 3-cell lowered ledge on the cliff side. Provides a tactile warning cue before the cliff edge.
- **Death plane**: `y < 0.1 * CELL_SIZE`. No row-0 collider floor.
- **Mirrored-pair bonus**: `min(left_goal_hits, right_goal_hits)` across all bilateral episodes (+20 per mirrored pair, max 60).

### Termination Conditions

| Condition | Trigger | Fitness Effect |
|-----------|---------|----------------|
| Max steps | step_count = max_steps | Normal score |
| Goal touched | robot contacts goal block | Goal touch bonus (+40) |
| Cliff fall | robot falls off cliff side | Cliff fall penalty (-20) |
| Death plane | any robot point y < 0.1 | Task-dependent penalty |
| Airborne + death | rose > margin above start AND hit death plane | Score = 0 |
| Left edge | robot COM x < 0 | Score = -20 |

### Fitness

**Bilateral goal-cliff**: `outward_progress` (10, capped) + `reversal_progress` (20, gated) + `goal_direction_progress` (30) + `goal_touch` (40) - `cliff_fall_penalty` (20) - `ledge_penalty` (10). At the candidate level: plus the mirrored-pair bonus (20/pair, max 60 over 6 episodes) minus the soft-loft penalty (`train.soft_loft_penalty_scale = 8.0` in `paper_run`, applied to mean lofted excess cells). The reversal reward gates on at 50% wrong-way progress and reaches full weight at 70%.

**Cliff-walk**: `progress_shaping` (20) + `safe_arrival` (40) + `safe_dwell` (30) + `urgency` (10). Airborne = 0. Left edge = -20.

### Cliff-Walk Task

```
W=30, H=10

x=0..9   fixed spawn plateau
x=10..19 phase-1 randomized zone
x=10..24 phase-2 / final randomized zone
x=25..29 empty remainder
```

- **Robot**: 7 wide x 2 high, all CONTRACTILE cells.
- **Platform**: a single FIXED row at `y=3`. Solid through `x=9`, then randomized cliff + optional gaps.
- **Curriculum**: phase 1 randomizes cliff in `x=10..19` with no gap. Phase 2 expands to `x=10..24` with a single 2-cell gap in alternating bands.

## Controller Architecture

### NCA-CTRNN

A Neural Cellular Automaton with Continuous-Time Recurrent Neural Network integration. Each cell in the robot morphology grid maintains a local state vector and communicates only with its 3x3 neighbourhood via a configurable Sobel/Laplacian/identity perception bank.

**Channel layout** (signed deformation, noise off):

| Channels | Count | Role |
|----------|-------|------|
| 0-3 | 4 | Signed deformation deltas (abs_h, abs_v, comp_h, comp_v) |
| 4-5 | 2 | Action output (h_contraction, v_contraction) |
| 6+ | G | Grid-visible hidden channels (written by CTRNN, perceived via active communication filters) |

**Controller configurations:**

| Config | Hidden | Grid Hidden | D/C | Params | Use |
|--------|--------|-------------|-----|--------|-----|
| `sbt` | 12 | 6 | 2.0 | 624 | Paper run |

**CTRNN dynamics** (Dayan & Abbott rate-coded, exact exponential integration):

```
tau = tau_min + (tau_max - tau_min) * sigmoid(log_tau)
r = tanh(y)                          # firing rate
target = r @ W_rec + perception @ W_input + bias
alpha = exp(-dt / tau)               # exact decay factor
y_new = target + (y - target) * alpha
output = tanh(y_new) @ W_output
```

Unconditionally stable for any dt/tau ratio. The paper run uses deterministic updates (`p_update=1.0`) with OU noise disabled and zero hidden-state initialization (`hidden_init_scale=0.0`). Symmetry breaking arises from asymmetric deformation inputs, terrain geometry, and the CTRNN's recurrent dynamics.

**Perception**: Split architecture. Private channels (deformation, action) get Identity only. Communication channels (grid_hidden) get a configurable filter bank drawn from `sobel_x`, `sobel_y`, `laplacian`, and `identity`, concatenated in canonical order. Output: `n_private + n_comm * len(active_comm_filters)` perception dimensions. The current `paper_run` and `iter_run` presets explicitly activate all four filters.

**Action mapping**: contraction in [-1, +1] mapped to actuator goal multiplier via `clip(1.0 + 0.5 * contraction, 0.5, 1.5)`.

### MaCE (Off)

Mass-Action Coupling Exchange (Papadopoulos & Guichard, 2025) is disabled in the paper run. MaCE models physical transport of conserved quantities, which is not appropriate for the current cell-to-cell signaling model where cells read neighbors' secreted state without consuming it.

### GasNet (Off by Default)

Diffusive gaseous signalling following Husbands et al. (1998). Two roles:
- **Signal mode**: gas channels are perceived via the active communication filter bank but don't modulate CTRNN gain.
- **Gain mode**: gas concentration modulates the CTRNN transfer function: `target = gain * target` where `gain = clip(1 + sensitivity * concentration, -4, 4)`.

Single (1 species, 2 channels) or dual (excitatory + inhibitory, 4 channels). Concentration evolves via: `c_new = c_old * decay + relu(emission)`, with Gaussian blur diffusion.

### Deformation Channels

The SBT presets keep the 4 signed deformation channels enabled by default. Ablations remain available as runtime overrides:

1. **ON**: 4 signed deformation channels computed from physics each sense phase. Log-scale gain applied (absolute=10.0, compliance=85.0).
2. **OFF**: all 4 channels zeroed via `nca.ablate_deformation=true`. Same architecture, same parameter count. Zero is the neutral value (signed delta encoding: ratio - 1.0 = 0 when no deformation).

### Phased Environment Step

True sense-act-physics interleaving (not same-snapshot NCA updates):

```
for env_step in range(max_steps):           # outer scan
    for phase in range(3):                   # 3 sense-act cycles per step
        deformation = sense(sim_state)       # fresh each phase
        for micro in range(nca_steps):       # NCA updates on this snapshot
            nca_state, actions = nca_step(nca_state, deformation, ...)
        apply mean(actions over microsteps) → actuator goals
        run physics substeps (5 per phase at 15 total)
        check termination, update tracking
    step_count += 1
```

CTRNN dt is split across phases: `dt_per_update = ctrnn_dt / (phases * nca_steps_per_phase)`.

## Quality-Diversity: CMA-MAE

Evolution uses **CMA-MAE** (Covariance Matrix Adaptation MAP-Annealing for MAP-Elites), following Fontaine & Nikolaidis.

### Algorithm

1. **Sep-CMA-ES emitters** each propose a batch of candidates. The shipped `paper_run` preset uses 8 emitters x 25 = 200 candidates per generation (`qd.n_emitters`, `qd.emitter_pop_size` in `config/preset/paper_run.yaml`). The archived canonical specimen was produced under an earlier form of this preset with 12 emitters x 25 = 300; see the reproducibility caveat in the root README.
2. All candidates are evaluated on 6 randomised terrains. Fitness and 3 behavioural descriptors are computed per candidate.
3. The **GridArchive** (7x7x20, 3D) computes improvement scores: `improvement = fitness - cell_threshold`. Candidates exceeding their cell's threshold are accepted. Thresholds update via exponential moving average: `threshold = (1 - lr) * old + lr * winner_fitness`, with `lr = 0.1`. A **focus window** progressively narrows the active archive region, targeting `goal_hit_count` (max) and `fall_count` (min), tightening every 20 generations under `paper_run` (`qd.focus.tighten_every_n_gens`; the config-group default is 10).
4. Each emitter receives its slice of improvement scores as the CMA-ES fitness signal.
5. Emitters that stall (flat improvements for 3 consecutive generations, or sigma collapse below 1e-12) are restarted, centred on a randomly sampled archive elite.

### Strategy: Separable CMA-ES

Sep-CMA-ES maintains only the diagonal of the covariance matrix, reducing storage from O(n^2) to O(n). Even for the smaller `sbt` controller (624 params), separable CMA-ES is preferred for efficiency. Implemented via evosax v0.2.0.

### QD Descriptors

**Bilateral** (7x7x20 archive, ranges [[0,6],[0,6],[0.2,0.8]]):

| Descriptor | Range | Meaning |
|-----------|-------|---------|
| `goal_hit_count` | [0, 6] | Number of goal touches across episodes |
| `fall_count` | [0, 6] | Number of cliff falls across episodes |
| `completion_time` | [0.2, 0.8] | Mean time to goal touch (fraction of budget; 1.0 if not reached) |

## Experiment Design

### Body Variants

| Body | Mode | Shape | Cells | Use |
|------|------|-------|-------|-----|
| `rect_7x2` | Fixed | rectangle 7x2 | 14 | Paper run (bilateral goal-cliff) |
| `rect_6x2` | Fixed | rectangle 6x2 | 12 | Smaller body variant |
| `rect_7x3` | Fixed | rectangle 7x3 | 21 | Full-support body |
| `comb_7x3` | Fixed | comb 7x3 | varies | Sparse-support body |
| `grown:7x3` | Grown | 7x3 canvas | evolved | Body-controller coevolution |

### Physics

- **ACTUATOR_CONVERGENCE = 0.003** is the 30-substep reference setting. `derive_physics_params()` rescales the per-substep convergence for 12- and 15-substep presets so all presets preserve the same effective per-env-step actuation (~8.6%).
- **spring_stiffness_scale = 0.8**: CONTRACTILE cells are 20% softer than standard actuators.
- **GROUND_VEL_DAMPING = 5238.0**: JAX-specific ground-normal velocity damping (not in C++ EvoGym). Tuned for ~0.05 coefficient of restitution.
- **RK4 integration** with dt = 0.0001.

### Evaluation Protocol

- **Training**: 4--6 episodes per candidate per generation, randomised from frozen terrain schedule.
- **Model selection**: top-5--10 elites from archive re-evaluated on frozen validation terrains; best selected.
- **Held-out reporting**: selected elite evaluated on frozen test terrains from the latest durable checkpoint. Bootstrap 95% CI on median fitness.
- **Statistics**: matched-seed ON vs OFF comparison on canonical publication summaries. The release stats store paired rows, paired effect sizes, bootstrap CI, and Wilcoxon p-values.

### Post-Training Analysis

The post-training pipeline covers specimen selection, the 14-family interpretability battery (native core, initial direction, long-distance generalization, double cliff, signaling ablation, deformation ablation, frozen sensing control, body transfer, state-space traces, combined ablation, tau, PCA bundle), publication figures, and statistical analysis. It is orchestrated by `sensing_edges/post_run.py` and driven through `scripts/modal_post_run.py`.

### Terrain Schedules

Frozen train/val/test terrain pools generated from a master seed with 3 disjoint RNG streams. SHA256-hashed for integrity verification. Stored as npz (grids) + JSON (metadata). All candidates in a generation see the same terrain set.

## Project Structure

Key files and directories; the listing is curated, not exhaustive.

```
sensing_edges/
  pyproject.toml                        # standalone uv project
  Makefile                              # preflight, smoke, test, figures, bundle
  README.md                             # this file
  REPRODUCTION.md                       # exact reproduction commands
  friction_audit.md                     # floor friction / stiction model notes

  sensing_edges/                        # --- Core pipeline ---
    terrain.py                          #   terrain generator (cliff + gaps + bilateral)
    terrain_schedule.py                 #   frozen train/val/test pools with SHA256
    nca.py                              #   NCA-CTRNN controller
    env.py                              #   phased episode runner
    fitness.py                          #   fitness + QD descriptors
    qd_descriptors.py                   #   descriptor registry
    rollout.py                          #   shared evaluation helper
    batch_eval.py                       #   chunked batched evaluation
    train.py                            #   CMA-MAE training loop
    evaluate.py                         #   held-out evaluation with bootstrap CI
    render.py                           #   on-demand GIF rendering
    checkpoint.py                       #   npz + JSON checkpoints
    manifest.py                         #   run provenance (git, JAX, hardware)
    wandb_metrics.py                    #   dashboard metrics + archive heatmaps
    jax_cache.py                        #   JAX compilation cache
    runtime_config.py                   #   runtime configuration helpers
    collision_tuning.py                 #   collision capacity configuration
    deformation_probe.py                #   signed deformation helpers
    training_reporting.py               #   training metrics formatting
    logging_utils.py                    #   logging configuration
    experiment.py                       #   Hydra entry point
    post_run.py                         #   post-run assay battery orchestration
    elite_import.py                     #   archive elite -> specimen bundle export
    publication.py                      #   publication sets and release trees
    pca_analogue.py                     #   PCA-analogue state-space bundles
    qd/                                 #   --- Quality-Diversity ---
      grid_archive.py                   #     CMA-MAE grid archive
      cma_emitter.py                    #     Sep-CMA-ES emitters with restart
      evolution_engine.py               #     evosax v0.2.0 strategy wrapper
      focus.py                          #     generation-ramped focus window
      gait_descriptor.py                #     gait spectral-purity descriptor

  config/                               # --- Hydra configs ---
    config.yaml                         #   root defaults
    nca/{default,sbt,sbt_6h_5grid,sbt_32h_8grid}.yaml  # controller configs
    sim/{full,half,twelve,twenty}.yaml  #   substep configs (30/15/12/20)
    terrain/*.yaml                      #   24 terrain variants
    preset/{paper_run,iter_run}.yaml    #   the two run presets behind the shipped specimens
    gasnet/{off,signal,single}.yaml     #   GasNet modes
    gpu_profile/*.yaml                  #   GPU-specific batch sizing
    {body,train,eval,qd,collision}/     #   component configs
    {publication,specimens,audit}/      #   release set, specimen export, audit knobs
    {jax,logging,wandb,batching,render}/

  scripts/                              # --- Tooling ---
    _modal_common.py                    #   shared Modal image/volumes
    modal_train.py                      #   GPU training
    modal_smoke.py                      #   GPU smoke test
    modal_eval.py                       #   GPU held-out evaluation
    modal_post_run.py                   #   GPU post-run assay battery
    modal_release.py                    #   Modal-side release materialization / figures / bundle
    generate_terrain_suite.py           #   emit frozen terrain schedules
    paper_protocol.py                   #   orchestrator (preflight→core→eval→figures→bundle)
    make_figures.py                     #   figures from canonical release artifacts
    make_tables.py                      #   paired ON/OFF stats from canonical release artifacts
    validate_manifest.py                #   JSON Schema validation
    validate_checkpoint.py              #   checkpoint integrity check
    export_release_bundle.py            #   tar.gz with SHA256 from a release tree
    import_elite_specimens.py           #   export archive elites as specimen bundles
    make_pca_analogue.py                #   PCA-analogue bundles from trace outputs
    compare_pca_bundle_stats.py         #   PCA bundle comparison stats

  schemas/                              #   JSON Schema for manifest, checkpoint, release, summaries
  terrain_suites/                       #   frozen terrain schedule bundles (npz + JSON)
  tests/                                #   555 test functions in 30 files (all local, no GPU)
  outputs/                              #   raw run artifacts (gitignored)
    publication/                        #   canonical paper artifacts on Modal volume
```

## Infrastructure

- **Modal**: GPU deployment. L4 for smoke and held-out eval; training entrypoints expose L4, A100-40GB, A100-80GB, H100, H200, and B200 targets. The `paper_run` preset selects the `h100` profile by default; the archived canonical specimen was trained on A100-40GB. Preemption retry with `initial_delay=0`, `max_retries=3`. Two persistent volumes: `se-jax-cache` (compilation cache) and `se-outputs` (run artifacts).
- **wandb**: 7 dashboard sections -- `fitness/` (hero), `descriptors/` (population means), `qd/`, `trials/` (GIF renders), `archive/` (heatmaps), `emitters/`, `timing/`. Run names are body/budget first; groups drop the seed for visual alignment.
- **Checkpoints**: durable JSON + NPZ payload bundles with `.done` completion markers. Publication tooling only indexes the latest durable checkpoint for each included run.
- **Manifests**: per-run JSON capturing git commit, JAX/Python versions, hardware backend, resolved config hash.

## Usage

```bash
cd sensing_edges
uv sync

# Local tests (no GPU)
uv run --with pytest python -m pytest tests/ -v

# Generate frozen terrain schedules (bilateral)
uv run python scripts/generate_terrain_suite.py \
    --variant bilateral_goal_cliff_asymmetric_pairs_7x2_grid30 \
    --seed 137 --n-gens 150 --n-episodes 6 --n-val 6 --n-test 6

# GPU smoke test
modal run scripts/modal_smoke.py

# Bilateral training run
modal run scripts/modal_train.py \
    --run paper_run \
    --body rect_7x2 \
    --seed 137 \
    --schedule terrain_suites/seed_137_bilateral_goal_cliff_asymmetric_pairs_7x2_grid30_g150_ep6_val6_test6

# `grid30` here refers to the terrain grid width, not physics substeps.

# Held-out evaluation
uv run python scripts/paper_protocol.py eval \
    --outputs-dir local_outputs \
    --schedules-dir terrain_suites

# Paper figures and statistics (runs make_figures/make_tables on the Modal
# release tree; pass --release-dir to invoke either script directly)
uv run python scripts/paper_protocol.py figures
```

See [REPRODUCTION.md](REPRODUCTION.md) for the complete reproduction protocol.

## Related Documents

- [`REPRODUCTION.md`](REPRODUCTION.md) -- Step-by-step reproduction commands
- [`friction_audit.md`](friction_audit.md) -- Friction-model methodology notes
- [`../preprint/README.md`](../preprint/README.md) -- Preprint analysis and figure scripts
- [`../s4/README.md`](../s4/README.md) -- Specimen audit and probe tooling

## References

- Keijzer, F. (2015). Moving and sensing without input and output: early nervous systems and the origins of the animal sensorimotor organization. *Biology & Philosophy*, 30(3), 311-331.
- Husbands, P., Smith, T., Jakobi, N., & O'Shea, M. (1998). Better living through chemistry: Evolving GasNets for robot control. *Connection Science*, 10(3-4), 185-210.
- Dayan, P., & Abbott, L. F. (2001). *Theoretical Neuroscience*. MIT Press.
- Fontaine, M. C., & Nikolaidis, S. (2023). Covariance Matrix Adaptation MAP-Annealing. *GECCO 2023*.
- Bhatia, J., et al. (2021). Evolution Gym: A large-scale benchmark for evolving soft robots. *NeurIPS 2021*.
