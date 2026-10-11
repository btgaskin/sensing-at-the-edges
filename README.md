# Sensing at the Edges

Research code and specimen records for the ALIFE 2026 paper
[*Sensing at the edges: synthetic specimens for early contractile
coordination*](https://doi.org/10.1162/isal.a.959).

The experiment asks how a multicellular body can coordinate adaptive behavior
before cells specialize into sensor or effector roles. A fixed 7 × 2 soft body
has fourteen identical contractile cells, each running the same continuous-time
recurrent neural network (CTRNN). Cells sense local deformation, exchange local
signals, and contract within the [jax-evogym](https://github.com/btgaskin/jax-evogym)
simulator. Evolution selects controller parameters in a bilateral goal-cliff
environment, where the body must reach a goal on one side while avoiding a cliff
on the other.

Eight selected specimens achieved mean native success of 89.6% (SD 12.4%).
Deformation and signaling ablations reveal distinct dependencies of edge
responses and locomotor coordination. One specimen, S4, also reverses at edges
under deformation ablation, with its ongoing controller drive redirected through
body mechanics. Performance approaches the 50% unilateral-commitment baseline
on longer, untrained distances; the cohort describes capabilities of selected
specimens rather than their frequency across evolutionary searches.

The repository organizes training, analysis, and specimen replay in three layers:

- **`sensing_edges/`** — training and evaluation. A CMA-MAE quality-diversity
  loop over a neural cellular automaton (NCA) with a shared CTRNN controller
  in a bilateral goal-cliff environment, with Hydra
  configuration (`config/`), the frozen terrain-schedule bundles behind the
  shipped specimens (`terrain_suites/`), post-run assay orchestration,
  and Modal entrypoints for
  GPU training. The paper preset is `config/preset/paper_run.yaml`.
- **`preprint/`** — analysis and figure scripts that turn run outputs into the
  manuscript's tables and figures.
- **`s4/`** — the specimen audit layer: a preset-driven probe runner
  (`probe.py`), audit deliverables (`audit/`), and curated specimen bundles
  (`results/`) — genome, controller parameters, resolved config, and
  representative renders for each featured robot.

## Paper

**Sensing at the edges: synthetic specimens for early contractile coordination**

Ben Gaskin. *ALIFE 2026: Proceedings of the 2026 Artificial Life Conference*,
article 33. MIT Press, published 17 August 2026.
DOI: [10.1162/isal.a.959](https://doi.org/10.1162/isal.a.959).

[Published paper](https://doi.org/10.1162/isal.a.959) ·
[Bundled camera-ready PDF](gaskin_alife.pdf) ·
[Illustrated research companion](https://www.bengaskin.com/sensing-edges/)

Use [CITATION.cff](CITATION.cff) or GitHub's **Cite this repository** menu to
cite the paper.

## Install

`sensing_edges/` and `s4/` are independent [uv](https://docs.astral.sh/uv/)
projects; `preprint/` is a script collection that runs against the
`sensing_edges/` environment. `jax-evogym` is pinned as a git dependency
(tag `v0.1.0`), so no separate checkout is needed:

```bash
git clone https://github.com/btgaskin/sensing-at-the-edges.git
cd sensing-at-the-edges
cd sensing_edges
uv sync
```

`s4/` additionally depends on `sensing_edges/` via a relative path, so keep the
repository layout intact:

```bash
cd ../s4
uv sync
```

## Reproducing runs

See [the reproduction guide](sensing_edges/REPRODUCTION.md) for the training
protocol and [the specimen guide](s4/results/README.md) for the stored replay
inputs. GPU training runs on [Modal](https://modal.com) and expects your own
Modal account:
the app is named `sensing-edges`, with volumes `se-jax-cache` and `se-outputs`,
and a Modal secret named `wandb-key` holding a `WANDB_API_KEY` for logging
(`modal secret create wandb-key WANDB_API_KEY=your_key_here`).

### Reproducibility caveat

The preset in `sensing_edges/config/preset/paper_run.yaml` has moved on from the
configuration that produced the archived specimens. The canonical specimen in
`s4/results/bin_03_rank_00_cell_6-0-3_flat_0423/` was trained with a 4000-step
episode horizon, 12 emitters (population 300), a `[7, 7, 10]` archive, a fall
margin of 1.5 cells, and a five-condition deformation suite; the shipped preset
uses 3000 steps, 8 emitters (population 200), `[7, 7, 20]`, a margin of 1.0, and
six conditions. The drift table in
[`s4/audit/AUDIT_REPORT.md`](s4/audit/AUDIT_REPORT.md) lists the differences
term by term.

The consequence is concrete: running `paper_run` as shipped is a related
experiment, not a rerun of the archived specimen. Each bundle under
`s4/results/` carries its own `config_snapshot.yaml`. That snapshot — not the
repository preset — is the authoritative record of the final run's
configuration.

Three specimens were produced by branch-mode runs resumed from generation-49
parent checkpoints. Those parent checkpoints and configurations are not
included, so the public bundles support specimen replay but not evolutionary
regeneration from scratch for those specimens. Every manifest records the
producing Git commit as `unknown`, so exact code provenance is not pinned.

## Data

What ships in this repository:

- two frozen terrain-schedule bundles used by the cohort
  (`sensing_edges/terrain_suites/`): the seed-137 `paper_run` bundle and the
  seed-42 `iter_run` bundle. The reproduction protocol regenerates training
  suites for other seeds with `scripts/generate_terrain_suite.py`
- the eight curated specimen bundles under `s4/results/` — each with provenance
  (preset, seed, run), the genome and evolved controller parameters, the
  resolved training config, and a few representative render GIFs

The complete run outputs behind the paper figures (assay trees, trace bundles,
and derived analyses, tens of gigabytes) are not hosted here. They are
available on request — open an issue on this repository.

## Related

- [jax-evogym](https://github.com/btgaskin/jax-evogym) — the JAX-native
  soft-body simulator this work is built on
- [Documentation site](https://jax-evogym.pages.dev/) — simulator docs
  and the browser-based world designer

## Acknowledgements

This work is built on [jax-evogym](https://github.com/btgaskin/jax-evogym), a
JAX-native reimplementation of [Evolution Gym](https://github.com/EvolutionGym/evogym)
(Bhatia, Jackson, Tian, Xu, and Matusik, NeurIPS 2021; MIT License, Copyright ©
2022 jagdeepsb). See [`NOTICE`](NOTICE) and the jax-evogym NOTICE for full
attribution.

## License

The code and data in this repository are licensed under Apache-2.0, matching
`jax-evogym`. See [LICENSE](LICENSE).

The paper [`gaskin_alife.pdf`](gaskin_alife.pdf) is © 2026 Ben Gaskin and is
licensed under [Creative Commons Attribution 4.0 International (CC BY
4.0)](https://creativecommons.org/licenses/by/4.0/).
