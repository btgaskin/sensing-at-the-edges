"""Tests for shared training config helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import logging
import numpy as np
import pytest
from omegaconf import OmegaConf

import sensing_edges.manifest as manifest_module
import sensing_edges.train as train_module
from sensing_edges.batch_eval import GrownEvalAssets, PopulationBatchSummary
from sensing_edges.checkpoint import payload_done_path
from sensing_edges.env import build_static_collider_template, infer_static_template_capacity
from sensing_edges.render import RenderArtifact
from sensing_edges.qd.cma_emitter import CMAEmitter
from sensing_edges.qd.grid_archive import GridArchive, GridArchiveBatchDecision
from sensing_edges.runtime_config import (
    compose_experiment_config,
    emitter_kwargs_from_cfg,
    rollout_kwargs_from_cfg,
    resolve_runtime_config,
    total_population,
)
from sensing_edges.terrain_schedule import generate_schedule, save_schedule
from sensing_edges.train import _prepare_training_inputs
from sensing_edges.terrain import build_cliff_terrain
from jax_evogym.types import default_physics_constants


def _compose(config_name: str = "config"):
    return compose_experiment_config(preset=None if config_name == "config" else config_name)


class DummyAuditEnv(NamedTuple):
    constants: object
    payload: int = 0


class TestRuntimeConfigHelpers:
    def test_qd_config_is_threaded_to_emitters(self):
        cfg = _compose("config")
        kwargs = emitter_kwargs_from_cfg(cfg)
        emitter = CMAEmitter(genotype_length=4, **kwargs)
        emitter.init(jax.random.PRNGKey(0))

        assert kwargs["stale_generations"] == cfg.qd.emitter_restart_stale_generations
        assert kwargs["max_age"] == cfg.qd.emitter_restart_max_age
        assert kwargs["meaningful_improvement_eps"] == pytest.approx(
            cfg.qd.emitter_restart_improvement_eps
        )
        assert kwargs["std_epsilon"] == pytest.approx(cfg.qd.emitter_std_epsilon)
        assert kwargs["condition_cov_limit"] == pytest.approx(
            cfg.qd.emitter_condition_cov_limit
        )
        assert kwargs["param_overrides"] == {"std_init": cfg.qd.initial_sigma}
        assert float(emitter.es_state.std) == pytest.approx(cfg.qd.initial_sigma)

    def test_qd_config_can_select_snes_emitters(self):
        cfg = compose_experiment_config(
            overrides=[
                "qd.emitter_strategy=snes",
                "qd.emitter_pop_size=10",
            ],
        )
        kwargs = emitter_kwargs_from_cfg(cfg)
        emitter = CMAEmitter(genotype_length=4, **kwargs)
        emitter.init(jax.random.PRNGKey(0))

        assert kwargs["strategy"] == "snes"
        key = jax.random.PRNGKey(1)
        population = emitter.ask(key)
        assert population.shape == (10, 4)
        assert emitter.es_state is not None

    def test_total_population_is_explicit_qd_population(self):
        cfg = _compose("config")
        assert total_population(cfg) == cfg.qd.n_emitters * cfg.qd.emitter_pop_size

    def test_rollout_kwargs_use_canonical_train_max_steps_and_static_shapes(self):
        cfg = _compose("config")
        kwargs = rollout_kwargs_from_cfg(cfg)
        assert kwargs["max_steps"] == 500
        assert kwargs["grid_h"] == cfg.terrain.grid_h
        assert kwargs["grid_w"] == cfg.terrain.grid_w
        assert kwargs["robot_h"] == cfg.terrain.robot_h
        assert kwargs["robot_w"] == cfg.terrain.robot_w
        assert kwargs["collision_strategy"] == "dynamic_broadphase"
        assert kwargs["dynamic_impl"] == "sparse_candidate_list"
        assert kwargs["dynamic_max_candidates"] == 4096
        assert kwargs["deformation_gain_absolute"] == pytest.approx(10.0 / 0.8)
        assert kwargs["deformation_gain_compliance"] == pytest.approx(85.0 / 0.8)

    def test_rollout_kwargs_can_disable_deformation_gain_autoscale(self):
        cfg = compose_experiment_config(
            overrides=[
                "nca.deformation_gain_absolute=12.0",
                "nca.deformation_gain_compliance=34.0",
                "nca.deformation_gain_auto_scale=false",
            ],
        )
        kwargs = rollout_kwargs_from_cfg(cfg)

        assert kwargs["deformation_gain_absolute"] == pytest.approx(12.0)
        assert kwargs["deformation_gain_compliance"] == pytest.approx(34.0)

    def test_train_config_no_longer_exposes_dead_population_knobs(self):
        cfg = _compose("config")
        assert "population_size" not in cfg.train
        assert "chunk_size" not in cfg.train

    def test_runtime_resolution_exposes_only_active_batching_workloads(self):
        cfg = _compose("config")
        resolution = resolve_runtime_config(cfg)
        assert set(resolution.resolved_batching) == {"generation", "validation", "test", "trace"}
        assert resolution.resolved_batching["generation"].exact_fit_small_run_threshold == 32

    def test_prepare_training_inputs_grown_returns_self_contained_eval_assets(self):
        cfg = compose_experiment_config(
            preset="iter_run",
            overrides=[
                "body.mode=grown",
                "terrain=cliff_only",
                "train.n_generations=1",
                "train.n_episodes=1",
                "train.require_schedule_dir=false",
            ],
        )
        inputs = _prepare_training_inputs(cfg, np.random.default_rng(0))
        envs = inputs.envs_for_generation(0)

        assert isinstance(envs, GrownEvalAssets)
        assert envs.grid_data is not None
        assert envs.runtime_build_capacity == inputs.collision_tuning.runtime_build_capacity
        assert int(envs.static_templates.cliff_x.shape[0]) == 1
        assert len(envs.templates) == 1
        assert envs.templates[0].render_info is not None

    def test_append_elite_specimens_saves_both_historical_zero_fall_tiers(self, tmp_path):
        cfg = compose_experiment_config(preset="paper_run")
        run_manifest_path = tmp_path / "manifest.json"
        run_manifest_path.write_text(
            json.dumps(
                {
                    "run_id": "run-test",
                    "config": {},
                    "runtime": {},
                    "hardware": {},
                    "system": {},
                    "git": {},
                    "wandb": {},
                    "lineage": {},
                }
            )
        )
        run_resolved_config_path = tmp_path / "resolved_config.yaml"
        run_resolved_config_path.write_text(OmegaConf.to_yaml(cfg))
        genotype_dim = train_module.genotype_dim_from_cfg(cfg)
        genotype_a = np.arange(genotype_dim, dtype=np.float32)
        genotype_b = np.arange(genotype_dim, dtype=np.float32) + 1000.0
        genotype_c = np.arange(genotype_dim, dtype=np.float32) + 2000.0
        candidate_batch = jnp.asarray(
            [
                genotype_a,
                genotype_b,
                genotype_c,
            ],
            dtype=jnp.float32,
        )
        batch_result = PopulationBatchSummary(
            fitness=np.asarray([10.0, 20.0, 30.0], dtype=np.float32),
            goal_hit_count=np.asarray([6.0, 6.0, 5.0], dtype=np.float32),
            mirrored_pair_count=np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
            episode_fell=np.asarray(
                [
                    [False, False, False, False, False, False],
                    [False, True, False, False, False, False],
                    [False, False, False, False, False, False],
                ],
                dtype=bool,
            ),
            episode_left_edge=np.asarray(
                [
                    [False, False, False, False, False, False],
                    [False, False, False, False, False, False],
                    [False, False, False, False, False, False],
                ],
                dtype=bool,
            ),
            deformation_channel_max=np.zeros((3, 4), dtype=np.float32),
            controller_hidden_output_max_abs=np.zeros((3,), dtype=np.float32),
            action_output_max_abs=np.zeros((3,), dtype=np.float32),
            descriptor_columns={"fall_count": np.asarray([0.0, 1.0, 0.0], dtype=np.float32)},
        )

        archive = GridArchive(
            dims=(2, 8, 10),
            ranges=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            descriptor_names=("fall_count", "goal_hit_count", "stable_turn_count"),
        )
        accepted_flat_index = archive._ravel((1, 6, 3))
        archive_decision = GridArchiveBatchDecision(
            improvements=np.zeros((3,), dtype=np.float64),
            statuses=np.asarray(
                [GridArchive.NEW, GridArchive.IMPROVE, GridArchive.NOT_ADDED],
                dtype=np.uint8,
            ),
            accepted_indices=np.asarray([0], dtype=np.int64),
            accepted_flat_indices=np.asarray([accepted_flat_index], dtype=np.int64),
            accepted_fitnesses=np.asarray([10.0], dtype=np.float64),
            accepted_statuses=np.asarray([GridArchive.NEW], dtype=np.uint8),
            accepted_thresholds=np.asarray([10.0], dtype=np.float64),
            accepted_pre_fitnesses=np.asarray([-np.inf], dtype=np.float64),
            descriptor_finite_mask=np.asarray([True, True, True], dtype=bool),
            invalid_descriptor_count=0,
        )

        saved = train_module._append_elite_specimens(
            output_dir=tmp_path,
            cfg=cfg,
            run_manifest_path=run_manifest_path,
            run_resolved_config_path=run_resolved_config_path,
            generation=7,
            candidate_batch=candidate_batch,
            batch_result=batch_result,
            fitnesses=np.asarray([10.0, 20.0, 30.0], dtype=np.float32),
            descriptors=np.asarray(
                [
                    [0.0, 6.0, 1.0],
                    [1.0, 6.0, 2.0],
                    [0.0, 5.0, 0.0],
                ],
                dtype=np.float32,
            ),
            n_episodes=6,
            descriptor_names=("fall_count", "goal_hit_count", "stable_turn_count"),
            statuses=np.asarray([GridArchive.NEW, GridArchive.IMPROVE, GridArchive.NOT_ADDED], dtype=np.uint8),
            archive=archive,
            archive_decision=archive_decision,
        )

        assert [path.name for path in saved] == [
            "specimen_gen_0007_candidate_0000_cell_1-6-3_hits6_falls0",
            "specimen_gen_0007_candidate_0002_hits5_falls0",
        ]
        perfect_specimen_dir = saved[0]
        assert (perfect_specimen_dir / "manifest.json").exists()
        assert (perfect_specimen_dir / "selection_summary.json").exists()
        assert (perfect_specimen_dir / "genotype.npz").exists()
        assert (perfect_specimen_dir / "controller_params.npz").exists()
        assert (perfect_specimen_dir / "config_snapshot.yaml").exists()
        assert (perfect_specimen_dir / "historical_snapshot.npz").exists()

        perfect_specimen = np.load(perfect_specimen_dir / "historical_snapshot.npz", allow_pickle=False)
        np.testing.assert_allclose(perfect_specimen["genotype"], genotype_a)
        np.testing.assert_allclose(perfect_specimen["descriptor"], np.asarray([0.0, 6.0, 1.0], dtype=np.float32))
        assert float(perfect_specimen["goal_hit_count"]) == pytest.approx(6.0)
        assert float(perfect_specimen["fall_count"]) == pytest.approx(0.0)
        assert str(perfect_specimen["historical_elite_tier"]) == "perfect_zero_fall"
        assert str(perfect_specimen["genotype_sha256"]) == train_module._genotype_sha256(
            genotype_a
        )
        assert int(perfect_specimen["archive_status"]) == GridArchive.NEW
        assert int(perfect_specimen["archive_flat_index"]) == accepted_flat_index
        np.testing.assert_array_equal(
            perfect_specimen["archive_cell_indices"],
            np.asarray([1, 6, 3], dtype=np.int32),
        )
        assert str(perfect_specimen["archive_cell_id"]) == "cell_1-6-3"
        assert str(perfect_specimen["archive_cell_label"]) == (
            "fall_count=1|goal_hit_count=6|stable_turn_count=3"
        )
        perfect_summary = json.loads((perfect_specimen_dir / "selection_summary.json").read_text())
        assert perfect_summary["selection_summary"]["specimen_tier"] == "perfect"

        near_perfect_specimen = np.load(saved[1] / "historical_snapshot.npz", allow_pickle=False)
        np.testing.assert_allclose(near_perfect_specimen["genotype"], genotype_c)
        np.testing.assert_allclose(near_perfect_specimen["descriptor"], np.asarray([0.0, 5.0, 0.0], dtype=np.float32))
        assert float(near_perfect_specimen["goal_hit_count"]) == pytest.approx(5.0)
        assert float(near_perfect_specimen["fall_count"]) == pytest.approx(0.0)
        assert str(near_perfect_specimen["historical_elite_tier"]) == "near_perfect_zero_fall"
        assert int(near_perfect_specimen["archive_flat_index"]) == -1

    def test_load_historical_elite_candidates_dedupes_genotypes_and_upgrades_tier(self, tmp_path):
        specimen_dir = tmp_path / "elite_specimens"
        specimen_dir.mkdir(parents=True)
        genotype = np.asarray([1.0, 2.0], dtype=np.float32)

        np.savez_compressed(
            specimen_dir / "gen_0002_candidate_0001_hits5_falls0.npz",
            genotype=genotype,
            fitness=np.asarray(50.0, dtype=np.float32),
            descriptor=np.asarray([0.0, 5.0, 0.2], dtype=np.float32),
            goal_hit_count=np.asarray(5.0, dtype=np.float32),
            fall_count=np.asarray(0.0, dtype=np.float32),
            generation=np.asarray(2, dtype=np.int32),
            candidate_index=np.asarray(1, dtype=np.int32),
            archive_flat_index=np.asarray(-1, dtype=np.int32),
            archive_cell_indices=np.asarray([-1, -1, -1], dtype=np.int32),
            archive_cell_id=np.asarray("", dtype=str),
            archive_cell_label=np.asarray("", dtype=str),
        )
        np.savez_compressed(
            specimen_dir / "gen_0005_candidate_0003_hits6_falls0.npz",
            genotype=genotype,
            genotype_sha256=np.asarray(train_module._genotype_sha256(genotype), dtype=str),
            fitness=np.asarray(75.0, dtype=np.float32),
            descriptor=np.asarray([0.0, 6.0, 0.4], dtype=np.float32),
            goal_hit_count=np.asarray(6.0, dtype=np.float32),
            fall_count=np.asarray(0.0, dtype=np.float32),
            historical_elite_tier=np.asarray("perfect_zero_fall", dtype=str),
            generation=np.asarray(5, dtype=np.int32),
            candidate_index=np.asarray(3, dtype=np.int32),
            archive_flat_index=np.asarray(12, dtype=np.int32),
            archive_cell_indices=np.asarray([1, 5, 2], dtype=np.int32),
            archive_cell_id=np.asarray("cell_1-5-2", dtype=str),
            archive_cell_label=np.asarray("goal_hit_count=1|fall_count=5|completion_time=2", dtype=str),
        )

        archive = GridArchive(
            dims=(7, 7, 10),
            ranges=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            descriptor_names=("goal_hit_count", "fall_count", "completion_time"),
        )
        candidates, file_count = train_module._load_historical_elite_candidates(
            output_dir=tmp_path,
            cfg=compose_experiment_config(preset="paper_run"),
            archive=archive,
            n_episodes=6,
        )

        assert file_count == 2
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate["source"] == "historical_elite"
        assert candidate["historical_elite_tier"] == "perfect_zero_fall"
        assert candidate["historical_first_generation"] == 2
        assert candidate["historical_last_generation"] == 5
        assert candidate["historical_snapshot_count"] == 2
        assert candidate["archive_flat_index"] == 12
        np.testing.assert_array_equal(
            candidate["archive_descriptor"],
            np.asarray([0.0, 6.0, 0.4], dtype=np.float32),
        )

    def test_load_historical_elite_candidates_reads_canonical_specimen_style_layout(self, tmp_path):
        cfg = compose_experiment_config(preset="paper_run")
        historical_root = (
            train_module.resolve_specimen_root(
                tmp_path,
                output_root=str(cfg.specimens.get("output_root", "specimens")),
            )
            / "historical_elites"
        )
        specimen_dir = historical_root / "specimen_gen_0004_candidate_0001_hits6_falls0"
        specimen_dir.mkdir(parents=True)
        genotype = np.asarray([1.0, 2.0], dtype=np.float32)
        np.savez_compressed(
            specimen_dir / "historical_snapshot.npz",
            genotype=genotype,
            genotype_sha256=np.asarray(train_module._genotype_sha256(genotype), dtype=str),
            fitness=np.asarray(91.0, dtype=np.float32),
            descriptor=np.asarray([0.0, 6.0, 0.4], dtype=np.float32),
            goal_hit_count=np.asarray(6.0, dtype=np.float32),
            fall_count=np.asarray(0.0, dtype=np.float32),
            historical_elite_tier=np.asarray("perfect_zero_fall", dtype=str),
            generation=np.asarray(4, dtype=np.int32),
            candidate_index=np.asarray(1, dtype=np.int32),
            archive_flat_index=np.asarray(12, dtype=np.int32),
            archive_cell_indices=np.asarray([1, 5, 2], dtype=np.int32),
            archive_cell_id=np.asarray("cell_1-5-2", dtype=str),
            archive_cell_label=np.asarray("goal_hit_count=1|fall_count=5|completion_time=2", dtype=str),
        )

        archive = GridArchive(
            dims=(7, 7, 10),
            ranges=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            descriptor_names=("goal_hit_count", "fall_count", "completion_time"),
        )
        candidates, file_count = train_module._load_historical_elite_candidates(
            output_dir=tmp_path,
            cfg=cfg,
            archive=archive,
            n_episodes=6,
        )

        assert file_count == 1
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate["source"] == "historical_elite"
        assert candidate["historical_elite_tier"] == "perfect_zero_fall"
        assert candidate["historical_first_generation"] == 4
        assert candidate["historical_last_generation"] == 4
        assert candidate["archive_flat_index"] == 12

    def test_dedupe_pool_candidates_merges_historical_and_archive_duplicates(self):
        genotype = np.asarray([1.0, 2.0], dtype=np.float32)
        historical_candidate = train_module._base_pool_candidate(
            source="historical_elite",
            source_index=5,
            genotype=genotype,
            archive_flat_index=-1,
            historical_elite_tier="perfect_zero_fall",
            historical_first_generation=4,
            historical_last_generation=9,
            historical_snapshot_count=2,
        )
        archive_candidate = train_module._base_pool_candidate(
            source="archive",
            source_index=0,
            genotype=genotype,
            archive_flat_index=123,
            archive_cell_indices=(6, 0, 4),
            archive_cell_id="cell_6-0-4",
            archive_cell_label="goal_hit_count=6|fall_count=0|completion_time=4",
            archive_fitness=99.0,
            archive_descriptor=np.asarray([6.0, 0.0, 0.4], dtype=np.float32),
        )

        deduped = train_module._dedupe_pool_candidates([archive_candidate, historical_candidate])

        assert len(deduped) == 1
        candidate = deduped[0]
        assert candidate["source"] == "historical_elite"
        assert candidate["candidate_sources"] == ["archive", "historical_elite"]
        assert candidate["historical_elite_tier"] == "perfect_zero_fall"
        assert candidate["archive_flat_index"] == 123
        assert candidate["archive_cell_id"] == "cell_6-0-4"
        assert candidate["historical_snapshot_count"] == 2

    def test_specimen_tier_helpers_map_to_perfect_and_imperfect(self):
        assert train_module._historical_specimen_tier("perfect_zero_fall") == "perfect"
        assert train_module._historical_specimen_tier("near_perfect_zero_fall") == "imperfect"
        assert train_module._historical_specimen_tier(None) is None
        assert train_module._held_out_specimen_tier(perfect_hits=True, zero_falls=True) == "perfect"
        assert train_module._held_out_specimen_tier(perfect_hits=False, zero_falls=True) == "imperfect"
        assert train_module._held_out_specimen_tier(perfect_hits=True, zero_falls=False) == "imperfect"

    def test_archive_pool_candidates_include_explicit_archive_cell_metadata(self):
        archive = GridArchive(
            dims=(7, 7, 10),
            ranges=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            descriptor_names=("goal_hit_count", "fall_count", "completion_time"),
        )
        flat_index = archive._ravel((6, 0, 2))
        archive._fitnesses[flat_index] = 42.0
        archive._genotypes[flat_index] = np.asarray([1.0, 2.0], dtype=np.float32)
        archive._descriptors[flat_index] = np.asarray([1.0, 0.0, 0.2], dtype=np.float32)

        candidates = train_module._archive_pool_candidates(archive)

        assert len(candidates) == 1
        assert candidates[0]["archive_flat_index"] == flat_index
        assert candidates[0]["archive_cell_indices"] == (6, 0, 2)
        assert candidates[0]["archive_cell_id"] == "cell_6-0-2"
        assert candidates[0]["archive_cell_label"] == (
            "goal_hit_count=6|fall_count=0|completion_time=2"
        )

    def test_select_specimen_rows_uses_historical_tiers_without_completion_bin_cap(self):
        cfg = compose_experiment_config(
            preset="paper_run",
            overrides=[
                "specimens.priority_min_mirrored_pair_count=4",
                "specimens.per_completion_bin=2",
            ],
        )
        rows = [
            {
                "name": "perfect_archive",
                "mirrored_pair_count": 5.0,
                "perfect_hits": True,
                "zero_falls": True,
                "goal_hit_count": 12.0,
                "fall_count": 0.0,
                "completion_time": 0.4,
                "activation_cost": 0.2,
                "source": "archive",
                "source_index": 0,
                "completion_bin": 0,
                "historical_elite_tier": "perfect_zero_fall",
            },
            {
                "name": "perfect_historical",
                "mirrored_pair_count": 3.0,
                "perfect_hits": True,
                "zero_falls": True,
                "goal_hit_count": 12.0,
                "fall_count": 0.0,
                "completion_time": 0.2,
                "activation_cost": 0.5,
                "source": "historical_elite",
                "source_index": 1,
                "completion_bin": 0,
                "historical_elite_tier": "perfect_zero_fall",
            },
            {
                "name": "near_perfect_one",
                "mirrored_pair_count": 4.0,
                "perfect_hits": True,
                "zero_falls": True,
                "goal_hit_count": 11.0,
                "fall_count": 0.0,
                "completion_time": 0.3,
                "activation_cost": 0.1,
                "source": "fresh_sample",
                "source_index": 0,
                "completion_bin": 0,
                "historical_elite_tier": "near_perfect_zero_fall",
            },
            {
                "name": "no_history_even_if_good",
                "mirrored_pair_count": 6.0,
                "perfect_hits": True,
                "zero_falls": True,
                "goal_hit_count": 12.0,
                "fall_count": 0.0,
                "completion_time": 0.1,
                "activation_cost": 0.0,
                "source": "archive",
                "source_index": 2,
                "completion_bin": 0,
                "historical_elite_tier": None,
            },
            {
                "name": "near_perfect_two",
                "mirrored_pair_count": 2.0,
                "perfect_hits": False,
                "zero_falls": True,
                "goal_hit_count": 10.0,
                "fall_count": 0.0,
                "completion_time": 0.6,
                "activation_cost": 0.2,
                "source": "archive",
                "source_index": 3,
                "completion_bin": 0,
                "historical_elite_tier": "near_perfect_zero_fall",
            },
        ]

        selected, qualifying_count = train_module._select_specimen_rows(rows, cfg=cfg)

        assert qualifying_count == 4
        assert [row["name"] for row in selected] == [
            "perfect_archive",
            "perfect_historical",
            "near_perfect_one",
            "near_perfect_two",
        ]

    def test_render_selected_specimen_previews_batches_once_and_fans_out_per_specimen(
        self,
        tmp_path,
        monkeypatch,
    ):
        cfg = _compose("iter_run")
        runtime_resolution = resolve_runtime_config(cfg)
        calls: list[dict[str, object]] = []

        def _fake_render_ranked_elites(
            *,
            elites,
            env_datas,
            alive_mask,
            cfg,
            generation,
            stage,
            output_dir,
            runtime_resolution,
        ):
            del env_datas, alive_mask, cfg, generation, runtime_resolution
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            calls.append(
                {
                    "elite_count": len(elites),
                    "stage": str(stage),
                    "output_dir": str(output_dir),
                }
            )
            artifacts = []
            for rank in range(len(elites)):
                for episode_idx in range(2):
                    gif_path = output_dir / f"rank_{rank}_ep_{episode_idx}.gif"
                    gif_path.write_bytes(b"GIF89a")
                    artifacts.append(
                        RenderArtifact(
                            stage=str(stage),
                            rank=int(rank),
                            episode_idx=int(episode_idx),
                            flat_index=int(rank),
                            archive_fitness=1.0,
                            selection_fitness=1.0,
                            cliff_x=10,
                            terrain_idx=int(episode_idx),
                            gif_path=str(gif_path),
                            episode_fitness=1.0,
                            goal_hit_count=6.0,
                            fall_count=0.0,
                            completion_time=0.5,
                            peak_front_x=1.0,
                            safe_front_x=1.0,
                            probe_extent=0.0,
                            turn_count=0.0,
                            stable_turn_count=0.0,
                            turn_motion_threshold=0.0,
                        )
                    )
            return artifacts

        monkeypatch.setattr(train_module, "render_ranked_elites", _fake_render_ranked_elites)

        preview_map = train_module._render_selected_specimen_previews(
            specimen_root=tmp_path,
            selected_candidates=[
                {
                    "specimen_id": "alpha",
                    "selection_row": {
                        "archive_flat_index": 7,
                        "fitness": 101.0,
                        "goal_hit_count": 12.0,
                        "fall_count": 0.0,
                        "completion_time": 0.4,
                    },
                    "candidate": {
                        "genotype": np.asarray([1.0, 2.0], dtype=np.float32),
                        "archive_descriptor": np.asarray([0.1, 0.2], dtype=np.float32),
                    },
                },
                {
                    "specimen_id": "beta",
                    "selection_row": {
                        "archive_flat_index": 8,
                        "fitness": 99.0,
                        "goal_hit_count": 11.0,
                        "fall_count": 0.0,
                        "completion_time": 0.5,
                    },
                    "candidate": {
                        "genotype": np.asarray([3.0, 4.0], dtype=np.float32),
                        "archive_descriptor": np.asarray([0.3, 0.4], dtype=np.float32),
                    },
                },
            ],
            preview_entry_indices=[0, 5],
            env_datas=[object(), object()],
            alive_mask=jnp.ones((2, 2), dtype=bool),
            cfg=cfg,
            generation=7,
            runtime_resolution=runtime_resolution,
        )

        assert calls == [
            {
                "elite_count": 2,
                "stage": "specimen_preview",
                "output_dir": str(tmp_path / "_preview_batch"),
            }
        ]
        assert sorted(preview_map) == ["alpha", "beta"]
        assert len(preview_map["alpha"]) == 2
        assert len(preview_map["beta"]) == 2
        assert (tmp_path / "specimen_alpha" / "preview_renders" / "preview_00.gif").exists()
        assert (tmp_path / "specimen_alpha" / "preview_renders" / "preview_01.gif").exists()
        assert (tmp_path / "specimen_beta" / "preview_renders" / "preview_00.gif").exists()
        assert not (tmp_path / "_preview_batch").exists()

        manifest = json.loads(
            (tmp_path / "specimen_alpha" / "preview_renders" / "manifest.json").read_text()
        )
        assert manifest["specimen_id"] == "alpha"
        assert manifest["render_count"] == 2
        assert manifest["preview_entry_indices"] == [0, 5]

    def test_prepare_training_inputs_grown_reuses_global_static_capacity(self):
        cfg = compose_experiment_config(
            preset="iter_run",
            overrides=[
                "body.mode=grown",
                "terrain=cliff_only",
                "train.n_generations=2",
                "train.n_episodes=1",
                "train.require_schedule_dir=false",
            ],
        )
        inputs = _prepare_training_inputs(cfg, np.random.default_rng(0))

        terrain_a = build_cliff_terrain(cfg.terrain.grid_h, cfg.terrain.grid_w, cliff_x=12)
        terrain_b = build_cliff_terrain(cfg.terrain.grid_h, cfg.terrain.grid_w, cliff_x=18)
        inputs.terrain_generations = [
            [(terrain_a, 12, [])],
            [(terrain_b, 18, [])],
        ]
        template_a = build_static_collider_template(
            terrain_a,
            cliff_x=12,
            canvas_h=cfg.body.growth.max_robot_h,
            canvas_w=cfg.body.growth.max_robot_w,
            spawn_x=cfg.terrain.spawn_x,
            spawn_y=cfg.terrain.spawn_y,
        )
        template_b = build_static_collider_template(
            terrain_b,
            cliff_x=18,
            canvas_h=cfg.body.growth.max_robot_h,
            canvas_w=cfg.body.growth.max_robot_w,
            spawn_x=cfg.terrain.spawn_x,
            spawn_y=cfg.terrain.spawn_y,
        )
        inputs.generation_static_capacity = infer_static_template_capacity([template_a, template_b])
        inputs._generation_env_cache.clear()

        envs_a = inputs.envs_for_generation(0)
        envs_b = inputs.envs_for_generation(1)

        assert envs_a.static_capacity == inputs.generation_static_capacity
        assert envs_b.static_capacity == inputs.generation_static_capacity
        assert envs_a.static_templates.static_collider_data.point_positions.shape[1] == inputs.generation_static_capacity.max_static_points
        assert envs_b.static_templates.static_collider_data.point_positions.shape[1] == inputs.generation_static_capacity.max_static_points
        assert envs_a.static_templates.static_collider_data.cell_vertices.shape[1] == inputs.generation_static_capacity.max_static_cells
        assert envs_b.static_templates.static_collider_data.cell_vertices.shape[1] == inputs.generation_static_capacity.max_static_cells

    def test_prepare_training_terrains_rejects_schedule_metadata_mismatch(self, tmp_path):
        cfg = compose_experiment_config(
            preset="iter_run",
            overrides=[
                "train.n_generations=1",
                "train.n_episodes=1",
            ],
        )
        terrain_cfg = dict(OmegaConf.to_container(cfg.terrain, resolve=True))
        schedule = generate_schedule(
            seed=int(cfg.seed),
            n_gens=1,
            n_episodes=2,
            n_val=int(cfg.eval.rerank_episodes),
            n_test=int(cfg.eval.test_episodes),
            terrain_cfg=terrain_cfg,
        )
        schedule_dir = save_schedule(schedule, tmp_path / "sched")
        cfg.train.terrain_schedule_dir = str(schedule_dir)

        with pytest.raises(ValueError, match="Terrain schedule metadata mismatch"):
            train_module._prepare_training_terrains(cfg, np.random.default_rng(0))

    def test_prepare_training_terrains_accepts_legacy_alias_variant_for_iter_run(self, tmp_path):
        cfg = compose_experiment_config(
            preset="iter_run",
            overrides=[
                "train.n_generations=1",
                "train.n_episodes=6",
                "eval.rerank_episodes=6",
                "eval.test_episodes=6",
            ],
        )
        terrain_cfg = dict(OmegaConf.to_container(cfg.terrain, resolve=True))
        terrain_cfg["variant"] = "bilateral_goal_cliff_asymmetric_pairs_8x2_30"
        schedule = generate_schedule(
            seed=int(cfg.seed),
            n_gens=1,
            n_episodes=6,
            n_val=6,
            n_test=6,
            terrain_cfg=terrain_cfg,
        )
        schedule_dir = save_schedule(schedule, tmp_path / "sched_alias")
        cfg.train.terrain_schedule_dir = str(schedule_dir)

        terrains, metadata = train_module._prepare_training_terrains(cfg, np.random.default_rng(0))

        assert len(terrains) == 1
        assert len(terrains[0]) == 6
        assert metadata is not None

    def test_periodic_render_failure_is_best_effort(self, monkeypatch, caplog):
        cfg = compose_experiment_config(preset="iter_run", overrides=["body.mode=grown"])

        def _raise(*args, **kwargs):
            raise RuntimeError("render boom")

        monkeypatch.setattr(train_module, "render_periodic_archive", _raise)
        with caplog.at_level(logging.ERROR):
            artifacts = train_module._render_periodic_archive_best_effort(
                archive=object(),
                env_datas=object(),
                alive_mask=object(),
                cfg=cfg,
                generation=3,
                output_root="renders",
            )

        assert artifacts == []
        assert "Periodic render failed | generation=3" in caplog.text

    def test_update_archive_scored_batch_commits_only_accepted_winners(self, monkeypatch):
        archive = GridArchive(
            dims=(10,),
            ranges=[(0.0, 1.0)],
            learning_rate=0.5,
            threshold_min=0.0,
        )
        candidate_batch = jnp.asarray(
            [[10.0], [20.0], [30.0], [40.0]],
            dtype=jnp.float32,
        )
        fitnesses = np.array([5.0, 7.0, -1.0, 4.0], dtype=np.float64)
        descriptors = np.array([[0.1], [0.1], [0.4], [0.8]], dtype=np.float64)

        captured: dict[str, np.ndarray] = {}
        original_commit = archive.commit_scored_batch

        def _capture_commit(decision, genotypes, winner_descriptors):
            captured["accepted_indices"] = np.array(decision.accepted_indices, copy=True)
            captured["genotypes"] = np.array(genotypes, copy=True)
            captured["descriptors"] = np.array(winner_descriptors, copy=True)
            original_commit(decision, genotypes, winner_descriptors)

        monkeypatch.setattr(archive, "commit_scored_batch", _capture_commit)

        decision = train_module._update_archive_scored_batch(
            archive,
            candidate_batch,
            fitnesses,
            descriptors,
        )

        np.testing.assert_array_equal(
            captured["accepted_indices"],
            np.array([1, 3], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            captured["genotypes"],
            np.array([[20.0], [40.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            captured["descriptors"],
            np.array([[0.1], [0.8]], dtype=np.float64),
        )
        np.testing.assert_allclose(decision.improvements, [5.0, 7.0, -1.0, 4.0])
        np.testing.assert_array_equal(
            decision.statuses,
            np.array(
                [
                    GridArchive.NOT_ADDED,
                    GridArchive.NEW,
                    GridArchive.NOT_ADDED,
                    GridArchive.NEW,
                ],
                dtype=np.uint8,
            ),
        )

    def test_update_archive_scored_batch_forwards_active_cell_mask(self, monkeypatch):
        archive = GridArchive(dims=(4,), ranges=[(0.0, 1.0)])
        candidate_batch = jnp.asarray([[1.0], [2.0]], dtype=jnp.float32)
        fitnesses = np.asarray([3.0, 4.0], dtype=np.float64)
        descriptors = np.asarray([[0.2], [0.8]], dtype=np.float64)
        active_mask = np.asarray([True, False, True, False], dtype=bool)
        captured: dict[str, np.ndarray] = {}

        def _fake_score_batch(fitnesses_arg, descriptors_arg, active_cell_mask=None):
            _ = fitnesses_arg
            _ = descriptors_arg
            captured["mask"] = np.asarray(active_cell_mask, dtype=bool)
            return GridArchiveBatchDecision(
                improvements=np.zeros(2, dtype=np.float64),
                statuses=np.zeros(2, dtype=np.uint8),
                accepted_indices=np.zeros((0,), dtype=np.int64),
                accepted_flat_indices=np.zeros((0,), dtype=np.int64),
                accepted_fitnesses=np.zeros((0,), dtype=np.float64),
                accepted_statuses=np.zeros((0,), dtype=np.uint8),
                accepted_thresholds=np.zeros((0,), dtype=np.float64),
                accepted_pre_fitnesses=np.zeros((0,), dtype=np.float64),
                descriptor_finite_mask=np.ones(2, dtype=bool),
                invalid_descriptor_count=0,
            )

        monkeypatch.setattr(archive, "score_batch", _fake_score_batch)

        train_module._update_archive_scored_batch(
            archive,
            candidate_batch,
            fitnesses,
            descriptors,
            active_cell_mask=active_mask,
        )

        np.testing.assert_array_equal(captured["mask"], active_mask)

    def test_write_generation_probe_artifacts_writes_full_batch_repeat_and_trace_outputs(
        self, tmp_path, monkeypatch
    ):
        cfg = compose_experiment_config(
            preset="paper_run",
            overrides=[
                "audit.enabled=true",
                "audit.generation_eval_mode=full_host",
                "audit.repeat_full_generation_batch=true",
                "audit.compare_trace_on_full_generation_batch=true",
                "audit.compare_floor_stiction_counterfactual=true",
                "audit.save_full_generation_genotypes=true",
                "audit.probe_internal_phase_divergence=true",
                "qd.descriptors.names=[activation_cost,peak_progress_rate]",
            ],
        )
        archive_spec = train_module.resolve_qd_archive(cfg)
        archive = GridArchive(
            dims=archive_spec.dims,
            ranges=list(archive_spec.ranges),
            descriptor_names=archive_spec.descriptor_names,
        )
        candidate_batch = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
        candidate_keys = jnp.asarray([[1, 2], [3, 4]], dtype=jnp.uint32)
        fitnesses = np.asarray([10.0, 20.0], dtype=np.float32)
        descriptors = np.asarray([[0.1, 0.2], [0.7, 0.8]], dtype=np.float32)
        batch_result = PopulationBatchSummary(
            fitness=fitnesses,
            goal_hit_count=np.asarray([1.0, 2.0], dtype=np.float32),
            mirrored_pair_count=np.asarray([0.0, 1.0], dtype=np.float32),
            episode_fell=np.zeros((2, 2), dtype=bool),
            episode_left_edge=np.zeros((2, 2), dtype=bool),
            deformation_channel_max=np.zeros((2, 2), dtype=np.float32),
            controller_hidden_output_max_abs=np.zeros((2,), dtype=np.float32),
            action_output_max_abs=np.zeros((2,), dtype=np.float32),
            descriptor_columns={
                "fall_count": np.asarray([0.0, 0.0], dtype=np.float32),
                "time_to_edge": np.asarray([12.0, 24.0], dtype=np.float32),
            },
        )
        decision = train_module._update_archive_scored_batch(
            archive,
            candidate_batch,
            fitnesses,
            descriptors,
        )

        repeat_results = [
            SimpleNamespace(
                batch_result=PopulationBatchSummary(
                    fitness=np.asarray([10.0, 18.0], dtype=np.float32),
                    goal_hit_count=np.asarray([1.0, 2.0], dtype=np.float32),
                    mirrored_pair_count=np.asarray([0.0, 1.0], dtype=np.float32),
                    episode_fell=np.zeros((2, 2), dtype=bool),
                    episode_left_edge=np.zeros((2, 2), dtype=bool),
                    deformation_channel_max=np.zeros((2, 2), dtype=np.float32),
                    controller_hidden_output_max_abs=np.zeros((2,), dtype=np.float32),
                    action_output_max_abs=np.zeros((2,), dtype=np.float32),
                    descriptor_columns={
                        "fall_count": np.asarray([0.0, 0.0], dtype=np.float32),
                        "time_to_edge": np.asarray([12.0, 23.0], dtype=np.float32),
                    },
                ),
                fitnesses=np.asarray([10.0, 18.0], dtype=np.float32),
                descriptors=np.asarray([[0.1, 0.2], [0.7, 0.85]], dtype=np.float32),
            ),
            SimpleNamespace(
                batch_result=PopulationBatchSummary(
                    fitness=np.asarray([10.0, 18.0], dtype=np.float32),
                    goal_hit_count=np.asarray([1.0, 2.0], dtype=np.float32),
                    mirrored_pair_count=np.asarray([0.0, 1.0], dtype=np.float32),
                    episode_fell=np.zeros((2, 2), dtype=bool),
                    episode_left_edge=np.zeros((2, 2), dtype=bool),
                    deformation_channel_max=np.zeros((2, 2), dtype=np.float32),
                    controller_hidden_output_max_abs=np.zeros((2,), dtype=np.float32),
                    action_output_max_abs=np.zeros((2,), dtype=np.float32),
                    descriptor_columns={
                        "fall_count": np.asarray([0.0, 0.0], dtype=np.float32),
                        "time_to_edge": np.asarray([12.0, 23.0], dtype=np.float32),
                    },
                ),
                fitnesses=np.asarray([10.0, 18.0], dtype=np.float32),
                descriptors=np.asarray([[0.1, 0.2], [0.7, 0.85]], dtype=np.float32),
            ),
            SimpleNamespace(
                batch_result=PopulationBatchSummary(
                    fitness=np.asarray([10.0, 17.5], dtype=np.float32),
                    goal_hit_count=np.asarray([1.0, 2.0], dtype=np.float32),
                    mirrored_pair_count=np.asarray([0.0, 1.0], dtype=np.float32),
                    episode_fell=np.zeros((2, 2), dtype=bool),
                    episode_left_edge=np.zeros((2, 2), dtype=bool),
                    deformation_channel_max=np.zeros((2, 2), dtype=np.float32),
                    controller_hidden_output_max_abs=np.zeros((2,), dtype=np.float32),
                    action_output_max_abs=np.zeros((2,), dtype=np.float32),
                    descriptor_columns={
                        "fall_count": np.asarray([0.0, 0.0], dtype=np.float32),
                        "time_to_edge": np.asarray([12.0, 23.0], dtype=np.float32),
                    },
                ),
                fitnesses=np.asarray([10.0, 17.5], dtype=np.float32),
                descriptors=np.asarray([[0.1, 0.2], [0.7, 0.75]], dtype=np.float32),
            ),
        ]
        trace_batch_result = PopulationBatchSummary(
            fitness=np.asarray([9.5, 20.0], dtype=np.float32),
            goal_hit_count=np.asarray([1.0, 2.0], dtype=np.float32),
            mirrored_pair_count=np.asarray([0.0, 1.0], dtype=np.float32),
            episode_fell=np.zeros((2, 2), dtype=bool),
            episode_left_edge=np.zeros((2, 2), dtype=bool),
            deformation_channel_max=np.zeros((2, 2), dtype=np.float32),
            controller_hidden_output_max_abs=np.zeros((2,), dtype=np.float32),
            action_output_max_abs=np.zeros((2,), dtype=np.float32),
            descriptor_columns={
                "fall_count": np.asarray([0.0, 0.0], dtype=np.float32),
                "time_to_edge": np.asarray([13.0, 24.0], dtype=np.float32),
            },
        )
        captured: dict[str, object] = {}
        env_datas = [
            DummyAuditEnv(
                constants=default_physics_constants()._replace(
                    floor_static_friction_const=0.4,
                    floor_stiction_stiffness=20_000.0,
                    floor_stiction_switch_speed=0.05,
                ),
                payload=7,
            )
        ]

        def _fake_evaluate_probe_candidates(**kwargs):
            captured.setdefault("repeat_modes", []).append(kwargs["reduced_host_result"])
            captured.setdefault("env_datas", []).append(kwargs["env_datas"])
            return repeat_results.pop(0)

        def _fake_trace_probe_candidates(**kwargs):
            captured["trace_workload"] = kwargs["workload"]
            return SimpleNamespace(episode_results="trace-episodes")

        def _fake_materialize_trace_batch_result(*, trace_eval, alive_mask, cfg):
            assert trace_eval.episode_results == "trace-episodes"
            assert alive_mask.shape == (2, 2)
            return (
                trace_batch_result,
                np.asarray([9.5, 20.0], dtype=np.float32),
                np.asarray([[0.15, 0.2], [0.7, 0.8]], dtype=np.float32),
            )

        def _fake_internal_phase_probe(**kwargs):
            assert kwargs["generation"] == 0
            assert kwargs["candidate_genotypes"].shape == (2, 2)
            return {
                "summary_path": str(
                    tmp_path
                    / "audit"
                    / "generations"
                    / "gen_0000_internal_phase_probe_summary.json"
                ),
                "rows_path": str(
                    tmp_path
                    / "audit"
                    / "generations"
                    / "gen_0000_internal_phase_probe_episodes.jsonl"
                ),
                "selected_candidate_index": 1,
                "selected_episode_index": 0,
                "selected_episode_first_mismatch_category": "physics_or_collision",
                "selected_episode_first_mismatch_stage": "phase_post_positions",
                "selected_episode_first_mismatch_step": 0,
                "selected_episode_first_mismatch_phase": 1,
                "selected_episode_substep_probe_initial_step_index": 0,
                "selected_episode_substep_probe_initial_phase_index": 2,
                "selected_episode_substep_probe_step_index": 0,
                "selected_episode_substep_probe_phase_index": 1,
                "selected_episode_substep_probe_backtrack_count": 1,
                "selected_episode_substep_probe_backtrack_exhausted": False,
                "selected_episode_substep_first_mismatch_category": "integrator",
                "selected_episode_substep_first_mismatch_stage": "substep_post_integrator_positions",
                "selected_episode_substep_first_mismatch_substep": 2,
                "selected_episode_substep_trace_npz_path": str(
                    tmp_path
                    / "audit"
                    / "generations"
                    / "gen_0000_internal_substep_probe_candidate_0001_episode_00_step_0000_phase_01.npz"
                ),
                "controller_first_mismatch_count": 0,
                "physics_or_collision_first_mismatch_count": 1,
                "pre_physics_state_first_mismatch_count": 0,
            }

        monkeypatch.setattr(
            train_module, "_evaluate_probe_candidates", _fake_evaluate_probe_candidates
        )
        monkeypatch.setattr(
            train_module, "_trace_probe_candidates", _fake_trace_probe_candidates
        )
        monkeypatch.setattr(
            train_module,
            "_materialize_trace_batch_result",
            _fake_materialize_trace_batch_result,
        )
        monkeypatch.setattr(
            train_module,
            "_write_internal_phase_divergence_probe_artifacts",
            _fake_internal_phase_probe,
        )

        summary_path = train_module._write_generation_probe_artifacts(
            output_dir=tmp_path,
            generation=0,
            candidate_batch=candidate_batch,
            candidate_keys=candidate_keys,
            batch_result=batch_result,
            fitnesses=fitnesses,
            descriptors=descriptors,
            archive=archive,
            decision=decision,
            cfg=cfg,
            env_datas=env_datas,
            alive_mask=jnp.ones((2, 2), dtype=jnp.bool_),
            runtime_resolution=SimpleNamespace(),
            static_capacity=None,
        )

        assert summary_path == tmp_path / "audit" / "generations" / "gen_0000_summary.json"
        assert captured["repeat_modes"] == [False, False, False]
        assert captured["env_datas"][0][0].constants.floor_static_friction_const == pytest.approx(
            0.4
        )
        assert captured["env_datas"][2][0].constants.floor_static_friction_const == pytest.approx(
            0.0
        )
        assert captured["env_datas"][2][0].constants.floor_stiction_stiffness == pytest.approx(
            0.0
        )
        assert captured["env_datas"][2][0].constants.floor_stiction_switch_speed == pytest.approx(
            0.0
        )
        assert captured["trace_workload"] == "generation"

        summary = json.loads(summary_path.read_text())
        assert summary["archive_fitness_mismatch_count"] == 0
        assert summary["archive_descriptor_mismatch_count"] == 0
        assert summary["full_batch_main_vs_repeat_eval_1_fitness_mismatch_count"] == 1
        assert summary["full_batch_repeat_eval_1_vs_repeat_eval_2_fitness_mismatch_count"] == 0
        assert summary["full_batch_main_vs_floor_counterfactual_fitness_mismatch_count"] == 1
        assert summary["full_batch_main_vs_trace_eval_fitness_mismatch_count"] == 1

        accepted_rows_path = tmp_path / "audit" / "generations" / "gen_0000_accepted_winners.jsonl"
        assert accepted_rows_path.exists()
        accepted_rows = [
            json.loads(line)
            for line in accepted_rows_path.read_text().splitlines()
            if line.strip()
        ]
        assert len(accepted_rows) == 2

        full_batch_summary_path = (
            tmp_path / "audit" / "generations" / "gen_0000_full_batch_summary.json"
        )
        full_batch_rows_path = (
            tmp_path / "audit" / "generations" / "gen_0000_full_batch_rows.jsonl"
        )
        full_batch_inputs_path = (
            tmp_path / "audit" / "generations" / "gen_0000_full_batch_inputs.npz"
        )
        full_batch_summary = json.loads(full_batch_summary_path.read_text())
        assert full_batch_summary["repeated_count"] == 2
        assert full_batch_summary["floor_counterfactual_mode"] == "floor_stiction_disabled"
        assert full_batch_summary["floor_counterfactual_compared_count"] == 2
        assert full_batch_summary["trace_compared_count"] == 2
        assert full_batch_summary["main_vs_repeat_eval_1_fitness_mismatch_count"] == 1
        assert full_batch_summary["repeat_eval_1_vs_repeat_eval_2_fitness_mismatch_count"] == 0
        assert full_batch_summary["main_vs_floor_counterfactual_fitness_mismatch_count"] == 1
        assert full_batch_summary["main_vs_trace_eval_descriptor_mismatch_count"] == 1
        assert (
            full_batch_summary["internal_phase_probe_selected_episode_first_mismatch_category"]
            == "physics_or_collision"
        )
        assert (
            full_batch_summary["internal_phase_probe_selected_episode_first_mismatch_stage"]
            == "phase_post_positions"
        )
        assert (
            full_batch_summary[
                "internal_phase_probe_selected_episode_substep_probe_initial_phase_index"
            ]
            == 2
        )
        assert (
            full_batch_summary[
                "internal_phase_probe_selected_episode_substep_probe_backtrack_count"
            ]
            == 1
        )
        assert (
            full_batch_summary[
                "internal_phase_probe_selected_episode_substep_first_mismatch_category"
            ]
            == "integrator"
        )
        assert (
            full_batch_summary[
                "internal_phase_probe_selected_episode_substep_first_mismatch_stage"
            ]
            == "substep_post_integrator_positions"
        )
        assert (
            full_batch_summary[
                "internal_phase_probe_selected_episode_substep_first_mismatch_substep"
            ]
            == 2
        )

        full_batch_rows = [
            json.loads(line)
            for line in full_batch_rows_path.read_text().splitlines()
            if line.strip()
        ]
        assert full_batch_rows[0]["main_vs_repeat_eval_1_fitness_match"] is True
        assert full_batch_rows[1]["main_vs_repeat_eval_1_fitness_match"] is False
        assert full_batch_rows[1]["repeat_eval_1_vs_repeat_eval_2_fitness_match"] is True
        assert full_batch_rows[1]["main_vs_floor_counterfactual_fitness_match"] is False
        assert full_batch_rows[0]["main_vs_trace_eval_descriptor_match"] is False

        with np.load(full_batch_inputs_path) as saved:
            np.testing.assert_allclose(saved["genotypes"], np.asarray(candidate_batch))
            np.testing.assert_array_equal(saved["candidate_keys"], np.asarray(candidate_keys))
            np.testing.assert_allclose(saved["main_fitnesses"], fitnesses)
            np.testing.assert_allclose(saved["main_descriptors"], descriptors)

    def test_first_phase_trace_divergence_classifies_controller_stage(self):
        trace_first = SimpleNamespace(
            initial_positions=np.zeros((2, 2), dtype=np.float32),
            initial_velocities=np.zeros((2, 2), dtype=np.float32),
            initial_spring_rest_length=np.zeros((3,), dtype=np.float32),
            initial_nca_state=np.zeros((1, 1, 2), dtype=np.float32),
            initial_ctrnn_carry=np.zeros((1, 1, 2), dtype=np.float32),
            phase_pre_positions=np.zeros((1, 1, 2, 2), dtype=np.float32),
            phase_pre_velocities=np.zeros((1, 1, 2, 2), dtype=np.float32),
            phase_pre_spring_rest_length=np.zeros((1, 1, 3), dtype=np.float32),
            phase_h_compact=np.zeros((1, 1, 2), dtype=np.float32),
            phase_v_compact=np.zeros((1, 1, 2), dtype=np.float32),
            phase_controller_hidden_output_max_abs=np.zeros((1, 1), dtype=np.float32),
            phase_action_output_max_abs=np.zeros((1, 1), dtype=np.float32),
            phase_post_positions=np.zeros((1, 1, 2, 2), dtype=np.float32),
            phase_post_velocities=np.zeros((1, 1, 2, 2), dtype=np.float32),
            phase_post_spring_rest_length=np.zeros((1, 1, 3), dtype=np.float32),
            phase_done_before=np.zeros((1, 1), dtype=bool),
            phase_done_after=np.zeros((1, 1), dtype=bool),
        )
        trace_second = SimpleNamespace(
            **{
                field: np.copy(getattr(trace_first, field))
                for field in trace_first.__dict__.keys()
            }
        )
        trace_second.phase_h_compact[0, 0, 1] = 1.5

        divergence = train_module._first_phase_trace_divergence(
            trace_first, trace_second
        )

        assert divergence["category"] == "controller"
        assert divergence["stage"] == "phase_h_compact"
        assert divergence["step"] == 0
        assert divergence["phase"] == 0
        assert divergence["max_abs_diff"] == pytest.approx(1.5)

    def test_first_substep_trace_divergence_classifies_integrator_stage(self):
        trace_first = SimpleNamespace(
            phase_pre_positions=np.zeros((2, 2), dtype=np.float32),
            phase_pre_velocities=np.zeros((2, 2), dtype=np.float32),
            phase_pre_spring_rest_length=np.zeros((3,), dtype=np.float32),
            phase_h_compact=np.zeros((2,), dtype=np.float32),
            phase_v_compact=np.zeros((2,), dtype=np.float32),
            phase_controller_hidden_output_max_abs=np.float32(0.0),
            phase_action_output_max_abs=np.float32(0.0),
            substep_dynamic_selected_pair_mask=np.zeros((2, 4), dtype=bool),
            substep_dynamic_overflow=np.zeros((2,), dtype=bool),
            substep_dynamic_selected_triple_active=np.zeros((2, 5), dtype=bool),
            substep_dynamic_triple_active=np.zeros((2, 5), dtype=bool),
            substep_dynamic_forces=np.zeros((2, 3, 2), dtype=np.float32),
            substep_static_forces=np.zeros((2, 3, 2), dtype=np.float32),
            substep_static_contact_mask=np.zeros((2, 3), dtype=bool),
            substep_static_memory_contact_mask=np.zeros((2, 3), dtype=bool),
            substep_static_manifold_active=np.zeros((2, 3), dtype=bool),
            substep_tangential_deformation=np.zeros((2, 3, 2), dtype=np.float32),
            substep_post_update_actuator_spring_rest_length=np.zeros((2, 3), dtype=np.float32),
            substep_post_integrator_positions=np.zeros((2, 3, 2), dtype=np.float32),
            substep_post_integrator_velocities=np.zeros((2, 3, 2), dtype=np.float32),
            substep_post_constraint_positions=np.zeros((2, 3, 2), dtype=np.float32),
            substep_post_anchor_positions=np.zeros((2, 3, 2), dtype=np.float32),
            substep_final_positions=np.zeros((2, 3, 2), dtype=np.float32),
            substep_velocity_true_positions_last=np.zeros((2, 3, 2), dtype=np.float32),
            substep_velocity_true_displacement=np.zeros((2, 3, 2), dtype=np.float32),
            substep_final_velocities=np.zeros((2, 3, 2), dtype=np.float32),
            substep_final_velocities_true=np.zeros((2, 3, 2), dtype=np.float32),
        )
        trace_second = SimpleNamespace(
            **{
                field: np.copy(getattr(trace_first, field))
                for field in trace_first.__dict__.keys()
            }
        )
        trace_second.substep_post_integrator_positions[1, 2, 0] = 0.25

        divergence = train_module._first_substep_trace_divergence(
            trace_first,
            trace_second,
        )

        assert divergence["category"] == "integrator"
        assert divergence["stage"] == "substep_post_integrator_positions"
        assert divergence["substep"] == 1
        assert divergence["max_abs_diff"] == pytest.approx(0.25)

    def test_first_substep_trace_divergence_classifies_velocity_true_reconstruction_stage(
        self,
    ):
        trace_first = SimpleNamespace(
            phase_pre_positions=np.zeros((1, 2), dtype=np.float32),
            phase_pre_velocities=np.zeros((1, 2), dtype=np.float32),
            phase_pre_spring_rest_length=np.zeros((3,), dtype=np.float32),
            phase_h_compact=np.zeros((2,), dtype=np.float32),
            phase_v_compact=np.zeros((2,), dtype=np.float32),
            phase_controller_hidden_output_max_abs=np.float32(0.0),
            phase_action_output_max_abs=np.float32(0.0),
            substep_dynamic_selected_pair_mask=np.zeros((2, 4), dtype=bool),
            substep_dynamic_overflow=np.zeros((2,), dtype=bool),
            substep_dynamic_selected_triple_active=np.zeros((2, 5), dtype=bool),
            substep_dynamic_triple_active=np.zeros((2, 5), dtype=bool),
            substep_dynamic_forces=np.zeros((2, 3, 2), dtype=np.float32),
            substep_static_forces=np.zeros((2, 3, 2), dtype=np.float32),
            substep_static_contact_mask=np.zeros((2, 3), dtype=bool),
            substep_static_memory_contact_mask=np.zeros((2, 3), dtype=bool),
            substep_static_manifold_active=np.zeros((2, 3), dtype=bool),
            substep_tangential_deformation=np.zeros((2, 3, 2), dtype=np.float32),
            substep_post_update_actuator_spring_rest_length=np.zeros((2, 3), dtype=np.float32),
            substep_post_integrator_positions=np.zeros((2, 3, 2), dtype=np.float32),
            substep_post_integrator_velocities=np.zeros((2, 3, 2), dtype=np.float32),
            substep_post_constraint_positions=np.zeros((2, 3, 2), dtype=np.float32),
            substep_post_anchor_positions=np.zeros((2, 3, 2), dtype=np.float32),
            substep_final_positions=np.zeros((2, 3, 2), dtype=np.float32),
            substep_velocity_true_positions_last=np.zeros((2, 3, 2), dtype=np.float32),
            substep_velocity_true_displacement=np.zeros((2, 3, 2), dtype=np.float32),
            substep_final_velocities=np.zeros((2, 3, 2), dtype=np.float32),
            substep_final_velocities_true=np.zeros((2, 3, 2), dtype=np.float32),
        )
        trace_second = SimpleNamespace(
            **{
                field: np.copy(getattr(trace_first, field))
                for field in trace_first.__dict__.keys()
            }
        )
        trace_second.substep_velocity_true_positions_last[0, 1, 0] = 0.125

        divergence = train_module._first_substep_trace_divergence(
            trace_first,
            trace_second,
        )

        assert divergence["category"] == "velocity_true_reconstruction"
        assert divergence["stage"] == "substep_velocity_true_positions_last"
        assert divergence["substep"] == 0
        assert divergence["max_abs_diff"] == pytest.approx(0.125)

    def test_probe_selected_episode_substeps_with_backtrack_chases_pre_velocity_boundaries(
        self, monkeypatch
    ):
        calls: list[tuple[int, int]] = []
        divergences = {
            (1, 2): {
                "category": "pre_substep_state",
                "stage": "phase_pre_velocities",
                "substep": None,
                "max_abs_diff": 1.0,
            },
            (1, 1): {
                "category": "integrator",
                "stage": "substep_post_integrator_velocities",
                "substep": 0,
                "max_abs_diff": 0.25,
            },
        }

        def _fake_probe_candidate_episode_substeps(
            genotype,
            env_datas,
            alive_mask,
            key,
            cfg,
            *,
            selected_episode_index,
            selected_step_index,
            selected_phase_index,
            runtime_resolution,
            static_capacity,
        ):
            del genotype
            del env_datas
            del alive_mask
            del key
            del cfg
            del selected_episode_index
            del runtime_resolution
            del static_capacity
            calls.append((int(selected_step_index), int(selected_phase_index)))
            return SimpleNamespace(
                substep_trace=SimpleNamespace(
                    boundary=(int(selected_step_index), int(selected_phase_index))
                )
            )

        def _fake_first_substep_trace_divergence(trace_first, trace_second):
            assert trace_first.boundary == trace_second.boundary
            return divergences[trace_first.boundary]

        monkeypatch.setattr(
            train_module,
            "probe_candidate_episode_substeps",
            _fake_probe_candidate_episode_substeps,
        )
        monkeypatch.setattr(
            train_module,
            "_first_substep_trace_divergence",
            _fake_first_substep_trace_divergence,
        )

        summary, final_first, final_second = (
            train_module._probe_selected_episode_substeps_with_backtrack(
                selected_genotype=np.zeros((2,), dtype=np.float32),
                env_datas=[],
                alive_mask=jnp.ones((1, 1), dtype=jnp.bool_),
                selected_key=np.asarray([1, 2], dtype=np.uint32),
                cfg=compose_experiment_config(preset="paper_run"),
                selected_episode_index=0,
                selected_step_index=1,
                selected_phase_index=2,
                phase_count=3,
                runtime_resolution=SimpleNamespace(),
                static_capacity=None,
            )
        )

        assert calls == [(1, 2), (1, 2), (1, 1), (1, 1)]
        assert (
            summary["selected_episode_substep_probe_initial_step_index"],
            summary["selected_episode_substep_probe_initial_phase_index"],
        ) == (1, 2)
        assert (
            summary["selected_episode_substep_probe_step_index"],
            summary["selected_episode_substep_probe_phase_index"],
        ) == (1, 1)
        assert summary["selected_episode_substep_probe_backtrack_count"] == 1
        assert summary["selected_episode_substep_probe_backtrack_exhausted"] is False
        assert len(summary["selected_episode_substep_probe_attempts"]) == 2
        assert (
            summary["selected_episode_substep_first_mismatch_category"] == "integrator"
        )
        assert (
            summary["selected_episode_substep_first_mismatch_stage"]
            == "substep_post_integrator_velocities"
        )
        assert summary["selected_episode_substep_first_mismatch_substep"] == 0
        assert summary["selected_episode_substep_first_mismatch_max_abs_diff"] == pytest.approx(
            0.25
        )
        assert final_first.substep_trace.boundary == (1, 1)
        assert final_second.substep_trace.boundary == (1, 1)

    def test_phase_probe_write_site_diagnostics_detects_trace_write_only_divergence(self):
        trace_first = SimpleNamespace(
            phase_last_substep_post_integrator_velocities=np.zeros((1, 2, 3, 2), dtype=np.float32),
            phase_last_substep_final_velocities=np.zeros((1, 2, 3, 2), dtype=np.float32),
            phase_last_substep_final_velocities_true=np.zeros((1, 2, 3, 2), dtype=np.float32),
            phase_post_velocities=np.zeros((1, 2, 3, 2), dtype=np.float32),
            phase_post_velocities_true=np.zeros((1, 2, 3, 2), dtype=np.float32),
        )
        trace_second = SimpleNamespace(
            **{
                field: np.copy(getattr(trace_first, field))
                for field in trace_first.__dict__.keys()
            }
        )
        trace_second.phase_post_velocities[0, 1, 2, 0] = 0.5

        diagnostics = train_module._phase_probe_write_site_diagnostics(
            trace_first,
            trace_second,
            step=0,
            phase=1,
        )

        assert (
            diagnostics["selected_episode_phase_write_site_first_divergent_field"]
            == "phase_post_velocities"
        )
        assert (
            diagnostics["selected_episode_phase_write_site_first_divergent_category"]
            == "phase_trace_write"
        )
        assert diagnostics[
            "selected_episode_phase_write_site_last_substep_final_velocities_repeat_delta"
        ] == pytest.approx(0.0)
        assert diagnostics[
            "selected_episode_phase_write_site_post_velocities_repeat_delta"
        ] == pytest.approx(0.5)
        assert diagnostics[
            "selected_episode_phase_write_site_repeat_1_post_vs_last_substep_final_velocities_delta"
        ] == pytest.approx(0.0)
        assert diagnostics[
            "selected_episode_phase_write_site_repeat_2_post_vs_last_substep_final_velocities_delta"
        ] == pytest.approx(0.5)


class TestPostTrainScheduleResolution:
    def test_variant_only_paper_preset_skips_inline_post_train_eval(self):
        cfg = compose_experiment_config(preset="paper_run")

        assert train_module._resolve_post_train_schedule_dirs(cfg) is None

    def test_train_schedule_only_skips_inline_post_train_eval(self):
        cfg = compose_experiment_config(
            preset="paper_run",
            overrides=[
                "train.terrain_schedule_dir=/tmp/paper_run_schedule",
                "train.require_schedule_dir=true",
            ],
        )

        assert train_module._resolve_post_train_schedule_dirs(cfg) is None

    @pytest.mark.parametrize(
        ("overrides", "missing"),
        [
            (
                ["eval.rerank_schedule_dir=terrain_suites/seed_42_single_gap2_zone10to24_30"],
                "eval.test_schedule_dir",
            ),
            (
                ["eval.test_schedule_dir=terrain_suites/seed_42_single_gap3_zone10to24_30_ood"],
                "eval.rerank_schedule_dir",
            ),
        ],
    )
    def test_partial_inline_post_train_eval_config_names_missing_schedule(self, overrides, missing):
        cfg = compose_experiment_config(preset="paper_run", overrides=overrides)

        with pytest.raises(ValueError, match=missing):
            train_module._resolve_post_train_schedule_dirs(cfg)

    def test_explicit_inline_post_train_eval_config_enables_schedule_resolution(self):
        cfg = compose_experiment_config(
            preset="paper_run",
            overrides=[
                "eval.rerank_schedule_dir=terrain_suites/seed_42_single_gap2_zone10to24_30",
                "eval.test_schedule_dir=terrain_suites/seed_42_single_gap3_zone10to24_30_ood",
            ],
        )

        assert train_module._resolve_post_train_schedule_dirs(cfg) == (
            "terrain_suites/seed_42_single_gap2_zone10to24_30",
            "terrain_suites/seed_42_single_gap3_zone10to24_30_ood",
        )


class TestSpecimenStagePaths:
    def test_specimen_output_dir_uses_stage_subtree_for_paused_stages(self, tmp_path):
        cfg = compose_experiment_config(preset="paper_run")
        output_dir = tmp_path / "outputs" / "paper_run_rect_7x2_seed42"

        final_root = train_module._specimen_output_dir(output_dir, cfg)
        stage_root = train_module._specimen_output_dir(
            output_dir,
            cfg,
            generation=99,
            final_stage=False,
        )

        assert final_root == output_dir.parent / "specimens" / output_dir.name
        assert stage_root == final_root / "stages" / "gen_0099"


class TestResumeBootstrap:
    def test_bootstrap_reuses_in_progress_manifest_and_latest_durable_checkpoint(self, tmp_path):
        output_dir = tmp_path / "run"
        cfg = compose_experiment_config(
            preset="iter_run",
            overrides=[f"train.output_dir={output_dir}"],
        )
        resolved_config_str = OmegaConf.to_yaml(cfg)
        manifest_path = output_dir / cfg.logging.manifest_file
        output_dir.mkdir(parents=True)
        (output_dir / "resolved_config.yaml").write_text(resolved_config_str)
        manifest_module.save_manifest(
            manifest_module.create_manifest(
                run_id="run-1",
                wandb_run_id="wb-1",
                preset_id=cfg.get("preset_id"),
                resolved_config_str=resolved_config_str,
                logical_config_sha256=train_module.logical_config_sha256(cfg),
                resolved_config_path=str(output_dir / "resolved_config.yaml"),
                attempt_count=1,
            ),
            manifest_path,
        )

        checkpoint_dir = output_dir / cfg.train.checkpoint_dir
        checkpoint_dir.mkdir()
        valid_prefix = checkpoint_dir / "gen_0003"
        valid_prefix.with_suffix(".json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "sensing_edges_checkpoint",
                    "run_id": "run-1",
                    "generation": 3,
                    "created_at_utc": "2026-01-01T00:00:03+00:00",
                    "config_sha256": train_module.logical_config_sha256(cfg),
                }
            )
        )
        valid_prefix.with_suffix(".npz").write_bytes(b"npz")
        payload_done_path(valid_prefix).write_text("{}")

        partial_prefix = checkpoint_dir / "gen_0004"
        partial_prefix.with_suffix(".json").write_text(valid_prefix.with_suffix(".json").read_text())
        partial_prefix.with_suffix(".npz").write_bytes(b"npz")

        bootstrap = train_module._bootstrap_training_run(cfg)

        assert bootstrap.run_id == "run-1"
        assert bootstrap.wandb_run_id == "wb-1"
        assert bootstrap.attempt_count == 2
        assert bootstrap.resume_from == str(valid_prefix)
        assert bootstrap.resumed_generation == 3

    def test_bootstrap_prefers_rolling_latest_checkpoint_when_present(self, tmp_path):
        output_dir = tmp_path / "run"
        cfg = compose_experiment_config(
            preset="iter_run",
            overrides=[f"train.output_dir={output_dir}"],
        )
        resolved_config_str = OmegaConf.to_yaml(cfg)
        manifest_path = output_dir / cfg.logging.manifest_file
        output_dir.mkdir(parents=True)
        (output_dir / "resolved_config.yaml").write_text(resolved_config_str)
        manifest_module.save_manifest(
            manifest_module.create_manifest(
                run_id="run-1",
                wandb_run_id="wb-1",
                preset_id=cfg.get("preset_id"),
                resolved_config_str=resolved_config_str,
                logical_config_sha256=train_module.logical_config_sha256(cfg),
                resolved_config_path=str(output_dir / "resolved_config.yaml"),
                attempt_count=1,
            ),
            manifest_path,
        )

        checkpoint_dir = output_dir / cfg.train.checkpoint_dir
        checkpoint_dir.mkdir()
        archival_prefix = checkpoint_dir / "gen_0003"
        archival_prefix.with_suffix(".json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "sensing_edges_checkpoint",
                    "run_id": "run-1",
                    "generation": 3,
                    "created_at_utc": "2026-01-01T00:00:03+00:00",
                    "config_sha256": train_module.logical_config_sha256(cfg),
                }
            )
        )
        archival_prefix.with_suffix(".npz").write_bytes(b"npz")
        payload_done_path(archival_prefix).write_text("{}")

        latest_prefix = checkpoint_dir / "latest"
        latest_prefix.with_suffix(".json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "sensing_edges_checkpoint",
                    "run_id": "run-1",
                    "generation": 12,
                    "created_at_utc": "2026-01-01T00:00:12+00:00",
                    "config_sha256": train_module.logical_config_sha256(cfg),
                }
            )
        )
        latest_prefix.with_suffix(".npz").write_bytes(b"npz")
        payload_done_path(latest_prefix).write_text("{}")

        bootstrap = train_module._bootstrap_training_run(cfg)

        assert bootstrap.resume_from == str(latest_prefix)
        assert bootstrap.resumed_generation == 12

    def test_bootstrap_fails_for_completed_matching_output_dir(self, tmp_path):
        output_dir = tmp_path / "run"
        cfg = compose_experiment_config(
            preset="iter_run",
            overrides=[f"train.output_dir={output_dir}"],
        )
        resolved_config_str = OmegaConf.to_yaml(cfg)
        output_dir.mkdir(parents=True)
        (output_dir / "resolved_config.yaml").write_text(resolved_config_str)
        manifest_path = output_dir / cfg.logging.manifest_file
        manifest = manifest_module.create_manifest(
            run_id="run-1",
            wandb_run_id="wb-1",
            preset_id=cfg.get("preset_id"),
            resolved_config_str=resolved_config_str,
            logical_config_sha256=train_module.logical_config_sha256(cfg),
            resolved_config_path=str(output_dir / "resolved_config.yaml"),
        )
        manifest["status"] = "completed"
        manifest_module.save_manifest(manifest, manifest_path)

        with pytest.raises(ValueError, match="completed matching run"):
            train_module._bootstrap_training_run(cfg)

    def test_bootstrap_allows_paused_matching_run(self, tmp_path):
        output_dir = tmp_path / "run"
        cfg = compose_experiment_config(
            preset="iter_run",
            overrides=[
                f"train.output_dir={output_dir}",
                "train.stop_after_generations=20",
                "train.resume_mode=staged",
            ],
        )
        resolved_config_str = OmegaConf.to_yaml(cfg)
        output_dir.mkdir(parents=True)
        manifest = manifest_module.create_manifest(
            run_id="run-1",
            wandb_run_id="wb-1",
            preset_id=cfg.get("preset_id"),
            resolved_config_str=resolved_config_str,
            logical_config_sha256=train_module.logical_config_sha256(cfg),
            resolved_config_path=str(output_dir / "resolved_config.yaml"),
        )
        manifest["status"] = "paused"
        manifest_module.save_manifest(manifest, output_dir / cfg.logging.manifest_file)

        bootstrap = train_module._bootstrap_training_run(cfg)

        assert bootstrap.run_id == "run-1"
        assert bootstrap.resume_mode == "staged"
        assert bootstrap.stop_after_generations == 20

    def test_bootstrap_allows_legacy_hash_mismatch_resume_when_flagged(self, tmp_path):
        output_dir = tmp_path / "run"
        base_cfg = compose_experiment_config(
            preset="iter_run",
            overrides=[f"train.output_dir={output_dir}"],
        )
        resolved_config_str = OmegaConf.to_yaml(base_cfg)
        output_dir.mkdir(parents=True)
        (output_dir / "resolved_config.yaml").write_text(resolved_config_str)
        manifest = manifest_module.create_manifest(
            run_id="run-1",
            wandb_run_id="wb-1",
            preset_id=base_cfg.get("preset_id"),
            resolved_config_str=resolved_config_str,
            logical_config_sha256=train_module.logical_config_sha256(base_cfg),
            resolved_config_path=str(output_dir / "resolved_config.yaml"),
        )
        manifest["status"] = "paused"
        manifest_module.save_manifest(manifest, output_dir / base_cfg.logging.manifest_file)

        checkpoint_dir = output_dir / base_cfg.train.checkpoint_dir
        checkpoint_dir.mkdir()
        checkpoint_prefix = checkpoint_dir / "latest"
        checkpoint_prefix.with_suffix(".json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "sensing_edges_checkpoint",
                    "run_id": "run-1",
                    "generation": 12,
                    "created_at_utc": "2026-01-01T00:00:12+00:00",
                    "config_sha256": train_module.logical_config_sha256(base_cfg),
                }
            )
        )
        checkpoint_prefix.with_suffix(".npz").write_bytes(b"npz")
        payload_done_path(checkpoint_prefix).write_text("{}")

        resumed_cfg = compose_experiment_config(
            preset="iter_run",
            overrides=[
                f"train.output_dir={output_dir}",
                "train.resume_mode=staged",
                "train.allow_output_dir_hash_mismatch_resume=true",
                "eval.post_run.enabled=true",
            ],
        )

        bootstrap = train_module._bootstrap_training_run(resumed_cfg)

        assert bootstrap.run_id == "run-1"
        assert bootstrap.resume_from == str(checkpoint_prefix)
        assert bootstrap.logical_config_sha256 == train_module.logical_config_sha256(base_cfg)

    def test_bootstrap_fresh_mode_refuses_matching_in_progress_run(self, tmp_path):
        output_dir = tmp_path / "run"
        cfg = compose_experiment_config(
            preset="iter_run",
            overrides=[
                f"train.output_dir={output_dir}",
                "train.resume_mode=fresh",
            ],
        )
        resolved_config_str = OmegaConf.to_yaml(cfg)
        output_dir.mkdir(parents=True)
        manifest = manifest_module.create_manifest(
            run_id="run-1",
            wandb_run_id="wb-1",
            preset_id=cfg.get("preset_id"),
            resolved_config_str=resolved_config_str,
            logical_config_sha256=train_module.logical_config_sha256(cfg),
            resolved_config_path=str(output_dir / "resolved_config.yaml"),
        )
        manifest["status"] = "running"
        manifest_module.save_manifest(manifest, output_dir / cfg.logging.manifest_file)

        with pytest.raises(ValueError, match="resume_mode='fresh'"):
            train_module._bootstrap_training_run(cfg)

    def test_branch_bootstrap_creates_child_lineage_from_completed_parent(self, tmp_path):
        parent_output_dir = tmp_path / "parent"
        parent_cfg = compose_experiment_config(
            preset="iter_run",
            overrides=[f"train.output_dir={parent_output_dir}"],
        )
        parent_output_dir.mkdir(parents=True)
        (parent_output_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(parent_cfg))
        parent_manifest = manifest_module.create_manifest(
            run_id="parent-run",
            wandb_run_id="parent-wb",
            preset_id=parent_cfg.get("preset_id"),
            resolved_config_str=OmegaConf.to_yaml(parent_cfg),
            logical_config_sha256=train_module.logical_config_sha256(parent_cfg),
            resolved_config_path=str(parent_output_dir / "resolved_config.yaml"),
        )
        parent_manifest["status"] = "completed"
        parent_manifest["summary"]["completed_generations"] = int(parent_cfg.train.n_generations)
        parent_manifest["terrain_schedule"] = {"train_hash": "train-hash"}
        manifest_module.save_manifest(parent_manifest, parent_output_dir / parent_cfg.logging.manifest_file)
        checkpoint_dir = parent_output_dir / parent_cfg.train.checkpoint_dir
        checkpoint_dir.mkdir()
        checkpoint_prefix = checkpoint_dir / "gen_0029"
        checkpoint_prefix.with_suffix(".json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "sensing_edges_checkpoint",
                    "run_id": "parent-run",
                    "generation": 29,
                    "created_at_utc": "2026-01-01T00:00:29+00:00",
                    "config_sha256": train_module.logical_config_sha256(parent_cfg),
                }
            )
        )
        checkpoint_prefix.with_suffix(".npz").write_bytes(b"npz")
        payload_done_path(checkpoint_prefix).write_text("{}")

        child_output_dir = tmp_path / "child"
        child_cfg = compose_experiment_config(
            preset="iter_run",
            overrides=[
                f"train.output_dir={child_output_dir}",
                f"train.resume_from={checkpoint_prefix}",
                "train.resume_mode=branch",
                "train.n_generations=60",
            ],
        )

        bootstrap = train_module._bootstrap_training_run(child_cfg)

        assert bootstrap.run_id != "parent-run"
        assert bootstrap.resume_mode == "branch"
        assert bootstrap.allow_checkpoint_config_mismatch is True
        assert bootstrap.lineage["mode"] == "branch"
        assert bootstrap.lineage["parent_run_id"] == "parent-run"
        assert bootstrap.lineage["parent_checkpoint_path"] == str(checkpoint_prefix)
        assert bootstrap.lineage["parent_completed_generations"] == int(parent_cfg.train.n_generations)
        assert bootstrap.lineage["parent_schedule_hashes"] == {"train_hash": "train-hash"}

    def test_initialize_artifacts_truncates_metrics_to_resumed_generation(self, tmp_path):
        output_dir = tmp_path / "run"
        cfg = compose_experiment_config(
            preset="iter_run",
            overrides=[f"train.output_dir={output_dir}"],
        )
        resolution = resolve_runtime_config(cfg, preset="iter_run")
        resolved_config_str = OmegaConf.to_yaml(cfg)
        output_dir.mkdir(parents=True)
        (output_dir / "resolved_config.yaml").write_text(resolved_config_str)
        manifest_module.save_manifest(
            manifest_module.create_manifest(
                run_id="run-1",
                wandb_run_id="wb-1",
                preset_id=cfg.get("preset_id"),
                resolved_config_str=resolved_config_str,
                logical_config_sha256=train_module.logical_config_sha256(cfg),
                resolved_config_path=str(output_dir / "resolved_config.yaml"),
            ),
            output_dir / cfg.logging.manifest_file,
        )
        metrics_path = output_dir / cfg.logging.metrics_file
        metrics_path.write_text(
            "\n".join(
                json.dumps({"generation": generation})
                for generation in range(5)
            )
            + "\n"
        )
        checkpoint_dir = output_dir / cfg.train.checkpoint_dir
        checkpoint_dir.mkdir()
        checkpoint_prefix = checkpoint_dir / "gen_0002"
        checkpoint_prefix.with_suffix(".json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "sensing_edges_checkpoint",
                    "run_id": "run-1",
                    "generation": 2,
                    "created_at_utc": "2026-01-01T00:00:02+00:00",
                    "config_sha256": train_module.logical_config_sha256(cfg),
                }
            )
        )
        checkpoint_prefix.with_suffix(".npz").write_bytes(b"npz")
        payload_done_path(checkpoint_prefix).write_text("{}")

        bootstrap = train_module._bootstrap_training_run(cfg)
        artifacts = train_module._initialize_artifacts(
            cfg,
            terrain_schedule_info=None,
            runtime_resolution=resolution,
            bootstrap=bootstrap,
        )

        retained_rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        assert [row["generation"] for row in retained_rows] == [0, 1, 2]
        assert artifacts.manifest["retry"]["resumed_generation"] == 2
        assert artifacts.manifest["wandb"]["run_id"] == "wb-1"
        assert artifacts.manifest["config"]["logical_config_sha256"] == train_module.logical_config_sha256(cfg)
