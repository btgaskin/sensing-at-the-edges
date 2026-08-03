"""Tests for held-out evaluation (bootstrap CI, elite selection logic)."""

from types import SimpleNamespace

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import sensing_edges.evaluate as evaluate_module
from sensing_edges.batch_eval import GrownEvalAssets
from sensing_edges.evaluate import (
    PhaseContext,
    _phase_cfg_from_schedule,
    _phase_context_from_schedule,
    bootstrap_ci,
    evaluate_archive_with_schedules,
    evaluate_gap_generalization_suite,
    evaluate_symmetric_double_cliff_probe,
    summarize_test_per_terrain,
)
from sensing_edges.runtime_config import compose_experiment_config, genotype_dim_from_cfg, resolve_runtime_config
from sensing_edges.terrain import build_cliff_terrain
from sensing_edges.terrain_schedule import TerrainEntry, TerrainSchedule


class TestBootstrapCI:
    def test_known_distribution(self):
        """Uniform [0, 100] → median ≈ 50, CI should contain 50."""
        values = np.linspace(0, 100, 101)
        lo, hi = bootstrap_ci(values, n_boot=5000)
        assert lo < 50.0 < hi
        assert lo > 30.0  # CI shouldn't be absurdly wide
        assert hi < 70.0

    def test_determinism(self):
        """Same seed → same CI."""
        values = np.random.default_rng(42).normal(50, 10, size=30)
        lo1, hi1 = bootstrap_ci(values, seed=0)
        lo2, hi2 = bootstrap_ci(values, seed=0)
        assert lo1 == lo2
        assert hi1 == hi2

    def test_narrow_distribution(self):
        """Constant values → CI is a point."""
        values = np.full(20, 42.0)
        lo, hi = bootstrap_ci(values)
        assert lo == pytest.approx(42.0)
        assert hi == pytest.approx(42.0)

    def test_single_value(self):
        values = np.array([99.0])
        lo, hi = bootstrap_ci(values)
        assert lo == pytest.approx(99.0)
        assert hi == pytest.approx(99.0)

    def test_wider_alpha(self):
        """90% CI should be narrower than 95%."""
        values = np.random.default_rng(42).normal(50, 10, size=50)
        lo_95, hi_95 = bootstrap_ci(values, alpha=0.05, seed=0)
        lo_90, hi_90 = bootstrap_ci(values, alpha=0.10, seed=0)
        assert (hi_90 - lo_90) <= (hi_95 - lo_95)


def test_phase_context_from_schedule_builds_grown_assets():
    cfg = compose_experiment_config(
        preset="iter_run",
        overrides=["body.mode=grown", "train.max_steps=3", "eval.eval_episodes=1"],
    )
    resolution = resolve_runtime_config(cfg, preset="iter_run")
    entry = TerrainEntry(
        grid=build_cliff_terrain(cfg.terrain.grid_h, cfg.terrain.grid_w, cliff_x=15),
        cliff_x=15,
        gaps=[],
    )
    terrain_cfg = {
        "grid_h": int(cfg.terrain.grid_h),
        "grid_w": int(cfg.terrain.grid_w),
        "platform_y": int(cfg.terrain.platform_y),
        "base_width": int(cfg.terrain.base_width),
        "cliff_x_min": int(cfg.terrain.cliff_x_min),
        "cliff_x_max": int(cfg.terrain.cliff_x_max),
        "gap_widths": [],
        "gap_budget": 0,
        "exact_gap_widths": [],
        "min_gap_separation": int(cfg.terrain.get("min_gap_separation", 2)),
        "min_post_gap_runway": int(cfg.terrain.get("min_post_gap_runway", 2)),
    }
    schedule = TerrainSchedule(
        train=[],
        val=[entry],
        test=[entry],
        seed=0,
        terrain_cfg=terrain_cfg,
        train_hash="train",
        val_hash="val",
        test_hash="test",
    )

    ctx = _phase_context_from_schedule(
        cfg,
        resolution,
        "schedule_dir",
        schedule,
        split="val",
        max_steps=3,
        limit=1,
    )

    assert isinstance(ctx.envs, GrownEvalAssets)
    assert ctx.runtime_resolution.resolved_collision is not None
    assert ctx.envs.runtime_build_capacity == ctx.runtime_resolution.resolved_collision.runtime_build_capacity
    assert ctx.alive_mask.shape == (cfg.body.growth.max_robot_h, cfg.body.growth.max_robot_w)
    assert int(ctx.envs.static_templates.cliff_x.shape[0]) == 1


def test_validate_loaded_schedule_accepts_legacy_alias_variant():
    schedule = TerrainSchedule(
        train=[],
        val=[],
        test=[],
        seed=42,
        terrain_cfg={"variant": "bilateral_goal_cliff_asymmetric_pairs_8x2_30"},
        train_hash="train",
        val_hash="val",
        test_hash="test",
    )

    evaluate_module._validate_loaded_schedule(
        "terrain_suites/fake",
        schedule,
        expected_seed=42,
        expected_variant="bilateral_goal_cliff_asymmetric_pairs_7x2_grid30",
    )


def test_summarize_test_per_terrain_includes_stable_turn_metrics():
    summary = summarize_test_per_terrain(
        [
            {
                "turn_count": 1.0,
                "stable_turn_count": 0.5,
                "turn_motion_threshold": 0.012,
                "fall_count": 1.0,
                "goal_touched": True,
                "entered_cliff_band": False,
                "progress_shaping": 0.0,
                "any_progress": 0.0,
                "goal_progress": 0.0,
                "correct_progress": 0.0,
                "incorrect_progress": 0.0,
                "toward_goal_frontier": 0.0,
                "reversal_progress": 0.0,
                "safe_arrival_bonus": 0.0,
                "safe_dwell_bonus": 0.0,
                "urgency_shaping": 0.0,
                "cliff_vicinity_fraction": 0.0,
                "probe_extent": 0.0,
                "time_to_edge": 0.0,
                "time_to_goal": 1.0,
                "activation_cost": 0.1,
                "movement_speed": 0.2,
            },
            {
                "turn_count": 0.0,
                "stable_turn_count": 0.0,
                "turn_motion_threshold": 0.014,
                "fall_count": 0.0,
                "goal_touched": False,
                "entered_cliff_band": False,
                "progress_shaping": 0.0,
                "any_progress": 0.0,
                "goal_progress": 0.0,
                "correct_progress": 0.0,
                "incorrect_progress": 0.0,
                "toward_goal_frontier": 0.0,
                "reversal_progress": 0.0,
                "safe_arrival_bonus": 0.0,
                "safe_dwell_bonus": 0.0,
                "urgency_shaping": 0.0,
                "cliff_vicinity_fraction": 0.0,
                "probe_extent": 0.0,
                "time_to_edge": 0.0,
                "time_to_goal": 1.0,
                "activation_cost": 0.1,
                "movement_speed": 0.2,
            },
        ]
    )

    assert summary["mean_turn_count"] == pytest.approx(0.5)
    assert summary["mean_stable_turn_count"] == pytest.approx(0.25)
    assert summary["mean_turn_motion_threshold"] == pytest.approx(0.013)


def test_phase_cfg_from_schedule_preserves_zone_single_gap_metadata():
    cfg = compose_experiment_config(preset="iter_run")
    schedule = TerrainSchedule(
        train=[],
        val=[],
        test=[],
        seed=0,
        terrain_cfg={
            "grid_h": 10,
            "grid_w": 30,
            "platform_y": 3,
            "base_width": 10,
            "cliff_x_min": 10,
            "cliff_x_max": 24,
            "gap_widths": [2],
            "gap_budget": 1,
            "exact_gap_widths": [2],
            "min_gap_separation": 2,
            "min_post_gap_runway": 2,
            "hazard_mode": "zone_single_gap",
            "spawn_plateau_width": 10,
            "approach_width": 5,
            "randomized_zone_width": 10,
            "remainder_width": 5,
            "single_gap_width": 2,
            "gap_start_bands": [[10, 14], [15, 19]],
            "layout_sampling": "uniform_legal_layouts",
        },
        train_hash="train",
        val_hash="val",
        test_hash="test",
    )

    phase_cfg = _phase_cfg_from_schedule(
        cfg,
        schedule,
        max_steps=7,
        variant_override="single_gap2_zone10to24_30",
    )

    assert phase_cfg.terrain.variant == "single_gap2_zone10to24_30"
    assert phase_cfg.terrain.grid_w == 30
    assert phase_cfg.terrain.base_width == 10
    assert phase_cfg.terrain.cliff_x_min == 10
    assert phase_cfg.terrain.cliff_x_max == 24
    assert phase_cfg.terrain.hazard_mode == "zone_single_gap"
    assert phase_cfg.terrain.spawn_plateau_width == 10
    assert phase_cfg.terrain.approach_width == 5
    assert phase_cfg.terrain.randomized_zone_width == 10
    assert phase_cfg.terrain.remainder_width == 5
    assert phase_cfg.terrain.single_gap_width == 2
    assert list(phase_cfg.terrain.gap_start_bands) == [[10, 14], [15, 19]]
    assert phase_cfg.terrain.layout_sampling == "uniform_legal_layouts"
    assert list(phase_cfg.terrain.exact_gap_widths) == [2]
    assert phase_cfg.train.max_steps == 7


def test_summarize_test_per_terrain_aggregates_ablation_metrics():
    summary = summarize_test_per_terrain(
        [
            {
                "progress_shaping": 16.0,
                "safe_arrival_bonus": 40.0,
                "safe_dwell_bonus": 18.0,
                "fell": False,
                "entered_cliff_band": True,
                "cliff_vicinity_fraction": 0.6,
                "probe_extent": 0.3,
                "time_to_edge": 0.4,
                "urgency_shaping": 6.0,
                "activation_cost": 0.1,
                "movement_speed": 0.5,
            },
            {
                "progress_shaping": 8.0,
                "safe_arrival_bonus": 0.0,
                "safe_dwell_bonus": 0.0,
                "fell": True,
                "entered_cliff_band": False,
                "cliff_vicinity_fraction": 0.2,
                "probe_extent": 0.1,
                "time_to_edge": 0.8,
                "urgency_shaping": 0.0,
                "activation_cost": 0.3,
                "movement_speed": 0.7,
            },
        ]
    )

    assert summary["mean_progress_shaping"] == pytest.approx(12.0)
    assert summary["mean_safe_arrival_bonus"] == pytest.approx(20.0)
    assert summary["mean_safe_dwell_bonus"] == pytest.approx(9.0)
    assert summary["mean_fall_count"] == pytest.approx(0.5)
    assert summary["edge_entry_rate"] == pytest.approx(0.5)
    assert summary["mean_probe_extent"] == pytest.approx(0.2)
    assert summary["mean_urgency_shaping"] == pytest.approx(3.0)


def test_symmetric_double_cliff_probe_runs_full_distance_sweep(tmp_path):
    cfg = compose_experiment_config(
        preset="iter_run",
        overrides=[
            "terrain.robot_w=6",
            "terrain.robot_h=3",
            "train.max_steps=2",
            "eval.test_max_steps=2",
            "eval.robustness_probe.enabled=true",
            "eval.robustness_probe.max_steps=2",
        ],
    )
    resolution = resolve_runtime_config(
        cfg,
        preset="iter_run",
        runtime_overrides=[
            "terrain.robot_w=6",
            "terrain.robot_h=3",
            "train.max_steps=2",
            "eval.test_max_steps=2",
            "eval.robustness_probe.enabled=true",
            "eval.robustness_probe.max_steps=2",
        ],
    )
    elite = {
        "flat_index": 7,
        "fitness": 0.0,
        "selection_fitness": 0.0,
        "genotype": np.zeros((genotype_dim_from_cfg(cfg),), dtype=np.float32),
        "descriptor": np.zeros((2,), dtype=np.float32),
    }

    results = evaluate_symmetric_double_cliff_probe(
        elite,
        cfg,
        runtime_resolution=resolution,
        output_path=tmp_path / "probe.json",
    )

    assert results.selected_elite_flat_index == 7
    assert results.n_probe_terrains == 6
    assert [item["distance"] for item in results.per_distance] == [1, 2, 3, 4, 5, 6]
    assert results.probe_metadata["spawn_x"] == 12
    assert results.probe_metadata["plateau_width"] == 10
    assert (tmp_path / "probe.json").exists()


def test_post_train_eval_renders_top_five_and_saves_top_five_genotypes(tmp_path, monkeypatch):
    overrides = ["render.render_evals=5"]
    cfg = compose_experiment_config(preset="paper_run", overrides=overrides)
    resolution = resolve_runtime_config(
        cfg,
        preset="paper_run",
        runtime_overrides=overrides,
    )

    fake_schedule = TerrainSchedule(
        train=[],
        val=[],
        test=[],
        seed=137,
        terrain_cfg={"variant": "bilateral_goal_cliff_asymmetric_pairs_7x2_grid30"},
        train_hash="train",
        val_hash="val",
        test_hash="test",
    )

    monkeypatch.setattr(evaluate_module, "load_schedule", lambda _path, **kwargs: fake_schedule)
    monkeypatch.setattr(evaluate_module, "_validate_loaded_schedule", lambda *args, **kwargs: None)

    ctx = PhaseContext(
        cfg=cfg,
        runtime_resolution=resolution,
        envs=[object(), object(), object(), object()],
        alive_mask=jnp.ones((2, 6), dtype=jnp.bool_),
        static_capacity=None,
        entries=[],
        schedule=fake_schedule,
        schedule_dir="terrain_suites/fake",
    )
    monkeypatch.setattr(evaluate_module, "_phase_context_from_schedule", lambda *args, **kwargs: ctx)

    elites = [
        {
            "flat_index": idx,
            "fitness": float(100 - idx),
            "selection_fitness": float(80 - idx),
            "descriptor": np.asarray([idx, idx + 1], dtype=np.float32),
            "genotype": np.asarray([idx, idx + 0.5], dtype=np.float32),
            "archive_rank": idx,
        }
        for idx in range(6)
    ]

    class _FakeArchive:
        size = 6

        def top_elites_by_fitness(self, n):
            return elites[:n]

    monkeypatch.setattr(
        evaluate_module,
        "_score_elites_on_phase",
        lambda elites, _ctx, root_key, workload: list(elites),
    )
    monkeypatch.setattr(
        evaluate_module,
        "_evaluate_selected_elite",
        lambda elite, _ctx, root_key: ([elite["selection_fitness"]] * 4, [{"fitness": elite["selection_fitness"]}] * 4),
    )
    monkeypatch.setattr(evaluate_module, "bootstrap_ci", lambda values, seed=0: (float(np.min(values)), float(np.max(values))))
    monkeypatch.setattr(
        evaluate_module,
        "summarize_test_per_terrain",
        lambda per_terrain: {"mean_fall_count": 0.0, "edge_entry_rate": 0.0, "mean_progress_shaping": 0.0,
                             "mean_safe_arrival_bonus": 0.0, "mean_safe_dwell_bonus": 0.0,
                             "mean_probe_extent": 0.0, "mean_time_to_edge": 0.0,
                             "mean_urgency_shaping": 0.0, "mean_activation_cost": 0.0},
    )
    monkeypatch.setattr(evaluate_module, "_schedule_metadata", lambda *args, **kwargs: {"schedule_dir": "terrain_suites/fake"})

    render_calls: list[tuple[str, int]] = []

    def _fake_render_ranked_elites(elites, env_datas, alive_mask, cfg, *, generation, stage, output_dir, runtime_resolution=None):
        render_calls.append((stage, len(elites)))
        return []

    monkeypatch.setattr(evaluate_module, "render_ranked_elites", _fake_render_ranked_elites)

    results = evaluate_archive_with_schedules(
        _FakeArchive(),
        cfg,
        rerank_schedule_dir="terrain_suites/fake_rerank",
        eval_schedule_dir="terrain_suites/fake_eval",
        runtime_resolution=resolution,
        output_path=tmp_path / "post_train_eval.json",
        render_output_root=tmp_path / "renders",
        generation=99,
    )

    assert render_calls == [("rerank", 5), ("eval", 5)]
    assert len(results.saved_elites) == 5
    assert results.saved_elites[0]["flat_index"] == 0
    assert results.saved_elites[-1]["flat_index"] == 4
    assert results.saved_elites[0]["genotype"] == [0.0, 0.5]


def test_run_deformation_suite_emits_variant_and_per_terrain_summaries(monkeypatch):
    cfg = compose_experiment_config(
        preset="iter_run",
        overrides=[
            "eval.deformation_suite.enabled=true",
            "eval.deformation_suite.elite_count=2",
            "eval.deformation_suite.render=true",
            "eval.deformation_suite.render_top_n=1",
        ],
    )
    resolution = resolve_runtime_config(cfg, preset="iter_run")
    entries = [
        TerrainEntry(grid=build_cliff_terrain(5, 6, cliff_x=2), cliff_x=2, gaps=[]),
        TerrainEntry(grid=build_cliff_terrain(5, 6, cliff_x=3), cliff_x=3, gaps=[]),
    ]
    ctx = PhaseContext(
        cfg=cfg,
        runtime_resolution=resolution,
        envs=[object(), object()],
        alive_mask=jnp.ones((2, 2), dtype=jnp.bool_),
        static_capacity=None,
        entries=entries,
        schedule=None,
        schedule_dir="terrain_suites/fake",
    )
    elites = [
        {
            "flat_index": idx,
            "fitness": float(100 - idx),
            "selection_fitness": float(80 - idx),
            "descriptor": np.asarray([idx, idx + 1], dtype=np.float32),
            "genotype": np.asarray([idx, idx + 0.5], dtype=np.float32),
            "archive_rank": idx,
        }
        for idx in range(2)
    ]

    variant_offsets = {"clean": 0.0, "zero": -10.0, "noisy": -5.0, "shuffled": -7.5}

    def _fake_batch_for_variant(variant: str):
        offset = variant_offsets[variant]
        fitnesses = np.asarray(
            [
                [90.0 + offset, 80.0 + offset],
                [70.0 + offset, 60.0 + offset],
            ],
            dtype=np.float32,
        )
        safe_front = np.asarray(
            [
                [0.025, 0.035],
                [0.015, 0.025],
            ],
            dtype=np.float32,
        )
        return SimpleNamespace(
            episode_fitnesses=fitnesses,
            episode_activation_costs=np.full((2, 2), 0.1, dtype=np.float32),
            episode_h_activations=np.full((2, 2), 0.2, dtype=np.float32),
            episode_v_activations=np.full((2, 2), 0.3, dtype=np.float32),
            episode_movement_speeds=np.full((2, 2), 0.4, dtype=np.float32),
            episode_init_front_xs=np.zeros((2, 2), dtype=np.float32),
            episode_peak_front_xs=safe_front + 0.005,
            episode_safe_front_xs=safe_front,
            episode_cliff_vicinity_fractions=np.asarray(
                [[0.6, 0.5], [0.4, 0.3]],
                dtype=np.float32,
            ),
            episode_cliff_vicinity_steps=np.asarray(
                [[2, 2], [1, 1]],
                dtype=np.int32,
            ),
            episode_fell=np.zeros((2, 2), dtype=bool),
            episode_airborne=np.zeros((2, 2), dtype=bool),
            episode_left_edge=np.zeros((2, 2), dtype=bool),
            episode_times_to_edge=np.asarray(
                [[0.4, 0.3], [0.6, 0.5]],
                dtype=np.float32,
            ),
            episode_probe_extents=np.full((2, 2), 0.01, dtype=np.float32),
        )

    monkeypatch.setattr(
        evaluate_module,
        "_evaluate_elite_batch",
        lambda elites, ctx, root_key, workload: SimpleNamespace(
            batch_result=_fake_batch_for_variant(
                str(ctx.cfg.eval.deformation_suite.active_variant)
            )
        ),
    )
    render_calls: list[tuple[str, int, str]] = []

    def _fake_render_ranked_elites(
        elites,
        env_datas,
        alive_mask,
        cfg,
        *,
        generation,
        stage,
        output_dir,
        runtime_resolution=None,
    ):
        render_calls.append((stage, len(elites), str(output_dir)))
        return []

    monkeypatch.setattr(evaluate_module, "render_ranked_elites", _fake_render_ranked_elites)

    suite = evaluate_module._run_deformation_suite(
        elites,
        cfg,
        ctx,
        root_key=jax.random.PRNGKey(0),
        render_output_root="renders",
        generation=12,
    )

    assert suite is not None
    assert suite["elite_count"] == 2
    assert suite["variants"] == ["clean", "zero", "noisy", "shuffled"]
    assert suite["render_enabled"] is True
    assert suite["render_top_n"] == 1
    assert len(suite["per_elite"]) == 2
    assert set(suite["variant_aggregate_summary"]) == {
        "clean",
        "zero",
        "noisy",
        "shuffled",
    }
    assert len(suite["per_terrain_rows"]) == 16
    assert suite["per_elite"][0]["variants"]["clean"]["test_mean_fitness"] == pytest.approx(85.0)
    assert suite["per_elite"][0]["average_across_variants_summary"]["variant_count"] == 4
    assert len(render_calls) == 4
    assert {stage for stage, _, _ in render_calls} == {
        "deformation_clean",
        "deformation_zero",
        "deformation_noisy",
        "deformation_shuffled",
    }


def test_symmetric_double_cliff_probe_supports_multiple_elites_and_rendering(monkeypatch):
    cfg = compose_experiment_config(
        preset="iter_run",
        overrides=[
            "eval.robustness_probe.enabled=true",
            "eval.robustness_probe.elite_count=2",
            "eval.robustness_probe.render=true",
        ],
    )
    resolution = resolve_runtime_config(cfg, preset="iter_run")
    probe_ctx = PhaseContext(
        cfg=cfg,
        runtime_resolution=resolution,
        envs=[object(), object()],
        alive_mask=jnp.ones((2, 6), dtype=jnp.bool_),
        static_capacity=None,
        entries=[
            TerrainEntry(grid=build_cliff_terrain(5, 6, cliff_x=2), cliff_x=2, gaps=[]),
            TerrainEntry(grid=build_cliff_terrain(5, 6, cliff_x=3), cliff_x=3, gaps=[]),
        ],
        schedule=None,
        schedule_dir="",
    )
    probe_cases = [
        {"distance": 1, "platform_start_x": 4, "platform_end_x": 8},
        {"distance": 2, "platform_start_x": 3, "platform_end_x": 9},
    ]
    monkeypatch.setattr(
        evaluate_module,
        "_symmetric_double_cliff_probe_context",
        lambda cfg, runtime_resolution: (probe_ctx, probe_cases),
    )
    monkeypatch.setattr(evaluate_module, "_warmup_phase", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluate_module, "split_key_sequence", lambda key, count: (key, [key] * count))
    monkeypatch.setattr(
        evaluate_module,
        "evaluate_population_chunked",
        lambda *args, **kwargs: SimpleNamespace(
            batch_result=SimpleNamespace(
                episode_fell=np.asarray([[False, True], [False, False]], dtype=bool),
                episode_airborne=np.zeros((2, 2), dtype=bool),
                episode_left_edge=np.zeros((2, 2), dtype=bool),
                episode_steps_alive=np.asarray([[10, 5], [9, 8]], dtype=np.int32),
                episode_fitnesses=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            )
        ),
    )
    render_calls: list[tuple[int, str]] = []

    def _fake_render_ranked_elites(
        elites,
        env_datas,
        alive_mask,
        cfg,
        *,
        generation,
        stage,
        output_dir,
        runtime_resolution=None,
    ):
        render_calls.append((len(elites), str(output_dir)))
        return []

    monkeypatch.setattr(evaluate_module, "render_ranked_elites", _fake_render_ranked_elites)

    results = evaluate_symmetric_double_cliff_probe(
        [
            {
                "rerank_rank": 0,
                "archive_rank": 0,
                "flat_index": 11,
                "fitness": 100.0,
                "selection_fitness": 90.0,
                "descriptor": np.asarray([0.1, 0.2], dtype=np.float32),
                "genotype": np.asarray([1.0, 2.0], dtype=np.float32),
            },
            {
                "rerank_rank": 1,
                "archive_rank": 1,
                "flat_index": 22,
                "fitness": 95.0,
                "selection_fitness": 80.0,
                "descriptor": np.asarray([0.3, 0.4], dtype=np.float32),
                "genotype": np.asarray([3.0, 4.0], dtype=np.float32),
            },
        ],
        cfg,
        runtime_resolution=resolution,
        render_output_root="renders",
        generation=7,
    )

    assert results.leading_elite_flat_index == 11
    assert results.elite_count == 2
    assert results.aggregate_summary["n_probe_terrains"] == 2
    assert results.aggregate_summary["mean_death_rate"] == pytest.approx(0.25)
    assert len(results.per_elite) == 2
    assert results.per_elite[0]["death_rate"] == pytest.approx(0.5)
    assert results.per_elite[1]["mean_steps_alive"] == pytest.approx(8.5)
    assert render_calls == [(2, "renders/robustness_probe/renders")]


def test_gap_generalization_suite_emits_surface_summaries_and_rows(tmp_path, monkeypatch):
    cfg = compose_experiment_config(
        preset="paper_run",
        overrides=["eval.gap_generalization_suite.render=false"],
    )
    resolution = resolve_runtime_config(
        cfg,
        preset="paper_run",
        runtime_overrides=["eval.gap_generalization_suite.render=false"],
    )
    elite = {
        "flat_index": 3,
        "fitness": 12.0,
        "selection_fitness": 12.0,
        "descriptor": np.asarray([0.1, 0.2], dtype=np.float32),
        "genotype": np.asarray([1.0, 2.0], dtype=np.float32),
    }

    monkeypatch.setattr(
        evaluate_module,
        "_build_phase_assets",
        lambda phase_cfg, runtime_resolution, entries: (
            runtime_resolution,
            [object()] * len(entries),
            jnp.ones((2, 6), dtype=jnp.bool_),
            None,
        ),
    )

    def _fake_evaluate_selected_elite(elite, ctx, root_key):
        surface = str(ctx.entries[0].metadata["gap_generalization_surface"])
        surface_offset = {"gap_w1": 10.0, "gap_w2": 20.0, "gap_w3": 30.0, "bare_cliff": 40.0}[surface]
        rows = []
        fitnesses = []
        for idx, entry in enumerate(ctx.entries):
            fitness = surface_offset + float(idx)
            fitnesses.append(fitness)
            rows.append(
                {
                    "terrain_idx": int(idx),
                    "fitness": fitness,
                    "gaps": [list(gap) for gap in entry.gaps],
                    "correct_progress": float(entry.metadata.get("cliff_front_gap_width", 0)) / 3.0,
                    "incorrect_progress": float(entry.metadata.get("cliff_front_gap_width", 0)) / 10.0,
                    "toward_goal_frontier": float(entry.metadata.get("cliff_front_gap_width", 0)) / 3.0,
                    "goal_touched": surface != "gap_w3",
                    "fell": False,
                    "entered_cliff_band": False,
                    "progress_shaping": 0.0,
                    "safe_arrival_bonus": 0.0,
                    "safe_dwell_bonus": 0.0,
                    "urgency_shaping": 0.0,
                    "cliff_vicinity_fraction": 0.0,
                    "probe_extent": 0.0,
                    "time_to_edge": 0.0,
                    "time_to_goal": 1.0,
                    "turn_count": 0.0,
                    "activation_cost": 0.1,
                    "movement_speed": 0.2,
                    "any_progress": 0.0,
                    "goal_progress": 0.0,
                }
            )
        return np.asarray(fitnesses, dtype=np.float32), rows

    monkeypatch.setattr(evaluate_module, "_evaluate_selected_elite", _fake_evaluate_selected_elite)

    results = evaluate_gap_generalization_suite(
        elite,
        cfg,
        runtime_resolution=resolution,
        output_path=tmp_path / "post_train_gap_generalization_eval.json",
    )

    assert [summary["surface"] for summary in results.surface_summaries] == [
        "gap_w1",
        "gap_w2",
        "gap_w3",
        "bare_cliff",
    ]
    assert all(summary["n_trials"] == 12 for summary in results.surface_summaries)
    assert results.overall_summary["surface_count"] == 4
    assert results.overall_summary["n_trials"] == 48
    assert len(results.per_terrain_rows) == 48
    assert results.surface_summaries[0]["cliff_front_gap_width"] == 1
    assert results.surface_summaries[2]["cliff_front_gap_width"] == 3
    assert results.surface_summaries[3]["cliff_front_gap_width"] == 0
    assert any(row["gaps"] == [[22, 1]] or row["gaps"] == [[7, 1]] for row in results.per_terrain_rows if row["surface"] == "gap_w1")
    assert (tmp_path / "post_train_gap_generalization_eval.json").exists()


def test_evaluate_checkpoint_with_schedules_rejects_config_hash_mismatch(monkeypatch):
    cfg = compose_experiment_config(preset="iter_run")
    runtime_resolution = resolve_runtime_config(cfg, preset="iter_run")
    fake_archive = SimpleNamespace(size=1, best_fitness=1.0)

    monkeypatch.setattr(
        evaluate_module,
        "load_checkpoint",
        lambda *args, **kwargs: {"archive": fake_archive, "meta": {"config_sha256": "mismatch"}},
    )
    monkeypatch.setattr(
        evaluate_module,
        "evaluate_archive_with_schedules",
        lambda *args, **kwargs: pytest.fail("evaluate_archive_with_schedules should not run on mismatch"),
    )

    with pytest.raises(ValueError, match="Checkpoint config hash does not match current logical config"):
        evaluate_module.evaluate_checkpoint_with_schedules(
            checkpoint_path="checkpoint/latest",
            rerank_schedule_dir="terrain_suites/fake_rerank",
            eval_schedule_dir="terrain_suites/fake_eval",
            cfg=cfg,
            runtime_resolution=runtime_resolution,
        )


def test_evaluate_checkpoint_with_schedules_allows_config_hash_mismatch_when_enabled(monkeypatch):
    cfg = compose_experiment_config(
        preset="iter_run",
        overrides=["eval.allow_checkpoint_config_mismatch=true"],
    )
    runtime_resolution = resolve_runtime_config(
        cfg,
        preset="iter_run",
        runtime_overrides=["eval.allow_checkpoint_config_mismatch=true"],
    )
    fake_archive = SimpleNamespace(size=1, best_fitness=1.0)
    expected = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        evaluate_module,
        "load_checkpoint",
        lambda *args, **kwargs: {"archive": fake_archive, "meta": {"config_sha256": "mismatch"}},
    )

    def _fake_evaluate_archive_with_schedules(
        archive,
        cfg,
        *,
        rerank_schedule_dir,
        eval_schedule_dir,
        runtime_resolution,
        output_path,
    ):
        captured["archive"] = archive
        captured["cfg"] = cfg
        captured["rerank_schedule_dir"] = rerank_schedule_dir
        captured["eval_schedule_dir"] = eval_schedule_dir
        captured["runtime_resolution"] = runtime_resolution
        captured["output_path"] = output_path
        return expected

    monkeypatch.setattr(
        evaluate_module,
        "evaluate_archive_with_schedules",
        _fake_evaluate_archive_with_schedules,
    )

    result = evaluate_module.evaluate_checkpoint_with_schedules(
        checkpoint_path="checkpoint/latest",
        rerank_schedule_dir="terrain_suites/fake_rerank",
        eval_schedule_dir="terrain_suites/fake_eval",
        cfg=cfg,
        runtime_resolution=runtime_resolution,
    )

    assert result is expected
    assert captured["archive"] is fake_archive
    assert captured["rerank_schedule_dir"] == "terrain_suites/fake_rerank"
    assert captured["eval_schedule_dir"] == "terrain_suites/fake_eval"
    assert bool(captured["cfg"].eval.allow_checkpoint_config_mismatch) is True
