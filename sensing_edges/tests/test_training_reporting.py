"""Tests for generation reporting and logging metrics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sensing_edges.batch_eval import PopulationBatchSummary
from sensing_edges.fitness import CandidateBatchResult
from sensing_edges.training_reporting import build_generation_report
from sensing_edges.wandb_metrics import build_generation_metrics


def _batch_result() -> CandidateBatchResult:
    return CandidateBatchResult(
        fitness=np.asarray([10.0, 8.0, 6.0], dtype=np.float32),
        fall_count=np.asarray([0.0, 2.0, 0.0], dtype=np.float32),
        goal_hit_count=np.asarray([2.0, 1.0, 0.0], dtype=np.float32),
        cliff_vicinity=np.asarray([0.6, 0.4, 0.2], dtype=np.float32),
        alive_cell_count=np.asarray([14.0, 14.0, 14.0], dtype=np.float32),
        activation_cost=np.asarray([0.5, 0.5, 0.5], dtype=np.float32),
        h_activation=np.asarray([0.25, 0.25, 0.25], dtype=np.float32),
        v_activation=np.asarray([0.25, 0.25, 0.25], dtype=np.float32),
        movement_speed=np.asarray([0.25, 0.25, 0.25], dtype=np.float32),
        peak_progress_rate=np.asarray([1.0, 0.8, 0.6], dtype=np.float32),
        time_to_edge=np.asarray([0.5, 0.5, 0.5], dtype=np.float32),
        correct_progress=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        incorrect_progress=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        correct_progress_top2_mean=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        incorrect_progress_top2_mean=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        toward_goal_frontier=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        turn_count=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        mirrored_pair_count=np.asarray([2.0, 1.0, 0.0], dtype=np.float32),
        episode_fitnesses=np.asarray(
            [
                [10.0, 10.0],
                [8.0, 8.0],
                [6.0, 6.0],
            ],
            dtype=np.float32,
        ),
        episode_activation_costs=np.asarray(
            [
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        ),
        episode_h_activations=np.asarray(
            [
                [0.25, 0.25],
                [0.25, 0.25],
                [0.25, 0.25],
            ],
            dtype=np.float32,
        ),
        episode_v_activations=np.asarray(
            [
                [0.25, 0.25],
                [0.25, 0.25],
                [0.25, 0.25],
            ],
            dtype=np.float32,
        ),
        episode_movement_speeds=np.asarray(
            [
                [0.25, 0.25],
                [0.25, 0.25],
                [0.25, 0.25],
            ],
            dtype=np.float32,
        ),
        episode_peak_progress_rates=np.asarray(
            [
                [1.0, 1.0],
                [0.8, 0.8],
                [0.6, 0.6],
            ],
            dtype=np.float32,
        ),
        episode_times_to_edge=np.asarray(
            [
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        ),
        episode_cliff_vicinity_steps=np.asarray(
            [
                [12, 12],
                [8, 8],
                [4, 4],
            ],
            dtype=np.int32,
        ),
        episode_cliff_vicinity_fractions=np.asarray(
            [
                [0.6, 0.6],
                [0.4, 0.4],
                [0.2, 0.2],
            ],
            dtype=np.float32,
        ),
        episode_init_front_xs=np.asarray(
            [
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        ),
        episode_peak_front_xs=np.asarray(
            [
                [1.0, 1.0],
                [0.9, 0.9],
                [0.8, 0.8],
            ],
            dtype=np.float32,
        ),
        episode_safe_front_xs=np.asarray(
            [
                [0.95, 0.95],
                [0.85, 0.85],
                [0.75, 0.75],
            ],
            dtype=np.float32,
        ),
        episode_probe_extents=np.asarray(
            [
                [0.05, 0.05],
                [0.05, 0.05],
                [0.05, 0.05],
            ],
            dtype=np.float32,
        ),
        episode_cliff_xs=np.asarray(
            [
                [10, 10],
                [10, 10],
                [10, 10],
            ],
            dtype=np.int32,
        ),
        episode_any_progresses=np.asarray(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        episode_goal_progresses=np.asarray(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        episode_incorrect_progresses=np.asarray(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        episode_toward_goal_frontiers=np.asarray(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        episode_goal_touched=np.asarray(
            [
                [True, True],
                [True, False],
                [False, False],
            ],
            dtype=bool,
        ),
        episode_cliff_fell=np.asarray(
            [
                [False, False],
                [True, False],
                [False, False],
            ],
            dtype=bool,
        ),
        episode_goal_directions=np.asarray(
            [
                [0, 0],
                [0, 0],
                [0, 0],
            ],
            dtype=np.int32,
        ),
        episode_goal_distances=np.asarray(
            [
                [0, 0],
                [0, 0],
                [0, 0],
            ],
            dtype=np.int32,
        ),
        episode_cliff_distances=np.asarray(
            [
                [0, 0],
                [0, 0],
                [0, 0],
            ],
            dtype=np.int32,
        ),
        episode_times_to_goal=np.asarray(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        ),
        episode_turn_counts=np.asarray(
            [
                [0, 0],
                [1, 1],
                [2, 2],
            ],
            dtype=np.int32,
        ),
        episode_fell=np.asarray(
            [
                [False, False],
                [True, True],
                [False, False],
            ],
            dtype=bool,
        ),
        episode_airborne=np.asarray(
            [
                [False, False],
                [True, False],
                [False, False],
            ],
            dtype=bool,
        ),
        episode_left_edge=np.asarray(
            [
                [False, False],
                [False, False],
                [True, False],
            ],
            dtype=bool,
        ),
        episode_steps_alive=np.asarray(
            [
                [20, 20],
                [20, 20],
                [20, 20],
            ],
            dtype=np.int32,
        ),
        deformation_channel_max=np.asarray(
            [
                [0.1] * 4,
                [0.2] * 4,
                [0.3] * 4,
            ],
            dtype=np.float32,
        ),
        controller_hidden_output_max_abs=np.asarray([0.2, 0.4, 0.3], dtype=np.float32),
        action_output_max_abs=np.asarray([0.5, 0.7, 0.6], dtype=np.float32),
        lower_ledge_penalty=np.asarray([0.5, 1.5, 2.5], dtype=np.float32),
        stable_turn_count=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        episode_stable_turn_counts=np.asarray(
            [
                [0, 0],
                [0, 1],
                [1, 1],
            ],
            dtype=np.int32,
        ),
        turn_motion_threshold=np.asarray([0.01, 0.012, 0.014], dtype=np.float32),
        episode_turn_motion_thresholds=np.asarray(
            [
                [0.01, 0.01],
                [0.012, 0.012],
                [0.014, 0.014],
            ],
            dtype=np.float32,
        ),
    )


def _bilateral_bucket_batch_result() -> CandidateBatchResult:
    return _batch_result()._replace(
        episode_fitnesses=np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            ],
            dtype=np.float32,
        ),
        episode_goal_touched=np.asarray(
            [
                [True, False, False, True, True, False],
                [False, False, True, True, False, False],
                [True, True, False, False, False, False],
            ],
            dtype=bool,
        ),
        episode_cliff_fell=np.asarray(
            [
                [False, True, False, False, True, False],
                [False, False, True, False, False, True],
                [False, False, False, True, True, True],
            ],
            dtype=bool,
        ),
        episode_fell=np.asarray(
            [
                [False, True, False, False, True, False],
                [False, False, True, False, False, True],
                [False, False, False, True, True, True],
            ],
            dtype=bool,
        ),
        episode_goal_distances=np.asarray(
            [
                [2, 2, 4, 4, 6, 6],
                [2, 2, 4, 4, 6, 6],
                [2, 2, 4, 4, 6, 6],
            ],
            dtype=np.int32,
        ),
        episode_cliff_distances=np.asarray(
            [
                [1, 1, 2, 2, 3, 3],
                [1, 1, 2, 2, 3, 3],
                [1, 1, 2, 2, 3, 3],
            ],
            dtype=np.int32,
        ),
        episode_times_to_goal=np.asarray(
            [
                [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                [0.35, 0.3, 0.25, 0.2, 0.15, 0.1],
                [0.95, 0.85, 0.75, 0.65, 0.55, 0.45],
            ],
            dtype=np.float32,
        ),
        episode_first_significant_directions=np.asarray(
            [
                [1, 1, -1, -1, 1, 0],
                [0, -1, 1, 1, -1, 1],
                [-1, -1, 0, 0, 1, -1],
            ],
            dtype=np.int32,
        ),
    )


def test_generation_report_drives_jsonl_and_wandb_metrics():
    archive = SimpleNamespace(
        best_fitness=12.5,
        qd_score=40.0,
        size=4,
        coverage=0.1,
        occupied_threshold_mean=0.75,
        axis_coverage={"goal_hit_count": 2 / 7, "fall_count": 3 / 7},
        axis_occupied_bins={"goal_hit_count": 2, "fall_count": 3},
    )
    emitters = [
        SimpleNamespace(total_restarts=1, es_state=SimpleNamespace(std=0.25)),
        SimpleNamespace(total_restarts=2, es_state=SimpleNamespace(std=0.5)),
    ]
    report = build_generation_report(
        gen=7,
        archive=archive,
        emitters=emitters,
        batch_result=_batch_result(),
        n_new=2,
        n_improved=1,
        invalid_descriptor_count=2,
        elapsed_s=3.0,
        render_s=0.6,
        wandb_s=0.2,
        qd_focus_metrics={
            "qd_focus_active_cells": 180.0,
            "qd_focus_active_fraction": 180.0 / 980.0,
            "qd_focus_occupied_in_active": 2.0,
            "qd_focus_goal_hit_count_window_size": 3.0,
            "qd_focus_goal_hit_count_lower": 4.0,
            "qd_focus_goal_hit_count_upper": 6.0,
            "qd_focus_tightened_this_gen": 1.0,
        },
    ).with_checkpoint_path("checkpoints/gen_0007")

    row = report.metrics_row()
    wandb_metrics = build_generation_metrics(report)

    assert row["generation"] == 7
    assert row["population"] == 3
    assert row["mean_fall_count"] == pytest.approx(2 / 3, rel=1e-3)
    assert row["goal_hit_mean"] == pytest.approx(1.0)
    assert "null_rate" not in row
    assert row["mean_mirrored_pair_count"] == pytest.approx(1.0)
    assert row["lower_ledge_penalty_mean"] == pytest.approx(1.5)
    assert row["invalid_descriptor_count"] == 2
    assert row["total_restarts"] == 3
    assert row["checkpoint_path"] == "checkpoints/gen_0007"
    assert row["candidates_per_s"] == pytest.approx(1.0)
    assert row["render_s"] == pytest.approx(0.6)
    assert row["wandb_s"] == pytest.approx(0.2)
    assert row["mean_chunk_size"] == pytest.approx(0.0)
    assert row["max_chunk_size"] == 0
    assert row["mean_turn_count"] == pytest.approx(1.0)
    assert row["mean_stable_turn_count"] == pytest.approx(0.5)
    assert row["mean_turn_motion_threshold"] == pytest.approx(0.012, rel=1e-3)
    assert row["max_turn_count"] == pytest.approx(2.0)
    assert row["max_stable_turn_count"] == pytest.approx(1.0)
    assert row["nonzero_turn_fraction"] == pytest.approx(2 / 3, rel=1e-3)
    assert row["nonzero_stable_turn_fraction"] == pytest.approx(2 / 3, rel=1e-3)
    assert row["descriptor_means"]["turn_count"] == pytest.approx(1.0)
    assert row["qd_axis_coverage"]["goal_hit_count"] == pytest.approx(2 / 7)
    assert row["qd_axis_occupied_bins"]["fall_count"] == 3
    assert row["qd_focus_metrics"]["qd_focus_active_cells"] == pytest.approx(180.0)
    assert row["deformation_channel_max"]["abs_h_delta"] == pytest.approx(0.3)
    assert row["controller_hidden_output_max_abs"] == pytest.approx(0.4)
    assert row["action_output_max_abs"] == pytest.approx(0.7)
    assert "padding_ratio" not in row
    assert "terrain_diagnostics" not in row

    assert wandb_metrics["generation"] == 7
    assert wandb_metrics["fitness/best"] == pytest.approx(12.5)
    assert wandb_metrics["fitness/best_gen"] == pytest.approx(10.0)
    assert wandb_metrics["fitness/mean_fall_count"] == pytest.approx(2 / 3)
    assert wandb_metrics["fitness/goal_hit_mean"] == pytest.approx(1.0)
    assert "fitness/null_rate" not in wandb_metrics
    assert wandb_metrics["fitness/mean_mirrored_pair_count"] == pytest.approx(1.0)
    assert wandb_metrics["fitness/lower_ledge_penalty_mean"] == pytest.approx(1.5)
    assert wandb_metrics["descriptors/mean_turn_count"] == pytest.approx(1.0)
    assert wandb_metrics["descriptors/mean_stable_turn_count"] == pytest.approx(0.5)
    assert wandb_metrics["descriptors/mean_turn_motion_threshold"] == pytest.approx(0.012, rel=1e-3)
    assert wandb_metrics["descriptors/max_turn_count"] == pytest.approx(2.0)
    assert wandb_metrics["descriptors/max_stable_turn_count"] == pytest.approx(1.0)
    assert wandb_metrics["descriptors/nonzero_turn_fraction"] == pytest.approx(2 / 3, rel=1e-3)
    assert wandb_metrics["descriptors/nonzero_stable_turn_fraction"] == pytest.approx(2 / 3, rel=1e-3)
    assert wandb_metrics["descriptors/mean_goal_hit_count"] == pytest.approx(1.0)
    assert wandb_metrics["qd/score"] == pytest.approx(40.0)
    assert wandb_metrics["qd/axis_coverage/goal_hit_count"] == pytest.approx(2 / 7)
    assert wandb_metrics["qd/axis_occupied_bins/fall_count"] == pytest.approx(3.0)
    assert wandb_metrics["qd/focus/active_cell_count"] == pytest.approx(180.0)
    assert wandb_metrics["qd/focus/window_size_cells/goal_hit_count"] == pytest.approx(3.0)
    assert wandb_metrics["qd/focus/lower_cell/goal_hit_count"] == pytest.approx(4.0)
    assert wandb_metrics["qd/focus/tightened_this_gen"] == pytest.approx(1.0)
    assert wandb_metrics["qd/invalid_descriptors"] == 2
    assert wandb_metrics["emitters/total_restarts"] == 3
    assert wandb_metrics["emitters/mean_sigma"] == pytest.approx(0.375)
    assert wandb_metrics["emitters/improvement_rate"] == pytest.approx(1.0)
    assert wandb_metrics["timing/render_s"] == pytest.approx(0.6)
    assert wandb_metrics["timing/mean_chunk_size"] == pytest.approx(0.0)
    assert wandb_metrics["timing/max_chunk_size"] == 0
    assert wandb_metrics["deformation/abs_h_delta_max_abs"] == pytest.approx(0.3)
    assert wandb_metrics["alignment/controller_hidden_output_max_abs"] == pytest.approx(0.4)
    assert wandb_metrics["alignment/action_output_max_abs"] == pytest.approx(0.7)
    assert "timing/padding_ratio" not in wandb_metrics
    assert not any(key.startswith("terrain_diag/") for key in wandb_metrics)


def test_generation_report_emits_terrain_bucket_diagnostics():
    archive = SimpleNamespace(
        best_fitness=12.5,
        qd_score=40.0,
        size=4,
        coverage=0.1,
        occupied_threshold_mean=0.75,
    )
    emitters = [SimpleNamespace(total_restarts=0, es_state=SimpleNamespace(std=0.25))]

    report = build_generation_report(
        gen=3,
        archive=archive,
        emitters=emitters,
        batch_result=_bilateral_bucket_batch_result(),
        n_new=1,
        n_improved=0,
        invalid_descriptor_count=0,
        elapsed_s=2.0,
    )
    row = report.metrics_row()
    wandb_metrics = build_generation_metrics(report)

    assert report.terrain_diagnostics is not None
    assert report.terrain_diagnostics["g2_c1/hit_rate"] == pytest.approx(0.5)
    assert report.terrain_diagnostics["g2_c1/fall_rate"] == pytest.approx(1 / 6)
    assert report.terrain_diagnostics["g2_c1/mean_fitness"] == pytest.approx(7.5)
    assert report.terrain_diagnostics["g2_c1/directional_regularity"] == pytest.approx(5 / 6)
    assert report.terrain_diagnostics["g2_c1/best_goal_time"] == pytest.approx(0.85)
    assert report.terrain_diagnostics["g4_c2/hit_rate"] == pytest.approx(0.5)
    assert report.terrain_diagnostics["g4_c2/fall_rate"] == pytest.approx(1 / 3)
    assert report.terrain_diagnostics["g4_c2/mean_fitness"] == pytest.approx(9.5)
    assert report.terrain_diagnostics["g4_c2/directional_regularity"] == pytest.approx(2 / 3)
    assert report.terrain_diagnostics["g4_c2/best_goal_time"] == pytest.approx(0.2)
    assert report.terrain_diagnostics["g6_c3/hit_rate"] == pytest.approx(1 / 6)
    assert report.terrain_diagnostics["g6_c3/fall_rate"] == pytest.approx(2 / 3)
    assert report.terrain_diagnostics["g6_c3/mean_fitness"] == pytest.approx(11.5)
    assert report.terrain_diagnostics["g6_c3/directional_regularity"] == pytest.approx(0.5)
    assert report.terrain_diagnostics["g6_c3/best_goal_time"] == pytest.approx(0.5)
    assert report.terrain_diagnostics["all/hit_rate"] == pytest.approx(7 / 18)
    assert report.terrain_diagnostics["all/fall_rate"] == pytest.approx(7 / 18)
    assert report.terrain_diagnostics["all/mean_fitness"] == pytest.approx(9.5)
    assert report.terrain_diagnostics["all/directional_regularity"] == pytest.approx(0.5)
    assert report.terrain_diagnostics["all/best_goal_time"] == pytest.approx(0.2)
    assert row["terrain_diagnostics"]["g2_c1/hit_rate"] == pytest.approx(0.5)
    assert wandb_metrics["terrain_diag/g2_c1/hit_rate"] == pytest.approx(0.5)
    assert wandb_metrics["terrain_diag/g4_c2/best_goal_time"] == pytest.approx(0.2)
    assert wandb_metrics["terrain_diag/all/directional_regularity"] == pytest.approx(0.5)


def test_generation_report_accepts_population_batch_summary_without_bilateral_episode_arrays():
    archive = SimpleNamespace(
        best_fitness=12.5,
        qd_score=40.0,
        size=4,
        coverage=0.1,
        occupied_threshold_mean=0.75,
    )
    emitters = [SimpleNamespace(total_restarts=0, es_state=SimpleNamespace(std=0.25))]
    batch_summary = PopulationBatchSummary(
        fitness=np.asarray([10.0, 8.0, 6.0], dtype=np.float32),
        goal_hit_count=np.asarray([2.0, 1.0, 0.0], dtype=np.float32),
        mirrored_pair_count=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        episode_fell=np.asarray(
            [
                [False, False],
                [True, True],
                [False, False],
            ],
            dtype=bool,
        ),
        episode_left_edge=np.asarray(
            [
                [False, False],
                [False, False],
                [True, False],
            ],
            dtype=bool,
        ),
        deformation_channel_max=np.asarray(
            [
                [0.1] * 4,
                [0.2] * 4,
                [0.3] * 4,
            ],
            dtype=np.float32,
        ),
        controller_hidden_output_max_abs=np.asarray([0.2, 0.4, 0.3], dtype=np.float32),
        action_output_max_abs=np.asarray([0.5, 0.7, 0.6], dtype=np.float32),
        lower_ledge_penalty=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        descriptor_columns={
            "turn_count": np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
            "stable_turn_count": np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
            "turn_motion_threshold": np.asarray([0.01, 0.012, 0.014], dtype=np.float32),
            "goal_hit_count": np.asarray([2.0, 1.0, 0.0], dtype=np.float32),
        },
    )

    report = build_generation_report(
        gen=7,
        archive=archive,
        emitters=emitters,
        batch_result=batch_summary,
        n_new=2,
        n_improved=1,
        invalid_descriptor_count=0,
        elapsed_s=3.0,
    )

    assert report.mean_fall_count == pytest.approx(2 / 3)
    assert report.goal_hit_mean == pytest.approx(1.0)
    assert report.mean_mirrored_pair_count == pytest.approx(1 / 3, rel=1e-3)
    assert report.lower_ledge_penalty_mean == pytest.approx(1.0)
    assert report.terrain_diagnostics is None
    assert report.descriptor_means["turn_count"] == pytest.approx(1.0)
    assert report.descriptor_means["stable_turn_count"] == pytest.approx(0.5)
    assert report.descriptor_means["turn_motion_threshold"] == pytest.approx(0.012, rel=1e-3)
    assert report.max_turn_count == pytest.approx(2.0)
    assert report.max_stable_turn_count == pytest.approx(1.0)
