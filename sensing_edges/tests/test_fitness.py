"""Tests for fitness computation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_evogym.constants import CELL_SIZE

from sensing_edges.env import EpisodeResult
from sensing_edges.fitness import (
    PROGRESS_SHAPING_SCALE,
    SAFE_ARRIVAL_BONUS,
    SAFE_DWELL_BONUS_SCALE,
    URGENCY_SHAPING_SCALE,
    candidate_batch_fitness,
    candidate_fitness,
    episode_descriptors,
    episode_fitness,
    episode_fitness_array,
    episode_peak_progress_rate,
)

_ZERO_DEFORMATION_MAX = jnp.zeros((4,), dtype=jnp.float32)


def _make_result(
    peak_front_x=0.15,
    safe_front_x=None,
    init_front_x=0.08,
    touched_death_plane=False,
    went_airborne=False,
    com_x_went_negative=False,
    mean_activation=0.5,
    mean_h_activation=None,
    mean_v_activation=None,
    mean_speed=0.01,
    cliff_vicinity_steps=0,
    cliff_vicinity_fraction=0.0,
    time_to_edge=1.0,
    elapsed_time_fraction=0.05,
    steps_alive=100,
    cliff_x=15,
    init_com_x=0.08,
    task_mode=0,
    goal_direction=0,
    cliff_side=1,
    goal_distance=0,
    cliff_distance=None,
    goal_block_x=-1,
    goal_block_y=-1,
    goal_x=-1.0,
    any_progress=0.0,
    goal_progress=0.0,
    incorrect_progress=0.0,
    toward_goal_frontier=0.0,
    reversal_progress=0.0,
    goal_touched=False,
    cliff_fell=False,
    time_to_goal=1.0,
    turn_count=0,
    stable_turn_count=0,
    lower_ledge_exposure_steps=0,
    lower_ledge_exposure_fraction=0.0,
    lofted_fraction=0.0,
    mean_lofted_excess_cells=0.0,
    peak_com_y_delta_cells=0.0,
) -> EpisodeResult:
    if safe_front_x is None:
        safe_front_x = peak_front_x
    if mean_h_activation is None:
        mean_h_activation = mean_activation / 2.0
    if mean_v_activation is None:
        mean_v_activation = mean_activation / 2.0
    if cliff_distance is None:
        cliff_distance = goal_distance
    return EpisodeResult(
        peak_front_x=jnp.float32(peak_front_x),
        safe_front_x=jnp.float32(safe_front_x),
        init_front_x=jnp.float32(init_front_x),
        init_com_x=jnp.float32(init_com_x),
        touched_death_plane=jnp.bool_(touched_death_plane),
        went_airborne=jnp.bool_(went_airborne),
        com_x_went_negative=jnp.bool_(com_x_went_negative),
        mean_activation=jnp.float32(mean_activation),
        mean_h_activation=jnp.float32(mean_h_activation),
        mean_v_activation=jnp.float32(mean_v_activation),
        mean_speed=jnp.float32(mean_speed),
        cliff_vicinity_steps=jnp.int32(cliff_vicinity_steps),
        cliff_vicinity_fraction=jnp.float32(cliff_vicinity_fraction),
        time_to_edge=jnp.float32(time_to_edge),
        elapsed_time_fraction=jnp.float32(elapsed_time_fraction),
        steps_alive=jnp.int32(steps_alive),
        cliff_x=jnp.int32(cliff_x),
        task_mode=jnp.int32(task_mode),
        goal_direction=jnp.int32(goal_direction),
        cliff_side=jnp.int32(cliff_side),
        goal_distance=jnp.int32(goal_distance),
        cliff_distance=jnp.int32(cliff_distance),
        goal_block_x=jnp.int32(goal_block_x),
        goal_block_y=jnp.int32(goal_block_y),
        goal_x=jnp.float32(goal_x),
        any_progress=jnp.float32(any_progress),
        goal_progress=jnp.float32(goal_progress),
        incorrect_progress=jnp.float32(incorrect_progress),
        toward_goal_frontier=jnp.float32(toward_goal_frontier),
        goal_touched=jnp.bool_(goal_touched),
        cliff_fell=jnp.bool_(cliff_fell),
        time_to_goal=jnp.float32(time_to_goal),
        turn_count=jnp.int32(turn_count),
        deformation_channel_max=_ZERO_DEFORMATION_MAX,
        controller_hidden_output_max_abs=jnp.float32(0.0),
        action_output_max_abs=jnp.float32(0.0),
        lower_ledge_exposure_steps=jnp.int32(lower_ledge_exposure_steps),
        lower_ledge_exposure_fraction=jnp.float32(lower_ledge_exposure_fraction),
        reversal_progress=jnp.float32(reversal_progress),
        stable_turn_count=jnp.int32(stable_turn_count),
        lofted_fraction=jnp.float32(lofted_fraction),
        mean_lofted_excess_cells=jnp.float32(mean_lofted_excess_cells),
        peak_com_y_delta_cells=jnp.float32(peak_com_y_delta_cells),
    )


class TestEpisodeFitness:
    def test_bilateral_goal_cliff_reward_matches_goal_touch_equation_without_time_bonus(self):
        r = _make_result(
            task_mode=1,
            incorrect_progress=0.05,
            toward_goal_frontier=0.25,
            goal_touched=True,
            cliff_fell=False,
            time_to_goal=0.25,
        )

        assert episode_fitness(r) == pytest.approx(48.0)

    def test_bilateral_goal_cliff_fall_penalty_is_softened_to_minus_20(self):
        r = _make_result(
            task_mode=1,
            incorrect_progress=0.1,
            toward_goal_frontier=0.25,
            goal_touched=False,
            cliff_fell=True,
            time_to_goal=0.0,
        )

        assert episode_fitness(r) == pytest.approx(-11.5)

    def test_bilateral_lower_ledge_penalty_ramps_with_exposure_fraction(self):
        r = _make_result(
            task_mode=1,
            incorrect_progress=0.1,
            toward_goal_frontier=0.25,
            lower_ledge_exposure_fraction=0.5,
        )

        assert episode_fitness(r) == pytest.approx(3.5)

    def test_bilateral_lower_ledge_penalty_caps_at_half_fall_penalty(self):
        r = _make_result(
            task_mode=1,
            incorrect_progress=0.1,
            toward_goal_frontier=0.25,
            lower_ledge_exposure_fraction=2.0,
        )

        assert episode_fitness(r) == pytest.approx(-1.5)

    def test_bilateral_wrong_frontier_bonus_scales_across_full_normalized_range(self):
        small = _make_result(task_mode=1, incorrect_progress=0.05, toward_goal_frontier=0.0)
        large = _make_result(task_mode=1, incorrect_progress=0.3, toward_goal_frontier=0.0)

        assert episode_fitness(small) == pytest.approx(0.5)
        assert episode_fitness(large) == pytest.approx(3.0)

    def test_bilateral_reversal_progress_adds_midcourse_return_reward(self):
        r = _make_result(
            task_mode=1,
            incorrect_progress=0.4,
            reversal_progress=0.5,
            toward_goal_frontier=0.2,
        )

        assert episode_fitness(r) == pytest.approx(10.0)

    def test_bilateral_reversal_progress_is_fully_enabled_near_frontier(self):
        r = _make_result(
            task_mode=1,
            incorrect_progress=0.8,
            reversal_progress=0.5,
            toward_goal_frontier=0.2,
        )

        assert episode_fitness(r) == pytest.approx(24.0)

    def test_bilateral_reversal_progress_softly_ramps_before_frontier(self):
        r = _make_result(
            task_mode=1,
            incorrect_progress=0.6,
            reversal_progress=0.5,
            toward_goal_frontier=0.2,
        )

        assert episode_fitness(r) == pytest.approx(17.0)

    def test_bilateral_mean_activation_no_longer_changes_scalar_fitness(self):
        quiet = _make_result(
            task_mode=1,
            incorrect_progress=0.1,
            toward_goal_frontier=0.25,
            mean_activation=0.1,
        )
        loud = _make_result(
            task_mode=1,
            incorrect_progress=0.1,
            toward_goal_frontier=0.25,
            mean_activation=1.0,
        )

        assert episode_fitness(quiet) == pytest.approx(episode_fitness(loud))

    def test_bilateral_peak_progress_rate_uses_toward_goal_frontier(self):
        r = _make_result(
            task_mode=1,
            toward_goal_frontier=0.5,
            goal_progress=0.0,
            elapsed_time_fraction=0.25,
        )

        assert episode_peak_progress_rate(r) == pytest.approx(2.0)

    def test_reached_cliff_edge(self):
        r = _make_result(peak_front_x=1.6, init_front_x=0.8, cliff_x=15)
        f = episode_fitness(r)
        assert f == pytest.approx(20.0, abs=0.1)

    def test_half_route(self):
        r = _make_result(peak_front_x=1.2, safe_front_x=1.2, init_front_x=0.8, cliff_x=15)
        f = episode_fitness(r)
        assert f == pytest.approx(10.0, abs=0.1)

    def test_progress_clips_at_100_even_if_safe_front_passes_edge(self):
        r = _make_result(peak_front_x=2.0, safe_front_x=2.0, init_front_x=0.8, cliff_x=15)
        f = episode_fitness(r)
        assert f == pytest.approx(20.0, abs=0.1)

    def test_safe_edge_arrival_gets_arrival_dwell_and_urgency_terms(self):
        r = _make_result(
            peak_front_x=1.6,
            safe_front_x=1.6,
            init_front_x=0.8,
            cliff_x=15,
            cliff_vicinity_steps=25,
            cliff_vicinity_fraction=0.25,
            time_to_edge=0.5,
        )
        f = episode_fitness(r)
        assert f == pytest.approx(72.5, abs=0.1)

    def test_fall_keeps_only_progress_shaping(self):
        r = _make_result(
            peak_front_x=1.6,
            safe_front_x=1.6,
            init_front_x=0.8,
            cliff_x=15,
            touched_death_plane=True,
            cliff_vicinity_steps=25,
            cliff_vicinity_fraction=0.25,
        )
        f = episode_fitness(r)
        assert f == pytest.approx(20.0, abs=0.1)

    def test_progress_shaping_uses_safe_front_even_if_peak_front_lags(self):
        r = _make_result(
            peak_front_x=2.0,
            safe_front_x=1.2,
            init_front_x=0.8,
            cliff_x=15,
            touched_death_plane=True,
        )
        f = episode_fitness(r)
        assert f == pytest.approx(10.0, abs=0.1)

    def test_airborne_death_scores_zero(self):
        r = _make_result(
            peak_front_x=1.6,
            safe_front_x=1.6,
            init_front_x=0.8,
            cliff_x=15,
            touched_death_plane=True,
            went_airborne=True,
            cliff_vicinity_steps=10,
            cliff_vicinity_fraction=0.10,
        )
        f = episode_fitness(r)
        assert f == pytest.approx(0.0)

    def test_left_edge_scores_minus_20(self):
        r = _make_result(peak_front_x=1.0, init_front_x=0.8, cliff_x=15,
                         com_x_went_negative=True)
        f = episode_fitness(r)
        assert f == pytest.approx(-20.0)

    def test_no_movement(self):
        r = _make_result(peak_front_x=0.8, init_front_x=0.8, cliff_x=15)
        f = episode_fitness(r)
        assert f == pytest.approx(0.0, abs=0.1)

    def test_non_edge_survivor_gets_only_progress_shaping(self):
        r = _make_result(
            peak_front_x=1.2,
            init_front_x=0.8,
            cliff_x=15,
            cliff_vicinity_steps=0,
            cliff_vicinity_fraction=0.25,
        )
        f = episode_fitness(r)
        assert f == pytest.approx(10.0, abs=0.1)

    def test_safe_edge_arrival_beats_non_arrival_with_similar_progress(self):
        safe_arrival = _make_result(
            peak_front_x=1.6,
            safe_front_x=1.4,
            init_front_x=0.8,
            cliff_x=15,
            cliff_vicinity_steps=10,
            cliff_vicinity_fraction=0.10,
            time_to_edge=0.5,
        )
        non_arrival = _make_result(
            peak_front_x=1.4,
            safe_front_x=1.4,
            init_front_x=0.8,
            cliff_x=15,
            cliff_vicinity_steps=0,
            cliff_vicinity_fraction=0.0,
            time_to_edge=1.0,
        )

        assert episode_fitness(safe_arrival) > episode_fitness(non_arrival)

    def test_earlier_safe_arrival_gets_more_urgency_shaping(self):
        early = _make_result(
            peak_front_x=1.6,
            safe_front_x=1.6,
            init_front_x=0.8,
            cliff_x=15,
            cliff_vicinity_steps=25,
            cliff_vicinity_fraction=0.25,
            time_to_edge=0.25,
        )
        late = _make_result(
            peak_front_x=1.6,
            safe_front_x=1.6,
            init_front_x=0.8,
            cliff_x=15,
            cliff_vicinity_steps=25,
            cliff_vicinity_fraction=0.25,
            time_to_edge=0.75,
        )

        early_fitness = episode_fitness(early)
        late_fitness = episode_fitness(late)

        assert early_fitness == pytest.approx(75.0, abs=0.1)
        assert late_fitness == pytest.approx(70.0, abs=0.1)
        assert early_fitness > late_fitness

    def test_safe_arrival_at_time_to_edge_one_gets_no_urgency_shaping(self):
        r = _make_result(
            peak_front_x=1.6,
            safe_front_x=1.6,
            init_front_x=0.8,
            cliff_x=15,
            cliff_vicinity_steps=25,
            cliff_vicinity_fraction=0.25,
            time_to_edge=1.0,
        )
        f = episode_fitness(r)
        assert f == pytest.approx(67.5, abs=0.1)

    def test_dwell_uses_fraction_not_steps_alive(self):
        r = _make_result(
            peak_front_x=1.2,
            safe_front_x=1.2,
            init_front_x=0.8,
            cliff_x=15,
            touched_death_plane=True,
            cliff_vicinity_steps=25,
            cliff_vicinity_fraction=0.25,
            steps_alive=40,
        )
        f = episode_fitness(r)
        assert f == pytest.approx(10.0, abs=0.1)

    def test_fall_after_arrival_gets_no_edge_reward(self):
        r = _make_result(
            peak_front_x=1.6,
            safe_front_x=1.6,
            init_front_x=0.8,
            cliff_x=15,
            touched_death_plane=True,
            cliff_vicinity_steps=25,
            cliff_vicinity_fraction=0.25,
            time_to_edge=0.25,
        )
        f = episode_fitness(r)
        assert f == pytest.approx(20.0, abs=0.1)

    def test_left_edge_vectorized_matches_scalar(self):
        result = _make_result(
            peak_front_x=1.4,
            safe_front_x=1.1,
            init_front_x=0.8,
            cliff_x=15,
            touched_death_plane=True,
            com_x_went_negative=True,
        )
        scalar = episode_fitness(result)
        vectorized = float(episode_fitness_array(result))
        assert scalar == pytest.approx(-20.0)
        assert vectorized == pytest.approx(scalar)

    def test_airborne_without_death_scores_zero(self):
        r = _make_result(
            peak_front_x=1.2,
            init_front_x=0.8,
            cliff_x=15,
            went_airborne=True,
            touched_death_plane=False,
            cliff_vicinity_steps=1,
            cliff_vicinity_fraction=0.01,
        )
        f = episode_fitness(r)
        assert f == pytest.approx(0.0)

    def test_airborne_overrides_left_edge_penalty(self):
        result = _make_result(
            peak_front_x=1.4,
            safe_front_x=1.1,
            init_front_x=0.8,
            cliff_x=15,
            went_airborne=True,
            com_x_went_negative=True,
        )
        scalar = episode_fitness(result)
        vectorized = float(episode_fitness_array(result))
        assert scalar == pytest.approx(0.0)
        assert vectorized == pytest.approx(0.0)

    def test_vectorized_matches_scalar(self):
        result = _make_result(
            peak_front_x=1.6,
            safe_front_x=1.4,
            init_front_x=0.8,
            cliff_x=15,
            touched_death_plane=False,
            cliff_vicinity_steps=5,
            cliff_vicinity_fraction=0.05,
            time_to_edge=0.5,
        )
        scalar = episode_fitness(result)
        vectorized = float(episode_fitness_array(result))
        assert vectorized == pytest.approx(scalar)

    def test_bilateral_lower_ledge_penalty_vectorized_matches_scalar(self):
        result = _make_result(
            task_mode=1,
            incorrect_progress=0.1,
            toward_goal_frontier=0.25,
            lower_ledge_exposure_fraction=0.5,
        )

        scalar = episode_fitness(result)
        vectorized = float(episode_fitness_array(result))
        assert vectorized == pytest.approx(scalar)

    def test_bilateral_left_edge_vectorized_matches_scalar(self):
        result = _make_result(
            task_mode=1,
            incorrect_progress=0.4,
            toward_goal_frontier=0.3,
            com_x_went_negative=True,
            goal_touched=True,
            time_to_goal=0.25,
        )

        scalar = episode_fitness(result)
        vectorized = float(episode_fitness_array(result))
        assert scalar == pytest.approx(-20.0)
        assert vectorized == pytest.approx(scalar)

    def test_fall_vectorized_matches_scalar_without_multiplier(self):
        result = _make_result(
            peak_front_x=1.6,
            safe_front_x=1.6,
            init_front_x=0.8,
            cliff_x=15,
            touched_death_plane=True,
            cliff_vicinity_steps=25,
            cliff_vicinity_fraction=0.25,
        )
        scalar = episode_fitness(result)
        vectorized = float(episode_fitness_array(result))
        assert scalar == pytest.approx(20.0, abs=0.1)
        assert vectorized == pytest.approx(scalar)

    def test_urgency_shaping_scales_with_time_to_edge(self):
        result = _make_result(
            peak_front_x=1.6,
            safe_front_x=1.6,
            init_front_x=0.8,
            cliff_x=15,
            cliff_vicinity_steps=25,
            cliff_vicinity_fraction=0.25,
            time_to_edge=0.4,
        )
        baseline = PROGRESS_SHAPING_SCALE + SAFE_ARRIVAL_BONUS + (SAFE_DWELL_BONUS_SCALE * 0.25)
        expected = baseline + (URGENCY_SHAPING_SCALE * 0.6)
        assert episode_fitness(result) == pytest.approx(expected, abs=0.1)


class TestEpisodeDescriptors:
    def test_activation_cost_descriptor(self):
        r = _make_result(mean_activation=0.75)
        activation_cost, _ = episode_descriptors(r)
        assert activation_cost == pytest.approx(0.75)

    def test_movement_speed_descriptor(self):
        r = _make_result(mean_speed=0.02)
        _, movement_speed = episode_descriptors(r)
        assert movement_speed == pytest.approx(0.02 / CELL_SIZE)

    def test_time_to_edge_defaults_to_one_when_band_never_reached(self):
        r = _make_result(time_to_edge=1.0)
        assert float(r.time_to_edge) == pytest.approx(1.0)

    def test_peak_progress_rate_uses_clipped_peak_progress_over_elapsed_budget(self):
        r = _make_result(
            peak_front_x=1.2,
            init_front_x=0.8,
            cliff_x=15,
            elapsed_time_fraction=0.5,
        )
        assert episode_peak_progress_rate(r) == pytest.approx(1.0)

    def test_peak_progress_rate_clips_overshoot_at_edge(self):
        r = _make_result(
            peak_front_x=2.0,
            init_front_x=0.8,
            cliff_x=15,
            elapsed_time_fraction=0.5,
        )
        assert episode_peak_progress_rate(r) == pytest.approx(2.0)

    def test_peak_progress_rate_caps_fast_outliers(self):
        r = _make_result(
            peak_front_x=1.6,
            init_front_x=0.8,
            cliff_x=15,
            elapsed_time_fraction=0.1,
        )
        assert episode_peak_progress_rate(r) == pytest.approx(2.0)


class TestCandidateFitness:
    def test_mean_across_episodes(self):
        r1 = _make_result(peak_front_x=1.6, init_front_x=0.8, cliff_x=15)
        r2 = _make_result(peak_front_x=1.2, init_front_x=0.8, cliff_x=15)
        c = candidate_fitness([r1, r2])
        assert c.fitness == pytest.approx(15.0, abs=0.1)
        assert c.n_episodes == 2
        assert len(c.episode_fitnesses) == 2

    def test_descriptors_averaged(self):
        r1 = _make_result(
            mean_activation=0.5,
            mean_h_activation=0.3,
            mean_v_activation=0.2,
            peak_front_x=1.2,
            safe_front_x=1.2,
            init_front_x=0.8,
            cliff_x=15,
            mean_speed=0.01,
            elapsed_time_fraction=0.5,
        )
        r2 = _make_result(
            mean_activation=1.0,
            mean_h_activation=0.8,
            mean_v_activation=0.2,
            peak_front_x=1.2,
            safe_front_x=1.0,
            init_front_x=0.8,
            cliff_x=15,
            mean_speed=0.03,
            elapsed_time_fraction=0.5,
        )
        c = candidate_fitness([r1, r2])
        assert c.fall_count == pytest.approx(0.0)
        assert c.cliff_vicinity == pytest.approx(0.0)
        assert c.activation_cost == pytest.approx(0.75)
        assert c.h_activation == pytest.approx(0.55)
        assert c.v_activation == pytest.approx(0.2)
        assert c.movement_speed == pytest.approx((0.01 / CELL_SIZE + 0.03 / CELL_SIZE) / 2.0)
        assert c.peak_progress_rate == pytest.approx((1.0 + 1.0) / 2.0)
        assert c.time_to_edge == pytest.approx(1.0)

    def test_bilateral_candidate_metrics_average_correct_incorrect_progress_and_turns(self):
        r1 = _make_result(
            task_mode=1,
            goal_progress=0.5,
            incorrect_progress=0.25,
            toward_goal_frontier=0.625,
            turn_count=1,
            stable_turn_count=0,
            time_to_goal=0.5,
        )
        r2 = _make_result(
            task_mode=1,
            goal_progress=0.25,
            incorrect_progress=0.5,
            toward_goal_frontier=0.125,
            turn_count=3,
            stable_turn_count=1,
            time_to_goal=0.75,
        )

        c = candidate_fitness([r1, r2])

        assert c.correct_progress == pytest.approx(0.375)
        assert c.incorrect_progress == pytest.approx(0.375)
        assert c.toward_goal_frontier == pytest.approx(0.375)
        assert c.reversal_progress == pytest.approx(0.0)
        assert c.turn_count == pytest.approx(2.0)
        assert c.stable_turn_count == pytest.approx(0.5)

    def test_bilateral_candidate_metrics_expose_top_two_progress_means(self):
        r1 = _make_result(task_mode=1, incorrect_progress=0.2, toward_goal_frontier=0.1)
        r2 = _make_result(task_mode=1, incorrect_progress=0.5, toward_goal_frontier=0.6)
        r3 = _make_result(task_mode=1, incorrect_progress=0.4, toward_goal_frontier=0.3)

        c = candidate_fitness([r1, r2, r3])

        assert c.correct_progress_top2_mean == pytest.approx((0.6 + 0.3) / 2.0)
        assert c.incorrect_progress_top2_mean == pytest.approx((0.5 + 0.4) / 2.0)

    def test_bilateral_candidate_exposes_mirrored_roundtrip_top2_mean(self):
        episodes = [
            _make_result(task_mode=1, goal_direction=-1, goal_distance=2, cliff_distance=1, incorrect_progress=0.8, toward_goal_frontier=0.6),
            _make_result(task_mode=1, goal_direction=1, goal_distance=2, cliff_distance=1, incorrect_progress=0.7, toward_goal_frontier=0.9),
            _make_result(task_mode=1, goal_direction=-1, goal_distance=4, cliff_distance=2, incorrect_progress=0.4, toward_goal_frontier=0.5),
            _make_result(task_mode=1, goal_direction=1, goal_distance=4, cliff_distance=2, incorrect_progress=0.3, toward_goal_frontier=0.9),
            _make_result(task_mode=1, goal_direction=-1, goal_distance=6, cliff_distance=3, incorrect_progress=0.9, toward_goal_frontier=0.2),
            _make_result(task_mode=1, goal_direction=1, goal_distance=6, cliff_distance=3, incorrect_progress=0.8, toward_goal_frontier=0.4),
        ]

        c = candidate_fitness(episodes)

        assert c.mirrored_roundtrip_top2_mean == pytest.approx((0.6 + 0.3) / 2.0)

    def test_bilateral_candidate_counts_balanced_left_right_hits(self):
        left_d1 = _make_result(
            task_mode=1,
            goal_direction=-1,
            goal_distance=2,
            cliff_distance=1,
            goal_touched=True,
        )
        right_d2 = _make_result(
            task_mode=1,
            goal_direction=1,
            goal_distance=4,
            cliff_distance=2,
            goal_touched=True,
        )
        left_d2 = _make_result(
            task_mode=1,
            goal_direction=-1,
            goal_distance=4,
            cliff_distance=2,
            goal_touched=True,
        )
        right_d3 = _make_result(
            task_mode=1,
            goal_direction=1,
            goal_distance=6,
            cliff_distance=3,
            goal_touched=False,
        )

        c = candidate_fitness([left_d1, right_d2, left_d2, right_d3])

        assert c.mirrored_pair_count == pytest.approx(1.0)
        assert c.mirrored_best_hit_count == pytest.approx(2.0)
        assert c.mirrored_total_hit_count == pytest.approx(3.0)
        assert c.fitness == pytest.approx(50.0)

    def test_candidate_level_activation_penalty_no_longer_changes_scalar_fitness(self):
        r1 = _make_result(peak_front_x=1.6, init_front_x=0.8, cliff_x=15, mean_activation=0.5)
        r2 = _make_result(peak_front_x=1.2, init_front_x=0.8, cliff_x=15, mean_activation=1.0)

        c = candidate_fitness([r1, r2], activation_cost_penalty_scale=2.0)

        assert c.activation_cost == pytest.approx(0.75)
        assert c.fitness == pytest.approx(15.0, abs=0.1)

    def test_gait_spectral_purity_bonus_applies_only_when_explicitly_passed(self):
        r1 = _make_result(peak_front_x=1.6, init_front_x=0.8, cliff_x=15)
        r2 = _make_result(peak_front_x=1.2, init_front_x=0.8, cliff_x=15)

        c = candidate_fitness(
            [r1, r2],
            gait_spectral_purity=0.8,
            gait_spectral_purity_bonus_scale=5.0,
        )

        assert c.gait_spectral_purity == pytest.approx(0.8)
        assert c.fitness == pytest.approx(19.0, abs=0.1)

    def test_soft_loft_penalty_reduces_scalar_fitness(self):
        r1 = _make_result(
            peak_front_x=1.6,
            init_front_x=0.8,
            cliff_x=15,
            mean_lofted_excess_cells=0.1,
        )
        r2 = _make_result(
            peak_front_x=1.2,
            init_front_x=0.8,
            cliff_x=15,
            mean_lofted_excess_cells=0.2,
        )

        c = candidate_fitness(
            [r1, r2],
            soft_loft_penalty_scale=8.0,
        )

        assert c.mean_lofted_excess_cells == pytest.approx(0.15)
        assert c.fitness == pytest.approx(13.8, abs=0.1)


class TestCandidateBatchFitness:
    def test_batch_bilateral_mirrored_pair_count_tracks_balanced_side_hits(self):
        result = EpisodeResult(
            peak_front_x=jnp.asarray([[0.8, 0.8, 0.8, 0.8]], dtype=jnp.float32),
            safe_front_x=jnp.asarray([[0.8, 0.8, 0.8, 0.8]], dtype=jnp.float32),
            init_front_x=jnp.asarray([[0.8, 0.8, 0.8, 0.8]], dtype=jnp.float32),
            init_com_x=jnp.asarray([[0.8, 0.8, 0.8, 0.8]], dtype=jnp.float32),
            touched_death_plane=jnp.asarray([[False, False, False, False]], dtype=jnp.bool_),
            went_airborne=jnp.asarray([[False, False, False, False]], dtype=jnp.bool_),
            com_x_went_negative=jnp.asarray([[False, False, False, False]], dtype=jnp.bool_),
            mean_activation=jnp.asarray([[0.5, 0.5, 0.5, 0.5]], dtype=jnp.float32),
            mean_h_activation=jnp.asarray([[0.25, 0.25, 0.25, 0.25]], dtype=jnp.float32),
            mean_v_activation=jnp.asarray([[0.25, 0.25, 0.25, 0.25]], dtype=jnp.float32),
            mean_speed=jnp.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32),
            cliff_vicinity_steps=jnp.asarray([[0, 0, 0, 0]], dtype=jnp.int32),
            cliff_vicinity_fraction=jnp.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32),
            time_to_edge=jnp.asarray([[1.0, 1.0, 1.0, 1.0]], dtype=jnp.float32),
            elapsed_time_fraction=jnp.asarray([[1.0, 1.0, 1.0, 1.0]], dtype=jnp.float32),
            steps_alive=jnp.asarray([[100, 100, 100, 100]], dtype=jnp.int32),
            cliff_x=jnp.asarray([[15, 15, 15, 15]], dtype=jnp.int32),
            task_mode=jnp.asarray([[1, 1, 1, 1]], dtype=jnp.int32),
            goal_direction=jnp.asarray([[-1, 1, -1, 1]], dtype=jnp.int32),
            cliff_side=jnp.asarray([[1, -1, 1, -1]], dtype=jnp.int32),
            goal_distance=jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32),
            cliff_distance=jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32),
            goal_block_x=jnp.asarray([[0, 0, 0, 0]], dtype=jnp.int32),
            goal_block_y=jnp.asarray([[0, 0, 0, 0]], dtype=jnp.int32),
            goal_x=jnp.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32),
            any_progress=jnp.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32),
            goal_progress=jnp.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32),
            incorrect_progress=jnp.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32),
            toward_goal_frontier=jnp.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32),
            goal_touched=jnp.asarray([[True, True, True, False]], dtype=jnp.bool_),
            cliff_fell=jnp.asarray([[False, False, False, False]], dtype=jnp.bool_),
            time_to_goal=jnp.asarray([[1.0, 1.0, 1.0, 1.0]], dtype=jnp.float32),
            turn_count=jnp.asarray([[0, 0, 0, 0]], dtype=jnp.int32),
            deformation_channel_max=jnp.zeros((1, 4, 4), dtype=jnp.float32),
            controller_hidden_output_max_abs=jnp.zeros((1, 4), dtype=jnp.float32),
            action_output_max_abs=jnp.zeros((1, 4), dtype=jnp.float32),
        )

        batch = candidate_batch_fitness(result)

        assert batch.mirrored_pair_count[0] == pytest.approx(1.0)
        assert batch.mirrored_best_hit_count[0] == pytest.approx(2.0)
        assert batch.mirrored_total_hit_count[0] == pytest.approx(3.0)
        assert batch.fitness[0] == pytest.approx(50.0)

    def test_batch_metrics_include_safe_peak_and_probe_arrays(self):
        result = EpisodeResult(
            peak_front_x=jnp.asarray([[1.6, 2.0]], dtype=jnp.float32),
            safe_front_x=jnp.asarray([[1.6, 1.2]], dtype=jnp.float32),
            init_front_x=jnp.asarray([[0.8, 0.8]], dtype=jnp.float32),
            init_com_x=jnp.asarray([[0.8, 0.8]], dtype=jnp.float32),
            touched_death_plane=jnp.asarray([[False, True]], dtype=jnp.bool_),
            went_airborne=jnp.asarray([[False, False]], dtype=jnp.bool_),
            com_x_went_negative=jnp.asarray([[False, False]], dtype=jnp.bool_),
            mean_activation=jnp.asarray([[0.5, 0.5]], dtype=jnp.float32),
            mean_h_activation=jnp.asarray([[0.3, 0.2]], dtype=jnp.float32),
            mean_v_activation=jnp.asarray([[0.2, 0.3]], dtype=jnp.float32),
            mean_speed=jnp.asarray([[0.01, 0.01]], dtype=jnp.float32),
            cliff_vicinity_steps=jnp.asarray([[10, 5]], dtype=jnp.int32),
            cliff_vicinity_fraction=jnp.asarray([[0.25, 0.125]], dtype=jnp.float32),
            time_to_edge=jnp.asarray([[0.3, 1.0]], dtype=jnp.float32),
            elapsed_time_fraction=jnp.asarray([[0.25, 1.0]], dtype=jnp.float32),
            steps_alive=jnp.asarray([[100, 100]], dtype=jnp.int32),
            cliff_x=jnp.asarray([[15, 15]], dtype=jnp.int32),
            task_mode=jnp.asarray([[0, 0]], dtype=jnp.int32),
            goal_direction=jnp.asarray([[0, 0]], dtype=jnp.int32),
            cliff_side=jnp.asarray([[1, 1]], dtype=jnp.int32),
            goal_distance=jnp.asarray([[0, 0]], dtype=jnp.int32),
            cliff_distance=jnp.asarray([[0, 0]], dtype=jnp.int32),
            goal_block_x=jnp.asarray([[-1, -1]], dtype=jnp.int32),
            goal_block_y=jnp.asarray([[-1, -1]], dtype=jnp.int32),
            goal_x=jnp.asarray([[-1.0, -1.0]], dtype=jnp.float32),
            any_progress=jnp.asarray([[0.0, 0.0]], dtype=jnp.float32),
            goal_progress=jnp.asarray([[0.0, 0.0]], dtype=jnp.float32),
            incorrect_progress=jnp.asarray([[0.0, 0.0]], dtype=jnp.float32),
            toward_goal_frontier=jnp.asarray([[0.0, 0.0]], dtype=jnp.float32),
            goal_touched=jnp.asarray([[False, False]], dtype=jnp.bool_),
            cliff_fell=jnp.asarray([[False, True]], dtype=jnp.bool_),
            time_to_goal=jnp.asarray([[1.0, 1.0]], dtype=jnp.float32),
            turn_count=jnp.asarray([[0, 0]], dtype=jnp.int32),
            stable_turn_count=jnp.asarray([[0, 1]], dtype=jnp.int32),
            deformation_channel_max=jnp.asarray(
                [[[0.1] * 4, [0.3] * 4]],
                dtype=jnp.float32,
            ),
            controller_hidden_output_max_abs=jnp.asarray([[0.25, 0.4]], dtype=jnp.float32),
            action_output_max_abs=jnp.asarray([[0.5, 0.8]], dtype=jnp.float32),
        )

        batch = candidate_batch_fitness(result)

        assert batch.episode_peak_front_xs.shape == (1, 2)
        assert batch.episode_safe_front_xs.shape == (1, 2)
        assert batch.episode_probe_extents.shape == (1, 2)
        assert batch.episode_h_activations.shape == (1, 2)
        assert batch.episode_v_activations.shape == (1, 2)
        assert batch.episode_cliff_vicinity_steps.shape == (1, 2)
        assert batch.episode_cliff_vicinity_fractions.shape == (1, 2)
        assert batch.deformation_channel_max.shape == (1, 4)
        assert batch.deformation_channel_max[0, 0] == pytest.approx(0.3)
        assert batch.controller_hidden_output_max_abs[0] == pytest.approx(0.4)
        assert batch.action_output_max_abs[0] == pytest.approx(0.8)
        assert batch.fitness[0] == pytest.approx((74.5 + 10.0) / 2.0)
        assert batch.fall_count[0] == pytest.approx(1.0)
        assert batch.cliff_vicinity[0] == pytest.approx((0.25 + 0.125) / 2.0)
        assert batch.peak_progress_rate[0] == pytest.approx((2.0 + 1.0) / 2.0)
        assert batch.time_to_edge[0] == pytest.approx((0.3 + 1.0) / 2.0)
        assert batch.correct_progress[0] == pytest.approx(0.0)
        assert batch.incorrect_progress[0] == pytest.approx(0.0)
        assert batch.toward_goal_frontier[0] == pytest.approx(0.0)
        assert batch.turn_count[0] == pytest.approx(0.0)
        assert batch.stable_turn_count[0] == pytest.approx(0.5)
        assert batch.mirrored_pair_count[0] == pytest.approx(0.0)
        assert batch.episode_peak_progress_rates[0, 0] == pytest.approx(2.0)
        assert batch.episode_peak_progress_rates[0, 1] == pytest.approx(1.0)
        assert batch.episode_times_to_edge[0, 0] == pytest.approx(0.3)
        assert batch.episode_times_to_edge[0, 1] == pytest.approx(1.0)
        assert batch.episode_cliff_vicinity_steps[0, 0] == 10
        assert batch.episode_cliff_vicinity_steps[0, 1] == 5
        assert batch.episode_probe_extents[0, 0] == pytest.approx(0.0)
        assert batch.episode_probe_extents[0, 1] == pytest.approx(0.8)
        assert batch.episode_incorrect_progresses[0, 0] == pytest.approx(0.0)
        assert batch.episode_toward_goal_frontiers[0, 1] == pytest.approx(0.0)
        assert batch.episode_turn_counts[0, 1] == 0
        assert batch.episode_stable_turn_counts[0, 1] == 1

    def test_batch_bilateral_progress_metrics_expose_top_two_means(self):
        result = EpisodeResult(
            peak_front_x=jnp.asarray([[0.8, 0.8, 0.8]], dtype=jnp.float32),
            safe_front_x=jnp.asarray([[0.8, 0.8, 0.8]], dtype=jnp.float32),
            init_front_x=jnp.asarray([[0.8, 0.8, 0.8]], dtype=jnp.float32),
            init_com_x=jnp.asarray([[0.8, 0.8, 0.8]], dtype=jnp.float32),
            touched_death_plane=jnp.asarray([[False, False, False]], dtype=jnp.bool_),
            went_airborne=jnp.asarray([[False, False, False]], dtype=jnp.bool_),
            com_x_went_negative=jnp.asarray([[False, False, False]], dtype=jnp.bool_),
            mean_activation=jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float32),
            mean_h_activation=jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float32),
            mean_v_activation=jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float32),
            mean_speed=jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float32),
            cliff_vicinity_steps=jnp.asarray([[0, 0, 0]], dtype=jnp.int32),
            cliff_vicinity_fraction=jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float32),
            time_to_edge=jnp.asarray([[1.0, 1.0, 1.0]], dtype=jnp.float32),
            elapsed_time_fraction=jnp.asarray([[1.0, 1.0, 1.0]], dtype=jnp.float32),
            steps_alive=jnp.asarray([[10, 10, 10]], dtype=jnp.int32),
            cliff_x=jnp.asarray([[15, 15, 15]], dtype=jnp.int32),
            task_mode=jnp.asarray([[1, 1, 1]], dtype=jnp.int32),
            goal_direction=jnp.asarray([[1, 1, 1]], dtype=jnp.int32),
            cliff_side=jnp.asarray([[-1, -1, -1]], dtype=jnp.int32),
            goal_distance=jnp.asarray([[2, 2, 2]], dtype=jnp.int32),
            cliff_distance=jnp.asarray([[1, 1, 1]], dtype=jnp.int32),
            goal_block_x=jnp.asarray([[0, 0, 0]], dtype=jnp.int32),
            goal_block_y=jnp.asarray([[0, 0, 0]], dtype=jnp.int32),
            goal_x=jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float32),
            any_progress=jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float32),
            goal_progress=jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float32),
            incorrect_progress=jnp.asarray([[0.2, 0.5, 0.4]], dtype=jnp.float32),
            toward_goal_frontier=jnp.asarray([[0.1, 0.6, 0.3]], dtype=jnp.float32),
            goal_touched=jnp.asarray([[False, False, False]], dtype=jnp.bool_),
            cliff_fell=jnp.asarray([[False, False, False]], dtype=jnp.bool_),
            time_to_goal=jnp.asarray([[1.0, 1.0, 1.0]], dtype=jnp.float32),
            turn_count=jnp.asarray([[0, 0, 0]], dtype=jnp.int32),
            deformation_channel_max=jnp.zeros((1, 3, 4), dtype=jnp.float32),
            controller_hidden_output_max_abs=jnp.zeros((1, 3), dtype=jnp.float32),
            action_output_max_abs=jnp.zeros((1, 3), dtype=jnp.float32),
        )

        batch = candidate_batch_fitness(result)

        assert batch.correct_progress[0] == pytest.approx((0.1 + 0.6 + 0.3) / 3.0)
        assert batch.incorrect_progress[0] == pytest.approx((0.2 + 0.5 + 0.4) / 3.0)
        assert batch.correct_progress_top2_mean[0] == pytest.approx((0.6 + 0.3) / 2.0)
        assert batch.incorrect_progress_top2_mean[0] == pytest.approx((0.5 + 0.4) / 2.0)

    def test_batch_bilateral_exposes_mirrored_roundtrip_top2_mean(self):
        result = EpisodeResult(
            peak_front_x=jnp.asarray([[0.8] * 6], dtype=jnp.float32),
            safe_front_x=jnp.asarray([[0.8] * 6], dtype=jnp.float32),
            init_front_x=jnp.asarray([[0.8] * 6], dtype=jnp.float32),
            init_com_x=jnp.asarray([[0.8] * 6], dtype=jnp.float32),
            touched_death_plane=jnp.asarray([[False] * 6], dtype=jnp.bool_),
            went_airborne=jnp.asarray([[False] * 6], dtype=jnp.bool_),
            com_x_went_negative=jnp.asarray([[False] * 6], dtype=jnp.bool_),
            mean_activation=jnp.asarray([[0.0] * 6], dtype=jnp.float32),
            mean_h_activation=jnp.asarray([[0.0] * 6], dtype=jnp.float32),
            mean_v_activation=jnp.asarray([[0.0] * 6], dtype=jnp.float32),
            mean_speed=jnp.asarray([[0.0] * 6], dtype=jnp.float32),
            cliff_vicinity_steps=jnp.asarray([[0] * 6], dtype=jnp.int32),
            cliff_vicinity_fraction=jnp.asarray([[0.0] * 6], dtype=jnp.float32),
            time_to_edge=jnp.asarray([[1.0] * 6], dtype=jnp.float32),
            elapsed_time_fraction=jnp.asarray([[1.0] * 6], dtype=jnp.float32),
            steps_alive=jnp.asarray([[10] * 6], dtype=jnp.int32),
            cliff_x=jnp.asarray([[15] * 6], dtype=jnp.int32),
            task_mode=jnp.asarray([[1] * 6], dtype=jnp.int32),
            goal_direction=jnp.asarray([[-1, 1, -1, 1, -1, 1]], dtype=jnp.int32),
            cliff_side=jnp.asarray([[1, -1, 1, -1, 1, -1]], dtype=jnp.int32),
            goal_distance=jnp.asarray([[2, 2, 4, 4, 6, 6]], dtype=jnp.int32),
            cliff_distance=jnp.asarray([[1, 1, 2, 2, 3, 3]], dtype=jnp.int32),
            goal_block_x=jnp.asarray([[0] * 6], dtype=jnp.int32),
            goal_block_y=jnp.asarray([[0] * 6], dtype=jnp.int32),
            goal_x=jnp.asarray([[0.0] * 6], dtype=jnp.float32),
            any_progress=jnp.asarray([[0.0] * 6], dtype=jnp.float32),
            goal_progress=jnp.asarray([[0.0] * 6], dtype=jnp.float32),
            incorrect_progress=jnp.asarray([[0.8, 0.7, 0.4, 0.3, 0.9, 0.8]], dtype=jnp.float32),
            toward_goal_frontier=jnp.asarray([[0.6, 0.9, 0.5, 0.9, 0.2, 0.4]], dtype=jnp.float32),
            goal_touched=jnp.asarray([[False] * 6], dtype=jnp.bool_),
            cliff_fell=jnp.asarray([[False] * 6], dtype=jnp.bool_),
            time_to_goal=jnp.asarray([[1.0] * 6], dtype=jnp.float32),
            turn_count=jnp.asarray([[0] * 6], dtype=jnp.int32),
            deformation_channel_max=jnp.zeros((1, 6, 4), dtype=jnp.float32),
            controller_hidden_output_max_abs=jnp.zeros((1, 6), dtype=jnp.float32),
            action_output_max_abs=jnp.zeros((1, 6), dtype=jnp.float32),
        )

        batch = candidate_batch_fitness(result)

        assert batch.mirrored_roundtrip_top2_mean[0] == pytest.approx((0.6 + 0.3) / 2.0)

    def test_batch_bilateral_time_to_edge_uses_time_to_goal_per_episode(self):
        result = EpisodeResult(
            peak_front_x=jnp.asarray([[0.8, 0.8]], dtype=jnp.float32),
            safe_front_x=jnp.asarray([[0.8, 0.8]], dtype=jnp.float32),
            init_front_x=jnp.asarray([[0.8, 0.8]], dtype=jnp.float32),
            init_com_x=jnp.asarray([[0.8, 0.8]], dtype=jnp.float32),
            touched_death_plane=jnp.asarray([[False, False]], dtype=jnp.bool_),
            went_airborne=jnp.asarray([[False, False]], dtype=jnp.bool_),
            com_x_went_negative=jnp.asarray([[False, False]], dtype=jnp.bool_),
            mean_activation=jnp.asarray([[0.0, 0.0]], dtype=jnp.float32),
            mean_h_activation=jnp.asarray([[0.0, 0.0]], dtype=jnp.float32),
            mean_v_activation=jnp.asarray([[0.0, 0.0]], dtype=jnp.float32),
            mean_speed=jnp.asarray([[0.0, 0.0]], dtype=jnp.float32),
            cliff_vicinity_steps=jnp.asarray([[0, 0]], dtype=jnp.int32),
            cliff_vicinity_fraction=jnp.asarray([[0.0, 0.0]], dtype=jnp.float32),
            time_to_edge=jnp.asarray([[0.9, 0.8]], dtype=jnp.float32),
            elapsed_time_fraction=jnp.asarray([[1.0, 1.0]], dtype=jnp.float32),
            steps_alive=jnp.asarray([[10, 10]], dtype=jnp.int32),
            cliff_x=jnp.asarray([[15, 15]], dtype=jnp.int32),
            task_mode=jnp.asarray([[1, 1]], dtype=jnp.int32),
            goal_direction=jnp.asarray([[-1, 1]], dtype=jnp.int32),
            cliff_side=jnp.asarray([[1, -1]], dtype=jnp.int32),
            goal_distance=jnp.asarray([[2, 2]], dtype=jnp.int32),
            cliff_distance=jnp.asarray([[1, 1]], dtype=jnp.int32),
            goal_block_x=jnp.asarray([[0, 0]], dtype=jnp.int32),
            goal_block_y=jnp.asarray([[0, 0]], dtype=jnp.int32),
            goal_x=jnp.asarray([[0.0, 0.0]], dtype=jnp.float32),
            any_progress=jnp.asarray([[0.0, 0.0]], dtype=jnp.float32),
            goal_progress=jnp.asarray([[0.0, 0.0]], dtype=jnp.float32),
            incorrect_progress=jnp.asarray([[0.0, 0.0]], dtype=jnp.float32),
            toward_goal_frontier=jnp.asarray([[0.0, 0.0]], dtype=jnp.float32),
            goal_touched=jnp.asarray([[False, False]], dtype=jnp.bool_),
            cliff_fell=jnp.asarray([[False, False]], dtype=jnp.bool_),
            time_to_goal=jnp.asarray([[0.25, 0.5]], dtype=jnp.float32),
            turn_count=jnp.asarray([[0, 0]], dtype=jnp.int32),
            deformation_channel_max=jnp.zeros((1, 2, 4), dtype=jnp.float32),
            controller_hidden_output_max_abs=jnp.zeros((1, 2), dtype=jnp.float32),
            action_output_max_abs=jnp.zeros((1, 2), dtype=jnp.float32),
        )

        batch = candidate_batch_fitness(result)

        assert batch.time_to_edge[0] == pytest.approx((0.25 + 0.5) / 2.0)
        assert batch.episode_times_to_edge[0, 0] == pytest.approx(0.25)
        assert batch.episode_times_to_edge[0, 1] == pytest.approx(0.5)

    def test_batch_candidate_level_activation_penalty_no_longer_changes_scalar_fitness(self):
        result = EpisodeResult(
            peak_front_x=jnp.asarray([[1.6, 1.2]], dtype=jnp.float32),
            safe_front_x=jnp.asarray([[1.6, 1.2]], dtype=jnp.float32),
            init_front_x=jnp.asarray([[0.8, 0.8]], dtype=jnp.float32),
            init_com_x=jnp.asarray([[0.8, 0.8]], dtype=jnp.float32),
            touched_death_plane=jnp.asarray([[False, False]], dtype=jnp.bool_),
            went_airborne=jnp.asarray([[False, False]], dtype=jnp.bool_),
            com_x_went_negative=jnp.asarray([[False, False]], dtype=jnp.bool_),
            mean_activation=jnp.asarray([[0.5, 1.0]], dtype=jnp.float32),
            mean_h_activation=jnp.asarray([[0.25, 0.5]], dtype=jnp.float32),
            mean_v_activation=jnp.asarray([[0.25, 0.5]], dtype=jnp.float32),
            mean_speed=jnp.asarray([[0.0, 0.0]], dtype=jnp.float32),
            cliff_vicinity_steps=jnp.asarray([[0, 0]], dtype=jnp.int32),
            cliff_vicinity_fraction=jnp.asarray([[0.0, 0.0]], dtype=jnp.float32),
            time_to_edge=jnp.asarray([[1.0, 1.0]], dtype=jnp.float32),
            elapsed_time_fraction=jnp.asarray([[1.0, 1.0]], dtype=jnp.float32),
            steps_alive=jnp.asarray([[100, 100]], dtype=jnp.int32),
            cliff_x=jnp.asarray([[15, 15]], dtype=jnp.int32),
            task_mode=jnp.asarray([[0, 0]], dtype=jnp.int32),
            goal_direction=jnp.asarray([[0, 0]], dtype=jnp.int32),
            cliff_side=jnp.asarray([[1, 1]], dtype=jnp.int32),
            goal_distance=jnp.asarray([[0, 0]], dtype=jnp.int32),
            cliff_distance=jnp.asarray([[0, 0]], dtype=jnp.int32),
            goal_block_x=jnp.asarray([[-1, -1]], dtype=jnp.int32),
            goal_block_y=jnp.asarray([[-1, -1]], dtype=jnp.int32),
            goal_x=jnp.asarray([[-1.0, -1.0]], dtype=jnp.float32),
            any_progress=jnp.asarray([[0.0, 0.0]], dtype=jnp.float32),
            goal_progress=jnp.asarray([[0.0, 0.0]], dtype=jnp.float32),
            incorrect_progress=jnp.asarray([[0.0, 0.0]], dtype=jnp.float32),
            toward_goal_frontier=jnp.asarray([[0.0, 0.0]], dtype=jnp.float32),
            goal_touched=jnp.asarray([[False, False]], dtype=jnp.bool_),
            cliff_fell=jnp.asarray([[False, False]], dtype=jnp.bool_),
            time_to_goal=jnp.asarray([[1.0, 1.0]], dtype=jnp.float32),
            turn_count=jnp.asarray([[0, 0]], dtype=jnp.int32),
            deformation_channel_max=jnp.zeros((1, 2, 4), dtype=jnp.float32),
            controller_hidden_output_max_abs=jnp.zeros((1, 2), dtype=jnp.float32),
            action_output_max_abs=jnp.zeros((1, 2), dtype=jnp.float32),
        )

        batch = candidate_batch_fitness(result, activation_cost_penalty_scale=2.0)

        assert batch.activation_cost[0] == pytest.approx(0.75)
        assert batch.fitness[0] == pytest.approx(15.0, abs=0.1)

    def test_batch_soft_loft_penalty_reduces_scalar_fitness(self):
        episode_a = _make_result(
            peak_front_x=1.6,
            init_front_x=0.8,
            cliff_x=15,
            mean_lofted_excess_cells=0.1,
            peak_com_y_delta_cells=0.3,
        )
        episode_b = _make_result(
            peak_front_x=1.2,
            init_front_x=0.8,
            cliff_x=15,
            mean_lofted_excess_cells=0.2,
            peak_com_y_delta_cells=0.4,
        )
        stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), episode_a, episode_b)
        result = jax.tree_util.tree_map(lambda x: x[None, ...], stacked)

        batch = candidate_batch_fitness(
            result,
            soft_loft_penalty_scale=8.0,
        )

        assert batch.mean_lofted_excess_cells[0] == pytest.approx(0.15)
        assert batch.peak_com_y_delta_cells[0] == pytest.approx(0.35)
        assert batch.fitness[0] == pytest.approx(13.8, abs=0.1)

    def test_batch_noncanonical_bool_storage_matches_canonical_semantics(self):
        noncanonical = np.frombuffer(b"\x00\x01\xff", dtype=np.bool_)
        canonical = np.asarray([False, True, True], dtype=np.bool_)

        def _make_bilateral(goal_touched: np.ndarray, cliff_fell: np.ndarray) -> EpisodeResult:
            touched = jnp.asarray(goal_touched[None, :], dtype=jnp.bool_)
            fell = jnp.asarray(cliff_fell[None, :], dtype=jnp.bool_)
            return EpisodeResult(
                peak_front_x=jnp.asarray([[0.8, 0.8, 0.8]], dtype=jnp.float32),
                safe_front_x=jnp.asarray([[0.8, 0.8, 0.8]], dtype=jnp.float32),
                init_front_x=jnp.asarray([[0.8, 0.8, 0.8]], dtype=jnp.float32),
                init_com_x=jnp.asarray([[0.8, 0.8, 0.8]], dtype=jnp.float32),
                touched_death_plane=jnp.asarray([[False, False, False]], dtype=jnp.bool_),
                went_airborne=jnp.asarray([[False, False, False]], dtype=jnp.bool_),
                com_x_went_negative=jnp.asarray([[False, False, False]], dtype=jnp.bool_),
                mean_activation=jnp.asarray([[0.1, 0.1, 0.1]], dtype=jnp.float32),
                mean_h_activation=jnp.asarray([[0.05, 0.05, 0.05]], dtype=jnp.float32),
                mean_v_activation=jnp.asarray([[0.05, 0.05, 0.05]], dtype=jnp.float32),
                mean_speed=jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float32),
                cliff_vicinity_steps=jnp.asarray([[0, 0, 0]], dtype=jnp.int32),
                cliff_vicinity_fraction=jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float32),
                time_to_edge=jnp.asarray([[1.0, 1.0, 1.0]], dtype=jnp.float32),
                elapsed_time_fraction=jnp.asarray([[1.0, 1.0, 1.0]], dtype=jnp.float32),
                steps_alive=jnp.asarray([[20, 20, 20]], dtype=jnp.int32),
                cliff_x=jnp.asarray([[15, 15, 15]], dtype=jnp.int32),
                task_mode=jnp.asarray([[1, 1, 1]], dtype=jnp.int32),
                goal_direction=jnp.asarray([[1, -1, 1]], dtype=jnp.int32),
                cliff_side=jnp.asarray([[-1, 1, -1]], dtype=jnp.int32),
                goal_distance=jnp.asarray([[2, 2, 4]], dtype=jnp.int32),
                cliff_distance=jnp.asarray([[1, 1, 2]], dtype=jnp.int32),
                goal_block_x=jnp.asarray([[0, 0, 0]], dtype=jnp.int32),
                goal_block_y=jnp.asarray([[0, 0, 0]], dtype=jnp.int32),
                goal_x=jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float32),
                any_progress=jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float32),
                goal_progress=jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float32),
                incorrect_progress=jnp.asarray([[0.25, 0.25, 0.5]], dtype=jnp.float32),
                toward_goal_frontier=jnp.asarray([[0.4, 0.4, 0.7]], dtype=jnp.float32),
                goal_touched=touched,
                cliff_fell=fell,
                time_to_goal=jnp.asarray([[0.7, 0.5, 0.3]], dtype=jnp.float32),
                turn_count=jnp.asarray([[0, 0, 0]], dtype=jnp.int32),
                deformation_channel_max=jnp.zeros((1, 3, 4), dtype=jnp.float32),
                controller_hidden_output_max_abs=jnp.zeros((1, 3), dtype=jnp.float32),
                action_output_max_abs=jnp.zeros((1, 3), dtype=jnp.float32),
            )

        noncanonical_result = _make_bilateral(noncanonical, noncanonical)
        canonical_result = _make_bilateral(canonical, canonical)

        batch_noncanonical = candidate_batch_fitness(noncanonical_result)
        batch_canonical = candidate_batch_fitness(canonical_result)

        np.testing.assert_allclose(
            np.asarray(batch_noncanonical.fitness),
            np.asarray(batch_canonical.fitness),
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(batch_noncanonical.fall_count),
            np.asarray(batch_canonical.fall_count),
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(batch_noncanonical.goal_hit_count),
            np.asarray(batch_canonical.goal_hit_count),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_episode_fitness_array_jit_matches_eager_for_noncanonical_booleans(self):
        noncanonical = np.frombuffer(b"\x00\x01\xff", dtype=np.bool_)
        result = EpisodeResult(
            peak_front_x=jnp.asarray([0.8, 0.8, 0.8], dtype=jnp.float32),
            safe_front_x=jnp.asarray([0.8, 0.8, 0.8], dtype=jnp.float32),
            init_front_x=jnp.asarray([0.8, 0.8, 0.8], dtype=jnp.float32),
            init_com_x=jnp.asarray([0.8, 0.8, 0.8], dtype=jnp.float32),
            touched_death_plane=jnp.asarray([False, False, False], dtype=jnp.bool_),
            went_airborne=jnp.asarray([False, False, False], dtype=jnp.bool_),
            com_x_went_negative=jnp.asarray([False, False, False], dtype=jnp.bool_),
            mean_activation=jnp.asarray([0.1, 0.1, 0.1], dtype=jnp.float32),
            mean_h_activation=jnp.asarray([0.05, 0.05, 0.05], dtype=jnp.float32),
            mean_v_activation=jnp.asarray([0.05, 0.05, 0.05], dtype=jnp.float32),
            mean_speed=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
            cliff_vicinity_steps=jnp.asarray([0, 0, 0], dtype=jnp.int32),
            cliff_vicinity_fraction=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
            time_to_edge=jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float32),
            elapsed_time_fraction=jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float32),
            steps_alive=jnp.asarray([20, 20, 20], dtype=jnp.int32),
            cliff_x=jnp.asarray([15, 15, 15], dtype=jnp.int32),
            task_mode=jnp.asarray([1, 1, 1], dtype=jnp.int32),
            goal_direction=jnp.asarray([1, -1, 1], dtype=jnp.int32),
            cliff_side=jnp.asarray([-1, 1, -1], dtype=jnp.int32),
            goal_distance=jnp.asarray([2, 2, 4], dtype=jnp.int32),
            cliff_distance=jnp.asarray([1, 1, 2], dtype=jnp.int32),
            goal_block_x=jnp.asarray([0, 0, 0], dtype=jnp.int32),
            goal_block_y=jnp.asarray([0, 0, 0], dtype=jnp.int32),
            goal_x=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
            any_progress=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
            goal_progress=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
            incorrect_progress=jnp.asarray([0.25, 0.25, 0.5], dtype=jnp.float32),
            toward_goal_frontier=jnp.asarray([0.4, 0.4, 0.7], dtype=jnp.float32),
            goal_touched=jnp.asarray(noncanonical, dtype=jnp.bool_),
            cliff_fell=jnp.asarray(noncanonical, dtype=jnp.bool_),
            time_to_goal=jnp.asarray([0.7, 0.5, 0.3], dtype=jnp.float32),
            turn_count=jnp.asarray([0, 0, 0], dtype=jnp.int32),
            deformation_channel_max=jnp.zeros((3, 4), dtype=jnp.float32),
            controller_hidden_output_max_abs=jnp.zeros((3,), dtype=jnp.float32),
            action_output_max_abs=jnp.zeros((3,), dtype=jnp.float32),
        )

        eager = episode_fitness_array(result)
        jitted = jax.jit(episode_fitness_array)(result)
        np.testing.assert_allclose(
            np.asarray(jitted),
            np.asarray(eager),
            atol=1e-6,
            rtol=1e-6,
        )
