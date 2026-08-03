"""Fitness computation and QD descriptor extraction.

Fitness rewards safe edge arrival and safe dwell near the cliff band, with
safe forward progress and earlier arrival acting as lower-weight shaping
terms. Backward escape off the left edge scores -20.

QD descriptors:
  - activation_cost: mean |contraction| from NCA action channels
  - movement_speed: mean absolute COM-x speed in cells per env step
  - peak_progress_rate: clipped normalized peak route progress / elapsed budget
  - time_to_edge: first cliff-band arrival time as a fraction of max_steps
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from jax_evogym.constants import CELL_SIZE

from .env import EpisodeResult


PROGRESS_SHAPING_SCALE = 20.0
SAFE_ARRIVAL_BONUS = 40.0
SAFE_DWELL_BONUS_SCALE = 30.0
URGENCY_SHAPING_SCALE = 10.0
LEFT_EDGE_PENALTY = 20.0
MAX_PEAK_PROGRESS_RATE = 2.0
OUTWARD_PROGRESS_SCALE = 10.0
REVERSAL_PROGRESS_SCALE = 20.0
TARGET_PROGRESS_SCALE = 30.0
REVERSAL_GATE_START = 0.5
REVERSAL_GATE_FULL = 0.7
WRONG_FRONTIER_REWARD_CAP = 1.0
GOAL_TOUCH_BONUS = 40.0
CLIFF_FALL_PENALTY = 20.0
LOWER_LEDGE_PENALTY_CAP = CLIFF_FALL_PENALTY * 0.5
MIRRORED_PAIR_BONUS = 20.0
_TASK_MODE_BILATERAL_GOAL_CLIFF = 1

CANDIDATE_SCALAR_METRIC_ATTRS = (
    "fall_count",
    "goal_hit_count",
    "cliff_vicinity",
    "alive_cell_count",
    "activation_cost",
    "h_activation",
    "v_activation",
    "movement_speed",
    "peak_progress_rate",
    "time_to_edge",
    "correct_progress",
    "incorrect_progress",
    "correct_progress_top2_mean",
    "incorrect_progress_top2_mean",
    "toward_goal_frontier",
    "reversal_progress",
    "turn_count",
    "stable_turn_count",
    "turn_motion_threshold",
    "mirrored_pair_count",
    "mirrored_best_hit_count",
    "mirrored_total_hit_count",
    "mirrored_roundtrip_top2_mean",
    "lower_ledge_penalty",
    "lofted_fraction",
    "mean_lofted_excess_cells",
    "peak_com_y_delta_cells",
    "gait_spectral_purity",
)

# Backward-compatible aliases for older imports.
ANY_PROGRESS_SCALE = OUTWARD_PROGRESS_SCALE
GOAL_PROGRESS_SCALE = TARGET_PROGRESS_SCALE

SURVIVAL_BONUS = SAFE_ARRIVAL_BONUS
DWELL_BONUS_SCALE = SAFE_DWELL_BONUS_SCALE
EARLY_EDGE_BONUS_SCALE = URGENCY_SHAPING_SCALE


def _bool_to_f32(mask: jnp.ndarray) -> jnp.ndarray:
    """Convert bool mask to float32 without relying on bool->float casts."""
    return jnp.where(mask, jnp.float32(1.0), jnp.float32(0.0))


def _count_true(mask: jnp.ndarray, *, axis: int = -1) -> jnp.ndarray:
    """Count true values without relying on bool->int casts."""
    return jnp.sum(
        jnp.where(mask, jnp.int32(1), jnp.int32(0)),
        axis=axis,
        dtype=jnp.int32,
    )


def _safe_edge_gate_scalar(
    *,
    cliff_vicinity_steps: int,
    touched_death_plane: bool,
    went_airborne: bool,
    com_x_went_negative: bool,
) -> bool:
    """Return whether a run safely entered the cliff band."""
    return bool(
        cliff_vicinity_steps > 0
        and not touched_death_plane
        and not went_airborne
        and not com_x_went_negative
    )


def _safe_edge_gate_array(result: EpisodeResult) -> jnp.ndarray:
    """Vectorized safe-cliff-band arrival gate for one or more episodes."""
    return (
        (result.cliff_vicinity_steps > 0)
        & (~result.touched_death_plane)
        & (~result.went_airborne)
        & (~result.com_x_went_negative)
    )


def _progress_shaping_scalar(result: EpisodeResult) -> float:
    """Reward normalized safe forward progress as a shaping signal."""
    edge_x = (int(result.cliff_x) + 1) * CELL_SIZE
    route_length = edge_x - float(result.init_front_x)
    if route_length <= 0.0:
        return 0.0
    safe_progress = min(
        max(float(result.safe_front_x) - float(result.init_front_x), 0.0),
        route_length,
    )
    return (safe_progress / route_length) * PROGRESS_SHAPING_SCALE


def _progress_shaping_array(result: EpisodeResult) -> jnp.ndarray:
    """Vectorized normalized safe forward-progress shaping signal."""
    init_front_x = result.init_front_x.astype(jnp.float32)
    edge_x = (result.cliff_x.astype(jnp.float32) + 1.0) * CELL_SIZE
    route_length = jnp.maximum(edge_x - init_front_x, 0.0)
    safe_route = jnp.where(route_length > 0.0, route_length, 1.0)
    safe_progress = jnp.clip(
        result.safe_front_x.astype(jnp.float32) - init_front_x,
        0.0,
        route_length,
    )
    return jnp.where(
        route_length > 0.0,
        (safe_progress / safe_route) * jnp.float32(PROGRESS_SHAPING_SCALE),
        0.0,
    )


def _urgency_bonus_scalar(
    *,
    time_to_edge: float,
    cliff_vicinity_steps: int,
    touched_death_plane: bool,
    went_airborne: bool,
    com_x_went_negative: bool,
) -> float:
    """Reward earlier safe cliff-band arrival without rewarding failed rushes."""
    if not _safe_edge_gate_scalar(
        cliff_vicinity_steps=cliff_vicinity_steps,
        touched_death_plane=touched_death_plane,
        went_airborne=went_airborne,
        com_x_went_negative=com_x_went_negative,
    ):
        return 0.0
    return URGENCY_SHAPING_SCALE * max(1.0 - float(time_to_edge), 0.0)


def _urgency_bonus_array(result: EpisodeResult) -> jnp.ndarray:
    """Vectorized urgency reward for one or more episodes."""
    arrived_safely = _safe_edge_gate_array(result)
    return jnp.where(
        arrived_safely,
        jnp.maximum(
            jnp.float32(URGENCY_SHAPING_SCALE)
            * (jnp.float32(1.0) - result.time_to_edge.astype(jnp.float32)),
            0.0,
        ),
        0.0,
    )


def _lower_ledge_exposure_fraction_array(result: EpisodeResult) -> jnp.ndarray:
    """Broadcast lower-ledge exposure onto the episode batch shape."""
    return jnp.clip(
        jnp.broadcast_to(
            jnp.asarray(result.lower_ledge_exposure_fraction, dtype=jnp.float32),
            result.task_mode.shape,
        ),
        0.0,
        1.0,
    )


def _lower_ledge_penalty_scalar(result: EpisodeResult) -> float:
    """Convert lower-ledge exposure into a capped soft penalty."""
    return min(
        max(float(result.lower_ledge_exposure_fraction), 0.0),
        1.0,
    ) * LOWER_LEDGE_PENALTY_CAP


def _lower_ledge_penalty_array(result: EpisodeResult) -> jnp.ndarray:
    """Vectorized lowered-ledge exposure penalty."""
    return (
        _lower_ledge_exposure_fraction_array(result)
        * jnp.float32(LOWER_LEDGE_PENALTY_CAP)
    )


def _top_two_mean_scalar(values: list[float]) -> float:
    """Return the mean of the top two values, or fewer when unavailable."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    top_two = sorted((float(value) for value in values), reverse=True)[:2]
    return float(sum(top_two) / len(top_two))


def _top_two_mean_array(values: jnp.ndarray) -> jnp.ndarray:
    """Vectorized top-two mean over the episode axis."""
    values = values.astype(jnp.float32)
    n_episodes = int(values.shape[-1])
    if n_episodes <= 0:
        return jnp.zeros(values.shape[:-1], dtype=jnp.float32)
    if n_episodes == 1:
        return jnp.squeeze(values, axis=-1)
    return jnp.mean(jnp.sort(values, axis=-1)[..., -2:], axis=-1)


def _top_two_mean_masked_array(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Vectorized top-two mean over valid entries only."""
    values = values.astype(jnp.float32)
    mask = mask.astype(jnp.bool_)
    n_values = int(values.shape[-1])
    if n_values <= 0:
        return jnp.zeros(values.shape[:-1], dtype=jnp.float32)
    masked = jnp.where(mask, values, -jnp.inf)
    top_two = jnp.sort(masked, axis=-1)[..., -min(n_values, 2):]
    top_two = jnp.where(jnp.isfinite(top_two), top_two, 0.0)
    valid_count = _count_true(mask, axis=-1).astype(jnp.float32)
    denom = jnp.minimum(valid_count, jnp.float32(2.0))
    return jnp.where(valid_count > 0.0, jnp.sum(top_two, axis=-1) / denom, 0.0)


def _reversal_reward_gate_scalar(incorrect_progress: float) -> float:
    """Ramp reversal reward on only after meaningful outbound frontier approach."""
    gate_width = max(REVERSAL_GATE_FULL - REVERSAL_GATE_START, 1e-6)
    return min(max((float(incorrect_progress) - REVERSAL_GATE_START) / gate_width, 0.0), 1.0)


def _reversal_reward_gate_array(incorrect_progress: jnp.ndarray) -> jnp.ndarray:
    """Vectorized soft gate for reversal reward based on wrong-side frontier approach."""
    gate_width = jnp.maximum(
        jnp.float32(REVERSAL_GATE_FULL - REVERSAL_GATE_START),
        jnp.float32(1e-6),
    )
    return jnp.clip(
        (incorrect_progress.astype(jnp.float32) - jnp.float32(REVERSAL_GATE_START)) / gate_width,
        jnp.float32(0.0),
        jnp.float32(1.0),
    )


def _mirrored_roundtrip_top2_mean_scalar(episode_results: list[EpisodeResult]) -> float:
    """Return the top-two mean paired roundtrip symmetry score across mirrored pairs."""
    pair_scores: list[float] = []
    idx = 0
    while idx + 1 < len(episode_results):
        left = episode_results[idx]
        right = episode_results[idx + 1]
        valid_pair = (
            int(left.task_mode) == _TASK_MODE_BILATERAL_GOAL_CLIFF
            and int(right.task_mode) == _TASK_MODE_BILATERAL_GOAL_CLIFF
            and int(left.goal_direction) != 0
            and int(right.goal_direction) != 0
            and int(left.goal_direction) == -int(right.goal_direction)
            and int(left.goal_distance) == int(right.goal_distance)
            and int(left.cliff_distance) == int(right.cliff_distance)
        )
        if valid_pair:
            left_roundtrip = min(float(left.incorrect_progress), float(left.toward_goal_frontier))
            right_roundtrip = min(float(right.incorrect_progress), float(right.toward_goal_frontier))
            pair_scores.append(min(left_roundtrip, right_roundtrip))
            idx += 2
            continue
        idx += 1
    return _top_two_mean_scalar(pair_scores)


def _mirrored_roundtrip_top2_mean_array(episode_results: EpisodeResult) -> jnp.ndarray:
    """Vectorized top-two mean paired roundtrip symmetry score across mirrored pairs."""
    n_episodes = int(episode_results.goal_direction.shape[-1])
    n_pairs = n_episodes // 2
    if n_pairs <= 0:
        return jnp.zeros(episode_results.goal_direction.shape[:-1], dtype=jnp.float32)

    roundtrip = jnp.minimum(
        episode_results.incorrect_progress.astype(jnp.float32),
        episode_results.toward_goal_frontier.astype(jnp.float32),
    )
    pair_a = roundtrip[..., : (2 * n_pairs) : 2]
    pair_b = roundtrip[..., 1 : (2 * n_pairs) : 2]
    dir_a = episode_results.goal_direction[..., : (2 * n_pairs) : 2].astype(jnp.int32)
    dir_b = episode_results.goal_direction[..., 1 : (2 * n_pairs) : 2].astype(jnp.int32)
    goal_distance_a = episode_results.goal_distance[..., : (2 * n_pairs) : 2].astype(jnp.int32)
    goal_distance_b = episode_results.goal_distance[..., 1 : (2 * n_pairs) : 2].astype(jnp.int32)
    cliff_distance_a = episode_results.cliff_distance[..., : (2 * n_pairs) : 2].astype(jnp.int32)
    cliff_distance_b = episode_results.cliff_distance[..., 1 : (2 * n_pairs) : 2].astype(jnp.int32)
    bilateral_a = (
        episode_results.task_mode[..., : (2 * n_pairs) : 2].astype(jnp.int32)
        == jnp.int32(_TASK_MODE_BILATERAL_GOAL_CLIFF)
    )
    bilateral_b = (
        episode_results.task_mode[..., 1 : (2 * n_pairs) : 2].astype(jnp.int32)
        == jnp.int32(_TASK_MODE_BILATERAL_GOAL_CLIFF)
    )
    valid_pairs = (
        bilateral_a
        & bilateral_b
        & (dir_a != 0)
        & (dir_b != 0)
        & (dir_a == -dir_b)
        & (goal_distance_a == goal_distance_b)
        & (cliff_distance_a == cliff_distance_b)
    )
    pair_scores = jnp.minimum(pair_a, pair_b)
    return _top_two_mean_masked_array(pair_scores, valid_pairs)


class CandidateResult(NamedTuple):
    """Aggregated fitness and descriptors for one candidate across episodes."""
    fitness: float
    fall_count: float
    goal_hit_count: float
    cliff_vicinity: float
    alive_cell_count: float
    activation_cost: float
    h_activation: float
    v_activation: float
    movement_speed: float
    peak_progress_rate: float
    time_to_edge: float
    correct_progress: float
    incorrect_progress: float
    correct_progress_top2_mean: float
    incorrect_progress_top2_mean: float
    toward_goal_frontier: float
    turn_count: float
    mirrored_pair_count: float
    n_episodes: int
    # Per-episode detail for logging.
    episode_fitnesses: list[float]
    episode_cliff_xs: list[int]
    episode_fell: list[bool]
    episode_airborne: list[bool]
    episode_left_edge: list[bool]
    episode_steps_alive: list[int]
    lower_ledge_penalty: float = 0.0
    mirrored_best_hit_count: float = 0.0
    mirrored_total_hit_count: float = 0.0
    mirrored_roundtrip_top2_mean: float = 0.0
    reversal_progress: float = 0.0
    stable_turn_count: float = 0.0
    turn_motion_threshold: float = 0.0
    lofted_fraction: float = 0.0
    mean_lofted_excess_cells: float = 0.0
    peak_com_y_delta_cells: float = 0.0
    gait_spectral_purity: float = 0.0


class CandidateBatchResult(NamedTuple):
    """Aggregated candidate metrics for a whole candidate batch."""
    fitness: jnp.ndarray
    fall_count: jnp.ndarray
    goal_hit_count: jnp.ndarray
    cliff_vicinity: jnp.ndarray
    alive_cell_count: jnp.ndarray
    activation_cost: jnp.ndarray
    h_activation: jnp.ndarray
    v_activation: jnp.ndarray
    movement_speed: jnp.ndarray
    peak_progress_rate: jnp.ndarray
    time_to_edge: jnp.ndarray
    correct_progress: jnp.ndarray
    incorrect_progress: jnp.ndarray
    correct_progress_top2_mean: jnp.ndarray
    incorrect_progress_top2_mean: jnp.ndarray
    toward_goal_frontier: jnp.ndarray
    turn_count: jnp.ndarray
    mirrored_pair_count: jnp.ndarray
    episode_fitnesses: jnp.ndarray
    episode_activation_costs: jnp.ndarray
    episode_h_activations: jnp.ndarray
    episode_v_activations: jnp.ndarray
    episode_movement_speeds: jnp.ndarray
    episode_peak_progress_rates: jnp.ndarray
    episode_times_to_edge: jnp.ndarray
    episode_cliff_vicinity_steps: jnp.ndarray
    episode_cliff_vicinity_fractions: jnp.ndarray
    episode_init_front_xs: jnp.ndarray
    episode_peak_front_xs: jnp.ndarray
    episode_safe_front_xs: jnp.ndarray
    episode_probe_extents: jnp.ndarray
    episode_cliff_xs: jnp.ndarray
    episode_any_progresses: jnp.ndarray
    episode_goal_progresses: jnp.ndarray
    episode_incorrect_progresses: jnp.ndarray
    episode_toward_goal_frontiers: jnp.ndarray
    episode_goal_touched: jnp.ndarray
    episode_cliff_fell: jnp.ndarray
    episode_goal_directions: jnp.ndarray
    episode_goal_distances: jnp.ndarray
    episode_cliff_distances: jnp.ndarray
    episode_times_to_goal: jnp.ndarray
    episode_turn_counts: jnp.ndarray
    episode_fell: jnp.ndarray
    episode_airborne: jnp.ndarray
    episode_left_edge: jnp.ndarray
    episode_steps_alive: jnp.ndarray
    deformation_channel_max: jnp.ndarray
    controller_hidden_output_max_abs: jnp.ndarray
    action_output_max_abs: jnp.ndarray
    lower_ledge_penalty: jnp.ndarray = 0.0
    mirrored_best_hit_count: jnp.ndarray = 0.0
    mirrored_total_hit_count: jnp.ndarray = 0.0
    mirrored_roundtrip_top2_mean: jnp.ndarray = 0.0
    reversal_progress: jnp.ndarray = 0.0
    episode_reversal_progresses: jnp.ndarray = 0.0
    stable_turn_count: jnp.ndarray = 0.0
    episode_stable_turn_counts: jnp.ndarray = 0
    turn_motion_threshold: jnp.ndarray = 0.0
    episode_turn_motion_thresholds: jnp.ndarray = 0.0
    episode_first_significant_directions: jnp.ndarray = 0
    lofted_fraction: jnp.ndarray = 0.0
    mean_lofted_excess_cells: jnp.ndarray = 0.0
    peak_com_y_delta_cells: jnp.ndarray = 0.0
    episode_lofted_fractions: jnp.ndarray = 0.0
    episode_mean_lofted_excess_cells: jnp.ndarray = 0.0
    episode_peak_com_y_delta_cells: jnp.ndarray = 0.0
    gait_spectral_purity: jnp.ndarray = 0.0


def _mirrored_side_hit_counts_scalar(
    episode_results: list[EpisodeResult],
) -> tuple[float, float]:
    """Count bilateral goal hits on each mirrored side."""
    left_hits = 0
    right_hits = 0
    for result in episode_results:
        if int(result.task_mode) != _TASK_MODE_BILATERAL_GOAL_CLIFF:
            continue
        direction = int(result.goal_direction)
        if direction == 0:
            continue
        if not bool(result.goal_touched):
            continue
        if direction < 0:
            left_hits += 1
        elif direction > 0:
            right_hits += 1
    return float(left_hits), float(right_hits)


def _mirrored_side_hit_counts_array(
    episode_results: EpisodeResult,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Vectorized bilateral goal hits on each mirrored side per candidate."""
    bilateral = (
        episode_results.task_mode.astype(jnp.int32)
        == jnp.int32(_TASK_MODE_BILATERAL_GOAL_CLIFF)
    )
    direction = episode_results.goal_direction.astype(jnp.int32)
    goal_touched = episode_results.goal_touched.astype(jnp.bool_)
    left_hits = _count_true(
        bilateral & goal_touched & (direction < 0),
        axis=-1,
    )
    right_hits = _count_true(
        bilateral & goal_touched & (direction > 0),
        axis=-1,
    )
    return left_hits.astype(jnp.float32), right_hits.astype(jnp.float32)


def _mirrored_pair_count_scalar(episode_results: list[EpisodeResult]) -> float:
    """Count balanced left/right bilateral successes, regardless of geometry."""
    left_hits, right_hits = _mirrored_side_hit_counts_scalar(episode_results)
    return float(min(left_hits, right_hits))


def _mirrored_best_hit_count_scalar(episode_results: list[EpisodeResult]) -> float:
    """Return the larger of the left/right bilateral goal-hit counts."""
    left_hits, right_hits = _mirrored_side_hit_counts_scalar(episode_results)
    return float(max(left_hits, right_hits))


def _mirrored_total_hit_count_scalar(episode_results: list[EpisodeResult]) -> float:
    """Return the total bilateral goal-hit count across both mirrored sides."""
    left_hits, right_hits = _mirrored_side_hit_counts_scalar(episode_results)
    return float(left_hits + right_hits)


def _mirrored_pair_count_array(episode_results: EpisodeResult) -> jnp.ndarray:
    """Vectorized balanced left/right bilateral success count per candidate."""
    left_hits, right_hits = _mirrored_side_hit_counts_array(episode_results)
    return jnp.minimum(left_hits, right_hits).astype(jnp.float32)


def _mirrored_best_hit_count_array(episode_results: EpisodeResult) -> jnp.ndarray:
    """Vectorized larger left/right bilateral goal-hit count per candidate."""
    left_hits, right_hits = _mirrored_side_hit_counts_array(episode_results)
    return jnp.maximum(left_hits, right_hits).astype(jnp.float32)


def _mirrored_total_hit_count_array(episode_results: EpisodeResult) -> jnp.ndarray:
    """Vectorized total bilateral goal-hit count across both mirrored sides."""
    left_hits, right_hits = _mirrored_side_hit_counts_array(episode_results)
    return (left_hits + right_hits).astype(jnp.float32)


def episode_fitness(result: EpisodeResult) -> float:
    """Compute edge-centric fitness for a single episode.

    Returns:
        Fitness score. Safe route progress contributes up to 20 as shaping,
        safe cliff-band arrival contributes 40, safe cliff-band dwell
        contributes up to 30, and earlier safe arrival contributes up to 10.
        Airborne runs score 0. Backward escape off the left edge scores -20
        when the run was not already nullified as airborne.
    """
    if int(result.task_mode) == _TASK_MODE_BILATERAL_GOAL_CLIFF:
        if bool(result.com_x_went_negative):
            return -LEFT_EDGE_PENALTY
        reversal_reward = (
            REVERSAL_PROGRESS_SCALE
            * float(result.reversal_progress)
            * _reversal_reward_gate_scalar(float(result.incorrect_progress))
        )
        goal_touch_bonus = GOAL_TOUCH_BONUS if bool(result.goal_touched) else 0.0
        return (
            (
                OUTWARD_PROGRESS_SCALE
                * min(float(result.incorrect_progress), WRONG_FRONTIER_REWARD_CAP)
            )
            + reversal_reward
            + (TARGET_PROGRESS_SCALE * float(result.toward_goal_frontier))
            + goal_touch_bonus
            - (CLIFF_FALL_PENALTY if bool(result.cliff_fell) else 0.0)
            - _lower_ledge_penalty_scalar(result)
        )

    if bool(result.went_airborne):
        return 0.0
    if bool(result.com_x_went_negative):
        return -LEFT_EDGE_PENALTY

    progress_shaping = _progress_shaping_scalar(result)
    safe_edge_arrival = _safe_edge_gate_scalar(
        cliff_vicinity_steps=int(result.cliff_vicinity_steps),
        touched_death_plane=bool(result.touched_death_plane),
        went_airborne=bool(result.went_airborne),
        com_x_went_negative=bool(result.com_x_went_negative),
    )
    safe_arrival_bonus = SAFE_ARRIVAL_BONUS if safe_edge_arrival else 0.0
    safe_dwell_bonus = (
        SAFE_DWELL_BONUS_SCALE * float(result.cliff_vicinity_fraction)
        if safe_edge_arrival
        else 0.0
    )
    urgency_shaping = _urgency_bonus_scalar(
        time_to_edge=float(result.time_to_edge),
        cliff_vicinity_steps=int(result.cliff_vicinity_steps),
        touched_death_plane=bool(result.touched_death_plane),
        went_airborne=bool(result.went_airborne),
        com_x_went_negative=bool(result.com_x_went_negative),
    )
    return progress_shaping + safe_arrival_bonus + safe_dwell_bonus + urgency_shaping


def episode_fitness_array(result: EpisodeResult) -> jnp.ndarray:
    """Vectorized edge-centric fitness for one or more episodes."""
    bilateral = result.task_mode.astype(jnp.int32) == jnp.int32(_TASK_MODE_BILATERAL_GOAL_CLIFF)
    goal_touched_f = _bool_to_f32(result.goal_touched)
    cliff_fell_f = _bool_to_f32(result.cliff_fell)
    reversal_progress = jnp.broadcast_to(
        jnp.asarray(result.reversal_progress, dtype=jnp.float32),
        result.task_mode.shape,
    )
    reversal_reward = (
        reversal_progress
        * jnp.float32(REVERSAL_PROGRESS_SCALE)
        * _reversal_reward_gate_array(result.incorrect_progress)
    )
    bilateral_score = (
        jnp.minimum(
            result.incorrect_progress.astype(jnp.float32),
            jnp.float32(WRONG_FRONTIER_REWARD_CAP),
        )
        * jnp.float32(OUTWARD_PROGRESS_SCALE)
        + reversal_reward
        + result.toward_goal_frontier.astype(jnp.float32) * jnp.float32(TARGET_PROGRESS_SCALE)
        + goal_touched_f * jnp.float32(GOAL_TOUCH_BONUS)
        - cliff_fell_f * jnp.float32(CLIFF_FALL_PENALTY)
        - _lower_ledge_penalty_array(result)
    )
    bilateral_score = jnp.where(
        result.com_x_went_negative,
        -jnp.float32(LEFT_EDGE_PENALTY),
        bilateral_score,
    )
    progress_shaping = _progress_shaping_array(result)
    safe_edge_arrival = _safe_edge_gate_array(result)
    safe_arrival_bonus = jnp.where(
        safe_edge_arrival,
        jnp.float32(SAFE_ARRIVAL_BONUS),
        0.0,
    )
    safe_dwell_bonus = jnp.where(
        safe_edge_arrival,
        result.cliff_vicinity_fraction.astype(jnp.float32)
        * jnp.float32(SAFE_DWELL_BONUS_SCALE),
        0.0,
    )
    urgency_shaping = _urgency_bonus_array(result)
    score = progress_shaping + safe_arrival_bonus + safe_dwell_bonus + urgency_shaping
    score = jnp.where(
        result.com_x_went_negative,
        -jnp.float32(LEFT_EDGE_PENALTY),
        score,
    )
    legacy_score = jnp.where(
        result.went_airborne,
        0.0,
        score,
    )
    return jnp.where(bilateral, bilateral_score, legacy_score)


def episode_descriptors(result: EpisodeResult) -> tuple[float, float]:
    """Extract QD descriptors from a single episode.

    Returns:
        (activation_cost, movement_speed)
    """
    activation_cost = float(result.mean_activation)
    movement_speed = float(result.mean_speed) / CELL_SIZE
    return activation_cost, movement_speed


def episode_descriptors_array(result: EpisodeResult) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Vectorized QD descriptors for one or more episodes."""
    activation_cost = result.mean_activation.astype(jnp.float32)
    movement_speed = result.mean_speed.astype(jnp.float32) / jnp.float32(CELL_SIZE)
    return activation_cost, movement_speed


def episode_peak_progress_rate(result: EpisodeResult) -> float:
    """Compute normalized clipped peak route progress per elapsed budget."""
    if int(result.task_mode) == _TASK_MODE_BILATERAL_GOAL_CLIFF:
        elapsed_time_fraction = max(float(result.elapsed_time_fraction), 1e-6)
        return min(
            max(float(result.toward_goal_frontier) / elapsed_time_fraction, 0.0),
            MAX_PEAK_PROGRESS_RATE,
        )
    edge_x = (float(result.cliff_x) + 1.0) * CELL_SIZE
    init_front_x = float(result.init_front_x)
    route_length = max(edge_x - init_front_x, 1e-6)
    clipped_peak_x = min(max(float(result.peak_front_x), init_front_x), edge_x)
    peak_progress_fraction = (clipped_peak_x - init_front_x) / route_length
    elapsed_time_fraction = max(float(result.elapsed_time_fraction), 1e-6)
    return min(max(peak_progress_fraction / elapsed_time_fraction, 0.0), MAX_PEAK_PROGRESS_RATE)


def episode_peak_progress_rate_array(result: EpisodeResult) -> jnp.ndarray:
    """Vectorized normalized clipped peak route progress per elapsed budget."""
    bilateral = result.task_mode.astype(jnp.int32) == jnp.int32(_TASK_MODE_BILATERAL_GOAL_CLIFF)
    edge_x = (result.cliff_x.astype(jnp.float32) + 1.0) * jnp.float32(CELL_SIZE)
    init_front_x = result.init_front_x.astype(jnp.float32)
    route_length = jnp.maximum(edge_x - init_front_x, jnp.float32(1e-6))
    clipped_peak_x = jnp.clip(result.peak_front_x.astype(jnp.float32), init_front_x, edge_x)
    peak_progress_fraction = (clipped_peak_x - init_front_x) / route_length
    elapsed_time_fraction = jnp.maximum(result.elapsed_time_fraction.astype(jnp.float32), jnp.float32(1e-6))
    legacy_rate = jnp.clip(
        peak_progress_fraction / elapsed_time_fraction,
        jnp.float32(0.0),
        jnp.float32(MAX_PEAK_PROGRESS_RATE),
    )
    bilateral_rate = jnp.clip(
        result.toward_goal_frontier.astype(jnp.float32) / elapsed_time_fraction,
        jnp.float32(0.0),
        jnp.float32(MAX_PEAK_PROGRESS_RATE),
    )
    return jnp.where(bilateral, bilateral_rate, legacy_rate)


def candidate_fitness(
    episode_results: list[EpisodeResult],
    *,
    alive_cell_count: float = 0.0,
    activation_cost_penalty_scale: float = 0.0,
    soft_loft_penalty_scale: float = 0.0,
    gait_spectral_purity: float = 0.0,
    gait_spectral_purity_bonus_scale: float = 0.0,
) -> CandidateResult:
    """Aggregate fitness and descriptors across episodes.

    Fitness = mean(episode_fitness).
    Descriptors = mean(per-episode energy and speed).
    """
    fitnesses = []
    activations = []
    h_activations = []
    v_activations = []
    movement_speeds = []
    peak_progress_rates = []
    times_to_edge = []
    correct_progresses = []
    incorrect_progresses = []
    toward_goal_frontiers = []
    reversal_progresses = []
    turn_counts = []
    stable_turn_counts = []
    turn_motion_thresholds = []
    lower_ledge_penalties = []
    lofted_fractions = []
    mean_lofted_excesses = []
    peak_com_y_deltas = []
    cliff_xs = []
    fells = []
    airbornes = []
    left_edges = []
    steps = []

    for r in episode_results:
        f = episode_fitness(r)
        a, s = episode_descriptors(r)
        fitnesses.append(f)
        activations.append(a)
        h_activations.append(float(r.mean_h_activation))
        v_activations.append(float(r.mean_v_activation))
        movement_speeds.append(s)
        peak_progress_rates.append(episode_peak_progress_rate(r))
        times_to_edge.append(float(r.time_to_goal if int(r.task_mode) == _TASK_MODE_BILATERAL_GOAL_CLIFF else r.time_to_edge))
        correct_progresses.append(float(r.toward_goal_frontier))
        incorrect_progresses.append(float(r.incorrect_progress))
        toward_goal_frontiers.append(float(r.toward_goal_frontier))
        reversal_progresses.append(float(r.reversal_progress))
        turn_counts.append(float(r.turn_count))
        stable_turn_counts.append(float(r.stable_turn_count))
        turn_motion_thresholds.append(float(r.turn_motion_threshold))
        lower_ledge_penalties.append(_lower_ledge_penalty_scalar(r))
        lofted_fractions.append(float(r.lofted_fraction))
        mean_lofted_excesses.append(float(r.mean_lofted_excess_cells))
        peak_com_y_deltas.append(float(r.peak_com_y_delta_cells))
        cliff_xs.append(int(r.cliff_x))
        fells.append(bool(r.cliff_fell))
        airbornes.append(bool(r.went_airborne))
        left_edges.append(bool(r.com_x_went_negative))
        steps.append(int(r.steps_alive))

    n = len(fitnesses)
    mirrored_pair_count = _mirrored_pair_count_scalar(episode_results)
    mirrored_best_hit_count = _mirrored_best_hit_count_scalar(episode_results)
    mirrored_total_hit_count = _mirrored_total_hit_count_scalar(episode_results)
    mean_episode_fitness = sum(fitnesses) / max(n, 1)
    mean_activation_cost = sum(activations) / max(n, 1)
    del activation_cost_penalty_scale
    loft_penalty = (
        float(soft_loft_penalty_scale)
        * (sum(mean_lofted_excesses) / max(n, 1))
    )
    gait_bonus = float(gait_spectral_purity_bonus_scale) * float(gait_spectral_purity)
    return CandidateResult(
        fitness=(
            mean_episode_fitness
            + (mirrored_pair_count * MIRRORED_PAIR_BONUS)
            - loft_penalty
            + gait_bonus
        ),
        fall_count=sum(float(bool(r.cliff_fell)) for r in episode_results),
        goal_hit_count=sum(float(bool(r.goal_touched)) for r in episode_results),
        cliff_vicinity=sum(float(r.cliff_vicinity_fraction) for r in episode_results) / max(n, 1),
        alive_cell_count=float(alive_cell_count),
        activation_cost=mean_activation_cost,
        h_activation=sum(h_activations) / max(n, 1),
        v_activation=sum(v_activations) / max(n, 1),
        movement_speed=sum(movement_speeds) / max(n, 1),
        peak_progress_rate=sum(peak_progress_rates) / max(n, 1),
        time_to_edge=sum(times_to_edge) / max(n, 1),
        correct_progress=sum(correct_progresses) / max(n, 1),
        incorrect_progress=sum(incorrect_progresses) / max(n, 1),
        correct_progress_top2_mean=_top_two_mean_scalar(correct_progresses),
        incorrect_progress_top2_mean=_top_two_mean_scalar(incorrect_progresses),
        toward_goal_frontier=sum(toward_goal_frontiers) / max(n, 1),
        turn_count=sum(turn_counts) / max(n, 1),
        mirrored_pair_count=mirrored_pair_count,
        n_episodes=n,
        episode_fitnesses=fitnesses,
        episode_cliff_xs=cliff_xs,
        episode_fell=fells,
        episode_airborne=airbornes,
        episode_left_edge=left_edges,
        episode_steps_alive=steps,
        lower_ledge_penalty=sum(lower_ledge_penalties) / max(n, 1),
        mirrored_best_hit_count=mirrored_best_hit_count,
        mirrored_total_hit_count=mirrored_total_hit_count,
        mirrored_roundtrip_top2_mean=_mirrored_roundtrip_top2_mean_scalar(episode_results),
        reversal_progress=sum(reversal_progresses) / max(n, 1),
        stable_turn_count=sum(stable_turn_counts) / max(n, 1),
        turn_motion_threshold=sum(turn_motion_thresholds) / max(n, 1),
        lofted_fraction=sum(lofted_fractions) / max(n, 1),
        mean_lofted_excess_cells=sum(mean_lofted_excesses) / max(n, 1),
        peak_com_y_delta_cells=sum(peak_com_y_deltas) / max(n, 1),
        gait_spectral_purity=float(gait_spectral_purity),
    )


def candidate_batch_fitness(
    episode_results: EpisodeResult,
    *,
    alive_cell_count: jnp.ndarray | None = None,
    activation_cost_penalty_scale: float = 0.0,
    soft_loft_penalty_scale: float = 0.0,
) -> CandidateBatchResult:
    """Aggregate fitness and descriptors across episodes for a candidate batch."""
    episode_fitnesses = episode_fitness_array(episode_results)
    episode_activation, episode_movement_speed = episode_descriptors_array(episode_results)
    episode_activation = episode_results.mean_activation.astype(jnp.float32)
    episode_h_activation = episode_results.mean_h_activation.astype(jnp.float32)
    episode_v_activation = episode_results.mean_v_activation.astype(jnp.float32)
    episode_peak_progress = episode_peak_progress_rate_array(episode_results)
    episode_lower_ledge_penalty = _lower_ledge_penalty_array(episode_results)
    mirrored_pair_count = _mirrored_pair_count_array(episode_results)
    mirrored_best_hit_count = _mirrored_best_hit_count_array(episode_results)
    mirrored_total_hit_count = _mirrored_total_hit_count_array(episode_results)
    episode_reversal_progress = jnp.broadcast_to(
        jnp.asarray(episode_results.reversal_progress, dtype=jnp.float32),
        episode_results.task_mode.shape,
    )
    episode_stable_turn_count = jnp.broadcast_to(
        jnp.asarray(episode_results.stable_turn_count, dtype=jnp.int32),
        episode_results.task_mode.shape,
    )
    episode_turn_motion_threshold = jnp.broadcast_to(
        jnp.asarray(episode_results.turn_motion_threshold, dtype=jnp.float32),
        episode_results.task_mode.shape,
    )
    episode_first_significant_direction = jnp.broadcast_to(
        jnp.asarray(episode_results.first_significant_direction, dtype=jnp.int32),
        episode_results.task_mode.shape,
    )
    episode_lofted_fraction = jnp.broadcast_to(
        jnp.asarray(episode_results.lofted_fraction, dtype=jnp.float32),
        episode_results.task_mode.shape,
    )
    episode_mean_lofted_excess_cells = jnp.broadcast_to(
        jnp.asarray(episode_results.mean_lofted_excess_cells, dtype=jnp.float32),
        episode_results.task_mode.shape,
    )
    episode_peak_com_y_delta_cells = jnp.broadcast_to(
        jnp.asarray(episode_results.peak_com_y_delta_cells, dtype=jnp.float32),
        episode_results.task_mode.shape,
    )
    episode_effective_times_to_edge = jnp.where(
        episode_results.task_mode.astype(jnp.int32) == jnp.int32(_TASK_MODE_BILATERAL_GOAL_CLIFF),
        episode_results.time_to_goal.astype(jnp.float32),
        episode_results.time_to_edge.astype(jnp.float32),
    )
    mean_episode_fitness = jnp.mean(episode_fitnesses, axis=-1)
    mean_activation_cost = jnp.mean(episode_activation, axis=-1)
    mean_lofted_fraction = jnp.mean(
        episode_lofted_fraction,
        axis=-1,
    )
    mean_lofted_excess_cells = jnp.mean(
        episode_mean_lofted_excess_cells,
        axis=-1,
    )
    mean_peak_com_y_delta_cells = jnp.mean(
        episode_peak_com_y_delta_cells,
        axis=-1,
    )
    del activation_cost_penalty_scale
    if alive_cell_count is None:
        alive_cell_count = jnp.zeros_like(jnp.mean(episode_fitnesses, axis=-1))
    return CandidateBatchResult(
        fitness=(
            mean_episode_fitness
            + (mirrored_pair_count * jnp.float32(MIRRORED_PAIR_BONUS))
            - (jnp.float32(soft_loft_penalty_scale) * mean_lofted_excess_cells)
        ),
        fall_count=_count_true(episode_results.cliff_fell, axis=-1).astype(jnp.float32),
        goal_hit_count=_count_true(episode_results.goal_touched, axis=-1).astype(jnp.float32),
        cliff_vicinity=jnp.mean(episode_results.cliff_vicinity_fraction.astype(jnp.float32), axis=-1),
        alive_cell_count=alive_cell_count.astype(jnp.float32),
        activation_cost=mean_activation_cost,
        h_activation=jnp.mean(episode_h_activation, axis=-1),
        v_activation=jnp.mean(episode_v_activation, axis=-1),
        movement_speed=jnp.mean(episode_movement_speed, axis=-1),
        peak_progress_rate=jnp.mean(episode_peak_progress, axis=-1),
        time_to_edge=jnp.mean(
            episode_effective_times_to_edge,
            axis=-1,
        ),
        correct_progress=jnp.mean(episode_results.toward_goal_frontier.astype(jnp.float32), axis=-1),
        incorrect_progress=jnp.mean(episode_results.incorrect_progress.astype(jnp.float32), axis=-1),
        correct_progress_top2_mean=_top_two_mean_array(
            episode_results.toward_goal_frontier.astype(jnp.float32)
        ),
        incorrect_progress_top2_mean=_top_two_mean_array(
            episode_results.incorrect_progress.astype(jnp.float32)
        ),
        toward_goal_frontier=jnp.mean(
            episode_results.toward_goal_frontier.astype(jnp.float32),
            axis=-1,
        ),
        turn_count=jnp.mean(episode_results.turn_count.astype(jnp.float32), axis=-1),
        mirrored_pair_count=mirrored_pair_count,
        episode_fitnesses=episode_fitnesses,
        episode_activation_costs=episode_activation,
        episode_h_activations=episode_h_activation,
        episode_v_activations=episode_v_activation,
        episode_movement_speeds=episode_movement_speed,
        episode_peak_progress_rates=episode_peak_progress,
        episode_times_to_edge=episode_effective_times_to_edge,
        episode_cliff_vicinity_steps=episode_results.cliff_vicinity_steps.astype(jnp.int32),
        episode_cliff_vicinity_fractions=episode_results.cliff_vicinity_fraction.astype(jnp.float32),
        episode_init_front_xs=episode_results.init_front_x.astype(jnp.float32),
        episode_peak_front_xs=episode_results.peak_front_x.astype(jnp.float32),
        episode_safe_front_xs=episode_results.safe_front_x.astype(jnp.float32),
        episode_probe_extents=jnp.maximum(
            episode_results.peak_front_x.astype(jnp.float32)
            - episode_results.safe_front_x.astype(jnp.float32),
            0.0,
        ),
        episode_cliff_xs=episode_results.cliff_x.astype(jnp.int32),
        episode_any_progresses=episode_results.any_progress.astype(jnp.float32),
        episode_goal_progresses=episode_results.goal_progress.astype(jnp.float32),
        episode_incorrect_progresses=episode_results.incorrect_progress.astype(jnp.float32),
        episode_toward_goal_frontiers=episode_results.toward_goal_frontier.astype(jnp.float32),
        episode_goal_touched=episode_results.goal_touched.astype(jnp.bool_),
        episode_cliff_fell=episode_results.cliff_fell.astype(jnp.bool_),
        episode_goal_directions=episode_results.goal_direction.astype(jnp.int32),
        episode_goal_distances=episode_results.goal_distance.astype(jnp.int32),
        episode_cliff_distances=episode_results.cliff_distance.astype(jnp.int32),
        episode_times_to_goal=episode_results.time_to_goal.astype(jnp.float32),
        episode_turn_counts=episode_results.turn_count.astype(jnp.int32),
        episode_fell=episode_results.cliff_fell.astype(jnp.bool_),
        episode_airborne=episode_results.went_airborne.astype(jnp.bool_),
        episode_left_edge=episode_results.com_x_went_negative.astype(jnp.bool_),
        episode_steps_alive=episode_results.steps_alive.astype(jnp.int32),
        deformation_channel_max=jnp.max(
            episode_results.deformation_channel_max.astype(jnp.float32),
            axis=-2,
        ),
        controller_hidden_output_max_abs=jnp.max(
            episode_results.controller_hidden_output_max_abs.astype(jnp.float32),
            axis=-1,
        ),
        action_output_max_abs=jnp.max(
            episode_results.action_output_max_abs.astype(jnp.float32),
            axis=-1,
        ),
        lower_ledge_penalty=jnp.mean(episode_lower_ledge_penalty, axis=-1),
        mirrored_best_hit_count=mirrored_best_hit_count,
        mirrored_total_hit_count=mirrored_total_hit_count,
        mirrored_roundtrip_top2_mean=_mirrored_roundtrip_top2_mean_array(episode_results),
        reversal_progress=jnp.mean(episode_reversal_progress, axis=-1),
        episode_reversal_progresses=episode_reversal_progress,
        stable_turn_count=jnp.mean(
            episode_stable_turn_count.astype(jnp.float32),
            axis=-1,
        ),
        episode_stable_turn_counts=episode_stable_turn_count,
        turn_motion_threshold=jnp.mean(
            episode_turn_motion_threshold.astype(jnp.float32),
            axis=-1,
        ),
        episode_turn_motion_thresholds=episode_turn_motion_threshold,
        episode_first_significant_directions=episode_first_significant_direction,
        lofted_fraction=mean_lofted_fraction,
        mean_lofted_excess_cells=mean_lofted_excess_cells,
        peak_com_y_delta_cells=mean_peak_com_y_delta_cells,
        episode_lofted_fractions=episode_lofted_fraction,
        episode_mean_lofted_excess_cells=episode_mean_lofted_excess_cells,
        episode_peak_com_y_delta_cells=episode_peak_com_y_delta_cells,
        gait_spectral_purity=jnp.zeros_like(mean_episode_fitness),
    )
