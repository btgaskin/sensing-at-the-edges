"""Held-out archive evaluation and post-train rerank/eval helpers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

from jax_evogym.constants import CELL_SIZE

from .batch_eval import (
    GrownEvalAssets,
    evaluate_population_chunked,
    planned_warmup_chunk_sizes,
    split_key_sequence,
    warmup_population_evaluator,
)
from .checkpoint import load_checkpoint
from .collision_tuning import resolve_collision_tuning_from_terrains
from .env import (
    CliffEnvData,
    build_static_collider_template,
    assemble_cliff_env,
    build_cliff_dynamic_core,
    build_cliff_terrain_assets,
    infer_static_collider_capacity,
    infer_static_template_capacity,
    precompute_shared_grid_data,
    stack_static_templates,
)
from .fitness import (
    PROGRESS_SHAPING_SCALE,
    SAFE_ARRIVAL_BONUS,
    SAFE_DWELL_BONUS_SCALE,
    URGENCY_SHAPING_SCALE,
)
from .logging_utils import configure_logging
from .render import RenderArtifact, render_ranked_elites, resolve_render_limit
from .runtime_config import (
    RuntimeResolution,
    attach_collision_tuning,
    emitter_kwargs_from_cfg,
    genotype_dim_from_cfg,
    logical_config_sha256,
    resolved_spawn_x,
    run_metadata,
    resolve_runtime_config,
    total_population,
)
from .terrain import (
    build_bilateral_goal_cliff_terrain,
    build_symmetric_double_cliff_terrain,
    fixed_body_alive_mask,
    make_robot_body,
    max_canvas_size,
)
from .terrain_schedule import (
    TerrainEntry,
    TerrainSchedule,
    load_schedule,
    schedule_variant_matches,
    validate_bilateral_schedule_semantics,
)

log = logging.getLogger(__name__)


class EvalResults(NamedTuple):
    """Backward-compatible held-out evaluation summary."""

    selected_elite_idx: int
    selected_elite_fitness_val: float
    test_mean_fitness: float
    test_median_fitness: float
    test_ci95_low: float
    test_ci95_high: float
    test_per_terrain: list[dict]
    n_test_terrains: int


@dataclass(frozen=True)
class PhaseContext:
    """Resolved config, runtime, and environments for one evaluation phase."""

    cfg: DictConfig
    runtime_resolution: RuntimeResolution
    envs: list[CliffEnvData] | GrownEvalAssets
    alive_mask: jnp.ndarray
    static_capacity: Any
    entries: list[TerrainEntry]
    schedule: TerrainSchedule | None
    schedule_dir: str


def _env_count(envs: list[CliffEnvData] | GrownEvalAssets) -> int:
    if isinstance(envs, GrownEvalAssets):
        return int(envs.static_templates.cliff_x.shape[0])
    return len(envs)


@dataclass(frozen=True)
class PostTrainEvalResults:
    """Analysis-ready post-train rerank/eval outputs."""

    run_metadata: dict[str, object]
    test_summary: dict[str, float]
    selected_elite_rank: int
    selected_elite_flat_index: int
    selected_elite_fitness_rerank: float
    test_mean_fitness: float
    test_median_fitness: float
    test_ci95_low: float
    test_ci95_high: float
    rerank_scores: list[dict[str, object]]
    test_per_terrain: list[dict[str, object]]
    n_rerank_terrains: int
    n_test_terrains: int
    rerank_schedule: dict[str, object]
    eval_schedule: dict[str, object]
    rerank_artifacts: list[RenderArtifact] = field(default_factory=list)
    eval_artifacts: list[RenderArtifact] = field(default_factory=list)
    saved_elites: list[dict[str, object]] = field(default_factory=list)
    deformation_suite: dict[str, object] | None = None
    selected_elite_genotype: np.ndarray | None = field(default=None, repr=False, compare=False)

    def as_json_dict(self) -> dict[str, object]:
        """Serialize results for durable storage."""
        return {
            "run_metadata": self.run_metadata,
            "test_summary": self.test_summary,
            "selected_elite_rank": int(self.selected_elite_rank),
            "selected_elite_flat_index": int(self.selected_elite_flat_index),
            "selected_elite_fitness_rerank": float(self.selected_elite_fitness_rerank),
            "test_mean_fitness": float(self.test_mean_fitness),
            "test_median_fitness": float(self.test_median_fitness),
            "test_ci95_low": float(self.test_ci95_low),
            "test_ci95_high": float(self.test_ci95_high),
            "rerank_scores": self.rerank_scores,
            "test_per_terrain": self.test_per_terrain,
            "n_rerank_terrains": int(self.n_rerank_terrains),
            "n_test_terrains": int(self.n_test_terrains),
            "rerank_schedule": self.rerank_schedule,
            "eval_schedule": self.eval_schedule,
            "rerank_artifacts": [item._asdict() for item in self.rerank_artifacts],
            "eval_artifacts": [item._asdict() for item in self.eval_artifacts],
            "saved_elites": self.saved_elites,
            "deformation_suite": _serialize_deformation_suite(self.deformation_suite),
        }


@dataclass(frozen=True)
class RobustnessProbeResults:
    """Analysis-ready outputs for the symmetric double-cliff robustness probe."""

    run_metadata: dict[str, object]
    probe_metadata: dict[str, object]
    leading_elite_flat_index: int
    elite_count: int
    aggregate_summary: dict[str, object]
    per_elite: list[dict[str, object]]
    artifacts: list[RenderArtifact] = field(default_factory=list)

    @property
    def selected_elite_flat_index(self) -> int:
        return int(self.leading_elite_flat_index)

    @property
    def death_rate(self) -> float:
        return float(self.aggregate_summary.get("mean_death_rate", 0.0))

    @property
    def mean_steps_alive(self) -> float:
        return float(self.aggregate_summary.get("mean_mean_steps_alive", 0.0))

    @property
    def n_probe_terrains(self) -> int:
        return int(self.aggregate_summary.get("n_probe_terrains", 0))

    @property
    def per_distance(self) -> list[dict[str, object]]:
        if not self.per_elite:
            return []
        return list(self.per_elite[0].get("per_distance", []))

    def as_json_dict(self) -> dict[str, object]:
        return {
            "run_metadata": self.run_metadata,
            "probe_metadata": self.probe_metadata,
            "leading_elite_flat_index": int(self.leading_elite_flat_index),
            "elite_count": int(self.elite_count),
            "aggregate_summary": self.aggregate_summary,
            "per_elite": self.per_elite,
            "artifacts": [item._asdict() for item in self.artifacts],
        }


@dataclass(frozen=True)
class GapGeneralizationSuiteResults:
    """Analysis-ready outputs for the plateau-gap width generalization sweep."""

    run_metadata: dict[str, object]
    suite_metadata: dict[str, object]
    selected_elite_flat_index: int
    surface_summaries: list[dict[str, object]]
    overall_summary: dict[str, object]
    per_terrain_rows: list[dict[str, object]]
    artifacts: list[RenderArtifact] = field(default_factory=list)

    def as_json_dict(self) -> dict[str, object]:
        return {
            "run_metadata": self.run_metadata,
            "suite_metadata": self.suite_metadata,
            "selected_elite_flat_index": int(self.selected_elite_flat_index),
            "surface_summaries": self.surface_summaries,
            "overall_summary": self.overall_summary,
            "per_terrain_rows": self.per_terrain_rows,
            "artifacts": [item._asdict() for item in self.artifacts],
        }


def _serialize_deformation_suite(
    suite: dict[str, object] | None,
) -> dict[str, object] | None:
    """Convert deformation-suite render artifacts into JSON-ready payloads."""
    if suite is None:
        return None
    payload = dict(suite)
    render_artifacts = payload.get("render_artifacts")
    if isinstance(render_artifacts, dict):
        payload["render_artifacts"] = {
            str(variant): [item._asdict() for item in artifacts]
            for variant, artifacts in render_artifacts.items()
        }
    return payload


def summarize_test_per_terrain(test_per_terrain: list[dict[str, object]]) -> dict[str, float]:
    """Aggregate per-terrain eval signals into analysis-ready summary metrics."""
    if not test_per_terrain:
        return {
            "mean_progress_shaping": 0.0,
            "mean_any_progress": 0.0,
            "mean_goal_progress": 0.0,
            "mean_correct_progress": 0.0,
            "mean_incorrect_progress": 0.0,
            "mean_toward_goal_frontier": 0.0,
            "mean_reversal_progress": 0.0,
            "mean_safe_arrival_bonus": 0.0,
            "mean_safe_dwell_bonus": 0.0,
            "mean_urgency_shaping": 0.0,
            "mean_fall_count": 0.0,
            "goal_touch_rate": 0.0,
            "edge_entry_rate": 0.0,
            "mean_cliff_vicinity_fraction": 0.0,
            "mean_probe_extent": 0.0,
            "mean_time_to_edge": 0.0,
            "mean_time_to_goal": 0.0,
            "mean_turn_count": 0.0,
            "mean_stable_turn_count": 0.0,
            "mean_turn_motion_threshold": 0.0,
            "mean_activation_cost": 0.0,
            "mean_movement_speed": 0.0,
        }

    def _mean(key: str) -> float:
        return float(np.mean([float(item.get(key, 0.0)) for item in test_per_terrain]))

    def _rate(key: str) -> float:
        return float(np.mean([1.0 if bool(item.get(key, False)) else 0.0 for item in test_per_terrain]))

    def _mean_fall_count() -> float:
        return float(
            np.mean(
                [
                    float(item.get("fall_count", 1.0 if bool(item.get("fell", False)) else 0.0))
                    for item in test_per_terrain
                ]
            )
        )

    return {
        "mean_progress_shaping": _mean("progress_shaping"),
        "mean_any_progress": _mean("any_progress"),
        "mean_goal_progress": _mean("goal_progress"),
        "mean_correct_progress": _mean("correct_progress"),
        "mean_incorrect_progress": _mean("incorrect_progress"),
        "mean_toward_goal_frontier": _mean("toward_goal_frontier"),
        "mean_reversal_progress": _mean("reversal_progress"),
        "mean_safe_arrival_bonus": _mean("safe_arrival_bonus"),
        "mean_safe_dwell_bonus": _mean("safe_dwell_bonus"),
        "mean_urgency_shaping": _mean("urgency_shaping"),
        "mean_fall_count": _mean_fall_count(),
        "goal_touch_rate": _rate("goal_touched"),
        "edge_entry_rate": _rate("entered_cliff_band"),
        "mean_cliff_vicinity_fraction": _mean("cliff_vicinity_fraction"),
        "mean_probe_extent": _mean("probe_extent"),
        "mean_time_to_edge": _mean("time_to_edge"),
        "mean_time_to_goal": _mean("time_to_goal"),
        "mean_turn_count": _mean("turn_count"),
        "mean_stable_turn_count": _mean("stable_turn_count"),
        "mean_turn_motion_threshold": _mean("turn_motion_threshold"),
        "mean_activation_cost": _mean("activation_cost"),
        "mean_movement_speed": _mean("movement_speed"),
    }


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap confidence interval on the median."""
    rng = np.random.default_rng(seed)
    n = len(values)
    medians = np.array([
        np.median(rng.choice(values, size=n, replace=True))
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(medians, 100 * alpha / 2))
    hi = float(np.percentile(medians, 100 * (1 - alpha / 2)))
    return lo, hi


def _build_envs_from_entries(
    entries: list[TerrainEntry],
    dynamic_core,
    robot_morph: np.ndarray,
    cfg: DictConfig,
    *,
    full_build_capacity,
) -> list[CliffEnvData]:
    """Build CliffEnvData for each terrain entry."""
    spawn_x = resolved_spawn_x(cfg)
    spawn_y = int(cfg.terrain.spawn_y)
    return [
        assemble_cliff_env(
            dynamic_core,
            build_cliff_terrain_assets(
                e.grid,
                robot_morph,
                spawn_x,
                spawn_y,
                e.cliff_x,
                task_metadata=e.metadata,
                spring_stiffness_scale=float(cfg.sim.spring_stiffness_scale),
                collision_build_capacity=full_build_capacity,
            ),
        )
        for e in entries
    ]


def _resolve_limit(value: int | str | None, total: int, *, allow_zero: bool = False) -> int:
    if total <= 0 or value is None:
        return 0
    if isinstance(value, str) and value.strip().lower() == "all":
        return int(total)
    limit = int(value)
    if allow_zero:
        limit = max(0, limit)
    else:
        limit = max(1, limit)
    return min(limit, int(total))


def _schedule_metadata(schedule_dir: str | Path, schedule: TerrainSchedule, split: str) -> dict[str, object]:
    return {
        "dir": str(schedule_dir),
        "split": split,
        "hash_version": int(schedule.hash_version),
        "seed": int(schedule.seed),
        "n_gens": len(schedule.train),
        "n_episodes": len(schedule.train[0]) if schedule.train else 0,
        "n_val": len(schedule.val),
        "n_test": len(schedule.test),
        "train_hash": schedule.train_hash,
        "val_hash": schedule.val_hash,
        "test_hash": schedule.test_hash,
        "train_content_hash": schedule.train_content_hash,
        "val_content_hash": schedule.val_content_hash,
        "test_content_hash": schedule.test_content_hash,
        "terrain_cfg": dict(schedule.terrain_cfg),
    }


def _validate_loaded_schedule(
    schedule_dir: str | Path,
    schedule: TerrainSchedule,
    *,
    expected_seed: int,
    expected_variant: str,
    expected_val: int | None = None,
    expected_test: int | None = None,
) -> None:
    mismatches: list[str] = []
    schedule_variant = str(schedule.terrain_cfg.get("variant", ""))
    if int(schedule.seed) != int(expected_seed):
        mismatches.append(f"seed expected {expected_seed}, got {schedule.seed}")
    if not schedule_variant_matches(str(expected_variant), schedule_variant):
        mismatches.append(f"variant expected {expected_variant!r}, got {schedule_variant!r}")
    if expected_val is not None and len(schedule.val) != int(expected_val):
        mismatches.append(f"n_val expected {expected_val}, got {len(schedule.val)}")
    if expected_test is not None and len(schedule.test) != int(expected_test):
        mismatches.append(f"n_test expected {expected_test}, got {len(schedule.test)}")
    if mismatches:
        raise ValueError(
            f"Terrain schedule metadata mismatch for {schedule_dir}: "
            + "; ".join(mismatches)
        )


def _phase_cfg_from_schedule(
    cfg: DictConfig,
    schedule: TerrainSchedule,
    *,
    max_steps: int | None,
    variant_override: str | None = None,
) -> DictConfig:
    terrain_cfg = dict(schedule.terrain_cfg)
    phase_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    phase_cfg.terrain.grid_h = int(terrain_cfg["grid_h"])
    phase_cfg.terrain.grid_w = int(terrain_cfg["grid_w"])
    phase_cfg.terrain.platform_y = int(
        terrain_cfg.get("platform_y", terrain_cfg.get("plateau_height", phase_cfg.terrain.platform_y))
    )
    phase_cfg.terrain.base_width = int(terrain_cfg.get("base_width", phase_cfg.terrain.base_width))
    phase_cfg.terrain.cliff_x_min = int(terrain_cfg["cliff_x_min"])
    phase_cfg.terrain.cliff_x_max = int(terrain_cfg["cliff_x_max"])
    phase_cfg.terrain.gap_widths = list(terrain_cfg.get("gap_widths") or [])
    phase_cfg.terrain.gap_budget = int(terrain_cfg.get("gap_budget", 0))
    phase_cfg.terrain.exact_gap_widths = list(terrain_cfg.get("exact_gap_widths") or [])
    phase_cfg.terrain.min_gap_separation = int(terrain_cfg.get("min_gap_separation", 2))
    phase_cfg.terrain.min_post_gap_runway = int(terrain_cfg.get("min_post_gap_runway", 2))
    phase_cfg.terrain.hazard_mode = str(terrain_cfg.get("hazard_mode", "legacy"))
    phase_cfg.terrain.task_mode = str(terrain_cfg.get("task_mode", "legacy_cliff"))
    phase_cfg.terrain.spawn_plateau_width = terrain_cfg.get("spawn_plateau_width")
    phase_cfg.terrain.approach_width = terrain_cfg.get("approach_width")
    phase_cfg.terrain.randomized_zone_width = terrain_cfg.get("randomized_zone_width")
    phase_cfg.terrain.remainder_width = terrain_cfg.get("remainder_width")
    phase_cfg.terrain.center_plateau_width = terrain_cfg.get("center_plateau_width")
    phase_cfg.terrain.symmetric_distance_min = terrain_cfg.get("symmetric_distance_min")
    phase_cfg.terrain.symmetric_distance_max = terrain_cfg.get("symmetric_distance_max")
    phase_cfg.terrain.correct_side_distance_scale = terrain_cfg.get("correct_side_distance_scale", 1.0)
    phase_cfg.terrain.incorrect_side_distance_scale = terrain_cfg.get("incorrect_side_distance_scale", 1.0)
    phase_cfg.terrain.distance_pairs = list(terrain_cfg.get("distance_pairs") or [])
    phase_cfg.terrain.cliff_front_gap_width = terrain_cfg.get("cliff_front_gap_width")
    phase_cfg.terrain.cliff_front_gap_widths = list(terrain_cfg.get("cliff_front_gap_widths") or [])
    phase_cfg.terrain.cliff_post_gap_runway = terrain_cfg.get("cliff_post_gap_runway")
    phase_cfg.terrain.cliff_probe_gap_width = terrain_cfg.get("cliff_probe_gap_width")
    phase_cfg.terrain.cliff_probe_ledge_width = terrain_cfg.get("cliff_probe_ledge_width")
    phase_cfg.terrain.single_gap_width = terrain_cfg.get("single_gap_width")
    phase_cfg.terrain.gap_start_min = terrain_cfg.get("gap_start_min")
    phase_cfg.terrain.gap_start_max = terrain_cfg.get("gap_start_max")
    phase_cfg.terrain.gap_start_bands = list(terrain_cfg.get("gap_start_bands") or [])
    phase_cfg.terrain.layout_sampling = str(
        terrain_cfg.get("layout_sampling", "uniform_legal_layouts")
    )
    if variant_override is not None:
        phase_cfg.terrain.variant = str(variant_override)
    if max_steps is not None:
        phase_cfg.train.max_steps = int(max_steps)
    return phase_cfg


def _phase_context_from_schedule(
    cfg: DictConfig,
    runtime_resolution: RuntimeResolution,
    schedule_dir: str | Path,
    schedule: TerrainSchedule,
    *,
    split: str,
    max_steps: int | None,
    limit: int,
    variant_override: str | None = None,
) -> PhaseContext:
    phase_cfg = _phase_cfg_from_schedule(
        cfg,
        schedule,
        max_steps=max_steps,
        variant_override=variant_override,
    )
    entries = list(getattr(schedule, split)[:limit])
    if not entries:
        raise ValueError(f"Schedule '{schedule_dir}' has no entries for split '{split}'")

    phase_resolution, envs, alive_mask, static_capacity = _build_phase_assets(
        phase_cfg,
        runtime_resolution,
        entries,
    )
    return PhaseContext(
        cfg=phase_cfg,
        runtime_resolution=phase_resolution,
        envs=envs,
        alive_mask=alive_mask,
        static_capacity=static_capacity,
        entries=entries,
        schedule=schedule,
        schedule_dir=str(schedule_dir),
    )


def _build_phase_assets(
    phase_cfg: DictConfig,
    runtime_resolution: RuntimeResolution,
    entries: list[TerrainEntry],
) -> tuple[RuntimeResolution, list[CliffEnvData] | GrownEvalAssets, jnp.ndarray, Any]:
    """Build evaluation env assets for a resolved config and terrain entries."""
    if not entries:
        raise ValueError("Phase asset construction requires at least one terrain entry.")

    body_mode = str(phase_cfg.body.mode).strip().lower()
    body_shape = str(phase_cfg.body.get("shape", "rectangle"))
    spawn_x = resolved_spawn_x(phase_cfg)
    spawn_y = int(phase_cfg.terrain.spawn_y)
    robot_morph = make_robot_body(
        phase_cfg.terrain.robot_w,
        phase_cfg.terrain.robot_h,
        shape=body_shape,
    )
    collision_tuning = resolve_collision_tuning_from_terrains(
        [np.asarray(entry.grid, dtype=np.int32) for entry in entries],
        grid_h=int(phase_cfg.terrain.grid_h),
        grid_w=int(phase_cfg.terrain.grid_w),
        robot_morph=(
            make_robot_body(int(phase_cfg.body.growth.max_robot_w), int(phase_cfg.body.growth.max_robot_h))
            if body_mode == "grown"
            else robot_morph
        ),
        spawn_x=spawn_x,
        spawn_y=spawn_y,
        mode=str(phase_cfg.collision.tuning.mode),
    )
    phase_resolution = attach_collision_tuning(runtime_resolution, collision_tuning)
    if body_mode == "grown":
        canvas_h = int(phase_cfg.body.growth.max_robot_h)
        canvas_w = int(phase_cfg.body.growth.max_robot_w)
        max_h, max_w = max_canvas_size(
            int(phase_cfg.terrain.grid_h),
            int(phase_cfg.terrain.grid_w),
            spawn_x=spawn_x,
            spawn_y=spawn_y,
        )
        if canvas_h > max_h or canvas_w > max_w:
            raise ValueError("Configured grown body canvas does not fit within the evaluation terrain")
        templates = [
            build_static_collider_template(
                entry.grid,
                cliff_x=int(entry.cliff_x),
                task_metadata=entry.metadata,
                canvas_h=canvas_h,
                canvas_w=canvas_w,
                spawn_x=spawn_x,
                spawn_y=spawn_y,
                spring_stiffness_scale=float(phase_cfg.sim.spring_stiffness_scale),
                collision_build_capacity=phase_resolution.resolved_collision.build_capacity,
            )
            for entry in entries
        ]
        static_capacity = infer_static_template_capacity(templates)
        envs = GrownEvalAssets(
            static_templates=stack_static_templates(templates, static_capacity=static_capacity),
            templates=tuple(templates),
            static_capacity=static_capacity,
            grid_data=precompute_shared_grid_data(
                int(phase_cfg.terrain.grid_h),
                int(phase_cfg.terrain.grid_w),
            ),
            runtime_build_capacity=phase_resolution.resolved_collision.runtime_build_capacity,
        )
        alive_mask = jnp.ones((canvas_h, canvas_w), dtype=jnp.bool_)
    else:
        dynamic_core = build_cliff_dynamic_core(
            robot_morph,
            spawn_x=spawn_x,
            spawn_y=spawn_y,
            physics_substeps=int(phase_cfg.sim.physics_substeps),
            actuator_convergence=float(phase_cfg.sim.actuator_convergence),
            spring_stiffness_scale=float(phase_cfg.sim.spring_stiffness_scale),
            floor_static_friction_const=float(phase_cfg.sim.floor_static_friction_const),
            floor_stiction_stiffness=float(phase_cfg.sim.floor_stiction_stiffness),
            floor_stiction_switch_speed=float(phase_cfg.sim.floor_stiction_switch_speed),
            collision_build_capacity=phase_resolution.resolved_collision.runtime_build_capacity,
        )
        envs = _build_envs_from_entries(
            entries,
            dynamic_core,
            robot_morph,
            phase_cfg,
            full_build_capacity=phase_resolution.resolved_collision.build_capacity,
        )
        static_capacity = infer_static_collider_capacity(envs)
        alive_mask = jnp.asarray(fixed_body_alive_mask(robot_morph), dtype=jnp.bool_)
    return phase_resolution, envs, alive_mask, static_capacity


def _center_spawn_x_for_plateau(*, plateau_start: int, plateau_width: int, robot_w: int) -> int:
    plateau_width = int(plateau_width)
    robot_w = int(robot_w)
    if robot_w <= 0:
        raise ValueError("robot_w must be positive for centered spawn placement.")
    if robot_w > plateau_width:
        raise ValueError(
            "Robustness probe requires the fixed body to fit within the centered plateau: "
            f"robot_w={robot_w}, plateau_width={plateau_width}"
        )
    return int(plateau_start + (plateau_width - robot_w) // 2)


def _probe_max_steps(cfg: DictConfig) -> int:
    probe_cfg = cfg.eval.get("robustness_probe")
    configured = probe_cfg.get("max_steps") if probe_cfg is not None else None
    if configured is not None:
        return int(configured)
    test_max_steps = cfg.eval.get("test_max_steps")
    if test_max_steps is not None:
        return int(test_max_steps)
    return int(cfg.train.max_steps)


def _gap_generalization_max_steps(cfg: DictConfig) -> int:
    suite_cfg = cfg.eval.get("gap_generalization_suite")
    configured = suite_cfg.get("max_steps") if suite_cfg is not None else None
    if configured is not None:
        return int(configured)
    test_max_steps = cfg.eval.get("test_max_steps")
    if test_max_steps is not None:
        return int(test_max_steps)
    return int(cfg.train.max_steps)


def _bilateral_distance_pairs_for_suite(cfg: DictConfig) -> list[tuple[int, int]]:
    raw_pairs = cfg.terrain.get("distance_pairs")
    if raw_pairs:
        normalized: list[tuple[int, int]] = []
        for pair in raw_pairs:
            if hasattr(pair, "keys") and "goal_distance" in pair and "cliff_distance" in pair:
                normalized.append((int(pair["goal_distance"]), int(pair["cliff_distance"])))
            else:
                normalized.append((int(pair[0]), int(pair[1])))
        return list(dict.fromkeys(normalized))

    distance_min = cfg.terrain.get("symmetric_distance_min")
    distance_max = cfg.terrain.get("symmetric_distance_max")
    if distance_min is None or distance_max is None:
        raise ValueError(
            "gap_generalization_suite requires terrain.distance_pairs or terrain.symmetric_distance_min/max."
        )
    correct_scale = float(cfg.terrain.get("correct_side_distance_scale", 1.0))
    incorrect_scale = float(cfg.terrain.get("incorrect_side_distance_scale", 1.0))
    return list(
        dict.fromkeys(
            (
                int(np.ceil(float(distance) * correct_scale)),
                int(np.ceil(float(distance) * incorrect_scale)),
            )
            for distance in range(int(distance_min), int(distance_max) + 1)
        )
    )


def _gap_suite_trial_conditions(
    cfg: DictConfig,
    *,
    trials_per_surface: int,
) -> list[tuple[int, int, str]]:
    base_conditions = [
        (int(goal_distance), int(cliff_distance), str(goal_side))
        for goal_distance, cliff_distance in _bilateral_distance_pairs_for_suite(cfg)
        for goal_side in ("left", "right")
    ]
    if not base_conditions:
        raise ValueError("gap_generalization_suite requires at least one bilateral condition.")
    if trials_per_surface <= 0:
        raise ValueError("gap_generalization_suite.trials_per_surface must be positive.")
    conditions: list[tuple[int, int, str]] = []
    while len(conditions) < int(trials_per_surface):
        for condition in base_conditions:
            if len(conditions) >= int(trials_per_surface):
                break
            conditions.append(condition)
    return conditions


def _symmetric_double_cliff_probe_context(
    cfg: DictConfig,
    runtime_resolution: RuntimeResolution,
) -> tuple[PhaseContext, list[dict[str, int]]]:
    probe_cfg = cfg.eval.get("robustness_probe")
    if probe_cfg is None:
        raise ValueError("eval.robustness_probe must be configured before running the probe.")
    if str(cfg.body.mode).strip().lower() != "fixed":
        raise ValueError("The symmetric double-cliff robustness probe currently supports body.mode='fixed' only.")

    grid_h = int(probe_cfg.get("grid_h", 10))
    grid_w = int(probe_cfg.get("grid_w", 30))
    plateau_width = int(probe_cfg.get("plateau_width", 10))
    platform_y = int(probe_cfg.get("platform_y", 3))
    min_distance = int(probe_cfg.get("min_distance", 1))
    max_distance = int(probe_cfg.get("max_distance", 6))
    center_start = (grid_w - plateau_width) // 2
    spawn_x = _center_spawn_x_for_plateau(
        plateau_start=center_start,
        plateau_width=plateau_width,
        robot_w=int(cfg.terrain.robot_w),
    )
    spawn_y = int(cfg.terrain.spawn_y)
    phase_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    phase_cfg.terrain.variant = str(probe_cfg.get("variant", "symmetric_double_cliff_probe"))
    phase_cfg.terrain.grid_h = grid_h
    phase_cfg.terrain.grid_w = grid_w
    phase_cfg.terrain.platform_y = platform_y
    phase_cfg.terrain.base_width = plateau_width
    phase_cfg.terrain.cliff_x_min = center_start + plateau_width - 1 + min_distance
    phase_cfg.terrain.cliff_x_max = center_start + plateau_width - 1 + max_distance
    phase_cfg.terrain.spawn_x = spawn_x
    phase_cfg.terrain.spawn_y = spawn_y
    phase_cfg.train.max_steps = _probe_max_steps(cfg)

    entries: list[TerrainEntry] = []
    metadata: list[dict[str, int]] = []
    for distance in range(min_distance, max_distance + 1):
        grid, platform_start, platform_end = build_symmetric_double_cliff_terrain(
            grid_h,
            grid_w,
            symmetric_distance=distance,
            plateau_width=plateau_width,
            platform_y=platform_y,
        )
        entries.append(TerrainEntry(grid=grid, cliff_x=int(platform_end), gaps=[]))
        metadata.append(
            {
                "distance": int(distance),
                "platform_start_x": int(platform_start),
                "platform_end_x": int(platform_end),
            }
        )

    phase_resolution, envs, alive_mask, static_capacity = _build_phase_assets(
        phase_cfg,
        runtime_resolution,
        entries,
    )
    return (
        PhaseContext(
            cfg=phase_cfg,
            runtime_resolution=phase_resolution,
            envs=envs,
            alive_mask=alive_mask,
            static_capacity=static_capacity,
            entries=entries,
            schedule=None,
            schedule_dir="",
        ),
        metadata,
    )


def _warmup_phase(
    ctx: PhaseContext,
    *,
    workload: str,
    candidate_count: int,
) -> None:
    batching = ctx.runtime_resolution.resolved_batching[workload]
    if not batching.warmup_enabled:
        return
    warmup_chunk_sizes = planned_warmup_chunk_sizes(candidate_count, batching)
    log.info(
        "Warmup | workload=%s terrains=%d planned=%s",
        workload,
        _env_count(ctx.envs),
        list(warmup_chunk_sizes),
    )
    warmup_timings = warmup_population_evaluator(
        genotype_dim_from_cfg(ctx.cfg),
        ctx.envs,
        ctx.alive_mask,
        ctx.cfg,
        workload=workload,
        runtime_resolution=ctx.runtime_resolution,
        static_capacity=ctx.static_capacity,
        candidate_count=candidate_count,
    )
    for bucket_size, elapsed_s in warmup_timings.items():
        log.info(
            "Warmup | workload=%s bucket=%d elapsed_s=%.2f",
            workload,
            bucket_size,
            elapsed_s,
        )


def _phase_context_with_cfg(ctx: PhaseContext, cfg: DictConfig) -> PhaseContext:
    """Reuse prebuilt environments with a config clone that changes only rollout knobs."""
    return PhaseContext(
        cfg=cfg,
        runtime_resolution=ctx.runtime_resolution,
        envs=ctx.envs,
        alive_mask=ctx.alive_mask,
        static_capacity=ctx.static_capacity,
        entries=ctx.entries,
        schedule=ctx.schedule,
        schedule_dir=ctx.schedule_dir,
    )


def _cfg_with_deformation_variant(cfg: DictConfig, variant: str) -> DictConfig:
    """Clone the eval config with one active deformation-suite variant."""
    variant_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if variant_cfg.eval.get("deformation_suite") is None:
        variant_cfg.eval.deformation_suite = {}
    variant_cfg.eval.deformation_suite.active_variant = str(variant)
    return variant_cfg


def _evaluate_elite_batch(
    elites: list[dict[str, np.ndarray | float | int]],
    ctx: PhaseContext,
    *,
    root_key: jax.Array,
    workload: str,
):
    if not elites:
        raise ValueError("Cannot evaluate an empty elite batch.")
    elite_genotypes = np.stack([np.asarray(elite["genotype"]) for elite in elites], axis=0)
    _warmup_phase(ctx, workload=workload, candidate_count=len(elites))
    _, candidate_keys = split_key_sequence(root_key, len(elites))
    return evaluate_population_chunked(
        elite_genotypes,
        ctx.envs,
        ctx.alive_mask,
        candidate_keys,
        ctx.cfg,
        workload=workload,
        runtime_resolution=ctx.runtime_resolution,
        static_capacity=ctx.static_capacity,
    )


def _score_elites_on_phase(
    elites: list[dict[str, np.ndarray | float | int]],
    ctx: PhaseContext,
    *,
    root_key: jax.Array,
    workload: str,
) -> list[dict[str, np.ndarray | float | int]]:
    if not elites:
        return []
    phase_eval = _evaluate_elite_batch(
        elites,
        ctx,
        root_key=root_key,
        workload=workload,
    )
    scored: list[dict[str, np.ndarray | float | int]] = []
    for idx, elite in enumerate(elites):
        elite_copy = dict(elite)
        elite_copy["selection_fitness"] = float(phase_eval.fitnesses[idx])
        elite_copy["archive_rank"] = int(idx)
        scored.append(elite_copy)
    scored.sort(
        key=lambda elite: (-float(elite["selection_fitness"]), int(elite["flat_index"])),
    )
    return scored


def _test_rows_for_candidate(
    batch,
    entries: list[TerrainEntry],
    *,
    candidate_idx: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Convert one candidate's held-out batch outputs into per-terrain rows."""
    def _episode_value(attr: str, idx: int, default: float = 0.0) -> float:
        values = getattr(batch, attr, None)
        if values is None:
            return float(default)
        return float(values[candidate_idx, idx])

    test_fitnesses = np.asarray(batch.episode_fitnesses[candidate_idx])
    test_per_terrain: list[dict[str, object]] = []
    for idx, entry in enumerate(entries):
        task_mode = str((entry.metadata or {}).get("task_mode", "legacy_cliff"))
        common = {
            "terrain_idx": int(idx),
            "fitness": float(batch.episode_fitnesses[candidate_idx, idx]),
            "activation_cost": float(batch.episode_activation_costs[candidate_idx, idx]),
            "h_activation": float(batch.episode_h_activations[candidate_idx, idx]),
            "v_activation": float(batch.episode_v_activations[candidate_idx, idx]),
            "movement_speed": float(batch.episode_movement_speeds[candidate_idx, idx]),
            "turn_motion_threshold": _episode_value(
                "episode_turn_motion_thresholds",
                idx,
            ),
            "cliff_x": int(entry.cliff_x),
            "gaps": [list(gap) for gap in entry.gaps],
        }
        if task_mode == "bilateral_goal_cliff":
            fell = bool(batch.episode_cliff_fell[candidate_idx, idx])
            goal_touched = bool(batch.episode_goal_touched[candidate_idx, idx])
            test_per_terrain.append(
                {
                    **common,
                    "task_mode": task_mode,
                    "fell": fell,
                    "fall_count": float(fell),
                    "goal_touched": goal_touched,
                    "goal_side": (entry.metadata or {}).get("goal_side"),
                    "goal_distance": int(batch.episode_goal_distances[candidate_idx, idx]),
                    "cliff_distance": int(batch.episode_cliff_distances[candidate_idx, idx]),
                    "correct_side_distance_scale": float((entry.metadata or {}).get("correct_side_distance_scale", 1.0)),
                    "incorrect_side_distance_scale": float((entry.metadata or {}).get("incorrect_side_distance_scale", 1.0)),
                    "goal_direction": int(batch.episode_goal_directions[candidate_idx, idx]),
                    "any_progress": float(batch.episode_any_progresses[candidate_idx, idx]),
                    "goal_progress": float(batch.episode_goal_progresses[candidate_idx, idx]),
                    "correct_progress": float(batch.episode_toward_goal_frontiers[candidate_idx, idx]),
                    "incorrect_progress": float(batch.episode_incorrect_progresses[candidate_idx, idx]),
                    "toward_goal_frontier": float(batch.episode_toward_goal_frontiers[candidate_idx, idx]),
                    "reversal_progress": float(batch.episode_reversal_progresses[candidate_idx, idx]),
                    "time_to_goal": float(batch.episode_times_to_goal[candidate_idx, idx]),
                    "turn_count": _episode_value("episode_turn_counts", idx),
                    "stable_turn_count": _episode_value(
                        "episode_stable_turn_counts",
                        idx,
                    ),
                    "time_to_edge": 0.0,
                    "entered_cliff_band": False,
                    "safe_edge_arrival": False,
                    "progress_shaping": 0.0,
                    "safe_arrival_bonus": 0.0,
                    "safe_dwell_bonus": 0.0,
                    "urgency_shaping": 0.0,
                    "cliff_vicinity_steps": 0,
                    "cliff_vicinity_fraction": 0.0,
                    "peak_front_x": float(batch.episode_peak_front_xs[candidate_idx, idx]),
                    "safe_front_x": float(batch.episode_safe_front_xs[candidate_idx, idx]),
                    "probe_extent": float(batch.episode_probe_extents[candidate_idx, idx]),
                    "airborne": bool(batch.episode_airborne[candidate_idx, idx]),
                    "left_edge": bool(batch.episode_left_edge[candidate_idx, idx]),
                }
            )
            continue

        init_front_x = float(batch.episode_init_front_xs[candidate_idx, idx])
        edge_x = float((int(entry.cliff_x) + 1) * CELL_SIZE)
        route_length = max(edge_x - init_front_x, 0.0)
        peak_front_x = float(batch.episode_peak_front_xs[candidate_idx, idx])
        safe_front_x = float(batch.episode_safe_front_xs[candidate_idx, idx])
        safe_progress = min(max(safe_front_x - init_front_x, 0.0), route_length)
        progress_shaping = ((safe_progress / route_length) * float(PROGRESS_SHAPING_SCALE)) if route_length > 0.0 else 0.0
        cliff_vicinity_fraction = float(batch.episode_cliff_vicinity_fractions[candidate_idx, idx])
        cliff_vicinity_steps = int(batch.episode_cliff_vicinity_steps[candidate_idx, idx])
        entered_cliff_band = bool(cliff_vicinity_steps > 0)
        fell = bool(batch.episode_fell[candidate_idx, idx])
        airborne = bool(batch.episode_airborne[candidate_idx, idx])
        left_edge = bool(batch.episode_left_edge[candidate_idx, idx])
        safe_edge_arrival = entered_cliff_band and not fell and not airborne and not left_edge
        safe_arrival_bonus = float(SAFE_ARRIVAL_BONUS) if safe_edge_arrival else 0.0
        safe_dwell_bonus = float(SAFE_DWELL_BONUS_SCALE) * cliff_vicinity_fraction if safe_edge_arrival else 0.0
        urgency_shaping = (
            float(URGENCY_SHAPING_SCALE) * max(1.0 - float(batch.episode_times_to_edge[candidate_idx, idx]), 0.0)
            if safe_edge_arrival else 0.0
        )
        test_per_terrain.append(
            {
                **common,
                "task_mode": task_mode,
                "time_to_edge": float(batch.episode_times_to_edge[candidate_idx, idx]),
                "cliff_vicinity_steps": cliff_vicinity_steps,
                "cliff_vicinity_fraction": cliff_vicinity_fraction,
                "entered_cliff_band": entered_cliff_band,
                "safe_edge_arrival": safe_edge_arrival,
                "fell": fell,
                "fall_count": float(fell),
                "goal_touched": False,
                "airborne": airborne,
                "left_edge": left_edge,
                "any_progress": 0.0,
                "goal_progress": 0.0,
                "correct_progress": 0.0,
                "incorrect_progress": 0.0,
                "toward_goal_frontier": 0.0,
                "reversal_progress": 0.0,
                "time_to_goal": 1.0,
                "turn_count": 0.0,
                "stable_turn_count": _episode_value(
                    "episode_stable_turn_counts",
                    idx,
                ),
                "progress_shaping": progress_shaping,
                "safe_arrival_bonus": safe_arrival_bonus,
                "safe_dwell_bonus": safe_dwell_bonus,
                "urgency_shaping": urgency_shaping,
                "peak_front_x": peak_front_x,
                "safe_front_x": safe_front_x,
                "probe_extent": float(batch.episode_probe_extents[candidate_idx, idx]),
            }
        )
    return test_fitnesses, test_per_terrain


def _evaluate_selected_elite(
    elite: dict[str, np.ndarray | float | int],
    ctx: PhaseContext,
    *,
    root_key: jax.Array,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    test_eval = _evaluate_elite_batch(
        [elite],
        ctx,
        root_key=root_key,
        workload="test",
    )
    return _test_rows_for_candidate(
        test_eval.batch_result,
        ctx.entries,
        candidate_idx=0,
    )


def _average_summary_dicts(summaries: list[dict[str, float]]) -> dict[str, float]:
    """Average a homogeneous list of numeric summary dicts."""
    if not summaries:
        return {}
    keys = sorted({key for summary in summaries for key in summary})
    return {
        key: float(np.mean([float(summary.get(key, 0.0)) for summary in summaries]))
        for key in keys
    }


def _run_deformation_suite(
    elites: list[dict[str, np.ndarray | float | int]],
    cfg: DictConfig,
    test_ctx: PhaseContext,
    *,
    root_key: jax.Array,
    render_output_root: str | Path | None = None,
    generation: int = 0,
) -> dict[str, object] | None:
    """Evaluate the top clean-reranked elites across deformation variants."""
    suite_cfg = cfg.eval.get("deformation_suite")
    if suite_cfg is None or not bool(suite_cfg.get("enabled", False)):
        return None

    elite_count = min(int(suite_cfg.get("elite_count", 5)), len(elites))
    if elite_count <= 0:
        return None
    variants = [str(variant) for variant in suite_cfg.get("variants", [])]
    if not variants:
        return None

    selected_elites = elites[:elite_count]
    per_elite: list[dict[str, object]] = [
        {
            "rerank_rank": int(rerank_rank),
            "archive_rank": int(elite.get("archive_rank", rerank_rank)),
            "flat_index": int(elite["flat_index"]),
            "archive_fitness": float(elite["fitness"]),
            "rerank_fitness": float(elite["selection_fitness"]),
            "descriptor": np.asarray(elite["descriptor"]).tolist(),
            "variants": {},
        }
        for rerank_rank, elite in enumerate(selected_elites)
    ]
    per_terrain_rows: list[dict[str, object]] = []
    variant_aggregate_summary: dict[str, dict[str, object]] = {}
    render_artifacts: dict[str, list[RenderArtifact]] = {}
    render_top_n = min(
        int(suite_cfg.get("render_top_n", 0)),
        elite_count,
    )

    for variant_index, variant in enumerate(variants):
        variant_ctx = _phase_context_with_cfg(
            test_ctx,
            _cfg_with_deformation_variant(cfg, variant),
        )
        variant_eval = _evaluate_elite_batch(
            selected_elites,
            variant_ctx,
            root_key=jax.random.fold_in(root_key, variant_index + 1),
            workload="test",
        )
        batch = variant_eval.batch_result
        variant_rows: list[dict[str, object]] = []
        variant_summaries: list[dict[str, object]] = []
        for elite_idx, elite in enumerate(selected_elites):
            fitnesses, rows = _test_rows_for_candidate(
                batch,
                test_ctx.entries,
                candidate_idx=elite_idx,
            )
            tagged_rows = [
                {
                    "variant": variant,
                    "rerank_rank": int(elite_idx),
                    "archive_rank": int(elite.get("archive_rank", elite_idx)),
                    "flat_index": int(elite["flat_index"]),
                    **row,
                }
                for row in rows
            ]
            summary = summarize_test_per_terrain(rows)
            ci_lo, ci_hi = bootstrap_ci(
                np.asarray(fitnesses),
                seed=int(cfg.eval.ou_noise_seed) + (variant_index * 1000) + elite_idx,
            )
            variant_summary = {
                "test_mean_fitness": float(np.mean(fitnesses)),
                "test_median_fitness": float(np.median(fitnesses)),
                "test_ci95_low": float(ci_lo),
                "test_ci95_high": float(ci_hi),
                "test_summary": summary,
                "n_test_terrains": int(len(rows)),
            }
            per_elite[elite_idx]["variants"][variant] = variant_summary
            variant_summaries.append(variant_summary)
            variant_rows.extend(tagged_rows)
            per_terrain_rows.extend(tagged_rows)

        variant_aggregate_summary[variant] = {
            "elite_count": int(len(variant_summaries)),
            "mean_test_mean_fitness": float(
                np.mean([summary["test_mean_fitness"] for summary in variant_summaries])
            ),
            "mean_test_median_fitness": float(
                np.mean([summary["test_median_fitness"] for summary in variant_summaries])
            ),
            "test_summary": summarize_test_per_terrain(variant_rows),
        }
        if (
            bool(suite_cfg.get("render", False))
            and render_output_root is not None
            and render_top_n > 0
        ):
            render_artifacts[variant] = render_ranked_elites(
                selected_elites[:render_top_n],
                variant_ctx.envs,
                variant_ctx.alive_mask,
                variant_ctx.cfg,
                generation=generation,
                stage=f"deformation_{variant}",
                output_dir=Path(render_output_root) / "deformation_suite" / "renders" / variant,
                runtime_resolution=variant_ctx.runtime_resolution,
            )

    for elite_entry in per_elite:
        variant_summaries = list(elite_entry["variants"].values())
        elite_entry["average_across_variants_summary"] = {
            "variant_count": int(len(variant_summaries)),
            "mean_test_mean_fitness": float(
                np.mean([summary["test_mean_fitness"] for summary in variant_summaries])
            ),
            "mean_test_median_fitness": float(
                np.mean([summary["test_median_fitness"] for summary in variant_summaries])
            ),
            "test_summary": _average_summary_dicts(
                [summary["test_summary"] for summary in variant_summaries]
            ),
        }

    return {
        "enabled": True,
        "elite_count": int(elite_count),
        "variants": variants,
        "noise_sigma_signed_delta": float(
            suite_cfg.get("noise_sigma_signed_delta", 0.05)
        ),
        "shuffle_mode": str(suite_cfg.get("shuffle_mode", "episode_fixed")),
        "render_enabled": bool(suite_cfg.get("render", False)),
        "render_top_n": int(render_top_n),
        "per_elite": per_elite,
        "per_terrain_rows": per_terrain_rows,
        "variant_aggregate_summary": variant_aggregate_summary,
        "render_artifacts": render_artifacts,
    }


def evaluate_gap_generalization_suite(
    elite: dict[str, np.ndarray | float | int],
    cfg: DictConfig,
    *,
    runtime_resolution: RuntimeResolution | None = None,
    output_path: str | Path | None = None,
    render_output_root: str | Path | None = None,
    generation: int = 0,
) -> GapGeneralizationSuiteResults:
    """Evaluate one selected elite across plateau-gap width surfaces."""
    suite_cfg = cfg.eval.get("gap_generalization_suite")
    if suite_cfg is None or not bool(suite_cfg.get("enabled", False)):
        raise ValueError("eval.gap_generalization_suite.enabled must be true to run the suite.")
    if str(cfg.terrain.get("task_mode", "legacy_cliff")).strip().lower() != "bilateral_goal_cliff":
        raise ValueError("gap_generalization_suite currently supports bilateral_goal_cliff terrains only.")
    if str(cfg.body.mode).strip().lower() != "fixed":
        raise ValueError("gap_generalization_suite currently supports body.mode='fixed' only.")

    surfaces = list(suite_cfg.get("surfaces") or [])
    if not surfaces:
        raise ValueError("gap_generalization_suite.surfaces must be non-empty.")

    trials_per_surface = int(suite_cfg.get("trials_per_surface", 12))
    trial_conditions = _gap_suite_trial_conditions(cfg, trials_per_surface=trials_per_surface)
    runtime_resolution = runtime_resolution or resolve_runtime_config(cfg)
    plateau_width = int(cfg.terrain.get("center_plateau_width", cfg.terrain.base_width))
    platform_y = int(cfg.terrain.get("platform_y", cfg.terrain.get("plateau_height", 3)))
    grid_h = int(cfg.terrain.grid_h)
    grid_w = int(cfg.terrain.grid_w)
    max_steps = _gap_generalization_max_steps(cfg)

    surface_summaries: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    all_fitnesses: list[float] = []
    artifacts: list[RenderArtifact] = []

    for surface_index, surface in enumerate(surfaces):
        surface_name = str(surface.get("name", f"surface_{surface_index}"))
        gap_width = int(surface.get("cliff_front_gap_width", 0))
        post_gap_runway = int(surface.get("cliff_post_gap_runway", 0))
        phase_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        phase_cfg.terrain.variant = str(surface.get("variant", f"{cfg.terrain.variant}_{surface_name}"))
        phase_cfg.terrain.cliff_front_gap_width = gap_width
        phase_cfg.terrain.cliff_front_gap_widths = [gap_width]
        phase_cfg.terrain.cliff_post_gap_runway = post_gap_runway
        phase_cfg.terrain.cliff_probe_gap_width = 0
        phase_cfg.terrain.cliff_probe_ledge_width = 0
        phase_cfg.train.max_steps = max_steps

        entries: list[TerrainEntry] = []
        for goal_distance, cliff_distance, goal_side in trial_conditions:
            grid, metadata = build_bilateral_goal_cliff_terrain(
                grid_h,
                grid_w,
                goal_distance=int(goal_distance),
                cliff_distance=int(cliff_distance),
                goal_side=str(goal_side),
                plateau_width=plateau_width,
                platform_y=platform_y,
                correct_side_distance_scale=float(cfg.terrain.get("correct_side_distance_scale", 1.0)),
                incorrect_side_distance_scale=float(cfg.terrain.get("incorrect_side_distance_scale", 1.0)),
                cliff_front_gap_width=gap_width,
                cliff_post_gap_runway=post_gap_runway,
            )
            gaps = (
                [(int(metadata["cliff_front_gap_start_x"]), gap_width)]
                if gap_width > 0 and int(metadata.get("cliff_front_gap_start_x", -1)) >= 0
                else []
            )
            entries.append(
                TerrainEntry(
                    grid=grid,
                    cliff_x=int(metadata["cliff_edge_x"]),
                    gaps=gaps,
                    metadata={**metadata, "gap_generalization_surface": surface_name},
                )
            )

        phase_resolution, envs, alive_mask, static_capacity = _build_phase_assets(
            phase_cfg,
            runtime_resolution,
            entries,
        )
        phase_ctx = PhaseContext(
            cfg=phase_cfg,
            runtime_resolution=phase_resolution,
            envs=envs,
            alive_mask=alive_mask,
            static_capacity=static_capacity,
            entries=entries,
            schedule=None,
            schedule_dir="",
        )
        fitnesses, rows = _evaluate_selected_elite(
            elite,
            phase_ctx,
            root_key=jax.random.fold_in(
                jax.random.PRNGKey(int(cfg.eval.ou_noise_seed)),
                surface_index + 1,
            ),
        )
        ci_lo, ci_hi = bootstrap_ci(
            np.asarray(fitnesses),
            seed=int(cfg.eval.ou_noise_seed) + (surface_index * 1000),
        )
        tagged_rows = [{**row, "surface": surface_name} for row in rows]
        summary = summarize_test_per_terrain(rows)
        surface_summaries.append(
            {
                "surface": surface_name,
                "cliff_front_gap_width": int(gap_width),
                "cliff_post_gap_runway": int(post_gap_runway),
                "n_trials": int(len(rows)),
                "test_mean_fitness": float(np.mean(fitnesses)),
                "test_median_fitness": float(np.median(fitnesses)),
                "test_ci95_low": float(ci_lo),
                "test_ci95_high": float(ci_hi),
                "test_summary": summary,
            }
        )
        all_rows.extend(tagged_rows)
        all_fitnesses.extend(np.asarray(fitnesses, dtype=np.float32).tolist())

        if bool(suite_cfg.get("render", False)) and render_output_root is not None:
            artifacts.extend(
                render_ranked_elites(
                    [elite],
                    phase_ctx.envs,
                    phase_ctx.alive_mask,
                    phase_ctx.cfg,
                    generation=generation,
                    stage=f"gap_suite_{surface_name}",
                    output_dir=Path(render_output_root) / "gap_generalization" / "renders" / surface_name,
                    runtime_resolution=phase_ctx.runtime_resolution,
                )
            )

    overall_summary = {
        "surface_count": int(len(surface_summaries)),
        "n_trials": int(len(all_rows)),
        "test_mean_fitness": float(np.mean(all_fitnesses)) if all_fitnesses else 0.0,
        "test_median_fitness": float(np.median(all_fitnesses)) if all_fitnesses else 0.0,
        "test_summary": summarize_test_per_terrain(all_rows),
    }
    results = GapGeneralizationSuiteResults(
        run_metadata=run_metadata(cfg),
        suite_metadata={
            "trials_per_surface": int(trials_per_surface),
            "surfaces": [
                {
                    "name": str(surface.get("name", f"surface_{idx}")),
                    "cliff_front_gap_width": int(surface.get("cliff_front_gap_width", 0)),
                    "cliff_post_gap_runway": int(surface.get("cliff_post_gap_runway", 0)),
                }
                for idx, surface in enumerate(surfaces)
            ],
            "max_steps": int(max_steps),
        },
        selected_elite_flat_index=int(elite["flat_index"]),
        surface_summaries=surface_summaries,
        overall_summary=overall_summary,
        per_terrain_rows=all_rows,
        artifacts=artifacts,
    )

    log.info(
        "Post-train gap suite | surfaces=%d selected_flat_index=%d mean=%.2f",
        len(surface_summaries),
        results.selected_elite_flat_index,
        float(results.overall_summary.get("test_mean_fitness", 0.0)),
    )

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results.as_json_dict(), f, indent=2)
        log.info("Post-train gap-suite results saved | path=%s", out)

    return results


def evaluate_symmetric_double_cliff_probe(
    elite: dict[str, np.ndarray | float | int]
    | list[dict[str, np.ndarray | float | int]],
    cfg: DictConfig,
    *,
    runtime_resolution: RuntimeResolution | None = None,
    output_path: str | Path | None = None,
    render_output_root: str | Path | None = None,
    generation: int = 0,
) -> RobustnessProbeResults:
    """Evaluate one selected elite on the deterministic symmetric double-cliff probe."""
    probe_cfg = cfg.eval.get("robustness_probe")
    if probe_cfg is None or not bool(probe_cfg.get("enabled", False)):
        raise ValueError("eval.robustness_probe.enabled must be true to run the robustness probe.")

    mode = str(probe_cfg.get("mode", "symmetric_double_cliff")).strip().lower()
    if mode != "symmetric_double_cliff":
        raise ValueError(f"Unsupported robustness_probe.mode={mode!r}")

    runtime_resolution = runtime_resolution or resolve_runtime_config(cfg)
    elites = [elite] if isinstance(elite, dict) else list(elite)
    if not elites:
        raise ValueError("Robustness probe requires at least one elite.")
    probe_ctx, probe_cases = _symmetric_double_cliff_probe_context(cfg, runtime_resolution)
    _warmup_phase(probe_ctx, workload="test", candidate_count=len(elites))
    _, candidate_keys = split_key_sequence(
        jax.random.PRNGKey(int(cfg.eval.ou_noise_seed)),
        len(elites),
    )
    probe_eval = evaluate_population_chunked(
        np.asarray([candidate["genotype"] for candidate in elites]),
        probe_ctx.envs,
        probe_ctx.alive_mask,
        candidate_keys,
        probe_ctx.cfg,
        workload="test",
        runtime_resolution=probe_ctx.runtime_resolution,
        static_capacity=probe_ctx.static_capacity,
    )
    batch = probe_eval.batch_result
    per_elite: list[dict[str, object]] = []
    death_rates: list[float] = []
    mean_steps_alive_values: list[float] = []
    for elite_idx, candidate in enumerate(elites):
        per_distance: list[dict[str, object]] = []
        death_flags: list[float] = []
        steps_alive: list[float] = []
        for idx, case in enumerate(probe_cases):
            fell = bool(batch.episode_fell[elite_idx, idx])
            airborne = bool(batch.episode_airborne[elite_idx, idx])
            left_edge = bool(batch.episode_left_edge[elite_idx, idx])
            died = bool(fell or airborne or left_edge)
            steps = int(batch.episode_steps_alive[elite_idx, idx])
            death_flags.append(1.0 if died else 0.0)
            steps_alive.append(float(steps))
            per_distance.append(
                {
                    "terrain_idx": int(idx),
                    "distance": int(case["distance"]),
                    "platform_start_x": int(case["platform_start_x"]),
                    "platform_end_x": int(case["platform_end_x"]),
                    "cliff_x": int(probe_ctx.entries[idx].cliff_x),
                    "fitness": float(batch.episode_fitnesses[elite_idx, idx]),
                    "fell": fell,
                    "airborne": airborne,
                    "left_edge": left_edge,
                    "died": died,
                    "steps_alive": steps,
                }
            )
        death_rate = float(np.mean(death_flags)) if death_flags else 0.0
        mean_steps_alive = float(np.mean(steps_alive)) if steps_alive else 0.0
        death_rates.append(death_rate)
        mean_steps_alive_values.append(mean_steps_alive)
        per_elite.append(
            {
                "rerank_rank": int(candidate.get("rerank_rank", elite_idx)),
                "archive_rank": int(candidate.get("archive_rank", elite_idx)),
                "flat_index": int(candidate["flat_index"]),
                "archive_fitness": float(candidate.get("fitness", 0.0)),
                "rerank_fitness": float(candidate.get("selection_fitness", candidate.get("fitness", 0.0))),
                "descriptor": np.asarray(candidate.get("descriptor", np.zeros((0,), dtype=np.float32))).tolist(),
                "death_rate": death_rate,
                "mean_steps_alive": mean_steps_alive,
                "n_probe_terrains": int(len(per_distance)),
                "per_distance": per_distance,
            }
        )

    artifacts: list[RenderArtifact] = []
    if bool(probe_cfg.get("render", False)) and render_output_root is not None:
        artifacts = render_ranked_elites(
            elites,
            probe_ctx.envs,
            probe_ctx.alive_mask,
            probe_ctx.cfg,
            generation=generation,
            stage="probe",
            output_dir=Path(render_output_root) / "robustness_probe" / "renders",
            runtime_resolution=probe_ctx.runtime_resolution,
        )

    probe_metadata = {
        "mode": mode,
        "grid_h": int(probe_ctx.cfg.terrain.grid_h),
        "grid_w": int(probe_ctx.cfg.terrain.grid_w),
        "platform_y": int(probe_ctx.cfg.terrain.platform_y),
        "plateau_width": int(probe_cfg.get("plateau_width", 10)),
        "spawn_x": int(probe_ctx.cfg.terrain.spawn_x),
        "spawn_y": int(probe_ctx.cfg.terrain.spawn_y),
        "min_distance": int(probe_cfg.get("min_distance", 1)),
        "max_distance": int(probe_cfg.get("max_distance", 6)),
        "max_steps": int(probe_ctx.cfg.train.max_steps),
    }
    results = RobustnessProbeResults(
        run_metadata=run_metadata(cfg),
        probe_metadata=probe_metadata,
        leading_elite_flat_index=int(elites[0]["flat_index"]),
        elite_count=int(len(per_elite)),
        aggregate_summary={
            "elite_count": int(len(per_elite)),
            "n_probe_terrains": int(len(probe_cases)),
            "mean_death_rate": float(np.mean(death_rates)) if death_rates else 0.0,
            "mean_mean_steps_alive": (
                float(np.mean(mean_steps_alive_values)) if mean_steps_alive_values else 0.0
            ),
        },
        per_elite=per_elite,
        artifacts=artifacts,
    )

    log.info(
        "Post-train probe | elites=%d terrains=%d leading_flat_index=%d mean_death_rate=%.3f mean_steps_alive=%.1f",
        results.elite_count,
        int(results.aggregate_summary.get("n_probe_terrains", 0)),
        results.leading_elite_flat_index,
        float(results.aggregate_summary.get("mean_death_rate", 0.0)),
        float(results.aggregate_summary.get("mean_mean_steps_alive", 0.0)),
    )

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results.as_json_dict(), f, indent=2)
        log.info("Post-train probe results saved | path=%s", out)

    return results


def evaluate_archive_with_schedules(
    archive,
    cfg: DictConfig,
    *,
    rerank_schedule_dir: str | Path,
    eval_schedule_dir: str | Path,
    runtime_resolution: RuntimeResolution | None = None,
    output_path: str | Path | None = None,
    render_output_root: str | Path | None = None,
    generation: int = 0,
) -> PostTrainEvalResults:
    """Rerank archive elites on one frozen suite and evaluate on another."""
    if archive.size <= 0:
        raise ValueError("Cannot evaluate an empty archive")

    runtime_resolution = runtime_resolution or resolve_runtime_config(cfg)
    allow_legacy_schedule_hashes = bool(cfg.eval.get("allow_legacy_schedule_hashes", False))
    rerank_schedule = load_schedule(
        rerank_schedule_dir,
        allow_legacy_hashes=allow_legacy_schedule_hashes,
    )
    eval_schedule = load_schedule(
        eval_schedule_dir,
        allow_legacy_hashes=allow_legacy_schedule_hashes,
    )
    validate_bilateral_schedule_semantics(rerank_schedule)
    validate_bilateral_schedule_semantics(eval_schedule)
    rerank_variant = str(cfg.eval.get("rerank_variant") or cfg.terrain.variant)
    test_variant = str(cfg.eval.get("test_variant") or rerank_variant)
    _validate_loaded_schedule(
        rerank_schedule_dir,
        rerank_schedule,
        expected_seed=int(cfg.get("seed", rerank_schedule.seed)),
        expected_variant=rerank_variant,
        expected_val=int(cfg.eval.get("rerank_episodes", cfg.eval.eval_episodes)),
    )
    _validate_loaded_schedule(
        eval_schedule_dir,
        eval_schedule,
        expected_seed=int(cfg.get("seed", eval_schedule.seed)),
        expected_variant=test_variant,
        expected_test=int(cfg.eval.get("test_episodes", cfg.eval.eval_episodes)),
    )
    rerank_episode_limit = _resolve_limit(
        cfg.eval.get("rerank_episodes", cfg.eval.eval_episodes),
        len(rerank_schedule.val),
    )
    rerank_episode_limit = max(1, rerank_episode_limit)
    test_episode_limit = _resolve_limit(
        cfg.eval.get("test_episodes", cfg.eval.eval_episodes),
        len(eval_schedule.test),
    )
    test_episode_limit = max(1, test_episode_limit)
    elite_limit = _resolve_limit(cfg.eval.eval_individuals, archive.size)
    rerank_max_steps = cfg.eval.get("rerank_max_steps")
    test_max_steps = cfg.eval.get("test_max_steps")

    rerank_ctx = _phase_context_from_schedule(
        cfg,
        runtime_resolution,
        rerank_schedule_dir,
        rerank_schedule,
        split="val",
        max_steps=int(rerank_max_steps) if rerank_max_steps is not None else int(cfg.train.max_steps),
        limit=rerank_episode_limit,
        variant_override=rerank_variant,
    )
    test_ctx = _phase_context_from_schedule(
        cfg,
        runtime_resolution,
        eval_schedule_dir,
        eval_schedule,
        split="test",
        max_steps=int(test_max_steps) if test_max_steps is not None else int(cfg.train.max_steps),
        limit=test_episode_limit,
        variant_override=test_variant,
    )

    elites = archive.top_elites_by_fitness(elite_limit)
    rerank_root = jax.random.fold_in(jax.random.PRNGKey(int(cfg.eval.ou_noise_seed)), 0)
    test_root = jax.random.fold_in(jax.random.PRNGKey(int(cfg.eval.ou_noise_seed)), 1)
    scored = _score_elites_on_phase(elites, rerank_ctx, root_key=rerank_root, workload="validation")
    if not scored:
        raise ValueError("Rerank produced no scored elites")

    for idx, elite in enumerate(scored):
        log.info(
            "Rerank elite | rank=%d rerank_fitness=%.2f train_fitness=%.2f flat_index=%d",
            idx,
            float(elite["selection_fitness"]),
            float(elite["fitness"]),
            int(elite["flat_index"]),
        )

    selected = scored[0]
    test_fitnesses, test_per_terrain = _evaluate_selected_elite(
        selected,
        test_ctx,
        root_key=test_root,
    )
    ci_lo, ci_hi = bootstrap_ci(np.asarray(test_fitnesses), seed=int(cfg.eval.ou_noise_seed))
    test_summary = summarize_test_per_terrain(test_per_terrain)

    render_limit = resolve_render_limit(cfg.render.get("render_evals"), len(scored))
    deformation_suite_cfg = cfg.eval.get("deformation_suite")
    deformation_save_limit = 0
    if deformation_suite_cfg is not None and bool(deformation_suite_cfg.get("enabled", False)):
        deformation_save_limit = int(deformation_suite_cfg.get("elite_count", 0))
    probe_cfg = cfg.eval.get("robustness_probe")
    probe_save_limit = 0
    if probe_cfg is not None and bool(probe_cfg.get("enabled", False)):
        probe_save_limit = int(probe_cfg.get("elite_count", 1))
    save_limit = min(max(5, deformation_save_limit, probe_save_limit), len(scored))
    rerank_artifacts: list[RenderArtifact] = []
    eval_artifacts: list[RenderArtifact] = []
    if bool(cfg.render.get("post_train", {}).get("enabled", True)) and render_output_root is not None:
        rerank_output_dir = Path(render_output_root) / "archive_eval" / "renders" / "rerank"
        eval_output_dir = Path(render_output_root) / "archive_eval" / "renders" / "eval"
        if render_limit > 0:
            rerank_artifacts = render_ranked_elites(
                scored[:render_limit],
                rerank_ctx.envs,
                rerank_ctx.alive_mask,
                rerank_ctx.cfg,
                generation=generation,
                stage="rerank",
                output_dir=rerank_output_dir,
                runtime_resolution=rerank_ctx.runtime_resolution,
            )
        if render_limit > 0:
            eval_artifacts = render_ranked_elites(
                scored[:render_limit],
                test_ctx.envs,
                test_ctx.alive_mask,
                test_ctx.cfg,
                generation=generation,
                stage="eval",
                output_dir=eval_output_dir,
                runtime_resolution=test_ctx.runtime_resolution,
            )

    rerank_scores: list[dict[str, object]] = []
    for rerank_rank, elite in enumerate(scored):
        rerank_scores.append(
            {
                "rerank_rank": int(rerank_rank),
                "archive_rank": int(elite.get("archive_rank", rerank_rank)),
                "flat_index": int(elite["flat_index"]),
                "archive_fitness": float(elite["fitness"]),
                "rerank_fitness": float(elite["selection_fitness"]),
                "descriptor": np.asarray(elite["descriptor"]).tolist(),
            }
        )

    saved_elites: list[dict[str, object]] = []
    for rerank_rank, elite in enumerate(scored[:save_limit]):
        saved_elites.append(
            {
                "rerank_rank": int(rerank_rank),
                "archive_rank": int(elite.get("archive_rank", rerank_rank)),
                "flat_index": int(elite["flat_index"]),
                "archive_fitness": float(elite["fitness"]),
                "rerank_fitness": float(elite["selection_fitness"]),
                "descriptor": np.asarray(elite["descriptor"]).tolist(),
                "genotype": np.asarray(elite["genotype"]).tolist(),
            }
        )

    deformation_suite = _run_deformation_suite(
        scored[:save_limit],
        cfg,
        test_ctx,
        root_key=jax.random.fold_in(test_root, 97),
        render_output_root=render_output_root,
        generation=generation,
    )

    results = PostTrainEvalResults(
        run_metadata=run_metadata(cfg),
        test_summary=test_summary,
        selected_elite_rank=0,
        selected_elite_flat_index=int(selected["flat_index"]),
        selected_elite_fitness_rerank=float(selected["selection_fitness"]),
        test_mean_fitness=float(np.mean(test_fitnesses)),
        test_median_fitness=float(np.median(test_fitnesses)),
        test_ci95_low=ci_lo,
        test_ci95_high=ci_hi,
        rerank_scores=rerank_scores,
        test_per_terrain=test_per_terrain,
        n_rerank_terrains=_env_count(rerank_ctx.envs),
        n_test_terrains=_env_count(test_ctx.envs),
        rerank_schedule=_schedule_metadata(rerank_schedule_dir, rerank_schedule, "val"),
        eval_schedule=_schedule_metadata(eval_schedule_dir, eval_schedule, "test"),
        rerank_artifacts=rerank_artifacts,
        eval_artifacts=eval_artifacts,
        saved_elites=saved_elites,
        deformation_suite=deformation_suite,
        selected_elite_genotype=np.asarray(selected["genotype"]),
    )

    log.info(
        "Post-train eval | rerank_terrains=%d test_terrains=%d selected_flat_index=%d mean=%.2f median=%.2f ci95=[%.2f, %.2f]",
        results.n_rerank_terrains,
        results.n_test_terrains,
        results.selected_elite_flat_index,
        results.test_mean_fitness,
        results.test_median_fitness,
        results.test_ci95_low,
        results.test_ci95_high,
    )

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results.as_json_dict(), f, indent=2)
        log.info("Post-train results saved | path=%s", out)

    return results


def evaluate_archive(
    checkpoint_path: str | Path,
    terrain_schedule_dir: str | Path,
    cfg: DictConfig,
    *,
    top_k: int | None = None,
    ou_noise_seed: int | None = None,
    output_path: str | Path | None = None,
    runtime_resolution: RuntimeResolution | None = None,
) -> EvalResults:
    """Load a checkpoint and evaluate it on one frozen suite."""
    results = evaluate_checkpoint_with_schedules(
        checkpoint_path=checkpoint_path,
        rerank_schedule_dir=terrain_schedule_dir,
        eval_schedule_dir=terrain_schedule_dir,
        cfg=cfg,
        top_k=top_k,
        ou_noise_seed=ou_noise_seed,
        output_path=output_path,
        runtime_resolution=runtime_resolution,
    )
    return EvalResults(
        selected_elite_idx=int(results.selected_elite_rank),
        selected_elite_fitness_val=float(results.selected_elite_fitness_rerank),
        test_mean_fitness=float(results.test_mean_fitness),
        test_median_fitness=float(results.test_median_fitness),
        test_ci95_low=float(results.test_ci95_low),
        test_ci95_high=float(results.test_ci95_high),
        test_per_terrain=list(results.test_per_terrain),
        n_test_terrains=int(results.n_test_terrains),
    )


def evaluate_checkpoint_with_schedules(
    checkpoint_path: str | Path,
    *,
    rerank_schedule_dir: str | Path,
    eval_schedule_dir: str | Path,
    cfg: DictConfig,
    top_k: int | None = None,
    ou_noise_seed: int | None = None,
    output_path: str | Path | None = None,
    runtime_resolution: RuntimeResolution | None = None,
) -> PostTrainEvalResults:
    """Load a checkpoint and evaluate it with distinct rerank and test suites."""
    genotype_dim = genotype_dim_from_cfg(cfg)
    runtime_resolution = runtime_resolution or resolve_runtime_config(cfg)

    ckpt = load_checkpoint(
        checkpoint_path,
        genotype_length=genotype_dim,
        **emitter_kwargs_from_cfg(cfg),
    )
    checkpoint_config_sha256 = str(ckpt.get("meta", {}).get("config_sha256", "")).strip()
    expected_config_sha256 = logical_config_sha256(cfg)
    if (
        not bool(cfg.eval.get("allow_checkpoint_config_mismatch", False))
        and checkpoint_config_sha256
        and expected_config_sha256
        and checkpoint_config_sha256 != expected_config_sha256
    ):
        raise ValueError(
            "Checkpoint config hash does not match current logical config. "
            f"checkpoint={checkpoint_config_sha256} current={expected_config_sha256}"
        )
    archive = ckpt["archive"]
    log.info(
        "Eval config | checkpoint=%s top_k=%s ou_noise_seed=%d total_pop=%d",
        checkpoint_path,
        "config" if top_k is None else top_k,
        int(cfg.eval.ou_noise_seed if ou_noise_seed is None else ou_noise_seed),
        total_population(cfg),
    )
    log.info("Loaded archive | elites=%d best=%.2f", archive.size, archive.best_fitness)

    eval_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if top_k is not None:
        eval_cfg.eval.eval_individuals = int(top_k)
    if ou_noise_seed is not None:
        eval_cfg.eval.ou_noise_seed = int(ou_noise_seed)

    results = evaluate_archive_with_schedules(
        archive,
        eval_cfg,
        rerank_schedule_dir=rerank_schedule_dir,
        eval_schedule_dir=eval_schedule_dir,
        runtime_resolution=runtime_resolution,
        output_path=output_path,
    )
    return results


def main():
    """CLI entry point for held-out evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a trained archive on held-out terrains.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path prefix (without extension)")
    parser.add_argument("--schedule", required=True, help="Terrain schedule directory")
    parser.add_argument("--preset", default="config", help="Hydra config name")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    configure_logging(logging.INFO)

    from .runtime_config import compose_experiment_config

    cfg = compose_experiment_config(
        preset=args.preset,
        config_dir=Path(__file__).parent.parent / "config",
    )

    output = args.output or f"{args.checkpoint}_eval_results.json"
    evaluate_archive(
        checkpoint_path=args.checkpoint,
        terrain_schedule_dir=args.schedule,
        cfg=cfg,
        top_k=args.top_k,
        output_path=output,
    )
