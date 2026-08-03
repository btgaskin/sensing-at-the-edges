"""Training loop: CMA-MAE ask-evaluate-tell with QD archive."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import uuid

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

from jax_evogym.constants import EMPTY

from .batch_eval import (
    ChunkProgress,
    CandidateEpisodePhaseProbeResult,
    CandidateEpisodeSubstepProbeResult,
    GrownEvalAssets,
    evaluate_population_chunked,
    planned_warmup_chunk_sizes,
    probe_candidate_episode_phases,
    probe_candidate_episode_substeps,
    split_key_sequence,
    trace_population_chunked,
    warmup_population_evaluator,
)
from .checkpoint import (
    build_archive_export_payload,
    build_checkpoint_payload,
    find_latest_checkpoint,
    load_checkpoint_meta,
    write_archive_export_payload,
    write_checkpoint_payload,
)
from .collision_tuning import (
    ResolvedCollisionTuning,
    resolve_collision_tuning_from_terrains,
)
from .env import (
    CliffEnvData,
    CliffEvalEnvData,
    CliffDynamicCore,
    StaticColliderCapacity,
    assemble_cliff_env,
    build_static_collider_template,
    build_cliff_dynamic_core,
    build_cliff_terrain_assets,
    derive_physics_params,
    infer_static_template_capacity,
    precompute_shared_grid_data,
    stack_static_templates,
)
from .evaluate import (
    _build_phase_assets,
    _phase_cfg_from_schedule,
    _test_rows_for_candidate,
    summarize_test_per_terrain,
)
from .jax_cache import setup_jax_cache
from .logging_utils import configure_logging
from .manifest import (
    config_sha256,
    create_manifest,
    load_manifest,
    record_manifest_stage_result,
    save_manifest,
    update_manifest_retry_state,
)
from .post_run import (
    materialize_historical_elite_artifacts,
    materialize_specimen_artifacts,
    resolve_historical_elite_root,
    resolve_specimen_root,
    run_post_run_for_stored_specimen,
    specimen_directory,
)
from .qd.cma_emitter import CMAEmitter
from .qd_descriptors import build_descriptor_matrix, resolve_qd_archive
from .qd.focus import (
    FocusConfig,
    FocusWindowBounds,
    build_mask_from_bounds,
    focus_config_from_section,
    focus_metrics,
    initial_window_bounds,
    maybe_tighten,
    validate_focus_config,
)
from .qd.grid_archive import GridArchive, GridArchiveBatchDecision
from .render import (
    build_wandb_render_payload,
    render_periodic_archive,
    render_ranked_elites,
)
from .runtime_config import (
    RuntimeResolution,
    attach_collision_tuning,
    controller_size_label,
    deformation_label,
    diff_config_paths,
    emitter_kwargs_from_cfg,
    genotype_dim_from_cfg,
    logical_config_sha256,
    path_is_allowed,
    resolved_spawn_x,
    resolved_noise_settings,
    run_metadata,
    resolve_runtime_config,
    total_population,
)
from .terrain import fixed_body_alive_mask, generate_generation_terrains, make_robot_body
from .terrain import max_canvas_size
from .terrain_schedule import (
    TerrainEntry,
    load_schedule,
    resolve_generation_terrain_cfg,
    validate_bilateral_schedule_semantics,
)
from .training_reporting import (
    append_metrics_row,
    build_generation_report,
    log_generation_report,
    truncate_metrics_rows,
)
from .wandb_metrics import (
    build_generation_metrics,
    init_wandb,
    render_archive_heatmap,
    render_archive_named_heatmaps,
    render_archive_third_descriptor_heatmap,
    render_fitness_histogram,
)
from .fitness import candidate_batch_fitness, episode_fitness_array

log = logging.getLogger(__name__)
BRANCH_ALLOWED_CONFIG_PATHS = frozenset(
    {
        "train.n_generations",
        "train.output_dir",
        "train.resume_from",
        "train.stop_after_generations",
        "train.resume_mode",
        "train.checkpoint_interval",
        "train.terrain_schedule_dir",
        "train.allow_legacy_schedule_hashes",
        "train.allow_output_dir_hash_mismatch_resume",
        "eval.rerank_episodes",
        "eval.test_episodes",
        "eval.rerank_max_steps",
        "eval.test_max_steps",
        "eval.rerank_schedule_dir",
        "eval.test_schedule_dir",
        "eval.allow_legacy_schedule_hashes",
        "eval.allow_checkpoint_config_mismatch",
        "eval.post_run",
        "notes",
        "tags",
        "wandb",
        "logging",
        "render",
        "gpu_profile",
        "batching",
        "specimens.run_on_stage_pause",
    }
)
_SPECIMEN_EVAL_SCOPE = 1401
_HISTORICAL_PERFECT_ZERO_FALL_TIER = "perfect_zero_fall"
_HISTORICAL_NEAR_PERFECT_ZERO_FALL_TIER = "near_perfect_zero_fall"
_PERFECT_SPECIMEN_TIER = "perfect"
_IMPERFECT_SPECIMEN_TIER = "imperfect"


@dataclass
class _AsyncCheckpointWriter:
    thread: threading.Thread | None = None
    error: BaseException | None = None
    path_prefix: Path | None = None

    def start(
        self,
        path_prefix: Path,
        *,
        meta: dict,
        arrays: dict[str, np.ndarray],
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        self.join()
        self.error = None
        self.path_prefix = path_prefix

        def _run() -> None:
            try:
                write_checkpoint_payload(
                    path_prefix,
                    meta=meta,
                    arrays=arrays,
                    on_complete=on_complete,
                )
            except BaseException as exc:  # pragma: no cover - exercised via join path
                self.error = exc

        self.thread = threading.Thread(
            target=_run,
            name=f"checkpoint-{path_prefix.name}",
            daemon=True,
        )
        self.thread.start()

    def join(self) -> None:
        if self.thread is None:
            return
        self.thread.join()
        self.thread = None
        if self.error is not None:
            exc = self.error
            self.error = None
            raise RuntimeError(
                f"Checkpoint write failed for {self.path_prefix}"
            ) from exc


def _metadata_digest(metadata: dict[str, object] | None) -> bytes:
    """Return a deterministic metadata digest for terrain asset cache keys."""
    canonical = json.dumps(
        dict(metadata or {}),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).digest()


@dataclass
class PreparedTrainingInputs:
    """Static inputs prepared before the generation loop."""

    body_mode: str
    genotype_dim: int
    alive_mask: jnp.ndarray
    terrain_generations: list[list[tuple[np.ndarray, int, dict[str, object]]]]
    dynamic_core: CliffDynamicCore | None
    robot_morph: np.ndarray | None
    grid_data: object | None
    spawn_x: int
    spawn_y: int
    spring_stiffness_scale: float
    generation_static_capacity: StaticColliderCapacity | None
    terrain_schedule_info: dict | None
    collision_tuning: ResolvedCollisionTuning
    _generation_env_cache: dict[int, list[CliffEnvData] | GrownEvalAssets] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _terrain_asset_cache: dict[tuple[int, bytes, bytes], object] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def generation_terrain_count(self, gen: int) -> int:
        return len(self.terrain_generations[gen])

    def envs_for_generation(self, gen: int) -> list[CliffEnvData] | GrownEvalAssets:
        """Build or reuse the terrain batch for one generation."""
        cached = self._generation_env_cache.get(gen)
        if cached is not None:
            return cached

        if self.body_mode == "grown":
            canvas_h, canvas_w = self.alive_mask.shape
            templates = []
            for grid, cliff_x, metadata in self.terrain_generations[gen]:
                terrain_np = np.asarray(grid, dtype=np.int32)
                cache_key = (
                    int(cliff_x),
                    terrain_np.tobytes(),
                    _metadata_digest(metadata),
                )
                terrain_assets = self._terrain_asset_cache.get(cache_key)
                if terrain_assets is None:
                    terrain_assets = build_static_collider_template(
                        terrain_np,
                        cliff_x=int(cliff_x),
                        task_metadata=metadata,
                        canvas_h=int(canvas_h),
                        canvas_w=int(canvas_w),
                        spawn_x=self.spawn_x,
                        spawn_y=self.spawn_y,
                        spring_stiffness_scale=self.spring_stiffness_scale,
                        collision_build_capacity=self.collision_tuning.build_capacity,
                    )
                    self._terrain_asset_cache[cache_key] = terrain_assets
                templates.append(terrain_assets)
            static_capacity = self.generation_static_capacity or infer_static_template_capacity(templates)
            grown_assets = GrownEvalAssets(
                static_templates=stack_static_templates(templates, static_capacity=static_capacity),
                templates=tuple(templates),
                static_capacity=static_capacity,
                grid_data=self.grid_data,
                runtime_build_capacity=self.collision_tuning.runtime_build_capacity,
            )
            self._generation_env_cache[gen] = grown_assets
            return grown_assets

        generation_envs: list[CliffEnvData] = []
        for grid, cliff_x, metadata in self.terrain_generations[gen]:
            terrain_np = np.asarray(grid, dtype=np.int32)
            cache_key = (
                int(cliff_x),
                terrain_np.tobytes(),
                _metadata_digest(metadata),
            )
            terrain_assets = self._terrain_asset_cache.get(cache_key)
            if terrain_assets is None:
                terrain_assets = build_cliff_terrain_assets(
                    terrain_np,
                    self.robot_morph,
                    self.dynamic_core.spawn_x,
                    self.dynamic_core.spawn_y,
                    cliff_x,
                    task_metadata=metadata,
                    spring_stiffness_scale=self.spring_stiffness_scale,
                    collision_build_capacity=self.collision_tuning.build_capacity,
                )
                self._terrain_asset_cache[cache_key] = terrain_assets
            generation_envs.append(assemble_cliff_env(self.dynamic_core, terrain_assets))

        self._generation_env_cache[gen] = generation_envs
        return generation_envs


@dataclass(frozen=True)
class TrainingArtifacts:
    """Output paths and persisted run metadata."""

    output_dir: Path
    metrics_path: Path
    manifest_path: Path
    resolved_config_path: Path
    resolved_config_str: str
    invocation_config_sha256: str
    logical_config_sha256: str
    manifest: dict


@dataclass(frozen=True)
class TrainingBootstrap:
    """Resolved logical run identity and resume source before training starts."""

    output_dir: Path
    metrics_path: Path
    manifest_path: Path
    resolved_config_path: Path
    resolved_config_str: str
    invocation_config_sha256: str
    logical_config_sha256: str
    existing_manifest: dict | None
    run_id: str
    wandb_run_id: str
    attempt_count: int
    resume_mode: str
    stop_after_generations: int
    lineage: dict
    allow_checkpoint_config_mismatch: bool = False
    resume_from: str | None = None
    resumed_generation: int | None = None
    latest_checkpoint_path: str | None = None


@dataclass
class TrainingState:
    """Mutable search state that evolves across generations."""

    archive: GridArchive
    emitters: list[CMAEmitter]
    key: jax.Array
    rng: np.random.Generator
    start_gen: int = 0
    focus_bounds: FocusWindowBounds | None = None


@dataclass(frozen=True)
class SpecimenExtractionResults:
    """Summary of final-stage specimen extraction outputs."""

    output_dir: Path
    manifest_path: Path
    terrain_manifest_path: Path
    pool_path: Path
    pool_rows_path: Path
    selected_path: Path
    selected_genotypes_path: Path
    pool_size: int
    pre_dedup_pool_size: int
    archive_elite_count: int
    historical_elite_file_count: int
    historical_elite_candidate_count: int
    fresh_sample_count: int
    qualifying_count: int
    selected_count: int
    perfect_zero_fall_count: int
    near_perfect_zero_fall_count: int
    perfect_specimen_count: int
    imperfect_specimen_count: int
    occupied_completion_bins: int


def _render_periodic_archive_best_effort(
    archive,
    env_datas: list[CliffEnvData] | GrownEvalAssets,
    alive_mask: jnp.ndarray,
    cfg: DictConfig,
    *,
    generation: int,
    output_root: str | Path,
    runtime_resolution: RuntimeResolution | None = None,
):
    """Render a periodic archive snapshot without aborting training on failure."""
    try:
        return render_periodic_archive(
            archive,
            env_datas,
            alive_mask,
            cfg,
            generation=generation,
            output_root=output_root,
            runtime_resolution=runtime_resolution,
        )
    except Exception:
        log.exception("Periodic render failed | generation=%d", generation)
        return []


def _render_periodic_qd_projection_best_effort(
    archive,
    alive_mask: jnp.ndarray,
    cfg: DictConfig,
    *,
    generation: int,
    output_root: str | Path,
    runtime_resolution: RuntimeResolution | None = None,
):
    """Render periodic QD projection snapshots without aborting training on failure."""
    try:
        return []
    except Exception:
        log.exception("Periodic QD render failed | generation=%d", generation)
        return []


def _update_archive_scored_batch(
    archive: GridArchive,
    candidate_batch: jax.Array,
    fitnesses: np.ndarray,
    descriptors: np.ndarray,
    *,
    active_cell_mask: np.ndarray | None = None,
) -> GridArchiveBatchDecision:
    """Score the evaluated batch, then copy only accepted winners to the host."""
    decision = archive.score_batch(
        fitnesses,
        descriptors,
        active_cell_mask=active_cell_mask,
    )
    if decision.accepted_indices.size == 0:
        return decision

    accepted_indices = jnp.asarray(decision.accepted_indices, dtype=jnp.int32)
    accepted_candidates = np.asarray(jax.block_until_ready(candidate_batch[accepted_indices]))
    archive.commit_scored_batch(
        decision,
        accepted_candidates,
        descriptors[decision.accepted_indices],
    )
    return decision


def _resolve_post_train_schedule_dirs(cfg: DictConfig) -> tuple[str, str] | None:
    """Return explicit post-train schedule dirs, or skip inline eval."""
    rerank_schedule_dir = cfg.eval.get("rerank_schedule_dir")
    eval_schedule_dir = cfg.eval.get("test_schedule_dir")
    has_rerank_schedule = bool(rerank_schedule_dir)
    has_eval_schedule = bool(eval_schedule_dir)

    if has_rerank_schedule and has_eval_schedule:
        return str(rerank_schedule_dir), str(eval_schedule_dir)
    if has_rerank_schedule or has_eval_schedule:
        missing = (
            "eval.rerank_schedule_dir"
            if not has_rerank_schedule
            else "eval.test_schedule_dir"
        )
        raise ValueError(
            "Post-train rerank/eval requires both eval.rerank_schedule_dir and "
            f"eval.test_schedule_dir; missing {missing}"
        )
    return None


def _prepare_training_terrains(
    cfg: DictConfig,
    rng: np.random.Generator,
) -> tuple[list[list[tuple[np.ndarray, int, dict[str, object]]]], dict | None]:
    """Load frozen schedule terrains or deterministically pre-generate them."""
    def _normalize_training_metadata(
        metadata: dict[str, object] | None,
        gaps: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None,
    ) -> dict[str, object]:
        normalized = dict(metadata or {})
        if "gaps" not in normalized:
            normalized["gaps"] = [tuple(gap) for gap in (gaps or [])]
        return normalized

    schedule_dir = cfg.train.get("terrain_schedule_dir", None)
    require_schedule_dir = bool(cfg.train.get("require_schedule_dir", False))
    allow_legacy_schedule_hashes = bool(cfg.train.get("allow_legacy_schedule_hashes", False))
    if schedule_dir:
        from .terrain_schedule import (
            load_schedule,
            schedule_train_terrains_for_gen,
            schedule_variant_matches,
        )

        schedule = load_schedule(
            schedule_dir,
            allow_legacy_hashes=allow_legacy_schedule_hashes,
        )
        validate_bilateral_schedule_semantics(schedule)
        expected_seed = int(cfg.get("seed", schedule.seed))
        expected_variant = str(cfg.terrain.variant)
        expected_gens = int(cfg.train.n_generations)
        expected_episodes = int(cfg.train.n_episodes)
        schedule_variant = str(schedule.terrain_cfg.get("variant", ""))
        generation_sizes = [len(generation) for generation in schedule.train]
        mismatches: list[str] = []
        if int(schedule.seed) != expected_seed:
            mismatches.append(f"seed expected {expected_seed}, got {schedule.seed}")
        if not schedule_variant_matches(expected_variant, schedule_variant):
            mismatches.append(f"variant expected {expected_variant!r}, got {schedule_variant!r}")
        if len(schedule.train) != expected_gens:
            mismatches.append(f"n_gens expected {expected_gens}, got {len(schedule.train)}")
        if any(size != expected_episodes for size in generation_sizes):
            mismatches.append(
                "n_episodes expected "
                f"{expected_episodes} for every generation, got {generation_sizes}"
            )
        if mismatches:
            raise ValueError(
                "Terrain schedule metadata mismatch for training schedule "
                f"{schedule_dir}: {'; '.join(mismatches)}"
            )
        log.info(
            "Terrain source | schedule=%s train_hash=%s",
            schedule_dir,
            schedule.train_hash[:16],
        )
        terrains = [
            [
                (
                    entry.grid,
                    entry.cliff_x,
                    _normalize_training_metadata(entry.metadata, entry.gaps),
                )
                for entry in schedule_train_terrains_for_gen(schedule, gen)
            ]
            for gen in range(cfg.train.n_generations)
        ]
        return terrains, {
            "dir": str(schedule_dir),
            "hash_version": int(schedule.hash_version),
            "train_hash": schedule.train_hash,
            "val_hash": schedule.val_hash,
            "test_hash": schedule.test_hash,
            "train_content_hash": schedule.train_content_hash,
            "val_content_hash": schedule.val_content_hash,
            "test_content_hash": schedule.test_content_hash,
        }

    if require_schedule_dir:
        raise ValueError(
            "This preset requires train.terrain_schedule_dir; procedural terrain fallback is disabled."
        )

    log.info(
        "Terrain source | procedural gens=%d episodes_per_gen=%d",
        cfg.train.n_generations,
        cfg.train.n_episodes,
    )
    terrains = []
    terrain_cfg = OmegaConf.to_container(cfg.terrain, resolve=True)
    assert isinstance(terrain_cfg, dict)
    for gen_idx in range(cfg.train.n_generations):
        gen_seed = int(rng.integers(0, 2**31))
        gen_cfg, _ = resolve_generation_terrain_cfg(
            terrain_cfg,
            gen_idx=gen_idx,
            n_gens=int(cfg.train.n_generations),
        )
        generation_terrains = generate_generation_terrains(
                gen_seed,
                n_episodes=cfg.train.n_episodes,
                grid_h=int(gen_cfg["grid_h"]),
                grid_w=int(gen_cfg["grid_w"]),
                platform_y=int(gen_cfg.get("platform_y", gen_cfg.get("plateau_height", 3))),
                base_width=int(gen_cfg.get("base_width", 10)),
                cliff_x_min=int(gen_cfg.get("cliff_x_min", 0)),
                cliff_x_max=int(gen_cfg.get("cliff_x_max", 0)),
                gap_widths=list(gen_cfg.get("gap_widths") or []) or None,
                gap_budget=int(gen_cfg.get("gap_budget", 0)),
                exact_gap_widths=list(gen_cfg.get("exact_gap_widths") or []) or None,
                min_gap_separation=int(gen_cfg.get("min_gap_separation", 2)),
                min_post_gap_runway=int(gen_cfg.get("min_post_gap_runway", 2)),
                hazard_mode=str(gen_cfg.get("hazard_mode", "legacy")),
                task_mode=str(gen_cfg.get("task_mode", "legacy_cliff")),
                spawn_plateau_width=(
                    int(gen_cfg["spawn_plateau_width"])
                    if gen_cfg.get("spawn_plateau_width") is not None
                    else None
                ),
                approach_width=(
                    int(gen_cfg["approach_width"])
                    if gen_cfg.get("approach_width") is not None
                    else None
                ),
                randomized_zone_width=(
                    int(gen_cfg["randomized_zone_width"])
                    if gen_cfg.get("randomized_zone_width") is not None
                    else None
                ),
                remainder_width=(
                    int(gen_cfg["remainder_width"])
                    if gen_cfg.get("remainder_width") is not None
                    else None
                ),
                single_gap_width=(
                    int(gen_cfg["single_gap_width"])
                    if gen_cfg.get("single_gap_width") is not None
                    else None
                ),
                layout_sampling=str(gen_cfg.get("layout_sampling", "uniform_legal_layouts")),
                gap_start_min=(
                    int(gen_cfg["gap_start_min"])
                    if gen_cfg.get("gap_start_min") is not None
                    else None
                ),
                gap_start_max=(
                    int(gen_cfg["gap_start_max"])
                    if gen_cfg.get("gap_start_max") is not None
                    else None
                ),
                gap_start_bands=list(gen_cfg.get("gap_start_bands") or []) or None,
                center_plateau_width=(
                    int(gen_cfg["center_plateau_width"])
                    if gen_cfg.get("center_plateau_width") is not None
                    else None
                ),
                symmetric_distance_min=(
                    int(gen_cfg["symmetric_distance_min"])
                    if gen_cfg.get("symmetric_distance_min") is not None
                    else None
                ),
                symmetric_distance_max=(
                    int(gen_cfg["symmetric_distance_max"])
                    if gen_cfg.get("symmetric_distance_max") is not None
                    else None
                ),
                goal_side=gen_cfg.get("goal_side"),
                correct_side_distance_scale=float(gen_cfg.get("correct_side_distance_scale", 1.0)),
                incorrect_side_distance_scale=float(gen_cfg.get("incorrect_side_distance_scale", 1.0)),
                cliff_front_gap_width=int(gen_cfg.get("cliff_front_gap_width", 0)),
                cliff_front_gap_widths=list(gen_cfg.get("cliff_front_gap_widths") or []) or None,
                cliff_post_gap_runway=int(gen_cfg.get("cliff_post_gap_runway", 0)),
                cliff_probe_gap_width=int(gen_cfg.get("cliff_probe_gap_width", 0)),
                cliff_probe_ledge_width=int(gen_cfg.get("cliff_probe_ledge_width", 0)),
            )
        terrains.append(
            [
                (
                    grid,
                    cliff_x,
                    _normalize_training_metadata(metadata, gaps),
                )
                for grid, cliff_x, gaps, metadata in generation_terrains
            ]
        )
    return terrains, None


def _infer_static_capacity_from_terrains(
    terrain_grids: list[np.ndarray],
) -> StaticColliderCapacity:
    """Infer fixed static-collider capacities without instantiating every environment."""
    if not terrain_grids:
        raise ValueError("Cannot infer static capacity from an empty terrain set")

    max_static_cells = 0
    max_static_points = 0
    max_grid_columns = 0
    max_cells_per_column = 0
    for terrain in terrain_grids:
        terrain_np = np.asarray(terrain, dtype=np.int32)
        max_grid_columns = max(max_grid_columns, int(terrain_np.shape[1]))
        non_empty_per_column = np.count_nonzero(terrain_np != EMPTY, axis=0)
        if non_empty_per_column.size > 0:
            max_cells_per_column = max(max_cells_per_column, int(np.max(non_empty_per_column)))
        active_y, active_x = np.nonzero(terrain_np != EMPTY)
        static_cell_count = int(active_x.shape[0])
        max_static_cells = max(max_static_cells, static_cell_count)
        if static_cell_count == 0:
            continue

        point_coords: set[tuple[int, int]] = set()
        for y, x in zip(active_y.tolist(), active_x.tolist()):
            point_coords.add((x, y))
            point_coords.add((x + 1, y))
            point_coords.add((x, y + 1))
            point_coords.add((x + 1, y + 1))
        max_static_points = max(max_static_points, len(point_coords))

    return StaticColliderCapacity(
        max_static_points=max_static_points,
        max_static_cells=max_static_cells,
        max_grid_columns=max_grid_columns,
        max_cells_per_column=max_cells_per_column,
    )


def _prepare_training_inputs(
    cfg: DictConfig,
    rng: np.random.Generator,
) -> PreparedTrainingInputs:
    """Prepare genotype dimensions, terrain batches, and shared environment assets."""
    body_mode = str(cfg.body.mode).strip().lower()
    body_shape = str(cfg.body.get("shape", "rectangle"))
    spawn_x = resolved_spawn_x(cfg)
    spawn_y = int(cfg.terrain.spawn_y)
    robot_morph = make_robot_body(cfg.terrain.robot_w, cfg.terrain.robot_h, shape=body_shape)
    alive_mask = jnp.asarray(fixed_body_alive_mask(robot_morph), dtype=jnp.bool_)
    genotype_dim = genotype_dim_from_cfg(cfg)

    all_terrains, terrain_schedule_info = _prepare_training_terrains(cfg, rng)
    terrain_grids = [
        np.asarray(grid, dtype=np.int32)
        for generation_terrains in all_terrains
        for grid, _cliff_x, _metadata in generation_terrains
    ]
    collision_robot = robot_morph
    dynamic_core: CliffDynamicCore | None = None
    grid_data = None
    generation_static_capacity: StaticColliderCapacity | None = _infer_static_capacity_from_terrains(
        terrain_grids,
    )

    if body_mode == "grown":
        canvas_h = int(cfg.body.growth.max_robot_h)
        canvas_w = int(cfg.body.growth.max_robot_w)
        max_h, max_w = max_canvas_size(
            int(cfg.terrain.grid_h),
            int(cfg.terrain.grid_w),
            spawn_x=spawn_x,
            spawn_y=spawn_y,
        )
        if canvas_h > max_h or canvas_w > max_w:
            raise ValueError("Configured grown body canvas does not fit within the terrain grid")
        collision_robot = make_robot_body(canvas_w, canvas_h)
        alive_mask = jnp.ones((canvas_h, canvas_w), dtype=jnp.bool_)
        grid_data = precompute_shared_grid_data(cfg.terrain.grid_h, cfg.terrain.grid_w)

    collision_tuning = resolve_collision_tuning_from_terrains(
        terrain_grids,
        grid_h=int(cfg.terrain.grid_h),
        grid_w=int(cfg.terrain.grid_w),
        robot_morph=collision_robot,
        spawn_x=spawn_x,
        spawn_y=spawn_y,
        mode=str(cfg.collision.tuning.mode),
    )
    if body_mode == "fixed":
        log.info(
            "Environment build | generations=%d environments=%d mode=lazy_generation_cache",
            cfg.train.n_generations,
            cfg.train.n_generations * cfg.train.n_episodes,
        )
        dynamic_core = build_cliff_dynamic_core(
            robot_morph,
            spawn_x=spawn_x,
            spawn_y=spawn_y,
            physics_substeps=int(cfg.sim.physics_substeps),
            actuator_convergence=float(cfg.sim.actuator_convergence),
            spring_stiffness_scale=float(cfg.sim.spring_stiffness_scale),
            floor_static_friction_const=float(cfg.sim.floor_static_friction_const),
            floor_stiction_stiffness=float(cfg.sim.floor_stiction_stiffness),
            floor_stiction_switch_speed=float(cfg.sim.floor_stiction_switch_speed),
            collision_build_capacity=collision_tuning.runtime_build_capacity,
        )
    else:
        log.info(
            "Environment build | generations=%d environments=%d mode=grown_static_templates",
            cfg.train.n_generations,
            cfg.train.n_generations * cfg.train.n_episodes,
        )

    return PreparedTrainingInputs(
        body_mode=body_mode,
        genotype_dim=genotype_dim,
        alive_mask=alive_mask,
        terrain_generations=all_terrains,
        dynamic_core=dynamic_core,
        robot_morph=None if body_mode == "grown" else robot_morph,
        grid_data=grid_data,
        spawn_x=spawn_x,
        spawn_y=spawn_y,
        spring_stiffness_scale=float(cfg.sim.spring_stiffness_scale),
        generation_static_capacity=generation_static_capacity,
        terrain_schedule_info=terrain_schedule_info,
        collision_tuning=collision_tuning,
    )


def _create_archive(cfg: DictConfig) -> GridArchive:
    """Create an empty QD archive from config."""
    resolved_archive = resolve_qd_archive(cfg)
    return GridArchive(
        dims=resolved_archive.dims,
        ranges=list(resolved_archive.ranges),
        learning_rate=cfg.qd.archive.learning_rate,
        threshold_min=cfg.qd.archive.threshold_min,
        descriptor_names=resolved_archive.descriptor_names,
        descriptor_labels=resolved_archive.descriptor_labels,
    )


def _resolve_focus_config(
    cfg: DictConfig,
    *,
    descriptor_names: tuple[str, ...],
) -> tuple[FocusConfig, dict[str, int]]:
    focus_cfg = focus_config_from_section(cfg.qd.get("focus"))
    if not focus_cfg.enabled:
        return focus_cfg, {}
    return focus_cfg, validate_focus_config(focus_cfg, descriptor_names)


def _create_emitters(
    cfg: DictConfig,
    genotype_dim: int,
    key: jax.Array,
) -> tuple[list[CMAEmitter], jax.Array]:
    """Initialize emitters and return the updated JAX key."""
    emitters = []
    emitter_kwargs = emitter_kwargs_from_cfg(cfg)
    for _ in range(cfg.qd.n_emitters):
        emitter = CMAEmitter(genotype_length=genotype_dim, **emitter_kwargs)
        key, init_key = jax.random.split(key)
        emitter.init(init_key)
        emitters.append(emitter)
    return emitters, key


def _canonical_checkpoint_prefix(path: str | Path) -> Path:
    """Normalize checkpoint paths to the shared prefix-without-extension form."""
    prefix = Path(path)
    if prefix.suffix in {".json", ".npz", ".done"}:
        return prefix.with_suffix("")
    return prefix


def _checkpoint_within_output_dir(path_prefix: str | Path, output_dir: Path) -> bool:
    """Return whether a checkpoint prefix is rooted inside one run output dir."""
    output_root = output_dir.resolve()
    checkpoint_prefix = _canonical_checkpoint_prefix(path_prefix).resolve()
    return output_root == checkpoint_prefix or output_root in checkpoint_prefix.parents


def _resume_checkpoint_meta(
    path_prefix: str | Path,
    *,
    expected_config_sha256: str | None,
) -> tuple[str, dict]:
    """Load explicit checkpoint metadata and verify it matches the resolved config."""
    prefix = _canonical_checkpoint_prefix(path_prefix)
    try:
        meta = load_checkpoint_meta(prefix)
    except OSError as exc:
        raise FileNotFoundError(f"Resume checkpoint metadata not found: {prefix}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Resume checkpoint metadata unreadable: {prefix}") from exc

    checkpoint_config_sha256 = str(meta.get("config_sha256", "")).strip()
    if (
        expected_config_sha256
        and checkpoint_config_sha256
        and checkpoint_config_sha256 != expected_config_sha256
    ):
        raise ValueError(
            "Resume checkpoint config hash does not match the current resolved config. "
            f"checkpoint={checkpoint_config_sha256} current={expected_config_sha256}"
        )
    return str(prefix), meta


def _manifest_logical_hash(manifest: dict) -> str:
    config_block = manifest.get("config", {})
    logical_hash = str(config_block.get("logical_config_sha256", "")).strip()
    if logical_hash:
        return logical_hash
    return str(config_block.get("resolved_config_sha256", "")).strip()


def _default_lineage(*, mode: str) -> dict:
    return {
        "mode": str(mode),
        "parent_run_id": None,
        "parent_checkpoint_path": None,
        "parent_completed_generations": None,
        "parent_schedule_hashes": None,
    }


def _output_dir_from_checkpoint(path_prefix: str | Path) -> Path:
    checkpoint_prefix = _canonical_checkpoint_prefix(path_prefix)
    return checkpoint_prefix.parent.parent


def _load_resolved_config(output_dir: str | Path) -> DictConfig:
    config_path = Path(output_dir) / "resolved_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Resolved config not found: {config_path}")
    return OmegaConf.load(config_path)


def _validate_branch_config_delta(parent_cfg: DictConfig, child_cfg: DictConfig) -> None:
    changed_paths = diff_config_paths(parent_cfg, child_cfg)
    disallowed = sorted(
        path for path in changed_paths if not path_is_allowed(path, BRANCH_ALLOWED_CONFIG_PATHS)
    )
    if disallowed:
        raise ValueError(
            "Branch continuation changes config paths outside the allowed delta surface: "
            + ", ".join(disallowed)
        )


def _stage_artifact_dir(output_dir: Path, *, generation: int) -> Path:
    return output_dir / "stages" / f"gen_{int(generation):04d}"


def _phase_artifact_root(artifact_output_dir: Path, *, final_stage: bool) -> Path:
    return artifact_output_dir / ("post_train" if final_stage else "stage_eval")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def _audit_enabled(cfg: DictConfig) -> bool:
    return bool(cfg.get("audit", {}).get("enabled", False))


def _audit_flag(cfg: DictConfig, key: str, default: bool = False) -> bool:
    return bool(cfg.get("audit", {}).get(key, default))


def _audit_output_dir(output_dir: Path, cfg: DictConfig) -> Path:
    root = Path(str(cfg.get("audit", {}).get("output_root", "audit")))
    if root.is_absolute():
        return root / output_dir.name
    return output_dir / root


def _generation_eval_mode(cfg: DictConfig) -> str:
    return str(
        cfg.get("audit", {}).get("generation_eval_mode", "reduced_host")
    ).strip().lower()


def _generation_eval_reduced_host(cfg: DictConfig) -> bool:
    return _generation_eval_mode(cfg) != "full_host"


def _max_generation_probe_winners(cfg: DictConfig) -> int:
    return int(cfg.get("audit", {}).get("max_generation_probe_winners", 16))


def _genotype_sha256(genotype: np.ndarray) -> str:
    arr = np.asarray(genotype, dtype=np.float32)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _candidate_metric_values(batch_result, attr: str) -> np.ndarray | None:
    descriptor_columns = getattr(batch_result, "descriptor_columns", None) or {}
    if hasattr(batch_result, attr):
        return np.asarray(getattr(batch_result, attr), dtype=np.float32).reshape(-1)
    if attr in descriptor_columns:
        return np.asarray(descriptor_columns[attr], dtype=np.float32).reshape(-1)
    return None


def _candidate_metric_scalar(batch_result, attr: str, idx: int) -> float | None:
    values = _candidate_metric_values(batch_result, attr)
    if values is None or idx < 0 or idx >= int(values.shape[0]):
        return None
    return float(values[idx])


def _candidate_eval_fields(
    *,
    prefix: str,
    batch_result,
    fitnesses: np.ndarray,
    descriptors: np.ndarray,
    idx: int,
) -> dict[str, object]:
    fitness_arr = np.asarray(fitnesses, dtype=np.float32).reshape(-1)
    descriptor_arr = np.asarray(descriptors, dtype=np.float32)
    return {
        f"{prefix}_fitness": float(fitness_arr[idx]),
        f"{prefix}_descriptor": np.asarray(descriptor_arr[idx], dtype=np.float32).tolist(),
        f"{prefix}_goal_hit_count": _candidate_metric_scalar(batch_result, "goal_hit_count", idx),
        f"{prefix}_fall_count": _candidate_metric_scalar(batch_result, "fall_count", idx),
        f"{prefix}_completion_time": _candidate_metric_scalar(batch_result, "time_to_edge", idx),
        f"{prefix}_mirrored_pair_count": _candidate_metric_scalar(
            batch_result, "mirrored_pair_count", idx
        ),
    }


def _fitness_match(lhs: float | None, rhs: float | None) -> bool | None:
    if lhs is None or rhs is None:
        return None
    return bool(np.isclose(float(lhs), float(rhs), atol=1.0e-6, rtol=1.0e-6))


def _descriptor_match(lhs: np.ndarray | None, rhs: np.ndarray | None) -> bool | None:
    if lhs is None or rhs is None:
        return None
    return bool(
        np.allclose(
            np.asarray(lhs, dtype=np.float32),
            np.asarray(rhs, dtype=np.float32),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
    )


def _fitness_delta(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    return float(rhs - lhs)


def _evaluate_probe_candidates(
    *,
    genotypes: np.ndarray,
    candidate_keys: np.ndarray,
    env_datas: list[CliffEnvData] | GrownEvalAssets,
    alive_mask: jnp.ndarray,
    cfg: DictConfig,
    runtime_resolution: RuntimeResolution,
    static_capacity: StaticColliderCapacity | None,
    reduced_host_result: bool,
):
    return evaluate_population_chunked(
        genotypes,
        env_datas,
        alive_mask,
        jnp.asarray(candidate_keys),
        cfg,
        workload="generation",
        runtime_resolution=runtime_resolution,
        static_capacity=static_capacity,
        reduced_host_result=reduced_host_result,
    )


def _floor_stiction_is_disabled(cfg: DictConfig) -> bool:
    return (
        float(cfg.sim.floor_static_friction_const) == 0.0
        and float(cfg.sim.floor_stiction_stiffness) == 0.0
        and float(cfg.sim.floor_stiction_switch_speed) == 0.0
    )


def _replace_floor_stiction_constants(
    constants,
    *,
    floor_static_friction_const: float,
    floor_stiction_stiffness: float,
    floor_stiction_switch_speed: float,
):
    return constants._replace(
        floor_static_friction_const=float(floor_static_friction_const),
        floor_stiction_stiffness=float(floor_stiction_stiffness),
        floor_stiction_switch_speed=float(floor_stiction_switch_speed),
    )


def _clone_env_datas_with_floor_stiction(
    env_datas: list[CliffEnvData] | list[CliffEvalEnvData],
    *,
    floor_static_friction_const: float,
    floor_stiction_stiffness: float,
    floor_stiction_switch_speed: float,
) -> list[CliffEnvData] | list[CliffEvalEnvData]:
    cloned_env_datas = []
    for env_data in env_datas:
        cloned_env_datas.append(
            env_data._replace(
                constants=_replace_floor_stiction_constants(
                    env_data.constants,
                    floor_static_friction_const=floor_static_friction_const,
                    floor_stiction_stiffness=floor_stiction_stiffness,
                    floor_stiction_switch_speed=floor_stiction_switch_speed,
                )
            )
        )
    return cloned_env_datas


def _trace_probe_candidates(
    *,
    genotypes: np.ndarray,
    candidate_keys: np.ndarray,
    env_datas: list[CliffEnvData] | GrownEvalAssets,
    alive_mask: jnp.ndarray,
    cfg: DictConfig,
    runtime_resolution: RuntimeResolution,
    static_capacity: StaticColliderCapacity | None,
    workload: str = "generation",
):
    return trace_population_chunked(
        genotypes,
        env_datas,
        alive_mask,
        jnp.asarray(candidate_keys),
        cfg,
        workload=workload,
        runtime_resolution=runtime_resolution,
        static_capacity=static_capacity,
    )


def _materialize_trace_batch_result(
    *,
    trace_eval,
    alive_mask: jnp.ndarray,
    cfg: DictConfig,
) -> tuple[object, np.ndarray, np.ndarray]:
    alive_count = float(jnp.sum(alive_mask.astype(jnp.float32)))
    trace_batch_result = candidate_batch_fitness(
        trace_eval.episode_results,
        alive_cell_count=jnp.full(
            (int(np.asarray(trace_eval.episode_results.peak_front_x).shape[0]),),
            alive_count,
            dtype=jnp.float32,
        ),
        activation_cost_penalty_scale=float(
            cfg.train.get("activation_cost_penalty_scale", 0.0)
        ),
        soft_loft_penalty_scale=float(
            cfg.train.get("soft_loft_penalty_scale", 0.0)
        ),
    )
    jax.block_until_ready(trace_batch_result.fitness)
    host_batch_result = jax.tree_util.tree_map(
        lambda array: np.asarray(array),
        trace_batch_result,
    )
    descriptor_specs = resolve_qd_archive(cfg).descriptor_specs
    descriptors = build_descriptor_matrix(host_batch_result, descriptor_specs)
    return (
        host_batch_result,
        np.asarray(host_batch_result.fitness, dtype=np.float32),
        np.asarray(descriptors, dtype=np.float32),
    )


def _candidate_probe_fitness(
    *,
    episode_results,
    alive_mask: jnp.ndarray,
    cfg: DictConfig,
) -> float:
    alive_count = float(jnp.sum(alive_mask.astype(jnp.float32)))
    candidate_episode_results = jax.tree_util.tree_map(
        lambda array: jnp.asarray(array)[None, ...],
        episode_results,
    )
    batch_result = candidate_batch_fitness(
        candidate_episode_results,
        alive_cell_count=jnp.asarray([alive_count], dtype=jnp.float32),
        activation_cost_penalty_scale=float(
            cfg.train.get("activation_cost_penalty_scale", 0.0)
        ),
        soft_loft_penalty_scale=float(
            cfg.train.get("soft_loft_penalty_scale", 0.0)
        ),
    )
    return float(np.asarray(jax.device_get(batch_result.fitness), dtype=np.float32)[0])


def _episode_probe_fitnesses(episode_results) -> np.ndarray:
    episode_fitnesses = episode_fitness_array(
        jax.tree_util.tree_map(lambda array: jnp.asarray(array), episode_results)
    )
    return np.asarray(jax.device_get(episode_fitnesses), dtype=np.float32).reshape(-1)


def _arrays_close(lhs: np.ndarray, rhs: np.ndarray) -> bool:
    lhs_arr = np.asarray(lhs)
    rhs_arr = np.asarray(rhs)
    if lhs_arr.dtype == np.bool_ or rhs_arr.dtype == np.bool_:
        return bool(np.array_equal(lhs_arr.astype(bool), rhs_arr.astype(bool)))
    if np.issubdtype(lhs_arr.dtype, np.integer) and np.issubdtype(
        rhs_arr.dtype, np.integer
    ):
        return bool(np.array_equal(lhs_arr, rhs_arr))
    return bool(
        np.allclose(
            lhs_arr.astype(np.float32),
            rhs_arr.astype(np.float32),
            atol=1.0e-6,
            rtol=1.0e-6,
            equal_nan=True,
        )
    )


def _max_abs_diff(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_arr = np.asarray(lhs)
    rhs_arr = np.asarray(rhs)
    if lhs_arr.dtype == np.bool_ or rhs_arr.dtype == np.bool_:
        return float(np.max(lhs_arr.astype(np.int8) != rhs_arr.astype(np.int8)))
    return float(
        np.max(
            np.abs(lhs_arr.astype(np.float32) - rhs_arr.astype(np.float32)),
            initial=0.0,
        )
    )


def _phase_trace_category(stage: str) -> str:
    if stage in {
        "initial_positions",
        "initial_velocities",
        "initial_spring_rest_length",
        "phase_pre_positions",
        "phase_pre_velocities",
        "phase_pre_spring_rest_length",
    }:
        return "pre_physics_state"
    if stage in {
        "initial_nca_state",
        "initial_ctrnn_carry",
        "phase_h_compact",
        "phase_v_compact",
        "phase_controller_hidden_output_max_abs",
        "phase_action_output_max_abs",
    }:
        return "controller"
    if stage in {
        "phase_post_positions",
        "phase_post_velocities",
        "phase_post_spring_rest_length",
    }:
        return "physics_or_collision"
    return "unknown"


def _first_phase_trace_divergence(trace_first, trace_second) -> dict[str, object]:
    initial_fields = (
        "initial_positions",
        "initial_velocities",
        "initial_spring_rest_length",
        "initial_nca_state",
        "initial_ctrnn_carry",
    )
    for field in initial_fields:
        lhs = np.asarray(getattr(trace_first, field))
        rhs = np.asarray(getattr(trace_second, field))
        if not _arrays_close(lhs, rhs):
            return {
                "category": _phase_trace_category(field),
                "stage": field,
                "step": None,
                "phase": None,
                "max_abs_diff": _max_abs_diff(lhs, rhs),
            }

    phase_fields = (
        "phase_pre_positions",
        "phase_pre_velocities",
        "phase_pre_spring_rest_length",
        "phase_h_compact",
        "phase_v_compact",
        "phase_controller_hidden_output_max_abs",
        "phase_action_output_max_abs",
        "phase_post_positions",
        "phase_post_velocities",
        "phase_post_spring_rest_length",
    )
    n_steps, n_phases = np.asarray(trace_first.phase_done_after).shape[:2]
    for step in range(int(n_steps)):
        for phase in range(int(n_phases)):
            for field in phase_fields:
                lhs = np.asarray(getattr(trace_first, field)[step, phase])
                rhs = np.asarray(getattr(trace_second, field)[step, phase])
                if not _arrays_close(lhs, rhs):
                    return {
                        "category": _phase_trace_category(field),
                        "stage": field,
                        "step": int(step),
                        "phase": int(phase),
                        "max_abs_diff": _max_abs_diff(lhs, rhs),
                    }
    return {
        "category": "none",
        "stage": None,
        "step": None,
        "phase": None,
        "max_abs_diff": 0.0,
    }


def _phase_probe_write_site_diagnostics(
    trace_first,
    trace_second,
    *,
    step: int | None,
    phase: int | None,
) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "selected_episode_phase_write_site_first_divergent_field": None,
        "selected_episode_phase_write_site_first_divergent_category": "none",
        "selected_episode_phase_write_site_last_substep_post_integrator_velocities_repeat_delta": 0.0,
        "selected_episode_phase_write_site_last_substep_final_velocities_repeat_delta": 0.0,
        "selected_episode_phase_write_site_last_substep_final_velocities_true_repeat_delta": 0.0,
        "selected_episode_phase_write_site_post_velocities_repeat_delta": 0.0,
        "selected_episode_phase_write_site_post_velocities_true_repeat_delta": 0.0,
        "selected_episode_phase_write_site_repeat_1_post_vs_last_substep_final_velocities_delta": 0.0,
        "selected_episode_phase_write_site_repeat_2_post_vs_last_substep_final_velocities_delta": 0.0,
        "selected_episode_phase_write_site_repeat_1_post_true_vs_last_substep_final_true_delta": 0.0,
        "selected_episode_phase_write_site_repeat_2_post_true_vs_last_substep_final_true_delta": 0.0,
    }
    if step is None or phase is None:
        return diagnostics

    step_i = int(step)
    phase_i = int(phase)

    field_pairs = (
        (
            "phase_last_substep_post_integrator_velocities",
            "selected_episode_phase_write_site_last_substep_post_integrator_velocities_repeat_delta",
            "integrator",
        ),
        (
            "phase_last_substep_final_velocities_true",
            "selected_episode_phase_write_site_last_substep_final_velocities_true_repeat_delta",
            "final_state_true",
        ),
        (
            "phase_last_substep_final_velocities",
            "selected_episode_phase_write_site_last_substep_final_velocities_repeat_delta",
            "final_state",
        ),
        (
            "phase_post_velocities_true",
            "selected_episode_phase_write_site_post_velocities_true_repeat_delta",
            "phase_trace_write_true",
        ),
        (
            "phase_post_velocities",
            "selected_episode_phase_write_site_post_velocities_repeat_delta",
            "phase_trace_write",
        ),
    )
    for field_name, summary_key, category in field_pairs:
        lhs = np.asarray(getattr(trace_first, field_name)[step_i, phase_i])
        rhs = np.asarray(getattr(trace_second, field_name)[step_i, phase_i])
        diagnostics[summary_key] = _max_abs_diff(lhs, rhs)
        if (
            diagnostics["selected_episode_phase_write_site_first_divergent_field"]
            is None
            and not _arrays_close(lhs, rhs)
        ):
            diagnostics["selected_episode_phase_write_site_first_divergent_field"] = (
                field_name
            )
            diagnostics[
                "selected_episode_phase_write_site_first_divergent_category"
            ] = category

    repeat_1_post = np.asarray(trace_first.phase_post_velocities[step_i, phase_i])
    repeat_2_post = np.asarray(trace_second.phase_post_velocities[step_i, phase_i])
    repeat_1_post_true = np.asarray(
        trace_first.phase_post_velocities_true[step_i, phase_i]
    )
    repeat_2_post_true = np.asarray(
        trace_second.phase_post_velocities_true[step_i, phase_i]
    )
    repeat_1_last_final = np.asarray(
        trace_first.phase_last_substep_final_velocities[step_i, phase_i]
    )
    repeat_2_last_final = np.asarray(
        trace_second.phase_last_substep_final_velocities[step_i, phase_i]
    )
    repeat_1_last_final_true = np.asarray(
        trace_first.phase_last_substep_final_velocities_true[step_i, phase_i]
    )
    repeat_2_last_final_true = np.asarray(
        trace_second.phase_last_substep_final_velocities_true[step_i, phase_i]
    )
    diagnostics[
        "selected_episode_phase_write_site_repeat_1_post_vs_last_substep_final_velocities_delta"
    ] = _max_abs_diff(repeat_1_post, repeat_1_last_final)
    diagnostics[
        "selected_episode_phase_write_site_repeat_2_post_vs_last_substep_final_velocities_delta"
    ] = _max_abs_diff(repeat_2_post, repeat_2_last_final)
    diagnostics[
        "selected_episode_phase_write_site_repeat_1_post_true_vs_last_substep_final_true_delta"
    ] = _max_abs_diff(repeat_1_post_true, repeat_1_last_final_true)
    diagnostics[
        "selected_episode_phase_write_site_repeat_2_post_true_vs_last_substep_final_true_delta"
    ] = _max_abs_diff(repeat_2_post_true, repeat_2_last_final_true)
    return diagnostics


def _substep_trace_category(stage: str) -> str:
    if stage in {
        "phase_pre_positions",
        "phase_pre_velocities",
        "phase_pre_spring_rest_length",
    }:
        return "pre_substep_state"
    if stage in {
        "phase_h_compact",
        "phase_v_compact",
        "phase_controller_hidden_output_max_abs",
        "phase_action_output_max_abs",
    }:
        return "controller"
    if stage in {
        "substep_dynamic_selected_pair_mask",
        "substep_dynamic_overflow",
        "substep_dynamic_selected_triple_active",
        "substep_dynamic_triple_active",
    }:
        return "dynamic_broadphase_selection"
    if stage == "substep_dynamic_forces":
        return "dynamic_collision_forces"
    if stage in {
        "substep_static_forces",
        "substep_static_contact_mask",
        "substep_static_memory_contact_mask",
        "substep_static_manifold_active",
    }:
        return "static_collision"
    if stage == "substep_tangential_deformation":
        return "tangential_deformation"
    if stage == "substep_post_update_actuator_spring_rest_length":
        return "actuator_update"
    if stage in {
        "substep_post_integrator_positions",
        "substep_post_integrator_velocities",
    }:
        return "integrator"
    if stage == "substep_post_constraint_positions":
        return "edge_constraints"
    if stage == "substep_post_anchor_positions":
        return "friction_anchor"
    if stage in {
        "substep_velocity_true_positions_last",
        "substep_velocity_true_displacement",
    }:
        return "velocity_true_reconstruction"
    if stage in {
        "substep_final_positions",
        "substep_final_velocities",
        "substep_final_velocities_true",
    }:
        return "final_state"
    return "unknown"


def _first_substep_trace_divergence(trace_first, trace_second) -> dict[str, object]:
    phase_fields = (
        "phase_pre_positions",
        "phase_pre_velocities",
        "phase_pre_spring_rest_length",
        "phase_h_compact",
        "phase_v_compact",
        "phase_controller_hidden_output_max_abs",
        "phase_action_output_max_abs",
    )
    for field in phase_fields:
        lhs = np.asarray(getattr(trace_first, field))
        rhs = np.asarray(getattr(trace_second, field))
        if not _arrays_close(lhs, rhs):
            return {
                "category": _substep_trace_category(field),
                "stage": field,
                "substep": None,
                "max_abs_diff": _max_abs_diff(lhs, rhs),
            }

    substep_fields = (
        "substep_dynamic_selected_pair_mask",
        "substep_dynamic_overflow",
        "substep_dynamic_selected_triple_active",
        "substep_dynamic_triple_active",
        "substep_dynamic_forces",
        "substep_static_forces",
        "substep_static_contact_mask",
        "substep_static_memory_contact_mask",
        "substep_static_manifold_active",
        "substep_tangential_deformation",
        "substep_post_update_actuator_spring_rest_length",
        "substep_post_integrator_positions",
        "substep_post_integrator_velocities",
        "substep_post_constraint_positions",
        "substep_post_anchor_positions",
        "substep_final_positions",
        "substep_velocity_true_positions_last",
        "substep_velocity_true_displacement",
        "substep_final_velocities",
        "substep_final_velocities_true",
    )
    n_substeps = int(np.asarray(trace_first.substep_dynamic_overflow).shape[0])
    for substep in range(n_substeps):
        for field in substep_fields:
            lhs = np.asarray(getattr(trace_first, field)[substep])
            rhs = np.asarray(getattr(trace_second, field)[substep])
            if not _arrays_close(lhs, rhs):
                return {
                    "category": _substep_trace_category(field),
                    "stage": field,
                    "substep": int(substep),
                    "max_abs_diff": _max_abs_diff(lhs, rhs),
                }
    return {
        "category": "none",
        "stage": None,
        "substep": None,
        "max_abs_diff": 0.0,
    }


def _previous_phase_boundary(
    step: int | None,
    phase: int | None,
    *,
    phase_count: int,
) -> tuple[int, int] | None:
    if step is None or phase is None:
        return None
    step_i = int(step)
    phase_i = int(phase)
    phase_count_i = int(phase_count)
    if phase_count_i <= 0 or step_i < 0 or phase_i < 0:
        return None
    if phase_i > 0:
        return step_i, phase_i - 1
    if step_i == 0:
        return None
    return step_i - 1, phase_count_i - 1


def _probe_selected_episode_substeps_with_backtrack(
    *,
    selected_genotype: np.ndarray,
    env_datas: list[CliffEnvData] | GrownEvalAssets,
    alive_mask: jnp.ndarray,
    selected_key: np.ndarray,
    cfg: DictConfig,
    selected_episode_index: int,
    selected_step_index: int,
    selected_phase_index: int,
    phase_count: int,
    runtime_resolution: RuntimeResolution,
    static_capacity: StaticColliderCapacity | None,
) -> tuple[
    dict[str, object],
    CandidateEpisodeSubstepProbeResult,
    CandidateEpisodeSubstepProbeResult,
]:
    current_step_index = int(selected_step_index)
    current_phase_index = int(selected_phase_index)
    max_attempts = max(
        1,
        (current_step_index + 1) * max(1, int(phase_count)),
    )
    attempts: list[dict[str, object]] = []
    backtrack_count = 0
    backtrack_exhausted = False
    final_first: CandidateEpisodeSubstepProbeResult | None = None
    final_second: CandidateEpisodeSubstepProbeResult | None = None
    final_divergence: dict[str, object] = {
        "category": None,
        "stage": None,
        "substep": None,
        "max_abs_diff": 0.0,
    }
    for attempt_index in range(max_attempts):
        first_probe = probe_candidate_episode_substeps(
            selected_genotype,
            env_datas,
            alive_mask,
            selected_key,
            cfg,
            selected_episode_index=selected_episode_index,
            selected_step_index=current_step_index,
            selected_phase_index=current_phase_index,
            runtime_resolution=runtime_resolution,
            static_capacity=static_capacity,
        )
        second_probe = probe_candidate_episode_substeps(
            selected_genotype,
            env_datas,
            alive_mask,
            selected_key,
            cfg,
            selected_episode_index=selected_episode_index,
            selected_step_index=current_step_index,
            selected_phase_index=current_phase_index,
            runtime_resolution=runtime_resolution,
            static_capacity=static_capacity,
        )
        divergence = _first_substep_trace_divergence(
            first_probe.substep_trace,
            second_probe.substep_trace,
        )
        attempts.append(
            {
                "attempt_index": int(attempt_index),
                "step_index": int(current_step_index),
                "phase_index": int(current_phase_index),
                "mismatch_category": divergence["category"],
                "mismatch_stage": divergence["stage"],
                "mismatch_substep": divergence["substep"],
                "mismatch_max_abs_diff": divergence["max_abs_diff"],
            }
        )
        final_first = first_probe
        final_second = second_probe
        final_divergence = divergence
        if (
            divergence["category"] == "pre_substep_state"
            and divergence["stage"] == "phase_pre_velocities"
        ):
            previous_boundary = _previous_phase_boundary(
                current_step_index,
                current_phase_index,
                phase_count=phase_count,
            )
            if previous_boundary is None:
                backtrack_exhausted = True
                break
            current_step_index, current_phase_index = previous_boundary
            backtrack_count += 1
            continue
        break
    if final_first is None or final_second is None:
        raise RuntimeError("Selected substep probe did not execute")
    return (
        {
            "selected_episode_substep_probe_initial_step_index": int(
                selected_step_index
            ),
            "selected_episode_substep_probe_initial_phase_index": int(
                selected_phase_index
            ),
            "selected_episode_substep_probe_step_index": int(current_step_index),
            "selected_episode_substep_probe_phase_index": int(current_phase_index),
            "selected_episode_substep_probe_backtrack_count": int(backtrack_count),
            "selected_episode_substep_probe_backtrack_exhausted": bool(
                backtrack_exhausted
            ),
            "selected_episode_substep_probe_attempts": attempts,
            "selected_episode_substep_first_mismatch_category": final_divergence[
                "category"
            ],
            "selected_episode_substep_first_mismatch_stage": final_divergence[
                "stage"
            ],
            "selected_episode_substep_first_mismatch_substep": final_divergence[
                "substep"
            ],
            "selected_episode_substep_first_mismatch_max_abs_diff": final_divergence[
                "max_abs_diff"
            ],
        },
        final_first,
        final_second,
    )


def _write_internal_phase_divergence_probe_artifacts(
    *,
    audit_dir: Path,
    generation: int,
    candidate_genotypes: np.ndarray,
    candidate_keys: np.ndarray,
    main_fitnesses: np.ndarray,
    decision: GridArchiveBatchDecision,
    repeat_eval_first,
    cfg: DictConfig,
    env_datas: list[CliffEnvData] | GrownEvalAssets,
    alive_mask: jnp.ndarray,
    runtime_resolution: RuntimeResolution,
    static_capacity: StaticColliderCapacity | None,
) -> dict[str, object] | None:
    if not _audit_flag(cfg, "probe_internal_phase_divergence", False):
        return None

    summary_path = audit_dir / f"gen_{int(generation):04d}_internal_phase_probe_summary.json"
    rows_path = audit_dir / f"gen_{int(generation):04d}_internal_phase_probe_episodes.jsonl"
    if repeat_eval_first is None:
        raise ValueError(
            "audit.probe_internal_phase_divergence requires repeat_full_generation_batch"
        )
    if str(cfg.body.mode).strip().lower() == "grown" or isinstance(
        env_datas, GrownEvalAssets
    ):
        summary_payload = {
            "generation": int(generation),
            "skipped_reason": "body_mode_grown_not_supported",
        }
        _write_json(summary_path, summary_payload)
        _write_jsonl(rows_path, [])
        summary_payload["summary_path"] = str(summary_path)
        summary_payload["rows_path"] = str(rows_path)
        return summary_payload

    repeat_fitnesses = np.asarray(repeat_eval_first.fitnesses, dtype=np.float32).reshape(-1)
    main_fitnesses_arr = np.asarray(main_fitnesses, dtype=np.float32).reshape(-1)
    candidate_deltas = np.abs(repeat_fitnesses - main_fitnesses_arr)
    selected_candidate_idx = int(np.argmax(candidate_deltas))
    selected_genotype = np.asarray(
        candidate_genotypes[selected_candidate_idx], dtype=np.float32
    )
    selected_key = np.asarray(candidate_keys[selected_candidate_idx], dtype=np.uint32)
    accepted_flat_index_by_candidate = {
        int(candidate_idx): int(flat_idx)
        for candidate_idx, flat_idx in zip(
            np.asarray(decision.accepted_indices, dtype=np.int64).tolist(),
            np.asarray(decision.accepted_flat_indices, dtype=np.int64).tolist(),
            strict=False,
        )
    }

    isolated_eval_first = _evaluate_probe_candidates(
        genotypes=selected_genotype[None, ...],
        candidate_keys=selected_key[None, ...],
        env_datas=env_datas,
        alive_mask=alive_mask,
        cfg=cfg,
        runtime_resolution=runtime_resolution,
        static_capacity=static_capacity,
        reduced_host_result=_generation_eval_reduced_host(cfg),
    )
    isolated_eval_second = _evaluate_probe_candidates(
        genotypes=selected_genotype[None, ...],
        candidate_keys=selected_key[None, ...],
        env_datas=env_datas,
        alive_mask=alive_mask,
        cfg=cfg,
        runtime_resolution=runtime_resolution,
        static_capacity=static_capacity,
        reduced_host_result=_generation_eval_reduced_host(cfg),
    )

    phase_probe_first: CandidateEpisodePhaseProbeResult = probe_candidate_episode_phases(
        selected_genotype,
        env_datas,
        alive_mask,
        selected_key,
        cfg,
        runtime_resolution=runtime_resolution,
        static_capacity=static_capacity,
    )
    phase_probe_second: CandidateEpisodePhaseProbeResult = probe_candidate_episode_phases(
        selected_genotype,
        env_datas,
        alive_mask,
        selected_key,
        cfg,
        runtime_resolution=runtime_resolution,
        static_capacity=static_capacity,
    )

    phase_probe_candidate_fitness_first = _candidate_probe_fitness(
        episode_results=phase_probe_first.episode_results,
        alive_mask=alive_mask,
        cfg=cfg,
    )
    phase_probe_candidate_fitness_second = _candidate_probe_fitness(
        episode_results=phase_probe_second.episode_results,
        alive_mask=alive_mask,
        cfg=cfg,
    )
    episode_fitnesses_first = _episode_probe_fitnesses(phase_probe_first.episode_results)
    episode_fitnesses_second = _episode_probe_fitnesses(phase_probe_second.episode_results)
    episode_deltas = np.abs(episode_fitnesses_second - episode_fitnesses_first)

    episode_count = int(episode_fitnesses_first.shape[0])
    rows: list[dict[str, object]] = []
    category_counts = {
        "pre_physics_state": 0,
        "controller": 0,
        "physics_or_collision": 0,
        "none": 0,
        "unknown": 0,
    }
    for episode_idx in range(episode_count):
        trace_first_env = jax.tree_util.tree_map(
            lambda array: np.asarray(array[episode_idx]),
            phase_probe_first.phase_trace,
        )
        trace_second_env = jax.tree_util.tree_map(
            lambda array: np.asarray(array[episode_idx]),
            phase_probe_second.phase_trace,
        )
        divergence = _first_phase_trace_divergence(trace_first_env, trace_second_env)
        category_counts[str(divergence["category"])] = (
            category_counts.get(str(divergence["category"]), 0) + 1
        )
        rows.append(
            {
                "generation": int(generation),
                "candidate_index": int(selected_candidate_idx),
                "episode_index": int(episode_idx),
                "probe_repeat_1_episode_fitness": float(episode_fitnesses_first[episode_idx]),
                "probe_repeat_2_episode_fitness": float(episode_fitnesses_second[episode_idx]),
                "probe_repeat_episode_fitness_delta": float(
                    episode_fitnesses_second[episode_idx]
                    - episode_fitnesses_first[episode_idx]
                ),
                "probe_repeat_1_goal_touched": bool(
                    np.asarray(phase_probe_first.episode_results.goal_touched)[episode_idx]
                ),
                "probe_repeat_2_goal_touched": bool(
                    np.asarray(phase_probe_second.episode_results.goal_touched)[episode_idx]
                ),
                "probe_repeat_1_cliff_fell": bool(
                    np.asarray(phase_probe_first.episode_results.cliff_fell)[episode_idx]
                ),
                "probe_repeat_2_cliff_fell": bool(
                    np.asarray(phase_probe_second.episode_results.cliff_fell)[episode_idx]
                ),
                "first_mismatch_category": divergence["category"],
                "first_mismatch_stage": divergence["stage"],
                "first_mismatch_step": divergence["step"],
                "first_mismatch_phase": divergence["phase"],
                "first_mismatch_max_abs_diff": divergence["max_abs_diff"],
            }
        )

    _write_jsonl(rows_path, rows)
    selected_episode_idx = int(np.argmax(episode_deltas)) if episode_count > 0 else 0
    selected_episode_row = (
        rows[selected_episode_idx]
        if 0 <= selected_episode_idx < len(rows)
        else {
            "first_mismatch_category": "none",
            "first_mismatch_stage": None,
            "first_mismatch_step": None,
            "first_mismatch_phase": None,
            "first_mismatch_max_abs_diff": 0.0,
            "probe_repeat_1_episode_fitness": None,
            "probe_repeat_2_episode_fitness": None,
            "probe_repeat_episode_fitness_delta": None,
        }
    )
    selected_episode_step_idx = selected_episode_row["first_mismatch_step"]
    selected_episode_phase_idx = selected_episode_row["first_mismatch_phase"]
    trace_npz_path = audit_dir / (
        f"gen_{int(generation):04d}_internal_phase_probe_candidate_"
        f"{selected_candidate_idx:04d}_episode_{selected_episode_idx:02d}.npz"
    )
    substep_trace_npz_path: Path | None = None
    if episode_count > 0:
        trace_first_env = jax.tree_util.tree_map(
            lambda array: np.asarray(array[selected_episode_idx]),
            phase_probe_first.phase_trace,
        )
        trace_second_env = jax.tree_util.tree_map(
            lambda array: np.asarray(array[selected_episode_idx]),
            phase_probe_second.phase_trace,
        )
        save_payload: dict[str, np.ndarray] = {}
        for prefix, trace in (("first", trace_first_env), ("second", trace_second_env)):
            for field in trace._fields:
                save_payload[f"{prefix}_{field}"] = np.asarray(getattr(trace, field))
        np.savez_compressed(trace_npz_path, **save_payload)
        write_site_diagnostics = _phase_probe_write_site_diagnostics(
            trace_first_env,
            trace_second_env,
            step=selected_episode_step_idx,
            phase=selected_episode_phase_idx,
        )
    else:
        write_site_diagnostics = _phase_probe_write_site_diagnostics(
            None,
            None,
            step=None,
            phase=None,
        )

    selected_episode_substep_summary: dict[str, object] = {
        "selected_episode_substep_probe_initial_step_index": selected_episode_step_idx,
        "selected_episode_substep_probe_initial_phase_index": selected_episode_phase_idx,
        "selected_episode_substep_probe_step_index": selected_episode_step_idx,
        "selected_episode_substep_probe_phase_index": selected_episode_phase_idx,
        "selected_episode_substep_probe_backtrack_count": 0,
        "selected_episode_substep_probe_backtrack_exhausted": False,
        "selected_episode_substep_probe_attempts": [],
        "selected_episode_substep_first_mismatch_category": None,
        "selected_episode_substep_first_mismatch_stage": None,
        "selected_episode_substep_first_mismatch_substep": None,
        "selected_episode_substep_first_mismatch_max_abs_diff": 0.0,
        "selected_episode_substep_trace_npz_path": None,
    }
    if (
        episode_count > 0
        and selected_episode_row["first_mismatch_category"] == "physics_or_collision"
        and selected_episode_step_idx is not None
        and selected_episode_phase_idx is not None
    ):
        selected_episode_trace = jax.tree_util.tree_map(
            lambda array: np.asarray(array[selected_episode_idx]),
            phase_probe_first.phase_trace,
        )
        phase_done_after = np.asarray(selected_episode_trace.phase_done_after)
        phase_count = (
            int(phase_done_after.shape[1]) if phase_done_after.ndim >= 2 else 0
        )
        (
            selected_episode_substep_summary,
            substep_probe_first,
            substep_probe_second,
        ) = _probe_selected_episode_substeps_with_backtrack(
            selected_genotype=selected_genotype,
            env_datas=env_datas,
            alive_mask=alive_mask,
            selected_key=selected_key,
            cfg=cfg,
            selected_episode_index=selected_episode_idx,
            selected_step_index=int(selected_episode_step_idx),
            selected_phase_index=int(selected_episode_phase_idx),
            phase_count=phase_count,
            runtime_resolution=runtime_resolution,
            static_capacity=static_capacity,
        )
        final_substep_probe_step_index = int(
            selected_episode_substep_summary["selected_episode_substep_probe_step_index"]
        )
        final_substep_probe_phase_index = int(
            selected_episode_substep_summary["selected_episode_substep_probe_phase_index"]
        )
        substep_trace_npz_path = audit_dir / (
            f"gen_{int(generation):04d}_internal_substep_probe_candidate_"
            f"{selected_candidate_idx:04d}_episode_{selected_episode_idx:02d}_"
            f"step_{final_substep_probe_step_index:04d}_phase_{final_substep_probe_phase_index:02d}.npz"
        )
        np.savez_compressed(
            substep_trace_npz_path,
            **{
                f"{prefix}_{field}": np.asarray(getattr(trace, field))
                for prefix, trace in (
                    ("first", substep_probe_first.substep_trace),
                    ("second", substep_probe_second.substep_trace),
                )
                for field in trace._fields
            },
        )
        selected_episode_substep_summary.update(
            {
                "selected_episode_substep_trace_npz_path": str(substep_trace_npz_path),
            }
        )

    summary_payload = {
        "generation": int(generation),
        "main_eval_mode": _generation_eval_mode(cfg),
        "selected_candidate_index": int(selected_candidate_idx),
        "selected_candidate_key": selected_key.astype(np.uint32).tolist(),
        "selected_candidate_accepted": bool(
            selected_candidate_idx
            in {int(idx) for idx in np.asarray(decision.accepted_indices, dtype=np.int64)}
        ),
        "selected_candidate_archive_flat_index": accepted_flat_index_by_candidate.get(
            selected_candidate_idx
        ),
        "selected_candidate_main_eval_fitness": float(
            main_fitnesses_arr[selected_candidate_idx]
        ),
        "selected_candidate_full_batch_repeat_1_fitness": float(
            repeat_fitnesses[selected_candidate_idx]
        ),
        "selected_candidate_full_batch_repeat_1_fitness_delta": float(
            repeat_fitnesses[selected_candidate_idx]
            - main_fitnesses_arr[selected_candidate_idx]
        ),
        "selected_candidate_isolated_repeat_1_fitness": float(
            np.asarray(isolated_eval_first.fitnesses, dtype=np.float32).reshape(-1)[0]
        ),
        "selected_candidate_isolated_repeat_2_fitness": float(
            np.asarray(isolated_eval_second.fitnesses, dtype=np.float32).reshape(-1)[0]
        ),
        "selected_candidate_isolated_repeat_fitness_delta": float(
            np.asarray(isolated_eval_second.fitnesses, dtype=np.float32).reshape(-1)[0]
            - np.asarray(isolated_eval_first.fitnesses, dtype=np.float32).reshape(-1)[0]
        ),
        "selected_candidate_phase_probe_repeat_1_fitness": phase_probe_candidate_fitness_first,
        "selected_candidate_phase_probe_repeat_2_fitness": phase_probe_candidate_fitness_second,
        "selected_candidate_phase_probe_repeat_fitness_delta": float(
            phase_probe_candidate_fitness_second - phase_probe_candidate_fitness_first
        ),
        "episode_count": int(episode_count),
        "pre_physics_state_first_mismatch_count": int(
            category_counts.get("pre_physics_state", 0)
        ),
        "controller_first_mismatch_count": int(
            category_counts.get("controller", 0)
        ),
        "physics_or_collision_first_mismatch_count": int(
            category_counts.get("physics_or_collision", 0)
        ),
        "no_internal_mismatch_count": int(category_counts.get("none", 0)),
        "selected_episode_index": int(selected_episode_idx),
        "selected_episode_probe_repeat_1_fitness": selected_episode_row[
            "probe_repeat_1_episode_fitness"
        ],
        "selected_episode_probe_repeat_2_fitness": selected_episode_row[
            "probe_repeat_2_episode_fitness"
        ],
        "selected_episode_probe_repeat_fitness_delta": selected_episode_row[
            "probe_repeat_episode_fitness_delta"
        ],
        "selected_episode_first_mismatch_category": selected_episode_row[
            "first_mismatch_category"
        ],
        "selected_episode_first_mismatch_stage": selected_episode_row[
            "first_mismatch_stage"
        ],
        "selected_episode_first_mismatch_step": selected_episode_row[
            "first_mismatch_step"
        ],
        "selected_episode_first_mismatch_phase": selected_episode_row[
            "first_mismatch_phase"
        ],
        "selected_episode_first_mismatch_max_abs_diff": selected_episode_row[
            "first_mismatch_max_abs_diff"
        ],
        "rows_path": str(rows_path),
        "trace_npz_path": str(trace_npz_path) if episode_count > 0 else None,
    }
    summary_payload.update(write_site_diagnostics)
    summary_payload.update(selected_episode_substep_summary)
    _write_json(summary_path, summary_payload)
    summary_payload["summary_path"] = str(summary_path)
    return summary_payload


def _status_label(status: int) -> str:
    if int(status) == GridArchive.NEW:
        return "new"
    if int(status) == GridArchive.IMPROVE:
        return "improve"
    return "not_added"


def _write_full_generation_batch_probe_artifacts(
    *,
    audit_dir: Path,
    generation: int,
    candidate_batch: jax.Array,
    candidate_keys: jax.Array,
    batch_result,
    fitnesses: np.ndarray,
    descriptors: np.ndarray,
    archive: GridArchive,
    decision: GridArchiveBatchDecision,
    cfg: DictConfig,
    env_datas: list[CliffEnvData] | GrownEvalAssets,
    alive_mask: jnp.ndarray,
    runtime_resolution: RuntimeResolution,
    static_capacity: StaticColliderCapacity | None,
) -> dict[str, object] | None:
    repeat_full_generation_batch = _audit_flag(
        cfg, "repeat_full_generation_batch", False
    ) or _audit_flag(cfg, "probe_internal_phase_divergence", False)
    compare_trace_on_full_generation_batch = _audit_flag(
        cfg, "compare_trace_on_full_generation_batch", False
    )
    compare_floor_stiction_counterfactual = _audit_flag(
        cfg, "compare_floor_stiction_counterfactual", False
    )
    save_full_generation_genotypes = _audit_flag(
        cfg, "save_full_generation_genotypes", False
    )
    if not (
        repeat_full_generation_batch
        or compare_trace_on_full_generation_batch
        or compare_floor_stiction_counterfactual
        or save_full_generation_genotypes
    ):
        return None

    generation_key_mode = _generation_eval_mode(cfg)
    candidate_genotypes = np.asarray(jax.block_until_ready(candidate_batch), dtype=np.float32)
    candidate_key_arr = np.asarray(jax.device_get(candidate_keys), dtype=np.uint32)
    main_fitnesses = np.asarray(fitnesses, dtype=np.float32).reshape(-1)
    main_descriptors = np.asarray(descriptors, dtype=np.float32)
    candidate_count = int(main_fitnesses.shape[0])
    accepted_index_set = {int(idx) for idx in np.asarray(decision.accepted_indices, dtype=np.int64)}
    accepted_flat_index_by_candidate = {
        int(candidate_idx): int(flat_idx)
        for candidate_idx, flat_idx in zip(
            np.asarray(decision.accepted_indices, dtype=np.int64).tolist(),
            np.asarray(decision.accepted_flat_indices, dtype=np.int64).tolist(),
            strict=False,
        )
    }
    full_batch_rows_path = audit_dir / f"gen_{int(generation):04d}_full_batch_rows.jsonl"
    full_batch_summary_path = audit_dir / f"gen_{int(generation):04d}_full_batch_summary.json"
    full_batch_inputs_path = (
        audit_dir / f"gen_{int(generation):04d}_full_batch_inputs.npz"
        if save_full_generation_genotypes
        else None
    )

    if full_batch_inputs_path is not None:
        full_batch_inputs_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            full_batch_inputs_path,
            genotypes=candidate_genotypes,
            candidate_keys=candidate_key_arr,
            main_fitnesses=main_fitnesses,
            main_descriptors=main_descriptors,
            accepted_indices=np.asarray(decision.accepted_indices, dtype=np.int64),
            accepted_flat_indices=np.asarray(decision.accepted_flat_indices, dtype=np.int64),
        )

    repeat_eval_first = None
    repeat_eval_second = None
    if repeat_full_generation_batch and candidate_count > 0:
        reduced_host_result = _generation_eval_reduced_host(cfg)
        repeat_eval_first = _evaluate_probe_candidates(
            genotypes=candidate_genotypes,
            candidate_keys=candidate_key_arr,
            env_datas=env_datas,
            alive_mask=alive_mask,
            cfg=cfg,
            runtime_resolution=runtime_resolution,
            static_capacity=static_capacity,
            reduced_host_result=reduced_host_result,
        )
        repeat_eval_second = _evaluate_probe_candidates(
            genotypes=candidate_genotypes,
            candidate_keys=candidate_key_arr,
            env_datas=env_datas,
            alive_mask=alive_mask,
            cfg=cfg,
            runtime_resolution=runtime_resolution,
            static_capacity=static_capacity,
            reduced_host_result=reduced_host_result,
        )

    floor_counterfactual_eval = None
    floor_counterfactual_mode: str | None = None
    floor_counterfactual_skipped_reason: str | None = None
    if compare_floor_stiction_counterfactual:
        if candidate_count <= 0:
            floor_counterfactual_skipped_reason = "no_candidates"
        elif str(cfg.body.mode).strip().lower() == "grown" or isinstance(
            env_datas, GrownEvalAssets
        ):
            floor_counterfactual_skipped_reason = "body_mode_grown_not_supported"
        elif _floor_stiction_is_disabled(cfg):
            floor_counterfactual_skipped_reason = "main_floor_stiction_already_disabled"
        elif len(env_datas) <= 0:
            floor_counterfactual_skipped_reason = "no_env_datas"
        else:
            floor_counterfactual_mode = "floor_stiction_disabled"
            floor_counterfactual_eval = _evaluate_probe_candidates(
                genotypes=candidate_genotypes,
                candidate_keys=candidate_key_arr,
                env_datas=_clone_env_datas_with_floor_stiction(
                    env_datas,
                    floor_static_friction_const=0.0,
                    floor_stiction_stiffness=0.0,
                    floor_stiction_switch_speed=0.0,
                ),
                alive_mask=alive_mask,
                cfg=cfg,
                runtime_resolution=runtime_resolution,
                static_capacity=static_capacity,
                reduced_host_result=_generation_eval_reduced_host(cfg),
            )

    trace_batch_result = None
    trace_fitnesses = None
    trace_descriptors = None
    if compare_trace_on_full_generation_batch and candidate_count > 0:
        trace_eval = _trace_probe_candidates(
            genotypes=candidate_genotypes,
            candidate_keys=candidate_key_arr,
            env_datas=env_datas,
            alive_mask=alive_mask,
            cfg=cfg,
            runtime_resolution=runtime_resolution,
            static_capacity=static_capacity,
            workload="generation",
        )
        trace_batch_result, trace_fitnesses, trace_descriptors = _materialize_trace_batch_result(
            trace_eval=trace_eval,
            alive_mask=alive_mask,
            cfg=cfg,
        )

    internal_phase_probe_summary = _write_internal_phase_divergence_probe_artifacts(
        audit_dir=audit_dir,
        generation=generation,
        candidate_genotypes=candidate_genotypes,
        candidate_keys=candidate_key_arr,
        main_fitnesses=main_fitnesses,
        decision=decision,
        repeat_eval_first=repeat_eval_first,
        cfg=cfg,
        env_datas=env_datas,
        alive_mask=alive_mask,
        runtime_resolution=runtime_resolution,
        static_capacity=static_capacity,
    )

    rows: list[dict[str, object]] = []
    main_vs_repeat_1_fitness_mismatches = 0
    main_vs_repeat_1_descriptor_mismatches = 0
    repeat_1_vs_repeat_2_fitness_mismatches = 0
    repeat_1_vs_repeat_2_descriptor_mismatches = 0
    main_vs_floor_counterfactual_fitness_mismatches = 0
    main_vs_floor_counterfactual_descriptor_mismatches = 0
    main_vs_trace_fitness_mismatches = 0
    main_vs_trace_descriptor_mismatches = 0
    floor_counterfactual_max_abs_fitness_delta = 0.0
    floor_counterfactual_max_abs_fitness_delta_candidate_index: int | None = None

    for candidate_idx in range(candidate_count):
        main_descriptor = np.asarray(main_descriptors[candidate_idx], dtype=np.float32)
        main_fitness = float(main_fitnesses[candidate_idx])
        accepted = candidate_idx in accepted_index_set
        flat_idx = accepted_flat_index_by_candidate.get(candidate_idx)
        row: dict[str, object] = {
            "generation": int(generation),
            "candidate_index": int(candidate_idx),
            "candidate_key": candidate_key_arr[candidate_idx].astype(np.uint32).tolist(),
            "accepted": bool(accepted),
            "status": _status_label(int(np.asarray(decision.statuses, dtype=np.uint8)[candidate_idx])),
            "descriptor_finite": bool(
                np.asarray(decision.descriptor_finite_mask, dtype=bool)[candidate_idx]
            ),
            "archive_flat_index": None if flat_idx is None else int(flat_idx),
            "main_eval_mode": generation_key_mode,
        }
        row.update(
            _candidate_eval_fields(
                prefix="main_eval",
                batch_result=batch_result,
                fitnesses=main_fitnesses,
                descriptors=main_descriptors,
                idx=candidate_idx,
            )
        )
        if flat_idx is not None:
            archive_descriptor = np.asarray(archive._descriptors[flat_idx], dtype=np.float32)
            archive_fitness = float(archive._fitnesses[flat_idx])
            row.update(
                {
                    "archive_post_fitness": archive_fitness,
                    "archive_post_descriptor": archive_descriptor.tolist(),
                    "archive_fitness_matches_main_eval": _fitness_match(
                        archive_fitness, main_fitness
                    ),
                    "archive_descriptor_matches_main_eval": _descriptor_match(
                        archive_descriptor, main_descriptor
                    ),
                }
            )

        if repeat_eval_first is not None:
            repeat_1_fitnesses = np.asarray(repeat_eval_first.fitnesses, dtype=np.float32).reshape(-1)
            repeat_1_descriptors = np.asarray(repeat_eval_first.descriptors, dtype=np.float32)
            repeat_1_fitness = float(repeat_1_fitnesses[candidate_idx])
            repeat_1_descriptor = np.asarray(
                repeat_1_descriptors[candidate_idx], dtype=np.float32
            )
            row.update(
                _candidate_eval_fields(
                    prefix="repeat_eval_1",
                    batch_result=repeat_eval_first.batch_result,
                    fitnesses=repeat_1_fitnesses,
                    descriptors=repeat_1_descriptors,
                    idx=candidate_idx,
                )
            )
            main_vs_repeat_1_fitness_match = _fitness_match(main_fitness, repeat_1_fitness)
            main_vs_repeat_1_descriptor_match = _descriptor_match(
                main_descriptor, repeat_1_descriptor
            )
            main_vs_repeat_1_fitness_mismatches += int(main_vs_repeat_1_fitness_match is False)
            main_vs_repeat_1_descriptor_mismatches += int(
                main_vs_repeat_1_descriptor_match is False
            )
            row.update(
                {
                    "main_vs_repeat_eval_1_fitness_match": main_vs_repeat_1_fitness_match,
                    "main_vs_repeat_eval_1_descriptor_match": main_vs_repeat_1_descriptor_match,
                    "main_vs_repeat_eval_1_fitness_delta": _fitness_delta(
                        main_fitness, repeat_1_fitness
                    ),
                }
            )

            if repeat_eval_second is not None:
                repeat_2_fitnesses = np.asarray(
                    repeat_eval_second.fitnesses, dtype=np.float32
                ).reshape(-1)
                repeat_2_descriptors = np.asarray(
                    repeat_eval_second.descriptors, dtype=np.float32
                )
                repeat_2_fitness = float(repeat_2_fitnesses[candidate_idx])
                repeat_2_descriptor = np.asarray(
                    repeat_2_descriptors[candidate_idx], dtype=np.float32
                )
                row.update(
                    _candidate_eval_fields(
                        prefix="repeat_eval_2",
                        batch_result=repeat_eval_second.batch_result,
                        fitnesses=repeat_2_fitnesses,
                        descriptors=repeat_2_descriptors,
                        idx=candidate_idx,
                    )
                )
                repeat_1_vs_repeat_2_fitness_match = _fitness_match(
                    repeat_1_fitness, repeat_2_fitness
                )
                repeat_1_vs_repeat_2_descriptor_match = _descriptor_match(
                    repeat_1_descriptor, repeat_2_descriptor
                )
                repeat_1_vs_repeat_2_fitness_mismatches += int(
                    repeat_1_vs_repeat_2_fitness_match is False
                )
                repeat_1_vs_repeat_2_descriptor_mismatches += int(
                    repeat_1_vs_repeat_2_descriptor_match is False
                )
                row.update(
                    {
                        "repeat_eval_1_vs_repeat_eval_2_fitness_match": repeat_1_vs_repeat_2_fitness_match,
                        "repeat_eval_1_vs_repeat_eval_2_descriptor_match": repeat_1_vs_repeat_2_descriptor_match,
                        "repeat_eval_1_vs_repeat_eval_2_fitness_delta": _fitness_delta(
                            repeat_1_fitness, repeat_2_fitness
                        ),
                }
            )

        if floor_counterfactual_eval is not None:
            floor_counterfactual_fitnesses = np.asarray(
                floor_counterfactual_eval.fitnesses, dtype=np.float32
            ).reshape(-1)
            floor_counterfactual_descriptors = np.asarray(
                floor_counterfactual_eval.descriptors, dtype=np.float32
            )
            floor_counterfactual_fitness = float(
                floor_counterfactual_fitnesses[candidate_idx]
            )
            floor_counterfactual_descriptor = np.asarray(
                floor_counterfactual_descriptors[candidate_idx], dtype=np.float32
            )
            row.update(
                _candidate_eval_fields(
                    prefix="floor_counterfactual_eval",
                    batch_result=floor_counterfactual_eval.batch_result,
                    fitnesses=floor_counterfactual_fitnesses,
                    descriptors=floor_counterfactual_descriptors,
                    idx=candidate_idx,
                )
            )
            main_vs_floor_counterfactual_fitness_match = _fitness_match(
                main_fitness, floor_counterfactual_fitness
            )
            main_vs_floor_counterfactual_descriptor_match = _descriptor_match(
                main_descriptor, floor_counterfactual_descriptor
            )
            main_vs_floor_counterfactual_fitness_delta = _fitness_delta(
                main_fitness, floor_counterfactual_fitness
            )
            main_vs_floor_counterfactual_fitness_mismatches += int(
                main_vs_floor_counterfactual_fitness_match is False
            )
            main_vs_floor_counterfactual_descriptor_mismatches += int(
                main_vs_floor_counterfactual_descriptor_match is False
            )
            if main_vs_floor_counterfactual_fitness_delta is not None:
                abs_fitness_delta = abs(main_vs_floor_counterfactual_fitness_delta)
            else:
                abs_fitness_delta = None
            if abs_fitness_delta is not None and (
                floor_counterfactual_max_abs_fitness_delta_candidate_index is None
                or abs_fitness_delta > floor_counterfactual_max_abs_fitness_delta
            ):
                floor_counterfactual_max_abs_fitness_delta = abs_fitness_delta
                floor_counterfactual_max_abs_fitness_delta_candidate_index = int(
                    candidate_idx
                )
            row.update(
                {
                    "main_vs_floor_counterfactual_fitness_match": main_vs_floor_counterfactual_fitness_match,
                    "main_vs_floor_counterfactual_descriptor_match": main_vs_floor_counterfactual_descriptor_match,
                    "main_vs_floor_counterfactual_fitness_delta": main_vs_floor_counterfactual_fitness_delta,
                }
            )

        if trace_batch_result is not None and trace_fitnesses is not None and trace_descriptors is not None:
            trace_fitness_arr = np.asarray(trace_fitnesses, dtype=np.float32).reshape(-1)
            trace_descriptor_arr = np.asarray(trace_descriptors, dtype=np.float32)
            trace_fitness = float(trace_fitness_arr[candidate_idx])
            trace_descriptor = np.asarray(trace_descriptor_arr[candidate_idx], dtype=np.float32)
            row.update(
                _candidate_eval_fields(
                    prefix="trace_eval",
                    batch_result=trace_batch_result,
                    fitnesses=trace_fitness_arr,
                    descriptors=trace_descriptor_arr,
                    idx=candidate_idx,
                )
            )
            main_vs_trace_fitness_match = _fitness_match(main_fitness, trace_fitness)
            main_vs_trace_descriptor_match = _descriptor_match(
                main_descriptor, trace_descriptor
            )
            main_vs_trace_fitness_mismatches += int(main_vs_trace_fitness_match is False)
            main_vs_trace_descriptor_mismatches += int(
                main_vs_trace_descriptor_match is False
            )
            row.update(
                {
                    "main_vs_trace_eval_fitness_match": main_vs_trace_fitness_match,
                    "main_vs_trace_eval_descriptor_match": main_vs_trace_descriptor_match,
                    "main_vs_trace_eval_fitness_delta": _fitness_delta(
                        main_fitness, trace_fitness
                    ),
                }
            )

        rows.append(row)

    _write_jsonl(full_batch_rows_path, rows)
    summary_payload = {
        "generation": int(generation),
        "main_eval_mode": generation_key_mode,
        "candidate_count": int(candidate_count),
        "accepted_count": int(np.asarray(decision.accepted_indices, dtype=np.int64).shape[0]),
        "invalid_descriptor_count": int(decision.invalid_descriptor_count),
        "repeated_count": int(candidate_count if repeat_eval_first is not None else 0),
        "repeat_pair_count": int(candidate_count if repeat_eval_second is not None else 0),
        "floor_counterfactual_mode": floor_counterfactual_mode,
        "floor_counterfactual_skipped_reason": floor_counterfactual_skipped_reason,
        "floor_counterfactual_compared_count": int(
            candidate_count if floor_counterfactual_eval is not None else 0
        ),
        "main_floor_static_friction_const": float(cfg.sim.floor_static_friction_const),
        "main_floor_stiction_stiffness": float(cfg.sim.floor_stiction_stiffness),
        "main_floor_stiction_switch_speed": float(cfg.sim.floor_stiction_switch_speed),
        "main_vs_floor_counterfactual_fitness_mismatch_count": int(
            main_vs_floor_counterfactual_fitness_mismatches
        ),
        "main_vs_floor_counterfactual_descriptor_mismatch_count": int(
            main_vs_floor_counterfactual_descriptor_mismatches
        ),
        "main_vs_floor_counterfactual_max_abs_fitness_delta": float(
            floor_counterfactual_max_abs_fitness_delta
        ),
        "main_vs_floor_counterfactual_max_abs_fitness_delta_candidate_index": floor_counterfactual_max_abs_fitness_delta_candidate_index,
        "trace_compared_count": int(candidate_count if trace_batch_result is not None else 0),
        "main_vs_repeat_eval_1_fitness_mismatch_count": int(
            main_vs_repeat_1_fitness_mismatches
        ),
        "main_vs_repeat_eval_1_descriptor_mismatch_count": int(
            main_vs_repeat_1_descriptor_mismatches
        ),
        "repeat_eval_1_vs_repeat_eval_2_fitness_mismatch_count": int(
            repeat_1_vs_repeat_2_fitness_mismatches
        ),
        "repeat_eval_1_vs_repeat_eval_2_descriptor_mismatch_count": int(
            repeat_1_vs_repeat_2_descriptor_mismatches
        ),
        "main_vs_trace_eval_fitness_mismatch_count": int(
            main_vs_trace_fitness_mismatches
        ),
        "main_vs_trace_eval_descriptor_mismatch_count": int(
            main_vs_trace_descriptor_mismatches
        ),
        "rows_path": str(full_batch_rows_path),
        "inputs_path": None
        if full_batch_inputs_path is None
        else str(full_batch_inputs_path),
    }
    if internal_phase_probe_summary is not None:
        summary_payload.update(
            {
                "internal_phase_probe_summary_path": internal_phase_probe_summary.get(
                    "summary_path"
                ),
                "internal_phase_probe_rows_path": internal_phase_probe_summary.get(
                    "rows_path"
                ),
                "internal_phase_probe_selected_candidate_index": internal_phase_probe_summary.get(
                    "selected_candidate_index"
                ),
                "internal_phase_probe_selected_episode_index": internal_phase_probe_summary.get(
                    "selected_episode_index"
                ),
                "internal_phase_probe_selected_episode_first_mismatch_category": internal_phase_probe_summary.get(
                    "selected_episode_first_mismatch_category"
                ),
                "internal_phase_probe_selected_episode_first_mismatch_stage": internal_phase_probe_summary.get(
                    "selected_episode_first_mismatch_stage"
                ),
                "internal_phase_probe_selected_episode_first_mismatch_step": internal_phase_probe_summary.get(
                    "selected_episode_first_mismatch_step"
                ),
                "internal_phase_probe_selected_episode_first_mismatch_phase": internal_phase_probe_summary.get(
                    "selected_episode_first_mismatch_phase"
                ),
                "internal_phase_probe_selected_episode_phase_write_site_first_divergent_field": internal_phase_probe_summary.get(
                    "selected_episode_phase_write_site_first_divergent_field"
                ),
                "internal_phase_probe_selected_episode_phase_write_site_first_divergent_category": internal_phase_probe_summary.get(
                    "selected_episode_phase_write_site_first_divergent_category"
                ),
                "internal_phase_probe_selected_episode_phase_write_site_last_substep_post_integrator_velocities_repeat_delta": internal_phase_probe_summary.get(
                    "selected_episode_phase_write_site_last_substep_post_integrator_velocities_repeat_delta"
                ),
                "internal_phase_probe_selected_episode_phase_write_site_last_substep_final_velocities_repeat_delta": internal_phase_probe_summary.get(
                    "selected_episode_phase_write_site_last_substep_final_velocities_repeat_delta"
                ),
                "internal_phase_probe_selected_episode_phase_write_site_last_substep_final_velocities_true_repeat_delta": internal_phase_probe_summary.get(
                    "selected_episode_phase_write_site_last_substep_final_velocities_true_repeat_delta"
                ),
                "internal_phase_probe_selected_episode_phase_write_site_post_velocities_repeat_delta": internal_phase_probe_summary.get(
                    "selected_episode_phase_write_site_post_velocities_repeat_delta"
                ),
                "internal_phase_probe_selected_episode_phase_write_site_post_velocities_true_repeat_delta": internal_phase_probe_summary.get(
                    "selected_episode_phase_write_site_post_velocities_true_repeat_delta"
                ),
                "internal_phase_probe_selected_episode_phase_write_site_repeat_1_post_vs_last_substep_final_velocities_delta": internal_phase_probe_summary.get(
                    "selected_episode_phase_write_site_repeat_1_post_vs_last_substep_final_velocities_delta"
                ),
                "internal_phase_probe_selected_episode_phase_write_site_repeat_2_post_vs_last_substep_final_velocities_delta": internal_phase_probe_summary.get(
                    "selected_episode_phase_write_site_repeat_2_post_vs_last_substep_final_velocities_delta"
                ),
                "internal_phase_probe_selected_episode_phase_write_site_repeat_1_post_true_vs_last_substep_final_true_delta": internal_phase_probe_summary.get(
                    "selected_episode_phase_write_site_repeat_1_post_true_vs_last_substep_final_true_delta"
                ),
                "internal_phase_probe_selected_episode_phase_write_site_repeat_2_post_true_vs_last_substep_final_true_delta": internal_phase_probe_summary.get(
                    "selected_episode_phase_write_site_repeat_2_post_true_vs_last_substep_final_true_delta"
                ),
                "internal_phase_probe_selected_episode_substep_probe_initial_step_index": internal_phase_probe_summary.get(
                    "selected_episode_substep_probe_initial_step_index"
                ),
                "internal_phase_probe_selected_episode_substep_probe_initial_phase_index": internal_phase_probe_summary.get(
                    "selected_episode_substep_probe_initial_phase_index"
                ),
                "internal_phase_probe_selected_episode_substep_probe_step_index": internal_phase_probe_summary.get(
                    "selected_episode_substep_probe_step_index"
                ),
                "internal_phase_probe_selected_episode_substep_probe_phase_index": internal_phase_probe_summary.get(
                    "selected_episode_substep_probe_phase_index"
                ),
                "internal_phase_probe_selected_episode_substep_probe_backtrack_count": internal_phase_probe_summary.get(
                    "selected_episode_substep_probe_backtrack_count"
                ),
                "internal_phase_probe_selected_episode_substep_probe_backtrack_exhausted": internal_phase_probe_summary.get(
                    "selected_episode_substep_probe_backtrack_exhausted"
                ),
                "internal_phase_probe_selected_episode_substep_first_mismatch_category": internal_phase_probe_summary.get(
                    "selected_episode_substep_first_mismatch_category"
                ),
                "internal_phase_probe_selected_episode_substep_first_mismatch_stage": internal_phase_probe_summary.get(
                    "selected_episode_substep_first_mismatch_stage"
                ),
                "internal_phase_probe_selected_episode_substep_first_mismatch_substep": internal_phase_probe_summary.get(
                    "selected_episode_substep_first_mismatch_substep"
                ),
                "internal_phase_probe_selected_episode_substep_trace_npz_path": internal_phase_probe_summary.get(
                    "selected_episode_substep_trace_npz_path"
                ),
                "internal_phase_probe_controller_first_mismatch_count": internal_phase_probe_summary.get(
                    "controller_first_mismatch_count"
                ),
                "internal_phase_probe_physics_or_collision_first_mismatch_count": internal_phase_probe_summary.get(
                    "physics_or_collision_first_mismatch_count"
                ),
                "internal_phase_probe_pre_physics_state_first_mismatch_count": internal_phase_probe_summary.get(
                    "pre_physics_state_first_mismatch_count"
                ),
            }
        )
    _write_json(full_batch_summary_path, summary_payload)
    summary_payload["summary_path"] = str(full_batch_summary_path)
    return summary_payload


def _write_generation_probe_artifacts(
    *,
    output_dir: Path,
    generation: int,
    candidate_batch: jax.Array,
    candidate_keys: jax.Array,
    batch_result,
    fitnesses: np.ndarray,
    descriptors: np.ndarray,
    archive: GridArchive,
    decision: GridArchiveBatchDecision,
    cfg: DictConfig,
    env_datas: list[CliffEnvData] | GrownEvalAssets,
    alive_mask: jnp.ndarray,
    runtime_resolution: RuntimeResolution,
    static_capacity: StaticColliderCapacity | None,
) -> Path | None:
    if not _audit_enabled(cfg):
        return None

    capture_archive_winners = _audit_flag(cfg, "capture_archive_winners", True)
    replay_accepted_winners = _audit_flag(cfg, "replay_accepted_winners", False)
    compare_generation_eval_modes = _audit_flag(
        cfg, "compare_generation_eval_modes", False
    )
    full_batch_probe_summary = _write_full_generation_batch_probe_artifacts(
        audit_dir=_audit_output_dir(output_dir, cfg) / "generations",
        generation=generation,
        candidate_batch=candidate_batch,
        candidate_keys=candidate_keys,
        batch_result=batch_result,
        fitnesses=fitnesses,
        descriptors=descriptors,
        archive=archive,
        decision=decision,
        cfg=cfg,
        env_datas=env_datas,
        alive_mask=alive_mask,
        runtime_resolution=runtime_resolution,
        static_capacity=static_capacity,
    )
    needs_accepted_winner_analysis = (
        capture_archive_winners or replay_accepted_winners or compare_generation_eval_modes
    )
    if not needs_accepted_winner_analysis and full_batch_probe_summary is None:
        return None

    accepted_indices = np.asarray(decision.accepted_indices, dtype=np.int64)
    accepted_flat_indices = np.asarray(decision.accepted_flat_indices, dtype=np.int64)
    accepted_fitnesses = np.asarray(decision.accepted_fitnesses, dtype=np.float64)
    accepted_thresholds = np.asarray(decision.accepted_thresholds, dtype=np.float64)
    accepted_pre_fitnesses = np.asarray(decision.accepted_pre_fitnesses, dtype=np.float64)
    accepted_statuses = np.asarray(decision.accepted_statuses, dtype=np.uint8)
    descriptor_names = tuple(str(name) for name in archive.descriptor_names)

    audit_dir = _audit_output_dir(output_dir, cfg) / "generations"
    audit_dir.mkdir(parents=True, exist_ok=True)
    rows_path = audit_dir / f"gen_{int(generation):04d}_accepted_winners.jsonl"
    summary_path = audit_dir / f"gen_{int(generation):04d}_summary.json"

    descriptors_arr = np.asarray(descriptors, dtype=np.float32)
    fitnesses_arr = np.asarray(fitnesses, dtype=np.float32).reshape(-1)
    archive_fitness_mismatches = 0
    archive_descriptor_mismatches = 0
    archive_hash_mismatches = 0
    compare_fitness_mismatches = 0
    compare_descriptor_mismatches = 0
    replay_limit = 0
    rows: list[dict[str, object]] = []

    full_eval = None
    reduced_eval = None
    if accepted_indices.size > 0:
        accepted_index_array = jnp.asarray(accepted_indices, dtype=jnp.int32)
        accepted_genotypes = np.asarray(
            jax.block_until_ready(candidate_batch[accepted_index_array]),
            dtype=np.float32,
        )
        accepted_keys = np.asarray(
            jax.device_get(candidate_keys[accepted_index_array]),
            dtype=np.uint32,
        )

        replay_limit = min(int(accepted_indices.size), _max_generation_probe_winners(cfg))
        replay_indices = accepted_indices[:replay_limit]
        replay_genotypes = accepted_genotypes[:replay_limit]
        replay_keys = accepted_keys[:replay_limit]
        replay_lookup = {
            int(candidate_idx): int(local_idx)
            for local_idx, candidate_idx in enumerate(replay_indices)
        }

        if replay_accepted_winners and replay_limit > 0:
            full_eval = _evaluate_probe_candidates(
                genotypes=replay_genotypes,
                candidate_keys=replay_keys,
                env_datas=env_datas,
                alive_mask=alive_mask,
                cfg=cfg,
                runtime_resolution=runtime_resolution,
                static_capacity=static_capacity,
                reduced_host_result=False,
            )
        if compare_generation_eval_modes and replay_limit > 0:
            if full_eval is None:
                full_eval = _evaluate_probe_candidates(
                    genotypes=replay_genotypes,
                    candidate_keys=replay_keys,
                    env_datas=env_datas,
                    alive_mask=alive_mask,
                    cfg=cfg,
                    runtime_resolution=runtime_resolution,
                    static_capacity=static_capacity,
                    reduced_host_result=False,
                )
            reduced_eval = _evaluate_probe_candidates(
                genotypes=replay_genotypes,
                candidate_keys=replay_keys,
                env_datas=env_datas,
                alive_mask=alive_mask,
                cfg=cfg,
                runtime_resolution=runtime_resolution,
                static_capacity=static_capacity,
                reduced_host_result=True,
            )

        for accepted_rank, candidate_idx in enumerate(accepted_indices.tolist()):
            flat_idx = int(accepted_flat_indices[accepted_rank])
            candidate_descriptor = np.asarray(
                descriptors_arr[candidate_idx], dtype=np.float32
            )
            candidate_genotype = np.asarray(
                accepted_genotypes[accepted_rank], dtype=np.float32
            )
            archive_genotype = np.asarray(archive._genotypes[flat_idx], dtype=np.float32)
            archive_descriptor = np.asarray(
                archive._descriptors[flat_idx], dtype=np.float32
            )
            candidate_hash = _genotype_sha256(candidate_genotype)
            archive_hash = _genotype_sha256(archive_genotype)
            archive_fitness = float(archive._fitnesses[flat_idx])
            archive_descriptor_match = _descriptor_match(
                archive_descriptor, candidate_descriptor
            )
            archive_fitness_match = _fitness_match(
                archive_fitness, float(accepted_fitnesses[accepted_rank])
            )
            archive_hash_match = bool(candidate_hash == archive_hash)
            archive_descriptor_mismatches += int(archive_descriptor_match is False)
            archive_fitness_mismatches += int(archive_fitness_match is False)
            archive_hash_mismatches += int(not archive_hash_match)

            row: dict[str, object] = {
                "generation": int(generation),
                "accepted_rank": int(accepted_rank),
                "candidate_index": int(candidate_idx),
                "candidate_key": accepted_keys[accepted_rank].astype(np.uint32).tolist(),
                "archive_flat_index": flat_idx,
                "status": _status_label(int(accepted_statuses[accepted_rank])),
                "main_eval_mode": _generation_eval_mode(cfg),
                "accepted_fitness": float(accepted_fitnesses[accepted_rank]),
                "accepted_threshold": float(accepted_thresholds[accepted_rank]),
                "accepted_pre_fitness": float(accepted_pre_fitnesses[accepted_rank]),
                "candidate_genotype_sha256": candidate_hash,
                "archive_genotype_sha256": archive_hash,
                "archive_post_fitness": archive_fitness,
                "archive_post_descriptor": archive_descriptor.tolist(),
                "archive_hash_matches_candidate": archive_hash_match,
                "archive_fitness_matches_main_eval": archive_fitness_match,
                "archive_descriptor_matches_main_eval": archive_descriptor_match,
            }
            row.update(
                _candidate_eval_fields(
                    prefix="main_eval",
                    batch_result=batch_result,
                    fitnesses=fitnesses_arr,
                    descriptors=descriptors_arr,
                    idx=candidate_idx,
                )
            )

            replay_pos = replay_lookup.get(int(candidate_idx))
            if replay_pos is not None and full_eval is not None:
                full_descriptor = np.asarray(
                    full_eval.descriptors[replay_pos], dtype=np.float32
                )
                full_fitness = float(
                    np.asarray(full_eval.fitnesses, dtype=np.float32)[replay_pos]
                )
                row.update(
                    _candidate_eval_fields(
                        prefix="replay_full",
                        batch_result=full_eval.batch_result,
                        fitnesses=np.asarray(full_eval.fitnesses, dtype=np.float32),
                        descriptors=np.asarray(full_eval.descriptors, dtype=np.float32),
                        idx=replay_pos,
                    )
                )
                row.update(
                    {
                        "archive_vs_replay_full_fitness_match": _fitness_match(
                            archive_fitness, full_fitness
                        ),
                        "archive_vs_replay_full_descriptor_match": _descriptor_match(
                            archive_descriptor, full_descriptor
                        ),
                    }
                )
            if replay_pos is not None and reduced_eval is not None and full_eval is not None:
                reduced_descriptor = np.asarray(
                    reduced_eval.descriptors[replay_pos], dtype=np.float32
                )
                full_fitness = float(
                    np.asarray(full_eval.fitnesses, dtype=np.float32)[replay_pos]
                )
                reduced_fitness = float(
                    np.asarray(reduced_eval.fitnesses, dtype=np.float32)[replay_pos]
                )
                full_descriptor = np.asarray(
                    full_eval.descriptors[replay_pos], dtype=np.float32
                )
                fitness_match = _fitness_match(full_fitness, reduced_fitness)
                descriptor_match = _descriptor_match(
                    full_descriptor, reduced_descriptor
                )
                compare_fitness_mismatches += int(fitness_match is False)
                compare_descriptor_mismatches += int(descriptor_match is False)
                row.update(
                    _candidate_eval_fields(
                        prefix="replay_reduced",
                        batch_result=reduced_eval.batch_result,
                        fitnesses=np.asarray(reduced_eval.fitnesses, dtype=np.float32),
                        descriptors=np.asarray(reduced_eval.descriptors, dtype=np.float32),
                        idx=replay_pos,
                    )
                )
                row.update(
                    {
                        "replay_full_vs_reduced_fitness_match": fitness_match,
                        "replay_full_vs_reduced_descriptor_match": descriptor_match,
                    }
                )
            rows.append(row)

    if capture_archive_winners:
        _write_jsonl(rows_path, rows)
    summary_payload = {
        "generation": int(generation),
        "main_eval_mode": _generation_eval_mode(cfg),
        "descriptor_names": list(descriptor_names),
        "accepted_count": int(accepted_indices.size),
        "invalid_descriptor_count": int(decision.invalid_descriptor_count),
        "replayed_count": int(replay_limit if full_eval is not None else 0),
        "compared_count": int(replay_limit if reduced_eval is not None else 0),
        "archive_fitness_mismatch_count": int(archive_fitness_mismatches),
        "archive_descriptor_mismatch_count": int(archive_descriptor_mismatches),
        "archive_genotype_hash_mismatch_count": int(archive_hash_mismatches),
        "full_vs_reduced_fitness_mismatch_count": int(compare_fitness_mismatches),
        "full_vs_reduced_descriptor_mismatch_count": int(compare_descriptor_mismatches),
        "rows_path": str(rows_path) if capture_archive_winners else None,
    }
    if full_batch_probe_summary is not None:
        summary_payload.update(
            {
                "full_batch_summary_path": str(full_batch_probe_summary["summary_path"]),
                "full_batch_rows_path": str(full_batch_probe_summary["rows_path"]),
                "full_batch_inputs_path": full_batch_probe_summary["inputs_path"],
                "full_batch_repeated_count": int(full_batch_probe_summary["repeated_count"]),
                "full_batch_repeat_pair_count": int(
                    full_batch_probe_summary["repeat_pair_count"]
                ),
                "full_batch_trace_compared_count": int(
                    full_batch_probe_summary["trace_compared_count"]
                ),
                "full_batch_main_vs_repeat_eval_1_fitness_mismatch_count": int(
                    full_batch_probe_summary[
                        "main_vs_repeat_eval_1_fitness_mismatch_count"
                    ]
                ),
                "full_batch_main_vs_repeat_eval_1_descriptor_mismatch_count": int(
                    full_batch_probe_summary[
                        "main_vs_repeat_eval_1_descriptor_mismatch_count"
                    ]
                ),
                "full_batch_repeat_eval_1_vs_repeat_eval_2_fitness_mismatch_count": int(
                    full_batch_probe_summary[
                        "repeat_eval_1_vs_repeat_eval_2_fitness_mismatch_count"
                    ]
                ),
                "full_batch_repeat_eval_1_vs_repeat_eval_2_descriptor_mismatch_count": int(
                    full_batch_probe_summary[
                        "repeat_eval_1_vs_repeat_eval_2_descriptor_mismatch_count"
                    ]
                ),
                "full_batch_floor_counterfactual_mode": full_batch_probe_summary.get(
                    "floor_counterfactual_mode"
                ),
                "full_batch_floor_counterfactual_skipped_reason": full_batch_probe_summary.get(
                    "floor_counterfactual_skipped_reason"
                ),
                "full_batch_floor_counterfactual_compared_count": int(
                    full_batch_probe_summary[
                        "floor_counterfactual_compared_count"
                    ]
                ),
                "full_batch_main_vs_floor_counterfactual_fitness_mismatch_count": int(
                    full_batch_probe_summary[
                        "main_vs_floor_counterfactual_fitness_mismatch_count"
                    ]
                ),
                "full_batch_main_vs_floor_counterfactual_descriptor_mismatch_count": int(
                    full_batch_probe_summary[
                        "main_vs_floor_counterfactual_descriptor_mismatch_count"
                    ]
                ),
                "full_batch_main_vs_floor_counterfactual_max_abs_fitness_delta": float(
                    full_batch_probe_summary[
                        "main_vs_floor_counterfactual_max_abs_fitness_delta"
                    ]
                ),
                "full_batch_main_vs_floor_counterfactual_max_abs_fitness_delta_candidate_index": full_batch_probe_summary.get(
                    "main_vs_floor_counterfactual_max_abs_fitness_delta_candidate_index"
                ),
                "full_batch_main_vs_trace_eval_fitness_mismatch_count": int(
                    full_batch_probe_summary[
                        "main_vs_trace_eval_fitness_mismatch_count"
                    ]
                ),
                "full_batch_main_vs_trace_eval_descriptor_mismatch_count": int(
                    full_batch_probe_summary[
                        "main_vs_trace_eval_descriptor_mismatch_count"
                    ]
                ),
                "full_batch_internal_phase_probe_summary_path": full_batch_probe_summary.get(
                    "internal_phase_probe_summary_path"
                ),
                "full_batch_internal_phase_probe_rows_path": full_batch_probe_summary.get(
                    "internal_phase_probe_rows_path"
                ),
                "full_batch_internal_phase_probe_selected_candidate_index": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_candidate_index"
                ),
                "full_batch_internal_phase_probe_selected_episode_index": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_episode_index"
                ),
                "full_batch_internal_phase_probe_selected_episode_first_mismatch_category": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_episode_first_mismatch_category"
                ),
                "full_batch_internal_phase_probe_selected_episode_first_mismatch_stage": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_episode_first_mismatch_stage"
                ),
                "full_batch_internal_phase_probe_selected_episode_first_mismatch_step": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_episode_first_mismatch_step"
                ),
                "full_batch_internal_phase_probe_selected_episode_first_mismatch_phase": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_episode_first_mismatch_phase"
                ),
                "full_batch_internal_phase_probe_selected_episode_phase_write_site_first_divergent_field": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_episode_phase_write_site_first_divergent_field"
                ),
                "full_batch_internal_phase_probe_selected_episode_phase_write_site_first_divergent_category": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_episode_phase_write_site_first_divergent_category"
                ),
                "full_batch_internal_phase_probe_selected_episode_phase_write_site_last_substep_post_integrator_velocities_repeat_delta": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_episode_phase_write_site_last_substep_post_integrator_velocities_repeat_delta"
                ),
                "full_batch_internal_phase_probe_selected_episode_phase_write_site_last_substep_final_velocities_repeat_delta": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_episode_phase_write_site_last_substep_final_velocities_repeat_delta"
                ),
                "full_batch_internal_phase_probe_selected_episode_phase_write_site_last_substep_final_velocities_true_repeat_delta": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_episode_phase_write_site_last_substep_final_velocities_true_repeat_delta"
                ),
                "full_batch_internal_phase_probe_selected_episode_phase_write_site_post_velocities_repeat_delta": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_episode_phase_write_site_post_velocities_repeat_delta"
                ),
                "full_batch_internal_phase_probe_selected_episode_phase_write_site_post_velocities_true_repeat_delta": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_episode_phase_write_site_post_velocities_true_repeat_delta"
                ),
                "full_batch_internal_phase_probe_selected_episode_phase_write_site_repeat_1_post_vs_last_substep_final_velocities_delta": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_episode_phase_write_site_repeat_1_post_vs_last_substep_final_velocities_delta"
                ),
                "full_batch_internal_phase_probe_selected_episode_phase_write_site_repeat_2_post_vs_last_substep_final_velocities_delta": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_episode_phase_write_site_repeat_2_post_vs_last_substep_final_velocities_delta"
                ),
                "full_batch_internal_phase_probe_selected_episode_phase_write_site_repeat_1_post_true_vs_last_substep_final_true_delta": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_episode_phase_write_site_repeat_1_post_true_vs_last_substep_final_true_delta"
                ),
                "full_batch_internal_phase_probe_selected_episode_phase_write_site_repeat_2_post_true_vs_last_substep_final_true_delta": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_episode_phase_write_site_repeat_2_post_true_vs_last_substep_final_true_delta"
                ),
                "full_batch_internal_phase_probe_selected_episode_substep_probe_initial_step_index": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_episode_substep_probe_initial_step_index"
                ),
                "full_batch_internal_phase_probe_selected_episode_substep_probe_initial_phase_index": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_episode_substep_probe_initial_phase_index"
                ),
                "full_batch_internal_phase_probe_selected_episode_substep_probe_step_index": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_episode_substep_probe_step_index"
                ),
                "full_batch_internal_phase_probe_selected_episode_substep_probe_phase_index": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_episode_substep_probe_phase_index"
                ),
                "full_batch_internal_phase_probe_selected_episode_substep_probe_backtrack_count": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_episode_substep_probe_backtrack_count"
                ),
                "full_batch_internal_phase_probe_selected_episode_substep_probe_backtrack_exhausted": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_episode_substep_probe_backtrack_exhausted"
                ),
                "full_batch_internal_phase_probe_selected_episode_substep_first_mismatch_category": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_episode_substep_first_mismatch_category"
                ),
                "full_batch_internal_phase_probe_selected_episode_substep_first_mismatch_stage": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_episode_substep_first_mismatch_stage"
                ),
                "full_batch_internal_phase_probe_selected_episode_substep_first_mismatch_substep": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_episode_substep_first_mismatch_substep"
                ),
                "full_batch_internal_phase_probe_selected_episode_substep_trace_npz_path": full_batch_probe_summary.get(
                    "internal_phase_probe_selected_episode_substep_trace_npz_path"
                ),
                "full_batch_internal_phase_probe_controller_first_mismatch_count": full_batch_probe_summary.get(
                    "internal_phase_probe_controller_first_mismatch_count"
                ),
                "full_batch_internal_phase_probe_physics_or_collision_first_mismatch_count": full_batch_probe_summary.get(
                    "internal_phase_probe_physics_or_collision_first_mismatch_count"
                ),
                "full_batch_internal_phase_probe_pre_physics_state_first_mismatch_count": full_batch_probe_summary.get(
                    "internal_phase_probe_pre_physics_state_first_mismatch_count"
                ),
            }
        )
    _write_json(summary_path, summary_payload)
    return summary_path


def _append_elite_specimens(
    *,
    output_dir: Path,
    cfg: DictConfig,
    run_manifest_path: Path,
    run_resolved_config_path: Path,
    generation: int,
    candidate_batch: jax.Array,
    batch_result,
    fitnesses: np.ndarray,
    descriptors: np.ndarray,
    n_episodes: int,
    descriptor_names: tuple[str, ...] | None = None,
    statuses: np.ndarray | None = None,
    archive: GridArchive | None = None,
    archive_decision: GridArchiveBatchDecision | None = None,
) -> list[Path]:
    """Persist any genomes that achieve the historical zero-fall specimen tiers."""
    goal_hit_count = getattr(batch_result, "goal_hit_count", None)
    if goal_hit_count is None:
        return []
    goal_hits = np.asarray(goal_hit_count, dtype=np.float32).reshape(-1)

    fall_counts = None
    if hasattr(batch_result, "fall_count"):
        fall_counts = np.asarray(getattr(batch_result, "fall_count"), dtype=np.float32).reshape(-1)
    elif (
        hasattr(batch_result, "descriptor_columns")
        and getattr(batch_result, "descriptor_columns") is not None
        and "fall_count" in getattr(batch_result, "descriptor_columns")
    ):
        fall_counts = np.asarray(
            getattr(batch_result, "descriptor_columns")["fall_count"],
            dtype=np.float32,
        ).reshape(-1)
    elif hasattr(batch_result, "episode_fell"):
        fall_counts = np.sum(
            np.asarray(getattr(batch_result, "episode_fell"), dtype=np.float32),
            axis=1,
        )
    if fall_counts is None:
        return []

    historical_tiers = [
        _historical_elite_tier(
            goal_hit_count=float(goal_hits[candidate_idx]),
            fall_count=float(fall_counts[candidate_idx]),
            n_episodes=n_episodes,
        )
        for candidate_idx in range(goal_hits.shape[0])
    ]
    qualifying = np.flatnonzero(np.asarray([tier is not None for tier in historical_tiers], dtype=bool))
    if qualifying.size == 0:
        return []

    historical_root = _historical_elite_output_dir(output_dir, cfg)
    historical_root.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    candidate_indices = jnp.asarray(qualifying, dtype=jnp.int32)
    selected_candidates = np.asarray(jax.block_until_ready(candidate_batch[candidate_indices]))
    fitnesses_arr = np.asarray(fitnesses, dtype=np.float32).reshape(-1)
    descriptors_arr = np.asarray(descriptors, dtype=np.float32)
    descriptor_names_arr = (
        np.asarray(tuple(str(name) for name in descriptor_names), dtype=str)
        if descriptor_names is not None
        else None
    )
    status_arr = (
        np.asarray(statuses, dtype=np.uint8).reshape(-1)
        if statuses is not None
        else None
    )
    archive_metadata_by_candidate = (
        _candidate_archive_metadata_lookup(archive, archive_decision)
        if archive is not None and archive_decision is not None
        else {}
    )
    archive_dim_count = int(len(descriptor_names or ()))

    for local_idx, candidate_idx in enumerate(qualifying.tolist()):
        goal_hit_count_int = int(round(float(goal_hits[candidate_idx])))
        fall_count_int = int(round(float(fall_counts[candidate_idx])))
        historical_elite_tier = historical_tiers[candidate_idx]
        if historical_elite_tier is None:
            continue
        archive_metadata = archive_metadata_by_candidate.get(int(candidate_idx), {})
        archive_cell_id = str(archive_metadata.get("archive_cell_id", "")).strip()
        specimen_id = _historical_elite_id(
            generation=generation,
            candidate_index=int(candidate_idx),
            archive_cell_id=archive_cell_id,
            goal_hit_count=goal_hit_count_int,
            fall_count=fall_count_int,
        )
        specimen_dir = specimen_directory(historical_root, specimen_id)
        snapshot_path = specimen_dir / "historical_snapshot.npz"
        if snapshot_path.exists():
            continue
        genotype = np.asarray(selected_candidates[local_idx], dtype=np.float32)
        payload: dict[str, object] = {
            "genotype": genotype,
            "genotype_sha256": np.asarray(_genotype_sha256(genotype), dtype=str),
            "fitness": np.asarray(fitnesses_arr[candidate_idx], dtype=np.float32),
            "descriptor": np.asarray(descriptors_arr[candidate_idx], dtype=np.float32),
            "goal_hit_count": np.asarray(goal_hits[candidate_idx], dtype=np.float32),
            "fall_count": np.asarray(fall_counts[candidate_idx], dtype=np.float32),
            "historical_elite_tier": np.asarray(historical_elite_tier, dtype=str),
            "generation": np.asarray(int(generation), dtype=np.int32),
            "candidate_index": np.asarray(int(candidate_idx), dtype=np.int32),
            "archive_flat_index": np.asarray(
                int(archive_metadata.get("archive_flat_index", -1)),
                dtype=np.int32,
            ),
            "archive_cell_indices": np.asarray(
                archive_metadata.get(
                    "archive_cell_indices",
                    tuple(-1 for _ in range(archive_dim_count)),
                ),
                dtype=np.int32,
            ),
            "archive_cell_id": np.asarray(archive_cell_id, dtype=str),
            "archive_cell_label": np.asarray(
                str(archive_metadata.get("archive_cell_label", "")),
                dtype=str,
            ),
            "specimen_id": np.asarray(specimen_id, dtype=str),
        }
        if descriptor_names_arr is not None:
            payload["descriptor_names"] = descriptor_names_arr
        if status_arr is not None:
            payload["archive_status"] = np.asarray(int(status_arr[candidate_idx]), dtype=np.uint8)
        archive_status = (
            int(status_arr[candidate_idx])
            if status_arr is not None
            else None
        )
        selection_summary = {
            "specimen_id": str(specimen_id),
            "generation": int(generation),
            "candidate_index": int(candidate_idx),
            "fitness": float(fitnesses_arr[candidate_idx]),
            "descriptor": np.asarray(descriptors_arr[candidate_idx], dtype=np.float32).tolist(),
            "descriptor_names": (
                descriptor_names_arr.tolist()
                if descriptor_names_arr is not None
                else []
            ),
            "goal_hit_count": float(goal_hits[candidate_idx]),
            "fall_count": float(fall_counts[candidate_idx]),
            "historical_elite_tier": str(historical_elite_tier),
            "specimen_tier": str(_historical_specimen_tier(historical_elite_tier)),
            "genotype_sha256": str(np.asarray(payload["genotype_sha256"]).item()),
            "archive_status": archive_status,
            "archive_flat_index": int(archive_metadata.get("archive_flat_index", -1)),
            "archive_cell_indices": list(
                int(value)
                for value in archive_metadata.get(
                    "archive_cell_indices",
                    tuple(-1 for _ in range(archive_dim_count)),
                )
            ),
            "archive_cell_id": str(archive_cell_id),
            "archive_cell_label": str(archive_metadata.get("archive_cell_label", "")),
        }
        selection_rule = {
            "kind": "historical_zero_fall_tier",
            "fall_threshold": 0,
            "perfect_hits_threshold": int(n_episodes),
            "near_perfect_hits_threshold": max(int(n_episodes) - 1, 0),
        }
        artifact_paths = materialize_historical_elite_artifacts(
            historical_root=historical_root,
            specimen_id=specimen_id,
            selection_summary=selection_summary,
            selection_rule=selection_rule,
            genotype=genotype,
            cfg=cfg,
            run_manifest_path=run_manifest_path,
            run_resolved_config_path=run_resolved_config_path,
            generation=int(generation),
            snapshot_payload=payload,
        )
        saved_paths.append(Path(artifact_paths["specimen_dir"]))
    return saved_paths


def _relative_path(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    return os.path.relpath(path, root)


def _event_root_key(base_seed: int, generation: int, scope: int) -> jax.Array:
    key = jax.random.PRNGKey(int(base_seed))
    key = jax.random.fold_in(key, int(scope))
    key = jax.random.fold_in(key, int(generation))
    return key


def _specimen_output_dir(
    output_dir: Path,
    cfg: DictConfig,
    *,
    generation: int | None = None,
    final_stage: bool = True,
) -> Path:
    specimen_root = resolve_specimen_root(
        output_dir,
        output_root=str(cfg.specimens.get("output_root", "specimens")),
    )
    if final_stage:
        return specimen_root
    if generation is None:
        raise ValueError("Stage specimen output dir requires a generation.")
    return specimen_root / "stages" / f"gen_{int(generation):04d}"


def _historical_elite_output_dir(output_dir: Path, cfg: DictConfig) -> Path:
    return resolve_historical_elite_root(
        output_dir,
        output_root=str(cfg.specimens.get("output_root", "specimens")),
    )


def _historical_elite_id(
    *,
    generation: int,
    candidate_index: int,
    archive_cell_id: str,
    goal_hit_count: int,
    fall_count: int,
) -> str:
    archive_name_suffix = f"_{archive_cell_id}" if archive_cell_id else ""
    return (
        f"gen_{int(generation):04d}_candidate_{int(candidate_index):04d}"
        f"{archive_name_suffix}"
        f"_hits{int(goal_hit_count)}_falls{int(fall_count)}"
    )


def _archive_cell_indices(
    archive: GridArchive,
    flat_index: int,
) -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in np.unravel_index(int(flat_index), archive.dims)
    )


def _archive_cell_id(cell_indices: tuple[int, ...]) -> str:
    return "cell_" + "-".join(str(int(value)) for value in cell_indices)


def _archive_cell_label(
    archive: GridArchive,
    cell_indices: tuple[int, ...],
) -> str:
    return "|".join(
        f"{name}={int(value)}"
        for name, value in zip(archive.descriptor_names, cell_indices, strict=False)
    )


def _archive_cell_metadata(
    archive: GridArchive,
    flat_index: int,
) -> dict[str, object]:
    cell_indices = _archive_cell_indices(archive, flat_index)
    return {
        "archive_flat_index": int(flat_index),
        "archive_cell_indices": cell_indices,
        "archive_cell_id": _archive_cell_id(cell_indices),
        "archive_cell_label": _archive_cell_label(archive, cell_indices),
    }


def _candidate_archive_metadata_lookup(
    archive: GridArchive,
    decision: GridArchiveBatchDecision,
) -> dict[int, dict[str, object]]:
    return {
        int(candidate_idx): _archive_cell_metadata(archive, int(flat_idx))
        for candidate_idx, flat_idx in zip(
            np.asarray(decision.accepted_indices, dtype=np.int64).tolist(),
            np.asarray(decision.accepted_flat_indices, dtype=np.int64).tolist(),
            strict=False,
        )
    }


def _historical_elite_tier(
    *,
    goal_hit_count: float,
    fall_count: float,
    n_episodes: int,
) -> str | None:
    if float(fall_count) > 0.0:
        return None
    rounded_hits = int(round(float(goal_hit_count)))
    if rounded_hits >= int(n_episodes):
        return _HISTORICAL_PERFECT_ZERO_FALL_TIER
    if rounded_hits >= max(int(n_episodes) - 1, 0):
        return _HISTORICAL_NEAR_PERFECT_ZERO_FALL_TIER
    return None


def _historical_elite_tier_priority(tier: str | None) -> int:
    if str(tier or "") == _HISTORICAL_PERFECT_ZERO_FALL_TIER:
        return 0
    if str(tier or "") == _HISTORICAL_NEAR_PERFECT_ZERO_FALL_TIER:
        return 1
    return 2


def _historical_specimen_tier(tier: str | None) -> str | None:
    normalized = str(tier or "").strip()
    if normalized == _HISTORICAL_PERFECT_ZERO_FALL_TIER:
        return _PERFECT_SPECIMEN_TIER
    if normalized == _HISTORICAL_NEAR_PERFECT_ZERO_FALL_TIER:
        return _IMPERFECT_SPECIMEN_TIER
    return None


def _held_out_specimen_tier(*, perfect_hits: bool, zero_falls: bool) -> str:
    if bool(perfect_hits) and bool(zero_falls):
        return _PERFECT_SPECIMEN_TIER
    return _IMPERFECT_SPECIMEN_TIER


def _candidate_source_priority(source: str) -> int:
    normalized = str(source).strip().lower()
    if normalized == "historical_elite":
        return 0
    if normalized == "archive":
        return 1
    if normalized == "fresh_sample":
        return 2
    return 3


def _genotype_sha256(genotype: np.ndarray) -> str:
    genotype_arr = np.asarray(genotype, dtype=np.float32)
    return hashlib.sha256(genotype_arr.tobytes()).hexdigest()


def _candidate_has_archive_metadata(candidate: dict[str, object]) -> bool:
    return int(candidate.get("archive_flat_index", -1)) >= 0


def _copy_candidate_archive_metadata(target: dict[str, object], source: dict[str, object]) -> None:
    target["archive_flat_index"] = int(source.get("archive_flat_index", -1))
    target["archive_cell_indices"] = tuple(int(value) for value in source.get("archive_cell_indices", ()))
    target["archive_cell_id"] = str(source.get("archive_cell_id", ""))
    target["archive_cell_label"] = str(source.get("archive_cell_label", ""))


def _candidate_sort_key(candidate: dict[str, object]) -> tuple[object, ...]:
    return (
        _candidate_source_priority(str(candidate.get("source", ""))),
        int(candidate.get("source_index", 0)),
    )


def _specimen_row_priority(row: dict[str, object]) -> tuple[object, ...]:
    return (
        _historical_elite_tier_priority(str(row.get("historical_elite_tier") or "")),
        0 if bool(row["perfect_hits"]) else 1,
        0 if bool(row["zero_falls"]) else 1,
        -float(row["mirrored_pair_count"]),
        -float(row["goal_hit_count"]),
        float(row["fall_count"]),
        float(row["completion_time"]),
        float(row["activation_cost"]),
        _candidate_source_priority(str(row["source"])),
        int(row["source_index"]),
    )


def _merge_candidate_records(
    existing: dict[str, object],
    incoming: dict[str, object],
) -> dict[str, object]:
    merged_sources = sorted(
        {
            *(str(source) for source in existing.get("candidate_sources", [existing.get("source", "")])),
            *(str(source) for source in incoming.get("candidate_sources", [incoming.get("source", "")])),
        }
    )
    existing["candidate_sources"] = merged_sources

    incoming_tier = str(incoming.get("historical_elite_tier") or "")
    existing_tier = str(existing.get("historical_elite_tier") or "")
    if _historical_elite_tier_priority(incoming_tier) < _historical_elite_tier_priority(existing_tier):
        existing["historical_elite_tier"] = incoming_tier

    incoming_first_generation = incoming.get("historical_first_generation")
    existing_first_generation = existing.get("historical_first_generation")
    if incoming_first_generation is not None:
        if existing_first_generation is None:
            existing["historical_first_generation"] = int(incoming_first_generation)
        else:
            existing["historical_first_generation"] = min(
                int(existing_first_generation),
                int(incoming_first_generation),
            )

    incoming_last_generation = incoming.get("historical_last_generation")
    existing_last_generation = existing.get("historical_last_generation")
    if incoming_last_generation is not None:
        if existing_last_generation is None:
            existing["historical_last_generation"] = int(incoming_last_generation)
        else:
            existing["historical_last_generation"] = max(
                int(existing_last_generation),
                int(incoming_last_generation),
            )

    existing["historical_snapshot_count"] = int(existing.get("historical_snapshot_count", 0)) + int(
        incoming.get("historical_snapshot_count", 0)
    )
    existing["genotype_sha256"] = str(existing.get("genotype_sha256") or incoming.get("genotype_sha256") or "")

    incoming_source_key = _candidate_sort_key(incoming)
    existing_source_key = _candidate_sort_key(existing)
    if incoming_source_key < existing_source_key:
        existing["source"] = str(incoming["source"])
        existing["source_index"] = int(incoming["source_index"])

    if _candidate_has_archive_metadata(incoming) and not _candidate_has_archive_metadata(existing):
        _copy_candidate_archive_metadata(existing, incoming)

    existing_archive_fitness = float(existing.get("archive_fitness", float("nan")))
    incoming_archive_fitness = float(incoming.get("archive_fitness", float("nan")))
    if np.isfinite(incoming_archive_fitness) and (
        not np.isfinite(existing_archive_fitness)
        or incoming_archive_fitness > existing_archive_fitness
    ):
        existing["archive_fitness"] = incoming_archive_fitness
        existing["archive_descriptor"] = np.asarray(
            incoming.get("archive_descriptor", ()),
            dtype=np.float32,
        )
    elif np.asarray(existing.get("archive_descriptor", ()), dtype=np.float32).size == 0:
        existing["archive_descriptor"] = np.asarray(
            incoming.get("archive_descriptor", ()),
            dtype=np.float32,
        )

    return existing


def _base_pool_candidate(
    *,
    source: str,
    source_index: int,
    genotype: np.ndarray,
    archive_flat_index: int = -1,
    archive_cell_indices: tuple[int, ...] | tuple[()] = (),
    archive_cell_id: str = "",
    archive_cell_label: str = "",
    archive_fitness: float = float("nan"),
    archive_descriptor: np.ndarray | tuple[()] = (),
    genotype_sha256: str | None = None,
    historical_elite_tier: str | None = None,
    historical_first_generation: int | None = None,
    historical_last_generation: int | None = None,
    historical_snapshot_count: int = 0,
) -> dict[str, object]:
    genotype_arr = np.asarray(genotype, dtype=np.float32)
    return {
        "source": str(source),
        "source_index": int(source_index),
        "archive_flat_index": int(archive_flat_index),
        "archive_cell_indices": tuple(int(value) for value in archive_cell_indices),
        "archive_cell_id": str(archive_cell_id),
        "archive_cell_label": str(archive_cell_label),
        "genotype": genotype_arr,
        "archive_fitness": float(archive_fitness),
        "archive_descriptor": np.asarray(archive_descriptor, dtype=np.float32),
        "genotype_sha256": str(genotype_sha256 or _genotype_sha256(genotype_arr)),
        "candidate_sources": [str(source)],
        "historical_elite_tier": None if historical_elite_tier is None else str(historical_elite_tier),
        "historical_first_generation": (
            None if historical_first_generation is None else int(historical_first_generation)
        ),
        "historical_last_generation": (
            None if historical_last_generation is None else int(historical_last_generation)
        ),
        "historical_snapshot_count": int(historical_snapshot_count),
    }


def _load_historical_elite_candidates(
    *,
    output_dir: Path,
    cfg: DictConfig,
    archive: GridArchive,
    n_episodes: int,
) -> tuple[list[dict[str, object]], int]:
    canonical_root = _historical_elite_output_dir(output_dir, cfg)
    canonical_snapshot_paths = sorted(
        specimen_dir / "historical_snapshot.npz"
        for specimen_dir in canonical_root.glob("specimen_*")
        if specimen_dir.is_dir() and (specimen_dir / "historical_snapshot.npz").exists()
    ) if canonical_root.exists() else []
    legacy_root = output_dir / "elite_specimens"
    legacy_snapshot_paths = sorted(legacy_root.glob("*.npz")) if legacy_root.exists() else []
    specimen_paths = canonical_snapshot_paths + legacy_snapshot_paths
    if not specimen_paths:
        return [], 0

    by_hash: dict[str, dict[str, object]] = {}
    for source_index, specimen_path in enumerate(specimen_paths):
        with np.load(specimen_path, allow_pickle=False) as specimen:
            genotype = np.asarray(specimen["genotype"], dtype=np.float32)
            genotype_sha256 = (
                str(np.asarray(specimen["genotype_sha256"]).item())
                if "genotype_sha256" in specimen.files
                else _genotype_sha256(genotype)
            )
            goal_hit_count = float(np.asarray(specimen["goal_hit_count"], dtype=np.float32).reshape(()))
            fall_count = float(np.asarray(specimen["fall_count"], dtype=np.float32).reshape(()))
            historical_elite_tier = (
                str(np.asarray(specimen["historical_elite_tier"]).item())
                if "historical_elite_tier" in specimen.files
                else _historical_elite_tier(
                    goal_hit_count=goal_hit_count,
                    fall_count=fall_count,
                    n_episodes=n_episodes,
                )
            )
            if historical_elite_tier is None:
                continue
            generation = int(np.asarray(specimen["generation"], dtype=np.int32).reshape(()))
            archive_flat_index = (
                int(np.asarray(specimen["archive_flat_index"], dtype=np.int32).reshape(()))
                if "archive_flat_index" in specimen.files
                else -1
            )
            archive_cell_indices = (
                tuple(int(value) for value in np.asarray(specimen["archive_cell_indices"], dtype=np.int32).reshape(-1).tolist())
                if "archive_cell_indices" in specimen.files
                else tuple(-1 for _ in archive.dims)
            )
            archive_cell_id = (
                str(np.asarray(specimen["archive_cell_id"]).item())
                if "archive_cell_id" in specimen.files
                else ""
            )
            archive_cell_label = (
                str(np.asarray(specimen["archive_cell_label"]).item())
                if "archive_cell_label" in specimen.files
                else ""
            )
            candidate = _base_pool_candidate(
                source="historical_elite",
                source_index=source_index,
                genotype=genotype,
                archive_flat_index=archive_flat_index,
                archive_cell_indices=archive_cell_indices,
                archive_cell_id=archive_cell_id,
                archive_cell_label=archive_cell_label,
                archive_fitness=float(np.asarray(specimen["fitness"], dtype=np.float32).reshape(())),
                archive_descriptor=(
                    np.asarray(specimen["descriptor"], dtype=np.float32)
                    if "descriptor" in specimen.files
                    else np.zeros((0,), dtype=np.float32)
                ),
                genotype_sha256=genotype_sha256,
                historical_elite_tier=historical_elite_tier,
                historical_first_generation=generation,
                historical_last_generation=generation,
                historical_snapshot_count=1,
            )
        if genotype_sha256 in by_hash:
            by_hash[genotype_sha256] = _merge_candidate_records(by_hash[genotype_sha256], candidate)
        else:
            by_hash[genotype_sha256] = candidate
    return list(by_hash.values()), len(specimen_paths)


def _dedupe_pool_candidates(
    candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_hash: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        genotype_sha256 = str(candidate.get("genotype_sha256") or _genotype_sha256(candidate["genotype"]))
        if genotype_sha256 in by_hash:
            by_hash[genotype_sha256] = _merge_candidate_records(by_hash[genotype_sha256], candidate)
        else:
            by_hash[genotype_sha256] = dict(candidate)
            by_hash[genotype_sha256]["candidate_sources"] = list(
                candidate.get("candidate_sources", [candidate.get("source", "")])
            )
            by_hash[genotype_sha256]["genotype_sha256"] = genotype_sha256
    return list(by_hash.values())


def _archive_pool_candidates(archive: GridArchive) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for source_index, flat_index in enumerate(sorted(int(index) for index in archive._genotypes)):
        archive_metadata = _archive_cell_metadata(archive, int(flat_index))
        candidates.append(
            _base_pool_candidate(
                source="archive",
                source_index=int(source_index),
                genotype=np.asarray(archive._genotypes[flat_index], dtype=np.float32),
                archive_fitness=float(archive._fitnesses[flat_index]),
                archive_descriptor=np.asarray(archive._descriptors[flat_index], dtype=np.float32),
                **archive_metadata,
            )
        )
    return candidates


def _sample_final_population(
    emitters: list[CMAEmitter],
    *,
    base_key: jax.Array,
    limit: int | None,
) -> list[dict[str, object]]:
    if not emitters:
        return []
    _, emitter_keys = split_key_sequence(base_key, len(emitters))
    populations: list[np.ndarray] = []
    for emitter, emitter_key in zip(emitters, emitter_keys, strict=False):
        original_state = emitter.es_state
        population = emitter.ask(emitter_key)
        emitter.es_state = original_state
        populations.append(np.asarray(jax.block_until_ready(population), dtype=np.float32))
    if not populations:
        return []
    combined = np.concatenate(populations, axis=0)
    if limit is not None:
        combined = combined[: int(limit)]
    return [
        _base_pool_candidate(
            source="fresh_sample",
            source_index=int(index),
            genotype=np.asarray(genotype, dtype=np.float32),
        )
        for index, genotype in enumerate(combined)
    ]


def _specimen_entries_from_schedule(
    schedule,
    *,
    n_terrains: int,
) -> tuple[list[TerrainEntry], list[dict[str, object]]]:
    per_split = int(n_terrains) // 2
    if len(schedule.val) < per_split or len(schedule.test) < per_split:
        raise ValueError(
            "Specimen extraction requires at least "
            f"{per_split} val and {per_split} test terrains; "
            f"got val={len(schedule.val)} test={len(schedule.test)}"
        )
    entries: list[TerrainEntry] = []
    terrain_manifest: list[dict[str, object]] = []
    for split, split_entries in (("val", list(schedule.val[:per_split])), ("test", list(schedule.test[:per_split]))):
        for split_idx, entry in enumerate(split_entries):
            entries.append(entry)
            terrain_manifest.append(
                {
                    "split": split,
                    "split_index": int(split_idx),
                    "cliff_x": int(entry.cliff_x),
                    "gaps": [list(gap) for gap in entry.gaps],
                    "metadata": dict(entry.metadata or {}),
                }
            )
    return entries, terrain_manifest


def _completion_time_bin(
    value: float,
    *,
    dims: tuple[int, ...],
    ranges: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    descriptor_names: tuple[str, ...],
) -> int:
    axis = descriptor_names.index("completion_time")
    lo, hi = ranges[axis]
    normed = (float(value) - float(lo)) / max(float(hi) - float(lo), 1.0e-8)
    normed = min(max(normed, 0.0), 1.0 - 1.0e-8)
    return int(min(int(normed * int(dims[axis])), int(dims[axis]) - 1))


def _specimen_preview_entry_indices(total_entries: int, preview_count: int) -> list[int]:
    if total_entries <= 0 or preview_count <= 0:
        return []
    return sorted({int(index) for index in np.linspace(0, total_entries - 1, num=min(total_entries, preview_count))})


def _specimen_preview_dir(specimen_root: Path, specimen_id: str) -> Path:
    return specimen_directory(specimen_root, specimen_id) / "preview_renders"


def _render_selected_specimen_previews(
    *,
    specimen_root: Path,
    selected_candidates: list[dict[str, object]],
    preview_entry_indices: list[int],
    env_datas: list[CliffEnvData] | GrownEvalAssets,
    alive_mask: jnp.ndarray,
    cfg: DictConfig,
    generation: int,
    runtime_resolution: RuntimeResolution,
) -> dict[str, list[dict[str, object]]]:
    if not selected_candidates or not preview_entry_indices:
        return {}

    batch_output_dir = specimen_root / "_preview_batch"
    shutil.rmtree(batch_output_dir, ignore_errors=True)
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    elites: list[dict[str, object]] = []
    specimen_ids: list[str] = []
    for selected_rank, candidate_info in enumerate(selected_candidates):
        row = dict(candidate_info["selection_row"])
        candidate = dict(candidate_info["candidate"])
        specimen_id = str(candidate_info["specimen_id"])
        specimen_ids.append(specimen_id)
        elites.append(
            {
                "rerank_rank": int(selected_rank),
                "archive_rank": 0,
                "flat_index": (
                    -1
                    if row["archive_flat_index"] is None
                    else int(row["archive_flat_index"])
                ),
                "fitness": float(row["fitness"]),
                "selection_fitness": float(row["fitness"]),
                "descriptor": np.asarray(candidate.get("archive_descriptor", ()), dtype=np.float32),
                "genotype": np.asarray(candidate["genotype"], dtype=np.float32),
                "goal_hit_count": float(row["goal_hit_count"]),
                "fall_count": float(row["fall_count"]),
                "completion_time": float(row["completion_time"]),
            }
        )

    artifacts = render_ranked_elites(
        elites=elites,
        env_datas=env_datas,
        alive_mask=alive_mask,
        cfg=cfg,
        generation=int(generation),
        stage="specimen_preview",
        output_dir=batch_output_dir,
        runtime_resolution=runtime_resolution,
    )

    preview_artifacts_by_specimen: dict[str, list[dict[str, object]]] = {
        specimen_id: [] for specimen_id in specimen_ids
    }
    for artifact in artifacts:
        specimen_id = specimen_ids[int(artifact.rank)]
        preview_dir = _specimen_preview_dir(specimen_root, specimen_id)
        preview_dir.mkdir(parents=True, exist_ok=True)
        target_path = preview_dir / f"preview_{int(artifact.episode_idx):02d}.gif"
        source_path = Path(str(artifact.gif_path))
        if source_path.exists():
            if target_path.exists():
                target_path.unlink()
            shutil.move(str(source_path), str(target_path))
        payload = dict(artifact._asdict())
        payload["gif_path"] = str(target_path)
        preview_artifacts_by_specimen[specimen_id].append(payload)

    for specimen_id, artifact_rows in preview_artifacts_by_specimen.items():
        preview_dir = _specimen_preview_dir(specimen_root, specimen_id)
        ordered_rows = sorted(
            artifact_rows,
            key=lambda row: (int(row["episode_idx"]), str(row["gif_path"])),
        )
        preview_artifacts_by_specimen[specimen_id] = ordered_rows
        _write_json(
            preview_dir / "manifest.json",
            {
                "kind": "sensing_edges_specimen_preview_manifest",
                "specimen_id": str(specimen_id),
                "preview_entry_indices": list(preview_entry_indices),
                "render_count": int(len(ordered_rows)),
                "renders": ordered_rows,
            },
        )

    shutil.rmtree(batch_output_dir, ignore_errors=True)
    return preview_artifacts_by_specimen


def _candidate_pool_row(
    *,
    batch,
    entries: list[TerrainEntry],
    candidate_idx: int,
    candidate: dict[str, object],
    dims: tuple[int, ...],
    ranges: tuple[tuple[float, float], ...],
    descriptor_names: tuple[str, ...],
    n_terrains: int,
    min_mirrored_pair_count: float,
) -> dict[str, object]:
    _test_fitnesses, terrain_rows = _test_rows_for_candidate(
        batch,
        entries,
        candidate_idx=int(candidate_idx),
    )
    terrain_summary = summarize_test_per_terrain(terrain_rows)
    goal_hit_count = float(np.asarray(batch.goal_hit_count)[candidate_idx])
    fall_count = float(np.asarray(batch.fall_count)[candidate_idx])
    mirrored_pair_count = float(np.asarray(batch.mirrored_pair_count)[candidate_idx])
    completion_time = float(np.asarray(batch.time_to_edge)[candidate_idx])
    activation_cost = float(np.asarray(batch.activation_cost)[candidate_idx])
    archive_fitness = float(candidate["archive_fitness"])
    perfect_hits = goal_hit_count >= float(n_terrains)
    zero_falls = fall_count <= 0.0
    return {
        "candidate_index": int(candidate_idx),
        "source": str(candidate["source"]),
        "candidate_sources": [str(source) for source in candidate.get("candidate_sources", [candidate["source"]])],
        "source_index": int(candidate["source_index"]),
        "genotype_sha256": str(candidate.get("genotype_sha256", "")),
        "archive_flat_index": (
            None if int(candidate["archive_flat_index"]) < 0 else int(candidate["archive_flat_index"])
        ),
        "archive_cell_indices": (
            None
            if int(candidate["archive_flat_index"]) < 0
            else [int(value) for value in candidate.get("archive_cell_indices", ())]
        ),
        "archive_cell_id": (
            None
            if int(candidate["archive_flat_index"]) < 0
            else str(candidate.get("archive_cell_id", ""))
        ),
        "archive_cell_label": (
            None
            if int(candidate["archive_flat_index"]) < 0
            else str(candidate.get("archive_cell_label", ""))
        ),
        "archive_fitness": None if np.isnan(archive_fitness) else archive_fitness,
        "fitness": float(np.asarray(batch.fitness)[candidate_idx]),
        "goal_hit_count": goal_hit_count,
        "fall_count": fall_count,
        "historical_elite_tier": candidate.get("historical_elite_tier"),
        "historical_first_generation": candidate.get("historical_first_generation"),
        "historical_last_generation": candidate.get("historical_last_generation"),
        "historical_snapshot_count": int(candidate.get("historical_snapshot_count", 0)),
        "mirrored_pair_count": mirrored_pair_count,
        "completion_time": completion_time,
        "completion_bin": _completion_time_bin(
            completion_time,
            dims=dims,
            ranges=ranges,
            descriptor_names=descriptor_names,
        ),
        "activation_cost": activation_cost,
        "perfect_hits": bool(perfect_hits),
        "zero_falls": bool(zero_falls),
        "specimen_tier": _held_out_specimen_tier(
            perfect_hits=bool(perfect_hits),
            zero_falls=bool(zero_falls),
        ),
        "qualified_mirrored_priority": bool(mirrored_pair_count >= float(min_mirrored_pair_count)),
        "descriptor": np.asarray(candidate.get("archive_descriptor", ()), dtype=np.float32).tolist(),
        "summary": terrain_summary,
    }


def _select_specimen_rows(
    rows: list[dict[str, object]],
    *,
    cfg: DictConfig,
) -> tuple[list[dict[str, object]], int]:
    _ = cfg
    qualifying = [
        row for row in rows
        if _historical_elite_tier_priority(str(row.get("historical_elite_tier") or "")) < 2
    ]
    ordered = sorted(qualifying, key=_specimen_row_priority)
    return ordered, len(qualifying)


def _extract_specimens(
    *,
    archive: GridArchive,
    emitters: list[CMAEmitter],
    cfg: DictConfig,
    runtime_resolution: RuntimeResolution,
    output_dir: Path,
    run_manifest_path: Path,
    run_resolved_config_path: Path,
    metrics_path: Path,
    final_archive_prefix: Path,
    generation: int,
    specimen_root: Path | None = None,
) -> SpecimenExtractionResults | None:
    if not bool(cfg.get("specimens", {}).get("enabled", False)):
        return None

    specimen_dir = Path(specimen_root) if specimen_root is not None else _specimen_output_dir(output_dir, cfg)
    specimen_dir.mkdir(parents=True, exist_ok=True)
    schedule_dir = str(cfg.train.terrain_schedule_dir)
    schedule = load_schedule(
        schedule_dir,
        allow_legacy_hashes=bool(cfg.train.get("allow_legacy_schedule_hashes", False)),
    )
    validate_bilateral_schedule_semantics(schedule)
    n_terrains = int(cfg.specimens.get("n_terrains", 12))
    entries, terrain_manifest = _specimen_entries_from_schedule(schedule, n_terrains=n_terrains)
    phase_cfg = _phase_cfg_from_schedule(
        cfg,
        schedule,
        max_steps=int(cfg.train.max_steps),
    )
    phase_resolution, envs, alive_mask, static_capacity = _build_phase_assets(
        phase_cfg,
        runtime_resolution,
        entries,
    )

    archive_candidates = _archive_pool_candidates(archive)
    historical_candidates, historical_elite_file_count = _load_historical_elite_candidates(
        output_dir=output_dir,
        cfg=cfg,
        archive=archive,
        n_episodes=int(cfg.train.n_episodes),
    )
    fresh_limit = cfg.specimens.get("fresh_population_size")
    fresh_candidates = _sample_final_population(
        emitters,
        base_key=_event_root_key(int(cfg.seed), generation, _SPECIMEN_EVAL_SCOPE),
        limit=(None if fresh_limit is None else int(fresh_limit)),
    )
    pre_dedup_pool_size = len(archive_candidates) + len(historical_candidates) + len(fresh_candidates)
    pool_candidates = _dedupe_pool_candidates(
        archive_candidates + historical_candidates + fresh_candidates
    )
    if not pool_candidates:
        return None

    pool_genotypes = np.stack(
        [np.asarray(candidate["genotype"], dtype=np.float32) for candidate in pool_candidates],
        axis=0,
    )
    _, candidate_keys = split_key_sequence(
        _event_root_key(int(cfg.seed), generation, _SPECIMEN_EVAL_SCOPE + 1),
        len(pool_candidates),
    )
    extraction_eval = evaluate_population_chunked(
        pool_genotypes,
        envs,
        alive_mask,
        candidate_keys,
        phase_cfg,
        workload="test",
        runtime_resolution=phase_resolution,
        static_capacity=static_capacity,
    )

    descriptor_names = tuple(str(name) for name in archive.descriptor_names)
    pool_rows = [
        _candidate_pool_row(
            batch=extraction_eval.batch_result,
            entries=entries,
            candidate_idx=candidate_idx,
            candidate=candidate,
            dims=archive.dims,
            ranges=tuple(tuple(bounds) for bounds in archive.ranges),
            descriptor_names=descriptor_names,
            n_terrains=n_terrains,
            min_mirrored_pair_count=float(
                cfg.specimens.get("priority_min_mirrored_pair_count", 4)
            ),
        )
        for candidate_idx, candidate in enumerate(pool_candidates)
    ]
    selected_rows, qualifying_count = _select_specimen_rows(pool_rows, cfg=cfg)
    perfect_zero_fall_count = sum(
        1
        for row in selected_rows
        if str(row.get("historical_elite_tier") or "") == _HISTORICAL_PERFECT_ZERO_FALL_TIER
    )
    near_perfect_zero_fall_count = sum(
        1
        for row in selected_rows
        if str(row.get("historical_elite_tier") or "") == _HISTORICAL_NEAR_PERFECT_ZERO_FALL_TIER
    )
    perfect_specimen_count = sum(
        1
        for row in selected_rows
        if str(row.get("specimen_tier") or "") == _PERFECT_SPECIMEN_TIER
    )
    imperfect_specimen_count = sum(
        1
        for row in selected_rows
        if str(row.get("specimen_tier") or "") == _IMPERFECT_SPECIMEN_TIER
    )

    preview_indices = _specimen_preview_entry_indices(
        len(entries),
        int(cfg.specimens.get("preview_renders_per_specimen", 6)),
    )
    preview_entries = [entries[index] for index in preview_indices]
    preview_phase_cfg = phase_cfg
    preview_phase_resolution = phase_resolution
    preview_envs = envs
    preview_alive_mask = alive_mask
    if preview_entries and len(preview_entries) != len(entries):
        preview_phase_resolution, preview_envs, preview_alive_mask, _preview_static_capacity = _build_phase_assets(
            preview_phase_cfg,
            runtime_resolution,
            preview_entries,
        )

    selected_specimens: list[dict[str, object]] = []
    selected_genotypes: list[np.ndarray] = []
    selected_preview_candidates: list[dict[str, object]] = []
    selection_rule = {
        "mode": "historical_zero_fall_tiers",
        "coarse_tiers": [_PERFECT_SPECIMEN_TIER, _IMPERFECT_SPECIMEN_TIER],
        "historical_tiers": [
            _HISTORICAL_PERFECT_ZERO_FALL_TIER,
            _HISTORICAL_NEAR_PERFECT_ZERO_FALL_TIER,
        ],
        "tier_definitions": {
            _HISTORICAL_PERFECT_ZERO_FALL_TIER: {
                "goal_hit_count_at_least": int(cfg.train.n_episodes),
                "fall_count_at_most": 0,
            },
            _HISTORICAL_NEAR_PERFECT_ZERO_FALL_TIER: {
                "goal_hit_count_at_least": max(int(cfg.train.n_episodes) - 1, 0),
                "fall_count_at_most": 0,
            },
            _PERFECT_SPECIMEN_TIER: {
                "held_out_perfect_hits": True,
                "held_out_zero_falls": True,
            },
            _IMPERFECT_SPECIMEN_TIER: {
                "held_out_perfect_hits": False,
                "held_out_zero_falls": "any",
            },
        },
        "ranking_within_tier": [
            "held_out_perfect_hits",
            "held_out_zero_falls",
            "held_out_mirrored_pair_count_desc",
            "held_out_goal_hit_count_desc",
            "held_out_fall_count_asc",
            "held_out_completion_time_asc",
            "held_out_activation_cost_asc",
            "source_priority_historical_archive_fresh",
            "source_index_asc",
        ],
        "completion_bin_cap_applied": False,
    }
    for specimen_rank, row in enumerate(selected_rows):
        candidate = pool_candidates[int(row["candidate_index"])]
        if row["archive_flat_index"] is None:
            specimen_suffix = f"{str(row['source'])}_{int(row['source_index']):04d}"
        else:
            specimen_suffix = (
                f"{str(row['archive_cell_id'])}_flat_{int(row['archive_flat_index']):04d}"
            )
        specimen_id = (
            f"bin_{int(row['completion_bin']):02d}_rank_{int(specimen_rank):02d}_"
            f"{specimen_suffix}"
        )
        selected_specimen = dict(row)
        selected_specimen["specimen_id"] = specimen_id
        selected_specimen["selected_rank"] = int(specimen_rank)
        selected_specimen["preview_entry_indices"] = list(preview_indices)
        selected_specimen["preview_artifacts"] = []
        selected_specimens.append(selected_specimen)
        selected_genotypes.append(np.asarray(candidate["genotype"], dtype=np.float32))
        selected_preview_candidates.append(
            {
                "specimen_id": specimen_id,
                "selection_row": dict(row),
                "candidate": candidate,
            }
        )

    preview_artifacts_by_specimen = _render_selected_specimen_previews(
        specimen_root=specimen_dir,
        selected_candidates=selected_preview_candidates,
        preview_entry_indices=preview_indices,
        env_datas=preview_envs,
        alive_mask=preview_alive_mask,
        cfg=phase_cfg,
        generation=int(generation),
        runtime_resolution=preview_phase_resolution,
    )
    for selected_specimen, preview_candidate in zip(selected_specimens, selected_preview_candidates):
        specimen_id = str(preview_candidate["specimen_id"])
        candidate = dict(preview_candidate["candidate"])
        selected_specimen["preview_artifacts"] = list(
            preview_artifacts_by_specimen.get(specimen_id, [])
        )
        selected_specimen["preview_dir"] = str(_specimen_preview_dir(specimen_dir, specimen_id))
        selected_specimen.update(
            materialize_specimen_artifacts(
                specimen_root=specimen_dir,
                specimen_id=specimen_id,
                selected_rank=int(selected_specimen["selected_rank"]),
                selection_summary=selected_specimen,
                selection_rule=selection_rule,
                genotype=np.asarray(candidate["genotype"], dtype=np.float32),
                cfg=cfg,
                run_manifest_path=run_manifest_path,
                run_resolved_config_path=run_resolved_config_path,
                generation=int(generation),
            )
        )

    terrain_manifest_path = specimen_dir / "terrain_manifest.json"
    pool_path = specimen_dir / "pool.npz"
    pool_rows_path = specimen_dir / "pool_rows.jsonl"
    selected_path = specimen_dir / "selected_specimens.json"
    selected_genotypes_path = specimen_dir / "selected_genotypes.npz"
    manifest_path = specimen_dir / "manifest.json"

    _write_json(
        terrain_manifest_path,
        {
            "schedule_dir": schedule_dir,
            "hash_version": int(schedule.hash_version),
            "train_hash": schedule.train_hash,
            "val_hash": schedule.val_hash,
            "test_hash": schedule.test_hash,
            "train_content_hash": schedule.train_content_hash,
            "val_content_hash": schedule.val_content_hash,
            "test_content_hash": schedule.test_content_hash,
            "entries": terrain_manifest,
        },
    )
    np.savez_compressed(
        pool_path,
        genotypes=pool_genotypes.astype(np.float32),
        source=np.asarray([candidate["source"] for candidate in pool_candidates], dtype=str),
        candidate_sources=np.asarray(
            ["|".join(candidate.get("candidate_sources", [candidate["source"]])) for candidate in pool_candidates],
            dtype=str,
        ),
        source_index=np.asarray([candidate["source_index"] for candidate in pool_candidates], dtype=np.int32),
        genotype_sha256=np.asarray([candidate["genotype_sha256"] for candidate in pool_candidates], dtype=str),
        archive_flat_index=np.asarray([candidate["archive_flat_index"] for candidate in pool_candidates], dtype=np.int32),
        archive_cell_indices=np.asarray(
            [
                list(candidate["archive_cell_indices"])
                if candidate["archive_cell_indices"]
                else [-1 for _ in archive.dims]
                for candidate in pool_candidates
            ],
            dtype=np.int32,
        ),
        archive_cell_id=np.asarray(
            [str(candidate.get("archive_cell_id", "")) for candidate in pool_candidates],
            dtype=str,
        ),
        archive_cell_label=np.asarray(
            [str(candidate.get("archive_cell_label", "")) for candidate in pool_candidates],
            dtype=str,
        ),
        archive_fitness=np.asarray([candidate["archive_fitness"] for candidate in pool_candidates], dtype=np.float32),
        historical_elite_tier=np.asarray(
            [str(candidate.get("historical_elite_tier") or "") for candidate in pool_candidates],
            dtype=str,
        ),
        historical_first_generation=np.asarray(
            [
                -1 if candidate.get("historical_first_generation") is None else int(candidate["historical_first_generation"])
                for candidate in pool_candidates
            ],
            dtype=np.int32,
        ),
        historical_last_generation=np.asarray(
            [
                -1 if candidate.get("historical_last_generation") is None else int(candidate["historical_last_generation"])
                for candidate in pool_candidates
            ],
            dtype=np.int32,
        ),
        historical_snapshot_count=np.asarray(
            [int(candidate.get("historical_snapshot_count", 0)) for candidate in pool_candidates],
            dtype=np.int32,
        ),
        specimen_tier=np.asarray(
            [str(row.get("specimen_tier") or "") for row in pool_rows],
            dtype=str,
        ),
        fitness=np.asarray(extraction_eval.fitnesses, dtype=np.float32),
        descriptors=np.asarray(extraction_eval.descriptors, dtype=np.float32),
        goal_hit_count=np.asarray(extraction_eval.batch_result.goal_hit_count, dtype=np.float32),
        fall_count=np.asarray(extraction_eval.batch_result.fall_count, dtype=np.float32),
        mirrored_pair_count=np.asarray(extraction_eval.batch_result.mirrored_pair_count, dtype=np.float32),
        completion_time=np.asarray(extraction_eval.batch_result.time_to_edge, dtype=np.float32),
        activation_cost=np.asarray(extraction_eval.batch_result.activation_cost, dtype=np.float32),
    )
    _write_jsonl(pool_rows_path, pool_rows)
    _write_json(
        selected_path,
        {
            "selection_rule": selection_rule,
            "selected_specimens": selected_specimens,
        },
    )
    np.savez_compressed(
        selected_genotypes_path,
        genotypes=(
            np.stack(selected_genotypes, axis=0).astype(np.float32)
            if selected_genotypes
            else np.zeros((0, pool_genotypes.shape[1]), dtype=np.float32)
        ),
        specimen_id=np.asarray([item["specimen_id"] for item in selected_specimens], dtype=str),
        source=np.asarray([item["source"] for item in selected_specimens], dtype=str),
        candidate_sources=np.asarray(
            ["|".join(item.get("candidate_sources", [item["source"]])) for item in selected_specimens],
            dtype=str,
        ),
        genotype_sha256=np.asarray([str(item.get("genotype_sha256", "")) for item in selected_specimens], dtype=str),
        archive_flat_index=np.asarray(
            [
                -1 if item["archive_flat_index"] is None else int(item["archive_flat_index"])
                for item in selected_specimens
            ],
            dtype=np.int32,
        ),
        archive_cell_indices=np.asarray(
            [
                list(item["archive_cell_indices"])
                if item["archive_cell_indices"] is not None
                else [-1 for _ in archive.dims]
                for item in selected_specimens
            ],
            dtype=np.int32,
        ),
        archive_cell_id=np.asarray(
            [
                "" if item["archive_cell_id"] is None else str(item["archive_cell_id"])
                for item in selected_specimens
            ],
            dtype=str,
        ),
        archive_cell_label=np.asarray(
            [
                "" if item["archive_cell_label"] is None else str(item["archive_cell_label"])
                for item in selected_specimens
            ],
            dtype=str,
        ),
        completion_bin=np.asarray([int(item["completion_bin"]) for item in selected_specimens], dtype=np.int32),
        historical_elite_tier=np.asarray(
            [str(item.get("historical_elite_tier") or "") for item in selected_specimens],
            dtype=str,
        ),
        specimen_tier=np.asarray(
            [str(item.get("specimen_tier") or "") for item in selected_specimens],
            dtype=str,
        ),
        historical_first_generation=np.asarray(
            [
                -1 if item.get("historical_first_generation") is None else int(item["historical_first_generation"])
                for item in selected_specimens
            ],
            dtype=np.int32,
        ),
        historical_last_generation=np.asarray(
            [
                -1 if item.get("historical_last_generation") is None else int(item["historical_last_generation"])
                for item in selected_specimens
            ],
            dtype=np.int32,
        ),
        historical_snapshot_count=np.asarray(
            [int(item.get("historical_snapshot_count", 0)) for item in selected_specimens],
            dtype=np.int32,
        ),
    )
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "kind": "sensing_edges_specimen_manifest",
            "run_metadata": run_metadata(cfg),
            "noise": resolved_noise_settings(cfg),
            "generation": int(generation),
            "archive_elite_count": len(archive_candidates),
            "historical_elite_file_count": int(historical_elite_file_count),
            "historical_elite_candidate_count": len(historical_candidates),
            "fresh_sample_count": len(fresh_candidates),
            "pre_dedup_pool_size": int(pre_dedup_pool_size),
            "pool_size": len(pool_candidates),
            "qualifying_count": int(qualifying_count),
            "selected_count": len(selected_specimens),
            "perfect_zero_fall_count": int(perfect_zero_fall_count),
            "near_perfect_zero_fall_count": int(near_perfect_zero_fall_count),
            "perfect_specimen_count": int(perfect_specimen_count),
            "imperfect_specimen_count": int(imperfect_specimen_count),
            "occupied_completion_bins": len({int(item["completion_bin"]) for item in selected_specimens}),
            "paths": {
                "run_root": str(output_dir),
                "run_manifest": str(run_manifest_path),
                "resolved_config": str(run_resolved_config_path),
                "metrics": str(metrics_path),
                "final_archive_npz": str(final_archive_prefix.with_suffix(".npz")),
                "terrain_manifest": str(terrain_manifest_path),
                "pool_npz": str(pool_path),
                "pool_rows": str(pool_rows_path),
                "selected_specimens": str(selected_path),
                "selected_genotypes": str(selected_genotypes_path),
            },
        },
    )
    return SpecimenExtractionResults(
        output_dir=specimen_dir,
        manifest_path=manifest_path,
        terrain_manifest_path=terrain_manifest_path,
        pool_path=pool_path,
        pool_rows_path=pool_rows_path,
        selected_path=selected_path,
        selected_genotypes_path=selected_genotypes_path,
        pool_size=len(pool_candidates),
        pre_dedup_pool_size=int(pre_dedup_pool_size),
        archive_elite_count=len(archive_candidates),
        historical_elite_file_count=int(historical_elite_file_count),
        historical_elite_candidate_count=len(historical_candidates),
        fresh_sample_count=len(fresh_candidates),
        qualifying_count=int(qualifying_count),
        selected_count=len(selected_specimens),
        perfect_zero_fall_count=int(perfect_zero_fall_count),
        near_perfect_zero_fall_count=int(near_perfect_zero_fall_count),
        perfect_specimen_count=int(perfect_specimen_count),
        imperfect_specimen_count=int(imperfect_specimen_count),
        occupied_completion_bins=len({int(item["completion_bin"]) for item in selected_specimens}),
    )


def _bootstrap_training_run(cfg: DictConfig) -> TrainingBootstrap:
    """Resolve output-dir reuse, logical run identity, and effective resume source."""
    output_dir = Path(cfg.train.get("output_dir", "outputs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / cfg.logging.metrics_file
    manifest_path = output_dir / cfg.logging.manifest_file
    resolved_config_path = output_dir / "resolved_config.yaml"
    resolved_config_str = OmegaConf.to_yaml(cfg)
    invocation_config_sha256 = config_sha256(resolved_config_str)
    logical_hash = logical_config_sha256(cfg)
    resume_mode = str(cfg.train.get("resume_mode", "auto")).strip().lower()
    allow_output_dir_hash_mismatch_resume = bool(
        cfg.train.get("allow_output_dir_hash_mismatch_resume", False)
    )
    stop_after_generations = int(
        cfg.train.get("stop_after_generations") or cfg.train.n_generations
    )
    explicit_resume = cfg.train.get("resume_from", None)
    explicit_resume_prefix = (
        _canonical_checkpoint_prefix(explicit_resume)
        if explicit_resume
        else None
    )

    existing_manifest = load_manifest(manifest_path) if manifest_path.exists() else None
    if existing_manifest is not None:
        existing_hash = _manifest_logical_hash(existing_manifest)
        if not str(existing_manifest.get("config", {}).get("logical_config_sha256", "")).strip():
            if resolved_config_path.exists():
                existing_hash = logical_config_sha256(OmegaConf.load(resolved_config_path))
        if existing_hash != logical_hash:
            if allow_output_dir_hash_mismatch_resume and resume_mode != "branch":
                log.warning(
                    "Reusing existing logical config hash for staged resume | "
                    "output_dir=%s existing=%s current=%s",
                    output_dir,
                    existing_hash,
                    logical_hash,
                )
                logical_hash = existing_hash
            else:
                raise ValueError(
                    "Output directory already contains a manifest for a different logical config. "
                    f"output_dir={output_dir} existing={existing_hash} current={logical_hash}"
                )
        existing_status = str(existing_manifest.get("status", "")).strip().lower()
        if resume_mode == "branch":
            raise ValueError(
                "Branch continuation requires a fresh child output_dir. "
                f"Refusing to reuse {output_dir}."
            )
        if resume_mode == "fresh":
            raise ValueError(
                "Output directory already contains a matching run, and train.resume_mode='fresh' "
                f"refuses automatic reuse: {output_dir}"
            )
        if existing_status == "completed":
            raise ValueError(
                "Output directory already contains a completed matching run. "
                f"Refusing to reuse {output_dir}."
            )

    same_output_dir_manual_resume = (
        explicit_resume_prefix is not None
        and _checkpoint_within_output_dir(explicit_resume_prefix, output_dir)
    )
    if (
        explicit_resume_prefix is not None
        and existing_manifest is not None
        and not same_output_dir_manual_resume
    ):
        raise ValueError(
            "Explicit train.resume_from points outside the current output_dir, but the "
            f"current output_dir already contains an in-progress run: {output_dir}. "
            "Choose a fresh output_dir for a new logical run."
        )

    run_id: str
    wandb_run_id: str
    attempt_count = 1
    lineage = _default_lineage(
        mode=(
            "branch"
            if resume_mode == "branch"
            else "staged"
            if resume_mode == "staged" or stop_after_generations < int(cfg.train.n_generations)
            else "root"
        )
    )
    resume_from: str | None = None
    resumed_generation: int | None = None
    latest_checkpoint_path: str | None = None
    allow_checkpoint_config_mismatch = False

    if existing_manifest is not None and (
        explicit_resume_prefix is None or same_output_dir_manual_resume
    ):
        run_id = str(existing_manifest["run_id"])
        wandb_run_id = str(
            existing_manifest.get("wandb", {}).get("run_id") or run_id
        )
        attempt_count = max(
            int(existing_manifest.get("retry", {}).get("attempt_count", 1)),
            1,
        ) + 1
        lineage = dict(existing_manifest.get("lineage") or lineage)
    else:
        run_id = str(uuid.uuid4())
        wandb_run_id = run_id

    if explicit_resume_prefix is not None:
        resume_from, explicit_meta = _resume_checkpoint_meta(
            explicit_resume_prefix,
            expected_config_sha256=None if resume_mode == "branch" else logical_hash,
        )
        resumed_generation = int(explicit_meta.get("generation", -1))
        latest_checkpoint_path = resume_from
        if resume_mode == "branch":
            allow_checkpoint_config_mismatch = True
            parent_output_dir = _output_dir_from_checkpoint(resume_from)
            parent_cfg = _load_resolved_config(parent_output_dir)
            _validate_branch_config_delta(parent_cfg, cfg)
            parent_manifest = None
            parent_manifest_path = parent_output_dir / cfg.logging.manifest_file
            if parent_manifest_path.exists():
                parent_manifest = load_manifest(parent_manifest_path)
            lineage = {
                "mode": "branch",
                "parent_run_id": (
                    str(parent_manifest.get("run_id"))
                    if parent_manifest is not None
                    else None
                ),
                "parent_checkpoint_path": resume_from,
                "parent_completed_generations": (
                    int(parent_manifest.get("summary", {}).get("completed_generations"))
                    if parent_manifest is not None
                    and parent_manifest.get("summary", {}).get("completed_generations") is not None
                    else int(resumed_generation) + 1
                ),
                "parent_schedule_hashes": (
                    dict(parent_manifest.get("terrain_schedule", {}))
                    if parent_manifest is not None
                    and parent_manifest.get("terrain_schedule") is not None
                    else None
                ),
            }
    elif existing_manifest is not None and resume_mode != "fresh":
        latest_checkpoint = find_latest_checkpoint(
            output_dir / cfg.train.checkpoint_dir,
            config_sha256=logical_hash,
            run_id=run_id,
            require_done=True,
        )
        if latest_checkpoint is not None:
            checkpoint_prefix, checkpoint_meta = latest_checkpoint
            resume_from = str(checkpoint_prefix)
            resumed_generation = int(checkpoint_meta.get("generation", -1))
            latest_checkpoint_path = resume_from
    elif resume_mode == "branch":
        raise ValueError("train.resume_mode=branch requires train.resume_from")

    if (
        resumed_generation is not None
        and stop_after_generations <= int(resumed_generation) + 1
    ):
        raise ValueError(
            "train.stop_after_generations must exceed the last durable generation when resuming. "
            f"stop_after_generations={stop_after_generations} resumed_generation={resumed_generation}"
        )

    return TrainingBootstrap(
        output_dir=output_dir,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        resolved_config_path=resolved_config_path,
        resolved_config_str=resolved_config_str,
        invocation_config_sha256=invocation_config_sha256,
        logical_config_sha256=logical_hash,
        existing_manifest=existing_manifest,
        run_id=run_id,
        wandb_run_id=wandb_run_id,
        attempt_count=attempt_count,
        resume_mode=resume_mode,
        stop_after_generations=stop_after_generations,
        lineage=lineage,
        allow_checkpoint_config_mismatch=allow_checkpoint_config_mismatch,
        resume_from=resume_from,
        resumed_generation=resumed_generation,
        latest_checkpoint_path=latest_checkpoint_path,
    )


def _initialize_training_state(
    cfg: DictConfig,
    genotype_dim: int,
    key: jax.Array,
    rng: np.random.Generator,
    *,
    focus_cfg: FocusConfig,
    focus_target_axes: dict[str, int],
    resume_from: str | None = None,
    expected_config_sha256: str | None = None,
) -> TrainingState:
    """Create fresh search state or restore it from a checkpoint."""
    archive = _create_archive(cfg)
    emitters, key = _create_emitters(cfg, genotype_dim, key)
    focus_bounds = None
    if focus_cfg.enabled:
        focus_bounds = initial_window_bounds(
            focus_cfg,
            focus_target_axes,
            archive.dims,
        )
    state = TrainingState(
        archive=archive,
        emitters=emitters,
        key=key,
        rng=rng,
        focus_bounds=focus_bounds,
    )

    if not resume_from:
        return state

    from .checkpoint import load_checkpoint

    log.info("Resume | checkpoint=%s", resume_from)
    checkpoint = load_checkpoint(
        resume_from,
        genotype_length=genotype_dim,
        **emitter_kwargs_from_cfg(cfg),
    )
    checkpoint_config_sha256 = str(checkpoint["meta"].get("config_sha256", "")).strip()
    if (
        expected_config_sha256
        and checkpoint_config_sha256
        and checkpoint_config_sha256 != expected_config_sha256
    ):
        raise ValueError(
            "Resume checkpoint config hash does not match the current resolved config. "
            f"checkpoint={checkpoint_config_sha256} current={expected_config_sha256}"
        )
    expected_archive = state.archive
    loaded_archive = checkpoint["archive"]
    if (
        loaded_archive.dims != expected_archive.dims
        or tuple(tuple(r) for r in loaded_archive.ranges)
        != tuple(tuple(r) for r in expected_archive.ranges)
        or loaded_archive.descriptor_names != expected_archive.descriptor_names
    ):
        raise ValueError(
            "Checkpoint archive layout does not match the current config. "
            f"expected dims={expected_archive.dims} names={expected_archive.descriptor_names}, "
            f"got dims={loaded_archive.dims} names={loaded_archive.descriptor_names}"
        )
    state.archive = loaded_archive
    state.emitters = checkpoint["emitters"]
    state.key = checkpoint["jax_key"]
    state.rng = checkpoint["rng"]
    state.start_gen = checkpoint["generation"] + 1
    if focus_cfg.enabled:
        state.focus_bounds = checkpoint.get("focus_window_bounds") or initial_window_bounds(
            focus_cfg,
            focus_target_axes,
            state.archive.dims,
        )

    for emitter in state.emitters:
        if emitter.es_state is None:
            state.key, init_key = jax.random.split(state.key)
            emitter.init(init_key)

    log.info(
        "Resume state | generation=%d occupied=%d best=%.2f",
        state.start_gen,
        state.archive.size,
        state.archive.best_fitness,
    )
    return state


def _initialize_artifacts(
    cfg: DictConfig,
    terrain_schedule_info: dict | None,
    runtime_resolution: RuntimeResolution,
    bootstrap: TrainingBootstrap,
) -> TrainingArtifacts:
    """Create output paths, save manifest, and persist the resolved config."""
    output_dir = bootstrap.output_dir
    metrics_path = bootstrap.metrics_path
    manifest_path = bootstrap.manifest_path
    resolved_config_path = bootstrap.resolved_config_path
    resolved_config_str = bootstrap.resolved_config_str
    physics_dt, effective_actuator_convergence = derive_physics_params(
        int(cfg.sim.physics_substeps),
        float(cfg.sim.actuator_convergence),
    )
    run_meta = run_metadata(cfg)
    condition = {
        "body_variant": run_meta["body_variant"],
        "controller_size": controller_size_label(cfg),
        "deformation_condition": deformation_label(cfg),
        "physics_substeps": cfg.sim.physics_substeps,
        "train_variant": run_meta["train_variant"],
        "rerank_variant": run_meta["rerank_variant"],
        "test_variant": run_meta["test_variant"],
        "reference_actuator_convergence": cfg.sim.actuator_convergence,
        "actuator_convergence": effective_actuator_convergence,
        "physics_dt": physics_dt,
        "collision_strategy": str(cfg.collision.strategy),
        "collision_impl": str(cfg.collision.dynamic_impl),
        "collision_tuning_mode": (
            runtime_resolution.resolved_collision.mode
            if runtime_resolution.resolved_collision is not None
            else str(cfg.collision.tuning.mode)
        ),
        "dynamic_max_candidates": (
            runtime_resolution.resolved_collision.dynamic_max_candidates
            if runtime_resolution.resolved_collision is not None
            else int(cfg.collision.dynamic_max_candidates)
        ),
        "dynamic_max_narrow": (
            runtime_resolution.resolved_collision.dynamic_max_narrow
            if runtime_resolution.resolved_collision is not None
            else None
        ),
        "build_max_triples": (
            runtime_resolution.resolved_collision.build_capacity.max_triples
            if runtime_resolution.resolved_collision is not None
            else None
        ),
        "build_max_self_triples": (
            runtime_resolution.resolved_collision.build_capacity.max_self_triples
            if runtime_resolution.resolved_collision is not None
            else None
        ),
        "runtime_build_max_triples": (
            runtime_resolution.resolved_collision.runtime_build_capacity.max_triples
            if runtime_resolution.resolved_collision is not None
            else None
        ),
        "runtime_build_max_self_triples": (
            runtime_resolution.resolved_collision.runtime_build_capacity.max_self_triples
            if runtime_resolution.resolved_collision is not None
            else None
        ),
    }
    manifest = (
        dict(bootstrap.existing_manifest)
        if bootstrap.existing_manifest is not None
        else create_manifest(
            run_id=bootstrap.run_id,
            wandb_run_id=bootstrap.wandb_run_id,
            preset_id=cfg.get("preset_id"),
            resolved_config_str=resolved_config_str,
            resolved_config_path=str(resolved_config_path),
            logical_config_sha256=bootstrap.logical_config_sha256,
            last_invocation_sha256=bootstrap.invocation_config_sha256,
            seed=cfg.seed,
            notes=cfg.get("notes"),
            tags=list(cfg.get("tags", [])),
            condition=condition,
            paper_core_id=runtime_resolution.paper_core_id,
            paper_core_sha256=runtime_resolution.paper_core_sha256,
            preset_changed_paths=list(runtime_resolution.preset_changed_paths),
            runtime_override_keys=list(runtime_resolution.runtime_override_keys),
            gpu_profile=runtime_resolution.gpu_profile,
            resolved_batching=runtime_resolution.batching_manifest(),
            resolved_collision=runtime_resolution.collision_manifest(),
            attempt_count=bootstrap.attempt_count,
            resumed_from_checkpoint=bootstrap.resume_from,
            resumed_generation=bootstrap.resumed_generation,
            latest_checkpoint_path=bootstrap.latest_checkpoint_path,
            lineage=bootstrap.lineage,
        )
    )
    manifest.update(
        {
            "preset_id": cfg.get("preset_id"),
            "notes": cfg.get("notes"),
            "tags": list(cfg.get("tags", [])),
            "condition": condition,
            "paper_core_id": runtime_resolution.paper_core_id,
            "paper_core_sha256": runtime_resolution.paper_core_sha256,
            "preset_changed_paths": list(runtime_resolution.preset_changed_paths),
            "runtime_override_keys": list(runtime_resolution.runtime_override_keys),
            "config": {
                "logical_config_sha256": bootstrap.logical_config_sha256,
                "last_invocation_sha256": bootstrap.invocation_config_sha256,
                "resolved_config_sha256": bootstrap.invocation_config_sha256,
                "resolved_config_path": str(resolved_config_path),
            },
            "runtime": {
                "gpu_profile": runtime_resolution.gpu_profile,
                "resolved_batching": runtime_resolution.batching_manifest(),
                "resolved_collision": runtime_resolution.collision_manifest(),
            },
            "lineage": bootstrap.lineage,
            "wandb": {
                "run_id": bootstrap.wandb_run_id,
            },
            "retry": {
                "attempt_count": int(bootstrap.attempt_count),
                "resumed_from_checkpoint": bootstrap.resume_from,
                "resumed_generation": bootstrap.resumed_generation,
                "latest_checkpoint_path": bootstrap.latest_checkpoint_path,
            },
            "seeds": {"master_seed": int(cfg.seed)},
            "status": "running",
        }
    )
    manifest.pop("completed_at_utc", None)
    if terrain_schedule_info:
        manifest["terrain_schedule"] = terrain_schedule_info
    manifest["training"] = {
        "n_generations": int(cfg.train.n_generations),
        "stop_after_generations": int(bootstrap.stop_after_generations),
        "resume_mode": str(bootstrap.resume_mode),
        "n_episodes": int(cfg.train.n_episodes),
        "max_steps": int(cfg.train.max_steps),
        "n_emitters": int(cfg.qd.n_emitters),
        "emitter_pop_size": int(cfg.qd.emitter_pop_size),
        "total_population": int(total_population(cfg)),
        "checkpoint_interval": int(cfg.train.checkpoint_interval),
    }
    save_manifest(manifest, manifest_path)
    resolved_config_path.write_text(resolved_config_str)
    if bootstrap.resume_from is not None and bootstrap.resumed_generation is not None:
        truncate_metrics_rows(
            metrics_path,
            max_generation=int(bootstrap.resumed_generation),
        )
    elif bootstrap.existing_manifest is not None and bootstrap.attempt_count > 1:
        truncate_metrics_rows(metrics_path, max_generation=-1)

    return TrainingArtifacts(
        output_dir=output_dir,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        resolved_config_path=resolved_config_path,
        resolved_config_str=resolved_config_str,
        invocation_config_sha256=bootstrap.invocation_config_sha256,
        logical_config_sha256=bootstrap.logical_config_sha256,
        manifest=manifest,
    )


def train(
    cfg: DictConfig,
    *,
    runtime_resolution: RuntimeResolution | None = None,
    on_checkpoint_complete: Callable[[], None] | None = None,
) -> GridArchive:
    """Run the full training loop and return the final archive."""
    configure_logging(cfg.logging.level)
    setup_jax_cache(cfg)
    runtime_resolution = runtime_resolution or resolve_runtime_config(cfg)
    bootstrap = _bootstrap_training_run(cfg)

    seed = int(cfg.seed)
    key = jax.random.PRNGKey(seed)
    rng = np.random.default_rng(seed)
    resolved_archive = resolve_qd_archive(cfg)
    focus_cfg, focus_target_axes = _resolve_focus_config(
        cfg,
        descriptor_names=resolved_archive.descriptor_names,
    )

    inputs = _prepare_training_inputs(cfg, rng)
    runtime_resolution = attach_collision_tuning(runtime_resolution, inputs.collision_tuning)
    state = _initialize_training_state(
        cfg,
        inputs.genotype_dim,
        key,
        rng,
        focus_cfg=focus_cfg,
        focus_target_axes=focus_target_axes,
        resume_from=bootstrap.resume_from,
        expected_config_sha256=(
            None
            if bootstrap.allow_checkpoint_config_mismatch
            else bootstrap.logical_config_sha256
        ),
    )
    artifacts = _initialize_artifacts(
        cfg,
        inputs.terrain_schedule_info,
        runtime_resolution,
        bootstrap,
    )
    wandb_active = init_wandb(
        cfg,
        run_id=bootstrap.wandb_run_id,
        resume_mode="allow",
    )

    terrain_source = (
        f"schedule:{inputs.terrain_schedule_info['dir']}"
        if inputs.terrain_schedule_info
        else "procedural"
    )
    target_generations = int(cfg.train.n_generations)
    stop_after_generations = int(bootstrap.stop_after_generations)
    final_stage = stop_after_generations >= target_generations
    render_output_root = artifacts.output_dir / str(cfg.render.output_dir)
    log.info(
        "Run config | core=%s preset=%s seed=%d output=%s generation_range=%d-%d target_gens=%d episodes_per_gen=%d total_pop=%d (%d emitters x %d) terrain=%s gpu=%s",
        runtime_resolution.paper_core_id,
        cfg.get("preset_id") or "config",
        seed,
        artifacts.output_dir,
        state.start_gen,
        stop_after_generations - 1,
        target_generations,
        cfg.train.n_episodes,
        total_population(cfg),
        cfg.qd.n_emitters,
        cfg.qd.emitter_pop_size,
        terrain_source,
        runtime_resolution.gpu_profile,
    )
    log.info(
        "Logical run | run_id=%s wandb_run_id=%s attempt=%d resume_mode=%s resume_from=%s resumed_generation=%s",
        bootstrap.run_id,
        bootstrap.wandb_run_id,
        bootstrap.attempt_count,
        bootstrap.resume_mode,
        bootstrap.resume_from,
        bootstrap.resumed_generation,
    )
    log.info(
        "Controller | size=%s genotype_dim=%d ctrnn_hidden=%d grid_hidden=%d deform=%s mace=%s",
        controller_size_label(cfg),
        inputs.genotype_dim,
        cfg.nca.ctrnn_hidden,
        cfg.nca.ctrnn_grid_hidden,
        deformation_label(cfg),
        cfg.nca.get("mace_mode", "cross_channel"),
    )
    log.info(
        "Collision | strategy=%s impl=%s max_candidates=%d bin=%.3f max_narrow=%d runtime_build_max_triples=%d runtime_build_max_self=%d suite_bound_max_triples=%d suite_bound_max_self=%d mode=%s",
        cfg.collision.strategy,
        cfg.collision.dynamic_impl,
        int(runtime_resolution.resolved_collision.dynamic_max_candidates),
        float(cfg.collision.dynamic_bin_size),
        int(runtime_resolution.resolved_collision.dynamic_max_narrow),
        int(runtime_resolution.resolved_collision.runtime_build_capacity.max_triples),
        int(runtime_resolution.resolved_collision.runtime_build_capacity.max_self_triples),
        int(runtime_resolution.resolved_collision.build_capacity.max_triples),
        int(runtime_resolution.resolved_collision.build_capacity.max_self_triples),
        runtime_resolution.resolved_collision.mode,
    )
    log.info(
        "Collision tuning | suite_pairs=%d suite_triples=%d suite_self=%d synthetic_pairs=%d synthetic_triples=%d synthetic_self=%d runtime_pairs=%d runtime_triples=%d runtime_self=%d",
        int(runtime_resolution.resolved_collision.suite_counts.max_candidate_pairs),
        int(runtime_resolution.resolved_collision.suite_counts.max_triples),
        int(runtime_resolution.resolved_collision.suite_counts.max_self_triples),
        int(runtime_resolution.resolved_collision.synthetic_counts.max_candidate_pairs),
        int(runtime_resolution.resolved_collision.synthetic_counts.max_triples),
        int(runtime_resolution.resolved_collision.synthetic_counts.max_self_triples),
        int(runtime_resolution.resolved_collision.runtime_counts.max_candidate_pairs),
        int(runtime_resolution.resolved_collision.runtime_counts.max_triples),
        int(runtime_resolution.resolved_collision.runtime_counts.max_self_triples),
    )
    generation_batching = runtime_resolution.resolved_batching["generation"]
    log.info(
        "Batching | workload=generation max_chunk=%d preferred=%s exact_fit_small=%d warmup=%s memory_budget_gib=%.2f",
        generation_batching.candidate_chunk_size,
        list(generation_batching.candidate_bucket_sizes),
        generation_batching.exact_fit_small_run_threshold,
        generation_batching.warmup_enabled,
        generation_batching.memory_budget_bytes / (1024**3),
    )

    warmup_total_s = 0.0
    if generation_batching.warmup_enabled:
        warmup_chunk_sizes = planned_warmup_chunk_sizes(total_population(cfg), generation_batching)
        warmup_started = time.monotonic()
        log.info(
            "Warmup | workload=generation terrains=%d planned=%s",
            inputs.generation_terrain_count(0),
            list(warmup_chunk_sizes),
        )
        warmup_timings = warmup_population_evaluator(
            inputs.genotype_dim,
            inputs.envs_for_generation(0),
            inputs.alive_mask,
            cfg,
            workload="generation",
            runtime_resolution=runtime_resolution,
            static_capacity=inputs.generation_static_capacity,
            candidate_count=total_population(cfg),
        )
        warmup_total_s = time.monotonic() - warmup_started
        for bucket_size, elapsed_s in warmup_timings.items():
            log.info(
                "Warmup | workload=generation bucket=%d elapsed_s=%.2f",
                bucket_size,
                elapsed_s,
            )
        log.info("Warmup complete | workload=generation total_s=%.2f", warmup_total_s)
    else:
        log.info("Warmup disabled | workload=generation")

    device_profile_path = os.environ.get("SE_DEVICE_MEMORY_PROFILE_PATH")
    if device_profile_path:
        jax.profiler.save_device_memory_profile(device_profile_path)
        log.info("Saved device memory profile | path=%s", device_profile_path)

    total_start = time.monotonic()
    latest_checkpoint_writer = _AsyncCheckpointWriter()
    archive_checkpoint_writer = _AsyncCheckpointWriter()
    last_completed_generation = state.start_gen - 1
    restart_axis_directions = (
        tuple((focus_target_axes[target.name], target.direction) for target in focus_cfg.targets)
        if focus_cfg.enabled
        else ()
    )

    for gen in range(state.start_gen, stop_after_generations):
        generation_start = time.monotonic()
        generation_envs = inputs.envs_for_generation(gen)
        terrain_count = inputs.generation_terrain_count(gen)
        active_cell_mask = None
        generation_focus_metrics = None
        if focus_cfg.enabled and state.focus_bounds is not None:
            tightened_this_gen = False
            if gen > 0 and gen % focus_cfg.tighten_every_n_gens == 0:
                updated_bounds = maybe_tighten(
                    state.focus_bounds,
                    focus_cfg,
                    focus_target_axes,
                    state.archive.dims,
                    state.archive.occupied_flat_indices,
                )
                tightened_this_gen = updated_bounds != state.focus_bounds
                state.focus_bounds = updated_bounds
            active_cell_mask = build_mask_from_bounds(
                state.focus_bounds,
                focus_target_axes,
                state.archive.dims,
            )
            generation_focus_metrics = focus_metrics(
                state.focus_bounds,
                focus_target_axes,
                state.archive.dims,
                state.archive.occupied_flat_indices,
            )
            generation_focus_metrics["qd_focus_tightened_this_gen"] = float(
                tightened_this_gen
            )

        ask_start = time.monotonic()
        candidates = []
        emitter_sizes = []
        for emitter in state.emitters:
            state.key, ask_key = jax.random.split(state.key)
            population = emitter.ask(ask_key)
            candidates.append(population)
            emitter_sizes.append(int(population.shape[0]))
        candidate_batch = jnp.concatenate(candidates, axis=0)
        ask_s = time.monotonic() - ask_start

        total_candidates = int(candidate_batch.shape[0])
        total_episode_evals = total_candidates * terrain_count
        log.info(
            "Gen %3d start | candidates=%d episodes=%d max_steps=%d chunk=%d",
            gen,
            total_candidates,
            total_episode_evals,
            cfg.train.max_steps,
            generation_batching.candidate_chunk_size,
        )
        state.key, candidate_keys = split_key_sequence(state.key, total_candidates)

        def _log_chunk(progress: ChunkProgress) -> None:
            log.info(
                "Gen %3d chunk %d/%d | candidates=%d/%d episodes=%d/%d best=%.2f mean=%.2f eval_s=%.1f bucket=%d thrpt=%.1f/s",
                gen,
                progress.chunk_index,
                progress.total_chunks,
                progress.evaluated_candidates,
                progress.total_candidates,
                progress.evaluated_candidates * terrain_count,
                total_episode_evals,
                progress.best_fitness,
                progress.mean_fitness,
                progress.elapsed_s,
                progress.bucket_size,
                progress.evaluated_candidates / max(progress.elapsed_s, 1e-6),
            )

        eval_result = evaluate_population_chunked(
            candidate_batch,
            generation_envs,
            inputs.alive_mask,
            candidate_keys,
            cfg,
            workload="generation",
            runtime_resolution=runtime_resolution,
            static_capacity=inputs.generation_static_capacity,
            progress_callback=_log_chunk,
            reduced_host_result=_generation_eval_reduced_host(cfg),
        )
        eval_s = eval_result.stats.elapsed_s
        fitnesses = eval_result.fitnesses
        descriptors = eval_result.descriptors

        archive_start = time.monotonic()
        archive_decision = _update_archive_scored_batch(
            state.archive,
            candidate_batch,
            fitnesses,
            descriptors,
            active_cell_mask=active_cell_mask,
        )
        improvements = archive_decision.improvements
        statuses = archive_decision.statuses
        n_new = int(np.sum(statuses == GridArchive.NEW))
        n_improved = int(np.sum(statuses == GridArchive.IMPROVE))
        archive_s = time.monotonic() - archive_start
        generation_audit_summary_path = _write_generation_probe_artifacts(
            output_dir=artifacts.output_dir,
            generation=gen,
            candidate_batch=candidate_batch,
            candidate_keys=candidate_keys,
            batch_result=eval_result.batch_result,
            fitnesses=fitnesses,
            descriptors=descriptors,
            archive=state.archive,
            decision=archive_decision,
            cfg=cfg,
            env_datas=generation_envs,
            alive_mask=inputs.alive_mask,
            runtime_resolution=runtime_resolution,
            static_capacity=inputs.generation_static_capacity,
        )
        if generation_audit_summary_path is not None:
            log.info(
                "Generation audit written | generation=%d path=%s",
                gen,
                generation_audit_summary_path,
            )
        saved_specimens = _append_elite_specimens(
            output_dir=artifacts.output_dir,
            cfg=cfg,
            run_manifest_path=artifacts.manifest_path,
            run_resolved_config_path=artifacts.resolved_config_path,
            generation=gen,
            candidate_batch=candidate_batch,
            batch_result=eval_result.batch_result,
            fitnesses=fitnesses,
            descriptors=descriptors,
            n_episodes=int(cfg.train.n_episodes),
            descriptor_names=tuple(state.archive.descriptor_names),
            statuses=statuses,
            archive=state.archive,
            archive_decision=archive_decision,
        )
        if saved_specimens:
            log.info(
                "Saved elite specimens | generation=%d count=%d dir=%s",
                gen,
                len(saved_specimens),
                _historical_elite_output_dir(artifacts.output_dir, cfg),
            )
            if on_checkpoint_complete is not None:
                on_checkpoint_complete()

        tell_start = time.monotonic()
        offset = 0
        for emitter, size in zip(state.emitters, emitter_sizes):
            state.key, tell_key = jax.random.split(state.key)
            emitter_population = candidate_batch[offset:offset + size]
            emitter_improvements = improvements[offset:offset + size]
            emitter_statuses = statuses[offset:offset + size]
            emitter.tell(
                tell_key,
                emitter_population,
                emitter_improvements,
                emitter_statuses,
            )

            restart_reason = emitter.restart_reason
            if restart_reason is not None:
                state.key, restart_key, sample_key = jax.random.split(state.key, 3)
                try:
                    if restart_axis_directions:
                        if active_cell_mask is not None:
                            new_mean = jnp.array(
                                state.archive.sample_frontier_elites(
                                    sample_key,
                                    1,
                                    axis_directions=restart_axis_directions,
                                    cell_mask=active_cell_mask,
                                )[0]
                            )
                        else:
                            new_mean = jnp.array(
                                state.archive.sample_frontier_elites(
                                    sample_key,
                                    1,
                                    axis_directions=restart_axis_directions,
                                )[0]
                            )
                    elif active_cell_mask is not None:
                        new_mean = jnp.array(
                            state.archive.sample_elites(
                                sample_key,
                                1,
                                cell_mask=active_cell_mask,
                            )[0]
                        )
                    else:
                        new_mean = jnp.array(state.archive.sample_elites(sample_key, 1)[0])
                except ValueError:
                    if (
                        active_cell_mask is not None
                        and focus_cfg.fallback_to_all_if_active_empty
                    ):
                        try:
                            if restart_axis_directions:
                                new_mean = jnp.array(
                                    state.archive.sample_frontier_elites(
                                        sample_key,
                                        1,
                                        axis_directions=restart_axis_directions,
                                    )[0]
                                )
                            else:
                                new_mean = jnp.array(state.archive.sample_elites(sample_key, 1)[0])
                        except ValueError:
                            new_mean = None
                    else:
                        new_mean = None
                emitter.restart(restart_key, new_mean, reason=restart_reason)
            offset += size
        tell_s = time.monotonic() - tell_start

        checkpoint_path = None
        checkpoint_s = 0.0
        latest_checkpoint_prefix = artifacts.output_dir / cfg.train.checkpoint_dir / "latest"
        checkpoint_start = time.monotonic()
        checkpoint_meta, checkpoint_arrays = build_checkpoint_payload(
            archive=state.archive,
            emitters=state.emitters,
            rng=state.rng,
            jax_key=state.key,
            generation=gen,
            config_sha256=artifacts.logical_config_sha256,
            run_id=artifacts.manifest["run_id"],
            focus_window_bounds=state.focus_bounds,
        )

        def _latest_checkpoint_complete(path: str = str(latest_checkpoint_prefix)) -> None:
            update_manifest_retry_state(
                artifacts.manifest_path,
                latest_checkpoint_path=path,
                wandb_run_id=bootstrap.wandb_run_id,
            )
            if on_checkpoint_complete is not None:
                on_checkpoint_complete()

        latest_checkpoint_writer.start(
            latest_checkpoint_prefix,
            meta=checkpoint_meta,
            arrays=checkpoint_arrays,
            on_complete=_latest_checkpoint_complete,
        )
        checkpoint_s = time.monotonic() - checkpoint_start
        checkpoint_path = str(latest_checkpoint_prefix)
        if (gen + 1) % cfg.train.checkpoint_interval == 0 or gen == stop_after_generations - 1:
            checkpoint_start = time.monotonic()
            checkpoint_prefix = artifacts.output_dir / cfg.train.checkpoint_dir / f"gen_{gen:04d}"
            archive_checkpoint_writer.start(
                checkpoint_prefix,
                meta=checkpoint_meta,
                arrays=checkpoint_arrays,
                on_complete=None,
            )
            checkpoint_s += time.monotonic() - checkpoint_start
            log.info(
                "Checkpoint scheduled | generation=%d latest=%s archive=%s dispatch_s=%.2f",
                gen,
                latest_checkpoint_prefix,
                checkpoint_prefix,
                checkpoint_s,
            )
        else:
            log.info(
                "Resume checkpoint scheduled | generation=%d path=%s dispatch_s=%.2f",
                gen,
                latest_checkpoint_prefix,
                checkpoint_s,
            )

        wandb_payload: dict[str, object] = {}
        render_s = 0.0
        if bool(cfg.render.enabled):
            periodic_cfg = cfg.render.periodic
            periodic_trigger = (
                bool(periodic_cfg.enabled)
                and int(periodic_cfg.every_generations) > 0
                and (gen + 1) % int(periodic_cfg.every_generations) == 0
            )

            if periodic_trigger:
                render_start = time.monotonic()
                periodic_artifacts = _render_periodic_archive_best_effort(
                    state.archive,
                    generation_envs,
                    inputs.alive_mask,
                    cfg,
                    generation=gen,
                    output_root=render_output_root,
                    runtime_resolution=runtime_resolution,
                )
                render_s = time.monotonic() - render_start
                if wandb_active and bool(periodic_cfg.upload_to_wandb):
                    wandb_payload.update(
                        build_wandb_render_payload(
                            periodic_artifacts,
                            key_prefix=f"trials/periodic/gen_{gen:04d}",
                            fps=int(periodic_cfg.get("fps", cfg.render.fps)),
                        )
                    )
                if wandb_active and bool(periodic_cfg.get("include_archive_visuals", True)):
                    wandb_payload["archive/heatmap"] = render_archive_heatmap(
                        state.archive,
                        focus_bounds=state.focus_bounds,
                    )
                    wandb_payload.update(
                        {
                            f"archive/{key}": image
                            for key, image in render_archive_named_heatmaps(
                                state.archive,
                                periodic_cfg.get("archive_heatmaps"),
                                focus_bounds=state.focus_bounds,
                            ).items()
                        }
                    )
                    third_descriptor_heatmap = render_archive_third_descriptor_heatmap(
                        state.archive,
                        focus_bounds=state.focus_bounds,
                    )
                    if third_descriptor_heatmap is not None:
                        wandb_payload["archive/descriptor_3_heatmap"] = third_descriptor_heatmap
                    wandb_payload["archive/fitness_hist"] = render_fitness_histogram(state.archive)

        # Build the scalar payload before wandb.log, then rebuild the final
        # report after logging so persisted elapsed_s includes wandb stall time.
        pre_wandb_report = build_generation_report(
            gen,
            state.archive,
            state.emitters,
            eval_result.batch_result,
            n_new,
            n_improved,
            int(archive_decision.invalid_descriptor_count),
            time.monotonic() - generation_start,
            ask_s=ask_s,
            eval_s=eval_s,
            archive_s=archive_s,
            tell_s=tell_s,
            checkpoint_s=checkpoint_s,
            render_s=render_s,
            episodes_per_candidate=terrain_count,
            chunk_count=eval_result.stats.chunk_count,
            mean_chunk_size=eval_result.stats.mean_chunk_size,
            max_chunk_size=eval_result.stats.max_chunk_size,
            bucket_chunk_counts=eval_result.stats.bucket_counts,
            qd_focus_metrics=generation_focus_metrics,
        )
        if checkpoint_path is not None:
            pre_wandb_report = pre_wandb_report.with_checkpoint_path(checkpoint_path)
        if wandb_active:
            wandb_payload.update(build_generation_metrics(pre_wandb_report))

        wandb_s = 0.0
        if wandb_active and wandb_payload:
            import wandb

            wandb_start = time.monotonic()
            wandb.log(wandb_payload)
            wandb_s = time.monotonic() - wandb_start

        report = build_generation_report(
            gen,
            state.archive,
            state.emitters,
            eval_result.batch_result,
            n_new,
            n_improved,
            int(archive_decision.invalid_descriptor_count),
            time.monotonic() - generation_start,
            ask_s=ask_s,
            eval_s=eval_s,
            archive_s=archive_s,
            tell_s=tell_s,
            checkpoint_s=checkpoint_s,
            render_s=render_s,
            wandb_s=wandb_s,
            episodes_per_candidate=terrain_count,
            chunk_count=eval_result.stats.chunk_count,
            mean_chunk_size=eval_result.stats.mean_chunk_size,
            max_chunk_size=eval_result.stats.max_chunk_size,
            bucket_chunk_counts=eval_result.stats.bucket_counts,
            qd_focus_metrics=generation_focus_metrics,
        )
        if checkpoint_path is not None:
            report = report.with_checkpoint_path(checkpoint_path)

        append_metrics_row(artifacts.metrics_path, report)
        log_generation_report(report, log)
        last_completed_generation = gen

        inputs._generation_env_cache.pop(gen - 1, None)

    latest_checkpoint_writer.join()
    archive_checkpoint_writer.join()

    completed_generations = int(last_completed_generation) + 1
    metric_generation = float(max(0, int(last_completed_generation)))
    artifact_output_dir = artifacts.output_dir
    stage_status = "completed" if final_stage else "paused"
    if not final_stage:
        artifact_output_dir = _stage_artifact_dir(
            artifacts.output_dir,
            generation=int(last_completed_generation),
        )
        artifact_output_dir.mkdir(parents=True, exist_ok=True)
        (artifact_output_dir / "resolved_config.yaml").write_text(artifacts.resolved_config_str)

    final_archive_meta, final_archive_arrays = build_archive_export_payload(
        archive=state.archive,
        config_sha256=artifacts.logical_config_sha256,
        run_id=artifacts.manifest["run_id"],
    )
    write_archive_export_payload(
        artifact_output_dir / "final_archive",
        meta=final_archive_meta,
        arrays=final_archive_arrays,
    )
    final_archive_prefix = artifact_output_dir / "final_archive"
    specimen_results = None
    post_run_results = None
    post_run_error: dict[str, str] | None = None
    specimen_stage_enabled = bool(cfg.get("specimens", {}).get("enabled", False)) and (
        final_stage or bool(cfg.get("specimens", {}).get("run_on_stage_pause", False))
    )
    post_run_cfg = cfg.eval.get("post_run")
    auto_mode = (
        "off"
        if post_run_cfg is None
        else str(post_run_cfg.get("auto_mode", "blocking")).strip().lower()
    )
    post_run_stage_enabled = (
        specimen_stage_enabled
        and post_run_cfg is not None
        and bool(post_run_cfg.get("enabled", False))
        and auto_mode not in {"disabled", "off"}
        and (final_stage or bool(post_run_cfg.get("run_on_stage_pause", False)))
    )
    specimen_output_root = _specimen_output_dir(
        artifacts.output_dir,
        cfg,
        generation=max(0, int(last_completed_generation)),
        final_stage=final_stage,
    )
    if specimen_stage_enabled:
        specimen_results = _extract_specimens(
            archive=state.archive,
            emitters=state.emitters,
            cfg=cfg,
            runtime_resolution=runtime_resolution,
            output_dir=artifacts.output_dir,
            run_manifest_path=artifacts.manifest_path,
            run_resolved_config_path=artifacts.resolved_config_path,
            metrics_path=artifacts.metrics_path,
            final_archive_prefix=final_archive_prefix,
            generation=max(0, int(last_completed_generation)),
            specimen_root=specimen_output_root,
        )
        if specimen_results is not None and post_run_stage_enabled:
            try:
                post_run_results = run_post_run_for_stored_specimen(
                    cfg=cfg,
                    runtime_resolution=runtime_resolution,
                    run_dir=artifacts.output_dir,
                    specimen_root=specimen_results.output_dir,
                    specimen_rank=int(post_run_cfg.get("selected_rank", 0)),
                    raise_on_failure=False,
                    on_progress=on_checkpoint_complete,
                )
                if post_run_results.status != "completed":
                    post_run_error = {
                        "failed_assay": (
                            str(post_run_results.failed_assay)
                            if post_run_results.failed_assay is not None
                            else ""
                        ),
                        "message": str(post_run_results.error_message or ""),
                    }
                    log.error(
                        "Post-run failed after specimens were saved | specimen_id=%s failed_assay=%s error=%s",
                        post_run_results.specimen_id,
                        post_run_results.failed_assay,
                        post_run_results.error_message,
                    )
            except Exception as exc:  # pragma: no cover - defensive safety net
                post_run_error = {
                    "failed_assay": "",
                    "message": str(exc),
                }
                log.exception("Post-run failed before completion, but specimens remain saved")
    if wandb_active and specimen_results is not None:
        import wandb

        wandb.log(
            {
                "generation": metric_generation,
                "specimens/pool_size": float(specimen_results.pool_size),
                "specimens/pre_dedup_pool_size": float(specimen_results.pre_dedup_pool_size),
                "specimens/archive_elite_count": float(specimen_results.archive_elite_count),
                "specimens/historical_elite_file_count": float(specimen_results.historical_elite_file_count),
                "specimens/historical_elite_candidate_count": float(specimen_results.historical_elite_candidate_count),
                "specimens/fresh_sample_count": float(specimen_results.fresh_sample_count),
                "specimens/qualifying_count": float(specimen_results.qualifying_count),
                "specimens/selected_count": float(specimen_results.selected_count),
                "specimens/perfect_zero_fall_count": float(specimen_results.perfect_zero_fall_count),
                "specimens/near_perfect_zero_fall_count": float(specimen_results.near_perfect_zero_fall_count),
                "specimens/perfect_specimen_count": float(specimen_results.perfect_specimen_count),
                "specimens/imperfect_specimen_count": float(specimen_results.imperfect_specimen_count),
                "specimens/occupied_completion_bins": float(specimen_results.occupied_completion_bins),
                "post_run/completed": float(
                    post_run_results is not None and post_run_results.status == "completed"
                ),
                "post_run/failed": float(
                    (post_run_results is not None and post_run_results.status == "failed")
                    or post_run_error is not None
                ),
            }
        )

    total_time = time.monotonic() - total_start
    stage_entry = {
        "status": stage_status,
        "attempt_count": int(bootstrap.attempt_count),
        "resume_mode": str(bootstrap.resume_mode),
        "stop_after_generations": int(stop_after_generations),
        "completed_generations": completed_generations,
        "completed_generation": int(last_completed_generation),
        "latest_checkpoint_path": (
            str(artifacts.output_dir / cfg.train.checkpoint_dir / "latest")
        ),
        "artifact_dir": str(artifact_output_dir),
        "audit_dir": str(_audit_output_dir(artifacts.output_dir, cfg)) if _audit_enabled(cfg) else None,
        "specimen_dir": None if specimen_results is None else str(specimen_results.output_dir),
        "post_run_index": None if post_run_results is None else str(post_run_results.index_path),
        "post_run_status": None if post_run_results is None else str(post_run_results.status),
        "post_run_failed_assay": None if post_run_results is None else post_run_results.failed_assay,
        "post_run_error": (
            None
            if post_run_results is None
            else post_run_results.error_message
        ),
        "post_run_failure": None if post_run_error is None else dict(post_run_error),
        "invocation_config_sha256": artifacts.invocation_config_sha256,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    record_manifest_stage_result(
        artifacts.manifest_path,
        status=stage_status,
        stage_entry=stage_entry,
        completed_generations=completed_generations,
        best_train_fitness=state.archive.best_fitness,
        final_qd_coverage=state.archive.coverage,
        final_qd_score=state.archive.qd_score,
    )

    if wandb_active:
        import wandb

        summary_payload: dict[str, object] = {
            "summary/status": stage_status,
            "summary/best_fitness": state.archive.best_fitness,
            "summary/final_coverage": state.archive.coverage,
            "summary/final_qd_score": state.archive.qd_score,
            "summary/completed_generations": completed_generations,
            "summary/n_generations": cfg.train.n_generations,
            "summary/stop_after_generations": stop_after_generations,
            "summary/total_time_s": round(total_time, 1),
            "summary/warmup_generation_total_s": round(warmup_total_s, 2),
        }
        if specimen_results is not None:
            summary_payload.update(
                {
                    "summary/specimen_pool_size": specimen_results.pool_size,
                    "summary/specimen_pre_dedup_pool_size": specimen_results.pre_dedup_pool_size,
                    "summary/specimen_historical_elite_file_count": specimen_results.historical_elite_file_count,
                    "summary/specimen_historical_elite_candidate_count": specimen_results.historical_elite_candidate_count,
                    "summary/specimen_qualifying_count": specimen_results.qualifying_count,
                    "summary/specimen_selected_count": specimen_results.selected_count,
                    "summary/specimen_perfect_zero_fall_count": specimen_results.perfect_zero_fall_count,
                    "summary/specimen_near_perfect_zero_fall_count": specimen_results.near_perfect_zero_fall_count,
                    "summary/specimen_perfect_count": specimen_results.perfect_specimen_count,
                    "summary/specimen_imperfect_count": specimen_results.imperfect_specimen_count,
                    "summary/specimen_occupied_completion_bins": specimen_results.occupied_completion_bins,
                    "summary/post_run_completed": float(
                        post_run_results is not None and post_run_results.status == "completed"
                    ),
                    "summary/post_run_failed": float(
                        (post_run_results is not None and post_run_results.status == "failed")
                        or post_run_error is not None
                    ),
                    "summary/post_run_status": (
                        None
                        if post_run_results is None
                        else str(post_run_results.status)
                    ),
                }
            )
        wandb.run.summary.update(summary_payload)
        wandb.finish()

    log.info(
        "Training %s | best=%.2f coverage=%.3f qd=%.1f total_time_s=%.0f output=%s",
        stage_status,
        state.archive.best_fitness,
        state.archive.coverage,
        state.archive.qd_score,
        total_time,
        artifacts.output_dir,
    )
    return state.archive
