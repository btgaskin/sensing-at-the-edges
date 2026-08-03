"""Deterministic GIF rendering for training snapshots and final summaries."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from jax_evogym.constants import DT
from jax_evogym.render import RenderConfig, render_episode, save_gif
from jax_evogym.types import RenderInfo, RenderStepOutput

from .batch_eval import (
    GrownEvalAssets,
    evaluate_population_chunked,
    planned_warmup_chunk_sizes,
    trace_population_chunked,
    warmup_population_evaluator,
    warmup_trace_evaluator,
)
from .env import (
    CliffEnvData,
    CliffStaticTemplateBatch,
    StaticColliderCapacity,
    assemble_cliff_env,
    build_cliff_dynamic_core,
    build_cliff_eval_dynamic_core,
    build_grown_dynamic_render_info,
    build_static_collider_template,
    build_cliff_terrain_assets,
    infer_static_collider_capacity,
    infer_static_template_capacity,
    merge_render_info,
    precompute_shared_grid_data,
    stack_static_templates,
)
from .fitness import episode_fitness
from .runtime_config import RuntimeResolution, resolve_runtime_config, resolved_spawn_x
from .terrain import make_robot_body
from .terrain_schedule import TerrainEntry, TerrainSchedule, load_schedule

log = logging.getLogger(__name__)

_PERIODIC_RENDER_SCOPE = 1101
_PERIODIC_TRACE_SCOPE = 1102
_PERIODIC_QD_TRACE_SCOPE = 1103
_FINAL_RERANK_SCOPE = 1201
_FINAL_TRACE_SCOPE = 1202
_POST_TRAIN_RERANK_SCOPE = 1301
_POST_TRAIN_EVAL_SCOPE = 1302
_PERIODIC_TERRAIN_SELECTION_MODES = frozenset({"all", "prefix"})
_PERIODIC_QD_DESCRIPTOR_PRIORITIES = (
    ("goal_hit_count", "max"),
    ("fall_count", "min"),
    ("completion_time", "min"),
)


class RenderArtifact(NamedTuple):
    """One rendered GIF plus deterministic selection metadata."""

    stage: str
    rank: int
    episode_idx: int
    flat_index: int
    archive_fitness: float
    selection_fitness: float
    cliff_x: int
    terrain_idx: int
    gif_path: str
    episode_fitness: float
    goal_hit_count: float
    fall_count: float
    completion_time: float
    peak_front_x: float
    safe_front_x: float
    probe_extent: float
    turn_count: float
    stable_turn_count: float
    turn_motion_threshold: float


class RenderableEpisode(NamedTuple):
    """One episode trace plus static render metadata ready for host-side encoding."""

    stage: str
    rank: int
    episode_idx: int
    flat_index: int
    archive_fitness: float
    selection_fitness: float
    cliff_x: int
    terrain_idx: int
    render_info: RenderInfo
    episode_result: object
    trace: RenderStepOutput
    output_path: Path
    goal_hit_count: float
    fall_count: float
    completion_time: float


def render_episode_gif(
    render_outputs: RenderStepOutput,
    render_info: RenderInfo,
    output_path: str | Path,
    *,
    config: RenderConfig | None = None,
    fps: int = 24,
    width: int = 480,
    height: int = 240,
    target_frames: int | None = None,
    real_time: bool = True,
    frame_skip: int = 1,
    sim_time_per_step: float | None = None,
) -> str:
    """Render a position trace to an animated GIF."""
    render_config = config or RenderConfig(
        enabled=True,
        width=width,
        height=height,
        fps=fps,
        real_time=real_time,
        frame_skip=frame_skip,
        dynamic_actuator_colors=True,
        target_frames=target_frames,
        sim_time_per_step=sim_time_per_step,
    )
    frames = render_episode(render_outputs, render_info, render_config)
    return save_gif(frames, str(output_path), render_config)


def select_render_elites(
    archive,
    *,
    max_cells: int,
    max_individuals: int,
) -> list[dict[str, np.ndarray | float | int]]:
    """Select elites deterministically from the highest-fitness archive cells."""
    if max_cells <= 0 or max_individuals <= 0 or archive.size == 0:
        return []
    cell_limited = archive.top_elites_by_fitness(min(int(max_cells), archive.size))
    return [dict(elite) for elite in cell_limited[: min(len(cell_limited), int(max_individuals))]]


def select_projected_render_elites(
    archive,
    *,
    descriptor_targets: dict[str, str],
    max_individuals: int,
) -> list[dict[str, np.ndarray | float | int]]:
    """Select elites from one named projected archive cell."""
    if max_individuals <= 0 or archive.size == 0:
        return []
    descriptor_names = tuple(str(name) for name in getattr(archive, "descriptor_names", ()))
    if not descriptor_names:
        return []

    axis_cells: dict[int, int] = {}
    for descriptor_name, target in descriptor_targets.items():
        if descriptor_name not in descriptor_names:
            log.info(
                "Projected render selection skipped | missing descriptor=%s archive_descriptors=%s",
                descriptor_name,
                descriptor_names,
            )
            return []
        axis = descriptor_names.index(descriptor_name)
        normalized_target = str(target).strip().lower()
        if normalized_target == "min":
            axis_cells[axis] = 0
        elif normalized_target == "max":
            axis_cells[axis] = int(archive.dims[axis]) - 1
        else:
            raise ValueError(
                f"Unsupported projected render target {target!r} for descriptor {descriptor_name!r}"
            )

    projected = archive.top_elites_in_projection(
        axis_cells=axis_cells,
        n=int(max_individuals),
    )
    return [dict(elite) for elite in projected]


def select_priority_render_elites(
    archive,
    *,
    descriptor_priorities: tuple[tuple[str, str], ...],
    max_individuals: int,
) -> list[dict[str, np.ndarray | float | int]]:
    """Select elites using lexicographic descriptor priorities."""
    if max_individuals <= 0 or archive.size == 0:
        return []
    descriptor_names = tuple(str(name) for name in getattr(archive, "descriptor_names", ()))
    if not descriptor_names:
        return []

    axis_directions: list[tuple[int, str]] = []
    for descriptor_name, direction in descriptor_priorities:
        if descriptor_name not in descriptor_names:
            log.info(
                "Priority render selection skipped | missing descriptor=%s archive_descriptors=%s",
                descriptor_name,
                descriptor_names,
            )
            return []
        axis_directions.append((descriptor_names.index(descriptor_name), str(direction)))

    prioritized = archive.top_elites_by_descriptor_priority(
        axis_directions=tuple(axis_directions),
        n=int(max_individuals),
    )
    return [dict(elite) for elite in prioritized]


def build_wandb_render_payload(
    artifacts: list[RenderArtifact],
    *,
    key_prefix: str,
    fps: int,
) -> dict[str, object]:
    """Convert rendered GIF artifacts into a W&B media payload."""
    if not artifacts:
        return {}

    import wandb

    payload: dict[str, object] = {}
    for item in artifacts:
        rank_key = f"rank_{item.rank}_ep_{item.episode_idx}"
        payload[f"{key_prefix}/{rank_key}"] = wandb.Video(
            item.gif_path,
            fps=int(fps),
            format="gif",
        )
        meta_prefix = f"other/{item.stage}/{rank_key}"
        payload[f"{meta_prefix}/flat_index"] = float(item.flat_index)
        payload[f"{meta_prefix}/archive_fitness"] = float(item.archive_fitness)
        payload[f"{meta_prefix}/selection_fitness"] = float(item.selection_fitness)
        payload[f"{meta_prefix}/episode_fitness"] = float(item.episode_fitness)
        payload[f"{meta_prefix}/goal_hit_count"] = float(item.goal_hit_count)
        payload[f"{meta_prefix}/fall_count"] = float(item.fall_count)
        payload[f"{meta_prefix}/completion_time"] = float(item.completion_time)
        payload[f"{meta_prefix}/terrain_idx"] = float(item.terrain_idx)
        payload[f"{meta_prefix}/cliff_x"] = float(item.cliff_x)
        payload[f"{meta_prefix}/peak_front_x"] = float(item.peak_front_x)
        payload[f"{meta_prefix}/safe_front_x"] = float(item.safe_front_x)
        payload[f"{meta_prefix}/probe_extent"] = float(item.probe_extent)
        payload[f"{meta_prefix}/turn_count"] = float(item.turn_count)
        payload[f"{meta_prefix}/stable_turn_count"] = float(item.stable_turn_count)
        payload[f"{meta_prefix}/turn_motion_threshold"] = float(item.turn_motion_threshold)
    return payload


def _stage_render_cfg(cfg: DictConfig, stage: str) -> DictConfig:
    if stage == "periodic":
        return cfg.render.get("periodic", cfg.render)
    if stage == "specimen_preview":
        return (
            cfg.get("specimens", {}).get("preview_render_quality")
            or cfg.render.get("periodic", cfg.render)
        )
    if stage in {"rerank", "eval"}:
        return cfg.render.get("post_train", cfg.render)
    if stage == "final":
        return cfg.render.get("final") or cfg.render.get("post_train", cfg.render)
    return cfg.render


def _render_target_frames(cfg: DictConfig, stage: str) -> int | None:
    stage_cfg = _stage_render_cfg(cfg, stage)
    value = stage_cfg.get("target_frames")
    if value is None:
        value = cfg.render.get("target_frames")
    if value is None:
        return None
    resolved = int(value)
    return resolved if resolved > 0 else None


def _render_temporal_mode(cfg: DictConfig, stage: str) -> str:
    stage_cfg = _stage_render_cfg(cfg, stage)
    value = stage_cfg.get("temporal_mode")
    if value is None:
        value = cfg.render.get("temporal_mode")
    if value is None:
        return "diagnostic" if stage in {"periodic", "adhoc"} else "realtime"
    resolved = str(value).strip().lower()
    if resolved not in {"diagnostic", "realtime"}:
        raise ValueError(f"Unsupported render temporal_mode={value!r}")
    return resolved


def _render_parallel_workers(cfg: DictConfig, job_count: int) -> int:
    configured = int(cfg.render.get("parallel_workers", 1))
    return max(1, min(int(job_count), configured))


def _render_sim_time_per_step(cfg: DictConfig) -> float:
    return float(int(cfg.sim.physics_substeps) * DT)


def _render_config_for_stage(cfg: DictConfig, stage: str) -> RenderConfig:
    temporal_mode = _render_temporal_mode(cfg, stage)
    stage_cfg = _stage_render_cfg(cfg, stage)

    def _stage_value(key: str) -> int:
        value = stage_cfg.get(key)
        if value is None:
            value = cfg.render.get(key)
        return int(value)

    return RenderConfig(
        enabled=True,
        width=_stage_value("width"),
        height=_stage_value("height"),
        fps=_stage_value("fps"),
        real_time=(temporal_mode == "realtime"),
        frame_skip=1,
        dynamic_actuator_colors=True,
        target_frames=(
            None
            if temporal_mode == "realtime"
            else _render_target_frames(cfg, stage)
        ),
        sim_time_per_step=_render_sim_time_per_step(cfg),
    )


def _encode_renderable_episode(
    item: RenderableEpisode,
    *,
    cfg: DictConfig,
) -> RenderArtifact:
    render_config = _render_config_for_stage(cfg, item.stage)
    render_episode_gif(
        item.trace,
        item.render_info,
        item.output_path,
        config=render_config,
    )
    episode_result = item.episode_result
    return RenderArtifact(
        stage=item.stage,
        rank=int(item.rank),
        episode_idx=int(item.episode_idx),
        flat_index=int(item.flat_index),
        archive_fitness=float(item.archive_fitness),
        selection_fitness=float(item.selection_fitness),
        cliff_x=int(item.cliff_x),
        terrain_idx=int(item.terrain_idx),
        gif_path=str(item.output_path),
        episode_fitness=float(episode_fitness(episode_result)),
        goal_hit_count=float(item.goal_hit_count),
        fall_count=float(item.fall_count),
        completion_time=float(item.completion_time),
        peak_front_x=float(episode_result.peak_front_x),
        safe_front_x=float(episode_result.safe_front_x),
        probe_extent=float(max(0.0, float(episode_result.peak_front_x - episode_result.safe_front_x))),
        turn_count=float(episode_result.turn_count),
        stable_turn_count=float(episode_result.stable_turn_count),
        turn_motion_threshold=float(episode_result.turn_motion_threshold),
    )


def _render_episode_batch(
    episodes: list[RenderableEpisode],
    cfg: DictConfig,
    *,
    stage: str,
    output_dir: Path,
) -> list[RenderArtifact]:
    if not episodes:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    worker_count = _render_parallel_workers(cfg, len(episodes))
    if worker_count == 1:
        artifacts = [_encode_renderable_episode(item, cfg=cfg) for item in episodes]
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="render-gif") as executor:
            artifacts = list(executor.map(lambda item: _encode_renderable_episode(item, cfg=cfg), episodes))

    artifacts.sort(key=lambda item: (int(item.rank), int(item.episode_idx)))
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps([item._asdict() for item in artifacts], indent=2))
    log.info(
        "Render complete | stage=%s elites=%d episodes=%d output=%s workers=%d",
        stage,
        len({int(item.rank) for item in artifacts}),
        len(artifacts),
        output_dir,
        worker_count,
    )
    return artifacts


def render_periodic_archive(
    archive,
    env_datas: list[CliffEnvData] | GrownEvalAssets,
    alive_mask: jnp.ndarray,
    cfg: DictConfig,
    *,
    generation: int,
    output_root: str | Path,
    runtime_resolution: RuntimeResolution | None = None,
) -> list[RenderArtifact]:
    """Render a deterministic periodic snapshot from current-generation terrains."""
    render_cfg = cfg.render.periodic
    elites = select_priority_render_elites(
        archive,
        descriptor_priorities=_PERIODIC_QD_DESCRIPTOR_PRIORITIES,
        max_individuals=int(render_cfg.max_individuals),
    )
    if not elites:
        elites = select_render_elites(
            archive,
            max_cells=int(render_cfg.get("select_cells", archive.size)),
            max_individuals=int(render_cfg.max_individuals),
        )
    elites = _annotate_elite_descriptor_metrics(
        elites,
        descriptor_names=tuple(str(name) for name in getattr(archive, "descriptor_names", ())),
    )
    if not elites:
        return []
    if str(cfg.body.mode).strip().lower() == "grown":
        if not isinstance(env_datas, GrownEvalAssets):
            raise TypeError("Grown periodic rendering expects GrownEvalAssets")
        selected_envs = _select_periodic_grown_render_assets(env_datas, render_cfg)
    else:
        selected_envs = _select_periodic_render_envs(env_datas, render_cfg)
    return _render_selected_elites(
        elites,
        selected_envs,
        alive_mask,
        cfg,
        generation=generation,
        key_scope=_PERIODIC_TRACE_SCOPE,
        stage="periodic",
        output_dir=Path(output_root) / f"gen_{generation:04d}" / "periodic",
        runtime_resolution=runtime_resolution,
    )


def _periodic_qd_render_generations(schedule: TerrainSchedule, *, generation: int) -> list[int]:
    if not schedule.train:
        return []
    phase_ends = list(schedule.curriculum_phase_end_gens or [])
    if not phase_ends:
        return [min(max(int(generation), 0), len(schedule.train) - 1)]

    representative_generations: list[int] = []
    phase_start = 0
    for phase_end in phase_ends:
        phase_end_clamped = min(max(int(phase_end), int(phase_start)), len(schedule.train))
        if phase_end_clamped > phase_start:
            representative_generations.append(phase_end_clamped - 1)
        phase_start = phase_end_clamped
    return representative_generations


def _periodic_qd_render_entries(
    schedule: TerrainSchedule,
    *,
    generation: int,
) -> list[TerrainEntry]:
    entries: list[TerrainEntry] = []
    for gen_idx in _periodic_qd_render_generations(schedule, generation=generation):
        entries.extend(list(schedule.train[gen_idx]))
    return entries


def render_periodic_qd_projection(
    archive,
    alive_mask: jnp.ndarray,
    cfg: DictConfig,
    *,
    generation: int,
    output_root: str | Path,
    runtime_resolution: RuntimeResolution | None = None,
) -> list[RenderArtifact]:
    """Render priority-ranked archive elites across fixed cross-stage curriculum terrains."""
    descriptor_names = tuple(str(name) for name in getattr(archive, "descriptor_names", ()))
    elites = _annotate_elite_descriptor_metrics(
        select_priority_render_elites(
            archive,
            descriptor_priorities=_PERIODIC_QD_DESCRIPTOR_PRIORITIES,
            max_individuals=3,
        ),
        descriptor_names=descriptor_names,
    )
    if not elites:
        return []

    schedule_dir = cfg.train.get("terrain_schedule_dir", None)
    if not schedule_dir:
        log.info("Periodic QD render skipped | no frozen terrain schedule configured")
        return []

    runtime_resolution = runtime_resolution or resolve_runtime_config(cfg)
    schedule = load_schedule(
        schedule_dir,
        allow_legacy_hashes=bool(cfg.train.get("allow_legacy_schedule_hashes", False)),
    )
    entries = _periodic_qd_render_entries(schedule, generation=generation)
    if not entries:
        log.info("Periodic QD render skipped | no render entries resolved from schedule=%s", schedule_dir)
        return []
    env_datas = _build_envs_from_entries(entries, cfg, runtime_resolution)
    log.info(
        "Periodic QD render | elites=%d terrains=%d priorities=%s",
        len(elites),
        _render_env_count(env_datas),
        ",".join(f"{name}:{direction}" for name, direction in _PERIODIC_QD_DESCRIPTOR_PRIORITIES),
    )
    return _render_selected_elites(
        elites,
        env_datas,
        alive_mask,
        cfg,
        generation=generation,
        key_scope=_PERIODIC_QD_TRACE_SCOPE,
        stage="periodic_qd",
        output_dir=Path(output_root) / f"gen_{generation:04d}" / "periodic_qd_directional",
        runtime_resolution=runtime_resolution,
    )


def render_final_archive(
    archive,
    fallback_envs: list[CliffEnvData] | GrownEvalAssets,
    alive_mask: jnp.ndarray,
    cfg: DictConfig,
    *,
    generation: int,
    output_root: str | Path,
    runtime_resolution: RuntimeResolution | None = None,
) -> list[RenderArtifact]:
    """Render the final archive with deterministic reranking when a schedule exists."""
    runtime_resolution = runtime_resolution or resolve_runtime_config(cfg)
    render_cfg = cfg.render.get("final") or cfg.render.get("post_train")
    elites = select_render_elites(
        archive,
        max_cells=int(render_cfg.get("select_cells", render_cfg.get("max_individuals", 0))),
        max_individuals=int(render_cfg.get("max_individuals", 0)),
    )
    if not elites:
        return []

    if str(cfg.body.mode).strip().lower() == "grown":
        if not isinstance(fallback_envs, GrownEvalAssets):
            raise TypeError("Grown final rendering expects GrownEvalAssets")
        render_envs = _select_grown_render_assets(
            fallback_envs,
            int(render_cfg.get("episodes_per_individual", len(fallback_envs.templates))),
        )
    else:
        render_envs = _select_render_envs(
            fallback_envs,
            int(render_cfg.get("episodes_per_individual", len(fallback_envs))),
        )
    reranked = elites
    elites = _annotate_elite_descriptor_metrics(
        elites,
        descriptor_names=tuple(str(name) for name in getattr(archive, "descriptor_names", ())),
    )
    reranked = elites

    schedule_dir = cfg.train.get("terrain_schedule_dir", None)
    if bool(render_cfg.get("rerank_on_validation", True)) and schedule_dir:
        schedule = load_schedule(
            schedule_dir,
            allow_legacy_hashes=bool(cfg.train.get("allow_legacy_schedule_hashes", False)),
        )
        val_envs = _build_envs_from_entries(
            schedule.val,
            cfg,
            runtime_resolution,
        )
        test_envs = _build_envs_from_entries(
            schedule.test,
            cfg,
            runtime_resolution,
        )
        reranked = _rerank_elites(
            elites,
            val_envs,
            alive_mask,
            cfg,
            generation=generation,
            key_scope=_FINAL_RERANK_SCOPE,
            runtime_resolution=runtime_resolution,
        )
        if str(cfg.body.mode).strip().lower() == "grown":
            render_envs = _select_grown_render_assets(
                test_envs,
                int(render_cfg.get("episodes_per_individual", len(test_envs.templates))),
            )
        else:
            render_envs = _select_render_envs(
                test_envs,
                int(render_cfg.get("episodes_per_individual", len(test_envs))),
            )
        log.info(
            "Final render rerank | candidates=%d val_terrains=%d render_terrains=%d",
            len(elites),
            _render_env_count(val_envs),
            _render_env_count(render_envs),
        )
    elif bool(render_cfg.get("rerank_on_validation", True)):
        log.info("Final render rerank skipped | no frozen terrain schedule configured")

    return _render_selected_elites(
        reranked[: int(render_cfg.get("max_individuals", len(reranked)))],
        render_envs,
        alive_mask,
        cfg,
        generation=generation,
        key_scope=_FINAL_TRACE_SCOPE,
        stage="final",
        output_dir=Path(output_root) / "final",
        runtime_resolution=runtime_resolution,
    )


def resolve_render_limit(value: int | str | None, total: int) -> int:
    """Resolve an integer-or-'all' render count against the available population."""
    if total <= 0 or value is None:
        return 0
    if isinstance(value, str) and value.strip().lower() == "all":
        return int(total)
    return max(0, min(int(value), int(total)))


def render_ranked_elites(
    elites: list[dict[str, np.ndarray | float | int]],
    env_datas: list[CliffEnvData] | GrownEvalAssets,
    alive_mask: jnp.ndarray,
    cfg: DictConfig,
    *,
    generation: int,
    stage: str,
    output_dir: str | Path,
    runtime_resolution: RuntimeResolution | None = None,
) -> list[RenderArtifact]:
    """Render an explicit ranked elite list for a post-train stage."""
    scope = {
        "rerank": _POST_TRAIN_RERANK_SCOPE,
        "eval": _POST_TRAIN_EVAL_SCOPE,
    }.get(stage, _POST_TRAIN_EVAL_SCOPE)
    return _render_selected_elites(
        elites,
        env_datas,
        alive_mask,
        cfg,
        generation=generation,
        key_scope=scope,
        stage=stage,
        output_dir=Path(output_dir),
        runtime_resolution=runtime_resolution,
    )


def render_top_k_from_archive(
    archive,
    env_datas: list[CliffEnvData],
    alive_mask: jnp.ndarray,
    key: jax.Array,
    cfg: DictConfig,
    *,
    k: int = 3,
    output_dir: str | Path = "renders/",
    runtime_resolution: RuntimeResolution | None = None,
) -> list[dict]:
    """Backward-compatible wrapper for ad hoc top-k rendering."""
    del key
    artifacts = _render_selected_elites(
        select_render_elites(archive, max_cells=k, max_individuals=k),
        _select_render_envs(env_datas, 1),
        alive_mask,
        cfg,
        generation=0,
        key_scope=_PERIODIC_TRACE_SCOPE,
        stage="adhoc",
        output_dir=Path(output_dir),
        runtime_resolution=runtime_resolution,
    )
    return [artifact._asdict() for artifact in artifacts]


def _event_root_key(base_seed: int, generation: int, scope: int) -> jax.Array:
    key = jax.random.PRNGKey(int(base_seed))
    key = jax.random.fold_in(key, int(scope))
    key = jax.random.fold_in(key, int(generation))
    return key


def _select_render_envs(
    env_datas: list[CliffEnvData],
    episodes_per_individual: int,
) -> list[CliffEnvData]:
    if episodes_per_individual <= 0:
        return []
    return list(env_datas[: min(len(env_datas), int(episodes_per_individual))])


def _periodic_terrain_selection_mode(render_cfg: DictConfig) -> str:
    mode = str(render_cfg.get("terrain_selection", "all")).strip().lower()
    if mode not in _PERIODIC_TERRAIN_SELECTION_MODES:
        raise ValueError(
            "render.periodic.terrain_selection must be one of "
            f"{sorted(_PERIODIC_TERRAIN_SELECTION_MODES)}"
        )
    return mode


def _select_periodic_render_envs(
    env_datas: list[CliffEnvData],
    render_cfg: DictConfig,
) -> list[CliffEnvData]:
    mode = _periodic_terrain_selection_mode(render_cfg)
    if mode == "all":
        return list(env_datas)
    return _select_render_envs(
        env_datas,
        resolve_render_limit(
            render_cfg.get("episodes_per_individual", len(env_datas)),
            len(env_datas),
        ),
    )


def _slice_static_template_batch(
    template_batch: CliffStaticTemplateBatch,
    limit: int,
) -> CliffStaticTemplateBatch:
    return CliffStaticTemplateBatch(
        static_collider_data=jax.tree_util.tree_map(
            lambda array: array[:limit],
            template_batch.static_collider_data,
        ),
        cliff_x=template_batch.cliff_x[:limit],
        death_plane_y=template_batch.death_plane_y[:limit],
        task_mode=template_batch.task_mode[:limit],
        goal_direction=template_batch.goal_direction[:limit],
        cliff_side=template_batch.cliff_side[:limit],
        goal_distance=template_batch.goal_distance[:limit],
        cliff_distance=template_batch.cliff_distance[:limit],
        goal_block_x=template_batch.goal_block_x[:limit],
        goal_block_y=template_batch.goal_block_y[:limit],
        goal_x=template_batch.goal_x[:limit],
        goal_object_id=template_batch.goal_object_id[:limit],
        cliff_probe_ledge_width=template_batch.cliff_probe_ledge_width[:limit],
        cliff_probe_ledge_cell_x=template_batch.cliff_probe_ledge_cell_x[:limit],
        cliff_probe_ledge_cell_y=template_batch.cliff_probe_ledge_cell_y[:limit],
    )


def _select_grown_render_assets(
    env_assets: GrownEvalAssets,
    episodes_per_individual: int,
) -> GrownEvalAssets:
    limit = (
        0
        if episodes_per_individual <= 0
        else min(len(env_assets.templates), int(episodes_per_individual))
    )
    return GrownEvalAssets(
        static_templates=_slice_static_template_batch(env_assets.static_templates, limit),
        templates=env_assets.templates[:limit],
        static_capacity=env_assets.static_capacity,
        grid_data=env_assets.grid_data,
        runtime_build_capacity=env_assets.runtime_build_capacity,
    )


def _select_periodic_grown_render_assets(
    env_assets: GrownEvalAssets,
    render_cfg: DictConfig,
) -> GrownEvalAssets:
    mode = _periodic_terrain_selection_mode(render_cfg)
    if mode == "all":
        return env_assets
    return _select_grown_render_assets(
        env_assets,
        resolve_render_limit(
            render_cfg.get("episodes_per_individual", len(env_assets.templates)),
            len(env_assets.templates),
        ),
    )


def _render_env_count(env_datas: list[CliffEnvData] | GrownEvalAssets) -> int:
    if isinstance(env_datas, GrownEvalAssets):
        return len(env_datas.templates)
    return len(env_datas)


def _build_envs_from_entries(
    entries: list[TerrainEntry],
    cfg: DictConfig,
    runtime_resolution: RuntimeResolution,
) -> list[CliffEnvData] | GrownEvalAssets:
    spawn_x = resolved_spawn_x(cfg)
    spawn_y = int(cfg.terrain.spawn_y)
    if str(cfg.body.mode).strip().lower() == "grown":
        templates = [
            build_static_collider_template(
                entry.grid,
                cliff_x=int(entry.cliff_x),
                task_metadata=entry.metadata,
                canvas_h=int(cfg.body.growth.max_robot_h),
                canvas_w=int(cfg.body.growth.max_robot_w),
                spawn_x=spawn_x,
                spawn_y=spawn_y,
                spring_stiffness_scale=float(cfg.sim.spring_stiffness_scale),
                collision_build_capacity=runtime_resolution.resolved_collision.build_capacity,
            )
            for entry in entries
        ]
        static_capacity = infer_static_template_capacity(templates)
        return GrownEvalAssets(
            static_templates=stack_static_templates(templates, static_capacity=static_capacity),
            templates=tuple(templates),
            static_capacity=static_capacity,
            grid_data=precompute_shared_grid_data(
                int(cfg.terrain.grid_h),
                int(cfg.terrain.grid_w),
            ),
            runtime_build_capacity=runtime_resolution.resolved_collision.runtime_build_capacity,
        )

    robot_morph = make_robot_body(
        cfg.terrain.robot_w,
        cfg.terrain.robot_h,
        shape=str(cfg.body.get("shape", "rectangle")),
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
        collision_build_capacity=runtime_resolution.resolved_collision.runtime_build_capacity,
    )
    return [
        assemble_cliff_env(
            dynamic_core,
            build_cliff_terrain_assets(
                entry.grid,
                robot_morph,
                spawn_x,
                spawn_y,
                entry.cliff_x,
                task_metadata=entry.metadata,
                spring_stiffness_scale=float(cfg.sim.spring_stiffness_scale),
                collision_build_capacity=runtime_resolution.resolved_collision.build_capacity,
            ),
        )
        for entry in entries
    ]


def _rerank_elites(
    elites: list[dict[str, np.ndarray | float | int]],
    env_datas: list[CliffEnvData] | GrownEvalAssets,
    alive_mask: jnp.ndarray,
    cfg: DictConfig,
    *,
    generation: int,
    key_scope: int,
    runtime_resolution: RuntimeResolution,
) -> list[dict[str, np.ndarray | float | int]]:
    if not elites or not env_datas:
        return elites

    val_static_capacity = (
        None
        if isinstance(env_datas, GrownEvalAssets)
        else infer_static_collider_capacity(env_datas)
    )
    genotypes = np.stack([np.asarray(elite["genotype"]) for elite in elites], axis=0)
    eval_key = _event_root_key(int(cfg.seed), generation, key_scope)
    candidate_keys = _candidate_keys(eval_key, len(elites))
    validation_batching = runtime_resolution.resolved_batching["validation"]
    if validation_batching.warmup_enabled:
        warmup_chunk_sizes = planned_warmup_chunk_sizes(len(elites), validation_batching)
        log.info(
            "Warmup | workload=validation_render terrains=%d planned=%s",
            _render_env_count(env_datas),
            list(warmup_chunk_sizes),
        )
        warmup_timings = warmup_population_evaluator(
            int(genotypes.shape[1]),
            env_datas,
            alive_mask,
            cfg,
            workload="validation",
            runtime_resolution=runtime_resolution,
            static_capacity=val_static_capacity,
            candidate_count=len(elites),
        )
        for bucket_size, elapsed_s in warmup_timings.items():
            log.info(
                "Warmup | workload=validation_render bucket=%d elapsed_s=%.2f",
                bucket_size,
                elapsed_s,
            )

    val_eval = evaluate_population_chunked(
        genotypes,
        env_datas,
        alive_mask,
        candidate_keys,
        cfg,
        workload="validation",
        runtime_resolution=runtime_resolution,
        static_capacity=val_static_capacity,
    )
    scored: list[dict[str, np.ndarray | float | int]] = []
    for idx, elite in enumerate(elites):
        elite_copy = dict(elite)
        elite_copy["selection_fitness"] = float(val_eval.fitnesses[idx])
        scored.append(elite_copy)
    scored.sort(
        key=lambda elite: (-float(elite["selection_fitness"]), int(elite["flat_index"])),
    )
    return scored


def _render_selected_elites_grown(
    elites: list[dict[str, np.ndarray | float | int]],
    env_assets: GrownEvalAssets,
    alive_mask: jnp.ndarray,
    cfg: DictConfig,
    *,
    generation: int,
    key_scope: int,
    stage: str,
    output_dir: Path,
    runtime_resolution: RuntimeResolution,
) -> list[RenderArtifact]:
    if not elites or not env_assets.templates:
        return []

    trace_batching = runtime_resolution.resolved_batching["trace"]
    genotypes = np.stack([np.asarray(elite["genotype"]) for elite in elites], axis=0)
    genotype_dim = int(genotypes.shape[1])
    trace_key = _event_root_key(int(cfg.seed), generation, key_scope)
    candidate_keys = _candidate_keys(trace_key, len(elites))
    if trace_batching.warmup_enabled:
        trace_warmup_chunk_sizes = planned_warmup_chunk_sizes(len(elites), trace_batching)
        log.info(
            "Warmup | workload=%s terrains=%d planned=%s",
            f"trace_{stage}",
            len(env_assets.templates),
            list(trace_warmup_chunk_sizes),
        )
        warmup_timings = warmup_trace_evaluator(
            genotype_dim,
            env_assets,
            alive_mask,
            cfg,
            workload="trace",
            runtime_resolution=runtime_resolution,
            candidate_count=len(elites),
        )
        for bucket_size, elapsed_s in warmup_timings.items():
            log.info(
                "Warmup | workload=%s bucket=%d elapsed_s=%.2f",
                f"trace_{stage}",
                bucket_size,
                elapsed_s,
            )

    trace_eval = trace_population_chunked(
        genotypes,
        env_assets,
        alive_mask,
        candidate_keys,
        cfg,
        workload="trace",
        runtime_resolution=runtime_resolution,
    )
    if trace_eval.render_preps is None:
        raise ValueError("Grown trace rendering requires dynamic render prep data")

    renderables: list[RenderableEpisode] = []
    n_dynamic_points = int(np.asarray(env_assets.grid_data.positions).shape[0])
    for rank, elite in enumerate(elites):
        render_prep = jax.tree_util.tree_map(
            lambda array: array[rank],
            trace_eval.render_preps,
        )
        dynamic_render_info = build_grown_dynamic_render_info(
            render_prep.robot_morph,
            env_assets.grid_data,
            spawn_x=resolved_spawn_x(cfg),
            spawn_y=int(cfg.terrain.spawn_y),
            spring_init_rest_length=render_prep.spring_init_rest_length,
        )
        selection_fitness = float(elite.get("selection_fitness", elite["fitness"]))
        for episode_idx, template in enumerate(env_assets.templates):
            def _select_episode_leaf(array):
                arr = jnp.asarray(array)
                return array[rank, episode_idx] if arr.ndim >= 2 else array

            ep_result = jax.tree_util.tree_map(
                _select_episode_leaf,
                trace_eval.episode_results,
            )
            trace = jax.tree_util.tree_map(
                lambda array: array[rank, episode_idx],
                trace_eval.traces,
            )
            render_info = merge_render_info(
                dynamic_render_info,
                template.render_info,
                n_dynamic_points=n_dynamic_points,
                grid_h=template.grid_h,
                grid_w=template.grid_w,
            )
            if render_info is None:
                continue
            renderables.append(
                RenderableEpisode(
                    stage=stage,
                    rank=int(rank),
                    episode_idx=int(episode_idx),
                    flat_index=int(elite["flat_index"]),
                    archive_fitness=float(elite["fitness"]),
                    selection_fitness=selection_fitness,
                    cliff_x=int(ep_result.cliff_x),
                    terrain_idx=int(episode_idx),
                    render_info=render_info,
                    episode_result=ep_result,
                    trace=trace,
                    output_path=output_dir / f"rank_{rank}_ep_{episode_idx}.gif",
                    goal_hit_count=float(elite.get("goal_hit_count", np.nan)),
                    fall_count=float(elite.get("fall_count", np.nan)),
                    completion_time=float(elite.get("completion_time", np.nan)),
                )
            )
    return _render_episode_batch(
        renderables,
        cfg,
        stage=stage,
        output_dir=output_dir,
    )


def _render_selected_elites(
    elites: list[dict[str, np.ndarray | float | int]],
    env_datas: list[CliffEnvData] | GrownEvalAssets,
    alive_mask: jnp.ndarray,
    cfg: DictConfig,
    *,
    generation: int,
    key_scope: int,
    stage: str,
    output_dir: Path,
    runtime_resolution: RuntimeResolution | None = None,
) -> list[RenderArtifact]:
    if not elites or not env_datas:
        return []

    runtime_resolution = runtime_resolution or resolve_runtime_config(cfg)
    if str(cfg.body.mode).strip().lower() == "grown":
        if not isinstance(env_datas, GrownEvalAssets):
            raise TypeError("Grown rendering expects GrownEvalAssets")
        return _render_selected_elites_grown(
            elites,
            env_datas,
            alive_mask,
            cfg,
            generation=generation,
            key_scope=key_scope,
            stage=stage,
            output_dir=output_dir,
            runtime_resolution=runtime_resolution,
        )

    trace_batching = runtime_resolution.resolved_batching["trace"]
    trace_static_capacity: StaticColliderCapacity = infer_static_collider_capacity(env_datas)
    genotypes = np.stack([np.asarray(elite["genotype"]) for elite in elites], axis=0)
    genotype_dim = int(genotypes.shape[1])
    trace_key = _event_root_key(int(cfg.seed), generation, key_scope)
    candidate_keys = _candidate_keys(trace_key, len(elites))

    if trace_batching.warmup_enabled:
        trace_warmup_chunk_sizes = planned_warmup_chunk_sizes(len(elites), trace_batching)
        log.info(
            "Warmup | workload=%s terrains=%d planned=%s",
            f"trace_{stage}",
            len(env_datas),
            list(trace_warmup_chunk_sizes),
        )
        warmup_timings = warmup_trace_evaluator(
            genotype_dim,
            env_datas,
            alive_mask,
            cfg,
            workload="trace",
            runtime_resolution=runtime_resolution,
            static_capacity=trace_static_capacity,
            candidate_count=len(elites),
        )
        for bucket_size, elapsed_s in warmup_timings.items():
            log.info(
                "Warmup | workload=%s bucket=%d elapsed_s=%.2f",
                f"trace_{stage}",
                bucket_size,
                elapsed_s,
            )

    trace_eval = trace_population_chunked(
        genotypes,
        env_datas,
        alive_mask,
        candidate_keys,
        cfg,
        workload="trace",
        runtime_resolution=runtime_resolution,
        static_capacity=trace_static_capacity,
    )

    renderables: list[RenderableEpisode] = []
    for rank, elite in enumerate(elites):
        selection_fitness = float(elite.get("selection_fitness", elite["fitness"]))
        for episode_idx, env_data in enumerate(env_datas):
            ep_result = jax.tree_util.tree_map(
                lambda array: array[rank, episode_idx],
                trace_eval.episode_results,
            )
            trace = jax.tree_util.tree_map(
                lambda array: array[rank, episode_idx],
                trace_eval.traces,
            )
            if env_data.render_info is None:
                continue
            renderables.append(
                RenderableEpisode(
                    stage=stage,
                    rank=int(rank),
                    episode_idx=int(episode_idx),
                    flat_index=int(elite["flat_index"]),
                    archive_fitness=float(elite["fitness"]),
                    selection_fitness=selection_fitness,
                    cliff_x=int(ep_result.cliff_x),
                    terrain_idx=int(episode_idx),
                    render_info=env_data.render_info,
                    episode_result=ep_result,
                    trace=trace,
                    output_path=output_dir / f"rank_{rank}_ep_{episode_idx}.gif",
                    goal_hit_count=float(elite.get("goal_hit_count", np.nan)),
                    fall_count=float(elite.get("fall_count", np.nan)),
                    completion_time=float(elite.get("completion_time", np.nan)),
                )
            )
    return _render_episode_batch(
        renderables,
        cfg,
        stage=stage,
        output_dir=output_dir,
    )


def _descriptor_metric_value(
    descriptor_names: tuple[str, ...],
    descriptor: np.ndarray,
    *,
    name: str,
) -> float:
    if name not in descriptor_names:
        return float("nan")
    return float(descriptor[descriptor_names.index(name)])


def _annotate_elite_descriptor_metrics(
    elites: list[dict[str, np.ndarray | float | int]],
    *,
    descriptor_names: tuple[str, ...],
) -> list[dict[str, np.ndarray | float | int]]:
    annotated: list[dict[str, np.ndarray | float | int]] = []
    for elite in elites:
        elite_copy = dict(elite)
        descriptor = np.asarray(elite_copy.get("descriptor", ()), dtype=np.float32)
        elite_copy["goal_hit_count"] = _descriptor_metric_value(
            descriptor_names,
            descriptor,
            name="goal_hit_count",
        )
        elite_copy["fall_count"] = _descriptor_metric_value(
            descriptor_names,
            descriptor,
            name="fall_count",
        )
        elite_copy["completion_time"] = _descriptor_metric_value(
            descriptor_names,
            descriptor,
            name="completion_time",
        )
        annotated.append(elite_copy)
    return annotated


def _candidate_keys(root_key: jax.Array, count: int) -> jax.Array:
    keys: list[jax.Array] = []
    for rank in range(int(count)):
        key = jax.random.fold_in(root_key, int(rank))
        keys.append(key)
    return jnp.stack(keys) if keys else jnp.zeros((0, 2), dtype=jnp.uint32)
