"""Chunked batched evaluators for sensing_edges training, eval, and trace capture."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from jax_evogym.constants import CONTRACTILE, EMPTY
from jax_evogym.types import GridData

from .env import (
    CliffEnvData,
    CliffEvalEnvData,
    EpisodeComTrace,
    EpisodePhaseProbeTrace,
    EpisodeSubstepProbeTrace,
    EpisodeResult,
    CliffStaticTemplate,
    CliffStaticTemplateBatch,
    StaticColliderCapacity,
    attach_static_template_to_eval_env,
    build_cliff_eval_dynamic_core,
    infer_static_collider_capacity,
    run_episode,
    run_episode_with_com_trace,
    run_episode_with_phase_probe,
    run_episode_with_substep_probe,
    run_episode_with_trace,
    stack_eval_envs,
)
from .fitness import (
    CANDIDATE_SCALAR_METRIC_ATTRS,
    CandidateBatchResult,
    candidate_batch_fitness,
)
from .collision_tuning import CollisionBuildCapacity
from .nca import ctrnn_flat_to_params, run_growth_phase
from .qd.gait_descriptor import compute_candidate_gait_spectral_purity
from .qd_descriptors import QDDescriptorSpec, build_descriptor_matrix, resolve_qd_archive
from .runtime_config import (
    RuntimeResolution,
    WorkloadBatching,
    noise_enabled_from_cfg,
    rollout_kwargs_from_cfg,
    resolved_spawn_x,
)
from .terrain import resolve_growth_seed_position

_POPULATION_EVALUATOR_CACHE: dict[tuple, Callable] = {}
_TRACE_EVALUATOR_CACHE: dict[tuple, Callable] = {}
_COM_TRACE_EVALUATOR_CACHE: dict[tuple, Callable] = {}
_EPISODE_PHASE_PROBE_EVALUATOR_CACHE: dict[tuple, Callable] = {}
_EPISODE_SUBSTEP_PROBE_EVALUATOR_CACHE: dict[tuple, Callable] = {}


@dataclass(frozen=True)
class ChunkProgress:
    """Host-side progress information emitted after each completed chunk."""

    workload: str
    chunk_index: int
    total_chunks: int
    evaluated_candidates: int
    total_candidates: int
    bucket_size: int
    elapsed_s: float
    best_fitness: float
    mean_fitness: float


@dataclass(frozen=True)
class ChunkedEvalStats:
    """Execution statistics for one chunked evaluation call."""

    workload: str
    elapsed_s: float
    candidate_count: int
    chunk_count: int
    bucket_counts: dict[int, int]

    @property
    def mean_chunk_size(self) -> float:
        if self.chunk_count <= 0:
            return 0.0
        return float(self.candidate_count) / float(self.chunk_count)

    @property
    def max_chunk_size(self) -> int:
        if not self.bucket_counts:
            return 0
        return max(int(bucket) for bucket in self.bucket_counts)


@dataclass(frozen=True)
class ChunkedPopulationEvalResult:
    """Batched population evaluation outputs plus host-friendly summaries."""

    batch_result: CandidateBatchResult | "PopulationBatchSummary"
    fitnesses: np.ndarray
    descriptors: np.ndarray
    stats: ChunkedEvalStats


@dataclass(frozen=True)
class PopulationBatchSummary:
    """Training-only host summary for archive insertion and reporting."""

    fitness: np.ndarray
    goal_hit_count: np.ndarray
    mirrored_pair_count: np.ndarray
    episode_fell: np.ndarray
    episode_left_edge: np.ndarray
    deformation_channel_max: np.ndarray
    controller_hidden_output_max_abs: np.ndarray
    action_output_max_abs: np.ndarray
    lower_ledge_penalty: np.ndarray = 0.0
    descriptor_columns: dict[str, np.ndarray] | None = None
    episode_fitnesses: np.ndarray | None = None
    episode_goal_touched: np.ndarray | None = None
    episode_goal_distances: np.ndarray | None = None
    episode_cliff_distances: np.ndarray | None = None
    episode_times_to_goal: np.ndarray | None = None
    episode_first_significant_directions: np.ndarray | None = None


@dataclass(frozen=True)
class ChunkedTraceEvalResult:
    """Trace capture outputs for a candidate batch."""

    episode_results: EpisodeResult
    traces: object
    render_preps: object | None
    stats: ChunkedEvalStats


@dataclass(frozen=True)
class CandidateEpisodePhaseProbeResult:
    """Repeated single-candidate phase trace outputs for divergence analysis."""

    episode_results: EpisodeResult
    phase_trace: EpisodePhaseProbeTrace


@dataclass(frozen=True)
class CandidateEpisodeSubstepProbeResult:
    """Repeated single-candidate selected-phase substep outputs."""

    episode_result: EpisodeResult
    substep_trace: EpisodeSubstepProbeTrace


@dataclass(frozen=True)
class GrownEvalAssets:
    """Shared grown-mode evaluation assets used by batched numeric evaluation."""

    static_templates: CliffStaticTemplateBatch
    templates: tuple[CliffStaticTemplate, ...]
    static_capacity: StaticColliderCapacity
    grid_data: GridData
    runtime_build_capacity: CollisionBuildCapacity


class GrownRenderPrep(NamedTuple):
    """Compact dynamic render metadata emitted by the grown trace evaluator."""

    robot_morph: jnp.ndarray
    spring_init_rest_length: jnp.ndarray


def split_key_sequence(key: jax.Array, count: int) -> tuple[jax.Array, jax.Array]:
    """Recursively split a JAX key `count` times, matching serial split semantics."""

    def _body(carry, _):
        next_key, out_key = jax.random.split(carry)
        return next_key, out_key

    return jax.lax.scan(_body, key, None, length=count)


def _requires_gait_spectral_purity(
    descriptor_specs: tuple[QDDescriptorSpec, ...],
) -> bool:
    return any(spec.metric_attr == "gait_spectral_purity" for spec in descriptor_specs)


def _gait_spectral_purity_bonus_scale_from_cfg(cfg: DictConfig) -> float:
    return float(cfg.train.get("gait_spectral_purity_bonus_scale", 0.0))


def _soft_loft_penalty_scale_from_cfg(cfg: DictConfig) -> float:
    return float(cfg.train.get("soft_loft_penalty_scale", 0.0))


def _requires_gait_com_trace(
    descriptor_specs: tuple[QDDescriptorSpec, ...],
    *,
    gait_spectral_purity_bonus_scale: float,
) -> bool:
    return (
        _requires_gait_spectral_purity(descriptor_specs)
        or gait_spectral_purity_bonus_scale > 0.0
    )


def _gait_bonus_gate(
    goal_hit_count: np.ndarray,
    episode_fell: np.ndarray,
) -> np.ndarray:
    goal_hits = np.asarray(goal_hit_count, dtype=np.float32).reshape(-1)
    fell = np.asarray(episode_fell, dtype=bool)
    zero_falls = ~np.any(fell, axis=-1)
    return ((goal_hits > 0.0) & zero_falls).astype(np.float32)


def _augment_batch_result_with_gait_host(
    batch_result: CandidateBatchResult | PopulationBatchSummary,
    gait_spectral_purity: np.ndarray,
    *,
    gait_spectral_purity_bonus_scale: float = 0.0,
) -> CandidateBatchResult | PopulationBatchSummary:
    gait_values = np.asarray(gait_spectral_purity, dtype=np.float32).reshape(-1)
    gait_bonus = (
        _gait_bonus_gate(batch_result.goal_hit_count, batch_result.episode_fell)
        * gait_values
        * np.float32(gait_spectral_purity_bonus_scale)
    )
    if isinstance(batch_result, PopulationBatchSummary):
        descriptor_columns = dict(batch_result.descriptor_columns or {})
        descriptor_columns["gait_spectral_purity"] = gait_values
        return replace(
            batch_result,
            fitness=np.asarray(batch_result.fitness, dtype=np.float32) + gait_bonus,
            descriptor_columns=descriptor_columns,
        )
    return batch_result._replace(
        fitness=np.asarray(batch_result.fitness, dtype=np.float32) + gait_bonus,
        gait_spectral_purity=gait_values,
    )


def make_population_evaluator(
    cfg: DictConfig,
    runtime_resolution: RuntimeResolution,
) -> Callable[[jnp.ndarray, CliffEvalEnvData, jnp.ndarray, jax.Array], CandidateBatchResult]:
    """Build the shared jitted candidate-evaluation kernel."""
    signature = _evaluator_signature(cfg, runtime_resolution)
    cached = _POPULATION_EVALUATOR_CACHE.get(signature)
    if cached is not None:
        return cached

    (
        ctrnn_hidden,
        ctrnn_grid_hidden,
        gasnet_mode,
        ou_enabled,
        growth_enabled,
        activation_cost_penalty_scale,
        soft_loft_penalty_scale,
        rollout_items,
    ) = signature
    if growth_enabled:
        raise ValueError("Fixed population evaluator cannot be used for grown mode")
    rollout_kwargs = dict(rollout_items)
    active_comm_filters = tuple(rollout_kwargs["active_comm_filters"])

    def _params_from_genotype(genotype: jnp.ndarray):
        return ctrnn_flat_to_params(
            genotype,
            ctrnn_hidden=ctrnn_hidden,
            ctrnn_grid_hidden=ctrnn_grid_hidden,
            gasnet_mode=gasnet_mode,
            ou_enabled=ou_enabled,
            active_comm_filters=active_comm_filters,
        )

    def _evaluate_chunk(
        genotypes: jnp.ndarray,
        env_batch: CliffEvalEnvData,
        alive_mask: jnp.ndarray,
        candidate_keys: jax.Array,
    ) -> CandidateBatchResult:
        params_batch = jax.vmap(_params_from_genotype)(genotypes)
        n_envs = env_batch.cliff_x.shape[0]
        episode_keys = jax.vmap(lambda key: split_key_sequence(key, n_envs)[1])(candidate_keys)

        def _evaluate_candidate(params, keys):
            return jax.vmap(
                lambda env, ep_key: run_episode(
                    env,
                    params,
                    alive_mask,
                    ep_key,
                    **rollout_kwargs,
                )
            )(env_batch, keys)

        episode_results = jax.vmap(_evaluate_candidate)(params_batch, episode_keys)
        alive_count = jnp.sum(alive_mask.astype(jnp.float32))
        return candidate_batch_fitness(
            episode_results,
            alive_cell_count=jnp.full((genotypes.shape[0],), alive_count, dtype=jnp.float32),
            activation_cost_penalty_scale=activation_cost_penalty_scale,
            soft_loft_penalty_scale=soft_loft_penalty_scale,
        )

    evaluator = jax.jit(_evaluate_chunk)
    _POPULATION_EVALUATOR_CACHE[signature] = evaluator
    return evaluator


def make_com_trace_evaluator(
    cfg: DictConfig,
    runtime_resolution: RuntimeResolution,
) -> Callable[[jnp.ndarray, CliffEvalEnvData, jnp.ndarray, jax.Array], tuple[EpisodeResult, EpisodeComTrace]]:
    """Build a jitted evaluator that returns episode results plus COM-x traces."""
    signature = ("com_trace", *_evaluator_signature(cfg, runtime_resolution))
    cached = _COM_TRACE_EVALUATOR_CACHE.get(signature)
    if cached is not None:
        return cached

    (
        ctrnn_hidden,
        ctrnn_grid_hidden,
        gasnet_mode,
        ou_enabled,
        growth_enabled,
        _activation_cost_penalty_scale,
        _soft_loft_penalty_scale,
        rollout_items,
    ) = _evaluator_signature(cfg, runtime_resolution)
    if growth_enabled:
        raise ValueError("COM trace evaluator cannot be used for grown mode")
    rollout_kwargs = dict(rollout_items)
    active_comm_filters = tuple(rollout_kwargs["active_comm_filters"])

    def _params_from_genotype(genotype: jnp.ndarray):
        return ctrnn_flat_to_params(
            genotype,
            ctrnn_hidden=ctrnn_hidden,
            ctrnn_grid_hidden=ctrnn_grid_hidden,
            gasnet_mode=gasnet_mode,
            ou_enabled=ou_enabled,
            active_comm_filters=active_comm_filters,
        )

    def _trace_chunk(
        genotypes: jnp.ndarray,
        env_batch: CliffEvalEnvData,
        alive_mask: jnp.ndarray,
        candidate_keys: jax.Array,
    ) -> tuple[EpisodeResult, EpisodeComTrace]:
        params_batch = jax.vmap(_params_from_genotype)(genotypes)
        n_envs = env_batch.cliff_x.shape[0]
        episode_keys = jax.vmap(lambda key: split_key_sequence(key, n_envs)[1])(candidate_keys)

        def _trace_candidate(params, keys):
            return jax.vmap(
                lambda env, ep_key: run_episode_with_com_trace(
                    env,
                    params,
                    alive_mask,
                    ep_key,
                    **rollout_kwargs,
                )
            )(env_batch, keys)

        return jax.vmap(_trace_candidate)(params_batch, episode_keys)

    evaluator = jax.jit(_trace_chunk)
    _COM_TRACE_EVALUATOR_CACHE[signature] = evaluator
    return evaluator


def make_grown_population_evaluator(
    cfg: DictConfig,
    runtime_resolution: RuntimeResolution,
    env_assets: GrownEvalAssets,
) -> Callable[[jnp.ndarray, CliffStaticTemplateBatch, jax.Array], CandidateBatchResult]:
    """Build the shared jitted grown-body candidate-evaluation kernel."""
    runtime_build_capacity = env_assets.runtime_build_capacity
    grid_data = env_assets.grid_data
    base_signature = _evaluator_signature(cfg, runtime_resolution)
    signature = (
        *base_signature,
        int(runtime_build_capacity.max_triples),
        int(runtime_build_capacity.max_self_triples),
    )
    cached = _POPULATION_EVALUATOR_CACHE.get(signature)
    if cached is not None:
        return cached

    (
        ctrnn_hidden,
        ctrnn_grid_hidden,
        gasnet_mode,
        ou_enabled,
        growth_enabled,
        activation_cost_penalty_scale,
        soft_loft_penalty_scale,
        rollout_items,
    ) = base_signature
    if not growth_enabled:
        raise ValueError("make_grown_population_evaluator requires body.mode='grown'")

    rollout_kwargs = dict(rollout_items)
    active_comm_filters = tuple(rollout_kwargs["active_comm_filters"])
    growth_cfg = cfg.body.growth
    canvas_h = int(growth_cfg.max_robot_h)
    canvas_w = int(growth_cfg.max_robot_w)
    seed_y, seed_x = resolve_growth_seed_position(
        canvas_h,
        canvas_w,
        str(growth_cfg.get("seed_position", "center")),
    )
    state_transition = str(growth_cfg.get("state_transition", "reset")).strip().lower()
    if state_transition not in {"reset", "pass_through"}:
        raise ValueError(f"Unsupported growth.state_transition={state_transition!r}")
    growth_steps = int(growth_cfg.steps)
    growth_threshold = float(growth_cfg.get("threshold", 0.5))
    frontier_identity_only = bool(growth_cfg.get("frontier_identity_only", True))
    connectivity_max_iters = int(growth_cfg.get("connectivity_max_iters", 10))
    noise_init_scale = float(growth_cfg.get("noise_init_scale", 0.0))
    p_update = float(cfg.nca.p_update)
    update_mode = str(cfg.nca.update_mode)
    state_clip_value = float(cfg.nca.state_clip_value)
    ctrnn_tau_min = float(cfg.nca.ctrnn_tau_min)
    ctrnn_tau_max = float(cfg.nca.ctrnn_tau_max)
    ctrnn_dt = float(cfg.nca.ctrnn_dt)
    spawn_x = resolved_spawn_x(cfg)
    spawn_y = int(cfg.terrain.spawn_y)
    physics_substeps = int(cfg.sim.physics_substeps)
    actuator_convergence = float(cfg.sim.actuator_convergence)
    spring_stiffness_scale = float(cfg.sim.spring_stiffness_scale)
    floor_static_friction_const = float(cfg.sim.floor_static_friction_const)
    floor_stiction_stiffness = float(cfg.sim.floor_stiction_stiffness)
    floor_stiction_switch_speed = float(cfg.sim.floor_stiction_switch_speed)
    max_triples = int(runtime_build_capacity.max_triples)
    max_self_triples = int(runtime_build_capacity.max_self_triples)

    def _params_from_genotype(genotype: jnp.ndarray):
        return ctrnn_flat_to_params(
            genotype,
            ctrnn_hidden=ctrnn_hidden,
            ctrnn_grid_hidden=ctrnn_grid_hidden,
            gasnet_mode=gasnet_mode,
            ou_enabled=ou_enabled,
            growth_enabled=True,
            active_comm_filters=active_comm_filters,
        )

    def _evaluate_chunk(
        genotypes: jnp.ndarray,
        template_batch: CliffStaticTemplateBatch,
        candidate_keys: jax.Array,
    ) -> CandidateBatchResult:
        params_batch = jax.vmap(_params_from_genotype)(genotypes)
        n_envs = template_batch.cliff_x.shape[0]
        key_batch = jax.vmap(lambda key: jax.random.split(key, n_envs + 1))(candidate_keys)
        growth_keys = key_batch[:, 0]
        episode_keys = key_batch[:, 1:]

        def _evaluate_candidate(params, growth_key, keys):
            alive_mask, frozen_identity, growth_state, growth_carry = run_growth_phase(
                params,
                growth_key,
                growth_steps=growth_steps,
                canvas_h=canvas_h,
                canvas_w=canvas_w,
                ctrnn_hidden=ctrnn_hidden,
                ctrnn_grid_hidden=ctrnn_grid_hidden,
                gasnet_mode=gasnet_mode,
                ou_enabled=ou_enabled,
                seed_y=seed_y,
                seed_x=seed_x,
                alive_threshold=growth_threshold,
                frontier_identity_only=frontier_identity_only,
                p_update=p_update,
                update_mode=update_mode,
                state_clip_value=state_clip_value,
                connectivity_max_iters=connectivity_max_iters,
                noise_init_scale=noise_init_scale,
                ctrnn_tau_min=ctrnn_tau_min,
                ctrnn_tau_max=ctrnn_tau_max,
                ctrnn_dt=ctrnn_dt,
                active_comm_filters=active_comm_filters,
            )
            robot_morph = jnp.where(
                alive_mask,
                jnp.int32(CONTRACTILE),
                jnp.int32(EMPTY),
            )
            env_core = build_cliff_eval_dynamic_core(
                robot_morph,
                grid_data,
                spawn_x,
                spawn_y,
                physics_substeps=physics_substeps,
                actuator_convergence=actuator_convergence,
                spring_stiffness_scale=spring_stiffness_scale,
                floor_static_friction_const=floor_static_friction_const,
                floor_stiction_stiffness=floor_stiction_stiffness,
                floor_stiction_switch_speed=floor_stiction_switch_speed,
                max_triples=max_triples,
                max_self_triples=max_self_triples,
            )
            initial_nca_state = growth_state if state_transition == "pass_through" else None
            initial_ctrnn_carry = growth_carry if state_transition == "pass_through" else None

            def _on_terrain(
                static_template,
                cliff_x,
                death_plane_y,
                task_mode,
                goal_direction,
                cliff_side,
                goal_distance,
                cliff_distance,
                goal_block_x,
                goal_block_y,
                goal_x,
                goal_object_id,
                cliff_probe_ledge_width,
                cliff_probe_ledge_cell_x,
                cliff_probe_ledge_cell_y,
                ep_key,
            ):
                env = attach_static_template_to_eval_env(
                    env_core,
                    static_template,
                    cliff_x=cliff_x,
                    task_mode=task_mode,
                    goal_direction=goal_direction,
                    cliff_side=cliff_side,
                    goal_distance=goal_distance,
                    cliff_distance=cliff_distance,
                    goal_block_x=goal_block_x,
                    goal_block_y=goal_block_y,
                    goal_x=goal_x,
                    goal_object_id=goal_object_id,
                    cliff_probe_ledge_width=cliff_probe_ledge_width,
                    cliff_probe_ledge_cell_x=cliff_probe_ledge_cell_x,
                    cliff_probe_ledge_cell_y=cliff_probe_ledge_cell_y,
                    death_plane_y=death_plane_y,
                    max_surface_boxels=canvas_h * canvas_w,
                )
                return run_episode(
                    env,
                    params,
                    alive_mask,
                    ep_key,
                    frozen_identity=frozen_identity,
                    initial_nca_state=initial_nca_state,
                    initial_ctrnn_carry=initial_ctrnn_carry,
                    **rollout_kwargs,
                )

            return jax.vmap(_on_terrain)(
                template_batch.static_collider_data,
                template_batch.cliff_x,
                template_batch.death_plane_y,
                template_batch.task_mode,
                template_batch.goal_direction,
                template_batch.cliff_side,
                template_batch.goal_distance,
                template_batch.cliff_distance,
                template_batch.goal_block_x,
                template_batch.goal_block_y,
                template_batch.goal_x,
                template_batch.goal_object_id,
                template_batch.cliff_probe_ledge_width,
                template_batch.cliff_probe_ledge_cell_x,
                template_batch.cliff_probe_ledge_cell_y,
                keys,
            ), jnp.sum(alive_mask.astype(jnp.float32))

        episode_results, alive_counts = jax.vmap(_evaluate_candidate)(params_batch, growth_keys, episode_keys)
        return candidate_batch_fitness(
            episode_results,
            alive_cell_count=alive_counts,
            activation_cost_penalty_scale=activation_cost_penalty_scale,
            soft_loft_penalty_scale=soft_loft_penalty_scale,
        )

    evaluator = jax.jit(_evaluate_chunk)
    _POPULATION_EVALUATOR_CACHE[signature] = evaluator
    return evaluator


def make_grown_com_trace_evaluator(
    cfg: DictConfig,
    runtime_resolution: RuntimeResolution,
    env_assets: GrownEvalAssets,
) -> Callable[[jnp.ndarray, CliffStaticTemplateBatch, jax.Array], tuple[EpisodeResult, EpisodeComTrace, jnp.ndarray]]:
    """Build a grown-body evaluator that returns episode results plus COM-x traces."""
    runtime_build_capacity = env_assets.runtime_build_capacity
    grid_data = env_assets.grid_data
    base_signature = _evaluator_signature(cfg, runtime_resolution)
    signature = (
        "grown_com_trace",
        *base_signature,
        int(runtime_build_capacity.max_triples),
        int(runtime_build_capacity.max_self_triples),
    )
    cached = _COM_TRACE_EVALUATOR_CACHE.get(signature)
    if cached is not None:
        return cached

    (
        ctrnn_hidden,
        ctrnn_grid_hidden,
        gasnet_mode,
        ou_enabled,
        growth_enabled,
        _activation_cost_penalty_scale,
        _soft_loft_penalty_scale,
        rollout_items,
    ) = base_signature
    if not growth_enabled:
        raise ValueError("make_grown_com_trace_evaluator requires body.mode='grown'")

    rollout_kwargs = dict(rollout_items)
    active_comm_filters = tuple(rollout_kwargs["active_comm_filters"])
    growth_cfg = cfg.body.growth
    canvas_h = int(growth_cfg.max_robot_h)
    canvas_w = int(growth_cfg.max_robot_w)
    seed_y, seed_x = resolve_growth_seed_position(
        canvas_h,
        canvas_w,
        str(growth_cfg.get("seed_position", "center")),
    )
    state_transition = str(growth_cfg.get("state_transition", "reset")).strip().lower()
    if state_transition not in {"reset", "pass_through"}:
        raise ValueError(f"Unsupported growth.state_transition={state_transition!r}")
    growth_steps = int(growth_cfg.steps)
    growth_threshold = float(growth_cfg.get("threshold", 0.5))
    frontier_identity_only = bool(growth_cfg.get("frontier_identity_only", True))
    connectivity_max_iters = int(growth_cfg.get("connectivity_max_iters", 10))
    noise_init_scale = float(growth_cfg.get("noise_init_scale", 0.0))
    p_update = float(cfg.nca.p_update)
    update_mode = str(cfg.nca.update_mode)
    state_clip_value = float(cfg.nca.state_clip_value)
    ctrnn_tau_min = float(cfg.nca.ctrnn_tau_min)
    ctrnn_tau_max = float(cfg.nca.ctrnn_tau_max)
    ctrnn_dt = float(cfg.nca.ctrnn_dt)
    spawn_x = resolved_spawn_x(cfg)
    spawn_y = int(cfg.terrain.spawn_y)
    physics_substeps = int(cfg.sim.physics_substeps)
    actuator_convergence = float(cfg.sim.actuator_convergence)
    spring_stiffness_scale = float(cfg.sim.spring_stiffness_scale)
    floor_static_friction_const = float(cfg.sim.floor_static_friction_const)
    floor_stiction_stiffness = float(cfg.sim.floor_stiction_stiffness)
    floor_stiction_switch_speed = float(cfg.sim.floor_stiction_switch_speed)
    max_triples = int(runtime_build_capacity.max_triples)
    max_self_triples = int(runtime_build_capacity.max_self_triples)

    def _params_from_genotype(genotype: jnp.ndarray):
        return ctrnn_flat_to_params(
            genotype,
            ctrnn_hidden=ctrnn_hidden,
            ctrnn_grid_hidden=ctrnn_grid_hidden,
            gasnet_mode=gasnet_mode,
            ou_enabled=ou_enabled,
            growth_enabled=True,
            active_comm_filters=active_comm_filters,
        )

    def _trace_chunk(
        genotypes: jnp.ndarray,
        template_batch: CliffStaticTemplateBatch,
        candidate_keys: jax.Array,
    ) -> tuple[EpisodeResult, EpisodeComTrace, jnp.ndarray]:
        params_batch = jax.vmap(_params_from_genotype)(genotypes)
        n_envs = template_batch.cliff_x.shape[0]
        key_batch = jax.vmap(lambda key: jax.random.split(key, n_envs + 1))(candidate_keys)
        growth_keys = key_batch[:, 0]
        episode_keys = key_batch[:, 1:]

        def _trace_candidate(params, growth_key, keys):
            alive_mask, frozen_identity, growth_state, growth_carry = run_growth_phase(
                params,
                growth_key,
                growth_steps=growth_steps,
                canvas_h=canvas_h,
                canvas_w=canvas_w,
                ctrnn_hidden=ctrnn_hidden,
                ctrnn_grid_hidden=ctrnn_grid_hidden,
                gasnet_mode=gasnet_mode,
                ou_enabled=ou_enabled,
                seed_y=seed_y,
                seed_x=seed_x,
                alive_threshold=growth_threshold,
                frontier_identity_only=frontier_identity_only,
                p_update=p_update,
                update_mode=update_mode,
                state_clip_value=state_clip_value,
                connectivity_max_iters=connectivity_max_iters,
                noise_init_scale=noise_init_scale,
                ctrnn_tau_min=ctrnn_tau_min,
                ctrnn_tau_max=ctrnn_tau_max,
                ctrnn_dt=ctrnn_dt,
                active_comm_filters=active_comm_filters,
            )
            robot_morph = jnp.where(
                alive_mask,
                jnp.int32(CONTRACTILE),
                jnp.int32(EMPTY),
            )
            env_core = build_cliff_eval_dynamic_core(
                robot_morph,
                grid_data,
                spawn_x,
                spawn_y,
                physics_substeps=physics_substeps,
                actuator_convergence=actuator_convergence,
                spring_stiffness_scale=spring_stiffness_scale,
                floor_static_friction_const=floor_static_friction_const,
                floor_stiction_stiffness=floor_stiction_stiffness,
                floor_stiction_switch_speed=floor_stiction_switch_speed,
                max_triples=max_triples,
                max_self_triples=max_self_triples,
            )
            initial_nca_state = growth_state if state_transition == "pass_through" else None
            initial_ctrnn_carry = growth_carry if state_transition == "pass_through" else None

            def _on_terrain(
                static_template,
                cliff_x,
                death_plane_y,
                task_mode,
                goal_direction,
                cliff_side,
                goal_distance,
                cliff_distance,
                goal_block_x,
                goal_block_y,
                goal_x,
                goal_object_id,
                cliff_probe_ledge_width,
                cliff_probe_ledge_cell_x,
                cliff_probe_ledge_cell_y,
                ep_key,
            ):
                env = attach_static_template_to_eval_env(
                    env_core,
                    static_template,
                    cliff_x=cliff_x,
                    task_mode=task_mode,
                    goal_direction=goal_direction,
                    cliff_side=cliff_side,
                    goal_distance=goal_distance,
                    cliff_distance=cliff_distance,
                    goal_block_x=goal_block_x,
                    goal_block_y=goal_block_y,
                    goal_x=goal_x,
                    goal_object_id=goal_object_id,
                    cliff_probe_ledge_width=cliff_probe_ledge_width,
                    cliff_probe_ledge_cell_x=cliff_probe_ledge_cell_x,
                    cliff_probe_ledge_cell_y=cliff_probe_ledge_cell_y,
                    death_plane_y=death_plane_y,
                    max_surface_boxels=canvas_h * canvas_w,
                )
                return run_episode_with_com_trace(
                    env,
                    params,
                    alive_mask,
                    ep_key,
                    frozen_identity=frozen_identity,
                    initial_nca_state=initial_nca_state,
                    initial_ctrnn_carry=initial_ctrnn_carry,
                    **rollout_kwargs,
                )

            episode_results, com_traces = jax.vmap(_on_terrain)(
                template_batch.static_collider_data,
                template_batch.cliff_x,
                template_batch.death_plane_y,
                template_batch.task_mode,
                template_batch.goal_direction,
                template_batch.cliff_side,
                template_batch.goal_distance,
                template_batch.cliff_distance,
                template_batch.goal_block_x,
                template_batch.goal_block_y,
                template_batch.goal_x,
                template_batch.goal_object_id,
                template_batch.cliff_probe_ledge_width,
                template_batch.cliff_probe_ledge_cell_x,
                template_batch.cliff_probe_ledge_cell_y,
                keys,
            )
            return (
                episode_results,
                com_traces,
                jnp.sum(alive_mask.astype(jnp.float32)),
            )

        return jax.vmap(_trace_candidate)(params_batch, growth_keys, episode_keys)

    evaluator = jax.jit(_trace_chunk)
    _COM_TRACE_EVALUATOR_CACHE[signature] = evaluator
    return evaluator


def make_trace_evaluator(
    cfg: DictConfig,
    runtime_resolution: RuntimeResolution,
) -> Callable[[jnp.ndarray, CliffEvalEnvData, jnp.ndarray, jax.Array], tuple[EpisodeResult, object]]:
    """Build the shared jitted trace-capture kernel."""
    signature = _evaluator_signature(cfg, runtime_resolution)
    cached = _TRACE_EVALUATOR_CACHE.get(signature)
    if cached is not None:
        return cached

    (
        ctrnn_hidden,
        ctrnn_grid_hidden,
        gasnet_mode,
        ou_enabled,
        growth_enabled,
        _activation_cost_penalty_scale,
        _soft_loft_penalty_scale,
        rollout_items,
    ) = signature
    if growth_enabled:
        raise ValueError("Trace evaluator cannot be used for grown mode")
    rollout_kwargs = dict(rollout_items)
    active_comm_filters = tuple(rollout_kwargs["active_comm_filters"])

    def _params_from_genotype(genotype: jnp.ndarray):
        return ctrnn_flat_to_params(
            genotype,
            ctrnn_hidden=ctrnn_hidden,
            ctrnn_grid_hidden=ctrnn_grid_hidden,
            gasnet_mode=gasnet_mode,
            ou_enabled=ou_enabled,
            active_comm_filters=active_comm_filters,
        )

    def _trace_chunk(
        genotypes: jnp.ndarray,
        env_batch: CliffEvalEnvData,
        alive_mask: jnp.ndarray,
        candidate_keys: jax.Array,
    ) -> tuple[EpisodeResult, object]:
        params_batch = jax.vmap(_params_from_genotype)(genotypes)
        n_envs = env_batch.cliff_x.shape[0]
        episode_keys = jax.vmap(lambda key: split_key_sequence(key, n_envs)[1])(candidate_keys)

        def _trace_candidate(params, keys):
            return jax.vmap(
                lambda env, ep_key: run_episode_with_trace(
                    env,
                    params,
                    alive_mask,
                    ep_key,
                    **rollout_kwargs,
                )
            )(env_batch, keys)

        return jax.vmap(_trace_candidate)(params_batch, episode_keys)

    evaluator = jax.jit(_trace_chunk)
    _TRACE_EVALUATOR_CACHE[signature] = evaluator
    return evaluator


def make_episode_phase_probe_evaluator(
    cfg: DictConfig,
    runtime_resolution: RuntimeResolution,
) -> Callable[[jnp.ndarray, CliffEvalEnvData, jnp.ndarray, jax.Array], tuple[EpisodeResult, EpisodePhaseProbeTrace]]:
    """Build the shared jitted single-candidate phase-probe kernel."""
    signature = ("phase_probe", *_evaluator_signature(cfg, runtime_resolution))
    cached = _EPISODE_PHASE_PROBE_EVALUATOR_CACHE.get(signature)
    if cached is not None:
        return cached

    (
        ctrnn_hidden,
        ctrnn_grid_hidden,
        gasnet_mode,
        ou_enabled,
        growth_enabled,
        _activation_cost_penalty_scale,
        _soft_loft_penalty_scale,
        rollout_items,
    ) = _evaluator_signature(cfg, runtime_resolution)
    if growth_enabled:
        raise ValueError("Episode phase probe is only supported for fixed-body evaluation")
    rollout_kwargs = dict(rollout_items)
    active_comm_filters = tuple(rollout_kwargs["active_comm_filters"])

    def _params_from_genotype(genotype: jnp.ndarray):
        return ctrnn_flat_to_params(
            genotype,
            ctrnn_hidden=ctrnn_hidden,
            ctrnn_grid_hidden=ctrnn_grid_hidden,
            gasnet_mode=gasnet_mode,
            ou_enabled=ou_enabled,
            active_comm_filters=active_comm_filters,
        )

    def _probe_candidate(
        genotype: jnp.ndarray,
        env_batch: CliffEvalEnvData,
        alive_mask: jnp.ndarray,
        candidate_key: jax.Array,
    ) -> tuple[EpisodeResult, EpisodePhaseProbeTrace]:
        params = _params_from_genotype(genotype)
        n_envs = env_batch.cliff_x.shape[0]
        episode_keys = split_key_sequence(candidate_key, n_envs)[1]
        return jax.vmap(
            lambda env, ep_key: run_episode_with_phase_probe(
                env,
                params,
                alive_mask,
                ep_key,
                **rollout_kwargs,
            )
        )(env_batch, episode_keys)

    evaluator = jax.jit(_probe_candidate)
    _EPISODE_PHASE_PROBE_EVALUATOR_CACHE[signature] = evaluator
    return evaluator


def make_episode_substep_probe_evaluator(
    cfg: DictConfig,
    runtime_resolution: RuntimeResolution,
) -> Callable[
    [jnp.ndarray, CliffEvalEnvData, jnp.ndarray, jax.Array, jnp.ndarray, jnp.ndarray],
    tuple[EpisodeResult, EpisodeSubstepProbeTrace],
]:
    """Build the shared jitted single-episode selected-phase substep-probe kernel."""
    signature = ("substep_probe", *_evaluator_signature(cfg, runtime_resolution))
    cached = _EPISODE_SUBSTEP_PROBE_EVALUATOR_CACHE.get(signature)
    if cached is not None:
        return cached

    (
        ctrnn_hidden,
        ctrnn_grid_hidden,
        gasnet_mode,
        ou_enabled,
        growth_enabled,
        _activation_cost_penalty_scale,
        _soft_loft_penalty_scale,
        rollout_items,
    ) = _evaluator_signature(cfg, runtime_resolution)
    if growth_enabled:
        raise ValueError("Episode substep probe is only supported for fixed-body evaluation")
    rollout_kwargs = dict(rollout_items)
    active_comm_filters = tuple(rollout_kwargs["active_comm_filters"])

    def _params_from_genotype(genotype: jnp.ndarray):
        return ctrnn_flat_to_params(
            genotype,
            ctrnn_hidden=ctrnn_hidden,
            ctrnn_grid_hidden=ctrnn_grid_hidden,
            gasnet_mode=gasnet_mode,
            ou_enabled=ou_enabled,
            active_comm_filters=active_comm_filters,
        )

    def _probe_candidate(
        genotype: jnp.ndarray,
        env_data: CliffEvalEnvData,
        alive_mask: jnp.ndarray,
        episode_key: jax.Array,
        probe_step_index: jnp.ndarray,
        probe_phase_index: jnp.ndarray,
    ) -> tuple[EpisodeResult, EpisodeSubstepProbeTrace]:
        params = _params_from_genotype(genotype)
        return run_episode_with_substep_probe(
            env_data,
            params,
            alive_mask,
            episode_key,
            substep_probe_step_index=probe_step_index,
            substep_probe_phase_index=probe_phase_index,
            **rollout_kwargs,
        )

    evaluator = jax.jit(_probe_candidate)
    _EPISODE_SUBSTEP_PROBE_EVALUATOR_CACHE[signature] = evaluator
    return evaluator


def make_grown_trace_evaluator(
    cfg: DictConfig,
    runtime_resolution: RuntimeResolution,
    env_assets: GrownEvalAssets,
) -> Callable[[jnp.ndarray, CliffStaticTemplateBatch, jax.Array], tuple[EpisodeResult, object, GrownRenderPrep]]:
    """Build the shared jitted grown-body trace-capture kernel."""
    runtime_build_capacity = env_assets.runtime_build_capacity
    grid_data = env_assets.grid_data
    base_signature = _evaluator_signature(cfg, runtime_resolution)
    signature = (
        *base_signature,
        int(runtime_build_capacity.max_triples),
        int(runtime_build_capacity.max_self_triples),
    )
    cached = _TRACE_EVALUATOR_CACHE.get(signature)
    if cached is not None:
        return cached

    (
        ctrnn_hidden,
        ctrnn_grid_hidden,
        gasnet_mode,
        ou_enabled,
        growth_enabled,
        _activation_cost_penalty_scale,
        _soft_loft_penalty_scale,
        rollout_items,
    ) = base_signature
    if not growth_enabled:
        raise ValueError("make_grown_trace_evaluator requires body.mode='grown'")

    rollout_kwargs = dict(rollout_items)
    active_comm_filters = tuple(rollout_kwargs["active_comm_filters"])
    growth_cfg = cfg.body.growth
    canvas_h = int(growth_cfg.max_robot_h)
    canvas_w = int(growth_cfg.max_robot_w)
    seed_y, seed_x = resolve_growth_seed_position(
        canvas_h,
        canvas_w,
        str(growth_cfg.get("seed_position", "center")),
    )
    state_transition = str(growth_cfg.get("state_transition", "reset")).strip().lower()
    if state_transition not in {"reset", "pass_through"}:
        raise ValueError(f"Unsupported growth.state_transition={state_transition!r}")
    growth_steps = int(growth_cfg.steps)
    growth_threshold = float(growth_cfg.get("threshold", 0.5))
    frontier_identity_only = bool(growth_cfg.get("frontier_identity_only", True))
    connectivity_max_iters = int(growth_cfg.get("connectivity_max_iters", 10))
    noise_init_scale = float(growth_cfg.get("noise_init_scale", 0.0))
    p_update = float(cfg.nca.p_update)
    update_mode = str(cfg.nca.update_mode)
    state_clip_value = float(cfg.nca.state_clip_value)
    ctrnn_tau_min = float(cfg.nca.ctrnn_tau_min)
    ctrnn_tau_max = float(cfg.nca.ctrnn_tau_max)
    ctrnn_dt = float(cfg.nca.ctrnn_dt)
    spawn_x = resolved_spawn_x(cfg)
    spawn_y = int(cfg.terrain.spawn_y)
    physics_substeps = int(cfg.sim.physics_substeps)
    actuator_convergence = float(cfg.sim.actuator_convergence)
    spring_stiffness_scale = float(cfg.sim.spring_stiffness_scale)
    floor_static_friction_const = float(cfg.sim.floor_static_friction_const)
    floor_stiction_stiffness = float(cfg.sim.floor_stiction_stiffness)
    floor_stiction_switch_speed = float(cfg.sim.floor_stiction_switch_speed)
    max_triples = int(runtime_build_capacity.max_triples)
    max_self_triples = int(runtime_build_capacity.max_self_triples)

    def _params_from_genotype(genotype: jnp.ndarray):
        return ctrnn_flat_to_params(
            genotype,
            ctrnn_hidden=ctrnn_hidden,
            ctrnn_grid_hidden=ctrnn_grid_hidden,
            gasnet_mode=gasnet_mode,
            ou_enabled=ou_enabled,
            growth_enabled=True,
            active_comm_filters=active_comm_filters,
        )

    def _trace_chunk(
        genotypes: jnp.ndarray,
        template_batch: CliffStaticTemplateBatch,
        candidate_keys: jax.Array,
    ) -> tuple[EpisodeResult, object, GrownRenderPrep]:
        params_batch = jax.vmap(_params_from_genotype)(genotypes)
        n_envs = template_batch.cliff_x.shape[0]
        key_batch = jax.vmap(lambda key: jax.random.split(key, n_envs + 1))(candidate_keys)
        growth_keys = key_batch[:, 0]
        episode_keys = key_batch[:, 1:]

        def _trace_candidate(params, growth_key, keys):
            alive_mask, frozen_identity, growth_state, growth_carry = run_growth_phase(
                params,
                growth_key,
                growth_steps=growth_steps,
                canvas_h=canvas_h,
                canvas_w=canvas_w,
                ctrnn_hidden=ctrnn_hidden,
                ctrnn_grid_hidden=ctrnn_grid_hidden,
                gasnet_mode=gasnet_mode,
                ou_enabled=ou_enabled,
                seed_y=seed_y,
                seed_x=seed_x,
                alive_threshold=growth_threshold,
                frontier_identity_only=frontier_identity_only,
                p_update=p_update,
                update_mode=update_mode,
                state_clip_value=state_clip_value,
                connectivity_max_iters=connectivity_max_iters,
                noise_init_scale=noise_init_scale,
                ctrnn_tau_min=ctrnn_tau_min,
                ctrnn_tau_max=ctrnn_tau_max,
                ctrnn_dt=ctrnn_dt,
                active_comm_filters=active_comm_filters,
            )
            robot_morph = jnp.where(
                alive_mask,
                jnp.int32(CONTRACTILE),
                jnp.int32(EMPTY),
            )
            env_core = build_cliff_eval_dynamic_core(
                robot_morph,
                grid_data,
                spawn_x,
                spawn_y,
                physics_substeps=physics_substeps,
                actuator_convergence=actuator_convergence,
                spring_stiffness_scale=spring_stiffness_scale,
                floor_static_friction_const=floor_static_friction_const,
                floor_stiction_stiffness=floor_stiction_stiffness,
                floor_stiction_switch_speed=floor_stiction_switch_speed,
                max_triples=max_triples,
                max_self_triples=max_self_triples,
            )
            initial_nca_state = growth_state if state_transition == "pass_through" else None
            initial_ctrnn_carry = growth_carry if state_transition == "pass_through" else None

            def _on_terrain(
                static_template,
                cliff_x,
                death_plane_y,
                task_mode,
                goal_direction,
                cliff_side,
                goal_distance,
                cliff_distance,
                goal_block_x,
                goal_block_y,
                goal_x,
                goal_object_id,
                cliff_probe_ledge_width,
                cliff_probe_ledge_cell_x,
                cliff_probe_ledge_cell_y,
                ep_key,
            ):
                env = attach_static_template_to_eval_env(
                    env_core,
                    static_template,
                    cliff_x=cliff_x,
                    task_mode=task_mode,
                    goal_direction=goal_direction,
                    cliff_side=cliff_side,
                    goal_distance=goal_distance,
                    cliff_distance=cliff_distance,
                    goal_block_x=goal_block_x,
                    goal_block_y=goal_block_y,
                    goal_x=goal_x,
                    goal_object_id=goal_object_id,
                    cliff_probe_ledge_width=cliff_probe_ledge_width,
                    cliff_probe_ledge_cell_x=cliff_probe_ledge_cell_x,
                    cliff_probe_ledge_cell_y=cliff_probe_ledge_cell_y,
                    death_plane_y=death_plane_y,
                    max_surface_boxels=canvas_h * canvas_w,
                )
                return run_episode_with_trace(
                    env,
                    params,
                    alive_mask,
                    ep_key,
                    frozen_identity=frozen_identity,
                    initial_nca_state=initial_nca_state,
                    initial_ctrnn_carry=initial_ctrnn_carry,
                    **rollout_kwargs,
                )

            episode_results, traces = jax.vmap(_on_terrain)(
                template_batch.static_collider_data,
                template_batch.cliff_x,
                template_batch.death_plane_y,
                template_batch.task_mode,
                template_batch.goal_direction,
                template_batch.cliff_side,
                template_batch.goal_distance,
                template_batch.cliff_distance,
                template_batch.goal_block_x,
                template_batch.goal_block_y,
                template_batch.goal_x,
                template_batch.goal_object_id,
                template_batch.cliff_probe_ledge_width,
                template_batch.cliff_probe_ledge_cell_x,
                template_batch.cliff_probe_ledge_cell_y,
                keys,
            )
            render_prep = GrownRenderPrep(
                robot_morph=robot_morph,
                spring_init_rest_length=env_core.sim_state.spring_init_rest_length,
            )
            return episode_results, traces, render_prep

        return jax.vmap(_trace_candidate)(params_batch, growth_keys, episode_keys)

    evaluator = jax.jit(_trace_chunk)
    _TRACE_EVALUATOR_CACHE[signature] = evaluator
    return evaluator


def warmup_population_evaluator(
    genotype_dim: int,
    env_datas: list[CliffEnvData | CliffEvalEnvData] | GrownEvalAssets,
    alive_mask: jnp.ndarray,
    cfg: DictConfig,
    *,
    workload: str,
    runtime_resolution: RuntimeResolution,
    static_capacity: StaticColliderCapacity | None = None,
    candidate_count: int | None = None,
) -> dict[int, float]:
    """Compile the population evaluator for the chunk sizes this workload will actually use."""
    descriptor_specs = resolve_qd_archive(cfg).descriptor_specs
    gait_bonus_scale = _gait_spectral_purity_bonus_scale_from_cfg(cfg)
    requires_gait = _requires_gait_com_trace(
        descriptor_specs,
        gait_spectral_purity_bonus_scale=gait_bonus_scale,
    )
    if str(cfg.body.mode).strip().lower() == "grown":
        if not isinstance(env_datas, GrownEvalAssets):
            raise TypeError("Grown warmup expects GrownEvalAssets")
        evaluator = (
            make_grown_com_trace_evaluator(cfg, runtime_resolution, env_datas)
            if requires_gait
            else make_grown_population_evaluator(cfg, runtime_resolution, env_datas)
        )
        timings: dict[int, float] = {}
        batching = runtime_resolution.resolved_batching[workload]
        for bucket_size in planned_warmup_chunk_sizes(candidate_count, batching):
            dummy_genotypes = jnp.zeros((bucket_size, genotype_dim), dtype=jnp.float32)
            _, candidate_keys = split_key_sequence(jax.random.PRNGKey(0), bucket_size)
            started = time.monotonic()
            result = evaluator(dummy_genotypes, env_datas.static_templates, candidate_keys)
            if requires_gait:
                episode_result, com_trace, alive_counts = result
                jax.block_until_ready(episode_result.peak_front_x)
                jax.block_until_ready(com_trace.com_x)
                jax.block_until_ready(alive_counts)
            else:
                jax.block_until_ready(result.fitness)
            timings[int(bucket_size)] = time.monotonic() - started
        return timings

    env_batch = stack_eval_envs(
        env_datas,
        static_capacity=static_capacity or infer_static_collider_capacity(env_datas),
    )
    if requires_gait:
        evaluator = make_com_trace_evaluator(cfg, runtime_resolution)
        timings: dict[int, float] = {}
        batching = runtime_resolution.resolved_batching[workload]
        for bucket_size in planned_warmup_chunk_sizes(candidate_count, batching):
            dummy_genotypes = jnp.zeros((bucket_size, genotype_dim), dtype=jnp.float32)
            _, candidate_keys = split_key_sequence(jax.random.PRNGKey(0), bucket_size)
            started = time.monotonic()
            episode_result, com_trace = evaluator(
                dummy_genotypes,
                env_batch,
                alive_mask,
                candidate_keys,
            )
            jax.block_until_ready(episode_result.peak_front_x)
            jax.block_until_ready(com_trace.com_x)
            timings[int(bucket_size)] = time.monotonic() - started
        return timings
    evaluator = make_population_evaluator(cfg, runtime_resolution)
    return _warmup_evaluator(
        evaluator,
        genotype_dim,
        env_batch,
        alive_mask,
        runtime_resolution.resolved_batching[workload],
        result_attr="fitness",
        candidate_count=candidate_count,
    )


def warmup_trace_evaluator(
    genotype_dim: int,
    env_datas: list[CliffEnvData | CliffEvalEnvData] | GrownEvalAssets,
    alive_mask: jnp.ndarray,
    cfg: DictConfig,
    *,
    workload: str,
    runtime_resolution: RuntimeResolution,
    static_capacity: StaticColliderCapacity | None = None,
    candidate_count: int | None = None,
) -> dict[int, float]:
    """Compile the trace evaluator for the chunk sizes this workload will actually use."""
    if str(cfg.body.mode).strip().lower() == "grown":
        if not isinstance(env_datas, GrownEvalAssets):
            raise TypeError("Grown trace warmup expects GrownEvalAssets")
        evaluator = make_grown_trace_evaluator(
            cfg,
            runtime_resolution,
            env_datas,
        )
        timings: dict[int, float] = {}
        batching = runtime_resolution.resolved_batching[workload]
        for bucket_size in planned_warmup_chunk_sizes(candidate_count, batching):
            dummy_genotypes = jnp.zeros((bucket_size, genotype_dim), dtype=jnp.float32)
            _, candidate_keys = split_key_sequence(jax.random.PRNGKey(0), bucket_size)
            started = time.monotonic()
            episode_result, trace, render_prep = evaluator(
                dummy_genotypes,
                env_datas.static_templates,
                candidate_keys,
            )
            jax.block_until_ready(episode_result.peak_front_x)
            jax.block_until_ready(trace.positions)
            jax.block_until_ready(render_prep.robot_morph)
            timings[int(bucket_size)] = time.monotonic() - started
        return timings

    env_batch = stack_eval_envs(
        env_datas,
        static_capacity=static_capacity or infer_static_collider_capacity(env_datas),
    )
    evaluator = make_trace_evaluator(cfg, runtime_resolution)
    return _warmup_evaluator(
        evaluator,
        genotype_dim,
        env_batch,
        alive_mask,
        runtime_resolution.resolved_batching[workload],
        result_attr="trace",
        candidate_count=candidate_count,
    )


def evaluate_population_chunked(
    genotypes: np.ndarray | jax.Array,
    env_datas: list[CliffEnvData | CliffEvalEnvData] | GrownEvalAssets,
    alive_mask: jnp.ndarray,
    root_keys: jax.Array,
    cfg: DictConfig,
    *,
    workload: str,
    runtime_resolution: RuntimeResolution,
    static_capacity: StaticColliderCapacity | None = None,
    progress_callback: Callable[[ChunkProgress], None] | None = None,
    reduced_host_result: bool = False,
) -> ChunkedPopulationEvalResult:
    """Evaluate a whole candidate population with exact-sized chunking."""
    descriptor_specs = resolve_qd_archive(cfg).descriptor_specs
    gait_bonus_scale = _gait_spectral_purity_bonus_scale_from_cfg(cfg)
    requires_gait = _requires_gait_com_trace(
        descriptor_specs,
        gait_spectral_purity_bonus_scale=gait_bonus_scale,
    )
    if str(cfg.body.mode).strip().lower() == "grown":
        if not isinstance(env_datas, GrownEvalAssets):
            raise TypeError("Grown evaluation expects GrownEvalAssets")
        batching = runtime_resolution.resolved_batching[workload]
        if requires_gait:
            evaluator = make_grown_com_trace_evaluator(
                cfg,
                runtime_resolution,
                env_datas,
            )
            return _chunked_grown_population_eval_with_com_traces(
                evaluator,
                genotypes,
                env_assets=env_datas,
                root_keys=root_keys,
                workload=workload,
                batching=batching,
                descriptor_specs=descriptor_specs,
                activation_cost_penalty_scale=float(
                    cfg.train.get("activation_cost_penalty_scale", 0.0)
                ),
                soft_loft_penalty_scale=_soft_loft_penalty_scale_from_cfg(cfg),
                gait_spectral_purity_bonus_scale=gait_bonus_scale,
                progress_callback=progress_callback,
                reduced_host_result=reduced_host_result,
            )
        evaluator = make_grown_population_evaluator(
            cfg,
            runtime_resolution,
            env_datas,
        )
        return _chunked_grown_population_eval(
            evaluator,
            genotypes,
            env_datas,
            root_keys,
            workload=workload,
            batching=batching,
            descriptor_specs=descriptor_specs,
            progress_callback=progress_callback,
            reduced_host_result=reduced_host_result,
        )

    env_batch = stack_eval_envs(env_datas, static_capacity=static_capacity)
    batching = runtime_resolution.resolved_batching[workload]
    if requires_gait:
        evaluator = make_com_trace_evaluator(cfg, runtime_resolution)
        return _chunked_population_eval_with_com_traces(
            evaluator,
            genotypes,
            env_batch,
            alive_mask,
            root_keys,
            workload=workload,
            batching=batching,
            descriptor_specs=descriptor_specs,
            activation_cost_penalty_scale=float(
                cfg.train.get("activation_cost_penalty_scale", 0.0)
            ),
            soft_loft_penalty_scale=_soft_loft_penalty_scale_from_cfg(cfg),
            gait_spectral_purity_bonus_scale=gait_bonus_scale,
            progress_callback=progress_callback,
            reduced_host_result=reduced_host_result,
        )
    evaluator = make_population_evaluator(cfg, runtime_resolution)
    return _chunked_population_eval(
        evaluator,
        genotypes,
        env_batch,
        alive_mask,
        root_keys,
        workload=workload,
        batching=batching,
        descriptor_specs=descriptor_specs,
        progress_callback=progress_callback,
        reduced_host_result=reduced_host_result,
    )


def trace_population_chunked(
    genotypes: np.ndarray,
    env_datas: list[CliffEnvData | CliffEvalEnvData] | GrownEvalAssets,
    alive_mask: jnp.ndarray,
    root_keys: jax.Array,
    cfg: DictConfig,
    *,
    workload: str,
    runtime_resolution: RuntimeResolution,
    static_capacity: StaticColliderCapacity | None = None,
    progress_callback: Callable[[ChunkProgress], None] | None = None,
) -> ChunkedTraceEvalResult:
    """Capture rollout traces for a candidate batch with the same chunking model."""
    if str(cfg.body.mode).strip().lower() == "grown":
        if not isinstance(env_datas, GrownEvalAssets):
            raise TypeError("Grown trace evaluation expects GrownEvalAssets")
        evaluator = make_grown_trace_evaluator(
            cfg,
            runtime_resolution,
            env_datas,
        )
        batching = runtime_resolution.resolved_batching[workload]
        return _chunked_grown_trace_eval(
            evaluator,
            genotypes,
            env_assets=env_datas,
            root_keys=root_keys,
            workload=workload,
            batching=batching,
            activation_cost_penalty_scale=float(
                cfg.train.get("activation_cost_penalty_scale", 0.0)
            ),
            soft_loft_penalty_scale=_soft_loft_penalty_scale_from_cfg(cfg),
            progress_callback=progress_callback,
        )
    env_batch = stack_eval_envs(env_datas, static_capacity=static_capacity)
    evaluator = make_trace_evaluator(cfg, runtime_resolution)
    batching = runtime_resolution.resolved_batching[workload]

    start = time.monotonic()
    activation_cost_penalty_scale = float(cfg.train.get("activation_cost_penalty_scale", 0.0))
    soft_loft_penalty_scale = _soft_loft_penalty_scale_from_cfg(cfg)
    total_candidates = int(genotypes.shape[0])
    total_chunks = len(_plan_candidate_chunk_sizes(total_candidates, batching))
    bucket_counts: dict[int, int] = {}
    episode_chunks = []
    trace_chunks = []
    partial_fitnesses: list[np.ndarray] = []

    for chunk_index, (chunk_genotypes, chunk_keys, bucket_size) in enumerate(
        _iter_candidate_chunks(genotypes, root_keys, batching), start=1
    ):
        bucket_counts[bucket_size] = bucket_counts.get(bucket_size, 0) + 1

        episode_chunk, trace_chunk = evaluator(
            jnp.asarray(chunk_genotypes),
            env_batch,
            alive_mask,
            chunk_keys,
        )
        jax.block_until_ready(episode_chunk.peak_front_x)
        trimmed_episodes = jax.tree_util.tree_map(
            lambda array: np.asarray(array),
            episode_chunk,
        )
        trimmed_traces = jax.tree_util.tree_map(
            lambda array: np.asarray(array),
            trace_chunk,
        )
        episode_chunks.append(trimmed_episodes)
        trace_chunks.append(trimmed_traces)

        batch_metrics = candidate_batch_fitness(
            trimmed_episodes,
            activation_cost_penalty_scale=activation_cost_penalty_scale,
            soft_loft_penalty_scale=soft_loft_penalty_scale,
        )
        partial_fitnesses.append(np.asarray(batch_metrics.fitness))
        if progress_callback is not None:
            progress_callback(
                ChunkProgress(
                    workload=workload,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    evaluated_candidates=sum(len(values) for values in partial_fitnesses),
                    total_candidates=total_candidates,
                    bucket_size=bucket_size,
                    elapsed_s=time.monotonic() - start,
                    best_fitness=float(np.max(np.concatenate(partial_fitnesses, axis=0))),
                    mean_fitness=float(np.mean(np.concatenate(partial_fitnesses, axis=0))),
                )
            )

    combined_episodes = _concatenate_pytrees(episode_chunks)
    combined_traces = _concatenate_pytrees(trace_chunks)
    return ChunkedTraceEvalResult(
        episode_results=combined_episodes,
        traces=combined_traces,
        render_preps=None,
        stats=ChunkedEvalStats(
            workload=workload,
            elapsed_s=time.monotonic() - start,
            candidate_count=total_candidates,
            chunk_count=total_chunks,
            bucket_counts=bucket_counts,
        ),
    )


def probe_candidate_episode_phases(
    genotype: np.ndarray | jax.Array,
    env_datas: list[CliffEnvData | CliffEvalEnvData],
    alive_mask: jnp.ndarray,
    candidate_key: jax.Array,
    cfg: DictConfig,
    *,
    runtime_resolution: RuntimeResolution,
    static_capacity: StaticColliderCapacity | None = None,
) -> CandidateEpisodePhaseProbeResult:
    """Run a single candidate through the generation rollout with phase-level capture."""
    if str(cfg.body.mode).strip().lower() == "grown":
        raise ValueError("Episode phase probe does not support body.mode='grown'")
    env_batch = stack_eval_envs(
        env_datas,
        static_capacity=static_capacity or infer_static_collider_capacity(env_datas),
    )
    evaluator = make_episode_phase_probe_evaluator(cfg, runtime_resolution)
    episode_results, phase_trace = evaluator(
        jnp.asarray(genotype, dtype=jnp.float32),
        env_batch,
        alive_mask,
        jnp.asarray(candidate_key),
    )
    jax.block_until_ready(episode_results.peak_front_x)
    jax.block_until_ready(phase_trace.phase_post_positions)
    return CandidateEpisodePhaseProbeResult(
        episode_results=jax.tree_util.tree_map(lambda array: np.asarray(array), episode_results),
        phase_trace=jax.tree_util.tree_map(lambda array: np.asarray(array), phase_trace),
    )


def probe_candidate_episode_substeps(
    genotype: np.ndarray | jax.Array,
    env_datas: list[CliffEnvData | CliffEvalEnvData],
    alive_mask: jnp.ndarray,
    candidate_key: jax.Array,
    cfg: DictConfig,
    *,
    selected_episode_index: int,
    selected_step_index: int,
    selected_phase_index: int,
    runtime_resolution: RuntimeResolution,
    static_capacity: StaticColliderCapacity | None = None,
) -> CandidateEpisodeSubstepProbeResult:
    """Run one selected terrain episode through the generation rollout with substep capture."""
    if str(cfg.body.mode).strip().lower() == "grown":
        raise ValueError("Episode substep probe does not support body.mode='grown'")
    episode_index = int(selected_episode_index)
    if episode_index < 0 or episode_index >= len(env_datas):
        raise IndexError(
            f"selected_episode_index={episode_index} is out of range for {len(env_datas)} envs"
        )
    selected_env = env_datas[episode_index]
    env_batch = stack_eval_envs(
        [selected_env],
        static_capacity=static_capacity or infer_static_collider_capacity(env_datas),
    )
    episode_keys = split_key_sequence(jnp.asarray(candidate_key), len(env_datas))[1]
    evaluator = make_episode_substep_probe_evaluator(cfg, runtime_resolution)
    episode_result, substep_trace = evaluator(
        jnp.asarray(genotype, dtype=jnp.float32),
        jax.tree_util.tree_map(lambda array: array[0], env_batch),
        alive_mask,
        episode_keys[episode_index],
        jnp.int32(selected_step_index),
        jnp.int32(selected_phase_index),
    )
    jax.block_until_ready(episode_result.peak_front_x)
    jax.block_until_ready(substep_trace.substep_final_positions)
    return CandidateEpisodeSubstepProbeResult(
        episode_result=jax.tree_util.tree_map(lambda array: np.asarray(array), episode_result),
        substep_trace=jax.tree_util.tree_map(lambda array: np.asarray(array), substep_trace),
    )


def _materialize_population_summary(
    batch_result: CandidateBatchResult,
    *,
    descriptor_specs: tuple[QDDescriptorSpec, ...],
) -> tuple[PopulationBatchSummary, dict[str, np.ndarray]]:
    metric_attrs = tuple(
        dict.fromkeys(
            tuple(CANDIDATE_SCALAR_METRIC_ATTRS)
            + tuple(spec.metric_attr for spec in descriptor_specs)
        )
    )
    selected = {
        "fitness": batch_result.fitness,
        "goal_hit_count": jnp.sum(batch_result.episode_goal_touched.astype(jnp.float32), axis=-1),
        "mirrored_pair_count": batch_result.mirrored_pair_count.astype(jnp.float32),
        "lower_ledge_penalty": batch_result.lower_ledge_penalty.astype(jnp.float32),
        "episode_fitnesses": batch_result.episode_fitnesses.astype(jnp.float32),
        "episode_goal_touched": batch_result.episode_goal_touched.astype(jnp.bool_),
        "episode_goal_distances": batch_result.episode_goal_distances.astype(jnp.int32),
        "episode_cliff_distances": batch_result.episode_cliff_distances.astype(jnp.int32),
        "episode_times_to_goal": batch_result.episode_times_to_goal.astype(jnp.float32),
        "episode_first_significant_directions": batch_result.episode_first_significant_directions.astype(jnp.int32),
        "episode_fell": batch_result.episode_fell,
        "episode_left_edge": batch_result.episode_left_edge,
        "deformation_channel_max": batch_result.deformation_channel_max,
        "controller_hidden_output_max_abs": batch_result.controller_hidden_output_max_abs,
        "action_output_max_abs": batch_result.action_output_max_abs,
        "descriptor_metrics": tuple(getattr(batch_result, attr) for attr in metric_attrs),
    }
    host_selected = jax.device_get(selected)
    descriptor_columns = {
        attr: np.asarray(value, dtype=np.float32)
        for attr, value in zip(
            metric_attrs,
            host_selected["descriptor_metrics"],
            strict=False,
        )
    }
    return PopulationBatchSummary(
        fitness=np.asarray(host_selected["fitness"], dtype=np.float32),
        goal_hit_count=np.asarray(host_selected["goal_hit_count"], dtype=np.float32),
        mirrored_pair_count=np.asarray(host_selected["mirrored_pair_count"], dtype=np.float32),
        lower_ledge_penalty=np.asarray(host_selected["lower_ledge_penalty"], dtype=np.float32),
        episode_fell=np.asarray(host_selected["episode_fell"], dtype=bool),
        episode_left_edge=np.asarray(host_selected["episode_left_edge"], dtype=bool),
        deformation_channel_max=np.asarray(host_selected["deformation_channel_max"], dtype=np.float32),
        controller_hidden_output_max_abs=np.asarray(host_selected["controller_hidden_output_max_abs"], dtype=np.float32),
        action_output_max_abs=np.asarray(host_selected["action_output_max_abs"], dtype=np.float32),
        descriptor_columns=descriptor_columns,
        episode_fitnesses=np.asarray(host_selected["episode_fitnesses"], dtype=np.float32),
        episode_goal_touched=np.asarray(host_selected["episode_goal_touched"], dtype=bool),
        episode_goal_distances=np.asarray(host_selected["episode_goal_distances"], dtype=np.int32),
        episode_cliff_distances=np.asarray(host_selected["episode_cliff_distances"], dtype=np.int32),
        episode_times_to_goal=np.asarray(host_selected["episode_times_to_goal"], dtype=np.float32),
        episode_first_significant_directions=np.asarray(
            host_selected["episode_first_significant_directions"],
            dtype=np.int32,
        ),
    ), descriptor_columns


def _combine_population_summaries(
    chunk_summaries: list[tuple[PopulationBatchSummary, dict[str, np.ndarray]]],
    *,
    descriptor_specs: tuple[QDDescriptorSpec, ...],
) -> tuple[PopulationBatchSummary, np.ndarray]:
    def _concat_optional(attr: str):
        first = getattr(chunk_summaries[0][0], attr)
        if first is None:
            return None
        return np.concatenate(
            [getattr(chunk[0], attr) for chunk in chunk_summaries],
            axis=0,
        )

    summary = PopulationBatchSummary(
        fitness=np.concatenate([chunk[0].fitness for chunk in chunk_summaries], axis=0),
        goal_hit_count=np.concatenate([chunk[0].goal_hit_count for chunk in chunk_summaries], axis=0),
        mirrored_pair_count=np.concatenate([chunk[0].mirrored_pair_count for chunk in chunk_summaries], axis=0),
        lower_ledge_penalty=np.concatenate(
            [chunk[0].lower_ledge_penalty for chunk in chunk_summaries],
            axis=0,
        ),
        episode_fell=np.concatenate([chunk[0].episode_fell for chunk in chunk_summaries], axis=0),
        episode_left_edge=np.concatenate(
            [chunk[0].episode_left_edge for chunk in chunk_summaries],
            axis=0,
        ),
        deformation_channel_max=np.concatenate(
            [chunk[0].deformation_channel_max for chunk in chunk_summaries],
            axis=0,
        ),
        controller_hidden_output_max_abs=np.concatenate(
            [chunk[0].controller_hidden_output_max_abs for chunk in chunk_summaries],
            axis=0,
        ),
        action_output_max_abs=np.concatenate(
            [chunk[0].action_output_max_abs for chunk in chunk_summaries],
            axis=0,
        ),
        descriptor_columns={
            attr: np.concatenate([chunk[1][attr] for chunk in chunk_summaries], axis=0)
            for attr in chunk_summaries[0][1]
        },
        episode_fitnesses=_concat_optional("episode_fitnesses"),
        episode_goal_touched=_concat_optional("episode_goal_touched"),
        episode_goal_distances=_concat_optional("episode_goal_distances"),
        episode_cliff_distances=_concat_optional("episode_cliff_distances"),
        episode_times_to_goal=_concat_optional("episode_times_to_goal"),
        episode_first_significant_directions=_concat_optional("episode_first_significant_directions"),
    )
    descriptor_columns = {
        attr: np.concatenate([chunk[1][attr] for chunk in chunk_summaries], axis=0)
        for attr in dict.fromkeys(spec.metric_attr for spec in descriptor_specs)
    }
    descriptors = np.stack(
        [descriptor_columns[spec.metric_attr] for spec in descriptor_specs],
        axis=1,
    )
    return summary, descriptors


def _chunked_population_eval(
    evaluator: Callable[[jnp.ndarray, CliffEvalEnvData, jnp.ndarray, jax.Array], CandidateBatchResult],
    genotypes: np.ndarray | jax.Array,
    env_batch: CliffEvalEnvData,
    alive_mask: jnp.ndarray,
    root_keys: jax.Array,
    *,
    workload: str,
    batching: WorkloadBatching,
    descriptor_specs: tuple[QDDescriptorSpec, ...],
    progress_callback: Callable[[ChunkProgress], None] | None = None,
    reduced_host_result: bool = False,
) -> ChunkedPopulationEvalResult:
    start = time.monotonic()
    total_candidates = int(genotypes.shape[0])
    total_chunks = len(_plan_candidate_chunk_sizes(total_candidates, batching))
    bucket_counts: dict[int, int] = {}
    chunk_results = []
    partial_fitnesses: list[np.ndarray] = []

    for chunk_index, (chunk_genotypes, chunk_keys, bucket_size) in enumerate(
        _iter_candidate_chunks(genotypes, root_keys, batching), start=1
    ):
        bucket_counts[bucket_size] = bucket_counts.get(bucket_size, 0) + 1

        chunk_result = evaluator(
            jnp.asarray(chunk_genotypes),
            env_batch,
            alive_mask,
            chunk_keys,
        )
        if reduced_host_result:
            summary_result = _materialize_population_summary(
                chunk_result,
                descriptor_specs=descriptor_specs,
            )
            chunk_results.append(summary_result)
            partial_fitnesses.append(np.asarray(summary_result[0].fitness))
        else:
            jax.block_until_ready(chunk_result.fitness)
            trimmed_result = jax.tree_util.tree_map(
                lambda array: np.asarray(array),
                chunk_result,
            )
            chunk_results.append(trimmed_result)
            partial_fitnesses.append(np.asarray(trimmed_result.fitness))

        if progress_callback is not None:
            concatenated = np.concatenate(partial_fitnesses, axis=0)
            progress_callback(
                ChunkProgress(
                    workload=workload,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    evaluated_candidates=int(concatenated.shape[0]),
                    total_candidates=total_candidates,
                    bucket_size=bucket_size,
                    elapsed_s=time.monotonic() - start,
                    best_fitness=float(np.max(concatenated)),
                    mean_fitness=float(np.mean(concatenated)),
                )
            )

    if reduced_host_result:
        combined, descriptors = _combine_population_summaries(
            chunk_results,
            descriptor_specs=descriptor_specs,
        )
    else:
        combined = _concatenate_pytrees(chunk_results)
        descriptors = build_descriptor_matrix(combined, descriptor_specs)
    return ChunkedPopulationEvalResult(
        batch_result=combined,
        fitnesses=np.asarray(combined.fitness),
        descriptors=descriptors,
        stats=ChunkedEvalStats(
            workload=workload,
            elapsed_s=time.monotonic() - start,
            candidate_count=total_candidates,
            chunk_count=total_chunks,
            bucket_counts=bucket_counts,
        ),
    )


def _chunked_grown_population_eval(
    evaluator: Callable[[jnp.ndarray, CliffStaticTemplateBatch, jax.Array], CandidateBatchResult],
    genotypes: np.ndarray | jax.Array,
    env_assets: GrownEvalAssets,
    root_keys: jax.Array,
    *,
    workload: str,
    batching: WorkloadBatching,
    descriptor_specs: tuple[QDDescriptorSpec, ...],
    progress_callback: Callable[[ChunkProgress], None] | None = None,
    reduced_host_result: bool = False,
) -> ChunkedPopulationEvalResult:
    start = time.monotonic()
    total_candidates = int(genotypes.shape[0])
    total_chunks = len(_plan_candidate_chunk_sizes(total_candidates, batching))
    bucket_counts: dict[int, int] = {}
    chunk_results = []
    partial_fitnesses: list[np.ndarray] = []
    n_envs = int(env_assets.static_templates.cliff_x.shape[0])

    for chunk_index, (chunk_genotypes, chunk_keys, bucket_size) in enumerate(
        _iter_candidate_chunks(genotypes, root_keys, batching), start=1
    ):
        bucket_counts[bucket_size] = bucket_counts.get(bucket_size, 0) + 1
        chunk_result = evaluator(
            jnp.asarray(chunk_genotypes),
            env_assets.static_templates,
            chunk_keys,
        )
        if reduced_host_result:
            summary_result = _materialize_population_summary(
                chunk_result,
                descriptor_specs=descriptor_specs,
            )
            chunk_results.append(summary_result)
            partial_fitnesses.append(np.asarray(summary_result[0].fitness))
        else:
            jax.block_until_ready(chunk_result.fitness)
            trimmed_result = jax.tree_util.tree_map(lambda array: np.asarray(array), chunk_result)
            chunk_results.append(trimmed_result)
            partial_fitnesses.append(np.asarray(trimmed_result.fitness))

        if progress_callback is not None:
            concatenated = np.concatenate(partial_fitnesses, axis=0)
            progress_callback(
                ChunkProgress(
                    workload=workload,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    evaluated_candidates=int(concatenated.shape[0]),
                    total_candidates=total_candidates,
                    bucket_size=bucket_size,
                    elapsed_s=time.monotonic() - start,
                    best_fitness=float(np.max(concatenated)),
                    mean_fitness=float(np.mean(concatenated)),
                )
            )

    if reduced_host_result:
        combined, descriptors = _combine_population_summaries(
            chunk_results,
            descriptor_specs=descriptor_specs,
        )
    else:
        combined = _concatenate_pytrees(chunk_results)
        descriptors = build_descriptor_matrix(combined, descriptor_specs)
    return ChunkedPopulationEvalResult(
        batch_result=combined,
        fitnesses=np.asarray(combined.fitness),
        descriptors=descriptors,
        stats=ChunkedEvalStats(
            workload=workload,
            elapsed_s=time.monotonic() - start,
            candidate_count=total_candidates,
            chunk_count=total_chunks,
            bucket_counts=bucket_counts,
        ),
    )


def _chunked_population_eval_with_com_traces(
    evaluator: Callable[[jnp.ndarray, CliffEvalEnvData, jnp.ndarray, jax.Array], tuple[EpisodeResult, EpisodeComTrace]],
    genotypes: np.ndarray | jax.Array,
    env_batch: CliffEvalEnvData,
    alive_mask: jnp.ndarray,
    root_keys: jax.Array,
    *,
    workload: str,
    batching: WorkloadBatching,
    descriptor_specs: tuple[QDDescriptorSpec, ...],
    activation_cost_penalty_scale: float = 0.0,
    soft_loft_penalty_scale: float = 0.0,
    gait_spectral_purity_bonus_scale: float = 0.0,
    progress_callback: Callable[[ChunkProgress], None] | None = None,
    reduced_host_result: bool = False,
) -> ChunkedPopulationEvalResult:
    start = time.monotonic()
    total_candidates = int(genotypes.shape[0])
    total_chunks = len(_plan_candidate_chunk_sizes(total_candidates, batching))
    bucket_counts: dict[int, int] = {}
    chunk_results = []
    partial_fitnesses: list[np.ndarray] = []
    alive_count = jnp.sum(alive_mask.astype(jnp.float32))

    for chunk_index, (chunk_genotypes, chunk_keys, bucket_size) in enumerate(
        _iter_candidate_chunks(genotypes, root_keys, batching), start=1
    ):
        bucket_counts[bucket_size] = bucket_counts.get(bucket_size, 0) + 1
        episode_chunk, com_trace_chunk = evaluator(
            jnp.asarray(chunk_genotypes),
            env_batch,
            alive_mask,
            chunk_keys,
        )
        batch_metrics = candidate_batch_fitness(
            episode_chunk,
            alive_cell_count=jnp.full(
                (chunk_genotypes.shape[0],),
                alive_count,
                dtype=jnp.float32,
            ),
            activation_cost_penalty_scale=activation_cost_penalty_scale,
            soft_loft_penalty_scale=soft_loft_penalty_scale,
        )
        jax.block_until_ready(batch_metrics.fitness)
        jax.block_until_ready(com_trace_chunk.com_x)
        batch_metrics_host = jax.tree_util.tree_map(
            lambda array: np.asarray(array),
            batch_metrics,
        )
        com_trace_host = jax.tree_util.tree_map(
            lambda array: np.asarray(array),
            com_trace_chunk,
        )
        gait_values = compute_candidate_gait_spectral_purity(
            com_trace_host.com_x,
            np.asarray(batch_metrics_host.episode_steps_alive, dtype=np.int32),
        )

        if reduced_host_result:
            summary, descriptor_columns = _materialize_population_summary(
                batch_metrics,
                descriptor_specs=descriptor_specs,
            )
            summary = _augment_batch_result_with_gait_host(
                replace(summary, descriptor_columns=descriptor_columns),
                gait_values,
                gait_spectral_purity_bonus_scale=gait_spectral_purity_bonus_scale,
            )
            chunk_results.append((summary, dict(summary.descriptor_columns or {})))
            partial_fitnesses.append(np.asarray(summary.fitness))
        else:
            trimmed_result = _augment_batch_result_with_gait_host(
                batch_metrics_host,
                gait_values,
                gait_spectral_purity_bonus_scale=gait_spectral_purity_bonus_scale,
            )
            chunk_results.append(trimmed_result)
            partial_fitnesses.append(np.asarray(trimmed_result.fitness))

        if progress_callback is not None:
            concatenated = np.concatenate(partial_fitnesses, axis=0)
            progress_callback(
                ChunkProgress(
                    workload=workload,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    evaluated_candidates=int(concatenated.shape[0]),
                    total_candidates=total_candidates,
                    bucket_size=bucket_size,
                    elapsed_s=time.monotonic() - start,
                    best_fitness=float(np.max(concatenated)),
                    mean_fitness=float(np.mean(concatenated)),
                )
            )

    if reduced_host_result:
        combined, descriptors = _combine_population_summaries(
            chunk_results,
            descriptor_specs=descriptor_specs,
        )
    else:
        combined = _concatenate_pytrees(chunk_results)
        descriptors = build_descriptor_matrix(combined, descriptor_specs)
    return ChunkedPopulationEvalResult(
        batch_result=combined,
        fitnesses=np.asarray(combined.fitness),
        descriptors=descriptors,
        stats=ChunkedEvalStats(
            workload=workload,
            elapsed_s=time.monotonic() - start,
            candidate_count=total_candidates,
            chunk_count=total_chunks,
            bucket_counts=bucket_counts,
        ),
    )


def _chunked_grown_population_eval_with_com_traces(
    evaluator: Callable[[jnp.ndarray, CliffStaticTemplateBatch, jax.Array], tuple[EpisodeResult, EpisodeComTrace, jnp.ndarray]],
    genotypes: np.ndarray | jax.Array,
    env_assets: GrownEvalAssets,
    root_keys: jax.Array,
    *,
    workload: str,
    batching: WorkloadBatching,
    descriptor_specs: tuple[QDDescriptorSpec, ...],
    activation_cost_penalty_scale: float = 0.0,
    soft_loft_penalty_scale: float = 0.0,
    gait_spectral_purity_bonus_scale: float = 0.0,
    progress_callback: Callable[[ChunkProgress], None] | None = None,
    reduced_host_result: bool = False,
) -> ChunkedPopulationEvalResult:
    start = time.monotonic()
    total_candidates = int(genotypes.shape[0])
    total_chunks = len(_plan_candidate_chunk_sizes(total_candidates, batching))
    bucket_counts: dict[int, int] = {}
    chunk_results = []
    partial_fitnesses: list[np.ndarray] = []

    for chunk_index, (chunk_genotypes, chunk_keys, bucket_size) in enumerate(
        _iter_candidate_chunks(genotypes, root_keys, batching), start=1
    ):
        bucket_counts[bucket_size] = bucket_counts.get(bucket_size, 0) + 1
        episode_chunk, com_trace_chunk, alive_counts = evaluator(
            jnp.asarray(chunk_genotypes),
            env_assets.static_templates,
            chunk_keys,
        )
        batch_metrics = candidate_batch_fitness(
            episode_chunk,
            alive_cell_count=alive_counts,
            activation_cost_penalty_scale=activation_cost_penalty_scale,
            soft_loft_penalty_scale=soft_loft_penalty_scale,
        )
        jax.block_until_ready(batch_metrics.fitness)
        jax.block_until_ready(com_trace_chunk.com_x)
        batch_metrics_host = jax.tree_util.tree_map(
            lambda array: np.asarray(array),
            batch_metrics,
        )
        com_trace_host = jax.tree_util.tree_map(
            lambda array: np.asarray(array),
            com_trace_chunk,
        )
        gait_values = compute_candidate_gait_spectral_purity(
            com_trace_host.com_x,
            np.asarray(batch_metrics_host.episode_steps_alive, dtype=np.int32),
        )

        if reduced_host_result:
            summary, descriptor_columns = _materialize_population_summary(
                batch_metrics,
                descriptor_specs=descriptor_specs,
            )
            summary = _augment_batch_result_with_gait_host(
                replace(summary, descriptor_columns=descriptor_columns),
                gait_values,
                gait_spectral_purity_bonus_scale=gait_spectral_purity_bonus_scale,
            )
            chunk_results.append((summary, dict(summary.descriptor_columns or {})))
            partial_fitnesses.append(np.asarray(summary.fitness))
        else:
            trimmed_result = _augment_batch_result_with_gait_host(
                batch_metrics_host,
                gait_values,
                gait_spectral_purity_bonus_scale=gait_spectral_purity_bonus_scale,
            )
            chunk_results.append(trimmed_result)
            partial_fitnesses.append(np.asarray(trimmed_result.fitness))

        if progress_callback is not None:
            concatenated = np.concatenate(partial_fitnesses, axis=0)
            progress_callback(
                ChunkProgress(
                    workload=workload,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    evaluated_candidates=int(concatenated.shape[0]),
                    total_candidates=total_candidates,
                    bucket_size=bucket_size,
                    elapsed_s=time.monotonic() - start,
                    best_fitness=float(np.max(concatenated)),
                    mean_fitness=float(np.mean(concatenated)),
                )
            )

    if reduced_host_result:
        combined, descriptors = _combine_population_summaries(
            chunk_results,
            descriptor_specs=descriptor_specs,
        )
    else:
        combined = _concatenate_pytrees(chunk_results)
        descriptors = build_descriptor_matrix(combined, descriptor_specs)
    return ChunkedPopulationEvalResult(
        batch_result=combined,
        fitnesses=np.asarray(combined.fitness),
        descriptors=descriptors,
        stats=ChunkedEvalStats(
            workload=workload,
            elapsed_s=time.monotonic() - start,
            candidate_count=total_candidates,
            chunk_count=total_chunks,
            bucket_counts=bucket_counts,
        ),
    )


def _chunked_grown_trace_eval(
    evaluator: Callable[[jnp.ndarray, CliffStaticTemplateBatch, jax.Array], tuple[EpisodeResult, object, GrownRenderPrep]],
    genotypes: np.ndarray | jax.Array,
    env_assets: GrownEvalAssets,
    root_keys: jax.Array,
    *,
    workload: str,
    batching: WorkloadBatching,
    activation_cost_penalty_scale: float = 0.0,
    soft_loft_penalty_scale: float = 0.0,
    progress_callback: Callable[[ChunkProgress], None] | None = None,
) -> ChunkedTraceEvalResult:
    start = time.monotonic()
    total_candidates = int(genotypes.shape[0])
    total_chunks = len(_plan_candidate_chunk_sizes(total_candidates, batching))
    bucket_counts: dict[int, int] = {}
    episode_chunks = []
    trace_chunks = []
    render_prep_chunks = []
    partial_fitnesses: list[np.ndarray] = []

    for chunk_index, (chunk_genotypes, chunk_keys, bucket_size) in enumerate(
        _iter_candidate_chunks(genotypes, root_keys, batching), start=1
    ):
        bucket_counts[bucket_size] = bucket_counts.get(bucket_size, 0) + 1
        episode_chunk, trace_chunk, render_prep_chunk = evaluator(
            jnp.asarray(chunk_genotypes),
            env_assets.static_templates,
            chunk_keys,
        )
        jax.block_until_ready(episode_chunk.peak_front_x)
        trimmed_episodes = jax.tree_util.tree_map(
            lambda array: np.asarray(array),
            episode_chunk,
        )
        trimmed_traces = jax.tree_util.tree_map(
            lambda array: np.asarray(array),
            trace_chunk,
        )
        trimmed_render_preps = jax.tree_util.tree_map(
            lambda array: np.asarray(array),
            render_prep_chunk,
        )
        episode_chunks.append(trimmed_episodes)
        trace_chunks.append(trimmed_traces)
        render_prep_chunks.append(trimmed_render_preps)

        batch_metrics = candidate_batch_fitness(
            trimmed_episodes,
            activation_cost_penalty_scale=activation_cost_penalty_scale,
            soft_loft_penalty_scale=soft_loft_penalty_scale,
        )
        partial_fitnesses.append(np.asarray(batch_metrics.fitness))
        if progress_callback is not None:
            progress_callback(
                ChunkProgress(
                    workload=workload,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    evaluated_candidates=sum(len(values) for values in partial_fitnesses),
                    total_candidates=total_candidates,
                    bucket_size=bucket_size,
                    elapsed_s=time.monotonic() - start,
                    best_fitness=float(np.max(np.concatenate(partial_fitnesses, axis=0))),
                    mean_fitness=float(np.mean(np.concatenate(partial_fitnesses, axis=0))),
                )
            )

    combined_episodes = _concatenate_pytrees(episode_chunks)
    combined_traces = _concatenate_pytrees(trace_chunks)
    combined_render_preps = _concatenate_pytrees(render_prep_chunks)
    return ChunkedTraceEvalResult(
        episode_results=combined_episodes,
        traces=combined_traces,
        render_preps=combined_render_preps,
        stats=ChunkedEvalStats(
            workload=workload,
            elapsed_s=time.monotonic() - start,
            candidate_count=total_candidates,
            chunk_count=total_chunks,
            bucket_counts=bucket_counts,
        ),
    )


def _warmup_evaluator(
    evaluator: Callable,
    genotype_dim: int,
    env_batch: CliffEvalEnvData,
    alive_mask: jnp.ndarray,
    batching: WorkloadBatching,
    *,
    result_attr: str,
    candidate_count: int | None = None,
) -> dict[int, float]:
    timings: dict[int, float] = {}
    for bucket_size in planned_warmup_chunk_sizes(candidate_count, batching):
        dummy_genotypes = jnp.zeros((bucket_size, genotype_dim), dtype=jnp.float32)
        _, candidate_keys = split_key_sequence(jax.random.PRNGKey(0), bucket_size)
        started = time.monotonic()
        result = evaluator(dummy_genotypes, env_batch, alive_mask, candidate_keys)
        if result_attr == "trace":
            episode_result, trace = result
            jax.block_until_ready(episode_result.peak_front_x)
            jax.block_until_ready(trace.positions)
        else:
            jax.block_until_ready(getattr(result, result_attr))
        timings[int(bucket_size)] = time.monotonic() - started
    return timings


def _iter_candidate_chunks(
    genotypes: np.ndarray | jax.Array,
    root_keys: jax.Array,
    batching: WorkloadBatching,
):
    total_candidates = int(genotypes.shape[0])
    if total_candidates == 0:
        return

    start = 0
    for chunk_size in _plan_candidate_chunk_sizes(total_candidates, batching):
        stop = min(start + chunk_size, total_candidates)
        chunk = genotypes[start:stop]
        chunk_keys = root_keys[start:stop]
        if int(chunk.shape[0]) != int(chunk_size):
            raise ValueError(
                f"Planned chunk size {chunk_size} did not match extracted size {chunk.shape[0]}"
            )
        yield chunk, chunk_keys, int(chunk_size)
        start = stop


def _plan_candidate_chunk_sizes(total_candidates: int, batching: WorkloadBatching) -> list[int]:
    if total_candidates <= 0:
        return []
    if total_candidates <= batching.exact_fit_small_run_threshold:
        return [int(total_candidates)]

    preferred_sizes = tuple(
        sorted(
            {
                int(size)
                for size in batching.candidate_bucket_sizes
                if 0 < int(size) <= batching.candidate_chunk_size
            }
        )
    )
    if not preferred_sizes:
        raise ValueError(f"No preferred chunk sizes configured for workload '{batching.workload}'")

    planned: list[int] = []
    remaining = int(total_candidates)
    while remaining > 0:
        if remaining <= batching.exact_fit_small_run_threshold:
            planned.append(int(remaining))
            break

        chunk_size = next(
            (size for size in reversed(preferred_sizes) if size <= remaining),
            None,
        )
        if chunk_size is None:
            chunk_size = min(remaining, batching.candidate_chunk_size)
        planned.append(int(chunk_size))
        remaining -= int(chunk_size)
    return planned


def _warmup_chunk_sizes(batching: WorkloadBatching) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(size)
                for size in batching.candidate_bucket_sizes
                if 0 < int(size) <= batching.candidate_chunk_size
            }
        )
    )


def planned_warmup_chunk_sizes(candidate_count: int | None, batching: WorkloadBatching) -> tuple[int, ...]:
    """Return the deduplicated chunk sizes that a warm workload will actually execute."""
    if candidate_count is None:
        return _warmup_chunk_sizes(batching)
    if candidate_count <= 0:
        return ()
    return tuple(dict.fromkeys(_plan_candidate_chunk_sizes(int(candidate_count), batching)))


def _concatenate_pytrees(chunks):
    if not chunks:
        raise ValueError("Expected at least one chunk result")
    if len(chunks) == 1:
        return chunks[0]
    return jax.tree_util.tree_map(lambda *arrays: np.concatenate(arrays, axis=0), *chunks)


def _evaluator_signature(cfg: DictConfig, runtime_resolution: RuntimeResolution) -> tuple:
    rollout_kwargs = rollout_kwargs_from_cfg(
        cfg,
        collision_tuning=runtime_resolution.resolved_collision,
    )
    growth_enabled = str(cfg.body.mode).strip().lower() == "grown"
    return (
        int(cfg.nca.ctrnn_hidden),
        int(cfg.nca.ctrnn_grid_hidden),
        str(cfg.nca.get("gasnet_mode", "off")),
        bool(noise_enabled_from_cfg(cfg)),
        bool(growth_enabled),
        float(cfg.train.get("activation_cost_penalty_scale", 0.0)),
        float(cfg.train.get("soft_loft_penalty_scale", 0.0)),
        tuple(sorted(rollout_kwargs.items())),
    )
