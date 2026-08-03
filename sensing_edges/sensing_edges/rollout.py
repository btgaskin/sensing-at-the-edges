"""Serial reference rollout helpers.

These helpers are kept for tests and parity checks. The production train/eval
path uses `sensing_edges.batch_eval`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from jax_evogym.constants import CONTRACTILE, EMPTY

from .batch_eval import GrownEvalAssets
from .env import (
    CliffEnvData,
    EpisodeResult,
    make_cliff_env_split_colliders,
    run_episode,
    run_episode_with_com_trace,
)
from .fitness import candidate_fitness, CandidateResult
from .nca import ctrnn_flat_to_params, run_growth_phase
from .qd.gait_descriptor import compute_candidate_gait_spectral_purity
from .runtime_config import (
    active_comm_filters_from_cfg,
    noise_enabled_from_cfg,
    rollout_kwargs_from_cfg,
    resolved_spawn_x,
)
from .terrain import resolve_growth_seed_position


def _params_from_genotype(genotype: np.ndarray, cfg: DictConfig):
    """Unflatten a genotype vector into CTRNNParams."""
    return ctrnn_flat_to_params(
        jnp.array(genotype),
        ctrnn_hidden=cfg.nca.ctrnn_hidden,
        ctrnn_grid_hidden=cfg.nca.ctrnn_grid_hidden,
        gasnet_mode=cfg.nca.get("gasnet_mode", "off"),
        ou_enabled=noise_enabled_from_cfg(cfg),
        growth_enabled=str(cfg.body.mode).strip().lower() == "grown",
        active_comm_filters=active_comm_filters_from_cfg(cfg),
    )


def run_candidate_episodes(
    genotype: np.ndarray,
    env_datas: list[CliffEnvData] | GrownEvalAssets,
    alive_mask: jnp.ndarray,
    key: jax.Array,
    cfg: DictConfig,
) -> CandidateResult:
    """Evaluate one candidate across episode terrains with serial Python loops."""
    params = _params_from_genotype(genotype, cfg)
    nca_kwargs = rollout_kwargs_from_cfg(cfg)
    gait_bonus_scale = float(cfg.train.get("gait_spectral_purity_bonus_scale", 0.0))
    soft_loft_penalty_scale = float(cfg.train.get("soft_loft_penalty_scale", 0.0))
    capture_com_trace = gait_bonus_scale > 0.0

    episode_results: list[EpisodeResult] = []
    com_x_traces: list[np.ndarray] = []
    if str(cfg.body.mode).strip().lower() == "grown":
        if not isinstance(env_datas, GrownEvalAssets):
            raise TypeError("Grown serial rollout expects GrownEvalAssets")
        seed_y, seed_x = resolve_growth_seed_position(
            int(cfg.body.growth.max_robot_h),
            int(cfg.body.growth.max_robot_w),
            str(cfg.body.growth.get("seed_position", "center")),
        )
        key, growth_key = jax.random.split(key)
        grown_alive_mask, frozen_identity, growth_state, growth_carry = run_growth_phase(
            params,
            growth_key,
            growth_steps=int(cfg.body.growth.steps),
            canvas_h=int(cfg.body.growth.max_robot_h),
            canvas_w=int(cfg.body.growth.max_robot_w),
            ctrnn_hidden=int(cfg.nca.ctrnn_hidden),
            ctrnn_grid_hidden=int(cfg.nca.ctrnn_grid_hidden),
            gasnet_mode=str(cfg.nca.get("gasnet_mode", "off")),
            ou_enabled=bool(noise_enabled_from_cfg(cfg)),
            seed_y=seed_y,
            seed_x=seed_x,
            alive_threshold=float(cfg.body.growth.get("threshold", 0.5)),
            frontier_identity_only=bool(cfg.body.growth.get("frontier_identity_only", True)),
            p_update=float(cfg.nca.p_update),
            update_mode=str(cfg.nca.update_mode),
            state_clip_value=float(cfg.nca.state_clip_value),
            connectivity_max_iters=int(cfg.body.growth.get("connectivity_max_iters", 10)),
            noise_init_scale=float(cfg.body.growth.get("noise_init_scale", 0.0)),
            ctrnn_tau_min=float(cfg.nca.ctrnn_tau_min),
            ctrnn_tau_max=float(cfg.nca.ctrnn_tau_max),
            ctrnn_dt=float(cfg.nca.ctrnn_dt),
            active_comm_filters=active_comm_filters_from_cfg(cfg),
        )
        robot_morph = jnp.where(grown_alive_mask, jnp.int32(CONTRACTILE), jnp.int32(EMPTY))
        for idx in range(int(env_datas.static_templates.cliff_x.shape[0])):
            key, ep_key = jax.random.split(key)
            template = jax.tree_util.tree_map(
                lambda array: array[idx],
                env_datas.static_templates.static_collider_data,
            )
            env_data = make_cliff_env_split_colliders(
                robot_morph,
                env_datas.grid_data,
                resolved_spawn_x(cfg),
                int(cfg.terrain.spawn_y),
                cliff_x=env_datas.static_templates.cliff_x[idx],
                task_mode=env_datas.static_templates.task_mode[idx],
                goal_direction=env_datas.static_templates.goal_direction[idx],
                cliff_side=env_datas.static_templates.cliff_side[idx],
                goal_distance=env_datas.static_templates.goal_distance[idx],
                cliff_distance=env_datas.static_templates.cliff_distance[idx],
                goal_block_x=env_datas.static_templates.goal_block_x[idx],
                goal_block_y=env_datas.static_templates.goal_block_y[idx],
                goal_x=env_datas.static_templates.goal_x[idx],
                goal_object_id=env_datas.static_templates.goal_object_id[idx],
                cliff_probe_ledge_width=env_datas.static_templates.cliff_probe_ledge_width[idx],
                cliff_probe_ledge_cell_x=env_datas.static_templates.cliff_probe_ledge_cell_x[idx],
                cliff_probe_ledge_cell_y=env_datas.static_templates.cliff_probe_ledge_cell_y[idx],
                death_plane_y=env_datas.static_templates.death_plane_y[idx],
                static_collider_template=template,
                physics_substeps=int(cfg.sim.physics_substeps),
                actuator_convergence=float(cfg.sim.actuator_convergence),
                spring_stiffness_scale=float(cfg.sim.spring_stiffness_scale),
                floor_static_friction_const=float(cfg.sim.floor_static_friction_const),
                floor_stiction_stiffness=float(cfg.sim.floor_stiction_stiffness),
                floor_stiction_switch_speed=float(cfg.sim.floor_stiction_switch_speed),
                max_triples=int(env_datas.runtime_build_capacity.max_triples),
                max_self_triples=int(env_datas.runtime_build_capacity.max_self_triples),
            )
            rollout_kwargs = dict(
                frozen_identity=frozen_identity,
                initial_nca_state=(
                    growth_state
                    if str(cfg.body.growth.state_transition).strip().lower() == "pass_through"
                    else None
                ),
                initial_ctrnn_carry=(
                    growth_carry
                    if str(cfg.body.growth.state_transition).strip().lower() == "pass_through"
                    else None
                ),
                **nca_kwargs,
            )
            if capture_com_trace:
                result, com_trace = run_episode_with_com_trace(
                    env_data,
                    params,
                    grown_alive_mask,
                    ep_key,
                    **rollout_kwargs,
                )
                com_x_traces.append(np.asarray(com_trace.com_x, dtype=np.float32))
            else:
                result = run_episode(
                    env_data,
                    params,
                    grown_alive_mask,
                    ep_key,
                    **rollout_kwargs,
                )
            episode_results.append(result)
        gait_purity = 0.0
        if capture_com_trace:
            gait_purity = float(
                compute_candidate_gait_spectral_purity(
                    np.asarray([com_x_traces], dtype=np.float32),
                    np.asarray([[int(r.steps_alive) for r in episode_results]], dtype=np.int32),
                )[0]
            )
        return candidate_fitness(
            episode_results,
            alive_cell_count=float(jnp.sum(grown_alive_mask.astype(jnp.float32))),
            activation_cost_penalty_scale=float(
                cfg.train.get("activation_cost_penalty_scale", 0.0)
            ),
            soft_loft_penalty_scale=soft_loft_penalty_scale,
            gait_spectral_purity=gait_purity,
            gait_spectral_purity_bonus_scale=gait_bonus_scale,
        )

    for env_data in env_datas:
        key, ep_key = jax.random.split(key)
        if capture_com_trace:
            result, com_trace = run_episode_with_com_trace(
                env_data,
                params,
                alive_mask,
                ep_key,
                **nca_kwargs,
            )
            com_x_traces.append(np.asarray(com_trace.com_x, dtype=np.float32))
        else:
            result = run_episode(env_data, params, alive_mask, ep_key, **nca_kwargs)
        episode_results.append(result)

    gait_purity = 0.0
    if capture_com_trace:
        gait_purity = float(
            compute_candidate_gait_spectral_purity(
                np.asarray([com_x_traces], dtype=np.float32),
                np.asarray([[int(r.steps_alive) for r in episode_results]], dtype=np.int32),
            )[0]
        )
    return candidate_fitness(
        episode_results,
        alive_cell_count=float(jnp.sum(alive_mask.astype(jnp.float32))),
        activation_cost_penalty_scale=float(
            cfg.train.get("activation_cost_penalty_scale", 0.0)
        ),
        soft_loft_penalty_scale=soft_loft_penalty_scale,
        gait_spectral_purity=gait_purity,
        gait_spectral_purity_bonus_scale=gait_bonus_scale,
    )
