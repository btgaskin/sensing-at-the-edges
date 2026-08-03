"""Tests for collision capacity tuning."""

from __future__ import annotations

import numpy as np

from jax_evogym import constants as C

from sensing_edges.collision_tuning import (
    resolve_collision_tuning_from_env_datas,
    resolve_collision_tuning_from_terrains,
)
from sensing_edges.env import (
    assemble_cliff_env,
    build_cliff_dynamic_core,
    build_cliff_env,
    build_cliff_terrain_assets,
)
from sensing_edges.runtime_config import compose_experiment_config
from sensing_edges.terrain import build_cliff_terrain, make_robot_body


def _round_up(value: float, quantum: int) -> int:
    base = max(1, int(np.ceil(float(value))))
    return int(((base + quantum - 1) // quantum) * quantum)


def test_collision_tuning_matches_hybrid_probe_formula():
    cfg = compose_experiment_config()
    robot = make_robot_body(cfg.terrain.robot_w, cfg.terrain.robot_h)
    terrains = [
        build_cliff_terrain(cfg.terrain.grid_h, cfg.terrain.grid_w, cliff_x=cfg.terrain.cliff_x_min),
        build_cliff_terrain(cfg.terrain.grid_h, cfg.terrain.grid_w, cliff_x=cfg.terrain.cliff_x_max),
    ]

    tuning = resolve_collision_tuning_from_terrains(
        terrains,
        grid_h=int(cfg.terrain.grid_h),
        grid_w=int(cfg.terrain.grid_w),
        robot_morph=robot,
        spawn_x=int(cfg.terrain.spawn_x),
        spawn_y=int(cfg.terrain.spawn_y),
        mode=str(cfg.collision.tuning.mode),
    )

    observed_pairs = max(
        tuning.suite_counts.max_candidate_pairs,
        tuning.synthetic_counts.max_candidate_pairs,
    )
    observed_triples = max(
        tuning.suite_counts.max_triples,
        tuning.synthetic_counts.max_triples,
    )
    observed_self = max(
        tuning.suite_counts.max_self_triples,
        tuning.synthetic_counts.max_self_triples,
    )

    assert tuning.runtime_counts.max_candidate_pairs > 0
    assert tuning.runtime_counts.max_triples > 0
    assert tuning.runtime_counts.max_self_triples > 0
    assert tuning.synthetic_counts.max_candidate_pairs > tuning.suite_counts.max_candidate_pairs
    assert tuning.synthetic_counts.max_triples > tuning.suite_counts.max_triples
    assert tuning.synthetic_counts.max_self_triples >= tuning.suite_counts.max_self_triples
    assert tuning.build_capacity.max_triples == _round_up(observed_triples * 1.10 + 128.0, 256)
    assert tuning.build_capacity.max_self_triples == _round_up(observed_self * 1.10 + 64.0, 64)
    assert tuning.runtime_build_capacity.max_triples == _round_up(
        tuning.runtime_counts.max_triples * 1.10 + 128.0,
        256,
    )
    assert tuning.runtime_build_capacity.max_self_triples == _round_up(
        tuning.runtime_counts.max_self_triples * 1.10 + 64.0,
        64,
    )
    assert tuning.dynamic_max_candidates == _round_up(tuning.runtime_counts.max_candidate_pairs * 1.10, 64)
    assert tuning.dynamic_max_narrow == min(
        tuning.runtime_build_capacity.max_triples,
        4 * tuning.dynamic_max_candidates,
    )


def test_build_cliff_env_uses_tuned_collision_buffers():
    cfg = compose_experiment_config()
    robot = make_robot_body(cfg.terrain.robot_w, cfg.terrain.robot_h)
    terrain = build_cliff_terrain(cfg.terrain.grid_h, cfg.terrain.grid_w, cliff_x=cfg.terrain.cliff_x_min)
    tuning = resolve_collision_tuning_from_terrains(
        [terrain],
        grid_h=int(cfg.terrain.grid_h),
        grid_w=int(cfg.terrain.grid_w),
        robot_morph=robot,
        spawn_x=int(cfg.terrain.spawn_x),
        spawn_y=int(cfg.terrain.spawn_y),
        mode=str(cfg.collision.tuning.mode),
    )

    env = build_cliff_env(
        terrain,
        robot,
        spawn_x=cfg.terrain.spawn_x,
        spawn_y=cfg.terrain.spawn_y,
        cliff_x=cfg.terrain.cliff_x_min,
        collision_build_capacity=tuning.build_capacity,
    )

    assert int(env.collision_data.triple_active.shape[0]) == tuning.build_capacity.max_triples
    assert int(env.collision_data.self_triple_active.shape[0]) == tuning.build_capacity.max_self_triples
    assert int(env.collision_data.n_triples) < tuning.build_capacity.max_triples


def test_split_env_build_uses_runtime_and_full_capacities():
    cfg = compose_experiment_config()
    robot = make_robot_body(cfg.terrain.robot_w, cfg.terrain.robot_h)
    terrain = build_cliff_terrain(cfg.terrain.grid_h, cfg.terrain.grid_w, cliff_x=cfg.terrain.cliff_x_min)
    tuning = resolve_collision_tuning_from_terrains(
        [terrain],
        grid_h=int(cfg.terrain.grid_h),
        grid_w=int(cfg.terrain.grid_w),
        robot_morph=robot,
        spawn_x=int(cfg.terrain.spawn_x),
        spawn_y=int(cfg.terrain.spawn_y),
        mode=str(cfg.collision.tuning.mode),
    )

    dynamic_core = build_cliff_dynamic_core(
        robot,
        spawn_x=cfg.terrain.spawn_x,
        spawn_y=cfg.terrain.spawn_y,
        physics_substeps=cfg.sim.physics_substeps,
        actuator_convergence=cfg.sim.actuator_convergence,
        spring_stiffness_scale=cfg.sim.spring_stiffness_scale,
        collision_build_capacity=tuning.runtime_build_capacity,
    )
    terrain_assets = build_cliff_terrain_assets(
        terrain,
        robot,
        cfg.terrain.spawn_x,
        cfg.terrain.spawn_y,
        cfg.terrain.cliff_x_min,
        spring_stiffness_scale=cfg.sim.spring_stiffness_scale,
        collision_build_capacity=tuning.build_capacity,
    )
    env = assemble_cliff_env(dynamic_core, terrain_assets)

    assert int(env.collision_data.triple_active.shape[0]) == tuning.runtime_build_capacity.max_triples
    assert int(env.collision_data.self_triple_active.shape[0]) == tuning.runtime_build_capacity.max_self_triples
    assert int(env.static_collider_data.cell_vertices.shape[0]) > 0
    assert int(env.static_collider_data.cell_aabb_min.shape[0]) == int(
        env.static_collider_data.cell_vertices.shape[0]
    )


def test_env_data_tuning_matches_terrain_tuning_for_runtime_caps():
    cfg = compose_experiment_config()
    robot = make_robot_body(cfg.terrain.robot_w, cfg.terrain.robot_h)
    terrain = np.full((cfg.terrain.grid_h, cfg.terrain.grid_w), C.RIGID, dtype=np.int32)

    terrain_tuning = resolve_collision_tuning_from_terrains(
        [terrain],
        grid_h=int(cfg.terrain.grid_h),
        grid_w=int(cfg.terrain.grid_w),
        robot_morph=robot,
        spawn_x=int(cfg.terrain.spawn_x),
        spawn_y=int(cfg.terrain.spawn_y),
        mode=str(cfg.collision.tuning.mode),
    )
    env = build_cliff_env(
        terrain,
        robot,
        spawn_x=cfg.terrain.spawn_x,
        spawn_y=cfg.terrain.spawn_y,
        cliff_x=cfg.terrain.cliff_x_max,
    )
    env_tuning = resolve_collision_tuning_from_env_datas(
        [env],
        grid_h=int(cfg.terrain.grid_h),
        grid_w=int(cfg.terrain.grid_w),
        robot_morph=robot,
        spawn_x=int(cfg.terrain.spawn_x),
        spawn_y=int(cfg.terrain.spawn_y),
        mode=str(cfg.collision.tuning.mode),
    )

    assert env_tuning.dynamic_max_candidates == terrain_tuning.dynamic_max_candidates
    assert env_tuning.dynamic_max_narrow == terrain_tuning.dynamic_max_narrow
    assert env_tuning.runtime_build_capacity == terrain_tuning.runtime_build_capacity
