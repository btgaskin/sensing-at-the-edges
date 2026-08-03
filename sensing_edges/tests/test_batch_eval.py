"""Tests for the chunked batched evaluator."""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from jax_evogym.constants import CONTRACTILE, EMPTY

from sensing_edges.batch_eval import (
    GrownEvalAssets,
    _plan_candidate_chunk_sizes,
    evaluate_population_chunked,
    planned_warmup_chunk_sizes,
    probe_candidate_episode_phases,
    probe_candidate_episode_substeps,
    split_key_sequence,
    trace_population_chunked,
    warmup_population_evaluator,
)
from sensing_edges.collision_tuning import resolve_collision_tuning_from_terrains
from sensing_edges.env import (
    build_cliff_env,
    build_static_collider_template,
    infer_static_collider_capacity,
    infer_static_template_capacity,
    make_cliff_env_split_colliders,
    precompute_shared_grid_data,
    run_episode_with_trace,
    stack_static_templates,
)
from sensing_edges.nca import ctrnn_flat_to_params, ctrnn_params_to_flat, init_ctrnn_params, run_growth_phase
from sensing_edges.qd_descriptors import resolve_qd_archive
from sensing_edges.rollout import run_candidate_episodes
from sensing_edges.runtime_config import (
    compose_experiment_config,
    noise_enabled_from_cfg,
    resolved_spawn_x,
    resolve_runtime_config,
    rollout_kwargs_from_cfg,
)
from sensing_edges.terrain import (
    build_cliff_terrain,
    fixed_body_alive_mask,
    make_robot_body,
    resolve_growth_seed_position,
)
from sensing_edges.fitness import candidate_batch_fitness
from sensing_edges.qd_descriptors import build_descriptor_matrix


def _make_tiny_eval_fixture(extra_overrides: list[str] | None = None):
    overrides = [
        "train.max_steps=3",
        "train.gait_spectral_purity_bonus_scale=0.0",
        "sim.physics_substeps=6",
        "batching.generation.candidate_chunk_size=2",
        "batching.generation.candidate_bucket_sizes=[2]",
        "batching.generation.exact_fit_small_run_threshold=2",
    ]
    overrides.extend(extra_overrides or [])
    cfg = compose_experiment_config(preset="paper_run", overrides=overrides)
    resolution = resolve_runtime_config(cfg, preset="paper_run")
    robot = make_robot_body(
        cfg.terrain.robot_w,
        cfg.terrain.robot_h,
        shape=str(cfg.body.shape),
    )
    envs = [
        build_cliff_env(
            build_cliff_terrain(cfg.terrain.grid_h, cfg.terrain.grid_w, cliff_x=15),
            robot,
            spawn_x=cfg.terrain.spawn_x,
            spawn_y=cfg.terrain.spawn_y,
            cliff_x=15,
            physics_substeps=cfg.sim.physics_substeps,
            actuator_convergence=cfg.sim.actuator_convergence,
        ),
        build_cliff_env(
            build_cliff_terrain(cfg.terrain.grid_h, cfg.terrain.grid_w, cliff_x=12),
            robot,
            spawn_x=cfg.terrain.spawn_x,
            spawn_y=cfg.terrain.spawn_y,
            cliff_x=12,
            physics_substeps=cfg.sim.physics_substeps,
            actuator_convergence=cfg.sim.actuator_convergence,
        ),
    ]
    alive_mask = jnp.asarray(fixed_body_alive_mask(robot), dtype=jnp.bool_)
    params = init_ctrnn_params(
        jax.random.PRNGKey(7),
        ctrnn_hidden=int(cfg.nca.ctrnn_hidden),
        ctrnn_grid_hidden=int(cfg.nca.ctrnn_grid_hidden),
        ou_enabled=noise_enabled_from_cfg(cfg),
    )
    genotype = np.asarray(ctrnn_params_to_flat(params))
    return cfg, resolution, envs, alive_mask, genotype


def _make_tiny_grown_eval_fixture():
    cfg = compose_experiment_config(
        preset="iter_run",
        overrides=[
            "body.mode=grown",
            "train.max_steps=3",
            "sim.physics_substeps=6",
            "body.growth.steps=3",
            "batching.generation.candidate_chunk_size=2",
            "batching.generation.candidate_bucket_sizes=[2]",
            "batching.generation.exact_fit_small_run_threshold=2",
        ],
    )
    resolution = resolve_runtime_config(cfg, preset="iter_run")
    spawn_x = resolved_spawn_x(cfg)
    terrains = [
        (build_cliff_terrain(cfg.terrain.grid_h, cfg.terrain.grid_w, cliff_x=15), 15),
        (build_cliff_terrain(cfg.terrain.grid_h, cfg.terrain.grid_w, cliff_x=12), 12),
    ]
    templates = [
        build_static_collider_template(
            terrain,
            cliff_x=cliff_x,
            canvas_h=cfg.body.growth.max_robot_h,
            canvas_w=cfg.body.growth.max_robot_w,
            spawn_x=spawn_x,
            spawn_y=cfg.terrain.spawn_y,
        )
        for terrain, cliff_x in terrains
    ]
    collision_tuning = resolve_collision_tuning_from_terrains(
        [np.asarray(terrain, dtype=np.int32) for terrain, _cliff_x in terrains],
        grid_h=int(cfg.terrain.grid_h),
        grid_w=int(cfg.terrain.grid_w),
        robot_morph=make_robot_body(cfg.body.growth.max_robot_w, cfg.body.growth.max_robot_h),
        spawn_x=int(spawn_x),
        spawn_y=int(cfg.terrain.spawn_y),
        mode=str(cfg.collision.tuning.mode),
    )
    static_capacity = infer_static_template_capacity(templates)
    assets = GrownEvalAssets(
        static_templates=stack_static_templates(templates, static_capacity=static_capacity),
        templates=tuple(templates),
        static_capacity=static_capacity,
        grid_data=precompute_shared_grid_data(cfg.terrain.grid_h, cfg.terrain.grid_w),
        runtime_build_capacity=collision_tuning.runtime_build_capacity,
    )
    alive_mask = jnp.ones((cfg.body.growth.max_robot_h, cfg.body.growth.max_robot_w), dtype=jnp.bool_)
    params = init_ctrnn_params(
        jax.random.PRNGKey(7),
        ctrnn_hidden=int(cfg.nca.ctrnn_hidden),
        ctrnn_grid_hidden=int(cfg.nca.ctrnn_grid_hidden),
        ou_enabled=noise_enabled_from_cfg(cfg),
        growth_enabled=True,
    )
    genotype = np.asarray(ctrnn_params_to_flat(params))
    return cfg, resolution, assets, alive_mask, genotype


@pytest.fixture
def tiny_eval_fixture():
    return _make_tiny_eval_fixture()


def test_chunked_population_eval_matches_serial_candidate_eval(tiny_eval_fixture):
    cfg, resolution, envs, alive_mask, genotype = tiny_eval_fixture
    descriptor_specs = resolve_qd_archive(cfg).descriptor_specs
    root_key = jax.random.PRNGKey(123)
    _, candidate_keys = split_key_sequence(root_key, 1)
    static_capacity = infer_static_collider_capacity(envs)

    serial = run_candidate_episodes(genotype, envs, alive_mask, candidate_keys[0], cfg)
    chunked = evaluate_population_chunked(
        np.asarray([genotype]),
        envs,
        alive_mask,
        candidate_keys,
        cfg,
        workload="generation",
        runtime_resolution=resolution,
        static_capacity=static_capacity,
    )

    assert chunked.stats.chunk_count == 1
    assert chunked.stats.candidate_count == 1
    assert chunked.stats.mean_chunk_size == pytest.approx(1.0)
    assert chunked.stats.max_chunk_size == 1
    assert chunked.fitnesses[0] == pytest.approx(serial.fitness, rel=1e-5, abs=1e-5)
    assert chunked.descriptors[0, 0] == pytest.approx(
        getattr(serial, descriptor_specs[0].metric_attr),
        rel=1e-6,
        abs=1e-6,
    )
    assert chunked.descriptors[0, 1] == pytest.approx(
        getattr(chunked.batch_result, descriptor_specs[1].metric_attr)[0],
        rel=1e-6,
        abs=1e-6,
    )
    assert np.asarray(chunked.batch_result.episode_fitnesses[0]) == pytest.approx(
        serial.episode_fitnesses,
        rel=2e-5,
        abs=2e-5,
    )


def test_chunked_population_eval_reduced_host_result_matches_full_result(tiny_eval_fixture):
    cfg, resolution, envs, alive_mask, genotype = tiny_eval_fixture
    root_key = jax.random.PRNGKey(123)
    _, candidate_keys = split_key_sequence(root_key, 1)
    static_capacity = infer_static_collider_capacity(envs)

    full = evaluate_population_chunked(
        np.asarray([genotype]),
        envs,
        alive_mask,
        candidate_keys,
        cfg,
        workload="generation",
        runtime_resolution=resolution,
        static_capacity=static_capacity,
    )
    reduced = evaluate_population_chunked(
        jnp.asarray([genotype]),
        envs,
        alive_mask,
        candidate_keys,
        cfg,
        workload="generation",
        runtime_resolution=resolution,
        static_capacity=static_capacity,
        reduced_host_result=True,
    )

    assert reduced.fitnesses == pytest.approx(full.fitnesses)
    assert reduced.descriptors == pytest.approx(full.descriptors)
    assert np.array_equal(
        np.asarray(reduced.batch_result.episode_fell),
        np.asarray(full.batch_result.episode_fell),
    )
    assert np.array_equal(
        np.asarray(reduced.batch_result.episode_left_edge),
        np.asarray(full.batch_result.episode_left_edge),
    )
    assert np.asarray(reduced.batch_result.deformation_channel_max) == pytest.approx(
        np.asarray(full.batch_result.deformation_channel_max)
    )
    assert np.asarray(reduced.batch_result.controller_hidden_output_max_abs) == pytest.approx(
        np.asarray(full.batch_result.controller_hidden_output_max_abs)
    )
    assert np.asarray(reduced.batch_result.action_output_max_abs) == pytest.approx(
        np.asarray(full.batch_result.action_output_max_abs)
    )


@pytest.mark.parametrize("reduced_host_result", [False, True])
def test_chunked_population_eval_repeat_same_inputs_is_stable(
    tiny_eval_fixture, reduced_host_result
):
    cfg, resolution, envs, alive_mask, genotype = tiny_eval_fixture
    genotypes = np.stack([genotype, genotype], axis=0)
    root_key = jax.random.PRNGKey(123)
    _, candidate_keys = split_key_sequence(root_key, 2)
    static_capacity = infer_static_collider_capacity(envs)

    first = evaluate_population_chunked(
        genotypes,
        envs,
        alive_mask,
        candidate_keys,
        cfg,
        workload="generation",
        runtime_resolution=resolution,
        static_capacity=static_capacity,
        reduced_host_result=reduced_host_result,
    )
    second = evaluate_population_chunked(
        genotypes,
        envs,
        alive_mask,
        candidate_keys,
        cfg,
        workload="generation",
        runtime_resolution=resolution,
        static_capacity=static_capacity,
        reduced_host_result=reduced_host_result,
    )

    np.testing.assert_allclose(first.fitnesses, second.fitnesses, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(first.descriptors, second.descriptors, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(
        np.asarray(first.batch_result.episode_fell),
        np.asarray(second.batch_result.episode_fell),
    )
    np.testing.assert_array_equal(
        np.asarray(first.batch_result.episode_left_edge),
        np.asarray(second.batch_result.episode_left_edge),
    )


def test_trace_population_materialized_batch_matches_generation_eval_same_inputs(
    tiny_eval_fixture,
):
    cfg, resolution, envs, alive_mask, genotype = tiny_eval_fixture
    genotypes = np.stack([genotype, genotype], axis=0)
    root_key = jax.random.PRNGKey(123)
    _, candidate_keys = split_key_sequence(root_key, 2)
    static_capacity = infer_static_collider_capacity(envs)

    full = evaluate_population_chunked(
        genotypes,
        envs,
        alive_mask,
        candidate_keys,
        cfg,
        workload="generation",
        runtime_resolution=resolution,
        static_capacity=static_capacity,
    )
    trace_eval = trace_population_chunked(
        genotypes,
        envs,
        alive_mask,
        candidate_keys,
        cfg,
        workload="generation",
        runtime_resolution=resolution,
        static_capacity=static_capacity,
    )
    alive_count = float(jnp.sum(alive_mask.astype(jnp.float32)))
    trace_batch_result = candidate_batch_fitness(
        trace_eval.episode_results,
        alive_cell_count=jnp.full((genotypes.shape[0],), alive_count, dtype=jnp.float32),
        activation_cost_penalty_scale=float(cfg.train.get("activation_cost_penalty_scale", 0.0)),
        soft_loft_penalty_scale=float(cfg.train.get("soft_loft_penalty_scale", 0.0)),
    )
    trace_host_result = jax.tree_util.tree_map(lambda array: np.asarray(array), trace_batch_result)
    trace_descriptors = build_descriptor_matrix(
        trace_host_result,
        resolve_qd_archive(cfg).descriptor_specs,
    )

    np.testing.assert_allclose(
        full.fitnesses,
        np.asarray(trace_host_result.fitness, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        full.descriptors,
        np.asarray(trace_descriptors, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_probe_candidate_episode_phases_repeat_same_inputs_is_stable(tiny_eval_fixture):
    cfg, resolution, envs, alive_mask, genotype = tiny_eval_fixture
    _, candidate_keys = split_key_sequence(jax.random.PRNGKey(123), 1)
    static_capacity = infer_static_collider_capacity(envs)

    first = probe_candidate_episode_phases(
        genotype,
        envs,
        alive_mask,
        candidate_keys[0],
        cfg,
        runtime_resolution=resolution,
        static_capacity=static_capacity,
    )
    second = probe_candidate_episode_phases(
        genotype,
        envs,
        alive_mask,
        candidate_keys[0],
        cfg,
        runtime_resolution=resolution,
        static_capacity=static_capacity,
    )

    np.testing.assert_allclose(
        np.asarray(first.episode_results.peak_front_x),
        np.asarray(second.episode_results.peak_front_x),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(first.phase_trace.phase_h_compact),
        np.asarray(second.phase_trace.phase_h_compact),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(first.phase_trace.phase_post_positions),
        np.asarray(second.phase_trace.phase_post_positions),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(first.phase_trace.phase_done_after),
        np.asarray(second.phase_trace.phase_done_after),
    )


def test_probe_candidate_episode_substeps_repeat_same_inputs_is_stable(
    tiny_eval_fixture,
):
    cfg, resolution, envs, alive_mask, genotype = tiny_eval_fixture
    _, candidate_keys = split_key_sequence(jax.random.PRNGKey(123), 1)
    static_capacity = infer_static_collider_capacity(envs)

    first = probe_candidate_episode_substeps(
        genotype,
        envs,
        alive_mask,
        candidate_keys[0],
        cfg,
        selected_episode_index=0,
        selected_step_index=0,
        selected_phase_index=0,
        runtime_resolution=resolution,
        static_capacity=static_capacity,
    )
    second = probe_candidate_episode_substeps(
        genotype,
        envs,
        alive_mask,
        candidate_keys[0],
        cfg,
        selected_episode_index=0,
        selected_step_index=0,
        selected_phase_index=0,
        runtime_resolution=resolution,
        static_capacity=static_capacity,
    )

    np.testing.assert_allclose(
        np.asarray(first.episode_result.peak_front_x),
        np.asarray(second.episode_result.peak_front_x),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(first.substep_trace.substep_dynamic_overflow),
        np.asarray(second.substep_trace.substep_dynamic_overflow),
    )
    np.testing.assert_allclose(
        np.asarray(first.substep_trace.substep_dynamic_forces),
        np.asarray(second.substep_trace.substep_dynamic_forces),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(first.substep_trace.substep_final_velocities),
        np.asarray(second.substep_trace.substep_final_velocities),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(first.substep_trace.substep_final_velocities_true),
        np.asarray(second.substep_trace.substep_final_velocities_true),
        rtol=1e-6,
        atol=1e-6,
    )


def test_chunked_population_eval_uses_selected_qd_descriptor_pair():
    cfg, resolution, envs, alive_mask, genotype = _make_tiny_eval_fixture(
        extra_overrides=[
            "qd.descriptors.names=[fall_count,cliff_vicinity]",
            "qd.focus.enabled=false",
        ]
    )
    root_key = jax.random.PRNGKey(123)
    _, candidate_keys = split_key_sequence(root_key, 1)
    static_capacity = infer_static_collider_capacity(envs)

    serial = run_candidate_episodes(genotype, envs, alive_mask, candidate_keys[0], cfg)
    chunked = evaluate_population_chunked(
        np.asarray([genotype]),
        envs,
        alive_mask,
        candidate_keys,
        cfg,
        workload="generation",
        runtime_resolution=resolution,
        static_capacity=static_capacity,
    )

    assert chunked.descriptors[0, 0] == pytest.approx(serial.fall_count, rel=1e-6, abs=1e-6)
    assert chunked.descriptors[0, 1] == pytest.approx(serial.cliff_vicinity, rel=1e-6, abs=1e-6)


def test_chunked_population_eval_matches_serial_candidate_eval_grown():
    cfg, resolution, envs, alive_mask, genotype = _make_tiny_grown_eval_fixture()
    root_key = jax.random.PRNGKey(123)
    _, candidate_keys = split_key_sequence(root_key, 1)

    serial = run_candidate_episodes(genotype, envs, alive_mask, candidate_keys[0], cfg)
    chunked = evaluate_population_chunked(
        np.asarray([genotype]),
        envs,
        alive_mask,
        candidate_keys,
        cfg,
        workload="generation",
        runtime_resolution=resolution,
    )

    assert chunked.fitnesses[0] == pytest.approx(serial.fitness, rel=1e-5, abs=1e-5)
    assert chunked.batch_result.alive_cell_count[0] == pytest.approx(serial.alive_cell_count, rel=1e-5)


def test_chunked_grown_trace_eval_matches_serial_trace_rollout():
    cfg, resolution, envs, alive_mask, genotype = _make_tiny_grown_eval_fixture()
    root_key = jax.random.PRNGKey(123)
    _, candidate_keys = split_key_sequence(root_key, 1)

    chunked = trace_population_chunked(
        np.asarray([genotype]),
        envs,
        alive_mask,
        candidate_keys,
        cfg,
        workload="trace",
        runtime_resolution=resolution,
    )
    assert chunked.render_preps is not None

    params = ctrnn_flat_to_params(
        jnp.asarray(genotype),
        ctrnn_hidden=int(cfg.nca.ctrnn_hidden),
        ctrnn_grid_hidden=int(cfg.nca.ctrnn_grid_hidden),
        gasnet_mode=str(cfg.nca.get("gasnet_mode", "off")),
        ou_enabled=bool(noise_enabled_from_cfg(cfg)),
        growth_enabled=True,
    )
    seed_y, seed_x = resolve_growth_seed_position(
        int(cfg.body.growth.max_robot_h),
        int(cfg.body.growth.max_robot_w),
        str(cfg.body.growth.get("seed_position", "center")),
    )
    key_batch = jax.random.split(candidate_keys[0], len(envs.templates) + 1)
    growth_key = key_batch[0]
    episode_keys = key_batch[1:]
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
    )
    robot_morph = jnp.where(grown_alive_mask, jnp.int32(CONTRACTILE), jnp.int32(EMPTY))
    rollout_kwargs = rollout_kwargs_from_cfg(
        cfg,
        collision_tuning=resolution.resolved_collision,
    )
    spawn_x = resolved_spawn_x(cfg)
    initial_nca_state = (
        growth_state
        if str(cfg.body.growth.state_transition).strip().lower() == "pass_through"
        else None
    )
    initial_ctrnn_carry = (
        growth_carry
        if str(cfg.body.growth.state_transition).strip().lower() == "pass_through"
        else None
    )

    serial_results = []
    serial_traces = []
    for template, ep_key in zip(envs.templates, episode_keys, strict=False):
        env = make_cliff_env_split_colliders(
            robot_morph,
            envs.grid_data,
            int(spawn_x),
            int(cfg.terrain.spawn_y),
            cliff_x=int(template.cliff_x),
            death_plane_y=float(template.death_plane_y),
            static_collider_template=template.static_collider_data,
            physics_substeps=int(cfg.sim.physics_substeps),
            actuator_convergence=float(cfg.sim.actuator_convergence),
            spring_stiffness_scale=float(cfg.sim.spring_stiffness_scale),
            max_triples=int(envs.runtime_build_capacity.max_triples),
            max_self_triples=int(envs.runtime_build_capacity.max_self_triples),
        )
        result, trace = run_episode_with_trace(
            env,
            params,
            grown_alive_mask,
            ep_key,
            frozen_identity=frozen_identity,
            initial_nca_state=initial_nca_state,
            initial_ctrnn_carry=initial_ctrnn_carry,
            **rollout_kwargs,
        )
        serial_results.append(result)
        serial_traces.append(trace)

    chunked_results = jax.tree_util.tree_map(lambda array: np.asarray(array)[0], chunked.episode_results)
    chunked_traces = jax.tree_util.tree_map(lambda array: np.asarray(array)[0], chunked.traces)
    chunked_render_prep = jax.tree_util.tree_map(lambda array: np.asarray(array)[0], chunked.render_preps)
    for episode_idx, serial_result in enumerate(serial_results):
        for field in serial_result._fields:
            np.testing.assert_allclose(
                np.asarray(getattr(chunked_results, field)[episode_idx]),
                np.asarray(getattr(serial_result, field)),
                rtol=1e-5,
                atol=1e-5,
            )
        np.testing.assert_allclose(
            np.asarray(chunked_traces.positions[episode_idx]),
            np.asarray(serial_traces[episode_idx].positions),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(chunked_traces.spring_rest_length[episode_idx]),
            np.asarray(serial_traces[episode_idx].spring_rest_length),
            rtol=1e-5,
            atol=1e-5,
        )
    np.testing.assert_array_equal(
        np.asarray(chunked_render_prep.robot_morph),
        np.asarray(robot_morph),
    )


def test_population_warmup_compiles_only_planned_chunk_sizes(tiny_eval_fixture):
    cfg, resolution, envs, alive_mask, genotype = tiny_eval_fixture
    timings = warmup_population_evaluator(
        genotype.shape[0],
        envs,
        alive_mask,
        cfg,
        workload="generation",
        runtime_resolution=resolution,
        static_capacity=infer_static_collider_capacity(envs),
        candidate_count=1,
    )

    assert set(timings) == {1}
    assert timings[1] >= 0.0


def test_candidate_chunk_planner_uses_exact_generation_breakpoints():
    cfg = compose_experiment_config()
    batching = resolve_runtime_config(cfg).resolved_batching["generation"]

    assert _plan_candidate_chunk_sizes(50, batching) == [50]
    assert _plan_candidate_chunk_sizes(100, batching) == [100]
    assert _plan_candidate_chunk_sizes(150, batching) == [100, 50]
    assert _plan_candidate_chunk_sizes(200, batching) == [100, 100]
    assert _plan_candidate_chunk_sizes(300, batching) == [100, 100, 100]


def test_candidate_chunk_planner_uses_exact_fit_for_small_and_tail_chunks():
    cfg = compose_experiment_config()
    batching = resolve_runtime_config(cfg).resolved_batching["generation"]

    assert _plan_candidate_chunk_sizes(20, batching) == [20]
    assert _plan_candidate_chunk_sizes(75, batching) == [50, 25]
    assert _plan_candidate_chunk_sizes(130, batching) == [100, 30]


def test_warmup_chunk_sizes_follow_actual_planned_run_shapes():
    cfg = compose_experiment_config()
    batching = resolve_runtime_config(cfg).resolved_batching["generation"]

    assert planned_warmup_chunk_sizes(50, batching) == (50,)
    assert planned_warmup_chunk_sizes(150, batching) == (100, 50)
    assert planned_warmup_chunk_sizes(75, batching) == (50, 25)
