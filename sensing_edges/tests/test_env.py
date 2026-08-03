"""Tests for cliff-walk environment build and episode runner."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_evogym.constants import CELL_SIZE, FIXED, CONTRACTILE

from sensing_edges.terrain import (
    build_bilateral_goal_cliff_terrain,
    build_cliff_gap_terrain,
    build_cliff_terrain,
    make_robot_body,
)
from sensing_edges.env import (
    CliffEnvData,
    CliffEpisodeState,
    EpisodeResult,
    RobotKinematics,
    _bilateral_progress_metrics,
    _episode_turn_motion_threshold,
    _episode_result_from_state,
    _lower_ledge_exposure_now,
    _significant_step_direction,
    _update_stable_turn_tracking,
    _update_turn_tracking,
    attach_static_template_to_eval_env,
    build_static_collider_template,
    build_cliff_eval_dynamic_core,
    build_cliff_env,
    build_grown_dynamic_render_info,
    derive_physics_params,
    infer_static_collider_capacity,
    make_cliff_env_split_colliders,
    precompute_shared_grid_data,
    reset_episode,
    run_episode,
    stack_eval_envs,
)
from sensing_edges.nca import (
    NCAStepDiagnostics,
    base_channel_count,
    init_ctrnn_params,
    ctrnn_params_to_flat,
    ctrnn_flat_to_params,
)
import sensing_edges.env as env_module


@pytest.fixture
def simple_env():
    """Build a minimal cliff environment for testing."""
    terrain = build_cliff_terrain(10, 35, cliff_x=15)
    robot = make_robot_body(7, 2)
    return build_cliff_env(terrain, robot, spawn_x=1, spawn_y=4, cliff_x=15)


@pytest.fixture
def short_env():
    """Cliff very close to robot for termination tests."""
    terrain = build_cliff_terrain(10, 35, cliff_x=10)
    robot = make_robot_body(7, 2)
    return build_cliff_env(terrain, robot, spawn_x=1, spawn_y=4, cliff_x=10)


@pytest.fixture
def comb_env():
    """Sparse fixed comb morphology for world-alignment regressions."""
    terrain = build_cliff_terrain(10, 35, cliff_x=15)
    robot = make_robot_body(7, 2, shape="comb")
    return build_cliff_env(terrain, robot, spawn_x=1, spawn_y=4, cliff_x=15)


@pytest.fixture
def comb_7x3_env():
    """Tall sparse fixed comb morphology for world-alignment regressions."""
    terrain = build_cliff_terrain(10, 35, cliff_x=15)
    robot = make_robot_body(7, 3, shape="comb_7x3")
    return build_cliff_env(terrain, robot, spawn_x=1, spawn_y=4, cliff_x=15)


class TestBilateralBookkeepingHelpers:
    def test_bilateral_progress_metrics_normalize_correct_and_incorrect_excursions(self):
        any_progress, goal_progress, incorrect_progress, toward_goal_frontier, reversal_progress = _bilateral_progress_metrics(
            bilateral_task=jnp.bool_(True),
            init_com_x=jnp.float32(10.5 * CELL_SIZE),
            init_min_x=jnp.float32(10.0 * CELL_SIZE),
            init_front_x=jnp.float32(11.0 * CELL_SIZE),
            goal_x=jnp.float32(14.5 * CELL_SIZE),
            goal_direction=jnp.int32(1),
            cliff_x=jnp.int32(9),
            cliff_side=jnp.int32(-1),
            goal_touched=jnp.bool_(False),
            final_min_x=jnp.float32(12.0 * CELL_SIZE),
            final_front_x=jnp.float32(13.0 * CELL_SIZE),
            peak_any_direction_progress=jnp.float32(1.5 * CELL_SIZE),
            peak_goal_progress=jnp.float32(2.0 * CELL_SIZE),
            peak_incorrect_progress=jnp.float32(0.5 * CELL_SIZE),
        )

        assert float(any_progress) == pytest.approx(1.5 / 3.5)
        assert float(goal_progress) == pytest.approx(2.0 / 3.5)
        assert float(incorrect_progress) == pytest.approx(0.5)
        assert float(toward_goal_frontier) == pytest.approx(2.0 / 3.5)
        assert float(reversal_progress) == pytest.approx(0.5)

    def test_lower_ledge_exposure_requires_x_overlap_and_depth(self):
        ledge_kwargs = dict(
            cliff_probe_ledge_width=3,
            cliff_probe_ledge_cell_x=6,
            cliff_probe_ledge_cell_y=2,
        )
        exposed = _lower_ledge_exposure_now(
            RobotKinematics(
                min_x=jnp.float32(0.62),
                front_x=jnp.float32(0.74),
                min_y=jnp.float32((3.0 * CELL_SIZE) - 1e-3),
                max_y=jnp.float32(0.36),
                com_x=jnp.float32(0.68),
                com_y=jnp.float32(0.33),
            ),
            **ledge_kwargs,
        )
        above = _lower_ledge_exposure_now(
            RobotKinematics(
                min_x=jnp.float32(0.62),
                front_x=jnp.float32(0.74),
                min_y=jnp.float32((3.0 * CELL_SIZE) + 1e-3),
                max_y=jnp.float32(0.37),
                com_x=jnp.float32(0.68),
                com_y=jnp.float32(0.34),
            ),
            **ledge_kwargs,
        )
        outside = _lower_ledge_exposure_now(
            RobotKinematics(
                min_x=jnp.float32(0.10),
                front_x=jnp.float32(0.20),
                min_y=jnp.float32((3.0 * CELL_SIZE) - 1e-3),
                max_y=jnp.float32(0.36),
                com_x=jnp.float32(0.15),
                com_y=jnp.float32(0.33),
            ),
            **ledge_kwargs,
        )

        assert bool(exposed)
        assert not bool(above)
        assert not bool(outside)

    def test_update_turn_tracking_ignores_small_jitter_and_counts_significant_flips(self):
        last_direction = jnp.int32(0)
        turn_count = jnp.int32(0)
        turn_motion_threshold = _episode_turn_motion_threshold(
            init_min_x=jnp.float32(0.0),
            init_front_x=jnp.float32(1.0 * CELL_SIZE),
        )

        last_direction, turn_count = _update_turn_tracking(
            last_significant_direction=last_direction,
            turn_count=turn_count,
            delta_com_x=jnp.float32(0.05 * CELL_SIZE),
            turn_motion_threshold=turn_motion_threshold,
        )
        assert int(last_direction) == 0
        assert int(turn_count) == 0

        last_direction, turn_count = _update_turn_tracking(
            last_significant_direction=last_direction,
            turn_count=turn_count,
            delta_com_x=jnp.float32(0.2 * CELL_SIZE),
            turn_motion_threshold=turn_motion_threshold,
        )
        assert int(last_direction) == 1
        assert int(turn_count) == 0

        last_direction, turn_count = _update_turn_tracking(
            last_significant_direction=last_direction,
            turn_count=turn_count,
            delta_com_x=jnp.float32(-0.2 * CELL_SIZE),
            turn_motion_threshold=turn_motion_threshold,
        )
        assert int(last_direction) == -1
        assert int(turn_count) == 1

    def test_significant_step_direction_reuses_turn_tracking_threshold(self):
        turn_motion_threshold = _episode_turn_motion_threshold(
            init_min_x=jnp.float32(0.0),
            init_front_x=jnp.float32(1.0 * CELL_SIZE),
        )

        assert int(
            _significant_step_direction(
                delta_com_x=jnp.float32(0.05 * CELL_SIZE),
                turn_motion_threshold=turn_motion_threshold,
            )
        ) == 0
        assert int(
            _significant_step_direction(
                delta_com_x=jnp.float32(0.2 * CELL_SIZE),
                turn_motion_threshold=turn_motion_threshold,
            )
        ) == 1
        assert int(
            _significant_step_direction(
                delta_com_x=jnp.float32(-0.2 * CELL_SIZE),
                turn_motion_threshold=turn_motion_threshold,
            )
        ) == -1

    def test_episode_turn_motion_threshold_scales_with_initial_robot_span(self):
        threshold = _episode_turn_motion_threshold(
            init_min_x=jnp.float32(0.0),
            init_front_x=jnp.float32(1.0 * CELL_SIZE),
        )
        larger = _episode_turn_motion_threshold(
            init_min_x=jnp.float32(0.0),
            init_front_x=jnp.float32(8.0 * CELL_SIZE),
        )

        assert float(threshold) == pytest.approx(0.1 * CELL_SIZE)
        assert float(larger) == pytest.approx(0.02 * 8.0 * CELL_SIZE)

    def test_update_stable_turn_tracking_requires_enduring_motion_after_flip(self):
        stable_direction = jnp.int32(0)
        stable_distance = jnp.float32(0.0)
        stable_eligible = jnp.bool_(False)
        stable_count = jnp.int32(0)
        turn_motion_threshold = _episode_turn_motion_threshold(
            init_min_x=jnp.float32(0.0),
            init_front_x=jnp.float32(1.0 * CELL_SIZE),
        )

        stable_direction, stable_distance, stable_eligible, stable_count = _update_stable_turn_tracking(
            stable_turn_direction=stable_direction,
            stable_turn_distance=stable_distance,
            stable_turn_eligible=stable_eligible,
            stable_turn_count=stable_count,
            delta_com_x=jnp.float32(0.2 * CELL_SIZE),
            turn_motion_threshold=turn_motion_threshold,
        )
        assert int(stable_direction) == 1
        assert float(stable_distance) == pytest.approx(0.2 * CELL_SIZE)
        assert bool(stable_eligible) is False
        assert int(stable_count) == 0

        stable_direction, stable_distance, stable_eligible, stable_count = _update_stable_turn_tracking(
            stable_turn_direction=stable_direction,
            stable_turn_distance=stable_distance,
            stable_turn_eligible=stable_eligible,
            stable_turn_count=stable_count,
            delta_com_x=jnp.float32(-0.2 * CELL_SIZE),
            turn_motion_threshold=turn_motion_threshold,
        )
        assert int(stable_direction) == -1
        assert float(stable_distance) == pytest.approx(0.2 * CELL_SIZE)
        assert bool(stable_eligible) is True
        assert int(stable_count) == 0

        stable_direction, stable_distance, stable_eligible, stable_count = _update_stable_turn_tracking(
            stable_turn_direction=stable_direction,
            stable_turn_distance=stable_distance,
            stable_turn_eligible=stable_eligible,
            stable_turn_count=stable_count,
            delta_com_x=jnp.float32(-0.05 * CELL_SIZE),
            turn_motion_threshold=turn_motion_threshold,
        )
        assert int(stable_direction) == -1
        assert float(stable_distance) == pytest.approx(0.25 * CELL_SIZE)
        assert bool(stable_eligible) is True
        assert int(stable_count) == 0

        stable_direction, stable_distance, stable_eligible, stable_count = _update_stable_turn_tracking(
            stable_turn_direction=stable_direction,
            stable_turn_distance=stable_distance,
            stable_turn_eligible=stable_eligible,
            stable_turn_count=stable_count,
            delta_com_x=jnp.float32(-0.75 * CELL_SIZE),
            turn_motion_threshold=turn_motion_threshold,
        )
        assert int(stable_direction) == -1
        assert float(stable_distance) == pytest.approx(1.0 * CELL_SIZE)
        assert bool(stable_eligible) is False
        assert int(stable_count) == 1

        stable_direction, stable_distance, stable_eligible, stable_count = _update_stable_turn_tracking(
            stable_turn_direction=stable_direction,
            stable_turn_distance=stable_distance,
            stable_turn_eligible=stable_eligible,
            stable_turn_count=stable_count,
            delta_com_x=jnp.float32(-0.3 * CELL_SIZE),
            turn_motion_threshold=turn_motion_threshold,
        )
        assert int(stable_direction) == -1
        assert float(stable_distance) == pytest.approx(1.3 * CELL_SIZE)
        assert bool(stable_eligible) is False
        assert int(stable_count) == 1

    def test_update_stable_turn_tracking_small_opposite_sign_motion_resets_pending_segment(self):
        turn_motion_threshold = _episode_turn_motion_threshold(
            init_min_x=jnp.float32(0.0),
            init_front_x=jnp.float32(1.0 * CELL_SIZE),
        )
        stable_direction = jnp.int32(0)
        stable_distance = jnp.float32(0.0)
        stable_eligible = jnp.bool_(False)
        stable_count = jnp.int32(0)

        for delta in (
            jnp.float32(0.2 * CELL_SIZE),
            jnp.float32(-0.2 * CELL_SIZE),
            jnp.float32(-0.2 * CELL_SIZE),
            jnp.float32(0.05 * CELL_SIZE),
            jnp.float32(-0.2 * CELL_SIZE),
            jnp.float32(-0.2 * CELL_SIZE),
            jnp.float32(-0.2 * CELL_SIZE),
            jnp.float32(-0.2 * CELL_SIZE),
        ):
            stable_direction, stable_distance, stable_eligible, stable_count = _update_stable_turn_tracking(
                stable_turn_direction=stable_direction,
                stable_turn_distance=stable_distance,
                stable_turn_eligible=stable_eligible,
                stable_turn_count=stable_count,
                delta_com_x=delta,
                turn_motion_threshold=turn_motion_threshold,
            )

        assert int(stable_direction) == -1
        assert float(stable_distance) == pytest.approx(0.8 * CELL_SIZE)
        assert bool(stable_eligible) is True
        assert int(stable_count) == 0

    def test_update_stable_turn_tracking_ignores_exact_zero_motion(self):
        stable_direction = jnp.int32(-1)
        stable_distance = jnp.float32(0.25 * CELL_SIZE)
        stable_eligible = jnp.bool_(True)
        stable_count = jnp.int32(0)
        turn_motion_threshold = _episode_turn_motion_threshold(
            init_min_x=jnp.float32(0.0),
            init_front_x=jnp.float32(1.0 * CELL_SIZE),
        )

        stable_direction, stable_distance, stable_eligible, stable_count = _update_stable_turn_tracking(
            stable_turn_direction=stable_direction,
            stable_turn_distance=stable_distance,
            stable_turn_eligible=stable_eligible,
            stable_turn_count=stable_count,
            delta_com_x=jnp.float32(0.0),
            turn_motion_threshold=turn_motion_threshold,
        )

        assert int(stable_direction) == -1
        assert float(stable_distance) == pytest.approx(0.25 * CELL_SIZE)
        assert bool(stable_eligible) is True
        assert int(stable_count) == 0

    def test_update_stable_turn_tracking_caps_count_at_six(self):
        stable_direction = jnp.int32(1)
        stable_distance = jnp.float32(2.0 * CELL_SIZE)
        stable_eligible = jnp.bool_(False)
        stable_count = jnp.int32(6)
        turn_motion_threshold = _episode_turn_motion_threshold(
            init_min_x=jnp.float32(0.0),
            init_front_x=jnp.float32(1.0 * CELL_SIZE),
        )

        stable_direction, stable_distance, stable_eligible, stable_count = _update_stable_turn_tracking(
            stable_turn_direction=stable_direction,
            stable_turn_distance=stable_distance,
            stable_turn_eligible=stable_eligible,
            stable_turn_count=stable_count,
            delta_com_x=jnp.float32(-1.1 * CELL_SIZE),
            turn_motion_threshold=turn_motion_threshold,
        )
        assert int(stable_direction) == -1
        assert float(stable_distance) == pytest.approx(1.1 * CELL_SIZE)
        assert bool(stable_eligible) is False
        assert int(stable_count) == 6


class TestBuildCliffEnv:
    def test_builds_without_error(self, simple_env):
        assert isinstance(simple_env, CliffEnvData)

    def test_grid_dims(self, simple_env):
        assert simple_env.grid_h == 10
        assert simple_env.grid_w == 35
        assert simple_env.render_info.grid_h == 10
        assert simple_env.render_info.grid_w == 35

    def test_death_plane_y(self, simple_env):
        assert simple_env.death_plane_y == pytest.approx((3.0 * CELL_SIZE) - (0.1 * CELL_SIZE))

    def test_init_front_x_positive(self, simple_env):
        assert simple_env.init_front_x > 0.0

    def test_robot_point_mask_has_points(self, simple_env):
        assert jnp.sum(simple_env.robot_point_mask) > 0

    def test_actuator_mask_has_actuators(self, simple_env):
        assert jnp.sum(simple_env.actuator_mask) > 0

    def test_deform_mask_has_cells(self, simple_env):
        assert jnp.sum(simple_env.deform_mask) > 0

    def test_physics_constants_convergence(self, simple_env):
        assert simple_env.constants.actuator_convergence == pytest.approx(0.003)

    def test_physics_constants_reference_dt(self, simple_env):
        assert simple_env.constants.dt == pytest.approx(0.0001)

    def test_fixed_env_threads_floor_stiction_constants(self):
        terrain = build_cliff_terrain(10, 35, cliff_x=15)
        robot = make_robot_body(7, 2)
        env = build_cliff_env(
            terrain,
            robot,
            spawn_x=1,
            spawn_y=4,
            cliff_x=15,
            floor_static_friction_const=0.4,
            floor_stiction_stiffness=20_000.0,
            floor_stiction_switch_speed=0.05,
        )

        assert env.constants.floor_static_friction_const == pytest.approx(0.4)
        assert env.constants.floor_stiction_stiffness == pytest.approx(20_000.0)
        assert env.constants.floor_stiction_switch_speed == pytest.approx(0.05)
        assert bool(env.constants.stiction_enabled) is False

    @pytest.mark.parametrize(
        ("substeps", "expected_dt", "expected_convergence"),
        (
            (30, 0.0001, 0.00300),
            (20, 0.00015, 0.00450),
            (15, 0.0002, 0.00599),
            (12, 0.00025, 0.00748),
        ),
    )
    def test_derive_physics_params_autoscales_reference_convergence(
        self,
        substeps,
        expected_dt,
        expected_convergence,
    ):
        dt, convergence = derive_physics_params(substeps, 0.003)
        assert dt == pytest.approx(expected_dt)
        assert convergence == pytest.approx(expected_convergence, rel=1e-3, abs=1e-5)

    def test_lower_probe_ledge_lowers_death_plane(self):
        terrain, metadata = build_bilateral_goal_cliff_terrain(
            10,
            30,
            symmetric_distance=6,
            goal_side="left",
            plateau_width=10,
            cliff_probe_gap_width=1,
            cliff_probe_ledge_width=3,
        )
        robot = make_robot_body(6, 2)
        env = build_cliff_env(
            terrain,
            robot,
            spawn_x=12,
            spawn_y=4,
            cliff_x=int(metadata["platform_end_x"]),
            task_metadata=metadata,
        )

        assert env.death_plane_y == pytest.approx((2.0 * CELL_SIZE) - (0.1 * CELL_SIZE))

    def test_lower_probe_ledge_without_gap_lowers_death_plane(self):
        terrain, metadata = build_bilateral_goal_cliff_terrain(
            10,
            30,
            symmetric_distance=6,
            goal_side="left",
            plateau_width=10,
            cliff_probe_gap_width=0,
            cliff_probe_ledge_width=3,
        )
        robot = make_robot_body(6, 2)
        env = build_cliff_env(
            terrain,
            robot,
            spawn_x=12,
            spawn_y=4,
            cliff_x=int(metadata["platform_end_x"]),
            task_metadata=metadata,
        )

        assert env.death_plane_y == pytest.approx((2.0 * CELL_SIZE) - (0.1 * CELL_SIZE))

    def test_gap_and_cliff_templates_keep_vertical_side_edges(self):
        terrain = build_cliff_gap_terrain(10, 35, cliff_x=15, gaps=[(10, 2)])
        template = build_static_collider_template(
            terrain,
            cliff_x=15,
            canvas_h=4,
            canvas_w=10,
            spawn_x=1,
            spawn_y=4,
        )
        aabb_min = np.asarray(template.static_collider_data.cell_aabb_min)
        aabb_max = np.asarray(template.static_collider_data.cell_aabb_max)
        surface = np.asarray(template.static_collider_data.surface_edge_mask)
        platform_band = np.isclose(aabb_min[:, 1], 3.0 * CELL_SIZE)

        gap_left_wall = platform_band & np.isclose(aabb_max[:, 0], 10.0 * CELL_SIZE)
        gap_right_wall = platform_band & np.isclose(aabb_min[:, 0], 12.0 * CELL_SIZE)
        cliff_wall = platform_band & np.isclose(aabb_max[:, 0], 16.0 * CELL_SIZE)

        assert np.any(gap_left_wall & surface[:, 1])
        assert np.any(gap_right_wall & surface[:, 3])
        assert np.any(cliff_wall & surface[:, 1])

    def test_deformation_info_is_world_aligned(self, simple_env):
        expected_cells = simple_env.grid_h * simple_env.grid_w
        assert simple_env.deformation_info.n_robot_voxels == expected_cells
        assert simple_env.deformation_info.voxel_horiz_springs.shape == (expected_cells, 2)
        assert simple_env.deformation_info.voxel_vert_springs.shape == (expected_cells, 2)

    def test_comb_sparse_body_builds_without_shape_mismatch(self, comb_env):
        assert isinstance(comb_env, CliffEnvData)
        expected_cells = comb_env.grid_h * comb_env.grid_w
        assert comb_env.deformation_info.voxel_horiz_springs.shape == (expected_cells, 2)
        assert comb_env.deformation_info.voxel_vert_springs.shape == (expected_cells, 2)
        assert int(jnp.sum(comb_env.deform_mask)) == 11

    def test_comb_7x3_sparse_body_builds_without_shape_mismatch(self, comb_7x3_env):
        assert isinstance(comb_7x3_env, CliffEnvData)
        expected_cells = comb_7x3_env.grid_h * comb_7x3_env.grid_w
        assert comb_7x3_env.deformation_info.voxel_horiz_springs.shape == (expected_cells, 2)
        assert comb_7x3_env.deformation_info.voxel_vert_springs.shape == (expected_cells, 2)
        assert int(jnp.sum(comb_7x3_env.deform_mask)) == 18

    def test_compact_actuator_indices_are_world_slots(self, simple_env):
        actuator_slots = np.flatnonzero(np.asarray(simple_env.actuator_mask))
        h_count = int(np.asarray(simple_env.per_axis_info.h_compact_count))
        v_count = int(np.asarray(simple_env.per_axis_info.v_compact_count))
        h_indices = np.asarray(simple_env.per_axis_info.h_compact_cell_indices[:h_count])
        v_indices = np.asarray(simple_env.per_axis_info.v_compact_cell_indices[:v_count])

        assert np.array_equal(h_indices, actuator_slots)
        assert np.array_equal(v_indices, actuator_slots)

    def test_stack_eval_envs_respects_explicit_static_capacity(self, simple_env, short_env):
        capacity = infer_static_collider_capacity([simple_env, short_env])
        simple_batch = stack_eval_envs([simple_env], static_capacity=capacity)
        short_batch = stack_eval_envs([short_env], static_capacity=capacity)

        assert simple_batch.static_collider_data.point_positions.shape == (
            1,
            capacity.max_static_points,
            2,
        )
        assert short_batch.static_collider_data.point_positions.shape == (
            1,
            capacity.max_static_points,
            2,
        )
        assert simple_batch.static_collider_data.cell_vertices.shape == (
            1,
            capacity.max_static_cells,
            4,
        )
        assert short_batch.static_collider_data.cell_vertices.shape == (
            1,
            capacity.max_static_cells,
            4,
        )
        assert simple_batch.static_collider_data.cell_aabb_min.shape == (
            1,
            capacity.max_static_cells,
            2,
        )
        assert short_batch.static_collider_data.cell_aabb_max.shape == (
            1,
            capacity.max_static_cells,
            2,
        )
        assert simple_batch.static_collider_data.column_cell_indices.shape == (
            1,
            capacity.max_grid_columns,
            capacity.max_cells_per_column,
        )
        assert simple_batch.static_collision_worklist.triple_i.shape == short_batch.static_collision_worklist.triple_i.shape
        expected_static_slots = capacity.max_static_cells
        assert simple_batch.static_collision_worklist.triple_active.shape == (
            1,
            simple_batch.collision_data.boxel_mask.shape[1] * expected_static_slots * 4,
        )


class TestResetEpisode:
    def test_reset_shapes(self, simple_env):
        key = jax.random.PRNGKey(0)
        state = reset_episode(simple_env, key, ctrnn_hidden=8, ctrnn_grid_hidden=12)
        assert isinstance(state, CliffEpisodeState)
        assert state.nca_state.shape == (2, 7, 19)  # 7 base + 12 hidden
        assert state.ctrnn_carry.shape == (2, 7, 8)
        assert state.step_count == 0
        assert not state.done
        assert not state.touched_death_plane

    def test_peak_front_x_initialized(self, simple_env):
        key = jax.random.PRNGKey(0)
        state = reset_episode(simple_env, key, ctrnn_hidden=8, ctrnn_grid_hidden=12)
        assert state.peak_front_x == pytest.approx(simple_env.init_front_x)

    def test_fixed_hidden_init_scale_seeds_only_hidden_grid_channels(self, simple_env):
        key = jax.random.PRNGKey(0)
        state = reset_episode(
            simple_env,
            key,
            ctrnn_hidden=8,
            ctrnn_grid_hidden=12,
            fixed_hidden_init_scale=0.05,
        )
        gh_start = base_channel_count(ou_enabled=True)
        hidden = state.nca_state[:, :, gh_start:gh_start + 12]

        assert jnp.any(hidden != 0.0)
        assert jnp.all(jnp.abs(hidden) <= 0.05 + 1e-6)
        assert jnp.allclose(state.nca_state[:, :, :gh_start], 0.0)
        assert jnp.allclose(state.ctrnn_carry, 0.0)

    def test_fixed_hidden_init_scale_is_reproducible_for_same_key(self, simple_env):
        key = jax.random.PRNGKey(7)
        state_a = reset_episode(
            simple_env,
            key,
            ctrnn_hidden=8,
            ctrnn_grid_hidden=12,
            fixed_hidden_init_scale=0.05,
        )
        state_b = reset_episode(
            simple_env,
            key,
            ctrnn_hidden=8,
            ctrnn_grid_hidden=12,
            fixed_hidden_init_scale=0.05,
        )

        assert jnp.allclose(state_a.nca_state, state_b.nca_state)
        assert jnp.allclose(state_a.ctrnn_carry, state_b.ctrnn_carry)

    def test_episode_result_normalizes_speed_by_env_steps(self, simple_env):
        key = jax.random.PRNGKey(0)
        state = reset_episode(simple_env, key, ctrnn_hidden=8, ctrnn_grid_hidden=12)
        final_state = state._replace(
            step_count=jnp.int32(2),
            steps_alive=jnp.int32(6),
            cumulative_speed=jnp.float32(0.3),
            first_significant_direction=jnp.int32(-1),
        )
        result = _episode_result_from_state(final_state, simple_env, max_steps=10)
        assert result.mean_speed == pytest.approx(0.15)
        assert int(result.first_significant_direction) == -1

    def test_mace_init_mass_seeds_hidden_channels(self, simple_env):
        key = jax.random.PRNGKey(0)
        state = reset_episode(
            simple_env,
            key,
            ctrnn_hidden=8,
            ctrnn_grid_hidden=12,
            mace_mode="cross_channel",
            mace_init_mass=1.25,
        )
        gh_start = base_channel_count(ou_enabled=True)
        hidden = state.nca_state[:, :, gh_start:gh_start + 12]
        assert jnp.allclose(hidden, 1.25)


class TestRunEpisode:
    def test_short_episode_completes(self, simple_env):
        """Run 3 steps to verify the loop works."""
        key = jax.random.PRNGKey(42)
        params = init_ctrnn_params(key, ctrnn_hidden=8, ctrnn_grid_hidden=12)
        alive_mask = jnp.ones((2, 7), dtype=jnp.bool_)
        result = run_episode(
            simple_env, params, alive_mask, key,
            max_steps=3,
            physics_substeps=6,      # fast: 6 substeps
            sense_act_phases=3,      # 3 phases → 2 substeps each
            ctrnn_hidden=8,
            ctrnn_grid_hidden=12,
        )
        assert isinstance(result, EpisodeResult)
        assert not jnp.isnan(result.peak_front_x)
        assert not jnp.isnan(result.mean_activation)
        assert not jnp.isnan(result.mean_speed)
        assert result.steps_alive >= 0

    def test_state_space_trace_uses_fixed_deformation_shape(self, simple_env):
        key = jax.random.PRNGKey(43)
        params = init_ctrnn_params(key, ctrnn_hidden=8, ctrnn_grid_hidden=12)
        alive_mask = jnp.ones((2, 7), dtype=jnp.bool_)

        _, trace = env_module.run_episode_with_state_space_trace(
            simple_env,
            params,
            alive_mask,
            key,
            max_steps=3,
            physics_substeps=6,
            sense_act_phases=3,
            ctrnn_hidden=8,
            ctrnn_grid_hidden=12,
        )

        assert trace.phase.deformation_inputs.shape == (
            3,
            3,
            simple_env.deform_mask.shape[0],
            4,
        )

    def test_peak_front_x_nondecreasing(self, simple_env):
        """Peak front_x should never decrease."""
        key = jax.random.PRNGKey(99)
        params = init_ctrnn_params(key, ctrnn_hidden=8, ctrnn_grid_hidden=12)
        alive_mask = jnp.ones((2, 7), dtype=jnp.bool_)
        result = run_episode(
            simple_env, params, alive_mask, key,
            max_steps=5, physics_substeps=6, sense_act_phases=3,
            ctrnn_hidden=8, ctrnn_grid_hidden=12,
        )
        # Peak should be >= initial.
        assert result.peak_front_x >= result.init_front_x - 1e-5

    def test_determinism(self, simple_env):
        """Same seed → same result."""
        key = jax.random.PRNGKey(42)
        params = init_ctrnn_params(key, ctrnn_hidden=8, ctrnn_grid_hidden=12)
        alive_mask = jnp.ones((2, 7), dtype=jnp.bool_)
        r1 = run_episode(
            simple_env, params, alive_mask, key,
            max_steps=3, physics_substeps=6, sense_act_phases=3,
            ctrnn_hidden=8, ctrnn_grid_hidden=12,
        )
        r2 = run_episode(
            simple_env, params, alive_mask, key,
            max_steps=3, physics_substeps=6, sense_act_phases=3,
            ctrnn_hidden=8, ctrnn_grid_hidden=12,
        )
        assert r1.peak_front_x == r2.peak_front_x
        assert r1.mean_activation == r2.mean_activation

    def test_airborne_terminates_episode_early(self, simple_env, monkeypatch):
        original = env_module._robot_kinematics

        def _forced_airborne(sim_state, robot_point_mask):
            base = original(sim_state, robot_point_mask)
            return env_module.RobotKinematics(
                min_x=base.min_x,
                front_x=base.front_x,
                min_y=base.min_y,
                max_y=base.max_y,
                com_x=base.com_x,
                com_y=jnp.float32(simple_env.init_com_y + (2.25 * CELL_SIZE)),
            )

        monkeypatch.setattr(env_module, "_robot_kinematics", _forced_airborne)
        key = jax.random.PRNGKey(7)
        params = init_ctrnn_params(key, ctrnn_hidden=8, ctrnn_grid_hidden=12)
        alive_mask = jnp.ones((2, 7), dtype=jnp.bool_)
        result = run_episode(
            simple_env,
            params,
            alive_mask,
            key,
            max_steps=5,
            physics_substeps=6,
            sense_act_phases=3,
            ctrnn_hidden=8,
            ctrnn_grid_hidden=12,
        )
        assert bool(result.went_airborne)
        assert int(result.steps_alive) < 5 * 3

    def test_subthreshold_leap_does_not_trigger_airborne_kill(self, simple_env, monkeypatch):
        original = env_module._robot_kinematics

        def _nearly_airborne(sim_state, robot_point_mask):
            base = original(sim_state, robot_point_mask)
            return env_module.RobotKinematics(
                min_x=base.min_x,
                front_x=base.front_x,
                min_y=base.min_y,
                max_y=base.max_y,
                com_x=base.com_x,
                com_y=jnp.float32(simple_env.init_com_y + (1.25 * CELL_SIZE)),
            )

        monkeypatch.setattr(env_module, "_robot_kinematics", _nearly_airborne)
        key = jax.random.PRNGKey(11)
        params = init_ctrnn_params(key, ctrnn_hidden=8, ctrnn_grid_hidden=12)
        alive_mask = jnp.ones((2, 7), dtype=jnp.bool_)
        result = run_episode(
            simple_env,
            params,
            alive_mask,
            key,
            max_steps=5,
            physics_substeps=6,
            sense_act_phases=3,
            ctrnn_hidden=8,
            ctrnn_grid_hidden=12,
        )
        assert not bool(result.went_airborne)

    def test_soft_loft_metrics_track_subcritical_lift(self, simple_env, monkeypatch):
        original = env_module._robot_kinematics

        def _soft_lofted(sim_state, robot_point_mask):
            base = original(sim_state, robot_point_mask)
            return env_module.RobotKinematics(
                min_x=base.min_x,
                front_x=base.front_x,
                min_y=base.min_y,
                max_y=base.max_y,
                com_x=base.com_x,
                com_y=jnp.float32(simple_env.init_com_y + (0.25 * CELL_SIZE)),
            )

        monkeypatch.setattr(env_module, "_robot_kinematics", _soft_lofted)
        key = jax.random.PRNGKey(13)
        params = init_ctrnn_params(key, ctrnn_hidden=8, ctrnn_grid_hidden=12)
        alive_mask = jnp.ones((2, 7), dtype=jnp.bool_)
        result = run_episode(
            simple_env,
            params,
            alive_mask,
            key,
            max_steps=5,
            physics_substeps=6,
            sense_act_phases=3,
            ctrnn_hidden=8,
            ctrnn_grid_hidden=12,
            soft_loft_threshold_cells=0.2,
        )
        assert not bool(result.went_airborne)
        assert float(result.lofted_fraction) == pytest.approx(1.0)
        assert float(result.mean_lofted_excess_cells) == pytest.approx(0.05, abs=1e-5)
        assert float(result.peak_com_y_delta_cells) == pytest.approx(0.25, abs=1e-5)

    def test_drop_below_platform_bottom_counts_as_death(self, simple_env, monkeypatch):
        original = env_module._robot_kinematics

        def _forced_drop(sim_state, robot_point_mask):
            base = original(sim_state, robot_point_mask)
            return env_module.RobotKinematics(
                min_x=base.min_x,
                front_x=base.front_x,
                min_y=jnp.float32((3.0 * CELL_SIZE) - 0.02),
                max_y=base.max_y,
                com_x=base.com_x,
                com_y=base.com_y,
            )

        monkeypatch.setattr(env_module, "_robot_kinematics", _forced_drop)
        key = jax.random.PRNGKey(17)
        params = init_ctrnn_params(key, ctrnn_hidden=8, ctrnn_grid_hidden=12)
        alive_mask = jnp.ones((2, 7), dtype=jnp.bool_)
        result = run_episode(
            simple_env,
            params,
            alive_mask,
            key,
            max_steps=5,
            physics_substeps=6,
            sense_act_phases=3,
            ctrnn_hidden=8,
            ctrnn_grid_hidden=12,
        )
        assert bool(result.touched_death_plane)
        assert int(result.steps_alive) < 5 * 3

    def test_ablation_runs(self, simple_env):
        """Deformation OFF should run without error."""
        key = jax.random.PRNGKey(42)
        params = init_ctrnn_params(key, ctrnn_hidden=8, ctrnn_grid_hidden=12)
        alive_mask = jnp.ones((2, 7), dtype=jnp.bool_)
        result = run_episode(
            simple_env, params, alive_mask, key,
            max_steps=3, physics_substeps=6, sense_act_phases=3,
            ctrnn_hidden=8, ctrnn_grid_hidden=12,
            ablate_deformation=True,
        )
        assert not jnp.isnan(result.peak_front_x)

    def test_cliff_x_in_result(self, simple_env):
        key = jax.random.PRNGKey(42)
        params = init_ctrnn_params(key, ctrnn_hidden=8, ctrnn_grid_hidden=12)
        alive_mask = jnp.ones((2, 7), dtype=jnp.bool_)
        result = run_episode(
            simple_env, params, alive_mask, key,
            max_steps=3, physics_substeps=6, sense_act_phases=3,
            ctrnn_hidden=8, ctrnn_grid_hidden=12,
        )
        assert int(result.cliff_x) == 15

    def test_phase_averages_micro_actions_before_physics(self, simple_env, monkeypatch):
        key = jax.random.PRNGKey(42)
        params = init_ctrnn_params(key, ctrnn_hidden=8, ctrnn_grid_hidden=12)
        alive_mask = jnp.ones((2, 7), dtype=jnp.bool_)
        h_values = jnp.asarray([0.6, 1.0, 1.4], dtype=jnp.float32)
        v_values = jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float32)

        def _stub_nca_step(
            nca_state,
            ctrnn_carry,
            deformation,
            deform_mask,
            actuator_mask,
            alive_mask,
            params,
            key,
            robot_offset_y,
            robot_offset_x,
            h,
            w,
            **kwargs,
        ):
            del (
                deformation,
                deform_mask,
                actuator_mask,
                alive_mask,
                params,
                key,
                robot_offset_y,
                robot_offset_x,
                kwargs,
            )
            micro_idx = jnp.clip(
                ctrnn_carry[0, 0, 0].astype(jnp.int32),
                0,
                jnp.int32(h_values.shape[0] - 1),
            )
            h_action = jnp.full((h * w,), h_values[micro_idx], dtype=jnp.float32)
            v_action = jnp.full((h * w,), v_values[micro_idx], dtype=jnp.float32)
            next_ctrnn_carry = ctrnn_carry.at[0, 0, 0].set(
                jnp.asarray(micro_idx + 1, dtype=ctrnn_carry.dtype)
            )
            return nca_state, next_ctrnn_carry, h_action, v_action, NCAStepDiagnostics(
                controller_hidden_output_max_abs=jnp.float32(0.0),
                action_output_max_abs=jnp.float32(0.0),
            )

        def _record_applied_goals(state, info, h_action_compact, v_action_compact):
            del info
            spring_rest_length = (
                state.spring_rest_length
                .at[0].set(h_action_compact[0])
                .at[1].set(v_action_compact[0])
            )
            return state._replace(spring_rest_length=spring_rest_length)

        def _identity_physics(sim_state, *args, **kwargs):
            del args, kwargs
            return sim_state

        monkeypatch.setattr(env_module, "nca_step_with_diagnostics", _stub_nca_step)
        monkeypatch.setattr(env_module, "set_per_axis_goals_compact", _record_applied_goals)
        monkeypatch.setattr(env_module, "physics_substep_dynamic_broadphase", _identity_physics)

        avg_result, avg_trace = env_module.run_episode_with_trace(
            simple_env,
            params,
            alive_mask,
            key,
            max_steps=1,
            physics_substeps=1,
            sense_act_phases=1,
            nca_steps_per_phase=3,
            ctrnn_hidden=8,
            ctrnn_grid_hidden=12,
        )
        np.testing.assert_allclose(
            np.asarray(avg_trace.spring_rest_length[0, :2]),
            np.asarray([1.0, 1.0], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        assert float(avg_result.mean_activation) == pytest.approx(0.0, abs=1e-6)
        assert float(avg_result.mean_h_activation) == pytest.approx(0.0, abs=1e-6)
        assert float(avg_result.mean_v_activation) == pytest.approx(0.0, abs=1e-6)

        h_count = float(np.asarray(simple_env.per_axis_info.h_compact_count))
        v_count = float(np.asarray(simple_env.per_axis_info.v_compact_count))
        expected_single_h = (2.0 * abs(0.6 - 1.0) * h_count) / max(h_count + v_count, 1.0)
        single_result, single_trace = env_module.run_episode_with_trace(
            simple_env,
            params,
            alive_mask,
            key,
            max_steps=1,
            physics_substeps=1,
            sense_act_phases=1,
            nca_steps_per_phase=1,
            ctrnn_hidden=8,
            ctrnn_grid_hidden=12,
        )
        np.testing.assert_allclose(
            np.asarray(single_trace.spring_rest_length[0, :2]),
            np.asarray([0.6, 1.0], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        assert float(single_result.mean_activation) == pytest.approx(expected_single_h, abs=1e-6)
        assert float(single_result.mean_h_activation) == pytest.approx(expected_single_h, abs=1e-6)
        assert float(single_result.mean_v_activation) == pytest.approx(0.0, abs=1e-6)

    def test_grown_env_episode_runs(self):
        terrain = build_cliff_terrain(10, 35, cliff_x=15)
        grid_data = precompute_shared_grid_data(10, 35)
        template = build_static_collider_template(
            terrain,
            cliff_x=15,
            canvas_h=4,
            canvas_w=10,
            spawn_x=1,
            spawn_y=4,
        )
        alive_mask = jnp.zeros((4, 10), dtype=jnp.bool_).at[1:3, 2:5].set(True)
        robot_morph = jnp.where(alive_mask, jnp.int32(CONTRACTILE), jnp.int32(0))
        env = make_cliff_env_split_colliders(
            robot_morph,
            grid_data,
            1,
            4,
            cliff_x=15,
            static_collider_template=template.static_collider_data,
            death_plane_y=template.death_plane_y,
            physics_substeps=6,
            actuator_convergence=0.003,
        )
        params = init_ctrnn_params(
            jax.random.PRNGKey(7),
            ctrnn_hidden=8,
            ctrnn_grid_hidden=12,
            growth_enabled=True,
        )
        frozen_identity = alive_mask.astype(jnp.float32)
        result = run_episode(
            env,
            params,
            alive_mask,
            jax.random.PRNGKey(11),
            max_steps=3,
            physics_substeps=6,
            sense_act_phases=3,
            ctrnn_hidden=8,
            ctrnn_grid_hidden=12,
            growth_enabled=True,
            frozen_identity=frozen_identity,
        )
        assert isinstance(result, EpisodeResult)
        assert not jnp.isnan(result.peak_front_x)

    def test_grown_split_env_threads_floor_stiction_constants(self):
        terrain = build_cliff_terrain(10, 35, cliff_x=15)
        grid_data = precompute_shared_grid_data(10, 35)
        template = build_static_collider_template(
            terrain,
            cliff_x=15,
            canvas_h=4,
            canvas_w=10,
            spawn_x=1,
            spawn_y=4,
        )
        alive_mask = jnp.zeros((4, 10), dtype=jnp.bool_).at[1:3, 2:5].set(True)
        robot_morph = jnp.where(alive_mask, jnp.int32(CONTRACTILE), jnp.int32(0))
        env = make_cliff_env_split_colliders(
            robot_morph,
            grid_data,
            1,
            4,
            cliff_x=15,
            static_collider_template=template.static_collider_data,
            death_plane_y=template.death_plane_y,
            physics_substeps=6,
            actuator_convergence=0.003,
            floor_static_friction_const=0.4,
            floor_stiction_stiffness=20_000.0,
            floor_stiction_switch_speed=0.05,
        )

        assert env.constants.floor_static_friction_const == pytest.approx(0.4)
        assert env.constants.floor_stiction_stiffness == pytest.approx(20_000.0)
        assert env.constants.floor_stiction_switch_speed == pytest.approx(0.05)
        assert bool(env.constants.stiction_enabled) is False

    def test_grown_dynamic_core_attach_matches_direct_builder(self):
        terrain = build_cliff_terrain(10, 35, cliff_x=15)
        grid_data = precompute_shared_grid_data(10, 35)
        template = build_static_collider_template(
            terrain,
            cliff_x=15,
            canvas_h=4,
            canvas_w=10,
            spawn_x=1,
            spawn_y=4,
        )
        alive_mask = jnp.zeros((4, 10), dtype=jnp.bool_).at[1:3, 2:5].set(True)
        robot_morph = jnp.where(alive_mask, jnp.int32(CONTRACTILE), jnp.int32(0))

        direct_env = make_cliff_env_split_colliders(
            robot_morph,
            grid_data,
            1,
            4,
            cliff_x=15,
            static_collider_template=template.static_collider_data,
            death_plane_y=template.death_plane_y,
            physics_substeps=6,
            actuator_convergence=0.003,
        )
        env_core = build_cliff_eval_dynamic_core(
            robot_morph,
            grid_data,
            1,
            4,
            physics_substeps=6,
            actuator_convergence=0.003,
        )
        attached_env = attach_static_template_to_eval_env(
            env_core,
            template.static_collider_data,
            cliff_x=15,
            death_plane_y=template.death_plane_y,
        )

        np.testing.assert_allclose(
            np.asarray(attached_env.sim_state.positions),
            np.asarray(direct_env.sim_state.positions),
        )
        np.testing.assert_array_equal(
            np.asarray(attached_env.actuator_mask),
            np.asarray(direct_env.actuator_mask),
        )
        np.testing.assert_array_equal(
            np.asarray(attached_env.collision_data.boxel_mask),
            np.asarray(direct_env.collision_data.boxel_mask),
        )
        np.testing.assert_array_equal(
            np.asarray(attached_env.static_collision_worklist.triple_i),
            np.asarray(direct_env.static_collision_worklist.triple_i),
        )
        np.testing.assert_array_equal(
            np.asarray(attached_env.static_collision_worklist.triple_j),
            np.asarray(direct_env.static_collision_worklist.triple_j),
        )
        np.testing.assert_array_equal(
            np.asarray(attached_env.static_collision_worklist.triple_k),
            np.asarray(direct_env.static_collision_worklist.triple_k),
        )
        np.testing.assert_array_equal(
            np.asarray(attached_env.static_collision_worklist.triple_active),
            np.asarray(direct_env.static_collision_worklist.triple_active),
        )
        assert attached_env.constants.dt == pytest.approx(direct_env.constants.dt)
        assert int(attached_env.cliff_x) == int(direct_env.cliff_x)
        assert float(attached_env.death_plane_y) == pytest.approx(float(direct_env.death_plane_y))

    def test_build_grown_dynamic_render_info_uses_world_grid_indices(self):
        grid_data = precompute_shared_grid_data(10, 35)
        alive_mask = jnp.zeros((4, 10), dtype=jnp.bool_).at[1:3, 2:5].set(True)
        robot_morph = jnp.where(alive_mask, jnp.int32(CONTRACTILE), jnp.int32(0))
        env_core = build_cliff_eval_dynamic_core(
            robot_morph,
            grid_data,
            1,
            4,
            physics_substeps=6,
            actuator_convergence=0.003,
        )
        render_info = build_grown_dynamic_render_info(
            robot_morph,
            grid_data,
            spawn_x=1,
            spawn_y=4,
            spring_init_rest_length=np.asarray(env_core.sim_state.spring_init_rest_length),
        )

        assert render_info.grid_h == 10
        assert render_info.grid_w == 35
        assert render_info.cell_vertices.shape[0] == int(np.sum(np.asarray(alive_mask)))
        assert np.all(render_info.cell_vertex_count == 4)
        assert render_info.robot_point_indices.shape[0] == int(np.asarray(env_core.robot_point_mask).sum())
        assert np.max(render_info.cell_vertices, initial=-1) < int(np.asarray(env_core.sim_state.positions).shape[0])
        np.testing.assert_array_equal(
            render_info.robot_point_indices,
            np.flatnonzero(np.asarray(env_core.robot_point_mask)),
        )
        assert np.all(render_info.actuator_cell_indices >= 0)
        assert np.all(render_info.cell_h_spring_pairs >= 0)
        assert np.all(render_info.cell_v_spring_pairs >= 0)
