from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf


SCRIPT_DIR = Path(__file__).resolve().parent
S4_ROOT = SCRIPT_DIR.parent
REPO_ROOT = S4_ROOT.parent
SENSING_EDGES_ROOT = REPO_ROOT / "sensing_edges"
DEFAULT_BUNDLE_ROOT = S4_ROOT / "results" / "bin_03_rank_00_cell_6-0-3_flat_0423"


def ensure_import_path() -> None:
    path = str(SENSING_EDGES_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_controller_params(path: Path):
    ensure_import_path()
    from sensing_edges.nca import CTRNNParams

    with np.load(path, allow_pickle=False) as data:
        return CTRNNParams(
            w_input=jnp.asarray(np.asarray(data["w_input"], dtype=np.float32)),
            w_rec=jnp.asarray(np.asarray(data["w_rec"], dtype=np.float32)),
            w_output=jnp.asarray(np.asarray(data["w_output"], dtype=np.float32)),
            bias=jnp.asarray(np.asarray(data["bias"], dtype=np.float32)),
            log_tau=jnp.asarray(np.asarray(data["log_tau"], dtype=np.float32)),
        )


def load_cfg(config_snapshot_path: Path, *, max_steps: int) -> Any:
    ensure_import_path()
    from sensing_edges.runtime_config import compose_experiment_config, default_config_dir

    snapshot_cfg = OmegaConf.load(config_snapshot_path)
    preset_name = (
        str(snapshot_cfg.get("preset_id")).strip()
        if snapshot_cfg.get("preset_id") is not None
        else None
    )
    base_cfg = compose_experiment_config(
        preset=preset_name,
        config_dir=default_config_dir(),
    )
    cfg = OmegaConf.merge(
        OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False)),
        snapshot_cfg,
    )
    OmegaConf.set_struct(cfg, False)
    cfg.train.max_steps = int(max_steps)
    cfg.eval.post_run.active_deformation_condition = "deformation_off"
    cfg.eval.post_run.active_signaling_condition = "intact"
    return cfg


def build_free_space_env(cfg) -> tuple[Any, Any, jnp.ndarray]:
    ensure_import_path()
    from jax_evogym.jax_utils import precompute_grid
    from sensing_edges.env import build_cliff_dynamic_core, build_cliff_eval_dynamic_core
    from sensing_edges.runtime_config import resolved_spawn_x
    from sensing_edges.terrain import fixed_body_alive_mask, make_robot_body

    robot_morph = make_robot_body(
        int(cfg.terrain.robot_w),
        int(cfg.terrain.robot_h),
        shape=str(cfg.body.get("shape", "rectangle")),
    )
    spawn_x = int(resolved_spawn_x(cfg))
    spawn_y = int(cfg.terrain.spawn_y)
    physics_substeps = int(cfg.sim.physics_substeps)
    actuator_convergence = float(cfg.sim.actuator_convergence)
    spring_stiffness_scale = float(cfg.sim.spring_stiffness_scale)

    dynamic_core = build_cliff_dynamic_core(
        robot_morph,
        spawn_x=spawn_x,
        spawn_y=spawn_y,
        physics_substeps=physics_substeps,
        actuator_convergence=actuator_convergence,
        spring_stiffness_scale=spring_stiffness_scale,
        floor_static_friction_const=0.0,
        floor_stiction_stiffness=0.0,
        floor_stiction_switch_speed=0.0,
    )
    env = build_cliff_eval_dynamic_core(
        jnp.asarray(robot_morph, dtype=jnp.int32),
        precompute_grid(int(cfg.terrain.grid_h), int(cfg.terrain.grid_w)),
        spawn_x,
        spawn_y,
        physics_substeps=physics_substeps,
        actuator_convergence=actuator_convergence,
        spring_stiffness_scale=spring_stiffness_scale,
        floor_static_friction_const=0.0,
        floor_stiction_stiffness=0.0,
        floor_stiction_switch_speed=0.0,
    )

    free_space_constants = env.constants._replace(
        gravity=0.0,
        viscous_drag=0.0,
        collision_const_ground=0.0,
        ground_vel_damping=0.0,
        ground_friction_const=0.0,
        ground_static_friction_const=0.0,
        static_friction_const=0.0,
        terrain_stiction_stiffness=0.0,
        ground_stiction_stiffness=0.0,
        terrain_stiction_switch_speed=0.0,
        ground_stiction_switch_speed=0.0,
        floor_static_friction_const=0.0,
        floor_stiction_stiffness=0.0,
        floor_stiction_switch_speed=0.0,
        stiction_enabled=False,
        diagnostics_enabled=False,
    )
    env = env._replace(
        constants=free_space_constants,
        death_plane_y=jnp.float32(-1.0e6),
        task_mode=jnp.int32(1),
        goal_direction=jnp.int32(0),
        cliff_side=jnp.int32(0),
        goal_distance=jnp.int32(0),
        cliff_distance=jnp.int32(0),
        goal_block_x=jnp.int32(-100000),
        goal_block_y=jnp.int32(-100000),
        goal_x=jnp.float32(-1.0),
        goal_object_id=jnp.int32(0),
        cliff_probe_ledge_width=jnp.int32(0),
        cliff_probe_ledge_cell_x=jnp.int32(-1),
        cliff_probe_ledge_cell_y=jnp.int32(-1),
        cliff_x=jnp.int32(0),
    )
    alive_mask = jnp.asarray(fixed_body_alive_mask(robot_morph), dtype=jnp.bool_)
    return env, dynamic_core.render_info, alive_mask


def build_flat_ground_env(
    cfg,
    *,
    ground_width: int = 20,
    ground_clearance_cells: float = 0.1,
) -> tuple[Any, Any, jnp.ndarray]:
    ensure_import_path()
    from jax_evogym.constants import CELL_SIZE, EMPTY, FIXED
    from sensing_edges.env import assemble_cliff_env, build_cliff_dynamic_core, build_cliff_terrain_assets
    from sensing_edges.terrain import fixed_body_alive_mask, make_robot_body

    robot_morph = make_robot_body(
        int(cfg.terrain.robot_w),
        int(cfg.terrain.robot_h),
        shape=str(cfg.body.get("shape", "rectangle")),
    )
    robot_h, robot_w = robot_morph.shape
    grid_h = int(cfg.terrain.grid_h)
    grid_w = int(cfg.terrain.grid_w)
    platform_y = int(cfg.terrain.get("platform_y", cfg.terrain.get("plateau_height", 3)))
    ground_width = int(ground_width)
    if ground_width <= 0:
        raise ValueError("ground_width must be positive.")
    if ground_width > grid_w:
        raise ValueError("ground_width must not exceed grid width.")
    if robot_w > ground_width:
        raise ValueError("ground_width must fit the fixed robot width.")

    platform_start = int((grid_w - ground_width) // 2)
    platform_end = int(platform_start + ground_width - 1)
    spawn_x = int(platform_start + (ground_width - robot_w) // 2)
    spawn_y = int(platform_y + 1)

    terrain = np.full((grid_h, grid_w), EMPTY, dtype=np.int32)
    terrain[platform_y, platform_start : platform_end + 1] = FIXED

    physics_substeps = int(cfg.sim.physics_substeps)
    actuator_convergence = float(cfg.sim.actuator_convergence)
    spring_stiffness_scale = float(cfg.sim.spring_stiffness_scale)
    floor_static_friction_const = float(cfg.sim.floor_static_friction_const)
    floor_stiction_stiffness = float(cfg.sim.floor_stiction_stiffness)
    floor_stiction_switch_speed = float(cfg.sim.floor_stiction_switch_speed)

    dynamic_core = build_cliff_dynamic_core(
        robot_morph,
        spawn_x=spawn_x,
        spawn_y=spawn_y,
        physics_substeps=physics_substeps,
        actuator_convergence=actuator_convergence,
        spring_stiffness_scale=spring_stiffness_scale,
        floor_static_friction_const=floor_static_friction_const,
        floor_stiction_stiffness=floor_stiction_stiffness,
        floor_stiction_switch_speed=floor_stiction_switch_speed,
    )
    terrain_assets = build_cliff_terrain_assets(
        terrain,
        robot_morph,
        spawn_x,
        spawn_y,
        cliff_x=platform_end,
        task_metadata={
            "task_mode": "legacy_cliff",
            "goal_direction": 0,
            "cliff_side": 0,
            "goal_distance": 0,
            "cliff_distance": 0,
            "goal_block_x": -1,
            "goal_block_y": -1,
            "goal_x": -1.0,
            "goal_object_id": 0,
            "cliff_probe_ledge_width": 0,
            "cliff_probe_ledge_cell_x": -1,
            "cliff_probe_ledge_cell_y": -1,
        },
        spring_stiffness_scale=spring_stiffness_scale,
    )
    env = assemble_cliff_env(dynamic_core, terrain_assets)

    robot_mask = np.asarray(env.robot_point_mask, dtype=bool)
    positions = np.asarray(env.sim_state.positions, dtype=np.float32)
    desired_min_y = float(platform_y + 1) * float(CELL_SIZE) + (
        float(ground_clearance_cells) * float(CELL_SIZE)
    )
    current_min_y = float(np.min(positions[robot_mask, 1])) if np.any(robot_mask) else 0.0
    delta_y = desired_min_y - current_min_y
    if abs(delta_y) > 1.0e-12:
        offset = np.asarray([0.0, delta_y], dtype=np.float32)
        shifted_positions = positions + offset
        shifted_positions_last = np.asarray(env.sim_state.positions_last, dtype=np.float32) + offset
        env = env._replace(
            sim_state=env.sim_state._replace(
                positions=jnp.asarray(shifted_positions, dtype=jnp.float32),
                positions_last=jnp.asarray(shifted_positions_last, dtype=jnp.float32),
            ),
            init_com_y=jnp.asarray(float(env.init_com_y) + delta_y, dtype=jnp.float32),
        )

    alive_mask = jnp.asarray(fixed_body_alive_mask(robot_morph), dtype=jnp.bool_)
    return env, env.render_info, alive_mask


def build_branch_env(
    cfg,
    *,
    environment_mode: str,
    ground_width: int = 20,
    ground_clearance_cells: float = 0.1,
) -> tuple[Any, Any, jnp.ndarray]:
    if str(environment_mode) == "flat_ground":
        return build_flat_ground_env(
            cfg,
            ground_width=int(ground_width),
            ground_clearance_cells=float(ground_clearance_cells),
        )

    env, _legacy_render_info, alive_mask = build_free_space_env(cfg)
    render_info = build_standard_render_info(
        cfg,
        spring_init_rest_length=np.asarray(env.sim_state.spring_init_rest_length, dtype=np.float32),
    )
    return env, render_info, alive_mask


def build_standard_render_info(
    cfg,
    *,
    spring_init_rest_length: np.ndarray | jnp.ndarray,
):
    ensure_import_path()
    from jax_evogym.jax_utils import precompute_grid
    from sensing_edges.env import build_grown_dynamic_render_info
    from sensing_edges.runtime_config import resolved_spawn_x
    from sensing_edges.terrain import make_robot_body

    robot_morph = make_robot_body(
        int(cfg.terrain.robot_w),
        int(cfg.terrain.robot_h),
        shape=str(cfg.body.get("shape", "rectangle")),
    )
    grid_data = precompute_grid(int(cfg.terrain.grid_h), int(cfg.terrain.grid_w))
    return build_grown_dynamic_render_info(
        robot_morph,
        grid_data,
        spawn_x=int(resolved_spawn_x(cfg)),
        spawn_y=int(cfg.terrain.spawn_y),
        spring_init_rest_length=np.asarray(spring_init_rest_length, dtype=np.float32),
    )


def rollout_kwargs(cfg, *, max_steps: int) -> dict[str, Any]:
    ensure_import_path()
    from sensing_edges.runtime_config import rollout_kwargs_from_cfg

    kwargs = rollout_kwargs_from_cfg(cfg)
    kwargs["max_steps"] = int(max_steps)
    kwargs["airborne_kill_height_margin_cells"] = 1.0e6
    kwargs["soft_loft_threshold_cells"] = 1.0e6
    return kwargs


def run_phase_probe(
    *,
    env,
    params,
    alive_mask: jnp.ndarray,
    cfg,
    seed: int,
    max_steps: int,
    initial_nca_state: jnp.ndarray | None = None,
    initial_ctrnn_carry: jnp.ndarray | None = None,
):
    ensure_import_path()
    from sensing_edges.env import run_episode_with_phase_probe

    return run_episode_with_phase_probe(
        env,
        params,
        alive_mask,
        jax.random.PRNGKey(int(seed)),
        initial_nca_state=initial_nca_state,
        initial_ctrnn_carry=initial_ctrnn_carry,
        **rollout_kwargs(cfg, max_steps=max_steps),
    )


def run_controller_trace(
    *,
    env,
    params,
    alive_mask: jnp.ndarray,
    cfg,
    seed: int,
    max_steps: int,
    initial_nca_state: jnp.ndarray | None = None,
    initial_ctrnn_carry: jnp.ndarray | None = None,
):
    ensure_import_path()
    from sensing_edges.env import run_episode_with_controller_trace

    return run_episode_with_controller_trace(
        env,
        params,
        alive_mask,
        jax.random.PRNGKey(int(seed)),
        initial_nca_state=initial_nca_state,
        initial_ctrnn_carry=initial_ctrnn_carry,
        **rollout_kwargs(cfg, max_steps=max_steps),
    )


def phase_reshape(x: np.ndarray, nca_steps_per_phase: int) -> np.ndarray:
    trim = (x.shape[0] // int(nca_steps_per_phase)) * int(nca_steps_per_phase)
    if trim <= 0:
        raise ValueError("Trace ended before a full phase was captured.")
    phase_count = trim // int(nca_steps_per_phase)
    return x[:trim].reshape(phase_count, int(nca_steps_per_phase), *x.shape[1:])


def reduce_controller_trace(trace, *, nca_steps_per_phase: int) -> dict[str, np.ndarray]:
    active = np.asarray(trace.active, dtype=bool)
    nca_micro = np.asarray(trace.nca_state[active], dtype=np.float32)
    ctrnn_micro = np.asarray(trace.ctrnn_carry[active], dtype=np.float32)
    action_micro = np.asarray(trace.action_outputs[active], dtype=np.float32)

    nca_phase = phase_reshape(nca_micro, nca_steps_per_phase)[:, -1]
    ctrnn_phase = phase_reshape(ctrnn_micro, nca_steps_per_phase)[:, -1]
    action_phase = phase_reshape(action_micro, nca_steps_per_phase).mean(axis=1)
    controller_flat = np.concatenate(
        [nca_phase.reshape(nca_phase.shape[0], -1), ctrnn_phase.reshape(ctrnn_phase.shape[0], -1)],
        axis=1,
    )
    return {
        "nca_phase": nca_phase,
        "ctrnn_phase": ctrnn_phase,
        "action_phase": action_phase,
        "controller_flat": controller_flat,
    }


def flatten_phase_trace(phase_trace) -> dict[str, np.ndarray]:
    return {
        "initial_positions": np.asarray(phase_trace.initial_positions, dtype=np.float32),
        "initial_velocities": np.asarray(phase_trace.initial_velocities, dtype=np.float32),
        "initial_spring_rest_length": np.asarray(
            phase_trace.initial_spring_rest_length,
            dtype=np.float32,
        ),
        "phase_post_positions": np.asarray(phase_trace.phase_post_positions, dtype=np.float32).reshape(
            -1,
            *np.asarray(phase_trace.phase_post_positions).shape[2:],
        ),
        "phase_post_velocities": np.asarray(phase_trace.phase_post_velocities, dtype=np.float32).reshape(
            -1,
            *np.asarray(phase_trace.phase_post_velocities).shape[2:],
        ),
        "phase_post_velocities_true": np.asarray(
            phase_trace.phase_post_velocities_true,
            dtype=np.float32,
        ).reshape(
            -1,
            *np.asarray(phase_trace.phase_post_velocities_true).shape[2:],
        ),
        "phase_post_spring_rest_length": np.asarray(
            phase_trace.phase_post_spring_rest_length,
            dtype=np.float32,
        ).reshape(
            -1,
            *np.asarray(phase_trace.phase_post_spring_rest_length).shape[2:],
        ),
        "phase_h_compact": np.asarray(phase_trace.phase_h_compact, dtype=np.float32).reshape(
            -1,
            *np.asarray(phase_trace.phase_h_compact).shape[2:],
        ),
        "phase_v_compact": np.asarray(phase_trace.phase_v_compact, dtype=np.float32).reshape(
            -1,
            *np.asarray(phase_trace.phase_v_compact).shape[2:],
        ),
    }


def robot_render_indices(render_info) -> np.ndarray:
    return np.flatnonzero(
        np.asarray(render_info.cell_is_robot, dtype=bool)
        & np.asarray(render_info.cell_is_dynamic, dtype=bool)
        & np.asarray(render_info.cell_mask, dtype=bool)
    ).astype(np.int32, copy=False)


def robot_cell_colors(
    *,
    local_cell_index: int,
    flash: bool,
    branch_is_perturbed: bool,
    cell_count: int,
) -> list[tuple[float, float, float]]:
    target_rgb = (
        np.asarray([239, 68, 68], dtype=np.float32)
        if flash
        else np.asarray([245, 196, 67], dtype=np.float32)
    ) / 255.0
    del branch_is_perturbed
    colors = [(-1.0, -1.0, -1.0) for _ in range(int(cell_count))]
    colors[int(local_cell_index)] = tuple(target_rgb.tolist())
    return colors


def highlight_overrides(
    render_info,
    *,
    local_cell_index: int,
    flash: bool,
    branch_is_perturbed: bool,
) -> np.ndarray:
    overrides = np.full((int(render_info.cell_mask.shape[0]), 3), -1, dtype=np.int32)
    robot_indices = robot_render_indices(render_info)
    colors = robot_cell_colors(
        local_cell_index=int(local_cell_index),
        flash=bool(flash),
        branch_is_perturbed=bool(branch_is_perturbed),
        cell_count=int(robot_indices.shape[0]),
    )
    for idx, render_idx in enumerate(robot_indices):
        rgb = np.asarray(colors[idx], dtype=np.float32)
        if np.all(rgb < 0.0):
            continue
        overrides[int(render_idx)] = np.clip(np.round(rgb * 255.0), 0.0, 255.0).astype(np.int32)
    return overrides


def default_analysis_dir(bundle_root: Path, *, max_steps: int) -> Path:
    return bundle_root / "analysis_runs" / f"free_space_branch_steps_{int(max_steps)}"


def local_cell_point_indices(
    render_info,
    *,
    local_cell_index: int,
) -> np.ndarray:
    vertex_count = int(np.asarray(render_info.cell_vertex_count, dtype=np.int32)[int(local_cell_index)])
    return np.asarray(
        render_info.cell_vertices[int(local_cell_index), :vertex_count],
        dtype=np.int32,
    )


def default_release_local_cell_index(
    *,
    render_info,
    positions: np.ndarray,
) -> int:
    robot_indices = robot_render_indices(render_info)
    cell_vertices = np.asarray(render_info.cell_vertices[robot_indices], dtype=np.int32)
    centroids = np.asarray(positions[cell_vertices], dtype=np.float32).mean(axis=1)
    max_x = float(np.max(centroids[:, 0]))
    right_candidates = np.where(centroids[:, 0] >= (max_x - 1.0e-5))[0]
    if right_candidates.size <= 1:
        return int(right_candidates[0]) if right_candidates.size == 1 else 0
    candidate_y = centroids[right_candidates, 1]
    return int(right_candidates[int(np.argmin(candidate_y))])


def support_release_profile(
    *,
    dt: float,
    gravity: float,
    physics_substeps: int,
    duration_env_steps: float,
    perturb_strength: float,
) -> dict[str, float]:
    support_duration = float(dt) * float(physics_substeps) * float(duration_env_steps)
    delta_vy = -float(gravity) * support_duration * float(perturb_strength)
    delta_y = -0.5 * float(gravity) * (support_duration ** 2) * float(perturb_strength)
    return {
        "duration_seconds": float(support_duration),
        "delta_vy": float(delta_vy),
        "delta_y": float(delta_y),
    }


def apply_support_release_perturbation(
    sim_state,
    *,
    render_info,
    local_cell_index: int,
    dt: float,
    delta_y: float,
    delta_vy: float,
):
    robot_indices = robot_render_indices(render_info)
    target_render_idx = int(robot_indices[int(local_cell_index)])
    corner_indices = np.asarray(
        render_info.cell_vertices[target_render_idx, : int(render_info.cell_vertex_count[target_render_idx])],
        dtype=np.int32,
    )
    if corner_indices.shape[0] != 4:
        raise ValueError("Expected a quad cell for the perturbation target.")

    positions = np.asarray(sim_state.positions, dtype=np.float32).copy()
    velocities = np.asarray(sim_state.velocities, dtype=np.float32).copy()
    velocities_true = np.asarray(sim_state.velocities_true, dtype=np.float32).copy()

    sag_weights = np.asarray([1.0, 1.0, 0.65, 0.65], dtype=np.float32)
    for corner_idx, point_idx in enumerate(corner_indices):
        y_shift = float(delta_y) * float(sag_weights[corner_idx])
        vy_shift = float(delta_vy) * float(sag_weights[corner_idx])
        positions[point_idx, 1] = positions[point_idx, 1] + y_shift
        velocities[point_idx, 1] = velocities[point_idx, 1] + vy_shift
        velocities_true[point_idx, 1] = velocities_true[point_idx, 1] + vy_shift

    positions_last = positions - (velocities_true * float(dt))
    return sim_state._replace(
        positions=jnp.asarray(positions, dtype=jnp.float32),
        velocities=jnp.asarray(velocities, dtype=jnp.float32),
        positions_last=jnp.asarray(positions_last, dtype=jnp.float32),
        velocities_true=jnp.asarray(velocities_true, dtype=jnp.float32),
    )


def apply_point_hold(
    sim_state,
    *,
    point_indices: np.ndarray,
    anchor_positions: np.ndarray,
):
    positions = np.asarray(sim_state.positions, dtype=np.float32).copy()
    velocities = np.asarray(sim_state.velocities, dtype=np.float32).copy()
    velocities_true = np.asarray(sim_state.velocities_true, dtype=np.float32).copy()
    positions_last = np.asarray(sim_state.positions_last, dtype=np.float32).copy()

    positions[point_indices] = anchor_positions
    positions_last[point_indices] = anchor_positions
    velocities[point_indices] = 0.0
    velocities_true[point_indices] = 0.0
    return sim_state._replace(
        positions=jnp.asarray(positions, dtype=jnp.float32),
        positions_last=jnp.asarray(positions_last, dtype=jnp.float32),
        velocities=jnp.asarray(velocities, dtype=jnp.float32),
        velocities_true=jnp.asarray(velocities_true, dtype=jnp.float32),
    )


def apply_point_hold_jax(
    sim_state,
    *,
    point_indices: jnp.ndarray,
    anchor_positions: jnp.ndarray,
):
    positions = sim_state.positions.at[point_indices].set(anchor_positions)
    positions_last = sim_state.positions_last.at[point_indices].set(anchor_positions)
    velocities = sim_state.velocities.at[point_indices].set(0.0)
    velocities_true = sim_state.velocities_true.at[point_indices].set(0.0)
    return sim_state._replace(
        positions=positions,
        positions_last=positions_last,
        velocities=velocities,
        velocities_true=velocities_true,
    )


def resume_branch_env(
    *,
    env,
    render_info,
    controller_phase: dict[str, np.ndarray],
    phase_arrays: dict[str, np.ndarray],
    resume_env_step: int,
    sense_act_phases: int,
    perturb_local_cell_index: int,
    support_release: dict[str, float],
) -> tuple[Any, jnp.ndarray, jnp.ndarray]:
    prefix_phase_count = int(resume_env_step) * int(sense_act_phases)
    if prefix_phase_count <= 0:
        raise ValueError("resume_env_step must be positive.")
    resume_phase_idx = prefix_phase_count - 1
    base_state = env.sim_state

    positions = phase_arrays["phase_post_positions"][resume_phase_idx]
    velocities = phase_arrays["phase_post_velocities"][resume_phase_idx]
    velocities_true = phase_arrays["phase_post_velocities_true"][resume_phase_idx]
    spring_rest_length = phase_arrays["phase_post_spring_rest_length"][resume_phase_idx]
    positions_last = positions - (velocities_true * float(np.asarray(env.constants.dt).item()))

    resumed_state = base_state._replace(
        positions=jnp.asarray(positions, dtype=jnp.float32),
        velocities=jnp.asarray(velocities, dtype=jnp.float32),
        positions_last=jnp.asarray(positions_last, dtype=jnp.float32),
        velocities_true=jnp.asarray(velocities_true, dtype=jnp.float32),
        spring_rest_length=jnp.asarray(spring_rest_length, dtype=jnp.float32),
        spring_rest_length_goal=jnp.asarray(spring_rest_length, dtype=jnp.float32),
        external_forces=jnp.zeros_like(base_state.external_forces),
        tangential_deformation=jnp.zeros_like(base_state.tangential_deformation),
        friction_anchor=jnp.zeros_like(base_state.friction_anchor),
        friction_anchor_active=jnp.zeros_like(base_state.friction_anchor_active),
        friction_anchor_no_contact_count=jnp.zeros_like(base_state.friction_anchor_no_contact_count),
        static_manifold_cell_idx=jnp.full_like(base_state.static_manifold_cell_idx, -1),
        static_manifold_edge_idx=jnp.full_like(base_state.static_manifold_edge_idx, -1),
        static_manifold_active=jnp.zeros_like(base_state.static_manifold_active),
        static_manifold_no_contact_count=jnp.zeros_like(base_state.static_manifold_no_contact_count),
        static_manifold_normal_force=jnp.zeros_like(base_state.static_manifold_normal_force),
        static_manifold_normal=jnp.zeros_like(base_state.static_manifold_normal),
    )
    resumed_state = apply_support_release_perturbation(
        resumed_state,
        render_info=render_info,
        local_cell_index=int(perturb_local_cell_index),
        dt=float(np.asarray(env.constants.dt).item()),
        delta_y=float(support_release["delta_y"]),
        delta_vy=float(support_release["delta_vy"]),
    )

    resumed_env = env._replace(sim_state=resumed_state)
    initial_nca_state = jnp.asarray(controller_phase["nca_phase"][resume_phase_idx], dtype=jnp.float32)
    initial_ctrnn_carry = jnp.asarray(controller_phase["ctrnn_phase"][resume_phase_idx], dtype=jnp.float32)
    return resumed_env, initial_nca_state, initial_ctrnn_carry


def replay_with_phase_actions_and_hold(
    *,
    env,
    phase_arrays: dict[str, np.ndarray],
    render_info,
    local_cell_index: int,
    hold_until_env_step: int,
    com_hold_until_env_step: int | None,
    sense_act_phases: int,
    physics_substeps: int,
    dynamic_max_candidates: int = 4096,
    dynamic_bin_size: float = 0.2,
    dynamic_impl: str = "sparse_candidate_list",
    dynamic_max_narrow: int | None = None,
) -> dict[str, np.ndarray]:
    ensure_import_path()
    from jax_evogym.actuators import set_per_axis_goals_compact
    from jax_evogym.sim import physics_substep_dynamic_broadphase_with_probe

    point_indices_np = local_cell_point_indices(render_info, local_cell_index=int(local_cell_index))
    point_indices = jnp.asarray(point_indices_np, dtype=jnp.int32)
    robot_point_indices_np = np.asarray(render_info.robot_point_indices, dtype=np.int32)
    robot_point_indices = jnp.asarray(robot_point_indices_np, dtype=jnp.int32)
    free_point_indices_np = np.setdiff1d(robot_point_indices_np, point_indices_np, assume_unique=False)
    free_point_indices = jnp.asarray(free_point_indices_np, dtype=jnp.int32)
    anchor_positions = jnp.asarray(
        np.asarray(env.sim_state.positions, dtype=np.float32)[point_indices_np].copy(),
        dtype=jnp.float32,
    )
    anchor_com = jnp.asarray(
        np.asarray(env.sim_state.positions, dtype=np.float32)[robot_point_indices_np].mean(axis=0),
        dtype=jnp.float32,
    )
    substeps_per_phase = int(physics_substeps) // int(sense_act_phases)
    hold_phase_count = int(hold_until_env_step) * int(sense_act_phases)
    com_hold_phase_count = (
        int(com_hold_until_env_step) * int(sense_act_phases)
        if com_hold_until_env_step is not None
        else 0
    )
    h_compact_all = jnp.asarray(phase_arrays["phase_h_compact"], dtype=jnp.float32)
    v_compact_all = jnp.asarray(phase_arrays["phase_v_compact"], dtype=jnp.float32)
    robot_point_count = max(1, int(robot_point_indices_np.shape[0]))
    free_point_count = int(free_point_indices_np.shape[0])

    def _apply_com_hold(sim_state):
        current_com = jnp.mean(sim_state.positions[robot_point_indices], axis=0)
        delta = current_com - anchor_com
        mean_vel = jnp.mean(sim_state.velocities[robot_point_indices], axis=0)
        mean_vel_true = jnp.mean(sim_state.velocities_true[robot_point_indices], axis=0)

        if free_point_count <= 0 or free_point_count == robot_point_count:
            positions = sim_state.positions.at[robot_point_indices].add(-delta)
            positions_last = sim_state.positions_last.at[robot_point_indices].add(-delta)
            velocities = sim_state.velocities.at[robot_point_indices].add(-mean_vel)
            velocities_true = sim_state.velocities_true.at[robot_point_indices].add(-mean_vel_true)
        else:
            scale = jnp.float32(float(robot_point_count) / float(free_point_count))
            positions = sim_state.positions.at[free_point_indices].add(-delta * scale)
            positions_last = sim_state.positions_last.at[free_point_indices].add(-delta * scale)
            velocities = sim_state.velocities.at[free_point_indices].add(-mean_vel * scale)
            velocities_true = sim_state.velocities_true.at[free_point_indices].add(-mean_vel_true * scale)
        return sim_state._replace(
            positions=positions,
            positions_last=positions_last,
            velocities=velocities,
            velocities_true=velocities_true,
        )

    def _phase_step(sim_carry, inputs):
        phase_idx, h_compact, v_compact = inputs
        sim = set_per_axis_goals_compact(sim_carry, env.per_axis_info, h_compact, v_compact)

        def _substep(sub_sim, _):
            next_sim, _probe = physics_substep_dynamic_broadphase_with_probe(
                sub_sim,
                env.topology,
                env.constants,
                collision_data=env.collision_data,
                static_collider_data=env.static_collider_data,
                static_collision_worklist=env.static_collision_worklist,
                dynamic_max_candidates=jnp.int32(dynamic_max_candidates),
                dynamic_bin_size=jnp.float32(dynamic_bin_size),
                dynamic_impl=dynamic_impl,
                dynamic_max_narrow=dynamic_max_narrow,
            )
            held_sim = jax.lax.cond(
                phase_idx < jnp.int32(hold_phase_count),
                lambda s: apply_point_hold_jax(
                    s,
                    point_indices=point_indices,
                    anchor_positions=anchor_positions,
                ),
                lambda s: s,
                next_sim,
            )
            pinned_sim = jax.lax.cond(
                phase_idx < jnp.int32(com_hold_phase_count),
                _apply_com_hold,
                lambda s: s,
                held_sim,
            )
            return pinned_sim, None

        sim, _ = jax.lax.scan(
            _substep,
            sim,
            None,
            length=int(substeps_per_phase),
        )
        return sim, (
            sim.positions,
            sim.spring_rest_length,
        )

    _, (positions_out, spring_out) = jax.lax.scan(
        _phase_step,
        env.sim_state,
        (
            jnp.arange(h_compact_all.shape[0], dtype=jnp.int32),
            h_compact_all,
            v_compact_all,
        ),
    )
    return {
        "phase_post_positions": np.asarray(positions_out, dtype=np.float32),
        "phase_post_spring_rest_length": np.asarray(spring_out, dtype=np.float32),
    }


def concat_branch_series(
    prefix: np.ndarray,
    tail: np.ndarray,
    *,
    prefix_phase_count: int,
) -> np.ndarray:
    return np.concatenate([prefix[:prefix_phase_count], tail], axis=0)


def compute_raw_abs_hv_ratios(
    positions_seq: np.ndarray,
    *,
    topology,
    spring_init_rest_length: jnp.ndarray,
    deformation_info,
) -> np.ndarray:
    a_idx = np.asarray(topology.a_idx, dtype=np.int32)
    b_idx = np.asarray(topology.b_idx, dtype=np.int32)
    init_rest = np.asarray(spring_init_rest_length, dtype=np.float32)
    safe_rest = np.where(init_rest > 1.0e-12, init_rest, 1.0)
    h_idx = np.asarray(deformation_info.voxel_horiz_springs, dtype=np.int32)
    v_idx = np.asarray(deformation_info.voxel_vert_springs, dtype=np.int32)
    sentinel_index = int(safe_rest.shape[0])
    h_take = np.where(h_idx >= 0, h_idx, sentinel_index)
    v_take = np.where(v_idx >= 0, v_idx, sentinel_index)

    features = np.zeros((positions_seq.shape[0], h_idx.shape[0] * 2), dtype=np.float32)
    for phase_idx, positions in enumerate(positions_seq):
        lengths = np.linalg.norm(positions[a_idx] - positions[b_idx], axis=1).astype(np.float32, copy=False)
        ratios = lengths / safe_rest
        padded = np.concatenate([ratios, np.ones((1,), dtype=np.float32)], axis=0)
        h_ratio = padded[h_take].mean(axis=1)
        v_ratio = padded[v_take].mean(axis=1)
        features[phase_idx] = np.concatenate([h_ratio, v_ratio], axis=0)
    return features


def fit_pca(features_a: np.ndarray, features_b: np.ndarray, *, n_components: int = 2) -> dict[str, np.ndarray]:
    pooled = np.concatenate([features_a, features_b], axis=0).astype(np.float64, copy=False)
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0)
    std = np.where(std > 1.0e-8, std, 1.0)
    centered = (pooled - mean) / std
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[: int(n_components)].astype(np.float64, copy=False)
    return {
        "mean": mean.astype(np.float32, copy=False),
        "std": std.astype(np.float32, copy=False),
        "components": components.astype(np.float32, copy=False),
    }


def transform_pca(features: np.ndarray, pca: dict[str, np.ndarray]) -> np.ndarray:
    centered = (features.astype(np.float32, copy=False) - pca["mean"]) / pca["std"]
    return centered @ pca["components"].T


def sample_phase_indices(total_phases: int, *, max_frames: int, required_phase_index: int) -> np.ndarray:
    if total_phases <= max_frames:
        base = np.arange(total_phases, dtype=np.int32)
    else:
        base = np.linspace(0, total_phases - 1, num=max_frames, dtype=np.int32)
    augmented = np.concatenate([base, np.asarray([required_phase_index], dtype=np.int32)])
    return np.unique(np.clip(augmented, 0, total_phases - 1)).astype(np.int32, copy=False)


def camera_spec_from_sequences(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    *,
    render_info,
    sample_phase_indices_arr: np.ndarray,
    panel_aspect: float,
) -> tuple[np.ndarray, float]:
    robot_point_indices = np.asarray(render_info.robot_point_indices, dtype=np.int32)
    sampled = np.concatenate(
        [
            positions_a[sample_phase_indices_arr][:, robot_point_indices, :],
            positions_b[sample_phase_indices_arr][:, robot_point_indices, :],
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    min_xy = sampled.min(axis=(0, 1))
    max_xy = sampled.max(axis=(0, 1))
    center = 0.5 * (min_xy + max_xy)
    width = float(max_xy[0] - min_xy[0]) + 0.25
    height = float(max_xy[1] - min_xy[1]) + 0.25
    viewport_width = max(width, height * float(panel_aspect), 1.0)
    return center.astype(np.float32, copy=False), float(viewport_width)


def encode_frames(frames_dir: Path, *, fps: int, mp4_path: Path, gif_path: Path) -> None:
    palette_path = gif_path.with_name("palette.png")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(int(fps)),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-pix_fmt",
            "yuv420p",
            str(mp4_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(int(fps)),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-vf",
            "palettegen",
            str(palette_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(int(fps)),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-i",
            str(palette_path),
            "-lavfi",
            "paletteuse",
            str(gif_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def capture_branch_trace(
    *,
    bundle_root: Path,
    controller_params_path: Path,
    config_snapshot_path: Path,
    output_dir: Path,
    max_steps: int,
    environment_mode: str,
    ground_width: int,
    ground_clearance_cells: float,
    resume_env_step: int | None,
    perturb_local_cell_index: int | None,
    perturb_strength: float,
    support_duration_env_steps: float,
    perturb_mode: str = "support_release",
    hold_until_env_step: int = 50,
    com_hold_until_env_step: int | None = None,
    other_hold_until_env_step: int | None = None,
    seed: int,
) -> dict[str, Any]:
    ensure_import_path()
    from jax_evogym.constants import GRAVITY

    cfg = load_cfg(config_snapshot_path, max_steps=int(max_steps))
    controller_env, _controller_render_info, alive_mask = build_free_space_env(cfg)
    env, render_info, _branch_alive_mask = build_branch_env(
        cfg,
        environment_mode=str(environment_mode),
        ground_width=int(ground_width),
        ground_clearance_cells=float(ground_clearance_cells),
    )
    params = load_controller_params(controller_params_path)

    if perturb_mode == "hold_release":
        resolved_resume_env_step = int(hold_until_env_step)
    elif resume_env_step is None:
        resolved_resume_env_step = max(1, int(max_steps) // 4)
    else:
        resolved_resume_env_step = int(resume_env_step)
    if resolved_resume_env_step >= int(max_steps):
        raise ValueError("resume_env_step must be smaller than max_steps.")

    baseline_result_phase, baseline_phase_trace = run_phase_probe(
        env=controller_env,
        params=params,
        alive_mask=alive_mask,
        cfg=cfg,
        seed=int(seed),
        max_steps=int(max_steps),
    )
    baseline_result_controller, baseline_controller_trace = run_controller_trace(
        env=controller_env,
        params=params,
        alive_mask=alive_mask,
        cfg=cfg,
        seed=int(seed),
        max_steps=int(max_steps),
    )
    del baseline_result_phase, baseline_result_controller

    sense_act_phases = int(cfg.sim.sense_act_phases_per_env_step)
    nca_steps_per_phase = int(cfg.nca.nca_steps_per_phase)
    prefix_phase_count = int(resolved_resume_env_step) * int(sense_act_phases)
    remaining_steps = int(max_steps) - int(resolved_resume_env_step)

    baseline_phase_arrays = flatten_phase_trace(baseline_phase_trace)
    baseline_controller_phase = reduce_controller_trace(
        baseline_controller_trace,
        nca_steps_per_phase=nca_steps_per_phase,
    )
    standard_render_info = render_info
    cell_count = int(np.count_nonzero(np.asarray(standard_render_info.cell_is_robot, dtype=bool)))
    resolved_local_cell_index = (
        default_release_local_cell_index(
            render_info=standard_render_info,
            positions=np.asarray(env.sim_state.positions, dtype=np.float32),
        )
        if perturb_local_cell_index is None
        else int(perturb_local_cell_index)
    )
    if not (0 <= resolved_local_cell_index < cell_count):
        raise ValueError("perturb_local_cell_index fell outside the robot body.")

    baseline_positions = baseline_phase_arrays["phase_post_positions"]
    baseline_spring_rest = baseline_phase_arrays["phase_post_spring_rest_length"]
    baseline_controller_flat = baseline_controller_phase["controller_flat"]
    baseline_action_phase = baseline_controller_phase["action_phase"]
    if perturb_mode == "hold_release":
        display_baseline_phase_arrays = replay_with_phase_actions_and_hold(
            env=env,
            phase_arrays=baseline_phase_arrays,
            render_info=standard_render_info,
            local_cell_index=int(resolved_local_cell_index),
            hold_until_env_step=int(hold_until_env_step),
            com_hold_until_env_step=com_hold_until_env_step,
            sense_act_phases=int(sense_act_phases),
            physics_substeps=int(cfg.sim.physics_substeps),
        )
        display_other_hold_until = (
            int(max_steps) if other_hold_until_env_step is None else int(other_hold_until_env_step)
        )
        perturbed_phase_arrays = replay_with_phase_actions_and_hold(
            env=env,
            phase_arrays=baseline_phase_arrays,
            render_info=standard_render_info,
            local_cell_index=int(resolved_local_cell_index),
            hold_until_env_step=int(display_other_hold_until),
            com_hold_until_env_step=com_hold_until_env_step,
            sense_act_phases=int(sense_act_phases),
            physics_substeps=int(cfg.sim.physics_substeps),
        )
        baseline_positions = display_baseline_phase_arrays["phase_post_positions"]
        baseline_spring_rest = display_baseline_phase_arrays["phase_post_spring_rest_length"]
        perturbed_positions = perturbed_phase_arrays["phase_post_positions"]
        perturbed_spring_rest = perturbed_phase_arrays["phase_post_spring_rest_length"]
        perturbed_controller_flat = baseline_controller_flat.copy()
        perturbed_action_phase = baseline_action_phase.copy()
        release = {
            "duration_seconds": 0.0,
            "delta_vy": 0.0,
            "delta_y": 0.0,
        }
        remaining_steps = int(max_steps) - int(hold_until_env_step)
        prefix_phase_count = int(hold_until_env_step) * int(sense_act_phases)
        row_label_top = f"Release @ {int(hold_until_env_step)}"
        row_label_bottom = f"Held {int(display_other_hold_until)}"
    else:
        release = support_release_profile(
            dt=float(np.asarray(env.constants.dt).item()),
            gravity=float(GRAVITY),
            physics_substeps=int(cfg.sim.physics_substeps),
            duration_env_steps=float(support_duration_env_steps),
            perturb_strength=float(perturb_strength),
        )
        resumed_env, initial_nca_state, initial_ctrnn_carry = resume_branch_env(
            env=env,
            render_info=standard_render_info,
            controller_phase=baseline_controller_phase,
            phase_arrays=baseline_phase_arrays,
            resume_env_step=int(resolved_resume_env_step),
            sense_act_phases=int(sense_act_phases),
            perturb_local_cell_index=int(resolved_local_cell_index),
            support_release=release,
        )

        perturbed_result_phase, perturbed_phase_trace = run_phase_probe(
            env=resumed_env,
            params=params,
            alive_mask=alive_mask,
            cfg=cfg,
            seed=int(seed),
            max_steps=int(remaining_steps),
            initial_nca_state=initial_nca_state,
            initial_ctrnn_carry=initial_ctrnn_carry,
        )
        perturbed_result_controller, perturbed_controller_trace = run_controller_trace(
            env=resumed_env,
            params=params,
            alive_mask=alive_mask,
            cfg=cfg,
            seed=int(seed),
            max_steps=int(remaining_steps),
            initial_nca_state=initial_nca_state,
            initial_ctrnn_carry=initial_ctrnn_carry,
        )
        del perturbed_result_phase, perturbed_result_controller

        perturbed_phase_arrays_tail = flatten_phase_trace(perturbed_phase_trace)
        perturbed_controller_phase_tail = reduce_controller_trace(
            perturbed_controller_trace,
            nca_steps_per_phase=nca_steps_per_phase,
        )

        perturbed_positions = concat_branch_series(
            baseline_positions,
            perturbed_phase_arrays_tail["phase_post_positions"],
            prefix_phase_count=int(prefix_phase_count),
        )
        perturbed_spring_rest = concat_branch_series(
            baseline_spring_rest,
            perturbed_phase_arrays_tail["phase_post_spring_rest_length"],
            prefix_phase_count=int(prefix_phase_count),
        )
        perturbed_controller_flat = concat_branch_series(
            baseline_controller_flat,
            perturbed_controller_phase_tail["controller_flat"],
            prefix_phase_count=int(prefix_phase_count),
        )
        perturbed_action_phase = concat_branch_series(
            baseline_action_phase,
            perturbed_controller_phase_tail["action_phase"],
            prefix_phase_count=int(prefix_phase_count),
        )
        row_label_top = "Baseline"
        row_label_bottom = "Perturbed"

    baseline_raw_abs_hv = compute_raw_abs_hv_ratios(
        baseline_positions,
        topology=env.topology,
        spring_init_rest_length=env.sim_state.spring_init_rest_length,
        deformation_info=env.deformation_info,
    )
    perturbed_raw_abs_hv = compute_raw_abs_hv_ratios(
        perturbed_positions,
        topology=env.topology,
        spring_init_rest_length=env.sim_state.spring_init_rest_length,
        deformation_info=env.deformation_info,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "branch_trace.npz",
        baseline_positions=baseline_positions,
        perturbed_positions=perturbed_positions,
        baseline_spring_rest_length=baseline_spring_rest,
        perturbed_spring_rest_length=perturbed_spring_rest,
        baseline_controller_flat=baseline_controller_flat,
        perturbed_controller_flat=perturbed_controller_flat,
        baseline_action_phase=baseline_action_phase,
        perturbed_action_phase=perturbed_action_phase,
        baseline_raw_abs_hv=baseline_raw_abs_hv,
        perturbed_raw_abs_hv=perturbed_raw_abs_hv,
        perturb_phase_index=np.asarray([prefix_phase_count], dtype=np.int32),
        perturb_local_cell_index=np.asarray([resolved_local_cell_index], dtype=np.int32),
        sense_act_phases=np.asarray([sense_act_phases], dtype=np.int32),
        nca_steps_per_phase=np.asarray([nca_steps_per_phase], dtype=np.int32),
        max_steps=np.asarray([max_steps], dtype=np.int32),
        resume_env_step=np.asarray([resolved_resume_env_step], dtype=np.int32),
        perturb_mode=np.asarray([0 if perturb_mode == "support_release" else 1], dtype=np.int32),
    )

    post_slice = slice(int(prefix_phase_count), None)
    plant_post_delta = np.linalg.norm(
        baseline_raw_abs_hv[post_slice] - perturbed_raw_abs_hv[post_slice],
        axis=1,
    )
    controller_post_delta = np.linalg.norm(
        baseline_controller_flat[post_slice] - perturbed_controller_flat[post_slice],
        axis=1,
    )

    summary = {
        "bundle_root": str(bundle_root),
        "controller_params_path": str(controller_params_path),
        "config_snapshot_path": str(config_snapshot_path),
        "trace_path": str(output_dir / "branch_trace.npz"),
        "max_steps": int(max_steps),
        "environment_mode": str(environment_mode),
        "ground_width": int(ground_width),
        "ground_clearance_cells": float(ground_clearance_cells),
        "sense_act_phases": int(sense_act_phases),
        "nca_steps_per_phase": int(nca_steps_per_phase),
        "phase_count": int(baseline_positions.shape[0]),
        "resume_env_step": int(resolved_resume_env_step),
        "perturb_phase_index": int(prefix_phase_count),
        "remaining_steps": int(remaining_steps),
        "perturb_local_cell_index": int(resolved_local_cell_index),
        "perturb_mode": str(perturb_mode),
        "event_label": "RELEASE" if perturb_mode == "hold_release" else "PERTURBATION",
        "row_label_top": row_label_top,
        "row_label_bottom": row_label_bottom,
        "perturb_strength": float(perturb_strength),
        "support_duration_env_steps": float(support_duration_env_steps),
        "hold_until_env_step": int(hold_until_env_step),
        "com_hold_until_env_step": (
            None if com_hold_until_env_step is None else int(com_hold_until_env_step)
        ),
        "other_hold_until_env_step": (
            int(max_steps) if other_hold_until_env_step is None else int(other_hold_until_env_step)
        ),
        "support_release_duration_seconds": float(release["duration_seconds"]),
        "support_release_delta_y": float(release["delta_y"]),
        "support_release_delta_vy": float(release["delta_vy"]),
        "plant_post_delta_mean": float(np.mean(plant_post_delta)),
        "plant_post_delta_max": float(np.max(plant_post_delta)),
        "controller_post_delta_mean": float(np.mean(controller_post_delta)),
        "controller_post_delta_max": float(np.max(controller_post_delta)),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def load_branch_trace(trace_dir: Path) -> dict[str, np.ndarray]:
    trace_path = trace_dir / "branch_trace.npz"
    with np.load(trace_path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}
