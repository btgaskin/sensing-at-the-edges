from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np
from omegaconf import OmegaConf


SCRIPT_DIR = Path(__file__).resolve().parent
S4_ROOT = SCRIPT_DIR.parent
REPO_ROOT = S4_ROOT.parent
SENSING_EDGES_ROOT = REPO_ROOT / "sensing_edges"
DEFAULT_BUNDLE_ROOT = S4_ROOT / "results" / "bin_03_rank_00_cell_6-0-3_flat_0423"


def _ensure_import_path() -> None:
    path = str(SENSING_EDGES_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_controller_params(path: Path):
    _ensure_import_path()
    from sensing_edges.nca import CTRNNParams

    with np.load(path, allow_pickle=False) as data:
        return CTRNNParams(
            w_input=jnp.asarray(np.asarray(data["w_input"], dtype=np.float32)),
            w_rec=jnp.asarray(np.asarray(data["w_rec"], dtype=np.float32)),
            w_output=jnp.asarray(np.asarray(data["w_output"], dtype=np.float32)),
            bias=jnp.asarray(np.asarray(data["bias"], dtype=np.float32)),
            log_tau=jnp.asarray(np.asarray(data["log_tau"], dtype=np.float32)),
        )


def _load_cfg(config_snapshot_path: Path, *, max_steps: int) -> Any:
    _ensure_import_path()
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


def _build_free_space_env(cfg) -> tuple[Any, Any, jnp.ndarray]:
    _ensure_import_path()
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


def _rollout_kwargs(cfg, *, max_steps: int) -> dict[str, Any]:
    _ensure_import_path()
    from sensing_edges.runtime_config import rollout_kwargs_from_cfg

    kwargs = rollout_kwargs_from_cfg(cfg)
    kwargs["max_steps"] = int(max_steps)
    kwargs["airborne_kill_height_margin_cells"] = 1.0e6
    kwargs["soft_loft_threshold_cells"] = 1.0e6
    return kwargs


def _run_phase_probe(
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
    _ensure_import_path()
    from sensing_edges.env import run_episode_with_phase_probe

    return run_episode_with_phase_probe(
        env,
        params,
        alive_mask,
        jax.random.PRNGKey(int(seed)),
        initial_nca_state=initial_nca_state,
        initial_ctrnn_carry=initial_ctrnn_carry,
        **_rollout_kwargs(cfg, max_steps=max_steps),
    )


def _run_controller_trace(
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
    _ensure_import_path()
    from sensing_edges.env import run_episode_with_controller_trace

    return run_episode_with_controller_trace(
        env,
        params,
        alive_mask,
        jax.random.PRNGKey(int(seed)),
        initial_nca_state=initial_nca_state,
        initial_ctrnn_carry=initial_ctrnn_carry,
        **_rollout_kwargs(cfg, max_steps=max_steps),
    )


def _phase_reshape(x: np.ndarray, nca_steps_per_phase: int) -> np.ndarray:
    trim = (x.shape[0] // int(nca_steps_per_phase)) * int(nca_steps_per_phase)
    if trim <= 0:
        raise ValueError("Trace ended before a full phase was captured.")
    phase_count = trim // int(nca_steps_per_phase)
    return x[:trim].reshape(phase_count, int(nca_steps_per_phase), *x.shape[1:])


def _reduce_controller_trace(trace, *, nca_steps_per_phase: int) -> dict[str, np.ndarray]:
    active = np.asarray(trace.active, dtype=bool)
    nca_micro = np.asarray(trace.nca_state[active], dtype=np.float32)
    ctrnn_micro = np.asarray(trace.ctrnn_carry[active], dtype=np.float32)
    action_micro = np.asarray(trace.action_outputs[active], dtype=np.float32)

    nca_phase = _phase_reshape(nca_micro, nca_steps_per_phase)[:, -1]
    ctrnn_phase = _phase_reshape(ctrnn_micro, nca_steps_per_phase)[:, -1]
    action_phase = _phase_reshape(action_micro, nca_steps_per_phase).mean(axis=1)
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


def _flatten_phase_trace(phase_trace) -> dict[str, np.ndarray]:
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
    }


def _robot_render_indices(render_info) -> np.ndarray:
    return np.flatnonzero(
        np.asarray(render_info.cell_is_robot, dtype=bool)
        & np.asarray(render_info.cell_is_dynamic, dtype=bool)
        & np.asarray(render_info.cell_mask, dtype=bool)
    ).astype(np.int32, copy=False)


def _robot_cell_polygons(
    positions: np.ndarray,
    *,
    render_info,
) -> np.ndarray:
    robot_indices = _robot_render_indices(render_info)
    return np.asarray(
        positions[np.asarray(render_info.cell_vertices[robot_indices], dtype=np.int32)],
        dtype=np.float32,
    )


def _robot_cell_colors(
    *,
    local_cell_index: int,
    flash: bool,
    branch_is_perturbed: bool,
    cell_count: int,
) -> list[tuple[float, float, float]]:
    base_rgb = (
        np.asarray([244, 180, 120], dtype=np.float32)
        if branch_is_perturbed
        else np.asarray([148, 194, 255], dtype=np.float32)
    ) / 255.0
    target_rgb = (
        np.asarray([239, 68, 68], dtype=np.float32)
        if flash
        else np.asarray([245, 196, 67], dtype=np.float32)
    ) / 255.0
    colors = [tuple(base_rgb.tolist()) for _ in range(int(cell_count))]
    colors[int(local_cell_index)] = tuple(target_rgb.tolist())
    return colors


def _highlight_overrides(
    render_info,
    *,
    local_cell_index: int,
    flash: bool,
    branch_is_perturbed: bool,
) -> np.ndarray:
    overrides = np.full((int(render_info.cell_mask.shape[0]), 3), -1, dtype=np.int32)
    robot_indices = _robot_render_indices(render_info)
    base_colors = _robot_cell_colors(
        local_cell_index=int(local_cell_index),
        flash=bool(flash),
        branch_is_perturbed=bool(branch_is_perturbed),
        cell_count=int(robot_indices.shape[0]),
    )
    for idx, render_idx in enumerate(robot_indices):
        rgb = np.asarray(base_colors[idx], dtype=np.float32)
        overrides[int(render_idx)] = np.clip(np.round(rgb * 255.0), 0.0, 255.0).astype(np.int32)
    return overrides


def _default_release_local_cell_index(
    *,
    render_info,
    positions: np.ndarray,
) -> int:
    polygons = _robot_cell_polygons(positions, render_info=render_info)
    centroids = polygons.mean(axis=1)
    max_x = float(np.max(centroids[:, 0]))
    right_candidates = np.where(centroids[:, 0] >= (max_x - 1.0e-5))[0]
    if right_candidates.size <= 1:
        return int(right_candidates[0]) if right_candidates.size == 1 else 0
    candidate_y = centroids[right_candidates, 1]
    return int(right_candidates[int(np.argmin(candidate_y))])


def _support_release_profile(
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


def _apply_support_release_perturbation(
    sim_state,
    *,
    render_info,
    local_cell_index: int,
    dt: float,
    delta_y: float,
    delta_vy: float,
):
    robot_indices = _robot_render_indices(render_info)
    target_render_idx = int(robot_indices[int(local_cell_index)])
    corner_indices = np.asarray(
        render_info.cell_vertices[target_render_idx, : int(render_info.cell_vertex_count[target_render_idx])],
        dtype=np.int32,
    )
    if corner_indices.shape[0] != 4:
        raise ValueError("Expected a quad cell for the perturbation target.")

    positions = np.asarray(sim_state.positions, dtype=np.float32).copy()
    velocities = np.asarray(sim_state.velocities, dtype=np.float32).copy()
    velocities_true = np.asarray(sim_state.velocities_true, dtype=np.float32)
    velocities_true = velocities_true.copy()

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


def _resume_branch_env(
    *,
    env,
    render_info,
    controller_phase: dict[str, np.ndarray],
    phase_arrays: dict[str, np.ndarray],
    resume_env_step: int,
    sense_act_phases: int,
    perturb_local_cell_index: int,
    support_release_profile: dict[str, float],
    apply_perturbation: bool,
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
    if apply_perturbation:
        resumed_state = _apply_support_release_perturbation(
            resumed_state,
            render_info=render_info,
            local_cell_index=int(perturb_local_cell_index),
            dt=float(np.asarray(env.constants.dt).item()),
            delta_y=float(support_release_profile["delta_y"]),
            delta_vy=float(support_release_profile["delta_vy"]),
        )

    resumed_env = env._replace(sim_state=resumed_state)
    initial_nca_state = jnp.asarray(controller_phase["nca_phase"][resume_phase_idx], dtype=jnp.float32)
    initial_ctrnn_carry = jnp.asarray(controller_phase["ctrnn_phase"][resume_phase_idx], dtype=jnp.float32)
    return resumed_env, initial_nca_state, initial_ctrnn_carry


def _concat_branch_series(
    prefix: np.ndarray,
    tail: np.ndarray,
    *,
    prefix_phase_count: int,
) -> np.ndarray:
    return np.concatenate([prefix[:prefix_phase_count], tail], axis=0)


def _compute_abs_deformation_features(
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
        h_def = padded[h_take].mean(axis=1) - 1.0
        v_def = padded[v_take].mean(axis=1) - 1.0
        features[phase_idx] = np.concatenate([h_def, v_def], axis=0)
    return features


def _fit_pca(features_a: np.ndarray, features_b: np.ndarray, *, n_components: int = 2) -> dict[str, np.ndarray]:
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


def _transform_pca(features: np.ndarray, pca: dict[str, np.ndarray]) -> np.ndarray:
    centered = (features.astype(np.float32, copy=False) - pca["mean"]) / pca["std"]
    return centered @ pca["components"].T


def _sample_phase_indices(total_phases: int, *, max_frames: int, required_phase_index: int) -> np.ndarray:
    if total_phases <= max_frames:
        base = np.arange(total_phases, dtype=np.int32)
    else:
        base = np.linspace(0, total_phases - 1, num=max_frames, dtype=np.int32)
    augmented = np.concatenate([base, np.asarray([required_phase_index], dtype=np.int32)])
    return np.unique(np.clip(augmented, 0, total_phases - 1)).astype(np.int32, copy=False)


def _robot_plot_layout(
    baseline_positions: np.ndarray,
    perturbed_positions: np.ndarray,
    *,
    render_info,
    y_exaggeration: float,
) -> dict[str, float]:
    robot_point_indices = np.asarray(render_info.robot_point_indices, dtype=np.int32)
    sampled_points = np.concatenate(
        [
            baseline_positions[:, robot_point_indices, :],
            perturbed_positions[:, robot_point_indices, :],
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    center_x = float(np.mean(sampled_points[:, :, 0]))
    center_y = float(np.mean(sampled_points[:, :, 1]))

    scaled = sampled_points.copy()
    scaled[:, :, 1] = center_y + (scaled[:, :, 1] - center_y) * float(y_exaggeration)
    min_x = float(np.min(scaled[:, :, 0]))
    max_x = float(np.max(scaled[:, :, 0]))
    min_y = float(np.min(scaled[:, :, 1]))
    max_y = float(np.max(scaled[:, :, 1]))
    span_x = max(max_x - min_x, 0.25)
    span_y = max(max_y - min_y, 0.25)
    pad_x = 0.08 * span_x + 0.03
    pad_y = 0.12 * span_y + 0.03
    return {
        "center_y": center_y,
        "y_exaggeration": float(y_exaggeration),
        "x_min": min_x - pad_x,
        "x_max": max_x + pad_x,
        "y_min": min_y - pad_y,
        "y_max": max_y + pad_y,
    }


def _scaled_robot_polygons(
    positions: np.ndarray,
    *,
    render_info,
    center_y: float,
    y_exaggeration: float,
) -> np.ndarray:
    polygons = _robot_cell_polygons(positions, render_info=render_info).copy()
    polygons[:, :, 1] = float(center_y) + (
        polygons[:, :, 1] - float(center_y)
    ) * float(y_exaggeration)
    return polygons


def _robot_schematic_polygons(
    plant_features: np.ndarray,
    *,
    robot_h: int,
    robot_w: int,
    y_scale: float = 1.0,
) -> np.ndarray:
    cell_count = int(robot_h) * int(robot_w)
    h_def = np.asarray(plant_features[:cell_count], dtype=np.float32)
    v_def = np.asarray(plant_features[cell_count:cell_count * 2], dtype=np.float32)
    base_w = 0.88
    base_h = 0.88 * float(y_scale)
    pitch_x = 0.92
    pitch_y = 1.02 * float(y_scale)
    polygons = np.zeros((cell_count, 4, 2), dtype=np.float32)
    for idx in range(cell_count):
        row = idx // int(robot_w)
        col = idx % int(robot_w)
        center_x = (float(col) - (float(robot_w) - 1.0) / 2.0) * pitch_x
        center_y = ((float(robot_h) - 1.0) / 2.0 - float(row)) * pitch_y
        width = float(np.clip(base_w * (1.0 + h_def[idx]), 0.35, 1.45))
        height = float(np.clip(base_h * (1.0 + v_def[idx]), 0.35, 1.45))
        polygons[idx] = np.asarray(
            [
                [center_x - width / 2.0, center_y - height / 2.0],
                [center_x + width / 2.0, center_y - height / 2.0],
                [center_x + width / 2.0, center_y + height / 2.0],
                [center_x - width / 2.0, center_y + height / 2.0],
            ],
            dtype=np.float32,
        )
    return polygons


def _camera_spec_from_sequences(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    *,
    render_info,
    sample_phase_indices: np.ndarray,
    panel_aspect: float,
) -> tuple[np.ndarray, float]:
    robot_point_indices = np.asarray(render_info.robot_point_indices, dtype=np.int32)
    sampled = np.concatenate(
        [
            positions_a[sample_phase_indices][:, robot_point_indices, :],
            positions_b[sample_phase_indices][:, robot_point_indices, :],
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


def _render_robot_images(
    *,
    positions_seq: np.ndarray,
    spring_seq: np.ndarray,
    render_info,
    sample_phase_indices: np.ndarray,
    camera_center: np.ndarray,
    viewport_width: float,
    local_cell_index: int,
    flash_window_phase_start: int,
    flash_window_phase_stop: int,
    branch_is_perturbed: bool,
    trail_length: int,
) -> list[np.ndarray]:
    _ensure_import_path()
    from jax_evogym.render import RenderConfig, render_frame

    config = RenderConfig(
        enabled=True,
        width=900,
        height=340,
        fps=12,
        real_time=False,
        frame_skip=1,
        show_grid=False,
        show_edges=True,
        dynamic_actuator_colors=False,
        camera_padding=0.0,
        viewport_width=float(viewport_width),
        camera_mode="fixed",
        camera_smoothing=0.0,
        supersample=3,
        edge_width_px=2,
    )
    images: list[np.ndarray] = []
    history: list[np.ndarray] = []

    def _with_trail(current: np.ndarray) -> np.ndarray:
        if int(trail_length) <= 0 or not history:
            return current
        bg = current[0, 0].astype(np.float32)
        blended = current.astype(np.float32).copy()
        for idx, prev in enumerate(reversed(history[-int(trail_length):]), start=1):
            alpha = 0.22 * (0.6 ** float(idx - 1))
            delta = np.abs(prev.astype(np.float32) - bg[None, None, :])
            mask = np.any(delta > 6.0, axis=2)
            blended[mask] = ((1.0 - alpha) * blended[mask]) + (alpha * prev[mask].astype(np.float32))
        return np.clip(np.round(blended), 0.0, 255.0).astype(np.uint8)

    for phase_idx in sample_phase_indices:
        flash = bool(branch_is_perturbed and flash_window_phase_start <= int(phase_idx) < flash_window_phase_stop)
        overrides = _highlight_overrides(
            render_info,
            local_cell_index=int(local_cell_index),
            flash=flash,
            branch_is_perturbed=bool(branch_is_perturbed),
        )
        image, _ = render_frame(
            positions_seq[int(phase_idx)],
            render_info,
            config,
            spring_rest_length_np=spring_seq[int(phase_idx)],
            camera_pos=np.asarray(camera_center, dtype=np.float32),
            cell_color_overrides_np=overrides,
        )
        image_np = np.asarray(image)
        images.append(_with_trail(image_np))
        history.append(image_np)
    return images


def _save_frames_and_encode(
    *,
    output_dir: Path,
    frame_phase_indices: np.ndarray,
    baseline_plant_xy: np.ndarray,
    perturbed_plant_xy: np.ndarray,
    baseline_controller_xy: np.ndarray,
    perturbed_controller_xy: np.ndarray,
    baseline_robot_images: list[np.ndarray],
    perturbed_robot_images: list[np.ndarray],
    local_cell_index: int,
    perturb_phase_start: int,
    fps: int,
    mp4_name: str,
    gif_name: str,
    robot_y_exaggeration: float,
) -> dict[str, str]:
    frames_dir = output_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    def _limits(xy_a: np.ndarray, xy_b: np.ndarray) -> tuple[float, float, float, float]:
        pooled = np.concatenate([xy_a, xy_b], axis=0)
        min_x = float(np.min(pooled[:, 0]))
        max_x = float(np.max(pooled[:, 0]))
        min_y = float(np.min(pooled[:, 1]))
        max_y = float(np.max(pooled[:, 1]))
        pad_x = max(0.1, 0.08 * max(max_x - min_x, 1.0))
        pad_y = max(0.1, 0.08 * max(max_y - min_y, 1.0))
        return min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y

    del local_cell_index
    del robot_y_exaggeration

    plant_limits = _limits(baseline_plant_xy, perturbed_plant_xy)
    controller_limits = _limits(baseline_controller_xy, perturbed_controller_xy)
    flash_window_stop = int(perturb_phase_start) + 18

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(19.2, 10.8),
        dpi=100,
        gridspec_kw={"width_ratios": [1.1, 1.9, 1.1]},
    )
    fig.patch.set_facecolor("#fbfaf7")
    flash_overlay = Rectangle(
        (0.0, 0.0),
        1.0,
        1.0,
        transform=fig.transFigure,
        facecolor="#fff4bf",
        alpha=0.0,
        zorder=10,
    )
    fig.add_artist(flash_overlay)
    perturb_text = fig.text(
        0.5,
        0.985,
        "PERTURBATION",
        ha="center",
        va="top",
        fontsize=22,
        fontweight="bold",
        color="#9f1239",
        alpha=0.0,
        zorder=11,
    )
    status_text = fig.text(
        0.5,
        0.952,
        "",
        ha="center",
        va="top",
        fontsize=11,
        color="#44403c",
        zorder=11,
    )

    for ax in axes.ravel():
        ax.set_facecolor("#fffdf8")
        for spine in ax.spines.values():
            spine.set_color("#d6d3d1")

    axes[0, 0].set_title("Plant PCA", fontsize=13, fontweight="bold")
    axes[0, 1].set_title("Robot Render", fontsize=13, fontweight="bold")
    axes[0, 2].set_title("Controller PCA", fontsize=13, fontweight="bold")

    axes[0, 0].set_ylabel("Unperturbed", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Perturbed", fontsize=12, fontweight="bold")

    top_robot = axes[0, 1].imshow(baseline_robot_images[0], aspect="auto")
    bottom_robot = axes[1, 1].imshow(perturbed_robot_images[0], aspect="auto")
    for ax in (axes[0, 1], axes[1, 1]):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(0.72)

    plant_line_top, = axes[0, 0].plot([], [], color="#2563eb", linewidth=2.2)
    plant_line_bottom, = axes[1, 0].plot([], [], color="#d97706", linewidth=2.2)
    plant_dot_top = axes[0, 0].scatter([], [], s=48, color="#1d4ed8", zorder=5)
    plant_dot_bottom = axes[1, 0].scatter([], [], s=48, color="#b45309", zorder=5)
    ctrl_line_top, = axes[0, 2].plot([], [], color="#2563eb", linewidth=2.2)
    ctrl_line_bottom, = axes[1, 2].plot([], [], color="#d97706", linewidth=2.2)
    ctrl_dot_top = axes[0, 2].scatter([], [], s=48, color="#1d4ed8", zorder=5)
    ctrl_dot_bottom = axes[1, 2].scatter([], [], s=48, color="#b45309", zorder=5)

    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_xlim(plant_limits[0], plant_limits[1])
        ax.set_ylim(plant_limits[2], plant_limits[3])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.18, linewidth=0.6)
    for ax in (axes[0, 2], axes[1, 2]):
        ax.set_xlim(controller_limits[0], controller_limits[1])
        ax.set_ylim(controller_limits[2], controller_limits[3])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.18, linewidth=0.6)

    sampled_baseline_plant = baseline_plant_xy[frame_phase_indices]
    sampled_perturbed_plant = perturbed_plant_xy[frame_phase_indices]
    sampled_baseline_ctrl = baseline_controller_xy[frame_phase_indices]
    sampled_perturbed_ctrl = perturbed_controller_xy[frame_phase_indices]

    for frame_idx, phase_idx in enumerate(frame_phase_indices):
        is_flash = int(perturb_phase_start) <= int(phase_idx) < int(flash_window_stop)
        top_robot.set_data(baseline_robot_images[frame_idx])
        bottom_robot.set_data(perturbed_robot_images[frame_idx])

        plant_line_top.set_data(
            sampled_baseline_plant[: frame_idx + 1, 0],
            sampled_baseline_plant[: frame_idx + 1, 1],
        )
        plant_line_bottom.set_data(
            sampled_perturbed_plant[: frame_idx + 1, 0],
            sampled_perturbed_plant[: frame_idx + 1, 1],
        )
        plant_dot_top.set_offsets(sampled_baseline_plant[frame_idx : frame_idx + 1])
        plant_dot_bottom.set_offsets(sampled_perturbed_plant[frame_idx : frame_idx + 1])

        ctrl_line_top.set_data(
            sampled_baseline_ctrl[: frame_idx + 1, 0],
            sampled_baseline_ctrl[: frame_idx + 1, 1],
        )
        ctrl_line_bottom.set_data(
            sampled_perturbed_ctrl[: frame_idx + 1, 0],
            sampled_perturbed_ctrl[: frame_idx + 1, 1],
        )
        ctrl_dot_top.set_offsets(sampled_baseline_ctrl[frame_idx : frame_idx + 1])
        ctrl_dot_bottom.set_offsets(sampled_perturbed_ctrl[frame_idx : frame_idx + 1])

        flash_overlay.set_alpha(0.24 if is_flash else 0.0)
        perturb_text.set_alpha(1.0 if is_flash else 0.0)
        status_text.set_text(f"Phase {int(phase_idx) + 1} / {int(frame_phase_indices[-1]) + 1}")

        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
        fig.savefig(frames_dir / f"frame_{frame_idx:04d}.png", dpi=100)

    plt.close(fig)

    mp4_path = output_dir / mp4_name
    gif_path = output_dir / gif_name
    palette_path = output_dir / "palette.png"

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

    return {
        "frames_dir": str(frames_dir),
        "mp4_path": str(mp4_path),
        "gif_path": str(gif_path),
    }


def run_assay(
    *,
    bundle_root: Path,
    controller_params_path: Path,
    config_snapshot_path: Path,
    output_dir: Path,
    max_steps: int,
    resume_env_step: int | None,
    perturb_local_cell_index: int | None,
    perturb_strength: float,
    support_duration_env_steps: float,
    robot_y_exaggeration: float,
    seed: int,
    fps: int,
    max_frames: int,
) -> dict[str, Any]:
    _ensure_import_path()
    from jax_evogym.constants import GRAVITY

    cfg = _load_cfg(config_snapshot_path, max_steps=int(max_steps))
    env, render_info, alive_mask = _build_free_space_env(cfg)
    params = _load_controller_params(controller_params_path)

    if resume_env_step is None:
        resolved_resume_env_step = max(1, int(max_steps) // 4)
    else:
        resolved_resume_env_step = int(resume_env_step)
    if resolved_resume_env_step >= int(max_steps):
        raise ValueError("resume_env_step must be smaller than max_steps.")

    baseline_result_phase, baseline_phase_trace = _run_phase_probe(
        env=env,
        params=params,
        alive_mask=alive_mask,
        cfg=cfg,
        seed=int(seed),
        max_steps=int(max_steps),
    )
    baseline_result_controller, baseline_controller_trace = _run_controller_trace(
        env=env,
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

    baseline_phase_arrays = _flatten_phase_trace(baseline_phase_trace)
    baseline_controller_phase = _reduce_controller_trace(
        baseline_controller_trace,
        nca_steps_per_phase=nca_steps_per_phase,
    )
    cell_count = int(_robot_render_indices(render_info).shape[0])
    resolved_local_cell_index = (
        _default_release_local_cell_index(
            render_info=render_info,
            positions=baseline_phase_arrays["initial_positions"],
        )
        if perturb_local_cell_index is None
        else int(perturb_local_cell_index)
    )
    if not (0 <= resolved_local_cell_index < cell_count):
        raise ValueError("perturb_local_cell_index fell outside the robot body.")
    support_release = _support_release_profile(
        dt=float(np.asarray(env.constants.dt).item()),
        gravity=float(GRAVITY),
        physics_substeps=int(cfg.sim.physics_substeps),
        duration_env_steps=float(support_duration_env_steps),
        perturb_strength=float(perturb_strength),
    )

    resumed_env, initial_nca_state, initial_ctrnn_carry = _resume_branch_env(
        env=env,
        render_info=render_info,
        controller_phase=baseline_controller_phase,
        phase_arrays=baseline_phase_arrays,
        resume_env_step=int(resolved_resume_env_step),
        sense_act_phases=int(sense_act_phases),
        perturb_local_cell_index=int(resolved_local_cell_index),
        support_release_profile=support_release,
        apply_perturbation=True,
    )

    perturbed_result_phase, perturbed_phase_trace = _run_phase_probe(
        env=resumed_env,
        params=params,
        alive_mask=alive_mask,
        cfg=cfg,
        seed=int(seed),
        max_steps=int(remaining_steps),
        initial_nca_state=initial_nca_state,
        initial_ctrnn_carry=initial_ctrnn_carry,
    )
    perturbed_result_controller, perturbed_controller_trace = _run_controller_trace(
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

    perturbed_phase_arrays_tail = _flatten_phase_trace(perturbed_phase_trace)
    perturbed_controller_phase_tail = _reduce_controller_trace(
        perturbed_controller_trace,
        nca_steps_per_phase=nca_steps_per_phase,
    )

    baseline_positions = baseline_phase_arrays["phase_post_positions"]
    baseline_spring_rest = baseline_phase_arrays["phase_post_spring_rest_length"]
    baseline_controller_flat = baseline_controller_phase["controller_flat"]

    perturbed_positions = _concat_branch_series(
        baseline_positions,
        perturbed_phase_arrays_tail["phase_post_positions"],
        prefix_phase_count=int(prefix_phase_count),
    )
    perturbed_spring_rest = _concat_branch_series(
        baseline_spring_rest,
        perturbed_phase_arrays_tail["phase_post_spring_rest_length"],
        prefix_phase_count=int(prefix_phase_count),
    )
    perturbed_controller_flat = _concat_branch_series(
        baseline_controller_flat,
        perturbed_controller_phase_tail["controller_flat"],
        prefix_phase_count=int(prefix_phase_count),
    )

    baseline_plant_features = _compute_abs_deformation_features(
        baseline_positions,
        topology=env.topology,
        spring_init_rest_length=env.sim_state.spring_init_rest_length,
        deformation_info=env.deformation_info,
    )
    perturbed_plant_features = _compute_abs_deformation_features(
        perturbed_positions,
        topology=env.topology,
        spring_init_rest_length=env.sim_state.spring_init_rest_length,
        deformation_info=env.deformation_info,
    )

    plant_pca = _fit_pca(baseline_plant_features, perturbed_plant_features)
    controller_pca = _fit_pca(baseline_controller_flat, perturbed_controller_flat)
    baseline_plant_xy = _transform_pca(baseline_plant_features, plant_pca)
    perturbed_plant_xy = _transform_pca(perturbed_plant_features, plant_pca)
    baseline_controller_xy = _transform_pca(baseline_controller_flat, controller_pca)
    perturbed_controller_xy = _transform_pca(perturbed_controller_flat, controller_pca)

    sample_phase_indices = _sample_phase_indices(
        baseline_positions.shape[0],
        max_frames=int(max_frames),
        required_phase_index=int(prefix_phase_count),
    )
    camera_center, viewport_width = _camera_spec_from_sequences(
        baseline_positions,
        perturbed_positions,
        render_info=render_info,
        sample_phase_indices=sample_phase_indices,
        panel_aspect=900.0 / 340.0,
    )
    baseline_robot_images = _render_robot_images(
        positions_seq=baseline_positions,
        spring_seq=baseline_spring_rest,
        render_info=render_info,
        sample_phase_indices=sample_phase_indices,
        camera_center=camera_center,
        viewport_width=viewport_width,
        local_cell_index=int(resolved_local_cell_index),
        flash_window_phase_start=int(prefix_phase_count),
        flash_window_phase_stop=int(prefix_phase_count) + 18,
        branch_is_perturbed=False,
        trail_length=6,
    )
    perturbed_robot_images = _render_robot_images(
        positions_seq=perturbed_positions,
        spring_seq=perturbed_spring_rest,
        render_info=render_info,
        sample_phase_indices=sample_phase_indices,
        camera_center=camera_center,
        viewport_width=viewport_width,
        local_cell_index=int(resolved_local_cell_index),
        flash_window_phase_start=int(prefix_phase_count),
        flash_window_phase_stop=int(prefix_phase_count) + 18,
        branch_is_perturbed=True,
        trail_length=6,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "branch_traces.npz",
        sample_phase_indices=sample_phase_indices,
        baseline_plant_xy=baseline_plant_xy,
        perturbed_plant_xy=perturbed_plant_xy,
        baseline_controller_xy=baseline_controller_xy,
        perturbed_controller_xy=perturbed_controller_xy,
        baseline_plant_features=baseline_plant_features,
        perturbed_plant_features=perturbed_plant_features,
        baseline_controller_flat=baseline_controller_flat,
        perturbed_controller_flat=perturbed_controller_flat,
    )

    encoded = _save_frames_and_encode(
        output_dir=output_dir,
        frame_phase_indices=sample_phase_indices,
        baseline_plant_xy=baseline_plant_xy,
        perturbed_plant_xy=perturbed_plant_xy,
        baseline_controller_xy=baseline_controller_xy,
        perturbed_controller_xy=perturbed_controller_xy,
        baseline_robot_images=baseline_robot_images,
        perturbed_robot_images=perturbed_robot_images,
        local_cell_index=int(resolved_local_cell_index),
        perturb_phase_start=int(prefix_phase_count),
        fps=int(fps),
        mp4_name="free_space_branch.mp4",
        gif_name="free_space_branch.gif",
        robot_y_exaggeration=float(robot_y_exaggeration),
    )

    post_slice = slice(int(prefix_phase_count), None)
    plant_post_delta = np.linalg.norm(
        baseline_plant_features[post_slice] - perturbed_plant_features[post_slice],
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
        "max_steps": int(max_steps),
        "sense_act_phases": int(sense_act_phases),
        "nca_steps_per_phase": int(nca_steps_per_phase),
        "resume_env_step": int(resolved_resume_env_step),
        "perturb_phase_index": int(prefix_phase_count),
        "remaining_steps": int(remaining_steps),
        "perturb_local_cell_index": int(resolved_local_cell_index),
        "perturb_strength": float(perturb_strength),
        "support_duration_env_steps": float(support_duration_env_steps),
        "support_release_duration_seconds": float(support_release["duration_seconds"]),
        "support_release_delta_y": float(support_release["delta_y"]),
        "support_release_delta_vy": float(support_release["delta_vy"]),
        "robot_y_exaggeration": float(robot_y_exaggeration),
        "plant_post_delta_mean": float(np.mean(plant_post_delta)),
        "plant_post_delta_max": float(np.max(plant_post_delta)),
        "controller_post_delta_mean": float(np.mean(controller_post_delta)),
        "controller_post_delta_max": float(np.max(controller_post_delta)),
        **encoded,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a free-space branched rollout with plant/controller PCA panels.",
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT,
        help="Specimen bundle root. Defaults to the canonical local S4 bundle.",
    )
    parser.add_argument(
        "--controller-params",
        type=Path,
        default=None,
        help="Override path to controller_params.npz.",
    )
    parser.add_argument(
        "--config-snapshot",
        type=Path,
        default=None,
        help="Override path to config_snapshot.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory. Defaults under analysis_runs/free_space_branch_pca.",
    )
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument("--resume-env-step", type=int, default=None)
    parser.add_argument("--perturb-local-cell-index", type=int, default=None)
    parser.add_argument(
        "--perturb-strength",
        type=float,
        default=1.0,
        help="Scale factor for the support-release perturbation.",
    )
    parser.add_argument(
        "--support-duration-env-steps",
        type=float,
        default=1.0,
        help="Equivalent unsupported duration, in env steps, used to size the release perturbation.",
    )
    parser.add_argument(
        "--robot-y-exaggeration",
        type=float,
        default=6.0,
        help="Vertical exaggeration factor for the robot panel.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=240)
    args = parser.parse_args()

    bundle_root = args.bundle_root.resolve()
    controller_params_path = (
        args.controller_params.resolve()
        if args.controller_params is not None
        else (bundle_root / "controller_params.npz").resolve()
    )
    config_snapshot_path = (
        args.config_snapshot.resolve()
        if args.config_snapshot is not None
        else (bundle_root / "config_snapshot.yaml").resolve()
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (
            bundle_root
            / "analysis_runs"
            / f"free_space_branch_pca_steps_{int(args.max_steps)}"
        ).resolve()
    )

    summary = run_assay(
        bundle_root=bundle_root,
        controller_params_path=controller_params_path,
        config_snapshot_path=config_snapshot_path,
        output_dir=output_dir,
        max_steps=int(args.max_steps),
        resume_env_step=args.resume_env_step,
        perturb_local_cell_index=args.perturb_local_cell_index,
        perturb_strength=float(args.perturb_strength),
        support_duration_env_steps=float(args.support_duration_env_steps),
        robot_y_exaggeration=float(args.robot_y_exaggeration),
        seed=int(args.seed),
        fps=int(args.fps),
        max_frames=int(args.max_frames),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
