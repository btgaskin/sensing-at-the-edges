from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import jax
import numpy as np

from free_space_branch_common import (
    DEFAULT_BUNDLE_ROOT,
    ensure_import_path,
    fit_pca,
    flatten_phase_trace,
    load_cfg,
    load_controller_params,
    reduce_controller_trace,
    transform_pca,
    write_json,
)
from terrain_width_compare_probe import (
    _phase_env_steps,
    _render_robot_compare,
    _render_state_pca,
    _robot_corner_com_features,
    _save_trace_bundle,
    _trim_pair,
    _valid_event_phase_index,
)


RESULT_ROOT = DEFAULT_BUNDLE_ROOT
DEFAULT_OUTPUT_DIR = RESULT_ROOT / "analysis_runs" / "right_native_goalblock_doublecliff_probe"
DEFAULT_CONFIG_PATH = RESULT_ROOT / "config_snapshot.yaml"
DEFAULT_PARAMS_PATH = RESULT_ROOT / "controller_params.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare right-native (2,1) with goal block variants and against native double-cliff 10.",
    )
    parser.add_argument("--config-snapshot-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--controller-params-path", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--goal-distance", type=int, default=2)
    parser.add_argument("--cliff-distance", type=int, default=1)
    parser.add_argument("--goal-side", type=str, default="right")
    parser.add_argument("--double-cliff-width", type=int, default=10)
    parser.add_argument("--double-cliff-plateau-width", type=int, default=None)
    parser.add_argument("--deformation-condition", type=str, default="deformation_off")
    parser.add_argument("--signaling-condition", type=str, default="intact")
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--flash-frames", type=int, default=8)
    parser.add_argument("--supersample", type=int, default=2)
    return parser.parse_args()


def _robot_render_indices(render_info) -> np.ndarray:
    return np.flatnonzero(
        np.asarray(render_info.cell_is_robot, dtype=bool)
        & np.asarray(render_info.cell_is_dynamic, dtype=bool)
        & np.asarray(render_info.cell_mask, dtype=bool)
    ).astype(np.int32, copy=False)


def _edge_lengths_from_cell(cell_points: np.ndarray) -> tuple[float, float]:
    x = np.asarray(cell_points[:, 0], dtype=np.float32)
    y = np.asarray(cell_points[:, 1], dtype=np.float32)
    top_idx = np.argsort(y)[-2:]
    bot_idx = np.argsort(y)[:2]
    left_idx = np.argsort(x)[:2]
    right_idx = np.argsort(x)[-2:]
    h_top = float(np.linalg.norm(cell_points[top_idx[0]] - cell_points[top_idx[1]]))
    h_bot = float(np.linalg.norm(cell_points[bot_idx[0]] - cell_points[bot_idx[1]]))
    v_left = float(np.linalg.norm(cell_points[left_idx[0]] - cell_points[left_idx[1]]))
    v_right = float(np.linalg.norm(cell_points[right_idx[0]] - cell_points[right_idx[1]]))
    return 0.5 * (h_top + h_bot), 0.5 * (v_left + v_right)


def _compute_robot_cell_hv_ratios(
    *,
    positions_seq: np.ndarray,
    initial_positions: np.ndarray,
    render_info,
) -> np.ndarray:
    robot_indices = _robot_render_indices(render_info)
    cell_vertices = np.asarray(render_info.cell_vertices[robot_indices], dtype=np.int32)
    cell_vertex_count = np.asarray(render_info.cell_vertex_count[robot_indices], dtype=np.int32)
    cell_count = int(cell_vertices.shape[0])
    if cell_count <= 0:
        raise ValueError("No robot render cells were found.")
    ref_h = np.zeros((cell_count,), dtype=np.float32)
    ref_v = np.zeros((cell_count,), dtype=np.float32)
    for cell_idx in range(cell_count):
        count = int(cell_vertex_count[cell_idx])
        points = np.asarray(initial_positions[cell_vertices[cell_idx, :count]], dtype=np.float32)
        h, v = _edge_lengths_from_cell(points)
        ref_h[cell_idx] = max(h, 1.0e-6)
        ref_v[cell_idx] = max(v, 1.0e-6)

    features = np.zeros((positions_seq.shape[0], cell_count * 2), dtype=np.float32)
    for phase_idx, positions in enumerate(positions_seq):
        h_vals = np.zeros((cell_count,), dtype=np.float32)
        v_vals = np.zeros((cell_count,), dtype=np.float32)
        for cell_idx in range(cell_count):
            count = int(cell_vertex_count[cell_idx])
            points = np.asarray(positions[cell_vertices[cell_idx, :count]], dtype=np.float32)
            h, v = _edge_lengths_from_cell(points)
            h_vals[cell_idx] = float(h) / float(ref_h[cell_idx])
            v_vals[cell_idx] = float(v) / float(ref_v[cell_idx])
        features[phase_idx] = np.concatenate([h_vals, v_vals], axis=0)
    return features


def _centered_max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_centered = a - np.mean(a, axis=0, keepdims=True)
    b_centered = b - np.mean(b, axis=0, keepdims=True)
    return float(np.max(np.abs(a_centered - b_centered)))


def _first_diff_phase(a: np.ndarray, b: np.ndarray, *, atol: float = 1.0e-8) -> int | None:
    diffs = np.max(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)), axis=1)
    hits = np.flatnonzero(diffs > float(atol))
    if hits.size == 0:
        return None
    return int(hits[0])


def _native_edge_event_phase(run: dict[str, Any], *, phase_count: int) -> int | None:
    from jax_evogym.constants import CELL_SIZE

    metadata = dict(run["entry_metadata"])
    cliff_side = str(metadata.get("cliff_side", "")).strip().lower()
    cliff_edge_x = int(metadata.get("terminal_cliff_edge_x", metadata.get("cliff_edge_x", -1)))
    if cliff_edge_x < 0:
        return None
    if cliff_side == "right":
        hits = np.flatnonzero(
            np.asarray(run["front_x_phase"][:phase_count], dtype=np.float32)
            > (float(cliff_edge_x + 1) * float(CELL_SIZE))
        )
    elif cliff_side == "left":
        hits = np.flatnonzero(
            np.asarray(run["back_x_phase"][:phase_count], dtype=np.float32)
            < (float(cliff_edge_x) * float(CELL_SIZE))
        )
    else:
        return None
    if hits.size == 0:
        return None
    return int(hits[0])


def _capture_native_variant(
    *,
    cfg,
    params,
    goal_distance: int,
    cliff_distance: int,
    goal_side: str,
    goal_block_present: bool,
    disable_goal_touch: bool,
    deformation_condition: str,
    signaling_condition: str,
    seed: int,
) -> dict[str, Any]:
    ensure_import_path()
    from jax_evogym.constants import EMPTY
    from sensing_edges.evaluate import _build_phase_assets
    from sensing_edges.env import run_episode_with_controller_trace, run_episode_with_phase_probe
    from sensing_edges.post_run import _build_bilateral_entry, _native_phase_cfg
    from sensing_edges.runtime_config import resolve_runtime_config, rollout_kwargs_from_cfg
    from sensing_edges.terrain_schedule import TerrainEntry

    phase_cfg = _native_phase_cfg(
        cfg,
        active_deformation_condition=str(deformation_condition),
        active_signaling_condition=str(signaling_condition),
    )
    phase_cfg.train.max_steps = int(cfg.train.max_steps)
    runtime_resolution = resolve_runtime_config(
        phase_cfg,
        preset=(str(phase_cfg.get("preset_id")) if phase_cfg.get("preset_id") is not None else None),
    )
    base_entry = _build_bilateral_entry(
        phase_cfg,
        goal_distance=int(goal_distance),
        cliff_distance=int(cliff_distance),
        goal_side=str(goal_side),
        extra_metadata={"analysis_mode": "right_native_goalblock_doublecliff_probe"},
    )
    grid = np.asarray(base_entry.grid, dtype=np.int32).copy()
    metadata = dict(base_entry.metadata or {})
    goal_block_x = int(metadata.get("goal_block_x", -1))
    goal_block_y = int(metadata.get("goal_block_y", -1))
    if (not goal_block_present) and goal_block_x >= 0 and goal_block_y >= 0:
        grid[goal_block_y, goal_block_x] = EMPTY
    if disable_goal_touch:
        metadata["goal_block_x"] = -1
        metadata["goal_block_y"] = -1
        metadata["goal_x"] = -1.0
        metadata["goal_object_id"] = 0
    metadata["goal_block_geometry_present"] = bool(goal_block_present)
    entry = TerrainEntry(
        grid=grid,
        cliff_x=int(base_entry.cliff_x),
        gaps=list(base_entry.gaps),
        metadata=metadata,
    )
    _phase_resolution, envs, alive_mask, _ = _build_phase_assets(
        phase_cfg,
        runtime_resolution,
        [entry],
    )
    env = envs[0]
    rollout_kwargs = rollout_kwargs_from_cfg(phase_cfg)
    rollout_kwargs["max_steps"] = int(phase_cfg.train.max_steps)
    phase_result, phase_trace = run_episode_with_phase_probe(
        env_data=env,
        params=params,
        alive_mask=alive_mask,
        key=jax.random.PRNGKey(int(seed)),
        **rollout_kwargs,
    )
    _controller_result, controller_trace = run_episode_with_controller_trace(
        env_data=env,
        params=params,
        alive_mask=alive_mask,
        key=jax.random.PRNGKey(int(seed)),
        **rollout_kwargs,
    )
    phase_arrays = flatten_phase_trace(phase_trace)
    controller_phase = reduce_controller_trace(
        controller_trace,
        nca_steps_per_phase=int(phase_cfg.nca.nca_steps_per_phase),
    )
    initial_positions = np.asarray(phase_arrays["initial_positions"], dtype=np.float32)
    positions = np.asarray(phase_arrays["phase_post_positions"], dtype=np.float32)
    spring_rest = np.asarray(phase_arrays["phase_post_spring_rest_length"], dtype=np.float32)
    robot_point_mask = np.asarray(env.robot_point_mask, dtype=bool)
    if not np.any(robot_point_mask):
        raise ValueError("Robot point mask is empty.")
    robot_positions = positions[:, robot_point_mask, :]
    plant_features = _compute_robot_cell_hv_ratios(
        positions_seq=positions,
        initial_positions=initial_positions,
        render_info=env.render_info,
    )
    return {
        "phase_result": phase_result,
        "phase_arrays": phase_arrays,
        "initial_positions": initial_positions,
        "positions": positions,
        "robot_positions": robot_positions,
        "robot_point_mask": robot_point_mask,
        "spring_rest": spring_rest,
        "plant_features": plant_features,
        "controller_features": np.asarray(controller_phase["controller_flat"], dtype=np.float32),
        "front_x_phase": np.max(robot_positions[..., 0], axis=1).astype(np.float32, copy=False),
        "back_x_phase": np.min(robot_positions[..., 0], axis=1).astype(np.float32, copy=False),
        "render_info": env.render_info,
        "entry_metadata": metadata,
        "phase_cfg": phase_cfg,
    }


def _capture_double_cliff_variant(
    *,
    cfg,
    params,
    width: int,
    plateau_width: int,
    deformation_condition: str,
    signaling_condition: str,
    seed: int,
) -> dict[str, Any]:
    ensure_import_path()
    from sensing_edges.evaluate import _build_phase_assets
    from sensing_edges.env import run_episode_with_controller_trace, run_episode_with_phase_probe
    from sensing_edges.post_run import _build_double_cliff_entry, _double_cliff_phase_cfg
    from sensing_edges.runtime_config import resolve_runtime_config, rollout_kwargs_from_cfg

    phase_cfg = _double_cliff_phase_cfg(
        cfg,
        active_deformation_condition=str(deformation_condition),
        active_signaling_condition=str(signaling_condition),
    )
    phase_cfg.train.max_steps = int(cfg.train.max_steps)
    runtime_resolution = resolve_runtime_config(
        phase_cfg,
        preset=(str(phase_cfg.get("preset_id")) if phase_cfg.get("preset_id") is not None else None),
    )
    entry = _build_double_cliff_entry(
        phase_cfg,
        width=int(width),
        plateau_width=int(plateau_width),
        extra_metadata={"analysis_mode": "right_native_goalblock_doublecliff_probe"},
    )
    _phase_resolution, envs, alive_mask, _ = _build_phase_assets(
        phase_cfg,
        runtime_resolution,
        [entry],
    )
    env = envs[0]
    rollout_kwargs = rollout_kwargs_from_cfg(phase_cfg)
    rollout_kwargs["max_steps"] = int(phase_cfg.train.max_steps)
    phase_result, phase_trace = run_episode_with_phase_probe(
        env_data=env,
        params=params,
        alive_mask=alive_mask,
        key=jax.random.PRNGKey(int(seed)),
        **rollout_kwargs,
    )
    _controller_result, controller_trace = run_episode_with_controller_trace(
        env_data=env,
        params=params,
        alive_mask=alive_mask,
        key=jax.random.PRNGKey(int(seed)),
        **rollout_kwargs,
    )
    phase_arrays = flatten_phase_trace(phase_trace)
    controller_phase = reduce_controller_trace(
        controller_trace,
        nca_steps_per_phase=int(phase_cfg.nca.nca_steps_per_phase),
    )
    initial_positions = np.asarray(phase_arrays["initial_positions"], dtype=np.float32)
    positions = np.asarray(phase_arrays["phase_post_positions"], dtype=np.float32)
    spring_rest = np.asarray(phase_arrays["phase_post_spring_rest_length"], dtype=np.float32)
    robot_point_mask = np.asarray(env.robot_point_mask, dtype=bool)
    if not np.any(robot_point_mask):
        raise ValueError("Robot point mask is empty.")
    robot_positions = positions[:, robot_point_mask, :]
    plant_features = _compute_robot_cell_hv_ratios(
        positions_seq=positions,
        initial_positions=initial_positions,
        render_info=env.render_info,
    )
    return {
        "phase_result": phase_result,
        "phase_arrays": phase_arrays,
        "initial_positions": initial_positions,
        "positions": positions,
        "robot_positions": robot_positions,
        "robot_point_mask": robot_point_mask,
        "spring_rest": spring_rest,
        "plant_features": plant_features,
        "controller_features": np.asarray(controller_phase["controller_flat"], dtype=np.float32),
        "front_x_phase": np.max(robot_positions[..., 0], axis=1).astype(np.float32, copy=False),
        "back_x_phase": np.min(robot_positions[..., 0], axis=1).astype(np.float32, copy=False),
        "render_info": env.render_info,
        "entry_metadata": dict(entry.metadata),
        "phase_cfg": phase_cfg,
    }


def _compare_runs(
    *,
    run_a: dict[str, Any],
    run_b: dict[str, Any],
    output_dir: Path,
    top_label: str,
    bottom_label: str,
    event_phase_index: int | None,
    event_name: str,
) -> dict[str, Any]:
    pos_a, pos_b, phase_count = _trim_pair(run_a["positions"], run_b["positions"])
    spring_a, spring_b, _ = _trim_pair(run_a["spring_rest"], run_b["spring_rest"])
    plant_a, plant_b, _ = _trim_pair(run_a["plant_features"], run_b["plant_features"])
    ctrl_a, ctrl_b, _ = _trim_pair(run_a["controller_features"], run_b["controller_features"])
    aug_a = _robot_corner_com_features(
        positions_seq=pos_a,
        robot_point_mask=np.asarray(run_a["robot_point_mask"], dtype=bool),
    )
    aug_b = _robot_corner_com_features(
        positions_seq=pos_b,
        robot_point_mask=np.asarray(run_b["robot_point_mask"], dtype=bool),
    )
    plant_pca = fit_pca(plant_a, plant_b, n_components=3)
    ctrl_pca = fit_pca(ctrl_a, ctrl_b, n_components=3)
    aug_pca = fit_pca(aug_a, aug_b, n_components=3)
    plant_a_xy = transform_pca(plant_a, plant_pca)
    plant_b_xy = transform_pca(plant_b, plant_pca)
    ctrl_a_xy = transform_pca(ctrl_a, ctrl_pca)
    ctrl_b_xy = transform_pca(ctrl_b, ctrl_pca)
    aug_a_xy = transform_pca(aug_a, aug_pca)
    aug_b_xy = transform_pca(aug_b, aug_pca)
    sense_act_phases = int(run_a["phase_cfg"].sim.sense_act_phases_per_env_step)
    env_steps = _phase_env_steps(phase_count, sense_act_phases=sense_act_phases)
    event_phase_index = _valid_event_phase_index(event_phase_index, phase_count=int(phase_count))

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_trace_bundle(
        output_dir / "trace_bundle.npz",
        {
            "top_positions": pos_a,
            "bottom_positions": pos_b,
            "top_spring_rest_length": spring_a,
            "bottom_spring_rest_length": spring_b,
            "top_raw_abs_hv": plant_a,
            "bottom_raw_abs_hv": plant_b,
            "top_controller_flat": ctrl_a,
            "bottom_controller_flat": ctrl_b,
            "top_corner_com": aug_a,
            "bottom_corner_com": aug_b,
            "env_steps": env_steps,
            "event_phase_index": np.asarray([-1 if event_phase_index is None else int(event_phase_index)], dtype=np.int32),
        },
    )
    np.savez_compressed(
        output_dir / "state_pca_coords.npz",
        top_plant_xy=plant_a_xy,
        bottom_plant_xy=plant_b_xy,
        top_controller_xy=ctrl_a_xy,
        bottom_controller_xy=ctrl_b_xy,
        top_aug_xy=aug_a_xy,
        bottom_aug_xy=aug_b_xy,
    )

    robot_summary = _render_robot_compare(
        output_dir=output_dir / "robot_compare",
        plateau_run={
            "positions": pos_a,
            "spring_rest": spring_a,
            "render_info": run_a["render_info"],
        },
        ledge_run={
            "positions": pos_b,
            "spring_rest": spring_b,
            "render_info": run_b["render_info"],
        },
        phase_count=int(phase_count),
        sense_act_phases=int(sense_act_phases),
        event_phase_index=event_phase_index,
        fps=int(args.fps),
        max_frames=int(args.max_frames),
        flash_frames=int(args.flash_frames),
        supersample=int(args.supersample),
        top_width_cells=int(run_a["entry_metadata"]["platform_end_x"] - run_a["entry_metadata"]["platform_start_x"] + 1),
        bottom_width_cells=int(run_b["entry_metadata"]["platform_end_x"] - run_b["entry_metadata"]["platform_start_x"] + 1),
        top_label=str(top_label),
        bottom_label=str(bottom_label),
    )
    pca_basic_summary = _render_state_pca(
        output_dir=output_dir / "state_pca_basic",
        plateau_plant_xy=plant_a_xy,
        ledge_plant_xy=plant_b_xy,
        plateau_ctrl_xy=ctrl_a_xy,
        ledge_ctrl_xy=ctrl_b_xy,
        env_steps=env_steps,
        event_phase_index=event_phase_index,
        fps=int(args.fps),
        max_frames=int(args.max_frames),
        flash_frames=int(args.flash_frames),
    )
    pca_extended_summary = _render_state_pca(
        output_dir=output_dir / "state_pca_extended",
        plateau_plant_xy=plant_a_xy,
        ledge_plant_xy=plant_b_xy,
        plateau_ctrl_xy=ctrl_a_xy,
        ledge_ctrl_xy=ctrl_b_xy,
        plateau_aug_xy=aug_a_xy,
        ledge_aug_xy=aug_b_xy,
        aug_title="Plant PCA + COM/corners",
        env_steps=env_steps,
        event_phase_index=event_phase_index,
        fps=int(args.fps),
        max_frames=int(args.max_frames),
        flash_frames=int(args.flash_frames),
    )

    summary = {
        "top_label": str(top_label),
        "bottom_label": str(bottom_label),
        "phase_count": int(phase_count),
        "sense_act_phases": int(sense_act_phases),
        "nca_steps_per_phase": int(run_a["phase_cfg"].nca.nca_steps_per_phase),
        "event_name": str(event_name),
        "event_phase_index": None if event_phase_index is None else int(event_phase_index),
        "event_env_step": None if event_phase_index is None else int(env_steps[int(event_phase_index)]),
        "initial_centered_robot_max_abs_diff": _centered_max_abs_diff(
            np.asarray(run_a["initial_positions"], dtype=np.float32)[np.asarray(run_a["robot_point_mask"], dtype=bool)],
            np.asarray(run_b["initial_positions"], dtype=np.float32)[np.asarray(run_b["robot_point_mask"], dtype=bool)],
        ),
        "phase0_centered_robot_max_abs_diff": _centered_max_abs_diff(
            pos_a[0][np.asarray(run_a["robot_point_mask"], dtype=bool)],
            pos_b[0][np.asarray(run_b["robot_point_mask"], dtype=bool)],
        ),
        "controller_max_abs_diff": float(np.max(np.abs(ctrl_a - ctrl_b))),
        "plant_first_diff_phase": _first_diff_phase(plant_a, plant_b),
        "robot_render": robot_summary,
        "state_pca_basic": pca_basic_summary,
        "state_pca_extended": pca_extended_summary,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    global args
    args = parse_args()
    cfg = load_cfg(args.config_snapshot_path.resolve(), max_steps=int(args.max_steps))
    params = load_controller_params(args.controller_params_path.resolve())
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    plateau_width = (
        int(args.double_cliff_plateau_width)
        if args.double_cliff_plateau_width is not None
        else int(cfg.eval.post_run.double_cliff.plateau_width)
    )

    native_goal_block = _capture_native_variant(
        cfg=cfg,
        params=params,
        goal_distance=int(args.goal_distance),
        cliff_distance=int(args.cliff_distance),
        goal_side=str(args.goal_side),
        goal_block_present=True,
        disable_goal_touch=True,
        deformation_condition=str(args.deformation_condition),
        signaling_condition=str(args.signaling_condition),
        seed=int(args.seed),
    )
    native_goal_block_touchable = _capture_native_variant(
        cfg=cfg,
        params=params,
        goal_distance=int(args.goal_distance),
        cliff_distance=int(args.cliff_distance),
        goal_side=str(args.goal_side),
        goal_block_present=True,
        disable_goal_touch=False,
        deformation_condition=str(args.deformation_condition),
        signaling_condition=str(args.signaling_condition),
        seed=int(args.seed),
    )
    native_no_goal_block = _capture_native_variant(
        cfg=cfg,
        params=params,
        goal_distance=int(args.goal_distance),
        cliff_distance=int(args.cliff_distance),
        goal_side=str(args.goal_side),
        goal_block_present=False,
        disable_goal_touch=True,
        deformation_condition=str(args.deformation_condition),
        signaling_condition=str(args.signaling_condition),
        seed=int(args.seed),
    )
    double_cliff = _capture_double_cliff_variant(
        cfg=cfg,
        params=params,
        width=int(args.double_cliff_width),
        plateau_width=int(plateau_width),
        deformation_condition=str(args.deformation_condition),
        signaling_condition=str(args.signaling_condition),
        seed=int(args.seed),
    )

    native_event_phase_index = _native_edge_event_phase(
        native_goal_block,
        phase_count=int(min(native_goal_block["positions"].shape[0], native_no_goal_block["positions"].shape[0])),
    )
    compare_goal = _compare_runs(
        run_a=native_goal_block,
        run_b=native_no_goal_block,
        output_dir=output_root / "native_goal_block_compare",
        top_label="Right Native (2,1) Block On",
        bottom_label="Right Native (2,1) Block Off",
        event_phase_index=native_event_phase_index,
        event_name="native_edge",
    )
    touchable_event_phase_index = _native_edge_event_phase(
        native_goal_block_touchable,
        phase_count=int(
            min(
                native_goal_block_touchable["positions"].shape[0],
                native_no_goal_block["positions"].shape[0],
            )
        ),
    )
    compare_goal_touchable = _compare_runs(
        run_a=native_goal_block_touchable,
        run_b=native_no_goal_block,
        output_dir=output_root / "native_goal_block_touchable_compare",
        top_label="Right Native (2,1) Block Touchable",
        bottom_label="Right Native (2,1) Block Off",
        event_phase_index=touchable_event_phase_index,
        event_name="native_edge",
    )
    compare_double = _compare_runs(
        run_a=native_goal_block,
        run_b=double_cliff,
        output_dir=output_root / "native_vs_double_cliff10",
        top_label="Right Native (2,1) Block On",
        bottom_label=f"Double Cliff {int(args.double_cliff_width)}",
        event_phase_index=native_event_phase_index,
        event_name="native_edge",
    )
    compare_double_touchable = _compare_runs(
        run_a=native_goal_block_touchable,
        run_b=double_cliff,
        output_dir=output_root / "native_touchable_vs_double_cliff10",
        top_label="Right Native (2,1) Block Touchable",
        bottom_label=f"Double Cliff {int(args.double_cliff_width)}",
        event_phase_index=touchable_event_phase_index,
        event_name="native_edge",
    )

    summary = {
        "deformation_condition": str(args.deformation_condition),
        "signaling_condition": str(args.signaling_condition),
        "goal_distance": int(args.goal_distance),
        "cliff_distance": int(args.cliff_distance),
        "goal_side": str(args.goal_side),
        "double_cliff_width": int(args.double_cliff_width),
        "double_cliff_plateau_width": int(plateau_width),
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "native_goal_block_compare": compare_goal,
        "native_goal_block_touchable_compare": compare_goal_touchable,
        "native_vs_double_cliff10": compare_double,
        "native_touchable_vs_double_cliff10": compare_double_touchable,
    }
    write_json(output_root / "summary.json", summary)
    print(compare_goal["robot_render"]["mp4_path"])
    print(compare_goal["state_pca_basic"]["mp4_path"])
    print(compare_goal["state_pca_extended"]["mp4_path"])
    print(compare_goal_touchable["robot_render"]["mp4_path"])
    print(compare_goal_touchable["state_pca_basic"]["mp4_path"])
    print(compare_goal_touchable["state_pca_extended"]["mp4_path"])
    print(compare_double["robot_render"]["mp4_path"])
    print(compare_double["state_pca_basic"]["mp4_path"])
    print(compare_double["state_pca_extended"]["mp4_path"])
    print(compare_double_touchable["robot_render"]["mp4_path"])
    print(compare_double_touchable["state_pca_basic"]["mp4_path"])
    print(compare_double_touchable["state_pca_extended"]["mp4_path"])


if __name__ == "__main__":
    main()
