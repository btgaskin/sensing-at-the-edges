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
    _robot_corner_com_features,
    _phase_env_steps,
    _render_robot_compare,
    _render_state_pca,
    _save_trace_bundle,
    _trim_pair,
    _valid_event_phase_index,
)


RESULT_ROOT = DEFAULT_BUNDLE_ROOT
DEFAULT_OUTPUT_DIR = RESULT_ROOT / "analysis_runs" / "native_core_pair_2_1_broad_compare_deformation_off"
DEFAULT_CONFIG_PATH = RESULT_ROOT / "config_snapshot.yaml"
DEFAULT_PARAMS_PATH = RESULT_ROOT / "controller_params.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare native-core (2,1) mirrored tasks against a broadened bilateral terrain.",
    )
    parser.add_argument("--config-snapshot-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--controller-params-path", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--goal-distance", type=int, default=2)
    parser.add_argument("--cliff-distance", type=int, default=1)
    parser.add_argument("--compare-goal-distance", type=int, default=None)
    parser.add_argument("--compare-cliff-distance", type=int, default=None)
    parser.add_argument("--deformation-condition", type=str, default="deformation_off")
    parser.add_argument("--signaling-condition", type=str, default="intact")
    parser.add_argument("--dynamic-impl", type=str, default="sparse_candidate_list")
    parser.add_argument("--sides", nargs="+", default=("left", "right"))
    parser.add_argument("--broad-grid-width", type=int, default=50)
    parser.add_argument("--broad-plateau-width", type=int, default=24)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--flash-frames", type=int, default=8)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--seed", type=int, default=137)
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


def _capture_bilateral(
    *,
    cfg,
    params,
    goal_distance: int,
    cliff_distance: int,
    goal_side: str,
    deformation_condition: str,
    signaling_condition: str,
    dynamic_impl: str,
    seed: int,
    broad_grid_width: int | None,
    broad_plateau_width: int | None,
) -> dict[str, Any]:
    ensure_import_path()
    from sensing_edges.evaluate import _build_phase_assets
    from sensing_edges.env import run_episode_with_controller_trace, run_episode_with_phase_probe
    from sensing_edges.post_run import _build_bilateral_entry, _native_phase_cfg
    from sensing_edges.runtime_config import resolve_runtime_config, rollout_kwargs_from_cfg

    phase_cfg = _native_phase_cfg(
        cfg,
        active_deformation_condition=str(deformation_condition),
        active_signaling_condition=str(signaling_condition),
    )
    phase_cfg.train.max_steps = int(cfg.train.max_steps)
    phase_cfg.collision.dynamic_impl = str(dynamic_impl)
    if broad_grid_width is not None:
        phase_cfg.terrain.grid_w = int(broad_grid_width)
    if broad_plateau_width is not None:
        phase_cfg.terrain.center_plateau_width = int(broad_plateau_width)
        phase_cfg.terrain.base_width = int(broad_plateau_width)

    runtime_resolution = resolve_runtime_config(
        phase_cfg,
        preset=(str(phase_cfg.get("preset_id")) if phase_cfg.get("preset_id") is not None else None),
    )
    entry = _build_bilateral_entry(
        phase_cfg,
        goal_distance=int(goal_distance),
        cliff_distance=int(cliff_distance),
        goal_side=str(goal_side),
        extra_metadata={"analysis_mode": "native_core_task_compare_probe"},
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
        initial_positions=np.asarray(phase_arrays["initial_positions"], dtype=np.float32),
        render_info=env.render_info,
    )
    return {
        "phase_result": phase_result,
        "phase_arrays": phase_arrays,
        "initial_positions": initial_positions,
        "positions": positions,
        "robot_positions": robot_positions,
        "spring_rest": spring_rest,
        "plant_features": plant_features,
        "controller_features": np.asarray(controller_phase["controller_flat"], dtype=np.float32),
        "robot_point_mask": robot_point_mask,
        "front_x_phase": np.max(robot_positions[..., 0], axis=1).astype(np.float32, copy=False),
        "back_x_phase": np.min(robot_positions[..., 0], axis=1).astype(np.float32, copy=False),
        "render_info": env.render_info,
        "entry_metadata": dict(entry.metadata),
        "phase_cfg": phase_cfg,
    }


def _native_edge_event_phase(run: dict[str, Any], *, phase_count: int) -> int | None:
    from jax_evogym.constants import CELL_SIZE

    metadata = dict(run["entry_metadata"])
    cliff_side = str(metadata.get("cliff_side", "")).strip().lower()
    cliff_edge_x = int(metadata.get("terminal_cliff_edge_x", metadata.get("cliff_edge_x", -1)))
    if cliff_edge_x < 0:
        return None
    if cliff_side == "right":
        hits = np.flatnonzero(np.asarray(run["front_x_phase"][:phase_count], dtype=np.float32) > (float(cliff_edge_x + 1) * float(CELL_SIZE)))
    elif cliff_side == "left":
        hits = np.flatnonzero(np.asarray(run["back_x_phase"][:phase_count], dtype=np.float32) < (float(cliff_edge_x) * float(CELL_SIZE)))
    else:
        return None
    if hits.size == 0:
        return None
    return int(hits[0])


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


def _run_side(
    *,
    cfg,
    params,
    output_dir: Path,
    goal_distance: int,
    cliff_distance: int,
    compare_goal_distance: int,
    compare_cliff_distance: int,
    goal_side: str,
    deformation_condition: str,
    signaling_condition: str,
    dynamic_impl: str,
    seed: int,
    broad_grid_width: int,
    broad_plateau_width: int,
    fps: int,
    max_frames: int,
    flash_frames: int,
    supersample: int,
) -> dict[str, Any]:
    native_run = _capture_bilateral(
        cfg=cfg,
        params=params,
        goal_distance=int(goal_distance),
        cliff_distance=int(cliff_distance),
        goal_side=str(goal_side),
        deformation_condition=str(deformation_condition),
        signaling_condition=str(signaling_condition),
        dynamic_impl=str(dynamic_impl),
        seed=int(seed),
        broad_grid_width=None,
        broad_plateau_width=None,
    )
    broad_run = _capture_bilateral(
        cfg=cfg,
        params=params,
        goal_distance=int(compare_goal_distance),
        cliff_distance=int(compare_cliff_distance),
        goal_side=str(goal_side),
        deformation_condition=str(deformation_condition),
        signaling_condition=str(signaling_condition),
        dynamic_impl=str(dynamic_impl),
        seed=int(seed),
        broad_grid_width=int(broad_grid_width),
        broad_plateau_width=int(broad_plateau_width),
    )

    native_positions, broad_positions, phase_count = _trim_pair(
        native_run["positions"],
        broad_run["positions"],
    )
    native_spring, broad_spring, _ = _trim_pair(
        native_run["spring_rest"],
        broad_run["spring_rest"],
    )
    native_plant, broad_plant, _ = _trim_pair(
        native_run["plant_features"],
        broad_run["plant_features"],
    )
    native_ctrl, broad_ctrl, _ = _trim_pair(
        native_run["controller_features"],
        broad_run["controller_features"],
    )
    native_front, broad_front, _ = _trim_pair(
        native_run["front_x_phase"][:, None],
        broad_run["front_x_phase"][:, None],
    )
    native_back, broad_back, _ = _trim_pair(
        native_run["back_x_phase"][:, None],
        broad_run["back_x_phase"][:, None],
    )
    native_front = native_front[:, 0]
    native_back = native_back[:, 0]
    broad_front = broad_front[:, 0]
    broad_back = broad_back[:, 0]

    event_phase_index = _native_edge_event_phase(native_run, phase_count=int(phase_count))
    event_phase_index = _valid_event_phase_index(event_phase_index, phase_count=int(phase_count))
    native_aug = _robot_corner_com_features(
        positions_seq=native_positions,
        robot_point_mask=np.asarray(native_run["robot_point_mask"], dtype=bool),
    )
    broad_aug = _robot_corner_com_features(
        positions_seq=broad_positions,
        robot_point_mask=np.asarray(broad_run["robot_point_mask"], dtype=bool),
    )
    plant_pca = fit_pca(native_plant, broad_plant, n_components=3)
    ctrl_pca = fit_pca(native_ctrl, broad_ctrl, n_components=3)
    aug_pca = fit_pca(native_aug, broad_aug, n_components=3)
    native_plant_xy = transform_pca(native_plant, plant_pca)
    broad_plant_xy = transform_pca(broad_plant, plant_pca)
    native_ctrl_xy = transform_pca(native_ctrl, ctrl_pca)
    broad_ctrl_xy = transform_pca(broad_ctrl, ctrl_pca)
    native_aug_xy = transform_pca(native_aug, aug_pca)
    broad_aug_xy = transform_pca(broad_aug, aug_pca)
    sense_act_phases = int(native_run["phase_cfg"].sim.sense_act_phases_per_env_step)
    env_steps = _phase_env_steps(phase_count, sense_act_phases=sense_act_phases)
    native_robot_initial = np.asarray(native_run["initial_positions"], dtype=np.float32)[
        np.asarray(native_run["robot_point_mask"], dtype=bool)
    ]
    broad_robot_initial = np.asarray(broad_run["initial_positions"], dtype=np.float32)[
        np.asarray(broad_run["robot_point_mask"], dtype=bool)
    ]
    initial_centered_max_diff = _centered_max_abs_diff(native_robot_initial, broad_robot_initial)
    phase0_centered_max_diff = _centered_max_abs_diff(
        np.asarray(native_run["robot_positions"][0], dtype=np.float32),
        np.asarray(broad_run["robot_positions"][0], dtype=np.float32),
    )
    plant_phase0_max_abs_diff = float(np.max(np.abs(native_plant[0] - broad_plant[0])))
    controller_max_abs_diff = float(np.max(np.abs(native_ctrl - broad_ctrl)))
    plant_first_diff_phase = _first_diff_phase(native_plant, broad_plant)

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_trace_bundle(
        output_dir / "trace_bundle.npz",
        {
            "native_positions": native_positions,
            "broad_positions": broad_positions,
            "native_spring_rest_length": native_spring,
            "broad_spring_rest_length": broad_spring,
            "native_raw_abs_hv": native_plant,
            "broad_raw_abs_hv": broad_plant,
            "native_controller_flat": native_ctrl,
            "broad_controller_flat": broad_ctrl,
            "native_corner_com": native_aug,
            "broad_corner_com": broad_aug,
            "native_front_x_phase": native_front[:phase_count],
            "native_back_x_phase": native_back[:phase_count],
            "broad_front_x_phase": broad_front[:phase_count],
            "broad_back_x_phase": broad_back[:phase_count],
            "env_steps": env_steps,
            "event_phase_index": np.asarray([-1 if event_phase_index is None else int(event_phase_index)], dtype=np.int32),
        },
    )
    np.savez_compressed(
        output_dir / "state_pca_coords.npz",
        native_plant_xy=native_plant_xy,
        broad_plant_xy=broad_plant_xy,
        native_controller_xy=native_ctrl_xy,
        broad_controller_xy=broad_ctrl_xy,
        native_aug_xy=native_aug_xy,
        broad_aug_xy=broad_aug_xy,
        plant_mean=plant_pca["mean"],
        plant_std=plant_pca["std"],
        plant_components=plant_pca["components"],
        controller_mean=ctrl_pca["mean"],
        controller_std=ctrl_pca["std"],
        controller_components=ctrl_pca["components"],
        aug_mean=aug_pca["mean"],
        aug_std=aug_pca["std"],
        aug_components=aug_pca["components"],
    )

    robot_summary = _render_robot_compare(
        output_dir=output_dir / "robot_compare",
        plateau_run={
            "positions": native_positions,
            "spring_rest": native_spring,
            "render_info": native_run["render_info"],
        },
        ledge_run={
            "positions": broad_positions,
            "spring_rest": broad_spring,
            "render_info": broad_run["render_info"],
        },
        phase_count=int(phase_count),
        sense_act_phases=int(sense_act_phases),
        event_phase_index=event_phase_index,
        fps=int(fps),
        max_frames=int(max_frames),
        flash_frames=int(flash_frames),
        supersample=int(supersample),
        top_width_cells=int(native_run["entry_metadata"]["platform_end_x"] - native_run["entry_metadata"]["platform_start_x"] + 1),
        bottom_width_cells=int(broad_run["entry_metadata"]["platform_end_x"] - broad_run["entry_metadata"]["platform_start_x"] + 1),
        top_label=f"Native ({int(goal_distance)},{int(cliff_distance)}) {str(goal_side).title()}",
        bottom_label=f"Compare ({int(compare_goal_distance)},{int(compare_cliff_distance)}) {str(goal_side).title()}",
    )
    pca_summary = _render_state_pca(
        output_dir=output_dir / "state_pca",
        plateau_plant_xy=native_plant_xy,
        ledge_plant_xy=broad_plant_xy,
        plateau_ctrl_xy=native_ctrl_xy,
        ledge_ctrl_xy=broad_ctrl_xy,
        plateau_aug_xy=native_aug_xy,
        ledge_aug_xy=broad_aug_xy,
        aug_title="Plant PCA + COM/corners",
        env_steps=env_steps,
        event_phase_index=event_phase_index,
        fps=int(fps),
        max_frames=int(max_frames),
        flash_frames=int(flash_frames),
    )

    summary = {
        "goal_side": str(goal_side),
        "goal_distance": int(goal_distance),
        "cliff_distance": int(cliff_distance),
        "compare_goal_distance": int(compare_goal_distance),
        "compare_cliff_distance": int(compare_cliff_distance),
        "deformation_condition": str(deformation_condition),
        "signaling_condition": str(signaling_condition),
        "dynamic_impl": str(dynamic_impl),
        "max_steps": int(cfg.train.max_steps),
        "seed": int(seed),
        "native_grid_width": int(native_run["phase_cfg"].terrain.grid_w),
        "native_plateau_width": int(native_run["phase_cfg"].terrain.center_plateau_width),
        "broad_grid_width": int(broad_run["phase_cfg"].terrain.grid_w),
        "broad_plateau_width": int(broad_run["phase_cfg"].terrain.center_plateau_width),
        "phase_count": int(phase_count),
        "sense_act_phases": int(sense_act_phases),
        "nca_steps_per_phase": int(native_run["phase_cfg"].nca.nca_steps_per_phase),
        "native_platform_bounds": [
            int(native_run["entry_metadata"]["platform_start_x"]),
            int(native_run["entry_metadata"]["platform_end_x"]),
        ],
        "broad_platform_bounds": [
            int(broad_run["entry_metadata"]["platform_start_x"]),
            int(broad_run["entry_metadata"]["platform_end_x"]),
        ],
        "initial_centered_robot_max_abs_diff": initial_centered_max_diff,
        "phase0_centered_robot_max_abs_diff": phase0_centered_max_diff,
        "phase0_plant_max_abs_diff": plant_phase0_max_abs_diff,
        "controller_max_abs_diff": controller_max_abs_diff,
        "plant_first_diff_phase": None if plant_first_diff_phase is None else int(plant_first_diff_phase),
        "native_edge_event_phase_index": None if event_phase_index is None else int(event_phase_index),
        "native_edge_event_env_step": None if event_phase_index is None else int(env_steps[int(event_phase_index)]),
        "robot_render": robot_summary,
        "state_pca": pca_summary,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config_snapshot_path.resolve(), max_steps=int(args.max_steps))
    params = load_controller_params(args.controller_params_path.resolve())
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, Any] = {}
    compare_goal_distance = (
        int(args.goal_distance) if args.compare_goal_distance is None else int(args.compare_goal_distance)
    )
    compare_cliff_distance = (
        int(args.cliff_distance)
        if args.compare_cliff_distance is None
        else int(args.compare_cliff_distance)
    )
    for goal_side in [str(side) for side in args.sides]:
        summaries[str(goal_side)] = _run_side(
            cfg=cfg,
            params=params,
            output_dir=output_root / str(goal_side),
            goal_distance=int(args.goal_distance),
            cliff_distance=int(args.cliff_distance),
            compare_goal_distance=int(compare_goal_distance),
            compare_cliff_distance=int(compare_cliff_distance),
            goal_side=str(goal_side),
            deformation_condition=str(args.deformation_condition),
            signaling_condition=str(args.signaling_condition),
            dynamic_impl=str(args.dynamic_impl),
            seed=int(args.seed),
            broad_grid_width=int(args.broad_grid_width),
            broad_plateau_width=int(args.broad_plateau_width),
            fps=int(args.fps),
            max_frames=int(args.max_frames),
            flash_frames=int(args.flash_frames),
            supersample=int(args.supersample),
        )

    write_json(output_root / "summary.json", summaries)
    for goal_side, summary in summaries.items():
        print(f"{goal_side}: {summary['robot_render']['mp4_path']}")
        print(f"{goal_side}: {summary['state_pca']['mp4_path']}")


if __name__ == "__main__":
    main()
