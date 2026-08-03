from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import jax
import numpy as np

from free_space_branch_common import (
    DEFAULT_BUNDLE_ROOT,
    compute_raw_abs_hv_ratios,
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
    _first_edge_phase,
    _phase_env_steps,
    _robot_corner_com_features,
    _render_robot_compare,
    _render_state_pca,
    _save_trace_bundle,
    _trim_pair,
    _valid_event_phase_index,
)


RESULT_ROOT = DEFAULT_BUNDLE_ROOT
DEFAULT_OUTPUT_DIR = (
    RESULT_ROOT / "analysis_runs" / "double_cliff_compare_steps_800_width2_6_deformation_off"
)
DEFAULT_CONFIG_PATH = RESULT_ROOT / "config_snapshot.yaml"
DEFAULT_PARAMS_PATH = RESULT_ROOT / "controller_params.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a native double-cliff width comparison probe with shared PCA/render outputs.",
    )
    parser.add_argument("--config-snapshot-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--controller-params-path", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--narrow-width", type=int, default=2)
    parser.add_argument("--wide-width", type=int, default=6)
    parser.add_argument("--plateau-width", type=int, default=None)
    parser.add_argument("--deformation-condition", type=str, default="deformation_off")
    parser.add_argument("--signaling-condition", type=str, default="intact")
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--flash-frames", type=int, default=8)
    parser.add_argument("--supersample", type=int, default=2)
    return parser.parse_args()


def _capture_double_cliff(
    *,
    cfg,
    params,
    width: int,
    plateau_width: int,
    seed: int,
    deformation_condition: str,
    signaling_condition: str,
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
        extra_metadata={"analysis_mode": "double_cliff_compare_probe"},
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
    positions = np.asarray(phase_arrays["phase_post_positions"], dtype=np.float32)
    spring_rest = np.asarray(phase_arrays["phase_post_spring_rest_length"], dtype=np.float32)
    robot_point_mask = np.asarray(env.robot_point_mask, dtype=bool)
    if not np.any(robot_point_mask):
        raise ValueError("Robot point mask is empty.")
    robot_positions = positions[:, robot_point_mask, :]
    plant_features = compute_raw_abs_hv_ratios(
        positions,
        topology=env.topology,
        spring_init_rest_length=env.sim_state.spring_init_rest_length,
        deformation_info=env.deformation_info,
    )
    return {
        "phase_result": phase_result,
        "phase_arrays": phase_arrays,
        "initial_positions": np.asarray(phase_arrays["initial_positions"], dtype=np.float32),
        "positions": positions,
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


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config_snapshot_path.resolve(), max_steps=int(args.max_steps))
    params = load_controller_params(args.controller_params_path.resolve())
    plateau_width = (
        int(args.plateau_width)
        if args.plateau_width is not None
        else int(cfg.eval.post_run.double_cliff.plateau_width)
    )
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    wide_run = _capture_double_cliff(
        cfg=cfg,
        params=params,
        width=int(args.wide_width),
        plateau_width=int(plateau_width),
        seed=int(args.seed),
        deformation_condition=str(args.deformation_condition),
        signaling_condition=str(args.signaling_condition),
    )
    narrow_run = _capture_double_cliff(
        cfg=cfg,
        params=params,
        width=int(args.narrow_width),
        plateau_width=int(plateau_width),
        seed=int(args.seed),
        deformation_condition=str(args.deformation_condition),
        signaling_condition=str(args.signaling_condition),
    )

    wide_positions, narrow_positions, phase_count = _trim_pair(
        wide_run["positions"],
        narrow_run["positions"],
    )
    wide_spring, narrow_spring, _ = _trim_pair(
        wide_run["spring_rest"],
        narrow_run["spring_rest"],
    )
    wide_plant, narrow_plant, _ = _trim_pair(
        wide_run["plant_features"],
        narrow_run["plant_features"],
    )
    wide_ctrl, narrow_ctrl, _ = _trim_pair(
        wide_run["controller_features"],
        narrow_run["controller_features"],
    )
    wide_front, narrow_front, _ = _trim_pair(
        wide_run["front_x_phase"][:, None],
        narrow_run["front_x_phase"][:, None],
    )
    wide_back, narrow_back, _ = _trim_pair(
        wide_run["back_x_phase"][:, None],
        narrow_run["back_x_phase"][:, None],
    )
    wide_front = wide_front[:, 0]
    wide_back = wide_back[:, 0]
    narrow_front = narrow_front[:, 0]
    narrow_back = narrow_back[:, 0]

    narrow_start = int(narrow_run["entry_metadata"]["platform_start_x"])
    narrow_end = int(narrow_run["entry_metadata"]["platform_end_x"])
    from jax_evogym.constants import CELL_SIZE

    event_phase_index = _first_edge_phase(
        back_x_phase=narrow_back[:phase_count],
        front_x_phase=narrow_front[:phase_count],
        left_frontier_x=float(narrow_start) * float(CELL_SIZE),
        right_frontier_x=float(narrow_end + 1) * float(CELL_SIZE),
    )
    event_phase_index = _valid_event_phase_index(event_phase_index, phase_count=int(phase_count))

    wide_aug = _robot_corner_com_features(
        positions_seq=wide_positions,
        robot_point_mask=np.asarray(wide_run["robot_point_mask"], dtype=bool),
    )
    narrow_aug = _robot_corner_com_features(
        positions_seq=narrow_positions,
        robot_point_mask=np.asarray(narrow_run["robot_point_mask"], dtype=bool),
    )
    plant_pca = fit_pca(wide_plant, narrow_plant, n_components=3)
    ctrl_pca = fit_pca(wide_ctrl, narrow_ctrl, n_components=3)
    aug_pca = fit_pca(wide_aug, narrow_aug, n_components=3)
    wide_plant_xy = transform_pca(wide_plant, plant_pca)
    narrow_plant_xy = transform_pca(narrow_plant, plant_pca)
    wide_ctrl_xy = transform_pca(wide_ctrl, ctrl_pca)
    narrow_ctrl_xy = transform_pca(narrow_ctrl, ctrl_pca)
    wide_aug_xy = transform_pca(wide_aug, aug_pca)
    narrow_aug_xy = transform_pca(narrow_aug, aug_pca)
    sense_act_phases = int(wide_run["phase_cfg"].sim.sense_act_phases_per_env_step)
    env_steps = _phase_env_steps(phase_count, sense_act_phases=sense_act_phases)
    wide_robot_initial = np.asarray(wide_run["initial_positions"], dtype=np.float32)[
        np.asarray(wide_run["robot_point_mask"], dtype=bool)
    ]
    narrow_robot_initial = np.asarray(narrow_run["initial_positions"], dtype=np.float32)[
        np.asarray(narrow_run["robot_point_mask"], dtype=bool)
    ]
    initial_centered_max_diff = _centered_max_abs_diff(wide_robot_initial, narrow_robot_initial)
    phase0_centered_max_diff = _centered_max_abs_diff(
        wide_positions[0][np.asarray(wide_run["robot_point_mask"], dtype=bool)],
        narrow_positions[0][np.asarray(narrow_run["robot_point_mask"], dtype=bool)],
    )
    controller_max_abs_diff = float(np.max(np.abs(wide_ctrl - narrow_ctrl)))
    plant_first_diff_phase = _first_diff_phase(wide_plant, narrow_plant)

    _save_trace_bundle(
        output_root / "trace_bundle.npz",
        {
            "wide_positions": wide_positions,
            "narrow_positions": narrow_positions,
            "wide_spring_rest_length": wide_spring,
            "narrow_spring_rest_length": narrow_spring,
            "wide_raw_abs_hv": wide_plant,
            "narrow_raw_abs_hv": narrow_plant,
            "wide_controller_flat": wide_ctrl,
            "narrow_controller_flat": narrow_ctrl,
            "wide_corner_com": wide_aug,
            "narrow_corner_com": narrow_aug,
            "wide_front_x_phase": wide_front[:phase_count],
            "wide_back_x_phase": wide_back[:phase_count],
            "narrow_front_x_phase": narrow_front[:phase_count],
            "narrow_back_x_phase": narrow_back[:phase_count],
            "env_steps": env_steps,
            "event_phase_index": np.asarray([-1 if event_phase_index is None else int(event_phase_index)], dtype=np.int32),
        },
    )
    np.savez_compressed(
        output_root / "state_pca_coords.npz",
        wide_plant_xy=wide_plant_xy,
        narrow_plant_xy=narrow_plant_xy,
        wide_controller_xy=wide_ctrl_xy,
        narrow_controller_xy=narrow_ctrl_xy,
        wide_aug_xy=wide_aug_xy,
        narrow_aug_xy=narrow_aug_xy,
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
        output_dir=output_root / "robot_compare",
        plateau_run={
            "positions": wide_positions,
            "spring_rest": wide_spring,
            "render_info": wide_run["render_info"],
        },
        ledge_run={
            "positions": narrow_positions,
            "spring_rest": narrow_spring,
            "render_info": narrow_run["render_info"],
        },
        phase_count=int(phase_count),
        sense_act_phases=int(sense_act_phases),
        event_phase_index=event_phase_index,
        fps=int(args.fps),
        max_frames=int(args.max_frames),
        flash_frames=int(args.flash_frames),
        supersample=int(args.supersample),
        top_width_cells=int(wide_run["entry_metadata"]["platform_end_x"] - wide_run["entry_metadata"]["platform_start_x"] + 1),
        bottom_width_cells=int(narrow_run["entry_metadata"]["platform_end_x"] - narrow_run["entry_metadata"]["platform_start_x"] + 1),
        top_label=f"Double Cliff {int(args.wide_width)}",
        bottom_label=f"Double Cliff {int(args.narrow_width)}",
    )
    pca_basic_summary = _render_state_pca(
        output_dir=output_root / "state_pca_basic",
        plateau_plant_xy=wide_plant_xy,
        ledge_plant_xy=narrow_plant_xy,
        plateau_ctrl_xy=wide_ctrl_xy,
        ledge_ctrl_xy=narrow_ctrl_xy,
        env_steps=env_steps,
        event_phase_index=event_phase_index,
        fps=int(args.fps),
        max_frames=int(args.max_frames),
        flash_frames=int(args.flash_frames),
    )
    pca_extended_summary = _render_state_pca(
        output_dir=output_root / "state_pca_extended",
        plateau_plant_xy=wide_plant_xy,
        ledge_plant_xy=narrow_plant_xy,
        plateau_ctrl_xy=wide_ctrl_xy,
        ledge_ctrl_xy=narrow_ctrl_xy,
        plateau_aug_xy=wide_aug_xy,
        ledge_aug_xy=narrow_aug_xy,
        aug_title="Plant PCA + COM/corners",
        env_steps=env_steps,
        event_phase_index=event_phase_index,
        fps=int(args.fps),
        max_frames=int(args.max_frames),
        flash_frames=int(args.flash_frames),
    )

    summary = {
        "deformation_condition": str(args.deformation_condition),
        "signaling_condition": str(args.signaling_condition),
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "grid_width": int(cfg.terrain.grid_w),
        "plateau_width": int(plateau_width),
        "wide_width": int(args.wide_width),
        "narrow_width": int(args.narrow_width),
        "phase_count": int(phase_count),
        "sense_act_phases": int(sense_act_phases),
        "nca_steps_per_phase": int(wide_run["phase_cfg"].nca.nca_steps_per_phase),
        "wide_platform_bounds": [
            int(wide_run["entry_metadata"]["platform_start_x"]),
            int(wide_run["entry_metadata"]["platform_end_x"]),
        ],
        "narrow_platform_bounds": [
            int(narrow_run["entry_metadata"]["platform_start_x"]),
            int(narrow_run["entry_metadata"]["platform_end_x"]),
        ],
        "initial_centered_robot_max_abs_diff": initial_centered_max_diff,
        "phase0_centered_robot_max_abs_diff": phase0_centered_max_diff,
        "controller_max_abs_diff": controller_max_abs_diff,
        "plant_first_diff_phase": plant_first_diff_phase,
        "narrow_event_phase_index": None if event_phase_index is None else int(event_phase_index),
        "narrow_event_env_step": None if event_phase_index is None else int(env_steps[int(event_phase_index)]),
        "robot_render": robot_summary,
        "state_pca_basic": pca_basic_summary,
        "state_pca_extended": pca_extended_summary,
    }
    write_json(output_root / "summary.json", summary)
    print(robot_summary["mp4_path"])
    print(pca_basic_summary["mp4_path"])
    print(pca_extended_summary["mp4_path"])


if __name__ == "__main__":
    main()
