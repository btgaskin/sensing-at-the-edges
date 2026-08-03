from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from free_space_branch_common import (
    DEFAULT_BUNDLE_ROOT,
    fit_pca,
    load_cfg,
    load_controller_params,
    transform_pca,
    write_json,
)
from right_native_goalblock_doublecliff_probe import (
    _capture_native_variant,
    _centered_max_abs_diff,
    _first_diff_phase,
    _native_edge_event_phase,
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
DEFAULT_OUTPUT_DIR = RESULT_ROOT / "analysis_runs" / "right_native_pair_compare"
DEFAULT_CONFIG_PATH = RESULT_ROOT / "config_snapshot.yaml"
DEFAULT_PARAMS_PATH = RESULT_ROOT / "controller_params.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two right-native bilateral terrains with block-off and block-touchable variants.",
    )
    parser.add_argument("--config-snapshot-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--controller-params-path", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--goal-side", type=str, default="right")
    parser.add_argument("--goal-distance", type=int, default=2)
    parser.add_argument("--cliff-distance", type=int, default=1)
    parser.add_argument("--compare-goal-distance", type=int, default=6)
    parser.add_argument("--compare-cliff-distance", type=int, default=3)
    parser.add_argument("--deformation-condition", type=str, default="deformation_off")
    parser.add_argument("--signaling-condition", type=str, default="intact")
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--flash-frames", type=int, default=8)
    parser.add_argument("--supersample", type=int, default=2)
    return parser.parse_args()


def _compare_runs(
    *,
    run_a: dict[str, Any],
    run_b: dict[str, Any],
    output_dir: Path,
    top_label: str,
    bottom_label: str,
    event_phase_index: int | None,
    event_name: str,
    fps: int,
    max_frames: int,
    flash_frames: int,
    supersample: int,
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
            "event_phase_index": np.asarray(
                [-1 if event_phase_index is None else int(event_phase_index)],
                dtype=np.int32,
            ),
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
        fps=int(fps),
        max_frames=int(max_frames),
        flash_frames=int(flash_frames),
        supersample=int(supersample),
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
        fps=int(fps),
        max_frames=int(max_frames),
        flash_frames=int(flash_frames),
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
        fps=int(fps),
        max_frames=int(max_frames),
        flash_frames=int(flash_frames),
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
    args = parse_args()
    cfg = load_cfg(args.config_snapshot_path.resolve(), max_steps=int(args.max_steps))
    params = load_controller_params(args.controller_params_path.resolve())
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    common_kwargs = {
        "cfg": cfg,
        "params": params,
        "goal_side": str(args.goal_side),
        "deformation_condition": str(args.deformation_condition),
        "signaling_condition": str(args.signaling_condition),
        "seed": int(args.seed),
    }
    base_off = _capture_native_variant(
        goal_distance=int(args.goal_distance),
        cliff_distance=int(args.cliff_distance),
        goal_block_present=False,
        disable_goal_touch=True,
        **common_kwargs,
    )
    base_touchable = _capture_native_variant(
        goal_distance=int(args.goal_distance),
        cliff_distance=int(args.cliff_distance),
        goal_block_present=True,
        disable_goal_touch=False,
        **common_kwargs,
    )
    compare_off = _capture_native_variant(
        goal_distance=int(args.compare_goal_distance),
        cliff_distance=int(args.compare_cliff_distance),
        goal_block_present=False,
        disable_goal_touch=True,
        **common_kwargs,
    )
    compare_touchable = _capture_native_variant(
        goal_distance=int(args.compare_goal_distance),
        cliff_distance=int(args.compare_cliff_distance),
        goal_block_present=True,
        disable_goal_touch=False,
        **common_kwargs,
    )

    base_event_off = _native_edge_event_phase(
        base_off,
        phase_count=int(min(base_off["positions"].shape[0], compare_off["positions"].shape[0])),
    )
    base_event_touchable = _native_edge_event_phase(
        base_touchable,
        phase_count=int(min(base_touchable["positions"].shape[0], compare_touchable["positions"].shape[0])),
    )

    compare_block_off = _compare_runs(
        run_a=base_off,
        run_b=compare_off,
        output_dir=output_root / "block_off_pair_compare",
        top_label=f"Right Native ({int(args.goal_distance)},{int(args.cliff_distance)}) Block Off",
        bottom_label=f"Right Native ({int(args.compare_goal_distance)},{int(args.compare_cliff_distance)}) Block Off",
        event_phase_index=base_event_off,
        event_name="native_edge",
        fps=int(args.fps),
        max_frames=int(args.max_frames),
        flash_frames=int(args.flash_frames),
        supersample=int(args.supersample),
    )
    compare_block_touchable = _compare_runs(
        run_a=base_touchable,
        run_b=compare_touchable,
        output_dir=output_root / "block_touchable_pair_compare",
        top_label=f"Right Native ({int(args.goal_distance)},{int(args.cliff_distance)}) Block Touchable",
        bottom_label=f"Right Native ({int(args.compare_goal_distance)},{int(args.compare_cliff_distance)}) Block Touchable",
        event_phase_index=base_event_touchable,
        event_name="native_edge",
        fps=int(args.fps),
        max_frames=int(args.max_frames),
        flash_frames=int(args.flash_frames),
        supersample=int(args.supersample),
    )

    summary = {
        "deformation_condition": str(args.deformation_condition),
        "signaling_condition": str(args.signaling_condition),
        "goal_side": str(args.goal_side),
        "goal_distance": int(args.goal_distance),
        "cliff_distance": int(args.cliff_distance),
        "compare_goal_distance": int(args.compare_goal_distance),
        "compare_cliff_distance": int(args.compare_cliff_distance),
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "block_off_pair_compare": compare_block_off,
        "block_touchable_pair_compare": compare_block_touchable,
    }
    write_json(output_root / "summary.json", summary)
    print(compare_block_off["robot_render"]["mp4_path"])
    print(compare_block_off["state_pca_basic"]["mp4_path"])
    print(compare_block_off["state_pca_extended"]["mp4_path"])
    print(compare_block_touchable["robot_render"]["mp4_path"])
    print(compare_block_touchable["state_pca_basic"]["mp4_path"])
    print(compare_block_touchable["state_pca_extended"]["mp4_path"])


if __name__ == "__main__":
    main()
