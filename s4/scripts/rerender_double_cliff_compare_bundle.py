from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from free_space_branch_common import DEFAULT_BUNDLE_ROOT, ensure_import_path, fit_pca, load_cfg, transform_pca, write_json
from terrain_width_compare_probe import (
    _phase_env_steps,
    _render_robot_compare,
    _render_state_pca,
    _robot_corner_com_features,
)


RESULT_ROOT = DEFAULT_BUNDLE_ROOT
DEFAULT_INPUT_DIR = (
    RESULT_ROOT / "analysis_runs" / "double_cliff_compare_steps_4000_width2_6_deformation_off_native"
)
DEFAULT_CONFIG_PATH = RESULT_ROOT / "config_snapshot.yaml"
DEFAULT_OUTPUT_DIR = (
    RESULT_ROOT / "analysis_runs" / "double_cliff_compare_steps_4000_width2_6_deformation_off_native_rerender_3d"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerender an existing native double-cliff compare bundle with the current stacked robot and 3D PCA views.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--config-snapshot-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--flash-frames", type=int, default=8)
    parser.add_argument("--supersample", type=int, default=2)
    return parser.parse_args()


def _build_double_cliff_render_bundle(
    *,
    cfg,
    width: int,
    plateau_width: int,
    deformation_condition: str,
    signaling_condition: str,
) -> dict[str, Any]:
    ensure_import_path()
    from sensing_edges.evaluate import _build_phase_assets
    from sensing_edges.post_run import _build_double_cliff_entry, _double_cliff_phase_cfg
    from sensing_edges.runtime_config import resolve_runtime_config

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
        extra_metadata={"analysis_mode": "rerender_double_cliff_compare_bundle"},
    )
    _phase_resolution, envs, _alive_mask, _ = _build_phase_assets(
        phase_cfg,
        runtime_resolution,
        [entry],
    )
    env = envs[0]
    return {
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
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = input_dir / "summary.json"
    trace_bundle_path = input_dir / "trace_bundle.npz"
    summary = json.loads(summary_path.read_text())
    cfg = load_cfg(args.config_snapshot_path.resolve(), max_steps=int(summary["max_steps"]))

    wide_width = int(summary["wide_width"])
    narrow_width = int(summary["narrow_width"])
    plateau_width = int(summary["plateau_width"])
    deformation_condition = str(summary["deformation_condition"])
    signaling_condition = str(summary["signaling_condition"])

    with np.load(trace_bundle_path, allow_pickle=False) as bundle:
        wide_positions = np.asarray(bundle["wide_positions"], dtype=np.float32)
        narrow_positions = np.asarray(bundle["narrow_positions"], dtype=np.float32)
        wide_spring = np.asarray(bundle["wide_spring_rest_length"], dtype=np.float32)
        narrow_spring = np.asarray(bundle["narrow_spring_rest_length"], dtype=np.float32)
        wide_plant = np.asarray(bundle["wide_raw_abs_hv"], dtype=np.float32)
        narrow_plant = np.asarray(bundle["narrow_raw_abs_hv"], dtype=np.float32)
        wide_ctrl = np.asarray(bundle["wide_controller_flat"], dtype=np.float32)
        narrow_ctrl = np.asarray(bundle["narrow_controller_flat"], dtype=np.float32)
        env_steps = np.asarray(bundle["env_steps"], dtype=np.int32)
        raw_event = int(np.asarray(bundle["event_phase_index"], dtype=np.int32).reshape(-1)[0])
        event_phase_index = None if raw_event < 0 else raw_event

    robot_point_mask = np.ones((wide_positions.shape[1],), dtype=bool)
    wide_aug = _robot_corner_com_features(
        positions_seq=wide_positions,
        robot_point_mask=robot_point_mask,
    )
    narrow_aug = _robot_corner_com_features(
        positions_seq=narrow_positions,
        robot_point_mask=robot_point_mask,
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

    wide_render = _build_double_cliff_render_bundle(
        cfg=cfg,
        width=int(wide_width),
        plateau_width=int(plateau_width),
        deformation_condition=str(deformation_condition),
        signaling_condition=str(signaling_condition),
    )
    narrow_render = _build_double_cliff_render_bundle(
        cfg=cfg,
        width=int(narrow_width),
        plateau_width=int(plateau_width),
        deformation_condition=str(deformation_condition),
        signaling_condition=str(signaling_condition),
    )

    phase_count = min(
        int(wide_positions.shape[0]),
        int(narrow_positions.shape[0]),
        int(wide_spring.shape[0]),
        int(narrow_spring.shape[0]),
        int(wide_plant.shape[0]),
        int(narrow_plant.shape[0]),
        int(wide_ctrl.shape[0]),
        int(narrow_ctrl.shape[0]),
        int(wide_aug.shape[0]),
        int(narrow_aug.shape[0]),
        int(env_steps.shape[0]),
    )
    env_steps = np.asarray(env_steps[:phase_count], dtype=np.int32)

    robot_summary = _render_robot_compare(
        output_dir=output_dir / "robot_compare",
        plateau_run={
            "positions": wide_positions[:phase_count],
            "spring_rest": wide_spring[:phase_count],
            "render_info": wide_render["render_info"],
        },
        ledge_run={
            "positions": narrow_positions[:phase_count],
            "spring_rest": narrow_spring[:phase_count],
            "render_info": narrow_render["render_info"],
        },
        phase_count=int(phase_count),
        sense_act_phases=int(summary["sense_act_phases"]),
        event_phase_index=event_phase_index,
        fps=int(args.fps),
        max_frames=int(args.max_frames),
        flash_frames=int(args.flash_frames),
        supersample=int(args.supersample),
        top_width_cells=int(wide_render["entry_metadata"]["platform_end_x"] - wide_render["entry_metadata"]["platform_start_x"] + 1),
        bottom_width_cells=int(narrow_render["entry_metadata"]["platform_end_x"] - narrow_render["entry_metadata"]["platform_start_x"] + 1),
        top_label=f"Long Double Cliff ({wide_width})",
        bottom_label=f"Short Double Cliff ({narrow_width})",
    )
    pca_basic_summary = _render_state_pca(
        output_dir=output_dir / "state_pca_basic",
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
        output_dir=output_dir / "state_pca_extended",
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

    rerender_summary = {
        "source_input_dir": str(input_dir),
        "deformation_condition": str(deformation_condition),
        "signaling_condition": str(signaling_condition),
        "grid_width": int(summary["grid_width"]),
        "plateau_width": int(plateau_width),
        "wide_width": int(wide_width),
        "narrow_width": int(narrow_width),
        "phase_count": int(phase_count),
        "sense_act_phases": int(summary["sense_act_phases"]),
        "nca_steps_per_phase": int(summary["nca_steps_per_phase"]),
        "narrow_event_phase_index": None if event_phase_index is None else int(event_phase_index),
        "narrow_event_env_step": None if event_phase_index is None else int(env_steps[int(event_phase_index)]),
        "initial_centered_robot_max_abs_diff": None,
        "phase0_centered_robot_max_abs_diff": _centered_max_abs_diff(wide_positions[0], narrow_positions[0]),
        "controller_max_abs_diff": float(np.max(np.abs(wide_ctrl[:phase_count] - narrow_ctrl[:phase_count]))),
        "plant_first_diff_phase": _first_diff_phase(wide_plant[:phase_count], narrow_plant[:phase_count]),
        "robot_render": robot_summary,
        "state_pca_basic": pca_basic_summary,
        "state_pca_extended": pca_extended_summary,
    }
    write_json(output_dir / "summary.json", rerender_summary)
    print(robot_summary["mp4_path"])
    print(pca_basic_summary["mp4_path"])
    print(pca_extended_summary["mp4_path"])


if __name__ == "__main__":
    main()
