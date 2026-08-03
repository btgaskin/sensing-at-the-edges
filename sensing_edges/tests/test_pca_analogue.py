from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sensing_edges.pca_analogue import (
    GRID_HIDDEN_START,
    GRID_HIDDEN_STOP,
    GOAL_TASK_CODE_THRESHOLD,
    PHASE_UNIQUE_MICRO_INDEX,
    analyze_pca_bundle,
    derive_trial_cliff_edge_labels,
    derive_trial_phase_labels,
    flatten_grid_hidden_features,
    load_pca_bundle,
)


def _trial_rows(
    *,
    trial_id: str,
    side: str,
    goal_direction: int,
    first_peak_direction: str,
    phase_com_x: list[float],
    phase_task_code: list[float],
    selected: bool,
    trial_repeat_index: int,
) -> tuple[list[dict[str, object]], dict[str, object], np.ndarray]:
    assert len(phase_com_x) == len(phase_task_code)
    init_com_x = float(phase_com_x[0])
    label_rows: list[dict[str, object]] = []
    nca_rows: list[np.ndarray] = []
    controller_tick = 0

    for phase_index, (com_x, task_code) in enumerate(zip(phase_com_x, phase_task_code), start=1):
        for micro_index in range(5):
            label_rows.append(
                {
                    "specimen_id": "specimen_alpha",
                    "trial_id": trial_id,
                    "distance_pair": "(4,2)",
                    "side": side,
                    "goal_direction": goal_direction,
                    "env_step": phase_index,
                    "phase_index": phase_index - 1,
                    "micro_index": micro_index,
                    "controller_tick": controller_tick,
                    "com_x": float(com_x + 0.005 * micro_index),
                    "front_x": float(com_x + 0.25),
                    "back_x": float(com_x - 0.25),
                    "task_code_geom": float(task_code),
                    "goal_touched": True,
                    "fell_flag": False,
                    "is_terminal_sample": controller_tick == (len(phase_com_x) * 5 - 1),
                }
            )

            state = np.zeros((2, 7, 12), dtype=np.float32)
            body_pattern = np.add.outer(np.arange(2, dtype=np.float32), np.arange(7, dtype=np.float32))
            for channel_offset, channel_index in enumerate(range(GRID_HIDDEN_START, GRID_HIDDEN_STOP)):
                state[:, :, channel_index] = (
                    (phase_index * 1.5)
                    + (micro_index * 0.1)
                    + (trial_repeat_index * 0.25)
                    + (channel_offset * 0.05)
                    + body_pattern * (0.02 + 0.01 * channel_offset)
                )
            nca_rows.append(state)
            controller_tick += 1

    trial_row = {
        "specimen_id": "specimen_alpha",
        "trial_id": trial_id,
        "distance_pair": "(4,2)",
        "goal_distance": 4,
        "cliff_distance": 2,
        "side": side,
        "goal_direction": goal_direction,
        "trial_repeat_index": trial_repeat_index,
        "init_com_x": init_com_x,
        "init_front_x": init_com_x + 0.25,
        "init_back_x": init_com_x - 0.25,
        "final_com_x": float(phase_com_x[-1]),
        "final_front_x": float(phase_com_x[-1] + 0.25),
        "final_back_x": float(phase_com_x[-1] - 0.25),
        "steps_alive": len(phase_com_x),
        "phase_steps_alive": len(phase_com_x),
        "controller_step_count": len(phase_com_x) * 5,
        "goal_touched": True,
        "fell_flag": False,
        "touched_death_plane": False,
        "went_airborne": False,
        "com_x_went_negative": False,
        "termination_reason": "goal_touched",
        "turn_motion_threshold": 0.01,
        "first_peak_x": float(max(phase_com_x)),
        "first_peak_goalward_x": 0.0,
        "first_peak_direction": first_peak_direction,
        "reached_cliff_frontier": first_peak_direction == "cliff_first",
        "reversed_pre_ledge": False,
        "task_code_geom_peak": float(max(phase_task_code)),
        "selected_for_primary_pca": selected,
    }
    return label_rows, trial_row, np.stack(nca_rows, axis=0)


def _write_tiny_bundle(bundle_dir: Path) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    left_cliff_rows, left_cliff_trial, left_cliff_nca = _trial_rows(
        trial_id="pair_4_2_left_000",
        side="left",
        goal_direction=-1,
        first_peak_direction="cliff_first",
        phase_com_x=[1.0, 1.2, 1.4, 1.3, 1.1, 0.9, 0.6],
        phase_task_code=[0.0, 0.3, 0.7, 1.0, 1.4, 1.8, 2.0],
        selected=True,
        trial_repeat_index=0,
    )
    left_goal_rows, left_goal_trial, left_goal_nca = _trial_rows(
        trial_id="pair_4_2_left_001",
        side="left",
        goal_direction=-1,
        first_peak_direction="goal_first",
        phase_com_x=[1.0, 0.9, 0.6, 0.7, 0.8, 0.95, 1.2],
        phase_task_code=[0.0, 0.2, 0.8, 1.0, 1.4, 1.8, 2.0],
        selected=True,
        trial_repeat_index=1,
    )
    right_rows, right_trial, right_nca = _trial_rows(
        trial_id="pair_4_2_right_000",
        side="right",
        goal_direction=1,
        first_peak_direction="goal_first",
        phase_com_x=[1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0],
        phase_task_code=[0.0, 0.5, 1.0, 1.3, 1.6, 1.9, 2.0],
        selected=False,
        trial_repeat_index=2,
    )

    label_df = pd.DataFrame(left_cliff_rows + left_goal_rows + right_rows)
    trial_df = pd.DataFrame([left_cliff_trial, left_goal_trial, right_trial])
    nca_states = np.concatenate([left_cliff_nca, left_goal_nca, right_nca], axis=0)

    np.savez_compressed(bundle_dir / "nca_states.npz", states=nca_states)
    label_df.to_csv(bundle_dir / "trace_labels.csv", index=False)
    trial_df.to_csv(bundle_dir / "trial_manifest.csv", index=False)
    with open(bundle_dir / "cohort_summary.json", "w") as handle:
        json.dump(
            {
                "schema_version": 1,
                "kind": "sensing_edges_post_run_pca_bundle_cohort",
                "selected_side": "left",
                "selected_trial_ids": ["pair_4_2_left_000", "pair_4_2_left_001"],
            },
            handle,
            indent=2,
        )
    return bundle_dir


def test_load_bundle_and_flatten_grid_hidden_features(tmp_path):
    bundle_dir = _write_tiny_bundle(tmp_path / "bundle")
    bundle = load_pca_bundle(bundle_dir)

    assert bundle.selected_side == "left"
    assert bundle.selected_trial_ids == ["pair_4_2_left_000", "pair_4_2_left_001"]
    assert bundle.nca_states.shape == (70, 2, 7, 12)
    assert bundle.label_df["trial_id"].nunique() == 2

    features = flatten_grid_hidden_features(bundle.nca_states)
    assert features.shape == (70, 84)

    phase_unique_features = flatten_grid_hidden_features(
        bundle.nca_states[bundle.label_df["micro_index"].to_numpy(dtype=np.int32) == PHASE_UNIQUE_MICRO_INDEX]
    )
    assert phase_unique_features.shape == (14, 84)


def test_derive_trial_phase_labels_for_cliff_first_and_goal_first(tmp_path):
    bundle_dir = _write_tiny_bundle(tmp_path / "bundle")
    bundle = load_pca_bundle(bundle_dir)
    phase_unique_df = bundle.label_df.loc[bundle.label_df["micro_index"] == PHASE_UNIQUE_MICRO_INDEX].copy()
    trial_lookup = bundle.trial_df.set_index("trial_id", drop=False)

    cliff_phase_df = phase_unique_df.loc[phase_unique_df["trial_id"] == "pair_4_2_left_000"].copy()
    labeled_cliff = derive_trial_phase_labels(cliff_phase_df, trial_lookup.loc["pair_4_2_left_000"])
    assert labeled_cliff["phase_label"].tolist() == [
        "probe",
        "turnaround",
        "turnaround",
        "turnaround",
        "return",
        "return",
        "dwell",
    ]
    assert int(labeled_cliff["turn_idx"].iloc[0]) == 2
    assert int(labeled_cliff["goal_idx"].iloc[0]) == 6
    assert float(labeled_cliff["task_code_geom"].iloc[-1]) >= GOAL_TASK_CODE_THRESHOLD
    cliff_edge_labeled = derive_trial_cliff_edge_labels(labeled_cliff, trial_lookup.loc["pair_4_2_left_000"])
    assert cliff_edge_labeled["cliff_edge_label"].tolist() == [
        "probe",
        "near_cliff",
        "over_frontier",
        "return",
        "return",
        "return",
        "return",
    ]

    goal_phase_df = phase_unique_df.loc[phase_unique_df["trial_id"] == "pair_4_2_left_001"].copy()
    labeled_goal = derive_trial_phase_labels(goal_phase_df, trial_lookup.loc["pair_4_2_left_001"])
    assert labeled_goal["phase_label"].tolist() == [
        "probe",
        "turnaround",
        "turnaround",
        "turnaround",
        "return",
        "return",
        "dwell",
    ]
    assert int(labeled_goal["turn_idx"].iloc[0]) == 2


def test_analyze_bundle_and_cli_smoke(tmp_path, monkeypatch):
    bundle_dir = _write_tiny_bundle(tmp_path / "bundle")
    output_dir = tmp_path / "analysis"

    summary = analyze_pca_bundle(
        bundle_dir,
        output_dir,
        max_plot_points=1000,
        max_trajectory_points_per_trial=200,
        max_animation_points=60,
    )
    assert summary["selected_trial_count"] == 2
    assert summary["feature_count"] == 84
    assert summary["pca_fit_basis"] == "phase_unique_samples"
    assert summary["selected_identity"]["deterministic_replicas"] is False
    assert summary["selected_identity"]["unique_trajectory_count"] == 2
    assert summary["comparison_identity"]["deterministic_replicas"] is False
    assert summary["comparison_identity"]["unique_trajectory_count"] == 1

    summary_path = output_dir / "summary.json"
    components_path = output_dir / "components.npz"
    task_plot = output_dir / "figures" / "pc1_pc2_task_code_geom.png"
    phase_plot = output_dir / "figures" / "pc1_pc2_phase_label.png"
    exemplar_plot = output_dir / "figures" / "exemplar_trial_alignment.png"
    loading_plot = output_dir / "figures" / "pc_loading_maps.png"
    overlay_3d = output_dir / "figures" / "pc1_pc2_pc3_phase_overlay.png"
    exemplar_3d_phase = output_dir / "figures" / "exemplar_trial_pc1_pc2_pc3_phase.png"
    exemplar_3d_stage = output_dir / "figures" / "exemplar_trial_pc1_pc2_pc3_stage.png"
    exemplar_3d_probe_return = output_dir / "figures" / "exemplar_trial_pc1_pc2_pc3_probe_return.png"
    exemplar_3d_time = output_dir / "figures" / "exemplar_trial_pc1_pc2_pc3_time.png"
    exemplar_all_steps_3d_phase = output_dir / "figures" / "exemplar_trial_all_steps_pc1_pc2_pc3_phase.png"
    exemplar_all_steps_3d_stage = output_dir / "figures" / "exemplar_trial_all_steps_pc1_pc2_pc3_stage.png"
    exemplar_all_steps_3d_probe_return = output_dir / "figures" / "exemplar_trial_all_steps_pc1_pc2_pc3_probe_return.png"
    exemplar_all_steps_3d_cliff_edge = output_dir / "figures" / "exemplar_trial_all_steps_pc1_pc2_pc3_cliff_edge.png"
    exemplar_all_steps_3d_transition = output_dir / "figures" / "exemplar_trial_all_steps_pc1_pc2_pc3_transition.png"
    exemplar_all_steps_3d_time = output_dir / "figures" / "exemplar_trial_all_steps_pc1_pc2_pc3_time.png"
    exemplar_all_steps_animation = output_dir / "figures" / "exemplar_trial_all_steps_pc1_pc2_pc3_probe_return.gif"
    exemplar_all_steps_cliff_edge_animation = output_dir / "figures" / "exemplar_trial_all_steps_pc1_pc2_pc3_cliff_edge.gif"
    exemplar_all_steps_transition_animation = output_dir / "figures" / "exemplar_trial_all_steps_pc1_pc2_pc3_transition.gif"
    reversal_window_pca_plot = output_dir / "figures" / "exemplar_trial_reversal_window_pca_cliff_edge.png"
    reversal_window_pca_gif = output_dir / "figures" / "exemplar_trial_reversal_window_pca_cliff_edge.gif"
    stage_augmented_pca_plot = output_dir / "figures" / "exemplar_trial_all_steps_stage_augmented_pca_cliff_edge.png"
    stage_augmented_pca_gif = output_dir / "figures" / "exemplar_trial_all_steps_stage_augmented_pca_cliff_edge.gif"
    dpca_plot = output_dir / "figures" / "exemplar_trial_all_steps_dpca_cliff_edge.png"
    dpca_gif = output_dir / "figures" / "exemplar_trial_all_steps_dpca_cliff_edge.gif"
    compare_3d = output_dir / "figures" / "selected_vs_mirrored_pc1_pc2_pc3.png"
    mirrored_probe_return = output_dir / "figures" / "mirrored_exemplar_all_steps_pc1_pc2_pc3_probe_return.png"
    mirrored_time = output_dir / "figures" / "mirrored_exemplar_all_steps_pc1_pc2_pc3_time.png"
    for path in [
        summary_path,
        components_path,
        task_plot,
        phase_plot,
        exemplar_plot,
        loading_plot,
        overlay_3d,
        exemplar_3d_phase,
        exemplar_3d_stage,
        exemplar_3d_probe_return,
        exemplar_3d_time,
        exemplar_all_steps_3d_phase,
        exemplar_all_steps_3d_stage,
        exemplar_all_steps_3d_probe_return,
        exemplar_all_steps_3d_cliff_edge,
        exemplar_all_steps_3d_transition,
        exemplar_all_steps_3d_time,
        exemplar_all_steps_animation,
        exemplar_all_steps_cliff_edge_animation,
        exemplar_all_steps_transition_animation,
        reversal_window_pca_plot,
        reversal_window_pca_gif,
        stage_augmented_pca_plot,
        stage_augmented_pca_gif,
        dpca_plot,
        dpca_gif,
        compare_3d,
        mirrored_probe_return,
        mirrored_time,
    ]:
        assert path.exists()

    payload = np.load(components_path)
    assert payload["components"].shape == (3, 84)
    assert payload["phase_unique_projection"].shape[1] == 3
    assert payload["reversal_window_pca_projection"].shape[1] == 3
    assert payload["stage_augmented_pca_projection"].shape[1] == 3
    assert payload["dpca_projection"].shape[1] == 3

    script_path = Path(__file__).parent.parent / "scripts" / "make_pca_analogue.py"
    spec = importlib.util.spec_from_file_location("sensing_edges_make_pca_analogue", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    cli_output_dir = tmp_path / "cli-analysis"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_pca_analogue.py",
            "--bundle-dir",
            str(bundle_dir),
            "--output-dir",
            str(cli_output_dir),
            "--max-plot-points",
            "1000",
            "--max-trajectory-points-per-trial",
            "200",
            "--max-animation-points",
            "60",
        ],
    )
    module.main()

    assert (cli_output_dir / "summary.json").exists()


def test_make_pca_analogue_cli_resolves_named_cohort_from_assay_root(tmp_path, monkeypatch):
    assay_dir = tmp_path / "pca_bundle"
    bundle_dir = _write_tiny_bundle(assay_dir / "cohorts" / "deformation_off_native_core")

    script_path = Path(__file__).parent.parent / "scripts" / "make_pca_analogue.py"
    spec = importlib.util.spec_from_file_location("sensing_edges_make_pca_analogue", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    cli_output_dir = tmp_path / "cli-analysis-cohort"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_pca_analogue.py",
            "--bundle-dir",
            str(assay_dir),
            "--cohort-id",
            "deformation_off_native_core",
            "--output-dir",
            str(cli_output_dir),
            "--max-plot-points",
            "1000",
            "--max-trajectory-points-per-trial",
            "200",
            "--max-animation-points",
            "60",
        ],
    )
    module.main()

    assert bundle_dir.exists()
    assert (cli_output_dir / "summary.json").exists()
