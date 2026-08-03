"""Offline PCA analogue for post-run sensing_edges trace bundles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from jax_evogym.constants import CELL_SIZE

GRID_HIDDEN_START = 6
GRID_HIDDEN_STOP = 12
GRID_HIDDEN_CHANNELS = GRID_HIDDEN_STOP - GRID_HIDDEN_START
PHASE_UNIQUE_MICRO_INDEX = 0
GOAL_TASK_CODE_THRESHOLD = 1.95
DEFAULT_MAX_TRAJECTORY_POINTS_PER_TRIAL = 2000
DEFAULT_MAX_ANIMATION_POINTS = 240
DEFAULT_STAGE_RESAMPLE_BINS = 60
CLIFF_VICINITY_TASK_THRESHOLD = 0.85
REVERSAL_RANGE_FRACTION = 0.05
PHASE_LABELS = ("probe", "turnaround", "return", "dwell")
PHASE_COLORS = {
    "probe": "#33658a",
    "turnaround": "#f26419",
    "return": "#2f9c95",
    "dwell": "#758e4f",
}
STAGE_LABELS = ("approach", "cliff_vicinity", "reversal", "goal_return", "goal_dwell")
STAGE_COLORS = {
    "approach": "#355070",
    "cliff_vicinity": "#6d597a",
    "reversal": "#e56b6f",
    "goal_return": "#2a9d8f",
    "goal_dwell": "#8ab17d",
}
PROBE_RETURN_LABELS = ("probe", "return")
PROBE_RETURN_COLORS = {
    "probe": "#5b2a86",
    "return": "#1b998b",
}
CLIFF_EDGE_LABELS = ("probe", "near_cliff", "over_frontier", "return")
CLIFF_EDGE_COLORS = {
    "probe": "#355070",
    "near_cliff": "#e56b6f",
    "over_frontier": "#f4a261",
    "return": "#2a9d8f",
}
TRANSITION_LABELS = ("approach", "proximity_in", "probe", "proximity_out", "return")
TRANSITION_COLORS = {
    "approach": "#355070",
    "proximity_in": "#b56576",
    "probe": "#f4a261",
    "proximity_out": "#e56b6f",
    "return": "#2a9d8f",
}
CLIFF_NEAR_BAND_CELLS = 2.0
CLIFF_NEAR_BAND_WIDTH = CLIFF_NEAR_BAND_CELLS * float(CELL_SIZE)


@dataclass(frozen=True)
class LoadedPcaBundle:
    bundle_dir: Path
    selected_side: str
    selected_trial_ids: list[str]
    trial_df: pd.DataFrame
    label_df: pd.DataFrame
    nca_states: np.ndarray
    all_trial_df: pd.DataFrame
    all_label_df: pd.DataFrame
    all_nca_states: np.ndarray


def _read_json(path: str | Path) -> dict[str, Any]:
    with open(Path(path)) as handle:
        return json.load(handle)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as handle:
        json.dump(payload, handle, indent=2)


def _require_columns(frame: pd.DataFrame, required: set[str], *, label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _trial_order_map(trial_ids: list[str]) -> dict[str, int]:
    return {trial_id: index for index, trial_id in enumerate(trial_ids)}


def _sorted_selected_indices(label_df: pd.DataFrame, selected_trial_ids: list[str]) -> np.ndarray:
    selected_positions = np.flatnonzero(label_df["trial_id"].isin(selected_trial_ids).to_numpy(dtype=bool))
    selected_rows = label_df.iloc[selected_positions].copy()
    trial_order = _trial_order_map(selected_trial_ids)
    selected_rows["_trial_order"] = selected_rows["trial_id"].map(trial_order)
    selected_rows["_source_index"] = selected_positions
    selected_rows = selected_rows.sort_values(
        by=["_trial_order", "env_step", "phase_index", "micro_index", "controller_tick"],
        kind="stable",
    )
    return selected_rows["_source_index"].to_numpy(dtype=np.int64)


def _sorted_indices_for_trial_ids(label_df: pd.DataFrame, trial_ids: list[str]) -> np.ndarray:
    trial_order = _trial_order_map(trial_ids)
    positions = np.flatnonzero(label_df["trial_id"].isin(trial_ids).to_numpy(dtype=bool))
    subset = label_df.iloc[positions].copy()
    subset["_trial_order"] = subset["trial_id"].map(trial_order)
    subset["_source_index"] = positions
    subset = subset.sort_values(
        by=["_trial_order", "env_step", "phase_index", "micro_index", "controller_tick"],
        kind="stable",
    )
    return subset["_source_index"].to_numpy(dtype=np.int64)


def _select_trial_subset(
    *,
    trial_df: pd.DataFrame,
    label_df: pd.DataFrame,
    nca_states: np.ndarray,
    trial_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    if not trial_ids:
        return trial_df.iloc[0:0].copy(), label_df.iloc[0:0].copy(), nca_states[:0]
    subset_indices = _sorted_indices_for_trial_ids(label_df, trial_ids)
    subset_label_df = label_df.iloc[subset_indices].copy().reset_index(drop=True)
    subset_nca_states = nca_states[subset_indices]
    subset_trial_df = trial_df.loc[trial_df["trial_id"].isin(trial_ids)].copy()
    subset_trial_df["_trial_order"] = subset_trial_df["trial_id"].map(_trial_order_map(trial_ids))
    subset_trial_df = (
        subset_trial_df.sort_values(by="_trial_order", kind="stable")
        .drop(columns=["_trial_order"])
        .reset_index(drop=True)
    )
    return subset_trial_df, subset_label_df, subset_nca_states


def _trajectory_identity_report(
    trial_ids: list[str],
    label_df: pd.DataFrame,
    nca_states: np.ndarray,
) -> dict[str, Any]:
    if not trial_ids:
        return {
            "trial_count": 0,
            "samples_per_trial": [],
            "unique_trajectory_count": 0,
            "deterministic_replicas": False,
            "representative_trial_id": None,
        }

    digests: dict[str, str] = {}
    samples_per_trial: list[int] = []
    for trial_id in trial_ids:
        idx = label_df.index[label_df["trial_id"] == trial_id].to_numpy()
        block = np.ascontiguousarray(nca_states[idx])
        samples_per_trial.append(int(block.shape[0]))
        digests[str(trial_id)] = hashlib.sha1(block.view(np.uint8)).hexdigest()

    unique_digest_count = len(set(digests.values()))
    return {
        "trial_count": len(trial_ids),
        "samples_per_trial": samples_per_trial,
        "unique_trajectory_count": int(unique_digest_count),
        "deterministic_replicas": bool(unique_digest_count == 1 and len(trial_ids) > 1),
        "representative_trial_id": str(trial_ids[0]),
        "trial_ids": [str(trial_id) for trial_id in trial_ids],
    }


def load_pca_bundle(bundle_dir: str | Path) -> LoadedPcaBundle:
    bundle_path = Path(bundle_dir)
    cohort_summary = _read_json(bundle_path / "cohort_summary.json")
    selected_trial_ids = [str(item) for item in cohort_summary.get("selected_trial_ids", [])]
    if not selected_trial_ids:
        raise ValueError(f"No selected_trial_ids found in {bundle_path / 'cohort_summary.json'}")

    label_df = pd.read_csv(bundle_path / "trace_labels.csv")
    trial_df = pd.read_csv(bundle_path / "trial_manifest.csv")
    _require_columns(
        label_df,
        {
            "trial_id",
            "env_step",
            "phase_index",
            "micro_index",
            "controller_tick",
            "com_x",
            "task_code_geom",
        },
        label="trace_labels.csv",
    )
    _require_columns(
        trial_df,
        {
            "trial_id",
            "side",
            "goal_direction",
            "init_com_x",
            "first_peak_direction",
            "controller_step_count",
        },
        label="trial_manifest.csv",
    )

    nca_payload = np.load(bundle_path / "nca_states.npz")
    nca_states = np.asarray(nca_payload["states"], dtype=np.float32)
    if len(label_df) != int(nca_states.shape[0]):
        raise ValueError(
            "trace_labels.csv row count does not match nca_states.npz sample count: "
            f"{len(label_df)} vs {int(nca_states.shape[0])}"
        )

    selected_indices = _sorted_selected_indices(label_df, selected_trial_ids)
    if selected_indices.size <= 0:
        raise ValueError("Selected trials are not present in trace_labels.csv")

    selected_label_df = label_df.iloc[selected_indices].copy().reset_index(drop=True)
    selected_nca_states = nca_states[selected_indices]
    selected_trial_df = trial_df.loc[trial_df["trial_id"].isin(selected_trial_ids)].copy()
    if len(selected_trial_df) != len(selected_trial_ids):
        found = set(selected_trial_df["trial_id"].astype(str))
        missing = [trial_id for trial_id in selected_trial_ids if trial_id not in found]
        raise ValueError(f"Selected trials missing from trial_manifest.csv: {missing}")

    selected_trial_df["_trial_order"] = selected_trial_df["trial_id"].map(_trial_order_map(selected_trial_ids))
    selected_trial_df = (
        selected_trial_df.sort_values(by="_trial_order", kind="stable")
        .drop(columns=["_trial_order"])
        .reset_index(drop=True)
    )

    expected_selected_samples = int(selected_trial_df["controller_step_count"].sum())
    if expected_selected_samples != len(selected_label_df):
        raise ValueError(
            "Selected sample count mismatch between trial_manifest.csv and trace_labels.csv: "
            f"{expected_selected_samples} vs {len(selected_label_df)}"
        )

    return LoadedPcaBundle(
        bundle_dir=bundle_path,
        selected_side=str(cohort_summary["selected_side"]),
        selected_trial_ids=selected_trial_ids,
        trial_df=selected_trial_df,
        label_df=selected_label_df,
        nca_states=selected_nca_states,
        all_trial_df=trial_df.copy(),
        all_label_df=label_df.copy(),
        all_nca_states=nca_states,
    )


def flatten_grid_hidden_features(nca_states: np.ndarray) -> np.ndarray:
    if nca_states.ndim != 4:
        raise ValueError(f"Expected nca_states to have 4 dimensions, got shape {nca_states.shape}")
    grid_hidden = np.asarray(nca_states[..., GRID_HIDDEN_START:GRID_HIDDEN_STOP], dtype=np.float32)
    return grid_hidden.reshape(grid_hidden.shape[0], -1)


def _phase_unique_mask(label_df: pd.DataFrame) -> np.ndarray:
    return label_df["micro_index"].to_numpy(dtype=np.int32) == PHASE_UNIQUE_MICRO_INDEX


def derive_trial_phase_labels(phase_trial_df: pd.DataFrame, trial_row: pd.Series) -> pd.DataFrame:
    if phase_trial_df.empty:
        raise ValueError(f"Cannot derive phase labels for empty trial {trial_row['trial_id']}")

    phase_frame = phase_trial_df.copy().reset_index(drop=True)
    goal_direction = int(trial_row["goal_direction"])
    init_com_x = float(trial_row["init_com_x"])
    goalward_x = goal_direction * (phase_frame["com_x"].to_numpy(dtype=np.float32) - init_com_x)
    first_peak_direction = str(trial_row["first_peak_direction"])

    if first_peak_direction == "cliff_first":
        turn_idx = int(np.argmin(goalward_x))
    elif first_peak_direction == "goal_first":
        turn_idx = int(np.argmax(goalward_x))
    else:
        turn_idx = 0

    turn_start = max(0, turn_idx - 1)
    turn_end = min(len(phase_frame) - 1, turn_idx + 1)
    labels = np.full(len(phase_frame), "probe", dtype=object)
    labels[turn_start : turn_end + 1] = "turnaround"

    task_code = phase_frame["task_code_geom"].to_numpy(dtype=np.float32)
    goal_hits = np.flatnonzero(task_code >= GOAL_TASK_CODE_THRESHOLD)
    goal_idx = int(goal_hits[0]) if goal_hits.size else None

    if goal_idx is not None:
        if turn_end + 1 <= goal_idx - 1:
            labels[turn_end + 1 : goal_idx] = "return"
        labels[goal_idx:] = "dwell"
    elif first_peak_direction != "none" and turn_end + 1 < len(labels):
        labels[turn_end + 1 :] = "return"

    phase_frame["goalward_x"] = goalward_x.astype(np.float32)
    phase_frame["phase_label"] = labels
    phase_frame["turn_idx"] = int(turn_idx)
    phase_frame["goal_idx"] = goal_idx if goal_idx is not None else -1
    phase_frame["phase_tick"] = np.arange(len(phase_frame), dtype=np.int32)
    return phase_frame


def derive_trial_stage_labels(phase_trial_df: pd.DataFrame, trial_row: pd.Series) -> pd.DataFrame:
    phase_frame = phase_trial_df.copy().reset_index(drop=True)
    task_code = phase_frame["task_code_geom"].to_numpy(dtype=np.float32)
    goalward_x = phase_frame["goalward_x"].to_numpy(dtype=np.float32)
    stage = np.full(len(phase_frame), "approach", dtype=object)

    goal_hits = np.flatnonzero(task_code >= GOAL_TASK_CODE_THRESHOLD)
    goal_idx = int(goal_hits[0]) if goal_hits.size else None
    first_peak_direction = str(trial_row["first_peak_direction"])

    if first_peak_direction == "cliff_first":
        vicinity_hits = np.flatnonzero(task_code >= CLIFF_VICINITY_TASK_THRESHOLD)
        if vicinity_hits.size:
            vicinity_start = int(vicinity_hits[0])
            stage[vicinity_start:] = "cliff_vicinity"
        else:
            vicinity_start = None

        movement_range = max(float(np.max(goalward_x) - np.min(goalward_x)), 1e-6)
        reversal_threshold = float(np.min(goalward_x)) + REVERSAL_RANGE_FRACTION * movement_range
        reversal_hits = np.flatnonzero(goalward_x <= reversal_threshold)
        if reversal_hits.size:
            reversal_start = int(reversal_hits[0])
            reversal_end = int(reversal_hits[-1])
            if vicinity_start is not None:
                reversal_start = max(reversal_start, vicinity_start)
            stage[reversal_start : reversal_end + 1] = "reversal"
            if reversal_end + 1 < len(stage):
                stage[reversal_end + 1 :] = "goal_return"
        elif vicinity_start is not None:
            stage[vicinity_start:] = "goal_return"
    else:
        phase_label = phase_frame["phase_label"].to_numpy(dtype=object)
        stage[phase_label == "turnaround"] = "reversal"
        stage[phase_label == "return"] = "goal_return"

    if goal_idx is not None:
        stage[goal_idx:] = "goal_dwell"

    phase_frame["stage_label"] = stage
    probe_return = np.where(
        np.isin(stage, ("approach", "cliff_vicinity", "reversal")),
        "probe",
        "return",
    )
    phase_frame["probe_return_label"] = probe_return
    return phase_frame


def derive_trial_cliff_edge_labels(phase_trial_df: pd.DataFrame, trial_row: pd.Series) -> pd.DataFrame:
    phase_frame = phase_trial_df.copy().reset_index(drop=True)
    goal_direction = int(trial_row["goal_direction"])
    cliff_distance = int(trial_row["cliff_distance"])
    first_peak_direction = str(trial_row["first_peak_direction"])

    if goal_direction < 0:
        cliff_facing_edge = phase_frame["front_x"].to_numpy(dtype=np.float32)
        init_cliff_edge_x = float(trial_row["init_front_x"])
    else:
        cliff_facing_edge = phase_frame["back_x"].to_numpy(dtype=np.float32)
        init_cliff_edge_x = float(trial_row["init_back_x"])

    cliff_frontier_x = float(init_cliff_edge_x - (goal_direction * cliff_distance * float(CELL_SIZE)))
    cliff_clearance_x = goal_direction * (cliff_facing_edge - cliff_frontier_x)
    outward_progress_x = np.maximum(
        -goal_direction * (cliff_facing_edge - float(init_cliff_edge_x)),
        0.0,
    ).astype(np.float32)
    peak_idx = int(np.argmax(outward_progress_x)) if len(outward_progress_x) > 0 else 0
    labels = np.full(len(phase_frame), "return", dtype=object)

    if first_peak_direction == "cliff_first":
        for idx in range(len(phase_frame)):
            if idx > peak_idx:
                labels[idx] = "return"
            elif float(cliff_clearance_x[idx]) < 0.0:
                labels[idx] = "over_frontier"
            elif float(cliff_clearance_x[idx]) <= CLIFF_NEAR_BAND_WIDTH:
                labels[idx] = "near_cliff"
            else:
                labels[idx] = "probe"

    phase_frame["cliff_facing_edge_x"] = cliff_facing_edge.astype(np.float32)
    phase_frame["cliff_frontier_x"] = np.full(len(phase_frame), cliff_frontier_x, dtype=np.float32)
    phase_frame["cliff_clearance_x"] = cliff_clearance_x.astype(np.float32)
    phase_frame["cliff_outward_progress_x"] = outward_progress_x
    phase_frame["cliff_peak_idx"] = int(peak_idx)
    phase_frame["cliff_edge_label"] = labels
    return phase_frame


def derive_trial_transition_labels(phase_trial_df: pd.DataFrame) -> pd.DataFrame:
    phase_frame = phase_trial_df.copy().reset_index(drop=True)
    clearance = phase_frame["cliff_clearance_x"].to_numpy(dtype=np.float32)
    over_mask = clearance < 0.0
    near_mask = (clearance >= 0.0) & (clearance <= float(CLIFF_NEAR_BAND_WIDTH))
    labels = np.full(len(phase_frame), "return", dtype=object)

    over_hits = np.flatnonzero(over_mask)
    if over_hits.size > 0:
        first_over = int(over_hits[0])
        last_over = int(over_hits[-1])
        for idx in range(len(phase_frame)):
            if idx < first_over:
                labels[idx] = "proximity_in" if near_mask[idx] else "approach"
            elif idx <= last_over:
                labels[idx] = "probe"
            else:
                labels[idx] = "proximity_out" if near_mask[idx] else "return"
    else:
        peak_idx = int(phase_frame["cliff_peak_idx"].iloc[0])
        for idx in range(len(phase_frame)):
            if idx <= peak_idx:
                labels[idx] = "proximity_in" if near_mask[idx] else "approach"
            else:
                labels[idx] = "proximity_out" if near_mask[idx] else "return"

    phase_frame["transition_label"] = labels
    return phase_frame


def _annotate_selected_labels(bundle: LoadedPcaBundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase_unique_df = bundle.label_df.loc[_phase_unique_mask(bundle.label_df)].copy().reset_index(drop=True)
    phase_rows: list[pd.DataFrame] = []
    full_rows: list[pd.DataFrame] = []
    trial_lookup = bundle.trial_df.set_index("trial_id", drop=False)
    required_broadcast_cols = [
        "trial_id",
        "env_step",
        "phase_index",
        "phase_label",
        "stage_label",
        "probe_return_label",
        "cliff_facing_edge_x",
        "cliff_frontier_x",
        "cliff_clearance_x",
        "cliff_outward_progress_x",
        "cliff_peak_idx",
        "cliff_edge_label",
        "transition_label",
    ]

    for trial_id in bundle.selected_trial_ids:
        trial_row = trial_lookup.loc[trial_id]
        phase_trial_df = phase_unique_df.loc[phase_unique_df["trial_id"] == trial_id].copy()
        labeled_phase_trial = derive_trial_phase_labels(phase_trial_df, trial_row)
        labeled_phase_trial = derive_trial_stage_labels(labeled_phase_trial, trial_row)
        labeled_phase_trial = derive_trial_cliff_edge_labels(labeled_phase_trial, trial_row)
        labeled_phase_trial = derive_trial_transition_labels(labeled_phase_trial)
        phase_rows.append(labeled_phase_trial)

        full_trial_df = bundle.label_df.loc[bundle.label_df["trial_id"] == trial_id].copy()
        full_trial_df["goalward_x"] = (
            int(trial_row["goal_direction"])
            * (full_trial_df["com_x"].to_numpy(dtype=np.float32) - float(trial_row["init_com_x"]))
        )
        phase_map = labeled_phase_trial[required_broadcast_cols].copy()
        full_trial_df = full_trial_df.merge(
            phase_map,
            on=["trial_id", "env_step", "phase_index"],
            how="left",
            suffixes=("", "_phase"),
        )
        if full_trial_df["phase_label"].isna().any():
            raise ValueError(f"Failed to broadcast phase labels for trial {trial_id}")
        full_rows.append(full_trial_df)

    labeled_phase_df = pd.concat(phase_rows, ignore_index=True)
    labeled_full_df = pd.concat(full_rows, ignore_index=True)
    return labeled_phase_df, labeled_full_df


def compute_pca_basis(features: np.ndarray, *, n_components: int = 3) -> dict[str, np.ndarray]:
    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {features.shape}")
    if features.shape[0] < 2:
        raise ValueError("Need at least two samples to fit PCA")

    centered = np.asarray(features, dtype=np.float64)
    mean = centered.mean(axis=0, keepdims=True)
    centered = centered - mean
    covariance = centered.T @ centered / float(centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total_variance = float(np.sum(eigenvalues))
    if total_variance <= 0.0:
        raise ValueError("PCA failed because the selected features have zero total variance")

    max_components = min(n_components, eigenvectors.shape[1])
    components = eigenvectors[:, :max_components].T.astype(np.float32)
    explained_variance = eigenvalues[:max_components].astype(np.float32)
    explained_variance_ratio = (eigenvalues[:max_components] / total_variance).astype(np.float32)
    return {
        "mean": mean.astype(np.float32).reshape(-1),
        "components": components,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
    }


def project_features(features: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    centered = np.asarray(features, dtype=np.float32) - np.asarray(mean, dtype=np.float32)[None, :]
    return centered @ np.asarray(components, dtype=np.float32).T


def _phase_plot_stride(n_points: int, *, max_plot_points: int) -> int:
    return max(1, int(math.ceil(float(n_points) / float(max_plot_points))))


def _cell_and_channel_loading_magnitudes(components: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reshaped = np.asarray(components, dtype=np.float32).reshape(components.shape[0], 2, 7, GRID_HIDDEN_CHANNELS)
    cell_magnitude = np.linalg.norm(reshaped, axis=-1)
    channel_magnitude = np.linalg.norm(reshaped, axis=(1, 2))
    return cell_magnitude.astype(np.float32), channel_magnitude.astype(np.float32)


def _trajectory_stride(n_points: int, *, max_points: int) -> int:
    return max(1, int(math.ceil(float(n_points) / float(max_points))))


def _downsample_trajectory(
    coords: np.ndarray,
    phase_labels: np.ndarray,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    stride = _trajectory_stride(len(coords), max_points=max_points)
    downsampled_coords = coords[::stride]
    downsampled_labels = phase_labels[::stride]
    if len(downsampled_coords) < len(coords):
        downsampled_coords = np.concatenate([downsampled_coords, coords[-1:]], axis=0)
        downsampled_labels = np.concatenate([downsampled_labels, phase_labels[-1:]], axis=0)
    return downsampled_coords, downsampled_labels


def _one_hot_labels(labels: np.ndarray, label_order: tuple[str, ...]) -> np.ndarray:
    label_array = np.asarray(labels, dtype=object)
    encoded = np.zeros((label_array.shape[0], len(label_order)), dtype=np.float32)
    for idx, label in enumerate(label_order):
        encoded[:, idx] = (label_array == label).astype(np.float32)
    return encoded


def _augment_features_with_labels(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    label_order: tuple[str, ...],
) -> tuple[np.ndarray, float]:
    base = np.asarray(features, dtype=np.float32)
    label_one_hot = _one_hot_labels(labels, label_order)
    feature_std = np.std(base, axis=0, dtype=np.float64)
    scale = float(np.mean(feature_std)) if feature_std.size else 1.0
    scale = max(scale, 1e-3)
    augmented = np.concatenate([base, label_one_hot * scale], axis=1)
    return augmented.astype(np.float32), float(scale)


def _reversal_window_mask(label_df: pd.DataFrame) -> np.ndarray:
    labels = label_df["cliff_edge_label"].to_numpy(dtype=object)
    clearance = label_df["cliff_clearance_x"].to_numpy(dtype=np.float32)
    if labels.size <= 0:
        raise ValueError("Cannot build a reversal window for an empty label frame")

    active_hits = np.flatnonzero(labels != "probe")
    if active_hits.size <= 0:
        mask = np.zeros(labels.shape[0], dtype=bool)
        mask[:] = True
        return mask

    start_idx = int(active_hits[0])
    retreat_hits = np.flatnonzero(
        (np.arange(labels.shape[0]) >= start_idx)
        & (labels == "return")
        & (clearance > float(CLIFF_NEAR_BAND_WIDTH))
    )
    end_idx = int(retreat_hits[0]) if retreat_hits.size > 0 else int(labels.shape[0] - 1)
    mask = np.zeros(labels.shape[0], dtype=bool)
    mask[start_idx : end_idx + 1] = True
    return mask


def _resample_feature_rows(features: np.ndarray, *, target_len: int) -> np.ndarray:
    rows = np.asarray(features, dtype=np.float32)
    if rows.ndim != 2:
        raise ValueError(f"Expected 2D feature rows, got shape {rows.shape}")
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if rows.shape[0] <= 0:
        raise ValueError("Cannot resample an empty feature block")
    if rows.shape[0] == 1:
        return np.repeat(rows, target_len, axis=0)

    source_x = np.linspace(0.0, 1.0, rows.shape[0], dtype=np.float64)
    target_x = np.linspace(0.0, 1.0, target_len, dtype=np.float64)
    resampled = np.empty((target_len, rows.shape[1]), dtype=np.float32)
    for feature_idx in range(rows.shape[1]):
        resampled[:, feature_idx] = np.interp(target_x, source_x, rows[:, feature_idx]).astype(np.float32)
    return resampled


def _build_stage_tensor(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    stage_order: tuple[str, ...],
    bins_per_stage: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_rows = np.asarray(features, dtype=np.float32)
    label_array = np.asarray(labels, dtype=object)
    stage_slices: list[np.ndarray] = []
    stage_lengths = np.zeros(len(stage_order), dtype=np.int32)
    for stage_idx, stage_label in enumerate(stage_order):
        stage_rows = feature_rows[label_array == stage_label]
        if stage_rows.shape[0] <= 0:
            raise ValueError(f"Stage {stage_label!r} has no samples for stage tensor construction")
        stage_lengths[stage_idx] = int(stage_rows.shape[0])
        stage_slices.append(_resample_feature_rows(stage_rows, target_len=bins_per_stage))

    tensor = np.stack(stage_slices, axis=1)  # (time, stage, feature)
    tensor = np.transpose(tensor, (2, 1, 0))  # (feature, stage, time)
    return tensor.astype(np.float32), stage_lengths


def compute_plain_dpca(
    stage_tensor: np.ndarray,
) -> dict[str, np.ndarray | float]:
    tensor = np.asarray(stage_tensor, dtype=np.float64)
    if tensor.ndim != 3:
        raise ValueError(f"Expected stage_tensor with 3 dims, got shape {tensor.shape}")
    n_features, n_stages, n_time = tensor.shape
    if n_features <= 0 or n_stages <= 0 or n_time <= 0:
        raise ValueError("stage_tensor must have positive dimensions")

    grand = tensor.mean(axis=(1, 2), keepdims=True)
    stage_marg = tensor.mean(axis=2, keepdims=True) - grand
    time_marg = tensor.mean(axis=1, keepdims=True) - grand
    interaction_marg = tensor - stage_marg - time_marg - grand

    def _top_component(matrix: np.ndarray) -> tuple[np.ndarray, float]:
        flat = matrix.reshape(n_features, -1)
        covariance = flat @ flat.T / max(flat.shape[1] - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        component = eigenvectors[:, order[0]].astype(np.float32)
        return component, float(eigenvalues[0])

    time_component, time_var = _top_component(time_marg)
    stage_component, stage_var = _top_component(stage_marg)
    interaction_component, interaction_var = _top_component(interaction_marg)
    component_matrix = np.stack([time_component, stage_component, interaction_component], axis=0).astype(np.float32)

    flat_tensor = tensor.reshape(n_features, -1).T.astype(np.float32)
    projected = flat_tensor @ component_matrix.T
    total_var = max(float(np.var(flat_tensor, axis=0, ddof=1).sum()), 1e-8)
    explained = np.asarray([time_var, stage_var, interaction_var], dtype=np.float32)
    explained_ratio = (explained / total_var).astype(np.float32)
    return {
        "components": component_matrix,
        "projection": projected.astype(np.float32),
        "explained_variance": explained,
        "explained_variance_ratio": explained_ratio,
    }


def _set_3d_axis_limits(ax: Any, coords: np.ndarray) -> None:
    mins = np.min(coords, axis=0)
    maxs = np.max(coords, axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    half_width = 0.55 * span if span > 0.0 else 1.0
    ax.set_xlim(float(center[0] - half_width), float(center[0] + half_width))
    ax.set_ylim(float(center[1] - half_width), float(center[1] + half_width))
    ax.set_zlim(float(center[2] - half_width), float(center[2] + half_width))


def _set_3d_axis_limits_raw(ax: Any, coords: np.ndarray) -> None:
    ax.set_xlim(float(np.min(coords[:, 0])), float(np.max(coords[:, 0])))
    ax.set_ylim(float(np.min(coords[:, 1])), float(np.max(coords[:, 1])))
    ax.set_zlim(float(np.min(coords[:, 2])), float(np.max(coords[:, 2])))


def _label_legend_handles(label_order: tuple[str, ...], color_map: dict[str, str]) -> list[Line2D]:
    return [
        Line2D([0], [0], color=color_map[label], lw=2, label=label)
        for label in label_order
    ]


def _plot_task_code_projection(
    projection: np.ndarray,
    task_code: np.ndarray,
    *,
    output_path: Path,
    title: str,
    plot_stride: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    coords = projection[::plot_stride]
    colors = task_code[::plot_stride]
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors, cmap="viridis", s=3, alpha=0.35, linewidths=0.0)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="task_code_geom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_phase_projection(
    projection: np.ndarray,
    phase_labels: np.ndarray,
    *,
    output_path: Path,
    title: str,
    plot_stride: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    coords = projection[::plot_stride]
    labels = phase_labels[::plot_stride]
    for phase_label in PHASE_LABELS:
        mask = labels == phase_label
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=4,
            alpha=0.45,
            linewidths=0.0,
            color=PHASE_COLORS[phase_label],
            label=phase_label,
        )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_exemplar_alignment(
    phase_trial_df: pd.DataFrame,
    phase_trial_projection: np.ndarray,
    *,
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax_left = axes[0]
    ax_left.plot(phase_trial_projection[:, 0], phase_trial_projection[:, 1], color="#777777", linewidth=1.25, alpha=0.8)
    for phase_label in PHASE_LABELS:
        mask = phase_trial_df["phase_label"] == phase_label
        if not np.any(mask):
            continue
        ax_left.scatter(
            phase_trial_projection[mask, 0],
            phase_trial_projection[mask, 1],
            color=PHASE_COLORS[phase_label],
            s=32,
            label=phase_label,
        )
    ax_left.scatter(phase_trial_projection[0, 0], phase_trial_projection[0, 1], color="#111111", s=40, marker="o")
    ax_left.scatter(phase_trial_projection[-1, 0], phase_trial_projection[-1, 1], color="#111111", s=40, marker="s")
    ax_left.set_xlabel("PC1")
    ax_left.set_ylabel("PC2")
    ax_left.set_title(f"{title}: trajectory")
    ax_left.legend(frameon=False)

    ax_right = axes[1]
    phase_ticks = phase_trial_df["phase_tick"].to_numpy(dtype=np.int32)
    ax_right.plot(phase_ticks, phase_trial_df["goalward_x"].to_numpy(dtype=np.float32), color="#111111", linewidth=1.5)
    ax_right.scatter(
        phase_ticks,
        phase_trial_df["goalward_x"].to_numpy(dtype=np.float32),
        c=[PHASE_COLORS[label] for label in phase_trial_df["phase_label"]],
        s=24,
    )
    ax_right.set_xlabel("Phase Tick")
    ax_right.set_ylabel("goalward_x")
    ax_right.set_title(f"{title}: movement alignment")

    ax_right_task = ax_right.twinx()
    ax_right_task.plot(
        phase_ticks,
        phase_trial_df["task_code_geom"].to_numpy(dtype=np.float32),
        color="#8a8a8a",
        linestyle="--",
        linewidth=1.2,
    )
    ax_right_task.set_ylabel("task_code_geom")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_trajectory_overlay_3d(
    labeled_phase_df: pd.DataFrame,
    phase_unique_projection: np.ndarray,
    *,
    output_path: Path,
    title: str,
    max_points_per_trial: int,
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    all_coords: list[np.ndarray] = []

    for trial_id, trial_df in labeled_phase_df.groupby("trial_id", sort=False):
        del trial_id
        trial_mask = labeled_phase_df["trial_id"].to_numpy(dtype=object) == trial_df["trial_id"].iloc[0]
        coords = phase_unique_projection[trial_mask]
        labels = trial_df["phase_label"].to_numpy(dtype=object)
        coords, labels = _downsample_trajectory(coords, labels, max_points=max_points_per_trial)
        if len(coords) < 2:
            continue
        all_coords.append(coords)
        segments = np.stack([coords[:-1], coords[1:]], axis=1)
        segment_colors = [PHASE_COLORS[str(label)] for label in labels[:-1]]
        line_collection = Line3DCollection(segments, colors=segment_colors, linewidths=1.0, alpha=0.28)
        ax.add_collection3d(line_collection)

    if all_coords:
        merged = np.concatenate(all_coords, axis=0)
        ax.set_xlim(float(np.min(merged[:, 0])), float(np.max(merged[:, 0])))
        ax.set_ylim(float(np.min(merged[:, 1])), float(np.max(merged[:, 1])))
        ax.set_zlim(float(np.min(merged[:, 2])), float(np.max(merged[:, 2])))
    ax.view_init(elev=28, azim=42)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(handles=_label_legend_handles(PHASE_LABELS, PHASE_COLORS), frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_exemplar_trajectory_3d_time(
    phase_trial_projection: np.ndarray,
    *,
    output_path: Path,
    title: str,
    max_points: int,
) -> None:
    coords, _labels = _downsample_trajectory(
        phase_trial_projection,
        np.full(len(phase_trial_projection), "return", dtype=object),
        max_points=max_points,
    )
    if len(coords) < 2:
        raise ValueError("Need at least two points to plot a 3D trajectory")

    time_fraction = np.linspace(0.0, 1.0, len(coords), dtype=np.float32)
    segments = np.stack([coords[:-1], coords[1:]], axis=1)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    line_collection = Line3DCollection(
        segments,
        cmap="viridis",
        linewidths=1.6,
        alpha=0.95,
    )
    line_collection.set_array(time_fraction[:-1])
    ax.add_collection3d(line_collection)
    ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color="#111111", s=28, marker="o")
    ax.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], color="#111111", s=34, marker="s")
    ax.set_xlim(float(np.min(coords[:, 0])), float(np.max(coords[:, 0])))
    ax.set_ylim(float(np.min(coords[:, 1])), float(np.max(coords[:, 1])))
    ax.set_zlim(float(np.min(coords[:, 2])), float(np.max(coords[:, 2])))
    ax.view_init(elev=28, azim=42)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    colorbar = fig.colorbar(line_collection, ax=ax, shrink=0.65, pad=0.08)
    colorbar.set_label("normalized time")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_exemplar_trajectory_3d_phase(
    phase_trial_df: pd.DataFrame,
    phase_trial_projection: np.ndarray,
    *,
    output_path: Path,
    title: str,
    max_points: int,
) -> None:
    coords, labels = _downsample_trajectory(
        phase_trial_projection,
        phase_trial_df["phase_label"].to_numpy(dtype=object),
        max_points=max_points,
    )
    if len(coords) < 2:
        raise ValueError("Need at least two points to plot a 3D trajectory")

    segments = np.stack([coords[:-1], coords[1:]], axis=1)
    segment_colors = [PHASE_COLORS[str(label)] for label in labels[:-1]]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    line_collection = Line3DCollection(segments, colors=segment_colors, linewidths=1.6, alpha=0.95)
    ax.add_collection3d(line_collection)
    ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color="#111111", s=28, marker="o")
    ax.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], color="#111111", s=34, marker="s")
    ax.set_xlim(float(np.min(coords[:, 0])), float(np.max(coords[:, 0])))
    ax.set_ylim(float(np.min(coords[:, 1])), float(np.max(coords[:, 1])))
    ax.set_zlim(float(np.min(coords[:, 2])), float(np.max(coords[:, 2])))
    ax.view_init(elev=28, azim=42)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(handles=_label_legend_handles(PHASE_LABELS, PHASE_COLORS), frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_exemplar_trajectory_3d_stage(
    phase_trial_df: pd.DataFrame,
    phase_trial_projection: np.ndarray,
    *,
    output_path: Path,
    title: str,
    max_points: int,
) -> None:
    coords, labels = _downsample_trajectory(
        phase_trial_projection,
        phase_trial_df["stage_label"].to_numpy(dtype=object),
        max_points=max_points,
    )
    if len(coords) < 2:
        raise ValueError("Need at least two points to plot a 3D trajectory")

    segments = np.stack([coords[:-1], coords[1:]], axis=1)
    segment_colors = [STAGE_COLORS[str(label)] for label in labels[:-1]]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    line_collection = Line3DCollection(segments, colors=segment_colors, linewidths=1.0, alpha=0.95)
    ax.add_collection3d(line_collection)
    ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color="#111111", s=28, marker="o")
    ax.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], color="#111111", s=34, marker="s")
    ax.set_xlim(float(np.min(coords[:, 0])), float(np.max(coords[:, 0])))
    ax.set_ylim(float(np.min(coords[:, 1])), float(np.max(coords[:, 1])))
    ax.set_zlim(float(np.min(coords[:, 2])), float(np.max(coords[:, 2])))
    ax.view_init(elev=28, azim=42)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(handles=_label_legend_handles(STAGE_LABELS, STAGE_COLORS), frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_exemplar_trajectory_3d_probe_return(
    phase_trial_df: pd.DataFrame,
    phase_trial_projection: np.ndarray,
    *,
    output_path: Path,
    title: str,
    max_points: int,
) -> None:
    coords, labels = _downsample_trajectory(
        phase_trial_projection,
        phase_trial_df["probe_return_label"].to_numpy(dtype=object),
        max_points=max_points,
    )
    if len(coords) < 2:
        raise ValueError("Need at least two points to plot a 3D trajectory")

    segments = np.stack([coords[:-1], coords[1:]], axis=1)
    segment_colors = [PROBE_RETURN_COLORS[str(label)] for label in labels[:-1]]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    line_collection = Line3DCollection(segments, colors=segment_colors, linewidths=0.9, alpha=0.95)
    ax.add_collection3d(line_collection)
    ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color="#111111", s=24, marker="o")
    ax.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], color="#111111", s=30, marker="s")
    _set_3d_axis_limits(ax, coords)
    ax.view_init(elev=28, azim=42)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(handles=_label_legend_handles(PROBE_RETURN_LABELS, PROBE_RETURN_COLORS), frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_exemplar_trajectory_3d_cliff_edge(
    phase_trial_df: pd.DataFrame,
    phase_trial_projection: np.ndarray,
    *,
    output_path: Path,
    title: str,
    max_points: int,
) -> None:
    coords, labels = _downsample_trajectory(
        phase_trial_projection,
        phase_trial_df["cliff_edge_label"].to_numpy(dtype=object),
        max_points=max_points,
    )
    if len(coords) < 2:
        raise ValueError("Need at least two points to plot a 3D trajectory")

    segments = np.stack([coords[:-1], coords[1:]], axis=1)
    segment_colors = [CLIFF_EDGE_COLORS[str(label)] for label in labels[:-1]]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    line_collection = Line3DCollection(segments, colors=segment_colors, linewidths=0.95, alpha=0.95)
    ax.add_collection3d(line_collection)
    ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color="#111111", s=24, marker="o")
    ax.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], color="#111111", s=30, marker="s")
    _set_3d_axis_limits(ax, coords)
    ax.view_init(elev=28, azim=42)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(handles=_label_legend_handles(CLIFF_EDGE_LABELS, CLIFF_EDGE_COLORS), frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_labeled_trajectory_3d(
    labels: np.ndarray,
    projection: np.ndarray,
    *,
    output_path: Path,
    title: str,
    max_points: int,
    label_order: tuple[str, ...],
    color_map: dict[str, str],
    axis_labels: tuple[str, str, str],
    equalized_axes: bool = True,
) -> None:
    coords, downsampled_labels = _downsample_trajectory(
        projection,
        np.asarray(labels, dtype=object),
        max_points=max_points,
    )
    if len(coords) < 2:
        raise ValueError("Need at least two points to plot a 3D trajectory")

    segments = np.stack([coords[:-1], coords[1:]], axis=1)
    segment_colors = [color_map[str(label)] for label in downsampled_labels[:-1]]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    line_collection = Line3DCollection(segments, colors=segment_colors, linewidths=0.95, alpha=0.95)
    ax.add_collection3d(line_collection)
    ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color="#111111", s=24, marker="o")
    ax.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], color="#111111", s=30, marker="s")
    if equalized_axes:
        _set_3d_axis_limits(ax, coords)
    else:
        _set_3d_axis_limits_raw(ax, coords)
    ax.view_init(elev=28, azim=42)
    ax.set_title(title)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.legend(handles=_label_legend_handles(label_order, color_map), frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _animate_exemplar_trajectory_3d_probe_return(
    phase_trial_df: pd.DataFrame,
    phase_trial_projection: np.ndarray,
    *,
    output_path: Path,
    title: str,
    max_points: int,
    fps: int = 20,
) -> None:
    coords, labels = _downsample_trajectory(
        phase_trial_projection,
        phase_trial_df["probe_return_label"].to_numpy(dtype=object),
        max_points=max_points,
    )
    if len(coords) < 2:
        raise ValueError("Need at least two points to animate a 3D trajectory")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    _set_3d_axis_limits(ax, coords)
    ax.view_init(elev=28, azim=42)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(handles=_label_legend_handles(PROBE_RETURN_LABELS, PROBE_RETURN_COLORS), frameon=False, loc="upper right")

    background_segments = np.stack([coords[:-1], coords[1:]], axis=1)
    background_colors = [PROBE_RETURN_COLORS[str(label)] for label in labels[:-1]]
    background = Line3DCollection(background_segments, colors=background_colors, linewidths=0.8, alpha=0.12)
    ax.add_collection3d(background)

    progress_line, = ax.plot([], [], [], color="#2f2f2f", linewidth=1.1, alpha=0.35)
    tracer_line, = ax.plot([], [], [], color=PROBE_RETURN_COLORS["probe"], linewidth=2.2, alpha=0.95)
    head_marker, = ax.plot([], [], [], linestyle="", marker="o", markersize=6, color=PROBE_RETURN_COLORS["probe"])
    status_text = ax.text2D(0.03, 0.95, "", transform=ax.transAxes)
    tracer_length = max(12, min(72, len(coords) // 10))

    def _update(frame_idx: int) -> tuple[Any, ...]:
        end = frame_idx + 1
        history = coords[:end]
        progress_line.set_data(history[:, 0], history[:, 1])
        progress_line.set_3d_properties(history[:, 2])

        start = max(0, end - tracer_length)
        tracer = coords[start:end]
        current_label = str(labels[end - 1])
        tracer_color = PROBE_RETURN_COLORS[current_label]
        tracer_line.set_color(tracer_color)
        tracer_line.set_data(tracer[:, 0], tracer[:, 1])
        tracer_line.set_3d_properties(tracer[:, 2])

        head = coords[end - 1]
        head_marker.set_color(tracer_color)
        head_marker.set_data([head[0]], [head[1]])
        head_marker.set_3d_properties([head[2]])
        status_text.set_text(f"{current_label}  {end}/{len(coords)}")
        return progress_line, tracer_line, head_marker, status_text

    animation = FuncAnimation(fig, _update, frames=len(coords), interval=1000 / float(fps), blit=False)
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=120)
    plt.close(fig)


def _animate_labeled_trajectory_3d(
    labels: np.ndarray,
    projection: np.ndarray,
    *,
    output_path: Path,
    title: str,
    max_points: int,
    label_order: tuple[str, ...],
    color_map: dict[str, str],
    axis_labels: tuple[str, str, str],
    equalized_axes: bool = True,
    fps: int = 20,
) -> None:
    coords, downsampled_labels = _downsample_trajectory(
        projection,
        np.asarray(labels, dtype=object),
        max_points=max_points,
    )
    if len(coords) < 2:
        raise ValueError("Need at least two points to animate a 3D trajectory")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    if equalized_axes:
        _set_3d_axis_limits(ax, coords)
    else:
        _set_3d_axis_limits_raw(ax, coords)
    ax.view_init(elev=28, azim=42)
    ax.set_title(title)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.legend(handles=_label_legend_handles(label_order, color_map), frameon=False, loc="upper right")

    background_segments = np.stack([coords[:-1], coords[1:]], axis=1)
    background_colors = [color_map[str(label)] for label in downsampled_labels[:-1]]
    background = Line3DCollection(background_segments, colors=background_colors, linewidths=0.8, alpha=0.12)
    ax.add_collection3d(background)

    progress_line, = ax.plot([], [], [], color="#2f2f2f", linewidth=1.1, alpha=0.35)
    tracer_line, = ax.plot([], [], [], color=color_map[label_order[0]], linewidth=2.2, alpha=0.95)
    head_marker, = ax.plot([], [], [], linestyle="", marker="o", markersize=6, color=color_map[label_order[0]])
    status_text = ax.text2D(0.03, 0.95, "", transform=ax.transAxes)
    tracer_length = max(12, min(72, len(coords) // 10))

    def _update(frame_idx: int) -> tuple[Any, ...]:
        end = frame_idx + 1
        history = coords[:end]
        progress_line.set_data(history[:, 0], history[:, 1])
        progress_line.set_3d_properties(history[:, 2])

        start = max(0, end - tracer_length)
        tracer = coords[start:end]
        current_label = str(downsampled_labels[end - 1])
        tracer_color = color_map[current_label]
        tracer_line.set_color(tracer_color)
        tracer_line.set_data(tracer[:, 0], tracer[:, 1])
        tracer_line.set_3d_properties(tracer[:, 2])

        head = coords[end - 1]
        head_marker.set_color(tracer_color)
        head_marker.set_data([head[0]], [head[1]])
        head_marker.set_3d_properties([head[2]])
        status_text.set_text(f"{current_label}  {end}/{len(coords)}")
        return progress_line, tracer_line, head_marker, status_text

    animation = FuncAnimation(fig, _update, frames=len(coords), interval=1000 / float(fps), blit=False)
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=120)
    plt.close(fig)


def _animate_exemplar_trajectory_3d_cliff_edge(
    phase_trial_df: pd.DataFrame,
    phase_trial_projection: np.ndarray,
    *,
    output_path: Path,
    title: str,
    max_points: int,
    fps: int = 20,
) -> None:
    coords, labels = _downsample_trajectory(
        phase_trial_projection,
        phase_trial_df["cliff_edge_label"].to_numpy(dtype=object),
        max_points=max_points,
    )
    if len(coords) < 2:
        raise ValueError("Need at least two points to animate a 3D trajectory")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    _set_3d_axis_limits(ax, coords)
    ax.view_init(elev=28, azim=42)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(handles=_label_legend_handles(CLIFF_EDGE_LABELS, CLIFF_EDGE_COLORS), frameon=False, loc="upper right")

    background_segments = np.stack([coords[:-1], coords[1:]], axis=1)
    background_colors = [CLIFF_EDGE_COLORS[str(label)] for label in labels[:-1]]
    background = Line3DCollection(background_segments, colors=background_colors, linewidths=0.8, alpha=0.12)
    ax.add_collection3d(background)

    progress_line, = ax.plot([], [], [], color="#2f2f2f", linewidth=1.1, alpha=0.35)
    tracer_line, = ax.plot([], [], [], color=CLIFF_EDGE_COLORS["probe"], linewidth=2.2, alpha=0.95)
    head_marker, = ax.plot([], [], [], linestyle="", marker="o", markersize=6, color=CLIFF_EDGE_COLORS["probe"])
    status_text = ax.text2D(0.03, 0.95, "", transform=ax.transAxes)
    tracer_length = max(12, min(72, len(coords) // 10))

    def _update(frame_idx: int) -> tuple[Any, ...]:
        end = frame_idx + 1
        history = coords[:end]
        progress_line.set_data(history[:, 0], history[:, 1])
        progress_line.set_3d_properties(history[:, 2])

        start = max(0, end - tracer_length)
        tracer = coords[start:end]
        current_label = str(labels[end - 1])
        tracer_color = CLIFF_EDGE_COLORS[current_label]
        tracer_line.set_color(tracer_color)
        tracer_line.set_data(tracer[:, 0], tracer[:, 1])
        tracer_line.set_3d_properties(tracer[:, 2])

        head = coords[end - 1]
        head_marker.set_color(tracer_color)
        head_marker.set_data([head[0]], [head[1]])
        head_marker.set_3d_properties([head[2]])
        status_text.set_text(f"{current_label}  {end}/{len(coords)}")
        return progress_line, tracer_line, head_marker, status_text

    animation = FuncAnimation(fig, _update, frames=len(coords), interval=1000 / float(fps), blit=False)
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=120)
    plt.close(fig)


def _plot_condition_comparison_3d(
    selected_projection: np.ndarray,
    comparison_projection: np.ndarray,
    *,
    output_path: Path,
    title: str,
    selected_label: str,
    comparison_label: str,
    max_points: int,
) -> None:
    selected_coords, _ = _downsample_trajectory(
        selected_projection,
        np.full(len(selected_projection), selected_label, dtype=object),
        max_points=max_points,
    )
    comparison_coords, _ = _downsample_trajectory(
        comparison_projection,
        np.full(len(comparison_projection), comparison_label, dtype=object),
        max_points=max_points,
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        selected_coords[:, 0],
        selected_coords[:, 1],
        selected_coords[:, 2],
        color="#33658a",
        linewidth=2.0,
        alpha=0.95,
        label=selected_label,
    )
    ax.plot(
        comparison_coords[:, 0],
        comparison_coords[:, 1],
        comparison_coords[:, 2],
        color="#b4413c",
        linewidth=2.0,
        alpha=0.95,
        label=comparison_label,
    )
    ax.scatter(selected_coords[0, 0], selected_coords[0, 1], selected_coords[0, 2], color="#33658a", s=26, marker="o")
    ax.scatter(selected_coords[-1, 0], selected_coords[-1, 1], selected_coords[-1, 2], color="#33658a", s=32, marker="s")
    ax.scatter(comparison_coords[0, 0], comparison_coords[0, 1], comparison_coords[0, 2], color="#b4413c", s=26, marker="o")
    ax.scatter(comparison_coords[-1, 0], comparison_coords[-1, 1], comparison_coords[-1, 2], color="#b4413c", s=32, marker="s")

    merged = np.concatenate([selected_coords, comparison_coords], axis=0)
    ax.set_xlim(float(np.min(merged[:, 0])), float(np.max(merged[:, 0])))
    ax.set_ylim(float(np.min(merged[:, 1])), float(np.max(merged[:, 1])))
    ax.set_zlim(float(np.min(merged[:, 2])), float(np.max(merged[:, 2])))
    ax.view_init(elev=28, azim=42)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_loading_maps(
    components: np.ndarray,
    *,
    output_path: Path,
    title: str,
) -> tuple[np.ndarray, np.ndarray]:
    cell_magnitude, channel_magnitude = _cell_and_channel_loading_magnitudes(components)
    fig, axes = plt.subplots(components.shape[0], 2, figsize=(10, 3.4 * components.shape[0]))
    if components.shape[0] == 1:
        axes = np.asarray([axes])

    channel_positions = np.arange(GRID_HIDDEN_CHANNELS, dtype=np.int32)
    for row_index in range(components.shape[0]):
        heat_ax = axes[row_index, 0]
        bar_ax = axes[row_index, 1]

        heat = heat_ax.imshow(cell_magnitude[row_index], cmap="magma", aspect="auto")
        heat_ax.set_title(f"PC{row_index + 1} cell magnitude")
        heat_ax.set_xlabel("x")
        heat_ax.set_ylabel("y")
        heat_ax.set_xticks(np.arange(7))
        heat_ax.set_yticks(np.arange(2))
        fig.colorbar(heat, ax=heat_ax, fraction=0.046, pad=0.04)

        bar_ax.bar(channel_positions, channel_magnitude[row_index], color="#33658a")
        bar_ax.set_title(f"PC{row_index + 1} channel magnitude")
        bar_ax.set_xlabel("grid_hidden channel")
        bar_ax.set_ylabel("L2 magnitude")
        bar_ax.set_xticks(channel_positions)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return cell_magnitude, channel_magnitude


def analyze_pca_bundle(
    bundle_dir: str | Path,
    output_dir: str | Path,
    *,
    n_components: int = 3,
    max_plot_points: int = 200_000,
    max_trajectory_points_per_trial: int = DEFAULT_MAX_TRAJECTORY_POINTS_PER_TRIAL,
    max_animation_points: int = DEFAULT_MAX_ANIMATION_POINTS,
) -> dict[str, Any]:
    bundle = load_pca_bundle(bundle_dir)
    labeled_phase_df, labeled_full_df = _annotate_selected_labels(bundle)
    selected_identity = _trajectory_identity_report(bundle.selected_trial_ids, bundle.label_df, bundle.nca_states)
    comparison_side = "right" if bundle.selected_side == "left" else "left"
    comparison_trial_ids = [
        str(trial_id)
        for trial_id in bundle.all_trial_df.loc[bundle.all_trial_df["side"] == comparison_side, "trial_id"].tolist()
    ]
    comparison_trial_df, comparison_label_df, comparison_nca_states = _select_trial_subset(
        trial_df=bundle.all_trial_df,
        label_df=bundle.all_label_df,
        nca_states=bundle.all_nca_states,
        trial_ids=comparison_trial_ids,
    )
    comparison_identity = _trajectory_identity_report(comparison_trial_ids, comparison_label_df, comparison_nca_states)

    phase_unique_features = flatten_grid_hidden_features(
        bundle.nca_states[_phase_unique_mask(bundle.label_df)]
    )
    all_features = flatten_grid_hidden_features(bundle.nca_states)

    pca = compute_pca_basis(phase_unique_features, n_components=n_components)
    phase_unique_projection = project_features(phase_unique_features, pca["mean"], pca["components"])
    all_projection = project_features(all_features, pca["mean"], pca["components"])
    comparison_phase_df = pd.DataFrame()
    comparison_full_df = pd.DataFrame()
    comparison_all_projection = np.zeros((0, pca["components"].shape[0]), dtype=np.float32)
    comparison_phase_projection = np.zeros((0, pca["components"].shape[0]), dtype=np.float32)
    if len(comparison_trial_ids) > 0:
        comparison_phase_df, comparison_full_df = _annotate_selected_labels(
            LoadedPcaBundle(
                bundle_dir=bundle.bundle_dir,
                selected_side=comparison_side,
                selected_trial_ids=comparison_trial_ids,
                trial_df=comparison_trial_df,
                label_df=comparison_label_df,
                nca_states=comparison_nca_states,
                all_trial_df=bundle.all_trial_df,
                all_label_df=bundle.all_label_df,
                all_nca_states=bundle.all_nca_states,
            )
        )
        comparison_phase_features = flatten_grid_hidden_features(
            comparison_nca_states[_phase_unique_mask(comparison_label_df)]
        )
        comparison_all_features = flatten_grid_hidden_features(comparison_nca_states)
        comparison_phase_projection = project_features(comparison_phase_features, pca["mean"], pca["components"])
        comparison_all_projection = project_features(comparison_all_features, pca["mean"], pca["components"])

    output_path = Path(output_dir)
    figures_dir = output_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_stride = _phase_plot_stride(len(all_projection), max_plot_points=max_plot_points)
    _plot_task_code_projection(
        all_projection,
        labeled_full_df["task_code_geom"].to_numpy(dtype=np.float32),
        output_path=figures_dir / "pc1_pc2_task_code_geom.png",
        title="Selected Cohort PCA Projection Colored by task_code_geom",
        plot_stride=plot_stride,
    )
    _plot_phase_projection(
        all_projection,
        labeled_full_df["phase_label"].to_numpy(dtype=object),
        output_path=figures_dir / "pc1_pc2_phase_label.png",
        title="Selected Cohort PCA Projection Colored by Movement Phase",
        plot_stride=plot_stride,
    )

    exemplar_trial_id = bundle.selected_trial_ids[0]
    exemplar_phase_df = labeled_phase_df.loc[labeled_phase_df["trial_id"] == exemplar_trial_id].reset_index(drop=True)
    exemplar_projection = phase_unique_projection[
        labeled_phase_df["trial_id"].to_numpy(dtype=object) == exemplar_trial_id
    ]
    exemplar_full_df = labeled_full_df.loc[labeled_full_df["trial_id"] == exemplar_trial_id].reset_index(drop=True)
    exemplar_full_projection = all_projection[
        labeled_full_df["trial_id"].to_numpy(dtype=object) == exemplar_trial_id
    ]
    exemplar_full_features = all_features[
        labeled_full_df["trial_id"].to_numpy(dtype=object) == exemplar_trial_id
    ]
    _plot_exemplar_alignment(
        exemplar_phase_df,
        exemplar_projection,
        output_path=figures_dir / "exemplar_trial_alignment.png",
        title=f"Exemplar {exemplar_trial_id}",
    )
    if phase_unique_projection.shape[1] >= 3:
        _plot_trajectory_overlay_3d(
            labeled_phase_df,
            phase_unique_projection,
            output_path=figures_dir / "pc1_pc2_pc3_phase_overlay.png",
            title="Selected Cohort Trajectories in PC1-PC2-PC3",
            max_points_per_trial=max_trajectory_points_per_trial,
        )
        _plot_exemplar_trajectory_3d_phase(
            exemplar_phase_df,
            exemplar_projection,
            output_path=figures_dir / "exemplar_trial_pc1_pc2_pc3_phase.png",
            title=f"Exemplar {exemplar_trial_id}: PC1-PC2-PC3 by Phase",
            max_points=max_trajectory_points_per_trial,
        )
        _plot_exemplar_trajectory_3d_stage(
            exemplar_phase_df,
            exemplar_projection,
            output_path=figures_dir / "exemplar_trial_pc1_pc2_pc3_stage.png",
            title=f"Exemplar {exemplar_trial_id}: PC1-PC2-PC3 by Stage",
            max_points=max_trajectory_points_per_trial,
        )
        _plot_exemplar_trajectory_3d_probe_return(
            exemplar_phase_df,
            exemplar_projection,
            output_path=figures_dir / "exemplar_trial_pc1_pc2_pc3_probe_return.png",
            title=f"Exemplar {exemplar_trial_id}: PC1-PC2-PC3 Probe vs Return",
            max_points=max_trajectory_points_per_trial,
        )
        _plot_exemplar_trajectory_3d_time(
            exemplar_projection,
            output_path=figures_dir / "exemplar_trial_pc1_pc2_pc3_time.png",
            title=f"Exemplar {exemplar_trial_id}: PC1-PC2-PC3 by Time",
            max_points=max_trajectory_points_per_trial,
        )
        _plot_exemplar_trajectory_3d_phase(
            exemplar_full_df,
            exemplar_full_projection,
            output_path=figures_dir / "exemplar_trial_all_steps_pc1_pc2_pc3_phase.png",
            title=f"Exemplar {exemplar_trial_id}: All Controller Steps by Phase",
            max_points=max_trajectory_points_per_trial,
        )
        _plot_exemplar_trajectory_3d_stage(
            exemplar_full_df,
            exemplar_full_projection,
            output_path=figures_dir / "exemplar_trial_all_steps_pc1_pc2_pc3_stage.png",
            title=f"Exemplar {exemplar_trial_id}: All Controller Steps by Stage",
            max_points=max_trajectory_points_per_trial,
        )
        _plot_exemplar_trajectory_3d_probe_return(
            exemplar_full_df,
            exemplar_full_projection,
            output_path=figures_dir / "exemplar_trial_all_steps_pc1_pc2_pc3_probe_return.png",
            title=f"Exemplar {exemplar_trial_id}: All Controller Steps Probe vs Return",
            max_points=max_trajectory_points_per_trial,
        )
        _plot_exemplar_trajectory_3d_cliff_edge(
            exemplar_full_df,
            exemplar_full_projection,
            output_path=figures_dir / "exemplar_trial_all_steps_pc1_pc2_pc3_cliff_edge.png",
            title=f"Exemplar {exemplar_trial_id}: All Controller Steps by Cliff Edge State",
            max_points=max_trajectory_points_per_trial,
        )
        _plot_labeled_trajectory_3d(
            exemplar_full_df["transition_label"].to_numpy(dtype=object),
            exemplar_full_projection,
            output_path=figures_dir / "exemplar_trial_all_steps_pc1_pc2_pc3_transition.png",
            title=f"Exemplar {exemplar_trial_id}: Standard PCA by Transition Sequence",
            max_points=max_trajectory_points_per_trial,
            label_order=TRANSITION_LABELS,
            color_map=TRANSITION_COLORS,
            axis_labels=("PC1", "PC2", "PC3"),
            equalized_axes=False,
        )
        _plot_exemplar_trajectory_3d_time(
            exemplar_full_projection,
            output_path=figures_dir / "exemplar_trial_all_steps_pc1_pc2_pc3_time.png",
            title=f"Exemplar {exemplar_trial_id}: All Controller Steps by Time",
            max_points=max_trajectory_points_per_trial,
        )
        _animate_exemplar_trajectory_3d_probe_return(
            exemplar_full_df,
            exemplar_full_projection,
            output_path=figures_dir / "exemplar_trial_all_steps_pc1_pc2_pc3_probe_return.gif",
            title=f"Exemplar {exemplar_trial_id}: All Controller Steps Probe vs Return",
            max_points=max_animation_points,
        )
        _animate_exemplar_trajectory_3d_cliff_edge(
            exemplar_full_df,
            exemplar_full_projection,
            output_path=figures_dir / "exemplar_trial_all_steps_pc1_pc2_pc3_cliff_edge.gif",
            title=f"Exemplar {exemplar_trial_id}: All Controller Steps by Cliff Edge State",
            max_points=max_animation_points,
        )
        _animate_labeled_trajectory_3d(
            exemplar_full_df["transition_label"].to_numpy(dtype=object),
            exemplar_full_projection,
            output_path=figures_dir / "exemplar_trial_all_steps_pc1_pc2_pc3_transition.gif",
            title=f"Exemplar {exemplar_trial_id}: Standard PCA by Transition Sequence",
            max_points=max_animation_points,
            label_order=TRANSITION_LABELS,
            color_map=TRANSITION_COLORS,
            axis_labels=("PC1", "PC2", "PC3"),
            equalized_axes=False,
        )

        cliff_edge_labels = exemplar_full_df["cliff_edge_label"].to_numpy(dtype=object)
        reversal_window_mask = _reversal_window_mask(exemplar_full_df)
        reversal_window_df = exemplar_full_df.loc[reversal_window_mask].reset_index(drop=True)
        reversal_window_features = exemplar_full_features[reversal_window_mask]
        reversal_window_labels = reversal_window_df["cliff_edge_label"].to_numpy(dtype=object)
        reversal_window_pca = compute_pca_basis(reversal_window_features, n_components=3)
        reversal_window_projection = project_features(
            reversal_window_features,
            reversal_window_pca["mean"],
            reversal_window_pca["components"],
        )
        _plot_labeled_trajectory_3d(
            reversal_window_labels,
            reversal_window_projection,
            output_path=figures_dir / "exemplar_trial_reversal_window_pca_cliff_edge.png",
            title=f"Exemplar {exemplar_trial_id}: Reversal-Window PCA",
            max_points=max_trajectory_points_per_trial,
            label_order=CLIFF_EDGE_LABELS,
            color_map=CLIFF_EDGE_COLORS,
            axis_labels=("PC1", "PC2", "PC3"),
        )
        _animate_labeled_trajectory_3d(
            reversal_window_labels,
            reversal_window_projection,
            output_path=figures_dir / "exemplar_trial_reversal_window_pca_cliff_edge.gif",
            title=f"Exemplar {exemplar_trial_id}: Reversal-Window PCA",
            max_points=max_animation_points,
            label_order=CLIFF_EDGE_LABELS,
            color_map=CLIFF_EDGE_COLORS,
            axis_labels=("PC1", "PC2", "PC3"),
        )

        stage_augmented_features, stage_channel_scale = _augment_features_with_labels(
            exemplar_full_features,
            cliff_edge_labels,
            label_order=CLIFF_EDGE_LABELS,
        )
        stage_augmented_pca = compute_pca_basis(stage_augmented_features, n_components=3)
        stage_augmented_projection = project_features(
            stage_augmented_features,
            stage_augmented_pca["mean"],
            stage_augmented_pca["components"],
        )
        _plot_labeled_trajectory_3d(
            cliff_edge_labels,
            stage_augmented_projection,
            output_path=figures_dir / "exemplar_trial_all_steps_stage_augmented_pca_cliff_edge.png",
            title=f"Exemplar {exemplar_trial_id}: Stage-Augmented PCA",
            max_points=max_trajectory_points_per_trial,
            label_order=CLIFF_EDGE_LABELS,
            color_map=CLIFF_EDGE_COLORS,
            axis_labels=("PC1", "PC2", "PC3"),
        )
        _animate_labeled_trajectory_3d(
            cliff_edge_labels,
            stage_augmented_projection,
            output_path=figures_dir / "exemplar_trial_all_steps_stage_augmented_pca_cliff_edge.gif",
            title=f"Exemplar {exemplar_trial_id}: Stage-Augmented PCA",
            max_points=max_animation_points,
            label_order=CLIFF_EDGE_LABELS,
            color_map=CLIFF_EDGE_COLORS,
            axis_labels=("PC1", "PC2", "PC3"),
        )

        dpca_stage_tensor, dpca_stage_lengths = _build_stage_tensor(
            exemplar_full_features,
            cliff_edge_labels,
            stage_order=CLIFF_EDGE_LABELS,
            bins_per_stage=DEFAULT_STAGE_RESAMPLE_BINS,
        )
        dpca = compute_plain_dpca(dpca_stage_tensor)
        dpca_projection = np.asarray(dpca["projection"], dtype=np.float32)
        dpca_sequence_labels = np.repeat(np.asarray(CLIFF_EDGE_LABELS, dtype=object), DEFAULT_STAGE_RESAMPLE_BINS)
        _plot_labeled_trajectory_3d(
            dpca_sequence_labels,
            dpca_projection,
            output_path=figures_dir / "exemplar_trial_all_steps_dpca_cliff_edge.png",
            title=f"Exemplar {exemplar_trial_id}: Plain dPCA",
            max_points=max_trajectory_points_per_trial,
            label_order=CLIFF_EDGE_LABELS,
            color_map=CLIFF_EDGE_COLORS,
            axis_labels=("dPC time", "dPC stage", "dPC interaction"),
        )
        _animate_labeled_trajectory_3d(
            dpca_sequence_labels,
            dpca_projection,
            output_path=figures_dir / "exemplar_trial_all_steps_dpca_cliff_edge.gif",
            title=f"Exemplar {exemplar_trial_id}: Plain dPCA",
            max_points=max_animation_points,
            label_order=CLIFF_EDGE_LABELS,
            color_map=CLIFF_EDGE_COLORS,
            axis_labels=("dPC time", "dPC stage", "dPC interaction"),
        )
        if len(comparison_trial_ids) > 0:
            comparison_exemplar_trial_id = comparison_trial_ids[0]
            comparison_exemplar_projection = comparison_phase_projection[
                comparison_phase_df["trial_id"].to_numpy(dtype=object) == comparison_exemplar_trial_id
            ]
            comparison_exemplar_full_df = comparison_full_df.loc[
                comparison_full_df["trial_id"] == comparison_exemplar_trial_id
            ].reset_index(drop=True)
            comparison_exemplar_full_projection = comparison_all_projection[
                comparison_full_df["trial_id"].to_numpy(dtype=object) == comparison_exemplar_trial_id
            ]
            _plot_condition_comparison_3d(
                exemplar_projection,
                comparison_exemplar_projection,
                output_path=figures_dir / "selected_vs_mirrored_pc1_pc2_pc3.png",
                title="Selected vs Mirrored Trajectory in Shared PCA Space",
                selected_label=f"{bundle.selected_side} exemplar",
                comparison_label=f"{comparison_side} exemplar",
                max_points=max_trajectory_points_per_trial,
            )
            _plot_exemplar_trajectory_3d_probe_return(
                comparison_exemplar_full_df,
                comparison_exemplar_full_projection,
                output_path=figures_dir / "mirrored_exemplar_all_steps_pc1_pc2_pc3_probe_return.png",
                title=f"Exemplar {comparison_exemplar_trial_id}: All Controller Steps Probe vs Return",
                max_points=max_trajectory_points_per_trial,
            )
            _plot_exemplar_trajectory_3d_time(
                comparison_exemplar_full_projection,
                output_path=figures_dir / "mirrored_exemplar_all_steps_pc1_pc2_pc3_time.png",
                title=f"Exemplar {comparison_exemplar_trial_id}: All Controller Steps by Time",
                max_points=max_trajectory_points_per_trial,
            )

    cell_loading_magnitude, channel_loading_magnitude = _plot_loading_maps(
        pca["components"],
        output_path=figures_dir / "pc_loading_maps.png",
        title="Grid-Hidden PCA Loading Magnitudes",
    )

    component_payload: dict[str, np.ndarray] = {
        "mean": pca["mean"],
        "components": pca["components"],
        "explained_variance": pca["explained_variance"],
        "explained_variance_ratio": pca["explained_variance_ratio"],
        "phase_unique_projection": phase_unique_projection.astype(np.float32),
        "all_projection": all_projection.astype(np.float32),
        "cell_loading_magnitude": cell_loading_magnitude,
        "channel_loading_magnitude": channel_loading_magnitude,
    }
    if phase_unique_projection.shape[1] >= 3:
        component_payload.update(
            {
                "reversal_window_pca_components": reversal_window_pca["components"],
                "reversal_window_pca_explained_variance_ratio": reversal_window_pca["explained_variance_ratio"],
                "reversal_window_pca_projection": reversal_window_projection.astype(np.float32),
                "stage_augmented_pca_components": stage_augmented_pca["components"],
                "stage_augmented_pca_explained_variance_ratio": stage_augmented_pca["explained_variance_ratio"],
                "stage_augmented_pca_projection": stage_augmented_projection.astype(np.float32),
                "dpca_components": np.asarray(dpca["components"], dtype=np.float32),
                "dpca_projection": dpca_projection.astype(np.float32),
                "dpca_explained_variance_ratio": np.asarray(dpca["explained_variance_ratio"], dtype=np.float32),
            }
        )
    np.savez_compressed(output_path / "components.npz", **component_payload)

    summary = {
        "bundle_dir": str(Path(bundle_dir)),
        "selected_side": bundle.selected_side,
        "selected_trial_ids": bundle.selected_trial_ids,
        "selected_trial_count": len(bundle.selected_trial_ids),
        "phase_unique_sample_count": int(len(phase_unique_features)),
        "all_selected_sample_count": int(len(all_features)),
        "feature_count": int(phase_unique_features.shape[1]),
        "grid_hidden_slice": [GRID_HIDDEN_START, GRID_HIDDEN_STOP],
        "pca_fit_basis": "phase_unique_samples",
        "phase_unique_micro_index": PHASE_UNIQUE_MICRO_INDEX,
        "n_components": int(pca["components"].shape[0]),
        "explained_variance": [float(value) for value in pca["explained_variance"]],
        "explained_variance_ratio": [float(value) for value in pca["explained_variance_ratio"]],
        "plot_stride": int(plot_stride),
        "max_plot_points": int(max_plot_points),
        "max_trajectory_points_per_trial": int(max_trajectory_points_per_trial),
        "max_animation_points": int(max_animation_points),
        "phase_label_counts_all_samples": {
            phase_label: int(np.sum(labeled_full_df["phase_label"].to_numpy(dtype=object) == phase_label))
            for phase_label in PHASE_LABELS
        },
        "phase_label_counts_phase_unique": {
            phase_label: int(np.sum(labeled_phase_df["phase_label"].to_numpy(dtype=object) == phase_label))
            for phase_label in PHASE_LABELS
        },
        "stage_label_counts_all_samples": {
            stage_label: int(np.sum(labeled_full_df["stage_label"].to_numpy(dtype=object) == stage_label))
            for stage_label in STAGE_LABELS
        },
        "stage_label_counts_phase_unique": {
            stage_label: int(np.sum(labeled_phase_df["stage_label"].to_numpy(dtype=object) == stage_label))
            for stage_label in STAGE_LABELS
        },
        "probe_return_counts_all_samples": {
            label: int(np.sum(labeled_full_df["probe_return_label"].to_numpy(dtype=object) == label))
            for label in PROBE_RETURN_LABELS
        },
        "probe_return_counts_phase_unique": {
            label: int(np.sum(labeled_phase_df["probe_return_label"].to_numpy(dtype=object) == label))
            for label in PROBE_RETURN_LABELS
        },
        "cliff_edge_label_counts_all_samples": {
            label: int(np.sum(labeled_full_df["cliff_edge_label"].to_numpy(dtype=object) == label))
            for label in CLIFF_EDGE_LABELS
        },
        "cliff_edge_label_counts_phase_unique": {
            label: int(np.sum(labeled_phase_df["cliff_edge_label"].to_numpy(dtype=object) == label))
            for label in CLIFF_EDGE_LABELS
        },
        "transition_label_counts_all_samples": {
            label: int(np.sum(labeled_full_df["transition_label"].to_numpy(dtype=object) == label))
            for label in TRANSITION_LABELS
        },
        "transition_label_counts_phase_unique": {
            label: int(np.sum(labeled_phase_df["transition_label"].to_numpy(dtype=object) == label))
            for label in TRANSITION_LABELS
        },
        "reversal_window_sample_count": int(np.sum(reversal_window_mask)) if phase_unique_projection.shape[1] >= 3 else None,
        "reversal_window_cliff_edge_counts": {
            label: int(np.sum(reversal_window_labels == label))
            for label in CLIFF_EDGE_LABELS
        } if phase_unique_projection.shape[1] >= 3 else {},
        "reversal_window_explained_variance_ratio": [
            float(value) for value in reversal_window_pca["explained_variance_ratio"]
        ] if phase_unique_projection.shape[1] >= 3 else [],
        "stage_augmented_channel_scale": float(stage_channel_scale) if phase_unique_projection.shape[1] >= 3 else None,
        "dpca_stage_resample_bins": int(DEFAULT_STAGE_RESAMPLE_BINS) if phase_unique_projection.shape[1] >= 3 else None,
        "dpca_stage_lengths_all_samples": {
            label: int(length)
            for label, length in zip(CLIFF_EDGE_LABELS, dpca_stage_lengths.tolist(), strict=False)
        } if phase_unique_projection.shape[1] >= 3 else {},
        "stage_augmented_explained_variance_ratio": [
            float(value) for value in stage_augmented_pca["explained_variance_ratio"]
        ] if phase_unique_projection.shape[1] >= 3 else [],
        "dpca_explained_variance_ratio": [
            float(value) for value in np.asarray(dpca["explained_variance_ratio"], dtype=np.float32)
        ] if phase_unique_projection.shape[1] >= 3 else [],
        "exemplar_trial_id": exemplar_trial_id,
        "selected_identity": selected_identity,
        "comparison_side": comparison_side,
        "comparison_trial_count": int(len(comparison_trial_ids)),
        "comparison_identity": comparison_identity,
        "figures": {
            "task_code_projection": str(figures_dir / "pc1_pc2_task_code_geom.png"),
            "phase_projection": str(figures_dir / "pc1_pc2_phase_label.png"),
            "exemplar_alignment": str(figures_dir / "exemplar_trial_alignment.png"),
            "loading_maps": str(figures_dir / "pc_loading_maps.png"),
        },
    }
    if phase_unique_projection.shape[1] >= 3:
        summary["figures"].update(
            {
                "phase_overlay_3d": str(figures_dir / "pc1_pc2_pc3_phase_overlay.png"),
                "exemplar_3d_phase": str(figures_dir / "exemplar_trial_pc1_pc2_pc3_phase.png"),
                "exemplar_3d_stage": str(figures_dir / "exemplar_trial_pc1_pc2_pc3_stage.png"),
                "exemplar_3d_probe_return": str(figures_dir / "exemplar_trial_pc1_pc2_pc3_probe_return.png"),
                "exemplar_3d_time": str(figures_dir / "exemplar_trial_pc1_pc2_pc3_time.png"),
                "exemplar_all_steps_3d_phase": str(figures_dir / "exemplar_trial_all_steps_pc1_pc2_pc3_phase.png"),
                "exemplar_all_steps_3d_stage": str(figures_dir / "exemplar_trial_all_steps_pc1_pc2_pc3_stage.png"),
                "exemplar_all_steps_3d_probe_return": str(figures_dir / "exemplar_trial_all_steps_pc1_pc2_pc3_probe_return.png"),
                "exemplar_all_steps_3d_cliff_edge": str(figures_dir / "exemplar_trial_all_steps_pc1_pc2_pc3_cliff_edge.png"),
                "exemplar_all_steps_3d_transition": str(
                    figures_dir / "exemplar_trial_all_steps_pc1_pc2_pc3_transition.png"
                ),
                "exemplar_all_steps_3d_time": str(figures_dir / "exemplar_trial_all_steps_pc1_pc2_pc3_time.png"),
                "exemplar_all_steps_3d_probe_return_animation": str(
                    figures_dir / "exemplar_trial_all_steps_pc1_pc2_pc3_probe_return.gif"
                ),
                "exemplar_all_steps_3d_cliff_edge_animation": str(
                    figures_dir / "exemplar_trial_all_steps_pc1_pc2_pc3_cliff_edge.gif"
                ),
                "exemplar_all_steps_3d_transition_animation": str(
                    figures_dir / "exemplar_trial_all_steps_pc1_pc2_pc3_transition.gif"
                ),
                "exemplar_reversal_window_pca_cliff_edge": str(
                    figures_dir / "exemplar_trial_reversal_window_pca_cliff_edge.png"
                ),
                "exemplar_reversal_window_pca_cliff_edge_animation": str(
                    figures_dir / "exemplar_trial_reversal_window_pca_cliff_edge.gif"
                ),
                "exemplar_all_steps_stage_augmented_pca_cliff_edge": str(
                    figures_dir / "exemplar_trial_all_steps_stage_augmented_pca_cliff_edge.png"
                ),
                "exemplar_all_steps_stage_augmented_pca_cliff_edge_animation": str(
                    figures_dir / "exemplar_trial_all_steps_stage_augmented_pca_cliff_edge.gif"
                ),
                "exemplar_all_steps_dpca_cliff_edge": str(
                    figures_dir / "exemplar_trial_all_steps_dpca_cliff_edge.png"
                ),
                "exemplar_all_steps_dpca_cliff_edge_animation": str(
                    figures_dir / "exemplar_trial_all_steps_dpca_cliff_edge.gif"
                ),
            }
        )
        if len(comparison_trial_ids) > 0:
            summary["figures"].update(
                {
                    "selected_vs_mirrored_3d": str(figures_dir / "selected_vs_mirrored_pc1_pc2_pc3.png"),
                    "mirrored_exemplar_all_steps_3d_probe_return": str(
                        figures_dir / "mirrored_exemplar_all_steps_pc1_pc2_pc3_probe_return.png"
                    ),
                    "mirrored_exemplar_all_steps_3d_time": str(
                        figures_dir / "mirrored_exemplar_all_steps_pc1_pc2_pc3_time.png"
                    ),
                }
            )
    _write_json(output_path / "summary.json", summary)
    return summary
