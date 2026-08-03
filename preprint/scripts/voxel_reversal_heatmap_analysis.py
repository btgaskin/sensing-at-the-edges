#!/usr/bin/env python
"""Voxel-wise local-state predictiveness heatmaps around cliff reach."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from committed_direction_decoding import (
    _auc,
    _balanced_accuracy,
    _committed_labels,
    _mcc,
    _ridge_scores,
    _stable_seed,
)
from leading_edge_reversal_analysis import (
    DEFAULT_TRACE_ROOT,
    _drive_features,
    _leading_edge_indices,
    _local_geometry_features,
    _shuffle_blocks,
    _spring_features,
    _trace_path_for_trial,
)


DEFAULT_OUTPUT_DIR = Path(
    "paper_data/preprint/native_core_ablation_v1/figures/"
    "voxel_reversal_heatmap_deformation_off"
)


WINDOWS = {
    "pre_256_0": (-256, -1),
    "edge_impulse": (-32, 96),
    "early_ramification": (24, 160),
    "late_ramification": (160, 512),
    "edge_to_return": (0, 512),
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path)
    try:
        return {key: loaded[key] for key in loaded.files}
    finally:
        loaded.close()


def _cell_spring_indices(trace: dict[str, np.ndarray], cell_index: int) -> dict[str, np.ndarray]:
    points = np.asarray(trace["boxel_corners"][int(cell_index)], dtype=np.int32)
    a_idx = np.asarray(trace["spring_a_idx"], dtype=np.int32)
    b_idx = np.asarray(trace["spring_b_idx"], dtype=np.int32)
    a_in = np.isin(a_idx, points)
    b_in = np.isin(b_idx, points)
    return {
        "points": points,
        "attached_springs": np.flatnonzero(a_in | b_in).astype(np.int32),
        "connection_springs": np.flatnonzero(a_in ^ b_in).astype(np.int32),
        "internal_springs": np.flatnonzero(a_in & b_in).astype(np.int32),
    }


def _cell_features(
    *,
    trace: dict[str, np.ndarray],
    cell_index: int,
    inward_sign: float,
) -> np.ndarray:
    indices = _cell_spring_indices(trace, int(cell_index))
    attached = _spring_features(trace, indices["attached_springs"])
    drives = _drive_features(trace, np.asarray([int(cell_index)], dtype=np.int32))
    geometry = _local_geometry_features(
        trace=trace,
        cells=np.asarray([int(cell_index)], dtype=np.int32),
        points=indices["points"],
        inward_sign=float(inward_sign),
    )
    return np.concatenate([attached, drives, geometry], axis=1).astype(np.float32)


def _load_trials(
    *,
    trace_root: Path,
    future_window: int,
    min_displacement: float,
    persistence_steps: int,
    persistence_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[int, int], list[int]]:
    step_manifest = pd.read_csv(trace_root / "step_manifest.csv")
    trial_manifest = pd.read_csv(trace_root / "trial_manifest.csv")
    true_reach = trial_manifest[
        trial_manifest["first_cliff_reach_step"].notna()
        & (trial_manifest["goal_direction"] != 0)
    ].copy()
    true_reach = true_reach.sort_values(["surface_id", "goal_distance", "cliff_distance"], kind="stable")
    with np.load(trace_root / "static_arrays.npz") as static:
        robot_shape = tuple(int(v) for v in static["robot_morph"].shape)
    trials: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    leading_cells_seen: list[int] = []
    for _, trial_row in true_reach.iterrows():
        trial_id = str(trial_row["trial_id"])
        rows = step_manifest[step_manifest["trial_id"] == trial_id].sort_values("step_index", kind="stable").reset_index(drop=True)
        event_index = int(trial_row["first_cliff_reach_step"])
        trace = _load_npz(_trace_path_for_trial(trace_root, rows))
        edge = _leading_edge_indices(rows=rows, trace=trace, event_index=event_index)
        leading_cells_seen.extend(int(value) for value in edge["leading_cells"])
        labels, future_delta = _committed_labels(
            rows,
            future_window=int(future_window),
            min_displacement=float(min_displacement),
            persistence_steps=int(persistence_steps),
            persistence_fraction=float(persistence_fraction),
        )
        cell_count = int(trace["boxel_corners"].shape[0])
        cell_features = {
            int(cell_index): _cell_features(
                trace=trace,
                cell_index=int(cell_index),
                inward_sign=float(edge["inward_sign"]),
            )
            for cell_index in range(cell_count)
        }
        step_count = min([len(rows), len(labels), *(features.shape[0] for features in cell_features.values())])
        rows = rows.iloc[:step_count].reset_index(drop=True)
        cell_features = {key: value[:step_count] for key, value in cell_features.items()}
        trials.append(
            {
                "trial_id": trial_id,
                "surface_id": str(trial_row["surface_id"]),
                "rows": rows,
                "labels": labels[:step_count],
                "future_delta": future_delta[:step_count],
                "event_index": int(event_index),
                "returned_from_edge": bool(trial_row["returned_from_edge"]),
                "success": bool(trial_row["success"]),
                "cell_features": cell_features,
                "leading_cells": [int(value) for value in edge["leading_cells"]],
                "edge_side": str(edge["edge_side"]),
            }
        )
        for cell_index in range(cell_count):
            springs = _cell_spring_indices(trace, int(cell_index))
            metadata_rows.append(
                {
                    "trial_id": trial_id,
                    "surface_id": str(trial_row["surface_id"]),
                    "cell_index": int(cell_index),
                    "is_event_leading_cell": int(cell_index) in set(int(v) for v in edge["leading_cells"]),
                    "edge_side": str(edge["edge_side"]),
                    "attached_spring_count": int(springs["attached_springs"].size),
                    "connection_spring_count": int(springs["connection_springs"].size),
                    "internal_spring_count": int(springs["internal_springs"].size),
                    "event_index": int(event_index),
                    "returned_from_edge": bool(trial_row["returned_from_edge"]),
                    "success": bool(trial_row["success"]),
                    "committed_toward_count": int(np.sum(labels[:step_count] == 1)),
                    "committed_away_count": int(np.sum(labels[:step_count] == -1)),
                }
            )
    return trials, metadata_rows, robot_shape, sorted(set(leading_cells_seen))


def _windowed_xy(
    trial: dict[str, Any],
    *,
    cell_index: int,
    lag: int,
    rel_start: int,
    rel_stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(trial["cell_features"][int(cell_index)], dtype=np.float32)
    labels = np.asarray(trial["labels"], dtype=np.int8)
    usable = features.shape[0] - int(lag)
    if usable <= 0:
        return np.zeros((0, features.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int8)
    indices = np.arange(usable, dtype=np.int32)
    rel = indices - int(trial["event_index"])
    y = labels[indices + int(lag)]
    mask = (rel >= int(rel_start)) & (rel <= int(rel_stop)) & (y != 0)
    mask &= np.all(np.isfinite(features[indices]), axis=1)
    return features[indices][mask], y[mask].astype(np.int8)


def _evaluate_cell(
    trials: list[dict[str, Any]],
    *,
    cell_index: int,
    window_name: str,
    rel_start: int,
    rel_stop: int,
    lag: int,
    ridge_alpha: float,
    permutations: int,
    null_block_size: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(_stable_seed(int(random_seed), f"voxel-{cell_index}-{window_name}-{lag}"))
    for heldout in trials:
        train_x_blocks: list[np.ndarray] = []
        train_y_blocks: list[np.ndarray] = []
        for trial in trials:
            if trial["trial_id"] == heldout["trial_id"]:
                continue
            x_block, y_block = _windowed_xy(
                trial,
                cell_index=int(cell_index),
                lag=int(lag),
                rel_start=int(rel_start),
                rel_stop=int(rel_stop),
            )
            if y_block.size:
                train_x_blocks.append(x_block)
                train_y_blocks.append(y_block)
        test_x, test_y = _windowed_xy(
            heldout,
            cell_index=int(cell_index),
            lag=int(lag),
            rel_start=int(rel_start),
            rel_stop=int(rel_stop),
        )
        base = {
            "cell_index": int(cell_index),
            "window": str(window_name),
            "relative_start": int(rel_start),
            "relative_stop": int(rel_stop),
            "lag": int(lag),
            "heldout_trial_id": heldout["trial_id"],
            "surface_id": heldout["surface_id"],
            "sample_count": int(test_y.size),
            "toward_count": int(np.sum(test_y == 1)),
            "away_count": int(np.sum(test_y == -1)),
            "returned_from_edge": bool(heldout["returned_from_edge"]),
            "is_event_leading_cell": int(cell_index) in set(heldout["leading_cells"]),
        }
        if not train_y_blocks or test_y.size == 0:
            rows.append({**base, "skip_reason": "empty_train_or_test"})
            continue
        train_x = np.concatenate(train_x_blocks, axis=0)
        train_y = np.concatenate(train_y_blocks, axis=0)
        if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
            rows.append({**base, "skip_reason": "single_class_train_or_test"})
            continue
        scores = _ridge_scores(train_x, train_y, test_x, alpha=float(ridge_alpha))
        null_scores = []
        for _ in range(int(permutations)):
            shuffled_y = np.concatenate(
                [_shuffle_blocks(rng, block, int(null_block_size)) for block in train_y_blocks]
            )
            shuffled_scores = _ridge_scores(train_x, shuffled_y, test_x, alpha=float(ridge_alpha))
            null_scores.append(_balanced_accuracy(test_y, shuffled_scores))
        null_arr = np.asarray(null_scores, dtype=np.float32)
        actual_ba = _balanced_accuracy(test_y, scores)
        rows.append(
            {
                **base,
                "accuracy": float(np.mean(np.where(scores >= 0.0, 1, -1) == test_y)),
                "balanced_accuracy": actual_ba,
                "auc": _auc(test_y, scores),
                "mcc": _mcc(test_y, scores),
                "null_balanced_accuracy_mean": float(np.nanmean(null_arr)),
                "actual_minus_null_balanced_accuracy": float(actual_ba - np.nanmean(null_arr)),
                "skip_reason": "",
            }
        )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    if frame.empty or "balanced_accuracy" not in frame.columns:
        return []
    valid = frame[(frame["skip_reason"].fillna("") == "") & np.isfinite(frame["balanced_accuracy"])]
    out: list[dict[str, Any]] = []
    for (cell_index, window, lag), group in valid.groupby(["cell_index", "window", "lag"], sort=True):
        out.append(
            {
                "cell_index": int(cell_index),
                "window": str(window),
                "lag": int(lag),
                "heldout_trial_count": int(group["heldout_trial_id"].nunique()),
                "mean_balanced_accuracy": float(group["balanced_accuracy"].mean()),
                "mean_auc": float(group["auc"].mean()),
                "mean_mcc": float(group["mcc"].mean()),
                "mean_null_balanced_accuracy": float(group["null_balanced_accuracy_mean"].mean()),
                "mean_actual_minus_null_balanced_accuracy": float(group["actual_minus_null_balanced_accuracy"].mean()),
                "is_event_leading_cell": bool(group["is_event_leading_cell"].any()),
            }
        )
    return out


def _rank_rows(aggregate_rows: list[dict[str, Any]], *, leading_cells: list[int]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(aggregate_rows)
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for (window, lag), group in frame.groupby(["window", "lag"], sort=True):
        ordered = group.sort_values(
            ["mean_auc", "mean_actual_minus_null_balanced_accuracy"],
            ascending=False,
            kind="stable",
        ).reset_index(drop=True)
        for rank, record in ordered.iterrows():
            rows.append(
                {
                    "window": str(window),
                    "lag": int(lag),
                    "cell_index": int(record["cell_index"]),
                    "rank_by_auc": int(rank) + 1,
                    "mean_auc": float(record["mean_auc"]),
                    "mean_balanced_accuracy": float(record["mean_balanced_accuracy"]),
                    "mean_actual_minus_null_balanced_accuracy": float(record["mean_actual_minus_null_balanced_accuracy"]),
                    "is_event_leading_cell": int(record["cell_index"]) in set(leading_cells),
                }
            )
    return rows


def _earliest_rows(
    aggregate_rows: list[dict[str, Any]],
    *,
    windows: dict[str, tuple[int, int]],
    lag: int,
    auc_threshold: float,
    min_folds: int,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(aggregate_rows)
    if frame.empty:
        return []
    window_order = {name: index for index, name in enumerate(windows)}
    rows: list[dict[str, Any]] = []
    for cell_index, group in frame[frame["lag"] == int(lag)].groupby("cell_index", sort=True):
        group = group.copy()
        group["window_order"] = group["window"].map(window_order)
        group = group.sort_values("window_order", kind="stable")
        hit = group[
            (group["heldout_trial_count"] >= int(min_folds))
            & (group["mean_auc"] >= float(auc_threshold))
        ]
        if hit.empty:
            rows.append(
                {
                    "cell_index": int(cell_index),
                    "lag": int(lag),
                    "earliest_window": "",
                    "earliest_window_order": None,
                    "earliest_mean_auc": None,
                    "earliest_heldout_trial_count": 0,
                }
            )
        else:
            first = hit.iloc[0]
            rows.append(
                {
                    "cell_index": int(cell_index),
                    "lag": int(lag),
                    "earliest_window": str(first["window"]),
                    "earliest_window_order": int(first["window_order"]),
                    "earliest_mean_auc": float(first["mean_auc"]),
                    "earliest_heldout_trial_count": int(first["heldout_trial_count"]),
                }
            )
    return rows


def _grid_for_metric(
    aggregate: pd.DataFrame,
    *,
    robot_shape: tuple[int, int],
    window: str,
    lag: int,
    metric: str,
) -> np.ndarray:
    grid = np.full(robot_shape, np.nan, dtype=np.float32)
    subset = aggregate[(aggregate["window"] == str(window)) & (aggregate["lag"] == int(lag))]
    for _, row in subset.iterrows():
        cell = int(row["cell_index"])
        r = cell // int(robot_shape[1])
        c = cell % int(robot_shape[1])
        if 0 <= r < int(robot_shape[0]) and 0 <= c < int(robot_shape[1]):
            grid[r, c] = float(row[metric])
    return grid


def _plot_spatial_heatmap(
    *,
    aggregate_rows: list[dict[str, Any]],
    robot_shape: tuple[int, int],
    window: str,
    lag: int,
    metric: str,
    leading_cells: list[int],
    output_path: Path,
) -> None:
    aggregate = pd.DataFrame(aggregate_rows)
    grid = _grid_for_metric(aggregate, robot_shape=robot_shape, window=window, lag=int(lag), metric=metric)
    fig, ax = plt.subplots(figsize=(9, 3.3))
    if np.all(~np.isfinite(grid)):
        image = ax.imshow(np.zeros_like(grid), cmap="viridis", vmin=0.0, vmax=1.0)
    else:
        image = ax.imshow(grid, cmap="viridis", vmin=np.nanmin(grid), vmax=np.nanmax(grid))
    for r in range(robot_shape[0]):
        for c in range(robot_shape[1]):
            cell = r * robot_shape[1] + c
            value = grid[r, c]
            text = f"{cell}\n{value:.2f}" if np.isfinite(value) else f"{cell}\nNA"
            ax.text(c, r, text, ha="center", va="center", color="white", fontsize=8)
            if cell in set(leading_cells):
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2.5))
    ax.set_xticks(range(robot_shape[1]))
    ax.set_yticks(range(robot_shape[0]))
    ax.set_title(f"{metric} | {window}, lag {lag}")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_temporal_heatmap(
    *,
    aggregate_rows: list[dict[str, Any]],
    windows: dict[str, tuple[int, int]],
    lag: int,
    metric: str,
    leading_cells: list[int],
    output_path: Path,
) -> None:
    frame = pd.DataFrame(aggregate_rows)
    cell_indices = sorted(int(v) for v in frame["cell_index"].dropna().unique())
    matrix = np.full((len(cell_indices), len(windows)), np.nan, dtype=np.float32)
    for row_index, cell in enumerate(cell_indices):
        for col_index, window in enumerate(windows):
            subset = frame[
                (frame["cell_index"] == int(cell))
                & (frame["window"] == str(window))
                & (frame["lag"] == int(lag))
            ]
            if not subset.empty:
                matrix[row_index, col_index] = float(subset.iloc[0][metric])
    fig_height = max(4.0, 0.35 * len(cell_indices))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(list(windows.keys()), rotation=35, ha="right")
    ax.set_yticks(range(len(cell_indices)))
    ax.set_yticklabels([str(cell) + (" *" if cell in set(leading_cells) else "") for cell in cell_indices])
    ax.set_title(f"Voxel {metric} by time window | lag {lag} (* event leading cell)")
    fig.colorbar(image, ax=ax, fraction=0.026, pad=0.03)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def analyze_voxel_heatmaps(
    trace_root: Path,
    output_dir: Path,
    *,
    future_window: int,
    min_displacement: float,
    persistence_steps: int,
    persistence_fraction: float,
    lags: list[int],
    ridge_alpha: float,
    permutations: int,
    null_block_size: int,
    random_seed: int,
    heatmap_lag: int,
    auc_threshold: float,
    min_folds: int,
) -> dict[str, Any]:
    trace_root = trace_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trials, metadata_rows, robot_shape, leading_cells = _load_trials(
        trace_root=trace_root,
        future_window=int(future_window),
        min_displacement=float(min_displacement),
        persistence_steps=int(persistence_steps),
        persistence_fraction=float(persistence_fraction),
    )
    cell_count = int(robot_shape[0]) * int(robot_shape[1])
    metric_rows: list[dict[str, Any]] = []
    for cell_index in range(cell_count):
        for window_name, (rel_start, rel_stop) in WINDOWS.items():
            for lag in lags:
                metric_rows.extend(
                    _evaluate_cell(
                        trials,
                        cell_index=int(cell_index),
                        window_name=str(window_name),
                        rel_start=int(rel_start),
                        rel_stop=int(rel_stop),
                        lag=int(lag),
                        ridge_alpha=float(ridge_alpha),
                        permutations=int(permutations),
                        null_block_size=int(null_block_size),
                        random_seed=int(random_seed),
                    )
                )
    aggregate_rows = _aggregate(metric_rows)
    rank_rows = _rank_rows(aggregate_rows, leading_cells=leading_cells)
    earliest = _earliest_rows(
        aggregate_rows,
        windows=WINDOWS,
        lag=int(heatmap_lag),
        auc_threshold=float(auc_threshold),
        min_folds=int(min_folds),
    )
    _write_csv(output_dir / "voxel_trial_metadata.csv", metadata_rows)
    _write_csv(output_dir / "voxel_decoder_metrics.csv", metric_rows)
    _write_csv(output_dir / "voxel_decoder_summary.csv", aggregate_rows)
    _write_csv(output_dir / "voxel_window_ranks.csv", rank_rows)
    _write_csv(output_dir / "voxel_earliest_predictive_window.csv", earliest)
    _plot_spatial_heatmap(
        aggregate_rows=aggregate_rows,
        robot_shape=robot_shape,
        window="edge_impulse",
        lag=int(heatmap_lag),
        metric="mean_auc",
        leading_cells=leading_cells,
        output_path=output_dir / f"spatial_heatmap_edge_impulse_lag{heatmap_lag}_auc.png",
    )
    _plot_spatial_heatmap(
        aggregate_rows=aggregate_rows,
        robot_shape=robot_shape,
        window="edge_to_return",
        lag=int(heatmap_lag),
        metric="mean_auc",
        leading_cells=leading_cells,
        output_path=output_dir / f"spatial_heatmap_edge_to_return_lag{heatmap_lag}_auc.png",
    )
    _plot_temporal_heatmap(
        aggregate_rows=aggregate_rows,
        windows=WINDOWS,
        lag=int(heatmap_lag),
        metric="mean_auc",
        leading_cells=leading_cells,
        output_path=output_dir / f"temporal_voxel_heatmap_lag{heatmap_lag}_auc.png",
    )
    _plot_temporal_heatmap(
        aggregate_rows=aggregate_rows,
        windows=WINDOWS,
        lag=int(heatmap_lag),
        metric="mean_actual_minus_null_balanced_accuracy",
        leading_cells=leading_cells,
        output_path=output_dir / f"temporal_voxel_heatmap_lag{heatmap_lag}_actual_minus_null_ba.png",
    )
    summary = {
        "schema_version": 1,
        "kind": "voxel_reversal_heatmap_analysis",
        "trace_root": str(trace_root),
        "output_dir": str(output_dir),
        "included_trials": [trial["trial_id"] for trial in trials],
        "robot_shape": list(robot_shape),
        "event_leading_cells": [int(value) for value in leading_cells],
        "feature_policy": "per-cell attached springs, h/v drives, local corner/point geometry and velocities; no absolute COM/cliff clearance",
        "windows": {key: list(value) for key, value in WINDOWS.items()},
        "lags": [int(value) for value in lags],
        "heatmap_lag": int(heatmap_lag),
        "auc_threshold_for_earliest": float(auc_threshold),
        "min_folds_for_earliest": int(min_folds),
        "outputs": {
            "trial_metadata": str(output_dir / "voxel_trial_metadata.csv"),
            "decoder_metrics": str(output_dir / "voxel_decoder_metrics.csv"),
            "decoder_summary": str(output_dir / "voxel_decoder_summary.csv"),
            "window_ranks": str(output_dir / "voxel_window_ranks.csv"),
            "earliest_predictive_window": str(output_dir / "voxel_earliest_predictive_window.csv"),
            "spatial_heatmap_edge_impulse_auc": str(output_dir / f"spatial_heatmap_edge_impulse_lag{heatmap_lag}_auc.png"),
            "spatial_heatmap_edge_to_return_auc": str(output_dir / f"spatial_heatmap_edge_to_return_lag{heatmap_lag}_auc.png"),
            "temporal_heatmap_auc": str(output_dir / f"temporal_voxel_heatmap_lag{heatmap_lag}_auc.png"),
            "temporal_heatmap_actual_minus_null_ba": str(output_dir / f"temporal_voxel_heatmap_lag{heatmap_lag}_actual_minus_null_ba.png"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def _parse_lags(value: str) -> list[int]:
    lags = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not lags:
        raise argparse.ArgumentTypeError("expected at least one lag")
    if any(lag < 0 for lag in lags):
        raise argparse.ArgumentTypeError("lags must be non-negative")
    return lags


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare voxel-local predictiveness across the whole body."
    )
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--future-window", type=int, default=32)
    parser.add_argument("--min-displacement", type=float, default=0.025)
    parser.add_argument("--persistence-steps", type=int, default=12)
    parser.add_argument("--persistence-fraction", type=float, default=0.75)
    parser.add_argument("--lags", type=_parse_lags, default=_parse_lags("0,8,16"))
    parser.add_argument("--heatmap-lag", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--permutations", type=int, default=25)
    parser.add_argument("--null-block-size", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=137)
    parser.add_argument("--auc-threshold", type=float, default=0.75)
    parser.add_argument("--min-folds", type=int, default=2)
    args = parser.parse_args()
    summary = analyze_voxel_heatmaps(
        args.trace_root,
        args.output_dir,
        future_window=int(args.future_window),
        min_displacement=float(args.min_displacement),
        persistence_steps=int(args.persistence_steps),
        persistence_fraction=float(args.persistence_fraction),
        lags=list(args.lags),
        ridge_alpha=float(args.ridge_alpha),
        permutations=int(args.permutations),
        null_block_size=int(args.null_block_size),
        random_seed=int(args.random_seed),
        heatmap_lag=int(args.heatmap_lag),
        auc_threshold=float(args.auc_threshold),
        min_folds=int(args.min_folds),
    )
    print(json.dumps(summary["outputs"], indent=2))


if __name__ == "__main__":
    main()
