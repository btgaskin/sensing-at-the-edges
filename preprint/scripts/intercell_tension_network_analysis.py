#!/usr/bin/env python
"""Inter-cell spring/tension analysis around cliff reach."""

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
from PIL import Image

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
    _safe_scale,
    _shuffle_blocks,
    _spring_features,
    _trace_path_for_trial,
)
from voxel_reversal_heatmap_analysis import WINDOWS, _cell_features


DEFAULT_OUTPUT_DIR = Path(
    "paper_data/preprint/native_core_ablation_v1/figures/"
    "intercell_tension_deformation_off"
)


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


def _derive_spring_metadata(height: int, width: int) -> list[dict[str, Any]]:
    """Mirror ``precompute_grid`` spring ordering without importing JAX."""
    rows: list[dict[str, Any]] = []
    n_horiz = (height + 1) * width
    vert_offset = n_horiz
    diag_offset = n_horiz + height * (width + 1)

    for gy in range(height + 1):
        for gx in range(width):
            spring_index = gy * width + gx
            cell_above = gy * width + gx if 0 <= gy < height else -1
            cell_below = (gy - 1) * width + gx if 0 <= gy - 1 < height else -1
            cell_a = cell_above if cell_above >= 0 else cell_below
            cell_b = cell_below if cell_above >= 0 and cell_below >= 0 else -1
            rows.append(
                {
                    "spring_index": spring_index,
                    "spring_type": "horizontal",
                    "grid_x": gx,
                    "grid_y": gy,
                    "cell_a": cell_a,
                    "cell_b": cell_b,
                    "is_intercell": cell_a >= 0 and cell_b >= 0,
                    "cell_pair": _pair_key(cell_a, cell_b),
                }
            )

    for gy in range(height):
        for gx in range(width + 1):
            spring_index = vert_offset + gy * (width + 1) + gx
            cell_left = gy * width + gx if 0 <= gx < width else -1
            cell_right = gy * width + gx - 1 if 0 <= gx - 1 < width else -1
            cell_a = cell_left if cell_left >= 0 else cell_right
            cell_b = cell_right if cell_left >= 0 and cell_right >= 0 else -1
            rows.append(
                {
                    "spring_index": spring_index,
                    "spring_type": "vertical",
                    "grid_x": gx,
                    "grid_y": gy,
                    "cell_a": cell_a,
                    "cell_b": cell_b,
                    "is_intercell": cell_a >= 0 and cell_b >= 0,
                    "cell_pair": _pair_key(cell_a, cell_b),
                }
            )

    for cell in range(height * width):
        row = cell // width
        col = cell % width
        for local_diag in range(2):
            spring_index = diag_offset + 2 * cell + local_diag
            rows.append(
                {
                    "spring_index": spring_index,
                    "spring_type": "diagonal",
                    "grid_x": col,
                    "grid_y": row,
                    "cell_a": cell,
                    "cell_b": -1,
                    "is_intercell": False,
                    "cell_pair": "",
                }
            )
    return rows


def _pair_key(a: int, b: int) -> str:
    if int(a) < 0 or int(b) < 0:
        return ""
    low, high = sorted((int(a), int(b)))
    return f"{low}-{high}"


def _standardize_train_test(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale > 1.0e-6, scale, 1.0).astype(np.float32)
    return ((train - mean) / scale).astype(np.float32), ((test - mean) / scale).astype(np.float32)


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    train_z, test_z = _standardize_train_test(train_x, test_x)
    train_design = np.column_stack([np.ones(train_z.shape[0], dtype=np.float32), train_z])
    test_design = np.column_stack([np.ones(test_z.shape[0], dtype=np.float32), test_z])
    penalty = np.eye(train_design.shape[1], dtype=np.float32) * float(alpha)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(
        train_design.T @ train_design + penalty,
        train_design.T @ train_y.astype(np.float32),
    )
    return (test_design @ weights).astype(np.float32)


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if y_true.size < 3 or float(np.std(y_true)) <= 1.0e-8 or float(np.std(y_pred)) <= 1.0e-8:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= 1.0e-8:
        return float("nan")
    return float(1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom)


def _spring_force_features(
    trace: dict[str, np.ndarray],
    spring_indices: np.ndarray,
    *,
    spring_const: np.ndarray,
) -> np.ndarray:
    if spring_indices.size == 0:
        return np.zeros((int(trace["spring_current_length"].shape[0]), 0), dtype=np.float32)
    indices = np.asarray(spring_indices, dtype=np.int32)
    current = np.asarray(trace["spring_current_length"][:, indices], dtype=np.float32)
    rest = np.asarray(trace["spring_rest_length"][:, indices], dtype=np.float32)
    init = np.asarray(trace["spring_init_rest_length"], dtype=np.float32)[indices]
    init = np.where(np.abs(init) > 1.0e-6, init, 1.0).astype(np.float32)
    k = np.asarray(spring_const, dtype=np.float32)[indices]
    signed_force = (current - rest) * k
    force_per_init = signed_force / init
    strain_to_rest = current / np.maximum(rest, 1.0e-6) - 1.0
    actuation = rest / init - 1.0
    return np.concatenate(
        [
            _safe_scale(current, init),
            _safe_scale(rest, init),
            strain_to_rest,
            signed_force,
            force_per_init,
            actuation,
        ],
        axis=1,
    ).astype(np.float32)


def _intercell_stream(
    trace: dict[str, np.ndarray],
    spring_index: int,
    *,
    spring_const: np.ndarray,
) -> np.ndarray:
    spring_indices = np.asarray([int(spring_index)], dtype=np.int32)
    length_features = _spring_features(trace, spring_indices)
    force_features = _spring_force_features(trace, spring_indices, spring_const=spring_const)
    return np.concatenate([length_features, force_features], axis=1).astype(np.float32)


def _intercell_group_stream(
    trace: dict[str, np.ndarray],
    spring_indices: list[int],
    *,
    spring_const: np.ndarray,
) -> np.ndarray:
    indices = np.asarray([int(value) for value in spring_indices], dtype=np.int32)
    length_features = _spring_features(trace, indices)
    force_features = _spring_force_features(trace, indices, spring_const=spring_const)
    return np.concatenate([length_features, force_features], axis=1).astype(np.float32)


def _group_rows(intercell_springs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spring_by_pair = {str(row["cell_pair"]): int(row["spring_index"]) for row in intercell_springs}

    def row(name: str, pairs: list[str]) -> dict[str, Any]:
        member_pairs = [pair for pair in pairs if pair in spring_by_pair]
        member_indices = [spring_by_pair[pair] for pair in member_pairs]
        return {
            "spring_index": -1,
            "spring_type": "group",
            "grid_x": -1,
            "grid_y": -1,
            "cell_a": -1,
            "cell_b": -1,
            "is_intercell": True,
            "cell_pair": name,
            "member_cell_pairs": ",".join(member_pairs),
            "member_spring_indices": ",".join(str(value) for value in member_indices),
        }

    return [
        row("group_cell0_direct_shared", ["0-1", "0-7"]),
        row("group_nexus_0_1_7_incident", ["0-1", "0-7", "1-2", "1-8", "7-8"]),
        row("group_square_0_1_7_8", ["0-1", "0-7", "1-8", "7-8"]),
        row("group_leading_band_shared", ["0-1", "0-7", "7-8"]),
        row("group_all_intercell", sorted(spring_by_pair)),
    ]


def _windowed_xy(
    trial: dict[str, Any],
    feature_key: tuple[str, int] | tuple[str, str],
    *,
    lag: int,
    rel_start: int,
    rel_stop: int,
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    feature_group, feature_id = feature_key
    if feature_group == "cell":
        features = trial["cell_features"][int(feature_id)]
    elif feature_group == "edge":
        features = trial["edge_features"][str(feature_id)]
    else:
        raise ValueError(f"unknown feature group: {feature_group}")
    features = np.asarray(features, dtype=np.float32)
    values = np.asarray(trial[target], dtype=np.float32 if target == "future_delta" else np.int8)
    usable = features.shape[0] - int(lag)
    if usable <= 0:
        return np.zeros((0, features.shape[1]), dtype=np.float32), np.zeros((0,), dtype=values.dtype)
    indices = np.arange(usable, dtype=np.int32)
    rel = indices - int(trial["event_index"])
    y = values[indices + int(lag)]
    mask = (rel >= int(rel_start)) & (rel <= int(rel_stop))
    if target == "labels":
        mask &= y != 0
    else:
        mask &= np.isfinite(y)
    mask &= np.all(np.isfinite(features[indices]), axis=1)
    return features[indices][mask], y[mask]


def _load_trials(
    *,
    trace_root: Path,
    future_window: int,
    min_displacement: float,
    persistence_steps: int,
    persistence_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[int, int], np.ndarray, list[dict[str, Any]]]:
    step_manifest = pd.read_csv(trace_root / "step_manifest.csv")
    trial_manifest = pd.read_csv(trace_root / "trial_manifest.csv")
    true_reach = trial_manifest[
        trial_manifest["first_cliff_reach_step"].notna()
        & (trial_manifest["goal_direction"] != 0)
    ].copy()
    true_reach = true_reach.sort_values(["surface_id", "goal_distance", "cliff_distance"], kind="stable")
    with np.load(trace_root / "static_arrays.npz") as static:
        robot_shape = tuple(int(v) for v in static["robot_morph"].shape)
        spring_const = np.asarray(static["spring_const"], dtype=np.float32)
    spring_meta = _derive_spring_metadata(robot_shape[0], robot_shape[1])
    intercell_springs = [row for row in spring_meta if bool(row["is_intercell"])]
    group_springs = _group_rows(intercell_springs)
    spring_meta = spring_meta + group_springs

    trials: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    for _, trial_row in true_reach.iterrows():
        trial_id = str(trial_row["trial_id"])
        rows = step_manifest[step_manifest["trial_id"] == trial_id].sort_values("step_index", kind="stable").reset_index(drop=True)
        event_index = int(trial_row["first_cliff_reach_step"])
        trace = _load_npz(_trace_path_for_trial(trace_root, rows))
        edge = _leading_edge_indices(rows=rows, trace=trace, event_index=event_index)
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
        edge_features = {
            str(row["cell_pair"]): _intercell_stream(
                trace,
                int(row["spring_index"]),
                spring_const=spring_const,
            )
            for row in intercell_springs
        }
        for row in group_springs:
            member_indices = [
                int(value)
                for value in str(row["member_spring_indices"]).split(",")
                if value.strip()
            ]
            edge_features[str(row["cell_pair"])] = _intercell_group_stream(
                trace,
                member_indices,
                spring_const=spring_const,
            )
        step_count = min(
            [
                len(rows),
                len(labels),
                *(value.shape[0] for value in cell_features.values()),
                *(value.shape[0] for value in edge_features.values()),
            ]
        )
        rows = rows.iloc[:step_count].reset_index(drop=True)
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
                "cell_features": {key: value[:step_count] for key, value in cell_features.items()},
                "edge_features": {key: value[:step_count] for key, value in edge_features.items()},
                "leading_cells": [int(value) for value in edge["leading_cells"]],
                "edge_side": str(edge["edge_side"]),
                "trace": trace,
            }
        )
        trial_rows.append(
            {
                "trial_id": trial_id,
                "surface_id": str(trial_row["surface_id"]),
                "event_index": int(event_index),
                "returned_from_edge": bool(trial_row["returned_from_edge"]),
                "success": bool(trial_row["success"]),
                "edge_side": str(edge["edge_side"]),
                "leading_cells": ",".join(str(int(value)) for value in edge["leading_cells"]),
                "step_count": int(step_count),
            }
        )
    return trials, trial_rows, robot_shape, spring_const, spring_meta


def _evaluate_intercell_binary(
    trials: list[dict[str, Any]],
    *,
    cell_pairs: list[str],
    windows: dict[str, tuple[int, int]],
    lags: list[int],
    ridge_alpha: float,
    permutations: int,
    null_block_size: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(_stable_seed(int(random_seed), "intercell-null"))
    for pair in cell_pairs:
        for window_name, (rel_start, rel_stop) in windows.items():
            for lag in lags:
                for heldout in trials:
                    train_x_blocks: list[np.ndarray] = []
                    train_y_blocks: list[np.ndarray] = []
                    for trial in trials:
                        if trial["trial_id"] == heldout["trial_id"]:
                            continue
                        x_block, y_block = _windowed_xy(
                            trial,
                            ("edge", pair),
                            lag=int(lag),
                            rel_start=int(rel_start),
                            rel_stop=int(rel_stop),
                            target="labels",
                        )
                        if y_block.size:
                            train_x_blocks.append(x_block)
                            train_y_blocks.append(y_block.astype(np.int8))
                    test_x, test_y = _windowed_xy(
                        heldout,
                        ("edge", pair),
                        lag=int(lag),
                        rel_start=int(rel_start),
                        rel_stop=int(rel_stop),
                        target="labels",
                    )
                    base = {
                        "cell_pair": pair,
                        "window": window_name,
                        "lag": int(lag),
                        "heldout_trial_id": heldout["trial_id"],
                        "sample_count": int(test_y.size),
                        "toward_count": int(np.sum(test_y == 1)),
                        "away_count": int(np.sum(test_y == -1)),
                    }
                    if not train_y_blocks or test_y.size == 0:
                        rows.append({**base, "skip_reason": "empty_train_or_test"})
                        continue
                    train_x = np.concatenate(train_x_blocks, axis=0)
                    train_y = np.concatenate(train_y_blocks, axis=0).astype(np.int8)
                    test_y = test_y.astype(np.int8)
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
                    null_mean = float(np.nanmean(np.asarray(null_scores, dtype=np.float32)))
                    ba = _balanced_accuracy(test_y, scores)
                    rows.append(
                        {
                            **base,
                            "balanced_accuracy": ba,
                            "auc": _auc(test_y, scores),
                            "mcc": _mcc(test_y, scores),
                            "null_balanced_accuracy_mean": null_mean,
                            "actual_minus_null_balanced_accuracy": float(ba - null_mean),
                            "skip_reason": "",
                        }
                    )
    return rows


def _summarize_binary(rows: list[dict[str, Any]], spring_meta: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    valid = frame[frame["skip_reason"].fillna("") == ""].copy()
    if valid.empty:
        return []
    meta_by_pair = {str(row["cell_pair"]): row for row in spring_meta if str(row["cell_pair"])}
    summaries: list[dict[str, Any]] = []
    for (pair, window, lag), group in valid.groupby(["cell_pair", "window", "lag"], sort=True):
        meta = meta_by_pair[str(pair)]
        summaries.append(
            {
                "cell_pair": str(pair),
                "spring_index": int(meta["spring_index"]),
                "spring_type": str(meta["spring_type"]),
                "cell_a": int(meta["cell_a"]),
                "cell_b": int(meta["cell_b"]),
                "window": str(window),
                "lag": int(lag),
                "heldout_trial_count": int(group["heldout_trial_id"].nunique()),
                "mean_balanced_accuracy": float(group["balanced_accuracy"].mean()),
                "mean_auc": float(group["auc"].mean()),
                "mean_mcc": float(group["mcc"].mean()),
                "mean_null_balanced_accuracy": float(group["null_balanced_accuracy_mean"].mean()),
                "mean_actual_minus_null_balanced_accuracy": float(
                    group["actual_minus_null_balanced_accuracy"].mean()
                ),
            }
        )
    return summaries


def _sliding_cell_regression(
    trials: list[dict[str, Any]],
    *,
    robot_shape: tuple[int, int],
    centers: list[int],
    half_width: int,
    lag: int,
    ridge_alpha: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cell_count = int(robot_shape[0]) * int(robot_shape[1])
    for center in centers:
        rel_start = int(center) - int(half_width)
        rel_stop = int(center) + int(half_width)
        for cell_index in range(cell_count):
            y_true_blocks: list[np.ndarray] = []
            y_pred_blocks: list[np.ndarray] = []
            valid_folds = 0
            for heldout in trials:
                train_x_blocks: list[np.ndarray] = []
                train_y_blocks: list[np.ndarray] = []
                for trial in trials:
                    if trial["trial_id"] == heldout["trial_id"]:
                        continue
                    x_block, y_block = _windowed_xy(
                        trial,
                        ("cell", cell_index),
                        lag=int(lag),
                        rel_start=rel_start,
                        rel_stop=rel_stop,
                        target="future_delta",
                    )
                    if y_block.size:
                        train_x_blocks.append(x_block)
                        train_y_blocks.append(y_block.astype(np.float32))
                test_x, test_y = _windowed_xy(
                    heldout,
                    ("cell", cell_index),
                    lag=int(lag),
                    rel_start=rel_start,
                    rel_stop=rel_stop,
                    target="future_delta",
                )
                if not train_y_blocks or test_y.size < 3:
                    continue
                train_x = np.concatenate(train_x_blocks, axis=0)
                train_y = np.concatenate(train_y_blocks, axis=0).astype(np.float32)
                if train_x.shape[0] <= 3 or float(np.std(train_y)) <= 1.0e-8:
                    continue
                pred = _ridge_predict(train_x, train_y, test_x, alpha=float(ridge_alpha))
                y_true_blocks.append(test_y.astype(np.float32))
                y_pred_blocks.append(pred.astype(np.float32))
                valid_folds += 1
            if y_true_blocks:
                y_true = np.concatenate(y_true_blocks, axis=0)
                y_pred = np.concatenate(y_pred_blocks, axis=0)
                rows.append(
                    {
                        "relative_center": int(center),
                        "relative_start": rel_start,
                        "relative_stop": rel_stop,
                        "lag": int(lag),
                        "cell_index": int(cell_index),
                        "valid_folds": int(valid_folds),
                        "sample_count": int(y_true.size),
                        "pearson": _pearson(y_true, y_pred),
                        "r2": _r2(y_true, y_pred),
                    }
                )
    return rows


def _plot_edge_network(
    *,
    summary_rows: list[dict[str, Any]],
    robot_shape: tuple[int, int],
    window: str,
    lag: int,
    metric: str,
    output_path: Path,
) -> None:
    frame = pd.DataFrame(summary_rows)
    subset = frame[(frame["window"] == str(window)) & (frame["lag"] == int(lag))]
    if "spring_type" in subset.columns:
        subset = subset[subset["spring_type"] != "group"]
    values = {str(row["cell_pair"]): float(row[metric]) for _, row in subset.iterrows()}
    finite = [v for v in values.values() if np.isfinite(v)]
    vmin = min(finite) if finite else 0.0
    vmax = max(finite) if finite else 1.0
    if abs(vmax - vmin) < 1.0e-9:
        vmax = vmin + 1.0

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    cmap = plt.get_cmap("viridis")
    for row in range(robot_shape[0]):
        for col in range(robot_shape[1]):
            cell = row * robot_shape[1] + col
            rect = plt.Rectangle((col - 0.45, row - 0.35), 0.9, 0.7, fill=False, edgecolor="0.75", linewidth=1.0)
            ax.add_patch(rect)
            ax.text(col, row, str(cell), ha="center", va="center", fontsize=8, color="0.2")
    for pair, value in values.items():
        a, b = [int(part) for part in pair.split("-")]
        ar, ac = divmod(a, robot_shape[1])
        br, bc = divmod(b, robot_shape[1])
        color = cmap((value - vmin) / (vmax - vmin))
        ax.plot([ac, bc], [ar, br], color=color, linewidth=2.0 + 5.0 * (value - vmin) / (vmax - vmin), solid_capstyle="round")
        ax.text((ac + bc) / 2.0, (ar + br) / 2.0 + 0.12, f"{value:.2f}", ha="center", va="center", fontsize=6, color="black")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.03, label=metric)
    ax.set_xlim(-0.75, robot_shape[1] - 0.25)
    ax.set_ylim(-0.65, robot_shape[0] - 0.35)
    ax.set_aspect("equal")
    ax.set_xticks(range(robot_shape[1]))
    ax.set_yticks(range(robot_shape[0]))
    ax.set_title(f"Inter-cell shared-spring {metric} | {window}, lag {lag}")
    ax.set_xlabel("cell column")
    ax.set_ylabel("physical row, bottom=0")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_nexus_tension(
    *,
    trials: list[dict[str, Any]],
    spring_meta: list[dict[str, Any]],
    spring_const: np.ndarray,
    output_path: Path,
    rel_start: int,
    rel_stop: int,
) -> list[dict[str, Any]]:
    meta_by_pair = {str(row["cell_pair"]): row for row in spring_meta if str(row["cell_pair"])}
    pairs = ["0-1", "0-7"]
    rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(len(pairs), 1, figsize=(10, 6), sharex=True)
    for ax, pair in zip(axes, pairs, strict=True):
        spring_index = int(meta_by_pair[pair]["spring_index"])
        for trial in trials:
            trace = trial["trace"]
            current = np.asarray(trace["spring_current_length"][:, spring_index], dtype=np.float32)
            rest = np.asarray(trace["spring_rest_length"][:, spring_index], dtype=np.float32)
            k = float(spring_const[spring_index])
            signed_force = (current - rest) * k
            rel = np.arange(len(signed_force), dtype=np.int32) - int(trial["event_index"])
            mask = (rel >= int(rel_start)) & (rel <= int(rel_stop))
            line_style = "-" if bool(trial["returned_from_edge"]) else "--"
            alpha = 0.85 if bool(trial["returned_from_edge"]) else 1.0
            ax.plot(rel[mask], signed_force[mask], line_style, linewidth=1.2, alpha=alpha, label=trial["trial_id"])
            event_mask = (rel >= -32) & (rel <= 96)
            rows.append(
                {
                    "trial_id": trial["trial_id"],
                    "cell_pair": pair,
                    "spring_index": spring_index,
                    "returned_from_edge": bool(trial["returned_from_edge"]),
                    "event_window_mean_signed_force": float(np.mean(signed_force[event_mask])),
                    "event_window_max_abs_signed_force": float(np.max(np.abs(signed_force[event_mask]))),
                    "pre_window_mean_signed_force": float(np.mean(signed_force[(rel >= -128) & (rel <= -1)])),
                    "post_window_mean_signed_force": float(np.mean(signed_force[(rel >= 0) & (rel <= 256)])),
                }
            )
        ax.axvline(0, color="black", linewidth=1.0)
        ax.axhline(0, color="0.7", linewidth=0.8)
        ax.set_title(f"Shared spring signed force proxy, cells {pair}")
        ax.set_ylabel("(current - rest) * k")
    axes[-1].set_xlabel("steps relative to first cliff reach")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return rows


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if int(window) <= 1 or values.size == 0:
        return values
    kernel = np.ones(int(window), dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def _plot_nexus_smoothed_load(
    *,
    trials: list[dict[str, Any]],
    spring_meta: list[dict[str, Any]],
    spring_const: np.ndarray,
    output_path: Path,
    rel_start: int,
    rel_stop: int,
    smooth_window: int,
) -> None:
    meta_by_pair = {str(row["cell_pair"]): row for row in spring_meta if str(row["cell_pair"])}
    pairs = ["0-1", "0-7"]
    fig, axes = plt.subplots(len(pairs), 1, figsize=(10, 6), sharex=True)
    for ax, pair in zip(axes, pairs, strict=True):
        spring_index = int(meta_by_pair[pair]["spring_index"])
        for trial in trials:
            trace = trial["trace"]
            current = np.asarray(trace["spring_current_length"][:, spring_index], dtype=np.float32)
            rest = np.asarray(trace["spring_rest_length"][:, spring_index], dtype=np.float32)
            k = float(spring_const[spring_index])
            signed_force = (current - rest) * k
            load = _rolling_mean(np.abs(signed_force), int(smooth_window))
            rel = np.arange(len(load), dtype=np.int32) - int(trial["event_index"])
            mask = (rel >= int(rel_start)) & (rel <= int(rel_stop))
            line_style = "-" if bool(trial["returned_from_edge"]) else "--"
            alpha = 0.85 if bool(trial["returned_from_edge"]) else 1.0
            ax.plot(rel[mask], load[mask], line_style, linewidth=1.4, alpha=alpha, label=trial["trial_id"])
        ax.axvline(0, color="black", linewidth=1.0)
        ax.set_title(f"Smoothed shared-spring load, cells {pair}")
        ax.set_ylabel(f"rolling mean |force| ({smooth_window})")
    axes[-1].set_xlabel("steps relative to first cliff reach")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_sliding_frame(
    *,
    frame_rows: pd.DataFrame,
    robot_shape: tuple[int, int],
    center: int,
    metric: str,
    vmin: float,
    vmax: float,
    timeline_start: int,
    timeline_stop: int,
    output_path: Path,
) -> None:
    grid = np.full(robot_shape, np.nan, dtype=np.float32)
    subset = frame_rows[frame_rows["relative_center"] == int(center)]
    for _, row in subset.iterrows():
        cell = int(row["cell_index"])
        rr, cc = divmod(cell, robot_shape[1])
        grid[rr, cc] = float(row[metric])
    fig, ax = plt.subplots(figsize=(9, 3.3))
    image = ax.imshow(grid, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
    for rr in range(robot_shape[0]):
        for cc in range(robot_shape[1]):
            cell = rr * robot_shape[1] + cc
            value = grid[rr, cc]
            text = f"{cell}\n{value:.2f}" if np.isfinite(value) else f"{cell}\nNA"
            ax.text(cc, rr, text, ha="center", va="center", color="white", fontsize=8)
    ax.axvline(-0.5, color="red", linewidth=2.0)
    ax.set_xticks(range(robot_shape[1]))
    ax.set_yticks(range(robot_shape[0]))
    ax.set_xlabel("cell column")
    ax.set_ylabel("physical row, bottom=0")
    ax.set_title(f"Future-displacement predictivity by cell | center {center:+d} steps")
    _add_event_timeline(ax, center=int(center), start=int(timeline_start), stop=int(timeline_stop))
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.30, top=0.86)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=metric)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def _add_event_timeline(ax: plt.Axes, *, center: int, start: int, stop: int) -> None:
    if int(stop) <= int(start):
        return
    event_x = (0.0 - float(start)) / (float(stop) - float(start))
    center_x = (float(center) - float(start)) / (float(stop) - float(start))
    event_x = float(np.clip(event_x, 0.0, 1.0))
    center_x = float(np.clip(center_x, 0.0, 1.0))
    y = -0.28
    ax.plot([0.0, 1.0], [y, y], transform=ax.transAxes, color="0.55", linewidth=1.1, clip_on=False)
    ax.plot(
        [event_x, event_x],
        [y - 0.05, y + 0.05],
        transform=ax.transAxes,
        color="red",
        linewidth=2.0,
        clip_on=False,
    )
    ax.scatter(
        [center_x],
        [y],
        transform=ax.transAxes,
        s=30,
        color="black",
        zorder=5,
        clip_on=False,
    )
    ax.text(
        event_x,
        y - 0.09,
        "edge event",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color="red",
        fontsize=7,
        clip_on=False,
    )
    ax.text(
        center_x,
        y + 0.07,
        f"{int(center):+d}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color="black",
        fontsize=7,
        clip_on=False,
    )


def _clear_frame_dir(frames_dir: Path) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in frames_dir.glob("frame_*.png"):
        stale_path.unlink()


def _make_animation(
    *,
    sliding_rows: list[dict[str, Any]],
    robot_shape: tuple[int, int],
    metric: str,
    output_dir: Path,
    gif_path: Path,
    duration_ms: int,
) -> None:
    frame = pd.DataFrame(sliding_rows)
    if frame.empty:
        return
    centers = sorted(int(value) for value in frame["relative_center"].dropna().unique())
    finite = frame[metric].replace([np.inf, -np.inf], np.nan).dropna()
    vmin = float(finite.quantile(0.05)) if not finite.empty else 0.0
    vmax = float(finite.quantile(0.95)) if not finite.empty else 1.0
    if metric == "pearson":
        bound = max(abs(vmin), abs(vmax), 0.1)
        vmin, vmax = -bound, bound
    if abs(vmax - vmin) < 1.0e-9:
        vmax = vmin + 1.0
    _clear_frame_dir(output_dir)
    frame_paths: list[Path] = []
    timeline_start = int(centers[0]) if centers else -256
    timeline_stop = int(centers[-1]) if centers else 256
    for center in centers:
        path = output_dir / f"frame_{center:+05d}.png"
        _plot_sliding_frame(
            frame_rows=frame,
            robot_shape=robot_shape,
            center=int(center),
            metric=metric,
            vmin=vmin,
            vmax=vmax,
            timeline_start=timeline_start,
            timeline_stop=timeline_stop,
            output_path=path,
        )
        frame_paths.append(path)
    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frame_paths]
    if images:
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=int(duration_ms),
            loop=0,
            optimize=False,
        )
        for image in images:
            image.close()


def analyze(
    *,
    trace_root: Path,
    output_dir: Path,
    lags: list[int],
    heatmap_lag: int,
    ridge_alpha: float,
    permutations: int,
    null_block_size: int,
    future_window: int,
    min_displacement: float,
    persistence_steps: int,
    persistence_fraction: float,
    animation_start: int,
    animation_stop: int,
    animation_step: int,
    animation_half_width: int,
    random_seed: int,
) -> dict[str, Any]:
    trials, trial_rows, robot_shape, spring_const, spring_meta = _load_trials(
        trace_root=trace_root,
        future_window=int(future_window),
        min_displacement=float(min_displacement),
        persistence_steps=int(persistence_steps),
        persistence_fraction=float(persistence_fraction),
    )
    intercell_pairs = sorted(
        str(row["cell_pair"]) for row in spring_meta if bool(row["is_intercell"]) and str(row["cell_pair"])
    )
    topology_rows = []
    for row in spring_meta:
        enriched = dict(row)
        idx = int(enriched["spring_index"])
        enriched["spring_const"] = float(spring_const[idx]) if idx >= 0 else float("nan")
        enriched.setdefault("member_cell_pairs", "")
        enriched.setdefault("member_spring_indices", "")
        topology_rows.append(enriched)

    metric_rows = _evaluate_intercell_binary(
        trials,
        cell_pairs=intercell_pairs,
        windows=WINDOWS,
        lags=lags,
        ridge_alpha=float(ridge_alpha),
        permutations=int(permutations),
        null_block_size=int(null_block_size),
        random_seed=int(random_seed),
    )
    summary_rows = _summarize_binary(metric_rows, topology_rows)

    centers = list(range(int(animation_start), int(animation_stop) + 1, int(animation_step)))
    sliding_rows = _sliding_cell_regression(
        trials,
        robot_shape=robot_shape,
        centers=centers,
        half_width=int(animation_half_width),
        lag=int(heatmap_lag),
        ridge_alpha=float(ridge_alpha),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "trial_metadata.csv", trial_rows)
    _write_csv(output_dir / "spring_topology.csv", topology_rows)
    _write_csv(output_dir / "intercell_decoder_metrics.csv", metric_rows)
    _write_csv(output_dir / "intercell_decoder_summary.csv", summary_rows)
    _write_csv(output_dir / "sliding_cell_future_delta_regression.csv", sliding_rows)
    nexus_rows = _plot_nexus_tension(
        trials=trials,
        spring_meta=topology_rows,
        spring_const=spring_const,
        output_path=output_dir / "nexus_signed_force_timeseries.png",
        rel_start=-256,
        rel_stop=512,
    )
    _write_csv(output_dir / "nexus_signed_force_summary.csv", nexus_rows)
    _plot_nexus_smoothed_load(
        trials=trials,
        spring_meta=topology_rows,
        spring_const=spring_const,
        output_path=output_dir / "nexus_smoothed_abs_force_timeseries.png",
        rel_start=-256,
        rel_stop=512,
        smooth_window=24,
    )
    _plot_edge_network(
        summary_rows=summary_rows,
        robot_shape=robot_shape,
        window="edge_impulse",
        lag=int(heatmap_lag),
        metric="mean_auc",
        output_path=output_dir / f"intercell_network_edge_impulse_lag{heatmap_lag}_auc.png",
    )
    _plot_edge_network(
        summary_rows=summary_rows,
        robot_shape=robot_shape,
        window="edge_impulse",
        lag=int(heatmap_lag),
        metric="mean_actual_minus_null_balanced_accuracy",
        output_path=output_dir / f"intercell_network_edge_impulse_lag{heatmap_lag}_actual_minus_null_ba.png",
    )
    _make_animation(
        sliding_rows=sliding_rows,
        robot_shape=robot_shape,
        metric="pearson",
        output_dir=output_dir / "cell_predictivity_future_delta_frames",
        gif_path=output_dir / f"cell_predictivity_future_delta_lag{heatmap_lag}.gif",
        duration_ms=140,
    )

    top_frame = pd.DataFrame(summary_rows)
    if summary_rows:
        edge_frame = top_frame.query("window == 'edge_impulse' and lag == @heatmap_lag").copy()
        top_edge = (
            edge_frame[edge_frame["spring_type"] != "group"]
            .sort_values("mean_auc", ascending=False)
            .head(8)
            .to_dict(orient="records")
        )
        top_groups = (
            edge_frame[edge_frame["spring_type"] == "group"]
            .sort_values("mean_auc", ascending=False)
            .to_dict(orient="records")
        )
    else:
        top_edge = []
        top_groups = []
    summary = {
        "schema_version": 1,
        "kind": "intercell_tension_network_analysis",
        "trace_root": str(trace_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "included_trials": [trial["trial_id"] for trial in trials],
        "robot_shape": list(robot_shape),
        "physical_cell_layout": ["7 8 9 10 11 12 13", "0 1 2 3 4 5 6"],
        "array_cell_layout": ["0 1 2 3 4 5 6", "7 8 9 10 11 12 13"],
        "force_proxy": "signed spring force = (current_length - actuated_rest_length) * spring_const; positive means extension/tension",
        "windows": {key: [int(v[0]), int(v[1])] for key, v in WINDOWS.items()},
        "lags": [int(value) for value in lags],
        "heatmap_lag": int(heatmap_lag),
        "sliding_animation": {
            "target": "goal-oriented future COM displacement",
            "metric": "leave-one-trial-out Pearson r",
            "relative_centers": [int(value) for value in centers],
            "half_width": int(animation_half_width),
            "lag": int(heatmap_lag),
        },
        "top_intercell_edge_impulse_auc": top_edge,
        "top_intercell_group_edge_impulse_auc": top_groups,
        "outputs": {
            "trial_metadata": str((output_dir / "trial_metadata.csv").resolve()),
            "spring_topology": str((output_dir / "spring_topology.csv").resolve()),
            "intercell_decoder_metrics": str((output_dir / "intercell_decoder_metrics.csv").resolve()),
            "intercell_decoder_summary": str((output_dir / "intercell_decoder_summary.csv").resolve()),
            "nexus_signed_force_summary": str((output_dir / "nexus_signed_force_summary.csv").resolve()),
            "nexus_signed_force_timeseries": str((output_dir / "nexus_signed_force_timeseries.png").resolve()),
            "nexus_smoothed_abs_force_timeseries": str((output_dir / "nexus_smoothed_abs_force_timeseries.png").resolve()),
            "intercell_network_auc": str((output_dir / f"intercell_network_edge_impulse_lag{heatmap_lag}_auc.png").resolve()),
            "intercell_network_actual_minus_null_ba": str(
                (output_dir / f"intercell_network_edge_impulse_lag{heatmap_lag}_actual_minus_null_ba.png").resolve()
            ),
            "sliding_cell_future_delta_regression": str((output_dir / "sliding_cell_future_delta_regression.csv").resolve()),
            "cell_predictivity_animation": str((output_dir / f"cell_predictivity_future_delta_lag{heatmap_lag}.gif").resolve()),
            "cell_predictivity_frames": str((output_dir / "cell_predictivity_future_delta_frames").resolve()),
            "summary": str((output_dir / "summary.json").resolve()),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lags", type=str, default="0,8,16")
    parser.add_argument("--heatmap-lag", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=1.0e-2)
    parser.add_argument("--permutations", type=int, default=25)
    parser.add_argument("--null-block-size", type=int, default=32)
    parser.add_argument("--future-window", type=int, default=96)
    parser.add_argument("--min-displacement", type=float, default=0.02)
    parser.add_argument("--persistence-steps", type=int, default=24)
    parser.add_argument("--persistence-fraction", type=float, default=0.75)
    parser.add_argument("--animation-start", type=int, default=-256)
    parser.add_argument("--animation-stop", type=int, default=256)
    parser.add_argument("--animation-step", type=int, default=16)
    parser.add_argument("--animation-half-width", type=int, default=48)
    parser.add_argument("--random-seed", type=int, default=20260608)
    args = parser.parse_args()

    lags = [int(part.strip()) for part in args.lags.split(",") if part.strip()]
    summary = analyze(
        trace_root=args.trace_root,
        output_dir=args.output_dir,
        lags=lags,
        heatmap_lag=int(args.heatmap_lag),
        ridge_alpha=float(args.ridge_alpha),
        permutations=int(args.permutations),
        null_block_size=int(args.null_block_size),
        future_window=int(args.future_window),
        min_displacement=float(args.min_displacement),
        persistence_steps=int(args.persistence_steps),
        persistence_fraction=float(args.persistence_fraction),
        animation_start=int(args.animation_start),
        animation_stop=int(args.animation_stop),
        animation_step=int(args.animation_step),
        animation_half_width=int(args.animation_half_width),
        random_seed=int(args.random_seed),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
