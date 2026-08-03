#!/usr/bin/env python
"""Leading-edge local-state reversal analysis for deformation-off traces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

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


DEFAULT_TRACE_ROOT = Path(
    "paper_data/preprint/native_core_ablation_v1/"
    "raw_modal_edge_state_space_deformation_off_20260608T041827/"
    "state_space_trace_edges_deformation_off"
)
DEFAULT_OUTPUT_DIR = Path(
    "paper_data/preprint/native_core_ablation_v1/figures/"
    "leading_edge_reversal_deformation_off"
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


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


def _trace_path_for_trial(trace_root: Path, rows: pd.DataFrame) -> Path:
    remote_path = Path(str(rows.iloc[0]["trace_npz"]))
    candidate = trace_root / "traces" / remote_path.name
    if candidate.exists():
        return candidate
    fallback = trace_root / "traces" / f"{rows.iloc[0]['trial_id']}.npz"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not locate trace NPZ for {rows.iloc[0]['trial_id']!r}")


def _flatten(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        return array.reshape((-1, 1))
    return array.reshape((array.shape[0], -1))


def _safe_scale(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float32)
    reference = np.where(np.abs(reference) > 1.0e-6, reference, 1.0).astype(np.float32)
    return np.asarray(values, dtype=np.float32) / reference - 1.0


def _nearest_edge_side(rows: pd.DataFrame, event_index: int) -> str:
    for column in ("nearest_cliff_side", "cliff_side"):
        if column in rows.columns:
            value = str(rows.iloc[int(event_index)][column]).strip().lower()
            if value in {"left", "right"}:
                return value
    goal_direction = int(rows.iloc[0]["goal_direction"])
    return "left" if goal_direction > 0 else "right"


def _leading_edge_indices(
    *,
    rows: pd.DataFrame,
    trace: dict[str, np.ndarray],
    event_index: int,
) -> dict[str, Any]:
    edge_side = _nearest_edge_side(rows, int(event_index))
    centers = np.asarray(trace["cell_corner_positions"][int(event_index)], dtype=np.float32).mean(axis=1)
    boxel_mask = np.asarray(trace["boxel_mask"], dtype=bool)
    valid_cells = np.flatnonzero(boxel_mask)
    valid_x = centers[valid_cells, 0]
    if edge_side == "left":
        target_x = float(np.min(valid_x))
        leading_cells = valid_cells[np.isclose(valid_x, target_x, atol=1.0e-4)]
        inward_sign = 1.0
    else:
        target_x = float(np.max(valid_x))
        leading_cells = valid_cells[np.isclose(valid_x, target_x, atol=1.0e-4)]
        inward_sign = -1.0
    if leading_cells.size == 0:
        chosen = int(valid_cells[int(np.argmin(valid_x) if edge_side == "left" else np.argmax(valid_x))])
        leading_cells = np.asarray([chosen], dtype=np.int32)
    boxel_corners = np.asarray(trace["boxel_corners"], dtype=np.int32)
    leading_points = np.unique(boxel_corners[leading_cells].reshape(-1)).astype(np.int32)
    a_idx = np.asarray(trace["spring_a_idx"], dtype=np.int32)
    b_idx = np.asarray(trace["spring_b_idx"], dtype=np.int32)
    a_in = np.isin(a_idx, leading_points)
    b_in = np.isin(b_idx, leading_points)
    attached = np.flatnonzero(a_in | b_in).astype(np.int32)
    internal = np.flatnonzero(a_in & b_in).astype(np.int32)
    connection = np.flatnonzero(a_in ^ b_in).astype(np.int32)
    return {
        "edge_side": edge_side,
        "inward_sign": float(inward_sign),
        "leading_cells": leading_cells.astype(np.int32),
        "leading_points": leading_points,
        "attached_springs": attached,
        "internal_springs": internal,
        "connection_springs": connection,
    }


def _local_geometry_features(
    *,
    trace: dict[str, np.ndarray],
    cells: np.ndarray,
    points: np.ndarray,
    inward_sign: float,
) -> np.ndarray:
    corners = np.asarray(trace["cell_corner_positions"][:, cells], dtype=np.float32)
    center = corners.mean(axis=(1, 2), keepdims=True)
    rel_corners = corners - center
    rel_corners[..., 0] *= float(inward_sign)
    point_pos = np.asarray(trace["plant_positions"][:, points], dtype=np.float32)
    point_center = point_pos.mean(axis=1, keepdims=True)
    rel_points = point_pos - point_center
    rel_points[..., 0] *= float(inward_sign)
    point_vel = np.asarray(trace["plant_velocities_true"][:, points], dtype=np.float32)
    point_vel = point_vel - point_vel.mean(axis=1, keepdims=True)
    point_vel[..., 0] *= float(inward_sign)
    return np.concatenate([_flatten(rel_corners), _flatten(rel_points), _flatten(point_vel)], axis=1)


def _spring_features(trace: dict[str, np.ndarray], spring_indices: np.ndarray) -> np.ndarray:
    if spring_indices.size == 0:
        return np.zeros((int(trace["spring_current_length"].shape[0]), 0), dtype=np.float32)
    init = np.asarray(trace["spring_init_rest_length"], dtype=np.float32)[spring_indices]
    current = _safe_scale(trace["spring_current_length"][:, spring_indices], init)
    rest = _safe_scale(trace["spring_rest_length"][:, spring_indices], init)
    pre_rest = _safe_scale(trace["spring_pre_rest_length"][:, spring_indices], init)
    phase_rest = _safe_scale(trace["phase_post_spring_rest_length"][:, :, spring_indices], init)
    return np.concatenate([current, rest, pre_rest, _flatten(phase_rest)], axis=1).astype(np.float32)


def _drive_features(trace: dict[str, np.ndarray], cells: np.ndarray) -> np.ndarray:
    if cells.size == 0:
        return np.zeros((int(trace["phase_h_drive"].shape[0]), 0), dtype=np.float32)
    h_drive = np.asarray(trace["phase_h_drive"][:, :, cells], dtype=np.float32)
    v_drive = np.asarray(trace["phase_v_drive"][:, :, cells], dtype=np.float32)
    return np.concatenate([_flatten(h_drive), _flatten(v_drive)], axis=1).astype(np.float32)


def _standardize_all(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale > 1.0e-6, scale, 1.0).astype(np.float32)
    return ((features - mean) / scale).astype(np.float32)


def _trial_features(
    *,
    trace_root: Path,
    rows: pd.DataFrame,
    event_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    trace_path = _trace_path_for_trial(trace_root, rows)
    loaded = np.load(trace_path)
    try:
        trace = {key: loaded[key] for key in loaded.files}
    finally:
        loaded.close()

    edge = _leading_edge_indices(rows=rows, trace=trace, event_index=int(event_index))
    connection = _spring_features(trace, edge["connection_springs"])
    attached = _spring_features(trace, edge["attached_springs"])
    internal = _spring_features(trace, edge["internal_springs"])
    drives = _drive_features(trace, edge["leading_cells"])
    geometry = _local_geometry_features(
        trace=trace,
        cells=edge["leading_cells"],
        points=edge["leading_points"],
        inward_sign=float(edge["inward_sign"]),
    )
    feature_streams = {
        "connection_springs": connection,
        "attached_springs": attached,
        "leading_voxel_state": np.concatenate([attached, drives, geometry], axis=1).astype(np.float32),
    }
    meta = {
        "edge_side": edge["edge_side"],
        "leading_cell_count": int(edge["leading_cells"].size),
        "leading_cells": [int(value) for value in edge["leading_cells"]],
        "leading_point_count": int(edge["leading_points"].size),
        "connection_spring_count": int(edge["connection_springs"].size),
        "attached_spring_count": int(edge["attached_springs"].size),
        "internal_spring_count": int(edge["internal_springs"].size),
        "trace_npz": str(trace_path),
    }
    return feature_streams, meta


def _load_trials(
    *,
    trace_root: Path,
    future_window: int,
    min_displacement: float,
    persistence_steps: int,
    persistence_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    step_manifest = pd.read_csv(trace_root / "step_manifest.csv")
    trial_manifest = pd.read_csv(trace_root / "trial_manifest.csv")
    true_reach = trial_manifest[
        trial_manifest["first_cliff_reach_step"].notna()
        & (trial_manifest["goal_direction"] != 0)
    ].copy()
    true_reach = true_reach.sort_values(["surface_id", "goal_distance", "cliff_distance"], kind="stable")
    trials: list[dict[str, Any]] = []
    trial_meta: list[dict[str, Any]] = []
    for _, trial_row in true_reach.iterrows():
        trial_id = str(trial_row["trial_id"])
        rows = step_manifest[step_manifest["trial_id"] == trial_id].sort_values("step_index", kind="stable").reset_index(drop=True)
        event_index = int(trial_row["first_cliff_reach_step"])
        labels, future_delta = _committed_labels(
            rows,
            future_window=int(future_window),
            min_displacement=float(min_displacement),
            persistence_steps=int(persistence_steps),
            persistence_fraction=float(persistence_fraction),
        )
        streams, meta = _trial_features(trace_root=trace_root, rows=rows, event_index=event_index)
        step_count = min([len(rows), len(labels), *(stream.shape[0] for stream in streams.values())])
        rows = rows.iloc[:step_count].reset_index(drop=True)
        streams = {key: value[:step_count] for key, value in streams.items()}
        trials.append(
            {
                "trial_id": trial_id,
                "rows": rows,
                "labels": labels[:step_count],
                "future_delta": future_delta[:step_count],
                "event_index": int(event_index),
                "returned_from_edge": bool(trial_row["returned_from_edge"]),
                "success": bool(trial_row["success"]),
                "surface_id": str(trial_row["surface_id"]),
                "feature_streams": streams,
            }
        )
        trial_meta.append(
            {
                **meta,
                "trial_id": trial_id,
                "surface_id": str(trial_row["surface_id"]),
                "distance_pair": str(trial_row.get("distance_pair", "")),
                "returned_from_edge": bool(trial_row["returned_from_edge"]),
                "success": bool(trial_row["success"]),
                "event_index": int(event_index),
                "step_count": int(step_count),
                "committed_toward_count": int(np.sum(labels[:step_count] == 1)),
                "committed_away_count": int(np.sum(labels[:step_count] == -1)),
            }
        )
    return trials, trial_meta


def _windowed_xy(
    trial: dict[str, Any],
    stream: str,
    *,
    lag: int,
    rel_start: int,
    rel_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(trial["feature_streams"][stream], dtype=np.float32)
    labels = np.asarray(trial["labels"], dtype=np.int8)
    usable = features.shape[0] - int(lag)
    if usable <= 0:
        return np.zeros((0, features.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int8), np.zeros((0,), dtype=np.int32)
    indices = np.arange(usable, dtype=np.int32)
    rel = indices - int(trial["event_index"])
    y = labels[indices + int(lag)]
    mask = (rel >= int(rel_start)) & (rel <= int(rel_stop)) & (y != 0)
    mask &= np.all(np.isfinite(features[indices]), axis=1)
    return features[indices][mask], y[mask].astype(np.int8), indices[mask].astype(np.int32)


def _shuffle_blocks(rng: np.random.Generator, values: np.ndarray, block_size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int8)
    if int(block_size) <= 1:
        return rng.permutation(values).astype(np.int8)
    chunks = [values[index : index + int(block_size)] for index in range(0, len(values), int(block_size))]
    order = rng.permutation(len(chunks))
    return np.concatenate([chunks[int(item)] for item in order]).astype(np.int8)


def _evaluate(
    trials: list[dict[str, Any]],
    *,
    streams: list[str],
    windows: dict[str, tuple[int, int]],
    lags: list[int],
    ridge_alpha: float,
    permutations: int,
    null_block_size: int,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(_stable_seed(int(random_seed), "leading-edge-null"))
    for window_name, (rel_start, rel_stop) in windows.items():
        for stream in streams:
            for lag in lags:
                for heldout in trials:
                    train_x_blocks: list[np.ndarray] = []
                    train_y_blocks: list[np.ndarray] = []
                    for trial in trials:
                        if trial["trial_id"] == heldout["trial_id"]:
                            continue
                        x_block, y_block, _ = _windowed_xy(
                            trial,
                            stream,
                            lag=int(lag),
                            rel_start=int(rel_start),
                            rel_stop=int(rel_stop),
                        )
                        if y_block.size:
                            train_x_blocks.append(x_block)
                            train_y_blocks.append(y_block)
                    test_x, test_y, test_indices = _windowed_xy(
                        heldout,
                        stream,
                        lag=int(lag),
                        rel_start=int(rel_start),
                        rel_stop=int(rel_stop),
                    )
                    base = {
                        "window": window_name,
                        "relative_start": int(rel_start),
                        "relative_stop": int(rel_stop),
                        "stream": stream,
                        "lag": int(lag),
                        "heldout_trial_id": heldout["trial_id"],
                        "sample_count": int(test_y.size),
                        "toward_count": int(np.sum(test_y == 1)),
                        "away_count": int(np.sum(test_y == -1)),
                        "surface_id": heldout["surface_id"],
                        "returned_from_edge": bool(heldout["returned_from_edge"]),
                        "success": bool(heldout["success"]),
                    }
                    if not train_y_blocks or test_y.size == 0:
                        metric_rows.append({**base, "skip_reason": "empty_train_or_test"})
                        continue
                    train_x = np.concatenate(train_x_blocks, axis=0)
                    train_y = np.concatenate(train_y_blocks, axis=0)
                    if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
                        metric_rows.append({**base, "skip_reason": "single_class_train_or_test"})
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
                    metric_rows.append(
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
                    if int(lag) == 0:
                        rows_by_step = heldout["rows"].set_index("step_index", drop=False)
                        for index, y_value, score in zip(test_indices, test_y, scores, strict=True):
                            source = rows_by_step.loc[int(index)]
                            prediction_rows.append(
                                {
                                    "window": window_name,
                                    "stream": stream,
                                    "trial_id": heldout["trial_id"],
                                    "step_index": int(index),
                                    "relative_to_event": int(index) - int(heldout["event_index"]),
                                    "env_step": int(source["env_step"]),
                                    "cliff_clearance": float(source["cliff_clearance"]),
                                    "label": int(y_value),
                                    "score": float(score),
                                }
                            )
    return metric_rows, prediction_rows


def _aggregate(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(metrics)
    if frame.empty or "balanced_accuracy" not in frame.columns:
        return []
    valid = frame[(frame.get("skip_reason", "") == "") & np.isfinite(frame["balanced_accuracy"])]
    rows: list[dict[str, Any]] = []
    for (window, stream, lag), group in valid.groupby(["window", "stream", "lag"], sort=True):
        rows.append(
            {
                "window": str(window),
                "stream": str(stream),
                "lag": int(lag),
                "heldout_trial_count": int(group["heldout_trial_id"].nunique()),
                "mean_balanced_accuracy": float(group["balanced_accuracy"].mean()),
                "mean_auc": float(group["auc"].mean()),
                "mean_mcc": float(group["mcc"].mean()),
                "mean_null_balanced_accuracy": float(group["null_balanced_accuracy_mean"].mean()),
                "mean_actual_minus_null_balanced_accuracy": float(group["actual_minus_null_balanced_accuracy"].mean()),
            }
        )
    return rows


def _outcome_descriptives(
    trials: list[dict[str, Any]],
    *,
    streams: list[str],
    rel_start: int,
    rel_stop: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stream in streams:
        summaries = []
        trial_ids = []
        returned = []
        for trial in trials:
            event = int(trial["event_index"])
            start = max(0, event + int(rel_start))
            stop = min(trial["feature_streams"][stream].shape[0], event + int(rel_stop) + 1)
            if stop <= start:
                continue
            features = _standardize_all(trial["feature_streams"][stream][start:stop])
            summary = np.concatenate(
                [
                    features.mean(axis=0),
                    features.std(axis=0),
                    np.quantile(features, 0.10, axis=0),
                    np.quantile(features, 0.90, axis=0),
                ]
            ).astype(np.float32)
            summaries.append(summary)
            trial_ids.append(trial["trial_id"])
            returned.append(bool(trial["returned_from_edge"]))
        if not summaries:
            continue
        matrix = np.vstack(summaries).astype(np.float32)
        return_mask = np.asarray(returned, dtype=bool)
        if np.sum(return_mask) < 2:
            continue
        return_mean = matrix[return_mask].mean(axis=0)
        return_scale = matrix[return_mask].std(axis=0)
        return_scale = np.where(return_scale > 1.0e-6, return_scale, 1.0)
        for trial_id, is_returned, vector in zip(trial_ids, returned, matrix, strict=True):
            z = (vector - return_mean) / return_scale
            rows.append(
                {
                    "stream": stream,
                    "trial_id": trial_id,
                    "returned_from_edge": bool(is_returned),
                    "relative_start": int(rel_start),
                    "relative_stop": int(rel_stop),
                    "distance_to_return_profile": float(np.sqrt(np.mean(z * z))),
                }
            )
    return rows


def analyze_leading_edge(
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
) -> dict[str, Any]:
    trace_root = trace_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _ = _read_json(trace_root / "manifest.json")
    trials, trial_meta = _load_trials(
        trace_root=trace_root,
        future_window=int(future_window),
        min_displacement=float(min_displacement),
        persistence_steps=int(persistence_steps),
        persistence_fraction=float(persistence_fraction),
    )
    streams = ["connection_springs", "attached_springs", "leading_voxel_state"]
    windows = {
        "pre_edge": (-256, -1),
        "edge_to_return": (0, 512),
        "edge_impulse": (-32, 96),
    }
    metrics, predictions = _evaluate(
        trials,
        streams=streams,
        windows=windows,
        lags=[int(value) for value in lags],
        ridge_alpha=float(ridge_alpha),
        permutations=int(permutations),
        null_block_size=int(null_block_size),
        random_seed=int(random_seed),
    )
    aggregate = _aggregate(metrics)
    outcome_rows = _outcome_descriptives(
        trials,
        streams=streams,
        rel_start=-64,
        rel_stop=256,
    )
    _write_csv(output_dir / "leading_edge_trial_metadata.csv", trial_meta)
    _write_csv(output_dir / "leading_edge_decoder_metrics.csv", metrics)
    _write_csv(output_dir / "leading_edge_decoder_summary.csv", aggregate)
    _write_csv(output_dir / "leading_edge_predictions_lag0.csv", predictions)
    _write_csv(output_dir / "leading_edge_outcome_descriptives.csv", outcome_rows)
    summary = {
        "schema_version": 1,
        "kind": "leading_edge_reversal_analysis",
        "trace_root": str(trace_root),
        "output_dir": str(output_dir),
        "included_trials": [trial["trial_id"] for trial in trials],
        "feature_policy": {
            "leading_edge_cells": "cell column whose event-time center is closest to the cliff frontier",
            "connection_springs": "springs with exactly one endpoint on a leading-edge cell corner point",
            "attached_springs": "springs with at least one endpoint on a leading-edge cell corner point",
            "leading_voxel_state": "attached spring state plus leading-cell h/v drives plus local corner/point geometry and velocities; no absolute COM or cliff clearance included",
        },
        "label_policy": {
            "axis": "goal-facing future COM displacement",
            "future_window": int(future_window),
            "min_displacement": float(min_displacement),
            "persistence_steps": int(persistence_steps),
            "persistence_fraction": float(persistence_fraction),
        },
        "windows": {key: list(value) for key, value in windows.items()},
        "lags": [int(value) for value in lags],
        "trial_metadata": trial_meta,
        "aggregate": aggregate,
        "outputs": {
            "trial_metadata": str(output_dir / "leading_edge_trial_metadata.csv"),
            "decoder_metrics": str(output_dir / "leading_edge_decoder_metrics.csv"),
            "decoder_summary": str(output_dir / "leading_edge_decoder_summary.csv"),
            "predictions_lag0": str(output_dir / "leading_edge_predictions_lag0.csv"),
            "outcome_descriptives": str(output_dir / "leading_edge_outcome_descriptives.csv"),
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
        description="Test whether leading-edge springs/local state predict reversal after cliff reach."
    )
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--future-window", type=int, default=32)
    parser.add_argument("--min-displacement", type=float, default=0.025)
    parser.add_argument("--persistence-steps", type=int, default=12)
    parser.add_argument("--persistence-fraction", type=float, default=0.75)
    parser.add_argument("--lags", type=_parse_lags, default=_parse_lags("0,8,16,32,64"))
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--permutations", type=int, default=50)
    parser.add_argument("--null-block-size", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=137)
    args = parser.parse_args()
    summary = analyze_leading_edge(
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
    )
    print(json.dumps(summary["outputs"], indent=2))


if __name__ == "__main__":
    main()
