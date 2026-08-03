#!/usr/bin/env python
"""Cell-wise signalling predictivity and trace-only susceptibility probes."""

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

from committed_direction_decoding import _committed_labels, _stable_seed
from leading_edge_reversal_analysis import DEFAULT_TRACE_ROOT, _leading_edge_indices, _trace_path_for_trial
from voxel_reversal_heatmap_analysis import WINDOWS, _cell_features


DEFAULT_OUTPUT_DIR = Path(
    "paper_data/preprint/native_core_ablation_v1/figures/"
    "signal_susceptibility_deformation_off"
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


def _flatten(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    return array.reshape((array.shape[0], -1))


def _standardize_train_test(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale > 1.0e-6, scale, 1.0).astype(np.float32)
    return ((train - mean) / scale).astype(np.float32), ((test - mean) / scale).astype(np.float32)


def _standardize_all(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale > 1.0e-6, scale, 1.0).astype(np.float32)
    return ((values - mean) / scale).astype(np.float32)


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    train_z, test_z = _standardize_train_test(train_x, test_x)
    y_mean = train_y.mean(axis=0, dtype=np.float64).astype(np.float32)
    y_scale = train_y.std(axis=0, dtype=np.float64).astype(np.float32)
    y_scale = np.where(y_scale > 1.0e-6, y_scale, 1.0).astype(np.float32)
    train_y_z = ((train_y - y_mean) / y_scale).astype(np.float32)
    train_design = np.column_stack([np.ones(train_z.shape[0], dtype=np.float32), train_z])
    test_design = np.column_stack([np.ones(test_z.shape[0], dtype=np.float32), test_z])
    penalty = np.eye(train_design.shape[1], dtype=np.float32) * float(alpha)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(
        train_design.T @ train_design + penalty,
        train_design.T @ train_y_z,
    )
    pred_z = (test_design @ weights).astype(np.float32)
    return (pred_z * y_scale + y_mean).astype(np.float32)


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(np.sum(mask)) < 3:
        return float("nan")
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if float(np.std(y_true)) <= 1.0e-8 or float(np.std(y_pred)) <= 1.0e-8:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    denom = float(np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2))
    if denom <= 1.0e-8:
        return float("nan")
    return float(1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom)


def _signal_cell_features(
    trace: dict[str, np.ndarray],
    *,
    cell_index: int,
    robot_shape: tuple[int, int],
    hidden_start_channel: int,
) -> np.ndarray:
    row, col = divmod(int(cell_index), int(robot_shape[1]))
    nca = np.asarray(trace["phase_nca_state"][:, :, row, col, :], dtype=np.float32)
    hidden = nca[:, :, int(hidden_start_channel):] if nca.shape[-1] > int(hidden_start_channel) else nca[:, :, 0:0]
    ctrnn = np.asarray(trace["phase_ctrnn_carry"][:, :, row, col, :], dtype=np.float32)
    return np.concatenate([_flatten(hidden), _flatten(ctrnn)], axis=1).astype(np.float32)


def _drive_global_features(trace: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            _flatten(np.asarray(trace["phase_h_drive"], dtype=np.float32)),
            _flatten(np.asarray(trace["phase_v_drive"], dtype=np.float32)),
        ],
        axis=1,
    ).astype(np.float32)


def _stream_window_xy(
    trial: dict[str, Any],
    *,
    stream: str,
    cell_index: int,
    lag: int,
    rel_start: int,
    rel_stop: int,
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(trial[f"{stream}_cell_features"][int(cell_index)], dtype=np.float32)
    y_values = np.asarray(trial[target], dtype=np.float32)
    usable = features.shape[0] - int(lag)
    if usable <= 0:
        return np.zeros((0, features.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    indices = np.arange(usable, dtype=np.int32)
    rel = indices - int(trial["event_index"])
    y = y_values[indices + int(lag)]
    mask = (rel >= int(rel_start)) & (rel <= int(rel_stop))
    mask &= np.isfinite(y)
    mask &= np.all(np.isfinite(features[indices]), axis=1)
    return features[indices][mask], y[mask].astype(np.float32)


def _global_window_xy(
    trial: dict[str, Any],
    *,
    x_stream: str,
    y_stream: str,
    lag: int,
    rel_start: int,
    rel_stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(trial[f"{x_stream}_global"], dtype=np.float32)
    y = np.asarray(trial[f"{y_stream}_global"], dtype=np.float32)
    usable = min(x.shape[0], y.shape[0] - int(lag))
    if usable <= 0:
        return np.zeros((0, x.shape[1]), dtype=np.float32), np.zeros((0, y.shape[1]), dtype=np.float32)
    indices = np.arange(usable, dtype=np.int32)
    rel = indices - int(trial["event_index"])
    mask = (rel >= int(rel_start)) & (rel <= int(rel_stop))
    mask &= np.all(np.isfinite(x[indices]), axis=1)
    mask &= np.all(np.isfinite(y[indices + int(lag)]), axis=1)
    return x[indices][mask], y[indices + int(lag)][mask]


def _load_trials(
    *,
    trace_root: Path,
    future_window: int,
    min_displacement: float,
    persistence_steps: int,
    persistence_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[int, int], dict[str, Any]]:
    with (trace_root / "manifest.json").open() as handle:
        manifest = json.load(handle)
    hidden_start = int((manifest.get("nca_channel_schema") or {}).get("hidden_start_channel", 6))
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
        cell_count = int(robot_shape[0]) * int(robot_shape[1])
        signal_cell_features = {
            cell: _signal_cell_features(
                trace,
                cell_index=cell,
                robot_shape=robot_shape,
                hidden_start_channel=int(hidden_start),
            )
            for cell in range(cell_count)
        }
        body_cell_features = {
            cell: _cell_features(
                trace=trace,
                cell_index=cell,
                inward_sign=float(edge["inward_sign"]),
            )
            for cell in range(cell_count)
        }
        signal_global = np.concatenate([signal_cell_features[cell] for cell in range(cell_count)], axis=1).astype(np.float32)
        body_global = np.concatenate([body_cell_features[cell] for cell in range(cell_count)], axis=1).astype(np.float32)
        drive_global = _drive_global_features(trace)
        step_count = min(
            [
                len(rows),
                len(labels),
                signal_global.shape[0],
                body_global.shape[0],
                drive_global.shape[0],
            ]
        )
        trials.append(
            {
                "trial_id": trial_id,
                "surface_id": str(trial_row["surface_id"]),
                "rows": rows.iloc[:step_count].reset_index(drop=True),
                "labels": labels[:step_count],
                "future_delta": future_delta[:step_count],
                "event_index": int(event_index),
                "returned_from_edge": bool(trial_row["returned_from_edge"]),
                "success": bool(trial_row["success"]),
                "edge_side": str(edge["edge_side"]),
                "signal_cell_features": {key: value[:step_count] for key, value in signal_cell_features.items()},
                "body_cell_features": {key: value[:step_count] for key, value in body_cell_features.items()},
                "signal_global": signal_global[:step_count],
                "body_global": body_global[:step_count],
                "drive_global": drive_global[:step_count],
            }
        )
        trial_rows.append(
            {
                "trial_id": trial_id,
                "surface_id": str(trial_row["surface_id"]),
                "event_index": int(event_index),
                "edge_side": str(edge["edge_side"]),
                "returned_from_edge": bool(trial_row["returned_from_edge"]),
                "success": bool(trial_row["success"]),
                "step_count": int(step_count),
            }
        )

    feature_manifest = {
        "hidden_start_channel": int(hidden_start),
        "source_nca_channel_schema": dict(manifest.get("nca_channel_schema") or {}),
        "signal_feature_policy": "per-cell phase_nca_state hidden channels plus phase_ctrnn_carry only; deformation inputs, action outputs, drives, body geometry, COM, and cliff geometry excluded",
        "body_feature_policy": "per-cell attached spring state, h/v drives, local corner/point geometry and velocities; absolute COM/cliff clearance excluded",
        "drive_feature_policy": "phase_h_drive plus phase_v_drive only",
        "signal_global_feature_count": int(trials[0]["signal_global"].shape[1]) if trials else 0,
        "body_global_feature_count": int(trials[0]["body_global"].shape[1]) if trials else 0,
        "drive_global_feature_count": int(trials[0]["drive_global"].shape[1]) if trials else 0,
    }
    return trials, trial_rows, robot_shape, feature_manifest


def _sliding_cell_regression(
    trials: list[dict[str, Any]],
    *,
    stream: str,
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
                    x_block, y_block = _stream_window_xy(
                        trial,
                        stream=stream,
                        cell_index=int(cell_index),
                        lag=int(lag),
                        rel_start=rel_start,
                        rel_stop=rel_stop,
                        target="future_delta",
                    )
                    if y_block.size:
                        train_x_blocks.append(x_block)
                        train_y_blocks.append(y_block.reshape((-1, 1)))
                test_x, test_y = _stream_window_xy(
                    heldout,
                    stream=stream,
                    cell_index=int(cell_index),
                    lag=int(lag),
                    rel_start=rel_start,
                    rel_stop=rel_stop,
                    target="future_delta",
                )
                if not train_y_blocks or test_y.size < 3:
                    continue
                train_x = np.concatenate(train_x_blocks, axis=0)
                train_y = np.concatenate(train_y_blocks, axis=0)
                if train_x.shape[0] <= 3 or float(np.std(train_y)) <= 1.0e-8:
                    continue
                pred = _ridge_predict(train_x, train_y, test_x, alpha=float(ridge_alpha)).reshape(-1)
                y_true_blocks.append(test_y.astype(np.float32))
                y_pred_blocks.append(pred.astype(np.float32))
                valid_folds += 1
            if y_true_blocks:
                y_true = np.concatenate(y_true_blocks, axis=0)
                y_pred = np.concatenate(y_pred_blocks, axis=0)
                rows.append(
                    {
                        "stream": stream,
                        "relative_center": int(center),
                        "relative_start": int(rel_start),
                        "relative_stop": int(rel_stop),
                        "lag": int(lag),
                        "cell_index": int(cell_index),
                        "valid_folds": int(valid_folds),
                        "sample_count": int(y_true.size),
                        "pearson": _pearson(y_true, y_pred),
                        "r2": _r2(y_true.reshape((-1, 1)), y_pred.reshape((-1, 1))),
                    }
                )
    return rows


def _coupling_summary(
    trials: list[dict[str, Any]],
    *,
    windows: dict[str, tuple[int, int]],
    ridge_alpha: float,
    coupling_lag: int,
) -> list[dict[str, Any]]:
    pairs = [
        ("signal_to_drive_same_step", "signal", "drive", 0),
        ("signal_to_future_body", "signal", "body", int(coupling_lag)),
        ("drive_to_future_body", "drive", "body", int(coupling_lag)),
    ]
    rows: list[dict[str, Any]] = []
    for window_name, (rel_start, rel_stop) in windows.items():
        for name, x_stream, y_stream, lag in pairs:
            y_true_blocks: list[np.ndarray] = []
            y_pred_blocks: list[np.ndarray] = []
            valid_folds = 0
            for heldout in trials:
                train_x_blocks: list[np.ndarray] = []
                train_y_blocks: list[np.ndarray] = []
                for trial in trials:
                    if trial["trial_id"] == heldout["trial_id"]:
                        continue
                    x_block, y_block = _global_window_xy(
                        trial,
                        x_stream=x_stream,
                        y_stream=y_stream,
                        lag=int(lag),
                        rel_start=int(rel_start),
                        rel_stop=int(rel_stop),
                    )
                    if x_block.shape[0]:
                        train_x_blocks.append(x_block)
                        train_y_blocks.append(y_block)
                test_x, test_y = _global_window_xy(
                    heldout,
                    x_stream=x_stream,
                    y_stream=y_stream,
                    lag=int(lag),
                    rel_start=int(rel_start),
                    rel_stop=int(rel_stop),
                )
                if not train_x_blocks or test_x.shape[0] < 3:
                    continue
                train_x = np.concatenate(train_x_blocks, axis=0)
                train_y = np.concatenate(train_y_blocks, axis=0)
                if train_x.shape[0] <= 3:
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
                        "coupling": name,
                        "x_stream": x_stream,
                        "y_stream": y_stream,
                        "window": window_name,
                        "relative_start": int(rel_start),
                        "relative_stop": int(rel_stop),
                        "lag": int(lag),
                        "valid_folds": int(valid_folds),
                        "sample_count": int(y_true.shape[0]),
                        "target_feature_count": int(y_true.shape[1]),
                        "flattened_pearson": _pearson(y_true, y_pred),
                        "global_r2": _r2(y_true, y_pred),
                    }
                )
    return rows


def _window_matrix(trials: list[dict[str, Any]], *, stream: str, rel_start: int, rel_stop: int) -> list[np.ndarray]:
    blocks: list[np.ndarray] = []
    for trial in trials:
        features = np.asarray(trial[f"{stream}_global"], dtype=np.float32)
        indices = np.arange(features.shape[0], dtype=np.int32)
        rel = indices - int(trial["event_index"])
        mask = (rel >= int(rel_start)) & (rel <= int(rel_stop))
        mask &= np.all(np.isfinite(features), axis=1)
        if np.any(mask):
            blocks.append(features[mask])
    return blocks


def _lag1_autocorr(blocks: list[np.ndarray]) -> float:
    values: list[float] = []
    for block in blocks:
        if block.shape[0] < 3:
            continue
        z = _standardize_all(block)
        a = z[:-1].reshape(-1)
        b = z[1:].reshape(-1)
        corr = _pearson(a, b)
        if np.isfinite(corr):
            values.append(corr)
    return float(np.mean(values)) if values else float("nan")


def _effective_rank(block: np.ndarray, *, max_samples: int, random_seed: int) -> tuple[float, float, int]:
    if block.shape[0] < 3:
        return float("nan"), float("nan"), int(block.shape[0])
    if block.shape[0] > int(max_samples):
        rng = np.random.default_rng(int(random_seed))
        indices = np.sort(rng.choice(block.shape[0], size=int(max_samples), replace=False))
        block = block[indices]
    z = _standardize_all(block)
    singular = np.linalg.svd(z, compute_uv=False)
    eig = (singular ** 2) / max(1, z.shape[0] - 1)
    denom = float(np.sum(eig ** 2))
    rank = float((float(np.sum(eig)) ** 2) / denom) if denom > 1.0e-12 else float("nan")
    top3 = float(np.sum(eig[:3]) / np.sum(eig)) if float(np.sum(eig)) > 1.0e-12 else float("nan")
    return rank, top3, int(z.shape[0])


def _criticality_window_summary(
    trials: list[dict[str, Any]],
    *,
    windows: dict[str, tuple[int, int]],
    max_svd_samples: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window_name, (rel_start, rel_stop) in windows.items():
        for stream in ("signal", "drive", "body"):
            blocks = _window_matrix(trials, stream=stream, rel_start=int(rel_start), rel_stop=int(rel_stop))
            if not blocks:
                continue
            block = np.concatenate(blocks, axis=0)
            rank, top3, svd_samples = _effective_rank(
                block,
                max_samples=int(max_svd_samples),
                random_seed=_stable_seed(int(random_seed), f"{stream}-{window_name}-rank"),
            )
            rows.append(
                {
                    "stream": stream,
                    "window": window_name,
                    "relative_start": int(rel_start),
                    "relative_stop": int(rel_stop),
                    "sample_count": int(block.shape[0]),
                    "feature_count": int(block.shape[1]),
                    "mean_feature_variance": float(np.mean(np.var(block, axis=0))),
                    "median_feature_variance": float(np.median(np.var(block, axis=0))),
                    "lag1_autocorr": _lag1_autocorr(blocks),
                    "effective_pca_rank": rank,
                    "top3_variance_fraction": top3,
                    "svd_sample_count": int(svd_samples),
                }
            )
    return rows


def _nearest_neighbor_susceptibility(
    trials: list[dict[str, Any]],
    *,
    windows: dict[str, tuple[int, int]],
    future_offset: int,
    max_samples: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window_name, (rel_start, rel_stop) in windows.items():
        for stream in ("signal", "drive", "body"):
            x_blocks: list[np.ndarray] = []
            y_blocks: list[np.ndarray] = []
            trial_blocks: list[np.ndarray] = []
            for trial_index, trial in enumerate(trials):
                x = np.asarray(trial[f"{stream}_global"], dtype=np.float32)
                body = np.asarray(trial["body_global"], dtype=np.float32)
                usable = min(x.shape[0], body.shape[0] - int(future_offset))
                if usable <= 0:
                    continue
                indices = np.arange(usable, dtype=np.int32)
                rel = indices - int(trial["event_index"])
                mask = (rel >= int(rel_start)) & (rel <= int(rel_stop))
                mask &= np.all(np.isfinite(x[indices]), axis=1)
                mask &= np.all(np.isfinite(body[indices + int(future_offset)]), axis=1)
                if np.any(mask):
                    x_blocks.append(x[indices][mask])
                    y_blocks.append(body[indices + int(future_offset)][mask])
                    trial_blocks.append(np.full(int(np.sum(mask)), int(trial_index), dtype=np.int32))
            if not x_blocks:
                continue
            x_all = np.concatenate(x_blocks, axis=0)
            y_all = np.concatenate(y_blocks, axis=0)
            trial_ids = np.concatenate(trial_blocks, axis=0)
            if x_all.shape[0] > int(max_samples):
                rng = np.random.default_rng(_stable_seed(int(random_seed), f"{stream}-{window_name}-nn"))
                selected = np.sort(rng.choice(x_all.shape[0], size=int(max_samples), replace=False))
                x_all = x_all[selected]
                y_all = y_all[selected]
                trial_ids = trial_ids[selected]
            x_z = _standardize_all(x_all)
            y_z = _standardize_all(y_all)
            current_dist: list[float] = []
            future_dist: list[float] = []
            ratios: list[float] = []
            for start in range(0, x_z.shape[0], 128):
                stop = min(x_z.shape[0], start + 128)
                diff = x_z[start:stop, None, :] - x_z[None, :, :]
                dist = np.sqrt(np.mean(diff * diff, axis=2))
                same_trial = trial_ids[start:stop, None] == trial_ids[None, :]
                dist[same_trial] = np.inf
                nearest = np.argmin(dist, axis=1)
                nearest_dist = dist[np.arange(stop - start), nearest]
                valid = np.isfinite(nearest_dist) & (nearest_dist > 1.0e-8)
                if not np.any(valid):
                    continue
                ydiff = y_z[start:stop][valid] - y_z[nearest[valid]]
                ydist = np.sqrt(np.mean(ydiff * ydiff, axis=1))
                current_dist.extend(float(value) for value in nearest_dist[valid])
                future_dist.extend(float(value) for value in ydist)
                ratios.extend(float(a / b) for a, b in zip(ydist, nearest_dist[valid], strict=True))
            if ratios:
                ratio_arr = np.asarray(ratios, dtype=np.float32)
                rows.append(
                    {
                        "stream": stream,
                        "window": window_name,
                        "relative_start": int(rel_start),
                        "relative_stop": int(rel_stop),
                        "future_body_offset": int(future_offset),
                        "sample_count": int(x_z.shape[0]),
                        "median_current_distance": float(np.median(np.asarray(current_dist, dtype=np.float32))),
                        "median_future_body_distance": float(np.median(np.asarray(future_dist, dtype=np.float32))),
                        "median_amplification_ratio": float(np.median(ratio_arr)),
                        "p90_amplification_ratio": float(np.quantile(ratio_arr, 0.9)),
                        "mean_amplification_ratio": float(np.mean(ratio_arr)),
                    }
                )
    return rows


def _plot_grid_frame(
    *,
    rows: pd.DataFrame,
    stream: str,
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
    subset = rows[(rows["stream"] == stream) & (rows["relative_center"] == int(center))]
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
    ax.set_title(f"{stream} future-displacement predictivity | center {center:+d}")
    _add_event_timeline(ax, center=int(center), start=int(timeline_start), stop=int(timeline_stop))
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.30, top=0.86)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=metric)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def _plot_comparison_frame(
    *,
    rows: pd.DataFrame,
    robot_shape: tuple[int, int],
    center: int,
    metric: str,
    vmin: float,
    vmax: float,
    timeline_start: int,
    timeline_stop: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
    image = None
    for ax, stream in zip(axes, ("signal", "body"), strict=True):
        grid = np.full(robot_shape, np.nan, dtype=np.float32)
        subset = rows[(rows["stream"] == stream) & (rows["relative_center"] == int(center))]
        for _, row in subset.iterrows():
            cell = int(row["cell_index"])
            rr, cc = divmod(cell, robot_shape[1])
            grid[rr, cc] = float(row[metric])
        image = ax.imshow(grid, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
        for rr in range(robot_shape[0]):
            for cc in range(robot_shape[1]):
                cell = rr * robot_shape[1] + cc
                value = grid[rr, cc]
                text = f"{cell}\n{value:.2f}" if np.isfinite(value) else f"{cell}\nNA"
                ax.text(cc, rr, text, ha="center", va="center", color="white", fontsize=8)
        ax.axvline(-0.5, color="red", linewidth=2.0)
        ax.set_title(stream)
        ax.set_xticks(range(robot_shape[1]))
        ax.set_yticks(range(robot_shape[0]))
        ax.set_xlabel("cell column")
        ax.set_ylabel("physical row, bottom=0")
        _add_event_timeline(ax, center=int(center), start=int(timeline_start), stop=int(timeline_stop))
    fig.suptitle(f"Future-displacement predictivity | center {center:+d} steps")
    fig.subplots_adjust(left=0.05, right=0.88, bottom=0.28, top=0.78, wspace=0.12)
    if image is not None:
        fig.colorbar(image, ax=axes, fraction=0.046, pad=0.04, label=metric)
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


def _gif_from_frames(frame_paths: list[Path], gif_path: Path, *, duration_ms: int) -> None:
    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frame_paths]
    if not images:
        return
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


def _make_animations(
    *,
    rows: list[dict[str, Any]],
    robot_shape: tuple[int, int],
    output_dir: Path,
    metric: str,
    heatmap_lag: int,
    duration_ms: int,
) -> dict[str, str]:
    frame = pd.DataFrame(rows)
    centers = sorted(int(value) for value in frame["relative_center"].dropna().unique())
    finite = frame[metric].replace([np.inf, -np.inf], np.nan).dropna()
    bound = max(abs(float(finite.quantile(0.05))), abs(float(finite.quantile(0.95))), 0.1) if not finite.empty else 1.0
    vmin, vmax = -bound, bound
    outputs: dict[str, str] = {}
    timeline_start = int(centers[0]) if centers else -256
    timeline_stop = int(centers[-1]) if centers else 256
    for stream in ("signal", "body"):
        frames_dir = output_dir / f"{stream}_cell_predictivity_future_delta_frames"
        _clear_frame_dir(frames_dir)
        frame_paths: list[Path] = []
        for center in centers:
            path = frames_dir / f"frame_{center:+05d}.png"
            _plot_grid_frame(
                rows=frame,
                stream=stream,
                robot_shape=robot_shape,
                center=center,
                metric=metric,
                vmin=vmin,
                vmax=vmax,
                timeline_start=timeline_start,
                timeline_stop=timeline_stop,
                output_path=path,
            )
            frame_paths.append(path)
        gif_path = output_dir / f"{stream}_cell_predictivity_future_delta_lag{heatmap_lag}.gif"
        _gif_from_frames(frame_paths, gif_path, duration_ms=int(duration_ms))
        outputs[f"{stream}_gif"] = str(gif_path.resolve())
        outputs[f"{stream}_frames"] = str(frames_dir.resolve())

    compare_dir = output_dir / "signal_vs_body_predictivity_future_delta_frames"
    _clear_frame_dir(compare_dir)
    compare_paths: list[Path] = []
    for center in centers:
        path = compare_dir / f"frame_{center:+05d}.png"
        _plot_comparison_frame(
            rows=frame,
            robot_shape=robot_shape,
            center=center,
            metric=metric,
            vmin=vmin,
            vmax=vmax,
            timeline_start=timeline_start,
            timeline_stop=timeline_stop,
            output_path=path,
        )
        compare_paths.append(path)
    compare_gif = output_dir / f"signal_vs_body_predictivity_future_delta_lag{heatmap_lag}.gif"
    _gif_from_frames(compare_paths, compare_gif, duration_ms=int(duration_ms))
    outputs["signal_vs_body_gif"] = str(compare_gif.resolve())
    outputs["signal_vs_body_frames"] = str(compare_dir.resolve())
    return outputs


def analyze(
    *,
    trace_root: Path,
    output_dir: Path,
    heatmap_lag: int,
    ridge_alpha: float,
    future_window: int,
    min_displacement: float,
    persistence_steps: int,
    persistence_fraction: float,
    animation_start: int,
    animation_stop: int,
    animation_step: int,
    animation_half_width: int,
    coupling_lag: int,
    nearest_future_offset: int,
    max_svd_samples: int,
    max_nn_samples: int,
    random_seed: int,
) -> dict[str, Any]:
    trials, trial_rows, robot_shape, feature_manifest = _load_trials(
        trace_root=trace_root,
        future_window=int(future_window),
        min_displacement=float(min_displacement),
        persistence_steps=int(persistence_steps),
        persistence_fraction=float(persistence_fraction),
    )
    centers = list(range(int(animation_start), int(animation_stop) + 1, int(animation_step)))
    sliding_rows: list[dict[str, Any]] = []
    for stream in ("signal", "body"):
        sliding_rows.extend(
            _sliding_cell_regression(
                trials,
                stream=stream,
                robot_shape=robot_shape,
                centers=centers,
                half_width=int(animation_half_width),
                lag=int(heatmap_lag),
                ridge_alpha=float(ridge_alpha),
            )
        )
    coupling_rows = _coupling_summary(
        trials,
        windows=WINDOWS,
        ridge_alpha=float(ridge_alpha),
        coupling_lag=int(coupling_lag),
    )
    critical_rows = _criticality_window_summary(
        trials,
        windows=WINDOWS,
        max_svd_samples=int(max_svd_samples),
        random_seed=int(random_seed),
    )
    nearest_rows = _nearest_neighbor_susceptibility(
        trials,
        windows=WINDOWS,
        future_offset=int(nearest_future_offset),
        max_samples=int(max_nn_samples),
        random_seed=int(random_seed),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "trial_metadata.csv", trial_rows)
    _write_csv(output_dir / "sliding_signal_body_future_delta_regression.csv", sliding_rows)
    _write_csv(output_dir / "signal_body_coupling_summary.csv", coupling_rows)
    _write_csv(output_dir / "criticality_window_summary.csv", critical_rows)
    _write_csv(output_dir / "nearest_neighbor_susceptibility.csv", nearest_rows)
    animation_outputs = _make_animations(
        rows=sliding_rows,
        robot_shape=robot_shape,
        output_dir=output_dir,
        metric="pearson",
        heatmap_lag=int(heatmap_lag),
        duration_ms=140,
    )

    sliding_frame = pd.DataFrame(sliding_rows)
    signal_top = (
        sliding_frame[sliding_frame["stream"] == "signal"]
        .sort_values("pearson", ascending=False)
        .head(12)
        .to_dict(orient="records")
        if sliding_rows
        else []
    )
    body_top = (
        sliding_frame[sliding_frame["stream"] == "body"]
        .sort_values("pearson", ascending=False)
        .head(12)
        .to_dict(orient="records")
        if sliding_rows
        else []
    )
    summary = {
        "schema_version": 1,
        "kind": "signal_susceptibility_analysis",
        "trace_root": str(trace_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "included_trials": [trial["trial_id"] for trial in trials],
        "robot_shape": list(robot_shape),
        "physical_cell_layout": ["7 8 9 10 11 12 13", "0 1 2 3 4 5 6"],
        "feature_manifest": feature_manifest,
        "sliding_animation": {
            "target": "goal-oriented future COM displacement",
            "metric": "leave-one-trial-out Pearson r",
            "relative_centers": [int(value) for value in centers],
            "half_width": int(animation_half_width),
            "lag": int(heatmap_lag),
            "shared_color_scale": True,
        },
        "susceptibility_policy": "trace-only observational proxies; not causal proof of criticality",
        "top_signal_cell_predictivity": signal_top,
        "top_body_cell_predictivity": body_top,
        "outputs": {
            "trial_metadata": str((output_dir / "trial_metadata.csv").resolve()),
            "sliding_signal_body_future_delta_regression": str((output_dir / "sliding_signal_body_future_delta_regression.csv").resolve()),
            "signal_body_coupling_summary": str((output_dir / "signal_body_coupling_summary.csv").resolve()),
            "criticality_window_summary": str((output_dir / "criticality_window_summary.csv").resolve()),
            "nearest_neighbor_susceptibility": str((output_dir / "nearest_neighbor_susceptibility.csv").resolve()),
            "summary": str((output_dir / "summary.json").resolve()),
            **animation_outputs,
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--heatmap-lag", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=1.0e-2)
    parser.add_argument("--future-window", type=int, default=96)
    parser.add_argument("--min-displacement", type=float, default=0.02)
    parser.add_argument("--persistence-steps", type=int, default=24)
    parser.add_argument("--persistence-fraction", type=float, default=0.75)
    parser.add_argument("--animation-start", type=int, default=-256)
    parser.add_argument("--animation-stop", type=int, default=256)
    parser.add_argument("--animation-step", type=int, default=16)
    parser.add_argument("--animation-half-width", type=int, default=48)
    parser.add_argument("--coupling-lag", type=int, default=8)
    parser.add_argument("--nearest-future-offset", type=int, default=96)
    parser.add_argument("--max-svd-samples", type=int, default=1800)
    parser.add_argument("--max-nn-samples", type=int, default=1200)
    parser.add_argument("--random-seed", type=int, default=20260610)
    args = parser.parse_args()

    summary = analyze(
        trace_root=args.trace_root,
        output_dir=args.output_dir,
        heatmap_lag=int(args.heatmap_lag),
        ridge_alpha=float(args.ridge_alpha),
        future_window=int(args.future_window),
        min_displacement=float(args.min_displacement),
        persistence_steps=int(args.persistence_steps),
        persistence_fraction=float(args.persistence_fraction),
        animation_start=int(args.animation_start),
        animation_stop=int(args.animation_stop),
        animation_step=int(args.animation_step),
        animation_half_width=int(args.animation_half_width),
        coupling_lag=int(args.coupling_lag),
        nearest_future_offset=int(args.nearest_future_offset),
        max_svd_samples=int(args.max_svd_samples),
        max_nn_samples=int(args.max_nn_samples),
        random_seed=int(args.random_seed),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
