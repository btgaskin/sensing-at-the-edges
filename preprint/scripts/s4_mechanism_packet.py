#!/usr/bin/env python
"""Build a compact S4 mechanism and eight-specimen sanity packet.

This script is intentionally analysis-only. It reads existing deterministic
state-space traces and assay manifests, then writes paper-facing CSV summaries,
simple charts, and a markdown note for the current eight-specimen revision.
"""

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
    _leading_edge_indices,
    _shuffle_blocks,
    _trace_path_for_trial,
)
from voxel_reversal_heatmap_analysis import _cell_features


DATA_ROOT = Path("paper_data/preprint/native_core_ablation_v1")
S4_SPECIMEN = "bin_03_rank_00_cell_6-0-3_flat_0423"
INTACT_NATIVE_TRACE_ROOT = (
    DATA_ROOT
    / "raw_modal_state_space_20260607T145100"
    / "preprint_native_state_space_v1"
    / "paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed137_e4fcc2fb"
    / S4_SPECIMEN
    / "state_space_trace"
)
DEFORMATION_OFF_EDGE_TRACE_ROOT = (
    DATA_ROOT
    / "raw_modal_edge_state_space_deformation_off_20260608T041827"
    / "state_space_trace_edges_deformation_off"
)
DEFAULT_OUTPUT_DIR = DATA_ROOT / "figures" / "s4_mechanism_packet"

WINDOWS = {
    "pre_edge": (-256, -1),
    "edge_impulse": (-32, 96),
    "early_ramification": (24, 160),
    "late_ramification": (160, 512),
    "edge_to_return": (0, 512),
}
WINDOW_LABELS = {
    "pre_edge": "pre-edge",
    "edge_impulse": "edge impulse",
    "early_ramification": "early",
    "late_ramification": "late",
    "edge_to_return": "edge-to-return",
}
STREAM_COLORS = {"body": "#8CB87C", "signal": "#3B5B92"}
BAND_COLORS = {"frontier": "#3B5B92", "middle": "#8CB87C", "rear": "#D4736C"}
OUTCOME_COLORS = {"success": "#8CB87C", "fall": "#D4736C", "stall": "#AAAAAA"}


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


def _bool_series(series: pd.Series) -> pd.Series:
    def coerce(value: Any) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return int(value) != 0
        text = str(value).strip().lower()
        return text in {"true", "1", "yes", "y", "survived", "success"}

    return series.map(coerce).astype(bool)


def _sem(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def _rate(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return float("nan")
    return float(_bool_series(frame[column]).mean())


def _mean_or_nan(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _load_trace_frames(trace_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_manifest = pd.read_csv(trace_root / "trial_manifest.csv")
    step_manifest = pd.read_csv(trace_root / "step_manifest.csv", low_memory=False)
    return trial_manifest, step_manifest


def _event_index_from_rows(rows: pd.DataFrame, trial_row: pd.Series | None = None) -> tuple[int, bool, str]:
    if trial_row is not None and "first_cliff_reach_step" in trial_row.index and pd.notna(trial_row["first_cliff_reach_step"]):
        event_step = int(trial_row["first_cliff_reach_step"])
        matches = np.flatnonzero(rows["step_index"].to_numpy(dtype=np.int64) == event_step)
        if matches.size:
            return int(matches[0]), True, "true_cliff_frontier"
    clearances = rows["cliff_clearance"].to_numpy(dtype=np.float32)
    hits = np.flatnonzero(clearances <= 0.0)
    if hits.size:
        return int(hits[0]), True, "true_cliff_frontier"
    return int(np.argmin(clearances)), False, "closest_approach"


def _edge_timing_for_trial(
    rows: pd.DataFrame,
    trial_row: pd.Series,
    *,
    condition: str,
    future_window: int,
    min_displacement: float,
    persistence_steps: int,
    persistence_fraction: float,
) -> dict[str, Any]:
    rows = rows.sort_values("step_index", kind="stable").reset_index(drop=True)
    labels, future_delta = _committed_labels(
        rows,
        future_window=int(future_window),
        min_displacement=float(min_displacement),
        persistence_steps=int(persistence_steps),
        persistence_fraction=float(persistence_fraction),
    )
    event_i, true_event, event_class = _event_index_from_rows(rows, trial_row)
    clearances = rows["cliff_clearance"].to_numpy(dtype=np.float32)
    step_index = rows["step_index"].to_numpy(dtype=np.int64)
    com_x = rows["com_x"].to_numpy(dtype=np.float32)
    goal_direction = int(rows.iloc[0].get("goal_direction", trial_row.get("goal_direction", 0)))
    oriented_com = com_x * float(goal_direction if goal_direction != 0 else 1)
    post = np.arange(event_i, len(rows), dtype=np.int64)
    post_labels = labels[post] if post.size else np.asarray([], dtype=np.int8)
    nonzero_labels = post_labels[post_labels != 0]

    goal_commit = np.flatnonzero(post_labels == 1)
    away_commit = np.flatnonzero(post_labels == -1)
    goal_commit_i = int(post[int(goal_commit[0])]) if goal_commit.size else None
    away_commit_i = int(post[int(away_commit[0])]) if away_commit.size else None
    commit_stop = goal_commit_i if goal_commit_i is not None else len(rows) - 1
    event_to_goal_commit = (
        int(step_index[goal_commit_i] - step_index[event_i]) if goal_commit_i is not None else None
    )
    event_to_away_commit = (
        int(step_index[away_commit_i] - step_index[event_i]) if away_commit_i is not None else None
    )
    if nonzero_labels.size > 1:
        direction_changes = int(np.sum(nonzero_labels[1:] != nonzero_labels[:-1]))
    else:
        direction_changes = 0

    if true_event and goal_commit_i is not None:
        overhang_before_goal = float(np.maximum(0.0, -clearances[event_i : commit_stop + 1]).max())
    else:
        overhang_before_goal = float("nan")
    overhang_after_event = (
        float(np.maximum(0.0, -clearances[event_i:]).max()) if true_event else float("nan")
    )
    returned_after_event = (
        bool(np.any(clearances[event_i:] > 0.0)) if true_event else False
    )
    if "returned_from_edge" in trial_row.index and pd.notna(trial_row["returned_from_edge"]):
        returned_after_event = bool(trial_row["returned_from_edge"])

    return {
        "condition": condition,
        "trial_id": str(trial_row["trial_id"]),
        "surface_id": str(trial_row.get("surface_id", "native_core")),
        "distance_pair": str(trial_row.get("distance_pair", "")),
        "side": str(trial_row.get("side", "")),
        "goal_direction": int(goal_direction),
        "success": bool(trial_row.get("success", False)),
        "event_class": event_class,
        "true_cliff_event": bool(true_event),
        "event_step": int(step_index[event_i]),
        "event_clearance": float(clearances[event_i]),
        "min_cliff_clearance": float(np.nanmin(clearances)),
        "max_cliff_overhang_any": float(np.maximum(0.0, -clearances).max()),
        "max_cliff_overhang_after_event": overhang_after_event,
        "max_cliff_overhang_before_goal_commit": overhang_before_goal,
        "event_to_goal_commit_steps": event_to_goal_commit,
        "event_to_away_commit_steps": event_to_away_commit,
        "com_displacement_after_event_goal_frame": float(oriented_com[-1] - oriented_com[event_i]),
        "future_delta_at_event_goal_frame": float(future_delta[event_i]) if event_i < len(future_delta) else float("nan"),
        "direction_changes_after_event": direction_changes,
        "returned_after_event": returned_after_event,
        "committed_toward_count_after_event": int(np.sum(post_labels == 1)),
        "committed_away_count_after_event": int(np.sum(post_labels == -1)),
        "uncommitted_count_after_event": int(np.sum(post_labels == 0)),
        "steps_after_event": int(step_index[-1] - step_index[event_i]),
        "step_count": int(len(rows)),
    }


def build_s4_edge_timing(
    *,
    output_dir: Path,
    future_window: int,
    min_displacement: float,
    persistence_steps: int,
    persistence_fraction: float,
) -> dict[str, Any]:
    rows_out: list[dict[str, Any]] = []
    for condition, trace_root in [
        ("intact", INTACT_NATIVE_TRACE_ROOT),
        ("deformation_off", DEFORMATION_OFF_EDGE_TRACE_ROOT),
    ]:
        trial_manifest, step_manifest = _load_trace_frames(trace_root)
        if "surface_id" in trial_manifest.columns:
            trial_manifest = trial_manifest[trial_manifest["surface_id"] == "native_core"].copy()
        trial_manifest = trial_manifest.sort_values(["goal_distance", "cliff_distance", "side"], kind="stable")
        for _, trial_row in trial_manifest.iterrows():
            trial_id = str(trial_row["trial_id"])
            rows = step_manifest[step_manifest["trial_id"] == trial_id]
            if rows.empty:
                continue
            rows_out.append(
                _edge_timing_for_trial(
                    rows,
                    trial_row,
                    condition=condition,
                    future_window=int(future_window),
                    min_displacement=float(min_displacement),
                    persistence_steps=int(persistence_steps),
                    persistence_fraction=float(persistence_fraction),
                )
            )
    _write_csv(output_dir / "s4_edge_timing_episode_metrics.csv", rows_out)
    frame = pd.DataFrame(rows_out)
    summary_rows: list[dict[str, Any]] = []
    if not frame.empty:
        for condition, group in frame.groupby("condition", sort=False):
            true = group[group["true_cliff_event"]]
            summary_rows.append(
                {
                    "condition": condition,
                    "episode_count": int(len(group)),
                    "true_cliff_event_count": int(len(true)),
                    "closest_approach_count": int(len(group) - len(true)),
                    "success_rate": float(group["success"].mean()),
                    "mean_event_to_goal_commit_steps_true_events": float(true["event_to_goal_commit_steps"].mean()),
                    "min_event_to_goal_commit_steps_true_events": float(true["event_to_goal_commit_steps"].min()),
                    "max_event_to_goal_commit_steps_true_events": float(true["event_to_goal_commit_steps"].max()),
                    "mean_max_overhang_before_goal_commit_true_events": float(
                        true["max_cliff_overhang_before_goal_commit"].mean()
                    ),
                    "max_overhang_before_goal_commit_true_events": float(
                        true["max_cliff_overhang_before_goal_commit"].max()
                    ),
                    "mean_direction_changes_after_event_true_events": float(
                        true["direction_changes_after_event"].mean()
                    ),
                    "mean_com_displacement_after_event_goal_frame_true_events": float(
                        true["com_displacement_after_event_goal_frame"].mean()
                    ),
                }
            )
    _write_csv(output_dir / "s4_edge_timing_summary.csv", summary_rows)

    paired_rows: list[dict[str, Any]] = []
    if not frame.empty:
        true = frame[frame["true_cliff_event"]].copy()
        # S4's true edge encounters can occur on opposite mirrored sides across
        # intact and deformation-off replay. Pair by distance pair, then retain
        # the side labels so this does not look like same-side pairing.
        for distance_pair, group in true.groupby("distance_pair", sort=True):
            if set(group["condition"]) >= {"intact", "deformation_off"}:
                intact = group[group["condition"] == "intact"].iloc[0]
                off = group[group["condition"] == "deformation_off"].iloc[0]
                paired_rows.append(
                    {
                        "distance_pair": str(distance_pair),
                        "intact_side": str(intact["side"]),
                        "deformation_off_side": str(off["side"]),
                        "event_to_goal_commit_steps_deformation_off_minus_intact": float(
                            off["event_to_goal_commit_steps"] - intact["event_to_goal_commit_steps"]
                        ),
                        "max_overhang_before_goal_commit_deformation_off_minus_intact": float(
                            off["max_cliff_overhang_before_goal_commit"]
                            - intact["max_cliff_overhang_before_goal_commit"]
                        ),
                        "direction_changes_after_event_deformation_off_minus_intact": float(
                            off["direction_changes_after_event"] - intact["direction_changes_after_event"]
                        ),
                        "com_displacement_after_event_deformation_off_minus_intact": float(
                            off["com_displacement_after_event_goal_frame"]
                            - intact["com_displacement_after_event_goal_frame"]
                        ),
                    }
                )
    _write_csv(output_dir / "s4_edge_timing_paired_differences.csv", paired_rows)
    return {"episode_rows": rows_out, "summary_rows": summary_rows, "paired_rows": paired_rows}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path)
    try:
        return {key: loaded[key] for key in loaded.files}
    finally:
        loaded.close()


def _signal_cell_features(
    trace: dict[str, np.ndarray],
    *,
    cell_index: int,
    hidden_start_channel: int,
    robot_shape: tuple[int, int],
) -> np.ndarray:
    rows, cols = robot_shape
    row = int(cell_index) // int(cols)
    col = int(cell_index) % int(cols)
    nca_state = np.asarray(trace["phase_nca_state"][:, :, row, col, int(hidden_start_channel) :], dtype=np.float32)
    ctrnn_carry = np.asarray(trace["phase_ctrnn_carry"][:, :, row, col, :], dtype=np.float32)
    return np.concatenate(
        [nca_state.reshape(nca_state.shape[0], -1), ctrnn_carry.reshape(ctrnn_carry.shape[0], -1)],
        axis=1,
    ).astype(np.float32)


def _band_cells(
    *,
    trace: dict[str, np.ndarray],
    edge: dict[str, Any],
    event_index: int,
) -> dict[str, list[int]]:
    centers = np.asarray(trace["cell_corner_positions"][int(event_index)], dtype=np.float32).mean(axis=1)
    valid_cells = np.flatnonzero(np.asarray(trace["boxel_mask"], dtype=bool))
    xs = centers[valid_cells, 0]
    rounded_x = np.round(xs, 4)
    ordered = sorted(np.unique(rounded_x).tolist(), reverse=str(edge["edge_side"]) == "right")
    ranks: dict[int, int] = {}
    for cell in valid_cells:
        value = float(np.round(centers[int(cell), 0], 4))
        nearest = int(np.argmin([abs(value - float(candidate)) for candidate in ordered]))
        ranks[int(cell)] = nearest
    max_rank = max(ranks.values()) if ranks else 0
    frontier = [cell for cell, rank in ranks.items() if rank <= 1]
    rear = [cell for cell, rank in ranks.items() if rank >= max_rank - 1]
    middle = [cell for cell, rank in ranks.items() if cell not in frontier and cell not in rear]
    if not middle:
        ordered_cells = [cell for cell, _ in sorted(ranks.items(), key=lambda item: item[1])]
        split = max(1, len(ordered_cells) // 3)
        frontier = ordered_cells[:split]
        middle = ordered_cells[split : len(ordered_cells) - split]
        rear = ordered_cells[len(ordered_cells) - split :]
    return {
        "frontier": sorted(int(cell) for cell in frontier),
        "middle": sorted(int(cell) for cell in middle),
        "rear": sorted(int(cell) for cell in rear),
    }


def _load_band_trials(
    *,
    trace_root: Path,
    future_window: int,
    min_displacement: float,
    persistence_steps: int,
    persistence_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[int, int]]:
    manifest = _read_json(trace_root / "manifest.json")
    hidden_start = int((manifest.get("nca_channel_schema") or {}).get("hidden_start_channel", 6))
    trial_manifest, step_manifest = _load_trace_frames(trace_root)
    trial_manifest = trial_manifest[
        (trial_manifest["surface_id"] == "native_core")
        & trial_manifest["first_cliff_reach_step"].notna()
        & (trial_manifest["goal_direction"] != 0)
    ].copy()
    trial_manifest = trial_manifest.sort_values(["goal_distance", "cliff_distance", "side"], kind="stable")
    with np.load(trace_root / "static_arrays.npz") as static:
        robot_shape = tuple(int(v) for v in static["robot_morph"].shape)

    trials: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    for _, trial_row in trial_manifest.iterrows():
        trial_id = str(trial_row["trial_id"])
        rows = step_manifest[step_manifest["trial_id"] == trial_id].sort_values("step_index", kind="stable").reset_index(drop=True)
        if rows.empty:
            continue
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
        body_by_cell = {
            cell: _cell_features(trace=trace, cell_index=cell, inward_sign=float(edge["inward_sign"]))
            for cell in range(cell_count)
        }
        signal_by_cell = {
            cell: _signal_cell_features(
                trace,
                cell_index=cell,
                hidden_start_channel=hidden_start,
                robot_shape=robot_shape,
            )
            for cell in range(cell_count)
        }
        step_count = min(
            [len(rows), len(labels), *(feature.shape[0] for feature in body_by_cell.values())]
        )
        rows = rows.iloc[:step_count].reset_index(drop=True)
        bands = _band_cells(trace=trace, edge=edge, event_index=event_index)
        stream_features: dict[str, dict[str, np.ndarray]] = {"body": {}, "signal": {}}
        for band, cells in bands.items():
            stream_features["body"][band] = np.concatenate(
                [body_by_cell[cell][:step_count] for cell in cells],
                axis=1,
            ).astype(np.float32)
            stream_features["signal"][band] = np.concatenate(
                [signal_by_cell[cell][:step_count] for cell in cells],
                axis=1,
            ).astype(np.float32)
        trials.append(
            {
                "trial_id": trial_id,
                "rows": rows,
                "labels": labels[:step_count],
                "future_delta": future_delta[:step_count],
                "event_index": event_index,
                "surface_id": str(trial_row["surface_id"]),
                "distance_pair": str(trial_row["distance_pair"]),
                "side": str(trial_row["side"]),
                "success": bool(trial_row["success"]),
                "returned_from_edge": bool(trial_row["returned_from_edge"]),
                "bands": bands,
                "features": stream_features,
            }
        )
        meta_rows.append(
            {
                "trial_id": trial_id,
                "distance_pair": str(trial_row["distance_pair"]),
                "side": str(trial_row["side"]),
                "event_index": int(event_index),
                "edge_side": str(edge["edge_side"]),
                "frontier_cells": " ".join(str(cell) for cell in bands["frontier"]),
                "middle_cells": " ".join(str(cell) for cell in bands["middle"]),
                "rear_cells": " ".join(str(cell) for cell in bands["rear"]),
                "event_leading_cells": " ".join(str(int(cell)) for cell in edge["leading_cells"]),
                "returned_from_edge": bool(trial_row["returned_from_edge"]),
                "success": bool(trial_row["success"]),
                "committed_toward_count": int(np.sum(labels[:step_count] == 1)),
                "committed_away_count": int(np.sum(labels[:step_count] == -1)),
            }
        )
    return trials, meta_rows, robot_shape


def _windowed_band_xy(
    trial: dict[str, Any],
    *,
    stream: str,
    band: str,
    lag: int,
    rel_start: int,
    rel_stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(trial["features"][stream][band], dtype=np.float32)
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


def _evaluate_band_decoders(
    trials: list[dict[str, Any]],
    *,
    lags: list[int],
    ridge_alpha: float,
    permutations: int,
    null_block_size: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(_stable_seed(int(random_seed), "s4-band-predictivity"))
    for window_name, (rel_start, rel_stop) in WINDOWS.items():
        for stream in ("body", "signal"):
            for band in ("frontier", "middle", "rear"):
                for lag in lags:
                    for heldout in trials:
                        train_x_blocks: list[np.ndarray] = []
                        train_y_blocks: list[np.ndarray] = []
                        for trial in trials:
                            if trial["trial_id"] == heldout["trial_id"]:
                                continue
                            x_block, y_block = _windowed_band_xy(
                                trial,
                                stream=stream,
                                band=band,
                                lag=int(lag),
                                rel_start=int(rel_start),
                                rel_stop=int(rel_stop),
                            )
                            if y_block.size:
                                train_x_blocks.append(x_block)
                                train_y_blocks.append(y_block)
                        test_x, test_y = _windowed_band_xy(
                            heldout,
                            stream=stream,
                            band=band,
                            lag=int(lag),
                            rel_start=int(rel_start),
                            rel_stop=int(rel_stop),
                        )
                        base = {
                            "window": window_name,
                            "relative_start": int(rel_start),
                            "relative_stop": int(rel_stop),
                            "stream": stream,
                            "band": band,
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
                        train_y = np.concatenate(train_y_blocks, axis=0)
                        if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
                            rows.append({**base, "skip_reason": "single_class_train_or_test"})
                            continue
                        scores = _ridge_scores(train_x, train_y, test_x, alpha=float(ridge_alpha))
                        actual_ba = _balanced_accuracy(test_y, scores)
                        actual_auc = _auc(test_y, scores)
                        null_ba: list[float] = []
                        null_auc: list[float] = []
                        for _ in range(int(permutations)):
                            shuffled_y = np.concatenate(
                                [_shuffle_blocks(rng, block, int(null_block_size)) for block in train_y_blocks]
                            )
                            shuffled_scores = _ridge_scores(train_x, shuffled_y, test_x, alpha=float(ridge_alpha))
                            null_ba.append(_balanced_accuracy(test_y, shuffled_scores))
                            null_auc.append(_auc(test_y, shuffled_scores))
                        null_ba_arr = np.asarray(null_ba, dtype=np.float32)
                        null_auc_arr = np.asarray(null_auc, dtype=np.float32)
                        rows.append(
                            {
                                **base,
                                "accuracy": float(np.mean(np.where(scores >= 0.0, 1, -1) == test_y)),
                                "balanced_accuracy": actual_ba,
                                "auc": actual_auc,
                                "mcc": _mcc(test_y, scores),
                                "null_balanced_accuracy_mean": float(np.nanmean(null_ba_arr)),
                                "null_auc_mean": float(np.nanmean(null_auc_arr)),
                                "actual_minus_null_balanced_accuracy": float(actual_ba - np.nanmean(null_ba_arr)),
                                "actual_minus_null_auc": float(actual_auc - np.nanmean(null_auc_arr)),
                                "empirical_p_auc": float((1 + np.sum(null_auc_arr >= actual_auc)) / (1 + len(null_auc_arr))),
                                "skip_reason": "",
                            }
                        )
    return rows


def _aggregate_decoder_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    if frame.empty or "auc" not in frame.columns:
        return []
    valid = frame[(frame["skip_reason"].fillna("") == "") & np.isfinite(frame["auc"])]
    out: list[dict[str, Any]] = []
    for (window, stream, band, lag), group in valid.groupby(["window", "stream", "band", "lag"], sort=True):
        out.append(
            {
                "window": str(window),
                "stream": str(stream),
                "band": str(band),
                "lag": int(lag),
                "relative_start": int(group["relative_start"].iloc[0]),
                "relative_stop": int(group["relative_stop"].iloc[0]),
                "heldout_trial_count": int(group["heldout_trial_id"].nunique()),
                "sample_count": int(group["sample_count"].sum()),
                "mean_auc": float(group["auc"].mean()),
                "mean_balanced_accuracy": float(group["balanced_accuracy"].mean()),
                "mean_mcc": float(group["mcc"].mean()),
                "mean_null_auc": float(group["null_auc_mean"].mean()),
                "mean_actual_minus_null_auc": float(group["actual_minus_null_auc"].mean()),
                "mean_null_balanced_accuracy": float(group["null_balanced_accuracy_mean"].mean()),
                "mean_actual_minus_null_balanced_accuracy": float(
                    group["actual_minus_null_balanced_accuracy"].mean()
                ),
                "mean_empirical_p_auc": float(group["empirical_p_auc"].mean()),
            }
        )
    return out


def _build_onset_latency_rows(
    summary_rows: list[dict[str, Any]],
    *,
    auc_threshold: float = 0.70,
    delta_threshold: float = 0.10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize where direction predictivity first becomes clearly above null.

    This is intentionally a small descriptive regression over the three coarse
    anatomical bands. It should be read as an onset-latency audit, not as a
    strong wave-propagation test.
    """

    frame = pd.DataFrame(summary_rows)
    if frame.empty:
        return [], []

    sequential_windows = ["pre_edge", "edge_impulse", "early_ramification", "late_ramification"]
    band_position = {"frontier": 0.0, "middle": 1.0, "rear": 2.0}
    rows: list[dict[str, Any]] = []
    for stream in ("body", "signal"):
        for band in ("frontier", "middle", "rear"):
            subset = frame[
                (frame["stream"] == stream)
                & (frame["band"] == band)
                & (frame["window"].isin(sequential_windows))
            ].copy()
            if subset.empty:
                rows.append(
                    {
                        "stream": stream,
                        "band": band,
                        "band_position": band_position[band],
                        "status": "not_estimable",
                        "auc_threshold": float(auc_threshold),
                        "delta_threshold": float(delta_threshold),
                    }
                )
                continue

            subset["_window_order"] = subset["window"].map({name: index for index, name in enumerate(sequential_windows)})
            subset = subset.sort_values(["_window_order", "lag"], kind="stable")
            valid = subset[
                np.isfinite(subset["mean_auc"].astype(float))
                & np.isfinite(subset["mean_actual_minus_null_auc"].astype(float))
            ].copy()
            hits = valid[
                (valid["mean_auc"].astype(float) >= float(auc_threshold))
                & (valid["mean_actual_minus_null_auc"].astype(float) >= float(delta_threshold))
            ]
            best = valid.sort_values("mean_auc", ascending=False).iloc[0] if not valid.empty else subset.iloc[0]
            if hits.empty:
                rows.append(
                    {
                        "stream": stream,
                        "band": band,
                        "band_position": band_position[band],
                        "status": "no_threshold_crossing",
                        "auc_threshold": float(auc_threshold),
                        "delta_threshold": float(delta_threshold),
                        "best_window": str(best["window"]),
                        "best_lag": int(best["lag"]) if np.isfinite(best.get("lag", np.nan)) else "",
                        "best_auc": float(best.get("mean_auc", np.nan)),
                        "best_actual_minus_null_auc": float(best.get("mean_actual_minus_null_auc", np.nan)),
                    }
                )
                continue

            onset = hits.iloc[0]
            rel_start = float(onset["relative_start"])
            rel_stop = float(onset["relative_stop"])
            rows.append(
                {
                    "stream": stream,
                    "band": band,
                    "band_position": band_position[band],
                    "status": "threshold_crossing",
                    "auc_threshold": float(auc_threshold),
                    "delta_threshold": float(delta_threshold),
                    "onset_window": str(onset["window"]),
                    "onset_lag": int(onset["lag"]),
                    "onset_relative_start": rel_start,
                    "onset_relative_stop": rel_stop,
                    "onset_relative_center": (rel_start + rel_stop) / 2.0,
                    "onset_auc": float(onset["mean_auc"]),
                    "onset_actual_minus_null_auc": float(onset["mean_actual_minus_null_auc"]),
                    "best_window": str(best["window"]),
                    "best_lag": int(best["lag"]) if np.isfinite(best.get("lag", np.nan)) else "",
                    "best_auc": float(best.get("mean_auc", np.nan)),
                    "best_actual_minus_null_auc": float(best.get("mean_actual_minus_null_auc", np.nan)),
                }
            )

    regression_rows: list[dict[str, Any]] = []
    onset_frame = pd.DataFrame(rows)
    for stream in ("body", "signal"):
        valid = onset_frame[
            (onset_frame["stream"] == stream)
            & (onset_frame["status"] == "threshold_crossing")
            & np.isfinite(onset_frame.get("onset_relative_center", np.nan).astype(float))
        ].copy()
        if len(valid) < 2:
            regression_rows.append(
                {
                    "stream": stream,
                    "status": "not_estimable",
                    "n_bands": int(len(valid)),
                    "reason": "fewer_than_two_threshold_crossings",
                    "auc_threshold": float(auc_threshold),
                    "delta_threshold": float(delta_threshold),
                }
            )
            continue
        x = valid["band_position"].to_numpy(dtype=float)
        y = valid["onset_relative_center"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, deg=1)
        if len(valid) >= 3:
            fitted = intercept + slope * x
            ss_res = float(np.sum((y - fitted) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
        else:
            r2 = float("nan")
        regression_rows.append(
            {
                "stream": stream,
                "status": "estimated",
                "n_bands": int(len(valid)),
                "slope_steps_per_band_position": float(slope),
                "intercept_relative_center": float(intercept),
                "r2": r2,
                "auc_threshold": float(auc_threshold),
                "delta_threshold": float(delta_threshold),
            }
        )
    return rows, regression_rows


def _frontier_control_rows(
    *,
    metric_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    lag: int = 8,
) -> list[dict[str, Any]]:
    metrics = pd.DataFrame(metric_rows)
    summary = pd.DataFrame(summary_rows)
    rows: list[dict[str, Any]] = []
    for stream in ("body", "signal"):
        for window in ("pre_edge", "edge_impulse"):
            if not summary.empty:
                valid = summary[
                    (summary["window"] == window)
                    & (summary["stream"] == stream)
                    & (summary["band"] == "frontier")
                    & (summary["lag"] == int(lag))
                ]
            else:
                valid = pd.DataFrame()
            if not valid.empty:
                row = dict(valid.iloc[0])
                row["control_status"] = "estimated"
                row["dominant_skip_reason"] = ""
                rows.append(row)
                continue
            raw = metrics[
                (metrics["window"] == window)
                & (metrics["stream"] == stream)
                & (metrics["band"] == "frontier")
                & (metrics["lag"] == int(lag))
            ]
            reason = ""
            if not raw.empty and "skip_reason" in raw.columns:
                counts = raw["skip_reason"].fillna("").replace("", "estimated").value_counts()
                reason = str(counts.index[0]) if not counts.empty else ""
            rows.append(
                {
                    "window": window,
                    "stream": stream,
                    "band": "frontier",
                    "lag": int(lag),
                    "control_status": "not_estimable",
                    "dominant_skip_reason": reason,
                    "heldout_trial_count": int(raw["heldout_trial_id"].nunique()) if not raw.empty else 0,
                    "sample_count": int(raw["sample_count"].sum()) if not raw.empty else 0,
                    "toward_count": int(raw["toward_count"].sum()) if not raw.empty else 0,
                    "away_count": int(raw["away_count"].sum()) if not raw.empty else 0,
                    "mean_auc": float("nan"),
                    "mean_balanced_accuracy": float("nan"),
                    "mean_null_auc": float("nan"),
                    "mean_actual_minus_null_auc": float("nan"),
                    "mean_null_balanced_accuracy": float("nan"),
                    "mean_actual_minus_null_balanced_accuracy": float("nan"),
                    "mean_empirical_p_auc": float("nan"),
                }
            )
    return rows


def build_s4_band_predictivity(
    *,
    output_dir: Path,
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
    trials, meta_rows, robot_shape = _load_band_trials(
        trace_root=DEFORMATION_OFF_EDGE_TRACE_ROOT,
        future_window=int(future_window),
        min_displacement=float(min_displacement),
        persistence_steps=int(persistence_steps),
        persistence_fraction=float(persistence_fraction),
    )
    metric_rows = _evaluate_band_decoders(
        trials,
        lags=[int(value) for value in lags],
        ridge_alpha=float(ridge_alpha),
        permutations=int(permutations),
        null_block_size=int(null_block_size),
        random_seed=int(random_seed),
    )
    summary_rows = _aggregate_decoder_metrics(metric_rows)
    onset_rows, onset_regression_rows = _build_onset_latency_rows(summary_rows)
    _write_csv(output_dir / "s4_band_predictivity_trial_metadata.csv", meta_rows)
    _write_csv(output_dir / "s4_band_predictivity_decoder_metrics.csv", metric_rows)
    _write_csv(output_dir / "s4_band_predictivity_by_window.csv", summary_rows)
    _write_csv(output_dir / "s4_band_predictivity_onsets.csv", onset_rows)
    _write_csv(output_dir / "s4_band_predictivity_onset_regression.csv", onset_regression_rows)
    controls = _frontier_control_rows(metric_rows=metric_rows, summary_rows=summary_rows, lag=8)
    _write_csv(output_dir / "s4_decoder_controls.csv", controls)
    return {
        "trial_rows": meta_rows,
        "metric_rows": metric_rows,
        "summary_rows": summary_rows,
        "onset_rows": onset_rows,
        "onset_regression_rows": onset_regression_rows,
        "control_rows": controls,
        "robot_shape": robot_shape,
    }


def _subset_rate_by_specimen(
    manifest: pd.DataFrame,
    *,
    selector: pd.Series,
    outcome: str,
) -> pd.Series:
    subset = manifest[selector].copy()
    if subset.empty or outcome not in subset.columns:
        return pd.Series(dtype=float)
    subset[outcome] = _bool_series(subset[outcome])
    return subset.groupby("specimen_id")[outcome].mean()


def build_cohort_tables(output_dir: Path) -> dict[str, Any]:
    manifest = pd.read_csv(DATA_ROOT / "tables" / "episode_manifest.csv")
    for column in ("success", "fell", "survived"):
        if column in manifest.columns:
            manifest[column] = _bool_series(manifest[column])
    specimens = sorted(str(value) for value in manifest["specimen_id"].dropna().unique())
    rows: list[dict[str, Any]] = []
    for specimen in specimens:
        base = manifest["specimen_id"].astype(str) == specimen
        native_intact = (
            base
            & (manifest["assay_family"] == "signaling_ablation")
            & (manifest["surface_id"] == "native_core")
            & (manifest["condition_id"] == "intact")
        )
        extended_intact = (
            base
            & (manifest["assay_family"] == "signaling_ablation")
            & (manifest["surface_id"] == "long_distance_generalization")
            & (manifest["condition_id"] == "intact")
        )
        double_intact = (
            base
            & (manifest["assay_family"] == "signaling_ablation")
            & (manifest["surface_id"] == "double_cliff")
            & (manifest["condition_id"] == "intact")
        )
        def_native = (
            base
            & (manifest["assay_family"] == "deformation_ablation")
            & (manifest["surface_id"] == "native_core")
        )
        sig_native = (
            base
            & (manifest["assay_family"] == "signaling_ablation")
            & (manifest["surface_id"] == "native_core")
        )
        transfer = (
            base
            & (manifest["assay_family"] == "body_transfer")
            & (manifest["surface_id"] == "native_core")
        )
        non_native_transfer_rates = []
        for body in ("rect_6x2", "rect_8x2", "rect_7x3"):
            non_native_transfer_rates.append(
                _rate(manifest[transfer & (manifest["condition_id"] == body)], "success")
            )
        rows.append(
            {
                "specimen_id": specimen,
                "is_s4": specimen == S4_SPECIMEN,
                "native_intact_success": _rate(manifest[native_intact], "success"),
                "extended_intact_success": _rate(manifest[extended_intact], "success"),
                "double_cliff_intact_survival": _rate(manifest[double_intact], "survived"),
                "native_deformation_off_success": _rate(
                    manifest[def_native & (manifest["condition_id"] == "deformation_off")],
                    "success",
                ),
                "native_absolute_only_success": _rate(
                    manifest[def_native & (manifest["condition_id"] == "absolute_only")],
                    "success",
                ),
                "native_compliance_only_success": _rate(
                    manifest[def_native & (manifest["condition_id"] == "compliance_only")],
                    "success",
                ),
                "native_sobel_x_off_success": _rate(
                    manifest[sig_native & (manifest["condition_id"] == "sobel_x_off")],
                    "success",
                ),
                "native_identity_comm_off_success": _rate(
                    manifest[sig_native & (manifest["condition_id"] == "identity_comm_off")],
                    "success",
                ),
                "native_all_comm_off_success": _rate(
                    manifest[sig_native & (manifest["condition_id"] == "all_comm_off")],
                    "success",
                ),
                "non_native_body_transfer_mean_success": _mean_or_nan(non_native_transfer_rates),
                "initial_direction_observed_count_native_intact": int(
                    np.sum(
                        native_intact
                        & manifest.get("initial_direction", pd.Series("", index=manifest.index)).notna()
                        & ~manifest.get("initial_direction", pd.Series("", index=manifest.index)).astype(str).eq("none")
                    )
                ),
                "mean_turn_count_native_intact": float(
                    manifest.loc[native_intact, "turn_count"].mean()
                )
                if "turn_count" in manifest.columns
                else float("nan"),
            }
        )
    _write_csv(output_dir / "eight_specimen_profile_table.csv", rows)

    exclude_rows: list[dict[str, Any]] = []
    selectors = [
        (
            "native_deformation_off_success",
            (manifest["assay_family"] == "deformation_ablation")
            & (manifest["surface_id"] == "native_core")
            & (manifest["condition_id"] == "deformation_off"),
            "success",
        ),
        (
            "extended_intact_success",
            (manifest["assay_family"] == "signaling_ablation")
            & (manifest["surface_id"] == "long_distance_generalization")
            & (manifest["condition_id"] == "intact"),
            "success",
        ),
        (
            "double_cliff_intact_survival",
            (manifest["assay_family"] == "signaling_ablation")
            & (manifest["surface_id"] == "double_cliff")
            & (manifest["condition_id"] == "intact"),
            "survived",
        ),
    ]
    for label, selector, outcome in selectors:
        subset = manifest[selector].copy()
        subset[outcome] = _bool_series(subset[outcome])
        for group_label, group in [
            ("all_8", subset),
            ("excluding_s4", subset[subset["specimen_id"].astype(str) != S4_SPECIMEN]),
        ]:
            specimen_rates = group.groupby("specimen_id")[outcome].mean()
            exclude_rows.append(
                {
                    "metric": label,
                    "group": group_label,
                    "specimen_count": int(specimen_rates.size),
                    "episode_count": int(len(group)),
                    "positive_count": int(group[outcome].sum()),
                    "episode_rate": float(group[outcome].mean()) if len(group) else float("nan"),
                    "specimen_mean_rate": float(specimen_rates.mean()) if specimen_rates.size else float("nan"),
                    "specimen_sem_rate": _sem(specimen_rates.to_numpy(dtype=np.float64)),
                    "min_specimen_rate": float(specimen_rates.min()) if specimen_rates.size else float("nan"),
                    "max_specimen_rate": float(specimen_rates.max()) if specimen_rates.size else float("nan"),
                }
            )
    _write_csv(output_dir / "excluding_s4_summary.csv", exclude_rows)
    return {"profile_rows": rows, "excluding_rows": exclude_rows}


def _plot_edge_timing(output_dir: Path, timing: dict[str, Any]) -> None:
    frame = pd.DataFrame(timing["episode_rows"])
    true = frame[frame["true_cliff_event"]].copy()
    if true.empty:
        return
    metrics = [
        ("event_to_goal_commit_steps", "Steps to goal-side\ncommitment"),
        ("max_cliff_overhang_before_goal_commit", "Max overhang before\ncommitment"),
        ("direction_changes_after_event", "Direction changes\nafter edge"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.2, 2.6), dpi=180)
    for ax, (metric, label) in zip(axes, metrics, strict=True):
        means = true.groupby("condition")[metric].mean().reindex(["intact", "deformation_off"])
        sems = true.groupby("condition")[metric].sem().reindex(["intact", "deformation_off"])
        x = np.arange(len(means))
        ax.bar(x, means.to_numpy(dtype=float), yerr=sems.to_numpy(dtype=float), color=["#8CB87C", "#3B5B92"], width=0.62)
        for xi, condition in enumerate(means.index):
            values = true[true["condition"] == condition][metric].to_numpy(dtype=float)
            ax.scatter(np.full(values.shape, xi), values, s=13, color="#222222", zorder=3, alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels(["Intact", "Def. off"], rotation=25, ha="right")
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"s4_edge_timing_summary.{suffix}", bbox_inches="tight")
    plt.close(fig)


def _plot_band_predictivity(output_dir: Path, band_summary: list[dict[str, Any]], *, lag: int = 8) -> None:
    frame = pd.DataFrame(band_summary)
    frame = frame[(frame["lag"] == int(lag)) & frame["window"].isin(WINDOWS.keys())].copy()
    if frame.empty:
        return
    windows = list(WINDOWS.keys())
    bands = ["frontier", "middle", "rear"]
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.5), dpi=180, sharex=True, sharey=True)
    for ax, stream in zip(axes, ["body", "signal"], strict=True):
        subset = frame[frame["stream"] == stream]
        x = np.arange(len(windows), dtype=np.float32)
        width = 0.22
        for offset, band in zip([-width, 0.0, width], bands, strict=True):
            values = []
            for window in windows:
                value = subset[(subset["window"] == window) & (subset["band"] == band)]["mean_auc"]
                values.append(float(value.iloc[0]) if not value.empty else np.nan)
            ax.bar(
                x + offset,
                values,
                width=width,
                color=BAND_COLORS[band],
                label=band if stream == "body" else None,
            )
        for window_index, window in enumerate(windows):
            present = subset[subset["window"] == window]
            if present.empty:
                ax.text(window_index, 0.515, "n/e", ha="center", va="bottom", fontsize=7, color="#555555")
        ax.axhline(0.5, color="#777777", linestyle="--", linewidth=0.9)
        ax.axvline(0.5, color="#B13B35", linewidth=1.0)
        ax.set_ylim(0.45, 1.02)
        ax.set_ylabel(f"{stream.title()} AUC")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=3, loc="upper right", fontsize=7)
    axes[-1].set_xticks(np.arange(len(windows)))
    axes[-1].set_xticklabels([WINDOW_LABELS[item] for item in windows], rotation=25, ha="right")
    axes[-1].set_xlabel("Window relative to first cliff-frontier event")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"s4_band_predictivity_lag{lag}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def _plot_specimen_profile(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    frame = frame.sort_values("specimen_id", kind="stable").reset_index(drop=True)
    labels: list[str] = []
    next_index = 1
    for _, row in frame.iterrows():
        if bool(row["is_s4"]):
            labels.append("S4")
            continue
        labels.append(f"O{next_index}")
        next_index += 1
    s4_index = int(frame.index[frame["is_s4"].astype(bool)][0]) if frame["is_s4"].any() else None
    metrics = [
        ("native_intact_success", "native", "#8CB87C"),
        ("extended_intact_success", "extended", "#3B5B92"),
        ("double_cliff_intact_survival", "double\nsurvival", "#5A4E8C"),
        ("native_deformation_off_success", "def. off", "#A9C99A"),
        ("native_sobel_x_off_success", "Sobel-X off", "#D4736C"),
    ]
    x = np.arange(len(labels), dtype=np.float32)
    width = 0.16
    fig, ax = plt.subplots(figsize=(7.2, 2.9), dpi=180)
    for index, (column, label, color) in enumerate(metrics):
        values = frame[column].to_numpy(dtype=float)
        ax.bar(x + (index - 2) * width, values, width=width, label=label, color=color, alpha=0.95)
    if s4_index is not None:
        ax.axvspan(s4_index - 0.5, s4_index + 0.5, color="#3B5B92", alpha=0.08)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.legend(frameon=False, ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"eight_specimen_profile_bars.{suffix}", bbox_inches="tight")
    plt.close(fig)


def _fmt_rate(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{100.0 * value:.1f}%"


def _write_markdown_summary(
    output_dir: Path,
    *,
    timing: dict[str, Any],
    band: dict[str, Any],
    cohort: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    timing_summary = pd.DataFrame(timing["summary_rows"])
    paired = pd.DataFrame(timing["paired_rows"])
    band_summary = pd.DataFrame(band["summary_rows"])
    controls = pd.DataFrame(band["control_rows"])
    onsets = pd.DataFrame(band["onset_rows"])
    onset_regression = pd.DataFrame(band["onset_regression_rows"])
    profile = pd.DataFrame(cohort["profile_rows"])
    excluding = pd.DataFrame(cohort["excluding_rows"])

    def best_row(stream: str, band_name: str = "frontier", window: str = "edge_impulse", lag: int = 8) -> str:
        subset = band_summary[
            (band_summary["stream"] == stream)
            & (band_summary["band"] == band_name)
            & (band_summary["window"] == window)
            & (band_summary["lag"] == int(lag))
        ]
        if subset.empty:
            return "not estimable"
        row = subset.iloc[0]
        return (
            f"AUC {row['mean_auc']:.3f}, balanced accuracy {row['mean_balanced_accuracy']:.3f}, "
            f"actual-minus-null AUC {row['mean_actual_minus_null_auc']:.3f}"
        )

    lines: list[str] = []
    lines.append("# S4 mechanism packet and eight-specimen sanity notes")
    lines.append("")
    lines.append("This packet belongs to the new eight-specimen analysis directory:")
    lines.append(f"`{DATA_ROOT}`.")
    lines.append("")
    lines.append("## Source data")
    lines.append("")
    lines.append(f"- Eight-specimen endpoint manifest: `{DATA_ROOT / 'tables' / 'episode_manifest.csv'}`")
    lines.append(f"- S4 intact native state-space trace: `{INTACT_NATIVE_TRACE_ROOT}`")
    lines.append(f"- S4 deformation-off edge state-space trace: `{DEFORMATION_OFF_EDGE_TRACE_ROOT}`")
    lines.append("")
    lines.append("The script uses existing deterministic replay/state-space outputs only. It does not run training, stochastic sweeps, or new evaluations.")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    for name in [
        "s4_edge_timing_episode_metrics.csv",
        "s4_edge_timing_summary.csv",
        "s4_edge_timing_paired_differences.csv",
        "s4_band_predictivity_trial_metadata.csv",
        "s4_band_predictivity_decoder_metrics.csv",
        "s4_band_predictivity_by_window.csv",
        "s4_band_predictivity_onsets.csv",
        "s4_band_predictivity_onset_regression.csv",
        "s4_decoder_controls.csv",
        "eight_specimen_profile_table.csv",
        "excluding_s4_summary.csv",
        "s4_edge_timing_summary.pdf/png",
        "s4_band_predictivity_lag8.pdf/png",
        "eight_specimen_profile_bars.pdf/png",
    ]:
        lines.append(f"- `{name}`")
    lines.append("")
    lines.append("## Event, label, and lag definitions")
    lines.append("")
    lines.append("- The cliff event is the first timestep with `cliff_clearance <= 0`, or the recorded `first_cliff_reach_step` when present.")
    lines.append("- Episodes without this event are kept as closest-approach/direct-goal episodes, but true event timing summaries are computed only from true cliff-frontier events.")
    lines.append("- In the figures, the red vertical line separates the pre-edge window from event-centered/post-event windows; in the prior spatial heatmaps, a red square marks event-leading/frontier cells.")
    lines.append(f"- Committed-direction labels use future COM displacement over `{args.future_window}` env steps, minimum displacement `{args.min_displacement}`, persistence `{args.persistence_steps}` steps, and persistence fraction `{args.persistence_fraction}`.")
    lines.append("- A decoder lag of `8` means features at relative time `t` are used to predict the committed-direction label at `t + 8`; the label itself is still based on the future-COM rule above.")
    lines.append("- Signal features are hidden NCA channels plus CTRNN carry state for the selected band. Deformation channels, action outputs, body geometry, COM, cliff clearance, and target/cliff coordinates are excluded.")
    lines.append("- Body features are per-cell attached spring state, h/v drives, local corner/point geometry, and velocities, using the same local body-state feature construction as the voxel heatmap analysis.")
    lines.append("")
    lines.append("## S4 timing summary")
    lines.append("")
    if not timing_summary.empty:
        for _, row in timing_summary.iterrows():
            lines.append(
                f"- `{row['condition']}`: {int(row['true_cliff_event_count'])}/{int(row['episode_count'])} native episodes have true cliff-frontier events; "
                f"mean event-to-goal-side commitment {row['mean_event_to_goal_commit_steps_true_events']:.1f} steps; "
                f"mean max overhang before commitment {row['mean_max_overhang_before_goal_commit_true_events']:.3f}; "
                f"mean post-event direction changes {row['mean_direction_changes_after_event_true_events']:.2f}."
            )
    if not paired.empty:
        lines.append("")
        lines.append("Paired true-event differences are reported as deformation-off minus intact:")
        for _, row in paired.iterrows():
            lines.append(
                f"- `{row['distance_pair']}` mirrored true-edge episodes "
                f"(intact `{row['intact_side']}`, deformation-off `{row['deformation_off_side']}`): "
                f"commitment latency {row['event_to_goal_commit_steps_deformation_off_minus_intact']:.1f} steps, "
                f"overhang {row['max_overhang_before_goal_commit_deformation_off_minus_intact']:.3f}, "
                f"direction changes {row['direction_changes_after_event_deformation_off_minus_intact']:.1f}."
            )
    lines.append("")
    lines.append("Interpretation guardrail: these timing metrics can support claims about delayed, deeper, or less stable recommitment only when the paired differences actually show that pattern.")
    lines.append("")
    lines.append("## S4 band predictivity")
    lines.append("")
    lines.append(f"- Frontier body, edge-impulse window, lag 8: {best_row('body')}.")
    lines.append(f"- Frontier signal, edge-impulse window, lag 8: {best_row('signal')}.")
    if not controls.empty:
        statuses = ", ".join(sorted(set(str(value) for value in controls.get("control_status", []))))
        lines.append(
            "- `s4_decoder_controls.csv` contains frontier pre-edge and edge-impulse control attempts at lag 8. "
            f"Rows are marked by `control_status` (`{statuses}`); not-estimable rows preserve the dominant skip reason."
        )
    if not onsets.empty:
        lines.append("")
        lines.append("Onset-latency audit:")
        for stream in ("body", "signal"):
            subset = onsets[onsets["stream"] == stream]
            if subset.empty:
                continue
            readable: list[str] = []
            for _, row in subset.iterrows():
                if row.get("status") == "threshold_crossing":
                    readable.append(
                        f"{row['band']} at `{row['onset_window']}` "
                        f"(center {float(row['onset_relative_center']):.1f}, AUC {float(row['onset_auc']):.3f})"
                    )
                else:
                    readable.append(f"{row['band']} {row.get('status', 'not_estimable')}")
            lines.append(f"- `{stream}`: " + "; ".join(readable) + ".")
        if not onset_regression.empty:
            for _, row in onset_regression.iterrows():
                if row.get("status") == "estimated":
                    r2 = row.get("r2", np.nan)
                    r2_text = "NA" if not np.isfinite(r2) else f"{float(r2):.3f}"
                    lines.append(
                        f"- `{row['stream']}` onset slope: {float(row['slope_steps_per_band_position']):.1f} "
                        f"relative steps per band position across {int(row['n_bands'])} bands; R2 {r2_text}."
                    )
                else:
                    lines.append(
                        f"- `{row['stream']}` onset slope not estimable "
                        f"({row.get('reason', 'insufficient threshold crossings')})."
                    )
    lines.append("")
    lines.append("Paper-safe reading: use this as an observational decoder result. It asks whether body or signal state carries linearly decodable information about committed direction around the edge event; it does not prove that the decoded stream causes reversal.")
    lines.append("")
    lines.append("## Eight-specimen sanity table")
    lines.append("")
    if not profile.empty:
        s4 = profile[profile["is_s4"].astype(bool)]
        if not s4.empty:
            row = s4.iloc[0]
            lines.append(
                f"S4 profile: native intact {_fmt_rate(row['native_intact_success'])}, "
                f"extended intact {_fmt_rate(row['extended_intact_success'])}, "
                f"double-cliff survival {_fmt_rate(row['double_cliff_intact_survival'])}, "
                f"native deformation-off {_fmt_rate(row['native_deformation_off_success'])}, "
                f"Sobel-X-off {_fmt_rate(row['native_sobel_x_off_success'])}, "
                f"all-comm-off {_fmt_rate(row['native_all_comm_off_success'])}."
            )
        lines.append("")
        lines.append("The per-specimen table is the main place to support the diversity framing: specimens vary across native, extended, symmetric survival, deformation removal, signaling removal, and body transfer axes.")
    lines.append("")
    lines.append("## Excluding-S4 checks")
    lines.append("")
    if not excluding.empty:
        for metric, group in excluding.groupby("metric", sort=False):
            all_row = group[group["group"] == "all_8"].iloc[0]
            ex_row = group[group["group"] == "excluding_s4"].iloc[0]
            lines.append(
                f"- `{metric}`: all eight specimen-mean rate {_fmt_rate(all_row['specimen_mean_rate'])}; "
                f"excluding S4 {_fmt_rate(ex_row['specimen_mean_rate'])}."
            )
    lines.append("")
    lines.append("## Initial commitment / behavioral grammar note")
    lines.append("")
    lines.append("The consolidated episode manifest has sparse `initial_direction` values, so this packet records the count of observed initial-commitment labels but does not use them for a strong blind-commitment analysis. A fuller behavioral grammar would need trace-level event extraction for all specimens/conditions.")
    lines.append("")
    lines.append("## Deferred broader study / interpolation probe")
    lines.append("")
    lines.append("The interpolation and timing-trap idea should remain a follow-up note rather than a main result in this packet. The useful broader study would run S4, and only later the wider cohort if warranted, across a dense sweep of cliff distances/plateau widths with intact and deformation-off traces. The key question would be whether reversal timing aligns with the physical edge event across interpolated distances, or with an approximately fixed internal phase/time.")
    lines.append("")
    lines.append("Suggested follow-up design:")
    lines.append("")
    lines.append("- S4-only first, intact and deformation-off.")
    lines.append("- Native-style mirrored goal-cliff terrains with cliff distance varied one unit at a time across and between the trained native/extended gaps.")
    lines.append("- Full state-space traces only for successful/probed episodes needed for mechanism analysis.")
    lines.append("- Primary metrics: first edge event, reversal latency, commitment depth/overhang, COM displacement after edge, direction-change count, and band-level body/signal AUC aligned to the event.")
    lines.append("- Interpretation: if reversal tracks edge event under deformation-off, that supports event-coupled body dynamics; if it tracks elapsed time, describe S4 as a task-calibrated internally phased regime rather than a clean edge-triggered mechanism.")
    lines.append("")
    lines.append("Existing interpolation probe outputs are treated as exploratory side probes, not replacement native results:")
    lines.append("")
    lines.append(f"- `{DATA_ROOT / 'raw_modal_s4_interpolated_state_space_20260613T004230' / 'RUN_SUMMARY.md'}`")
    lines.append(f"- `{DATA_ROOT / 'raw_modal_s4_interpolated_pairwise_state_space_20260613T005525' / 'RUN_SUMMARY.md'}`")
    lines.append(f"- `{DATA_ROOT / 'raw_modal_s4_timing_trap_dense_state_space_20260613T020953' / 'RUN_SUMMARY.md'}`")
    lines.append("")
    lines.append("The dense timing-trap probe has now been run on Modal for S4 only. It crossed goal distances `2,4,6,8,10` with cliff distances `1-6`, for intact and deformation-off state-space replay. It should remain exploratory: the first audit suggests event timing tracks physical cliff distance, while deformation-off post-edge success varies non-monotonically with cliff distance, consistent with phase/rhythm susceptibility rather than a simple fixed-time reversal account.")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append(".venv/bin/python preprint/scripts/s4_mechanism_packet.py")
    lines.append("```")
    lines.append("")
    lines.append("The command writes all outputs into this directory and records the key parameter settings in `summary.json`.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mechanism_packet_summary.md").write_text("\n".join(lines) + "\n")


def build_packet(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timing = build_s4_edge_timing(
        output_dir=output_dir,
        future_window=int(args.future_window),
        min_displacement=float(args.min_displacement),
        persistence_steps=int(args.persistence_steps),
        persistence_fraction=float(args.persistence_fraction),
    )
    band = build_s4_band_predictivity(
        output_dir=output_dir,
        future_window=int(args.future_window),
        min_displacement=float(args.min_displacement),
        persistence_steps=int(args.persistence_steps),
        persistence_fraction=float(args.persistence_fraction),
        lags=[int(value) for value in args.lags],
        ridge_alpha=float(args.ridge_alpha),
        permutations=int(args.permutations),
        null_block_size=int(args.null_block_size),
        random_seed=int(args.random_seed),
    )
    cohort = build_cohort_tables(output_dir=output_dir)
    _plot_edge_timing(output_dir, timing)
    _plot_band_predictivity(output_dir, band["summary_rows"], lag=8)
    _plot_specimen_profile(output_dir, cohort["profile_rows"])
    _write_markdown_summary(output_dir, timing=timing, band=band, cohort=cohort, args=args)
    summary = {
        "schema_version": 1,
        "kind": "s4_mechanism_packet",
        "output_dir": str(output_dir),
        "source_data_root": str(DATA_ROOT.resolve()),
        "intact_native_trace_root": str(INTACT_NATIVE_TRACE_ROOT.resolve()),
        "deformation_off_edge_trace_root": str(DEFORMATION_OFF_EDGE_TRACE_ROOT.resolve()),
        "s4_specimen_id": S4_SPECIMEN,
        "label_policy": {
            "future_window": int(args.future_window),
            "min_displacement": float(args.min_displacement),
            "persistence_steps": int(args.persistence_steps),
            "persistence_fraction": float(args.persistence_fraction),
        },
        "decoder_policy": {
            "lags": [int(value) for value in args.lags],
            "ridge_alpha": float(args.ridge_alpha),
            "permutations": int(args.permutations),
            "null_block_size": int(args.null_block_size),
            "random_seed": int(args.random_seed),
        },
        "outputs": {
            "edge_episode_metrics": str(output_dir / "s4_edge_timing_episode_metrics.csv"),
            "edge_summary": str(output_dir / "s4_edge_timing_summary.csv"),
            "edge_paired_differences": str(output_dir / "s4_edge_timing_paired_differences.csv"),
            "band_trial_metadata": str(output_dir / "s4_band_predictivity_trial_metadata.csv"),
            "band_decoder_metrics": str(output_dir / "s4_band_predictivity_decoder_metrics.csv"),
            "band_summary": str(output_dir / "s4_band_predictivity_by_window.csv"),
            "band_onsets": str(output_dir / "s4_band_predictivity_onsets.csv"),
            "band_onset_regression": str(output_dir / "s4_band_predictivity_onset_regression.csv"),
            "decoder_controls": str(output_dir / "s4_decoder_controls.csv"),
            "specimen_profile": str(output_dir / "eight_specimen_profile_table.csv"),
            "excluding_s4_summary": str(output_dir / "excluding_s4_summary.csv"),
            "markdown_summary": str(output_dir / "mechanism_packet_summary.md"),
            "summary_json": str(output_dir / "summary.json"),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--future-window", type=int, default=32)
    parser.add_argument("--min-displacement", type=float, default=0.025)
    parser.add_argument("--persistence-steps", type=int, default=12)
    parser.add_argument("--persistence-fraction", type=float, default=0.75)
    parser.add_argument("--lags", type=_parse_lags, default=_parse_lags("0,8,16"))
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--permutations", type=int, default=30)
    parser.add_argument("--null-block-size", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=137)
    args = parser.parse_args()
    summary = build_packet(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
