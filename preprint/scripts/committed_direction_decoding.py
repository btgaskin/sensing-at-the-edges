#!/usr/bin/env python
"""Decode committed movement direction from deformation-off state-space traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_TRACE_ROOT = Path(
    "paper_data/preprint/native_core_ablation_v1/"
    "raw_modal_state_space_deformation_off_latest/"
    "preprint_native_state_space_deformation_off_v1/"
    "paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed137_e4fcc2fb/"
    "bin_03_rank_00_cell_6-0-3_flat_0423/"
    "state_space_trace_deformation_off"
)
DEFAULT_OUTPUT_DIR = Path(
    "paper_data/preprint/native_core_ablation_v1/figures/committed_direction_deformation_off"
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], *, label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _load_step_manifest(trace_root: Path) -> pd.DataFrame:
    path = trace_root / "step_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing step manifest: {path}")
    frame = pd.read_csv(path)
    _require_columns(
        frame,
        (
            "trial_id",
            "step_index",
            "env_step",
            "goal_direction",
            "com_x",
            "com_y",
            "cliff_clearance",
            "trace_npz",
        ),
        label=str(path),
    )
    return frame


def _trace_path_for_trial(trace_root: Path, rows: pd.DataFrame) -> Path:
    remote_path = Path(str(rows.iloc[0]["trace_npz"]))
    candidate = trace_root / "traces" / remote_path.name
    if candidate.exists():
        return candidate
    fallback = trace_root / "traces" / f"{rows.iloc[0]['trial_id']}.npz"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not locate trace NPZ for trial {rows.iloc[0]['trial_id']!r}")


def _flatten(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        return array.reshape((-1, 1))
    return array.reshape((array.shape[0], -1))


def _summary_stats(components: list[np.ndarray]) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for component in components:
        flat = _flatten(component)
        if flat.shape[1] == 0:
            continue
        blocks.append(
            np.column_stack(
                [
                    np.mean(flat, axis=1),
                    np.std(flat, axis=1),
                    np.min(flat, axis=1),
                    np.quantile(flat, 0.25, axis=1),
                    np.median(flat, axis=1),
                    np.quantile(flat, 0.75, axis=1),
                    np.max(flat, axis=1),
                ]
            ).astype(np.float32)
        )
    if not blocks:
        length = int(components[0].shape[0]) if components else 0
        return np.zeros((length, 0), dtype=np.float32)
    return np.concatenate(blocks, axis=1).astype(np.float32)


def _stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(str(label).encode("utf-8")).digest()
    label_seed = int.from_bytes(digest[:4], byteorder="little", signed=False)
    return int((int(base_seed) + label_seed) % (2**32 - 1))


def _projection(feature_count: int, projection_dim: int, *, seed: int, label: str) -> np.ndarray:
    if feature_count <= 0 or projection_dim <= 0:
        return np.zeros((feature_count, 0), dtype=np.float32)
    rng = np.random.default_rng(_stable_seed(seed, label))
    matrix = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(float(projection_dim)),
        size=(feature_count, projection_dim),
    )
    return matrix.astype(np.float32)


def _project(raw: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[1] == 0:
        return np.zeros((raw.shape[0], 0), dtype=np.float32)
    return (raw @ matrix).astype(np.float32)


def _goal_oriented_grid(array: np.ndarray, goal_direction: int) -> np.ndarray:
    oriented = np.asarray(array, dtype=np.float32)
    if int(goal_direction) < 0 and oriented.ndim >= 5:
        oriented = np.flip(oriented, axis=-2)
    return oriented


def _centered_goal_oriented_positions(
    positions: np.ndarray,
    *,
    com_x: np.ndarray,
    com_y: np.ndarray,
    goal_direction: int,
) -> np.ndarray:
    centered = np.asarray(positions, dtype=np.float32).copy()
    centered[..., 0] = (centered[..., 0] - com_x[:, None]) * float(int(goal_direction))
    centered[..., 1] = centered[..., 1] - com_y[:, None]
    return centered


def _committed_labels(
    rows: pd.DataFrame,
    *,
    future_window: int,
    min_displacement: float,
    persistence_steps: int,
    persistence_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    com_x = rows["com_x"].to_numpy(dtype=np.float32)
    goal_direction = int(rows.iloc[0]["goal_direction"])
    oriented_com = com_x * float(goal_direction)
    count = len(rows)
    raw = np.zeros(count, dtype=np.int8)
    future_indices = np.arange(count, dtype=np.int64) + int(future_window)
    valid = future_indices < count
    deltas = np.zeros(count, dtype=np.float32)
    deltas[valid] = oriented_com[future_indices[valid]] - oriented_com[np.arange(count)[valid]]
    raw[np.logical_and(valid, deltas >= float(min_displacement))] = 1
    raw[np.logical_and(valid, deltas <= -float(min_displacement))] = -1
    labels = raw.copy()
    if int(persistence_steps) > 1:
        persisted = np.zeros_like(raw)
        for index, value in enumerate(raw):
            if value == 0:
                continue
            stop = min(count, index + int(persistence_steps))
            if stop <= index:
                continue
            same_fraction = float(np.mean(raw[index:stop] == value))
            if same_fraction >= float(persistence_fraction):
                persisted[index] = value
        labels = persisted
    labels[~valid] = 0
    return labels, deltas


def _event_index(rows: pd.DataFrame) -> tuple[int, str]:
    clearances = rows["cliff_clearance"].to_numpy(dtype=np.float32)
    hits = np.flatnonzero(clearances <= 0.0)
    if len(hits):
        return int(hits[0]), "cliff_reach"
    return int(np.argmin(clearances)), "closest_cliff_approach"


def _raw_streams(
    *,
    trace_root: Path,
    rows: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = rows.sort_values("step_index", kind="stable").reset_index(drop=True)
    trace_path = _trace_path_for_trial(trace_root, rows)
    step_count = len(rows)
    goal_direction = int(rows.iloc[0]["goal_direction"])
    channel_schema = dict(manifest.get("nca_channel_schema") or {})
    hidden_start = int(channel_schema.get("hidden_start_channel", 7))

    with np.load(trace_path) as trace:
        step_count = min(step_count, int(trace["plant_positions"].shape[0]))
        rows = rows.iloc[:step_count].reset_index(drop=True)
        nca_state = _goal_oriented_grid(trace["phase_nca_state"][:step_count], goal_direction)
        nca_hidden = nca_state[..., hidden_start:] if nca_state.shape[-1] > hidden_start else nca_state[..., 0:0]
        ctrnn_carry = _goal_oriented_grid(trace["phase_ctrnn_carry"][:step_count], goal_direction)
        signal_components = [nca_hidden, ctrnn_carry]
        signal_raw = np.concatenate([_flatten(component) for component in signal_components], axis=1)
        signal_summary = _summary_stats(signal_components)

        com_x = rows["com_x"].to_numpy(dtype=np.float32)
        com_y = rows["com_y"].to_numpy(dtype=np.float32)
        centered_positions = _centered_goal_oriented_positions(
            trace["plant_positions"][:step_count],
            com_x=com_x,
            com_y=com_y,
            goal_direction=goal_direction,
        )
        init_rest = np.asarray(trace["spring_init_rest_length"], dtype=np.float32)
        init_rest = np.where(np.abs(init_rest) > 1.0e-6, init_rest, 1.0).astype(np.float32)
        spring_current_delta = np.asarray(trace["spring_current_length"][:step_count], dtype=np.float32) / init_rest - 1.0
        spring_rest_delta = np.asarray(trace["spring_rest_length"][:step_count], dtype=np.float32) / init_rest - 1.0
        phase_rest_delta = np.asarray(trace["phase_post_spring_rest_length"][:step_count], dtype=np.float32) / init_rest - 1.0
        body_components = [
            trace["phase_h_drive"][:step_count],
            trace["phase_v_drive"][:step_count],
            trace["phase_action_outputs"][:step_count],
            spring_current_delta,
            spring_rest_delta,
            phase_rest_delta,
            centered_positions,
        ]
        body_raw = np.concatenate([_flatten(component) for component in body_components], axis=1)
        body_summary = _summary_stats(body_components)

    return (
        signal_raw.astype(np.float32),
        signal_summary.astype(np.float32),
        body_raw.astype(np.float32),
        body_summary.astype(np.float32),
    )


def _load_trials(
    *,
    trace_root: Path,
    projection_dim: int,
    random_seed: int,
    future_window: int,
    min_displacement: float,
    persistence_steps: int,
    persistence_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = _read_json(trace_root / "manifest.json")
    step_manifest = _load_step_manifest(trace_root)
    grouped = list(step_manifest.groupby("trial_id", sort=False))
    if not grouped:
        raise ValueError("state-space trace has no trials")

    first_rows = grouped[0][1].sort_values("step_index", kind="stable").reset_index(drop=True)
    first_signal_raw, _, first_body_raw, _ = _raw_streams(
        trace_root=trace_root,
        rows=first_rows,
        manifest=manifest,
    )
    signal_projection = _projection(
        first_signal_raw.shape[1],
        int(projection_dim),
        seed=int(random_seed),
        label="signal",
    )
    body_projection = _projection(
        first_body_raw.shape[1],
        int(projection_dim),
        seed=int(random_seed),
        label="body",
    )

    trials: list[dict[str, Any]] = []
    for trial_id, rows in grouped:
        rows = rows.sort_values("step_index", kind="stable").reset_index(drop=True)
        signal_raw, signal_summary, body_raw, body_summary = _raw_streams(
            trace_root=trace_root,
            rows=rows,
            manifest=manifest,
        )
        step_count = min(len(rows), signal_raw.shape[0], body_raw.shape[0])
        rows = rows.iloc[:step_count].reset_index(drop=True)
        labels, future_delta = _committed_labels(
            rows,
            future_window=int(future_window),
            min_displacement=float(min_displacement),
            persistence_steps=int(persistence_steps),
            persistence_fraction=float(persistence_fraction),
        )
        event_local_index, event_type = _event_index(rows)
        trials.append(
            {
                "trial_id": str(trial_id),
                "rows": rows,
                "signal": np.concatenate(
                    [
                        signal_summary[:step_count],
                        _project(signal_raw[:step_count], signal_projection),
                    ],
                    axis=1,
                ),
                "body": np.concatenate(
                    [
                        body_summary[:step_count],
                        _project(body_raw[:step_count], body_projection),
                    ],
                    axis=1,
                ),
                "labels": labels[:step_count],
                "future_delta": future_delta[:step_count],
                "event_local_index": event_local_index,
                "event_type": event_type,
                "committed_forward_count": int(np.sum(labels == 1)),
                "committed_backward_count": int(np.sum(labels == -1)),
                "stationary_or_uncommitted_count": int(np.sum(labels == 0)),
            }
        )

    feature_manifest = {
        "projection_dim": int(projection_dim),
        "random_seed": int(random_seed),
        "signal_raw_feature_count": int(first_signal_raw.shape[1]),
        "body_raw_feature_count": int(first_body_raw.shape[1]),
        "signal_projected_feature_count": int(signal_projection.shape[1]),
        "body_projected_feature_count": int(body_projection.shape[1]),
        "signal_final_feature_count": int(trials[0]["signal"].shape[1]),
        "body_final_feature_count": int(trials[0]["body"].shape[1]),
        "source_nca_channel_schema": dict(manifest.get("nca_channel_schema") or {}),
        "signal_feature_policy": "phase_nca_state hidden channels plus phase_ctrnn_carry; deformation/action channels excluded from nca_state",
        "body_feature_policy": "contraction drives, action outputs, spring length patterns, and centered body geometry; absolute COM and COM velocity excluded",
    }
    return trials, feature_manifest


def _lagged_xy(trial: dict[str, Any], stream: str, lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(trial[stream], dtype=np.float32)
    labels = np.asarray(trial["labels"], dtype=np.int8)
    step_indices = trial["rows"]["step_index"].to_numpy(dtype=np.int32)
    if int(lag) > 0:
        features = features[:-int(lag)]
        labels = labels[int(lag):]
        step_indices = step_indices[:-int(lag)]
    mask = labels != 0
    mask &= np.all(np.isfinite(features), axis=1)
    return features[mask], labels[mask].astype(np.int8), step_indices[mask]


def _standardize_train_test(
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale > 1.0e-6, scale, 1.0).astype(np.float32)
    return ((train - mean) / scale).astype(np.float32), ((test - mean) / scale).astype(np.float32)


def _ridge_scores(
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


def _balanced_accuracy(y_true: np.ndarray, scores: np.ndarray) -> float:
    pred = np.where(scores >= 0.0, 1, -1).astype(np.int8)
    pos = y_true == 1
    neg = y_true == -1
    if not np.any(pos) or not np.any(neg):
        return float("nan")
    tpr = float(np.mean(pred[pos] == 1))
    tnr = float(np.mean(pred[neg] == -1))
    return 0.5 * (tpr + tnr)


def _mcc(y_true: np.ndarray, scores: np.ndarray) -> float:
    pred = np.where(scores >= 0.0, 1, -1).astype(np.int8)
    tp = float(np.sum((pred == 1) & (y_true == 1)))
    tn = float(np.sum((pred == -1) & (y_true == -1)))
    fp = float(np.sum((pred == 1) & (y_true == -1)))
    fn = float(np.sum((pred == -1) & (y_true == 1)))
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denom) if denom > 0 else float("nan")


def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == -1]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    comparisons = pos_scores[:, None] - neg_scores[None, :]
    return float(np.mean(comparisons > 0.0) + 0.5 * np.mean(comparisons == 0.0))


def _metric_row(
    *,
    stream: str,
    lag: int,
    heldout_trial_id: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    null_balanced_accuracy: list[float],
) -> dict[str, Any]:
    return {
        "stream": stream,
        "lag": int(lag),
        "heldout_trial_id": heldout_trial_id,
        "sample_count": int(len(y_true)),
        "forward_count": int(np.sum(y_true == 1)),
        "backward_count": int(np.sum(y_true == -1)),
        "accuracy": float(np.mean(np.where(scores >= 0.0, 1, -1) == y_true)),
        "balanced_accuracy": _balanced_accuracy(y_true, scores),
        "auc": _auc(y_true, scores),
        "mcc": _mcc(y_true, scores),
        "null_balanced_accuracy_mean": float(np.nanmean(null_balanced_accuracy)) if null_balanced_accuracy else float("nan"),
        "null_balanced_accuracy_std": float(np.nanstd(null_balanced_accuracy)) if null_balanced_accuracy else float("nan"),
    }


def _evaluate_stream(
    trials: list[dict[str, Any]],
    *,
    stream: str,
    lags: list[int],
    ridge_alpha: float,
    permutations: int,
    null_block_size: int,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(_stable_seed(random_seed, f"decoder-null-{stream}"))
    for lag in lags:
        for heldout in trials:
            train_x_blocks: list[np.ndarray] = []
            train_y_blocks: list[np.ndarray] = []
            for trial in trials:
                if trial["trial_id"] == heldout["trial_id"]:
                    continue
                x_block, y_block, _ = _lagged_xy(trial, stream, int(lag))
                if len(y_block):
                    train_x_blocks.append(x_block)
                    train_y_blocks.append(y_block)
            test_x, test_y, test_steps = _lagged_xy(heldout, stream, int(lag))
            if not train_y_blocks or len(test_y) == 0:
                continue
            train_x = np.concatenate(train_x_blocks, axis=0)
            train_y = np.concatenate(train_y_blocks, axis=0)
            if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
                metric_rows.append(
                    {
                        "stream": stream,
                        "lag": int(lag),
                        "heldout_trial_id": heldout["trial_id"],
                        "sample_count": int(len(test_y)),
                        "forward_count": int(np.sum(test_y == 1)),
                        "backward_count": int(np.sum(test_y == -1)),
                        "accuracy": float("nan"),
                        "balanced_accuracy": float("nan"),
                        "auc": float("nan"),
                        "mcc": float("nan"),
                        "null_balanced_accuracy_mean": float("nan"),
                        "null_balanced_accuracy_std": float("nan"),
                        "skip_reason": "single_class_train_or_test",
                    }
                )
                continue
            scores = _ridge_scores(train_x, train_y, test_x, alpha=float(ridge_alpha))
            null_scores: list[float] = []
            for _ in range(int(permutations)):
                shuffled_blocks: list[np.ndarray] = []
                for block in train_y_blocks:
                    if int(null_block_size) <= 1:
                        shuffled = rng.permutation(block)
                    else:
                        chunks = [
                            block[index : index + int(null_block_size)]
                            for index in range(0, len(block), int(null_block_size))
                        ]
                        order = rng.permutation(len(chunks))
                        shuffled = np.concatenate([chunks[int(item)] for item in order])
                    shuffled_blocks.append(shuffled.astype(np.int8))
                shuffled_y = np.concatenate(shuffled_blocks, axis=0)
                shuffled_scores = _ridge_scores(train_x, shuffled_y, test_x, alpha=float(ridge_alpha))
                null_scores.append(_balanced_accuracy(test_y, shuffled_scores))
            row = _metric_row(
                stream=stream,
                lag=int(lag),
                heldout_trial_id=heldout["trial_id"],
                y_true=test_y,
                scores=scores,
                null_balanced_accuracy=null_scores,
            )
            row["skip_reason"] = ""
            metric_rows.append(row)
            if int(lag) == 0:
                rows = heldout["rows"].set_index("step_index", drop=False)
                for step_index, y_value, score in zip(test_steps, test_y, scores, strict=True):
                    source_row = rows.loc[int(step_index)]
                    prediction_rows.append(
                        {
                            "trial_id": heldout["trial_id"],
                            "stream": stream,
                            "step_index": int(step_index),
                            "env_step": int(source_row["env_step"]),
                            "label": int(y_value),
                            "score": float(score),
                            "event_type": heldout["event_type"],
                            "event_local_index": int(heldout["event_local_index"]),
                            "relative_to_event": int(step_index) - int(heldout["event_local_index"]),
                            "future_delta_goal_frame": float(heldout["future_delta"][int(step_index)]),
                        }
                    )
    return metric_rows, prediction_rows


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


def _plot_lag_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    valid = metrics[np.isfinite(metrics["balanced_accuracy"])]
    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=180)
    colors = {"signal": "#2864ad", "body": "#b04a2f"}
    for stream, group in valid.groupby("stream", sort=False):
        summary = group.groupby("lag", as_index=False).agg(
            balanced_accuracy=("balanced_accuracy", "mean"),
            null_balanced_accuracy=("null_balanced_accuracy_mean", "mean"),
        )
        ax.plot(
            summary["lag"],
            summary["balanced_accuracy"],
            marker="o",
            linewidth=2.0,
            color=colors.get(str(stream), None),
            label=f"{stream} decoder",
        )
        ax.plot(
            summary["lag"],
            summary["null_balanced_accuracy"],
            linestyle="--",
            linewidth=1.4,
            color=colors.get(str(stream), None),
            alpha=0.7,
            label=f"{stream} shuffled null",
        )
    ax.axhline(0.5, color="#666666", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Prediction lag, env steps")
    ax.set_ylabel("Leave-one-trial-out balanced accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_trial_panels(
    *,
    trials: list[dict[str, Any]],
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    stream_colors = {"signal": "#2864ad", "body": "#b04a2f"}
    fig, axes = plt.subplots(
        len(trials),
        3,
        figsize=(12.0, max(2.0 * len(trials), 4.0)),
        dpi=170,
        sharex=False,
    )
    if len(trials) == 1:
        axes = np.asarray([axes])
    for row_index, trial in enumerate(trials):
        rows = trial["rows"]
        step = rows["step_index"].to_numpy(dtype=np.int32)
        oriented_com = rows["com_x"].to_numpy(dtype=np.float32) * float(int(rows.iloc[0]["goal_direction"]))
        labels = trial["labels"]
        ax0, ax1, ax2 = axes[row_index]
        ax0.plot(step, oriented_com, color="#222222", linewidth=1.3)
        ax0.scatter(step[labels == 1], oriented_com[labels == 1], s=3, color="#2b7a4b", alpha=0.65)
        ax0.scatter(step[labels == -1], oriented_com[labels == -1], s=3, color="#7a2b5e", alpha=0.65)
        ax0.axvline(trial["event_local_index"], color="#777777", linestyle="--", linewidth=1.0)
        ax0.set_ylabel(trial["trial_id"], fontsize=7)
        ax0.grid(True, alpha=0.18)
        ax0.set_title("Committed movement", fontsize=9) if row_index == 0 else None

        for ax, stream in ((ax1, "signal"), (ax2, "body")):
            subset = predictions[
                (predictions["trial_id"] == trial["trial_id"]) & (predictions["stream"] == stream)
            ]
            if not subset.empty:
                ax.plot(
                    subset["step_index"],
                    subset["score"],
                    color=stream_colors[stream],
                    linewidth=1.1,
                )
                ax.scatter(
                    subset.loc[subset["label"] == 1, "step_index"],
                    subset.loc[subset["label"] == 1, "score"],
                    s=3,
                    color="#2b7a4b",
                    alpha=0.55,
                )
                ax.scatter(
                    subset.loc[subset["label"] == -1, "step_index"],
                    subset.loc[subset["label"] == -1, "score"],
                    s=3,
                    color="#7a2b5e",
                    alpha=0.55,
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "single-class held-out",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#555555",
                    transform=ax.transAxes,
                )
                ax.set_xlim(float(step[0]), float(step[-1]))
            ax.axhline(0.0, color="#777777", linestyle=":", linewidth=0.9)
            ax.axvline(trial["event_local_index"], color="#777777", linestyle="--", linewidth=1.0)
            ax.grid(True, alpha=0.18)
            ax.set_title(f"{stream} held-out score", fontsize=9) if row_index == 0 else None
    for ax in axes[-1]:
        ax.set_xlabel("Env step")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def analyze_committed_direction(
    trace_root: Path,
    output_dir: Path,
    *,
    future_window: int,
    min_displacement: float,
    persistence_steps: int,
    persistence_fraction: float,
    projection_dim: int,
    lags: list[int],
    ridge_alpha: float,
    permutations: int,
    null_block_size: int,
    random_seed: int,
) -> dict[str, Any]:
    trace_root = trace_root.resolve()
    output_dir = output_dir.resolve()
    trials, feature_manifest = _load_trials(
        trace_root=trace_root,
        projection_dim=int(projection_dim),
        random_seed=int(random_seed),
        future_window=int(future_window),
        min_displacement=float(min_displacement),
        persistence_steps=int(persistence_steps),
        persistence_fraction=float(persistence_fraction),
    )
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for stream in ("signal", "body"):
        stream_metrics, stream_predictions = _evaluate_stream(
            trials,
            stream=stream,
            lags=[int(value) for value in lags],
            ridge_alpha=float(ridge_alpha),
            permutations=int(permutations),
            null_block_size=int(null_block_size),
            random_seed=int(random_seed),
        )
        metric_rows.extend(stream_metrics)
        prediction_rows.extend(stream_predictions)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "direction_decoder_metrics.csv"
    predictions_path = output_dir / "direction_decoder_predictions_lag0.csv"
    event_response_path = output_dir / "event_aligned_response.csv"
    _write_csv(metrics_path, metric_rows)
    _write_csv(predictions_path, prediction_rows)
    _write_csv(event_response_path, prediction_rows)

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    if not metrics_df.empty:
        _plot_lag_metrics(metrics_df, output_dir / "direction_decoder_by_lag.png")
    if not predictions_df.empty:
        _plot_trial_panels(
            trials=trials,
            predictions=predictions_df,
            output_path=output_dir / "per_trial_committed_direction_panels.png",
        )
        _plot_trial_panels(
            trials=trials,
            predictions=predictions_df[
                predictions_df["relative_to_event"].between(-240, 360)
            ],
            output_path=output_dir / "signal_body_event_response.png",
        )

    valid_metrics = metrics_df[np.isfinite(metrics_df["balanced_accuracy"])] if not metrics_df.empty else metrics_df
    aggregate_rows = []
    if valid_metrics is not None and not valid_metrics.empty:
        for (stream, lag), group in valid_metrics.groupby(["stream", "lag"], sort=True):
            aggregate_rows.append(
                {
                    "stream": str(stream),
                    "lag": int(lag),
                    "heldout_trial_count": int(group["heldout_trial_id"].nunique()),
                    "mean_balanced_accuracy": float(group["balanced_accuracy"].mean()),
                    "std_balanced_accuracy": float(group["balanced_accuracy"].std(ddof=0)),
                    "mean_auc": float(group["auc"].mean()),
                    "mean_mcc": float(group["mcc"].mean()),
                    "mean_null_balanced_accuracy": float(group["null_balanced_accuracy_mean"].mean()),
                }
            )
    aggregate_path = output_dir / "direction_decoder_summary.csv"
    _write_csv(aggregate_path, aggregate_rows)
    summary = {
        "schema_version": 1,
        "kind": "committed_direction_decoding",
        "trace_root": str(trace_root),
        "output_dir": str(output_dir),
        "label_policy": {
            "axis": "goal-facing COM displacement",
            "future_window": int(future_window),
            "min_displacement": float(min_displacement),
            "persistence_steps": int(persistence_steps),
            "persistence_fraction": float(persistence_fraction),
            "label_values": {"1": "committed toward goal", "-1": "committed away from goal", "0": "uncommitted_or_stationary"},
        },
        "feature_manifest": feature_manifest,
        "lags": [int(value) for value in lags],
        "ridge_alpha": float(ridge_alpha),
        "permutations": int(permutations),
        "null_block_size": int(null_block_size),
        "trial_labels": [
            {
                "trial_id": trial["trial_id"],
                "step_count": int(len(trial["rows"])),
                "committed_forward_count": int(trial["committed_forward_count"]),
                "committed_backward_count": int(trial["committed_backward_count"]),
                "stationary_or_uncommitted_count": int(trial["stationary_or_uncommitted_count"]),
                "event_type": trial["event_type"],
                "event_local_index": int(trial["event_local_index"]),
            }
            for trial in trials
        ],
        "aggregate": aggregate_rows,
        "outputs": {
            "direction_decoder_metrics": str(metrics_path),
            "direction_decoder_summary": str(aggregate_path),
            "direction_decoder_predictions_lag0": str(predictions_path),
            "event_aligned_response": str(event_response_path),
            "direction_decoder_by_lag_png": str(output_dir / "direction_decoder_by_lag.png"),
            "per_trial_committed_direction_panels_png": str(output_dir / "per_trial_committed_direction_panels.png"),
            "signal_body_event_response_png": str(output_dir / "signal_body_event_response.png"),
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
        description="Test whether signal or body features decode committed movement direction."
    )
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--future-window", type=int, default=32)
    parser.add_argument("--min-displacement", type=float, default=0.025)
    parser.add_argument("--persistence-steps", type=int, default=12)
    parser.add_argument("--persistence-fraction", type=float, default=0.75)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--lags", type=_parse_lags, default=_parse_lags("0,4,8,16,32,64"))
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--permutations", type=int, default=25)
    parser.add_argument("--null-block-size", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=137)
    args = parser.parse_args()

    summary = analyze_committed_direction(
        args.trace_root,
        args.output_dir,
        future_window=int(args.future_window),
        min_displacement=float(args.min_displacement),
        persistence_steps=int(args.persistence_steps),
        persistence_fraction=float(args.persistence_fraction),
        projection_dim=int(args.projection_dim),
        lags=list(args.lags),
        ridge_alpha=float(args.ridge_alpha),
        permutations=int(args.permutations),
        null_block_size=int(args.null_block_size),
        random_seed=int(args.random_seed),
    )
    print(json.dumps(summary["outputs"], indent=2))


if __name__ == "__main__":
    main()
