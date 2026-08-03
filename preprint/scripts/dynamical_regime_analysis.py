#!/usr/bin/env python
"""Event-locked dynamical-regime analysis for deformation-off dense traces."""

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
    _event_index,
    _mcc,
    _projection,
    _project,
    _raw_streams,
    _read_json,
    _ridge_scores,
    _stable_seed,
    _load_step_manifest,
    _write_csv,
    _write_json,
)


DEFAULT_TRACE_ROOT = Path(
    "paper_data/preprint/native_core_ablation_v1/"
    "raw_modal_state_space_deformation_off_20260608T013533/"
    "paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed137_e4fcc2fb/"
    "bin_03_rank_00_cell_6-0-3_flat_0423/"
    "state_space_trace_deformation_off"
)
DEFAULT_OUTPUT_DIR = Path(
    "paper_data/preprint/native_core_ablation_v1/figures/dynamical_regime_deformation_off"
)

VALID_BINARY_TRIALS = (
    "pair_2_1_left_000",
    "pair_2_1_right_000",
    "pair_4_2_right_000",
    "pair_6_3_left_000",
    "pair_6_3_right_000",
)
EXCLUDED_SINGLE_CLASS_TRIAL = "pair_4_2_left_000"
STREAM_COLORS = {"signal": "#2864ad", "body": "#b04a2f"}
STATE_COLORS = {
    "committed_toward": "#2b7a4b",
    "committed_away": "#7a2b5e",
    "meander_toward": "#78b892",
    "meander_away": "#c57aa6",
    "uncommitted": "#c9c9c9",
}
WINDOWS = {
    "pre_baseline": (-256, -32),
    "event_impulse": (-8, 24),
    "early_ramification": (24, 160),
    "late_ramification": (160, 512),
}


def _standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale > 1.0e-6, scale, 1.0).astype(np.float32)
    return ((features - mean) / scale).astype(np.float32), mean, scale


def _fit_pca(features: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, dict[str, Any]]:
    z, _mean, _scale = _standardize(np.asarray(features, dtype=np.float32))
    centered = z - z.mean(axis=0, dtype=np.float64).astype(np.float32)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[: int(n_components)].astype(np.float32)
    denom = max(centered.shape[0] - 1, 1)
    variances = (singular_values.astype(np.float64) ** 2) / denom
    explained = variances[: int(n_components)] / np.maximum(np.sum(variances), 1.0e-12)
    coords = centered @ components.T
    return coords.astype(np.float32), {
        "sample_count": int(features.shape[0]),
        "feature_count": int(features.shape[1]),
        "explained_variance_ratio": [float(value) for value in explained],
    }


def _meander_labels(
    rows: pd.DataFrame,
    committed: np.ndarray,
    *,
    window: int,
    min_displacement: float,
) -> tuple[np.ndarray, np.ndarray]:
    oriented_com = rows["com_x"].to_numpy(dtype=np.float32) * float(int(rows.iloc[0]["goal_direction"]))
    count = len(rows)
    local_delta = np.zeros((count,), dtype=np.float32)
    labels = np.zeros((count,), dtype=np.int8)
    future = np.arange(count, dtype=np.int64) + int(window)
    valid = future < count
    local_delta[valid] = oriented_com[future[valid]] - oriented_com[np.arange(count)[valid]]
    toward = np.logical_and(valid, local_delta >= float(min_displacement))
    away = np.logical_and(valid, local_delta <= -float(min_displacement))
    labels[np.logical_and(toward, committed == 0)] = 1
    labels[np.logical_and(away, committed == 0)] = -1
    return labels, local_delta


def _state_class(committed_label: int, meander_label: int) -> str:
    if int(committed_label) == 1:
        return "committed_toward"
    if int(committed_label) == -1:
        return "committed_away"
    if int(meander_label) == 1:
        return "meander_toward"
    if int(meander_label) == -1:
        return "meander_away"
    return "uncommitted"


def _load_trials(
    *,
    trace_root: Path,
    projection_dim: int,
    random_seed: int,
    future_window: int,
    min_displacement: float,
    persistence_steps: int,
    persistence_fraction: float,
    meander_window: int,
    meander_min_displacement: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = _read_json(trace_root / "manifest.json")
    step_manifest = _load_step_manifest(trace_root)
    grouped = list(step_manifest.groupby("trial_id", sort=False))
    if not grouped:
        raise ValueError("state-space trace has no trials")

    first_rows = grouped[0][1].sort_values("step_index", kind="stable").reset_index(drop=True)
    first_signal_raw, _first_signal_summary, first_body_raw, _first_body_summary = _raw_streams(
        trace_root=trace_root,
        rows=first_rows,
        manifest=manifest,
    )
    signal_projection = _projection(
        first_signal_raw.shape[1],
        int(projection_dim),
        seed=int(random_seed),
        label="signal-regime",
    )
    body_projection = _projection(
        first_body_raw.shape[1],
        int(projection_dim),
        seed=int(random_seed),
        label="body-regime",
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
        committed, future_delta = _committed_labels(
            rows,
            future_window=int(future_window),
            min_displacement=float(min_displacement),
            persistence_steps=int(persistence_steps),
            persistence_fraction=float(persistence_fraction),
        )
        meander, local_delta = _meander_labels(
            rows,
            committed,
            window=int(meander_window),
            min_displacement=float(meander_min_displacement),
        )
        event_local_index, event_type = _event_index(rows)
        signal = np.concatenate(
            [
                signal_summary[:step_count],
                _project(signal_raw[:step_count], signal_projection),
            ],
            axis=1,
        )
        body = np.concatenate(
            [
                body_summary[:step_count],
                _project(body_raw[:step_count], body_projection),
            ],
            axis=1,
        )
        trials.append(
            {
                "trial_id": str(trial_id),
                "rows": rows,
                "signal": signal.astype(np.float32),
                "body": body.astype(np.float32),
                "labels": committed[:step_count],
                "meander_labels": meander[:step_count],
                "future_delta": future_delta[:step_count],
                "local_delta": local_delta[:step_count],
                "event_local_index": int(event_local_index),
                "event_type": str(event_type),
                "valid_binary": str(trial_id) in VALID_BINARY_TRIALS,
                "true_cliff_reach": str(event_type) == "cliff_reach",
                "committed_forward_count": int(np.sum(committed == 1)),
                "committed_backward_count": int(np.sum(committed == -1)),
                "meander_forward_count": int(np.sum(meander == 1)),
                "meander_backward_count": int(np.sum(meander == -1)),
            }
        )

    feature_manifest = {
        "projection_dim": int(projection_dim),
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


def _baseline_distance(
    features: np.ndarray,
    *,
    center: int,
    pre_window: int,
    pre_guard: int,
    post_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start = int(center) - int(pre_window)
    stop = int(center) + int(post_window) + 1
    baseline_start = int(center) - int(pre_window)
    baseline_stop = int(center) - int(pre_guard)
    if baseline_start < 0 or baseline_stop <= baseline_start or stop > features.shape[0]:
        raise ValueError("insufficient baseline or post window")
    baseline = features[baseline_start:baseline_stop]
    mean = baseline.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = baseline.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale > 1.0e-6, scale, 1.0).astype(np.float32)
    window_features = features[start:stop]
    z = (window_features - mean) / scale
    rms = np.sqrt(np.mean(z * z, axis=1)).astype(np.float32)
    rel_steps = np.arange(start, stop, dtype=np.int32) - int(center)
    baseline_rms = rms[rel_steps < -int(pre_guard)]
    return rel_steps, rms, baseline_rms, z


def _candidate_null_centers(
    *,
    count: int,
    event_index: int,
    pre_window: int,
    post_window: int,
    exclusion: int,
    limit: int,
    seed: int,
    trial_id: str,
) -> list[int]:
    low = int(pre_window)
    high = int(count) - int(post_window) - 1
    if high <= low:
        return []
    centers = np.arange(low, high + 1, dtype=np.int32)
    centers = centers[np.abs(centers - int(event_index)) >= int(exclusion)]
    if centers.size == 0:
        return []
    rng = np.random.default_rng(_stable_seed(seed, f"null-centers-{trial_id}"))
    if centers.size > int(limit):
        centers = rng.choice(centers, size=int(limit), replace=False)
    return [int(value) for value in np.sort(centers)]


def _empirical_p_greater(observed: float, null_values: np.ndarray) -> float:
    null_values = np.asarray(null_values, dtype=np.float64)
    null_values = null_values[np.isfinite(null_values)]
    if null_values.size == 0 or not np.isfinite(observed):
        return float("nan")
    return float((1.0 + np.sum(null_values >= float(observed))) / (float(null_values.size) + 1.0))


def _empirical_p_two_sided(observed: float, null_values: np.ndarray) -> float:
    null_values = np.asarray(null_values, dtype=np.float64)
    null_values = null_values[np.isfinite(null_values)]
    if null_values.size == 0 or not np.isfinite(observed):
        return float("nan")
    null_mean = float(np.mean(null_values))
    observed_distance = abs(float(observed) - null_mean)
    null_distance = np.abs(null_values - null_mean)
    return float((1.0 + np.sum(null_distance >= observed_distance)) / (float(null_values.size) + 1.0))


def _empirical_percentile(observed: float, null_values: np.ndarray) -> float:
    null_values = np.asarray(null_values, dtype=np.float64)
    null_values = null_values[np.isfinite(null_values)]
    if null_values.size == 0 or not np.isfinite(observed):
        return float("nan")
    return float(np.mean(null_values <= float(observed)))


def _event_perturbation(
    trials: list[dict[str, Any]],
    *,
    pre_window: int,
    pre_guard: int,
    post_window: int,
    null_count: int,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    timeseries_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    persistence_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    for trial in trials:
        if not bool(trial["true_cliff_reach"]):
            continue
        event_index = int(trial["event_local_index"])
        if event_index < int(pre_window) or event_index + int(post_window) >= len(trial["rows"]):
            continue
        for stream in ("signal", "body"):
            rel_steps, rms, baseline_rms, _z = _baseline_distance(
                trial[stream],
                center=event_index,
                pre_window=int(pre_window),
                pre_guard=int(pre_guard),
                post_window=int(post_window),
            )
            baseline_mean = float(np.mean(baseline_rms))
            baseline_threshold = float(np.quantile(baseline_rms, 0.95))
            for rel, value in zip(rel_steps, rms, strict=True):
                source_index = int(event_index + int(rel))
                row = trial["rows"].iloc[source_index]
                timeseries_rows.append(
                    {
                        "trial_id": trial["trial_id"],
                        "stream": stream,
                        "relative_step": int(rel),
                        "step_index": source_index,
                        "env_step": int(row["env_step"]),
                        "z_rms": float(value),
                        "delta_from_pre_mean": float(value - baseline_mean),
                        "cliff_clearance": float(row["cliff_clearance"]),
                        "committed_label": int(trial["labels"][source_index]),
                        "meander_label": int(trial["meander_labels"][source_index]),
                    }
                )
            for window_name, (start, stop) in WINDOWS.items():
                mask = (rel_steps >= int(start)) & (rel_steps <= int(stop))
                if not np.any(mask):
                    continue
                summary_rows.append(
                    {
                        "cohort": "true_cliff_reach",
                        "trial_id": trial["trial_id"],
                        "stream": stream,
                        "window": window_name,
                        "relative_start": int(start),
                        "relative_stop": int(stop),
                        "mean_z_rms": float(np.mean(rms[mask])),
                        "delta_from_pre_mean": float(np.mean(rms[mask]) - baseline_mean),
                        "baseline_mean_z_rms": baseline_mean,
                    }
                )
            post_mask = (rel_steps >= 0) & (rel_steps <= int(post_window))
            post_rel = rel_steps[post_mask]
            post_above = rms[post_mask] > baseline_threshold
            if post_above.size:
                last_above = int(post_rel[np.flatnonzero(post_above)[-1]]) if np.any(post_above) else 0
                early_mask = (post_rel >= 24) & (post_rel <= 160)
                late_mask = (post_rel >= 160) & (post_rel <= int(post_window))
                persistence_rows.append(
                    {
                        "trial_id": trial["trial_id"],
                        "stream": stream,
                        "threshold": baseline_threshold,
                        "last_relative_step_above_threshold": last_above,
                        "fraction_above_0_160": float(np.mean(post_above[post_rel <= 160])),
                        "fraction_above_24_160": float(np.mean(post_above[early_mask])) if np.any(early_mask) else float("nan"),
                        "fraction_above_160_post": float(np.mean(post_above[late_mask])) if np.any(late_mask) else float("nan"),
                    }
                )

            centers = _candidate_null_centers(
                count=len(trial["rows"]),
                event_index=event_index,
                pre_window=int(pre_window),
                post_window=int(post_window),
                exclusion=256,
                limit=int(null_count),
                seed=int(random_seed),
                trial_id=str(trial["trial_id"]),
            )
            for center in centers:
                null_rel, null_rms, null_baseline, _null_z = _baseline_distance(
                    trial[stream],
                    center=center,
                    pre_window=int(pre_window),
                    pre_guard=int(pre_guard),
                    post_window=int(post_window),
                )
                null_baseline_mean = float(np.mean(null_baseline))
                for window_name, (start, stop) in WINDOWS.items():
                    mask = (null_rel >= int(start)) & (null_rel <= int(stop))
                    if not np.any(mask):
                        continue
                    null_rows.append(
                        {
                            "trial_id": trial["trial_id"],
                            "stream": stream,
                            "window": window_name,
                            "null_center_step_index": int(center),
                            "mean_z_rms": float(np.mean(null_rms[mask])),
                            "delta_from_pre_mean": float(np.mean(null_rms[mask]) - null_baseline_mean),
                            "baseline_mean_z_rms": null_baseline_mean,
                        }
                    )
    return timeseries_rows, summary_rows, persistence_rows, null_rows


def _autocorr_peak(values: np.ndarray, *, min_lag: int = 16, max_lag: int = 300) -> tuple[int | None, float | None]:
    values = np.asarray(values, dtype=np.float32)
    values = values - float(np.mean(values))
    denom = float(np.dot(values, values))
    if denom <= 1.0e-12 or values.shape[0] <= min_lag + 2:
        return None, None
    upper = min(int(max_lag), values.shape[0] - 2)
    if upper <= int(min_lag):
        return None, None
    scores = []
    lags = np.arange(int(min_lag), upper + 1, dtype=np.int32)
    for lag in lags:
        scores.append(float(np.dot(values[:-lag], values[lag:]) / denom))
    scores_arr = np.asarray(scores, dtype=np.float32)
    idx = int(np.argmax(scores_arr))
    return int(lags[idx]), float(scores_arr[idx])


def _signal_cycle_metrics(trials: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    valid_trials = [trial for trial in trials if bool(trial["valid_binary"])]
    signal_features = np.concatenate([trial["signal"] for trial in valid_trials], axis=0)
    coords, model = _fit_pca(signal_features, n_components=3)
    coord_by_trial: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    offset = 0
    for trial in valid_trials:
        count = int(trial["signal"].shape[0])
        trial_coords = coords[offset : offset + count]
        coord_by_trial[str(trial["trial_id"])] = trial_coords
        offset += count
        phase = np.unwrap(np.arctan2(trial_coords[:, 1], trial_coords[:, 0]))
        angular_velocity = np.diff(phase)
        radius = np.sqrt(np.sum(trial_coords[:, :2] * trial_coords[:, :2], axis=1))
        dominant_fraction = float("nan")
        if angular_velocity.size:
            signs = np.sign(angular_velocity[np.abs(angular_velocity) > 1.0e-4])
            if signs.size:
                dominant_fraction = float(max(np.mean(signs > 0), np.mean(signs < 0)))
        peak_lag, peak_value = _autocorr_peak(trial_coords[:, 0])
        event_index = int(trial["event_local_index"])
        pre_speed = float("nan")
        post_speed = float("nan")
        if bool(trial["true_cliff_reach"]) and event_index >= 256 and event_index + 160 < count:
            pre = np.abs(angular_velocity[max(0, event_index - 256) : max(0, event_index - 32)])
            post = np.abs(angular_velocity[event_index : event_index + 160])
            pre_speed = float(np.mean(pre)) if pre.size else float("nan")
            post_speed = float(np.mean(post)) if post.size else float("nan")
        rows.append(
            {
                "trial_id": trial["trial_id"],
                "event_type": trial["event_type"],
                "step_count": count,
                "radius_mean": float(np.mean(radius)),
                "radius_cv": float(np.std(radius) / np.maximum(np.mean(radius), 1.0e-12)),
                "angular_velocity_abs_median": float(np.median(np.abs(angular_velocity))) if angular_velocity.size else float("nan"),
                "angular_velocity_abs_iqr": float(np.quantile(np.abs(angular_velocity), 0.75) - np.quantile(np.abs(angular_velocity), 0.25)) if angular_velocity.size else float("nan"),
                "dominant_rotation_fraction": dominant_fraction,
                "pc1_autocorr_peak_lag": peak_lag,
                "pc1_autocorr_peak_value": peak_value,
                "pre_event_abs_angular_velocity_mean": pre_speed,
                "post_event_abs_angular_velocity_mean": post_speed,
                "post_over_pre_abs_angular_velocity": (post_speed / pre_speed) if np.isfinite(pre_speed) and pre_speed > 0 else float("nan"),
            }
        )
    return rows, coord_by_trial, model


def _lagged_event_xy(
    trial: dict[str, Any],
    stream: str,
    lag: int,
    *,
    rel_start: int,
    rel_stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(trial[stream], dtype=np.float32)
    labels = np.asarray(trial["labels"], dtype=np.int8)
    event_index = int(trial["event_local_index"])
    indices = np.arange(features.shape[0] - int(lag), dtype=np.int64)
    rel = indices - event_index
    mask = (rel >= int(rel_start)) & (rel <= int(rel_stop))
    y = labels[indices + int(lag)]
    mask &= y != 0
    x = features[indices]
    mask &= np.all(np.isfinite(x), axis=1)
    return x[mask], y[mask].astype(np.int8)


def _evaluate_future_commitment(
    trials: list[dict[str, Any]],
    *,
    cohort_name: str,
    lags: list[int],
    rel_start: int,
    rel_stop: int,
    ridge_alpha: float,
    random_seed: int,
    permutations: int,
    null_block_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(_stable_seed(random_seed, f"future-{cohort_name}"))
    for stream in ("signal", "body"):
        for lag in lags:
            for heldout in trials:
                train_x_blocks: list[np.ndarray] = []
                train_y_blocks: list[np.ndarray] = []
                for trial in trials:
                    if trial["trial_id"] == heldout["trial_id"]:
                        continue
                    x_block, y_block = _lagged_event_xy(
                        trial,
                        stream,
                        int(lag),
                        rel_start=int(rel_start),
                        rel_stop=int(rel_stop),
                    )
                    if y_block.size:
                        train_x_blocks.append(x_block)
                        train_y_blocks.append(y_block)
                test_x, test_y = _lagged_event_xy(
                    heldout,
                    stream,
                    int(lag),
                    rel_start=int(rel_start),
                    rel_stop=int(rel_stop),
                )
                base = {
                    "cohort": cohort_name,
                    "stream": stream,
                    "lag": int(lag),
                    "heldout_trial_id": heldout["trial_id"],
                    "relative_start": int(rel_start),
                    "relative_stop": int(rel_stop),
                    "sample_count": int(test_y.size),
                    "forward_count": int(np.sum(test_y == 1)),
                    "backward_count": int(np.sum(test_y == -1)),
                }
                if not train_y_blocks or test_y.size == 0 or len(np.unique(test_y)) < 2:
                    rows.append({**base, "skip_reason": "single_class_or_empty_test"})
                    continue
                train_x = np.concatenate(train_x_blocks, axis=0)
                train_y = np.concatenate(train_y_blocks, axis=0)
                if len(np.unique(train_y)) < 2:
                    rows.append({**base, "skip_reason": "single_class_train"})
                    continue
                scores = _ridge_scores(train_x, train_y, test_x, alpha=float(ridge_alpha))
                null_scores = []
                for _ in range(int(permutations)):
                    shuffled_blocks: list[np.ndarray] = []
                    for block in train_y_blocks:
                        if int(null_block_size) <= 1:
                            shuffled_blocks.append(rng.permutation(block).astype(np.int8))
                        else:
                            chunks = [
                                block[index : index + int(null_block_size)]
                                for index in range(0, len(block), int(null_block_size))
                            ]
                            order = rng.permutation(len(chunks))
                            shuffled_blocks.append(np.concatenate([chunks[int(item)] for item in order]).astype(np.int8))
                    shuffled_y = np.concatenate(shuffled_blocks, axis=0)
                    shuffled_scores = _ridge_scores(train_x, shuffled_y, test_x, alpha=float(ridge_alpha))
                    null_scores.append(_balanced_accuracy(test_y, shuffled_scores))
                actual_balanced_accuracy = _balanced_accuracy(test_y, scores)
                null_scores_arr = np.asarray(null_scores, dtype=np.float32)
                rows.append(
                    {
                        **base,
                        "accuracy": float(np.mean(np.where(scores >= 0.0, 1, -1) == test_y)),
                        "balanced_accuracy": actual_balanced_accuracy,
                        "auc": _auc(test_y, scores),
                        "mcc": _mcc(test_y, scores),
                        "null_balanced_accuracy_mean": float(np.nanmean(null_scores_arr)),
                        "null_balanced_accuracy_std": float(np.nanstd(null_scores_arr)),
                        "null_balanced_accuracy_q05": float(np.nanquantile(null_scores_arr, 0.05)),
                        "null_balanced_accuracy_q95": float(np.nanquantile(null_scores_arr, 0.95)),
                        "actual_minus_null_balanced_accuracy": float(actual_balanced_accuracy - np.nanmean(null_scores_arr)),
                        "empirical_p_greater_equal": _empirical_p_greater(actual_balanced_accuracy, null_scores_arr),
                        "skip_reason": "",
                    }
                )
    return rows


def _write_meander_rows(path: Path, trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in trials:
        if not bool(trial["valid_binary"]):
            continue
        step_rows = trial["rows"]
        for index, row in step_rows.iterrows():
            committed = int(trial["labels"][index])
            meander = int(trial["meander_labels"][index])
            rows.append(
                {
                    "trial_id": trial["trial_id"],
                    "env_step": int(row["env_step"]),
                    "step_index": int(row["step_index"]),
                    "event_type": trial["event_type"],
                    "event_local_index": int(trial["event_local_index"]),
                    "relative_to_event": int(index) - int(trial["event_local_index"]),
                    "goal_direction": int(row["goal_direction"]),
                    "goal_frame_com_x": float(row["com_x"]) * float(int(row["goal_direction"])),
                    "cliff_clearance": float(row["cliff_clearance"]),
                    "committed_label": committed,
                    "meander_label": meander,
                    "state_class": _state_class(committed, meander),
                    "future_delta_goal_frame": float(trial["future_delta"][index]),
                    "local_delta_goal_frame": float(trial["local_delta"][index]),
                }
            )
    _write_csv(path, rows)
    return rows


def _aggregate_future(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    if frame.empty or "balanced_accuracy" not in frame.columns:
        return []
    valid = frame[np.isfinite(frame["balanced_accuracy"])]
    out: list[dict[str, Any]] = []
    for (cohort, stream, lag), group in valid.groupby(["cohort", "stream", "lag"], sort=True):
        out.append(
            {
                "cohort": str(cohort),
                "stream": str(stream),
                "lag": int(lag),
                "heldout_trial_count": int(group["heldout_trial_id"].nunique()),
                "mean_balanced_accuracy": float(group["balanced_accuracy"].mean()),
                "mean_auc": float(group["auc"].mean()),
                "mean_mcc": float(group["mcc"].mean()),
                "mean_null_balanced_accuracy": float(group["null_balanced_accuracy_mean"].mean()),
                "mean_actual_minus_null_balanced_accuracy": float(group["actual_minus_null_balanced_accuracy"].mean()),
                "mean_empirical_p_greater_equal": float(group["empirical_p_greater_equal"].mean()),
            }
        )
    return out


def _null_control_summary(
    actual_rows: list[dict[str, Any]],
    null_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    actual = pd.DataFrame(actual_rows)
    null = pd.DataFrame(null_rows)
    if actual.empty:
        return []
    out: list[dict[str, Any]] = []
    for (stream, window), group in actual.groupby(["stream", "window"], sort=True):
        null_group = null[(null["stream"] == stream) & (null["window"] == window)] if not null.empty else pd.DataFrame()
        actual_delta = float(group["delta_from_pre_mean"].mean())
        actual_z_rms = float(group["mean_z_rms"].mean())
        null_delta_values = (
            null_group["delta_from_pre_mean"].to_numpy(dtype=np.float64)
            if not null_group.empty
            else np.asarray([], dtype=np.float64)
        )
        null_z_values = (
            null_group["mean_z_rms"].to_numpy(dtype=np.float64)
            if not null_group.empty
            else np.asarray([], dtype=np.float64)
        )
        null_delta_mean = float(np.mean(null_delta_values)) if null_delta_values.size else float("nan")
        null_z_mean = float(np.mean(null_z_values)) if null_z_values.size else float("nan")
        out.append(
            {
                "stream": str(stream),
                "window": str(window),
                "actual_trial_count": int(group["trial_id"].nunique()),
                "actual_mean_delta": actual_delta,
                "actual_mean_z_rms": actual_z_rms,
                "null_window_count": int(len(null_group)),
                "null_mean_delta": null_delta_mean,
                "null_delta_std": float(np.std(null_delta_values)) if null_delta_values.size else float("nan"),
                "null_delta_q05": float(np.quantile(null_delta_values, 0.05)) if null_delta_values.size else float("nan"),
                "null_delta_q50": float(np.quantile(null_delta_values, 0.50)) if null_delta_values.size else float("nan"),
                "null_delta_q95": float(np.quantile(null_delta_values, 0.95)) if null_delta_values.size else float("nan"),
                "null_mean_z_rms": null_z_mean,
                "actual_minus_null_delta": float(actual_delta - null_delta_mean) if np.isfinite(null_delta_mean) else float("nan"),
                "actual_delta_null_percentile": _empirical_percentile(actual_delta, null_delta_values),
                "empirical_p_greater_equal": _empirical_p_greater(actual_delta, null_delta_values),
                "empirical_p_two_sided": _empirical_p_two_sided(actual_delta, null_delta_values),
            }
        )
    return out


def _plot_event_shift(timeseries: pd.DataFrame, null_summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=180)
    for stream, group in timeseries.groupby("stream", sort=False):
        summary = group.groupby("relative_step", as_index=False).agg(z_rms=("z_rms", "mean"))
        smoothed = summary["z_rms"].rolling(window=9, center=True, min_periods=1).mean()
        ax.plot(
            summary["relative_step"],
            summary["z_rms"],
            color=STREAM_COLORS.get(stream),
            linewidth=0.8,
            alpha=0.22,
        )
        ax.plot(
            summary["relative_step"],
            smoothed,
            color=STREAM_COLORS.get(stream),
            linewidth=2.0,
                    label=f"{stream} true-cliff aligned",
        )
        if not null_summary.empty:
            early_null = null_summary[
                (null_summary["stream"] == stream) & (null_summary["window"] == "early_ramification")
            ]
            if not early_null.empty:
                ax.axhline(
                    float(early_null["null_mean_z_rms"].iloc[0]),
                    color=STREAM_COLORS.get(stream),
                    linestyle="--",
                    alpha=0.55,
                    linewidth=1.2,
                    label=f"{stream} matched-null early mean",
                )
    ax.axvline(0, color="#333333", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Env steps relative to true cliff reach")
    ax.set_ylabel("RMS z-distance from pre-event baseline")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_persistence(persistence: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=180)
    labels = []
    values = []
    colors = []
    for stream in ("signal", "body"):
        group = persistence[persistence["stream"] == stream]
        labels.extend([f"{stream}\n0-160", f"{stream}\n24-160", f"{stream}\n160-512"])
        values.extend(
            [
                float(group["fraction_above_0_160"].mean()) if not group.empty else 0.0,
                float(group["fraction_above_24_160"].mean()) if not group.empty else 0.0,
                float(group["fraction_above_160_post"].mean()) if not group.empty else 0.0,
            ]
        )
        colors.extend([STREAM_COLORS.get(stream)] * 3)
    ax.bar(np.arange(len(values)), values, color=colors, alpha=0.84)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction above 95% pre-event baseline")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_future_prediction(summary: pd.DataFrame, output_path: Path) -> None:
    valid = summary[np.isfinite(summary["mean_balanced_accuracy"])] if not summary.empty else summary
    cohorts = [
        ("all_valid_aligned_window", "All five valid aligned trials"),
        ("true_cliff_reach_event_window", "True cliff-reach trials"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=180, sharey=True)
    for ax, (cohort_name, title) in zip(axes, cohorts, strict=True):
        cohort = valid[valid["cohort"] == cohort_name]
        for stream, group in cohort.groupby("stream", sort=False):
            ax.plot(
                group["lag"],
                group["mean_balanced_accuracy"],
                marker="o",
                linewidth=2.0,
                color=STREAM_COLORS.get(stream),
                label=f"{stream} decoder",
            )
            ax.plot(
                group["lag"],
                group["mean_null_balanced_accuracy"],
                linestyle="--",
                linewidth=1.2,
                color=STREAM_COLORS.get(stream),
                alpha=0.6,
                label=f"{stream} shuffled null",
            )
        ax.axhline(0.5, color="#666666", linestyle=":", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Prediction lag, env steps")
        ax.set_ylim(0.45, 0.70)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Mean leave-one-trial-out balanced accuracy")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_signal_cycle(
    trials: list[dict[str, Any]],
    coord_by_trial: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    valid_trials = [trial for trial in trials if bool(trial["valid_binary"])]
    fig, axes = plt.subplots(len(valid_trials), 1, figsize=(6.6, max(2.0 * len(valid_trials), 4.0)), dpi=170)
    if len(valid_trials) == 1:
        axes = np.asarray([axes])
    for ax, trial in zip(axes, valid_trials, strict=True):
        coords = coord_by_trial[str(trial["trial_id"])]
        keep = np.linspace(0, len(coords) - 1, min(len(coords), 1600), dtype=np.int64)
        ax.plot(coords[keep, 0], coords[keep, 1], color="#2864ad", linewidth=0.8, alpha=0.8)
        event_idx = int(trial["event_local_index"])
        ax.scatter(coords[0, 0], coords[0, 1], s=14, color="#222222", marker="o")
        ax.scatter(coords[event_idx, 0], coords[event_idx, 1], s=42, color="#d95f02", marker="X")
        ax.set_ylabel(str(trial["trial_id"]), fontsize=7)
        ax.grid(True, alpha=0.2)
    axes[0].set_title("Signal PCA cycle by trial")
    axes[-1].set_xlabel("Signal PC1")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_meander_state_space(
    trials: list[dict[str, Any]],
    output_path: Path,
    *,
    random_seed: int,
) -> None:
    valid_trials = [trial for trial in trials if bool(trial["valid_binary"])]
    fig, axes = plt.subplots(
        len(valid_trials),
        2,
        figsize=(11.0, max(2.35 * len(valid_trials), 5.0)),
        dpi=170,
        squeeze=False,
    )
    rng = np.random.default_rng(_stable_seed(random_seed, "meander-plot"))
    for column, stream in enumerate(("signal", "body")):
        features = np.concatenate([trial[stream] for trial in valid_trials], axis=0)
        coords, model = _fit_pca(features, n_components=2)
        offset = 0
        for row, trial in enumerate(valid_trials):
            ax = axes[row, column]
            count = int(trial[stream].shape[0])
            trial_coords = coords[offset : offset + count]
            offset += count
            labels = np.asarray(
                [
                    _state_class(int(committed), int(meander))
                    for committed, meander in zip(trial["labels"], trial["meander_labels"], strict=True)
                ],
                dtype=object,
            )
            max_points = min(count, 1200)
            keep = rng.choice(count, size=max_points, replace=False) if count > max_points else np.arange(count)
            for state_class in ("uncommitted", "meander_toward", "meander_away", "committed_toward", "committed_away"):
                mask = labels[keep] == state_class
                if not np.any(mask):
                    continue
                alpha = 0.12 if state_class == "uncommitted" else 0.64
                size = 3 if state_class == "uncommitted" else 7
                ax.scatter(
                    trial_coords[keep[mask], 0],
                    trial_coords[keep[mask], 1],
                    s=size,
                    alpha=alpha,
                    color=STATE_COLORS[state_class],
                    label=state_class.replace("_", " "),
                    linewidths=0,
                )
            if row == 0:
                ax.set_title(f"{stream} state space")
            if column == 0:
                ax.set_ylabel(str(trial["trial_id"]), fontsize=7)
            if row == len(valid_trials) - 1:
                ax.set_xlabel(f"PC1 ({model['explained_variance_ratio'][0] * 100:.1f}%)")
            ax.grid(True, alpha=0.2)
    axes[0, 1].legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_leading_edge(
    timeseries: pd.DataFrame,
    output_path: Path,
) -> None:
    if timeseries.empty:
        return
    fig, ax1 = plt.subplots(figsize=(8.4, 4.8), dpi=180)
    signal = timeseries[timeseries["stream"] == "signal"].groupby("relative_step", as_index=False).agg(z_rms=("z_rms", "mean"))
    body = timeseries[timeseries["stream"] == "body"].groupby("relative_step", as_index=False).agg(z_rms=("z_rms", "mean"))
    clearance = timeseries[timeseries["stream"] == "body"].groupby("relative_step", as_index=False).agg(cliff_clearance=("cliff_clearance", "mean"))
    ax1.plot(body["relative_step"], body["z_rms"], color=STREAM_COLORS["body"], linewidth=2.0, label="body shift")
    ax1.plot(signal["relative_step"], signal["z_rms"], color=STREAM_COLORS["signal"], linewidth=1.7, label="signal shift")
    ax1.set_ylabel("RMS z-distance from pre-event baseline")
    ax1.set_xlabel("Env steps relative to true cliff reach")
    ax1.axvline(0, color="#333333", linestyle=":", linewidth=1.0)
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(clearance["relative_step"], clearance["cliff_clearance"], color="#555555", linestyle="--", linewidth=1.4, label="cliff clearance")
    ax2.axhline(0.0, color="#555555", linestyle=":", linewidth=1.0)
    ax2.set_ylabel("Mean cliff clearance")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def analyze_dynamical_regime(
    trace_root: Path,
    output_dir: Path,
    *,
    future_window: int,
    min_displacement: float,
    persistence_steps: int,
    persistence_fraction: float,
    meander_window: int,
    meander_min_displacement: float,
    projection_dim: int,
    pre_window: int,
    pre_guard: int,
    post_window: int,
    lags: list[int],
    ridge_alpha: float,
    permutations: int,
    null_block_size: int,
    null_event_count: int,
    random_seed: int,
) -> dict[str, Any]:
    trace_root = trace_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trials, feature_manifest = _load_trials(
        trace_root=trace_root,
        projection_dim=int(projection_dim),
        random_seed=int(random_seed),
        future_window=int(future_window),
        min_displacement=float(min_displacement),
        persistence_steps=int(persistence_steps),
        persistence_fraction=float(persistence_fraction),
        meander_window=int(meander_window),
        meander_min_displacement=float(meander_min_displacement),
    )
    valid_trials = [trial for trial in trials if bool(trial["valid_binary"])]
    true_cliff_trials = [trial for trial in valid_trials if bool(trial["true_cliff_reach"])]

    label_rows = _write_meander_rows(output_dir / "meander_commitment_labels.csv", trials)
    cycle_rows, signal_coords, signal_pca = _signal_cycle_metrics(trials)
    _write_csv(output_dir / "signal_cycle_metrics.csv", cycle_rows)

    timeseries_rows, event_summary_rows, persistence_rows, null_rows = _event_perturbation(
        valid_trials,
        pre_window=int(pre_window),
        pre_guard=int(pre_guard),
        post_window=int(post_window),
        null_count=int(null_event_count),
        random_seed=int(random_seed),
    )
    null_summary_rows = _null_control_summary(event_summary_rows, null_rows)
    _write_csv(output_dir / "event_locked_perturbation_by_trial.csv", timeseries_rows)
    _write_csv(output_dir / "body_regime_shift_summary.csv", event_summary_rows)
    _write_csv(output_dir / "persistence_summary.csv", persistence_rows)
    _write_csv(output_dir / "null_control_summary.csv", null_summary_rows)

    future_rows: list[dict[str, Any]] = []
    future_rows.extend(
        _evaluate_future_commitment(
            valid_trials,
            cohort_name="all_valid_aligned_window",
            lags=[int(value) for value in lags],
            rel_start=0,
            rel_stop=512,
            ridge_alpha=float(ridge_alpha),
            random_seed=int(random_seed),
            permutations=int(permutations),
            null_block_size=int(null_block_size),
        )
    )
    future_rows.extend(
        _evaluate_future_commitment(
            true_cliff_trials,
            cohort_name="true_cliff_reach_event_window",
            lags=[int(value) for value in lags],
            rel_start=0,
            rel_stop=512,
            ridge_alpha=float(ridge_alpha),
            random_seed=int(random_seed),
            permutations=int(permutations),
            null_block_size=int(null_block_size),
        )
    )
    future_summary_rows = _aggregate_future(future_rows)
    _write_csv(output_dir / "future_commitment_prediction.csv", future_rows)
    _write_csv(output_dir / "future_commitment_prediction_summary.csv", future_summary_rows)

    timeseries_df = pd.DataFrame(timeseries_rows)
    null_summary_df = pd.DataFrame(null_summary_rows)
    persistence_df = pd.DataFrame(persistence_rows)
    future_summary_df = pd.DataFrame(future_summary_rows)
    if not timeseries_df.empty:
        _plot_event_shift(timeseries_df, null_summary_df, output_dir / "event_locked_body_vs_signal_shift.png")
        _plot_leading_edge(timeseries_df, output_dir / "leading_edge_threshold_response.png")
    if not persistence_df.empty:
        _plot_persistence(persistence_df, output_dir / "regime_shift_persistence.png")
    if not future_summary_df.empty:
        _plot_future_prediction(future_summary_df, output_dir / "future_commitment_prediction_by_lag.png")
    _plot_signal_cycle(trials, signal_coords, output_dir / "signal_cycle_phase.png")
    _plot_meander_state_space(trials, output_dir / "meander_vs_commitment_state_space.png", random_seed=int(random_seed))

    summary = {
        "schema_version": 1,
        "kind": "dynamical_regime_deformation_off_analysis",
        "trace_root": str(trace_root),
        "output_dir": str(output_dir),
        "included_valid_binary_trials": [trial["trial_id"] for trial in valid_trials],
        "true_cliff_reach_trials": [trial["trial_id"] for trial in true_cliff_trials],
        "excluded_single_class_trial": EXCLUDED_SINGLE_CLASS_TRIAL,
        "label_policy": {
            "committed_future_window": int(future_window),
            "committed_min_displacement": float(min_displacement),
            "committed_persistence_steps": int(persistence_steps),
            "committed_persistence_fraction": float(persistence_fraction),
            "meander_window": int(meander_window),
            "meander_min_displacement": float(meander_min_displacement),
        },
        "event_policy": {
            "pre_window": int(pre_window),
            "pre_guard": int(pre_guard),
            "post_window": int(post_window),
            "windows": {key: list(value) for key, value in WINDOWS.items()},
        },
        "feature_manifest": feature_manifest,
        "signal_pca": signal_pca,
        "trial_label_counts": [
            {
                "trial_id": trial["trial_id"],
                "valid_binary": bool(trial["valid_binary"]),
                "event_type": trial["event_type"],
                "event_local_index": int(trial["event_local_index"]),
                "committed_forward_count": int(trial["committed_forward_count"]),
                "committed_backward_count": int(trial["committed_backward_count"]),
                "meander_forward_count": int(trial["meander_forward_count"]),
                "meander_backward_count": int(trial["meander_backward_count"]),
            }
            for trial in trials
        ],
        "event_regime_shift": null_summary_rows,
        "future_commitment_prediction": future_summary_rows,
        "outputs": {
            "meander_commitment_labels": str(output_dir / "meander_commitment_labels.csv"),
            "signal_cycle_metrics": str(output_dir / "signal_cycle_metrics.csv"),
            "body_regime_shift_summary": str(output_dir / "body_regime_shift_summary.csv"),
            "event_locked_perturbation_by_trial": str(output_dir / "event_locked_perturbation_by_trial.csv"),
            "persistence_summary": str(output_dir / "persistence_summary.csv"),
            "future_commitment_prediction": str(output_dir / "future_commitment_prediction.csv"),
            "future_commitment_prediction_summary": str(output_dir / "future_commitment_prediction_summary.csv"),
            "null_control_summary": str(output_dir / "null_control_summary.csv"),
            "signal_cycle_phase_png": str(output_dir / "signal_cycle_phase.png"),
            "event_locked_body_vs_signal_shift_png": str(output_dir / "event_locked_body_vs_signal_shift.png"),
            "meander_vs_commitment_state_space_png": str(output_dir / "meander_vs_commitment_state_space.png"),
            "regime_shift_persistence_png": str(output_dir / "regime_shift_persistence.png"),
            "leading_edge_threshold_response_png": str(output_dir / "leading_edge_threshold_response.png"),
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
        description="Analyze signal cycles and body regime shifts in deformation-off traces."
    )
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--future-window", type=int, default=32)
    parser.add_argument("--min-displacement", type=float, default=0.025)
    parser.add_argument("--persistence-steps", type=int, default=12)
    parser.add_argument("--persistence-fraction", type=float, default=0.75)
    parser.add_argument("--meander-window", type=int, default=8)
    parser.add_argument("--meander-min-displacement", type=float, default=0.006)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--pre-window", type=int, default=256)
    parser.add_argument("--pre-guard", type=int, default=32)
    parser.add_argument("--post-window", type=int, default=512)
    parser.add_argument("--lags", type=_parse_lags, default=_parse_lags("0,8,16,32,64,128"))
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--permutations", type=int, default=25)
    parser.add_argument("--null-block-size", type=int, default=32)
    parser.add_argument("--null-event-count", type=int, default=25)
    parser.add_argument("--random-seed", type=int, default=137)
    args = parser.parse_args()

    summary = analyze_dynamical_regime(
        args.trace_root,
        args.output_dir,
        future_window=int(args.future_window),
        min_displacement=float(args.min_displacement),
        persistence_steps=int(args.persistence_steps),
        persistence_fraction=float(args.persistence_fraction),
        meander_window=int(args.meander_window),
        meander_min_displacement=float(args.meander_min_displacement),
        projection_dim=int(args.projection_dim),
        pre_window=int(args.pre_window),
        pre_guard=int(args.pre_guard),
        post_window=int(args.post_window),
        lags=list(args.lags),
        ridge_alpha=float(args.ridge_alpha),
        permutations=int(args.permutations),
        null_block_size=int(args.null_block_size),
        null_event_count=int(args.null_event_count),
        random_seed=int(args.random_seed),
    )
    print(json.dumps(summary["outputs"], indent=2))


if __name__ == "__main__":
    main()
