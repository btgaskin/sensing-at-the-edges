#!/usr/bin/env python
"""Batch descriptive comparison of two sensing_edges PCA bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import subspace_angles

from sensing_edges.pca_analogue import (
    GRID_HIDDEN_START,
    GRID_HIDDEN_STOP,
    PHASE_UNIQUE_MICRO_INDEX,
    TRANSITION_COLORS,
    TRANSITION_LABELS,
    compute_pca_basis,
    derive_trial_cliff_edge_labels,
    derive_trial_phase_labels,
    derive_trial_stage_labels,
    derive_trial_transition_labels,
    flatten_grid_hidden_features,
    project_features,
)


def _read_json(path: str | Path) -> dict[str, Any]:
    with open(Path(path)) as handle:
        return json.load(handle)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as handle:
        json.dump(payload, handle, indent=2)


def _orient_nca_to_cliff_right(nca_states: np.ndarray, *, goal_direction: int) -> np.ndarray:
    oriented = np.asarray(nca_states, dtype=np.float32)
    if int(goal_direction) > 0:
        oriented = np.flip(oriented, axis=2).copy()
    return oriented


def _load_bundle(bundle_dir: str | Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, np.ndarray]:
    bundle_path = Path(bundle_dir)
    cohort_summary = _read_json(bundle_path / "cohort_summary.json")
    trial_df = pd.read_csv(bundle_path / "trial_manifest.csv")
    label_df = pd.read_csv(bundle_path / "trace_labels.csv")
    nca_states = np.asarray(np.load(bundle_path / "nca_states.npz")["states"], dtype=np.float32)
    if len(label_df) != int(nca_states.shape[0]):
        raise ValueError(f"Sample mismatch in {bundle_path}: {len(label_df)} labels vs {nca_states.shape[0]} states")
    return cohort_summary, trial_df, label_df, nca_states


def _choose_reference_trial(cohort_summary: dict[str, Any], trial_df: pd.DataFrame) -> tuple[str, str]:
    selected_ids = [str(item) for item in cohort_summary.get("selected_trial_ids", [])]
    if selected_ids:
        return selected_ids[0], "selected_primary"

    selected_side = str(cohort_summary.get("selected_side", "right"))
    success = trial_df.loc[(trial_df["goal_touched"] == True) & (trial_df["fell_flag"] == False)].copy()  # noqa: E712
    success_side = success.loc[success["side"] == selected_side].copy()
    if len(success_side) > 0:
        return str(success_side.iloc[0]["trial_id"]), "fallback_success_selected_side"
    if len(success) > 0:
        return str(success.iloc[0]["trial_id"]), "fallback_success_any_side"
    return str(trial_df.iloc[0]["trial_id"]), "fallback_first_trial"


def _label_single_trial(
    *,
    trial_df: pd.DataFrame,
    label_df: pd.DataFrame,
    trial_id: str,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    trial_lookup = trial_df.set_index("trial_id", drop=False)
    trial_row = trial_lookup.loc[trial_id]
    full_trial_df = label_df.loc[label_df["trial_id"] == trial_id].copy().reset_index(drop=True)
    phase_trial_df = full_trial_df.loc[
        full_trial_df["micro_index"].to_numpy(dtype=np.int32) == PHASE_UNIQUE_MICRO_INDEX
    ].copy().reset_index(drop=True)
    phase_trial_df = derive_trial_phase_labels(phase_trial_df, trial_row)
    phase_trial_df = derive_trial_stage_labels(phase_trial_df, trial_row)
    phase_trial_df = derive_trial_cliff_edge_labels(phase_trial_df, trial_row)
    phase_trial_df = derive_trial_transition_labels(phase_trial_df)

    broadcast_cols = [
        "trial_id",
        "env_step",
        "phase_index",
        "goalward_x",
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
    full_trial_df = full_trial_df.merge(
        phase_trial_df[broadcast_cols],
        on=["trial_id", "env_step", "phase_index"],
        how="left",
        suffixes=("", "_phase"),
    )
    return trial_row, phase_trial_df, full_trial_df


def _window_reduce(values: np.ndarray, *, window: int) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    if data.ndim != 1:
        raise ValueError("Expected 1D series for window reduction")
    if len(data) <= 0:
        return np.zeros((0,), dtype=np.float32)
    n_windows = max(1, len(data) // max(window, 1))
    trimmed = data[: n_windows * window]
    if len(trimmed) <= 0:
        return np.asarray([float(np.mean(data))], dtype=np.float32)
    return trimmed.reshape(n_windows, window).mean(axis=1).astype(np.float32)


def _features_to_grid_hidden(features: np.ndarray) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    expected_dim = 2 * 7 * (GRID_HIDDEN_STOP - GRID_HIDDEN_START)
    if arr.ndim != 2 or arr.shape[1] != expected_dim:
        raise ValueError(f"Expected flattened grid-hidden features with shape (n, {expected_dim}), got {arr.shape}")
    return arr.reshape((-1, 2, 7, GRID_HIDDEN_STOP - GRID_HIDDEN_START))


def _welch_summary(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    if len(x) < 2 or len(y) < 2:
        return {"n_x": int(len(x)), "n_y": int(len(y)), "t_stat": float("nan"), "p_value": float("nan")}
    test = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
    return {"n_x": int(len(x)), "n_y": int(len(y)), "t_stat": float(test.statistic), "p_value": float(test.pvalue)}


def _mannwhitney_summary(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    if len(x) <= 0 or len(y) <= 0:
        return {"n_x": int(len(x)), "n_y": int(len(y)), "u_stat": float("nan"), "p_value": float("nan")}
    test = stats.mannwhitneyu(x, y, alternative="two-sided")
    return {"n_x": int(len(x)), "n_y": int(len(y)), "u_stat": float(test.statistic), "p_value": float(test.pvalue)}


def _ks_summary(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    if len(x) <= 0 or len(y) <= 0:
        return {"n_x": int(len(x)), "n_y": int(len(y)), "ks_stat": float("nan"), "p_value": float("nan")}
    test = stats.ks_2samp(x, y)
    return {"n_x": int(len(x)), "n_y": int(len(y)), "ks_stat": float(test.statistic), "p_value": float(test.pvalue)}


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pooled = np.sqrt(((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1)) / max(len(x) + len(y) - 2, 1))
    if pooled <= 0.0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / pooled)


def _resample_rows(rows: np.ndarray, *, target_len: int) -> np.ndarray:
    arr = np.asarray(rows, dtype=np.float32)
    if arr.shape[0] == target_len:
        return arr
    if arr.shape[0] == 1:
        return np.repeat(arr, target_len, axis=0)
    source_x = np.linspace(0.0, 1.0, arr.shape[0], dtype=np.float64)
    target_x = np.linspace(0.0, 1.0, target_len, dtype=np.float64)
    out = np.empty((target_len, arr.shape[1]), dtype=np.float32)
    for col in range(arr.shape[1]):
        out[:, col] = np.interp(target_x, source_x, arr[:, col]).astype(np.float32)
    return out


def _trajectory_distance(a: np.ndarray, b: np.ndarray, *, target_len: int = 400) -> dict[str, float]:
    aa = _resample_rows(a, target_len=target_len)
    bb = _resample_rows(b, target_len=target_len)
    dist = np.linalg.norm(aa - bb, axis=1)
    return {
        "target_len": int(target_len),
        "mean_distance": float(np.mean(dist)),
        "median_distance": float(np.median(dist)),
        "max_distance": float(np.max(dist)),
        "end_distance": float(dist[-1]),
    }


def _hidden_channel_means(oriented_nca: np.ndarray) -> np.ndarray:
    if oriented_nca.ndim == 2:
        grid_hidden = _features_to_grid_hidden(oriented_nca)
    else:
        grid_hidden = np.asarray(oriented_nca[..., GRID_HIDDEN_START:GRID_HIDDEN_STOP], dtype=np.float32)
    return grid_hidden.mean(axis=(1, 2)).astype(np.float32)


def _left_right_asymmetry(oriented_nca: np.ndarray) -> np.ndarray:
    if oriented_nca.ndim == 2:
        grid_hidden = _features_to_grid_hidden(oriented_nca)
    else:
        grid_hidden = np.asarray(oriented_nca[..., GRID_HIDDEN_START:GRID_HIDDEN_STOP], dtype=np.float32)
    left = grid_hidden[:, :, :3, :]
    right = np.flip(grid_hidden[:, :, 4:, :], axis=2)
    return np.mean(np.abs(left - right), axis=(1, 2, 3)).astype(np.float32)


def _derivative_norm(features: np.ndarray) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    if arr.shape[0] < 2:
        return np.zeros((0,), dtype=np.float32)
    return np.linalg.norm(np.diff(arr, axis=0), axis=1).astype(np.float32)


def _trial_metrics(
    *,
    cohort_name: str,
    trial_row: pd.Series,
    phase_trial_df: pd.DataFrame,
    full_trial_df: pd.DataFrame,
    phase_projection: np.ndarray,
    oriented_phase_states: np.ndarray,
    oriented_phase_features: np.ndarray,
) -> dict[str, Any]:
    transitions = full_trial_df["transition_label"].to_numpy(dtype=object)
    cliff_clearance = full_trial_df["cliff_clearance_x"].to_numpy(dtype=np.float32)
    state_norm = np.linalg.norm(oriented_phase_features, axis=1)
    deriv_norm = _derivative_norm(oriented_phase_features)
    phase_steps = phase_trial_df["phase_tick"].to_numpy(dtype=np.int32)
    stage_counts = {
        label: int(np.sum(transitions == label))
        for label in TRANSITION_LABELS
    }
    arc = float(np.sum(np.linalg.norm(np.diff(phase_projection, axis=0), axis=1))) if len(phase_projection) > 1 else 0.0
    displacement = float(np.linalg.norm(phase_projection[-1] - phase_projection[0])) if len(phase_projection) > 1 else 0.0

    def _first_idx(label: str) -> int | None:
        hits = np.flatnonzero(transitions == label)
        return int(hits[0]) if hits.size > 0 else None

    return {
        "cohort_name": cohort_name,
        "trial_id": str(trial_row["trial_id"]),
        "side": str(trial_row["side"]),
        "goal_direction": int(trial_row["goal_direction"]),
        "goal_touched": bool(trial_row["goal_touched"]),
        "fell_flag": bool(trial_row["fell_flag"]),
        "termination_reason": str(trial_row["termination_reason"]),
        "first_peak_direction": str(trial_row["first_peak_direction"]),
        "reached_cliff_frontier": bool(trial_row["reached_cliff_frontier"]),
        "reversed_pre_ledge": bool(trial_row["reversed_pre_ledge"]),
        "steps_alive": int(trial_row["steps_alive"]),
        "controller_step_count": int(trial_row["controller_step_count"]),
        "phase_unique_sample_count": int(len(phase_trial_df)),
        "transition_counts": stage_counts,
        "first_proximity_in_idx": _first_idx("proximity_in"),
        "first_probe_idx": _first_idx("probe"),
        "first_proximity_out_idx": _first_idx("proximity_out"),
        "first_return_idx": _first_idx("return"),
        "peak_overreach_cells": float(max(0.0, -float(np.min(cliff_clearance)) / 0.1)),
        "min_cliff_clearance": float(np.min(cliff_clearance)),
        "max_outward_progress": float(np.max(full_trial_df["cliff_outward_progress_x"].to_numpy(dtype=np.float32))),
        "pca_arc_length": arc,
        "pca_displacement": displacement,
        "pca_straightness": float(displacement / arc) if arc > 0.0 else float("nan"),
        "state_norm_mean": float(np.mean(state_norm)),
        "state_norm_std": float(np.std(state_norm)),
        "state_derivative_mean": float(np.mean(deriv_norm)) if len(deriv_norm) > 0 else 0.0,
        "state_derivative_std": float(np.std(deriv_norm)) if len(deriv_norm) > 0 else 0.0,
        "left_right_asymmetry_mean": float(np.mean(_left_right_asymmetry(oriented_phase_states))),
    }


def _plot_timeline_2d(path: Path, projection: np.ndarray, title: str) -> None:
    cmap = LinearSegmentedColormap.from_list("timeline", ["#1d3557", "#6d597a", "#e56b6f", "#f4a261", "#2a9d8f"])
    xy = np.asarray(projection[:, :2], dtype=np.float32)
    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    time = np.linspace(0.0, 1.0, len(xy), dtype=np.float32)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot(xy[:, 0], xy[:, 1], color="#d9dde3", linewidth=1.0, alpha=0.45, zorder=1)
    line = LineCollection(segments, cmap=cmap, linewidths=2.2, alpha=0.98, zorder=2)
    line.set_array(time[:-1])
    ax.add_collection(line)
    ax.scatter(xy[0, 0], xy[0, 1], s=42, color="#111111", marker="o", zorder=3)
    ax.scatter(xy[-1, 0], xy[-1, 1], s=48, color="#111111", marker="s", zorder=3)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.14, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    colorbar = fig.colorbar(line, ax=ax, shrink=0.82, pad=0.02)
    colorbar.set_label("normalized time")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_pair_overlay_2d(path: Path, projection_a: np.ndarray, projection_b: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot(projection_a[:, 0], projection_a[:, 1], color="#33658a", linewidth=2.0, alpha=0.92, label="trial_a")
    ax.plot(projection_b[:, 0], projection_b[:, 1], color="#bc4b51", linewidth=2.0, alpha=0.92, label="trial_b")
    ax.scatter(projection_a[0, 0], projection_a[0, 1], s=36, color="#33658a", marker="o")
    ax.scatter(projection_a[-1, 0], projection_a[-1, 1], s=40, color="#33658a", marker="s")
    ax.scatter(projection_b[0, 0], projection_b[0, 1], s=36, color="#bc4b51", marker="o")
    ax.scatter(projection_b[-1, 0], projection_b[-1, 1], s=40, color="#bc4b51", marker="s")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.14, linewidth=0.6)
    ax.legend(frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _resampled_corr(x: np.ndarray, y: np.ndarray, *, target_len: int = 400) -> float:
    xx = _resample_rows(np.asarray(x, dtype=np.float32).reshape(-1, 1), target_len=target_len)[:, 0]
    yy = _resample_rows(np.asarray(y, dtype=np.float32).reshape(-1, 1), target_len=target_len)[:, 0]
    if np.std(xx) <= 0.0 or np.std(yy) <= 0.0:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def _segment_metrics(
    full_trial_df: pd.DataFrame,
    oriented_full_features: np.ndarray,
    full_projection: np.ndarray,
) -> dict[str, dict[str, Any]]:
    transitions = full_trial_df["transition_label"].to_numpy(dtype=object)
    state_norm = np.linalg.norm(oriented_full_features, axis=1)
    deriv_norm = np.concatenate(
        [np.asarray([0.0], dtype=np.float32), _derivative_norm(oriented_full_features)],
        axis=0,
    )
    asym = _left_right_asymmetry(oriented_full_features)
    channel_means = _hidden_channel_means(oriented_full_features)

    out: dict[str, dict[str, Any]] = {}
    for label in TRANSITION_LABELS:
        mask = transitions == label
        if not np.any(mask):
            out[label] = {"count": 0}
            continue
        out[label] = {
            "count": int(np.sum(mask)),
            "state_norm_mean": float(np.mean(state_norm[mask])),
            "state_derivative_mean": float(np.mean(deriv_norm[mask])),
            "asymmetry_mean": float(np.mean(asym[mask])),
            "pc_centroid": [float(v) for v in np.mean(full_projection[mask, :3], axis=0)],
            "channel_means": [float(v) for v in np.mean(channel_means[mask], axis=0)],
        }
    return out


def _event_window_metrics(
    full_trial_df: pd.DataFrame,
    oriented_full_features: np.ndarray,
    full_projection: np.ndarray,
    *,
    window_before: int = 400,
    window_after: int = 400,
) -> dict[str, dict[str, Any]]:
    transitions = full_trial_df["transition_label"].to_numpy(dtype=object)
    channel_means = _hidden_channel_means(oriented_full_features)
    state_norm = np.linalg.norm(oriented_full_features, axis=1)
    asym = _left_right_asymmetry(oriented_full_features)
    out: dict[str, dict[str, Any]] = {}
    for event_label in ("proximity_in", "probe", "proximity_out", "return"):
        hits = np.flatnonzero(transitions == event_label)
        if hits.size <= 0:
            out[event_label] = {"present": False}
            continue
        idx = int(hits[0])
        lo = max(0, idx - window_before)
        hi = min(len(transitions), idx + window_after + 1)
        out[event_label] = {
            "present": True,
            "index": idx,
            "window_start": lo,
            "window_end": hi,
            "pc_centroid": [float(v) for v in np.mean(full_projection[lo:hi, :3], axis=0)],
            "state_norm_mean": float(np.mean(state_norm[lo:hi])),
            "asymmetry_mean": float(np.mean(asym[lo:hi])),
            "channel_means": [float(v) for v in np.mean(channel_means[lo:hi], axis=0)],
        }
    return out


def _subspace_summary(a_features: np.ndarray, b_features: np.ndarray, *, n_components: int = 3) -> dict[str, Any]:
    a_pca = compute_pca_basis(a_features, n_components=n_components)
    b_pca = compute_pca_basis(b_features, n_components=n_components)
    angles = np.degrees(subspace_angles(a_pca["components"].T, b_pca["components"].T))
    return {
        "n_components": int(n_components),
        "principal_angles_deg": [float(v) for v in angles],
        "a_explained_variance_ratio": [float(v) for v in a_pca["explained_variance_ratio"]],
        "b_explained_variance_ratio": [float(v) for v in b_pca["explained_variance_ratio"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch compare two sensing_edges PCA bundles.")
    parser.add_argument("--intact-bundle-dir", type=str, required=True)
    parser.add_argument("--comparison-bundle-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    bundle_specs = [
        ("intact_native_core", Path(args.intact_bundle_dir)),
        ("deformation_off_native_core", Path(args.comparison_bundle_dir)),
    ]

    trial_records: list[dict[str, Any]] = []
    shared_phase_features: list[np.ndarray] = []
    trial_data: dict[str, dict[str, Any]] = {}

    for cohort_name, bundle_dir in bundle_specs:
        cohort_summary, trial_df, label_df, nca_states = _load_bundle(bundle_dir)
        chosen_trial_id, selection_reason = _choose_reference_trial(cohort_summary, trial_df)

        for _, trial_row in trial_df.iterrows():
            trial_id = str(trial_row["trial_id"])
            labeled_trial_row, phase_trial_df, full_trial_df = _label_single_trial(
                trial_df=trial_df,
                label_df=label_df,
                trial_id=trial_id,
            )
            full_mask = label_df["trial_id"].astype(str).to_numpy() == trial_id
            full_states = nca_states[full_mask]
            oriented_full_states = _orient_nca_to_cliff_right(full_states, goal_direction=int(trial_row["goal_direction"]))
            oriented_full_features = flatten_grid_hidden_features(oriented_full_states)
            oriented_phase_states = oriented_full_states[
                full_trial_df["micro_index"].to_numpy(dtype=np.int32) == PHASE_UNIQUE_MICRO_INDEX
            ]
            oriented_phase_features = oriented_full_features[
                full_trial_df["micro_index"].to_numpy(dtype=np.int32) == PHASE_UNIQUE_MICRO_INDEX
            ]
            shared_phase_features.append(oriented_phase_features)
            trial_data[f"{cohort_name}:{trial_id}"] = {
                "cohort_name": cohort_name,
                "bundle_dir": str(bundle_dir),
                "selection_reason": selection_reason if trial_id == chosen_trial_id else "",
                "trial_row": labeled_trial_row,
                "phase_trial_df": phase_trial_df,
                "full_trial_df": full_trial_df,
                "oriented_phase_states": oriented_phase_states,
                "oriented_phase_features": oriented_phase_features,
                "oriented_full_states": oriented_full_states,
                "oriented_full_features": oriented_full_features,
            }

    shared_pca = compute_pca_basis(np.concatenate(shared_phase_features, axis=0), n_components=5)

    for key, item in trial_data.items():
        del key
        item["phase_projection"] = project_features(
            item["oriented_phase_features"],
            shared_pca["mean"],
            shared_pca["components"],
        )
        item["full_projection"] = project_features(
            item["oriented_full_features"],
            shared_pca["mean"],
            shared_pca["components"],
        )
        metrics = _trial_metrics(
            cohort_name=item["cohort_name"],
            trial_row=item["trial_row"],
            phase_trial_df=item["phase_trial_df"],
            full_trial_df=item["full_trial_df"],
            phase_projection=item["phase_projection"],
            oriented_phase_states=item["oriented_phase_states"],
            oriented_phase_features=item["oriented_phase_features"],
        )
        metrics["bundle_dir"] = item["bundle_dir"]
        metrics["selection_reason"] = item["selection_reason"]
        metrics["segment_metrics"] = _segment_metrics(
            item["full_trial_df"],
            item["oriented_full_features"],
            item["full_projection"],
        )
        metrics["event_window_metrics"] = _event_window_metrics(
            item["full_trial_df"],
            item["oriented_full_features"],
            item["full_projection"],
        )
        trial_records.append(metrics)

        _plot_timeline_2d(
            figures_dir / f"{item['cohort_name']}_{str(item['trial_row']['trial_id'])}_pc1_pc2_time.png",
            item["full_projection"],
            f"{item['cohort_name']}: {str(item['trial_row']['trial_id'])}",
        )

    cliff_pair = (
        trial_data["intact_native_core:pair_4_2_left_000"],
        trial_data["deformation_off_native_core:pair_4_2_right_000"],
    )
    goal_pair = (
        trial_data["intact_native_core:pair_4_2_right_000"],
        trial_data["deformation_off_native_core:pair_4_2_left_000"],
    )

    def _pair_summary(name: str, left_item: dict[str, Any], right_item: dict[str, Any]) -> dict[str, Any]:
        left_channels = _hidden_channel_means(left_item["oriented_phase_features"])
        right_channels = _hidden_channel_means(right_item["oriented_phase_features"])
        window = 50
        channel_tests: dict[str, Any] = {}
        for channel_idx in range(left_channels.shape[1]):
            x = _window_reduce(left_channels[:, channel_idx], window=window)
            y = _window_reduce(right_channels[:, channel_idx], window=window)
            channel_tests[f"grid_hidden_{channel_idx}"] = {
                "welch": _welch_summary(x, y),
                "mannwhitney": _mannwhitney_summary(x, y),
                "ks": _ks_summary(x, y),
                "cohens_d": _cohens_d(x, y),
                "mean_x": float(np.mean(x)),
                "mean_y": float(np.mean(y)),
                "delta_mean": float(np.mean(x) - np.mean(y)),
            }

        left_norm = np.linalg.norm(left_item["oriented_phase_features"], axis=1)
        right_norm = np.linalg.norm(right_item["oriented_phase_features"], axis=1)
        left_deriv = _derivative_norm(left_item["oriented_phase_features"])
        right_deriv = _derivative_norm(right_item["oriented_phase_features"])
        left_asym = _left_right_asymmetry(left_item["oriented_phase_features"])
        right_asym = _left_right_asymmetry(right_item["oriented_phase_features"])

        segment_deltas: dict[str, Any] = {}
        left_segment_metrics = _segment_metrics(
            left_item["full_trial_df"],
            left_item["oriented_full_features"],
            left_item["full_projection"],
        )
        right_segment_metrics = _segment_metrics(
            right_item["full_trial_df"],
            right_item["oriented_full_features"],
            right_item["full_projection"],
        )
        for label in TRANSITION_LABELS:
            left_seg = left_segment_metrics.get(label, {"count": 0})
            right_seg = right_segment_metrics.get(label, {"count": 0})
            segment_deltas[label] = {
                "count_a": int(left_seg.get("count", 0)),
                "count_b": int(right_seg.get("count", 0)),
                "delta_state_norm_mean": float(left_seg.get("state_norm_mean", np.nan) - right_seg.get("state_norm_mean", np.nan)),
                "delta_state_derivative_mean": float(
                    left_seg.get("state_derivative_mean", np.nan) - right_seg.get("state_derivative_mean", np.nan)
                ),
                "delta_asymmetry_mean": float(left_seg.get("asymmetry_mean", np.nan) - right_seg.get("asymmetry_mean", np.nan)),
                "delta_pc_centroid": [
                    float(a - b)
                    for a, b in zip(left_seg.get("pc_centroid", [np.nan, np.nan, np.nan]), right_seg.get("pc_centroid", [np.nan, np.nan, np.nan]))
                ],
                "delta_channel_means": [
                    float(a - b)
                    for a, b in zip(left_seg.get("channel_means", [np.nan] * (GRID_HIDDEN_STOP - GRID_HIDDEN_START)),
                                    right_seg.get("channel_means", [np.nan] * (GRID_HIDDEN_STOP - GRID_HIDDEN_START)))
                ],
            }

        event_windows = {
            "trial_a": _event_window_metrics(
                left_item["full_trial_df"],
                left_item["oriented_full_features"],
                left_item["full_projection"],
            ),
            "trial_b": _event_window_metrics(
                right_item["full_trial_df"],
                right_item["oriented_full_features"],
                right_item["full_projection"],
            ),
        }

        return {
            "pair_name": name,
            "trial_a": str(left_item["trial_row"]["trial_id"]),
            "trial_b": str(right_item["trial_row"]["trial_id"]),
            "trajectory_distance_pc": _trajectory_distance(left_item["phase_projection"][:, :3], right_item["phase_projection"][:, :3]),
            "trajectory_distance_raw": _trajectory_distance(left_item["oriented_phase_features"], right_item["oriented_phase_features"]),
            "subspace_similarity": _subspace_summary(left_item["oriented_phase_features"], right_item["oriented_phase_features"]),
            "phase_projection_corr": {
                f"pc{idx + 1}": _resampled_corr(left_item["phase_projection"][:, idx], right_item["phase_projection"][:, idx])
                for idx in range(min(3, left_item["phase_projection"].shape[1], right_item["phase_projection"].shape[1]))
            },
            "state_norm_window_tests": {
                "welch": _welch_summary(_window_reduce(left_norm, window=window), _window_reduce(right_norm, window=window)),
                "mannwhitney": _mannwhitney_summary(_window_reduce(left_norm, window=window), _window_reduce(right_norm, window=window)),
                "ks": _ks_summary(_window_reduce(left_norm, window=window), _window_reduce(right_norm, window=window)),
                "cohens_d": _cohens_d(_window_reduce(left_norm, window=window), _window_reduce(right_norm, window=window)),
            },
            "derivative_window_tests": {
                "welch": _welch_summary(_window_reduce(left_deriv, window=window), _window_reduce(right_deriv, window=window)),
                "mannwhitney": _mannwhitney_summary(_window_reduce(left_deriv, window=window), _window_reduce(right_deriv, window=window)),
                "ks": _ks_summary(_window_reduce(left_deriv, window=window), _window_reduce(right_deriv, window=window)),
                "cohens_d": _cohens_d(_window_reduce(left_deriv, window=window), _window_reduce(right_deriv, window=window)),
            },
            "asymmetry_window_tests": {
                "welch": _welch_summary(_window_reduce(left_asym, window=window), _window_reduce(right_asym, window=window)),
                "mannwhitney": _mannwhitney_summary(_window_reduce(left_asym, window=window), _window_reduce(right_asym, window=window)),
                "ks": _ks_summary(_window_reduce(left_asym, window=window), _window_reduce(right_asym, window=window)),
                "cohens_d": _cohens_d(_window_reduce(left_asym, window=window), _window_reduce(right_asym, window=window)),
            },
            "channel_window_tests": channel_tests,
            "segment_deltas": segment_deltas,
            "event_windows": event_windows,
            "trial_a_transition_counts": {
                label: int(np.sum(left_item["full_trial_df"]["transition_label"].to_numpy(dtype=object) == label))
                for label in TRANSITION_LABELS
            },
            "trial_b_transition_counts": {
                label: int(np.sum(right_item["full_trial_df"]["transition_label"].to_numpy(dtype=object) == label))
                for label in TRANSITION_LABELS
            },
        }

    pairwise = {
        "cliff_first_motif": _pair_summary("cliff_first_motif", cliff_pair[0], cliff_pair[1]),
        "goal_first_motif": _pair_summary("goal_first_motif", goal_pair[0], goal_pair[1]),
    }
    _plot_pair_overlay_2d(
        figures_dir / "cliff_first_motif_overlay_pc1_pc2.png",
        cliff_pair[0]["full_projection"],
        cliff_pair[1]["full_projection"],
        "cliff_first motif overlay",
    )
    _plot_pair_overlay_2d(
        figures_dir / "goal_first_motif_overlay_pc1_pc2.png",
        goal_pair[0]["full_projection"],
        goal_pair[1]["full_projection"],
        "goal_first motif overlay",
    )

    trial_table = pd.json_normalize(trial_records, sep=".")
    trial_table.to_csv(output_dir / "trial_metrics.csv", index=False)

    summary = {
        "note": (
            "These are descriptive comparisons over deterministic single-trial traces. "
            "Windowed p-values treat coarse time windows as units and should not be read as population-level inference."
        ),
        "shared_pca_explained_variance_ratio": [float(v) for v in shared_pca["explained_variance_ratio"]],
        "trial_count": len(trial_records),
        "trial_metrics_path": str(output_dir / "trial_metrics.csv"),
        "figures_dir": str(figures_dir),
        "pairwise": pairwise,
    }
    _write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
