#!/usr/bin/env python
"""Pre-contact continue-vs-turn decoder for the expanded S4 goal-cliff traces.

The older S4 decoder asks whether state near the release event predicts future
movement direction. This analysis asks a sharper pre-contact question: before
formal loss of support, can body-internal state distinguish samples that are
near release ("about to turn") from samples that are still far from release
("continue")?
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
SENSING_EDGES_ROOT = REPO_ROOT / "sensing_edges"
for path in (SCRIPT_DIR, SENSING_EDGES_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from committed_direction_decoding import (  # noqa: E402
    _auc,
    _balanced_accuracy,
    _mcc,
    _ridge_scores,
    _stable_seed,
    _standardize_train_test,
)
from leading_edge_reversal_analysis import (  # noqa: E402
    _drive_features,
    _leading_edge_indices,
    _local_geometry_features,
    _spring_features,
    _trace_path_for_trial,
)
from s4_mechanism_packet import _band_cells, _signal_cell_features  # noqa: E402
from voxel_reversal_heatmap_analysis import _cell_spring_indices  # noqa: E402


DEFAULT_PACKAGE_ROOT = REPO_ROOT / "paper_data" / "preprint" / "s4_goal_cliff_propagation_20260614"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_ROOT / "precontact_proximity_decoder"

PRIMARY_CLEARANCE_MARGIN = 0.05
PRIMARY_NEAR_THRESHOLD = 128
DEFAULT_FAR_THRESHOLD = 512
DEFAULT_PHASE_BINS = 12
DEFAULT_RIDGE_ALPHA = 10.0
DEFAULT_COMMON_DIM = 32
DEFAULT_RANDOM_SEED = 137

STREAM_ORDER = [
    "mechanical_frontier",
    "mechanical_plus_drive",
    "drive_only",
    "signal_frontier",
    "phase_only",
    "position_leak",
]
STREAM_LABELS = {
    "mechanical_frontier": "Mechanical",
    "mechanical_plus_drive": "Mech.+drive",
    "drive_only": "Drive only",
    "signal_frontier": "Signal",
    "phase_only": "Phase only",
    "position_leak": "Position leak",
}
STREAM_COLORS = {
    "mechanical_frontier": "#8CB87C",
    "mechanical_plus_drive": "#6F9F63",
    "drive_only": "#B9A46A",
    "signal_frontier": "#7D8EC7",
    "phase_only": "#AAAAAA",
    "position_leak": "#D4736C",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


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


def _bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return bool(value)


def _trace_roots(package_root: Path) -> list[tuple[int, Path]]:
    inventory_path = package_root / "tables" / "trace_inventory.csv"
    inventory = pd.read_csv(inventory_path)
    roots: list[tuple[int, Path]] = []
    for _, row in inventory.sort_values("plateau_width", kind="stable").iterrows():
        root = Path(str(row["assay_root"]))
        if not root.exists():
            raise FileNotFoundError(root)
        roots.append((int(row["plateau_width"]), root))
    return roots


def _phase_from_trace(trace: dict[str, np.ndarray], *, phase_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a locomotor phase proxy from controller drives only.

    This intentionally excludes COM, terrain, cliff clearance, and global x. It
    is used as a matching/control variable rather than as evidence for sensing.
    """

    h = np.asarray(trace["phase_h_drive"], dtype=np.float32).reshape(trace["phase_h_drive"].shape[0], -1)
    v = np.asarray(trace["phase_v_drive"], dtype=np.float32).reshape(trace["phase_v_drive"].shape[0], -1)
    h_mean = h.mean(axis=1)
    v_mean = v.mean(axis=1)
    h_centered = h_mean - float(np.mean(h_mean))
    v_centered = v_mean - float(np.mean(v_mean))
    angle = np.arctan2(v_centered, h_centered).astype(np.float32)
    bins = np.floor(((angle + np.pi) / (2.0 * np.pi)) * int(phase_bins)).astype(np.int32)
    bins = np.clip(bins, 0, int(phase_bins) - 1)
    return angle, bins


def _mechanical_cell_features(
    *,
    trace: dict[str, np.ndarray],
    cell_index: int,
    inward_sign: float,
) -> np.ndarray:
    indices = _cell_spring_indices(trace, int(cell_index))
    attached = _spring_features(trace, indices["attached_springs"])
    geometry = _local_geometry_features(
        trace=trace,
        cells=np.asarray([int(cell_index)], dtype=np.int32),
        points=np.asarray(indices["points"], dtype=np.int32),
        inward_sign=float(inward_sign),
    )
    return np.concatenate([attached, geometry], axis=1).astype(np.float32)


def _band_features(
    *,
    trace: dict[str, np.ndarray],
    cells: list[int],
    inward_sign: float,
    hidden_start_channel: int,
    robot_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    mechanical_blocks = [
        _mechanical_cell_features(trace=trace, cell_index=int(cell), inward_sign=float(inward_sign))
        for cell in cells
    ]
    drive_blocks = [
        _drive_features(trace, np.asarray([int(cell)], dtype=np.int32))
        for cell in cells
    ]
    signal_blocks = [
        _signal_cell_features(
            trace,
            cell_index=int(cell),
            hidden_start_channel=int(hidden_start_channel),
            robot_shape=robot_shape,
        )
        for cell in cells
    ]
    mechanical = np.concatenate(mechanical_blocks, axis=1).astype(np.float32)
    drive = np.concatenate(drive_blocks, axis=1).astype(np.float32)
    signal = np.concatenate(signal_blocks, axis=1).astype(np.float32)
    return {
        "mechanical_frontier": mechanical,
        "mechanical_plus_drive": np.concatenate([mechanical, drive], axis=1).astype(np.float32),
        "drive_only": drive,
        "signal_frontier": signal,
    }


def _position_leak_features(rows: pd.DataFrame) -> np.ndarray:
    columns = [
        "com_x",
        "front_x",
        "back_x",
        "cliff_clearance",
        "com_x_to_cliff",
        "cliff_edge_body_x",
    ]
    available = [column for column in columns if column in rows.columns]
    return rows[available].to_numpy(dtype=np.float32)


def _load_events(package_root: Path, *, phase_bins: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events: list[dict[str, Any]] = []
    hidden_start = 6
    robot_shape: tuple[int, int] | None = None
    metadata: dict[str, Any] = {
        "trace_roots": [],
        "hidden_start_channel": hidden_start,
        "phase_bins": int(phase_bins),
    }

    for plateau_width, trace_root in _trace_roots(package_root):
        manifest = json.loads((trace_root / "manifest.json").read_text())
        metadata["trace_roots"].append(str(trace_root))
        hidden_start = int((manifest.get("nca_channel_schema") or {}).get("hidden_start_channel", hidden_start))
        metadata["hidden_start_channel"] = hidden_start
        trial_manifest = pd.read_csv(trace_root / "trial_manifest.csv")
        step_manifest = pd.read_csv(trace_root / "step_manifest.csv", low_memory=False)
        with np.load(trace_root / "static_arrays.npz") as static:
            robot_shape = tuple(int(value) for value in static["robot_morph"].shape)
        trials = trial_manifest[
            trial_manifest["first_cliff_reach_step"].notna()
            & (trial_manifest["goal_direction"] != 0)
        ].copy()
        trials = trials.sort_values(["goal_distance", "cliff_distance", "side"], kind="stable")
        for _, trial_row in trials.iterrows():
            trial_id = str(trial_row["trial_id"])
            rows = step_manifest[step_manifest["trial_id"] == trial_id].sort_values("step_index", kind="stable").reset_index(drop=True)
            if rows.empty:
                continue
            event_index = int(trial_row["first_cliff_reach_step"])
            trace_path = _trace_path_for_trial(trace_root, rows)
            trace = _load_npz(trace_path)
            step_count = min(int(len(rows)), int(trace["cell_corner_positions"].shape[0]))
            rows = rows.iloc[:step_count].reset_index(drop=True)
            trace = {
                key: value[:step_count] if getattr(value, "shape", ())[:1] == (trace["cell_corner_positions"].shape[0],) else value
                for key, value in trace.items()
            }
            edge = _leading_edge_indices(rows=rows, trace=trace, event_index=event_index)
            bands = _band_cells(trace=trace, edge=edge, event_index=event_index)
            frontier_cells = bands["frontier"]
            stream_features = _band_features(
                trace=trace,
                cells=frontier_cells,
                inward_sign=float(edge["inward_sign"]),
                hidden_start_channel=int(hidden_start),
                robot_shape=robot_shape,
            )
            phase_angle, phase_bin = _phase_from_trace(trace, phase_bins=int(phase_bins))
            stream_features["phase_only"] = np.column_stack([np.sin(phase_angle), np.cos(phase_angle)]).astype(np.float32)
            stream_features["position_leak"] = _position_leak_features(rows)

            time_to_release = np.arange(step_count, dtype=np.int32)
            time_to_release = int(event_index) - time_to_release
            event_id = f"pw{int(plateau_width)}_{trial_id}"
            events.append(
                {
                    "event_id": event_id,
                    "plateau_width": int(plateau_width),
                    "trial_id": trial_id,
                    "distance_pair": str(trial_row.get("distance_pair", "")),
                    "goal_distance": int(trial_row["goal_distance"]),
                    "cliff_distance": int(trial_row["cliff_distance"]),
                    "side": str(trial_row.get("side", "")),
                    "goal_direction": int(trial_row.get("goal_direction", 0)),
                    "event_index": int(event_index),
                    "edge_side": str(edge["edge_side"]),
                    "frontier_cells": list(int(cell) for cell in frontier_cells),
                    "success": _bool_value(trial_row.get("success")),
                    "returned_from_edge": _bool_value(trial_row.get("returned_from_edge")),
                    "step_count": int(step_count),
                    "trace_npz": str(trace_path),
                    "rows": rows,
                    "time_to_release": time_to_release,
                    "phase_angle": phase_angle[:step_count],
                    "phase_bin": phase_bin[:step_count],
                    "features": {key: value[:step_count] for key, value in stream_features.items()},
                }
            )
    if robot_shape is None:
        raise FileNotFoundError(f"No traces found under {package_root}")
    metadata["robot_shape"] = list(robot_shape)
    metadata["event_count"] = len(events)
    metadata["streams"] = {
        stream: int(events[0]["features"][stream].shape[1]) if events else 0 for stream in STREAM_ORDER
    }
    return events, metadata


def _candidate_indices(
    event: dict[str, Any],
    *,
    clearance_margin: float,
    near_threshold: int,
    far_threshold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = event["rows"]
    clearance = rows["cliff_clearance"].to_numpy(dtype=np.float32)
    ttr = np.asarray(event["time_to_release"], dtype=np.int32)
    pre_contact = (ttr > 0) & (clearance >= float(clearance_margin))
    near = pre_contact & (ttr <= int(near_threshold))
    far = pre_contact & (ttr >= int(far_threshold))
    indices = np.flatnonzero(near | far).astype(np.int32)
    labels = np.where(near[indices], 1, -1).astype(np.int8)
    phase_bins = np.asarray(event["phase_bin"], dtype=np.int32)[indices]
    return indices, labels, phase_bins


def _build_scenario_samples(
    events: list[dict[str, Any]],
    *,
    clearance_margin: float,
    near_threshold: int,
    far_threshold: int,
    phase_bins: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, int]] = set()

    for event in events:
        indices, labels, bins = _candidate_indices(
            event,
            clearance_margin=float(clearance_margin),
            near_threshold=int(near_threshold),
            far_threshold=int(far_threshold),
        )
        for phase_bin in range(int(phase_bins)):
            for label in (-1, 1):
                count_rows.append(
                    {
                        "clearance_margin": float(clearance_margin),
                        "near_threshold": int(near_threshold),
                        "far_threshold": int(far_threshold),
                        "event_id": str(event["event_id"]),
                        "phase_bin": int(phase_bin),
                        "label": int(label),
                        "candidate_count": int(np.sum((bins == phase_bin) & (labels == label))),
                    }
                )
            pos = indices[(bins == phase_bin) & (labels == 1)]
            neg = indices[(bins == phase_bin) & (labels == -1)]
            k = int(min(pos.size, neg.size))
            if k <= 0:
                continue
            pos_choice = rng.choice(pos, size=k, replace=False)
            neg_choice = rng.choice(neg, size=k, replace=False)
            for idx in np.concatenate([pos_choice, neg_choice]).astype(np.int32):
                selected_keys.add((str(event["event_id"]), int(idx)))

        for idx, label, phase_bin in zip(indices, labels, bins, strict=True):
            selected = (str(event["event_id"]), int(idx)) in selected_keys
            if selected:
                rows = event["rows"]
                raw_rows.append(
                    {
                        "event_id": str(event["event_id"]),
                        "trial_id": str(event["trial_id"]),
                        "plateau_width": int(event["plateau_width"]),
                        "distance_pair": str(event["distance_pair"]),
                        "goal_distance": int(event["goal_distance"]),
                        "cliff_distance": int(event["cliff_distance"]),
                        "step_index": int(rows.iloc[int(idx)]["step_index"]),
                        "sample_index": int(idx),
                        "time_to_release": int(event["time_to_release"][int(idx)]),
                        "cliff_clearance": float(rows.iloc[int(idx)]["cliff_clearance"]),
                        "phase_bin": int(phase_bin),
                        "label": int(label),
                        "label_name": "near_about_to_turn" if int(label) == 1 else "far_continue",
                    }
                )
    sample_frame = pd.DataFrame(raw_rows)
    audit_rows: list[dict[str, Any]] = []
    if not sample_frame.empty:
        for (phase_bin, label), group in sample_frame.groupby(["phase_bin", "label"], sort=True):
            audit_rows.append(
                {
                    "clearance_margin": float(clearance_margin),
                    "near_threshold": int(near_threshold),
                    "far_threshold": int(far_threshold),
                    "phase_bin": int(phase_bin),
                    "label": int(label),
                    "selected_count": int(len(group)),
                    "event_count": int(group["event_id"].nunique()),
                }
            )
    return sample_frame, count_rows, audit_rows


def _matrix_for_samples(
    events_by_id: dict[str, dict[str, Any]],
    samples: pd.DataFrame,
    stream: str,
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for event_id, group in samples.groupby("event_id", sort=False):
        event = events_by_id[str(event_id)]
        features = np.asarray(event["features"][stream], dtype=np.float32)
        blocks.append(features[group["sample_index"].to_numpy(dtype=np.int32)])
    if not blocks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(blocks, axis=0).astype(np.float32)


def _pca_project_train_test(
    train_x: np.ndarray,
    test_x: np.ndarray,
    *,
    dim: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    train_z, test_z = _standardize_train_test(train_x, test_x)
    max_dim = int(min(dim, train_z.shape[1], max(1, train_z.shape[0] - 1)))
    if max_dim <= 0:
        return train_z[:, :0], test_z[:, :0], 0
    _, _, vt = np.linalg.svd(train_z, full_matrices=False)
    components = vt[:max_dim].T.astype(np.float32)
    return (train_z @ components).astype(np.float32), (test_z @ components).astype(np.float32), max_dim


def _decode_cv(
    *,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    ridge_alpha: float,
    feature_policy: str,
    common_dim: int,
) -> tuple[np.ndarray, list[dict[str, Any]], str]:
    scores = np.full((len(y),), np.nan, dtype=np.float32)
    fold_rows: list[dict[str, Any]] = []
    unique_groups = sorted(str(value) for value in pd.unique(groups))
    for heldout in unique_groups:
        test_mask = groups.astype(str) == str(heldout)
        train_mask = ~test_mask
        train_y = y[train_mask]
        test_y = y[test_mask]
        base = {
            "heldout_group": str(heldout),
            "train_sample_count": int(np.sum(train_mask)),
            "test_sample_count": int(np.sum(test_mask)),
            "train_near_count": int(np.sum(train_y == 1)),
            "train_far_count": int(np.sum(train_y == -1)),
            "test_near_count": int(np.sum(test_y == 1)),
            "test_far_count": int(np.sum(test_y == -1)),
        }
        if np.sum(train_mask) == 0 or np.sum(test_mask) == 0:
            fold_rows.append({**base, "skip_reason": "empty_train_or_test"})
            continue
        if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
            fold_rows.append({**base, "skip_reason": "single_class_train_or_test"})
            continue
        train_x = x[train_mask]
        test_x = x[test_mask]
        projected_dim = train_x.shape[1]
        if feature_policy == "common_pca":
            train_x, test_x, projected_dim = _pca_project_train_test(train_x, test_x, dim=int(common_dim))
        if train_x.shape[1] == 0:
            fold_rows.append({**base, "skip_reason": "empty_feature_matrix"})
            continue
        fold_scores = _ridge_scores(train_x, train_y, test_x, alpha=float(ridge_alpha))
        scores[test_mask] = fold_scores
        fold_rows.append(
            {
                **base,
                "skip_reason": "",
                "projected_dim": int(projected_dim),
                "auc": _auc(test_y, fold_scores),
                "balanced_accuracy": _balanced_accuracy(test_y, fold_scores),
                "mcc": _mcc(test_y, fold_scores),
            }
        )
    valid = np.isfinite(scores)
    if not np.any(valid):
        return scores, fold_rows, "no_estimable_folds"
    if len(np.unique(y[valid])) < 2:
        return scores, fold_rows, "single_class_pooled_scores"
    return scores, fold_rows, ""


def _metric_summary(
    *,
    y: np.ndarray,
    scores: np.ndarray,
    rng: np.random.Generator,
    permutations: int,
) -> dict[str, float]:
    valid = np.isfinite(scores)
    y_valid = y[valid]
    scores_valid = scores[valid]
    if y_valid.size == 0 or len(np.unique(y_valid)) < 2:
        return {
            "auc": float("nan"),
            "balanced_accuracy": float("nan"),
            "mcc": float("nan"),
            "null_auc_mean": float("nan"),
            "null_auc_q95": float("nan"),
            "empirical_p_auc": float("nan"),
        }
    auc = _auc(y_valid, scores_valid)
    null_auc = np.asarray(
        [_auc(rng.permutation(y_valid), scores_valid) for _ in range(int(permutations))],
        dtype=np.float32,
    )
    return {
        "auc": auc,
        "balanced_accuracy": _balanced_accuracy(y_valid, scores_valid),
        "mcc": _mcc(y_valid, scores_valid),
        "null_auc_mean": float(np.nanmean(null_auc)),
        "null_auc_q95": float(np.nanquantile(null_auc, 0.95)),
        "empirical_p_auc": float((1.0 + np.sum(null_auc >= auc)) / (1.0 + len(null_auc))),
    }


def _evaluate_scenarios(
    *,
    events: list[dict[str, Any]],
    scenario_frames: dict[tuple[float, int, int], pd.DataFrame],
    streams: list[str],
    cv_modes: list[str],
    ridge_alpha: float,
    permutations: int,
    random_seed: int,
    common_dim: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events_by_id = {str(event["event_id"]): event for event in events}
    metric_rows: list[dict[str, Any]] = []
    fold_rows_all: list[dict[str, Any]] = []
    for scenario, samples in scenario_frames.items():
        clearance_margin, near_threshold, far_threshold = scenario
        if samples.empty:
            continue
        samples = samples.reset_index(drop=True)
        y = samples["label"].to_numpy(dtype=np.int8)
        groups_by_mode = {
            "event": samples["event_id"].astype(str).to_numpy(),
            "distance_pair": samples["distance_pair"].astype(str).to_numpy(),
            "plateau_width": samples["plateau_width"].astype(str).to_numpy(),
        }
        is_primary = (
            abs(float(clearance_margin) - PRIMARY_CLEARANCE_MARGIN) < 1.0e-9
            and int(near_threshold) == PRIMARY_NEAR_THRESHOLD
            and int(far_threshold) == DEFAULT_FAR_THRESHOLD
        )
        for stream in streams:
            x = _matrix_for_samples(events_by_id, samples, stream)
            if x.shape[0] != len(samples):
                raise ValueError(f"Feature/sample length mismatch for {stream}: {x.shape[0]} vs {len(samples)}")
            feature_policies = ["raw", "common_pca"] if is_primary else ["raw"]
            for cv_mode in cv_modes:
                groups = groups_by_mode[cv_mode]
                for feature_policy in feature_policies:
                    rng = np.random.default_rng(
                        _stable_seed(
                            int(random_seed),
                            f"{stream}-{cv_mode}-{feature_policy}-{clearance_margin}-{near_threshold}-{far_threshold}",
                        )
                    )
                    scores, fold_rows, skip_reason = _decode_cv(
                        x=x,
                        y=y,
                        groups=groups,
                        ridge_alpha=float(ridge_alpha),
                        feature_policy=feature_policy,
                        common_dim=int(common_dim),
                    )
                    summary = _metric_summary(y=y, scores=scores, rng=rng, permutations=int(permutations))
                    valid = np.isfinite(scores)
                    row = {
                        "clearance_margin": float(clearance_margin),
                        "near_threshold": int(near_threshold),
                        "far_threshold": int(far_threshold),
                        "stream": stream,
                        "stream_label": STREAM_LABELS.get(stream, stream),
                        "feature_policy": feature_policy,
                        "cv_mode": cv_mode,
                        "feature_dim": int(x.shape[1]),
                        "common_dim": int(common_dim) if feature_policy == "common_pca" else "",
                        "sample_count": int(len(samples)),
                        "scored_sample_count": int(np.sum(valid)),
                        "near_count": int(np.sum(y == 1)),
                        "far_count": int(np.sum(y == -1)),
                        "scored_near_count": int(np.sum(y[valid] == 1)),
                        "scored_far_count": int(np.sum(y[valid] == -1)),
                        "heldout_group_count": int(len(set(groups))),
                        "estimable_fold_count": int(sum(1 for fold in fold_rows if not fold.get("skip_reason"))),
                        "skip_reason": skip_reason,
                        "ridge_alpha": float(ridge_alpha),
                        "permutations": int(permutations),
                        **summary,
                    }
                    metric_rows.append(row)
                    for fold in fold_rows:
                        fold_rows_all.append(
                            {
                                "clearance_margin": float(clearance_margin),
                                "near_threshold": int(near_threshold),
                                "far_threshold": int(far_threshold),
                                "stream": stream,
                                "feature_policy": feature_policy,
                                "cv_mode": cv_mode,
                                **fold,
                            }
                        )
    return metric_rows, fold_rows_all


def _ridge_regression_scores(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, *, alpha: float) -> np.ndarray:
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


def _continuous_proximity_rows(
    *,
    events: list[dict[str, Any]],
    streams: list[str],
    clearance_margin: float,
    min_ttr: int,
    max_ttr: int,
    ridge_alpha: float,
    random_seed: int,
    max_samples_per_event: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(_stable_seed(int(random_seed), "continuous-proximity"))
    rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for event in events:
        frame = event["rows"]
        clearance = frame["cliff_clearance"].to_numpy(dtype=np.float32)
        ttr = np.asarray(event["time_to_release"], dtype=np.int32)
        mask = (ttr >= int(min_ttr)) & (ttr <= int(max_ttr)) & (clearance >= float(clearance_margin))
        indices = np.flatnonzero(mask).astype(np.int32)
        if indices.size > int(max_samples_per_event):
            indices = np.sort(rng.choice(indices, size=int(max_samples_per_event), replace=False)).astype(np.int32)
        for idx in indices:
            sample_rows.append(
                {
                    "event_id": str(event["event_id"]),
                    "distance_pair": str(event["distance_pair"]),
                    "plateau_width": int(event["plateau_width"]),
                    "sample_index": int(idx),
                    "log_time_to_release": float(np.log1p(float(ttr[int(idx)]))),
                }
            )
    samples = pd.DataFrame(sample_rows)
    if samples.empty:
        return rows
    y = samples["log_time_to_release"].to_numpy(dtype=np.float32)
    groups_by_mode = {
        "event": samples["event_id"].astype(str).to_numpy(),
        "distance_pair": samples["distance_pair"].astype(str).to_numpy(),
        "plateau_width": samples["plateau_width"].astype(str).to_numpy(),
    }
    events_by_id = {str(event["event_id"]): event for event in events}
    for stream in streams:
        x = _matrix_for_samples(events_by_id, samples, stream)
        for cv_mode, groups in groups_by_mode.items():
            preds = np.full((len(y),), np.nan, dtype=np.float32)
            estimable = 0
            for heldout in sorted(str(value) for value in pd.unique(groups)):
                test = groups.astype(str) == heldout
                train = ~test
                if np.sum(train) < 8 or np.sum(test) < 2:
                    continue
                preds[test] = _ridge_regression_scores(x[train], y[train], x[test], alpha=float(ridge_alpha))
                estimable += 1
            valid = np.isfinite(preds)
            if np.sum(valid) < 4:
                rows.append(
                    {
                        "stream": stream,
                        "cv_mode": cv_mode,
                        "sample_count": int(len(y)),
                        "scored_sample_count": int(np.sum(valid)),
                        "estimable_fold_count": int(estimable),
                        "r2": float("nan"),
                        "pearson_r": float("nan"),
                        "mae": float("nan"),
                    }
                )
                continue
            yv = y[valid]
            pv = preds[valid]
            ss_res = float(np.sum((yv - pv) ** 2))
            ss_tot = float(np.sum((yv - float(np.mean(yv))) ** 2))
            rows.append(
                {
                    "stream": stream,
                    "cv_mode": cv_mode,
                    "sample_count": int(len(y)),
                    "scored_sample_count": int(np.sum(valid)),
                    "estimable_fold_count": int(estimable),
                    "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                    "pearson_r": float(np.corrcoef(yv, pv)[0, 1]) if np.std(yv) > 0 and np.std(pv) > 0 else float("nan"),
                    "mae": float(np.mean(np.abs(yv - pv))),
                }
            )
    return rows


def _plot_primary_auc(metrics: pd.DataFrame, output_dir: Path) -> None:
    subset = metrics[
        (metrics["clearance_margin"] == PRIMARY_CLEARANCE_MARGIN)
        & (metrics["near_threshold"] == PRIMARY_NEAR_THRESHOLD)
        & (metrics["far_threshold"] == DEFAULT_FAR_THRESHOLD)
        & (metrics["cv_mode"] == "event")
        & (metrics["feature_policy"] == "raw")
    ].copy()
    subset["stream"] = pd.Categorical(subset["stream"], categories=STREAM_ORDER, ordered=True)
    subset = subset.sort_values("stream")
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    x = np.arange(len(subset))
    colors = [STREAM_COLORS.get(str(row["stream"]), "#999999") for _, row in subset.iterrows()]
    ax.bar(x, subset["auc"].to_numpy(dtype=float), color=colors, edgecolor="white", linewidth=0.6)
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=0.8)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("AUC")
    ax.set_title("Pre-contact near-release decoding", fontsize=10, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([STREAM_LABELS.get(str(s), str(s)) for s in subset["stream"]], rotation=25, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "precontact_primary_auc.png", dpi=240)
    fig.savefig(output_dir / "precontact_primary_auc.pdf")
    plt.close(fig)


def _plot_cv_controls(metrics: pd.DataFrame, output_dir: Path) -> None:
    subset = metrics[
        (metrics["clearance_margin"] == PRIMARY_CLEARANCE_MARGIN)
        & (metrics["near_threshold"] == PRIMARY_NEAR_THRESHOLD)
        & (metrics["far_threshold"] == DEFAULT_FAR_THRESHOLD)
        & (metrics["feature_policy"] == "raw")
        & (metrics["stream"].isin(["mechanical_frontier", "signal_frontier", "phase_only", "position_leak"]))
    ].copy()
    cv_order = ["event", "distance_pair", "plateau_width"]
    stream_order = ["mechanical_frontier", "signal_frontier", "phase_only", "position_leak"]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    width = 0.18
    x = np.arange(len(cv_order))
    for offset, stream in enumerate(stream_order):
        vals = []
        for cv in cv_order:
            row = subset[(subset["cv_mode"] == cv) & (subset["stream"] == stream)]
            vals.append(float(row.iloc[0]["auc"]) if not row.empty else np.nan)
        ax.bar(
            x + (offset - 1.5) * width,
            vals,
            width=width,
            color=STREAM_COLORS.get(stream, "#999999"),
            edgecolor="white",
            linewidth=0.5,
            label=STREAM_LABELS.get(stream, stream),
        )
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=0.8)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("AUC")
    ax.set_title("Held-out group controls", fontsize=10, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["event", "distance", "plateau"], fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "precontact_cv_controls.png", dpi=240)
    fig.savefig(output_dir / "precontact_cv_controls.pdf")
    plt.close(fig)


def _plot_margin_sensitivity(metrics: pd.DataFrame, output_dir: Path) -> None:
    subset = metrics[
        (metrics["near_threshold"] == PRIMARY_NEAR_THRESHOLD)
        & (metrics["far_threshold"] == DEFAULT_FAR_THRESHOLD)
        & (metrics["cv_mode"] == "event")
        & (metrics["feature_policy"] == "raw")
        & (metrics["stream"].isin(["mechanical_frontier", "signal_frontier", "phase_only", "position_leak"]))
    ].copy()
    margins = sorted(float(v) for v in subset["clearance_margin"].dropna().unique())
    if len(margins) < 2:
        return
    stream_order = ["mechanical_frontier", "signal_frontier", "phase_only", "position_leak"]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for stream in stream_order:
        vals = []
        for margin in margins:
            row = subset[(subset["clearance_margin"] == margin) & (subset["stream"] == stream)]
            vals.append(float(row.iloc[0]["auc"]) if not row.empty else np.nan)
        ax.plot(
            margins,
            vals,
            marker="o",
            linewidth=1.4,
            markersize=4,
            color=STREAM_COLORS.get(stream, "#999999"),
            label=STREAM_LABELS.get(stream, stream),
        )
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=0.8)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("Minimum clearance before release")
    ax.set_ylabel("AUC")
    ax.set_title("Pre-contact clearance sensitivity", fontsize=10, weight="bold")
    ax.tick_params(labelsize=8)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "precontact_margin_sensitivity.png", dpi=240)
    fig.savefig(output_dir / "precontact_margin_sensitivity.pdf")
    plt.close(fig)


def _write_report(
    output_dir: Path,
    *,
    package_root: Path,
    metadata: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    continuous_rows: list[dict[str, Any]],
    sample_rows: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    metrics = pd.DataFrame(metric_rows)
    primary = metrics[
        (metrics["clearance_margin"] == PRIMARY_CLEARANCE_MARGIN)
        & (metrics["near_threshold"] == PRIMARY_NEAR_THRESHOLD)
        & (metrics["far_threshold"] == DEFAULT_FAR_THRESHOLD)
        & (metrics["cv_mode"] == "event")
        & (metrics["feature_policy"] == "raw")
    ].copy()
    primary["stream"] = pd.Categorical(primary["stream"], categories=STREAM_ORDER, ordered=True)
    primary = primary.sort_values("stream")
    common = metrics[
        (metrics["clearance_margin"] == PRIMARY_CLEARANCE_MARGIN)
        & (metrics["near_threshold"] == PRIMARY_NEAR_THRESHOLD)
        & (metrics["far_threshold"] == DEFAULT_FAR_THRESHOLD)
        & (metrics["cv_mode"] == "event")
        & (metrics["feature_policy"] == "common_pca")
    ].copy()
    common["stream"] = pd.Categorical(common["stream"], categories=STREAM_ORDER, ordered=True)
    common = common.sort_values("stream")

    lines = [
        "# S4 pre-contact continue-vs-turn decoder",
        "",
        "This analysis uses the canonical A100-40GB S4 deformation-off goal-cliff expansion. It is separate from the earlier edge-impulse decoder: the target here is pre-contact proximity to release, not sustained reversal success versus failure.",
        "",
        "## Source",
        "",
        f"- Source package: `{package_root}`",
        f"- Events loaded: `{metadata.get('event_count')}`",
        f"- Streams and feature dimensions: `{metadata.get('streams')}`",
        f"- Hidden start channel: `{metadata.get('hidden_start_channel')}`",
        f"- Phase bins: `{metadata.get('phase_bins')}`",
        "",
        "## Method",
        "",
        f"- Candidate samples are before release and require `cliff_clearance >= margin`; the primary margin is `{PRIMARY_CLEARANCE_MARGIN}`.",
        f"- Binary label: `near/about-to-turn = time_to_release <= {PRIMARY_NEAR_THRESHOLD}`; `far/continue = time_to_release >= {DEFAULT_FAR_THRESHOLD}`.",
        "- Intermediate samples are excluded.",
        "- Samples are balanced within locomotor-phase bins before decoding.",
        "- Cross-validation holds out whole events, distance pairs, or plateau widths. Timepoints from the same rollout are never split independently across train/test.",
        "- The strict mechanical stream uses frontier-band spring state plus local relative geometry and point velocities. It excludes COM, cliff clearance, terrain, and absolute/global x.",
        "- The signal stream uses frontier-band NCA hidden channels plus CTRNN carried state. It excludes deformation inputs, action outputs, body geometry, COM, terrain, and cliff geometry.",
        "- `phase_only` is a rhythm-control decoder from the phase proxy alone.",
        "- `position_leak` deliberately includes clearance/global-position variables as a positive leakage baseline.",
        f"- Null p-values are score-label permutation checks with `{int(args.permutations)}` permutations.",
        "",
        "## Primary Event-Held-Out Result",
        "",
    ]
    if primary.empty:
        lines.append("No primary event-held-out rows were estimable.")
    else:
        lines.append("| Stream | Feature dim | Samples | AUC | Balanced accuracy | p(AUC) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, row in primary.iterrows():
            lines.append(
                f"| {row['stream_label']} | {int(row['feature_dim'])} | {int(row['scored_sample_count'])} | "
                f"{float(row['auc']):.3f} | {float(row['balanced_accuracy']):.3f} | {float(row['empirical_p_auc']):.4f} |"
            )
    lines.extend(["", "## Common-Dimension Control", ""])
    if common.empty:
        lines.append("No common-dimension rows were estimable.")
    else:
        lines.append(f"Each stream was PCA-reduced inside each training fold to at most `{int(args.common_dim)}` dimensions.")
        lines.append("")
        lines.append("| Stream | AUC | Balanced accuracy | p(AUC) |")
        lines.append("|---|---:|---:|---:|")
        for _, row in common.iterrows():
            lines.append(
                f"| {row['stream_label']} | {float(row['auc']):.3f} | "
                f"{float(row['balanced_accuracy']):.3f} | {float(row['empirical_p_auc']):.4f} |"
            )
    lines.extend(["", "## Group-Held-Out Controls", ""])
    group = metrics[
        (metrics["clearance_margin"] == PRIMARY_CLEARANCE_MARGIN)
        & (metrics["near_threshold"] == PRIMARY_NEAR_THRESHOLD)
        & (metrics["far_threshold"] == DEFAULT_FAR_THRESHOLD)
        & (metrics["feature_policy"] == "raw")
    ].copy()
    if not group.empty:
        lines.append("| CV mode | Mechanical AUC | Signal AUC | Phase-only AUC | Position-leak AUC |")
        lines.append("|---|---:|---:|---:|---:|")
        for cv_mode in ["event", "distance_pair", "plateau_width"]:
            vals: dict[str, float] = {}
            for stream in ["mechanical_frontier", "signal_frontier", "phase_only", "position_leak"]:
                row = group[(group["cv_mode"] == cv_mode) & (group["stream"] == stream)]
                vals[stream] = float(row.iloc[0]["auc"]) if not row.empty else float("nan")
            lines.append(
                f"| {cv_mode} | {vals['mechanical_frontier']:.3f} | {vals['signal_frontier']:.3f} | "
                f"{vals['phase_only']:.3f} | {vals['position_leak']:.3f} |"
            )
    lines.extend(["", "## Sample Balance", ""])
    if not sample_rows.empty:
        lines.append(
            f"Primary phase-matched sample set: `{len(sample_rows)}` samples from "
            f"`{sample_rows['event_id'].nunique()}` events; "
            f"`{int(np.sum(sample_rows['label'].to_numpy(dtype=int) == 1))}` near and "
            f"`{int(np.sum(sample_rows['label'].to_numpy(dtype=int) == -1))}` far."
        )
    lines.extend(["", "## Continuous Proximity Probe", ""])
    if continuous_rows:
        cont = pd.DataFrame(continuous_rows)
        focus = cont[cont["cv_mode"] == "event"].copy()
        focus["stream"] = pd.Categorical(focus["stream"], categories=STREAM_ORDER, ordered=True)
        focus = focus.sort_values("stream")
        lines.append("Continuous probe predicts `log1p(time_to_release)` from pre-contact samples. This is exploratory and not the main figure target.")
        lines.append("")
        lines.append("| Stream | Event-held-out R2 | Pearson r | MAE |")
        lines.append("|---|---:|---:|---:|")
        for _, row in focus.iterrows():
            lines.append(
                f"| {STREAM_LABELS.get(str(row['stream']), str(row['stream']))} | "
                f"{float(row['r2']):.3f} | {float(row['pearson_r']):.3f} | {float(row['mae']):.3f} |"
            )
    else:
        lines.append("Continuous proximity probe was not run.")
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- A positive body result here would mean pre-contact body-internal state contains proximity-to-release structure under this deterministic S4 deformation-off expansion.",
            "- It does not by itself prove causal body control.",
            "- If `phase_only` approaches the body AUC, the result is mainly rhythm/phase separability.",
            "- If only `position_leak` is strong, the real body/signal streams do not support a pre-contact sensing claim.",
            "- Signal predictivity should be described as controller rhythm/proximity readout unless it survives the controls in a way distinct from phase alone.",
            "",
            "## Outputs",
            "",
            "- `precontact_decoder_metrics.csv`",
            "- `precontact_decoder_folds.csv`",
            "- `precontact_primary_samples.csv`",
            "- `precontact_phase_candidate_counts.csv`",
            "- `precontact_phase_balance_audit.csv`",
            "- `continuous_proximity_metrics.csv`",
            "- `precontact_primary_auc.png/pdf`",
            "- `precontact_cv_controls.png/pdf`",
        ]
    )
    if len(set(float(row["clearance_margin"]) for row in metric_rows)) > 1:
        lines.append("- `precontact_margin_sensitivity.png/pdf`")
    else:
        lines.append(
            "- Margin sensitivity was not generated in this targeted primary run; pass multiple `--clearance-margins` to generate it."
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    package_root = args.package_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    events, metadata = _load_events(package_root, phase_bins=int(args.phase_bins))
    event_rows = [
        {
            "event_id": event["event_id"],
            "plateau_width": int(event["plateau_width"]),
            "trial_id": str(event["trial_id"]),
            "distance_pair": str(event["distance_pair"]),
            "goal_distance": int(event["goal_distance"]),
            "cliff_distance": int(event["cliff_distance"]),
            "event_index": int(event["event_index"]),
            "edge_side": str(event["edge_side"]),
            "frontier_cells": " ".join(str(cell) for cell in event["frontier_cells"]),
            "success": bool(event["success"]),
            "returned_from_edge": bool(event["returned_from_edge"]),
            "step_count": int(event["step_count"]),
            "trace_npz": str(event["trace_npz"]),
        }
        for event in events
    ]

    margins = [float(value) for value in args.clearance_margins]
    near_thresholds = [int(value) for value in args.near_thresholds]
    if args.smoke:
        margins = [PRIMARY_CLEARANCE_MARGIN]
        near_thresholds = [PRIMARY_NEAR_THRESHOLD]
    scenario_frames: dict[tuple[float, int, int], pd.DataFrame] = {}
    candidate_count_rows: list[dict[str, Any]] = []
    balance_rows: list[dict[str, Any]] = []
    primary_samples = pd.DataFrame()
    for margin in margins:
        for near in near_thresholds:
            rng = np.random.default_rng(_stable_seed(int(args.random_seed), f"phase-match-{margin}-{near}-{args.far_threshold}"))
            samples, counts, audit = _build_scenario_samples(
                events,
                clearance_margin=float(margin),
                near_threshold=int(near),
                far_threshold=int(args.far_threshold),
                phase_bins=int(args.phase_bins),
                rng=rng,
            )
            scenario = (float(margin), int(near), int(args.far_threshold))
            scenario_frames[scenario] = samples
            candidate_count_rows.extend(counts)
            balance_rows.extend(audit)
            if (
                abs(float(margin) - PRIMARY_CLEARANCE_MARGIN) < 1.0e-9
                and int(near) == PRIMARY_NEAR_THRESHOLD
                and int(args.far_threshold) == DEFAULT_FAR_THRESHOLD
            ):
                primary_samples = samples.copy()

    streams = list(STREAM_ORDER)
    if args.smoke:
        streams = ["mechanical_frontier", "signal_frontier", "phase_only", "position_leak"]
    cv_modes = ["event", "distance_pair", "plateau_width"]
    metric_rows, fold_rows = _evaluate_scenarios(
        events=events,
        scenario_frames=scenario_frames,
        streams=streams,
        cv_modes=cv_modes,
        ridge_alpha=float(args.ridge_alpha),
        permutations=int(args.permutations),
        random_seed=int(args.random_seed),
        common_dim=int(args.common_dim),
    )
    continuous_rows: list[dict[str, Any]] = []
    if not args.skip_continuous:
        continuous_rows = _continuous_proximity_rows(
            events=events,
            streams=streams,
            clearance_margin=PRIMARY_CLEARANCE_MARGIN,
            min_ttr=int(args.continuous_min_ttr),
            max_ttr=int(args.continuous_max_ttr),
            ridge_alpha=float(args.ridge_alpha),
            random_seed=int(args.random_seed),
            max_samples_per_event=int(args.continuous_max_samples_per_event),
        )

    _write_csv(output_dir / "precontact_event_manifest.csv", event_rows)
    _write_csv(output_dir / "precontact_decoder_metrics.csv", metric_rows)
    _write_csv(output_dir / "precontact_decoder_folds.csv", fold_rows)
    _write_csv(output_dir / "precontact_phase_candidate_counts.csv", candidate_count_rows)
    _write_csv(output_dir / "precontact_phase_balance_audit.csv", balance_rows)
    _write_csv(output_dir / "continuous_proximity_metrics.csv", continuous_rows)
    if primary_samples.empty:
        (output_dir / "precontact_primary_samples.csv").write_text("")
    else:
        primary_samples.to_csv(output_dir / "precontact_primary_samples.csv", index=False)

    metrics_frame = pd.DataFrame(metric_rows)
    if not metrics_frame.empty:
        figures_dir = output_dir / "figures"
        _plot_primary_auc(metrics_frame, figures_dir)
        _plot_cv_controls(metrics_frame, figures_dir)
        _plot_margin_sensitivity(metrics_frame, figures_dir)

    summary = {
        "schema_version": 1,
        "kind": "s4_precontact_proximity_decoder",
        "package_root": str(package_root),
        "output_dir": str(output_dir),
        "metadata": metadata,
        "primary_design": {
            "clearance_margin": PRIMARY_CLEARANCE_MARGIN,
            "near_threshold": PRIMARY_NEAR_THRESHOLD,
            "far_threshold": DEFAULT_FAR_THRESHOLD,
            "phase_bins": int(args.phase_bins),
            "phase_matching": "deterministic downsampling to equal near/far counts within event-local phase bins",
        },
        "decoder": {
            "streams": streams,
            "cv_modes": cv_modes,
            "ridge_alpha": float(args.ridge_alpha),
            "permutations": int(args.permutations),
            "null_policy": "score-label permutation over pooled held-out predictions",
            "common_dim": int(args.common_dim),
        },
        "outputs": {
            "event_manifest": str(output_dir / "precontact_event_manifest.csv"),
            "decoder_metrics": str(output_dir / "precontact_decoder_metrics.csv"),
            "decoder_folds": str(output_dir / "precontact_decoder_folds.csv"),
            "primary_samples": str(output_dir / "precontact_primary_samples.csv"),
            "phase_candidate_counts": str(output_dir / "precontact_phase_candidate_counts.csv"),
            "phase_balance_audit": str(output_dir / "precontact_phase_balance_audit.csv"),
            "continuous_proximity": str(output_dir / "continuous_proximity_metrics.csv"),
            "figures": str(output_dir / "figures"),
            "readme": str(output_dir / "README.md"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    _write_report(
        output_dir,
        package_root=package_root,
        metadata=metadata,
        metric_rows=metric_rows,
        continuous_rows=continuous_rows,
        sample_rows=primary_samples,
        args=args,
    )
    print(json.dumps(summary["outputs"], indent=2))
    return summary


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clearance-margins", type=_parse_float_list, default=_parse_float_list("0.05"))
    parser.add_argument("--near-thresholds", type=_parse_int_list, default=_parse_int_list("128"))
    parser.add_argument("--far-threshold", type=int, default=DEFAULT_FAR_THRESHOLD)
    parser.add_argument("--phase-bins", type=int, default=DEFAULT_PHASE_BINS)
    parser.add_argument("--ridge-alpha", type=float, default=DEFAULT_RIDGE_ALPHA)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--common-dim", type=int, default=DEFAULT_COMMON_DIM)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--skip-continuous", action="store_true")
    parser.add_argument("--continuous-min-ttr", type=int, default=32)
    parser.add_argument("--continuous-max-ttr", type=int, default=1024)
    parser.add_argument("--continuous-max-samples-per-event", type=int, default=400)
    parser.add_argument("--smoke", action="store_true", help="Run the primary scenario only with the smaller stream set.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
