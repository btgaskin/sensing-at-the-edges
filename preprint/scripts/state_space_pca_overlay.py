#!/usr/bin/env python
"""Generate shared 3D PCA overlays for dense plant and drive state traces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib.animation import FFMpegWriter, PillowWriter, FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DEFAULT_TRACE_ROOT = Path(
    "paper_data/preprint/native_core_ablation_v1/"
    "raw_modal_state_space_20260607T145100/"
    "preprint_native_state_space_v1/"
    "paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed137_e4fcc2fb/"
    "bin_03_rank_00_cell_6-0-3_flat_0423/"
    "state_space_trace"
)
DEFAULT_OUTPUT_DIR = Path(
    "paper_data/preprint/native_core_ablation_v1/figures/state_space_pca_overlay"
)

PLANT_SCALAR_COLUMNS = (
    "com_y",
    "min_y",
    "max_y",
    "com_x_from_start",
    "com_x_to_goal",
    "com_x_to_cliff",
    "goal_distance_remaining",
    "cliff_clearance",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


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
            "env_step",
            "cliff_side",
            "cliff_frontier_x",
            "cliff_clearance",
            "trace_npz",
            *PLANT_SCALAR_COLUMNS,
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


def _orient_x_to_cliff(values: np.ndarray, *, cliff_side: str, cliff_frontier_x: np.ndarray) -> np.ndarray:
    frontier = np.asarray(cliff_frontier_x, dtype=np.float32)
    while frontier.ndim < values.ndim:
        frontier = frontier[..., None]
    if str(cliff_side).strip().lower() == "left":
        return values - frontier
    return frontier - values


def _orient_velocity_x_to_cliff(values: np.ndarray, *, cliff_side: str) -> np.ndarray:
    if str(cliff_side).strip().lower() == "left":
        return values
    return -values


def _flatten_float(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    return array.reshape((array.shape[0], -1))


def _trial_features(
    *,
    trace_root: Path,
    trial_id: str,
    rows: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rows = rows.sort_values("step_index", kind="stable").reset_index(drop=True)
    trace_path = _trace_path_for_trial(trace_root, rows)
    with np.load(trace_path) as trace:
        step_count = min(len(rows), int(trace["plant_positions"].shape[0]))
        rows = rows.iloc[:step_count].reset_index(drop=True)
        cliff_side = str(rows.iloc[0]["cliff_side"])
        frontier = rows["cliff_frontier_x"].to_numpy(dtype=np.float32)

        positions = np.asarray(trace["plant_positions"][:step_count], dtype=np.float32).copy()
        positions[..., 0] = _orient_x_to_cliff(
            positions[..., 0],
            cliff_side=cliff_side,
            cliff_frontier_x=frontier,
        )
        velocities = np.asarray(trace["plant_velocities"][:step_count], dtype=np.float32).copy()
        velocities[..., 0] = _orient_velocity_x_to_cliff(
            velocities[..., 0],
            cliff_side=cliff_side,
        )
        corners = np.asarray(trace["cell_corner_positions"][:step_count], dtype=np.float32).copy()
        corners[..., 0] = _orient_x_to_cliff(
            corners[..., 0],
            cliff_side=cliff_side,
            cliff_frontier_x=frontier,
        )
        plant_scalars = rows.loc[:, PLANT_SCALAR_COLUMNS].to_numpy(dtype=np.float32)
        plant = np.concatenate(
            [
                _flatten_float(positions),
                _flatten_float(velocities),
                _flatten_float(corners),
                _flatten_float(trace["spring_current_length"][:step_count]),
                _flatten_float(trace["spring_rest_length"][:step_count]),
                plant_scalars,
            ],
            axis=1,
        )
        drive = np.concatenate(
            [
                _flatten_float(trace["phase_h_drive"][:step_count]),
                _flatten_float(trace["phase_v_drive"][:step_count]),
                _flatten_float(trace["phase_action_outputs"][:step_count]),
                _flatten_float(trace["phase_nca_state"][:step_count]),
                _flatten_float(trace["phase_ctrnn_carry"][:step_count]),
            ],
            axis=1,
        )

    clearances = rows["cliff_clearance"].to_numpy(dtype=np.float32)
    event_positions = np.flatnonzero(clearances <= 0.0)
    event_local_index = int(event_positions[0]) if len(event_positions) else None
    metadata = {
        "trial_id": trial_id,
        "trace_npz": str(trace_path),
        "step_count": int(len(rows)),
        "distance_pair": str(rows.iloc[0]["distance_pair"]),
        "side": str(rows.iloc[0]["side"]),
        "cliff_side": str(rows.iloc[0]["cliff_side"]),
        "goal_distance": int(rows.iloc[0]["goal_distance"]),
        "cliff_distance": int(rows.iloc[0]["cliff_distance"]),
        "event_local_index": event_local_index,
        "event_env_step": (
            int(rows.iloc[event_local_index]["env_step"])
            if event_local_index is not None
            else None
        ),
        "min_cliff_clearance": float(np.min(clearances)),
        "final_env_step": int(rows.iloc[-1]["env_step"]),
    }
    return plant, drive, metadata


def _standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale > 1.0e-6, scale, 1.0).astype(np.float32)
    return ((features - mean) / scale).astype(np.float32), mean, scale


def _fit_shared_pca(
    plant_features: np.ndarray,
    drive_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    plant_z, plant_mean, plant_scale = _standardize(plant_features)
    drive_z, drive_mean, drive_scale = _standardize(drive_features)
    zero_plant = np.zeros_like(plant_z)
    zero_drive = np.zeros_like(drive_z)
    plant_full = np.concatenate([plant_z, zero_drive], axis=1)
    drive_full = np.concatenate([zero_plant, drive_z], axis=1)
    combined_full = np.concatenate([plant_z, drive_z], axis=1)
    fit_matrix = np.vstack([plant_full, drive_full]).astype(np.float32)
    pca_mean = fit_matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = fit_matrix - pca_mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:3].astype(np.float32)
    denom = max(fit_matrix.shape[0] - 1, 1)
    variances = (singular_values.astype(np.float64) ** 2) / denom
    explained = (variances[:3] / np.maximum(np.sum(variances), 1.0e-12)).astype(float)
    plant_coords = (plant_full - pca_mean) @ components.T
    drive_coords = (drive_full - pca_mean) @ components.T
    combined_coords = (combined_full - pca_mean) @ components.T
    model = {
        "plant_feature_count": int(plant_features.shape[1]),
        "drive_feature_count": int(drive_features.shape[1]),
        "sample_count": int(plant_features.shape[0]),
        "fit_sample_count": int(fit_matrix.shape[0]),
        "explained_variance_ratio": [float(value) for value in explained],
        "plant_mean_shape": list(plant_mean.shape),
        "plant_scale_shape": list(plant_scale.shape),
        "drive_mean_shape": list(drive_mean.shape),
        "drive_scale_shape": list(drive_scale.shape),
    }
    return (
        plant_coords.astype(np.float32),
        drive_coords.astype(np.float32),
        combined_coords.astype(np.float32),
        model,
    )


def _write_projected_csv(
    path: Path,
    *,
    step_manifest: pd.DataFrame,
    trial_metadata: list[dict[str, Any]],
    plant_coords: np.ndarray,
    drive_coords: np.ndarray,
    combined_coords: np.ndarray,
) -> None:
    rows_by_trial = {
        trial_id: frame.sort_values("step_index", kind="stable").reset_index(drop=True)
        for trial_id, frame in step_manifest.groupby("trial_id", sort=False)
    }
    meta_by_trial = {meta["trial_id"]: meta for meta in trial_metadata}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "trial_id",
                "stream",
                "env_step",
                "step_index",
                "side",
                "distance_pair",
                "cliff_clearance",
                "is_cliff_reach_event",
                "pc1",
                "pc2",
                "pc3",
            ],
        )
        writer.writeheader()
        offset = 0
        for meta in trial_metadata:
            trial_id = meta["trial_id"]
            rows = rows_by_trial[trial_id]
            count = int(meta["step_count"])
            slices = {
                "plant": plant_coords[offset : offset + count],
                "drive": drive_coords[offset : offset + count],
                "combined": combined_coords[offset : offset + count],
            }
            for stream, coords in slices.items():
                for local_index, coord in enumerate(coords):
                    row = rows.iloc[local_index]
                    writer.writerow(
                        {
                            "trial_id": trial_id,
                            "stream": stream,
                            "env_step": int(row["env_step"]),
                            "step_index": int(row["step_index"]),
                            "side": str(row["side"]),
                            "distance_pair": str(row["distance_pair"]),
                            "cliff_clearance": float(row["cliff_clearance"]),
                            "is_cliff_reach_event": bool(
                                meta_by_trial[trial_id]["event_local_index"] == local_index
                            ),
                            "pc1": float(coord[0]),
                            "pc2": float(coord[1]),
                            "pc3": float(coord[2]),
                        }
                    )
            offset += count


def _downsample_indices(count: int, *, max_points: int, keep_index: int | None = None) -> np.ndarray:
    if count <= max_points:
        return np.arange(count, dtype=np.int64)
    indices = np.linspace(0, count - 1, max_points, dtype=np.int64)
    if keep_index is not None:
        indices = np.unique(np.concatenate([indices, np.asarray([keep_index], dtype=np.int64)]))
    return indices


def _set_equal_axes(ax, coords: list[np.ndarray]) -> None:
    points = np.concatenate([c.reshape((-1, 3)) for c in coords if c.size], axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    if not np.isfinite(radius) or radius <= 0.0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _line3d_set_point(line, point: np.ndarray) -> None:
    line.set_data([float(point[0])], [float(point[1])])
    line.set_3d_properties([float(point[2])])


def _line3d_set_path(line, points: np.ndarray) -> None:
    if points.size == 0:
        line.set_data([], [])
        line.set_3d_properties([])
        return
    line.set_data(points[:, 0], points[:, 1])
    line.set_3d_properties(points[:, 2])


def _animation_entries(
    *,
    trial_metadata: list[dict[str, Any]],
    plant_coords: np.ndarray,
    drive_coords: np.ndarray,
    event_window: tuple[int, int],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    offset = 0
    before, after = event_window
    for meta in trial_metadata:
        count = int(meta["step_count"])
        event_index = meta.get("event_local_index")
        if event_index is None:
            offset += count
            continue
        event_index = int(event_index)
        start = max(0, event_index - int(before))
        stop = min(count, event_index + int(after) + 1)
        entries.append(
            {
                "meta": meta,
                "relative_steps": np.arange(start, stop, dtype=np.int32) - event_index,
                "plant": plant_coords[offset + start : offset + stop],
                "drive": drive_coords[offset + start : offset + stop],
                "event_local_window_index": event_index - start,
            }
        )
        offset += count
    return entries


def _points_through_relative_step(
    entry: dict[str, Any],
    *,
    stream: str,
    relative_step: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    relative_steps = entry["relative_steps"]
    included = np.flatnonzero(relative_steps <= int(relative_step))
    if len(included) == 0:
        return np.zeros((0, 3), dtype=np.float32), None
    last = int(included[-1])
    points = entry[stream][: last + 1]
    return points, entry[stream][last]


def _animate_cliff_reach_overlay(
    output_dir: Path,
    *,
    trial_metadata: list[dict[str, Any]],
    plant_coords: np.ndarray,
    drive_coords: np.ndarray,
    model: dict[str, Any],
    event_window: tuple[int, int],
    frame_stride: int,
    fps: int,
) -> dict[str, str]:
    entries = _animation_entries(
        trial_metadata=trial_metadata,
        plant_coords=plant_coords,
        drive_coords=drive_coords,
        event_window=event_window,
    )
    if not entries:
        return {}
    before, after = event_window
    frames = np.arange(-int(before), int(after) + 1, int(frame_stride), dtype=np.int32)
    if 0 not in frames:
        frames = np.sort(np.unique(np.concatenate([frames, np.asarray([0], dtype=np.int32)])))
    all_points = []
    for entry in entries:
        all_points.extend([entry["plant"], entry["drive"]])

    colors = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(8.8, 6.6), dpi=125)
    ax = fig.add_subplot(111, projection="3d")
    _set_equal_axes(ax, all_points)
    explained = model["explained_variance_ratio"]
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)", labelpad=8)
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)", labelpad=8)
    ax.set_zlabel(f"PC3 ({explained[2] * 100:.1f}%)", labelpad=8)
    ax.set_title("Event-Aligned Plant/Drive PCA Animation", pad=18)
    ax.view_init(elev=24, azim=-58)
    ax.grid(True, alpha=0.25)
    time_text = ax.text2D(0.02, 0.96, "", transform=ax.transAxes)
    artists: list[Any] = [time_text]
    entry_artists: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        color = colors(index % 10)
        meta = entry["meta"]
        label = f"{meta['distance_pair']} {meta['side']}"
        (plant_line,) = ax.plot([], [], [], color=color, linewidth=2.1, alpha=0.86, label=label)
        (drive_line,) = ax.plot([], [], [], color=color, linestyle="--", linewidth=1.65, alpha=0.78)
        (plant_current,) = ax.plot([], [], [], marker="o", color=color, markersize=5, linestyle="None")
        (drive_current,) = ax.plot([], [], [], marker="^", color=color, markersize=5, linestyle="None")
        event_index = int(entry["event_local_window_index"])
        event_plant = entry["plant"][event_index]
        event_drive = entry["drive"][event_index]
        (plant_event,) = ax.plot(
            [event_plant[0]],
            [event_plant[1]],
            [event_plant[2]],
            marker="o",
            color=color,
            markeredgecolor="black",
            markersize=8,
            linestyle="None",
            alpha=0.28,
        )
        (drive_event,) = ax.plot(
            [event_drive[0]],
            [event_drive[1]],
            [event_drive[2]],
            marker="X",
            color=color,
            markeredgecolor="black",
            markersize=8,
            linestyle="None",
            alpha=0.28,
        )
        entry_artists.append(
            {
                "entry": entry,
                "plant_line": plant_line,
                "drive_line": drive_line,
                "plant_current": plant_current,
                "drive_current": drive_current,
                "plant_event": plant_event,
                "drive_event": drive_event,
            }
        )
        artists.extend(
            [
                plant_line,
                drive_line,
                plant_current,
                drive_current,
                plant_event,
                drive_event,
            ]
        )

    stream_handles = [
        Line2D([0], [0], color="black", linewidth=2.0, label="Plant trail"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.6, label="Drive trail"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor="#777777", markeredgecolor="black", linestyle="None", markersize=7, label="Cliff reach"),
    ]
    legend1 = ax.legend(handles=stream_handles, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    ax.add_artist(legend1)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=8, title="Trials")
    fig.tight_layout()

    def update(frame_value: int):
        time_text.set_text(f"relative env step: {int(frame_value):+d}  |  t=0 first body edge reaches cliff")
        for item in entry_artists:
            entry = item["entry"]
            plant_path, plant_current = _points_through_relative_step(
                entry,
                stream="plant",
                relative_step=int(frame_value),
            )
            drive_path, drive_current = _points_through_relative_step(
                entry,
                stream="drive",
                relative_step=int(frame_value),
            )
            _line3d_set_path(item["plant_line"], plant_path)
            _line3d_set_path(item["drive_line"], drive_path)
            if plant_current is None:
                item["plant_current"].set_data([], [])
                item["plant_current"].set_3d_properties([])
                item["drive_current"].set_data([], [])
                item["drive_current"].set_3d_properties([])
            else:
                _line3d_set_point(item["plant_current"], plant_current)
                _line3d_set_point(item["drive_current"], drive_current)
            alpha = 1.0 if int(frame_value) >= 0 else 0.20
            item["plant_event"].set_alpha(alpha)
            item["drive_event"].set_alpha(alpha)
        return artists

    animation = FuncAnimation(
        fig,
        update,
        frames=[int(frame) for frame in frames],
        interval=1000 / max(int(fps), 1),
        blit=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / "plant_drive_shared_pca_3d_event_aligned.gif"
    mp4_path = output_dir / "plant_drive_shared_pca_3d_event_aligned.mp4"
    animation.save(gif_path, writer=PillowWriter(fps=int(fps)))
    animation.save(mp4_path, writer=FFMpegWriter(fps=int(fps), bitrate=1800))
    plt.close(fig)
    return {"event_aligned_gif": str(gif_path), "event_aligned_mp4": str(mp4_path)}


def _plot_overlay(
    output_path: Path,
    *,
    trial_metadata: list[dict[str, Any]],
    plant_coords: np.ndarray,
    drive_coords: np.ndarray,
    model: dict[str, Any],
    title: str,
    event_only: bool,
    max_points_per_trial: int,
    event_window: tuple[int, int] | None = None,
) -> None:
    colors = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(12.5, 9.5), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    offset = 0
    plotted_coords: list[np.ndarray] = []
    for index, meta in enumerate(trial_metadata):
        count = int(meta["step_count"])
        if event_only and meta["event_local_index"] is None:
            offset += count
            continue
        label = f"{meta['distance_pair']} {meta['side']}"
        color = colors(index % 10)
        plant = plant_coords[offset : offset + count]
        drive = drive_coords[offset : offset + count]
        if event_window is not None:
            if meta["event_local_index"] is None:
                offset += count
                continue
            before, after = event_window
            event_index = int(meta["event_local_index"])
            start = max(0, event_index - int(before))
            stop = min(count, event_index + int(after) + 1)
            base_indices = np.arange(start, stop, dtype=np.int64)
            keep_local = _downsample_indices(
                len(base_indices),
                max_points=max_points_per_trial,
                keep_index=event_index - start,
            )
            keep = base_indices[keep_local]
        else:
            keep = _downsample_indices(
                count,
            max_points=max_points_per_trial,
            keep_index=meta["event_local_index"],
            )
        plant_plot = plant[keep]
        drive_plot = drive[keep]
        plotted_coords.extend([plant_plot, drive_plot])
        ax.plot(
            plant_plot[:, 0],
            plant_plot[:, 1],
            plant_plot[:, 2],
            color=color,
            linewidth=1.8,
            alpha=0.84,
            label=label,
        )
        ax.plot(
            drive_plot[:, 0],
            drive_plot[:, 1],
            drive_plot[:, 2],
            color=color,
            linestyle="--",
            linewidth=1.35,
            alpha=0.74,
        )
        ax.scatter(
            plant[0, 0],
            plant[0, 1],
            plant[0, 2],
            color=color,
            marker="o",
            s=14,
            alpha=0.75,
        )
        ax.scatter(
            drive[0, 0],
            drive[0, 1],
            drive[0, 2],
            color=color,
            marker="^",
            s=14,
            alpha=0.75,
        )
        if meta["event_local_index"] is not None:
            event_index = int(meta["event_local_index"])
            ax.scatter(
                plant[event_index, 0],
                plant[event_index, 1],
                plant[event_index, 2],
                marker="o",
                s=72,
                color=color,
                edgecolor="black",
                linewidth=0.9,
                zorder=10,
            )
            ax.scatter(
                drive[event_index, 0],
                drive[event_index, 1],
                drive[event_index, 2],
                marker="X",
                s=76,
                color=color,
                edgecolor="black",
                linewidth=0.8,
                zorder=10,
            )
        offset += count

    if not plotted_coords:
        raise ValueError("No trajectories selected for plotting")
    _set_equal_axes(ax, plotted_coords)
    explained = model["explained_variance_ratio"]
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)", labelpad=12)
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)", labelpad=12)
    ax.set_zlabel(f"PC3 ({explained[2] * 100:.1f}%)", labelpad=12)
    ax.set_title(title, pad=22)
    ax.view_init(elev=24, azim=-58)
    ax.grid(True, alpha=0.25)
    stream_handles = [
        Line2D([0], [0], color="black", linewidth=2.0, label="Plant"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.6, label="Drive"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor="#888888",
            markeredgecolor="black",
            linestyle="None",
            markersize=7,
            label="Plant cliff reach",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="white",
            markerfacecolor="#888888",
            markeredgecolor="black",
            linestyle="None",
            markersize=7,
            label="Drive at same step",
        ),
    ]
    legend1 = ax.legend(handles=stream_handles, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    ax.add_artist(legend1)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=8, title="Trials")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def analyze_state_space_trace(
    trace_root: Path,
    output_dir: Path,
    *,
    max_points_per_trial: int,
    animate: bool,
    animation_stride: int,
    animation_fps: int,
) -> dict[str, Any]:
    trace_root = trace_root.resolve()
    output_dir = output_dir.resolve()
    manifest = _read_json(trace_root / "manifest.json")
    step_manifest = _load_step_manifest(trace_root)
    plant_blocks: list[np.ndarray] = []
    drive_blocks: list[np.ndarray] = []
    trial_metadata: list[dict[str, Any]] = []
    for trial_id, rows in step_manifest.groupby("trial_id", sort=False):
        plant, drive, metadata = _trial_features(
            trace_root=trace_root,
            trial_id=str(trial_id),
            rows=rows,
        )
        plant_blocks.append(plant)
        drive_blocks.append(drive)
        trial_metadata.append(metadata)
    plant_features = np.concatenate(plant_blocks, axis=0)
    drive_features = np.concatenate(drive_blocks, axis=0)
    plant_coords, drive_coords, combined_coords, model = _fit_shared_pca(
        plant_features,
        drive_features,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_overlay(
        output_dir / "plant_drive_shared_pca_3d_all_trials.png",
        trial_metadata=trial_metadata,
        plant_coords=plant_coords,
        drive_coords=drive_coords,
        model=model,
        title="Plant and Drive Trajectories in Shared 3D PCA Space",
        event_only=False,
        max_points_per_trial=max_points_per_trial,
    )
    _plot_overlay(
        output_dir / "plant_drive_shared_pca_3d_cliff_reach_trials.png",
        trial_metadata=trial_metadata,
        plant_coords=plant_coords,
        drive_coords=drive_coords,
        model=model,
        title="Cliff-Reach Trials: Plant and Drive in Shared 3D PCA Space",
        event_only=True,
        max_points_per_trial=max_points_per_trial,
    )
    _plot_overlay(
        output_dir / "plant_drive_shared_pca_3d_cliff_reach_window.png",
        trial_metadata=trial_metadata,
        plant_coords=plant_coords,
        drive_coords=drive_coords,
        model=model,
        title="Cliff-Reach Window: Plant and Drive in Shared 3D PCA Space",
        event_only=True,
        max_points_per_trial=max_points_per_trial,
        event_window=(180, 540),
    )
    _write_projected_csv(
        output_dir / "projected_trajectory_points.csv",
        step_manifest=step_manifest,
        trial_metadata=trial_metadata,
        plant_coords=plant_coords,
        drive_coords=drive_coords,
        combined_coords=combined_coords,
    )
    animation_outputs = {}
    if animate:
        animation_outputs = _animate_cliff_reach_overlay(
            output_dir,
            trial_metadata=trial_metadata,
            plant_coords=plant_coords,
            drive_coords=drive_coords,
            model=model,
            event_window=(180, 540),
            frame_stride=int(animation_stride),
            fps=int(animation_fps),
        )
    event_table = [
        {
            "trial_id": meta["trial_id"],
            "distance_pair": meta["distance_pair"],
            "side": meta["side"],
            "cliff_side": meta["cliff_side"],
            "step_count": meta["step_count"],
            "event_env_step": meta["event_env_step"],
            "event_local_index": meta["event_local_index"],
            "min_cliff_clearance": meta["min_cliff_clearance"],
        }
        for meta in trial_metadata
    ]
    summary = {
        "schema_version": 1,
        "kind": "state_space_plant_drive_shared_pca_overlay",
        "trace_root": str(trace_root),
        "output_dir": str(output_dir),
        "source_manifest": manifest,
        "pca": model,
        "trial_count": len(trial_metadata),
        "step_count": int(sum(meta["step_count"] for meta in trial_metadata)),
        "events": event_table,
        "outputs": {
            "all_trials_png": str(output_dir / "plant_drive_shared_pca_3d_all_trials.png"),
            "cliff_reach_trials_png": str(
                output_dir / "plant_drive_shared_pca_3d_cliff_reach_trials.png"
            ),
            "cliff_reach_window_png": str(
                output_dir / "plant_drive_shared_pca_3d_cliff_reach_window.png"
            ),
            "projected_trajectory_points": str(output_dir / "projected_trajectory_points.csv"),
            **animation_outputs,
            "summary": str(output_dir / "summary.json"),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate shared 3D PCA overlay figures for plant and drive state traces."
    )
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-points-per-trial", type=int, default=1400)
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--animation-stride", type=int, default=6)
    parser.add_argument("--animation-fps", type=int, default=12)
    args = parser.parse_args()

    summary = analyze_state_space_trace(
        args.trace_root,
        args.output_dir,
        max_points_per_trial=int(args.max_points_per_trial),
        animate=not bool(args.no_animation),
        animation_stride=int(args.animation_stride),
        animation_fps=int(args.animation_fps),
    )
    print(json.dumps(summary["outputs"], indent=2))


if __name__ == "__main__":
    main()
