from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import pandas as pd
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
S4_ROOT = SCRIPT_DIR.parent
DEFAULT_BUNDLE_ROOT = (
    S4_ROOT
    / "results"
    / "bin_03_rank_00_cell_6-0-3_flat_0423"
    / "source_runs"
    / "current_core_full"
    / "pca_bundle"
)
ABS_CHANNEL_LABELS = ("abs_h_delta", "abs_v_delta")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_states(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as data:
        key = data.files[0]
        return np.asarray(data[key], dtype=np.float32)


def _fit_pca(x: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0, dtype=np.float64)
    centered = x.astype(np.float64) - mean
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:n_components].astype(np.float32)
    total_var = np.sum(singular_values ** 2)
    explained = (
        (singular_values[:n_components] ** 2) / total_var
        if total_var > 0
        else np.zeros((n_components,), dtype=np.float64)
    )
    return mean.astype(np.float32), components, explained.astype(np.float32)


def _project(x: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    centered = x.astype(np.float32) - mean[None, :]
    return centered @ components.T


def _phase_unique_mask(labels: pd.DataFrame) -> np.ndarray:
    return np.asarray(labels["micro_index"].to_numpy() == 0, dtype=bool)


def _sample_rows(x: np.ndarray, max_points: int, *, seed: int = 0) -> np.ndarray:
    if x.shape[0] <= max_points:
        return x
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(x.shape[0], size=max_points, replace=False))
    return x[indices]


def _scatter_condition_plot(
    *,
    intact_proj: np.ndarray,
    ablated_proj: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    intact_plot = _sample_rows(intact_proj[:, :2], 25000, seed=1)
    ablated_plot = _sample_rows(ablated_proj[:, :2], 25000, seed=2)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.scatter(intact_plot[:, 0], intact_plot[:, 1], s=6, alpha=0.18, label="intact", color="#1f77b4", linewidths=0)
    ax.scatter(ablated_plot[:, 0], ablated_plot[:, 1], s=6, alpha=0.30, label="deformation_off", color="#d95f02", linewidths=0)
    ax.set_title("Absolute Deformation PCA: Intact vs Deformation Off")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _scatter_side_plot(
    *,
    intact_proj: np.ndarray,
    intact_labels: pd.DataFrame,
    ablated_proj: np.ndarray,
    ablated_labels: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharex=True, sharey=True)
    for axis, side in zip(axes, ("left", "right"), strict=True):
        intact_side = intact_proj[intact_labels["side"].to_numpy() == side, :2]
        ablated_side = ablated_proj[ablated_labels["side"].to_numpy() == side, :2]
        intact_plot = _sample_rows(intact_side, 15000, seed=3 if side == "left" else 4)
        ablated_plot = _sample_rows(ablated_side, 15000, seed=5 if side == "left" else 6)
        axis.scatter(intact_plot[:, 0], intact_plot[:, 1], s=6, alpha=0.18, label="intact", color="#1f77b4", linewidths=0)
        axis.scatter(ablated_plot[:, 0], ablated_plot[:, 1], s=6, alpha=0.30, label="deformation_off", color="#d95f02", linewidths=0)
        axis.set_title(f"{side.capitalize()} trials")
        axis.set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[1].legend(frameon=False)
    fig.suptitle("Absolute Deformation PCA by Side")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _selected_trial_plot(
    *,
    intact_proj: np.ndarray,
    intact_labels: pd.DataFrame,
    intact_trial_id: str,
    ablated_proj: np.ndarray,
    ablated_labels: pd.DataFrame,
    ablated_trial_id: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    intact_mask = np.asarray(intact_labels["trial_id"].to_numpy() == intact_trial_id, dtype=bool)
    ablated_mask = np.asarray(ablated_labels["trial_id"].to_numpy() == ablated_trial_id, dtype=bool)
    intact_xy = intact_proj[intact_mask, :2]
    ablated_xy = ablated_proj[ablated_mask, :2]
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.plot(intact_xy[:, 0], intact_xy[:, 1], color="#1f77b4", linewidth=2.2, label=f"intact {intact_trial_id}")
    ax.plot(ablated_xy[:, 0], ablated_xy[:, 1], color="#d95f02", linewidth=2.2, linestyle="--", label=f"deformation_off {ablated_trial_id}")
    ax.scatter([intact_xy[0, 0], ablated_xy[0, 0]], [intact_xy[0, 1], ablated_xy[0, 1]], color=["#1f77b4", "#d95f02"], s=24, zorder=3)
    ax.set_title("Selected-Trial Absolute Deformation PCA Trajectories")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _goalward_colors(goalward_x: np.ndarray) -> np.ndarray:
    goalward_x = np.asarray(goalward_x, dtype=np.float32)
    colors = np.zeros((goalward_x.shape[0], 4), dtype=np.float32)
    pos_max = float(np.max(goalward_x[goalward_x > 0.0])) if np.any(goalward_x > 0.0) else 1.0
    neg_min = float(np.min(goalward_x[goalward_x < 0.0])) if np.any(goalward_x < 0.0) else -1.0
    for idx, value in enumerate(goalward_x):
        if value >= 0.0:
            alpha = float(value / pos_max) if pos_max > 0 else 0.0
            color = np.array([1.0, alpha, 0.0, 1.0], dtype=np.float32)  # red -> yellow
        else:
            alpha = float(value / neg_min) if neg_min < 0 else 0.0
            cool = np.array([0.20, 0.55, 0.95, 1.0], dtype=np.float32)
            warm = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
            color = cool * (1.0 - alpha) + warm * alpha
        colors[idx] = color
    return colors


def _add_gradient_line(ax: plt.Axes, xy: np.ndarray, goalward_x: np.ndarray, *, title: str) -> None:
    if xy.shape[0] < 2:
        ax.plot(xy[:, 0], xy[:, 1], color="black", linewidth=1.5)
        ax.set_title(title)
        return
    colors = _goalward_colors(goalward_x)
    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    segment_colors = 0.5 * (colors[:-1] + colors[1:])
    collection = LineCollection(segments, colors=segment_colors, linewidths=2.2)
    ax.add_collection(collection)
    ax.scatter([xy[0, 0]], [xy[0, 1]], color="black", s=22, zorder=3)
    ax.scatter([xy[-1, 0]], [xy[-1, 1]], color="#ffd92f", edgecolor="black", linewidth=0.4, s=28, zorder=3)
    ax.autoscale_view()
    ax.set_title(title)


def _selected_trial_location_plot(
    *,
    intact_proj: np.ndarray,
    intact_labels: pd.DataFrame,
    intact_trial_id: str,
    ablated_proj: np.ndarray,
    ablated_labels: pd.DataFrame,
    ablated_trial_id: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    intact_mask = np.asarray(intact_labels["trial_id"].to_numpy() == intact_trial_id, dtype=bool)
    ablated_mask = np.asarray(ablated_labels["trial_id"].to_numpy() == ablated_trial_id, dtype=bool)
    intact_xy = intact_proj[intact_mask, :2]
    ablated_xy = ablated_proj[ablated_mask, :2]
    intact_rows = intact_labels.loc[intact_mask].reset_index(drop=True)
    ablated_rows = ablated_labels.loc[ablated_mask].reset_index(drop=True)
    intact_goalward = (
        intact_rows["goal_direction"].astype(float).to_numpy()
        * (intact_rows["com_x"].astype(float).to_numpy() - float(intact_rows["com_x"].iloc[0]))
    )
    ablated_goalward = (
        ablated_rows["goal_direction"].astype(float).to_numpy()
        * (ablated_rows["com_x"].astype(float).to_numpy() - float(ablated_rows["com_x"].iloc[0]))
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), sharex=True, sharey=True)
    _add_gradient_line(
        axes[0],
        intact_xy,
        intact_goalward,
        title=f"intact {intact_trial_id}",
    )
    _add_gradient_line(
        axes[1],
        ablated_xy,
        ablated_goalward,
        title=f"deformation_off {ablated_trial_id}",
    )
    axes[0].set_ylabel("PC2")
    for axis in axes:
        axis.set_xlabel("PC1")
    fig.suptitle("Selected-trial absolute deformation PCA, colored by location")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _set_3d_limits(axes: list, xyz_blocks: list[np.ndarray]) -> None:
    all_xyz = np.concatenate(xyz_blocks, axis=0)
    mins = np.min(all_xyz, axis=0)
    maxs = np.max(all_xyz, axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    if radius <= 0:
        radius = 1.0
    for ax in axes:
        ax.set_xlim(centers[0] - radius, centers[0] + radius)
        ax.set_ylim(centers[1] - radius, centers[1] + radius)
        ax.set_zlim(centers[2] - radius, centers[2] + radius)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")


def _add_gradient_line_3d(ax, xyz: np.ndarray, goalward_x: np.ndarray, *, title: str) -> None:
    if xyz.shape[0] < 2:
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="black", linewidth=1.5)
        ax.set_title(title)
        return
    colors = _goalward_colors(goalward_x)
    segments = np.stack([xyz[:-1], xyz[1:]], axis=1)
    segment_colors = 0.5 * (colors[:-1] + colors[1:])
    collection = Line3DCollection(segments, colors=segment_colors, linewidths=2.2)
    ax.add_collection3d(collection)
    ax.scatter([xyz[0, 0]], [xyz[0, 1]], [xyz[0, 2]], color="black", s=18, zorder=3)
    ax.scatter([xyz[-1, 0]], [xyz[-1, 1]], [xyz[-1, 2]], color="#ffd92f", edgecolor="black", linewidth=0.4, s=24, zorder=3)
    ax.set_title(title)


def _add_gradient_backdrop_3d(ax, xyz: np.ndarray, goalward_x: np.ndarray) -> None:
    if xyz.shape[0] < 2:
        ax.scatter([xyz[0, 0]], [xyz[0, 1]], [xyz[0, 2]], color="#888888", s=14, alpha=0.25)
        return
    colors = _goalward_colors(goalward_x)
    colors[:, 3] = 0.12
    segments = np.stack([xyz[:-1], xyz[1:]], axis=1)
    segment_colors = 0.5 * (colors[:-1] + colors[1:])
    collection = Line3DCollection(segments, colors=segment_colors, linewidths=1.2)
    ax.add_collection3d(collection)


def _add_gradient_line_3d_progress(
    ax,
    xyz: np.ndarray,
    goalward_x: np.ndarray,
    *,
    upto: int,
    title: str,
) -> None:
    capped = max(1, min(int(upto), int(xyz.shape[0])))
    if capped <= 1 or xyz.shape[0] < 2:
        ax.scatter([xyz[0, 0]], [xyz[0, 1]], [xyz[0, 2]], color="black", s=18, zorder=3)
        ax.set_title(title)
        return
    colors = _goalward_colors(goalward_x[:capped])
    segments = np.stack([xyz[: capped - 1], xyz[1:capped]], axis=1)
    segment_colors = 0.5 * (colors[:-1] + colors[1:])
    collection = Line3DCollection(segments, colors=segment_colors, linewidths=2.4)
    ax.add_collection3d(collection)
    ax.scatter([xyz[0, 0]], [xyz[0, 1]], [xyz[0, 2]], color="black", s=18, zorder=3)
    ax.scatter([xyz[capped - 1, 0]], [xyz[capped - 1, 1]], [xyz[capped - 1, 2]], color=colors[capped - 1], s=24, zorder=4)
    ax.set_title(title)


def _selected_trial_location_3d_outputs(
    *,
    intact_proj: np.ndarray,
    intact_labels: pd.DataFrame,
    intact_trial_id: str,
    ablated_proj: np.ndarray,
    ablated_labels: pd.DataFrame,
    ablated_trial_id: str,
    figures_dir: Path,
) -> None:
    intact_mask = np.asarray(intact_labels["trial_id"].to_numpy() == intact_trial_id, dtype=bool)
    ablated_mask = np.asarray(ablated_labels["trial_id"].to_numpy() == ablated_trial_id, dtype=bool)
    intact_xyz = intact_proj[intact_mask, :3]
    ablated_xyz = ablated_proj[ablated_mask, :3]
    intact_rows = intact_labels.loc[intact_mask].reset_index(drop=True)
    ablated_rows = ablated_labels.loc[ablated_mask].reset_index(drop=True)
    intact_goalward = (
        intact_rows["goal_direction"].astype(float).to_numpy()
        * (intact_rows["com_x"].astype(float).to_numpy() - float(intact_rows["com_x"].iloc[0]))
    )
    ablated_goalward = (
        ablated_rows["goal_direction"].astype(float).to_numpy()
        * (ablated_rows["com_x"].astype(float).to_numpy() - float(ablated_rows["com_x"].iloc[0]))
    )

    single_path = figures_dir / "selected_trial_location_gradient_3d.png"
    fig = plt.figure(figsize=(12.0, 5.4))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    _add_gradient_line_3d(ax1, intact_xyz, intact_goalward, title=f"intact {intact_trial_id}")
    _add_gradient_line_3d(ax2, ablated_xyz, ablated_goalward, title=f"deformation_off {ablated_trial_id}")
    _set_3d_limits([ax1, ax2], [intact_xyz, ablated_xyz])
    ax1.view_init(elev=24, azim=-58)
    ax2.view_init(elev=24, azim=-58)
    fig.suptitle("Selected-trial absolute deformation PCA, colored by location")
    fig.tight_layout()
    fig.savefig(single_path, dpi=180)
    plt.close(fig)

    multiview_path = figures_dir / "selected_trial_location_gradient_3d_multiview.png"
    views = ((24, -58), (24, 18), (24, 102))
    fig = plt.figure(figsize=(15.0, 8.8))
    axes: list = []
    for col, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 3, col + 1, projection="3d")
        _add_gradient_line_3d(ax, intact_xyz, intact_goalward, title=f"intact view {col + 1}")
        ax.view_init(elev=elev, azim=azim)
        axes.append(ax)
        ax = fig.add_subplot(2, 3, col + 4, projection="3d")
        _add_gradient_line_3d(ax, ablated_xyz, ablated_goalward, title=f"deformation_off view {col + 1}")
        ax.view_init(elev=elev, azim=azim)
        axes.append(ax)
    _set_3d_limits(axes, [intact_xyz, ablated_xyz])
    fig.suptitle("Selected-trial absolute deformation PCA, multiple 3D views")
    fig.tight_layout()
    fig.savefig(multiview_path, dpi=180)
    plt.close(fig)

    gif_path = figures_dir / "selected_trial_location_gradient_3d.gif"
    frames: list[Image.Image] = []
    max_len = max(intact_xyz.shape[0], ablated_xyz.shape[0])
    frame_counts = np.unique(np.linspace(1, max_len, num=min(72, max_len), endpoint=True).astype(int))
    for count in frame_counts:
        fig = plt.figure(figsize=(12.0, 5.4))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        _add_gradient_backdrop_3d(ax1, intact_xyz, intact_goalward)
        _add_gradient_backdrop_3d(ax2, ablated_xyz, ablated_goalward)
        _add_gradient_line_3d_progress(ax1, intact_xyz, intact_goalward, upto=int(count), title="intact")
        _add_gradient_line_3d_progress(ax2, ablated_xyz, ablated_goalward, upto=int(count), title="deformation_off")
        _set_3d_limits([ax1, ax2], [intact_xyz, ablated_xyz])
        ax1.view_init(elev=24, azim=-58)
        ax2.view_init(elev=24, azim=-58)
        fig.suptitle("Selected-trial absolute deformation PCA, colored by location")
        fig.tight_layout()
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=140)
        plt.close(fig)
        buffer.seek(0)
        frames.append(Image.open(buffer).convert("P", palette=Image.Palette.ADAPTIVE))
    if frames:
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=80,
            loop=0,
            optimize=False,
            disposal=2,
        )


def _save_loading_maps(
    *,
    components: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = components.shape[0]
    fig, axes = plt.subplots(n_rows, 2, figsize=(10.5, 3.3 * n_rows))
    axes = np.atleast_2d(axes)
    for row in range(n_rows):
        component = components[row].reshape(2, 7, 2)
        cell_mag = np.linalg.norm(component, axis=-1)
        channel_mag = np.linalg.norm(component.reshape(-1, 2), axis=0)
        vmax = float(np.max(np.abs(cell_mag))) if cell_mag.size else 1.0
        image = axes[row, 0].imshow(cell_mag, cmap="viridis", vmin=0.0, vmax=vmax if vmax > 0 else 1.0)
        axes[row, 0].set_title(f"PC{row + 1} cell magnitude")
        axes[row, 0].set_xlabel("x")
        axes[row, 0].set_ylabel("y")
        plt.colorbar(image, ax=axes[row, 0], fraction=0.046, pad=0.04)
        axes[row, 1].bar(range(len(ABS_CHANNEL_LABELS)), channel_mag, color=["#4daf4a", "#984ea3"])
        axes[row, 1].set_xticks(range(len(ABS_CHANNEL_LABELS)))
        axes[row, 1].set_xticklabels(ABS_CHANNEL_LABELS, rotation=15)
        axes[row, 1].set_ylabel("L2 loading")
        axes[row, 1].set_title(f"PC{row + 1} channel contribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_deformation_absolute_pca(
    *,
    bundle_dir: Path,
    output_dir: Path,
    n_components: int = 3,
) -> dict[str, object]:
    intact_dir = bundle_dir
    ablated_dir = bundle_dir / "cohorts" / "deformation_off_native_core"

    intact_summary = _load_json(intact_dir / "cohort_summary.json")
    ablated_summary = _load_json(ablated_dir / "cohort_summary.json")
    intact_labels_all = pd.read_csv(intact_dir / "trace_labels.csv")
    ablated_labels_all = pd.read_csv(ablated_dir / "trace_labels.csv")
    intact_states_all = _load_states(intact_dir / "nca_states.npz")[..., 0:2]
    ablated_states_all = _load_states(ablated_dir / "nca_states.npz")[..., 0:2]

    intact_mask = _phase_unique_mask(intact_labels_all)
    ablated_mask = _phase_unique_mask(ablated_labels_all)
    intact_labels = intact_labels_all.loc[intact_mask].reset_index(drop=True)
    ablated_labels = ablated_labels_all.loc[ablated_mask].reset_index(drop=True)
    intact_x = intact_states_all[intact_mask].reshape(int(np.sum(intact_mask)), -1)
    ablated_x = ablated_states_all[ablated_mask].reshape(int(np.sum(ablated_mask)), -1)

    mean, components, explained = _fit_pca(intact_x, n_components=n_components)
    intact_proj = _project(intact_x, mean, components)
    ablated_proj = _project(ablated_x, mean, components)

    figures_dir = output_dir / "figures"
    _scatter_condition_plot(
        intact_proj=intact_proj,
        ablated_proj=ablated_proj,
        output_path=figures_dir / "pc1_pc2_by_condition.png",
    )
    _scatter_side_plot(
        intact_proj=intact_proj,
        intact_labels=intact_labels,
        ablated_proj=ablated_proj,
        ablated_labels=ablated_labels,
        output_path=figures_dir / "pc1_pc2_by_side.png",
    )
    _selected_trial_plot(
        intact_proj=intact_proj,
        intact_labels=intact_labels,
        intact_trial_id=str(intact_summary["selected_trial_ids"][0]),
        ablated_proj=ablated_proj,
        ablated_labels=ablated_labels,
        ablated_trial_id=str(ablated_summary["selected_trial_ids"][0]),
        output_path=figures_dir / "selected_trial_trajectories.png",
    )
    _selected_trial_location_plot(
        intact_proj=intact_proj,
        intact_labels=intact_labels,
        intact_trial_id=str(intact_summary["selected_trial_ids"][0]),
        ablated_proj=ablated_proj,
        ablated_labels=ablated_labels,
        ablated_trial_id=str(ablated_summary["selected_trial_ids"][0]),
        output_path=figures_dir / "selected_trial_location_gradient.png",
    )
    _selected_trial_location_3d_outputs(
        intact_proj=intact_proj,
        intact_labels=intact_labels,
        intact_trial_id=str(intact_summary["selected_trial_ids"][0]),
        ablated_proj=ablated_proj,
        ablated_labels=ablated_labels,
        ablated_trial_id=str(ablated_summary["selected_trial_ids"][0]),
        figures_dir=figures_dir,
    )
    _save_loading_maps(
        components=components[: min(3, components.shape[0])],
        output_path=figures_dir / "pc_loading_maps.png",
    )

    np.savez_compressed(
        output_dir / "deformation_absolute_pca.npz",
        mean=mean,
        components=components,
        explained_variance_ratio=explained,
        intact_projection=intact_proj,
        deformation_off_projection=ablated_proj,
    )

    intact_centroid = np.mean(intact_proj[:, :2], axis=0, dtype=np.float64)
    ablated_centroid = np.mean(ablated_proj[:, :2], axis=0, dtype=np.float64)
    summary = {
        "bundle_dir": str(bundle_dir),
        "feature_definition": {
            "channels": list(ABS_CHANNEL_LABELS),
            "description": "Signed absolute deformation channels only: horizontal and vertical deformation minus 1.0, excluding compliance channels.",
            "feature_dim": int(intact_x.shape[1]),
        },
        "phase_unique_only": True,
        "fit_condition": "intact",
        "intact_phase_unique_samples": int(intact_x.shape[0]),
        "deformation_off_phase_unique_samples": int(ablated_x.shape[0]),
        "intact_trial_count": int(intact_labels["trial_id"].nunique()),
        "deformation_off_trial_count": int(ablated_labels["trial_id"].nunique()),
        "intact_selected_trial_id": str(intact_summary["selected_trial_ids"][0]),
        "deformation_off_selected_trial_id": str(ablated_summary["selected_trial_ids"][0]),
        "deformation_off_all_zero": bool(np.allclose(ablated_x, 0.0)),
        "explained_variance_ratio": [float(x) for x in explained],
        "pc1_pc2_centroid_distance": float(np.linalg.norm(intact_centroid - ablated_centroid)),
        "intact_pc1_pc2_span": {
            "pc1": [float(np.min(intact_proj[:, 0])), float(np.max(intact_proj[:, 0]))],
            "pc2": [float(np.min(intact_proj[:, 1])), float(np.max(intact_proj[:, 1]))],
        },
        "deformation_off_pc1_pc2_span": {
            "pc1": [float(np.min(ablated_proj[:, 0])), float(np.max(ablated_proj[:, 0]))],
            "pc2": [float(np.min(ablated_proj[:, 1])), float(np.max(ablated_proj[:, 1]))],
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PCA on the absolute deformation channels across the intact PCA bundle and the deformation-off cohort."
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT,
        help="Path to the copied pca_bundle directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for summaries and figures.",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=3,
        help="Number of PCA components to retain.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    bundle_dir = args.bundle_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (bundle_dir.parents[2] / "analysis_runs" / "deformation_absolute_pca").resolve()
    )
    summary = run_deformation_absolute_pca(
        bundle_dir=bundle_dir,
        output_dir=output_dir,
        n_components=int(args.components),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
