from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from free_space_branch_common import (
    DEFAULT_BUNDLE_ROOT,
    default_analysis_dir,
    encode_frames,
    fit_pca,
    load_branch_trace,
    transform_pca,
    write_json,
)


def load_summary(trace_dir: Path) -> dict[str, object]:
    return json.loads((trace_dir / "summary.json").read_text())


def limits(xy_a: np.ndarray, xy_b: np.ndarray) -> tuple[float, float, float, float]:
    pooled = np.concatenate([xy_a, xy_b], axis=0)
    min_x = float(np.min(pooled[:, 0]))
    max_x = float(np.max(pooled[:, 0]))
    min_y = float(np.min(pooled[:, 1]))
    max_y = float(np.max(pooled[:, 1]))
    pad_x = max(0.08, 0.1 * max(max_x - min_x, 1.0))
    pad_y = max(0.08, 0.1 * max(max_y - min_y, 1.0))
    return (min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render separate plant/controller PCA movies from a saved free-space branched trace.",
    )
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <trace-dir>/state_pca.",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--flash-frames", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_dir = (
        args.trace_dir.resolve()
        if args.trace_dir is not None
        else default_analysis_dir(args.bundle_root.resolve(), max_steps=int(args.max_steps)).resolve()
    )
    if not (trace_dir / "branch_trace.npz").exists():
        raise FileNotFoundError(
            f"Missing branch trace under {trace_dir}. Run free_space_branch_capture.py first."
        )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (trace_dir / "state_pca").resolve()
    )
    if output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    trace = load_branch_trace(trace_dir)
    summary = load_summary(trace_dir)
    top_label = str(summary.get("row_label_top", "Baseline"))
    bottom_label = str(summary.get("row_label_bottom", "Perturbed"))
    event_label = str(summary.get("event_label", "PERTURBATION"))

    baseline_plant = np.asarray(trace["baseline_raw_abs_hv"], dtype=np.float32)
    perturbed_plant = np.asarray(trace["perturbed_raw_abs_hv"], dtype=np.float32)
    baseline_controller = np.asarray(trace["baseline_controller_flat"], dtype=np.float32)
    perturbed_controller = np.asarray(trace["perturbed_controller_flat"], dtype=np.float32)

    plant_pca = fit_pca(baseline_plant, perturbed_plant)
    ctrl_pca = fit_pca(baseline_controller, perturbed_controller)
    baseline_plant_xy = transform_pca(baseline_plant, plant_pca)
    perturbed_plant_xy = transform_pca(perturbed_plant, plant_pca)
    baseline_ctrl_xy = transform_pca(baseline_controller, ctrl_pca)
    perturbed_ctrl_xy = transform_pca(perturbed_controller, ctrl_pca)

    np.savez_compressed(
        output_dir / "state_pca_coords.npz",
        baseline_plant_xy=baseline_plant_xy,
        perturbed_plant_xy=perturbed_plant_xy,
        baseline_controller_xy=baseline_ctrl_xy,
        perturbed_controller_xy=perturbed_ctrl_xy,
        plant_mean=plant_pca["mean"],
        plant_std=plant_pca["std"],
        plant_components=plant_pca["components"],
        controller_mean=ctrl_pca["mean"],
        controller_std=ctrl_pca["std"],
        controller_components=ctrl_pca["components"],
    )

    phase_count = int(baseline_plant.shape[0])
    perturb_phase_index = int(trace["perturb_phase_index"][0])
    sense_act_phases = int(trace["sense_act_phases"][0])
    frame_phase_indices = np.linspace(0, phase_count - 1, num=min(int(args.max_frames), phase_count), dtype=np.int32)
    if perturb_phase_index not in frame_phase_indices:
        frame_phase_indices = np.unique(
            np.concatenate([frame_phase_indices, np.asarray([perturb_phase_index], dtype=np.int32)], axis=0)
        ).astype(np.int32, copy=False)
    frame_phase_indices.sort()
    perturb_frame_idx = int(np.searchsorted(frame_phase_indices, perturb_phase_index))

    plant_bounds = limits(baseline_plant_xy, perturbed_plant_xy)
    ctrl_bounds = limits(baseline_ctrl_xy, perturbed_ctrl_xy)

    fig, axes = plt.subplots(2, 1, figsize=(11.8, 11.0), dpi=110)
    fig.patch.set_facecolor("#fbfaf7")
    flash_overlay = Rectangle(
        (0.0, 0.0),
        1.0,
        1.0,
        transform=fig.transFigure,
        facecolor="#fff4bf",
        alpha=0.0,
        zorder=10,
    )
    fig.add_artist(flash_overlay)
    perturb_text = fig.text(
        0.5,
        0.982,
        event_label,
        ha="center",
        va="top",
        fontsize=24,
        fontweight="bold",
        color="#9f1239",
        alpha=0.0,
        zorder=11,
    )
    status_text = fig.text(
        0.5,
        0.948,
        "",
        ha="center",
        va="top",
        fontsize=11,
        color="#44403c",
        zorder=11,
    )

    titles = ["Plant PCA: raw absolute H/V state", "Controller PCA"]
    bounds = [plant_bounds, ctrl_bounds]
    sequences = [
        (baseline_plant_xy, perturbed_plant_xy),
        (baseline_ctrl_xy, perturbed_ctrl_xy),
    ]
    colors = {"top": "#2563eb", "bottom": "#d97706"}

    lines = []
    dots = []
    for idx, ax in enumerate(axes):
        ax.set_facecolor("#fffdf8")
        for spine in ax.spines.values():
            spine.set_color("#d6d3d1")
        ax.set_title(titles[idx], fontsize=14, fontweight="bold")
        ax.set_xlim(bounds[idx][0], bounds[idx][1])
        ax.set_ylim(bounds[idx][2], bounds[idx][3])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.plot(
            sequences[idx][0][:, 0],
            sequences[idx][0][:, 1],
            color=colors["top"],
            alpha=0.15,
            linewidth=1.3,
        )
        ax.plot(
            sequences[idx][1][:, 0],
            sequences[idx][1][:, 1],
            color=colors["bottom"],
            alpha=0.15,
            linewidth=1.3,
        )
        baseline_line, = ax.plot(
            [],
            [],
            color=colors["top"],
            linewidth=2.1,
            linestyle=(0, (5, 2)),
            label=top_label,
            zorder=6,
        )
        perturbed_line, = ax.plot(
            [],
            [],
            color=colors["bottom"],
            linewidth=2.3,
            label=bottom_label,
            zorder=5,
        )
        baseline_dot = ax.scatter(
            [],
            [],
            s=64,
            facecolors="white",
            edgecolors=colors["top"],
            linewidths=1.8,
            marker="o",
            zorder=8,
        )
        perturbed_dot = ax.scatter(
            [],
            [],
            s=64,
            color=colors["bottom"],
            linewidths=1.8,
            marker="X",
            zorder=7,
        )
        ax.legend(loc="upper right", frameon=False)
        lines.append((baseline_line, perturbed_line))
        dots.append((baseline_dot, perturbed_dot))

    frames_dir = output_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, phase_idx in enumerate(frame_phase_indices):
        for seq_idx, (baseline_xy, perturbed_xy) in enumerate(sequences):
            baseline_line, perturbed_line = lines[seq_idx]
            baseline_dot, perturbed_dot = dots[seq_idx]
            baseline_line.set_data(
                baseline_xy[: phase_idx + 1, 0],
                baseline_xy[: phase_idx + 1, 1],
            )
            perturbed_line.set_data(
                perturbed_xy[: phase_idx + 1, 0],
                perturbed_xy[: phase_idx + 1, 1],
            )
            baseline_dot.set_offsets(baseline_xy[phase_idx : phase_idx + 1])
            perturbed_dot.set_offsets(perturbed_xy[phase_idx : phase_idx + 1])

        is_flash = bool(perturb_frame_idx <= int(frame_idx) < (perturb_frame_idx + int(args.flash_frames)))
        flash_overlay.set_alpha(0.24 if is_flash else 0.0)
        perturb_text.set_alpha(1.0 if is_flash else 0.0)
        env_step = int(phase_idx) // max(1, sense_act_phases)
        status_text.set_text(f"Phase {int(phase_idx) + 1} / {phase_count}   Env step {env_step + 1}")

        fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.92])
        fig.savefig(frames_dir / f"frame_{frame_idx:04d}.png", dpi=110)

    plt.close(fig)

    mp4_path = output_dir / "state_pca.mp4"
    gif_path = output_dir / "state_pca.gif"
    encode_frames(frames_dir, fps=int(args.fps), mp4_path=mp4_path, gif_path=gif_path)

    payload = {
        "trace_dir": str(trace_dir),
        "mp4_path": str(mp4_path),
        "gif_path": str(gif_path),
        "coords_path": str(output_dir / "state_pca_coords.npz"),
        "frame_count": int(frame_phase_indices.shape[0]),
        "fps": int(args.fps),
        "flash_frames": int(args.flash_frames),
        "phase_count": int(phase_count),
        "perturb_phase_index": int(perturb_phase_index),
    }
    write_json(output_dir / "summary.json", payload)
    print(f"mp4_path: {mp4_path}")
    print(f"gif_path: {gif_path}")


if __name__ == "__main__":
    main()
