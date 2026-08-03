from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from free_space_branch_common import DEFAULT_BUNDLE_ROOT, encode_frames, fit_pca, transform_pca, write_json


SCRIPT_DIR = Path(__file__).resolve().parent
S4_ROOT = SCRIPT_DIR.parent
DEFAULT_BRANCH_STATE_PCA_DIR = (
    DEFAULT_BUNDLE_ROOT
    / "analysis_runs"
    / "flat_ground_hold_compare_steps_300_hold20_clear005_paperfriction"
    / "state_pca"
)
DEFAULT_MATCHED_BUNDLE_DIR = (
    DEFAULT_BUNDLE_ROOT
    / "source_runs"
    / "current_core_full"
    / "_previous"
    / "2026-04-13T06-27-50.032355_00-00"
    / "pca_bundle"
)
DEFAULT_MATCHED_OUTPUT_DIR = (
    DEFAULT_BUNDLE_ROOT / "analysis_runs" / "matched_pair_4_2_state_pca_spectral_purity"
)
DEFAULT_BRANCH_OUTPUT_DIR = (
    DEFAULT_BRANCH_STATE_PCA_DIR.parent / "state_pca_spectral_purity"
)
INTACT_TRIAL_ID = "pair_4_2_left_000"
DEFORMATION_OFF_TRIAL_ID = "pair_4_2_right_000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze spectral purity on current PCA trajectories and build the matched pair PCA movie from pca_bundle.",
    )
    parser.add_argument(
        "--branch-state-pca-dir",
        type=Path,
        default=DEFAULT_BRANCH_STATE_PCA_DIR,
        help="Directory containing an existing state_pca_coords.npz and summary.json.",
    )
    parser.add_argument(
        "--branch-output-dir",
        type=Path,
        default=DEFAULT_BRANCH_OUTPUT_DIR,
        help="Where to save spectral summaries for the current branch PCA.",
    )
    parser.add_argument(
        "--matched-bundle-dir",
        type=Path,
        default=DEFAULT_MATCHED_BUNDLE_DIR,
        help="Archived pca_bundle directory with intact and deformation_off cohorts.",
    )
    parser.add_argument(
        "--matched-output-dir",
        type=Path,
        default=DEFAULT_MATCHED_OUTPUT_DIR,
        help="Where to save the matched pair PCA movie and purity summaries.",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--top-k-peaks", type=int, default=5)
    parser.add_argument(
        "--skip-branch",
        action="store_true",
        help="Skip the current branch PCA spectral analysis.",
    )
    parser.add_argument(
        "--skip-matched",
        action="store_true",
        help="Skip the matched intact vs deformation_off reconstruction.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _windowed_xy_spectrum(
    xy: np.ndarray,
    *,
    sample_spacing_env_steps: float,
    top_k_peaks: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    xy = np.asarray(xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] < 1:
        raise ValueError(f"Expected 2D trajectory, got shape={xy.shape}")
    centered = xy - np.mean(xy, axis=0, keepdims=True, dtype=np.float32)
    if centered.shape[0] < 4:
        freqs = np.zeros((0,), dtype=np.float32)
        power = np.zeros((0,), dtype=np.float32)
        metrics = {
            "sample_count": int(centered.shape[0]),
            "dominant_frequency_cycles_per_env_step": None,
            "dominant_period_env_steps": None,
            "dominant_power_fraction": 0.0,
            "top3_power_fraction": 0.0,
            "peak_to_second_ratio": None,
            "spectral_entropy": 0.0,
            "peaks": [],
        }
        return freqs, power, metrics

    window = np.hanning(centered.shape[0]).astype(np.float32)[:, None]
    windowed = centered * window
    spectrum = np.fft.rfft(windowed, axis=0)
    power = np.sum(np.abs(spectrum) ** 2, axis=1).astype(np.float64, copy=False)
    freqs = np.fft.rfftfreq(centered.shape[0], d=float(sample_spacing_env_steps)).astype(
        np.float32, copy=False
    )
    if power.size:
        power[0] = 0.0

    total_power = float(np.sum(power))
    if not np.isfinite(total_power) or total_power <= 0.0:
        metrics = {
            "sample_count": int(centered.shape[0]),
            "dominant_frequency_cycles_per_env_step": None,
            "dominant_period_env_steps": None,
            "dominant_power_fraction": 0.0,
            "top3_power_fraction": 0.0,
            "peak_to_second_ratio": None,
            "spectral_entropy": 0.0,
            "peaks": [],
        }
        return freqs, power.astype(np.float32, copy=False), metrics

    ranked = np.argsort(power)[::-1]
    peak_index = int(ranked[0])
    second_index = int(ranked[1]) if ranked.shape[0] > 1 else peak_index
    normalized = power / total_power
    nz = normalized[normalized > 0.0]
    entropy = 0.0
    if nz.size > 1:
        entropy = float(-(nz * np.log(nz)).sum() / np.log(float(normalized.size)))

    peaks = []
    for index in ranked[: int(top_k_peaks)]:
        freq = float(freqs[int(index)])
        peaks.append(
            {
                "frequency_cycles_per_env_step": freq,
                "period_env_steps": (1.0 / freq) if freq > 0.0 else None,
                "power_fraction": float(power[int(index)] / total_power),
            }
        )

    peak_frequency = float(freqs[peak_index])
    second_power = float(power[second_index])
    peak_to_second = (
        float(power[peak_index] / second_power)
        if second_power > 0.0 and second_index != peak_index
        else None
    )
    metrics = {
        "sample_count": int(centered.shape[0]),
        "dominant_frequency_cycles_per_env_step": peak_frequency,
        "dominant_period_env_steps": (1.0 / peak_frequency) if peak_frequency > 0.0 else None,
        "dominant_power_fraction": float(power[peak_index] / total_power),
        "top3_power_fraction": float(np.sum(power[ranked[:3]]) / total_power),
        "peak_to_second_ratio": peak_to_second,
        "spectral_entropy": entropy,
        "peaks": peaks,
    }
    return freqs, power.astype(np.float32, copy=False), metrics


def _save_overlay_spectrum(
    *,
    freqs_a: np.ndarray,
    power_a: np.ndarray,
    label_a: str,
    color_a: str,
    freqs_b: np.ndarray,
    power_b: np.ndarray,
    label_b: str,
    color_b: str,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(freqs_a, np.sqrt(np.maximum(power_a, 0.0)), color=color_a, linewidth=2.0, label=label_a)
    ax.plot(freqs_b, np.sqrt(np.maximum(power_b, 0.0)), color=color_b, linewidth=2.0, label=label_b)
    ax.set_title(title)
    ax.set_xlabel("cycles / env step")
    ax.set_ylabel("sqrt(power)")
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _limits(xy_a: np.ndarray, xy_b: np.ndarray) -> tuple[float, float, float, float]:
    pooled = np.concatenate([xy_a, xy_b], axis=0)
    min_x = float(np.min(pooled[:, 0]))
    max_x = float(np.max(pooled[:, 0]))
    min_y = float(np.min(pooled[:, 1]))
    max_y = float(np.max(pooled[:, 1]))
    pad_x = max(0.08, 0.1 * max(max_x - min_x, 1.0))
    pad_y = max(0.08, 0.1 * max(max_y - min_y, 1.0))
    return (min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y)


def _render_state_pca_movie(
    *,
    top_a_xy: np.ndarray,
    top_b_xy: np.ndarray,
    bottom_a_xy: np.ndarray,
    bottom_b_xy: np.ndarray,
    label_a: str,
    label_b: str,
    top_title: str,
    bottom_title: str,
    env_steps_a: np.ndarray,
    env_steps_b: np.ndarray,
    phase_count: int,
    output_dir: Path,
    fps: int,
    max_frames: int,
    event_label: str | None = None,
    event_frame_idx: int | None = None,
    flash_frames: int = 8,
) -> dict[str, Any]:
    frame_indices = np.linspace(
        0,
        phase_count - 1,
        num=min(int(max_frames), int(phase_count)),
        dtype=np.int32,
    )
    if event_frame_idx is not None and int(event_frame_idx) not in frame_indices:
        frame_indices = np.unique(
            np.concatenate([frame_indices, np.asarray([event_frame_idx], dtype=np.int32)], axis=0)
        ).astype(np.int32, copy=False)
        frame_indices.sort()

    top_bounds = _limits(top_a_xy, top_b_xy)
    bottom_bounds = _limits(bottom_a_xy, bottom_b_xy)

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
    event_text = fig.text(
        0.5,
        0.982,
        "" if event_label is None else event_label,
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

    sequences = [
        (top_a_xy, top_b_xy, top_title, top_bounds),
        (bottom_a_xy, bottom_b_xy, bottom_title, bottom_bounds),
    ]
    colors = {"a": "#2563eb", "b": "#d97706"}
    lines: list[tuple[Any, Any]] = []
    dots: list[tuple[Any, Any]] = []
    for ax, (xy_a, xy_b, title, bounds) in zip(axes, sequences, strict=True):
        ax.set_facecolor("#fffdf8")
        for spine in ax.spines.values():
            spine.set_color("#d6d3d1")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.plot(xy_a[:, 0], xy_a[:, 1], color=colors["a"], alpha=0.15, linewidth=1.3)
        ax.plot(xy_b[:, 0], xy_b[:, 1], color=colors["b"], alpha=0.15, linewidth=1.3)
        line_a, = ax.plot(
            [],
            [],
            color=colors["a"],
            linewidth=2.1,
            linestyle=(0, (5, 2)),
            label=label_a,
            zorder=6,
        )
        line_b, = ax.plot(
            [],
            [],
            color=colors["b"],
            linewidth=2.3,
            label=label_b,
            zorder=5,
        )
        dot_a = ax.scatter(
            [],
            [],
            s=64,
            facecolors="white",
            edgecolors=colors["a"],
            linewidths=1.8,
            marker="o",
            zorder=8,
        )
        dot_b = ax.scatter(
            [],
            [],
            s=64,
            color=colors["b"],
            linewidths=1.8,
            marker="X",
            zorder=7,
        )
        ax.legend(loc="upper right", frameon=False)
        lines.append((line_a, line_b))
        dots.append((dot_a, dot_b))

    frames_dir = output_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, phase_idx in enumerate(frame_indices):
        for seq_idx, (xy_a, xy_b, _title, _bounds) in enumerate(sequences):
            line_a, line_b = lines[seq_idx]
            dot_a, dot_b = dots[seq_idx]
            line_a.set_data(xy_a[: phase_idx + 1, 0], xy_a[: phase_idx + 1, 1])
            line_b.set_data(xy_b[: phase_idx + 1, 0], xy_b[: phase_idx + 1, 1])
            dot_a.set_offsets(xy_a[phase_idx : phase_idx + 1])
            dot_b.set_offsets(xy_b[phase_idx : phase_idx + 1])

        is_flash = (
            event_label is not None
            and event_frame_idx is not None
            and int(event_frame_idx) <= int(frame_idx) < (int(event_frame_idx) + int(flash_frames))
        )
        flash_overlay.set_alpha(0.24 if is_flash else 0.0)
        event_text.set_alpha(1.0 if is_flash else 0.0)
        status_text.set_text(
            f"Sample {int(phase_idx) + 1} / {phase_count}"
            f"   A env step {int(env_steps_a[phase_idx])}"
            f"   B env step {int(env_steps_b[phase_idx])}"
        )

        fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.92])
        fig.savefig(frames_dir / f"frame_{frame_idx:04d}.png", dpi=110)

    plt.close(fig)

    mp4_path = output_dir / "state_pca.mp4"
    gif_path = output_dir / "state_pca.gif"
    encode_frames(frames_dir, fps=int(fps), mp4_path=mp4_path, gif_path=gif_path)
    return {
        "coords_path": str(output_dir / "state_pca_coords.npz"),
        "mp4_path": str(mp4_path),
        "gif_path": str(gif_path),
        "frame_count": int(frame_indices.shape[0]),
        "fps": int(fps),
        "phase_count": int(phase_count),
        "event_frame_idx": None if event_frame_idx is None else int(event_frame_idx),
    }


def _save_condition_spectra(
    *,
    output_dir: Path,
    panel_name: str,
    label_a: str,
    label_b: str,
    color_a: str,
    color_b: str,
    freqs_a: np.ndarray,
    power_a: np.ndarray,
    freqs_b: np.ndarray,
    power_b: np.ndarray,
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
) -> dict[str, Any]:
    _save_overlay_spectrum(
        freqs_a=freqs_a,
        power_a=power_a,
        label_a=label_a,
        color_a=color_a,
        freqs_b=freqs_b,
        power_b=power_b,
        label_b=label_b,
        color_b=color_b,
        title=f"{panel_name} PCA spectral purity",
        output_path=output_dir / f"{panel_name.lower().replace(' ', '_')}_spectrum.png",
    )
    return {
        "panel": panel_name,
        "condition_a": {"label": label_a, **metrics_a},
        "condition_b": {"label": label_b, **metrics_b},
    }


def analyze_current_branch(
    *,
    state_pca_dir: Path,
    output_dir: Path,
    top_k_peaks: int,
) -> dict[str, Any]:
    coords = load_npz(state_pca_dir / "state_pca_coords.npz")
    state_pca_summary = load_json(state_pca_dir / "summary.json")
    trace_dir = Path(str(state_pca_summary["trace_dir"]))
    trace_summary = load_json(trace_dir / "summary.json")
    sample_spacing_env_steps = 1.0 / float(trace_summary["sense_act_phases"])

    output_dir.mkdir(parents=True, exist_ok=True)
    plant_freqs_a, plant_power_a, plant_metrics_a = _windowed_xy_spectrum(
        coords["baseline_plant_xy"],
        sample_spacing_env_steps=sample_spacing_env_steps,
        top_k_peaks=top_k_peaks,
    )
    plant_freqs_b, plant_power_b, plant_metrics_b = _windowed_xy_spectrum(
        coords["perturbed_plant_xy"],
        sample_spacing_env_steps=sample_spacing_env_steps,
        top_k_peaks=top_k_peaks,
    )
    ctrl_freqs_a, ctrl_power_a, ctrl_metrics_a = _windowed_xy_spectrum(
        coords["baseline_controller_xy"],
        sample_spacing_env_steps=sample_spacing_env_steps,
        top_k_peaks=top_k_peaks,
    )
    ctrl_freqs_b, ctrl_power_b, ctrl_metrics_b = _windowed_xy_spectrum(
        coords["perturbed_controller_xy"],
        sample_spacing_env_steps=sample_spacing_env_steps,
        top_k_peaks=top_k_peaks,
    )

    plant_summary = _save_condition_spectra(
        output_dir=output_dir,
        panel_name="Plant",
        label_a=str(trace_summary["row_label_top"]),
        label_b=str(trace_summary["row_label_bottom"]),
        color_a="#2563eb",
        color_b="#d97706",
        freqs_a=plant_freqs_a,
        power_a=plant_power_a,
        freqs_b=plant_freqs_b,
        power_b=plant_power_b,
        metrics_a=plant_metrics_a,
        metrics_b=plant_metrics_b,
    )
    controller_summary = _save_condition_spectra(
        output_dir=output_dir,
        panel_name="Controller",
        label_a=str(trace_summary["row_label_top"]),
        label_b=str(trace_summary["row_label_bottom"]),
        color_a="#2563eb",
        color_b="#d97706",
        freqs_a=ctrl_freqs_a,
        power_a=ctrl_power_a,
        freqs_b=ctrl_freqs_b,
        power_b=ctrl_power_b,
        metrics_a=ctrl_metrics_a,
        metrics_b=ctrl_metrics_b,
    )

    summary = {
        "state_pca_dir": str(state_pca_dir),
        "coords_path": str(state_pca_dir / "state_pca_coords.npz"),
        "sample_spacing_env_steps": float(sample_spacing_env_steps),
        "panels": [plant_summary, controller_summary],
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def _sorted_trial_rows(labels: pd.DataFrame, trial_id: str) -> np.ndarray:
    mask = (labels["trial_id"] == trial_id) & (labels["micro_index"] == 0)
    rows = labels.loc[mask].copy()
    if rows.empty:
        raise ValueError(f"Could not find phase-unique rows for trial_id={trial_id}.")
    order = np.lexsort(
        [
            rows["phase_index"].to_numpy(dtype=np.int32),
            rows["env_step"].to_numpy(dtype=np.int32),
        ]
    )
    return rows.index.to_numpy(dtype=np.int64)[order]


def build_matched_pair_from_bundle(
    *,
    bundle_dir: Path,
    output_dir: Path,
    fps: int,
    max_frames: int,
    top_k_peaks: int,
) -> dict[str, Any]:
    intact_dir = bundle_dir
    deformation_off_dir = bundle_dir / "cohorts" / "deformation_off_native_core"
    if not deformation_off_dir.exists():
        raise FileNotFoundError(f"Missing deformation_off cohort under {bundle_dir}.")

    intact_labels = pd.read_csv(intact_dir / "trace_labels.csv")
    deformation_off_labels = pd.read_csv(deformation_off_dir / "trace_labels.csv")
    intact_order = _sorted_trial_rows(intact_labels, INTACT_TRIAL_ID)
    deformation_off_order = _sorted_trial_rows(deformation_off_labels, DEFORMATION_OFF_TRIAL_ID)

    intact_nca = load_npz(intact_dir / "nca_states.npz")
    intact_ctrnn = load_npz(intact_dir / "ctrnn_states.npz")
    deformation_off_nca = load_npz(deformation_off_dir / "nca_states.npz")
    deformation_off_ctrnn = load_npz(deformation_off_dir / "ctrnn_states.npz")

    intact_nca_arr = next(iter(intact_nca.values()))[intact_order].astype(np.float32, copy=False)
    intact_ctrnn_arr = next(iter(intact_ctrnn.values()))[intact_order].astype(np.float32, copy=False)
    deformation_off_nca_arr = next(iter(deformation_off_nca.values()))[deformation_off_order].astype(
        np.float32, copy=False
    )
    deformation_off_ctrnn_arr = next(iter(deformation_off_ctrnn.values()))[
        deformation_off_order
    ].astype(np.float32, copy=False)

    intact_env_steps = intact_labels.loc[intact_order, "env_step"].to_numpy(dtype=np.int32, copy=True)
    deformation_off_env_steps = deformation_off_labels.loc[deformation_off_order, "env_step"].to_numpy(
        dtype=np.int32, copy=True
    )
    intact_phase_index = intact_labels.loc[intact_order, "phase_index"].to_numpy(dtype=np.int32, copy=True)
    deformation_off_phase_index = deformation_off_labels.loc[deformation_off_order, "phase_index"].to_numpy(
        dtype=np.int32, copy=True
    )

    intact_plant = intact_nca_arr[..., 0:2].reshape(intact_nca_arr.shape[0], -1)
    deformation_off_plant = deformation_off_nca_arr[..., 0:2].reshape(
        deformation_off_nca_arr.shape[0],
        -1,
    )
    intact_controller = np.concatenate(
        [intact_nca_arr.reshape(intact_nca_arr.shape[0], -1), intact_ctrnn_arr.reshape(intact_ctrnn_arr.shape[0], -1)],
        axis=1,
    )
    deformation_off_controller = np.concatenate(
        [
            deformation_off_nca_arr.reshape(deformation_off_nca_arr.shape[0], -1),
            deformation_off_ctrnn_arr.reshape(deformation_off_ctrnn_arr.shape[0], -1),
        ],
        axis=1,
    )

    plant_pca = fit_pca(intact_plant, deformation_off_plant)
    controller_pca = fit_pca(intact_controller, deformation_off_controller)
    intact_plant_xy = transform_pca(intact_plant, plant_pca)
    deformation_off_plant_xy = transform_pca(deformation_off_plant, plant_pca)
    intact_controller_xy = transform_pca(intact_controller, controller_pca)
    deformation_off_controller_xy = transform_pca(deformation_off_controller, controller_pca)

    common_phase_count = int(min(intact_plant_xy.shape[0], deformation_off_plant_xy.shape[0]))
    intact_plant_xy = intact_plant_xy[:common_phase_count]
    deformation_off_plant_xy = deformation_off_plant_xy[:common_phase_count]
    intact_controller_xy = intact_controller_xy[:common_phase_count]
    deformation_off_controller_xy = deformation_off_controller_xy[:common_phase_count]
    intact_env_steps = intact_env_steps[:common_phase_count]
    deformation_off_env_steps = deformation_off_env_steps[:common_phase_count]
    intact_phase_index = intact_phase_index[:common_phase_count]
    deformation_off_phase_index = deformation_off_phase_index[:common_phase_count]

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "state_pca_coords.npz",
        intact_plant_xy=intact_plant_xy,
        deformation_off_plant_xy=deformation_off_plant_xy,
        intact_controller_xy=intact_controller_xy,
        deformation_off_controller_xy=deformation_off_controller_xy,
        intact_env_steps=intact_env_steps,
        deformation_off_env_steps=deformation_off_env_steps,
        intact_phase_index=intact_phase_index,
        deformation_off_phase_index=deformation_off_phase_index,
        plant_mean=plant_pca["mean"],
        plant_std=plant_pca["std"],
        plant_components=plant_pca["components"],
        controller_mean=controller_pca["mean"],
        controller_std=controller_pca["std"],
        controller_components=controller_pca["components"],
    )

    manifest = load_json(intact_dir / "trace_manifest.json")
    sample_spacing_env_steps = 1.0 / float(manifest["sense_act_phases"])

    plant_freqs_a, plant_power_a, plant_metrics_a = _windowed_xy_spectrum(
        intact_plant_xy,
        sample_spacing_env_steps=sample_spacing_env_steps,
        top_k_peaks=top_k_peaks,
    )
    plant_freqs_b, plant_power_b, plant_metrics_b = _windowed_xy_spectrum(
        deformation_off_plant_xy,
        sample_spacing_env_steps=sample_spacing_env_steps,
        top_k_peaks=top_k_peaks,
    )
    ctrl_freqs_a, ctrl_power_a, ctrl_metrics_a = _windowed_xy_spectrum(
        intact_controller_xy,
        sample_spacing_env_steps=sample_spacing_env_steps,
        top_k_peaks=top_k_peaks,
    )
    ctrl_freqs_b, ctrl_power_b, ctrl_metrics_b = _windowed_xy_spectrum(
        deformation_off_controller_xy,
        sample_spacing_env_steps=sample_spacing_env_steps,
        top_k_peaks=top_k_peaks,
    )

    plant_summary = _save_condition_spectra(
        output_dir=output_dir,
        panel_name="Plant",
        label_a="Intact (4,2)",
        label_b="Deformation Off (4,2)",
        color_a="#2563eb",
        color_b="#d97706",
        freqs_a=plant_freqs_a,
        power_a=plant_power_a,
        freqs_b=plant_freqs_b,
        power_b=plant_power_b,
        metrics_a=plant_metrics_a,
        metrics_b=plant_metrics_b,
    )
    controller_summary = _save_condition_spectra(
        output_dir=output_dir,
        panel_name="Controller",
        label_a="Intact (4,2)",
        label_b="Deformation Off (4,2)",
        color_a="#2563eb",
        color_b="#d97706",
        freqs_a=ctrl_freqs_a,
        power_a=ctrl_power_a,
        freqs_b=ctrl_freqs_b,
        power_b=ctrl_power_b,
        metrics_a=ctrl_metrics_a,
        metrics_b=ctrl_metrics_b,
    )

    movie_summary = _render_state_pca_movie(
        top_a_xy=intact_plant_xy,
        top_b_xy=deformation_off_plant_xy,
        bottom_a_xy=intact_controller_xy,
        bottom_b_xy=deformation_off_controller_xy,
        label_a="Intact (4,2)",
        label_b="Deformation Off (4,2)",
        top_title="Plant PCA: raw absolute H/V state",
        bottom_title="Controller PCA",
        env_steps_a=intact_env_steps,
        env_steps_b=deformation_off_env_steps,
        phase_count=common_phase_count,
        output_dir=output_dir,
        fps=fps,
        max_frames=max_frames,
        event_label=None,
        event_frame_idx=None,
    )

    summary = {
        "bundle_dir": str(bundle_dir),
        "intact_trial_id": INTACT_TRIAL_ID,
        "deformation_off_trial_id": DEFORMATION_OFF_TRIAL_ID,
        "sample_spacing_env_steps": float(sample_spacing_env_steps),
        "intact_original_phase_count": int(intact_plant.shape[0]),
        "deformation_off_original_phase_count": int(deformation_off_plant.shape[0]),
        "common_phase_count": int(common_phase_count),
        "movie": movie_summary,
        "panels": [plant_summary, controller_summary],
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    outputs: dict[str, Any] = {}
    if not args.skip_branch:
        outputs["current_branch"] = analyze_current_branch(
            state_pca_dir=args.branch_state_pca_dir.resolve(),
            output_dir=args.branch_output_dir.resolve(),
            top_k_peaks=int(args.top_k_peaks),
        )
    if not args.skip_matched:
        outputs["matched_pair_4_2"] = build_matched_pair_from_bundle(
            bundle_dir=args.matched_bundle_dir.resolve(),
            output_dir=args.matched_output_dir.resolve(),
            fps=int(args.fps),
            max_frames=int(args.max_frames),
            top_k_peaks=int(args.top_k_peaks),
        )
    if outputs:
        write_json(
            (args.matched_output_dir.resolve().parent / "state_pca_spectral_purity_index.json"),
            outputs,
        )
    for key, payload in outputs.items():
        print(f"{key}: {payload.get('movie', {}).get('mp4_path', payload.get('state_pca_dir'))}")


if __name__ == "__main__":
    main()
