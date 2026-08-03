#!/usr/bin/env python3
"""Create the curated eight-specimen Sensing at the Edges analysis packet."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import warnings
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from committed_direction_decoding import _auc, _balanced_accuracy, _mcc, _ridge_scores
import make_sensing_edges_revision_materials as base
import make_sensing_edges_tau_histogram as tau_mod
from signal_susceptibility_analysis import _load_trials as _load_signal_trials


PACKET_ROOT = (
    base.PAPER_ROOT / "sensing_edges_8_specimen_analysis_2026-06-13"
)
REVISION_ROOT = base.DEFAULT_OUTPUT_ROOT
EDGE_TRACE_ROOT = (
    base.DATA_ROOT
    / "raw_modal_edge_state_space_deformation_off_20260608T041827"
    / "state_space_trace_edges_deformation_off"
)

OUTCOME_COLORS = {
    "success": base.C_SUCCESS,
    "survival": base.C_SUCCESS,
    "fall": base.C_FALL,
    "stall": base.C_STALL,
}


def _set_original_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.titlesize": 12.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 10.0,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def _save(fig: plt.Figure, path: Path, *, dpi: int = 240) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _fmt_pct(value: float) -> str:
    rounded = Decimal(str(100.0 * float(value))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return f"{rounded}%"


def _fmt_pp(value: float) -> str:
    rounded = Decimal(str(100.0 * float(value))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return f"{rounded} pp"


def _sem(values: np.ndarray | pd.Series) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def _rate_by_specimen(df: pd.DataFrame, expr: pd.Series) -> pd.Series:
    tmp = df[["specimen_id"]].copy()
    tmp["value"] = expr.astype(float).to_numpy()
    return tmp.groupby("specimen_id", sort=True)["value"].mean()


def _outcome_mean_sem(
    eps: pd.DataFrame,
    *,
    assay_family: str,
    surface_id: str,
    condition_id: str,
    distance_pair: str | None = None,
    width: int | None = None,
    specimen_id: str | None = None,
) -> dict[str, tuple[float, float]]:
    df = eps[
        (eps["assay_family"] == assay_family)
        & (eps["surface_id"] == surface_id)
        & (eps["condition_id"] == condition_id)
    ].copy()
    if distance_pair is not None:
        df = df[df["distance_pair"] == distance_pair]
    if width is not None:
        df = df[df["width"].fillna(-1).astype(int) == int(width)]
    if specimen_id is not None:
        df = df[df["specimen_id"] == specimen_id]
    if df.empty:
        raise ValueError((assay_family, surface_id, condition_id, distance_pair, width, specimen_id))

    if surface_id == "double_cliff":
        survival = _rate_by_specimen(df, df["survived"])
        fall = _rate_by_specimen(df, df["fell"])
        stall = 1.0 - survival - fall
        return {
            "survival": (float(survival.mean()), _sem(survival)),
            "fall": (float(fall.mean()), _sem(fall)),
            "stall": (float(stall.mean()), _sem(stall)),
        }

    success = _rate_by_specimen(df, df["success"])
    fall = _rate_by_specimen(df, df["fell"])
    stall = _rate_by_specimen(df, (~df["success"]) & (~df["fell"]))
    return {
        "success": (float(success.mean()), _sem(success)),
        "fall": (float(fall.mean()), _sem(fall)),
        "stall": (float(stall.mean()), _sem(stall)),
    }


def _plot_grouped_outcomes(
    ax: plt.Axes,
    *,
    specs: list[dict[str, object]],
    outcomes: tuple[str, ...],
    title: str | None = None,
    show_ylabel: bool = True,
    legend: bool = False,
) -> None:
    x = np.arange(len(specs), dtype=float)
    width = 0.22 if len(outcomes) == 3 else 0.28
    offsets = (np.arange(len(outcomes)) - (len(outcomes) - 1) / 2.0) * width
    for offset, outcome in zip(offsets, outcomes, strict=True):
        means = [100.0 * float(spec["rates"][outcome][0]) for spec in specs]
        sems = [100.0 * float(spec["rates"][outcome][1]) for spec in specs]
        label = {"success": "Success", "survival": "Survival", "fall": "Fall", "stall": "Stall"}[outcome]
        ax.bar(
            x + offset,
            means,
            width=width * 0.92,
            yerr=sems,
            color=OUTCOME_COLORS[outcome],
            edgecolor="white",
            linewidth=0.7,
            ecolor="black",
            capsize=2.5,
            label=label,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(spec["label"]) for spec in specs])
    ax.set_ylim(0, 105)
    if show_ylabel:
        ax.set_ylabel("Rate (%)")
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(frameon=True, fontsize=8, loc="upper right")
    _style_axes(ax)


def plot_performance_original_style(eps: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.1), constrained_layout=True)

    native_specs = []
    for pair in ["(2,1)", "(4,2)", "(6,3)"]:
        native_specs.append(
            {
                "label": pair,
                "rates": _outcome_mean_sem(
                    eps,
                    assay_family="signaling_ablation",
                    surface_id="native_core",
                    condition_id="intact",
                    distance_pair=pair,
                ),
            }
        )
    _plot_grouped_outcomes(
        axes[0],
        specs=native_specs,
        outcomes=("success", "fall", "stall"),
        title="A  Native core",
    )

    extended_specs = []
    for pair in ["(8,4)", "(10,5)"]:
        extended_specs.append(
            {
                "label": pair,
                "rates": _outcome_mean_sem(
                    eps,
                    assay_family="signaling_ablation",
                    surface_id="long_distance_generalization",
                    condition_id="intact",
                    distance_pair=pair,
                ),
            }
        )
    _plot_grouped_outcomes(
        axes[1],
        specs=extended_specs,
        outcomes=("success", "fall", "stall"),
        title="B  Extended",
        show_ylabel=False,
    )

    double_specs = []
    for width in [2, 4, 6]:
        double_specs.append(
            {
                "label": f"w={width}",
                "rates": _outcome_mean_sem(
                    eps,
                    assay_family="signaling_ablation",
                    surface_id="double_cliff",
                    condition_id="intact",
                    width=width,
                ),
            }
        )
    _plot_grouped_outcomes(
        axes[2],
        specs=double_specs,
        outcomes=("survival", "fall", "stall"),
        title="C  Double cliff",
        show_ylabel=False,
        legend=True,
    )
    return _save(fig, out_dir / "performance_by_surface_original_style.pdf")


def plot_ablations_original_style(eps: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(7.8, 5.35), constrained_layout=True)

    signaling_specs = []
    signaling_display_order = [
        "intact",
        "sobel_x_off",
        "sobel_y_off",
        "laplacian_off",
        "identity_comm_off",
        "all_comm_off",
    ]
    for cond in signaling_display_order:
        signaling_specs.append(
            {
                "label": base.SIGNALING_LABELS[cond],
                "rates": _outcome_mean_sem(
                    eps,
                    assay_family="signaling_ablation",
                    surface_id="native_core",
                    condition_id=cond,
                ),
            }
        )
    _plot_grouped_outcomes(
        axes[0],
        specs=signaling_specs,
        outcomes=("success", "fall", "stall"),
        title="A  Signalling ablations",
        legend=True,
    )

    deformation_specs = []
    deformation_display_order = ["intact", "absolute_only", "compliance_only", "deformation_off"]
    for cond in deformation_display_order:
        deformation_specs.append(
            {
                "label": base.DEFORMATION_LABELS[cond],
                "rates": _outcome_mean_sem(
                    eps,
                    assay_family="deformation_ablation",
                    surface_id="native_core",
                    condition_id=cond,
                ),
            }
        )
    _plot_grouped_outcomes(
        axes[1],
        specs=deformation_specs,
        outcomes=("success", "fall", "stall"),
        title="B  Deformation ablations",
    )
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=0)
    return _save(fig, out_dir / "ablations_original_style.pdf")


def plot_body_transfer_original_style(eps: pd.DataFrame, out_dir: Path) -> Path:
    labels = {
        "rect_7x2": "7x2 (native)",
        "rect_6x2": "6x2",
        "rect_8x2": "8x2",
        "rect_7x3": "7x3",
    }
    specs = []
    for cond in ["rect_7x2", "rect_6x2", "rect_8x2", "rect_7x3"]:
        specs.append(
            {
                "label": labels[cond],
                "rates": _outcome_mean_sem(
                    eps,
                    assay_family="body_transfer",
                    surface_id="native_core",
                    condition_id=cond,
                ),
            }
        )
    fig, ax = plt.subplots(figsize=(5.0, 3.0), constrained_layout=True)
    _plot_grouped_outcomes(
        ax,
        specs=specs,
        outcomes=("success", "fall", "stall"),
        legend=True,
    )
    return _save(fig, out_dir / "body_transfer_original_style.pdf")


def _s4_rate_spec(
    eps: pd.DataFrame,
    *,
    label: str,
    assay_family: str,
    surface_id: str,
    condition_id: str,
    distance_pair: str | None = None,
    width: int | None = None,
) -> dict[str, object]:
    return {
        "label": label,
        "rates": _outcome_mean_sem(
            eps,
            assay_family=assay_family,
            surface_id=surface_id,
            condition_id=condition_id,
            distance_pair=distance_pair,
            width=width,
            specimen_id=base.S4_SPECIMEN,
        ),
    }


def plot_s4_profile_original_style(eps: pd.DataFrame, out_dir: Path) -> Path:
    specs = [
        _s4_rate_spec(
            eps,
            label="Native core",
            assay_family="signaling_ablation",
            surface_id="native_core",
            condition_id="intact",
        ),
        _s4_rate_spec(
            eps,
            label="Extended (8,4)",
            assay_family="signaling_ablation",
            surface_id="long_distance_generalization",
            condition_id="intact",
            distance_pair="(8,4)",
        ),
        _s4_rate_spec(
            eps,
            label="Extended (10,5)",
            assay_family="signaling_ablation",
            surface_id="long_distance_generalization",
            condition_id="intact",
            distance_pair="(10,5)",
        ),
        _s4_rate_spec(
            eps,
            label="Double cliff w=2",
            assay_family="signaling_ablation",
            surface_id="double_cliff",
            condition_id="intact",
            width=2,
        ),
        _s4_rate_spec(
            eps,
            label="Double cliff w=4",
            assay_family="signaling_ablation",
            surface_id="double_cliff",
            condition_id="intact",
            width=4,
        ),
        _s4_rate_spec(
            eps,
            label="Double cliff w=6",
            assay_family="signaling_ablation",
            surface_id="double_cliff",
            condition_id="intact",
            width=6,
        ),
    ]
    signaling_display_order = [
        "intact",
        "sobel_x_off",
        "sobel_y_off",
        "laplacian_off",
        "identity_comm_off",
        "all_comm_off",
    ]
    for cond in signaling_display_order:
        specs.append(
            _s4_rate_spec(
                eps,
                label=base.SIGNALING_LABELS[cond],
                assay_family="signaling_ablation",
                surface_id="native_core",
                condition_id=cond,
            )
        )
    for cond in ["intact", "deformation_off", "compliance_only", "absolute_only"]:
        label = "Def. intact" if cond == "intact" else base.DEFORMATION_LABELS[cond]
        specs.append(
            _s4_rate_spec(
                eps,
                label=label,
                assay_family="deformation_ablation",
                surface_id="native_core",
                condition_id=cond,
            )
        )

    fig, ax = plt.subplots(figsize=(9.7, 3.1), constrained_layout=True)
    x = np.arange(len(specs), dtype=float)
    width = 0.19
    outcomes = ("success", "fall", "stall")
    for offset, outcome in zip([-width, 0.0, width], outcomes, strict=True):
        vals = []
        for spec in specs:
            rates = spec["rates"]
            if outcome == "success" and "survival" in rates:
                vals.append(100.0 * float(rates["survival"][0]))
            else:
                vals.append(100.0 * float(rates[outcome][0]))
        ax.bar(
            x + offset,
            vals,
            width=width * 0.92,
            color=OUTCOME_COLORS[outcome],
            edgecolor="white",
            linewidth=0.7,
            label="Success / survival" if outcome == "success" else outcome.title(),
        )
    for cut in [5.5, 11.5]:
        ax.axvline(cut, color="#DDDDDD", linewidth=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels([str(spec["label"]) for spec in specs], rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Rate (%)")
    ax.legend(frameon=True, fontsize=8, loc="upper right")
    _style_axes(ax)
    return _save(fig, out_dir / "s4_profile_original_style.pdf")


def plot_tau_original_style(out_dir: Path, tables_dir: Path) -> tuple[Path, dict[str, object]]:
    tau_values, specimen_summary = tau_mod.load_tau_values()
    tau_values.to_csv(tables_dir / "eight_specimen_tau_values.csv", index=False)
    specimen_summary.to_csv(tables_dir / "eight_specimen_tau_summary.csv", index=False)

    tau_min = float(tau_values["tau_min"].iloc[0])
    tau_max = float(tau_values["tau_max"].iloc[0])
    bins = np.arange(tau_min, tau_max + 1.0001, 1.0)
    bin_counts = tau_mod._binned_counts_by_specimen(tau_values, bins)
    bin_counts.to_csv(tables_dir / "eight_specimen_tau_bin_counts.csv", index=False)
    binned = (
        bin_counts.groupby(["bin_index", "bin_left", "bin_right"], as_index=False)
        .agg(
            mean_count=("count", "mean"),
            sem_count=("count", lambda values: _sem(np.asarray(values, dtype=float))),
        )
        .sort_values("bin_index")
    )
    values = tau_values["tau"].to_numpy(dtype=float)
    centers = (binned["bin_left"].to_numpy(float) + binned["bin_right"].to_numpy(float)) / 2.0
    widths = binned["bin_right"].to_numpy(float) - binned["bin_left"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(4.7, 3.0), constrained_layout=True)
    ax.bar(
        centers,
        binned["mean_count"].to_numpy(float),
        yerr=binned["sem_count"].to_numpy(float),
        width=widths * 0.92,
        color=base.C_ACCENT,
        edgecolor="white",
        linewidth=0.7,
        ecolor="black",
        capsize=2.5,
    )
    ax.axvline(np.median(values), color=base.C_NEUTRAL, linestyle="--", linewidth=1.0, label="Median")
    ax.set_xlim(tau_min, tau_max)
    ax.set_xticks(np.arange(tau_min, tau_max + 0.1, 1.0))
    ax.set_xlabel(f"Tau (allowed range {tau_min:g}-{tau_max:g})")
    ax.set_ylabel("Mean hidden units / specimen")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _style_axes(ax)
    path = _save(fig, out_dir / "tau_distribution_original_style.pdf")

    payload = {
        "n_specimens": int(tau_values["specimen_id"].nunique()),
        "ctrnn_units_per_specimen": int(tau_values.groupby("specimen_id").size().iloc[0]),
        "n_tau_values": int(values.size),
        "allowed_tau_min": tau_min,
        "allowed_tau_max": tau_max,
        "observed_min": float(np.min(values)),
        "observed_q25": float(np.quantile(values, 0.25)),
        "observed_median": float(np.median(values)),
        "observed_mean": float(np.mean(values)),
        "observed_q75": float(np.quantile(values, 0.75)),
        "observed_max": float(np.max(values)),
        "observed_std": float(np.std(values, ddof=1)),
        "histogram_y_axis": "Mean hidden-unit count per tau bin across specimens; whiskers are SEM across specimens.",
    }
    (tables_dir / "eight_specimen_tau_overall_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return path, payload


def plot_s4_predictivity_heatmaps(out_dir: Path) -> Path:
    source = (
        base.DATA_ROOT
        / "figures"
        / "signal_susceptibility_deformation_off"
        / "sliding_signal_body_future_delta_regression.csv"
    )
    data = pd.read_csv(source)
    sub = data[
        (data["relative_center"] == 0)
        & (data["lag"] == 8)
        & (data["stream"].isin(["body", "signal"]))
    ].copy()
    if len(sub) != 28:
        raise ValueError(f"Expected 28 body/signal cell rows at event center, got {len(sub)}")

    layout = np.array([[7, 8, 9, 10, 11, 12, 13], [0, 1, 2, 3, 4, 5, 6]], dtype=int)
    values: dict[str, np.ndarray] = {}
    for stream in ["body", "signal"]:
        arr = np.full(layout.shape, np.nan, dtype=float)
        s = sub[sub["stream"] == stream].set_index("cell_index")
        for r in range(layout.shape[0]):
            for c in range(layout.shape[1]):
                arr[r, c] = float(s.loc[int(layout[r, c]), "pearson"])
        values[stream] = arr

    vmax = float(np.nanmax(np.abs(np.concatenate([values["body"].ravel(), values["signal"].ravel()]))))
    vmax = max(0.30, math.ceil(vmax * 20.0) / 20.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 3.35), constrained_layout=True)
    titles = [
        "Body-state predictivity",
        "Signaling-state predictivity",
    ]
    image = None
    for ax, stream, title in zip(axes, ["body", "signal"], titles, strict=True):
        image = ax.imshow(values[stream], cmap="PRGn", norm=norm, aspect="equal")
        for r in range(layout.shape[0]):
            for c in range(layout.shape[1]):
                cell = int(layout[r, c])
                val = values[stream][r, c]
                text_color = "white" if abs(val) >= 0.10 else "#222222"
                ax.text(c, r, f"{cell}\n{val:+.2f}", ha="center", va="center", fontsize=8, color=text_color)
        ax.set_title(title)
        ax.set_xticks(np.arange(7))
        ax.set_xticklabels([str(i) for i in range(7)])
        ax.set_yticks([])
        ax.spines[:].set_visible(True)
    axes[-1].set_xlabel("cell column")
    fig.suptitle("S4 event-centered cell predictivity | center +0, lag 8", fontweight="bold", fontsize=11)
    if image is None:
        raise RuntimeError("No heatmap image generated")
    cbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.03)
    cbar.set_label("Pearson r")
    return _save(fig, out_dir / "s4_body_signal_predictivity_turn_heatmaps.pdf")


def plot_s4_frontier_band_auc_heatmap(out_dir: Path) -> Path:
    source = (
        base.DATA_ROOT
        / "figures"
        / "voxel_reversal_heatmap_deformation_off"
        / "voxel_decoder_summary.csv"
    )
    data = pd.read_csv(source)
    sub = data[(data["window"] == "edge_impulse") & (data["lag"] == 8)].copy()
    if sub["cell_index"].nunique() != 14:
        raise ValueError(f"Expected 14 voxel decoder rows, got {sub['cell_index'].nunique()}")

    layout = np.array([[7, 8, 9, 10, 11, 12, 13], [0, 1, 2, 3, 4, 5, 6]], dtype=int)
    values = np.full(layout.shape, np.nan, dtype=float)
    indexed = sub.set_index("cell_index")
    for r in range(layout.shape[0]):
        for c in range(layout.shape[1]):
            values[r, c] = float(indexed.loc[int(layout[r, c]), "mean_auc"])

    fig, ax = plt.subplots(figsize=(6.35, 2.25), constrained_layout=True)
    image = ax.imshow(values, cmap="viridis", vmin=0.5, vmax=1.0, aspect="equal")
    for r in range(layout.shape[0]):
        for c in range(layout.shape[1]):
            cell = int(layout[r, c])
            auc = values[r, c]
            ax.text(c, r, f"{cell}\n{auc:.3f}", ha="center", va="center", fontsize=8, color="white")

    ax.add_patch(
        plt.Rectangle(
            (-0.5, -0.5),
            2.0,
            2.0,
            fill=False,
            edgecolor="#D62728",
            linewidth=2.3,
        )
    )
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels([str(i) for i in range(7)])
    ax.set_yticks([])
    ax.set_xlabel("cell column")
    ax.set_title("S4 frontier-band direction predictivity | edge impulse, lag 8")
    cbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.03)
    cbar.set_label("AUC")
    for spine in ax.spines.values():
        spine.set_visible(True)
    return _save(fig, out_dir / "s4_frontier_band_auc_edge_impulse_lag8.pdf")


def _signal_cell_windowed_xy(
    trial: dict[str, object],
    *,
    cell_index: int,
    lag: int,
    rel_start: int,
    rel_stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(trial["signal_cell_features"][int(cell_index)], dtype=np.float32)
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


def _compute_signal_cell_auc(
    *,
    trace_root: Path,
    lag: int,
    rel_start: int,
    rel_stop: int,
    ridge_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.DtypeWarning)
        trials, trial_rows, robot_shape, feature_manifest = _load_signal_trials(
            trace_root=trace_root,
            future_window=32,
            min_displacement=0.025,
            persistence_steps=12,
            persistence_fraction=0.75,
        )
    fold_rows: list[dict[str, object]] = []
    cell_count = int(robot_shape[0]) * int(robot_shape[1])
    for cell_index in range(cell_count):
        for heldout in trials:
            train_x_blocks: list[np.ndarray] = []
            train_y_blocks: list[np.ndarray] = []
            for trial in trials:
                if trial["trial_id"] == heldout["trial_id"]:
                    continue
                x_block, y_block = _signal_cell_windowed_xy(
                    trial,
                    cell_index=int(cell_index),
                    lag=int(lag),
                    rel_start=int(rel_start),
                    rel_stop=int(rel_stop),
                )
                if y_block.size:
                    train_x_blocks.append(x_block)
                    train_y_blocks.append(y_block)
            test_x, test_y = _signal_cell_windowed_xy(
                heldout,
                cell_index=int(cell_index),
                lag=int(lag),
                rel_start=int(rel_start),
                rel_stop=int(rel_stop),
            )
            base_row = {
                "cell_index": int(cell_index),
                "heldout_trial_id": str(heldout["trial_id"]),
                "lag": int(lag),
                "relative_start": int(rel_start),
                "relative_stop": int(rel_stop),
                "sample_count": int(test_y.size),
                "toward_count": int(np.sum(test_y == 1)),
                "away_count": int(np.sum(test_y == -1)),
            }
            if not train_y_blocks or test_y.size == 0:
                fold_rows.append({**base_row, "skip_reason": "empty_train_or_test"})
                continue
            train_x = np.concatenate(train_x_blocks, axis=0)
            train_y = np.concatenate(train_y_blocks, axis=0)
            if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
                fold_rows.append({**base_row, "skip_reason": "single_class_train_or_test"})
                continue
            scores = _ridge_scores(train_x, train_y, test_x, alpha=float(ridge_alpha))
            fold_rows.append(
                {
                    **base_row,
                    "accuracy": float(np.mean(np.where(scores >= 0.0, 1, -1) == test_y)),
                    "balanced_accuracy": _balanced_accuracy(test_y, scores),
                    "auc": _auc(test_y, scores),
                    "mcc": _mcc(test_y, scores),
                    "skip_reason": "",
                }
            )

    folds = pd.DataFrame(fold_rows)
    valid = folds[(folds["skip_reason"].fillna("") == "") & np.isfinite(folds["auc"])]
    summary_rows: list[dict[str, object]] = []
    for cell_index, group in valid.groupby("cell_index", sort=True):
        summary_rows.append(
            {
                "cell_index": int(cell_index),
                "heldout_trial_count": int(group["heldout_trial_id"].nunique()),
                "lag": int(lag),
                "relative_start": int(rel_start),
                "relative_stop": int(rel_stop),
                "mean_auc": float(group["auc"].mean()),
                "mean_balanced_accuracy": float(group["balanced_accuracy"].mean()),
                "mean_mcc": float(group["mcc"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    metadata = {
        "trace_root": str(trace_root),
        "included_trials": [str(trial["trial_id"]) for trial in trials],
        "trial_metadata": trial_rows,
        "robot_shape": [int(robot_shape[0]), int(robot_shape[1])],
        "feature_manifest": feature_manifest,
        "window": "edge_impulse",
        "lag": int(lag),
        "relative_start": int(rel_start),
        "relative_stop": int(rel_stop),
        "ridge_alpha": float(ridge_alpha),
        "metric": "leave-one-trial-out AUC",
        "signal_feature_policy": feature_manifest.get("signal_feature_policy"),
    }
    return summary, folds, metadata


def plot_s4_signal_body_predictivity_stacked(
    out_dir: Path,
    tables_dir: Path,
) -> tuple[Path, dict[str, Path], dict[str, float]]:
    body_source = (
        base.DATA_ROOT
        / "figures"
        / "voxel_reversal_heatmap_deformation_off"
        / "voxel_decoder_summary.csv"
    )
    signal_data, signal_folds, signal_metadata = _compute_signal_cell_auc(
        trace_root=EDGE_TRACE_ROOT,
        lag=8,
        rel_start=-32,
        rel_stop=96,
        ridge_alpha=10.0,
    )
    body_data = pd.read_csv(body_source)
    body_sub = body_data[(body_data["window"] == "edge_impulse") & (body_data["lag"] == 8)].copy()
    if signal_data["cell_index"].nunique() != 14:
        raise ValueError(f"Expected 14 signaling cell rows, got {signal_data['cell_index'].nunique()}")
    if body_sub["cell_index"].nunique() != 14:
        raise ValueError(f"Expected 14 body cell rows, got {body_sub['cell_index'].nunique()}")

    tables_dir.mkdir(parents=True, exist_ok=True)
    signal_summary_path = tables_dir / "s4_signal_cell_edge_impulse_lag8_auc_summary.csv"
    signal_folds_path = tables_dir / "s4_signal_cell_edge_impulse_lag8_auc_folds.csv"
    signal_metadata_path = tables_dir / "s4_signal_cell_edge_impulse_lag8_auc_metadata.json"
    signal_data.to_csv(signal_summary_path, index=False)
    signal_folds.to_csv(signal_folds_path, index=False)
    signal_metadata_path.write_text(json.dumps(signal_metadata, indent=2) + "\n", encoding="utf-8")

    layout = np.array([[7, 8, 9, 10, 11, 12, 13], [0, 1, 2, 3, 4, 5, 6]], dtype=int)
    signal_values = np.full(layout.shape, np.nan, dtype=float)
    body_values = np.full(layout.shape, np.nan, dtype=float)
    signal_indexed = signal_data.set_index("cell_index")
    body_indexed = body_sub.set_index("cell_index")
    for r in range(layout.shape[0]):
        for c in range(layout.shape[1]):
            cell = int(layout[r, c])
            signal_values[r, c] = float(signal_indexed.loc[cell, "mean_auc"])
            body_values[r, c] = float(body_indexed.loc[cell, "mean_auc"])

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(6.35, 3.65),
        constrained_layout=True,
    )
    panels = [
        (
            axes[0],
            signal_values,
            "A  Signalling state",
        ),
        (
            axes[1],
            body_values,
            "B  Body state",
        ),
    ]

    image = None
    for ax, values, title in panels:
        image = ax.imshow(values, cmap="viridis", vmin=0.5, vmax=1.0, aspect="equal")
        for r in range(layout.shape[0]):
            for c in range(layout.shape[1]):
                cell = int(layout[r, c])
                text_color = "white" if values[r, c] >= 0.72 else "#222222"
                ax.text(c, r, str(cell), ha="center", va="center", fontsize=8, color=text_color)
        ax.add_patch(
            plt.Rectangle(
                (-0.5, -0.5),
                2.0,
                2.0,
                fill=False,
                edgecolor="#D62728",
                linewidth=2.1,
            )
        )
        ax.set_title(title, loc="left", fontsize=10.5, pad=4)
        ax.set_xticks(np.arange(7))
        ax.set_xticklabels([str(i) for i in range(7)])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)

    axes[-1].set_xlabel("cell column")
    if image is None:
        raise RuntimeError("No heatmap image generated")
    cbar = fig.colorbar(image, ax=axes, shrink=0.76, pad=0.025)
    cbar.set_label("AUC")
    path = _save(fig, out_dir / "s4_signal_body_predictivity_stacked.pdf")
    top_signal = signal_data.sort_values("mean_auc", ascending=False).iloc[0]
    return (
        path,
        {
            "s4_signal_cell_auc_summary": signal_summary_path,
            "s4_signal_cell_auc_folds": signal_folds_path,
            "s4_signal_cell_auc_metadata": signal_metadata_path,
        },
        {
            "top_signal_cell": float(top_signal["cell_index"]),
            "top_signal_auc": float(top_signal["mean_auc"]),
            "frontier_signal_cell_7_auc": float(signal_indexed.loc[7, "mean_auc"]),
            "frontier_signal_cell_8_auc": float(signal_indexed.loc[8, "mean_auc"]),
            "frontier_signal_cell_0_auc": float(signal_indexed.loc[0, "mean_auc"]),
            "frontier_signal_cell_1_auc": float(signal_indexed.loc[1, "mean_auc"]),
        },
    )


def _copy_file(src: Path, dst: Path) -> Path:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _copy_matching(src_dir: Path, dst_dir: Path, patterns: tuple[str, ...]) -> list[Path]:
    copied: list[Path] = []
    if not src_dir.exists():
        return copied
    for pattern in patterns:
        for src in sorted(src_dir.glob(pattern)):
            if src.name == ".DS_Store":
                continue
            copied.append(_copy_file(src, dst_dir / src.name))
    return copied


def _write_tables(packet_root: Path, eps: pd.DataFrame) -> dict[str, Path]:
    tables_dir = packet_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary = base._build_endpoint_summary(eps)
    deltas, specimen_deltas = base._build_paired_deltas(eps)
    with_without_s4 = base._build_with_without_s4_summary(eps)
    axes = base._build_failure_axes(eps)
    pairs = base._build_distance_pair_summary(eps)
    outputs = {
        "endpoint_summary": tables_dir / "eight_specimen_endpoint_summary.csv",
        "paired_deltas": tables_dir / "eight_specimen_paired_deltas.csv",
        "specimen_deltas": tables_dir / "eight_specimen_specimen_deltas.csv",
        "with_without_s4": tables_dir / "eight_specimen_with_without_s4_summary.csv",
        "failure_axes": tables_dir / "specimen_failure_axes.csv",
        "distance_pairs": tables_dir / "intact_distance_pair_summary.csv",
    }
    summary.to_csv(outputs["endpoint_summary"], index=False)
    deltas.to_csv(outputs["paired_deltas"], index=False)
    specimen_deltas.to_csv(outputs["specimen_deltas"], index=False)
    with_without_s4.to_csv(outputs["with_without_s4"], index=False)
    axes.to_csv(outputs["failure_axes"], index=False)
    pairs.to_csv(outputs["distance_pairs"], index=False)
    return outputs


def _summary_row(summary: pd.DataFrame, assay: str, surface: str, cond: str) -> pd.Series:
    row = summary[
        (summary["assay_family"] == assay)
        & (summary["surface_id"] == surface)
        & (summary["condition_id"] == cond)
    ]
    if row.empty:
        raise KeyError((assay, surface, cond))
    return row.iloc[0]


def _delta_row(deltas: pd.DataFrame, assay: str, surface: str, cond: str) -> pd.Series:
    row = deltas[
        (deltas["assay_family"] == assay)
        & (deltas["surface_id"] == surface)
        & (deltas["condition_id"] == cond)
    ]
    if row.empty:
        raise KeyError((assay, surface, cond))
    return row.iloc[0]


def _short_run_id(run_id: str) -> str:
    text = str(run_id)
    if len(text) <= 34:
        return text
    if "_seed" in text:
        prefix, suffix = text.rsplit("_seed", 1)
        return f"{prefix.split('_')[0]}...seed{suffix}"
    return f"{text[:18]}...{text[-12:]}"


def _build_provenance_summary_lines() -> list[str]:
    provenance = pd.read_csv(base.DATA_ROOT / "tables" / "provenance.csv")
    grouped = (
        provenance.groupby(["group_id", "run_id"], sort=True)["specimen_id"]
        .agg(lambda values: ", ".join(sorted(str(value) for value in values)))
        .reset_index()
    )
    lines = ["| Group | Source run | Specimens |", "|---|---|---|"]
    for _, row in grouped.iterrows():
        specimen_ids = str(row["specimen_id"])
        lines.append(
            f"| `{row['group_id']}` | `{_short_run_id(str(row['run_id']))}` | `{specimen_ids}` |"
        )
    return lines


def _episode_count_label(df: pd.DataFrame, *, endpoint: str) -> str:
    if endpoint == "survival":
        hits = int(df["survived"].fillna(False).sum())
    else:
        hits = int(df["success"].fillna(False).sum())
    total = int(len(df))
    return f"{hits}/{total} = {_fmt_pct(hits / total if total else float('nan'))}"


def _failure_comp_label(df: pd.DataFrame, *, endpoint: str) -> str:
    if endpoint == "survival":
        falls = int(df["fell"].fillna(False).sum())
        stalls = int((~df["survived"].fillna(False) & ~df["fell"].fillna(False)).sum())
    else:
        falls = int(df["fell"].fillna(False).sum())
        stalls = int((~df["success"].fillna(False) & ~df["fell"].fillna(False)).sum())
    total = int(len(df))
    fall_rate = _fmt_pct(falls / total if total else float("nan"))
    stall_rate = _fmt_pct(stalls / total if total else float("nan"))
    return f"fall {falls}/{total} = {fall_rate}; stall {stalls}/{total} = {stall_rate}"


def _s4_subset(
    eps: pd.DataFrame,
    *,
    assay_family: str,
    surface_id: str,
    condition_id: str,
    distance_pair: str | None = None,
) -> pd.DataFrame:
    df = eps[
        (eps["specimen_id"] == base.S4_SPECIMEN)
        & (eps["assay_family"] == assay_family)
        & (eps["surface_id"] == surface_id)
        & (eps["condition_id"] == condition_id)
    ].copy()
    if distance_pair is not None:
        df = df[df["distance_pair"] == distance_pair]
    if df.empty:
        raise KeyError((assay_family, surface_id, condition_id, distance_pair))
    return df


def _build_s4_profile_lines(eps: pd.DataFrame) -> list[str]:
    specs = [
        ("Native intact", "signaling_ablation", "native_core", "intact", None, "success"),
        ("Extended intact `(8,4)`", "signaling_ablation", "long_distance_generalization", "intact", "(8,4)", "success"),
        ("Extended intact `(10,5)`", "signaling_ablation", "long_distance_generalization", "intact", "(10,5)", "success"),
        ("Double-cliff intact", "signaling_ablation", "double_cliff", "intact", None, "survival"),
        ("Native deformation off", "deformation_ablation", "native_core", "deformation_off", None, "success"),
        ("Native Sobel-X off", "signaling_ablation", "native_core", "sobel_x_off", None, "success"),
        ("Native identity off", "signaling_ablation", "native_core", "identity_comm_off", None, "success"),
        ("Native all comm off", "signaling_ablation", "native_core", "all_comm_off", None, "success"),
    ]
    lines = ["| Assay | Endpoint | Failure composition |", "|---|---:|---|"]
    for label, assay, surface, condition, pair, endpoint in specs:
        df = _s4_subset(
            eps,
            assay_family=assay,
            surface_id=surface,
            condition_id=condition,
            distance_pair=pair,
        )
        endpoint_label = "survival" if endpoint == "survival" else "success"
        lines.append(
            f"| {label} | {endpoint_label}: `{_episode_count_label(df, endpoint=endpoint)}` | {_failure_comp_label(df, endpoint=endpoint)} |"
        )
    return lines


def _chart_list(paths: list[Path], packet_root: Path) -> str:
    return "\n".join(f"- `{path.relative_to(packet_root)}`" for path in paths)


def _write_readme(
    packet_root: Path,
    *,
    original_paths: list[Path],
    copied_paths: list[Path],
    table_paths: dict[str, Path],
    tau_payload: dict[str, object],
) -> Path:
    text = f"""# Sensing at the Edges - 8 Specimen Analysis Packet

Created: 2026-06-13

This directory is the curated packet for the new eight-specimen Sensing at the Edges analysis. It supersedes the earlier manuscript-revision materials for this purpose, but it does not edit the manuscript directly.

## What This Packet Is

- Data source: `{base.DATA_ROOT}`
- Primary manifest: `{base.DATA_ROOT / "tables" / "episode_manifest.csv"}`
- Cohort: `n=8` selected specimens from multiple source runs.
- Statistical unit: specimen-level rates. Deterministic episodes are terrain/configuration evaluations, not independent stochastic replicates.
- Reporting convention: mean +/- SEM across specimens, with raw episode counts included where useful.
- Double-cliff is reported as stress survival, not a goal-completion endpoint.

## Preferred Paper-Facing Charts

{_chart_list(original_paths, packet_root)}

These charts follow the original paper visual style: grouped outcome bars, `Rate (%)` axes, black SEM whiskers, and the success/fall/stall palette used in the previous figures. Visible chart text is intentionally minimal because the manuscript captions should carry method details.

## Additional Copied Outputs

All prior generated outputs are kept for reference in organized folders:

- `figures/composite/`
- `figures/main_bars/`
- `figures/simple/`
- `figures/tau_previous/`
- `tables/`

Copied output files: `{len(copied_paths)}`.

## Core Tables

{_chart_list(list(table_paths.values()), packet_root)}

Tau summary: allowed range `{tau_payload["allowed_tau_min"]:.1f}-{tau_payload["allowed_tau_max"]:.1f}`, observed range `{tau_payload["observed_min"]:.3f}-{tau_payload["observed_max"]:.3f}`, median `{tau_payload["observed_median"]:.3f}`, mean `{tau_payload["observed_mean"]:.3f}`.

## Main Report

See `analysis_report_8_specimens.md` for chart descriptions, statistics, interpretation notes, and manuscript-safe wording.
"""
    path = packet_root / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def _write_report(
    packet_root: Path,
    *,
    eps: pd.DataFrame,
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    pairs: pd.DataFrame,
    tau_payload: dict[str, object],
    original_paths: list[Path],
    s4_signal_auc_payload: dict[str, float],
) -> Path:
    native = _summary_row(summary, "signaling_ablation", "native_core", "intact")
    extended = _summary_row(summary, "signaling_ablation", "long_distance_generalization", "intact")
    double = _summary_row(summary, "signaling_ablation", "double_cliff", "intact")
    provenance_lines = _build_provenance_summary_lines()
    s4_profile_lines = _build_s4_profile_lines(eps)

    pair_order = {"(2,1)": 0, "(4,2)": 1, "(6,3)": 2, "(8,4)": 3, "(10,5)": 4}
    pairs = pairs.assign(_order=pairs["distance_pair"].map(pair_order).fillna(99)).sort_values("_order")
    pair_lines = [
        f"| {row['surface_id']} | {row['distance_pair']} | {int(row['count'])}/{int(row['total'])} = {_fmt_pct(row['episode_rate'])} | {_fmt_pct(row['mean_specimen_rate'])} +/- {_fmt_pct(row['sem_specimen_rate'])} |"
        for _, row in pairs.iterrows()
    ]

    signaling_lines = []
    for cond in base.SIGNALING_ORDER:
        row = _summary_row(summary, "signaling_ablation", "native_core", cond)
        delta = _delta_row(deltas, "signaling_ablation", "native_core", cond)
        signaling_lines.append(
            f"| {base.SIGNALING_LABELS[cond]} | {int(row['count'])}/{int(row['total'])} = {_fmt_pct(row['episode_rate'])} | {_fmt_pct(row['fall_episode_rate'])} | {_fmt_pct(row['stall_episode_rate'])} | {_fmt_pct(row['mean_specimen_rate'])} +/- {_fmt_pct(row['sem_specimen_rate'])} | {_fmt_pp(delta['mean_delta'])} | {delta['wilcoxon_p_two_sided']:.4f} |"
        )

    deformation_lines = []
    for cond in base.DEFORMATION_ORDER:
        row = _summary_row(summary, "deformation_ablation", "native_core", cond)
        delta = _delta_row(deltas, "deformation_ablation", "native_core", cond)
        deformation_lines.append(
            f"| {base.DEFORMATION_LABELS[cond]} | {int(row['count'])}/{int(row['total'])} = {_fmt_pct(row['episode_rate'])} | {_fmt_pct(row['fall_episode_rate'])} | {_fmt_pct(row['stall_episode_rate'])} | {_fmt_pct(row['mean_specimen_rate'])} +/- {_fmt_pct(row['sem_specimen_rate'])} | {_fmt_pp(delta['mean_delta'])} | {delta['wilcoxon_p_two_sided']:.4f} |"
        )

    body_transfer_rows = []
    for cond, label in [
        ("rect_7x2", "7x2 native"),
        ("rect_6x2", "6x2"),
        ("rect_8x2", "8x2"),
        ("rect_7x3", "7x3"),
    ]:
        row = _summary_row(summary, "body_transfer", "native_core", cond)
        body_transfer_rows.append(
            f"| {label} | {int(row['count'])}/{int(row['total'])} = {_fmt_pct(row['episode_rate'])} | {_fmt_pct(row['fall_episode_rate'])} | {_fmt_pct(row['stall_episode_rate'])} | {_fmt_pct(row['mean_specimen_rate'])} +/- {_fmt_pct(row['sem_specimen_rate'])} |"
        )

    frozen_lines = []
    for cond, label in [
        ("deformation_frozen_start", "Deformation frozen"),
        ("perception_frozen_start", "Perception frozen"),
    ]:
        row = _summary_row(summary, "frozen_sensing_control", "native_core", cond)
        delta = _delta_row(deltas, "frozen_sensing_control", "native_core", cond)
        frozen_lines.append(
            f"| {label} | {int(row['count'])}/{int(row['total'])} = {_fmt_pct(row['episode_rate'])} | {_fmt_pct(row['fall_episode_rate'])} | {_fmt_pct(row['stall_episode_rate'])} | {_fmt_pct(row['mean_specimen_rate'])} +/- {_fmt_pct(row['sem_specimen_rate'])} | {_fmt_pp(delta['mean_delta'])} |"
        )

    joint_lines = []
    for cond, label in [
        ("deformation_off__sobel_x_off", "Def. off + Sobel-X off"),
        ("deformation_off__sobel_y_off", "Def. off + Sobel-Y off"),
        ("deformation_off__laplacian_off", "Def. off + Laplacian off"),
        ("deformation_off__identity_comm_off", "Def. off + Identity off"),
        ("deformation_off__all_comm_off", "Def. off + All comm off"),
    ]:
        row = _summary_row(summary, "combined_ablation_quick", "native_core", cond)
        joint_lines.append(
            f"| {label} | {int(row['count'])}/{int(row['total'])} = {_fmt_pct(row['episode_rate'])} | {_fmt_pct(row['fall_episode_rate'])} | {_fmt_pct(row['stall_episode_rate'])} |"
        )

    with_without = base._build_with_without_s4_summary(eps)

    def _ww_cell(row: pd.Series | None) -> str:
        if row is None:
            return "not applicable"
        return (
            f"{int(row['positive_count'])}/{int(row['episode_count'])} = {_fmt_pct(row['episode_rate'])}; "
            f"mean {_fmt_pct(row['mean_specimen_rate'])} +/- {_fmt_pct(row['sem_specimen_rate'])}"
        )

    def _ww_line(label: str, assay: str, surface: str, cond: str) -> str:
        subset = with_without[
            (with_without["assay_family"] == assay)
            & (with_without["surface_id"] == surface)
            & (with_without["condition_id"] == cond)
        ]
        all_rows = subset[subset["group"] == "all_8"]
        excluding_rows = subset[subset["group"] == "excluding_s4"]
        all_row = all_rows.iloc[0] if not all_rows.empty else None
        excluding_row = excluding_rows.iloc[0] if not excluding_rows.empty else None
        return f"| {label} | {_ww_cell(all_row)} | {_ww_cell(excluding_row)} |"

    with_without_lines = [
        _ww_line("Native intact", "signaling_ablation", "native_core", "intact"),
        _ww_line("Extended intact", "signaling_ablation", "long_distance_generalization", "intact"),
        _ww_line("Double-cliff intact survival", "signaling_ablation", "double_cliff", "intact"),
        _ww_line("Native deformation off", "deformation_ablation", "native_core", "deformation_off"),
        _ww_line("Native Sobel-X off", "signaling_ablation", "native_core", "sobel_x_off"),
        _ww_line("Native all comm off", "signaling_ablation", "native_core", "all_comm_off"),
        _ww_line("Deformation frozen", "frozen_sensing_control", "native_core", "deformation_frozen_start"),
        _ww_line("Perception frozen", "frozen_sensing_control", "native_core", "perception_frozen_start"),
    ]

    s4_mechanism_dir = base.DATA_ROOT / "figures" / "s4_mechanism_packet"
    s4_mechanism_lines: list[str] = [
        f"- Detailed mechanism packet: `{s4_mechanism_dir / 'mechanism_packet_summary.md'}`",
        f"- Summary JSON: `{s4_mechanism_dir / 'summary.json'}`",
    ]
    timing_path = s4_mechanism_dir / "s4_edge_timing_summary.csv"
    paired_timing_path = s4_mechanism_dir / "s4_edge_timing_paired_differences.csv"
    band_path = s4_mechanism_dir / "s4_band_predictivity_by_window.csv"
    control_path = s4_mechanism_dir / "s4_decoder_controls.csv"
    onset_path = s4_mechanism_dir / "s4_band_predictivity_onsets.csv"
    onset_regression_path = s4_mechanism_dir / "s4_band_predictivity_onset_regression.csv"
    if timing_path.exists():
        timing_frame = pd.read_csv(timing_path)
        for _, row in timing_frame.iterrows():
            s4_mechanism_lines.append(
                f"- `{row['condition']}` timing: {int(row['true_cliff_event_count'])}/{int(row['episode_count'])} "
                f"native episodes have true cliff-frontier events; mean event-to-goal-side commitment "
                f"{row['mean_event_to_goal_commit_steps_true_events']:.1f} steps; mean max overhang "
                f"{row['mean_max_overhang_before_goal_commit_true_events']:.3f}; mean post-event direction changes "
                f"{row['mean_direction_changes_after_event_true_events']:.2f}."
            )
    if paired_timing_path.exists():
        paired_frame = pd.read_csv(paired_timing_path)
        for _, row in paired_frame.iterrows():
            s4_mechanism_lines.append(
                f"- Paired `{row['distance_pair']}` true-edge difference, deformation-off minus intact: "
                f"commitment latency {row['event_to_goal_commit_steps_deformation_off_minus_intact']:.1f} steps, "
                f"overhang {row['max_overhang_before_goal_commit_deformation_off_minus_intact']:.3f}, "
                f"direction changes {row['direction_changes_after_event_deformation_off_minus_intact']:.1f}."
            )
    if band_path.exists():
        band_frame = pd.read_csv(band_path)
        for stream in ["body", "signal"]:
            for band_name in ["frontier", "middle", "rear"]:
                subset = band_frame[
                    (band_frame["window"] == "edge_impulse")
                    & (band_frame["lag"] == 8)
                    & (band_frame["stream"] == stream)
                    & (band_frame["band"] == band_name)
                ]
                if subset.empty:
                    continue
                row = subset.iloc[0]
                s4_mechanism_lines.append(
                    f"- Edge-impulse `{stream}` `{band_name}` lag 8: AUC {row['mean_auc']:.3f}, "
                    f"balanced accuracy {row['mean_balanced_accuracy']:.3f}, "
                    f"actual-minus-null AUC {row['mean_actual_minus_null_auc']:.3f}."
                )
    if control_path.exists():
        controls = pd.read_csv(control_path)
        for _, row in controls.iterrows():
            if row["control_status"] == "estimated":
                s4_mechanism_lines.append(
                    f"- Control `{row['stream']}` `{row['window']}` frontier lag 8: estimated AUC "
                    f"{row['mean_auc']:.3f}; actual-minus-null AUC {row['mean_actual_minus_null_auc']:.3f}."
                )
            else:
                s4_mechanism_lines.append(
                    f"- Control `{row['stream']}` `{row['window']}` frontier lag 8: not estimable "
                    f"({row['dominant_skip_reason']})."
                )
    if onset_path.exists():
        onsets = pd.read_csv(onset_path)
        for stream in ["body", "signal"]:
            subset = onsets[onsets["stream"] == stream]
            if subset.empty:
                continue
            pieces = []
            for _, row in subset.iterrows():
                if row["status"] == "threshold_crossing":
                    pieces.append(
                        f"{row['band']} at `{row['onset_window']}` "
                        f"(center {row['onset_relative_center']:.1f}, AUC {row['onset_auc']:.3f})"
                    )
                else:
                    pieces.append(f"{row['band']} {row['status']}")
            s4_mechanism_lines.append(f"- Onset audit `{stream}`: " + "; ".join(pieces) + ".")
    if onset_regression_path.exists():
        onset_regression = pd.read_csv(onset_regression_path)
        for _, row in onset_regression.iterrows():
            if row["status"] == "estimated":
                r2 = row.get("r2", np.nan)
                r2_text = "NA" if not np.isfinite(r2) else f"{r2:.3f}"
                s4_mechanism_lines.append(
                    f"- Onset slope `{row['stream']}`: {row['slope_steps_per_band_position']:.1f} "
                    f"relative steps per band position across {int(row['n_bands'])} bands; R2 {r2_text}."
                )
            else:
                s4_mechanism_lines.append(
                    f"- Onset slope `{row['stream']}`: not estimable ({row.get('reason', 'insufficient threshold crossings')})."
                )

    text = f"""# Analysis Report - New 8 Specimen Sensing at the Edges Packet

This report describes the curated figure and table packet in this directory. The analysis is based on the new eight selected specimens in `{base.DATA_ROOT.name}`.

## Data And Statistical Unit

- Primary data: `{base.DATA_ROOT / "tables" / "episode_manifest.csv"}`
- Provenance table: `{base.DATA_ROOT / "tables" / "provenance.csv"}`
- Cohort: `n=8` selected specimens, not eight independent evolutionary runs.
- Unit of aggregation: specimen-level rates. Episodes are deterministic terrain/configuration evaluations.
- Error bars: SEM across specimens.
- Tests: paired within-specimen deltas use Wilcoxon/sign-test values as secondary checks.
- Bootstrap 95% confidence intervals are included in `eight_specimen_endpoint_summary.csv`, `eight_specimen_paired_deltas.csv`, and `eight_specimen_with_without_s4_summary.csv`.
- Double-cliff values are survival/stress-survival endpoints.

## Cohort Provenance

The eight specimens are selected controllers from multiple source runs, not eight independent evolutionary runs. Treat the specimen as the unit of descriptive analysis and avoid claims that require run-level independence.

{chr(10).join(provenance_lines)}

## Preferred Charts

{_chart_list(original_paths, packet_root)}

## Paper-Facing Versus Reference Outputs

Use the six preferred charts above as the paper-facing figure pool. Other copied outputs in `figures/composite/`, `figures/main_bars/`, `figures/simple/`, `figures/tau_previous/`, and extra files under `figures/s4_predictivity/` are audit/reference material. In particular, `s4_body_signal_predictivity_turn_heatmaps` is the older Pearson future-displacement view and should not be used as the main S4 predictivity panel; `s4_signal_body_predictivity_stacked` is the matched AUC-vs-AUC paper-facing chart.

## Figure Caption Guidance

The preferred charts are caption-driven: visible plot text should be limited to compact panel headers, axis labels, legends, and colorbar labels. Do not put method-heavy strings such as `edge impulse`, `lag 8`, `deformation-off replay`, `n=8`, or `mean held-out AUC` inside the plot area unless a caption will not accompany the figure.

Suggested caption content:

- `performance_by_surface_original_style`: Baseline outcomes across the eight selected specimens. Bars show specimen-level mean +/- SEM for success or survival, falls, and stalls; deterministic episodes are terrain/configuration evaluations. Double-cliff is a survival/stress endpoint, not goal-completion success.
- `ablations_original_style`: Native-core signalling and deformation ablation outcomes. Bars show specimen-level mean +/- SEM; the current deformation set is intact, absolute only, compliance only, and deformation off.
- `body_transfer_original_style`: Native-task body-transfer outcomes for the same controllers. Use as inline support for the body-as-sensory-interface claim rather than as a main standalone result if space is tight.
- `s4_profile_original_style`: S4 endpoint profile across current assay families. Use to show why S4 is a mechanistic case study: robust under deformation-off replay and double-cliff stress, but sensitive to signalling removals.
- `tau_distribution_original_style`: Evolved CTRNN time constants across the allowed range; bars are mean hidden-unit counts per tau bin per specimen, with SEM across specimens.
- `s4_signal_body_predictivity_stacked`: S4 deformation-off replay, edge-impulse window `[-32,+96]`, lag `8`. Cell-wise held-out AUC for decoding committed edge outcome from hidden/internal signalling state versus body state. The grid follows the physical 7x2 cell layout: upper row cells 7-13 and lower row cells 0-6. The red outline marks the leading-edge/frontier-band cells at the cliff-edge event window, i.e. the anatomical region whose state is evaluated as the body reaches/releases over the edge.

Ready-to-paste caption drafts:

- **Performance:** Baseline outcomes for the eight selected specimens. Bars show specimen-level mean +/- SEM across deterministic terrain episodes. Native and extended panels report goal-cliff success, falls, and stalls; the double-cliff panel reports survival, falls, and stalls under a symmetric stress endpoint.
- **Ablations:** Native-core outcome composition under signalling and deformation ablations. Bars show specimen-level mean +/- SEM; ablations are paired to the intact native condition at the specimen level.
- **Body transfer:** Native-task outcomes after transferring the same controllers into altered body geometries. Reduced success and increased falls support the interpretation that the body is a calibrated sensory interface rather than a passive output device.
- **S4 profile:** Endpoint profile for specimen S4 across intact, deformation-off, and signalling-ablation assays. S4 is robust under deformation-off replay and double-cliff stress but collapses under key signalling removals, motivating a mechanistic case study rather than a population-wide mechanism claim.
- **Tau distribution:** Distribution of evolved CTRNN time constants across the eight selected specimens. Bars show mean hidden-unit counts per tau bin per specimen; whiskers are SEM across specimens.
- **S4 predictivity:** S4 deformation-off replay. Cell-wise leave-one-trial-out AUC for predicting committed edge outcome from hidden/internal signalling state versus body state in the edge-impulse window `[-32,+96]`, lag `8`. The red outline marks the leading-edge/frontier-band cells at the cliff-edge event window.

### `performance_by_surface_original_style`

This chart keeps the original paper grammar: grouped bars for success/survival, fall, and stall, with SEM across specimens. Panel A shows native distance pairs, Panel B shows extended distance pairs, and Panel C shows double-cliff stress survival by width. The 50% mirrored-goal intuition applies only to native/extended goal-cliff tasks, not to double-cliff survival.

Native intact success is `{int(native['count'])}/{int(native['total'])} = {_fmt_pct(native['episode_rate'])}` with mean specimen success `{_fmt_pct(native['mean_specimen_rate'])} +/- {_fmt_pct(native['sem_specimen_rate'])}`. Extended intact success is `{int(extended['count'])}/{int(extended['total'])} = {_fmt_pct(extended['episode_rate'])}` with mean specimen success `{_fmt_pct(extended['mean_specimen_rate'])} +/- {_fmt_pct(extended['sem_specimen_rate'])}`. Double-cliff intact survival is `{int(double['count'])}/{int(double['total'])} = {_fmt_pct(double['episode_rate'])}` with mean specimen survival `{_fmt_pct(double['mean_specimen_rate'])} +/- {_fmt_pct(double['sem_specimen_rate'])}`.

| Surface | Pair | Episode success | Mean specimen success +/- SEM |
|---|---:|---:|---:|
{chr(10).join(pair_lines)}

### `ablations_original_style`

This chart shows native-core signaling and deformation ablations as outcome composition, not only success loss. It includes only the current new-manifest deformation conditions: intact, absolute only, compliance only, and deformation off. Legacy deformation conditions from older manifests are intentionally excluded from the preferred chart set.

Native signaling ablations:

| Condition | Success | Fall | Stall | Mean success +/- SEM | Paired delta | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(signaling_lines)}

Native deformation ablations:

| Condition | Success | Fall | Stall | Mean success +/- SEM | Paired delta | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(deformation_lines)}

### Frozen-Sensing Native Control

This follow-up control freezes test-time sensory input while keeping recurrent controller dynamics and actions intact. `Deformation frozen` holds episode-start deformation/sensory channels fixed. `Perception frozen` holds the full episode-start CTRNN perception tensor fixed, making it the harsher supplement. Both are native-core deterministic replays across all eight specimens and are paired to the intact native baseline in `eight_specimen_paired_deltas.csv`.

| Condition | Success | Fall | Stall | Mean success +/- SEM | Paired delta vs intact |
|---|---:|---:|---:|---:|---:|
{chr(10).join(frozen_lines)}

Paper-safe reading: freezing test-time sensing collapses native success well below intact performance. This is a direct replay control showing that ongoing contact/body-derived sensing is load-bearing; it does not require any new evolutionary run claim.

### `body_transfer_original_style`

This chart supports the body-as-sensory-interface framing. Locomotor activity partly transfers, but altered bodies sharply reduce native success and often increase falls.

| Body | Success | Fall | Stall | Mean success +/- SEM |
|---|---:|---:|---:|---:|
{chr(10).join(body_transfer_rows)}

### `s4_profile_original_style`

This chart shows S4 across current assay families only. It keeps S4 as a case study rather than making it the population-level result. S4 remains perfect on native intact, extended intact, double-cliff intact, and native deformation-off replay, but collapses under Sobel-X, identity-communication, and all-communication ablations.

| Assay | Endpoint | Failure composition |
|---|---:|---|
{chr(10).join(s4_profile_lines[2:])}

### S4 Joint Deformation-Off And Signalling Ablation Surface

This S4-only follow-up crosses deformation-off replay with each signalling removal. It is a case-study assay, not an eight-specimen population result.

| Condition | Success | Fall | Stall |
|---|---:|---:|---:|
{chr(10).join(joint_lines)}

Paper-safe reading: S4 remains deformation-off robust when signalling is intact, but the joint surface shows that some signalling cuts still collapse or destabilize the regime. Sobel-X off and all-communication off produce 0/6 success with stalls; Sobel-Y off preserves 4/6 success; identity-off and Laplacian-off introduce falls and partial success. This supports the two-capacities interpretation for S4: the body-state dynamics modulate the return regime, while internal signalling remains necessary to sustain it.

### With/Without-S4 Robustness Checks

These rows keep S4 from silently driving the cohort interpretation. Use them as robustness notes rather than separate primary claims.

| Metric | All 8 | Excluding S4 |
|---|---:|---:|
{chr(10).join(with_without_lines)}

### New Follow-Up Run Provenance

The added controls were run on Modal and then merged into `tables/episode_manifest.csv` by `preprint/scripts/update_eight_specimen_followups.py`. The original manifest was backed up as `tables/episode_manifest.pre_followups_20260613.csv`.

- Frozen-sensing control raw pull: `{base.DATA_ROOT / 'raw_modal_frozen_sensing_control_20260613T041500'}`
- S4 joint-ablation raw pull: `{base.DATA_ROOT / 'raw_modal_s4_joint_ablation_20260613T041600'}`
- Follow-up summary table: `{base.DATA_ROOT / 'tables' / 'followup_control_summary.csv'}`
- Final manifest row counts: 1,358 rows total, with 96 `frozen_sensing_control` rows and 30 `combined_ablation_quick` rows.

### `tau_distribution_original_style`

This chart shows evolved CTRNN time constants across the allowed range. Each specimen contributes one binned 12-unit profile; bars are mean hidden-unit count per bin per specimen, and whiskers are SEM across specimens. The allowed range is `{tau_payload['allowed_tau_min']:.1f}-{tau_payload['allowed_tau_max']:.1f}`. Observed values span `{tau_payload['observed_min']:.3f}-{tau_payload['observed_max']:.3f}`, with median `{tau_payload['observed_median']:.3f}` and mean `{tau_payload['observed_mean']:.3f}`.

### `s4_signal_body_predictivity_stacked`

This is the preferred S4 mechanistic heatmap. It places the event-centered signalling map above the body/frontier-band map so the contrast is visible in one figure. Both panels use the same 7x2 body layout, the same marked frontier band, and the same AUC scale. The visible cells are numbered in physical layout: upper row cells 7-13, lower row cells 0-6. Row labels are intentionally omitted from the plot to keep the paper panel compact.

Panel A shows a new per-cell signalling-state direction decoder computed from the same saved edge-trace deformation-off replays. It uses hidden NCA channels plus CTRNN carry only, excluding deformation channels, action outputs, body geometry, COM, and cliff geometry. The metric is mean held-out AUC in the edge-impulse window `[-32,+96]`, at lag `8`. The strongest signalling cell is cell `{int(s4_signal_auc_payload['top_signal_cell'])}` with AUC `{s4_signal_auc_payload['top_signal_auc']:.3f}`; frontier-band signalling AUCs are cell 7 `{s4_signal_auc_payload['frontier_signal_cell_7_auc']:.3f}`, cell 8 `{s4_signal_auc_payload['frontier_signal_cell_8_auc']:.3f}`, cell 0 `{s4_signal_auc_payload['frontier_signal_cell_0_auc']:.3f}`, and cell 1 `{s4_signal_auc_payload['frontier_signal_cell_1_auc']:.3f}`.

Panel B shows body-state direction predictivity in the edge-impulse window `[-32,+96]` around edge crossing/release, at lag `8`. The metric is mean held-out AUC for predicting whether the specimen returns from the edge or continues away. The red outline encloses the leading-edge/frontier-band cells at the cliff-edge event window: cells 7, 8, 0, and 1. This outline should be described as marking the frontier band at the edge moment, not as an additional measured variable. The frontier band is visible at the leading edge: cell 7 AUC `0.989`, cell 1 AUC `0.981`, cell 0 AUC `0.980`, and cell 8 AUC `0.958`.

This is the figure that supports the claim that predictive body-state structure is already apparent around the crossing/release event whereas the hidden/internal signalling map does not show an equivalent direction-predictive frontier-band pattern. It should be described as leading-edge/frontier-band predictivity, not as single-voxel cell 0 alone.

Reference versions are also generated in `figures/s4_predictivity/`: `s4_frontier_band_auc_edge_impulse_lag8` contains the body AUC panel alone, and `s4_body_signal_predictivity_turn_heatmaps` contains the older Pearson body/signalling comparison. Use the stacked AUC-vs-AUC chart for the paper-facing S4 panel.

#### S4 Heatmap Method Details

- **Event window:** `edge_impulse` means the model uses feature samples whose time index is from `32` steps before to `96` steps after the first cliff-edge reach/release event. This centers the analysis on the moment when the leading/frontier part of the body reaches or releases over the cliff edge.
- **Lag:** `lag 8` means features at time `t` are used to predict the committed edge outcome label at `t + 8` environment steps. The window is applied to the feature time `t`, not the future label time.
- **Outcome labels:** The decoder uses nonzero committed-direction labels derived from subsequent movement: whether the specimen returns from the edge or continues away over the future commitment window. Uncommitted/stationary samples are excluded from the decoder fits.
- **AUC calculation:** For each cell, a ridge decoder is trained in a leave-one-trial-out scheme. Training uses all other included edge-trace deformation-off trials; testing uses the held-out trial. The reported value is mean held-out AUC across valid folds. AUC near `0.5` is chance-level direction decoding; larger values mean the cell state separates return-from-edge versus continue-away samples better.
- **Signalling features:** Panel A uses only hidden/internal carried state: per-cell hidden NCA channels plus CTRNN carry. It excludes deformation channels, action outputs, body geometry, COM, cliff geometry, and explicit drive/body measurements.
- **Body features:** Panel B uses the previously computed voxel reversal decoder summary for per-cell body state in the same edge-impulse window and lag. These features include local body/spring geometry and drive-related body state for each voxel, and are treated as body-state predictors rather than hidden/internal signalling predictors.
- **Red outline:** The red outline is a spatial annotation, not a learned mask or extra measurement. It marks the physical frontier-band cells in the first two body columns at the cliff-edge event: upper cells `7` and `8`, lower cells `0` and `1`. It is drawn identically in both panels so signalling and body predictivity can be compared over the same frontier region.

## S4 Edge-Dynamics Follow-Up Packet

{chr(10).join(s4_mechanism_lines)}

Paper-safe reading: these analyses are observational. They support the claim that S4's deformation-off replay contains stronger body/frontier-band direction predictivity than hidden/internal signalling direction predictivity around edge events. They do not prove that the decoded body stream causes reversal, and the onset audit should be treated as a descriptive check rather than a formal propagation proof.

## Remaining Future Work Notes

- The frozen-sensing native control has been run and supports ongoing sensing as load-bearing at test time.
- The headline tables now include within-specimen paired deltas and bootstrap confidence intervals across specimens.
- S4 space/time, label-shuffle/null, pre-edge control, and onset-latency audit outputs are present in the S4 mechanism packet.
- The S4 joint deformation-off/signalling surface has been run for the five signalling removals and should be treated as a case-study assay.
- A stronger tau/timescale claim still requires a timescale-necessity assay, such as fixed/homogeneous tau evaluation or re-evolution. Without that, keep the tau chart descriptive only.
- A causal clamp/settling intervention would have high scientific value but remains future work. The current packet is correlational and decoder-based.
- Broader interpolation/timing-trap sweeps should remain exploratory unless they are promoted into a preregistered figure. Existing S4-only timing-trap outputs are documented in `figures/s4_mechanism_packet/mechanism_packet_summary.md`.

## Paper-Safe Interpretation

- Eight selected specimens show strong native performance, with perfect success on shorter native pairs and failures concentrated on the hardest native pair.
- Extended generalization is lower than native performance but present and heterogeneous.
- Double-cliff is a harsh survival/stress endpoint; S4 is uniquely robust across all widths.
- Native signaling and deformation ablations both impair success, with distinct fall/stall profiles.
- S4 is a mechanistic case study: it is deformation-off robust but signaling-sensitive.
- In S4 deformation-off replay, leading-edge/frontier-band body state predicts committed direction better than hidden/internal signaling state.
- Do not claim the body causes reversal, that signaling does not matter, or that S4 explains every specimen.
"""
    path = packet_root / "analysis_report_8_specimens.md"
    path.write_text(text, encoding="utf-8")
    return path


def _write_manifest(
    packet_root: Path,
    *,
    original_paths: list[Path],
    copied_paths: list[Path],
    table_paths: dict[str, Path],
    eps: pd.DataFrame,
) -> Path:
    provenance = pd.read_csv(base.DATA_ROOT / "tables" / "provenance.csv")
    payload = {
        "schema_version": 1,
        "kind": "sensing_edges_8_specimen_analysis_packet",
        "created": "2026-06-13",
        "packet_root": str(packet_root),
        "source_data_root": str(base.DATA_ROOT),
        "episode_manifest": str(base.DATA_ROOT / "tables" / "episode_manifest.csv"),
        "provenance_csv": str(base.DATA_ROOT / "tables" / "provenance.csv"),
        "n_specimens": int(eps["specimen_id"].nunique()),
        "specimen_ids": sorted(eps["specimen_id"].dropna().unique().tolist()),
        "source_run_ids": sorted(provenance["run_id"].dropna().unique().tolist()),
        "preferred_charts": [str(path.relative_to(packet_root)) for path in original_paths],
        "tables": {key: str(path.relative_to(packet_root)) for key, path in table_paths.items()},
        "copied_reference_outputs": [str(path.relative_to(packet_root)) for path in copied_paths],
        "generator_script": "preprint/scripts/make_sensing_edges_clean_packet.py",
    }
    path = packet_root / "metadata" / "packet_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _copy_reference_outputs(packet_root: Path) -> list[Path]:
    copied: list[Path] = []
    copied.extend(_copy_matching(REVISION_ROOT / "figures", packet_root / "figures" / "composite", ("*.pdf", "*.png")))
    copied.extend(_copy_matching(REVISION_ROOT / "bar_figures", packet_root / "figures" / "main_bars", ("*.pdf", "*.png")))
    copied.extend(_copy_matching(REVISION_ROOT / "simple_figures", packet_root / "figures" / "simple", ("*.pdf", "*.png")))
    copied.extend(_copy_matching(REVISION_ROOT / "tau_figures", packet_root / "figures" / "tau_previous", ("*.pdf", "*.png")))
    copied.extend(_copy_matching(REVISION_ROOT, packet_root / "metadata" / "previous_indexes", ("*.md", "*.txt")))
    return copied


def build_packet(packet_root: Path) -> dict[str, object]:
    _set_original_style()
    packet_root.mkdir(parents=True, exist_ok=True)
    figures_dir = packet_root / "figures" / "original_style"
    tables_dir = packet_root / "tables"

    eps = base._load_episode_manifest()
    table_paths = _write_tables(packet_root, eps)

    original_paths = [
        plot_performance_original_style(eps, figures_dir),
        plot_ablations_original_style(eps, figures_dir),
        plot_body_transfer_original_style(eps, figures_dir),
        plot_s4_profile_original_style(eps, figures_dir),
    ]
    tau_path, tau_payload = plot_tau_original_style(figures_dir, tables_dir)
    original_paths.append(tau_path)
    s4_predictivity_dir = packet_root / "figures" / "s4_predictivity"
    s4_path, s4_table_paths, s4_signal_auc_payload = plot_s4_signal_body_predictivity_stacked(
        s4_predictivity_dir,
        tables_dir,
    )
    original_paths.append(s4_path)
    table_paths.update(s4_table_paths)
    reference_paths = [
        plot_s4_frontier_band_auc_heatmap(s4_predictivity_dir),
        plot_s4_predictivity_heatmaps(s4_predictivity_dir),
    ]

    copied_paths = _copy_reference_outputs(packet_root)
    summary = base._build_endpoint_summary(eps)
    deltas, _specimen_deltas = base._build_paired_deltas(eps)
    pairs = base._build_distance_pair_summary(eps)

    readme = _write_readme(
        packet_root,
        original_paths=original_paths,
        copied_paths=copied_paths,
        table_paths=table_paths,
        tau_payload=tau_payload,
    )
    report = _write_report(
        packet_root,
        eps=eps,
        summary=summary,
        deltas=deltas,
        pairs=pairs,
        tau_payload=tau_payload,
        original_paths=original_paths,
        s4_signal_auc_payload=s4_signal_auc_payload,
    )
    manifest = _write_manifest(
        packet_root,
        original_paths=original_paths,
        copied_paths=copied_paths,
        table_paths=table_paths,
        eps=eps,
    )
    return {
        "packet_root": str(packet_root),
        "readme": str(readme),
        "report": str(report),
        "manifest": str(manifest),
        "preferred_charts": [str(path) for path in original_paths],
        "copied_reference_outputs": len(copied_paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PACKET_ROOT)
    args = parser.parse_args()
    payload = build_packet(args.output_root)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
