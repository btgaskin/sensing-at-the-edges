#!/usr/bin/env python3
"""Generate simple single-panel Sensing at the Edges paper charts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import make_sensing_edges_revision_materials as base


DEFAULT_OUTPUT_ROOT = base.DEFAULT_OUTPUT_ROOT


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _style_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#E2E2E2", linewidth=0.55)
        ax.set_axisbelow(True)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def _condition_rates(
    eps: pd.DataFrame,
    *,
    assay_family: str,
    surface_id: str,
    condition_id: str,
    endpoint: str,
) -> pd.Series:
    return base._condition_rates(
        eps,
        assay_family=assay_family,
        surface_id=surface_id,
        condition_id=condition_id,
        endpoint=endpoint,
    )


def plot_baseline_outcomes(eps: pd.DataFrame, output_dir: Path) -> Path:
    labels = ["Native\nsuccess", "Extended\nsuccess", "Double-cliff\nsurvival"]
    specs = [
        ("signaling_ablation", "native_core", "intact", False),
        ("signaling_ablation", "long_distance_generalization", "intact", False),
        ("signaling_ablation", "double_cliff", "intact", True),
    ]
    success_vals: list[float] = []
    fall_vals: list[float] = []
    stall_vals: list[float] = []
    for assay, surface, cond, is_survival in specs:
        success, fall, stall = base._condition_outcomes(
            eps,
            assay_family=assay,
            surface_id=surface,
            condition_id=cond,
        )
        success_vals.append(success)
        fall_vals.append(fall)
        stall_vals.append(0.0 if is_survival else stall)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(3.35, 2.6), constrained_layout=True)
    ax.bar(x, success_vals, color=base.C_SUCCESS, edgecolor="white", linewidth=0.7, label="Success / survival")
    ax.bar(x, fall_vals, bottom=success_vals, color=base.C_FALL, edgecolor="white", linewidth=0.7, label="Fall")
    ax.bar(
        x,
        stall_vals,
        bottom=np.asarray(success_vals) + np.asarray(fall_vals),
        color=base.C_STALL,
        edgecolor="white",
        linewidth=0.7,
        label="Stall",
    )
    for i, value in enumerate(success_vals):
        ax.text(i, max(value / 2, 0.045), f"{100 * value:.0f}%", ha="center", va="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean specimen rate")
    ax.set_title("Baseline endpoints")
    ax.legend(
        frameon=False,
        fontsize=7,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        handlelength=1.1,
        columnspacing=0.9,
    )
    _style_axes(ax)
    return _save(fig, output_dir / "baseline_outcomes.pdf")


def plot_distance_pair_success(eps: pd.DataFrame, output_dir: Path) -> Path:
    base_df = eps[(eps["assay_family"] == "signaling_ablation") & (eps["condition_id"] == "intact")]
    pair_order = ["(2,1)", "(4,2)", "(6,3)", "(8,4)", "(10,5)"]
    fig, ax = plt.subplots(figsize=(3.6, 2.45), constrained_layout=True)
    rng = np.random.default_rng(137)
    means: list[float] = []
    sems: list[float] = []
    for i, pair in enumerate(pair_order):
        rates = base_df[base_df["distance_pair"] == pair].groupby("specimen_id")["success"].mean()
        means.append(float(rates.mean()))
        sems.append(base._sem(rates.values))
        ax.scatter(
            np.full(len(rates), i) + rng.normal(0, 0.035, len(rates)),
            rates,
            s=15,
            color="#595959",
            alpha=0.75,
            linewidth=0,
        )
    ax.errorbar(np.arange(len(pair_order)), means, yerr=sems, fmt="o", color=base.C_ACCENT, capsize=3, markersize=4)
    ax.axhline(0.5, color=base.C_NEUTRAL, linestyle="--", linewidth=0.8)
    ax.text(4.2, 0.515, "50%", color=base.C_NEUTRAL, fontsize=7, va="bottom")
    ax.set_xticks(np.arange(len(pair_order)))
    ax.set_xticklabels(pair_order)
    ax.set_ylim(-0.03, 1.04)
    ax.set_ylabel("Success rate")
    ax.set_title("Success by distance pair")
    _style_axes(ax)
    return _save(fig, output_dir / "distance_pair_success.pdf")


def _delta_chart(
    specimen_deltas: pd.DataFrame,
    *,
    assay_family: str,
    surface_id: str,
    condition_order: list[str],
    labels: dict[str, str],
    title: str,
    output_path: Path,
) -> Path:
    rows = specimen_deltas[
        (specimen_deltas["assay_family"] == assay_family)
        & (specimen_deltas["surface_id"] == surface_id)
        & (specimen_deltas["condition_id"].isin(condition_order))
    ]
    fig_height = 0.46 * len(condition_order) + 1.1
    fig, ax = plt.subplots(figsize=(3.55, fig_height), constrained_layout=True)
    rng = np.random.default_rng(139)
    y = np.arange(len(condition_order))
    for i, cond in enumerate(condition_order):
        vals = rows[rows["condition_id"] == cond]["delta"].to_numpy(dtype=float) * 100.0
        ax.scatter(vals, np.full(len(vals), i) + rng.normal(0, 0.045, len(vals)), s=15, color="#666666", alpha=0.75, linewidth=0)
        ax.errorbar(np.nanmean(vals), i, xerr=base._sem(vals), fmt="o", color=base.C_ACCENT, capsize=3, markersize=4)
    ax.axvline(0, color=base.C_NEUTRAL, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[c] for c in condition_order])
    ax.invert_yaxis()
    ax.set_xlim(-105, 12)
    ax.set_xlabel("Success delta vs intact (percentage points)")
    ax.set_title(title)
    _style_axes(ax, grid_axis="x")
    return _save(fig, output_path)


def plot_signaling_delta(specimen_deltas: pd.DataFrame, output_dir: Path) -> Path:
    return _delta_chart(
        specimen_deltas,
        assay_family="signaling_ablation",
        surface_id="native_core",
        condition_order=base.SIGNALING_ORDER,
        labels=base.SIGNALING_LABELS,
        title="Native signalling ablation",
        output_path=output_dir / "native_signaling_success_delta.pdf",
    )


def plot_deformation_delta(specimen_deltas: pd.DataFrame, output_dir: Path) -> Path:
    return _delta_chart(
        specimen_deltas,
        assay_family="deformation_ablation",
        surface_id="native_core",
        condition_order=base.DEFORMATION_ORDER,
        labels=base.DEFORMATION_LABELS,
        title="Native deformation ablation",
        output_path=output_dir / "native_deformation_success_delta.pdf",
    )


def plot_double_cliff_survival(eps: pd.DataFrame, output_dir: Path) -> Path:
    specs = [
        ("Intact", "signaling_ablation", "intact"),
        ("All comm off", "signaling_ablation", "all_comm_off"),
        ("Sobel-X off", "signaling_ablation", "sobel_x_off"),
        ("Def. off", "deformation_ablation", "deformation_off"),
        ("Compliance only", "deformation_ablation", "compliance_only"),
    ]
    fig, ax = plt.subplots(figsize=(3.6, 2.55), constrained_layout=True)
    rng = np.random.default_rng(140)
    means: list[float] = []
    sems: list[float] = []
    for i, (_label, assay, cond) in enumerate(specs):
        rates = _condition_rates(eps, assay_family=assay, surface_id="double_cliff", condition_id=cond, endpoint="survived")
        means.append(float(rates.mean()))
        sems.append(base._sem(rates.values))
        ax.scatter(
            np.full(len(rates), i) + rng.normal(0, 0.035, len(rates)),
            rates,
            s=15,
            color="#666666",
            alpha=0.7,
            linewidth=0,
        )
    ax.errorbar(np.arange(len(specs)), means, yerr=sems, fmt="o", color=base.C_ACCENT_2, capsize=3, markersize=4)
    ax.set_xticks(np.arange(len(specs)))
    ax.set_xticklabels([label for label, _assay, _cond in specs], rotation=30, ha="right")
    ax.set_ylim(-0.03, 1.04)
    ax.set_ylabel("Survival rate")
    ax.set_title("Double-cliff stress survival")
    _style_axes(ax)
    return _save(fig, output_dir / "double_cliff_stress_survival.pdf")


def plot_body_transfer(eps: pd.DataFrame, output_dir: Path) -> Path:
    order = ["rect_7x2", "rect_6x2", "rect_8x2", "rect_7x3"]
    labels = ["7x2", "6x2", "8x2", "7x3"]
    fig, ax = plt.subplots(figsize=(3.25, 2.4), constrained_layout=True)
    rng = np.random.default_rng(141)
    means: list[float] = []
    sems: list[float] = []
    for i, cond in enumerate(order):
        rates = _condition_rates(eps, assay_family="body_transfer", surface_id="native_core", condition_id=cond, endpoint="success")
        means.append(float(rates.mean()))
        sems.append(base._sem(rates.values))
        ax.scatter(
            np.full(len(rates), i) + rng.normal(0, 0.035, len(rates)),
            rates,
            s=15,
            color="#666666",
            alpha=0.72,
            linewidth=0,
        )
    ax.errorbar(np.arange(len(order)), means, yerr=sems, fmt="o", color=base.C_ACCENT, capsize=3, markersize=4)
    ax.axhline(0.5, color=base.C_NEUTRAL, linestyle="--", linewidth=0.8)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(labels)
    ax.set_ylim(-0.03, 1.04)
    ax.set_xlabel("Body")
    ax.set_ylabel("Native success rate")
    ax.set_title("Body-transfer success")
    _style_axes(ax)
    return _save(fig, output_dir / "body_transfer_success.pdf")


def plot_s4_profile(eps: pd.DataFrame, output_dir: Path) -> Path:
    scenarios = [
        base.Scenario("Native intact", "signaling_ablation", "native_core", "intact", "success"),
        base.Scenario("Extended intact", "signaling_ablation", "long_distance_generalization", "intact", "success"),
        base.Scenario("Double intact", "signaling_ablation", "double_cliff", "intact", "survived"),
        base.Scenario("Native def. off", "deformation_ablation", "native_core", "deformation_off", "success"),
        base.Scenario("Double def. off", "deformation_ablation", "double_cliff", "deformation_off", "survived"),
        base.Scenario("Sobel-X off", "signaling_ablation", "native_core", "sobel_x_off", "success"),
        base.Scenario("All comm off", "signaling_ablation", "native_core", "all_comm_off", "success"),
    ]
    fig, ax = plt.subplots(figsize=(3.75, 2.95), constrained_layout=True)
    rng = np.random.default_rng(142)
    y = np.arange(len(scenarios))
    for i, scenario in enumerate(scenarios):
        rates = _condition_rates(
            eps,
            assay_family=scenario.assay_family,
            surface_id=scenario.surface_id,
            condition_id=scenario.condition_id,
            endpoint=scenario.endpoint,
        )
        ax.scatter(
            rates.to_numpy(dtype=float),
            np.full(len(rates), i) + rng.normal(0, 0.045, len(rates)),
            s=15,
            color="#B7B7B7",
            alpha=0.82,
            linewidth=0,
            zorder=2,
        )
        if base.S4_SPECIMEN in rates.index:
            ax.scatter(float(rates.loc[base.S4_SPECIMEN]), i, s=34, color=base.C_ACCENT, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([s.label for s in scenarios])
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.04)
    ax.set_xlabel("Rate")
    ax.set_title("S4 profile against cohort")
    ax.text(0.98, -0.45, "blue = S4", color=base.C_ACCENT, ha="right", fontsize=7)
    _style_axes(ax, grid_axis="x")
    return _save(fig, output_dir / "s4_profile.pdf")


def plot_s4_decoder(output_dir: Path) -> Path:
    df = pd.DataFrame(
        [
            {"group": "Body\nlag 0", "auc": 0.837, "balanced_accuracy": 0.616},
            {"group": "Signal\nlag 0", "auc": 0.586, "balanced_accuracy": 0.500},
            {"group": "Body\nlag 8", "auc": 0.822, "balanced_accuracy": 0.571},
            {"group": "Signal\nlag 8", "auc": 0.643, "balanced_accuracy": 0.500},
        ]
    )
    fig, ax = plt.subplots(figsize=(3.45, 2.45), constrained_layout=True)
    x = np.arange(len(df))
    width = 0.34
    ax.bar(x - width / 2, df["auc"], width=width, color=base.C_ACCENT, label="AUC")
    ax.bar(x + width / 2, df["balanced_accuracy"], width=width, color=base.C_ACCENT_2, label="Balanced acc.")
    ax.axhline(0.5, color=base.C_NEUTRAL, linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["group"])
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("Decoder score")
    ax.set_title("S4 direction decoding")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    _style_axes(ax)
    return _save(fig, output_dir / "s4_body_signal_decoder.pdf")


def plot_frontier_auc(output_dir: Path) -> Path:
    voxel_path = base.DATA_ROOT / "figures" / "voxel_reversal_heatmap_deformation_off" / "voxel_decoder_summary.csv"
    voxel = pd.read_csv(voxel_path)
    subset = voxel[(voxel["window"] == "edge_impulse") & (voxel["lag"] == 8)]
    grid = np.full((2, 7), np.nan, dtype=float)
    top_cells = set(subset.sort_values("mean_auc", ascending=False)["cell_index"].head(4).astype(int).tolist())
    leading_cells = set(subset[subset["is_event_leading_cell"].astype(bool)]["cell_index"].astype(int).tolist())
    for _, row in subset.iterrows():
        cell = int(row["cell_index"])
        grid[cell // 7, cell % 7] = float(row["mean_auc"])

    fig, ax = plt.subplots(figsize=(3.8, 1.75), constrained_layout=True)
    image = ax.imshow(grid, cmap="viridis", vmin=0.5, vmax=1.0)
    for r in range(2):
        for c in range(7):
            cell = r * 7 + c
            value = grid[r, c]
            ax.text(c, r, f"{cell}\n{value:.2f}", ha="center", va="center", color="white", fontsize=7)
            if cell in top_cells:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=base.C_ACCENT, linewidth=1.6))
            if cell in leading_cells:
                ax.add_patch(plt.Rectangle((c - 0.4, r - 0.4), 0.8, 0.8, fill=False, edgecolor=base.C_FALL, linewidth=1.2))
    ax.set_xticks(range(7))
    ax.set_yticks(range(2))
    ax.set_title("S4 frontier-band voxel AUC")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="AUC")
    return _save(fig, output_dir / "s4_frontier_band_auc.pdf")


def write_index(output_root: Path, paths: list[Path]) -> Path:
    index = output_root / "simple_chart_index_2026-06-12.md"
    text = "# Sensing at the Edges - Simple Chart Set\n\n"
    text += "Generated from the eight-specimen manifest and saved S4 analysis summaries.\n\n"
    text += "These are single-panel, paper-readable charts intended to replace the earlier composite/dashboard figures.\n\n"
    text += "## Charts\n\n"
    for path in paths:
        text += f"- `{path}`\n"
    text += "\nEach PDF also has a PNG preview with the same basename.\n"
    index.write_text(text, encoding="utf-8")
    return index


def generate(output_root: Path) -> list[Path]:
    _set_style()
    eps = base._load_episode_manifest()
    _deltas, specimen_deltas = base._build_paired_deltas(eps)
    out = output_root / "simple_figures"
    out.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_baseline_outcomes(eps, out),
        plot_distance_pair_success(eps, out),
        plot_signaling_delta(specimen_deltas, out),
        plot_deformation_delta(specimen_deltas, out),
        plot_double_cliff_survival(eps, out),
        plot_body_transfer(eps, out),
        plot_s4_profile(eps, out),
        plot_s4_decoder(out),
        plot_frontier_auc(out),
    ]
    index = write_index(output_root, paths)
    print("WROTE", out)
    print("INDEX", index)
    for path in paths:
        print(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    generate(args.output_root)


if __name__ == "__main__":
    main()
