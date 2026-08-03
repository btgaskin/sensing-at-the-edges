#!/usr/bin/env python3
"""Generate clean bar-first Sensing at the Edges paper charts."""

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
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
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
        ax.grid(axis=grid_axis, color="#E4E4E4", linewidth=0.55)
        ax.set_axisbelow(True)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def _rate_stats(
    eps: pd.DataFrame,
    *,
    assay_family: str,
    surface_id: str,
    condition_id: str,
    endpoint: str,
) -> tuple[float, float, pd.Series]:
    rates = base._condition_rates(
        eps,
        assay_family=assay_family,
        surface_id=surface_id,
        condition_id=condition_id,
        endpoint=endpoint,
    )
    return float(rates.mean()), base._sem(rates.values), rates


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

    fig, ax = plt.subplots(figsize=(3.35, 2.25), constrained_layout=True)
    x = np.arange(len(labels))
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
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean specimen rate")
    ax.set_title("Baseline endpoints")
    ax.legend(
        frameon=False,
        fontsize=6.8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        handlelength=1.1,
        columnspacing=0.9,
    )
    _style_axes(ax)
    return _save(fig, output_dir / "baseline_outcomes_bars.pdf")


def plot_distance_pair_success(eps: pd.DataFrame, output_dir: Path) -> Path:
    df = eps[(eps["assay_family"] == "signaling_ablation") & (eps["condition_id"] == "intact")]
    pair_order = ["(2,1)", "(4,2)", "(6,3)", "(8,4)", "(10,5)"]
    means: list[float] = []
    sems: list[float] = []
    for pair in pair_order:
        rates = df[df["distance_pair"] == pair].groupby("specimen_id")["success"].mean()
        means.append(float(rates.mean()))
        sems.append(base._sem(rates.values))

    fig, ax = plt.subplots(figsize=(3.55, 2.05), constrained_layout=True)
    x = np.arange(len(pair_order))
    ax.bar(x, means, yerr=sems, color=base.C_SUCCESS, edgecolor="white", linewidth=0.7, capsize=3)
    ax.axhline(0.5, color=base.C_NEUTRAL, linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_order)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Success rate")
    ax.set_title("Success by distance pair")
    _style_axes(ax)
    return _save(fig, output_dir / "distance_pair_success_bars.pdf")


def _delta_bar_chart(
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
    means: list[float] = []
    sems: list[float] = []
    for cond in condition_order:
        vals = rows[rows["condition_id"] == cond]["delta"].to_numpy(dtype=float) * 100.0
        means.append(float(np.nanmean(vals)))
        sems.append(base._sem(vals))

    fig_height = 0.34 * len(condition_order) + 0.95
    fig, ax = plt.subplots(figsize=(3.55, fig_height), constrained_layout=True)
    y = np.arange(len(condition_order))
    colors = [base.C_FALL if i == 0 else "#DFA09B" for i in range(len(condition_order))]
    ax.barh(y, means, xerr=sems, color=colors, edgecolor="white", linewidth=0.6, capsize=3)
    for i, mean in enumerate(means):
        x_label = mean - sems[i] - 2.0 if np.isfinite(sems[i]) else mean - 2.0
        ax.text(x_label, i, f"{mean:.1f}", ha="right", va="center", fontsize=6.8, color=base.C_DARK)
    ax.axvline(0, color=base.C_NEUTRAL, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[c] for c in condition_order])
    ax.invert_yaxis()
    ax.set_xlim(-95 if assay_family == "signaling_ablation" else -66, 5)
    ax.set_xlabel("Success delta vs intact (percentage points)")
    ax.set_title(title)
    _style_axes(ax, grid_axis="x")
    return _save(fig, output_path)


def plot_signaling_delta(specimen_deltas: pd.DataFrame, output_dir: Path) -> Path:
    return _delta_bar_chart(
        specimen_deltas,
        assay_family="signaling_ablation",
        surface_id="native_core",
        condition_order=base.SIGNALING_ORDER,
        labels=base.SIGNALING_LABELS,
        title="Native signaling ablation",
        output_path=output_dir / "native_signaling_success_delta_bars.pdf",
    )


def plot_deformation_delta(specimen_deltas: pd.DataFrame, output_dir: Path) -> Path:
    return _delta_bar_chart(
        specimen_deltas,
        assay_family="deformation_ablation",
        surface_id="native_core",
        condition_order=base.DEFORMATION_ORDER,
        labels=base.DEFORMATION_LABELS,
        title="Native deformation ablation",
        output_path=output_dir / "native_deformation_success_delta_bars.pdf",
    )


def _success_rate_bar_chart(
    eps: pd.DataFrame,
    *,
    assay_family: str,
    condition_order: list[str],
    labels: dict[str, str],
    title: str,
    output_path: Path,
) -> Path:
    intact_mean, _intact_sem, _intact_rates = _rate_stats(
        eps,
        assay_family=assay_family,
        surface_id="native_core",
        condition_id="intact",
        endpoint="success",
    )
    means: list[float] = []
    sems: list[float] = []
    for cond in condition_order:
        mean, sem, _rates = _rate_stats(
            eps,
            assay_family=assay_family,
            surface_id="native_core",
            condition_id=cond,
            endpoint="success",
        )
        means.append(mean)
        sems.append(sem)

    fig_width = max(3.25, 0.55 * len(condition_order) + 1.1)
    fig, ax = plt.subplots(figsize=(fig_width, 2.05), constrained_layout=True)
    x = np.arange(len(condition_order))
    ax.bar(x, means, yerr=sems, color=base.C_SUCCESS, edgecolor="white", linewidth=0.7, capsize=3)
    ax.axhline(intact_mean, color=base.C_NEUTRAL, linestyle="--", linewidth=0.9, label="Intact mean")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[c].replace(" ", "\n", 1) for c in condition_order])
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Native success rate")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=6.8, loc="upper right")
    _style_axes(ax)
    return _save(fig, output_path)


def plot_signaling_success_rates(eps: pd.DataFrame, output_dir: Path) -> Path:
    return _success_rate_bar_chart(
        eps,
        assay_family="signaling_ablation",
        condition_order=base.SIGNALING_ORDER,
        labels=base.SIGNALING_LABELS,
        title="Native signaling ablation",
        output_path=output_dir / "native_signaling_success_rate_bars.pdf",
    )


def plot_deformation_success_rates(eps: pd.DataFrame, output_dir: Path) -> Path:
    return _success_rate_bar_chart(
        eps,
        assay_family="deformation_ablation",
        condition_order=base.DEFORMATION_ORDER,
        labels=base.DEFORMATION_LABELS,
        title="Native deformation ablation",
        output_path=output_dir / "native_deformation_success_rate_bars.pdf",
    )


def plot_double_cliff_survival(eps: pd.DataFrame, output_dir: Path) -> Path:
    specs = [
        ("Intact", "signaling_ablation", "intact"),
        ("All comm\noff", "signaling_ablation", "all_comm_off"),
        ("Sobel-X\noff", "signaling_ablation", "sobel_x_off"),
        ("Def.\noff", "deformation_ablation", "deformation_off"),
        ("Compliance\nonly", "deformation_ablation", "compliance_only"),
    ]
    means: list[float] = []
    sems: list[float] = []
    for _label, assay, cond in specs:
        mean, sem, _rates = _rate_stats(eps, assay_family=assay, surface_id="double_cliff", condition_id=cond, endpoint="survived")
        means.append(mean)
        sems.append(sem)

    fig, ax = plt.subplots(figsize=(3.55, 2.05), constrained_layout=True)
    x = np.arange(len(specs))
    ax.bar(x, means, yerr=sems, color=base.C_SUCCESS, edgecolor="white", linewidth=0.7, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _assay, _cond in specs])
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Survival rate")
    ax.set_title("Double-cliff stress survival")
    _style_axes(ax)
    return _save(fig, output_dir / "double_cliff_stress_survival_bars.pdf")


def plot_body_transfer(eps: pd.DataFrame, output_dir: Path) -> Path:
    order = ["rect_7x2", "rect_6x2", "rect_8x2", "rect_7x3"]
    labels = ["7x2", "6x2", "8x2", "7x3"]
    means: list[float] = []
    sems: list[float] = []
    for cond in order:
        mean, sem, _rates = _rate_stats(eps, assay_family="body_transfer", surface_id="native_core", condition_id=cond, endpoint="success")
        means.append(mean)
        sems.append(sem)

    fig, ax = plt.subplots(figsize=(3.2, 2.0), constrained_layout=True)
    x = np.arange(len(order))
    ax.bar(x, means, yerr=sems, color=base.C_SUCCESS, edgecolor="white", linewidth=0.7, capsize=3)
    ax.axhline(0.5, color=base.C_NEUTRAL, linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("Body")
    ax.set_ylabel("Native success rate")
    ax.set_title("Body-transfer success")
    _style_axes(ax)
    return _save(fig, output_dir / "body_transfer_success_bars.pdf")


def plot_s4_profile(eps: pd.DataFrame, output_dir: Path) -> Path:
    scenarios = [
        base.Scenario("Native\nintact", "signaling_ablation", "native_core", "intact", "success"),
        base.Scenario("Extended\nintact", "signaling_ablation", "long_distance_generalization", "intact", "success"),
        base.Scenario("Double\nintact", "signaling_ablation", "double_cliff", "intact", "survived"),
        base.Scenario("Native\ndef. off", "deformation_ablation", "native_core", "deformation_off", "success"),
        base.Scenario("Double\ndef. off", "deformation_ablation", "double_cliff", "deformation_off", "survived"),
        base.Scenario("Sobel-X\noff", "signaling_ablation", "native_core", "sobel_x_off", "success"),
        base.Scenario("All comm\noff", "signaling_ablation", "native_core", "all_comm_off", "success"),
    ]
    cohort_means: list[float] = []
    cohort_sems: list[float] = []
    s4_rates: list[float] = []
    for scenario in scenarios:
        mean, sem, rates = _rate_stats(
            eps,
            assay_family=scenario.assay_family,
            surface_id=scenario.surface_id,
            condition_id=scenario.condition_id,
            endpoint=scenario.endpoint,
        )
        cohort_means.append(mean)
        cohort_sems.append(sem)
        s4_rates.append(float(rates.loc[base.S4_SPECIMEN]))

    fig, ax = plt.subplots(figsize=(3.85, 2.25), constrained_layout=True)
    x = np.arange(len(scenarios))
    width = 0.36
    ax.bar(x - width / 2, cohort_means, yerr=cohort_sems, width=width, color="#B8B8B8", edgecolor="white", linewidth=0.6, capsize=2.5, label="Cohort mean")
    ax.bar(x + width / 2, s4_rates, width=width, color=base.C_ACCENT, edgecolor="white", linewidth=0.6, label="S4")
    ax.set_xticks(x)
    ax.set_xticklabels([scenario.label for scenario in scenarios], rotation=30, ha="right")
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Rate")
    ax.set_title("S4 profile against cohort")
    ax.legend(frameon=False, fontsize=6.8, loc="upper right")
    _style_axes(ax)
    return _save(fig, output_dir / "s4_profile_bars.pdf")


def plot_s4_decoder(output_dir: Path) -> Path:
    labels = ["Body\nlag 0", "Signal\nlag 0", "Body\nlag 8", "Signal\nlag 8"]
    auc = [0.837, 0.586, 0.822, 0.643]
    ba = [0.616, 0.500, 0.571, 0.500]
    fig, ax = plt.subplots(figsize=(3.35, 2.05), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, auc, width=width, color=base.C_ACCENT, label="AUC")
    ax.bar(x + width / 2, ba, width=width, color=base.C_ACCENT_2, label="Balanced acc.")
    ax.axhline(0.5, color=base.C_NEUTRAL, linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("Decoder score")
    ax.set_title("S4 direction decoding")
    ax.legend(frameon=False, fontsize=6.8, loc="upper right")
    _style_axes(ax)
    return _save(fig, output_dir / "s4_body_signal_decoder_bars.pdf")


def plot_frontier_top_cells(output_dir: Path) -> Path:
    voxel_path = base.DATA_ROOT / "figures" / "voxel_reversal_heatmap_deformation_off" / "voxel_decoder_summary.csv"
    voxel = pd.read_csv(voxel_path)
    subset = voxel[(voxel["window"] == "edge_impulse") & (voxel["lag"] == 8)]
    top = subset.sort_values("mean_auc", ascending=False).head(6).copy()
    top["label"] = top["cell_index"].astype(int).map(lambda cell: f"Cell {cell}")

    fig, ax = plt.subplots(figsize=(3.15, 2.0), constrained_layout=True)
    y = np.arange(len(top))
    colors = [base.C_ACCENT if i < 4 else "#8BA0C6" for i in range(len(top))]
    ax.barh(y, top["mean_auc"], color=colors, edgecolor="white", linewidth=0.6)
    for i, value in enumerate(top["mean_auc"]):
        ax.text(float(value) - 0.018, i, f"{float(value):.3f}", ha="right", va="center", fontsize=6.8, color="white")
    ax.axvline(0.5, color=base.C_NEUTRAL, linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(top["label"])
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("AUC")
    ax.set_title("S4 frontier-band voxel AUC")
    _style_axes(ax, grid_axis="x")
    return _save(fig, output_dir / "s4_frontier_top_cell_auc_bars.pdf")


def write_index(output_root: Path, paths: list[Path]) -> Path:
    index = output_root / "bar_chart_index_2026-06-12.md"
    text = "# Sensing at the Edges - Bar Chart Set\n\n"
    text += "Generated from the eight-specimen manifest and saved S4 analysis summaries.\n\n"
    text += "These charts prioritize clean manuscript readability: bars show means, whiskers show SEM across specimens, and specimen-level variability is left to the text/tables.\n\n"
    text += "Style note: colors follow the previous paper figures (success/survival green, fall red, stall grey), with compact single-panel layouts and no inline percentage labels.\n\n"
    text += "## Recommended Text Treatment\n\n"
    text += "- In main figures, use bars plus SEM whiskers for quick interpretation.\n"
    text += "- In text, mention heterogeneity explicitly where it matters: extended generalization, double-cliff survival, and S4's distinctive profile.\n"
    text += "- Put exact per-specimen values and paired deltas in the generated CSV tables rather than in the main figure marks.\n"
    text += "- Do not use the double-cliff bar chart as ordinary success; label it stress survival.\n\n"
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
    out = output_root / "bar_figures"
    out.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_baseline_outcomes(eps, out),
        plot_distance_pair_success(eps, out),
        plot_signaling_success_rates(eps, out),
        plot_deformation_success_rates(eps, out),
        plot_signaling_delta(specimen_deltas, out),
        plot_deformation_delta(specimen_deltas, out),
        plot_s4_profile(eps, out),
        plot_s4_decoder(out),
        plot_frontier_top_cells(out),
        plot_double_cliff_survival(eps, out),
        plot_body_transfer(eps, out),
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
