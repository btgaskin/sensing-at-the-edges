#!/usr/bin/env python3
"""Create paper revision materials for the Sensing at the Edges ALife paper.

This script is intentionally a post-run analysis/artifact generator. It reads
the consolidated deterministic assay manifest and saved S4 analysis summaries;
it does not run training or evaluation sweeps.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import textwrap
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# Repo root; the data/paper roots point at local (non-repository) datasets and
# can be overridden via environment variables.
JAX_ROOT = Path(os.environ.get("SENSING_EDGES_ROOT", Path(__file__).resolve().parents[2]))
DATA_ROOT = JAX_ROOT / "paper_data" / "preprint" / "native_core_ablation_v1"
PAPER_ROOT = Path(os.environ.get("SENSING_EDGES_PAPER_ROOT", JAX_ROOT / "paper_workspace"))
DEFAULT_OUTPUT_ROOT = PAPER_ROOT / "revision_materials_2026-06-12"
S4_SPECIMEN = "bin_03_rank_00_cell_6-0-3_flat_0423"

C_SUCCESS = "#8CB87C"
C_FALL = "#D4736C"
C_STALL = "#AAAAAA"
C_ACCENT = "#3B5B92"
C_ACCENT_2 = "#5A4E8C"
C_NEUTRAL = "#777777"
C_LIGHT = "#E8E8E8"
C_DARK = "#222222"


SIGNALING_ORDER = [
    "sobel_x_off",
    "all_comm_off",
    "laplacian_off",
    "sobel_y_off",
    "identity_comm_off",
]
SIGNALING_LABELS = {
    "intact": "Intact",
    "sobel_x_off": "Sobel-X off",
    "sobel_y_off": "Sobel-Y off",
    "laplacian_off": "Laplacian off",
    "identity_comm_off": "Identity off",
    "all_comm_off": "All comm off",
}
DEFORMATION_ORDER = ["deformation_off", "absolute_only", "compliance_only"]
DEFORMATION_LABELS = {
    "intact": "Intact",
    "deformation_off": "Def. off",
    "absolute_only": "Absolute only",
    "compliance_only": "Compliance only",
}
SUMMARY_ASSAYS = [
    "signaling_ablation",
    "deformation_ablation",
    "frozen_sensing_control",
    "combined_ablation_quick",
    "body_transfer",
]


@dataclass(frozen=True)
class Scenario:
    label: str
    assay_family: str
    surface_id: str
    condition_id: str
    endpoint: str


def _bool_series(series: pd.Series) -> pd.Series:
    return series.map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})


def _load_episode_manifest() -> pd.DataFrame:
    path = DATA_ROOT / "tables" / "episode_manifest.csv"
    df = pd.read_csv(path)
    for col in ["success", "fell", "survived", "deathplane_contact", "left_exit"]:
        if col in df.columns:
            df[col] = _bool_series(df[col])
    return df


def _sem(values: np.ndarray | pd.Series | list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1) / math.sqrt(len(arr)))


def _bootstrap_ci(values: np.ndarray | pd.Series, *, n_boot: int = 10000, seed: int = 137) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def _stable_int_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _endpoint_for_surface(surface_id: str) -> str:
    return "survived" if surface_id == "double_cliff" else "success"


def _safe_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.mean())


def _specimen_rates(df: pd.DataFrame, endpoint: str) -> pd.Series:
    return df.groupby("specimen_id", sort=True)[endpoint].mean()


def _summarize_group(df: pd.DataFrame, endpoint: str) -> dict[str, object]:
    rates = _specimen_rates(df, endpoint)
    ci_lo, ci_hi = _bootstrap_ci(rates.values)
    count = int(df[endpoint].sum())
    total = int(len(df))
    return {
        "count": count,
        "total": total,
        "episode_rate": count / total if total else np.nan,
        "n_specimens": int(len(rates)),
        "mean_specimen_rate": float(rates.mean()),
        "sem_specimen_rate": _sem(rates.values),
        "ci95_boot_lo": ci_lo,
        "ci95_boot_hi": ci_hi,
        "specimen_rates": ";".join(f"{idx}:{value:.3f}" for idx, value in rates.items()),
    }


def _build_endpoint_summary(eps: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for assay in SUMMARY_ASSAYS:
        adf = eps[eps["assay_family"] == assay]
        for surface in sorted(adf["surface_id"].dropna().unique()):
            endpoint = _endpoint_for_surface(str(surface))
            sdf = adf[adf["surface_id"] == surface]
            for cond, cdf in sdf.groupby("condition_id", dropna=False, sort=True):
                row = _summarize_group(cdf, endpoint)
                row.update(
                    {
                        "assay_family": assay,
                        "surface_id": surface,
                        "condition_id": cond,
                        "condition_label": str(cdf["condition_label"].dropna().iloc[0])
                        if len(cdf["condition_label"].dropna())
                        else str(cond),
                        "endpoint": endpoint,
                        "success_episode_rate": _safe_rate(cdf["success"]),
                        "fall_episode_rate": _safe_rate(cdf["fell"]),
                        "stall_episode_rate": _safe_rate((~cdf["success"]) & (~cdf["fell"]))
                        if surface != "double_cliff"
                        else np.nan,
                        "survival_episode_rate": _safe_rate(cdf["survived"])
                        if "survived" in cdf.columns
                        else np.nan,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def _native_intact_baseline(eps: pd.DataFrame, endpoint: str) -> tuple[str, pd.Series] | None:
    if endpoint != "success":
        return None
    priority = [
        ("signaling_ablation", "intact"),
        ("deformation_ablation", "intact"),
        ("native_core", None),
    ]
    for assay, condition in priority:
        selector = (eps["assay_family"] == assay) & (eps["surface_id"] == "native_core")
        if condition is not None:
            selector &= eps["condition_id"] == condition
            label = f"{assay}:{condition}"
        else:
            label = assay
        base_df = eps[selector]
        if not base_df.empty and endpoint in base_df.columns:
            return label, base_df.groupby("specimen_id")[endpoint].mean()
    return None


def _build_paired_deltas(eps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    specimen_rows: list[dict[str, object]] = []
    for assay in SUMMARY_ASSAYS:
        adf = eps[eps["assay_family"] == assay]
        for surface in sorted(adf["surface_id"].dropna().unique()):
            endpoint = _endpoint_for_surface(str(surface))
            sdf = adf[adf["surface_id"] == surface]
            if "intact" in set(sdf["condition_id"].dropna()):
                base_cond = "intact"
                base_label = base_cond
                base = sdf[sdf["condition_id"] == base_cond].groupby("specimen_id")[endpoint].mean()
            elif "rect_7x2" in set(sdf["condition_id"].dropna()):
                base_cond = "rect_7x2"
                base_label = base_cond
                base = sdf[sdf["condition_id"] == base_cond].groupby("specimen_id")[endpoint].mean()
            else:
                baseline = _native_intact_baseline(eps, endpoint)
                if baseline is None:
                    continue
                base_label, base = baseline
                base_cond = base_label
            for cond in sorted(c for c in sdf["condition_id"].dropna().unique() if c != base_cond):
                cond_rates = sdf[sdf["condition_id"] == cond].groupby("specimen_id")[endpoint].mean()
                common = base.index.intersection(cond_rates.index)
                if len(common) == 0:
                    continue
                diffs = cond_rates.loc[common].to_numpy(dtype=float) - base.loc[common].to_numpy(dtype=float)
                ci_lo, ci_hi = _bootstrap_ci(
                    diffs,
                    seed=_stable_int_seed(assay, surface, cond, base_label),
                )
                nonzero = diffs[diffs != 0]
                try:
                    wilcoxon_p = (
                        1.0
                        if np.allclose(diffs, 0)
                        else float(stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue)
                    )
                except Exception:
                    wilcoxon_p = float("nan")
                if len(nonzero) == 0:
                    sign_p = 1.0
                else:
                    positive = int(np.sum(nonzero > 0))
                    sign_p = float(stats.binomtest(positive, len(nonzero), 0.5, alternative="two-sided").pvalue)
                summary_rows.append(
                    {
                        "assay_family": assay,
                        "surface_id": surface,
                        "condition_id": cond,
                        "endpoint": endpoint,
                        "baseline_condition": base_label,
                        "n_specimens": int(len(common)),
                        "mean_delta": float(np.mean(diffs)),
                        "sem_delta": _sem(diffs),
                        "ci95_boot_lo": ci_lo,
                        "ci95_boot_hi": ci_hi,
                        "median_delta": float(np.median(diffs)),
                        "positive_deltas": int(np.sum(diffs > 0)),
                        "negative_deltas": int(np.sum(diffs < 0)),
                        "zero_deltas": int(np.sum(diffs == 0)),
                        "wilcoxon_p_two_sided": wilcoxon_p,
                        "sign_test_p_two_sided": sign_p,
                        "baseline_rates": ";".join(f"{k}:{v:.3f}" for k, v in base.loc[common].items()),
                        "condition_rates": ";".join(f"{k}:{v:.3f}" for k, v in cond_rates.loc[common].items()),
                    }
                )
                for specimen, b, c, d in zip(common, base.loc[common], cond_rates.loc[common], diffs):
                    specimen_rows.append(
                        {
                            "assay_family": assay,
                            "surface_id": surface,
                            "condition_id": cond,
                            "endpoint": endpoint,
                            "baseline_condition": base_label,
                            "specimen_id": specimen,
                            "baseline_rate": float(b),
                            "condition_rate": float(c),
                            "delta": float(d),
                        }
                    )
    return pd.DataFrame(summary_rows), pd.DataFrame(specimen_rows)


def _build_with_without_s4_summary(eps: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for assay in SUMMARY_ASSAYS:
        adf = eps[eps["assay_family"] == assay]
        for surface in sorted(adf["surface_id"].dropna().unique()):
            endpoint = _endpoint_for_surface(str(surface))
            sdf = adf[adf["surface_id"] == surface]
            for cond, cdf in sdf.groupby("condition_id", dropna=False, sort=True):
                for group_label, gdf in [
                    ("all_8", cdf),
                    ("excluding_s4", cdf[cdf["specimen_id"].astype(str) != S4_SPECIMEN]),
                ]:
                    if gdf.empty or endpoint not in gdf.columns:
                        continue
                    specimen_rates = gdf.groupby("specimen_id")[endpoint].mean()
                    ci_lo, ci_hi = _bootstrap_ci(
                        specimen_rates.values,
                        seed=_stable_int_seed(assay, surface, cond, group_label),
                    )
                    rows.append(
                        {
                            "assay_family": assay,
                            "surface_id": surface,
                            "condition_id": cond,
                            "condition_label": str(cdf["condition_label"].dropna().iloc[0])
                            if len(cdf["condition_label"].dropna())
                            else str(cond),
                            "endpoint": endpoint,
                            "group": group_label,
                            "n_specimens": int(specimen_rates.size),
                            "episode_count": int(len(gdf)),
                            "positive_count": int(gdf[endpoint].sum()),
                            "episode_rate": float(gdf[endpoint].mean()),
                            "mean_specimen_rate": float(specimen_rates.mean()),
                            "sem_specimen_rate": _sem(specimen_rates.values),
                            "ci95_boot_lo": ci_lo,
                            "ci95_boot_hi": ci_hi,
                            "min_specimen_rate": float(specimen_rates.min()),
                            "max_specimen_rate": float(specimen_rates.max()),
                        }
                    )
    return pd.DataFrame(rows)


def _build_failure_axes(eps: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for specimen in sorted(eps["specimen_id"].dropna().unique()):
        row: dict[str, object] = {"specimen_id": specimen}
        for surface, endpoint, name in [
            ("native_core", "success", "native_intact_success"),
            ("long_distance_generalization", "success", "extended_intact_success"),
            ("double_cliff", "survived", "symmetric_intact_survival"),
        ]:
            df = eps[
                (eps["specimen_id"] == specimen)
                & (eps["assay_family"] == "signaling_ablation")
                & (eps["surface_id"] == surface)
                & (eps["condition_id"] == "intact")
            ]
            row[name] = _safe_rate(df[endpoint]) if len(df) else np.nan
            row[f"{name}_count"] = int(df[endpoint].sum()) if len(df) else 0
            row[f"{name}_n"] = int(len(df))
        for assay, name in [
            ("signaling_ablation", "signaling_goal_ablation_success"),
            ("deformation_ablation", "deformation_goal_ablation_success"),
        ]:
            df = eps[
                (eps["specimen_id"] == specimen)
                & (eps["assay_family"] == assay)
                & (eps["surface_id"].isin(["native_core", "long_distance_generalization"]))
                & (eps["condition_id"] != "intact")
            ]
            row[name] = _safe_rate(df["success"]) if len(df) else np.nan
            row[f"{name}_count"] = int(df["success"].sum()) if len(df) else 0
            row[f"{name}_n"] = int(len(df))
            row[name.replace("success", "fall")] = _safe_rate(df["fell"]) if len(df) else np.nan
        df = eps[
            (eps["specimen_id"] == specimen)
            & (eps["assay_family"] == "deformation_ablation")
            & (eps["surface_id"] == "native_core")
            & (eps["condition_id"] == "deformation_off")
        ]
        row["native_deformation_off_success"] = _safe_rate(df["success"]) if len(df) else np.nan
        df = eps[
            (eps["specimen_id"] == specimen)
            & (eps["assay_family"] == "deformation_ablation")
            & (eps["surface_id"] == "double_cliff")
            & (eps["condition_id"] == "deformation_off")
        ]
        row["symmetric_deformation_off_survival"] = _safe_rate(df["survived"]) if len(df) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _build_distance_pair_summary(eps: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base = eps[(eps["assay_family"] == "signaling_ablation") & (eps["condition_id"] == "intact")]
    for surface in ["native_core", "long_distance_generalization"]:
        sdf = base[base["surface_id"] == surface]
        for pair, pdf in sdf.groupby("distance_pair", sort=True):
            row = _summarize_group(pdf, "success")
            row.update({"surface_id": surface, "distance_pair": pair})
            rows.append(row)
    return pd.DataFrame(rows)


def _condition_rates(
    eps: pd.DataFrame,
    *,
    assay_family: str,
    surface_id: str,
    condition_id: str,
    endpoint: str,
) -> pd.Series:
    df = eps[
        (eps["assay_family"] == assay_family)
        & (eps["surface_id"] == surface_id)
        & (eps["condition_id"] == condition_id)
    ]
    return df.groupby("specimen_id", sort=True)[endpoint].mean()


def _condition_outcomes(
    eps: pd.DataFrame,
    *,
    assay_family: str,
    surface_id: str,
    condition_id: str,
) -> tuple[float, float, float]:
    df = eps[
        (eps["assay_family"] == assay_family)
        & (eps["surface_id"] == surface_id)
        & (eps["condition_id"] == condition_id)
    ]
    if surface_id == "double_cliff":
        survival = df.groupby("specimen_id")["survived"].mean().mean()
        fall = df.groupby("specimen_id")["fell"].mean().mean()
        return float(survival), float(fall), 0.0
    grouped = df.groupby("specimen_id").agg(success=("success", "mean"), fell=("fell", "mean"))
    success = float(grouped["success"].mean())
    fall = float(grouped["fell"].mean())
    stall = float((1.0 - grouped["success"] - grouped["fell"]).mean())
    return success, fall, stall


def _short_specimen_label(specimen_id: str) -> str:
    if specimen_id == S4_SPECIMEN:
        return "S4"
    match = re.search(r"bin_(\d+)_rank_(\d+)_cell_([0-9]+-[0-9]+-[0-9]+)", specimen_id)
    if match:
        bin_id, rank, cell = match.groups()
        return f"b{bin_id}r{rank} {cell}"
    return specimen_id[:12]


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_population_diversity(eps: pd.DataFrame, axes_table: pd.DataFrame, output_dir: Path) -> Path:
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    fig = plt.figure(figsize=(7.2, 6.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], width_ratios=[1.0, 1.15])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # Panel A: baseline outcomes by surface.
    surfaces = [
        ("native_core", "Native\nsuccess"),
        ("long_distance_generalization", "Extended\nsuccess"),
        ("double_cliff", "Double-cliff\nsurvival"),
    ]
    bottoms = np.zeros(len(surfaces))
    outcome_values = {"success": [], "fall": [], "stall": []}
    for surface, _label in surfaces:
        success, fall, stall = _condition_outcomes(
            eps,
            assay_family="signaling_ablation",
            surface_id=surface,
            condition_id="intact",
        )
        outcome_values["success"].append(success)
        outcome_values["fall"].append(fall)
        outcome_values["stall"].append(stall)
    x = np.arange(len(surfaces))
    ax_a.bar(x, outcome_values["success"], color=C_SUCCESS, edgecolor="white", label="Success")
    bottoms += np.asarray(outcome_values["success"])
    ax_a.bar(x, outcome_values["fall"], bottom=bottoms, color=C_FALL, edgecolor="white", label="Fall")
    bottoms += np.asarray(outcome_values["fall"])
    ax_a.bar(x, outcome_values["stall"], bottom=bottoms, color=C_STALL, edgecolor="white", label="Stall")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([label for _surface, label in surfaces])
    ax_a.set_ylim(0, 1)
    ax_a.set_ylabel("Mean specimen rate")
    ax_a.set_title("A  Baseline endpoints")
    ax_a.legend(loc="upper right", frameon=False, fontsize=7, handlelength=1.4)
    _style_axes(ax_a)

    # Panel B: distance-pair performance.
    base = eps[(eps["assay_family"] == "signaling_ablation") & (eps["condition_id"] == "intact")]
    pair_order = ["(2,1)", "(4,2)", "(6,3)", "(8,4)", "(10,5)"]
    pair_labels = ["(2,1)", "(4,2)", "(6,3)", "(8,4)", "(10,5)"]
    rng = np.random.default_rng(137)
    means: list[float] = []
    sems: list[float] = []
    for i, pair in enumerate(pair_order):
        rates = base[base["distance_pair"] == pair].groupby("specimen_id")["success"].mean()
        jitter = rng.normal(0, 0.035, len(rates))
        ax_b.scatter(np.full(len(rates), i) + jitter, rates, s=18, color="#555555", alpha=0.78, linewidth=0)
        means.append(float(rates.mean()))
        sems.append(_sem(rates.values))
    ax_b.errorbar(np.arange(len(pair_order)), means, yerr=sems, fmt="o", color=C_ACCENT, capsize=3, markersize=4)
    ax_b.axhline(0.5, color=C_NEUTRAL, linestyle="--", linewidth=0.8)
    ax_b.text(len(pair_order) - 0.1, 0.515, "50% mirrored baseline", ha="right", va="bottom", fontsize=7, color=C_NEUTRAL)
    ax_b.set_xticks(np.arange(len(pair_order)))
    ax_b.set_xticklabels(pair_labels)
    ax_b.set_ylim(-0.03, 1.03)
    ax_b.set_ylabel("Success rate")
    ax_b.set_title("B  Distance-pair success")
    _style_axes(ax_b)

    # Panel C: failure-axis heatmap.
    heat_cols = [
        ("native_intact_success", "Native\nintact"),
        ("extended_intact_success", "Extended\nintact"),
        ("symmetric_intact_survival", "Double\nsurvival"),
        ("signaling_goal_ablation_success", "Signal\nabl. succ."),
        ("signaling_goal_ablation_fall", "Signal\nabl. fall"),
        ("deformation_goal_ablation_success", "Def.\nabl. succ."),
        ("deformation_goal_ablation_fall", "Def.\nabl. fall"),
        ("native_deformation_off_success", "Native\ndef. off"),
    ]
    axes_table = axes_table.copy().sort_values("specimen_id")
    grid = axes_table[[col for col, _label in heat_cols]].to_numpy(dtype=float)
    image = ax_c.imshow(grid, cmap="cividis", vmin=0, vmax=1, aspect="auto")
    ax_c.set_xticks(np.arange(len(heat_cols)))
    ax_c.set_xticklabels([label for _col, label in heat_cols])
    labels = [_short_specimen_label(v) for v in axes_table["specimen_id"]]
    ax_c.set_yticks(np.arange(len(labels)))
    ax_c.set_yticklabels(labels)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            value = grid[r, c]
            ax_c.text(c, r, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.55 else "#111111", fontsize=6.8)
    s4_rows = np.where(axes_table["specimen_id"].to_numpy() == S4_SPECIMEN)[0]
    if len(s4_rows):
        r = int(s4_rows[0])
        ax_c.add_patch(plt.Rectangle((-0.5, r - 0.5), len(heat_cols), 1, fill=False, edgecolor=C_ACCENT, linewidth=2.0))
    ax_c.set_title("C  Specimen failure axes (rates; S4 outlined)")
    fig.colorbar(image, ax=ax_c, fraction=0.025, pad=0.012, label="Rate")

    path = output_dir / "fig2_population_diversity.pdf"
    _save_figure(fig, path)
    return path


def plot_ablation_cuts(eps: pd.DataFrame, specimen_deltas: pd.DataFrame, output_dir: Path) -> Path:
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    fig = plt.figure(figsize=(7.2, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    rng = np.random.default_rng(138)

    def delta_panel(ax: plt.Axes, assay: str, order: list[str], labels: dict[str, str], title: str) -> None:
        rows = specimen_deltas[
            (specimen_deltas["assay_family"] == assay)
            & (specimen_deltas["surface_id"] == "native_core")
            & (specimen_deltas["condition_id"].isin(order))
        ]
        for i, cond in enumerate(order):
            vals = rows[rows["condition_id"] == cond]["delta"].to_numpy(dtype=float) * 100.0
            jitter = rng.normal(0, 0.035, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=18, color="#555555", alpha=0.75, linewidth=0)
            ax.errorbar(
                [i],
                [np.nanmean(vals)],
                yerr=[_sem(vals)],
                fmt="o",
                color=C_ACCENT,
                capsize=3,
                markersize=4,
            )
        ax.axhline(0, color=C_NEUTRAL, linewidth=0.8)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels([labels[c] for c in order], rotation=30, ha="right")
        ax.set_ylabel("Success delta vs intact (pp)")
        ax.set_title(title)
        ax.set_ylim(-105, 20)
        _style_axes(ax)

    delta_panel(ax_a, "signaling_ablation", SIGNALING_ORDER, SIGNALING_LABELS, "A  Native signalling cuts")
    delta_panel(ax_b, "deformation_ablation", DEFORMATION_ORDER, DEFORMATION_LABELS, "B  Native deformation cuts")

    # Panel C: failure composition for representative native cuts.
    comp_specs = [
        ("Intact", "signaling_ablation", "intact"),
        ("Sobel-X off", "signaling_ablation", "sobel_x_off"),
        ("All comm off", "signaling_ablation", "all_comm_off"),
        ("Def. off", "deformation_ablation", "deformation_off"),
    ]
    x = np.arange(len(comp_specs))
    success_vals: list[float] = []
    fall_vals: list[float] = []
    stall_vals: list[float] = []
    for _label, assay, cond in comp_specs:
        success, fall, stall = _condition_outcomes(eps, assay_family=assay, surface_id="native_core", condition_id=cond)
        success_vals.append(success)
        fall_vals.append(fall)
        stall_vals.append(stall)
    ax_c.bar(x, success_vals, color=C_SUCCESS, edgecolor="white", label="Success")
    ax_c.bar(x, fall_vals, bottom=success_vals, color=C_FALL, edgecolor="white", label="Fall")
    ax_c.bar(x, stall_vals, bottom=np.asarray(success_vals) + np.asarray(fall_vals), color=C_STALL, edgecolor="white", label="Stall")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([label for label, _assay, _cond in comp_specs], rotation=25, ha="right")
    ax_c.set_ylim(0, 1)
    ax_c.set_ylabel("Mean specimen rate")
    ax_c.set_title("C  Failure composition for selected native cuts")
    ax_c.legend(frameon=False, fontsize=7, loc="upper right")
    _style_axes(ax_c)

    # Panel D: double-cliff stress survival, explicitly separate endpoint.
    stress_specs = [
        ("Intact", "signaling_ablation", "intact"),
        ("All comm off", "signaling_ablation", "all_comm_off"),
        ("Sobel-X off", "signaling_ablation", "sobel_x_off"),
        ("Def. off", "deformation_ablation", "deformation_off"),
        ("Compliance only", "deformation_ablation", "compliance_only"),
    ]
    for i, (_label, assay, cond) in enumerate(stress_specs):
        rates = _condition_rates(
            eps,
            assay_family=assay,
            surface_id="double_cliff",
            condition_id=cond,
            endpoint="survived",
        )
        jitter = rng.normal(0, 0.035, len(rates))
        ax_d.scatter(np.full(len(rates), i) + jitter, rates, s=18, color="#555555", alpha=0.72, linewidth=0)
        ax_d.errorbar([i], [rates.mean()], yerr=[_sem(rates.values)], fmt="o", color=C_ACCENT_2, capsize=3, markersize=4)
    ax_d.set_xticks(np.arange(len(stress_specs)))
    ax_d.set_xticklabels([label for label, _assay, _cond in stress_specs], rotation=30, ha="right")
    ax_d.set_ylim(-0.03, 1.03)
    ax_d.set_ylabel("Survival rate")
    ax_d.set_title("D  Double-cliff stress survival")
    _style_axes(ax_d)

    path = output_dir / "fig3_ablation_cuts.pdf"
    _save_figure(fig, path)
    return path


def _scenario_rates(eps: pd.DataFrame, scenario: Scenario) -> pd.Series:
    return _condition_rates(
        eps,
        assay_family=scenario.assay_family,
        surface_id=scenario.surface_id,
        condition_id=scenario.condition_id,
        endpoint=scenario.endpoint,
    )


def plot_s4_mechanism(eps: pd.DataFrame, output_dir: Path) -> Path:
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    fig = plt.figure(figsize=(7.2, 6.7), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    rng = np.random.default_rng(139)

    scenarios = [
        Scenario("Native", "signaling_ablation", "native_core", "intact", "success"),
        Scenario("Extended", "signaling_ablation", "long_distance_generalization", "intact", "success"),
        Scenario("Double", "signaling_ablation", "double_cliff", "intact", "survived"),
        Scenario("Nat def. off", "deformation_ablation", "native_core", "deformation_off", "success"),
        Scenario("Dbl def. off", "deformation_ablation", "double_cliff", "deformation_off", "survived"),
        Scenario("Sobel-X off", "signaling_ablation", "native_core", "sobel_x_off", "success"),
        Scenario("All comm off", "signaling_ablation", "native_core", "all_comm_off", "success"),
    ]
    s4_values: list[float] = []
    for i, scenario in enumerate(scenarios):
        rates = _scenario_rates(eps, scenario)
        jitter = rng.normal(0, 0.035, len(rates))
        ax_a.scatter(np.full(len(rates), i) + jitter, rates, s=18, color="#BBBBBB", alpha=0.85, linewidth=0)
        if S4_SPECIMEN in rates.index:
            s4_values.append(float(rates.loc[S4_SPECIMEN]))
        else:
            s4_values.append(float("nan"))
    ax_a.plot(np.arange(len(scenarios)), s4_values, color=C_ACCENT, linewidth=1.6, marker="o", markersize=4, label="S4")
    ax_a.set_xticks(np.arange(len(scenarios)))
    ax_a.set_xticklabels([s.label for s in scenarios], rotation=35, ha="right", fontsize=7)
    ax_a.set_ylim(-0.03, 1.03)
    ax_a.set_ylabel("Rate")
    ax_a.set_title("A  S4 profile against cohort")
    ax_a.legend(frameon=False, loc="lower left", fontsize=7)
    _style_axes(ax_a)

    # True-cliff decoder summary from the 2026-06-11 S4 note.
    decoder = pd.DataFrame(
        [
            {"stream": "Body", "lag": 0, "auc": 0.837, "balanced_accuracy": 0.616},
            {"stream": "Signal", "lag": 0, "auc": 0.586, "balanced_accuracy": 0.500},
            {"stream": "Body", "lag": 8, "auc": 0.822, "balanced_accuracy": 0.571},
            {"stream": "Signal", "lag": 8, "auc": 0.643, "balanced_accuracy": 0.500},
        ]
    )
    labels = [f"{row.stream}\nlag {int(row.lag)}" for row in decoder.itertuples()]
    x = np.arange(len(decoder))
    width = 0.36
    ax_b.bar(x - width / 2, decoder["auc"], width=width, color=C_ACCENT, label="AUC")
    ax_b.bar(x + width / 2, decoder["balanced_accuracy"], width=width, color=C_ACCENT_2, label="Balanced acc.")
    ax_b.axhline(0.5, color=C_NEUTRAL, linewidth=0.8, linestyle="--")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylim(0.45, 1.0)
    ax_b.set_ylabel("Decoder score")
    ax_b.set_title("B  True-cliff committed-direction decoding")
    ax_b.legend(frameon=False, fontsize=7, loc="upper right")
    _style_axes(ax_b)

    # Leading-edge/frontier-band voxel AUC heatmap.
    voxel_path = DATA_ROOT / "figures" / "voxel_reversal_heatmap_deformation_off" / "voxel_decoder_summary.csv"
    voxel = pd.read_csv(voxel_path)
    subset = voxel[(voxel["window"] == "edge_impulse") & (voxel["lag"] == 8)]
    grid = np.full((2, 7), np.nan, dtype=float)
    top_cells = set(subset.sort_values("mean_auc", ascending=False)["cell_index"].head(4).astype(int).tolist())
    leading_cells = set(subset[subset["is_event_leading_cell"].astype(bool)]["cell_index"].astype(int).tolist())
    for _, row in subset.iterrows():
        cell = int(row["cell_index"])
        r = cell // 7
        c = cell % 7
        grid[r, c] = float(row["mean_auc"])
    image = ax_c.imshow(grid, cmap="viridis", vmin=0.5, vmax=1.0)
    for r in range(2):
        for c in range(7):
            cell = r * 7 + c
            value = grid[r, c]
            ax_c.text(c, r, f"{cell}\n{value:.2f}", ha="center", va="center", color="white", fontsize=7)
            if cell in top_cells:
                ax_c.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=C_ACCENT, linewidth=2.0))
            if cell in leading_cells:
                ax_c.add_patch(plt.Rectangle((c - 0.42, r - 0.42), 0.84, 0.84, fill=False, edgecolor="#D4736C", linewidth=1.5))
    ax_c.set_xticks(np.arange(7))
    ax_c.set_yticks(np.arange(2))
    ax_c.set_title("C  Frontier-band voxel AUC\n(edge impulse, lag 8)")
    fig.colorbar(image, ax=ax_c, fraction=0.046, pad=0.04, label="AUC")

    # Coupling metrics.
    coupling_path = DATA_ROOT / "figures" / "signal_susceptibility_deformation_off" / "signal_body_coupling_summary.csv"
    coupling = pd.read_csv(coupling_path)
    edge = coupling[coupling["window"] == "edge_impulse"].copy()
    label_map = {
        "signal_to_drive_same_step": "Signal -> drive",
        "signal_to_future_body": "Signal -> future body",
        "drive_to_future_body": "Drive -> future body",
    }
    edge["label"] = edge["coupling"].map(label_map)
    edge = edge[edge["label"].notna()]
    y = np.arange(len(edge))
    ax_d.barh(y - 0.18, edge["flattened_pearson"], height=0.34, color=C_ACCENT, label="Pearson r")
    ax_d.barh(y + 0.18, edge["global_r2"], height=0.34, color=C_ACCENT_2, label="R2")
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(edge["label"])
    ax_d.set_xlim(0, 1.05)
    ax_d.set_xlabel("Coupling score")
    ax_d.set_title("D  Signal/drive/body coupling\n(edge impulse window)")
    ax_d.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    _style_axes(ax_d)

    path = output_dir / "fig4_s4_mechanism.pdf"
    _save_figure(fig, path)
    return path


def _fmt_pct(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    rounded = Decimal(str(100.0 * float(value))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return f"{rounded}%"


def _fmt_pp(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    rounded = Decimal(str(100.0 * float(value))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return f"{rounded} pp"


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


def _write_revision_document(
    *,
    output_root: Path,
    figure_paths: list[Path],
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    pairs: pd.DataFrame,
    axes_table: pd.DataFrame,
) -> Path:
    native = _summary_row(summary, "signaling_ablation", "native_core", "intact")
    ext = _summary_row(summary, "signaling_ablation", "long_distance_generalization", "intact")
    sym = _summary_row(summary, "signaling_ablation", "double_cliff", "intact")
    s4_axes = axes_table[axes_table["specimen_id"] == S4_SPECIMEN].iloc[0]
    non_s4_def = summary[
        (summary["assay_family"] == "deformation_ablation")
        & (summary["surface_id"] == "native_core")
        & (summary["condition_id"] == "deformation_off")
    ].iloc[0]

    pair_order = {"(2,1)": 0, "(4,2)": 1, "(6,3)": 2, "(8,4)": 3, "(10,5)": 4}
    pairs = pairs.assign(_pair_order=pairs["distance_pair"].map(pair_order).fillna(99))
    native_pairs = pairs[pairs["surface_id"] == "native_core"].sort_values("_pair_order")
    extended_pairs = pairs[pairs["surface_id"] == "long_distance_generalization"].sort_values("_pair_order")

    signaling_lines = []
    for cond in SIGNALING_ORDER:
        row = _summary_row(summary, "signaling_ablation", "native_core", cond)
        delta = _delta_row(deltas, "signaling_ablation", "native_core", cond)
        signaling_lines.append(
            f"| {SIGNALING_LABELS[cond]} | {int(row['count'])}/{int(row['total'])} = {_fmt_pct(row['episode_rate'])} | "
            f"{_fmt_pct(row['fall_episode_rate'])} | {_fmt_pct(row['stall_episode_rate'])} | "
            f"{_fmt_pct(row['mean_specimen_rate'])} +/- {_fmt_pct(row['sem_specimen_rate'])} | "
            f"{_fmt_pp(delta['mean_delta'])} | {delta['wilcoxon_p_two_sided']:.4f} |"
        )

    deformation_lines = []
    for cond in DEFORMATION_ORDER:
        row = _summary_row(summary, "deformation_ablation", "native_core", cond)
        delta = _delta_row(deltas, "deformation_ablation", "native_core", cond)
        deformation_lines.append(
            f"| {DEFORMATION_LABELS[cond]} | {int(row['count'])}/{int(row['total'])} = {_fmt_pct(row['episode_rate'])} | "
            f"{_fmt_pct(row['fall_episode_rate'])} | {_fmt_pct(row['stall_episode_rate'])} | "
            f"{_fmt_pct(row['mean_specimen_rate'])} +/- {_fmt_pct(row['sem_specimen_rate'])} | "
            f"{_fmt_pp(delta['mean_delta'])} | {delta['wilcoxon_p_two_sided']:.4f} |"
        )

    pair_lines = []
    for _, row in pd.concat([native_pairs, extended_pairs]).iterrows():
        pair_lines.append(
            f"| {row['surface_id']} | {row['distance_pair']} | {int(row['count'])}/{int(row['total'])} = "
            f"{_fmt_pct(row['episode_rate'])} | {_fmt_pct(row['mean_specimen_rate'])} +/- {_fmt_pct(row['sem_specimen_rate'])} |"
        )

    figure_lines = "\n".join(f"- `{path}`" for path in figure_paths)
    table_lines = "\n".join(
        f"- `{path}`"
        for path in [
            output_root / "computed" / "eight_specimen_endpoint_summary.csv",
            output_root / "computed" / "eight_specimen_paired_deltas.csv",
            output_root / "computed" / "eight_specimen_specimen_deltas.csv",
            output_root / "computed" / "intact_distance_pair_summary.csv",
            output_root / "computed" / "specimen_failure_axes.csv",
        ]
    )

    text = f"""# Sensing at the Edges - Eight-Specimen Revision Materials

Created: 2026-06-12

This document prepares the manuscript revision around the new eight-specimen assay and the S4 mechanistic analysis. It is a working materials packet: it does not edit the manuscript directly, but it gives the statistical claims, chart set, caption guidance, and guardrails needed to revise the paper.

## Artifact Outputs

Figures:

{figure_lines}

Computed tables:

{table_lines}

Primary data source:

- `{DATA_ROOT / "tables" / "episode_manifest.csv"}`

S4 analysis sources:

- `{DATA_ROOT / "figures" / "voxel_reversal_heatmap_deformation_off" / "voxel_decoder_summary.csv"}`
- `{DATA_ROOT / "figures" / "signal_susceptibility_deformation_off" / "signal_body_coupling_summary.csv"}`
- `{PAPER_ROOT / "docs" / "perfect-specimen-deformation-off-state-space-2026-06-08.md"}`

## Reporting Policy

- Use specimens as the statistical unit. The cohort is `n=8` selected specimens, not eight independent evolutionary runs.
- Deterministic episodes are terrain/configuration evaluations, not stochastic replicates.
- Compute rates per specimen first, then summarize across specimens.
- Report mean +/- SEM across specimens for continuity with the current manuscript.
- Include raw episode counts where they are useful for readability.
- Use paired within-specimen deltas for ablation contrasts.
- Treat Wilcoxon/sign-test p-values as secondary support, not as the main evidentiary object.
- Apply the 50% baseline only to mirrored goal-cliff success tasks. Do not apply it to double-cliff survival.
- Call double-cliff an endpoint of survival or stress survival, not success.

## Core Population Results

### Native Core

Native intact success is `{int(native['count'])}/{int(native['total'])} = {_fmt_pct(native['episode_rate'])}`. The mean specimen success rate is `{_fmt_pct(native['mean_specimen_rate'])} +/- {_fmt_pct(native['sem_specimen_rate'])} SEM`. Falls are `{_fmt_pct(native['fall_episode_rate'])}` and stalls/no-hit/no-fall cases are `{_fmt_pct(native['stall_episode_rate'])}`.

Distance-pair performance:

| Surface | Pair | Episode success | Mean specimen success +/- SEM |
|---|---:|---:|---:|
{chr(10).join(pair_lines)}

Paper-safe claim:

> Eight selected specimens show strong native performance. The two shorter native pairs are perfect, and native intact failures are concentrated on the hardest `(6,3)` pair.

Avoid claiming this is eight independent evolutionary runs.

### Extended Generalization

Combined extended intact success is `{int(ext['count'])}/{int(ext['total'])} = {_fmt_pct(ext['episode_rate'])}`, with mean specimen success `{_fmt_pct(ext['mean_specimen_rate'])} +/- {_fmt_pct(ext['sem_specimen_rate'])} SEM`.

Paper-safe claim:

> Extended-distance generalization is materially lower than native performance but present, and it is heterogeneous across specimens.

Avoid saying that only S4 generalizes. S4 is distinctive because it combines perfect extended performance, all-width double-cliff survival, and deformation-off robustness.

### Double-Cliff / Symmetric Survival

Intact double-cliff survival is `{int(sym['count'])}/{int(sym['total'])} = {_fmt_pct(sym['episode_rate'])}`, with mean specimen survival `{_fmt_pct(sym['mean_specimen_rate'])} +/- {_fmt_pct(sym['sem_specimen_rate'])} SEM`.

S4 survives all three intact double-cliff widths. One other specimen survives one width; the remaining specimens survive none.

Paper-safe claim:

> Double-cliff terrains are a harsh no-goal stress assay. S4 is uniquely robust across all widths, while most specimens fail this endpoint.

Do not call this goal success and do not apply the 50% mirrored-goal baseline.

## Native Signalling Ablations

| Condition | Success | Fall | Stall | Mean success +/- SEM | Paired delta vs intact | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(signaling_lines)}

Paper-safe claim:

> Native signalling ablations consistently impair success. Sobel-X removal is the largest single-channel success cut, while the other removals produce distinct fall/stall profiles.

Avoid a clean one-filter-one-function mapping unless additional behavioral/state evidence is added. The endpoint data support distinct impairment profiles, not a settled decomposition such as "Sobel-Y detects cliffs" and "Sobel-X supports commitment."

Extended signalling contrast:

- Sobel-X off and all communication off both collapse to `3/32 = 9.4%` extended success.
- Both have paired extended deltas of about `-53.1 pp` relative to intact.

Double-cliff signalling:

- Intact survival: `4/24 = 16.7%`.
- All communication off: `12/24 = 50.0%`.
- Sobel-X off: `11/24 = 45.8%`.

Interpret this as stress survival under dampened or settled behavior, not as better goal-directed task performance.

## Native Deformation Ablations

| Condition | Success | Fall | Stall | Mean success +/- SEM | Paired delta vs intact | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(deformation_lines)}

Paper-safe claim:

> Native deformation ablations impair success and increase falls, but deformation dependence is not universal.

Excluding S4, native deformation-off success is `17/42 = 40.5%` and native deformation-off falls are `17/42 = 40.5%`. Use this to preserve the population-level impairment while acknowledging that S4 is not deformation-dependent under replay.

Use only the new manifest deformation conditions in the main text:

- intact
- absolute only
- compliance only
- deformation off

Do not include old noise, shuffled, or reversed deformation conditions in the main result unless new data are explicitly added.

## Body Transfer / Morphology

Do not keep a standalone morphology chart in the main figure set. Use body transfer inline as support for the body-as-sensory-interface argument.

Native body-transfer values:

- Native `7x2`: `43/48 = 89.6%` success.
- `6x2`: `24/48 = 50.0%` success.
- `8x2`: `19/48 = 39.6%` success.
- `7x3`: `20/48 = 41.7%` success.

Paper-safe claim:

> Locomotor activity partly transfers across bodies, but edge detection and reversal do not transfer cleanly. The body is not a passive output device; it is part of the sensory interface.

Use this to clarify "without input": the system is without specialized labelled sensory channels, not without perturbation or contact. The body's own deformation under environmental contact is the sensory medium.

## S4 Mechanistic Case Study

Specimen:

- `{S4_SPECIMEN}`

Assay profile:

- Native intact: `6/6` success.
- Extended intact: `4/4` success.
- Double-cliff intact: `3/3` survival.
- Native deformation off: `6/6` success.
- Double-cliff deformation off: `3/3` survival.
- Native Sobel-X off: `0/6` success, all stalls.
- Native identity-comm off: `0/6` success, all stalls.
- Native all-comm off: `0/6` success, all stalls.

S4 failure-axis values in the generated table:

- Native intact success: `{s4_axes['native_intact_success']:.3f}`.
- Extended intact success: `{s4_axes['extended_intact_success']:.3f}`.
- Symmetric intact survival: `{s4_axes['symmetric_intact_survival']:.3f}`.
- Native deformation-off success: `{s4_axes['native_deformation_off_success']:.3f}`.
- Symmetric deformation-off survival: `{s4_axes['symmetric_deformation_off_survival']:.3f}`.

### Decoder Result

True-cliff event-window direction decoding:

| Stream | Lag | AUC | Balanced accuracy | Empirical p |
|---|---:|---:|---:|---:|
| Body | 0 | 0.837 | 0.616 | 0.038 |
| Signal | 0 | 0.586 | 0.500 | 1.000 |
| Body | 8 | 0.822 | 0.571 | 0.038 |
| Signal | 8 | 0.643 | 0.500 | 1.000 |

Leading-edge/frontier-band, edge-impulse window, lag 8:

- Leading voxel state: AUC `0.980`, balanced accuracy `0.926`, MCC `0.853`.
- Attached springs: AUC `0.994`, balanced accuracy `0.861`.
- Connection springs: AUC `0.996`, balanced accuracy `0.770`.
- Top whole-body voxel AUCs: cell 7 `0.989`, cell 1 `0.981`, cell 0 `0.980`, cell 8 `0.958`.

Signal/drive/body coupling in the same edge-impulse window:

- Signal to drive: r `0.992`, R2 `0.984`.
- Signal to future body: r `0.953`, R2 `0.842`.
- Drive to future body: r `0.975`, R2 `0.921`.

Paper-safe claim:

> In S4 deformation-off replay, committed direction is predicted much better by body/leading-edge state than by hidden/internal signalling state. Signalling is regular and strongly coupled to drive/body evolution, but it does not itself linearly predict committed direction under this replay.

Avoid:

- "the body causes the reversal"
- "signalling does not matter"
- "direction is stored in the body"
- "this proves embodied control"
- universalizing S4 to all specimens

Use "leading-edge/frontier band", not "single voxel 0 alone."

## Figure Guide

### Figure 2: Population Diversity

File:

- `{figure_paths[0]}`

Purpose:

- Establish the eight-specimen population before S4.
- Show baseline endpoints, distance-pair generalization, and failure-axis diversity.

Draft caption:

> Eight-specimen performance and failure diversity. (A) Baseline endpoints across native, extended, and double-cliff surfaces. Native and extended panels report goal success, falls, and stalls; double-cliff reports survival/fall as a separate stress endpoint. Bars are means of specimen-level rates. (B) Distance-pair success with per-specimen dots and mean +/- SEM. The dashed 50% line applies only to mirrored goal-cliff tasks. (C) Specimen-level failure-axis heatmap; S4 is outlined.

### Figure 3: Ablation Cuts

File:

- `{figure_paths[1]}`

Purpose:

- Show what breaks under native signalling and deformation removals.
- Keep double-cliff survival separate from goal success.

Draft caption:

> Informative ablation cuts. (A,B) Paired within-specimen native success deltas relative to intact for signalling and deformation ablations. Dots are specimens; blue markers show mean +/- SEM. (C) Failure composition for selected native cuts. (D) Double-cliff stress survival under selected ablations, reported as survival rather than success.

### Figure 4: S4 Mechanistic Case Study

File:

- `{figure_paths[2]}`

Purpose:

- Show S4 as a discovered case study, not the whole population result.
- Put the S4 decoding result beside its assay profile.

Draft caption:

> S4 mechanistic case study under deformation-off replay. (A) S4 profile against the cohort across native, extended, double-cliff, deformation-off, and signalling-ablation endpoints. (B) True-cliff committed-direction decoding: body state outperforms hidden/internal signal state at lag 0 and lag 8. (C) Leading-edge/frontier-band voxel AUCs in the edge-impulse window at lag 8. (D) Signal, drive, and body streams are strongly coupled in the same window. These analyses are observational and do not by themselves establish causal control by the body state.

## Manuscript Replacement Checklist

Replace old five-specimen language:

- `5 independent runs`
- `five specimens`
- `remaining four`
- `n=5`

Use:

- `eight selected specimens`
- `specimen-level summaries across n=8`
- `selected from multiple source runs`

Remove old deformation conditions from main Results:

- `noise substitution`
- `shuffled`
- `reversed`
- `noise_only`

Replace clean filter-function claims with distinct impairment-profile language.

Replace "state-space analyses are deferred to future work" with the cautious S4 case-study result.

Ensure every double-cliff claim says survival or stress survival, not success.

## Fitness Prose

The source fitness constants are in:

- `{JAX_ROOT / "sensing_edges" / "sensing_edges" / "fitness.py"}`

Short replacement prose:

> Per-episode fitness rewards wrong-side exploration, gated reversal, subsequent goal-side progress, and goal contact, while penalizing cliff falls and lower-ledge exposure. Candidate fitness averages this score across mirrored terrain episodes and adds a bonus for balanced left/right goal hits.

The exact constants can remain in the equation or supplement, but they should not dominate the ecological task description.

## Verification Checklist

- Recompute the tables from `episode_manifest.csv`.
- Regenerate all three figures.
- Search the manuscript for obsolete claims: `n=5`, `five specimens`, `remaining four`, `noise`, `shuffled`, `reversed`.
- Confirm figure captions state endpoint and aggregation unit.
- Confirm double-cliff panels are labelled survival/stress survival.
- Confirm S4 language remains observational rather than causal.
- Confirm morphology is inline text, not a standalone main chart.
- Confirm deterministic episodes are not described as independent replicates.

## Minimal Revised Result Summary

Eight selected specimens achieve strong native performance on the bilateral goal-cliff task, with perfect success on shorter native pairs and failures concentrated on the hardest native pair. Extended generalization is lower but present and heterogeneous. Double-cliff survival is harsh, with S4 uniquely robust across all widths. Native signalling and deformation ablations both impair success, but they do so with distinct failure profiles. S4 is a distinctive case: it remains successful under deformation-off replay but collapses under key signalling ablations. In S4 deformation-off replay, committed direction is better predicted by leading-edge/frontier-band body state than by hidden/internal signalling state, while signalling remains dynamically coupled to drive/body evolution. Morphology transfer further supports the body-as-sensory-interface interpretation: altering body geometry degrades edge/reversal performance even when locomotor activity persists.
"""

    path = output_root / "sensing_edges_revision_materials_2026-06-12.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_outputs(output_root: Path) -> list[Path]:
    eps = _load_episode_manifest()
    computed_dir = output_root / "computed"
    figures_dir = output_root / "figures"
    computed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = _build_endpoint_summary(eps)
    deltas, specimen_deltas = _build_paired_deltas(eps)
    with_without_s4 = _build_with_without_s4_summary(eps)
    axes_table = _build_failure_axes(eps)
    pairs = _build_distance_pair_summary(eps)

    summary.to_csv(computed_dir / "eight_specimen_endpoint_summary.csv", index=False)
    deltas.to_csv(computed_dir / "eight_specimen_paired_deltas.csv", index=False)
    specimen_deltas.to_csv(computed_dir / "eight_specimen_specimen_deltas.csv", index=False)
    with_without_s4.to_csv(computed_dir / "eight_specimen_with_without_s4_summary.csv", index=False)
    axes_table.to_csv(computed_dir / "specimen_failure_axes.csv", index=False)
    pairs.to_csv(computed_dir / "intact_distance_pair_summary.csv", index=False)

    figure_paths = [
        plot_population_diversity(eps, axes_table, figures_dir),
        plot_ablation_cuts(eps, specimen_deltas, figures_dir),
        plot_s4_mechanism(eps, figures_dir),
    ]
    doc_path = _write_revision_document(
        output_root=output_root,
        figure_paths=figure_paths,
        summary=summary,
        deltas=deltas,
        pairs=pairs,
        axes_table=axes_table,
    )

    stdout = textwrap.dedent(
        f"""\
        WROTE {output_root}
        document={doc_path}
        figures:
        {chr(10).join(str(path) for path in figure_paths)}
        tables:
        {computed_dir / "eight_specimen_endpoint_summary.csv"}
        {computed_dir / "eight_specimen_paired_deltas.csv"}
        {computed_dir / "eight_specimen_specimen_deltas.csv"}
        {computed_dir / "eight_specimen_with_without_s4_summary.csv"}
        {computed_dir / "intact_distance_pair_summary.csv"}
        {computed_dir / "specimen_failure_axes.csv"}
        """
    )
    (output_root / "generation_stdout.txt").write_text(stdout, encoding="utf-8")
    print(stdout)
    return [doc_path, *figure_paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    write_outputs(args.output_root)


if __name__ == "__main__":
    main()
