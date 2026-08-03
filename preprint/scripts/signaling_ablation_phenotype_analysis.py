#!/usr/bin/env python
"""Phenotype analysis for native-core signalling ablations.

This is a manifest-level analysis over the eight-specimen post-run assay.
It intentionally separates endpoint/failure-behaviour summaries from true
trace-level tortuosity and contraction-rhythm claims, which require raw
state-space traces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "paper_data" / "preprint" / "native_core_ablation_v1"
MANIFEST_PATH = DATA_ROOT / "tables" / "episode_manifest.csv"
OUTPUT_ROOT = DATA_ROOT / "signaling_ablation_phenotype_20260621"
TABLES_DIR = OUTPUT_ROOT / "tables"
FIGURES_DIR = OUTPUT_ROOT / "figures"

CONDITION_ORDER = [
    "intact",
    "sobel_x_off",
    "sobel_y_off",
    "laplacian_off",
    "identity_comm_off",
    "all_comm_off",
]
CONDITION_LABELS = {
    "intact": "Intact",
    "sobel_x_off": "Sobel-X off",
    "sobel_y_off": "Sobel-Y off",
    "laplacian_off": "Laplacian off",
    "identity_comm_off": "Identity off",
    "all_comm_off": "All comm off",
}
OUTCOME_COLORS = {
    "success": "#8CB87C",
    "fall": "#D4736C",
    "stall": "#AAAAAA",
}
ACCENT = "#3B5B92"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _condition_sort(values: pd.Series) -> pd.Categorical:
    return pd.Categorical(values, categories=CONDITION_ORDER, ordered=True)


def _load_native_signaling() -> pd.DataFrame:
    df = pd.read_csv(MANIFEST_PATH)
    native = df[
        (df["assay_family"] == "signaling_ablation")
        & (df["surface_id"] == "native_core")
    ].copy()
    native["condition_id"] = native["condition_id"].astype(str)
    native["condition_label_short"] = native["condition_id"].map(CONDITION_LABELS).fillna(native["condition_id"])
    native["condition_order"] = _condition_sort(native["condition_id"])
    native["success"] = native["success"].astype(bool)
    native["fell"] = native["fell"].astype(bool)
    native["stall"] = (~native["success"]) & (~native["fell"])
    native["outcome"] = np.where(
        native["success"],
        "success",
        np.where(native["fell"], "fall", "stall"),
    )
    native["steps_alive_fraction_of_full_budget"] = native["steps_alive"].astype(float) / float(native["steps_alive"].max())
    native["abs_final_com_x"] = native["final_com_x"].astype(float).abs()
    native["abs_peak_front_x"] = native["peak_front_x"].astype(float).abs()
    native["probe_extent_filled"] = native["probe_extent"].astype(float)
    native["mean_speed_scaled"] = native["mean_speed"].astype(float) * 1000.0
    native["low_speed"] = native["mean_speed"] < native["mean_speed"].quantile(0.25)
    return native.sort_values(["condition_order", "specimen_id", "distance_pair", "side"], kind="stable")


def _rate_summary(group: pd.DataFrame) -> dict[str, Any]:
    n = int(len(group))
    return {
        "episode_count": n,
        "success_count": int(group["success"].sum()),
        "fall_count": int(group["fell"].sum()),
        "stall_count": int(group["stall"].sum()),
        "success_rate": float(group["success"].mean()) if n else float("nan"),
        "fall_rate": float(group["fell"].mean()) if n else float("nan"),
        "stall_rate": float(group["stall"].mean()) if n else float("nan"),
    }


def build_outcome_tables(native: pd.DataFrame) -> dict[str, pd.DataFrame]:
    condition_rows: list[dict[str, Any]] = []
    for condition, group in native.groupby("condition_id", sort=False):
        row = {
            "condition_id": condition,
            "condition_label": CONDITION_LABELS.get(condition, condition),
            **_rate_summary(group),
            "median_steps_alive": float(group["steps_alive"].median()),
            "median_mean_speed": float(group["mean_speed"].median()),
            "median_mean_speed_x1000": float(group["mean_speed_scaled"].median()),
            "median_mean_activation": float(group["mean_activation"].median()),
            "median_probe_extent": float(group["probe_extent_filled"].median()),
            "median_abs_final_com_x": float(group["abs_final_com_x"].median()),
            "max_steps_episode_count": int((group["termination_reason"] == "max_steps").sum()),
            "death_plane_count": int((group["termination_reason"] == "death_plane").sum()),
            "airborne_count": int((group["termination_reason"] == "airborne").sum()),
        }
        condition_rows.append(row)
    condition_summary = pd.DataFrame(condition_rows)
    condition_summary["condition_order"] = _condition_sort(condition_summary["condition_id"])
    condition_summary = condition_summary.sort_values("condition_order").drop(columns=["condition_order"])

    by_pair_side = (
        native.groupby(["condition_id", "distance_pair", "side"], sort=False)
        .apply(lambda g: pd.Series(_rate_summary(g)))
        .reset_index()
    )
    by_pair_side["condition_order"] = _condition_sort(by_pair_side["condition_id"])
    by_pair_side = by_pair_side.sort_values(["condition_order", "distance_pair", "side"]).drop(columns=["condition_order"])

    term = (
        native.groupby(["condition_id", "termination_reason"], sort=False)
        .size()
        .rename("count")
        .reset_index()
    )
    term["condition_order"] = _condition_sort(term["condition_id"])
    term = term.sort_values(["condition_order", "termination_reason"]).drop(columns=["condition_order"])
    return {
        "condition_summary": condition_summary,
        "pair_side_summary": by_pair_side,
        "termination_summary": term,
    }


def build_specimen_tables(native: pd.DataFrame) -> dict[str, pd.DataFrame]:
    metric_cols = [
        "success",
        "fell",
        "stall",
        "steps_alive",
        "steps_alive_fraction_of_full_budget",
        "mean_speed",
        "mean_speed_scaled",
        "mean_activation",
        "probe_extent_filled",
        "abs_final_com_x",
        "abs_peak_front_x",
    ]
    per_specimen = (
        native.groupby(["specimen_id", "condition_id"], sort=False)[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    per_specimen["condition_order"] = _condition_sort(per_specimen["condition_id"])
    per_specimen = per_specimen.sort_values(["condition_order", "specimen_id"]).drop(columns=["condition_order"])

    across_rows: list[dict[str, Any]] = []
    for condition, group in per_specimen.groupby("condition_id", sort=False):
        row: dict[str, Any] = {
            "condition_id": condition,
            "condition_label": CONDITION_LABELS.get(condition, condition),
            "specimen_count": int(group["specimen_id"].nunique()),
        }
        for col in metric_cols:
            values = group[col].to_numpy(dtype=float)
            row[f"{col}_mean"] = float(np.nanmean(values))
            row[f"{col}_sd"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else float("nan")
            row[f"{col}_median"] = float(np.nanmedian(values))
            row[f"{col}_min"] = float(np.nanmin(values))
            row[f"{col}_max"] = float(np.nanmax(values))
        across_rows.append(row)
    specimen_summary = pd.DataFrame(across_rows)
    specimen_summary["condition_order"] = _condition_sort(specimen_summary["condition_id"])
    specimen_summary = specimen_summary.sort_values("condition_order").drop(columns=["condition_order"])

    baseline = per_specimen[per_specimen["condition_id"] == "intact"].set_index("specimen_id")
    delta_rows: list[dict[str, Any]] = []
    for _, row in per_specimen.iterrows():
        specimen = str(row["specimen_id"])
        condition = str(row["condition_id"])
        if condition == "intact" or specimen not in baseline.index:
            continue
        out = {"specimen_id": specimen, "condition_id": condition}
        for col in metric_cols:
            out[f"{col}_delta_vs_intact"] = float(row[col] - baseline.loc[specimen, col])
        delta_rows.append(out)
    deltas = pd.DataFrame(delta_rows)
    if not deltas.empty:
        deltas["condition_order"] = _condition_sort(deltas["condition_id"])
        deltas = deltas.sort_values(["condition_order", "specimen_id"]).drop(columns=["condition_order"])
    return {
        "specimen_condition_metrics": per_specimen,
        "specimen_condition_summary": specimen_summary,
        "specimen_deltas_vs_intact": deltas,
    }


def _save_fig(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_outcomes(condition_summary: pd.DataFrame) -> None:
    labels = condition_summary["condition_label"].tolist()
    x = np.arange(len(labels))
    success = condition_summary["success_rate"].to_numpy(dtype=float) * 100
    fall = condition_summary["fall_rate"].to_numpy(dtype=float) * 100
    stall = condition_summary["stall_rate"].to_numpy(dtype=float) * 100
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    ax.bar(x, success, color=OUTCOME_COLORS["success"], label="Success", width=0.72)
    ax.bar(x, fall, bottom=success, color=OUTCOME_COLORS["fall"], label="Fall", width=0.72)
    ax.bar(x, stall, bottom=success + fall, color=OUTCOME_COLORS["stall"], label="Stall", width=0.72)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Episodes (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=8, loc="upper right", ncols=3)
    ax.set_title("Native signalling ablation outcomes", fontsize=10, weight="bold")
    _save_fig(fig, "signaling_outcome_composition")


def plot_behaviour_metrics(specimen_summary: pd.DataFrame) -> None:
    metrics = [
        ("mean_speed_scaled", "Speed x1000"),
        ("steps_alive_fraction_of_full_budget", "Lifetime fraction"),
        ("mean_activation", "Mean activation"),
        ("probe_extent_filled", "Probe extent"),
    ]
    labels = specimen_summary["condition_label"].tolist()
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.3), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics, strict=True):
        means = specimen_summary[f"{metric}_mean"].to_numpy(dtype=float)
        sd = specimen_summary[f"{metric}_sd"].to_numpy(dtype=float)
        ax.bar(x, means, yerr=sd, color=ACCENT, alpha=0.82, capsize=2, width=0.68)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Native signalling ablation behavioural proxies", fontsize=10, weight="bold")
    _save_fig(fig, "signaling_behaviour_proxy_bars")


def plot_termination(term: pd.DataFrame) -> None:
    pivot = (
        term.pivot_table(index="condition_id", columns="termination_reason", values="count", fill_value=0)
        .reindex(CONDITION_ORDER)
        .fillna(0)
    )
    order = [c for c in ["goal_touched", "death_plane", "airborne", "max_steps"] if c in pivot.columns]
    colors = {
        "goal_touched": OUTCOME_COLORS["success"],
        "death_plane": OUTCOME_COLORS["fall"],
        "airborne": "#C35F5F",
        "max_steps": OUTCOME_COLORS["stall"],
    }
    labels = [CONDITION_LABELS[c] for c in pivot.index]
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    for col in order:
        values = pivot[col].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, color=colors.get(col, "#777777"), label=col.replace("_", " "), width=0.72)
        bottom += values
    ax.set_ylabel("Episodes")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=8, loc="upper right", ncols=2)
    ax.set_title("Termination grammar", fontsize=10, weight="bold")
    _save_fig(fig, "signaling_termination_grammar")


def plot_specimen_heatmap(per_specimen: pd.DataFrame) -> None:
    specimens = sorted(per_specimen["specimen_id"].unique())
    conditions = CONDITION_ORDER
    metrics = ["success", "fell", "stall", "mean_speed_scaled", "mean_activation"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(10.5, 3.4), constrained_layout=True)
    for ax, metric in zip(axes, metrics, strict=True):
        mat = np.full((len(specimens), len(conditions)), np.nan, dtype=float)
        for i, specimen in enumerate(specimens):
            for j, condition in enumerate(conditions):
                hit = per_specimen[(per_specimen["specimen_id"] == specimen) & (per_specimen["condition_id"] == condition)]
                if not hit.empty:
                    mat[i, j] = float(hit.iloc[0][metric])
        im = ax.imshow(mat, aspect="auto", vmin=np.nanmin(mat), vmax=np.nanmax(mat), cmap="viridis")
        ax.set_title(metric.replace("_", " "), fontsize=8)
        ax.set_xticks(np.arange(len(conditions)))
        ax.set_xticklabels([CONDITION_LABELS[c].replace(" ", "\n") for c in conditions], fontsize=6, rotation=0)
        ax.set_yticks(np.arange(len(specimens)))
        ax.set_yticklabels([f"S{i+1}" for i in range(len(specimens))], fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    _save_fig(fig, "signaling_specimen_metric_heatmaps")


def write_report(
    *,
    native: pd.DataFrame,
    condition_summary: pd.DataFrame,
    specimen_summary: pd.DataFrame,
    term: pd.DataFrame,
) -> Path:
    report = OUTPUT_ROOT / "signaling_ablation_phenotype_report.md"
    limitations = []
    if native["turn_count"].nunique(dropna=False) <= 1:
        limitations.append("`turn_count` is constant in the native signalling manifest and is not useful as tortuosity evidence.")
    if native["time_to_first_cliff_vicinity"].isna().all():
        limitations.append("Cliff-vicinity timing fields are empty for native signalling rows.")
    if native["max_cliff_protrusion"].isna().all():
        limitations.append("Cliff protrusion/stopping-distance fields are empty for native signalling rows.")
    lines = [
        "# Native signalling ablation phenotype analysis",
        "",
        "## Scope",
        "",
        f"- Source manifest: `{MANIFEST_PATH}`",
        f"- Rows analysed: `{len(native)}` native-core signalling episodes",
        f"- Specimens: `{native['specimen_id'].nunique()}`",
        f"- Conditions: `{', '.join(CONDITION_ORDER)}`",
        "",
        "This is a manifest-level phenotype analysis. It can characterise endpoint/failure grammar and behavioural proxies, but true geometric tortuosity and contraction rhythm require raw state-space traces.",
        "",
        "## Main endpoint grammar",
        "",
        "| Condition | Success | Fall | Stall | Median steps | Median speed x1000 | Median activation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in condition_summary.iterrows():
        n = int(row["episode_count"])
        lines.append(
            f"| {row['condition_label']} | "
            f"{int(row['success_count'])}/{n} | {int(row['fall_count'])}/{n} | {int(row['stall_count'])}/{n} | "
            f"{float(row['median_steps_alive']):.0f} | {float(row['median_mean_speed_x1000']):.3f} | "
            f"{float(row['median_mean_activation']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Phenotype notes",
            "",
            "- Sobel-X off is the strongest single-filter success cut and has the largest stall/max-step component.",
            "- Sobel-Y off has the highest fall count among single-filter cuts, matching the subjective impression of unstable edge interaction.",
            "- Laplacian off is mixed, with substantial stalls and falls.",
            "- Identity-communication off retains some successes but combines early deaths with a sizeable full-budget stall component.",
            "- All communication off produces broad dampening: many max-step stalls but also a non-trivial set of successes and falls, so it is not a uniform total collapse across specimens.",
            "",
            "## What this can and cannot say",
            "",
        ]
    )
    for item in limitations:
        lines.append(f"- {item}")
    lines.extend(
        [
            "- Use `mean_speed`, `steps_alive`, `termination_reason`, `probe_extent`, and `mean_activation` as behavioural proxies only.",
            "- Do not describe this table as measuring contraction rhythms. It measures aggregate activation, not phase/amplitude/coherence of contractile drive.",
            "- The next required step for tortuosity and locomotor-contraction pattern is native-core state-space capture for the signalling conditions.",
            "",
            "## Output files",
            "",
            "- `tables/signaling_condition_summary.csv`",
            "- `tables/signaling_pair_side_summary.csv`",
            "- `tables/signaling_termination_summary.csv`",
            "- `tables/signaling_specimen_condition_metrics.csv`",
            "- `tables/signaling_specimen_condition_summary.csv`",
            "- `tables/signaling_specimen_deltas_vs_intact.csv`",
            "- `figures/signaling_outcome_composition.{png,pdf}`",
            "- `figures/signaling_termination_grammar.{png,pdf}`",
            "- `figures/signaling_behaviour_proxy_bars.{png,pdf}`",
            "- `figures/signaling_specimen_metric_heatmaps.{png,pdf}`",
        ]
    )
    report.write_text("\n".join(lines) + "\n")
    return report


def run() -> dict[str, Path]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    native = _load_native_signaling()
    outcome_tables = build_outcome_tables(native)
    specimen_tables = build_specimen_tables(native)

    paths: dict[str, Path] = {}
    for name, frame in {**outcome_tables, **specimen_tables}.items():
        path = TABLES_DIR / f"signaling_{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path

    plot_outcomes(outcome_tables["condition_summary"])
    plot_behaviour_metrics(specimen_tables["specimen_condition_summary"])
    plot_termination(outcome_tables["termination_summary"])
    plot_specimen_heatmap(specimen_tables["specimen_condition_metrics"])

    summary = {
        "schema_version": 1,
        "kind": "native_signaling_ablation_phenotype_analysis",
        "source_manifest": MANIFEST_PATH,
        "output_root": OUTPUT_ROOT,
        "row_count": int(len(native)),
        "specimen_count": int(native["specimen_id"].nunique()),
        "condition_order": CONDITION_ORDER,
        "limitations": {
            "turn_count_unique_values": sorted(native["turn_count"].dropna().unique().tolist()),
            "time_to_first_cliff_vicinity_non_null": int(native["time_to_first_cliff_vicinity"].notna().sum()),
            "max_cliff_protrusion_non_null": int(native["max_cliff_protrusion"].notna().sum()),
            "contraction_rhythm_requires_state_space_trace": True,
        },
        "tables": {key: str(value) for key, value in paths.items()},
        "figures": {
            "outcome_composition": str(FIGURES_DIR / "signaling_outcome_composition.png"),
            "termination_grammar": str(FIGURES_DIR / "signaling_termination_grammar.png"),
            "behaviour_proxy_bars": str(FIGURES_DIR / "signaling_behaviour_proxy_bars.png"),
            "specimen_metric_heatmaps": str(FIGURES_DIR / "signaling_specimen_metric_heatmaps.png"),
        },
    }
    summary_path = OUTPUT_ROOT / "signaling_ablation_phenotype_summary.json"
    _write_json(summary_path, summary)
    paths["summary"] = summary_path
    paths["report"] = write_report(
        native=native,
        condition_summary=outcome_tables["condition_summary"],
        specimen_summary=specimen_tables["specimen_condition_summary"],
        term=outcome_tables["termination_summary"],
    )
    return paths


def main() -> None:
    for key, value in run().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
