from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SURFACE_ORDER = [
    "native_core",
    "long_distance_generalization",
    "double_cliff",
]
SURFACE_LABELS = {
    "native_core": "Native Core",
    "long_distance_generalization": "Extended",
    "double_cliff": "Double Cliff",
}
CONDITION_ORDER = [
    "intact",
    "absolute_only",
    "compliance_only",
    "deformation_off",
]
CONDITION_LABELS = {
    "intact": "Intact",
    "absolute_only": "Absolute Only",
    "compliance_only": "Compliance Only",
    "deformation_off": "All Deformation Off",
}
CONDITION_COLORS = {
    "intact": "#1b5e20",
    "absolute_only": "#1565c0",
    "compliance_only": "#ef6c00",
    "deformation_off": "#8e24aa",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create compact summary charts for a deterministic deformation suite run.",
    )
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _load_episode_df(suite_root: Path) -> pd.DataFrame:
    path = suite_root / "deformation_ablation" / "per_episode.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing per-episode table: {path}")
    df = pd.read_csv(path)
    for col in ("success", "fell", "survived"):
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for surface_id in SURFACE_ORDER:
        surface_df = df[df["surface_id"] == surface_id]
        for condition_id in CONDITION_ORDER:
            sub = surface_df[surface_df["deformation_condition"] == condition_id]
            if sub.empty:
                continue
            if surface_id == "double_cliff":
                outcome_rate = float(sub["survived"].mean())
            else:
                outcome_rate = float(sub["success"].mean())
            rows.append(
                {
                    "surface_id": surface_id,
                    "condition_id": condition_id,
                    "n": int(len(sub)),
                    "outcome_rate": outcome_rate,
                    "fall_rate": float(sub["fell"].mean()),
                    "mean_fitness": float(sub["fitness"].mean()),
                    "mean_peak_front_x": float(sub["peak_front_x"].mean()),
                    "mean_final_com_x": float(sub["final_com_x"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _plot_surface_summary(agg_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(
        nrows=len(SURFACE_ORDER),
        ncols=4,
        figsize=(15, 9),
        constrained_layout=True,
    )
    metric_specs = [
        ("outcome_rate", "Outcome Rate", (0.0, 1.05)),
        ("fall_rate", "Fall Rate", (0.0, 1.05)),
        ("mean_fitness", "Mean Fitness", None),
        ("mean_peak_front_x", "Mean Peak Front X", None),
    ]
    x = np.arange(len(CONDITION_ORDER))
    labels = [CONDITION_LABELS[c] for c in CONDITION_ORDER]
    colors = [CONDITION_COLORS[c] for c in CONDITION_ORDER]

    for row_idx, surface_id in enumerate(SURFACE_ORDER):
        surface_df = agg_df[agg_df["surface_id"] == surface_id].set_index("condition_id")
        for col_idx, (metric_key, metric_label, y_lim) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            values = [float(surface_df.loc[c, metric_key]) for c in CONDITION_ORDER]
            ax.bar(x, values, color=colors, width=0.72)
            ax.set_xticks(x, labels, rotation=25, ha="right")
            if y_lim is not None:
                ax.set_ylim(*y_lim)
            ax.grid(axis="y", alpha=0.2, linewidth=0.8)
            if row_idx == 0:
                ax.set_title(metric_label, fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(SURFACE_LABELS[surface_id], fontsize=11, fontweight="bold")
            for i, value in enumerate(values):
                if metric_key in ("outcome_rate", "fall_rate"):
                    text = f"{value:.2f}"
                else:
                    text = f"{value:.1f}"
                ax.text(i, value, text, ha="center", va="bottom", fontsize=8)

    fig.suptitle("Deterministic Deformation Suite Summary", fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_rate_comparison(agg_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.8), constrained_layout=True)
    metric_specs = [
        ("outcome_rate", "Success / Survival Rate"),
        ("fall_rate", "Fall Rate"),
    ]
    x = np.arange(len(SURFACE_ORDER))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, num=len(CONDITION_ORDER))

    for ax, (metric_key, title) in zip(axes, metric_specs):
        for offset, condition_id in zip(offsets, CONDITION_ORDER):
            values = []
            for surface_id in SURFACE_ORDER:
                row = agg_df[
                    (agg_df["surface_id"] == surface_id) & (agg_df["condition_id"] == condition_id)
                ]
                values.append(float(row.iloc[0][metric_key]))
            ax.bar(
                x + offset,
                values,
                width=width,
                label=CONDITION_LABELS[condition_id],
                color=CONDITION_COLORS[condition_id],
            )
        ax.set_xticks(x, [SURFACE_LABELS[s] for s in SURFACE_ORDER])
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.2, linewidth=0.8)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Outcome Comparison", fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_double_cliff_timing(df: pd.DataFrame, out_path: Path) -> None:
    cliff_df = df[df["surface_id"] == "double_cliff"].copy()
    cliff_df["width"] = cliff_df["width"].astype(int)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.8), constrained_layout=True)
    metric_specs = [
        ("time_to_first_edge_approach", "Time To First Edge Approach"),
        ("time_to_first_arrest", "Time To First Arrest / Fall"),
    ]

    for ax, (metric_key, title) in zip(axes, metric_specs):
        for condition_id in CONDITION_ORDER:
            sub = cliff_df[cliff_df["deformation_condition"] == condition_id].sort_values("width")
            ax.plot(
                sub["width"].to_numpy(),
                sub[metric_key].to_numpy(),
                marker="o",
                linewidth=1.5,
                markersize=5,
                color=CONDITION_COLORS[condition_id],
                label=CONDITION_LABELS[condition_id],
            )
            for _, row in sub.iterrows():
                marker = "S" if bool(row["survived"]) else "F"
                ax.text(
                    float(row["width"]),
                    float(row[metric_key]),
                    marker,
                    ha="left",
                    va="bottom",
                    fontsize=8,
                    color=CONDITION_COLORS[condition_id],
                )
        ax.set_xticks(sorted(cliff_df["width"].unique()))
        ax.set_xlabel("Double-Cliff Width")
        ax.set_ylabel("Env Step")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.2, linewidth=0.8)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Double-Cliff Timing From Actual Runs", fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _heatmap_data(df: pd.DataFrame, surface_id: str, value_key: str) -> tuple[np.ndarray, list[str]]:
    surface_df = df[df["surface_id"] == surface_id].copy()
    if surface_id == "double_cliff":
        row_ids = [(int(width),) for width in sorted(surface_df["width"].dropna().astype(int).unique())]
        labels = [f"width {width}" for (width,) in row_ids]
        matrix = np.full((len(row_ids), len(CONDITION_ORDER)), np.nan, dtype=float)
        for r, (width,) in enumerate(row_ids):
            for c, condition_id in enumerate(CONDITION_ORDER):
                sub = surface_df[
                    (surface_df["width"].fillna(-1).astype(int) == width)
                    & (surface_df["deformation_condition"] == condition_id)
                ]
                if not sub.empty:
                    matrix[r, c] = float(sub.iloc[0][value_key])
        return matrix, labels

    row_ids = []
    for goal_distance, cliff_distance in sorted(
        {
            (int(goal), int(cliff))
            for goal, cliff in zip(
                surface_df["goal_distance"].dropna().astype(int),
                surface_df["cliff_distance"].dropna().astype(int),
            )
        }
    ):
        for side in ("left", "right"):
            row_ids.append((goal_distance, cliff_distance, side))
    labels = [f"({goal},{cliff}) {side[0].upper()}" for goal, cliff, side in row_ids]
    matrix = np.full((len(row_ids), len(CONDITION_ORDER)), np.nan, dtype=float)
    for r, (goal_distance, cliff_distance, side) in enumerate(row_ids):
        for c, condition_id in enumerate(CONDITION_ORDER):
            sub = surface_df[
                (surface_df["goal_distance"].fillna(-1).astype(int) == goal_distance)
                & (surface_df["cliff_distance"].fillna(-1).astype(int) == cliff_distance)
                & (surface_df["side"] == side)
                & (surface_df["deformation_condition"] == condition_id)
            ]
            if not sub.empty:
                matrix[r, c] = float(sub.iloc[0][value_key])
    return matrix, labels


def _plot_episode_heatmaps(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(SURFACE_ORDER),
        figsize=(14, 8),
        constrained_layout=True,
    )
    if len(SURFACE_ORDER) == 1:
        axes = [axes]
    for ax, surface_id in zip(axes, SURFACE_ORDER):
        matrix, row_labels = _heatmap_data(df, surface_id, "success" if surface_id != "double_cliff" else "survived")
        im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="YlGn", aspect="auto")
        ax.set_title(SURFACE_LABELS[surface_id], fontsize=11, fontweight="bold")
        ax.set_xticks(np.arange(len(CONDITION_ORDER)), [CONDITION_LABELS[c] for c in CONDITION_ORDER], rotation=25, ha="right")
        ax.set_yticks(np.arange(len(row_labels)), row_labels)
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                value = matrix[r, c]
                if np.isnan(value):
                    continue
                ax.text(c, r, f"{value:.0f}", ha="center", va="center", fontsize=8, color="black")
    cbar = fig.colorbar(im, ax=axes, shrink=0.75)
    cbar.set_label("Outcome (1=success/survival, 0=failure)", rotation=90)
    fig.suptitle("Per-Episode Outcome Grid", fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_fitness_heatmaps(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(SURFACE_ORDER),
        figsize=(14, 8),
        constrained_layout=True,
    )
    if len(SURFACE_ORDER) == 1:
        axes = [axes]
    matrices = []
    for surface_id in SURFACE_ORDER:
        matrix, _ = _heatmap_data(df, surface_id, "fitness")
        matrices.append(matrix)
    vmax = np.nanmax(np.concatenate([m.flatten() for m in matrices]))
    vmin = np.nanmin(np.concatenate([m.flatten() for m in matrices]))
    for ax, surface_id, matrix in zip(axes, SURFACE_ORDER, matrices):
        _, row_labels = _heatmap_data(df, surface_id, "fitness")
        im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="Blues", aspect="auto")
        ax.set_title(SURFACE_LABELS[surface_id], fontsize=11, fontweight="bold")
        ax.set_xticks(np.arange(len(CONDITION_ORDER)), [CONDITION_LABELS[c] for c in CONDITION_ORDER], rotation=25, ha="right")
        ax.set_yticks(np.arange(len(row_labels)), row_labels)
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                value = matrix[r, c]
                if np.isnan(value):
                    continue
                ax.text(c, r, f"{value:.0f}", ha="center", va="center", fontsize=8, color="black")
    cbar = fig.colorbar(im, ax=axes, shrink=0.75)
    cbar.set_label("Fitness", rotation=90)
    fig.suptitle("Per-Episode Fitness Grid", fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    suite_root = args.suite_root.resolve()
    output_dir = (args.output_dir or (suite_root / "clean_charts")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_df = _load_episode_df(suite_root)
    agg_df = _aggregate(episode_df)
    agg_df.to_csv(output_dir / "surface_condition_summary.csv", index=False)

    _plot_surface_summary(agg_df, output_dir / "surface_summary.png")
    _plot_rate_comparison(agg_df, output_dir / "rate_comparison.png")
    _plot_double_cliff_timing(episode_df, output_dir / "double_cliff_timing.png")
    _plot_episode_heatmaps(episode_df, output_dir / "episode_outcomes.png")
    _plot_fitness_heatmaps(episode_df, output_dir / "episode_fitness.png")


if __name__ == "__main__":
    main()
