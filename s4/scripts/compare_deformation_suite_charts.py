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
        description="Create direct comparison charts across deterministic deformation suite runs.",
    )
    parser.add_argument(
        "--suite",
        action="append",
        required=True,
        help="Suite label and root in the form LABEL=/abs/path/to/deterministic_suite_run",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _parse_suite_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Suite must be LABEL=PATH, got: {raw}")
    label, path = raw.split("=", 1)
    label = label.strip()
    suite_root = Path(path).expanduser().resolve()
    if not label:
        raise ValueError(f"Suite label is empty: {raw}")
    return label, suite_root


def _load_episode_df(suite_root: Path, suite_label: str) -> pd.DataFrame:
    path = suite_root / "deformation_ablation" / "per_episode.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing per-episode table: {path}")
    df = pd.read_csv(path)
    df["suite_label"] = suite_label
    for col in ("success", "fell", "survived"):
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for suite_label in df["suite_label"].drop_duplicates().tolist():
        suite_df = df[df["suite_label"] == suite_label]
        for surface_id in SURFACE_ORDER:
            surface_df = suite_df[suite_df["surface_id"] == surface_id]
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
                        "suite_label": suite_label,
                        "surface_id": surface_id,
                        "condition_id": condition_id,
                        "outcome_rate": outcome_rate,
                        "fall_rate": float(sub["fell"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def _plot_rate_compare(agg_df: pd.DataFrame, suite_labels: list[str], out_path: Path) -> None:
    fig, axes = plt.subplots(
        nrows=len(suite_labels),
        ncols=2,
        figsize=(12, 4.8 * len(suite_labels)),
        constrained_layout=True,
    )
    if len(suite_labels) == 1:
        axes = np.asarray([axes])
    metrics = [
        ("outcome_rate", "Success / Survival Rate"),
        ("fall_rate", "Fall Rate"),
    ]
    x = np.arange(len(SURFACE_ORDER))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, num=len(CONDITION_ORDER))

    for row_idx, suite_label in enumerate(suite_labels):
        suite_df = agg_df[agg_df["suite_label"] == suite_label]
        for col_idx, (metric_key, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for offset, condition_id in zip(offsets, CONDITION_ORDER):
                values = []
                for surface_id in SURFACE_ORDER:
                    row = suite_df[
                        (suite_df["surface_id"] == surface_id)
                        & (suite_df["condition_id"] == condition_id)
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
            ax.grid(axis="y", alpha=0.2, linewidth=0.8)
            if row_idx == 0:
                ax.set_title(title, fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(suite_label, fontsize=12, fontweight="bold")
    axes[0, 0].legend(frameon=False, loc="upper right")
    fig.suptitle("Deformation Suite Comparison", fontsize=15, fontweight="bold")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_double_cliff_timing(df: pd.DataFrame, suite_labels: list[str], out_path: Path) -> None:
    fig, axes = plt.subplots(
        nrows=len(suite_labels),
        ncols=2,
        figsize=(12, 4.8 * len(suite_labels)),
        constrained_layout=True,
    )
    if len(suite_labels) == 1:
        axes = np.asarray([axes])
    metrics = [
        ("time_to_first_edge_approach", "Time To First Edge Approach"),
        ("time_to_first_arrest", "Time To First Arrest / Fall"),
    ]

    for row_idx, suite_label in enumerate(suite_labels):
        suite_df = df[(df["suite_label"] == suite_label) & (df["surface_id"] == "double_cliff")].copy()
        suite_df["width"] = suite_df["width"].astype(int)
        for col_idx, (metric_key, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for condition_id in CONDITION_ORDER:
                sub = suite_df[suite_df["deformation_condition"] == condition_id].sort_values("width")
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
            ax.set_xticks(sorted(suite_df["width"].dropna().astype(int).unique()))
            ax.set_xlabel("Double-Cliff Width")
            ax.set_ylabel("Env Step")
            ax.grid(alpha=0.2, linewidth=0.8)
            if row_idx == 0:
                ax.set_title(title, fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{suite_label}\nEnv Step", fontsize=12, fontweight="bold")
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.suptitle("Double-Cliff Timing Comparison", fontsize=15, fontweight="bold")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    suite_pairs = [_parse_suite_arg(raw) for raw in args.suite]
    suite_labels = [label for label, _ in suite_pairs]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = [_load_episode_df(path, label) for label, path in suite_pairs]
    episode_df = pd.concat(frames, ignore_index=True)
    agg_df = _aggregate(episode_df)
    agg_df.to_csv(output_dir / "surface_condition_comparison.csv", index=False)

    _plot_rate_compare(agg_df, suite_labels, output_dir / "rate_comparison.png")
    _plot_double_cliff_timing(episode_df, suite_labels, output_dir / "double_cliff_timing.png")


if __name__ == "__main__":
    main()
