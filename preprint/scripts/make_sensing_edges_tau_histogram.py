#!/usr/bin/env python3
"""Generate a tau-range histogram for the eight selected Sensing Edges specimens."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import make_sensing_edges_revision_materials as base


DEFAULT_OUTPUT_ROOT = base.DEFAULT_OUTPUT_ROOT


def _config_number(config_path: Path, key: str) -> float:
    pattern = re.compile(rf"^\s*{re.escape(key)}:\s*([^#\n]+)")
    for line in config_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            raw = match.group(1).strip().strip("'\"")
            return float(raw)
    raise KeyError(f"Missing {key!r} in {config_path}")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


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


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E4E4E4", linewidth=0.55)
    ax.set_axisbelow(True)


def load_tau_values() -> tuple[pd.DataFrame, pd.DataFrame]:
    provenance_path = base.DATA_ROOT / "tables" / "provenance.csv"
    provenance = pd.read_csv(provenance_path)

    rows: list[dict[str, object]] = []
    for _, row in provenance.iterrows():
        config_path = Path(str(row["local_config_path"]))
        controller_path = config_path.parent / "controller_params.npz"
        if not controller_path.exists():
            raise FileNotFoundError(f"Missing controller params for {row['specimen_id']}: {controller_path}")

        tau_min = _config_number(config_path, "ctrnn_tau_min")
        tau_max = _config_number(config_path, "ctrnn_tau_max")
        hidden = int(_config_number(config_path, "ctrnn_hidden"))

        with np.load(controller_path, allow_pickle=False) as params:
            log_tau = np.asarray(params["log_tau"], dtype=np.float64)
        if log_tau.shape != (hidden,):
            raise ValueError(
                f"{controller_path} log_tau shape {log_tau.shape} does not match ctrnn_hidden={hidden}"
            )

        tau = tau_min + (tau_max - tau_min) * _sigmoid(log_tau)
        for unit_index, (log_tau_value, tau_value) in enumerate(zip(log_tau, tau, strict=True)):
            rows.append(
                {
                    "group_id": row["group_id"],
                    "run_id": row["run_id"],
                    "specimen_id": row["specimen_id"],
                    "unit_index": unit_index,
                    "log_tau": float(log_tau_value),
                    "tau": float(tau_value),
                    "tau_min": float(tau_min),
                    "tau_max": float(tau_max),
                    "controller_params_path": str(controller_path),
                    "config_snapshot_path": str(config_path),
                }
            )

    tau_values = pd.DataFrame(rows)
    specimen_summary = (
        tau_values.groupby(["group_id", "run_id", "specimen_id"], as_index=False)
        .agg(
            n_units=("tau", "size"),
            tau_min_allowed=("tau_min", "first"),
            tau_max_allowed=("tau_max", "first"),
            tau_min_observed=("tau", "min"),
            tau_q25=("tau", lambda values: float(np.quantile(values, 0.25))),
            tau_median=("tau", "median"),
            tau_mean=("tau", "mean"),
            tau_q75=("tau", lambda values: float(np.quantile(values, 0.75))),
            tau_max_observed=("tau", "max"),
            tau_std=("tau", "std"),
        )
        .sort_values(["group_id", "specimen_id"])
    )
    return tau_values, specimen_summary


def _binned_counts_by_specimen(tau_values: pd.DataFrame, bins: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["group_id", "run_id", "specimen_id"]
    for keys, group in tau_values.groupby(group_cols, sort=True):
        counts, _edges = np.histogram(group["tau"].to_numpy(dtype=float), bins=bins)
        group_id, run_id, specimen_id = keys
        for bin_index, count in enumerate(counts):
            rows.append(
                {
                    "group_id": group_id,
                    "run_id": run_id,
                    "specimen_id": specimen_id,
                    "bin_index": bin_index,
                    "bin_left": float(bins[bin_index]),
                    "bin_right": float(bins[bin_index + 1]),
                    "count": int(count),
                }
            )
    return pd.DataFrame(rows)


def save_histogram(tau_values: pd.DataFrame, output_dir: Path) -> tuple[Path, pd.DataFrame]:
    allowed_min_values = tau_values["tau_min"].unique()
    allowed_max_values = tau_values["tau_max"].unique()
    if len(allowed_min_values) != 1 or len(allowed_max_values) != 1:
        raise ValueError("Expected a single tau range across the selected cohort")

    tau_min = float(allowed_min_values[0])
    tau_max = float(allowed_max_values[0])
    bins = np.arange(tau_min, tau_max + 1.0001, 1.0)
    if bins[-1] < tau_max:
        bins = np.append(bins, tau_max)
    bin_counts = _binned_counts_by_specimen(tau_values, bins)
    binned = (
        bin_counts.groupby(["bin_index", "bin_left", "bin_right"], as_index=False)
        .agg(
            mean_count=("count", "mean"),
            sem_count=("count", lambda values: float(np.std(values, ddof=1) / np.sqrt(len(values)))),
        )
        .sort_values("bin_index")
    )
    values = tau_values["tau"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(3.3, 2.05), constrained_layout=True)
    centers = (binned["bin_left"].to_numpy(dtype=float) + binned["bin_right"].to_numpy(dtype=float)) / 2.0
    widths = binned["bin_right"].to_numpy(dtype=float) - binned["bin_left"].to_numpy(dtype=float)
    ax.bar(
        centers,
        binned["mean_count"].to_numpy(dtype=float),
        yerr=binned["sem_count"].to_numpy(dtype=float),
        width=widths * 0.95,
        align="center",
        color=base.C_ACCENT,
        edgecolor="white",
        linewidth=0.7,
        capsize=2.5,
    )
    ax.axvline(np.median(values), color=base.C_NEUTRAL, linestyle="--", linewidth=0.9, label="Median")
    ax.set_xlim(tau_min, tau_max)
    ax.set_xticks(np.arange(tau_min, tau_max + 0.1, 1.0))
    ax.set_xlabel(f"Tau (allowed range {tau_min:g}-{tau_max:g})")
    ax.set_ylabel("Mean hidden units / specimen")
    ax.set_title("CTRNN tau distribution")
    ax.legend(frameon=False, fontsize=6.8, loc="upper right")
    _style_axes(ax)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tau_histogram.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path, bin_counts


def save_outputs(output_root: Path) -> dict[str, object]:
    _set_style()
    output_dir = output_root / "tau_figures"
    computed_dir = output_root / "computed"
    output_dir.mkdir(parents=True, exist_ok=True)
    computed_dir.mkdir(parents=True, exist_ok=True)

    tau_values, specimen_summary = load_tau_values()
    tau_values_path = computed_dir / "eight_specimen_tau_values.csv"
    specimen_summary_path = computed_dir / "eight_specimen_tau_summary.csv"
    tau_values.to_csv(tau_values_path, index=False)
    specimen_summary.to_csv(specimen_summary_path, index=False)

    histogram_path, bin_counts = save_histogram(tau_values, output_dir)
    bin_counts_path = computed_dir / "eight_specimen_tau_bin_counts.csv"
    values = tau_values["tau"].to_numpy(dtype=float)
    bin_counts.to_csv(bin_counts_path, index=False)
    payload = {
        "n_specimens": int(tau_values["specimen_id"].nunique()),
        "ctrnn_units_per_specimen": int(tau_values.groupby("specimen_id").size().iloc[0]),
        "n_tau_values": int(values.size),
        "allowed_tau_min": float(tau_values["tau_min"].iloc[0]),
        "allowed_tau_max": float(tau_values["tau_max"].iloc[0]),
        "observed_min": float(np.min(values)),
        "observed_q25": float(np.quantile(values, 0.25)),
        "observed_median": float(np.median(values)),
        "observed_mean": float(np.mean(values)),
        "observed_q75": float(np.quantile(values, 0.75)),
        "observed_max": float(np.max(values)),
        "observed_std": float(np.std(values, ddof=1)),
        "histogram_pdf": str(histogram_path),
        "histogram_png": str(histogram_path.with_suffix(".png")),
        "tau_values_csv": str(tau_values_path),
        "specimen_summary_csv": str(specimen_summary_path),
        "bin_counts_csv": str(bin_counts_path),
        "histogram_y_axis": "Mean hidden-unit count per tau bin across specimens; whiskers are SEM across specimens.",
    }
    summary_path = computed_dir / "eight_specimen_tau_overall_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("WROTE", histogram_path)
    print("WROTE", histogram_path.with_suffix(".png"))
    print("WROTE", tau_values_path)
    print("WROTE", specimen_summary_path)
    print("WROTE", bin_counts_path)
    print("WROTE", summary_path)
    print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    save_outputs(args.output_root)


if __name__ == "__main__":
    main()
