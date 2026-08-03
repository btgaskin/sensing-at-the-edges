#!/usr/bin/env python
"""Run the earlier S4 band decoder on the expanded goal-cliff traces.

This is intentionally a compatibility analysis: it reuses the same band/window
future-COM decoder from ``s4_mechanism_packet.py`` while reading the 30-event
S4 goal-cliff propagation expansion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

import s4_mechanism_packet as old


DEFAULT_PACKAGE_ROOT = Path("paper_data/preprint/s4_goal_cliff_propagation_20260614")
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_ROOT / "old_mechanism_decoder"


def _trace_roots(package_root: Path) -> list[tuple[int, Path]]:
    inventory_path = package_root / "tables" / "trace_inventory.csv"
    inventory = pd.read_csv(inventory_path)
    roots: list[tuple[int, Path]] = []
    for _, row in inventory.sort_values("plateau_width", kind="stable").iterrows():
        root = Path(str(row["assay_root"]))
        if not root.exists():
            raise FileNotFoundError(root)
        roots.append((int(row["plateau_width"]), root))
    return roots


def _load_expanded_trials(
    *,
    package_root: Path,
    future_window: int,
    min_displacement: float,
    persistence_steps: int,
    persistence_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[int, int]]:
    trials: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    robot_shape: tuple[int, int] | None = None
    for plateau_width, trace_root in _trace_roots(package_root):
        loaded_trials, loaded_meta, loaded_shape = old._load_band_trials(
            trace_root=trace_root,
            future_window=int(future_window),
            min_displacement=float(min_displacement),
            persistence_steps=int(persistence_steps),
            persistence_fraction=float(persistence_fraction),
        )
        robot_shape = loaded_shape if robot_shape is None else robot_shape
        for trial, meta in zip(loaded_trials, loaded_meta, strict=True):
            unique_id = f"pw{int(plateau_width):02d}_{trial['trial_id']}"
            trial = dict(trial)
            trial["raw_trial_id"] = str(trial["trial_id"])
            trial["trial_id"] = unique_id
            trial["plateau_width"] = int(plateau_width)
            meta = dict(meta)
            meta["raw_trial_id"] = str(meta["trial_id"])
            meta["trial_id"] = unique_id
            meta["plateau_width"] = int(plateau_width)
            trials.append(trial)
            meta_rows.append(meta)
    if robot_shape is None:
        raise ValueError(f"No traces found under {package_root}")
    return trials, meta_rows, robot_shape


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _focus_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(summary_rows)
    if frame.empty:
        return []
    focus = frame[
        (frame["window"] == "edge_impulse")
        & (frame["lag"] == 8)
        & (frame["band"].isin(["frontier", "middle", "rear"]))
    ].copy()
    focus = focus.sort_values(["stream", "band"], kind="stable")
    return focus.to_dict("records")


def _write_summary_note(
    output_dir: Path,
    *,
    package_root: Path,
    trials: list[dict[str, Any]],
    meta_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    onset_rows: list[dict[str, Any]],
    regression_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    focus = pd.DataFrame(_focus_rows(summary_rows))
    lines: list[str] = [
        "# Earlier S4 band decoder on expanded goal-cliff traces",
        "",
        "This package reruns the earlier S4 mechanism packet's band/window decoder on the 30-event goal-cliff propagation expansion.",
        "",
        "Important scope note: the expanded traces are one-sided right-goal episodes. All included events reach the left cliff and return rightward, so this analysis tests sample-level future movement/return phase, not an event-level sustained-reversal success-versus-failure outcome.",
        "",
        f"- Source package: `{package_root}`",
        f"- Trial/event count: `{len(trials)}`",
        f"- Future window: `{int(args.future_window)}` steps",
        f"- Persistence: `{int(args.persistence_steps)}` steps at `{float(args.persistence_fraction):.2f}`",
        f"- Lags: `{','.join(str(int(v)) for v in args.lags)}`",
        f"- Permutations: `{int(args.permutations)}`",
        "",
        "## Edge-Impulse Lag-8 Summary",
        "",
    ]
    if focus.empty:
        lines.append("No edge-impulse lag-8 summary rows were estimable.")
    else:
        for _, row in focus.iterrows():
            lines.append(
                "- "
                f"{row['stream']} {row['band']}: "
                f"AUC {float(row['mean_auc']):.3f}, "
                f"balanced accuracy {float(row['mean_balanced_accuracy']):.3f}, "
                f"actual-minus-null AUC {float(row['mean_actual_minus_null_auc']):.3f}, "
                f"held-out trials {int(row['heldout_trial_count'])}, "
                f"samples {int(row['sample_count'])}."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If signal AUC is high here, it should be interpreted as internal controller state predicting near-future locomotor/return phase in an all-success one-sided expansion. It should not be read as evidence that signalling alone predicts sustained reversal outcome.",
            "",
            "For sustained reversal, a different event-level target is needed: reversed inward and cleared the edge zone versus fall, stall, continue outward, or fail to clear. This 30-event right-goal expansion has no negative event outcomes for that classifier.",
            "",
            "## Outputs",
            "",
            "- `expanded_band_predictivity_trial_metadata.csv`",
            "- `expanded_band_predictivity_decoder_metrics.csv`",
            "- `expanded_band_predictivity_by_window.csv`",
            "- `expanded_band_predictivity_onsets.csv`",
            "- `expanded_band_predictivity_onset_regression.csv`",
            "- `expanded_decoder_controls.csv`",
            "- `expanded_band_predictivity_lag8.png/pdf`",
            "",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text("\n".join(lines))


def run(args: argparse.Namespace) -> dict[str, Any]:
    package_root = args.package_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trials, meta_rows, robot_shape = _load_expanded_trials(
        package_root=package_root,
        future_window=int(args.future_window),
        min_displacement=float(args.min_displacement),
        persistence_steps=int(args.persistence_steps),
        persistence_fraction=float(args.persistence_fraction),
    )
    metric_rows = old._evaluate_band_decoders(
        trials,
        lags=[int(value) for value in args.lags],
        ridge_alpha=float(args.ridge_alpha),
        permutations=int(args.permutations),
        null_block_size=int(args.null_block_size),
        random_seed=int(args.random_seed),
    )
    summary_rows = old._aggregate_decoder_metrics(metric_rows)
    onset_rows, regression_rows = old._build_onset_latency_rows(summary_rows)
    control_rows = old._frontier_control_rows(metric_rows=metric_rows, summary_rows=summary_rows, lag=8)

    old._write_csv(output_dir / "expanded_band_predictivity_trial_metadata.csv", meta_rows)
    old._write_csv(output_dir / "expanded_band_predictivity_decoder_metrics.csv", metric_rows)
    old._write_csv(output_dir / "expanded_band_predictivity_by_window.csv", summary_rows)
    old._write_csv(output_dir / "expanded_band_predictivity_onsets.csv", onset_rows)
    old._write_csv(output_dir / "expanded_band_predictivity_onset_regression.csv", regression_rows)
    old._write_csv(output_dir / "expanded_decoder_controls.csv", control_rows)
    old._plot_band_predictivity(output_dir, summary_rows, lag=8)
    generated_png = output_dir / "s4_band_predictivity_lag8.png"
    generated_pdf = output_dir / "s4_band_predictivity_lag8.pdf"
    if generated_png.exists():
        generated_png.rename(output_dir / "expanded_band_predictivity_lag8.png")
    if generated_pdf.exists():
        generated_pdf.rename(output_dir / "expanded_band_predictivity_lag8.pdf")

    _write_summary_note(
        output_dir,
        package_root=package_root,
        trials=trials,
        meta_rows=meta_rows,
        summary_rows=summary_rows,
        control_rows=control_rows,
        onset_rows=onset_rows,
        regression_rows=regression_rows,
        args=args,
    )
    summary = {
        "schema_version": 1,
        "kind": "s4_goal_cliff_old_decoder_on_expansion",
        "package_root": str(package_root),
        "output_dir": str(output_dir),
        "trial_count": int(len(trials)),
        "robot_shape": list(robot_shape),
        "label_policy": {
            "future_window": int(args.future_window),
            "min_displacement": float(args.min_displacement),
            "persistence_steps": int(args.persistence_steps),
            "persistence_fraction": float(args.persistence_fraction),
        },
        "decoder_policy": {
            "lags": [int(value) for value in args.lags],
            "ridge_alpha": float(args.ridge_alpha),
            "permutations": int(args.permutations),
            "null_block_size": int(args.null_block_size),
            "random_seed": int(args.random_seed),
        },
        "outputs": {
            "trial_metadata": str(output_dir / "expanded_band_predictivity_trial_metadata.csv"),
            "decoder_metrics": str(output_dir / "expanded_band_predictivity_decoder_metrics.csv"),
            "summary": str(output_dir / "expanded_band_predictivity_by_window.csv"),
            "controls": str(output_dir / "expanded_decoder_controls.csv"),
            "readme": str(output_dir / "README.md"),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--future-window", type=int, default=32)
    parser.add_argument("--min-displacement", type=float, default=0.025)
    parser.add_argument("--persistence-steps", type=int, default=12)
    parser.add_argument("--persistence-fraction", type=float, default=0.75)
    parser.add_argument("--lags", type=old._parse_lags, default=old._parse_lags("0,8,16"))
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--permutations", type=int, default=30)
    parser.add_argument("--null-block-size", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=137)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
