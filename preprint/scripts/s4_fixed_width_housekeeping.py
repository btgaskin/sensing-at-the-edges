#!/usr/bin/env python
"""Final housekeeping analyses for the canonical S4 fixed-width packet.

This script assumes the A100 fixed-width traces have been pulled into
paper_data/preprint/s4_fixed_width_distance_sweep_20260618/modal/a100-40gb.
It adds two small paper-facing outputs:

- per-channel signalling decoders for sustained post-release direction;
- a per-event COM flip table for the supplement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import s4_fixed_width_distance_sweep as fixed  # noqa: E402


REL_TIMES = fixed.PROPAGATION_REL_TIMES
WINDOWS = (32, 128, 336)


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


def _signal_channel_specs(events: list[dict[str, Any]]) -> list[tuple[str, int, int]]:
    first = events[0]
    trace = first["trace"]
    hidden_start = int(first["hidden_start_channel"])
    nca_channels = int(trace["phase_nca_state"].shape[-1]) - hidden_start
    ctrnn_units = int(trace["phase_ctrnn_carry"].shape[-1])
    specs: list[tuple[str, int, int]] = []
    for channel in range(nca_channels):
        specs.append(("nca_hidden", int(channel), hidden_start + int(channel)))
    for unit in range(ctrnn_units):
        specs.append(("ctrnn_carry", int(unit), int(unit)))
    return specs


def _signal_channel_feature(
    event: dict[str, Any],
    *,
    channel_kind: str,
    absolute_index: int,
    cell: int,
    step: int,
) -> np.ndarray | None:
    if step < 0 or step >= int(event["step_count"]):
        return None
    rows, cols = event["robot_shape"]
    row = int(cell) // int(cols)
    col = int(cell) % int(cols)
    trace = event["trace"]
    if channel_kind == "nca_hidden":
        values = trace["phase_nca_state"][int(step), :, row, col, int(absolute_index)]
    elif channel_kind == "ctrnn_carry":
        values = trace["phase_ctrnn_carry"][int(step), :, row, col, int(absolute_index)]
    else:
        raise ValueError(f"Unsupported signal channel kind: {channel_kind}")
    feature = np.asarray(values, dtype=np.float32).reshape(-1)
    feature[np.abs(feature) < fixed.PROPAGATION_FEATURE_DUST_FLOOR] = 0.0
    return feature


def _decode_samples(samples: list[dict[str, Any]], *, rng: np.random.Generator, permutations: int) -> dict[str, Any]:
    y = np.asarray([sample["label"] for sample in samples], dtype=np.int8)
    scores = fixed._scores_leave_one_source_out_no_intercept(samples)
    if scores is None or not np.all(np.isfinite(scores)):
        return {
            "auc": float("nan"),
            "balanced_accuracy": float("nan"),
            "null_auc_q95": float("nan"),
            "one_sided_p": float("nan"),
            "estimable": False,
        }
    auc = fixed._auc(y, scores)
    balanced = fixed._balanced_accuracy(y, scores)
    null = np.asarray([fixed._auc(rng.permutation(y), scores) for _ in range(int(permutations))], dtype=np.float32)
    return {
        "auc": float(auc),
        "balanced_accuracy": float(balanced),
        "null_auc_q95": float(np.quantile(null, 0.95)),
        "one_sided_p": float((np.sum(null >= auc) + 1.0) / (len(null) + 1.0)),
        "estimable": True,
    }


def signal_channel_decoder_rows(
    events: list[dict[str, Any]],
    *,
    random_seed: int,
    permutations: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(random_seed))
    direction_by_event = {event["event_id"]: fixed._edge_oriented_committed_labels(event) for event in events}
    rank_count = min(int(event["cell_count"]) for event in events)
    specs = _signal_channel_specs(events)
    rows: list[dict[str, Any]] = []
    for channel_kind, channel_index, absolute_index in specs:
        for rank in range(rank_count):
            for rel_time in REL_TIMES:
                samples: list[dict[str, Any]] = []
                for event in events:
                    cell = event["rank_to_cell"].get(int(rank))
                    if cell is None:
                        continue
                    sample_step = int(event["release_step"]) + int(rel_time)
                    labels, deltas = direction_by_event[event["event_id"]]
                    if sample_step < 0 or sample_step >= len(labels):
                        continue
                    label = int(labels[sample_step])
                    if label not in {-1, 1}:
                        continue
                    feature = _signal_channel_feature(
                        event,
                        channel_kind=channel_kind,
                        absolute_index=absolute_index,
                        cell=int(cell),
                        step=sample_step,
                    )
                    if feature is None or not np.all(np.isfinite(feature)):
                        continue
                    samples.append(
                        {
                            "source_event_id": str(event["event_id"]),
                            "feature": feature,
                            "label": label,
                        }
                    )
                decoded = _decode_samples(samples, rng=rng, permutations=permutations)
                y = np.asarray([sample["label"] for sample in samples], dtype=np.int8)
                rows.append(
                    {
                        "channel_kind": channel_kind,
                        "channel_index": int(channel_index),
                        "absolute_channel_index": int(absolute_index),
                        "rank_from_edge": int(rank),
                        "rel_time": int(rel_time),
                        "sample_count": int(len(samples)),
                        "positive_count": int(np.sum(y == 1)) if y.size else 0,
                        "negative_count": int(np.sum(y == -1)) if y.size else 0,
                        "source_event_count": int(len({sample["source_event_id"] for sample in samples})),
                        "feature_dim": int(samples[0]["feature"].shape[0]) if samples else 0,
                        "feature_kind": "single_signal_channel_across_three_sense_act_phases",
                        "classifier": "leave-one-event-out ridge without intercept",
                        "direction_label": "positive=away_from_cliff; negative=toward_cliff",
                        "future_window": int(fixed.PROPAGATION_DIRECTION_FUTURE_WINDOW),
                        "min_displacement": float(fixed.PROPAGATION_DIRECTION_MIN_DISPLACEMENT),
                        "persistence_steps": int(fixed.PROPAGATION_DIRECTION_PERSISTENCE_STEPS),
                        "persistence_fraction": float(fixed.PROPAGATION_DIRECTION_PERSISTENCE_FRACTION),
                        **decoded,
                    }
                )
    return rows


def summarize_signal_channels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    summary: list[dict[str, Any]] = []
    for (kind, index), group in frame.groupby(["channel_kind", "channel_index"], sort=True):
        estimable = group[group["estimable"].astype(bool)].copy()
        if estimable.empty:
            summary.append(
                {
                    "channel_kind": str(kind),
                    "channel_index": int(index),
                    "estimable_rows": 0,
                    "max_auc": float("nan"),
                    "max_auc_rank_from_edge": "",
                    "max_auc_rel_time": "",
                    "max_balanced_accuracy": float("nan"),
                    "min_one_sided_p": float("nan"),
                    "significant_rows_p_le_0_05": 0,
                    "significant_rows_auc_gt_null_q95": 0,
                }
            )
            continue
        best = estimable.sort_values(["auc", "balanced_accuracy"], ascending=False).iloc[0]
        summary.append(
            {
                "channel_kind": str(kind),
                "channel_index": int(index),
                "estimable_rows": int(len(estimable)),
                "max_auc": float(best["auc"]),
                "max_auc_rank_from_edge": int(best["rank_from_edge"]),
                "max_auc_rel_time": int(best["rel_time"]),
                "max_auc_sample_count": int(best["sample_count"]),
                "max_auc_positive_count": int(best["positive_count"]),
                "max_auc_negative_count": int(best["negative_count"]),
                "max_balanced_accuracy": float(best["balanced_accuracy"]),
                "min_one_sided_p": float(estimable["one_sided_p"].min()),
                "median_auc": float(estimable["auc"].median()),
                "mean_auc": float(estimable["auc"].mean()),
                "significant_rows_p_le_0_05": int(np.sum(estimable["one_sided_p"].to_numpy(dtype=np.float64) <= 0.05)),
                "significant_rows_auc_gt_null_q95": int(
                    np.sum(estimable["auc"].to_numpy(dtype=np.float64) > estimable["null_auc_q95"].to_numpy(dtype=np.float64))
                ),
            }
        )
    return summary


def _first_sustained_away_step(oriented_com: np.ndarray, release_step: int) -> int | None:
    threshold = float(fixed.PROPAGATION_DIRECTION_MIN_DISPLACEMENT)
    persistence = int(fixed.PROPAGATION_DIRECTION_PERSISTENCE_STEPS)
    fraction = float(fixed.PROPAGATION_DIRECTION_PERSISTENCE_FRACTION)
    baseline = float(oriented_com[int(release_step)])
    for step in range(int(release_step), len(oriented_com)):
        stop = min(len(oriented_com), step + persistence)
        if stop <= step:
            continue
        ok = oriented_com[step:stop] - baseline >= threshold
        if float(np.mean(ok)) >= fraction:
            return int(step - int(release_step))
    return None


def com_flip_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        com_x = fixed._com_x_series(event)
        oriented = com_x * float(fixed._edge_away_sign(event))
        release = int(event["release_step"])
        labels, deltas = fixed._edge_oriented_committed_labels(event)
        base = float(oriented[release])
        pre_start = max(0, release - 128)
        row: dict[str, Any] = {
            "event_id": event["event_id"],
            "trial_id": event["trial_id"],
            "cliff_distance": int(event["cliff_distance"]),
            "goal_distance": int(event["goal_distance"]),
            "cliff_side": event["cliff_side"],
            "release_step": release,
            "step_count": int(event["step_count"]),
            "success": bool(event["success"]),
            "returned_from_edge": bool(event["returned_from_edge"]),
            "cliff_fell": bool(event["cliff_fell"]),
            "oriented_com_at_release": base,
            "pre_delta_128": float(oriented[release] - oriented[pre_start]),
            "pre_cliffward_128": bool(oriented[release] - oriented[pre_start] <= -float(fixed.PROPAGATION_DIRECTION_MIN_DISPLACEMENT)),
            "first_sustained_away_rel_step": _first_sustained_away_step(oriented, release),
        }
        for window in WINDOWS:
            index = min(len(oriented) - 1, release + int(window))
            post_delta = float(oriented[index] - base)
            label_at_window = int(labels[index]) if 0 <= index < len(labels) else 0
            future_delta_at_window = float(deltas[index]) if 0 <= index < len(deltas) else float("nan")
            row[f"post_delta_{window}"] = post_delta
            row[f"away_by_{window}"] = bool(post_delta >= float(fixed.PROPAGATION_DIRECTION_MIN_DISPLACEMENT))
            row[f"toward_by_{window}"] = bool(post_delta <= -float(fixed.PROPAGATION_DIRECTION_MIN_DISPLACEMENT))
            row[f"direction_label_at_plus_{window}"] = label_at_window
            row[f"future_delta_at_plus_{window}"] = future_delta_at_window
        rows.append(row)
    return rows


def summarize_com(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    summary: dict[str, Any] = {
        "event_count": int(len(frame)),
        "cliff_distance_min": int(frame["cliff_distance"].min()),
        "cliff_distance_max": int(frame["cliff_distance"].max()),
        "success_count": int(frame["success"].sum()),
        "returned_from_edge_count": int(frame["returned_from_edge"].sum()),
        "fall_count": int(frame["cliff_fell"].sum()),
        "pre_cliffward_128_count": int(frame["pre_cliffward_128"].sum()),
    }
    for window in WINDOWS:
        summary[f"away_by_{window}_count"] = int(frame[f"away_by_{window}"].sum())
        summary[f"toward_by_{window}_count"] = int(frame[f"toward_by_{window}"].sum())
        summary[f"post_delta_{window}_mean"] = float(frame[f"post_delta_{window}"].mean())
        summary[f"post_delta_{window}_median"] = float(frame[f"post_delta_{window}"].median())
        summary[f"post_delta_{window}_min"] = float(frame[f"post_delta_{window}"].min())
        summary[f"post_delta_{window}_max"] = float(frame[f"post_delta_{window}"].max())
    rel = frame["first_sustained_away_rel_step"].dropna().to_numpy(dtype=np.float64)
    summary["first_sustained_away_count"] = int(rel.size)
    if rel.size:
        summary["first_sustained_away_median"] = float(np.median(rel))
        summary["first_sustained_away_min"] = float(np.min(rel))
        summary["first_sustained_away_max"] = float(np.max(rel))
    return summary


def write_report(
    *,
    events: list[dict[str, Any]],
    channel_summary: list[dict[str, Any]],
    com_summary: dict[str, Any],
    permutations: int,
) -> Path:
    out = fixed.PACKAGE_ROOT / "S4_signal_channel_and_com_flip_report.md"
    channel_frame = pd.DataFrame(channel_summary)
    lines = [
        "# S4 signalling-channel and COM flip housekeeping",
        "",
        "## Inputs",
        "",
        f"- Packet root: `{fixed.PACKAGE_ROOT}`",
        f"- Data root: `{fixed._data_root()}`",
        f"- Events loaded: `{len(events)}`",
        "- Condition: deformation off, intact signalling, fixed-width A100 packet.",
        f"- Decoder permutations: `{int(permutations)}`",
        f"- Direction label: positive = away from cliff; negative = toward cliff.",
        "",
        "## Signalling-channel decoder",
        "",
        "- Each NCA hidden channel and CTRNN carry unit is decoded separately.",
        "- Feature per sample is the single channel/unit value across the three sense-act phases for one cell/rank at one event-relative time.",
        "- Cross-validation holds out whole cliff-distance events.",
        "- Body, action, geometry, COM, terrain, and explicit cliff-position variables are excluded.",
        "",
    ]
    if channel_frame.empty:
        lines.append("- No channel rows were produced.")
    else:
        best = channel_frame.sort_values(["max_auc", "max_balanced_accuracy"], ascending=False).head(8)
        lines.append("| Channel | Max AUC | Balanced acc. | n (+/-) | Rank | Rel. time | min p | Sig. rows |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in best.iterrows():
            lines.append(
                f"| {row['channel_kind']} {int(row['channel_index'])} | "
                f"{float(row['max_auc']):.3f} | {float(row['max_balanced_accuracy']):.3f} | "
                f"{int(row['max_auc_sample_count'])} ({int(row['max_auc_positive_count'])}/{int(row['max_auc_negative_count'])}) | "
                f"{row['max_auc_rank_from_edge']} | {row['max_auc_rel_time']} | "
                f"{float(row['min_one_sided_p']):.4f} | {int(row['significant_rows_p_le_0_05'])} |"
            )
        total_rows = int(channel_frame["estimable_rows"].sum())
        minority_rows = pd.read_csv(fixed.PACKAGE_ROOT / "tables" / "s4_signal_channel_direction_decoder.csv")
        minority_rows = minority_rows[minority_rows["estimable"].astype(bool)]
        if not minority_rows.empty:
            minority = np.minimum(
                minority_rows["positive_count"].to_numpy(dtype=np.int32),
                minority_rows["negative_count"].to_numpy(dtype=np.int32),
            )
            lines.extend(
                [
                    "",
                    f"- Estimable channel/rank/time rows: `{total_rows}`.",
                    f"- Minority-class count across estimable rows: median `{float(np.median(minority)):.0f}`, max `{int(np.max(minority))}`.",
                    "- These are raw signalling-state decoders. They can capture post-release locomotor phase/timing information and should not be read as a release-response decoder.",
                    "- The matched release-versus-continuing analyses remain the relevant evidence for whether the signalling stream changes in response to support loss.",
                ]
            )
    lines.extend(
        [
            "",
            "## COM directional flip",
            "",
            f"- Events: `{com_summary.get('event_count', 0)}`",
            f"- Successes: `{com_summary.get('success_count', 0)}/{com_summary.get('event_count', 0)}`",
            f"- Returned from edge: `{com_summary.get('returned_from_edge_count', 0)}/{com_summary.get('event_count', 0)}`",
            f"- Falls: `{com_summary.get('fall_count', 0)}/{com_summary.get('event_count', 0)}`",
            f"- Pre-release 128-step cliffward approaches: `{com_summary.get('pre_cliffward_128_count', 0)}/{com_summary.get('event_count', 0)}`",
        ]
    )
    for window in WINDOWS:
        lines.append(
            f"- Away by +{window}: `{com_summary.get(f'away_by_{window}_count', 0)}/{com_summary.get('event_count', 0)}`; "
            f"post-delta mean/median/range = "
            f"{com_summary.get(f'post_delta_{window}_mean', float('nan')):.3f} / "
            f"{com_summary.get(f'post_delta_{window}_median', float('nan')):.3f} / "
            f"[{com_summary.get(f'post_delta_{window}_min', float('nan')):.3f}, "
            f"{com_summary.get(f'post_delta_{window}_max', float('nan')):.3f}]"
        )
    if com_summary.get("first_sustained_away_count", 0):
        lines.append(
            f"- First sustained away step: `{com_summary['first_sustained_away_count']}/{com_summary.get('event_count', 0)}` events; "
            f"median/range = {com_summary['first_sustained_away_median']:.0f} "
            f"[{com_summary['first_sustained_away_min']:.0f}, {com_summary['first_sustained_away_max']:.0f}] steps after release."
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            "These outputs are housekeeping/supporting analyses. The channel decoder tests whether individual carried signalling variables linearly predict the release-locked future direction; it is not a causal lesion. The COM table verifies the physical return trajectory event by event and should be compared with the preserved-support forced-command control.",
            "",
            "## Files",
            "",
            "- `tables/s4_signal_channel_direction_decoder.csv`",
            "- `tables/s4_signal_channel_direction_decoder_summary.csv`",
            "- `tables/s4_com_directional_flip_by_event.csv`",
            "- `tables/s4_signal_channel_and_com_flip_summary.json`",
        ]
    )
    out.write_text("\n".join(lines) + "\n")
    return out


def run(*, random_seed: int, permutations: int) -> dict[str, Path]:
    events, _robot_shape, _hidden_start = fixed._load_events()
    tables = fixed.PACKAGE_ROOT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    channel_rows = signal_channel_decoder_rows(events, random_seed=random_seed, permutations=permutations)
    channel_summary = summarize_signal_channels(channel_rows)
    com_rows = com_flip_rows(events)
    com_summary = summarize_com(com_rows)

    channel_path = tables / "s4_signal_channel_direction_decoder.csv"
    channel_summary_path = tables / "s4_signal_channel_direction_decoder_summary.csv"
    com_path = tables / "s4_com_directional_flip_by_event.csv"
    summary_path = tables / "s4_signal_channel_and_com_flip_summary.json"

    pd.DataFrame(channel_rows).to_csv(channel_path, index=False)
    pd.DataFrame(channel_summary).to_csv(channel_summary_path, index=False)
    pd.DataFrame(com_rows).to_csv(com_path, index=False)
    _write_json(
        summary_path,
        {
            "schema_version": 1,
            "kind": "s4_signal_channel_and_com_flip_summary",
            "packet_root": fixed.PACKAGE_ROOT,
            "data_root": fixed._data_root(),
            "event_count": len(events),
            "random_seed": int(random_seed),
            "permutations": int(permutations),
            "signal_channel_count": len(_signal_channel_specs(events)),
            "signal_decoder_rows": len(channel_rows),
            "signal_channel_summary": channel_summary,
            "com_summary": com_summary,
            "paths": {
                "signal_channel_decoder": channel_path,
                "signal_channel_summary": channel_summary_path,
                "com_flip_by_event": com_path,
                "report": fixed.PACKAGE_ROOT / "S4_signal_channel_and_com_flip_report.md",
            },
        },
    )
    report_path = write_report(
        events=events,
        channel_summary=channel_summary,
        com_summary=com_summary,
        permutations=permutations,
    )
    return {
        "signal_channel_decoder": channel_path,
        "signal_channel_summary": channel_summary_path,
        "com_flip_by_event": com_path,
        "summary": summary_path,
        "report": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-seed", type=int, default=137)
    parser.add_argument("--permutations", type=int, default=fixed.PROPAGATION_PERMUTATIONS)
    args = parser.parse_args()
    for key, value in run(random_seed=int(args.random_seed), permutations=int(args.permutations)).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
