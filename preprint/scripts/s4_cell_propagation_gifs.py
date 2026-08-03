#!/usr/bin/env python
"""Render cell-wise S4 signal/body propagation GIFs from fixed-width tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import s4_fixed_width_distance_sweep as sweep


DEFAULT_PACKET_ROOT = Path("paper_data/preprint/s4_fixed_width_distance_sweep_20260618")
ROBOT_SHAPE = (2, 7)
REL_TIMES = tuple(range(-64, 513, 8))
DEFAULT_DURATION_MS = 120


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _cell_direction_decoder_rows(
    raw_rows: pd.DataFrame,
    *,
    random_seed: int,
    permutations: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(random_seed))
    rows: list[dict[str, Any]] = []
    for (stream, cell, rel_time), group in raw_rows.groupby(["stream", "cell", "rel_time"], sort=True):
        samples = []
        for row in group.to_dict("records"):
            label = int(row["direction_label"])
            if label not in {-1, 1}:
                continue
            feature = sweep._phase_response_feature(row)
            if not np.all(np.isfinite(feature)):
                continue
            samples.append(
                {
                    "source_event_id": str(row["event_id"]),
                    "feature": feature,
                    "label": label,
                }
            )
        y = np.asarray([sample["label"] for sample in samples], dtype=np.int8)
        scores = sweep._scores_leave_one_source_out_no_intercept(samples)
        auc = float("nan") if scores is None else sweep._auc(y, scores)
        balanced = float("nan") if scores is None else sweep._balanced_accuracy(y, scores)
        if scores is None or not np.isfinite(auc):
            null_q95 = float("nan")
            p_value = float("nan")
        else:
            null = np.asarray([sweep._auc(rng.permutation(y), scores) for _ in range(int(permutations))], dtype=np.float32)
            null_q95 = float(np.quantile(null, 0.95))
            p_value = float((np.sum(null >= auc) + 1.0) / (len(null) + 1.0))
        rows.append(
            {
                "stream": str(stream),
                "cell": int(cell),
                "rel_time": int(rel_time),
                "sample_count": int(len(samples)),
                "positive_count": int(np.sum(y == 1)) if y.size else 0,
                "negative_count": int(np.sum(y == -1)) if y.size else 0,
                "source_event_count": int(len({sample["source_event_id"] for sample in samples})),
                "auc": auc,
                "balanced_accuracy": balanced,
                "null_auc_q95": null_q95,
                "one_sided_p": p_value,
                "estimable": bool(scores is not None and np.isfinite(auc)),
                "feature_kind": "phase_response",
                "classifier": "leave-one-event-out ridge without intercept",
                "feature_dust_floor": sweep.PROPAGATION_FEATURE_DUST_FLOOR,
            }
        )
    return rows


def _cell_slip_rows(raw_rows: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (stream, cell, rel_time), group in raw_rows.groupby(["stream", "cell", "rel_time"], sort=True):
        rows.append(
            {
                "stream": str(stream),
                "cell": int(cell),
                "rel_time": int(rel_time),
                "n": int(len(group)),
                "event_count": int(group["event_id"].nunique()),
                "mean_abs_phase_slip": float(group["abs_phase_slip"].mean()),
                "median_abs_phase_slip": float(group["abs_phase_slip"].median()),
                "mean_pc_l2_delta": float(group["pc_l2_delta"].mean()),
                "median_pc_l2_delta": float(group["pc_l2_delta"].median()),
                "mean_radius_delta": float(group["radius_delta"].mean()),
                "away_label_count": int(np.sum(group["direction_label"].to_numpy(dtype=np.int8) == 1)),
                "toward_label_count": int(np.sum(group["direction_label"].to_numpy(dtype=np.int8) == -1)),
                "uncommitted_label_count": int(np.sum(group["direction_label"].to_numpy(dtype=np.int8) == 0)),
            }
        )
    return rows


def _grid_from_rows(
    frame: pd.DataFrame,
    *,
    stream: str,
    rel_time: int,
    value_col: str,
    fill_value: float | None = None,
) -> np.ndarray:
    grid = np.full(ROBOT_SHAPE, np.nan, dtype=np.float32)
    subset = frame[(frame["stream"] == stream) & (frame["rel_time"] == int(rel_time))]
    for _, row in subset.iterrows():
        cell = int(row["cell"])
        rr, cc = divmod(cell, ROBOT_SHAPE[1])
        value = row[value_col]
        if pd.isna(value):
            value = fill_value if fill_value is not None else np.nan
        grid[rr, cc] = float(value)
    return grid


def _add_timeline(ax: plt.Axes, *, rel_time: int, start: int, stop: int) -> None:
    if int(stop) <= int(start):
        return
    event_x = (0.0 - float(start)) / (float(stop) - float(start))
    center_x = (float(rel_time) - float(start)) / (float(stop) - float(start))
    event_x = float(np.clip(event_x, 0.0, 1.0))
    center_x = float(np.clip(center_x, 0.0, 1.0))
    y = -0.34
    ax.plot([0.0, 1.0], [y, y], transform=ax.transAxes, color="0.55", linewidth=1.0, clip_on=False)
    ax.plot(
        [event_x, event_x],
        [y - 0.05, y + 0.05],
        transform=ax.transAxes,
        color="#D4736C",
        linewidth=2.0,
        clip_on=False,
    )
    ax.scatter([center_x], [y], transform=ax.transAxes, s=24, color="black", clip_on=False, zorder=5)
    ax.text(event_x, y - 0.08, "release", transform=ax.transAxes, ha="center", va="top", color="#D4736C", fontsize=7)
    ax.text(center_x, y + 0.06, f"{int(rel_time):+d}", transform=ax.transAxes, ha="center", va="bottom", color="black", fontsize=7)


def _plot_comparison_frame(
    *,
    frame: pd.DataFrame,
    left_stream: str,
    right_stream: str,
    left_title: str,
    right_title: str,
    value_col: str,
    rel_time: int,
    vmin: float,
    vmax: float,
    cmap: str,
    fill_value: float | None,
    suptitle: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.05))
    image = None
    for ax, stream, title in zip(axes, (left_stream, right_stream), (left_title, right_title), strict=True):
        grid = _grid_from_rows(frame, stream=stream, rel_time=int(rel_time), value_col=value_col, fill_value=fill_value)
        image = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        for rr in range(ROBOT_SHAPE[0]):
            for cc in range(ROBOT_SHAPE[1]):
                cell = rr * ROBOT_SHAPE[1] + cc
                value = grid[rr, cc]
                text = f"{cell}\n{value:.2f}" if np.isfinite(value) else f"{cell}\nNA"
                if np.isfinite(value) and vmax > vmin:
                    norm = (float(value) - float(vmin)) / (float(vmax) - float(vmin))
                    color = "#111111" if norm >= 0.72 else "white"
                else:
                    color = "white"
                ax.text(cc, rr, text, ha="center", va="center", color=color, fontsize=7, weight="bold")
        ax.axvline(-0.5, color="#D4736C", linewidth=2.0)
        ax.set_title(title, fontsize=10, weight="bold")
        ax.set_xticks(range(ROBOT_SHAPE[1]))
        ax.set_yticks(range(ROBOT_SHAPE[0]))
        ax.set_xlabel("cell column", fontsize=8)
        ax.set_ylabel("physical row, bottom=0", fontsize=8)
        ax.tick_params(labelsize=7)
        _add_timeline(ax, rel_time=int(rel_time), start=REL_TIMES[0], stop=REL_TIMES[-1])
    fig.suptitle(f"{suptitle} | t={int(rel_time):+d}", fontsize=11, weight="bold", y=0.96)
    fig.subplots_adjust(left=0.055, right=0.88, bottom=0.36, top=0.82, wspace=0.15)
    if image is not None:
        fig.colorbar(image, ax=axes, fraction=0.046, pad=0.035, label=value_col)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _clear_frames(frames_dir: Path) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames_dir.glob("frame_*.png"):
        frame.unlink()


def _gif_from_frames(frame_paths: list[Path], gif_path: Path, *, duration_ms: int) -> None:
    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frame_paths]
    if not images:
        return
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=int(duration_ms),
        loop=0,
        optimize=False,
    )
    for image in images:
        image.close()


def _render_gif(
    *,
    frame: pd.DataFrame,
    output_dir: Path,
    stem: str,
    left_stream: str,
    right_stream: str,
    left_title: str,
    right_title: str,
    value_col: str,
    vmin: float,
    vmax: float,
    cmap: str,
    fill_value: float | None,
    suptitle: str,
    duration_ms: int,
) -> dict[str, str]:
    frames_dir = output_dir / f"{stem}_frames"
    _clear_frames(frames_dir)
    frame_paths: list[Path] = []
    for rel_time in REL_TIMES:
        frame_path = frames_dir / f"frame_{int(rel_time):+05d}.png"
        _plot_comparison_frame(
            frame=frame,
            left_stream=left_stream,
            right_stream=right_stream,
            left_title=left_title,
            right_title=right_title,
            value_col=value_col,
            rel_time=int(rel_time),
            vmin=float(vmin),
            vmax=float(vmax),
            cmap=cmap,
            fill_value=fill_value,
            suptitle=suptitle,
            output_path=frame_path,
        )
        frame_paths.append(frame_path)
    gif_path = output_dir / f"{stem}.gif"
    _gif_from_frames(frame_paths, gif_path, duration_ms=int(duration_ms))
    return {"gif": str(gif_path.resolve()), "frames": str(frames_dir.resolve())}


def analyze(
    *,
    packet_root: Path,
    duration_ms: int,
    permutations: int,
    random_seed: int,
) -> dict[str, Any]:
    tables_dir = packet_root / "tables"
    figures_dir = packet_root / "figures"
    raw_path = tables_dir / "phase_propagation_local_slip_by_event.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing phase propagation by-event table: {raw_path}")
    raw = pd.read_csv(raw_path)

    cell_slip_rows = _cell_slip_rows(raw)
    cell_decoder_rows = _cell_direction_decoder_rows(
        raw,
        random_seed=sweep._stable_seed(int(random_seed), "cell-propagation-gifs"),
        permutations=int(permutations),
    )
    slip_path = tables_dir / "phase_propagation_cell_slip.csv"
    decoder_path = tables_dir / "phase_propagation_cell_direction_decoder.csv"
    _write_csv(slip_path, cell_slip_rows)
    _write_csv(decoder_path, cell_decoder_rows)

    slip_frame = pd.DataFrame(cell_slip_rows)
    decoder_frame = pd.DataFrame(cell_decoder_rows)
    body_slip_values = slip_frame[
        slip_frame["stream"].isin(["body_no_contractile", "spring_current", "spring_contractile"])
    ]["mean_abs_phase_slip"].replace([np.inf, -np.inf], np.nan).dropna()
    slip_vmax = float(body_slip_values.quantile(0.98)) if not body_slip_values.empty else 1.0
    slip_vmax = max(slip_vmax, sweep.PROPAGATION_PHASE_ONSET_MIN_RAD)

    outputs = {
        "signal_vs_body_direction_auc": _render_gif(
            frame=decoder_frame,
            output_dir=figures_dir,
            stem="signal_vs_body_cell_direction_auc_fixed_width",
            left_stream="signal_hidden",
            right_stream="body_no_contractile",
            left_title="hidden/internal signalling",
            right_title="body: spring length + kinematics",
            value_col="auc",
            vmin=0.5,
            vmax=1.0,
            cmap="viridis",
            fill_value=None,
            suptitle="Sustained-direction AUC from local phase response",
            duration_ms=int(duration_ms),
        ),
        "signal_vs_body_phase_slip": _render_gif(
            frame=slip_frame,
            output_dir=figures_dir,
            stem="signal_vs_body_cell_phase_slip_fixed_width",
            left_stream="signal_hidden",
            right_stream="body_no_contractile",
            left_title="hidden/internal signalling",
            right_title="body: spring length + kinematics",
            value_col="mean_abs_phase_slip",
            vmin=0.0,
            vmax=slip_vmax,
            cmap="magma",
            fill_value=0.0,
            suptitle="Local phase/deviation from continuing baseline",
            duration_ms=int(duration_ms),
        ),
        "signal_vs_spring_phase_slip": _render_gif(
            frame=slip_frame,
            output_dir=figures_dir,
            stem="signal_vs_spring_cell_phase_slip_fixed_width",
            left_stream="signal_hidden",
            right_stream="spring_current",
            left_title="hidden/internal signalling",
            right_title="spring length state",
            value_col="mean_abs_phase_slip",
            vmin=0.0,
            vmax=slip_vmax,
            cmap="magma",
            fill_value=0.0,
            suptitle="Local phase/deviation from continuing baseline",
            duration_ms=int(duration_ms),
        ),
    }

    summary = {
        "schema_version": 1,
        "kind": "s4_cell_propagation_gifs",
        "packet_root": str(packet_root.resolve()),
        "source_table": str(raw_path.resolve()),
        "robot_shape": list(ROBOT_SHAPE),
        "physical_cell_layout": ["7 8 9 10 11 12 13", "0 1 2 3 4 5 6"],
        "relative_times": list(REL_TIMES),
        "direction_auc_policy": {
            "label": "positive=away_from_cliff; negative=toward_cliff",
            "classifier": "leave-one-event-out ridge without intercept",
            "feature_kind": "phase_response",
            "feature_dust_floor": sweep.PROPAGATION_FEATURE_DUST_FLOOR,
            "permutations": int(permutations),
        },
        "render_policy": {
            "red_line": "left frontier/release edge",
            "red_timeline_tick": "release event",
            "timeline_dot": "current relative frame",
            "auc_scale": [0.5, 1.0],
            "phase_slip_scale": [0.0, slip_vmax],
        },
        "outputs": {
            "cell_slip_table": str(slip_path.resolve()),
            "cell_direction_decoder_table": str(decoder_path.resolve()),
            **outputs,
        },
    }
    summary_path = figures_dir / "s4_cell_propagation_gifs_summary.json"
    _write_json(summary_path, summary)
    summary["outputs"]["summary"] = str(summary_path.resolve())
    _write_report(packet_root=packet_root, summary=summary, decoder_frame=decoder_frame, slip_frame=slip_frame)
    return summary


def _best_stream_lines(decoder_frame: pd.DataFrame, slip_frame: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    post_decoder = decoder_frame[(decoder_frame["rel_time"] >= 0) & (decoder_frame["estimable"])]
    for stream in ("signal_hidden", "body_no_contractile", "spring_current", "spring_contractile"):
        group = post_decoder[post_decoder["stream"] == stream]
        if group.empty:
            lines.append(f"- `{stream}` direction AUC: no estimable rows.")
            continue
        best = group.sort_values("auc", ascending=False, kind="stable").iloc[0]
        sig = group[(group["auc"] > group["null_auc_q95"]) & (group["one_sided_p"] <= 0.05)]
        lines.append(
            "- "
            f"`{stream}` direction AUC: best {float(best['auc']):.3f} "
            f"at cell {int(best['cell'])}, t={int(best['rel_time']):+d}; "
            f"significant rows {len(sig)}/{len(group)}."
        )
    post_slip = slip_frame[slip_frame["rel_time"] >= 0]
    for stream in ("signal_hidden", "body_no_contractile", "spring_current", "spring_contractile"):
        group = post_slip[post_slip["stream"] == stream]
        if group.empty:
            lines.append(f"- `{stream}` phase slip: no rows.")
            continue
        best = group.sort_values("mean_abs_phase_slip", ascending=False, kind="stable").iloc[0]
        lines.append(
            "- "
            f"`{stream}` phase slip: max mean |slip| {float(best['mean_abs_phase_slip']):.3f} "
            f"at cell {int(best['cell'])}, t={int(best['rel_time']):+d}."
        )
    return lines


def _write_report(
    *,
    packet_root: Path,
    summary: dict[str, Any],
    decoder_frame: pd.DataFrame,
    slip_frame: pd.DataFrame,
) -> None:
    lines = [
        "# S4 cell-wise propagation GIFs",
        "",
        "## Purpose",
        "",
        "These animations replicate the useful visual style of the earlier signal/body susceptibility GIF on the larger fixed-width S4 deformation-off family. They are cell-wise, not rank-only: values are aggregated by physical cell id in the 7x2 body.",
        "",
        "## Source",
        "",
        f"- Packet root: `{packet_root}`",
        "- Source table: `tables/phase_propagation_local_slip_by_event.csv`",
        "- Event family: fixed-width S4 deformation-off, 30/30 cliff-release events.",
        "- Physical layout: top row `7 8 9 10 11 12 13`, bottom row `0 1 2 3 4 5 6`.",
        "",
        "## Outputs",
        "",
    ]
    outputs = summary["outputs"]
    for key in (
        "cell_slip_table",
        "cell_direction_decoder_table",
        "signal_vs_body_direction_auc",
        "signal_vs_body_phase_slip",
        "signal_vs_spring_phase_slip",
    ):
        value = outputs[key]
        if isinstance(value, dict):
            lines.append(f"- `{key}`: `{value['gif']}`")
        else:
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- The red vertical line marks the release/frontier edge; the red timeline tick marks release; the black dot marks the current relative frame.",
            "- Direction AUC uses sustained future COM motion away from vs toward the cliff, decoded from local phase-response features.",
            "- Phase-slip panels show mean absolute local phase/deviation from the still-continuing baseline.",
            "- Hidden/internal signalling is the negative control; body/spring streams are the mechanical readout.",
            "",
            "## Numeric Summary",
            "",
        ]
    )
    lines.extend(_best_stream_lines(decoder_frame, slip_frame))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These GIFs support the cell-wise version of the mechanism story: the hidden/internal signalling panel stays flat/chance, while body and spring cells show release-locked perturbation and direction-predictive structure. They should not be phrased as causal proof or as a clean front-to-back wave unless the separate onset-regression gate is satisfied.",
        ]
    )
    (packet_root / "S4_cell_propagation_gifs_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, default=DEFAULT_PACKET_ROOT)
    parser.add_argument("--duration-ms", type=int, default=DEFAULT_DURATION_MS)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=137)
    args = parser.parse_args()
    summary = analyze(
        packet_root=args.packet_root,
        duration_ms=int(args.duration_ms),
        permutations=int(args.permutations),
        random_seed=int(args.random_seed),
    )
    print(json.dumps(summary["outputs"], indent=2))


if __name__ == "__main__":
    main()
