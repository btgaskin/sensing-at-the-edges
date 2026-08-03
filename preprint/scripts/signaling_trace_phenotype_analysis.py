#!/usr/bin/env python
"""Trace-level phenotype analysis for native signalling ablations.

This consumes `state_space_trace` outputs, not the endpoint ablation manifest.
It separates:

- trajectory geometry: COM path tortuosity and heading reversals;
- contraction commands: horizontal/vertical drive and action-output rhythms;
- mechanical body state: velocities and spring strain.

The signalling/internal stream is intentionally not collapsed into the
contraction stream. Contraction channels are motor output, not carried hidden
state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "paper_data" / "preprint" / "native_core_ablation_v1"
DEFAULT_TRACE_ROOT = DATA_ROOT / "raw_modal_native_signaling_state_space_v1"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "signaling_ablation_trace_phenotype_20260621"
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
OUTCOME_COLORS = {"success": "#8CB87C", "fall": "#D4736C", "stall": "#AAAAAA"}
ACCENT = "#3B5B92"


def _condition_sort(values: pd.Series) -> pd.Categorical:
    observed = [str(value) for value in values.dropna().unique()]
    categories = list(CONDITION_ORDER) + [value for value in observed if value not in CONDITION_ORDER]
    return pd.Categorical(values, categories=categories, ordered=True)


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


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-9:
        return float("nan")
    return float(num / den)


def _rms(array: np.ndarray) -> float:
    if array.size == 0:
        return float("nan")
    arr = np.asarray(array, dtype=np.float64)
    return float(np.sqrt(np.nanmean(arr * arr)))


def _spectral_features(signal: np.ndarray) -> dict[str, float]:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 16:
        return {
            "dominant_frequency_bin": float("nan"),
            "dominant_frequency_fraction": float("nan"),
            "spectral_entropy": float("nan"),
        }
    x = x - float(np.mean(x))
    power = np.abs(np.fft.rfft(x)) ** 2
    if power.size <= 1 or float(np.sum(power[1:])) <= 0.0:
        return {
            "dominant_frequency_bin": float("nan"),
            "dominant_frequency_fraction": float("nan"),
            "spectral_entropy": float("nan"),
        }
    non_dc = power[1:]
    total = float(np.sum(non_dc))
    dominant_index = int(np.argmax(non_dc) + 1)
    probs = non_dc / total
    entropy = -float(np.sum(probs * np.log2(np.maximum(probs, 1e-12))))
    normalized_entropy = entropy / float(np.log2(len(probs))) if len(probs) > 1 else 0.0
    return {
        "dominant_frequency_bin": float(dominant_index),
        "dominant_frequency_fraction": float(power[dominant_index] / total),
        "spectral_entropy": normalized_entropy,
    }


def _pc1_fraction(features: np.ndarray) -> float:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim < 2:
        x = x.reshape(-1, 1)
    x = x.reshape(x.shape[0], -1)
    mask = np.isfinite(x).all(axis=1)
    x = x[mask]
    if x.shape[0] < 4 or x.shape[1] < 2:
        return float("nan")
    x = x - np.mean(x, axis=0, keepdims=True)
    denom = np.linalg.norm(x)
    if denom <= 1e-12:
        return float("nan")
    _u, s, _vt = np.linalg.svd(x, full_matrices=False)
    total = float(np.sum(s * s))
    return float((s[0] * s[0]) / total) if total > 0 else float("nan")


def _heading_reversal_count(dx: np.ndarray) -> int:
    vals = np.asarray(dx, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0
    scale = float(np.nanmedian(np.abs(vals)))
    threshold = max(scale * 0.25, 1e-5)
    signs = np.sign(vals[np.abs(vals) >= threshold])
    if signs.size <= 1:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def _trajectory_metrics(steps: pd.DataFrame) -> dict[str, float | int]:
    ordered = steps.sort_values("step_index", kind="stable")
    x = ordered["com_x"].to_numpy(dtype=float)
    y = ordered["com_y"].to_numpy(dtype=float)
    if x.size < 2:
        return {
            "path_length": float("nan"),
            "net_displacement": float("nan"),
            "tortuosity_path_over_net": float("nan"),
            "x_heading_reversal_count": 0,
            "pre_edge_x_heading_reversal_count": 0,
            "x_wavering_fraction": float("nan"),
        }
    dx = np.diff(x)
    dy = np.diff(y)
    path = float(np.nansum(np.sqrt(dx * dx + dy * dy)))
    net = float(np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2))
    x_reversals = _heading_reversal_count(dx)
    x_abs = float(np.nansum(np.abs(dx)))
    x_net = float(abs(x[-1] - x[0]))

    first_reach = None
    if "cliff_clearance" in ordered:
        hits = np.flatnonzero(ordered["cliff_clearance"].to_numpy(dtype=float) <= 0.0)
        if hits.size:
            first_reach = int(hits[0])
    if first_reach is None:
        pre_dx = dx
    else:
        pre_dx = np.diff(x[: max(first_reach + 1, 1)])
    return {
        "path_length": path,
        "net_displacement": net,
        "tortuosity_path_over_net": _safe_ratio(path, net),
        "x_heading_reversal_count": x_reversals,
        "pre_edge_x_heading_reversal_count": _heading_reversal_count(pre_dx),
        "x_wavering_fraction": _safe_ratio(max(x_abs - x_net, 0.0), x_abs),
    }


def _resolve_trace_npz(assay_dir: Path, trace_npz: str) -> Path:
    raw = Path(str(trace_npz))
    if raw.exists():
        return raw
    candidate = assay_dir / "traces" / raw.name
    if candidate.exists():
        return candidate
    matches = list(assay_dir.glob(f"traces/{raw.name}"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Trace NPZ not found for {trace_npz!r} under {assay_dir}")


def _trace_metrics(npz_path: Path) -> tuple[dict[str, float], pd.DataFrame]:
    with np.load(npz_path) as trace:
        h_drive = np.asarray(trace["phase_h_drive"], dtype=np.float32)
        v_drive = np.asarray(trace["phase_v_drive"], dtype=np.float32)
        phase_actions = np.asarray(trace["phase_action_outputs"], dtype=np.float32)
        action_outputs = np.asarray(trace["action_outputs"], dtype=np.float32)
        spring_current = np.asarray(trace["spring_current_length"], dtype=np.float32)
        spring_rest = np.asarray(trace["spring_rest_length"], dtype=np.float32)
        plant_vel = np.asarray(trace["plant_velocities"], dtype=np.float32)
        nca_state = np.asarray(trace["nca_state"], dtype=np.float32)
        ctrnn_carry = np.asarray(trace["ctrnn_carry"], dtype=np.float32)

        drive = np.concatenate(
            [h_drive.reshape(h_drive.shape[0], -1), v_drive.reshape(v_drive.shape[0], -1)],
            axis=1,
        )
        drive_norm = np.linalg.norm(drive, axis=1)
        spring_strain = spring_current - spring_rest
        velocity_norm = np.linalg.norm(plant_vel.reshape(plant_vel.shape[0], -1), axis=1)
        hidden = np.concatenate(
            [nca_state.reshape(nca_state.shape[0], -1), ctrnn_carry.reshape(ctrnn_carry.shape[0], -1)],
            axis=1,
        )
        metrics = {
            "step_count_npz": int(h_drive.shape[0]),
            "drive_rms": _rms(drive),
            "h_drive_rms": _rms(h_drive),
            "v_drive_rms": _rms(v_drive),
            "drive_abs_mean": float(np.nanmean(np.abs(drive))) if drive.size else float("nan"),
            "drive_pc1_fraction": _pc1_fraction(drive),
            "phase_action_rms": _rms(phase_actions),
            "action_output_rms": _rms(action_outputs),
            "spring_strain_rms": _rms(spring_strain),
            "spring_strain_abs_mean": float(np.nanmean(np.abs(spring_strain))) if spring_strain.size else float("nan"),
            "spring_strain_pc1_fraction": _pc1_fraction(spring_strain),
            "plant_velocity_rms": _rms(plant_vel),
            "velocity_norm_mean": float(np.nanmean(velocity_norm)) if velocity_norm.size else float("nan"),
            "hidden_state_rms": _rms(hidden),
            "hidden_state_pc1_fraction": _pc1_fraction(hidden),
            **{f"drive_{key}": value for key, value in _spectral_features(drive_norm).items()},
            **{f"spring_strain_{key}": value for key, value in _spectral_features(np.linalg.norm(spring_strain, axis=1)).items()},
        }

        h_by_cell = np.nanmean(np.abs(h_drive), axis=(0, 1))
        v_by_cell = np.nanmean(np.abs(v_drive), axis=(0, 1))
        cell_rows = []
        for cell_index, (h_value, v_value) in enumerate(zip(h_by_cell, v_by_cell, strict=False)):
            cell_rows.append(
                {
                    "cell_index": int(cell_index),
                    "mean_abs_h_drive": float(h_value),
                    "mean_abs_v_drive": float(v_value),
                    "mean_abs_total_drive": float(0.5 * (h_value + v_value)),
                }
            )
        return metrics, pd.DataFrame(cell_rows)


def _assay_dirs(trace_root: Path) -> list[Path]:
    return sorted(
        {
            path.parent
            for path in trace_root.glob("**/state_space_trace/trial_manifest.csv")
            if "_previous" not in path.parts
        }
    )


def collect_metrics(trace_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_rows: list[dict[str, Any]] = []
    cell_rows: list[pd.DataFrame] = []
    for assay_dir in _assay_dirs(trace_root):
        manifest_path = assay_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        condition = str(manifest.get("signaling_condition", "unknown"))
        specimen_id = str(manifest.get("specimen_id", "unknown"))
        run_id = str(manifest.get("run_id", "unknown"))
        trial_df = pd.read_csv(assay_dir / "trial_manifest.csv")
        step_df = pd.read_csv(assay_dir / "step_manifest.csv")
        for _, trial in trial_df.iterrows():
            trial_id = str(trial["trial_id"])
            steps = step_df[step_df["trial_id"] == trial_id]
            npz_path = _resolve_trace_npz(assay_dir, str(trial["trace_npz"]))
            trajectory = _trajectory_metrics(steps)
            trace_metrics, per_cell = _trace_metrics(npz_path)
            success = bool(trial.get("success", False))
            fell = bool(trial.get("cliff_fell", False))
            row = {
                "run_id": run_id,
                "specimen_id": specimen_id,
                "condition_id": condition,
                "condition_label": CONDITION_LABELS.get(condition, condition),
                "surface_id": str(trial.get("surface_id", "native_core")),
                "trial_id": trial_id,
                "distance_pair": str(trial.get("distance_pair", "")),
                "side": str(trial.get("side", "")),
                "goal_direction": int(trial.get("goal_direction", 0)),
                "steps_alive": int(trial.get("steps_alive", 0)),
                "success": success,
                "fell": fell,
                "stall": bool((not success) and (not fell)),
                "termination_reason": str(trial.get("termination_reason", "")),
                "first_cliff_reach_step": trial.get("first_cliff_reach_step", np.nan),
                "max_cliff_protrusion": trial.get("max_cliff_protrusion", np.nan),
                "trace_npz": str(npz_path),
                **trajectory,
                **trace_metrics,
            }
            trial_rows.append(row)
            per_cell = per_cell.assign(
                run_id=run_id,
                specimen_id=specimen_id,
                condition_id=condition,
                trial_id=trial_id,
                distance_pair=str(trial.get("distance_pair", "")),
                side=str(trial.get("side", "")),
            )
            cell_rows.append(per_cell)
    if not trial_rows:
        raise FileNotFoundError(
            f"No state_space_trace/trial_manifest.csv files found under {trace_root}. "
            "Pull or generate the native signalling state-space packet first."
        )
    return pd.DataFrame(trial_rows), pd.concat(cell_rows, ignore_index=True)


def summarize(trials: pd.DataFrame, cells: pd.DataFrame) -> dict[str, pd.DataFrame]:
    trials = trials.copy()
    cells = cells.copy()
    trials["condition_order"] = _condition_sort(trials["condition_id"])
    cells["condition_order"] = _condition_sort(cells["condition_id"])
    metric_cols = [
        "success",
        "fell",
        "stall",
        "steps_alive",
        "tortuosity_path_over_net",
        "x_heading_reversal_count",
        "pre_edge_x_heading_reversal_count",
        "x_wavering_fraction",
        "drive_rms",
        "h_drive_rms",
        "v_drive_rms",
        "drive_pc1_fraction",
        "drive_dominant_frequency_fraction",
        "drive_spectral_entropy",
        "phase_action_rms",
        "spring_strain_rms",
        "spring_strain_pc1_fraction",
        "plant_velocity_rms",
        "hidden_state_rms",
    ]
    per_specimen = (
        trials.groupby(["condition_id", "specimen_id"], sort=False)[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    condition_rows = []
    for condition, group in per_specimen.groupby("condition_id", sort=False):
        row: dict[str, Any] = {
            "condition_id": condition,
            "condition_label": CONDITION_LABELS.get(condition, condition),
            "specimen_count": int(group["specimen_id"].nunique()),
        }
        for col in metric_cols:
            values = group[col].to_numpy(dtype=float)
            row[f"{col}_mean"] = float(np.nanmean(values))
            row[f"{col}_sd"] = float(np.nanstd(values, ddof=1)) if values.size > 1 else float("nan")
            row[f"{col}_median"] = float(np.nanmedian(values))
        condition_rows.append(row)
    condition_summary = pd.DataFrame(condition_rows)
    if not condition_summary.empty:
        condition_summary["condition_order"] = _condition_sort(condition_summary["condition_id"])
        condition_summary = condition_summary.sort_values("condition_order").drop(columns=["condition_order"])

    outcome = (
        trials.groupby(["condition_id", "success", "fell", "stall"], sort=False)
        .size()
        .rename("episode_count")
        .reset_index()
    )

    cell_summary = (
        cells.groupby(["condition_id", "cell_index"], sort=False)[
            ["mean_abs_h_drive", "mean_abs_v_drive", "mean_abs_total_drive"]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    cell_summary["condition_order"] = _condition_sort(cell_summary["condition_id"])
    cell_summary = cell_summary.sort_values(["condition_order", "cell_index"]).drop(columns=["condition_order"])
    return {
        "trial_metrics": trials.drop(columns=["condition_order"]),
        "per_specimen_metrics": per_specimen,
        "condition_summary": condition_summary,
        "outcome_counts": outcome,
        "per_cell_drive_summary": cell_summary,
    }


def _save_fig(fig: plt.Figure, output_dir: Path, name: str) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_figures(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    summary = tables["condition_summary"].copy()
    if summary.empty:
        return
    summary["condition_order"] = _condition_sort(summary["condition_id"])
    summary = summary.sort_values("condition_order")
    labels = [CONDITION_LABELS.get(c, c) for c in summary["condition_id"]]
    x = np.arange(len(summary))

    fig, ax = plt.subplots(figsize=(7.2, 2.8), constrained_layout=True)
    bottom = np.zeros(len(summary), dtype=float)
    for outcome, color in [("success", OUTCOME_COLORS["success"]), ("fell", OUTCOME_COLORS["fall"]), ("stall", OUTCOME_COLORS["stall"])]:
        values = summary[f"{outcome}_mean"].to_numpy(dtype=float) * 100.0
        ax.bar(x, values, bottom=bottom, color=color, label=outcome.replace("fell", "fall").title(), width=0.72)
        bottom += values
    ax.set_ylabel("Episodes (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(frameon=True, loc="upper right", fontsize=8)
    ax.set_title("Trace packet outcomes by signalling condition", fontsize=11, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    _save_fig(fig, output_dir, "trace_outcome_composition")

    metric_specs = [
        ("tortuosity_path_over_net", "Tortuosity"),
        ("x_heading_reversal_count", "Heading reversals"),
        ("drive_pc1_fraction", "Drive PC1 frac."),
        ("spring_strain_rms", "Spring strain RMS"),
    ]
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(9.4, 2.6), constrained_layout=True)
    for ax, (metric, title) in zip(axes, metric_specs, strict=True):
        values = summary[f"{metric}_median"].to_numpy(dtype=float)
        ax.bar(x, values, color=ACCENT, alpha=0.78, width=0.72)
        ax.set_title(title, fontsize=9, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    _save_fig(fig, output_dir, "trace_tortuosity_drive_mechanics")

    cell = tables["per_cell_drive_summary"].copy()
    conditions = [c for c in CONDITION_ORDER if c in set(cell["condition_id"])]
    if conditions:
        matrix = np.vstack(
            [
                cell[cell["condition_id"] == condition]
                .sort_values("cell_index")["mean_abs_total_drive"]
                .to_numpy(dtype=float)
                for condition in conditions
            ]
        )
        fig, ax = plt.subplots(figsize=(7.2, 2.5), constrained_layout=True)
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_yticks(np.arange(len(conditions)))
        ax.set_yticklabels([CONDITION_LABELS.get(c, c) for c in conditions])
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xlabel("Cell index")
        ax.set_title("Mean absolute contraction drive by cell", fontsize=11, weight="bold")
        fig.colorbar(im, ax=ax, label="Mean |drive|", shrink=0.75)
        _save_fig(fig, output_dir, "trace_per_cell_drive_heatmap")


def write_report(tables: dict[str, pd.DataFrame], output_dir: Path, trace_root: Path) -> Path:
    summary = tables["condition_summary"].copy()
    trial_metrics = tables["trial_metrics"].copy()
    trial_metrics["outcome"] = np.where(
        trial_metrics["success"],
        "success",
        np.where(trial_metrics["fell"], "fall", "stall"),
    )
    outcome_summary = (
        trial_metrics.groupby(["condition_id", "outcome"], sort=False)
        .agg(
            episode_count=("trial_id", "size"),
            steps_median=("steps_alive", "median"),
            tortuosity_median=("tortuosity_path_over_net", "median"),
            heading_reversal_median=("x_heading_reversal_count", "median"),
            drive_rms_median=("drive_rms", "median"),
            spring_strain_rms_median=("spring_strain_rms", "median"),
            velocity_rms_median=("plant_velocity_rms", "median"),
        )
        .reset_index()
    )
    outcome_summary["condition_order"] = _condition_sort(outcome_summary["condition_id"])
    outcome_order = {"success": 0, "fall": 1, "stall": 2}
    outcome_summary["outcome_order"] = outcome_summary["outcome"].map(outcome_order)
    outcome_summary = outcome_summary.sort_values(["condition_order", "outcome_order"]).drop(
        columns=["condition_order", "outcome_order"]
    )
    lines = [
        "# Native signalling ablation trace phenotype analysis",
        "",
        "## Scope",
        "",
        f"- Trace root: `{trace_root}`",
        f"- Trial rows: `{len(tables['trial_metrics'])}`",
        f"- Specimens: `{tables['trial_metrics']['specimen_id'].nunique()}`",
        f"- Conditions: `{', '.join(summary['condition_id'].astype(str).tolist())}`",
        "",
        "This is the trace-level companion to the endpoint phenotype packet. It measures geometric tortuosity, contraction-command rhythm, and mechanical-state variation from state-space traces.",
        "",
        "## Ontology used",
        "",
        "- Trajectory geometry: COM path, heading reversals, and cliff-contact timing from `step_manifest`.",
        "- Contraction command: `phase_h_drive`, `phase_v_drive`, `action_outputs`, and `phase_action_outputs`.",
        "- Mechanical body state: spring current/rest length, spring strain, plant velocities, and cell/point geometry.",
        "- Hidden/internal signalling state: `nca_state`, `ctrnn_carry`, and their phase traces. Deformation inputs, action outputs, COM, and terrain are not treated as signalling.",
        "",
        "## Condition summary",
        "",
        "Tortuosity is path length divided by net displacement. It can become large when a specimen spends the full budget wandering or stalling with little net displacement, so the table below uses specimen-level medians for trajectory and rhythm descriptors.",
        "",
        "| Condition | Success | Fall | Stall | Tortuosity med. | Heading reversals med. | Pre-edge reversals med. | Drive RMS med. | Drive PC1 med. | Drive entropy med. | Spring strain RMS med. | Hidden RMS med. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {label} | {succ:.1f}% | {fall:.1f}% | {stall:.1f}% | {tort:.3f} | {rev:.1f} | {pre:.1f} | {drive:.4f} | {pc1:.3f} | {entropy:.3f} | {strain:.4f} | {hidden:.2f} |".format(
                label=row["condition_label"],
                succ=100.0 * float(row["success_mean"]),
                fall=100.0 * float(row["fell_mean"]),
                stall=100.0 * float(row["stall_mean"]),
                tort=float(row["tortuosity_path_over_net_median"]),
                rev=float(row["x_heading_reversal_count_median"]),
                pre=float(row["pre_edge_x_heading_reversal_count_median"]),
                drive=float(row["drive_rms_median"]),
                pc1=float(row["drive_pc1_fraction_median"]),
                entropy=float(row["drive_spectral_entropy_median"]),
                strain=float(row["spring_strain_rms_median"]),
                hidden=float(row["hidden_state_rms_median"]),
            )
        )
    lines.extend(
        [
            "",
            "## Outcome-conditioned behaviour",
            "",
            "| Condition | Outcome | Episodes | Steps med. | Tortuosity med. | Heading reversals med. | Drive RMS med. | Spring strain RMS med. | Velocity RMS med. |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in outcome_summary.iterrows():
        lines.append(
            "| {condition} | {outcome} | {n:d} | {steps:.0f} | {tort:.3f} | {rev:.1f} | {drive:.4f} | {strain:.4f} | {vel:.3f} |".format(
                condition=CONDITION_LABELS.get(str(row["condition_id"]), str(row["condition_id"])),
                outcome=str(row["outcome"]),
                n=int(row["episode_count"]),
                steps=float(row["steps_median"]),
                tort=float(row["tortuosity_median"]),
                rev=float(row["heading_reversal_median"]),
                drive=float(row["drive_rms_median"]),
                strain=float(row["spring_strain_rms_median"]),
                vel=float(row["velocity_rms_median"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Sobel-X off is the cleanest tortuous-stall phenotype: half of episodes stall, and stalled episodes have very high median tortuosity and heading reversals.",
            "- Sobel-Y off is more fall-prone and less wandering: failures tend to occur with lower tortuosity and lower velocity, consistent with unstable or poorly regulated edge interaction rather than long indecision.",
            "- Laplacian off is mixed: it produces both stalls and falls, with intermediate heading reversals and a less coherent drive pattern than all-communication off.",
            "- Identity-communication off has rapid outcomes when it succeeds or falls, but its stalls are extremely tortuous; this matches a split phenotype rather than a uniform dampening.",
            "- All communication off does not remove contraction commands. It collapses hidden/internal signalling magnitude, but the drive becomes highly low-dimensional and mechanically strain-heavy; this looks like a simplified open-loop motor regime with many stalls, not complete motor silence.",
            "- Contraction-command metrics are not evidence of internal signalling. The report keeps `phase_h_drive`, `phase_v_drive`, and action outputs separate from `nca_state`/`ctrnn_carry`.",
            "",
            "## Outputs",
            "",
            "- `tables/trace_trial_metrics.csv`",
            "- `tables/trace_per_specimen_metrics.csv`",
            "- `tables/trace_condition_summary.csv`",
            "- `tables/trace_outcome_counts.csv`",
            "- `tables/trace_per_cell_drive_summary.csv`",
            "- `figures/trace_outcome_composition.{png,pdf}`",
            "- `figures/trace_tortuosity_drive_mechanics.{png,pdf}`",
            "- `figures/trace_per_cell_drive_heatmap.{png,pdf}`",
        ]
    )
    report_path = output_dir / "signaling_trace_phenotype_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def analyze(trace_root: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    trials, cells = collect_metrics(trace_root)
    tables = summarize(trials, cells)
    outputs: dict[str, Path] = {}
    for name, df in tables.items():
        out_path = tables_dir / f"trace_{name}.csv"
        df.to_csv(out_path, index=False)
        outputs[name] = out_path
    make_figures(tables, output_dir)
    report_path = write_report(tables, output_dir, trace_root)
    outputs["report"] = report_path
    _write_json(
        output_dir / "signaling_trace_phenotype_summary.json",
        {
            "trace_root": trace_root,
            "output_dir": output_dir,
            "trial_count": int(len(tables["trial_metrics"])),
            "specimen_count": int(tables["trial_metrics"]["specimen_id"].nunique()),
            "conditions": list(tables["condition_summary"]["condition_id"].astype(str)),
            "outputs": outputs,
        },
    )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", default=str(DEFAULT_TRACE_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    outputs = analyze(Path(args.trace_root).resolve(), Path(args.output_dir).resolve())
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
