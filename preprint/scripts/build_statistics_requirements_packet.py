#!/usr/bin/env python
"""Build the consolidated camera-ready statistics requirements packet.

This script is intentionally read-only with respect to the source manifests. It
recomputes specimen-level rates from `episode_manifest.csv`, copies in existing
S4 mechanism tables, computes a small mechanical snapshot table from saved S4
traces, and writes a single markdown report plus CSV exports.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "paper_data" / "preprint" / "native_core_ablation_v1"
OUT_ROOT = DATA_ROOT / "statistics_requirements_packet_20260614"
TABLES_DIR = OUT_ROOT / "tables"
S4_ID = "bin_03_rank_00_cell_6-0-3_flat_0423"

EPISODE_MANIFEST = DATA_ROOT / "tables" / "episode_manifest.csv"
PROVENANCE_CSV = DATA_ROOT / "tables" / "provenance.csv"
FOLLOWUP_SUMMARY = DATA_ROOT / "tables" / "followup_control_summary.csv"
MECH_DIR = DATA_ROOT / "figures" / "s4_mechanism_packet"
S4_CLEAN_TABLES = DATA_ROOT / "eight_specimen_clean_packet_20260613" / "tables"
HARDWARE_ROOT = REPO_ROOT / "paper_data" / "preprint" / "s4_cross_hardware_state_space_20260614"


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _positive_mask(df: pd.DataFrame, endpoint: str) -> pd.Series:
    if endpoint == "success":
        return _bool_series(df["success"])
    if endpoint == "fall":
        return _bool_series(df["fell"])
    if endpoint == "stall":
        if "survived" in df.columns and df["survived"].notna().any() and not df["success"].notna().any():
            return ~_bool_series(df["survived"]) & ~_bool_series(df["fell"])
        return ~_bool_series(df["success"]) & ~_bool_series(df["fell"])
    if endpoint == "survived":
        return _bool_series(df["survived"])
    raise ValueError(f"Unsupported endpoint: {endpoint}")


def _fmt_pct(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{100.0 * float(value):.1f}%"


def _fmt_pp(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{100.0 * float(value):+.1f} pp"


def _fmt_num(value: float | int | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _vector_str(values: dict[str, float]) -> str:
    return ";".join(f"{key}:{value:.3f}" for key, value in sorted(values.items()))


def _rate_vector(df: pd.DataFrame, endpoint: str) -> dict[str, float]:
    if df.empty:
        return {}
    mask = _positive_mask(df, endpoint)
    tmp = df.assign(_positive=mask.to_numpy(dtype=bool))
    return {
        str(specimen): float(group["_positive"].mean())
        for specimen, group in tmp.groupby("specimen_id", sort=True)
    }


def _mean_vector(df: pd.DataFrame, value_col: str) -> dict[str, float]:
    if df.empty:
        return {}
    return {
        str(specimen): float(group[value_col].astype(float).mean())
        for specimen, group in df.groupby("specimen_id", sort=True)
    }


def _summarize_vector(
    values: dict[str, float],
    *,
    count: int | None = None,
    total: int | None = None,
) -> dict[str, Any]:
    arr = np.asarray(list(values.values()), dtype=float)
    if arr.size == 0:
        return {
            "n_specimens": 0,
            "mean": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "positive_count": count,
            "episode_count": total,
            "episode_rate": np.nan,
            "specimen_rates": "",
        }
    return {
        "n_specimens": int(arr.size),
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "positive_count": count,
        "episode_count": total,
        "episode_rate": (float(count) / float(total)) if count is not None and total else np.nan,
        "specimen_rates": _vector_str(values),
    }


@dataclass(frozen=True)
class RateSpec:
    section: str
    label: str
    assay_family: str
    surface_id: str
    condition_id: str
    endpoint: str
    distance_pair: str | None = None
    body: str | None = None
    value_col: str | None = None


def _filter_for_spec(df: pd.DataFrame, spec: RateSpec) -> pd.DataFrame:
    sub = df[
        (df["assay_family"] == spec.assay_family)
        & (df["surface_id"] == spec.surface_id)
        & (df["condition_id"] == spec.condition_id)
    ].copy()
    if spec.distance_pair is not None:
        sub = sub[sub["distance_pair"] == spec.distance_pair]
    if spec.body is not None:
        sub = sub[sub["body"] == spec.body]
    return sub


def build_rate_specs() -> list[RateSpec]:
    specs: list[RateSpec] = []
    for endpoint in ("success", "fall", "stall"):
        specs.append(RateSpec("native", "Native overall", "signaling_ablation", "native_core", "intact", endpoint))
        for pair in ("(2,1)", "(4,2)", "(6,3)"):
            specs.append(
                RateSpec("native_pair", f"Native {pair}", "signaling_ablation", "native_core", "intact", endpoint, distance_pair=pair)
            )
    for endpoint in ("success", "fall", "stall"):
        specs.append(
            RateSpec(
                "extended",
                "Extended combined",
                "signaling_ablation",
                "long_distance_generalization",
                "intact",
                endpoint,
            )
        )
        for pair in ("(8,4)", "(10,5)"):
            specs.append(
                RateSpec(
                    "extended_pair",
                    f"Extended {pair}",
                    "signaling_ablation",
                    "long_distance_generalization",
                    "intact",
                    endpoint,
                    distance_pair=pair,
                )
            )
    for condition, label in (
        ("intact", "Intact"),
        ("absolute_only", "Absolute only"),
        ("compliance_only", "Compliance only"),
        ("deformation_off", "Deformation off"),
    ):
        for endpoint in ("success", "fall", "stall"):
            specs.append(
                RateSpec("deformation_ablation", label, "deformation_ablation", "native_core", condition, endpoint)
            )
    for condition, label in (
        ("intact", "Intact"),
        ("sobel_x_off", "Sobel-X off"),
        ("sobel_y_off", "Sobel-Y off"),
        ("laplacian_off", "Laplacian off"),
        ("identity_comm_off", "Identity off"),
        ("all_comm_off", "All communication off"),
    ):
        for endpoint in ("success", "fall", "stall"):
            specs.append(
                RateSpec("signaling_ablation", label, "signaling_ablation", "native_core", condition, endpoint)
            )
    for condition, body, label in (
        ("rect_7x2", "rect_7x2", "7x2 native"),
        ("rect_6x2", "rect_6x2", "6x2"),
        ("rect_8x2", "rect_8x2", "8x2"),
        ("rect_7x3", "rect_7x3", "7x3"),
    ):
        for endpoint in ("success", "fall", "stall"):
            specs.append(
                RateSpec("body_transfer", label, "body_transfer", "native_core", condition, endpoint, body=body)
            )
    for condition, label in (
        ("deformation_frozen_start", "Deformation frozen"),
        ("perception_frozen_start", "Perception frozen"),
    ):
        for endpoint in ("success", "fall", "stall"):
            specs.append(
                RateSpec("frozen_sensing", label, "frozen_sensing_control", "native_core", condition, endpoint)
            )
    specs.append(
        RateSpec(
            "fitness",
            "Native intact fitness",
            "signaling_ablation",
            "native_core",
            "intact",
            "fitness",
            value_col="fitness",
        )
    )
    return specs


def build_master_rates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    for spec in build_rate_specs():
        sub = _filter_for_spec(df, spec)
        if spec.value_col:
            values = _mean_vector(sub, spec.value_col)
            summary = _summarize_vector(values, count=None, total=None)
        else:
            values = _rate_vector(sub, spec.endpoint)
            positive = int(_positive_mask(sub, spec.endpoint).sum()) if not sub.empty else 0
            summary = _summarize_vector(values, count=positive, total=int(len(sub)))
        row = {
            "section": spec.section,
            "label": spec.label,
            "assay_family": spec.assay_family,
            "surface_id": spec.surface_id,
            "condition_id": spec.condition_id,
            "endpoint": spec.endpoint,
            "distance_pair": spec.distance_pair or "",
            "body": spec.body or "",
            **summary,
        }
        rows.append(row)
        for specimen_id, value in sorted(values.items()):
            vector_rows.append(
                {
                    "section": spec.section,
                    "label": spec.label,
                    "assay_family": spec.assay_family,
                    "surface_id": spec.surface_id,
                    "condition_id": spec.condition_id,
                    "endpoint": spec.endpoint,
                    "distance_pair": spec.distance_pair or "",
                    "body": spec.body or "",
                    "specimen_id": specimen_id,
                    "rate": value,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(vector_rows)


def build_double_cliff_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = df[
        (df["assay_family"] == "signaling_ablation")
        & (df["surface_id"] == "double_cliff")
        & (df["condition_id"] == "intact")
    ].copy()
    rows: list[dict[str, Any]] = []
    for _, row in sub.iterrows():
        if bool(row.get("survived")):
            outcome = "survive"
        elif bool(row.get("fell")):
            outcome = "fall"
        else:
            outcome = "stall"
        rows.append({"specimen_id": row["specimen_id"], "width": int(row["width"]), "outcome": outcome})
    long = pd.DataFrame(rows)
    matrix = long.pivot(index="specimen_id", columns="width", values="outcome").reset_index()
    matrix["survived_any_width"] = matrix[[2, 4, 6]].eq("survive").any(axis=1)
    counts = (
        long.assign(survive=lambda x: x["outcome"].eq("survive"))
        .groupby("width", sort=True)
        .agg(n=("survive", "size"), survival_count=("survive", "sum"))
        .reset_index()
    )
    counts["fraction"] = counts["survival_count"].astype(int).astype(str) + "/" + counts["n"].astype(int).astype(str)
    return matrix, counts


def _bootstrap_ci(values: np.ndarray, *, draws: int = 10_000, seed: int = 137) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan, np.nan
    idx = rng.integers(0, arr.size, size=(draws, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def build_paired_deltas(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = df[
        (df["assay_family"] == "signaling_ablation")
        & (df["surface_id"] == "native_core")
        & (df["condition_id"] == "intact")
    ]
    base_rates = _rate_vector(base, "success")
    rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    conditions: list[tuple[str, str, str, str]] = [
        ("deformation", "intact", "deformation_ablation", "Intact"),
        ("deformation", "absolute_only", "deformation_ablation", "Absolute only"),
        ("deformation", "compliance_only", "deformation_ablation", "Compliance only"),
        ("deformation", "deformation_off", "deformation_ablation", "Deformation off"),
        ("signaling", "sobel_x_off", "signaling_ablation", "Sobel-X off"),
        ("signaling", "sobel_y_off", "signaling_ablation", "Sobel-Y off"),
        ("signaling", "laplacian_off", "signaling_ablation", "Laplacian off"),
        ("signaling", "identity_comm_off", "signaling_ablation", "Identity off"),
        ("signaling", "all_comm_off", "signaling_ablation", "All communication off"),
    ]
    for family, condition, assay_family, label in conditions:
        sub = df[
            (df["assay_family"] == assay_family)
            & (df["surface_id"] == "native_core")
            & (df["condition_id"] == condition)
        ]
        condition_rates = _rate_vector(sub, "success")
        deltas = {
            specimen: condition_rates.get(specimen, np.nan) - base_rates.get(specimen, np.nan)
            for specimen in sorted(base_rates)
        }
        arr = np.asarray([value for value in deltas.values() if np.isfinite(value)], dtype=float)
        lo, hi = _bootstrap_ci(arr, seed=137 + len(rows))
        rows.append(
            {
                "family": family,
                "condition_id": condition,
                "condition_label": label,
                "n_specimens": int(arr.size),
                "mean_delta": float(arr.mean()) if arr.size else np.nan,
                "sd_delta": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "ci95_boot_lo": lo,
                "ci95_boot_hi": hi,
                "median_delta": float(np.median(arr)) if arr.size else np.nan,
                "min_delta": float(arr.min()) if arr.size else np.nan,
                "max_delta": float(arr.max()) if arr.size else np.nan,
                "delta_vector": _vector_str({k: v for k, v in deltas.items() if np.isfinite(v)}),
            }
        )
        for specimen_id, delta in sorted(deltas.items()):
            vector_rows.append(
                {
                    "family": family,
                    "condition_id": condition,
                    "condition_label": label,
                    "specimen_id": specimen_id,
                    "baseline_rate": base_rates.get(specimen_id, np.nan),
                    "condition_rate": condition_rates.get(specimen_id, np.nan),
                    "delta": delta,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(vector_rows)


def build_with_without_s4(master_specs: list[RateSpec], df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, group_df in (("all_8", df), ("excluding_s4", df[df["specimen_id"] != S4_ID])):
        for spec in master_specs:
            sub = _filter_for_spec(group_df, spec)
            if spec.value_col:
                values = _mean_vector(sub, spec.value_col)
                summary = _summarize_vector(values)
            else:
                values = _rate_vector(sub, spec.endpoint)
                positive = int(_positive_mask(sub, spec.endpoint).sum()) if not sub.empty else 0
                summary = _summarize_vector(values, count=positive, total=int(len(sub)))
            rows.append(
                {
                    "group": group_name,
                    "section": spec.section,
                    "label": spec.label,
                    "assay_family": spec.assay_family,
                    "surface_id": spec.surface_id,
                    "condition_id": spec.condition_id,
                    "endpoint": spec.endpoint,
                    "distance_pair": spec.distance_pair or "",
                    "body": spec.body or "",
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def build_s8_timing() -> pd.DataFrame:
    edge_path = MECH_DIR / "s4_edge_timing_episode_metrics.csv"
    if not edge_path.exists():
        return pd.DataFrame()
    metrics = pd.read_csv(edge_path)
    true = metrics[metrics["true_cliff_event"].astype(bool)].copy()

    with open(MECH_DIR / "summary.json") as handle:
        summary = json.load(handle)
    trace_roots = {
        "intact": Path(summary["intact_native_trace_root"]),
        "deformation_off": Path(summary["deformation_off_edge_trace_root"]),
    }

    extra_rows: list[dict[str, Any]] = []
    for _, row in true.iterrows():
        trace_root = trace_roots[str(row["condition"])]
        trace_path = trace_root / "traces" / f"{row['trial_id']}.npz"
        if not trace_path.exists():
            continue
        loaded = np.load(trace_path)
        try:
            positions = np.asarray(loaded["plant_positions"], dtype=np.float32)
            if "robot_point_mask" in loaded.files:
                mask = np.asarray(loaded["robot_point_mask"], dtype=bool)
            else:
                mask = np.ones((positions.shape[1],), dtype=bool)
        finally:
            loaded.close()
        com = positions[:, mask, :].mean(axis=1)
        event = int(row["event_step"])
        pre = com[: max(1, event + 1)]
        diffs = np.diff(pre, axis=0)
        path_length = float(np.linalg.norm(diffs, axis=1).sum())
        net = float(np.linalg.norm(pre[-1] - pre[0]))
        dx = diffs[:, 0]
        active = np.abs(dx) > 1e-4
        signs = np.sign(dx[active])
        heading_reversals = int(np.sum(signs[1:] != signs[:-1])) if signs.size > 1 else 0
        extra_rows.append(
            {
                "condition": row["condition"],
                "distance_pair": row["distance_pair"],
                "side": row["side"],
                "trial_id": row["trial_id"],
                "approach_tortuosity": (path_length / net) if net > 1e-9 else np.nan,
                "approach_heading_reversals": heading_reversals,
                "time_to_edge": event,
            }
        )
    extra = pd.DataFrame(extra_rows)
    merged = true.merge(extra, on=["condition", "distance_pair", "side", "trial_id"], how="left")
    fields = [
        ("max_cliff_overhang_before_goal_commit", "reversal_overhang"),
        ("direction_changes_after_event", "post_event_direction_switches"),
        ("event_to_goal_commit_steps", "recommitment_latency_steps"),
        ("approach_tortuosity", "approach_tortuosity"),
        ("approach_heading_reversals", "approach_heading_reversals"),
        ("time_to_edge", "time_to_edge_steps"),
    ]
    rows: list[dict[str, Any]] = []
    for condition, group in merged.groupby("condition", sort=True):
        for source_col, metric in fields:
            values = group[source_col].dropna().astype(float).to_numpy()
            rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "n_true_events": int(values.size),
                    "median": float(np.median(values)) if values.size else np.nan,
                    "mean": float(values.mean()) if values.size else np.nan,
                    "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "min": float(values.min()) if values.size else np.nan,
                    "max": float(values.max()) if values.size else np.nan,
                    "values": ";".join(f"{v:.6g}" for v in values),
                }
            )
    return pd.DataFrame(rows)


def build_s6_snapshots() -> pd.DataFrame:
    trial_meta = MECH_DIR / "s4_band_predictivity_trial_metadata.csv"
    summary_json = MECH_DIR / "summary.json"
    if not trial_meta.exists() or not summary_json.exists():
        return pd.DataFrame()
    meta = pd.read_csv(trial_meta)
    meta = meta[meta["surface_id"].fillna("native_core").eq("native_core")] if "surface_id" in meta.columns else meta
    with open(summary_json) as handle:
        summary = json.load(handle)
    trace_root = Path(summary["deformation_off_edge_trace_root"])
    rel_steps = [-32, 0, 32, 96]
    trial_rows: list[dict[str, Any]] = []
    for _, row in meta.iterrows():
        trial_id = str(row["trial_id"])
        event_index = int(row["event_index"])
        trace_path = trace_root / "traces" / f"{trial_id}.npz"
        if not trace_path.exists():
            continue
        loaded = np.load(trace_path)
        try:
            corners = np.asarray(loaded["cell_corner_positions"], dtype=np.float32)
        finally:
            loaded.close()
        local_shape = corners - corners.mean(axis=2, keepdims=True)
        base_start = max(0, event_index - 128)
        base_stop = max(base_start + 1, event_index - 32)
        baseline = local_shape[base_start:base_stop].mean(axis=0)
        for rel in rel_steps:
            idx = int(np.clip(event_index + rel, 0, local_shape.shape[0] - 1))
            mag = np.sqrt(np.mean((local_shape[idx] - baseline) ** 2, axis=(1, 2)))
            for cell_index, value in enumerate(mag.tolist()):
                trial_rows.append(
                    {
                        "trial_id": trial_id,
                        "distance_pair": row.get("distance_pair", ""),
                        "event_index": event_index,
                        "relative_step": rel,
                        "absolute_step": idx,
                        "cell_index": int(cell_index),
                        "mechanical_perturbation": float(value),
                        "baseline_relative_start": base_start - event_index,
                        "baseline_relative_stop": base_stop - event_index,
                        "baseline_policy": "RMS local cell-shape deviation from pre-release baseline",
                    }
                )
    trial_frame = pd.DataFrame(trial_rows)
    if trial_frame.empty:
        return trial_frame
    summary_rows = (
        trial_frame.groupby(["relative_step", "cell_index"], sort=True)
        .agg(
            event_count=("trial_id", "nunique"),
            mean_mechanical_perturbation=("mechanical_perturbation", "mean"),
            sd_mechanical_perturbation=("mechanical_perturbation", "std"),
            min_mechanical_perturbation=("mechanical_perturbation", "min"),
            max_mechanical_perturbation=("mechanical_perturbation", "max"),
        )
        .reset_index()
    )
    summary_rows["sd_mechanical_perturbation"] = summary_rows["sd_mechanical_perturbation"].fillna(0.0)
    return summary_rows


def build_s12_hardware() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inventory = pd.read_csv(HARDWARE_ROOT / "tables" / "hardware_trace_inventory.csv")
    matrix = pd.read_csv(HARDWARE_ROOT / "tables" / "hardware_trial_matrix.csv")
    complete_assays = inventory[
        (inventory["trace_count"] == inventory["expected_trace_count"])
        & (inventory["trial_count"] == inventory["expected_trial_count"])
        & (inventory["has_manifest_json"].astype(bool))
        & (inventory["has_summary_json"].astype(bool))
        & (inventory["has_step_manifest_parquet"].astype(bool))
        & (inventory["has_static_arrays_npz"].astype(bool))
    ]
    counts = complete_assays.groupby("runtime")["assay"].nunique()
    complete_runtimes = sorted(counts[counts >= 2].index.tolist())
    m = matrix[matrix["runtime"].isin(complete_runtimes)].copy()

    overall_rows: list[dict[str, Any]] = []
    for (runtime, assay), group in m.groupby(["runtime", "assay"], sort=True):
        endpoint_counts = group["endpoint_class"].value_counts().to_dict()
        overall_rows.append(
            {
                "runtime": runtime,
                "assay": assay,
                "n": int(len(group)),
                "success_count": int(_bool_series(group["success"]).sum()),
                "fall_count": int(_bool_series(group["cliff_fell"]).sum()),
                "edge_reach_count": int(group["first_cliff_reach_step"].notna().sum()),
                "returned_from_edge_count": int(_bool_series(group["returned_from_edge"]).sum()),
                "full_reverse_and_goal": int(endpoint_counts.get("full_reverse_and_goal", 0)),
                "cliff_stop_only": int(endpoint_counts.get("cliff_stop_only", 0)),
                "direct_goal_no_edge": int(endpoint_counts.get("direct_goal_no_edge", 0)),
                "returned_no_goal": int(endpoint_counts.get("returned_no_goal", 0)),
                "stall_no_edge": int(endpoint_counts.get("stall_no_edge", 0)),
                "complete_runtime": True,
            }
        )
    overall = pd.DataFrame(overall_rows)

    regions = {
        "native_interpolated_6_pairs": {"(1,1)", "(2,1)", "(3,2)", "(4,2)", "(5,3)", "(6,3)"},
        "original_native_3_pairs": {"(2,1)", "(4,2)", "(6,3)"},
        "extended_interpolated_4_pairs": {"(7,4)", "(8,4)", "(9,5)", "(10,5)"},
        "all_10_pairs": {"(1,1)", "(2,1)", "(3,2)", "(4,2)", "(5,3)", "(6,3)", "(7,4)", "(8,4)", "(9,5)", "(10,5)"},
    }
    region_rows: list[dict[str, Any]] = []
    for region, pairs in regions.items():
        subset = m[m["distance_pair"].isin(pairs)]
        for (runtime, assay), group in subset.groupby(["runtime", "assay"], sort=True):
            endpoint_counts = group["endpoint_class"].value_counts().to_dict()
            region_rows.append(
                {
                    "region": region,
                    "runtime": runtime,
                    "assay": assay,
                    "n": int(len(group)),
                    "success_count": int(_bool_series(group["success"]).sum()),
                    "fall_count": int(_bool_series(group["cliff_fell"]).sum()),
                    "full_reverse_and_goal": int(endpoint_counts.get("full_reverse_and_goal", 0)),
                    "cliff_stop_only": int(endpoint_counts.get("cliff_stop_only", 0)),
                    "direct_goal_no_edge": int(endpoint_counts.get("direct_goal_no_edge", 0)),
                    "returned_no_goal": int(endpoint_counts.get("returned_no_goal", 0)),
                    "stall_no_edge": int(endpoint_counts.get("stall_no_edge", 0)),
                }
            )
    regions_df = pd.DataFrame(region_rows)
    pair_rows = (
        m.groupby(["runtime", "assay", "surface_id", "distance_pair", "side"], sort=True)
        .agg(
            success=("success", lambda x: int(_bool_series(x).sum())),
            fall=("cliff_fell", lambda x: int(_bool_series(x).sum())),
            endpoint_class=("endpoint_class", lambda x: ",".join(map(str, x))),
            first_cliff_reach_step=("first_cliff_reach_step", lambda x: ";".join("" if pd.isna(v) else str(int(v)) for v in x)),
            final_com_x=("final_com_x", lambda x: ";".join(f"{float(v):.4f}" for v in x)),
        )
        .reset_index()
    )
    return overall, regions_df, pair_rows


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _table(df: pd.DataFrame, columns: list[str], *, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows available._"
    sub = df.loc[:, columns].copy()
    if max_rows is not None:
        sub = sub.head(max_rows)
    headers = [str(col) for col in sub.columns]
    body = sub.fillna("").astype(str).values.tolist()

    def clean(value: str) -> str:
        return value.replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(clean(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(clean(cell) for cell in row) + " |")
    return "\n".join(lines)


def _condition_comp(master: pd.DataFrame, section: str, labels: Iterable[str]) -> pd.DataFrame:
    subset = master[(master["section"] == section) & (master["label"].isin(labels))].copy()
    rows: list[dict[str, Any]] = []
    for label, group in subset.groupby("label", sort=False):
        out = {"Condition": label}
        for endpoint in ("success", "fall", "stall", "survived"):
            row = group[group["endpoint"] == endpoint]
            if row.empty:
                continue
            r = row.iloc[0]
            if pd.notna(r["positive_count"]) and pd.notna(r["episode_count"]):
                out[endpoint.capitalize()] = f"{int(r['positive_count'])}/{int(r['episode_count'])} = {_fmt_pct(r['episode_rate'])}"
            out[f"{endpoint.capitalize()} mean ± SD"] = f"{_fmt_pct(r['mean'])} ± {_fmt_pct(r['sd'])}"
            out[f"{endpoint.capitalize()} vector"] = r["specimen_rates"]
        rows.append(out)
    return pd.DataFrame(rows)


def write_markdown(
    *,
    master: pd.DataFrame,
    vectors: pd.DataFrame,
    double_matrix: pd.DataFrame,
    double_counts: pd.DataFrame,
    deltas: pd.DataFrame,
    with_without: pd.DataFrame,
    s6: pd.DataFrame,
    s8: pd.DataFrame,
    s12_overall: pd.DataFrame,
    s12_regions: pd.DataFrame,
    s12_pairs: pd.DataFrame,
) -> Path:
    provenance = pd.read_csv(PROVENANCE_CSV)
    followup = _read_csv_if_exists(FOLLOWUP_SUMMARY)
    band = _read_csv_if_exists(MECH_DIR / "s4_band_predictivity_by_window.csv")
    onsets = _read_csv_if_exists(MECH_DIR / "s4_band_predictivity_onsets.csv")
    onset_reg = _read_csv_if_exists(MECH_DIR / "s4_band_predictivity_onset_regression.csv")
    decoder_controls = _read_csv_if_exists(MECH_DIR / "s4_decoder_controls.csv")
    timing_episode = _read_csv_if_exists(MECH_DIR / "s4_edge_timing_episode_metrics.csv")
    profile = _read_csv_if_exists(MECH_DIR / "eight_specimen_profile_table.csv")
    signal_cell = _read_csv_if_exists(S4_CLEAN_TABLES / "s4_signal_cell_edge_impulse_lag8_auc_summary.csv")
    body_cell = _read_csv_if_exists(DATA_ROOT / "figures" / "voxel_reversal_heatmap_deformation_off" / "voxel_decoder_summary.csv")

    lines: list[str] = []
    lines.append("# Sensing at the edges - consolidated statistics and data response")
    lines.append("")
    lines.append(f"Created: 2026-06-14")
    lines.append("")
    lines.append("This packet responds to S1-S12 in the consolidated data request. It uses specimen-level aggregation (`n=8`) and reports SD, not SEM, unless an existing S4 decoder table is explicitly quoted from a prior mechanism packet.")
    lines.append("")
    lines.append("## Source data and outputs")
    lines.append("")
    lines.append(f"- Primary manifest: `{EPISODE_MANIFEST}`")
    lines.append(f"- Provenance: `{PROVENANCE_CSV}`")
    lines.append(f"- Existing S4 mechanism packet: `{MECH_DIR}`")
    lines.append(f"- Cross-hardware package: `{HARDWARE_ROOT}`")
    lines.append(f"- New output directory: `{OUT_ROOT}`")
    lines.append("")
    lines.append("## Global conventions")
    lines.append("")
    lines.append("- Unit of aggregation is the selected specimen. Deterministic terrain episodes are not treated as independent stochastic replicates.")
    lines.append("- Dispersion is SD across specimen-level rates.")
    lines.append("- Episode fractions are included as counts for sanity checks.")
    lines.append("- Double-cliff is a survival/stress endpoint, not a goal-completion success endpoint.")
    lines.append("")

    lines.append("## S1. Master rates table")
    lines.append("")
    lines.append("Full exports: `tables/s1_master_rates.csv` and `tables/s1_rate_vectors.csv`.")
    lines.append("")
    native_rows = _condition_comp(master, "native", ["Native overall"])
    lines.append("### Native overall")
    lines.append(_table(native_rows, list(native_rows.columns)))
    lines.append("")
    pair_rows = master[(master["section"].isin(["native_pair", "extended_pair"])) & (master["endpoint"] == "success")].copy()
    pair_rows["Rate"] = pair_rows.apply(lambda r: f"{int(r['positive_count'])}/{int(r['episode_count'])} = {_fmt_pct(r['episode_rate'])}", axis=1)
    pair_rows["Mean ± SD"] = pair_rows.apply(lambda r: f"{_fmt_pct(r['mean'])} ± {_fmt_pct(r['sd'])}", axis=1)
    pair_rows["Median [min,max]"] = pair_rows.apply(lambda r: f"{_fmt_pct(r['median'])} [{_fmt_pct(r['min'])}, {_fmt_pct(r['max'])}]", axis=1)
    lines.append("### Native and extended pair success")
    lines.append(_table(pair_rows, ["label", "Rate", "Mean ± SD", "Median [min,max]", "specimen_rates"]))
    lines.append("")
    def_rows = _condition_comp(master, "deformation_ablation", ["Intact", "Absolute only", "Compliance only", "Deformation off"])
    lines.append("### Native deformation ablations")
    lines.append(_table(def_rows, list(def_rows.columns)))
    lines.append("")
    sig_rows = _condition_comp(master, "signaling_ablation", ["Intact", "Sobel-X off", "Sobel-Y off", "Laplacian off", "Identity off", "All communication off"])
    lines.append("### Native signalling ablations")
    lines.append(_table(sig_rows, list(sig_rows.columns)))
    lines.append("")
    body_rows = _condition_comp(master, "body_transfer", ["7x2 native", "6x2", "8x2", "7x3"])
    lines.append("### Body transfer")
    lines.append(_table(body_rows, list(body_rows.columns)))
    lines.append("")
    fit = master[(master["section"] == "fitness") & (master["endpoint"] == "fitness")].iloc[0]
    lines.append(f"### Fitness")
    lines.append(f"Native intact mean fitness across specimens is `{fit['mean']:.2f} ± {fit['sd']:.2f} SD`; median `{fit['median']:.2f}`, range `[{fit['min']:.2f}, {fit['max']:.2f}]`. Per-specimen vector: `{fit['specimen_rates']}`.")
    lines.append("")

    lines.append("## S2. Double-cliff survival matrix")
    lines.append("")
    lines.append("Full export: `tables/s2_double_cliff_survival_matrix.csv` and `tables/s2_double_cliff_width_counts.csv`.")
    lines.append("")
    matrix_md = double_matrix.rename(columns={2: "w=2", 4: "w=4", 6: "w=6"})
    lines.append(_table(matrix_md, ["specimen_id", "w=2", "w=4", "w=6", "survived_any_width"]))
    lines.append("")
    lines.append(_table(double_counts, ["width", "survival_count", "n", "fraction"]))
    survivors = double_matrix[double_matrix["survived_any_width"]]["specimen_id"].tolist()
    lines.append("")
    lines.append(f"Specimens surviving at least one width: `{', '.join(survivors)}`.")
    lines.append("")

    lines.append("## S3. Paired ablation deltas")
    lines.append("")
    lines.append("Full exports: `tables/s3_paired_ablation_deltas.csv` and `tables/s3_paired_ablation_delta_vectors.csv`. Deltas are condition success rate minus intact native success rate within specimen; CI is a 10,000-draw bootstrap over specimens.")
    display = deltas.copy()
    display["Mean delta"] = display["mean_delta"].map(_fmt_pp)
    display["95% CI"] = display.apply(lambda r: f"[{_fmt_pp(r['ci95_boot_lo'])}, {_fmt_pp(r['ci95_boot_hi'])}]", axis=1)
    display["SD"] = display["sd_delta"].map(_fmt_pp)
    lines.append(_table(display, ["family", "condition_label", "n_specimens", "Mean delta", "SD", "95% CI", "delta_vector"]))
    lines.append("")

    lines.append("## S4. With / without S4")
    lines.append("")
    lines.append("Full export: `tables/s4_with_without_s4.csv`. Selected headline rows:")
    headline_keys = [
        ("native", "Native overall", "success"),
        ("extended", "Extended combined", "success"),
        ("deformation_ablation", "Deformation off", "success"),
        ("signaling_ablation", "Sobel-X off", "success"),
        ("signaling_ablation", "All communication off", "success"),
        ("body_transfer", "6x2", "success"),
        ("frozen_sensing", "Perception frozen", "success"),
        ("fitness", "Native intact fitness", "fitness"),
    ]
    ww_rows: list[dict[str, Any]] = []
    for section, label, endpoint in headline_keys:
        sub = with_without[(with_without["section"] == section) & (with_without["label"] == label) & (with_without["endpoint"] == endpoint)]
        for _, r in sub.iterrows():
            rate = f"{int(r['positive_count'])}/{int(r['episode_count'])} = {_fmt_pct(r['episode_rate'])}" if pd.notna(r["positive_count"]) else "NA"
            ww_rows.append(
                {
                    "metric": label,
                    "endpoint": endpoint,
                    "group": r["group"],
                    "n": int(r["n_specimens"]),
                    "episode fraction": rate,
                    "mean ± SD": f"{_fmt_pct(r['mean'])} ± {_fmt_pct(r['sd'])}" if endpoint != "fitness" else f"{r['mean']:.2f} ± {r['sd']:.2f}",
                    "range": f"[{_fmt_pct(r['min'])}, {_fmt_pct(r['max'])}]" if endpoint != "fitness" else f"[{r['min']:.2f}, {r['max']:.2f}]",
                }
            )
    lines.append(_table(pd.DataFrame(ww_rows), ["metric", "endpoint", "group", "n", "episode fraction", "mean ± SD", "range"]))
    lines.append("")

    lines.append("## S5. S4 predictivity package")
    lines.append("")
    lines.append("Existing completed S4 mechanism exports are in `figures/s4_mechanism_packet/`. The completed package provides body and hidden/internal signal streams with permutation controls, but not the newly requested third Sobel-filtered signalling-field-gradient stream as a full position-by-lag map. Therefore, the safe current claim should remain a two-stream body-vs-hidden/internal comparison unless that additional field-gradient decoder is run.")
    lines.append("")
    if not signal_cell.empty:
        lines.append("- Hidden/internal cell-wise signal map: `eight_specimen_clean_packet_20260613/tables/s4_signal_cell_edge_impulse_lag8_auc_summary.csv`.")
        top_signal = signal_cell.sort_values("mean_auc", ascending=False).head(6)
        top_signal = top_signal.assign(mean_auc_fmt=top_signal["mean_auc"].map(lambda v: f"{v:.3f}"))
        signal_cols = ["cell_index", "heldout_trial_count", "mean_auc_fmt", "mean_balanced_accuracy"]
        if "sample_count" in top_signal.columns:
            signal_cols.append("sample_count")
        lines.append(_table(top_signal, signal_cols, max_rows=6))
    if not band.empty:
        lines.append("")
        edge_lag8 = band[(band["window"] == "edge_impulse") & (band["lag"] == 8)].copy()
        edge_lag8["AUC"] = edge_lag8["mean_auc"].map(lambda v: f"{v:.3f}")
        edge_lag8["actual-null AUC"] = edge_lag8["mean_actual_minus_null_auc"].map(lambda v: f"{v:.3f}")
        lines.append("Band-level edge-impulse summary at lag 8:")
        lines.append(_table(edge_lag8, ["stream", "band", "heldout_trial_count", "sample_count", "AUC", "actual-null AUC", "mean_balanced_accuracy", "mean_empirical_p_auc"]))
    if not onsets.empty:
        lines.append("")
        lines.append("Onset audit:")
        lines.append(_table(onsets, ["stream", "band", "status", "onset_window", "onset_lag", "onset_relative_center", "onset_auc", "best_window", "best_auc"]))
    if not onset_reg.empty:
        lines.append("")
        lines.append("Onset-latency regression:")
        lines.append(_table(onset_reg, ["stream", "status", "n_bands", "slope_steps_per_band_position", "r2"]))
    if not decoder_controls.empty:
        lines.append("")
        lines.append("Decoder controls:")
        dc = decoder_controls.copy()
        dc["AUC"] = dc["mean_auc"].map(lambda v: "NA" if pd.isna(v) else f"{v:.3f}")
        lines.append(_table(dc, ["stream", "band", "window", "lag", "control_status", "dominant_skip_reason", "heldout_trial_count", "sample_count", "AUC", "mean_actual_minus_null_auc"]))
    lines.append("")
    lines.append("Confirmations for the completed S4 mechanism packet: event count is 3 native deformation-off true cliff-frontier events for the band decoder; window is `[-32,+96]`; the main single-lag panel uses lag `8`. The prior band null used 30 permutations, not the newly requested 1000. Treat the p/null values as audit-level rather than final high-resolution permutation tests.")
    lines.append("")

    lines.append("## S6. S4 propagation snapshot data")
    lines.append("")
    lines.append("Full export: `tables/s6_s4_mechanical_snapshot.csv`. Values are per-cell RMS local cell-shape deviations from a pre-release baseline, averaged across the three native deformation-off true cliff events. Baseline window is `[-128,-32)` relative to each event.")
    if not s6.empty:
        pivot = s6.pivot(index="cell_index", columns="relative_step", values="mean_mechanical_perturbation").reset_index()
        pivot.columns = ["cell_index"] + [f"t{int(c):+d}" for c in pivot.columns[1:]]
        for col in pivot.columns[1:]:
            pivot[col] = pivot[col].map(lambda v: f"{v:.4f}")
        lines.append(_table(pivot, list(pivot.columns)))
    else:
        lines.append("_No S6 snapshot data could be computed._")
    lines.append("")

    lines.append("## S7. S4 directional flip and symmetry")
    lines.append("")
    if not timing_episode.empty:
        s4_native = timing_episode[["condition", "distance_pair", "side", "event_class", "true_cliff_event", "returned_after_event", "success", "event_to_goal_commit_steps", "direction_changes_after_event"]].copy()
        lines.append("S4 native trace-level event grammar:")
        lines.append(_table(s4_native, list(s4_native.columns)))
        true_by_cond = timing_episode[timing_episode["true_cliff_event"].astype(bool)].groupby("condition")["side"].apply(lambda x: ",".join(x.astype(str))).to_dict()
        lines.append("")
        lines.append(f"In the current trace set, intact true-cliff episodes occur on side labels `{true_by_cond.get('intact', 'NA')}` and deformation-off true-cliff episodes occur on side labels `{true_by_cond.get('deformation_off', 'NA')}`. Matched by distance pair, the true-cliff side flips for all three native pairs. This is a trace-level observation, not an all-specimen initial-commitment result.")
    lines.append("")
    lines.append("The consolidated endpoint manifest has sparse `initial_direction` labels (`none`/missing for almost all rows), so the requested all-other-specimen flip analysis is not estimable from the current endpoint tables. A trace-level behavioral grammar pass would be needed for the other seven specimens.")
    lines.append("")

    lines.append("## S8. S4 reversal timing")
    lines.append("")
    lines.append("Full export: `tables/s8_s4_reversal_timing_summary.csv`. The matched set contains three true-cliff events per condition when matched by distance pair. The side label flips across intact and deformation-off, so this is not same-side matching.")
    if not s8.empty:
        display_s8 = s8.copy()
        for col in ("median", "mean", "sd", "min", "max"):
            display_s8[col] = display_s8[col].map(lambda v: "NA" if pd.isna(v) else f"{v:.3f}")
        lines.append(_table(display_s8, ["condition", "metric", "n_true_events", "median", "mean", "sd", "min", "max", "values"]))
        lines.append("")
        lines.append("Interpretation: deformation-off preserves true-edge reversal in S4, but it does not support the specific claim of greater overhang. In these true-edge events, deformation-off has lower overhang than intact and more post-event direction switching; recommitment latency after true edge contact is shorter, not delayed. The stronger paper-safe claim is that deformation feedback modulates the precision/stability and mirror-side expression of an otherwise preserved reversal regime.")
    else:
        lines.append("_No S8 timing data available._")
    lines.append("")

    lines.append("## S9. S4 joint ablation")
    lines.append("")
    if not followup.empty:
        joint = followup[followup["assay_family"] == "combined_ablation_quick"].copy()
        joint["Success"] = joint.apply(lambda r: f"{int(r['positive_count'])}/{int(r['episode_count'])} = {_fmt_pct(r['episode_rate'])}", axis=1)
        joint["Fall"] = joint["fall_episode_rate"].map(_fmt_pct)
        joint["Stall"] = joint.apply(lambda r: _fmt_pct(1.0 - float(r["episode_rate"]) - float(r["fall_episode_rate"])), axis=1)
        lines.append(_table(joint, ["condition_label", "n_specimens", "Success", "Fall", "Stall"]))
        lines.append("")
        lines.append("These were new S4-only Modal follow-up rollouts, merged into the consolidated manifest. Deformation-off + Sobel-X off and deformation-off + all-communication off both give 0/6 success and 100% stall.")
    else:
        lines.append("_No joint-ablation rows found._")
    lines.append("")

    lines.append("## S10. Frozen-sensing native control")
    lines.append("")
    if not followup.empty:
        frozen = master[(master["section"] == "frozen_sensing")].copy()
        frozen_comp = _condition_comp(master, "frozen_sensing", ["Deformation frozen", "Perception frozen"])
        lines.append(_table(frozen_comp, list(frozen_comp.columns)))
        lines.append("")
        lines.append("This control was run for all eight specimens. Perception-frozen success is 12/48 = 25.0%; deformation-frozen success is 21/48 = 43.8%. This supports a test-time load-bearing sensing claim without making a new evolution-needed claim.")
    else:
        lines.append("_Frozen-sensing rows not found._")
    lines.append("")

    lines.append("## S11. Confirmations")
    lines.append("")
    group_rows = provenance.groupby(["group_id", "run_id"], sort=True)["specimen_id"].apply(lambda x: ", ".join(sorted(x))).reset_index()
    lines.append("### Selection and provenance")
    lines.append(_table(group_rows, ["group_id", "run_id", "specimen_id"]))
    lines.append("")
    descriptor_cells = sorted({sid.split("_cell_")[-1].split("_flat_")[0] for sid in provenance["specimen_id"].astype(str)})
    lines.append(f"Descriptor cells represented by specimen IDs: `{', '.join(descriptor_cells)}`. The local packet confirms `n=8` selected specimens from 5 source runs; it does not by itself prove the full archive-selection rule beyond the recorded selected set, so manuscript wording should say selected specimens from multiple source runs rather than independent runs.")
    lines.append("")
    bt = master[(master["section"] == "body_transfer") & (master["endpoint"] == "success")]
    six = bt[bt["label"] == "6x2"].iloc[0]
    lines.append(f"Body-transfer n is `{int(six['n_specimens'])}` for all four bodies. The 6x2 transfer has SD `{_fmt_pct(six['sd'])}` because every specimen is at `{_fmt_pct(six['mean'])}` success.")
    lines.append("")
    lines.append("Simulator release status was not verified from these local tables. Keep the availability sentence conditional on the actual repository/archive release state.")
    lines.append("")
    lines.append("Timescale separation should be omitted from experimental claims unless a fixed/homogeneous-tau necessity assay is added. The tau distribution can remain descriptive architecture context only.")
    lines.append("")

    lines.append("## S12. S4 validity / reproducibility")
    lines.append("")
    lines.append("Full exports: `tables/s12_complete_runtime_summary.csv`, `tables/s12_region_summary.csv`, and `tables/s12_pair_matrix.csv`. The expanded probe includes native/interpolated pairs `(1,1)`, `(2,1)`, `(3,2)`, `(4,2)`, `(5,3)`, `(6,3)` and extended/interpolated pairs `(7,4)`, `(8,4)`, `(9,5)`, `(10,5)`, mirrored left/right, for intact and deformation-off. Complete runtimes exclude the interrupted `local_cpu_jax0901` attempt.")
    lines.append("")
    overall = s12_overall.copy()
    overall["success"] = overall.apply(lambda r: f"{int(r['success_count'])}/{int(r['n'])}", axis=1)
    lines.append("### Complete runtime totals across all 10 pairs")
    lines.append(_table(overall, ["runtime", "assay", "n", "success", "fall_count", "edge_reach_count", "returned_from_edge_count", "full_reverse_and_goal", "cliff_stop_only", "direct_goal_no_edge", "returned_no_goal", "stall_no_edge"]))
    lines.append("")
    orig = s12_regions[s12_regions["region"] == "original_native_3_pairs"].copy()
    orig["success"] = orig.apply(lambda r: f"{int(r['success_count'])}/{int(r['n'])}", axis=1)
    lines.append("### Original native three-pair subset")
    lines.append(_table(orig, ["runtime", "assay", "n", "success", "fall_count", "full_reverse_and_goal", "cliff_stop_only", "direct_goal_no_edge", "returned_no_goal", "stall_no_edge"]))
    lines.append("")
    lines.append("Interpretation: safety is robust in these completed runs (`fall_count = 0` throughout), but reversal-and-goal behavior is backend-sensitive, especially under deformation-off. A100-40GB, A100-80GB, and L4 agree on the full expanded deformation-off total (15/20 success, 8 full reverse-and-goal); local CPU JAX 0.9.2 gives 2/20 success with 0 full reverse-and-goal; T4 gives 10/20 success but those are direct-goal/no-edge successes rather than full reversals; H100 gives 5/20. Matching local JAX to 0.9.2 did not recover the A100/L4 deformation-off behavior, so the divergence is not just the earlier local JAX version.")
    lines.append("")
    lines.append("Float64, bit-identical repeated-run determinism, and RK4 substep-resolution checks were not completed in this packet. The cross-hardware state-space traces do provide numeric trajectory sanity checks: successful `full_reverse_and_goal` rows have a recorded cliff reach, `returned_from_edge=True`, no fall, and goal touch, so the reversal is a physical trajectory endpoint rather than a scoring-only artifact.")
    lines.append("")

    lines.append("## Figure-data exports")
    lines.append("")
    lines.append("- Generalization vectors: `tables/s1_rate_vectors.csv` for native/extended success; `tables/s2_double_cliff_survival_matrix.csv` and `tables/s2_double_cliff_width_counts.csv` for double-cliff survival.")
    lines.append("- Ablation bars/dots: `tables/s1_master_rates.csv`, `tables/s1_rate_vectors.csv`, `tables/s3_paired_ablation_deltas.csv`, `tables/s3_paired_ablation_delta_vectors.csv`.")
    lines.append("- S4 figure: existing `figures/s4_mechanism_packet/*`, `eight_specimen_clean_packet_20260613/tables/s4_signal_cell_edge_impulse_lag8_auc_summary.csv`, and new `tables/s6_s4_mechanical_snapshot.csv`.")
    lines.append("- Cross-hardware validity: `tables/s12_complete_runtime_summary.csv`, `tables/s12_region_summary.csv`, `tables/s12_pair_matrix.csv`.")
    lines.append("")

    lines.append("## Paper-safe summary")
    lines.append("")
    lines.append("- The eight selected specimens retain strong native performance under specimen-level SD reporting, with all native failures concentrated on `(6,3)`.")
    lines.append("- Native signalling and deformation ablations have large negative within-specimen deltas; these paired deltas are the safest headline quantities.")
    lines.append("- Body transfer is complete for all eight specimens; 6x2 is exactly 50% success for every specimen.")
    lines.append("- Frozen sensing has been run and supports the claim that ongoing sensing is load-bearing at test time.")
    lines.append("- S4 remains a useful mechanistic case study, but its deformation-off reversal regime is backend-sensitive. Report this explicitly and avoid causal overclaiming.")
    lines.append("- The current S4 decoder evidence supports body/frontier-band state being more direction-predictive than hidden/internal signalling state in the completed two-stream analysis. The newly requested Sobel-filtered signalling-field-gradient stream remains an additional analysis if you want the conduit hypothesis in the main text.")
    lines.append("")

    path = OUT_ROOT / "sensing_edges_statistics_requirements_response_2026-06-14.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(EPISODE_MANIFEST)
    master, vectors = build_master_rates(df)
    double_matrix, double_counts = build_double_cliff_matrix(df)
    deltas, delta_vectors = build_paired_deltas(df)
    with_without = build_with_without_s4(build_rate_specs(), df)
    s6 = build_s6_snapshots()
    s8 = build_s8_timing()
    s12_overall, s12_regions, s12_pairs = build_s12_hardware()

    outputs = {
        "s1_master_rates.csv": master,
        "s1_rate_vectors.csv": vectors,
        "s2_double_cliff_survival_matrix.csv": double_matrix,
        "s2_double_cliff_width_counts.csv": double_counts,
        "s3_paired_ablation_deltas.csv": deltas,
        "s3_paired_ablation_delta_vectors.csv": delta_vectors,
        "s4_with_without_s4.csv": with_without,
        "s6_s4_mechanical_snapshot.csv": s6,
        "s8_s4_reversal_timing_summary.csv": s8,
        "s12_complete_runtime_summary.csv": s12_overall,
        "s12_region_summary.csv": s12_regions,
        "s12_pair_matrix.csv": s12_pairs,
    }
    for name, frame in outputs.items():
        frame.to_csv(TABLES_DIR / name, index=False)

    md_path = write_markdown(
        master=master,
        vectors=vectors,
        double_matrix=double_matrix,
        double_counts=double_counts,
        deltas=deltas,
        with_without=with_without,
        s6=s6,
        s8=s8,
        s12_overall=s12_overall,
        s12_regions=s12_regions,
        s12_pairs=s12_pairs,
    )
    manifest = {
        "schema_version": 1,
        "kind": "sensing_edges_statistics_requirements_packet",
        "created": "2026-06-14",
        "output_root": str(OUT_ROOT),
        "markdown": str(md_path),
        "tables": {name: str(TABLES_DIR / name) for name in outputs},
        "source_episode_manifest": str(EPISODE_MANIFEST),
        "source_s4_mechanism_packet": str(MECH_DIR),
        "source_cross_hardware_package": str(HARDWARE_ROOT),
    }
    (OUT_ROOT / "packet_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(md_path)


if __name__ == "__main__":
    main()
