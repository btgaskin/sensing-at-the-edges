from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPECIMEN_ROOT = ROOT / "results/bin_03_rank_00_cell_6-0-3_flat_0423"
ARCHIVE_ROOT = SPECIMEN_ROOT / (
    "source_runs/current_core_full/_previous/"
    "2026-04-13T06-27-50.032355_00-00"
)


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    path: Path
    lane: str
    archived_distance_pair: str | None = None
    archived_side: str | None = None


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _find_row(
    rows: list[dict[str, str]],
    **filters: str,
) -> dict[str, str] | None:
    for row in rows:
        if all(row.get(key) == value for key, value in filters.items()):
            return row
    return None


def _probe_metrics(npz_path: Path, lane: str) -> dict[str, float | int]:
    bundle = np.load(npz_path)
    positions = bundle[f"{lane}_positions"]
    env_steps = bundle["env_steps"]
    com_x = positions[:, :, 0].mean(axis=1)
    front_x = positions[:, :, 0].max(axis=1)

    # Keep one sample per env step for a more summary-like motion metric.
    step_change = np.r_[True, env_steps[1:] != env_steps[:-1]]
    env_step_com_x = com_x[step_change]

    return {
        "phase_count": int(len(com_x)),
        "env_step_count": int(np.unique(env_steps).size),
        "final_com_x": float(com_x[-1]),
        "peak_front_x": float(front_x.max()),
        "min_com_x": float(com_x.min()),
        "max_com_x": float(com_x.max()),
        "mean_abs_com_x_delta_per_env_step": float(
            np.mean(np.abs(np.diff(env_step_com_x)))
        ),
    }


def _float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _archived_native_core_metrics(
    per_episode_rows: list[dict[str, str]],
    distance_pair: str,
    side: str,
) -> dict[str, float | str | None] | None:
    row = _find_row(
        per_episode_rows,
        assay_family="native_core",
        distance_pair=distance_pair,
        side=side,
    )
    if row is None:
        return None
    return {
        "condition": row["condition_id"],
        "deformation_condition": row["deformation_condition"],
        "signaling_condition": row["signaling_condition"],
        "success": row["success"],
        "fell": row["fell"],
        "final_com_x": _float_or_none(row["final_com_x"]),
        "peak_front_x": _float_or_none(row["peak_front_x"]),
        "mean_speed": _float_or_none(row["mean_speed"]),
    }


def _archived_pca_bundle_metrics(trial_manifest_rows: list[dict[str, str]]) -> dict[str, str | float]:
    row = _find_row(trial_manifest_rows, trial_id="pair_4_2_right_000")
    if row is None:
        raise ValueError("Missing pair_4_2_right_000 in deformation_off trial manifest")
    return {
        "trial_id": row["trial_id"],
        "distance_pair": row["distance_pair"],
        "side": row["side"],
        "final_com_x": float(row["final_com_x"]),
        "final_front_x": float(row["final_front_x"]),
        "steps_alive": int(row["steps_alive"]),
        "phase_steps_alive": int(row["phase_steps_alive"]),
        "goal_touched": row["goal_touched"],
        "fell_flag": row["fell_flag"],
        "termination_reason": row["termination_reason"],
    }


def main() -> None:
    probes = [
        ProbeSpec(
            name="current_right_native_2_1_deformation_off",
            path=SPECIMEN_ROOT
            / "analysis_runs/right_native_pair_2_1_vs_6_3_4000/block_off_pair_compare/trace_bundle.npz",
            lane="top",
            archived_distance_pair="(2,1)",
            archived_side="right",
        ),
        ProbeSpec(
            name="current_right_native_6_3_deformation_off",
            path=SPECIMEN_ROOT
            / "analysis_runs/right_native_pair_2_1_vs_6_3_4000/block_off_pair_compare/trace_bundle.npz",
            lane="bottom",
            archived_distance_pair="(6,3)",
            archived_side="right",
        ),
        ProbeSpec(
            name="current_double_cliff_10_deformation_off",
            path=SPECIMEN_ROOT
            / "analysis_runs/right_native_goalblock_doublecliff_probe_4000_touchable_v2/native_vs_double_cliff10/trace_bundle.npz",
            lane="bottom",
        ),
    ]

    native_core_rows = _load_csv_rows(ARCHIVE_ROOT / "native_core/per_episode.csv")
    pca_trial_rows = _load_csv_rows(
        ARCHIVE_ROOT / "pca_bundle/cohorts/deformation_off_native_core/trial_manifest.csv"
    )

    report: dict[str, object] = {
        "archived_sources": {
            "native_core_per_episode": str(ARCHIVE_ROOT / "native_core/per_episode.csv"),
            "deformation_off_pca_trial_manifest": str(
                ARCHIVE_ROOT / "pca_bundle/cohorts/deformation_off_native_core/trial_manifest.csv"
            ),
        },
        "note": (
            "Archived native_core/per_episode.csv is intact-only. "
            "The archived deformation_off time-resolved bundle is only the selected right (4,2) trial."
        ),
        "current_vs_archived": {},
        "archived_deformation_off_reference": _archived_pca_bundle_metrics(pca_trial_rows),
    }

    current_vs_archived = report["current_vs_archived"]
    assert isinstance(current_vs_archived, dict)

    for probe in probes:
        probe_summary = _probe_metrics(probe.path, probe.lane)
        entry: dict[str, object] = {
            "probe_bundle": str(probe.path),
            "lane": probe.lane,
            "current_probe_metrics": probe_summary,
        }
        if probe.archived_distance_pair and probe.archived_side:
            archived = _archived_native_core_metrics(
                native_core_rows,
                distance_pair=probe.archived_distance_pair,
                side=probe.archived_side,
            )
            entry["archived_native_core_intact"] = archived
            if archived is not None:
                entry["delta_vs_archived_intact"] = {
                    "final_com_x": probe_summary["final_com_x"] - archived["final_com_x"],
                    "peak_front_x": probe_summary["peak_front_x"] - archived["peak_front_x"],
                }
        current_vs_archived[probe.name] = entry

    output_dir = SPECIMEN_ROOT / "analysis_runs/direct_summary_compare"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(report, indent=2))
    print(output_path)


if __name__ == "__main__":
    main()
