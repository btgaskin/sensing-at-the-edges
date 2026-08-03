#!/usr/bin/env python3
"""Merge targeted follow-up assays into the eight-specimen preprint package.

This is an analysis/package-maintenance script. It reads deterministic Modal
post-run outputs that were pulled locally, stamps the same provenance fields as
the original package, and rewrites the package episode manifest after removing
older copies of the same follow-up assay families.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd


DATA_ROOT = Path("paper_data/preprint/native_core_ablation_v1")
DEFAULT_FROZEN_ROOT = (
    DATA_ROOT
    / "raw_modal_frozen_sensing_control_20260613T041500"
    / "preprint_native_core_ablation_v1"
)
DEFAULT_JOINT_ROOT = (
    DATA_ROOT
    / "raw_modal_s4_joint_ablation_20260613T041600"
    / "preprint_native_core_ablation_v1"
)
FOLLOWUP_ASSAYS = {"frozen_sensing_control", "combined_ablation_quick"}


def _bool_coerce(series: pd.Series) -> pd.Series:
    return series.map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})


def _load_provenance(data_root: Path) -> pd.DataFrame:
    provenance = pd.read_csv(data_root / "tables" / "provenance.csv")
    required = {"group_id", "run_id", "specimen_id"}
    missing = sorted(required - set(provenance.columns))
    if missing:
        raise ValueError(f"provenance.csv missing required columns: {missing}")
    return provenance


def _stamp_provenance(frame: pd.DataFrame, provenance: pd.DataFrame, *, source_root: Path) -> pd.DataFrame:
    if frame.empty:
        return frame
    merged = frame.merge(
        provenance[["group_id", "run_id", "specimen_id"]].rename(
            columns={"run_id": "source_run_id"}
        ),
        how="left",
        on="specimen_id",
        validate="many_to_one",
    )
    missing = sorted(merged.loc[merged["source_run_id"].isna(), "specimen_id"].astype(str).unique())
    if missing:
        raise ValueError(f"follow-up rows missing provenance for specimens: {missing}")
    merged["source_specimen_id"] = merged["specimen_id"].astype(str)
    merged["assay"] = merged["assay_family"].astype(str)
    merged["surface_package_label"] = merged["surface_id"].astype(str).map(
        {
            "native_core": "native_core",
            "long_distance_generalization": "extended",
            "double_cliff": "symmetric",
        }
    ).fillna(merged["surface_id"].astype(str))
    merged["followup_source_root"] = str(source_root)
    return merged


def _load_followup_csvs(root: Path, assay: str, provenance: pd.DataFrame) -> pd.DataFrame:
    if not root.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob(f"*/*/{assay}/per_episode.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["assay_family"] = str(assay)
        if "surface_id" not in frame.columns:
            frame["surface_id"] = "native_core"
        frames.append(_stamp_provenance(frame, provenance, source_root=root))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _write_followup_summary(data_root: Path, followups: pd.DataFrame) -> None:
    if followups.empty:
        summary = pd.DataFrame()
    else:
        work = followups.copy()
        for column in ("success", "fell", "survived"):
            if column in work.columns:
                work[column] = _bool_coerce(work[column])
        rows: list[dict[str, object]] = []
        for (assay, surface, condition), group in work.groupby(
            ["assay_family", "surface_id", "condition_id"],
            dropna=False,
            sort=True,
        ):
            endpoint = "survived" if str(surface) == "double_cliff" else "success"
            rates = group.groupby("specimen_id")[endpoint].mean() if endpoint in group.columns else pd.Series(dtype=float)
            rows.append(
                {
                    "assay_family": assay,
                    "surface_id": surface,
                    "condition_id": condition,
                    "condition_label": str(group["condition_label"].dropna().iloc[0])
                    if "condition_label" in group.columns and len(group["condition_label"].dropna())
                    else str(condition),
                    "endpoint": endpoint,
                    "n_specimens": int(rates.size),
                    "episode_count": int(len(group)),
                    "positive_count": int(group[endpoint].sum()) if endpoint in group.columns else 0,
                    "episode_rate": float(group[endpoint].mean()) if endpoint in group.columns and len(group) else float("nan"),
                    "mean_specimen_rate": float(rates.mean()) if rates.size else float("nan"),
                    "fall_episode_rate": float(group["fell"].mean()) if "fell" in group.columns and len(group) else float("nan"),
                }
            )
        summary = pd.DataFrame(rows)
    summary.to_csv(data_root / "tables" / "followup_control_summary.csv", index=False)


def merge_followups(*, data_root: Path, frozen_root: Path, joint_root: Path) -> dict[str, object]:
    tables_dir = data_root / "tables"
    manifest_path = tables_dir / "episode_manifest.csv"
    backup_path = tables_dir / "episode_manifest.pre_followups_20260613.csv"
    if not backup_path.exists():
        shutil.copy2(manifest_path, backup_path)

    base = pd.read_csv(manifest_path)
    provenance = _load_provenance(data_root)
    followup_frames = [
        _load_followup_csvs(frozen_root, "frozen_sensing_control", provenance),
        _load_followup_csvs(joint_root, "combined_ablation_quick", provenance),
    ]
    followups = pd.concat(
        [frame for frame in followup_frames if not frame.empty],
        ignore_index=True,
    ) if any(not frame.empty for frame in followup_frames) else pd.DataFrame()
    retained = base[~base["assay_family"].astype(str).isin(FOLLOWUP_ASSAYS)].copy()
    merged = pd.concat([retained, followups], ignore_index=True, sort=False)
    merged.to_csv(manifest_path, index=False)
    _write_followup_summary(data_root, followups)
    summary = {
        "episode_manifest": str(manifest_path),
        "backup_manifest": str(backup_path),
        "retained_rows": int(len(retained)),
        "followup_rows": int(len(followups)),
        "total_rows": int(len(merged)),
        "followup_assays": sorted(followups["assay_family"].dropna().unique().tolist())
        if not followups.empty
        else [],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--frozen-root", type=Path, default=DEFAULT_FROZEN_ROOT)
    parser.add_argument("--joint-root", type=Path, default=DEFAULT_JOINT_ROOT)
    args = parser.parse_args()
    summary = merge_followups(
        data_root=args.data_root,
        frozen_root=args.frozen_root,
        joint_root=args.joint_root,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
