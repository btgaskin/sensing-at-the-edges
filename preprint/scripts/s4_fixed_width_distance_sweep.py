#!/usr/bin/env python
"""Run and analyze the fixed-width S4 cliff-distance sweep.

This packet supersedes the plateau-width event-multiplication probe for the
main S4 mechanism analysis. It keeps the grid, plateau width, spawn, specimen,
deformation condition, and signalling condition fixed, then varies only the
cliff distance along the direction of S4's cliffward travel.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def _resolve_repo_root() -> Path:
    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        if (parent / "sensing_edges").exists():
            return parent
    if Path("/root/experiment").exists():
        return Path("/root")
    return module_path.parents[2]


REPO_ROOT = _resolve_repo_root()
SCRIPT_DIR = Path(__file__).resolve().parent
SENSING_EDGES_ROOT = REPO_ROOT / "sensing_edges"
for import_path in (SCRIPT_DIR, SENSING_EDGES_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from committed_direction_decoding import _auc, _balanced_accuracy, _ridge_scores, _stable_seed
from voxel_reversal_heatmap_analysis import _cell_spring_indices


DEFAULT_PACKAGE_ROOT = REPO_ROOT / "paper_data" / "preprint" / "s4_fixed_width_distance_sweep_20260618"
PACKAGE_ROOT = Path(os.environ.get("S4_FIXED_WIDTH_PACKAGE_ROOT", str(DEFAULT_PACKAGE_ROOT)))
REMOTE_ROOT = "/outputs/preprint_s4_fixed_width_distance_sweep_20260618"
RUN_ID = "paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed137_e4fcc2fb"
SPECIMEN_ID = "bin_03_rank_00_cell_6-0-3_flat_0423"
SOURCE_STAGE_ROOT = REPO_ROOT / "run_specimens" / RUN_ID / "stages" / "gen_0099"
SOURCE_SPECIMEN_DIR = SOURCE_STAGE_ROOT / f"specimen_{SPECIMEN_ID}"
PACKAGE_STAGE_ROOT = PACKAGE_ROOT / "specimen" / "stages" / "gen_0099"
PACKAGE_SPECIMEN_DIR = PACKAGE_STAGE_ROOT / f"specimen_{SPECIMEN_ID}"
PACKAGE_RUN_CONTEXT_DIR = PACKAGE_ROOT / "specimen" / "run_context" / RUN_ID

GPU_PROFILE = "a100-40gb"
ASSAY = "state_space_trace_deformation_off"
GOAL_SIDE = "right"
GRID_W = 72
PLATEAU_WIDTH = 12
GOAL_DISTANCE = 12
CLIFF_DISTANCES = tuple(range(1, 31))
MAX_STEPS = 30_000
BATCH_SIZE = 2
REL_TIMES = tuple(range(-128, 337, 8))
PHASE_REL_TIMES = tuple(range(0, 337, 8))
CONTINUER_MARGIN_STEPS = 64
RIDGE_ALPHA = 10.0
PERMUTATIONS = 1000
PHASE_PERMUTATIONS = 1000
PHASE_FIT_STRIDE = 4
PROPAGATION_REL_TIMES = tuple(range(-64, 513, 8))
PROPAGATION_PERMUTATIONS = 1000
PROPAGATION_DIRECTION_FUTURE_WINDOW = 32
PROPAGATION_DIRECTION_MIN_DISPLACEMENT = 0.025
PROPAGATION_DIRECTION_PERSISTENCE_STEPS = 12
PROPAGATION_DIRECTION_PERSISTENCE_FRACTION = 0.75
PROPAGATION_PHASE_ONSET_MIN_RAD = 0.02
PROPAGATION_PHASE_ONSET_SD_MULTIPLIER = 3.0
PROPAGATION_FEATURE_DUST_FLOOR = 1.0e-5
GRAPH_PROPAGATION_REL_TIMES = tuple(range(-64, 513, 8))
GRAPH_PROPAGATION_ONSET_SD_MULTIPLIER = 3.0
GRAPH_PROPAGATION_ONSET_MIN_EFFECT = 1.0e-4
PCA_REGIME_REL_TIMES = tuple(range(-64, 513, 8))
PCA_REGIME_FIT_STRIDE = 4
PCA_REGIME_COMPONENTS = 3
PCA_REGIME_ONSET_SUSTAIN_STEPS = 3
EXACT_TOLERANCE = 0.0
REQUIRED_ASSAY_FILES = (
    "manifest.json",
    "trial_manifest.csv",
    "summary.json",
    "step_manifest.csv",
    "static_arrays.npz",
    "edge_event_audit.csv",
    "aggregate.json",
)
REQUIRED_TRACE_ARRAYS = (
    "boxel_corners",
    "cell_corner_positions",
    "plant_positions",
    "plant_velocities_true",
    "spring_a_idx",
    "spring_b_idx",
    "spring_current_length",
    "spring_rest_length",
    "spring_pre_rest_length",
    "phase_post_spring_rest_length",
    "spring_init_rest_length",
    "phase_nca_state",
    "phase_ctrnn_carry",
    "phase_h_drive",
    "phase_v_drive",
    "phase_action_outputs",
)
PREDICTIVE_STREAMS = (
    "signal_hidden",
    "controller_action",
    "body_kinematic",
    "spring_current",
    "spring_contractile",
    "body_no_contractile",
    "body_combined",
)
LEGACY_STREAM_ALIASES = {"signal": "signal_hidden", "body": "body_combined"}
FOCUS_STREAMS = (
    "signal_hidden",
    "controller_action",
    "body_kinematic",
    "spring_current",
    "spring_contractile",
    "body_no_contractile",
    "body_combined",
)
PHASE_STREAMS = FOCUS_STREAMS
PHASE_FIGURE_STREAMS = (
    "signal_hidden",
    "controller_action",
    "spring_contractile",
    "spring_current",
    "body_kinematic",
    "body_no_contractile",
)
PROPAGATION_STREAMS = PHASE_STREAMS
PROPAGATION_FIGURE_STREAMS = PHASE_FIGURE_STREAMS


def _data_root() -> Path:
    return Path(os.environ.get("S4_FIXED_WIDTH_DATA_ROOT", str(PACKAGE_ROOT / "modal" / GPU_PROFILE)))


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


def _omega_pairs(pairs: Iterable[tuple[int, int]]) -> str:
    return "[" + ",".join(
        f'{{"goal_distance":{int(goal)},"cliff_distance":{int(cliff)}}}'
        for goal, cliff in pairs
    ) + "]"


def _omega_list(values: Iterable[str]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def _shell_join(argv: list[str]) -> str:
    import shlex

    return shlex.join(argv)


def _pairs() -> tuple[tuple[int, int], ...]:
    return tuple((GOAL_DISTANCE, cliff_distance) for cliff_distance in CLIFF_DISTANCES)


def _distance_batches() -> tuple[tuple[int, tuple[tuple[int, int], ...]], ...]:
    pairs = _pairs()
    return tuple(
        (batch_index, pairs[start : start + BATCH_SIZE])
        for batch_index, start in enumerate(range(0, len(pairs), BATCH_SIZE))
    )


def _remote_analysis_root(batch_index: int | None = None) -> str:
    if batch_index is None:
        return REMOTE_ROOT
    return f"{REMOTE_ROOT}/batch_{int(batch_index):02d}"


def _modal_volume_path() -> str:
    return _remote_analysis_root().removeprefix("/outputs/")


def _s4_item() -> dict[str, str]:
    import native_core_ablation as native

    return native.state_space_specimen()


def _copy_required_file(source: Path, dest: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Required S4 artifact missing: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def initialize_package() -> Path:
    for rel in ("specimen", "manifests", "modal", "tables", "figures", "notes"):
        (PACKAGE_ROOT / rel).mkdir(parents=True, exist_ok=True)

    for name in ("manifest.json", "selected_specimens.json", "selected_genotypes.npz", "terrain_manifest.json"):
        source = SOURCE_STAGE_ROOT / name
        if source.exists():
            _copy_required_file(source, PACKAGE_STAGE_ROOT / name)

    if not SOURCE_SPECIMEN_DIR.exists():
        raise FileNotFoundError(f"S4 source specimen directory not found: {SOURCE_SPECIMEN_DIR}")
    if not PACKAGE_SPECIMEN_DIR.exists():
        shutil.copytree(
            SOURCE_SPECIMEN_DIR,
            PACKAGE_SPECIMEN_DIR,
            ignore=shutil.ignore_patterns("post_run", "preview_renders", "charts"),
        )
    PACKAGE_RUN_CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    if (PACKAGE_STAGE_ROOT / "manifest.json").exists():
        _copy_required_file(PACKAGE_STAGE_ROOT / "manifest.json", PACKAGE_RUN_CONTEXT_DIR / "manifest.json")
    _copy_required_file(PACKAGE_SPECIMEN_DIR / "config_snapshot.yaml", PACKAGE_RUN_CONTEXT_DIR / "resolved_config.yaml")

    manifest = {
        "schema_version": 1,
        "kind": "s4_fixed_width_distance_sweep_package",
        "created_for": "S4 fixed-width deformation-off cliff-distance sweep",
        "package_root": PACKAGE_ROOT,
        "run_id": RUN_ID,
        "specimen_id": SPECIMEN_ID,
        "gpu_profile": GPU_PROFILE,
        "assay": ASSAY,
        "condition": "deformation_off",
        "goal_side": GOAL_SIDE,
        "grid_w": GRID_W,
        "plateau_width": PLATEAU_WIDTH,
        "goal_distance": GOAL_DISTANCE,
        "cliff_distances": list(CLIFF_DISTANCES),
        "max_steps": MAX_STEPS,
        "batch_size": BATCH_SIZE,
        "batch_count": len(_distance_batches()),
        "expected_config_count": len(CLIFF_DISTANCES),
        "decoder": {
            "relative_times": list(REL_TIMES),
            "continuer_margin_steps": CONTINUER_MARGIN_STEPS,
            "ridge_alpha": RIDGE_ALPHA,
            "permutations": PERMUTATIONS,
            "label": "release-aligned trace versus same-absolute-time continuing trace",
        },
    }
    _write_json(PACKAGE_ROOT / "manifests" / "package_manifest.json", manifest)
    (PACKAGE_ROOT / "notes" / "README.md").write_text(
        "# S4 fixed-width cliff-distance sweep\n\n"
        "Canonical Modal A100-40GB deformation-off S4 state-space packet.\n\n"
        "- Goal-cliff only; no double-cliff traces are included.\n"
        f"- Fixed grid width `{GRID_W}`, plateau/base width `{PLATEAU_WIDTH}`, goal side `{GOAL_SIDE}`, and goal distance `{GOAL_DISTANCE}`.\n"
        f"- Cliff distance varies from `{CLIFF_DISTANCES[0]}` to `{CLIFF_DISTANCES[-1]}`.\n"
        "- The packet is isolated from the earlier plateau-width sweep.\n"
        "- S4 genotype/specimen artifacts are copied under `specimen/` for provenance.\n",
    )
    return PACKAGE_ROOT


def _override_tokens(
    *,
    analysis_root: str,
    specimens_output_root: str,
    pairs: Iterable[tuple[int, int]],
) -> list[str]:
    return [
        f"eval.post_run.analysis_root={analysis_root}",
        "eval.post_run.state_space_surfaces=[native_core]",
        "eval.post_run.state_space_deformation_condition=deformation_off",
        "eval.post_run.state_space_signaling_condition=intact",
        f"eval.post_run.native_pairs={_omega_pairs(pairs)}",
        f"eval.post_run.goal_sides={_omega_list((GOAL_SIDE,))}",
        f"eval.post_run.native_max_steps={MAX_STEPS}",
        f"eval.post_run.max_steps={MAX_STEPS}",
        "eval.post_run.fast_ablation_eval=false",
        f"terrain.grid_w={GRID_W}",
        f"terrain.center_plateau_width={PLATEAU_WIDTH}",
        f"terrain.base_width={PLATEAU_WIDTH}",
        f"specimens.output_root={specimens_output_root}",
    ]


def modal_launch_argv(*, batch_index: int, pairs: Iterable[tuple[int, int]]) -> list[str]:
    item = _s4_item()
    return [
        "uv",
        "run",
        "--python",
        ".venv/bin/python",
        "modal",
        "run",
        "--detach",
        "sensing_edges/scripts/modal_post_run.py",
        "--run-dir",
        item["remote_run_dir"],
        "--specimen-id",
        item["specimen_id"],
        "--config-path",
        item["remote_config_path"],
        "--assays",
        ASSAY,
        "--override",
        ",".join(
            _override_tokens(
                analysis_root=_remote_analysis_root(batch_index),
                specimens_output_root=item["specimens_output_root"],
                pairs=pairs,
            )
        ),
        "--gpu-profile",
        GPU_PROFILE,
        "--detach-launch",
    ]


def pull_argv() -> list[str]:
    return [
        "uv",
        "run",
        "--python",
        ".venv/bin/python",
        "modal",
        "volume",
        "get",
        "se-outputs",
        _modal_volume_path(),
        str(PACKAGE_ROOT / "modal" / GPU_PROFILE),
    ]


def write_launch_manifest() -> Path:
    initialize_package()
    jobs = []
    for batch_index, pairs in _distance_batches():
        launch = modal_launch_argv(batch_index=batch_index, pairs=pairs)
        jobs.append(
            {
                "batch_index": int(batch_index),
                "cliff_distances": [int(cliff) for _, cliff in pairs],
                "remote_analysis_root": _remote_analysis_root(batch_index),
                "modal_volume_path": _remote_analysis_root(batch_index).removeprefix("/outputs/"),
                "launch_argv": launch,
                "launch_command": _shell_join(launch),
                "expected_trace_count": len(pairs),
            }
        )
    pull = pull_argv()
    path = PACKAGE_ROOT / "manifests" / "modal_launch_manifest.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "kind": "s4_fixed_width_distance_sweep_modal_launch_manifest",
            "jobs": jobs,
            "pull": {
                "remote_analysis_root": _remote_analysis_root(),
                "modal_volume_path": _modal_volume_path(),
                "pull_argv": pull,
                "pull_command": _shell_join(pull),
            },
            "expected": {
                "jobs": len(jobs),
                "batch_size": BATCH_SIZE,
                "trace_npz_total": len(CLIFF_DISTANCES),
            },
        },
    )
    return path


def _load_launch_manifest() -> dict[str, Any]:
    with write_launch_manifest().open() as handle:
        return json.load(handle)


def launch_modal(*, dry_run: bool = False) -> None:
    jobs = _load_launch_manifest()["jobs"]
    for job in jobs:
        print(job["launch_command"])
        if not dry_run:
            subprocess.run(job["launch_argv"], cwd=REPO_ROOT, check=True)


def pull_modal(*, dry_run: bool = False) -> None:
    job = _load_launch_manifest()["pull"]
    print(job["pull_command"])
    if not dry_run:
        dest = Path(job["pull_argv"][-1])
        if dest.exists():
            shutil.rmtree(dest)
        subprocess.run(job["pull_argv"], cwd=REPO_ROOT, check=True)


def _find_assay_roots() -> list[Path]:
    root = _data_root()
    return sorted(path.parent for path in root.glob(f"**/{ASSAY}/trial_manifest.csv"))


def _expected_counts_by_batch_root() -> dict[str, int]:
    expected: dict[str, int] = {}
    manifest_path = PACKAGE_ROOT / "manifests" / "modal_launch_manifest.json"
    if not manifest_path.exists():
        return expected
    payload = json.loads(manifest_path.read_text())
    for job in payload.get("jobs", []):
        marker = f"batch_{int(job['batch_index']):02d}"
        expected[marker] = int(job.get("expected_trace_count", 0))
    return expected


def _expected_count_for_assay_root(assay_root: Path) -> int:
    text = str(assay_root)
    marker = _batch_marker(assay_root)
    if marker.startswith("batch_"):
        try:
            batch_index = int(marker.split("_", 1)[1])
            batches = dict(_distance_batches())
            if batch_index in batches:
                return len(batches[batch_index])
        except (ValueError, IndexError):
            pass
    for marker, count in _expected_counts_by_batch_root().items():
        if marker in text:
            return int(count)
    return len(CLIFF_DISTANCES)


def _bool_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def summarize(*, init_package: bool = True) -> dict[str, Path]:
    if init_package:
        initialize_package()
    else:
        for rel in ("tables", "figures", "notes"):
            (PACKAGE_ROOT / rel).mkdir(parents=True, exist_ok=True)
    roots = _find_assay_roots()
    trial_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    for assay_root in roots:
        trial_df = pd.read_csv(assay_root / "trial_manifest.csv")
        expected_count = _expected_count_for_assay_root(assay_root)
        traces_dir = assay_root / "traces"
        trace_count = len(list(traces_dir.glob("*.npz"))) if traces_dir.exists() else 0
        reached = int(trial_df["first_cliff_reach_step"].notna().sum()) if "first_cliff_reach_step" in trial_df else 0
        returned = int(trial_df["returned_from_edge"].map(_bool_value).sum()) if "returned_from_edge" in trial_df else 0
        inventory_rows.append(
            {
                "assay_root": str(assay_root),
                "trial_count": int(len(trial_df)),
                "expected_trial_count": expected_count,
                "trace_count": trace_count,
                "expected_trace_count": expected_count,
                "edge_reach_count": reached,
                "returned_from_edge_count": returned,
                "complete": int(len(trial_df) == expected_count and trace_count == expected_count),
            }
        )
        for _, row in trial_df.iterrows():
            record = row.to_dict()
            record["assay_root"] = str(assay_root)
            record["event_id"] = f"cd{int(record.get('cliff_distance')):02d}_{record.get('trial_id')}"
            record["true_release_event"] = bool(pd.notna(record.get("first_cliff_reach_step")))
            trial_rows.append(record)

    trial_path = PACKAGE_ROOT / "tables" / "trial_manifest_combined.csv"
    inventory_path = PACKAGE_ROOT / "tables" / "trace_inventory.csv"
    pd.DataFrame(trial_rows).to_csv(trial_path, index=False)
    pd.DataFrame(inventory_rows).to_csv(inventory_path, index=False)
    event_count = int(sum(1 for row in trial_rows if row.get("true_release_event")))
    returned_count = int(sum(1 for row in trial_rows if _bool_value(row.get("returned_from_edge"))))
    status_path = PACKAGE_ROOT / "notes" / "status.md"
    status_path.write_text(
        "# S4 fixed-width distance sweep status\n\n"
        f"Assay roots found: {len(inventory_rows)} / {len(_distance_batches())}\n\n"
        f"Trials found: {len(trial_rows)} / {len(CLIFF_DISTANCES)}\n\n"
        f"True release events: {event_count}\n\n"
        f"Returned-from-edge events: {returned_count}\n\n"
        "Tables:\n"
        f"- `{trial_path}`\n"
        f"- `{inventory_path}`\n",
    )
    return {"trial_manifest": trial_path, "trace_inventory": inventory_path, "status": status_path}


def _batch_marker(path: Path) -> str:
    for part in path.parts:
        if part.startswith("batch_"):
            return part
    return "unknown"


def _trace_npz_keys(path: Path) -> set[str]:
    with np.load(path) as loaded:
        return set(str(key) for key in loaded.files)


def audit_remote_data(*, init_package: bool = False) -> dict[str, Path]:
    """Audit the fixed-width raw trace packet without copying trace payloads.

    This is intended to run next to the Modal volume. It also works locally if
    the raw packet has already been pulled into the configured data root.
    """
    if init_package:
        initialize_package()
    else:
        for rel in ("tables", "figures", "notes"):
            (PACKAGE_ROOT / rel).mkdir(parents=True, exist_ok=True)

    roots = _find_assay_roots()
    assay_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    observed_cliff_distances: list[int] = []
    required_trace_arrays = set(REQUIRED_TRACE_ARRAYS)

    for assay_root in roots:
        batch = _batch_marker(assay_root)
        expected_count = _expected_count_for_assay_root(assay_root)
        missing_files = [name for name in REQUIRED_ASSAY_FILES if not (assay_root / name).exists()]
        traces_dir = assay_root / "traces"
        trial_df = pd.read_csv(assay_root / "trial_manifest.csv") if (assay_root / "trial_manifest.csv").exists() else pd.DataFrame()
        trace_count = len(list(traces_dir.glob("*.npz"))) if traces_dir.exists() else 0
        reached = int(trial_df["first_cliff_reach_step"].notna().sum()) if "first_cliff_reach_step" in trial_df else 0
        returned = int(trial_df["returned_from_edge"].map(_bool_value).sum()) if "returned_from_edge" in trial_df else 0
        successes = int(trial_df["success"].map(_bool_value).sum()) if "success" in trial_df else 0
        falls = int(trial_df["cliff_fell"].map(_bool_value).sum()) if "cliff_fell" in trial_df else 0
        cliff_values: list[int] = []
        trace_missing_arrays: list[str] = []
        trace_missing_count = 0
        trace_missing_file_count = 0

        for _, trial_row in trial_df.iterrows():
            trial_id = str(trial_row.get("trial_id", ""))
            cliff_distance = int(trial_row["cliff_distance"]) if "cliff_distance" in trial_row and not pd.isna(trial_row["cliff_distance"]) else -1
            if cliff_distance >= 0:
                cliff_values.append(cliff_distance)
                observed_cliff_distances.append(cliff_distance)
            trace_npz = trial_row.get("trace_npz", "")
            try:
                trace_path = _trace_path_for_trial(assay_root, trace_npz, trial_id)
                trace_exists = True
            except FileNotFoundError:
                trace_path = assay_root / "traces" / Path(str(trace_npz)).name
                trace_exists = False
            missing_arrays: list[str] = []
            trace_size = 0
            if trace_exists:
                trace_size = int(trace_path.stat().st_size)
                try:
                    keys = _trace_npz_keys(trace_path)
                    missing_arrays = sorted(required_trace_arrays - keys)
                except Exception as exc:  # pragma: no cover - diagnostic path
                    missing_arrays = [f"npz_read_error:{type(exc).__name__}:{exc}"]
            else:
                trace_missing_file_count += 1
            if missing_arrays:
                trace_missing_count += 1
                trace_missing_arrays.extend(missing_arrays)
            trace_rows.append(
                {
                    "batch": batch,
                    "assay_root": str(assay_root),
                    "trial_id": trial_id,
                    "event_id": f"cd{cliff_distance:02d}_{trial_id}" if cliff_distance >= 0 else trial_id,
                    "goal_distance": int(trial_row["goal_distance"]) if "goal_distance" in trial_row and not pd.isna(trial_row["goal_distance"]) else "",
                    "cliff_distance": cliff_distance if cliff_distance >= 0 else "",
                    "goal_side": trial_row.get("goal_side", ""),
                    "cliff_side": trial_row.get("cliff_side", ""),
                    "first_cliff_reach_step": trial_row.get("first_cliff_reach_step", ""),
                    "returned_from_edge": _bool_value(trial_row.get("returned_from_edge")),
                    "success": _bool_value(trial_row.get("success")),
                    "cliff_fell": _bool_value(trial_row.get("cliff_fell")),
                    "trace_npz": str(trace_npz),
                    "trace_path": str(trace_path),
                    "trace_exists": trace_exists,
                    "trace_size_bytes": trace_size,
                    "missing_required_arrays": ";".join(sorted(set(missing_arrays))),
                }
            )

        assay_complete = (
            int(len(trial_df)) == int(expected_count)
            and int(trace_count) == int(expected_count)
            and not missing_files
            and trace_missing_count == 0
            and trace_missing_file_count == 0
        )
        assay_rows.append(
            {
                "batch": batch,
                "assay_root": str(assay_root),
                "trial_count": int(len(trial_df)),
                "expected_trial_count": int(expected_count),
                "trace_count": int(trace_count),
                "expected_trace_count": int(expected_count),
                "edge_reach_count": reached,
                "returned_from_edge_count": returned,
                "success_count": successes,
                "fall_count": falls,
                "cliff_distances": ";".join(str(value) for value in sorted(cliff_values)),
                "missing_required_files": ";".join(missing_files),
                "trace_missing_file_count": int(trace_missing_file_count),
                "trace_missing_required_array_count": int(trace_missing_count),
                "trace_missing_required_arrays": ";".join(sorted(set(trace_missing_arrays))),
                "complete": int(assay_complete),
            }
        )

    expected_distances = set(int(value) for value in CLIFF_DISTANCES)
    observed_distances = [int(value) for value in observed_cliff_distances]
    missing_distances = sorted(expected_distances - set(observed_distances))
    duplicate_distances = sorted(value for value in set(observed_distances) if observed_distances.count(value) > 1)
    all_complete = (
        len(roots) == len(_distance_batches())
        and len(trace_rows) == len(CLIFF_DISTANCES)
        and not missing_distances
        and not duplicate_distances
        and all(int(row["complete"]) == 1 for row in assay_rows)
    )

    tables_dir = PACKAGE_ROOT / "tables"
    assay_path = tables_dir / "fixed_width_remote_data_audit.csv"
    trace_path = tables_dir / "fixed_width_trace_inventory_corrected.csv"
    summary_path = tables_dir / "fixed_width_remote_data_audit_summary.json"
    report_path = PACKAGE_ROOT / "S4_fixed_width_data_audit_report.md"
    _write_csv(assay_path, assay_rows)
    _write_csv(trace_path, trace_rows)
    summary = {
        "schema_version": 1,
        "kind": "s4_fixed_width_remote_data_audit",
        "package_root": str(PACKAGE_ROOT),
        "data_root": str(_data_root()),
        "expected_batch_count": len(_distance_batches()),
        "observed_batch_count": len(roots),
        "expected_trace_count": len(CLIFF_DISTANCES),
        "observed_trace_rows": len(trace_rows),
        "observed_trace_files": int(sum(1 for row in trace_rows if row["trace_exists"])),
        "missing_cliff_distances": missing_distances,
        "duplicate_cliff_distances": duplicate_distances,
        "complete": bool(all_complete),
        "required_assay_files": list(REQUIRED_ASSAY_FILES),
        "required_trace_arrays": list(REQUIRED_TRACE_ARRAYS),
        "paths": {
            "assay_audit": str(assay_path),
            "trace_inventory": str(trace_path),
            "report": str(report_path),
        },
    }
    _write_json(summary_path, summary)

    lines = [
        "# S4 fixed-width data audit",
        "",
        "## Source",
        "",
        f"- Package root: `{PACKAGE_ROOT}`",
        f"- Data root inspected: `{_data_root()}`",
        f"- Canonical Modal root: `{REMOTE_ROOT}`",
        f"- Specimen: `{SPECIMEN_ID}`",
        f"- Condition: deformation off, intact signalling",
        f"- Fixed goal distance: `{GOAL_DISTANCE}`",
        f"- Expected cliff distances: `{CLIFF_DISTANCES[0]}..{CLIFF_DISTANCES[-1]}`",
        "",
        "## Result",
        "",
        f"- Batches found: `{len(roots)}/{len(_distance_batches())}`",
        f"- Trace rows found: `{len(trace_rows)}/{len(CLIFF_DISTANCES)}`",
        f"- Trace files found: `{summary['observed_trace_files']}/{len(CLIFF_DISTANCES)}`",
        f"- Missing cliff distances: `{missing_distances}`",
        f"- Duplicate cliff distances: `{duplicate_distances}`",
        f"- Complete: `{bool(all_complete)}`",
        "",
        "## Outputs",
        "",
        "- `tables/fixed_width_remote_data_audit.csv`",
        "- `tables/fixed_width_trace_inventory_corrected.csv`",
        "- `tables/fixed_width_remote_data_audit_summary.json`",
        "",
        "## Interpretation",
        "",
    ]
    if all_complete:
        lines.append("The remote canonical fixed-width packet is complete for the remaining derived analyses.")
    else:
        lines.append("The packet is not complete enough for the remaining derived analyses; inspect the audit tables before proceeding.")
    report_path.write_text("\n".join(lines) + "\n")
    return {
        "remote_data_audit": assay_path,
        "trace_inventory_corrected": trace_path,
        "remote_data_audit_summary": summary_path,
        "remote_data_audit_report": report_path,
    }


def _trace_path_for_trial(assay_root: Path, trace_npz_value: Any, trial_id: str) -> Path:
    remote_path = Path(str(trace_npz_value))
    candidate = assay_root / "traces" / remote_path.name
    if candidate.exists():
        return candidate
    fallback = assay_root / "traces" / f"{trial_id}.npz"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not locate trace NPZ for trial {trial_id!r} under {assay_root}")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path)
    try:
        return {key: loaded[key] for key in loaded.files}
    finally:
        loaded.close()


def _slice_time(array: np.ndarray, step_count: int) -> np.ndarray:
    if array.shape[:1] == (step_count,):
        return array
    if array.ndim > 0 and array.shape[0] > step_count:
        return array[:step_count]
    return array


def _safe_scale(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    reference = np.where(np.abs(reference) > 1.0e-6, reference, 1.0).astype(np.float32)
    if values.ndim > reference.ndim:
        extra = values.ndim - reference.ndim
        if values.shape[: reference.ndim] == reference.shape:
            reference = reference.reshape(reference.shape + (1,) * extra)
        elif values.shape[-reference.ndim :] == reference.shape:
            reference = reference.reshape((1,) * extra + reference.shape)
    return values / reference - 1.0


def _cell_rank_map(trace: dict[str, np.ndarray], release_step: int, cliff_side: str) -> dict[int, int]:
    centers = np.asarray(trace["cell_corner_positions"][int(release_step)], dtype=np.float32).mean(axis=1)
    valid = np.flatnonzero(np.asarray(trace["boxel_mask"], dtype=bool))
    reverse = str(cliff_side).strip().lower() == "right"
    ordered = sorted(valid.tolist(), key=lambda cell: float(centers[int(cell), 0]), reverse=reverse)
    return {rank: int(cell) for rank, cell in enumerate(ordered)}


def _signal_hidden_rank_feature(
    event: dict[str, Any],
    *,
    rank: int,
    step: int,
) -> np.ndarray | None:
    cell = event["rank_to_cell"].get(int(rank))
    if cell is None:
        return None
    rows, cols = event["robot_shape"]
    row = int(cell) // int(cols)
    col = int(cell) % int(cols)
    trace = event["trace"]
    hidden_start = int(event["hidden_start_channel"])
    nca = np.asarray(trace["phase_nca_state"][int(step), :, row, col, hidden_start:], dtype=np.float32)
    carry = np.asarray(trace["phase_ctrnn_carry"][int(step), :, row, col, :], dtype=np.float32)
    return np.concatenate([nca.reshape(-1), carry.reshape(-1)]).astype(np.float32)


def _controller_action_rank_feature(
    event: dict[str, Any],
    *,
    rank: int,
    step: int,
) -> np.ndarray | None:
    cell = event["rank_to_cell"].get(int(rank))
    if cell is None:
        return None
    rows, cols = event["robot_shape"]
    row = int(cell) // int(cols)
    col = int(cell) % int(cols)
    trace = event["trace"]
    parts: list[np.ndarray] = []
    if "action_outputs" in trace:
        action = np.asarray(trace["action_outputs"][int(step), row, col, :], dtype=np.float32)
        parts.append(action.reshape(-1))
    if "phase_action_outputs" in trace:
        phase_action = np.asarray(trace["phase_action_outputs"][int(step), :, row, col, :], dtype=np.float32)
        parts.append(phase_action.reshape(-1))
    if not parts:
        return None
    return np.concatenate(parts).astype(np.float32)


def _spring_component_blocks(trace: dict[str, np.ndarray], spring_indices: np.ndarray, step: int) -> dict[str, np.ndarray]:
    if spring_indices.size == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return {"current": empty, "contractile": empty, "combined": empty}
    init = np.asarray(trace["spring_init_rest_length"], dtype=np.float32)[spring_indices]
    current = _safe_scale(trace["spring_current_length"][int(step), spring_indices], init)
    rest = _safe_scale(trace["spring_rest_length"][int(step), spring_indices], init)
    pre_rest = _safe_scale(trace["spring_pre_rest_length"][int(step), spring_indices], init)
    phase_rest = _safe_scale(trace["phase_post_spring_rest_length"][int(step), :, spring_indices], init)
    current_block = current.reshape(-1).astype(np.float32)
    contractile_block = np.concatenate([rest.reshape(-1), pre_rest.reshape(-1), phase_rest.reshape(-1)]).astype(np.float32)
    return {
        "current": current_block,
        "contractile": contractile_block,
        "combined": np.concatenate([current_block, contractile_block]).astype(np.float32),
    }


def _body_kinematic_rank_feature(
    event: dict[str, Any],
    *,
    rank: int,
    step: int,
) -> np.ndarray | None:
    cell = event["rank_to_cell"].get(int(rank))
    if cell is None:
        return None
    trace = event["trace"]
    springs = event["spring_indices"][int(cell)]
    points = np.asarray(springs["points"], dtype=np.int32)
    corners = np.asarray(trace["cell_corner_positions"][int(step), [int(cell)]], dtype=np.float32)
    center = corners.mean(axis=(1, 2), keepdims=True)
    rel_corners = (corners - center).reshape(-1)
    velocities = np.asarray(trace["plant_velocities_true"][int(step), points], dtype=np.float32)
    velocities = (velocities - velocities.mean(axis=0, keepdims=True)).reshape(-1)
    return np.concatenate([rel_corners, velocities]).astype(np.float32)


def _spring_rank_feature(
    event: dict[str, Any],
    *,
    rank: int,
    step: int,
    component: str,
) -> np.ndarray | None:
    cell = event["rank_to_cell"].get(int(rank))
    if cell is None:
        return None
    trace = event["trace"]
    springs = event["spring_indices"][int(cell)]
    blocks = _spring_component_blocks(trace, np.asarray(springs["attached_springs"], dtype=np.int32), int(step))
    return blocks[str(component)]


def _stream_feature(event: dict[str, Any], *, stream: str, rank: int, step: int) -> np.ndarray | None:
    if step < 0 or step >= event["step_count"]:
        return None
    stream = LEGACY_STREAM_ALIASES.get(stream, stream)
    if stream == "signal_hidden":
        return _signal_hidden_rank_feature(event, rank=rank, step=step)
    if stream == "controller_action":
        return _controller_action_rank_feature(event, rank=rank, step=step)
    if stream == "body_kinematic":
        return _body_kinematic_rank_feature(event, rank=rank, step=step)
    if stream == "spring_current":
        return _spring_rank_feature(event, rank=rank, step=step, component="current")
    if stream == "spring_contractile":
        return _spring_rank_feature(event, rank=rank, step=step, component="contractile")
    if stream == "body_no_contractile":
        kinematic = _body_kinematic_rank_feature(event, rank=rank, step=step)
        current = _spring_rank_feature(event, rank=rank, step=step, component="current")
        if kinematic is None or current is None:
            return None
        return np.concatenate([current, kinematic]).astype(np.float32)
    if stream == "body_combined":
        kinematic = _body_kinematic_rank_feature(event, rank=rank, step=step)
        spring = _spring_rank_feature(event, rank=rank, step=step, component="combined")
        if kinematic is None or spring is None:
            return None
        return np.concatenate([spring, kinematic]).astype(np.float32)
    raise ValueError(f"Unknown stream: {stream}")


def _cell_stream_feature(event: dict[str, Any], *, stream: str, cell: int, step: int) -> np.ndarray | None:
    if step < 0 or step >= event["step_count"]:
        return None
    stream = LEGACY_STREAM_ALIASES.get(stream, stream)
    rows, cols = event["robot_shape"]
    row = int(cell) // int(cols)
    col = int(cell) % int(cols)
    trace = event["trace"]
    if stream == "signal_hidden":
        hidden_start = int(event["hidden_start_channel"])
        nca = np.asarray(trace["phase_nca_state"][int(step), :, row, col, hidden_start:], dtype=np.float32)
        carry = np.asarray(trace["phase_ctrnn_carry"][int(step), :, row, col, :], dtype=np.float32)
        return np.concatenate([nca.reshape(-1), carry.reshape(-1)]).astype(np.float32)
    if stream == "controller_action":
        parts: list[np.ndarray] = []
        if "action_outputs" in trace:
            parts.append(np.asarray(trace["action_outputs"][int(step), row, col, :], dtype=np.float32).reshape(-1))
        if "phase_action_outputs" in trace:
            parts.append(np.asarray(trace["phase_action_outputs"][int(step), :, row, col, :], dtype=np.float32).reshape(-1))
        if not parts:
            return None
        return np.concatenate(parts).astype(np.float32)
    if stream == "body_kinematic":
        springs = event["spring_indices"][int(cell)]
        points = np.asarray(springs["points"], dtype=np.int32)
        corners = np.asarray(trace["cell_corner_positions"][int(step), [int(cell)]], dtype=np.float32)
        center = corners.mean(axis=(1, 2), keepdims=True)
        rel_corners = (corners - center).reshape(-1)
        velocities = np.asarray(trace["plant_velocities_true"][int(step), points], dtype=np.float32)
        velocities = (velocities - velocities.mean(axis=0, keepdims=True)).reshape(-1)
        return np.concatenate([rel_corners, velocities]).astype(np.float32)
    if stream in {"spring_current", "spring_contractile", "body_no_contractile", "body_combined"}:
        springs = event["spring_indices"][int(cell)]
        blocks = _spring_component_blocks(trace, np.asarray(springs["attached_springs"], dtype=np.int32), int(step))
        if stream == "spring_current":
            return blocks["current"]
        if stream == "spring_contractile":
            return blocks["contractile"]
        kinematic = _cell_stream_feature(event, stream="body_kinematic", cell=cell, step=step)
        if kinematic is None:
            return None
        if stream == "body_no_contractile":
            return np.concatenate([blocks["current"], kinematic]).astype(np.float32)
        return np.concatenate([blocks["combined"], kinematic]).astype(np.float32)
    raise ValueError(f"Unknown stream: {stream}")


def _load_events() -> tuple[list[dict[str, Any]], tuple[int, int], int]:
    roots = _find_assay_roots()
    if not roots:
        raise FileNotFoundError(f"No {ASSAY} roots found under {_data_root()}")
    events: list[dict[str, Any]] = []
    robot_shape: tuple[int, int] | None = None
    hidden_start: int | None = None
    for assay_root in roots:
        manifest = json.loads((assay_root / "manifest.json").read_text())
        root_hidden_start = int((manifest.get("nca_channel_schema") or {}).get("hidden_start_channel", 6))
        trial_df = pd.read_csv(assay_root / "trial_manifest.csv")
        with np.load(assay_root / "static_arrays.npz") as static:
            root_robot_shape = tuple(int(value) for value in static["robot_morph"].shape)
        if robot_shape is None:
            robot_shape = root_robot_shape
        elif robot_shape != root_robot_shape:
            raise ValueError(f"Mismatched robot shape in {assay_root}: {root_robot_shape} != {robot_shape}")
        if hidden_start is None:
            hidden_start = root_hidden_start
        elif hidden_start != root_hidden_start:
            raise ValueError(f"Mismatched hidden start in {assay_root}: {root_hidden_start} != {hidden_start}")

        for _, trial_row in trial_df.sort_values("cliff_distance", kind="stable").iterrows():
            if pd.isna(trial_row.get("first_cliff_reach_step")):
                continue
            trial_id = str(trial_row["trial_id"])
            trace_path = _trace_path_for_trial(assay_root, trial_row["trace_npz"], trial_id)
            trace = _load_npz(trace_path)
            step_count = min(
                int(trial_row.get("steps_alive", trace["cell_corner_positions"].shape[0])),
                int(trace["cell_corner_positions"].shape[0]),
            )
            trace = {key: _slice_time(value, step_count) for key, value in trace.items()}
            release_step = int(trial_row["first_cliff_reach_step"])
            if release_step < 0 or release_step >= step_count:
                continue
            cliff_side = str(trial_row.get("cliff_side", "left"))
            rank_to_cell = _cell_rank_map(trace, release_step, cliff_side)
            cell_count = int(trace["boxel_corners"].shape[0])
            events.append(
                {
                    "event_id": f"cd{int(trial_row['cliff_distance']):02d}_{trial_id}",
                    "trial_id": trial_id,
                    "trace": trace,
                    "trace_path": str(trace_path),
                    "assay_root": str(assay_root),
                    "goal_distance": int(trial_row["goal_distance"]),
                    "cliff_distance": int(trial_row["cliff_distance"]),
                    "release_step": release_step,
                    "step_count": step_count,
                    "steps_alive": step_count,
                    "success": _bool_value(trial_row.get("success")),
                    "returned_from_edge": _bool_value(trial_row.get("returned_from_edge")),
                    "cliff_fell": _bool_value(trial_row.get("cliff_fell")),
                    "goal_touched": _bool_value(trial_row.get("goal_touched")),
                    "termination_reason": str(trial_row.get("termination_reason", "")),
                    "cliff_side": cliff_side,
                    "rank_to_cell": rank_to_cell,
                    "cell_count": cell_count,
                    "robot_shape": robot_shape,
                    "hidden_start_channel": hidden_start,
                    "spring_indices": {cell: _cell_spring_indices(trace, int(cell)) for cell in range(cell_count)},
                }
            )
    if robot_shape is None or hidden_start is None:
        raise ValueError("No valid state-space roots were loaded.")
    events.sort(key=lambda event: int(event["cliff_distance"]))
    return events, robot_shape, hidden_start


def _compare_prefix(a: np.ndarray, b: np.ndarray, horizon: int) -> tuple[float, float, int | None]:
    aa = np.asarray(a[:horizon], dtype=np.float32)
    bb = np.asarray(b[:horizon], dtype=np.float32)
    diff = np.abs(aa - bb)
    if diff.size == 0:
        return 0.0, 0.0, None
    max_abs = float(np.max(diff))
    rms = float(np.sqrt(np.mean(np.square(diff))))
    first = None
    if max_abs > EXACT_TOLERANCE:
        flat = diff.reshape((diff.shape[0], -1))
        hit = np.flatnonzero(np.max(flat, axis=1) > EXACT_TOLERANCE)
        first = int(hit[0]) if hit.size else None
    return max_abs, rms, first


def pre_release_identity(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    arrays = {
        "signal_nca_hidden": lambda event: event["trace"]["phase_nca_state"][..., int(event["hidden_start_channel"]) :],
        "signal_ctrnn": lambda event: event["trace"]["phase_ctrnn_carry"],
        "controller_action": lambda event: event["trace"]["phase_action_outputs"],
        "body_positions": lambda event: event["trace"]["plant_positions"],
        "body_velocities": lambda event: event["trace"]["plant_velocities_true"],
        "body_corners": lambda event: event["trace"]["cell_corner_positions"],
        "spring_current": lambda event: event["trace"]["spring_current_length"],
        "spring_rest": lambda event: event["trace"]["spring_rest_length"],
        "spring_pre_rest": lambda event: event["trace"]["spring_pre_rest_length"],
        "spring_phase_rest": lambda event: event["trace"]["phase_post_spring_rest_length"],
        "drive_h": lambda event: event["trace"]["phase_h_drive"],
        "drive_v": lambda event: event["trace"]["phase_v_drive"],
    }
    pair_rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(events):
        for right in events[left_index + 1 :]:
            horizon = min(int(left["release_step"]), int(right["release_step"]))
            for name, getter in arrays.items():
                max_abs, rms, first = _compare_prefix(getter(left), getter(right), horizon)
                pair_rows.append(
                    {
                        "stream": name,
                        "event_a": left["event_id"],
                        "event_b": right["event_id"],
                        "cliff_distance_a": left["cliff_distance"],
                        "cliff_distance_b": right["cliff_distance"],
                        "horizon_steps": horizon,
                        "max_abs": max_abs,
                        "rms": rms,
                        "first_diff_step": first,
                        "exact": bool(max_abs <= EXACT_TOLERANCE),
                    }
                )
    summary_rows: list[dict[str, Any]] = []
    frame = pd.DataFrame(pair_rows)
    for stream, group in frame.groupby("stream", sort=False):
        max_abs_values = group["max_abs"].to_numpy(dtype=np.float64)
        rms_values = group["rms"].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "stream": str(stream),
                "pair_count": int(len(group)),
                "exact_pair_count": int(group["exact"].sum()),
                "exact_fraction": float(group["exact"].mean()) if len(group) else float("nan"),
                "max_abs_max": float(np.max(max_abs_values)) if max_abs_values.size else float("nan"),
                "max_abs_median": float(np.median(max_abs_values)) if max_abs_values.size else float("nan"),
                "rms_max": float(np.max(rms_values)) if rms_values.size else float("nan"),
                "rms_median": float(np.median(rms_values)) if rms_values.size else float("nan"),
                "first_diff_min": None if group["first_diff_step"].dropna().empty else int(group["first_diff_step"].dropna().min()),
            }
        )
    return pair_rows, summary_rows


def _scores_leave_one_source_out(samples: list[dict[str, Any]]) -> np.ndarray | None:
    if len(samples) < 6:
        return None
    y = np.asarray([sample["label"] for sample in samples], dtype=np.int8)
    if len(np.unique(y)) < 2 or min(np.sum(y == -1), np.sum(y == 1)) < 2:
        return None
    sources = np.asarray([sample["source_event_id"] for sample in samples], dtype=object)
    unique_sources = sorted(set(sources.tolist()))
    scores = np.full((len(samples),), np.nan, dtype=np.float32)
    for source in unique_sources:
        test_mask = sources == source
        train_mask = ~test_mask
        train_y = y[train_mask]
        if len(np.unique(train_y)) < 2:
            continue
        train_x = np.vstack([samples[index]["feature"] for index in np.flatnonzero(train_mask)]).astype(np.float32)
        test_x = np.vstack([samples[index]["feature"] for index in np.flatnonzero(test_mask)]).astype(np.float32)
        scores[test_mask] = _ridge_scores(train_x, train_y, test_x, alpha=RIDGE_ALPHA)
    if not np.all(np.isfinite(scores)):
        return None
    return scores


def _ridge_scores_no_intercept(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, *, alpha: float) -> np.ndarray:
    train_x = np.vstack(train_x).astype(np.float32)
    test_x = np.vstack(test_x).astype(np.float32)
    mean = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd = np.where(sd > 1.0e-8, sd, 1.0).astype(np.float32)
    train_z = (train_x - mean) / sd
    test_z = (test_x - mean) / sd
    penalty = np.eye(train_z.shape[1], dtype=np.float32) * float(alpha)
    weights = np.linalg.solve(
        train_z.T @ train_z + penalty,
        train_z.T @ train_y.astype(np.float32),
    )
    return (test_z @ weights).astype(np.float32)


def _scores_leave_one_source_out_no_intercept(samples: list[dict[str, Any]]) -> np.ndarray | None:
    if len(samples) < 6:
        return None
    y = np.asarray([sample["label"] for sample in samples], dtype=np.int8)
    if len(np.unique(y)) < 2 or min(np.sum(y == -1), np.sum(y == 1)) < 2:
        return None
    sources = np.asarray([sample["source_event_id"] for sample in samples], dtype=object)
    unique_sources = sorted(set(sources.tolist()))
    scores = np.full((len(samples),), np.nan, dtype=np.float32)
    for source in unique_sources:
        test_mask = sources == source
        train_mask = ~test_mask
        train_y = y[train_mask]
        if len(np.unique(train_y)) < 2:
            continue
        train_x = [samples[index]["feature"] for index in np.flatnonzero(train_mask)]
        test_x = [samples[index]["feature"] for index in np.flatnonzero(test_mask)]
        scores[test_mask] = _ridge_scores_no_intercept(train_x, train_y, test_x, alpha=RIDGE_ALPHA)
    if not np.all(np.isfinite(scores)):
        return None
    return scores


def _decoder_rows(events: list[dict[str, Any]], *, random_seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(random_seed))
    rows: list[dict[str, Any]] = []
    rank_count = min(int(event["cell_count"]) for event in events)
    for stream in PREDICTIVE_STREAMS:
        for rank in range(rank_count):
            for rel_time in REL_TIMES:
                samples: list[dict[str, Any]] = []
                for event in events:
                    sample_step = int(event["release_step"]) + int(rel_time)
                    positive = _stream_feature(event, stream=stream, rank=rank, step=sample_step)
                    if positive is None or not np.all(np.isfinite(positive)):
                        continue
                    samples.append(
                        {
                            "source_event_id": event["event_id"],
                            "feature": positive,
                            "label": 1,
                        }
                    )
                    for continuer in events:
                        if continuer["event_id"] == event["event_id"]:
                            continue
                        if int(continuer["release_step"]) <= sample_step + CONTINUER_MARGIN_STEPS:
                            continue
                        negative = _stream_feature(continuer, stream=stream, rank=rank, step=sample_step)
                        if negative is None or not np.all(np.isfinite(negative)):
                            continue
                        samples.append(
                            {
                                "source_event_id": continuer["event_id"],
                                "feature": negative,
                                "label": -1,
                            }
                        )
                y = np.asarray([sample["label"] for sample in samples], dtype=np.int8)
                scores = _scores_leave_one_source_out(samples)
                auc = float("nan") if scores is None else _auc(y, scores)
                bal = float("nan") if scores is None else _balanced_accuracy(y, scores)
                if scores is None or not np.isfinite(auc):
                    null_q95 = float("nan")
                    p_value = float("nan")
                else:
                    null = np.asarray([_auc(rng.permutation(y), scores) for _ in range(PERMUTATIONS)], dtype=np.float32)
                    null_q95 = float(np.quantile(null, 0.95))
                    p_value = float((np.sum(null >= auc) + 1.0) / (len(null) + 1.0))
                rows.append(
                    {
                        "stream": stream,
                        "rank_from_edge": int(rank),
                        "rel_time": int(rel_time),
                        "sample_count": int(len(samples)),
                        "positive_count": int(np.sum(y == 1)) if y.size else 0,
                        "negative_count": int(np.sum(y == -1)) if y.size else 0,
                        "source_event_count": int(len({sample["source_event_id"] for sample in samples})),
                        "auc": auc,
                        "balanced_accuracy": bal,
                        "null_auc_q95": null_q95,
                        "one_sided_p": p_value,
                        "estimable": bool(scores is not None and np.isfinite(auc)),
                    }
                )
    return rows


def _matched_perturbation_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rank_count = min(int(event["cell_count"]) for event in events)
    for stream in PREDICTIVE_STREAMS:
        for rank in range(rank_count):
            for rel_time in REL_TIMES:
                deltas: list[float] = []
                for event in events:
                    sample_step = int(event["release_step"]) + int(rel_time)
                    positive = _stream_feature(event, stream=stream, rank=rank, step=sample_step)
                    if positive is None or not np.all(np.isfinite(positive)):
                        continue
                    negatives = []
                    for continuer in events:
                        if int(continuer["release_step"]) <= sample_step + CONTINUER_MARGIN_STEPS:
                            continue
                        negative = _stream_feature(continuer, stream=stream, rank=rank, step=sample_step)
                        if negative is not None and np.all(np.isfinite(negative)):
                            negatives.append(negative)
                    if not negatives:
                        continue
                    baseline = np.mean(np.vstack(negatives).astype(np.float32), axis=0)
                    deltas.append(float(np.linalg.norm(positive - baseline) / np.sqrt(max(1, positive.size))))
                arr = np.asarray(deltas, dtype=np.float32)
                rows.append(
                    {
                        "stream": stream,
                        "rank_from_edge": int(rank),
                        "rel_time": int(rel_time),
                        "n": int(arr.size),
                        "mean_l2_delta": float(np.mean(arr)) if arr.size else float("nan"),
                        "median_l2_delta": float(np.median(arr)) if arr.size else float("nan"),
                        "sd_l2_delta": float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan"),
                    }
                )
    return rows


def _onset_rows(decoder_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame = pd.DataFrame(decoder_rows)
    onset_rows: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []
    for stream, group in frame.groupby("stream", sort=False):
        stream_onsets: list[dict[str, Any]] = []
        for rank, rank_group in group.groupby("rank_from_edge", sort=True):
            eligible = rank_group[
                (rank_group["estimable"])
                & (rank_group["rel_time"] >= 0)
                & (rank_group["auc"] > rank_group["null_auc_q95"])
                & (rank_group["one_sided_p"] <= 0.05)
            ].sort_values("rel_time", kind="stable")
            onset = None if eligible.empty else int(eligible.iloc[0]["rel_time"])
            row = {
                "stream": str(stream),
                "rank_from_edge": int(rank),
                "onset_rel_time": onset,
                "onset_found": onset is not None,
            }
            onset_rows.append(row)
            stream_onsets.append(row)
        valid = [row for row in stream_onsets if row["onset_found"]]
        if len(valid) >= 3:
            x = np.asarray([row["rank_from_edge"] for row in valid], dtype=np.float64)
            y = np.asarray([row["onset_rel_time"] for row in valid], dtype=np.float64)
            slope, intercept = np.polyfit(x, y, deg=1)
            pred = slope * x + intercept
            denom = float(np.sum((y - y.mean()) ** 2))
            r2 = float(1.0 - np.sum((y - pred) ** 2) / denom) if denom > 0 else float("nan")
        else:
            slope = intercept = r2 = float("nan")
        regression_rows.append(
            {
                "stream": str(stream),
                "rank_count_with_onset": int(len(valid)),
                "slope_steps_per_rank": float(slope),
                "intercept": float(intercept),
                "r2": float(r2),
            }
        )
    return onset_rows, regression_rows


def _plot_heatmap(
    frame: pd.DataFrame,
    *,
    value_col: str,
    title: str,
    path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranks = sorted(int(value) for value in frame["rank_from_edge"].dropna().unique())
    rel_times = sorted(int(value) for value in frame["rel_time"].dropna().unique())
    matrix = np.full((len(ranks), len(rel_times)), np.nan, dtype=np.float32)
    rank_to_row = {rank: idx for idx, rank in enumerate(ranks)}
    time_to_col = {rel_time: idx for idx, rel_time in enumerate(rel_times)}
    for _, row in frame.iterrows():
        matrix[rank_to_row[int(row["rank_from_edge"])], time_to_col[int(row["rel_time"])]] = float(row[value_col])
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    image = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    if 0 in time_to_col:
        ax.axvline(time_to_col[0], color="#D4736C", linewidth=1.1)
    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_xlabel("Steps from release")
    ax.set_ylabel("Rank from edge")
    tick_cols = [idx for idx, rel_time in enumerate(rel_times) if rel_time % 64 == 0]
    ax.set_xticks(tick_cols)
    ax.set_xticklabels([str(rel_times[idx]) for idx in tick_cols], fontsize=8)
    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels([str(rank) for rank in ranks], fontsize=7)
    fig.colorbar(image, ax=ax, shrink=0.82)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_release_times(events: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [event["cliff_distance"] for event in events]
    y = [event["release_step"] for event in events]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.plot(x, y, marker="o", color="#3B5B92", linewidth=1.5, markersize=4)
    ax.set_xlabel("Cliff distance")
    ax.set_ylabel("Release step")
    ax.set_title("S4 release timing", fontsize=10, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_identity_summary(summary_rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(summary_rows)
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ax.bar(np.arange(len(frame)), frame["exact_fraction"].to_numpy(dtype=float), color="#8CB87C", width=0.7)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Exact pair fraction")
    ax.set_xticks(np.arange(len(frame)))
    ax.set_xticklabels(frame["stream"].astype(str), rotation=35, ha="right", fontsize=8)
    ax.set_title("Pre-release trace identity", fontsize=10, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _clean_derived_outputs() -> None:
    for pattern in (
        "matched_release_auc_*.png",
        "matched_release_auc_*.pdf",
        "matched_release_perturbation_*.png",
        "matched_release_perturbation_*.pdf",
        "matched_release_perturbation_*.gif",
        "phase_reset_*.png",
        "phase_reset_*.pdf",
    ):
        for path in (PACKAGE_ROOT / "figures").glob(pattern):
            path.unlink()
    for name in (
        "analysis_summary.json",
        "matched_release_decoder_map.csv",
        "matched_release_decoder_onsets.csv",
        "matched_release_onset_regression.csv",
        "matched_release_perturbation_map.csv",
        "phase_reset_events.csv",
        "phase_reset_reference_summary.csv",
        "phase_reset_release_phase_correlations.csv",
        "phase_reset_slip_by_event.csv",
        "phase_reset_slip_summary.csv",
    ):
        path = PACKAGE_ROOT / "tables" / name
        if path.exists():
            path.unlink()


def _plot_perturbation_gif(
    perturbation_rows: list[dict[str, Any]],
    path: Path,
    *,
    streams: tuple[str, ...] = (
        "body_no_contractile",
        "spring_contractile",
        "controller_action",
        "signal_hidden",
    ),
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    frame = pd.DataFrame(perturbation_rows)
    frame = frame[frame["stream"].isin(streams)].copy()
    rel_times = sorted(int(value) for value in frame["rel_time"].dropna().unique())
    ranks = sorted(int(value) for value in frame["rank_from_edge"].dropna().unique())
    if not rel_times or not ranks:
        return

    stream_vmax: dict[str, float] = {}
    for stream in streams:
        values = frame.loc[frame["stream"] == stream, "mean_l2_delta"].to_numpy(dtype=np.float32)
        values = values[np.isfinite(values)]
        stream_vmax[stream] = float(np.quantile(values, 0.98)) if values.size else 1.0
        if stream_vmax[stream] <= 0 or not np.isfinite(stream_vmax[stream]):
            stream_vmax[stream] = 1.0

    value_lookup = {
        (str(row["stream"]), int(row["rank_from_edge"]), int(row["rel_time"])): float(row["mean_l2_delta"])
        for _, row in frame.iterrows()
    }
    labels = {
        "body_no_contractile": "body: strain + kinematics",
        "spring_contractile": "contractile rest lengths",
        "controller_action": "controller action channels",
        "signal_hidden": "hidden/internal signalling",
    }
    colors = {
        "body_no_contractile": "#3B5B92",
        "spring_contractile": "#8CB87C",
        "controller_action": "#B58B4A",
        "signal_hidden": "#5A4E8C",
    }

    fig, axes = plt.subplots(len(streams), 1, figsize=(6.2, 1.25 * len(streams)), sharex=True)
    if len(streams) == 1:
        axes = [axes]

    def _draw(frame_index: int):
        rel_time = rel_times[frame_index]
        for ax, stream in zip(axes, streams, strict=True):
            ax.clear()
            values = np.asarray(
                [value_lookup.get((stream, rank, rel_time), np.nan) for rank in ranks],
                dtype=np.float32,
            )
            ax.bar(ranks, values, color=colors.get(stream, "#777777"), width=0.75)
            ax.set_ylim(0.0, stream_vmax[stream] * 1.08)
            ax.set_ylabel(labels.get(stream, stream), fontsize=8)
            ax.tick_params(axis="y", labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)
        axes[0].set_title(f"Mean released-vs-continuing deviation | t = {rel_time:+d} steps", fontsize=10, weight="bold")
        axes[-1].set_xlabel("Rank from edge (0 = frontier-nearest)", fontsize=8)
        axes[-1].set_xticks(ranks)
        axes[-1].tick_params(axis="x", labelsize=7)

    animation = FuncAnimation(fig, _draw, frames=len(rel_times), interval=120, repeat=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=8), dpi=140)
    plt.close(fig)


def _wrap_angle(value: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def _circular_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.arctan2(np.mean(np.sin(values)), np.mean(np.cos(values))))


def _whole_stream_feature(event: dict[str, Any], *, stream: str, step: int) -> np.ndarray | None:
    if step < 0 or step >= event["step_count"]:
        return None
    parts: list[np.ndarray] = []
    for cell in range(int(event["cell_count"])):
        feature = _cell_stream_feature(event, stream=stream, cell=cell, step=step)
        if feature is None or not np.all(np.isfinite(feature)):
            return None
        parts.append(np.asarray(feature, dtype=np.float32).reshape(-1))
    return np.concatenate(parts).astype(np.float32)


def _fit_phase_model(
    reference_event: dict[str, Any],
    *,
    stream: str,
) -> dict[str, Any] | None:
    max_step = int(reference_event["release_step"])
    steps = list(range(0, max_step, PHASE_FIT_STRIDE))
    features: list[np.ndarray] = []
    used_steps: list[int] = []
    for step in steps:
        feature = _whole_stream_feature(reference_event, stream=stream, step=step)
        if feature is None:
            continue
        features.append(feature)
        used_steps.append(int(step))
    if len(features) < 16:
        return None
    x = np.vstack(features).astype(np.float64)
    mean = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd > 1.0e-8, sd, 1.0)
    z = (x - mean) / sd
    try:
        _, singular_values, vt = np.linalg.svd(z, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if vt.shape[0] < 2:
        return None
    components = vt[:2].T
    scores = z @ components
    phase = np.unwrap(np.arctan2(scores[:, 1], scores[:, 0]))
    radius = np.sqrt(np.sum(np.square(scores[:, :2]), axis=1))
    total_var = float(np.sum(np.square(singular_values)))
    var1 = float(singular_values[0] ** 2 / total_var) if total_var > 0 else float("nan")
    var2 = float(singular_values[1] ** 2 / total_var) if total_var > 0 else float("nan")
    return {
        "stream": stream,
        "reference_event_id": reference_event["event_id"],
        "reference_cliff_distance": int(reference_event["cliff_distance"]),
        "fit_steps": np.asarray(used_steps, dtype=np.int32),
        "mean": mean,
        "sd": sd,
        "components": components,
        "scores": scores,
        "phase": phase,
        "radius": radius,
        "explained_var_pc1": var1,
        "explained_var_pc2": var2,
    }


def _project_phase(model: dict[str, Any], feature: np.ndarray) -> tuple[float, float, np.ndarray]:
    z = (np.asarray(feature, dtype=np.float64) - model["mean"]) / model["sd"]
    score = z @ model["components"]
    phase = float(np.arctan2(score[1], score[0]))
    radius = float(np.sqrt(np.sum(np.square(score[:2]))))
    return phase, radius, score.astype(np.float32)


def _fit_local_phase_model(
    reference_event: dict[str, Any],
    *,
    stream: str,
    cell: int,
) -> dict[str, Any] | None:
    max_step = int(reference_event["release_step"])
    steps = list(range(0, max_step, PHASE_FIT_STRIDE))
    features: list[np.ndarray] = []
    used_steps: list[int] = []
    for step in steps:
        feature = _cell_stream_feature(reference_event, stream=stream, cell=int(cell), step=step)
        if feature is None or not np.all(np.isfinite(feature)):
            continue
        features.append(feature)
        used_steps.append(int(step))
    if len(features) < 16:
        return None
    x = np.vstack(features).astype(np.float64)
    mean = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd > 1.0e-8, sd, 1.0)
    z = (x - mean) / sd
    try:
        _, singular_values, vt = np.linalg.svd(z, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if vt.shape[0] < 2:
        return None
    components = vt[:2].T
    scores = z @ components
    phase = np.unwrap(np.arctan2(scores[:, 1], scores[:, 0]))
    radius = np.sqrt(np.sum(np.square(scores[:, :2]), axis=1))
    total_var = float(np.sum(np.square(singular_values)))
    var1 = float(singular_values[0] ** 2 / total_var) if total_var > 0 else float("nan")
    var2 = float(singular_values[1] ** 2 / total_var) if total_var > 0 else float("nan")
    return {
        "stream": stream,
        "cell": int(cell),
        "reference_event_id": reference_event["event_id"],
        "reference_cliff_distance": int(reference_event["cliff_distance"]),
        "fit_steps": np.asarray(used_steps, dtype=np.int32),
        "feature_dim": int(x.shape[1]),
        "mean": mean,
        "sd": sd,
        "components": components,
        "scores": scores,
        "phase": phase,
        "radius": radius,
        "explained_var_pc1": var1,
        "explained_var_pc2": var2,
    }


def _com_x_series(event: dict[str, Any]) -> np.ndarray:
    trace = event["trace"]
    if "plant_positions" in trace:
        positions = np.asarray(trace["plant_positions"], dtype=np.float32)
        if positions.ndim >= 3 and positions.shape[-1] >= 1:
            axes = tuple(range(1, positions.ndim - 1))
            return np.nanmean(positions[..., 0], axis=axes).astype(np.float32)
    corners = np.asarray(trace["cell_corner_positions"], dtype=np.float32)
    if corners.ndim >= 4 and corners.shape[-1] >= 1:
        axes = tuple(range(1, corners.ndim - 1))
        return np.nanmean(corners[..., 0], axis=axes).astype(np.float32)
    raise ValueError(f"Trace for {event['event_id']} has no supported COM-position array")


def _edge_away_sign(event: dict[str, Any]) -> float:
    return -1.0 if str(event.get("cliff_side", "left")).strip().lower() == "right" else 1.0


def _edge_oriented_committed_labels(
    event: dict[str, Any],
    *,
    future_window: int = PROPAGATION_DIRECTION_FUTURE_WINDOW,
    min_displacement: float = PROPAGATION_DIRECTION_MIN_DISPLACEMENT,
    persistence_steps: int = PROPAGATION_DIRECTION_PERSISTENCE_STEPS,
    persistence_fraction: float = PROPAGATION_DIRECTION_PERSISTENCE_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    com_x = _com_x_series(event)
    oriented_com = com_x * float(_edge_away_sign(event))
    count = len(oriented_com)
    raw = np.zeros(count, dtype=np.int8)
    future_indices = np.arange(count, dtype=np.int64) + int(future_window)
    valid = future_indices < count
    deltas = np.zeros(count, dtype=np.float32)
    deltas[valid] = oriented_com[future_indices[valid]] - oriented_com[np.arange(count)[valid]]
    raw[np.logical_and(valid, deltas >= float(min_displacement))] = 1
    raw[np.logical_and(valid, deltas <= -float(min_displacement))] = -1
    labels = raw.copy()
    if int(persistence_steps) > 1:
        persisted = np.zeros_like(raw)
        for index, value in enumerate(raw):
            if value == 0:
                continue
            stop = min(count, index + int(persistence_steps))
            if stop <= index:
                continue
            same_fraction = float(np.mean(raw[index:stop] == value))
            if same_fraction >= float(persistence_fraction):
                persisted[index] = value
        labels = persisted
    labels[~valid] = 0
    return labels, deltas


def _phase_response_feature(row: dict[str, Any]) -> np.ndarray:
    values = np.asarray(
        [
            float(row["phase_slip"]),
            float(row["abs_phase_slip"]),
            float(row["radius_delta"]),
            float(row["pc_l2_delta"]),
        ],
        dtype=np.float32,
    )
    values[np.abs(values) < PROPAGATION_FEATURE_DUST_FLOOR] = 0.0
    return values


def _rank_local_phase_rows(
    events: list[dict[str, Any]],
    *,
    models: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    direction_by_event = {
        event["event_id"]: _edge_oriented_committed_labels(event)
        for event in events
    }
    rank_count = min(int(event["cell_count"]) for event in events)
    for stream in PROPAGATION_STREAMS:
        for rank in range(rank_count):
            for rel_time in PROPAGATION_REL_TIMES:
                for event in events:
                    cell = event["rank_to_cell"].get(int(rank))
                    if cell is None:
                        continue
                    model = models.get((stream, int(cell)))
                    if model is None:
                        continue
                    sample_step = int(event["release_step"]) + int(rel_time)
                    positive = _cell_stream_feature(event, stream=stream, cell=int(cell), step=sample_step)
                    if positive is None or not np.all(np.isfinite(positive)):
                        continue
                    pos_phase, pos_radius, pos_score = _project_phase(model, positive)
                    negative_phases: list[float] = []
                    negative_radii: list[float] = []
                    negative_scores: list[np.ndarray] = []
                    for continuer in events:
                        if continuer["event_id"] == event["event_id"]:
                            continue
                        if int(continuer["release_step"]) <= sample_step + CONTINUER_MARGIN_STEPS:
                            continue
                        negative = _cell_stream_feature(continuer, stream=stream, cell=int(cell), step=sample_step)
                        if negative is None or not np.all(np.isfinite(negative)):
                            continue
                        neg_phase, neg_radius, neg_score = _project_phase(model, negative)
                        negative_phases.append(neg_phase)
                        negative_radii.append(neg_radius)
                        negative_scores.append(neg_score)
                    if not negative_phases:
                        continue
                    labels, direction_deltas = direction_by_event[event["event_id"]]
                    direction_label = int(labels[sample_step]) if 0 <= sample_step < len(labels) else 0
                    direction_delta = float(direction_deltas[sample_step]) if 0 <= sample_step < len(direction_deltas) else float("nan")
                    baseline_phase = _circular_mean(np.asarray(negative_phases, dtype=np.float64))
                    baseline_radius = float(np.mean(np.asarray(negative_radii, dtype=np.float64)))
                    baseline_score = np.mean(np.vstack(negative_scores).astype(np.float32), axis=0)
                    phase_slip = float(_wrap_angle(pos_phase - baseline_phase))
                    radius_delta = float(pos_radius - baseline_radius)
                    pc_l2_delta = float(np.linalg.norm(pos_score - baseline_score))
                    rows.append(
                        {
                            "stream": stream,
                            "event_id": event["event_id"],
                            "trial_id": event["trial_id"],
                            "cliff_distance": int(event["cliff_distance"]),
                            "release_step": int(event["release_step"]),
                            "rank_from_edge": int(rank),
                            "cell": int(cell),
                            "rel_time": int(rel_time),
                            "sample_step": int(sample_step),
                            "success": bool(event["success"]),
                            "returned_from_edge": bool(event["returned_from_edge"]),
                            "steps_after_release": int(event["steps_alive"]) - int(event["release_step"]),
                            "direction_label": direction_label,
                            "direction_delta": direction_delta,
                            "positive_phase": pos_phase,
                            "baseline_phase": baseline_phase,
                            "phase_slip": phase_slip,
                            "abs_phase_slip": abs(phase_slip),
                            "positive_radius": pos_radius,
                            "baseline_radius": baseline_radius,
                            "radius_delta": radius_delta,
                            "pc_l2_delta": pc_l2_delta,
                            "continuing_count": int(len(negative_phases)),
                        }
                    )
    return rows


def _rank_local_phase_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    summary: list[dict[str, Any]] = []
    if frame.empty:
        return summary
    for (stream, rank, rel_time), group in frame.groupby(["stream", "rank_from_edge", "rel_time"], sort=True):
        slips = group["phase_slip"].to_numpy(dtype=np.float64)
        abs_slips = np.abs(slips)
        summary.append(
            {
                "stream": str(stream),
                "rank_from_edge": int(rank),
                "rel_time": int(rel_time),
                "n": int(len(group)),
                "event_count": int(group["event_id"].nunique()),
                "mean_phase_slip": _circular_mean(slips),
                "mean_abs_phase_slip": float(np.mean(abs_slips)),
                "median_abs_phase_slip": float(np.median(abs_slips)),
                "sd_abs_phase_slip": float(np.std(abs_slips, ddof=1)) if len(abs_slips) > 1 else float("nan"),
                "mean_radius_delta": float(group["radius_delta"].mean()),
                "median_radius_delta": float(group["radius_delta"].median()),
                "mean_pc_l2_delta": float(group["pc_l2_delta"].mean()),
                "median_pc_l2_delta": float(group["pc_l2_delta"].median()),
                "away_label_count": int(np.sum(group["direction_label"].to_numpy(dtype=np.int8) == 1)),
                "toward_label_count": int(np.sum(group["direction_label"].to_numpy(dtype=np.int8) == -1)),
                "uncommitted_label_count": int(np.sum(group["direction_label"].to_numpy(dtype=np.int8) == 0)),
            }
        )
    return summary


def _phase_propagation_direction_decoder_rows(
    local_rows: list[dict[str, Any]],
    *,
    random_seed: int,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(local_rows)
    if frame.empty:
        return []
    rng = np.random.default_rng(int(random_seed))
    rows: list[dict[str, Any]] = []
    for (stream, rank, rel_time), group in frame.groupby(["stream", "rank_from_edge", "rel_time"], sort=True):
        samples = [
            {
                "source_event_id": str(row["event_id"]),
                "feature": _phase_response_feature(row),
                "label": int(row["direction_label"]),
            }
            for row in group.to_dict("records")
            if int(row["direction_label"]) in {-1, 1}
            and np.all(np.isfinite(_phase_response_feature(row)))
        ]
        y = np.asarray([sample["label"] for sample in samples], dtype=np.int8)
        scores = _scores_leave_one_source_out_no_intercept(samples)
        auc = float("nan") if scores is None else _auc(y, scores)
        bal = float("nan") if scores is None else _balanced_accuracy(y, scores)
        if scores is None or not np.isfinite(auc):
            null_q95 = float("nan")
            p_value = float("nan")
        else:
            null = np.asarray([_auc(rng.permutation(y), scores) for _ in range(PROPAGATION_PERMUTATIONS)], dtype=np.float32)
            null_q95 = float(np.quantile(null, 0.95))
            p_value = float((np.sum(null >= auc) + 1.0) / (len(null) + 1.0))
        rows.append(
            {
                "stream": str(stream),
                "rank_from_edge": int(rank),
                "rel_time": int(rel_time),
                "sample_count": int(len(samples)),
                "positive_count": int(np.sum(y == 1)) if y.size else 0,
                "negative_count": int(np.sum(y == -1)) if y.size else 0,
                "source_event_count": int(len({sample["source_event_id"] for sample in samples})),
                "feature_kind": "phase_response",
                "classifier": "leave-one-event-out ridge without intercept",
                "direction_label": "positive=away_from_cliff; negative=toward_cliff",
                "future_window": int(PROPAGATION_DIRECTION_FUTURE_WINDOW),
                "min_displacement": float(PROPAGATION_DIRECTION_MIN_DISPLACEMENT),
                "persistence_steps": int(PROPAGATION_DIRECTION_PERSISTENCE_STEPS),
                "persistence_fraction": float(PROPAGATION_DIRECTION_PERSISTENCE_FRACTION),
                "auc": auc,
                "balanced_accuracy": bal,
                "null_auc_q95": null_q95,
                "one_sided_p": p_value,
                "estimable": bool(scores is not None and np.isfinite(auc)),
            }
        )
    return rows


def _rank_onset_thresholds(summary_rows: list[dict[str, Any]]) -> dict[tuple[str, int], float]:
    frame = pd.DataFrame(summary_rows)
    thresholds: dict[tuple[str, int], float] = {}
    if frame.empty:
        return thresholds
    for (stream, rank), group in frame.groupby(["stream", "rank_from_edge"], sort=True):
        pre = group[group["rel_time"] < 0]["mean_abs_phase_slip"].to_numpy(dtype=np.float64)
        pre = pre[np.isfinite(pre)]
        if pre.size:
            threshold = float(np.mean(pre) + PROPAGATION_PHASE_ONSET_SD_MULTIPLIER * np.std(pre, ddof=1)) if pre.size > 1 else float(pre[0])
            threshold = max(threshold, float(np.quantile(pre, 0.95)))
        else:
            threshold = float("nan")
        if not np.isfinite(threshold):
            threshold = PROPAGATION_PHASE_ONSET_MIN_RAD
        thresholds[(str(stream), int(rank))] = max(float(threshold), PROPAGATION_PHASE_ONSET_MIN_RAD)
    return thresholds


def _linear_slope_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    rng: np.random.Generator,
    permutations: int = 1000,
) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 3:
        return {
            "slope": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "intercept": float("nan"),
            "r2": float("nan"),
            "permutation_p_positive": float("nan"),
        }
    slope, intercept = np.polyfit(x, y, deg=1)
    pred = slope * x + intercept
    denom = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum((y - pred) ** 2) / denom) if denom > 0 else float("nan")
    boot: list[float] = []
    for _ in range(int(permutations)):
        idx = rng.integers(0, len(x), size=len(x))
        if len(np.unique(x[idx])) < 2:
            continue
        boot_slope, _ = np.polyfit(x[idx], y[idx], deg=1)
        boot.append(float(boot_slope))
    if boot:
        ci_low, ci_high = np.quantile(np.asarray(boot, dtype=np.float64), [0.025, 0.975])
    else:
        ci_low = ci_high = float("nan")
    null: list[float] = []
    for _ in range(int(permutations)):
        shuffled = rng.permutation(y)
        null_slope, _ = np.polyfit(x, shuffled, deg=1)
        null.append(float(null_slope))
    null_arr = np.asarray(null, dtype=np.float64)
    p_positive = float((np.sum(null_arr >= float(slope)) + 1.0) / (len(null_arr) + 1.0))
    return {
        "slope": float(slope),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "intercept": float(intercept),
        "r2": float(r2),
        "permutation_p_positive": p_positive,
    }


def _phase_propagation_onset_rows(
    local_summary_rows: list[dict[str, Any]],
    decoder_rows: list[dict[str, Any]],
    *,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    thresholds = _rank_onset_thresholds(local_summary_rows)
    summary_frame = pd.DataFrame(local_summary_rows)
    decoder_frame = pd.DataFrame(decoder_rows)
    onset_rows: list[dict[str, Any]] = []
    if not summary_frame.empty:
        for (stream, rank), group in summary_frame.groupby(["stream", "rank_from_edge"], sort=True):
            threshold = thresholds[(str(stream), int(rank))]
            eligible = group[
                (group["rel_time"] >= 0)
                & (group["mean_abs_phase_slip"] > threshold)
            ].sort_values("rel_time", kind="stable")
            onset = None if eligible.empty else int(eligible.iloc[0]["rel_time"])
            onset_rows.append(
                {
                    "stream": str(stream),
                    "rank_from_edge": int(rank),
                    "metric": "mean_abs_phase_slip",
                    "onset_rel_time": onset,
                    "onset_found": onset is not None,
                    "threshold": float(threshold),
                }
            )
    if not decoder_frame.empty:
        for (stream, rank), group in decoder_frame.groupby(["stream", "rank_from_edge"], sort=True):
            eligible = group[
                (group["estimable"])
                & (group["rel_time"] >= 0)
                & (group["auc"] > group["null_auc_q95"])
                & (group["one_sided_p"] <= 0.05)
            ].sort_values("rel_time", kind="stable")
            onset = None if eligible.empty else int(eligible.iloc[0]["rel_time"])
            onset_rows.append(
                {
                    "stream": str(stream),
                    "rank_from_edge": int(rank),
                    "metric": "direction_auc",
                    "onset_rel_time": onset,
                    "onset_found": onset is not None,
                    "threshold": float("nan"),
                }
            )
    rng = np.random.default_rng(_stable_seed(random_seed, "phase-propagation-onsets"))
    regression_rows: list[dict[str, Any]] = []
    onset_frame = pd.DataFrame(onset_rows)
    if onset_frame.empty:
        return onset_rows, regression_rows
    for (stream, metric), group in onset_frame[onset_frame["onset_found"]].groupby(["stream", "metric"], sort=True):
        x = group["rank_from_edge"].to_numpy(dtype=np.float64)
        y = group["onset_rel_time"].to_numpy(dtype=np.float64)
        fit = _linear_slope_with_ci(x, y, rng=rng, permutations=1000)
        regression_rows.append(
            {
                "stream": str(stream),
                "metric": str(metric),
                "rank_count_with_onset": int(len(group)),
                "slope_steps_per_rank": fit["slope"],
                "slope_ci_low": fit["ci_low"],
                "slope_ci_high": fit["ci_high"],
                "intercept": fit["intercept"],
                "r2": fit["r2"],
                "permutation_p_positive": fit["permutation_p_positive"],
            }
        )
    return onset_rows, regression_rows


def _plot_phase_propagation_heatmaps(
    rows: list[dict[str, Any]],
    *,
    value_col: str,
    path: Path,
    title_prefix: str,
    streams: tuple[str, ...] = PROPAGATION_FIGURE_STREAMS,
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(rows)
    frame = frame[frame["stream"].isin(streams)].copy()
    if frame.empty:
        return
    ranks = sorted(int(value) for value in frame["rank_from_edge"].dropna().unique())
    rel_times = sorted(int(value) for value in frame["rel_time"].dropna().unique())
    if not ranks or not rel_times:
        return
    stream_count = len(streams)
    fig, axes = plt.subplots(stream_count, 1, figsize=(6.4, 1.18 * stream_count), sharex=True)
    if stream_count == 1:
        axes = [axes]
    labels = {
        "signal_hidden": "hidden/internal",
        "controller_action": "action",
        "spring_contractile": "contractile",
        "spring_current": "spring length",
        "body_kinematic": "kinematics",
        "body_no_contractile": "body no contraction",
    }
    for ax, stream in zip(axes, streams, strict=True):
        stream_frame = frame[frame["stream"] == stream]
        matrix = np.full((len(ranks), len(rel_times)), np.nan, dtype=np.float32)
        rank_to_row = {rank: idx for idx, rank in enumerate(ranks)}
        time_to_col = {rel_time: idx for idx, rel_time in enumerate(rel_times)}
        for _, row in stream_frame.iterrows():
            matrix[rank_to_row[int(row["rank_from_edge"])], time_to_col[int(row["rel_time"])]] = float(row[value_col])
        stream_vmax = vmax
        if stream_vmax is None:
            finite = matrix[np.isfinite(matrix)]
            stream_vmax = float(np.quantile(finite, 0.98)) if finite.size else 1.0
            if vmin == 0.0 and stream_vmax < PROPAGATION_PHASE_ONSET_MIN_RAD:
                stream_vmax = PROPAGATION_PHASE_ONSET_MIN_RAD
        image = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=stream_vmax)
        if 0 in time_to_col:
            ax.axvline(time_to_col[0], color="#D4736C", linewidth=1.0)
        ax.set_ylabel(labels.get(stream, stream), fontsize=7)
        ax.set_yticks([0, len(ranks) - 1] if len(ranks) > 1 else [0])
        ax.set_yticklabels([str(ranks[0]), str(ranks[-1])] if len(ranks) > 1 else [str(ranks[0])], fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
        fig.colorbar(image, ax=ax, shrink=0.78, pad=0.012)
    axes[0].set_title(title_prefix, fontsize=10, weight="bold")
    axes[-1].set_xlabel("Steps from release", fontsize=8)
    tick_cols = [idx for idx, rel_time in enumerate(rel_times) if rel_time % 128 == 0]
    axes[-1].set_xticks(tick_cols)
    axes[-1].set_xticklabels([str(rel_times[idx]) for idx in tick_cols], fontsize=7)
    fig.text(0.008, 0.5, "Rank from edge", va="center", rotation="vertical", fontsize=8)
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_phase_propagation_onsets(onset_rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(onset_rows)
    frame = frame[frame["stream"].isin(PROPAGATION_FIGURE_STREAMS) & frame["onset_found"]].copy()
    if frame.empty:
        return
    metrics = ("mean_abs_phase_slip", "direction_auc")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    colors = {
        "signal_hidden": "#5A4E8C",
        "controller_action": "#B58B4A",
        "spring_contractile": "#8CB87C",
        "spring_current": "#3B5B92",
        "body_kinematic": "#6E9DAA",
        "body_no_contractile": "#4E6F8C",
    }
    labels = {
        "signal_hidden": "hidden/internal",
        "controller_action": "action",
        "spring_contractile": "contractile",
        "spring_current": "spring length",
        "body_kinematic": "kinematics",
        "body_no_contractile": "body no contraction",
    }
    for ax, metric in zip(axes, metrics, strict=True):
        metric_frame = frame[frame["metric"] == metric]
        for stream in PROPAGATION_FIGURE_STREAMS:
            stream_frame = metric_frame[metric_frame["stream"] == stream].sort_values("rank_from_edge")
            if stream_frame.empty:
                continue
            ax.plot(
                stream_frame["rank_from_edge"].to_numpy(dtype=float),
                stream_frame["onset_rel_time"].to_numpy(dtype=float),
                marker="o",
                markersize=3,
                linewidth=1.1,
                color=colors.get(stream, "#777777"),
                label=labels.get(stream, stream),
            )
        ax.axhline(0, color="#D4736C", linewidth=0.8)
        ax.set_xlabel("Rank from edge", fontsize=8)
        ax.set_title("Phase-slip onset" if metric == "mean_abs_phase_slip" else "Direction-AUC onset", fontsize=9, weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Onset rel. time", fontsize=8)
    axes[1].legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_phase_propagation_gif(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    streams: tuple[str, ...] = ("body_no_contractile", "spring_current", "spring_contractile", "signal_hidden"),
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    frame = pd.DataFrame(rows)
    frame = frame[frame["stream"].isin(streams)].copy()
    if frame.empty:
        return
    rel_times = sorted(int(value) for value in frame["rel_time"].dropna().unique())
    ranks = sorted(int(value) for value in frame["rank_from_edge"].dropna().unique())
    if not rel_times or not ranks:
        return
    value_lookup = {
        (str(row["stream"]), int(row["rank_from_edge"]), int(row["rel_time"])): float(row["mean_abs_phase_slip"])
        for _, row in frame.iterrows()
    }
    labels = {
        "body_no_contractile": "body no contraction",
        "spring_current": "spring length",
        "spring_contractile": "contractile",
        "signal_hidden": "hidden/internal",
    }
    colors = {
        "body_no_contractile": "#4E6F8C",
        "spring_current": "#3B5B92",
        "spring_contractile": "#8CB87C",
        "signal_hidden": "#5A4E8C",
    }
    stream_vmax: dict[str, float] = {}
    for stream in streams:
        values = frame.loc[frame["stream"] == stream, "mean_abs_phase_slip"].to_numpy(dtype=np.float32)
        values = values[np.isfinite(values)]
        stream_vmax[stream] = float(np.quantile(values, 0.98)) if values.size else 1.0
        if not np.isfinite(stream_vmax[stream]) or stream_vmax[stream] <= 0:
            stream_vmax[stream] = 1.0
    fig, axes = plt.subplots(len(streams), 1, figsize=(6.2, 1.15 * len(streams)), sharex=True)
    if len(streams) == 1:
        axes = [axes]

    def _draw(frame_index: int) -> None:
        rel_time = rel_times[frame_index]
        for ax, stream in zip(axes, streams, strict=True):
            ax.clear()
            values = np.asarray([value_lookup.get((stream, rank, rel_time), np.nan) for rank in ranks], dtype=np.float32)
            ax.bar(ranks, values, color=colors.get(stream, "#777777"), width=0.75)
            ax.set_ylim(0.0, stream_vmax[stream] * 1.08)
            ax.set_ylabel(labels.get(stream, stream), fontsize=8)
            ax.tick_params(axis="y", labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)
        axes[0].set_title(f"Local phase/deviation from continuing baseline | t = {rel_time:+d}", fontsize=9, weight="bold")
        axes[-1].set_xlabel("Rank from edge (0 = frontier-nearest)", fontsize=8)
        axes[-1].set_xticks(ranks)
        axes[-1].tick_params(axis="x", labelsize=7)

    animation = FuncAnimation(fig, _draw, frames=len(rel_times), interval=120, repeat=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=8), dpi=140)
    plt.close(fig)


GRAPH_PROPAGATION_STREAMS = (
    "signal_hidden",
    "controller_action",
    "body_kinematic",
    "body_no_contractile",
    "connection_spring_current",
    "internal_spring_current",
    "connection_spring_contractile",
    "internal_spring_contractile",
)

GRAPH_PROPAGATION_FIGURE_STREAMS = (
    "connection_spring_current",
    "connection_spring_contractile",
    "body_kinematic",
    "body_no_contractile",
    "signal_hidden",
    "controller_action",
)


def _frontier_cells(event: dict[str, Any]) -> list[int]:
    trace = event["trace"]
    centers = np.asarray(trace["cell_corner_positions"][int(event["release_step"])], dtype=np.float32).mean(axis=1)
    valid = np.flatnonzero(np.asarray(trace["boxel_mask"], dtype=bool))
    if valid.size == 0:
        return []
    reverse = str(event.get("cliff_side", "left")).strip().lower() == "right"
    xs = centers[valid, 0]
    edge_x = float(np.max(xs) if reverse else np.min(xs))
    # Both cells in the frontier column share the same x center up to float dust.
    tol = max(1.0e-4, 1.0e-4 * float(np.ptp(xs) if xs.size else 1.0))
    return [int(cell) for cell in valid.tolist() if abs(float(centers[int(cell), 0]) - edge_x) <= tol]


def _cell_adjacency(event: dict[str, Any]) -> dict[int, set[int]]:
    trace = event["trace"]
    cell_count = int(event["cell_count"])
    adjacency: dict[int, set[int]] = {cell: set() for cell in range(cell_count)}
    boxel_corners = np.asarray(trace["boxel_corners"], dtype=np.int32)
    point_to_cells: dict[int, set[int]] = {}
    for cell in range(cell_count):
        for point in boxel_corners[int(cell)].reshape(-1):
            point_to_cells.setdefault(int(point), set()).add(int(cell))
    a_idx = np.asarray(trace["spring_a_idx"], dtype=np.int32)
    b_idx = np.asarray(trace["spring_b_idx"], dtype=np.int32)
    for a, b in zip(a_idx.tolist(), b_idx.tolist(), strict=False):
        cells = point_to_cells.get(int(a), set()) | point_to_cells.get(int(b), set())
        if len(cells) < 2:
            continue
        for left in cells:
            for right in cells:
                if left != right:
                    adjacency[int(left)].add(int(right))
    return adjacency


def _cell_graph_distances(event: dict[str, Any]) -> dict[int, int]:
    frontier = _frontier_cells(event)
    adjacency = _cell_adjacency(event)
    distances: dict[int, int] = {}
    queue: list[int] = []
    for cell in frontier:
        distances[int(cell)] = 0
        queue.append(int(cell))
    while queue:
        current = queue.pop(0)
        for neighbor in sorted(adjacency.get(current, set())):
            if neighbor in distances:
                continue
            distances[int(neighbor)] = distances[current] + 1
            queue.append(int(neighbor))
    for cell in range(int(event["cell_count"])):
        distances.setdefault(int(cell), int(event["rank_to_cell"].get(cell, cell)))
    return distances


def _spring_subset_feature(
    event: dict[str, Any],
    *,
    cell: int,
    step: int,
    subset: str,
    component: str,
) -> np.ndarray | None:
    if step < 0 or step >= event["step_count"]:
        return None
    springs = event["spring_indices"][int(cell)]
    if subset == "connection":
        spring_indices = np.asarray(springs["connection_springs"], dtype=np.int32)
    elif subset == "internal":
        spring_indices = np.asarray(springs["internal_springs"], dtype=np.int32)
    else:
        raise ValueError(f"Unknown spring subset: {subset}")
    blocks = _spring_component_blocks(event["trace"], spring_indices, int(step))
    return blocks[str(component)]


def _graph_stream_feature(event: dict[str, Any], *, stream: str, cell: int, step: int) -> np.ndarray | None:
    if stream in {"signal_hidden", "controller_action", "body_kinematic", "body_no_contractile"}:
        return _cell_stream_feature(event, stream=stream, cell=int(cell), step=int(step))
    if stream == "connection_spring_current":
        return _spring_subset_feature(event, cell=int(cell), step=int(step), subset="connection", component="current")
    if stream == "internal_spring_current":
        return _spring_subset_feature(event, cell=int(cell), step=int(step), subset="internal", component="current")
    if stream == "connection_spring_contractile":
        return _spring_subset_feature(event, cell=int(cell), step=int(step), subset="connection", component="contractile")
    if stream == "internal_spring_contractile":
        return _spring_subset_feature(event, cell=int(cell), step=int(step), subset="internal", component="contractile")
    raise ValueError(f"Unknown graph propagation stream: {stream}")


def _feature_delta_stats(positive: np.ndarray, negatives: list[np.ndarray]) -> dict[str, float] | None:
    if positive.size == 0 or not negatives:
        return None
    try:
        baseline = np.mean(np.vstack([np.asarray(value, dtype=np.float32).reshape(1, -1) for value in negatives]), axis=0).reshape(-1)
    except ValueError:
        return None
    pos = np.asarray(positive, dtype=np.float32).reshape(-1)
    if pos.shape != baseline.shape:
        return None
    delta = pos - baseline
    return {
        "l2_delta": float(np.linalg.norm(delta)),
        "rms_delta": float(np.sqrt(np.mean(np.square(delta)))) if delta.size else 0.0,
        "max_abs_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "feature_dim": int(delta.size),
    }


def _graph_propagation_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    distances_by_event = {event["event_id"]: _cell_graph_distances(event) for event in events}
    for stream in GRAPH_PROPAGATION_STREAMS:
        for event in events:
            distances = distances_by_event[event["event_id"]]
            frontier = set(_frontier_cells(event))
            for cell in range(int(event["cell_count"])):
                distance = int(distances.get(int(cell), -1))
                if distance < 0:
                    continue
                for rel_time in GRAPH_PROPAGATION_REL_TIMES:
                    sample_step = int(event["release_step"]) + int(rel_time)
                    positive = _graph_stream_feature(event, stream=stream, cell=int(cell), step=sample_step)
                    if positive is None or not np.all(np.isfinite(positive)):
                        continue
                    negatives: list[np.ndarray] = []
                    for continuer in events:
                        if continuer["event_id"] == event["event_id"]:
                            continue
                        if int(continuer["release_step"]) <= sample_step + CONTINUER_MARGIN_STEPS:
                            continue
                        negative = _graph_stream_feature(continuer, stream=stream, cell=int(cell), step=sample_step)
                        if negative is None or not np.all(np.isfinite(negative)):
                            continue
                        negatives.append(negative)
                    stats = _feature_delta_stats(positive, negatives)
                    if stats is None:
                        continue
                    rows.append(
                        {
                            "stream": stream,
                            "event_id": event["event_id"],
                            "trial_id": event["trial_id"],
                            "cliff_distance": int(event["cliff_distance"]),
                            "release_step": int(event["release_step"]),
                            "rel_time": int(rel_time),
                            "sample_step": int(sample_step),
                            "cell": int(cell),
                            "graph_distance": int(distance),
                            "frontier_cell": bool(int(cell) in frontier),
                            "success": bool(event["success"]),
                            "returned_from_edge": bool(event["returned_from_edge"]),
                            "continuing_count": int(len(negatives)),
                            **stats,
                        }
                    )
    return rows


def _graph_propagation_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    summary: list[dict[str, Any]] = []
    for (stream, distance, rel_time), group in frame.groupby(["stream", "graph_distance", "rel_time"], sort=True):
        values = group["l2_delta"].to_numpy(dtype=np.float64)
        summary.append(
            {
                "stream": str(stream),
                "graph_distance": int(distance),
                "rel_time": int(rel_time),
                "n": int(len(group)),
                "event_count": int(group["event_id"].nunique()),
                "cell_count": int(group["cell"].nunique()),
                "mean_l2_delta": float(np.mean(values)),
                "median_l2_delta": float(np.median(values)),
                "sd_l2_delta": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                "mean_rms_delta": float(group["rms_delta"].mean()),
                "mean_max_abs_delta": float(group["max_abs_delta"].mean()),
            }
        )
    return summary


def _graph_onset_rows(summary_rows: list[dict[str, Any]], *, random_seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame = pd.DataFrame(summary_rows)
    onset_rows: list[dict[str, Any]] = []
    if frame.empty:
        return onset_rows, []
    for (stream, distance), group in frame.groupby(["stream", "graph_distance"], sort=True):
        pre = group[group["rel_time"] < 0]["mean_l2_delta"].to_numpy(dtype=np.float64)
        pre = pre[np.isfinite(pre)]
        if pre.size > 1:
            threshold = max(
                float(np.mean(pre) + GRAPH_PROPAGATION_ONSET_SD_MULTIPLIER * np.std(pre, ddof=1)),
                float(np.quantile(pre, 0.95)),
                GRAPH_PROPAGATION_ONSET_MIN_EFFECT,
            )
        elif pre.size == 1:
            threshold = max(float(pre[0]), GRAPH_PROPAGATION_ONSET_MIN_EFFECT)
        else:
            threshold = GRAPH_PROPAGATION_ONSET_MIN_EFFECT
        eligible = group[
            (group["rel_time"] >= 0)
            & (group["mean_l2_delta"] > threshold)
        ].sort_values("rel_time", kind="stable")
        onset = None if eligible.empty else int(eligible.iloc[0]["rel_time"])
        onset_rows.append(
            {
                "stream": str(stream),
                "graph_distance": int(distance),
                "metric": "mean_l2_delta",
                "threshold": float(threshold),
                "onset_rel_time": onset,
                "onset_found": onset is not None,
            }
        )
    onset_frame = pd.DataFrame(onset_rows)
    regression_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(_stable_seed(random_seed, "graph-propagation-onsets"))
    if not onset_frame.empty:
        for stream, group in onset_frame[onset_frame["onset_found"]].groupby("stream", sort=True):
            fit = _linear_slope_with_ci(
                group["graph_distance"].to_numpy(dtype=np.float64),
                group["onset_rel_time"].to_numpy(dtype=np.float64),
                rng=rng,
                permutations=1000,
            )
            regression_rows.append(
                {
                    "stream": str(stream),
                    "metric": "mean_l2_delta",
                    "distance_count_with_onset": int(len(group)),
                    "slope_steps_per_graph_distance": fit["slope"],
                    "slope_ci_low": fit["ci_low"],
                    "slope_ci_high": fit["ci_high"],
                    "intercept": fit["intercept"],
                    "r2": fit["r2"],
                    "permutation_p_positive": fit["permutation_p_positive"],
                }
            )
    return onset_rows, regression_rows


def _plot_graph_heatmaps(
    rows: list[dict[str, Any]],
    *,
    path: Path,
    streams: tuple[str, ...] = GRAPH_PROPAGATION_FIGURE_STREAMS,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    rel_times = sorted(int(value) for value in frame["rel_time"].dropna().unique())
    distances = sorted(int(value) for value in frame["graph_distance"].dropna().unique())
    ncols = 2
    nrows = int(np.ceil(len(streams) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 2.15 * nrows), squeeze=False)
    for ax, stream in zip(axes.ravel(), streams, strict=False):
        subset = frame[frame["stream"] == stream]
        matrix = np.full((len(distances), len(rel_times)), np.nan, dtype=np.float32)
        dist_to_row = {distance: idx for idx, distance in enumerate(distances)}
        time_to_col = {rel_time: idx for idx, rel_time in enumerate(rel_times)}
        for _, row in subset.iterrows():
            matrix[dist_to_row[int(row["graph_distance"])], time_to_col[int(row["rel_time"])]] = float(row["mean_l2_delta"])
        finite = matrix[np.isfinite(matrix)]
        vmax = float(np.quantile(finite, 0.98)) if finite.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
        if 0 in time_to_col:
            ax.axvline(time_to_col[0], color="#D4736C", linewidth=1.0)
        ax.set_title(stream, fontsize=9, weight="bold")
        tick_cols = [idx for idx, rel_time in enumerate(rel_times) if rel_time % 128 == 0]
        ax.set_xticks(tick_cols)
        ax.set_xticklabels([str(rel_times[idx]) for idx in tick_cols], fontsize=7)
        ax.set_yticks(range(len(distances)))
        ax.set_yticklabels([str(distance) for distance in distances], fontsize=7)
        ax.set_xlabel("Steps from release", fontsize=8)
        ax.set_ylabel("Graph distance", fontsize=8)
        fig.colorbar(image, ax=ax, shrink=0.72)
    for ax in axes.ravel()[len(streams) :]:
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _write_graph_propagation_report(
    events: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    regression_rows: list[dict[str, Any]],
    paths: dict[str, Path],
) -> Path:
    event_frame = pd.DataFrame(events)
    lines = [
        "# S4 graph-distance mechanical propagation report",
        "",
        "## Purpose",
        "",
        "This analysis tests whether the release perturbation appears as a local mechanical redistribution through cell-adjacent springs. It uses released-vs-continuing deltas, not AUC, as the primary propagation object.",
        "",
        "## Source",
        "",
        f"- Package root: `{PACKAGE_ROOT}`",
        f"- Data root: `{_data_root()}`",
        f"- Event count: `{len(events)}`",
        f"- Cliff distances: `{int(event_frame['cliff_distance'].min())}` to `{int(event_frame['cliff_distance'].max())}`",
        f"- Relative-time grid: `{GRAPH_PROPAGATION_REL_TIMES[0]}..{GRAPH_PROPAGATION_REL_TIMES[-1]}` in 8-step increments",
        "",
        "## Streams",
        "",
        "- `connection_spring_current`: physical length state of springs connecting neighbouring voxels.",
        "- `connection_spring_contractile`: rest-length/contraction state on those connection springs.",
        "- `internal_spring_current` / `internal_spring_contractile`: within-voxel spring controls.",
        "- `body_kinematic` and `body_no_contractile`: cell geometry/velocity readouts.",
        "- `signal_hidden` and `controller_action`: negative/control streams.",
        "",
        "## Onset regressions",
        "",
    ]
    for row in regression_rows:
        lines.append(
            "- "
            f"`{row['stream']}`: distance count {int(row['distance_count_with_onset'])}; "
            f"slope {float(row['slope_steps_per_graph_distance']):.3f} steps/distance "
            f"[{float(row['slope_ci_low']):.3f}, {float(row['slope_ci_high']):.3f}], "
            f"R2 {float(row['r2']):.3f}, p+ {float(row['permutation_p_positive']):.4f}."
        )
    mechanical_streams = {
        "connection_spring_current",
        "connection_spring_contractile",
        "body_kinematic",
        "body_no_contractile",
    }
    gate_pass = any(
        row.get("stream") in mechanical_streams
        and np.isfinite(float(row.get("slope_ci_low", float("nan"))))
        and float(row.get("slope_ci_low", float("nan"))) > 0.0
        for row in regression_rows
    )
    lines.extend(
        [
            "",
            "## Gate decision",
            "",
            (
                "Gate passed: a mechanical stream has a positive graph-distance onset slope with its interval above zero."
                if gate_pass
                else "Gate failed for a clean travelling-wave claim: mechanical streams show release-locked perturbation, but their onset slopes do not establish ordered graph-distance propagation."
            ),
        ]
    )
    summary_frame = pd.DataFrame(summary_rows)
    if not summary_frame.empty:
        lines.extend(["", "## Peak post-release effects", ""])
        for stream in GRAPH_PROPAGATION_FIGURE_STREAMS:
            post = summary_frame[(summary_frame["stream"] == stream) & (summary_frame["rel_time"] >= 0)]
            if post.empty:
                continue
            best = post.sort_values("mean_l2_delta", ascending=False, kind="stable").iloc[0]
            lines.append(
                "- "
                f"`{stream}`: max mean L2 {float(best['mean_l2_delta']):.3f} "
                f"at distance {int(best['graph_distance'])}, t={int(best['rel_time']):+d}."
            )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
        ]
    )
    for key, value in paths.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation gate",
            "",
            "Use this as a propagation result only if a mechanical connection/body stream has a positive onset slope with an interval mostly above zero while signal/action controls remain flat. Otherwise, report release-locked mechanical redistribution without claiming a clean travelling wave.",
        ]
    )
    report_path = PACKAGE_ROOT / "S4_graph_propagation_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def analyze_graph_propagation_only(*, random_seed: int = 137, init_package: bool = False) -> dict[str, Path]:
    audit_paths = audit_remote_data(init_package=init_package)
    summary = json.loads((PACKAGE_ROOT / "tables" / "fixed_width_remote_data_audit_summary.json").read_text())
    if not bool(summary.get("complete")):
        raise RuntimeError("Fixed-width packet audit is incomplete; refusing graph propagation analysis.")
    events, _, _ = _load_events()
    if not events:
        raise ValueError("No true release events available for graph propagation analysis.")
    tables_dir = PACKAGE_ROOT / "tables"
    figures_dir = PACKAGE_ROOT / "figures"
    rows = _graph_propagation_rows(events)
    summary_rows = _graph_propagation_summary_rows(rows)
    onset_rows, regression_rows = _graph_onset_rows(summary_rows, random_seed=random_seed)
    paths = {
        "graph_propagation_by_cell": tables_dir / "graph_propagation_by_cell.csv",
        "graph_propagation_by_distance": tables_dir / "graph_propagation_by_distance.csv",
        "graph_propagation_onsets": tables_dir / "graph_propagation_onsets.csv",
        "graph_propagation_onset_regression": tables_dir / "graph_propagation_onset_regression.csv",
        "graph_propagation_heatmaps": figures_dir / "graph_propagation_heatmaps.png",
    }
    _write_csv(paths["graph_propagation_by_cell"], rows)
    _write_csv(paths["graph_propagation_by_distance"], summary_rows)
    _write_csv(paths["graph_propagation_onsets"], onset_rows)
    _write_csv(paths["graph_propagation_onset_regression"], regression_rows)
    _plot_graph_heatmaps(summary_rows, path=paths["graph_propagation_heatmaps"])
    payload = {
        "schema_version": 1,
        "kind": "s4_graph_propagation_analysis",
        "event_count": len(events),
        "streams": list(GRAPH_PROPAGATION_STREAMS),
        "relative_times": list(GRAPH_PROPAGATION_REL_TIMES),
        "continuer_margin_steps": CONTINUER_MARGIN_STEPS,
        "paths": {key: str(value) for key, value in paths.items()},
        "audit": {key: str(value) for key, value in audit_paths.items()},
    }
    summary_path = tables_dir / "graph_propagation_summary.json"
    _write_json(summary_path, payload)
    report_path = _write_graph_propagation_report(events, summary_rows, regression_rows, {**paths, "graph_propagation_summary": summary_path})
    return {
        **audit_paths,
        **paths,
        "graph_propagation_summary": summary_path,
        "graph_propagation_report": report_path,
    }


PCA_REGIME_STREAMS = (
    "signal_hidden",
    "controller_action",
    "body_kinematic",
    "spring_current",
    "spring_contractile",
    "body_no_contractile",
    "body_combined",
)

PCA_REGIME_FIGURE_STREAMS = (
    "body_kinematic",
    "body_no_contractile",
    "spring_current",
    "spring_contractile",
    "signal_hidden",
    "controller_action",
)


def _fit_pca_model_from_series(
    feature_rows: list[np.ndarray],
    *,
    feature_steps: list[int],
    n_components: int = PCA_REGIME_COMPONENTS,
) -> dict[str, Any] | None:
    if len(feature_rows) < 16:
        return None
    x = np.vstack([np.asarray(feature, dtype=np.float64).reshape(1, -1) for feature in feature_rows])
    mean = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd > 1.0e-8, sd, 1.0)
    z = (x - mean) / sd
    try:
        _, singular_values, vt = np.linalg.svd(z, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    component_count = min(int(n_components), int(vt.shape[0]))
    if component_count < 2:
        return None
    components = vt[:component_count].T
    scores = z @ components
    total_var = float(np.sum(np.square(singular_values)))
    explained = [
        float(singular_values[index] ** 2 / total_var) if total_var > 0 else float("nan")
        for index in range(component_count)
    ]
    return {
        "fit_steps": np.asarray(feature_steps, dtype=np.int32),
        "mean": mean,
        "sd": sd,
        "components": components,
        "scores": scores.astype(np.float32),
        "explained_variance": explained,
        "feature_dim": int(x.shape[1]),
        "component_count": int(component_count),
    }


def _project_pca_score(model: dict[str, Any], feature: np.ndarray) -> np.ndarray:
    z = (np.asarray(feature, dtype=np.float64).reshape(-1) - model["mean"]) / model["sd"]
    return (z @ model["components"]).astype(np.float32)


def _fit_pca_regime_models(events: list[dict[str, Any]]) -> tuple[dict[tuple[str, str, int], dict[str, Any]], list[dict[str, Any]]]:
    reference_event = max(events, key=lambda event: int(event["release_step"]))
    models: dict[tuple[str, str, int], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for stream in PCA_REGIME_STREAMS:
        for scope, cells in (("whole", [-1]), ("cell", list(range(int(reference_event["cell_count"]))))):
            for cell in cells:
                features: list[np.ndarray] = []
                steps: list[int] = []
                for step in range(0, int(reference_event["release_step"]), PCA_REGIME_FIT_STRIDE):
                    if scope == "whole":
                        feature = _whole_stream_feature(reference_event, stream=stream, step=step)
                    else:
                        feature = _cell_stream_feature(reference_event, stream=stream, cell=int(cell), step=step)
                    if feature is None or not np.all(np.isfinite(feature)):
                        continue
                    features.append(feature)
                    steps.append(int(step))
                model = _fit_pca_model_from_series(features, feature_steps=steps)
                key = (stream, scope, int(cell))
                if model is not None:
                    models[key] = model
                explained = [] if model is None else model["explained_variance"]
                rows.append(
                    {
                        "stream": stream,
                        "scope": scope,
                        "cell": "" if scope == "whole" else int(cell),
                        "reference_event_id": reference_event["event_id"],
                        "reference_cliff_distance": int(reference_event["cliff_distance"]),
                        "fit_sample_count": len(steps),
                        "fit_stride": PCA_REGIME_FIT_STRIDE,
                        "feature_dim": 0 if model is None else int(model["feature_dim"]),
                        "component_count": 0 if model is None else int(model["component_count"]),
                        "explained_var_pc1": explained[0] if len(explained) > 0 else float("nan"),
                        "explained_var_pc2": explained[1] if len(explained) > 1 else float("nan"),
                        "explained_var_pc3": explained[2] if len(explained) > 2 else float("nan"),
                        "explained_var_pc1_pc2": float(np.nansum(explained[:2])) if explained else float("nan"),
                        "explained_var_pc1_pc3": float(np.nansum(explained[:3])) if explained else float("nan"),
                        "fit_available": model is not None,
                    }
                )
    return models, rows


def _pca_sample_score(
    event: dict[str, Any],
    *,
    model: dict[str, Any],
    stream: str,
    scope: str,
    cell: int,
    step: int,
) -> np.ndarray | None:
    if scope == "whole":
        feature = _whole_stream_feature(event, stream=stream, step=int(step))
    else:
        feature = _cell_stream_feature(event, stream=stream, cell=int(cell), step=int(step))
    if feature is None or not np.all(np.isfinite(feature)):
        return None
    return _project_pca_score(model, feature)


def _pca_regime_rows(
    events: list[dict[str, Any]],
    models: dict[tuple[str, str, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # First pass: collect post-release PCA scores to define a stream/scope/cell
    # post-release regime centroid and threshold.
    post_scores: dict[tuple[str, str, int], list[np.ndarray]] = {key: [] for key in models}
    for key, model in models.items():
        stream, scope, cell = key
        for event in events:
            for rel_time in PCA_REGIME_REL_TIMES:
                if int(rel_time) < 128 or int(rel_time) > 336:
                    continue
                step = int(event["release_step"]) + int(rel_time)
                score = _pca_sample_score(event, model=model, stream=stream, scope=scope, cell=cell, step=step)
                if score is not None:
                    post_scores[key].append(score)

    regime_rows: list[dict[str, Any]] = []
    regime_definitions: dict[tuple[str, str, int], dict[str, Any]] = {}
    for key, scores in post_scores.items():
        stream, scope, cell = key
        if len(scores) < 4:
            continue
        stack = np.vstack(scores).astype(np.float32)
        centroid = np.mean(stack, axis=0)
        distances = np.linalg.norm(stack - centroid.reshape(1, -1), axis=1)
        threshold = float(np.quantile(distances, 0.75))
        regime_definitions[key] = {"centroid": centroid, "threshold": threshold}
        regime_rows.append(
            {
                "stream": stream,
                "scope": scope,
                "cell": "" if scope == "whole" else int(cell),
                "post_sample_count": int(len(scores)),
                "post_distance_threshold_q75": threshold,
                "post_distance_median": float(np.median(distances)),
                "post_distance_q95": float(np.quantile(distances, 0.95)),
            }
        )

    rows: list[dict[str, Any]] = []
    for key, model in models.items():
        if key not in regime_definitions:
            continue
        stream, scope, cell = key
        definition = regime_definitions[key]
        centroid = np.asarray(definition["centroid"], dtype=np.float32)
        threshold = float(definition["threshold"])
        for event in events:
            for rel_time in PCA_REGIME_REL_TIMES:
                sample_step = int(event["release_step"]) + int(rel_time)
                score = _pca_sample_score(event, model=model, stream=stream, scope=scope, cell=cell, step=sample_step)
                if score is None:
                    continue
                negative_scores: list[np.ndarray] = []
                for continuer in events:
                    if continuer["event_id"] == event["event_id"]:
                        continue
                    if int(continuer["release_step"]) <= sample_step + CONTINUER_MARGIN_STEPS:
                        continue
                    negative = _pca_sample_score(continuer, model=model, stream=stream, scope=scope, cell=cell, step=sample_step)
                    if negative is None:
                        continue
                    negative_scores.append(negative)
                if negative_scores:
                    baseline_score = np.mean(np.vstack(negative_scores).astype(np.float32), axis=0)
                    released_vs_continuing_l2 = float(np.linalg.norm(score - baseline_score))
                else:
                    released_vs_continuing_l2 = float("nan")
                distance_to_post = float(np.linalg.norm(score - centroid))
                rows.append(
                    {
                        "stream": stream,
                        "scope": scope,
                        "cell": "" if scope == "whole" else int(cell),
                        "event_id": event["event_id"],
                        "trial_id": event["trial_id"],
                        "cliff_distance": int(event["cliff_distance"]),
                        "release_step": int(event["release_step"]),
                        "rel_time": int(rel_time),
                        "sample_step": int(sample_step),
                        "success": bool(event["success"]),
                        "returned_from_edge": bool(event["returned_from_edge"]),
                        "distance_to_post_regime": distance_to_post,
                        "post_regime_threshold": threshold,
                        "in_post_regime": bool(distance_to_post <= threshold),
                        "released_vs_continuing_l2": released_vs_continuing_l2,
                        "continuing_count": int(len(negative_scores)),
                    }
                )
    return rows, regime_rows


def _pca_regime_summary_rows(rows: list[dict[str, Any]], events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return [], [], []
    distance_reference = _cell_graph_distances(max(events, key=lambda event: int(event["release_step"])))
    summary_rows: list[dict[str, Any]] = []
    for (stream, scope, cell, rel_time), group in frame.groupby(["stream", "scope", "cell", "rel_time"], sort=True):
        summary_rows.append(
            {
                "stream": str(stream),
                "scope": str(scope),
                "cell": cell,
                "graph_distance": "" if str(scope) == "whole" else int(distance_reference.get(int(cell), -1)),
                "rel_time": int(rel_time),
                "n": int(len(group)),
                "event_count": int(group["event_id"].nunique()),
                "fraction_in_post_regime": float(np.mean(group["in_post_regime"].to_numpy(dtype=bool))),
                "mean_distance_to_post_regime": float(group["distance_to_post_regime"].mean()),
                "median_distance_to_post_regime": float(group["distance_to_post_regime"].median()),
                "mean_released_vs_continuing_l2": float(group["released_vs_continuing_l2"].mean()),
                "median_released_vs_continuing_l2": float(group["released_vs_continuing_l2"].median()),
            }
        )

    onset_rows: list[dict[str, Any]] = []
    summary_frame = pd.DataFrame(summary_rows)
    for (stream, scope, cell), group in summary_frame.groupby(["stream", "scope", "cell"], sort=True):
        group = group.sort_values("rel_time", kind="stable")
        post = group[group["rel_time"] >= 0].reset_index(drop=True)
        onset = None
        if not post.empty:
            in_regime = post["fraction_in_post_regime"].to_numpy(dtype=np.float64) >= 0.5
            times = post["rel_time"].to_numpy(dtype=np.int32)
            for index in range(len(in_regime)):
                stop = min(len(in_regime), index + PCA_REGIME_ONSET_SUSTAIN_STEPS)
                if stop - index < PCA_REGIME_ONSET_SUSTAIN_STEPS:
                    continue
                if bool(np.all(in_regime[index:stop])):
                    onset = int(times[index])
                    break
        onset_rows.append(
            {
                "stream": str(stream),
                "scope": str(scope),
                "cell": cell,
                "graph_distance": "" if str(scope) == "whole" else int(distance_reference.get(int(cell), -1)),
                "metric": "fraction_in_post_regime>=0.5",
                "sustain_frames": PCA_REGIME_ONSET_SUSTAIN_STEPS,
                "onset_rel_time": onset,
                "onset_found": onset is not None,
            }
        )

    regression_rows: list[dict[str, Any]] = []
    onset_frame = pd.DataFrame(onset_rows)
    rng = np.random.default_rng(_stable_seed(137, "pca-regime-entry-onsets"))
    for stream, group in onset_frame[(onset_frame["scope"] == "cell") & (onset_frame["onset_found"])].groupby("stream", sort=True):
        fit = _linear_slope_with_ci(
            group["graph_distance"].to_numpy(dtype=np.float64),
            group["onset_rel_time"].to_numpy(dtype=np.float64),
            rng=rng,
            permutations=1000,
        )
        regression_rows.append(
            {
                "stream": str(stream),
                "scope": "cell",
                "distance_count_with_onset": int(len(group)),
                "slope_steps_per_graph_distance": fit["slope"],
                "slope_ci_low": fit["ci_low"],
                "slope_ci_high": fit["ci_high"],
                "intercept": fit["intercept"],
                "r2": fit["r2"],
                "permutation_p_positive": fit["permutation_p_positive"],
            }
        )
    return summary_rows, onset_rows, regression_rows


def _plot_pca_regime_whole(summary_rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(summary_rows)
    frame = frame[(frame["scope"] == "whole") & (frame["stream"].isin(PCA_REGIME_FIGURE_STREAMS))]
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    colors = {
        "body_kinematic": "#3B5B92",
        "body_no_contractile": "#8CB87C",
        "spring_current": "#5A4E8C",
        "spring_contractile": "#B58B4A",
        "signal_hidden": "#777777",
        "controller_action": "#AAAAAA",
    }
    for stream in PCA_REGIME_FIGURE_STREAMS:
        subset = frame[frame["stream"] == stream].sort_values("rel_time", kind="stable")
        if subset.empty:
            continue
        ax.plot(
            subset["rel_time"].to_numpy(dtype=np.int32),
            subset["fraction_in_post_regime"].to_numpy(dtype=np.float32),
            label=stream,
            linewidth=1.4,
            color=colors.get(stream),
        )
    ax.axvline(0, color="#D4736C", linewidth=1.0)
    ax.set_xlabel("Steps from release")
    ax.set_ylabel("Fraction in post-release PCA regime")
    ax.set_ylim(-0.03, 1.03)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_pca_regime_cell_heatmaps(summary_rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(summary_rows)
    frame = frame[(frame["scope"] == "cell") & (frame["stream"].isin(PCA_REGIME_FIGURE_STREAMS))]
    if frame.empty:
        return
    rel_times = sorted(int(value) for value in frame["rel_time"].dropna().unique())
    cells = sorted(int(value) for value in frame["cell"].dropna().unique())
    ncols = 2
    nrows = int(np.ceil(len(PCA_REGIME_FIGURE_STREAMS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 2.25 * nrows), squeeze=False)
    for ax, stream in zip(axes.ravel(), PCA_REGIME_FIGURE_STREAMS, strict=False):
        subset = frame[frame["stream"] == stream]
        matrix = np.full((len(cells), len(rel_times)), np.nan, dtype=np.float32)
        cell_to_row = {cell: idx for idx, cell in enumerate(cells)}
        time_to_col = {rel_time: idx for idx, rel_time in enumerate(rel_times)}
        for _, row in subset.iterrows():
            matrix[cell_to_row[int(row["cell"])], time_to_col[int(row["rel_time"])]] = float(row["fraction_in_post_regime"])
        image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
        if 0 in time_to_col:
            ax.axvline(time_to_col[0], color="#D4736C", linewidth=1.0)
        ax.set_title(stream, fontsize=9, weight="bold")
        tick_cols = [idx for idx, rel_time in enumerate(rel_times) if rel_time % 128 == 0]
        ax.set_xticks(tick_cols)
        ax.set_xticklabels([str(rel_times[idx]) for idx in tick_cols], fontsize=7)
        ax.set_yticks(range(len(cells)))
        ax.set_yticklabels([str(cell) for cell in cells], fontsize=7)
        ax.set_xlabel("Steps from release", fontsize=8)
        ax.set_ylabel("Cell", fontsize=8)
        fig.colorbar(image, ax=ax, shrink=0.72)
    for ax in axes.ravel()[len(PCA_REGIME_FIGURE_STREAMS) :]:
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _write_pca_regime_report(
    events: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    regime_definition_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    regression_rows: list[dict[str, Any]],
    paths: dict[str, Path],
) -> Path:
    event_frame = pd.DataFrame(events)
    summary_frame = pd.DataFrame(summary_rows)
    lines = [
        "# S4 PCA regime-entry report",
        "",
        "## Purpose",
        "",
        "This analysis asks whether the body enters a post-release dynamical regime after the frontier release, and whether that entry is visible in body/spring streams more than in hidden/internal signalling or action controls.",
        "",
        "## Source",
        "",
        f"- Package root: `{PACKAGE_ROOT}`",
        f"- Event count: `{len(events)}`",
        f"- Cliff distances: `{int(event_frame['cliff_distance'].min())}` to `{int(event_frame['cliff_distance'].max())}`",
        f"- PCA components: `{PCA_REGIME_COMPONENTS}`",
        f"- Fit stride: `{PCA_REGIME_FIT_STRIDE}`",
        f"- Regime definition: post-release samples from `+128..+336`; threshold is the 75th percentile distance to the post-release centroid.",
        "",
        "## Whole-body entry summary",
        "",
    ]
    whole = summary_frame[(summary_frame["scope"] == "whole") & (summary_frame["rel_time"] >= 0)] if not summary_frame.empty else pd.DataFrame()
    control_peak = float("nan")
    body_peak = float("nan")
    if not whole.empty:
        control_peak = float(
            whole[whole["stream"].isin(["signal_hidden", "controller_action"])]["fraction_in_post_regime"].max()
        )
        body_peak = float(
            whole[whole["stream"].isin(["body_kinematic", "body_no_contractile", "spring_current", "spring_contractile"])]["fraction_in_post_regime"].max()
        )
    for stream in PCA_REGIME_FIGURE_STREAMS:
        subset = whole[whole["stream"] == stream]
        if subset.empty:
            continue
        best = subset.sort_values("fraction_in_post_regime", ascending=False, kind="stable").iloc[0]
        at_128 = subset[subset["rel_time"] == 128]
        lines.append(
            "- "
            f"`{stream}`: max fraction {float(best['fraction_in_post_regime']):.3f} "
            f"at t={int(best['rel_time']):+d}"
            + (f"; +128 fraction {float(at_128.iloc[0]['fraction_in_post_regime']):.3f}" if not at_128.empty else "")
            + "."
        )
    gate_pass = bool(np.isfinite(body_peak) and body_peak >= 0.5 and np.isfinite(control_peak) and control_peak < 0.5)
    lines.extend(
        [
            "",
            "## Gate decision",
            "",
            (
                "Gate passed: body/spring streams enter the operational post-release PCA regime while signal/action controls do not."
                if gate_pass
                else f"Gate failed for a body-specific PCA-regime claim: body/spring streams enter the operational post-release regime, but signal/action controls also enter it (control peak {control_peak:.3f}; body/spring peak {body_peak:.3f}). Treat this PCA readout as a broad phase/regime diagnostic, not as evidence for a body-only propagation mechanism."
            ),
        ]
    )
    lines.extend(["", "## Cell onset regressions", ""])
    for row in regression_rows:
        lines.append(
            "- "
            f"`{row['stream']}`: distance count {int(row['distance_count_with_onset'])}; "
            f"slope {float(row['slope_steps_per_graph_distance']):.3f} steps/distance "
            f"[{float(row['slope_ci_low']):.3f}, {float(row['slope_ci_high']):.3f}], "
            f"R2 {float(row['r2']):.3f}, p+ {float(row['permutation_p_positive']):.4f}."
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
        ]
    )
    for key, value in paths.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation gate",
            "",
            "Use this as a regime-entry result only if body/spring streams enter the post-release PCA regime clearly while signal/action controls do not. If entry appears diffuse or simultaneous across the whole body, describe it as a global phase reset rather than local propagation.",
        ]
    )
    report_path = PACKAGE_ROOT / "S4_pca_regime_entry_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def analyze_pca_regime_entry_only(*, random_seed: int = 137, init_package: bool = False) -> dict[str, Path]:
    _ = random_seed
    audit_paths = audit_remote_data(init_package=init_package)
    summary = json.loads((PACKAGE_ROOT / "tables" / "fixed_width_remote_data_audit_summary.json").read_text())
    if not bool(summary.get("complete")):
        raise RuntimeError("Fixed-width packet audit is incomplete; refusing PCA regime-entry analysis.")
    events, _, _ = _load_events()
    if not events:
        raise ValueError("No true release events available for PCA regime-entry analysis.")
    tables_dir = PACKAGE_ROOT / "tables"
    figures_dir = PACKAGE_ROOT / "figures"
    models, model_rows = _fit_pca_regime_models(events)
    rows, regime_definition_rows = _pca_regime_rows(events, models)
    summary_rows, onset_rows, regression_rows = _pca_regime_summary_rows(rows, events)
    paths = {
        "pca_regime_entry_samples": tables_dir / "pca_regime_entry_samples.csv",
        "pca_regime_entry_model_summary": tables_dir / "pca_regime_entry_model_summary.csv",
        "pca_regime_entry_definitions": tables_dir / "pca_regime_entry_definitions.csv",
        "pca_regime_entry_summary": tables_dir / "pca_regime_entry_summary.csv",
        "pca_regime_entry_onsets": tables_dir / "pca_regime_entry_onsets.csv",
        "pca_regime_entry_onset_regression": tables_dir / "pca_regime_entry_onset_regression.csv",
        "pca_regime_entry_whole_body": figures_dir / "pca_regime_entry_whole_body.png",
        "pca_regime_entry_cell_heatmaps": figures_dir / "pca_regime_entry_cell_heatmaps.png",
    }
    _write_csv(paths["pca_regime_entry_samples"], rows)
    _write_csv(paths["pca_regime_entry_model_summary"], model_rows)
    _write_csv(paths["pca_regime_entry_definitions"], regime_definition_rows)
    _write_csv(paths["pca_regime_entry_summary"], summary_rows)
    _write_csv(paths["pca_regime_entry_onsets"], onset_rows)
    _write_csv(paths["pca_regime_entry_onset_regression"], regression_rows)
    _plot_pca_regime_whole(summary_rows, paths["pca_regime_entry_whole_body"])
    _plot_pca_regime_cell_heatmaps(summary_rows, paths["pca_regime_entry_cell_heatmaps"])
    payload = {
        "schema_version": 1,
        "kind": "s4_pca_regime_entry_analysis",
        "event_count": len(events),
        "streams": list(PCA_REGIME_STREAMS),
        "relative_times": list(PCA_REGIME_REL_TIMES),
        "components": PCA_REGIME_COMPONENTS,
        "regime_definition": "post-release samples +128..+336; threshold=q75 distance to post centroid",
        "paths": {key: str(value) for key, value in paths.items()},
        "audit": {key: str(value) for key, value in audit_paths.items()},
    }
    summary_path = tables_dir / "pca_regime_entry_payload.json"
    _write_json(summary_path, payload)
    report_path = _write_pca_regime_report(events, model_rows, regime_definition_rows, summary_rows, regression_rows, {**paths, "pca_regime_entry_payload": summary_path})
    return {
        **audit_paths,
        **paths,
        "pca_regime_entry_payload": summary_path,
        "pca_regime_entry_report": report_path,
    }


def analyze_remaining_mechanism(*, random_seed: int = 137, init_package: bool = False) -> dict[str, Path]:
    audit_paths = audit_remote_data(init_package=init_package)
    summary = json.loads((PACKAGE_ROOT / "tables" / "fixed_width_remote_data_audit_summary.json").read_text())
    if not bool(summary.get("complete")):
        raise RuntimeError("Fixed-width packet audit is incomplete; refusing remaining mechanism analyses.")
    graph_paths = analyze_graph_propagation_only(random_seed=random_seed, init_package=False)
    pca_paths = analyze_pca_regime_entry_only(random_seed=random_seed, init_package=False)
    report_path = PACKAGE_ROOT / "S4_remaining_mechanism_experiments_report.md"
    lines = [
        "# S4 remaining mechanism experiments",
        "",
        "## Completed packet components",
        "",
        "- P0 remote-canonical data audit.",
        "- P1 graph-distance mechanical propagation analysis.",
        "- P2 PCA regime-entry analysis.",
        "",
        "## Reports",
        "",
        f"- Data audit: `{audit_paths['remote_data_audit_report']}`",
        f"- Graph propagation: `{graph_paths['graph_propagation_report']}`",
        f"- PCA regime entry: `{pca_paths['pca_regime_entry_report']}`",
        "",
        "## Paper-use rule",
        "",
        "Use the graph-propagation or PCA-regime panels only if their report-level interpretation gates pass. Otherwise, retain the safer body-vs-signal phase/direction results and describe these outputs as exploratory support for release-locked mechanical redistribution.",
    ]
    report_path.write_text("\n".join(lines) + "\n")
    return {
        **audit_paths,
        **{f"graph_{key}": value for key, value in graph_paths.items()},
        **{f"pca_{key}": value for key, value in pca_paths.items()},
        "remaining_mechanism_report": report_path,
    }


def _run_phase_propagation_analysis(
    events: list[dict[str, Any]],
    figures_dir: Path,
    *,
    random_seed: int,
) -> dict[str, Path]:
    if not events:
        return {}
    reference_event = max(events, key=lambda event: int(event["release_step"]))
    cell_count = min(int(event["cell_count"]) for event in events)
    models: dict[tuple[str, int], dict[str, Any]] = {}
    reference_rows: list[dict[str, Any]] = []
    for stream in PROPAGATION_STREAMS:
        for cell in range(cell_count):
            model = _fit_local_phase_model(reference_event, stream=stream, cell=cell)
            if model is not None:
                models[(stream, int(cell))] = model
            reference_rows.append(
                {
                    "stream": stream,
                    "cell": int(cell),
                    "reference_event_id": reference_event["event_id"],
                    "reference_cliff_distance": int(reference_event["cliff_distance"]),
                    "fit_sample_count": 0 if model is None else int(len(model["fit_steps"])),
                    "fit_stride": int(PHASE_FIT_STRIDE),
                    "feature_dim": 0 if model is None else int(model["feature_dim"]),
                    "explained_var_pc1": float("nan") if model is None else float(model["explained_var_pc1"]),
                    "explained_var_pc2": float("nan") if model is None else float(model["explained_var_pc2"]),
                    "explained_var_pc1_pc2": float("nan") if model is None else float(model["explained_var_pc1"] + model["explained_var_pc2"]),
                    "fit_available": model is not None,
                }
            )

    local_rows = _rank_local_phase_rows(events, models=models)
    local_summary = _rank_local_phase_summary_rows(local_rows)
    decoder_rows = _phase_propagation_direction_decoder_rows(
        local_rows,
        random_seed=_stable_seed(random_seed, "phase-propagation-direction"),
    )
    onset_rows, regression_rows = _phase_propagation_onset_rows(
        local_summary,
        decoder_rows,
        random_seed=random_seed,
    )

    tables_dir = PACKAGE_ROOT / "tables"
    _write_csv(tables_dir / "phase_propagation_reference_summary.csv", reference_rows)
    _write_csv(tables_dir / "phase_propagation_local_slip_by_event.csv", local_rows)
    _write_csv(tables_dir / "phase_propagation_local_slip.csv", local_summary)
    _write_csv(tables_dir / "phase_propagation_direction_decoder.csv", decoder_rows)
    _write_csv(tables_dir / "phase_propagation_onsets.csv", onset_rows)
    _write_csv(tables_dir / "phase_propagation_onset_regression.csv", regression_rows)

    _plot_phase_propagation_heatmaps(
        local_summary,
        value_col="mean_abs_phase_slip",
        title_prefix="Local phase/deviation after release",
        path=figures_dir / "phase_propagation_local_slip_heatmaps.png",
        cmap="magma",
        vmin=0.0,
    )
    _plot_phase_propagation_heatmaps(
        decoder_rows,
        value_col="auc",
        title_prefix="Sustained-direction decodability from local phase response",
        path=figures_dir / "phase_propagation_direction_auc_heatmaps.png",
        cmap="viridis",
        vmin=0.5,
        vmax=1.0,
    )
    _plot_phase_propagation_onsets(onset_rows, figures_dir / "phase_propagation_onsets.png")
    _plot_phase_propagation_gif(local_summary, figures_dir / "phase_propagation_local_slip.gif")

    return {
        "phase_propagation_reference_summary": tables_dir / "phase_propagation_reference_summary.csv",
        "phase_propagation_local_slip_by_event": tables_dir / "phase_propagation_local_slip_by_event.csv",
        "phase_propagation_local_slip": tables_dir / "phase_propagation_local_slip.csv",
        "phase_propagation_direction_decoder": tables_dir / "phase_propagation_direction_decoder.csv",
        "phase_propagation_onsets": tables_dir / "phase_propagation_onsets.csv",
        "phase_propagation_onset_regression": tables_dir / "phase_propagation_onset_regression.csv",
        "phase_propagation_local_slip_heatmaps": figures_dir / "phase_propagation_local_slip_heatmaps.png",
        "phase_propagation_direction_auc_heatmaps": figures_dir / "phase_propagation_direction_auc_heatmaps.png",
        "phase_propagation_onsets_figure": figures_dir / "phase_propagation_onsets.png",
        "phase_propagation_local_slip_gif": figures_dir / "phase_propagation_local_slip.gif",
    }


def _phase_propagation_report_lines(phase_paths: dict[str, Path]) -> list[str]:
    if not phase_paths:
        return [
            "",
            "## Phase-propagation analysis",
            "",
            "Phase-propagation outputs were not generated.",
        ]
    local_path = phase_paths.get("phase_propagation_local_slip")
    decoder_path = phase_paths.get("phase_propagation_direction_decoder")
    regression_path = phase_paths.get("phase_propagation_onset_regression")
    if local_path is None or not Path(local_path).exists():
        return [
            "",
            "## Phase-propagation analysis",
            "",
            "Phase-propagation output paths were registered, but the expected local-slip table was not found.",
        ]
    local = pd.read_csv(local_path)
    decoder = pd.read_csv(decoder_path) if decoder_path is not None and Path(decoder_path).exists() else pd.DataFrame()
    regression = pd.read_csv(regression_path) if regression_path is not None and Path(regression_path).exists() else pd.DataFrame()
    lines = [
        "",
        "## Phase-propagation analysis",
        "",
        "This is the derived test for local communication of the release event through the body. Rank 0 is the frontier-nearest cell at release. Phase/deviation is measured per fixed cell identity from a pre-release PCA rhythm fitted on the longest-distance trace, then summarized by rank-from-edge at each event.",
        "",
        "Direction labels:",
        f"- Future window: `{PROPAGATION_DIRECTION_FUTURE_WINDOW}` steps.",
        f"- Minimum COM displacement: `{PROPAGATION_DIRECTION_MIN_DISPLACEMENT}` in the away-from-cliff coordinate.",
        f"- Persistence: `{PROPAGATION_DIRECTION_PERSISTENCE_STEPS}` steps with fraction `{PROPAGATION_DIRECTION_PERSISTENCE_FRACTION}`.",
        "- Positive label means sustained movement away from the cliff; negative label means sustained movement toward the cliff; uncommitted samples are excluded from the direction decoder.",
        f"- Direction decoding uses leave-one-event-out ridge regression without an intercept and zeroes phase-response features below `{PROPAGATION_FEATURE_DUST_FLOOR:g}`, so flat control streams score at chance rather than inheriting fold-level class-balance shifts or numerical dust.",
        "",
        "Phase/deviation stream summaries:",
    ]
    for stream in PROPAGATION_FIGURE_STREAMS:
        stream_frame = local[(local["stream"] == stream) & (local["rel_time"] >= 0)]
        if stream_frame.empty:
            lines.append(f"- `{stream}`: no local-slip rows.")
            continue
        best = stream_frame.sort_values("mean_abs_phase_slip", ascending=False, kind="stable").iloc[0]
        lines.append(
            "- "
            f"`{stream}`: max mean |phase slip| {float(best['mean_abs_phase_slip']):.3f} rad "
            f"at rank {int(best['rank_from_edge'])}, +{int(best['rel_time'])} steps; "
            f"mean PC-L2 delta {float(best['mean_pc_l2_delta']):.3f}."
        )
    if not decoder.empty:
        lines.extend(["", "Sustained-direction decoder summaries:"])
        for stream in PROPAGATION_FIGURE_STREAMS:
            stream_frame = decoder[
                (decoder["stream"] == stream)
                & (decoder["rel_time"] >= 0)
                & (decoder["estimable"])
            ]
            if stream_frame.empty:
                lines.append(f"- `{stream}`: no estimable away/toward rows.")
                continue
            best = stream_frame.sort_values("auc", ascending=False, kind="stable").iloc[0]
            significant = stream_frame[(stream_frame["auc"] > stream_frame["null_auc_q95"]) & (stream_frame["one_sided_p"] <= 0.05)]
            lines.append(
                "- "
                f"`{stream}`: best AUC {float(best['auc']):.3f} "
                f"at rank {int(best['rank_from_edge'])}, +{int(best['rel_time'])} steps; "
                f"significant rows {len(significant)}/{len(stream_frame)}."
            )
    if not regression.empty:
        lines.extend(["", "Onset regressions:"])
        for _, row in regression.iterrows():
            lines.append(
                "- "
                f"`{row['stream']}` `{row['metric']}`: onset ranks {int(row['rank_count_with_onset'])}; "
                f"slope {float(row['slope_steps_per_rank']):.3f} steps/rank "
                f"[{float(row['slope_ci_low']):.3f}, {float(row['slope_ci_high']):.3f}], "
                f"R2 {float(row['r2']):.3f}, positive-slope permutation p {float(row['permutation_p_positive']):.4f}."
            )
    lines.extend(
        [
            "",
            "Interpretation gate: claim a front-to-back propagation only if the body or spring phase-slip onset slope is positive with its interval mostly above zero and the hidden/action controls do not show the same structure. If that gate fails, the paper-safe claim is narrower: release perturbs plant phase locally relative to the still-continuing open-loop baseline, while direction-predictive structure is evaluated separately by the AUC map.",
        ]
    )
    return lines


def _phase_slip_rows(
    events: list[dict[str, Any]],
    *,
    stream: str,
    model: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        for rel_time in PHASE_REL_TIMES:
            sample_step = int(event["release_step"]) + int(rel_time)
            positive = _whole_stream_feature(event, stream=stream, step=sample_step)
            if positive is None:
                continue
            pos_phase, pos_radius, pos_score = _project_phase(model, positive)
            negative_phases: list[float] = []
            negative_radii: list[float] = []
            negative_scores: list[np.ndarray] = []
            for continuer in events:
                if continuer["event_id"] == event["event_id"]:
                    continue
                if int(continuer["release_step"]) <= sample_step + CONTINUER_MARGIN_STEPS:
                    continue
                negative = _whole_stream_feature(continuer, stream=stream, step=sample_step)
                if negative is None:
                    continue
                neg_phase, neg_radius, neg_score = _project_phase(model, negative)
                negative_phases.append(neg_phase)
                negative_radii.append(neg_radius)
                negative_scores.append(neg_score)
            if not negative_phases:
                continue
            baseline_phase = _circular_mean(np.asarray(negative_phases, dtype=np.float64))
            baseline_radius = float(np.mean(np.asarray(negative_radii, dtype=np.float64)))
            baseline_score = np.mean(np.vstack(negative_scores).astype(np.float32), axis=0)
            phase_slip = float(_wrap_angle(pos_phase - baseline_phase))
            radius_delta = float(pos_radius - baseline_radius)
            pc_l2_delta = float(np.linalg.norm(pos_score - baseline_score))
            rows.append(
                {
                    "stream": stream,
                    "event_id": event["event_id"],
                    "trial_id": event["trial_id"],
                    "cliff_distance": int(event["cliff_distance"]),
                    "release_step": int(event["release_step"]),
                    "rel_time": int(rel_time),
                    "sample_step": int(sample_step),
                    "success": bool(event["success"]),
                    "returned_from_edge": bool(event["returned_from_edge"]),
                    "steps_after_release": int(event["steps_alive"]) - int(event["release_step"]),
                    "positive_phase": pos_phase,
                    "baseline_phase": baseline_phase,
                    "phase_slip": phase_slip,
                    "abs_phase_slip": abs(phase_slip),
                    "positive_radius": pos_radius,
                    "baseline_radius": baseline_radius,
                    "radius_delta": radius_delta,
                    "pc_l2_delta": pc_l2_delta,
                    "continuing_count": int(len(negative_phases)),
                }
            )
    return rows


def _phase_summary_rows(phase_slip_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(phase_slip_rows)
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return rows
    for (stream, rel_time), group in frame.groupby(["stream", "rel_time"], sort=True):
        slips = group["phase_slip"].to_numpy(dtype=np.float64)
        abs_slips = np.abs(slips)
        rows.append(
            {
                "stream": str(stream),
                "rel_time": int(rel_time),
                "n": int(len(group)),
                "mean_phase_slip": _circular_mean(slips),
                "mean_abs_phase_slip": float(np.mean(abs_slips)),
                "median_abs_phase_slip": float(np.median(abs_slips)),
                "sd_abs_phase_slip": float(np.std(abs_slips, ddof=1)) if len(abs_slips) > 1 else float("nan"),
                "mean_radius_delta": float(group["radius_delta"].mean()),
                "median_radius_delta": float(group["radius_delta"].median()),
                "mean_pc_l2_delta": float(group["pc_l2_delta"].mean()),
                "median_pc_l2_delta": float(group["pc_l2_delta"].median()),
            }
        )
    return rows


def _phase_event_rows(
    events: list[dict[str, Any]],
    *,
    stream: str,
    model: dict[str, Any],
    phase_slip_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    slip_frame = pd.DataFrame(phase_slip_rows)
    rows: list[dict[str, Any]] = []
    for event in events:
        release_pre_step = max(0, int(event["release_step"]) - 1)
        release_feature = _whole_stream_feature(event, stream=stream, step=release_pre_step)
        if release_feature is None:
            continue
        phase, radius, score = _project_phase(model, release_feature)
        event_slip = slip_frame[
            (slip_frame["stream"] == stream)
            & (slip_frame["event_id"] == event["event_id"])
            & (slip_frame["rel_time"] >= 0)
            & (slip_frame["rel_time"] <= 128)
        ]
        rows.append(
            {
                "stream": stream,
                "event_id": event["event_id"],
                "trial_id": event["trial_id"],
                "cliff_distance": int(event["cliff_distance"]),
                "release_step": int(event["release_step"]),
                "release_phase_pre": phase,
                "release_radius_pre": radius,
                "release_pc1_pre": float(score[0]),
                "release_pc2_pre": float(score[1]),
                "success": bool(event["success"]),
                "returned_from_edge": bool(event["returned_from_edge"]),
                "cliff_fell": bool(event["cliff_fell"]),
                "steps_after_release": int(event["steps_alive"]) - int(event["release_step"]),
                "mean_abs_phase_slip_0_128": float(event_slip["abs_phase_slip"].mean()) if not event_slip.empty else float("nan"),
                "mean_pc_l2_delta_0_128": float(event_slip["pc_l2_delta"].mean()) if not event_slip.empty else float("nan"),
                "mean_radius_delta_0_128": float(event_slip["radius_delta"].mean()) if not event_slip.empty else float("nan"),
            }
        )
    return rows


def _phase_predictor_r2(phase: np.ndarray, y: np.ndarray) -> float:
    phase = np.asarray(phase, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(phase) & np.isfinite(y)
    phase = phase[ok]
    y = y[ok]
    if len(y) < 5:
        return float("nan")
    design = np.column_stack([np.ones_like(phase), np.sin(phase), np.cos(phase)])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ beta
    denom = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - np.sum((y - pred) ** 2) / denom) if denom > 0 else float("nan")


def _phase_correlation_rows(
    event_rows: list[dict[str, Any]],
    *,
    random_seed: int,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(event_rows)
    if frame.empty:
        return []
    rng = np.random.default_rng(int(random_seed))
    rows: list[dict[str, Any]] = []
    metrics = (
        "steps_after_release",
        "mean_abs_phase_slip_0_128",
        "mean_pc_l2_delta_0_128",
        "mean_radius_delta_0_128",
    )
    for stream, group in frame.groupby("stream", sort=False):
        phase = group["release_phase_pre"].to_numpy(dtype=np.float64)
        for metric in metrics:
            y = group[metric].to_numpy(dtype=np.float64)
            actual = _phase_predictor_r2(phase, y)
            if not np.isfinite(actual):
                p_value = float("nan")
                null_q95 = float("nan")
            else:
                null = np.asarray(
                    [_phase_predictor_r2(rng.permutation(phase), y) for _ in range(PHASE_PERMUTATIONS)],
                    dtype=np.float32,
                )
                null_q95 = float(np.quantile(null[np.isfinite(null)], 0.95)) if np.any(np.isfinite(null)) else float("nan")
                p_value = float((np.sum(null >= actual) + 1.0) / (len(null) + 1.0))
            rows.append(
                {
                    "stream": str(stream),
                    "metric": metric,
                    "n": int(np.sum(np.isfinite(phase) & np.isfinite(y))),
                    "phase_r2": actual,
                    "null_r2_q95": null_q95,
                    "one_sided_p": p_value,
                }
            )
    return rows


def _plot_phase_slip_summary(summary_rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(summary_rows)
    if frame.empty:
        return
    labels = {
        "signal_hidden": "hidden/internal",
        "controller_action": "action",
        "spring_contractile": "contractile",
        "spring_current": "spring length",
        "body_kinematic": "kinematics",
        "body_no_contractile": "body no contraction",
    }
    colors = {
        "signal_hidden": "#5A4E8C",
        "controller_action": "#B58B4A",
        "spring_contractile": "#8CB87C",
        "spring_current": "#3B5B92",
        "body_kinematic": "#6E9DAA",
        "body_no_contractile": "#4E6F8C",
    }
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for stream in PHASE_FIGURE_STREAMS:
        stream_frame = frame[frame["stream"] == stream].sort_values("rel_time")
        if stream_frame.empty:
            continue
        ax.plot(
            stream_frame["rel_time"].to_numpy(dtype=float),
            stream_frame["mean_abs_phase_slip"].to_numpy(dtype=float),
            label=labels.get(stream, stream),
            color=colors.get(stream, "#777777"),
            linewidth=1.5,
        )
    ax.axvline(0, color="#D4736C", linewidth=1.0)
    ax.set_xlabel("Steps from release")
    ax.set_ylabel("Mean absolute phase slip (rad)")
    ax.set_title("Release-aligned phase slip", fontsize=10, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_phase_loop(
    model: dict[str, Any],
    event_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(event_rows)
    stream = str(model["stream"])
    stream_frame = frame[frame["stream"] == stream]
    if stream_frame.empty:
        return
    scores = np.asarray(model["scores"], dtype=np.float32)
    fit_steps = np.asarray(model["fit_steps"], dtype=np.int32)
    stride = max(1, len(scores) // 3000)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    scatter = ax.scatter(
        scores[::stride, 0],
        scores[::stride, 1],
        c=fit_steps[::stride],
        s=3,
        cmap="Greys",
        alpha=0.35,
        linewidths=0,
    )
    release = ax.scatter(
        stream_frame["release_pc1_pre"].to_numpy(dtype=float),
        stream_frame["release_pc2_pre"].to_numpy(dtype=float),
        c=stream_frame["cliff_distance"].to_numpy(dtype=float),
        s=18,
        cmap="viridis",
        edgecolors="#222222",
        linewidths=0.3,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{stream} pre-release rhythm", fontsize=10, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.colorbar(release, ax=ax, shrink=0.76, label="Cliff distance")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_release_phase_vs_slip(event_rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(event_rows)
    if frame.empty:
        return
    streams = tuple(stream for stream in ("signal_hidden", "controller_action", "spring_contractile", "body_no_contractile") if stream in set(frame["stream"]))
    if not streams:
        return
    fig, axes = plt.subplots(1, len(streams), figsize=(2.5 * len(streams), 2.5), sharey=True)
    if len(streams) == 1:
        axes = [axes]
    for ax, stream in zip(axes, streams, strict=True):
        group = frame[frame["stream"] == stream]
        ax.scatter(
            group["release_phase_pre"].to_numpy(dtype=float),
            group["mean_abs_phase_slip_0_128"].to_numpy(dtype=float),
            c=group["cliff_distance"].to_numpy(dtype=float),
            cmap="viridis",
            s=18,
            edgecolors="#222222",
            linewidths=0.25,
        )
        ax.set_title(stream.replace("_", " "), fontsize=8)
        ax.set_xlabel("Release phase (rad)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean |phase slip| 0..128", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _run_phase_reset_analysis(
    events: list[dict[str, Any]],
    figures_dir: Path,
    *,
    random_seed: int,
) -> dict[str, Path]:
    if not events:
        return {}
    reference_event = max(events, key=lambda event: int(event["release_step"]))
    reference_rows: list[dict[str, Any]] = []
    all_slip_rows: list[dict[str, Any]] = []
    all_event_rows: list[dict[str, Any]] = []
    models: dict[str, dict[str, Any]] = {}
    for stream in PHASE_STREAMS:
        model = _fit_phase_model(reference_event, stream=stream)
        if model is None:
            reference_rows.append(
                {
                    "stream": stream,
                    "reference_event_id": reference_event["event_id"],
                    "fit_sample_count": 0,
                    "explained_var_pc1": float("nan"),
                    "explained_var_pc2": float("nan"),
                    "explained_var_pc1_pc2": float("nan"),
                }
            )
            continue
        models[stream] = model
        reference_rows.append(
            {
                "stream": stream,
                "reference_event_id": reference_event["event_id"],
                "reference_cliff_distance": int(reference_event["cliff_distance"]),
                "fit_sample_count": int(len(model["fit_steps"])),
                "fit_stride": int(PHASE_FIT_STRIDE),
                "explained_var_pc1": float(model["explained_var_pc1"]),
                "explained_var_pc2": float(model["explained_var_pc2"]),
                "explained_var_pc1_pc2": float(model["explained_var_pc1"] + model["explained_var_pc2"]),
            }
        )
        stream_slip = _phase_slip_rows(events, stream=stream, model=model)
        all_slip_rows.extend(stream_slip)
        stream_event_rows = _phase_event_rows(events, stream=stream, model=model, phase_slip_rows=stream_slip)
        all_event_rows.extend(stream_event_rows)

    summary_rows = _phase_summary_rows(all_slip_rows)
    correlation_rows = _phase_correlation_rows(all_event_rows, random_seed=_stable_seed(random_seed, "phase-reset"))

    tables_dir = PACKAGE_ROOT / "tables"
    _write_csv(tables_dir / "phase_reset_reference_summary.csv", reference_rows)
    _write_csv(tables_dir / "phase_reset_slip_by_event.csv", all_slip_rows)
    _write_csv(tables_dir / "phase_reset_slip_summary.csv", summary_rows)
    _write_csv(tables_dir / "phase_reset_events.csv", all_event_rows)
    _write_csv(tables_dir / "phase_reset_release_phase_correlations.csv", correlation_rows)

    _plot_phase_slip_summary(summary_rows, figures_dir / "phase_reset_slip_summary.png")
    for stream, model in models.items():
        if stream in PHASE_FIGURE_STREAMS:
            _plot_phase_loop(model, all_event_rows, figures_dir / f"phase_reset_loop_{stream}.png")
    _plot_release_phase_vs_slip(all_event_rows, figures_dir / "phase_reset_release_phase_vs_slip.png")

    return {
        "phase_reset_reference_summary": tables_dir / "phase_reset_reference_summary.csv",
        "phase_reset_slip_by_event": tables_dir / "phase_reset_slip_by_event.csv",
        "phase_reset_slip_summary": tables_dir / "phase_reset_slip_summary.csv",
        "phase_reset_events": tables_dir / "phase_reset_events.csv",
        "phase_reset_release_phase_correlations": tables_dir / "phase_reset_release_phase_correlations.csv",
        "phase_reset_slip_summary_figure": figures_dir / "phase_reset_slip_summary.png",
        "phase_reset_release_phase_vs_slip_figure": figures_dir / "phase_reset_release_phase_vs_slip.png",
    }


def _write_phase_reset_report(events: list[dict[str, Any]], phase_paths: dict[str, Path]) -> Path:
    event_frame = pd.DataFrame(
        [
            {
                "cliff_distance": event["cliff_distance"],
                "release_step": event["release_step"],
                "success": event["success"],
                "returned_from_edge": event["returned_from_edge"],
                "cliff_fell": event["cliff_fell"],
            }
            for event in events
        ]
    )
    lines = [
        "# S4 phase-reset analysis report",
        "",
        "## Purpose",
        "",
        "This derived analysis tests the phase-interruption interpretation of the S4 deformation-off distance sweep. It asks whether the cliff release perturbs an otherwise identical locomotor/controller rhythm, rather than producing a clean spatial propagation wave.",
        "",
        "## Source packet",
        "",
        f"- Package root: `{PACKAGE_ROOT}`",
        f"- Specimen: `{SPECIMEN_ID}`",
        f"- Modal profile: `{GPU_PROFILE}`",
        "- Condition: deformation off, intact signalling",
        f"- Cliff distances: `{CLIFF_DISTANCES[0]}..{CLIFF_DISTANCES[-1]}`",
        f"- Release events: `{len(events)}/{len(CLIFF_DISTANCES)}`",
        f"- Returned from edge: `{int(event_frame['returned_from_edge'].sum())}/{len(events)}`",
        f"- Success: `{int(event_frame['success'].sum())}/{len(events)}`",
        f"- Falls: `{int(event_frame['cliff_fell'].sum())}/{len(events)}`",
        f"- Release-step range: `{int(event_frame['release_step'].min())}` to `{int(event_frame['release_step'].max())}`",
    ]
    lines.extend(_phase_report_lines(phase_paths))
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `tables/phase_reset_reference_summary.csv`",
            "- `tables/phase_reset_slip_by_event.csv`",
            "- `tables/phase_reset_slip_summary.csv`",
            "- `tables/phase_reset_events.csv`",
            "- `tables/phase_reset_release_phase_correlations.csv`",
            "- `figures/phase_reset_slip_summary.pdf`",
            "- `figures/phase_reset_loop_<stream>.pdf`",
            "- `figures/phase_reset_release_phase_vs_slip.pdf`",
            "",
            "## Interpretation",
            "",
            "Treat this as an indicative phase-reset analysis. The strongest supported version of the S4 mechanism is a controller/plant dissociation after exact pre-release identity: the controller streams remain open-loop, while the plant streams are phase-shifted by release. It should not be worded as proof of a literal front-to-back mechanical wave, or as proof that release phase predicts behavioral outcome, unless separate tests support those claims.",
        ]
    )
    path = PACKAGE_ROOT / "S4_phase_reset_report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def analyze_phase_reset_only(*, random_seed: int = 137, init_package: bool = False) -> dict[str, Path]:
    summarize(init_package=init_package)
    events, _, _ = _load_events()
    if not events:
        raise ValueError("No true release events available for phase-reset analysis.")
    figures_dir = PACKAGE_ROOT / "figures"
    phase_paths = _run_phase_reset_analysis(events, figures_dir, random_seed=random_seed)
    phase_report = _write_phase_reset_report(events, phase_paths)
    payload = {
        "schema_version": 1,
        "kind": "s4_fixed_width_phase_reset_analysis",
        "event_count": len(events),
        "streams": list(PHASE_STREAMS),
        "relative_times": list(PHASE_REL_TIMES),
        "fit_stride": PHASE_FIT_STRIDE,
        "continuer_margin_steps": CONTINUER_MARGIN_STEPS,
        "paths": {key: str(value) for key, value in phase_paths.items()},
        "phase_report": str(phase_report),
    }
    _write_json(PACKAGE_ROOT / "tables" / "phase_reset_summary.json", payload)
    outputs = {"phase_reset_summary": PACKAGE_ROOT / "tables" / "phase_reset_summary.json", "phase_report": phase_report}
    outputs.update(phase_paths)
    return outputs


def _write_phase_propagation_report(events: list[dict[str, Any]], propagation_paths: dict[str, Path]) -> Path:
    event_frame = pd.DataFrame(
        [
            {
                "cliff_distance": event["cliff_distance"],
                "release_step": event["release_step"],
                "success": event["success"],
                "returned_from_edge": event["returned_from_edge"],
                "cliff_fell": event["cliff_fell"],
            }
            for event in events
        ]
    )
    label_rows: list[dict[str, Any]] = []
    for event in events:
        labels, deltas = _edge_oriented_committed_labels(event)
        for rel_time in PROPAGATION_REL_TIMES:
            step = int(event["release_step"]) + int(rel_time)
            if step < 0 or step >= len(labels):
                continue
            label_rows.append(
                {
                    "event_id": event["event_id"],
                    "rel_time": int(rel_time),
                    "label": int(labels[step]),
                    "delta": float(deltas[step]),
                }
            )
    label_frame = pd.DataFrame(label_rows)
    lines = [
        "# S4 phase-propagation analysis report",
        "",
        "## Purpose",
        "",
        "This derived analysis tests whether the deformation-off release event locally perturbs body phase/deformation and whether that perturbation shows rank-ordered propagation away from the release frontier. It also asks whether local phase-response variables predict sustained direction of movement better in body streams than in hidden/internal or action-control streams.",
        "",
        "## Source packet",
        "",
        f"- Package root: `{PACKAGE_ROOT}`",
        f"- Specimen: `{SPECIMEN_ID}`",
        f"- Modal profile: `{GPU_PROFILE}`",
        "- Condition: deformation off, intact signalling",
        f"- Cliff distances: `{CLIFF_DISTANCES[0]}..{CLIFF_DISTANCES[-1]}`",
        f"- Release events: `{len(events)}/{len(CLIFF_DISTANCES)}`",
        f"- Returned from edge: `{int(event_frame['returned_from_edge'].sum())}/{len(events)}`",
        f"- Success: `{int(event_frame['success'].sum())}/{len(events)}`",
        f"- Falls: `{int(event_frame['cliff_fell'].sum())}/{len(events)}`",
        f"- Release-step range: `{int(event_frame['release_step'].min())}` to `{int(event_frame['release_step'].max())}`",
        "",
        "## Pre-release identity control",
        "",
    ]
    identity_path = PACKAGE_ROOT / "tables" / "pre_release_identity_summary.csv"
    if identity_path.exists():
        identity = pd.read_csv(identity_path)
        exact_all = bool((identity["exact_fraction"].to_numpy(dtype=np.float64) == 1.0).all()) if not identity.empty else False
        lines.append(f"- All streams exact across inspected pre-release pairwise prefixes: `{exact_all}`.")
        for _, row in identity.iterrows():
            lines.append(
                "- "
                f"`{row['stream']}`: exact {int(row['exact_pair_count'])}/{int(row['pair_count'])}; "
                f"max_abs_max {float(row['max_abs_max']):.6g}; rms_max {float(row['rms_max']):.6g}."
            )
    else:
        lines.append("- `pre_release_identity_summary.csv` was not present in this packet at report time.")
    lines.extend(
        [
            "",
            "This matters because the release-aligned comparisons are not comparing different pre-release controllers or bodies. The traces are identical until the earlier cliff-frontier release; the post-release differences are therefore event-locked to loss of support under this fixed-width family.",
            "",
        "## Direction-label inventory",
        "",
        ]
    )
    if label_frame.empty:
        lines.append("- No direction labels could be computed.")
    else:
        for rel_time in (-64, 0, 64, 128, 256, 384, 512):
            rel = label_frame[label_frame["rel_time"] == rel_time]
            if rel.empty:
                continue
            away = int(np.sum(rel["label"].to_numpy(dtype=np.int8) == 1))
            toward = int(np.sum(rel["label"].to_numpy(dtype=np.int8) == -1))
            uncommitted = int(np.sum(rel["label"].to_numpy(dtype=np.int8) == 0))
            lines.append(
                f"- t={rel_time:+d}: away `{away}`, toward `{toward}`, uncommitted `{uncommitted}`."
            )
    lines.extend(_phase_propagation_report_lines(propagation_paths))
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `tables/phase_propagation_reference_summary.csv`",
            "- `tables/phase_propagation_local_slip_by_event.csv`",
            "- `tables/phase_propagation_local_slip.csv`",
            "- `tables/phase_propagation_direction_decoder.csv`",
            "- `tables/phase_propagation_onsets.csv`",
            "- `tables/phase_propagation_onset_regression.csv`",
            "- `tables/phase_propagation_summary.json`",
            "- `figures/phase_propagation_local_slip_heatmaps.pdf`",
            "- `figures/phase_propagation_direction_auc_heatmaps.pdf`",
            "- `figures/phase_propagation_onsets.pdf`",
            "- `figures/phase_propagation_local_slip.gif`",
            "",
            "## Paper-use decision",
            "",
            "Use this as a propagation figure only if the onset-regression gate passes. Otherwise, keep it as an analysis packet supporting the narrower statement that the body-state phase response can be inspected locally and compared with hidden/action controls around release.",
        ]
    )
    path = PACKAGE_ROOT / "S4_phase_propagation_report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def analyze_phase_propagation_only(*, random_seed: int = 137, init_package: bool = False) -> dict[str, Path]:
    summarize(init_package=init_package)
    events, _, _ = _load_events()
    if not events:
        raise ValueError("No true release events available for phase-propagation analysis.")
    figures_dir = PACKAGE_ROOT / "figures"
    propagation_paths = _run_phase_propagation_analysis(events, figures_dir, random_seed=random_seed)
    propagation_report = _write_phase_propagation_report(events, propagation_paths)
    payload = {
        "schema_version": 1,
        "kind": "s4_fixed_width_phase_propagation_analysis",
        "event_count": len(events),
        "streams": list(PROPAGATION_STREAMS),
        "figure_streams": list(PROPAGATION_FIGURE_STREAMS),
        "relative_times": list(PROPAGATION_REL_TIMES),
        "fit_stride": PHASE_FIT_STRIDE,
        "continuer_margin_steps": CONTINUER_MARGIN_STEPS,
        "permutations": PROPAGATION_PERMUTATIONS,
        "classifier": "leave-one-event-out ridge without intercept for the phase-response direction decoder",
        "feature_dust_floor": PROPAGATION_FEATURE_DUST_FLOOR,
        "direction_label": {
            "positive": "sustained future COM motion away from cliff",
            "negative": "sustained future COM motion toward cliff",
            "future_window": PROPAGATION_DIRECTION_FUTURE_WINDOW,
            "min_displacement": PROPAGATION_DIRECTION_MIN_DISPLACEMENT,
            "persistence_steps": PROPAGATION_DIRECTION_PERSISTENCE_STEPS,
            "persistence_fraction": PROPAGATION_DIRECTION_PERSISTENCE_FRACTION,
        },
        "phase_onset_threshold": {
            "method": "max(pre-release mean_abs_phase_slip q95/mean+sd rule, fixed minimum)",
            "minimum_radians": PROPAGATION_PHASE_ONSET_MIN_RAD,
            "sd_multiplier": PROPAGATION_PHASE_ONSET_SD_MULTIPLIER,
        },
        "paths": {key: str(value) for key, value in propagation_paths.items()},
        "phase_propagation_report": str(propagation_report),
    }
    _write_json(PACKAGE_ROOT / "tables" / "phase_propagation_summary.json", payload)
    outputs = {
        "phase_propagation_summary": PACKAGE_ROOT / "tables" / "phase_propagation_summary.json",
        "phase_propagation_report": propagation_report,
    }
    outputs.update(propagation_paths)
    return outputs


def analyze(*, random_seed: int = 137, init_package: bool = True) -> dict[str, Path]:
    summarize(init_package=init_package)
    events, robot_shape, hidden_start = _load_events()
    if not events:
        raise ValueError("No true release events available for analysis.")
    _clean_derived_outputs()

    event_rows = [
        {
            "event_id": event["event_id"],
            "trial_id": event["trial_id"],
            "goal_distance": event["goal_distance"],
            "cliff_distance": event["cliff_distance"],
            "release_step": event["release_step"],
            "steps_alive": event["steps_alive"],
            "success": event["success"],
            "returned_from_edge": event["returned_from_edge"],
            "cliff_fell": event["cliff_fell"],
            "goal_touched": event["goal_touched"],
            "termination_reason": event["termination_reason"],
            "cliff_side": event["cliff_side"],
            "trace_path": event["trace_path"],
        }
        for event in events
    ]
    _write_csv(PACKAGE_ROOT / "tables" / "release_event_manifest.csv", event_rows)

    identity_rows, identity_summary = pre_release_identity(events)
    _write_csv(PACKAGE_ROOT / "tables" / "pre_release_identity_pairwise.csv", identity_rows)
    _write_csv(PACKAGE_ROOT / "tables" / "pre_release_identity_summary.csv", identity_summary)

    decoder_rows = _decoder_rows(events, random_seed=_stable_seed(random_seed, "decoder"))
    _write_csv(PACKAGE_ROOT / "tables" / "matched_release_decoder_map.csv", decoder_rows)
    perturbation_rows = _matched_perturbation_rows(events)
    _write_csv(PACKAGE_ROOT / "tables" / "matched_release_perturbation_map.csv", perturbation_rows)
    onset, regression = _onset_rows(decoder_rows)
    _write_csv(PACKAGE_ROOT / "tables" / "matched_release_decoder_onsets.csv", onset)
    _write_csv(PACKAGE_ROOT / "tables" / "matched_release_onset_regression.csv", regression)

    figures_dir = PACKAGE_ROOT / "figures"
    _plot_release_times(events, figures_dir / "release_time_vs_cliff_distance.png")
    _plot_identity_summary(identity_summary, figures_dir / "pre_release_identity_summary.png")
    decoder_frame = pd.DataFrame(decoder_rows)
    for stream in FOCUS_STREAMS:
        _plot_heatmap(
            decoder_frame[decoder_frame["stream"] == stream],
            value_col="auc",
            title=f"S4 {stream} released-vs-continuing AUC",
            path=figures_dir / f"matched_release_auc_{stream}.png",
            vmin=0.5,
            vmax=1.0,
        )
    perturbation_frame = pd.DataFrame(perturbation_rows)
    for stream in FOCUS_STREAMS:
        stream_frame = perturbation_frame[perturbation_frame["stream"] == stream]
        finite = stream_frame["mean_l2_delta"].to_numpy(dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        vmax = float(np.quantile(finite, 0.98)) if finite.size else None
        _plot_heatmap(
            stream_frame,
            value_col="mean_l2_delta",
            title=f"S4 {stream} released-vs-continuing deviation",
            path=figures_dir / f"matched_release_perturbation_{stream}.png",
            vmin=0.0,
            vmax=vmax,
            cmap="magma",
        )
    _plot_perturbation_gif(
        perturbation_rows,
        figures_dir / "matched_release_perturbation_split_streams.gif",
    )
    phase_paths = _run_phase_reset_analysis(events, figures_dir, random_seed=random_seed)

    all_identity_exact = all(float(row["exact_fraction"]) == 1.0 for row in identity_summary)
    summary = {
        "schema_version": 1,
        "kind": "s4_fixed_width_distance_sweep_analysis",
        "event_count": len(events),
        "target_event_count": len(CLIFF_DISTANCES),
        "robot_shape": robot_shape,
        "hidden_start_channel": hidden_start,
        "all_pre_release_streams_exact": bool(all_identity_exact),
        "decoder": {
            "streams": list(PREDICTIVE_STREAMS),
            "relative_times": list(REL_TIMES),
            "continuer_margin_steps": CONTINUER_MARGIN_STEPS,
            "ridge_alpha": RIDGE_ALPHA,
            "permutations": PERMUTATIONS,
            "cross_validation": "leave source trace out",
            "label_policy": "positive=release-aligned trace at release+relative time; negative=farther trace at same absolute time whose release is still beyond margin",
        },
        "paths": {
            "release_event_manifest": PACKAGE_ROOT / "tables" / "release_event_manifest.csv",
            "pre_release_identity_pairwise": PACKAGE_ROOT / "tables" / "pre_release_identity_pairwise.csv",
            "pre_release_identity_summary": PACKAGE_ROOT / "tables" / "pre_release_identity_summary.csv",
            "matched_release_decoder_map": PACKAGE_ROOT / "tables" / "matched_release_decoder_map.csv",
            "matched_release_perturbation_map": PACKAGE_ROOT / "tables" / "matched_release_perturbation_map.csv",
            "matched_release_decoder_onsets": PACKAGE_ROOT / "tables" / "matched_release_decoder_onsets.csv",
            "matched_release_onset_regression": PACKAGE_ROOT / "tables" / "matched_release_onset_regression.csv",
            "phase_reset_reference_summary": PACKAGE_ROOT / "tables" / "phase_reset_reference_summary.csv",
            "phase_reset_slip_by_event": PACKAGE_ROOT / "tables" / "phase_reset_slip_by_event.csv",
            "phase_reset_slip_summary": PACKAGE_ROOT / "tables" / "phase_reset_slip_summary.csv",
            "phase_reset_events": PACKAGE_ROOT / "tables" / "phase_reset_events.csv",
            "phase_reset_release_phase_correlations": PACKAGE_ROOT / "tables" / "phase_reset_release_phase_correlations.csv",
            "figures": figures_dir,
        },
    }
    _write_json(PACKAGE_ROOT / "tables" / "analysis_summary.json", summary)
    _write_report(events, identity_summary, decoder_rows, perturbation_rows, regression, summary, phase_paths)
    outputs = {
        "analysis_summary": PACKAGE_ROOT / "tables" / "analysis_summary.json",
        "report": PACKAGE_ROOT / "S4_fixed_width_distance_sweep_report.md",
    }
    outputs.update(phase_paths)
    return outputs


def _focus_auc_rows(decoder_rows: list[dict[str, Any]]) -> list[str]:
    frame = pd.DataFrame(decoder_rows)
    lines: list[str] = []
    for stream, group in frame.groupby("stream", sort=False):
        post = group[(group["rel_time"] >= 0) & (group["rel_time"] <= 128) & group["estimable"]]
        if post.empty:
            lines.append(f"- `{stream}`: no estimable post-release rows.")
            continue
        best = post.sort_values("auc", ascending=False, kind="stable").iloc[0]
        sig = post[(post["auc"] > post["null_auc_q95"]) & (post["one_sided_p"] <= 0.05)]
        lines.append(
            "- "
            f"`{stream}`: best post-release AUC {float(best['auc']):.3f} "
            f"at rank {int(best['rank_from_edge'])}, rel_time {int(best['rel_time'])}; "
            f"significant rows {len(sig)}/{len(post)}."
        )
    return lines


def _phase_report_lines(phase_paths: dict[str, Path]) -> list[str]:
    if not phase_paths:
        return [
            "",
            "## Phase-reset analysis",
            "",
            "Phase-reset outputs were not generated.",
        ]
    reference_path = phase_paths.get("phase_reset_reference_summary")
    summary_path = phase_paths.get("phase_reset_slip_summary")
    correlations_path = phase_paths.get("phase_reset_release_phase_correlations")
    if reference_path is None or summary_path is None or not Path(reference_path).exists() or not Path(summary_path).exists():
        return [
            "",
            "## Phase-reset analysis",
            "",
            "Phase-reset output paths were registered, but the expected summary tables were not found.",
        ]

    reference = pd.read_csv(reference_path)
    slip = pd.read_csv(summary_path)
    correlations = pd.read_csv(correlations_path) if correlations_path is not None and Path(correlations_path).exists() else pd.DataFrame()
    lines = [
        "",
        "## Phase-reset analysis",
        "",
        "- Phase coordinates are fitted separately for each stream by PCA on the longest-distance trace before release.",
        "- Whole-body phase vectors use fixed cell identity order, not rank-from-edge order, so phase comparisons do not depend on release-time spatial reordering.",
        f"- Fit sampling stride: `{PHASE_FIT_STRIDE}` steps.",
        "- Release phase is sampled at `release_step - 1`, before the loss-of-support frame.",
        f"- Phase slip compares the released trace with still-continuing traces at the same absolute timestep, requiring the continuer's release to be at least `{CONTINUER_MARGIN_STEPS}` steps later.",
        "- This analysis asks whether the cliff release interrupts an otherwise identical locomotor/controller rhythm; it is distinct from the spatial onset-regression test above.",
        "",
        "Stream summaries:",
    ]
    for stream in PHASE_FIGURE_STREAMS:
        ref_row = reference[reference["stream"] == stream]
        stream_slip = slip[slip["stream"] == stream]
        if ref_row.empty or stream_slip.empty:
            lines.append(f"- `{stream}`: no phase-reset summary available.")
            continue
        ref = ref_row.iloc[0]
        at_128 = stream_slip[stream_slip["rel_time"] == 128]
        at_336 = stream_slip[stream_slip["rel_time"] == 336]
        max_row = stream_slip.sort_values("mean_abs_phase_slip", ascending=False, kind="stable").iloc[0]
        pc12 = float(ref["explained_var_pc1_pc2"])
        text = (
            f"- `{stream}`: PC1+PC2 explain {pc12:.3f}; "
            f"max mean |phase slip| {float(max_row['mean_abs_phase_slip']):.3f} rad "
            f"at +{int(max_row['rel_time'])} steps"
        )
        if not at_128.empty:
            text += f"; +128 mean |slip| {float(at_128.iloc[0]['mean_abs_phase_slip']):.3f} rad"
        if not at_336.empty:
            text += f"; +336 mean |slip| {float(at_336.iloc[0]['mean_abs_phase_slip']):.3f} rad"
        lines.append(text + ".")

    if not correlations.empty:
        lines.extend(["", "Release-phase predictive checks:"])
        for metric in ("mean_abs_phase_slip_0_128", "steps_after_release"):
            metric_rows = correlations[correlations["metric"] == metric].copy()
            if metric_rows.empty:
                continue
            metric_rows = metric_rows.sort_values("phase_r2", ascending=False, kind="stable")
            best = metric_rows.iloc[0]
            lines.append(
                "- "
                f"`{metric}`: strongest stream `{best['stream']}` with circular phase R2 "
                f"{float(best['phase_r2']):.3f}, null q95 {float(best['null_r2_q95']):.3f}, "
                f"p {float(best['one_sided_p']):.4f}."
            )
    lines.extend(
        [
            "",
            "Phase-reset interpretation: this is the cleaner test for a phase-interruption account. The relevant positive result is not a clean rearward onset slope, but a fixed-cell dissociation: hidden/action streams remain effectively phase-locked after release, while plant streams, especially kinematics and body-without-contraction, accumulate post-release phase slip. Release phase should not be claimed to predict outcome unless the circular phase checks exceed their permutation null.",
        ]
    )
    return lines


def _write_report(
    events: list[dict[str, Any]],
    identity_summary: list[dict[str, Any]],
    decoder_rows: list[dict[str, Any]],
    perturbation_rows: list[dict[str, Any]],
    regression_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    phase_paths: dict[str, Path],
) -> None:
    event_frame = pd.DataFrame(
        [
            {
                "cliff_distance": event["cliff_distance"],
                "release_step": event["release_step"],
                "success": event["success"],
                "returned_from_edge": event["returned_from_edge"],
                "cliff_fell": event["cliff_fell"],
            }
            for event in events
        ]
    )
    identity_frame = pd.DataFrame(identity_summary)
    all_exact = bool(summary["all_pre_release_streams_exact"])
    lines: list[str] = [
        "# S4 fixed-width distance sweep report",
        "",
        "## Purpose",
        "",
        "This packet replaces the plateau-width event-multiplication probe for the main S4 mechanism analysis. The earlier plateau sweep produced a useful 24-trace matched family, but plateau width also changed static collision worklist shape. This packet holds width fixed and varies only cliff distance.",
        "",
        "## Run setup",
        "",
        f"- Package root: `{PACKAGE_ROOT}`",
        f"- Specimen: `{SPECIMEN_ID}`",
        f"- Modal profile: `{GPU_PROFILE}`",
        f"- Assay: `{ASSAY}`",
        "- Condition: deformation off, intact signalling",
        f"- Grid width: `{GRID_W}`",
        f"- Plateau/base width: `{PLATEAU_WIDTH}`",
        f"- Goal side: `{GOAL_SIDE}`",
        f"- Goal distance: `{GOAL_DISTANCE}`",
        f"- Cliff distances: `{CLIFF_DISTANCES[0]}..{CLIFF_DISTANCES[-1]}`",
        f"- Max steps: `{MAX_STEPS}`",
        "",
        "## Event yield",
        "",
        f"- Release events: `{len(events)}/{len(CLIFF_DISTANCES)}`",
        f"- Returned from edge: `{int(event_frame['returned_from_edge'].sum())}/{len(events)}`",
        f"- Success: `{int(event_frame['success'].sum())}/{len(events)}`",
        f"- Falls: `{int(event_frame['cliff_fell'].sum())}/{len(events)}`",
        f"- Release-step range: `{int(event_frame['release_step'].min())}` to `{int(event_frame['release_step'].max())}`",
        "",
        "## Pre-release identity",
        "",
        f"- All inspected pre-release streams exact across pairs: `{all_exact}`",
    ]
    for _, row in identity_frame.iterrows():
        lines.append(
            "- "
            f"`{row['stream']}`: exact {int(row['exact_pair_count'])}/{int(row['pair_count'])}; "
            f"max_abs_max {float(row['max_abs_max']):.6g}; rms_max {float(row['rms_max']):.6g}."
        )
    lines.extend(
        [
            "",
            "Interpretation gate: if pre-release identity is not exact, the released-vs-continuing decoder remains informative but should not be described as a perfectly matched open-loop interruption.",
            "",
            "## Matched released-vs-continuing decoder",
            "",
            f"- Relative-time grid: `{REL_TIMES[0]}..{REL_TIMES[-1]}` in 8-step increments.",
            f"- Negative controls require the continuing trace's release to be at least `{CONTINUER_MARGIN_STEPS}` steps in the future.",
            "- Cross-validation leaves out all samples from one source trace at a time.",
            "- Features exclude COM, terrain coordinates, cliff/goal positions, and global x.",
            "- Negative relative-time decoder rows are a rhythm/time diagnostic, not an anticipation result: the exact-identity table is the pre-release control. Because negative rows are sampled relative to each event's release, they mix different absolute times and locomotor phases across cliff distances.",
            "- `signal_hidden`: NCA hidden channels plus CTRNN carried state; excludes deformation channels and action/contraction output channels.",
            "- `controller_action`: local NCA action/contraction outputs, including phase-level action channels.",
            "- `body_kinematic`: local cell-corner geometry relative to the cell center plus local point velocities.",
            "- `spring_current`: current attached-spring length state, normalized to initial rest length.",
            "- `spring_contractile`: rest-length contraction state (`spring_rest_length`, `spring_pre_rest_length`, and phase post-rest lengths), normalized to initial rest length.",
            "- `body_no_contractile`: spring-current state plus kinematic state.",
            "- `body_combined`: legacy combined plant stream: spring-current, spring-contractile, and kinematic state.",
            "",
        ]
    )
    lines.extend(_focus_auc_rows(decoder_rows))
    lines.extend(["", "## Onset regression", ""])
    for row in regression_rows:
        lines.append(
            "- "
            f"`{row['stream']}`: onset ranks {int(row['rank_count_with_onset'])}; "
            f"slope {float(row['slope_steps_per_rank']):.3f} steps/rank; "
            f"R2 {float(row['r2']):.3f}."
        )
    lines.extend(
        [
            "",
            "Onset-regression interpretation: rank 0 is the cliff-nearest/frontier rank. A positive slope would support delayed rearward propagation. The current body slope is negative, so the onset regression should not be described as a clean front-to-back wave; the stronger supported result is post-release body-state decodability exceeding signalling-state decodability after exact pre-release identity.",
        ]
    )
    lines.extend(_phase_report_lines(phase_paths))
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `tables/release_event_manifest.csv`",
            "- `tables/pre_release_identity_pairwise.csv`",
            "- `tables/pre_release_identity_summary.csv`",
            "- `tables/matched_release_decoder_map.csv`",
            "- `tables/matched_release_perturbation_map.csv`",
            "- `tables/matched_release_decoder_onsets.csv`",
            "- `tables/matched_release_onset_regression.csv`",
            "- `tables/phase_reset_reference_summary.csv`",
            "- `tables/phase_reset_slip_by_event.csv`",
            "- `tables/phase_reset_slip_summary.csv`",
            "- `tables/phase_reset_events.csv`",
            "- `tables/phase_reset_release_phase_correlations.csv`",
            "- `figures/release_time_vs_cliff_distance.pdf`",
            "- `figures/pre_release_identity_summary.pdf`",
            "- `figures/matched_release_auc_<stream>.pdf` for split predictive streams",
            "- `figures/matched_release_perturbation_<stream>.pdf` for split predictive streams",
            "- `figures/matched_release_perturbation_split_streams.gif`",
            "- `figures/phase_reset_slip_summary.pdf`",
            "- `figures/phase_reset_loop_<stream>.pdf` for phase streams",
            "- `figures/phase_reset_release_phase_vs_slip.pdf`",
            "",
            "## Paper-use decision",
            "",
        ]
    )
    if all_exact and len(events) >= 25:
        lines.append("This packet is eligible for the main S4 mechanism figure, subject to the body-vs-signal decoder and propagation results being visually and statistically interpretable.")
    else:
        lines.append("This packet should be treated as exploratory unless the pre-release identity issue is resolved or a matched exact-family subset is explicitly selected.")
    (PACKAGE_ROOT / "S4_fixed_width_distance_sweep_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    launch_parser = sub.add_parser("launch-modal")
    launch_parser.add_argument("--dry-run", action="store_true")
    pull_parser = sub.add_parser("pull-modal")
    pull_parser.add_argument("--dry-run", action="store_true")
    sub.add_parser("summarize")
    sub.add_parser("audit")
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--random-seed", type=int, default=137)
    phase_parser = sub.add_parser("phase-reset")
    phase_parser.add_argument("--random-seed", type=int, default=137)
    propagation_parser = sub.add_parser("phase-propagation")
    propagation_parser.add_argument("--random-seed", type=int, default=137)
    graph_parser = sub.add_parser("graph-propagation")
    graph_parser.add_argument("--random-seed", type=int, default=137)
    pca_parser = sub.add_parser("pca-regime-entry")
    pca_parser.add_argument("--random-seed", type=int, default=137)
    remaining_parser = sub.add_parser("remaining-mechanism")
    remaining_parser.add_argument("--random-seed", type=int, default=137)
    args = parser.parse_args()

    if args.command == "init":
        write_launch_manifest()
        print(PACKAGE_ROOT)
    elif args.command == "launch-modal":
        launch_modal(dry_run=bool(args.dry_run))
    elif args.command == "pull-modal":
        pull_modal(dry_run=bool(args.dry_run))
    elif args.command == "summarize":
        for key, value in summarize().items():
            print(f"{key}: {value}")
    elif args.command == "audit":
        for key, value in audit_remote_data().items():
            print(f"{key}: {value}")
    elif args.command == "analyze":
        for key, value in analyze(random_seed=int(args.random_seed)).items():
            print(f"{key}: {value}")
    elif args.command == "phase-reset":
        for key, value in analyze_phase_reset_only(random_seed=int(args.random_seed)).items():
            print(f"{key}: {value}")
    elif args.command == "phase-propagation":
        for key, value in analyze_phase_propagation_only(random_seed=int(args.random_seed)).items():
            print(f"{key}: {value}")
    elif args.command == "graph-propagation":
        for key, value in analyze_graph_propagation_only(random_seed=int(args.random_seed)).items():
            print(f"{key}: {value}")
    elif args.command == "pca-regime-entry":
        for key, value in analyze_pca_regime_entry_only(random_seed=int(args.random_seed)).items():
            print(f"{key}: {value}")
    elif args.command == "remaining-mechanism":
        for key, value in analyze_remaining_mechanism(random_seed=int(args.random_seed)).items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
