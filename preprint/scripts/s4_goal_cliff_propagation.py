#!/usr/bin/env python
"""Run and analyze the S4 goal-cliff propagation expansion.

This is intentionally self-contained. It launches the existing Modal post-run
state-space assay with overrides, stores all artifacts under one preprint
package, and analyzes only the resulting deformation-off goal-cliff traces.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
SENSING_EDGES_ROOT = REPO_ROOT / "sensing_edges"
for path in (SCRIPT_DIR, SENSING_EDGES_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from committed_direction_decoding import _auc, _committed_labels, _ridge_scores, _stable_seed
from leading_edge_reversal_analysis import _leading_edge_indices, _trace_path_for_trial
from voxel_reversal_heatmap_analysis import _cell_spring_indices


PACKAGE_ROOT = REPO_ROOT / "paper_data" / "preprint" / "s4_goal_cliff_propagation_20260614"
REMOTE_ROOT = "/outputs/preprint_s4_goal_cliff_propagation_20260614"
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
GRID_W = 40
MAX_STEPS = 12_000
PAIRS = ((2, 1), (4, 2), (6, 3), (8, 4), (10, 5), (12, 6))
PLATEAU_WIDTHS = (8, 10, 12, 14, 16)
REL_TIMES = tuple(range(-64, 337, 8))
MECH_REL_TIMES = tuple(range(-64, 337, 8))
DECODER_LAG = 8
FUTURE_WINDOW = 32
MIN_DISPLACEMENT = 0.025
PERSISTENCE_STEPS = 12
PERSISTENCE_FRACTION = 0.75
RIDGE_ALPHA = 10.0
PERMUTATIONS = 1000


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
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


def _plateau_id(plateau_width: int) -> str:
    return f"plateau_{int(plateau_width):02d}"


def _remote_analysis_root(plateau_width: int) -> str:
    return f"{REMOTE_ROOT}/{_plateau_id(plateau_width)}"


def _modal_volume_path(plateau_width: int) -> str:
    return _remote_analysis_root(plateau_width).removeprefix("/outputs/")


def _s4_item() -> dict[str, str]:
    import native_core_ablation as native

    return native.state_space_specimen()


def _copy_required_file(source: Path, dest: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Required S4 artifact missing: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def initialize_package() -> Path:
    """Create the package and copy S4 genotype/source context."""
    for rel in ("specimen", "manifests", "modal", "tables", "figures", "notes"):
        (PACKAGE_ROOT / rel).mkdir(parents=True, exist_ok=True)

    for name in ("manifest.json", "selected_specimens.json", "selected_genotypes.npz", "terrain_manifest.json"):
        source = SOURCE_STAGE_ROOT / name
        if source.exists():
            _copy_required_file(source, PACKAGE_STAGE_ROOT / name)

    if not SOURCE_SPECIMEN_DIR.exists():
        raise FileNotFoundError(f"S4 source specimen directory not found: {SOURCE_SPECIMEN_DIR}")
    if PACKAGE_SPECIMEN_DIR.exists():
        shutil.rmtree(PACKAGE_SPECIMEN_DIR)
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
        "kind": "s4_goal_cliff_propagation_package",
        "created_for": "S4 deformation-off goal-cliff propagation expansion",
        "package_root": PACKAGE_ROOT,
        "run_id": RUN_ID,
        "specimen_id": SPECIMEN_ID,
        "gpu_profile": GPU_PROFILE,
        "assay": ASSAY,
        "condition": "deformation_off",
        "goal_side": GOAL_SIDE,
        "grid_w": GRID_W,
        "max_steps": MAX_STEPS,
        "pairs": [{"goal_distance": goal, "cliff_distance": cliff} for goal, cliff in PAIRS],
        "plateau_widths": list(PLATEAU_WIDTHS),
        "expected_config_count": len(PAIRS) * len(PLATEAU_WIDTHS),
        "decoder": {
            "relative_times": list(REL_TIMES),
            "lag": DECODER_LAG,
            "future_window": FUTURE_WINDOW,
            "min_displacement": MIN_DISPLACEMENT,
            "persistence_steps": PERSISTENCE_STEPS,
            "persistence_fraction": PERSISTENCE_FRACTION,
            "ridge_alpha": RIDGE_ALPHA,
            "permutations": PERMUTATIONS,
        },
    }
    _write_json(PACKAGE_ROOT / "manifests" / "package_manifest.json", manifest)
    (PACKAGE_ROOT / "notes" / "README.md").write_text(
        "# S4 goal-cliff propagation expansion\n\n"
        "Canonical Modal A100-40GB deformation-off S4 state-space expansion.\n\n"
        "- Goal-cliff only; no double-cliff traces are included.\n"
        "- Five plateau-width jobs x six right-goal pairs = 30 deterministic configs.\n"
        "- Shared post-run code is used through overrides; this package is isolated under `paper_data/preprint/s4_goal_cliff_propagation_20260614`.\n"
        "- S4 genotype/specimen artifacts are copied under `specimen/` for provenance.\n",
    )
    return PACKAGE_ROOT


def _override_tokens(*, plateau_width: int, analysis_root: str, specimens_output_root: str) -> list[str]:
    return [
        f"eval.post_run.analysis_root={analysis_root}",
        "eval.post_run.state_space_surfaces=[native_core]",
        "eval.post_run.state_space_deformation_condition=deformation_off",
        "eval.post_run.state_space_signaling_condition=intact",
        f"eval.post_run.native_pairs={_omega_pairs(PAIRS)}",
        f"eval.post_run.goal_sides={_omega_list((GOAL_SIDE,))}",
        f"eval.post_run.native_max_steps={MAX_STEPS}",
        f"eval.post_run.max_steps={MAX_STEPS}",
        "eval.post_run.fast_ablation_eval=false",
        f"terrain.grid_w={GRID_W}",
        f"terrain.center_plateau_width={int(plateau_width)}",
        f"terrain.base_width={int(plateau_width)}",
        f"specimens.output_root={specimens_output_root}",
    ]


def modal_launch_argv(plateau_width: int) -> list[str]:
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
                plateau_width=int(plateau_width),
                analysis_root=_remote_analysis_root(int(plateau_width)),
                specimens_output_root=item["specimens_output_root"],
            )
        ),
        "--gpu-profile",
        GPU_PROFILE,
        "--detach-launch",
    ]


def pull_argv(plateau_width: int) -> list[str]:
    return [
        "uv",
        "run",
        "--python",
        ".venv/bin/python",
        "modal",
        "volume",
        "get",
        "se-outputs",
        _modal_volume_path(int(plateau_width)),
        str(PACKAGE_ROOT / "modal" / GPU_PROFILE),
    ]


def write_launch_manifest() -> Path:
    initialize_package()
    jobs = []
    for plateau_width in PLATEAU_WIDTHS:
        launch = modal_launch_argv(int(plateau_width))
        pull = pull_argv(int(plateau_width))
        jobs.append(
            {
                "plateau_width": int(plateau_width),
                "plateau_id": _plateau_id(int(plateau_width)),
                "remote_analysis_root": _remote_analysis_root(int(plateau_width)),
                "modal_volume_path": _modal_volume_path(int(plateau_width)),
                "launch_argv": launch,
                "launch_command": _shell_join(launch),
                "pull_argv": pull,
                "pull_command": _shell_join(pull),
            }
        )
    path = PACKAGE_ROOT / "manifests" / "modal_launch_manifest.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "kind": "s4_goal_cliff_propagation_modal_launch_manifest",
            "jobs": jobs,
            "expected": {
                "jobs": len(jobs),
                "configs_per_job": len(PAIRS),
                "total_configs": len(PAIRS) * len(jobs),
                "trace_npz_per_job": len(PAIRS),
            },
        },
    )
    return path


def _load_launch_manifest() -> dict[str, Any]:
    path = write_launch_manifest()
    with path.open() as handle:
        return json.load(handle)


def launch_modal(*, dry_run: bool = False) -> None:
    manifest = _load_launch_manifest()
    for job in manifest["jobs"]:
        print(job["launch_command"])
        if not dry_run:
            subprocess.run(job["launch_argv"], cwd=REPO_ROOT, check=True)


def pull_modal(*, dry_run: bool = False) -> None:
    manifest = _load_launch_manifest()
    for job in manifest["jobs"]:
        print(job["pull_command"])
        if not dry_run:
            dest = Path(job["pull_argv"][-1]) / str(job["plateau_id"])
            if dest.exists():
                shutil.rmtree(dest)
            subprocess.run(job["pull_argv"], cwd=REPO_ROOT, check=True)


def _plateau_from_path(path: Path) -> int | None:
    for part in path.parts:
        if part.startswith("plateau_"):
            try:
                return int(part.split("_", 1)[1])
            except ValueError:
                return None
    return None


def _find_assay_roots() -> list[Path]:
    root = PACKAGE_ROOT / "modal" / GPU_PROFILE
    return sorted(path.parent for path in root.glob(f"**/{ASSAY}/trial_manifest.csv"))


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


def summarize() -> dict[str, Path]:
    initialize_package()
    trial_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    for assay_root in _find_assay_roots():
        plateau_width = _plateau_from_path(assay_root)
        trial_df = pd.read_csv(assay_root / "trial_manifest.csv")
        traces_dir = assay_root / "traces"
        trace_count = len(list(traces_dir.glob("*.npz"))) if traces_dir.exists() else 0
        reached = int(trial_df["first_cliff_reach_step"].notna().sum()) if "first_cliff_reach_step" in trial_df else 0
        returned = int(trial_df["returned_from_edge"].map(_bool_value).sum()) if "returned_from_edge" in trial_df else 0
        inventory_rows.append(
            {
                "plateau_width": plateau_width,
                "assay_root": str(assay_root),
                "trial_count": int(len(trial_df)),
                "expected_trial_count": len(PAIRS),
                "trace_count": trace_count,
                "expected_trace_count": len(PAIRS),
                "edge_reach_count": reached,
                "returned_from_edge_count": returned,
                "complete": int(len(trial_df) == len(PAIRS) and trace_count == len(PAIRS)),
            }
        )
        for _, row in trial_df.iterrows():
            record = row.to_dict()
            record["plateau_width"] = plateau_width
            record["assay_root"] = str(assay_root)
            record["event_id"] = f"pw{plateau_width}_{record.get('trial_id')}"
            record["true_release_event"] = bool(pd.notna(record.get("first_cliff_reach_step")))
            trial_rows.append(record)

    trial_path = PACKAGE_ROOT / "tables" / "trial_manifest_combined.csv"
    inventory_path = PACKAGE_ROOT / "tables" / "trace_inventory.csv"
    pd.DataFrame(trial_rows).to_csv(trial_path, index=False)
    pd.DataFrame(inventory_rows).to_csv(inventory_path, index=False)
    event_count = int(sum(1 for row in trial_rows if row.get("true_release_event")))
    returned_count = int(sum(1 for row in trial_rows if _bool_value(row.get("returned_from_edge"))))
    note = (
        "# S4 goal-cliff propagation status\n\n"
        f"Assay roots found: {len(inventory_rows)} / {len(PLATEAU_WIDTHS)}\n\n"
        f"Trials found: {len(trial_rows)} / {len(PAIRS) * len(PLATEAU_WIDTHS)}\n\n"
        f"True release events: {event_count}\n\n"
        f"Returned-from-edge events: {returned_count}\n\n"
        "Tables:\n"
        f"- `{trial_path}`\n"
        f"- `{inventory_path}`\n"
    )
    status_path = PACKAGE_ROOT / "notes" / "status.md"
    status_path.write_text(note)
    return {"trial_manifest": trial_path, "trace_inventory": inventory_path, "status": status_path}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path)
    try:
        return {key: loaded[key] for key in loaded.files}
    finally:
        loaded.close()


def _signal_cell_features(
    trace: dict[str, np.ndarray],
    *,
    cell_index: int,
    hidden_start_channel: int,
    robot_shape: tuple[int, int],
) -> np.ndarray:
    rows, cols = robot_shape
    row = int(cell_index) // int(cols)
    col = int(cell_index) % int(cols)
    nca_state = np.asarray(trace["phase_nca_state"][:, :, row, col, int(hidden_start_channel) :], dtype=np.float32)
    ctrnn_carry = np.asarray(trace["phase_ctrnn_carry"][:, :, row, col, :], dtype=np.float32)
    return np.concatenate(
        [nca_state.reshape(nca_state.shape[0], -1), ctrnn_carry.reshape(ctrnn_carry.shape[0], -1)],
        axis=1,
    ).astype(np.float32)


def _safe_scale(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float32)
    reference = np.where(np.abs(reference) > 1.0e-6, reference, 1.0).astype(np.float32)
    return np.asarray(values, dtype=np.float32) / reference - 1.0


def _spring_features(trace: dict[str, np.ndarray], spring_indices: np.ndarray) -> np.ndarray:
    if spring_indices.size == 0:
        return np.zeros((int(trace["spring_current_length"].shape[0]), 0), dtype=np.float32)
    init = np.asarray(trace["spring_init_rest_length"], dtype=np.float32)[spring_indices]
    current = _safe_scale(trace["spring_current_length"][:, spring_indices], init)
    rest = _safe_scale(trace["spring_rest_length"][:, spring_indices], init)
    pre_rest = _safe_scale(trace["spring_pre_rest_length"][:, spring_indices], init)
    phase_rest = _safe_scale(trace["phase_post_spring_rest_length"][:, :, spring_indices], init)
    return np.concatenate([current, rest, pre_rest, phase_rest.reshape(phase_rest.shape[0], -1)], axis=1).astype(np.float32)


def _body_cell_features(
    trace: dict[str, np.ndarray],
    *,
    cell_index: int,
    inward_sign: float,
) -> np.ndarray:
    springs = _cell_spring_indices(trace, int(cell_index))
    spring_block = _spring_features(trace, springs["attached_springs"])
    points = np.asarray(springs["points"], dtype=np.int32)
    corners = np.asarray(trace["cell_corner_positions"][:, [int(cell_index)]], dtype=np.float32)
    center = corners.mean(axis=(1, 2), keepdims=True)
    rel_corners = corners - center
    rel_corners[..., 0] *= float(inward_sign)
    point_vel = np.asarray(trace["plant_velocities_true"][:, points], dtype=np.float32)
    point_vel = point_vel - point_vel.mean(axis=1, keepdims=True)
    point_vel[..., 0] *= float(inward_sign)
    h_drive = np.asarray(trace["phase_h_drive"][:, :, [int(cell_index)]], dtype=np.float32)
    v_drive = np.asarray(trace["phase_v_drive"][:, :, [int(cell_index)]], dtype=np.float32)
    blocks = [
        spring_block,
        rel_corners.reshape(rel_corners.shape[0], -1),
        point_vel.reshape(point_vel.shape[0], -1),
        h_drive.reshape(h_drive.shape[0], -1),
        v_drive.reshape(v_drive.shape[0], -1),
    ]
    return np.concatenate(blocks, axis=1).astype(np.float32)


def _cell_tension_and_speed(trace: dict[str, np.ndarray], *, cell_index: int) -> tuple[np.ndarray, np.ndarray]:
    springs = _cell_spring_indices(trace, int(cell_index))
    points = np.asarray(springs["points"], dtype=np.int32)
    attached = np.asarray(springs["attached_springs"], dtype=np.int32)
    if attached.size:
        init = np.asarray(trace["spring_init_rest_length"], dtype=np.float32)[attached]
        strain = _safe_scale(trace["spring_current_length"][:, attached], init)
        tension = np.sqrt(np.mean(np.square(strain), axis=1)).astype(np.float32)
    else:
        tension = np.zeros((int(trace["spring_current_length"].shape[0]),), dtype=np.float32)
    velocities = np.asarray(trace["plant_velocities_true"][:, points], dtype=np.float32)
    speed = np.sqrt(np.mean(np.sum(np.square(velocities), axis=2), axis=1)).astype(np.float32)
    return tension, speed


def _event_cell_ranks(trace: dict[str, np.ndarray], event_index: int, edge_side: str) -> dict[int, int]:
    centers = np.asarray(trace["cell_corner_positions"][int(event_index)], dtype=np.float32).mean(axis=1)
    valid = np.flatnonzero(np.asarray(trace["boxel_mask"], dtype=bool))
    ordered = sorted(valid.tolist(), key=lambda cell: float(centers[int(cell), 0]), reverse=str(edge_side) == "right")
    return {int(cell): int(rank) for rank, cell in enumerate(ordered)}


def _load_events() -> tuple[list[dict[str, Any]], tuple[int, int], int]:
    events: list[dict[str, Any]] = []
    hidden_start = 6
    robot_shape: tuple[int, int] | None = None
    for assay_root in _find_assay_roots():
        plateau_width = _plateau_from_path(assay_root)
        manifest = json.loads((assay_root / "manifest.json").read_text())
        hidden_start = int((manifest.get("nca_channel_schema") or {}).get("hidden_start_channel", hidden_start))
        trial_manifest = pd.read_csv(assay_root / "trial_manifest.csv")
        step_manifest = pd.read_csv(assay_root / "step_manifest.csv", low_memory=False)
        with np.load(assay_root / "static_arrays.npz") as static:
            robot_shape = tuple(int(v) for v in static["robot_morph"].shape)
        trials = trial_manifest[
            trial_manifest["first_cliff_reach_step"].notna()
            & (trial_manifest["goal_direction"] != 0)
            & (trial_manifest["side"].astype(str) == GOAL_SIDE)
        ].copy()
        trials = trials.sort_values(["goal_distance", "cliff_distance"], kind="stable")
        for _, trial_row in trials.iterrows():
            trial_id = str(trial_row["trial_id"])
            rows = step_manifest[step_manifest["trial_id"] == trial_id].sort_values("step_index", kind="stable").reset_index(drop=True)
            if rows.empty:
                continue
            event_index = int(trial_row["first_cliff_reach_step"])
            trace = _load_npz(_trace_path_for_trial(assay_root, rows))
            edge = _leading_edge_indices(rows=rows, trace=trace, event_index=event_index)
            labels, future_delta = _committed_labels(
                rows,
                future_window=FUTURE_WINDOW,
                min_displacement=MIN_DISPLACEMENT,
                persistence_steps=PERSISTENCE_STEPS,
                persistence_fraction=PERSISTENCE_FRACTION,
            )
            step_count = min(int(len(rows)), int(len(labels)), int(trace["cell_corner_positions"].shape[0]))
            rows = rows.iloc[:step_count].reset_index(drop=True)
            labels = labels[:step_count]
            trace = {key: value[:step_count] if value.shape[:1] == (trace["cell_corner_positions"].shape[0],) else value for key, value in trace.items()}
            cell_count = int(trace["boxel_corners"].shape[0])
            body_features = {
                cell: _body_cell_features(trace, cell_index=cell, inward_sign=float(edge["inward_sign"]))[:step_count]
                for cell in range(cell_count)
            }
            signal_features = {
                cell: _signal_cell_features(
                    trace,
                    cell_index=cell,
                    hidden_start_channel=hidden_start,
                    robot_shape=robot_shape,
                )[:step_count]
                for cell in range(cell_count)
            }
            tension_speed = {
                cell: tuple(array[:step_count] for array in _cell_tension_and_speed(trace, cell_index=cell))
                for cell in range(cell_count)
            }
            events.append(
                {
                    "event_id": f"pw{plateau_width}_{trial_id}",
                    "plateau_width": int(plateau_width) if plateau_width is not None else None,
                    "trial_id": trial_id,
                    "goal_distance": int(trial_row["goal_distance"]),
                    "cliff_distance": int(trial_row["cliff_distance"]),
                    "event_index": int(event_index),
                    "success": _bool_value(trial_row.get("success")),
                    "returned_from_edge": _bool_value(trial_row.get("returned_from_edge")),
                    "steps_alive": int(trial_row.get("steps_alive", len(rows))),
                    "edge_side": str(edge["edge_side"]),
                    "cell_ranks": _event_cell_ranks(trace, int(event_index), str(edge["edge_side"])),
                    "labels": labels,
                    "future_delta": future_delta[:step_count],
                    "body_features": body_features,
                    "signal_features": signal_features,
                    "tension_speed": tension_speed,
                }
            )
    if robot_shape is None:
        raise FileNotFoundError("No pulled state-space traces found for propagation analysis.")
    return events, robot_shape, hidden_start


def _scores_leave_one_event_out(samples: list[tuple[str, np.ndarray, int]]) -> np.ndarray | None:
    if len(samples) < 4:
        return None
    y = np.asarray([label for _event_id, _x, label in samples], dtype=np.int8)
    if len(np.unique(y)) < 2 or min(np.sum(y == -1), np.sum(y == 1)) < 2:
        return None
    scores = np.full((len(samples),), np.nan, dtype=np.float32)
    for idx in range(len(samples)):
        train = [sample for sample_i, sample in enumerate(samples) if sample_i != idx]
        train_y = np.asarray([label for _event_id, _x, label in train], dtype=np.int8)
        if len(np.unique(train_y)) < 2:
            continue
        train_x = np.vstack([x for _event_id, x, _label in train]).astype(np.float32)
        test_x = np.asarray(samples[idx][1], dtype=np.float32).reshape(1, -1)
        scores[idx] = _ridge_scores(train_x, train_y, test_x, alpha=RIDGE_ALPHA)[0]
    if not np.all(np.isfinite(scores)):
        return None
    return scores


def _decoder_map(events: list[dict[str, Any]], *, stream: str, cell_count: int, random_seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(random_seed))
    cell_ranks = _mean_cell_ranks(events, cell_count)
    rows: list[dict[str, Any]] = []
    for cell_index in range(cell_count):
        for rel_time in REL_TIMES:
            samples: list[tuple[str, np.ndarray, int]] = []
            for event in events:
                feature_index = int(event["event_index"]) + int(rel_time)
                label_index = feature_index + DECODER_LAG
                if feature_index < 0 or label_index < 0:
                    continue
                features = event[f"{stream}_features"][cell_index]
                labels = event["labels"]
                if feature_index >= features.shape[0] or label_index >= len(labels):
                    continue
                label = int(labels[label_index])
                if label == 0 or not np.all(np.isfinite(features[feature_index])):
                    continue
                samples.append((str(event["event_id"]), features[feature_index], label))
            y = np.asarray([sample[2] for sample in samples], dtype=np.int8)
            scores = _scores_leave_one_event_out(samples)
            auc = float("nan") if scores is None else _auc(y, scores)
            null_values: np.ndarray
            if scores is None or not np.isfinite(auc):
                null_values = np.full((0,), np.nan, dtype=np.float32)
                p_value = float("nan")
                null_q95 = float("nan")
            else:
                null = []
                for _ in range(PERMUTATIONS):
                    shuffled = rng.permutation(y)
                    null.append(_auc(shuffled, scores))
                null_values = np.asarray(null, dtype=np.float32)
                p_value = float((np.sum(null_values >= auc) + 1.0) / (len(null_values) + 1.0))
                null_q95 = float(np.quantile(null_values, 0.95))
            rows.append(
                {
                    "stream": stream,
                    "cell_index": int(cell_index),
                    "cell_rank_from_edge": float(cell_ranks.get(int(cell_index), np.nan)),
                    "rel_time": int(rel_time),
                    "lag": int(DECODER_LAG),
                    "sample_count": int(len(samples)),
                    "positive_count": int(np.sum(y == 1)) if y.size else 0,
                    "negative_count": int(np.sum(y == -1)) if y.size else 0,
                    "auc": auc,
                    "null_auc_q95": null_q95,
                    "one_sided_p": p_value,
                    "estimable": bool(scores is not None and np.isfinite(auc)),
                }
            )
    return rows


def _mechanical_heatmaps(events: list[dict[str, Any]], *, cell_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for observable_index, observable in enumerate(("tension", "speed")):
        for cell_index in range(cell_count):
            for rel_time in MECH_REL_TIMES:
                values: list[float] = []
                for event in events:
                    event_i = int(event["event_index"])
                    sample_i = event_i + int(rel_time)
                    if sample_i < 0:
                        continue
                    series = event["tension_speed"][cell_index][observable_index]
                    if sample_i >= len(series):
                        continue
                    baseline_start = max(0, event_i - 64)
                    baseline_stop = max(baseline_start + 1, event_i - 16)
                    baseline = float(np.mean(series[baseline_start:baseline_stop]))
                    values.append(float(series[sample_i] - baseline))
                arr = np.asarray(values, dtype=np.float32)
                rows.append(
                    {
                        "observable": observable,
                        "cell_index": int(cell_index),
                        "rel_time": int(rel_time),
                        "n": int(arr.size),
                        "mean_deviation": float(np.mean(arr)) if arr.size else float("nan"),
                        "sd_deviation": float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan"),
                    }
                )
    return rows


def _mechanical_rank_heatmaps(events: list[dict[str, Any]], *, cell_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for observable_index, observable in enumerate(("tension", "speed")):
        for rank in range(cell_count):
            for rel_time in MECH_REL_TIMES:
                values: list[float] = []
                for event in events:
                    ranked_cells = {
                        int(cell): int(cell_rank)
                        for cell, cell_rank in event["cell_ranks"].items()
                    }
                    matching = [cell for cell, cell_rank in ranked_cells.items() if cell_rank == rank]
                    for cell_index in matching:
                        event_i = int(event["event_index"])
                        sample_i = event_i + int(rel_time)
                        if sample_i < 0:
                            continue
                        series = event["tension_speed"][cell_index][observable_index]
                        if sample_i >= len(series):
                            continue
                        baseline_start = max(0, event_i - 64)
                        baseline_stop = max(baseline_start + 1, event_i - 16)
                        baseline = float(np.mean(series[baseline_start:baseline_stop]))
                        values.append(float(series[sample_i] - baseline))
                arr = np.asarray(values, dtype=np.float32)
                rows.append(
                    {
                        "observable": observable,
                        "rank_from_edge": int(rank),
                        "rel_time": int(rel_time),
                        "n": int(arr.size),
                        "mean_deviation": float(np.mean(arr)) if arr.size else float("nan"),
                        "sd_deviation": float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan"),
                        "mean_abs_deviation": float(np.mean(np.abs(arr))) if arr.size else float("nan"),
                    }
                )
    return rows


def _mechanical_rank_release_delta(events: list[dict[str, Any]], *, cell_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for observable_index, observable in enumerate(("tension", "speed")):
        for rank in range(cell_count):
            for rel_time in MECH_REL_TIMES:
                values: list[float] = []
                for event in events:
                    ranked_cells = {
                        int(cell): int(cell_rank)
                        for cell, cell_rank in event["cell_ranks"].items()
                    }
                    matching = [cell for cell, cell_rank in ranked_cells.items() if cell_rank == rank]
                    for cell_index in matching:
                        event_i = int(event["event_index"])
                        sample_i = event_i + int(rel_time)
                        if sample_i < 0:
                            continue
                        series = event["tension_speed"][cell_index][observable_index]
                        if sample_i >= len(series) or event_i >= len(series):
                            continue
                        values.append(float(series[sample_i] - series[event_i]))
                arr = np.asarray(values, dtype=np.float32)
                rows.append(
                    {
                        "observable": observable,
                        "rank_from_edge": int(rank),
                        "rel_time": int(rel_time),
                        "n": int(arr.size),
                        "mean_delta_from_release": float(np.mean(arr)) if arr.size else float("nan"),
                        "mean_abs_delta_from_release": float(np.mean(np.abs(arr))) if arr.size else float("nan"),
                        "sd_delta_from_release": float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan"),
                    }
                )
    return rows


def _mean_cell_ranks(events: list[dict[str, Any]], cell_count: int) -> dict[int, float]:
    ranks: dict[int, list[int]] = {cell: [] for cell in range(cell_count)}
    for event in events:
        for cell, rank in event["cell_ranks"].items():
            ranks[int(cell)].append(int(rank))
    return {cell: float(np.mean(values)) for cell, values in ranks.items() if values}


def _onset_rows(map_rows: list[dict[str, Any]], *, cell_ranks: dict[int, float]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(map_rows)
    rows: list[dict[str, Any]] = []
    for stream, stream_group in frame.groupby("stream", sort=False):
        for cell_index, cell_group in stream_group.groupby("cell_index", sort=True):
            eligible = cell_group[
                (cell_group["estimable"])
                & (cell_group["rel_time"] >= -32)
                & (cell_group["auc"] > cell_group["null_auc_q95"])
                & (cell_group["one_sided_p"] <= 0.05)
            ].sort_values("rel_time", kind="stable")
            onset = int(eligible.iloc[0]["rel_time"]) if not eligible.empty else None
            rows.append(
                {
                    "stream": str(stream),
                    "cell_index": int(cell_index),
                    "cell_rank_from_edge": float(cell_ranks.get(int(cell_index), np.nan)),
                    "onset_rel_time": onset,
                    "onset_found": onset is not None,
                }
            )
    return rows


def _regression_rows(onsets: list[dict[str, Any]], *, random_seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(random_seed))
    rows: list[dict[str, Any]] = []
    frame = pd.DataFrame(onsets)
    for stream, group in frame.groupby("stream", sort=False):
        observed = group[group["onset_found"] & group["cell_rank_from_edge"].notna()].copy()
        if len(observed) < 3:
            rows.append(
                {
                    "stream": stream,
                    "cell_count_with_onset": int(len(observed)),
                    "slope_steps_per_cell": float("nan"),
                    "intercept": float("nan"),
                    "r2": float("nan"),
                    "permutation_p": float("nan"),
                }
            )
            continue
        x = observed["cell_rank_from_edge"].to_numpy(dtype=np.float64)
        y = observed["onset_rel_time"].to_numpy(dtype=np.float64)
        slope, intercept = np.polyfit(x, y, deg=1)
        pred = slope * x + intercept
        denom = float(np.sum((y - y.mean()) ** 2))
        r2 = float(1.0 - np.sum((y - pred) ** 2) / denom) if denom > 0 else float("nan")
        null_slopes = []
        for _ in range(PERMUTATIONS):
            shuffled = rng.permutation(y)
            null_slopes.append(float(np.polyfit(x, shuffled, deg=1)[0]))
        null_slopes_arr = np.asarray(null_slopes, dtype=np.float64)
        p_value = float((np.sum(null_slopes_arr >= slope) + 1.0) / (len(null_slopes_arr) + 1.0))
        rows.append(
            {
                "stream": stream,
                "cell_count_with_onset": int(len(observed)),
                "slope_steps_per_cell": float(slope),
                "intercept": float(intercept),
                "r2": r2,
                "permutation_p": p_value,
            }
        )
    return rows


def _mechanical_rank_onsets(rank_rows: list[dict[str, Any]], *, random_seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(random_seed))
    frame = pd.DataFrame(rank_rows)
    rows: list[dict[str, Any]] = []
    for observable, group in frame.groupby("observable", sort=False):
        onset_rows: list[dict[str, Any]] = []
        for rank, rank_group in group.groupby("rank_from_edge", sort=True):
            pre = rank_group[rank_group["rel_time"] < 0]["mean_abs_deviation"].to_numpy(dtype=np.float64)
            post = rank_group[rank_group["rel_time"] >= 0].sort_values("rel_time", kind="stable")
            threshold = float(np.quantile(pre[np.isfinite(pre)], 0.95)) if np.any(np.isfinite(pre)) else float("nan")
            eligible = post[post["mean_abs_deviation"] > threshold] if np.isfinite(threshold) else post.iloc[0:0]
            onset = int(eligible.iloc[0]["rel_time"]) if not eligible.empty else None
            peak_row = post.iloc[post["mean_abs_deviation"].to_numpy(dtype=np.float64).argmax()] if not post.empty else None
            onset_rows.append(
                {
                    "observable": str(observable),
                    "rank_from_edge": int(rank),
                    "pre_release_abs_q95": threshold,
                    "onset_rel_time": onset,
                    "onset_found": onset is not None,
                    "peak_rel_time": None if peak_row is None else int(peak_row["rel_time"]),
                    "peak_mean_abs_deviation": None if peak_row is None else float(peak_row["mean_abs_deviation"]),
                }
            )
        valid = [row for row in onset_rows if row["onset_found"]]
        if len(valid) >= 3:
            x = np.asarray([row["rank_from_edge"] for row in valid], dtype=np.float64)
            y = np.asarray([row["onset_rel_time"] for row in valid], dtype=np.float64)
            slope, intercept = np.polyfit(x, y, deg=1)
            null = np.asarray([np.polyfit(x, rng.permutation(y), deg=1)[0] for _ in range(PERMUTATIONS)], dtype=np.float64)
            p_value = float((np.sum(null >= slope) + 1.0) / (len(null) + 1.0))
        else:
            slope = intercept = p_value = float("nan")
        for row in onset_rows:
            row["onset_slope_steps_per_rank"] = float(slope)
            row["onset_intercept"] = float(intercept)
            row["onset_permutation_p"] = float(p_value)
        rows.extend(onset_rows)
    return rows


def _plot_heatmap(frame: pd.DataFrame, *, value_col: str, title: str, path: Path, vmin: float | None = None, vmax: float | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = sorted(int(v) for v in frame["cell_index"].dropna().unique())
    rel_times = sorted(int(v) for v in frame["rel_time"].dropna().unique())
    matrix = np.full((len(cells), len(rel_times)), np.nan, dtype=np.float32)
    cell_to_row = {cell: idx for idx, cell in enumerate(cells)}
    time_to_col = {rel: idx for idx, rel in enumerate(rel_times)}
    for _, row in frame.iterrows():
        matrix[cell_to_row[int(row["cell_index"])], time_to_col[int(row["rel_time"])]] = float(row[value_col])
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.axvline(time_to_col.get(0, 0), color="#D4736C", linewidth=1.0)
    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_xlabel("Steps from release")
    ax.set_ylabel("Cell")
    tick_cols = [idx for idx, rel in enumerate(rel_times) if rel % 64 == 0]
    ax.set_xticks(tick_cols)
    ax.set_xticklabels([str(rel_times[idx]) for idx in tick_cols], fontsize=8)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([str(cell) for cell in cells], fontsize=7)
    fig.colorbar(image, ax=ax, shrink=0.82)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_rank_heatmap(
    frame: pd.DataFrame,
    *,
    value_col: str,
    title: str,
    path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranks = sorted(int(v) for v in frame["rank_from_edge"].dropna().unique())
    rel_times = sorted(int(v) for v in frame["rel_time"].dropna().unique())
    matrix = np.full((len(ranks), len(rel_times)), np.nan, dtype=np.float32)
    rank_to_row = {rank: idx for idx, rank in enumerate(ranks)}
    time_to_col = {rel: idx for idx, rel in enumerate(rel_times)}
    for _, row in frame.iterrows():
        matrix[rank_to_row[int(row["rank_from_edge"])], time_to_col[int(row["rel_time"])]] = float(row[value_col])
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.axvline(time_to_col.get(0, 0), color="#D4736C", linewidth=1.0)
    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_xlabel("Steps from release")
    ax.set_ylabel("Rank from release edge")
    tick_cols = [idx for idx, rel in enumerate(rel_times) if rel % 64 == 0]
    ax.set_xticks(tick_cols)
    ax.set_xticklabels([str(rel_times[idx]) for idx in tick_cols], fontsize=8)
    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels([str(rank) for rank in ranks], fontsize=7)
    fig.colorbar(image, ax=ax, shrink=0.82)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def analyze(*, random_seed: int = 137) -> dict[str, Path]:
    summarize()
    events, robot_shape, hidden_start = _load_events()
    if not events:
        raise ValueError("No true release events available for analysis.")
    cell_count = int(next(iter(events[0]["body_features"].keys())) if False else len(events[0]["body_features"]))
    event_rows = [
        {
            "event_id": event["event_id"],
            "plateau_width": event["plateau_width"],
            "trial_id": event["trial_id"],
            "goal_distance": event["goal_distance"],
            "cliff_distance": event["cliff_distance"],
            "event_index": event["event_index"],
            "success": event["success"],
            "returned_from_edge": event["returned_from_edge"],
            "steps_alive": event["steps_alive"],
            "edge_side": event["edge_side"],
            "toward_label_count": int(np.sum(event["labels"] == 1)),
            "away_label_count": int(np.sum(event["labels"] == -1)),
            "uncommitted_label_count": int(np.sum(event["labels"] == 0)),
        }
        for event in events
    ]
    _write_csv(PACKAGE_ROOT / "tables" / "release_event_manifest.csv", event_rows)

    map_rows: list[dict[str, Any]] = []
    for stream in ("body", "signal"):
        map_rows.extend(_decoder_map(events, stream=stream, cell_count=cell_count, random_seed=_stable_seed(random_seed, stream)))
    _write_csv(PACKAGE_ROOT / "tables" / "event_aligned_auc_map.csv", map_rows)

    mech_rows = _mechanical_heatmaps(events, cell_count=cell_count)
    _write_csv(PACKAGE_ROOT / "tables" / "mechanical_deviation_heatmap.csv", mech_rows)
    mech_rank_rows = _mechanical_rank_heatmaps(events, cell_count=cell_count)
    _write_csv(PACKAGE_ROOT / "tables" / "mechanical_rank_deviation_heatmap.csv", mech_rank_rows)
    mech_release_delta_rows = _mechanical_rank_release_delta(events, cell_count=cell_count)
    _write_csv(PACKAGE_ROOT / "tables" / "mechanical_rank_release_delta_heatmap.csv", mech_release_delta_rows)
    mech_rank_onsets = _mechanical_rank_onsets(
        mech_rank_rows,
        random_seed=_stable_seed(random_seed, "mechanical_rank_onsets"),
    )
    _write_csv(PACKAGE_ROOT / "tables" / "mechanical_rank_onsets.csv", mech_rank_onsets)

    ranks = _mean_cell_ranks(events, cell_count)
    onset = _onset_rows(map_rows, cell_ranks=ranks)
    _write_csv(PACKAGE_ROOT / "tables" / "decoder_onsets.csv", onset)
    regression = _regression_rows(onset, random_seed=_stable_seed(random_seed, "onset_regression"))
    _write_csv(PACKAGE_ROOT / "tables" / "onset_regression.csv", regression)

    figures_dir = PACKAGE_ROOT / "figures"
    map_frame = pd.DataFrame(map_rows)
    for stream in ("body", "signal"):
        _plot_heatmap(
            map_frame[map_frame["stream"] == stream],
            value_col="auc",
            title=f"S4 {stream} event-aligned AUC",
            path=figures_dir / f"auc_map_{stream}.png",
            vmin=0.5,
            vmax=1.0,
        )
        rank_frame = (
            map_frame[map_frame["stream"] == stream]
            .dropna(subset=["cell_rank_from_edge"])
            .assign(rank_from_edge=lambda d: d["cell_rank_from_edge"].round().astype(int))
            .groupby(["rank_from_edge", "rel_time"], as_index=False)["auc"]
            .mean()
        )
        _plot_rank_heatmap(
            rank_frame,
            value_col="auc",
            title=f"S4 {stream} AUC by edge rank",
            path=figures_dir / f"auc_rank_map_{stream}.png",
            vmin=0.5,
            vmax=1.0,
        )
    mech_frame = pd.DataFrame(mech_rows)
    for observable in ("tension", "speed"):
        obs = mech_frame[mech_frame["observable"] == observable]
        finite = obs["mean_deviation"].to_numpy(dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        vmax = float(np.quantile(np.abs(finite), 0.98)) if finite.size else None
        _plot_heatmap(
            obs,
            value_col="mean_deviation",
            title=f"S4 mechanical {observable} deviation",
            path=figures_dir / f"mechanical_{observable}_deviation.png",
            vmin=-vmax if vmax else None,
            vmax=vmax,
        )
    mech_rank_frame = pd.DataFrame(mech_rank_rows)
    for observable in ("tension", "speed"):
        obs = mech_rank_frame[mech_rank_frame["observable"] == observable]
        finite = obs["mean_deviation"].to_numpy(dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        vmax = float(np.quantile(np.abs(finite), 0.98)) if finite.size else None
        _plot_rank_heatmap(
            obs,
            value_col="mean_deviation",
            title=f"S4 {observable} deviation by edge rank",
            path=figures_dir / f"mechanical_rank_{observable}_deviation.png",
            vmin=-vmax if vmax else None,
            vmax=vmax,
        )
    mech_release_delta_frame = pd.DataFrame(mech_release_delta_rows)
    for observable in ("tension", "speed"):
        obs = mech_release_delta_frame[mech_release_delta_frame["observable"] == observable]
        finite = obs["mean_delta_from_release"].to_numpy(dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        vmax = float(np.quantile(np.abs(finite), 0.98)) if finite.size else None
        _plot_rank_heatmap(
            obs,
            value_col="mean_delta_from_release",
            title=f"S4 {observable} change from release",
            path=figures_dir / f"mechanical_rank_{observable}_release_delta.png",
            vmin=-vmax if vmax else None,
            vmax=vmax,
        )

    summary = {
        "schema_version": 1,
        "kind": "s4_goal_cliff_propagation_analysis",
        "event_count": len(events),
        "target_event_count": len(PAIRS) * len(PLATEAU_WIDTHS),
        "robot_shape": robot_shape,
        "hidden_start_channel": hidden_start,
        "decoder": {
            "streams": ["body", "signal"],
            "relative_times": list(REL_TIMES),
            "lag": DECODER_LAG,
            "future_window": FUTURE_WINDOW,
            "min_displacement": MIN_DISPLACEMENT,
            "persistence_steps": PERSISTENCE_STEPS,
            "persistence_fraction": PERSISTENCE_FRACTION,
            "ridge_alpha": RIDGE_ALPHA,
            "permutations": PERMUTATIONS,
            "null_policy": "one-sided score-label permutation null using leave-one-event-out scores",
        },
        "paths": {
            "release_event_manifest": PACKAGE_ROOT / "tables" / "release_event_manifest.csv",
            "event_aligned_auc_map": PACKAGE_ROOT / "tables" / "event_aligned_auc_map.csv",
            "mechanical_deviation_heatmap": PACKAGE_ROOT / "tables" / "mechanical_deviation_heatmap.csv",
            "mechanical_rank_deviation_heatmap": PACKAGE_ROOT / "tables" / "mechanical_rank_deviation_heatmap.csv",
            "mechanical_rank_release_delta_heatmap": PACKAGE_ROOT / "tables" / "mechanical_rank_release_delta_heatmap.csv",
            "mechanical_rank_onsets": PACKAGE_ROOT / "tables" / "mechanical_rank_onsets.csv",
            "decoder_onsets": PACKAGE_ROOT / "tables" / "decoder_onsets.csv",
            "onset_regression": PACKAGE_ROOT / "tables" / "onset_regression.csv",
            "figures": figures_dir,
        },
    }
    _write_json(PACKAGE_ROOT / "tables" / "analysis_summary.json", summary)
    (PACKAGE_ROOT / "notes" / "analysis_summary.md").write_text(
        "# S4 goal-cliff propagation analysis\n\n"
        f"True release events analyzed: {len(events)} / {len(PAIRS) * len(PLATEAU_WIDTHS)}.\n\n"
        "Decoder maps use event-aligned per-cell features to predict future committed movement "
        f"at lag {DECODER_LAG}, with committed labels defined by future COM displacement over "
        f"{FUTURE_WINDOW} steps.\n\n"
        "The red vertical line in each figure marks the release frame (`rel_time = 0`).\n\n"
        "Generated tables:\n"
        "- `tables/release_event_manifest.csv`\n"
        "- `tables/event_aligned_auc_map.csv`\n"
        "- `tables/mechanical_deviation_heatmap.csv`\n"
        "- `tables/mechanical_rank_deviation_heatmap.csv`\n"
        "- `tables/mechanical_rank_release_delta_heatmap.csv`\n"
        "- `tables/mechanical_rank_onsets.csv`\n"
        "- `tables/decoder_onsets.csv`\n"
        "- `tables/onset_regression.csv`\n",
    )
    return {
        "analysis_summary": PACKAGE_ROOT / "tables" / "analysis_summary.json",
        "analysis_note": PACKAGE_ROOT / "notes" / "analysis_summary.md",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    launch_parser = sub.add_parser("launch-modal")
    launch_parser.add_argument("--dry-run", action="store_true")
    pull_parser = sub.add_parser("pull-modal")
    pull_parser.add_argument("--dry-run", action="store_true")
    sub.add_parser("summarize")
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--random-seed", type=int, default=137)
    args = parser.parse_args()

    if args.command == "init":
        write_launch_manifest()
        print(PACKAGE_ROOT)
    elif args.command == "launch-modal":
        launch_modal(dry_run=bool(args.dry_run))
    elif args.command == "pull-modal":
        pull_modal(dry_run=bool(args.dry_run))
    elif args.command == "summarize":
        paths = summarize()
        for key, value in paths.items():
            print(f"{key}: {value}")
    elif args.command == "analyze":
        paths = analyze(random_seed=int(args.random_seed))
        for key, value in paths.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
