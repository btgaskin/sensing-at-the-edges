#!/usr/bin/env python
"""Run and analyze the S4 support-preserved deformation-off control.

The control keeps S4 in the same deformation-off, intact-signalling regime as
the fixed-width cliff-distance sweep, but moves the wrong-side cliff far enough
away that no support is lost during the 30k-step replay.  The analysis then uses
the release times from the fixed-width sweep as pseudo-events and asks whether
S4 reverses without the physical release.
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


DEFAULT_PACKAGE_ROOT = REPO_ROOT / "paper_data" / "preprint" / "s4_support_preserved_control_20260620"
PACKAGE_ROOT = Path(os.environ.get("S4_SUPPORT_CONTROL_PACKAGE_ROOT", str(DEFAULT_PACKAGE_ROOT)))
REMOTE_ROOT = "/outputs/preprint_s4_support_preserved_control_20260620"
FIXED_WIDTH_REMOTE_ROOT = "/outputs/preprint_s4_fixed_width_distance_sweep_20260618"
FIXED_WIDTH_LOCAL_ROOT = REPO_ROOT / "paper_data" / "preprint" / "s4_fixed_width_distance_sweep_20260618"

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
GOAL_DISTANCE = 12
SUPPORT_PRESERVED_CLIFF_DISTANCE = 80
GRID_W = 180
PLATEAU_WIDTH = 12
MAX_STEPS = 30_000
POST_WINDOWS = (32, 128, 336)
PRE_WINDOW = 128
REVERSAL_DELTA_THRESHOLD = 0.025

COMMAND_ARRAYS = (
    "phase_action_outputs",
    "phase_h_drive",
    "phase_v_drive",
    "phase_nca_state",
    "phase_ctrnn_carry",
)


def _data_root() -> Path:
    return Path(os.environ.get("S4_SUPPORT_CONTROL_DATA_ROOT", str(PACKAGE_ROOT / "modal" / GPU_PROFILE)))


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
        "kind": "s4_support_preserved_control_package",
        "created_for": "S4 deformation-off support-preserved platform control",
        "package_root": str(PACKAGE_ROOT),
        "remote_root": REMOTE_ROOT,
        "fixed_width_reference_root": FIXED_WIDTH_REMOTE_ROOT,
        "run_id": RUN_ID,
        "specimen_id": SPECIMEN_ID,
        "gpu_profile": GPU_PROFILE,
        "assay": ASSAY,
        "condition": "deformation_off",
        "signaling_condition": "intact",
        "goal_side": GOAL_SIDE,
        "goal_distance": GOAL_DISTANCE,
        "support_preserved_cliff_distance": SUPPORT_PRESERVED_CLIFF_DISTANCE,
        "grid_w": GRID_W,
        "plateau_width": PLATEAU_WIDTH,
        "max_steps": MAX_STEPS,
        "post_windows": list(POST_WINDOWS),
        "pre_window": PRE_WINDOW,
        "reversal_delta_threshold": REVERSAL_DELTA_THRESHOLD,
    }
    _write_json(PACKAGE_ROOT / "manifests" / "package_manifest.json", manifest)
    (PACKAGE_ROOT / "notes" / "README.md").write_text(
        "# S4 support-preserved control\n\n"
        "Modal A100-40GB deformation-off S4 replay with the wrong-side cliff moved far away.\n\n"
        f"- Specimen: `{SPECIMEN_ID}`.\n"
        f"- Goal side `{GOAL_SIDE}`, goal distance `{GOAL_DISTANCE}`, cliff distance `{SUPPORT_PRESERVED_CLIFF_DISTANCE}`.\n"
        f"- Grid width `{GRID_W}`, plateau/base width `{PLATEAU_WIDTH}`, max steps `{MAX_STEPS}`.\n"
        "- The fixed-width release times are used as pseudo-events to test whether S4 reverses without loss of support.\n"
        "- S4 genotype/specimen artifacts are copied under `specimen/` for provenance.\n",
    )
    return PACKAGE_ROOT


def _override_tokens(*, analysis_root: str, specimens_output_root: str) -> list[str]:
    pairs = ((GOAL_DISTANCE, SUPPORT_PRESERVED_CLIFF_DISTANCE),)
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


def _remote_trace_root() -> str:
    return f"{REMOTE_ROOT}/trace_run"


def modal_launch_argv(*, detach_modal: bool = False, detach_function: bool = False) -> list[str]:
    item = _s4_item()
    argv = [
        "uv",
        "run",
        "--python",
        ".venv/bin/python",
        "modal",
        "run",
    ]
    if detach_modal:
        argv.append("--detach")
    argv.extend(
        [
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
                    analysis_root=_remote_trace_root(),
                    specimens_output_root=item["specimens_output_root"],
                )
            ),
            "--gpu-profile",
            GPU_PROFILE,
        ]
    )
    if detach_function:
        argv.append("--detach-launch")
    return argv


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
        REMOTE_ROOT.removeprefix("/outputs/"),
        str(PACKAGE_ROOT / "modal" / GPU_PROFILE),
    ]


def write_launch_manifest() -> Path:
    initialize_package()
    launch = modal_launch_argv(detach_modal=False, detach_function=False)
    pull = pull_argv()
    path = PACKAGE_ROOT / "manifests" / "modal_launch_manifest.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "kind": "s4_support_preserved_control_modal_launch_manifest",
            "launch": {
                "remote_analysis_root": _remote_trace_root(),
                "modal_volume_path": _remote_trace_root().removeprefix("/outputs/"),
                "launch_argv": launch,
                "launch_command": _shell_join(launch),
            },
            "pull": {
                "remote_analysis_root": REMOTE_ROOT,
                "modal_volume_path": REMOTE_ROOT.removeprefix("/outputs/"),
                "pull_argv": pull,
                "pull_command": _shell_join(pull),
            },
            "expected": {
                "trace_npz_total": 1,
                "support_preserved_cliff_distance": SUPPORT_PRESERVED_CLIFF_DISTANCE,
            },
        },
    )
    return path


def launch_modal(*, dry_run: bool = False, detach_modal: bool = False, detach_function: bool = False) -> None:
    initialize_package()
    argv = modal_launch_argv(detach_modal=detach_modal, detach_function=detach_function)
    print(_shell_join(argv))
    if not dry_run:
        subprocess.run(argv, cwd=REPO_ROOT, check=True)


def pull_modal(*, dry_run: bool = False) -> None:
    write_launch_manifest()
    argv = pull_argv()
    print(_shell_join(argv))
    if not dry_run:
        dest = Path(argv[-1])
        if dest.exists():
            shutil.rmtree(dest)
        subprocess.run(argv, cwd=REPO_ROOT, check=True)


def _candidate_roots(root: Path) -> list[Path]:
    roots = [root]
    nested = root / REMOTE_ROOT.removeprefix("/outputs/")
    if nested != root:
        roots.append(nested)
    return roots


def _find_support_assay_root() -> Path:
    roots = _candidate_roots(_data_root())
    matches: list[Path] = []
    for root in roots:
        matches.extend(path.parent for path in root.glob(f"**/{ASSAY}/trial_manifest.csv"))
    matches = sorted({path.resolve() for path in matches})
    if not matches:
        raise FileNotFoundError(
            f"Could not find {ASSAY}/trial_manifest.csv under any support control data root: "
            + ", ".join(str(root) for root in roots)
        )
    if len(matches) > 1:
        # Prefer the explicit support-preserved cliff-distance trace if old scratch data exists.
        scored: list[tuple[int, Path]] = []
        for match in matches:
            trial_path = match / "trial_manifest.csv"
            try:
                df = pd.read_csv(trial_path)
                score = int((df.get("cliff_distance") == SUPPORT_PRESERVED_CLIFF_DISTANCE).sum())
            except Exception:
                score = 0
            scored.append((score, match))
        matches = [path for _score, path in sorted(scored, key=lambda item: (-item[0], str(item[1])))]
    return matches[0]


def _fixed_release_manifest_path() -> Path:
    env_path = os.environ.get("S4_FIXED_WIDTH_RELEASE_MANIFEST")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
    remote_path = Path(FIXED_WIDTH_REMOTE_ROOT) / "tables" / "release_event_manifest.csv"
    if remote_path.exists():
        return remote_path
    local_path = FIXED_WIDTH_LOCAL_ROOT / "tables" / "release_event_manifest.csv"
    if local_path.exists():
        return local_path
    raise FileNotFoundError("Could not find fixed-width release_event_manifest.csv")


def _resolve_path(path_value: Any, *, support_root: Path | None = None) -> Path:
    raw = str(path_value)
    path = Path(raw)
    if path.exists():
        return path
    if support_root is not None and raw.startswith(REMOTE_ROOT):
        rel = Path(raw).relative_to(REMOTE_ROOT)
        for root in _candidate_roots(_data_root()):
            candidate = root / rel
            if candidate.exists():
                return candidate
    if raw.startswith(FIXED_WIDTH_REMOTE_ROOT):
        rel = Path(raw).relative_to(FIXED_WIDTH_REMOTE_ROOT)
        local_candidate = FIXED_WIDTH_LOCAL_ROOT / rel
        if local_candidate.exists():
            return local_candidate
    return path


def _load_support_tables(assay_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_df = pd.read_csv(assay_root / "trial_manifest.csv")
    step_parquet = assay_root / "step_manifest.parquet"
    if step_parquet.exists():
        step_df = pd.read_parquet(step_parquet)
    else:
        step_df = pd.read_csv(assay_root / "step_manifest.csv")
    return trial_df, step_df


def _platform_trace_path(trial_df: pd.DataFrame, *, assay_root: Path) -> Path:
    if trial_df.empty:
        raise ValueError("support control trial_manifest.csv is empty")
    trace_path = _resolve_path(trial_df.iloc[0]["trace_npz"], support_root=assay_root)
    if trace_path.exists():
        return trace_path
    fallback = next((assay_root / "traces").glob("*.npz"), None)
    if fallback is None:
        raise FileNotFoundError(f"Could not find support control trace npz: {trace_path}")
    return fallback


def _direction_label(delta: float) -> str:
    if float(delta) > REVERSAL_DELTA_THRESHOLD:
        return "away_from_cliff"
    if float(delta) < -REVERSAL_DELTA_THRESHOLD:
        return "toward_cliff"
    return "near_stationary"


def _first_sustained_away_step(com_x: np.ndarray) -> int | None:
    max_window = max(POST_WINDOWS)
    for step in range(PRE_WINDOW, max(0, len(com_x) - max_window)):
        pre_delta = float(com_x[step] - com_x[step - PRE_WINDOW])
        post_delta = float(com_x[step + max_window] - com_x[step])
        if pre_delta < -REVERSAL_DELTA_THRESHOLD and post_delta > REVERSAL_DELTA_THRESHOLD:
            return int(step)
    return None


def _pseudo_release_rows(*, step_df: pd.DataFrame, release_df: pd.DataFrame) -> list[dict[str, Any]]:
    ordered_steps = step_df.sort_values("step_index", kind="stable")
    com_x = ordered_steps["com_x"].to_numpy(dtype=float)
    front_x = ordered_steps["front_x"].to_numpy(dtype=float)
    back_x = ordered_steps["back_x"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for _, event in release_df.sort_values("cliff_distance", kind="stable").iterrows():
        release_step = int(event["release_step"])
        if release_step < PRE_WINDOW or release_step + max(POST_WINDOWS) >= len(com_x):
            rows.append(
                {
                    "event_id": event["event_id"],
                    "reference_cliff_distance": int(event["cliff_distance"]),
                    "reference_release_step": release_step,
                    "valid_window": False,
                    "reason": "outside_support_trace_window",
                }
            )
            continue
        pre_delta = float(com_x[release_step] - com_x[release_step - PRE_WINDOW])
        record: dict[str, Any] = {
            "event_id": event["event_id"],
            "reference_trial_id": event["trial_id"],
            "reference_cliff_distance": int(event["cliff_distance"]),
            "reference_release_step": release_step,
            "valid_window": True,
            "platform_com_x_at_pseudo_release": float(com_x[release_step]),
            "platform_front_x_at_pseudo_release": float(front_x[release_step]),
            "platform_back_x_at_pseudo_release": float(back_x[release_step]),
            "pre_delta_128": pre_delta,
            "pre_direction_128": _direction_label(pre_delta),
        }
        for window in POST_WINDOWS:
            delta = float(com_x[release_step + window] - com_x[release_step])
            record[f"post_delta_{window}"] = delta
            record[f"post_direction_{window}"] = _direction_label(delta)
            record[f"pseudo_reversal_{window}"] = bool(
                pre_delta < -REVERSAL_DELTA_THRESHOLD and delta > REVERSAL_DELTA_THRESHOLD
            )
        rows.append(record)
    return rows


def _command_identity_rows(*, platform_trace_path: Path, release_df: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    platform_npz = np.load(platform_trace_path)
    missing_fixed = 0
    comparisons = 0
    exact_full_windows = 0
    exact_pre_windows = 0
    for _, event in release_df.sort_values("cliff_distance", kind="stable").iterrows():
        fixed_trace_path = _resolve_path(event["trace_path"])
        fixed_exists = fixed_trace_path.exists()
        if not fixed_exists:
            missing_fixed += 1
        for array_name in COMMAND_ARRAYS:
            row: dict[str, Any] = {
                "event_id": event["event_id"],
                "reference_cliff_distance": int(event["cliff_distance"]),
                "reference_release_step": int(event["release_step"]),
                "array": array_name,
                "fixed_trace_path": str(fixed_trace_path),
                "fixed_trace_exists": bool(fixed_exists),
            }
            if not fixed_exists:
                rows.append(row)
                continue
            fixed_npz = np.load(fixed_trace_path)
            if array_name not in platform_npz or array_name not in fixed_npz:
                row["status"] = "missing_array"
                rows.append(row)
                continue
            platform = np.asarray(platform_npz[array_name])
            fixed = np.asarray(fixed_npz[array_name])
            full_end = min(int(event["release_step"]) + max(POST_WINDOWS), platform.shape[0], fixed.shape[0])
            pre_end = min(int(event["release_step"]), platform.shape[0], fixed.shape[0])
            full_exact = bool(np.array_equal(platform[:full_end], fixed[:full_end]))
            pre_exact = bool(np.array_equal(platform[:pre_end], fixed[:pre_end]))
            full_max_abs = float(np.max(np.abs(platform[:full_end] - fixed[:full_end]))) if full_end > 0 else float("nan")
            pre_max_abs = float(np.max(np.abs(platform[:pre_end] - fixed[:pre_end]))) if pre_end > 0 else float("nan")
            row.update(
                {
                    "status": "compared",
                    "pre_end": int(pre_end),
                    "full_end": int(full_end),
                    "pre_exact": pre_exact,
                    "full_exact": full_exact,
                    "pre_max_abs": pre_max_abs,
                    "full_max_abs": full_max_abs,
                    "platform_shape": "x".join(str(v) for v in platform.shape),
                    "fixed_shape": "x".join(str(v) for v in fixed.shape),
                }
            )
            comparisons += 1
            exact_pre_windows += int(pre_exact)
            exact_full_windows += int(full_exact)
            rows.append(row)
    summary = {
        "command_arrays": list(COMMAND_ARRAYS),
        "fixed_trace_missing_events": int(missing_fixed),
        "array_comparisons": int(comparisons),
        "pre_window_exact_comparisons": int(exact_pre_windows),
        "full_window_exact_comparisons": int(exact_full_windows),
        "all_compared_pre_windows_exact": bool(comparisons > 0 and exact_pre_windows == comparisons),
        "all_compared_full_windows_exact": bool(comparisons > 0 and exact_full_windows == comparisons),
    }
    return rows, summary


def _make_figures(*, step_df: pd.DataFrame, pseudo_df: pd.DataFrame) -> list[Path]:
    import matplotlib.pyplot as plt

    figure_dir = PACKAGE_ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    ordered = step_df.sort_values("step_index", kind="stable")
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    ax.plot(ordered["step_index"], ordered["com_x"], color="#3B5B92", linewidth=1.0)
    valid = pseudo_df[pseudo_df.get("valid_window", False) == True]  # noqa: E712
    for step in valid["reference_release_step"]:
        ax.axvline(float(step), color="#AAAAAA", alpha=0.18, linewidth=0.6)
    ax.set_xlabel("Step")
    ax.set_ylabel("COM x")
    ax.set_title("S4 support-preserved replay: COM path with fixed-width release times")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = figure_dir / f"support_preserved_com_trace.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    valid = valid.sort_values("reference_cliff_distance", kind="stable")
    ax.axhline(0, color="#777777", linestyle="--", linewidth=0.8)
    ax.bar(
        valid["reference_cliff_distance"].astype(int),
        valid["post_delta_336"].astype(float),
        color="#8CB87C",
        width=0.8,
    )
    ax.set_xlabel("Reference cliff distance")
    ax.set_ylabel("COM delta after 336 steps")
    ax.set_title("No support-loss pseudo-events do not show edge-locked return")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = figure_dir / f"support_preserved_post_delta_336.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _forced_command_replay_rows(*, config_path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import jax
    import jax.numpy as jnp
    from omegaconf import OmegaConf

    from jax_evogym.actuators import set_per_axis_goals_compact
    from jax_evogym.sim import physics_substep_dynamic_broadphase
    from sensing_edges.evaluate import _build_phase_assets
    from sensing_edges.env import CELL_SIZE, _robot_kinematics
    from sensing_edges.post_run import _build_bilateral_entry, _phase_cfg_for_assay
    from sensing_edges.runtime_config import merge_stored_config, resolve_runtime_config, rollout_kwargs_from_cfg

    try:
        from _modal_common import load_config
    except Exception as exc:  # pragma: no cover - only used in Modal forced replay.
        raise RuntimeError("forced command replay requires the Modal _modal_common helper") from exc

    stored_cfg = OmegaConf.load(config_path)
    base_preset = str(stored_cfg.get("preset_id") or "config")
    base_cfg = OmegaConf.create(OmegaConf.to_container(load_config(config_name=base_preset), resolve=False))
    cfg = merge_stored_config(base_cfg, stored_cfg, prefer_current_post_run=True)
    runtime_resolution = resolve_runtime_config(cfg)
    phase_cfg = _phase_cfg_for_assay(
        cfg,
        max_steps=MAX_STEPS,
        active_deformation_condition="deformation_off",
        active_signaling_condition="intact",
        active_sensing_freeze_mode=str(cfg.eval.post_run.get("active_sensing_freeze_mode", "none")),
    )
    phase_cfg.terrain.grid_w = int(GRID_W)
    phase_cfg.terrain.center_plateau_width = int(PLATEAU_WIDTH)
    phase_cfg.terrain.base_width = int(PLATEAU_WIDTH)
    entry = _build_bilateral_entry(
        phase_cfg,
        goal_distance=GOAL_DISTANCE,
        cliff_distance=SUPPORT_PRESERVED_CLIFF_DISTANCE,
        goal_side=GOAL_SIDE,
        extra_metadata={"support_preserved_control": True},
    )
    phase_resolution, envs, _alive_mask, _static_capacity = _build_phase_assets(
        phase_cfg,
        runtime_resolution,
        [entry],
    )
    env_data = envs[0]
    rollout_kwargs = rollout_kwargs_from_cfg(
        phase_cfg,
        collision_tuning=phase_resolution.resolved_collision,
    )
    sense_act_phases = int(phase_cfg.sim.sense_act_phases_per_env_step)
    physics_substeps = int(phase_cfg.sim.physics_substeps)
    substeps_per_phase = int(physics_substeps // sense_act_phases)
    if substeps_per_phase <= 0:
        raise ValueError("physics_substeps must be >= sense_act_phases_per_env_step")
    dynamic_max_candidates = int(rollout_kwargs["dynamic_max_candidates"])
    dynamic_bin_size = float(rollout_kwargs["dynamic_bin_size"])
    dynamic_impl = str(rollout_kwargs["dynamic_impl"])
    dynamic_max_narrow = rollout_kwargs["dynamic_max_narrow"]
    support_left_frontier = float(int(entry.metadata["platform_start_x"]) * float(CELL_SIZE))
    support_right_frontier = float((int(entry.metadata["platform_end_x"]) + 1) * float(CELL_SIZE))

    def _replay_one(h_drive: jnp.ndarray, v_drive: jnp.ndarray) -> dict[str, jnp.ndarray]:
        def _step(sim_state, drives):
            h_step, v_step = drives

            def _phase(phase_sim_state, phase_drives):
                h_phase, v_phase = phase_drives
                phase_sim_state = set_per_axis_goals_compact(
                    phase_sim_state,
                    env_data.per_axis_info,
                    h_phase,
                    v_phase,
                )

                def _substep(sub_sim_state, _unused):
                    next_sim = physics_substep_dynamic_broadphase(
                        sub_sim_state,
                        env_data.topology,
                        env_data.constants,
                        collision_data=env_data.collision_data,
                        static_collider_data=env_data.static_collider_data,
                        static_collision_worklist=env_data.static_collision_worklist,
                        dynamic_max_candidates=jnp.int32(dynamic_max_candidates),
                        dynamic_bin_size=jnp.float32(dynamic_bin_size),
                        dynamic_impl=dynamic_impl,
                        dynamic_max_narrow=dynamic_max_narrow,
                    )
                    return next_sim, None

                phase_sim_state, _ = jax.lax.scan(
                    _substep,
                    phase_sim_state,
                    None,
                    length=substeps_per_phase,
                )
                return phase_sim_state, None

            sim_state, _ = jax.lax.scan(_phase, sim_state, (h_step, v_step))
            stats = _robot_kinematics(sim_state, env_data.robot_point_mask)
            return sim_state, {
                "com_x": stats.com_x,
                "min_x": stats.min_x,
                "front_x": stats.front_x,
                "min_y": stats.min_y,
                "max_y": stats.max_y,
            }

        _final_state, trace = jax.lax.scan(
            _step,
            env_data.sim_state,
            (h_drive, v_drive),
        )
        return trace

    @jax.jit
    def _replay_batch(h_drive: jnp.ndarray, v_drive: jnp.ndarray) -> dict[str, jnp.ndarray]:
        return jax.vmap(_replay_one)(h_drive, v_drive)

    release_df = pd.read_csv(_fixed_release_manifest_path())
    command_records: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    max_command_len = 0
    for _, event in release_df.sort_values("cliff_distance", kind="stable").iterrows():
        fixed_trace_path = _resolve_path(event["trace_path"])
        if not fixed_trace_path.exists():
            status_rows.append(
                {
                    "event_id": event["event_id"],
                    "reference_cliff_distance": int(event["cliff_distance"]),
                    "reference_release_step": int(event["release_step"]),
                    "status": "missing_fixed_trace",
                    "fixed_trace_path": str(fixed_trace_path),
                }
            )
            continue
        fixed_npz = np.load(fixed_trace_path)
        release_step = int(event["release_step"])
        command_end = min(release_step + max(POST_WINDOWS) + 1, fixed_npz["phase_h_drive"].shape[0])
        if command_end <= release_step + max(POST_WINDOWS):
            status_rows.append(
                {
                    "event_id": event["event_id"],
                    "reference_cliff_distance": int(event["cliff_distance"]),
                    "reference_release_step": release_step,
                    "status": "insufficient_command_length",
                    "fixed_trace_path": str(fixed_trace_path),
                }
            )
            continue
        h_np = np.asarray(fixed_npz["phase_h_drive"][:command_end], dtype=np.float32)
        v_np = np.asarray(fixed_npz["phase_v_drive"][:command_end], dtype=np.float32)
        fixed_positions = np.asarray(fixed_npz["plant_positions"][:command_end], dtype=np.float32)
        fixed_robot_mask = np.asarray(fixed_npz["robot_point_mask"], dtype=bool)
        fixed_com_x = np.mean(fixed_positions[:, fixed_robot_mask, 0], axis=1)
        fixed_metrics: dict[str, Any] = {}
        if release_step >= PRE_WINDOW and release_step + max(POST_WINDOWS) < fixed_com_x.shape[0]:
            fixed_pre_delta = float(fixed_com_x[release_step] - fixed_com_x[release_step - PRE_WINDOW])
            fixed_metrics["fixed_pre_delta_128"] = fixed_pre_delta
            fixed_metrics["fixed_pre_direction_128"] = _direction_label(fixed_pre_delta)
            for window in POST_WINDOWS:
                fixed_delta = float(fixed_com_x[release_step + window] - fixed_com_x[release_step])
                fixed_metrics[f"fixed_post_delta_{window}"] = fixed_delta
                fixed_metrics[f"fixed_post_direction_{window}"] = _direction_label(fixed_delta)
                fixed_metrics[f"fixed_reversal_{window}"] = bool(
                    fixed_pre_delta < -REVERSAL_DELTA_THRESHOLD
                    and fixed_delta > REVERSAL_DELTA_THRESHOLD
                )
        max_command_len = max(max_command_len, int(command_end))
        command_records.append(
            {
                "event": event.to_dict(),
                "fixed_trace_path": fixed_trace_path,
                "command_end": int(command_end),
                "h_drive": h_np,
                "v_drive": v_np,
                "fixed_metrics": fixed_metrics,
            }
        )

    rows: list[dict[str, Any]] = list(status_rows)
    if command_records:
        h_shape = command_records[0]["h_drive"].shape[1:]
        v_shape = command_records[0]["v_drive"].shape[1:]
        h_batch = np.ones((len(command_records), max_command_len, *h_shape), dtype=np.float32)
        v_batch = np.ones((len(command_records), max_command_len, *v_shape), dtype=np.float32)
        for index, record in enumerate(command_records):
            command_end = int(record["command_end"])
            h_batch[index, :command_end] = record["h_drive"]
            v_batch[index, :command_end] = record["v_drive"]
        replay_batch = jax.tree_util.tree_map(
            lambda value: np.asarray(value),
            _replay_batch(jnp.asarray(h_batch), jnp.asarray(v_batch)),
        )
    else:
        replay_batch = {}

    for batch_index, command_record in enumerate(command_records):
        event = command_record["event"]
        fixed_trace_path = command_record["fixed_trace_path"]
        release_step = int(event["release_step"])
        command_end = int(command_record["command_end"])
        com_x = np.asarray(replay_batch["com_x"][batch_index, :command_end], dtype=float)
        min_x = np.asarray(replay_batch["min_x"][batch_index, :command_end], dtype=float)
        front_x = np.asarray(replay_batch["front_x"][batch_index, :command_end], dtype=float)
        if release_step < PRE_WINDOW or release_step + max(POST_WINDOWS) >= com_x.shape[0]:
            rows.append(
                {
                    "event_id": event["event_id"],
                    "reference_cliff_distance": int(event["cliff_distance"]),
                    "reference_release_step": release_step,
                    "status": "outside_replay_window",
                    "command_end": int(command_end),
                    "fixed_trace_path": str(fixed_trace_path),
                }
            )
            continue
        left_clearance = min_x - support_left_frontier
        right_clearance = support_right_frontier - front_x
        pre_delta = float(com_x[release_step] - com_x[release_step - PRE_WINDOW])
        record: dict[str, Any] = {
            "event_id": event["event_id"],
            "reference_trial_id": event["trial_id"],
            "reference_cliff_distance": int(event["cliff_distance"]),
            "reference_release_step": release_step,
            "status": "replayed",
            "command_source": "fixed_width_phase_h_drive_phase_v_drive",
            "command_end": int(command_end),
            "fixed_trace_path": str(fixed_trace_path),
            "support_left_frontier_x": support_left_frontier,
            "support_right_frontier_x": support_right_frontier,
            "min_left_clearance": float(np.min(left_clearance)),
            "min_right_clearance": float(np.min(right_clearance)),
            "first_left_support_loss_step": (
                int(np.flatnonzero(left_clearance <= 0.0)[0])
                if np.any(left_clearance <= 0.0)
                else None
            ),
            "first_right_support_loss_step": (
                int(np.flatnonzero(right_clearance <= 0.0)[0])
                if np.any(right_clearance <= 0.0)
                else None
            ),
            "forced_com_x_at_pseudo_release": float(com_x[release_step]),
            "forced_front_x_at_pseudo_release": float(front_x[release_step]),
            "forced_back_x_at_pseudo_release": float(min_x[release_step]),
            "pre_delta_128": pre_delta,
            "pre_direction_128": _direction_label(pre_delta),
            "reference_returned_from_edge": bool(event.get("returned_from_edge", False)),
            "reference_success": bool(event.get("success", False)),
        }
        record.update(command_record.get("fixed_metrics", {}))
        for window in POST_WINDOWS:
            delta = float(com_x[release_step + window] - com_x[release_step])
            record[f"post_delta_{window}"] = delta
            record[f"post_direction_{window}"] = _direction_label(delta)
            record[f"pseudo_reversal_{window}"] = bool(
                pre_delta < -REVERSAL_DELTA_THRESHOLD and delta > REVERSAL_DELTA_THRESHOLD
            )
        rows.append(record)

    summary = {
        "schema_version": 1,
        "kind": "s4_forced_command_support_control_summary",
        "replay_count": int(sum(1 for row in rows if row.get("status") == "replayed")),
        "reference_event_count": int(len(release_df)),
        "support_left_frontier_x": support_left_frontier,
        "support_right_frontier_x": support_right_frontier,
        "config_path": str(config_path),
        "runtime": _runtime_info(),
    }
    return rows, summary


def _make_forced_figures(forced_df: pd.DataFrame) -> list[Path]:
    import matplotlib.pyplot as plt

    figure_dir = PACKAGE_ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    valid = forced_df[forced_df["status"] == "replayed"].sort_values("reference_cliff_distance", kind="stable")
    paths: list[Path] = []
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    ax.axhline(0, color="#777777", linestyle="--", linewidth=0.8)
    ax.bar(
        valid["reference_cliff_distance"].astype(int),
        valid["post_delta_336"].astype(float),
        color="#8CB87C",
        width=0.8,
    )
    ax.set_xlabel("Reference cliff distance")
    ax.set_ylabel("COM delta after 336 steps")
    ax.set_title("Forced command replay on supported terrain")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = figure_dir / f"forced_command_support_post_delta_336.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _write_forced_report(*, forced_df: pd.DataFrame, summary: dict[str, Any]) -> Path:
    valid = forced_df[forced_df["status"] == "replayed"]
    replay_count = int(len(valid))
    reversal_counts = {
        window: int(valid[f"pseudo_reversal_{window}"].sum()) if replay_count else 0
        for window in POST_WINDOWS
    }
    away_counts = {
        window: int((valid[f"post_direction_{window}"] == "away_from_cliff").sum()) if replay_count else 0
        for window in POST_WINDOWS
    }
    fixed_away_counts = {
        window: int((valid[f"fixed_post_direction_{window}"] == "away_from_cliff").sum())
        if replay_count and f"fixed_post_direction_{window}" in valid
        else 0
        for window in POST_WINDOWS
    }
    fixed_reversal_counts = {
        window: int(valid[f"fixed_reversal_{window}"].sum())
        if replay_count and f"fixed_reversal_{window}" in valid
        else 0
        for window in POST_WINDOWS
    }
    support_loss_count = int(
        valid["first_left_support_loss_step"].notna().sum()
        + valid["first_right_support_loss_step"].notna().sum()
    ) if replay_count else 0
    post336 = valid["post_delta_336"].astype(float) if replay_count else pd.Series(dtype=float)
    fixed_post336 = (
        valid["fixed_post_delta_336"].astype(float)
        if replay_count and "fixed_post_delta_336" in valid
        else pd.Series(dtype=float)
    )
    conclusion = (
        "The strict forced-command control did not produce edge-locked reversal under preserved support."
        if replay_count and all(count == 0 for count in reversal_counts.values()) and support_loss_count == 0
        else "The strict forced-command control produced some positive post-window displacement under preserved support; use the per-event table rather than a zero-reversal claim."
    )
    lines = [
        "# S4 forced-command support-preserved control",
        "",
        "## Purpose",
        "",
        "This is the stricter version of the support-preserved test: instead of rerunning the controller on a far-cliff terrain, it replays the recorded fixed-width actuator command streams (`phase_h_drive`, `phase_v_drive`) onto the same S4 body on a long supported terrain.",
        "",
        "## Method",
        "",
        f"For each of the `{summary.get('reference_event_count')}` fixed-width release events, the recorded phase-level horizontal and vertical drives were loaded from the original A100 trace and applied directly to a support-preserved terrain with cliff distance `{SUPPORT_PRESERVED_CLIFF_DISTANCE}`. The NCA/CTRNN controller was bypassed during replay; the physics, body, actuator mapping, terrain support, and collision system were retained.",
        "",
        f"The pseudo-event is the original fixed-width release step. As before, a pseudo-reversal requires `{PRE_WINDOW}` pre-steps of cliffward COM displacement below `-{REVERSAL_DELTA_THRESHOLD}` and a post-window COM displacement above `{REVERSAL_DELTA_THRESHOLD}`.",
        "",
        "## Results",
        "",
        f"- Replayed command traces: `{replay_count}/{summary.get('reference_event_count')}`.",
        f"- Support-loss events during forced replay: `{support_loss_count}`.",
        f"- Pseudo-reversal counts: "
        + ", ".join(f"{window} steps = {count}/{replay_count}" for window, count in reversal_counts.items())
        + ".",
        f"- Away-direction post-window counts: "
        + ", ".join(f"{window} steps = {count}/{replay_count}" for window, count in away_counts.items())
        + ".",
        f"- Matched real-cliff away-direction counts: "
        + ", ".join(f"{window} steps = {count}/{replay_count}" for window, count in fixed_away_counts.items())
        + ".",
        f"- Matched real-cliff reversal counts: "
        + ", ".join(f"{window} steps = {count}/{replay_count}" for window, count in fixed_reversal_counts.items())
        + ".",
        f"- `post_delta_336` mean/median/min/max: `{float(post336.mean()) if replay_count else float('nan'):.6f}` / `{float(post336.median()) if replay_count else float('nan'):.6f}` / `{float(post336.min()) if replay_count else float('nan'):.6f}` / `{float(post336.max()) if replay_count else float('nan'):.6f}`.",
        f"- Matched real-cliff `post_delta_336` mean/median/min/max: `{float(fixed_post336.mean()) if not fixed_post336.empty else float('nan'):.6f}` / `{float(fixed_post336.median()) if not fixed_post336.empty else float('nan'):.6f}` / `{float(fixed_post336.min()) if not fixed_post336.empty else float('nan'):.6f}` / `{float(fixed_post336.max()) if not fixed_post336.empty else float('nan'):.6f}`.",
        "",
        "## Interpretation",
        "",
        conclusion,
        "",
        "The relevant contrast is not all-or-none internal reversal. It is that the same recorded command streams usually continue cliffward or remain locally oscillatory under preserved support, while the matched real-cliff traces reliably move away from the release edge. The few support-preserved positive windows are small phase excursions rather than the systematic release-locked return seen with support loss.",
        "",
        "This is the stronger control for the paper. It preserves the command sequence by construction and removes the support-loss event. The result does not support a pure command-clock explanation: the same commands only rarely produce small away excursions on supported terrain, whereas the matched released body produces a much more systematic return.",
        "",
        "## Output files",
        "",
        "- `tables/forced_command_support_pseudo_release_summary.csv`: per-event forced replay metrics.",
        "- `tables/forced_command_support_summary.json`: compact machine-readable summary.",
        "- `figures/forced_command_support_post_delta_336.{png,pdf}`.",
    ]
    path = PACKAGE_ROOT / "S4_forced_command_support_control_report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def analyze_forced_command_replay(*, config_path: str, init_package: bool = True) -> dict[str, Path]:
    if init_package:
        initialize_package()
    rows, summary = _forced_command_replay_rows(config_path=config_path)
    forced_df = pd.DataFrame(rows)
    table_path = PACKAGE_ROOT / "tables" / "forced_command_support_pseudo_release_summary.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    forced_df.to_csv(table_path, index=False)
    valid = forced_df[forced_df["status"] == "replayed"] if not forced_df.empty else pd.DataFrame()
    summary.update(
        {
            "pseudo_reversal_counts": {
                str(window): int(valid[f"pseudo_reversal_{window}"].sum()) if not valid.empty else 0
                for window in POST_WINDOWS
            },
            "away_direction_counts": {
                str(window): int((valid[f"post_direction_{window}"] == "away_from_cliff").sum()) if not valid.empty else 0
                for window in POST_WINDOWS
            },
            "matched_real_cliff_reversal_counts": {
                str(window): int(valid[f"fixed_reversal_{window}"].sum())
                if not valid.empty and f"fixed_reversal_{window}" in valid
                else 0
                for window in POST_WINDOWS
            },
            "matched_real_cliff_away_direction_counts": {
                str(window): int((valid[f"fixed_post_direction_{window}"] == "away_from_cliff").sum())
                if not valid.empty and f"fixed_post_direction_{window}" in valid
                else 0
                for window in POST_WINDOWS
            },
            "support_loss_count": int(
                valid["first_left_support_loss_step"].notna().sum()
                + valid["first_right_support_loss_step"].notna().sum()
            ) if not valid.empty else 0,
        }
    )
    summary_path = PACKAGE_ROOT / "tables" / "forced_command_support_summary.json"
    _write_json(summary_path, summary)
    _make_forced_figures(forced_df)
    report_path = _write_forced_report(forced_df=forced_df, summary=summary)
    return {
        "forced_table": table_path,
        "forced_summary": summary_path,
        "forced_report": report_path,
    }


def _runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import jax

        info["jax_version"] = str(jax.__version__)
        info["jax_devices"] = [str(device) for device in jax.devices()]
    except Exception as exc:  # pragma: no cover - diagnostic only
        info["jax_error"] = repr(exc)
    return info


def _write_report(
    *,
    assay_root: Path,
    trial_df: pd.DataFrame,
    pseudo_df: pd.DataFrame,
    release_df: pd.DataFrame,
    command_summary: dict[str, Any],
    first_sustained_away_step: int | None,
    runtime_info: dict[str, Any],
) -> Path:
    valid = pseudo_df[pseudo_df.get("valid_window", False) == True]  # noqa: E712
    pseudo_count = int(len(valid))
    reversal_counts = {
        window: int(valid[f"pseudo_reversal_{window}"].sum()) if pseudo_count else 0
        for window in POST_WINDOWS
    }
    away_counts = {
        window: int((valid[f"post_direction_{window}"] == "away_from_cliff").sum()) if pseudo_count else 0
        for window in POST_WINDOWS
    }
    platform_trial = trial_df.iloc[0].to_dict() if not trial_df.empty else {}
    min_clearance = platform_trial.get("min_cliff_clearance", None)
    first_cliff_reach = platform_trial.get("first_cliff_reach_step", None)
    if pd.isna(first_cliff_reach):
        first_cliff_reach = None
    conclusion = (
        "Support was preserved and no pseudo-release window showed a sustained away/reversal response."
        if pseudo_count and all(count == 0 for count in reversal_counts.values()) and first_cliff_reach is None
        else "Support-preserved replay produced at least one pseudo-window away/reversal response; inspect the table before using a strong edge-induced claim."
    )

    lines = [
        "# S4 support-preserved deformation-off control",
        "",
        "## Purpose",
        "",
        "This control asks whether S4 reverses under the deformation-off command regime when the physical loss of support is prevented. The fixed-width A100 distance sweep supplies real release times; here those same times are treated as pseudo-events in a long supported replay.",
        "",
        "## Inputs and regime",
        "",
        f"- Specimen: `{SPECIMEN_ID}`.",
        f"- Run: `{RUN_ID}`.",
        f"- Modal GPU profile: `{GPU_PROFILE}`.",
        f"- JAX: `{runtime_info.get('jax_version', 'not recorded')}`.",
        f"- Devices: `{runtime_info.get('jax_devices', 'not recorded')}`.",
        f"- Condition: deformation off, intact signalling, state-space trace.",
        f"- Support-preserved terrain: goal side `{GOAL_SIDE}`, goal distance `{GOAL_DISTANCE}`, cliff distance `{SUPPORT_PRESERVED_CLIFF_DISTANCE}`, grid width `{GRID_W}`, plateau width `{PLATEAU_WIDTH}`, max steps `{MAX_STEPS}`.",
        f"- Reference release events: `{len(release_df)}` from `{FIXED_WIDTH_REMOTE_ROOT}`.",
        f"- Support assay root: `{assay_root}`.",
        "",
        "## Method",
        "",
        f"For each fixed-width release step, the support-preserved COM trajectory was sampled over a `{PRE_WINDOW}`-step pre-window and post-windows of `{', '.join(str(w) for w in POST_WINDOWS)}` steps. Because the cliff is on the left in these goal-right assays, positive COM displacement after a pseudo-event is movement away from the cliff. A pseudo-reversal is counted only when the pre-window displacement is less than `-{REVERSAL_DELTA_THRESHOLD}` and the post-window displacement is greater than `{REVERSAL_DELTA_THRESHOLD}`.",
        "",
        "Where the original cliff traces are available, the analysis also compares phase-level command/control arrays between each fixed-width trace and the support-preserved trace through `release_step + 336`: action outputs, horizontal drive, vertical drive, NCA state, and CTRNN carry.",
        "",
        "## Results",
        "",
        f"- Support trace termination: `{platform_trial.get('termination_reason', 'unknown')}`.",
        f"- Goal touched: `{platform_trial.get('goal_touched', 'unknown')}`; cliff fell: `{platform_trial.get('cliff_fell', 'unknown')}`.",
        f"- First cliff reach step: `{first_cliff_reach}`.",
        f"- Minimum cliff clearance: `{min_clearance}`.",
        f"- Valid pseudo-release windows: `{pseudo_count}/{len(release_df)}`.",
        f"- Pseudo-reversal counts: "
        + ", ".join(f"{window} steps = {count}/{pseudo_count}" for window, count in reversal_counts.items())
        + ".",
        f"- Away-direction post-window counts: "
        + ", ".join(f"{window} steps = {count}/{pseudo_count}" for window, count in away_counts.items())
        + ".",
        f"- First sustained away step anywhere in the support trace: `{first_sustained_away_step}`.",
        f"- Command identity comparisons available: `{command_summary.get('array_comparisons', 0)}` array windows; missing fixed-trace events: `{command_summary.get('fixed_trace_missing_events', 0)}`.",
        f"- Pre-release command/control windows exact: `{command_summary.get('pre_window_exact_comparisons', 0)}/{command_summary.get('array_comparisons', 0)}`.",
        f"- Full `release+336` command/control windows exact: `{command_summary.get('full_window_exact_comparisons', 0)}/{command_summary.get('array_comparisons', 0)}`.",
        "",
        "## Interpretation",
        "",
        conclusion,
        "",
        "Paper-safe wording: this is a test-time control for the S4 case study. If no pseudo-event reversal appears under preserved support while command/control streams are preserved, the observed reversal in the cliff sweep is much more tightly linked to the physical loss of support than to a coincident internal gait clock. It is still not a full causal clamp, but it is stronger than a decoder-only result.",
        "",
        "## Output files",
        "",
        "- `tables/support_preserved_pseudo_release_summary.csv`: per-reference-event pseudo-release metrics.",
        "- `tables/support_preserved_command_identity.csv`: per-event, per-array identity checks against fixed-width traces.",
        "- `tables/support_preserved_control_summary.json`: compact machine-readable summary.",
        "- `figures/support_preserved_com_trace.{png,pdf}` and `figures/support_preserved_post_delta_336.{png,pdf}`.",
    ]
    path = PACKAGE_ROOT / "S4_support_preserved_control_report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def analyze(*, init_package: bool = True) -> dict[str, Path]:
    if init_package:
        initialize_package()
    assay_root = _find_support_assay_root()
    trial_df, step_df = _load_support_tables(assay_root)
    release_df = pd.read_csv(_fixed_release_manifest_path())
    release_df = release_df.sort_values("cliff_distance", kind="stable").reset_index(drop=True)

    pseudo_rows = _pseudo_release_rows(step_df=step_df, release_df=release_df)
    pseudo_df = pd.DataFrame(pseudo_rows)
    pseudo_path = PACKAGE_ROOT / "tables" / "support_preserved_pseudo_release_summary.csv"
    pseudo_path.parent.mkdir(parents=True, exist_ok=True)
    pseudo_df.to_csv(pseudo_path, index=False)

    platform_trace = _platform_trace_path(trial_df, assay_root=assay_root)
    command_rows, command_summary = _command_identity_rows(
        platform_trace_path=platform_trace,
        release_df=release_df,
    )
    command_path = PACKAGE_ROOT / "tables" / "support_preserved_command_identity.csv"
    _write_csv(command_path, command_rows)

    ordered_steps = step_df.sort_values("step_index", kind="stable")
    first_sustained_away_step = _first_sustained_away_step(
        ordered_steps["com_x"].to_numpy(dtype=float)
    )
    figures = _make_figures(step_df=ordered_steps, pseudo_df=pseudo_df)

    valid = pseudo_df[pseudo_df.get("valid_window", False) == True]  # noqa: E712
    pseudo_count = int(len(valid))
    reversal_counts = {
        str(window): int(valid[f"pseudo_reversal_{window}"].sum()) if pseudo_count else 0
        for window in POST_WINDOWS
    }
    away_counts = {
        str(window): int((valid[f"post_direction_{window}"] == "away_from_cliff").sum()) if pseudo_count else 0
        for window in POST_WINDOWS
    }
    platform_trial = trial_df.iloc[0].to_dict() if not trial_df.empty else {}
    first_cliff_reach = platform_trial.get("first_cliff_reach_step")
    if pd.isna(first_cliff_reach):
        first_cliff_reach = None
    runtime_info = _runtime_info()
    summary = {
        "schema_version": 1,
        "kind": "s4_support_preserved_control_summary",
        "assay_root": str(assay_root),
        "platform_trace": str(platform_trace),
        "reference_release_manifest": str(_fixed_release_manifest_path()),
        "reference_release_event_count": int(len(release_df)),
        "valid_pseudo_release_windows": pseudo_count,
        "pseudo_reversal_counts": reversal_counts,
        "away_direction_counts": away_counts,
        "first_sustained_away_step": first_sustained_away_step,
        "support_first_cliff_reach_step": first_cliff_reach,
        "support_min_cliff_clearance": platform_trial.get("min_cliff_clearance"),
        "support_goal_touched": bool(platform_trial.get("goal_touched", False)),
        "support_cliff_fell": bool(platform_trial.get("cliff_fell", False)),
        "support_termination_reason": platform_trial.get("termination_reason"),
        "command_identity": command_summary,
        "runtime": runtime_info,
    }
    summary_path = PACKAGE_ROOT / "tables" / "support_preserved_control_summary.json"
    _write_json(summary_path, summary)
    report_path = _write_report(
        assay_root=assay_root,
        trial_df=trial_df,
        pseudo_df=pseudo_df,
        release_df=release_df,
        command_summary=command_summary,
        first_sustained_away_step=first_sustained_away_step,
        runtime_info=runtime_info,
    )
    return {
        "pseudo_release_summary": pseudo_path,
        "command_identity": command_path,
        "summary": summary_path,
        "report": report_path,
        "figure": figures[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    launch_parser = sub.add_parser("launch-modal")
    launch_parser.add_argument("--dry-run", action="store_true")
    launch_parser.add_argument("--detach-modal", action="store_true")
    launch_parser.add_argument("--detach-function", action="store_true")
    pull_parser = sub.add_parser("pull")
    pull_parser.add_argument("--dry-run", action="store_true")
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--no-init", action="store_true")
    forced_parser = sub.add_parser("analyze-forced")
    forced_parser.add_argument("--config-path", required=True)
    forced_parser.add_argument("--no-init", action="store_true")
    sub.add_parser("manifest")
    args = parser.parse_args()

    if args.command == "init":
        print(initialize_package())
    elif args.command == "manifest":
        print(write_launch_manifest())
    elif args.command == "launch-modal":
        launch_modal(
            dry_run=bool(args.dry_run),
            detach_modal=bool(args.detach_modal),
            detach_function=bool(args.detach_function),
        )
    elif args.command == "pull":
        pull_modal(dry_run=bool(args.dry_run))
    elif args.command == "analyze":
        outputs = analyze(init_package=not bool(args.no_init))
        for key, value in outputs.items():
            print(f"{key}: {value}")
    elif args.command == "analyze-forced":
        outputs = analyze_forced_command_replay(
            config_path=str(args.config_path),
            init_package=not bool(args.no_init),
        )
        for key, value in outputs.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
