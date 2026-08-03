#!/usr/bin/env python
"""Launch and verify the S4 cross-hardware full state-space probe.

The package produced by this script is intentionally self-contained:

    paper_data/preprint/s4_cross_hardware_state_space_20260614/

It copies the S4 specimen artifacts into the package, writes launch manifests,
and stores local/Modal trace outputs under runtime-specific directories.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SENSING_EDGES_ROOT = REPO_ROOT / "sensing_edges"
if str(SENSING_EDGES_ROOT) not in sys.path:
    sys.path.insert(0, str(SENSING_EDGES_ROOT))
PACKAGE_ROOT = REPO_ROOT / "paper_data" / "preprint" / "s4_cross_hardware_state_space_20260614"
RUN_ID = "paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed137_e4fcc2fb"
SPECIMEN_ID = "bin_03_rank_00_cell_6-0-3_flat_0423"
SOURCE_STAGE_ROOT = REPO_ROOT / "run_specimens" / RUN_ID / "stages" / "gen_0099"
SOURCE_SPECIMEN_DIR = SOURCE_STAGE_ROOT / f"specimen_{SPECIMEN_ID}"
PACKAGE_STAGE_ROOT = PACKAGE_ROOT / "specimen" / "stages" / "gen_0099"
PACKAGE_SPECIMEN_DIR = PACKAGE_STAGE_ROOT / f"specimen_{SPECIMEN_ID}"
PACKAGE_RUN_CONTEXT_DIR = PACKAGE_ROOT / "specimen" / "run_context" / RUN_ID
REMOTE_ANALYSIS_ROOT_PREFIX = "/outputs/preprint_s4_cross_hardware_state_space_20260614"
GPU_PROFILES = ("a100-40gb", "a100-80gb", "h100", "l4", "t4")
TRACE_ASSAYS = ("state_space_trace", "state_space_trace_deformation_off")
SIDES = ("left", "right")

NATIVE_GRID = ((1, 1), (2, 1), (3, 2), (4, 2), (5, 3), (6, 3))
EXTENDED_GRID = ((7, 4), (8, 4), (9, 5), (10, 5))
EDGE_GRID = NATIVE_GRID + EXTENDED_GRID
PAIR_STEP_BUDGETS = {
    (1, 1): 4000,
    (2, 1): 4000,
    (3, 2): 4000,
    (4, 2): 4000,
    (5, 3): 4000,
    (6, 3): 4000,
    (7, 4): 5000,
    (8, 4): 5000,
    (9, 5): 7500,
    (10, 5): 7500,
}


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _omega_pairs(pairs: Iterable[tuple[int, int]]) -> str:
    return "[" + ",".join(
        f"{{\"goal_distance\":{goal},\"cliff_distance\":{cliff}}}"
        for goal, cliff in pairs
    ) + "]"


def _omega_pair_budgets(pairs: Iterable[tuple[int, int]]) -> str:
    return "[" + ",".join(
        f"{{\"goal_distance\":{goal},\"cliff_distance\":{cliff},\"max_steps\":{PAIR_STEP_BUDGETS[(goal, cliff)]}}}"
        for goal, cliff in pairs
    ) + "]"


def _omega_list(values: Iterable[str]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def _remote_analysis_root(profile: str) -> str:
    return f"{REMOTE_ANALYSIS_ROOT_PREFIX}_{profile}"


def _modal_volume_path(profile: str) -> str:
    return _remote_analysis_root(profile).removeprefix("/outputs")


def _expected_trial_count() -> int:
    return len(EDGE_GRID) * len(SIDES)


def _expected_trace_count() -> int:
    return _expected_trial_count() * len(TRACE_ASSAYS)


def _override_tokens(*, analysis_root: str, specimens_output_root: str) -> list[str]:
    return [
        f"eval.post_run.analysis_root={analysis_root}",
        "eval.post_run.state_space_surfaces=[native_core,long_distance_generalization]",
        f"eval.post_run.native_pairs={_omega_pairs(NATIVE_GRID)}",
        f"eval.post_run.long_distance_pairs={_omega_pairs(EXTENDED_GRID)}",
        f"eval.post_run.long_distance_pair_max_steps={_omega_pair_budgets(EXTENDED_GRID)}",
        "eval.post_run.native_max_steps=4000",
        "eval.post_run.max_steps=7500",
        "eval.post_run.state_space_deformation_condition=intact",
        "eval.post_run.state_space_signaling_condition=intact",
        "eval.post_run.fast_ablation_eval=false",
        f"specimens.output_root={specimens_output_root}",
    ]


def _override_csv(*, analysis_root: str, specimens_output_root: str) -> str:
    return ",".join(_override_tokens(analysis_root=analysis_root, specimens_output_root=specimens_output_root))


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
    """Create the self-contained package and copy S4 specimen artifacts."""
    for rel in ("specimen", "manifests", "local", "modal", "tables", "notes"):
        (PACKAGE_ROOT / rel).mkdir(parents=True, exist_ok=True)

    stage_files = (
        "manifest.json",
        "selected_specimens.json",
        "selected_genotypes.npz",
        "terrain_manifest.json",
    )
    for name in stage_files:
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
    stage_manifest = PACKAGE_STAGE_ROOT / "manifest.json"
    if stage_manifest.exists():
        _copy_required_file(stage_manifest, PACKAGE_RUN_CONTEXT_DIR / "manifest.json")
    _copy_required_file(PACKAGE_SPECIMEN_DIR / "config_snapshot.yaml", PACKAGE_RUN_CONTEXT_DIR / "resolved_config.yaml")

    package_manifest = {
        "schema_version": 1,
        "kind": "s4_cross_hardware_state_space_package",
        "created_for": "S4 cross-hardware full state-space probe",
        "package_root": str(PACKAGE_ROOT),
        "source_stage_root": str(SOURCE_STAGE_ROOT),
        "source_specimen_dir": str(SOURCE_SPECIMEN_DIR),
        "run_dir_name": RUN_ID,
        "specimen_id": SPECIMEN_ID,
        "conditions": list(TRACE_ASSAYS),
        "surfaces": ["native_core", "long_distance_generalization"],
        "native_pairs": [{"goal_distance": g, "cliff_distance": c} for g, c in NATIVE_GRID],
        "extended_pairs": [{"goal_distance": g, "cliff_distance": c} for g, c in EXTENDED_GRID],
        "step_budgets": {
            f"{goal},{cliff}": int(PAIR_STEP_BUDGETS[(goal, cliff)])
            for goal, cliff in EDGE_GRID
        },
        "expected": {
            "trials_per_assay": _expected_trial_count(),
            "trace_npz_per_runtime": _expected_trace_count(),
            "assays_per_runtime": list(TRACE_ASSAYS),
        },
        "copied_paths": {
            "package_stage_root": str(PACKAGE_STAGE_ROOT),
            "package_specimen_dir": str(PACKAGE_SPECIMEN_DIR),
            "package_run_context_dir": str(PACKAGE_RUN_CONTEXT_DIR),
        },
    }
    _write_json(PACKAGE_ROOT / "manifests" / "package_manifest.json", package_manifest)
    note_path = PACKAGE_ROOT / "notes" / "README.md"
    note_path.write_text(
        "# S4 cross-hardware full state-space package\n\n"
        "This directory contains the expanded S4 goal-cliff state-space probe.\n\n"
        "- Specimen artifacts are copied under `specimen/`.\n"
        "- Local CPU outputs should be written under `local/<runtime>/`.\n"
        "- Pulled Modal GPU outputs should be written under `modal/<profile>/`.\n"
        "- Summary tables and verification outputs are written under `tables/` and `notes/`.\n\n"
        "Expected per runtime: 2 trace assays x 20 mirrored goal-cliff episodes = 40 trace NPZ files.\n",
    )
    return PACKAGE_ROOT


def modal_launch_argv(profile: str) -> list[str]:
    item = _s4_item()
    analysis_root = _remote_analysis_root(profile)
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
        ",".join(TRACE_ASSAYS),
        "--override",
        _override_csv(
            analysis_root=analysis_root,
            specimens_output_root=item["specimens_output_root"],
        ),
        "--gpu-profile",
        profile,
        "--detach-launch",
    ]


def pull_argv(profile: str) -> list[str]:
    return [
        "uv",
        "run",
        "--python",
        ".venv/bin/python",
        "modal",
        "volume",
        "get",
        "se-outputs",
        _modal_volume_path(profile),
        str(PACKAGE_ROOT / "modal" / profile),
    ]


def write_launch_manifest(profiles: tuple[str, ...] = GPU_PROFILES) -> Path:
    initialize_package()
    jobs = []
    for profile in profiles:
        launch = modal_launch_argv(profile)
        pull = pull_argv(profile)
        jobs.append(
            {
                "profile": profile,
                "remote_analysis_root": _remote_analysis_root(profile),
                "modal_volume_path": _modal_volume_path(profile),
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
            "kind": "s4_cross_hardware_state_space_modal_launch_manifest",
            "profiles": list(profiles),
            "run_dir_name": RUN_ID,
            "specimen_id": SPECIMEN_ID,
            "assays": list(TRACE_ASSAYS),
            "expected": {
                "trials_per_assay": _expected_trial_count(),
                "trace_npz_per_runtime": _expected_trace_count(),
            },
            "jobs": jobs,
        },
    )
    return path


def run_local(*, runtime_id: str, x64: bool, overwrite: bool = False) -> Path:
    initialize_package()
    import jax
    from omegaconf import OmegaConf
    import jaxlib
    import pyarrow
    from sensing_edges.logging_utils import configure_logging
    from sensing_edges.post_run import run_post_run_for_stored_specimen
    from sensing_edges.runtime_config import compose_stored_config, resolve_runtime_config

    jax.config.update("jax_platform_name", "cpu")
    if x64:
        jax.config.update("jax_enable_x64", True)

    configure_logging("INFO")
    analysis_root = PACKAGE_ROOT / "local" / runtime_id
    if analysis_root.exists() and not overwrite:
        raise FileExistsError(
            f"Local runtime output already exists: {analysis_root}. "
            "Choose a new --runtime-id or pass --overwrite."
        )
    if analysis_root.exists() and overwrite:
        shutil.rmtree(analysis_root)
    _write_json(
        PACKAGE_ROOT / "manifests" / f"{runtime_id}_runtime.json",
        {
            "runtime_id": runtime_id,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "pyarrow": pyarrow.__version__,
            "default_backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "x64_requested": bool(x64),
        },
    )
    cfg = compose_stored_config(PACKAGE_SPECIMEN_DIR / "config_snapshot.yaml", preset="paper_run")
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.from_dotlist(
            _override_tokens(
                analysis_root=str(analysis_root),
                specimens_output_root=str(PACKAGE_ROOT / "specimen" / "stages"),
            )
        ),
    )
    runtime_resolution = resolve_runtime_config(cfg)
    result = run_post_run_for_stored_specimen(
        cfg=cfg,
        runtime_resolution=runtime_resolution,
        run_dir=PACKAGE_RUN_CONTEXT_DIR,
        specimen_root=PACKAGE_STAGE_ROOT,
        specimen_id=SPECIMEN_ID,
        assays=set(TRACE_ASSAYS),
    )
    _write_json(
        PACKAGE_ROOT / "manifests" / f"{runtime_id}_local_result.json",
        {
            "runtime_id": runtime_id,
            "x64_requested": bool(x64),
            "index_path": str(result.index_path),
            "specimen_dir": str(result.specimen_dir),
            "assays": dict(result.assay_dirs),
            "status": str(result.status),
        },
    )
    return analysis_root


def _bool_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        import pandas as pd

        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _classify_trial(row: Any) -> str:
    success = _bool_value(row.get("success"))
    goal_touched = _bool_value(row.get("goal_touched"))
    cliff_fell = _bool_value(row.get("cliff_fell"))
    returned = _bool_value(row.get("returned_from_edge"))
    first_edge = row.get("first_cliff_reach_step")
    edge_reached = first_edge is not None and str(first_edge).lower() != "nan"
    if cliff_fell:
        return "fall"
    if success and goal_touched and edge_reached and returned:
        return "full_reverse_and_goal"
    if success and goal_touched and not edge_reached:
        return "direct_goal_no_edge"
    if success and goal_touched:
        return "goal_after_edge_no_return"
    if edge_reached and returned:
        return "returned_no_goal"
    if edge_reached:
        return "cliff_stop_only"
    return "stall_no_edge"


def _runtime_from_path(path: Path) -> str:
    parts = path.parts
    if "local" in parts:
        index = parts.index("local")
        if index + 1 < len(parts):
            return f"local:{parts[index + 1]}"
    if "modal" in parts:
        index = parts.index("modal")
        if index + 1 < len(parts):
            return f"modal:{parts[index + 1]}"
    return "unknown"


def _find_assay_roots(input_root: Path) -> list[Path]:
    roots: list[Path] = []
    for assay in TRACE_ASSAYS:
        roots.extend(sorted(path.parent for path in input_root.glob(f"**/{assay}/trial_manifest.csv")))
    return sorted(set(roots))


def summarize(input_roots: list[Path] | None = None) -> dict[str, Path]:
    import pandas as pd

    initialize_package()
    roots = input_roots or [PACKAGE_ROOT / "local", PACKAGE_ROOT / "modal"]
    assay_roots: list[Path] = []
    for root in roots:
        if root.exists():
            assay_roots.extend(_find_assay_roots(root))

    trial_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for assay_root in sorted(set(assay_roots)):
        assay = assay_root.name
        runtime = _runtime_from_path(assay_root)
        trial_path = assay_root / "trial_manifest.csv"
        trial_df = pd.read_csv(trial_path)
        traces_dir = assay_root / "traces"
        present_trace_names = {path.name for path in traces_dir.glob("*.npz")} if traces_dir.exists() else set()
        expected_trace_names = set()
        for _, row in trial_df.iterrows():
            trace_name = Path(str(row.get("trace_npz", ""))).name
            if trace_name:
                expected_trace_names.add(trace_name)
            record = row.to_dict()
            record.update(
                {
                    "runtime": runtime,
                    "assay": assay,
                    "endpoint_class": _classify_trial(record),
                    "assay_root": str(assay_root),
                }
            )
            trial_rows.append(record)

        required_files = {
            "manifest_json": assay_root / "manifest.json",
            "summary_json": assay_root / "summary.json",
            "trial_manifest_csv": assay_root / "trial_manifest.csv",
            "step_manifest_csv": assay_root / "step_manifest.csv",
            "step_manifest_parquet": assay_root / "step_manifest.parquet",
            "static_arrays_npz": assay_root / "static_arrays.npz",
            "traces_dir": traces_dir,
        }
        missing_trace_names = sorted(expected_trace_names - present_trace_names)
        extra_trace_names = sorted(present_trace_names - expected_trace_names)
        inventory_rows.append(
            {
                "runtime": runtime,
                "assay": assay,
                "assay_root": str(assay_root),
                "trial_count": int(len(trial_df)),
                "expected_trial_count": _expected_trial_count(),
                "trace_count": int(len(present_trace_names)),
                "expected_trace_count": int(len(expected_trace_names) or _expected_trial_count()),
                "missing_trace_count": int(len(missing_trace_names)),
                "extra_trace_count": int(len(extra_trace_names)),
                "missing_trace_names": ";".join(missing_trace_names),
                "extra_trace_names": ";".join(extra_trace_names),
                **{f"has_{name}": path.exists() for name, path in required_files.items()},
            }
        )

    if trial_rows:
        trial_all = pd.DataFrame(trial_rows)
        group_cols = ["runtime", "assay", "surface_id", "distance_pair", "goal_distance", "cliff_distance"]
        for keys, group in trial_all.groupby(group_cols, dropna=False):
            row = dict(zip(group_cols, keys, strict=True))
            counts = group["endpoint_class"].value_counts().to_dict()
            row.update(
                {
                    "n": int(len(group)),
                    "success_count": int(group["success"].map(_bool_value).sum()),
                    "fall_count": int(group["cliff_fell"].map(_bool_value).sum()),
                    "edge_reach_count": int(group["first_cliff_reach_step"].notna().sum()),
                    "returned_from_edge_count": int(group["returned_from_edge"].map(_bool_value).sum()),
                }
            )
            for endpoint in sorted(set(trial_all["endpoint_class"])):
                row[f"endpoint_{endpoint}"] = int(counts.get(endpoint, 0))
            summary_rows.append(row)
    else:
        trial_all = pd.DataFrame()

    tables_dir = PACKAGE_ROOT / "tables"
    trial_path = tables_dir / "hardware_trial_matrix.csv"
    inventory_path = tables_dir / "hardware_trace_inventory.csv"
    summary_path = tables_dir / "hardware_endpoint_summary.csv"
    pd.DataFrame(trial_rows).to_csv(trial_path, index=False)
    pd.DataFrame(inventory_rows).to_csv(inventory_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    note_path = PACKAGE_ROOT / "notes" / "cross_hardware_state_space_status.md"
    completed = 0
    if inventory_rows:
        inventory_df = pd.DataFrame(inventory_rows)
        completed = int(
            (
                (inventory_df["trial_count"] == inventory_df["expected_trial_count"])
                & (inventory_df["trace_count"] == inventory_df["expected_trace_count"])
                & (inventory_df["missing_trace_count"] == 0)
                & inventory_df.filter(like="has_").all(axis=1)
            ).sum()
        )
    note_path.write_text(
        "# S4 cross-hardware state-space status\n\n"
        f"Package root: `{PACKAGE_ROOT}`\n\n"
        f"Assay roots found: {len(assay_roots)}\n\n"
        f"Complete assay inventories: {completed}/{len(inventory_rows)}\n\n"
        "Expected per runtime: 2 assays, 20 trials per assay, 40 trace NPZ files total.\n\n"
        "Tables:\n"
        f"- `{trial_path}`\n"
        f"- `{inventory_path}`\n"
        f"- `{summary_path}`\n",
    )
    return {
        "trial_matrix": trial_path,
        "trace_inventory": inventory_path,
        "endpoint_summary": summary_path,
        "status_note": note_path,
    }


def _profiles_from_arg(raw: str) -> tuple[str, ...]:
    if str(raw).strip().lower() == "all":
        return GPU_PROFILES
    profiles = tuple(token.strip().lower() for token in str(raw).split(",") if token.strip())
    invalid = [profile for profile in profiles if profile not in GPU_PROFILES]
    if invalid:
        raise ValueError(f"Unknown GPU profile(s): {invalid}; expected one of {GPU_PROFILES}")
    return profiles


def command_init_package(_args: argparse.Namespace) -> int:
    print(initialize_package())
    return 0


def command_write_launch_manifest(args: argparse.Namespace) -> int:
    path = write_launch_manifest(_profiles_from_arg(args.profiles))
    print(path)
    return 0


def command_print_launch_commands(args: argparse.Namespace) -> int:
    initialize_package()
    for profile in _profiles_from_arg(args.profiles):
        print(_shell_join(modal_launch_argv(profile)))
    return 0


def command_print_pull_commands(args: argparse.Namespace) -> int:
    initialize_package()
    for profile in _profiles_from_arg(args.profiles):
        print(_shell_join(pull_argv(profile)))
    return 0


def command_run_local(args: argparse.Namespace) -> int:
    runtime_id = str(args.runtime_id or ("local_cpu_x64" if args.x64 else "local_cpu"))
    output = run_local(runtime_id=runtime_id, x64=bool(args.x64), overwrite=bool(args.overwrite))
    print(output)
    return 0


def command_summarize(args: argparse.Namespace) -> int:
    roots = [Path(value).resolve() for value in args.input_root] if args.input_root else None
    outputs = summarize(roots)
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


def command_launch_modal(args: argparse.Namespace) -> int:
    initialize_package()
    launched: list[dict[str, Any]] = []
    for profile in _profiles_from_arg(args.profiles):
        argv = modal_launch_argv(profile)
        if args.dry_run:
            print(_shell_join(argv))
            continue
        proc = subprocess.run(argv, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
        launched.append(
            {
                "profile": profile,
                "returncode": int(proc.returncode),
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "command": _shell_join(argv),
            }
        )
        print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="")
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)
    if launched:
        _write_json(PACKAGE_ROOT / "manifests" / "modal_launch_results.json", {"launched": launched})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-package")
    p_init.set_defaults(func=command_init_package)

    p_manifest = sub.add_parser("write-launch-manifest")
    p_manifest.add_argument("--profiles", default="all")
    p_manifest.set_defaults(func=command_write_launch_manifest)

    p_print = sub.add_parser("print-launch-commands")
    p_print.add_argument("--profiles", default="all")
    p_print.set_defaults(func=command_print_launch_commands)

    p_pull = sub.add_parser("print-pull-commands")
    p_pull.add_argument("--profiles", default="all")
    p_pull.set_defaults(func=command_print_pull_commands)

    p_launch = sub.add_parser("launch-modal")
    p_launch.add_argument("--profiles", default="all")
    p_launch.add_argument("--dry-run", action="store_true")
    p_launch.set_defaults(func=command_launch_modal)

    p_local = sub.add_parser("run-local")
    p_local.add_argument("--runtime-id", default=None)
    p_local.add_argument("--x64", action="store_true")
    p_local.add_argument("--overwrite", action="store_true")
    p_local.set_defaults(func=command_run_local)

    p_summary = sub.add_parser("summarize")
    p_summary.add_argument("--input-root", action="append", default=[])
    p_summary.set_defaults(func=command_summarize)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
