#!/usr/bin/env python
"""Prepare Modal state-space trace jobs for native signalling ablations.

The endpoint post-run assay records enough to summarize success/fall/stall, but
not enough to measure tortuosity, locomotor phase, or contraction rhythms. This
script prepares an isolated A100 batched state-space trace packet for the same
eight specimens and the six native-core signalling conditions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
from typing import Any

from native_core_ablation import (
    NATIVE_PAIRS,
    SIDES,
    SIGNALING_CONDITIONS,
    SPECIMENS,
    STEP_BUDGETS,
    SURFACE_PRESETS,
    state_space_override_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "sensing_edges"
    / "tmp"
    / "preprint_native_signaling_state_space_v1"
)
DEFAULT_ANALYSIS_ROOT = "/outputs/preprint_native_signaling_state_space_v1"


def _shell_join(argv: list[str]) -> str:
    return shlex.join(argv)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _launch_argv(
    *,
    item: dict[str, str],
    analysis_root: str,
    signaling_condition: str,
) -> list[str]:
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
        "state_space_trace",
        "--override",
        state_space_override_csv(
            specimens_output_root=item["specimens_output_root"],
            analysis_root=f"{analysis_root}/{signaling_condition}",
            deformation_condition="intact",
            signaling_condition=signaling_condition,
            state_space_surfaces=SURFACE_PRESETS["native_core"],
        ),
        "--gpu-profile",
        "a100-40gb",
        "--detach-launch",
    ]


def _expected_trace_count(condition_count: int) -> dict[str, int]:
    trials_per_specimen = len(NATIVE_PAIRS) * len(SIDES)
    return {
        "specimen_count": len(SPECIMENS),
        "condition_count": condition_count,
        "trials_per_specimen_per_condition": trials_per_specimen,
        "traces_per_condition": len(SPECIMENS) * trials_per_specimen,
        "total_traces": len(SPECIMENS) * trials_per_specimen * condition_count,
    }


def write_manifest(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = tuple(args.conditions or SIGNALING_CONDITIONS)
    unknown = sorted(set(conditions) - set(SIGNALING_CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown signalling condition(s): {', '.join(unknown)}")

    jobs: list[dict[str, Any]] = []
    for condition in conditions:
        for item in SPECIMENS:
            argv = _launch_argv(
                item=dict(item),
                analysis_root=str(args.analysis_root),
                signaling_condition=condition,
            )
            jobs.append(
                {
                    "condition_id": condition,
                    "group_id": item["group_id"],
                    "run_id": item["run_id"],
                    "specimen_id": item["specimen_id"],
                    "specimens_output_root": item["specimens_output_root"],
                    "analysis_root": f"{args.analysis_root}/{condition}",
                    "launch_argv": argv,
                    "launch_command": _shell_join(argv),
                }
            )

    manifest_path = output_dir / "launch_manifest_signaling_state_space.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "kind": "preprint_native_signaling_state_space_one_specimen_launch_manifest",
            "purpose": (
                "Native-core state-space traces for eight-specimen signalling "
                "ablation phenotype analysis: tortuosity, contraction phase, "
                "drive rhythm, and per-cell/state comparisons."
            ),
            "analysis_root": str(args.analysis_root),
            "modal_gpu": "A100-40GB",
            "launch_mode": "one_specimen_per_modal_function",
            "batched_state_space_supported": False,
            "assays": ["state_space_trace"],
            "surfaces": list(SURFACE_PRESETS["native_core"]),
            "deformation_condition": "intact",
            "signaling_conditions": list(conditions),
            "native_pairs": [
                {"goal_distance": goal, "cliff_distance": cliff}
                for goal, cliff in NATIVE_PAIRS
            ],
            "goal_sides": list(SIDES),
            "step_budgets": {"native_core": STEP_BUDGETS["native_core"]},
            "expected": _expected_trace_count(len(conditions)),
            "trace_contents": {
                "trajectory": [
                    "plant_positions",
                    "plant_velocities",
                    "cell_corner_positions",
                    "step_manifest.csv",
                    "step_manifest.parquet",
                ],
                "mechanical": [
                    "spring_rest_length",
                    "spring_current_length",
                    "spring_init_rest_length",
                    "phase_post_spring_rest_length",
                ],
                "controller": [
                    "nca_state",
                    "ctrnn_carry",
                    "action_outputs",
                    "phase_nca_state",
                    "phase_ctrnn_carry",
                    "phase_action_outputs",
                ],
                "drive": ["phase_h_drive", "phase_v_drive"],
            },
            "jobs": jobs,
        },
    )
    commands_path = output_dir / "launch_commands.txt"
    commands_path.write_text("\n".join(job["launch_command"] for job in jobs) + "\n")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--analysis-root", default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(SIGNALING_CONDITIONS),
        help="Subset of signalling conditions to trace.",
    )
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="Print the generated Modal launch commands after writing the manifest.",
    )
    args = parser.parse_args()
    manifest_path = write_manifest(args)
    print(manifest_path)
    if args.print_commands:
        manifest = json.loads(manifest_path.read_text())
        for job in manifest["jobs"]:
            print(job["launch_command"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
