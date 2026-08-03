from __future__ import annotations

import argparse
from pathlib import Path

from free_space_branch_common import (
    DEFAULT_BUNDLE_ROOT,
    capture_branch_trace,
    default_analysis_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a deformation-off branched rollout for later render/PCA analysis.",
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT,
        help="Specimen bundle root. Defaults to the canonical copied local bundle.",
    )
    parser.add_argument(
        "--controller-params-path",
        type=Path,
        default=None,
        help="Override controller parameter path. Defaults to <bundle-root>/controller_params.npz.",
    )
    parser.add_argument(
        "--config-snapshot-path",
        type=Path,
        default=None,
        help="Override config snapshot path. Defaults to <bundle-root>/config_snapshot.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override capture output directory. Defaults under analysis_runs/free_space_branch_steps_<max_steps>.",
    )
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument(
        "--environment-mode",
        type=str,
        choices=("free_space", "flat_ground"),
        default="free_space",
    )
    parser.add_argument("--ground-width", type=int, default=20)
    parser.add_argument("--ground-clearance-cells", type=float, default=0.1)
    parser.add_argument("--resume-env-step", type=int, default=None)
    parser.add_argument("--perturb-local-cell-index", type=int, default=None)
    parser.add_argument(
        "--perturb-strength",
        type=float,
        default=1.0,
        help="Scale factor applied to the support-release perturbation profile.",
    )
    parser.add_argument(
        "--support-duration-env-steps",
        type=float,
        default=1.0,
        help="How many env-step equivalents of sudden support release to inject.",
    )
    parser.add_argument(
        "--perturb-mode",
        type=str,
        choices=("support_release", "hold_release"),
        default="support_release",
    )
    parser.add_argument(
        "--hold-until-env-step",
        type=int,
        default=50,
        help="For hold_release mode: keep the target cell held through this env step, then release.",
    )
    parser.add_argument(
        "--com-hold-until-env-step",
        type=int,
        default=None,
        help="Optional env step through which both branches keep their center of mass pinned.",
    )
    parser.add_argument(
        "--other-hold-until-env-step",
        type=int,
        default=None,
        help="Optional second branch hold duration. In hold_release mode this lets the other row stay held longer.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_root = args.bundle_root.resolve()
    controller_params_path = (
        args.controller_params_path.resolve()
        if args.controller_params_path is not None
        else (bundle_root / "controller_params.npz").resolve()
    )
    config_snapshot_path = (
        args.config_snapshot_path.resolve()
        if args.config_snapshot_path is not None
        else (bundle_root / "config_snapshot.yaml").resolve()
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_analysis_dir(bundle_root, max_steps=int(args.max_steps)).resolve()
    )

    summary = capture_branch_trace(
        bundle_root=bundle_root,
        controller_params_path=controller_params_path,
        config_snapshot_path=config_snapshot_path,
        output_dir=output_dir,
        max_steps=int(args.max_steps),
        environment_mode=str(args.environment_mode),
        ground_width=int(args.ground_width),
        ground_clearance_cells=float(args.ground_clearance_cells),
        resume_env_step=args.resume_env_step,
        perturb_local_cell_index=args.perturb_local_cell_index,
        perturb_strength=float(args.perturb_strength),
        support_duration_env_steps=float(args.support_duration_env_steps),
        perturb_mode=str(args.perturb_mode),
        hold_until_env_step=int(args.hold_until_env_step),
        com_hold_until_env_step=args.com_hold_until_env_step,
        other_hold_until_env_step=args.other_hold_until_env_step,
        seed=int(args.seed),
    )

    print(f"trace_dir: {output_dir}")
    print(f"trace_path: {summary['trace_path']}")
    print(f"summary_path: {output_dir / 'summary.json'}")
    print(f"controller_post_delta_max: {summary['controller_post_delta_max']:.8f}")
    print(f"plant_post_delta_max: {summary['plant_post_delta_max']:.8f}")


if __name__ == "__main__":
    main()
