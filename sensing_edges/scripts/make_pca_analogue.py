#!/usr/bin/env python
"""Generate Briggman-inspired PCA analogue figures from a PCA bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from sensing_edges.pca_analogue import analyze_pca_bundle


def _resolve_bundle_dir(bundle_dir: str, cohort_id: str | None) -> Path:
    root = Path(bundle_dir)
    if cohort_id is None:
        return root
    if cohort_id == "intact_native_core" and (root / "cohort_summary.json").exists():
        return root
    candidate = root / "cohorts" / str(cohort_id)
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"PCA cohort not found under {root}: {cohort_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PCA analogue outputs from a post-run bundle.")
    parser.add_argument("--bundle-dir", type=str, required=True)
    parser.add_argument("--cohort-id", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--max-plot-points", type=int, default=200000)
    parser.add_argument("--max-trajectory-points-per-trial", type=int, default=2000)
    parser.add_argument("--max-animation-points", type=int, default=240)
    args = parser.parse_args()

    bundle_dir = _resolve_bundle_dir(args.bundle_dir, args.cohort_id)
    summary = analyze_pca_bundle(
        bundle_dir,
        args.output_dir,
        n_components=args.n_components,
        max_plot_points=args.max_plot_points,
        max_trajectory_points_per_trial=args.max_trajectory_points_per_trial,
        max_animation_points=args.max_animation_points,
    )
    print(f"Wrote PCA analogue outputs to {args.output_dir}")
    print(f"Bundle dir: {bundle_dir}")
    print(f"Selected trials: {summary['selected_trial_count']}")


if __name__ == "__main__":
    main()
