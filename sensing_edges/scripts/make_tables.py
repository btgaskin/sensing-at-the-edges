#!/usr/bin/env python
"""Generate deformation-ablation tables from canonical publication summaries."""

from __future__ import annotations

import argparse

from sensing_edges.publication import generate_deformation_ablation_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deformation-ablation tables.")
    parser.add_argument("--release-dir", type=str, required=True)
    args = parser.parse_args()

    comparisons = generate_deformation_ablation_tables(args.release_dir)
    if not comparisons:
        print(f"No paired deformation-ablation comparisons found in {args.release_dir}")
        return
    print(f"Wrote {len(comparisons)} paired deformation-ablation comparisons to {args.release_dir}/stats")


if __name__ == "__main__":
    main()
