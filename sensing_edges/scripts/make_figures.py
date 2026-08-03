#!/usr/bin/env python
"""Generate publication figures from canonical release artifacts."""

from __future__ import annotations

import argparse

from sensing_edges.publication import generate_publication_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.add_argument("--release-dir", type=str, required=True)
    args = parser.parse_args()

    generate_publication_figures(args.release_dir)
    print(f"Figures saved to {args.release_dir}/figures")


if __name__ == "__main__":
    main()
