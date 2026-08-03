#!/usr/bin/env python
"""Package a canonical publication release into a tar.gz bundle."""

from __future__ import annotations

import argparse

from sensing_edges.publication import bundle_publication_release


def main() -> None:
    parser = argparse.ArgumentParser(description="Export publication release bundle.")
    parser.add_argument("--release-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    bundle_path = bundle_publication_release(
        args.release_dir,
        output=args.output,
    )
    print(f"Bundle written to {bundle_path}")
    print(f"Checksum written to {bundle_path.with_suffix('.sha256')}")


if __name__ == "__main__":
    main()
