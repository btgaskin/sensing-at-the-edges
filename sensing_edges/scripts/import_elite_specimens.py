#!/usr/bin/env python
"""Pull and materialize raw elite specimen snapshots from Modal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sensing_edges.elite_import import (
    import_remote_elite_snapshots,
    run_post_run_for_imported_specimens_remote,
    run_post_run_for_local_specimen_root_remote,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "elite_specimens"


def _read_paths_file(path: str | Path) -> list[str]:
    raw_lines = Path(path).read_text().splitlines()
    return [
        line.strip()
        for line in raw_lines
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pull_parser = subparsers.add_parser("pull", help="Pull remote elite snapshots into a local import root.")
    pull_parser.add_argument("remote_path", nargs="*", help="Remote Modal paths to elite .npz snapshots.")
    pull_parser.add_argument(
        "--paths-file",
        action="append",
        default=[],
        help="Text file with one remote Modal path per line.",
    )
    pull_parser.add_argument("--volume", default="se-outputs", help="Modal volume name.")
    pull_parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Local import root. Defaults to repo_root/elite_specimens.",
    )

    render_parser = subparsers.add_parser(
        "render",
        help="Stage imported elites to Modal and run post-run assays on A100s.",
    )
    render_parser.add_argument("run_dir_name", help="Run directory name under the import root.")
    render_parser.add_argument(
        "--specimen-id",
        action="append",
        default=[],
        help="Imported specimen id(s) to render. Repeat for multiple specimens.",
    )
    render_parser.add_argument(
        "--all",
        action="store_true",
        help="Render all imported specimens for the run.",
    )
    render_parser.add_argument(
        "--assays",
        default=None,
        help="Comma-separated subset of post-run assays to execute.",
    )
    render_parser.add_argument(
        "--volume",
        default="se-outputs",
        help="Modal volume name used for staging and analysis outputs.",
    )
    render_parser.add_argument(
        "--detach-launch",
        action="store_true",
        help="Launch one detached Modal post-run job per specimen instead of waiting.",
    )
    render_parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Local import root. Defaults to repo_root/elite_specimens.",
    )

    local_parser = subparsers.add_parser(
        "rerun-local",
        help="Stage an existing local stored-specimen root and run post-run assays on Modal A100s.",
    )
    local_parser.add_argument(
        "source_path",
        help="Path to a stored specimen root containing selected_specimens.json, or a specimen_* directory inside it.",
    )
    local_parser.add_argument(
        "--specimen-id",
        action="append",
        default=[],
        help="Specimen id(s) to run when source_path points at a specimen root.",
    )
    local_parser.add_argument(
        "--config-path",
        default=None,
        help="Local config snapshot/resolved config to stage. Defaults to config_snapshot.yaml from the chosen specimen.",
    )
    local_parser.add_argument(
        "--assays",
        default=None,
        help="Comma-separated subset of post-run assays to execute.",
    )
    local_parser.add_argument(
        "--volume",
        default="se-outputs",
        help="Modal volume name used for staging and analysis outputs.",
    )
    local_parser.add_argument(
        "--detach-launch",
        action="store_true",
        help="Launch one detached Modal post-run job per specimen instead of waiting.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "pull":
        remote_paths = list(args.remote_path)
        for paths_file in list(args.paths_file):
            remote_paths.extend(_read_paths_file(paths_file))
        if not remote_paths:
            parser.error("pull requires at least one remote_path or --paths-file entry.")

        imported = import_remote_elite_snapshots(
            root_dir=args.root,
            volume_name=args.volume,
            remote_paths=remote_paths,
        )
        print(json.dumps(imported, indent=2, sort_keys=True))
        return

    if args.command == "render":
        specimen_ids = list(args.specimen_id)
        if not args.all and not specimen_ids:
            parser.error("render requires --specimen-id or --all.")
        results = run_post_run_for_imported_specimens_remote(
            root_dir=args.root,
            run_dir_name=args.run_dir_name,
            specimen_ids=(None if args.all else specimen_ids),
            assays=(None if args.assays is None else [token.strip() for token in args.assays.split(",") if token.strip()]),
            volume_name=args.volume,
            detach_launch=bool(args.detach_launch),
        )
        print(json.dumps(results, indent=2, sort_keys=True))
        return

    results = run_post_run_for_local_specimen_root_remote(
        source_path=args.source_path,
        specimen_ids=(None if not args.specimen_id else list(args.specimen_id)),
        config_path=args.config_path,
        assays=(None if args.assays is None else [token.strip() for token in args.assays.split(",") if token.strip()]),
        volume_name=args.volume,
        detach_launch=bool(args.detach_launch),
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
