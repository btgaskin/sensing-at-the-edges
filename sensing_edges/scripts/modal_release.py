"""Modal entrypoint for publication release materialization and packaging."""

from __future__ import annotations

from pathlib import Path

import modal

from _modal_common import VOLUMES, app, commit_volumes, se_image


@app.function(
    image=se_image,
    timeout=7200,
    volumes=VOLUMES,
)
def release_remote(
    *,
    action: str = "materialize",
    outputs_dir: str = "/outputs",
    publication_set_name: str = "paper_main",
    release_id: str | None = None,
):
    """Build canonical publication artifacts directly inside the shared outputs volume."""
    import os

    os.chdir("/root/experiment")

    from sensing_edges.publication import (
        bundle_publication_release,
        generate_deformation_ablation_tables,
        generate_publication_figures,
        load_publication_set,
        materialize_publication_release,
        release_root,
        resolve_release_id,
    )

    publication_set = load_publication_set(publication_set_name, se_root="/root/experiment")
    resolved_release_id = resolve_release_id(release_id=release_id, publication_set=publication_set)

    try:
        manifest, run_records = materialize_publication_release(
            outputs_dir,
            release_id=resolved_release_id,
            publication_set=publication_set,
            se_root="/root/experiment",
        )
        resolved_release_root = release_root(outputs_dir, resolved_release_id)
        bundle_path = None

        normalized_action = str(action).strip().lower()
        if normalized_action in {"tables", "figures", "bundle", "all"}:
            generate_deformation_ablation_tables(resolved_release_root)
        if normalized_action in {"figures", "bundle", "all"}:
            generate_publication_figures(resolved_release_root)
        if normalized_action in {"bundle", "all"}:
            bundle_path = bundle_publication_release(
                resolved_release_root,
                se_root="/root/experiment",
            )

        return {
            "action": normalized_action,
            "release_id": resolved_release_id,
            "release_root": str(resolved_release_root),
            "ready_runs": int(sum(1 for record in run_records if record.get("publication_ready"))),
            "bundle_path": None if bundle_path is None else str(bundle_path),
            "release_manifest": str(Path(manifest["paths"]["run_index"]).parent.parent / "release_manifest.json"),
        }
    finally:
        commit_volumes()


@app.local_entrypoint()
def main(
    action: str = "materialize",
    outputs_dir: str = "/outputs",
    publication_set: str = "paper_main",
    release_id: str | None = None,
):
    result = release_remote.remote(
        action=action,
        outputs_dir=outputs_dir,
        publication_set_name=publication_set,
        release_id=release_id,
    )
    print(f"\nRelease result: {result}")
