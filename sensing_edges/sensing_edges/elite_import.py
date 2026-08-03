"""Import raw elite specimen snapshots into a local, post-run-compatible tree."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from typing import Any, Iterable, Sequence

import numpy as np

from .runtime_config import (
    compose_experiment_config,
    compose_stored_config,
    genotype_dim_from_cfg,
    resolve_runtime_config,
)


_IMPORT_SCHEMA_VERSION = 1
_DEFAULT_MODAL_VOLUME = "se-outputs"
_REMOTE_IMPORTED_RUNS_ROOT = PurePosixPath("/imported_elite_runs")
_REMOTE_IMPORTED_SPECIMENS_ROOT = PurePosixPath("/imported_elite_specimens")
_REMOTE_IMPORTED_ANALYSES_ROOT = PurePosixPath("/imported_elite_analyses")
_REMOTE_LOCAL_RUNS_ROOT = PurePosixPath("/rerun_local_runs")
_REMOTE_LOCAL_SPECIMENS_ROOT = PurePosixPath("/rerun_local_specimens")
_REMOTE_LOCAL_ANALYSES_ROOT = PurePosixPath("/rerun_local_analyses")
_CONTAINER_OUTPUTS_ROOT = PurePosixPath("/outputs")
_SE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ImportedEliteLayout:
    root_dir: Path
    run_dir_name: str

    @property
    def run_dir(self) -> Path:
        return self.root_dir / "runs" / self.run_dir_name

    @property
    def raw_dir(self) -> Path:
        return self.root_dir / "raw" / self.run_dir_name

    @property
    def specimen_root(self) -> Path:
        return self.root_dir / "specimens" / self.run_dir_name

    @property
    def analyses_root(self) -> Path:
        return self.root_dir / "analyses"


def normalize_remote_elite_path(remote_path: str) -> PurePosixPath:
    normalized = str(remote_path).strip()
    if not normalized:
        raise ValueError("Remote elite path must be non-empty.")
    if not normalized.startswith("/"):
        normalized = "/" + normalized

    remote = PurePosixPath(normalized)
    parts = remote.parts
    if len(parts) == 5 and parts[1] == "outputs" and parts[3] == "elite_specimens":
        return PurePosixPath("/") / parts[2] / parts[3] / parts[4]
    if len(parts) == 4 and parts[2] == "elite_specimens":
        return remote
    raise ValueError(
        "Remote elite path must match /<run_dir>/elite_specimens/<filename>.npz "
        "or /outputs/<run_dir>/elite_specimens/<filename>.npz"
    )


def parse_remote_elite_path(remote_path: str) -> tuple[str, str]:
    remote = normalize_remote_elite_path(remote_path)
    parts = remote.parts
    if remote.suffix != ".npz":
        raise ValueError("Remote elite path must point to a .npz file.")
    return parts[1], parts[3]


def layout_for_run(root_dir: str | Path, run_dir_name: str) -> ImportedEliteLayout:
    return ImportedEliteLayout(
        root_dir=Path(root_dir),
        run_dir_name=str(run_dir_name).strip(),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as handle:
        return json.load(handle)


def _np_scalar(payload: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in payload.files:
        return default
    value = np.asarray(payload[key])
    if value.shape == ():
        return value.item()
    return value


def _archive_cell_indices_list(values: Any) -> list[int] | None:
    if values is None:
        return None
    indices = [int(value) for value in np.asarray(values, dtype=np.int32).reshape(-1).tolist()]
    if not indices or all(value < 0 for value in indices):
        return None
    return indices


def _completion_bin(indices: list[int] | None) -> int:
    if not indices:
        return -1
    return int(indices[0])


def _normalize_remote_paths(remote_paths: Sequence[str] | None, expected: int) -> list[str | None]:
    if remote_paths is None:
        return [None] * expected
    if len(remote_paths) != expected:
        raise ValueError("remote_paths must match raw_paths length.")
    return [None if path is None else str(path) for path in remote_paths]


def _localize_raw_paths(
    *,
    layout: ImportedEliteLayout,
    raw_paths: Sequence[str | Path],
) -> list[Path]:
    layout.raw_dir.mkdir(parents=True, exist_ok=True)
    localized: list[Path] = []
    for raw_path in raw_paths:
        source = Path(raw_path)
        if not source.exists():
            raise FileNotFoundError(f"Elite snapshot not found: {source}")
        target = layout.raw_dir / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        localized.append(target)
    return localized


def _copy_if_needed(source: str | Path, target: Path) -> None:
    source_path = Path(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() == target.resolve():
        return
    shutil.copy2(source_path, target)


def _sort_import_rows(row: dict[str, Any]) -> tuple[int, int, int, str]:
    archive_flat_index = int(row.get("archive_flat_index", -1))
    return (
        int(row.get("generation", -1)),
        int(row.get("candidate_index", -1)),
        archive_flat_index,
        str(row.get("specimen_id", "")),
    )


def _import_row_from_elite_payload(
    *,
    specimen_id: str,
    raw_path: Path,
    remote_path: str | None,
    run_dir_name: str,
    run_id: str,
    volume_name: str | None,
) -> tuple[dict[str, Any], np.ndarray]:
    with np.load(raw_path, allow_pickle=False) as elite:
        genotype = np.asarray(elite["genotype"], dtype=np.float32)
        descriptor = np.asarray(_np_scalar(elite, "descriptor", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        archive_cell_indices = _archive_cell_indices_list(_np_scalar(elite, "archive_cell_indices"))
        archive_flat_index = int(_np_scalar(elite, "archive_flat_index", -1))
        row = {
            "specimen_id": specimen_id,
            "source": "imported_modal_elite_snapshot",
            "candidate_sources": ["imported_modal_elite_snapshot"],
            "run_id": str(run_id),
            "run_dir_name": str(run_dir_name),
            "volume_name": None if volume_name is None else str(volume_name),
            "remote_path": None if remote_path is None else str(remote_path),
            "raw_snapshot_path": str(raw_path),
            "raw_snapshot_name": raw_path.name,
            "genotype_sha256": str(_np_scalar(elite, "genotype_sha256", "")),
            "generation": int(_np_scalar(elite, "generation", -1)),
            "candidate_index": int(_np_scalar(elite, "candidate_index", -1)),
            "fitness": float(_np_scalar(elite, "fitness", np.nan)),
            "descriptor": descriptor.astype(np.float32).tolist(),
            "descriptor_names": [
                str(value)
                for value in np.asarray(_np_scalar(elite, "descriptor_names", np.asarray([], dtype=str))).tolist()
            ],
            "goal_hit_count": float(_np_scalar(elite, "goal_hit_count", 0.0)),
            "fall_count": float(_np_scalar(elite, "fall_count", 0.0)),
            "historical_elite_tier": str(_np_scalar(elite, "historical_elite_tier", "")),
            "archive_flat_index": (None if archive_flat_index < 0 else archive_flat_index),
            "archive_cell_indices": archive_cell_indices,
            "archive_cell_id": str(_np_scalar(elite, "archive_cell_id", "")),
            "archive_cell_label": str(_np_scalar(elite, "archive_cell_label", "")),
            "archive_status": (
                None
                if "archive_status" not in elite.files
                else int(np.asarray(elite["archive_status"], dtype=np.uint8).reshape(()))
            ),
            "completion_bin": _completion_bin(archive_cell_indices),
            "imported_at_utc": _utc_now(),
        }
    return row, genotype


def _selection_rule(*, volume_name: str | None) -> dict[str, Any]:
    return {
        "mode": "imported_modal_elite_snapshots",
        "schema_version": _IMPORT_SCHEMA_VERSION,
        "volume_name": None if volume_name is None else str(volume_name),
        "ranking": [
            "generation_asc",
            "candidate_index_asc",
            "archive_flat_index_asc",
            "specimen_id_asc",
        ],
    }


def _write_selected_genotypes(
    *,
    path: Path,
    ordered_rows: list[dict[str, Any]],
    genotypes_by_specimen: dict[str, np.ndarray],
    genotype_dim: int,
) -> None:
    if ordered_rows:
        genotype_stack = np.stack(
            [np.asarray(genotypes_by_specimen[str(row["specimen_id"])], dtype=np.float32) for row in ordered_rows],
            axis=0,
        )
    else:
        genotype_stack = np.zeros((0, genotype_dim), dtype=np.float32)

    np.savez_compressed(
        path,
        genotypes=genotype_stack,
        specimen_id=np.asarray([str(row["specimen_id"]) for row in ordered_rows], dtype=str),
        generation=np.asarray([int(row.get("generation", -1)) for row in ordered_rows], dtype=np.int32),
        candidate_index=np.asarray([int(row.get("candidate_index", -1)) for row in ordered_rows], dtype=np.int32),
        archive_flat_index=np.asarray(
            [
                -1 if row.get("archive_flat_index") is None else int(row["archive_flat_index"])
                for row in ordered_rows
            ],
            dtype=np.int32,
        ),
        archive_cell_id=np.asarray([str(row.get("archive_cell_id", "")) for row in ordered_rows], dtype=str),
        goal_hit_count=np.asarray([float(row.get("goal_hit_count", 0.0)) for row in ordered_rows], dtype=np.float32),
        fall_count=np.asarray([float(row.get("fall_count", 0.0)) for row in ordered_rows], dtype=np.float32),
        historical_elite_tier=np.asarray(
            [str(row.get("historical_elite_tier", "")) for row in ordered_rows],
            dtype=str,
        ),
        genotype_sha256=np.asarray([str(row.get("genotype_sha256", "")) for row in ordered_rows], dtype=str),
    )


def _write_specimen_root_manifest(
    *,
    layout: ImportedEliteLayout,
    run_id: str,
    volume_name: str | None,
    selected_count: int,
) -> Path:
    manifest_path = layout.specimen_root / "manifest.json"
    payload = {
        "schema_version": _IMPORT_SCHEMA_VERSION,
        "kind": "sensing_edges_imported_elite_specimen_set",
        "run_id": str(run_id),
        "run_dir_name": str(layout.run_dir_name),
        "volume_name": None if volume_name is None else str(volume_name),
        "selected_count": int(selected_count),
        "imported_at_utc": _utc_now(),
        "paths": {
            "run_dir": str(layout.run_dir),
            "raw_dir": str(layout.raw_dir),
            "selected_specimens": str(layout.specimen_root / "selected_specimens.json"),
            "selected_genotypes": str(layout.specimen_root / "selected_genotypes.npz"),
        },
    }
    _write_json(manifest_path, payload)
    return manifest_path


def import_raw_elite_snapshots(
    *,
    root_dir: str | Path,
    run_dir_name: str,
    raw_paths: Sequence[str | Path],
    run_manifest_path: str | Path,
    run_resolved_config_path: str | Path,
    volume_name: str | None = None,
    remote_paths: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    from .post_run import materialize_specimen_artifacts

    layout = layout_for_run(root_dir, run_dir_name)
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    layout.specimen_root.mkdir(parents=True, exist_ok=True)
    localized_raw_paths = _localize_raw_paths(layout=layout, raw_paths=raw_paths)
    normalized_remote_paths = _normalize_remote_paths(remote_paths, len(localized_raw_paths))

    local_run_manifest_path = layout.run_dir / "manifest.json"
    local_run_resolved_config_path = layout.run_dir / "resolved_config.yaml"
    _copy_if_needed(run_manifest_path, local_run_manifest_path)
    _copy_if_needed(run_resolved_config_path, local_run_resolved_config_path)

    run_manifest = _load_json(local_run_manifest_path)
    run_id = str(run_manifest.get("run_id", ""))
    cfg = compose_stored_config(local_run_resolved_config_path)
    selected_path = layout.specimen_root / "selected_specimens.json"
    existing_payload = (
        _load_json(selected_path)
        if selected_path.exists()
        else {"selection_rule": _selection_rule(volume_name=volume_name), "selected_specimens": []}
    )
    rows_by_specimen = {
        str(row["specimen_id"]): dict(row)
        for row in list(existing_payload.get("selected_specimens", []))
    }
    genotypes_by_specimen: dict[str, np.ndarray] = {}

    for raw_path, remote_path in zip(localized_raw_paths, normalized_remote_paths, strict=True):
        specimen_id = raw_path.stem
        row, genotype = _import_row_from_elite_payload(
            specimen_id=specimen_id,
            raw_path=raw_path,
            remote_path=remote_path,
            run_dir_name=layout.run_dir_name,
            run_id=run_id,
            volume_name=volume_name,
        )
        rows_by_specimen[specimen_id] = row
        genotypes_by_specimen[specimen_id] = genotype

    for row in rows_by_specimen.values():
        specimen_id = str(row["specimen_id"])
        if specimen_id in genotypes_by_specimen:
            continue
        raw_snapshot_path = Path(str(row["raw_snapshot_path"]))
        if not raw_snapshot_path.exists():
            raise FileNotFoundError(
                f"Missing cached raw snapshot for imported specimen {specimen_id}: {raw_snapshot_path}"
            )
        with np.load(raw_snapshot_path, allow_pickle=False) as elite:
            genotypes_by_specimen[specimen_id] = np.asarray(elite["genotype"], dtype=np.float32)

    ordered_rows = sorted(rows_by_specimen.values(), key=_sort_import_rows)
    selection_rule = _selection_rule(volume_name=volume_name)
    for rank, row in enumerate(ordered_rows):
        row["selected_rank"] = int(rank)
        artifact_paths = materialize_specimen_artifacts(
            specimen_root=layout.specimen_root,
            specimen_id=str(row["specimen_id"]),
            selected_rank=int(rank),
            selection_summary=row,
            selection_rule=selection_rule,
            genotype=np.asarray(genotypes_by_specimen[str(row["specimen_id"])], dtype=np.float32),
            cfg=cfg,
            run_manifest_path=local_run_manifest_path,
            run_resolved_config_path=local_run_resolved_config_path,
            generation=int(row.get("generation", -1)),
        )
        row.update(artifact_paths)

    _write_json(
        selected_path,
        {
            "selection_rule": selection_rule,
            "selected_specimens": ordered_rows,
        },
    )
    _write_selected_genotypes(
        path=layout.specimen_root / "selected_genotypes.npz",
        ordered_rows=ordered_rows,
        genotypes_by_specimen=genotypes_by_specimen,
        genotype_dim=genotype_dim_from_cfg(cfg),
    )
    _write_specimen_root_manifest(
        layout=layout,
        run_id=run_id,
        volume_name=volume_name,
        selected_count=len(ordered_rows),
    )
    return ordered_rows


def _modal_get(*, volume_name: str, remote_path: str, local_destination: Path) -> None:
    local_destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "modal",
            "volume",
            "get",
            str(volume_name),
            str(remote_path),
            str(local_destination),
            "--force",
        ],
        check=True,
        cwd=str(_SE_ROOT),
    )


def _modal_put(*, volume_name: str, local_path: Path, remote_dir: PurePosixPath) -> None:
    subprocess.run(
        [
            "modal",
            "volume",
            "put",
            str(volume_name),
            str(local_path),
            f"{remote_dir.as_posix()}/",
            "--force",
        ],
        check=True,
        cwd=str(_SE_ROOT),
    )


def _modal_run_post_run(
    *,
    run_dir: PurePosixPath,
    specimen_id: str,
    assays: Sequence[str] | None,
    config_path: PurePosixPath,
    specimen_output_root: PurePosixPath,
    analysis_root: PurePosixPath,
    detach_launch: bool,
) -> subprocess.CompletedProcess[str]:
    override = ",".join(
        [
            f"specimens.output_root={specimen_output_root.as_posix()}",
            f"eval.post_run.analysis_root={analysis_root.as_posix()}",
        ]
    )
    command = [
        "modal",
        "run",
        "scripts/modal_post_run.py",
        "--run-dir",
        run_dir.as_posix(),
        "--specimen-id",
        str(specimen_id),
        "--config-path",
        config_path.as_posix(),
        "--override",
        override,
    ]
    if assays:
        command.extend(["--assays", ",".join(str(assay) for assay in assays)])
    command.append("--detach-launch" if detach_launch else "--no-detach-launch")
    return subprocess.run(
        command,
        check=True,
        cwd=str(_SE_ROOT),
        text=True,
        capture_output=bool(detach_launch),
    )


def _load_selected_rows(layout: ImportedEliteLayout) -> list[dict[str, Any]]:
    selected_path = layout.specimen_root / "selected_specimens.json"
    if not selected_path.exists():
        raise FileNotFoundError(f"No imported specimens found for run {layout.run_dir_name}: {selected_path}")
    payload = _load_json(selected_path)
    return list(payload.get("selected_specimens", []))


def _resolve_requested_rows(
    *,
    layout: ImportedEliteLayout,
    specimen_ids: Iterable[str] | None,
) -> list[dict[str, Any]]:
    selected_rows = _load_selected_rows(layout)
    requested = None if specimen_ids is None else {str(specimen_id) for specimen_id in specimen_ids}
    chosen_rows = [
        row
        for row in selected_rows
        if requested is None or str(row.get("specimen_id")) in requested
    ]
    if requested is not None and len(chosen_rows) != len(requested):
        found = {str(row.get("specimen_id")) for row in chosen_rows}
        missing = sorted(requested - found)
        raise ValueError(f"Unknown imported specimen_id(s): {missing}")
    return chosen_rows


def normalize_local_specimen_source_path(
    source_path: str | Path,
) -> tuple[Path, str | None]:
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Stored specimen path not found: {source_path}")
    if source_path.is_dir() and (source_path / "selected_specimens.json").exists():
        return source_path, None
    if (
        source_path.is_dir()
        and source_path.name.startswith("specimen_")
        and (source_path.parent / "selected_specimens.json").exists()
    ):
        return source_path.parent, source_path.name.removeprefix("specimen_")
    raise ValueError(
        "Stored specimen path must be a specimen root containing selected_specimens.json "
        "or a specimen_* directory inside such a root."
    )


def infer_run_dir_name_from_specimen_root(specimen_root: str | Path) -> str:
    specimen_root = Path(specimen_root)
    if specimen_root.parent.name == "stages" and specimen_root.parent.parent.name:
        return specimen_root.parent.parent.name
    return specimen_root.name


def infer_config_path_for_specimen_root(
    specimen_root: str | Path,
    *,
    specimen_id: str | None = None,
) -> Path:
    specimen_root = Path(specimen_root)
    candidate_ids: list[str] = []
    if specimen_id is not None:
        candidate_ids.append(str(specimen_id))
    selected_path = specimen_root / "selected_specimens.json"
    if selected_path.exists():
        payload = _load_json(selected_path)
        candidate_ids.extend(
            str(row.get("specimen_id"))
            for row in list(payload.get("selected_specimens", []))
            if row.get("specimen_id") is not None
        )
    seen: set[str] = set()
    for candidate_id in candidate_ids:
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        candidate_path = specimen_root / f"specimen_{candidate_id}" / "config_snapshot.yaml"
        if candidate_path.exists():
            return candidate_path
    raise FileNotFoundError(
        f"Could not infer config_snapshot.yaml from stored specimen root {specimen_root}"
    )


def _stage_directory_children_to_modal(
    *,
    local_root: Path,
    remote_root: PurePosixPath,
    volume_name: str,
) -> None:
    if not local_root.exists():
        raise FileNotFoundError(f"Local staging root not found: {local_root}")
    for path in sorted(local_root.iterdir()):
        _modal_put(
            volume_name=volume_name,
            local_path=path,
            remote_dir=remote_root,
        )


def stage_local_specimen_root_to_modal(
    *,
    specimen_root: str | Path,
    run_dir_name: str,
    config_path: str | Path,
    volume_name: str = _DEFAULT_MODAL_VOLUME,
) -> dict[str, str]:
    specimen_root = Path(specimen_root)
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config path not found: {config_path}")

    remote_run_root = _REMOTE_LOCAL_RUNS_ROOT / run_dir_name
    remote_specimen_root = _REMOTE_LOCAL_SPECIMENS_ROOT / run_dir_name
    _modal_put(
        volume_name=volume_name,
        local_path=config_path,
        remote_dir=remote_run_root,
    )
    _stage_directory_children_to_modal(
        local_root=specimen_root,
        remote_root=remote_specimen_root,
        volume_name=volume_name,
    )
    return {
        "volume_name": str(volume_name),
        "remote_run_root": remote_run_root.as_posix(),
        "remote_specimen_root": remote_specimen_root.as_posix(),
        "remote_analysis_root": _REMOTE_LOCAL_ANALYSES_ROOT.as_posix(),
        "container_run_dir": (_CONTAINER_OUTPUTS_ROOT / remote_run_root.relative_to("/")).as_posix(),
        "container_config_path": (
            _CONTAINER_OUTPUTS_ROOT / remote_run_root.relative_to("/") / config_path.name
        ).as_posix(),
        "container_specimen_output_root": (
            _CONTAINER_OUTPUTS_ROOT / _REMOTE_LOCAL_SPECIMENS_ROOT.relative_to("/")
        ).as_posix(),
        "container_analysis_root": (
            _CONTAINER_OUTPUTS_ROOT / _REMOTE_LOCAL_ANALYSES_ROOT.relative_to("/")
        ).as_posix(),
    }


def post_run_determinism_summary(config_path: str | Path) -> dict[str, Any]:
    from .post_run import (
        _effective_trace_trials_per_side,
        _effective_trials_per_side,
        _effective_trials_per_width,
        _post_run_condition_is_stochastic,
    )

    cfg = compose_stored_config(config_path)
    stochastic = bool(_post_run_condition_is_stochastic(cfg))
    return {
        "stochastic_base_condition": stochastic,
        "effective_trials_per_side": int(_effective_trials_per_side(cfg)),
        "effective_trials_per_width": int(_effective_trials_per_width(cfg)),
        "effective_trace_trials_per_side": int(_effective_trace_trials_per_side(cfg)),
    }


def stage_imported_run_to_modal(
    *,
    root_dir: str | Path,
    run_dir_name: str,
    volume_name: str = _DEFAULT_MODAL_VOLUME,
) -> dict[str, str]:
    layout = layout_for_run(root_dir, run_dir_name)
    if not layout.run_dir.exists():
        raise FileNotFoundError(f"Imported run metadata not found: {layout.run_dir}")
    if not layout.specimen_root.exists():
        raise FileNotFoundError(f"Imported specimen root not found: {layout.specimen_root}")

    remote_run_root = _REMOTE_IMPORTED_RUNS_ROOT / run_dir_name
    remote_specimen_root = _REMOTE_IMPORTED_SPECIMENS_ROOT / run_dir_name
    for path in sorted(layout.run_dir.iterdir()):
        _modal_put(
            volume_name=volume_name,
            local_path=path,
            remote_dir=remote_run_root,
        )
    for path in sorted(layout.specimen_root.iterdir()):
        _modal_put(
            volume_name=volume_name,
            local_path=path,
            remote_dir=remote_specimen_root,
        )
    return {
        "volume_name": str(volume_name),
        "remote_run_root": remote_run_root.as_posix(),
        "remote_specimen_root": remote_specimen_root.as_posix(),
        "remote_analysis_root": _REMOTE_IMPORTED_ANALYSES_ROOT.as_posix(),
        "container_run_dir": (_CONTAINER_OUTPUTS_ROOT / remote_run_root.relative_to("/")).as_posix(),
        "container_specimen_output_root": (
            _CONTAINER_OUTPUTS_ROOT / _REMOTE_IMPORTED_SPECIMENS_ROOT.relative_to("/")
        ).as_posix(),
        "container_analysis_root": (
            _CONTAINER_OUTPUTS_ROOT / _REMOTE_IMPORTED_ANALYSES_ROOT.relative_to("/")
        ).as_posix(),
    }


def import_remote_elite_snapshots(
    *,
    root_dir: str | Path,
    volume_name: str,
    remote_paths: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[str]] = {}
    for remote_path in remote_paths:
        normalized_remote_path = str(normalize_remote_elite_path(remote_path))
        run_dir_name, _filename = parse_remote_elite_path(normalized_remote_path)
        grouped.setdefault(run_dir_name, []).append(normalized_remote_path)

    imported: dict[str, list[dict[str, Any]]] = {}
    for run_dir_name, run_remote_paths in grouped.items():
        layout = layout_for_run(root_dir, run_dir_name)
        layout.run_dir.mkdir(parents=True, exist_ok=True)
        layout.raw_dir.mkdir(parents=True, exist_ok=True)
        _modal_get(
            volume_name=volume_name,
            remote_path=f"/{run_dir_name}/manifest.json",
            local_destination=layout.run_dir / "manifest.json",
        )
        _modal_get(
            volume_name=volume_name,
            remote_path=f"/{run_dir_name}/resolved_config.yaml",
            local_destination=layout.run_dir / "resolved_config.yaml",
        )

        local_raw_paths: list[Path] = []
        for remote_path in run_remote_paths:
            local_raw_path = layout.raw_dir / PurePosixPath(remote_path).name
            _modal_get(
                volume_name=volume_name,
                remote_path=remote_path,
                local_destination=local_raw_path,
            )
            local_raw_paths.append(local_raw_path)

        imported[run_dir_name] = import_raw_elite_snapshots(
            root_dir=root_dir,
            run_dir_name=run_dir_name,
            raw_paths=local_raw_paths,
            run_manifest_path=layout.run_dir / "manifest.json",
            run_resolved_config_path=layout.run_dir / "resolved_config.yaml",
            volume_name=volume_name,
            remote_paths=run_remote_paths,
        )
    return imported


def run_post_run_for_imported_specimens(
    *,
    root_dir: str | Path,
    run_dir_name: str,
    specimen_ids: Iterable[str] | None = None,
    assays: set[str] | None = None,
) -> list[dict[str, Any]]:
    from .post_run import run_post_run_for_stored_specimen

    layout = layout_for_run(root_dir, run_dir_name)
    selected_path = layout.specimen_root / "selected_specimens.json"
    if not selected_path.exists():
        raise FileNotFoundError(f"No imported specimens found for run {run_dir_name}: {selected_path}")

    payload = _load_json(selected_path)
    selected_rows = list(payload.get("selected_specimens", []))
    requested = None if specimen_ids is None else {str(specimen_id) for specimen_id in specimen_ids}
    chosen_rows = [
        row
        for row in selected_rows
        if requested is None or str(row.get("specimen_id")) in requested
    ]
    if requested is not None and len(chosen_rows) != len(requested):
        found = {str(row.get("specimen_id")) for row in chosen_rows}
        missing = sorted(requested - found)
        raise ValueError(f"Unknown imported specimen_id(s): {missing}")

    cfg = compose_stored_config(layout.run_dir / "resolved_config.yaml")
    cfg.eval.post_run.analysis_root = str((layout.analyses_root).resolve())
    runtime_resolution = resolve_runtime_config(cfg)
    results: list[dict[str, Any]] = []
    for row in chosen_rows:
        specimen_id = str(row["specimen_id"])
        result = run_post_run_for_stored_specimen(
            cfg=cfg,
            runtime_resolution=runtime_resolution,
            run_dir=layout.run_dir,
            specimen_root=layout.specimen_root,
            specimen_id=specimen_id,
            assays=assays,
        )
        results.append(
            {
                "run_dir_name": layout.run_dir_name,
                "specimen_id": result.specimen_id,
                "specimen_dir": str(result.specimen_dir),
                "index_path": str(result.index_path),
                "status": str(result.status),
                "assay_dirs": dict(result.assay_dirs),
            }
        )
    return results


def run_post_run_for_imported_specimens_remote(
    *,
    root_dir: str | Path,
    run_dir_name: str,
    specimen_ids: Iterable[str] | None = None,
    assays: Sequence[str] | None = None,
    volume_name: str = _DEFAULT_MODAL_VOLUME,
    detach_launch: bool = False,
) -> list[dict[str, Any]]:
    layout = layout_for_run(root_dir, run_dir_name)
    chosen_rows = _resolve_requested_rows(
        layout=layout,
        specimen_ids=specimen_ids,
    )
    staged = stage_imported_run_to_modal(
        root_dir=root_dir,
        run_dir_name=run_dir_name,
        volume_name=volume_name,
    )
    run_dir = PurePosixPath(staged["container_run_dir"])
    config_path = run_dir / "resolved_config.yaml"
    specimen_output_root = PurePosixPath(staged["container_specimen_output_root"])
    analysis_root = PurePosixPath(staged["container_analysis_root"])

    results: list[dict[str, Any]] = []
    for row in chosen_rows:
        specimen_id = str(row["specimen_id"])
        completed = _modal_run_post_run(
            run_dir=run_dir,
            specimen_id=specimen_id,
            assays=assays,
            config_path=config_path,
            specimen_output_root=specimen_output_root,
            analysis_root=analysis_root,
            detach_launch=detach_launch,
        )
        results.append(
            {
                "run_dir_name": run_dir_name,
                "specimen_id": specimen_id,
                "volume_name": volume_name,
                "container_run_dir": run_dir.as_posix(),
                "container_specimen_output_root": specimen_output_root.as_posix(),
                "container_analysis_root": analysis_root.as_posix(),
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
                "detach_launch": bool(detach_launch),
            }
        )
    return results


def run_post_run_for_local_specimen_root_remote(
    *,
    source_path: str | Path,
    specimen_ids: Iterable[str] | None = None,
    config_path: str | Path | None = None,
    assays: Sequence[str] | None = None,
    volume_name: str = _DEFAULT_MODAL_VOLUME,
    detach_launch: bool = False,
) -> list[dict[str, Any]]:
    specimen_root, inferred_specimen_id = normalize_local_specimen_source_path(source_path)
    run_dir_name = infer_run_dir_name_from_specimen_root(specimen_root)
    chosen_specimen_ids = (
        [str(specimen_id) for specimen_id in specimen_ids]
        if specimen_ids is not None
        else ([] if inferred_specimen_id is None else [str(inferred_specimen_id)])
    )
    if not chosen_specimen_ids:
        raise ValueError(
            "No specimen_id selected. Provide a specimen_* directory path or pass specimen_ids explicitly."
        )

    resolved_config_path = (
        infer_config_path_for_specimen_root(specimen_root, specimen_id=chosen_specimen_ids[0])
        if config_path is None
        else Path(config_path)
    )
    determinism = post_run_determinism_summary(resolved_config_path)
    staged = stage_local_specimen_root_to_modal(
        specimen_root=specimen_root,
        run_dir_name=run_dir_name,
        config_path=resolved_config_path,
        volume_name=volume_name,
    )
    run_dir = PurePosixPath(staged["container_run_dir"])
    container_config_path = PurePosixPath(staged["container_config_path"])
    specimen_output_root = PurePosixPath(staged["container_specimen_output_root"])
    analysis_root = PurePosixPath(staged["container_analysis_root"])

    results: list[dict[str, Any]] = []
    for specimen_id in chosen_specimen_ids:
        completed = _modal_run_post_run(
            run_dir=run_dir,
            specimen_id=specimen_id,
            assays=assays,
            config_path=container_config_path,
            specimen_output_root=specimen_output_root,
            analysis_root=analysis_root,
            detach_launch=detach_launch,
        )
        results.append(
            {
                "run_dir_name": run_dir_name,
                "specimen_root": str(specimen_root),
                "specimen_id": specimen_id,
                "volume_name": volume_name,
                "container_run_dir": run_dir.as_posix(),
                "container_config_path": container_config_path.as_posix(),
                "container_specimen_output_root": specimen_output_root.as_posix(),
                "container_analysis_root": analysis_root.as_posix(),
                "determinism": dict(determinism),
                "stdout": (completed.stdout or "").strip(),
                "stderr": (completed.stderr or "").strip(),
                "detach_launch": bool(detach_launch),
            }
        )
    return results
