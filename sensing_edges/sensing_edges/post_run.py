"""Specimen-centric post-run assay orchestration for sensing_edges."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
import tempfile
import time
import traceback
from typing import Any, Callable, Sequence

import jax
import numpy as np
import pandas as pd
from hydra.core.global_hydra import GlobalHydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

from jax_evogym.constants import CELL_SIZE

from .batch_eval import make_com_trace_evaluator, split_key_sequence, trace_population_chunked
from .env import (
    run_episode_with_controller_trace,
    run_episode_with_state_trace,
    run_episode_with_state_space_trace,
    stack_eval_envs,
)
from .evaluate import _build_phase_assets
from .fitness import episode_fitness_array
from .manifest import load_manifest
from .nca import ACTION_H_IDX, ACTION_V_IDX, ctrnn_flat_to_params, base_channel_count
from .render import _render_config_for_stage, _render_parallel_workers, render_episode_gif
from .runtime_config import (
    active_comm_filters_from_cfg,
    compatible_post_run_signaling_conditions,
    resolved_noise_settings,
    rollout_kwargs_from_cfg,
    run_metadata,
)
from .terrain import (
    build_bilateral_goal_cliff_terrain,
    build_symmetric_double_cliff_terrain,
    make_robot_body,
)
from .terrain_schedule import TerrainEntry

_GOAL_SIDES = ("left", "right")
_SPECIMEN_SCHEMA_VERSION = 1
_POST_RUN_SCHEMA_VERSION = 1
_CLIFF_VICINITY_BAND = 3.0 * float(CELL_SIZE)
_ARREST_WINDOW_STEPS = 20
_FULL_BATCH_CONTROLLER_TRACE_GPU_PROFILES = frozenset(
    {"a100_40gb", "a100_80gb", "h100", "h200", "b200"}
)
_ASSAY_FAMILY_ORDER = (
    "pca_bundle",
    "native_core",
    "initial_direction",
    "long_distance_generalization",
    "double_cliff",
    "signaling_ablation",
    "deformation_ablation",
    "frozen_sensing_control",
    "body_transfer",
    "state_space_trace",
    "state_space_trace_deformation_off",
    "state_space_trace_edges_deformation_off",
    "combined_ablation_quick",
    "tau",
)

_SURFACE_LABELS = {
    "native_core": "Native Core",
    "long_distance_generalization": "Long-Distance Generalization",
    "double_cliff": "Double Cliff",
}

_SIGNALING_LABELS = {
    "intact": "Intact",
    "sobel_x_off": "Sobel-X Off",
    "sobel_y_off": "Sobel-Y Off",
    "laplacian_off": "Laplacian Off",
    "identity_comm_off": "Identity Off",
    "all_comm_off": "All Communication Off",
    "sobel_x_sign_flip": "Sobel-X Sign Flip",
}

_DEFORMATION_LABELS = {
    "intact": "Intact",
    "absolute_only": "Absolute Only",
    "compliance_only": "Compliance Only",
    "deformation_off": "All Deformation Off",
    "noise_only": "Noise Only",
    "shuffled": "Shuffled",
    "deformation_reversed": "Deformation Reversed",
}

_DEFORMATION_METADATA = {
    "intact": {
        "label": "Intact",
        "description": "All signed deformation channels active.",
    },
    "absolute_only": {
        "label": "Absolute Only",
        "description": "Keep absolute channels active and zero compliance channels.",
    },
    "compliance_only": {
        "label": "Compliance Only",
        "description": "Keep compliance channels active and zero absolute channels.",
    },
    "deformation_off": {
        "label": "All Deformation Off",
        "description": "Neutralize all deformation channels.",
    },
    "noise_only": {
        "label": "Noise Only",
        "description": "Replace signed deformation deltas with noise-only input.",
    },
    "shuffled": {
        "label": "Shuffled",
        "description": "Shuffle deformation channels per episode using the configured shuffle mode.",
    },
    "deformation_reversed": {
        "label": "Deformation Reversed",
        "description": "Flip signed deformation deltas around the neutral point.",
    },
}

_SENSING_FREEZE_LABELS = {
    "deformation_frozen_start": "Deformation Frozen",
    "perception_frozen_start": "Perception Frozen",
}

_SENSING_FREEZE_METADATA = {
    "deformation_frozen_start": {
        "label": "Deformation Frozen",
        "description": "Hold episode-start deformation/sensory inputs fixed while internal signalling and controller dynamics continue.",
    },
    "perception_frozen_start": {
        "label": "Perception Frozen",
        "description": "Hold the episode-start CTRNN perception tensor fixed while recurrent dynamics and actions continue.",
    },
}

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PostRunResults:
    specimen_id: str
    specimen_dir: Path
    index_path: Path
    assay_dirs: dict[str, str]
    ran_assays: list[str]
    status: str
    failed_assay: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class BatchedPostRunJob:
    specimen_id: str
    run_id: str
    run_dir: Path
    specimen_root: Path
    specimen_dir: Path
    specimen_manifest_path: Path
    specimen_manifest: dict[str, Any]
    selection_row: dict[str, Any]
    analysis_root: Path
    analysis_manifest_path: Path
    analysis_index_path: Path
    pointer_path: Path
    genotype: np.ndarray


def resolve_specimen_root(run_dir: str | Path, output_root: str | Path = "specimens") -> Path:
    """Resolve the persistent specimen root for a completed run directory."""
    run_dir = Path(run_dir)
    output_root = Path(output_root)
    if output_root.is_absolute():
        return output_root / run_dir.name
    return run_dir.parent / output_root / run_dir.name


def resolve_historical_elite_root(
    run_dir: str | Path,
    output_root: str | Path = "specimens",
) -> Path:
    """Resolve the canonical historical-elite root for a run."""
    return resolve_specimen_root(run_dir, output_root=output_root) / "historical_elites"


def resolve_analysis_root(
    run_dir: str | Path,
    specimen_id: str,
    output_root: str | Path = "analyses",
) -> Path:
    """Resolve the canonical post-run analysis root for one stored specimen."""
    run_dir = Path(run_dir)
    output_root = Path(output_root)
    if output_root.is_absolute():
        return output_root / run_dir.name / str(specimen_id)
    return run_dir.parent / output_root / run_dir.name / str(specimen_id)


def resolve_batched_report_root(
    analysis_roots: Sequence[str | Path],
    *,
    batch_label: str | None = None,
) -> Path:
    resolved_roots = [Path(root) for root in analysis_roots]
    if not resolved_roots:
        raise ValueError("resolve_batched_report_root requires at least one analysis root")
    base_roots = {root.parent.parent for root in resolved_roots}
    if len(base_roots) != 1:
        raise ValueError(
            "Batched report root requires a shared analysis-root base; got "
            + ", ".join(str(root) for root in sorted(base_roots))
        )
    label = (
        str(batch_label).strip()
        if batch_label is not None and str(batch_label).strip()
        else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    return next(iter(base_roots)) / "_batched_reports" / label


def resolve_stored_specimen_root(run_dir: str | Path, output_root: str | Path = "specimens") -> Path:
    """Resolve the specimen root for replay, preferring staged outputs when needed."""
    specimen_root = resolve_specimen_root(run_dir, output_root=output_root)
    if (specimen_root / "selected_specimens.json").exists():
        return specimen_root

    stages_root = specimen_root / "stages"
    if not stages_root.exists():
        return specimen_root

    stage_dirs = sorted(
        stage_dir
        for stage_dir in stages_root.glob("gen_*")
        if stage_dir.is_dir() and (stage_dir / "selected_specimens.json").exists()
    )
    if not stage_dirs:
        return specimen_root

    run_manifest_path = Path(run_dir) / "manifest.json"
    if run_manifest_path.exists():
        try:
            run_manifest = load_manifest(run_manifest_path)
            completed_generations = run_manifest.get("summary", {}).get("completed_generations")
            if completed_generations is not None:
                preferred = stages_root / f"gen_{int(completed_generations) - 1:04d}"
                if (preferred / "selected_specimens.json").exists():
                    return preferred
        except Exception:
            pass
    return stage_dirs[-1]


def specimen_directory(specimen_root: str | Path, specimen_id: str) -> Path:
    return Path(specimen_root) / f"specimen_{specimen_id}"


def historical_elite_directory(historical_root: str | Path, specimen_id: str) -> Path:
    return Path(historical_root) / f"specimen_{specimen_id}"


def load_selected_specimens(specimen_root: str | Path) -> dict[str, Any]:
    selected_path = Path(specimen_root) / "selected_specimens.json"
    with open(selected_path) as handle:
        return json.load(handle)


def _derived_specimen_id(row: dict[str, Any], *, selected_rank: int) -> str:
    archive_flat_index = row.get("archive_flat_index")
    if archive_flat_index is None or int(archive_flat_index) < 0:
        source = str(row.get("source", "selected")).strip() or "selected"
        source_index = int(row.get("source_index", selected_rank))
        specimen_suffix = f"{source}_{source_index:04d}"
    else:
        archive_cell_id = str(row.get("archive_cell_id", "")).strip() or "legacy_cell"
        specimen_suffix = f"{archive_cell_id}_flat_{int(archive_flat_index):04d}"
    completion_bin = int(row.get("completion_bin", 0))
    return f"bin_{completion_bin:02d}_rank_{int(selected_rank):02d}_{specimen_suffix}"


def _canonicalize_selected_payload(
    specimen_root: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    payload = load_selected_specimens(specimen_root)
    selected_rows = list(payload.get("selected_specimens", []))
    changed = False
    canonical_rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(selected_rows):
        row = dict(raw_row)
        selected_rank = int(row.get("selected_rank", index))
        if int(row.get("selected_rank", index)) != selected_rank or "selected_rank" not in row:
            row["selected_rank"] = selected_rank
            changed = True
        specimen_id = str(row.get("specimen_id", "")).strip()
        if not specimen_id:
            row["specimen_id"] = _derived_specimen_id(row, selected_rank=selected_rank)
            changed = True
        canonical_rows.append(row)
    if changed:
        payload = dict(payload)
        payload["selected_specimens"] = canonical_rows
        _write_json(Path(specimen_root) / "selected_specimens.json", payload)
    return payload, canonical_rows, changed


def resolve_selected_specimen(
    specimen_root: str | Path,
    *,
    specimen_id: str | None = None,
    specimen_rank: int | None = None,
) -> tuple[dict[str, Any], Path]:
    _payload, selected, _changed = _canonicalize_selected_payload(specimen_root)
    if specimen_id is not None:
        for row in selected:
            if str(row.get("specimen_id")) == str(specimen_id):
                return row, specimen_directory(specimen_root, str(specimen_id))
        raise ValueError(f"Unknown specimen_id={specimen_id!r}")
    resolved_rank = 0 if specimen_rank is None else int(specimen_rank)
    if resolved_rank < 0 or resolved_rank >= len(selected):
        raise ValueError(
            f"specimen_rank={resolved_rank} is out of range for {len(selected)} selected specimens"
        )
    row = selected[resolved_rank]
    return row, specimen_directory(specimen_root, str(row["specimen_id"]))


def _resolve_selected_genotype(
    specimen_root: str | Path,
    *,
    row: dict[str, Any],
    selected_rank: int,
) -> np.ndarray:
    genotype = row.get("genotype")
    if genotype is not None:
        return np.asarray(genotype, dtype=np.float32)

    genotype_path = Path(specimen_root) / "selected_genotypes.npz"
    if not genotype_path.exists():
        raise FileNotFoundError(f"Selected genotype payload not found: {genotype_path}")

    with np.load(genotype_path, allow_pickle=False) as data:
        genotypes = np.asarray(data["genotypes"], dtype=np.float32)
        if "specimen_id" in data.files:
            specimen_ids = [str(value) for value in np.asarray(data["specimen_id"]).tolist()]
            desired_id = str(row["specimen_id"])
            if desired_id in specimen_ids:
                return np.asarray(genotypes[specimen_ids.index(desired_id)], dtype=np.float32)
        if 0 <= int(selected_rank) < int(genotypes.shape[0]):
            return np.asarray(genotypes[int(selected_rank)], dtype=np.float32)
    raise ValueError(
        f"Could not resolve genotype for specimen_id={row['specimen_id']!r} "
        f"selected_rank={selected_rank}"
    )


def _manifest_generation(run_manifest: dict[str, Any]) -> int:
    summary = dict(run_manifest.get("summary", {}))
    completed_generations = summary.get("completed_generations")
    if completed_generations is not None:
        return max(0, int(completed_generations) - 1)
    stages = list(run_manifest.get("stages", []))
    for stage in reversed(stages):
        completed_generation = stage.get("completed_generation")
        if completed_generation is not None:
            return max(0, int(completed_generation))
    retry = dict(run_manifest.get("retry", {}))
    resumed_generation = retry.get("resumed_generation")
    if resumed_generation is not None:
        return max(0, int(resumed_generation))
    return 0


def ensure_specimen_artifacts(
    *,
    cfg: DictConfig,
    run_dir: str | Path,
    specimen_root: str | Path,
    specimen_id: str | None = None,
    specimen_rank: int | None = None,
) -> tuple[dict[str, Any], Path]:
    run_dir = Path(run_dir)
    specimen_root = Path(specimen_root)
    payload, selected_rows, _changed = _canonicalize_selected_payload(specimen_root)
    resolved_rank = 0 if specimen_rank is None else int(specimen_rank)
    row: dict[str, Any] | None = None
    row_index: int | None = None
    if specimen_id is not None:
        for index, candidate in enumerate(selected_rows):
            if str(candidate.get("specimen_id")) == str(specimen_id):
                row = candidate
                row_index = index
                break
        if row is None:
            raise ValueError(f"Unknown specimen_id={specimen_id!r}")
    else:
        if resolved_rank < 0 or resolved_rank >= len(selected_rows):
            raise ValueError(
                f"specimen_rank={resolved_rank} is out of range for {len(selected_rows)} selected specimens"
            )
        row_index = resolved_rank
        row = selected_rows[row_index]

    assert row is not None
    assert row_index is not None
    resolved_rank = int(row.get("selected_rank", row_index))
    specimen_dir = specimen_directory(specimen_root, str(row["specimen_id"]))
    required_paths = (
        specimen_dir / "genotype.npz",
        specimen_dir / "controller_params.npz",
        specimen_dir / "config_snapshot.yaml",
        specimen_dir / "selection_summary.json",
        specimen_dir / "manifest.json",
    )
    if all(path.exists() for path in required_paths):
        return row, specimen_dir

    run_manifest_path = run_dir / "manifest.json"
    run_resolved_config_path = run_dir / "resolved_config.yaml"
    run_manifest = load_manifest(run_manifest_path)
    genotype = _resolve_selected_genotype(
        specimen_root,
        row=row,
        selected_rank=resolved_rank,
    )
    artifact_paths = materialize_specimen_artifacts(
        specimen_root=specimen_root,
        specimen_id=str(row["specimen_id"]),
        selected_rank=resolved_rank,
        selection_summary=row,
        selection_rule=dict(payload.get("selection_rule", {})),
        genotype=genotype,
        cfg=cfg,
        run_manifest_path=run_manifest_path,
        run_resolved_config_path=run_resolved_config_path,
        generation=_manifest_generation(run_manifest),
    )
    hydrated_row = dict(row)
    hydrated_row.update(artifact_paths)
    payload = dict(payload)
    selected_rows = list(selected_rows)
    selected_rows[row_index] = hydrated_row
    payload["selected_specimens"] = selected_rows
    _write_json(specimen_root / "selected_specimens.json", payload)
    return hydrated_row, specimen_dir


def materialize_specimen_artifacts(
    *,
    specimen_root: str | Path,
    specimen_id: str,
    selected_rank: int,
    selection_summary: dict[str, Any],
    selection_rule: dict[str, Any],
    genotype: np.ndarray,
    cfg: DictConfig,
    run_manifest_path: str | Path,
    run_resolved_config_path: str | Path,
    generation: int,
) -> dict[str, str]:
    """Create the durable per-specimen folder used by automatic and manual post-run."""
    specimen_root = Path(specimen_root)
    specimen_dir = specimen_directory(specimen_root, specimen_id)
    specimen_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = specimen_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    genotype = np.asarray(genotype, dtype=np.float32)
    run_manifest_path = Path(run_manifest_path)
    run_resolved_config_path = Path(run_resolved_config_path)
    run_manifest = load_manifest(run_manifest_path)
    config_text = run_resolved_config_path.read_text()
    params = ctrnn_flat_to_params(
        jax.numpy.asarray(genotype),
        ctrnn_hidden=int(cfg.nca.ctrnn_hidden),
        ctrnn_grid_hidden=int(cfg.nca.ctrnn_grid_hidden),
        gasnet_mode=str(cfg.nca.get("gasnet_mode", "off")),
        ou_enabled=bool(resolved_noise_settings(cfg)["ou_enabled"]),
        growth_enabled=str(cfg.body.mode).strip().lower() == "grown",
        active_comm_filters=active_comm_filters_from_cfg(cfg),
    )

    genotype_path = specimen_dir / "genotype.npz"
    controller_params_path = specimen_dir / "controller_params.npz"
    config_snapshot_path = specimen_dir / "config_snapshot.yaml"
    selection_summary_path = specimen_dir / "selection_summary.json"
    manifest_path = specimen_dir / "manifest.json"
    preview_dir = specimen_dir / "preview_renders"
    preview_manifest_path = preview_dir / "manifest.json"

    np.savez_compressed(
        genotype_path,
        genotype=genotype,
    )
    np.savez_compressed(
        controller_params_path,
        w_input=np.asarray(params.w_input, dtype=np.float32),
        w_rec=np.asarray(params.w_rec, dtype=np.float32),
        w_output=np.asarray(params.w_output, dtype=np.float32),
        bias=np.asarray(params.bias, dtype=np.float32),
        log_tau=np.asarray(params.log_tau, dtype=np.float32),
    )
    config_snapshot_path.write_text(config_text)

    selection_payload = {
        "schema_version": _SPECIMEN_SCHEMA_VERSION,
        "specimen_id": str(specimen_id),
        "selected_rank": int(selected_rank),
        "selection_rule": dict(selection_rule),
        "selection_summary": dict(selection_summary),
    }
    _write_json(selection_summary_path, selection_payload)
    manifest_payload = {
        "schema_version": _SPECIMEN_SCHEMA_VERSION,
        "kind": "sensing_edges_specimen_manifest",
        "specimen_id": str(specimen_id),
        "selected_rank": int(selected_rank),
        "generation": int(generation),
        "run_id": str(run_manifest.get("run_id", "")),
        "run_dir_name": run_manifest_path.parent.name,
        "condition": run_metadata(cfg),
        "body": {
            "mode": str(cfg.body.mode),
            "shape": str(cfg.body.get("shape", "rectangle")),
            "robot_w": int(cfg.terrain.robot_w),
            "robot_h": int(cfg.terrain.robot_h),
        },
        "controller": {
            "nca_name": str(cfg.nca.get("name", "sbt")),
            "ctrnn_hidden": int(cfg.nca.ctrnn_hidden),
            "ctrnn_grid_hidden": int(cfg.nca.ctrnn_grid_hidden),
            "ctrnn_tau_min": float(cfg.nca.ctrnn_tau_min),
            "ctrnn_tau_max": float(cfg.nca.ctrnn_tau_max),
        },
        "selection": selection_payload,
        "run_provenance": {
            "config": dict(run_manifest.get("config", {})),
            "runtime": dict(run_manifest.get("runtime", {})),
            "hardware": dict(run_manifest.get("hardware", {})),
            "system": dict(run_manifest.get("system", {})),
            "git": dict(run_manifest.get("git", {})),
            "wandb": dict(run_manifest.get("wandb", {})),
            "lineage": dict(run_manifest.get("lineage", {})),
        },
        "paths": {
            "run_manifest": str(run_manifest_path),
            "resolved_config": str(run_resolved_config_path),
            "genotype": str(genotype_path),
            "controller_params": str(controller_params_path),
            "config_snapshot": str(config_snapshot_path),
            "selection_summary": str(selection_summary_path),
        },
        "preview_renders": {
            "preview_dir": str(preview_dir),
            "manifest_path": str(preview_manifest_path),
            "render_count": int(len(selection_summary.get("preview_artifacts", ()))),
            "preview_entry_indices": list(selection_summary.get("preview_entry_indices", ())),
            "renders": list(selection_summary.get("preview_artifacts", ())),
        },
        "post_run": None,
    }
    _write_json(manifest_path, manifest_payload)
    return {
        "specimen_dir": str(specimen_dir),
        "genotype_path": str(genotype_path),
        "controller_params_path": str(controller_params_path),
        "config_snapshot_path": str(config_snapshot_path),
        "selection_summary_path": str(selection_summary_path),
        "manifest_path": str(manifest_path),
    }


def materialize_historical_elite_artifacts(
    *,
    historical_root: str | Path,
    specimen_id: str,
    selection_summary: dict[str, Any],
    selection_rule: dict[str, Any],
    genotype: np.ndarray,
    cfg: DictConfig,
    run_manifest_path: str | Path,
    run_resolved_config_path: str | Path,
    generation: int,
    snapshot_payload: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Create a durable specimen-like folder for a historical elite saved mid-run."""
    historical_root = Path(historical_root)
    specimen_dir = historical_elite_directory(historical_root, specimen_id)
    specimen_dir.mkdir(parents=True, exist_ok=True)

    genotype = np.asarray(genotype, dtype=np.float32)
    run_manifest_path = Path(run_manifest_path)
    run_resolved_config_path = Path(run_resolved_config_path)
    run_manifest = load_manifest(run_manifest_path)
    config_text = run_resolved_config_path.read_text()
    params = ctrnn_flat_to_params(
        jax.numpy.asarray(genotype),
        ctrnn_hidden=int(cfg.nca.ctrnn_hidden),
        ctrnn_grid_hidden=int(cfg.nca.ctrnn_grid_hidden),
        gasnet_mode=str(cfg.nca.get("gasnet_mode", "off")),
        ou_enabled=bool(resolved_noise_settings(cfg)["ou_enabled"]),
        growth_enabled=str(cfg.body.mode).strip().lower() == "grown",
        active_comm_filters=active_comm_filters_from_cfg(cfg),
    )

    genotype_path = specimen_dir / "genotype.npz"
    controller_params_path = specimen_dir / "controller_params.npz"
    config_snapshot_path = specimen_dir / "config_snapshot.yaml"
    selection_summary_path = specimen_dir / "selection_summary.json"
    snapshot_path = specimen_dir / "historical_snapshot.npz"
    manifest_path = specimen_dir / "manifest.json"

    np.savez_compressed(genotype_path, genotype=genotype)
    np.savez_compressed(
        controller_params_path,
        w_input=np.asarray(params.w_input, dtype=np.float32),
        w_rec=np.asarray(params.w_rec, dtype=np.float32),
        w_output=np.asarray(params.w_output, dtype=np.float32),
        bias=np.asarray(params.bias, dtype=np.float32),
        log_tau=np.asarray(params.log_tau, dtype=np.float32),
    )
    config_snapshot_path.write_text(config_text)
    if snapshot_payload is not None:
        np.savez_compressed(snapshot_path, **snapshot_payload)

    selection_payload = {
        "schema_version": _SPECIMEN_SCHEMA_VERSION,
        "kind": "sensing_edges_historical_elite_summary",
        "specimen_id": str(specimen_id),
        "selection_rule": dict(selection_rule),
        "selection_summary": dict(selection_summary),
    }
    _write_json(selection_summary_path, selection_payload)

    manifest_payload = {
        "schema_version": _SPECIMEN_SCHEMA_VERSION,
        "kind": "sensing_edges_historical_elite_manifest",
        "specimen_id": str(specimen_id),
        "generation": int(generation),
        "run_id": str(run_manifest.get("run_id", "")),
        "run_dir_name": run_manifest_path.parent.name,
        "condition": run_metadata(cfg),
        "body": {
            "mode": str(cfg.body.mode),
            "shape": str(cfg.body.get("shape", "rectangle")),
            "robot_w": int(cfg.terrain.robot_w),
            "robot_h": int(cfg.terrain.robot_h),
        },
        "controller": {
            "nca_name": str(cfg.nca.get("name", "sbt")),
            "ctrnn_hidden": int(cfg.nca.ctrnn_hidden),
            "ctrnn_grid_hidden": int(cfg.nca.ctrnn_grid_hidden),
            "ctrnn_tau_min": float(cfg.nca.ctrnn_tau_min),
            "ctrnn_tau_max": float(cfg.nca.ctrnn_tau_max),
        },
        "selection": selection_payload,
        "run_provenance": {
            "config": dict(run_manifest.get("config", {})),
            "runtime": dict(run_manifest.get("runtime", {})),
            "hardware": dict(run_manifest.get("hardware", {})),
            "system": dict(run_manifest.get("system", {})),
            "git": dict(run_manifest.get("git", {})),
            "wandb": dict(run_manifest.get("wandb", {})),
            "lineage": dict(run_manifest.get("lineage", {})),
        },
        "paths": {
            "run_manifest": str(run_manifest_path),
            "resolved_config": str(run_resolved_config_path),
            "genotype": str(genotype_path),
            "controller_params": str(controller_params_path),
            "config_snapshot": str(config_snapshot_path),
            "selection_summary": str(selection_summary_path),
            "historical_snapshot": str(snapshot_path) if snapshot_payload is not None else None,
        },
        "preview_renders": None,
        "post_run": None,
    }
    _write_json(manifest_path, manifest_payload)
    return {
        "specimen_dir": str(specimen_dir),
        "genotype_path": str(genotype_path),
        "controller_params_path": str(controller_params_path),
        "config_snapshot_path": str(config_snapshot_path),
        "selection_summary_path": str(selection_summary_path),
        "historical_snapshot_path": str(snapshot_path) if snapshot_payload is not None else "",
        "manifest_path": str(manifest_path),
    }


def _assay_output_dir(analysis_root: str | Path, assay_name: str) -> Path:
    return Path(analysis_root) / str(assay_name)


def _archive_existing_requested_outputs(
    *,
    analysis_root: Path,
    requested_assays: list[str],
    index_path: Path,
    manifest_path: Path,
    started_at_utc: str,
) -> dict[str, str]:
    archive_root = analysis_root / "_previous" / (
        started_at_utc.replace(":", "-").replace("+", "_")
    )
    archived_paths: dict[str, str] = {}
    for assay_name in requested_assays:
        assay_path = _assay_output_dir(analysis_root, assay_name)
        if not assay_path.exists():
            continue
        archive_root.mkdir(parents=True, exist_ok=True)
        archived_path = archive_root / assay_path.name
        assay_path.rename(archived_path)
        archived_paths[str(assay_name)] = str(archived_path)
        log.info(
            "Archived stale post-run output | specimen_id=%s assay=%s archived_path=%s",
            analysis_root.name,
            assay_name,
            archived_path,
        )
    if manifest_path.exists():
        archive_root.mkdir(parents=True, exist_ok=True)
        archived_manifest_path = archive_root / manifest_path.name
        manifest_path.rename(archived_manifest_path)
        archived_paths["analysis_manifest"] = str(archived_manifest_path)
    if index_path.exists():
        archive_root.mkdir(parents=True, exist_ok=True)
        archived_index_path = archive_root / index_path.name
        index_path.rename(archived_index_path)
        archived_paths["post_run_index"] = str(archived_index_path)
    return archived_paths


def _write_post_run_pointer(
    *,
    pointer_path: Path,
    analysis_root: Path,
    analysis_index_path: Path,
    status: str,
    specimen_id: str,
    requested_assays: list[str],
    completed_assays: list[str],
    current_assay: str | None,
    failed_assay: str | None,
    updated_at_utc: str,
) -> None:
    _write_json(
        pointer_path,
        {
            "schema_version": _POST_RUN_SCHEMA_VERSION,
            "kind": "sensing_edges_post_run_pointer",
            "specimen_id": str(specimen_id),
            "status": str(status),
            "requested_assays": list(requested_assays),
            "completed_assays": list(completed_assays),
            "current_assay": current_assay,
            "failed_assay": failed_assay,
            "analysis_root": str(analysis_root),
            "analysis_index_path": str(analysis_index_path),
            "updated_at_utc": str(updated_at_utc),
        },
    )


def _update_analysis_manifest(
    manifest_path: Path,
    *,
    specimen_manifest: dict[str, Any],
    run_dir: Path,
    specimen_root: Path,
    specimen_dir: Path,
    assay_dirs: dict[str, str],
    index_path: Path,
    status: str,
    started_at_utc: str,
    requested_assays: list[str],
    completed_assays: list[str],
    current_assay: str | None,
    assay_progress: dict[str, dict[str, Any]],
    archived_outputs: dict[str, str],
    failed_assay: str | None,
    error_payload: dict[str, Any] | None,
) -> None:
    payload = {
        "schema_version": _POST_RUN_SCHEMA_VERSION,
        "kind": "sensing_edges_post_run_analysis_manifest",
        "run_id": str(specimen_manifest.get("run_id", "")),
        "run_dir": str(run_dir),
        "specimen_root": str(specimen_root),
        "specimen_id": str(specimen_manifest.get("specimen_id", specimen_dir.name.removeprefix("specimen_"))),
        "specimen_dir": str(specimen_dir),
        "analysis_root": str(manifest_path.parent),
        "selection": dict(specimen_manifest.get("selection", {})),
        "body": dict(specimen_manifest.get("body", {})),
        "controller": dict(specimen_manifest.get("controller", {})),
        "run_provenance": dict(specimen_manifest.get("run_provenance", {})),
        "paths": dict(specimen_manifest.get("paths", {})),
        "post_run": {
            "status": str(status),
            "started_at_utc": str(started_at_utc),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_at_utc": (
                datetime.now(timezone.utc).isoformat()
                if str(status) == "completed"
                else None
            ),
            "failed_at_utc": (
                datetime.now(timezone.utc).isoformat()
                if str(status) == "failed"
                else None
            ),
            "requested_assays": list(requested_assays),
            "completed_assays": list(completed_assays),
            "current_assay": current_assay,
            "assay_progress": dict(assay_progress),
            "assays": dict(assay_dirs),
            "archived_outputs": dict(archived_outputs),
            "index_path": str(index_path),
            "failed_assay": failed_assay,
            "error": error_payload,
        },
    }
    _write_json(manifest_path, payload)


def run_post_run_for_stored_specimen(
    *,
    cfg: DictConfig,
    runtime_resolution,
    run_dir: str | Path,
    specimen_root: str | Path,
    specimen_id: str | None = None,
    specimen_rank: int | None = None,
    assays: set[str] | None = None,
    raise_on_failure: bool = True,
    on_progress: Callable[[], None] | None = None,
) -> PostRunResults:
    """Run the configured post-run battery for one stored specimen."""
    run_dir = Path(run_dir)
    specimen_root = Path(specimen_root)
    row, specimen_dir = ensure_specimen_artifacts(
        cfg=cfg,
        run_dir=run_dir,
        specimen_root=specimen_root,
        specimen_id=specimen_id,
        specimen_rank=specimen_rank,
    )
    genotype = np.load(specimen_dir / "genotype.npz", allow_pickle=False)["genotype"]
    specimen_manifest_path = specimen_dir / "manifest.json"
    manifest = load_manifest(specimen_manifest_path)
    post_run_cfg = cfg.eval.post_run
    selected_assays = (
        set(_ASSAY_FAMILY_ORDER)
        if assays is None
        else {str(assay).strip() for assay in assays}
    )
    requested_assays = [assay for assay in _ASSAY_FAMILY_ORDER if assay in selected_assays]

    analysis_root = resolve_analysis_root(
        run_dir,
        str(row["specimen_id"]),
        output_root=str(post_run_cfg.get("analysis_root", "analyses")),
    )
    analysis_root.mkdir(parents=True, exist_ok=True)
    analysis_manifest_path = analysis_root / "manifest.json"
    index_path = analysis_root / "index.json"
    run_index_dir = specimen_root / str(post_run_cfg.get("output_dir", "post_run"))
    run_index_dir.mkdir(parents=True, exist_ok=True)
    pointer_path = run_index_dir / f"{str(row['specimen_id'])}.json"
    started_at_utc = datetime.now(timezone.utc).isoformat()
    archived_outputs = _archive_existing_requested_outputs(
        analysis_root=analysis_root,
        requested_assays=requested_assays,
        index_path=index_path,
        manifest_path=analysis_manifest_path,
        started_at_utc=started_at_utc,
    )
    assay_dirs: dict[str, str] = {
        assay_name: str(_assay_output_dir(analysis_root, assay_name))
        for assay_name in requested_assays
    }
    ran_assays: list[str] = []
    assay_progress: dict[str, dict[str, Any]] = {}
    primary_failed_assay: str | None = None
    primary_error_payload: dict[str, Any] | None = None
    first_exception: Exception | None = None

    native_rows: pd.DataFrame | None = None
    native_aggregate: dict[str, Any] | None = None
    current_assay: str | None = None

    def _invoke_progress_callback() -> None:
        if on_progress is None:
            return
        try:
            on_progress()
        except Exception:
            log.warning("Post-run progress callback failed", exc_info=True)

    def _persist_progress(
        *,
        status: str,
        failed_assay: str | None = None,
        error_payload: dict[str, Any] | None = None,
    ) -> None:
        now_utc = datetime.now(timezone.utc).isoformat()
        effective_failed_assay = failed_assay if failed_assay is not None else primary_failed_assay
        effective_error_payload = error_payload if error_payload is not None else primary_error_payload
        index_payload = {
            "schema_version": _POST_RUN_SCHEMA_VERSION,
            "kind": "sensing_edges_post_run_index",
            "run_id": str(manifest["run_id"]),
            "run_dir": str(run_dir),
            "specimen_root": str(specimen_root),
            "specimen_id": str(row["specimen_id"]),
            "specimen_dir": str(specimen_dir),
            "status": str(status),
            "started_at_utc": started_at_utc,
            "updated_at_utc": now_utc,
            "completed_at_utc": (now_utc if status == "completed" else None),
            "failed_at_utc": (now_utc if status == "failed" else None),
            "selection_rank": int(row.get("selected_rank", specimen_rank or 0)),
            "selection_summary": dict(row),
            "requested_assays": list(requested_assays),
            "completed_assays": list(ran_assays),
            "current_assay": current_assay,
            "assay_progress": dict(assay_progress),
            "assays": dict(assay_dirs),
            "archived_outputs": dict(archived_outputs),
            "failed_assay": effective_failed_assay,
            "error": effective_error_payload,
        }
        _write_json(index_path, index_payload)
        _update_analysis_manifest(
            analysis_manifest_path,
            specimen_manifest=manifest,
            run_dir=run_dir,
            specimen_root=specimen_root,
            specimen_dir=specimen_dir,
            assay_dirs=assay_dirs,
            index_path=index_path,
            status=status,
            started_at_utc=started_at_utc,
            requested_assays=requested_assays,
            completed_assays=ran_assays,
            current_assay=current_assay,
            assay_progress=assay_progress,
            archived_outputs=archived_outputs,
            failed_assay=effective_failed_assay,
            error_payload=effective_error_payload,
        )
        _update_specimen_manifest_post_run(
            specimen_manifest_path,
            assay_dirs=assay_dirs,
            index_path=index_path,
            analysis_root=analysis_root,
            analysis_manifest_path=analysis_manifest_path,
            status=status,
            started_at_utc=started_at_utc,
            requested_assays=requested_assays,
            completed_assays=ran_assays,
            current_assay=current_assay,
            assay_progress=assay_progress,
            archived_outputs=archived_outputs,
            failed_assay=effective_failed_assay,
            error_payload=effective_error_payload,
        )
        _write_post_run_pointer(
            pointer_path=pointer_path,
            analysis_root=analysis_root,
            analysis_index_path=index_path,
            status=status,
            specimen_id=str(row["specimen_id"]),
            requested_assays=requested_assays,
            completed_assays=ran_assays,
            current_assay=current_assay,
            failed_assay=effective_failed_assay,
            updated_at_utc=now_utc,
        )
        _invoke_progress_callback()

    def _mark_assay_started(assay_name: str) -> None:
        nonlocal current_assay
        current_assay = str(assay_name)
        assay_progress[current_assay] = {
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_at_utc": None,
            "failed_at_utc": None,
            "assay_dir": assay_dirs.get(current_assay),
        }
        log.info(
            "Post-run assay started | specimen_id=%s assay=%s requested=%s completed=%s",
            row["specimen_id"],
            current_assay,
            ",".join(requested_assays),
            ",".join(ran_assays),
        )
        _persist_progress(status="running")

    def _mark_assay_completed(assay_name: str, assay_dir: Path) -> None:
        nonlocal current_assay
        assay_progress[str(assay_name)] = {
            "status": "completed",
            "started_at_utc": assay_progress.get(str(assay_name), {}).get("started_at_utc"),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "failed_at_utc": None,
            "assay_dir": str(assay_dir),
        }
        log.info(
            "Post-run assay completed | specimen_id=%s assay=%s assay_dir=%s completed_count=%d/%d",
            row["specimen_id"],
            assay_name,
            assay_dir,
            len(ran_assays),
            len(requested_assays),
        )
        _persist_progress(status="running")
        current_assay = None

    def _mark_assay_failed(
        assay_name: str,
        *,
        error_payload: dict[str, Any],
        status: str,
    ) -> None:
        nonlocal current_assay, primary_failed_assay, primary_error_payload
        assay_progress[str(assay_name)] = {
            "status": status,
            "started_at_utc": assay_progress.get(str(assay_name), {}).get("started_at_utc"),
            "completed_at_utc": None,
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "assay_dir": assay_dirs.get(str(assay_name)),
            "error": error_payload,
        }
        if primary_failed_assay is None:
            primary_failed_assay = str(assay_name)
            primary_error_payload = dict(error_payload)
        if status == "skipped":
            log.warning(
                "Post-run assay skipped | specimen_id=%s assay=%s reason=%s",
                row["specimen_id"],
                assay_name,
                error_payload.get("message"),
            )
        else:
            log.error(
                "Post-run assay failed | specimen_id=%s assay=%s completed=%s error=%s",
                row["specimen_id"],
                assay_name,
                ",".join(ran_assays),
                error_payload.get("message"),
            )
        _persist_progress(status="running")
        current_assay = None

    def _dependency_unavailable(assay_name: str, dependency: str) -> None:
        _mark_assay_failed(
            assay_name,
            error_payload={
                "type": "DependencyUnavailable",
                "message": f"{assay_name} requires completed {dependency}",
                "traceback": None,
            },
            status="skipped",
        )

    log.info(
        "Post-run started | specimen_id=%s run_dir=%s specimen_root=%s requested_assays=%s",
        row["specimen_id"],
        run_dir,
        specimen_root,
        ",".join(requested_assays),
    )
    _persist_progress(status="running")

    for assay_name in requested_assays:
        if assay_name in {"initial_direction", "long_distance_generalization", "body_transfer"} and (
            native_rows is None or native_aggregate is None
        ):
            _dependency_unavailable(assay_name, "native_core")
            continue

        _mark_assay_started(assay_name)
        try:
            if assay_name == "pca_bundle":
                assay_dir = _run_pca_bundle(
                    cfg=cfg,
                    runtime_resolution=runtime_resolution,
                    specimen_dir=analysis_root,
                    specimen_id=str(row["specimen_id"]),
                    genotype=genotype,
                    selection_row=row,
                    run_id=str(manifest["run_id"]),
                )
                assay_dirs["pca_bundle"] = str(assay_dir)
                ran_assays.append("pca_bundle")
                _mark_assay_completed("pca_bundle", assay_dir)
            elif assay_name == "native_core":
                native_rows, native_aggregate, assay_dir = _run_native_core_assay(
                    cfg=cfg,
                    runtime_resolution=runtime_resolution,
                    specimen_dir=analysis_root,
                    specimen_id=str(row["specimen_id"]),
                    genotype=genotype,
                    selection_row=row,
                    run_id=str(manifest["run_id"]),
                )
                assay_dirs["native_core"] = str(assay_dir)
                ran_assays.append("native_core")
                _mark_assay_completed("native_core", assay_dir)
            elif assay_name == "initial_direction":
                assay_dir = _run_initial_direction_summary(
                    specimen_dir=analysis_root,
                    native_rows=native_rows,
                    specimen_id=str(row["specimen_id"]),
                )
                assay_dirs["initial_direction"] = str(assay_dir)
                ran_assays.append("initial_direction")
                _mark_assay_completed("initial_direction", assay_dir)
            elif assay_name == "long_distance_generalization":
                _, aggregate, assay_dir = _run_long_distance_assay(
                    cfg=cfg,
                    runtime_resolution=runtime_resolution,
                    specimen_dir=analysis_root,
                    specimen_id=str(row["specimen_id"]),
                    genotype=genotype,
                    selection_row=row,
                    run_id=str(manifest["run_id"]),
                    native_rows=native_rows,
                    native_aggregate=native_aggregate,
                )
                _ = aggregate
                assay_dirs["long_distance_generalization"] = str(assay_dir)
                ran_assays.append("long_distance_generalization")
                _mark_assay_completed("long_distance_generalization", assay_dir)
            elif assay_name == "double_cliff":
                _, aggregate, assay_dir = _run_double_cliff_assay(
                    cfg=cfg,
                    runtime_resolution=runtime_resolution,
                    specimen_dir=analysis_root,
                    specimen_id=str(row["specimen_id"]),
                    genotype=genotype,
                    selection_row=row,
                    run_id=str(manifest["run_id"]),
                )
                _ = aggregate
                assay_dirs["double_cliff"] = str(assay_dir)
                ran_assays.append("double_cliff")
                _mark_assay_completed("double_cliff", assay_dir)
            elif assay_name == "signaling_ablation":
                _, aggregate, assay_dir = _run_signaling_ablation_assay(
                    cfg=cfg,
                    runtime_resolution=runtime_resolution,
                    specimen_dir=analysis_root,
                    specimen_id=str(row["specimen_id"]),
                    genotype=genotype,
                    selection_row=row,
                    run_id=str(manifest["run_id"]),
                )
                _ = aggregate
                assay_dirs["signaling_ablation"] = str(assay_dir)
                ran_assays.append("signaling_ablation")
                _mark_assay_completed("signaling_ablation", assay_dir)
            elif assay_name == "deformation_ablation":
                _, aggregate, assay_dir = _run_deformation_ablation_assay(
                    cfg=cfg,
                    runtime_resolution=runtime_resolution,
                    specimen_dir=analysis_root,
                    specimen_id=str(row["specimen_id"]),
                    genotype=genotype,
                    selection_row=row,
                    run_id=str(manifest["run_id"]),
                )
                _ = aggregate
                assay_dirs["deformation_ablation"] = str(assay_dir)
                ran_assays.append("deformation_ablation")
                _mark_assay_completed("deformation_ablation", assay_dir)
            elif assay_name == "frozen_sensing_control":
                _, aggregate, assay_dir = _run_frozen_sensing_control_assay(
                    cfg=cfg,
                    runtime_resolution=runtime_resolution,
                    specimen_dir=analysis_root,
                    specimen_id=str(row["specimen_id"]),
                    genotype=genotype,
                    selection_row=row,
                    run_id=str(manifest["run_id"]),
                )
                _ = aggregate
                assay_dirs["frozen_sensing_control"] = str(assay_dir)
                ran_assays.append("frozen_sensing_control")
                _mark_assay_completed("frozen_sensing_control", assay_dir)
            elif assay_name == "body_transfer":
                _, aggregate, assay_dir = _run_body_transfer_assay(
                    cfg=cfg,
                    runtime_resolution=runtime_resolution,
                    specimen_dir=analysis_root,
                    specimen_id=str(row["specimen_id"]),
                    genotype=genotype,
                    selection_row=row,
                    run_id=str(manifest["run_id"]),
                    native_rows=native_rows,
                    native_aggregate=native_aggregate,
                )
                _ = aggregate
                assay_dirs["body_transfer"] = str(assay_dir)
                ran_assays.append("body_transfer")
                _mark_assay_completed("body_transfer", assay_dir)
            elif assay_name == "state_space_trace":
                _, aggregate, assay_dir = _run_state_space_trace_assay(
                    cfg=cfg,
                    runtime_resolution=runtime_resolution,
                    specimen_dir=analysis_root,
                    specimen_id=str(row["specimen_id"]),
                    genotype=genotype,
                    selection_row=row,
                    run_id=str(manifest["run_id"]),
                )
                _ = aggregate
                assay_dirs["state_space_trace"] = str(assay_dir)
                ran_assays.append("state_space_trace")
                _mark_assay_completed("state_space_trace", assay_dir)
            elif assay_name == "state_space_trace_deformation_off":
                _, aggregate, assay_dir = _run_state_space_trace_assay(
                    cfg=cfg,
                    runtime_resolution=runtime_resolution,
                    specimen_dir=analysis_root,
                    specimen_id=str(row["specimen_id"]),
                    genotype=genotype,
                    selection_row=row,
                    run_id=str(manifest["run_id"]),
                    assay_name="state_space_trace_deformation_off",
                    deformation_condition="deformation_off",
                    signaling_condition="intact",
                )
                _ = aggregate
                assay_dirs["state_space_trace_deformation_off"] = str(assay_dir)
                ran_assays.append("state_space_trace_deformation_off")
                _mark_assay_completed("state_space_trace_deformation_off", assay_dir)
            elif assay_name == "state_space_trace_edges_deformation_off":
                _, aggregate, assay_dir = _run_state_space_trace_assay(
                    cfg=cfg,
                    runtime_resolution=runtime_resolution,
                    specimen_dir=analysis_root,
                    specimen_id=str(row["specimen_id"]),
                    genotype=genotype,
                    selection_row=row,
                    run_id=str(manifest["run_id"]),
                    assay_name="state_space_trace_edges_deformation_off",
                    deformation_condition="deformation_off",
                    signaling_condition="intact",
                )
                _ = aggregate
                assay_dirs["state_space_trace_edges_deformation_off"] = str(assay_dir)
                ran_assays.append("state_space_trace_edges_deformation_off")
                _mark_assay_completed("state_space_trace_edges_deformation_off", assay_dir)
            elif assay_name == "combined_ablation_quick":
                _, aggregate, assay_dir = _run_combined_ablation_quick_assay(
                    cfg=cfg,
                    runtime_resolution=runtime_resolution,
                    specimen_dir=analysis_root,
                    specimen_id=str(row["specimen_id"]),
                    genotype=genotype,
                    selection_row=row,
                    run_id=str(manifest["run_id"]),
                )
                _ = aggregate
                assay_dirs["combined_ablation_quick"] = str(assay_dir)
                ran_assays.append("combined_ablation_quick")
                _mark_assay_completed("combined_ablation_quick", assay_dir)
            elif assay_name == "tau":
                assay_dir = _run_tau_analysis(
                    cfg=cfg,
                    specimen_dir=analysis_root,
                    specimen_id=str(row["specimen_id"]),
                    genotype=genotype,
                )
                assay_dirs["tau"] = str(assay_dir)
                ran_assays.append("tau")
                _mark_assay_completed("tau", assay_dir)
            else:
                raise ValueError(f"Unsupported post-run assay: {assay_name}")
        except Exception as exc:
            if first_exception is None:
                first_exception = exc
            _mark_assay_failed(
                assay_name,
                error_payload=_exception_payload(exc),
                status="failed",
            )
            continue

    current_assay = None
    final_status = "completed" if primary_failed_assay is None else "failed"
    if final_status == "completed":
        log.info(
            "Post-run completed | specimen_id=%s completed_assays=%s index_path=%s",
            row["specimen_id"],
            ",".join(ran_assays),
            index_path,
        )
    else:
        log.warning(
            "Post-run completed with failures | specimen_id=%s failed_assay=%s completed_assays=%s",
            row["specimen_id"],
            primary_failed_assay,
            ",".join(ran_assays),
        )
    _persist_progress(
        status=final_status,
        failed_assay=primary_failed_assay,
        error_payload=primary_error_payload,
    )
    results = PostRunResults(
        specimen_id=str(row["specimen_id"]),
        specimen_dir=specimen_dir,
        index_path=index_path,
        assay_dirs=assay_dirs,
        ran_assays=ran_assays,
        status=final_status,
        failed_assay=primary_failed_assay,
        error_message=(
            None if primary_error_payload is None else str(primary_error_payload.get("message") or "")
        ),
    )
    if final_status != "completed" and raise_on_failure and first_exception is not None:
        raise first_exception
    return results


def _update_specimen_manifest_post_run(
    manifest_path: Path,
    *,
    assay_dirs: dict[str, str],
    index_path: Path,
    analysis_root: Path,
    analysis_manifest_path: Path,
    status: str,
    started_at_utc: str,
    requested_assays: list[str],
    completed_assays: list[str],
    current_assay: str | None,
    assay_progress: dict[str, dict[str, Any]],
    archived_outputs: dict[str, str],
    failed_assay: str | None,
    error_payload: dict[str, Any] | None,
) -> None:
    payload = load_manifest(manifest_path)
    payload["post_run"] = {
        "status": str(status),
        "started_at_utc": str(started_at_utc),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_at_utc": (
            datetime.now(timezone.utc).isoformat()
            if str(status) == "completed"
            else None
        ),
        "failed_at_utc": (
            datetime.now(timezone.utc).isoformat()
            if str(status) == "failed"
            else None
        ),
        "requested_assays": list(requested_assays),
        "completed_assays": list(completed_assays),
        "current_assay": current_assay,
        "assay_progress": dict(assay_progress),
        "assays": dict(assay_dirs),
        "archived_outputs": dict(archived_outputs),
        "analysis_root": str(analysis_root),
        "analysis_manifest_path": str(analysis_manifest_path),
        "index_path": str(index_path),
        "failed_assay": failed_assay,
        "error": error_payload,
    }
    _write_json(manifest_path, payload)


def _validate_unique_batched_specimen_ids(run_specs: Sequence[dict[str, Any]]) -> None:
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for spec in run_specs:
        specimen_id = str(spec["specimen_id"])
        run_dir = str(spec["run_dir"])
        if specimen_id in seen:
            duplicates.append(f"{specimen_id} ({seen[specimen_id]}, {run_dir})")
        else:
            seen[specimen_id] = run_dir
    if duplicates:
        raise ValueError(
            "Batched post-run requires unique specimen_id values across run_specs; duplicates: "
            + ", ".join(duplicates)
        )


def _persist_batched_job_progress(
    *,
    job: BatchedPostRunJob,
    assay_dirs: dict[str, str],
    started_at_utc: str,
    requested_assays: list[str],
    completed_assays: list[str],
    current_assay: str | None,
    assay_progress: dict[str, dict[str, Any]],
    archived_outputs: dict[str, str],
    status: str,
    failed_assay: str | None = None,
    error_payload: dict[str, Any] | None = None,
) -> None:
    now_utc = datetime.now(timezone.utc).isoformat()
    index_payload = {
        "schema_version": _POST_RUN_SCHEMA_VERSION,
        "kind": "sensing_edges_post_run_index",
        "run_id": str(job.run_id),
        "run_dir": str(job.run_dir),
        "specimen_root": str(job.specimen_root),
        "specimen_id": str(job.specimen_id),
        "specimen_dir": str(job.specimen_dir),
        "status": str(status),
        "started_at_utc": str(started_at_utc),
        "updated_at_utc": now_utc,
        "completed_at_utc": (now_utc if status == "completed" else None),
        "failed_at_utc": (now_utc if status == "failed" else None),
        "selection_rank": int(job.selection_row.get("selected_rank", 0)),
        "selection_summary": dict(job.selection_row),
        "requested_assays": list(requested_assays),
        "completed_assays": list(completed_assays),
        "current_assay": current_assay,
        "assay_progress": dict(assay_progress),
        "assays": dict(assay_dirs),
        "archived_outputs": dict(archived_outputs),
        "failed_assay": failed_assay,
        "error": error_payload,
    }
    _write_json(job.analysis_index_path, index_payload)
    _update_analysis_manifest(
        job.analysis_manifest_path,
        specimen_manifest=job.specimen_manifest,
        run_dir=job.run_dir,
        specimen_root=job.specimen_root,
        specimen_dir=job.specimen_dir,
        assay_dirs=assay_dirs,
        index_path=job.analysis_index_path,
        status=status,
        started_at_utc=started_at_utc,
        requested_assays=requested_assays,
        completed_assays=completed_assays,
        current_assay=current_assay,
        assay_progress=assay_progress,
        archived_outputs=archived_outputs,
        failed_assay=failed_assay,
        error_payload=error_payload,
    )
    _update_specimen_manifest_post_run(
        job.specimen_manifest_path,
        assay_dirs=assay_dirs,
        index_path=job.analysis_index_path,
        analysis_root=job.analysis_root,
        analysis_manifest_path=job.analysis_manifest_path,
        status=status,
        started_at_utc=started_at_utc,
        requested_assays=requested_assays,
        completed_assays=completed_assays,
        current_assay=current_assay,
        assay_progress=assay_progress,
        archived_outputs=archived_outputs,
        failed_assay=failed_assay,
        error_payload=error_payload,
    )
    _write_post_run_pointer(
        pointer_path=job.pointer_path,
        analysis_root=job.analysis_root,
        analysis_index_path=job.analysis_index_path,
        status=status,
        specimen_id=str(job.specimen_id),
        requested_assays=requested_assays,
        completed_assays=completed_assays,
        current_assay=current_assay,
        failed_assay=failed_assay,
        updated_at_utc=now_utc,
    )


def _exception_payload(exc: Exception) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _require_native_summary(
    native_rows: pd.DataFrame | None,
    native_aggregate: dict[str, Any] | None,
) -> None:
    if native_rows is None or native_aggregate is None:
        raise RuntimeError("The requested assay depends on native_core results")


def _clone_cfg(cfg: DictConfig) -> DictConfig:
    if GlobalHydra.instance().is_initialized():
        pass
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


def _stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], byteorder="big", signed=False)


def _side_code(side: str) -> int:
    normalized = str(side).strip().lower()
    if normalized == "left":
        return -1
    if normalized == "right":
        return 1
    return 0


def _pair_label(goal_distance: int, cliff_distance: int) -> str:
    return f"({int(goal_distance)},{int(cliff_distance)})"


def _pair_key(goal_distance: int, cliff_distance: int) -> tuple[int, int]:
    return (int(goal_distance), int(cliff_distance))


def _humanize_token(token: str) -> str:
    return str(token).replace("_", " ").strip().title()


def _surface_label(surface_id: str) -> str:
    return _SURFACE_LABELS.get(str(surface_id), _humanize_token(surface_id))


def _signaling_label(condition: str) -> str:
    return _SIGNALING_LABELS.get(str(condition), _humanize_token(condition))


def _deformation_label(condition: str) -> str:
    return _DEFORMATION_LABELS.get(str(condition), _humanize_token(condition))


def _sensing_freeze_label(condition: str) -> str:
    return _SENSING_FREEZE_LABELS.get(str(condition), _humanize_token(condition))


def _body_label(body_label: str) -> str:
    return str(body_label).replace("_", " ").strip()


def _long_distance_pair_step_map(cfg: DictConfig) -> dict[tuple[int, int], int]:
    post_run_cfg = cfg.eval.post_run
    fallback_steps = int(post_run_cfg.get("max_steps", 5000))
    pair_step_map: dict[tuple[int, int], int] = {}
    for pair_cfg in list(post_run_cfg.get("long_distance_pair_max_steps") or []):
        pair_step_map[_pair_key(pair_cfg["goal_distance"], pair_cfg["cliff_distance"])] = int(
            pair_cfg.get("max_steps", fallback_steps)
        )
    for pair_cfg in list(post_run_cfg.long_distance_pairs):
        pair_step_map.setdefault(
            _pair_key(pair_cfg["goal_distance"], pair_cfg["cliff_distance"]),
            fallback_steps,
        )
    return pair_step_map


def _native_max_steps(cfg: DictConfig) -> int:
    post_run_cfg = cfg.eval.post_run
    return int(post_run_cfg.get("native_max_steps", post_run_cfg.get("max_steps", 5000)))


def _double_cliff_max_steps(cfg: DictConfig) -> int:
    post_run_cfg = cfg.eval.post_run
    return int(post_run_cfg.get("double_cliff_max_steps", post_run_cfg.get("max_steps", 5000)))


def _long_distance_max_steps(cfg: DictConfig, *, goal_distance: int, cliff_distance: int) -> int:
    pair_step_map = _long_distance_pair_step_map(cfg)
    return int(
        pair_step_map.get(
            _pair_key(goal_distance, cliff_distance),
            cfg.eval.post_run.get("max_steps", 5000),
        )
    )


def _stochastic_deformation_condition(condition: str | None) -> bool:
    normalized = str(condition or "").strip().lower()
    return normalized in {"noise_only", "noisy", "shuffled"}


def _post_run_condition_is_stochastic(
    cfg: DictConfig,
    *,
    deformation_condition: str | None = None,
) -> bool:
    if bool(resolved_noise_settings(cfg)["ou_enabled"]):
        return True
    if float(cfg.nca.get("p_update", 1.0)) < 1.0:
        return True
    active_deformation = (
        str(deformation_condition)
        if deformation_condition is not None
        else str(cfg.eval.post_run.get("active_deformation_condition", "clean"))
    )
    return _stochastic_deformation_condition(active_deformation)


def _effective_trials_per_side(
    cfg: DictConfig,
    *,
    deformation_condition: str | None = None,
) -> int:
    requested = int(cfg.eval.post_run.trials_per_side)
    if _post_run_condition_is_stochastic(cfg, deformation_condition=deformation_condition):
        capped = cfg.eval.post_run.get("stochastic_trials_per_side")
        if capped is None:
            return requested
        return max(1, min(requested, int(capped)))
    return 1


def _effective_trials_per_width(
    cfg: DictConfig,
    *,
    deformation_condition: str | None = None,
) -> int:
    requested = int(cfg.eval.post_run.double_cliff.trials_per_width)
    if _post_run_condition_is_stochastic(cfg, deformation_condition=deformation_condition):
        capped = cfg.eval.post_run.double_cliff.get("stochastic_trials_per_width")
        if capped is None:
            return requested
        return max(1, min(requested, int(capped)))
    return 1


def _effective_trace_trials_per_side(
    cfg: DictConfig,
    *,
    deformation_condition: str | None = None,
) -> int:
    requested = int(cfg.eval.post_run.trace.requested_trials_per_side)
    if _post_run_condition_is_stochastic(cfg, deformation_condition=deformation_condition):
        capped = cfg.eval.post_run.trace.get(
            "stochastic_requested_trials_per_side",
            cfg.eval.post_run.get("stochastic_trials_per_side"),
        )
        if capped is None:
            return requested
        return max(1, min(requested, int(capped)))
    return 1


def _fast_ablation_eval_enabled(cfg: DictConfig) -> bool:
    return bool(cfg.eval.post_run.get("fast_ablation_eval", False))


def _normalized_text_list(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    return tuple(
        token
        for token in (
            str(value).strip().lower()
            for value in list(values)
        )
        if token
    )


def _configured_goal_sides(
    cfg: DictConfig,
    *,
    trace_cfg: DictConfig | None = None,
) -> tuple[str, ...]:
    configured = ()
    if trace_cfg is not None and "goal_sides" in trace_cfg:
        configured = _normalized_text_list(trace_cfg.get("goal_sides"))
    if not configured:
        configured = _normalized_text_list(cfg.eval.post_run.get("goal_sides"))
    if not configured:
        return _GOAL_SIDES
    invalid = sorted(set(configured) - set(_GOAL_SIDES))
    if invalid:
        raise ValueError(f"Unsupported goal_sides={invalid!r}; expected a subset of {_GOAL_SIDES!r}")
    ordered: list[str] = []
    seen: set[str] = set()
    for side in configured:
        if side not in seen:
            ordered.append(side)
            seen.add(side)
    return tuple(ordered) if ordered else _GOAL_SIDES


def _should_render_signaling_condition(cfg: DictConfig, condition: str) -> bool:
    configured = _normalized_text_list(cfg.eval.post_run.get("signaling_render_conditions"))
    if not configured:
        return True
    return str(condition).strip().lower() in configured


def _should_render_deformation_condition(cfg: DictConfig, condition: str) -> bool:
    configured = _normalized_text_list(cfg.eval.post_run.get("deformation_render_conditions"))
    if not configured:
        return True
    return str(condition).strip().lower() in configured


def _should_render_body_label(
    cfg: DictConfig,
    *,
    body_label: str,
    reference_body_label: str,
) -> bool:
    configured = _normalized_text_list(cfg.eval.post_run.get("body_transfer_render_bodies"))
    if not configured:
        return True
    normalized_label = str(body_label).strip().lower()
    normalized_reference = str(reference_body_label).strip().lower()
    return (
        normalized_label in configured
        or ("reference" in configured and normalized_label == normalized_reference)
    )


def _combined_ablation_tests(cfg: DictConfig) -> list[dict[str, str]]:
    tests: list[dict[str, str]] = []
    for raw in list(cfg.eval.post_run.get("combined_ablation_quick_tests", [])):
        payload = (
            OmegaConf.to_container(raw, resolve=True)
            if OmegaConf.is_config(raw)
            else dict(raw)
        )
        test_id = str(payload.get("test_id", "")).strip()
        if not test_id:
            raise ValueError("combined_ablation_quick_tests entries require test_id")
        deformation_condition = str(payload.get("deformation_condition", "")).strip()
        signaling_condition = str(payload.get("signaling_condition", "")).strip()
        if not deformation_condition or not signaling_condition:
            raise ValueError(
                "combined_ablation_quick_tests entries require deformation_condition and signaling_condition"
            )
        tests.append(
            {
                "test_id": test_id,
                "label": str(payload.get("label", test_id)).strip() or test_id,
                "deformation_condition": deformation_condition,
                "signaling_condition": signaling_condition,
            }
        )
    return tests


def _flatten_selected_episode_results(result) -> Any:
    return jax.tree_util.tree_map(lambda array: np.asarray(array[0]), result)


def _phase_cfg_for_assay(
    cfg: DictConfig,
    *,
    max_steps: int,
    active_deformation_condition: str | None = None,
    active_signaling_condition: str | None = None,
    active_sensing_freeze_mode: str | None = None,
    body_label: str | None = None,
    robot_w: int | None = None,
    robot_h: int | None = None,
    body_shape: str | None = None,
) -> DictConfig:
    phase_cfg = _clone_cfg(cfg)
    phase_cfg.train.max_steps = int(max_steps)
    if active_deformation_condition is not None:
        phase_cfg.eval.post_run.active_deformation_condition = str(active_deformation_condition)
    if active_signaling_condition is not None:
        phase_cfg.eval.post_run.active_signaling_condition = str(active_signaling_condition)
    else:
        phase_cfg.eval.post_run.active_signaling_condition = "intact"
    if active_sensing_freeze_mode is not None:
        phase_cfg.eval.post_run.active_sensing_freeze_mode = str(active_sensing_freeze_mode)
    else:
        phase_cfg.eval.post_run.active_sensing_freeze_mode = "none"
    if body_label is not None:
        phase_cfg.eval.post_run.active_body_label = str(body_label)
    if robot_w is not None:
        phase_cfg.terrain.robot_w = int(robot_w)
    if robot_h is not None:
        phase_cfg.terrain.robot_h = int(robot_h)
    if body_shape is not None:
        phase_cfg.body.shape = str(body_shape)
    return phase_cfg


def _native_phase_cfg(
    cfg: DictConfig,
    *,
    active_deformation_condition: str | None = None,
    active_signaling_condition: str | None = None,
    active_sensing_freeze_mode: str | None = None,
    body_label: str | None = None,
    robot_w: int | None = None,
    robot_h: int | None = None,
    body_shape: str | None = None,
) -> DictConfig:
    return _phase_cfg_for_assay(
        cfg,
        max_steps=_native_max_steps(cfg),
        active_deformation_condition=active_deformation_condition,
        active_signaling_condition=active_signaling_condition,
        active_sensing_freeze_mode=active_sensing_freeze_mode,
        body_label=body_label,
        robot_w=robot_w,
        robot_h=robot_h,
        body_shape=body_shape,
    )


def _double_cliff_phase_cfg(
    cfg: DictConfig,
    *,
    active_deformation_condition: str | None = None,
    active_signaling_condition: str | None = None,
    active_sensing_freeze_mode: str | None = None,
    body_label: str | None = None,
    robot_w: int | None = None,
    robot_h: int | None = None,
    body_shape: str | None = None,
) -> DictConfig:
    return _phase_cfg_for_assay(
        cfg,
        max_steps=_double_cliff_max_steps(cfg),
        active_deformation_condition=active_deformation_condition,
        active_signaling_condition=active_signaling_condition,
        active_sensing_freeze_mode=active_sensing_freeze_mode,
        body_label=body_label,
        robot_w=robot_w,
        robot_h=robot_h,
        body_shape=body_shape,
    )


def _long_distance_phase_cfg(
    cfg: DictConfig,
    *,
    goal_distance: int | None = None,
    cliff_distance: int | None = None,
    active_deformation_condition: str | None = None,
    active_signaling_condition: str | None = None,
    active_sensing_freeze_mode: str | None = None,
    body_label: str | None = None,
    robot_w: int | None = None,
    robot_h: int | None = None,
    body_shape: str | None = None,
) -> DictConfig:
    post_run_cfg = cfg.eval.post_run
    if goal_distance is None or cliff_distance is None:
        max_steps = int(post_run_cfg.get("max_steps", 5000))
    else:
        max_steps = _long_distance_max_steps(
            cfg,
            goal_distance=int(goal_distance),
            cliff_distance=int(cliff_distance),
        )
    phase_cfg = _phase_cfg_for_assay(
        cfg,
        max_steps=max_steps,
        active_deformation_condition=active_deformation_condition,
        active_signaling_condition=active_signaling_condition,
        active_sensing_freeze_mode=active_sensing_freeze_mode,
        body_label=body_label,
        robot_w=robot_w,
        robot_h=robot_h,
        body_shape=body_shape,
    )
    terrain_overrides = post_run_cfg.get("long_distance_terrain")
    if terrain_overrides is None:
        return phase_cfg
    grid_w = terrain_overrides.get("grid_w")
    if grid_w is not None:
        phase_cfg.terrain.grid_w = int(grid_w)
    plateau_width = terrain_overrides.get("plateau_width")
    if plateau_width is not None:
        phase_cfg.terrain.center_plateau_width = int(plateau_width)
        phase_cfg.terrain.base_width = int(plateau_width)
    return phase_cfg


def _build_bilateral_entry(
    phase_cfg: DictConfig,
    *,
    goal_distance: int,
    cliff_distance: int,
    goal_side: str,
    extra_metadata: dict[str, Any] | None = None,
) -> TerrainEntry:
    terrain_cfg = phase_cfg.terrain
    grid, metadata = build_bilateral_goal_cliff_terrain(
        int(terrain_cfg.grid_h),
        int(terrain_cfg.grid_w),
        goal_distance=int(goal_distance),
        cliff_distance=int(cliff_distance),
        goal_side=str(goal_side),
        plateau_width=int(terrain_cfg.get("center_plateau_width", terrain_cfg.get("base_width", 10))),
        platform_y=int(terrain_cfg.get("platform_y", terrain_cfg.get("plateau_height", 3))),
        correct_side_distance_scale=float(terrain_cfg.get("correct_side_distance_scale", 1.0)),
        incorrect_side_distance_scale=float(terrain_cfg.get("incorrect_side_distance_scale", 1.0)),
        cliff_front_gap_width=int(terrain_cfg.get("cliff_front_gap_width", 0)),
        cliff_post_gap_runway=int(terrain_cfg.get("cliff_post_gap_runway", 0)),
        cliff_probe_gap_width=int(terrain_cfg.get("cliff_probe_gap_width", 0)),
        cliff_probe_ledge_width=int(terrain_cfg.get("cliff_probe_ledge_width", 0)),
    )
    merged_metadata = dict(metadata)
    merged_metadata["assay_goal_side"] = str(goal_side)
    if extra_metadata:
        merged_metadata.update(dict(extra_metadata))
    front_gap_width = int(merged_metadata.get("cliff_front_gap_width", 0))
    gaps = (
        [(int(merged_metadata["cliff_front_gap_start_x"]), front_gap_width)]
        if front_gap_width > 0 and int(merged_metadata.get("cliff_front_gap_start_x", -1)) >= 0
        else []
    )
    cliff_x = int(
        merged_metadata.get(
            "terminal_cliff_edge_x",
            merged_metadata.get("cliff_edge_x", merged_metadata.get("platform_end_x", 0)),
        )
    )
    return TerrainEntry(grid=grid, cliff_x=cliff_x, gaps=gaps, metadata=merged_metadata)


def _build_double_cliff_entry(
    phase_cfg: DictConfig,
    *,
    width: int,
    plateau_width: int,
    extra_metadata: dict[str, Any] | None = None,
) -> TerrainEntry:
    grid, platform_start, platform_end = build_symmetric_double_cliff_terrain(
        int(phase_cfg.terrain.grid_h),
        int(phase_cfg.terrain.grid_w),
        int(width),
        plateau_width=int(plateau_width),
        platform_y=int(phase_cfg.terrain.get("platform_y", phase_cfg.terrain.get("plateau_height", 3))),
    )
    metadata = {
        "task_mode": "legacy_cliff",
        "platform_start_x": int(platform_start),
        "platform_end_x": int(platform_end),
        "width": int(width),
        "double_cliff_width": int(width),
        "assay_width": int(width),
    }
    if extra_metadata:
        metadata.update(dict(extra_metadata))
    return TerrainEntry(
        grid=grid,
        cliff_x=int(platform_end),
        gaps=[],
        metadata=metadata,
    )


def _build_native_entries(
    phase_cfg: DictConfig,
    pairs: list[dict[str, int]],
    *,
    trials_per_side: int,
    goal_sides: tuple[str, ...] = _GOAL_SIDES,
) -> list[TerrainEntry]:
    entries: list[TerrainEntry] = []
    for pair in pairs:
        goal_distance = int(pair["goal_distance"])
        cliff_distance = int(pair["cliff_distance"])
        for goal_side in goal_sides:
            for trial_repeat in range(int(trials_per_side)):
                entries.append(
                    _build_bilateral_entry(
                        phase_cfg,
                        goal_distance=goal_distance,
                        cliff_distance=cliff_distance,
                        goal_side=goal_side,
                        extra_metadata={
                            "distance_pair_label": _pair_label(goal_distance, cliff_distance),
                            "trial_repeat_index": int(trial_repeat),
                        },
                    )
                )
    return entries


def _build_targeted_trace_entries(
    phase_cfg: DictConfig,
    *,
    goal_distance: int,
    cliff_distance: int,
    trials_per_side: int,
    goal_sides: tuple[str, ...] = _GOAL_SIDES,
) -> list[TerrainEntry]:
    entries: list[TerrainEntry] = []
    for goal_side in goal_sides:
        for trial_repeat in range(int(trials_per_side)):
            entries.append(
                _build_bilateral_entry(
                    phase_cfg,
                    goal_distance=int(goal_distance),
                    cliff_distance=int(cliff_distance),
                    goal_side=goal_side,
                    extra_metadata={
                        "distance_pair_label": _pair_label(goal_distance, cliff_distance),
                        "trial_repeat_index": int(trial_repeat),
                        "pca_bundle_target": True,
                    },
                )
            )
    return entries


def _native_like_pair_groups(
    *,
    cfg: DictConfig,
    surface_id: str,
    pairs: list[dict[str, int]],
    trials_per_side: int,
    signaling_condition: str,
    deformation_condition: str,
    sensing_freeze_mode: str,
    body_label: str,
    robot_w: int | None,
    robot_h: int | None,
    body_shape: str | None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    goal_sides = _configured_goal_sides(cfg)
    for pair in pairs:
        goal_distance = int(pair["goal_distance"])
        cliff_distance = int(pair["cliff_distance"])
        if surface_id == "long_distance_generalization":
            phase_cfg = _long_distance_phase_cfg(
                cfg,
                goal_distance=goal_distance,
                cliff_distance=cliff_distance,
                active_deformation_condition=deformation_condition,
                active_signaling_condition=signaling_condition,
                active_sensing_freeze_mode=sensing_freeze_mode,
                body_label=body_label,
                robot_w=robot_w,
                robot_h=robot_h,
                body_shape=body_shape,
            )
            max_steps = _long_distance_max_steps(
                cfg,
                goal_distance=goal_distance,
                cliff_distance=cliff_distance,
            )
        else:
            phase_cfg = _native_phase_cfg(
                cfg,
                active_deformation_condition=deformation_condition,
                active_signaling_condition=signaling_condition,
                active_sensing_freeze_mode=sensing_freeze_mode,
                body_label=body_label,
                robot_w=robot_w,
                robot_h=robot_h,
                body_shape=body_shape,
            )
            max_steps = _native_max_steps(cfg)
        signature = (
            int(phase_cfg.train.max_steps),
            int(phase_cfg.terrain.grid_h),
            int(phase_cfg.terrain.grid_w),
            int(phase_cfg.terrain.get("center_plateau_width", phase_cfg.terrain.get("base_width", 0))),
            int(phase_cfg.terrain.get("base_width", 0)),
            int(phase_cfg.terrain.robot_w),
            int(phase_cfg.terrain.robot_h),
            str(phase_cfg.body.get("shape", "rectangle")),
        )
        group = grouped.setdefault(
            signature,
            {
                "phase_cfg": phase_cfg,
                "max_steps": int(max_steps),
                "entries": [],
                "budget_records": [],
            },
        )
        group["entries"].extend(
            _build_native_entries(
                phase_cfg,
                [{"goal_distance": goal_distance, "cliff_distance": cliff_distance}],
                trials_per_side=int(trials_per_side),
                goal_sides=goal_sides,
            )
        )
        group["budget_records"].append(
            {
                "goal_distance": int(goal_distance),
                "cliff_distance": int(cliff_distance),
                "distance_pair": _pair_label(goal_distance, cliff_distance),
                "max_steps": int(max_steps),
            }
        )
    return list(grouped.values())


def _build_double_cliff_entries(
    phase_cfg: DictConfig,
    widths: list[int],
    *,
    plateau_width: int,
    trials_per_width: int,
) -> list[TerrainEntry]:
    entries: list[TerrainEntry] = []
    for width in widths:
        for trial_repeat in range(int(trials_per_width)):
            entries.append(
                _build_double_cliff_entry(
                    phase_cfg,
                    width=int(width),
                    plateau_width=int(plateau_width),
                    extra_metadata={"trial_repeat_index": int(trial_repeat)},
                )
            )
    return entries


def _candidate_root_keys(seed: int | Sequence[int]) -> jax.Array:
    if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
        seeds = [int(value) for value in seed]
    else:
        seeds = [int(seed)]
    return jax.numpy.asarray([jax.random.PRNGKey(value) for value in seeds], dtype=jax.numpy.uint32)


def _split_candidate_results(result) -> list[Any]:
    leaves = jax.tree_util.tree_leaves(result)
    if not leaves:
        return []
    candidate_count = int(np.asarray(leaves[0]).shape[0])
    return [
        jax.tree_util.tree_map(lambda array, idx=index: np.asarray(array[idx]), result)
        for index in range(candidate_count)
    ]


def _run_trace_eval(
    *,
    phase_cfg: DictConfig,
    runtime_resolution,
    genotype: np.ndarray,
    entries: list[TerrainEntry],
    workload: str,
    seed: int,
) -> tuple[Any, Any, list[Any], Any]:
    phase_resolution, envs, alive_mask, static_capacity = _build_phase_assets(
        phase_cfg,
        runtime_resolution,
        entries,
    )
    trace_eval = trace_population_chunked(
        np.asarray(genotype, dtype=np.float32)[None, :],
        envs,
        alive_mask,
        _candidate_root_keys(seed),
        phase_cfg,
        workload=workload,
        runtime_resolution=phase_resolution,
        static_capacity=static_capacity,
    )
    episode_results = _flatten_selected_episode_results(trace_eval.episode_results)
    traces = _flatten_selected_episode_results(trace_eval.traces)
    fitnesses = np.asarray(episode_fitness_array(episode_results), dtype=np.float32)
    return episode_results, traces, envs, fitnesses


def _run_trace_eval_batched(
    *,
    phase_cfg: DictConfig,
    runtime_resolution,
    genotypes: np.ndarray,
    entries: list[TerrainEntry],
    workload: str,
    seeds: Sequence[int],
) -> tuple[list[Any], list[Any], list[Any], list[np.ndarray]]:
    phase_resolution, envs, alive_mask, static_capacity = _build_phase_assets(
        phase_cfg,
        runtime_resolution,
        entries,
    )
    trace_eval = trace_population_chunked(
        np.asarray(genotypes, dtype=np.float32),
        envs,
        alive_mask,
        _candidate_root_keys(seeds),
        phase_cfg,
        workload=workload,
        runtime_resolution=phase_resolution,
        static_capacity=static_capacity,
    )
    episode_results = jax.tree_util.tree_map(np.asarray, trace_eval.episode_results)
    traces = jax.tree_util.tree_map(np.asarray, trace_eval.traces)
    fitness_matrix = np.asarray(episode_fitness_array(episode_results), dtype=np.float32)
    return (
        _split_candidate_results(episode_results),
        _split_candidate_results(traces),
        envs,
        [np.asarray(fitness_matrix[index], dtype=np.float32) for index in range(int(fitness_matrix.shape[0]))],
    )


def _run_com_trace_eval(
    *,
    phase_cfg: DictConfig,
    runtime_resolution,
    genotype: np.ndarray,
    entries: list[TerrainEntry],
    seed: int,
) -> tuple[Any, Any, list[Any], np.ndarray]:
    phase_resolution, envs, alive_mask, static_capacity = _build_phase_assets(
        phase_cfg,
        runtime_resolution,
        entries,
    )
    env_batch = stack_eval_envs(envs, static_capacity=static_capacity)
    evaluator = make_com_trace_evaluator(phase_cfg, phase_resolution)
    episode_results, com_traces = evaluator(
        jax.numpy.asarray(genotype, dtype=jax.numpy.float32)[None, :],
        env_batch,
        alive_mask,
        _candidate_root_keys(seed),
    )
    jax.block_until_ready(episode_results.peak_front_x)
    jax.block_until_ready(com_traces.com_x)
    selected_episode_results = jax.tree_util.tree_map(lambda array: np.asarray(array[0]), episode_results)
    selected_com_traces = jax.tree_util.tree_map(lambda array: np.asarray(array[0]), com_traces)
    fitnesses = np.asarray(episode_fitness_array(selected_episode_results), dtype=np.float32)
    return selected_episode_results, selected_com_traces, envs, fitnesses


def _run_com_trace_eval_batched(
    *,
    phase_cfg: DictConfig,
    runtime_resolution,
    genotypes: np.ndarray,
    entries: list[TerrainEntry],
    seeds: Sequence[int],
) -> tuple[list[Any], list[Any], list[Any], list[np.ndarray]]:
    phase_resolution, envs, alive_mask, static_capacity = _build_phase_assets(
        phase_cfg,
        runtime_resolution,
        entries,
    )
    env_batch = stack_eval_envs(envs, static_capacity=static_capacity)
    evaluator = make_com_trace_evaluator(phase_cfg, phase_resolution)
    episode_results, com_traces = evaluator(
        jax.numpy.asarray(genotypes, dtype=jax.numpy.float32),
        env_batch,
        alive_mask,
        _candidate_root_keys(seeds),
    )
    jax.block_until_ready(episode_results.peak_front_x)
    jax.block_until_ready(com_traces.com_x)
    episode_results_np = jax.tree_util.tree_map(np.asarray, episode_results)
    com_traces_np = jax.tree_util.tree_map(np.asarray, com_traces)
    fitness_matrix = np.asarray(episode_fitness_array(episode_results_np), dtype=np.float32)
    return (
        _split_candidate_results(episode_results_np),
        _split_candidate_results(com_traces_np),
        envs,
        [np.asarray(fitness_matrix[index], dtype=np.float32) for index in range(int(fitness_matrix.shape[0]))],
    )


def _run_state_trace_eval(
    *,
    phase_cfg: DictConfig,
    runtime_resolution,
    genotype: np.ndarray,
    entries: list[TerrainEntry],
    seed: int,
) -> tuple[Any, Any, list[Any]]:
    phase_resolution, envs, alive_mask, static_capacity = _build_phase_assets(
        phase_cfg,
        runtime_resolution,
        entries,
    )
    env_batch = stack_eval_envs(envs, static_capacity=static_capacity)
    rollout_kwargs = rollout_kwargs_from_cfg(
        phase_cfg,
        collision_tuning=phase_resolution.resolved_collision,
    )
    params = ctrnn_flat_to_params(
        jax.numpy.asarray(genotype, dtype=jax.numpy.float32),
        ctrnn_hidden=int(phase_cfg.nca.ctrnn_hidden),
        ctrnn_grid_hidden=int(phase_cfg.nca.ctrnn_grid_hidden),
        gasnet_mode=str(phase_cfg.nca.get("gasnet_mode", "off")),
        ou_enabled=bool(resolved_noise_settings(phase_cfg)["ou_enabled"]),
        growth_enabled=False,
        active_comm_filters=active_comm_filters_from_cfg(phase_cfg),
    )

    @jax.jit
    def _evaluate(env_batch_arg, root_key):
        episode_keys = split_key_sequence(root_key, int(env_batch_arg.cliff_x.shape[0]))[1]
        return jax.vmap(
            lambda env, ep_key: run_episode_with_state_trace(
                env,
                params,
                alive_mask,
                ep_key,
                **rollout_kwargs,
            )
        )(env_batch_arg, episode_keys)

    episode_results, state_traces = _evaluate(env_batch, jax.random.PRNGKey(int(seed)))
    return (
        jax.tree_util.tree_map(lambda array: np.asarray(array), episode_results),
        jax.tree_util.tree_map(lambda array: np.asarray(array), state_traces),
        envs,
    )


def _native_like_condition_identity(
    *,
    assay_family: str,
    surface_id: str,
    signaling_condition: str,
    deformation_condition: str,
    sensing_freeze_mode: str = "none",
    body_label: str,
    condition_id_override: str | None = None,
    condition_label_override: str | None = None,
) -> tuple[str, str]:
    if condition_id_override is not None or condition_label_override is not None:
        if condition_id_override is None or condition_label_override is None:
            raise ValueError("condition_id_override and condition_label_override must be provided together")
        return str(condition_id_override), str(condition_label_override)
    if assay_family in {"native_core", "long_distance_generalization"}:
        return surface_id, _surface_label(surface_id)
    if assay_family == "body_transfer":
        return body_label, _body_label(body_label)
    if assay_family == "signaling_ablation":
        return signaling_condition, _signaling_label(signaling_condition)
    if assay_family == "frozen_sensing_control":
        return sensing_freeze_mode, _sensing_freeze_label(sensing_freeze_mode)
    return deformation_condition, _deformation_label(deformation_condition)


def _run_controller_trace_eval(
    *,
    phase_cfg: DictConfig,
    runtime_resolution,
    genotype: np.ndarray,
    entries: list[TerrainEntry],
    seed: int,
    episode_batch_size: int,
) -> tuple[Any, Any, list[Any]]:
    rollout_kwargs = rollout_kwargs_from_cfg(
        phase_cfg,
        collision_tuning=runtime_resolution.resolved_collision,
    )
    params = ctrnn_flat_to_params(
        jax.numpy.asarray(genotype, dtype=jax.numpy.float32),
        ctrnn_hidden=int(phase_cfg.nca.ctrnn_hidden),
        ctrnn_grid_hidden=int(phase_cfg.nca.ctrnn_grid_hidden),
        gasnet_mode=str(phase_cfg.nca.get("gasnet_mode", "off")),
        ou_enabled=bool(resolved_noise_settings(phase_cfg)["ou_enabled"]),
        growth_enabled=False,
        active_comm_filters=active_comm_filters_from_cfg(phase_cfg),
    )

    @jax.jit
    def _evaluate(env_batch_arg, alive_mask_arg, root_key):
        episode_keys = split_key_sequence(root_key, int(env_batch_arg.cliff_x.shape[0]))[1]
        return jax.vmap(
            lambda env, ep_key: run_episode_with_controller_trace(
                env,
                params,
                alive_mask_arg,
                ep_key,
                **rollout_kwargs,
            )
        )(env_batch_arg, episode_keys)

    chunk_size = max(int(episode_batch_size), 1)
    chunk_results: list[Any] = []
    chunk_traces: list[Any] = []
    chunk_envs: list[Any] = []
    chunk_count = (len(entries) + chunk_size - 1) // chunk_size
    chunk_keys = jax.random.split(jax.random.PRNGKey(int(seed)), chunk_count + 1)[1:]
    for chunk_index in range(chunk_count):
        start = chunk_index * chunk_size
        stop = min(start + chunk_size, len(entries))
        chunk_entries = entries[start:stop]
        phase_resolution, envs, alive_mask, static_capacity = _build_phase_assets(
            phase_cfg,
            runtime_resolution,
            chunk_entries,
        )
        env_batch = stack_eval_envs(envs, static_capacity=static_capacity)
        episode_results, controller_traces = _evaluate(
            env_batch,
            alive_mask,
            chunk_keys[chunk_index],
        )
        chunk_results.append(jax.tree_util.tree_map(lambda array: np.asarray(array), episode_results))
        chunk_traces.append(jax.tree_util.tree_map(lambda array: np.asarray(array), controller_traces))
        chunk_envs.extend(envs)
        _ = phase_resolution
    if not chunk_results:
        raise ValueError("controller-trace eval requires at least one terrain entry")
    return (
        jax.tree_util.tree_map(lambda *arrays: np.concatenate(arrays, axis=0), *chunk_results),
        jax.tree_util.tree_map(lambda *arrays: np.concatenate(arrays, axis=0), *chunk_traces),
        chunk_envs,
    )


def _run_state_space_trace_eval(
    *,
    phase_cfg: DictConfig,
    runtime_resolution,
    genotype: np.ndarray,
    entries: list[TerrainEntry],
    seed: int,
) -> tuple[Any, Any, list[Any]]:
    rollout_kwargs = rollout_kwargs_from_cfg(
        phase_cfg,
        collision_tuning=runtime_resolution.resolved_collision,
    )
    params = ctrnn_flat_to_params(
        jax.numpy.asarray(genotype, dtype=jax.numpy.float32),
        ctrnn_hidden=int(phase_cfg.nca.ctrnn_hidden),
        ctrnn_grid_hidden=int(phase_cfg.nca.ctrnn_grid_hidden),
        gasnet_mode=str(phase_cfg.nca.get("gasnet_mode", "off")),
        ou_enabled=bool(resolved_noise_settings(phase_cfg)["ou_enabled"]),
        growth_enabled=False,
        active_comm_filters=active_comm_filters_from_cfg(phase_cfg),
    )
    phase_resolution, envs, alive_mask, static_capacity = _build_phase_assets(
        phase_cfg,
        runtime_resolution,
        entries,
    )
    env_batch = stack_eval_envs(envs, static_capacity=static_capacity)

    @jax.jit
    def _evaluate(env_batch_arg, alive_mask_arg, root_key):
        episode_keys = split_key_sequence(root_key, int(env_batch_arg.cliff_x.shape[0]))[1]
        return jax.vmap(
            lambda env, ep_key: run_episode_with_state_space_trace(
                env,
                params,
                alive_mask_arg,
                ep_key,
                **rollout_kwargs,
            )
        )(env_batch_arg, episode_keys)

    episode_results, state_space_traces = _evaluate(
        env_batch,
        alive_mask,
        jax.random.PRNGKey(int(seed)),
    )
    _ = phase_resolution
    return (
        jax.tree_util.tree_map(lambda array: np.asarray(array), episode_results),
        jax.tree_util.tree_map(lambda array: np.asarray(array), state_space_traces),
        envs,
    )


def _cliff_frontier_world_x(cliff_edge_x: int, cliff_side: str) -> float:
    if str(cliff_side).strip().lower() == "left":
        return float(int(cliff_edge_x)) * float(CELL_SIZE)
    return float(int(cliff_edge_x) + 1) * float(CELL_SIZE)


def _cliff_facing_edge(min_x: np.ndarray, front_x: np.ndarray, cliff_side: str) -> np.ndarray:
    if str(cliff_side).strip().lower() == "left":
        return np.asarray(min_x, dtype=np.float32)
    return np.asarray(front_x, dtype=np.float32)


def _stopping_distance_for_trial(
    *,
    clearances: np.ndarray,
    cliff_fell: bool,
    first_cliff_vicinity_step: int | None,
) -> float | None:
    if cliff_fell or first_cliff_vicinity_step is None or clearances.size == 0:
        return None
    return float(max(float(clearances[-1]), 0.0))


def _max_cliff_protrusion_for_trial(
    *,
    min_edge_clearance: float | None,
    cliff_fell: bool,
    first_cliff_vicinity_step: int | None,
) -> float | None:
    if cliff_fell or first_cliff_vicinity_step is None or min_edge_clearance is None:
        return None
    return float(max(-float(min_edge_clearance), 0.0))


def _goal_facing_edge(back_x: np.ndarray, front_x: np.ndarray, goal_direction: int) -> np.ndarray:
    if int(goal_direction) < 0:
        return np.asarray(back_x, dtype=np.float32)
    return np.asarray(front_x, dtype=np.float32)


def _wrong_facing_edge(back_x: np.ndarray, front_x: np.ndarray, goal_direction: int) -> np.ndarray:
    if int(goal_direction) < 0:
        return np.asarray(front_x, dtype=np.float32)
    return np.asarray(back_x, dtype=np.float32)


def _goal_contact_face_x(goal_block_x: int, goal_direction: int) -> float:
    if int(goal_direction) < 0:
        return float(int(goal_block_x) + 1) * float(CELL_SIZE)
    return float(int(goal_block_x)) * float(CELL_SIZE)


def _controller_trace_trial_id(
    *,
    goal_distance: int,
    cliff_distance: int,
    side: str,
    trial_repeat_index: int,
) -> str:
    return (
        f"pair_{int(goal_distance)}_{int(cliff_distance)}"
        f"_{str(side).strip().lower()}_{int(trial_repeat_index):03d}"
    )


def _first_peak_summary(
    *,
    com_x: np.ndarray,
    init_com_x: float,
    goal_direction: int,
    turn_motion_threshold: float,
) -> tuple[str, float | None, float | None]:
    com_x = np.asarray(com_x, dtype=np.float32)
    if com_x.size == 0:
        return "none", None, None
    goalward_x = float(goal_direction) * (com_x - float(init_com_x))
    threshold = max(float(turn_motion_threshold), 0.0)
    significant_indices = np.flatnonzero(np.abs(goalward_x) >= threshold)
    if significant_indices.size == 0:
        return "none", None, None
    start_idx = int(significant_indices[0])
    initial_sign = 1 if float(goalward_x[start_idx]) > 0.0 else -1
    excursion = initial_sign * goalward_x[start_idx:]
    peak_offset = 0
    best_value = float(excursion[0])
    for offset, value in enumerate(excursion[1:], start=1):
        value_f = float(value)
        if value_f > best_value:
            best_value = value_f
            peak_offset = offset
        if best_value - value_f >= threshold:
            break
    peak_idx = start_idx + peak_offset
    return (
        "goal_first" if initial_sign > 0 else "cliff_first",
        float(com_x[peak_idx]),
        float(goalward_x[peak_idx]),
    )


def _first_frontier_index(
    *,
    cliff_edge_x: np.ndarray,
    frontier_x: float,
    goal_direction: int,
) -> int | None:
    cliff_edge_x = np.asarray(cliff_edge_x, dtype=np.float32)
    if cliff_edge_x.size == 0:
        return None
    reached = (float(goal_direction) * (cliff_edge_x - float(frontier_x))) <= 1e-6
    indices = np.flatnonzero(reached)
    if indices.size == 0:
        return None
    return int(indices[0])


def _task_code_geom(
    *,
    back_x: np.ndarray,
    front_x: np.ndarray,
    goal_direction: int,
    cliff_side: str,
    cliff_edge_x: int,
    goal_block_x: int,
    init_back_x: float,
    init_front_x: float,
) -> tuple[np.ndarray, bool, int | None]:
    back_x = np.asarray(back_x, dtype=np.float32)
    front_x = np.asarray(front_x, dtype=np.float32)
    if back_x.size == 0 or front_x.size == 0:
        return np.zeros((0,), dtype=np.float32), False, None
    goal_edge_x = _goal_facing_edge(back_x, front_x, goal_direction)
    cliff_facing_edge_x = _wrong_facing_edge(back_x, front_x, goal_direction)
    init_cliff_edge_x = float(
        _wrong_facing_edge(
            np.asarray([init_back_x], dtype=np.float32),
            np.asarray([init_front_x], dtype=np.float32),
            goal_direction,
        )[0]
    )
    frontier_x = _cliff_frontier_world_x(cliff_edge_x, cliff_side)
    cliff_route = max(abs(float(frontier_x) - init_cliff_edge_x), 1e-6)
    outward_progress = np.maximum(
        -float(goal_direction) * (cliff_facing_edge_x - init_cliff_edge_x),
        0.0,
    )
    task_code = np.clip(outward_progress / cliff_route, 0.0, 1.0).astype(np.float32)
    frontier_idx = _first_frontier_index(
        cliff_edge_x=cliff_facing_edge_x,
        frontier_x=frontier_x,
        goal_direction=goal_direction,
    )
    if frontier_idx is None:
        return task_code, False, None
    frontier_goal_edge_x = float(goal_edge_x[frontier_idx])
    goal_contact_x = _goal_contact_face_x(goal_block_x, goal_direction)
    goal_route = max(abs(goal_contact_x - frontier_goal_edge_x), 1e-6)
    return_progress = np.maximum(
        float(goal_direction) * (goal_edge_x - frontier_goal_edge_x),
        0.0,
    )
    task_code[frontier_idx:] = (
        1.0 + np.clip(return_progress[frontier_idx:] / goal_route, 0.0, 1.0)
    ).astype(np.float32)
    return task_code, True, frontier_idx


def _termination_reason(
    *,
    goal_touched: bool,
    touched_death_plane: bool,
    went_airborne: bool,
    steps_alive: int,
    max_steps: int,
    sense_act_phases: int,
) -> str:
    if goal_touched:
        return "goal_touched"
    if touched_death_plane:
        return "death_plane"
    if went_airborne:
        return "airborne"
    if int(steps_alive) >= int(max_steps) * int(sense_act_phases):
        return "max_steps"
    return "ended"


def _episode_termination_reason(
    *,
    goal_touched: bool,
    touched_death_plane: bool,
    went_airborne: bool,
    left_exit: bool,
    steps_alive: int,
    max_steps: int,
) -> str:
    if goal_touched:
        return "goal_touched"
    if touched_death_plane:
        return "death_plane"
    if went_airborne:
        return "airborne"
    if left_exit:
        return "left_exit"
    if int(steps_alive) >= int(max_steps):
        return "max_steps"
    return "ended"


def _trace_kinematics(positions: np.ndarray, robot_point_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.asarray(robot_point_mask, dtype=bool)
    point_x = np.asarray(positions, dtype=np.float32)[:, mask, 0]
    return (
        np.mean(point_x, axis=1),
        np.min(point_x, axis=1),
        np.max(point_x, axis=1),
    )


def _significant_direction_label(direction_code: int) -> str:
    if int(direction_code) < 0:
        return "left"
    if int(direction_code) > 0:
        return "right"
    return "none"


def _extract_initial_commit(
    *,
    com_x: np.ndarray,
    init_com_x: float,
    turn_motion_threshold: float,
) -> tuple[str, int | None, float | None]:
    if com_x.size == 0:
        return "none", None, None
    deltas = np.diff(np.concatenate([[float(init_com_x)], np.asarray(com_x, dtype=np.float32)]))
    for index, delta in enumerate(deltas, start=1):
        if abs(float(delta)) >= float(turn_motion_threshold):
            return (
                "right" if float(delta) > 0.0 else "left",
                int(index),
                float(com_x[index - 1]),
            )
    return "none", None, None


def _find_arrest_step(
    *,
    com_x: np.ndarray,
    init_com_x: float,
    turn_motion_threshold: float,
    first_cliff_vicinity_step: int | None,
) -> int | None:
    if first_cliff_vicinity_step is None or com_x.size <= 0:
        return None
    deltas = np.abs(
        np.diff(np.concatenate([[float(init_com_x)], np.asarray(com_x, dtype=np.float32)]))
    )
    if deltas.size < _ARREST_WINDOW_STEPS:
        return None
    start_index = max(int(first_cliff_vicinity_step) - 1, 0)
    for index in range(start_index, deltas.size - _ARREST_WINDOW_STEPS + 1):
        window = deltas[index:index + _ARREST_WINDOW_STEPS]
        if np.all(window < float(turn_motion_threshold)):
            return int(index + 1)
    return None


def _summary_stats(values: np.ndarray, *, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "bootstrap_ci95": None,
        }
    if values.size == 1:
        ci = [float(values[0]), float(values[0])]
    else:
        rng = np.random.default_rng(int(seed))
        draws = rng.choice(values, size=(512, values.size), replace=True)
        boot = np.mean(draws, axis=1)
        ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=0)),
        "bootstrap_ci95": ci,
    }


def _relative_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value) - float(baseline)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(_jsonify(payload), indent=2, default=_json_default)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(rendered)
            handle.write("\n")
            temp_path = Path(handle.name)
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _json_default(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _jsonify(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return _jsonify(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_dataframe_bundle(df: pd.DataFrame, output_dir: Path, *, stem: str = "per_episode") -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    parquet_path = output_dir / f"{stem}.parquet"
    df.to_csv(csv_path, index=False)
    parquet_written = False
    parquet_error = None
    try:
        df.to_parquet(parquet_path, index=False)
        parquet_written = True
    except Exception as exc:  # pragma: no cover - environment dependent
        parquet_error = str(exc)
    return {
        "csv": str(csv_path),
        "parquet": str(parquet_path) if parquet_written else None,
        "parquet_error": parquet_error,
    }


def _render_config_for_post_run(cfg: DictConfig):
    return _render_config_for_stage(cfg, "post_train")


def _select_best_render_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_group: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        group_key = str(candidate["group_key"])
        current = best_by_group.get(group_key)
        if current is None or tuple(candidate["score"]) > tuple(current["score"]):
            best_by_group[group_key] = candidate
    return sorted(best_by_group.values(), key=lambda candidate: tuple(candidate["sort_key"]))


def _render_selected_candidates(
    *,
    cfg: DictConfig,
    output_dir: Path,
    candidates: list[dict[str, Any]],
    manifest_kind: str,
) -> list[dict[str, Any]]:
    selected_candidates = _select_best_render_candidates(candidates)
    if not selected_candidates:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    render_cfg = _render_config_for_post_run(cfg)
    worker_count = _render_parallel_workers(cfg, len(selected_candidates))

    def _encode(candidate: dict[str, Any]) -> dict[str, Any]:
        output_path = output_dir / str(candidate["filename"])
        render_episode_gif(
            candidate["trace"],
            candidate["render_info"],
            output_path,
            config=render_cfg,
        )
        return {
            "group_key": str(candidate["group_key"]),
            "filename": str(candidate["filename"]),
            "path": str(output_path),
            "trial_id": str(candidate["trial_id"]),
            "selection_reason": str(candidate["selection_reason"]),
            "score": list(candidate["score"]),
            "fitness": float(candidate["fitness"]),
            "success": bool(candidate.get("success", False)),
            "fell": bool(candidate.get("fell", False)),
            "metadata": dict(candidate["metadata"]),
        }

    if worker_count == 1:
        manifest_items = [_encode(candidate) for candidate in selected_candidates]
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="post-run-render") as executor:
            manifest_items = list(executor.map(_encode, selected_candidates))
    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": _POST_RUN_SCHEMA_VERSION,
            "kind": str(manifest_kind),
            "render_count": int(len(manifest_items)),
            "renders": manifest_items,
        },
    )
    return manifest_items


def _native_render_selection_candidates(
    *,
    episode_results,
    entries: list[TerrainEntry],
    fitnesses: np.ndarray,
    trial_id_prefix: str,
    metadata_overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    episode_count = int(np.asarray(episode_results.steps_alive).shape[0])
    for index in range(episode_count):
        result = jax.tree_util.tree_map(lambda array: np.asarray(array[index]), episode_results)
        metadata = dict(entries[index].metadata or {})
        if metadata_overrides:
            metadata.update(dict(metadata_overrides))
        goal_distance = int(metadata.get("goal_distance", result.goal_distance))
        cliff_distance = int(metadata.get("cliff_distance", result.cliff_distance))
        side = str(metadata.get("goal_side", metadata.get("assay_goal_side", "unknown")))
        trial_repeat_index = int(metadata.get("trial_repeat_index", 0))
        success = bool(result.goal_touched) and not bool(result.cliff_fell)
        fell = bool(result.cliff_fell)
        candidates.append(
            {
                "episode_index": int(index),
                "group_key": f"pair_{goal_distance}_{cliff_distance}_{side}",
                "sort_key": (int(goal_distance), int(cliff_distance), 0 if side == "left" else 1),
                "filename": f"pair_{goal_distance}_{cliff_distance}_{side}.gif",
                "trial_id": f"{trial_id_prefix}_{index:04d}",
                "selection_reason": "success_no_fall_then_fitness_then_repeat",
                "score": (
                    int(success),
                    int(not fell),
                    float(fitnesses[index]),
                    -int(trial_repeat_index),
                    -int(index),
                ),
                "fitness": float(fitnesses[index]),
                "success": success,
                "fell": fell,
                "metadata": metadata,
            }
        )
    return candidates


def _native_render_candidates(
    *,
    episode_results,
    traces,
    envs: list[Any],
    entries: list[TerrainEntry],
    fitnesses: np.ndarray,
    trial_id_prefix: str,
    metadata_overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    episode_count = int(np.asarray(episode_results.steps_alive).shape[0])
    for index in range(episode_count):
        env_data = envs[index]
        render_info = getattr(env_data, "render_info", None)
        if render_info is None:
            continue
        result = jax.tree_util.tree_map(lambda array: np.asarray(array[index]), episode_results)
        trace = jax.tree_util.tree_map(lambda array: np.asarray(array[index]), traces)
        metadata = dict(entries[index].metadata or {})
        if metadata_overrides:
            metadata.update(dict(metadata_overrides))
        goal_distance = int(metadata.get("goal_distance", result.goal_distance))
        cliff_distance = int(metadata.get("cliff_distance", result.cliff_distance))
        side = str(metadata.get("goal_side", metadata.get("assay_goal_side", "unknown")))
        trial_repeat_index = int(metadata.get("trial_repeat_index", 0))
        success = bool(result.goal_touched) and not bool(result.cliff_fell)
        fell = bool(result.cliff_fell)
        candidates.append(
            {
                "group_key": f"pair_{goal_distance}_{cliff_distance}_{side}",
                "sort_key": (int(goal_distance), int(cliff_distance), 0 if side == "left" else 1),
                "filename": f"pair_{goal_distance}_{cliff_distance}_{side}.gif",
                "trial_id": f"{trial_id_prefix}_{index:04d}",
                "selection_reason": "success_no_fall_then_fitness_then_repeat",
                "score": (
                    int(success),
                    int(not fell),
                    float(fitnesses[index]),
                    -int(trial_repeat_index),
                    -int(index),
                ),
                "fitness": float(fitnesses[index]),
                "success": success,
                "fell": fell,
                "trace": trace,
                "render_info": render_info,
                "metadata": metadata,
            }
    )
    return candidates


def _double_cliff_render_selection_candidates(
    *,
    episode_results,
    entries: list[TerrainEntry],
    fitnesses: np.ndarray,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    episode_count = int(np.asarray(episode_results.steps_alive).shape[0])
    for index in range(episode_count):
        result = jax.tree_util.tree_map(lambda array: np.asarray(array[index]), episode_results)
        metadata = dict(entries[index].metadata or {})
        width = int(
            metadata.get(
                "double_cliff_width",
                metadata.get("width", metadata.get("assay_width", 0)),
            )
        )
        trial_repeat_index = int(metadata.get("trial_repeat_index", 0))
        candidates.append(
            {
                "episode_index": int(index),
                "group_key": f"width_{width}",
                "sort_key": (int(width),),
                "filename": f"width_{width}.gif",
                "trial_id": f"double_cliff_{index:04d}",
                "selection_reason": "survival_then_fitness_then_repeat",
                "score": (
                    int(not bool(result.cliff_fell)),
                    float(fitnesses[index]),
                    -int(trial_repeat_index),
                    -int(index),
                ),
                "fitness": float(fitnesses[index]),
                "success": not bool(result.cliff_fell),
                "fell": bool(result.cliff_fell),
                "metadata": metadata,
            }
        )
    return candidates


def _double_cliff_render_candidates(
    *,
    episode_results,
    traces,
    envs: list[Any],
    entries: list[TerrainEntry],
    fitnesses: np.ndarray,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    episode_count = int(np.asarray(episode_results.steps_alive).shape[0])
    for index in range(episode_count):
        env_data = envs[index]
        render_info = getattr(env_data, "render_info", None)
        if render_info is None:
            continue
        result = jax.tree_util.tree_map(lambda array: np.asarray(array[index]), episode_results)
        trace = jax.tree_util.tree_map(lambda array: np.asarray(array[index]), traces)
        metadata = dict(entries[index].metadata or {})
        width = int(
            metadata.get(
                "double_cliff_width",
                metadata.get("width", metadata.get("assay_width", 0)),
            )
        )
        trial_repeat_index = int(metadata.get("trial_repeat_index", 0))
        candidates.append(
            {
                "group_key": f"width_{width}",
                "sort_key": (int(width),),
                "filename": f"width_{width}.gif",
                "trial_id": f"double_cliff_{index:04d}",
                "selection_reason": "survival_then_fitness_then_repeat",
                "score": (
                    int(not bool(result.cliff_fell)),
                    float(fitnesses[index]),
                    -int(trial_repeat_index),
                    -int(index),
                ),
                "fitness": float(fitnesses[index]),
                "success": not bool(result.cliff_fell),
                "fell": bool(result.cliff_fell),
                "trace": trace,
                "render_info": render_info,
                "metadata": metadata,
            }
    )
    return candidates


def _render_native_candidates_from_summary(
    *,
    phase_cfg: DictConfig,
    runtime_resolution,
    genotype: np.ndarray,
    entries: list[TerrainEntry],
    episode_results,
    fitnesses: np.ndarray,
    seed: int,
    trial_id_prefix: str,
    metadata_overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    selection_candidates = _select_best_render_candidates(
        _native_render_selection_candidates(
            episode_results=episode_results,
            entries=entries,
            fitnesses=fitnesses,
            trial_id_prefix=trial_id_prefix,
            metadata_overrides=metadata_overrides,
        )
    )
    if not selection_candidates:
        return []
    selected_entries = [entries[int(candidate["episode_index"])] for candidate in selection_candidates]
    _, traces, selected_envs, _ = _run_trace_eval(
        phase_cfg=phase_cfg,
        runtime_resolution=runtime_resolution,
        genotype=genotype,
        entries=selected_entries,
        workload="trace",
        seed=int(seed),
    )
    render_candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(selection_candidates):
        render_info = getattr(selected_envs[index], "render_info", None)
        if render_info is None:
            continue
        render_candidates.append(
            {
                **{key: value for key, value in candidate.items() if key != "episode_index"},
                "trace": jax.tree_util.tree_map(lambda array: np.asarray(array[index]), traces),
                "render_info": render_info,
            }
        )
    return render_candidates


def _render_double_cliff_candidates_from_summary(
    *,
    cfg: DictConfig,
    output_dir: Path,
    phase_cfg: DictConfig,
    runtime_resolution,
    genotype: np.ndarray,
    entries: list[TerrainEntry],
    episode_results,
    fitnesses: np.ndarray,
    seed: int,
) -> list[dict[str, Any]]:
    selection_candidates = _select_best_render_candidates(
        _double_cliff_render_selection_candidates(
            episode_results=episode_results,
            entries=entries,
            fitnesses=fitnesses,
        )
    )
    if not selection_candidates:
        return []
    selected_entries = [entries[int(candidate["episode_index"])] for candidate in selection_candidates]
    _, traces, selected_envs, _ = _run_trace_eval(
        phase_cfg=phase_cfg,
        runtime_resolution=runtime_resolution,
        genotype=genotype,
        entries=selected_entries,
        workload="trace",
        seed=int(seed),
    )
    render_candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(selection_candidates):
        render_info = getattr(selected_envs[index], "render_info", None)
        if render_info is None:
            continue
        render_candidates.append(
            {
                **{key: value for key, value in candidate.items() if key != "episode_index"},
                "trace": jax.tree_util.tree_map(lambda array: np.asarray(array[index]), traces),
                "render_info": render_info,
            }
        )
    return _render_selected_candidates(
        cfg=cfg,
        output_dir=output_dir,
        candidates=render_candidates,
        manifest_kind="sensing_edges_post_run_render_manifest",
    )


def _copy_render_bundle(source_dir: Path, target_dir: Path) -> None:
    if not source_dir.exists():
        return
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)


def _save_figure_variants(fig, output_path: Path, *, dpi: int = 180) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")


def _style_publication_axis(ax, *, ylabel: str | None = None, xlabel: str | None = None) -> None:
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=10)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=10)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelrotation=20)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")


def _outcome_breakdown(
    df: pd.DataFrame,
    *,
    success_col: str,
    fall_col: str,
) -> dict[str, Any]:
    if df.empty:
        trial_count = 0
        success_count = 0
        fall_count = 0
    else:
        success_mask = df[success_col].astype(bool).to_numpy()
        fall_mask = df[fall_col].astype(bool).to_numpy()
        trial_count = int(len(df))
        success_count = int(np.sum(success_mask))
        fall_count = int(np.sum(fall_mask))
    no_hit_no_fall_count = max(int(trial_count) - success_count - fall_count, 0)
    denominator = max(int(trial_count), 1)
    return {
        "trial_count": int(trial_count),
        "success_count": int(success_count),
        "success_rate": float(success_count / denominator) if trial_count else 0.0,
        "fall_count": int(fall_count),
        "fall_rate": float(fall_count / denominator) if trial_count else 0.0,
        "no_hit_no_fall_count": int(no_hit_no_fall_count),
        "no_hit_no_fall_rate": float(no_hit_no_fall_count / denominator) if trial_count else 0.0,
    }


def _flatten_record(record: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for raw_key, raw_value in record.items():
        key = str(raw_key)
        value = _jsonify(raw_value)
        flat_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_record(value, prefix=flat_key))
            continue
        if isinstance(value, list):
            if key == "bootstrap_ci95" and len(value) == 2:
                flat[f"{prefix}_bootstrap_ci95_lower" if prefix else "bootstrap_ci95_lower"] = value[0]
                flat[f"{prefix}_bootstrap_ci95_upper" if prefix else "bootstrap_ci95_upper"] = value[1]
            else:
                flat[flat_key] = json.dumps(value, sort_keys=True)
            continue
        flat[flat_key] = value
    return flat


def _records_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame([_flatten_record(record) for record in records])


def _write_stats_tables(
    *,
    assay_dir: Path,
    condition_records: list[dict[str, Any]] | None = None,
    specimen_records: list[dict[str, Any]] | None = None,
    inventory_records: list[dict[str, Any]] | None = None,
    cross_surface_records: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    stats_dir = assay_dir / "stats"
    outputs: dict[str, dict[str, Any]] = {}
    table_specs = {
        "condition_summary": condition_records,
        "specimen_summary": specimen_records,
        "assay_inventory": inventory_records,
        "cross_surface_summary": cross_surface_records,
    }
    for stem, records in table_specs.items():
        if records is None:
            continue
        df = _records_dataframe(records)
        outputs[stem] = _write_dataframe_bundle(df, stats_dir, stem=stem)
    return outputs


def _chart_bar(
    *,
    categories: list[str],
    values: list[float],
    ci_bounds: list[list[float] | None] | None,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    positions = np.arange(len(categories), dtype=np.float32)
    clean_categories = [str(category) for category in categories]
    if ci_bounds is None:
        ax.bar(positions, values, color="#2E6F95", edgecolor="#173753", linewidth=0.8)
    else:
        lower = [
            0.0 if bounds is None else max(values[index] - bounds[0], 0.0)
            for index, bounds in enumerate(ci_bounds)
        ]
        upper = [
            0.0 if bounds is None else max(bounds[1] - values[index], 0.0)
            for index, bounds in enumerate(ci_bounds)
        ]
        ax.bar(
            positions,
            values,
            yerr=np.asarray([lower, upper], dtype=np.float32),
            color="#2E6F95",
            edgecolor="#173753",
            linewidth=0.8,
            capsize=4,
        )
    ax.set_xticks(positions, clean_categories)
    ax.set_title(title, fontsize=12)
    _style_publication_axis(ax, ylabel=ylabel)
    fig.tight_layout()
    _save_figure_variants(fig, output_path)
    plt.close(fig)


def _chart_line(
    *,
    categories: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(categories, values, marker="o", color="#1A5A99", linewidth=2.0, markersize=6)
    ax.set_title(title, fontsize=12)
    _style_publication_axis(ax, ylabel=ylabel)
    fig.tight_layout()
    _save_figure_variants(fig, output_path)
    plt.close(fig)


def _chart_histogram(
    *,
    values: np.ndarray,
    title: str,
    xlabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(
        np.asarray(values, dtype=np.float32),
        bins=min(10, max(len(values), 1)),
        color="#C97C5D",
        edgecolor="#8C4A2F",
        linewidth=0.8,
    )
    ax.set_title(title, fontsize=12)
    _style_publication_axis(ax, xlabel=xlabel)
    fig.tight_layout()
    _save_figure_variants(fig, output_path)
    plt.close(fig)


def _chart_outcome_breakdown(
    *,
    categories: list[str],
    outcome_breakdowns: list[dict[str, Any]],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    positions = np.arange(len(categories), dtype=np.float32)
    success = np.asarray(
        [float(record.get("success_rate", 0.0) or 0.0) for record in outcome_breakdowns],
        dtype=np.float32,
    )
    neutral = np.asarray(
        [float(record.get("no_hit_no_fall_rate", 0.0) or 0.0) for record in outcome_breakdowns],
        dtype=np.float32,
    )
    fall = np.asarray(
        [float(record.get("fall_rate", 0.0) or 0.0) for record in outcome_breakdowns],
        dtype=np.float32,
    )
    ax.bar(positions, success, color="#2E8B57", label="Success")
    ax.bar(positions, neutral, bottom=success, color="#A0A0A0", label="No Hit / No Fall")
    ax.bar(positions, fall, bottom=success + neutral, color="#B03052", label="Fall")
    for index, record in enumerate(outcome_breakdowns):
        ax.text(
            positions[index],
            1.02,
            f"n={int(record.get('trial_count', 0))}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(positions, [str(category) for category in categories])
    ax.set_ylim(0.0, 1.1)
    ax.set_title(title, fontsize=12)
    _style_publication_axis(ax, ylabel="fraction")
    ax.legend(frameon=False, fontsize=8, ncols=3, loc="upper center")
    fig.tight_layout()
    _save_figure_variants(fig, output_path)
    plt.close(fig)


def _finalize_assay_outputs(
    *,
    assay_dir: Path,
    manifest: dict[str, Any],
    rows: pd.DataFrame | None,
    aggregate: dict[str, Any],
    condition_records: list[dict[str, Any]] | None = None,
    specimen_records: list[dict[str, Any]] | None = None,
    inventory_records: list[dict[str, Any]] | None = None,
    cross_surface_records: list[dict[str, Any]] | None = None,
) -> None:
    assay_dir.mkdir(parents=True, exist_ok=True)
    _write_json(assay_dir / "manifest.json", manifest)
    if rows is not None:
        bundle = _write_dataframe_bundle(rows, assay_dir, stem="per_episode")
        aggregate["paths"] = bundle
    stats_outputs = _write_stats_tables(
        assay_dir=assay_dir,
        condition_records=condition_records,
        specimen_records=specimen_records,
        inventory_records=inventory_records,
        cross_surface_records=cross_surface_records,
    )
    if stats_outputs:
        aggregate["stats_paths"] = stats_outputs
    _write_json(assay_dir / "summary.json", aggregate)
    _write_json(assay_dir / "aggregate.json", aggregate)


def _condition_summary_records(
    df: pd.DataFrame,
    *,
    group_col: str,
    metrics: dict[str, str],
    seed_prefix: int,
    success_col: str | None = None,
    fall_col: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for group_index, (group_value, group_df) in enumerate(df.groupby(group_col, sort=False)):
        record = {group_col: group_value, "trial_count": int(len(group_df))}
        for metric_name, column in metrics.items():
            summary = _summary_stats(group_df[column].astype(float).to_numpy(), seed=seed_prefix + group_index)
            record[metric_name] = summary
        if success_col is not None and fall_col is not None:
            record["outcome_breakdown"] = _outcome_breakdown(
                group_df,
                success_col=success_col,
                fall_col=fall_col,
            )
        records.append(record)
    return records


def _write_group_summaries(
    *,
    root_dir: Path,
    records: list[dict[str, Any]],
    group_field: str,
) -> None:
    for record in records:
        group_value = str(record[group_field])
        summary_dir = root_dir / group_value
        summary_dir.mkdir(parents=True, exist_ok=True)
        _write_json(summary_dir / "summary.json", record)
        _write_json(summary_dir / "aggregate.json", record)


def _controller_trace_episode_batch_size(
    *,
    runtime_resolution,
    requested_batch_size: int,
    total_entries: int,
) -> int:
    if str(runtime_resolution.gpu_profile).strip().lower() in _FULL_BATCH_CONTROLLER_TRACE_GPU_PROFILES:
        return max(int(total_entries), 1)
    return max(int(requested_batch_size), 1)


def _native_metrics_schema() -> dict[str, str]:
    return {
        "success_rate": "success",
        "fall_rate": "fell",
        "mean_fitness": "fitness",
        "mean_max_cliff_protrusion": "max_cliff_protrusion",
        "mean_time_to_arrest": "time_to_arrest",
        "mean_cliff_vicinity_fraction": "cliff_vicinity_fraction",
        "mean_activation": "mean_activation",
        "mean_speed": "mean_speed",
        "mean_turn_count": "turn_count",
    }


def _native_surface_specimen_record(
    *,
    specimen_id: str,
    assay_family: str,
    surface_id: str,
    rows: pd.DataFrame,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "specimen_id": specimen_id,
        "assay_family": assay_family,
        "surface_id": surface_id,
        "surface_label": _surface_label(surface_id),
        "trial_count": int(len(rows)),
        "outcome_breakdown": _outcome_breakdown(rows, success_col="success", fall_col="fell"),
    }
    for metric_name, column in _native_metrics_schema().items():
        record[metric_name] = _summary_stats(
            rows[column].astype(float).to_numpy(),
            seed=_stable_seed(specimen_id, assay_family, surface_id, metric_name),
        )
    if extra_fields:
        record.update(dict(extra_fields))
    return record


def _double_cliff_surface_specimen_record(
    *,
    specimen_id: str,
    assay_family: str,
    surface_id: str,
    rows: pd.DataFrame,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = rows.copy()
    if "fell" not in rows.columns:
        fell_left = rows["fell_left"].fillna(False).astype(bool) if "fell_left" in rows.columns else pd.Series(False, index=rows.index)
        fell_right = rows["fell_right"].fillna(False).astype(bool) if "fell_right" in rows.columns else pd.Series(False, index=rows.index)
        rows["fell"] = fell_left | fell_right
    if "survived" not in rows.columns:
        rows["survived"] = ~rows["fell"].astype(bool)
    if "first_approached_side" in rows.columns:
        valid_side_rows = rows[rows["first_approached_side"].notna()]
        left_bias = (
            float(np.mean((valid_side_rows["first_approached_side"] == "left").astype(float)))
            if not valid_side_rows.empty
            else None
        )
        right_bias = (
            float(np.mean((valid_side_rows["first_approached_side"] == "right").astype(float)))
            if not valid_side_rows.empty
            else None
        )
    else:
        left_bias = None
        right_bias = None
    side_switch_values = (
        rows["side_switch_count"].astype(float).to_numpy()
        if "side_switch_count" in rows.columns
        else np.asarray([], dtype=np.float32)
    )
    record = {
        "specimen_id": specimen_id,
        "assay_family": assay_family,
        "surface_id": surface_id,
        "surface_label": _surface_label(surface_id),
        "trial_count": int(len(rows)),
        "outcome_breakdown": _outcome_breakdown(rows, success_col="survived", fall_col="fell"),
        "survival_rate": _summary_stats(rows["survived"].astype(float).to_numpy(), seed=_stable_seed(specimen_id, assay_family, surface_id, "survival_rate")),
        "mean_side_switch_count": _summary_stats(side_switch_values, seed=_stable_seed(specimen_id, assay_family, surface_id, "mean_side_switch_count")),
        "left_first_side_bias": left_bias,
        "right_first_side_bias": right_bias,
        "symmetricity_score": None if left_bias is None or right_bias is None else 1.0 - abs(left_bias - right_bias),
    }
    if extra_fields:
        record.update(dict(extra_fields))
    return record


def _surface_inventory_record(
    *,
    assay_family: str,
    surface_id: str,
    subgroup_type: str,
    subgroup_id: str,
    surface_dir: Path,
    subgroup_dir: Path | None = None,
) -> dict[str, Any]:
    target_dir = subgroup_dir if subgroup_dir is not None else surface_dir
    return {
        "assay_family": assay_family,
        "surface_id": surface_id,
        "surface_label": _surface_label(surface_id),
        "subgroup_type": subgroup_type,
        "subgroup_id": subgroup_id,
        "surface_dir": str(surface_dir),
        "target_dir": str(target_dir),
        "summary_path": str(target_dir / "summary.json"),
        "aggregate_path": str(target_dir / "aggregate.json"),
        "per_episode_csv": str(surface_dir / "per_episode.csv"),
        "per_episode_parquet": str(surface_dir / "per_episode.parquet"),
        "render_manifest_path": str(target_dir / "renders" / "manifest.json"),
        "charts_dir": str(surface_dir / "charts"),
    }


def _run_native_like_surface(
    *,
    cfg: DictConfig,
    runtime_resolution,
    genotype: np.ndarray,
    run_id: str,
    specimen_id: str,
    assay_family: str,
    surface_id: str,
    pairs: list[dict[str, int]],
    trials_per_side: int,
    signaling_condition: str = "intact",
    deformation_condition: str = "intact",
    sensing_freeze_mode: str = "none",
    body_label: str = "rect_7x2",
    robot_w: int | None = None,
    robot_h: int | None = None,
    body_shape: str | None = None,
    render_output_dir: Path | None = None,
    trial_id_prefix: str | None = None,
    render_metadata: dict[str, Any] | None = None,
    condition_id_override: str | None = None,
    condition_label_override: str | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    render_candidates: list[dict[str, Any]] = []
    budget_records: list[dict[str, Any]] = []
    capture_mode = (
        "com_trace"
        if _fast_ablation_eval_enabled(cfg)
        and assay_family in {"signaling_ablation", "deformation_ablation", "frozen_sensing_control", "body_transfer", "combined_ablation_quick"}
        else "trace"
    )
    condition_id, condition_label = _native_like_condition_identity(
        assay_family=assay_family,
        surface_id=surface_id,
        signaling_condition=signaling_condition,
        deformation_condition=deformation_condition,
        sensing_freeze_mode=sensing_freeze_mode,
        body_label=body_label,
        condition_id_override=condition_id_override,
        condition_label_override=condition_label_override,
    )
    pair_groups = _native_like_pair_groups(
        cfg=cfg,
        surface_id=surface_id,
        pairs=pairs,
        trials_per_side=int(trials_per_side),
        signaling_condition=signaling_condition,
        deformation_condition=deformation_condition,
        sensing_freeze_mode=sensing_freeze_mode,
        body_label=body_label,
        robot_w=robot_w,
        robot_h=robot_h,
        body_shape=body_shape,
    )
    for group_index, pair_group in enumerate(pair_groups):
        phase_cfg = pair_group["phase_cfg"]
        max_steps = int(pair_group["max_steps"])
        entries = list(pair_group["entries"])
        evaluation_seed = _stable_seed(
            run_id,
            specimen_id,
            assay_family,
            surface_id,
            signaling_condition,
            deformation_condition,
            sensing_freeze_mode,
            body_label,
            "group",
            group_index,
        )
        if capture_mode == "com_trace":
            episode_results, com_traces, _envs, fitnesses = _run_com_trace_eval(
                phase_cfg=phase_cfg,
                runtime_resolution=runtime_resolution,
                genotype=genotype,
                entries=entries,
                seed=evaluation_seed,
            )
            pair_rows = _native_com_trace_row_records(
                episode_results=episode_results,
                com_traces=com_traces,
                fitnesses=fitnesses,
                entries=entries,
                specimen_id=specimen_id,
                run_id=run_id,
                assay_family=assay_family,
                condition_id=condition_id,
                condition_label=condition_label,
                signaling_condition=signaling_condition,
                deformation_condition=deformation_condition,
                sensing_freeze_mode=sensing_freeze_mode,
                body_label=body_label,
                max_steps=int(max_steps),
            )
            if render_output_dir is not None:
                render_candidates.extend(
                    _render_native_candidates_from_summary(
                    phase_cfg=phase_cfg,
                    runtime_resolution=runtime_resolution,
                    genotype=genotype,
                    entries=entries,
                    episode_results=episode_results,
                    fitnesses=fitnesses,
                    seed=evaluation_seed,
                    trial_id_prefix=str(trial_id_prefix or f"{assay_family}_{surface_id}"),
                    metadata_overrides={
                        "surface_id": surface_id,
                        "surface_label": _surface_label(surface_id),
                        "surface_max_steps": int(max_steps),
                        "sensing_freeze_mode": sensing_freeze_mode,
                        **(dict(render_metadata) if render_metadata else {}),
                    },
                    )
                )
        else:
            episode_results, traces, envs, fitnesses = _run_trace_eval(
                phase_cfg=phase_cfg,
                runtime_resolution=runtime_resolution,
                genotype=genotype,
                entries=entries,
                workload="trace",
                seed=evaluation_seed,
            )
            pair_rows = _native_row_records(
                episode_results=episode_results,
                traces=traces,
                envs=envs,
                fitnesses=fitnesses,
                entries=entries,
                specimen_id=specimen_id,
                run_id=run_id,
                assay_family=assay_family,
                condition_id=condition_id,
                condition_label=condition_label,
                signaling_condition=signaling_condition,
                deformation_condition=deformation_condition,
                sensing_freeze_mode=sensing_freeze_mode,
                body_label=body_label,
                max_steps=int(max_steps),
            )
            if render_output_dir is not None:
                render_candidates.extend(
                    _native_render_candidates(
                        episode_results=episode_results,
                        traces=traces,
                        envs=envs,
                        entries=entries,
                        fitnesses=fitnesses,
                        trial_id_prefix=str(trial_id_prefix or f"{assay_family}_{surface_id}"),
                        metadata_overrides={
                            "surface_id": surface_id,
                            "surface_label": _surface_label(surface_id),
                            "surface_max_steps": int(max_steps),
                            "sensing_freeze_mode": sensing_freeze_mode,
                            **(dict(render_metadata) if render_metadata else {}),
                        },
                    )
                )
        for row in pair_rows:
            row["surface_id"] = surface_id
            row["surface_label"] = _surface_label(surface_id)
            row["surface_max_steps"] = int(max_steps)
        rows.extend(pair_rows)
        budget_records.extend(list(pair_group["budget_records"]))
    df = pd.DataFrame(rows)
    if render_output_dir is not None and render_candidates:
        _render_selected_candidates(
            cfg=cfg,
            output_dir=render_output_dir,
            candidates=render_candidates,
            manifest_kind="sensing_edges_post_run_render_manifest",
        )
    summaries = _condition_summary_records(
        df,
        group_col="distance_pair",
        metrics=_native_metrics_schema(),
        seed_prefix=_stable_seed(run_id, specimen_id, assay_family, surface_id, "summary"),
        success_col="success",
        fall_col="fell",
    )
    surface_record = _native_surface_specimen_record(
        specimen_id=specimen_id,
        assay_family=assay_family,
        surface_id=surface_id,
        rows=df,
        extra_fields={"max_steps_by_pair": budget_records},
    )
    return df, summaries, budget_records, surface_record


def _run_native_like_surface_batched(
    *,
    cfg: DictConfig,
    runtime_resolution,
    jobs: Sequence[BatchedPostRunJob],
    assay_family: str,
    surface_id: str,
    pairs: list[dict[str, int]],
    trials_per_side: int,
    signaling_condition: str = "intact",
    deformation_condition: str = "intact",
    sensing_freeze_mode: str = "none",
    body_label: str = "rect_7x2",
    robot_w: int | None = None,
    robot_h: int | None = None,
    body_shape: str | None = None,
    render_output_dirs: dict[str, Path | None] | None = None,
    trial_id_prefix: str | None = None,
    render_metadata: dict[str, Any] | None = None,
    condition_id_override: str | None = None,
    condition_label_override: str | None = None,
) -> dict[str, tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]]:
    rows_by_specimen: dict[str, list[dict[str, Any]]] = {
        job.specimen_id: []
        for job in jobs
    }
    render_candidates_by_specimen: dict[str, list[dict[str, Any]]] = {
        job.specimen_id: []
        for job in jobs
    }
    budget_records: list[dict[str, Any]] = []
    capture_mode = (
        "com_trace"
        if _fast_ablation_eval_enabled(cfg)
        and assay_family in {"signaling_ablation", "deformation_ablation", "frozen_sensing_control", "body_transfer", "combined_ablation_quick"}
        else "trace"
    )
    condition_id, condition_label = _native_like_condition_identity(
        assay_family=assay_family,
        surface_id=surface_id,
        signaling_condition=signaling_condition,
        deformation_condition=deformation_condition,
        sensing_freeze_mode=sensing_freeze_mode,
        body_label=body_label,
        condition_id_override=condition_id_override,
        condition_label_override=condition_label_override,
    )
    genotypes = np.stack([np.asarray(job.genotype, dtype=np.float32) for job in jobs], axis=0)
    pair_groups = _native_like_pair_groups(
        cfg=cfg,
        surface_id=surface_id,
        pairs=pairs,
        trials_per_side=int(trials_per_side),
        signaling_condition=signaling_condition,
        deformation_condition=deformation_condition,
        sensing_freeze_mode=sensing_freeze_mode,
        body_label=body_label,
        robot_w=robot_w,
        robot_h=robot_h,
        body_shape=body_shape,
    )
    for group_index, pair_group in enumerate(pair_groups):
        phase_cfg = pair_group["phase_cfg"]
        max_steps = int(pair_group["max_steps"])
        entries = list(pair_group["entries"])
        seeds = [
            _stable_seed(
                job.run_id,
                job.specimen_id,
                assay_family,
                surface_id,
                signaling_condition,
                deformation_condition,
                sensing_freeze_mode,
                body_label,
                "group",
                group_index,
            )
            for job in jobs
        ]
        if capture_mode == "com_trace":
            episode_results_batches, com_trace_batches, _envs, fitness_batches = _run_com_trace_eval_batched(
                phase_cfg=phase_cfg,
                runtime_resolution=runtime_resolution,
                genotypes=genotypes,
                entries=entries,
                seeds=seeds,
            )
            for job, episode_results, com_traces, fitnesses in zip(
                jobs,
                episode_results_batches,
                com_trace_batches,
                fitness_batches,
            ):
                pair_rows = _native_com_trace_row_records(
                    episode_results=episode_results,
                    com_traces=com_traces,
                    fitnesses=fitnesses,
                    entries=entries,
                    specimen_id=job.specimen_id,
                    run_id=job.run_id,
                    assay_family=assay_family,
                    condition_id=condition_id,
                    condition_label=condition_label,
                    signaling_condition=signaling_condition,
                    deformation_condition=deformation_condition,
                    sensing_freeze_mode=sensing_freeze_mode,
                    body_label=body_label,
                    max_steps=int(max_steps),
                )
                if render_output_dirs and render_output_dirs.get(job.specimen_id) is not None:
                    render_candidates_by_specimen[job.specimen_id].extend(
                        _render_native_candidates_from_summary(
                            phase_cfg=phase_cfg,
                            runtime_resolution=runtime_resolution,
                            genotype=job.genotype,
                            entries=entries,
                            episode_results=episode_results,
                            fitnesses=fitnesses,
                            seed=int(
                                _stable_seed(
                                    job.run_id,
                                    job.specimen_id,
                                    assay_family,
                                    surface_id,
                                    signaling_condition,
                                    deformation_condition,
                                    sensing_freeze_mode,
                                    body_label,
                                    "group",
                                    group_index,
                                    "render",
                                )
                            ),
                            trial_id_prefix=(
                                f"{trial_id_prefix or f'{assay_family}_{surface_id}'}_{job.specimen_id}"
                            ),
                            metadata_overrides={
                                "surface_id": surface_id,
                                "surface_label": _surface_label(surface_id),
                                "surface_max_steps": int(max_steps),
                                "sensing_freeze_mode": sensing_freeze_mode,
                                **(dict(render_metadata) if render_metadata else {}),
                            },
                        )
                    )
                for row in pair_rows:
                    row["surface_id"] = surface_id
                    row["surface_label"] = _surface_label(surface_id)
                    row["surface_max_steps"] = int(max_steps)
                rows_by_specimen[job.specimen_id].extend(pair_rows)
        else:
            episode_results_batches, trace_batches, envs, fitness_batches = _run_trace_eval_batched(
                phase_cfg=phase_cfg,
                runtime_resolution=runtime_resolution,
                genotypes=genotypes,
                entries=entries,
                workload="trace",
                seeds=seeds,
            )
            for job, episode_results, traces, fitnesses in zip(
                jobs,
                episode_results_batches,
                trace_batches,
                fitness_batches,
            ):
                pair_rows = _native_row_records(
                    episode_results=episode_results,
                    traces=traces,
                    envs=envs,
                    fitnesses=fitnesses,
                    entries=entries,
                    specimen_id=job.specimen_id,
                    run_id=job.run_id,
                    assay_family=assay_family,
                    condition_id=condition_id,
                    condition_label=condition_label,
                    signaling_condition=signaling_condition,
                    deformation_condition=deformation_condition,
                    sensing_freeze_mode=sensing_freeze_mode,
                    body_label=body_label,
                    max_steps=int(max_steps),
                )
                if render_output_dirs and render_output_dirs.get(job.specimen_id) is not None:
                    render_candidates_by_specimen[job.specimen_id].extend(
                        _native_render_candidates(
                            episode_results=episode_results,
                            traces=traces,
                            envs=envs,
                            entries=entries,
                            fitnesses=fitnesses,
                            trial_id_prefix=(
                                f"{trial_id_prefix or f'{assay_family}_{surface_id}'}_{job.specimen_id}"
                            ),
                            metadata_overrides={
                                "surface_id": surface_id,
                                "surface_label": _surface_label(surface_id),
                                "surface_max_steps": int(max_steps),
                                "sensing_freeze_mode": sensing_freeze_mode,
                                **(dict(render_metadata) if render_metadata else {}),
                            },
                        )
                    )
                for row in pair_rows:
                    row["surface_id"] = surface_id
                    row["surface_label"] = _surface_label(surface_id)
                    row["surface_max_steps"] = int(max_steps)
                rows_by_specimen[job.specimen_id].extend(pair_rows)
        budget_records.extend(list(pair_group["budget_records"]))
    results: dict[str, tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]] = {}
    for job in jobs:
        df = pd.DataFrame(rows_by_specimen[job.specimen_id])
        render_output_dir = None if render_output_dirs is None else render_output_dirs.get(job.specimen_id)
        if render_output_dir is not None and render_candidates_by_specimen[job.specimen_id]:
            _render_selected_candidates(
                cfg=cfg,
                output_dir=render_output_dir,
                candidates=render_candidates_by_specimen[job.specimen_id],
                manifest_kind="sensing_edges_post_run_render_manifest",
            )
        summaries = _condition_summary_records(
            df,
            group_col="distance_pair",
            metrics=_native_metrics_schema(),
            seed_prefix=_stable_seed(job.run_id, job.specimen_id, assay_family, surface_id, "summary"),
            success_col="success",
            fall_col="fell",
        )
        surface_record = _native_surface_specimen_record(
            specimen_id=job.specimen_id,
            assay_family=assay_family,
            surface_id=surface_id,
            rows=df,
            extra_fields={"max_steps_by_pair": list(budget_records)},
        )
        results[job.specimen_id] = (df, summaries, list(budget_records), surface_record)
    return results


def _native_com_trace_row_records(
    *,
    episode_results,
    com_traces,
    fitnesses: np.ndarray,
    entries: list[TerrainEntry],
    specimen_id: str,
    run_id: str,
    assay_family: str,
    condition_id: str,
    condition_label: str,
    signaling_condition: str = "intact",
    deformation_condition: str = "intact",
    sensing_freeze_mode: str = "none",
    body_label: str = "rect_7x2",
    max_steps: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    episode_count = int(np.asarray(episode_results.steps_alive).shape[0])
    for index in range(episode_count):
        result = jax.tree_util.tree_map(lambda array: np.asarray(array[index]), episode_results)
        trace = jax.tree_util.tree_map(lambda array: np.asarray(array[index]), com_traces)
        metadata = dict(entries[index].metadata or {})
        goal_distance = int(metadata.get("goal_distance", result.goal_distance))
        cliff_distance = int(metadata.get("cliff_distance", result.cliff_distance))
        goal_side = str(metadata.get("goal_side", metadata.get("assay_goal_side", "unknown")))
        steps_alive = int(result.steps_alive)
        com_x = np.asarray(trace.com_x, dtype=np.float32)[:steps_alive] if steps_alive > 0 else np.zeros((0,), dtype=np.float32)
        cliff_vicinity_steps = int(np.asarray(result.cliff_vicinity_steps))
        first_vicinity_step = None
        if cliff_vicinity_steps > 0:
            first_vicinity_step = max(
                1,
                min(
                    int(max_steps),
                    int(np.ceil(float(np.asarray(result.time_to_edge)) * float(max_steps))),
                ),
            )
        initial_direction, initial_commit_step, initial_commit_x = _extract_initial_commit(
            com_x=com_x,
            init_com_x=float(result.init_com_x),
            turn_motion_threshold=float(result.turn_motion_threshold),
        )
        time_to_arrest = _find_arrest_step(
            com_x=com_x,
            init_com_x=float(result.init_com_x),
            turn_motion_threshold=float(result.turn_motion_threshold),
            first_cliff_vicinity_step=first_vicinity_step,
        )
        rows.append(
            {
                "run_id": str(run_id),
                "specimen_id": str(specimen_id),
                "trial_id": f"{assay_family}_{index:04d}",
                "assay_family": assay_family,
                "condition_id": str(condition_id),
                "condition_label": str(condition_label),
                "goal_distance": int(goal_distance),
                "cliff_distance": int(cliff_distance),
                "distance_pair": _pair_label(goal_distance, cliff_distance),
                "side": str(goal_side),
                "body": str(body_label),
                "signaling_condition": str(signaling_condition),
                "deformation_condition": str(deformation_condition),
                "sensing_freeze_mode": str(sensing_freeze_mode),
                "success": bool(result.goal_touched),
                "fell": bool(result.cliff_fell),
                "deathplane_contact": bool(result.touched_death_plane),
                "left_exit": bool(result.com_x_went_negative),
                "steps_alive": int(steps_alive),
                "termination_reason": _episode_termination_reason(
                    goal_touched=bool(result.goal_touched),
                    touched_death_plane=bool(result.touched_death_plane),
                    went_airborne=bool(result.went_airborne),
                    left_exit=bool(result.com_x_went_negative),
                    steps_alive=int(steps_alive),
                    max_steps=int(max_steps),
                ),
                "fitness": float(fitnesses[index]),
                "final_com_x": None if steps_alive <= 0 else float(result.final_com_x),
                "peak_front_x": float(result.peak_front_x),
                "safe_front_x": float(result.safe_front_x),
                "probe_extent": float(max(0.0, float(result.peak_front_x) - float(result.safe_front_x))),
                "min_edge_clearance": None,
                "max_cliff_protrusion": None,
                "stopping_distance": None,
                "time_to_first_cliff_vicinity": first_vicinity_step,
                "time_to_arrest": time_to_arrest,
                "cliff_vicinity_fraction": float(result.cliff_vicinity_fraction),
                "mean_activation": float(result.mean_activation),
                "mean_speed": float(result.mean_speed),
                "turn_count": int(result.turn_count),
                "initial_direction": initial_direction,
                "initial_commit_step": initial_commit_step,
                "initial_commit_x": initial_commit_x,
            }
        )
    return rows


def _native_row_records(
    *,
    episode_results,
    traces,
    envs: list[Any],
    fitnesses: np.ndarray,
    entries: list[TerrainEntry],
    specimen_id: str,
    run_id: str,
    assay_family: str,
    condition_id: str,
    condition_label: str,
    signaling_condition: str = "intact",
    deformation_condition: str = "intact",
    sensing_freeze_mode: str = "none",
    body_label: str = "rect_7x2",
    max_steps: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    episode_count = int(np.asarray(episode_results.steps_alive).shape[0])
    for index in range(episode_count):
        result = jax.tree_util.tree_map(lambda array: np.asarray(array[index]), episode_results)
        trace = jax.tree_util.tree_map(lambda array: np.asarray(array[index]), traces)
        env_data = envs[index]
        metadata = dict((entries[index].metadata or {}))
        goal_distance = int(metadata.get("goal_distance", result.goal_distance))
        cliff_distance = int(metadata.get("cliff_distance", result.cliff_distance))
        goal_side = str(metadata.get("goal_side", metadata.get("assay_goal_side", "unknown")))
        cliff_side = str(metadata.get("cliff_side", "unknown"))
        cliff_edge_x = int(metadata.get("terminal_cliff_edge_x", metadata.get("cliff_edge_x", entries[index].cliff_x)))
        steps_alive = int(result.steps_alive)
        trace_positions = np.asarray(trace.positions, dtype=np.float32)
        valid_positions = trace_positions[:steps_alive] if steps_alive > 0 else trace_positions[:0]
        if steps_alive > 0:
            com_x, min_x, front_x = _trace_kinematics(valid_positions, np.asarray(env_data.robot_point_mask))
        else:
            com_x = np.zeros((0,), dtype=np.float32)
            min_x = np.zeros((0,), dtype=np.float32)
            front_x = np.zeros((0,), dtype=np.float32)
        cliff_facing_edge = _cliff_facing_edge(min_x, front_x, cliff_side) if steps_alive > 0 else np.zeros((0,), dtype=np.float32)
        frontier_x = _cliff_frontier_world_x(cliff_edge_x, cliff_side)
        clearances = (
            frontier_x - cliff_facing_edge
            if cliff_side == "right"
            else cliff_facing_edge - frontier_x
        )
        vicinity_flags = clearances <= _CLIFF_VICINITY_BAND
        first_vicinity_step = int(np.argmax(vicinity_flags) + 1) if steps_alive > 0 and bool(np.any(vicinity_flags)) else None
        initial_direction, initial_commit_step, initial_commit_x = _extract_initial_commit(
            com_x=com_x,
            init_com_x=float(result.init_com_x),
            turn_motion_threshold=float(result.turn_motion_threshold),
        )
        time_to_arrest = _find_arrest_step(
            com_x=com_x,
            init_com_x=float(result.init_com_x),
            turn_motion_threshold=float(result.turn_motion_threshold),
            first_cliff_vicinity_step=first_vicinity_step,
        )
        min_edge_clearance = None if steps_alive <= 0 else float(np.min(clearances))
        stopping_distance = _stopping_distance_for_trial(
            clearances=clearances,
            cliff_fell=bool(result.cliff_fell),
            first_cliff_vicinity_step=first_vicinity_step,
        )
        max_cliff_protrusion = _max_cliff_protrusion_for_trial(
            min_edge_clearance=min_edge_clearance,
            cliff_fell=bool(result.cliff_fell),
            first_cliff_vicinity_step=first_vicinity_step,
        )
        rows.append(
            {
                "run_id": str(run_id),
                "specimen_id": str(specimen_id),
                "trial_id": f"{assay_family}_{index:04d}",
                "assay_family": assay_family,
                "condition_id": str(condition_id),
                "condition_label": str(condition_label),
                "goal_distance": int(goal_distance),
                "cliff_distance": int(cliff_distance),
                "distance_pair": _pair_label(goal_distance, cliff_distance),
                "side": str(goal_side),
                "body": str(body_label),
                "signaling_condition": str(signaling_condition),
                "deformation_condition": str(deformation_condition),
                "sensing_freeze_mode": str(sensing_freeze_mode),
                "success": bool(result.goal_touched),
                "fell": bool(result.cliff_fell),
                "deathplane_contact": bool(result.touched_death_plane),
                "left_exit": bool(result.com_x_went_negative),
                "steps_alive": int(steps_alive),
                "termination_reason": _episode_termination_reason(
                    goal_touched=bool(result.goal_touched),
                    touched_death_plane=bool(result.touched_death_plane),
                    went_airborne=bool(result.went_airborne),
                    left_exit=bool(result.com_x_went_negative),
                    steps_alive=int(steps_alive),
                    max_steps=int(max_steps),
                ),
                "fitness": float(fitnesses[index]),
                "final_com_x": None if steps_alive <= 0 else float(com_x[-1]),
                "peak_front_x": float(result.peak_front_x),
                "min_edge_clearance": min_edge_clearance,
                "max_cliff_protrusion": max_cliff_protrusion,
                "stopping_distance": stopping_distance,
                "time_to_first_cliff_vicinity": first_vicinity_step,
                "time_to_arrest": time_to_arrest,
                "cliff_vicinity_fraction": (
                    float(np.mean(vicinity_flags.astype(np.float32)))
                    if steps_alive > 0
                    else 0.0
                ),
                "mean_activation": float(result.mean_activation),
                "mean_speed": float(result.mean_speed),
                "turn_count": int(result.turn_count),
                "initial_direction": initial_direction,
                "initial_commit_step": initial_commit_step,
                "initial_commit_x": initial_commit_x,
            }
        )
    return rows


def _run_native_core_assay(
    *,
    cfg: DictConfig,
    runtime_resolution,
    specimen_dir: Path,
    specimen_id: str,
    genotype: np.ndarray,
    selection_row: dict[str, Any],
    run_id: str,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    post_run_cfg = cfg.eval.post_run
    assay_dir = specimen_dir / "native_core"
    effective_trials_per_side = _effective_trials_per_side(cfg)
    rows, summaries, budget_records, surface_record = _run_native_like_surface(
        cfg=cfg,
        runtime_resolution=runtime_resolution,
        genotype=genotype,
        run_id=run_id,
        specimen_id=specimen_id,
        assay_family="native_core",
        surface_id="native_core",
        pairs=list(post_run_cfg.native_pairs),
        trials_per_side=int(effective_trials_per_side),
        render_output_dir=assay_dir / "renders",
        trial_id_prefix="native_core",
    )
    specimen_summary = {
        "pooled_native_success_rate": _summary_stats(
            rows["success"].astype(float).to_numpy(),
            seed=_stable_seed(run_id, specimen_id, "native_core", "pooled_success"),
        ),
        "pooled_native_fall_rate": _summary_stats(
            rows["fell"].astype(float).to_numpy(),
            seed=_stable_seed(run_id, specimen_id, "native_core", "pooled_fall"),
        ),
        "pooled_native_max_cliff_protrusion": _summary_stats(
            rows["max_cliff_protrusion"].astype(float).to_numpy(),
            seed=_stable_seed(run_id, specimen_id, "native_core", "pooled_max_cliff_protrusion"),
        ),
        "outcome_breakdown": _outcome_breakdown(rows, success_col="success", fall_col="fell"),
        "surface_summary": surface_record,
        "requested_trials_per_side": int(post_run_cfg.trials_per_side),
        "effective_trials_per_side": int(effective_trials_per_side),
        "deterministic_rollout": bool(effective_trials_per_side == 1),
        "selection_summary": dict(selection_row),
    }
    charts_dir = assay_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    _chart_bar(
        categories=[record["distance_pair"] for record in summaries],
        values=[float(record["success_rate"]["mean"] or 0.0) for record in summaries],
        ci_bounds=[record["success_rate"]["bootstrap_ci95"] for record in summaries],
        title="Success Rate By Distance Pair",
        ylabel="success rate",
        output_path=charts_dir / "success_by_pair.png",
    )
    _chart_bar(
        categories=[record["distance_pair"] for record in summaries],
        values=[float(record["fall_rate"]["mean"] or 0.0) for record in summaries],
        ci_bounds=[record["fall_rate"]["bootstrap_ci95"] for record in summaries],
        title="Fall Rate By Distance Pair",
        ylabel="fall rate",
        output_path=charts_dir / "fall_rate_by_pair.png",
    )
    _chart_bar(
        categories=[record["distance_pair"] for record in summaries],
        values=[float(record["mean_max_cliff_protrusion"]["mean"] or 0.0) for record in summaries],
        ci_bounds=[record["mean_max_cliff_protrusion"]["bootstrap_ci95"] for record in summaries],
        title="Max Cliff Protrusion By Distance Pair",
        ylabel="distance",
        output_path=charts_dir / "max_cliff_protrusion_by_pair.png",
    )
    _chart_bar(
        categories=[record["distance_pair"] for record in summaries],
        values=[float(record["mean_time_to_arrest"]["mean"] or 0.0) for record in summaries],
        ci_bounds=[record["mean_time_to_arrest"]["bootstrap_ci95"] for record in summaries],
        title="Time To Arrest By Distance Pair",
        ylabel="env steps",
        output_path=charts_dir / "time_to_arrest_by_pair.png",
    )
    aggregate = {
        "schema_version": _POST_RUN_SCHEMA_VERSION,
        "kind": "sensing_edges_post_run_native_core",
        "specimen_id": specimen_id,
        "surface_id": "native_core",
        "condition_summaries": summaries,
        "specimen_summary": specimen_summary,
    }
    condition_records = []
    inventory_records = []
    for record in summaries:
        condition_record = dict(record)
        condition_record.update(
            {
                "specimen_id": specimen_id,
                "assay_family": "native_core",
                "surface_id": "native_core",
                "surface_label": _surface_label("native_core"),
            }
        )
        condition_records.append(condition_record)
        inventory_records.append(
            _surface_inventory_record(
                assay_family="native_core",
                surface_id="native_core",
                subgroup_type="distance_pair",
                subgroup_id=str(record["distance_pair"]),
                surface_dir=assay_dir,
            )
        )
    _finalize_assay_outputs(
        assay_dir=assay_dir,
        manifest={
            "schema_version": _POST_RUN_SCHEMA_VERSION,
            "kind": "sensing_edges_post_run_assay_manifest",
            "assay_family": "native_core",
            "specimen_id": specimen_id,
            "run_id": run_id,
            "surface_label": _surface_label("native_core"),
            "max_steps": _native_max_steps(cfg),
            "pairs": list(post_run_cfg.native_pairs),
            "requested_trials_per_side": int(post_run_cfg.trials_per_side),
            "effective_trials_per_side": int(effective_trials_per_side),
            "deterministic_rollout": bool(effective_trials_per_side == 1),
        },
        rows=rows,
        aggregate=aggregate,
        condition_records=condition_records,
        specimen_records=[surface_record],
        inventory_records=inventory_records,
    )
    return rows, aggregate, assay_dir


def _group_surface_ids() -> tuple[str, ...]:
    return ("native_core", "long_distance_generalization", "double_cliff")


def _configured_surface_ids(cfg: DictConfig, *, config_key: str | None = None) -> tuple[str, ...]:
    surface_ids = _group_surface_ids()
    if config_key is None:
        return surface_ids
    configured = cfg.eval.post_run.get(config_key)
    if not configured:
        return surface_ids
    normalized = [str(surface_id).strip() for surface_id in list(configured) if str(surface_id).strip()]
    invalid = sorted(set(normalized) - set(surface_ids))
    if invalid:
        raise ValueError(f"Unsupported post-run surfaces for {config_key}: {invalid}")
    ordered = tuple(surface_id for surface_id in surface_ids if surface_id in normalized)
    return ordered if ordered else surface_ids


def _reference_body_spec(cfg: DictConfig) -> dict[str, Any]:
    return {
        "label": f"rect_{int(cfg.terrain.robot_w)}x{int(cfg.terrain.robot_h)}",
        "robot_w": int(cfg.terrain.robot_w),
        "robot_h": int(cfg.terrain.robot_h),
        "shape": str(cfg.body.shape),
    }


def _chart_native_surface_family(
    *,
    charts_dir: Path,
    title_prefix: str,
    category_field: str,
    records: list[dict[str, Any]],
) -> None:
    def _chart_metric(
        *,
        metric_name: str,
        title: str,
        ylabel: str,
        filename: str,
    ) -> None:
        filtered = [record for record in records if _summary_metric_has_samples(record, metric_name)]
        if not filtered:
            return
        _chart_bar(
            categories=[str(record[category_field]) for record in filtered],
            values=[float(record[metric_name]["mean"] or 0.0) for record in filtered],
            ci_bounds=[record[metric_name]["bootstrap_ci95"] for record in filtered],
            title=title,
            ylabel=ylabel,
            output_path=charts_dir / filename,
        )

    _chart_metric(
        metric_name="success_rate",
        title=f"{title_prefix} Success Rate",
        ylabel="success rate",
        filename="success_by_condition.png",
    )
    _chart_metric(
        metric_name="fall_rate",
        title=f"{title_prefix} Fall Rate",
        ylabel="fall rate",
        filename="fall_rate_by_condition.png",
    )
    _chart_metric(
        metric_name="mean_max_cliff_protrusion",
        title=f"{title_prefix} Max Cliff Protrusion",
        ylabel="distance",
        filename="max_cliff_protrusion_by_condition.png",
    )
    _chart_outcome_breakdown(
        categories=[str(record[category_field]) for record in records],
        outcome_breakdowns=[record["outcome_breakdown"] for record in records],
        title=f"{title_prefix} Outcome Composition",
        output_path=charts_dir / "outcome_breakdown.png",
    )


def _chart_double_cliff_surface_family(
    *,
    charts_dir: Path,
    title_prefix: str,
    category_field: str,
    records: list[dict[str, Any]],
) -> None:
    def _chart_metric(
        *,
        metric_name: str,
        title: str,
        ylabel: str,
        filename: str,
    ) -> None:
        filtered = [record for record in records if _summary_metric_has_samples(record, metric_name)]
        if not filtered:
            return
        _chart_bar(
            categories=[str(record[category_field]) for record in filtered],
            values=[float(record[metric_name]["mean"] or 0.0) for record in filtered],
            ci_bounds=[record[metric_name]["bootstrap_ci95"] for record in filtered],
            title=title,
            ylabel=ylabel,
            output_path=charts_dir / filename,
        )

    _chart_metric(
        metric_name="survival_rate",
        title=f"{title_prefix} Survival Rate",
        ylabel="survival rate",
        filename="survival_by_condition.png",
    )
    _chart_metric(
        metric_name="mean_side_switch_count",
        title=f"{title_prefix} Side Switch Count",
        ylabel="switch count",
        filename="side_switch_count_by_condition.png",
    )
    bias_records = [record for record in records if record.get("left_first_side_bias") is not None]
    if bias_records:
        _chart_bar(
            categories=[str(record[category_field]) for record in bias_records],
            values=[float(record.get("left_first_side_bias", 0.0) or 0.0) for record in bias_records],
            ci_bounds=None,
            title=f"{title_prefix} Left First-Side Bias",
            ylabel="fraction",
            output_path=charts_dir / "left_first_side_bias.png",
        )
    _chart_outcome_breakdown(
        categories=[str(record[category_field]) for record in records],
        outcome_breakdowns=[record["outcome_breakdown"] for record in records],
        title=f"{title_prefix} Outcome Composition",
        output_path=charts_dir / "outcome_breakdown.png",
    )


def _overall_multi_surface_group_records(
    df: pd.DataFrame,
    *,
    specimen_id: str,
    assay_family: str,
    group_field: str,
    group_label_field: str,
) -> list[dict[str, Any]]:
    if df.empty or group_field not in df.columns:
        return []

    metric_columns = {
        "success_rate": "success",
        "fall_rate": "fell",
        "mean_fitness": "fitness",
        "mean_max_cliff_protrusion": "max_cliff_protrusion",
        "mean_time_to_arrest": "time_to_arrest",
        "mean_cliff_vicinity_fraction": "cliff_vicinity_fraction",
        "mean_activation": "mean_activation",
        "mean_speed": "mean_speed",
        "mean_turn_count": "turn_count",
        "survival_rate": "survived",
        "mean_side_switch_count": "side_switch_count",
        "mean_transfer_ratio": "transfer_ratio",
    }

    records: list[dict[str, Any]] = []
    for group_value, group_df in df.groupby(group_field, sort=False):
        group_df = group_df.copy()
        label_series = (
            group_df[group_label_field].dropna()
            if group_label_field in group_df.columns
            else pd.Series(dtype=object)
        )
        group_label = str(label_series.iloc[0]) if not label_series.empty else str(group_value)
        surface_ids = (
            [str(surface_id) for surface_id in pd.unique(group_df["surface_id"].dropna())]
            if "surface_id" in group_df.columns
            else []
        )
        surface_trial_counts = (
            [
                {"surface_id": str(surface_id), "trial_count": int(count)}
                for surface_id, count in group_df.groupby("surface_id", sort=False).size().items()
            ]
            if "surface_id" in group_df.columns
            else []
        )
        record: dict[str, Any] = {
            "specimen_id": str(specimen_id),
            "assay_family": str(assay_family),
            group_field: str(group_value),
            group_label_field: str(group_label),
            "trial_count": int(len(group_df)),
            "surface_count": int(len(surface_ids)),
            "surface_ids": surface_ids,
            "surface_trial_counts": surface_trial_counts,
        }
        if "reference" in group_df.columns:
            record["reference"] = bool(np.any(group_df["reference"].fillna(False).astype(bool).to_numpy()))
        for metric_name, column in metric_columns.items():
            if column not in group_df.columns:
                continue
            record[metric_name] = _summary_stats(
                group_df[column].astype(float).to_numpy(),
                seed=_stable_seed(specimen_id, assay_family, "overall", group_field, group_value, metric_name),
            )
        if "success" in group_df.columns and "fell" in group_df.columns:
            native_df = group_df[group_df["success"].notna() & group_df["fell"].notna()]
            if not native_df.empty:
                record["native_outcome_breakdown"] = _outcome_breakdown(
                    native_df,
                    success_col="success",
                    fall_col="fell",
                )
        if "survived" in group_df.columns and "fell" in group_df.columns:
            double_cliff_df = group_df[group_df["survived"].notna() & group_df["fell"].notna()]
            if not double_cliff_df.empty:
                record["double_cliff_outcome_breakdown"] = _outcome_breakdown(
                    double_cliff_df,
                    success_col="survived",
                    fall_col="fell",
                )
        records.append(record)
    return records


def _summary_metric_has_samples(record: dict[str, Any], metric_name: str) -> bool:
    metric = record.get(metric_name)
    return isinstance(metric, dict) and int(metric.get("n", 0) or 0) > 0 and metric.get("mean") is not None


def _chart_overall_multi_surface_family(
    *,
    charts_dir: Path,
    title_prefix: str,
    category_field: str,
    records: list[dict[str, Any]],
) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)

    def _chart_metric(
        *,
        metric_name: str,
        title: str,
        ylabel: str,
        filename: str,
    ) -> None:
        filtered = [record for record in records if _summary_metric_has_samples(record, metric_name)]
        if not filtered:
            return
        _chart_bar(
            categories=[str(record[category_field]) for record in filtered],
            values=[float(record[metric_name]["mean"] or 0.0) for record in filtered],
            ci_bounds=[record[metric_name]["bootstrap_ci95"] for record in filtered],
            title=title,
            ylabel=ylabel,
            output_path=charts_dir / filename,
        )

    _chart_metric(
        metric_name="success_rate",
        title=f"{title_prefix} Success Rate",
        ylabel="success rate",
        filename="overall_success_by_condition.png",
    )
    _chart_metric(
        metric_name="fall_rate",
        title=f"{title_prefix} Fall Rate",
        ylabel="fall rate",
        filename="overall_fall_rate_by_condition.png",
    )
    _chart_metric(
        metric_name="mean_max_cliff_protrusion",
        title=f"{title_prefix} Max Cliff Protrusion",
        ylabel="distance",
        filename="overall_max_cliff_protrusion_by_condition.png",
    )
    _chart_metric(
        metric_name="mean_time_to_arrest",
        title=f"{title_prefix} Time To Arrest",
        ylabel="env steps",
        filename="overall_time_to_arrest_by_condition.png",
    )
    _chart_metric(
        metric_name="survival_rate",
        title=f"{title_prefix} Survival Rate",
        ylabel="survival rate",
        filename="overall_survival_by_condition.png",
    )
    _chart_metric(
        metric_name="mean_side_switch_count",
        title=f"{title_prefix} Side Switch Count",
        ylabel="switch count",
        filename="overall_side_switch_count_by_condition.png",
    )
    _chart_metric(
        metric_name="mean_transfer_ratio",
        title=f"{title_prefix} Transfer Ratio",
        ylabel="ratio",
        filename="overall_transfer_ratio_by_condition.png",
    )

    native_records = [record for record in records if "native_outcome_breakdown" in record]
    if native_records:
        _chart_outcome_breakdown(
            categories=[str(record[category_field]) for record in native_records],
            outcome_breakdowns=[record["native_outcome_breakdown"] for record in native_records],
            title=f"{title_prefix} Native Outcome Composition",
            output_path=charts_dir / "overall_native_outcome_breakdown.png",
        )
    double_cliff_records = [record for record in records if "double_cliff_outcome_breakdown" in record]
    if double_cliff_records:
        _chart_outcome_breakdown(
            categories=[str(record[category_field]) for record in double_cliff_records],
            outcome_breakdowns=[record["double_cliff_outcome_breakdown"] for record in double_cliff_records],
            title=f"{title_prefix} Double-Cliff Outcome Composition",
            output_path=charts_dir / "overall_double_cliff_outcome_breakdown.png",
        )

def _run_long_distance_assay(
    *,
    cfg: DictConfig,
    runtime_resolution,
    specimen_dir: Path,
    specimen_id: str,
    genotype: np.ndarray,
    selection_row: dict[str, Any],
    run_id: str,
    native_rows: pd.DataFrame,
    native_aggregate: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    del selection_row
    post_run_cfg = cfg.eval.post_run
    assay_dir = specimen_dir / "long_distance_generalization"
    effective_trials_per_side = _effective_trials_per_side(cfg)
    rows, summaries, budget_records, surface_record = _run_native_like_surface(
        cfg=cfg,
        runtime_resolution=runtime_resolution,
        genotype=genotype,
        run_id=run_id,
        specimen_id=specimen_id,
        assay_family="long_distance_generalization",
        surface_id="long_distance_generalization",
        pairs=list(post_run_cfg.long_distance_pairs),
        trials_per_side=int(effective_trials_per_side),
        render_output_dir=assay_dir / "renders",
        trial_id_prefix="long_distance_generalization",
    )
    native_success = native_aggregate["specimen_summary"]["pooled_native_success_rate"]["mean"]
    native_fall = native_aggregate["specimen_summary"]["pooled_native_fall_rate"]["mean"]
    native_protrusion = native_aggregate["specimen_summary"]["pooled_native_max_cliff_protrusion"]["mean"]
    native_arrest = _summary_stats(
        native_rows["time_to_arrest"].astype(float).to_numpy(),
        seed=_stable_seed(run_id, specimen_id, "long_distance_generalization", "native_arrest"),
    )["mean"]
    for summary in summaries:
        summary["delta_from_native"] = {
            "success_rate": _relative_delta(summary["success_rate"]["mean"], native_success),
            "fall_rate": _relative_delta(summary["fall_rate"]["mean"], native_fall),
            "max_cliff_protrusion": _relative_delta(
                summary["mean_max_cliff_protrusion"]["mean"],
                native_protrusion,
            ),
            "time_to_arrest": _relative_delta(summary["mean_time_to_arrest"]["mean"], native_arrest),
        }
    charts_dir = assay_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    _chart_line(
        categories=[record["distance_pair"] for record in summaries],
        values=[float(record["success_rate"]["mean"] or 0.0) for record in summaries],
        title="Success Rate Vs Distance Pair",
        ylabel="success rate",
        output_path=charts_dir / "success_vs_pair.png",
    )
    _chart_line(
        categories=[record["distance_pair"] for record in summaries],
        values=[float(record["fall_rate"]["mean"] or 0.0) for record in summaries],
        title="Fall Rate Vs Distance Pair",
        ylabel="fall rate",
        output_path=charts_dir / "fall_rate_vs_pair.png",
    )
    _chart_bar(
        categories=[record["distance_pair"] for record in summaries],
        values=[float(record["delta_from_native"]["success_rate"] or 0.0) for record in summaries],
        ci_bounds=None,
        title="Success Delta From Native",
        ylabel="delta success",
        output_path=charts_dir / "delta_from_native.png",
    )
    _chart_outcome_breakdown(
        categories=[record["distance_pair"] for record in summaries],
        outcome_breakdowns=[record["outcome_breakdown"] for record in summaries],
        title="Long-Distance Outcome Composition",
        output_path=charts_dir / "outcome_breakdown.png",
    )
    aggregate = {
        "schema_version": _POST_RUN_SCHEMA_VERSION,
        "kind": "sensing_edges_post_run_long_distance",
        "specimen_id": specimen_id,
        "surface_id": "long_distance_generalization",
        "condition_summaries": summaries,
        "specimen_summary": {
            "native_pooled": native_aggregate["specimen_summary"],
            "surface_summary": surface_record,
            "requested_trials_per_side": int(post_run_cfg.trials_per_side),
            "effective_trials_per_side": int(effective_trials_per_side),
            "deterministic_rollout": bool(effective_trials_per_side == 1),
        },
    }
    condition_records = []
    inventory_records = []
    for record in summaries:
        condition_record = dict(record)
        condition_record.update(
            {
                "specimen_id": specimen_id,
                "assay_family": "long_distance_generalization",
                "surface_id": "long_distance_generalization",
                "surface_label": _surface_label("long_distance_generalization"),
            }
        )
        condition_records.append(condition_record)
        inventory_records.append(
            _surface_inventory_record(
                assay_family="long_distance_generalization",
                surface_id="long_distance_generalization",
                subgroup_type="distance_pair",
                subgroup_id=str(record["distance_pair"]),
                surface_dir=assay_dir,
            )
        )
    _finalize_assay_outputs(
        assay_dir=assay_dir,
        manifest={
            "schema_version": _POST_RUN_SCHEMA_VERSION,
            "kind": "sensing_edges_post_run_assay_manifest",
            "assay_family": "long_distance_generalization",
            "specimen_id": specimen_id,
            "run_id": run_id,
            "surface_label": _surface_label("long_distance_generalization"),
            "max_steps_by_pair": budget_records,
            "pairs": list(post_run_cfg.long_distance_pairs),
            "requested_trials_per_side": int(post_run_cfg.trials_per_side),
            "effective_trials_per_side": int(effective_trials_per_side),
            "deterministic_rollout": bool(effective_trials_per_side == 1),
        },
        rows=rows,
        aggregate=aggregate,
        condition_records=condition_records,
        specimen_records=[surface_record],
        inventory_records=inventory_records,
    )
    return rows, aggregate, assay_dir


def _run_signaling_ablation_assay(
    *,
    cfg: DictConfig,
    runtime_resolution,
    specimen_dir: Path,
    specimen_id: str,
    genotype: np.ndarray,
    selection_row: dict[str, Any],
    run_id: str,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    del selection_row
    post_run_cfg = cfg.eval.post_run
    assay_dir = specimen_dir / "signaling_ablation"
    signaling_conditions = compatible_post_run_signaling_conditions(cfg)
    all_surface_frames: list[pd.DataFrame] = []
    surface_summaries: list[dict[str, Any]] = []
    cross_surface_records: list[dict[str, Any]] = []
    inventory_records: list[dict[str, Any]] = []
    configured_surfaces = _configured_surface_ids(cfg, config_key="signaling_surfaces")
    for surface_id in configured_surfaces:
        surface_dir = assay_dir / "surfaces" / surface_id
        condition_root = surface_dir / "conditions"
        surface_frames: list[pd.DataFrame] = []
        surface_condition_records: list[dict[str, Any]] = []
        max_steps_by_pair: list[dict[str, Any]] = []
        requested_trials_per_side = int(post_run_cfg.trials_per_side)
        requested_trials_per_width = int(post_run_cfg.double_cliff.trials_per_width)
        effective_trials_per_side = _effective_trials_per_side(cfg)
        effective_trials_per_width = _effective_trials_per_width(cfg)
        for condition in signaling_conditions:
            condition_dir = condition_root / str(condition)
            condition_render_dir = (
                condition_dir / "renders"
                if _should_render_signaling_condition(cfg, str(condition))
                else None
            )
            subgroup_started = time.monotonic()
            log.info(
                "Post-run subgroup started | specimen_id=%s assay=signaling_ablation surface=%s condition=%s render=%s",
                specimen_id,
                surface_id,
                condition,
                bool(condition_render_dir is not None),
            )
            if surface_id == "double_cliff":
                phase_cfg = _double_cliff_phase_cfg(
                    cfg,
                    active_signaling_condition=str(condition),
                )
                entries = _build_double_cliff_entries(
                    phase_cfg,
                    [int(width) for width in list(post_run_cfg.double_cliff.widths)],
                    plateau_width=int(post_run_cfg.double_cliff.plateau_width),
                    trials_per_width=int(effective_trials_per_width),
                )
                episode_results, traces, envs, fitnesses = _run_trace_eval(
                    phase_cfg=phase_cfg,
                    runtime_resolution=runtime_resolution,
                    genotype=genotype,
                    entries=entries,
                    workload="trace",
                    seed=_stable_seed(run_id, specimen_id, "signaling", surface_id, condition),
                )
                condition_df = pd.DataFrame(
                    _double_cliff_rows(
                        episode_results=episode_results,
                        traces=traces,
                        envs=envs,
                        entries=entries,
                        specimen_id=specimen_id,
                        run_id=run_id,
                        max_steps=_double_cliff_max_steps(cfg),
                    )
                )
                condition_df["fell"] = condition_df["fell_left"].astype(bool) | condition_df["fell_right"].astype(bool)
                condition_df["surface_id"] = surface_id
                condition_df["surface_label"] = _surface_label(surface_id)
                condition_df["surface_max_steps"] = _double_cliff_max_steps(cfg)
                condition_df["signaling_condition"] = str(condition)
                condition_df["condition_id"] = str(condition)
                condition_df["condition_label"] = _signaling_label(str(condition))
                if condition_render_dir is not None:
                    _render_selected_candidates(
                        cfg=cfg,
                        output_dir=condition_render_dir,
                        candidates=_double_cliff_render_candidates(
                            episode_results=episode_results,
                            traces=traces,
                            envs=envs,
                            entries=entries,
                            fitnesses=fitnesses,
                        ),
                        manifest_kind="sensing_edges_post_run_render_manifest",
                    )
                condition_record = _double_cliff_surface_specimen_record(
                    specimen_id=specimen_id,
                    assay_family="signaling_ablation",
                    surface_id=surface_id,
                    rows=condition_df,
                    extra_fields={
                        "signaling_condition": str(condition),
                        "condition_id": str(condition),
                        "condition_label": _signaling_label(str(condition)),
                        "max_steps": _double_cliff_max_steps(cfg),
                        "requested_trials_per_width": requested_trials_per_width,
                        "effective_trials_per_width": int(effective_trials_per_width),
                        "deterministic_rollout": bool(effective_trials_per_width == 1),
                        "render_enabled": bool(condition_render_dir is not None),
                    },
                )
            else:
                condition_df, pair_summaries, pair_budgets, condition_record = _run_native_like_surface(
                    cfg=cfg,
                    runtime_resolution=runtime_resolution,
                    genotype=genotype,
                    run_id=run_id,
                    specimen_id=specimen_id,
                    assay_family="signaling_ablation",
                    surface_id=surface_id,
                    pairs=(
                        list(post_run_cfg.native_pairs)
                        if surface_id == "native_core"
                        else list(post_run_cfg.long_distance_pairs)
                    ),
                    trials_per_side=int(effective_trials_per_side),
                    signaling_condition=str(condition),
                    render_output_dir=condition_render_dir,
                    trial_id_prefix=f"signaling_{surface_id}_{condition}",
                    render_metadata={"signaling_condition": str(condition)},
                )
                max_steps_by_pair.extend(pair_budgets)
                condition_record.update(
                    {
                        "signaling_condition": str(condition),
                        "condition_id": str(condition),
                        "condition_label": _signaling_label(str(condition)),
                        "distance_pair_summaries": pair_summaries,
                        "requested_trials_per_side": requested_trials_per_side,
                        "effective_trials_per_side": int(effective_trials_per_side),
                        "deterministic_rollout": bool(effective_trials_per_side == 1),
                        "render_enabled": bool(condition_render_dir is not None),
                    }
                )
            _write_json(condition_dir / "summary.json", condition_record)
            _write_json(condition_dir / "aggregate.json", condition_record)
            surface_frames.append(condition_df)
            surface_condition_records.append(condition_record)
            log.info(
                "Post-run subgroup completed | specimen_id=%s assay=signaling_ablation surface=%s condition=%s trial_count=%d elapsed_s=%.2f",
                specimen_id,
                surface_id,
                condition,
                len(condition_df),
                time.monotonic() - subgroup_started,
            )
            inventory_records.append(
                _surface_inventory_record(
                    assay_family="signaling_ablation",
                    surface_id=surface_id,
                    subgroup_type="condition",
                    subgroup_id=str(condition),
                    surface_dir=surface_dir,
                    subgroup_dir=condition_dir,
                )
            )
        surface_df = (
            pd.concat(surface_frames, ignore_index=True)
            if surface_frames
            else pd.DataFrame()
        )
        intact_summary = next(
            (record for record in surface_condition_records if record["signaling_condition"] == "intact"),
            None,
        )
        for record in surface_condition_records:
            delta_fields = (
                ("survival_rate",) if surface_id == "double_cliff" else (
                    "success_rate",
                    "fall_rate",
                    "mean_fitness",
                    "mean_max_cliff_protrusion",
                    "mean_time_to_arrest",
                )
            )
            record["delta_from_intact"] = {
                metric: _relative_delta(
                    record[metric]["mean"] if isinstance(record.get(metric), dict) else record.get(metric),
                    None
                    if intact_summary is None
                    else (
                        intact_summary[metric]["mean"]
                        if isinstance(intact_summary.get(metric), dict)
                        else intact_summary.get(metric)
                    ),
                )
                for metric in delta_fields
            }
        cross_surface_records.extend(dict(record) for record in surface_condition_records)
        surface_summary = {
            "surface_id": surface_id,
            "surface_label": _surface_label(surface_id),
            "condition_summaries": surface_condition_records,
            "condition_count": int(len(surface_condition_records)),
            "max_steps_by_pair": max_steps_by_pair,
            "requested_trials_per_side": (
                None if surface_id == "double_cliff" else requested_trials_per_side
            ),
            "effective_trials_per_side": (
                None if surface_id == "double_cliff" else int(effective_trials_per_side)
            ),
            "requested_trials_per_width": (
                requested_trials_per_width if surface_id == "double_cliff" else None
            ),
            "effective_trials_per_width": (
                int(effective_trials_per_width) if surface_id == "double_cliff" else None
            ),
            "deterministic_rollout": bool(
                effective_trials_per_width == 1 if surface_id == "double_cliff" else effective_trials_per_side == 1
            ),
        }
        surface_charts_dir = surface_dir / "charts"
        surface_charts_dir.mkdir(parents=True, exist_ok=True)
        if surface_id == "double_cliff":
            _chart_double_cliff_surface_family(
                charts_dir=surface_charts_dir,
                title_prefix=f"{_surface_label(surface_id)} Signaling",
                category_field="condition_label",
                records=surface_condition_records,
            )
        else:
            _chart_native_surface_family(
                charts_dir=surface_charts_dir,
                title_prefix=f"{_surface_label(surface_id)} Signaling",
                category_field="condition_label",
                records=surface_condition_records,
            )
        _finalize_assay_outputs(
            assay_dir=surface_dir,
            manifest={
                "schema_version": _POST_RUN_SCHEMA_VERSION,
                "kind": "sensing_edges_post_run_assay_manifest",
                "assay_family": "signaling_ablation",
                "surface_id": surface_id,
                "surface_label": _surface_label(surface_id),
                "specimen_id": specimen_id,
                "run_id": run_id,
                "conditions": list(signaling_conditions),
                "pairs": (
                    list(post_run_cfg.native_pairs)
                    if surface_id == "native_core"
                    else list(post_run_cfg.long_distance_pairs)
                    if surface_id == "long_distance_generalization"
                    else None
                ),
                "widths": (
                    [int(width) for width in list(post_run_cfg.double_cliff.widths)]
                    if surface_id == "double_cliff"
                    else None
                ),
                "max_steps": (
                    _double_cliff_max_steps(cfg)
                    if surface_id == "double_cliff"
                    else None
                ),
                "max_steps_by_pair": max_steps_by_pair if surface_id != "double_cliff" else None,
                "requested_trials_per_side": (
                    None if surface_id == "double_cliff" else requested_trials_per_side
                ),
                "effective_trials_per_side": (
                    None if surface_id == "double_cliff" else int(effective_trials_per_side)
                ),
                "requested_trials_per_width": (
                    requested_trials_per_width if surface_id == "double_cliff" else None
                ),
                "effective_trials_per_width": (
                    int(effective_trials_per_width) if surface_id == "double_cliff" else None
                ),
                "deterministic_rollout": bool(
                    effective_trials_per_width == 1 if surface_id == "double_cliff" else effective_trials_per_side == 1
                ),
            },
            rows=surface_df,
            aggregate={
                "schema_version": _POST_RUN_SCHEMA_VERSION,
                "kind": "sensing_edges_post_run_signaling_ablation_surface",
                "specimen_id": specimen_id,
                **surface_summary,
            },
            condition_records=surface_condition_records,
            specimen_records=[
                {
                    "specimen_id": specimen_id,
                    "assay_family": "signaling_ablation",
                    "surface_id": surface_id,
                    "surface_label": _surface_label(surface_id),
                    "condition_count": int(len(surface_condition_records)),
                    "trial_count": int(len(surface_df)),
                }
            ],
            inventory_records=[
                record for record in inventory_records if record["surface_id"] == surface_id
            ],
        )
        all_surface_frames.append(surface_df)
        surface_summaries.append(surface_summary)
    df = pd.concat(all_surface_frames, ignore_index=True) if all_surface_frames else pd.DataFrame()
    overall_condition_records = _overall_multi_surface_group_records(
        df,
        specimen_id=specimen_id,
        assay_family="signaling_ablation",
        group_field="condition_id",
        group_label_field="condition_label",
    )
    _chart_overall_multi_surface_family(
        charts_dir=assay_dir / "charts",
        title_prefix="Signaling Overall",
        category_field="condition_label",
        records=overall_condition_records,
    )
    aggregate = {
        "schema_version": _POST_RUN_SCHEMA_VERSION,
        "kind": "sensing_edges_post_run_signaling_ablation",
        "specimen_id": specimen_id,
        "conditions": list(signaling_conditions),
        "surface_summaries": surface_summaries,
        "cross_surface_condition_summaries": cross_surface_records,
        "overall_condition_summaries": overall_condition_records,
    }
    _finalize_assay_outputs(
        assay_dir=assay_dir,
        manifest={
            "schema_version": _POST_RUN_SCHEMA_VERSION,
            "kind": "sensing_edges_post_run_assay_manifest",
            "assay_family": "signaling_ablation",
            "specimen_id": specimen_id,
            "run_id": run_id,
            "conditions": list(signaling_conditions),
            "surfaces": list(configured_surfaces),
        },
        rows=df,
        aggregate=aggregate,
        condition_records=overall_condition_records,
        specimen_records=[
            {
                "specimen_id": specimen_id,
                "assay_family": "signaling_ablation",
                "surface_count": int(len(surface_summaries)),
                "condition_count": int(len(cross_surface_records)),
                "trial_count": int(len(df)),
            }
        ],
        inventory_records=inventory_records,
        cross_surface_records=cross_surface_records,
    )
    return df, aggregate, assay_dir


def _run_deformation_ablation_assay(
    *,
    cfg: DictConfig,
    runtime_resolution,
    specimen_dir: Path,
    specimen_id: str,
    genotype: np.ndarray,
    selection_row: dict[str, Any],
    run_id: str,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    del selection_row
    post_run_cfg = cfg.eval.post_run
    assay_dir = specimen_dir / "deformation_ablation"
    deformation_conditions = [str(condition) for condition in list(post_run_cfg.deformation_conditions)]
    all_surface_frames: list[pd.DataFrame] = []
    surface_summaries: list[dict[str, Any]] = []
    cross_surface_records: list[dict[str, Any]] = []
    inventory_records: list[dict[str, Any]] = []
    configured_surfaces = _configured_surface_ids(cfg, config_key="deformation_surfaces")
    for surface_id in configured_surfaces:
        surface_dir = assay_dir / "surfaces" / surface_id
        condition_root = surface_dir / "conditions"
        surface_frames: list[pd.DataFrame] = []
        surface_condition_records: list[dict[str, Any]] = []
        max_steps_by_pair: list[dict[str, Any]] = []
        requested_trials_per_side = int(post_run_cfg.trials_per_side)
        requested_trials_per_width = int(post_run_cfg.double_cliff.trials_per_width)
        condition_effective_trials_per_side: dict[str, int] = {}
        condition_effective_trials_per_width: dict[str, int] = {}
        condition_deterministic_rollout: dict[str, bool] = {}
        for condition in deformation_conditions:
            condition_dir = condition_root / str(condition)
            condition_render_dir = (
                condition_dir / "renders"
                if _should_render_deformation_condition(cfg, str(condition))
                else None
            )
            effective_trials_per_side = _effective_trials_per_side(
                cfg,
                deformation_condition=str(condition),
            )
            effective_trials_per_width = _effective_trials_per_width(
                cfg,
                deformation_condition=str(condition),
            )
            subgroup_started = time.monotonic()
            log.info(
                "Post-run subgroup started | specimen_id=%s assay=deformation_ablation surface=%s condition=%s render=%s effective_trials_side=%d effective_trials_width=%d",
                specimen_id,
                surface_id,
                condition,
                bool(condition_render_dir is not None),
                int(effective_trials_per_side),
                int(effective_trials_per_width),
            )
            condition_effective_trials_per_side[str(condition)] = int(effective_trials_per_side)
            condition_effective_trials_per_width[str(condition)] = int(effective_trials_per_width)
            condition_deterministic_rollout[str(condition)] = bool(
                effective_trials_per_width == 1 if surface_id == "double_cliff" else effective_trials_per_side == 1
            )
            if surface_id == "double_cliff":
                phase_cfg = _double_cliff_phase_cfg(
                    cfg,
                    active_deformation_condition=str(condition),
                )
                entries = _build_double_cliff_entries(
                    phase_cfg,
                    [int(width) for width in list(post_run_cfg.double_cliff.widths)],
                    plateau_width=int(post_run_cfg.double_cliff.plateau_width),
                    trials_per_width=int(effective_trials_per_width),
                )
                episode_results, traces, envs, fitnesses = _run_trace_eval(
                    phase_cfg=phase_cfg,
                    runtime_resolution=runtime_resolution,
                    genotype=genotype,
                    entries=entries,
                    workload="trace",
                    seed=_stable_seed(run_id, specimen_id, "deformation", surface_id, condition),
                )
                condition_df = pd.DataFrame(
                    _double_cliff_rows(
                        episode_results=episode_results,
                        traces=traces,
                        envs=envs,
                        entries=entries,
                        specimen_id=specimen_id,
                        run_id=run_id,
                        max_steps=_double_cliff_max_steps(cfg),
                    )
                )
                condition_df["fell"] = condition_df["fell_left"].astype(bool) | condition_df["fell_right"].astype(bool)
                condition_df["surface_id"] = surface_id
                condition_df["surface_label"] = _surface_label(surface_id)
                condition_df["surface_max_steps"] = _double_cliff_max_steps(cfg)
                condition_df["deformation_condition"] = str(condition)
                condition_df["condition_id"] = str(condition)
                condition_df["condition_label"] = _deformation_label(str(condition))
                if condition_render_dir is not None:
                    _render_selected_candidates(
                        cfg=cfg,
                        output_dir=condition_render_dir,
                        candidates=_double_cliff_render_candidates(
                            episode_results=episode_results,
                            traces=traces,
                            envs=envs,
                            entries=entries,
                            fitnesses=fitnesses,
                        ),
                        manifest_kind="sensing_edges_post_run_render_manifest",
                    )
                condition_record = _double_cliff_surface_specimen_record(
                    specimen_id=specimen_id,
                    assay_family="deformation_ablation",
                    surface_id=surface_id,
                    rows=condition_df,
                    extra_fields={
                        "deformation_condition": str(condition),
                        "condition_id": str(condition),
                        "condition_label": _deformation_label(str(condition)),
                        "deformation_metadata": _DEFORMATION_METADATA.get(str(condition), {}),
                        "deformation_channels": ["abs_h", "abs_v", "comp_h", "comp_v"],
                        "max_steps": _double_cliff_max_steps(cfg),
                        "requested_trials_per_width": requested_trials_per_width,
                        "effective_trials_per_width": int(effective_trials_per_width),
                        "deterministic_rollout": bool(effective_trials_per_width == 1),
                        "render_enabled": bool(condition_render_dir is not None),
                    },
                )
            else:
                condition_df, pair_summaries, pair_budgets, condition_record = _run_native_like_surface(
                    cfg=cfg,
                    runtime_resolution=runtime_resolution,
                    genotype=genotype,
                    run_id=run_id,
                    specimen_id=specimen_id,
                    assay_family="deformation_ablation",
                    surface_id=surface_id,
                    pairs=(
                        list(post_run_cfg.native_pairs)
                        if surface_id == "native_core"
                        else list(post_run_cfg.long_distance_pairs)
                    ),
                    trials_per_side=int(effective_trials_per_side),
                    deformation_condition=str(condition),
                    render_output_dir=condition_render_dir,
                    trial_id_prefix=f"deformation_{surface_id}_{condition}",
                    render_metadata={"deformation_condition": str(condition)},
                )
                max_steps_by_pair.extend(pair_budgets)
                condition_record.update(
                    {
                        "deformation_condition": str(condition),
                        "condition_id": str(condition),
                        "condition_label": _deformation_label(str(condition)),
                        "distance_pair_summaries": pair_summaries,
                        "deformation_metadata": _DEFORMATION_METADATA.get(str(condition), {}),
                        "deformation_channels": ["abs_h", "abs_v", "comp_h", "comp_v"],
                        "requested_trials_per_side": requested_trials_per_side,
                        "effective_trials_per_side": int(effective_trials_per_side),
                        "deterministic_rollout": bool(effective_trials_per_side == 1),
                        "render_enabled": bool(condition_render_dir is not None),
                    }
                )
            _write_json(condition_dir / "summary.json", condition_record)
            _write_json(condition_dir / "aggregate.json", condition_record)
            surface_frames.append(condition_df)
            surface_condition_records.append(condition_record)
            log.info(
                "Post-run subgroup completed | specimen_id=%s assay=deformation_ablation surface=%s condition=%s trial_count=%d elapsed_s=%.2f",
                specimen_id,
                surface_id,
                condition,
                len(condition_df),
                time.monotonic() - subgroup_started,
            )
            inventory_records.append(
                _surface_inventory_record(
                    assay_family="deformation_ablation",
                    surface_id=surface_id,
                    subgroup_type="condition",
                    subgroup_id=str(condition),
                    surface_dir=surface_dir,
                    subgroup_dir=condition_dir,
                )
            )
        intact_summary = next(
            (record for record in surface_condition_records if record["deformation_condition"] == "intact"),
            None,
        )
        off_summary = next(
            (record for record in surface_condition_records if record["deformation_condition"] == "deformation_off"),
            None,
        )
        for record in surface_condition_records:
            delta_fields = (
                ("survival_rate",) if surface_id == "double_cliff" else (
                    "success_rate",
                    "fall_rate",
                    "mean_fitness",
                    "mean_max_cliff_protrusion",
                    "mean_time_to_arrest",
                )
            )
            record["delta_from_intact"] = {
                metric: _relative_delta(
                    record[metric]["mean"] if isinstance(record.get(metric), dict) else record.get(metric),
                    None
                    if intact_summary is None
                    else (
                        intact_summary[metric]["mean"]
                        if isinstance(intact_summary.get(metric), dict)
                        else intact_summary.get(metric)
                    ),
                )
                for metric in delta_fields
            }
            record["delta_from_deformation_off"] = {
                metric: _relative_delta(
                    record[metric]["mean"] if isinstance(record.get(metric), dict) else record.get(metric),
                    None
                    if off_summary is None
                    else (
                        off_summary[metric]["mean"]
                        if isinstance(off_summary.get(metric), dict)
                        else off_summary.get(metric)
                    ),
                )
                for metric in delta_fields
            }
        cross_surface_records.extend(dict(record) for record in surface_condition_records)
        surface_df = (
            pd.concat(surface_frames, ignore_index=True)
            if surface_frames
            else pd.DataFrame()
        )
        surface_summary = {
            "surface_id": surface_id,
            "surface_label": _surface_label(surface_id),
            "condition_summaries": surface_condition_records,
            "condition_count": int(len(surface_condition_records)),
            "max_steps_by_pair": max_steps_by_pair,
            "requested_trials_per_side": (
                None if surface_id == "double_cliff" else requested_trials_per_side
            ),
            "requested_trials_per_width": (
                requested_trials_per_width if surface_id == "double_cliff" else None
            ),
            "condition_effective_trials_per_side": (
                None if surface_id == "double_cliff" else condition_effective_trials_per_side
            ),
            "condition_effective_trials_per_width": (
                condition_effective_trials_per_width if surface_id == "double_cliff" else None
            ),
            "condition_deterministic_rollout": condition_deterministic_rollout,
        }
        surface_charts_dir = surface_dir / "charts"
        surface_charts_dir.mkdir(parents=True, exist_ok=True)
        if surface_id == "double_cliff":
            _chart_double_cliff_surface_family(
                charts_dir=surface_charts_dir,
                title_prefix=f"{_surface_label(surface_id)} Deformation",
                category_field="condition_label",
                records=surface_condition_records,
            )
        else:
            _chart_native_surface_family(
                charts_dir=surface_charts_dir,
                title_prefix=f"{_surface_label(surface_id)} Deformation",
                category_field="condition_label",
                records=surface_condition_records,
            )
        _finalize_assay_outputs(
            assay_dir=surface_dir,
            manifest={
                "schema_version": _POST_RUN_SCHEMA_VERSION,
                "kind": "sensing_edges_post_run_assay_manifest",
                "assay_family": "deformation_ablation",
                "surface_id": surface_id,
                "surface_label": _surface_label(surface_id),
                "specimen_id": specimen_id,
                "run_id": run_id,
                "conditions": deformation_conditions,
                "pairs": (
                    list(post_run_cfg.native_pairs)
                    if surface_id == "native_core"
                    else list(post_run_cfg.long_distance_pairs)
                    if surface_id == "long_distance_generalization"
                    else None
                ),
                "widths": (
                    [int(width) for width in list(post_run_cfg.double_cliff.widths)]
                    if surface_id == "double_cliff"
                    else None
                ),
                "max_steps": (
                    _double_cliff_max_steps(cfg)
                    if surface_id == "double_cliff"
                    else None
                ),
                "max_steps_by_pair": max_steps_by_pair if surface_id != "double_cliff" else None,
                "deformation_channels": ["abs_h", "abs_v", "comp_h", "comp_v"],
                "requested_trials_per_side": (
                    None if surface_id == "double_cliff" else requested_trials_per_side
                ),
                "requested_trials_per_width": (
                    requested_trials_per_width if surface_id == "double_cliff" else None
                ),
                "condition_effective_trials_per_side": (
                    None if surface_id == "double_cliff" else condition_effective_trials_per_side
                ),
                "condition_effective_trials_per_width": (
                    condition_effective_trials_per_width if surface_id == "double_cliff" else None
                ),
                "condition_deterministic_rollout": condition_deterministic_rollout,
            },
            rows=surface_df,
            aggregate={
                "schema_version": _POST_RUN_SCHEMA_VERSION,
                "kind": "sensing_edges_post_run_deformation_ablation_surface",
                "specimen_id": specimen_id,
                **surface_summary,
            },
            condition_records=surface_condition_records,
            specimen_records=[
                {
                    "specimen_id": specimen_id,
                    "assay_family": "deformation_ablation",
                    "surface_id": surface_id,
                    "surface_label": _surface_label(surface_id),
                    "condition_count": int(len(surface_condition_records)),
                    "trial_count": int(len(surface_df)),
                }
            ],
            inventory_records=[
                record for record in inventory_records if record["surface_id"] == surface_id
            ],
        )
        all_surface_frames.append(surface_df)
        surface_summaries.append(surface_summary)
    df = pd.concat(all_surface_frames, ignore_index=True) if all_surface_frames else pd.DataFrame()
    overall_condition_records = _overall_multi_surface_group_records(
        df,
        specimen_id=specimen_id,
        assay_family="deformation_ablation",
        group_field="condition_id",
        group_label_field="condition_label",
    )
    _chart_overall_multi_surface_family(
        charts_dir=assay_dir / "charts",
        title_prefix="Deformation Overall",
        category_field="condition_label",
        records=overall_condition_records,
    )
    aggregate = {
        "schema_version": _POST_RUN_SCHEMA_VERSION,
        "kind": "sensing_edges_post_run_deformation_ablation",
        "specimen_id": specimen_id,
        "conditions": deformation_conditions,
        "surface_summaries": surface_summaries,
        "cross_surface_condition_summaries": cross_surface_records,
        "overall_condition_summaries": overall_condition_records,
    }
    _finalize_assay_outputs(
        assay_dir=assay_dir,
        manifest={
            "schema_version": _POST_RUN_SCHEMA_VERSION,
            "kind": "sensing_edges_post_run_assay_manifest",
            "assay_family": "deformation_ablation",
            "specimen_id": specimen_id,
            "run_id": run_id,
            "conditions": deformation_conditions,
            "surfaces": list(configured_surfaces),
            "deformation_channels": ["abs_h", "abs_v", "comp_h", "comp_v"],
        },
        rows=df,
        aggregate=aggregate,
        condition_records=overall_condition_records,
        specimen_records=[
            {
                "specimen_id": specimen_id,
                "assay_family": "deformation_ablation",
                "surface_count": int(len(surface_summaries)),
                "condition_count": int(len(cross_surface_records)),
                "trial_count": int(len(df)),
            }
        ],
        inventory_records=inventory_records,
        cross_surface_records=cross_surface_records,
    )
    return df, aggregate, assay_dir


def _run_frozen_sensing_control_assay(
    *,
    cfg: DictConfig,
    runtime_resolution,
    specimen_dir: Path,
    specimen_id: str,
    genotype: np.ndarray,
    selection_row: dict[str, Any],
    run_id: str,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    del selection_row
    post_run_cfg = cfg.eval.post_run
    assay_dir = specimen_dir / "frozen_sensing_control"
    conditions = [
        str(condition).strip().lower()
        for condition in list(post_run_cfg.get("frozen_sensing_conditions") or [])
        if str(condition).strip()
    ]
    if not conditions:
        conditions = ["deformation_frozen_start", "perception_frozen_start"]
    if any(condition == "none" for condition in conditions):
        raise ValueError("frozen_sensing_conditions should not include 'none'")

    all_frames: list[pd.DataFrame] = []
    condition_records: list[dict[str, Any]] = []
    inventory_records: list[dict[str, Any]] = []
    max_steps_by_pair: list[dict[str, Any]] = []
    requested_trials_per_side = int(post_run_cfg.trials_per_side)
    effective_trials_per_side = _effective_trials_per_side(cfg)

    for condition in conditions:
        condition_dir = assay_dir / "conditions" / condition
        subgroup_started = time.monotonic()
        log.info(
            "Post-run subgroup started | specimen_id=%s assay=frozen_sensing_control condition=%s effective_trials_side=%d",
            specimen_id,
            condition,
            int(effective_trials_per_side),
        )
        condition_df, pair_summaries, pair_budgets, condition_record = _run_native_like_surface(
            cfg=cfg,
            runtime_resolution=runtime_resolution,
            genotype=genotype,
            run_id=run_id,
            specimen_id=specimen_id,
            assay_family="frozen_sensing_control",
            surface_id="native_core",
            pairs=list(post_run_cfg.native_pairs),
            trials_per_side=int(effective_trials_per_side),
            signaling_condition="intact",
            deformation_condition="intact",
            sensing_freeze_mode=str(condition),
            render_output_dir=None,
            trial_id_prefix=f"frozen_sensing_{condition}",
            render_metadata={"sensing_freeze_mode": str(condition)},
        )
        max_steps_by_pair.extend(pair_budgets)
        condition_record.update(
            {
                "sensing_freeze_mode": str(condition),
                "condition_id": str(condition),
                "condition_label": _sensing_freeze_label(str(condition)),
                "sensing_freeze_metadata": _SENSING_FREEZE_METADATA.get(str(condition), {}),
                "distance_pair_summaries": pair_summaries,
                "requested_trials_per_side": requested_trials_per_side,
                "effective_trials_per_side": int(effective_trials_per_side),
                "deterministic_rollout": bool(effective_trials_per_side == 1),
                "render_enabled": False,
            }
        )
        _write_json(condition_dir / "summary.json", condition_record)
        _write_json(condition_dir / "aggregate.json", condition_record)
        all_frames.append(condition_df)
        condition_records.append(condition_record)
        inventory_records.append(
            _surface_inventory_record(
                assay_family="frozen_sensing_control",
                surface_id="native_core",
                subgroup_type="condition",
                subgroup_id=str(condition),
                surface_dir=assay_dir,
                subgroup_dir=condition_dir,
            )
        )
        log.info(
            "Post-run subgroup completed | specimen_id=%s assay=frozen_sensing_control condition=%s trial_count=%d elapsed_s=%.2f",
            specimen_id,
            condition,
            len(condition_df),
            time.monotonic() - subgroup_started,
        )

    df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    charts_dir = assay_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    _chart_native_surface_family(
        charts_dir=charts_dir,
        title_prefix="Frozen Sensing Native",
        category_field="condition_label",
        records=condition_records,
    )
    aggregate = {
        "schema_version": _POST_RUN_SCHEMA_VERSION,
        "kind": "sensing_edges_post_run_frozen_sensing_control",
        "specimen_id": specimen_id,
        "surface_id": "native_core",
        "surface_label": _surface_label("native_core"),
        "conditions": conditions,
        "condition_summaries": condition_records,
        "max_steps_by_pair": max_steps_by_pair,
        "requested_trials_per_side": requested_trials_per_side,
        "effective_trials_per_side": int(effective_trials_per_side),
        "deterministic_rollout": bool(effective_trials_per_side == 1),
    }
    _finalize_assay_outputs(
        assay_dir=assay_dir,
        manifest={
            "schema_version": _POST_RUN_SCHEMA_VERSION,
            "kind": "sensing_edges_post_run_assay_manifest",
            "assay_family": "frozen_sensing_control",
            "specimen_id": specimen_id,
            "run_id": run_id,
            "surface_id": "native_core",
            "surface_label": _surface_label("native_core"),
            "conditions": conditions,
            "pairs": list(post_run_cfg.native_pairs),
            "max_steps_by_pair": max_steps_by_pair,
            "requested_trials_per_side": requested_trials_per_side,
            "effective_trials_per_side": int(effective_trials_per_side),
            "deterministic_rollout": bool(effective_trials_per_side == 1),
        },
        rows=df,
        aggregate=aggregate,
        condition_records=condition_records,
        specimen_records=[
            {
                "specimen_id": specimen_id,
                "assay_family": "frozen_sensing_control",
                "surface_id": "native_core",
                "surface_label": _surface_label("native_core"),
                "condition_count": int(len(condition_records)),
                "trial_count": int(len(df)),
            }
        ],
        inventory_records=inventory_records,
    )
    return df, aggregate, assay_dir


def _run_body_transfer_assay(
    *,
    cfg: DictConfig,
    runtime_resolution,
    specimen_dir: Path,
    specimen_id: str,
    genotype: np.ndarray,
    selection_row: dict[str, Any],
    run_id: str,
    native_rows: pd.DataFrame,
    native_aggregate: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    del selection_row
    post_run_cfg = cfg.eval.post_run
    assay_dir = specimen_dir / "body_transfer"
    reference_body = _reference_body_spec(cfg)
    body_specs = [reference_body] + [dict(body) for body in list(post_run_cfg.body_transfer_bodies)]
    all_surface_frames: list[pd.DataFrame] = []
    surface_summaries: list[dict[str, Any]] = []
    cross_surface_records: list[dict[str, Any]] = []
    inventory_records: list[dict[str, Any]] = []
    configured_surfaces = _configured_surface_ids(cfg, config_key="signaling_surfaces")
    for surface_id in configured_surfaces:
        surface_dir = assay_dir / "surfaces" / surface_id
        body_root = surface_dir / "bodies"
        surface_frames: list[pd.DataFrame] = []
        surface_body_records: list[dict[str, Any]] = []
        max_steps_by_pair: list[dict[str, Any]] = []
        requested_trials_per_side = int(post_run_cfg.trials_per_side)
        requested_trials_per_width = int(post_run_cfg.double_cliff.trials_per_width)
        effective_trials_per_side = _effective_trials_per_side(cfg)
        effective_trials_per_width = _effective_trials_per_width(cfg)
        for body in body_specs:
            body_label = str(body["label"])
            body_dir = body_root / body_label
            body_render_dir = (
                body_dir / "renders"
                if _should_render_body_label(
                    cfg,
                    body_label=body_label,
                    reference_body_label=str(reference_body["label"]),
                )
                else None
            )
            subgroup_started = time.monotonic()
            log.info(
                "Post-run subgroup started | specimen_id=%s assay=body_transfer surface=%s body=%s render=%s",
                specimen_id,
                surface_id,
                body_label,
                bool(body_render_dir is not None),
            )
            if surface_id == "double_cliff":
                phase_cfg = _double_cliff_phase_cfg(
                    cfg,
                    body_label=body_label,
                    robot_w=int(body["robot_w"]),
                    robot_h=int(body["robot_h"]),
                    body_shape=str(body.get("shape", "rectangle")),
                )
                entries = _build_double_cliff_entries(
                    phase_cfg,
                    [int(width) for width in list(post_run_cfg.double_cliff.widths)],
                    plateau_width=int(post_run_cfg.double_cliff.plateau_width),
                    trials_per_width=int(effective_trials_per_width),
                )
                episode_results, traces, envs, fitnesses = _run_trace_eval(
                    phase_cfg=phase_cfg,
                    runtime_resolution=runtime_resolution,
                    genotype=genotype,
                    entries=entries,
                    workload="trace",
                    seed=_stable_seed(run_id, specimen_id, "body_transfer", surface_id, body_label),
                )
                body_df = pd.DataFrame(
                    _double_cliff_rows(
                        episode_results=episode_results,
                        traces=traces,
                        envs=envs,
                        entries=entries,
                        specimen_id=specimen_id,
                        run_id=run_id,
                        max_steps=_double_cliff_max_steps(cfg),
                    )
                )
                body_df["fell"] = body_df["fell_left"].astype(bool) | body_df["fell_right"].astype(bool)
                body_df["surface_id"] = surface_id
                body_df["surface_label"] = _surface_label(surface_id)
                body_df["surface_max_steps"] = _double_cliff_max_steps(cfg)
                body_df["body"] = body_label
                body_df["condition_id"] = body_label
                body_df["condition_label"] = _body_label(body_label)
                if body_render_dir is not None:
                    _render_selected_candidates(
                        cfg=cfg,
                        output_dir=body_render_dir,
                        candidates=_double_cliff_render_candidates(
                            episode_results=episode_results,
                            traces=traces,
                            envs=envs,
                            entries=entries,
                            fitnesses=fitnesses,
                        ),
                        manifest_kind="sensing_edges_post_run_render_manifest",
                    )
                body_record = _double_cliff_surface_specimen_record(
                    specimen_id=specimen_id,
                    assay_family="body_transfer",
                    surface_id=surface_id,
                    rows=body_df,
                    extra_fields={
                        "body": body_label,
                        "condition_id": body_label,
                        "condition_label": _body_label(body_label),
                        "reference": bool(body_label == reference_body["label"]),
                        "max_steps": _double_cliff_max_steps(cfg),
                        "requested_trials_per_width": requested_trials_per_width,
                        "effective_trials_per_width": int(effective_trials_per_width),
                        "deterministic_rollout": bool(effective_trials_per_width == 1),
                        "render_enabled": bool(body_render_dir is not None),
                    },
                )
            else:
                body_df, pair_summaries, pair_budgets, body_record = _run_native_like_surface(
                    cfg=cfg,
                    runtime_resolution=runtime_resolution,
                    genotype=genotype,
                    run_id=run_id,
                    specimen_id=specimen_id,
                    assay_family="body_transfer",
                    surface_id=surface_id,
                    pairs=(
                        list(post_run_cfg.native_pairs)
                        if surface_id == "native_core"
                        else list(post_run_cfg.long_distance_pairs)
                    ),
                    trials_per_side=int(effective_trials_per_side),
                    body_label=body_label,
                    robot_w=int(body["robot_w"]),
                    robot_h=int(body["robot_h"]),
                    body_shape=str(body.get("shape", "rectangle")),
                    render_output_dir=body_render_dir,
                    trial_id_prefix=f"body_transfer_{surface_id}_{body_label}",
                    render_metadata={"body_label": body_label},
                )
                max_steps_by_pair.extend(pair_budgets)
                body_record.update(
                    {
                        "body": body_label,
                        "condition_id": body_label,
                        "condition_label": _body_label(body_label),
                        "reference": bool(body_label == reference_body["label"]),
                        "distance_pair_summaries": pair_summaries,
                        "requested_trials_per_side": requested_trials_per_side,
                        "effective_trials_per_side": int(effective_trials_per_side),
                        "deterministic_rollout": bool(effective_trials_per_side == 1),
                        "render_enabled": bool(body_render_dir is not None),
                    }
                )
            _write_json(body_dir / "summary.json", body_record)
            _write_json(body_dir / "aggregate.json", body_record)
            surface_frames.append(body_df)
            surface_body_records.append(body_record)
            log.info(
                "Post-run subgroup completed | specimen_id=%s assay=body_transfer surface=%s body=%s trial_count=%d elapsed_s=%.2f",
                specimen_id,
                surface_id,
                body_label,
                len(body_df),
                time.monotonic() - subgroup_started,
            )
            inventory_records.append(
                _surface_inventory_record(
                    assay_family="body_transfer",
                    surface_id=surface_id,
                    subgroup_type="body",
                    subgroup_id=body_label,
                    surface_dir=surface_dir,
                    subgroup_dir=body_dir,
                )
            )
        reference_record = next(
            (record for record in surface_body_records if bool(record.get("reference"))),
            None,
        )
        baseline = None
        if reference_record is not None:
            baseline_metric = reference_record.get("survival_rate") if surface_id == "double_cliff" else reference_record.get("success_rate")
            baseline = baseline_metric["mean"] if isinstance(baseline_metric, dict) else baseline_metric
        for record in surface_body_records:
            metric = record.get("survival_rate") if surface_id == "double_cliff" else record.get("success_rate")
            metric_value = metric["mean"] if isinstance(metric, dict) else metric
            record["transfer_ratio"] = None if baseline in {None, 0.0} else float(metric_value) / float(baseline)
        cross_surface_records.extend(dict(record) for record in surface_body_records)
        surface_df = pd.concat(surface_frames, ignore_index=True) if surface_frames else pd.DataFrame()
        non_reference_ratios = [
            float(record["transfer_ratio"])
            for record in surface_body_records
            if record.get("transfer_ratio") is not None and not bool(record.get("reference"))
        ]
        surface_summary = {
            "surface_id": surface_id,
            "surface_label": _surface_label(surface_id),
            "body_summaries": surface_body_records,
            "body_count": int(len(surface_body_records)),
            "max_steps_by_pair": max_steps_by_pair,
            "reference_body": reference_body["label"],
            "requested_trials_per_side": (
                None if surface_id == "double_cliff" else requested_trials_per_side
            ),
            "effective_trials_per_side": (
                None if surface_id == "double_cliff" else int(effective_trials_per_side)
            ),
            "requested_trials_per_width": (
                requested_trials_per_width if surface_id == "double_cliff" else None
            ),
            "effective_trials_per_width": (
                int(effective_trials_per_width) if surface_id == "double_cliff" else None
            ),
            "deterministic_rollout": bool(
                effective_trials_per_width == 1 if surface_id == "double_cliff" else effective_trials_per_side == 1
            ),
            "body_robustness_score": (
                float(np.mean(non_reference_ratios)) if non_reference_ratios else None
            ),
        }
        surface_charts_dir = surface_dir / "charts"
        surface_charts_dir.mkdir(parents=True, exist_ok=True)
        if surface_id == "double_cliff":
            _chart_double_cliff_surface_family(
                charts_dir=surface_charts_dir,
                title_prefix=f"{_surface_label(surface_id)} Body Transfer",
                category_field="condition_label",
                records=surface_body_records,
            )
        else:
            _chart_native_surface_family(
                charts_dir=surface_charts_dir,
                title_prefix=f"{_surface_label(surface_id)} Body Transfer",
                category_field="condition_label",
                records=surface_body_records,
            )
        _chart_bar(
            categories=[record["condition_label"] for record in surface_body_records],
            values=[float(record["transfer_ratio"] or 0.0) for record in surface_body_records],
            ci_bounds=None,
            title=f"{_surface_label(surface_id)} Transfer Ratio",
            ylabel="ratio",
            output_path=surface_charts_dir / "transfer_ratio_by_body.png",
        )
        _finalize_assay_outputs(
            assay_dir=surface_dir,
            manifest={
                "schema_version": _POST_RUN_SCHEMA_VERSION,
                "kind": "sensing_edges_post_run_assay_manifest",
                "assay_family": "body_transfer",
                "surface_id": surface_id,
                "surface_label": _surface_label(surface_id),
                "specimen_id": specimen_id,
                "run_id": run_id,
                "bodies": body_specs,
                "reference_body": reference_body,
                "pairs": (
                    list(post_run_cfg.native_pairs)
                    if surface_id == "native_core"
                    else list(post_run_cfg.long_distance_pairs)
                    if surface_id == "long_distance_generalization"
                    else None
                ),
                "widths": (
                    [int(width) for width in list(post_run_cfg.double_cliff.widths)]
                    if surface_id == "double_cliff"
                    else None
                ),
                "max_steps": (
                    _double_cliff_max_steps(cfg)
                    if surface_id == "double_cliff"
                    else None
                ),
                "max_steps_by_pair": max_steps_by_pair if surface_id != "double_cliff" else None,
                "requested_trials_per_side": (
                    None if surface_id == "double_cliff" else requested_trials_per_side
                ),
                "effective_trials_per_side": (
                    None if surface_id == "double_cliff" else int(effective_trials_per_side)
                ),
                "requested_trials_per_width": (
                    requested_trials_per_width if surface_id == "double_cliff" else None
                ),
                "effective_trials_per_width": (
                    int(effective_trials_per_width) if surface_id == "double_cliff" else None
                ),
                "deterministic_rollout": bool(
                    effective_trials_per_width == 1 if surface_id == "double_cliff" else effective_trials_per_side == 1
                ),
            },
            rows=surface_df,
            aggregate={
                "schema_version": _POST_RUN_SCHEMA_VERSION,
                "kind": "sensing_edges_post_run_body_transfer_surface",
                "specimen_id": specimen_id,
                **surface_summary,
            },
            condition_records=surface_body_records,
            specimen_records=[
                {
                    "specimen_id": specimen_id,
                    "assay_family": "body_transfer",
                    "surface_id": surface_id,
                    "surface_label": _surface_label(surface_id),
                    "body_count": int(len(surface_body_records)),
                    "trial_count": int(len(surface_df)),
                    "body_robustness_score": surface_summary["body_robustness_score"],
                }
            ],
            inventory_records=[
                record for record in inventory_records if record["surface_id"] == surface_id
            ],
        )
        all_surface_frames.append(surface_df)
        surface_summaries.append(surface_summary)
    df = pd.concat(all_surface_frames, ignore_index=True) if all_surface_frames else pd.DataFrame()
    overall_body_records = _overall_multi_surface_group_records(
        df,
        specimen_id=specimen_id,
        assay_family="body_transfer",
        group_field="body",
        group_label_field="condition_label",
    )
    _chart_overall_multi_surface_family(
        charts_dir=assay_dir / "charts",
        title_prefix="Body Transfer Overall",
        category_field="condition_label",
        records=overall_body_records,
    )
    aggregate = {
        "schema_version": _POST_RUN_SCHEMA_VERSION,
        "kind": "sensing_edges_post_run_body_transfer",
        "specimen_id": specimen_id,
        "surface_summaries": surface_summaries,
        "cross_surface_condition_summaries": cross_surface_records,
        "overall_body_summaries": overall_body_records,
        "reference_body": reference_body["label"],
        "native_summary": native_aggregate["specimen_summary"],
        "native_trial_count": int(len(native_rows)),
    }
    _finalize_assay_outputs(
        assay_dir=assay_dir,
        manifest={
            "schema_version": _POST_RUN_SCHEMA_VERSION,
            "kind": "sensing_edges_post_run_assay_manifest",
            "assay_family": "body_transfer",
            "specimen_id": specimen_id,
            "run_id": run_id,
            "bodies": body_specs,
            "reference_body": reference_body,
            "surfaces": list(_group_surface_ids()),
        },
        rows=df,
        aggregate=aggregate,
        condition_records=overall_body_records,
        specimen_records=[
            {
                "specimen_id": specimen_id,
                "assay_family": "body_transfer",
                "surface_count": int(len(surface_summaries)),
                "body_count": int(len(cross_surface_records)),
                "trial_count": int(len(df)),
            }
        ],
        inventory_records=inventory_records,
        cross_surface_records=cross_surface_records,
    )
    return df, aggregate, assay_dir


def _spring_lengths_np(
    positions: np.ndarray,
    *,
    a_idx: np.ndarray,
    b_idx: np.ndarray,
) -> np.ndarray:
    pos_a = np.asarray(positions, dtype=np.float32)[:, np.asarray(a_idx, dtype=np.int32)]
    pos_b = np.asarray(positions, dtype=np.float32)[:, np.asarray(b_idx, dtype=np.int32)]
    return np.sqrt(np.sum((pos_a - pos_b) ** 2, axis=-1) + 1.0e-12).astype(np.float32)


def _npz_safe_trial_id(trial_id: str) -> str:
    return str(trial_id).replace("/", "_").replace(" ", "_")


def _configured_state_space_surfaces(cfg: DictConfig) -> tuple[str, ...]:
    raw = cfg.eval.post_run.get("state_space_surfaces", ["native_core"])
    raw_items = [raw] if isinstance(raw, str) else list(raw or ["native_core"])
    expanded: list[str] = []
    alias_map = {
        "extended": "long_distance_generalization",
        "symmetric": "double_cliff",
        "symmetrical": "double_cliff",
    }
    for item in raw_items:
        surface = str(item).strip()
        if not surface:
            continue
        if surface == "all":
            expanded.extend(["native_core", "long_distance_generalization", "double_cliff"])
            continue
        expanded.append(alias_map.get(surface, surface))
    allowed = {"native_core", "long_distance_generalization", "double_cliff"}
    unknown = [surface for surface in expanded if surface not in allowed]
    if unknown:
        raise ValueError(f"Unsupported state-space trace surfaces: {unknown}")
    return tuple(dict.fromkeys(expanded or ["native_core"]))


def _state_space_trace_groups(
    *,
    cfg: DictConfig,
    active_deformation_condition: str,
    active_signaling_condition: str,
    surfaces: Sequence[str],
) -> list[dict[str, Any]]:
    post_run_cfg = cfg.eval.post_run
    groups: list[dict[str, Any]] = []
    for surface_id in surfaces:
        if surface_id == "native_core":
            phase_cfg = _native_phase_cfg(
                cfg,
                active_deformation_condition=active_deformation_condition,
                active_signaling_condition=active_signaling_condition,
                active_sensing_freeze_mode=str(cfg.eval.post_run.get("active_sensing_freeze_mode", "none")),
            )
            groups.append(
                {
                    "surface_id": "native_core",
                    "phase_cfg": phase_cfg,
                    "entries": _build_native_entries(
                        phase_cfg,
                        list(post_run_cfg.native_pairs),
                        trials_per_side=1,
                        goal_sides=_configured_goal_sides(cfg),
                    ),
                    "max_steps": _native_max_steps(cfg),
                    "budget_records": [
                        {
                            "goal_distance": int(pair["goal_distance"]),
                            "cliff_distance": int(pair["cliff_distance"]),
                            "distance_pair": _pair_label(pair["goal_distance"], pair["cliff_distance"]),
                            "max_steps": _native_max_steps(cfg),
                        }
                        for pair in list(post_run_cfg.native_pairs)
                    ],
                }
            )
        elif surface_id == "long_distance_generalization":
            for group in _native_like_pair_groups(
                cfg=cfg,
                surface_id="long_distance_generalization",
                pairs=list(post_run_cfg.long_distance_pairs),
                trials_per_side=1,
                signaling_condition=active_signaling_condition,
                deformation_condition=active_deformation_condition,
                sensing_freeze_mode=str(cfg.eval.post_run.get("active_sensing_freeze_mode", "none")),
                body_label="rect_7x2",
                robot_w=None,
                robot_h=None,
                body_shape=None,
            ):
                groups.append(
                    {
                        "surface_id": "long_distance_generalization",
                        "phase_cfg": group["phase_cfg"],
                        "entries": list(group["entries"]),
                        "max_steps": int(group["max_steps"]),
                        "budget_records": list(group["budget_records"]),
                    }
                )
        elif surface_id == "double_cliff":
            double_cliff_cfg = post_run_cfg.double_cliff
            phase_cfg = _double_cliff_phase_cfg(
                cfg,
                active_deformation_condition=active_deformation_condition,
                active_signaling_condition=active_signaling_condition,
            )
            widths = [int(width) for width in list(double_cliff_cfg.widths)]
            groups.append(
                {
                    "surface_id": "double_cliff",
                    "phase_cfg": phase_cfg,
                    "entries": _build_double_cliff_entries(
                        phase_cfg,
                        widths,
                        plateau_width=int(double_cliff_cfg.plateau_width),
                        trials_per_width=1,
                    ),
                    "max_steps": _double_cliff_max_steps(cfg),
                    "budget_records": [
                        {"width": int(width), "max_steps": _double_cliff_max_steps(cfg)}
                        for width in widths
                    ],
                }
            )
    if not groups:
        raise ValueError("state-space trace requires at least one surface group")
    return groups


def _state_space_edge_audit_rows(trial_df: pd.DataFrame) -> list[dict[str, Any]]:
    if trial_df.empty:
        return []
    audit_cols = [
        "surface_id",
        "surface_label",
        "trial_id",
        "distance_pair",
        "side",
        "width",
        "deformation_condition",
        "signaling_condition",
        "max_steps",
        "steps_alive",
        "goal_touched",
        "cliff_fell",
        "success",
        "termination_reason",
        "min_cliff_clearance",
        "max_cliff_protrusion",
        "first_cliff_reach_step",
        "returned_from_edge",
        "final_cliff_clearance",
        "min_left_clearance",
        "min_right_clearance",
        "first_left_cliff_reach_step",
        "first_right_cliff_reach_step",
        "side_switch_count",
    ]
    existing = [col for col in audit_cols if col in trial_df.columns]
    return trial_df[existing].to_dict(orient="records")


def _run_state_space_trace_assay(
    *,
    cfg: DictConfig,
    runtime_resolution,
    specimen_dir: Path,
    specimen_id: str,
    genotype: np.ndarray,
    selection_row: dict[str, Any],
    run_id: str,
    assay_name: str = "state_space_trace",
    deformation_condition: str | None = None,
    signaling_condition: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    del selection_row
    post_run_cfg = cfg.eval.post_run
    active_deformation_condition = str(
        deformation_condition
        if deformation_condition is not None
        else post_run_cfg.get("state_space_deformation_condition", "intact")
    )
    active_signaling_condition = str(
        signaling_condition
        if signaling_condition is not None
        else post_run_cfg.get("state_space_signaling_condition", "intact")
    )
    assay_dir = specimen_dir / str(assay_name)
    traces_dir = assay_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    surfaces = _configured_state_space_surfaces(cfg)
    trace_groups = _state_space_trace_groups(
        cfg=cfg,
        active_deformation_condition=active_deformation_condition,
        active_signaling_condition=active_signaling_condition,
        surfaces=surfaces,
    )
    trial_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    trace_files: list[dict[str, Any]] = []
    surface_group_records: list[dict[str, Any]] = []
    env0 = None

    for group_index, group in enumerate(trace_groups):
        surface_id = str(group["surface_id"])
        phase_cfg = group["phase_cfg"]
        entries = list(group["entries"])
        max_steps = int(group["max_steps"])
        seed = (
            _stable_seed(run_id, specimen_id, "state_space_trace", "native_core")
            if len(trace_groups) == 1 and surface_id == "native_core"
            else _stable_seed(run_id, specimen_id, assay_name, surface_id, group_index)
        )
        episode_results, state_traces, envs = _run_state_space_trace_eval(
            phase_cfg=phase_cfg,
            runtime_resolution=runtime_resolution,
            genotype=genotype,
            entries=entries,
            seed=seed,
        )
        if env0 is None and envs:
            env0 = envs[0]
        sense_act_phases = int(phase_cfg.sim.sense_act_phases_per_env_step)
        physics_substeps = int(phase_cfg.sim.physics_substeps)
        nca_steps_per_phase = int(phase_cfg.nca.nca_steps_per_phase)
        surface_group_records.append(
            {
                "surface_id": surface_id,
                "surface_label": _surface_label(surface_id),
                "group_index": int(group_index),
                "trial_count": int(len(entries)),
                "max_steps": int(max_steps),
                "budget_records": list(group["budget_records"]),
                "sense_act_phases_per_env_step": int(sense_act_phases),
                "physics_substeps": int(physics_substeps),
                "nca_steps_per_phase": int(nca_steps_per_phase),
                "seed": int(seed),
            }
        )

        for index, (entry, env_data) in enumerate(zip(entries, envs, strict=True)):
            metadata = dict(entry.metadata or {})
            trial_repeat_index = int(metadata.get("trial_repeat_index", 0))
            width = None
            if surface_id == "double_cliff":
                width = int(metadata.get("assay_width", metadata.get("width", 0)))
                trial_id = f"width_{width}_{trial_repeat_index:03d}"
                side = "double"
                goal_distance = 0
                cliff_distance = 0
                distance_pair = ""
            else:
                side = str(metadata.get("goal_side", metadata.get("assay_goal_side", "unknown")))
                goal_distance = int(metadata.get("goal_distance", metadata.get("assay_goal_distance", 0)))
                if goal_distance <= 0:
                    goal_distance = int(np.asarray(episode_results.goal_distance[index]))
                cliff_distance = int(metadata.get("cliff_distance", metadata.get("assay_cliff_distance", 0)))
                if cliff_distance <= 0:
                    cliff_distance = int(np.asarray(episode_results.cliff_distance[index]))
                distance_pair = _pair_label(goal_distance, cliff_distance)
                trial_id = _controller_trace_trial_id(
                    goal_distance=goal_distance,
                    cliff_distance=cliff_distance,
                    side=side,
                    trial_repeat_index=trial_repeat_index,
                )
            active = np.asarray(state_traces.active[index], dtype=bool)
            active_count = int(np.sum(active))
            if active_count <= 0:
                active = np.asarray(state_traces.env_step[index] >= 0, dtype=bool)
                active_count = int(np.sum(active))
            step_slice = active
            positions = np.asarray(state_traces.post_positions[index], dtype=np.float32)[step_slice]
            pre_positions = np.asarray(state_traces.pre_positions[index], dtype=np.float32)[step_slice]
            velocities = np.asarray(state_traces.post_velocities[index], dtype=np.float32)[step_slice]
            velocities_true = np.asarray(state_traces.post_velocities_true[index], dtype=np.float32)[step_slice]
            pre_velocities = np.asarray(state_traces.pre_velocities[index], dtype=np.float32)[step_slice]
            spring_rest = np.asarray(state_traces.post_spring_rest_length[index], dtype=np.float32)[step_slice]
            pre_spring_rest = np.asarray(state_traces.pre_spring_rest_length[index], dtype=np.float32)[step_slice]
            a_idx = np.asarray(env_data.topology.a_idx, dtype=np.int32)
            b_idx = np.asarray(env_data.topology.b_idx, dtype=np.int32)
            spring_current_length = _spring_lengths_np(positions, a_idx=a_idx, b_idx=b_idx)
            boxel_corners = np.asarray(env_data.collision_data.boxel_corners, dtype=np.int32)
            cell_corner_positions = positions[:, boxel_corners, :] if boxel_corners.size else np.zeros((positions.shape[0], 0, 4, 2), dtype=np.float32)
            npz_name = f"{_npz_safe_trial_id(trial_id)}.npz"
            npz_path = traces_dir / npz_name
            np.savez_compressed(
                npz_path,
                plant_positions=positions,
                plant_pre_positions=pre_positions,
                plant_velocities=velocities,
                plant_pre_velocities=pre_velocities,
                plant_velocities_true=velocities_true,
                cell_corner_positions=cell_corner_positions.astype(np.float32),
                spring_rest_length=spring_rest,
                spring_pre_rest_length=pre_spring_rest,
                spring_init_rest_length=np.asarray(env_data.sim_state.spring_init_rest_length, dtype=np.float32),
                spring_current_length=spring_current_length,
                spring_a_idx=a_idx,
                spring_b_idx=b_idx,
                robot_point_mask=np.asarray(env_data.robot_point_mask, dtype=bool),
                boxel_corners=boxel_corners,
                boxel_mask=np.asarray(env_data.collision_data.boxel_mask, dtype=bool),
                surface_edge_mask=np.asarray(env_data.collision_data.surface_edge_mask, dtype=bool),
                nca_state=np.asarray(state_traces.nca_state[index], dtype=np.float32)[step_slice],
                ctrnn_carry=np.asarray(state_traces.ctrnn_carry[index], dtype=np.float32)[step_slice],
                action_outputs=np.asarray(state_traces.action_outputs[index], dtype=np.float32)[step_slice],
                phase_active=np.asarray(state_traces.phase.active[index], dtype=bool)[step_slice],
                phase_deformation_inputs=np.asarray(state_traces.phase.deformation_inputs[index], dtype=np.float32)[step_slice],
                phase_h_drive=np.asarray(state_traces.phase.h_compact[index], dtype=np.float32)[step_slice],
                phase_v_drive=np.asarray(state_traces.phase.v_compact[index], dtype=np.float32)[step_slice],
                phase_nca_state=np.asarray(state_traces.phase.nca_state[index], dtype=np.float32)[step_slice],
                phase_ctrnn_carry=np.asarray(state_traces.phase.ctrnn_carry[index], dtype=np.float32)[step_slice],
                phase_action_outputs=np.asarray(state_traces.phase.action_outputs[index], dtype=np.float32)[step_slice],
                phase_post_positions=np.asarray(state_traces.phase.post_positions[index], dtype=np.float32)[step_slice],
                phase_post_velocities=np.asarray(state_traces.phase.post_velocities[index], dtype=np.float32)[step_slice],
                phase_post_velocities_true=np.asarray(state_traces.phase.post_velocities_true[index], dtype=np.float32)[step_slice],
                phase_post_spring_rest_length=np.asarray(state_traces.phase.post_spring_rest_length[index], dtype=np.float32)[step_slice],
                phase_controller_hidden_output_max_abs=np.asarray(
                    state_traces.phase.controller_hidden_output_max_abs[index],
                    dtype=np.float32,
                )[step_slice],
                phase_action_output_max_abs=np.asarray(
                    state_traces.phase.action_output_max_abs[index],
                    dtype=np.float32,
                )[step_slice],
            )
            goal_direction = int(np.asarray(episode_results.goal_direction[index])) if surface_id != "double_cliff" else 0
            env_steps = np.asarray(state_traces.env_step[index], dtype=np.int32)[step_slice]
            com_x = np.asarray(state_traces.com_x[index], dtype=np.float32)[step_slice]
            com_y = np.asarray(state_traces.com_y[index], dtype=np.float32)[step_slice]
            min_x = np.asarray(state_traces.min_x[index], dtype=np.float32)[step_slice]
            max_x = np.asarray(state_traces.max_x[index], dtype=np.float32)[step_slice]
            min_y = np.asarray(state_traces.min_y[index], dtype=np.float32)[step_slice]
            max_y = np.asarray(state_traces.max_y[index], dtype=np.float32)[step_slice]
            front_x = np.asarray(state_traces.front_x[index], dtype=np.float32)[step_slice]
            back_x = np.asarray(state_traces.back_x[index], dtype=np.float32)[step_slice]
            done_before = np.asarray(state_traces.done_before[index], dtype=bool)[step_slice]
            done_after = np.asarray(state_traces.done_after[index], dtype=bool)[step_slice]
            init_com_x = float(np.asarray(episode_results.init_com_x[index]))
            if surface_id == "double_cliff":
                left_frontier_x = float(int(metadata["platform_start_x"]) * float(CELL_SIZE))
                right_frontier_x = float((int(metadata["platform_end_x"]) + 1) * float(CELL_SIZE))
                left_clearance = min_x - left_frontier_x
                right_clearance = right_frontier_x - front_x
                nearest_left = left_clearance <= right_clearance
                cliff_clearance = np.minimum(left_clearance, right_clearance).astype(np.float32)
                nearest_cliff_side = np.where(nearest_left, "left", "right")
                cliff_frontier_x = np.where(nearest_left, left_frontier_x, right_frontier_x).astype(np.float32)
                cliff_edge_x = np.where(nearest_left, int(metadata["platform_start_x"]), int(metadata["platform_end_x"]))
                cliff_edge_body_x = np.where(nearest_left, min_x, front_x).astype(np.float32)
                goal_block_x = None
                goal_block_y = None
                goal_contact_x = float("nan")
                goal_distance_remaining = np.full((active_count,), np.nan, dtype=np.float32)
                com_x_to_goal = np.full((active_count,), np.nan, dtype=np.float32)
            else:
                cliff_side_raw = metadata.get("cliff_side", "right" if goal_direction < 0 else "left")
                cliff_side = str(cliff_side_raw).strip().lower()
                cliff_edge_x_scalar = int(
                    metadata.get(
                        "terminal_cliff_edge_x",
                        metadata.get("cliff_edge_x", np.asarray(episode_results.cliff_x[index])),
                    )
                )
                cliff_frontier_scalar = _cliff_frontier_world_x(cliff_edge_x_scalar, cliff_side)
                goal_block_x = int(metadata.get("goal_block_x", np.asarray(episode_results.goal_block_x[index])))
                goal_block_y = int(metadata.get("goal_block_y", np.asarray(episode_results.goal_block_y[index])))
                goal_contact_x = _goal_contact_face_x(goal_block_x, goal_direction)
                goal_edge_x = _goal_facing_edge(back_x, front_x, goal_direction)
                cliff_edge_body_x = _cliff_facing_edge(min_x, front_x, cliff_side)
                if cliff_side == "left":
                    cliff_clearance = (cliff_edge_body_x - cliff_frontier_scalar).astype(np.float32)
                    left_clearance = cliff_clearance
                    right_clearance = np.full((active_count,), np.nan, dtype=np.float32)
                    left_frontier_x = float(cliff_frontier_scalar)
                    right_frontier_x = float("nan")
                else:
                    cliff_clearance = (cliff_frontier_scalar - cliff_edge_body_x).astype(np.float32)
                    left_clearance = np.full((active_count,), np.nan, dtype=np.float32)
                    right_clearance = cliff_clearance
                    left_frontier_x = float("nan")
                    right_frontier_x = float(cliff_frontier_scalar)
                nearest_cliff_side = np.full((active_count,), cliff_side, dtype=object)
                cliff_frontier_x = np.full((active_count,), float(cliff_frontier_scalar), dtype=np.float32)
                cliff_edge_x = np.full((active_count,), int(cliff_edge_x_scalar), dtype=np.int32)
                goal_distance_remaining = np.asarray(goal_direction, dtype=np.float32) * (
                    goal_contact_x - goal_edge_x
                )
                com_x_to_goal = goal_contact_x - com_x
            hit_indices = np.flatnonzero(cliff_clearance <= 0.0)
            first_cliff_reach_step = int(hit_indices[0]) if hit_indices.size else None
            left_hit_indices = np.flatnonzero(left_clearance <= 0.0)
            right_hit_indices = np.flatnonzero(right_clearance <= 0.0)
            first_left_cliff_reach_step = int(left_hit_indices[0]) if left_hit_indices.size else None
            first_right_cliff_reach_step = int(right_hit_indices[0]) if right_hit_indices.size else None
            min_cliff_clearance = float(np.min(cliff_clearance)) if active_count > 0 else None
            final_cliff_clearance = float(cliff_clearance[-1]) if active_count > 0 else None
            max_cliff_protrusion = None if min_cliff_clearance is None else float(max(-min_cliff_clearance, 0.0))
            returned_from_edge = bool(
                first_cliff_reach_step is not None
                and final_cliff_clearance is not None
                and final_cliff_clearance > 0.0
            )
            side_switch_count = None
            if surface_id == "double_cliff" and active_count > 0:
                vicinity = (left_clearance <= _CLIFF_VICINITY_BAND) | (right_clearance <= _CLIFF_VICINITY_BAND)
                active_sides = nearest_cliff_side[vicinity]
                side_switch_count = int(np.sum(active_sides[1:] != active_sides[:-1])) if active_sides.size > 1 else 0
            goal_touched = bool(np.asarray(episode_results.goal_touched[index]))
            cliff_fell = bool(np.asarray(episode_results.cliff_fell[index]))
            termination = _termination_reason(
                goal_touched=goal_touched,
                touched_death_plane=bool(np.asarray(episode_results.touched_death_plane[index])),
                went_airborne=bool(np.asarray(episode_results.went_airborne[index])),
                steps_alive=active_count * sense_act_phases,
                max_steps=max_steps,
                sense_act_phases=sense_act_phases,
            )
            for local_index in range(active_count):
                step_rows.append(
                    {
                        "run_id": run_id,
                        "specimen_id": specimen_id,
                        "assay_family": "state_space_trace",
                        "assay_id": str(assay_name),
                        "surface_id": surface_id,
                        "surface_label": _surface_label(surface_id),
                        "deformation_condition": active_deformation_condition,
                        "signaling_condition": active_signaling_condition,
                        "trial_id": trial_id,
                        "distance_pair": distance_pair,
                        "goal_distance": goal_distance,
                        "cliff_distance": cliff_distance,
                        "width": width,
                        "side": side,
                        "goal_direction": goal_direction,
                        "env_step": int(env_steps[local_index]),
                        "step_index": int(local_index),
                        "done_before": bool(done_before[local_index]),
                        "done_after": bool(done_after[local_index]),
                        "com_x": float(com_x[local_index]),
                        "com_y": float(com_y[local_index]),
                        "min_x": float(min_x[local_index]),
                        "max_x": float(max_x[local_index]),
                        "min_y": float(min_y[local_index]),
                        "max_y": float(max_y[local_index]),
                        "front_x": float(front_x[local_index]),
                        "back_x": float(back_x[local_index]),
                        "goal_block_x": goal_block_x,
                        "goal_block_y": goal_block_y,
                        "goal_contact_x": float(goal_contact_x),
                        "cliff_edge_x": int(cliff_edge_x[local_index]),
                        "cliff_side": str(nearest_cliff_side[local_index]),
                        "nearest_cliff_side": str(nearest_cliff_side[local_index]),
                        "cliff_frontier_x": float(cliff_frontier_x[local_index]),
                        "left_frontier_x": left_frontier_x,
                        "right_frontier_x": right_frontier_x,
                        "com_x_from_start": float(com_x[local_index] - init_com_x),
                        "com_x_to_goal": float(com_x_to_goal[local_index]),
                        "com_x_to_cliff": float(cliff_frontier_x[local_index] - com_x[local_index]),
                        "goal_distance_remaining": float(goal_distance_remaining[local_index]),
                        "cliff_edge_body_x": float(cliff_edge_body_x[local_index]),
                        "cliff_clearance": float(cliff_clearance[local_index]),
                        "left_clearance": float(left_clearance[local_index]),
                        "right_clearance": float(right_clearance[local_index]),
                        "goal_touched_final": goal_touched,
                        "cliff_fell_final": cliff_fell,
                        "termination_reason": termination,
                        "trace_npz": str(npz_path),
                    }
                )
            trial_rows.append(
                {
                    "run_id": run_id,
                    "specimen_id": specimen_id,
                    "assay_family": "state_space_trace",
                    "assay_id": str(assay_name),
                    "surface_id": surface_id,
                    "surface_label": _surface_label(surface_id),
                    "deformation_condition": active_deformation_condition,
                    "signaling_condition": active_signaling_condition,
                    "trial_id": trial_id,
                    "distance_pair": distance_pair,
                    "goal_distance": goal_distance,
                    "cliff_distance": cliff_distance,
                    "width": width,
                    "side": side,
                    "goal_direction": goal_direction,
                    "trial_repeat_index": trial_repeat_index,
                    "max_steps": max_steps,
                    "steps_alive": active_count,
                    "goal_touched": goal_touched,
                    "cliff_fell": cliff_fell,
                    "success": bool(goal_touched and not cliff_fell) if surface_id != "double_cliff" else bool(not cliff_fell),
                    "termination_reason": termination,
                    "init_com_x": init_com_x,
                    "final_com_x": float(com_x[-1]) if active_count > 0 else None,
                    "final_front_x": float(front_x[-1]) if active_count > 0 else None,
                    "final_back_x": float(back_x[-1]) if active_count > 0 else None,
                    "goal_block_x": goal_block_x,
                    "goal_block_y": goal_block_y,
                    "goal_contact_x": float(goal_contact_x),
                    "cliff_edge_x": None if surface_id == "double_cliff" else int(cliff_edge_x[0]),
                    "cliff_frontier_x": None if surface_id == "double_cliff" else float(cliff_frontier_x[0]),
                    "cliff_side": "both" if surface_id == "double_cliff" else str(nearest_cliff_side[0]),
                    "left_frontier_x": left_frontier_x,
                    "right_frontier_x": right_frontier_x,
                    "min_cliff_clearance": min_cliff_clearance,
                    "final_cliff_clearance": final_cliff_clearance,
                    "max_cliff_protrusion": max_cliff_protrusion,
                    "first_cliff_reach_step": first_cliff_reach_step,
                    "returned_from_edge": returned_from_edge,
                    "min_left_clearance": float(np.nanmin(left_clearance)) if np.any(np.isfinite(left_clearance)) else None,
                    "min_right_clearance": float(np.nanmin(right_clearance)) if np.any(np.isfinite(right_clearance)) else None,
                    "first_left_cliff_reach_step": first_left_cliff_reach_step,
                    "first_right_cliff_reach_step": first_right_cliff_reach_step,
                    "side_switch_count": side_switch_count,
                    "trace_npz": str(npz_path),
                }
            )
            trace_files.append(
                {
                    "surface_id": surface_id,
                    "trial_id": trial_id,
                    "path": str(npz_path),
                    "step_count": active_count,
                    "exists": npz_path.exists(),
                    "min_cliff_clearance": min_cliff_clearance,
                    "first_cliff_reach_step": first_cliff_reach_step,
                }
            )

    trial_df = pd.DataFrame(trial_rows)
    step_df = pd.DataFrame(step_rows)
    edge_audit_rows = _state_space_edge_audit_rows(trial_df)
    trial_df.to_csv(assay_dir / "trial_manifest.csv", index=False)
    step_df.to_csv(assay_dir / "step_manifest.csv", index=False)
    step_df.to_parquet(assay_dir / "step_manifest.parquet", index=False)
    pd.DataFrame(edge_audit_rows).to_csv(assay_dir / "edge_event_audit.csv", index=False)
    if env0 is None:
        raise ValueError("state-space trace produced no environments")
    reference_body = _reference_body_spec(cfg)
    robot_morph = make_robot_body(
        int(reference_body["robot_w"]),
        int(reference_body["robot_h"]),
        shape=str(reference_body.get("shape", "rectangle")),
    )
    np.savez_compressed(
        assay_dir / "static_arrays.npz",
        spring_a_idx=np.asarray(env0.topology.a_idx, dtype=np.int32),
        spring_b_idx=np.asarray(env0.topology.b_idx, dtype=np.int32),
        spring_mask=np.asarray(env0.sim_state.spring_mask, dtype=bool),
        spring_const=np.asarray(env0.sim_state.spring_const, dtype=np.float32),
        spring_init_rest_length=np.asarray(env0.sim_state.spring_init_rest_length, dtype=np.float32),
        robot_point_mask=np.asarray(env0.robot_point_mask, dtype=bool),
        robot_morph=np.asarray(robot_morph, dtype=np.int32),
        boxel_corners=np.asarray(env0.collision_data.boxel_corners, dtype=np.int32),
        boxel_mask=np.asarray(env0.collision_data.boxel_mask, dtype=bool),
        surface_edge_mask=np.asarray(env0.collision_data.surface_edge_mask, dtype=bool),
        h_compact_cell_indices=np.asarray(env0.per_axis_info.h_compact_cell_indices, dtype=np.int32),
        v_compact_cell_indices=np.asarray(env0.per_axis_info.v_compact_cell_indices, dtype=np.int32),
        h_compact_spring_pairs=np.asarray(env0.per_axis_info.h_compact_spring_pairs, dtype=np.int32),
        v_compact_spring_pairs=np.asarray(env0.per_axis_info.v_compact_spring_pairs, dtype=np.int32),
    )
    phase_cfg0 = trace_groups[0]["phase_cfg"]
    ou_enabled = bool(resolved_noise_settings(phase_cfg0)["ou_enabled"])
    hidden_start_channel = int(base_channel_count(ou_enabled=ou_enabled))
    nca_channel_schema = {
        "deformation_channels": [0, 1, 2, 3],
        "action_channels": [int(ACTION_H_IDX), int(ACTION_V_IDX)],
        "hidden_start_channel": hidden_start_channel,
        "ou_enabled": ou_enabled,
    }
    manifest = {
        "schema_version": _POST_RUN_SCHEMA_VERSION,
        "kind": "sensing_edges_post_run_state_space_trace",
        "assay_family": "state_space_trace",
        "assay_id": str(assay_name),
        "specimen_id": specimen_id,
        "run_id": run_id,
        "surface_id": surfaces[0] if len(surfaces) == 1 else "multi_surface",
        "surface_label": _surface_label(surfaces[0]) if len(surfaces) == 1 else "Multi Surface",
        "surfaces": list(surfaces),
        "surface_groups": surface_group_records,
        "deformation_condition": active_deformation_condition,
        "signaling_condition": active_signaling_condition,
        "body": _reference_body_spec(cfg)["label"],
        "native_pairs": list(post_run_cfg.native_pairs),
        "long_distance_pairs": list(post_run_cfg.long_distance_pairs),
        "double_cliff_widths": [int(width) for width in list(post_run_cfg.double_cliff.widths)],
        "goal_sides": list(_configured_goal_sides(cfg)),
        "trial_count": int(len(trial_df)),
        "step_count": int(len(step_df)),
        "max_steps": {
            str(record["surface_id"]): int(record["max_steps"])
            for record in surface_group_records
        },
        "sense_act_phases_per_env_step": int(phase_cfg0.sim.sense_act_phases_per_env_step),
        "physics_substeps": int(phase_cfg0.sim.physics_substeps),
        "substeps_per_phase": int(phase_cfg0.sim.physics_substeps) // int(phase_cfg0.sim.sense_act_phases_per_env_step),
        "nca_steps_per_phase": int(phase_cfg0.nca.nca_steps_per_phase),
        "nca_channel_schema": nca_channel_schema,
        "paths": {
            "trial_manifest": str(assay_dir / "trial_manifest.csv"),
            "step_manifest_csv": str(assay_dir / "step_manifest.csv"),
            "step_manifest_parquet": str(assay_dir / "step_manifest.parquet"),
            "edge_event_audit": str(assay_dir / "edge_event_audit.csv"),
            "static_arrays": str(assay_dir / "static_arrays.npz"),
            "traces_dir": str(traces_dir),
        },
        "trace_files": trace_files,
    }
    aggregate = {
        **manifest,
        "success_count": int(trial_df["success"].sum()) if "success" in trial_df.columns else 0,
        "mean_steps_alive": float(trial_df["steps_alive"].mean()) if "steps_alive" in trial_df.columns and not trial_df.empty else None,
        "edge_reach_count": int(trial_df["first_cliff_reach_step"].notna().sum()) if "first_cliff_reach_step" in trial_df.columns else 0,
        "returned_from_edge_count": int(trial_df["returned_from_edge"].sum()) if "returned_from_edge" in trial_df.columns else 0,
    }
    _write_json(assay_dir / "manifest.json", manifest)
    _write_json(assay_dir / "summary.json", aggregate)
    _write_json(assay_dir / "aggregate.json", aggregate)
    return trial_df, aggregate, assay_dir


def _run_combined_ablation_quick_assay(
    *,
    cfg: DictConfig,
    runtime_resolution,
    specimen_dir: Path,
    specimen_id: str,
    genotype: np.ndarray,
    selection_row: dict[str, Any],
    run_id: str,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    del selection_row
    post_run_cfg = cfg.eval.post_run
    assay_dir = specimen_dir / "combined_ablation_quick"
    test_specs = _combined_ablation_tests(cfg)
    all_frames: list[pd.DataFrame] = []
    condition_records: list[dict[str, Any]] = []
    inventory_records: list[dict[str, Any]] = []
    max_steps_by_pair: list[dict[str, Any]] = []
    requested_trials_per_side = int(post_run_cfg.trials_per_side)
    for test_spec in test_specs:
        test_id = str(test_spec["test_id"])
        condition_dir = assay_dir / "conditions" / test_id
        effective_trials_per_side = _effective_trials_per_side(
            cfg,
            deformation_condition=str(test_spec["deformation_condition"]),
        )
        subgroup_started = time.monotonic()
        log.info(
            "Post-run subgroup started | specimen_id=%s assay=combined_ablation_quick test=%s deformation=%s signaling=%s",
            specimen_id,
            test_id,
            test_spec["deformation_condition"],
            test_spec["signaling_condition"],
        )
        condition_df, pair_summaries, pair_budgets, condition_record = _run_native_like_surface(
            cfg=cfg,
            runtime_resolution=runtime_resolution,
            genotype=genotype,
            run_id=run_id,
            specimen_id=specimen_id,
            assay_family="combined_ablation_quick",
            surface_id="native_core",
            pairs=list(post_run_cfg.native_pairs),
            trials_per_side=int(effective_trials_per_side),
            signaling_condition=str(test_spec["signaling_condition"]),
            deformation_condition=str(test_spec["deformation_condition"]),
            render_output_dir=condition_dir / "renders",
            trial_id_prefix=f"combined_ablation_quick_{test_id}",
            render_metadata={
                "signaling_condition": str(test_spec["signaling_condition"]),
                "deformation_condition": str(test_spec["deformation_condition"]),
            },
            condition_id_override=test_id,
            condition_label_override=str(test_spec["label"]),
        )
        max_steps_by_pair.extend(pair_budgets)
        condition_record.update(
            {
                "test_id": test_id,
                "condition_id": test_id,
                "condition_label": str(test_spec["label"]),
                "distance_pair_summaries": pair_summaries,
                "signaling_condition": str(test_spec["signaling_condition"]),
                "deformation_condition": str(test_spec["deformation_condition"]),
                "requested_trials_per_side": requested_trials_per_side,
                "effective_trials_per_side": int(effective_trials_per_side),
                "deterministic_rollout": bool(effective_trials_per_side == 1),
                "render_enabled": True,
            }
        )
        _write_json(condition_dir / "summary.json", condition_record)
        _write_json(condition_dir / "aggregate.json", condition_record)
        all_frames.append(condition_df)
        condition_records.append(condition_record)
        inventory_records.append(
            _surface_inventory_record(
                assay_family="combined_ablation_quick",
                surface_id="native_core",
                subgroup_type="condition",
                subgroup_id=test_id,
                surface_dir=assay_dir,
                subgroup_dir=condition_dir,
            )
        )
        log.info(
            "Post-run subgroup completed | specimen_id=%s assay=combined_ablation_quick test=%s trial_count=%d elapsed_s=%.2f",
            specimen_id,
            test_id,
            len(condition_df),
            time.monotonic() - subgroup_started,
        )
    df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    charts_dir = assay_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    _chart_native_surface_family(
        charts_dir=charts_dir,
        title_prefix="Combined Ablation Quick",
        category_field="condition_label",
        records=condition_records,
    )
    aggregate = {
        "schema_version": _POST_RUN_SCHEMA_VERSION,
        "kind": "sensing_edges_post_run_combined_ablation_quick",
        "specimen_id": specimen_id,
        "surface_id": "native_core",
        "surface_label": _surface_label("native_core"),
        "condition_summaries": condition_records,
        "max_steps_by_pair": max_steps_by_pair,
        "requested_trials_per_side": requested_trials_per_side,
    }
    _finalize_assay_outputs(
        assay_dir=assay_dir,
        manifest={
            "schema_version": _POST_RUN_SCHEMA_VERSION,
            "kind": "sensing_edges_post_run_assay_manifest",
            "assay_family": "combined_ablation_quick",
            "specimen_id": specimen_id,
            "run_id": run_id,
            "surface_id": "native_core",
            "surface_label": _surface_label("native_core"),
            "tests": test_specs,
            "pairs": list(post_run_cfg.native_pairs),
            "max_steps_by_pair": max_steps_by_pair,
            "requested_trials_per_side": requested_trials_per_side,
        },
        rows=df,
        aggregate=aggregate,
        condition_records=condition_records,
        specimen_records=[
            {
                "specimen_id": specimen_id,
                "assay_family": "combined_ablation_quick",
                "surface_id": "native_core",
                "surface_label": _surface_label("native_core"),
                "condition_count": int(len(condition_records)),
                "trial_count": int(len(df)),
            }
        ],
        inventory_records=inventory_records,
    )
    return df, aggregate, assay_dir


def _prepare_batched_post_run_jobs(
    *,
    cfg: DictConfig,
    run_specs: Sequence[dict[str, Any]],
) -> list[BatchedPostRunJob]:
    jobs: list[BatchedPostRunJob] = []
    for spec in run_specs:
        run_dir = Path(str(spec["run_dir"]))
        specimen_id = str(spec["specimen_id"])
        specimen_root = (
            Path(str(spec["specimen_root"]))
            if spec.get("specimen_root") is not None
            else resolve_stored_specimen_root(
                run_dir,
                output_root=str(cfg.specimens.get("output_root", "specimens")),
            )
        )
        row, specimen_dir = ensure_specimen_artifacts(
            cfg=cfg,
            run_dir=run_dir,
            specimen_root=specimen_root,
            specimen_id=specimen_id,
        )
        specimen_manifest_path = specimen_dir / "manifest.json"
        genotype = np.load(specimen_dir / "genotype.npz", allow_pickle=False)["genotype"]
        manifest = load_manifest(specimen_manifest_path)
        analysis_root = resolve_analysis_root(
            run_dir,
            specimen_id,
            output_root=str(cfg.eval.post_run.get("analysis_root", "analyses")),
        )
        analysis_root.mkdir(parents=True, exist_ok=True)
        analysis_manifest_path = analysis_root / "manifest.json"
        analysis_index_path = analysis_root / "index.json"
        pointer_path = (
            specimen_root
            / str(cfg.eval.post_run.get("output_dir", "post_run"))
            / f"{specimen_id}.json"
        )
        jobs.append(
            BatchedPostRunJob(
                specimen_id=specimen_id,
                run_id=str(manifest["run_id"]),
                run_dir=run_dir,
                specimen_root=specimen_root,
                specimen_dir=specimen_dir,
                specimen_manifest_path=specimen_manifest_path,
                specimen_manifest=dict(manifest),
                selection_row=dict(row),
                analysis_root=analysis_root,
                analysis_manifest_path=analysis_manifest_path,
                analysis_index_path=analysis_index_path,
                pointer_path=pointer_path,
                genotype=np.asarray(genotype, dtype=np.float32),
            )
        )
    return jobs


def _run_signaling_ablation_assay_batched(
    *,
    cfg: DictConfig,
    runtime_resolution,
    jobs: Sequence[BatchedPostRunJob],
) -> dict[str, Path]:
    post_run_cfg = cfg.eval.post_run
    signaling_conditions = compatible_post_run_signaling_conditions(cfg)
    assay_dirs = {
        job.specimen_id: job.analysis_root / "signaling_ablation"
        for job in jobs
    }
    all_surface_frames: dict[str, list[pd.DataFrame]] = {job.specimen_id: [] for job in jobs}
    surface_summaries: dict[str, list[dict[str, Any]]] = {job.specimen_id: [] for job in jobs}
    cross_surface_records: dict[str, list[dict[str, Any]]] = {job.specimen_id: [] for job in jobs}
    inventory_records: dict[str, list[dict[str, Any]]] = {job.specimen_id: [] for job in jobs}
    configured_surfaces = _configured_surface_ids(cfg, config_key="signaling_surfaces")
    for surface_id in configured_surfaces:
        surface_frames: dict[str, list[pd.DataFrame]] = {job.specimen_id: [] for job in jobs}
        surface_condition_records: dict[str, list[dict[str, Any]]] = {job.specimen_id: [] for job in jobs}
        max_steps_by_pair: dict[str, list[dict[str, Any]]] = {job.specimen_id: [] for job in jobs}
        requested_trials_per_side = int(post_run_cfg.trials_per_side)
        requested_trials_per_width = int(post_run_cfg.double_cliff.trials_per_width)
        effective_trials_per_side = _effective_trials_per_side(cfg)
        effective_trials_per_width = _effective_trials_per_width(cfg)
        for condition in signaling_conditions:
            subgroup_started = time.monotonic()
            log.info(
                "Post-run batched subgroup started | assay=signaling_ablation surface=%s condition=%s specimen_count=%d render=%s",
                surface_id,
                condition,
                len(jobs),
                bool(_should_render_signaling_condition(cfg, str(condition))),
            )
            render_output_dirs = {
                job.specimen_id: (
                    assay_dirs[job.specimen_id] / "surfaces" / surface_id / "conditions" / str(condition) / "renders"
                    if _should_render_signaling_condition(cfg, str(condition))
                    else None
                )
                for job in jobs
            }
            if surface_id == "double_cliff":
                phase_cfg = _double_cliff_phase_cfg(
                    cfg,
                    active_signaling_condition=str(condition),
                )
                entries = _build_double_cliff_entries(
                    phase_cfg,
                    [int(width) for width in list(post_run_cfg.double_cliff.widths)],
                    plateau_width=int(post_run_cfg.double_cliff.plateau_width),
                    trials_per_width=int(effective_trials_per_width),
                )
                episode_batches, trace_batches, envs, fitness_batches = _run_trace_eval_batched(
                    phase_cfg=phase_cfg,
                    runtime_resolution=runtime_resolution,
                    genotypes=np.stack([job.genotype for job in jobs], axis=0),
                    entries=entries,
                    workload="trace",
                    seeds=[
                        _stable_seed(job.run_id, job.specimen_id, "signaling", surface_id, condition)
                        for job in jobs
                    ],
                )
                for job, episode_results, traces, fitnesses in zip(jobs, episode_batches, trace_batches, fitness_batches):
                    condition_dir = assay_dirs[job.specimen_id] / "surfaces" / surface_id / "conditions" / str(condition)
                    condition_df = pd.DataFrame(
                        _double_cliff_rows(
                            episode_results=episode_results,
                            traces=traces,
                            envs=envs,
                            entries=entries,
                            specimen_id=job.specimen_id,
                            run_id=job.run_id,
                            max_steps=_double_cliff_max_steps(cfg),
                        )
                    )
                    condition_df["fell"] = condition_df["fell_left"].astype(bool) | condition_df["fell_right"].astype(bool)
                    condition_df["surface_id"] = surface_id
                    condition_df["surface_label"] = _surface_label(surface_id)
                    condition_df["surface_max_steps"] = _double_cliff_max_steps(cfg)
                    condition_df["signaling_condition"] = str(condition)
                    condition_df["condition_id"] = str(condition)
                    condition_df["condition_label"] = _signaling_label(str(condition))
                    if render_output_dirs[job.specimen_id] is not None:
                        _render_selected_candidates(
                            cfg=cfg,
                            output_dir=render_output_dirs[job.specimen_id],
                            candidates=_double_cliff_render_candidates(
                                episode_results=episode_results,
                                traces=traces,
                                envs=envs,
                                entries=entries,
                                fitnesses=fitnesses,
                            ),
                            manifest_kind="sensing_edges_post_run_render_manifest",
                        )
                    condition_record = _double_cliff_surface_specimen_record(
                        specimen_id=job.specimen_id,
                        assay_family="signaling_ablation",
                        surface_id=surface_id,
                        rows=condition_df,
                        extra_fields={
                            "signaling_condition": str(condition),
                            "condition_id": str(condition),
                            "condition_label": _signaling_label(str(condition)),
                            "max_steps": _double_cliff_max_steps(cfg),
                            "requested_trials_per_width": requested_trials_per_width,
                            "effective_trials_per_width": int(effective_trials_per_width),
                            "deterministic_rollout": bool(effective_trials_per_width == 1),
                            "render_enabled": bool(render_output_dirs[job.specimen_id] is not None),
                        },
                    )
                    _write_json(condition_dir / "summary.json", condition_record)
                    _write_json(condition_dir / "aggregate.json", condition_record)
                    surface_frames[job.specimen_id].append(condition_df)
                    surface_condition_records[job.specimen_id].append(condition_record)
                    inventory_records[job.specimen_id].append(
                        _surface_inventory_record(
                            assay_family="signaling_ablation",
                            surface_id=surface_id,
                            subgroup_type="condition",
                            subgroup_id=str(condition),
                            surface_dir=assay_dirs[job.specimen_id] / "surfaces" / surface_id,
                            subgroup_dir=condition_dir,
                        )
                    )
            else:
                surface_results = _run_native_like_surface_batched(
                    cfg=cfg,
                    runtime_resolution=runtime_resolution,
                    jobs=jobs,
                    assay_family="signaling_ablation",
                    surface_id=surface_id,
                    pairs=(
                        list(post_run_cfg.native_pairs)
                        if surface_id == "native_core"
                        else list(post_run_cfg.long_distance_pairs)
                    ),
                    trials_per_side=int(effective_trials_per_side),
                    signaling_condition=str(condition),
                    render_output_dirs=render_output_dirs,
                    trial_id_prefix=f"signaling_{surface_id}_{condition}",
                    render_metadata={"signaling_condition": str(condition)},
                )
                for job in jobs:
                    condition_dir = assay_dirs[job.specimen_id] / "surfaces" / surface_id / "conditions" / str(condition)
                    condition_df, pair_summaries, pair_budgets, condition_record = surface_results[job.specimen_id]
                    max_steps_by_pair[job.specimen_id].extend(pair_budgets)
                    condition_record.update(
                        {
                            "signaling_condition": str(condition),
                            "condition_id": str(condition),
                            "condition_label": _signaling_label(str(condition)),
                            "distance_pair_summaries": pair_summaries,
                            "requested_trials_per_side": requested_trials_per_side,
                            "effective_trials_per_side": int(effective_trials_per_side),
                            "deterministic_rollout": bool(effective_trials_per_side == 1),
                            "render_enabled": bool(render_output_dirs[job.specimen_id] is not None),
                        }
                    )
                    _write_json(condition_dir / "summary.json", condition_record)
                    _write_json(condition_dir / "aggregate.json", condition_record)
                    surface_frames[job.specimen_id].append(condition_df)
                    surface_condition_records[job.specimen_id].append(condition_record)
                    inventory_records[job.specimen_id].append(
                        _surface_inventory_record(
                            assay_family="signaling_ablation",
                            surface_id=surface_id,
                            subgroup_type="condition",
                            subgroup_id=str(condition),
                            surface_dir=assay_dirs[job.specimen_id] / "surfaces" / surface_id,
                            subgroup_dir=condition_dir,
                        )
                    )
            log.info(
                "Post-run batched subgroup completed | assay=signaling_ablation surface=%s condition=%s specimen_count=%d elapsed_s=%.2f",
                surface_id,
                condition,
                len(jobs),
                time.monotonic() - subgroup_started,
            )
        for job in jobs:
            intact_summary = next(
                (record for record in surface_condition_records[job.specimen_id] if record["signaling_condition"] == "intact"),
                None,
            )
            for record in surface_condition_records[job.specimen_id]:
                delta_fields = (
                    ("survival_rate",) if surface_id == "double_cliff" else (
                        "success_rate",
                        "fall_rate",
                        "mean_fitness",
                        "mean_max_cliff_protrusion",
                        "mean_time_to_arrest",
                    )
                )
                record["delta_from_intact"] = {
                    metric: _relative_delta(
                        record[metric]["mean"] if isinstance(record.get(metric), dict) else record.get(metric),
                        None
                        if intact_summary is None
                        else (
                            intact_summary[metric]["mean"]
                            if isinstance(intact_summary.get(metric), dict)
                            else intact_summary.get(metric)
                        ),
                    )
                    for metric in delta_fields
                }
            cross_surface_records[job.specimen_id].extend(dict(record) for record in surface_condition_records[job.specimen_id])
            surface_df = pd.concat(surface_frames[job.specimen_id], ignore_index=True) if surface_frames[job.specimen_id] else pd.DataFrame()
            surface_summary = {
                "surface_id": surface_id,
                "surface_label": _surface_label(surface_id),
                "condition_summaries": surface_condition_records[job.specimen_id],
                "condition_count": int(len(surface_condition_records[job.specimen_id])),
                "max_steps_by_pair": max_steps_by_pair[job.specimen_id],
                "requested_trials_per_side": (
                    None if surface_id == "double_cliff" else requested_trials_per_side
                ),
                "effective_trials_per_side": (
                    None if surface_id == "double_cliff" else int(effective_trials_per_side)
                ),
                "requested_trials_per_width": (
                    requested_trials_per_width if surface_id == "double_cliff" else None
                ),
                "effective_trials_per_width": (
                    int(effective_trials_per_width) if surface_id == "double_cliff" else None
                ),
                "deterministic_rollout": bool(
                    effective_trials_per_width == 1 if surface_id == "double_cliff" else effective_trials_per_side == 1
                ),
            }
            surface_charts_dir = assay_dirs[job.specimen_id] / "surfaces" / surface_id / "charts"
            surface_charts_dir.mkdir(parents=True, exist_ok=True)
            if surface_id == "double_cliff":
                _chart_double_cliff_surface_family(
                    charts_dir=surface_charts_dir,
                    title_prefix=f"{_surface_label(surface_id)} Signaling",
                    category_field="condition_label",
                    records=surface_condition_records[job.specimen_id],
                )
            else:
                _chart_native_surface_family(
                    charts_dir=surface_charts_dir,
                    title_prefix=f"{_surface_label(surface_id)} Signaling",
                    category_field="condition_label",
                    records=surface_condition_records[job.specimen_id],
                )
            _finalize_assay_outputs(
                assay_dir=assay_dirs[job.specimen_id] / "surfaces" / surface_id,
                manifest={
                    "schema_version": _POST_RUN_SCHEMA_VERSION,
                    "kind": "sensing_edges_post_run_assay_manifest",
                    "assay_family": "signaling_ablation",
                    "surface_id": surface_id,
                    "surface_label": _surface_label(surface_id),
                    "specimen_id": job.specimen_id,
                    "run_id": job.run_id,
                    "conditions": list(signaling_conditions),
                    "pairs": (
                        list(post_run_cfg.native_pairs)
                        if surface_id == "native_core"
                        else list(post_run_cfg.long_distance_pairs)
                        if surface_id == "long_distance_generalization"
                        else None
                    ),
                    "widths": (
                        [int(width) for width in list(post_run_cfg.double_cliff.widths)]
                        if surface_id == "double_cliff"
                        else None
                    ),
                    "max_steps": (
                        _double_cliff_max_steps(cfg)
                        if surface_id == "double_cliff"
                        else None
                    ),
                    "max_steps_by_pair": max_steps_by_pair[job.specimen_id] if surface_id != "double_cliff" else None,
                    "requested_trials_per_side": (
                        None if surface_id == "double_cliff" else requested_trials_per_side
                    ),
                    "effective_trials_per_side": (
                        None if surface_id == "double_cliff" else int(effective_trials_per_side)
                    ),
                    "requested_trials_per_width": (
                        requested_trials_per_width if surface_id == "double_cliff" else None
                    ),
                    "effective_trials_per_width": (
                        int(effective_trials_per_width) if surface_id == "double_cliff" else None
                    ),
                    "deterministic_rollout": bool(
                        effective_trials_per_width == 1 if surface_id == "double_cliff" else effective_trials_per_side == 1
                    ),
                },
                rows=surface_df,
                aggregate={
                    "schema_version": _POST_RUN_SCHEMA_VERSION,
                    "kind": "sensing_edges_post_run_signaling_ablation_surface",
                    "specimen_id": job.specimen_id,
                    **surface_summary,
                },
                condition_records=surface_condition_records[job.specimen_id],
                specimen_records=[
                    {
                        "specimen_id": job.specimen_id,
                        "assay_family": "signaling_ablation",
                        "surface_id": surface_id,
                        "surface_label": _surface_label(surface_id),
                        "condition_count": int(len(surface_condition_records[job.specimen_id])),
                        "trial_count": int(len(surface_df)),
                    }
                ],
                inventory_records=[
                    record for record in inventory_records[job.specimen_id] if record["surface_id"] == surface_id
                ],
            )
            all_surface_frames[job.specimen_id].append(surface_df)
            surface_summaries[job.specimen_id].append(surface_summary)
    for job in jobs:
        df = pd.concat(all_surface_frames[job.specimen_id], ignore_index=True) if all_surface_frames[job.specimen_id] else pd.DataFrame()
        overall_condition_records = _overall_multi_surface_group_records(
            df,
            specimen_id=job.specimen_id,
            assay_family="signaling_ablation",
            group_field="condition_id",
            group_label_field="condition_label",
        )
        _chart_overall_multi_surface_family(
            charts_dir=assay_dirs[job.specimen_id] / "charts",
            title_prefix="Signaling Overall",
            category_field="condition_label",
            records=overall_condition_records,
        )
        aggregate = {
            "schema_version": _POST_RUN_SCHEMA_VERSION,
            "kind": "sensing_edges_post_run_signaling_ablation",
            "specimen_id": job.specimen_id,
            "conditions": list(signaling_conditions),
            "surface_summaries": surface_summaries[job.specimen_id],
            "cross_surface_condition_summaries": cross_surface_records[job.specimen_id],
            "overall_condition_summaries": overall_condition_records,
        }
        _finalize_assay_outputs(
            assay_dir=assay_dirs[job.specimen_id],
            manifest={
                "schema_version": _POST_RUN_SCHEMA_VERSION,
                "kind": "sensing_edges_post_run_assay_manifest",
                "assay_family": "signaling_ablation",
                "specimen_id": job.specimen_id,
                "run_id": job.run_id,
                "conditions": list(signaling_conditions),
                "surfaces": list(configured_surfaces),
            },
            rows=df,
            aggregate=aggregate,
            condition_records=overall_condition_records,
            specimen_records=[
                {
                    "specimen_id": job.specimen_id,
                    "assay_family": "signaling_ablation",
                    "surface_count": int(len(surface_summaries[job.specimen_id])),
                    "condition_count": int(len(cross_surface_records[job.specimen_id])),
                    "trial_count": int(len(df)),
                }
            ],
            inventory_records=inventory_records[job.specimen_id],
            cross_surface_records=cross_surface_records[job.specimen_id],
        )
    return assay_dirs


def _run_deformation_ablation_assay_batched(
    *,
    cfg: DictConfig,
    runtime_resolution,
    jobs: Sequence[BatchedPostRunJob],
) -> dict[str, Path]:
    post_run_cfg = cfg.eval.post_run
    deformation_conditions = [str(condition) for condition in list(post_run_cfg.deformation_conditions)]
    assay_dirs = {
        job.specimen_id: job.analysis_root / "deformation_ablation"
        for job in jobs
    }
    all_surface_frames: dict[str, list[pd.DataFrame]] = {job.specimen_id: [] for job in jobs}
    surface_summaries: dict[str, list[dict[str, Any]]] = {job.specimen_id: [] for job in jobs}
    cross_surface_records: dict[str, list[dict[str, Any]]] = {job.specimen_id: [] for job in jobs}
    inventory_records: dict[str, list[dict[str, Any]]] = {job.specimen_id: [] for job in jobs}
    configured_surfaces = _configured_surface_ids(cfg, config_key="deformation_surfaces")
    for surface_id in configured_surfaces:
        surface_frames: dict[str, list[pd.DataFrame]] = {job.specimen_id: [] for job in jobs}
        surface_condition_records: dict[str, list[dict[str, Any]]] = {job.specimen_id: [] for job in jobs}
        max_steps_by_pair: dict[str, list[dict[str, Any]]] = {job.specimen_id: [] for job in jobs}
        requested_trials_per_side = int(post_run_cfg.trials_per_side)
        requested_trials_per_width = int(post_run_cfg.double_cliff.trials_per_width)
        condition_effective_trials_per_side: dict[str, int] = {}
        condition_effective_trials_per_width: dict[str, int] = {}
        condition_deterministic_rollout: dict[str, bool] = {}
        for condition in deformation_conditions:
            effective_trials_per_side = _effective_trials_per_side(
                cfg,
                deformation_condition=str(condition),
            )
            effective_trials_per_width = _effective_trials_per_width(
                cfg,
                deformation_condition=str(condition),
            )
            condition_effective_trials_per_side[str(condition)] = int(effective_trials_per_side)
            condition_effective_trials_per_width[str(condition)] = int(effective_trials_per_width)
            condition_deterministic_rollout[str(condition)] = bool(
                effective_trials_per_width == 1 if surface_id == "double_cliff" else effective_trials_per_side == 1
            )
            subgroup_started = time.monotonic()
            log.info(
                "Post-run batched subgroup started | assay=deformation_ablation surface=%s condition=%s specimen_count=%d effective_trials_side=%d effective_trials_width=%d render=%s",
                surface_id,
                condition,
                len(jobs),
                int(effective_trials_per_side),
                int(effective_trials_per_width),
                bool(_should_render_deformation_condition(cfg, str(condition))),
            )
            render_output_dirs = {
                job.specimen_id: (
                    assay_dirs[job.specimen_id] / "surfaces" / surface_id / "conditions" / str(condition) / "renders"
                    if _should_render_deformation_condition(cfg, str(condition))
                    else None
                )
                for job in jobs
            }
            if surface_id == "double_cliff":
                phase_cfg = _double_cliff_phase_cfg(
                    cfg,
                    active_deformation_condition=str(condition),
                )
                entries = _build_double_cliff_entries(
                    phase_cfg,
                    [int(width) for width in list(post_run_cfg.double_cliff.widths)],
                    plateau_width=int(post_run_cfg.double_cliff.plateau_width),
                    trials_per_width=int(effective_trials_per_width),
                )
                episode_batches, trace_batches, envs, fitness_batches = _run_trace_eval_batched(
                    phase_cfg=phase_cfg,
                    runtime_resolution=runtime_resolution,
                    genotypes=np.stack([job.genotype for job in jobs], axis=0),
                    entries=entries,
                    workload="trace",
                    seeds=[
                        _stable_seed(job.run_id, job.specimen_id, "deformation", surface_id, condition)
                        for job in jobs
                    ],
                )
                for job, episode_results, traces, fitnesses in zip(jobs, episode_batches, trace_batches, fitness_batches):
                    condition_dir = assay_dirs[job.specimen_id] / "surfaces" / surface_id / "conditions" / str(condition)
                    condition_df = pd.DataFrame(
                        _double_cliff_rows(
                            episode_results=episode_results,
                            traces=traces,
                            envs=envs,
                            entries=entries,
                            specimen_id=job.specimen_id,
                            run_id=job.run_id,
                            max_steps=_double_cliff_max_steps(cfg),
                        )
                    )
                    condition_df["fell"] = condition_df["fell_left"].astype(bool) | condition_df["fell_right"].astype(bool)
                    condition_df["surface_id"] = surface_id
                    condition_df["surface_label"] = _surface_label(surface_id)
                    condition_df["surface_max_steps"] = _double_cliff_max_steps(cfg)
                    condition_df["deformation_condition"] = str(condition)
                    condition_df["condition_id"] = str(condition)
                    condition_df["condition_label"] = _deformation_label(str(condition))
                    if render_output_dirs[job.specimen_id] is not None:
                        _render_selected_candidates(
                            cfg=cfg,
                            output_dir=render_output_dirs[job.specimen_id],
                            candidates=_double_cliff_render_candidates(
                                episode_results=episode_results,
                                traces=traces,
                                envs=envs,
                                entries=entries,
                                fitnesses=fitnesses,
                            ),
                            manifest_kind="sensing_edges_post_run_render_manifest",
                        )
                    condition_record = _double_cliff_surface_specimen_record(
                        specimen_id=job.specimen_id,
                        assay_family="deformation_ablation",
                        surface_id=surface_id,
                        rows=condition_df,
                        extra_fields={
                            "deformation_condition": str(condition),
                            "condition_id": str(condition),
                            "condition_label": _deformation_label(str(condition)),
                            "deformation_metadata": _DEFORMATION_METADATA.get(str(condition), {}),
                            "deformation_channels": ["abs_h", "abs_v", "comp_h", "comp_v"],
                            "max_steps": _double_cliff_max_steps(cfg),
                            "requested_trials_per_width": requested_trials_per_width,
                            "effective_trials_per_width": int(effective_trials_per_width),
                            "deterministic_rollout": bool(effective_trials_per_width == 1),
                            "render_enabled": bool(render_output_dirs[job.specimen_id] is not None),
                        },
                    )
                    _write_json(condition_dir / "summary.json", condition_record)
                    _write_json(condition_dir / "aggregate.json", condition_record)
                    surface_frames[job.specimen_id].append(condition_df)
                    surface_condition_records[job.specimen_id].append(condition_record)
                    inventory_records[job.specimen_id].append(
                        _surface_inventory_record(
                            assay_family="deformation_ablation",
                            surface_id=surface_id,
                            subgroup_type="condition",
                            subgroup_id=str(condition),
                            surface_dir=assay_dirs[job.specimen_id] / "surfaces" / surface_id,
                            subgroup_dir=condition_dir,
                        )
                    )
            else:
                surface_results = _run_native_like_surface_batched(
                    cfg=cfg,
                    runtime_resolution=runtime_resolution,
                    jobs=jobs,
                    assay_family="deformation_ablation",
                    surface_id=surface_id,
                    pairs=(
                        list(post_run_cfg.native_pairs)
                        if surface_id == "native_core"
                        else list(post_run_cfg.long_distance_pairs)
                    ),
                    trials_per_side=int(effective_trials_per_side),
                    deformation_condition=str(condition),
                    render_output_dirs=render_output_dirs,
                    trial_id_prefix=f"deformation_{surface_id}_{condition}",
                    render_metadata={"deformation_condition": str(condition)},
                )
                for job in jobs:
                    condition_dir = assay_dirs[job.specimen_id] / "surfaces" / surface_id / "conditions" / str(condition)
                    condition_df, pair_summaries, pair_budgets, condition_record = surface_results[job.specimen_id]
                    max_steps_by_pair[job.specimen_id].extend(pair_budgets)
                    condition_record.update(
                        {
                            "deformation_condition": str(condition),
                            "condition_id": str(condition),
                            "condition_label": _deformation_label(str(condition)),
                            "distance_pair_summaries": pair_summaries,
                            "deformation_metadata": _DEFORMATION_METADATA.get(str(condition), {}),
                            "deformation_channels": ["abs_h", "abs_v", "comp_h", "comp_v"],
                            "requested_trials_per_side": requested_trials_per_side,
                            "effective_trials_per_side": int(effective_trials_per_side),
                            "deterministic_rollout": bool(effective_trials_per_side == 1),
                            "render_enabled": bool(render_output_dirs[job.specimen_id] is not None),
                        }
                    )
                    _write_json(condition_dir / "summary.json", condition_record)
                    _write_json(condition_dir / "aggregate.json", condition_record)
                    surface_frames[job.specimen_id].append(condition_df)
                    surface_condition_records[job.specimen_id].append(condition_record)
                    inventory_records[job.specimen_id].append(
                        _surface_inventory_record(
                            assay_family="deformation_ablation",
                            surface_id=surface_id,
                            subgroup_type="condition",
                            subgroup_id=str(condition),
                            surface_dir=assay_dirs[job.specimen_id] / "surfaces" / surface_id,
                            subgroup_dir=condition_dir,
                        )
                    )
            log.info(
                "Post-run batched subgroup completed | assay=deformation_ablation surface=%s condition=%s specimen_count=%d elapsed_s=%.2f",
                surface_id,
                condition,
                len(jobs),
                time.monotonic() - subgroup_started,
            )
        for job in jobs:
            intact_summary = next(
                (record for record in surface_condition_records[job.specimen_id] if record["deformation_condition"] == "intact"),
                None,
            )
            off_summary = next(
                (record for record in surface_condition_records[job.specimen_id] if record["deformation_condition"] == "deformation_off"),
                None,
            )
            for record in surface_condition_records[job.specimen_id]:
                delta_fields = (
                    ("survival_rate",) if surface_id == "double_cliff" else (
                        "success_rate",
                        "fall_rate",
                        "mean_fitness",
                        "mean_max_cliff_protrusion",
                        "mean_time_to_arrest",
                    )
                )
                record["delta_from_intact"] = {
                    metric: _relative_delta(
                        record[metric]["mean"] if isinstance(record.get(metric), dict) else record.get(metric),
                        None
                        if intact_summary is None
                        else (
                            intact_summary[metric]["mean"]
                            if isinstance(intact_summary.get(metric), dict)
                            else intact_summary.get(metric)
                        ),
                    )
                    for metric in delta_fields
                }
                record["delta_from_deformation_off"] = {
                    metric: _relative_delta(
                        record[metric]["mean"] if isinstance(record.get(metric), dict) else record.get(metric),
                        None
                        if off_summary is None
                        else (
                            off_summary[metric]["mean"]
                            if isinstance(off_summary.get(metric), dict)
                            else off_summary.get(metric)
                        ),
                    )
                    for metric in delta_fields
                }
            cross_surface_records[job.specimen_id].extend(dict(record) for record in surface_condition_records[job.specimen_id])
            surface_df = pd.concat(surface_frames[job.specimen_id], ignore_index=True) if surface_frames[job.specimen_id] else pd.DataFrame()
            surface_summary = {
                "surface_id": surface_id,
                "surface_label": _surface_label(surface_id),
                "condition_summaries": surface_condition_records[job.specimen_id],
                "condition_count": int(len(surface_condition_records[job.specimen_id])),
                "max_steps_by_pair": max_steps_by_pair[job.specimen_id],
                "requested_trials_per_side": (
                    None if surface_id == "double_cliff" else requested_trials_per_side
                ),
                "requested_trials_per_width": (
                    requested_trials_per_width if surface_id == "double_cliff" else None
                ),
                "condition_effective_trials_per_side": (
                    None if surface_id == "double_cliff" else dict(condition_effective_trials_per_side)
                ),
                "condition_effective_trials_per_width": (
                    dict(condition_effective_trials_per_width)
                    if surface_id == "double_cliff"
                    else None
                ),
                "condition_deterministic_rollout": dict(condition_deterministic_rollout),
            }
            surface_charts_dir = assay_dirs[job.specimen_id] / "surfaces" / surface_id / "charts"
            surface_charts_dir.mkdir(parents=True, exist_ok=True)
            if surface_id == "double_cliff":
                _chart_double_cliff_surface_family(
                    charts_dir=surface_charts_dir,
                    title_prefix=f"{_surface_label(surface_id)} Deformation",
                    category_field="condition_label",
                    records=surface_condition_records[job.specimen_id],
                )
            else:
                _chart_native_surface_family(
                    charts_dir=surface_charts_dir,
                    title_prefix=f"{_surface_label(surface_id)} Deformation",
                    category_field="condition_label",
                    records=surface_condition_records[job.specimen_id],
                )
            _finalize_assay_outputs(
                assay_dir=assay_dirs[job.specimen_id] / "surfaces" / surface_id,
                manifest={
                    "schema_version": _POST_RUN_SCHEMA_VERSION,
                    "kind": "sensing_edges_post_run_assay_manifest",
                    "assay_family": "deformation_ablation",
                    "surface_id": surface_id,
                    "surface_label": _surface_label(surface_id),
                    "specimen_id": job.specimen_id,
                    "run_id": job.run_id,
                    "conditions": list(deformation_conditions),
                    "pairs": (
                        list(post_run_cfg.native_pairs)
                        if surface_id == "native_core"
                        else list(post_run_cfg.long_distance_pairs)
                        if surface_id == "long_distance_generalization"
                        else None
                    ),
                    "widths": (
                        [int(width) for width in list(post_run_cfg.double_cliff.widths)]
                        if surface_id == "double_cliff"
                        else None
                    ),
                    "max_steps": (
                        _double_cliff_max_steps(cfg)
                        if surface_id == "double_cliff"
                        else None
                    ),
                    "max_steps_by_pair": max_steps_by_pair[job.specimen_id] if surface_id != "double_cliff" else None,
                    "requested_trials_per_side": (
                        None if surface_id == "double_cliff" else requested_trials_per_side
                    ),
                    "requested_trials_per_width": (
                        requested_trials_per_width if surface_id == "double_cliff" else None
                    ),
                    "condition_effective_trials_per_side": (
                        None if surface_id == "double_cliff" else dict(condition_effective_trials_per_side)
                    ),
                    "condition_effective_trials_per_width": (
                        dict(condition_effective_trials_per_width)
                        if surface_id == "double_cliff"
                        else None
                    ),
                    "condition_deterministic_rollout": dict(condition_deterministic_rollout),
                },
                rows=surface_df,
                aggregate={
                    "schema_version": _POST_RUN_SCHEMA_VERSION,
                    "kind": "sensing_edges_post_run_deformation_ablation_surface",
                    "specimen_id": job.specimen_id,
                    **surface_summary,
                },
                condition_records=surface_condition_records[job.specimen_id],
                specimen_records=[
                    {
                        "specimen_id": job.specimen_id,
                        "assay_family": "deformation_ablation",
                        "surface_id": surface_id,
                        "surface_label": _surface_label(surface_id),
                        "condition_count": int(len(surface_condition_records[job.specimen_id])),
                        "trial_count": int(len(surface_df)),
                    }
                ],
                inventory_records=[
                    record for record in inventory_records[job.specimen_id] if record["surface_id"] == surface_id
                ],
            )
            all_surface_frames[job.specimen_id].append(surface_df)
            surface_summaries[job.specimen_id].append(surface_summary)
    for job in jobs:
        df = pd.concat(all_surface_frames[job.specimen_id], ignore_index=True) if all_surface_frames[job.specimen_id] else pd.DataFrame()
        overall_condition_records = _overall_multi_surface_group_records(
            df,
            specimen_id=job.specimen_id,
            assay_family="deformation_ablation",
            group_field="condition_id",
            group_label_field="condition_label",
        )
        _chart_overall_multi_surface_family(
            charts_dir=assay_dirs[job.specimen_id] / "charts",
            title_prefix="Deformation Overall",
            category_field="condition_label",
            records=overall_condition_records,
        )
        aggregate = {
            "schema_version": _POST_RUN_SCHEMA_VERSION,
            "kind": "sensing_edges_post_run_deformation_ablation",
            "specimen_id": job.specimen_id,
            "conditions": list(deformation_conditions),
            "surface_summaries": surface_summaries[job.specimen_id],
            "cross_surface_condition_summaries": cross_surface_records[job.specimen_id],
            "overall_condition_summaries": overall_condition_records,
        }
        _finalize_assay_outputs(
            assay_dir=assay_dirs[job.specimen_id],
            manifest={
                "schema_version": _POST_RUN_SCHEMA_VERSION,
                "kind": "sensing_edges_post_run_assay_manifest",
                "assay_family": "deformation_ablation",
                "specimen_id": job.specimen_id,
                "run_id": job.run_id,
                "conditions": list(deformation_conditions),
                "surfaces": list(configured_surfaces),
            },
            rows=df,
            aggregate=aggregate,
            condition_records=overall_condition_records,
            specimen_records=[
                {
                    "specimen_id": job.specimen_id,
                    "assay_family": "deformation_ablation",
                    "surface_count": int(len(surface_summaries[job.specimen_id])),
                    "condition_count": int(len(cross_surface_records[job.specimen_id])),
                    "trial_count": int(len(df)),
                }
            ],
            inventory_records=inventory_records[job.specimen_id],
            cross_surface_records=cross_surface_records[job.specimen_id],
        )
    return assay_dirs


def run_batched_ablation_post_run_for_stored_specimens(
    *,
    cfg: DictConfig,
    runtime_resolution,
    run_specs: Sequence[dict[str, Any]],
    assays: set[str] | None = None,
    batch_label: str | None = None,
) -> dict[str, dict[str, str]]:
    requested = {"signaling_ablation", "deformation_ablation"} if assays is None else {
        str(assay).strip()
        for assay in assays
    }
    unsupported = sorted(requested - {"signaling_ablation", "deformation_ablation"})
    if unsupported:
        raise ValueError(f"Unsupported batched assays: {unsupported}")
    _validate_unique_batched_specimen_ids(run_specs)
    jobs = _prepare_batched_post_run_jobs(cfg=cfg, run_specs=run_specs)
    requested_assays = [
        assay_name for assay_name in ("signaling_ablation", "deformation_ablation") if assay_name in requested
    ]
    started_at_utc = datetime.now(timezone.utc).isoformat()
    assay_dirs_by_specimen = {
        job.specimen_id: {
            assay_name: str(_assay_output_dir(job.analysis_root, assay_name))
            for assay_name in requested_assays
        }
        for job in jobs
    }
    completed_assays_by_specimen = {job.specimen_id: [] for job in jobs}
    assay_progress_by_specimen: dict[str, dict[str, dict[str, Any]]] = {
        job.specimen_id: {}
        for job in jobs
    }
    current_assay_by_specimen: dict[str, str | None] = {
        job.specimen_id: None
        for job in jobs
    }
    archived_outputs_by_specimen: dict[str, dict[str, str]] = {}
    for job in jobs:
        archived_outputs_by_specimen[job.specimen_id] = _archive_existing_requested_outputs(
            analysis_root=job.analysis_root,
            requested_assays=requested_assays,
            index_path=job.analysis_index_path,
            manifest_path=job.analysis_manifest_path,
            started_at_utc=started_at_utc,
        )
    results = {
        job.specimen_id: {
            "specimen_id": job.specimen_id,
            "run_id": job.run_id,
            "analysis_root": str(job.analysis_root),
        }
        for job in jobs
    }
    for job in jobs:
        _persist_batched_job_progress(
            job=job,
            assay_dirs=assay_dirs_by_specimen[job.specimen_id],
            started_at_utc=started_at_utc,
            requested_assays=requested_assays,
            completed_assays=completed_assays_by_specimen[job.specimen_id],
            current_assay=current_assay_by_specimen[job.specimen_id],
            assay_progress=assay_progress_by_specimen[job.specimen_id],
            archived_outputs=archived_outputs_by_specimen[job.specimen_id],
            status="running",
        )
    report_root = resolve_batched_report_root(
        [job.analysis_root for job in jobs],
        batch_label=batch_label,
    )
    current_batch_assay: str | None = None
    try:
        for assay_name in requested_assays:
            current_batch_assay = str(assay_name)
            assay_started_at = datetime.now(timezone.utc).isoformat()
            for job in jobs:
                current_assay_by_specimen[job.specimen_id] = current_batch_assay
                assay_progress_by_specimen[job.specimen_id][current_batch_assay] = {
                    "status": "running",
                    "started_at_utc": assay_started_at,
                    "completed_at_utc": None,
                    "failed_at_utc": None,
                    "assay_dir": assay_dirs_by_specimen[job.specimen_id].get(current_batch_assay),
                }
                _persist_batched_job_progress(
                    job=job,
                    assay_dirs=assay_dirs_by_specimen[job.specimen_id],
                    started_at_utc=started_at_utc,
                    requested_assays=requested_assays,
                    completed_assays=completed_assays_by_specimen[job.specimen_id],
                    current_assay=current_assay_by_specimen[job.specimen_id],
                    assay_progress=assay_progress_by_specimen[job.specimen_id],
                    archived_outputs=archived_outputs_by_specimen[job.specimen_id],
                    status="running",
                )
            if current_batch_assay == "signaling_ablation":
                assay_dirs = _run_signaling_ablation_assay_batched(
                    cfg=cfg,
                    runtime_resolution=runtime_resolution,
                    jobs=jobs,
                )
            else:
                assay_dirs = _run_deformation_ablation_assay_batched(
                    cfg=cfg,
                    runtime_resolution=runtime_resolution,
                    jobs=jobs,
                )
            for specimen_id, assay_dir in assay_dirs.items():
                results[specimen_id][current_batch_assay] = str(assay_dir)
                completed_assays_by_specimen[specimen_id].append(current_batch_assay)
                assay_progress_by_specimen[specimen_id][current_batch_assay] = {
                    "status": "completed",
                    "started_at_utc": assay_progress_by_specimen[specimen_id][current_batch_assay].get(
                        "started_at_utc"
                    ),
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "failed_at_utc": None,
                    "assay_dir": str(assay_dir),
                }
                current_assay_by_specimen[specimen_id] = None
            for job in jobs:
                _persist_batched_job_progress(
                    job=job,
                    assay_dirs=assay_dirs_by_specimen[job.specimen_id],
                    started_at_utc=started_at_utc,
                    requested_assays=requested_assays,
                    completed_assays=completed_assays_by_specimen[job.specimen_id],
                    current_assay=current_assay_by_specimen[job.specimen_id],
                    assay_progress=assay_progress_by_specimen[job.specimen_id],
                    archived_outputs=archived_outputs_by_specimen[job.specimen_id],
                    status="running",
                )
            _write_batched_assay_report(
                assay_name=current_batch_assay,
                jobs=jobs,
                assay_dirs=assay_dirs,
                report_root=report_root,
            )
            current_batch_assay = None
    except Exception as exc:
        error_payload = _exception_payload(exc)
        failed_at = datetime.now(timezone.utc).isoformat()
        for job in jobs:
            if current_batch_assay is not None:
                assay_progress_by_specimen[job.specimen_id][current_batch_assay] = {
                    "status": "failed",
                    "started_at_utc": assay_progress_by_specimen[job.specimen_id]
                    .get(current_batch_assay, {})
                    .get("started_at_utc"),
                    "completed_at_utc": None,
                    "failed_at_utc": failed_at,
                    "assay_dir": assay_dirs_by_specimen[job.specimen_id].get(current_batch_assay),
                    "error": error_payload,
                }
            current_assay_by_specimen[job.specimen_id] = None
            _persist_batched_job_progress(
                job=job,
                assay_dirs=assay_dirs_by_specimen[job.specimen_id],
                started_at_utc=started_at_utc,
                requested_assays=requested_assays,
                completed_assays=completed_assays_by_specimen[job.specimen_id],
                current_assay=current_assay_by_specimen[job.specimen_id],
                assay_progress=assay_progress_by_specimen[job.specimen_id],
                archived_outputs=archived_outputs_by_specimen[job.specimen_id],
                status="failed",
                failed_assay=current_batch_assay,
                error_payload=error_payload,
            )
        raise
    for job in jobs:
        _persist_batched_job_progress(
            job=job,
            assay_dirs=assay_dirs_by_specimen[job.specimen_id],
            started_at_utc=started_at_utc,
            requested_assays=requested_assays,
            completed_assays=completed_assays_by_specimen[job.specimen_id],
            current_assay=None,
            assay_progress=assay_progress_by_specimen[job.specimen_id],
            archived_outputs=archived_outputs_by_specimen[job.specimen_id],
            status="completed",
        )
    for specimen_id in results:
        results[specimen_id]["batch_report_root"] = str(report_root)
    return results


def _write_batched_assay_report(
    *,
    assay_name: str,
    jobs: Sequence[BatchedPostRunJob],
    assay_dirs: dict[str, Path],
    report_root: Path,
) -> None:
    report_dir = report_root / assay_name
    report_dir.mkdir(parents=True, exist_ok=True)
    combined_rows: list[pd.DataFrame] = []
    combined_condition_records: list[pd.DataFrame] = []
    combined_cross_surface_records: list[pd.DataFrame] = []
    manifest_jobs: list[dict[str, Any]] = []
    for job in jobs:
        assay_dir = assay_dirs[job.specimen_id]
        manifest_jobs.append(
            {
                "specimen_id": job.specimen_id,
                "run_id": job.run_id,
                "run_dir": str(job.run_dir),
                "assay_dir": str(assay_dir),
            }
        )
        per_episode_path = assay_dir / "per_episode.csv"
        if per_episode_path.exists():
            df = pd.read_csv(per_episode_path)
            df["specimen_id"] = job.specimen_id
            df["run_id"] = job.run_id
            df["run_dir_name"] = job.run_dir.name
            combined_rows.append(df)
        condition_summary_path = assay_dir / "stats" / "condition_summary.csv"
        if condition_summary_path.exists():
            df = pd.read_csv(condition_summary_path)
            df["specimen_id"] = job.specimen_id
            df["run_id"] = job.run_id
            df["run_dir_name"] = job.run_dir.name
            combined_condition_records.append(df)
        cross_surface_path = assay_dir / "stats" / "cross_surface_summary.csv"
        if cross_surface_path.exists():
            df = pd.read_csv(cross_surface_path)
            df["specimen_id"] = job.specimen_id
            df["run_id"] = job.run_id
            df["run_dir_name"] = job.run_dir.name
            combined_cross_surface_records.append(df)
    manifest_payload = {
        "schema_version": _POST_RUN_SCHEMA_VERSION,
        "kind": "sensing_edges_post_run_batched_report",
        "assay_family": assay_name,
        "specimen_count": int(len(jobs)),
        "jobs": manifest_jobs,
    }
    _write_json(report_dir / "manifest.json", manifest_payload)
    if combined_rows:
        all_rows = pd.concat(combined_rows, ignore_index=True)
        _write_dataframe_bundle(all_rows, report_dir, stem="all_per_episode")
        pooled_condition_records = _overall_multi_surface_group_records(
            all_rows,
            specimen_id="__batch__",
            assay_family=assay_name,
            group_field="condition_id",
            group_label_field="condition_label",
        )
        pooled_df = _records_dataframe(pooled_condition_records)
        if not pooled_df.empty:
            specimen_counts = (
                all_rows.groupby("condition_id")["specimen_id"].nunique().rename("specimen_count").reset_index()
            )
            pooled_df = pooled_df.merge(specimen_counts, on="condition_id", how="left")
        _write_dataframe_bundle(pooled_df, report_dir / "stats", stem="pooled_condition_summary")
    if combined_condition_records:
        _write_dataframe_bundle(
            pd.concat(combined_condition_records, ignore_index=True),
            report_dir / "stats",
            stem="all_condition_summary",
        )
    if combined_cross_surface_records:
        _write_dataframe_bundle(
            pd.concat(combined_cross_surface_records, ignore_index=True),
            report_dir / "stats",
            stem="all_cross_surface_summary",
        )


def _double_cliff_rows(
    *,
    episode_results,
    traces,
    envs: list[Any],
    entries: list[TerrainEntry],
    specimen_id: str,
    run_id: str,
    max_steps: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    episode_count = int(np.asarray(episode_results.steps_alive).shape[0])
    for index in range(episode_count):
        result = jax.tree_util.tree_map(lambda array: np.asarray(array[index]), episode_results)
        trace = jax.tree_util.tree_map(lambda array: np.asarray(array[index]), traces)
        env_data = envs[index]
        metadata = dict(entries[index].metadata or {})
        width = int(metadata["assay_width"])
        steps_alive = int(result.steps_alive)
        valid_positions = np.asarray(trace.positions, dtype=np.float32)[:steps_alive] if steps_alive > 0 else np.zeros((0, 0, 2), dtype=np.float32)
        if steps_alive > 0:
            com_x, min_x, front_x = _trace_kinematics(valid_positions, np.asarray(env_data.robot_point_mask))
        else:
            com_x = np.zeros((0,), dtype=np.float32)
            min_x = np.zeros((0,), dtype=np.float32)
            front_x = np.zeros((0,), dtype=np.float32)
        left_frontier = float(int(metadata["platform_start_x"])) * float(CELL_SIZE)
        right_frontier = float(int(metadata["platform_end_x"]) + 1) * float(CELL_SIZE)
        left_clearance = min_x - left_frontier if steps_alive > 0 else np.zeros((0,), dtype=np.float32)
        right_clearance = right_frontier - front_x if steps_alive > 0 else np.zeros((0,), dtype=np.float32)
        left_vicinity = left_clearance <= _CLIFF_VICINITY_BAND
        right_vicinity = right_clearance <= _CLIFF_VICINITY_BAND
        first_left = int(np.argmax(left_vicinity) + 1) if steps_alive > 0 and bool(np.any(left_vicinity)) else None
        first_right = int(np.argmax(right_vicinity) + 1) if steps_alive > 0 and bool(np.any(right_vicinity)) else None
        if first_left is None and first_right is None:
            first_approached_side = "none"
            time_to_first_edge_approach = None
        elif first_right is None or (first_left is not None and first_left < first_right):
            first_approached_side = "left"
            time_to_first_edge_approach = first_left
        else:
            first_approached_side = "right"
            time_to_first_edge_approach = first_right
        turn_motion_threshold = float(result.turn_motion_threshold)
        arrest_step = _find_arrest_step(
            com_x=com_x,
            init_com_x=float(result.init_com_x),
            turn_motion_threshold=turn_motion_threshold,
            first_cliff_vicinity_step=time_to_first_edge_approach,
        )
        if arrest_step is None or steps_alive <= 0:
            first_arrest_side = "none"
        else:
            arrest_idx = max(min(arrest_step - 1, steps_alive - 1), 0)
            first_arrest_side = "left" if float(left_clearance[arrest_idx]) <= float(right_clearance[arrest_idx]) else "right"
        if steps_alive <= 0:
            side_switch_count = 0
        else:
            dominant_side = np.where(left_clearance <= right_clearance, "left", "right")
            switchable = left_vicinity | right_vicinity
            dominant_active = dominant_side[switchable]
            side_switch_count = int(np.sum(dominant_active[1:] != dominant_active[:-1])) if dominant_active.size > 1 else 0
        final_side = (
            "left"
            if steps_alive > 0 and float(left_clearance[-1]) <= float(right_clearance[-1])
            else "right"
        )
        rows.append(
            {
                "run_id": str(run_id),
                "specimen_id": str(specimen_id),
                "trial_id": f"double_cliff_{index:04d}",
                "width": int(width),
                "first_approached_side": first_approached_side,
                "first_arrest_side": first_arrest_side,
                "fell_left": bool(result.cliff_fell) and final_side == "left",
                "fell_right": bool(result.cliff_fell) and final_side == "right",
                "survived": not bool(result.cliff_fell),
                "steps_alive": int(steps_alive),
                "termination_reason": _episode_termination_reason(
                    goal_touched=False,
                    touched_death_plane=bool(result.touched_death_plane),
                    went_airborne=bool(result.went_airborne),
                    left_exit=False,
                    steps_alive=int(steps_alive),
                    max_steps=int(max_steps),
                ),
                "time_to_first_edge_approach": time_to_first_edge_approach,
                "time_to_first_arrest": arrest_step,
                "side_switch_count": side_switch_count,
                "min_left_clearance": None if steps_alive <= 0 else float(np.min(left_clearance)),
                "min_right_clearance": None if steps_alive <= 0 else float(np.min(right_clearance)),
            }
        )
    return rows


def _double_cliff_summary_rows(
    *,
    episode_results,
    entries: list[TerrainEntry],
    specimen_id: str,
    run_id: str,
    max_steps: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    episode_count = int(np.asarray(episode_results.steps_alive).shape[0])
    for index in range(episode_count):
        result = jax.tree_util.tree_map(lambda array: np.asarray(array[index]), episode_results)
        metadata = dict(entries[index].metadata or {})
        width = int(metadata["assay_width"])
        fell = bool(result.cliff_fell)
        rows.append(
            {
                "run_id": str(run_id),
                "specimen_id": str(specimen_id),
                "trial_id": f"double_cliff_{index:04d}",
                "width": int(width),
                "first_approached_side": None,
                "first_arrest_side": None,
                "fell_left": None,
                "fell_right": None,
                "fell": fell,
                "survived": not fell,
                "steps_alive": int(result.steps_alive),
                "termination_reason": _episode_termination_reason(
                    goal_touched=False,
                    touched_death_plane=bool(result.touched_death_plane),
                    went_airborne=bool(result.went_airborne),
                    left_exit=False,
                    steps_alive=int(result.steps_alive),
                    max_steps=int(max_steps),
                ),
                "time_to_first_edge_approach": None,
                "time_to_first_arrest": None,
                "side_switch_count": None,
                "min_left_clearance": None,
                "min_right_clearance": None,
            }
        )
    return rows


def _run_double_cliff_assay(
    *,
    cfg: DictConfig,
    runtime_resolution,
    specimen_dir: Path,
    specimen_id: str,
    genotype: np.ndarray,
    selection_row: dict[str, Any],
    run_id: str,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    del selection_row
    post_run_cfg = cfg.eval.post_run
    phase_cfg = _double_cliff_phase_cfg(cfg)
    double_cliff_cfg = post_run_cfg.double_cliff
    assay_dir = specimen_dir / "double_cliff"
    effective_trials_per_width = _effective_trials_per_width(cfg)
    entries = _build_double_cliff_entries(
        phase_cfg,
        [int(width) for width in list(double_cliff_cfg.widths)],
        plateau_width=int(double_cliff_cfg.plateau_width),
        trials_per_width=int(effective_trials_per_width),
    )
    episode_results, traces, envs, fitnesses = _run_trace_eval(
        phase_cfg=phase_cfg,
        runtime_resolution=runtime_resolution,
        genotype=genotype,
        entries=entries,
        workload="trace",
        seed=_stable_seed(run_id, specimen_id, "double_cliff"),
    )
    rows = pd.DataFrame(
        _double_cliff_rows(
            episode_results=episode_results,
            traces=traces,
            envs=envs,
            entries=entries,
            specimen_id=specimen_id,
            run_id=run_id,
            max_steps=_double_cliff_max_steps(cfg),
        )
    )
    rows["surface_id"] = "double_cliff"
    rows["surface_label"] = _surface_label("double_cliff")
    rows["surface_max_steps"] = _double_cliff_max_steps(cfg)
    rows["fell"] = rows["fell_left"].astype(bool) | rows["fell_right"].astype(bool)
    _render_selected_candidates(
        cfg=cfg,
        output_dir=assay_dir / "renders",
        candidates=_double_cliff_render_candidates(
            episode_results=episode_results,
            traces=traces,
            envs=envs,
            entries=entries,
            fitnesses=fitnesses,
        ),
        manifest_kind="sensing_edges_post_run_render_manifest",
    )
    summaries = _condition_summary_records(
        rows,
        group_col="width",
        metrics={
            "survival_rate": "survived",
            "mean_side_switch_count": "side_switch_count",
        },
        seed_prefix=_stable_seed(run_id, specimen_id, "double_cliff_summary"),
        success_col="survived",
        fall_col="fell",
    )
    left_bias = float(np.mean((rows["first_approached_side"] == "left").astype(float))) if not rows.empty else 0.0
    right_bias = float(np.mean((rows["first_approached_side"] == "right").astype(float))) if not rows.empty else 0.0
    symmetricity_score = 1.0 - abs(left_bias - right_bias)
    charts_dir = assay_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    _chart_bar(
        categories=[str(record["width"]) for record in summaries],
        values=[float(record["survival_rate"]["mean"] or 0.0) for record in summaries],
        ci_bounds=[record["survival_rate"]["bootstrap_ci95"] for record in summaries],
        title="Survival Rate By Width",
        ylabel="survival rate",
        output_path=charts_dir / "survival_by_width.png",
    )
    _chart_bar(
        categories=["left", "right"],
        values=[left_bias, right_bias],
        ci_bounds=None,
        title="First Side Bias",
        ylabel="fraction",
        output_path=charts_dir / "first_side_bias.png",
    )
    _chart_bar(
        categories=[str(record["width"]) for record in summaries],
        values=[float(record["mean_side_switch_count"]["mean"] or 0.0) for record in summaries],
        ci_bounds=[record["mean_side_switch_count"]["bootstrap_ci95"] for record in summaries],
        title="Side Switch Count By Width",
        ylabel="switch count",
        output_path=charts_dir / "side_switch_count_by_width.png",
    )
    _chart_outcome_breakdown(
        categories=[str(record["width"]) for record in summaries],
        outcome_breakdowns=[record["outcome_breakdown"] for record in summaries],
        title="Double-Cliff Outcome Composition",
        output_path=charts_dir / "outcome_breakdown.png",
    )
    surface_record = _double_cliff_surface_specimen_record(
        specimen_id=specimen_id,
        assay_family="double_cliff",
        surface_id="double_cliff",
        rows=rows,
        extra_fields={
            "plateau_width": int(double_cliff_cfg.plateau_width),
            "widths": [int(width) for width in list(double_cliff_cfg.widths)],
            "requested_trials_per_width": int(double_cliff_cfg.trials_per_width),
            "effective_trials_per_width": int(effective_trials_per_width),
            "deterministic_rollout": bool(effective_trials_per_width == 1),
            "max_steps": _double_cliff_max_steps(cfg),
        },
    )
    aggregate = {
        "schema_version": _POST_RUN_SCHEMA_VERSION,
        "kind": "sensing_edges_post_run_double_cliff",
        "specimen_id": specimen_id,
        "surface_id": "double_cliff",
        "condition_summaries": summaries,
        "specimen_summary": surface_record,
    }
    condition_records = []
    inventory_records = []
    for record in summaries:
        condition_record = dict(record)
        condition_record.update(
            {
                "specimen_id": specimen_id,
                "assay_family": "double_cliff",
                "surface_id": "double_cliff",
                "surface_label": _surface_label("double_cliff"),
            }
        )
        condition_records.append(condition_record)
        inventory_records.append(
            _surface_inventory_record(
                assay_family="double_cliff",
                surface_id="double_cliff",
                subgroup_type="width",
                subgroup_id=str(record["width"]),
                surface_dir=assay_dir,
            )
        )
    _finalize_assay_outputs(
        assay_dir=assay_dir,
        manifest={
            "schema_version": _POST_RUN_SCHEMA_VERSION,
            "kind": "sensing_edges_post_run_assay_manifest",
            "assay_family": "double_cliff",
            "specimen_id": specimen_id,
            "run_id": run_id,
            "surface_label": _surface_label("double_cliff"),
            "max_steps": _double_cliff_max_steps(cfg),
            "plateau_width": int(double_cliff_cfg.plateau_width),
            "widths": [int(width) for width in list(double_cliff_cfg.widths)],
            "requested_trials_per_width": int(double_cliff_cfg.trials_per_width),
            "effective_trials_per_width": int(effective_trials_per_width),
            "deterministic_rollout": bool(effective_trials_per_width == 1),
        },
        rows=rows,
        aggregate=aggregate,
        condition_records=condition_records,
        specimen_records=[surface_record],
        inventory_records=inventory_records,
    )
    return rows, aggregate, assay_dir


def _run_tau_analysis(
    *,
    cfg: DictConfig,
    specimen_dir: Path,
    specimen_id: str,
    genotype: np.ndarray,
) -> Path:
    params = ctrnn_flat_to_params(
        jax.numpy.asarray(genotype, dtype=jax.numpy.float32),
        ctrnn_hidden=int(cfg.nca.ctrnn_hidden),
        ctrnn_grid_hidden=int(cfg.nca.ctrnn_grid_hidden),
        gasnet_mode=str(cfg.nca.get("gasnet_mode", "off")),
        ou_enabled=bool(resolved_noise_settings(cfg)["ou_enabled"]),
        growth_enabled=False,
        active_comm_filters=active_comm_filters_from_cfg(cfg),
    )
    tau_values = np.asarray(
        float(cfg.nca.ctrnn_tau_min)
        + (
            float(cfg.nca.ctrnn_tau_max) - float(cfg.nca.ctrnn_tau_min)
        )
        * (1.0 / (1.0 + np.exp(-np.asarray(params.log_tau, dtype=np.float32)))),
        dtype=np.float32,
    )
    sorted_tau = np.sort(tau_values)
    assay_dir = specimen_dir / "tau"
    charts_dir = assay_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    _write_json(assay_dir / "tau_values.json", {"specimen_id": specimen_id, "tau_values": tau_values.tolist()})
    summary_payload = {
        "specimen_id": specimen_id,
        "sorted_tau_values": sorted_tau.tolist(),
        "median_tau": float(np.median(tau_values)),
        "tau_spread": {
            "std": float(np.std(tau_values)),
            "iqr": float(np.quantile(tau_values, 0.75) - np.quantile(tau_values, 0.25)),
            "max_min_ratio": (
                None
                if float(np.min(tau_values)) <= 0.0
                else float(np.max(tau_values)) / float(np.min(tau_values))
            ),
        },
    }
    _write_json(assay_dir / "tau_summary.json", summary_payload)
    _write_json(
        assay_dir / "manifest.json",
        {
            "schema_version": _POST_RUN_SCHEMA_VERSION,
            "kind": "sensing_edges_post_run_tau_manifest",
            "specimen_id": specimen_id,
        },
    )
    _write_json(assay_dir / "summary.json", summary_payload)
    _write_json(assay_dir / "aggregate.json", summary_payload)
    _write_stats_tables(
        assay_dir=assay_dir,
        specimen_records=[summary_payload],
        inventory_records=[
            {
                "assay_family": "tau",
                "surface_id": "tau",
                "surface_label": "Tau",
                "subgroup_type": "summary",
                "subgroup_id": specimen_id,
                "surface_dir": str(assay_dir),
                "target_dir": str(assay_dir),
                "summary_path": str(assay_dir / "summary.json"),
                "aggregate_path": str(assay_dir / "aggregate.json"),
                "per_episode_csv": None,
                "per_episode_parquet": None,
                "render_manifest_path": None,
                "charts_dir": str(charts_dir),
            }
        ],
    )
    _chart_line(
        categories=[str(index) for index in range(1, len(sorted_tau) + 1)],
        values=[float(value) for value in sorted_tau],
        title="Sorted Tau Profile",
        ylabel="tau",
        output_path=charts_dir / "sorted_tau_profile.png",
    )
    _chart_histogram(
        values=tau_values,
        title="Tau Histogram",
        xlabel="tau",
        output_path=charts_dir / "tau_histogram.png",
    )
    return assay_dir


def _pca_bundle_additional_native_cohorts(trace_cfg: DictConfig) -> list[dict[str, Any]]:
    raw_cohorts = list(trace_cfg.get("additional_native_cohorts", []))
    cohorts: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    target_pair_cfg = trace_cfg.target_pair
    default_goal_distance = int(target_pair_cfg.goal_distance)
    default_cliff_distance = int(target_pair_cfg.cliff_distance)
    for raw in raw_cohorts:
        payload = (
            OmegaConf.to_container(raw, resolve=True)
            if OmegaConf.is_config(raw)
            else dict(raw)
        )
        cohort_id = str(payload.get("cohort_id", "")).strip()
        if not cohort_id:
            raise ValueError("eval.post_run.trace.additional_native_cohorts entries require cohort_id")
        if cohort_id in seen_ids:
            raise ValueError(f"Duplicate PCA cohort_id configured: {cohort_id}")
        seen_ids.add(cohort_id)
        cohorts.append(
            {
                "cohort_id": cohort_id,
                "cohort_label": str(payload.get("cohort_label", cohort_id)).strip() or cohort_id,
                "goal_distance": int(payload.get("goal_distance", default_goal_distance)),
                "cliff_distance": int(payload.get("cliff_distance", default_cliff_distance)),
                "deformation_condition": str(payload.get("deformation_condition", "intact")),
                "signaling_condition": str(payload.get("signaling_condition", "intact")),
            }
        )
    return cohorts


def _run_pca_bundle_cohort(
    *,
    cfg: DictConfig,
    runtime_resolution,
    bundle_dir: Path,
    trace_cfg: DictConfig,
    specimen_id: str,
    genotype: np.ndarray,
    run_id: str,
    cohort_id: str,
    cohort_label: str,
    goal_distance: int,
    cliff_distance: int,
    deformation_condition: str = "intact",
    signaling_condition: str = "intact",
) -> dict[str, Any]:
    phase_cfg = _native_phase_cfg(
        cfg,
        active_deformation_condition=deformation_condition,
        active_signaling_condition=signaling_condition,
    )
    requested_trials_per_side = int(trace_cfg.requested_trials_per_side)
    effective_trials_per_side = _effective_trace_trials_per_side(
        cfg,
        deformation_condition=deformation_condition,
    )
    entries = _build_targeted_trace_entries(
        phase_cfg,
        goal_distance=int(goal_distance),
        cliff_distance=int(cliff_distance),
        trials_per_side=effective_trials_per_side,
        goal_sides=_configured_goal_sides(cfg, trace_cfg=trace_cfg),
    )
    episode_batch_size = _controller_trace_episode_batch_size(
        runtime_resolution=runtime_resolution,
        requested_batch_size=int(trace_cfg.get("episode_batch_size", 4)),
        total_entries=len(entries),
    )
    episode_results, controller_traces, _envs = _run_controller_trace_eval(
        phase_cfg=phase_cfg,
        runtime_resolution=runtime_resolution,
        genotype=genotype,
        entries=entries,
        seed=_stable_seed(run_id, specimen_id, "pca_bundle", cohort_id),
        episode_batch_size=episode_batch_size,
    )
    ctrnn_chunks: list[np.ndarray] = []
    nca_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    label_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    distance_pair = _pair_label(goal_distance, cliff_distance)
    sense_act_phases = int(phase_cfg.sim.sense_act_phases_per_env_step)
    nca_steps_per_phase = int(phase_cfg.nca.nca_steps_per_phase)
    max_steps = int(phase_cfg.train.max_steps)
    for index, entry in enumerate(entries):
        metadata = dict(entry.metadata or {})
        side = str(metadata.get("goal_side", metadata.get("assay_goal_side", "unknown")))
        goal_direction = int(metadata.get("goal_direction", _side_code(side)))
        trial_repeat_index = int(metadata.get("trial_repeat_index", 0))
        trial_id = _controller_trace_trial_id(
            goal_distance=goal_distance,
            cliff_distance=cliff_distance,
            side=side,
            trial_repeat_index=trial_repeat_index,
        )
        active_mask = np.asarray(controller_traces.active[index], dtype=bool)
        active_indices = np.argwhere(active_mask)
        nca_state = np.asarray(controller_traces.nca_state[index], dtype=np.float32)[active_mask]
        ctrnn_carry = np.asarray(controller_traces.ctrnn_carry[index], dtype=np.float32)[active_mask]
        action_outputs = np.asarray(controller_traces.action_outputs[index], dtype=np.float32)[active_mask]
        com_x = np.asarray(controller_traces.com_x[index], dtype=np.float32)[active_mask]
        back_x = np.asarray(controller_traces.back_x[index], dtype=np.float32)[active_mask]
        front_x = np.asarray(controller_traces.front_x[index], dtype=np.float32)[active_mask]
        phase_unique_mask = (
            active_indices[:, 2] == 0
            if active_indices.size
            else np.zeros((0,), dtype=bool)
        )
        phase_com_x = com_x[phase_unique_mask]
        init_com_x = float(com_x[0]) if com_x.size else float(np.asarray(episode_results.init_com_x[index]))
        init_front_x = float(front_x[0]) if front_x.size else float(np.asarray(episode_results.init_front_x[index]))
        init_back_x = float(back_x[0]) if back_x.size else float(np.asarray(episode_results.final_back_x[index]))
        first_peak_direction, first_peak_x, first_peak_goalward_x = _first_peak_summary(
            com_x=phase_com_x,
            init_com_x=init_com_x,
            goal_direction=goal_direction,
            turn_motion_threshold=float(np.asarray(episode_results.turn_motion_threshold[index])),
        )
        cliff_side = str(metadata.get("cliff_side", "right"))
        cliff_edge_x = int(metadata.get("terminal_cliff_edge_x", metadata.get("cliff_edge_x", entry.cliff_x)))
        task_code_geom, reached_cliff_frontier, _frontier_idx = _task_code_geom(
            back_x=back_x,
            front_x=front_x,
            goal_direction=goal_direction,
            cliff_side=cliff_side,
            cliff_edge_x=cliff_edge_x,
            goal_block_x=int(metadata.get("goal_block_x", -1)),
            init_back_x=init_back_x,
            init_front_x=init_front_x,
        )
        goal_touched = bool(np.asarray(episode_results.goal_touched[index]))
        fell_flag = bool(np.asarray(episode_results.cliff_fell[index]))
        reversed_pre_ledge = first_peak_direction == "cliff_first" and not reached_cliff_frontier
        primary_candidate = (
            goal_touched
            and (not fell_flag)
            and first_peak_direction == "cliff_first"
            and reached_cliff_frontier
            and (not reversed_pre_ledge)
        )
        successful_cliff_first = goal_touched and (not fell_flag) and first_peak_direction == "cliff_first"
        steps_alive = int(active_indices[-1, 0] + 1) if active_indices.size else 0
        phase_steps_alive = int(np.sum(phase_unique_mask)) if phase_unique_mask.size else 0
        controller_step_count = int(nca_state.shape[0])
        ctrnn_chunks.append(ctrnn_carry)
        nca_chunks.append(nca_state)
        action_chunks.append(action_outputs)
        for sample_index, (env_idx, phase_idx, micro_idx) in enumerate(active_indices):
            label_rows.append(
                {
                    "specimen_id": specimen_id,
                    "cohort_id": cohort_id,
                    "cohort_label": cohort_label,
                    "surface_id": "native_core",
                    "deformation_condition": str(deformation_condition),
                    "signaling_condition": str(signaling_condition),
                    "trial_id": trial_id,
                    "distance_pair": distance_pair,
                    "side": side,
                    "goal_direction": goal_direction,
                    "env_step": int(env_idx + 1),
                    "phase_index": int(phase_idx),
                    "micro_index": int(micro_idx),
                    "controller_tick": int(sample_index),
                    "com_x": float(com_x[sample_index]),
                    "front_x": float(front_x[sample_index]),
                    "back_x": float(back_x[sample_index]),
                    "task_code_geom": float(task_code_geom[sample_index]) if task_code_geom.size else 0.0,
                    "goal_touched": goal_touched,
                    "fell_flag": fell_flag,
                    "is_terminal_sample": bool(sample_index == (controller_step_count - 1)),
                }
            )
        trial_rows.append(
            {
                "specimen_id": specimen_id,
                "cohort_id": cohort_id,
                "cohort_label": cohort_label,
                "surface_id": "native_core",
                "surface_label": _surface_label("native_core"),
                "deformation_condition": str(deformation_condition),
                "signaling_condition": str(signaling_condition),
                "trial_id": trial_id,
                "distance_pair": distance_pair,
                "goal_distance": goal_distance,
                "cliff_distance": cliff_distance,
                "side": side,
                "goal_direction": goal_direction,
                "trial_repeat_index": trial_repeat_index,
                "init_com_x": init_com_x,
                "init_front_x": init_front_x,
                "init_back_x": init_back_x,
                "final_com_x": float(np.asarray(episode_results.final_com_x[index])),
                "final_front_x": float(np.asarray(episode_results.final_front_x[index])),
                "final_back_x": float(np.asarray(episode_results.final_back_x[index])),
                "steps_alive": steps_alive,
                "phase_steps_alive": phase_steps_alive,
                "controller_step_count": controller_step_count,
                "goal_touched": goal_touched,
                "fell_flag": fell_flag,
                "touched_death_plane": bool(np.asarray(episode_results.touched_death_plane[index])),
                "went_airborne": bool(np.asarray(episode_results.went_airborne[index])),
                "com_x_went_negative": bool(np.asarray(episode_results.com_x_went_negative[index])),
                "termination_reason": _termination_reason(
                    goal_touched=goal_touched,
                    touched_death_plane=bool(np.asarray(episode_results.touched_death_plane[index])),
                    went_airborne=bool(np.asarray(episode_results.went_airborne[index])),
                    steps_alive=phase_steps_alive,
                    max_steps=max_steps,
                    sense_act_phases=sense_act_phases,
                ),
                "turn_motion_threshold": float(np.asarray(episode_results.turn_motion_threshold[index])),
                "first_peak_x": first_peak_x,
                "first_peak_goalward_x": first_peak_goalward_x,
                "first_peak_direction": first_peak_direction,
                "reached_cliff_frontier": reached_cliff_frontier,
                "reversed_pre_ledge": reversed_pre_ledge,
                "task_code_geom_peak": float(np.max(task_code_geom)) if task_code_geom.size else 0.0,
                "_primary_candidate": primary_candidate,
                "_successful_cliff_first": successful_cliff_first,
            }
        )
    trial_df = pd.DataFrame(trial_rows)
    requested_trials_by_side = {side_name: int(requested_trials_per_side) for side_name in _GOAL_SIDES}
    effective_trials_by_side = {side_name: int(effective_trials_per_side) for side_name in _GOAL_SIDES}
    side_summary: dict[str, dict[str, Any]] = {}
    for side_name in _GOAL_SIDES:
        side_rows = trial_df[trial_df["side"] == side_name]
        side_summary[side_name] = {
            "requested_trial_count": int(requested_trials_per_side),
            "effective_trial_count": int(effective_trials_per_side),
            "realized_trial_count": int(len(side_rows)),
            "successful_no_fall_count": int(
                np.sum(np.asarray(side_rows["goal_touched"] & (~side_rows["fell_flag"]), dtype=np.int32))
            ),
            "successful_cliff_first_count": int(
                np.sum(np.asarray(side_rows["_successful_cliff_first"], dtype=np.int32))
            ),
            "primary_candidate_count": int(
                np.sum(np.asarray(side_rows["_primary_candidate"], dtype=np.int32))
            ),
        }
    left_primary = side_summary["left"]["primary_candidate_count"]
    right_primary = side_summary["right"]["primary_candidate_count"]
    left_cliff_first = side_summary["left"]["successful_cliff_first_count"]
    right_cliff_first = side_summary["right"]["successful_cliff_first_count"]
    if left_primary > right_primary:
        selected_side = "left"
        side_selection_reason = "primary_candidate_count"
    elif right_primary > left_primary:
        selected_side = "right"
        side_selection_reason = "primary_candidate_count"
    elif left_cliff_first > right_cliff_first:
        selected_side = "left"
        side_selection_reason = "successful_cliff_first_count"
    elif right_cliff_first > left_cliff_first:
        selected_side = "right"
        side_selection_reason = "successful_cliff_first_count"
    else:
        selected_side = "right"
        side_selection_reason = "tie_break_right"
    trial_df["selected_for_primary_pca"] = trial_df["_primary_candidate"] & (trial_df["side"] == selected_side)
    selected_trial_ids = [
        str(trial_id)
        for trial_id in trial_df.loc[trial_df["selected_for_primary_pca"], "trial_id"].tolist()
    ]
    goal_first_count = int(
        np.sum(
            np.asarray(
                (trial_df["side"] == selected_side)
                & trial_df["goal_touched"]
                & (~trial_df["fell_flag"])
                & (trial_df["first_peak_direction"] == "goal_first"),
                dtype=np.int32,
            )
        )
    )
    cliff_first_count = int(
        np.sum(
            np.asarray(
                (trial_df["side"] == selected_side)
                & trial_df["goal_touched"]
                & (~trial_df["fell_flag"])
                & (trial_df["first_peak_direction"] == "cliff_first"),
                dtype=np.int32,
            )
        )
    )
    none_count = int(
        np.sum(
            np.asarray(
                (trial_df["side"] == selected_side)
                & trial_df["goal_touched"]
                & (~trial_df["fell_flag"])
                & (trial_df["first_peak_direction"] == "none"),
                dtype=np.int32,
            )
        )
    )
    trial_df = trial_df.drop(columns=["_primary_candidate", "_successful_cliff_first"])
    bundle_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        bundle_dir / "ctrnn_states.npz",
        states=np.concatenate(ctrnn_chunks, axis=0) if ctrnn_chunks else np.zeros((0,), dtype=np.float32),
    )
    np.savez_compressed(
        bundle_dir / "nca_states.npz",
        states=np.concatenate(nca_chunks, axis=0) if nca_chunks else np.zeros((0,), dtype=np.float32),
    )
    np.savez_compressed(
        bundle_dir / "actions.npz",
        actions=np.concatenate(action_chunks, axis=0) if action_chunks else np.zeros((0,), dtype=np.float32),
    )
    label_df = pd.DataFrame(label_rows)
    label_df.to_csv(bundle_dir / "trace_labels.csv", index=False)
    trial_df.to_csv(bundle_dir / "trial_manifest.csv", index=False)
    cohort_summary = {
        "schema_version": _POST_RUN_SCHEMA_VERSION,
        "kind": "sensing_edges_post_run_pca_bundle_cohort",
        "specimen_id": specimen_id,
        "cohort_id": cohort_id,
        "cohort_label": cohort_label,
        "surface_id": "native_core",
        "surface_label": _surface_label("native_core"),
        "deformation_condition": str(deformation_condition),
        "signaling_condition": str(signaling_condition),
        "distance_pair": distance_pair,
        "goal_distance": goal_distance,
        "cliff_distance": cliff_distance,
        "requested_trials_per_side": requested_trials_per_side,
        "requested_trials_by_side": requested_trials_by_side,
        "effective_trials_per_side": int(effective_trials_per_side),
        "effective_trials_by_side": effective_trials_by_side,
        "deterministic_rollout": bool(effective_trials_per_side == 1),
        "side_summary": side_summary,
        "selected_side": selected_side,
        "side_selection_reason": side_selection_reason,
        "successful_no_fall_count_selected_side": side_summary[selected_side]["successful_no_fall_count"],
        "goal_first_count_selected_side": goal_first_count,
        "cliff_first_count_selected_side": cliff_first_count,
        "none_count_selected_side": none_count,
        "selected_trial_ids": selected_trial_ids,
        "selected_trial_count": int(len(selected_trial_ids)),
    }
    _write_json(bundle_dir / "cohort_summary.json", cohort_summary)
    trace_manifest = {
        "schema_version": _POST_RUN_SCHEMA_VERSION,
        "kind": "sensing_edges_post_run_pca_bundle",
        "specimen_id": specimen_id,
        "cohort_id": cohort_id,
        "cohort_label": cohort_label,
        "surface_id": "native_core",
        "surface_label": _surface_label("native_core"),
        "deformation_condition": str(deformation_condition),
        "signaling_condition": str(signaling_condition),
        "sample_resolution": str(trace_cfg.get("sample_resolution", "controller_step")),
        "distance_pair": distance_pair,
        "goal_distance": goal_distance,
        "cliff_distance": cliff_distance,
        "requested_trials_per_side": requested_trials_per_side,
        "requested_trials_by_side": requested_trials_by_side,
        "effective_trials_per_side": int(effective_trials_per_side),
        "effective_trials_by_side": effective_trials_by_side,
        "realized_trial_count": int(len(entries)),
        "logged_step_count": int(len(label_df)),
        "selected_side": selected_side,
        "sense_act_phases": sense_act_phases,
        "nca_steps_per_phase": nca_steps_per_phase,
        "ctrnn_hidden": int(cfg.nca.ctrnn_hidden),
        "nca_channels": int(nca_chunks[0].shape[-1]) if nca_chunks else 0,
        "action_channels": int(action_chunks[0].shape[-1]) if action_chunks else 0,
        "episode_batch_size": int(episode_batch_size),
        "gpu_profile": str(runtime_resolution.gpu_profile),
        "full_trial_batch": bool(int(episode_batch_size) >= int(len(entries))),
    }
    _write_json(bundle_dir / "trace_manifest.json", trace_manifest)
    _write_json(bundle_dir / "manifest.json", trace_manifest)
    summary_payload = {
        **cohort_summary,
        "kind": "sensing_edges_post_run_pca_bundle_summary",
        "realized_trial_count": int(len(entries)),
        "logged_step_count": int(len(label_df)),
        "episode_batch_size": int(episode_batch_size),
        "full_trial_batch": bool(int(episode_batch_size) >= int(len(entries))),
        "max_steps": _native_max_steps(cfg),
    }
    _write_json(bundle_dir / "summary.json", summary_payload)
    _write_json(bundle_dir / "aggregate.json", summary_payload)
    _write_stats_tables(
        assay_dir=bundle_dir,
        specimen_records=[summary_payload],
        inventory_records=[
            {
                "assay_family": "pca_bundle",
                "surface_id": "native_core",
                "surface_label": _surface_label("native_core"),
                "subgroup_type": "cohort",
                "subgroup_id": cohort_id,
                "surface_dir": str(bundle_dir),
                "target_dir": str(bundle_dir),
                "summary_path": str(bundle_dir / "summary.json"),
                "aggregate_path": str(bundle_dir / "aggregate.json"),
                "per_episode_csv": str(bundle_dir / "trace_labels.csv"),
                "per_episode_parquet": None,
                "render_manifest_path": None,
                "charts_dir": None,
            }
        ],
    )
    return {
        **summary_payload,
        "bundle_dir": str(bundle_dir),
        "trace_manifest_path": str(bundle_dir / "trace_manifest.json"),
        "cohort_summary_path": str(bundle_dir / "cohort_summary.json"),
        "summary_path": str(bundle_dir / "summary.json"),
        "aggregate_path": str(bundle_dir / "aggregate.json"),
        "trace_labels_path": str(bundle_dir / "trace_labels.csv"),
        "trial_manifest_path": str(bundle_dir / "trial_manifest.csv"),
    }


def _run_pca_bundle(
    *,
    cfg: DictConfig,
    runtime_resolution,
    specimen_dir: Path,
    specimen_id: str,
    genotype: np.ndarray,
    selection_row: dict[str, Any],
    run_id: str,
) -> Path:
    del selection_row
    post_run_cfg = cfg.eval.post_run
    trace_cfg = post_run_cfg.trace
    assay_dir = specimen_dir / "pca_bundle"
    if not bool(trace_cfg.enabled):
        return assay_dir

    target_pair_cfg = trace_cfg.target_pair
    primary_record = _run_pca_bundle_cohort(
        cfg=cfg,
        runtime_resolution=runtime_resolution,
        bundle_dir=assay_dir,
        trace_cfg=trace_cfg,
        specimen_id=specimen_id,
        genotype=genotype,
        run_id=run_id,
        cohort_id="intact_native_core",
        cohort_label="Intact Native Core",
        goal_distance=int(target_pair_cfg.goal_distance),
        cliff_distance=int(target_pair_cfg.cliff_distance),
        deformation_condition="intact",
        signaling_condition="intact",
    )

    cohort_records = [primary_record]
    inventory_records = [
        {
            "assay_family": "pca_bundle",
            "surface_id": "native_core",
            "surface_label": _surface_label("native_core"),
            "subgroup_type": "cohort",
            "subgroup_id": str(primary_record["cohort_id"]),
            "surface_dir": str(assay_dir),
            "target_dir": str(assay_dir),
            "summary_path": str(primary_record["summary_path"]),
            "aggregate_path": str(primary_record["aggregate_path"]),
            "per_episode_csv": str(primary_record["trace_labels_path"]),
            "per_episode_parquet": None,
            "render_manifest_path": None,
            "charts_dir": None,
        }
    ]
    for extra in _pca_bundle_additional_native_cohorts(trace_cfg):
        bundle_dir = assay_dir / "cohorts" / str(extra["cohort_id"])
        record = _run_pca_bundle_cohort(
            cfg=cfg,
            runtime_resolution=runtime_resolution,
            bundle_dir=bundle_dir,
            trace_cfg=trace_cfg,
            specimen_id=specimen_id,
            genotype=genotype,
            run_id=run_id,
            cohort_id=str(extra["cohort_id"]),
            cohort_label=str(extra["cohort_label"]),
            goal_distance=int(extra["goal_distance"]),
            cliff_distance=int(extra["cliff_distance"]),
            deformation_condition=str(extra["deformation_condition"]),
            signaling_condition=str(extra["signaling_condition"]),
        )
        cohort_records.append(record)
        inventory_records.append(
            {
                "assay_family": "pca_bundle",
                "surface_id": "native_core",
                "surface_label": _surface_label("native_core"),
                "subgroup_type": "cohort",
                "subgroup_id": str(record["cohort_id"]),
                "surface_dir": str(assay_dir),
                "target_dir": str(record["bundle_dir"]),
                "summary_path": str(record["summary_path"]),
                "aggregate_path": str(record["aggregate_path"]),
                "per_episode_csv": str(record["trace_labels_path"]),
                "per_episode_parquet": None,
                "render_manifest_path": None,
                "charts_dir": None,
            }
        )

    assay_index = {
        "schema_version": _POST_RUN_SCHEMA_VERSION,
        "kind": "sensing_edges_post_run_pca_bundle_index",
        "specimen_id": specimen_id,
        "primary_cohort_id": str(primary_record["cohort_id"]),
        "cohort_count": int(len(cohort_records)),
        "cohorts": [
            {
                "cohort_id": str(record["cohort_id"]),
                "cohort_label": str(record["cohort_label"]),
                "surface_id": str(record["surface_id"]),
                "surface_label": str(record["surface_label"]),
                "deformation_condition": str(record["deformation_condition"]),
                "signaling_condition": str(record["signaling_condition"]),
                "distance_pair": str(record["distance_pair"]),
                "selected_side": str(record["selected_side"]),
                "selected_trial_count": int(record["selected_trial_count"]),
                "bundle_dir": str(record["bundle_dir"]),
                "relative_bundle_dir": str(Path(record["bundle_dir"]).relative_to(assay_dir)),
                "summary_path": str(record["summary_path"]),
                "trace_manifest_path": str(record["trace_manifest_path"]),
            }
            for record in cohort_records
        ],
    }
    _write_json(assay_dir / "cohorts.json", assay_index)
    _write_stats_tables(
        assay_dir=assay_dir,
        condition_records=cohort_records,
        specimen_records=[assay_index],
        inventory_records=inventory_records,
    )
    return assay_dir


def _run_initial_direction_summary(
    *,
    specimen_dir: Path,
    native_rows: pd.DataFrame,
    specimen_id: str,
) -> Path:
    assay_dir = specimen_dir / "initial_direction"
    charts_dir = assay_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    direction_rows = native_rows[
        [
            "specimen_id",
            "trial_id",
            "distance_pair",
            "side",
            "initial_direction",
            "initial_commit_step",
            "initial_commit_x",
        ]
    ].copy()
    fractions = {
        "left": float(np.mean((direction_rows["initial_direction"] == "left").astype(float))),
        "right": float(np.mean((direction_rows["initial_direction"] == "right").astype(float))),
        "none": float(np.mean((direction_rows["initial_direction"] == "none").astype(float))),
    }
    direction_regularity = max(fractions.values()) if fractions else 0.0
    bundle = _write_dataframe_bundle(direction_rows, assay_dir, stem="per_trial")
    summary_payload = {
        "schema_version": _POST_RUN_SCHEMA_VERSION,
        "kind": "sensing_edges_post_run_initial_direction",
        "specimen_id": specimen_id,
        "fractions": fractions,
        "direction_regularity": float(direction_regularity),
        "paths": bundle,
    }
    _write_json(
        assay_dir / "manifest.json",
        {
            "schema_version": _POST_RUN_SCHEMA_VERSION,
            "kind": "sensing_edges_post_run_assay_manifest",
            "assay_family": "initial_direction",
            "specimen_id": specimen_id,
        },
    )
    _write_json(assay_dir / "summary.json", summary_payload)
    _write_json(assay_dir / "aggregate.json", summary_payload)
    _write_stats_tables(
        assay_dir=assay_dir,
        specimen_records=[summary_payload],
        inventory_records=[
            {
                "assay_family": "initial_direction",
                "surface_id": "native_core",
                "surface_label": _surface_label("native_core"),
                "subgroup_type": "summary",
                "subgroup_id": specimen_id,
                "surface_dir": str(assay_dir),
                "target_dir": str(assay_dir),
                "summary_path": str(assay_dir / "summary.json"),
                "aggregate_path": str(assay_dir / "aggregate.json"),
                "per_episode_csv": str(assay_dir / "per_trial.csv"),
                "per_episode_parquet": str(assay_dir / "per_trial.parquet"),
                "render_manifest_path": None,
                "charts_dir": str(charts_dir),
            }
        ],
    )
    _chart_bar(
        categories=list(fractions.keys()),
        values=[fractions[key] for key in ("left", "right", "none")],
        ci_bounds=None,
        title="Initial Direction Fractions",
        ylabel="fraction",
        output_path=charts_dir / "fractions.png",
    )
    _chart_bar(
        categories=["regularity"],
        values=[float(direction_regularity)],
        ci_bounds=None,
        title="Direction Regularity",
        ylabel="score",
        output_path=charts_dir / "direction_regularity.png",
    )
    return assay_dir
