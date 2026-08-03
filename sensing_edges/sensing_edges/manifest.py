"""Run provenance capture for reproducibility.

Captures git state, Python/JAX versions, hardware info, and config hash
into manifest.json at run start. Updated with summary at run end.
"""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import jax
import numpy as np


def _git_info() -> dict:
    """Capture git state."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, branch, dirty = "unknown", None, True
    return {"commit": commit, "branch": branch, "dirty": dirty}


def _system_info() -> dict:
    """Capture system and package versions."""
    import jaxlib
    import evosax

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "evosax_version": _package_version("evosax", evosax),
    }


def _package_version(package_name: str, module) -> str:
    """Resolve a package version from module metadata or installed distribution metadata."""
    version = getattr(module, "__version__", None)
    if version is not None:
        return str(version)

    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _hardware_info() -> dict:
    """Capture JAX backend and device info."""
    backend = jax.default_backend()
    devices = [str(d) for d in jax.devices()]
    gpu_name = None
    if backend == "gpu" and devices:
        gpu_name = devices[0]
    return {
        "backend": backend,
        "device_count": len(devices),
        "devices": devices,
        "gpu_name": gpu_name,
    }


def config_sha256(resolved_config: str) -> str:
    """SHA256 hash of the resolved config string."""
    return hashlib.sha256(resolved_config.encode()).hexdigest()


def load_manifest(path: str | Path) -> dict:
    """Load a manifest JSON payload from disk."""
    with open(Path(path)) as handle:
        return json.load(handle)


def _default_lineage(*, mode: str = "root") -> dict:
    return {
        "mode": str(mode),
        "parent_run_id": None,
        "parent_checkpoint_path": None,
        "parent_completed_generations": None,
        "parent_schedule_hashes": None,
    }


def _apply_retry_state(
    manifest: dict,
    *,
    attempt_count: int | None = None,
    resumed_from_checkpoint: str | None = None,
    resumed_generation: int | None = None,
    latest_checkpoint_path: str | None = None,
    status: str | None = None,
    wandb_run_id: str | None = None,
) -> dict:
    retry = dict(manifest.get("retry", {}))
    if attempt_count is not None:
        retry["attempt_count"] = int(attempt_count)
    if resumed_from_checkpoint is not None:
        retry["resumed_from_checkpoint"] = resumed_from_checkpoint
    if resumed_generation is not None:
        retry["resumed_generation"] = int(resumed_generation)
    if latest_checkpoint_path is not None:
        retry["latest_checkpoint_path"] = latest_checkpoint_path
    manifest["retry"] = retry

    wandb_block = dict(manifest.get("wandb", {}))
    if wandb_run_id is not None:
        wandb_block["run_id"] = str(wandb_run_id)
    if wandb_block:
        manifest["wandb"] = wandb_block

    if status is not None:
        manifest["status"] = status
    if status == "running":
        manifest.pop("completed_at_utc", None)
    return manifest


def create_manifest(
    run_id: str | None = None,
    wandb_run_id: str | None = None,
    preset_id: str | None = None,
    resolved_config_str: str = "",
    resolved_config_path: str | None = None,
    logical_config_sha256: str | None = None,
    last_invocation_sha256: str | None = None,
    seed: int = 0,
    notes: str | None = None,
    tags: list[str] | None = None,
    condition: dict | None = None,
    paper_core_id: str | None = None,
    paper_core_sha256: str | None = None,
    preset_changed_paths: list[str] | None = None,
    runtime_override_keys: list[str] | None = None,
    gpu_profile: str | None = None,
    resolved_batching: dict | None = None,
    resolved_collision: dict | None = None,
    attempt_count: int = 1,
    resumed_from_checkpoint: str | None = None,
    resumed_generation: int | None = None,
    latest_checkpoint_path: str | None = None,
    lineage: dict | None = None,
    stages: list[dict] | None = None,
) -> dict:
    """Create a manifest dict at run start."""
    manifest_run_id = str(run_id or uuid.uuid4())
    invocation_sha = str(last_invocation_sha256 or config_sha256(resolved_config_str))
    logical_sha = str(logical_config_sha256 or invocation_sha)
    return {
        "schema_version": 1,
        "kind": "sensing_edges_manifest",
        "run_id": manifest_run_id,
        "preset_id": preset_id,
        "status": "running",
        "notes": notes,
        "tags": tags or [],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "condition": condition or {},
        "paper_core_id": paper_core_id,
        "paper_core_sha256": paper_core_sha256,
        "preset_changed_paths": preset_changed_paths or [],
        "runtime_override_keys": runtime_override_keys or [],
        "config": {
            "logical_config_sha256": logical_sha,
            "last_invocation_sha256": invocation_sha,
            "resolved_config_sha256": config_sha256(resolved_config_str),
            "resolved_config_path": resolved_config_path,
        },
        "runtime": {
            "gpu_profile": gpu_profile,
            "resolved_batching": resolved_batching or {},
            "resolved_collision": resolved_collision,
        },
        "wandb": {
            "run_id": str(wandb_run_id or manifest_run_id),
        },
        "retry": {
            "attempt_count": int(attempt_count),
            "resumed_from_checkpoint": resumed_from_checkpoint,
            "resumed_generation": resumed_generation,
            "latest_checkpoint_path": latest_checkpoint_path,
        },
        "lineage": lineage or _default_lineage(),
        "stages": stages or [],
        "seeds": {"master_seed": seed},
        "git": _git_info(),
        "system": _system_info(),
        "hardware": _hardware_info(),
        "summary": {
            "completed_generations": None,
            "best_train_fitness": None,
            "final_qd_coverage": None,
            "final_qd_score": None,
        },
    }


def save_manifest(manifest: dict, path: str | Path) -> None:
    """Write manifest to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def update_manifest_summary(
    path: str | Path,
    *,
    status: str = "completed",
    completed_generations: int | None = None,
    best_train_fitness: float | None = None,
    final_qd_coverage: float | None = None,
    final_qd_score: float | None = None,
) -> None:
    """Update manifest with final summary."""
    path = Path(path)
    with open(path) as f:
        manifest = json.load(f)
    manifest["status"] = status
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["summary"].update({
        k: v for k, v in {
            "completed_generations": completed_generations,
            "best_train_fitness": best_train_fitness,
            "final_qd_coverage": final_qd_coverage,
            "final_qd_score": final_qd_score,
        }.items() if v is not None
    })
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def record_manifest_stage_result(
    path: str | Path,
    *,
    status: str,
    stage_entry: dict,
    completed_generations: int | None = None,
    best_train_fitness: float | None = None,
    final_qd_coverage: float | None = None,
    final_qd_score: float | None = None,
) -> dict:
    """Append one stage result and update the latest manifest summary."""
    path = Path(path)
    manifest = load_manifest(path)
    stages = list(manifest.get("stages", []))
    stages.append(stage_entry)
    manifest["stages"] = stages
    manifest["status"] = str(status)
    if status == "completed":
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    else:
        manifest.pop("completed_at_utc", None)
    manifest["summary"].update(
        {
            k: v
            for k, v in {
                "completed_generations": completed_generations,
                "best_train_fitness": best_train_fitness,
                "final_qd_coverage": final_qd_coverage,
                "final_qd_score": final_qd_score,
            }.items()
            if v is not None
        }
    )
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    return manifest


def update_manifest_retry_state(
    path: str | Path,
    *,
    attempt_count: int | None = None,
    resumed_from_checkpoint: str | None = None,
    resumed_generation: int | None = None,
    latest_checkpoint_path: str | None = None,
    status: str | None = None,
    wandb_run_id: str | None = None,
) -> dict:
    """Update retry/resume state on an existing manifest."""
    path = Path(path)
    manifest = load_manifest(path)
    manifest = _apply_retry_state(
        manifest,
        attempt_count=attempt_count,
        resumed_from_checkpoint=resumed_from_checkpoint,
        resumed_generation=resumed_generation,
        latest_checkpoint_path=latest_checkpoint_path,
        status=status,
        wandb_run_id=wandb_run_id,
    )

    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    return manifest
