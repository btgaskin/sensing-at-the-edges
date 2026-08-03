"""Checkpoint and archive export helpers using JSON + NPZ payloads.

No pickle — all state is serializable as numpy arrays or JSON primitives.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .qd.grid_archive import GridArchive
from .qd.cma_emitter import CMAEmitter
from .qd.focus import FocusWindowBounds

log = logging.getLogger(__name__)
_DONE_MARKER_VERSION = 1


def payload_done_path(path_prefix: str | Path) -> Path:
    """Return the completion marker path for a payload prefix."""
    return Path(path_prefix).with_suffix(".done")


def _write_payload_bundle(
    path_prefix: str | Path,
    *,
    meta: dict,
    arrays: dict[str, np.ndarray],
    kind: str,
    on_complete: Callable[[], None] | None = None,
) -> tuple[Path, Path]:
    """Persist a JSON+NPZ payload atomically and mark it complete."""
    prefix = Path(path_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    npz_path = prefix.with_suffix(".npz")
    done_path = payload_done_path(prefix)
    tmp_json_path = prefix.with_suffix(".tmp.json")
    tmp_npz_path = prefix.with_suffix(".tmp.npz")
    tmp_done_path = prefix.with_suffix(".tmp.done")

    for tmp_path in (tmp_json_path, tmp_npz_path, tmp_done_path):
        if tmp_path.exists():
            tmp_path.unlink()

    with open(tmp_json_path, "w") as handle:
        json.dump(meta, handle, indent=2, default=str)
    with open(tmp_npz_path, "wb") as handle:
        np.savez(handle, **arrays)

    tmp_json_path.replace(json_path)
    tmp_npz_path.replace(npz_path)

    done_payload = {
        "schema_version": _DONE_MARKER_VERSION,
        "kind": kind,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(tmp_done_path, "w") as handle:
        json.dump(done_payload, handle, indent=2, default=str)
    tmp_done_path.replace(done_path)

    if on_complete is not None:
        on_complete()

    return json_path, npz_path


def load_checkpoint_meta(path_prefix: str | Path) -> dict:
    """Load checkpoint JSON metadata without materializing arrays."""
    json_path = Path(path_prefix).with_suffix(".json")
    with open(json_path) as handle:
        return json.load(handle)


def is_payload_complete(
    path_prefix: str | Path,
    *,
    require_done: bool = False,
) -> bool:
    """Return whether a payload prefix looks complete on disk."""
    prefix = Path(path_prefix)
    json_path = prefix.with_suffix(".json")
    npz_path = prefix.with_suffix(".npz")
    done_path = payload_done_path(prefix)
    if done_path.exists():
        return json_path.exists() and npz_path.exists()
    if require_done:
        return False
    # Backward-compatible fallback for legacy checkpoints written before the
    # explicit completion marker existed.
    return json_path.exists() and npz_path.exists()


def find_latest_checkpoint(
    checkpoint_dir: str | Path,
    *,
    config_sha256: str | None = None,
    run_id: str | None = None,
    require_done: bool = True,
) -> tuple[Path, dict] | None:
    """Return the latest completed checkpoint prefix and metadata."""
    directory = Path(checkpoint_dir)
    if not directory.exists():
        return None

    candidates: list[tuple[int, str, Path, dict]] = []
    for json_path in sorted(directory.glob("*.json")):
        prefix = json_path.with_suffix("")
        if not is_payload_complete(prefix, require_done=require_done):
            continue
        try:
            meta = load_checkpoint_meta(prefix)
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("kind") != "sensing_edges_checkpoint":
            continue
        if config_sha256 and str(meta.get("config_sha256", "")) != str(config_sha256):
            continue
        if run_id and str(meta.get("run_id", "")) != str(run_id):
            continue
        generation = int(meta.get("generation", -1))
        created_at = str(meta.get("created_at_utc", ""))
        candidates.append((generation, created_at, prefix, meta))

    if not candidates:
        return None
    generation, _created_at, prefix, meta = max(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    return prefix, meta


def _restore_emitter_state(emitter: CMAEmitter, data: dict, emitter_idx: int) -> bool:
    """Restore an evosax state dataclass from saved per-field arrays."""
    template_state = emitter.engine.init(jax.random.PRNGKey(0))
    state_fields = tuple(template_state.__dataclass_fields__)
    saved_fields = {
        field_name: data[f"emitters/{emitter_idx}/{field_name}"]
        for field_name in state_fields
        if f"emitters/{emitter_idx}/{field_name}" in data
    }
    if not saved_fields:
        log.warning(
            "Checkpoint emitter %d has no serialized ES state; falling back to re-init.",
            emitter_idx,
        )
        return False

    missing = [field_name for field_name in state_fields if field_name not in saved_fields]
    if missing:
        log.warning(
            "Checkpoint emitter %d is missing ES state fields %s; falling back to re-init.",
            emitter_idx,
            missing,
        )
        return False

    emitter.es_state = type(template_state)(
        **{field_name: jnp.asarray(saved_fields[field_name]) for field_name in state_fields}
    )
    return True


def save_checkpoint(
    path_prefix: str | Path,
    *,
    archive: GridArchive,
    emitters: list[CMAEmitter],
    rng: np.random.Generator,
    jax_key: jax.Array,
    generation: int,
    config_sha256: str = "",
    run_id: str = "",
    focus_window_bounds: FocusWindowBounds | None = None,
    on_complete: Callable[[], None] | None = None,
) -> tuple[Path, Path]:
    """Save checkpoint as .json + .npz pair."""
    meta, arrays = build_checkpoint_payload(
        archive=archive,
        emitters=emitters,
        rng=rng,
        jax_key=jax_key,
        generation=generation,
        config_sha256=config_sha256,
        run_id=run_id,
        focus_window_bounds=focus_window_bounds,
    )
    return write_checkpoint_payload(
        path_prefix,
        meta=meta,
        arrays=arrays,
        on_complete=on_complete,
    )


def build_checkpoint_payload(
    *,
    archive: GridArchive,
    emitters: list[CMAEmitter],
    rng: np.random.Generator,
    jax_key: jax.Array,
    generation: int,
    config_sha256: str = "",
    run_id: str = "",
    focus_window_bounds: FocusWindowBounds | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Snapshot checkpoint metadata and arrays into immutable host payloads."""
    meta = {
        "schema_version": 1,
        "kind": "sensing_edges_checkpoint",
        "run_id": run_id,
        "generation": generation,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha256,
        "archive": {
            "dims": list(archive.dims),
            "ranges": archive.ranges,
            "descriptor_names": list(archive.descriptor_names),
            "descriptor_labels": list(archive.descriptor_labels),
            "learning_rate": archive.learning_rate,
            "threshold_min": archive.threshold_min,
            "occupied_count": archive.size,
        },
        "emitters": {
            "n_emitters": len(emitters),
            "total_restarts": [e.total_restarts for e in emitters],
            "stale_counters": [e.stale_counter for e in emitters],
            "age_since_restart": [e.age_since_restart for e in emitters],
            "last_restart_reasons": [e.last_restart_reason for e in emitters],
            "restart_reason_totals": {
                "stale": [e.total_restarts_by_reason["stale"] for e in emitters],
                "es_stop": [e.total_restarts_by_reason["es_stop"] for e in emitters],
                "age_cap": [e.total_restarts_by_reason["age_cap"] for e in emitters],
            },
            "flat_counters": [e.flat_counter for e in emitters],
            "strategy": emitters[0].engine.strategy_name if emitters else None,
            "population_size": emitters[0].engine.population_size if emitters else None,
            "stale_generations": emitters[0].stale_generations if emitters else None,
            "max_age": emitters[0].max_age if emitters else None,
            "meaningful_improvement_eps": (
                emitters[0].meaningful_improvement_eps if emitters else None
            ),
            "condition_cov_limit": (
                emitters[0].condition_cov_limit if emitters else None
            ),
            "flat_generations": emitters[0].flat_generations if emitters else None,
            "flat_tolerance": emitters[0].flat_tolerance if emitters else None,
            "std_epsilon": emitters[0].std_epsilon if emitters else None,
            "param_overrides": emitters[0].param_overrides if emitters else None,
        },
        "progress": {
            "best_fitness_so_far": archive.best_fitness,
            "n_occupied": archive.size,
        },
    }
    if focus_window_bounds is not None:
        meta["focus_window_bounds"] = focus_window_bounds.state_dict()
    arrays = {}
    state = archive.state_dict()
    arrays["archive/thresholds"] = state["thresholds"]
    arrays["archive/occupied_indices"] = state["occupied_indices"]
    if state["fitnesses"] is not None and len(state["occupied_indices"]) > 0:
        arrays["archive/fitnesses"] = state["fitnesses"]
        arrays["archive/genotypes"] = state["genotypes"]
        arrays["archive/descriptors"] = state["descriptors"]

    # Emitter ES states — flatten each to numpy arrays.
    for i, em in enumerate(emitters):
        if em.es_state is not None:
            for field_name in em.es_state.__dataclass_fields__:
                val = getattr(em.es_state, field_name)
                arr = np.asarray(val)
                arrays[f"emitters/{i}/{field_name}"] = arr
        arrays[f"emitters/{i}/stale_counter"] = np.array(em.stale_counter, dtype=np.int32)
        arrays[f"emitters/{i}/age_since_restart"] = np.array(
            em.age_since_restart, dtype=np.int32
        )
        arrays[f"emitters/{i}/flat_counter"] = np.array(em.flat_counter, dtype=np.int32)
        arrays[f"emitters/{i}/total_restarts"] = np.array(em.total_restarts, dtype=np.int32)
        arrays[f"emitters/{i}/total_restarts_stale"] = np.array(
            em.total_restarts_by_reason["stale"], dtype=np.int32
        )
        arrays[f"emitters/{i}/total_restarts_es_stop"] = np.array(
            em.total_restarts_by_reason["es_stop"], dtype=np.int32
        )
        arrays[f"emitters/{i}/total_restarts_age_cap"] = np.array(
            em.total_restarts_by_reason["age_cap"], dtype=np.int32
        )
        arrays[f"emitters/{i}/last_restart_reason"] = np.array(
            "" if em.last_restart_reason is None else em.last_restart_reason
        )

    # RNG state.
    arrays["rng/jax_key"] = np.asarray(jax_key)
    rng_state = rng.bit_generator.state
    arrays["rng/numpy_state_s_has_uint32"] = np.array(
        rng_state["has_uint32"], dtype=np.int32
    )
    arrays["rng/numpy_state_s_uinteger"] = np.array(
        rng_state["uinteger"], dtype=np.uint32
    )
    arrays["rng/numpy_state_state"] = np.array(str(rng_state["state"]["state"]))
    arrays["rng/numpy_state_inc"] = np.array(str(rng_state["state"]["inc"]))

    return meta, arrays


def write_checkpoint_payload(
    path_prefix: str | Path,
    *,
    meta: dict,
    arrays: dict[str, np.ndarray],
    on_complete: Callable[[], None] | None = None,
) -> tuple[Path, Path]:
    """Persist a checkpoint payload previously produced on the host."""
    return _write_payload_bundle(
        path_prefix,
        meta=meta,
        arrays=arrays,
        kind="sensing_edges_checkpoint",
        on_complete=on_complete,
    )


def build_archive_export_payload(
    *,
    archive: GridArchive,
    config_sha256: str = "",
    run_id: str = "",
) -> tuple[dict, dict[str, np.ndarray]]:
    """Snapshot a final archive export payload."""
    state = archive.state_dict()
    meta = {
        "schema_version": 1,
        "kind": "sensing_edges_final_archive",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha256,
        "archive": {
            "dims": list(archive.dims),
            "ranges": archive.ranges,
            "descriptor_names": list(archive.descriptor_names),
            "descriptor_labels": list(archive.descriptor_labels),
            "learning_rate": archive.learning_rate,
            "threshold_min": archive.threshold_min,
            "occupied_count": archive.size,
            "best_fitness": archive.best_fitness,
            "qd_score": archive.qd_score,
            "coverage": archive.coverage,
        },
    }
    arrays: dict[str, np.ndarray] = {
        "archive/thresholds": state["thresholds"],
        "archive/occupied_indices": state["occupied_indices"],
    }
    if state["fitnesses"] is not None and len(state["occupied_indices"]) > 0:
        arrays["archive/fitnesses"] = state["fitnesses"]
        arrays["archive/genotypes"] = state["genotypes"]
        arrays["archive/descriptors"] = state["descriptors"]
    return meta, arrays


def write_archive_export_payload(
    path_prefix: str | Path,
    *,
    meta: dict,
    arrays: dict[str, np.ndarray],
    on_complete: Callable[[], None] | None = None,
) -> tuple[Path, Path]:
    """Persist a final archive export payload."""
    return _write_payload_bundle(
        path_prefix,
        meta=meta,
        arrays=arrays,
        kind="sensing_edges_final_archive",
        on_complete=on_complete,
    )


def load_checkpoint(
    path_prefix: str | Path,
    *,
    genotype_length: int,
    population_size: int | None,
    strategy: str = "sep_cmaes",
    stale_generations: int = 5,
    max_age: int = 50,
    meaningful_improvement_eps: float = 0.5,
    flat_generations: int = 3,
    flat_tolerance: float | None = None,
    std_epsilon: float | None = None,
    condition_cov_limit: float = 1.0e14,
    param_overrides: dict | None = None,
) -> dict:
    """Load checkpoint and return components dict.

    Returns dict with keys: archive, emitters, rng, jax_key, generation, meta,
    focus_window_bounds.
    """
    prefix = Path(path_prefix)
    json_path = prefix.with_suffix(".json")
    npz_path = prefix.with_suffix(".npz")

    with open(json_path) as f:
        meta = json.load(f)

    data = dict(np.load(npz_path, allow_pickle=False))

    # Reconstruct archive.
    archive_meta = meta["archive"]
    archive = GridArchive(
        dims=tuple(archive_meta["dims"]),
        ranges=archive_meta["ranges"],
        descriptor_names=tuple(archive_meta.get("descriptor_names", [])) or None,
        descriptor_labels=tuple(archive_meta.get("descriptor_labels", [])) or None,
        learning_rate=archive_meta.get("learning_rate", 0.01),
        threshold_min=archive_meta.get("threshold_min", 0.0),
    )
    if "archive/genotypes" in data:
        archive_state = {
            "dims": tuple(archive_meta["dims"]),
            "ranges": archive_meta["ranges"],
            "descriptor_names": archive_meta.get("descriptor_names"),
            "descriptor_labels": archive_meta.get("descriptor_labels"),
            "learning_rate": archive_meta.get("learning_rate", 0.01),
            "threshold_min": archive_meta.get("threshold_min", 0.0),
            "thresholds": data["archive/thresholds"],
            "occupied_indices": data["archive/occupied_indices"],
            "fitnesses": data["archive/fitnesses"],
            "genotypes": data["archive/genotypes"],
            "descriptors": data["archive/descriptors"],
        }
        archive = GridArchive.from_state_dict(archive_state)

    # Reconstruct emitters.
    emitter_meta = meta.get("emitters", {})
    n_emitters = emitter_meta["n_emitters"]
    resolved_population_size = emitter_meta.get("population_size", population_size)
    if resolved_population_size is None:
        raise ValueError("population_size is required when checkpoint metadata does not include it.")
    resolved_strategy = emitter_meta.get("strategy", strategy)
    resolved_stale_generations = emitter_meta.get(
        "stale_generations",
        emitter_meta.get("flat_generations", stale_generations),
    )
    resolved_max_age = emitter_meta.get("max_age", max_age)
    resolved_meaningful_improvement_eps = emitter_meta.get(
        "meaningful_improvement_eps", meaningful_improvement_eps
    )
    resolved_flat_generations = emitter_meta.get("flat_generations", flat_generations)
    resolved_flat_tolerance = emitter_meta.get(
        "flat_tolerance", 1e-12 if flat_tolerance is None else flat_tolerance
    )
    resolved_std_epsilon = emitter_meta.get(
        "std_epsilon", 1e-12 if std_epsilon is None else std_epsilon
    )
    resolved_condition_cov_limit = emitter_meta.get(
        "condition_cov_limit", condition_cov_limit
    )
    resolved_param_overrides = emitter_meta.get("param_overrides", param_overrides)
    if resolved_param_overrides is not None:
        resolved_param_overrides = dict(resolved_param_overrides)

    emitters = []
    for i in range(n_emitters):
        em = CMAEmitter(
            genotype_length=genotype_length,
            population_size=int(resolved_population_size),
            strategy=resolved_strategy,
            stale_generations=int(resolved_stale_generations),
            max_age=int(resolved_max_age),
            meaningful_improvement_eps=float(resolved_meaningful_improvement_eps),
            flat_generations=int(resolved_flat_generations),
            flat_tolerance=float(resolved_flat_tolerance),
            std_epsilon=float(resolved_std_epsilon),
            condition_cov_limit=float(resolved_condition_cov_limit),
            param_overrides=resolved_param_overrides,
        )
        # Restore restart counters and totals.
        sc_key = f"emitters/{i}/stale_counter"
        age_key = f"emitters/{i}/age_since_restart"
        fc_key = f"emitters/{i}/flat_counter"
        tr_key = f"emitters/{i}/total_restarts"
        lrr_key = f"emitters/{i}/last_restart_reason"
        tr_stale_key = f"emitters/{i}/total_restarts_stale"
        tr_es_stop_key = f"emitters/{i}/total_restarts_es_stop"
        tr_age_cap_key = f"emitters/{i}/total_restarts_age_cap"
        if sc_key in data:
            em.stale_counter = int(data[sc_key])
        elif fc_key in data:
            em.stale_counter = int(data[fc_key])
        if age_key in data:
            em.age_since_restart = int(data[age_key])
        if fc_key in data:
            em.flat_counter = int(data[fc_key])
        if tr_key in data:
            em.total_restarts = int(data[tr_key])
        if tr_stale_key in data:
            em.total_restarts_by_reason["stale"] = int(data[tr_stale_key])
        if tr_es_stop_key in data:
            em.total_restarts_by_reason["es_stop"] = int(data[tr_es_stop_key])
        if tr_age_cap_key in data:
            em.total_restarts_by_reason["age_cap"] = int(data[tr_age_cap_key])
        if lrr_key in data:
            reason = str(data[lrr_key])
            em.last_restart_reason = reason or None
        _restore_emitter_state(em, data, i)
        emitters.append(em)

    # Reconstruct RNG.
    jax_key = jnp.array(data["rng/jax_key"])
    rng = np.random.default_rng()
    has_uint32_key = (
        "rng/numpy_state_s_has_uint32"
        if "rng/numpy_state_s_has_uint32" in data
        else "rng/numpy_state_has_uint32"
    )
    uinteger_key = (
        "rng/numpy_state_s_uinteger"
        if "rng/numpy_state_s_uinteger" in data
        else "rng/numpy_state_uinteger"
    )
    rng.bit_generator.state = {
        "bit_generator": "PCG64",
        "state": {
            "state": int(data["rng/numpy_state_state"]),
            "inc": int(data["rng/numpy_state_inc"]),
        },
        "has_uint32": int(data[has_uint32_key]),
        "uinteger": int(data[uinteger_key]),
    }

    return {
        "archive": archive,
        "emitters": emitters,
        "rng": rng,
        "jax_key": jax_key,
        "generation": meta["generation"],
        "meta": meta,
        "focus_window_bounds": (
            FocusWindowBounds.from_state_dict(meta["focus_window_bounds"])
            if "focus_window_bounds" in meta
            else None
        ),
    }
