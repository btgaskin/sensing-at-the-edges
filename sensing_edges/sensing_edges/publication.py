"""Publication artifact helpers for sensing_edges."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tarfile
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from .checkpoint import (
    find_latest_checkpoint,
    is_payload_complete,
    load_checkpoint_meta,
    payload_done_path,
)
from .evaluate import summarize_test_per_terrain
from .manifest import load_manifest
from .runtime_config import run_metadata

DEFAULT_RELEASE_ID = "paper_main"
PUBLICATION_DIRNAME = "publication"
_SUMMARY_METRICS = (
    ("test_mean_fitness", "Test Mean Fitness"),
    ("mean_any_progress", "Mean Any Progress"),
    ("mean_goal_progress", "Mean Goal Progress"),
    ("mean_correct_progress", "Mean Correct Progress"),
    ("mean_incorrect_progress", "Mean Incorrect Progress"),
    ("mean_toward_goal_frontier", "Mean Toward-Goal Frontier"),
    ("mean_progress_shaping", "Mean Progress Shaping"),
    ("mean_safe_arrival_bonus", "Mean Safe Arrival Bonus"),
    ("mean_safe_dwell_bonus", "Mean Safe Dwell Bonus"),
    ("mean_urgency_shaping", "Mean Urgency Shaping"),
    ("goal_touch_rate", "Goal Touch Rate"),
    ("mean_fall_count", "Mean Falls / Genome"),
    ("mean_turn_count", "Mean Turn Count"),
    ("mean_probe_extent", "Mean Probe Extent"),
)
_ABLATION_METRICS = (
    "test_mean_fitness",
    "test_median_fitness",
    "mean_any_progress",
    "mean_goal_progress",
    "mean_correct_progress",
    "mean_incorrect_progress",
    "mean_toward_goal_frontier",
    "mean_progress_shaping",
    "mean_safe_arrival_bonus",
    "mean_safe_dwell_bonus",
    "mean_urgency_shaping",
    "goal_touch_rate",
    "mean_fall_count",
    "edge_entry_rate",
    "mean_probe_extent",
    "mean_time_to_edge",
    "mean_time_to_goal",
    "mean_turn_count",
    "mean_activation_cost",
)
_DEFORM_COLORS = {
    "on": "#1b6ca8",
    "off": "#b4413c",
}


def _read_json(path: str | Path) -> dict[str, Any]:
    with open(Path(path)) as handle:
        return json.load(handle)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as handle:
        json.dump(payload, handle, indent=2)


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _publication_config_path(publication_set: str | Path, *, se_root: Path) -> Path:
    candidate = Path(publication_set)
    if candidate.exists():
        return candidate
    if candidate.suffix:
        return candidate
    return se_root / "config" / "publication" / f"{candidate}.yaml"


def load_publication_set(
    publication_set: str | Path = DEFAULT_RELEASE_ID,
    *,
    se_root: str | Path | None = None,
) -> dict[str, Any]:
    resolved_root = Path(se_root) if se_root is not None else Path(__file__).resolve().parent.parent
    config_path = _publication_config_path(publication_set, se_root=resolved_root)
    cfg = OmegaConf.load(config_path)
    payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Publication set must resolve to a mapping: {config_path}")
    payload["config_path"] = str(config_path)
    return payload


def resolve_release_id(
    *,
    release_id: str | None = None,
    publication_set: dict[str, Any] | None = None,
) -> str:
    if release_id:
        return str(release_id)
    if publication_set is not None:
        value = publication_set.get("release_id")
        if value:
            return str(value)
    return DEFAULT_RELEASE_ID


def _body_variant_from_body_spec(spec: str) -> str:
    token = str(spec).strip().lower()
    if token == "grown":
        token = "grown:7x3"
    if token.startswith("grown:"):
        return f"grown_{token.split(':', 1)[1]}"
    return token


def publication_body_variants(publication_set: dict[str, Any]) -> set[str]:
    if publication_set.get("body_variants") is not None:
        return {str(item) for item in publication_set.get("body_variants", [])}
    return {_body_variant_from_body_spec(str(item)) for item in publication_set.get("body_specs", [])}


def run_matches_publication_set(
    metadata: dict[str, Any] | None,
    publication_set: dict[str, Any],
) -> bool:
    if metadata is None:
        return False
    preset_ids = {str(item) for item in publication_set.get("preset_ids", [])}
    if preset_ids and str(metadata.get("preset_id")) not in preset_ids:
        return False
    body_variants = publication_body_variants(publication_set)
    if body_variants and str(metadata.get("body_variant")) not in body_variants:
        return False
    deformation_conditions = {str(item) for item in publication_set.get("deformation_conditions", [])}
    if deformation_conditions and str(metadata.get("deformation_condition")) not in deformation_conditions:
        return False
    return True


def publication_exclude_reason(
    manifest: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    publication_set: dict[str, Any],
) -> str | None:
    if manifest is None:
        return "missing_manifest"
    if metadata is None:
        return "missing_run_metadata"
    if not run_matches_publication_set(metadata, publication_set):
        return "outside_publication_set"
    status = str(manifest.get("status", "")).strip().lower()
    if status != "completed":
        return f"status:{status or 'unknown'}"
    lineage_mode = str(manifest.get("lineage", {}).get("mode", "root")).strip().lower()
    if lineage_mode == "branch":
        return "lineage:branch"
    return None


def release_root(outputs_dir: str | Path, release_id: str) -> Path:
    return Path(outputs_dir) / PUBLICATION_DIRNAME / str(release_id)


def _summary_path_for_run(release_root_dir: Path, run_id: str) -> Path:
    return release_root_dir / "summaries" / f"{run_id}.json"


def _release_dirs(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "manifests": root / "manifests",
        "summaries": root / "summaries",
        "stats": root / "stats",
        "figures": root / "figures",
        "bundles": root / "bundles",
        "schedules": root / "schedules",
    }


def _ensure_release_dirs(root: Path) -> dict[str, Path]:
    dirs = _release_dirs(root)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _local_payload_prefix(candidate: str | None, run_dir: Path) -> Path | None:
    if not candidate:
        return None
    raw = Path(candidate)
    if raw.exists():
        return raw
    checkpoint_candidate = run_dir / "checkpoints" / raw.name
    if checkpoint_candidate.exists():
        return checkpoint_candidate
    top_level_candidate = run_dir / raw.name
    if top_level_candidate.exists():
        return top_level_candidate
    return None


def resolve_latest_durable_checkpoint_prefix(run_dir: Path, manifest: dict[str, Any]) -> Path | None:
    logical_hash = str(manifest.get("config", {}).get("logical_config_sha256", "")).strip() or None
    run_id = str(manifest.get("run_id", "")).strip() or None
    retry_candidate = _local_payload_prefix(
        str(manifest.get("retry", {}).get("latest_checkpoint_path", "")).strip() or None,
        run_dir,
    )
    if retry_candidate is not None and is_payload_complete(retry_candidate, require_done=True):
        try:
            meta = load_checkpoint_meta(retry_candidate)
        except (OSError, json.JSONDecodeError):
            meta = None
        if meta is not None:
            run_ok = run_id is None or str(meta.get("run_id", "")) in {"", run_id}
            hash_ok = logical_hash is None or str(meta.get("config_sha256", "")) in {"", logical_hash}
            if run_ok and hash_ok:
                return retry_candidate

    latest = find_latest_checkpoint(
        run_dir / "checkpoints",
        config_sha256=logical_hash,
        run_id=run_id,
        require_done=True,
    )
    if latest is None:
        return None
    return latest[0]


def resolve_final_archive_prefix(run_dir: Path) -> Path | None:
    prefix = run_dir / "final_archive"
    if is_payload_complete(prefix, require_done=True):
        return prefix
    return None


def resolve_post_train_eval_path(run_dir: Path) -> Path | None:
    structured = run_dir / "post_train" / "archive_eval" / "summary.json"
    if structured.exists():
        return structured
    legacy = run_dir / "post_train_eval.json"
    if legacy.exists():
        return legacy
    return None


def resolve_probe_eval_path(run_dir: Path) -> Path | None:
    structured = run_dir / "post_train" / "robustness_probe" / "summary.json"
    if structured.exists():
        return structured
    legacy = run_dir / "post_train_probe_eval.json"
    if legacy.exists():
        return legacy
    return None


def _runtime_metadata_from_config(config_path: Path | None) -> dict[str, Any] | None:
    if config_path is None or not config_path.exists():
        return None
    cfg = OmegaConf.load(config_path)
    return run_metadata(cfg)


def _normalize_test_summary(test_summary: dict[str, Any] | None) -> dict[str, Any]:
    summary = dict(test_summary or {})
    if "mean_fall_count" not in summary and "fall_rate" in summary:
        summary["mean_fall_count"] = float(summary["fall_rate"])
    return summary


def canonicalize_post_train_eval_payload(
    payload: dict[str, Any],
    *,
    release_id: str,
    run_id: str,
    run_dir_name: str,
    manifest_path: Path,
    config_path: Path | None,
    checkpoint_prefix: Path | None,
    final_archive_prefix: Path | None,
    source_path: Path | None,
) -> dict[str, Any]:
    metadata = payload.get("run_metadata") or _runtime_metadata_from_config(config_path) or {}
    test_summary = payload.get("test_summary")
    if test_summary is None:
        test_summary = summarize_test_per_terrain(payload.get("test_per_terrain", []))
    test_summary = _normalize_test_summary(test_summary)
    return {
        "schema_version": 1,
        "kind": "sensing_edges_publication_summary",
        "release_id": str(release_id),
        "run_id": str(run_id),
        "run_dir_name": str(run_dir_name),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "kind": "post_train_eval",
            "path": None if source_path is None else str(source_path),
            "manifest_path": str(manifest_path),
            "resolved_config_path": None if config_path is None else str(config_path),
        },
        "run_metadata": metadata,
        "selection": {
            "policy": "rerank_selected_finalist",
            "selected_elite_rank": int(payload.get("selected_elite_rank", 0)),
            "selected_elite_flat_index": int(payload.get("selected_elite_flat_index", -1)),
            "selected_elite_fitness_rerank": float(payload.get("selected_elite_fitness_rerank", 0.0)),
            "rerank_scores": list(payload.get("rerank_scores", [])),
        },
        "reporting": {
            "test_mean_fitness": float(payload.get("test_mean_fitness", 0.0)),
            "test_median_fitness": float(payload.get("test_median_fitness", 0.0)),
            "test_ci95_low": float(payload.get("test_ci95_low", 0.0)),
            "test_ci95_high": float(payload.get("test_ci95_high", 0.0)),
            "test_summary": test_summary,
            "test_per_terrain": list(payload.get("test_per_terrain", [])),
            "n_rerank_terrains": int(payload.get("n_rerank_terrains", 0)),
            "n_test_terrains": int(payload.get("n_test_terrains", 0)),
        },
        "schedules": {
            "rerank": payload.get("rerank_schedule"),
            "test": payload.get("eval_schedule"),
        },
        "artifacts": {
            "checkpoint_prefix": None if checkpoint_prefix is None else str(checkpoint_prefix),
            "checkpoint_json": None if checkpoint_prefix is None else str(checkpoint_prefix.with_suffix(".json")),
            "checkpoint_npz": None if checkpoint_prefix is None else str(checkpoint_prefix.with_suffix(".npz")),
            "checkpoint_done": None if checkpoint_prefix is None else str(payload_done_path(checkpoint_prefix)),
            "final_archive_prefix": None if final_archive_prefix is None else str(final_archive_prefix),
            "final_archive_json": None if final_archive_prefix is None else str(final_archive_prefix.with_suffix(".json")),
            "final_archive_npz": None if final_archive_prefix is None else str(final_archive_prefix.with_suffix(".npz")),
            "final_archive_done": None if final_archive_prefix is None else str(payload_done_path(final_archive_prefix)),
            "rerank_artifacts": list(payload.get("rerank_artifacts", [])),
            "eval_artifacts": list(payload.get("eval_artifacts", [])),
        },
        "supplemental": {
            "saved_elites": list(payload.get("saved_elites", [])),
            "variant_suite": payload.get("deformation_suite"),
        },
    }


def _copy_tree_if_missing(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        target = dst / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _maybe_copy_schedule_dir(
    schedule_dir: str | None,
    *,
    destination_root: Path,
    search_roots: tuple[Path, ...],
) -> str | None:
    if not schedule_dir:
        return None
    raw = Path(schedule_dir)
    candidates: list[Path] = []
    if raw.exists():
        candidates.append(raw)
    for root in search_roots:
        candidate = (root / raw).resolve()
        if candidate.exists():
            candidates.append(candidate)
    if not candidates:
        return None
    source = candidates[0]
    if not source.is_dir():
        return None
    dest = destination_root / source.name
    _copy_tree_if_missing(source, dest)
    return str(dest)


def materialize_publication_release(
    outputs_dir: str | Path,
    *,
    release_id: str,
    publication_set: dict[str, Any],
    se_root: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    outputs_root = Path(outputs_dir)
    release_root_dir = release_root(outputs_root, release_id)
    release_dirs = _ensure_release_dirs(release_root_dir)
    search_roots = (
        Path(se_root) if se_root is not None else Path(__file__).resolve().parent.parent,
        Path.cwd(),
    )

    run_records: list[dict[str, Any]] = []
    copied_schedule_dirs: dict[str, str] = {}
    for run_dir in sorted(outputs_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name == PUBLICATION_DIRNAME:
            continue

        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = load_manifest(manifest_path)
        config_path = run_dir / "resolved_config.yaml"
        metrics_path = run_dir / "metrics.jsonl"
        post_train_eval_path = resolve_post_train_eval_path(run_dir)
        probe_eval_path = resolve_probe_eval_path(run_dir)
        metadata = _runtime_metadata_from_config(config_path if config_path.exists() else None)
        exclude_reason = publication_exclude_reason(manifest, metadata, publication_set)
        checkpoint_prefix = resolve_latest_durable_checkpoint_prefix(run_dir, manifest)
        final_archive_prefix = resolve_final_archive_prefix(run_dir)

        run_id = str(manifest.get("run_id", run_dir.name))
        summary_path = _summary_path_for_run(release_root_dir, run_id)
        if exclude_reason is None and post_train_eval_path is not None and post_train_eval_path.exists():
            payload = _read_json(post_train_eval_path)
            summary = canonicalize_post_train_eval_payload(
                payload,
                release_id=release_id,
                run_id=run_id,
                run_dir_name=run_dir.name,
                manifest_path=manifest_path,
                config_path=config_path if config_path.exists() else None,
                checkpoint_prefix=checkpoint_prefix,
                final_archive_prefix=final_archive_prefix,
                source_path=post_train_eval_path,
            )
            _write_json(summary_path, summary)

            for schedule_key in ("train", "rerank", "test"):
                if schedule_key == "train":
                    schedule_dir_value = str(manifest.get("terrain_schedule", {}).get("dir", "")).strip() or None
                else:
                    schedule_dir_value = str(summary.get("schedules", {}).get(schedule_key, {}).get("dir", "")).strip() or None
                copied = _maybe_copy_schedule_dir(
                    schedule_dir_value,
                    destination_root=release_dirs["schedules"],
                    search_roots=search_roots,
                )
                if copied is not None:
                    copied_schedule_dirs[schedule_dir_value or schedule_key] = copied

        ready_checks = {
            "has_manifest": manifest_path.exists(),
            "has_resolved_config": config_path.exists(),
            "has_metrics": metrics_path.exists(),
            "has_latest_durable_checkpoint": checkpoint_prefix is not None,
            "has_final_archive": final_archive_prefix is not None,
            "has_publication_summary": summary_path.exists(),
        }
        publication_ready = exclude_reason is None and all(ready_checks.values())
        run_records.append(
            {
                "schema_version": 1,
                "kind": "sensing_edges_publication_run",
                "release_id": str(release_id),
                "run_id": run_id,
                "run_dir_name": run_dir.name,
                "run_dir_path": str(run_dir),
                "preset_id": None if metadata is None else metadata.get("preset_id"),
                "status": str(manifest.get("status", "")),
                "lineage_mode": str(manifest.get("lineage", {}).get("mode", "root")),
                "eligible_for_publication": exclude_reason is None,
                "publication_ready": bool(publication_ready),
                "exclude_reason": exclude_reason,
                "completed_generations": manifest.get("summary", {}).get("completed_generations"),
                "run_metadata": metadata,
                "ready_checks": ready_checks,
                    "paths": {
                        "manifest": str(manifest_path),
                        "resolved_config": str(config_path) if config_path.exists() else None,
                        "metrics": str(metrics_path) if metrics_path.exists() else None,
                        "post_train_eval": (
                            str(post_train_eval_path)
                            if post_train_eval_path is not None and post_train_eval_path.exists()
                            else None
                        ),
                        "post_train_probe_eval": (
                            str(probe_eval_path)
                            if probe_eval_path is not None and probe_eval_path.exists()
                            else None
                        ),
                        "publication_summary": str(summary_path) if summary_path.exists() else None,
                    },
                "artifacts": {
                    "latest_durable_checkpoint_prefix": None if checkpoint_prefix is None else str(checkpoint_prefix),
                    "final_archive_prefix": None if final_archive_prefix is None else str(final_archive_prefix),
                },
                "schedule_hashes": manifest.get("terrain_schedule"),
            }
        )

    _write_jsonl(release_dirs["manifests"] / "runs.jsonl", run_records)

    exclude_counts = Counter(
        record["exclude_reason"]
        for record in run_records
        if record["exclude_reason"] is not None
    )
    release_manifest = {
        "schema_version": 1,
        "kind": "sensing_edges_publication_release",
        "release_id": str(release_id),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs_dir": str(outputs_root),
        "release_root": str(release_root_dir),
        "publication_set": publication_set,
        "counts": {
            "indexed_runs": len(run_records),
            "eligible_runs": sum(1 for record in run_records if record["eligible_for_publication"]),
            "ready_runs": sum(1 for record in run_records if record["publication_ready"]),
            "excluded_runs": sum(1 for record in run_records if not record["eligible_for_publication"]),
        },
        "exclude_reasons": dict(sorted(exclude_counts.items())),
        "copied_schedule_dirs": copied_schedule_dirs,
        "paths": {
            "run_index": str(release_dirs["manifests"] / "runs.jsonl"),
            "summaries_dir": str(release_dirs["summaries"]),
            "stats_dir": str(release_dirs["stats"]),
            "figures_dir": str(release_dirs["figures"]),
            "bundles_dir": str(release_dirs["bundles"]),
            "schedules_dir": str(release_dirs["schedules"]),
        },
    }
    _write_json(release_root_dir / "release_manifest.json", release_manifest)
    return release_manifest, run_records


def load_run_index(release_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(release_dir) / "manifests" / "runs.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        token = line.strip()
        if token:
            rows.append(json.loads(token))
    return rows


def load_ready_run_records(release_dir: str | Path) -> list[dict[str, Any]]:
    return [
        record
        for record in load_run_index(release_dir)
        if bool(record.get("publication_ready", False))
    ]


def _analysis_record_from_summary(
    summary: dict[str, Any],
    *,
    path: Path,
) -> dict[str, Any]:
    metadata = dict(summary.get("run_metadata") or {})
    reporting = dict(summary.get("reporting") or {})
    test_summary = dict(reporting.get("test_summary") or {})
    return {
        **metadata,
        "path": str(path),
        "test_mean_fitness": float(reporting.get("test_mean_fitness", 0.0)),
        "test_median_fitness": float(reporting.get("test_median_fitness", 0.0)),
        "test_ci95_low": float(reporting.get("test_ci95_low", 0.0)),
        "test_ci95_high": float(reporting.get("test_ci95_high", 0.0)),
        "test_summary": test_summary,
    }


def load_eval_analysis_records(release_dir: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run_record in load_ready_run_records(release_dir):
        summary_path = run_record.get("paths", {}).get("publication_summary")
        if not summary_path:
            continue
        path = Path(summary_path)
        if not path.exists():
            continue
        records.append(_analysis_record_from_summary(_read_json(path), path=path))
    return records


def load_metric_runs(release_dir: str | Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for record in load_ready_run_records(release_dir):
        metrics_path = record.get("paths", {}).get("metrics")
        if not metrics_path:
            continue
        path = Path(metrics_path)
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        runs.append(
            {
                "name": str(record.get("run_dir_name")),
                "metadata": dict(record.get("run_metadata") or {}),
                "metrics": rows,
            }
        )
    return runs


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    medians = np.array(
        [np.median(rng.choice(values, size=n, replace=True)) for _ in range(n_boot)],
        dtype=float,
    )
    return (
        float(np.percentile(medians, 100 * alpha / 2)),
        float(np.percentile(medians, 100 * (1 - alpha / 2))),
    )


def cohens_dz(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    std = float(np.std(values, ddof=1))
    if std < 1.0e-12:
        return 0.0
    return float(np.mean(values) / std)


def _metric_value(record: dict[str, Any], metric_name: str) -> float:
    if metric_name.startswith("test_"):
        return float(record.get(metric_name, 0.0))
    summary = _normalize_test_summary(record.get("test_summary"))
    return float(summary.get(metric_name, 0.0))


def _comparison_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["preset_id"],
        record["body_variant"],
        record["controller_size"],
        int(record["n_generations"]),
        int(record["total_population"]),
        int(record["physics_substeps"]),
        int(record["train_max_steps"]),
        int(record["rerank_max_steps"]),
        int(record["test_max_steps"]),
        str(record.get("reference_convergence_label", "")),
    )


def generate_deformation_ablation_tables(release_dir: str | Path) -> list[dict[str, Any]]:
    records = load_eval_analysis_records(release_dir)
    grouped: dict[tuple[Any, ...], dict[str, dict[int, dict[str, Any]]]] = defaultdict(
        lambda: {"on": {}, "off": {}}
    )
    for record in records:
        deform = str(record.get("deformation_condition", ""))
        if deform not in {"on", "off"}:
            continue
        grouped[_comparison_key(record)][deform][int(record["seed"])] = record

    comparisons: list[dict[str, Any]] = []
    for key, paired in sorted(grouped.items()):
        all_seeds = sorted(set(paired["on"]) | set(paired["off"]))
        seeds = sorted(set(paired["on"]) & set(paired["off"]))
        if not seeds:
            continue

        exemplar = paired["on"][seeds[0]]
        metrics: dict[str, dict[str, Any]] = {}
        for metric_name in _ABLATION_METRICS:
            on_values = np.asarray([_metric_value(paired["on"][seed], metric_name) for seed in seeds], dtype=float)
            off_values = np.asarray([_metric_value(paired["off"][seed], metric_name) for seed in seeds], dtype=float)
            deltas = on_values - off_values
            ci_low, ci_high = bootstrap_ci(deltas, seed=0)
            if len(deltas) < 2 or np.allclose(deltas, deltas[0]):
                wilcoxon_p = 1.0
            else:
                try:
                    from scipy import stats

                    wilcoxon = stats.wilcoxon(deltas, zero_method="wilcox", correction=False, alternative="two-sided")
                    wilcoxon_p = float(wilcoxon.pvalue)
                except (ImportError, ValueError):
                    wilcoxon_p = 1.0
            metrics[metric_name] = {
                "n_pairs": len(seeds),
                "on_mean": float(np.mean(on_values)),
                "off_mean": float(np.mean(off_values)),
                "delta_mean": float(np.mean(deltas)),
                "delta_median": float(np.median(deltas)),
                "delta_bootstrap_ci95_low": ci_low,
                "delta_bootstrap_ci95_high": ci_high,
                "cohens_dz": cohens_dz(deltas),
                "wilcoxon_p": wilcoxon_p,
                "paired_rows": [
                    {
                        "seed": int(seed),
                        "on": float(_metric_value(paired["on"][seed], metric_name)),
                        "off": float(_metric_value(paired["off"][seed], metric_name)),
                        "delta": float(_metric_value(paired["on"][seed], metric_name) - _metric_value(paired["off"][seed], metric_name)),
                    }
                    for seed in seeds
                ],
                "missing_on_seeds": [int(seed) for seed in sorted(set(all_seeds) - set(paired["on"]))],
                "missing_off_seeds": [int(seed) for seed in sorted(set(all_seeds) - set(paired["off"]))],
            }

        comparisons.append(
            {
                "preset_id": exemplar["preset_id"],
                "body_variant": exemplar["body_variant"],
                "controller_size": exemplar["controller_size"],
                "n_generations": int(exemplar["n_generations"]),
                "total_population": int(exemplar["total_population"]),
                "physics_substeps": int(exemplar["physics_substeps"]),
                "train_max_steps": int(exemplar["train_max_steps"]),
                "rerank_max_steps": int(exemplar["rerank_max_steps"]),
                "test_max_steps": int(exemplar["test_max_steps"]),
                "reference_convergence_label": str(exemplar.get("reference_convergence_label", "")),
                "paired_seeds": [int(seed) for seed in seeds],
                "metrics": metrics,
            }
        )

    stats_dir = Path(release_dir) / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    _write_json(stats_dir / "deformation_ablation.json", {"comparisons": comparisons})
    _write_json(stats_dir / "effect_sizes.json", {"comparisons": comparisons})
    return comparisons


def _condition_label(metadata: dict[str, Any]) -> str:
    rerank_steps = int(metadata["rerank_max_steps"])
    test_steps = int(metadata["test_max_steps"])
    eval_label = f"e{rerank_steps}" if rerank_steps == test_steps else f"er{rerank_steps}_et{test_steps}"
    return (
        f"{metadata['body_variant']}_g{metadata['n_generations']}_"
        f"p{metadata['total_population']}_{metadata['physics_substeps']}sub_"
        f"t{metadata['train_max_steps']}_{eval_label}_{metadata.get('reference_convergence_label', '')}"
    )


def _plot_grouped_curves(
    runs: list[dict[str, Any]],
    output_dir: Path,
    *,
    metric_key: str,
    ylabel: str,
    filename_prefix: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[_condition_label(run["metadata"])].append(run)

    for group_label, group_runs in sorted(grouped.items()):
        fig, ax = plt.subplots(figsize=(8, 5))
        legend_done: set[str] = set()
        for run in sorted(group_runs, key=lambda item: (item["metadata"]["deformation_condition"], item["metadata"]["seed"])):
            metadata = run["metadata"]
            deform = str(metadata["deformation_condition"])
            color = _DEFORM_COLORS.get(deform, "#555555")
            label = f"deform={deform}"
            if label in legend_done:
                label = None
            else:
                legend_done.add(label)
            ax.plot(
                [row.get("generation", 0) for row in run["metrics"]],
                [row.get(metric_key, 0.0) for row in run["metrics"]],
                color=color,
                alpha=0.35,
                linewidth=1.0,
                label=label,
            )

        ax.set_xlabel("Generation")
        ax.set_ylabel(ylabel)
        ax.set_title(group_label.replace("_", " "))
        if legend_done:
            ax.legend(frameon=False)
        plt.tight_layout()
        fig.savefig(output_dir / f"{filename_prefix}_{group_label}.pdf", dpi=150)
        plt.close(fig)


def generate_publication_figures(release_dir: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = Path(release_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    runs = load_metric_runs(release_dir)
    if runs:
        _plot_grouped_curves(
            runs,
            figure_dir,
            metric_key="best_fitness",
            ylabel="Best Fitness (%)",
            filename_prefix="fitness",
        )
        _plot_grouped_curves(
            runs,
            figure_dir,
            metric_key="coverage",
            ylabel="Archive Coverage",
            filename_prefix="coverage",
        )

    records = load_eval_analysis_records(release_dir)
    if not records:
        return

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[_condition_label(record)][str(record["deformation_condition"])].append(record)

    body_variants = [label for label in sorted(grouped) if grouped[label]]
    if not body_variants:
        return

    x = np.arange(len(body_variants))
    width = 0.34
    ncols = 2
    nrows = int(np.ceil(len(_SUMMARY_METRICS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, max(6, 3.2 * nrows)))
    flat_axes = np.asarray(axes).reshape(-1)

    for axis, (metric_key, title) in zip(flat_axes, _SUMMARY_METRICS):
        on_values: list[float] = []
        off_values: list[float] = []
        for body in body_variants:
            on_records = grouped[body].get("on", [])
            off_records = grouped[body].get("off", [])
            if metric_key.startswith("test_"):
                on_values.append(np.mean([record[metric_key] for record in on_records]) if on_records else np.nan)
                off_values.append(np.mean([record[metric_key] for record in off_records]) if off_records else np.nan)
            else:
                on_values.append(
                    np.mean([_metric_value(record, metric_key) for record in on_records]) if on_records else np.nan
                )
                off_values.append(
                    np.mean([_metric_value(record, metric_key) for record in off_records]) if off_records else np.nan
                )

        axis.bar(x - width / 2, on_values, width=width, color=_DEFORM_COLORS["on"], label="deform=on")
        axis.bar(x + width / 2, off_values, width=width, color=_DEFORM_COLORS["off"], label="deform=off")
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(body_variants, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.2)

    for axis in flat_axes[len(_SUMMARY_METRICS):]:
        axis.axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figure_dir / "deformation_ablation_summary.pdf", dpi=150)
    plt.close(fig)


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def bundle_publication_release(
    release_dir: str | Path,
    *,
    output: str | Path | None = None,
    se_root: str | Path | None = None,
) -> Path:
    root = Path(release_dir)
    base_root = Path(se_root) if se_root is not None else Path(__file__).resolve().parent.parent
    release_manifest = _read_json(root / "release_manifest.json")
    run_records = load_ready_run_records(root)

    bundle_dir = root / "bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output) if output is not None else bundle_dir / f"{root.name}.tar.gz"

    entries: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            entries.append((path, str(Path(PUBLICATION_DIRNAME) / root.name / path.relative_to(root))))

    for record in run_records:
        for key in ("manifest", "resolved_config", "metrics", "post_train_eval", "post_train_probe_eval"):
            raw_path = record.get("paths", {}).get(key)
            if raw_path:
                path = Path(raw_path)
                if path.exists():
                    entries.append((path, f"outputs/{record['run_dir_name']}/{path.name}"))
        for artifact_key in ("latest_durable_checkpoint_prefix", "final_archive_prefix"):
            raw_prefix = record.get("artifacts", {}).get(artifact_key)
            if not raw_prefix:
                continue
            prefix = Path(raw_prefix)
            for suffix in (".json", ".npz", ".done"):
                path = prefix.with_suffix(suffix)
                if path.exists():
                    entries.append((path, f"outputs/{record['run_dir_name']}/{path.parent.name}/{path.name}" if path.parent.name == "checkpoints" else f"outputs/{record['run_dir_name']}/{path.name}"))

    schema_names = (
        "checkpoint.schema.json",
        "final_archive.schema.json",
        "manifest.schema.json",
        "publication_release.schema.json",
        "publication_run.schema.json",
        "publication_summary.schema.json",
    )
    for schema_name in schema_names:
        path = base_root / "schemas" / schema_name
        if path.exists():
            entries.append((path, str(path.relative_to(base_root))))

    publication_config_path = release_manifest.get("publication_set", {}).get("config_path")
    if publication_config_path:
        config_path = Path(publication_config_path)
        if config_path.exists():
            entries.append((config_path, str(config_path.relative_to(base_root))))

    deduped: dict[str, Path] = {}
    for path, arcname in entries:
        deduped.setdefault(arcname, path)

    with tarfile.open(output_path, "w:gz") as tar:
        for arcname, path in sorted(deduped.items()):
            tar.add(path, arcname=arcname)

    sha_path = output_path.with_suffix(".sha256")
    sha_path.write_text(f"{_sha256_file(output_path)}  {output_path.name}\n")
    return output_path
