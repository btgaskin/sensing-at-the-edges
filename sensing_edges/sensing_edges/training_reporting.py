"""Generation-level training metrics and logging helpers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .deformation_probe import deformation_channel_max_dict
from .fitness import CANDIDATE_SCALAR_METRIC_ATTRS

if TYPE_CHECKING:
    from .batch_eval import PopulationBatchSummary
    from .fitness import CandidateBatchResult
    from .qd.cma_emitter import CMAEmitter
    from .qd.grid_archive import GridArchive


TERRAIN_DIAGNOSTIC_BUCKETS = (
    (2, 1, "g2_c1"),
    (4, 2, "g4_c2"),
    (6, 3, "g6_c3"),
)

TERRAIN_DIAGNOSTIC_METRICS = (
    "hit_rate",
    "fall_rate",
    "mean_fitness",
    "directional_regularity",
    "best_goal_time",
)


@dataclass(frozen=True)
class GenerationReport:
    """Canonical per-generation summary used by stdout, JSONL, and W&B."""

    generation: int
    population: int
    elapsed_s: float
    best_fitness: float
    best_generation_fitness: float
    mean_fitness: float
    qd_score: float
    coverage: float
    threshold_mean: float
    n_occupied: int
    new_cells: int
    improved_cells: int
    invalid_descriptor_count: int
    mean_fall_count: float
    goal_hit_mean: float
    mean_mirrored_pair_count: float
    emitter_restarts: list[int]
    emitter_total_restarts: int
    emitter_mean_sigma: float
    emitter_improvement_rate: float
    candidates_per_s: float
    episodes_per_s: float = 0.0
    ask_s: float = 0.0
    eval_s: float = 0.0
    archive_s: float = 0.0
    tell_s: float = 0.0
    checkpoint_s: float = 0.0
    render_s: float = 0.0
    wandb_s: float = 0.0
    chunk_count: int = 0
    mean_chunk_size: float = 0.0
    max_chunk_size: int = 0
    bucket_chunk_counts: dict[int, int] | None = None
    checkpoint_path: str | None = None
    deformation_channel_max: dict[str, float] | None = None
    controller_hidden_output_max_abs: float = 0.0
    action_output_max_abs: float = 0.0
    lower_ledge_penalty_mean: float = 0.0
    max_turn_count: float = 0.0
    max_stable_turn_count: float = 0.0
    nonzero_turn_fraction: float = 0.0
    nonzero_stable_turn_fraction: float = 0.0
    qd_axis_coverage: dict[str, float] | None = None
    qd_axis_occupied_bins: dict[str, int] | None = None
    qd_focus_metrics: dict[str, float] | None = None
    descriptor_means: dict[str, float] | None = None
    terrain_diagnostics: dict[str, float] | None = None

    def with_checkpoint_path(self, checkpoint_path: str | None) -> "GenerationReport":
        """Return a copy of the report with checkpoint metadata attached."""
        return replace(self, checkpoint_path=checkpoint_path)

    def metrics_row(self) -> dict:
        """Return the JSONL row persisted to metrics.jsonl."""
        row = {
            "generation": self.generation,
            "population": self.population,
            "elapsed_s": round(self.elapsed_s, 2),
            "best_fitness": round(self.best_fitness, 4),
            "best_generation_fitness": round(self.best_generation_fitness, 4),
            "mean_fitness": round(self.mean_fitness, 4),
            "qd_score": round(self.qd_score, 4),
            "coverage": round(self.coverage, 6),
            "threshold_mean": round(self.threshold_mean, 6),
            "n_occupied": self.n_occupied,
            "new_cells": self.new_cells,
            "improved_cells": self.improved_cells,
            "invalid_descriptor_count": self.invalid_descriptor_count,
            "mean_fall_count": round(self.mean_fall_count, 4),
            "goal_hit_mean": round(self.goal_hit_mean, 4),
            "mean_mirrored_pair_count": round(self.mean_mirrored_pair_count, 4),
            "lower_ledge_penalty_mean": round(self.lower_ledge_penalty_mean, 4),
            "emitter_restarts": self.emitter_restarts,
            "total_restarts": self.emitter_total_restarts,
            "mean_sigma": round(self.emitter_mean_sigma, 6),
            "improvement_rate": round(self.emitter_improvement_rate, 4),
            "candidates_per_s": round(self.candidates_per_s, 1),
            "episodes_per_s": round(self.episodes_per_s, 1),
            "ask_s": round(self.ask_s, 4),
            "eval_s": round(self.eval_s, 4),
            "archive_s": round(self.archive_s, 4),
            "tell_s": round(self.tell_s, 4),
            "checkpoint_s": round(self.checkpoint_s, 4),
            "render_s": round(self.render_s, 4),
            "wandb_s": round(self.wandb_s, 4),
            "chunk_count": self.chunk_count,
            "mean_chunk_size": round(self.mean_chunk_size, 6),
            "max_chunk_size": self.max_chunk_size,
            "bucket_chunk_counts": {
                str(bucket): count
                for bucket, count in (self.bucket_chunk_counts or {}).items()
            },
            "checkpoint_path": self.checkpoint_path,
            "deformation_channel_max": self.deformation_channel_max,
            "controller_hidden_output_max_abs": round(self.controller_hidden_output_max_abs, 6),
            "action_output_max_abs": round(self.action_output_max_abs, 6),
            "max_turn_count": round(self.max_turn_count, 4),
            "max_stable_turn_count": round(self.max_stable_turn_count, 4),
            "nonzero_turn_fraction": round(self.nonzero_turn_fraction, 4),
            "nonzero_stable_turn_fraction": round(self.nonzero_stable_turn_fraction, 4),
        }
        if self.descriptor_means:
            row["descriptor_means"] = {
                key: round(float(value), 4)
                for key, value in self.descriptor_means.items()
            }
            row.update(
                {
                    f"mean_{key}": round(float(value), 4)
                    for key, value in self.descriptor_means.items()
                }
            )
        if self.qd_axis_coverage:
            row["qd_axis_coverage"] = {
                key: round(float(value), 6)
                for key, value in self.qd_axis_coverage.items()
            }
            row.update(
                {
                    f"qd_axis_coverage_{key}": round(float(value), 6)
                    for key, value in self.qd_axis_coverage.items()
                }
            )
        if self.qd_axis_occupied_bins:
            row["qd_axis_occupied_bins"] = {
                key: int(value)
                for key, value in self.qd_axis_occupied_bins.items()
            }
            row.update(
                {
                    f"qd_axis_occupied_bins_{key}": int(value)
                    for key, value in self.qd_axis_occupied_bins.items()
                }
            )
        if self.qd_focus_metrics:
            row["qd_focus_metrics"] = {
                key: round(float(value), 6)
                for key, value in self.qd_focus_metrics.items()
            }
            row.update(
                {
                    key: round(float(value), 6)
                    for key, value in self.qd_focus_metrics.items()
                }
            )
        if self.terrain_diagnostics:
            row["terrain_diagnostics"] = {
                key: round(float(value), 4)
                for key, value in self.terrain_diagnostics.items()
            }
        return row

    def format_for_log(self) -> str:
        """Return a concise human-readable training line."""
        checkpoint_suffix = ""
        if self.checkpoint_path:
            checkpoint_suffix = f" ckpt={Path(self.checkpoint_path).name}"
        focus_suffix = ""
        if self.qd_focus_metrics:
            active_fraction = self.qd_focus_metrics.get("qd_focus_active_fraction")
            if active_fraction is not None:
                focus_suffix = f" focus={float(active_fraction):.3f}"

        return (
            "Gen %3d | best=%.2f gen_best=%.2f qd=%.1f cov=%.3f "
            "new=%d imp=%d inv_desc=%d falls=%.2f hits=%.2f mir=%.2f restart=%d "
            "thrpt=%.1f/s eps=%.1f/s chunks=%d mean_chunk=%.1f max_chunk=%d "
            "rend=%.1fs wb=%.2fs (%.1fs)%s%s"
        ) % (
            self.generation,
            self.best_fitness,
            self.best_generation_fitness,
            self.qd_score,
            self.coverage,
            self.new_cells,
            self.improved_cells,
            self.invalid_descriptor_count,
            self.mean_fall_count,
            self.goal_hit_mean,
            self.mean_mirrored_pair_count,
            self.emitter_total_restarts,
            self.candidates_per_s,
            self.episodes_per_s,
            self.chunk_count,
            self.mean_chunk_size,
            self.max_chunk_size,
            self.render_s,
            self.wandb_s,
            self.elapsed_s,
            focus_suffix,
            checkpoint_suffix,
        )


def _descriptor_mean_columns(
    batch_result: "CandidateBatchResult | PopulationBatchSummary",
) -> dict[str, float]:
    means: dict[str, float] = {}
    for attr in CANDIDATE_SCALAR_METRIC_ATTRS:
        values = _candidate_scalar_values(batch_result, attr)
        if values is None or not values.size:
            continue
        means[attr] = float(np.mean(values))
    return means


def _candidate_scalar_values(
    batch_result: "CandidateBatchResult | PopulationBatchSummary",
    attr: str,
) -> np.ndarray | None:
    descriptor_columns = getattr(batch_result, "descriptor_columns", None) or {}
    if hasattr(batch_result, attr):
        return np.asarray(getattr(batch_result, attr), dtype=np.float32)
    if attr in descriptor_columns:
        return np.asarray(descriptor_columns[attr], dtype=np.float32)
    return None


def _episode_metric_values(
    batch_result: "CandidateBatchResult | PopulationBatchSummary",
    attr: str,
    *,
    dtype,
) -> np.ndarray | None:
    values = getattr(batch_result, attr, None)
    if values is None:
        return None
    values = np.asarray(values, dtype=dtype)
    if values.ndim != 2:
        return None
    return values


def _terrain_bucket_diagnostics(
    *,
    episode_fitnesses: np.ndarray,
    episode_goal_touched: np.ndarray,
    episode_fell: np.ndarray,
    episode_goal_distances: np.ndarray,
    episode_cliff_distances: np.ndarray,
    episode_times_to_goal: np.ndarray,
    episode_first_significant_directions: np.ndarray,
) -> dict[str, float]:
    def _bucket_metrics(mask: np.ndarray) -> dict[str, float]:
        if not np.any(mask):
            return {
                "hit_rate": float("nan"),
                "fall_rate": float("nan"),
                "mean_fitness": float("nan"),
                "directional_regularity": float("nan"),
                "best_goal_time": float("nan"),
            }

        candidate_scores: list[float] = []
        for candidate_dirs, candidate_mask in zip(
            episode_first_significant_directions,
            mask,
            strict=False,
        ):
            selected_count = int(np.sum(candidate_mask))
            if selected_count <= 0:
                continue
            selected_dirs = np.asarray(candidate_dirs[candidate_mask], dtype=np.int32)
            left_count = int(np.sum(selected_dirs == -1))
            right_count = int(np.sum(selected_dirs == 1))
            candidate_scores.append(float(max(left_count, right_count)) / float(selected_dirs.size))

        success_mask = mask & episode_goal_touched
        return {
            "hit_rate": float(np.mean(episode_goal_touched[mask], dtype=np.float32)),
            "fall_rate": float(np.mean(episode_fell[mask], dtype=np.float32)),
            "mean_fitness": float(np.mean(episode_fitnesses[mask], dtype=np.float32)),
            "directional_regularity": (
                float(np.mean(np.asarray(candidate_scores, dtype=np.float32)))
                if candidate_scores
                else 0.0
            ),
            "best_goal_time": (
                float(np.min(episode_times_to_goal[success_mask]))
                if np.any(success_mask)
                else float("nan")
            ),
        }

    diagnostics: dict[str, float] = {}
    all_mask = np.zeros_like(episode_goal_touched, dtype=bool)
    for goal_distance, cliff_distance, label in TERRAIN_DIAGNOSTIC_BUCKETS:
        mask = (
            (episode_goal_distances == int(goal_distance))
            & (episode_cliff_distances == int(cliff_distance))
        )
        all_mask |= mask
        for metric_name, metric_value in _bucket_metrics(mask).items():
            diagnostics[f"{label}/{metric_name}"] = metric_value
    if not np.any(all_mask):
        return {}
    for metric_name, metric_value in _bucket_metrics(all_mask).items():
        diagnostics[f"all/{metric_name}"] = metric_value
    return diagnostics


def _build_terrain_diagnostics(
    batch_result: "CandidateBatchResult | PopulationBatchSummary",
) -> dict[str, float] | None:
    episode_fitnesses = _episode_metric_values(
        batch_result,
        "episode_fitnesses",
        dtype=np.float32,
    )
    episode_goal_touched = _episode_metric_values(
        batch_result,
        "episode_goal_touched",
        dtype=bool,
    )
    episode_fell = _episode_metric_values(
        batch_result,
        "episode_fell",
        dtype=bool,
    )
    episode_goal_distances = _episode_metric_values(
        batch_result,
        "episode_goal_distances",
        dtype=np.int32,
    )
    episode_cliff_distances = _episode_metric_values(
        batch_result,
        "episode_cliff_distances",
        dtype=np.int32,
    )
    episode_times_to_goal = _episode_metric_values(
        batch_result,
        "episode_times_to_goal",
        dtype=np.float32,
    )
    episode_first_significant_directions = _episode_metric_values(
        batch_result,
        "episode_first_significant_directions",
        dtype=np.int32,
    )
    if any(
        values is None
        for values in (
            episode_fitnesses,
            episode_goal_touched,
            episode_fell,
            episode_goal_distances,
            episode_cliff_distances,
            episode_times_to_goal,
            episode_first_significant_directions,
        )
    ):
        return None
    return _terrain_bucket_diagnostics(
        episode_fitnesses=episode_fitnesses,
        episode_goal_touched=episode_goal_touched,
        episode_fell=episode_fell,
        episode_goal_distances=episode_goal_distances,
        episode_cliff_distances=episode_cliff_distances,
        episode_times_to_goal=episode_times_to_goal,
        episode_first_significant_directions=episode_first_significant_directions,
    )


def build_generation_report(
    gen: int,
    archive: "GridArchive",
    emitters: list["CMAEmitter"],
    batch_result: "CandidateBatchResult | PopulationBatchSummary",
    n_new: int,
    n_improved: int,
    invalid_descriptor_count: int,
    elapsed_s: float,
    *,
    ask_s: float = 0.0,
    eval_s: float = 0.0,
    archive_s: float = 0.0,
    tell_s: float = 0.0,
    checkpoint_s: float = 0.0,
    render_s: float = 0.0,
    wandb_s: float = 0.0,
    episodes_per_candidate: int = 1,
    chunk_count: int = 0,
    mean_chunk_size: float = 0.0,
    max_chunk_size: int = 0,
    bucket_chunk_counts: dict[int, int] | None = None,
    qd_focus_metrics: dict[str, float] | None = None,
) -> GenerationReport:
    """Build the canonical generation report."""
    population = int(batch_result.fitness.shape[0])
    mean_fall_count = 0.0
    if hasattr(batch_result, "fall_count"):
        fall_count = np.asarray(batch_result.fall_count, dtype=np.float32)
        if fall_count.size:
            mean_fall_count = float(np.mean(fall_count))
    else:
        episode_fell = np.asarray(batch_result.episode_fell, dtype=np.float32)
        if episode_fell.size:
            mean_fall_count = float(np.mean(np.sum(episode_fell, axis=1)))
    goal_hit_mean = 0.0
    if hasattr(batch_result, "goal_hit_count"):
        goal_hit_count = np.asarray(batch_result.goal_hit_count, dtype=np.float32)
        if goal_hit_count.size:
            goal_hit_mean = float(np.mean(goal_hit_count))
    elif hasattr(batch_result, "episode_goal_touched"):
        episode_goal_touched = np.asarray(batch_result.episode_goal_touched, dtype=bool)
        if episode_goal_touched.size:
            goal_hit_mean = float(np.mean(np.sum(episode_goal_touched, axis=1, dtype=np.int32)))
    mean_mirrored_pair_count = 0.0
    if hasattr(batch_result, "mirrored_pair_count"):
        mirrored_pair_count = np.asarray(batch_result.mirrored_pair_count, dtype=np.float32)
        if mirrored_pair_count.size:
            mean_mirrored_pair_count = float(np.mean(mirrored_pair_count))
    lower_ledge_penalty_mean = 0.0
    if hasattr(batch_result, "lower_ledge_penalty"):
        lower_ledge_penalty = np.asarray(batch_result.lower_ledge_penalty, dtype=np.float32)
        if lower_ledge_penalty.size:
            lower_ledge_penalty_mean = float(np.mean(lower_ledge_penalty))
    turn_values = _candidate_scalar_values(batch_result, "turn_count")
    stable_turn_values = _candidate_scalar_values(batch_result, "stable_turn_count")
    max_turn_count = float(np.max(turn_values)) if turn_values is not None and turn_values.size else 0.0
    max_stable_turn_count = (
        float(np.max(stable_turn_values))
        if stable_turn_values is not None and stable_turn_values.size
        else 0.0
    )
    nonzero_turn_fraction = (
        float(np.mean(turn_values > 0.0))
        if turn_values is not None and turn_values.size
        else 0.0
    )
    nonzero_stable_turn_fraction = (
        float(np.mean(stable_turn_values > 0.0))
        if stable_turn_values is not None and stable_turn_values.size
        else 0.0
    )
    descriptor_means = _descriptor_mean_columns(batch_result)
    terrain_diagnostics = _build_terrain_diagnostics(batch_result)
    generation_fitnesses = np.asarray(batch_result.fitness, dtype=np.float64)
    best_generation_fitness = float(np.max(generation_fitnesses)) if population else 0.0
    mean_generation_fitness = float(np.mean(generation_fitnesses)) if population else 0.0
    deformation_channel_max = None
    controller_hidden_output_max_abs = 0.0
    action_output_max_abs = 0.0
    if population and hasattr(batch_result, "deformation_channel_max"):
        deformation_max = np.asarray(batch_result.deformation_channel_max, dtype=np.float32)
        if deformation_max.size:
            deformation_channel_max = deformation_channel_max_dict(
                np.max(deformation_max, axis=0)
            )
    if population and hasattr(batch_result, "controller_hidden_output_max_abs"):
        controller_max = np.asarray(
            batch_result.controller_hidden_output_max_abs,
            dtype=np.float32,
        )
        if controller_max.size:
            controller_hidden_output_max_abs = float(np.max(controller_max))
    if population and hasattr(batch_result, "action_output_max_abs"):
        action_max = np.asarray(batch_result.action_output_max_abs, dtype=np.float32)
        if action_max.size:
            action_output_max_abs = float(np.max(action_max))

    sigmas = []
    for emitter in emitters:
        if emitter.es_state is None:
            continue
        std = getattr(emitter.es_state, "std", None)
        if std is not None:
            sigmas.append(float(np.max(np.asarray(std))))

    return GenerationReport(
        generation=gen,
        population=population,
        elapsed_s=elapsed_s,
        best_fitness=float(archive.best_fitness),
        best_generation_fitness=float(best_generation_fitness),
        mean_fitness=mean_generation_fitness,
        qd_score=float(archive.qd_score),
        coverage=float(archive.coverage),
        threshold_mean=float(archive.occupied_threshold_mean),
        n_occupied=int(archive.size),
        new_cells=int(n_new),
        improved_cells=int(n_improved),
        invalid_descriptor_count=int(invalid_descriptor_count),
        mean_fall_count=mean_fall_count,
        goal_hit_mean=goal_hit_mean,
        mean_mirrored_pair_count=mean_mirrored_pair_count,
        lower_ledge_penalty_mean=lower_ledge_penalty_mean,
        emitter_restarts=[int(emitter.total_restarts) for emitter in emitters],
        emitter_total_restarts=int(sum(emitter.total_restarts for emitter in emitters)),
        emitter_mean_sigma=float(np.mean(sigmas)) if sigmas else 0.0,
        emitter_improvement_rate=float((n_new + n_improved) / max(population, 1)),
        candidates_per_s=float(population / max(elapsed_s, 1e-6)),
        episodes_per_s=float(
            (population * max(episodes_per_candidate, 1)) / max(elapsed_s, 1e-6)
        ),
        ask_s=float(ask_s),
        eval_s=float(eval_s),
        archive_s=float(archive_s),
        tell_s=float(tell_s),
        checkpoint_s=float(checkpoint_s),
        render_s=float(render_s),
        wandb_s=float(wandb_s),
        chunk_count=int(chunk_count),
        mean_chunk_size=float(mean_chunk_size),
        max_chunk_size=int(max_chunk_size),
        bucket_chunk_counts=dict(bucket_chunk_counts or {}),
        deformation_channel_max=deformation_channel_max,
        controller_hidden_output_max_abs=controller_hidden_output_max_abs,
        action_output_max_abs=action_output_max_abs,
        max_turn_count=max_turn_count,
        max_stable_turn_count=max_stable_turn_count,
        nonzero_turn_fraction=nonzero_turn_fraction,
        nonzero_stable_turn_fraction=nonzero_stable_turn_fraction,
        qd_axis_coverage=dict(getattr(archive, "axis_coverage", {})),
        qd_axis_occupied_bins=dict(getattr(archive, "axis_occupied_bins", {})),
        qd_focus_metrics=None if qd_focus_metrics is None else dict(qd_focus_metrics),
        descriptor_means=descriptor_means,
        terrain_diagnostics=terrain_diagnostics,
    )


def append_metrics_row(metrics_path: Path, report: GenerationReport) -> None:
    """Append a generation report to metrics.jsonl."""
    with open(metrics_path, "a") as handle:
        handle.write(json.dumps(report.metrics_row(), default=str) + "\n")


def truncate_metrics_rows(metrics_path: Path, *, max_generation: int) -> None:
    """Keep only metrics rows up to and including `max_generation`."""
    if not metrics_path.exists():
        return
    retained: list[str] = []
    with open(metrics_path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if int(row.get("generation", -1)) <= int(max_generation):
                retained.append(json.dumps(row, default=str))
    with open(metrics_path, "w") as handle:
        if retained:
            handle.write("\n".join(retained) + "\n")


def log_generation_report(report: GenerationReport, logger: logging.Logger | None = None) -> None:
    """Emit a concise stdout log line for a generation."""
    active_logger = logger or logging.getLogger(__name__)
    active_logger.info(report.format_for_log())
