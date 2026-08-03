"""wandb metric builders and archive visualization.

Direct namespaced keys — no aliasing, no routing, no profile tiers.
The key you construct IS the key wandb sees.

Dashboard sections (in display order):
  fitness/   — hero panel (best, mean, mean_fall_count, goal hits, mirrored pairs, soft ledge penalties)
  descriptors/ — population means for candidate-level descriptor metrics
  qd/        — archive health (coverage, score)
  trials/    — GIF renders of top candidates
  archive/   — heatmap and histogram images
  emitters/  — search health (restarts, sigma)
  timing/    — performance (gen_s, throughput)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .fitness import CandidateResult
    from .qd.focus import FocusWindowBounds
    from .qd.grid_archive import GridArchive
    from .training_reporting import GenerationReport

from .deformation_probe import DEFORMATION_CHANNEL_KEYS
from .runtime_config import run_group_label, run_metadata, run_name_label
from .training_reporting import TERRAIN_DIAGNOSTIC_BUCKETS


def define_wandb_metrics() -> None:
    """Set step metrics and summary types for clean charts.

    Call once after wandb.init().
    """
    import wandb

    wandb.define_metric("generation")

    # Fitness — summarize as max (best achievement matters).
    for key in [
        "fitness/best",
        "fitness/mean",
        "fitness/best_gen",
        "fitness/goal_hit_mean",
        "fitness/mean_mirrored_pair_count",
    ]:
        wandb.define_metric(key, step_metric="generation", summary="max")
    wandb.define_metric("fitness/mean_fall_count", step_metric="generation", summary="min")
    wandb.define_metric("fitness/lower_ledge_penalty_mean", step_metric="generation", summary="min")
    wandb.define_metric("descriptors/*", step_metric="generation", summary="last")

    # QD — coverage and score as max.
    for key in ["qd/coverage", "qd/score"]:
        wandb.define_metric(key, step_metric="generation", summary="max")
    wandb.define_metric("qd/axis_coverage/*", step_metric="generation", summary="max")
    wandb.define_metric("qd/axis_occupied_bins/*", step_metric="generation", summary="max")
    wandb.define_metric("qd/focus/*", step_metric="generation", summary="last")
    for key in ["qd/new_cells", "qd/improved", "qd/threshold_mean"]:
        wandb.define_metric(key, step_metric="generation", summary="last")
    wandb.define_metric("qd/invalid_descriptors", step_metric="generation", summary="mean")

    # Emitters — summary as last.
    wandb.define_metric("emitters/*", step_metric="generation", summary="last")

    # Timing — summary as mean.
    wandb.define_metric("timing/*", step_metric="generation", summary="mean")

    # Deformation probe — raw physical channel maxima, summarize as max.
    wandb.define_metric("deformation/*", step_metric="generation", summary="max")
    wandb.define_metric("alignment/*", step_metric="generation", summary="max")
    for _goal_distance, _cliff_distance, label in (*TERRAIN_DIAGNOSTIC_BUCKETS, (None, None, "all")):
        wandb.define_metric(
            f"terrain_diag/{label}/hit_rate",
            step_metric="generation",
            summary="max",
        )
        wandb.define_metric(
            f"terrain_diag/{label}/fall_rate",
            step_metric="generation",
            summary="min",
        )
        wandb.define_metric(
            f"terrain_diag/{label}/mean_fitness",
            step_metric="generation",
            summary="last",
        )
        wandb.define_metric(
            f"terrain_diag/{label}/directional_regularity",
            step_metric="generation",
            summary="max",
        )
        wandb.define_metric(
            f"terrain_diag/{label}/best_goal_time",
            step_metric="generation",
            summary="min",
        )

    # Post-train eval — logged once at the final generation.
    wandb.define_metric("post_train/*", step_metric="generation", summary="last")
    wandb.define_metric("post_train_deform/*", step_metric="generation", summary="last")
    wandb.define_metric("specimens/*", step_metric="generation", summary="last")


def build_generation_metrics(report: "GenerationReport") -> dict:
    """Build the full scalar metric dict for one generation."""
    metrics = {
        "generation": report.generation,
        # fitness/
        "fitness/best": report.best_fitness,
        "fitness/mean": report.mean_fitness,
        "fitness/best_gen": report.best_generation_fitness,
        "fitness/mean_fall_count": report.mean_fall_count,
        "fitness/goal_hit_mean": report.goal_hit_mean,
        "fitness/mean_mirrored_pair_count": report.mean_mirrored_pair_count,
        "fitness/lower_ledge_penalty_mean": report.lower_ledge_penalty_mean,
        # qd/
        "qd/coverage": report.coverage,
        "qd/score": report.qd_score,
        "qd/new_cells": report.new_cells,
        "qd/improved": report.improved_cells,
        "qd/threshold_mean": report.threshold_mean,
        "qd/invalid_descriptors": report.invalid_descriptor_count,
        # emitters/
        "emitters/total_restarts": report.emitter_total_restarts,
        "emitters/mean_sigma": report.emitter_mean_sigma,
        "emitters/improvement_rate": report.emitter_improvement_rate,
        # timing/
        "timing/gen_s": round(report.elapsed_s, 2),
        "timing/candidates_per_s": round(report.candidates_per_s, 1),
        "timing/episodes_per_s": round(report.episodes_per_s, 1),
        "timing/ask_s": round(report.ask_s, 4),
        "timing/eval_s": round(report.eval_s, 4),
        "timing/archive_s": round(report.archive_s, 4),
        "timing/tell_s": round(report.tell_s, 4),
        "timing/checkpoint_s": round(report.checkpoint_s, 4),
        "timing/render_s": round(report.render_s, 4),
        "timing/chunk_count": report.chunk_count,
        "timing/mean_chunk_size": round(report.mean_chunk_size, 6),
        "timing/max_chunk_size": report.max_chunk_size,
    }
    if report.deformation_channel_max:
        metrics.update(
            {
                f"deformation/{channel_key}_max_abs": float(
                    report.deformation_channel_max[channel_key]
                )
                for channel_key in DEFORMATION_CHANNEL_KEYS
            }
        )
    metrics["alignment/controller_hidden_output_max_abs"] = float(
        report.controller_hidden_output_max_abs
    )
    metrics["alignment/action_output_max_abs"] = float(
        report.action_output_max_abs
    )
    if report.descriptor_means:
        metrics.update(
            {
                f"descriptors/mean_{name}": float(value)
                for name, value in report.descriptor_means.items()
            }
        )
    if report.qd_axis_coverage:
        metrics.update(
            {
                f"qd/axis_coverage/{name}": float(value)
                for name, value in report.qd_axis_coverage.items()
            }
        )
    if report.qd_axis_occupied_bins:
        metrics.update(
            {
                f"qd/axis_occupied_bins/{name}": float(value)
                for name, value in report.qd_axis_occupied_bins.items()
            }
        )
    if report.qd_focus_metrics:
        for key, value in report.qd_focus_metrics.items():
            if key == "qd_focus_active_cells":
                metrics["qd/focus/active_cell_count"] = float(value)
            elif key == "qd_focus_active_fraction":
                metrics["qd/focus/active_fraction"] = float(value)
            elif key == "qd_focus_occupied_in_active":
                metrics["qd/focus/occupied_in_window"] = float(value)
            elif key == "qd_focus_tightened_this_gen":
                metrics["qd/focus/tightened_this_gen"] = float(value)
            elif key.startswith("qd_focus_") and key.endswith("_window_size"):
                name = key.removeprefix("qd_focus_").removesuffix("_window_size")
                metrics[f"qd/focus/window_size_cells/{name}"] = float(value)
            elif key.startswith("qd_focus_") and key.endswith("_lower"):
                name = key.removeprefix("qd_focus_").removesuffix("_lower")
                metrics[f"qd/focus/lower_cell/{name}"] = float(value)
            elif key.startswith("qd_focus_") and key.endswith("_upper"):
                name = key.removeprefix("qd_focus_").removesuffix("_upper")
                metrics[f"qd/focus/upper_cell/{name}"] = float(value)
    metrics["descriptors/max_turn_count"] = float(report.max_turn_count)
    metrics["descriptors/max_stable_turn_count"] = float(report.max_stable_turn_count)
    metrics["descriptors/nonzero_turn_fraction"] = float(report.nonzero_turn_fraction)
    metrics["descriptors/nonzero_stable_turn_fraction"] = float(
        report.nonzero_stable_turn_fraction
    )
    if report.terrain_diagnostics:
        metrics.update(
            {
                f"terrain_diag/{key}": float(value)
                for key, value in report.terrain_diagnostics.items()
            }
        )
    return metrics


def build_render_metrics(rank: int, result: "CandidateResult", episode_idx: int = 0) -> dict:
    """Per-render metadata logged alongside GIF."""
    return {
        f"trials/rank_{rank}/fitness": result.fitness,
        f"trials/rank_{rank}/cliff_x": result.episode_cliff_xs[episode_idx],
        f"trials/rank_{rank}/fell": int(result.episode_fell[episode_idx]),
        f"trials/rank_{rank}/steps_alive": result.episode_steps_alive[episode_idx],
    }


def _descriptor_axis_labels(archive: "GridArchive") -> tuple[str, ...]:
    labels = tuple(getattr(archive, "descriptor_labels", ()))
    if labels:
        return labels
    return tuple(f"Descriptor {index + 1}" for index in range(len(archive.dims)))


def _projection_suffix(
    labels: tuple[str, ...],
    *,
    x_axis: int,
    y_axis: int,
) -> str:
    projected = [label for idx, label in enumerate(labels) if idx not in {x_axis, y_axis}]
    if not projected:
        return ""
    return f"  |  max fitness over {', '.join(projected)}"


def _projection_heatmap_image(
    archive: "GridArchive",
    grid: np.ndarray,
    *,
    x_axis: int,
    y_axis: int,
    title: str,
    colorbar_label: str,
    cmap_name: str,
    focus_grid: np.ndarray | None = None,
):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    import wandb

    cmap = matplotlib.colormaps[cmap_name].copy()
    cmap.set_bad(color="#f4f4f4")
    labels = _descriptor_axis_labels(archive)
    fig, ax = plt.subplots(figsize=(8, 5))
    (lo_x, hi_x) = archive.ranges[x_axis]
    (lo_y, hi_y) = archive.ranges[y_axis]
    im = ax.imshow(
        grid.T,
        origin="lower",
        cmap=cmap,
        extent=[lo_x, hi_x, lo_y, hi_y],
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xlabel(labels[x_axis])
    ax.set_ylabel(labels[y_axis])
    ax.set_title(title)
    if focus_grid is not None and not np.all(focus_grid):
        inactive_overlay = np.ma.masked_where(focus_grid.T, np.ones_like(focus_grid.T, dtype=float))
        inactive_cmap = ListedColormap(["#c7c7c7"])
        inactive_cmap.set_bad((0.0, 0.0, 0.0, 0.0))
        ax.imshow(
            inactive_overlay,
            origin="lower",
            cmap=inactive_cmap,
            extent=[lo_x, hi_x, lo_y, hi_y],
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
            alpha=0.95,
        )
        ax.text(
            0.99,
            0.01,
            "grey = outside focus window",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#555555",
        )
    plt.colorbar(im, ax=ax, label=colorbar_label)
    plt.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _fitness_projection_grid(
    archive: "GridArchive",
    *,
    x_axis: int,
    y_axis: int,
) -> np.ndarray:
    grid = np.full(
        (int(archive.dims[x_axis]), int(archive.dims[y_axis])),
        np.nan,
        dtype=np.float64,
    )
    for flat_idx in archive._genotypes:
        cell = np.unravel_index(flat_idx, archive.dims)
        x_idx = int(cell[x_axis])
        y_idx = int(cell[y_axis])
        fitness = float(archive._fitnesses[flat_idx])
        current = grid[x_idx, y_idx]
        if np.isnan(current) or fitness > current:
            grid[x_idx, y_idx] = fitness
    return grid


def _descriptor_projection_grid(
    archive: "GridArchive",
    *,
    x_axis: int,
    y_axis: int,
    descriptor_axis: int,
) -> np.ndarray:
    grid = np.full(
        (int(archive.dims[x_axis]), int(archive.dims[y_axis])),
        np.nan,
        dtype=np.float64,
    )
    for flat_idx in archive._genotypes:
        descriptor = archive._descriptors.get(flat_idx)
        if descriptor is None or len(descriptor) <= descriptor_axis:
            continue
        cell = np.unravel_index(flat_idx, archive.dims)
        x_idx = int(cell[x_axis])
        y_idx = int(cell[y_axis])
        value = float(descriptor[descriptor_axis])
        current = grid[x_idx, y_idx]
        if np.isnan(current) or value > current:
            grid[x_idx, y_idx] = value
    return grid


def _projection_focus_grid(
    archive: "GridArchive",
    *,
    x_axis: int,
    y_axis: int,
    focus_bounds: "FocusWindowBounds | None",
) -> np.ndarray | None:
    if focus_bounds is None:
        return None
    x_lo = int(focus_bounds.lower.get(x_axis, 0))
    x_hi = int(focus_bounds.upper.get(x_axis, int(archive.dims[x_axis]) - 1))
    y_lo = int(focus_bounds.lower.get(y_axis, 0))
    y_hi = int(focus_bounds.upper.get(y_axis, int(archive.dims[y_axis]) - 1))
    focus_grid = np.ones(
        (int(archive.dims[x_axis]), int(archive.dims[y_axis])),
        dtype=bool,
    )
    for x_idx in range(int(archive.dims[x_axis])):
        for y_idx in range(int(archive.dims[y_axis])):
            focus_grid[x_idx, y_idx] = (
                x_lo <= x_idx <= x_hi and y_lo <= y_idx <= y_hi
            )
    return focus_grid


def render_archive_heatmap(
    archive: "GridArchive",
    *,
    focus_bounds: "FocusWindowBounds | None" = None,
):
    """Render a 2D fitness projection of the archive as a wandb.Image.

    For 3D+ archives, this projects by taking the best fitness across all
    remaining descriptor axes at each (x, y) bin.
    """
    if len(archive.dims) < 2:
        raise ValueError("Archive heatmaps require at least 2 descriptor axes")
    grid = _fitness_projection_grid(archive, x_axis=0, y_axis=1)

    labels = _descriptor_axis_labels(archive)
    return _projection_heatmap_image(
        archive,
        grid,
        x_axis=0,
        y_axis=1,
        title=(
            f"{labels[0]} × {labels[1]}  |  "
            f"{archive.size}/{archive.total_cells} cells  "
            f"({archive.coverage:.1%} coverage)"
            f"{_projection_suffix(labels, x_axis=0, y_axis=1)}"
        ),
        colorbar_label="Fitness",
        cmap_name="viridis",
        focus_grid=_projection_focus_grid(
            archive,
            x_axis=0,
            y_axis=1,
            focus_bounds=focus_bounds,
        ),
    )


def render_archive_third_descriptor_heatmap(
    archive: "GridArchive",
    *,
    focus_bounds: "FocusWindowBounds | None" = None,
):
    """Render a 2D projection colored by the best third-descriptor value.

    Uses the first two archive axes as x/y. For each occupied (x, y) bin,
    colors by the maximum realized value of descriptor 3 across all remaining
    dimensions.
    """
    if len(archive.dims) < 3:
        return None

    grid = _descriptor_projection_grid(archive, x_axis=0, y_axis=1, descriptor_axis=2)
    labels = _descriptor_axis_labels(archive)
    suffix = ""
    if len(labels) > 3:
        suffix = f"  |  max over {', '.join(labels[3:])}"
    return _projection_heatmap_image(
        archive,
        grid,
        x_axis=0,
        y_axis=1,
        title=f"{labels[2]} Projection{suffix}",
        colorbar_label=labels[2],
        cmap_name="magma",
        focus_grid=_projection_focus_grid(
            archive,
            x_axis=0,
            y_axis=1,
            focus_bounds=focus_bounds,
        ),
    )


def render_archive_heatmap_time_hits(
    archive: "GridArchive",
    *,
    focus_bounds: "FocusWindowBounds | None" = None,
):
    """Render completion_time × goal_hit_count, maxing over remaining axes."""
    if len(archive.dims) < 3:
        return None
    labels = _descriptor_axis_labels(archive)
    return _projection_heatmap_image(
        archive,
        _fitness_projection_grid(archive, x_axis=2, y_axis=0),
        x_axis=2,
        y_axis=0,
        title=f"{labels[2]} × {labels[0]}{_projection_suffix(labels, x_axis=2, y_axis=0)}",
        colorbar_label="Fitness",
        cmap_name="viridis",
        focus_grid=_projection_focus_grid(
            archive,
            x_axis=2,
            y_axis=0,
            focus_bounds=focus_bounds,
        ),
    )


def render_archive_heatmap_time_falls(
    archive: "GridArchive",
    *,
    focus_bounds: "FocusWindowBounds | None" = None,
):
    """Render completion_time × fall_count, maxing over remaining axes."""
    if len(archive.dims) < 3:
        return None
    labels = _descriptor_axis_labels(archive)
    return _projection_heatmap_image(
        archive,
        _fitness_projection_grid(archive, x_axis=2, y_axis=1),
        x_axis=2,
        y_axis=1,
        title=f"{labels[2]} × {labels[1]}{_projection_suffix(labels, x_axis=2, y_axis=1)}",
        colorbar_label="Fitness",
        cmap_name="viridis",
        focus_grid=_projection_focus_grid(
            archive,
            x_axis=2,
            y_axis=1,
            focus_bounds=focus_bounds,
        ),
    )


def render_archive_named_heatmaps(
    archive: "GridArchive",
    heatmap_names: tuple[str, ...] | list[str] | None,
    *,
    focus_bounds: "FocusWindowBounds | None" = None,
) -> dict[str, object]:
    """Render the requested archive heatmaps keyed for the archive/ namespace."""
    requested = []
    for name in list(heatmap_names or ()):
        normalized = str(name).strip().lower()
        if normalized and normalized not in requested:
            requested.append(normalized)

    payload: dict[str, object] = {}
    for name in requested:
        image = None
        key = ""
        if name == "hits_falls":
            image = render_archive_heatmap(archive, focus_bounds=focus_bounds)
            key = "hits_falls_heatmap"
        elif name == "time_hits":
            image = render_archive_heatmap_time_hits(
                archive,
                focus_bounds=focus_bounds,
            )
            key = "time_hits_heatmap"
        elif name == "time_falls":
            image = render_archive_heatmap_time_falls(
                archive,
                focus_bounds=focus_bounds,
            )
            key = "time_falls_heatmap"
        if image is not None and key:
            payload[key] = image
    return payload


def render_fitness_histogram(archive: "GridArchive"):
    """Render histogram of fitness values in the archive as a wandb.Image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wandb

    if archive.size == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Empty archive", ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        img = wandb.Image(fig)
        plt.close(fig)
        return img

    fitnesses = [archive._fitnesses[i] for i in archive._genotypes]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(fitnesses, bins=30, color="#2196F3", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Fitness (%)")
    ax.set_ylabel("Count")
    ax.set_title(f"Fitness Distribution  |  best={max(fitnesses):.1f}%  mean={np.mean(fitnesses):.1f}%")
    plt.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def init_wandb(cfg, *, run_id: str | None = None, resume_mode: str | None = None) -> bool:
    """Initialize wandb if enabled. Returns True if active.

    Run naming: {terrain}_{body}_{gens}_{pop}_{substeps}_{budget}_{ref}_seed{seed}
    Group: condition (terrain + body + budget surface + reference, without the seed)
    """
    if not cfg.wandb.enabled:
        return False

    import wandb

    metadata = run_metadata(cfg)
    preset_id = metadata["preset_id"]
    group_stem = run_group_label(cfg)
    run_stem = run_name_label(cfg)

    group = cfg.wandb.group or f"{preset_id}_{group_stem}"
    name = cfg.wandb.name or f"{preset_id}_{run_stem}"

    tags = list(cfg.get("tags", []))
    tags.extend(
        [
            str(metadata["body_variant"]),
            f"ea_{metadata['emitter_strategy']}",
            str(metadata["controller_size"]),
            f"deform_{metadata['deformation_condition']}",
            f"gpu_{metadata['gpu_profile']}",
            f"gens_{metadata['n_generations']}",
            f"pop_{metadata['total_population']}",
            f"substeps_{metadata['physics_substeps']}",
            f"train_{metadata['train_max_steps']}",
            f"rerank_{metadata['rerank_max_steps']}",
            f"test_{metadata['test_max_steps']}",
            str(metadata["reference_convergence_label"]),
            f"seed_{metadata['seed']}",
        ]
    )

    init_kwargs = dict(
        project=cfg.wandb.project,
        group=group,
        name=name,
        tags=tags,
        config=dict(cfg),
    )
    if run_id is not None:
        init_kwargs["id"] = str(run_id)
    if resume_mode is not None:
        init_kwargs["resume"] = str(resume_mode)

    wandb.init(**init_kwargs)
    define_wandb_metrics()
    return True
