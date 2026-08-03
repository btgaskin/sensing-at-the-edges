"""Config-driven QD descriptor registry and archive layout resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
from omegaconf import DictConfig

from .terrain import make_robot_body

if TYPE_CHECKING:
    from .fitness import CandidateBatchResult, CandidateResult


def default_qd_descriptor_names(cfg: DictConfig) -> tuple[str, str]:
    """Return the default descriptor pair for the current body mode."""
    body_mode = str(cfg.body.mode).strip().lower()
    if body_mode == "grown":
        return ("alive_cell_count", "time_to_edge")
    return ("activation_cost", "peak_progress_rate")


def _fixed_dim(value: int) -> Callable[[DictConfig], int]:
    return lambda _cfg: int(value)


def _fixed_range(lo: float, hi: float) -> Callable[[DictConfig], tuple[float, float]]:
    return lambda _cfg: (float(lo), float(hi))


def _episode_count_dim(cfg: DictConfig) -> int:
    return max(int(cfg.train.n_episodes) + 1, 2)


def _episode_count_range(cfg: DictConfig) -> tuple[float, float]:
    return (0.0, float(max(int(cfg.train.n_episodes), 1)))


def _alive_cell_count_range(cfg: DictConfig) -> tuple[float, float]:
    body_mode = str(cfg.body.mode).strip().lower()
    if body_mode == "grown":
        max_cells = int(cfg.body.growth.max_robot_h) * int(cfg.body.growth.max_robot_w)
    else:
        robot = make_robot_body(
            int(cfg.terrain.robot_w),
            int(cfg.terrain.robot_h),
            shape=str(cfg.body.get("shape", "rectangle")),
        )
        max_cells = int(np.count_nonzero(robot))
    return (0.0, float(max(max_cells, 1)))


@dataclass(frozen=True)
class QDDescriptorSpec:
    """Descriptor metadata plus default archive layout helpers."""

    name: str
    label: str
    metric_attr: str
    default_dim: Callable[[DictConfig], int]
    default_range: Callable[[DictConfig], tuple[float, float]]


@dataclass(frozen=True)
class ResolvedQDArchive:
    """Resolved descriptor set plus archive layout metadata."""

    descriptor_specs: tuple[QDDescriptorSpec, ...]
    dims: tuple[int, ...]
    ranges: tuple[tuple[float, float], ...]

    @property
    def descriptor_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.descriptor_specs)

    @property
    def descriptor_labels(self) -> tuple[str, ...]:
        return tuple(spec.label for spec in self.descriptor_specs)


QD_DESCRIPTOR_REGISTRY: dict[str, QDDescriptorSpec] = {
    "activation_cost": QDDescriptorSpec(
        name="activation_cost",
        label="Activation Cost",
        metric_attr="activation_cost",
        default_dim=_fixed_dim(20),
        default_range=_fixed_range(0.0, 2.0),
    ),
    "movement_speed": QDDescriptorSpec(
        name="movement_speed",
        label="Movement Speed",
        metric_attr="movement_speed",
        default_dim=_fixed_dim(20),
        default_range=_fixed_range(0.0, 1.0),
    ),
    "peak_progress_rate": QDDescriptorSpec(
        name="peak_progress_rate",
        label="Peak Progress Rate",
        metric_attr="peak_progress_rate",
        default_dim=_fixed_dim(20),
        default_range=_fixed_range(0.0, 2.0),
    ),
    "correct_progress": QDDescriptorSpec(
        name="correct_progress",
        label="Correct Progress",
        metric_attr="correct_progress",
        default_dim=_fixed_dim(20),
        default_range=_fixed_range(0.0, 1.0),
    ),
    "correct_progress_top2_mean": QDDescriptorSpec(
        name="correct_progress_top2_mean",
        label="Correct Progress Top-2 Mean",
        metric_attr="correct_progress_top2_mean",
        default_dim=_fixed_dim(20),
        default_range=_fixed_range(0.0, 1.0),
    ),
    "incorrect_progress": QDDescriptorSpec(
        name="incorrect_progress",
        label="Incorrect Progress",
        metric_attr="incorrect_progress",
        default_dim=_fixed_dim(20),
        default_range=_fixed_range(0.0, 1.0),
    ),
    "incorrect_progress_top2_mean": QDDescriptorSpec(
        name="incorrect_progress_top2_mean",
        label="Incorrect Progress Top-2 Mean",
        metric_attr="incorrect_progress_top2_mean",
        default_dim=_fixed_dim(20),
        default_range=_fixed_range(0.0, 1.0),
    ),
    "mirrored_roundtrip_top2_mean": QDDescriptorSpec(
        name="mirrored_roundtrip_top2_mean",
        label="Mirrored Roundtrip Top-2 Mean",
        metric_attr="mirrored_roundtrip_top2_mean",
        default_dim=_fixed_dim(20),
        default_range=_fixed_range(0.0, 1.0),
    ),
    "goal_hit_count": QDDescriptorSpec(
        name="goal_hit_count",
        label="Goal Hit Count",
        metric_attr="goal_hit_count",
        default_dim=_episode_count_dim,
        default_range=_episode_count_range,
    ),
    "mirrored_pair_count": QDDescriptorSpec(
        name="mirrored_pair_count",
        label="Mirrored Pair Count",
        metric_attr="mirrored_pair_count",
        default_dim=_episode_count_dim,
        default_range=_episode_count_range,
    ),
    "mirrored_best_hit_count": QDDescriptorSpec(
        name="mirrored_best_hit_count",
        label="Mirrored Best Hit Count",
        metric_attr="mirrored_best_hit_count",
        default_dim=_episode_count_dim,
        default_range=_episode_count_range,
    ),
    "mirrored_total_hit_count": QDDescriptorSpec(
        name="mirrored_total_hit_count",
        label="Mirrored Total Hit Count",
        metric_attr="mirrored_total_hit_count",
        default_dim=_episode_count_dim,
        default_range=_episode_count_range,
    ),
    "turn_count": QDDescriptorSpec(
        name="turn_count",
        label="Turn Count",
        metric_attr="turn_count",
        default_dim=_fixed_dim(7),
        default_range=_fixed_range(0.0, 6.0),
    ),
    "stable_turn_count": QDDescriptorSpec(
        name="stable_turn_count",
        label="Stable Turn Count",
        metric_attr="stable_turn_count",
        default_dim=_fixed_dim(24),
        default_range=_fixed_range(0.0, 6.0),
    ),
    "time_to_edge": QDDescriptorSpec(
        name="time_to_edge",
        label="Time To Edge",
        metric_attr="time_to_edge",
        default_dim=_fixed_dim(20),
        default_range=_fixed_range(0.0, 1.0),
    ),
    "completion_time": QDDescriptorSpec(
        name="completion_time",
        label="Completion Time",
        metric_attr="time_to_edge",
        default_dim=_fixed_dim(20),
        default_range=_fixed_range(0.0, 1.0),
    ),
    "gait_spectral_purity": QDDescriptorSpec(
        name="gait_spectral_purity",
        label="Gait Spectral Purity",
        metric_attr="gait_spectral_purity",
        default_dim=_fixed_dim(20),
        default_range=_fixed_range(0.0, 1.0),
    ),
    "fall_count": QDDescriptorSpec(
        name="fall_count",
        label="Fall Count",
        metric_attr="fall_count",
        default_dim=_episode_count_dim,
        default_range=_episode_count_range,
    ),
    "cliff_vicinity": QDDescriptorSpec(
        name="cliff_vicinity",
        label="Cliff Vicinity",
        metric_attr="cliff_vicinity",
        default_dim=_fixed_dim(20),
        default_range=_fixed_range(0.0, 1.0),
    ),
    "alive_cell_count": QDDescriptorSpec(
        name="alive_cell_count",
        label="Alive Cell Count",
        metric_attr="alive_cell_count",
        default_dim=_fixed_dim(20),
        default_range=_alive_cell_count_range,
    ),
    "h_activation": QDDescriptorSpec(
        name="h_activation",
        label="H Activation",
        metric_attr="h_activation",
        default_dim=_fixed_dim(20),
        default_range=_fixed_range(0.0, 1.0),
    ),
    "v_activation": QDDescriptorSpec(
        name="v_activation",
        label="V Activation",
        metric_attr="v_activation",
        default_dim=_fixed_dim(20),
        default_range=_fixed_range(0.0, 1.0),
    ),
}


def resolve_qd_descriptor_specs(cfg: DictConfig) -> tuple[QDDescriptorSpec, ...]:
    """Resolve and validate the selected archive descriptor set."""

    raw_names = cfg.qd.get("descriptors", {}).get("names")
    if raw_names is None:
        names = default_qd_descriptor_names(cfg)
    else:
        names = tuple(str(name).strip() for name in raw_names)

    if len(names) < 2:
        raise ValueError("qd.descriptors.names must contain at least 2 descriptor names")
    if len(set(names)) != len(names):
        raise ValueError("qd.descriptors.names must not contain duplicates")

    unknown = [name for name in names if name not in QD_DESCRIPTOR_REGISTRY]
    if unknown:
        raise ValueError(
            "Unknown QD descriptor name(s): " + ", ".join(sorted(unknown))
        )

    return tuple(QD_DESCRIPTOR_REGISTRY[name] for name in names)


def resolve_qd_archive(cfg: DictConfig) -> ResolvedQDArchive:
    """Resolve descriptor set plus archive dims/ranges from config defaults."""

    specs = resolve_qd_descriptor_specs(cfg)
    raw_dims = cfg.qd.archive.get("dims")
    raw_ranges = cfg.qd.archive.get("ranges")

    if raw_dims is None or len(raw_dims) != len(specs):
        dims = tuple(int(spec.default_dim(cfg)) for spec in specs)
    else:
        dims = tuple(int(value) for value in raw_dims)
    if any(dim <= 0 for dim in dims):
        raise ValueError("qd.archive.dims must contain only positive integers")

    if raw_ranges is None or len(raw_ranges) != len(specs):
        ranges = tuple(tuple(spec.default_range(cfg)) for spec in specs)
    else:
        ranges = tuple((float(bounds[0]), float(bounds[1])) for bounds in raw_ranges)
    for lo, hi in ranges:
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError("qd.archive.ranges must contain finite [lo, hi] pairs with hi > lo")

    return ResolvedQDArchive(
        descriptor_specs=specs,
        dims=tuple(int(dim) for dim in dims),
        ranges=tuple((float(lo), float(hi)) for lo, hi in ranges),
    )


def build_descriptor_matrix(
    batch_result: "CandidateBatchResult",
    descriptor_specs: tuple[QDDescriptorSpec, ...],
) -> np.ndarray:
    """Stack the selected candidate metrics into an archive descriptor matrix."""

    columns = [
        np.asarray(getattr(batch_result, spec.metric_attr), dtype=np.float32)
        for spec in descriptor_specs
    ]
    return np.stack(columns, axis=1)


def build_descriptor_vector(
    result: "CandidateResult",
    descriptor_specs: tuple[QDDescriptorSpec, ...],
) -> np.ndarray:
    """Extract the selected descriptor vector from a single candidate result."""

    return np.asarray(
        [float(getattr(result, spec.metric_attr)) for spec in descriptor_specs],
        dtype=np.float32,
    )
