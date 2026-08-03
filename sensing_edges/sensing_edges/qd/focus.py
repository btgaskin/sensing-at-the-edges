"""Generation-ramped focus window for QD archive.

Implements an automated Interactive QD-style hard window (extending Sfikas et al.,
GECCO 2023) on specified descriptor axes.  Every ``tighten_every_n_gens``
generations the window shrinks by one cell per axis toward the target corner,
subject to a minimum window size and an occupancy gate.

The window only controls which cells participate in improvement scoring and
restart sampling.  The full archive is preserved — frozen cells keep their
elites but stop accepting updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FocusTarget:
    """A single descriptor axis to tighten."""

    name: str
    direction: str  # "max" or "min"

    def __post_init__(self) -> None:
        if self.direction not in ("max", "min"):
            raise ValueError(
                f"FocusTarget direction must be 'max' or 'min', got {self.direction!r}"
            )


@dataclass(frozen=True)
class FocusConfig:
    """Top-level focus window configuration."""

    enabled: bool = False
    tighten_every_n_gens: int = 10
    min_occupied_to_tighten: int = 1
    min_window_size: int = 3
    targets: tuple[FocusTarget, ...] = ()
    fallback_to_all_if_active_empty: bool = True


def focus_config_from_section(section: Any) -> FocusConfig:
    """Parse a config-like focus section into a concrete ``FocusConfig``."""
    if section is None:
        return FocusConfig()

    raw_targets = list(section.get("targets") or [])
    targets = tuple(
        FocusTarget(
            name=str(target.get("name", "")).strip(),
            direction=str(target.get("direction", "")).strip().lower(),
        )
        for target in raw_targets
    )
    return FocusConfig(
        enabled=bool(section.get("enabled", False)),
        tighten_every_n_gens=int(section.get("tighten_every_n_gens", 10)),
        min_occupied_to_tighten=int(section.get("min_occupied_to_tighten", 1)),
        min_window_size=int(section.get("min_window_size", 3)),
        targets=targets,
        fallback_to_all_if_active_empty=bool(
            section.get("fallback_to_all_if_active_empty", True)
        ),
    )


# ---------------------------------------------------------------------------
# Window bounds — mutable training state, checkpointed
# ---------------------------------------------------------------------------

@dataclass
class FocusWindowBounds:
    """Current per-axis window bounds.  Mutated during training."""

    # {axis_index: current inclusive lower bound cell index}
    lower: dict[int, int] = field(default_factory=dict)
    # {axis_index: current inclusive upper bound cell index}
    upper: dict[int, int] = field(default_factory=dict)

    # ---- serialisation for checkpointing ----

    def state_dict(self) -> dict:
        return {
            "lower": dict(self.lower),
            "upper": dict(self.upper),
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> FocusWindowBounds:
        return cls(
            lower={int(k): int(v) for k, v in d["lower"].items()},
            upper={int(k): int(v) for k, v in d["upper"].items()},
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_focus_config(
    focus_cfg: FocusConfig,
    descriptor_names: tuple[str, ...],
) -> dict[str, int]:
    """Validate focus targets against the archive's descriptor names.

    Returns a mapping ``{target_name: axis_index}`` for every target.
    Raises ``ValueError`` if a target name is not present in
    *descriptor_names*.
    """
    if not focus_cfg.enabled:
        return {}
    if not focus_cfg.targets:
        raise ValueError("qd.focus.enabled=true requires at least one focus target")
    name_to_idx: dict[str, int] = {n: i for i, n in enumerate(descriptor_names)}
    target_axes: dict[str, int] = {}
    for t in focus_cfg.targets:
        if t.name not in name_to_idx:
            raise ValueError(
                f"Focus target {t.name!r} is not in the archive descriptor "
                f"set {list(descriptor_names)}"
            )
        target_axes[t.name] = name_to_idx[t.name]
    return target_axes


# ---------------------------------------------------------------------------
# Initial bounds
# ---------------------------------------------------------------------------

def initial_window_bounds(
    focus_cfg: FocusConfig,
    target_axes: dict[str, int],
    archive_dims: tuple[int, ...],
) -> FocusWindowBounds:
    """Return fully-open bounds for each target axis."""
    lower: dict[int, int] = {}
    upper: dict[int, int] = {}
    for t in focus_cfg.targets:
        axis = target_axes[t.name]
        lower[axis] = 0
        upper[axis] = int(archive_dims[axis]) - 1
    return FocusWindowBounds(lower=lower, upper=upper)


# ---------------------------------------------------------------------------
# Tightening
# ---------------------------------------------------------------------------

def _final_bounds_for_target(
    target: FocusTarget,
    axis: int,
    dim: int,
    min_window_size: int,
) -> tuple[int, int]:
    """Return the tightest (lower, upper) that respects *min_window_size*."""
    if target.direction == "max":
        # push lower bound up; upper stays at dim-1
        final_lower = max(dim - min_window_size, 0)
        return final_lower, dim - 1
    else:
        # push upper bound down; lower stays at 0
        final_upper = min(min_window_size - 1, dim - 1)
        return 0, final_upper


def maybe_tighten(
    bounds: FocusWindowBounds,
    focus_cfg: FocusConfig,
    target_axes: dict[str, int],
    archive_dims: tuple[int, ...],
    archive_occupied_flat_indices: set[int],
) -> FocusWindowBounds:
    """Propose shrinking by 1 cell per target axis.

    Only applies the tighter bounds when **both**:

    1. Every axis still has at least ``min_window_size`` cells after shrinking.
    2. The proposed region contains more than ``min_occupied_to_tighten``
       occupied cells in the archive.

    Returns the (possibly unchanged) bounds.
    """
    proposed_lower = dict(bounds.lower)
    proposed_upper = dict(bounds.upper)
    any_change = False

    for t in focus_cfg.targets:
        axis = target_axes[t.name]
        dim = int(archive_dims[axis])
        lo_final, hi_final = _final_bounds_for_target(
            t, axis, dim, focus_cfg.min_window_size,
        )

        if t.direction == "max":
            new_lo = min(bounds.lower[axis] + 1, lo_final)
            if new_lo != bounds.lower[axis]:
                proposed_lower[axis] = new_lo
                any_change = True
        else:
            new_hi = max(bounds.upper[axis] - 1, hi_final)
            if new_hi != bounds.upper[axis]:
                proposed_upper[axis] = new_hi
                any_change = True

    if not any_change:
        return bounds  # already at minimum window

    # Check occupancy in proposed region
    proposed_bounds = FocusWindowBounds(lower=proposed_lower, upper=proposed_upper)
    proposed_mask = build_mask_from_bounds(
        proposed_bounds, target_axes, archive_dims,
    )
    occupied_in_proposed = sum(
        1 for idx in archive_occupied_flat_indices if proposed_mask[idx]
    )

    if occupied_in_proposed > focus_cfg.min_occupied_to_tighten:
        return proposed_bounds

    return bounds  # hold — not enough occupancy yet


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def build_mask_from_bounds(
    bounds: FocusWindowBounds,
    target_axes: dict[str, int],
    archive_dims: tuple[int, ...],
) -> np.ndarray:
    """Return a flat boolean array of shape ``(prod(dims),)`` — ``True`` = active.

    Cells are active when every target axis' cell index falls within the
    inclusive ``[lower, upper]`` range.  Non-target axes are unconstrained.
    """
    ndim = len(archive_dims)
    total = int(np.prod(archive_dims))

    # Build per-axis boolean arrays, then broadcast via outer product
    axis_masks: list[np.ndarray] = []
    for d in range(ndim):
        dim = int(archive_dims[d])
        if d in bounds.lower or d in bounds.upper:
            lo = bounds.lower.get(d, 0)
            hi = bounds.upper.get(d, dim - 1)
            axis_masks.append(
                np.array([(lo <= c <= hi) for c in range(dim)], dtype=bool)
            )
        else:
            axis_masks.append(np.ones(dim, dtype=bool))

    # N-dimensional outer product via meshgrid, then flatten
    grids = np.meshgrid(*axis_masks, indexing="ij")
    combined = grids[0]
    for g in grids[1:]:
        combined = combined & g
    return combined.ravel()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def focus_metrics(
    bounds: FocusWindowBounds,
    target_axes: dict[str, int],
    archive_dims: tuple[int, ...],
    archive_occupied_flat_indices: set[int],
) -> dict[str, float]:
    """Return ``qd_focus_*`` metrics for generation reporting."""
    mask = build_mask_from_bounds(bounds, target_axes, archive_dims)
    active_count = int(np.sum(mask))
    total = int(np.prod(archive_dims))
    occupied_in_active = sum(
        1 for idx in archive_occupied_flat_indices if mask[idx]
    )

    metrics: dict[str, float] = {
        "qd_focus_active_cells": float(active_count),
        "qd_focus_active_fraction": float(active_count) / max(total, 1),
        "qd_focus_occupied_in_active": float(occupied_in_active),
    }

    # Per-axis bounds
    for name, axis in target_axes.items():
        lo = bounds.lower.get(axis, 0)
        hi = bounds.upper.get(axis, int(archive_dims[axis]) - 1)
        metrics[f"qd_focus_{name}_lower"] = float(lo)
        metrics[f"qd_focus_{name}_upper"] = float(hi)
        metrics[f"qd_focus_{name}_window_size"] = float(hi - lo + 1)

    return metrics
