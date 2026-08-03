"""Tests for the QD focus window module."""

from __future__ import annotations

import numpy as np
import pytest

from sensing_edges.qd.focus import (
    FocusConfig,
    FocusTarget,
    FocusWindowBounds,
    build_mask_from_bounds,
    focus_metrics,
    initial_window_bounds,
    maybe_tighten,
    validate_focus_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cliff_config(
    min_window_size: int = 3,
    min_occupied: int = 1,
) -> tuple[FocusConfig, dict[str, int], tuple[int, ...]]:
    """Return (config, target_axes, dims) for a 7×7×20 cliff archive."""
    cfg = FocusConfig(
        enabled=True,
        tighten_every_n_gens=10,
        min_occupied_to_tighten=min_occupied,
        min_window_size=min_window_size,
        targets=(
            FocusTarget(name="goal_hit_count", direction="max"),
            FocusTarget(name="fall_count", direction="min"),
        ),
    )
    target_axes = {"goal_hit_count": 0, "fall_count": 1}
    dims = (7, 7, 20)
    return cfg, target_axes, dims


def _occupied_everywhere(dims: tuple[int, ...]) -> set[int]:
    """Return all flat indices as occupied."""
    return set(range(int(np.prod(dims))))


def _occupied_at_cells(
    dims: tuple[int, ...],
    cells: list[tuple[int, ...]],
) -> set[int]:
    """Return flat indices for specific multi-dim cell coords."""
    occupied = set()
    for cell in cells:
        flat = int(np.ravel_multi_index(cell, dims))
        occupied.add(flat)
    return occupied


# ---------------------------------------------------------------------------
# validate_focus_config
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_descriptors(self):
        cfg = FocusConfig(
            enabled=True,
            targets=(FocusTarget("goal_hit_count", "max"),),
        )
        axes = validate_focus_config(
            cfg, ("goal_hit_count", "fall_count", "completion_time"),
        )
        assert axes == {"goal_hit_count": 0}

    def test_missing_descriptor_raises(self):
        cfg = FocusConfig(
            enabled=True,
            targets=(FocusTarget("nonexistent", "max"),),
        )
        with pytest.raises(ValueError, match="nonexistent"):
            validate_focus_config(cfg, ("goal_hit_count", "fall_count"))

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction"):
            FocusTarget("x", "up")


# ---------------------------------------------------------------------------
# initial_window_bounds
# ---------------------------------------------------------------------------

class TestInitialBounds:
    def test_covers_full_archive(self):
        cfg, target_axes, dims = _cliff_config()
        bounds = initial_window_bounds(cfg, target_axes, dims)

        assert bounds.lower[0] == 0  # hits
        assert bounds.upper[0] == 6
        assert bounds.lower[1] == 0  # falls
        assert bounds.upper[1] == 6

    def test_initial_mask_all_true(self):
        cfg, target_axes, dims = _cliff_config()
        bounds = initial_window_bounds(cfg, target_axes, dims)
        mask = build_mask_from_bounds(bounds, target_axes, dims)
        assert mask.shape == (7 * 7 * 20,)
        assert np.all(mask)


# ---------------------------------------------------------------------------
# maybe_tighten
# ---------------------------------------------------------------------------

class TestMaybeTighten:
    def test_shrinks_by_one_per_axis(self):
        cfg, target_axes, dims = _cliff_config()
        bounds = initial_window_bounds(cfg, target_axes, dims)
        occupied = _occupied_everywhere(dims)

        new_bounds = maybe_tighten(bounds, cfg, target_axes, dims, occupied)

        # hits (dir=max): lower 0 → 1
        assert new_bounds.lower[0] == 1
        assert new_bounds.upper[0] == 6
        # falls (dir=min): upper 6 → 5
        assert new_bounds.lower[1] == 0
        assert new_bounds.upper[1] == 5

    def test_multiple_tighten_steps(self):
        cfg, target_axes, dims = _cliff_config()
        bounds = initial_window_bounds(cfg, target_axes, dims)
        occupied = _occupied_everywhere(dims)

        for _ in range(4):
            bounds = maybe_tighten(bounds, cfg, target_axes, dims, occupied)

        # hits: lower 0 → 4 (min_window_size=3 → final lower=4, dim=7)
        assert bounds.lower[0] == 4
        # falls: upper 6 → 2 (min_window_size=3 → final upper=2)
        assert bounds.upper[1] == 2

    def test_stops_at_min_window_size(self):
        cfg, target_axes, dims = _cliff_config(min_window_size=3)
        bounds = initial_window_bounds(cfg, target_axes, dims)
        occupied = _occupied_everywhere(dims)

        for _ in range(10):
            bounds = maybe_tighten(bounds, cfg, target_axes, dims, occupied)

        # hits: lower capped at dim - min_window_size = 7-3 = 4
        assert bounds.lower[0] == 4
        assert bounds.upper[0] == 6
        # falls: upper capped at min_window_size - 1 = 2
        assert bounds.lower[1] == 0
        assert bounds.upper[1] == 2

    def test_holds_when_insufficient_occupancy(self):
        cfg, target_axes, dims = _cliff_config(min_occupied=1)
        bounds = initial_window_bounds(cfg, target_axes, dims)

        # Only occupy cells at (0, 6, 0) — far from the target corner
        occupied = _occupied_at_cells(dims, [(0, 6, 0)])

        # Tighten step 1: proposed region [1-6]×[0-5]×[0-19]
        # Cell (0, 6, 0) is NOT in proposed region → 0 occupied → hold
        new_bounds = maybe_tighten(bounds, cfg, target_axes, dims, occupied)
        assert new_bounds.lower[0] == 0  # unchanged
        assert new_bounds.upper[1] == 6  # unchanged

    def test_tightens_when_occupancy_sufficient(self):
        cfg, target_axes, dims = _cliff_config(min_occupied=1)
        bounds = initial_window_bounds(cfg, target_axes, dims)

        # Occupy a cell that will be in the proposed region
        occupied = _occupied_at_cells(dims, [(3, 2, 10), (5, 0, 5)])

        new_bounds = maybe_tighten(bounds, cfg, target_axes, dims, occupied)
        assert new_bounds.lower[0] == 1  # tightened
        assert new_bounds.upper[1] == 5  # tightened

    def test_already_at_minimum_returns_same(self):
        cfg, target_axes, dims = _cliff_config(min_window_size=3)
        bounds = FocusWindowBounds(lower={0: 4, 1: 0}, upper={0: 6, 1: 2})
        occupied = _occupied_everywhere(dims)

        new_bounds = maybe_tighten(bounds, cfg, target_axes, dims, occupied)
        assert new_bounds.lower[0] == 4
        assert new_bounds.upper[1] == 2


# ---------------------------------------------------------------------------
# build_mask_from_bounds
# ---------------------------------------------------------------------------

class TestBuildMask:
    def test_full_bounds_all_active(self):
        cfg, target_axes, dims = _cliff_config()
        bounds = initial_window_bounds(cfg, target_axes, dims)
        mask = build_mask_from_bounds(bounds, target_axes, dims)
        assert np.all(mask)

    def test_tight_bounds_correct_cells(self):
        target_axes = {"goal_hit_count": 0, "fall_count": 1}
        dims = (7, 7, 20)
        bounds = FocusWindowBounds(lower={0: 5, 1: 0}, upper={0: 6, 1: 1})
        mask = build_mask_from_bounds(bounds, target_axes, dims)

        # Check specific cells
        # (5, 0, 0) should be active
        assert mask[np.ravel_multi_index((5, 0, 0), dims)]
        # (6, 1, 19) should be active
        assert mask[np.ravel_multi_index((6, 1, 19), dims)]
        # (4, 0, 0) should be inactive (hits=4 < lower=5)
        assert not mask[np.ravel_multi_index((4, 0, 0), dims)]
        # (5, 2, 0) should be inactive (falls=2 > upper=1)
        assert not mask[np.ravel_multi_index((5, 2, 0), dims)]

        # Total active: 2 hits × 2 falls × 20 time = 80
        assert np.sum(mask) == 80

    def test_unconstrained_axis_always_open(self):
        target_axes = {"goal_hit_count": 0}
        dims = (7, 7, 20)
        bounds = FocusWindowBounds(lower={0: 5}, upper={0: 6})
        mask = build_mask_from_bounds(bounds, target_axes, dims)

        # All falls values should be active for hits=5
        for f in range(7):
            for t in range(20):
                assert mask[np.ravel_multi_index((5, f, t), dims)]

        # hits=4 should be inactive
        assert not mask[np.ravel_multi_index((4, 0, 0), dims)]


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_state_dict_roundtrip(self):
        bounds = FocusWindowBounds(lower={0: 3, 1: 0}, upper={0: 6, 1: 4})
        d = bounds.state_dict()
        restored = FocusWindowBounds.from_state_dict(d)
        assert restored.lower == bounds.lower
        assert restored.upper == bounds.upper


# ---------------------------------------------------------------------------
# focus_metrics
# ---------------------------------------------------------------------------

class TestFocusMetrics:
    def test_full_window_metrics(self):
        cfg, target_axes, dims = _cliff_config()
        bounds = initial_window_bounds(cfg, target_axes, dims)
        occupied = _occupied_at_cells(dims, [(5, 0, 10)])

        m = focus_metrics(bounds, target_axes, dims, occupied)
        assert m["qd_focus_active_cells"] == 7 * 7 * 20
        assert m["qd_focus_active_fraction"] == pytest.approx(1.0)
        assert m["qd_focus_occupied_in_active"] == 1.0
        assert m["qd_focus_goal_hit_count_window_size"] == 7.0
        assert m["qd_focus_fall_count_window_size"] == 7.0
