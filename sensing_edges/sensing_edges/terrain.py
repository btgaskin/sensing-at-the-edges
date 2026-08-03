"""Programmatic terrain generator for sensing_edges terrain families.

Grid convention: y-up (row 0 = bottom), matching jax_compose_grid.

The module supports the legacy cliff/gap generators plus the newer bilateral
goal-vs-cliff task family used by the main ablations:

- Legacy: base plateau, variable hazard zone, trailing dead space.
- Structured gap: spawn plateau `x=0..9`, phase-1 cliff randomization in
  `x=10..19`, and phase-2/final gap layouts extending that active span to
  `x=24`.
- Bilateral goal-cliff: centered 10-cell start plateau, symmetric outward
  runway on both sides, one raised goal block on one side, and a cliff on the
  opposite side.

The walking surface is a single fixed row at ``y=platform_y``. There is no
row-0 collider floor; falling below the experiment's death plane is episode
logic, not terrain geometry.
"""

from __future__ import annotations

import itertools

import numpy as np

from jax_evogym.constants import EMPTY, FIXED, CONTRACTILE


# Per-width maximum gap counts (hardcoded invariants).
_MAX_GAPS_PER_WIDTH = {1: 3, 2: 2, 3: 1}
_MIN_SOLID_RUN = 2
_HAZARD_MODE_LEGACY = "legacy"
_HAZARD_MODE_ZONE_SINGLE_GAP = "zone_single_gap"
_LAYOUT_SAMPLING_UNIFORM_LEGAL = "uniform_legal_layouts"
_TASK_MODE_LEGACY_CLIFF = "legacy_cliff"
_TASK_MODE_BILATERAL_GOAL_CLIFF = "bilateral_goal_cliff"
_GOAL_SIDE_LEFT = "left"
_GOAL_SIDE_RIGHT = "right"
_TERRAIN_OBJECT_ID = 1
_GOAL_OBJECT_ID = 2


def _normalize_body_shape(shape: str) -> str:
    normalized = str(shape).strip().lower()
    if normalized in {"rectangle", "rect"}:
        return "rectangle"
    if normalized == "comb":
        return "comb"
    if normalized == "comb_7x3":
        return "comb_7x3"
    raise ValueError(f"Unsupported body.shape={shape!r}")


def _resolve_min_solid_run(min_cells: int) -> int:
    """Clamp solid terrain run lengths to the global 2-cell invariant."""
    return max(int(min_cells), _MIN_SOLID_RUN)


def build_cliff_terrain(
    grid_h: int,
    grid_w: int,
    cliff_x: int,
    *,
    platform_y: int | None = None,
    plateau_height: int | None = None,
) -> np.ndarray:
    """Build a single-row platform terrain with a cliff edge.

    Args:
        grid_h: Grid height.
        grid_w: Grid width.
        cliff_x: Last FIXED column of the platform. x=cliff_x+1 onward is empty.
        platform_y: Row containing the walking surface. Defaults to 3.
        plateau_height: Backward-compatible alias for the old top-surface height.
            If provided, it maps directly onto ``platform_y``.

    Returns:
        (grid_h, grid_w) int32 array of cell types.
    """
    platform_y = _resolve_platform_y(platform_y=platform_y, plateau_height=plateau_height)
    grid = np.full((grid_h, grid_w), EMPTY, dtype=np.int32)
    grid[platform_y, : cliff_x + 1] = FIXED
    return grid


def build_cliff_gap_terrain(
    grid_h: int,
    grid_w: int,
    cliff_x: int,
    gaps: list[tuple[int, int]],
    *,
    platform_y: int | None = None,
    plateau_height: int | None = None,
) -> np.ndarray:
    """Build a cliff terrain with gaps cut into the platform row.

    Args:
        grid_h: Grid height.
        grid_w: Grid width.
        cliff_x: Last FIXED column of the platform.
        gaps: List of (start_col, width) pairs defining gap positions.
        platform_y: Row containing the walking surface. Defaults to 3.
        plateau_height: Backward-compatible alias for the old top-surface height.

    Returns:
        (grid_h, grid_w) int32 array of cell types.
    """
    platform_y = _resolve_platform_y(platform_y=platform_y, plateau_height=plateau_height)
    grid = build_cliff_terrain(grid_h, grid_w, cliff_x, platform_y=platform_y)
    for start_col, width in gaps:
        grid[platform_y, start_col : start_col + width] = EMPTY
    return grid


def build_symmetric_double_cliff_terrain(
    grid_h: int,
    grid_w: int,
    symmetric_distance: int,
    *,
    plateau_width: int = 10,
    platform_y: int | None = None,
    plateau_height: int | None = None,
) -> tuple[np.ndarray, int, int]:
    """Build a centered platform with equal cliff drop-offs on both sides.

    The middle plateau of width ``plateau_width`` is centered in the grid. The
    sampled ``symmetric_distance`` extends that plateau equally to the left and
    right, yielding a deterministic sweep from narrow to wide safe corridors.

    Returns:
        ``(grid, platform_start_x, platform_end_x)`` where the terrain row is
        FIXED on ``platform_start_x..platform_end_x`` inclusive and empty
        elsewhere.
    """
    platform_y = _resolve_platform_y(platform_y=platform_y, plateau_height=plateau_height)
    plateau_width = int(plateau_width)
    symmetric_distance = int(symmetric_distance)
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("grid_h and grid_w must be positive.")
    if plateau_width <= 0:
        raise ValueError("plateau_width must be positive.")
    if plateau_width > int(grid_w):
        raise ValueError("plateau_width cannot exceed grid_w.")
    if symmetric_distance < 0:
        raise ValueError("symmetric_distance must be non-negative.")

    center_start = (int(grid_w) - plateau_width) // 2
    center_end = center_start + plateau_width - 1
    platform_start = center_start - symmetric_distance
    platform_end = center_end + symmetric_distance
    if platform_start < 0 or platform_end >= int(grid_w):
        raise ValueError(
            "symmetric_distance does not fit within the grid: "
            f"grid_w={grid_w}, plateau_width={plateau_width}, symmetric_distance={symmetric_distance}"
        )

    grid = np.full((grid_h, grid_w), EMPTY, dtype=np.int32)
    grid[platform_y, platform_start : platform_end + 1] = FIXED
    return grid, int(platform_start), int(platform_end)


def build_asymmetric_double_cliff_terrain(
    grid_h: int,
    grid_w: int,
    *,
    left_distance: int,
    right_distance: int,
    plateau_width: int = 10,
    platform_y: int | None = None,
    plateau_height: int | None = None,
) -> tuple[np.ndarray, int, int, int, int]:
    """Build a centered platform with independently scaled left/right runways."""
    platform_y = _resolve_platform_y(platform_y=platform_y, plateau_height=plateau_height)
    plateau_width = int(plateau_width)
    left_distance = int(left_distance)
    right_distance = int(right_distance)
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("grid_h and grid_w must be positive.")
    if plateau_width <= 0:
        raise ValueError("plateau_width must be positive.")
    if plateau_width > int(grid_w):
        raise ValueError("plateau_width cannot exceed grid_w.")
    if left_distance < 0 or right_distance < 0:
        raise ValueError("left_distance and right_distance must be non-negative.")

    center_start = (int(grid_w) - plateau_width) // 2
    center_end = center_start + plateau_width - 1
    platform_start = center_start - left_distance
    platform_end = center_end + right_distance
    if platform_start < 0 or platform_end >= int(grid_w):
        raise ValueError(
            "asymmetric bilateral distances do not fit within the grid: "
            f"grid_w={grid_w}, plateau_width={plateau_width}, "
            f"left_distance={left_distance}, right_distance={right_distance}"
        )

    grid = np.full((grid_h, grid_w), EMPTY, dtype=np.int32)
    grid[platform_y, platform_start : platform_end + 1] = FIXED
    return (
        grid,
        int(platform_start),
        int(platform_end),
        int(center_start),
        int(center_end),
    )


def build_bilateral_goal_cliff_terrain(
    grid_h: int,
    grid_w: int,
    *,
    symmetric_distance: int | None = None,
    goal_distance: int | None = None,
    cliff_distance: int | None = None,
    goal_side: str,
    plateau_width: int = 10,
    platform_y: int | None = None,
    plateau_height: int | None = None,
    correct_side_distance_scale: float = 1.0,
    incorrect_side_distance_scale: float = 1.0,
    cliff_front_gap_width: int = 0,
    cliff_post_gap_runway: int = 0,
    cliff_probe_gap_width: int = 0,
    cliff_probe_ledge_width: int = 0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build a centered bilateral task with one raised goal block."""
    platform_y = _resolve_platform_y(platform_y=platform_y, plateau_height=plateau_height)
    normalized_goal_side = str(goal_side).strip().lower()
    if normalized_goal_side not in {_GOAL_SIDE_LEFT, _GOAL_SIDE_RIGHT}:
        raise ValueError(f"goal_side must be 'left' or 'right', got {goal_side!r}")
    if int(platform_y) + 1 >= int(grid_h):
        raise ValueError("goal block row must fit within the terrain grid.")
    cliff_front_gap_width = int(cliff_front_gap_width)
    cliff_post_gap_runway = int(cliff_post_gap_runway)
    cliff_probe_gap_width = int(cliff_probe_gap_width)
    cliff_probe_ledge_width = int(cliff_probe_ledge_width)
    if (
        cliff_front_gap_width < 0
        or cliff_post_gap_runway < 0
        or cliff_probe_gap_width < 0
        or cliff_probe_ledge_width < 0
    ):
        raise ValueError(
            "cliff_front_gap_width, cliff_post_gap_runway, cliff_probe_gap_width, "
            "and cliff_probe_ledge_width must be non-negative."
        )
    if cliff_front_gap_width == 0 and cliff_post_gap_runway > 0:
        raise ValueError("cliff_post_gap_runway requires cliff_front_gap_width > 0.")
    if cliff_probe_ledge_width > 0 and int(platform_y) <= 0:
        raise ValueError("cliff_probe_ledge_width requires platform_y > 0.")
    if cliff_front_gap_width > 0 and (cliff_probe_gap_width > 0 or cliff_probe_ledge_width > 0):
        raise ValueError("Plateau-level front gaps cannot be combined with cliff_probe gap/ledge features.")
    correct_side_distance_scale = float(correct_side_distance_scale)
    incorrect_side_distance_scale = float(incorrect_side_distance_scale)
    if correct_side_distance_scale <= 0.0 or incorrect_side_distance_scale <= 0.0:
        raise ValueError("correct/incorrect side distance scales must be positive.")

    if goal_distance is not None or cliff_distance is not None:
        if goal_distance is None or cliff_distance is None:
            raise ValueError(
                "goal_distance and cliff_distance must be provided together for "
                "explicit bilateral pair generation."
            )
        goal_distance = int(goal_distance)
        cliff_distance = int(cliff_distance)
        if goal_distance < 0 or cliff_distance < 0:
            raise ValueError("goal_distance and cliff_distance must be non-negative.")
        base_distance = int(cliff_distance)
    else:
        if symmetric_distance is None:
            raise ValueError(
                "build_bilateral_goal_cliff_terrain requires either symmetric_distance "
                "or explicit goal_distance/cliff_distance."
            )
        base_distance = int(symmetric_distance)
        goal_distance = int(np.ceil(float(base_distance) * correct_side_distance_scale))
        cliff_distance = int(np.ceil(float(base_distance) * incorrect_side_distance_scale))
    left_distance = goal_distance if normalized_goal_side == _GOAL_SIDE_LEFT else cliff_distance
    right_distance = goal_distance if normalized_goal_side == _GOAL_SIDE_RIGHT else cliff_distance
    grid, platform_start, platform_end, center_start, center_end = build_asymmetric_double_cliff_terrain(
        grid_h,
        grid_w,
        left_distance=int(left_distance),
        right_distance=int(right_distance),
        plateau_width=int(plateau_width),
        platform_y=platform_y,
    )
    goal_block_x = int(platform_start) if normalized_goal_side == _GOAL_SIDE_LEFT else int(platform_end)
    goal_block_y = int(platform_y) + 1
    grid[goal_block_y, goal_block_x] = FIXED
    goal_direction = -1 if normalized_goal_side == _GOAL_SIDE_LEFT else 1
    cliff_side = _GOAL_SIDE_RIGHT if normalized_goal_side == _GOAL_SIDE_LEFT else _GOAL_SIDE_LEFT
    cliff_front_gap_cell_x = -1
    cliff_front_gap_start_x = -1
    cliff_front_gap_end_x = -1
    cliff_post_gap_runway_start_x = -1
    cliff_post_gap_runway_end_x = -1
    terminal_cliff_edge_x = int(platform_start if cliff_side == _GOAL_SIDE_LEFT else platform_end)
    front_gap_cells: list[int] = []
    cliff_probe_gap_cell_x = -1
    cliff_probe_ledge_cell_x = -1
    cliff_probe_ledge_cell_y = -1
    cliff_direction = -1 if cliff_side == _GOAL_SIDE_LEFT else 1
    cliff_anchor_x = int(platform_start) if cliff_side == _GOAL_SIDE_LEFT else int(platform_end)
    if cliff_front_gap_width > 0:
        cliff_front_gap_cell_x = int(cliff_anchor_x + cliff_direction)
        front_gap_cells = [
            int(cliff_anchor_x + (cliff_direction * (offset + 1)))
            for offset in range(cliff_front_gap_width)
        ]
        runway_cells = [
            int(cliff_anchor_x + (cliff_direction * (cliff_front_gap_width + offset + 1)))
            for offset in range(cliff_post_gap_runway)
        ]
        occupied_front_feature_cells = front_gap_cells + runway_cells
        if occupied_front_feature_cells:
            if (
                min(occupied_front_feature_cells) < 0
                or max(occupied_front_feature_cells) >= int(grid_w)
            ):
                raise ValueError(
                    "Plateau-level front gap does not fit within the grid: "
                    f"grid_w={grid_w}, goal_distance={goal_distance}, cliff_distance={cliff_distance}, "
                    f"gap_width={cliff_front_gap_width}, post_gap_runway={cliff_post_gap_runway}"
                )
        if runway_cells:
            grid[platform_y, runway_cells] = FIXED
            cliff_post_gap_runway_start_x = int(min(runway_cells))
            cliff_post_gap_runway_end_x = int(max(runway_cells))
            terminal_cliff_edge_x = int(runway_cells[-1])
        cliff_front_gap_start_x = int(min(front_gap_cells))
        cliff_front_gap_end_x = int(max(front_gap_cells))
    if cliff_probe_gap_width > 0 or cliff_probe_ledge_width > 0:
        gap_cells: list[int] = []
        if cliff_probe_gap_width > 0:
            cliff_probe_gap_cell_x = int(cliff_anchor_x + cliff_direction)
            gap_cells = [
                int(cliff_anchor_x + (cliff_direction * (offset + 1)))
                for offset in range(cliff_probe_gap_width)
            ]
        cliff_probe_ledge_cell_x = (
            int(cliff_anchor_x + (cliff_direction * (cliff_probe_gap_width + 1)))
            if cliff_probe_ledge_width > 0
            else -1
        )
        cliff_probe_ledge_cell_y = int(platform_y) - 1 if cliff_probe_ledge_width > 0 else -1
        ledge_cells = [
            int(cliff_anchor_x + (cliff_direction * (cliff_probe_gap_width + 1 + offset)))
            for offset in range(cliff_probe_ledge_width)
        ]
        occupied_probe_cells = gap_cells + ledge_cells
        if occupied_probe_cells:
            if min(occupied_probe_cells) < 0 or max(occupied_probe_cells) >= int(grid_w):
                raise ValueError(
                    "cliff probe feature does not fit within the grid: "
                    f"grid_w={grid_w}, goal_distance={goal_distance}, cliff_distance={cliff_distance}, "
                    f"gap_width={cliff_probe_gap_width}, ledge_width={cliff_probe_ledge_width}"
                )
        if ledge_cells:
            grid[cliff_probe_ledge_cell_y, ledge_cells] = FIXED
    return grid, {
        "task_mode": _TASK_MODE_BILATERAL_GOAL_CLIFF,
        "goal_side": normalized_goal_side,
        "goal_direction": goal_direction,
        "cliff_side": cliff_side,
        "base_distance": int(base_distance),
        "goal_distance": int(goal_distance),
        "cliff_distance": int(cliff_distance),
        "correct_side_distance_scale": float(correct_side_distance_scale),
        "incorrect_side_distance_scale": float(incorrect_side_distance_scale),
        "platform_start_x": int(platform_start),
        "platform_end_x": int(platform_end),
        "center_start_x": int(center_start),
        "center_end_x": int(center_end),
        "cliff_edge_x": int(platform_start if cliff_side == _GOAL_SIDE_LEFT else platform_end),
        "goal_block_x": int(goal_block_x),
        "goal_block_y": int(goal_block_y),
        "goal_object_id": int(_GOAL_OBJECT_ID),
        "cliff_front_gap_width": int(cliff_front_gap_width),
        "cliff_post_gap_runway": int(cliff_post_gap_runway),
        "cliff_front_gap_cell_x": int(cliff_front_gap_cell_x),
        "cliff_front_gap_start_x": int(cliff_front_gap_start_x),
        "cliff_front_gap_end_x": int(cliff_front_gap_end_x),
        "cliff_post_gap_runway_start_x": int(cliff_post_gap_runway_start_x),
        "cliff_post_gap_runway_end_x": int(cliff_post_gap_runway_end_x),
        "terminal_cliff_edge_x": int(terminal_cliff_edge_x),
        "cliff_probe_gap_width": int(cliff_probe_gap_width),
        "cliff_probe_ledge_width": int(cliff_probe_ledge_width),
        "cliff_probe_gap_cell_x": int(cliff_probe_gap_cell_x),
        "cliff_probe_ledge_cell_x": int(cliff_probe_ledge_cell_x),
        "cliff_probe_ledge_cell_y": int(cliff_probe_ledge_cell_y),
    }


def _place_gaps(
    rng: np.random.Generator,
    zone_start: int,
    zone_end: int,
    gap_widths: list[int],
    gap_budget: int,
    *,
    min_gap_separation: int,
) -> list[tuple[int, int]]:
    """Place gaps in the variable zone, respecting constraints.

    Args:
        rng: Numpy random generator.
        zone_start: First column available for gaps (inclusive).
        zone_end: Last column available for gaps (inclusive).
        gap_widths: Allowed gap widths.
        gap_budget: Maximum total number of gaps.
        min_gap_separation: Minimum solid run required between adjacent gaps.

    Returns:
        List of (start_col, width) pairs.
    """
    if not gap_widths or gap_budget <= 0:
        return []

    zone_len = zone_end - zone_start + 1
    if zone_len < min(gap_widths):
        return []

    placed: list[tuple[int, int]] = []
    per_width_counts = {w: 0 for w in [1, 2, 3]}

    for _ in range(gap_budget):
        # Filter widths that haven't hit their per-width cap.
        available_widths = [
            w
            for w in gap_widths
            if per_width_counts.get(w, 0) < _MAX_GAPS_PER_WIDTH.get(w, 0)
        ]
        if not available_widths:
            break

        # Try widths in random order until one fits.
        rng.shuffle(available_widths)
        placed_this_round = False

        for width in available_widths:
            # Find valid start positions that preserve the minimum solid-run
            # invariant between this gap and every already placed gap.
            valid_starts = []
            for s in range(zone_len - width + 1):
                abs_start = zone_start + s
                abs_end = abs_start + width - 1
                is_valid = True
                for other_start, other_width in placed:
                    other_end = other_start + other_width - 1
                    if not (abs_end < other_start or other_end < abs_start):
                        is_valid = False
                        break
                    solid_run = (
                        abs_start - other_end - 1
                        if other_end < abs_start
                        else other_start - abs_end - 1
                    )
                    if solid_run < min_gap_separation:
                        is_valid = False
                        break
                if is_valid:
                    valid_starts.append(s)

            if not valid_starts:
                continue

            # Pick a random valid start.
            s = valid_starts[rng.integers(0, len(valid_starts))]
            abs_start = zone_start + s
            placed.append((abs_start, width))
            per_width_counts[width] = per_width_counts.get(width, 0) + 1
            placed_this_round = True
            break

        if not placed_this_round:
            break

    return placed


def _required_gap_span(
    gap_widths: list[int],
    *,
    min_gap_separation: int,
) -> int:
    """Return the minimum horizontal span required to place all listed gaps."""
    if not gap_widths:
        return 0
    return int(sum(gap_widths) + max(len(gap_widths) - 1, 0) * min_gap_separation)


def minimum_required_cliff_x(
    *,
    base_width: int,
    gap_widths: list[int],
    min_gap_separation: int = 2,
    min_post_gap_runway: int = 2,
) -> int:
    """Return the minimum cliff_x that can host the requested gap pattern.

    The gap placement zone starts at ``base_width`` and ends at
    ``cliff_x - min_post_gap_runway`` so a minimum solid runway remains after
    the final gap. The generator never allows single-cell solid terrain runs.
    """
    min_gap_separation = _resolve_min_solid_run(min_gap_separation)
    min_post_gap_runway = _resolve_min_solid_run(min_post_gap_runway)
    required_zone_len = _required_gap_span(
        list(gap_widths),
        min_gap_separation=min_gap_separation,
    )
    return int(base_width + required_zone_len + min_post_gap_runway - 1)


def _enumerate_exact_gap_layouts(
    zone_start: int,
    zone_end: int,
    gap_widths: list[int],
    *,
    min_gap_separation: int,
) -> list[list[tuple[int, int]]]:
    """Enumerate all exact gap layouts that fit inside the placement zone."""
    if not gap_widths:
        return [[]]
    if zone_end < zone_start:
        return []

    layouts: set[tuple[tuple[int, int], ...]] = set()
    for ordered_widths in sorted(set(itertools.permutations(gap_widths))):
        def _place(width_index: int, next_start: int, placed: list[tuple[int, int]]) -> None:
            if width_index >= len(ordered_widths):
                layouts.add(tuple(placed))
                return

            width = int(ordered_widths[width_index])
            remaining = list(ordered_widths[width_index + 1 :])
            tail_span = _required_gap_span(
                remaining,
                min_gap_separation=min_gap_separation,
            )
            gap_and_sep_tail = (
                min_gap_separation + tail_span
                if remaining
                else 0
            )
            max_start = int(zone_end - width + 1 - gap_and_sep_tail)
            for start in range(int(next_start), max_start + 1):
                placed.append((start, width))
                _place(
                    width_index + 1,
                    start + width + min_gap_separation,
                    placed,
                )
                placed.pop()

        _place(0, int(zone_start), [])

    return [list(layout) for layout in sorted(layouts)]


def _place_exact_gaps(
    rng: np.random.Generator,
    zone_start: int,
    zone_end: int,
    gap_widths: list[int],
    *,
    min_gap_separation: int,
) -> list[tuple[int, int]]:
    """Sample one exact gap layout from all legal placements."""
    layouts = _enumerate_exact_gap_layouts(
        zone_start,
        zone_end,
        list(gap_widths),
        min_gap_separation=min_gap_separation,
    )
    if not layouts:
        raise ValueError(
            "No valid exact gap layouts for "
            f"zone=[{zone_start}, {zone_end}] widths={gap_widths} "
            f"min_gap_separation={min_gap_separation}"
    )
    return list(layouts[int(rng.integers(0, len(layouts)))])


def _resolve_zone_partition(
    *,
    grid_w: int,
    spawn_plateau_width: int,
    approach_width: int,
    randomized_zone_width: int,
    remainder_width: int,
) -> tuple[int, int]:
    """Resolve the randomized-zone bounds for the structured W=30 layout."""
    spawn_plateau_width = int(spawn_plateau_width)
    approach_width = int(approach_width)
    randomized_zone_width = int(randomized_zone_width)
    remainder_width = int(remainder_width)
    if min(
        spawn_plateau_width,
        approach_width,
        randomized_zone_width,
        remainder_width,
    ) < 0:
        raise ValueError("Zone partition widths must be non-negative.")
    if randomized_zone_width <= 0:
        raise ValueError("randomized_zone_width must be positive.")
    total_width = (
        spawn_plateau_width
        + approach_width
        + randomized_zone_width
        + remainder_width
    )
    if total_width != int(grid_w):
        raise ValueError(
            "Zone partition must exactly match grid width: "
            f"spawn_plateau_width={spawn_plateau_width}, approach_width={approach_width}, "
            f"randomized_zone_width={randomized_zone_width}, remainder_width={remainder_width}, "
            f"grid_w={grid_w}."
        )
    zone_start = spawn_plateau_width + approach_width
    zone_end = zone_start + randomized_zone_width - 1
    return int(zone_start), int(zone_end)


def _resolve_zone_single_gap_width(
    *,
    single_gap_width: int | None,
    gap_widths: list[int],
    gap_budget: int,
    exact_gap_widths: list[int],
) -> int:
    """Resolve the exact single-gap width for the structured zone mode."""
    if single_gap_width is not None:
        return int(single_gap_width)
    if exact_gap_widths:
        if len(exact_gap_widths) != 1:
            raise ValueError(
                "zone_single_gap mode requires exactly one exact gap width when "
                "exact_gap_widths is provided."
            )
        return int(exact_gap_widths[0])
    if int(gap_budget) <= 0:
        return 0
    unique_widths = sorted({int(width) for width in gap_widths})
    if not unique_widths:
        return 0
    if len(unique_widths) != 1:
        raise ValueError(
            "zone_single_gap mode requires a single unambiguous gap width."
        )
    return int(unique_widths[0])


def _enumerate_zone_single_gap_layouts(
    zone_start: int,
    zone_end: int,
    *,
    gap_width: int,
    min_post_gap_runway: int,
    gap_start_min: int | None = None,
    gap_start_max: int | None = None,
) -> list[tuple[int, list[tuple[int, int]]]]:
    """Enumerate legal single-gap layouts within the structured randomized zone."""
    gap_width = int(gap_width)
    min_post_gap_runway = _resolve_min_solid_run(min_post_gap_runway)
    if gap_width <= 0:
        raise ValueError("gap_width must be positive when enumerating gap layouts.")

    latest_gap_start = zone_end - gap_width - min_post_gap_runway + 1
    if latest_gap_start < zone_start:
        return []
    resolved_gap_start_min = (
        int(zone_start) if gap_start_min is None else max(int(zone_start), int(gap_start_min))
    )
    resolved_gap_start_max = (
        int(latest_gap_start)
        if gap_start_max is None
        else min(int(latest_gap_start), int(gap_start_max))
    )
    if resolved_gap_start_max < resolved_gap_start_min:
        return []

    layouts: list[tuple[int, list[tuple[int, int]]]] = []
    for gap_start in range(int(resolved_gap_start_min), int(resolved_gap_start_max) + 1):
        gap_end = gap_start + gap_width - 1
        min_cliff_x = gap_end + min_post_gap_runway
        for cliff_x in range(int(min_cliff_x), int(zone_end) + 1):
            layouts.append((int(cliff_x), [(int(gap_start), int(gap_width))]))
    return layouts


def generate_terrain(
    seed: int,
    *,
    grid_h: int = 10,
    grid_w: int = 35,
    platform_y: int | None = None,
    plateau_height: int | None = None,
    base_width: int = 10,
    cliff_x_min: int = 10,
    cliff_x_max: int = 24,
    gap_widths: list[int] | None = None,
    gap_budget: int = 0,
    exact_gap_widths: list[int] | None = None,
    min_gap_separation: int = 2,
    min_post_gap_runway: int = 2,
    hazard_mode: str = _HAZARD_MODE_LEGACY,
    spawn_plateau_width: int | None = None,
    approach_width: int | None = None,
    randomized_zone_width: int | None = None,
    remainder_width: int | None = None,
    single_gap_width: int | None = None,
    layout_sampling: str = _LAYOUT_SAMPLING_UNIFORM_LEGAL,
    gap_start_min: int | None = None,
    gap_start_max: int | None = None,
    task_mode: str = _TASK_MODE_LEGACY_CLIFF,
    center_plateau_width: int | None = None,
    symmetric_distance_min: int | None = None,
    symmetric_distance_max: int | None = None,
    goal_side: str | None = None,
    correct_side_distance_scale: float = 1.0,
    incorrect_side_distance_scale: float = 1.0,
    cliff_front_gap_width: int = 0,
    cliff_front_gap_widths: list[int] | None = None,
    cliff_post_gap_runway: int = 0,
    cliff_probe_gap_width: int = 0,
    cliff_probe_ledge_width: int = 0,
    distance_pairs: list[dict[str, int] | list[int] | tuple[int, int]] | None = None,
) -> tuple[np.ndarray, int, list[tuple[int, int]], dict[str, object]]:
    """Generate a single terrain with randomized cliff position and gaps.

    Args:
        seed: Random seed for this terrain.
        grid_h: Grid height.
        grid_w: Grid width.
        platform_y: Row containing the walking surface. Defaults to 3.
        plateau_height: Backward-compatible alias for the old top-surface height.
        base_width: Width of the guaranteed base plateau (x=0..base_width-1).
        cliff_x_min: Minimum cliff_x (inclusive).
        cliff_x_max: Maximum cliff_x (inclusive).
        gap_widths: Allowed gap widths. None or empty = no gaps.
        gap_budget: Max gaps to place.
        exact_gap_widths: Exact multiset of gaps to carve, if provided.
        min_gap_separation: Number of solid cells required between adjacent gaps.
        min_post_gap_runway: Number of solid platform cells required between the
            final gap and the cliff edge. Both solid-run constraints are clamped
            to at least 2 to prevent 1-cell plateau fragments.
        hazard_mode: Terrain generator mode. ``legacy`` preserves the historical
            cliff-first generator; ``zone_single_gap`` uses the structured
            spawn/approach/randomized-zone layout.
        task_mode: High-level task family. ``legacy_cliff`` uses the existing
            cliff/gap semantics; ``bilateral_goal_cliff`` builds the centered
            bilateral wall-vs-cliff task.

    Returns:
        ``(terrain_grid, cliff_x, gaps, metadata)`` where metadata carries any
        task-specific fields needed downstream.
    """
    platform_y = _resolve_platform_y(platform_y=platform_y, plateau_height=plateau_height)
    rng = np.random.default_rng(seed)
    exact_gap_widths_list = list(exact_gap_widths) if exact_gap_widths else []
    gap_widths_list = list(gap_widths) if gap_widths else []
    hazard_mode = str(hazard_mode).strip().lower()
    task_mode = str(task_mode).strip().lower()
    active_gap_widths = exact_gap_widths_list or gap_widths_list
    min_gap_separation = _resolve_min_solid_run(min_gap_separation)
    min_post_gap_runway = _resolve_min_solid_run(min_post_gap_runway)

    if task_mode == _TASK_MODE_BILATERAL_GOAL_CLIFF:
        active_cliff_front_gap_widths = [int(width) for width in list(cliff_front_gap_widths or [])]
        sampled_cliff_front_gap_width = int(cliff_front_gap_width)
        if sampled_cliff_front_gap_width <= 0 and active_cliff_front_gap_widths:
            pair_idx = int(rng.integers(0, len(active_cliff_front_gap_widths)))
            sampled_cliff_front_gap_width = int(active_cliff_front_gap_widths[pair_idx])
        sampled_goal_side = (
            str(goal_side).strip().lower()
            if goal_side is not None
            else (_GOAL_SIDE_LEFT if int(rng.integers(0, 2)) == 0 else _GOAL_SIDE_RIGHT)
        )
        active_pairs = list(distance_pairs or [])
        build_kwargs = {
            "goal_side": sampled_goal_side,
            "plateau_width": int(center_plateau_width if center_plateau_width is not None else base_width),
            "platform_y": platform_y,
            "correct_side_distance_scale": float(correct_side_distance_scale),
            "incorrect_side_distance_scale": float(incorrect_side_distance_scale),
            "cliff_front_gap_width": int(sampled_cliff_front_gap_width),
            "cliff_post_gap_runway": int(cliff_post_gap_runway),
            "cliff_probe_gap_width": int(cliff_probe_gap_width),
            "cliff_probe_ledge_width": int(cliff_probe_ledge_width),
        }
        if active_pairs:
            normalized_pairs: list[tuple[int, int]] = []
            for pair in active_pairs:
                if hasattr(pair, "keys") and "goal_distance" in pair and "cliff_distance" in pair:
                    normalized_goal_distance = int(pair["goal_distance"])
                    normalized_cliff_distance = int(pair["cliff_distance"])
                else:
                    if len(pair) != 2:
                        raise ValueError("distance_pairs entries must have exactly two values.")
                    normalized_goal_distance = int(pair[0])
                    normalized_cliff_distance = int(pair[1])
                normalized_pairs.append((normalized_goal_distance, normalized_cliff_distance))
            pair_idx = int(rng.integers(0, len(normalized_pairs)))
            sampled_goal_distance, sampled_cliff_distance = normalized_pairs[pair_idx]
            grid, metadata = build_bilateral_goal_cliff_terrain(
                grid_h,
                grid_w,
                goal_distance=sampled_goal_distance,
                cliff_distance=sampled_cliff_distance,
                **build_kwargs,
            )
        else:
            if symmetric_distance_min is None or symmetric_distance_max is None:
                raise ValueError(
                    "bilateral_goal_cliff task mode requires symmetric_distance_min and "
                    "symmetric_distance_max when distance_pairs are not provided."
                )
            distance_min = int(symmetric_distance_min)
            distance_max = int(symmetric_distance_max)
            if distance_max < distance_min:
                raise ValueError("symmetric_distance_max must be >= symmetric_distance_min.")
            sampled_distance = int(rng.integers(distance_min, distance_max + 1))
            grid, metadata = build_bilateral_goal_cliff_terrain(
                grid_h,
                grid_w,
                symmetric_distance=sampled_distance,
                **build_kwargs,
            )
        front_gap_width = int(metadata.get("cliff_front_gap_width", 0))
        gaps = (
            [(int(metadata["cliff_front_gap_start_x"]), front_gap_width)]
            if front_gap_width > 0 and int(metadata.get("cliff_front_gap_start_x", -1)) >= 0
            else []
        )
        return (
            grid,
            int(metadata["cliff_edge_x"]) if front_gap_width > 0 else int(metadata["platform_end_x"]),
            gaps,
            metadata,
        )

    if hazard_mode == _HAZARD_MODE_ZONE_SINGLE_GAP:
        if None in {
            spawn_plateau_width,
            approach_width,
            randomized_zone_width,
            remainder_width,
        }:
            raise ValueError(
                "zone_single_gap mode requires spawn_plateau_width, approach_width, "
                "randomized_zone_width, and remainder_width."
            )
        zone_start, zone_end = _resolve_zone_partition(
            grid_w=grid_w,
            spawn_plateau_width=int(spawn_plateau_width),
            approach_width=int(approach_width),
            randomized_zone_width=int(randomized_zone_width),
            remainder_width=int(remainder_width),
        )
        cliff_x: int
        gaps: list[tuple[int, int]]
        resolved_gap_width = _resolve_zone_single_gap_width(
            single_gap_width=single_gap_width,
            gap_widths=gap_widths_list,
            gap_budget=gap_budget,
            exact_gap_widths=exact_gap_widths_list,
        )
        if resolved_gap_width <= 0:
            cliff_x = int(rng.integers(zone_start, zone_end + 1))
            gaps = []
        else:
            layout_sampling = str(layout_sampling).strip().lower()
            if layout_sampling != _LAYOUT_SAMPLING_UNIFORM_LEGAL:
                raise ValueError(
                    "zone_single_gap mode only supports "
                    f"layout_sampling={_LAYOUT_SAMPLING_UNIFORM_LEGAL!r}; got "
                    f"{layout_sampling!r}."
                )
            layouts = _enumerate_zone_single_gap_layouts(
                zone_start,
                zone_end,
                gap_width=resolved_gap_width,
                min_post_gap_runway=min_post_gap_runway,
                gap_start_min=gap_start_min,
                gap_start_max=gap_start_max,
            )
            if not layouts:
                raise ValueError(
                    "Structured terrain config cannot realize requested gap layout: "
                    f"zone=[{zone_start}, {zone_end}] gap_width={resolved_gap_width} "
                    f"min_post_gap_runway={min_post_gap_runway} "
                    f"gap_start_min={gap_start_min} gap_start_max={gap_start_max}"
                )
            cliff_x, gaps = layouts[int(rng.integers(0, len(layouts)))]
        grid = build_cliff_gap_terrain(
            grid_h,
            grid_w,
            cliff_x,
            gaps,
            platform_y=platform_y,
        )
        return grid, cliff_x, gaps, {
            "task_mode": _TASK_MODE_LEGACY_CLIFF,
            "gaps": [(int(start), int(width)) for start, width in gaps],
        }

    if hazard_mode != _HAZARD_MODE_LEGACY:
        raise ValueError(f"Unsupported hazard_mode={hazard_mode!r}")

    if active_gap_widths:
        min_feasible_cliff_x = minimum_required_cliff_x(
            base_width=base_width,
            gap_widths=active_gap_widths if exact_gap_widths_list else [min(active_gap_widths)],
            min_gap_separation=min_gap_separation,
            min_post_gap_runway=min_post_gap_runway,
        )
        if cliff_x_max < min_feasible_cliff_x:
            raise ValueError(
                f"Terrain config cannot realize requested hazards: "
                f"cliff_x_max={cliff_x_max} < minimum feasible {min_feasible_cliff_x}"
            )
        cliff_x_min = max(int(cliff_x_min), int(min_feasible_cliff_x))

    cliff_x = int(rng.integers(cliff_x_min, cliff_x_max + 1))

    gaps: list[tuple[int, int]] = []

    if active_gap_widths:
        # Gaps only in the variable zone, leaving a solid runway before the cliff.
        zone_start = base_width
        zone_end = cliff_x - int(min_post_gap_runway)
        if zone_end >= zone_start:
            if exact_gap_widths_list:
                gaps = _place_exact_gaps(
                    rng,
                    zone_start,
                    zone_end,
                    exact_gap_widths_list,
                    min_gap_separation=min_gap_separation,
                )
            elif gap_budget > 0:
                gaps = _place_gaps(
                    rng,
                    zone_start,
                    zone_end,
                    gap_widths_list,
                    gap_budget,
                    min_gap_separation=min_gap_separation,
                )

    if gaps:
        grid = build_cliff_gap_terrain(
            grid_h, grid_w, cliff_x, gaps, platform_y=platform_y
        )
    else:
        grid = build_cliff_terrain(grid_h, grid_w, cliff_x, platform_y=platform_y)

    return grid, cliff_x, gaps, {
        "task_mode": _TASK_MODE_LEGACY_CLIFF,
        "gaps": [(int(start), int(width)) for start, width in gaps],
    }


def generate_generation_terrains(
    gen_seed: int,
    n_episodes: int,
    *,
    grid_h: int = 10,
    grid_w: int = 35,
    platform_y: int | None = None,
    plateau_height: int | None = None,
    base_width: int = 10,
    cliff_x_min: int = 10,
    cliff_x_max: int = 24,
    gap_widths: list[int] | None = None,
    gap_budget: int = 0,
    exact_gap_widths: list[int] | None = None,
    min_gap_separation: int = 2,
    min_post_gap_runway: int = 2,
    hazard_mode: str = _HAZARD_MODE_LEGACY,
    spawn_plateau_width: int | None = None,
    approach_width: int | None = None,
    randomized_zone_width: int | None = None,
    remainder_width: int | None = None,
    single_gap_width: int | None = None,
    layout_sampling: str = _LAYOUT_SAMPLING_UNIFORM_LEGAL,
    gap_start_min: int | None = None,
    gap_start_max: int | None = None,
    gap_start_bands: list[list[int]] | None = None,
    task_mode: str = _TASK_MODE_LEGACY_CLIFF,
    center_plateau_width: int | None = None,
    symmetric_distance_min: int | None = None,
    symmetric_distance_max: int | None = None,
    goal_side: str | None = None,
    correct_side_distance_scale: float = 1.0,
    incorrect_side_distance_scale: float = 1.0,
    cliff_front_gap_width: int = 0,
    cliff_front_gap_widths: list[int] | None = None,
    cliff_post_gap_runway: int = 0,
    cliff_probe_gap_width: int = 0,
    cliff_probe_ledge_width: int = 0,
    distance_pairs: list[dict[str, int] | list[int] | tuple[int, int]] | None = None,
) -> list[tuple[np.ndarray, int, list[tuple[int, int]], dict[str, object]]]:
    """Generate n_episodes terrains for one generation.

    All candidates in a generation see the same terrain set.
    """
    platform_y = _resolve_platform_y(platform_y=platform_y, plateau_height=plateau_height)
    rng = np.random.default_rng(gen_seed)
    results = []
    resolved_gap_start_bands = list(gap_start_bands or [])
    for episode_idx in range(n_episodes):
        episode_seed = int(rng.integers(0, 2**31))
        resolved_gap_start_min = gap_start_min
        resolved_gap_start_max = gap_start_max
        if resolved_gap_start_bands:
            band = resolved_gap_start_bands[int(episode_idx) % len(resolved_gap_start_bands)]
            if len(band) != 2:
                raise ValueError("gap_start_bands entries must be [min, max] pairs.")
            resolved_gap_start_min = int(band[0])
            resolved_gap_start_max = int(band[1])
        results.append(
            generate_terrain(
                episode_seed,
                grid_h=grid_h,
                grid_w=grid_w,
                platform_y=platform_y,
                base_width=base_width,
                cliff_x_min=cliff_x_min,
                cliff_x_max=cliff_x_max,
                gap_widths=gap_widths,
                gap_budget=gap_budget,
                exact_gap_widths=exact_gap_widths,
                min_gap_separation=min_gap_separation,
                min_post_gap_runway=min_post_gap_runway,
                hazard_mode=hazard_mode,
                spawn_plateau_width=spawn_plateau_width,
                approach_width=approach_width,
                randomized_zone_width=randomized_zone_width,
                remainder_width=remainder_width,
                single_gap_width=single_gap_width,
                layout_sampling=layout_sampling,
                gap_start_min=resolved_gap_start_min,
                gap_start_max=resolved_gap_start_max,
                task_mode=task_mode,
                center_plateau_width=center_plateau_width,
                symmetric_distance_min=symmetric_distance_min,
                symmetric_distance_max=symmetric_distance_max,
                goal_side=goal_side,
                correct_side_distance_scale=correct_side_distance_scale,
                incorrect_side_distance_scale=incorrect_side_distance_scale,
                cliff_front_gap_width=cliff_front_gap_width,
                cliff_front_gap_widths=cliff_front_gap_widths,
                cliff_post_gap_runway=cliff_post_gap_runway,
                cliff_probe_gap_width=cliff_probe_gap_width,
                cliff_probe_ledge_width=cliff_probe_ledge_width,
                distance_pairs=distance_pairs,
            )
        )
    return results


def make_robot_body(
    robot_w: int = 7,
    robot_h: int = 2,
    *,
    shape: str = "rectangle",
) -> np.ndarray:
    """Create the fixed robot body for the configured shape."""
    resolved_shape = _normalize_body_shape(shape)
    if resolved_shape == "rectangle":
        return np.full((robot_h, robot_w), CONTRACTILE, dtype=np.int32)
    if resolved_shape == "comb":
        if robot_w != 7 or robot_h != 2:
            raise ValueError(
                "body.shape='comb' requires terrain.robot_w=7 and terrain.robot_h=2"
            )
        return np.array(
            [
                [CONTRACTILE, EMPTY, CONTRACTILE, EMPTY, CONTRACTILE, EMPTY, CONTRACTILE],
                [
                    CONTRACTILE,
                    CONTRACTILE,
                    CONTRACTILE,
                    CONTRACTILE,
                    CONTRACTILE,
                    CONTRACTILE,
                    CONTRACTILE,
                ],
            ],
            dtype=np.int32,
        )
    if robot_w != 7 or robot_h != 3:
        raise ValueError(
            "body.shape='comb_7x3' requires terrain.robot_w=7 and terrain.robot_h=3"
        )
    return np.array(
        [
            [CONTRACTILE, EMPTY, CONTRACTILE, EMPTY, CONTRACTILE, EMPTY, CONTRACTILE],
            [
                CONTRACTILE,
                CONTRACTILE,
                CONTRACTILE,
                CONTRACTILE,
                CONTRACTILE,
                CONTRACTILE,
                CONTRACTILE,
            ],
            [
                CONTRACTILE,
                CONTRACTILE,
                CONTRACTILE,
                CONTRACTILE,
                CONTRACTILE,
                CONTRACTILE,
                CONTRACTILE,
            ],
        ],
        dtype=np.int32,
    )


def fixed_body_alive_mask(robot_morph: np.ndarray) -> np.ndarray:
    """Return the active-cell mask for a fixed morphology."""
    return np.asarray(robot_morph != EMPTY, dtype=bool)


def max_canvas_size(
    grid_h: int,
    grid_w: int,
    *,
    spawn_x: int,
    spawn_y: int,
) -> tuple[int, int]:
    """Return the largest robot canvas that fits at the configured spawn."""
    return max(0, int(grid_h) - int(spawn_y)), max(0, int(grid_w) - int(spawn_x))


def resolve_growth_seed_position(
    canvas_h: int,
    canvas_w: int,
    seed_position: str = "center",
) -> tuple[int, int]:
    """Resolve a growth seed location on the local robot canvas."""
    position = str(seed_position).strip().lower()
    if position != "center":
        raise ValueError(f"Unsupported body.growth.seed_position={seed_position!r}")
    return int(canvas_h // 2), int(canvas_w // 2)


def _resolve_platform_y(
    *,
    platform_y: int | None,
    plateau_height: int | None,
) -> int:
    """Resolve the canonical platform row from current or legacy config names."""
    if platform_y is not None:
        return int(platform_y)
    if plateau_height is not None:
        return int(plateau_height)
    return 3
