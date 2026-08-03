"""Tests for terrain generation."""

import numpy as np
import pytest

from jax_evogym.constants import EMPTY, FIXED, CONTRACTILE

from sensing_edges.terrain import (
    _enumerate_zone_single_gap_layouts,
    build_bilateral_goal_cliff_terrain,
    build_cliff_terrain,
    build_cliff_gap_terrain,
    build_symmetric_double_cliff_terrain,
    generate_terrain,
    generate_generation_terrains,
    make_robot_body,
    minimum_required_cliff_x,
)


PHASE_1_ZONE_MODE_KWARGS = {
    "grid_h": 10,
    "grid_w": 30,
    "platform_y": 3,
    "base_width": 10,
    "cliff_x_min": 10,
    "cliff_x_max": 19,
    "hazard_mode": "zone_single_gap",
    "spawn_plateau_width": 10,
    "approach_width": 0,
    "randomized_zone_width": 10,
    "remainder_width": 10,
    "min_gap_separation": 2,
    "min_post_gap_runway": 2,
    "layout_sampling": "uniform_legal_layouts",
}


PHASE_2_ZONE_MODE_KWARGS = {
    "grid_h": 10,
    "grid_w": 30,
    "platform_y": 3,
    "base_width": 10,
    "cliff_x_min": 10,
    "cliff_x_max": 24,
    "hazard_mode": "zone_single_gap",
    "spawn_plateau_width": 10,
    "approach_width": 0,
    "randomized_zone_width": 15,
    "remainder_width": 5,
    "gap_start_min": 10,
    "gap_start_max": 19,
    "min_gap_separation": 2,
    "min_post_gap_runway": 2,
    "layout_sampling": "uniform_legal_layouts",
}


def _solid_run_lengths(gaps: list[tuple[int, int]], cliff_x: int) -> list[int]:
    ordered = sorted(gaps)
    if not ordered:
        return []
    runs: list[int] = []
    prev_end = ordered[0][0] + ordered[0][1] - 1
    for start, width in ordered[1:]:
        runs.append(start - prev_end - 1)
        prev_end = start + width - 1
    runs.append(cliff_x - prev_end)
    return runs


class TestBuildCliffTerrain:
    def test_shape(self):
        grid = build_cliff_terrain(10, 35, cliff_x=20)
        assert grid.shape == (10, 35)
        assert grid.dtype == np.int32

    def test_row_zero_is_empty(self):
        grid = build_cliff_terrain(10, 35, cliff_x=15)
        assert np.all(grid[0, :] == EMPTY)

    def test_platform_present(self):
        grid = build_cliff_terrain(10, 35, cliff_x=20)
        # Single platform row at y=3 from x=0 to x=20 inclusive.
        assert np.all(grid[3, :21] == FIXED)
        assert np.all(grid[2, :] == EMPTY)

    def test_platform_ends_at_cliff(self):
        grid = build_cliff_terrain(10, 35, cliff_x=15)
        # x=16 onward should be empty at platform height.
        assert np.all(grid[3, 16:] == EMPTY)

    def test_off_platform_rows_empty(self):
        grid = build_cliff_terrain(10, 35, cliff_x=20)
        assert np.all(grid[4:, :] == EMPTY)
        assert np.all(grid[:3, 21:] == EMPTY)

    def test_min_cliff(self):
        grid = build_cliff_terrain(10, 35, cliff_x=10)
        assert np.all(grid[3, :11] == FIXED)
        assert np.all(grid[3, 11:] == EMPTY)

    def test_max_cliff(self):
        grid = build_cliff_terrain(10, 35, cliff_x=24)
        assert np.all(grid[3, :25] == FIXED)
        assert np.all(grid[3, 25:] == EMPTY)


class TestBuildCliffGapTerrain:
    def test_gap_cuts_platform(self):
        gaps = [(14, 2)]  # 2-wide gap at x=14..15
        grid = build_cliff_gap_terrain(10, 35, cliff_x=20, gaps=gaps)
        assert grid[3, 14] == EMPTY
        assert grid[3, 15] == EMPTY
        # Adjacent cells still FIXED.
        assert grid[3, 13] == FIXED
        assert grid[3, 16] == FIXED
        assert np.all(grid[2, :] == EMPTY)

    def test_row_zero_remains_empty(self):
        gaps = [(12, 3)]
        grid = build_cliff_gap_terrain(10, 35, cliff_x=20, gaps=gaps)
        assert np.all(grid[0, :] == EMPTY)

    def test_multiple_gaps(self):
        gaps = [(11, 1), (14, 2)]
        grid = build_cliff_gap_terrain(10, 35, cliff_x=20, gaps=gaps)
        assert grid[3, 11] == EMPTY
        assert grid[3, 14] == EMPTY
        assert grid[3, 15] == EMPTY


class TestBuildSymmetricDoubleCliffTerrain:
    def test_distance_zero_keeps_only_center_plateau(self):
        grid, platform_start, platform_end = build_symmetric_double_cliff_terrain(
            10,
            30,
            symmetric_distance=0,
            plateau_width=10,
        )

        assert platform_start == 10
        assert platform_end == 19
        assert np.all(grid[3, 10:20] == FIXED)
        assert np.all(grid[3, :10] == EMPTY)
        assert np.all(grid[3, 20:] == EMPTY)

    def test_distance_five_extends_platform_symmetrically(self):
        grid, platform_start, platform_end = build_symmetric_double_cliff_terrain(
            10,
            30,
            symmetric_distance=5,
            plateau_width=10,
        )

        assert platform_start == 5
        assert platform_end == 24
        assert np.all(grid[3, 5:25] == FIXED)
        assert np.all(grid[3, :5] == EMPTY)
        assert np.all(grid[3, 25:] == EMPTY)

    def test_rejects_distance_that_overflows_grid(self):
        with pytest.raises(ValueError, match="does not fit within the grid"):
            build_symmetric_double_cliff_terrain(
                10,
                30,
                symmetric_distance=11,
                plateau_width=10,
            )


class TestBuildBilateralGoalCliffTerrain:
    def test_left_goal_places_single_raised_goal_block(self):
        grid, metadata = build_bilateral_goal_cliff_terrain(
            10,
            30,
            symmetric_distance=3,
            goal_side="left",
            plateau_width=10,
        )

        assert metadata["goal_side"] == "left"
        assert metadata["goal_distance"] == 3
        assert metadata["goal_block_x"] == 7
        assert metadata["goal_block_y"] == 4
        assert np.all(grid[3, 7:23] == FIXED)
        assert grid[4, 7] == FIXED
        assert np.count_nonzero(grid[4, :]) == 1

    def test_explicit_goal_and_cliff_distances_support_fixed_pairs(self):
        grid, metadata = build_bilateral_goal_cliff_terrain(
            10,
            30,
            goal_distance=6,
            cliff_distance=3,
            goal_side="right",
            plateau_width=10,
            cliff_probe_gap_width=1,
            cliff_probe_ledge_width=3,
        )

        assert metadata["goal_distance"] == 6
        assert metadata["cliff_distance"] == 3
        assert metadata["base_distance"] == 3
        assert metadata["goal_side"] == "right"
        assert metadata["cliff_side"] == "left"
        assert grid[3, 6] == EMPTY
        assert metadata["cliff_probe_ledge_cell_y"] == 2
        assert np.all(grid[2, 3:6] == FIXED)
        assert np.all(grid[3, 3:6] == EMPTY)
        assert grid[4, 25] == FIXED

    def test_asymmetric_distance_scales_extend_goal_side_farther(self):
        grid, metadata = build_bilateral_goal_cliff_terrain(
            10,
            30,
            symmetric_distance=3,
            goal_side="left",
            plateau_width=10,
            correct_side_distance_scale=1.5,
            incorrect_side_distance_scale=1.0,
        )

        assert metadata["base_distance"] == 3
        assert metadata["goal_distance"] == 5
        assert metadata["cliff_distance"] == 3
        assert metadata["goal_block_x"] == 5
        assert metadata["cliff_edge_x"] == 22
        assert np.all(grid[3, 5:23] == FIXED)
        assert np.all(grid[3, :5] == 0)
        assert np.all(grid[3, 23:] == 0)

    def test_right_cliff_probe_gap_adds_warning_ledge(self):
        grid, metadata = build_bilateral_goal_cliff_terrain(
            10,
            30,
            symmetric_distance=6,
            goal_side="left",
            plateau_width=10,
            cliff_probe_gap_width=1,
            cliff_probe_ledge_width=3,
        )

        assert metadata["cliff_side"] == "right"
        assert metadata["cliff_probe_gap_width"] == 1
        assert metadata["cliff_probe_ledge_width"] == 3
        assert metadata["cliff_probe_gap_cell_x"] == 26
        assert metadata["cliff_probe_ledge_cell_x"] == 27
        assert metadata["cliff_probe_ledge_cell_y"] == 2
        assert grid[3, 25] == FIXED
        assert grid[3, 26] == EMPTY
        assert np.all(grid[2, 27:30] == FIXED)
        assert np.all(grid[3, 27:30] == EMPTY)

    def test_right_cliff_probe_ledge_without_gap_is_supported(self):
        grid, metadata = build_bilateral_goal_cliff_terrain(
            10,
            30,
            symmetric_distance=6,
            goal_side="left",
            plateau_width=10,
            cliff_probe_gap_width=0,
            cliff_probe_ledge_width=3,
        )

        assert metadata["cliff_side"] == "right"
        assert metadata["cliff_probe_gap_width"] == 0
        assert metadata["cliff_probe_ledge_width"] == 3
        assert metadata["cliff_probe_gap_cell_x"] == -1
        assert metadata["cliff_probe_ledge_cell_x"] == 26
        assert metadata["cliff_probe_ledge_cell_y"] == 2
        assert grid[3, 25] == FIXED
        assert np.all(grid[2, 26:29] == FIXED)
        assert np.all(grid[3, 26:29] == EMPTY)

    def test_plateau_gap_cuts_plateau_and_adds_post_gap_runway(self):
        grid, metadata = build_bilateral_goal_cliff_terrain(
            10,
            30,
            goal_distance=4,
            cliff_distance=2,
            goal_side="left",
            plateau_width=10,
            cliff_front_gap_width=2,
            cliff_post_gap_runway=1,
        )

        assert metadata["cliff_side"] == "right"
        assert metadata["cliff_edge_x"] == 21
        assert metadata["cliff_front_gap_width"] == 2
        assert metadata["cliff_front_gap_start_x"] == 22
        assert metadata["cliff_front_gap_end_x"] == 23
        assert metadata["cliff_post_gap_runway_start_x"] == 24
        assert metadata["cliff_post_gap_runway_end_x"] == 24
        assert metadata["terminal_cliff_edge_x"] == 24
        assert grid[3, 21] == FIXED
        assert np.all(grid[3, 22:24] == EMPTY)
        assert grid[3, 24] == FIXED
        assert grid[3, 25] == EMPTY

    def test_plateau_gap_cannot_be_combined_with_probe_ledge_features(self):
        with pytest.raises(ValueError, match="Plateau-level front gaps cannot be combined"):
            build_bilateral_goal_cliff_terrain(
                10,
                30,
                symmetric_distance=3,
                goal_side="left",
                plateau_width=10,
                cliff_front_gap_width=1,
                cliff_probe_ledge_width=3,
            )


class TestGenerateTerrain:
    def test_determinism(self):
        t1, c1, g1, m1 = generate_terrain(42, cliff_x_min=10, cliff_x_max=24)
        t2, c2, g2, m2 = generate_terrain(42, cliff_x_min=10, cliff_x_max=24)
        assert np.array_equal(t1, t2)
        assert c1 == c2
        assert g1 == g2
        assert m1 == m2

    def test_different_seeds_differ(self):
        t1, c1, _, _ = generate_terrain(42, cliff_x_min=10, cliff_x_max=24)
        t2, c2, _, _ = generate_terrain(99, cliff_x_min=10, cliff_x_max=24)
        # Very unlikely to be identical with different seeds.
        assert not np.array_equal(t1, t2) or c1 != c2

    def test_cliff_in_range(self):
        for seed in range(50):
            _, cliff_x, _, _ = generate_terrain(
                seed, cliff_x_min=10, cliff_x_max=24
            )
            assert 10 <= cliff_x <= 24

    def test_no_gaps_when_budget_zero(self):
        _, _, gaps, _ = generate_terrain(42, gap_widths=[1, 2], gap_budget=0)
        assert gaps == []

    def test_gaps_placed_when_configured(self):
        _, cliff_x, gaps, _ = generate_terrain(
            42, cliff_x_min=15, cliff_x_max=24, gap_widths=[1], gap_budget=3
        )
        # With a reasonable extension, should place at least 1 gap.
        assert len(gaps) >= 1
        for start, width in gaps:
            assert 10 <= start < cliff_x
            assert start + width - 1 <= cliff_x - 1

    def test_gap_non_adjacency(self):
        """No placement should create a 1-cell solid plateau between gaps."""
        for seed in range(20):
            _, _, gaps, _ = generate_terrain(
                seed,
                cliff_x_min=15,
                cliff_x_max=24,
                gap_widths=[1],
                gap_budget=3,
            )
            if len(gaps) < 2:
                continue
            internal_runs = _solid_run_lengths(gaps, cliff_x=10**9)[:-1]
            assert all(run >= 2 for run in internal_runs), f"Seed {seed}: {gaps}"

    def test_terminal_runway_never_shrinks_to_one_cell(self):
        for seed in range(20):
            _, cliff_x, gaps, _ = generate_terrain(
                seed,
                cliff_x_min=15,
                cliff_x_max=19,
                gap_widths=[1],
                gap_budget=3,
                min_post_gap_runway=1,
            )
            if not gaps:
                continue
            terminal_runway = _solid_run_lengths(gaps, cliff_x)[-1]
            assert terminal_runway >= 2, f"Seed {seed}: cliff_x={cliff_x} gaps={gaps}"

    def test_per_width_caps(self):
        """Width-3 gaps: max 1. Width-2: max 2. Width-1: max 3."""
        for seed in range(30):
            _, _, gaps, _ = generate_terrain(
                seed,
                cliff_x_min=15,
                cliff_x_max=24,
                gap_widths=[1, 2, 3],
                gap_budget=10,  # High budget to test caps.
            )
            counts = {}
            for _, w in gaps:
                counts[w] = counts.get(w, 0) + 1
            assert counts.get(3, 0) <= 1
            assert counts.get(2, 0) <= 2
            assert counts.get(1, 0) <= 3

    def test_exact_gap_widths_respected(self):
        for seed in range(20):
            _, cliff_x, gaps, _ = generate_terrain(
                seed,
                cliff_x_min=15,
                cliff_x_max=19,
                gap_widths=[1, 2],
                gap_budget=2,
                exact_gap_widths=[1, 2],
                min_gap_separation=1,
                min_post_gap_runway=1,
            )
            assert sorted(width for _, width in gaps) == [1, 2]
            assert len(gaps) == 2
            for start, width in gaps:
                assert 10 <= start < cliff_x
                assert start + width - 1 <= cliff_x - 1

    def test_exact_gap_layout_preserves_terminal_runway(self):
        for seed in range(20):
            grid, cliff_x, gaps, _ = generate_terrain(
                seed,
                cliff_x_min=15,
                cliff_x_max=19,
                exact_gap_widths=[1, 2],
                min_gap_separation=1,
                min_post_gap_runway=1,
            )
            run_lengths = _solid_run_lengths(gaps, cliff_x)
            assert all(run >= 2 for run in run_lengths)
            last_gap_end = max(start + width - 1 for start, width in gaps)
            assert last_gap_end + 2 <= cliff_x
            assert grid[3, last_gap_end + 1] == FIXED
            assert grid[3, last_gap_end + 2] == FIXED
            assert grid[3, cliff_x] == FIXED

    def test_minimum_required_cliff_x_for_exact_gap_layout(self):
        assert minimum_required_cliff_x(
            base_width=10,
            gap_widths=[1, 2],
            min_gap_separation=1,
            min_post_gap_runway=1,
        ) == 16

    def test_exact_gap_layout_rejects_infeasible_cliff_range(self):
        with pytest.raises(ValueError, match="cannot realize requested hazards"):
            generate_terrain(
                42,
                cliff_x_min=10,
                cliff_x_max=15,
                exact_gap_widths=[1, 2],
                min_gap_separation=1,
                min_post_gap_runway=1,
            )

    def test_bilateral_goal_cliff_uses_centered_plateau_and_goal_metadata(self):
        grid, cliff_x, gaps, metadata = generate_terrain(
            42,
            grid_h=10,
            grid_w=30,
            platform_y=3,
            base_width=10,
            task_mode="bilateral_goal_cliff",
            center_plateau_width=10,
            symmetric_distance_min=0,
            symmetric_distance_max=5,
            goal_side="right",
        )

        assert gaps == []
        assert metadata["task_mode"] == "bilateral_goal_cliff"
        assert metadata["goal_side"] == "right"
        assert 19 <= cliff_x <= 24
        assert metadata["goal_block_y"] == 4
        assert grid[4, int(metadata["goal_block_x"])] == FIXED

    def test_bilateral_goal_cliff_accepts_explicit_distance_pairs(self):
        grid, cliff_x, gaps, metadata = generate_terrain(
            42,
            grid_h=10,
            grid_w=30,
            platform_y=3,
            base_width=10,
            task_mode="bilateral_goal_cliff",
            center_plateau_width=10,
            goal_side="left",
            distance_pairs=[{"goal_distance": 4, "cliff_distance": 2}],
            cliff_probe_gap_width=1,
            cliff_probe_ledge_width=3,
        )

        assert gaps == []
        assert cliff_x == int(metadata["platform_end_x"])
        assert metadata["goal_distance"] == 4
        assert metadata["cliff_distance"] == 2
        assert metadata["goal_side"] == "left"
        assert metadata["cliff_probe_gap_width"] == 1
        assert metadata["cliff_probe_ledge_width"] == 3
        assert grid[4, int(metadata["goal_block_x"])] == FIXED

    def test_bilateral_goal_cliff_probe_gap_adds_cliff_side_gap_and_ledge(self):
        grid, cliff_x, gaps, metadata = generate_terrain(
            42,
            grid_h=10,
            grid_w=30,
            platform_y=3,
            base_width=10,
            task_mode="bilateral_goal_cliff",
            center_plateau_width=10,
            symmetric_distance_min=2,
            symmetric_distance_max=2,
            goal_side="right",
            cliff_probe_gap_width=1,
            cliff_probe_ledge_width=3,
        )

        assert gaps == []
        assert cliff_x == 21
        assert metadata["goal_side"] == "right"
        assert metadata["cliff_side"] == "left"
        assert metadata["cliff_probe_gap_cell_x"] == 7
        assert metadata["cliff_probe_ledge_cell_x"] == 6
        assert metadata["cliff_probe_ledge_cell_y"] == 2
        assert grid[3, 8] == FIXED
        assert grid[3, 7] == EMPTY
        assert np.all(grid[2, 4:7] == FIXED)
        assert np.all(grid[3, 4:7] == EMPTY)

    def test_bilateral_goal_cliff_probe_ledge_without_gap_adds_cliff_side_ledge(self):
        grid, cliff_x, gaps, metadata = generate_terrain(
            42,
            grid_h=10,
            grid_w=30,
            platform_y=3,
            base_width=10,
            task_mode="bilateral_goal_cliff",
            center_plateau_width=10,
            symmetric_distance_min=2,
            symmetric_distance_max=2,
            goal_side="right",
            cliff_probe_gap_width=0,
            cliff_probe_ledge_width=3,
        )

        assert gaps == []
        assert cliff_x == 21
        assert metadata["goal_side"] == "right"
        assert metadata["cliff_side"] == "left"
        assert metadata["cliff_probe_gap_cell_x"] == -1
        assert metadata["cliff_probe_ledge_cell_x"] == 7
        assert metadata["cliff_probe_ledge_cell_y"] == 2
        assert grid[3, 8] == FIXED
        assert np.all(grid[2, 5:8] == FIXED)
        assert np.all(grid[3, 5:8] == EMPTY)

    def test_bilateral_goal_cliff_plateau_gap_returns_gap_metadata_and_frontier_cliff_x(self):
        grid, cliff_x, gaps, metadata = generate_terrain(
            42,
            grid_h=10,
            grid_w=30,
            platform_y=3,
            base_width=10,
            task_mode="bilateral_goal_cliff",
            center_plateau_width=10,
            goal_side="right",
            distance_pairs=[{"goal_distance": 4, "cliff_distance": 2}],
            cliff_front_gap_width=3,
            cliff_post_gap_runway=1,
        )

        assert cliff_x == 8
        assert gaps == [(5, 3)]
        assert metadata["goal_side"] == "right"
        assert metadata["cliff_side"] == "left"
        assert metadata["cliff_edge_x"] == 8
        assert metadata["cliff_front_gap_width"] == 3
        assert metadata["terminal_cliff_edge_x"] == 4
        assert grid[3, 8] == FIXED
        assert np.all(grid[3, 5:8] == EMPTY)
        assert grid[3, 4] == FIXED

    def test_zone_mode_gap_free_phase_keeps_fixed_prefix_and_empty_suffix(self):
        for seed in range(20):
            grid, cliff_x, gaps, _ = generate_terrain(
                seed,
                gap_widths=[],
                gap_budget=0,
                exact_gap_widths=[],
                single_gap_width=0,
                **PHASE_1_ZONE_MODE_KWARGS,
            )

            assert gaps == []
            assert 10 <= cliff_x <= 19
            assert np.all(grid[3, :10] == FIXED)
            assert np.all(grid[3, cliff_x + 1 :] == EMPTY)

    def test_zone_mode_single_gap_preserves_fixed_prefix_and_post_gap_runway(self):
        for seed in range(20):
            grid, cliff_x, gaps, _ = generate_terrain(
                seed,
                gap_widths=[2],
                gap_budget=1,
                exact_gap_widths=[2],
                single_gap_width=2,
                **PHASE_2_ZONE_MODE_KWARGS,
            )

            assert len(gaps) == 1
            gap_start, gap_width = gaps[0]
            gap_end = gap_start + gap_width - 1
            assert gap_width == 2
            assert 10 <= gap_start <= 19
            assert gap_end + 2 <= cliff_x <= 24
            assert np.all(grid[3, :10] == FIXED)
            assert np.all(grid[3, gap_start : gap_start + gap_width] == EMPTY)
            assert np.all(grid[3, gap_end + 1 : cliff_x + 1] == FIXED)

    def test_zone_mode_two_wide_layout_enumeration_respects_gap_start_cap(self):
        layouts = _enumerate_zone_single_gap_layouts(
            10,
            24,
            gap_width=2,
            min_post_gap_runway=2,
            gap_start_min=10,
            gap_start_max=19,
        )

        gap_starts = [gaps[0][0] for _, gaps in layouts]
        assert len(layouts) == 75
        assert sorted(set(gap_starts)) == list(range(10, 20))
        assert gap_starts.count(10) == 12
        assert gap_starts.count(19) == 3

    def test_zone_mode_three_wide_layout_enumeration_respects_gap_start_cap(self):
        layouts = _enumerate_zone_single_gap_layouts(
            10,
            24,
            gap_width=3,
            min_post_gap_runway=2,
            gap_start_min=10,
            gap_start_max=19,
        )

        gap_starts = [gaps[0][0] for _, gaps in layouts]
        assert len(layouts) == 65
        assert sorted(set(gap_starts)) == list(range(10, 20))
        assert gap_starts.count(10) == 11
        assert gap_starts.count(19) == 2


class TestGenerateGenerationTerrains:
    def test_count(self):
        results = generate_generation_terrains(42, n_episodes=4)
        assert len(results) == 4

    def test_determinism(self):
        r1 = generate_generation_terrains(42, n_episodes=4)
        r2 = generate_generation_terrains(42, n_episodes=4)
        for (t1, c1, g1, m1), (t2, c2, g2, m2) in zip(r1, r2):
            assert np.array_equal(t1, t2)
            assert c1 == c2
            assert g1 == g2
            assert m1 == m2

    def test_different_gen_seeds(self):
        r1 = generate_generation_terrains(42, n_episodes=4)
        r2 = generate_generation_terrains(99, n_episodes=4)
        any_different = any(
            not np.array_equal(t1, t2)
            for (t1, _, _, _), (t2, _, _, _) in zip(r1, r2)
        )
        assert any_different

    def test_zone_mode_threads_structured_gap_config(self):
        results = generate_generation_terrains(
            42,
            n_episodes=4,
            gap_widths=[2],
            gap_budget=1,
            exact_gap_widths=[2],
            single_gap_width=2,
            gap_start_bands=[[10, 14], [15, 19]],
            **{
                key: value
                for key, value in PHASE_2_ZONE_MODE_KWARGS.items()
                if key not in {"gap_start_min", "gap_start_max"}
            },
        )

        assert len(results) == 4
        for episode_idx, (grid, cliff_x, gaps, metadata) in enumerate(results):
            assert grid.shape == (10, 30)
            assert 10 <= cliff_x <= 24
            assert len(gaps) == 1
            assert metadata["task_mode"] == "legacy_cliff"
            if episode_idx % 2 == 0:
                assert 10 <= gaps[0][0] <= 14
            else:
                assert 15 <= gaps[0][0] <= 19
            assert gaps[0][1] == 2


class TestMakeRobotBody:
    def test_shape(self):
        body = make_robot_body(7, 2)
        assert body.shape == (2, 7)

    def test_all_contractile(self):
        body = make_robot_body()
        assert np.all(body == CONTRACTILE)

    def test_comb_shape_uses_sparse_bottom_row(self):
        body = make_robot_body(7, 2, shape="comb")
        expected = np.array(
            [
                [CONTRACTILE, EMPTY, CONTRACTILE, EMPTY, CONTRACTILE, EMPTY, CONTRACTILE],
                [CONTRACTILE, CONTRACTILE, CONTRACTILE, CONTRACTILE, CONTRACTILE, CONTRACTILE, CONTRACTILE],
            ],
            dtype=np.int32,
        )
        assert np.array_equal(body, expected)

    def test_comb_requires_7x2_bounds(self):
        with pytest.raises(ValueError, match="requires terrain.robot_w=7 and terrain.robot_h=2"):
            make_robot_body(7, 3, shape="comb")

    def test_comb_7x3_shape_uses_sparse_bottom_row_and_two_full_upper_rows(self):
        body = make_robot_body(7, 3, shape="comb_7x3")
        expected = np.array(
            [
                [CONTRACTILE, EMPTY, CONTRACTILE, EMPTY, CONTRACTILE, EMPTY, CONTRACTILE],
                [CONTRACTILE, CONTRACTILE, CONTRACTILE, CONTRACTILE, CONTRACTILE, CONTRACTILE, CONTRACTILE],
                [CONTRACTILE, CONTRACTILE, CONTRACTILE, CONTRACTILE, CONTRACTILE, CONTRACTILE, CONTRACTILE],
            ],
            dtype=np.int32,
        )
        assert np.array_equal(body, expected)

    def test_comb_7x3_requires_7x3_bounds(self):
        with pytest.raises(ValueError, match="requires terrain.robot_w=7 and terrain.robot_h=3"):
            make_robot_body(7, 2, shape="comb_7x3")
