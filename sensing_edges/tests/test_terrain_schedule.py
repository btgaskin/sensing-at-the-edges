"""Tests for frozen terrain schedule generation and loading."""

import numpy as np
import pytest

from jax_evogym.constants import FIXED

from sensing_edges.terrain_schedule import (
    TerrainEntry,
    TerrainSchedule,
    candidate_schedule_bundle_paths,
    compatible_schedule_variants,
    generate_schedule,
    load_schedule,
    resolve_schedule_bundle_path,
    schedule_variant_matches,
    save_schedule,
    schedule_bundle_name,
    schedule_bundle_path,
    schedule_train_terrains_for_gen,
    validate_bilateral_schedule_semantics,
)

TERRAIN_CFG = {
    "grid_h": 10,
    "grid_w": 35,
    "platform_y": 3,
    "base_width": 10,
    "cliff_x_min": 10,
    "cliff_x_max": 24,
    "gap_widths": None,
    "gap_budget": 0,
}

STRATIFIED_CLIFF_CFG = {
    "variant": "cliff_standard_30",
    "grid_h": 10,
    "grid_w": 30,
    "platform_y": 3,
    "base_width": 10,
    "spawn_x": 1,
    "spawn_y": 4,
    "robot_w": 7,
    "robot_h": 2,
    "cliff_x_min": 10,
    "cliff_x_max": 19,
    "sampling_mode": "stratified_cliff_x",
    "gap_widths": [],
    "gap_budget": 0,
}

BILATERAL_CFG = {
    "variant": "bilateral_goal_cliff_30",
    "task_mode": "bilateral_goal_cliff",
    "grid_h": 10,
    "grid_w": 30,
    "platform_y": 3,
    "base_width": 10,
    "center_plateau_width": 10,
    "spawn_x": 12,
    "spawn_y": 4,
    "robot_w": 6,
    "robot_h": 3,
    "cliff_x_min": 19,
    "cliff_x_max": 24,
    "symmetric_distance_min": 1,
    "symmetric_distance_max": 6,
    "sampling_mode": "canonical_bilateral_cycle",
    "gap_widths": [],
    "gap_budget": 0,
}

PROBE_GAP_BILATERAL_CFG = {
    **BILATERAL_CFG,
    "variant": "bilateral_goal_cliff_probe_gap_30",
    "cliff_probe_gap_width": 1,
    "cliff_probe_ledge_width": 3,
}

ASYMMETRIC_BILATERAL_CFG = {
    **BILATERAL_CFG,
    "variant": "bilateral_goal_cliff_asymmetric_30",
    "correct_side_distance_scale": 1.5,
    "incorrect_side_distance_scale": 1.0,
    "cliff_probe_gap_width": 0,
    "cliff_probe_ledge_width": 0,
}

ASYMMETRIC_FIXED_PAIR_CFG = {
    **BILATERAL_CFG,
    "variant": "bilateral_goal_cliff_asymmetric_pairs_30",
    "distance_pairs": [
        {"goal_distance": 2, "cliff_distance": 1},
        {"goal_distance": 4, "cliff_distance": 2},
        {"goal_distance": 6, "cliff_distance": 3},
    ],
    "cliff_probe_gap_width": 0,
    "cliff_probe_ledge_width": 0,
}

ASYMMETRIC_FIXED_PAIR_LEDGE_CFG = {
    **ASYMMETRIC_FIXED_PAIR_CFG,
    "variant": "bilateral_goal_cliff_asymmetric_pairs_probe_ledge_30",
    "cliff_probe_gap_width": 0,
    "cliff_probe_ledge_width": 3,
}

ASYMMETRIC_FIXED_PAIR_LEDGE_4_CFG = {
    **ASYMMETRIC_FIXED_PAIR_CFG,
    "variant": "bilateral_goal_cliff_asymmetric_pairs_probe_ledge_4_30",
    "cliff_probe_gap_width": 0,
    "cliff_probe_ledge_width": 4,
}

ASYMMETRIC_FIXED_PAIR_8X2_CFG = {
    **{
        **ASYMMETRIC_FIXED_PAIR_CFG,
        "distance_pairs": [
            {"goal_distance": 2, "cliff_distance": 1},
            {"goal_distance": 4, "cliff_distance": 2},
            {"goal_distance": 6, "cliff_distance": 3},
        ],
    },
    "variant": "bilateral_goal_cliff_asymmetric_pairs_8x2_30",
    "base_width": 12,
    "center_plateau_width": 12,
    "spawn_x": 11,
    "robot_w": 8,
}

ASYMMETRIC_FIXED_PAIR_7X2_GRID30_CFG = {
    **ASYMMETRIC_FIXED_PAIR_8X2_CFG,
    "variant": "bilateral_goal_cliff_asymmetric_pairs_7x2_grid30",
    "robot_w": 7,
}

ASYMMETRIC_FIXED_PAIR_LEDGE_8X2_CFG = {
    **ASYMMETRIC_FIXED_PAIR_8X2_CFG,
    "variant": "bilateral_goal_cliff_asymmetric_pairs_probe_ledge_8x2_30",
    "cliff_probe_gap_width": 0,
    "cliff_probe_ledge_width": 3,
}

ASYMMETRIC_FIXED_PAIR_PLATEAU_GAP_W1_CFG = {
    **ASYMMETRIC_FIXED_PAIR_CFG,
    "variant": "bilateral_goal_cliff_asymmetric_pairs_plateau_gap_w1_30",
    "cliff_front_gap_width": 1,
    "cliff_post_gap_runway": 1,
    "cliff_probe_gap_width": 0,
    "cliff_probe_ledge_width": 0,
}

ASYMMETRIC_FIXED_PAIR_PLATEAU_GAP_W3_8X2_CFG = {
    **ASYMMETRIC_FIXED_PAIR_8X2_CFG,
    "variant": "bilateral_goal_cliff_asymmetric_pairs_plateau_gap_w3_8x2_30",
    "cliff_front_gap_width": 3,
    "cliff_post_gap_runway": 1,
    "cliff_probe_gap_width": 0,
    "cliff_probe_ledge_width": 0,
}

CURRICULUM_ASYMMETRIC_FIXED_PAIR_GAP_CFG = {
    **{
        **ASYMMETRIC_FIXED_PAIR_CFG,
        "distance_pairs": [
            {"goal_distance": 2, "cliff_distance": 1},
            {"goal_distance": 4, "cliff_distance": 2},
            {"goal_distance": 6, "cliff_distance": 3},
        ],
    },
    "variant": "curriculum_bilateral_goal_cliff_asymmetric_pairs_gap_then_nogap_30",
    "curriculum": True,
    "curriculum_phases": [
        {"cliff_probe_gap_width": 1, "cliff_probe_ledge_width": 3},
        {"cliff_probe_gap_width": 0, "cliff_probe_ledge_width": 0},
    ],
}

CURRICULUM_ASYMMETRIC_FIXED_PAIR_PLATEAU_GAP_CFG = {
    **ASYMMETRIC_FIXED_PAIR_CFG,
    "variant": "curriculum_bilateral_goal_cliff_asymmetric_pairs_plateau_gap_mix123_30",
    "curriculum": True,
    "cliff_front_gap_width": 0,
    "cliff_post_gap_runway": 1,
    "curriculum_phase_switch_gens": [50, 100],
    "curriculum_phases": [
        {"cliff_front_gap_widths": [1]},
        {"cliff_front_gap_widths": [1, 2]},
        {"cliff_front_gap_widths": [1, 2, 3]},
    ],
}

CURRICULUM_ASYMMETRIC_FIXED_PAIR_PLATEAU_GAP_8X2_CFG = {
    **ASYMMETRIC_FIXED_PAIR_8X2_CFG,
    "variant": "curriculum_bilateral_goal_cliff_asymmetric_pairs_plateau_gap_mix123_8x2_30",
    "curriculum": True,
    "cliff_front_gap_width": 0,
    "cliff_post_gap_runway": 1,
    "curriculum_phase_switch_gens": [50, 100],
    "curriculum_phases": [
        {"cliff_front_gap_widths": [1]},
        {"cliff_front_gap_widths": [1, 2]},
        {"cliff_front_gap_widths": [1, 2, 3]},
    ],
}

CURRICULUM_ASYMMETRIC_FIXED_PAIR_GAP_8X2_CFG = {
    **{
        **ASYMMETRIC_FIXED_PAIR_8X2_CFG,
        "distance_pairs": [
            {"goal_distance": 2, "cliff_distance": 1},
            {"goal_distance": 4, "cliff_distance": 2},
            {"goal_distance": 6, "cliff_distance": 3},
        ],
    },
    "variant": "curriculum_bilateral_goal_cliff_asymmetric_pairs_gap_then_nogap_8x2_30",
    "curriculum": True,
    "curriculum_phases": [
        {"cliff_probe_gap_width": 1, "cliff_probe_ledge_width": 3},
        {"cliff_probe_gap_width": 0, "cliff_probe_ledge_width": 0},
    ],
}

CURRICULUM_BILATERAL_PROBE_GAP_CFG = {
    **PROBE_GAP_BILATERAL_CFG,
    "variant": "curriculum_bilateral_goal_cliff_probe_gap_30",
    "curriculum_phase_switch_gens": [50, 100],
    "curriculum": True,
    "curriculum_phases": [
        {"symmetric_distance_min": 1, "symmetric_distance_max": 2},
        {"symmetric_distance_min": 1, "symmetric_distance_max": 4},
        {"symmetric_distance_min": 1, "symmetric_distance_max": 6},
    ],
}

CURRICULUM_CFG = {
    "variant": "curriculum_zone10to19_extend24_gap2_30",
    "curriculum": True,
    "grid_h": 10,
    "grid_w": 30,
    "platform_y": 3,
    "base_width": 10,
    "spawn_x": 1,
    "spawn_y": 4,
    "robot_w": 7,
    "robot_h": 2,
    "hazard_mode": "zone_single_gap",
    "spawn_plateau_width": 10,
    "approach_width": 0,
    "randomized_zone_width": 10,
    "remainder_width": 10,
    "cliff_x_min": 10,
    "cliff_x_max": 19,
    "min_gap_separation": 2,
    "min_post_gap_runway": 2,
    "layout_sampling": "uniform_legal_layouts",
    "phase_1": {
        "gap_widths": [],
        "gap_budget": 0,
        "exact_gap_widths": [],
        "single_gap_width": 0,
    },
    "phase_2": {
        "gap_widths": [2],
        "gap_budget": 1,
        "exact_gap_widths": [2],
        "single_gap_width": 2,
        "gap_start_bands": [[10, 14], [15, 19]],
        "randomized_zone_width": 15,
        "remainder_width": 5,
        "cliff_x_min": 10,
        "cliff_x_max": 24,
    },
}


@pytest.fixture
def schedule():
    return generate_schedule(
        seed=42, n_gens=5, n_episodes=4, n_val=3, n_test=4, terrain_cfg=TERRAIN_CFG
    )


class TestGenerateSchedule:
    def test_schedule_bundle_name_includes_counts(self):
        assert schedule_bundle_name(
            seed=42,
            variant="bilateral_goal_cliff_probe_gap_30",
            n_gens=50,
            n_episodes=5,
            n_val=12,
            n_test=12,
        ) == "seed_42_bilateral_goal_cliff_probe_gap_30_g50_ep5_val12_test12"

    def test_resolve_schedule_bundle_path_prefers_explicit_then_legacy(self, tmp_path):
        preferred = schedule_bundle_path(
            tmp_path,
            seed=42,
            variant="bilateral_goal_cliff_probe_gap_30",
            n_gens=50,
            n_episodes=5,
            n_val=12,
            n_test=12,
        )
        legacy = schedule_bundle_path(
            tmp_path,
            seed=42,
            variant="bilateral_goal_cliff_probe_gap_30",
        )

        assert candidate_schedule_bundle_paths(
            tmp_path,
            seed=42,
            variant="bilateral_goal_cliff_probe_gap_30",
            n_gens=50,
            n_episodes=5,
            n_val=12,
            n_test=12,
        ) == [preferred, legacy]

        legacy.mkdir(parents=True)
        assert resolve_schedule_bundle_path(
            tmp_path,
            seed=42,
            variant="bilateral_goal_cliff_probe_gap_30",
            n_gens=50,
            n_episodes=5,
            n_val=12,
            n_test=12,
        ) == legacy

        preferred.mkdir(parents=True)
        assert resolve_schedule_bundle_path(
            tmp_path,
            seed=42,
            variant="bilateral_goal_cliff_probe_gap_30",
            n_gens=50,
            n_episodes=5,
            n_val=12,
            n_test=12,
        ) == preferred

    def test_resolve_schedule_bundle_path_accepts_legacy_alias_for_7x2_grid30(self, tmp_path):
        preferred = schedule_bundle_path(
            tmp_path,
            seed=42,
            variant="bilateral_goal_cliff_asymmetric_pairs_7x2_grid30",
            n_gens=150,
            n_episodes=6,
            n_val=6,
            n_test=6,
        )
        legacy = schedule_bundle_path(
            tmp_path,
            seed=42,
            variant="bilateral_goal_cliff_asymmetric_pairs_8x2_30",
            n_gens=150,
            n_episodes=6,
            n_val=6,
            n_test=6,
        )
        legacy.mkdir(parents=True)

        assert compatible_schedule_variants("bilateral_goal_cliff_asymmetric_pairs_7x2_grid30") == (
            "bilateral_goal_cliff_asymmetric_pairs_7x2_grid30",
            "bilateral_goal_cliff_asymmetric_pairs_8x2_30",
        )
        assert candidate_schedule_bundle_paths(
            tmp_path,
            seed=42,
            variant="bilateral_goal_cliff_asymmetric_pairs_7x2_grid30",
            n_gens=150,
            n_episodes=6,
            n_val=6,
            n_test=6,
        ) == [
            preferred,
            schedule_bundle_path(
                tmp_path,
                seed=42,
                variant="bilateral_goal_cliff_asymmetric_pairs_7x2_grid30",
            ),
            legacy,
            schedule_bundle_path(
                tmp_path,
                seed=42,
                variant="bilateral_goal_cliff_asymmetric_pairs_8x2_30",
            ),
        ]
        assert resolve_schedule_bundle_path(
            tmp_path,
            seed=42,
            variant="bilateral_goal_cliff_asymmetric_pairs_7x2_grid30",
            n_gens=150,
            n_episodes=6,
            n_val=6,
            n_test=6,
        ) == legacy
        assert schedule_variant_matches(
            "bilateral_goal_cliff_asymmetric_pairs_7x2_grid30",
            "bilateral_goal_cliff_asymmetric_pairs_8x2_30",
        )

    def test_shapes(self, schedule):
        assert len(schedule.train) == 5
        assert all(len(gen) == 4 for gen in schedule.train)
        assert len(schedule.val) == 3
        assert len(schedule.test) == 4

    def test_terrain_entry_contents(self, schedule):
        e = schedule.train[0][0]
        assert isinstance(e, TerrainEntry)
        assert e.grid.shape == (10, 35)
        assert 10 <= e.cliff_x <= 24

    def test_disjoint_rng_streams(self):
        """Train, val, and test pools use independent RNG streams.

        Verify by checking that changing n_val doesn't affect train terrains.
        """
        s_a = generate_schedule(seed=42, n_gens=3, n_episodes=2, n_val=2, n_test=2, terrain_cfg=TERRAIN_CFG)
        s_b = generate_schedule(seed=42, n_gens=3, n_episodes=2, n_val=5, n_test=2, terrain_cfg=TERRAIN_CFG)
        # Train should be identical despite different n_val.
        for gen_a, gen_b in zip(s_a.train, s_b.train):
            for a, b in zip(gen_a, gen_b):
                assert np.array_equal(a.grid, b.grid)
                assert a.cliff_x == b.cliff_x

    def test_determinism(self):
        s1 = generate_schedule(seed=42, n_gens=3, n_episodes=2, n_val=2, n_test=2, terrain_cfg=TERRAIN_CFG)
        s2 = generate_schedule(seed=42, n_gens=3, n_episodes=2, n_val=2, n_test=2, terrain_cfg=TERRAIN_CFG)
        for gen_a, gen_b in zip(s1.train, s2.train):
            for a, b in zip(gen_a, gen_b):
                assert np.array_equal(a.grid, b.grid)
                assert a.cliff_x == b.cliff_x

    def test_different_seeds_differ(self):
        s1 = generate_schedule(seed=42, n_gens=3, n_episodes=2, n_val=2, n_test=2, terrain_cfg=TERRAIN_CFG)
        s2 = generate_schedule(seed=99, n_gens=3, n_episodes=2, n_val=2, n_test=2, terrain_cfg=TERRAIN_CFG)
        any_different = any(
            not np.array_equal(a.grid, b.grid)
            for gen_a, gen_b in zip(s1.train, s2.train)
            for a, b in zip(gen_a, gen_b)
        )
        assert any_different

    def test_curriculum_schedule_transitions_halfway(self):
        schedule = generate_schedule(
            seed=42,
            n_gens=6,
            n_episodes=2,
            n_val=2,
            n_test=2,
            terrain_cfg=CURRICULUM_CFG,
        )

        assert schedule.curriculum is True
        assert schedule.transition_gen == 3
        assert schedule.phase_1_cfg == CURRICULUM_CFG["phase_1"]
        assert schedule.phase_2_cfg == CURRICULUM_CFG["phase_2"]

        for gen_idx, generation in enumerate(schedule.train):
            for ep_idx, entry in enumerate(generation):
                if gen_idx < 3:
                    assert 10 <= entry.cliff_x <= 19
                    assert entry.gaps == []
                else:
                    assert 10 <= entry.cliff_x <= 24
                    assert len(entry.gaps) == 1
                    if ep_idx % 2 == 0:
                        assert 10 <= entry.gaps[0][0] <= 14
                    else:
                        assert 15 <= entry.gaps[0][0] <= 19
                    assert entry.gaps[0][1] == 2

    def test_curriculum_val_and_test_use_final_task(self):
        schedule = generate_schedule(
            seed=42,
            n_gens=6,
            n_episodes=2,
            n_val=3,
            n_test=4,
            terrain_cfg=CURRICULUM_CFG,
        )

        for split in (schedule.val, schedule.test):
            for idx, entry in enumerate(split):
                assert 10 <= entry.cliff_x <= 24
                assert len(entry.gaps) == 1
                if idx % 2 == 0:
                    assert 10 <= entry.gaps[0][0] <= 14
                else:
                    assert 15 <= entry.gaps[0][0] <= 19
                assert entry.gaps[0][1] == 2

    def test_stratified_cliff_sampling_balances_each_generation(self):
        schedule = generate_schedule(
            seed=42,
            n_gens=5,
            n_episodes=4,
            n_val=4,
            n_test=4,
            terrain_cfg=STRATIFIED_CLIFF_CFG,
        )

        expected_strata = [
            {10, 11, 12},
            {13, 14, 15},
            {16, 17},
            {18, 19},
        ]

        for generation in schedule.train:
            cliffs = [entry.cliff_x for entry in generation]
            assert len(cliffs) == 4
            for stratum in expected_strata:
                assert sum(cliff in stratum for cliff in cliffs) == 1

        for split in (schedule.val, schedule.test):
            cliffs = [entry.cliff_x for entry in split]
            assert len(cliffs) == 4
            for stratum in expected_strata:
                assert sum(cliff in stratum for cliff in cliffs) == 1

    def test_stratified_cliff_sampling_rejects_gap_terrains(self):
        cfg = dict(STRATIFIED_CLIFF_CFG)
        cfg["gap_widths"] = [2]
        cfg["gap_budget"] = 1

        with pytest.raises(ValueError, match="legacy cliff-only terrains without gaps"):
            generate_schedule(
                seed=42,
                n_gens=2,
                n_episodes=4,
                n_val=2,
                n_test=2,
                terrain_cfg=cfg,
            )

    def test_bilateral_schedule_cycles_all_12_conditions_every_three_generations(self):
        schedule = generate_schedule(
            seed=42,
            n_gens=6,
            n_episodes=4,
            n_val=12,
            n_test=12,
            terrain_cfg=BILATERAL_CFG,
        )

        for cycle_start in (0, 3):
            seen = {
                (
                    int(entry.metadata["goal_distance"]),
                    str(entry.metadata["goal_side"]),
                )
                for generation in schedule.train[cycle_start: cycle_start + 3]
                for entry in generation
            }
            assert seen == {
                (distance, side)
                for distance in range(1, 7)
                for side in ("left", "right")
            }

        for generation in schedule.train:
            assert [str(entry.metadata["goal_side"]) for entry in generation] == [
                "left",
                "right",
                "left",
                "right",
            ]
            left_pairs = [
                (
                    int(entry.metadata["goal_distance"]),
                    int(entry.metadata["cliff_distance"]),
                )
                for entry in generation[::2]
            ]
            right_pairs = [
                (
                    int(entry.metadata["goal_distance"]),
                    int(entry.metadata["cliff_distance"]),
                )
                for entry in generation[1::2]
            ]
            assert left_pairs == right_pairs

    def test_bilateral_schedule_rejects_odd_episode_counts(self):
        with pytest.raises(ValueError, match="even train.n_episodes"):
            generate_schedule(
                seed=42,
                n_gens=12,
                n_episodes=5,
                n_val=12,
                n_test=12,
                terrain_cfg=BILATERAL_CFG,
            )

    def test_bilateral_probe_gap_schedule_preserves_probe_metadata(self):
        schedule = generate_schedule(
            seed=42,
            n_gens=3,
            n_episodes=4,
            n_val=12,
            n_test=12,
            terrain_cfg=PROBE_GAP_BILATERAL_CFG,
        )

        sample = schedule.train[0][0]
        assert sample.metadata["cliff_probe_gap_width"] == 1
        assert sample.metadata["cliff_probe_ledge_width"] == 3
        assert int(sample.metadata["cliff_probe_gap_cell_x"]) >= 0
        assert int(sample.metadata["cliff_probe_ledge_cell_x"]) >= 0
        assert int(sample.metadata["cliff_probe_ledge_cell_y"]) == 2

        for split in (schedule.val, schedule.test):
            seen = {
                (
                    int(entry.metadata["goal_distance"]),
                    str(entry.metadata["goal_side"]),
                )
                for entry in split
            }
            assert seen == {
                (distance, side)
                for distance in range(1, 7)
                for side in ("left", "right")
            }

    def test_asymmetric_bilateral_schedule_derives_goal_and_cliff_distances(self):
        schedule = generate_schedule(
            seed=42,
            n_gens=3,
            n_episodes=4,
            n_val=12,
            n_test=12,
            terrain_cfg=ASYMMETRIC_BILATERAL_CFG,
        )

        seen = {
            (int(entry.metadata["cliff_distance"]), str(entry.metadata["goal_side"]))
            for generation in schedule.train
            for entry in generation
        }
        assert seen == {
            (distance, side)
            for distance in range(1, 7)
            for side in ("left", "right")
        }
        for split in (schedule.train[0], schedule.val, schedule.test):
            for entry in split:
                cliff_distance = int(entry.metadata["cliff_distance"])
                goal_distance = int(entry.metadata["goal_distance"])
                assert cliff_distance >= 1
                assert goal_distance == int(np.ceil(1.5 * cliff_distance))
                assert entry.metadata["cliff_probe_gap_width"] == 0
                assert entry.metadata["cliff_probe_ledge_width"] == 0

    def test_fixed_pair_bilateral_schedule_uses_all_six_layouts_each_generation(self):
        schedule = generate_schedule(
            seed=42,
            n_gens=4,
            n_episodes=6,
            n_val=6,
            n_test=6,
            terrain_cfg=ASYMMETRIC_FIXED_PAIR_CFG,
        )

        expected = {
            (2, 1, "left"),
            (2, 1, "right"),
            (4, 2, "left"),
            (4, 2, "right"),
            (6, 3, "left"),
            (6, 3, "right"),
        }
        for generation in schedule.train:
            seen = {
                (
                    int(entry.metadata["goal_distance"]),
                    int(entry.metadata["cliff_distance"]),
                    str(entry.metadata["goal_side"]),
                )
                for entry in generation
            }
            assert seen == expected
            for entry in generation:
                assert entry.metadata["cliff_probe_gap_width"] == 0
                assert entry.metadata["cliff_probe_ledge_width"] == 0

        for split in (schedule.val, schedule.test):
            seen = {
                (
                    int(entry.metadata["goal_distance"]),
                    int(entry.metadata["cliff_distance"]),
                    str(entry.metadata["goal_side"]),
                )
                for entry in split
            }
            assert seen == expected

    def test_fixed_pair_bilateral_8x2_schedule_preserves_pairs_and_plateau_geometry(self):
        schedule = generate_schedule(
            seed=42,
            n_gens=3,
            n_episodes=6,
            n_val=6,
            n_test=6,
            terrain_cfg=ASYMMETRIC_FIXED_PAIR_8X2_CFG,
        )

        assert schedule.terrain_cfg["variant"] == "bilateral_goal_cliff_asymmetric_pairs_8x2_30"
        assert schedule.terrain_cfg["center_plateau_width"] == 12
        assert schedule.terrain_cfg["robot_w"] == 8
        expected = {
            (2, 1, "left"),
            (2, 1, "right"),
            (4, 2, "left"),
            (4, 2, "right"),
            (6, 3, "left"),
            (6, 3, "right"),
        }

        for generation in schedule.train:
            seen = {
                (
                    int(entry.metadata["goal_distance"]),
                    int(entry.metadata["cliff_distance"]),
                    str(entry.metadata["goal_side"]),
                )
                for entry in generation
            }
            assert seen == expected

    def test_fixed_pair_bilateral_7x2_grid30_schedule_preserves_pairs_and_plateau_geometry(self):
        schedule = generate_schedule(
            seed=42,
            n_gens=3,
            n_episodes=6,
            n_val=6,
            n_test=6,
            terrain_cfg=ASYMMETRIC_FIXED_PAIR_7X2_GRID30_CFG,
        )

        assert schedule.terrain_cfg["variant"] == "bilateral_goal_cliff_asymmetric_pairs_7x2_grid30"
        assert schedule.terrain_cfg["center_plateau_width"] == 12
        assert schedule.terrain_cfg["robot_w"] == 7
        expected = {
            (2, 1, "left"),
            (2, 1, "right"),
            (4, 2, "left"),
            (4, 2, "right"),
            (6, 3, "left"),
            (6, 3, "right"),
        }

        for generation in schedule.train:
            seen = {
                (
                    int(entry.metadata["goal_distance"]),
                    int(entry.metadata["cliff_distance"]),
                    str(entry.metadata["goal_side"]),
                )
                for entry in generation
            }
            assert seen == expected

    def test_fixed_pair_bilateral_ledge_schedule_preserves_pairs_and_ledge_metadata(self):
        schedule = generate_schedule(
            seed=137,
            n_gens=150,
            n_episodes=6,
            n_val=6,
            n_test=6,
            terrain_cfg=ASYMMETRIC_FIXED_PAIR_LEDGE_CFG,
        )

        assert schedule.terrain_cfg["variant"] == "bilateral_goal_cliff_asymmetric_pairs_probe_ledge_30"
        expected = {
            (2, 1, "left"),
            (2, 1, "right"),
            (4, 2, "left"),
            (4, 2, "right"),
            (6, 3, "left"),
            (6, 3, "right"),
        }

        for generation in schedule.train:
            seen = {
                (
                    int(entry.metadata["goal_distance"]),
                    int(entry.metadata["cliff_distance"]),
                    str(entry.metadata["goal_side"]),
                )
                for entry in generation
            }
            assert seen == expected
            for entry in generation:
                assert entry.metadata["cliff_probe_gap_width"] == 0
                assert entry.metadata["cliff_probe_gap_cell_x"] == -1
                assert entry.metadata["cliff_probe_ledge_width"] == 3
                assert int(entry.metadata["cliff_probe_ledge_cell_x"]) >= 0
                assert int(entry.metadata["cliff_probe_ledge_cell_y"]) == 2

        for split in (schedule.val, schedule.test):
            seen = {
                (
                    int(entry.metadata["goal_distance"]),
                    int(entry.metadata["cliff_distance"]),
                    str(entry.metadata["goal_side"]),
                )
                for entry in split
            }
            assert seen == expected
            for entry in split:
                assert entry.metadata["cliff_probe_gap_width"] == 0
                assert entry.metadata["cliff_probe_gap_cell_x"] == -1
                assert entry.metadata["cliff_probe_ledge_width"] == 3
                assert int(entry.metadata["cliff_probe_ledge_cell_y"]) == 2

    def test_fixed_pair_bilateral_ledge4_schedule_preserves_pairs_and_ledge_metadata(self):
        schedule = generate_schedule(
            seed=137,
            n_gens=150,
            n_episodes=6,
            n_val=6,
            n_test=6,
            terrain_cfg=ASYMMETRIC_FIXED_PAIR_LEDGE_4_CFG,
        )

        assert schedule.terrain_cfg["variant"] == "bilateral_goal_cliff_asymmetric_pairs_probe_ledge_4_30"
        expected = {
            (2, 1, "left"),
            (2, 1, "right"),
            (4, 2, "left"),
            (4, 2, "right"),
            (6, 3, "left"),
            (6, 3, "right"),
        }

        for generation in schedule.train:
            seen = {
                (
                    int(entry.metadata["goal_distance"]),
                    int(entry.metadata["cliff_distance"]),
                    str(entry.metadata["goal_side"]),
                )
                for entry in generation
            }
            assert seen == expected
            for entry in generation:
                assert entry.metadata["cliff_probe_gap_width"] == 0
                assert entry.metadata["cliff_probe_gap_cell_x"] == -1
                assert entry.metadata["cliff_probe_ledge_width"] == 4
                assert int(entry.metadata["cliff_probe_ledge_cell_x"]) >= 0
                assert int(entry.metadata["cliff_probe_ledge_cell_y"]) == 2

    def test_fixed_pair_bilateral_ledge_8x2_schedule_preserves_pairs_geometry_and_ledge_metadata(self):
        schedule = generate_schedule(
            seed=137,
            n_gens=150,
            n_episodes=6,
            n_val=6,
            n_test=6,
            terrain_cfg=ASYMMETRIC_FIXED_PAIR_LEDGE_8X2_CFG,
        )

        assert schedule.terrain_cfg["variant"] == "bilateral_goal_cliff_asymmetric_pairs_probe_ledge_8x2_30"
        assert schedule.terrain_cfg["center_plateau_width"] == 12
        assert schedule.terrain_cfg["robot_w"] == 8
        expected = {
            (2, 1, "left"),
            (2, 1, "right"),
            (4, 2, "left"),
            (4, 2, "right"),
            (6, 3, "left"),
            (6, 3, "right"),
        }

        for generation in schedule.train:
            seen = {
                (
                    int(entry.metadata["goal_distance"]),
                    int(entry.metadata["cliff_distance"]),
                    str(entry.metadata["goal_side"]),
                )
                for entry in generation
            }
            assert seen == expected
            for entry in generation:
                assert entry.metadata["cliff_probe_gap_width"] == 0
                assert entry.metadata["cliff_probe_gap_cell_x"] == -1
                assert entry.metadata["cliff_probe_ledge_width"] == 3
                assert int(entry.metadata["cliff_probe_ledge_cell_y"]) == 2

    def test_asymmetric_fixed_pair_gap_curriculum_splits_evenly_without_explicit_switches(self):
        schedule = generate_schedule(
            seed=42,
            n_gens=100,
            n_episodes=6,
            n_val=6,
            n_test=6,
            terrain_cfg=CURRICULUM_ASYMMETRIC_FIXED_PAIR_GAP_CFG,
        )

        assert schedule.curriculum is True
        assert schedule.transition_gen == 50
        assert schedule.curriculum_phase_end_gens == [50, 100]
        assert schedule.curriculum_phase_cfgs == CURRICULUM_ASYMMETRIC_FIXED_PAIR_GAP_CFG["curriculum_phases"]

        for generation in schedule.train[:50]:
            for entry in generation:
                assert entry.metadata["cliff_probe_gap_width"] == 1
                assert entry.metadata["cliff_probe_ledge_width"] == 3
        for generation in schedule.train[50:]:
            for entry in generation:
                assert entry.metadata["cliff_probe_gap_width"] == 0
                assert entry.metadata["cliff_probe_ledge_width"] == 0

    def test_fixed_pair_bilateral_plateau_gap_schedule_preserves_pairs_and_gap_metadata(self):
        schedule = generate_schedule(
            seed=42,
            n_gens=4,
            n_episodes=6,
            n_val=6,
            n_test=6,
            terrain_cfg=ASYMMETRIC_FIXED_PAIR_PLATEAU_GAP_W1_CFG,
        )

        expected = {
            (2, 1, "left"),
            (2, 1, "right"),
            (4, 2, "left"),
            (4, 2, "right"),
            (6, 3, "left"),
            (6, 3, "right"),
        }
        for generation in schedule.train:
            seen = {
                (
                    int(entry.metadata["goal_distance"]),
                    int(entry.metadata["cliff_distance"]),
                    str(entry.metadata["goal_side"]),
                )
                for entry in generation
            }
            assert seen == expected
            for entry in generation:
                assert entry.metadata["cliff_front_gap_width"] == 1
                assert entry.metadata["cliff_post_gap_runway"] == 1
                assert entry.metadata["cliff_probe_gap_width"] == 0
                assert entry.metadata["cliff_probe_ledge_width"] == 0
                assert entry.gaps and entry.gaps[0][1] == 1

        expected = {
            (2, 1, "left"),
            (2, 1, "right"),
            (4, 2, "left"),
            (4, 2, "right"),
            (6, 3, "left"),
            (6, 3, "right"),
        }
        for split in (schedule.val, schedule.test):
            seen = {
                (
                    int(entry.metadata["goal_distance"]),
                    int(entry.metadata["cliff_distance"]),
                    str(entry.metadata["goal_side"]),
                )
                for entry in split
            }
            assert seen == expected
            for entry in split:
                assert entry.metadata["cliff_probe_gap_width"] == 0
                assert entry.metadata["cliff_probe_ledge_width"] == 0

    def test_asymmetric_fixed_pair_gap_curriculum_halves_short_runs(self):
        schedule = generate_schedule(
            seed=42,
            n_gens=60,
            n_episodes=4,
            n_val=4,
            n_test=4,
            terrain_cfg=CURRICULUM_ASYMMETRIC_FIXED_PAIR_GAP_CFG,
        )

        assert schedule.transition_gen == 30
        assert schedule.curriculum_phase_end_gens == [30, 60]

    def test_asymmetric_fixed_pair_gap_curriculum_8x2_preserves_start_margin_and_final_eval_variant(self):
        schedule = generate_schedule(
            seed=42,
            n_gens=100,
            n_episodes=6,
            n_val=6,
            n_test=6,
            terrain_cfg=CURRICULUM_ASYMMETRIC_FIXED_PAIR_GAP_8X2_CFG,
        )

        assert schedule.terrain_cfg["variant"] == "curriculum_bilateral_goal_cliff_asymmetric_pairs_gap_then_nogap_8x2_30"
        assert schedule.terrain_cfg["center_plateau_width"] == 12
        assert schedule.terrain_cfg["robot_w"] == 8
        assert schedule.transition_gen == 50
        assert schedule.curriculum_phase_end_gens == [50, 100]

        for generation in schedule.train[:50]:
            for entry in generation:
                assert entry.metadata["cliff_probe_gap_width"] == 1
                assert entry.metadata["cliff_probe_ledge_width"] == 3
        for generation in schedule.train[50:]:
            for entry in generation:
                assert entry.metadata["cliff_probe_gap_width"] == 0
                assert entry.metadata["cliff_probe_ledge_width"] == 0

        expected = {
            (2, 1, "left"),
            (2, 1, "right"),
            (4, 2, "left"),
            (4, 2, "right"),
            (6, 3, "left"),
            (6, 3, "right"),
        }
        for split in (schedule.val, schedule.test):
            seen = {
                (
                    int(entry.metadata["goal_distance"]),
                    int(entry.metadata["cliff_distance"]),
                    str(entry.metadata["goal_side"]),
                )
                for entry in split
            }
            assert seen == expected
            for entry in split:
                assert entry.metadata["cliff_probe_gap_width"] == 0
                assert entry.metadata["cliff_probe_ledge_width"] == 0

    def test_asymmetric_fixed_pair_plateau_gap_curriculum_accumulates_active_widths(self):
        schedule = generate_schedule(
            seed=42,
            n_gens=150,
            n_episodes=6,
            n_val=18,
            n_test=18,
            terrain_cfg=CURRICULUM_ASYMMETRIC_FIXED_PAIR_PLATEAU_GAP_CFG,
        )

        assert schedule.curriculum is True
        assert schedule.curriculum_phase_cfgs == CURRICULUM_ASYMMETRIC_FIXED_PAIR_PLATEAU_GAP_CFG["curriculum_phases"]
        assert schedule.curriculum_phase_end_gens == [50, 100, 150]

        phase_1_widths = {
            int(entry.metadata["cliff_front_gap_width"])
            for generation in schedule.train[:50]
            for entry in generation
        }
        phase_2_widths = {
            int(entry.metadata["cliff_front_gap_width"])
            for generation in schedule.train[50:100]
            for entry in generation
        }
        phase_3_widths = {
            int(entry.metadata["cliff_front_gap_width"])
            for generation in schedule.train[100:]
            for entry in generation
        }
        assert phase_1_widths == {1}
        assert phase_2_widths == {1, 2}
        assert phase_3_widths == {1, 2, 3}

        for generation in schedule.train:
            assert [str(entry.metadata["goal_side"]) for entry in generation] == [
                "left",
                "right",
                "left",
                "right",
                "left",
                "right",
            ]
            left_triplets = [
                (
                    int(entry.metadata["goal_distance"]),
                    int(entry.metadata["cliff_distance"]),
                    int(entry.metadata["cliff_front_gap_width"]),
                )
                for entry in generation[::2]
            ]
            right_triplets = [
                (
                    int(entry.metadata["goal_distance"]),
                    int(entry.metadata["cliff_distance"]),
                    int(entry.metadata["cliff_front_gap_width"]),
                )
                for entry in generation[1::2]
            ]
            assert left_triplets == right_triplets

        for split in (schedule.val, schedule.test):
            assert {
                int(entry.metadata["cliff_front_gap_width"])
                for entry in split
            } == {1, 2, 3}
            for entry in split:
                assert entry.metadata["cliff_post_gap_runway"] == 1
                assert entry.metadata["cliff_probe_gap_width"] == 0
                assert entry.metadata["cliff_probe_ledge_width"] == 0

    def test_asymmetric_fixed_pair_plateau_gap_curriculum_8x2_preserves_geometry(self):
        schedule = generate_schedule(
            seed=42,
            n_gens=150,
            n_episodes=6,
            n_val=18,
            n_test=18,
            terrain_cfg=CURRICULUM_ASYMMETRIC_FIXED_PAIR_PLATEAU_GAP_8X2_CFG,
        )

        assert schedule.terrain_cfg["variant"] == "curriculum_bilateral_goal_cliff_asymmetric_pairs_plateau_gap_mix123_8x2_30"
        assert schedule.terrain_cfg["center_plateau_width"] == 12
        assert schedule.terrain_cfg["robot_w"] == 8
        assert schedule.curriculum_phase_end_gens == [50, 100, 150]
        assert {
            int(entry.metadata["cliff_front_gap_width"])
            for entry in schedule.test
        } == {1, 2, 3}

    def test_bilateral_probe_gap_curriculum_keeps_50_gen_run_in_easy_stage(self):
        schedule = generate_schedule(
            seed=42,
            n_gens=6,
            n_episodes=4,
            n_val=12,
            n_test=12,
            terrain_cfg=CURRICULUM_BILATERAL_PROBE_GAP_CFG,
        )

        assert schedule.curriculum is True
        assert schedule.transition_gen is None
        assert schedule.curriculum_phase_cfgs == CURRICULUM_BILATERAL_PROBE_GAP_CFG["curriculum_phases"]
        assert schedule.curriculum_phase_end_gens == [6, 6, 6]

        for generation in schedule.train:
            distances = {int(entry.metadata["goal_distance"]) for entry in generation}
            assert distances <= {1, 2}
            assert len(distances) >= 1
            for entry in generation:
                assert entry.metadata["cliff_probe_gap_width"] == 1
                assert entry.metadata["cliff_probe_ledge_width"] == 3

        for split in (schedule.val, schedule.test):
            seen = {
                (
                    int(entry.metadata["goal_distance"]),
                    str(entry.metadata["goal_side"]),
                )
                for entry in split
            }
            assert seen == {
                (distance, side)
                for distance in range(1, 7)
                for side in ("left", "right")
            }

    def test_bilateral_probe_gap_curriculum_switches_at_absolute_50_and_100(self):
        schedule = generate_schedule(
            seed=42,
            n_gens=120,
            n_episodes=4,
            n_val=12,
            n_test=12,
            terrain_cfg=CURRICULUM_BILATERAL_PROBE_GAP_CFG,
        )

        assert schedule.curriculum_phase_end_gens == [50, 100, 120]

        for generation in schedule.train[:50]:
            assert {int(entry.metadata["goal_distance"]) for entry in generation} <= {1, 2}
        for generation in schedule.train[50:100]:
            assert {int(entry.metadata["goal_distance"]) for entry in generation} <= {1, 2, 3, 4}
        for generation in schedule.train[100:]:
            assert {int(entry.metadata["goal_distance"]) for entry in generation} <= {1, 2, 3, 4, 5, 6}


class TestBilateralSemanticValidation:
    def test_generated_bilateral_schedule_passes_semantic_validation(self):
        schedule = generate_schedule(
            seed=42,
            n_gens=2,
            n_episodes=6,
            n_val=6,
            n_test=6,
            terrain_cfg=ASYMMETRIC_FIXED_PAIR_CFG,
        )

        validate_bilateral_schedule_semantics(schedule)

    def test_semantic_validation_rejects_mismatched_pair_goal_directions(self):
        bad_pair = [
            TerrainEntry(
                grid=np.zeros((10, 30), dtype=np.int32),
                cliff_x=20,
                gaps=[],
                metadata={
                    "task_mode": "bilateral_goal_cliff",
                    "goal_direction": 1,
                    "goal_distance": 2,
                    "cliff_distance": 1,
                },
            ),
            TerrainEntry(
                grid=np.zeros((10, 30), dtype=np.int32),
                cliff_x=20,
                gaps=[],
                metadata={
                    "task_mode": "bilateral_goal_cliff",
                    "goal_direction": 1,
                    "goal_distance": 2,
                    "cliff_distance": 1,
                },
            ),
        ]
        schedule = TerrainSchedule(
            train=[bad_pair],
            val=[],
            test=[],
            seed=0,
            terrain_cfg={"task_mode": "bilateral_goal_cliff"},
        )

        with pytest.raises(ValueError, match="invalid goal directions"):
            validate_bilateral_schedule_semantics(schedule)


class TestSaveLoadRoundtrip:
    def test_roundtrip(self, schedule, tmp_path):
        save_schedule(schedule, tmp_path / "sched")
        loaded = load_schedule(tmp_path / "sched")

        assert len(loaded.train) == len(schedule.train)
        assert len(loaded.val) == len(schedule.val)
        assert len(loaded.test) == len(schedule.test)

        for gen_a, gen_b in zip(schedule.train, loaded.train):
            for a, b in zip(gen_a, gen_b):
                assert np.array_equal(a.grid, b.grid)
                assert a.cliff_x == b.cliff_x
                assert a.gaps == b.gaps
                assert a.metadata == b.metadata

        for a, b in zip(schedule.val, loaded.val):
            assert np.array_equal(a.grid, b.grid)

    def test_curriculum_metadata_roundtrips(self, tmp_path):
        schedule = generate_schedule(
            seed=42,
            n_gens=6,
            n_episodes=2,
            n_val=2,
            n_test=2,
            terrain_cfg=CURRICULUM_CFG,
        )
        save_schedule(schedule, tmp_path / "sched")
        loaded = load_schedule(tmp_path / "sched")

        assert loaded.curriculum is True
        assert loaded.transition_gen == 3
        assert loaded.phase_1_cfg == CURRICULUM_CFG["phase_1"]
        assert loaded.phase_2_cfg == CURRICULUM_CFG["phase_2"]

    def test_multi_phase_curriculum_metadata_roundtrips(self, tmp_path):
        schedule = generate_schedule(
            seed=42,
            n_gens=6,
            n_episodes=4,
            n_val=12,
            n_test=12,
            terrain_cfg=CURRICULUM_BILATERAL_PROBE_GAP_CFG,
        )
        save_schedule(schedule, tmp_path / "sched")
        loaded = load_schedule(tmp_path / "sched")

        assert loaded.curriculum is True
        assert loaded.transition_gen is None
        assert loaded.curriculum_phase_cfgs == CURRICULUM_BILATERAL_PROBE_GAP_CFG["curriculum_phases"]
        assert loaded.curriculum_phase_end_gens == [6, 6, 6]

    def test_hashes_populated(self, schedule, tmp_path):
        save_schedule(schedule, tmp_path / "sched")
        loaded = load_schedule(tmp_path / "sched")
        assert loaded.hash_version == 2
        assert len(loaded.train_hash) == 64  # SHA256 hex
        assert len(loaded.val_hash) == 64
        assert len(loaded.test_hash) == 64
        assert len(loaded.train_content_hash) == 64
        assert len(loaded.val_content_hash) == 64
        assert len(loaded.test_content_hash) == 64

    def test_hash_determinism(self, tmp_path):
        s1 = generate_schedule(seed=42, n_gens=3, n_episodes=2, n_val=2, n_test=2, terrain_cfg=TERRAIN_CFG)
        s2 = generate_schedule(seed=42, n_gens=3, n_episodes=2, n_val=2, n_test=2, terrain_cfg=TERRAIN_CFG)
        save_schedule(s1, tmp_path / "a")
        save_schedule(s2, tmp_path / "b")
        a = load_schedule(tmp_path / "a")
        b = load_schedule(tmp_path / "b")
        assert a.train_hash == b.train_hash
        assert a.val_hash == b.val_hash
        assert a.test_hash == b.test_hash
        assert a.train_content_hash == b.train_content_hash
        assert a.val_content_hash == b.val_content_hash
        assert a.test_content_hash == b.test_content_hash

    def test_hash_verification_fails_on_tamper(self, schedule, tmp_path):
        save_schedule(schedule, tmp_path / "sched")
        # Tamper with train.npz.
        train_path = tmp_path / "sched" / "train.npz"
        data = bytearray(train_path.read_bytes())
        data[100] ^= 0xFF  # flip a byte
        train_path.write_bytes(bytes(data))
        with pytest.raises(ValueError, match="hash mismatch"):
            load_schedule(tmp_path / "sched", verify_hashes=True)

    def test_load_without_verification_flag(self, schedule, tmp_path):
        """verify_hashes=False skips hash check (file must still be valid npz)."""
        save_schedule(schedule, tmp_path / "sched")
        # Modify the content hash in the JSON (not the actual file).
        import json
        meta_path = tmp_path / "sched" / "schedule.json"
        meta = json.loads(meta_path.read_text())
        meta["train_content_hash"] = "0" * 64  # wrong hash
        meta["train_hash"] = "0" * 64  # legacy alias should not matter for v2 validation
        meta_path.write_text(json.dumps(meta))
        # With verification: should raise.
        with pytest.raises(ValueError, match="hash mismatch"):
            load_schedule(tmp_path / "sched", verify_hashes=True)
        # Without verification: should succeed.
        loaded = load_schedule(tmp_path / "sched", verify_hashes=False)
        assert len(loaded.train) > 0

    def test_hash_verification_detects_metadata_tamper(self, schedule, tmp_path):
        save_schedule(schedule, tmp_path / "sched")
        import json

        meta_path = tmp_path / "sched" / "schedule.json"
        meta = json.loads(meta_path.read_text())
        meta["train_metadata"][0][0]["metadata"]["tampered"] = True
        meta_path.write_text(json.dumps(meta))

        with pytest.raises(ValueError, match="hash mismatch"):
            load_schedule(tmp_path / "sched", verify_hashes=True)

    def test_legacy_hash_schema_requires_explicit_opt_in(self, schedule, tmp_path):
        save_schedule(schedule, tmp_path / "sched")
        import json
        import hashlib

        meta_path = tmp_path / "sched" / "schedule.json"
        meta = json.loads(meta_path.read_text())
        meta.pop("hash_version", None)
        meta.pop("train_content_hash", None)
        meta.pop("val_content_hash", None)
        meta.pop("test_content_hash", None)
        meta["train_hash"] = hashlib.sha256((tmp_path / "sched" / "train.npz").read_bytes()).hexdigest()
        meta["val_hash"] = hashlib.sha256((tmp_path / "sched" / "val.npz").read_bytes()).hexdigest()
        meta["test_hash"] = hashlib.sha256((tmp_path / "sched" / "test.npz").read_bytes()).hexdigest()
        meta_path.write_text(json.dumps(meta))

        with pytest.raises(ValueError, match="legacy hash schema"):
            load_schedule(tmp_path / "sched", verify_hashes=True)

        loaded = load_schedule(
            tmp_path / "sched",
            verify_hashes=True,
            allow_legacy_hashes=True,
        )
        assert len(loaded.train) > 0


class TestScheduleAccessor:
    def test_get_gen(self, schedule):
        entries = schedule_train_terrains_for_gen(schedule, 0)
        assert len(entries) == 4
        assert all(isinstance(e, TerrainEntry) for e in entries)

    def test_out_of_range(self, schedule):
        with pytest.raises(IndexError):
            schedule_train_terrains_for_gen(schedule, 100)


class TestWithGaps:
    def test_gap_schedule(self):
        cfg = {**TERRAIN_CFG, "gap_widths": [1, 2], "gap_budget": 2, "cliff_x_min": 15}
        s = generate_schedule(seed=42, n_gens=3, n_episodes=2, n_val=2, n_test=2, terrain_cfg=cfg)
        # At least some terrains should have gaps.
        all_entries = [e for gen in s.train for e in gen] + s.val + s.test
        has_gaps = any(len(e.gaps) > 0 for e in all_entries)
        assert has_gaps

    def test_exact_gap_schedule_preserves_terminal_runway(self):
        cfg = {
            **TERRAIN_CFG,
            "grid_w": 40,
            "cliff_x_min": 15,
            "cliff_x_max": 19,
            "gap_widths": [1, 2],
            "gap_budget": 2,
            "exact_gap_widths": [1, 2],
            "min_gap_separation": 1,
            "min_post_gap_runway": 1,
        }
        schedule = generate_schedule(
            seed=42,
            n_gens=3,
            n_episodes=2,
            n_val=2,
            n_test=2,
            terrain_cfg=cfg,
        )
        all_entries = [e for gen in schedule.train for e in gen] + schedule.val + schedule.test
        for entry in all_entries:
            assert sorted(width for _, width in entry.gaps) == [1, 2]
            last_gap_end = max(start + width - 1 for start, width in entry.gaps)
            assert last_gap_end + 2 <= entry.cliff_x
            assert entry.grid[3, last_gap_end + 1] == FIXED
            assert entry.grid[3, last_gap_end + 2] == FIXED
