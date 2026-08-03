"""Tests for the sensing_edges QD grid archive."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from sensing_edges.qd.grid_archive import GridArchive


def _cell_idx(archive: GridArchive, descriptor: np.ndarray) -> int:
    return archive._ravel(archive._to_cell(descriptor))


def test_add_new_cell_improvement_and_threshold_update():
    archive = GridArchive(
        dims=(10,),
        ranges=[(0.0, 1.0)],
        learning_rate=0.5,
        threshold_min=0.0,
    )
    improvements, statuses = archive.add(
        np.array([[1.0, 2.0]], dtype=np.float32),
        np.array([8.0], dtype=np.float64),
        np.array([[0.5]], dtype=np.float64),
    )

    flat_idx = _cell_idx(archive, np.array([0.5], dtype=np.float64))
    assert improvements[0] == pytest.approx(8.0)
    assert statuses[0] == GridArchive.NEW
    assert archive._thresholds[flat_idx] == pytest.approx(4.0)
    assert archive.size == 1


def test_add_same_cell_accepts_only_best_winner():
    archive = GridArchive(dims=(10,), ranges=[(0.0, 1.0)])
    improvements, statuses = archive.add(
        np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        np.array([3.0, 9.0, 7.0], dtype=np.float64),
        np.array([[0.5], [0.5], [0.5]], dtype=np.float64),
    )

    assert np.sum(statuses == GridArchive.NEW) == 1
    assert np.sum(statuses == GridArchive.NOT_ADDED) == 2
    assert archive.best_fitness == pytest.approx(9.0)
    assert float(np.max(improvements)) == pytest.approx(9.0)


def test_score_batch_commit_scored_batch_keeps_first_tied_winner_and_updates_cells():
    archive = GridArchive(
        dims=(10,),
        ranges=[(0.0, 1.0)],
        learning_rate=0.5,
        threshold_min=0.0,
    )
    archive.add(
        np.array([[11.0], [22.0]], dtype=np.float32),
        np.array([5.0, 4.0], dtype=np.float64),
        np.array([[0.1], [0.7]], dtype=np.float64),
    )

    genotypes = np.array([[101.0], [202.0], [303.0], [404.0], [505.0]], dtype=np.float32)
    fitnesses = np.array([8.0, 8.0, 6.0, 1.0, 7.0], dtype=np.float64)
    descriptors = np.array([[0.1], [0.1], [0.3], [0.7], [0.7]], dtype=np.float64)

    decision = archive.score_batch(fitnesses, descriptors)
    accepted_rows = decision.accepted_indices
    archive.commit_scored_batch(
        decision,
        genotypes[accepted_rows],
        descriptors[accepted_rows],
    )

    first_cell_idx = _cell_idx(archive, np.array([0.1], dtype=np.float64))
    second_cell_idx = _cell_idx(archive, np.array([0.3], dtype=np.float64))
    third_cell_idx = _cell_idx(archive, np.array([0.7], dtype=np.float64))

    np.testing.assert_allclose(decision.improvements, [5.5, 5.5, 6.0, -1.0, 5.0])
    np.testing.assert_array_equal(
        decision.statuses,
        np.array(
            [
                GridArchive.IMPROVE,
                GridArchive.NOT_ADDED,
                GridArchive.NEW,
                GridArchive.NOT_ADDED,
                GridArchive.IMPROVE,
            ],
            dtype=np.uint8,
        ),
    )
    np.testing.assert_array_equal(decision.accepted_indices, np.array([0, 2, 4], dtype=np.int64))
    assert archive._fitnesses[first_cell_idx] == pytest.approx(8.0)
    assert archive._fitnesses[second_cell_idx] == pytest.approx(6.0)
    assert archive._fitnesses[third_cell_idx] == pytest.approx(7.0)
    assert archive._thresholds[first_cell_idx] == pytest.approx(5.25)
    assert archive._thresholds[second_cell_idx] == pytest.approx(3.0)
    assert archive._thresholds[third_cell_idx] == pytest.approx(4.5)
    np.testing.assert_array_equal(archive._genotypes[first_cell_idx], genotypes[0])
    np.testing.assert_array_equal(archive._genotypes[second_cell_idx], genotypes[2])
    np.testing.assert_array_equal(archive._genotypes[third_cell_idx], genotypes[4])


def test_score_batch_empty_batch_returns_no_winners():
    archive = GridArchive(dims=(10,), ranges=[(0.0, 1.0)])

    decision = archive.score_batch(
        np.array([], dtype=np.float64),
        np.zeros((0, 1), dtype=np.float64),
    )

    assert decision.improvements.shape == (0,)
    assert decision.statuses.shape == (0,)
    assert decision.accepted_indices.shape == (0,)
    archive.commit_scored_batch(
        decision,
        np.zeros((0, 2), dtype=np.float32),
        np.zeros((0, 1), dtype=np.float64),
    )


def test_top_elites_in_projection_returns_best_occupied_matches():
    archive = GridArchive(
        dims=(7, 7, 4),
        ranges=[(0.0, 6.0), (0.0, 6.0), (0.0, 3.0)],
        descriptor_names=("fall_count", "goal_hit_count", "stable_turn_count"),
    )
    genotypes = np.array(
        [[11.0], [22.0], [33.0], [44.0], [55.0]],
        dtype=np.float32,
    )
    fitnesses = np.array([10.0, 30.0, 20.0, 40.0, 50.0], dtype=np.float64)
    descriptors = np.array(
        [
            [0.0, 6.0, 0.0],
            [0.0, 6.0, 1.0],
            [0.0, 6.0, 2.0],
            [1.0, 6.0, 3.0],
            [0.0, 5.0, 3.0],
        ],
        dtype=np.float64,
    )
    archive.add(genotypes, fitnesses, descriptors)

    elites = archive.top_elites_in_projection(axis_cells={0: 0, 1: 6}, n=3)

    assert [float(item["fitness"]) for item in elites] == [30.0, 20.0, 10.0]
    assert [int(item["flat_index"]) for item in elites] == [
        archive._ravel((0, 6, 1)),
        archive._ravel((0, 6, 2)),
        archive._ravel((0, 6, 0)),
    ]


def test_top_elites_by_descriptor_priority_uses_lexicographic_order():
    archive = GridArchive(
        dims=(7, 7, 24),
        ranges=[(0.0, 6.0), (0.0, 6.0), (0.0, 6.0)],
        descriptor_names=("fall_count", "goal_hit_count", "stable_turn_count"),
    )
    genotypes = np.array(
        [[11.0], [22.0], [33.0], [44.0], [55.0]],
        dtype=np.float32,
    )
    fitnesses = np.array([20.0, 100.0, 50.0, 10.0, 5.0], dtype=np.float64)
    descriptors = np.array(
        [
            [0.0, 6.0, 2.1],
            [1.0, 6.0, 0.1],
            [0.0, 5.0, 0.1],
            [0.0, 6.0, 1.1],
            [0.0, 6.0, 0.1],
        ],
        dtype=np.float64,
    )
    archive.add(genotypes, fitnesses, descriptors)

    elites = archive.top_elites_by_descriptor_priority(
        axis_directions=((1, "max"), (0, "min"), (2, "min")),
        n=5,
    )

    assert [float(item["fitness"]) for item in elites] == [5.0, 10.0, 20.0, 100.0, 50.0]
    assert [int(item["flat_index"]) for item in elites] == [
        archive._ravel(archive._to_cell(np.array([0.0, 6.0, 0.1], dtype=np.float64))),
        archive._ravel(archive._to_cell(np.array([0.0, 6.0, 1.1], dtype=np.float64))),
        archive._ravel(archive._to_cell(np.array([0.0, 6.0, 2.1], dtype=np.float64))),
        archive._ravel(archive._to_cell(np.array([1.0, 6.0, 0.1], dtype=np.float64))),
        archive._ravel(archive._to_cell(np.array([0.0, 5.0, 0.1], dtype=np.float64))),
    ]


def test_score_batch_ignores_non_finite_fitnesses():
    archive = GridArchive(dims=(10,), ranges=[(0.0, 1.0)])

    decision = archive.score_batch(
        np.array([np.nan, np.inf, 3.0], dtype=np.float64),
        np.array([[0.1], [0.2], [0.2]], dtype=np.float64),
    )

    np.testing.assert_allclose(decision.improvements, [0.0, 0.0, 3.0])
    np.testing.assert_array_equal(
        decision.statuses,
        np.array(
            [
                GridArchive.NOT_ADDED,
                GridArchive.NOT_ADDED,
                GridArchive.NEW,
            ],
            dtype=np.uint8,
        ),
    )
    archive.commit_scored_batch(
        decision,
        np.array([[33.0]], dtype=np.float32),
        np.array([[0.2]], dtype=np.float64),
    )
    assert archive.size == 1


def test_score_batch_rejects_non_finite_descriptors_and_tracks_invalid_count():
    archive = GridArchive(dims=(10,), ranges=[(0.0, 1.0)])

    decision = archive.score_batch(
        np.array([5.0, 7.0, 9.0], dtype=np.float64),
        np.array([[0.1], [np.nan], [np.inf]], dtype=np.float64),
    )

    np.testing.assert_allclose(decision.improvements, [5.0, 0.0, 0.0])
    np.testing.assert_array_equal(
        decision.statuses,
        np.array(
            [
                GridArchive.NEW,
                GridArchive.NOT_ADDED,
                GridArchive.NOT_ADDED,
            ],
            dtype=np.uint8,
        ),
    )
    np.testing.assert_array_equal(decision.descriptor_finite_mask, np.array([True, False, False]))
    assert decision.invalid_descriptor_count == 2

    archive.commit_scored_batch(
        decision,
        np.array([[11.0]], dtype=np.float32),
        np.array([[0.1]], dtype=np.float64),
    )
    assert archive.size == 1


def test_sample_elites_shape():
    archive = GridArchive(dims=(10,), ranges=[(0.0, 1.0)])
    archive.add(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([5.0, 7.0], dtype=np.float64),
        np.array([[0.1], [0.9]], dtype=np.float64),
    )

    samples = archive.sample_elites(jax.random.PRNGKey(0), 5)
    assert samples.shape == (5, 2)


# ---------------------------------------------------------------------------
# Active cell mask tests
# ---------------------------------------------------------------------------


def test_score_batch_inactive_cells_get_zero_improvement():
    """Candidates landing in inactive cells must have improvement=0."""
    archive = GridArchive(dims=(5, 5), ranges=[(0.0, 1.0), (0.0, 1.0)])
    fitnesses = np.array([10.0, 10.0], dtype=np.float64)
    descriptors = np.array([[0.1, 0.1], [0.9, 0.9]], dtype=np.float64)

    # Mask: only cell (4, 4) is active (where descriptor (0.9, 0.9) lands)
    mask = np.zeros(25, dtype=bool)
    mask[24] = True  # flat index for (4, 4)

    decision = archive.score_batch(fitnesses, descriptors, active_cell_mask=mask)

    # First candidate (0.1, 0.1) → cell (0, 0) → inactive → improvement=0
    assert decision.improvements[0] == 0.0
    assert decision.statuses[0] == GridArchive.NOT_ADDED
    # Second candidate (0.9, 0.9) → cell (4, 4) → active → improvement>0
    assert decision.improvements[1] > 0.0
    assert decision.statuses[1] == GridArchive.NEW


def test_score_batch_active_cells_score_normally():
    """Active cells should score and accept candidates as usual."""
    archive = GridArchive(dims=(5,), ranges=[(0.0, 1.0)])

    mask = np.ones(5, dtype=bool)  # all active
    fitnesses = np.array([5.0], dtype=np.float64)
    descriptors = np.array([[0.5]], dtype=np.float64)

    decision = archive.score_batch(fitnesses, descriptors, active_cell_mask=mask)
    assert decision.improvements[0] == pytest.approx(5.0)
    assert decision.statuses[0] == GridArchive.NEW


def test_sample_elites_with_cell_mask_filters():
    """sample_elites with cell_mask should only return elites from active cells."""
    archive = GridArchive(dims=(10,), ranges=[(0.0, 1.0)])
    archive.add(
        np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        np.array([5.0, 7.0, 9.0], dtype=np.float64),
        np.array([[0.05], [0.55], [0.95]], dtype=np.float64),
    )

    # Only allow the cell containing descriptor=0.95 (cell 9)
    mask = np.zeros(10, dtype=bool)
    mask[9] = True

    for _ in range(10):
        samples = archive.sample_elites(jax.random.PRNGKey(42), 3, cell_mask=mask)
        # All samples should be the elite at cell 9 (genotype=[3.0])
        np.testing.assert_allclose(samples, np.array([[3.0], [3.0], [3.0]]))


def test_sample_elites_empty_active_region_raises():
    """sample_elites should raise ValueError when active region has no elites."""
    archive = GridArchive(dims=(10,), ranges=[(0.0, 1.0)])
    archive.add(
        np.array([[1.0]], dtype=np.float32),
        np.array([5.0], dtype=np.float64),
        np.array([[0.05]], dtype=np.float64),
    )

    # Mask excludes the occupied cell
    mask = np.zeros(10, dtype=bool)
    mask[9] = True  # cell 9 is empty

    with pytest.raises(ValueError, match="Cannot sample from empty archive"):
        archive.sample_elites(jax.random.PRNGKey(0), 1, cell_mask=mask)


def test_sample_frontier_elites_prefers_emptier_target_neighborhoods():
    archive = GridArchive(
        dims=(5, 5, 3),
        ranges=[(0.0, 4.0), (0.0, 4.0), (0.0, 2.0)],
    )
    archive.add(
        np.array([[10.0], [20.0], [30.0], [40.0], [50.0]], dtype=np.float32),
        np.array([5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float64),
        np.array(
            [
                [2.0, 2.0, 0.0],  # isolated projected cell
                [4.0, 0.0, 0.0],  # crowded target corner
                [3.0, 0.0, 0.0],
                [4.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )

    samples = archive.sample_frontier_elites(
        jax.random.PRNGKey(7),
        1000,
        axis_directions=((0, "max"), (1, "min")),
    )
    counts = {
        float(value): int(np.sum(samples[:, 0] == value))
        for value in np.unique(samples[:, 0])
    }
    assert counts[10.0] > counts[20.0]


def test_score_batch_rejects_invalid_active_cell_mask_shape():
    archive = GridArchive(dims=(5, 5), ranges=[(0.0, 1.0), (0.0, 1.0)])

    with pytest.raises(ValueError, match="active_cell_mask"):
        archive.score_batch(
            np.array([1.0], dtype=np.float64),
            np.array([[0.5, 0.5]], dtype=np.float64),
            active_cell_mask=np.ones(5, dtype=bool),
        )


def test_sample_elites_rejects_invalid_cell_mask_shape():
    archive = GridArchive(dims=(5,), ranges=[(0.0, 1.0)])
    archive.add(
        np.array([[1.0]], dtype=np.float32),
        np.array([5.0], dtype=np.float64),
        np.array([[0.2]], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="cell_mask"):
        archive.sample_elites(jax.random.PRNGKey(0), 1, cell_mask=np.ones(2, dtype=bool))


def test_axis_coverage_tracks_marginal_occupied_bins():
    archive = GridArchive(
        dims=(7, 7, 20),
        ranges=[(0.0, 6.0), (0.0, 6.0), (0.2, 0.8)],
        descriptor_names=("goal_hit_count", "fall_count", "completion_time"),
    )
    archive.add(
        np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        np.array([5.0, 6.0, 7.0], dtype=np.float64),
        np.array(
            [
                [4.0, 2.0, 0.3],
                [6.0, 0.0, 0.7],
                [4.0, 1.0, 0.5],
            ],
            dtype=np.float64,
        ),
    )

    assert archive.axis_occupied_bins == {
        "goal_hit_count": 2,
        "fall_count": 3,
        "completion_time": 3,
    }
    assert archive.axis_coverage["goal_hit_count"] == pytest.approx(2 / 7)
    assert archive.axis_coverage["fall_count"] == pytest.approx(3 / 7)
    assert archive.axis_coverage["completion_time"] == pytest.approx(3 / 20)
