"""Smoke tests for the full training pipeline.

NOTE: test_full_smoke requires GPU and should be run via Modal, not locally.
Local tests verify wiring and config resolution only.
"""

import pytest
import numpy as np
import jax

from sensing_edges.terrain import make_robot_body, generate_generation_terrains
from sensing_edges.nca import controller_flat_dim, init_ctrnn_params, ctrnn_params_to_flat
from sensing_edges.qd.grid_archive import GridArchive
from sensing_edges.qd.cma_emitter import CMAEmitter
from sensing_edges.fitness import candidate_fitness, episode_fitness


class TestWiring:
    """Lightweight local tests that verify module imports and dimension consistency."""

    def test_genotype_dim_single(self):
        dim = controller_flat_dim(8, 12, "off", ou_enabled=True)
        assert dim == 632

    def test_archive_accepts_genotypes(self):
        dim = 632
        archive = GridArchive(
            dims=(20, 20),
            ranges=[(0.0, 2.0), (0.0, 1.0)],
            learning_rate=0.01,
        )
        genotypes = np.random.randn(10, dim).astype(np.float64)
        fitnesses = np.random.rand(10).astype(np.float64)
        descriptors = np.column_stack([
            np.random.rand(10) * 3.0,
            np.random.rand(10),
        ])
        improvements, statuses = archive.add(genotypes, fitnesses, descriptors)
        assert improvements.shape == (10,)
        assert archive.size > 0

    def test_empty_archive_best_fitness_uses_threshold_min(self):
        archive = GridArchive(
            dims=(20, 20),
            ranges=[(0.0, 2.0), (0.0, 1.0)],
            learning_rate=0.01,
            threshold_min=0.0,
        )
        assert archive.size == 0
        assert archive.best_fitness == pytest.approx(0.0)

    @pytest.mark.parametrize("strategy", ["sep_cmaes", "snes"])
    def test_emitter_ask_tell_cycle(self, strategy):
        dim = 632
        em = CMAEmitter(genotype_length=dim, population_size=10, strategy=strategy)
        key = jax.random.PRNGKey(0)
        em.init(key)
        key, k = jax.random.split(key)
        pop = em.ask(k)
        assert pop.shape == (10, dim)
        improvements = np.random.rand(10)
        key, k = jax.random.split(key)
        em.tell(k, pop, improvements)
        assert em.es_state is not None

    def test_terrain_generation_deterministic(self):
        t1 = generate_generation_terrains(42, 4, grid_h=10, grid_w=35)
        t2 = generate_generation_terrains(42, 4, grid_h=10, grid_w=35)
        for (g1, c1, _, _), (g2, c2, _, _) in zip(t1, t2):
            assert np.array_equal(g1, g2)
            assert c1 == c2

    def test_robot_body(self):
        body = make_robot_body(7, 2)
        assert body.shape == (2, 7)


@pytest.mark.skip(reason="Requires GPU — run via Modal smoke test")
class TestFullSmoke:
    """Full pipeline smoke test. Run on Modal only."""

    def test_2_generations(self):
        """Run 2 generations with pop=10 and verify no crash."""
        # This would be called via modal_smoke.py
        pass
