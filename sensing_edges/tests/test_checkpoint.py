"""Tests for checkpoint save/load round-trips."""

from __future__ import annotations

import copy
import json

import jax
import numpy as np

from sensing_edges.checkpoint import (
    build_archive_export_payload,
    find_latest_checkpoint,
    load_checkpoint,
    payload_done_path,
    save_checkpoint,
    write_archive_export_payload,
)
from sensing_edges.qd.cma_emitter import CMAEmitter
from sensing_edges.qd.focus import FocusWindowBounds
from sensing_edges.qd.grid_archive import GridArchive


def _make_archive(genotype_length: int) -> GridArchive:
    archive = GridArchive(
        dims=(4, 4),
        ranges=[(0.0, 2.0), (0.0, 1.0)],
        learning_rate=0.01,
        threshold_min=0.0,
    )
    genotypes = np.array(
        [
            np.linspace(-0.5, 0.5, genotype_length, dtype=np.float64),
            np.linspace(0.5, -0.5, genotype_length, dtype=np.float64),
        ]
    )
    fitnesses = np.array([1.5, 2.5], dtype=np.float64)
    descriptors = np.array([[0.2, 0.1], [0.8, 0.4]], dtype=np.float64)
    archive.add(genotypes, fitnesses, descriptors)
    return archive


def _make_emitter(genotype_length: int) -> CMAEmitter:
    emitter = CMAEmitter(
        genotype_length=genotype_length,
        population_size=6,
        strategy="sep_cmaes",
        stale_generations=4,
        max_age=12,
        meaningful_improvement_eps=0.5,
        flat_generations=4,
        flat_tolerance=1.0e-6,
        std_epsilon=1.0e-7,
        condition_cov_limit=1.0e12,
        param_overrides={"std_init": 0.5},
    )
    key = jax.random.PRNGKey(0)
    emitter.init(key)
    key, ask_key, tell_key = jax.random.split(key, 3)
    population = emitter.ask(ask_key)
    improvements = np.linspace(0.1, 0.6, 6, dtype=np.float64)
    statuses = np.zeros(6, dtype=np.uint8)
    statuses[-1] = 1
    emitter.tell(tell_key, population, improvements, statuses)
    emitter.stale_counter = 2
    emitter.age_since_restart = 3
    emitter.total_restarts = 1
    emitter.total_restarts_by_reason["stale"] = 1
    emitter.last_restart_reason = "stale"
    return emitter


class TestCheckpointRoundtrip:
    def test_save_and_load_roundtrip(self, tmp_path):
        genotype_length = 4
        archive = _make_archive(genotype_length)
        emitter = _make_emitter(genotype_length)
        rng = np.random.default_rng(123)
        rng.integers(0, 1000, size=4)
        expected_rng_state = copy.deepcopy(rng.bit_generator.state)
        jax_key = jax.random.PRNGKey(17)

        prefix = tmp_path / "checkpoints" / "gen_0000"
        json_path, npz_path = save_checkpoint(
            prefix,
            archive=archive,
            emitters=[emitter],
            rng=rng,
            jax_key=jax_key,
            generation=7,
            config_sha256="abc123",
            run_id="run-1",
        )

        assert json_path.exists()
        assert npz_path.exists()

        loaded = load_checkpoint(
            prefix,
            genotype_length=genotype_length,
            population_size=6,
            strategy="sep_cmaes",
            stale_generations=4,
            max_age=12,
            meaningful_improvement_eps=0.5,
            flat_generations=4,
            flat_tolerance=1.0e-6,
            std_epsilon=1.0e-7,
            condition_cov_limit=1.0e12,
            param_overrides={"std_init": 0.5},
        )

        assert loaded["generation"] == 7
        assert loaded["archive"].size == archive.size
        assert loaded["emitters"][0].es_state is not None
        assert loaded["emitters"][0].stale_counter == emitter.stale_counter
        assert loaded["emitters"][0].age_since_restart == emitter.age_since_restart
        assert loaded["emitters"][0].total_restarts == emitter.total_restarts
        assert loaded["emitters"][0].total_restarts_by_reason == emitter.total_restarts_by_reason
        assert loaded["emitters"][0].last_restart_reason == emitter.last_restart_reason
        assert loaded["meta"]["emitters"]["population_size"] == 6
        assert loaded["meta"]["emitters"]["param_overrides"] == {"std_init": 0.5}
        np.testing.assert_array_equal(np.array(loaded["jax_key"]), np.array(jax_key))

        original_state = emitter.es_state
        restored_state = loaded["emitters"][0].es_state
        for field_name in original_state.__dataclass_fields__:
            np.testing.assert_allclose(
                np.array(getattr(restored_state, field_name)),
                np.array(getattr(original_state, field_name)),
                atol=1e-6,
                equal_nan=True,
            )

        expected_rng = np.random.default_rng()
        expected_rng.bit_generator.state = expected_rng_state
        np.testing.assert_array_equal(
            loaded["rng"].integers(0, 1000, size=8),
            expected_rng.integers(0, 1000, size=8),
        )

    def test_save_and_load_roundtrip_preserves_focus_window_bounds(self, tmp_path):
        genotype_length = 4
        archive = _make_archive(genotype_length)
        emitter = _make_emitter(genotype_length)
        focus_window_bounds = FocusWindowBounds(
            lower={0: 4, 1: 0},
            upper={0: 6, 1: 2},
        )

        prefix = tmp_path / "checkpoints" / "gen_0001"
        save_checkpoint(
            prefix,
            archive=archive,
            emitters=[emitter],
            rng=np.random.default_rng(123),
            jax_key=jax.random.PRNGKey(5),
            generation=8,
            config_sha256="cfg-focus",
            run_id="run-focus",
            focus_window_bounds=focus_window_bounds,
        )

        loaded = load_checkpoint(
            prefix,
            genotype_length=genotype_length,
            population_size=6,
            strategy="sep_cmaes",
            stale_generations=4,
            max_age=12,
            meaningful_improvement_eps=0.5,
            flat_generations=4,
            flat_tolerance=1.0e-6,
            std_epsilon=1.0e-7,
            condition_cov_limit=1.0e12,
            param_overrides={"std_init": 0.5},
        )

        assert loaded["focus_window_bounds"] is not None
        assert loaded["focus_window_bounds"].lower == focus_window_bounds.lower
        assert loaded["focus_window_bounds"].upper == focus_window_bounds.upper

    def test_find_latest_checkpoint_requires_done_and_skips_newer_partial_or_unreadable(self, tmp_path):
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        def _write_stub(generation: int, *, valid_json: bool, done: bool) -> None:
            prefix = checkpoint_dir / f"gen_{generation:04d}"
            json_path = prefix.with_suffix(".json")
            npz_path = prefix.with_suffix(".npz")
            if valid_json:
                json_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "kind": "sensing_edges_checkpoint",
                            "run_id": "run-1",
                            "generation": generation,
                            "created_at_utc": f"2026-01-01T00:00:{generation:02d}+00:00",
                            "config_sha256": "cfg-1",
                        }
                    )
                )
            else:
                json_path.write_text("{not-json")
            npz_path.write_bytes(b"npz")
            if done:
                payload_done_path(prefix).write_text("{}")

        _write_stub(3, valid_json=True, done=True)
        _write_stub(4, valid_json=True, done=False)
        _write_stub(5, valid_json=False, done=True)

        latest = find_latest_checkpoint(
            checkpoint_dir,
            config_sha256="cfg-1",
            run_id="run-1",
            require_done=True,
        )

        assert latest is not None
        prefix, meta = latest
        assert prefix.name == "gen_0003"
        assert meta["generation"] == 3

    def test_write_archive_export_payload_marks_done(self, tmp_path):
        archive = _make_archive(genotype_length=4)
        meta, arrays = build_archive_export_payload(
            archive=archive,
            config_sha256="cfg-1",
            run_id="run-1",
        )

        prefix = tmp_path / "final_archive"
        json_path, npz_path = write_archive_export_payload(
            prefix,
            meta=meta,
            arrays=arrays,
        )

        assert json_path.exists()
        assert npz_path.exists()
        assert payload_done_path(prefix).exists()
