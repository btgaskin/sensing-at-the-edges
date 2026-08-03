"""Tests for publication artifact generation."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from sensing_edges.checkpoint import payload_done_path
from sensing_edges.publication import (
    bundle_publication_release,
    generate_deformation_ablation_tables,
    load_publication_set,
    load_run_index,
    materialize_publication_release,
    release_root,
)
from sensing_edges.runtime_config import compose_experiment_config, run_metadata, run_name_label


def _write_run_fixture(
    outputs_dir: Path,
    *,
    seed: int,
    deform_off: bool,
    status: str = "completed",
    lineage_mode: str = "staged",
    test_mean: float = 80.0,
    progress: float = 72.0,
    mean_fall_count: float = 0.10,
) -> Path:
    overrides = [f"seed={seed}", "body.mode=fixed", "body.shape=rectangle", "terrain.robot_w=7", "terrain.robot_h=3"]
    if deform_off:
        overrides.append("nca.ablate_deformation=true")
    cfg = compose_experiment_config(preset="paper_run", overrides=overrides)
    run_dir_name = f"{cfg.preset_id}_{run_name_label(cfg)}"
    run_dir = outputs_dir / run_dir_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    OmegaConf.save(cfg, run_dir / "resolved_config.yaml")

    run_id = f"run-{run_dir_name}"
    logical_hash = "cfg-hash"
    manifest = {
        "run_id": run_id,
        "preset_id": cfg.preset_id,
        "status": status,
        "lineage": {"mode": lineage_mode},
        "summary": {"completed_generations": int(cfg.train.n_generations)},
        "retry": {"latest_checkpoint_path": f"/outputs/{run_dir_name}/checkpoints/gen_0199"},
        "config": {"logical_config_sha256": logical_hash},
        "terrain_schedule": {"dir": "terrain_suites/seed_42_curriculum_zone10to19_extend24_gap2_30_g200_ep4_val8_test16"},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "metrics.jsonl").write_text(json.dumps({"generation": 0, "best_fitness": 1.0, "coverage": 0.1}) + "\n")

    checkpoint_prefix = checkpoint_dir / "gen_0199"
    checkpoint_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "sensing_edges_checkpoint",
                "run_id": run_id,
                "generation": 199,
                "config_sha256": logical_hash,
            }
        )
    )
    checkpoint_prefix.with_suffix(".npz").write_bytes(b"npz")
    payload_done_path(checkpoint_prefix).write_text("{}")

    final_archive_prefix = run_dir / "final_archive"
    final_archive_prefix.with_suffix(".json").write_text("{}")
    final_archive_prefix.with_suffix(".npz").write_bytes(b"npz")
    payload_done_path(final_archive_prefix).write_text("{}")

    payload = {
        "run_metadata": run_metadata(cfg),
        "test_mean_fitness": test_mean,
        "test_median_fitness": test_mean - 1.0,
        "test_summary": {
            "mean_progress_shaping": progress,
            "mean_any_progress": 0.0,
            "mean_goal_progress": 0.0,
            "mean_correct_progress": 0.0,
            "mean_incorrect_progress": 0.0,
            "mean_toward_goal_frontier": 0.0,
            "mean_safe_arrival_bonus": 24.0,
            "mean_safe_dwell_bonus": 9.0,
            "mean_urgency_shaping": 4.0,
            "goal_touch_rate": 0.0,
            "mean_fall_count": mean_fall_count,
            "edge_entry_rate": 0.5,
            "mean_probe_extent": 0.2,
            "mean_time_to_edge": 0.7,
            "mean_time_to_goal": 0.0,
            "mean_turn_count": 0.0,
            "mean_activation_cost": 0.1,
        },
        "test_ci95_low": test_mean - 5.0,
        "test_ci95_high": test_mean + 5.0,
        "selected_elite_rank": 0,
        "selected_elite_flat_index": 7,
        "selected_elite_fitness_rerank": test_mean + 3.0,
        "n_rerank_terrains": 8,
        "n_test_terrains": 16,
        "rerank_schedule": {"dir": "terrain_suites/seed_42_single_gap2_zone10to24_30_g200_ep4_val8_test16"},
        "eval_schedule": {"dir": "terrain_suites/seed_42_single_gap3_zone10to24_30_ood_g200_ep4_val8_test16"},
    }
    post_train_dir = run_dir / "post_train" / "archive_eval"
    post_train_dir.mkdir(parents=True)
    (post_train_dir / "summary.json").write_text(json.dumps(payload))
    return run_dir


def test_materialize_release_and_generate_tables(tmp_path):
    outputs_dir = tmp_path / "outputs"
    _write_run_fixture(outputs_dir, seed=42, deform_off=False, test_mean=80.0, progress=72.0, mean_fall_count=0.10)
    _write_run_fixture(outputs_dir, seed=42, deform_off=True, test_mean=65.0, progress=55.0, mean_fall_count=0.35)

    publication_set = load_publication_set("paper_main", se_root=Path(__file__).parent.parent)
    release_manifest, _records = materialize_publication_release(
        outputs_dir,
        release_id="paper_main",
        publication_set=publication_set,
        se_root=Path(__file__).parent.parent,
    )

    release_dir = release_root(outputs_dir, "paper_main")
    comparisons = generate_deformation_ablation_tables(release_dir)
    assert release_manifest["counts"]["ready_runs"] == 2
    assert len(load_run_index(release_dir)) == 2
    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert comparison["body_variant"] == "rect_7x3"
    assert comparison["metrics"]["test_mean_fitness"]["delta_mean"] == 15.0
    assert comparison["metrics"]["mean_progress_shaping"]["delta_mean"] == 17.0
    assert comparison["metrics"]["mean_fall_count"]["delta_mean"] == pytest.approx(-0.25)
    assert comparison["metrics"]["test_mean_fitness"]["delta_bootstrap_ci95_low"] <= 15.0
    assert comparison["metrics"]["test_mean_fitness"]["delta_bootstrap_ci95_high"] >= 15.0


def test_bundle_release_includes_durable_checkpoint_and_release_tree(tmp_path):
    outputs_dir = tmp_path / "outputs"
    run_dir = _write_run_fixture(outputs_dir, seed=42, deform_off=False)

    publication_set = load_publication_set("paper_main", se_root=Path(__file__).parent.parent)
    materialize_publication_release(
        outputs_dir,
        release_id="paper_main",
        publication_set=publication_set,
        se_root=Path(__file__).parent.parent,
    )
    release_dir = release_root(outputs_dir, "paper_main")
    generate_deformation_ablation_tables(release_dir)

    bundle_path = bundle_publication_release(
        release_dir,
        se_root=Path(__file__).parent.parent,
    )

    with tarfile.open(bundle_path, "r:gz") as tar:
        arcnames = set(tar.getnames())

    run_id = f"run-{run_dir.name}"
    assert f"publication/paper_main/release_manifest.json" in arcnames
    assert f"publication/paper_main/manifests/runs.jsonl" in arcnames
    assert f"publication/paper_main/summaries/{run_id}.json" in arcnames
    assert f"outputs/{run_dir.name}/checkpoints/gen_0199.json" in arcnames
    assert f"outputs/{run_dir.name}/checkpoints/gen_0199.npz" in arcnames
    assert f"outputs/{run_dir.name}/checkpoints/gen_0199.done" in arcnames
    assert f"outputs/{run_dir.name}/final_archive.json" in arcnames


def test_materialize_release_marks_paused_and_branch_runs_not_ready(tmp_path):
    outputs_dir = tmp_path / "outputs"
    _write_run_fixture(outputs_dir, seed=42, deform_off=False, status="paused")
    _write_run_fixture(outputs_dir, seed=99, deform_off=False, lineage_mode="branch")

    publication_set = load_publication_set("paper_main", se_root=Path(__file__).parent.parent)
    _manifest, records = materialize_publication_release(
        outputs_dir,
        release_id="paper_main",
        publication_set=publication_set,
        se_root=Path(__file__).parent.parent,
    )

    assert len(records) == 2
    reasons = {record["exclude_reason"] for record in records}
    assert "status:paused" in reasons
    assert "lineage:branch" in reasons
    assert all(not record["publication_ready"] for record in records)
