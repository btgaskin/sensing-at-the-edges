"""Tests for manifest metadata helpers."""

from __future__ import annotations

from importlib import metadata as importlib_metadata
from types import SimpleNamespace

from sensing_edges import manifest


def test_package_version_prefers_module_attr():
    module = SimpleNamespace(__version__="1.2.3")
    assert manifest._package_version("evosax", module) == "1.2.3"


def test_package_version_falls_back_to_distribution_metadata(monkeypatch):
    monkeypatch.setattr(
        manifest.importlib_metadata,
        "version",
        lambda package_name: "9.9.9" if package_name == "evosax" else "unexpected",
    )

    assert manifest._package_version("evosax", SimpleNamespace()) == "9.9.9"


def test_package_version_handles_missing_distribution(monkeypatch):
    def _missing(_: str) -> str:
        raise importlib_metadata.PackageNotFoundError

    monkeypatch.setattr(manifest.importlib_metadata, "version", _missing)

    assert manifest._package_version("evosax", SimpleNamespace()) == "unknown"


def test_create_manifest_records_runtime_provenance():
    created = manifest.create_manifest(
        run_id="run-1",
        wandb_run_id="wb-1",
        preset_id="paper_run",
        resolved_config_str="seed: 42\n",
        logical_config_sha256="logical-1",
        last_invocation_sha256="invoke-1",
        seed=42,
        paper_core_id="sensing_edges_sbt_v1",
        paper_core_sha256="abc123",
        preset_changed_paths=["nca.ctrnn_hidden"],
        runtime_override_keys=["seed"],
        gpu_profile="l4",
        resolved_batching={"generation": {"candidate_chunk_size": 64}},
        resolved_collision={"dynamic_max_candidates": 5312},
    )

    assert created["paper_core_id"] == "sensing_edges_sbt_v1"
    assert created["paper_core_sha256"] == "abc123"
    assert created["preset_changed_paths"] == ["nca.ctrnn_hidden"]
    assert created["runtime_override_keys"] == ["seed"]
    assert created["runtime"]["gpu_profile"] == "l4"
    assert created["runtime"]["resolved_collision"] == {"dynamic_max_candidates": 5312}
    assert created["run_id"] == "run-1"
    assert created["wandb"]["run_id"] == "wb-1"
    assert created["config"]["logical_config_sha256"] == "logical-1"
    assert created["config"]["last_invocation_sha256"] == "invoke-1"
    assert created["lineage"]["mode"] == "root"
    assert created["stages"] == []
    assert created["retry"]["attempt_count"] == 1


def test_update_manifest_retry_state_updates_wandb_and_resume_fields(tmp_path):
    path = tmp_path / "manifest.json"
    manifest.save_manifest(
        manifest.create_manifest(
            run_id="run-1",
            wandb_run_id="wb-1",
            resolved_config_str="seed: 42\n",
        ),
        path,
    )

    updated = manifest.update_manifest_retry_state(
        path,
        attempt_count=3,
        resumed_from_checkpoint="checkpoints/gen_0007",
        resumed_generation=7,
        latest_checkpoint_path="checkpoints/gen_0009",
        wandb_run_id="wb-stable",
        status="running",
    )

    assert updated["retry"] == {
        "attempt_count": 3,
        "resumed_from_checkpoint": "checkpoints/gen_0007",
        "resumed_generation": 7,
        "latest_checkpoint_path": "checkpoints/gen_0009",
    }
    assert updated["wandb"]["run_id"] == "wb-stable"


def test_record_manifest_stage_result_appends_stage_and_updates_status(tmp_path):
    path = tmp_path / "manifest.json"
    manifest.save_manifest(
        manifest.create_manifest(
            run_id="run-1",
            wandb_run_id="wb-1",
            resolved_config_str="seed: 42\n",
        ),
        path,
    )

    updated = manifest.record_manifest_stage_result(
        path,
        status="paused",
        stage_entry={
            "status": "paused",
            "attempt_count": 1,
            "resume_mode": "staged",
            "stop_after_generations": 50,
            "completed_generations": 50,
            "completed_generation": 49,
            "latest_checkpoint_path": "checkpoints/gen_0049",
            "artifact_dir": "stages/gen_0049",
            "invocation_config_sha256": "invoke-1",
            "finished_at_utc": "2026-01-01T00:00:00+00:00",
        },
        completed_generations=50,
        best_train_fitness=12.5,
        final_qd_coverage=0.25,
        final_qd_score=8.0,
    )

    assert updated["status"] == "paused"
    assert updated["summary"]["completed_generations"] == 50
    assert updated["stages"][0]["artifact_dir"] == "stages/gen_0049"
    assert "completed_at_utc" not in updated
