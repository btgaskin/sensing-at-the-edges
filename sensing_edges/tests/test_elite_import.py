from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
from omegaconf import OmegaConf

import sensing_edges.elite_import as elite_import_module
from sensing_edges.elite_import import (
    infer_run_dir_name_from_specimen_root,
    import_raw_elite_snapshots,
    normalize_local_specimen_source_path,
    parse_remote_elite_path,
    run_post_run_for_local_specimen_root_remote,
    run_post_run_for_imported_specimens_remote,
)
from sensing_edges.runtime_config import compose_experiment_config, genotype_dim_from_cfg


def test_parse_remote_elite_path_accepts_expected_modal_layout():
    run_dir_name, filename = parse_remote_elite_path(
        "/outputs/paper_run_rect_7x2_seed42/elite_specimens/gen_0007_candidate_0012_cell_6-0-4_hits6_falls0.npz"
    )

    assert run_dir_name == "paper_run_rect_7x2_seed42"
    assert filename == "gen_0007_candidate_0012_cell_6-0-4_hits6_falls0.npz"

    run_dir_name, filename = parse_remote_elite_path(
        "/paper_run_rect_7x2_seed42/elite_specimens/gen_0007_candidate_0012_cell_6-0-4_hits6_falls0.npz"
    )

    assert run_dir_name == "paper_run_rect_7x2_seed42"
    assert filename == "gen_0007_candidate_0012_cell_6-0-4_hits6_falls0.npz"


def test_import_raw_elite_snapshots_materializes_local_specimen_tree(tmp_path):
    cfg = compose_experiment_config(preset="paper_run")
    run_dir_name = "paper_run_rect_7x2_seed42"
    run_manifest_path = tmp_path / "remote_run_manifest.json"
    run_manifest_path.write_text(
        json.dumps(
            {
                "run_id": "run-42",
                "config": {},
                "runtime": {},
                "hardware": {},
                "system": {},
                "git": {},
                "wandb": {},
                "lineage": {},
            }
        )
    )
    run_resolved_config_path = tmp_path / "resolved_config.yaml"
    run_resolved_config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True))

    genotype = np.zeros((genotype_dim_from_cfg(cfg),), dtype=np.float32)
    raw_dir = tmp_path / "raw_source"
    raw_dir.mkdir()
    raw_a = raw_dir / "gen_0064_candidate_0092_cell_6-0-4_hits6_falls0.npz"
    np.savez_compressed(
        raw_a,
        genotype=genotype,
        genotype_sha256=np.asarray("sha-a", dtype=str),
        fitness=np.asarray(1.25, dtype=np.float32),
        descriptor=np.asarray([6.0, 0.0, 4.0], dtype=np.float32),
        descriptor_names=np.asarray(["completion_bin", "falls", "time_bin"], dtype=str),
        goal_hit_count=np.asarray(6.0, dtype=np.float32),
        fall_count=np.asarray(0.0, dtype=np.float32),
        historical_elite_tier=np.asarray("perfect_zero_fall", dtype=str),
        generation=np.asarray(64, dtype=np.int32),
        candidate_index=np.asarray(92, dtype=np.int32),
        archive_flat_index=np.asarray(423, dtype=np.int32),
        archive_cell_indices=np.asarray([6, 0, 4], dtype=np.int32),
        archive_cell_id=np.asarray("cell_6-0-4", dtype=str),
        archive_cell_label=np.asarray("completion_bin=6|falls=0|time_bin=4", dtype=str),
    )
    raw_b = raw_dir / "gen_0071_candidate_0207_cell_6-0-3_hits6_falls0.npz"
    np.savez_compressed(
        raw_b,
        genotype=genotype,
        genotype_sha256=np.asarray("sha-b", dtype=str),
        fitness=np.asarray(2.0, dtype=np.float32),
        descriptor=np.asarray([6.0, 0.0, 3.0], dtype=np.float32),
        descriptor_names=np.asarray(["completion_bin", "falls", "time_bin"], dtype=str),
        goal_hit_count=np.asarray(6.0, dtype=np.float32),
        fall_count=np.asarray(0.0, dtype=np.float32),
        historical_elite_tier=np.asarray("perfect_zero_fall", dtype=str),
        generation=np.asarray(71, dtype=np.int32),
        candidate_index=np.asarray(207, dtype=np.int32),
        archive_flat_index=np.asarray(422, dtype=np.int32),
        archive_cell_indices=np.asarray([6, 0, 3], dtype=np.int32),
        archive_cell_id=np.asarray("cell_6-0-3", dtype=str),
        archive_cell_label=np.asarray("completion_bin=6|falls=0|time_bin=3", dtype=str),
    )

    imported = import_raw_elite_snapshots(
        root_dir=tmp_path / "elite_specimens",
        run_dir_name=run_dir_name,
        raw_paths=[raw_b, raw_a],
        run_manifest_path=run_manifest_path,
        run_resolved_config_path=run_resolved_config_path,
        volume_name="se-outputs",
        remote_paths=[
            f"/outputs/{run_dir_name}/elite_specimens/{raw_b.name}",
            f"/outputs/{run_dir_name}/elite_specimens/{raw_a.name}",
        ],
    )

    assert [row["specimen_id"] for row in imported] == [
        "gen_0064_candidate_0092_cell_6-0-4_hits6_falls0",
        "gen_0071_candidate_0207_cell_6-0-3_hits6_falls0",
    ]

    specimen_root = tmp_path / "elite_specimens" / "specimens" / run_dir_name
    selected_payload = json.loads((specimen_root / "selected_specimens.json").read_text())
    assert selected_payload["selected_specimens"][0]["selected_rank"] == 0
    assert selected_payload["selected_specimens"][1]["selected_rank"] == 1
    assert (specimen_root / "selected_genotypes.npz").exists()
    assert (specimen_root / "manifest.json").exists()
    assert (
        specimen_root
        / "specimen_gen_0064_candidate_0092_cell_6-0-4_hits6_falls0"
        / "controller_params.npz"
    ).exists()
    assert (
        tmp_path
        / "elite_specimens"
        / "runs"
        / run_dir_name
        / "resolved_config.yaml"
    ).exists()


def test_run_post_run_for_imported_specimens_remote_stages_and_dispatches_modal(monkeypatch, tmp_path):
    cfg = compose_experiment_config(preset="paper_run")
    run_dir_name = "paper_run_rect_7x2_seed42"
    run_manifest_path = tmp_path / "remote_run_manifest.json"
    run_manifest_path.write_text(
        json.dumps(
            {
                "run_id": "run-42",
                "config": {},
                "runtime": {},
                "hardware": {},
                "system": {},
                "git": {},
                "wandb": {},
                "lineage": {},
            }
        )
    )
    run_resolved_config_path = tmp_path / "resolved_config.yaml"
    run_resolved_config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True))

    genotype = np.zeros((genotype_dim_from_cfg(cfg),), dtype=np.float32)
    raw_dir = tmp_path / "raw_source"
    raw_dir.mkdir()
    raw_path = raw_dir / "gen_0064_candidate_0092_cell_6-0-4_hits6_falls0.npz"
    np.savez_compressed(
        raw_path,
        genotype=genotype,
        genotype_sha256=np.asarray("sha-a", dtype=str),
        fitness=np.asarray(1.25, dtype=np.float32),
        descriptor=np.asarray([6.0, 0.0, 4.0], dtype=np.float32),
        descriptor_names=np.asarray(["completion_bin", "falls", "time_bin"], dtype=str),
        goal_hit_count=np.asarray(6.0, dtype=np.float32),
        fall_count=np.asarray(0.0, dtype=np.float32),
        historical_elite_tier=np.asarray("perfect_zero_fall", dtype=str),
        generation=np.asarray(64, dtype=np.int32),
        candidate_index=np.asarray(92, dtype=np.int32),
        archive_flat_index=np.asarray(423, dtype=np.int32),
        archive_cell_indices=np.asarray([6, 0, 4], dtype=np.int32),
        archive_cell_id=np.asarray("cell_6-0-4", dtype=str),
        archive_cell_label=np.asarray("completion_bin=6|falls=0|time_bin=4", dtype=str),
    )
    root_dir = tmp_path / "elite_specimens"
    import_raw_elite_snapshots(
        root_dir=root_dir,
        run_dir_name=run_dir_name,
        raw_paths=[raw_path],
        run_manifest_path=run_manifest_path,
        run_resolved_config_path=run_resolved_config_path,
        volume_name="se-outputs",
        remote_paths=[f"/{run_dir_name}/elite_specimens/{raw_path.name}"],
    )

    calls: list[list[str]] = []

    def _fake_run(command, check, cwd=None, text=False, capture_output=False):
        calls.append(list(command))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="Post-run result: {'ok': True}\n",
            stderr="",
        )

    monkeypatch.setattr(elite_import_module.subprocess, "run", _fake_run)

    results = run_post_run_for_imported_specimens_remote(
        root_dir=root_dir,
        run_dir_name=run_dir_name,
        specimen_ids=None,
        assays=["tau", "native_core"],
        volume_name="se-outputs",
        detach_launch=True,
    )

    put_commands = [command for command in calls if command[:3] == ["modal", "volume", "put"]]
    run_commands = [command for command in calls if command[:3] == ["modal", "run", "scripts/modal_post_run.py"]]

    assert any(command[5] == f"/imported_elite_runs/{run_dir_name}/" for command in put_commands)
    assert any(command[5] == f"/imported_elite_specimens/{run_dir_name}/" for command in put_commands)
    assert len(run_commands) == 1
    assert "--run-dir" in run_commands[0]
    assert f"/outputs/imported_elite_runs/{run_dir_name}" in run_commands[0]
    override_value = run_commands[0][run_commands[0].index("--override") + 1]
    assert "specimens.output_root=/outputs/imported_elite_specimens" in override_value
    assert "eval.post_run.analysis_root=/outputs/imported_elite_analyses" in override_value
    assert "--detach-launch" in run_commands[0]
    assert "--assays" in run_commands[0]
    assert results[0]["specimen_id"] == "gen_0064_candidate_0092_cell_6-0-4_hits6_falls0"


def test_normalize_local_specimen_source_path_accepts_stage_root_and_specimen_dir(tmp_path):
    run_root = tmp_path / "paper_specimens" / "paper_run_rect_7x2_seed42" / "stages" / "gen_0099"
    specimen_dir = run_root / "specimen_bin_03_rank_00_cell_6-0-3_flat_0423"
    specimen_dir.mkdir(parents=True)
    (run_root / "selected_specimens.json").write_text(
        json.dumps({"selected_specimens": [{"specimen_id": "bin_03_rank_00_cell_6-0-3_flat_0423"}]})
    )

    normalized_root, inferred_specimen_id = normalize_local_specimen_source_path(run_root)
    assert normalized_root == run_root
    assert inferred_specimen_id is None

    normalized_root, inferred_specimen_id = normalize_local_specimen_source_path(specimen_dir)
    assert normalized_root == run_root
    assert inferred_specimen_id == "bin_03_rank_00_cell_6-0-3_flat_0423"
    assert infer_run_dir_name_from_specimen_root(run_root) == "paper_run_rect_7x2_seed42"


def test_run_post_run_for_local_specimen_root_remote_stages_and_dispatches_modal(monkeypatch, tmp_path):
    cfg = compose_experiment_config(preset="paper_run")
    stage_root = tmp_path / "paper_specimens" / "paper_run_rect_7x2_seed42" / "stages" / "gen_0099"
    specimen_id = "bin_03_rank_00_cell_6-0-3_flat_0423"
    specimen_dir = stage_root / f"specimen_{specimen_id}"
    specimen_dir.mkdir(parents=True)
    (stage_root / "selected_specimens.json").write_text(
        json.dumps({"selected_specimens": [{"specimen_id": specimen_id}]})
    )
    genotype = np.zeros((genotype_dim_from_cfg(cfg),), dtype=np.float32)
    np.savez_compressed(specimen_dir / "genotype.npz", genotype=genotype)
    np.savez_compressed(
        specimen_dir / "controller_params.npz",
        w_input=np.zeros((1,), dtype=np.float32),
        w_rec=np.zeros((1,), dtype=np.float32),
        w_output=np.zeros((1,), dtype=np.float32),
        bias=np.zeros((1,), dtype=np.float32),
        log_tau=np.zeros((1,), dtype=np.float32),
    )
    (specimen_dir / "config_snapshot.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    (specimen_dir / "selection_summary.json").write_text(
        json.dumps({"specimen_id": specimen_id, "selection_summary": {"specimen_id": specimen_id}})
    )
    (specimen_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "sensing_edges_specimen_manifest",
                "specimen_id": specimen_id,
                "run_id": "run-42",
                "run_dir_name": "paper_run_rect_7x2_seed42",
                "selection": {},
                "body": {},
                "controller": {},
                "run_provenance": {},
                "paths": {},
                "post_run": None,
            }
        )
    )

    calls: list[list[str]] = []

    def _fake_run(command, check, cwd=None, text=False, capture_output=False):
        calls.append(list(command))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="Post-run result: {'ok': True}\n",
            stderr="",
        )

    monkeypatch.setattr(elite_import_module.subprocess, "run", _fake_run)

    results = run_post_run_for_local_specimen_root_remote(
        source_path=specimen_dir,
        assays=["tau"],
        volume_name="se-outputs",
        detach_launch=False,
    )

    put_commands = [command for command in calls if command[:3] == ["modal", "volume", "put"]]
    run_commands = [command for command in calls if command[:3] == ["modal", "run", "scripts/modal_post_run.py"]]

    assert any(command[5] == "/rerun_local_runs/paper_run_rect_7x2_seed42/" for command in put_commands)
    assert any(command[5] == "/rerun_local_specimens/paper_run_rect_7x2_seed42/" for command in put_commands)
    assert len(run_commands) == 1
    assert "--specimen-id" in run_commands[0]
    assert specimen_id in run_commands[0]
    override_value = run_commands[0][run_commands[0].index("--override") + 1]
    assert "specimens.output_root=/outputs/rerun_local_specimens" in override_value
    assert "eval.post_run.analysis_root=/outputs/rerun_local_analyses" in override_value
    assert results[0]["determinism"]["stochastic_base_condition"] is False
    assert results[0]["determinism"]["effective_trials_per_side"] == 1


def test_run_post_run_for_local_specimen_root_remote_attached_handles_empty_captured_output(
    monkeypatch, tmp_path
):
    cfg = compose_experiment_config(preset="paper_run")
    stage_root = tmp_path / "paper_specimens" / "paper_run_rect_7x2_seed42" / "stages" / "gen_0099"
    specimen_id = "bin_03_rank_00_cell_6-0-3_flat_0423"
    specimen_dir = stage_root / f"specimen_{specimen_id}"
    specimen_dir.mkdir(parents=True)
    (stage_root / "selected_specimens.json").write_text(
        json.dumps({"selected_specimens": [{"specimen_id": specimen_id}]})
    )
    genotype = np.zeros((genotype_dim_from_cfg(cfg),), dtype=np.float32)
    np.savez_compressed(specimen_dir / "genotype.npz", genotype=genotype)
    np.savez_compressed(
        specimen_dir / "controller_params.npz",
        w_input=np.zeros((1,), dtype=np.float32),
        w_rec=np.zeros((1,), dtype=np.float32),
        w_output=np.zeros((1,), dtype=np.float32),
        bias=np.zeros((1,), dtype=np.float32),
        log_tau=np.zeros((1,), dtype=np.float32),
    )
    (specimen_dir / "config_snapshot.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    (specimen_dir / "selection_summary.json").write_text(
        json.dumps({"specimen_id": specimen_id, "selection_summary": {"specimen_id": specimen_id}})
    )
    (specimen_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "sensing_edges_specimen_manifest",
                "specimen_id": specimen_id,
                "run_id": "run-42",
                "run_dir_name": "paper_run_rect_7x2_seed42",
                "selection": {},
                "body": {},
                "controller": {},
                "run_provenance": {},
                "paths": {},
                "post_run": None,
            }
        )
    )

    def _fake_run(command, check, cwd=None, text=False, capture_output=False):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=None,
            stderr=None,
        )

    monkeypatch.setattr(elite_import_module.subprocess, "run", _fake_run)

    results = run_post_run_for_local_specimen_root_remote(
        source_path=specimen_dir,
        assays=["tau"],
        volume_name="se-outputs",
        detach_launch=False,
    )

    assert results[0]["stdout"] == ""
    assert results[0]["stderr"] == ""
