"""Focused tests for specimen-centric post-run helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import pytest

from jax_evogym.constants import CELL_SIZE

from sensing_edges.env import (
    _apply_deformation_variant,
    _deformation_variant_forces_zero_channels,
    _validate_forced_zero_deformation_channels,
)
from sensing_edges.post_run import (
    _build_bilateral_entry,
    _combined_ablation_tests,
    _controller_trace_episode_batch_size,
    _double_cliff_phase_cfg,
    _double_cliff_render_candidates,
    _double_cliff_surface_specimen_record,
    _effective_trace_trials_per_side,
    _effective_trials_per_side,
    _effective_trials_per_width,
    _first_peak_summary,
    _max_cliff_protrusion_for_trial,
    _native_phase_cfg,
    _long_distance_phase_cfg,
    _overall_multi_surface_group_records,
    _pca_bundle_additional_native_cohorts,
    _post_run_condition_is_stochastic,
    _spring_lengths_np,
    _validate_unique_batched_specimen_ids,
    _write_stats_tables,
    _task_code_geom,
    _write_json,
    ensure_specimen_artifacts,
    materialize_specimen_artifacts,
    resolve_analysis_root,
    resolve_batched_report_root,
    resolve_specimen_root,
    resolve_stored_specimen_root,
    run_post_run_for_stored_specimen,
)
from sensing_edges.runtime_config import compose_experiment_config, genotype_dim_from_cfg
from sensing_edges.nca import _apply_filter_bank


def _compose(preset: str = "paper_run"):
    return compose_experiment_config(preset=preset)


def test_resolve_specimen_root_stays_under_outputs_volume():
    assert resolve_specimen_root(Path("/outputs/paper_run_rect_7x2_seed42")) == Path(
        "/outputs/specimens/paper_run_rect_7x2_seed42"
    )


def test_resolve_analysis_root_stays_under_outputs_volume():
    assert resolve_analysis_root(
        Path("/outputs/paper_run_rect_7x2_seed42"),
        "specimen_alpha",
    ) == Path("/outputs/analyses/paper_run_rect_7x2_seed42/specimen_alpha")


def test_resolve_batched_report_root_stays_under_analysis_outputs_volume():
    roots = [
        Path("/outputs/analyses/run_a/specimen_alpha"),
        Path("/outputs/analyses/run_b/specimen_beta"),
    ]

    assert resolve_batched_report_root(roots, batch_label="paper_deadline") == Path(
        "/outputs/analyses/_batched_reports/paper_deadline"
    )


def test_validate_unique_batched_specimen_ids_rejects_duplicates():
    with pytest.raises(ValueError, match="unique specimen_id"):
        _validate_unique_batched_specimen_ids(
            [
                {"run_dir": "/outputs/run_a", "specimen_id": "specimen_alpha"},
                {"run_dir": "/outputs/run_b", "specimen_id": "specimen_alpha"},
            ]
        )


def test_resolve_stored_specimen_root_prefers_stage_outputs_when_root_is_empty(tmp_path):
    run_dir = tmp_path / "outputs" / "paper_run_rect_7x2_seed42"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"summary": {"completed_generations": 100}}))
    specimen_root = resolve_specimen_root(run_dir, output_root="specimens")
    stage_root = specimen_root / "stages" / "gen_0099"
    stage_root.mkdir(parents=True)
    (stage_root / "selected_specimens.json").write_text(json.dumps({"selected_specimens": []}))

    assert resolve_stored_specimen_root(run_dir, output_root="specimens") == stage_root


def test_long_distance_phase_cfg_allows_10_5_pair_on_widened_grid():
    cfg = _compose("paper_run")
    phase_cfg = _long_distance_phase_cfg(cfg, goal_distance=10, cliff_distance=5)

    entry = _build_bilateral_entry(
        phase_cfg,
        goal_distance=10,
        cliff_distance=5,
        goal_side="left",
    )

    assert entry.grid.shape[1] == 32
    assert int(phase_cfg.train.max_steps) == 6000


def test_surface_phase_cfgs_use_surface_specific_step_budgets():
    cfg = _compose("paper_run")

    native_phase_cfg = _native_phase_cfg(cfg)
    double_cliff_phase_cfg = _double_cliff_phase_cfg(cfg)
    long_phase_cfg = _long_distance_phase_cfg(cfg, goal_distance=8, cliff_distance=4)

    assert int(native_phase_cfg.train.max_steps) == 4000
    assert int(double_cliff_phase_cfg.train.max_steps) == 6000
    assert int(long_phase_cfg.train.max_steps) == 5000


def test_spring_lengths_np_derives_lengths_for_state_space_trace():
    positions = np.asarray(
        [
            [[0.0, 0.0], [3.0, 4.0], [3.0, 0.0]],
            [[0.0, 0.0], [0.0, 6.0], [8.0, 0.0]],
        ],
        dtype=np.float32,
    )

    lengths = _spring_lengths_np(
        positions,
        a_idx=np.asarray([0, 1], dtype=np.int32),
        b_idx=np.asarray([1, 2], dtype=np.int32),
    )

    np.testing.assert_allclose(
        lengths,
        np.asarray([[5.0, 4.0], [6.0, 10.0]], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_controller_trace_episode_batch_size_uses_full_batch_on_a100():
    runtime_resolution = type("RuntimeResolution", (), {"gpu_profile": "a100_40gb"})()

    assert _controller_trace_episode_batch_size(
        runtime_resolution=runtime_resolution,
        requested_batch_size=4,
        total_entries=40,
    ) == 40


def test_deterministic_paper_run_collapses_repeated_trials():
    cfg = _compose("paper_run")

    assert _post_run_condition_is_stochastic(cfg) is False
    assert _effective_trials_per_side(cfg) == 1
    assert _effective_trials_per_width(cfg) == 1
    assert _effective_trace_trials_per_side(cfg) == 1


def test_stochastic_deformation_conditions_use_configured_caps():
    cfg = _compose("paper_run")

    capped_per_side = int(cfg.eval.post_run.stochastic_trials_per_side)
    capped_per_width = int(cfg.eval.post_run.double_cliff.stochastic_trials_per_width)

    assert _post_run_condition_is_stochastic(cfg, deformation_condition="noise_only") is True
    assert _effective_trials_per_side(cfg, deformation_condition="noise_only") == capped_per_side
    assert _effective_trials_per_width(cfg, deformation_condition="noise_only") == capped_per_width
    assert _effective_trace_trials_per_side(cfg, deformation_condition="noise_only") == capped_per_side

    assert _post_run_condition_is_stochastic(cfg, deformation_condition="shuffled") is True
    assert _effective_trials_per_side(cfg, deformation_condition="shuffled") == capped_per_side
    assert _effective_trials_per_width(cfg, deformation_condition="shuffled") == capped_per_width


def test_default_trace_config_includes_deformation_off_pca_cohort():
    cfg = _compose("paper_run")

    cohorts = _pca_bundle_additional_native_cohorts(cfg.eval.post_run.trace)

    assert cohorts == [
        {
            "cohort_id": "deformation_off_native_core",
            "cohort_label": "Deformation Off Native Core",
            "goal_distance": 4,
            "cliff_distance": 2,
            "deformation_condition": "deformation_off",
            "signaling_condition": "intact",
        }
    ]


def test_double_cliff_render_candidates_accept_legacy_assay_width_metadata():
    class _EpisodeResults(NamedTuple):
        steps_alive: np.ndarray
        cliff_fell: np.ndarray

    episode_results = _EpisodeResults(
        steps_alive=np.asarray([10, 10], dtype=np.int32),
        cliff_fell=np.asarray([False, False], dtype=np.bool_),
    )
    traces = np.asarray([[1.0], [2.0]], dtype=np.float32)
    envs = [
        SimpleNamespace(render_info={"episode": 0}),
        SimpleNamespace(render_info={"episode": 1}),
    ]
    entries = [
        SimpleNamespace(metadata={"assay_width": 2, "trial_repeat_index": 0}),
        SimpleNamespace(metadata={"assay_width": 4, "trial_repeat_index": 0}),
    ]

    candidates = _double_cliff_render_candidates(
        episode_results=episode_results,
        traces=traces,
        envs=envs,
        entries=entries,
        fitnesses=np.asarray([1.0, 2.0], dtype=np.float32),
    )

    assert [candidate["group_key"] for candidate in candidates] == ["width_2", "width_4"]
    assert [candidate["filename"] for candidate in candidates] == ["width_2.gif", "width_4.gif"]


def test_max_cliff_protrusion_only_counts_survived_approached_trials():
    assert _max_cliff_protrusion_for_trial(
        min_edge_clearance=-0.25,
        cliff_fell=False,
        first_cliff_vicinity_step=17,
    ) == pytest.approx(0.25)
    assert _max_cliff_protrusion_for_trial(
        min_edge_clearance=0.25,
        cliff_fell=False,
        first_cliff_vicinity_step=17,
    ) == 0.0
    assert _max_cliff_protrusion_for_trial(
        min_edge_clearance=-0.25,
        cliff_fell=True,
        first_cliff_vicinity_step=17,
    ) is None
    assert _max_cliff_protrusion_for_trial(
        min_edge_clearance=-0.25,
        cliff_fell=False,
        first_cliff_vicinity_step=None,
    ) is None


def test_overall_multi_surface_group_records_pool_native_and_double_cliff_metrics():
    df = pd.DataFrame(
        [
            {
                "surface_id": "native_core",
                "condition_id": "intact",
                "condition_label": "Intact",
                "success": 1.0,
                "fell": 0.0,
                "fitness": 10.0,
                "max_cliff_protrusion": 0.25,
                "time_to_arrest": 12.0,
            },
            {
                "surface_id": "long_distance_generalization",
                "condition_id": "intact",
                "condition_label": "Intact",
                "success": 0.0,
                "fell": 0.0,
                "fitness": 8.0,
                "max_cliff_protrusion": 0.0,
                "time_to_arrest": 18.0,
            },
            {
                "surface_id": "double_cliff",
                "condition_id": "intact",
                "condition_label": "Intact",
                "fell": 1.0,
                "survived": 0.0,
                "side_switch_count": 2.0,
            },
        ]
    )

    records = _overall_multi_surface_group_records(
        df,
        specimen_id="specimen_alpha",
        assay_family="deformation_ablation",
        group_field="condition_id",
        group_label_field="condition_label",
    )

    assert len(records) == 1
    record = records[0]
    assert record["condition_id"] == "intact"
    assert record["condition_label"] == "Intact"
    assert record["surface_count"] == 3
    assert record["surface_ids"] == [
        "native_core",
        "long_distance_generalization",
        "double_cliff",
    ]
    assert record["success_rate"]["n"] == 2
    assert record["success_rate"]["mean"] == pytest.approx(0.5)
    assert record["fall_rate"]["n"] == 3
    assert record["fall_rate"]["mean"] == pytest.approx(1.0 / 3.0)
    assert record["mean_max_cliff_protrusion"]["n"] == 2
    assert record["mean_max_cliff_protrusion"]["mean"] == pytest.approx(0.125)
    assert record["survival_rate"]["n"] == 1
    assert record["survival_rate"]["mean"] == 0.0
    assert record["mean_side_switch_count"]["n"] == 1
    assert record["mean_side_switch_count"]["mean"] == 2.0
    assert record["native_outcome_breakdown"]["trial_count"] == 2
    assert record["double_cliff_outcome_breakdown"]["trial_count"] == 1


def test_double_cliff_surface_specimen_record_handles_summary_only_rows():
    rows = pd.DataFrame(
        [
            {"survived": 1.0, "fell": 0.0, "width": 2},
            {"survived": 0.0, "fell": 1.0, "width": 4},
        ]
    )

    record = _double_cliff_surface_specimen_record(
        specimen_id="specimen_alpha",
        assay_family="deformation_ablation",
        surface_id="double_cliff",
        rows=rows,
    )

    assert record["survival_rate"]["n"] == 2
    assert record["survival_rate"]["mean"] == pytest.approx(0.5)
    assert record["mean_side_switch_count"]["n"] == 0
    assert record["left_first_side_bias"] is None
    assert record["right_first_side_bias"] is None
    assert record["symmetricity_score"] is None


def test_combined_ablation_quick_defaults_parse():
    cfg = _compose("paper_run")

    tests = _combined_ablation_tests(cfg)

    assert tests == [
        {
            "test_id": "deformation_off__sobel_x_off",
            "label": "Deformation Off + Sobel-X Off",
            "deformation_condition": "deformation_off",
            "signaling_condition": "sobel_x_off",
        },
        {
            "test_id": "deformation_off__sobel_y_off",
            "label": "Deformation Off + Sobel-Y Off",
            "deformation_condition": "deformation_off",
            "signaling_condition": "sobel_y_off",
        },
        {
            "test_id": "deformation_off__laplacian_off",
            "label": "Deformation Off + Laplacian Off",
            "deformation_condition": "deformation_off",
            "signaling_condition": "laplacian_off",
        },
        {
            "test_id": "deformation_off__identity_comm_off",
            "label": "Deformation Off + Identity Off",
            "deformation_condition": "deformation_off",
            "signaling_condition": "identity_comm_off",
        },
        {
            "test_id": "deformation_off__all_comm_off",
            "label": "Deformation Off + All Communication Off",
            "deformation_condition": "deformation_off",
            "signaling_condition": "all_comm_off",
        },
    ]


def test_materialize_specimen_artifacts_creates_per_specimen_tree(tmp_path):
    cfg = _compose("paper_run")
    run_dir = tmp_path / "outputs" / "paper_run_rect_7x2_seed42"
    run_dir.mkdir(parents=True)
    run_manifest_path = run_dir / "manifest.json"
    run_manifest_path.write_text(
        json.dumps(
            {
                "run_id": "run-123",
                "config": {"resolved_config_path": str(run_dir / "resolved_config.yaml")},
                "runtime": {"gpu_profile": "a100_40gb"},
                "hardware": {"backend": "cpu"},
                "system": {"platform": "test"},
                "git": {"commit": "deadbeef"},
                "wandb": {"run_id": "wandb-123"},
                "lineage": {"mode": "root"},
            }
        )
    )
    resolved_config_path = run_dir / "resolved_config.yaml"
    resolved_config_path.write_text("seed: 42\n")
    specimen_root = resolve_specimen_root(run_dir, output_root="specimens")
    genotype = np.zeros((genotype_dim_from_cfg(cfg),), dtype=np.float32)

    payload = materialize_specimen_artifacts(
        specimen_root=specimen_root,
        specimen_id="specimen_alpha",
        selected_rank=0,
        selection_summary={"goal_hit_count": 12.0, "fall_count": 0.0},
        selection_rule={"per_completion_bin": 3},
        genotype=genotype,
        cfg=cfg,
        run_manifest_path=run_manifest_path,
        run_resolved_config_path=resolved_config_path,
        generation=100,
    )

    specimen_dir = specimen_root / "specimen_specimen_alpha"
    assert specimen_dir.exists()
    assert (specimen_dir / "genotype.npz").exists()
    assert (specimen_dir / "controller_params.npz").exists()
    assert (specimen_dir / "config_snapshot.yaml").exists()
    assert (specimen_dir / "selection_summary.json").exists()
    assert (specimen_dir / "manifest.json").exists()
    assert payload["specimen_dir"] == str(specimen_dir)


def test_ensure_specimen_artifacts_hydrates_legacy_selected_specimen(tmp_path):
    cfg = _compose("paper_run")
    run_dir = tmp_path / "outputs" / "paper_run_rect_7x2_seed42"
    run_dir.mkdir(parents=True)
    run_manifest_path = run_dir / "manifest.json"
    run_manifest_path.write_text(
        json.dumps(
            {
                "run_id": "run-legacy",
                "summary": {"completed_generations": 100},
                "config": {"resolved_config_path": str(run_dir / "resolved_config.yaml")},
                "runtime": {"gpu_profile": "a100_40gb"},
                "hardware": {"backend": "cpu"},
                "system": {"platform": "test"},
                "git": {"commit": "deadbeef"},
                "wandb": {"run_id": "wandb-legacy"},
                "lineage": {"mode": "root"},
            }
        )
    )
    resolved_config_path = run_dir / "resolved_config.yaml"
    resolved_config_path.write_text("seed: 42\n")
    specimen_root = resolve_specimen_root(run_dir, output_root="specimens")
    specimen_root.mkdir(parents=True)
    (specimen_root / "selected_specimens.json").write_text(
        json.dumps(
            {
                "selection_rule": {"per_completion_bin": 3},
                "selected_specimens": [
                    {
                        "completion_bin": 2,
                        "source": "archive",
                        "source_index": 0,
                        "archive_flat_index": 7,
                        "archive_cell_id": "cell_1-2-3",
                        "goal_hit_count": 12.0,
                        "fall_count": 0.0,
                    }
                ],
            }
        )
    )
    np.savez_compressed(
        specimen_root / "selected_genotypes.npz",
        genotypes=np.zeros((1, genotype_dim_from_cfg(cfg)), dtype=np.float32),
    )

    row, specimen_dir = ensure_specimen_artifacts(
        cfg=cfg,
        run_dir=run_dir,
        specimen_root=specimen_root,
        specimen_rank=0,
    )

    assert row["specimen_id"] == "bin_02_rank_00_cell_1-2-3_flat_0007"
    assert specimen_dir == specimen_root / "specimen_bin_02_rank_00_cell_1-2-3_flat_0007"
    assert (specimen_dir / "genotype.npz").exists()
    assert (specimen_dir / "controller_params.npz").exists()
    updated_payload = json.loads((specimen_root / "selected_specimens.json").read_text())
    assert updated_payload["selected_specimens"][0]["specimen_id"] == row["specimen_id"]


def test_run_post_run_persists_partial_failure_state(tmp_path, monkeypatch):
    cfg = _compose("paper_run")
    run_dir = tmp_path / "outputs" / "paper_run_rect_7x2_seed42"
    run_dir.mkdir(parents=True)
    run_manifest_path = run_dir / "manifest.json"
    run_manifest_path.write_text(
        json.dumps(
            {
                "run_id": "run-post-run-failure",
                "summary": {"completed_generations": 100},
                "config": {"resolved_config_path": str(run_dir / "resolved_config.yaml")},
                "runtime": {"gpu_profile": "a100_40gb"},
                "hardware": {"backend": "cpu"},
                "system": {"platform": "test"},
                "git": {"commit": "deadbeef"},
                "wandb": {"run_id": "wandb-post-run"},
                "lineage": {"mode": "root"},
            }
        )
    )
    resolved_config_path = run_dir / "resolved_config.yaml"
    resolved_config_path.write_text("seed: 42\n")
    specimen_root = resolve_specimen_root(run_dir, output_root="specimens")
    genotype = np.zeros((genotype_dim_from_cfg(cfg),), dtype=np.float32)
    artifact_paths = materialize_specimen_artifacts(
        specimen_root=specimen_root,
        specimen_id="specimen_alpha",
        selected_rank=0,
        selection_summary={"goal_hit_count": 12.0, "fall_count": 0.0},
        selection_rule={"per_completion_bin": 3},
        genotype=genotype,
        cfg=cfg,
        run_manifest_path=run_manifest_path,
        run_resolved_config_path=resolved_config_path,
        generation=99,
    )
    (specimen_root / "selected_specimens.json").write_text(
        json.dumps(
            {
                "selection_rule": {"per_completion_bin": 3},
                "selected_specimens": [
                    {
                        "specimen_id": "specimen_alpha",
                        "selected_rank": 0,
                        "goal_hit_count": 12.0,
                        "fall_count": 0.0,
                        **artifact_paths,
                    }
                ],
            }
        )
    )

    def _fake_tau_analysis(*, specimen_dir, **_kwargs):
        assay_dir = specimen_dir / "tau"
        assay_dir.mkdir(parents=True, exist_ok=True)
        (assay_dir / "tau_summary.json").write_text("{}")
        return assay_dir

    def _fake_pca_bundle(**_kwargs):
        raise RuntimeError("synthetic pca failure")

    monkeypatch.setattr("sensing_edges.post_run._run_tau_analysis", _fake_tau_analysis)
    monkeypatch.setattr("sensing_edges.post_run._run_pca_bundle", _fake_pca_bundle)
    progress_events: list[str] = []

    results = run_post_run_for_stored_specimen(
        cfg=cfg,
        runtime_resolution=object(),
        run_dir=run_dir,
        specimen_root=specimen_root,
        specimen_rank=0,
        assays={"tau", "pca_bundle"},
        raise_on_failure=False,
        on_progress=lambda: progress_events.append("tick"),
    )

    assert results.status == "failed"
    assert results.failed_assay == "pca_bundle"
    assert results.ran_assays == ["tau"]
    assert len(progress_events) >= 4
    analysis_root = resolve_analysis_root(run_dir, "specimen_alpha")
    index_payload = json.loads(results.index_path.read_text())
    assert index_payload["status"] == "failed"
    assert index_payload["completed_assays"] == ["tau"]
    assert index_payload["failed_assay"] == "pca_bundle"
    assert index_payload["current_assay"] is None
    assert index_payload["assay_progress"]["tau"]["status"] == "completed"
    assert index_payload["assay_progress"]["pca_bundle"]["status"] == "failed"
    assert index_payload["assay_progress"]["pca_bundle"]["assay_dir"] == str(
        analysis_root / "pca_bundle"
    )
    assert "synthetic pca failure" in index_payload["error"]["message"]
    pointer_payload = json.loads((specimen_root / "post_run" / "specimen_alpha.json").read_text())
    assert pointer_payload["analysis_root"] == str(analysis_root)
    assert pointer_payload["analysis_index_path"] == str(results.index_path)

    specimen_manifest = json.loads((Path(artifact_paths["manifest_path"])).read_text())
    assert specimen_manifest["post_run"]["status"] == "failed"
    assert specimen_manifest["post_run"]["completed_assays"] == ["tau"]
    assert specimen_manifest["post_run"]["failed_assay"] == "pca_bundle"
    assert specimen_manifest["post_run"]["current_assay"] is None
    assert specimen_manifest["post_run"]["assay_progress"]["tau"]["status"] == "completed"
    assert specimen_manifest["post_run"]["assay_progress"]["pca_bundle"]["status"] == "failed"
    assert specimen_manifest["post_run"]["assay_progress"]["pca_bundle"]["assay_dir"] == str(
        analysis_root / "pca_bundle"
    )
    assert specimen_manifest["post_run"]["analysis_root"] == str(analysis_root)


def test_run_post_run_can_reraise_after_persisting_failure(tmp_path, monkeypatch):
    cfg = _compose("paper_run")
    run_dir = tmp_path / "outputs" / "paper_run_rect_7x2_seed42"
    run_dir.mkdir(parents=True)
    run_manifest_path = run_dir / "manifest.json"
    run_manifest_path.write_text(
        json.dumps(
            {
                "run_id": "run-post-run-reraise",
                "summary": {"completed_generations": 100},
                "config": {"resolved_config_path": str(run_dir / "resolved_config.yaml")},
                "runtime": {"gpu_profile": "a100_40gb"},
                "hardware": {"backend": "cpu"},
                "system": {"platform": "test"},
                "git": {"commit": "deadbeef"},
                "wandb": {"run_id": "wandb-post-run"},
                "lineage": {"mode": "root"},
            }
        )
    )
    resolved_config_path = run_dir / "resolved_config.yaml"
    resolved_config_path.write_text("seed: 42\n")
    specimen_root = resolve_specimen_root(run_dir, output_root="specimens")
    genotype = np.zeros((genotype_dim_from_cfg(cfg),), dtype=np.float32)
    artifact_paths = materialize_specimen_artifacts(
        specimen_root=specimen_root,
        specimen_id="specimen_alpha",
        selected_rank=0,
        selection_summary={"goal_hit_count": 12.0, "fall_count": 0.0},
        selection_rule={"per_completion_bin": 3},
        genotype=genotype,
        cfg=cfg,
        run_manifest_path=run_manifest_path,
        run_resolved_config_path=resolved_config_path,
        generation=99,
    )
    (specimen_root / "selected_specimens.json").write_text(
        json.dumps(
            {
                "selection_rule": {"per_completion_bin": 3},
                "selected_specimens": [
                    {
                        "specimen_id": "specimen_alpha",
                        "selected_rank": 0,
                        **artifact_paths,
                    }
                ],
            }
        )
    )

    monkeypatch.setattr(
        "sensing_edges.post_run._run_tau_analysis",
        lambda **_kwargs: (_kwargs["specimen_dir"] / "tau"),
    )
    monkeypatch.setattr(
        "sensing_edges.post_run._run_pca_bundle",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        run_post_run_for_stored_specimen(
            cfg=cfg,
            runtime_resolution=object(),
            run_dir=run_dir,
            specimen_root=specimen_root,
            specimen_rank=0,
            assays={"tau", "pca_bundle"},
            raise_on_failure=True,
        )

    analysis_root = resolve_analysis_root(run_dir, "specimen_alpha")
    index_payload = json.loads((analysis_root / "index.json").read_text())
    assert index_payload["status"] == "failed"
    assert index_payload["failed_assay"] == "pca_bundle"
    assert index_payload["completed_assays"] == ["tau"]
    assert index_payload["current_assay"] is None
    assert index_payload["assay_progress"]["tau"]["status"] == "completed"
    assert index_payload["assay_progress"]["pca_bundle"]["status"] == "failed"
    assert index_payload["assay_progress"]["pca_bundle"]["assay_dir"] == str(
        analysis_root / "pca_bundle"
    )


def test_run_post_run_archives_stale_assay_outputs_before_rerun(tmp_path, monkeypatch):
    cfg = _compose("paper_run")
    run_dir = tmp_path / "outputs" / "paper_run_rect_7x2_seed42"
    run_dir.mkdir(parents=True)
    run_manifest_path = run_dir / "manifest.json"
    run_manifest_path.write_text(
        json.dumps(
            {
                "run_id": "run-post-run-archive",
                "summary": {"completed_generations": 100},
                "config": {"resolved_config_path": str(run_dir / "resolved_config.yaml")},
                "runtime": {"gpu_profile": "a100_40gb"},
                "hardware": {"backend": "cpu"},
                "system": {"platform": "test"},
                "git": {"commit": "deadbeef"},
                "wandb": {"run_id": "wandb-post-run"},
                "lineage": {"mode": "root"},
            }
        )
    )
    resolved_config_path = run_dir / "resolved_config.yaml"
    resolved_config_path.write_text("seed: 42\n")
    specimen_root = resolve_specimen_root(run_dir, output_root="specimens")
    genotype = np.zeros((genotype_dim_from_cfg(cfg),), dtype=np.float32)
    artifact_paths = materialize_specimen_artifacts(
        specimen_root=specimen_root,
        specimen_id="specimen_alpha",
        selected_rank=0,
        selection_summary={"goal_hit_count": 12.0, "fall_count": 0.0},
        selection_rule={"per_completion_bin": 3},
        genotype=genotype,
        cfg=cfg,
        run_manifest_path=run_manifest_path,
        run_resolved_config_path=resolved_config_path,
        generation=99,
    )
    (specimen_root / "selected_specimens.json").write_text(
        json.dumps(
            {
                "selection_rule": {"per_completion_bin": 3},
                "selected_specimens": [
                    {
                        "specimen_id": "specimen_alpha",
                        "selected_rank": 0,
                        **artifact_paths,
                    }
                ],
            }
        )
    )
    analysis_root = resolve_analysis_root(run_dir, "specimen_alpha")
    analysis_root.mkdir(parents=True, exist_ok=True)
    stale_tau_dir = analysis_root / "tau"
    stale_tau_dir.mkdir(parents=True, exist_ok=True)
    (stale_tau_dir / "old.txt").write_text("stale")
    stale_index_path = analysis_root / "index.json"
    stale_index_path.write_text('{"status":"old"}\n')
    (analysis_root / "manifest.json").write_text('{"kind":"old"}\n')

    def _fake_tau_analysis(*, specimen_dir, **_kwargs):
        assay_dir = specimen_dir / "tau"
        assay_dir.mkdir(parents=True, exist_ok=True)
        (assay_dir / "tau_summary.json").write_text("{}")
        return assay_dir

    monkeypatch.setattr("sensing_edges.post_run._run_tau_analysis", _fake_tau_analysis)

    results = run_post_run_for_stored_specimen(
        cfg=cfg,
        runtime_resolution=object(),
        run_dir=run_dir,
        specimen_root=specimen_root,
        specimen_rank=0,
        assays={"tau"},
        raise_on_failure=False,
    )

    assert results.status == "completed"
    archived_tau_dirs = list((analysis_root / "_previous").glob("*/tau"))
    assert archived_tau_dirs
    assert (archived_tau_dirs[0] / "old.txt").exists()
    archived_index_paths = list((analysis_root / "_previous").glob("*/index.json"))
    assert archived_index_paths
    index_payload = json.loads(results.index_path.read_text())
    assert index_payload["archived_outputs"]["tau"].endswith("/tau")
    assert index_payload["archived_outputs"]["post_run_index"].endswith("/index.json")
    assert index_payload["archived_outputs"]["analysis_manifest"].endswith("/manifest.json")
    assert index_payload["assay_progress"]["tau"]["assay_dir"] == str(analysis_root / "tau")


def test_write_json_serializes_omegaconf_payloads(tmp_path):
    output_path = tmp_path / "assay_manifest.json"
    payload = {
        "pairs": [OmegaConf.create({"goal_distance": 2, "cliff_distance": 1})],
        "post_run": OmegaConf.create({"enabled": True, "max_steps": 5000}),
    }

    _write_json(output_path, payload)

    written = json.loads(output_path.read_text())
    assert written == {
        "pairs": [{"goal_distance": 2, "cliff_distance": 1}],
        "post_run": {"enabled": True, "max_steps": 5000},
    }


def test_write_json_serializes_nested_omegaconf_lists_and_numpy_scalars(tmp_path):
    output_path = tmp_path / "nested_assay_manifest.json"
    payload = {
        "pairs": list(
            OmegaConf.create(
                [
                    {"goal_distance": 2, "cliff_distance": 1},
                    {"goal_distance": 4, "cliff_distance": 2},
                ]
            )
        ),
        "flags": {
            "success": np.bool_(True),
            "count": np.int64(3),
        },
    }

    _write_json(output_path, payload)

    written = json.loads(output_path.read_text())
    assert written == {
        "pairs": [
            {"goal_distance": 2, "cliff_distance": 1},
            {"goal_distance": 4, "cliff_distance": 2},
        ],
        "flags": {
            "success": True,
            "count": 3,
        },
    }


def test_write_stats_tables_writes_flat_csv_outputs(tmp_path):
    assay_dir = tmp_path / "analysis" / "native_core"
    outputs = _write_stats_tables(
        assay_dir=assay_dir,
        condition_records=[
            {
                "distance_pair": "(2,1)",
                "success_rate": {"mean": 1.0, "bootstrap_ci95": [1.0, 1.0]},
                "outcome_breakdown": {
                    "success_count": 40,
                    "fall_count": 0,
                    "no_hit_no_fall_count": 0,
                },
            }
        ],
        specimen_records=[{"specimen_id": "specimen_alpha", "trial_count": 120}],
        inventory_records=[{"assay_family": "native_core", "surface_id": "native_core"}],
    )

    assert outputs["condition_summary"]["csv"].endswith("condition_summary.csv")
    assert (assay_dir / "stats" / "condition_summary.csv").exists()
    assert (assay_dir / "stats" / "specimen_summary.csv").exists()
    assert (assay_dir / "stats" / "assay_inventory.csv").exists()


def test_write_json_is_atomic_on_serialization_failure(tmp_path):
    output_path = tmp_path / "assay_manifest.json"
    output_path.write_text('{"status": "previous"}\n')

    with pytest.raises(TypeError, match="not JSON serializable"):
        _write_json(output_path, {"bad": {1, 2, 3}})

    assert output_path.read_text() == '{"status": "previous"}\n'


def test_apply_deformation_variant_supports_mechanistic_modes():
    deformation = jnp.array(
        [
            [1.2, 1.4, 0.8, 1.1],
            [0.9, 1.3, 1.5, 0.7],
        ],
        dtype=jnp.float32,
    )
    common_kwargs = {
        "noise_sigma_signed_delta": 0.05,
        "shuffle_mode": "episode_fixed",
        "variant_phase_key": jax.random.PRNGKey(0),
        "variant_episode_key": jax.random.PRNGKey(1),
    }

    absolute_only = np.asarray(
        _apply_deformation_variant(deformation, variant="absolute_only", **common_kwargs)
    )
    compliance_only = np.asarray(
        _apply_deformation_variant(deformation, variant="compliance_only", **common_kwargs)
    )
    reversed_variant = np.asarray(
        _apply_deformation_variant(deformation, variant="deformation_reversed", **common_kwargs)
    )
    noise_only = np.asarray(
        _apply_deformation_variant(deformation, variant="noise_only", **common_kwargs)
    )

    np.testing.assert_allclose(absolute_only[:, 2:], 1.0)
    np.testing.assert_allclose(compliance_only[:, :2], 1.0)
    np.testing.assert_allclose(
        reversed_variant - 1.0,
        -(np.asarray(deformation) - 1.0),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        noise_only,
        np.asarray(
            jax.random.normal(
                common_kwargs["variant_phase_key"],
                shape=deformation.shape,
                dtype=jnp.float32,
            )
            * jnp.float32(common_kwargs["noise_sigma_signed_delta"])
            + jnp.float32(1.0)
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_deformation_off_is_marked_for_explicit_controller_zeroing():
    assert _deformation_variant_forces_zero_channels("deformation_off")
    assert _deformation_variant_forces_zero_channels("zero")
    assert not _deformation_variant_forces_zero_channels("intact")
    assert not _deformation_variant_forces_zero_channels("noisy")
    assert not _deformation_variant_forces_zero_channels("noise_only")


def test_forced_zero_deformation_validation_allows_neutral_tensor():
    deformation = jnp.ones((2, 4), dtype=jnp.float32)
    _validate_forced_zero_deformation_channels(
        deformation,
        variant="deformation_off",
    )


def test_forced_zero_deformation_validation_rejects_nonzero_channels():
    deformation = jnp.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0001, 1.0],
        ],
        dtype=jnp.float32,
    )

    with pytest.raises(ValueError, match="produced non-zero signed deformation channels"):
        _validate_forced_zero_deformation_channels(
            deformation,
            variant="deformation_off",
        )


def test_apply_filter_bank_supports_signaling_variants():
    comm = jnp.arange(9, dtype=jnp.float32).reshape(3, 3, 1)
    intact = np.asarray(_apply_filter_bank(comm, signaling_variant="intact"))
    sobel_x_off = np.asarray(_apply_filter_bank(comm, signaling_variant="sobel_x_off"))
    sign_flip = np.asarray(_apply_filter_bank(comm, signaling_variant="sobel_x_sign_flip"))
    all_comm_off = np.asarray(_apply_filter_bank(comm, signaling_variant="all_comm_off"))

    np.testing.assert_allclose(sobel_x_off[..., 0], 0.0)
    np.testing.assert_allclose(sign_flip[..., 0], -intact[..., 0])
    np.testing.assert_allclose(all_comm_off, 0.0)


def test_first_peak_summary_detects_initial_cliffward_excursion():
    unit = float(CELL_SIZE)
    direction, first_peak_x, first_peak_goalward_x = _first_peak_summary(
        com_x=np.asarray([0.0, -0.8 * unit, -1.8 * unit, -1.1 * unit], dtype=np.float32),
        init_com_x=0.0,
        goal_direction=1,
        turn_motion_threshold=0.5 * unit,
    )

    assert direction == "cliff_first"
    assert first_peak_x == pytest.approx(-1.8 * unit)
    assert first_peak_goalward_x == pytest.approx(-1.8 * unit)


def test_task_code_geom_hits_cliff_and_goal_breakpoints():
    unit = float(CELL_SIZE)
    code, reached_frontier, frontier_idx = _task_code_geom(
        back_x=np.asarray([4.0, 3.0, 2.0, 2.0, 4.0, 6.0], dtype=np.float32) * unit,
        front_x=np.asarray([5.0, 4.0, 3.0, 3.0, 5.0, 7.0], dtype=np.float32) * unit,
        goal_direction=1,
        cliff_side="left",
        cliff_edge_x=2,
        goal_block_x=7,
        init_back_x=4.0 * unit,
        init_front_x=5.0 * unit,
    )

    assert reached_frontier is True
    assert frontier_idx == 2
    np.testing.assert_allclose(code[[0, 1, 2, 4, 5]], [0.0, 0.5, 1.0, 1.5, 2.0], atol=1e-6)


def test_apply_filter_bank_rejects_incompatible_signaling_variant():
    comm = jnp.arange(9, dtype=jnp.float32).reshape(3, 3, 1)
    with pytest.raises(ValueError, match="requires active communication filter"):
        _apply_filter_bank(
            comm,
            signaling_variant="sobel_x_off",
            active_comm_filters=("laplacian", "identity"),
        )
