"""Tests for Modal orchestration helpers in paper_protocol.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from sensing_edges.checkpoint import payload_done_path


def _load_module():
    path = Path(__file__).parent.parent / "scripts" / "paper_protocol.py"
    spec = importlib.util.spec_from_file_location("sensing_edges_paper_protocol", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


module = _load_module()


class TestPaperProtocolHelpers:
    def test_seed_from_run_dir_name(self):
        assert module._seed_from_run_dir_name("paper_run_rect_7x2_seed42") == 42

    def test_seed_from_run_dir_name_requires_suffix(self):
        with pytest.raises(ValueError):
            module._seed_from_run_dir_name("paper_run_rect_7x2")

    def test_preset_from_run_dir_name(self):
        assert module._preset_from_run_dir_name("paper_run_rect_7x2_seed42") == "paper_run_rect_7x2"

    def test_schedule_for_run_uses_explicit_override(self, tmp_path):
        run_dir = tmp_path / "paper_run_rect_7x2_seed42"
        run_dir.mkdir()
        assert module._schedule_for_run(
            run_dir,
            schedule="terrain_suites/seed_999",
            schedules_dir="terrain_suites",
        ) == "terrain_suites/seed_999"

    def test_schedule_for_run_derives_seed_path(self, tmp_path):
        run_dir = tmp_path / "paper_run_rect_7x2_seed42"
        run_dir.mkdir()
        schedules_dir = tmp_path / "terrain_suites"
        expected = schedules_dir / "seed_42"
        expected.mkdir(parents=True)

        assert module._schedule_for_run(
            run_dir,
            schedule=None,
            schedules_dir=str(schedules_dir),
        ) == str(expected)

    def test_discover_available_schedule_seeds_empty_when_dir_missing(self, tmp_path):
        seeds = module._discover_available_schedule_seeds(str(tmp_path / "missing"))
        assert seeds == []

    def test_discover_available_schedule_seeds_from_variant_dirs(self, tmp_path):
        schedules_dir = tmp_path / "terrain_suites"
        schedules_dir.mkdir()
        (schedules_dir / "seed_42_curriculum_zone10to19_extend24_gap2_30").mkdir()
        (schedules_dir / "seed_137_single_gap2_zone10to24_30").mkdir()
        (schedules_dir / "seed_42_audit10").mkdir()
        (schedules_dir / "notes").mkdir()

        seeds = module._discover_available_schedule_seeds(str(schedules_dir))
        assert seeds == [42, 137]

    def test_variant_schedule_path(self, tmp_path):
        assert module._variant_schedule_path(42, "curriculum_zone10to19_extend24_gap2_30", str(tmp_path)) == (
            tmp_path / "seed_42_curriculum_zone10to19_extend24_gap2_30"
        )

    def test_parse_body_spec_for_grown_and_comb_variants(self):
        grown = module._parse_body_spec("grown:7x3")
        comb = module._parse_body_spec("comb")
        comb_7x3 = module._parse_body_spec("comb_7x3")

        assert grown["label"] == "grown_7x3"
        assert grown["body"] == "grown"
        assert grown["canvas_w"] == 7
        assert grown["canvas_h"] == 3
        assert comb["label"] == "comb"
        assert comb["overrides"] == [
            "body.mode=fixed",
            "body.shape=comb",
            "terrain.robot_w=7",
            "terrain.robot_h=2",
        ]
        assert comb_7x3["label"] == "comb_7x3"
        assert comb_7x3["overrides"] == [
            "body.mode=fixed",
            "body.shape=comb_7x3",
            "terrain.robot_w=7",
            "terrain.robot_h=3",
        ]

    def test_body_specs_from_cli_uses_default_matrix(self):
        specs = module._body_specs_from_cli(None)
        assert [spec["label"] for spec in specs] == ["rect_7x2"]

    def test_schedule_bundle_for_paper_run_uses_cliff_task_variants(self):
        bundle = module._schedule_bundle_for_preset(
            "paper_run",
            seed=42,
            schedules_dir="terrain_suites",
        )

        assert bundle == {
            "train": (
                "terrain_suites/"
                "seed_42_bilateral_goal_cliff_asymmetric_pairs_7x2_grid30_g150_ep6_val6_test6"
            ),
            "rerank": (
                "terrain_suites/"
                "seed_42_bilateral_goal_cliff_asymmetric_pairs_7x2_grid30_g150_ep6_val6_test6"
            ),
            "eval": (
                "terrain_suites/"
                "seed_42_bilateral_goal_cliff_asymmetric_pairs_7x2_grid30_g150_ep6_val6_test6"
            ),
        }

    def test_schedule_bundle_for_preset_accepts_body_overrides(self):
        bundle = module._schedule_bundle_for_preset(
            "paper_run",
            seed=42,
            schedules_dir="terrain_suites",
            overrides=["body.mode=grown", "body.growth.max_robot_w=7", "body.growth.max_robot_h=3"],
        )

        assert bundle["train"] == (
            "terrain_suites/"
            "seed_42_bilateral_goal_cliff_asymmetric_pairs_7x2_grid30_g150_ep6_val6_test6"
        )
        assert bundle["eval"] == (
            "terrain_suites/"
            "seed_42_bilateral_goal_cliff_asymmetric_pairs_7x2_grid30_g150_ep6_val6_test6"
        )

    def test_schedule_bundle_for_resolved_config_uses_config_variants(self, tmp_path):
        config_path = tmp_path / "resolved_config.yaml"
        OmegaConf.save(
            OmegaConf.create(
                {
                    "seed": 42,
                    "train": {"n_generations": 150, "n_episodes": 6},
                    "terrain": {
                        "variant": "curriculum_bilateral_goal_cliff_asymmetric_pairs_plateau_gap_mix123_30"
                    },
                    "eval": {
                        "rerank_episodes": 6,
                        "test_episodes": 6,
                        "rerank_variant": "bilateral_goal_cliff_asymmetric_pairs_plateau_gap_w3_30",
                        "test_variant": "bilateral_goal_cliff_asymmetric_pairs_plateau_gap_w3_30",
                    },
                }
            ),
            config_path,
        )

        bundle = module._schedule_bundle_for_resolved_config(
            config_path,
            schedules_dir="terrain_suites",
        )

        assert bundle == {
            "train": (
                "terrain_suites/"
                "seed_42_curriculum_bilateral_goal_cliff_asymmetric_pairs_plateau_gap_mix123_30_g150_ep6_val6_test6"
            ),
            "rerank": (
                "terrain_suites/"
                "seed_42_bilateral_goal_cliff_asymmetric_pairs_plateau_gap_w3_30_g150_ep6_val6_test6"
            ),
            "eval": (
                "terrain_suites/"
                "seed_42_bilateral_goal_cliff_asymmetric_pairs_plateau_gap_w3_30_g150_ep6_val6_test6"
            ),
        }

    def test_config_path_for_run_requires_resolved_config(self, tmp_path):
        run_dir = tmp_path / "paper_run_rect_7x2_seed42"
        run_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            module._config_path_for_run(run_dir)

    def test_config_path_for_run_returns_resolved_config(self, tmp_path):
        run_dir = tmp_path / "paper_run_rect_7x2_seed42"
        run_dir.mkdir()
        config_path = run_dir / "resolved_config.yaml"
        config_path.write_text("seed: 42\n")
        assert module._config_path_for_run(run_dir) == str(config_path)

    def test_paused_stage_runs_filters_branch_and_completed_runs(self, tmp_path):
        outputs_dir = tmp_path / "outputs"
        paused = outputs_dir / "paper_run_rect_7x3_seed42"
        completed = outputs_dir / "paper_run_rect_7x3_seed137"
        branch = outputs_dir / "paper_run_rect_7x3_seed7"
        for run_dir, payload in (
            (paused, {"status": "paused", "lineage": {"mode": "staged"}}),
            (completed, {"status": "completed", "lineage": {"mode": "staged"}}),
            (branch, {"status": "paused", "lineage": {"mode": "branch"}}),
        ):
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text(__import__("json").dumps(payload))

        runs = module._paused_stage_runs(outputs_dir)

        assert runs == [paused]

    def test_cmd_core_threads_stage_boundary_to_modal_train(self, monkeypatch):
        commands = []
        train_sched = Path("/tmp/train_sched")
        rerank_sched = Path("/tmp/rerank_sched")
        eval_sched = Path("/tmp/eval_sched")
        for path in (train_sched, rerank_sched, eval_sched):
            path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(module, "_body_specs_from_cli", lambda _csv: [module._parse_body_spec("rect_7x3")])
        monkeypatch.setattr(module, "_body_specs_from_publication_set", lambda _csv, _set: [module._parse_body_spec("rect_7x3")])
        monkeypatch.setattr(module, "_discover_available_schedule_seeds", lambda _dir: [42])
        monkeypatch.setattr(
            module,
            "_schedule_bundle_for_preset",
            lambda *args, **kwargs: {
                "train": str(train_sched),
                "rerank": str(rerank_sched),
                "eval": str(eval_sched),
            },
        )
        monkeypatch.setattr(module, "_run", lambda cmd, check=False: commands.append(cmd) or 0)

        module.cmd_core(
            type("Args", (), {
                "seeds": None,
                "bodies": None,
                "schedules_dir": "terrain_suites",
                "publication_set": "paper_main",
                "release_id": None,
                "require_schedules": True,
                "stage_gens": 50,
            })()
        )

        assert any("--stop-after-gens" in command for command in commands)
        assert any("--override" in command and "nca.ablate_deformation=true" in command for command in commands)

    def test_cmd_extend_resumes_only_paused_staged_runs(self, monkeypatch, tmp_path):
        outputs_dir = tmp_path / "outputs"
        paused = outputs_dir / "paper_run_rect_7x3_seed42"
        complete = outputs_dir / "paper_run_rect_7x3_seed137"
        paused.mkdir(parents=True)
        complete.mkdir(parents=True)
        (paused / "manifest.json").write_text(
            __import__("json").dumps(
                {"status": "paused", "lineage": {"mode": "staged"}, "summary": {"completed_generations": 50}}
            )
        )
        (complete / "manifest.json").write_text(
            __import__("json").dumps(
                {"status": "completed", "lineage": {"mode": "staged"}, "summary": {"completed_generations": 50}}
            )
        )
        commands = []
        monkeypatch.setattr(module, "_run", lambda cmd, check=False: commands.append(cmd) or 0)

        module.cmd_extend(
            type("Args", (), {"outputs_dir": str(outputs_dir), "to_gens": 100})()
        )

        assert len(commands) == 1
        assert commands[0][-2:] == ["--stop-after-gens", "100"]
        assert commands[0][commands[0].index("--continue-output-dir") + 1] == "/outputs/paper_run_rect_7x3_seed42"

    def test_cmd_eval_uses_latest_durable_checkpoint_and_publication_summary(self, monkeypatch, tmp_path):
        outputs_dir = tmp_path / "outputs"
        run_dir = outputs_dir / "paper_run_rect_7x3_seed42"
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        config_path = run_dir / "resolved_config.yaml"
        OmegaConf.save(
            OmegaConf.create(
                {
                    "seed": 42,
                    "preset_id": "paper_run",
                    "body": {"mode": "fixed", "shape": "rectangle", "growth": {"max_robot_w": 7, "max_robot_h": 3}},
                    "terrain": {"robot_w": 7, "robot_h": 3, "variant": "curriculum_zone10to19_extend24_gap2_30"},
                    "nca": {"ablate_deformation": False, "ctrnn_hidden": 12, "ctrnn_grid_hidden": 6},
                    "sim": {"physics_substeps": 15, "actuator_convergence": 0.003},
                    "train": {"n_generations": 200, "n_episodes": 4, "max_steps": 2000},
                    "eval": {
                        "rerank_episodes": 8,
                        "test_episodes": 16,
                        "rerank_variant": "single_gap2_zone10to24_30",
                        "test_variant": "single_gap3_zone10to24_30_ood",
                    },
                    "qd": {
                        "emitter_strategy": "sep_cmaes",
                        "n_emitters": 10,
                        "emitter_pop_size": 20,
                        "archive": {"dims": [20, 20], "ranges": [[0, 2], [0, 2]]},
                        "descriptors": {"names": ["activation_cost", "peak_progress_rate"]},
                    },
                    "gpu_profile": {"id": "l4"},
                }
            ),
            config_path,
        )
        manifest = {
            "run_id": "run-42",
            "status": "completed",
            "lineage": {"mode": "staged"},
            "config": {"logical_config_sha256": "cfg-1"},
            "summary": {"completed_generations": 200},
            "retry": {"latest_checkpoint_path": "/outputs/paper_run_rect_7x3_seed42/checkpoints/gen_0003"},
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        for generation, done in ((3, True), (4, False)):
            prefix = checkpoint_dir / f"gen_{generation:04d}"
            prefix.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "sensing_edges_checkpoint",
                        "run_id": "run-42",
                        "generation": generation,
                        "config_sha256": "cfg-1",
                    }
                )
            )
            prefix.with_suffix(".npz").write_bytes(b"npz")
            if done:
                payload_done_path(prefix).write_text("{}")

        commands = []
        monkeypatch.setattr(module, "_run", lambda cmd, check=False: commands.append(cmd) or 0)

        module.cmd_eval(
            type("Args", (), {
                "outputs_dir": str(outputs_dir),
                "schedule": None,
                "schedules_dir": "terrain_suites",
                "publication_set": "paper_main",
                "release_id": None,
                "force": False,
            })()
        )

        eval_command = next(command for command in commands if "scripts/modal_eval.py" in command)
        assert "/outputs/paper_run_rect_7x3_seed42/checkpoints/gen_0003" in eval_command
        assert "--rerank-schedule" in eval_command
        assert "--eval-schedule" in eval_command
        assert "/outputs/publication/paper_main/summaries/run-42.json" in eval_command
        release_command = next(command for command in commands if "scripts/modal_release.py" in command)
        assert release_command[-2:] == ["--release-id", "paper_main"]
