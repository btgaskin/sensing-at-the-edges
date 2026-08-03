"""Tests for Modal train CLI wiring."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf


class _FakeApp:
    def function(self, **_kwargs):
        def decorator(fn):
            return fn

        return decorator

    def local_entrypoint(self):
        def decorator(fn):
            return fn

        return decorator


def _load_modal_train(monkeypatch):
    fake_modal = types.SimpleNamespace(
        Retries=lambda **kwargs: kwargs,
        Secret=types.SimpleNamespace(from_name=lambda name: name),
    )
    fake_common = types.SimpleNamespace(
        VOLUMES={},
        app=_FakeApp(),
        commit_volumes=lambda: None,
        compose_config=lambda **kwargs: kwargs,
        parse_override_csv=lambda override: [part.strip() for part in override.split(",") if part.strip()] if override else [],
        se_image=object(),
    )

    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setitem(sys.modules, "_modal_common", fake_common)

    path = Path(__file__).parent.parent / "scripts" / "modal_train.py"
    spec = importlib.util.spec_from_file_location("sensing_edges_modal_train", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_cli_overrides_preserves_hydra_semantics(monkeypatch):
    module = _load_modal_train(monkeypatch)

    overrides = module._parse_cli_overrides(
        seed=42,
        override="train.n_generations=10,+extra.flag=true,~wandb",
    )

    assert overrides == [
        "seed=42",
        "train.n_generations=10",
        "+extra.flag=true",
        "~wandb",
    ]


def test_train_result_payload_uses_python_scalars(monkeypatch):
    module = _load_modal_train(monkeypatch)

    archive = types.SimpleNamespace(
        best_fitness=np.float32(12.5),
        coverage=np.float64(0.125),
        qd_score=np.float32(40.0),
    )

    payload = module._train_result_payload(
        archive=archive,
        output_dir=Path("/outputs/example"),
    )

    assert payload == {
        "best_fitness": 12.5,
        "coverage": 0.125,
        "qd_score": 40.0,
        "output_dir": "/outputs/example",
    }
    assert all(type(payload[key]) is float for key in ("best_fitness", "coverage", "qd_score"))
    assert type(payload["output_dir"]) is str


def test_schedule_bundle_for_cfg_uses_count_qualified_names(monkeypatch):
    module = _load_modal_train(monkeypatch)
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "train": {"n_generations": 50, "n_episodes": 5},
            "terrain": {"variant": "curriculum_bilateral_goal_cliff_probe_gap_30"},
            "eval": {
                "rerank_episodes": 12,
                "test_episodes": 12,
                "rerank_variant": "bilateral_goal_cliff_probe_gap_30",
                "test_variant": "bilateral_goal_cliff_probe_gap_30",
            },
        }
    )

    bundle = module._schedule_bundle_for_cfg(cfg)

    assert bundle == {
        "train": "terrain_suites/seed_42_curriculum_bilateral_goal_cliff_probe_gap_30_g50_ep5_val12_test12",
        "rerank": "terrain_suites/seed_42_bilateral_goal_cliff_probe_gap_30_g50_ep5_val12_test12",
        "eval": "terrain_suites/seed_42_bilateral_goal_cliff_probe_gap_30_g50_ep5_val12_test12",
    }


def test_default_output_dir_override_adds_config_hash_suffix(monkeypatch):
    module = _load_modal_train(monkeypatch)
    monkeypatch.setattr(
        module,
        "compose_config",
        lambda **_kwargs: OmegaConf.create({"preset_id": "paper_run"}),
    )
    monkeypatch.setitem(
        sys.modules,
        "sensing_edges.manifest",
        types.SimpleNamespace(config_sha256=lambda _text: "deadbeefcafebabe"),
    )
    monkeypatch.setitem(
        sys.modules,
        "sensing_edges.runtime_config",
        types.SimpleNamespace(
            logical_config_sha256=lambda _cfg: "deadbeefcafebabe",
            run_name_label=lambda _cfg: (
                "cliff_rect_7x2_sep_cmaes_g150_p200_15sub_"
                "t3000_e3000_ref003_seed137"
            ),
        ),
    )

    override = module._default_output_dir_override(
        preset="paper_run",
        resolved_overrides=["seed=137"],
    )

    assert override == (
        "train.output_dir=/outputs/"
        "paper_run_cliff_rect_7x2_sep_cmaes_g150_p200_15sub_"
        "t3000_e3000_ref003_seed137_deadbeef"
    )


def test_resolve_schedule_args_auto_resolves_paper_run(monkeypatch, tmp_path):
    module = _load_modal_train(monkeypatch)
    expected = tmp_path / "bundle"
    expected.mkdir()
    monkeypatch.setattr(
        module,
        "compose_config",
        lambda **_kwargs: OmegaConf.create(
            {
                "seed": 137,
                "terrain": {"variant": "bilateral_goal_cliff_asymmetric_pairs_7x2_grid30"},
                "train": {
                    "n_generations": 150,
                    "n_episodes": 6,
                    "require_schedule_dir": False,
                },
                "eval": {
                    "rerank_episodes": 6,
                    "test_episodes": 6,
                    "rerank_variant": "bilateral_goal_cliff_asymmetric_pairs_7x2_grid30",
                    "test_variant": "bilateral_goal_cliff_asymmetric_pairs_7x2_grid30",
                },
            }
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "sensing_edges.terrain_schedule",
        types.SimpleNamespace(resolve_schedule_bundle_path=lambda *args, **kwargs: expected),
    )

    resolved = module._resolve_schedule_args(
        preset="paper_run",
        overrides=["seed=137"],
        schedule=None,
        rerank_schedule=None,
        eval_schedule=None,
    )

    assert resolved == (str(expected), str(expected), str(expected))


def test_resolve_schedule_args_auto_resolves_iter_run_when_schedule_dir_required(monkeypatch, tmp_path):
    module = _load_modal_train(monkeypatch)
    expected = tmp_path / "bundle"
    expected.mkdir()
    monkeypatch.setattr(
        module,
        "compose_config",
        lambda **_kwargs: OmegaConf.create(
            {
                "seed": 137,
                "terrain": {"variant": "bilateral_goal_cliff_asymmetric_pairs_probe_ledge_30"},
                "train": {
                    "n_generations": 150,
                    "n_episodes": 6,
                    "require_schedule_dir": True,
                },
                "eval": {
                    "rerank_episodes": 6,
                    "test_episodes": 6,
                    "rerank_variant": "bilateral_goal_cliff_asymmetric_pairs_probe_ledge_30",
                    "test_variant": "bilateral_goal_cliff_asymmetric_pairs_probe_ledge_30",
                },
            }
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "sensing_edges.terrain_schedule",
        types.SimpleNamespace(resolve_schedule_bundle_path=lambda *args, **kwargs: expected),
    )

    resolved = module._resolve_schedule_args(
        preset="iter_run",
        overrides=["seed=137", "train.require_schedule_dir=true"],
        schedule=None,
        rerank_schedule=None,
        eval_schedule=None,
    )

    assert resolved == (str(expected), str(expected), str(expected))


def test_append_schedule_overrides_threads_train_and_eval_paths(monkeypatch):
    module = _load_modal_train(monkeypatch)

    overrides = module._append_schedule_overrides(
        ["seed=42"],
        schedule="terrain_suites/seed_42_curriculum_zone10to19_extend24_gap2_30",
        rerank_schedule="terrain_suites/seed_42_single_gap2_zone10to24_30",
        eval_schedule="terrain_suites/seed_42_single_gap3_zone10to24_30_ood",
    )

    assert overrides == [
        "seed=42",
        "train.terrain_schedule_dir=terrain_suites/seed_42_curriculum_zone10to19_extend24_gap2_30",
        "eval.rerank_schedule_dir=terrain_suites/seed_42_single_gap2_zone10to24_30",
        "eval.test_schedule_dir=terrain_suites/seed_42_single_gap3_zone10to24_30_ood",
    ]


def test_main_uses_spawn_not_detach_arg(monkeypatch):
    module = _load_modal_train(monkeypatch)
    params = inspect.signature(module.main).parameters
    assert "spawn" in params
    assert "detach" not in params


def test_append_run_surface_overrides_supports_150_population_and_eval_budget(monkeypatch):
    module = _load_modal_train(monkeypatch)

    overrides = module._append_run_surface_overrides(
        ["seed=42"],
        run="iter_run",
        pop=150,
        gens=20,
        stop_after_gens=10,
        env_steps=2500,
        eval_env_steps=3000,
        phys_substeps=15,
    )

    assert overrides == [
        "seed=42",
        "qd.n_emitters=10",
        "qd.emitter_pop_size=15",
        "batching.generation.candidate_chunk_size=150",
        "batching.generation.candidate_bucket_sizes=[150]",
        "batching.generation.exact_fit_small_run_threshold=150",
        "train.n_generations=20",
        "train.stop_after_generations=10",
        "train.max_steps=2500",
        "eval.rerank_max_steps=3000",
        "eval.test_max_steps=3000",
        "sim=half",
    ]


def test_append_run_surface_overrides_exact_fits_300_population_on_high_end_gpu(monkeypatch):
    module = _load_modal_train(monkeypatch)

    overrides = module._append_run_surface_overrides(
        ["seed=42"],
        run="iter_run",
        gpu_profile="h100",
        pop=300,
        gens=100,
        env_steps=3000,
        eval_env_steps=5000,
        phys_substeps=15,
    )

    assert overrides == [
        "seed=42",
        "qd.n_emitters=12",
        "qd.emitter_pop_size=25",
        "batching.generation.candidate_chunk_size=300",
        "batching.generation.candidate_bucket_sizes=[300]",
        "batching.generation.exact_fit_small_run_threshold=300",
        "train.n_generations=100",
        "train.max_steps=3000",
        "eval.rerank_max_steps=5000",
        "eval.test_max_steps=5000",
        "sim=half",
    ]


def test_append_run_surface_overrides_exact_fits_400_population_on_high_end_gpu(monkeypatch):
    module = _load_modal_train(monkeypatch)

    overrides = module._append_run_surface_overrides(
        ["seed=42"],
        run="iter_run",
        gpu_profile="a100_40gb",
        pop=400,
        gens=50,
        env_steps=2000,
        eval_env_steps=4000,
        phys_substeps=15,
    )

    assert overrides == [
        "seed=42",
        "qd.n_emitters=10",
        "qd.emitter_pop_size=40",
        "batching.generation.candidate_chunk_size=400",
        "batching.generation.candidate_bucket_sizes=[400]",
        "batching.generation.exact_fit_small_run_threshold=400",
        "train.n_generations=50",
        "train.max_steps=2000",
        "eval.rerank_max_steps=4000",
        "eval.test_max_steps=4000",
        "sim=half",
    ]


def test_append_run_surface_overrides_prefers_8x25_for_200_population(monkeypatch):
    module = _load_modal_train(monkeypatch)

    overrides = module._append_run_surface_overrides(
        ["seed=42"],
        run="paper_run",
        gpu_profile="h100",
        pop=200,
        env_steps=3000,
    )

    assert overrides == [
        "seed=42",
        "qd.n_emitters=8",
        "qd.emitter_pop_size=25",
        "batching.generation.candidate_chunk_size=200",
        "batching.generation.candidate_bucket_sizes=[200]",
        "batching.generation.exact_fit_small_run_threshold=200",
        "train.max_steps=3000",
        "eval.rerank_max_steps=3000",
        "eval.test_max_steps=3000",
    ]


def test_append_run_surface_overrides_supports_twenty_substeps(monkeypatch):
    module = _load_modal_train(monkeypatch)

    overrides = module._append_run_surface_overrides(
        ["seed=42"],
        run="iter_run",
        phys_substeps=20,
    )

    assert overrides == [
        "seed=42",
        "sim=twenty",
    ]


def test_append_run_surface_overrides_keeps_300_population_conservative_on_l4(monkeypatch):
    module = _load_modal_train(monkeypatch)

    overrides = module._append_run_surface_overrides(
        ["seed=42"],
        run="iter_run",
        gpu_profile="l4",
        pop=300,
    )

    assert overrides == [
        "seed=42",
        "qd.n_emitters=12",
        "qd.emitter_pop_size=25",
    ]


def test_append_run_surface_overrides_defaults_eval_budget_to_train_budget(monkeypatch):
    module = _load_modal_train(monkeypatch)

    overrides = module._append_run_surface_overrides(
        ["seed=42"],
        run="iter_run",
        env_steps=2500,
    )

    assert overrides == [
        "seed=42",
        "train.max_steps=2500",
        "eval.rerank_max_steps=2500",
        "eval.test_max_steps=2500",
    ]


def test_force_gpu_profile_replaces_conflicting_override(monkeypatch):
    module = _load_modal_train(monkeypatch)

    overrides = module._force_gpu_profile(
        ["seed=42", "gpu_profile=l4", "train.n_generations=10"],
        gpu_profile="b200",
    )

    assert overrides == [
        "seed=42",
        "train.n_generations=10",
        "gpu_profile=b200",
    ]


def test_append_body_surface_overrides_for_comb(monkeypatch):
    module = _load_modal_train(monkeypatch)

    overrides = module._append_body_surface_overrides(["seed=42"], body="comb")

    assert overrides == [
        "seed=42",
        "body.mode=fixed",
        "body.shape=comb",
        "terrain.robot_w=7",
        "terrain.robot_h=2",
    ]


def test_append_body_surface_overrides_for_generic_rectangle(monkeypatch):
    module = _load_modal_train(monkeypatch)

    overrides = module._append_body_surface_overrides(["seed=42"], body="rect_6x1")

    assert overrides == [
        "seed=42",
        "body.mode=fixed",
        "body.shape=rectangle",
        "terrain.robot_w=6",
        "terrain.robot_h=1",
    ]


def test_append_body_surface_overrides_for_comb_7x3(monkeypatch):
    module = _load_modal_train(monkeypatch)

    overrides = module._append_body_surface_overrides(["seed=42"], body="comb_7x3")

    assert overrides == [
        "seed=42",
        "body.mode=fixed",
        "body.shape=comb_7x3",
        "terrain.robot_w=7",
        "terrain.robot_h=3",
    ]


def test_append_terrain_surface_overrides_gap_curriculum_uses_matching_eval_variants(monkeypatch):
    module = _load_modal_train(monkeypatch)

    overrides = module._append_terrain_surface_overrides(["seed=42"], terrain="gap_curriculum")

    assert overrides == [
        "seed=42",
        "terrain=curriculum_bilateral_goal_cliff_asymmetric_pairs_plateau_gap_mix123_30",
        "eval.rerank_variant=bilateral_goal_cliff_asymmetric_pairs_plateau_gap_w3_30",
        "eval.test_variant=bilateral_goal_cliff_asymmetric_pairs_plateau_gap_w3_30",
    ]


def test_append_terrain_surface_overrides_selects_8x_variants_for_wide_body(monkeypatch):
    module = _load_modal_train(monkeypatch)

    overrides = module._append_terrain_surface_overrides(
        ["seed=42"],
        terrain="gap_curriculum",
        body="rect_8x3",
    )

    assert overrides == [
        "seed=42",
        "terrain=curriculum_bilateral_goal_cliff_asymmetric_pairs_plateau_gap_mix123_8x2_30",
        "eval.rerank_variant=bilateral_goal_cliff_asymmetric_pairs_plateau_gap_w3_8x2_30",
        "eval.test_variant=bilateral_goal_cliff_asymmetric_pairs_plateau_gap_w3_8x2_30",
    ]


def test_append_terrain_surface_overrides_accepts_phase_end_generations(monkeypatch):
    module = _load_modal_train(monkeypatch)

    overrides = module._append_terrain_surface_overrides(
        ["seed=42"],
        terrain="gap_curriculum",
        curriculum_phase_ends="10,20,30",
    )

    assert overrides == [
        "seed=42",
        "terrain=curriculum_bilateral_goal_cliff_asymmetric_pairs_plateau_gap_mix123_30",
        "eval.rerank_variant=bilateral_goal_cliff_asymmetric_pairs_plateau_gap_w3_30",
        "eval.test_variant=bilateral_goal_cliff_asymmetric_pairs_plateau_gap_w3_30",
        "terrain.curriculum_phase_switch_gens=[10,20]",
    ]


def test_append_terrain_surface_overrides_rejects_phase_ends_without_terrain(monkeypatch):
    module = _load_modal_train(monkeypatch)

    with pytest.raises(ValueError, match="requires --terrain"):
        module._append_terrain_surface_overrides(
            ["seed=42"],
            curriculum_phase_ends="10,20,30",
        )


def test_append_body_surface_overrides_for_grown_defaults_canvas(monkeypatch):
    module = _load_modal_train(monkeypatch)

    overrides = module._append_body_surface_overrides(["seed=42"], body="grown")

    assert overrides == [
        "seed=42",
        "body.mode=grown",
        "body.growth.max_robot_w=7",
        "body.growth.max_robot_h=3",
    ]


def test_append_body_surface_overrides_rejects_partial_grown_canvas(monkeypatch):
    module = _load_modal_train(monkeypatch)

    try:
        module._append_body_surface_overrides(["seed=42"], body="grown", canvas_w=7)
    except ValueError as exc:
        assert "requires both --canvas-w and --canvas-h" in str(exc)
    else:
        raise AssertionError("Expected ValueError for partial grown canvas")


def test_force_wandb_enabled_removes_disable_overrides(monkeypatch):
    module = _load_modal_train(monkeypatch)

    overrides = module._force_wandb_enabled(
        ["seed=42", "wandb.enabled=false", "~wandb", "train.n_generations=10"]
    )

    assert overrides == [
        "seed=42",
        "train.n_generations=10",
        "wandb.enabled=true",
    ]


def test_force_wandb_enabled_appends_true_when_unspecified(monkeypatch):
    module = _load_modal_train(monkeypatch)

    overrides = module._force_wandb_enabled(["seed=42"])

    assert overrides == [
        "seed=42",
        "wandb.enabled=true",
    ]


def test_resolve_preset_name_prefers_named_run(monkeypatch):
    module = _load_modal_train(monkeypatch)

    assert module._resolve_preset_name(preset="config", run="paper_run") == "paper_run"


def test_main_routes_h100_runs_with_matching_gpu_profile(monkeypatch):
    module = _load_modal_train(monkeypatch)
    captured = {}

    module.train_remote_h100 = types.SimpleNamespace(
        remote=lambda **kwargs: captured.update(kwargs) or {"ok": True}
    )

    module.main(
        run="iter_run",
        body="comb",
        gpu="h100",
        pop=300,
        gens=100,
        env_steps=3000,
        eval_env_steps=5000,
        phys_substeps=15,
    )

    assert captured["preset"] == "iter_run"
    assert "gpu_profile=h100" in captured["overrides"]
    assert "batching.generation.candidate_chunk_size=300" in captured["overrides"]


def test_main_routes_a100_80gb_runs_with_matching_gpu_profile(monkeypatch):
    module = _load_modal_train(monkeypatch)
    captured = {}

    module.train_remote_a100_80gb = types.SimpleNamespace(
        remote=lambda **kwargs: captured.update(kwargs) or {"ok": True}
    )

    module.main(
        run="iter_run",
        body="comb_7x3",
        gpu="a100-80gb",
        pop=300,
        gens=50,
        env_steps=3000,
        eval_env_steps=5000,
        phys_substeps=30,
    )

    assert captured["preset"] == "iter_run"
    assert "gpu_profile=a100_80gb" in captured["overrides"]
    assert "batching.generation.candidate_chunk_size=300" in captured["overrides"]
    assert "sim=full" in captured["overrides"]


def test_main_defaults_to_preset_gpu_profile_when_gpu_omitted(monkeypatch):
    module = _load_modal_train(monkeypatch)
    captured = {}

    module.compose_config = lambda **_kwargs: OmegaConf.create(
        {"gpu_profile": {"id": "a100_80gb"}}
    )
    module.train_remote_a100_80gb = types.SimpleNamespace(
        remote=lambda **kwargs: captured.update(kwargs) or {"ok": True}
    )

    module.main(
        run="paper_run",
        body="rect_6x2",
    )

    assert captured["preset"] == "paper_run"
    assert "gpu_profile=a100_80gb" in captured["overrides"]


def test_main_routes_twenty_substep_runs_with_matching_sim(monkeypatch):
    module = _load_modal_train(monkeypatch)
    captured = {}

    module.train_remote_a100_80gb = types.SimpleNamespace(
        remote=lambda **kwargs: captured.update(kwargs) or {"ok": True}
    )

    module.main(
        run="iter_run",
        body="rect_7x3",
        gpu="a100-80gb",
        pop=300,
        gens=50,
        env_steps=3000,
        eval_env_steps=5000,
        phys_substeps=20,
        override="train.n_episodes=2",
    )

    assert captured["preset"] == "iter_run"
    assert "gpu_profile=a100_80gb" in captured["overrides"]
    assert "sim=twenty" in captured["overrides"]
    assert "train.n_episodes=2" in captured["overrides"]


def test_main_applies_explicit_overrides_after_run_surface(monkeypatch):
    module = _load_modal_train(monkeypatch)
    captured = {}

    module.train_remote_a100_40gb = types.SimpleNamespace(
        remote=lambda **kwargs: captured.update(kwargs) or {"ok": True}
    )

    module.main(
        run="iter_run",
        body="rect_7x2",
        gpu="a100-40gb",
        pop=500,
        gens=50,
        env_steps=3000,
        eval_env_steps=4000,
        phys_substeps=15,
        override="eval.test_max_steps=5000,render.periodic.every_generations=5",
    )

    assert captured["preset"] == "iter_run"
    assert "train.max_steps=3000" in captured["overrides"]
    assert "eval.rerank_max_steps=4000" in captured["overrides"]
    assert "eval.test_max_steps=5000" in captured["overrides"]
    assert "render.periodic.every_generations=5" in captured["overrides"]


def test_main_supports_gap_curriculum_terrain_args(monkeypatch):
    module = _load_modal_train(monkeypatch)
    captured = {}

    module.train_remote_a100_40gb = types.SimpleNamespace(
        remote=lambda **kwargs: captured.update(kwargs) or {"ok": True}
    )

    module.main(
        run="iter_run",
        gpu="a100-40gb",
        terrain="gap_curriculum",
        curriculum_phase_ends="10,20,30",
        gens=30,
        pop=150,
        env_steps=2500,
    )

    assert captured["preset"] == "iter_run"
    assert "terrain=curriculum_bilateral_goal_cliff_asymmetric_pairs_plateau_gap_mix123_30" in captured["overrides"]
    assert "eval.rerank_variant=bilateral_goal_cliff_asymmetric_pairs_plateau_gap_w3_30" in captured["overrides"]
    assert "eval.test_variant=bilateral_goal_cliff_asymmetric_pairs_plateau_gap_w3_30" in captured["overrides"]
    assert "terrain.curriculum_phase_switch_gens=[10,20]" in captured["overrides"]
    assert "train.n_generations=30" in captured["overrides"]
    assert "train.max_steps=2500" in captured["overrides"]
    assert "gpu_profile=a100_40gb" in captured["overrides"]


def test_default_output_dir_for_cfg_uses_logical_hash(monkeypatch):
    module = _load_modal_train(monkeypatch)
    monkeypatch.setitem(
        sys.modules,
        "sensing_edges.runtime_config",
        types.SimpleNamespace(
            logical_config_sha256=lambda _cfg: "cafefeed12345678",
            run_name_label=lambda _cfg: "rect_7x3_g200_p200_12sub_t2000_e2000_ref003_seed42",
        ),
    )

    path = module._default_output_dir_for_cfg(
        OmegaConf.create({"preset_id": "paper_run"})
    )

    assert path == (
        "/outputs/"
        "paper_run_rect_7x3_g200_p200_12sub_t2000_e2000_ref003_seed42_cafefeed"
    )


def test_train_remote_impl_continues_from_existing_output_dir(monkeypatch, tmp_path):
    module = _load_modal_train(monkeypatch)
    run_dir = tmp_path / "paper_run_rect_7x3"
    run_dir.mkdir()
    OmegaConf.save(
        OmegaConf.create(
            {
                "preset_id": "paper_run",
                "train": {"output_dir": str(run_dir), "n_generations": 200},
                "wandb": {"enabled": True},
                "gpu_profile": {"id": "l4"},
            }
        ),
        run_dir / "resolved_config.yaml",
    )
    captured = {}
    monkeypatch.setattr("os.chdir", lambda _path: None)
    def _capture_overrides(cfg, overrides):
        captured["overrides"] = list(overrides)
        return cfg

    def _capture_runtime_resolution(cfg, **kwargs):
        captured["runtime_overrides"] = list(kwargs["runtime_overrides"])
        return types.SimpleNamespace()

    monkeypatch.setattr(module, "_apply_assignment_overrides", _capture_overrides)
    monkeypatch.setattr(module, "_load_resolved_cfg_from_output_dir", lambda _path: OmegaConf.load(run_dir / "resolved_config.yaml"))
    monkeypatch.setitem(
        sys.modules,
        "sensing_edges.runtime_config",
        types.SimpleNamespace(
            parse_override_keys=lambda overrides: tuple(token.split("=", 1)[0] for token in overrides),
            resolve_runtime_config=_capture_runtime_resolution,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "sensing_edges.train",
        types.SimpleNamespace(
            train=lambda cfg, **kwargs: types.SimpleNamespace(best_fitness=1.0, coverage=0.1, qd_score=2.0)
        ),
    )

    module._train_remote_impl(
        continue_output_dir=str(run_dir),
        overrides=["wandb.enabled=true"],
    )

    assert f"train.output_dir={run_dir}" in captured["overrides"]
    assert "train.resume_mode=staged" in captured["overrides"]


def test_apply_assignment_overrides_preserves_structured_group_overrides(monkeypatch):
    module = _load_modal_train(monkeypatch)

    def _compose_config(*, preset, overrides=None):
        assert preset == "iter_run"
        if overrides == ["gpu_profile=h100"]:
            return OmegaConf.create({"gpu_profile": {"id": "h100", "hbm_gib": 80}})
        if overrides == ["sim=half"]:
            return OmegaConf.create({"sim": {"physics_substeps": 15}})
        raise AssertionError(f"Unexpected overrides: {overrides!r}")

    monkeypatch.setattr(module, "compose_config", _compose_config)

    cfg = OmegaConf.create(
        {
            "preset_id": "iter_run",
            "gpu_profile": {"id": "l4", "hbm_gib": 24},
            "sim": {"physics_substeps": 30},
            "train": {"max_steps": 2500},
        }
    )

    resolved = module._apply_assignment_overrides(
        cfg,
        ["gpu_profile=h100", "sim=half", "train.max_steps=3000"],
    )

    assert resolved.gpu_profile.id == "h100"
    assert resolved.gpu_profile.hbm_gib == 80
    assert resolved.sim.physics_substeps == 15
    assert resolved.train.max_steps == 3000


def test_load_resolved_cfg_from_output_dir_normalizes_legacy_scalar_gpu_profile(monkeypatch, tmp_path):
    module = _load_modal_train(monkeypatch)
    run_dir = tmp_path / "legacy_run"
    run_dir.mkdir()
    OmegaConf.save(
        OmegaConf.create(
            {
                "preset_id": "iter_run",
                "gpu_profile": "a100_40gb",
                "train": {"output_dir": str(run_dir)},
            }
        ),
        run_dir / "resolved_config.yaml",
    )

    def _compose_config(*, preset, overrides=None):
        assert preset == "iter_run"
        if overrides == ["gpu_profile=a100_40gb"]:
            return OmegaConf.create(
                {
                    "gpu_profile": {
                        "id": "a100_40gb",
                        "hbm_gib": 40,
                        "usable_fraction": 0.75,
                    }
                }
            )
        return OmegaConf.create(
            {
                "preset_id": "iter_run",
                "gpu_profile": {
                    "id": "l4",
                    "hbm_gib": 24,
                    "usable_fraction": 0.7,
                },
                "eval": {"post_run": {"enabled": False}},
            }
        )

    monkeypatch.setattr(module, "compose_config", _compose_config)

    resolved = module._load_resolved_cfg_from_output_dir(run_dir)

    assert resolved.gpu_profile.id == "a100_40gb"
    assert resolved.gpu_profile.hbm_gib == 40
    assert resolved.eval.post_run.enabled is False


def test_load_resolved_cfg_from_output_dir_prunes_removed_legacy_keys(monkeypatch, tmp_path):
    module = _load_modal_train(monkeypatch)
    run_dir = tmp_path / "legacy_run"
    run_dir.mkdir()
    OmegaConf.save(
        OmegaConf.create(
            {
                "preset_id": "paper_run",
                "eval": {
                    "post_run": {
                        "enabled": True,
                        "trace": {
                            "enabled": True,
                            "scope": "native_intact_all_trials",
                            "sample_resolution": "controller_step",
                        },
                    }
                },
                "train": {"output_dir": str(run_dir)},
            }
        ),
        run_dir / "resolved_config.yaml",
    )

    monkeypatch.setattr(
        module,
        "compose_config",
        lambda **_kwargs: OmegaConf.create(
            {
                "preset_id": "paper_run",
                "eval": {
                    "post_run": {
                        "enabled": False,
                        "trace": {
                            "enabled": True,
                            "sample_resolution": "controller_step",
                        },
                    }
                },
                "train": {"output_dir": "/outputs/default"},
            }
        ),
    )

    resolved = module._load_resolved_cfg_from_output_dir(run_dir)

    assert resolved.eval.post_run.enabled is True
    assert resolved.eval.post_run.trace.enabled is True
    assert resolved.eval.post_run.trace.sample_resolution == "controller_step"
    assert "scope" not in resolved.eval.post_run.trace


def test_main_rejects_gens_with_continue_output_dir(monkeypatch):
    module = _load_modal_train(monkeypatch)

    with pytest.raises(SystemExit, match="stop-after-gens"):
        module.main(
            continue_output_dir="/outputs/example",
            gens=100,
        )


def test_main_defaults_new_launches_to_fresh_resume_mode(monkeypatch):
    module = _load_modal_train(monkeypatch)
    captured = {}

    class _Remote:
        def remote(self, **kwargs):
            captured.update(kwargs)
            return None

    monkeypatch.setattr(module, "train_remote_a100_40gb", _Remote())
    monkeypatch.setattr(module, "_resolve_gpu_target", lambda **_kwargs: ("a100-40gb", {"profile_id": "a100_40gb"}))

    module.main(run="iter_run")

    assert "train.resume_mode=fresh" in captured["overrides"]
