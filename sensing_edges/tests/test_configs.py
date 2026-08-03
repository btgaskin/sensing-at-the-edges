"""Tests for Hydra config resolution and body-variant runtime wiring."""

from __future__ import annotations

import pytest

from sensing_edges.qd_descriptors import resolve_qd_archive
from sensing_edges.runtime_config import (
    body_variant_label,
    compatible_post_run_signaling_conditions,
    compose_experiment_config,
    genotype_dim_from_cfg,
    resolved_noise_settings,
    resolved_spawn_x,
    terrain_family_label,
    run_group_label,
    run_metadata,
    run_name_label,
    resolve_runtime_config,
)

PAPER_CORE_ID = "sensing_edges_sbt_v1"
RUN_PRESETS = ["iter_run", "paper_run"]

RUN_PRESET_NCA_SHAPES = {
    "iter_run": (12, 6),
    "paper_run": (12, 6),
}

RUN_PRESET_ACTIVE_COMM_FILTERS = {
    "iter_run": ("sobel_x", "sobel_y", "laplacian", "identity"),
    "paper_run": ("sobel_x", "sobel_y", "laplacian", "identity"),
}


def _compose(preset: str | None = None, overrides: list[str] | None = None):
    return compose_experiment_config(preset=preset, overrides=overrides)


class TestConfigResolution:
    def test_default_resolves(self):
        cfg = _compose()
        resolution = resolve_runtime_config(cfg)
        qd_archive = resolve_qd_archive(cfg)

        assert cfg.seed == 42
        assert cfg.paper_core_id == PAPER_CORE_ID
        assert cfg.body.mode == "fixed"
        assert cfg.body.shape == "rectangle"
        assert cfg.body.fixed.hidden_init_scale == pytest.approx(0.0)
        assert cfg.nca.ctrnn_hidden == 8
        assert cfg.nca.ctrnn_grid_hidden == 12
        assert cfg.nca.ctrnn_tau_min == pytest.approx(0.1)
        assert cfg.nca.gasnet_mode == "off"
        assert cfg.nca.deformation_gain_absolute == pytest.approx(10.0)
        assert cfg.nca.deformation_gain_compliance == pytest.approx(85.0)
        assert cfg.nca.deformation_gain_auto_scale is True
        assert cfg.eval.robustness_probe.enabled is False
        assert cfg.eval.gap_generalization_suite.enabled is False
        assert cfg.train.allow_legacy_schedule_hashes is False
        assert cfg.eval.allow_legacy_schedule_hashes is False
        assert cfg.eval.allow_checkpoint_config_mismatch is False
        assert cfg.eval.robustness_probe.mode == "symmetric_double_cliff"
        assert cfg.eval.robustness_probe.plateau_width == 10
        assert cfg.render.post_train.upload_to_wandb is False
        assert cfg.render.post_train.include_archive_visuals is True
        assert cfg.sim.physics_substeps == 30
        assert cfg.sim.actuator_convergence == pytest.approx(0.003)
        assert cfg.sim.spring_stiffness_scale == pytest.approx(0.8)
        assert cfg.sim.floor_static_friction_const == pytest.approx(0.4)
        assert cfg.sim.floor_stiction_stiffness == pytest.approx(20_000.0)
        assert cfg.sim.floor_stiction_switch_speed == pytest.approx(0.05)
        assert cfg.sim.airborne_kill_height_margin_cells == pytest.approx(2.0)
        assert resolution.paper_core_id == PAPER_CORE_ID
        assert qd_archive.descriptor_names == ("activation_cost", "peak_progress_rate")
        assert qd_archive.ranges == ((0.0, 2.0), (0.0, 2.0))

    @pytest.mark.parametrize("preset", RUN_PRESETS)
    def test_run_preset_resolves(self, preset):
        cfg = _compose(preset)
        resolution = resolve_runtime_config(cfg, preset=preset)
        expected_ctrnn_hidden, expected_ctrnn_grid_hidden = RUN_PRESET_NCA_SHAPES[preset]

        assert cfg.preset_id == preset
        assert cfg.nca.ctrnn_hidden == expected_ctrnn_hidden
        assert cfg.nca.ctrnn_grid_hidden == expected_ctrnn_grid_hidden
        assert cfg.nca.ctrnn_tau_min == pytest.approx(1.0)
        assert cfg.nca.gasnet_mode == "off"
        assert cfg.nca.gasnet_type == "signal"
        assert cfg.nca.ablate_deformation is False
        assert cfg.nca.deformation_gain_absolute == pytest.approx(10.0)
        assert cfg.nca.deformation_gain_compliance == pytest.approx(85.0)
        assert cfg.nca.deformation_gain_auto_scale is True
        assert tuple(cfg.nca.active_comm_filters) == RUN_PRESET_ACTIVE_COMM_FILTERS[preset]
        assert resolution.preset_name == preset
        assert resolution.paper_core_id == PAPER_CORE_ID

    def test_paper_run_sets_fixed_hidden_init_scale(self):
        cfg = _compose("paper_run")

        assert cfg.body.fixed.hidden_init_scale == pytest.approx(0.0)

    def test_negative_fixed_hidden_init_scale_fails_fast(self):
        cfg = _compose(
            "paper_run",
            overrides=["body.fixed.hidden_init_scale=-0.01"],
        )

        with pytest.raises(ValueError, match="body.fixed.hidden_init_scale must be non-negative"):
            resolve_runtime_config(cfg, preset="paper_run")

    def test_active_comm_filters_override_changes_genotype_dim(self):
        cfg = _compose("iter_run")
        reduced_cfg = _compose(
            "iter_run",
            overrides=["nca.active_comm_filters=[laplacian,identity]"],
        )

        assert genotype_dim_from_cfg(cfg) == 624
        assert genotype_dim_from_cfg(reduced_cfg) == 480

    def test_incompatible_active_signaling_override_fails_fast(self):
        cfg = _compose(
            "paper_run",
            overrides=[
                "nca.active_comm_filters=[laplacian,identity]",
                "eval.post_run.active_signaling_condition=sobel_x_off",
            ],
        )

        with pytest.raises(ValueError, match="requires active communication filter"):
            resolve_runtime_config(cfg, preset="paper_run")

    def test_post_run_signaling_conditions_auto_prune_for_reduced_filter_bank(self):
        cfg = _compose(
            "paper_run",
            overrides=["nca.active_comm_filters=[laplacian,identity]"],
        )
        resolution = resolve_runtime_config(cfg, preset="paper_run")

        assert resolution.paper_core_id == PAPER_CORE_ID
        assert compatible_post_run_signaling_conditions(cfg) == (
            "intact",
            "laplacian_off",
            "identity_comm_off",
            "all_comm_off",
        )

    def test_post_run_fast_ablation_defaults_and_render_subsets(self):
        cfg = _compose("paper_run")

        assert cfg.eval.post_run.fast_ablation_eval is True
        assert list(cfg.eval.post_run.signaling_render_conditions) == ["intact"]
        assert list(cfg.eval.post_run.deformation_render_conditions) == [
            "intact",
            "deformation_off",
            "noise_only",
            "shuffled",
        ]
        assert list(cfg.eval.post_run.body_transfer_render_bodies) == ["reference"]
        assert [item["test_id"] for item in cfg.eval.post_run.combined_ablation_quick_tests] == [
            "deformation_off__sobel_x_off",
            "deformation_off__sobel_y_off",
            "deformation_off__laplacian_off",
            "deformation_off__identity_comm_off",
            "deformation_off__all_comm_off",
        ]

    def test_iter_run_defaults_to_fixed_rect_7x2_cliff_iteration_profile(self):
        cfg = _compose("iter_run")
        qd_archive = resolve_qd_archive(cfg)

        assert cfg.body.mode == "fixed"
        assert cfg.body.shape == "rectangle"
        assert cfg.terrain.robot_w == 7
        assert cfg.terrain.robot_h == 2
        assert cfg.terrain.variant == "bilateral_goal_cliff_asymmetric_pairs_7x2_grid30"
        assert cfg.terrain.bilateral_spawn_x_offset == 1
        assert "curriculum" not in cfg.terrain
        assert cfg.terrain.grid_w == 30
        assert cfg.terrain.base_width == 12
        assert cfg.terrain.center_plateau_width == 12
        assert cfg.terrain.task_mode == "bilateral_goal_cliff"
        assert cfg.terrain.cliff_x_min == 20
        assert cfg.terrain.cliff_x_max == 25
        assert cfg.terrain.cliff_probe_gap_width == 0
        assert cfg.terrain.cliff_probe_ledge_width == 0
        assert list(cfg.terrain.distance_pairs) == [
            {"goal_distance": 2, "cliff_distance": 1},
            {"goal_distance": 4, "cliff_distance": 2},
            {"goal_distance": 6, "cliff_distance": 3},
        ]
        assert cfg.sim.physics_substeps == 15
        assert cfg.sim.spring_stiffness_scale == pytest.approx(0.8)
        assert cfg.sim.airborne_kill_height_margin_cells == pytest.approx(1.0)
        assert cfg.nca.mace_mode == "off"
        assert cfg.train.n_generations == 50
        assert cfg.train.n_episodes == 6
        assert cfg.train.max_steps == 2500
        assert cfg.train.checkpoint_interval == 10
        assert cfg.qd.n_emitters == 10
        assert cfg.qd.emitter_pop_size == 15
        assert cfg.qd.emitter_strategy == "sep_cmaes"
        assert cfg.qd.initial_sigma == pytest.approx(1.0)
        assert cfg.qd.archive.learning_rate == pytest.approx(0.1)
        assert cfg.nca.noise.noise_type == "off"
        assert cfg.specimens.enabled is True
        assert cfg.eval.rerank_variant == "bilateral_goal_cliff_asymmetric_pairs_7x2_grid30"
        assert cfg.eval.test_variant == "bilateral_goal_cliff_asymmetric_pairs_7x2_grid30"
        assert cfg.eval.rerank_episodes == 6
        assert cfg.eval.test_episodes == 6
        assert cfg.eval.rerank_max_steps == 2500
        assert cfg.eval.test_max_steps == 2500
        assert cfg.gpu_profile.id == "a100_40gb"
        assert body_variant_label(cfg) == "rect_7x2"
        assert resolved_spawn_x(cfg) == 12
        assert cfg.render.periodic.every_generations == 10
        assert cfg.render.periodic.select_cells == 2
        assert cfg.render.periodic.max_individuals == 3
        assert cfg.render.periodic.episodes_per_individual == 6
        assert cfg.render.periodic.width == 360
        assert cfg.render.periodic.height == 180
        assert cfg.render.periodic.fps == 16
        assert cfg.render.periodic.target_frames == 160
        assert cfg.render.periodic.terrain_selection == "all"
        assert cfg.render.periodic.include_archive_visuals is True
        assert cfg.render.post_train.upload_to_wandb is False
        assert cfg.render.post_train.include_archive_visuals is True
        assert qd_archive.descriptor_names == (
            "goal_hit_count",
            "fall_count",
            "completion_time",
        )
        assert qd_archive.dims == (7, 7, 10)
        assert qd_archive.ranges == ((0.0, 6.0), (0.0, 6.0), (0.0, 1.0))

    def test_iter_run_accepts_twenty_substep_override(self):
        cfg = _compose("iter_run", overrides=["sim=twenty"])
        resolution = resolve_runtime_config(
            cfg,
            preset="iter_run",
            runtime_overrides=["sim=twenty"],
        )

        assert cfg.sim.physics_substeps == 20
        assert cfg.sim.sense_act_phases_per_env_step == 4

    def test_iter_run_accepts_symmetry_and_speed_qd_override(self):
        cfg = _compose(
            "iter_run",
            overrides=["qd.descriptors.names=[mirrored_roundtrip_top2_mean,movement_speed]"],
        )
        qd_archive = resolve_qd_archive(cfg)

        assert qd_archive.descriptor_names == (
            "mirrored_roundtrip_top2_mean",
            "movement_speed",
        )
        assert qd_archive.ranges == ((0.0, 1.0), (0.0, 1.0))

    def test_iter_run_accepts_symmetry_and_completion_time_qd_override(self):
        cfg = _compose(
            "iter_run",
            overrides=["qd.descriptors.names=[mirrored_roundtrip_top2_mean,completion_time]"],
        )
        qd_archive = resolve_qd_archive(cfg)

        assert qd_archive.descriptor_names == (
            "mirrored_roundtrip_top2_mean",
            "completion_time",
        )
        assert qd_archive.ranges == ((0.0, 1.0), (0.0, 1.0))

    def test_noise_resolution_prefers_nested_config_and_normalizes_legacy_fields(self):
        nested_cfg = _compose(
            overrides=[
                "nca.noise.noise_type=ou_fixed",
                "nca.noise.ou_theta=0.25",
                "nca.noise.ou_sigma=0.5",
                "+nca.ou_enabled=false",
                "+nca.ou_theta=0.9",
                "+nca.ou_sigma=0.9",
            ],
        )
        nested_noise = resolved_noise_settings(nested_cfg)

        assert nested_noise["noise_type"] == "ou_fixed"
        assert nested_noise["ou_theta"] == pytest.approx(0.25)
        assert nested_noise["ou_sigma"] == pytest.approx(0.5)
        assert nested_noise["ou_enabled"] is True

        legacy_cfg = _compose(
            overrides=[
                "nca.noise=null",
                "+nca.ou_enabled=false",
                "+nca.ou_theta=0.2",
                "+nca.ou_sigma=0.4",
            ],
        )
        legacy_noise = resolved_noise_settings(legacy_cfg)

        assert legacy_noise["noise_type"] == "off"
        assert legacy_noise["ou_theta"] == pytest.approx(0.2)
        assert legacy_noise["ou_sigma"] == pytest.approx(0.4)
        assert legacy_noise["ou_enabled"] is False

    def test_ou_evolved_noise_type_is_rejected_until_implemented(self):
        cfg = _compose(overrides=["nca.noise.noise_type=ou_evolved"])

        with pytest.raises(ValueError, match="reserved but not implemented yet"):
            resolve_runtime_config(cfg)

    @pytest.mark.parametrize(
        ("terrain_variant", "robot_w", "robot_h", "expected_body_label", "expected_spawn_x"),
        [
            (
                "bilateral_goal_cliff_asymmetric_pairs_probe_ledge_30",
                6,
                2,
                "rect_6x2",
                13,
            ),
            (
                "bilateral_goal_cliff_asymmetric_pairs_probe_ledge_8x2_30",
                8,
                2,
                "rect_8x2",
                12,
            ),
        ],
    )
    def test_iter_run_accepts_ledge_only_fixed_pair_overrides(
        self,
        terrain_variant,
        robot_w,
        robot_h,
        expected_body_label,
        expected_spawn_x,
    ):
        overrides = [
            "nca=sbt_32h_8grid",
            "sim=half",
            "gpu_profile=a100_80gb",
            f"terrain={terrain_variant}",
            "body.mode=fixed",
            "body.shape=rectangle",
            f"terrain.robot_w={robot_w}",
            f"terrain.robot_h={robot_h}",
            "train.n_generations=150",
            "train.stop_after_generations=100",
            "train.n_episodes=6",
            "train.max_steps=2000",
            "train.checkpoint_interval=50",
            "train.require_schedule_dir=true",
            f"eval.rerank_variant={terrain_variant}",
            f"eval.test_variant={terrain_variant}",
            "eval.rerank_episodes=6",
            "eval.test_episodes=6",
            "eval.rerank_max_steps=3000",
            "eval.test_max_steps=3000",
            "qd.n_emitters=10",
            "qd.emitter_pop_size=30",
            "qd.descriptors.names=[correct_progress,incorrect_progress]",
            "qd.archive.dims=[10,10]",
        ]
        cfg = _compose("iter_run", overrides=overrides)
        resolution = resolve_runtime_config(
            cfg,
            preset="iter_run",
            runtime_overrides=overrides,
        )
        qd_archive = resolve_qd_archive(cfg)

        assert cfg.nca.ctrnn_hidden == 32
        assert cfg.nca.ctrnn_grid_hidden == 8
        assert cfg.terrain.variant == terrain_variant
        assert cfg.terrain.cliff_probe_gap_width == 0
        assert cfg.terrain.cliff_probe_ledge_width == 3
        assert cfg.sim.physics_substeps == 15
        assert cfg.sim.spring_stiffness_scale == pytest.approx(0.8)
        assert cfg.train.n_generations == 150
        assert cfg.train.stop_after_generations == 100
        assert cfg.train.n_episodes == 6
        assert cfg.train.max_steps == 2000
        assert cfg.train.checkpoint_interval == 50
        assert cfg.train.require_schedule_dir is True
        assert cfg.eval.rerank_variant == terrain_variant
        assert cfg.eval.test_variant == terrain_variant
        assert cfg.eval.rerank_episodes == 6
        assert cfg.eval.test_episodes == 6
        assert cfg.eval.rerank_max_steps == 3000
        assert cfg.eval.test_max_steps == 3000
        assert cfg.qd.n_emitters == 10
        assert cfg.qd.emitter_pop_size == 30
        assert cfg.gpu_profile.id == "a100_80gb"
        assert body_variant_label(cfg) == expected_body_label
        assert resolved_spawn_x(cfg) == expected_spawn_x
        assert resolution.gpu_profile == "a100_80gb"
        assert qd_archive.descriptor_names == ("correct_progress", "incorrect_progress")
        assert qd_archive.dims == (10, 10)
        assert qd_archive.ranges == ((0.0, 1.0), (0.0, 1.0))

    def test_iter_run_accepts_ledge4_signal_4000_step_surface(self):
        overrides = [
            "nca=sbt_32h_8grid",
            "gasnet=signal",
            "sim=half",
            "gpu_profile=a100_80gb",
            "terrain=bilateral_goal_cliff_asymmetric_pairs_probe_ledge_4_30",
            "body.mode=fixed",
            "body.shape=rectangle",
            "terrain.robot_w=7",
            "terrain.robot_h=2",
            "train.n_generations=150",
            "train.stop_after_generations=100",
            "train.n_episodes=6",
            "train.max_steps=4000",
            "train.checkpoint_interval=50",
            "train.require_schedule_dir=true",
            "eval.rerank_variant=bilateral_goal_cliff_asymmetric_pairs_probe_ledge_4_30",
            "eval.test_variant=bilateral_goal_cliff_asymmetric_pairs_probe_ledge_4_30",
            "eval.rerank_episodes=6",
            "eval.test_episodes=6",
            "eval.rerank_max_steps=4000",
            "eval.test_max_steps=4000",
            "qd.n_emitters=10",
            "qd.emitter_pop_size=30",
            "qd.descriptors.names=[correct_progress,incorrect_progress]",
            "qd.archive.dims=[10,10]",
        ]
        cfg = _compose("iter_run", overrides=overrides)
        resolution = resolve_runtime_config(
            cfg,
            preset="iter_run",
            runtime_overrides=overrides,
        )
        qd_archive = resolve_qd_archive(cfg)

        assert cfg.nca.ctrnn_hidden == 32
        assert cfg.nca.ctrnn_grid_hidden == 8
        assert cfg.nca.gasnet_mode == "single"
        assert cfg.nca.gasnet_type == "signal"
        assert cfg.terrain.variant == "bilateral_goal_cliff_asymmetric_pairs_probe_ledge_4_30"
        assert cfg.terrain.cliff_probe_gap_width == 0
        assert cfg.terrain.cliff_probe_ledge_width == 4
        assert cfg.sim.physics_substeps == 15
        assert cfg.train.n_generations == 150
        assert cfg.train.stop_after_generations == 100
        assert cfg.train.n_episodes == 6
        assert cfg.train.max_steps == 4000
        assert cfg.train.checkpoint_interval == 50
        assert cfg.eval.rerank_variant == "bilateral_goal_cliff_asymmetric_pairs_probe_ledge_4_30"
        assert cfg.eval.test_variant == "bilateral_goal_cliff_asymmetric_pairs_probe_ledge_4_30"
        assert cfg.eval.rerank_episodes == 6
        assert cfg.eval.test_episodes == 6
        assert cfg.eval.rerank_max_steps == 4000
        assert cfg.eval.test_max_steps == 4000
        assert cfg.gpu_profile.id == "a100_80gb"
        assert body_variant_label(cfg) == "rect_7x2"
        assert resolved_spawn_x(cfg) == 12
        assert resolution.gpu_profile == "a100_80gb"
        assert qd_archive.descriptor_names == ("correct_progress", "incorrect_progress")
        assert qd_archive.dims == (10, 10)

    def test_paper_run_defaults_to_fixed_rect_7x2_cliff_schedule_profile(self):
        cfg = _compose("paper_run")
        qd_archive = resolve_qd_archive(cfg)

        assert cfg.body.mode == "fixed"
        assert cfg.body.shape == "rectangle"
        assert cfg.terrain.robot_w == 7
        assert cfg.terrain.robot_h == 2
        assert cfg.terrain.variant == "bilateral_goal_cliff_asymmetric_pairs_7x2_grid30"
        assert "curriculum" not in cfg.terrain
        assert cfg.terrain.grid_w == 30
        assert cfg.terrain.base_width == 12
        assert cfg.terrain.center_plateau_width == 12
        assert cfg.terrain.task_mode == "bilateral_goal_cliff"
        assert cfg.terrain.cliff_x_min == 20
        assert cfg.terrain.cliff_x_max == 25
        assert cfg.terrain.cliff_probe_gap_width == 0
        assert cfg.terrain.cliff_probe_ledge_width == 0
        assert list(cfg.terrain.distance_pairs) == [
            {"goal_distance": 2, "cliff_distance": 1},
            {"goal_distance": 4, "cliff_distance": 2},
            {"goal_distance": 6, "cliff_distance": 3},
        ]
        assert cfg.sim.physics_substeps == 15
        assert cfg.sim.spring_stiffness_scale == pytest.approx(0.8)
        assert cfg.sim.airborne_kill_height_margin_cells == pytest.approx(1.0)
        assert cfg.nca.mace_mode == "off"
        assert cfg.train.n_generations == 150
        assert cfg.train.stop_after_generations == 100
        assert cfg.train.n_episodes == 6
        assert cfg.train.max_steps == 3000
        assert cfg.train.checkpoint_interval == 25
        assert cfg.train.get("activation_cost_penalty_scale") is None
        assert cfg.train.soft_loft_penalty_scale == pytest.approx(8.0)
        assert cfg.train.gait_spectral_purity_bonus_scale == pytest.approx(0.0)
        assert cfg.eval.rerank_variant == "bilateral_goal_cliff_asymmetric_pairs_7x2_grid30"
        assert cfg.eval.test_variant == "bilateral_goal_cliff_asymmetric_pairs_7x2_grid30"
        assert cfg.eval.rerank_episodes == 6
        assert cfg.eval.test_episodes == 6
        assert cfg.eval.rerank_max_steps == 3000
        assert cfg.eval.test_max_steps == 3000
        assert cfg.eval.deformation_suite.enabled is True
        assert cfg.eval.deformation_suite.render is True
        assert cfg.eval.deformation_suite.render_top_n == 3
        assert cfg.eval.gap_generalization_suite.enabled is True
        assert cfg.eval.robustness_probe.enabled is True
        assert cfg.eval.robustness_probe.elite_count == 3
        assert cfg.eval.robustness_probe.render is True
        assert cfg.eval.post_run.enabled is True
        assert cfg.eval.post_run.auto_mode == "off"
        assert cfg.eval.post_run.run_on_stage_pause is True
        assert cfg.eval.post_run.selected_rank == 0
        assert cfg.eval.post_run.max_steps == 6000
        assert cfg.eval.post_run.native_max_steps == 4000
        assert cfg.eval.post_run.double_cliff_max_steps == 6000
        assert cfg.eval.post_run.analysis_root == "analyses"
        assert list(cfg.eval.post_run.native_pairs) == [
            {"goal_distance": 2, "cliff_distance": 1},
            {"goal_distance": 4, "cliff_distance": 2},
            {"goal_distance": 6, "cliff_distance": 3},
        ]
        assert list(cfg.eval.post_run.long_distance_pairs) == [
            {"goal_distance": 8, "cliff_distance": 4},
            {"goal_distance": 10, "cliff_distance": 5},
        ]
        assert list(cfg.eval.post_run.long_distance_pair_max_steps) == [
            {"goal_distance": 8, "cliff_distance": 4, "max_steps": 5000},
            {"goal_distance": 10, "cliff_distance": 5, "max_steps": 6000},
        ]
        assert cfg.eval.post_run.long_distance_terrain.grid_w == 32
        assert cfg.eval.post_run.long_distance_terrain.plateau_width == 12
        assert cfg.eval.post_run.trials_per_side == 20
        assert cfg.eval.post_run.stochastic_trials_per_side == 5
        assert cfg.specimens.run_on_stage_pause is True
        assert cfg.eval.post_run.double_cliff.plateau_width == 9
        assert list(cfg.eval.post_run.double_cliff.widths) == [2, 4, 6]
        assert cfg.eval.post_run.double_cliff.stochastic_trials_per_width == 5
        assert cfg.qd.archive.learning_rate == pytest.approx(0.1)
        assert cfg.qd.initial_sigma == pytest.approx(1.0)
        assert cfg.qd.n_emitters == 8
        assert cfg.qd.emitter_pop_size == 25
        assert cfg.qd.emitter_strategy == "sep_cmaes"
        assert cfg.qd.emitter_flat_generations == 3
        assert cfg.qd.emitter_flat_tolerance == pytest.approx(1.0e-12)
        assert cfg.qd.emitter_restart_stale_generations == 5
        assert cfg.qd.emitter_restart_max_age == 50
        assert cfg.qd.emitter_restart_improvement_eps == pytest.approx(0.5)
        assert cfg.qd.emitter_std_epsilon == pytest.approx(1.0e-12)
        assert cfg.qd.emitter_condition_cov_limit == pytest.approx(1.0e14)
        assert cfg.qd.focus.enabled is True
        assert cfg.qd.focus.tighten_every_n_gens == 20
        assert cfg.qd.focus.min_occupied_to_tighten == 2
        assert cfg.qd.focus.min_window_size == 3
        assert list(cfg.qd.focus.targets) == [
            {"name": "goal_hit_count", "direction": "max"},
            {"name": "fall_count", "direction": "min"},
        ]
        assert tuple(cfg.nca.active_comm_filters) == (
            "sobel_x",
            "sobel_y",
            "laplacian",
            "identity",
        )
        assert compatible_post_run_signaling_conditions(cfg) == (
            "intact",
            "sobel_x_off",
            "sobel_y_off",
            "laplacian_off",
            "identity_comm_off",
            "all_comm_off",
        )
        assert list(cfg.eval.post_run.signaling_surfaces) == ["native_core"]
        assert list(cfg.eval.post_run.deformation_surfaces) == ["native_core"]
        assert cfg.nca.noise.noise_type == "off"
        assert cfg.specimens.enabled is True
        assert cfg.batching.generation.candidate_chunk_size == 200
        assert cfg.gpu_profile.id == "h100"
        assert cfg.render.periodic.every_generations == 10
        assert cfg.render.periodic.select_cells == 2
        assert cfg.render.periodic.max_individuals == 3
        assert cfg.render.periodic.episodes_per_individual == 6
        assert cfg.render.periodic.width == 360
        assert cfg.render.periodic.height == 180
        assert cfg.render.periodic.fps == 16
        assert cfg.render.periodic.target_frames == 160
        assert cfg.render.periodic.terrain_selection == "all"
        assert cfg.render.periodic.include_archive_visuals is True
        assert list(cfg.render.periodic.archive_heatmaps) == [
            "hits_falls",
            "time_hits",
            "time_falls",
        ]
        assert qd_archive.descriptor_names == (
            "goal_hit_count",
            "fall_count",
            "completion_time",
        )
        assert qd_archive.dims == (7, 7, 20)
        assert qd_archive.ranges == ((0.0, 6.0), (0.0, 6.0), (0.2, 0.8))

    def test_default_qd_focus_is_disabled(self):
        cfg = _compose()

        assert cfg.qd.focus.enabled is False
        assert cfg.qd.focus.tighten_every_n_gens == 10
        assert cfg.qd.focus.min_window_size == 3
        assert list(cfg.qd.focus.targets) == []

    def test_focus_validation_fails_when_target_descriptor_is_missing(self):
        cfg = _compose(
            "paper_run",
            overrides=[
                "qd.descriptors.names=[goal_hit_count,completion_time]",
                "qd.focus.enabled=true",
            ],
        )

        with pytest.raises(ValueError, match="fall_count"):
            resolve_runtime_config(cfg, preset="paper_run")

    def test_resolved_spawn_x_recenters_bilateral_rectangles_from_geometry(self):
        cfg = _compose(
            "paper_run",
            overrides=["terrain.robot_w=4"],
        )

        assert int(cfg.terrain.spawn_x) == 11
        assert resolved_spawn_x(cfg) == 14

    def test_comb_body_override_reduces_alive_cell_range(self):
        overrides = [
            "body.mode=fixed",
            "body.shape=comb",
            "terrain.robot_w=7",
            "terrain.robot_h=2",
        ]
        cfg = _compose("paper_run", overrides=overrides)
        qd_archive = resolve_qd_archive(cfg)

        assert body_variant_label(cfg) == "comb"
        assert qd_archive.descriptor_names == (
            "goal_hit_count",
            "fall_count",
            "completion_time",
        )
        assert qd_archive.ranges == ((0.0, 6.0), (0.0, 6.0), (0.2, 0.8))

    def test_rect_7x3_body_override_uses_expected_variant_label(self):
        overrides = [
            "body.mode=fixed",
            "body.shape=rectangle",
            "terrain.robot_w=7",
            "terrain.robot_h=3",
        ]
        cfg = _compose("paper_run", overrides=overrides)
        qd_archive = resolve_qd_archive(cfg)

        assert body_variant_label(cfg) == "rect_7x3"
        assert qd_archive.descriptor_names == (
            "goal_hit_count",
            "fall_count",
            "completion_time",
        )
        assert qd_archive.ranges == ((0.0, 6.0), (0.0, 6.0), (0.2, 0.8))

    def test_comb_7x3_body_override_uses_expected_variant_label(self):
        overrides = [
            "body.mode=fixed",
            "body.shape=comb_7x3",
            "terrain.robot_w=7",
            "terrain.robot_h=3",
        ]
        cfg = _compose("paper_run", overrides=overrides)
        qd_archive = resolve_qd_archive(cfg)

        assert body_variant_label(cfg) == "comb_7x3"
        assert qd_archive.descriptor_names == (
            "goal_hit_count",
            "fall_count",
            "completion_time",
        )
        assert qd_archive.ranges == ((0.0, 6.0), (0.0, 6.0), (0.2, 0.8))

    def test_grown_body_override_uses_canvas_range(self):
        overrides = [
            "body.mode=grown",
            "body.growth.max_robot_w=7",
            "body.growth.max_robot_h=3",
        ]
        cfg = _compose("paper_run", overrides=overrides)
        qd_archive = resolve_qd_archive(cfg)

        assert body_variant_label(cfg) == "grown_7x3"
        assert qd_archive.descriptor_names == (
            "goal_hit_count",
            "fall_count",
            "completion_time",
        )
        assert qd_archive.ranges == ((0.0, 6.0), (0.0, 6.0), (0.2, 0.8))

    @pytest.mark.parametrize(
        ("terrain_variant", "expected_family"),
        [
            ("bilateral_goal_cliff_asymmetric_pairs_30", "cliff"),
            ("curriculum_bilateral_goal_cliff_asymmetric_pairs_gap_then_nogap_30", "gaps"),
            ("bilateral_goal_cliff_asymmetric_pairs_probe_ledge_30", "ledge"),
        ],
    )
    def test_terrain_family_label_uses_terrain_family_prefixes(self, terrain_variant, expected_family):
        cfg = _compose("iter_run", overrides=[f"terrain={terrain_variant}"])

        assert terrain_family_label(cfg) == expected_family

    def test_iter_run_labels_include_terrain_body_gens_pop_substeps_and_seed(self):
        cfg = _compose("iter_run", overrides=["seed=137"])

        assert run_group_label(cfg) == "cliff_rect_7x2_sep_cmaes_g50_p150_15sub_t2500_e2500_ref003"
        assert run_name_label(cfg) == "cliff_rect_7x2_sep_cmaes_g50_p150_15sub_t2500_e2500_ref003_seed137"
        assert run_metadata(cfg)["total_population"] == 150
        assert run_metadata(cfg)["gpu_profile"] == "a100_40gb"
        assert run_metadata(cfg)["reference_convergence_label"] == "ref003"
        assert run_metadata(cfg)["terrain_family"] == "cliff"
        assert run_metadata(cfg)["emitter_strategy"] == "sep_cmaes"

    def test_nodeform_label_adds_suffix_without_dropping_core_fields(self):
        cfg = _compose(
            "iter_run",
            overrides=["seed=137", "nca.ablate_deformation=true"],
        )

        assert run_group_label(cfg) == "cliff_rect_7x2_sep_cmaes_g50_p150_15sub_t2500_e2500_ref003_nodeform"
        assert run_name_label(cfg) == "cliff_rect_7x2_sep_cmaes_g50_p150_15sub_t2500_e2500_ref003_nodeform_seed137"

    def test_invalid_comb_dimensions_are_rejected(self):
        overrides = [
            "body.mode=fixed",
            "body.shape=comb",
            "terrain.robot_w=7",
            "terrain.robot_h=3",
        ]
        cfg = _compose("paper_run", overrides=overrides)

        with pytest.raises(ValueError, match="requires terrain.robot_w=7 and terrain.robot_h=2"):
            resolve_runtime_config(
                cfg,
                preset="paper_run",
                runtime_overrides=overrides,
            )

    def test_invalid_comb_7x3_dimensions_are_rejected(self):
        overrides = [
            "body.mode=fixed",
            "body.shape=comb_7x3",
            "terrain.robot_w=7",
            "terrain.robot_h=2",
        ]
        cfg = _compose("paper_run", overrides=overrides)

        with pytest.raises(ValueError, match="requires terrain.robot_w=7 and terrain.robot_h=3"):
            resolve_runtime_config(
                cfg,
                preset="paper_run",
                runtime_overrides=overrides,
            )

    def test_gasnet_single_resolves(self):
        cfg = _compose(overrides=["gasnet=single"])
        resolution = resolve_runtime_config(
            cfg,
            runtime_overrides=["gasnet=single"],
        )

        assert cfg.nca.gasnet_mode == "single"
        assert cfg.nca.gasnet_type == "gain"
        assert resolution.preset_name is None

    def test_gasnet_signal_resolves(self):
        cfg = _compose(overrides=["gasnet=signal"])
        resolution = resolve_runtime_config(
            cfg,
            runtime_overrides=["gasnet=signal"],
        )

        assert cfg.nca.gasnet_mode == "single"
        assert cfg.nca.gasnet_type == "signal"
        assert resolution.preset_name is None

    def test_timing_divisibility(self):
        for preset in RUN_PRESETS:
            cfg = _compose(preset)
            assert cfg.sim.physics_substeps % cfg.sim.sense_act_phases_per_env_step == 0

    def test_seed_override(self):
        cfg = _compose("paper_run", overrides=["seed=99"])
        resolution = resolve_runtime_config(
            cfg,
            preset="paper_run",
            runtime_overrides=["seed=99"],
        )

        assert cfg.seed == 99
        assert resolution.runtime_override_keys == ("seed",)

    def test_runtime_resolution_allows_body_iteration_overrides(self):
        overrides = [
            "body.mode=grown",
            "body.growth.max_robot_w=7",
            "body.growth.max_robot_h=3",
            "train.n_generations=10",
            "train.stop_after_generations=5",
            "train.resume_mode=staged",
            "qd.n_emitters=3",
            "train.max_steps=750",
        ]
        cfg = _compose("iter_run", overrides=overrides)
        resolution = resolve_runtime_config(
            cfg,
            preset="iter_run",
            runtime_overrides=overrides,
        )

        assert cfg.body.mode == "grown"
        assert cfg.train.n_generations == 10
        assert cfg.train.stop_after_generations == 5
        assert cfg.train.resume_mode == "staged"
        assert cfg.qd.n_emitters == 3
        assert cfg.train.max_steps == 750
        assert resolution.runtime_override_keys == tuple(overrides_key.split("=")[0] for overrides_key in overrides)

    def test_stop_after_generations_cannot_exceed_target_generations(self):
        cfg = _compose(
            "iter_run",
            overrides=[
                "train.n_generations=10",
                "train.stop_after_generations=11",
            ],
        )

        with pytest.raises(ValueError, match="cannot exceed train.n_generations"):
            resolve_runtime_config(cfg, preset="iter_run")

    def test_resume_mode_must_be_known(self):
        cfg = _compose(
            "iter_run",
            overrides=["train.resume_mode=surprise"],
        )

        with pytest.raises(ValueError, match="train.resume_mode"):
            resolve_runtime_config(cfg, preset="iter_run")

    def test_runtime_resolution_allows_robustness_probe_overrides(self):
        overrides = [
            "eval.robustness_probe.enabled=true",
            "eval.robustness_probe.max_steps=4000",
        ]
        cfg = _compose("iter_run", overrides=overrides)
        resolution = resolve_runtime_config(
            cfg,
            preset="iter_run",
            runtime_overrides=overrides,
        )

        assert cfg.eval.robustness_probe.enabled is True
        assert cfg.eval.robustness_probe.max_steps == 4000
        assert resolution.runtime_override_keys == (
            "eval.robustness_probe.enabled",
            "eval.robustness_probe.max_steps",
        )

    def test_runtime_resolution_allows_post_run_overrides(self):
        overrides = [
            "eval.post_run.enabled=true",
            "eval.post_run.max_steps=4500",
            "eval.post_run.selected_rank=0",
        ]
        cfg = _compose("paper_run", overrides=overrides)
        resolution = resolve_runtime_config(
            cfg,
            preset="paper_run",
            runtime_overrides=overrides,
        )

        assert cfg.eval.post_run.enabled is True
        assert cfg.eval.post_run.max_steps == 4500
        assert cfg.eval.post_run.selected_rank == 0
        assert resolution.runtime_override_keys == (
            "eval.post_run.enabled",
            "eval.post_run.max_steps",
            "eval.post_run.selected_rank",
        )

    def test_runtime_resolution_allows_audit_probe_overrides(self):
        overrides = [
            "audit.enabled=true",
            "audit.generation_eval_mode=full_host",
            "audit.replay_accepted_winners=true",
            "audit.compare_generation_eval_modes=true",
            "audit.repeat_full_generation_batch=true",
            "audit.compare_trace_on_full_generation_batch=true",
            "audit.compare_floor_stiction_counterfactual=true",
            "audit.save_full_generation_genotypes=true",
            "audit.probe_internal_phase_divergence=true",
            "audit.max_generation_probe_winners=8",
        ]
        cfg = _compose("iter_run", overrides=overrides)
        resolution = resolve_runtime_config(
            cfg,
            preset="iter_run",
            runtime_overrides=overrides,
        )

        assert cfg.audit.enabled is True
        assert cfg.audit.generation_eval_mode == "full_host"
        assert cfg.audit.replay_accepted_winners is True
        assert cfg.audit.compare_generation_eval_modes is True
        assert cfg.audit.repeat_full_generation_batch is True
        assert cfg.audit.compare_trace_on_full_generation_batch is True
        assert cfg.audit.compare_floor_stiction_counterfactual is True
        assert cfg.audit.save_full_generation_genotypes is True
        assert cfg.audit.probe_internal_phase_divergence is True
        assert cfg.audit.max_generation_probe_winners == 8
        assert resolution.runtime_override_keys == (
            "audit.enabled",
            "audit.generation_eval_mode",
            "audit.replay_accepted_winners",
            "audit.compare_generation_eval_modes",
            "audit.repeat_full_generation_batch",
            "audit.compare_trace_on_full_generation_batch",
            "audit.compare_floor_stiction_counterfactual",
            "audit.save_full_generation_genotypes",
            "audit.probe_internal_phase_divergence",
            "audit.max_generation_probe_winners",
        )

    def test_runtime_resolution_rejects_unknown_audit_generation_eval_mode(self):
        cfg = _compose(
            "iter_run",
            overrides=[
                "audit.enabled=true",
                "audit.generation_eval_mode=surprise",
            ],
        )

        with pytest.raises(ValueError, match="audit.generation_eval_mode"):
            resolve_runtime_config(cfg, preset="iter_run")

    @pytest.mark.parametrize("gpu_profile", ["a100_40gb", "a100_80gb", "h100", "h200", "b200"])
    def test_gpu_profile_override_resolves_and_propagates_metadata(self, gpu_profile):
        overrides = [
            f"gpu_profile={gpu_profile}",
            "qd.n_emitters=10",
            "qd.emitter_pop_size=30",
        ]
        cfg = _compose("iter_run", overrides=overrides)
        resolution = resolve_runtime_config(
            cfg,
            preset="iter_run",
            runtime_overrides=overrides,
        )
        assert cfg.gpu_profile.id == gpu_profile
        assert resolution.gpu_profile == gpu_profile
        assert run_metadata(cfg)["gpu_profile"] == gpu_profile

    def test_qd_descriptor_override_derives_dynamic_archive_defaults(self):
        cfg = _compose(
            overrides=["qd.descriptors.names=[fall_count,cliff_vicinity]"],
        )
        qd_archive = resolve_qd_archive(cfg)

        assert qd_archive.descriptor_names == ("fall_count", "cliff_vicinity")
        assert qd_archive.descriptor_labels == ("Fall Count", "Cliff Vicinity")
        assert qd_archive.dims == (int(cfg.train.n_episodes) + 1, 20)
        assert qd_archive.ranges == ((0.0, float(cfg.train.n_episodes)), (0.0, 1.0))

    def test_qd_goal_hit_count_descriptor_uses_episode_count_defaults(self):
        cfg = _compose(
            overrides=["qd.descriptors.names=[goal_hit_count,fall_count]"],
        )
        qd_archive = resolve_qd_archive(cfg)

        assert qd_archive.descriptor_names == ("goal_hit_count", "fall_count")
        assert qd_archive.descriptor_labels == ("Goal Hit Count", "Fall Count")
        assert qd_archive.dims == (int(cfg.train.n_episodes) + 1, int(cfg.train.n_episodes) + 1)
        assert qd_archive.ranges == (
            (0.0, float(cfg.train.n_episodes)),
            (0.0, float(cfg.train.n_episodes)),
        )

    def test_qd_mirrored_hit_count_descriptors_use_episode_count_defaults(self):
        cfg = _compose(
            overrides=[
                "qd.descriptors.names=[mirrored_pair_count,mirrored_best_hit_count,mirrored_total_hit_count]"
            ],
        )
        qd_archive = resolve_qd_archive(cfg)

        assert qd_archive.descriptor_names == (
            "mirrored_pair_count",
            "mirrored_best_hit_count",
            "mirrored_total_hit_count",
        )
        assert qd_archive.descriptor_labels == (
            "Mirrored Pair Count",
            "Mirrored Best Hit Count",
            "Mirrored Total Hit Count",
        )
        assert qd_archive.dims == (
            int(cfg.train.n_episodes) + 1,
            int(cfg.train.n_episodes) + 1,
            int(cfg.train.n_episodes) + 1,
        )
        assert qd_archive.ranges == (
            (0.0, float(cfg.train.n_episodes)),
            (0.0, float(cfg.train.n_episodes)),
            (0.0, float(cfg.train.n_episodes)),
        )

    def test_qd_stable_turn_count_descriptor_uses_quarter_cell_archive_defaults(self):
        cfg = _compose(
            overrides=["qd.descriptors.names=[goal_hit_count,stable_turn_count]"],
        )
        qd_archive = resolve_qd_archive(cfg)

        assert qd_archive.descriptor_names == ("goal_hit_count", "stable_turn_count")
        assert qd_archive.dims == (int(cfg.train.n_episodes) + 1, 24)
        assert qd_archive.ranges == (
            (0.0, float(cfg.train.n_episodes)),
            (0.0, 6.0),
        )

    def test_qd_axis_activation_ranges_are_tight(self):
        cfg = _compose(
            "paper_run",
            overrides=["qd.descriptors.names=[h_activation,v_activation]"],
        )
        qd_archive = resolve_qd_archive(cfg)

        assert qd_archive.descriptor_names == ("h_activation", "v_activation")
        assert qd_archive.ranges == ((0.0, 1.0), (0.0, 1.0))

    def test_qd_archive_override_replaces_descriptor_defaults(self):
        cfg = _compose(
            "paper_run",
            overrides=[
                "qd.descriptors.names=[fall_count,cliff_vicinity]",
                "qd.archive.dims=[9,13]",
                "qd.archive.ranges=[[0.0,8.0],[0.2,0.9]]",
            ],
        )
        qd_archive = resolve_qd_archive(cfg)

        assert qd_archive.dims == (9, 13)
        assert qd_archive.ranges == ((0.0, 8.0), (0.2, 0.9))

    def test_qd_archive_supports_three_descriptors(self):
        cfg = _compose(
            "paper_run",
            overrides=[
                "qd.descriptors.names=[goal_hit_count,fall_count,mirrored_roundtrip_top2_mean]",
                "qd.archive.dims=null",
                "qd.archive.ranges=null",
            ],
        )
        qd_archive = resolve_qd_archive(cfg)

        assert qd_archive.descriptor_names == (
            "goal_hit_count",
            "fall_count",
            "mirrored_roundtrip_top2_mean",
        )
        assert qd_archive.dims == (
            int(cfg.train.n_episodes) + 1,
            int(cfg.train.n_episodes) + 1,
            20,
        )
        assert qd_archive.ranges == (
            (0.0, float(cfg.train.n_episodes)),
            (0.0, float(cfg.train.n_episodes)),
            (0.0, 1.0),
        )

    def test_duplicate_qd_descriptors_are_rejected(self):
        cfg = _compose(overrides=["qd.descriptors.names=[activation_cost,activation_cost]"])

        with pytest.raises(ValueError, match="must not contain duplicates"):
            resolve_runtime_config(cfg)
