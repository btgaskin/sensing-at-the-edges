"""Tests for render surface helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp
import pytest

import sensing_edges.render as render_module
from jax_evogym.types import RenderStepOutput
from sensing_edges.batch_eval import ChunkedEvalStats, ChunkedTraceEvalResult, GrownEvalAssets, GrownRenderPrep
from sensing_edges.collision_tuning import CollisionBuildCapacity
from sensing_edges.env import (
    EpisodeResult,
    build_static_collider_template,
    infer_static_template_capacity,
    precompute_shared_grid_data,
    stack_static_templates,
)
from sensing_edges.qd.grid_archive import GridArchive
from sensing_edges.runtime_config import compose_experiment_config, resolve_runtime_config
from sensing_edges.terrain import build_cliff_terrain, make_robot_body
from sensing_edges.terrain_schedule import TerrainEntry, TerrainSchedule


def test_render_ranked_elites_uses_grown_assets_directly(tmp_path, monkeypatch):
    cfg = compose_experiment_config(
        preset="iter_run",
        overrides=["body.mode=grown", "train.max_steps=3"],
    )
    resolution = resolve_runtime_config(cfg, preset="iter_run")
    terrain = build_cliff_terrain(cfg.terrain.grid_h, cfg.terrain.grid_w, cliff_x=15)
    template = build_static_collider_template(
        terrain,
        cliff_x=15,
        canvas_h=cfg.body.growth.max_robot_h,
        canvas_w=cfg.body.growth.max_robot_w,
        spawn_x=cfg.terrain.spawn_x,
        spawn_y=cfg.terrain.spawn_y,
    )
    static_capacity = infer_static_template_capacity([template])
    env_assets = GrownEvalAssets(
        static_templates=stack_static_templates([template], static_capacity=static_capacity),
        templates=(template,),
        static_capacity=static_capacity,
        grid_data=precompute_shared_grid_data(cfg.terrain.grid_h, cfg.terrain.grid_w),
        runtime_build_capacity=CollisionBuildCapacity(max_triples=256, max_self_triples=64),
    )
    captured: dict[str, object] = {}

    def _fake_render_selected_elites(
        elites,
        env_datas,
        alive_mask,
        cfg,
        *,
        generation,
        key_scope,
        stage,
        output_dir,
        runtime_resolution=None,
    ):
        captured["env_datas"] = env_datas
        return []

    monkeypatch.setattr(render_module, "_render_selected_elites", _fake_render_selected_elites)
    artifacts = render_module.render_ranked_elites(
        elites=[{"genotype": np.zeros((1,), dtype=np.float32), "fitness": 1.0, "flat_index": 0}],
        env_datas=env_assets,
        alive_mask=jnp.ones((cfg.body.growth.max_robot_h, cfg.body.growth.max_robot_w), dtype=jnp.bool_),
        cfg=cfg,
        generation=0,
        stage="eval",
        output_dir=tmp_path,
        runtime_resolution=resolution,
    )

    assert artifacts == []
    assert captured["env_datas"] is env_assets


def test_render_ranked_elites_grown_uses_trace_render_prep_without_growth_replay(tmp_path, monkeypatch):
    cfg = compose_experiment_config(
        preset="iter_run",
        overrides=["body.mode=grown", "train.max_steps=3"],
    )
    resolution = resolve_runtime_config(cfg, preset="iter_run")
    terrain = build_cliff_terrain(cfg.terrain.grid_h, cfg.terrain.grid_w, cliff_x=15)
    template = build_static_collider_template(
        terrain,
        cliff_x=15,
        canvas_h=cfg.body.growth.max_robot_h,
        canvas_w=cfg.body.growth.max_robot_w,
        spawn_x=cfg.terrain.spawn_x,
        spawn_y=cfg.terrain.spawn_y,
    )
    static_capacity = infer_static_template_capacity([template])
    env_assets = GrownEvalAssets(
        static_templates=stack_static_templates([template], static_capacity=static_capacity),
        templates=(template,),
        static_capacity=static_capacity,
        grid_data=precompute_shared_grid_data(cfg.terrain.grid_h, cfg.terrain.grid_w),
        runtime_build_capacity=CollisionBuildCapacity(max_triples=256, max_self_triples=64),
    )
    robot_morph = make_robot_body(cfg.body.growth.max_robot_w, cfg.body.growth.max_robot_h)
    n_points = int(np.asarray(env_assets.grid_data.positions).shape[0])
    n_springs = int(np.asarray(env_assets.grid_data.spring_rest_length).shape[0])

    def _fake_trace_population_chunked(*args, **kwargs):
        return ChunkedTraceEvalResult(
            episode_results=EpisodeResult(
                peak_front_x=jnp.asarray([[1.5]], dtype=jnp.float32),
                safe_front_x=jnp.asarray([[1.0]], dtype=jnp.float32),
                init_front_x=jnp.asarray([[0.5]], dtype=jnp.float32),
                init_com_x=jnp.asarray([[0.5]], dtype=jnp.float32),
                touched_death_plane=jnp.asarray([[False]]),
                went_airborne=jnp.asarray([[False]]),
                com_x_went_negative=jnp.asarray([[False]]),
                mean_activation=jnp.asarray([[0.1]], dtype=jnp.float32),
                mean_h_activation=jnp.asarray([[0.05]], dtype=jnp.float32),
                mean_v_activation=jnp.asarray([[0.05]], dtype=jnp.float32),
                mean_speed=jnp.asarray([[0.1]], dtype=jnp.float32),
                cliff_vicinity_steps=jnp.asarray([[1]], dtype=jnp.int32),
                cliff_vicinity_fraction=jnp.asarray([[0.25]], dtype=jnp.float32),
                time_to_edge=jnp.asarray([[0.5]], dtype=jnp.float32),
                elapsed_time_fraction=jnp.asarray([[0.5]], dtype=jnp.float32),
                steps_alive=jnp.asarray([[3]], dtype=jnp.int32),
                cliff_x=jnp.asarray([[15]], dtype=jnp.int32),
                task_mode=jnp.asarray([[0]], dtype=jnp.int32),
                goal_direction=jnp.asarray([[0]], dtype=jnp.int32),
                cliff_side=jnp.asarray([[1]], dtype=jnp.int32),
                goal_distance=jnp.asarray([[0]], dtype=jnp.int32),
                cliff_distance=jnp.asarray([[0]], dtype=jnp.int32),
                goal_block_x=jnp.asarray([[-1]], dtype=jnp.int32),
                goal_block_y=jnp.asarray([[-1]], dtype=jnp.int32),
                goal_x=jnp.asarray([[-1.0]], dtype=jnp.float32),
                any_progress=jnp.asarray([[0.0]], dtype=jnp.float32),
                goal_progress=jnp.asarray([[0.0]], dtype=jnp.float32),
                incorrect_progress=jnp.asarray([[0.0]], dtype=jnp.float32),
                toward_goal_frontier=jnp.asarray([[0.0]], dtype=jnp.float32),
                goal_touched=jnp.asarray([[False]], dtype=jnp.bool_),
                cliff_fell=jnp.asarray([[False]], dtype=jnp.bool_),
                time_to_goal=jnp.asarray([[1.0]], dtype=jnp.float32),
                turn_count=jnp.asarray([[0]], dtype=jnp.int32),
                deformation_channel_max=jnp.zeros((1, 1, 4), dtype=jnp.float32),
                controller_hidden_output_max_abs=jnp.zeros((1, 1), dtype=jnp.float32),
                action_output_max_abs=jnp.zeros((1, 1), dtype=jnp.float32),
            ),
            traces=RenderStepOutput(
                positions=jnp.zeros((1, 1, 2, n_points, 2), dtype=jnp.float32),
                spring_rest_length=jnp.ones((1, 1, 2, n_springs), dtype=jnp.float32),
                reward=jnp.zeros((1, 1, 2), dtype=jnp.float32),
                done=jnp.zeros((1, 1, 2), dtype=jnp.bool_),
            ),
            render_preps=GrownRenderPrep(
                robot_morph=jnp.asarray(robot_morph[None, ...], dtype=jnp.int32),
                spring_init_rest_length=jnp.asarray(
                    np.asarray(env_assets.grid_data.spring_rest_length)[None, ...],
                    dtype=jnp.float32,
                ),
            ),
            stats=ChunkedEvalStats(
                workload="trace",
                elapsed_s=0.0,
                candidate_count=1,
                chunk_count=1,
                bucket_counts={1: 1},
            ),
        )

    def _fake_render_episode_gif(*args, **kwargs):
        path = kwargs.get("output_path", args[2])
        Path(path).write_text("gif")
        return str(path)

    monkeypatch.setattr(render_module, "trace_population_chunked", _fake_trace_population_chunked)
    monkeypatch.setattr(render_module, "render_episode_gif", _fake_render_episode_gif)

    artifacts = render_module.render_ranked_elites(
        elites=[{"genotype": np.zeros((4,), dtype=np.float32), "fitness": 1.0, "flat_index": 0}],
        env_datas=env_assets,
        alive_mask=jnp.ones((cfg.body.growth.max_robot_h, cfg.body.growth.max_robot_w), dtype=jnp.bool_),
        cfg=cfg,
        generation=0,
        stage="eval",
        output_dir=tmp_path,
        runtime_resolution=resolution,
    )

    assert len(artifacts) == 1
    assert Path(artifacts[0].gif_path).exists()


def test_render_config_for_stage_uses_hybrid_temporal_policy():
    cfg = compose_experiment_config(preset="iter_run")

    periodic_cfg = render_module._render_config_for_stage(cfg, "periodic")
    post_train_cfg = render_module._render_config_for_stage(cfg, "eval")

    assert periodic_cfg.real_time is False
    assert periodic_cfg.frame_skip == 1
    assert periodic_cfg.width == 360
    assert periodic_cfg.height == 180
    assert periodic_cfg.fps == 16
    assert periodic_cfg.target_frames == 160
    assert periodic_cfg.sim_time_per_step == 15 * 0.0001

    assert post_train_cfg.real_time is True
    assert post_train_cfg.frame_skip == 1
    assert post_train_cfg.target_frames is None
    assert post_train_cfg.sim_time_per_step == 15 * 0.0001


def test_render_periodic_archive_uses_full_generation_batch_by_default(tmp_path, monkeypatch):
    cfg = compose_experiment_config(preset="iter_run")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        render_module,
        "select_priority_render_elites",
        lambda *args, **kwargs: [
            {
                "genotype": np.zeros((1,), dtype=np.float32),
                "fitness": 1.0,
                "flat_index": 0,
                "descriptor": np.asarray([6.0, 0.0, 0.1], dtype=np.float32),
            }
        ],
    )

    def _fake_render_selected_elites(
        elites,
        env_datas,
        alive_mask,
        cfg,
        *,
        generation,
        key_scope,
        stage,
        output_dir,
        runtime_resolution=None,
    ):
        del elites, alive_mask, cfg, generation, key_scope, stage, output_dir, runtime_resolution
        captured["env_datas"] = env_datas
        return []

    monkeypatch.setattr(render_module, "_render_selected_elites", _fake_render_selected_elites)

    env_datas = [object() for _ in range(6)]
    artifacts = render_module.render_periodic_archive(
        archive=SimpleNamespace(
            size=1,
            descriptor_names=("goal_hit_count", "fall_count", "completion_time"),
        ),
        env_datas=env_datas,
        alive_mask=jnp.ones((1, 1), dtype=jnp.bool_),
        cfg=cfg,
        generation=19,
        output_root=tmp_path,
    )

    assert artifacts == []
    assert captured["env_datas"] == env_datas


def test_render_periodic_archive_supports_prefix_subset_selection(tmp_path, monkeypatch):
    cfg = compose_experiment_config(
        preset="iter_run",
        overrides=["render.periodic.terrain_selection=prefix"],
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        render_module,
        "select_priority_render_elites",
        lambda *args, **kwargs: [
            {
                "genotype": np.zeros((1,), dtype=np.float32),
                "fitness": 1.0,
                "flat_index": 0,
                "descriptor": np.asarray([6.0, 0.0, 0.1], dtype=np.float32),
            }
        ],
    )

    def _fake_render_selected_elites(
        elites,
        env_datas,
        alive_mask,
        cfg,
        *,
        generation,
        key_scope,
        stage,
        output_dir,
        runtime_resolution=None,
    ):
        del elites, alive_mask, cfg, generation, key_scope, stage, output_dir, runtime_resolution
        captured["env_datas"] = env_datas
        return []

    monkeypatch.setattr(render_module, "_render_selected_elites", _fake_render_selected_elites)

    env_datas = [object() for _ in range(6)]
    artifacts = render_module.render_periodic_archive(
        archive=SimpleNamespace(
            size=1,
            descriptor_names=("goal_hit_count", "fall_count", "completion_time"),
        ),
        env_datas=env_datas,
        alive_mask=jnp.ones((1, 1), dtype=jnp.bool_),
        cfg=cfg,
        generation=19,
        output_root=tmp_path,
    )

    assert artifacts == []
    assert captured["env_datas"] == env_datas


def test_select_projected_render_elites_uses_named_descriptor_targets():
    archive = GridArchive(
        dims=(7, 7, 4),
        ranges=[(0.0, 6.0), (0.0, 6.0), (0.0, 3.0)],
        descriptor_names=("fall_count", "goal_hit_count", "stable_turn_count"),
    )
    archive.add(
        np.array([[11.0], [22.0], [33.0], [44.0]], dtype=np.float32),
        np.array([10.0, 40.0, 30.0, 20.0], dtype=np.float64),
        np.array(
            [
                [0.0, 6.0, 0.0],
                [0.0, 6.0, 1.0],
                [0.0, 6.0, 2.0],
                [1.0, 6.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )

    elites = render_module.select_projected_render_elites(
        archive,
        descriptor_targets={"fall_count": "min", "goal_hit_count": "max"},
        max_individuals=2,
    )

    assert [float(item["fitness"]) for item in elites] == [40.0, 30.0]


def test_select_priority_render_elites_uses_lexicographic_descriptor_order():
    archive = GridArchive(
        dims=(7, 7, 10),
        ranges=[(0.0, 6.0), (0.0, 6.0), (0.0, 1.0)],
        descriptor_names=("goal_hit_count", "fall_count", "completion_time"),
    )
    archive.add(
        np.array([[11.0], [22.0], [33.0], [44.0]], dtype=np.float32),
        np.array([20.0, 100.0, 10.0, 5.0], dtype=np.float64),
        np.array(
            [
                [6.0, 0.0, 0.3],
                [6.0, 1.0, 0.1],
                [6.0, 0.0, 0.5],
                [5.0, 0.0, 0.1],
            ],
            dtype=np.float64,
        ),
    )

    elites = render_module.select_priority_render_elites(
        archive,
        descriptor_priorities=(
            ("goal_hit_count", "max"),
            ("fall_count", "min"),
            ("completion_time", "min"),
        ),
        max_individuals=3,
    )

    assert [float(item["fitness"]) for item in elites] == [20.0, 10.0, 100.0]


def test_periodic_qd_render_entries_use_last_generation_of_each_phase():
    def _entry(gen: int, episode: int) -> TerrainEntry:
        return TerrainEntry(
            grid=np.full((2, 2), gen * 10 + episode, dtype=np.int32),
            cliff_x=15,
            gaps=[],
            metadata={"gen": gen, "episode": episode},
        )

    schedule = TerrainSchedule(
        train=[
            [_entry(0, 0), _entry(0, 1)],
            [_entry(1, 0), _entry(1, 1)],
            [_entry(2, 0), _entry(2, 1)],
            [_entry(3, 0), _entry(3, 1)],
            [_entry(4, 0), _entry(4, 1)],
            [_entry(5, 0), _entry(5, 1)],
        ],
        val=[],
        test=[],
        seed=0,
        terrain_cfg={},
        curriculum=True,
        curriculum_phase_end_gens=[2, 4, 6],
    )

    entries = render_module._periodic_qd_render_entries(schedule, generation=5)

    assert [int(entry.metadata["gen"]) for entry in entries] == [1, 1, 3, 3, 5, 5]


def test_render_periodic_qd_projection_uses_cross_phase_entries(tmp_path, monkeypatch):
    cfg = compose_experiment_config(
        preset="iter_run",
        overrides=["train.terrain_schedule_dir=/tmp/fake_schedule"],
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        render_module,
        "select_priority_render_elites",
        lambda *args, **kwargs: [
            {
                "genotype": np.zeros((1,), dtype=np.float32),
                "fitness": 1.0,
                "flat_index": 0,
                "descriptor": np.asarray([6.0, 0.0, 0.1], dtype=np.float32),
            }
        ],
    )
    monkeypatch.setattr(
        render_module,
        "load_schedule",
        lambda _schedule_dir, **kwargs: TerrainSchedule(
            train=[
                [
                    TerrainEntry(
                        grid=np.zeros((2, 2), dtype=np.int32),
                        cliff_x=15,
                        gaps=[],
                        metadata={"gen": gen, "episode": episode},
                    )
                    for episode in range(6)
                ]
                for gen in range(6)
            ],
            val=[],
            test=[],
            seed=0,
            terrain_cfg={},
            curriculum=True,
            curriculum_phase_end_gens=[2, 4, 6],
        ),
    )

    def _fake_build_envs_from_entries(entries, cfg, runtime_resolution):
        del cfg, runtime_resolution
        captured["entry_gens"] = [int(entry.metadata["gen"]) for entry in entries]
        return ["env"] * len(entries)

    def _fake_render_selected_elites(
        elites,
        env_datas,
        alive_mask,
        cfg,
        *,
        generation,
        key_scope,
        stage,
        output_dir,
        runtime_resolution=None,
    ):
        del elites, alive_mask, cfg, generation, key_scope, runtime_resolution
        captured["env_count"] = len(env_datas)
        captured["stage"] = stage
        captured["output_dir"] = output_dir
        return []

    monkeypatch.setattr(render_module, "_build_envs_from_entries", _fake_build_envs_from_entries)
    monkeypatch.setattr(render_module, "_render_selected_elites", _fake_render_selected_elites)

    artifacts = render_module.render_periodic_qd_projection(
        archive=SimpleNamespace(
            descriptor_names=("goal_hit_count", "fall_count", "completion_time")
        ),
        alive_mask=jnp.ones((1, 1), dtype=jnp.bool_),
        cfg=cfg,
        generation=5,
        output_root=tmp_path,
        runtime_resolution=resolve_runtime_config(cfg, preset="iter_run"),
    )

    assert artifacts == []
    assert captured["entry_gens"] == [1] * 6 + [3] * 6 + [5] * 6
    assert captured["env_count"] == 18
    assert captured["stage"] == "periodic_qd"


def test_render_periodic_qd_projection_skips_when_stable_turns_are_zero(tmp_path, monkeypatch):
    cfg = compose_experiment_config(
        preset="iter_run",
        overrides=["train.terrain_schedule_dir=/tmp/fake_schedule"],
    )

    archive = SimpleNamespace(size=1, descriptor_names=("goal_hit_count", "fall_count"))
    artifacts = render_module.render_periodic_qd_projection(
        archive=archive,
        alive_mask=jnp.ones((1, 1), dtype=jnp.bool_),
        cfg=cfg,
        generation=5,
        output_root=tmp_path,
        runtime_resolution=resolve_runtime_config(cfg, preset="iter_run"),
    )

    assert artifacts == []


def test_encode_renderable_episode_keeps_full_trace_until_renderer(tmp_path, monkeypatch):
    cfg = compose_experiment_config(preset="iter_run", overrides=["train.max_steps=2000"])
    captured: dict[str, object] = {}

    def _fake_render_episode_gif(trace, render_info, output_path, **kwargs):
        del render_info
        captured["n_steps"] = int(np.asarray(trace.positions).shape[0])
        captured["config"] = kwargs["config"]
        Path(output_path).write_text("gif")
        return str(output_path)

    monkeypatch.setattr(render_module, "render_episode_gif", _fake_render_episode_gif)

    artifact = render_module._encode_renderable_episode(
        render_module.RenderableEpisode(
            stage="periodic",
            rank=0,
            episode_idx=0,
            flat_index=0,
            archive_fitness=1.0,
            selection_fitness=1.0,
            cliff_x=15,
            terrain_idx=0,
            render_info=object(),
            episode_result=EpisodeResult(
                peak_front_x=jnp.float32(1.5),
                safe_front_x=jnp.float32(1.0),
                init_front_x=jnp.float32(0.5),
                init_com_x=jnp.float32(0.5),
                touched_death_plane=jnp.bool_(False),
                went_airborne=jnp.bool_(False),
                com_x_went_negative=jnp.bool_(False),
                mean_activation=jnp.float32(0.1),
                mean_h_activation=jnp.float32(0.05),
                mean_v_activation=jnp.float32(0.05),
                mean_speed=jnp.float32(0.1),
                cliff_vicinity_steps=jnp.int32(1),
                cliff_vicinity_fraction=jnp.float32(0.25),
                time_to_edge=jnp.float32(0.5),
                elapsed_time_fraction=jnp.float32(1.0),
                steps_alive=jnp.int32(2000),
                cliff_x=jnp.int32(15),
                task_mode=jnp.int32(0),
                goal_direction=jnp.int32(0),
                cliff_side=jnp.int32(1),
                goal_distance=jnp.int32(0),
                cliff_distance=jnp.int32(0),
                goal_block_x=jnp.int32(-1),
                goal_block_y=jnp.int32(-1),
                goal_x=jnp.float32(-1.0),
                any_progress=jnp.float32(0.0),
                goal_progress=jnp.float32(0.0),
                incorrect_progress=jnp.float32(0.0),
                toward_goal_frontier=jnp.float32(0.0),
                goal_touched=jnp.bool_(False),
                cliff_fell=jnp.bool_(False),
                time_to_goal=jnp.float32(1.0),
                turn_count=jnp.int32(0),
                deformation_channel_max=jnp.zeros((4,), dtype=jnp.float32),
                controller_hidden_output_max_abs=jnp.float32(0.0),
                action_output_max_abs=jnp.float32(0.0),
                stable_turn_count=jnp.int32(1),
                turn_motion_threshold=jnp.float32(0.012),
            ),
            trace=RenderStepOutput(
                positions=jnp.zeros((2000, 1, 2), dtype=jnp.float32),
                spring_rest_length=jnp.ones((2000, 1), dtype=jnp.float32),
                reward=jnp.zeros((2000,), dtype=jnp.float32),
                done=jnp.zeros((2000,), dtype=jnp.bool_),
            ),
            output_path=tmp_path / "rank_0_ep_0.gif",
            goal_hit_count=6.0,
            fall_count=0.0,
            completion_time=0.2,
        ),
        cfg=cfg,
    )

    assert captured["n_steps"] == 2000
    assert captured["config"].target_frames == 160
    assert Path(artifact.gif_path).exists()
    assert artifact.goal_hit_count == pytest.approx(6.0)
    assert artifact.fall_count == pytest.approx(0.0)
    assert artifact.completion_time == pytest.approx(0.2)
    assert artifact.turn_count == pytest.approx(0.0)
    assert artifact.stable_turn_count == pytest.approx(1.0)
    assert artifact.turn_motion_threshold == pytest.approx(0.012)
