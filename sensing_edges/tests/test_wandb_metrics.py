"""Tests for W&B metric naming and grouping."""

from __future__ import annotations

import numpy as np

from sensing_edges.qd.focus import FocusWindowBounds
from sensing_edges.qd.grid_archive import GridArchive
from sensing_edges.runtime_config import compose_experiment_config
from sensing_edges.wandb_metrics import (
    define_wandb_metrics,
    init_wandb,
    render_archive_heatmap,
    render_archive_named_heatmaps,
    render_archive_heatmap_time_falls,
    render_archive_heatmap_time_hits,
    render_archive_third_descriptor_heatmap,
)


class _FakeWandb:
    def __init__(self):
        self.defined = []
        self.init_kwargs = None
        self.images = []

    def define_metric(self, *args, **kwargs):
        self.defined.append((args, kwargs))

    def init(self, **kwargs):
        self.init_kwargs = kwargs

    def Image(self, fig):
        self.images.append(fig)
        return {"figure": fig}


def test_define_wandb_metrics_includes_post_train_namespace(monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake)

    define_wandb_metrics()

    assert any(args == ("post_train/*",) for args, _ in fake.defined)
    assert any(args == ("deformation/*",) for args, _ in fake.defined)
    assert any(args == ("alignment/*",) for args, _ in fake.defined)
    assert any(args == ("descriptors/*",) for args, _ in fake.defined)
    assert any(args == ("fitness/goal_hit_mean",) for args, _ in fake.defined)
    assert any(args == ("fitness/mean_mirrored_pair_count",) for args, _ in fake.defined)
    assert any(args == ("fitness/lower_ledge_penalty_mean",) for args, _ in fake.defined)
    assert any(args == ("qd/axis_coverage/*",) for args, _ in fake.defined)
    assert any(args == ("qd/axis_occupied_bins/*",) for args, _ in fake.defined)
    assert any(args == ("qd/focus/*",) for args, _ in fake.defined)
    assert any(args == ("terrain_diag/g2_c1/hit_rate",) for args, _ in fake.defined)
    assert any(args == ("terrain_diag/all/best_goal_time",) for args, _ in fake.defined)
    assert not any(args == ("fitness/null_rate",) for args, _ in fake.defined)


def test_init_wandb_uses_body_budget_ref_and_seed_naming(monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake)
    cfg = compose_experiment_config(preset="iter_run", overrides=["seed=137", "gpu_profile=h100"])

    assert init_wandb(cfg) is True
    assert fake.init_kwargs is not None
    assert fake.init_kwargs["group"] == "iter_run_cliff_rect_7x2_sep_cmaes_g50_p150_15sub_t2500_e2500_ref003"
    assert fake.init_kwargs["name"] == "iter_run_cliff_rect_7x2_sep_cmaes_g50_p150_15sub_t2500_e2500_ref003_seed137"
    assert "rect_7x2" in fake.init_kwargs["tags"]
    assert "ea_sep_cmaes" in fake.init_kwargs["tags"]
    assert "gpu_h100" in fake.init_kwargs["tags"]
    assert "gens_50" in fake.init_kwargs["tags"]
    assert "pop_150" in fake.init_kwargs["tags"]
    assert "substeps_15" in fake.init_kwargs["tags"]
    assert "train_2500" in fake.init_kwargs["tags"]
    assert "rerank_2500" in fake.init_kwargs["tags"]
    assert "test_2500" in fake.init_kwargs["tags"]
    assert "ref003" in fake.init_kwargs["tags"]


def test_init_wandb_accepts_stable_run_id_and_resume(monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake)
    cfg = compose_experiment_config(preset="iter_run")

    assert init_wandb(cfg, run_id="stable-run", resume_mode="allow") is True
    assert fake.init_kwargs["id"] == "stable-run"
    assert fake.init_kwargs["resume"] == "allow"


def test_render_archive_heatmap_projects_three_dimensional_archive(monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake)
    archive = GridArchive(
        dims=(7, 7, 20),
        ranges=[(0.0, 6.0), (0.0, 6.0), (0.0, 1.0)],
        descriptor_names=("goal_hit_count", "fall_count", "mirrored_roundtrip_top2_mean"),
        descriptor_labels=("Goal Hit Count", "Fall Count", "Mirrored Roundtrip Top-2 Mean"),
    )
    for cell, fitness, descriptor in [
        ((3, 1, 4), 12.0, np.asarray([3.0, 1.0, 0.2], dtype=np.float32)),
        ((3, 1, 16), 18.0, np.asarray([3.0, 1.0, 0.8], dtype=np.float32)),
    ]:
        flat_idx = archive._ravel(cell)
        archive._genotypes[flat_idx] = np.zeros((1,), dtype=np.float32)
        archive._fitnesses[flat_idx] = fitness
        archive._descriptors[flat_idx] = descriptor

    image = render_archive_heatmap(archive)

    assert image is not None
    assert len(fake.images) == 1


def test_render_archive_third_descriptor_heatmap_requires_three_axes(monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake)
    archive = GridArchive(
        dims=(10, 10),
        ranges=[(0.0, 1.0), (0.0, 1.0)],
        descriptor_names=("correct_progress_top2_mean", "incorrect_progress_top2_mean"),
        descriptor_labels=("Correct Progress Top-2 Mean", "Incorrect Progress Top-2 Mean"),
    )

    assert render_archive_third_descriptor_heatmap(archive) is None


def test_render_archive_third_descriptor_heatmap_projects_descriptor_values(monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake)
    archive = GridArchive(
        dims=(7, 7, 20),
        ranges=[(0.0, 6.0), (0.0, 6.0), (0.0, 1.0)],
        descriptor_names=("goal_hit_count", "fall_count", "mirrored_roundtrip_top2_mean"),
        descriptor_labels=("Goal Hit Count", "Fall Count", "Mirrored Roundtrip Top-2 Mean"),
    )
    for cell, fitness, descriptor in [
        ((4, 2, 6), 10.0, np.asarray([4.0, 2.0, 0.3], dtype=np.float32)),
        ((4, 2, 18), 9.0, np.asarray([4.0, 2.0, 0.9], dtype=np.float32)),
    ]:
        flat_idx = archive._ravel(cell)
        archive._genotypes[flat_idx] = np.zeros((1,), dtype=np.float32)
        archive._fitnesses[flat_idx] = fitness
        archive._descriptors[flat_idx] = descriptor

    image = render_archive_third_descriptor_heatmap(archive)

    assert image is not None
    assert len(fake.images) == 1


def test_render_archive_heatmap_time_hits_projects_three_dimensional_archive(monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake)
    archive = GridArchive(
        dims=(7, 7, 20),
        ranges=[(0.0, 6.0), (0.0, 6.0), (0.2, 0.8)],
        descriptor_names=("goal_hit_count", "fall_count", "completion_time"),
        descriptor_labels=("Goal Hit Count", "Fall Count", "Completion Time"),
    )
    archive.add(
        np.array([[1.0], [2.0]], dtype=np.float32),
        np.array([12.0, 18.0], dtype=np.float64),
        np.array([[4.0, 1.0, 0.3], [6.0, 0.0, 0.7]], dtype=np.float64),
    )

    image = render_archive_heatmap_time_hits(archive)

    assert image is not None
    assert len(fake.images) == 1


def test_render_archive_heatmap_time_falls_projects_three_dimensional_archive(monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake)
    archive = GridArchive(
        dims=(7, 7, 20),
        ranges=[(0.0, 6.0), (0.0, 6.0), (0.2, 0.8)],
        descriptor_names=("goal_hit_count", "fall_count", "completion_time"),
        descriptor_labels=("Goal Hit Count", "Fall Count", "Completion Time"),
    )
    archive.add(
        np.array([[1.0], [2.0]], dtype=np.float32),
        np.array([11.0, 19.0], dtype=np.float64),
        np.array([[4.0, 2.0, 0.4], [6.0, 0.0, 0.75]], dtype=np.float64),
    )

    image = render_archive_heatmap_time_falls(archive)

    assert image is not None
    assert len(fake.images) == 1


def test_render_archive_named_heatmaps_returns_requested_pairwise_images(monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake)
    archive = GridArchive(
        dims=(7, 7, 20),
        ranges=[(0.0, 6.0), (0.0, 6.0), (0.2, 0.8)],
        descriptor_names=("goal_hit_count", "fall_count", "completion_time"),
        descriptor_labels=("Goal Hit Count", "Fall Count", "Completion Time"),
    )
    archive.add(
        np.array([[1.0], [2.0]], dtype=np.float32),
        np.array([11.0, 19.0], dtype=np.float64),
        np.array([[4.0, 2.0, 0.4], [6.0, 0.0, 0.75]], dtype=np.float64),
    )

    images = render_archive_named_heatmaps(
        archive,
        ["hits_falls", "time_hits", "time_falls", "time_hits"],
    )

    assert tuple(images.keys()) == (
        "hits_falls_heatmap",
        "time_hits_heatmap",
        "time_falls_heatmap",
    )
    assert len(fake.images) == 3


def test_render_archive_heatmap_overlays_outside_focus_window(monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake)
    archive = GridArchive(
        dims=(3, 3, 2),
        ranges=[(0.0, 2.0), (0.0, 2.0), (0.0, 1.0)],
        descriptor_names=("goal_hit_count", "fall_count", "completion_time"),
        descriptor_labels=("Goal Hit Count", "Fall Count", "Completion Time"),
    )
    archive.add(
        np.array([[1.0], [2.0]], dtype=np.float32),
        np.array([4.0, 6.0], dtype=np.float64),
        np.array([[2.0, 0.0, 0.0], [2.0, 2.0, 1.0]], dtype=np.float64),
    )
    focus_bounds = FocusWindowBounds(lower={0: 2, 1: 0}, upper={0: 2, 1: 0})

    image = render_archive_heatmap(archive, focus_bounds=focus_bounds)

    assert image is not None
    fig = fake.images[0]
    ax = fig.axes[0]
    assert len(ax.images) == 2
    assert ax.texts[-1].get_text() == "grey = outside focus window"


def test_render_archive_heatmap_time_hits_greys_hits_outside_focus_window(monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake)
    archive = GridArchive(
        dims=(3, 3, 2),
        ranges=[(0.0, 2.0), (0.0, 2.0), (0.0, 1.0)],
        descriptor_names=("goal_hit_count", "fall_count", "completion_time"),
        descriptor_labels=("Goal Hit Count", "Fall Count", "Completion Time"),
    )
    archive.add(
        np.array([[1.0], [2.0]], dtype=np.float32),
        np.array([4.0, 9.0], dtype=np.float64),
        np.array([[2.0, 0.0, 1.0], [2.0, 2.0, 1.0]], dtype=np.float64),
    )
    focus_bounds = FocusWindowBounds(lower={0: 2, 1: 0}, upper={0: 2, 1: 1})

    image = render_archive_heatmap_time_hits(
        archive,
        focus_bounds=focus_bounds,
    )

    assert image is not None
    fig = fake.images[0]
    overlay = fig.axes[0].images[1].get_array()
    assert bool(np.ma.getmaskarray(overlay)[2, 1])
    assert bool(np.ma.getmaskarray(overlay)[2, 0])
    assert not bool(np.ma.getmaskarray(overlay)[1, 0])
