"""Tests for Modal post-run CLI wiring."""

from __future__ import annotations

import importlib.util
import importlib
import sys
import types
from pathlib import Path

from omegaconf import OmegaConf
import pytest


class _FakeApp:
    def function(self, **_kwargs):
        def decorator(fn):
            return fn

        return decorator

    def local_entrypoint(self):
        def decorator(fn):
            return fn

        return decorator


def _load_modal_post_run(monkeypatch, *, load_config_impl=None):
    fake_modal = types.SimpleNamespace()
    fake_common = types.SimpleNamespace(
        VOLUMES={},
        app=_FakeApp(),
        commit_volumes=lambda: None,
        load_config=(load_config_impl or (lambda **_kwargs: OmegaConf.create({}))),
        parse_override_csv=lambda override: [part.strip() for part in override.split(",") if part.strip()] if override else [],
        se_image=object(),
    )

    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setitem(sys.modules, "_modal_common", fake_common)

    path = Path(__file__).parent.parent / "scripts" / "modal_post_run.py"
    spec = importlib.util.spec_from_file_location("sensing_edges_modal_post_run", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compose_post_run_cfg_accepts_extra_keys_from_stored_config(tmp_path, monkeypatch):
    base_cfg = OmegaConf.create(
        {
            "preset_id": "paper_run",
            "specimens": {"output_root": "specimens"},
            "eval": {
                "post_run": {
                    "native_max_steps": 4000,
                    "deformation_conditions": [
                        "intact",
                        "absolute_only",
                        "compliance_only",
                        "deformation_off",
                        "noise_only",
                        "shuffled",
                    ],
                    "trace": {
                        "enabled": True,
                        "sample_resolution": "controller_step",
                    },
                },
            },
        }
    )
    module = _load_modal_post_run(
        monkeypatch,
        load_config_impl=lambda **_kwargs: base_cfg,
    )
    resolved_config_path = tmp_path / "resolved_config.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "preset_id": "paper_run",
                "specimens": {"output_root": "specimens"},
                "eval": {
                    "post_run": {
                        "native_max_steps": 2000,
                        "deformation_conditions": [
                            "intact",
                            "absolute_only",
                            "compliance_only",
                            "deformation_off",
                        ],
                        "trace": {
                            "enabled": True,
                            "sample_resolution": "episode_step",
                            "scope": "native_intact_all_trials",
                        }
                    }
                },
            }
        ),
        resolved_config_path,
    )

    cfg = module._compose_post_run_cfg(
        preset="paper_run",
        resolved_config_path=str(resolved_config_path),
    )

    assert cfg.eval.post_run.trace.scope == "native_intact_all_trials"
    assert cfg.eval.post_run.trace.sample_resolution == "controller_step"
    assert cfg.eval.post_run.native_max_steps == 4000
    assert list(cfg.eval.post_run.deformation_conditions) == [
        "intact",
        "absolute_only",
        "compliance_only",
        "deformation_off",
        "noise_only",
        "shuffled",
    ]


def test_main_uses_spawn_when_detach_launch_enabled(monkeypatch, capsys):
    module = _load_modal_post_run(monkeypatch)
    calls: list[tuple[str, dict[str, object]]] = []

    class _FakeRemote:
        def spawn(self, **kwargs):
            calls.append(("spawn", kwargs))
            return types.SimpleNamespace(object_id="fc-test")

        def remote(self, **kwargs):
            calls.append(("remote", kwargs))
            return {"ok": True}

    monkeypatch.setattr(module, "post_run_remote", _FakeRemote())

    module.main(
        run_dir="/outputs/run",
        specimen_id="specimen_alpha",
        detach_launch=True,
    )

    captured = capsys.readouterr()
    assert calls == [
        (
            "spawn",
            {
                "run_dir": "/outputs/run",
                "specimen_id": "specimen_alpha",
                "specimen_rank": None,
                "preset": "config",
                "config_path": None,
                "overrides": None,
                "assays": None,
            },
        )
    ]
    assert "fc-test" in captured.out


def test_default_post_run_gpu_is_a100_40gb(monkeypatch):
    module = _load_modal_post_run(monkeypatch)

    assert module._DEFAULT_POST_RUN_GPU == "A100-40GB"
    assert module._LOW_COST_POST_RUN_GPU == "L4"


def test_main_routes_single_specimen_to_l4_gpu_profile(monkeypatch, capsys):
    module = _load_modal_post_run(monkeypatch)
    calls: list[tuple[str, dict[str, object]]] = []

    class _FakeRemote:
        def spawn(self, **kwargs):
            calls.append(("spawn", kwargs))
            return types.SimpleNamespace(object_id="fc-l4")

        def remote(self, **kwargs):
            calls.append(("remote", kwargs))
            return {"ok": True}

    monkeypatch.setattr(module, "post_run_remote_l4", _FakeRemote())

    module.main(
        run_dir="/outputs/run",
        specimen_id="specimen_alpha",
        assays="native_core",
        gpu_profile="l4",
        detach_launch=True,
    )

    captured = capsys.readouterr()
    assert calls == [
        (
            "spawn",
            {
                "run_dir": "/outputs/run",
                "specimen_id": "specimen_alpha",
                "specimen_rank": None,
                "preset": "config",
                "config_path": None,
                "overrides": None,
                "assays": ["native_core"],
            },
        )
    ]
    assert "fc-l4" in captured.out
    assert "(l4)" in captured.out


def test_main_uses_batched_remote_when_batch_jobs_path_provided(monkeypatch, tmp_path, capsys):
    module = _load_modal_post_run(monkeypatch)
    calls: list[tuple[str, dict[str, object]]] = []

    class _FakeBatchedRemote:
        def spawn(self, **kwargs):
            calls.append(("spawn", kwargs))
            return types.SimpleNamespace(object_id="fc-batch")

        def remote(self, **kwargs):
            calls.append(("remote", kwargs))
            return {"ok": True}

    monkeypatch.setattr(module, "post_run_batched_remote", _FakeBatchedRemote())
    batch_jobs_path = tmp_path / "batch_jobs.json"
    batch_jobs_path.write_text(
        '[{"run_dir":"/outputs/run_a","specimen_id":"specimen_alpha","config_path":"/outputs/run_a/config_snapshot.yaml"}]'
    )

    module.main(
        run_dir="/outputs/ignored",
        batch_jobs_path=str(batch_jobs_path),
        assays="signaling_ablation,deformation_ablation",
        batch_label="paper_deadline",
    )

    captured = capsys.readouterr()
    assert calls == [
        (
            "remote",
            {
                "run_specs": [
                    {
                        "run_dir": "/outputs/run_a",
                        "specimen_id": "specimen_alpha",
                        "config_path": "/outputs/run_a/config_snapshot.yaml",
                    }
                ],
                "preset": "config",
                "overrides": None,
                "assays": ["signaling_ablation", "deformation_ablation"],
                "batch_label": "paper_deadline",
            },
        )
    ]
    assert "Batched post-run result" in captured.out


def test_main_rejects_low_cost_gpu_for_batched_jobs(monkeypatch, tmp_path):
    module = _load_modal_post_run(monkeypatch)
    batch_jobs_path = tmp_path / "batch_jobs.json"
    batch_jobs_path.write_text(
        '[{"run_dir":"/outputs/run_a","specimen_id":"specimen_alpha","config_path":"/outputs/run_a/config_snapshot.yaml"}]'
    )

    with pytest.raises(ValueError, match="non-batched"):
        module.main(
            run_dir="/outputs/ignored",
            batch_jobs_path=str(batch_jobs_path),
            assays="signaling_ablation,deformation_ablation",
            gpu_profile="l4",
        )


def test_compose_batched_post_run_cfg_rejects_mixed_logical_configs(monkeypatch):
    module = _load_modal_post_run(monkeypatch)
    runtime_config = importlib.import_module("sensing_edges.runtime_config")

    monkeypatch.setattr(
        module,
        "_compose_post_run_cfg",
        lambda *, preset, resolved_config_path, overrides=None: {"config_path": resolved_config_path},
    )
    monkeypatch.setattr(
        runtime_config,
        "logical_config_sha256",
        lambda cfg: "sig-a" if str(cfg["config_path"]).endswith("a.yaml") else "sig-b",
    )

    with pytest.raises(ValueError, match="logically compatible configs"):
        module._compose_batched_post_run_cfg(
            run_specs=[
                {"run_dir": "/outputs/run_a", "specimen_id": "specimen_alpha", "config_path": "/tmp/a.yaml"},
                {"run_dir": "/outputs/run_b", "specimen_id": "specimen_beta", "config_path": "/tmp/b.yaml"},
            ],
            preset="config",
            overrides=None,
        )
