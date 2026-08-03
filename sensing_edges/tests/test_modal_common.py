"""Tests for shared Modal image setup."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


class _FakeVolume:
    def __init__(self, name: str, create_if_missing: bool):
        self.name = name
        self.create_if_missing = create_if_missing

    @classmethod
    def from_name(cls, name: str, create_if_missing: bool = False):
        return cls(name=name, create_if_missing=create_if_missing)

    def commit(self) -> None:
        return None


class _FakeApp:
    def __init__(self, name: str):
        self.name = name


class _FakeImage:
    def __init__(self):
        self.local_dirs: list[tuple[str, str]] = []
        self.local_files: list[tuple[str, str]] = []
        self.env_vars: dict[str, str] = {}

    @classmethod
    def debian_slim(cls, python_version: str):
        image = cls()
        image.python_version = python_version
        return image

    def apt_install(self, *packages: str):
        self.apt_packages = packages
        return self

    def pip_install(self, *packages: str):
        self.pip_packages = packages
        return self

    def env(self, env_vars: dict[str, str]):
        self.env_vars.update(env_vars)
        return self

    def add_local_dir(self, local_dir: str, remote_dir: str):
        self.local_dirs.append((local_dir, remote_dir))
        return self

    def add_local_file(self, local_file: str, remote_path: str):
        self.local_files.append((local_file, remote_path))
        return self


def _load_modal_common(monkeypatch):
    fake_modal = types.SimpleNamespace(App=_FakeApp, Image=_FakeImage, Volume=_FakeVolume)
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    path = Path(__file__).parent.parent / "scripts" / "_modal_common.py"
    spec = importlib.util.spec_from_file_location("sensing_edges_modal_common", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_se_image_mounts_modal_helper(monkeypatch):
    module = _load_modal_common(monkeypatch)

    helper_path = str(module.SE_ROOT / "scripts" / "_modal_common.py")
    assert (helper_path, "/root/_modal_common.py") in module.se_image.local_files


def test_resolve_roots_supports_container_mount_layout(monkeypatch, tmp_path):
    module = _load_modal_common(monkeypatch)

    container_experiment_root = tmp_path / "root" / "experiment"
    (container_experiment_root / "config").mkdir(parents=True)
    (container_experiment_root / "sensing_edges").mkdir()
    module_path = tmp_path / "root" / "_modal_common.py"

    repo_root, se_root = module._resolve_roots(
        module_path,
        container_experiment_root=container_experiment_root,
        container_repo_root=tmp_path / "root",
    )

    assert repo_root == tmp_path / "root"
    assert se_root == container_experiment_root


def test_parse_override_csv_preserves_list_literals(monkeypatch):
    module = _load_modal_common(monkeypatch)

    overrides = module.parse_override_csv(
        "train.n_generations=10,batching.generation.candidate_bucket_sizes=[16, 32, 64],wandb.enabled=false"
    )

    assert overrides == [
        "train.n_generations=10",
        "batching.generation.candidate_bucket_sizes=[16, 32, 64]",
        "wandb.enabled=false",
    ]


def test_compose_config_uses_resolved_se_root_config_dir(monkeypatch):
    module = _load_modal_common(monkeypatch)
    captured: dict[str, object] = {}

    def _compose_experiment_config(*, preset, overrides, config_dir):
        captured["preset"] = preset
        captured["overrides"] = overrides
        captured["config_dir"] = config_dir
        return {"ok": True}

    monkeypatch.setitem(
        sys.modules,
        "sensing_edges.runtime_config",
        types.SimpleNamespace(compose_experiment_config=_compose_experiment_config),
    )

    result = module.compose_config(preset="iter_run", overrides=["seed=42"])

    assert result == {"ok": True}
    assert captured == {
        "preset": "iter_run",
        "overrides": ["seed=42"],
        "config_dir": str(module.SE_ROOT / "config"),
    }
