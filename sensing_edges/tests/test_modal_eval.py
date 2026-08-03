"""Tests for Modal eval CLI wiring."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


class _FakeApp:
    def function(self, **_kwargs):
        def decorator(fn):
            return fn

        return decorator

    def local_entrypoint(self):
        def decorator(fn):
            return fn

        return decorator


def _load_modal_eval(monkeypatch):
    fake_modal = types.SimpleNamespace()
    fake_common = types.SimpleNamespace(
        VOLUMES={},
        app=_FakeApp(),
        commit_volumes=lambda: None,
        load_config=lambda **kwargs: kwargs,
        parse_override_csv=lambda override: [part.strip() for part in override.split(",") if part.strip()] if override else [],
        se_image=object(),
    )

    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setitem(sys.modules, "_modal_common", fake_common)

    path = Path(__file__).parent.parent / "scripts" / "modal_eval.py"
    spec = importlib.util.spec_from_file_location("sensing_edges_modal_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_eval_result_payload_uses_python_scalars(monkeypatch):
    module = _load_modal_eval(monkeypatch)

    results = types.SimpleNamespace(
        test_mean_fitness=np.float32(11.5),
        test_median_fitness=np.float64(10.25),
        test_ci95_low=np.float32(-2.0),
        test_ci95_high=np.float64(20.0),
        selected_elite_rank=np.int32(0),
        selected_elite_flat_index=np.int32(7),
    )

    payload = module._eval_result_payload(
        results=results,
        output=Path("/outputs/example_eval.json"),
    )

    assert payload == {
        "test_mean": 11.5,
        "test_median": 10.25,
        "ci95": [-2.0, 20.0],
        "selected_elite_rank": 0,
        "selected_elite_flat_index": 7,
        "output": "/outputs/example_eval.json",
    }
    assert type(payload["test_mean"]) is float
    assert type(payload["test_median"]) is float
    assert all(type(value) is float for value in payload["ci95"])
    assert type(payload["selected_elite_rank"]) is int
    assert type(payload["selected_elite_flat_index"]) is int
    assert type(payload["output"]) is str
