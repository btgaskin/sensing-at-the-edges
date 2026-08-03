from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_probe_module():
    path = Path(__file__).resolve().parents[1] / "probe.py"
    spec = importlib.util.spec_from_file_location("s4_probe", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_builtin_trace_preset():
    module = _load_probe_module()
    job = module.load_probe_job("pair42_trace_compare")
    assert job.preset_name == "pair42_trace_compare"
    assert job.specimen_id == "bin_03_rank_00_cell_6-0-3_flat_0423"
    assert job.assays == ("pca_bundle",)
    assert job.seed == 137
    assert job.run_dir.name == module.DEFAULT_RUN_NAME
    assert job.analysis_root_base.name == "pair42_trace_compare"
    assert job.expected_analysis_root.parts[-2:] == (
        module.DEFAULT_RUN_NAME,
        "bin_03_rank_00_cell_6-0-3_flat_0423",
    )


def test_available_presets_contains_expected_names():
    module = _load_probe_module()
    names = {name for name, _description in module.available_presets()}
    assert "cliffblind" in names
    assert "native1n" in names
    assert "long5m" in names
    assert "pair42_trace_compare" in names
    assert "pair42_native_deformation_compare" in names
    assert "pair63_trace_compare" in names


def test_presets_token_lists_presets():
    module = _load_probe_module()
    assert module.main(["presets"]) == 0


def test_directory_like_preset_raises_clear_error():
    module = _load_probe_module()
    try:
        module.resolve_preset_path("presets/pair42_trace_compare")
    except IsADirectoryError:
        raise AssertionError("Expected a preset file path to resolve")
    try:
        module.resolve_preset_path(str(module.PRESET_ROOT))
    except IsADirectoryError as exc:
        assert "directory" in str(exc).lower()
    else:
        raise AssertionError("Expected IsADirectoryError for preset directory input")
