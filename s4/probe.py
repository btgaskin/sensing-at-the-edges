from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any


S4_ROOT = Path(__file__).resolve().parent
REPO_ROOT = S4_ROOT.parent
SENSING_EDGES_ROOT = REPO_ROOT / "sensing_edges"
PRESET_ROOT = S4_ROOT / "presets"

DEFAULT_SPECIMEN_ID = "bin_03_rank_00_cell_6-0-3_flat_0423"
DEFAULT_RUN_NAME = "paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed137_e4fcc2fb"
DEFAULT_SEED = 137
DEFAULT_RESULT_BUNDLE = S4_ROOT / "results" / DEFAULT_SPECIMEN_ID
DEFAULT_STORED_RUN_DIR = DEFAULT_RESULT_BUNDLE / "stored_run" / DEFAULT_RUN_NAME
DEFAULT_STORED_SPECIMEN_ROOT = DEFAULT_RESULT_BUNDLE / "stored_run" / "specimen_root"


@dataclass(frozen=True)
class ProbeJob:
    preset_name: str
    preset_path: Path
    description: str
    specimen_id: str
    bundle_root: Path
    run_dir: Path
    specimen_root: Path
    assays: tuple[str, ...]
    analysis_root_base: Path
    pointer_root: Path
    config_patch: dict[str, Any]
    force_laptop_safe: bool
    jax_platform: str
    seed: int

    @property
    def preset_root(self) -> Path:
        return self.analysis_root_base

    @property
    def expected_analysis_root(self) -> Path:
        return self.analysis_root_base / self.run_dir.name / self.specimen_id

    @property
    def expected_pointer_path(self) -> Path:
        return self.pointer_root / f"{self.specimen_id}.json"

    def to_summary(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "preset_path": str(self.preset_path),
                "bundle_root": str(self.bundle_root),
                "preset_root": str(self.preset_root),
                "run_dir": str(self.run_dir),
                "specimen_root": str(self.specimen_root),
                "analysis_root_base": str(self.analysis_root_base),
                "pointer_root": str(self.pointer_root),
                "expected_analysis_root": str(self.expected_analysis_root),
                "expected_pointer_path": str(self.expected_pointer_path),
            }
        )
        return payload


def _ensure_sensing_edges_import_path() -> None:
    path = str(SENSING_EDGES_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def _resolve_relative_path(raw: str | Path | None, *, anchor: Path, default: Path) -> Path:
    if raw is None:
        return default.resolve()
    path = Path(raw)
    if not path.is_absolute():
        path = (anchor / path).resolve()
    return path


def resolve_preset_path(preset: str, *, preset_root: Path = PRESET_ROOT) -> Path:
    candidate = Path(preset)
    options = []
    if candidate.is_absolute():
        options.append(candidate)
    else:
        options.append((Path.cwd() / candidate).resolve())
        options.append((preset_root / candidate).resolve())
        if candidate.suffix == "":
            options.append((preset_root / f"{candidate.name}.yaml").resolve())
            options.append((preset_root / f"{candidate.name}.yml").resolve())
    for option in options:
        if option.is_file():
            return option
        if option.is_dir():
            raise IsADirectoryError(
                f"{option} is a directory, not a preset file. "
                "Use --list-presets or pass a specific preset YAML name."
            )
    raise FileNotFoundError(f"Could not resolve preset {preset!r} under {preset_root}")


def _load_yaml_payload(path: Path) -> dict[str, Any]:
    _ensure_sensing_edges_import_path()
    from omegaconf import OmegaConf

    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Preset {path} must decode to a mapping")
    return dict(payload)


def load_probe_job(preset: str, *, preset_root: Path = PRESET_ROOT) -> ProbeJob:
    preset_path = resolve_preset_path(preset, preset_root=preset_root)
    payload = _load_yaml_payload(preset_path)
    anchor = preset_path.parent
    preset_name = str(payload.get("name") or preset_path.stem)
    bundle_root = _resolve_relative_path(
        payload.get("bundle_root"),
        anchor=anchor,
        default=DEFAULT_RESULT_BUNDLE,
    )
    run_dir = _resolve_relative_path(
        payload.get("run_dir"),
        anchor=anchor,
        default=DEFAULT_STORED_RUN_DIR,
    )
    specimen_root = _resolve_relative_path(
        payload.get("specimen_root"),
        anchor=anchor,
        default=DEFAULT_STORED_SPECIMEN_ROOT,
    )
    analysis_root_base = _resolve_relative_path(
        payload.get("analysis_root"),
        anchor=anchor,
        default=bundle_root / preset_name,
    )
    pointer_root = _resolve_relative_path(
        payload.get("pointer_root"),
        anchor=anchor,
        default=analysis_root_base / "_pointers",
    )
    assays = tuple(str(token) for token in payload.get("assays", []))
    if not assays:
        raise ValueError(f"Preset {preset_path} must define at least one assay")
    config_patch = payload.get("config_patch", {})
    if not isinstance(config_patch, dict):
        raise ValueError(f"Preset {preset_path} config_patch must be a mapping")
    return ProbeJob(
        preset_name=preset_name,
        preset_path=preset_path,
        description=str(payload.get("description", "")).strip(),
        specimen_id=str(payload.get("specimen_id") or DEFAULT_SPECIMEN_ID),
        bundle_root=bundle_root,
        run_dir=run_dir,
        specimen_root=specimen_root,
        assays=assays,
        analysis_root_base=analysis_root_base,
        pointer_root=pointer_root,
        config_patch=config_patch,
        force_laptop_safe=bool(payload.get("force_laptop_safe", True)),
        jax_platform=str(payload.get("jax_platform", "cpu")).strip().lower() or "cpu",
        seed=int(payload.get("seed", DEFAULT_SEED)),
    )


def available_presets(*, preset_root: Path = PRESET_ROOT) -> list[tuple[str, str]]:
    presets: list[tuple[str, str]] = []
    yaml_paths = sorted(set(preset_root.glob("*.yaml")) | set(preset_root.glob("*.yml")))
    for path in yaml_paths:
        try:
            payload = _load_yaml_payload(path)
            description = str(payload.get("description", "")).strip()
        except Exception:
            description = ""
        presets.append((path.stem, description))
    return presets


def _should_list_presets_from_args(presets: list[str]) -> bool:
    if len(presets) != 1:
        return False
    token = str(presets[0]).strip().lower()
    return token in {"presets", "preset", "list", "list-presets"}


def configure_local_runtime_env(jax_platform: str) -> None:
    normalized = str(jax_platform).strip().lower()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if normalized == "cpu":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif normalized == "metal":
        os.environ.pop("JAX_PLATFORM_NAME", None)
        os.environ["JAX_PLATFORMS"] = "metal,cpu"
    elif normalized in {"default", "auto"}:
        os.environ.pop("JAX_PLATFORM_NAME", None)
        os.environ.pop("JAX_PLATFORMS", None)
    else:
        raise ValueError(
            f"Unsupported jax_platform={jax_platform!r}; expected one of cpu, metal, default"
        )


def _local_l4_gpu_profile():
    _ensure_sensing_edges_import_path()
    from sensing_edges.runtime_config import compose_experiment_config

    l4_cfg = compose_experiment_config(
        overrides=["gpu_profile=l4"],
        config_dir=str(SENSING_EDGES_ROOT / "config"),
    )
    return l4_cfg.gpu_profile


def compose_probe_config(job: ProbeJob):
    _ensure_sensing_edges_import_path()
    from omegaconf import OmegaConf
    from sensing_edges.runtime_config import compose_stored_config, resolve_runtime_config

    config_path = job.run_dir / "resolved_config.yaml"
    cfg = compose_stored_config(
        config_path,
        prefer_current_post_run=True,
    )

    OmegaConf.set_struct(cfg, False)
    if job.force_laptop_safe:
        cfg.gpu_profile = _local_l4_gpu_profile()
        cfg.batching.generation.candidate_chunk_size = 8
        cfg.batching.generation.candidate_bucket_sizes = [1, 2, 4, 8]
        cfg.batching.generation.exact_fit_small_run_threshold = 8
        cfg.batching.validation.candidate_chunk_size = 4
        cfg.batching.validation.candidate_bucket_sizes = [1, 2, 4]
        cfg.batching.validation.exact_fit_small_run_threshold = 4
        cfg.batching.test.candidate_chunk_size = 2
        cfg.batching.test.candidate_bucket_sizes = [1, 2]
        cfg.batching.test.exact_fit_small_run_threshold = 2
        cfg.batching.trace.candidate_chunk_size = 1
        cfg.batching.trace.candidate_bucket_sizes = [1]
        cfg.batching.trace.exact_fit_small_run_threshold = 1
        cfg.render.parallel_workers = 1
        cfg.render.fps = 20
        cfg.render.width = 480
        cfg.render.height = 240
        cfg.render.trace_max_frames = 160
        cfg.render.post_train.enabled = True
        cfg.render.post_train.temporal_mode = "realtime"
        cfg.render.post_train.target_frames = 160
        cfg.render.post_train.upload_to_wandb = False
        cfg.render.post_train.include_archive_visuals = False
        cfg.wandb.enabled = False

    if job.config_patch:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(job.config_patch))

    cfg.seed = int(job.seed)
    cfg.eval.post_run.enabled = True
    cfg.eval.post_run.auto_mode = "off"
    cfg.eval.post_run.analysis_root = str(job.analysis_root_base)
    cfg.eval.post_run.output_dir = str(job.pointer_root)
    cfg.wandb.enabled = False

    runtime_resolution = resolve_runtime_config(
        cfg,
        config_dir=str(SENSING_EDGES_ROOT / "config"),
    )
    return cfg, runtime_resolution


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _copy_json_if_present(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.read_text())


class ProbeProgressBar:
    def __init__(self, *, index_path: Path, label: str) -> None:
        self.index_path = index_path
        self.label = label
        self.started = time.monotonic()
        self._stop = threading.Event()
        self._kick = threading.Event()
        self._thread: threading.Thread | None = None
        self._frames = ("-", "\\", "|", "/")
        self._frame_index = 0
        self._last_line = ""

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="s4-probe-progress", daemon=True)
        self._thread.start()

    def notify(self) -> None:
        self._kick.set()

    def finish(self) -> None:
        self._stop.set()
        self._kick.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._render(final=True)
        print()

    def _read_state(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {
                "status": "starting",
                "requested_assays": [],
                "completed_assays": [],
                "current_assay": None,
            }
        try:
            return json.loads(self.index_path.read_text())
        except Exception:
            return {
                "status": "starting",
                "requested_assays": [],
                "completed_assays": [],
                "current_assay": None,
            }

    def _line(self, *, final: bool = False) -> str:
        payload = self._read_state()
        requested = list(payload.get("requested_assays", []))
        completed = list(payload.get("completed_assays", []))
        current = str(payload.get("current_assay") or "idle")
        status = str(payload.get("status") or "running")
        total = max(len(requested), 1)
        done = min(len(completed), total)
        filled = int(round((done / total) * 18))
        bar = "#" * filled + "-" * (18 - filled)
        elapsed = time.monotonic() - self.started
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        frame = self._frames[self._frame_index % len(self._frames)]
        if final:
            if status == "completed":
                frame = "done"
            elif status == "failed":
                frame = "fail"
            else:
                frame = "stop"
        return (
            f"\r[{frame}] {self.label} [{bar}] {done}/{total} "
            f"current={current} status={status} elapsed={minutes:02d}:{seconds:02d}"
        )

    def _render(self, *, final: bool = False) -> None:
        self._frame_index += 1
        line = self._line(final=final)
        if final or line != self._last_line:
            print(line, end="", flush=True)
            self._last_line = line

    def _run(self) -> None:
        while not self._stop.is_set():
            self._render()
            self._kick.wait(timeout=0.5)
            self._kick.clear()


def run_probe_job(job: ProbeJob, *, dry_run: bool = False) -> dict[str, Any]:
    configure_local_runtime_env(job.jax_platform)
    cfg, runtime_resolution = compose_probe_config(job)
    launched_at = datetime.now(timezone.utc).isoformat()
    launch_payload = {
        "launched_at_utc": launched_at,
        "job": job.to_summary(),
        "assays": list(job.assays),
        "dry_run": bool(dry_run),
        "seed": int(job.seed),
    }
    _write_json(job.preset_root / "probe_job.json", launch_payload)
    if dry_run:
        dry_run_payload = {
            "status": "dry_run",
            "preset_name": job.preset_name,
            "preset_root": str(job.preset_root),
            "analysis_root": str(job.expected_analysis_root),
            "pointer_path": str(job.expected_pointer_path),
            "assays": list(job.assays),
        }
        _write_json(job.preset_root / "run_summary.json", dry_run_payload)
        return {
            **dry_run_payload,
        }

    _ensure_sensing_edges_import_path()
    from sensing_edges.logging_utils import configure_logging
    from sensing_edges.post_run import run_post_run_for_stored_specimen

    configure_logging("INFO")
    progress = ProbeProgressBar(
        index_path=job.expected_analysis_root / "index.json",
        label=job.preset_name,
    )
    progress.start()
    results = run_post_run_for_stored_specimen(
        cfg=cfg,
        runtime_resolution=runtime_resolution,
        run_dir=job.run_dir,
        specimen_root=job.specimen_root,
        specimen_id=job.specimen_id,
        assays=set(job.assays),
        raise_on_failure=True,
        on_progress=progress.notify,
    )
    progress.finish()
    result_payload = {
        "status": str(results.status),
        "preset_name": job.preset_name,
        "preset_root": str(job.preset_root),
        "specimen_id": str(results.specimen_id),
        "specimen_dir": str(results.specimen_dir),
        "index_path": str(results.index_path),
        "failed_assay": results.failed_assay,
        "error_message": results.error_message,
        "assay_dirs": {key: str(value) for key, value in results.assay_dirs.items()},
    }
    _write_json(job.preset_root / "probe_result.json", result_payload)
    _write_json(job.preset_root / "run_summary.json", result_payload)
    _copy_json_if_present(Path(results.index_path), job.preset_root / "latest_index.json")
    _copy_json_if_present(job.expected_analysis_root / "manifest.json", job.preset_root / "latest_manifest.json")
    return result_payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run preset-driven, specimen-local sensing_edges post-run probes.",
    )
    parser.add_argument(
        "presets",
        nargs="*",
        help="Preset file names or paths. Presets under s4/presets can omit .yaml.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available presets and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve jobs and output paths without running any workloads.",
    )
    parser.add_argument(
        "--jax-platform",
        choices=["cpu", "metal", "default"],
        help="Override the preset JAX platform for this invocation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_presets or _should_list_presets_from_args(list(args.presets)):
        for name, description in available_presets():
            suffix = f": {description}" if description else ""
            print(f"{name}{suffix}")
        return 0

    if not args.presets:
        parser.print_help()
        return 1

    if len(args.presets) > 5:
        parser.error("probe.py accepts at most 5 presets per invocation")

    exit_code = 0
    for preset in args.presets:
        job = load_probe_job(preset)
        if args.jax_platform:
            job = ProbeJob(
                preset_name=job.preset_name,
                preset_path=job.preset_path,
                description=job.description,
                specimen_id=job.specimen_id,
                bundle_root=job.bundle_root,
                run_dir=job.run_dir,
                specimen_root=job.specimen_root,
                assays=job.assays,
                analysis_root_base=job.analysis_root_base,
                pointer_root=job.pointer_root,
                config_patch=job.config_patch,
                force_laptop_safe=job.force_laptop_safe,
                jax_platform=str(args.jax_platform),
                seed=job.seed,
            )
        result = run_probe_job(job, dry_run=bool(args.dry_run))
        print(json.dumps(result, indent=2, sort_keys=True))
        if str(result.get("status")) not in {"completed", "dry_run"}:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
