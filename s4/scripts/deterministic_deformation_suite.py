from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from free_space_branch_common import DEFAULT_BUNDLE_ROOT, ensure_import_path, load_cfg, write_json


SCRIPT_DIR = Path(__file__).resolve().parent
S4_ROOT = SCRIPT_DIR.parent
RESULT_ROOT = DEFAULT_BUNDLE_ROOT
DEFAULT_OUTPUT_DIR = RESULT_ROOT / "analysis_runs" / "deterministic_deformation_suite_cpu_local"
DEFAULT_CONFIG_PATH = RESULT_ROOT / "config_snapshot.yaml"
DEFAULT_GENOTYPE_PATH = RESULT_ROOT / "genotype.npz"
DEFAULT_MANIFEST_PATH = RESULT_ROOT / "manifest.json"
DEFAULT_CONDITIONS = ("intact", "absolute_only", "compliance_only", "deformation_off")
DEFAULT_SURFACES = ("native_core", "long_distance_generalization", "double_cliff")
DEFAULT_NATIVE_MAX_STEPS = 6000
DEFAULT_EXTENDED_MAX_STEPS = 8000
DEFAULT_DOUBLE_CLIFF_MAX_STEPS = 10000


@dataclass(frozen=True)
class EpisodeSpec:
    surface_id: str
    condition_id: str
    max_steps: int
    goal_distance: int | None = None
    cliff_distance: int | None = None
    side: str | None = None
    width: int | None = None

    @property
    def render_filename(self) -> str:
        if self.surface_id == "double_cliff":
            return f"width_{int(self.width)}.gif"
        return (
            f"pair_{int(self.goal_distance)}_{int(self.cliff_distance)}"
            f"_{str(self.side).strip().lower()}.gif"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the deterministic deformation-ablation suite for the canonical S4 specimen "
            "across native core, long-distance generalization, and double-cliff surfaces."
        ),
    )
    parser.add_argument("--config-snapshot-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--genotype-path", type=Path, default=DEFAULT_GENOTYPE_PATH)
    parser.add_argument("--source-manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--native-max-steps", type=int, default=DEFAULT_NATIVE_MAX_STEPS)
    parser.add_argument("--extended-max-steps", type=int, default=DEFAULT_EXTENDED_MAX_STEPS)
    parser.add_argument("--double-cliff-max-steps", type=int, default=DEFAULT_DOUBLE_CLIFF_MAX_STEPS)
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Print the planned deterministic run counts and exit without executing the suite.",
    )
    return parser.parse_args()


def _load_source_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_genotype(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "genotype" in data:
            return np.asarray(data["genotype"], dtype=np.float32)
        if len(data.files) == 1:
            return np.asarray(data[data.files[0]], dtype=np.float32)
    raise KeyError(f"Could not resolve genotype array from {path}")


def _configured_cfg(
    *,
    config_snapshot_path: Path,
    native_max_steps: int,
    extended_max_steps: int,
    double_cliff_max_steps: int,
) -> Any:
    cfg = load_cfg(
        config_snapshot_path,
        max_steps=max(int(native_max_steps), int(extended_max_steps), int(double_cliff_max_steps)),
    )
    OmegaConf.set_struct(cfg, False)
    post_run = cfg.eval.post_run
    post_run.native_max_steps = int(native_max_steps)
    post_run.max_steps = int(extended_max_steps)
    post_run.double_cliff_max_steps = int(double_cliff_max_steps)
    post_run.deformation_conditions = list(DEFAULT_CONDITIONS)
    post_run.deformation_surfaces = list(DEFAULT_SURFACES)
    post_run.deformation_render_conditions = list(DEFAULT_CONDITIONS)
    post_run.signaling_conditions = ["intact"]
    post_run.signaling_surfaces = ["native_core"]
    post_run.signaling_render_conditions = ["intact"]
    post_run.trials_per_side = 1
    post_run.stochastic_trials_per_side = 1
    post_run.double_cliff.trials_per_width = 1
    post_run.double_cliff.stochastic_trials_per_width = 1
    post_run.trace.requested_trials_per_side = 1
    post_run.trace.stochastic_requested_trials_per_side = 1
    post_run.active_deformation_condition = "intact"
    post_run.active_signaling_condition = "intact"
    post_run.long_distance_pair_max_steps = [
        {
            "goal_distance": int(pair["goal_distance"]),
            "cliff_distance": int(pair["cliff_distance"]),
            "max_steps": int(extended_max_steps),
        }
        for pair in list(post_run.long_distance_pairs)
    ]
    return cfg


def _expected_episode_specs(cfg) -> list[EpisodeSpec]:
    specs: list[EpisodeSpec] = []
    post_run = cfg.eval.post_run
    native_max_steps = int(post_run.native_max_steps)
    extended_pair_step_map = {
        (int(pair["goal_distance"]), int(pair["cliff_distance"])): int(pair["max_steps"])
        for pair in list(post_run.long_distance_pair_max_steps)
    }
    double_cliff_max_steps = int(post_run.double_cliff_max_steps)

    for condition in DEFAULT_CONDITIONS:
        for pair in list(post_run.native_pairs):
            for side in ("left", "right"):
                specs.append(
                    EpisodeSpec(
                        surface_id="native_core",
                        condition_id=str(condition),
                        max_steps=native_max_steps,
                        goal_distance=int(pair["goal_distance"]),
                        cliff_distance=int(pair["cliff_distance"]),
                        side=side,
                    )
                )
        for pair in list(post_run.long_distance_pairs):
            key = (int(pair["goal_distance"]), int(pair["cliff_distance"]))
            for side in ("left", "right"):
                specs.append(
                    EpisodeSpec(
                        surface_id="long_distance_generalization",
                        condition_id=str(condition),
                        max_steps=int(extended_pair_step_map[key]),
                        goal_distance=key[0],
                        cliff_distance=key[1],
                        side=side,
                    )
                )
        for width in list(post_run.double_cliff.widths):
            specs.append(
                EpisodeSpec(
                    surface_id="double_cliff",
                    condition_id=str(condition),
                    max_steps=double_cliff_max_steps,
                    width=int(width),
                )
            )
    return specs


def _expected_counts(specs: list[EpisodeSpec]) -> dict[str, Any]:
    surface_counts: dict[str, int] = {}
    surface_condition_counts: dict[str, int] = {}
    env_steps = 0
    for spec in specs:
        surface_counts[spec.surface_id] = surface_counts.get(spec.surface_id, 0) + 1
        key = f"{spec.surface_id}:{spec.condition_id}"
        surface_condition_counts[key] = surface_condition_counts.get(key, 0) + 1
        env_steps += int(spec.max_steps)
    return {
        "episode_count": int(len(specs)),
        "gif_count": int(len(specs)),
        "env_steps": int(env_steps),
        "phase_count": int(env_steps * 3),
        "nca_micro_updates": int(env_steps * 3 * 5),
        "physics_substeps": int(env_steps * 15),
        "counts_by_surface": surface_counts,
        "counts_by_surface_condition": surface_condition_counts,
    }


def _render_path_for_row(output_dir: Path, row: pd.Series) -> Path:
    condition = str(row["deformation_condition"])
    surface_id = str(row["surface_id"])
    render_dir = output_dir / "deformation_ablation" / "surfaces" / surface_id / "conditions" / condition / "renders"
    if surface_id == "double_cliff":
        return render_dir / f"width_{int(row['width'])}.gif"
    return render_dir / (
        f"pair_{int(row['goal_distance'])}_{int(row['cliff_distance'])}_{str(row['side']).strip().lower()}.gif"
    )


def _episode_manifest(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    manifest_df = df.copy()
    render_paths = [_render_path_for_row(output_dir, row) for _, row in manifest_df.iterrows()]
    manifest_df["render_path"] = [str(path) for path in render_paths]
    manifest_df["render_exists"] = [path.exists() for path in render_paths]
    return manifest_df


def _top_level_manifest(
    *,
    cfg,
    source_manifest: dict[str, Any],
    source_manifest_path: Path,
    config_snapshot_path: Path,
    genotype_path: Path,
    output_dir: Path,
    assay_dir: Path,
    suite_df: pd.DataFrame,
    episode_manifest_df: pd.DataFrame,
    runtime_resolution,
    expected_summary: dict[str, Any],
) -> dict[str, Any]:
    counts_by_surface = {
        str(surface): int(count)
        for surface, count in suite_df.groupby("surface_id", sort=False).size().items()
    }
    counts_by_surface_condition = {
        f"{str(surface)}:{str(condition)}": int(count)
        for (surface, condition), count in suite_df.groupby(
            ["surface_id", "deformation_condition"],
            sort=False,
        ).size().items()
    }
    actual_gif_count = int(sum(bool(value) for value in episode_manifest_df["render_exists"].tolist()))
    post_run = cfg.eval.post_run
    return {
        "schema_version": 1,
        "kind": "s4_deterministic_deformation_suite",
        "suite_label": "deterministic_deformation_suite_cpu_local",
        "source_specimen_id": str(source_manifest.get("specimen_id")),
        "source_run_id": str(source_manifest.get("run_id")),
        "source_manifest_path": str(source_manifest_path),
        "config_snapshot_path": str(config_snapshot_path),
        "genotype_path": str(genotype_path),
        "output_dir": str(output_dir),
        "assay_dir": str(assay_dir),
        "conditions": list(DEFAULT_CONDITIONS),
        "surfaces": list(DEFAULT_SURFACES),
        "native_pairs": [dict(pair) for pair in list(post_run.native_pairs)],
        "long_distance_pairs": [dict(pair) for pair in list(post_run.long_distance_pairs)],
        "double_cliff_widths": [int(width) for width in list(post_run.double_cliff.widths)],
        "max_steps": {
            "native_core": int(post_run.native_max_steps),
            "long_distance_generalization": {
                f"({int(pair['goal_distance'])},{int(pair['cliff_distance'])})": int(pair["max_steps"])
                for pair in list(post_run.long_distance_pair_max_steps)
            },
            "double_cliff": int(post_run.double_cliff_max_steps),
        },
        "expected": dict(expected_summary),
        "actual": {
            "episode_count": int(len(suite_df)),
            "gif_count": actual_gif_count,
            "counts_by_surface": counts_by_surface,
            "counts_by_surface_condition": counts_by_surface_condition,
        },
        "deterministic_trials": {
            "trials_per_side": int(post_run.trials_per_side),
            "trials_per_width": int(post_run.double_cliff.trials_per_width),
            "trace_requested_trials_per_side": int(post_run.trace.requested_trials_per_side),
        },
        "runtime": {
            "backend": str(jax.default_backend()),
            "devices": [str(device) for device in jax.devices()],
            "gpu_profile": str(runtime_resolution.gpu_profile),
            "batching": runtime_resolution.batching_manifest(),
            "collision": runtime_resolution.collision_manifest(),
            "jax_version": str(jax.__version__),
            "numpy_version": str(np.__version__),
        },
        "archived_reference_warning": (
            "This suite is a new current-code CPU-local baseline and is not expected to reproduce the archived GPU post-run artifacts."
        ),
        "episode_manifest_csv": str(output_dir / "episode_manifest.csv"),
        "suite_summary_csv": str(output_dir / "suite_summary.csv"),
        "deformation_summary_json": str(assay_dir / "aggregate.json"),
    }


def _print_count_summary(specs: list[EpisodeSpec], summary: dict[str, Any]) -> None:
    payload = {
        "suite": "deterministic_deformation_suite_cpu_local",
        "conditions": list(DEFAULT_CONDITIONS),
        "surfaces": list(DEFAULT_SURFACES),
        "expected": summary,
        "episodes": [asdict(spec) for spec in specs],
    }
    print(json.dumps(payload, indent=2))


def main(args: argparse.Namespace) -> int:
    cfg = _configured_cfg(
        config_snapshot_path=args.config_snapshot_path.resolve(),
        native_max_steps=int(args.native_max_steps),
        extended_max_steps=int(args.extended_max_steps),
        double_cliff_max_steps=int(args.double_cliff_max_steps),
    )
    expected_specs = _expected_episode_specs(cfg)
    expected_summary = _expected_counts(expected_specs)

    if args.count_only:
        _print_count_summary(expected_specs, expected_summary)
        return 0

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}. "
            "Choose a new --output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_import_path()
    from sensing_edges.post_run import _run_deformation_ablation_assay
    from sensing_edges.runtime_config import resolve_runtime_config

    source_manifest = _load_source_manifest(args.source_manifest_path.resolve())
    genotype = _load_genotype(args.genotype_path.resolve())
    runtime_resolution = resolve_runtime_config(
        cfg,
        preset=(str(cfg.get("preset_id")) if cfg.get("preset_id") is not None else None),
    )

    suite_df, aggregate, assay_dir = _run_deformation_ablation_assay(
        cfg=cfg,
        runtime_resolution=runtime_resolution,
        specimen_dir=output_dir,
        specimen_id=str(source_manifest.get("specimen_id", RESULT_ROOT.name)),
        genotype=genotype,
        selection_row=dict(source_manifest.get("selection", {})),
        run_id=str(source_manifest.get("run_id", "unknown")),
    )
    if len(suite_df) != int(expected_summary["episode_count"]):
        raise RuntimeError(
            f"Unexpected episode count: got {len(suite_df)}, expected {expected_summary['episode_count']}"
        )

    episode_manifest_df = _episode_manifest(suite_df, output_dir)
    actual_gif_count = int(sum(bool(value) for value in episode_manifest_df["render_exists"].tolist()))
    if actual_gif_count != int(expected_summary["gif_count"]):
        raise RuntimeError(
            f"Unexpected GIF count: got {actual_gif_count}, expected {expected_summary['gif_count']}"
        )

    episode_manifest_df.to_csv(output_dir / "episode_manifest.csv", index=False)
    suite_df.to_csv(output_dir / "suite_summary.csv", index=False)
    write_json(
        output_dir / "run_manifest.json",
        _top_level_manifest(
            cfg=cfg,
            source_manifest=source_manifest,
            source_manifest_path=args.source_manifest_path.resolve(),
            config_snapshot_path=args.config_snapshot_path.resolve(),
            genotype_path=args.genotype_path.resolve(),
            output_dir=output_dir,
            assay_dir=assay_dir,
            suite_df=suite_df,
            episode_manifest_df=episode_manifest_df,
            runtime_resolution=runtime_resolution,
            expected_summary=expected_summary,
        ),
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main(args))
