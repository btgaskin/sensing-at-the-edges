#!/usr/bin/env python
"""Paper protocol orchestrator.

Subcommands for each stage of the experimental pipeline:
  preflight  — validate configs, terrain suites, git state
  pilot      — run smoke test + short pilot
  core       — launch the paper ablation matrix for the requested seeds
  eval       — held-out evaluation on all final checkpoints
  figures    — generate paper figures + stats tables
  bundle     — package release artifact

Usage:
    uv run python scripts/paper_protocol.py preflight
    uv run python scripts/paper_protocol.py pilot
    uv run python scripts/paper_protocol.py core --seeds 42,137 --schedules-dir terrain_suites
    uv run python scripts/paper_protocol.py core --bodies rect_7x3,comb_7x3,grown:7x3
    uv run python scripts/paper_protocol.py eval --outputs-dir local_outputs --schedules-dir terrain_suites
    uv run python scripts/paper_protocol.py figures
    uv run python scripts/paper_protocol.py bundle
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SE_ROOT = Path(__file__).parent.parent
DEFAULT_SCHEDULES_DIR = "terrain_suites"
DEFAULT_PUBLICATION_SET = "paper_main"
DEFAULT_BODY_SPECS = ("rect_7x2",)
REFERENCE_SEEDS = [42, 99, 7, 13, 21]
SEED_SUFFIX_RE = re.compile(r"_seed(?P<seed>\d+)$")
SCHEDULE_SEED_RE = re.compile(r"^seed_(?P<seed>\d+)(?:_|$)")


def _run(cmd: list[str], check: bool = True) -> int:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(SE_ROOT))
    if check and result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        sys.exit(result.returncode)
    return result.returncode


def _seed_from_run_dir_name(name: str) -> int:
    match = SEED_SUFFIX_RE.search(name)
    if not match:
        raise ValueError(f"Could not infer seed from run directory name: {name}")
    return int(match.group("seed"))


def _discover_available_schedule_seeds(schedules_dir: str) -> list[int]:
    """Infer ready-to-run seeds from schedule directory names on disk."""
    root = Path(schedules_dir)
    if not root.exists():
        return []

    seeds: set[int] = set()
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = SCHEDULE_SEED_RE.match(path.name)
        if match:
            seeds.add(int(match.group("seed")))
    return sorted(seeds)


def _schedule_for_run(run_dir: Path, *, schedule: str | None, schedules_dir: str) -> str:
    if schedule:
        return schedule
    seed = _seed_from_run_dir_name(run_dir.name)
    schedule_path = Path(schedules_dir) / f"seed_{seed}"
    if not schedule_path.exists():
        raise FileNotFoundError(f"Schedule not found for seed {seed}: {schedule_path}")
    return str(schedule_path)


def _preset_from_run_dir_name(name: str) -> str:
    return SEED_SUFFIX_RE.sub("", name)


def _remote_run_dir(run_dir_name: str) -> Path:
    return Path("/outputs") / run_dir_name


def _variant_schedule_path(seed: int, variant: str, schedules_dir: str) -> Path:
    from sensing_edges.terrain_schedule import schedule_bundle_path

    return schedule_bundle_path(
        schedules_dir,
        seed=seed,
        variant=variant,
    )


def _parse_body_spec(spec: str) -> dict[str, object]:
    token = str(spec).strip().lower()
    if token == "rect_7x2":
        return {
            "label": "rect_7x2",
            "body": "rect_7x2",
            "canvas_w": None,
            "canvas_h": None,
            "overrides": [
                "body.mode=fixed",
                "body.shape=rectangle",
                "terrain.robot_w=7",
                "terrain.robot_h=2",
            ],
        }
    if token == "rect_7x3":
        return {
            "label": "rect_7x3",
            "body": "rect_7x3",
            "canvas_w": None,
            "canvas_h": None,
            "overrides": [
                "body.mode=fixed",
                "body.shape=rectangle",
                "terrain.robot_w=7",
                "terrain.robot_h=3",
            ],
        }
    if token == "comb":
        return {
            "label": "comb",
            "body": "comb",
            "canvas_w": None,
            "canvas_h": None,
            "overrides": [
                "body.mode=fixed",
                "body.shape=comb",
                "terrain.robot_w=7",
                "terrain.robot_h=2",
            ],
        }
    if token == "comb_7x3":
        return {
            "label": "comb_7x3",
            "body": "comb_7x3",
            "canvas_w": None,
            "canvas_h": None,
            "overrides": [
                "body.mode=fixed",
                "body.shape=comb_7x3",
                "terrain.robot_w=7",
                "terrain.robot_h=3",
            ],
        }
    if token == "grown":
        token = "grown:7x3"
    if token.startswith("grown:"):
        canvas = token.split(":", 1)[1]
        match = re.fullmatch(r"(?P<w>\d+)x(?P<h>\d+)", canvas)
        if match is None:
            raise ValueError(f"Invalid grown body spec: {spec!r}")
        canvas_w = int(match.group("w"))
        canvas_h = int(match.group("h"))
        return {
            "label": f"grown_{canvas_w}x{canvas_h}",
            "body": "grown",
            "canvas_w": canvas_w,
            "canvas_h": canvas_h,
            "overrides": [
                "body.mode=grown",
                f"body.growth.max_robot_w={canvas_w}",
                f"body.growth.max_robot_h={canvas_h}",
            ],
        }
    raise ValueError(f"Unsupported body spec: {spec!r}")


def _body_specs_from_cli(body_csv: str | None) -> list[dict[str, object]]:
    raw_specs = (
        [token.strip() for token in body_csv.split(",") if token.strip()]
        if body_csv
        else list(DEFAULT_BODY_SPECS)
    )
    return [_parse_body_spec(spec) for spec in raw_specs]


def _body_specs_from_publication_set(body_csv: str | None, publication_set: dict[str, object]) -> list[dict[str, object]]:
    if body_csv:
        return _body_specs_from_cli(body_csv)
    raw_specs = [str(item) for item in publication_set.get("body_specs", [])]
    if not raw_specs:
        raw_specs = list(DEFAULT_BODY_SPECS)
    return [_parse_body_spec(spec) for spec in raw_specs]


def _schedule_bundle_for_preset(
    preset: str,
    *,
    seed: int,
    schedules_dir: str,
    overrides: list[str] | None = None,
) -> dict[str, str]:
    from sensing_edges.runtime_config import compose_experiment_config

    cfg = compose_experiment_config(
        preset=preset,
        overrides=overrides,
        config_dir=SE_ROOT / "config",
    )
    train_variant = str(cfg.terrain.variant)
    rerank_variant = str(cfg.eval.get("rerank_variant") or train_variant)
    eval_variant = str(cfg.eval.get("test_variant") or rerank_variant)
    from sensing_edges.terrain_schedule import resolve_schedule_bundle_path

    return {
        "train": str(
            resolve_schedule_bundle_path(
                schedules_dir,
                seed=seed,
                variant=train_variant,
                n_gens=int(cfg.train.n_generations),
                n_episodes=int(cfg.train.n_episodes),
                n_val=int(cfg.eval.rerank_episodes),
                n_test=int(cfg.eval.test_episodes),
            )
        ),
        "rerank": str(
            resolve_schedule_bundle_path(
                schedules_dir,
                seed=seed,
                variant=rerank_variant,
                n_gens=int(cfg.train.n_generations),
                n_episodes=int(cfg.train.n_episodes),
                n_val=int(cfg.eval.rerank_episodes),
                n_test=int(cfg.eval.test_episodes),
            )
        ),
        "eval": str(
            resolve_schedule_bundle_path(
                schedules_dir,
                seed=seed,
                variant=eval_variant,
                n_gens=int(cfg.train.n_generations),
                n_episodes=int(cfg.train.n_episodes),
                n_val=int(cfg.eval.rerank_episodes),
                n_test=int(cfg.eval.test_episodes),
            )
        ),
    }


def _schedule_bundle_for_resolved_config(config_path: str | Path, *, schedules_dir: str) -> dict[str, str]:
    from omegaconf import OmegaConf
    from sensing_edges.terrain_schedule import resolve_schedule_bundle_path

    cfg = OmegaConf.load(config_path)
    seed = int(cfg.get("seed", 0))
    train_variant = str(cfg.terrain.variant)
    rerank_variant = str(cfg.eval.get("rerank_variant") or train_variant)
    eval_variant = str(cfg.eval.get("test_variant") or rerank_variant)
    return {
        "train": str(
            resolve_schedule_bundle_path(
                schedules_dir,
                seed=seed,
                variant=train_variant,
                n_gens=int(cfg.train.n_generations),
                n_episodes=int(cfg.train.n_episodes),
                n_val=int(cfg.eval.rerank_episodes),
                n_test=int(cfg.eval.test_episodes),
            )
        ),
        "rerank": str(
            resolve_schedule_bundle_path(
                schedules_dir,
                seed=seed,
                variant=rerank_variant,
                n_gens=int(cfg.train.n_generations),
                n_episodes=int(cfg.train.n_episodes),
                n_val=int(cfg.eval.rerank_episodes),
                n_test=int(cfg.eval.test_episodes),
            )
        ),
        "eval": str(
            resolve_schedule_bundle_path(
                schedules_dir,
                seed=seed,
                variant=eval_variant,
                n_gens=int(cfg.train.n_generations),
                n_episodes=int(cfg.train.n_episodes),
                n_val=int(cfg.eval.rerank_episodes),
                n_test=int(cfg.eval.test_episodes),
            )
        ),
    }


def _config_path_for_run(run_dir: Path) -> str:
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Resolved config not found: {config_path}")
    return str(config_path)


def _paused_stage_runs(outputs_dir: Path) -> list[Path]:
    runs: list[Path] = []
    for run_dir in sorted(outputs_dir.iterdir()):
        manifest_path = run_dir / "manifest.json"
        if not run_dir.is_dir() or not manifest_path.exists():
            continue
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        if str(manifest.get("status", "")).strip().lower() != "paused":
            continue
        if str(manifest.get("lineage", {}).get("mode", "root")).strip().lower() == "branch":
            continue
        runs.append(run_dir)
    return runs


def _validate_run_matrix() -> None:
    from sensing_edges.runtime_config import compose_experiment_config, resolve_runtime_config

    default_bodies = _body_specs_from_cli(None)
    exploratory_bodies = [_parse_body_spec("grown:7x3")]
    run_cases = [
        ("iter_run", default_bodies[0]),
        ("paper_run", None),
        *[("paper_run", spec) for spec in default_bodies],
        *[("paper_run", spec) for spec in exploratory_bodies],
    ]
    for preset, body_spec in run_cases:
        overrides = [] if body_spec is None else list(body_spec["overrides"])
        cfg = compose_experiment_config(
            preset=preset,
            overrides=overrides,
            config_dir=SE_ROOT / "config",
        )
        resolve_runtime_config(
            cfg,
            preset=preset,
            runtime_overrides=overrides,
            config_dir=SE_ROOT / "config",
        )
        label = preset if body_spec is None else f"{preset}:{body_spec['label']}"
        print(f"  Run OK: {label}")


def cmd_preflight(args):
    """Validate configs, schemas, and git state."""
    print("=== Preflight ===")

    # Check git clean.
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    )
    if result.stdout.strip():
        print("WARNING: Git tree is dirty. Paper runs should use a clean tree.")
        if not args.allow_dirty:
            print("  Pass --allow-dirty to override.")
            sys.exit(1)

    # Run config tests.
    _run(["uv", "run", "--with", "pytest", "python", "-m", "pytest",
          "tests/test_configs.py", "-v", "--tb=short"])
    _validate_run_matrix()

    # Validate schemas parse.
    import json
    for schema_file in (SE_ROOT / "schemas").glob("*.json"):
        with open(schema_file) as f:
            json.load(f)  # Will raise on malformed JSON.
        print(f"  Schema OK: {schema_file.name}")

    print("Preflight PASSED")


def cmd_pilot(args):
    """Run smoke test via Modal."""
    print("=== Pilot ===")
    _run(["modal", "run", "scripts/modal_smoke.py"])
    print("Pilot PASSED")


def cmd_core(args):
    """Launch full ablation matrix via Modal."""
    from sensing_edges.publication import load_publication_set, resolve_release_id

    publication_set = load_publication_set(args.publication_set, se_root=SE_ROOT)
    release_id = resolve_release_id(release_id=args.release_id, publication_set=publication_set)
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = _discover_available_schedule_seeds(args.schedules_dir)
        if seeds:
            print(f"Auto-discovered schedule seeds in {args.schedules_dir}: {','.join(str(seed) for seed in seeds)}")
        else:
            seeds = list(REFERENCE_SEEDS)
            print(
                "No on-disk schedule seeds discovered; "
                f"falling back to reference seeds: {','.join(str(seed) for seed in seeds)}"
            )
    body_specs = _body_specs_from_publication_set(args.bodies, publication_set)
    deformation_conditions = [str(item) for item in publication_set.get("deformation_conditions", ["on"])]
    total_runs = len(body_specs) * len(seeds) * len(deformation_conditions)
    print(
        f"=== Core: {release_id} × {len(body_specs)} body variants × "
        f"{len(deformation_conditions)} deformation conditions × {len(seeds)} seeds = {total_runs} runs ==="
    )

    for body_spec in body_specs:
        for deformation_condition in deformation_conditions:
            for seed in seeds:
                print(
                    f"\nLaunching paper_run body={body_spec['label']} "
                    f"deform={deformation_condition} seed={seed}"
                )
                schedule_bundle = _schedule_bundle_for_preset(
                    "paper_run",
                    seed=seed,
                    schedules_dir=args.schedules_dir,
                    overrides=list(body_spec["overrides"]),
                )
                missing = [label for label, path in schedule_bundle.items() if not Path(path).exists()]
                if missing:
                    print(
                        f"Missing terrain schedules for paper_run body={body_spec['label']} "
                        f"deform={deformation_condition} seed={seed}: {missing}"
                    )
                    for label in missing:
                        print(f"  {label}: {schedule_bundle[label]}")
                    if args.require_schedules:
                        sys.exit(1)
                    print("  Continuing without missing schedule overrides.")

                overrides: list[str] = []
                if deformation_condition == "off":
                    overrides.append("nca.ablate_deformation=true")
                _run([
                    "modal", "run", "--detach",
                    "scripts/modal_train.py",
                    "--run", "paper_run",
                    "--body", str(body_spec["body"]),
                    *(
                        ["--canvas-w", str(body_spec["canvas_w"]), "--canvas-h", str(body_spec["canvas_h"])]
                        if body_spec["canvas_w"] is not None and body_spec["canvas_h"] is not None
                        else []
                    ),
                    "--seed", str(seed),
                    *(
                        ["--schedule", schedule_bundle["train"]]
                        if Path(schedule_bundle["train"]).exists()
                        else []
                    ),
                    *(
                        ["--rerank-schedule", schedule_bundle["rerank"]]
                        if Path(schedule_bundle["rerank"]).exists()
                        else []
                    ),
                    *(
                        ["--eval-schedule", schedule_bundle["eval"]]
                        if Path(schedule_bundle["eval"]).exists()
                        else []
                    ),
                    *(["--override", ",".join(overrides)] if overrides else []),
                    *(["--stop-after-gens", str(args.stage_gens)] if args.stage_gens is not None else []),
                ], check=False)

    print(f"\nLaunched {total_runs} runs. Monitor via: modal container list")


def cmd_extend(args):
    """Resume paused staged runs through a later generation boundary."""
    print("=== Extend ===")
    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        print(f"Outputs directory not found: {outputs_dir}")
        sys.exit(1)

    run_dirs = _paused_stage_runs(outputs_dir)
    if not run_dirs:
        print("No paused staged runs found.")
        return

    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.json"
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        completed_generations = manifest.get("summary", {}).get("completed_generations")
        if completed_generations is not None and int(completed_generations) >= int(args.to_gens):
            print(
                f"Skipping {run_dir.name}: already at completed_generations={completed_generations}"
            )
            continue
        print(f"\nExtending {run_dir.name} to generation budget {args.to_gens}")
        _run(
            [
                "modal",
                "run",
                "--detach",
                "scripts/modal_train.py",
                "--continue-output-dir",
                str(_remote_run_dir(run_dir.name)),
                "--stop-after-gens",
                str(args.to_gens),
            ],
            check=False,
        )


def cmd_eval(args):
    """Run held-out evaluation on all final checkpoints."""
    from sensing_edges.publication import (
        load_publication_set,
        publication_exclude_reason,
        resolve_latest_durable_checkpoint_prefix,
        resolve_release_id,
    )

    print("=== Eval ===")
    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        print(f"Outputs directory not found: {outputs_dir}")
        sys.exit(1)

    publication_set = load_publication_set(args.publication_set, se_root=SE_ROOT)
    release_id = resolve_release_id(release_id=args.release_id, publication_set=publication_set)
    local_release_dir = outputs_dir / "publication" / release_id
    for run_dir in sorted(outputs_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "publication":
            continue
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        config_path = run_dir / "resolved_config.yaml"
        metadata = None
        if config_path.exists():
            from omegaconf import OmegaConf
            from sensing_edges.runtime_config import run_metadata

            metadata = run_metadata(OmegaConf.load(config_path))
        exclude_reason = publication_exclude_reason(manifest, metadata, publication_set)
        if exclude_reason is not None:
            continue

        checkpoint_prefix = resolve_latest_durable_checkpoint_prefix(run_dir, manifest)
        if checkpoint_prefix is None:
            print(f"Skipping {run_dir.name}: no durable checkpoint found.")
            continue
        run_id = str(manifest.get("run_id", run_dir.name))
        local_summary_path = local_release_dir / "summaries" / f"{run_id}.json"
        if local_summary_path.exists() and not args.force:
            print(f"Skipping {run_dir.name}: publication summary already present at {local_summary_path}")
            continue

        print(f"\nEvaluating {run_dir.name} @ {checkpoint_prefix.name}")
        if args.schedule:
            rerank_schedule = args.schedule
            eval_schedule = args.schedule
        else:
            schedule_bundle = _schedule_bundle_for_resolved_config(
                _config_path_for_run(run_dir),
                schedules_dir=args.schedules_dir,
            )
            rerank_schedule = schedule_bundle["rerank"]
            eval_schedule = schedule_bundle["eval"]
        remote_checkpoint_prefix = _remote_run_dir(run_dir.name) / "checkpoints" / checkpoint_prefix.name
        remote_config_path = _remote_run_dir(run_dir.name) / "resolved_config.yaml"
        publication_summary_output = (
            Path("/outputs") / "publication" / release_id / "summaries" / f"{run_id}.json"
        )
        _run([
            "modal", "run", "scripts/modal_eval.py",
            "--checkpoint", str(remote_checkpoint_prefix),
            "--rerank-schedule", rerank_schedule,
            "--eval-schedule", eval_schedule,
            "--config-path", str(remote_config_path),
            "--publication-summary-output", str(publication_summary_output),
        ], check=False)

    _run(
        [
            "modal",
            "run",
            "scripts/modal_release.py",
            "--action",
            "materialize",
            "--outputs-dir",
            "/outputs",
            "--publication-set",
            args.publication_set,
            "--release-id",
            release_id,
        ],
        check=False,
    )


def cmd_figures(args):
    """Generate figures and tables."""
    from sensing_edges.publication import load_publication_set, resolve_release_id

    publication_set = load_publication_set(args.publication_set, se_root=SE_ROOT)
    release_id = resolve_release_id(release_id=args.release_id, publication_set=publication_set)
    print("=== Figures ===")
    _run(
        [
            "modal",
            "run",
            "scripts/modal_release.py",
            "--action",
            "figures",
            "--outputs-dir",
            "/outputs",
            "--publication-set",
            args.publication_set,
            "--release-id",
            release_id,
        ]
    )
    print("Figures and tables generated.")


def cmd_bundle(args):
    """Package release bundle."""
    from sensing_edges.publication import load_publication_set, resolve_release_id

    publication_set = load_publication_set(args.publication_set, se_root=SE_ROOT)
    release_id = resolve_release_id(release_id=args.release_id, publication_set=publication_set)
    print("=== Bundle ===")
    _run(
        [
            "modal",
            "run",
            "scripts/modal_release.py",
            "--action",
            "bundle",
            "--outputs-dir",
            "/outputs",
            "--publication-set",
            args.publication_set,
            "--release-id",
            release_id,
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Sensing Edges paper protocol.")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("preflight")
    p.add_argument("--allow-dirty", action="store_true")

    sub.add_parser("pilot")

    p = sub.add_parser("core")
    p.add_argument("--seeds", default=None)
    p.add_argument("--bodies", default=None)
    p.add_argument("--schedules-dir", default=DEFAULT_SCHEDULES_DIR)
    p.add_argument("--publication-set", default=DEFAULT_PUBLICATION_SET)
    p.add_argument("--release-id", default=None)
    p.add_argument("--stage-gens", type=int, default=None)
    p.add_argument("--require-schedules", action="store_true", default=True)
    p.add_argument("--allow-missing-schedules", action="store_false", dest="require_schedules")

    p = sub.add_parser("extend")
    p.add_argument("--outputs-dir", default="local_outputs")
    p.add_argument("--to-gens", type=int, required=True)

    p = sub.add_parser("eval")
    p.add_argument("--outputs-dir", default="local_outputs")
    p.add_argument("--schedule", default=None)
    p.add_argument("--schedules-dir", default=DEFAULT_SCHEDULES_DIR)
    p.add_argument("--publication-set", default=DEFAULT_PUBLICATION_SET)
    p.add_argument("--release-id", default=None)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("figures")
    p.add_argument("--publication-set", default=DEFAULT_PUBLICATION_SET)
    p.add_argument("--release-id", default=None)
    p = sub.add_parser("bundle")
    p.add_argument("--publication-set", default=DEFAULT_PUBLICATION_SET)
    p.add_argument("--release-id", default=None)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "preflight": cmd_preflight,
        "pilot": cmd_pilot,
        "core": cmd_core,
        "extend": cmd_extend,
        "eval": cmd_eval,
        "figures": cmd_figures,
        "bundle": cmd_bundle,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
