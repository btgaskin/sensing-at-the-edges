#!/usr/bin/env python
"""Generate frozen terrain schedules for reproducible experiments.

Usage:
    uv run python scripts/generate_terrain_suite.py --variant curriculum_zone10to19_extend24_gap2_30 --seed 42 --n-gens 50
    uv run python scripts/generate_terrain_suite.py --variant single_gap2_zone10to24_30 --seed 42 --n-gens 500
    uv run python scripts/generate_terrain_suite.py --variant curriculum_zone10to19_extend24_gap2_30 --seeds 42,99,7,13,21 --n-gens 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

# Add experiment root to path for imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sensing_edges.terrain_schedule import generate_schedule, save_schedule, schedule_bundle_path


def _load_variant_config(variant: str) -> dict:
    config_path = Path(__file__).resolve().parent.parent / "config" / "terrain" / f"{variant}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Terrain variant config not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    loaded = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"Terrain variant config must resolve to a mapping: {config_path}")
    return loaded


def main():
    parser = argparse.ArgumentParser(description="Generate frozen terrain schedules.")
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        help="Terrain variant name from config/terrain/<variant>.yaml",
    )
    parser.add_argument("--seed", type=int, default=42, help="Master seed (ignored if --seeds)")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds (generates one suite per seed)")
    parser.add_argument("--n-gens", type=int, default=50, help="Number of training generations")
    parser.add_argument("--n-episodes", type=int, default=4, help="Episodes per generation")
    parser.add_argument("--n-val", type=int, default=8, help="Validation terrains")
    parser.add_argument("--n-test", type=int, default=16, help="Test terrains")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output directory (default: "
            "terrain_suites/seed_<seed>_<variant>_g<gens>_ep<episodes>_val<val>_test<test>)"
        ),
    )
    args = parser.parse_args()

    terrain_cfg = _load_variant_config(args.variant)

    seeds = [args.seed]
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]

    for seed in seeds:
        output_dir = args.output or str(
            schedule_bundle_path(
                "terrain_suites",
                seed=seed,
                variant=args.variant,
                n_gens=args.n_gens,
                n_episodes=args.n_episodes,
                n_val=args.n_val,
                n_test=args.n_test,
            )
        )
        print(
            f"Generating schedule: variant={args.variant} seed={seed}, "
            f"gens={args.n_gens}, episodes={args.n_episodes}"
        )

        schedule = generate_schedule(
            seed=seed,
            n_gens=args.n_gens,
            n_episodes=args.n_episodes,
            n_val=args.n_val,
            n_test=args.n_test,
            terrain_cfg=terrain_cfg,
        )
        out_path = save_schedule(schedule, output_dir)

        n_train = args.n_gens * args.n_episodes
        print(f"  Output: {out_path}")
        print(f"  Train: {n_train} terrains ({args.n_gens} gens x {args.n_episodes} eps)")
        print(f"  Val:   {args.n_val} terrains")
        print(f"  Test:  {args.n_test} terrains")
        print(f"  Hashes: train={schedule.train_hash[:16]}... val={schedule.val_hash[:16]}... test={schedule.test_hash[:16]}...")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
