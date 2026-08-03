"""Modal entrypoint for sensing_edges training.

Usage:
    modal run scripts/modal_train.py
    modal run scripts/modal_train.py --run paper_run --body rect_7x2
    modal run scripts/modal_train.py --run paper_run --body rect_7x2 --seed 137 --schedule terrain_suites/seed_137_bilateral_goal_cliff_asymmetric_pairs_7x2_grid30_g150_ep6_val6_test6
    modal run scripts/modal_train.py --run iter_run --terrain gap_curriculum --curriculum-phase-ends 10,20,30 --gens 30 --pop 150 --env-steps 2500 --gpu a100-40gb
    modal run --detach scripts/modal_train.py --run iter_run --body grown --canvas-w 7 --canvas-h 3 --seed 137 --pop 150 --gens 20 --env-steps 2500 --eval-env-steps 3000 --phys-substeps 15
"""

from __future__ import annotations

from pathlib import Path
import re

import modal
from omegaconf import OmegaConf

from _modal_common import VOLUMES, app, commit_volumes, compose_config, parse_override_csv, se_image


# --- Training ---


_PHYS_SUBSTEP_TO_SIM = {
    12: "twelve",
    15: "half",
    20: "twenty",
    30: "full",
}
_DEFAULT_GROWN_CANVAS = (7, 3)
_PREFERRED_EMITTER_COUNTS = (12, 10, 8, 5)
_TRAIN_GPU_TARGETS = {
    "l4": {"modal_gpu": "L4", "profile_id": "l4"},
    "a100-40gb": {"modal_gpu": "A100-40GB", "profile_id": "a100_40gb"},
    "a100-80gb": {"modal_gpu": "A100-80GB", "profile_id": "a100_80gb"},
    "h100": {"modal_gpu": "H100!", "profile_id": "h100"},
    "h200": {"modal_gpu": "H200", "profile_id": "h200"},
    "b200": {"modal_gpu": "B200", "profile_id": "b200"},
}
_HIGH_END_GPU_PROFILES = frozenset({"a100_40gb", "a100_80gb", "h100", "h200", "b200"})
_STRUCTURED_CONFIG_GROUP_KEYS = frozenset(
    {
        "audit",
        "batching",
        "body",
        "collision",
        "eval",
        "gasnet",
        "gpu_profile",
        "jax",
        "logging",
        "nca",
        "qd",
        "render",
        "sim",
        "specimens",
        "terrain",
        "train",
        "wandb",
    }
)
_RECT_BODY_RE = re.compile(r"rect_(?P<w>\d+)x(?P<h>\d+)$")
_TERRAIN_ARG_ALIASES = {
    "gap": "gap_curriculum",
    "gaps": "gap_curriculum",
    "gap_curriculum": "gap_curriculum",
    "plateau_gap": "gap_curriculum",
    "plateau_gap_curriculum": "gap_curriculum",
    "cliff": "cliff",
    "probe_gap": "probe_gap",
    "ledge": "probe_ledge",
    "probe_ledge": "probe_ledge",
    "gap_then_nogap": "gap_then_nogap",
}
_TERRAIN_SURFACE_CONFIGS = {
    "gap_curriculum": {
        "train_variant": {
            "default": "curriculum_bilateral_goal_cliff_asymmetric_pairs_plateau_gap_mix123_30",
            "8x": "curriculum_bilateral_goal_cliff_asymmetric_pairs_plateau_gap_mix123_8x2_30",
        },
        "eval_variant": {
            "default": "bilateral_goal_cliff_asymmetric_pairs_plateau_gap_w3_30",
            "8x": "bilateral_goal_cliff_asymmetric_pairs_plateau_gap_w3_8x2_30",
        },
        "n_phases": 3,
    },
    "gap_then_nogap": {
        "train_variant": {
            "default": "curriculum_bilateral_goal_cliff_asymmetric_pairs_gap_then_nogap_30",
            "8x": "curriculum_bilateral_goal_cliff_asymmetric_pairs_gap_then_nogap_8x2_30",
        },
        "eval_variant": {
            "default": "bilateral_goal_cliff_asymmetric_pairs_30",
            "8x": "bilateral_goal_cliff_asymmetric_pairs_8x2_30",
        },
        "n_phases": 2,
    },
    "cliff": {
        "train_variant": {
            "default": "bilateral_goal_cliff_asymmetric_pairs_30",
            "8x": "bilateral_goal_cliff_asymmetric_pairs_8x2_30",
        },
        "eval_variant": {
            "default": "bilateral_goal_cliff_asymmetric_pairs_30",
            "8x": "bilateral_goal_cliff_asymmetric_pairs_8x2_30",
        },
    },
    "probe_gap": {
        "train_variant": {
            "default": "bilateral_goal_cliff_asymmetric_pairs_probe_gap_30",
            "8x": "bilateral_goal_cliff_asymmetric_pairs_probe_gap_30",
        },
        "eval_variant": {
            "default": "bilateral_goal_cliff_asymmetric_pairs_probe_gap_30",
            "8x": "bilateral_goal_cliff_asymmetric_pairs_probe_gap_30",
        },
    },
    "probe_ledge": {
        "train_variant": {
            "default": "bilateral_goal_cliff_asymmetric_pairs_probe_ledge_4_30",
            "8x": "bilateral_goal_cliff_asymmetric_pairs_probe_ledge_8x2_30",
        },
        "eval_variant": {
            "default": "bilateral_goal_cliff_asymmetric_pairs_probe_ledge_4_30",
            "8x": "bilateral_goal_cliff_asymmetric_pairs_probe_ledge_8x2_30",
        },
    },
}


def _train_result_payload(*, archive, output_dir: str) -> dict[str, object]:
    """Return a Modal-safe result payload using only Python scalars."""
    return {
        "best_fitness": float(archive.best_fitness),
        "coverage": float(archive.coverage),
        "qd_score": float(archive.qd_score),
        "output_dir": str(output_dir),
    }


def _parse_cli_overrides(seed: int | None, override: str | None) -> list[str]:
    """Parse comma-separated Hydra overrides without mutating their semantics."""
    overrides = []
    if seed is not None:
        overrides.append(f"seed={seed}")
    overrides.extend(parse_override_csv(override))
    return overrides


def _force_wandb_enabled(overrides: list[str] | None) -> list[str]:
    """Normalize overrides so Modal-backed runs always keep W&B enabled."""
    resolved: list[str] = []
    for override in overrides or []:
        token = str(override).strip()
        if not token:
            continue
        if token == "~wandb" or token.startswith("~wandb."):
            continue
        if token.startswith("wandb.enabled="):
            continue
        resolved.append(token)
    resolved.append("wandb.enabled=true")
    return resolved


def _force_gpu_profile(overrides: list[str] | None, *, gpu_profile: str) -> list[str]:
    """Normalize overrides so runtime metadata matches the selected Modal GPU."""
    resolved: list[str] = []
    for override in overrides or []:
        token = str(override).strip()
        if not token:
            continue
        if token == "~gpu_profile" or token.startswith("~gpu_profile."):
            continue
        if token.startswith("gpu_profile="):
            continue
        resolved.append(token)
    resolved.append(f"gpu_profile={gpu_profile}")
    return resolved


def _resolve_gpu_target(
    *,
    preset: str,
    overrides: list[str] | None,
    gpu: str | None,
) -> tuple[str, dict[str, str]]:
    """Resolve the Modal GPU target, defaulting to the preset gpu_profile."""
    gpu_token = str(gpu).strip().lower() if gpu is not None else "preset"
    if gpu_token in {"", "preset"}:
        preview_cfg = compose_config(preset=preset, overrides=overrides)
        preview_profile = str(preview_cfg.gpu_profile.id).strip().lower()
        gpu_token = preview_profile.replace("_", "-")
    gpu_target = _TRAIN_GPU_TARGETS.get(gpu_token)
    if gpu_target is None:
        raise SystemExit(
            f"Unsupported --gpu value: {gpu!r}. "
            f"Expected one of: preset, {', '.join(sorted(_TRAIN_GPU_TARGETS))}"
        )
    return gpu_token, gpu_target


def _default_output_dir_override(*, preset: str, resolved_overrides: list[str]) -> str:
    """Return the automatic train.output_dir override with a config identity suffix."""
    from sensing_edges.runtime_config import logical_config_sha256, run_name_label

    preview_cfg = compose_config(preset=preset, overrides=resolved_overrides)
    run_name = preview_cfg.get("preset_id") or "default"
    run_hash = logical_config_sha256(preview_cfg)[:8]
    return f"train.output_dir=/outputs/{run_name}_{run_name_label(preview_cfg)}_{run_hash}"


def _default_output_dir_for_cfg(cfg) -> str:
    """Return the automatic run output dir for an already-resolved config."""
    from sensing_edges.runtime_config import logical_config_sha256, run_name_label

    run_name = cfg.get("preset_id") or "default"
    run_hash = logical_config_sha256(cfg)[:8]
    return f"/outputs/{run_name}_{run_name_label(cfg)}_{run_hash}"


def _load_resolved_cfg_from_output_dir(output_dir: str | Path):
    config_path = Path(output_dir) / "resolved_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Resolved config not found: {config_path}")
    stored_cfg = OmegaConf.load(config_path)
    base_preset = str(stored_cfg.get("preset_id") or "config")
    stored_cfg = _normalize_legacy_resolved_cfg(stored_cfg, base_preset=base_preset)
    base_cfg = compose_config(preset=base_preset)
    pruned_cfg = OmegaConf.create(
        _prune_unknown_config_keys(
            OmegaConf.to_container(base_cfg, resolve=False),
            OmegaConf.to_container(stored_cfg, resolve=False),
        )
    )
    return OmegaConf.merge(base_cfg, pruned_cfg)


def _structured_group_override_node(*, base_preset: str, token: str):
    key = str(token).split("=", 1)[0].strip()
    group_cfg = compose_config(preset=base_preset, overrides=[token])
    return key, group_cfg.get(key)


def _normalize_legacy_resolved_cfg(stored_cfg, *, base_preset: str):
    normalized = OmegaConf.create(OmegaConf.to_container(stored_cfg, resolve=False))
    for key in _STRUCTURED_CONFIG_GROUP_KEYS:
        value = normalized.get(key)
        if not isinstance(value, str):
            continue
        group_key, group_node = _structured_group_override_node(
            base_preset=base_preset,
            token=f"{key}={value}",
        )
        normalized = OmegaConf.merge(
            normalized,
            OmegaConf.create(
                {
                    group_key: OmegaConf.to_container(group_node, resolve=False),
                }
            ),
        )
    return normalized


def _prune_unknown_config_keys(schema, value):
    if isinstance(schema, dict) and isinstance(value, dict):
        pruned: dict[str, object] = {}
        for key, child_value in value.items():
            if key not in schema:
                continue
            pruned[key] = _prune_unknown_config_keys(schema[key], child_value)
        return pruned
    return value


def _apply_assignment_overrides(cfg, overrides: list[str] | None):
    """Apply simple key=value overrides to a resolved config payload."""
    base_preset = str(cfg.get("preset_id") or "config")
    merged = cfg
    dotlist: list[str] = []
    for override in overrides or []:
        token = str(override).strip()
        if not token:
            continue
        if token.startswith("+") or token.startswith("~"):
            raise ValueError(
                "Continuation/branch execution only supports assignment overrides "
                f"(got {token!r})"
            )
        key = token.split("=", 1)[0].strip()
        if "." not in key and key in _STRUCTURED_CONFIG_GROUP_KEYS:
            group_key, group_node = _structured_group_override_node(
                base_preset=base_preset,
                token=token,
            )
            merged = OmegaConf.merge(
                merged,
                OmegaConf.create(
                    {
                        group_key: OmegaConf.to_container(group_node, resolve=False),
                    }
                ),
            )
            continue
        dotlist.append(token)
    if not dotlist:
        return merged
    return OmegaConf.merge(merged, OmegaConf.from_dotlist(dotlist))


def _latest_checkpoint_for_output_dir(output_dir: str | Path) -> str:
    from sensing_edges.checkpoint import find_latest_checkpoint

    checkpoint_dir = Path(output_dir) / "checkpoints"
    latest = find_latest_checkpoint(checkpoint_dir, require_done=True)
    if latest is None:
        raise FileNotFoundError(f"No durable checkpoints found in {checkpoint_dir}")
    checkpoint_prefix, _meta = latest
    return str(checkpoint_prefix)


def _resolve_preset_name(*, preset: str, run: str | None) -> str:
    if run is None:
        return preset
    if preset not in {"config", run}:
        raise ValueError(f"Cannot combine preset={preset!r} with run={run!r}")
    return run


def _append_run_surface_overrides(
    overrides: list[str] | None,
    *,
    run: str | None,
    gpu_profile: str | None = None,
    pop: int | None = None,
    gens: int | None = None,
    stop_after_gens: int | None = None,
    env_steps: int | None = None,
    eval_env_steps: int | None = None,
    phys_substeps: int | None = None,
) -> list[str]:
    resolved = list(overrides or [])
    if pop is not None:
        total_pop = int(pop)
        if total_pop <= 0:
            raise ValueError("pop must be positive")
        preferred_emitter_counts = (
            (12, 8, 10, 5)
            if total_pop == 200
            else _PREFERRED_EMITTER_COUNTS
        )
        population_layout = next(
            (
                (n_emitters, total_pop // n_emitters)
                for n_emitters in preferred_emitter_counts
                if total_pop % n_emitters == 0
            ),
            None,
        )
        if population_layout is None:
            raise ValueError("pop must be divisible by 12, 10, 8, or 5")
        n_emitters, emitter_pop_size = population_layout
        resolved.extend(
            [
                f"qd.n_emitters={n_emitters}",
                f"qd.emitter_pop_size={emitter_pop_size}",
            ]
        )
        if total_pop <= 200 or (
            gpu_profile in _HIGH_END_GPU_PROFILES and total_pop <= 400
        ):
            resolved.extend(
                [
                    f"batching.generation.candidate_chunk_size={total_pop}",
                    f"batching.generation.candidate_bucket_sizes=[{total_pop}]",
                    f"batching.generation.exact_fit_small_run_threshold={total_pop}",
                ]
            )
    if gens is not None:
        resolved.append(f"train.n_generations={int(gens)}")
    if stop_after_gens is not None:
        resolved.append(f"train.stop_after_generations={int(stop_after_gens)}")
    if env_steps is not None:
        steps = int(env_steps)
        resolved.append(f"train.max_steps={steps}")
        if eval_env_steps is None and run in {"iter_run", "paper_run"}:
            eval_env_steps = steps
    if eval_env_steps is not None:
        eval_steps = int(eval_env_steps)
        resolved.append(f"eval.rerank_max_steps={eval_steps}")
        resolved.append(f"eval.test_max_steps={eval_steps}")
    if phys_substeps is not None:
        substeps = int(phys_substeps)
        sim_preset = _PHYS_SUBSTEP_TO_SIM.get(substeps)
        if sim_preset is None:
            raise ValueError("phys_substeps must be one of 12, 15, 20, or 30")
        resolved.append(f"sim={sim_preset}")
    return resolved


def _append_body_surface_overrides(
    overrides: list[str] | None,
    *,
    body: str | None = None,
    canvas_w: int | None = None,
    canvas_h: int | None = None,
) -> list[str]:
    """Translate body-surface CLI args into Hydra overrides."""
    resolved = list(overrides or [])
    if body is None:
        if canvas_w is not None or canvas_h is not None:
            raise ValueError("--canvas-w/--canvas-h require --body grown")
        return resolved

    token = str(body).strip().lower()
    rect_match = _RECT_BODY_RE.fullmatch(token)
    if rect_match is not None:
        if canvas_w is not None or canvas_h is not None:
            raise ValueError("Canvas overrides are only valid with --body grown")
        rect_w = int(rect_match.group("w"))
        rect_h = int(rect_match.group("h"))
        resolved.extend(
            [
                "body.mode=fixed",
                "body.shape=rectangle",
                f"terrain.robot_w={rect_w}",
                f"terrain.robot_h={rect_h}",
            ]
        )
        return resolved
    if token == "comb":
        if canvas_w is not None or canvas_h is not None:
            raise ValueError("Canvas overrides are only valid with --body grown")
        resolved.extend(
            [
                "body.mode=fixed",
                "body.shape=comb",
                "terrain.robot_w=7",
                "terrain.robot_h=2",
            ]
        )
        return resolved
    if token == "comb_7x3":
        if canvas_w is not None or canvas_h is not None:
            raise ValueError("Canvas overrides are only valid with --body grown")
        resolved.extend(
            [
                "body.mode=fixed",
                "body.shape=comb_7x3",
                "terrain.robot_w=7",
                "terrain.robot_h=3",
            ]
        )
        return resolved
    if token == "grown":
        if canvas_w is None and canvas_h is None:
            canvas_w, canvas_h = _DEFAULT_GROWN_CANVAS
        elif canvas_w is None or canvas_h is None:
            raise ValueError("--body grown requires both --canvas-w and --canvas-h")
        resolved.extend(
            [
                "body.mode=grown",
                f"body.growth.max_robot_w={int(canvas_w)}",
                f"body.growth.max_robot_h={int(canvas_h)}",
            ]
        )
        return resolved
    raise ValueError(f"Unsupported --body value: {body!r}")


def _body_width_hint(*, body: str | None = None, canvas_w: int | None = None) -> int:
    """Infer an approximate body width from CLI args for terrain alias selection."""
    if body is None:
        return 6
    token = str(body).strip().lower()
    rect_match = _RECT_BODY_RE.fullmatch(token)
    if rect_match is not None:
        return int(rect_match.group("w"))
    if token == "comb":
        return 7
    if token == "comb_7x3":
        return 7
    if token == "grown":
        return int(canvas_w) if canvas_w is not None else _DEFAULT_GROWN_CANVAS[0]
    return 6


def _normalize_curriculum_switches(
    phase_spec: str,
    *,
    n_phases: int,
) -> list[int]:
    """Accept either switch generations or full phase-end generations."""
    values = [int(token.strip()) for token in str(phase_spec).split(",") if token.strip()]
    if len(values) == n_phases:
        switches = values[:-1]
    elif len(values) == n_phases - 1:
        switches = values
    else:
        raise ValueError(
            "curriculum phase spec must provide either "
            f"{n_phases - 1} switch gens or {n_phases} phase-end gens."
        )
    if any(value <= 0 for value in switches):
        raise ValueError("curriculum phase switch gens must be positive integers.")
    if any(curr <= prev for prev, curr in zip(switches, switches[1:])):
        raise ValueError("curriculum phase switch gens must be strictly increasing.")
    return switches


def _append_terrain_surface_overrides(
    overrides: list[str] | None,
    *,
    terrain: str | None = None,
    body: str | None = None,
    canvas_w: int | None = None,
    curriculum_phase_ends: str | None = None,
) -> list[str]:
    """Translate terrain aliases into consistent train/rerank/test overrides."""
    resolved = list(overrides or [])
    if terrain is None:
        if curriculum_phase_ends is not None:
            raise ValueError("--curriculum-phase-ends requires --terrain")
        return resolved

    token = str(terrain).strip().lower()
    canonical = _TERRAIN_ARG_ALIASES.get(token)
    if canonical is None:
        raise ValueError(f"Unsupported --terrain value: {terrain!r}")
    surface_cfg = _TERRAIN_SURFACE_CONFIGS[canonical]
    body_width = _body_width_hint(body=body, canvas_w=canvas_w)
    variant_key = "8x" if body_width >= 8 else "default"
    train_variant = str(surface_cfg["train_variant"][variant_key])
    eval_variant = str(surface_cfg["eval_variant"][variant_key])
    resolved.extend(
        [
            f"terrain={train_variant}",
            f"eval.rerank_variant={eval_variant}",
            f"eval.test_variant={eval_variant}",
        ]
    )
    if curriculum_phase_ends is not None:
        n_phases = surface_cfg.get("n_phases")
        if n_phases is None:
            raise ValueError(f"--curriculum-phase-ends is not supported for terrain={terrain!r}")
        switches = _normalize_curriculum_switches(curriculum_phase_ends, n_phases=int(n_phases))
        resolved.append(
            "terrain.curriculum_phase_switch_gens=[" + ",".join(str(value) for value in switches) + "]"
        )
    return resolved


def _append_schedule_overrides(
    overrides: list[str] | None,
    *,
    schedule: str | None = None,
    rerank_schedule: str | None = None,
    eval_schedule: str | None = None,
) -> list[str]:
    """Thread explicit schedule directories through Hydra overrides."""
    resolved = list(overrides or [])
    if schedule:
        resolved.append(f"train.terrain_schedule_dir={schedule}")
    if rerank_schedule:
        resolved.append(f"eval.rerank_schedule_dir={rerank_schedule}")
    if eval_schedule:
        resolved.append(f"eval.test_schedule_dir={eval_schedule}")
    return resolved


def _schedule_bundle_for_cfg(cfg) -> dict[str, str]:
    from sensing_edges.terrain_schedule import schedule_bundle_path

    seed = int(cfg.get("seed", 0))
    train_variant = str(cfg.terrain.variant)
    rerank_variant = str(cfg.eval.get("rerank_variant") or train_variant)
    eval_variant = str(cfg.eval.get("test_variant") or rerank_variant)
    n_gens = int(cfg.train.n_generations)
    n_episodes = int(cfg.train.n_episodes)
    n_val = int(cfg.eval.rerank_episodes)
    n_test = int(cfg.eval.test_episodes)
    return {
        "train": str(
            schedule_bundle_path(
                "terrain_suites",
                seed=seed,
                variant=train_variant,
                n_gens=n_gens,
                n_episodes=n_episodes,
                n_val=n_val,
                n_test=n_test,
            )
        ),
        "rerank": str(
            schedule_bundle_path(
                "terrain_suites",
                seed=seed,
                variant=rerank_variant,
                n_gens=n_gens,
                n_episodes=n_episodes,
                n_val=n_val,
                n_test=n_test,
            )
        ),
        "eval": str(
            schedule_bundle_path(
                "terrain_suites",
                seed=seed,
                variant=eval_variant,
                n_gens=n_gens,
                n_episodes=n_episodes,
                n_val=n_val,
                n_test=n_test,
            )
        ),
    }


def _resolve_schedule_args(
    *,
    preset: str,
    overrides: list[str] | None,
    schedule: str | None,
    rerank_schedule: str | None,
    eval_schedule: str | None,
) -> tuple[str | None, str | None, str | None]:
    from sensing_edges.terrain_schedule import resolve_schedule_bundle_path

    auto_resolve_presets = {"paper_run"}
    preview_cfg = compose_config(preset=preset, overrides=overrides)
    if preset not in auto_resolve_presets and not bool(preview_cfg.train.get("require_schedule_dir", False)):
        return schedule, rerank_schedule, eval_schedule
    resolved = (
        schedule
        or str(
            resolve_schedule_bundle_path(
                "terrain_suites",
                seed=int(preview_cfg.get("seed", 0)),
                variant=str(preview_cfg.terrain.variant),
                n_gens=int(preview_cfg.train.n_generations),
                n_episodes=int(preview_cfg.train.n_episodes),
                n_val=int(preview_cfg.eval.rerank_episodes),
                n_test=int(preview_cfg.eval.test_episodes),
            )
        ),
        rerank_schedule
        or str(
            resolve_schedule_bundle_path(
                "terrain_suites",
                seed=int(preview_cfg.get("seed", 0)),
                variant=str(preview_cfg.eval.get("rerank_variant") or preview_cfg.terrain.variant),
                n_gens=int(preview_cfg.train.n_generations),
                n_episodes=int(preview_cfg.train.n_episodes),
                n_val=int(preview_cfg.eval.rerank_episodes),
                n_test=int(preview_cfg.eval.test_episodes),
            )
        ),
        eval_schedule
        or str(
            resolve_schedule_bundle_path(
                "terrain_suites",
                seed=int(preview_cfg.get("seed", 0)),
                variant=str(
                    preview_cfg.eval.get("test_variant")
                    or preview_cfg.eval.get("rerank_variant")
                    or preview_cfg.terrain.variant
                ),
                n_gens=int(preview_cfg.train.n_generations),
                n_episodes=int(preview_cfg.train.n_episodes),
                n_val=int(preview_cfg.eval.rerank_episodes),
                n_test=int(preview_cfg.eval.test_episodes),
            )
        ),
    )
    for path in resolved:
        if path is not None and not Path(path).exists():
            raise FileNotFoundError(f"Required {preset} terrain schedule not found: {path}")
    return resolved


def _train_remote_impl(
    preset: str = "config",
    overrides: list[str] | None = None,
    schedule: str | None = None,
    rerank_schedule: str | None = None,
    eval_schedule: str | None = None,
    continue_output_dir: str | None = None,
    branch_from_output: str | None = None,
    branch_from_checkpoint: str | None = None,
):
    """Run training on GPU."""
    import os

    os.chdir("/root/experiment")
    from sensing_edges.logging_utils import configure_logging

    configure_logging("INFO")

    import jax
    import logging
    logging.getLogger(__name__).info(
        "Backend: %s  Devices: %s", jax.default_backend(), jax.devices()
    )

    from sensing_edges.runtime_config import parse_override_keys, resolve_runtime_config

    overrides = _force_wandb_enabled(overrides)
    continuation_sources = [
        value
        for value in (continue_output_dir, branch_from_output, branch_from_checkpoint)
        if value is not None
    ]
    if len(continuation_sources) > 1:
        raise ValueError(
            "Specify at most one of continue_output_dir, branch_from_output, or "
            "branch_from_checkpoint"
        )

    if continuation_sources:
        if continue_output_dir is not None:
            base_output_dir = Path(continue_output_dir)
            resolved_overrides = list(overrides or [])
            resolved_overrides.append(f"train.output_dir={base_output_dir}")
            resolved_overrides.append("train.resume_mode=staged")
        else:
            resume_prefix = (
                str(branch_from_checkpoint)
                if branch_from_checkpoint is not None
                else _latest_checkpoint_for_output_dir(str(branch_from_output))
            )
            base_output_dir = Path(branch_from_output) if branch_from_output is not None else Path(resume_prefix).with_suffix("").parent.parent
            resolved_overrides = list(overrides or [])
            resolved_overrides.append("train.resume_mode=branch")
            resolved_overrides.append(f"train.resume_from={resume_prefix}")

        resolved_overrides = _append_schedule_overrides(
            resolved_overrides,
            schedule=schedule,
            rerank_schedule=rerank_schedule,
            eval_schedule=eval_schedule,
        )
        cfg = _apply_assignment_overrides(
            _load_resolved_cfg_from_output_dir(base_output_dir),
            resolved_overrides,
        )
        override_keys = set(parse_override_keys(resolved_overrides))
        if "train.output_dir" not in override_keys:
            cfg.train.output_dir = _default_output_dir_for_cfg(cfg)
            resolved_overrides.append(f"train.output_dir={cfg.train.output_dir}")
        output_dir = str(cfg.train.output_dir)
        resolution = resolve_runtime_config(
            cfg,
            preset=cfg.get("preset_id"),
            runtime_overrides=resolved_overrides,
            config_dir="/root/experiment/config",
        )
    else:
        schedule, rerank_schedule, eval_schedule = _resolve_schedule_args(
            preset=preset,
            overrides=overrides,
            schedule=schedule,
            rerank_schedule=rerank_schedule,
            eval_schedule=eval_schedule,
        )
        resolved_overrides = _append_schedule_overrides(
            overrides,
            schedule=schedule,
            rerank_schedule=rerank_schedule,
            eval_schedule=eval_schedule,
        )

        override_keys = set(parse_override_keys(resolved_overrides))
        if "train.output_dir" not in override_keys:
            resolved_overrides.append(
                _default_output_dir_override(preset=preset, resolved_overrides=resolved_overrides)
            )

        cfg = compose_config(preset=preset, overrides=resolved_overrides)
        resolution = resolve_runtime_config(
            cfg,
            preset=preset,
            runtime_overrides=resolved_overrides,
            config_dir="/root/experiment/config",
        )

    output_dir = str(cfg.train.output_dir)

    from sensing_edges.train import train

    try:
        archive = train(
            cfg,
            runtime_resolution=resolution,
            on_checkpoint_complete=commit_volumes,
        )
        logging.getLogger(__name__).info(
            "Done. best=%.2f cov=%.3f qd=%.1f",
            archive.best_fitness, archive.coverage, archive.qd_score,
        )
        return _train_result_payload(archive=archive, output_dir=output_dir)
    finally:
        commit_volumes()


@app.function(
    image=se_image,
    gpu="L4",
    timeout=21600,  # 6 hours
    retries=modal.Retries(initial_delay=0.0, max_retries=3),
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("wandb-key")],
)
def train_remote_l4(
    preset: str = "config",
    overrides: list[str] | None = None,
    schedule: str | None = None,
    rerank_schedule: str | None = None,
    eval_schedule: str | None = None,
    continue_output_dir: str | None = None,
    branch_from_output: str | None = None,
    branch_from_checkpoint: str | None = None,
):
    return _train_remote_impl(
        preset=preset,
        overrides=overrides,
        schedule=schedule,
        rerank_schedule=rerank_schedule,
        eval_schedule=eval_schedule,
        continue_output_dir=continue_output_dir,
        branch_from_output=branch_from_output,
        branch_from_checkpoint=branch_from_checkpoint,
    )


@app.function(
    image=se_image,
    gpu="A100-40GB",
    timeout=21600,  # 6 hours
    retries=modal.Retries(initial_delay=0.0, max_retries=3),
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("wandb-key")],
)
def train_remote_a100_40gb(
    preset: str = "config",
    overrides: list[str] | None = None,
    schedule: str | None = None,
    rerank_schedule: str | None = None,
    eval_schedule: str | None = None,
    continue_output_dir: str | None = None,
    branch_from_output: str | None = None,
    branch_from_checkpoint: str | None = None,
):
    return _train_remote_impl(
        preset=preset,
        overrides=overrides,
        schedule=schedule,
        rerank_schedule=rerank_schedule,
        eval_schedule=eval_schedule,
        continue_output_dir=continue_output_dir,
        branch_from_output=branch_from_output,
        branch_from_checkpoint=branch_from_checkpoint,
    )


@app.function(
    image=se_image,
    gpu="A100-80GB",
    timeout=21600,  # 6 hours
    retries=modal.Retries(initial_delay=0.0, max_retries=3),
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("wandb-key")],
)
def train_remote_a100_80gb(
    preset: str = "config",
    overrides: list[str] | None = None,
    schedule: str | None = None,
    rerank_schedule: str | None = None,
    eval_schedule: str | None = None,
    continue_output_dir: str | None = None,
    branch_from_output: str | None = None,
    branch_from_checkpoint: str | None = None,
):
    return _train_remote_impl(
        preset=preset,
        overrides=overrides,
        schedule=schedule,
        rerank_schedule=rerank_schedule,
        eval_schedule=eval_schedule,
        continue_output_dir=continue_output_dir,
        branch_from_output=branch_from_output,
        branch_from_checkpoint=branch_from_checkpoint,
    )


@app.function(
    image=se_image,
    gpu="H100!",
    timeout=21600,  # 6 hours
    retries=modal.Retries(initial_delay=0.0, max_retries=3),
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("wandb-key")],
)
def train_remote_h100(
    preset: str = "config",
    overrides: list[str] | None = None,
    schedule: str | None = None,
    rerank_schedule: str | None = None,
    eval_schedule: str | None = None,
    continue_output_dir: str | None = None,
    branch_from_output: str | None = None,
    branch_from_checkpoint: str | None = None,
):
    return _train_remote_impl(
        preset=preset,
        overrides=overrides,
        schedule=schedule,
        rerank_schedule=rerank_schedule,
        eval_schedule=eval_schedule,
        continue_output_dir=continue_output_dir,
        branch_from_output=branch_from_output,
        branch_from_checkpoint=branch_from_checkpoint,
    )


@app.function(
    image=se_image,
    gpu="H200",
    timeout=21600,  # 6 hours
    retries=modal.Retries(initial_delay=0.0, max_retries=3),
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("wandb-key")],
)
def train_remote_h200(
    preset: str = "config",
    overrides: list[str] | None = None,
    schedule: str | None = None,
    rerank_schedule: str | None = None,
    eval_schedule: str | None = None,
    continue_output_dir: str | None = None,
    branch_from_output: str | None = None,
    branch_from_checkpoint: str | None = None,
):
    return _train_remote_impl(
        preset=preset,
        overrides=overrides,
        schedule=schedule,
        rerank_schedule=rerank_schedule,
        eval_schedule=eval_schedule,
        continue_output_dir=continue_output_dir,
        branch_from_output=branch_from_output,
        branch_from_checkpoint=branch_from_checkpoint,
    )


@app.function(
    image=se_image,
    gpu="B200",
    timeout=21600,  # 6 hours
    retries=modal.Retries(initial_delay=0.0, max_retries=3),
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("wandb-key")],
)
def train_remote_b200(
    preset: str = "config",
    overrides: list[str] | None = None,
    schedule: str | None = None,
    rerank_schedule: str | None = None,
    eval_schedule: str | None = None,
    continue_output_dir: str | None = None,
    branch_from_output: str | None = None,
    branch_from_checkpoint: str | None = None,
):
    return _train_remote_impl(
        preset=preset,
        overrides=overrides,
        schedule=schedule,
        rerank_schedule=rerank_schedule,
        eval_schedule=eval_schedule,
        continue_output_dir=continue_output_dir,
        branch_from_output=branch_from_output,
        branch_from_checkpoint=branch_from_checkpoint,
    )


@app.local_entrypoint()
def main(
    preset: str = "config",
    run: str | None = None,
    body: str | None = None,
    terrain: str | None = None,
    curriculum_phase_ends: str | None = None,
    canvas_w: int | None = None,
    canvas_h: int | None = None,
    seed: int | None = None,
    schedule: str | None = None,
    rerank_schedule: str | None = None,
    eval_schedule: str | None = None,
    continue_output_dir: str | None = None,
    branch_from_output: str | None = None,
    branch_from_checkpoint: str | None = None,
    override: str | None = None,
    pop: int | None = None,
    gens: int | None = None,
    stop_after_gens: int | None = None,
    env_steps: int | None = None,
    eval_env_steps: int | None = None,
    phys_substeps: int | None = None,
    gpu: str = "preset",
    spawn: bool = False,
):
    continuation_sources = [
        value
        for value in (continue_output_dir, branch_from_output, branch_from_checkpoint)
        if value is not None
    ]
    if len(continuation_sources) > 1:
        raise SystemExit(
            "Specify at most one of --continue-output-dir, --branch-from-output, "
            "or --branch-from-checkpoint."
        )
    if continue_output_dir is not None and gens is not None:
        raise SystemExit(
            "--continue-output-dir keeps the canonical train.n_generations. "
            "Use --stop-after-gens to choose the stage boundary."
        )

    resolved_preset = _resolve_preset_name(preset=preset, run=run)
    explicit_overrides = parse_override_csv(override)
    gpu_token, gpu_target = _resolve_gpu_target(
        preset=resolved_preset,
        overrides=_parse_cli_overrides(seed, None) + explicit_overrides,
        gpu=gpu,
    )
    gpu_profile_id = str(gpu_target["profile_id"])
    overrides = _parse_cli_overrides(seed, None)
    overrides = _append_run_surface_overrides(
        overrides,
        run=resolved_preset if run is not None else None,
        gpu_profile=gpu_profile_id,
        pop=pop,
        gens=gens,
        stop_after_gens=stop_after_gens,
        env_steps=env_steps,
        eval_env_steps=eval_env_steps,
        phys_substeps=phys_substeps,
    )
    overrides = _append_body_surface_overrides(
        overrides,
        body=body,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
    )
    overrides = _append_terrain_surface_overrides(
        overrides,
        terrain=terrain,
        body=body,
        canvas_w=canvas_w,
        curriculum_phase_ends=curriculum_phase_ends,
    )
    overrides.extend(explicit_overrides)
    explicit_override_keys = {token.split("=", 1)[0] for token in overrides if "=" in token}
    if not continuation_sources and "train.resume_mode" not in explicit_override_keys:
        overrides.append("train.resume_mode=fresh")
    overrides = _force_gpu_profile(overrides, gpu_profile=gpu_profile_id)
    overrides = _force_wandb_enabled(overrides)

    if spawn:
        raise SystemExit(
            "The `--spawn` mode is not supported for this entrypoint. "
            "Use `modal run --detach scripts/modal_train.py ...` so the training app "
            "stays alive after the local entrypoint exits."
        )
    if gpu_token == "l4":
        remote = train_remote_l4
    elif gpu_token == "a100-40gb":
        remote = train_remote_a100_40gb
    elif gpu_token == "a100-80gb":
        remote = train_remote_a100_80gb
    elif gpu_token == "h100":
        remote = train_remote_h100
    elif gpu_token == "h200":
        remote = train_remote_h200
    elif gpu_token == "b200":
        remote = train_remote_b200
    else:
        raise AssertionError(f"Unhandled GPU target: {gpu_token}")
    result = remote.remote(
        preset=resolved_preset,
        overrides=overrides,
        schedule=schedule,
        rerank_schedule=rerank_schedule,
        eval_schedule=eval_schedule,
        continue_output_dir=continue_output_dir,
        branch_from_output=branch_from_output,
        branch_from_checkpoint=branch_from_checkpoint,
    )
    print(f"Result: {result}")
