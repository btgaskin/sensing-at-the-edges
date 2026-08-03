"""Shared helpers for composing, validating, and deriving runtime parameters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from .collision_tuning import ResolvedCollisionTuning
from .manifest import config_sha256
from .nca import (
    compatible_signaling_variants,
    controller_flat_dim,
    normalize_active_comm_filters,
    validate_signaling_variant_compatibility,
)
from .qd_descriptors import resolve_qd_archive
from .qd.focus import focus_config_from_section, validate_focus_config
from .terrain import make_robot_body, max_canvas_size

_TASK_MODE_BILATERAL_GOAL_CLIFF = "bilateral_goal_cliff"
_NOISE_TYPE_OFF = "off"
_NOISE_TYPE_OU_FIXED = "ou_fixed"
_NOISE_TYPE_OU_EVOLVED = "ou_evolved"
_NOISE_TYPES = frozenset(
    {
        _NOISE_TYPE_OFF,
        _NOISE_TYPE_OU_FIXED,
        _NOISE_TYPE_OU_EVOLVED,
    }
)
LOGICAL_IDENTITY_EXCLUDED_PATHS = frozenset(
    {
        "train.output_dir",
        "train.resume_from",
        "train.stop_after_generations",
        "train.resume_mode",
    }
)


def _auto_scale_deformation_gain(
    gain: float, *, spring_stiffness_scale: float, enabled: bool,
) -> float:
    if not enabled or spring_stiffness_scale <= 0.0:
        return gain
    return gain / spring_stiffness_scale


PAPER_PRESET_ALLOWED_KEYS = frozenset(
    {
        "preset_id",
        "notes",
        "tags",
        "terrain",
        "batching",
        "collision.strategy",
        "collision.dynamic_impl",
        "nca.ctrnn_hidden",
        "nca.ctrnn_grid_hidden",
        "nca.active_comm_filters",
        "nca.ablate_deformation",
        "nca.noise",
        "nca.deformation_gain_absolute",
        "nca.deformation_gain_compliance",
        "nca.deformation_gain_auto_scale",
        "nca.p_update",
        "sim.physics_substeps",
        "sim.actuator_convergence",
        "sim.spring_stiffness_scale",
        "sim.floor_static_friction_const",
        "sim.floor_stiction_stiffness",
        "sim.floor_stiction_switch_speed",
        "sim.airborne_kill_height_margin_cells",
        "sim.soft_loft_threshold_cells",
        "train.max_steps",
        "train.activation_cost_penalty_scale",
        "train.soft_loft_penalty_scale",
        "train.gait_spectral_purity_bonus_scale",
        "train.require_schedule_dir",
        "eval",
        "qd.descriptors.names",
        "qd.emitter_strategy",
        "qd.archive.dims",
        "qd.archive.ranges",
        "qd.archive.learning_rate",
        "qd.focus.enabled",
        "qd.focus.tighten_every_n_gens",
        "qd.focus.min_occupied_to_tighten",
        "qd.focus.min_window_size",
        "qd.focus.targets",
        "qd.focus.fallback_to_all_if_active_empty",
        "render.target_frames",
        "render.trace_max_frames",
        "render.parallel_workers",
        "render.periodic.temporal_mode",
        "render.periodic.terrain_selection",
        "render.periodic.width",
        "render.periodic.height",
        "render.periodic.fps",
        "render.periodic.target_frames",
        "render.post_train.temporal_mode",
        "render.post_train.target_frames",
        "audit",
        "specimens",
    }
)
PAPER_RUNTIME_ALLOWED_KEYS = frozenset(
    {
        "seed",
        "body.mode",
        "body.shape",
        "body.growth.max_robot_h",
        "body.growth.max_robot_w",
        "train.output_dir",
        "train.resume_from",
        "train.stop_after_generations",
        "train.resume_mode",
        "train.terrain_schedule_dir",
        "train.require_schedule_dir",
        "train.n_generations",
        "train.n_episodes",
        "train.max_steps",
        "train.activation_cost_penalty_scale",
        "train.soft_loft_penalty_scale",
        "train.gait_spectral_purity_bonus_scale",
        "train.checkpoint_interval",
        "nca.p_update",
        "nca.active_comm_filters",
        "nca.noise",
        "qd.n_emitters",
        "qd.emitter_pop_size",
        "qd.emitter_strategy",
        "qd.initial_sigma",
        "qd.descriptors.names",
        "qd.archive.dims",
        "qd.archive.ranges",
        "qd.archive.learning_rate",
        "qd.focus.enabled",
        "qd.focus.tighten_every_n_gens",
        "qd.focus.min_occupied_to_tighten",
        "qd.focus.min_window_size",
        "qd.focus.targets",
        "qd.focus.fallback_to_all_if_active_empty",
        "sim.physics_substeps",
        "sim.actuator_convergence",
        "sim.spring_stiffness_scale",
        "sim.floor_static_friction_const",
        "sim.floor_stiction_stiffness",
        "sim.floor_stiction_switch_speed",
        "sim.airborne_kill_height_margin_cells",
        "sim.soft_loft_threshold_cells",
        "sim.sense_act_phases_per_env_step",
        "nca.deformation_gain_absolute",
        "nca.deformation_gain_compliance",
        "nca.deformation_gain_auto_scale",
        "eval.eval_episodes",
        "eval.eval_individuals",
        "eval.rerank_episodes",
        "eval.test_episodes",
        "eval.rerank_max_steps",
        "eval.test_max_steps",
        "eval.rerank_schedule_dir",
        "eval.test_schedule_dir",
        "eval.robustness_probe",
        "eval.deformation_suite",
        "eval.post_run",
        "collision.strategy",
        "collision.dynamic_max_candidates",
        "collision.dynamic_bin_size",
        "collision.dynamic_impl",
        "collision.dynamic_max_narrow",
        "wandb.group",
        "wandb.name",
        "wandb.enabled",
        "render",
        "render.render_evals",
        "render.target_frames",
        "render.trace_max_frames",
        "render.parallel_workers",
        "render.periodic.width",
        "render.periodic.height",
        "render.periodic.fps",
        "audit",
        "specimens",
        "gpu_profile",
        "terrain.robot_h",
        "terrain.robot_w",
        "batching.generation.candidate_chunk_size",
        "batching.generation.candidate_bucket_sizes",
        "batching.generation.exact_fit_small_run_threshold",
        "batching.generation.warmup_enabled",
        "batching.validation.candidate_chunk_size",
        "batching.validation.candidate_bucket_sizes",
        "batching.validation.exact_fit_small_run_threshold",
        "batching.validation.warmup_enabled",
        "batching.test.candidate_chunk_size",
        "batching.test.candidate_bucket_sizes",
        "batching.test.exact_fit_small_run_threshold",
        "batching.test.warmup_enabled",
        "batching.trace.candidate_chunk_size",
        "batching.trace.candidate_bucket_sizes",
        "batching.trace.exact_fit_small_run_threshold",
        "batching.trace.warmup_enabled",
    }
)
REQUIRED_TOP_LEVEL_KEYS = (
    "paper_core_id",
    "seed",
    "body",
    "train",
    "eval",
    "batching",
    "gpu_profile",
    "nca",
    "sim",
    "qd",
    "terrain",
    "collision",
    "logging",
    "wandb",
    "render",
    "audit",
    "specimens",
)
WORKLOAD_NAMES = ("generation", "validation", "test", "trace")

_POST_RUN_SIGNALING_CONDITIONS = frozenset(
    {
        "intact",
        "sobel_x_off",
        "sobel_y_off",
        "laplacian_off",
        "identity_comm_off",
        "all_comm_off",
        "sobel_x_sign_flip",
    }
)
_POST_RUN_DEFORMATION_CONDITIONS = frozenset(
    {
        "intact",
        "clean",
        "absolute_only",
        "compliance_only",
        "deformation_off",
        "noise_only",
        "deformation_reversed",
        "zero",
        "noisy",
        "shuffled",
    }
)
_POST_RUN_SENSING_FREEZE_MODES = frozenset(
    {
        "none",
        "deformation_frozen_start",
        "perception_frozen_start",
    }
)


@dataclass(frozen=True)
class WorkloadBatching:
    workload: str
    candidate_chunk_size: int
    candidate_bucket_sizes: tuple[int, ...]
    exact_fit_small_run_threshold: int
    warmup_enabled: bool
    memory_budget_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_chunk_size": self.candidate_chunk_size,
            "candidate_bucket_sizes": list(self.candidate_bucket_sizes),
            "exact_fit_small_run_threshold": self.exact_fit_small_run_threshold,
            "warmup_enabled": self.warmup_enabled,
            "memory_budget_bytes": self.memory_budget_bytes,
        }


@dataclass(frozen=True)
class RuntimeResolution:
    preset_name: str | None
    preset_changed_paths: tuple[str, ...]
    runtime_override_keys: tuple[str, ...]
    paper_core_id: str
    paper_core_sha256: str
    gpu_profile: str
    resolved_batching: dict[str, WorkloadBatching]
    resolved_collision: ResolvedCollisionTuning | None = None

    def batching_manifest(self) -> dict[str, dict[str, Any]]:
        return {
            workload: resolution.as_dict()
            for workload, resolution in self.resolved_batching.items()
        }

    def collision_manifest(self) -> dict[str, Any] | None:
        if self.resolved_collision is None:
            return None
        return self.resolved_collision.as_dict()


def default_config_dir() -> Path:
    """Return the sensing_edges Hydra config directory."""
    return Path(__file__).resolve().parent.parent / "config"


def normalize_preset_name(preset: str | None) -> str | None:
    """Normalize preset references to the canonical preset group key."""
    if preset is None:
        return None
    value = str(preset).strip()
    if value in ("", "config", "base", "default"):
        return None
    if "/" in value:
        raise ValueError(f"Unsupported preset reference: {preset}")
    return value


def parse_override_keys(overrides: list[str] | None) -> tuple[str, ...]:
    """Extract logical config paths from Hydra override strings."""
    parsed: list[str] = []
    for override in overrides or []:
        token = override.strip()
        if not token:
            continue
        token = token.lstrip("+~")
        key = token.split("=", 1)[0].strip()
        if key:
            parsed.append(key)
    return tuple(parsed)


def compose_experiment_config(
    *,
    preset: str | None = None,
    overrides: list[str] | None = None,
    config_dir: str | Path | None = None,
) -> DictConfig:
    """Compose the canonical sensing_edges config as base -> preset -> args."""
    compose_overrides = list(overrides or [])
    normalized_preset = normalize_preset_name(preset)
    if normalized_preset is not None:
        compose_overrides.insert(0, f"preset={normalized_preset}")

    resolved_dir = str(config_dir or default_config_dir())
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=resolved_dir, version_base=None):
        return compose(config_name="config", overrides=compose_overrides)


def merge_stored_config(
    base_cfg: DictConfig,
    stored_cfg: DictConfig,
    *,
    prefer_current_post_run: bool = True,
) -> DictConfig:
    """Merge a stored run config while keeping current post-run defaults authoritative.

    Stored configs should continue to drive run-specific morphology/controller details.
    Post-run assay definitions, however, should come from the current preset so the
    active assay battery is consistent across archived specimens.
    """
    merged_base = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
    merged_stored = OmegaConf.create(OmegaConf.to_container(stored_cfg, resolve=False))
    OmegaConf.set_struct(merged_base, False)
    OmegaConf.set_struct(merged_stored, False)

    merged_cfg = OmegaConf.merge(merged_base, merged_stored)
    if not prefer_current_post_run:
        return merged_cfg

    base_eval = merged_base.get("eval")
    base_post_run = None if base_eval is None else base_eval.get("post_run")
    if base_post_run is None:
        return merged_cfg

    stored_eval = merged_stored.get("eval")
    stored_post_run = None if stored_eval is None else stored_eval.get("post_run")
    merged_post_run = OmegaConf.merge(
        OmegaConf.create(
            {} if stored_post_run is None else OmegaConf.to_container(stored_post_run, resolve=False)
        ),
        OmegaConf.create(OmegaConf.to_container(base_post_run, resolve=False)),
    )
    if merged_cfg.get("eval") is None:
        merged_cfg.eval = OmegaConf.create({})
    merged_cfg.eval.post_run = merged_post_run
    return merged_cfg


def compose_stored_config(
    resolved_config_path: str | Path,
    *,
    preset: str | None = None,
    prefer_current_post_run: bool = True,
) -> DictConfig:
    """Compose a stored run config against the current preset defaults."""
    stored_cfg = OmegaConf.load(resolved_config_path)
    base_preset = str(stored_cfg.get("preset_id") or preset or "config")
    base_cfg = compose_experiment_config(preset=base_preset)
    return merge_stored_config(
        base_cfg,
        stored_cfg,
        prefer_current_post_run=prefer_current_post_run,
    )


def logical_identity_config(cfg: DictConfig) -> dict[str, Any]:
    """Return the resolved config payload used for logical run identity."""
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    return _prune_excluded_paths(
        resolved,
        excluded_paths=LOGICAL_IDENTITY_EXCLUDED_PATHS,
    )


def logical_config_sha256(cfg: DictConfig) -> str:
    """SHA256 hash of the logical config with execution-only paths removed."""
    logical_cfg = OmegaConf.create(logical_identity_config(cfg))
    return config_sha256(OmegaConf.to_yaml(logical_cfg, resolve=True))


def controller_size_label(cfg: DictConfig) -> str:
    """Return a coarse controller size label used in logs and metadata."""
    if int(cfg.nca.ctrnn_hidden) == 12 and int(cfg.nca.ctrnn_grid_hidden) == 6:
        return "sbt"
    if int(cfg.nca.ctrnn_hidden) == 8 and int(cfg.nca.ctrnn_grid_hidden) == 12:
        return "single"
    return f"h{int(cfg.nca.ctrnn_hidden)}_gh{int(cfg.nca.ctrnn_grid_hidden)}"


def deformation_label(cfg: DictConfig) -> str:
    """Return the deformation condition label used in logs and metadata."""
    return "off" if cfg.nca.ablate_deformation else "on"


def emitter_strategy_label(cfg: DictConfig) -> str:
    """Return the canonical EA strategy label used in run names and metadata."""
    return str(cfg.qd.emitter_strategy).strip().lower()


def body_variant_label(cfg: DictConfig) -> str:
    """Return the canonical body variant label used in run names."""
    body_mode = str(cfg.body.mode).strip().lower()
    if body_mode == "grown":
        return f"grown_{int(cfg.body.growth.max_robot_w)}x{int(cfg.body.growth.max_robot_h)}"

    shape = str(cfg.body.get("shape", "rectangle")).strip().lower()
    if shape in {"rectangle", "rect"}:
        return f"rect_{int(cfg.terrain.robot_w)}x{int(cfg.terrain.robot_h)}"
    return shape


def terrain_family_label(cfg: DictConfig) -> str:
    """Return a coarse terrain family label used in run names."""
    variant = str(cfg.terrain.variant).strip().lower()
    if "ledge" in variant:
        return "ledge"
    if "gap" in variant:
        return "gaps"
    if "cliff" in variant:
        return "cliff"

    if variant.startswith("curriculum_"):
        variant = variant[len("curriculum_") :]
    family = variant.split("_", 1)[0]
    return family or "terrain"


def resolved_noise_settings(cfg: DictConfig) -> dict[str, Any]:
    """Resolve the canonical controller-noise settings from nested or legacy config."""
    noise_cfg = cfg.nca.get("noise")
    if noise_cfg is not None:
        noise_type = str(noise_cfg.get("noise_type", _NOISE_TYPE_OU_FIXED)).strip().lower()
        ou_theta = float(noise_cfg.get("ou_theta", cfg.nca.get("ou_theta", 0.1)))
        ou_sigma = float(noise_cfg.get("ou_sigma", cfg.nca.get("ou_sigma", 0.3)))
    else:
        noise_type = (
            _NOISE_TYPE_OU_FIXED
            if bool(cfg.nca.get("ou_enabled", True))
            else _NOISE_TYPE_OFF
        )
        ou_theta = float(cfg.nca.get("ou_theta", 0.1))
        ou_sigma = float(cfg.nca.get("ou_sigma", 0.3))
    return {
        "noise_type": noise_type,
        "ou_theta": ou_theta,
        "ou_sigma": ou_sigma,
        "ou_enabled": noise_type != _NOISE_TYPE_OFF,
    }


def noise_enabled_from_cfg(cfg: DictConfig) -> bool:
    """Return whether the resolved controller-noise channel is active."""
    return bool(resolved_noise_settings(cfg)["ou_enabled"])


def active_comm_filters_from_cfg(cfg: DictConfig) -> tuple[str, ...]:
    """Return the normalized active communication filter bank for the controller."""
    return normalize_active_comm_filters(cfg.nca.get("active_comm_filters"))


def compatible_post_run_signaling_conditions(cfg: DictConfig) -> tuple[str, ...]:
    """Return post-run signaling conditions compatible with the run's filter bank."""
    post_run_cfg = cfg.eval.get("post_run")
    if post_run_cfg is None:
        return ()
    signaling_conditions = [
        str(condition).strip().lower()
        for condition in list(post_run_cfg.get("signaling_conditions") or [])
    ]
    return compatible_signaling_variants(
        signaling_conditions,
        active_comm_filters_from_cfg(cfg),
    )


def resolved_spawn_x(cfg: DictConfig) -> int:
    """Return the effective spawn_x after any task-specific geometry rules."""
    task_mode = str(cfg.terrain.get("task_mode", "")).strip().lower()
    if task_mode != _TASK_MODE_BILATERAL_GOAL_CLIFF:
        return int(cfg.terrain.spawn_x)

    plateau_width = int(cfg.terrain.get("center_plateau_width", cfg.terrain.get("base_width", 10)))
    grid_w = int(cfg.terrain.grid_w)
    plateau_start = (grid_w - plateau_width) // 2
    body_mode = str(cfg.body.mode).strip().lower()
    robot_w = (
        int(cfg.body.growth.max_robot_w)
        if body_mode == "grown"
        else int(cfg.terrain.robot_w)
    )
    if robot_w <= 0:
        raise ValueError("Resolved bilateral spawn requires a positive robot width.")
    if robot_w > plateau_width:
        raise ValueError(
            "Resolved bilateral spawn requires the body to fit within the centered plateau: "
            f"robot_w={robot_w}, plateau_width={plateau_width}"
        )
    centered_spawn_x = int(plateau_start + (plateau_width - robot_w) // 2)
    offset = int(cfg.terrain.get("bilateral_spawn_x_offset", 0))
    min_spawn_x = int(plateau_start)
    max_spawn_x = int(plateau_start + plateau_width - robot_w)
    return max(min_spawn_x, min(centered_spawn_x + offset, max_spawn_x))


def resolved_rerank_max_steps(cfg: DictConfig) -> int:
    """Return the effective rerank episode budget in env steps."""
    value = cfg.eval.get("rerank_max_steps")
    return int(value) if value is not None else int(cfg.train.max_steps)


def resolved_test_max_steps(cfg: DictConfig) -> int:
    """Return the effective held-out test episode budget in env steps."""
    value = cfg.eval.get("test_max_steps")
    return int(value) if value is not None else int(cfg.train.max_steps)


def reference_convergence_label(cfg: DictConfig) -> str:
    """Return a compact tag-friendly label for the reference actuator convergence."""
    decimals = f"{float(cfg.sim.actuator_convergence):.4f}".split(".", 1)[1].rstrip("0")
    if not decimals:
        decimals = "0"
    if len(decimals) < 3:
        decimals = decimals.ljust(3, "0")
    return f"ref{decimals}"


def eval_budget_label(cfg: DictConfig) -> str:
    """Return the canonical eval-step budget label for run naming."""
    rerank_steps = resolved_rerank_max_steps(cfg)
    test_steps = resolved_test_max_steps(cfg)
    if rerank_steps == test_steps:
        return f"e{rerank_steps}"
    return f"er{rerank_steps}_et{test_steps}"


def run_group_label(cfg: DictConfig) -> str:
    """Return a human-readable condition label for grouping related runs."""
    parts = [
        terrain_family_label(cfg),
        body_variant_label(cfg),
        emitter_strategy_label(cfg),
        f"g{int(cfg.train.n_generations)}",
        f"p{int(total_population(cfg))}",
        f"{int(cfg.sim.physics_substeps)}sub",
        f"t{int(cfg.train.max_steps)}",
        eval_budget_label(cfg),
        reference_convergence_label(cfg),
    ]
    if deformation_label(cfg) == "off":
        parts.append("nodeform")
    return "_".join(parts)


def run_name_label(cfg: DictConfig) -> str:
    """Return the canonical run stem used for output dirs and logging."""
    return f"{run_group_label(cfg)}_seed{int(cfg.seed)}"


def run_metadata(cfg: DictConfig) -> dict[str, Any]:
    """Return condition metadata used across manifests, evals, and analysis."""
    train_variant = str(cfg.terrain.variant)
    rerank_variant = str(cfg.eval.get("rerank_variant") or train_variant)
    test_variant = str(cfg.eval.get("test_variant") or rerank_variant)
    resolved_qd = resolve_qd_archive(cfg)
    noise_settings = resolved_noise_settings(cfg)
    return {
        "preset_id": str(cfg.get("preset_id") or "config"),
        "seed": int(cfg.seed),
        "body_mode": str(cfg.body.mode),
        "body_variant": body_variant_label(cfg),
        "emitter_strategy": emitter_strategy_label(cfg),
        "terrain_family": terrain_family_label(cfg),
        "group_label": run_group_label(cfg),
        "run_label": run_name_label(cfg),
        "controller_size": controller_size_label(cfg),
        "active_comm_filters": list(active_comm_filters_from_cfg(cfg)),
        "deformation_condition": deformation_label(cfg),
        "noise_type": str(noise_settings["noise_type"]),
        "physics_substeps": int(cfg.sim.physics_substeps),
        "n_generations": int(cfg.train.n_generations),
        "total_population": int(total_population(cfg)),
        "gpu_profile": str(cfg.gpu_profile.id),
        "reference_actuator_convergence": float(cfg.sim.actuator_convergence),
        "reference_convergence_label": reference_convergence_label(cfg),
        "train_max_steps": int(cfg.train.max_steps),
        "rerank_max_steps": int(resolved_rerank_max_steps(cfg)),
        "test_max_steps": int(resolved_test_max_steps(cfg)),
        "train_variant": train_variant,
        "rerank_variant": rerank_variant,
        "test_variant": test_variant,
        "qd_descriptor_names": list(resolved_qd.descriptor_names),
        "qd_descriptor_labels": list(resolved_qd.descriptor_labels),
    }


def total_population(cfg: DictConfig) -> int:
    """Return the total number of candidates evaluated per generation."""
    return int(cfg.qd.n_emitters) * int(cfg.qd.emitter_pop_size)


def genotype_dim_from_cfg(cfg: DictConfig) -> int:
    """Return the flattened controller genotype length for the config."""
    growth_enabled = str(cfg.body.mode).strip().lower() == "grown"
    noise_settings = resolved_noise_settings(cfg)
    return controller_flat_dim(
        ctrnn_hidden=cfg.nca.ctrnn_hidden,
        ctrnn_grid_hidden=cfg.nca.ctrnn_grid_hidden,
        gasnet_mode=cfg.nca.get("gasnet_mode", "off"),
        ou_enabled=bool(noise_settings["ou_enabled"]),
        growth_enabled=growth_enabled,
        active_comm_filters=active_comm_filters_from_cfg(cfg),
    )


def emitter_kwargs_from_cfg(cfg: DictConfig) -> dict:
    """Resolve CMA emitter kwargs from config."""
    param_overrides = None
    initial_sigma = cfg.qd.get("initial_sigma", None)
    if initial_sigma is not None:
        param_overrides = {"std_init": float(initial_sigma)}

    return {
        "population_size": int(cfg.qd.emitter_pop_size),
        "strategy": cfg.qd.emitter_strategy,
        "flat_generations": int(cfg.qd.get("emitter_flat_generations", 3)),
        "flat_tolerance": float(cfg.qd.get("emitter_flat_tolerance", 1.0e-12)),
        "stale_generations": int(
            cfg.qd.get(
                "emitter_restart_stale_generations",
                cfg.qd.get("emitter_flat_generations", 5),
            )
        ),
        "max_age": int(cfg.qd.get("emitter_restart_max_age", 50)),
        "meaningful_improvement_eps": float(
            cfg.qd.get("emitter_restart_improvement_eps", 0.5)
        ),
        "std_epsilon": float(cfg.qd.get("emitter_std_epsilon", 1.0e-12)),
        "condition_cov_limit": float(
            cfg.qd.get("emitter_condition_cov_limit", 1.0e14)
        ),
        "param_overrides": param_overrides,
    }


def _normalize_distance_pair_list(raw_pairs: Any, *, field_name: str) -> list[tuple[int, int]]:
    pairs = list(raw_pairs or [])
    if not pairs:
        raise ValueError(f"{field_name} must be non-empty")
    normalized: list[tuple[int, int]] = []
    for pair in pairs:
        if hasattr(pair, "keys"):
            goal_distance = int(pair.get("goal_distance"))
            cliff_distance = int(pair.get("cliff_distance"))
        else:
            if len(pair) != 2:
                raise ValueError(f"{field_name} entries must have exactly two values")
            goal_distance = int(pair[0])
            cliff_distance = int(pair[1])
        if goal_distance < 0 or cliff_distance < 0:
            raise ValueError(f"{field_name} distances must be non-negative")
        normalized.append((goal_distance, cliff_distance))
    return normalized


def rollout_kwargs_from_cfg(
    cfg: DictConfig,
    *,
    collision_tuning: ResolvedCollisionTuning | None = None,
) -> dict:
    """Extract NCA and physics kwargs for run_episode."""
    growth_enabled = str(cfg.body.mode).strip().lower() == "grown"
    deformation_gain_auto_scale = bool(cfg.nca.get("deformation_gain_auto_scale", False))
    spring_stiffness_scale = float(cfg.sim.spring_stiffness_scale)
    spawn_x = resolved_spawn_x(cfg)
    noise_settings = resolved_noise_settings(cfg)
    active_comm_filters = active_comm_filters_from_cfg(cfg)
    post_run_cfg = cfg.eval.get("post_run", {})
    active_deformation_variant = str(
        post_run_cfg.get(
            "active_deformation_condition",
            cfg.eval.get("deformation_suite", {}).get("active_variant", "clean"),
        )
    )
    active_signaling_variant = str(
        post_run_cfg.get("active_signaling_condition", "intact")
    ).strip().lower()
    validate_signaling_variant_compatibility(
        active_signaling_variant,
        active_comm_filters,
        field_name="eval.post_run.active_signaling_condition",
    )
    active_sensing_freeze_mode = str(
        post_run_cfg.get("active_sensing_freeze_mode", "none")
    ).strip().lower()
    if active_sensing_freeze_mode not in _POST_RUN_SENSING_FREEZE_MODES:
        raise ValueError(
            "eval.post_run.active_sensing_freeze_mode must be one of "
            f"{sorted(_POST_RUN_SENSING_FREEZE_MODES)}"
        )
    return dict(
        max_steps=cfg.train.max_steps,
        physics_substeps=cfg.sim.physics_substeps,
        sense_act_phases=cfg.sim.sense_act_phases_per_env_step,
        grid_h=cfg.terrain.grid_h,
        grid_w=cfg.terrain.grid_w,
        spawn_x=spawn_x,
        spawn_y=cfg.terrain.spawn_y,
        robot_h=None if growth_enabled else cfg.terrain.robot_h,
        robot_w=None if growth_enabled else cfg.terrain.robot_w,
        fixed_hidden_init_scale=float(
            cfg.body.get("fixed", {}).get("hidden_init_scale", 0.0)
        ),
        nca_steps_per_phase=cfg.nca.nca_steps_per_phase,
        ctrnn_hidden=cfg.nca.ctrnn_hidden,
        ctrnn_grid_hidden=cfg.nca.ctrnn_grid_hidden,
        ctrnn_dt=cfg.nca.ctrnn_dt,
        ctrnn_tau_min=cfg.nca.ctrnn_tau_min,
        ctrnn_tau_max=cfg.nca.ctrnn_tau_max,
        p_update=cfg.nca.p_update,
        active_comm_filters=active_comm_filters,
        ou_theta=float(noise_settings["ou_theta"]),
        ou_sigma=float(noise_settings["ou_sigma"]),
        ou_enabled=bool(noise_settings["ou_enabled"]),
        ablate_deformation=cfg.nca.ablate_deformation,
        deformation_variant=active_deformation_variant,
        deformation_noise_sigma_signed_delta=float(
            cfg.eval.get("deformation_suite", {}).get("noise_sigma_signed_delta", 0.05)
        ),
        deformation_shuffle_mode=str(
            cfg.eval.get("deformation_suite", {}).get("shuffle_mode", "episode_fixed")
        ),
        signaling_variant=active_signaling_variant,
        sensing_freeze_mode=active_sensing_freeze_mode,
        gasnet_mode=cfg.nca.get("gasnet_mode", "off"),
        gasnet_type=cfg.nca.get("gasnet_type", "gain"),
        gas_decay_rate=cfg.nca.get("gas_decay_rate", 0.95),
        gas_diffusion_radius=cfg.nca.get("gas_diffusion_radius", 3.0),
        gas_gain_sensitivity=cfg.nca.get("gas_gain_sensitivity", 2.0),
        gas_max_concentration=cfg.nca.get("gas_max_concentration", 1.0),
        gas_gain_min=cfg.nca.get("gas_gain_min", -4.0),
        gas_gain_max=cfg.nca.get("gas_gain_max", 4.0),
        update_mode=cfg.nca.update_mode,
        state_clip_value=cfg.nca.state_clip_value,
        mace_mode=cfg.nca.get("mace_mode", "cross_channel"),
        mace_beta=cfg.nca.get("mace_beta", 1.0),
        mace_init_mass=cfg.nca.get("mace_init_mass", 1.0),
        deformation_gain_absolute=_auto_scale_deformation_gain(
            float(cfg.nca.get("deformation_gain_absolute", 0.0)),
            spring_stiffness_scale=spring_stiffness_scale,
            enabled=deformation_gain_auto_scale,
        ),
        deformation_gain_compliance=_auto_scale_deformation_gain(
            float(cfg.nca.get("deformation_gain_compliance", 0.0)),
            spring_stiffness_scale=spring_stiffness_scale,
            enabled=deformation_gain_auto_scale,
        ),
        airborne_kill_height_margin_cells=float(
            cfg.sim.get("airborne_kill_height_margin_cells", 2.0)
        ),
        soft_loft_threshold_cells=float(
            cfg.sim.get("soft_loft_threshold_cells", 0.0)
        ),
        collision_strategy=str(cfg.collision.strategy),
        dynamic_max_candidates=(
            int(collision_tuning.dynamic_max_candidates)
            if collision_tuning is not None
            else int(cfg.collision.dynamic_max_candidates)
        ),
        dynamic_bin_size=float(cfg.collision.dynamic_bin_size),
        dynamic_impl=str(cfg.collision.dynamic_impl),
        dynamic_max_narrow=(
            int(collision_tuning.dynamic_max_narrow)
            if collision_tuning is not None
            else _resolve_dynamic_max_narrow(cfg)
        ),
        growth_enabled=growth_enabled,
    )


def resolve_runtime_config(
    cfg: DictConfig,
    *,
    preset: str | None = None,
    runtime_overrides: list[str] | None = None,
    config_dir: str | Path | None = None,
) -> RuntimeResolution:
    """Resolve runtime metadata and GPU-profile-backed batching for execution."""
    resolved_dir = config_dir or default_config_dir()
    normalized_preset = normalize_preset_name(preset or cfg.get("preset_id"))
    runtime_override_keys = parse_override_keys(runtime_overrides)

    _validate_required_sections(cfg)
    _validate_rollout_config(cfg)
    resolve_qd_archive(cfg)
    _validate_qd_focus_config(cfg)

    base_cfg = compose_experiment_config(config_dir=resolved_dir)
    paper_core_sha = config_sha256(OmegaConf.to_yaml(base_cfg, resolve=True))
    preset_changed_paths: tuple[str, ...] = ()
    if normalized_preset is not None:
        preset_cfg = compose_experiment_config(
            preset=normalized_preset,
            config_dir=resolved_dir,
        )
        preset_changed_paths = tuple(sorted(_diff_paths(base_cfg, preset_cfg)))

    resolved_batching = resolve_batching(cfg)
    return RuntimeResolution(
        preset_name=normalized_preset,
        preset_changed_paths=preset_changed_paths,
        runtime_override_keys=runtime_override_keys,
        paper_core_id=str(cfg.paper_core_id),
        paper_core_sha256=paper_core_sha,
        gpu_profile=str(cfg.gpu_profile.id),
        resolved_batching=resolved_batching,
    )


def validate_paper_config(
    cfg: DictConfig,
    *,
    preset: str | None = None,
    runtime_overrides: list[str] | None = None,
    config_dir: str | Path | None = None,
) -> RuntimeResolution:
    """Resolve runtime metadata and enforce the paper-only config policy."""
    resolution = resolve_runtime_config(
        cfg,
        preset=preset,
        runtime_overrides=runtime_overrides,
        config_dir=config_dir,
    )
    _validate_runtime_overrides(resolution.runtime_override_keys)
    _validate_preset_diff(resolution.preset_changed_paths)
    return resolution


def attach_collision_tuning(
    runtime_resolution: RuntimeResolution,
    collision_tuning: ResolvedCollisionTuning,
) -> RuntimeResolution:
    """Return a runtime resolution annotated with resolved collision capacities."""
    return replace(runtime_resolution, resolved_collision=collision_tuning)


def resolve_batching(cfg: DictConfig) -> dict[str, WorkloadBatching]:
    """Resolve workload batching from explicit config plus GPU-profile defaults."""
    memory_budget_bytes = _memory_budget_bytes(cfg)
    resolved: dict[str, WorkloadBatching] = {}
    for workload in WORKLOAD_NAMES:
        workload_cfg = cfg.batching.get(workload)
        profile_cfg = cfg.gpu_profile.workloads.get(workload)
        if workload_cfg is None or profile_cfg is None:
            raise ValueError(f"Missing batching config for workload '{workload}'")

        candidate_chunk_size = workload_cfg.get("candidate_chunk_size")
        if candidate_chunk_size is None:
            candidate_chunk_size = profile_cfg.candidate_chunk_size
        candidate_chunk_size = int(candidate_chunk_size)
        if candidate_chunk_size <= 0:
            raise ValueError(f"batching.{workload}.candidate_chunk_size must be positive")

        bucket_sizes = workload_cfg.get("candidate_bucket_sizes")
        if not bucket_sizes:
            bucket_sizes = profile_cfg.candidate_bucket_sizes
        candidate_bucket_sizes = tuple(int(size) for size in bucket_sizes)
        if not candidate_bucket_sizes:
            raise ValueError(f"batching.{workload}.candidate_bucket_sizes must be non-empty")
        if any(size <= 0 for size in candidate_bucket_sizes):
            raise ValueError(f"batching.{workload}.candidate_bucket_sizes must be positive")
        if sorted(candidate_bucket_sizes) != list(candidate_bucket_sizes):
            raise ValueError(
                f"batching.{workload}.candidate_bucket_sizes must be sorted ascending"
            )
        if candidate_bucket_sizes[-1] > candidate_chunk_size:
            raise ValueError(
                f"batching.{workload}.candidate_bucket_sizes cannot exceed "
                f"candidate_chunk_size={candidate_chunk_size}"
            )

        explicit_exact_fit_threshold = workload_cfg.get("exact_fit_small_run_threshold")
        exact_fit_small_run_threshold = explicit_exact_fit_threshold
        if exact_fit_small_run_threshold is None:
            exact_fit_small_run_threshold = profile_cfg.get("exact_fit_small_run_threshold")
        if exact_fit_small_run_threshold is None:
            exact_fit_small_run_threshold = candidate_bucket_sizes[0]
        exact_fit_small_run_threshold = int(exact_fit_small_run_threshold)
        if exact_fit_small_run_threshold <= 0:
            raise ValueError(
                f"batching.{workload}.exact_fit_small_run_threshold must be positive"
            )
        if exact_fit_small_run_threshold > candidate_chunk_size:
            if explicit_exact_fit_threshold is None:
                exact_fit_small_run_threshold = candidate_chunk_size
            else:
                raise ValueError(
                    f"batching.{workload}.exact_fit_small_run_threshold={exact_fit_small_run_threshold} "
                    f"exceeds candidate_chunk_size={candidate_chunk_size}"
                )

        warmup_enabled = bool(workload_cfg.get("warmup_enabled", True))
        resolved[workload] = WorkloadBatching(
            workload=workload,
            candidate_chunk_size=candidate_chunk_size,
            candidate_bucket_sizes=candidate_bucket_sizes,
            exact_fit_small_run_threshold=exact_fit_small_run_threshold,
            warmup_enabled=warmup_enabled,
            memory_budget_bytes=memory_budget_bytes,
        )
    return resolved


def _validate_required_sections(cfg: DictConfig) -> None:
    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in cfg]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")


def _validate_rollout_config(cfg: DictConfig) -> None:
    if int(cfg.train.n_generations) <= 0:
        raise ValueError("train.n_generations must be positive")
    if int(cfg.sim.physics_substeps) % int(cfg.sim.sense_act_phases_per_env_step) != 0:
        raise ValueError(
            "sim.physics_substeps must be divisible by sim.sense_act_phases_per_env_step"
        )
    if int(cfg.train.max_steps) <= 0:
        raise ValueError("train.max_steps must be positive")
    if int(cfg.train.n_episodes) <= 0:
        raise ValueError("train.n_episodes must be positive")
    stop_after_generations = cfg.train.get("stop_after_generations")
    if stop_after_generations is not None:
        stop_after_generations = int(stop_after_generations)
        if stop_after_generations <= 0:
            raise ValueError("train.stop_after_generations must be positive when set")
        if stop_after_generations > int(cfg.train.n_generations):
            raise ValueError(
                "train.stop_after_generations cannot exceed train.n_generations"
            )
    resume_mode = str(cfg.train.get("resume_mode", "auto")).strip().lower()
    if resume_mode not in {"auto", "fresh", "staged", "branch"}:
        raise ValueError("train.resume_mode must be 'auto', 'fresh', 'staged', or 'branch'")
    if float(cfg.sim.get("airborne_kill_height_margin_cells", 2.0)) <= 0.0:
        raise ValueError("sim.airborne_kill_height_margin_cells must be positive")
    if float(cfg.sim.get("soft_loft_threshold_cells", 0.0)) < 0.0:
        raise ValueError("sim.soft_loft_threshold_cells must be nonnegative")
    if float(cfg.train.get("activation_cost_penalty_scale", 0.0)) < 0.0:
        raise ValueError("train.activation_cost_penalty_scale must be nonnegative")
    if float(cfg.train.get("soft_loft_penalty_scale", 0.0)) < 0.0:
        raise ValueError("train.soft_loft_penalty_scale must be nonnegative")
    if float(cfg.train.get("gait_spectral_purity_bonus_scale", 0.0)) < 0.0:
        raise ValueError("train.gait_spectral_purity_bonus_scale must be nonnegative")
    noise_settings = resolved_noise_settings(cfg)
    active_comm_filters = active_comm_filters_from_cfg(cfg)
    noise_type = str(noise_settings["noise_type"])
    if noise_type not in _NOISE_TYPES:
        raise ValueError(
            "nca.noise.noise_type must be one of "
            f"{sorted(_NOISE_TYPES)}"
        )
    if noise_type == _NOISE_TYPE_OU_EVOLVED:
        raise ValueError(
            "nca.noise.noise_type='ou_evolved' is reserved but not implemented yet"
        )
    if float(noise_settings["ou_theta"]) < 0.0:
        raise ValueError("nca.noise.ou_theta must be nonnegative")
    if float(noise_settings["ou_sigma"]) < 0.0:
        raise ValueError("nca.noise.ou_sigma must be nonnegative")
    eval_episodes = int(cfg.eval.eval_episodes)
    if eval_episodes <= 0:
        raise ValueError("eval.eval_episodes must be positive")
    for key in ("rerank_episodes", "test_episodes"):
        value = cfg.eval.get(key)
        if value is not None and int(value) <= 0:
            raise ValueError(f"eval.{key} must be positive when set")
    eval_individuals = cfg.eval.eval_individuals
    if str(eval_individuals).strip().lower() != "all" and int(eval_individuals) <= 0:
        raise ValueError("eval.eval_individuals must be positive or 'all'")
    for key in ("rerank_max_steps", "test_max_steps"):
        value = cfg.eval.get(key)
        if value is not None and int(value) <= 0:
            raise ValueError(f"eval.{key} must be positive when set")
    probe_cfg = cfg.eval.get("robustness_probe")
    if probe_cfg is not None:
        probe_mode = str(probe_cfg.get("mode", "symmetric_double_cliff")).strip().lower()
        if probe_mode != "symmetric_double_cliff":
            raise ValueError("eval.robustness_probe.mode must be 'symmetric_double_cliff'")
        for key in ("grid_h", "grid_w", "platform_y", "plateau_width", "min_distance", "max_distance"):
            value = probe_cfg.get(key)
            if value is None:
                continue
            if int(value) < 0:
                raise ValueError(f"eval.robustness_probe.{key} must be non-negative when set")
        if int(probe_cfg.get("grid_h", 0)) <= 0 or int(probe_cfg.get("grid_w", 0)) <= 0:
            raise ValueError("eval.robustness_probe.grid_h/grid_w must be positive")
        if int(probe_cfg.get("plateau_width", 0)) <= 0:
            raise ValueError("eval.robustness_probe.plateau_width must be positive")
        if int(probe_cfg.get("platform_y", 0)) >= int(probe_cfg.get("grid_h", 0)):
            raise ValueError("eval.robustness_probe.platform_y must be within the probe grid")
        min_distance = int(probe_cfg.get("min_distance", 1))
        max_distance = int(probe_cfg.get("max_distance", 6))
        if max_distance < min_distance:
            raise ValueError("eval.robustness_probe.max_distance must be >= min_distance")
        if int(probe_cfg.get("plateau_width", 0)) + 2 * max_distance > int(probe_cfg.get("grid_w", 0)):
            raise ValueError(
                "eval.robustness_probe plateau width plus symmetric distance sweep must fit within grid_w"
            )
        max_steps = probe_cfg.get("max_steps")
        if max_steps is not None and int(max_steps) <= 0:
            raise ValueError("eval.robustness_probe.max_steps must be positive when set")
        elite_count = int(probe_cfg.get("elite_count", 1))
        if elite_count <= 0:
            raise ValueError("eval.robustness_probe.elite_count must be positive")
    deformation_suite_cfg = cfg.eval.get("deformation_suite")
    if deformation_suite_cfg is not None:
        elite_count = int(deformation_suite_cfg.get("elite_count", 0))
        if elite_count <= 0:
            raise ValueError("eval.deformation_suite.elite_count must be positive")
        render_top_n = int(deformation_suite_cfg.get("render_top_n", 0))
        if render_top_n < 0:
            raise ValueError("eval.deformation_suite.render_top_n must be non-negative")
        if render_top_n > elite_count:
            raise ValueError(
                "eval.deformation_suite.render_top_n cannot exceed eval.deformation_suite.elite_count"
            )
        variants = list(deformation_suite_cfg.get("variants", []))
        valid_variants = {"clean", "zero", "noisy", "shuffled"}
        if not variants:
            raise ValueError("eval.deformation_suite.variants must be non-empty")
        for variant in variants:
            if str(variant) not in valid_variants:
                raise ValueError(
                    "eval.deformation_suite.variants must be drawn from "
                    f"{sorted(valid_variants)}"
                )
        if float(deformation_suite_cfg.get("noise_sigma_signed_delta", 0.0)) < 0.0:
            raise ValueError("eval.deformation_suite.noise_sigma_signed_delta must be nonnegative")
        shuffle_mode = str(deformation_suite_cfg.get("shuffle_mode", "episode_fixed")).strip().lower()
        if shuffle_mode != "episode_fixed":
            raise ValueError("eval.deformation_suite.shuffle_mode must be 'episode_fixed'")
    post_run_cfg = cfg.eval.get("post_run")
    if post_run_cfg is not None:
        auto_mode = str(post_run_cfg.get("auto_mode", "blocking")).strip().lower()
        if auto_mode not in {"blocking", "disabled", "off"}:
            raise ValueError("eval.post_run.auto_mode must be 'blocking', 'disabled', or 'off'")
        if int(post_run_cfg.get("selected_rank", 0)) < 0:
            raise ValueError("eval.post_run.selected_rank must be non-negative")
        if int(post_run_cfg.get("max_steps", 0)) <= 0:
            raise ValueError("eval.post_run.max_steps must be positive")
        if int(post_run_cfg.get("native_max_steps", post_run_cfg.get("max_steps", 0))) <= 0:
            raise ValueError("eval.post_run.native_max_steps must be positive")
        if int(post_run_cfg.get("double_cliff_max_steps", post_run_cfg.get("max_steps", 0))) <= 0:
            raise ValueError("eval.post_run.double_cliff_max_steps must be positive")
        if not str(post_run_cfg.get("analysis_root", "analyses")).strip():
            raise ValueError("eval.post_run.analysis_root must be non-empty")
        native_pairs = _normalize_distance_pair_list(
            post_run_cfg.get("native_pairs"),
            field_name="eval.post_run.native_pairs",
        )
        long_distance_pairs = _normalize_distance_pair_list(
            post_run_cfg.get("long_distance_pairs"),
            field_name="eval.post_run.long_distance_pairs",
        )
        pair_budget_cfg = list(post_run_cfg.get("long_distance_pair_max_steps") or [])
        if not pair_budget_cfg:
            raise ValueError("eval.post_run.long_distance_pair_max_steps must be non-empty")
        pair_budget_map: dict[tuple[int, int], int] = {}
        for index, budget in enumerate(pair_budget_cfg):
            goal_distance = int(budget.get("goal_distance", 0))
            cliff_distance = int(budget.get("cliff_distance", 0))
            max_steps = int(budget.get("max_steps", 0))
            if goal_distance <= 0 or cliff_distance <= 0:
                raise ValueError(
                    "eval.post_run.long_distance_pair_max_steps entries must have positive "
                    "goal_distance and cliff_distance"
                )
            if max_steps <= 0:
                raise ValueError(
                    "eval.post_run.long_distance_pair_max_steps entries must have positive max_steps"
                )
            pair_key = (goal_distance, cliff_distance)
            if pair_key in pair_budget_map:
                raise ValueError(
                    "eval.post_run.long_distance_pair_max_steps cannot repeat distance pairs"
                )
            pair_budget_map[pair_key] = max_steps
            _ = index
        missing_budgets = [pair for pair in long_distance_pairs if pair not in pair_budget_map]
        extra_budgets = [pair for pair in pair_budget_map if pair not in long_distance_pairs]
        if missing_budgets or extra_budgets:
            raise ValueError(
                "eval.post_run.long_distance_pair_max_steps must match "
                "eval.post_run.long_distance_pairs exactly"
            )
        if int(post_run_cfg.get("trials_per_side", 0)) <= 0:
            raise ValueError("eval.post_run.trials_per_side must be positive")
        double_cliff_cfg = post_run_cfg.get("double_cliff")
        if double_cliff_cfg is None:
            raise ValueError("eval.post_run.double_cliff must be configured")
        if int(double_cliff_cfg.get("plateau_width", 0)) <= 0:
            raise ValueError("eval.post_run.double_cliff.plateau_width must be positive")
        widths = [int(width) for width in list(double_cliff_cfg.get("widths") or [])]
        if not widths or any(width <= 0 for width in widths):
            raise ValueError("eval.post_run.double_cliff.widths must be non-empty positive integers")
        if int(double_cliff_cfg.get("trials_per_width", 0)) <= 0:
            raise ValueError("eval.post_run.double_cliff.trials_per_width must be positive")
        signaling_conditions = [
            str(condition).strip().lower()
            for condition in list(post_run_cfg.get("signaling_conditions") or [])
        ]
        if not signaling_conditions:
            raise ValueError("eval.post_run.signaling_conditions must be non-empty")
        invalid_signaling = [
            condition
            for condition in signaling_conditions
            if condition not in _POST_RUN_SIGNALING_CONDITIONS
        ]
        if invalid_signaling:
            raise ValueError(
                "eval.post_run.signaling_conditions must be drawn from "
                f"{sorted(_POST_RUN_SIGNALING_CONDITIONS)}"
            )
        compatible_signaling = compatible_signaling_variants(
            signaling_conditions,
            active_comm_filters,
        )
        if not compatible_signaling:
            raise ValueError(
                "eval.post_run.signaling_conditions has no variants compatible with "
                f"nca.active_comm_filters={active_comm_filters}"
            )
        active_signaling_condition = str(
            post_run_cfg.get("active_signaling_condition", "intact")
        ).strip().lower()
        validate_signaling_variant_compatibility(
            active_signaling_condition,
            active_comm_filters,
            field_name="eval.post_run.active_signaling_condition",
        )
        active_sensing_freeze_mode = str(
            post_run_cfg.get("active_sensing_freeze_mode", "none")
        ).strip().lower()
        if active_sensing_freeze_mode not in _POST_RUN_SENSING_FREEZE_MODES:
            raise ValueError(
                "eval.post_run.active_sensing_freeze_mode must be one of "
                f"{sorted(_POST_RUN_SENSING_FREEZE_MODES)}"
            )
        frozen_sensing_conditions = [
            str(condition).strip().lower()
            for condition in list(post_run_cfg.get("frozen_sensing_conditions") or [])
        ]
        invalid_frozen_sensing = [
            condition
            for condition in frozen_sensing_conditions
            if condition not in _POST_RUN_SENSING_FREEZE_MODES or condition == "none"
        ]
        if invalid_frozen_sensing:
            raise ValueError(
                "eval.post_run.frozen_sensing_conditions must be drawn from "
                f"{sorted(_POST_RUN_SENSING_FREEZE_MODES - {'none'})}"
            )
        deformation_conditions = [
            str(condition).strip().lower()
            for condition in list(post_run_cfg.get("deformation_conditions") or [])
        ]
        if not deformation_conditions:
            raise ValueError("eval.post_run.deformation_conditions must be non-empty")
        invalid_deformation = [
            condition
            for condition in deformation_conditions
            if condition not in _POST_RUN_DEFORMATION_CONDITIONS
        ]
        if invalid_deformation:
            raise ValueError(
                "eval.post_run.deformation_conditions must be drawn from "
                f"{sorted(_POST_RUN_DEFORMATION_CONDITIONS)}"
            )
        bodies = list(post_run_cfg.get("body_transfer_bodies") or [])
        if not bodies:
            raise ValueError("eval.post_run.body_transfer_bodies must be non-empty")
        for index, body in enumerate(bodies):
            label = str(body.get("label", "")).strip()
            if not label:
                raise ValueError(
                    f"eval.post_run.body_transfer_bodies[{index}].label must be non-empty"
                )
            if int(body.get("robot_w", 0)) <= 0 or int(body.get("robot_h", 0)) <= 0:
                raise ValueError(
                    f"eval.post_run.body_transfer_bodies[{index}].robot_w/robot_h must be positive"
                )
            make_robot_body(
                int(body.get("robot_w")),
                int(body.get("robot_h")),
                shape=str(body.get("shape", "rectangle")),
            )
        trace_cfg = post_run_cfg.get("trace")
        if trace_cfg is None:
            raise ValueError("eval.post_run.trace must be configured")
        trace_sample_resolution = str(
            trace_cfg.get("sample_resolution", "controller_step")
        ).strip().lower()
        if trace_sample_resolution != "controller_step":
            raise ValueError(
                "eval.post_run.trace.sample_resolution must be 'controller_step'"
            )
        target_pair_cfg = trace_cfg.get("target_pair")
        if target_pair_cfg is None:
            raise ValueError("eval.post_run.trace.target_pair must be configured")
        if int(target_pair_cfg.get("goal_distance", 0)) <= 0:
            raise ValueError(
                "eval.post_run.trace.target_pair.goal_distance must be positive"
            )
        if int(target_pair_cfg.get("cliff_distance", 0)) <= 0:
            raise ValueError(
                "eval.post_run.trace.target_pair.cliff_distance must be positive"
            )
        if int(trace_cfg.get("requested_trials_per_side", 0)) <= 0:
            raise ValueError(
                "eval.post_run.trace.requested_trials_per_side must be positive"
            )
        if int(trace_cfg.get("episode_batch_size", 0)) <= 0:
            raise ValueError("eval.post_run.trace.episode_batch_size must be positive")
    render_evals = cfg.render.get("render_evals", 0)
    if str(render_evals).strip().lower() != "all" and int(render_evals) < 0:
        raise ValueError("render.render_evals must be non-negative or 'all'")
    temporal_mode = cfg.render.get("temporal_mode")
    if temporal_mode is not None and str(temporal_mode).strip().lower() not in {"diagnostic", "realtime"}:
        raise ValueError("render.temporal_mode must be 'diagnostic' or 'realtime' when set")
    for key in ("target_frames", "trace_max_frames", "parallel_workers", "width", "height", "fps"):
        value = cfg.render.get(key)
        if value is not None and int(value) <= 0:
            raise ValueError(f"render.{key} must be positive when set")
    for stage_name in ("periodic", "post_train", "final"):
        stage_cfg = cfg.render.get(stage_name)
        if stage_cfg is None:
            continue
        if stage_name == "periodic":
            terrain_selection = str(stage_cfg.get("terrain_selection", "all")).strip().lower()
            if terrain_selection not in {"all", "prefix"}:
                raise ValueError(
                    "render.periodic.terrain_selection must be 'all' or 'prefix'"
                )
        archive_heatmaps = stage_cfg.get("archive_heatmaps")
        if archive_heatmaps is not None:
            invalid_archive_heatmaps = [
                str(name)
                for name in list(archive_heatmaps)
                if str(name).strip().lower() not in {"hits_falls", "time_hits", "time_falls"}
            ]
            if invalid_archive_heatmaps:
                raise ValueError(
                    f"render.{stage_name}.archive_heatmaps must be drawn from "
                    "['hits_falls', 'time_hits', 'time_falls']"
                )
        stage_mode = stage_cfg.get("temporal_mode")
        if stage_mode is not None and str(stage_mode).strip().lower() not in {"diagnostic", "realtime"}:
            raise ValueError(
                f"render.{stage_name}.temporal_mode must be 'diagnostic' or 'realtime' when set"
            )
        for key in ("target_frames", "trace_max_frames", "width", "height", "fps"):
            value = stage_cfg.get(key)
            if value is not None and int(value) <= 0:
                raise ValueError(f"render.{stage_name}.{key} must be positive when set")
    specimens_cfg = cfg.get("specimens")
    if specimens_cfg is not None and bool(specimens_cfg.get("enabled", False)):
        source = str(specimens_cfg.get("source", "archive_plus_fresh_pop")).strip().lower()
        if source != "archive_plus_fresh_pop":
            raise ValueError("specimens.source must be 'archive_plus_fresh_pop'")
        terrain_source = str(
            specimens_cfg.get("terrain_source", "val_plus_test_from_train_schedule")
        ).strip().lower()
        if terrain_source != "val_plus_test_from_train_schedule":
            raise ValueError(
                "specimens.terrain_source must be 'val_plus_test_from_train_schedule'"
            )
        n_terrains = int(specimens_cfg.get("n_terrains", 0))
        if n_terrains <= 0 or n_terrains % 2 != 0:
            raise ValueError("specimens.n_terrains must be a positive even integer")
        fresh_population_size = specimens_cfg.get("fresh_population_size")
        if fresh_population_size is not None and int(fresh_population_size) <= 0:
            raise ValueError("specimens.fresh_population_size must be positive when set")

        if int(specimens_cfg.get("priority_min_mirrored_pair_count", 0)) < 0:
            raise ValueError("specimens.priority_min_mirrored_pair_count must be nonnegative")
        if int(specimens_cfg.get("per_completion_bin", 0)) <= 0:
            raise ValueError("specimens.per_completion_bin must be positive")
        if int(specimens_cfg.get("preview_renders_per_specimen", 0)) < 0:
            raise ValueError("specimens.preview_renders_per_specimen must be nonnegative")
        preview_cfg = specimens_cfg.get("preview_render_quality")
        if preview_cfg is not None:
            preview_mode = str(preview_cfg.get("temporal_mode", "diagnostic")).strip().lower()
            if preview_mode not in {"diagnostic", "realtime"}:
                raise ValueError(
                    "specimens.preview_render_quality.temporal_mode must be "
                    "'diagnostic' or 'realtime'"
                )
            for key in ("target_frames", "width", "height", "fps"):
                value = preview_cfg.get(key)
                if value is not None and int(value) <= 0:
                    raise ValueError(
                        f"specimens.preview_render_quality.{key} must be positive when set"
                    )
    audit_cfg = cfg.get("audit")
    if audit_cfg is not None and bool(audit_cfg.get("enabled", False)):
        generation_eval_mode = str(
            audit_cfg.get("generation_eval_mode", "reduced_host")
        ).strip().lower()
        if generation_eval_mode not in {"reduced_host", "full_host"}:
            raise ValueError(
                "audit.generation_eval_mode must be 'reduced_host' or 'full_host'"
            )
        max_generation_probe_winners = int(
            audit_cfg.get("max_generation_probe_winners", 0)
        )
        if max_generation_probe_winners <= 0:
            raise ValueError("audit.max_generation_probe_winners must be positive")
    if str(cfg.collision.strategy) != "dynamic_broadphase":
        raise ValueError("collision.strategy must be 'dynamic_broadphase'")
    if str(cfg.collision.dynamic_impl) not in {"dense_mask", "sparse_candidate_list"}:
        raise ValueError(
            "collision.dynamic_impl must be 'dense_mask' or 'sparse_candidate_list'"
        )
    tuning_mode = str(cfg.collision.get("tuning", {}).get("mode", "")).strip()
    if tuning_mode != "hybrid_suite_probe":
        raise ValueError("collision.tuning.mode must be 'hybrid_suite_probe'")
    body_mode = str(cfg.body.mode).strip().lower()
    if body_mode not in {"fixed", "grown"}:
        raise ValueError("body.mode must be 'fixed' or 'grown'")
    if body_mode == "grown":
        growth_cfg = cfg.body.growth
        if int(growth_cfg.max_robot_h) <= 0 or int(growth_cfg.max_robot_w) <= 0:
            raise ValueError("body.growth.max_robot_h/max_robot_w must be positive")
        spawn_x = resolved_spawn_x(cfg)
        max_h, max_w = max_canvas_size(
            int(cfg.terrain.grid_h),
            int(cfg.terrain.grid_w),
            spawn_x=spawn_x,
            spawn_y=int(cfg.terrain.spawn_y),
        )
        if int(growth_cfg.max_robot_h) > max_h or int(growth_cfg.max_robot_w) > max_w:
            raise ValueError(
                "body.growth.max_robot_h/max_robot_w must fit within the terrain at spawn"
            )
        if str(growth_cfg.get("state_transition", "reset")).strip().lower() not in {"reset", "pass_through"}:
            raise ValueError("body.growth.state_transition must be 'reset' or 'pass_through'")
    else:
        fixed_hidden_init_scale = float(cfg.body.get("fixed", {}).get("hidden_init_scale", 0.0))
        if fixed_hidden_init_scale < 0.0:
            raise ValueError("body.fixed.hidden_init_scale must be non-negative")
        make_robot_body(
            int(cfg.terrain.robot_w),
            int(cfg.terrain.robot_h),
            shape=str(cfg.body.get("shape", "rectangle")),
        )
    if post_run_cfg is not None and bool(post_run_cfg.get("enabled", False)):
        if body_mode != "fixed":
            raise ValueError("eval.post_run currently supports body.mode='fixed' only")
        if str(cfg.terrain.get("task_mode", "legacy_cliff")).strip().lower() != _TASK_MODE_BILATERAL_GOAL_CLIFF:
            raise ValueError(
                "eval.post_run currently supports terrain.task_mode='bilateral_goal_cliff' only"
            )
        if not bool(cfg.get("specimens", {}).get("enabled", False)):
            raise ValueError("eval.post_run.enabled requires specimens.enabled=true")
    if probe_cfg is not None and bool(probe_cfg.get("enabled", False)):
        if body_mode != "fixed":
            raise ValueError("eval.robustness_probe currently supports body.mode='fixed' only")
        if int(cfg.terrain.robot_w) > int(probe_cfg.get("plateau_width", 0)):
            raise ValueError("eval.robustness_probe.plateau_width must fit the fixed robot width")
        if int(cfg.terrain.spawn_y) + int(cfg.terrain.robot_h) > int(probe_cfg.get("grid_h", 0)):
            raise ValueError("eval.robustness_probe grid_h must fit the fixed robot at spawn_y")


def _validate_qd_focus_config(cfg: DictConfig) -> None:
    focus_cfg = focus_config_from_section(cfg.qd.get("focus"))
    if not focus_cfg.enabled:
        return
    if focus_cfg.tighten_every_n_gens <= 0:
        raise ValueError("qd.focus.tighten_every_n_gens must be positive")
    if focus_cfg.min_occupied_to_tighten < 0:
        raise ValueError("qd.focus.min_occupied_to_tighten must be nonnegative")
    if focus_cfg.min_window_size <= 0:
        raise ValueError("qd.focus.min_window_size must be positive")
    resolved_qd = resolve_qd_archive(cfg)
    validate_focus_config(focus_cfg, resolved_qd.descriptor_names)


def _validate_runtime_overrides(runtime_override_keys: tuple[str, ...]) -> None:
    for key in runtime_override_keys:
        if not _path_is_allowed(key, PAPER_RUNTIME_ALLOWED_KEYS):
            raise ValueError(f"Runtime override '{key}' is not allowed")


def _validate_preset_diff(changed_paths: tuple[str, ...]) -> None:
    disallowed = [
        path for path in changed_paths if not _path_is_allowed(path, PAPER_PRESET_ALLOWED_KEYS)
    ]
    if disallowed:
        raise ValueError(
            "Preset changed paths outside the allowed paper-variant surface: "
            + ", ".join(disallowed)
        )


def _path_is_allowed(path: str, allowed_paths: frozenset[str]) -> bool:
    return any(path == allowed or path.startswith(f"{allowed}.") for allowed in allowed_paths)


def path_is_allowed(path: str, allowed_paths: frozenset[str]) -> bool:
    """Public wrapper for prefix-based config-path allowance checks."""
    return _path_is_allowed(path, allowed_paths)


def _diff_paths(base_cfg: DictConfig, other_cfg: DictConfig) -> set[str]:
    base = _flatten_config(OmegaConf.to_container(base_cfg, resolve=True))
    other = _flatten_config(OmegaConf.to_container(other_cfg, resolve=True))
    changed: set[str] = set()
    for key in sorted(set(base) | set(other)):
        if base.get(key) != other.get(key):
            changed.add(key)
    return changed


def diff_config_paths(base_cfg: DictConfig, other_cfg: DictConfig) -> set[str]:
    """Return flattened config paths whose resolved values differ."""
    return _diff_paths(base_cfg, other_cfg)


def _flatten_config(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_config(child, child_prefix))
        return flattened
    if isinstance(value, list):
        return {prefix: list(value)}
    return {prefix: value}


def _prune_excluded_paths(
    value: Any,
    *,
    excluded_paths: frozenset[str],
    prefix: str = "",
) -> Any:
    if isinstance(value, dict):
        pruned: dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if child_prefix in excluded_paths:
                continue
            pruned[key] = _prune_excluded_paths(
                child,
                excluded_paths=excluded_paths,
                prefix=child_prefix,
            )
        return pruned
    if isinstance(value, list):
        return [
            _prune_excluded_paths(
                child,
                excluded_paths=excluded_paths,
                prefix=prefix,
            )
            for child in value
        ]
    return value


def _memory_budget_bytes(cfg: DictConfig) -> int:
    usable_gib = (
        float(cfg.gpu_profile.hbm_gib) * float(cfg.gpu_profile.usable_fraction)
        - float(cfg.gpu_profile.compile_headroom_gib)
        - float(cfg.gpu_profile.safety_margin_gib)
    )
    if usable_gib <= 0:
        raise ValueError(
            f"GPU profile '{cfg.gpu_profile.id}' leaves no usable device memory budget"
        )
    return int(usable_gib * (1024**3))


def _resolve_dynamic_max_narrow(cfg: DictConfig) -> int | None:
    value = cfg.collision.get("dynamic_max_narrow", None)
    if value is None:
        return max(1, int(cfg.collision.dynamic_max_candidates) * 4)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"auto", "none", ""}:
            return max(1, int(cfg.collision.dynamic_max_candidates) * 4)
    return int(value)
