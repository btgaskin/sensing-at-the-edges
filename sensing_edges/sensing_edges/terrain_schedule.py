"""Frozen train/val/test terrain schedules with SHA256 integrity hashes.

Generates disjoint terrain pools from a master seed, serializes to
npz (grids) + JSON (metadata), and provides load/verify for train.py.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .terrain import generate_terrain


_SAMPLING_MODE_IID = "iid"
_SAMPLING_MODE_STRATIFIED_CLIFF_X = "stratified_cliff_x"
_SAMPLING_MODE_CANONICAL_BILATERAL_CYCLE = "canonical_bilateral_cycle"
_TASK_MODE_BILATERAL_GOAL_CLIFF = "bilateral_goal_cliff"
_GOAL_SIDES = ("left", "right")
_SCHEDULE_HASH_VERSION = 2
_SCHEDULE_VARIANT_COMPAT_ALIASES = {
    "bilateral_goal_cliff_asymmetric_pairs_7x2_grid30": (
        "bilateral_goal_cliff_asymmetric_pairs_8x2_30",
    ),
}


def schedule_bundle_name(
    *,
    seed: int,
    variant: str,
    n_gens: int | None = None,
    n_episodes: int | None = None,
    n_val: int | None = None,
    n_test: int | None = None,
) -> str:
    """Return a human-readable schedule bundle directory name."""
    base = f"seed_{int(seed)}_{str(variant)}"
    counts = (n_gens, n_episodes, n_val, n_test)
    if all(value is not None for value in counts):
        return (
            f"{base}_g{int(n_gens)}_ep{int(n_episodes)}"
            f"_val{int(n_val)}_test{int(n_test)}"
        )
    return base


def schedule_bundle_path(
    schedules_dir: str | Path,
    *,
    seed: int,
    variant: str,
    n_gens: int | None = None,
    n_episodes: int | None = None,
    n_val: int | None = None,
    n_test: int | None = None,
) -> Path:
    """Return the canonical path for a schedule bundle."""
    return Path(schedules_dir) / schedule_bundle_name(
        seed=seed,
        variant=variant,
        n_gens=n_gens,
        n_episodes=n_episodes,
        n_val=n_val,
        n_test=n_test,
    )


def compatible_schedule_variants(variant: str) -> tuple[str, ...]:
    """Return acceptable on-disk schedule variants for a requested runtime variant."""
    normalized = str(variant)
    return (normalized, *_SCHEDULE_VARIANT_COMPAT_ALIASES.get(normalized, ()))


def schedule_variant_matches(expected_variant: str, actual_variant: str) -> bool:
    """Return whether a loaded schedule variant is valid for the expected runtime variant."""
    return str(actual_variant) in compatible_schedule_variants(str(expected_variant))


def candidate_schedule_bundle_paths(
    schedules_dir: str | Path,
    *,
    seed: int,
    variant: str,
    n_gens: int | None = None,
    n_episodes: int | None = None,
    n_val: int | None = None,
    n_test: int | None = None,
) -> list[Path]:
    """Return candidate paths, preferring explicit count-qualified names and canonical variants."""
    candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate_variant in compatible_schedule_variants(variant):
        preferred = schedule_bundle_path(
            schedules_dir,
            seed=seed,
            variant=candidate_variant,
            n_gens=n_gens,
            n_episodes=n_episodes,
            n_val=n_val,
            n_test=n_test,
        )
        legacy = schedule_bundle_path(
            schedules_dir,
            seed=seed,
            variant=candidate_variant,
        )
        for candidate in (preferred, legacy):
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)
    return candidates


def resolve_schedule_bundle_path(
    schedules_dir: str | Path,
    *,
    seed: int,
    variant: str,
    n_gens: int | None = None,
    n_episodes: int | None = None,
    n_val: int | None = None,
    n_test: int | None = None,
) -> Path:
    """Return the existing bundle path if found, otherwise the canonical path."""
    candidates = candidate_schedule_bundle_paths(
        schedules_dir,
        seed=seed,
        variant=variant,
        n_gens=n_gens,
        n_episodes=n_episodes,
        n_val=n_val,
        n_test=n_test,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


@dataclass
class TerrainEntry:
    """One terrain instance with its metadata."""
    grid: np.ndarray
    cliff_x: int
    gaps: list[tuple[int, int]]
    metadata: dict[str, object] | None = None


@dataclass
class TerrainSchedule:
    """Frozen train/val/test terrain pools."""
    train: list[list[TerrainEntry]]     # [gen][episode]
    val: list[TerrainEntry]
    test: list[TerrainEntry]
    seed: int
    terrain_cfg: dict
    curriculum: bool = False
    transition_gen: int | None = None
    phase_1_cfg: dict | None = None
    phase_2_cfg: dict | None = None
    curriculum_phase_cfgs: list[dict] | None = None
    curriculum_phase_end_gens: list[int] | None = None
    hash_version: int = _SCHEDULE_HASH_VERSION
    train_hash: str = ""
    val_hash: str = ""
    test_hash: str = ""
    train_content_hash: str = ""
    val_content_hash: str = ""
    test_content_hash: str = ""


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _hash_payload_and_metadata(payload_bytes: bytes, metadata_obj: object) -> str:
    hasher = hashlib.sha256()
    hasher.update(payload_bytes)
    hasher.update(b"\x00")
    hasher.update(_canonical_json_bytes(metadata_obj))
    return hasher.hexdigest()


def _merge_phase(terrain_cfg: dict, phase_cfg: dict | None) -> dict:
    """Overlay one curriculum phase onto the shared terrain config."""
    merged = {
        key: copy.deepcopy(value)
        for key, value in terrain_cfg.items()
        if key not in {
            "curriculum",
            "phase_1",
            "phase_2",
            "curriculum_phases",
            "curriculum_phase_switch_gens",
        }
    }
    if phase_cfg:
        for key, value in phase_cfg.items():
            merged[key] = copy.deepcopy(value)
    return merged


def _curriculum_phase_cfgs(terrain_cfg: dict) -> list[dict]:
    """Return the ordered curriculum phase configs."""
    if not terrain_cfg.get("curriculum", False):
        return []
    explicit_phases = terrain_cfg.get("curriculum_phases")
    if explicit_phases is not None:
        phases = [dict(copy.deepcopy(phase)) for phase in explicit_phases]
        if not phases:
            raise ValueError("curriculum_phases must be non-empty when curriculum=true.")
        return phases
    return [
        dict(copy.deepcopy(terrain_cfg.get("phase_1") or {})),
        dict(copy.deepcopy(terrain_cfg.get("phase_2") or {})),
    ]


def _curriculum_phase_end_gens(terrain_cfg: dict, *, n_gens: int, n_phases: int) -> list[int]:
    """Resolve curriculum phase boundaries for the current run length."""
    if n_phases <= 0:
        return []
    raw_switches = terrain_cfg.get("curriculum_phase_switch_gens")
    if raw_switches is not None:
        switches = [int(value) for value in raw_switches]
        if len(switches) != int(n_phases) - 1:
            raise ValueError(
                "curriculum_phase_switch_gens must have exactly n_phases - 1 entries."
            )
        if any(value <= 0 for value in switches):
            raise ValueError("curriculum_phase_switch_gens must contain only positive integers.")
        if any(curr <= prev for prev, curr in zip(switches, switches[1:])):
            raise ValueError("curriculum_phase_switch_gens must be strictly increasing.")
        phase_end_gens = [min(int(value), int(n_gens)) for value in switches]
        phase_end_gens.append(int(n_gens))
        last_end = 0
        resolved: list[int] = []
        for phase_end in phase_end_gens:
            clamped_end = max(int(phase_end), int(last_end))
            resolved.append(clamped_end)
            last_end = clamped_end
        return resolved
    return [
        int(math.ceil(((phase_idx + 1) * int(n_gens)) / int(n_phases)))
        for phase_idx in range(int(n_phases))
    ]


def resolve_generation_terrain_cfg(
    terrain_cfg: dict,
    *,
    gen_idx: int,
    n_gens: int,
) -> tuple[dict, int | None]:
    """Resolve the active terrain config for one training generation."""
    if not terrain_cfg.get("curriculum", False):
        return _merge_phase(terrain_cfg, None), None

    phase_cfgs = _curriculum_phase_cfgs(terrain_cfg)
    phase_end_gens = _curriculum_phase_end_gens(
        terrain_cfg,
        n_gens=int(n_gens),
        n_phases=len(phase_cfgs),
    )
    phase_idx = next(
        idx
        for idx, phase_end in enumerate(phase_end_gens)
        if int(gen_idx) < int(phase_end)
    )
    transition_gen = phase_end_gens[0] if len(phase_cfgs) == 2 else None
    return _merge_phase(terrain_cfg, phase_cfgs[phase_idx]), transition_gen


def resolve_final_terrain_cfg(terrain_cfg: dict) -> dict:
    """Resolve the final-task terrain config for validation and test splits."""
    if not terrain_cfg.get("curriculum", False):
        return _merge_phase(terrain_cfg, None)
    phase_cfgs = _curriculum_phase_cfgs(terrain_cfg)
    return _merge_phase(terrain_cfg, phase_cfgs[-1])


def _generate_pool(
    rng: np.random.Generator,
    n_terrains: int,
    terrain_cfg: dict,
) -> list[TerrainEntry]:
    """Generate a flat pool of terrains from an RNG stream."""
    sampling_mode = str(terrain_cfg.get("sampling_mode", _SAMPLING_MODE_IID))
    task_mode = str(terrain_cfg.get("task_mode", "legacy_cliff"))
    active_gap_widths = list(terrain_cfg.get("gap_widths") or [])
    exact_gap_widths = list(terrain_cfg.get("exact_gap_widths") or [])
    gap_budget = int(terrain_cfg.get("gap_budget", 0))
    hazard_mode = str(terrain_cfg.get("hazard_mode", "legacy"))

    cliff_x_sequence: list[int] | None = None
    if sampling_mode == _SAMPLING_MODE_STRATIFIED_CLIFF_X:
        if (
            hazard_mode != "legacy"
            or active_gap_widths
            or exact_gap_widths
            or gap_budget > 0
        ):
            raise ValueError(
                "sampling_mode='stratified_cliff_x' is only supported for "
                "legacy cliff-only terrains without gaps."
            )
        cliff_values = np.arange(
            int(terrain_cfg["cliff_x_min"]),
            int(terrain_cfg["cliff_x_max"]) + 1,
            dtype=np.int64,
        )
        if cliff_values.size == 0:
            raise ValueError("stratified_cliff_x requires a non-empty cliff range.")

        cliff_x_sequence = []
        full_cycles, remainder = divmod(int(n_terrains), int(cliff_values.size))
        for _ in range(full_cycles):
            cycle_values = cliff_values.copy()
            rng.shuffle(cycle_values)
            cliff_x_sequence.extend(int(value) for value in cycle_values)
        if remainder:
            strata = np.array_split(cliff_values, remainder)
            sampled = [int(stratum[rng.integers(0, len(stratum))]) for stratum in strata]
            sampled_arr = np.asarray(sampled, dtype=np.int64)
            rng.shuffle(sampled_arr)
            cliff_x_sequence.extend(int(value) for value in sampled_arr)
    elif sampling_mode == _SAMPLING_MODE_CANONICAL_BILATERAL_CYCLE:
        if task_mode != _TASK_MODE_BILATERAL_GOAL_CLIFF:
            raise ValueError(
                "sampling_mode='canonical_bilateral_cycle' requires "
                "task_mode='bilateral_goal_cliff'."
            )
        if n_terrains <= 0:
            return []
        return _generate_bilateral_eval_pool(rng, n_terrains, terrain_cfg)
    elif sampling_mode != _SAMPLING_MODE_IID:
        raise ValueError(f"Unsupported sampling_mode={sampling_mode!r}")

    entries = []
    gap_start_bands = list(terrain_cfg.get("gap_start_bands") or [])
    for terrain_idx in range(n_terrains):
        seed = int(rng.integers(0, 2**31))
        gap_start_min = terrain_cfg.get("gap_start_min")
        gap_start_max = terrain_cfg.get("gap_start_max")
        cliff_x_min = int(terrain_cfg["cliff_x_min"])
        cliff_x_max = int(terrain_cfg["cliff_x_max"])
        if gap_start_bands:
            band = gap_start_bands[int(terrain_idx) % len(gap_start_bands)]
            if len(band) != 2:
                raise ValueError("gap_start_bands entries must be [min, max] pairs.")
            gap_start_min = int(band[0])
            gap_start_max = int(band[1])
        if cliff_x_sequence is not None:
            cliff_x_min = cliff_x_sequence[terrain_idx]
            cliff_x_max = cliff_x_min
        grid, cliff_x, gaps, metadata = generate_terrain(
            seed,
            grid_h=terrain_cfg["grid_h"],
            grid_w=terrain_cfg["grid_w"],
            platform_y=terrain_cfg.get("platform_y", terrain_cfg.get("plateau_height", 3)),
            base_width=terrain_cfg.get("base_width", 10),
            cliff_x_min=cliff_x_min,
            cliff_x_max=cliff_x_max,
            gap_widths=terrain_cfg.get("gap_widths") or None,
            gap_budget=terrain_cfg.get("gap_budget", 0),
            exact_gap_widths=terrain_cfg.get("exact_gap_widths") or None,
            min_gap_separation=terrain_cfg.get("min_gap_separation", 2),
            min_post_gap_runway=terrain_cfg.get("min_post_gap_runway", 2),
            hazard_mode=terrain_cfg.get("hazard_mode", "legacy"),
            spawn_plateau_width=terrain_cfg.get("spawn_plateau_width"),
            approach_width=terrain_cfg.get("approach_width"),
            randomized_zone_width=terrain_cfg.get("randomized_zone_width"),
            remainder_width=terrain_cfg.get("remainder_width"),
            single_gap_width=terrain_cfg.get("single_gap_width"),
            layout_sampling=terrain_cfg.get("layout_sampling", "uniform_legal_layouts"),
            gap_start_min=gap_start_min,
            gap_start_max=gap_start_max,
            task_mode=terrain_cfg.get("task_mode", "legacy_cliff"),
            center_plateau_width=terrain_cfg.get("center_plateau_width"),
            symmetric_distance_min=terrain_cfg.get("symmetric_distance_min"),
            symmetric_distance_max=terrain_cfg.get("symmetric_distance_max"),
            goal_side=terrain_cfg.get("goal_side"),
            correct_side_distance_scale=terrain_cfg.get("correct_side_distance_scale", 1.0),
            incorrect_side_distance_scale=terrain_cfg.get("incorrect_side_distance_scale", 1.0),
            cliff_front_gap_width=terrain_cfg.get("cliff_front_gap_width", 0),
            cliff_front_gap_widths=terrain_cfg.get("cliff_front_gap_widths"),
            cliff_post_gap_runway=terrain_cfg.get("cliff_post_gap_runway", 0),
            cliff_probe_gap_width=terrain_cfg.get("cliff_probe_gap_width", 0),
            cliff_probe_ledge_width=terrain_cfg.get("cliff_probe_ledge_width", 0),
            distance_pairs=terrain_cfg.get("distance_pairs"),
        )
        entries.append(TerrainEntry(grid=grid, cliff_x=cliff_x, gaps=gaps, metadata=metadata))
    return entries


def _normalized_distance_pairs(terrain_cfg: dict) -> list[tuple[int, int]]:
    raw_pairs = terrain_cfg.get("distance_pairs")
    if raw_pairs is None:
        return []
    normalized_pairs: list[tuple[int, int]] = []
    for pair in raw_pairs:
        if hasattr(pair, "keys") and "goal_distance" in pair and "cliff_distance" in pair:
            goal_distance = int(pair["goal_distance"])
            cliff_distance = int(pair["cliff_distance"])
        else:
            if len(pair) != 2:
                raise ValueError("distance_pairs entries must have exactly two values.")
            goal_distance = int(pair[0])
            cliff_distance = int(pair[1])
        if goal_distance < 0 or cliff_distance < 0:
            raise ValueError("distance_pairs distances must be non-negative.")
        normalized_pairs.append((goal_distance, cliff_distance))
    return normalized_pairs


def _normalized_cliff_front_gap_widths(terrain_cfg: dict) -> list[int]:
    raw_widths = terrain_cfg.get("cliff_front_gap_widths")
    if raw_widths is not None:
        normalized = [int(width) for width in raw_widths]
        if any(width < 0 for width in normalized):
            raise ValueError("cliff_front_gap_widths must contain only non-negative integers.")
        if normalized:
            return list(dict.fromkeys(normalized))

    width = int(terrain_cfg.get("cliff_front_gap_width", 0))
    if width < 0:
        raise ValueError("cliff_front_gap_width must be non-negative.")
    return [width]


def _bilateral_conditions(terrain_cfg: dict) -> list[tuple[int, int, str, int]]:
    gap_widths = _normalized_cliff_front_gap_widths(terrain_cfg)
    explicit_pairs = _normalized_distance_pairs(terrain_cfg)
    if explicit_pairs:
        return [
            (int(goal_distance), int(cliff_distance), side, int(gap_width))
            for goal_distance, cliff_distance in list(dict.fromkeys(explicit_pairs))
            for side in _GOAL_SIDES
            for gap_width in gap_widths
        ]
    distance_min = int(terrain_cfg["symmetric_distance_min"])
    distance_max = int(terrain_cfg["symmetric_distance_max"])
    if distance_max < distance_min:
        raise ValueError("symmetric_distance_max must be >= symmetric_distance_min")
    correct_scale = float(terrain_cfg.get("correct_side_distance_scale", 1.0))
    incorrect_scale = float(terrain_cfg.get("incorrect_side_distance_scale", 1.0))
    return list(
        dict.fromkeys(
            (
                int(math.ceil(float(distance) * correct_scale)),
                int(math.ceil(float(distance) * incorrect_scale)),
                side,
                int(gap_width),
            )
            for distance in range(distance_min, distance_max + 1)
            for side in _GOAL_SIDES
            for gap_width in gap_widths
        )
    )


def _bilateral_pair_conditions(terrain_cfg: dict) -> list[tuple[int, int, int]]:
    """Return unique mirrored bilateral conditions without goal-side duplication."""
    return list(
        dict.fromkeys(
            (int(goal_distance), int(cliff_distance), int(gap_width))
            for goal_distance, cliff_distance, _side, gap_width in _bilateral_conditions(terrain_cfg)
        )
    )


def _build_bilateral_entry(
    goal_distance: int,
    cliff_distance: int,
    goal_side: str,
    cliff_front_gap_width: int,
    terrain_cfg: dict,
) -> TerrainEntry:
    grid, cliff_x, gaps, metadata = generate_terrain(
        0,
        grid_h=terrain_cfg["grid_h"],
        grid_w=terrain_cfg["grid_w"],
        platform_y=terrain_cfg.get("platform_y", terrain_cfg.get("plateau_height", 3)),
        base_width=terrain_cfg.get("base_width", 10),
        cliff_x_min=terrain_cfg.get("cliff_x_min", 0),
        cliff_x_max=terrain_cfg.get("cliff_x_max", 0),
        task_mode=terrain_cfg.get("task_mode", "legacy_cliff"),
        center_plateau_width=terrain_cfg.get("center_plateau_width", terrain_cfg.get("base_width", 10)),
        goal_side=str(goal_side),
        correct_side_distance_scale=terrain_cfg.get("correct_side_distance_scale", 1.0),
        incorrect_side_distance_scale=terrain_cfg.get("incorrect_side_distance_scale", 1.0),
        cliff_front_gap_width=int(cliff_front_gap_width),
        cliff_front_gap_widths=terrain_cfg.get("cliff_front_gap_widths"),
        cliff_post_gap_runway=terrain_cfg.get("cliff_post_gap_runway", 0),
        cliff_probe_gap_width=terrain_cfg.get("cliff_probe_gap_width", 0),
        cliff_probe_ledge_width=terrain_cfg.get("cliff_probe_ledge_width", 0),
        distance_pairs=[{"goal_distance": int(goal_distance), "cliff_distance": int(cliff_distance)}],
    )
    return TerrainEntry(grid=grid, cliff_x=cliff_x, gaps=gaps, metadata=metadata)


def _generate_bilateral_train_schedule(
    rng: np.random.Generator,
    *,
    n_gens: int,
    n_episodes: int,
    terrain_cfg: dict,
) -> list[list[TerrainEntry]]:
    def _generate_stage_schedule(stage_cfg: dict, *, stage_n_gens: int) -> list[list[TerrainEntry]]:
        resolved_n_episodes = int(n_episodes)
        if resolved_n_episodes <= 0:
            raise ValueError("canonical_bilateral_cycle requires train.n_episodes > 0.")
        if resolved_n_episodes % 2 != 0:
            raise ValueError(
                "canonical_bilateral_cycle requires an even train.n_episodes so each "
                "generation can be exactly 50/50 left/right."
            )

        pair_conditions = _bilateral_pair_conditions(stage_cfg)
        n_pair_conditions = len(pair_conditions)
        pairs_per_generation = resolved_n_episodes // 2
        if pairs_per_generation > n_pair_conditions:
            raise ValueError(
                "canonical_bilateral_cycle requires train.n_episodes <= "
                f"{n_pair_conditions * 2} so each generation can use unique mirrored layouts."
            )

        cycle_slots = math.lcm(n_pair_conditions, pairs_per_generation)
        repeats_per_pair = cycle_slots // n_pair_conditions
        generations_per_cycle = cycle_slots // pairs_per_generation

        stage_train: list[list[TerrainEntry]] = []
        gen_idx = 0
        while gen_idx < int(stage_n_gens):
            remaining = {condition: repeats_per_pair for condition in pair_conditions}
            for _ in range(generations_per_cycle):
                if gen_idx >= int(stage_n_gens):
                    break
                randomized = list(pair_conditions)
                rng.shuffle(randomized)
                layouts = sorted(
                    randomized,
                    key=lambda condition: remaining[condition],
                    reverse=True,
                )[:pairs_per_generation]
                rng.shuffle(layouts)
                generation: list[TerrainEntry] = []
                for goal_distance, cliff_distance, cliff_front_gap_width in layouts:
                    generation.append(
                        _build_bilateral_entry(
                            int(goal_distance),
                            int(cliff_distance),
                            "left",
                            int(cliff_front_gap_width),
                            stage_cfg,
                        )
                    )
                    generation.append(
                        _build_bilateral_entry(
                            int(goal_distance),
                            int(cliff_distance),
                            "right",
                            int(cliff_front_gap_width),
                            stage_cfg,
                        )
                    )
                for layout in layouts:
                    remaining[layout] -= 1
                stage_train.append(generation)
                gen_idx += 1
        return stage_train

    if not terrain_cfg.get("curriculum", False):
        return _generate_stage_schedule(terrain_cfg, stage_n_gens=int(n_gens))

    phase_cfgs = _curriculum_phase_cfgs(terrain_cfg)
    phase_end_gens = _curriculum_phase_end_gens(
        terrain_cfg,
        n_gens=int(n_gens),
        n_phases=len(phase_cfgs),
    )
    train: list[list[TerrainEntry]] = []
    phase_start = 0
    for phase_cfg, phase_end in zip(phase_cfgs, phase_end_gens):
        stage_n_gens = int(phase_end) - int(phase_start)
        if stage_n_gens <= 0:
            continue
        merged_cfg = _merge_phase(terrain_cfg, phase_cfg)
        train.extend(_generate_stage_schedule(merged_cfg, stage_n_gens=stage_n_gens))
        phase_start = int(phase_end)
    return train


def _generate_bilateral_eval_pool(
    rng: np.random.Generator,
    n_terrains: int,
    terrain_cfg: dict,
) -> list[TerrainEntry]:
    pair_conditions = _bilateral_pair_conditions(terrain_cfg)
    entries: list[TerrainEntry] = []
    while len(entries) < int(n_terrains):
        order = np.arange(len(pair_conditions), dtype=np.int32)
        rng.shuffle(order)
        for idx in order.tolist():
            if len(entries) >= int(n_terrains):
                break
            goal_distance, cliff_distance, cliff_front_gap_width = pair_conditions[int(idx)]
            if len(entries) + 2 > int(n_terrains):
                side = _GOAL_SIDES[int(rng.integers(0, len(_GOAL_SIDES)))]
                entries.append(
                    _build_bilateral_entry(
                        goal_distance,
                        cliff_distance,
                        side,
                        cliff_front_gap_width,
                        terrain_cfg,
                    )
                )
                break
            entries.append(
                _build_bilateral_entry(
                    goal_distance,
                    cliff_distance,
                    "left",
                    cliff_front_gap_width,
                    terrain_cfg,
                )
            )
            entries.append(
                _build_bilateral_entry(
                    goal_distance,
                    cliff_distance,
                    "right",
                    cliff_front_gap_width,
                    terrain_cfg,
                )
            )
    return entries


def generate_schedule(
    seed: int,
    n_gens: int,
    n_episodes: int,
    n_val: int,
    n_test: int,
    terrain_cfg: dict,
) -> TerrainSchedule:
    """Generate a complete terrain schedule with disjoint RNG streams.

    The master seed is split into 3 independent streams for
    train, val, and test — no cross-contamination.
    """
    master = np.random.default_rng(seed)
    seeds = master.integers(0, 2**31, size=3)
    train_rng = np.random.default_rng(int(seeds[0]))
    val_rng = np.random.default_rng(int(seeds[1]))
    test_rng = np.random.default_rng(int(seeds[2]))
    is_curriculum = bool(terrain_cfg.get("curriculum", False))
    phase_cfgs = _curriculum_phase_cfgs(terrain_cfg) if is_curriculum else []
    phase_end_gens = (
        _curriculum_phase_end_gens(
            terrain_cfg,
            n_gens=int(n_gens),
            n_phases=len(phase_cfgs),
        )
        if phase_cfgs
        else []
    )
    transition_gen = phase_end_gens[0] if len(phase_end_gens) == 2 else None
    sampling_mode = str(terrain_cfg.get("sampling_mode", _SAMPLING_MODE_IID))
    task_mode = str(terrain_cfg.get("task_mode", "legacy_cliff"))

    # Train: organized as [gen][episode].
    if (
        task_mode == _TASK_MODE_BILATERAL_GOAL_CLIFF
        and sampling_mode == _SAMPLING_MODE_CANONICAL_BILATERAL_CYCLE
    ):
        train = _generate_bilateral_train_schedule(
            train_rng,
            n_gens=n_gens,
            n_episodes=n_episodes,
            terrain_cfg=terrain_cfg,
        )
        final_cfg = resolve_final_terrain_cfg(terrain_cfg)
        val = _generate_bilateral_eval_pool(val_rng, n_val, final_cfg)
        test = _generate_bilateral_eval_pool(test_rng, n_test, final_cfg)
    else:
        train = []
        for gen_idx in range(n_gens):
            gen_cfg, _ = resolve_generation_terrain_cfg(
                terrain_cfg,
                gen_idx=gen_idx,
                n_gens=n_gens,
            )
            gen_entries = _generate_pool(train_rng, n_episodes, gen_cfg)
            train.append(gen_entries)

        final_cfg = resolve_final_terrain_cfg(terrain_cfg)
        val = _generate_pool(val_rng, n_val, final_cfg)
        test = _generate_pool(test_rng, n_test, final_cfg)

    return TerrainSchedule(
        train=train,
        val=val,
        test=test,
        seed=seed,
        terrain_cfg=copy.deepcopy(terrain_cfg),
        curriculum=is_curriculum,
        transition_gen=transition_gen,
        phase_1_cfg=copy.deepcopy(terrain_cfg.get("phase_1")) if is_curriculum else None,
        phase_2_cfg=copy.deepcopy(terrain_cfg.get("phase_2")) if is_curriculum else None,
        curriculum_phase_cfgs=copy.deepcopy(phase_cfgs) if phase_cfgs else None,
        curriculum_phase_end_gens=copy.deepcopy(phase_end_gens) if phase_end_gens else None,
    )


def _entries_to_npz_bytes(entries: list[TerrainEntry]) -> bytes:
    """Stack terrain grids and serialize to npz bytes."""
    import io
    grids = np.stack([e.grid for e in entries])
    buf = io.BytesIO()
    np.savez_compressed(buf, grids=grids)
    return buf.getvalue()


def _entries_metadata(entries: list[TerrainEntry]) -> list[dict]:
    """Extract metadata (cliff_x, gaps) per terrain entry."""
    return [
        {"cliff_x": e.cliff_x, "gaps": e.gaps, "metadata": e.metadata or {}}
        for e in entries
    ]


def save_schedule(schedule: TerrainSchedule, output_dir: str | Path) -> Path:
    """Save schedule to disk as npz + JSON.

    Output structure:
        output_dir/
            schedule.json    # metadata + hashes
            train.npz        # stacked grids
            val.npz
            test.npz
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Flatten train for npz (all gens × episodes).
    train_flat = [e for gen in schedule.train for e in gen]
    train_bytes = _entries_to_npz_bytes(train_flat)
    val_bytes = _entries_to_npz_bytes(schedule.val)
    test_bytes = _entries_to_npz_bytes(schedule.test)
    train_metadata = [_entries_metadata(gen) for gen in schedule.train]
    val_metadata = _entries_metadata(schedule.val)
    test_metadata = _entries_metadata(schedule.test)

    (out / "train.npz").write_bytes(train_bytes)
    (out / "val.npz").write_bytes(val_bytes)
    (out / "test.npz").write_bytes(test_bytes)

    schedule.hash_version = _SCHEDULE_HASH_VERSION
    schedule.train_content_hash = _hash_payload_and_metadata(train_bytes, train_metadata)
    schedule.val_content_hash = _hash_payload_and_metadata(val_bytes, val_metadata)
    schedule.test_content_hash = _hash_payload_and_metadata(test_bytes, test_metadata)
    # Backward-compatible aliases retained for downstream metadata payloads.
    schedule.train_hash = schedule.train_content_hash
    schedule.val_hash = schedule.val_content_hash
    schedule.test_hash = schedule.test_content_hash

    n_episodes = len(schedule.train[0]) if schedule.train else 0
    meta = {
        "hash_version": schedule.hash_version,
        "seed": schedule.seed,
        "n_gens": len(schedule.train),
        "n_episodes": n_episodes,
        "n_val": len(schedule.val),
        "n_test": len(schedule.test),
        "terrain_cfg": schedule.terrain_cfg,
        "curriculum": bool(schedule.curriculum),
        "transition_gen": schedule.transition_gen,
        "phase_1_cfg": schedule.phase_1_cfg,
        "phase_2_cfg": schedule.phase_2_cfg,
        "curriculum_phase_cfgs": schedule.curriculum_phase_cfgs,
        "curriculum_phase_end_gens": schedule.curriculum_phase_end_gens,
        "train_content_hash": schedule.train_content_hash,
        "val_content_hash": schedule.val_content_hash,
        "test_content_hash": schedule.test_content_hash,
        # Legacy field names preserved so older consumers can still introspect.
        "train_hash": schedule.train_hash,
        "val_hash": schedule.val_hash,
        "test_hash": schedule.test_hash,
        "train_metadata": train_metadata,
        "val_metadata": val_metadata,
        "test_metadata": test_metadata,
    }
    with open(out / "schedule.json", "w") as f:
        json.dump(meta, f, indent=2)

    return out


def load_schedule(
    schedule_dir: str | Path,
    verify_hashes: bool = True,
    *,
    allow_legacy_hashes: bool = False,
) -> TerrainSchedule:
    """Load schedule from disk and optionally verify integrity.

    Raises ValueError if hash verification fails.
    """
    d = Path(schedule_dir)
    with open(d / "schedule.json") as f:
        meta = json.load(f)

    train_bytes = (d / "train.npz").read_bytes()
    val_bytes = (d / "val.npz").read_bytes()
    test_bytes = (d / "test.npz").read_bytes()
    hash_version = int(meta.get("hash_version", 1))
    train_meta = meta["train_metadata"]
    val_meta = meta["val_metadata"]
    test_meta = meta["test_metadata"]

    if verify_hashes:
        if hash_version >= _SCHEDULE_HASH_VERSION:
            for name, data, split_metadata, expected in [
                ("train", train_bytes, train_meta, str(meta.get("train_content_hash", ""))),
                ("val", val_bytes, val_meta, str(meta.get("val_content_hash", ""))),
                ("test", test_bytes, test_meta, str(meta.get("test_content_hash", ""))),
            ]:
                if not expected:
                    raise ValueError(
                        f"Terrain schedule missing required content hash for {name} "
                        f"(hash_version={hash_version})."
                    )
                actual = _hash_payload_and_metadata(data, split_metadata)
                if actual != expected:
                    raise ValueError(
                        f"Terrain schedule hash mismatch for {name}: "
                        f"expected {expected[:16]}..., got {actual[:16]}..."
                    )
        elif allow_legacy_hashes:
            for name, data, expected in [
                ("train", train_bytes, str(meta.get("train_hash", ""))),
                ("val", val_bytes, str(meta.get("val_hash", ""))),
                ("test", test_bytes, str(meta.get("test_hash", ""))),
            ]:
                if not expected:
                    raise ValueError(
                        f"Terrain schedule missing legacy hash for {name}."
                    )
                actual = _hash_bytes(data)
                if actual != expected:
                    raise ValueError(
                        f"Terrain schedule hash mismatch for {name}: "
                        f"expected {expected[:16]}..., got {actual[:16]}..."
                    )
        else:
            raise ValueError(
                "Terrain schedule uses legacy hash schema without metadata coverage. "
                "Regenerate with hash_version=2 or load with allow_legacy_hashes=True."
            )

    import io
    train_grids = np.load(io.BytesIO(train_bytes))["grids"]
    val_grids = np.load(io.BytesIO(val_bytes))["grids"]
    test_grids = np.load(io.BytesIO(test_bytes))["grids"]

    n_gens = meta["n_gens"]
    n_episodes = meta["n_episodes"]

    # Reconstruct train as [gen][episode].
    train: list[list[TerrainEntry]] = []
    idx = 0
    for gen_idx in range(n_gens):
        gen_entries = []
        for ep_idx in range(n_episodes):
            m = train_meta[gen_idx][ep_idx]
            gen_entries.append(TerrainEntry(
                grid=train_grids[idx],
                cliff_x=m["cliff_x"],
                gaps=[tuple(g) for g in m["gaps"]],
                metadata=dict(m.get("metadata") or {}),
            ))
            idx += 1
        train.append(gen_entries)

    val = [
        TerrainEntry(
            grid=val_grids[i],
            cliff_x=m["cliff_x"],
            gaps=[tuple(g) for g in m["gaps"]],
            metadata=dict(m.get("metadata") or {}),
        )
        for i, m in enumerate(val_meta)
    ]
    test = [
        TerrainEntry(
            grid=test_grids[i],
            cliff_x=m["cliff_x"],
            gaps=[tuple(g) for g in m["gaps"]],
            metadata=dict(m.get("metadata") or {}),
        )
        for i, m in enumerate(test_meta)
    ]

    return TerrainSchedule(
        train=train,
        val=val,
        test=test,
        seed=meta["seed"],
        terrain_cfg=meta["terrain_cfg"],
        curriculum=bool(meta.get("curriculum", False)),
        transition_gen=meta.get("transition_gen"),
        phase_1_cfg=meta.get("phase_1_cfg"),
        phase_2_cfg=meta.get("phase_2_cfg"),
        curriculum_phase_cfgs=meta.get("curriculum_phase_cfgs"),
        curriculum_phase_end_gens=meta.get("curriculum_phase_end_gens"),
        hash_version=hash_version,
        train_hash=str(meta.get("train_content_hash", meta.get("train_hash", ""))),
        val_hash=str(meta.get("val_content_hash", meta.get("val_hash", ""))),
        test_hash=str(meta.get("test_content_hash", meta.get("test_hash", ""))),
        train_content_hash=str(meta.get("train_content_hash", meta.get("train_hash", ""))),
        val_content_hash=str(meta.get("val_content_hash", meta.get("val_hash", ""))),
        test_content_hash=str(meta.get("test_content_hash", meta.get("test_hash", ""))),
    )


def validate_bilateral_schedule_semantics(schedule: TerrainSchedule) -> None:
    """Validate mirrored ordering/balance assumptions for bilateral schedules."""

    def _is_bilateral_entry(entry: TerrainEntry) -> bool:
        metadata = dict(entry.metadata or {})
        return str(metadata.get("task_mode", "")).strip().lower() == _TASK_MODE_BILATERAL_GOAL_CLIFF

    def _entry_int(entry: TerrainEntry, key: str) -> int:
        return int((entry.metadata or {}).get(key, 0))

    def _validate_group(entries: list[TerrainEntry], *, label: str) -> None:
        if not entries:
            return
        bilateral_flags = [_is_bilateral_entry(entry) for entry in entries]
        if not any(bilateral_flags):
            return
        if not all(bilateral_flags):
            raise ValueError(f"Bilateral schedule semantics failed for {label}: mixed task modes in one group.")
        if len(entries) % 2 != 0:
            raise ValueError(
                f"Bilateral schedule semantics failed for {label}: expected even episode count, got {len(entries)}."
            )
        left_count = 0
        right_count = 0
        for pair_idx in range(0, len(entries), 2):
            left_entry = entries[pair_idx]
            right_entry = entries[pair_idx + 1]
            left_dir = _entry_int(left_entry, "goal_direction")
            right_dir = _entry_int(right_entry, "goal_direction")
            if left_dir == 0 or right_dir == 0 or left_dir != -right_dir:
                raise ValueError(
                    f"Bilateral schedule semantics failed for {label}: pair {pair_idx // 2} has invalid goal directions "
                    f"({left_dir}, {right_dir})."
                )
            if _entry_int(left_entry, "goal_distance") != _entry_int(right_entry, "goal_distance"):
                raise ValueError(
                    f"Bilateral schedule semantics failed for {label}: pair {pair_idx // 2} goal_distance mismatch."
                )
            if _entry_int(left_entry, "cliff_distance") != _entry_int(right_entry, "cliff_distance"):
                raise ValueError(
                    f"Bilateral schedule semantics failed for {label}: pair {pair_idx // 2} cliff_distance mismatch."
                )
            left_count += int(left_dir < 0) + int(right_dir < 0)
            right_count += int(left_dir > 0) + int(right_dir > 0)
        if left_count != right_count:
            raise ValueError(
                f"Bilateral schedule semantics failed for {label}: expected balanced left/right goals, got "
                f"left={left_count}, right={right_count}."
            )

    for generation_idx, generation_entries in enumerate(schedule.train):
        _validate_group(list(generation_entries), label=f"train[generation={generation_idx}]")
    _validate_group(list(schedule.val), label="val")
    _validate_group(list(schedule.test), label="test")


def schedule_train_terrains_for_gen(
    schedule: TerrainSchedule, gen: int
) -> list[TerrainEntry]:
    """Get the train terrain entries for a specific generation."""
    if gen < 0 or gen >= len(schedule.train):
        raise IndexError(f"Generation {gen} out of range [0, {len(schedule.train)})")
    return schedule.train[gen]
