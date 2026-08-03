"""Collision capacity tuning for the fixed-morphology sensing_edges pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jax_evogym import constants as C
from jax_evogym.collision import make_collision_data
from jax_evogym.types import CollisionData


@dataclass(frozen=True)
class CollisionBuildCapacity:
    """Build-time collision buffer capacities for world instantiation."""

    max_triples: int
    max_self_triples: int

    def as_dict(self) -> dict[str, int]:
        return {
            "max_triples": int(self.max_triples),
            "max_self_triples": int(self.max_self_triples),
        }


@dataclass(frozen=True)
class CollisionCountSummary:
    """Observed collision counts over a terrain collection."""

    terrain_count: int
    max_candidate_pairs: int
    max_triples: int
    max_self_triples: int

    def as_dict(self) -> dict[str, int]:
        return {
            "terrain_count": int(self.terrain_count),
            "max_candidate_pairs": int(self.max_candidate_pairs),
            "max_triples": int(self.max_triples),
            "max_self_triples": int(self.max_self_triples),
        }


@dataclass(frozen=True)
class ResolvedCollisionTuning:
    """Resolved runtime and build-time collision capacities."""

    mode: str
    build_capacity: CollisionBuildCapacity
    runtime_build_capacity: CollisionBuildCapacity
    dynamic_max_candidates: int
    dynamic_max_narrow: int
    suite_counts: CollisionCountSummary
    synthetic_counts: CollisionCountSummary
    runtime_counts: CollisionCountSummary

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": str(self.mode),
            "build_max_triples": int(self.build_capacity.max_triples),
            "build_max_self_triples": int(self.build_capacity.max_self_triples),
            "runtime_build_max_triples": int(self.runtime_build_capacity.max_triples),
            "runtime_build_max_self_triples": int(self.runtime_build_capacity.max_self_triples),
            "dynamic_max_candidates": int(self.dynamic_max_candidates),
            "dynamic_max_narrow": int(self.dynamic_max_narrow),
            "suite": self.suite_counts.as_dict(),
            "synthetic": self.synthetic_counts.as_dict(),
            "runtime": self.runtime_counts.as_dict(),
        }


def resolve_collision_tuning_from_terrains(
    terrain_grids: list[np.ndarray],
    *,
    grid_h: int,
    grid_w: int,
    robot_morph: np.ndarray,
    spawn_x: int,
    spawn_y: int,
    mode: str,
) -> ResolvedCollisionTuning:
    """Resolve tuned collision capacities from real terrain grids plus a synthetic probe."""
    if mode != "hybrid_suite_probe":
        raise ValueError(
            "collision.tuning.mode must be 'hybrid_suite_probe' for sensing_edges"
        )
    if not terrain_grids:
        raise ValueError("Collision tuning requires at least one terrain grid")

    suite_counts = summarize_collision_counts_from_terrains(
        terrain_grids,
        grid_h=grid_h,
        grid_w=grid_w,
        robot_morph=robot_morph,
        spawn_x=spawn_x,
        spawn_y=spawn_y,
    )
    synthetic_counts = synthetic_collision_probe(
        grid_h=grid_h,
        grid_w=grid_w,
        robot_morph=robot_morph,
        spawn_x=spawn_x,
        spawn_y=spawn_y,
    )
    runtime_counts = runtime_collision_probe(
        grid_h=grid_h,
        grid_w=grid_w,
        robot_morph=robot_morph,
        spawn_x=spawn_x,
        spawn_y=spawn_y,
    )

    observed_pairs = max(suite_counts.max_candidate_pairs, synthetic_counts.max_candidate_pairs)
    observed_triples = max(suite_counts.max_triples, synthetic_counts.max_triples)
    observed_self = max(suite_counts.max_self_triples, synthetic_counts.max_self_triples)

    build_capacity = CollisionBuildCapacity(
        max_triples=_round_up(observed_triples * 1.10 + 128.0, 256),
        max_self_triples=_round_up(observed_self * 1.10 + 64.0, 64),
    )
    runtime_build_capacity = CollisionBuildCapacity(
        max_triples=_round_up(runtime_counts.max_triples * 1.10 + 128.0, 256),
        max_self_triples=_round_up(runtime_counts.max_self_triples * 1.10 + 64.0, 64),
    )
    dynamic_max_candidates = _round_up(runtime_counts.max_candidate_pairs * 1.10, 64)
    dynamic_max_narrow = min(
        runtime_build_capacity.max_triples,
        max(1, 4 * dynamic_max_candidates),
    )

    return ResolvedCollisionTuning(
        mode=str(mode),
        build_capacity=build_capacity,
        runtime_build_capacity=runtime_build_capacity,
        dynamic_max_candidates=int(dynamic_max_candidates),
        dynamic_max_narrow=int(dynamic_max_narrow),
        suite_counts=suite_counts,
        synthetic_counts=synthetic_counts,
        runtime_counts=runtime_counts,
    )


def resolve_collision_tuning_from_env_datas(
    env_datas,
    *,
    grid_h: int,
    grid_w: int,
    robot_morph: np.ndarray,
    spawn_x: int,
    spawn_y: int,
    mode: str,
) -> ResolvedCollisionTuning:
    """Resolve runtime collision capacities from already-built environments.

    This fallback only has access to split runtime env payloads, so the returned
    `build_capacity` is set to the same value as `runtime_build_capacity`.
    """
    if mode != "hybrid_suite_probe":
        raise ValueError(
            "collision.tuning.mode must be 'hybrid_suite_probe' for sensing_edges"
        )
    if not env_datas:
        raise ValueError("Collision tuning requires at least one environment payload")

    suite_counts = CollisionCountSummary(
        terrain_count=len(env_datas),
        max_candidate_pairs=max(int(np.asarray(env_data.collision_data.n_candidate_pairs).item()) for env_data in env_datas),
        max_triples=max(int(np.asarray(env_data.collision_data.n_triples).item()) for env_data in env_datas),
        max_self_triples=max(int(np.asarray(env_data.collision_data.n_self_triples).item()) for env_data in env_datas),
    )
    synthetic_counts = synthetic_collision_probe(
        grid_h=grid_h,
        grid_w=grid_w,
        robot_morph=robot_morph,
        spawn_x=spawn_x,
        spawn_y=spawn_y,
    )
    runtime_build_capacity = CollisionBuildCapacity(
        max_triples=_round_up(suite_counts.max_triples * 1.10 + 128.0, 256),
        max_self_triples=_round_up(suite_counts.max_self_triples * 1.10 + 64.0, 64),
    )
    dynamic_max_candidates = _round_up(suite_counts.max_candidate_pairs * 1.10, 64)
    dynamic_max_narrow = min(
        runtime_build_capacity.max_triples,
        max(1, 4 * dynamic_max_candidates),
    )
    return ResolvedCollisionTuning(
        mode=str(mode),
        build_capacity=runtime_build_capacity,
        runtime_build_capacity=runtime_build_capacity,
        dynamic_max_candidates=int(dynamic_max_candidates),
        dynamic_max_narrow=int(dynamic_max_narrow),
        suite_counts=suite_counts,
        synthetic_counts=synthetic_counts,
        runtime_counts=suite_counts,
    )


def summarize_collision_counts_from_terrains(
    terrain_grids: list[np.ndarray],
    *,
    grid_h: int,
    grid_w: int,
    robot_morph: np.ndarray,
    spawn_x: int,
    spawn_y: int,
) -> CollisionCountSummary:
    """Summarize real collision counts over a terrain collection."""
    max_candidate_pairs = 0
    max_triples = 0
    max_self_triples = 0
    for terrain_grid in terrain_grids:
        collision_data = collision_data_from_terrain(
            terrain_grid,
            grid_h=grid_h,
            grid_w=grid_w,
            robot_morph=robot_morph,
            spawn_x=spawn_x,
            spawn_y=spawn_y,
        )
        max_candidate_pairs = max(
            max_candidate_pairs,
            int(np.asarray(collision_data.n_candidate_pairs).item()),
        )
        max_triples = max(
            max_triples,
            int(np.asarray(collision_data.n_triples).item()),
        )
        max_self_triples = max(
            max_self_triples,
            int(np.asarray(collision_data.n_self_triples).item()),
        )
    return CollisionCountSummary(
        terrain_count=len(terrain_grids),
        max_candidate_pairs=int(max_candidate_pairs),
        max_triples=int(max_triples),
        max_self_triples=int(max_self_triples),
    )


def synthetic_collision_probe(
    *,
    grid_h: int,
    grid_w: int,
    robot_morph: np.ndarray,
    spawn_x: int,
    spawn_y: int,
) -> CollisionCountSummary:
    """Construct a dense rigid terrain occupancy probe and summarize its counts."""
    terrain_grid = np.full((grid_h, grid_w), C.RIGID, dtype=np.int32)
    collision_data = collision_data_from_terrain(
        terrain_grid,
        grid_h=grid_h,
        grid_w=grid_w,
        robot_morph=robot_morph,
        spawn_x=spawn_x,
        spawn_y=spawn_y,
    )
    return CollisionCountSummary(
        terrain_count=1,
        max_candidate_pairs=int(np.asarray(collision_data.n_candidate_pairs).item()),
        max_triples=int(np.asarray(collision_data.n_triples).item()),
        max_self_triples=int(np.asarray(collision_data.n_self_triples).item()),
    )


def runtime_collision_probe(
    *,
    grid_h: int,
    grid_w: int,
    robot_morph: np.ndarray,
    spawn_x: int,
    spawn_y: int,
) -> CollisionCountSummary:
    """Summarize collision counts for the robot-only runtime world."""
    empty_terrain = np.full((grid_h, grid_w), C.EMPTY, dtype=np.int32)
    collision_data = collision_data_from_terrain(
        empty_terrain,
        grid_h=grid_h,
        grid_w=grid_w,
        robot_morph=robot_morph,
        spawn_x=spawn_x,
        spawn_y=spawn_y,
    )
    return CollisionCountSummary(
        terrain_count=1,
        max_candidate_pairs=int(np.asarray(collision_data.n_candidate_pairs).item()),
        max_triples=int(np.asarray(collision_data.n_triples).item()),
        max_self_triples=int(np.asarray(collision_data.n_self_triples).item()),
    )


def collision_data_from_terrain(
    terrain_grid: np.ndarray,
    *,
    grid_h: int,
    grid_w: int,
    robot_morph: np.ndarray,
    spawn_x: int,
    spawn_y: int,
) -> CollisionData:
    """Build collision metadata for a terrain + fixed robot placement."""
    composited, robot_mask = composited_collision_grid(
        terrain_grid,
        robot_morph,
        spawn_x=spawn_x,
        spawn_y=spawn_y,
    )
    return make_collision_data(
        composited,
        grid_h=grid_h,
        grid_w=grid_w,
        robot_cell_mask=robot_mask,
    )


def composited_collision_grid(
    terrain_grid: np.ndarray,
    robot_morph: np.ndarray,
    *,
    spawn_x: int,
    spawn_y: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose the fixed terrain and robot footprint into one collision grid."""
    terrain_np = np.asarray(terrain_grid, dtype=np.int32)
    robot_np = np.asarray(robot_morph, dtype=np.int32)
    composited = terrain_np.copy()
    robot_mask = np.zeros_like(composited, dtype=bool)
    robot_h, robot_w = robot_np.shape

    for ry in range(robot_h):
        for rx in range(robot_w):
            if robot_np[ry, rx] == C.EMPTY:
                continue
            grid_y = int(spawn_y + ry)
            grid_x = int(spawn_x + rx)
            if not (0 <= grid_y < composited.shape[0] and 0 <= grid_x < composited.shape[1]):
                raise ValueError("Robot footprint lies outside the collision grid")
            composited[grid_y, grid_x] = robot_np[ry, rx]
            robot_mask[grid_y, grid_x] = True
    return composited, robot_mask


def _round_up(value: float, quantum: int) -> int:
    quantum_int = int(quantum)
    if quantum_int <= 0:
        raise ValueError("quantum must be positive")
    base = max(1, int(np.ceil(float(value))))
    return int(((base + quantum_int - 1) // quantum_int) * quantum_int)
