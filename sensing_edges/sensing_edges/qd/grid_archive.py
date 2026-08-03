"""N-dimensional grid archive for MAP-Elites quality-diversity.

Implements CMA-MAE threshold updates with finite improvement scoring suitable
for CMA-ES ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class GridArchiveBatchDecision:
    """Scored archive update for a batch before elite payloads are committed."""

    improvements: np.ndarray
    statuses: np.ndarray
    accepted_indices: np.ndarray
    accepted_flat_indices: np.ndarray
    accepted_fitnesses: np.ndarray
    accepted_statuses: np.ndarray
    accepted_thresholds: np.ndarray
    accepted_pre_fitnesses: np.ndarray
    descriptor_finite_mask: np.ndarray
    invalid_descriptor_count: int


class GridArchive:
    """N-dimensional grid archive mapping behavioral descriptors to elite genotypes.

    Each cell stores a single elite (genotype + fitness). Improvement scoring
    uses finite values suitable for CMA-ES ranking.
    """

    NOT_ADDED = 0
    IMPROVE = 1
    NEW = 2

    def __init__(
        self,
        dims: tuple[int, ...],
        ranges: list[tuple[float, float]],
        learning_rate: float = 0.01,
        threshold_min: float = 0.0,
        descriptor_names: tuple[str, ...] | None = None,
        descriptor_labels: tuple[str, ...] | None = None,
    ):
        if len(dims) != len(ranges):
            raise ValueError(
                f"dims ({len(dims)}) and ranges ({len(ranges)}) must have same length"
            )
        if not (0.0 <= learning_rate <= 1.0):
            raise ValueError(f"learning_rate must be in [0.0, 1.0], got {learning_rate}")

        self.dims = tuple(dims)
        self.ranges = list(ranges)
        self.learning_rate = float(learning_rate)
        self.threshold_min = float(threshold_min)
        self._n_dims = len(dims)
        self._total_cells = int(np.prod(dims))
        default_names = tuple(f"descriptor_{index + 1}" for index in range(self._n_dims))
        resolved_names = tuple(descriptor_names) if descriptor_names is not None else default_names
        if len(resolved_names) != self._n_dims:
            raise ValueError(
                "descriptor_names must match archive dimensionality "
                f"({self._n_dims}), got {len(resolved_names)}"
            )
        default_labels = tuple(_format_descriptor_label(name) for name in resolved_names)
        resolved_labels = (
            tuple(descriptor_labels) if descriptor_labels is not None else default_labels
        )
        if len(resolved_labels) != self._n_dims:
            raise ValueError(
                "descriptor_labels must match archive dimensionality "
                f"({self._n_dims}), got {len(resolved_labels)}"
            )
        self.descriptor_names = tuple(str(name) for name in resolved_names)
        self.descriptor_labels = tuple(str(label) for label in resolved_labels)

        self._fitnesses = np.full(self._total_cells, -np.inf, dtype=np.float64)
        self._thresholds = np.full(self._total_cells, self.threshold_min, dtype=np.float64)
        self._genotypes: dict[int, np.ndarray] = {}
        self._descriptors: dict[int, np.ndarray] = {}

    def _to_cell(self, descriptor: np.ndarray) -> tuple[int, ...]:
        indices = []
        for i in range(self._n_dims):
            val = np.float64(descriptor[i])
            if not np.isfinite(val):
                indices.append(0)
                continue
            lo, hi = self.ranges[i]
            normed = (val - lo) / max(hi - lo, 1e-8)
            normed = np.clip(normed, 0.0, 1.0 - 1e-8)
            idx = min(int(normed * self.dims[i]), self.dims[i] - 1)
            indices.append(idx)
        return tuple(indices)

    def _ravel(self, cell: tuple[int, ...]) -> int:
        return int(np.ravel_multi_index(cell, self.dims))

    def _to_cells_array(self, descriptors: np.ndarray) -> np.ndarray:
        """Vectorized descriptor-to-cell mapping for a batch."""
        desc = np.asarray(descriptors, dtype=np.float64)
        if desc.ndim != 2 or desc.shape[1] != self._n_dims:
            raise ValueError(
                f"descriptors must have shape (N, {self._n_dims}), got {desc.shape}"
            )
        if desc.shape[0] == 0:
            return np.zeros((0, self._n_dims), dtype=np.int64)

        indices = np.zeros((desc.shape[0], self._n_dims), dtype=np.int64)
        for dim_idx in range(self._n_dims):
            values = desc[:, dim_idx]
            finite_mask = np.isfinite(values)
            lo, hi = self.ranges[dim_idx]
            safe_values = np.where(finite_mask, values, lo)
            normed = (safe_values - lo) / max(hi - lo, 1e-8)
            normed = np.clip(normed, 0.0, 1.0 - 1e-8)
            dim_cells = np.minimum(
                (normed * self.dims[dim_idx]).astype(np.int64),
                self.dims[dim_idx] - 1,
            )
            indices[:, dim_idx] = np.where(finite_mask, dim_cells, 0)
        return indices

    def _score_batch(
        self,
        fitnesses: np.ndarray,
        descriptors: np.ndarray,
        active_cell_mask: np.ndarray | None = None,
    ) -> GridArchiveBatchDecision:
        """Score a batch against the current archive without copying genotypes."""
        fitness_arr = np.asarray(fitnesses, dtype=np.float64).reshape(-1)
        desc_arr = np.asarray(descriptors, dtype=np.float64)
        if desc_arr.ndim != 2 or desc_arr.shape[0] != fitness_arr.shape[0]:
            raise ValueError(
                "descriptors must have shape (N, n_bd) aligned with fitnesses; "
                f"got fitnesses={fitness_arr.shape}, descriptors={desc_arr.shape}"
            )

        n = int(fitness_arr.shape[0])
        improvements = np.zeros(n, dtype=np.float64)
        statuses = np.full(n, self.NOT_ADDED, dtype=np.uint8)
        if n == 0:
            empty_i64 = np.zeros((0,), dtype=np.int64)
            empty_f64 = np.zeros((0,), dtype=np.float64)
            empty_u8 = np.zeros((0,), dtype=np.uint8)
            empty_bool = np.zeros((0,), dtype=bool)
            return GridArchiveBatchDecision(
                improvements=improvements,
                statuses=statuses,
                accepted_indices=empty_i64,
                accepted_flat_indices=empty_i64,
                accepted_fitnesses=empty_f64,
                accepted_statuses=empty_u8,
                accepted_thresholds=empty_f64,
                accepted_pre_fitnesses=empty_f64,
                descriptor_finite_mask=empty_bool,
                invalid_descriptor_count=0,
            )

        descriptor_finite_mask = np.all(np.isfinite(desc_arr), axis=1)
        invalid_descriptor_count = int(np.count_nonzero(~descriptor_finite_mask))
        cells = self._to_cells_array(desc_arr)
        flat_indices = np.ravel_multi_index(cells.T, self.dims)
        thresholds = self._thresholds[flat_indices]
        finite_mask = np.isfinite(fitness_arr)
        valid_mask = finite_mask & descriptor_finite_mask
        if active_cell_mask is not None:
            active_cell_mask = np.asarray(active_cell_mask, dtype=bool).reshape(-1)
            if active_cell_mask.shape != (self._total_cells,):
                raise ValueError(
                    "active_cell_mask must have shape "
                    f"({self._total_cells},), got {active_cell_mask.shape}"
                )
        improvements[valid_mask] = fitness_arr[valid_mask] - thresholds[valid_mask]

        # Focus window: zero improvements and block eligibility for inactive cells
        if active_cell_mask is not None:
            inactive = valid_mask & ~active_cell_mask[flat_indices]
            improvements[inactive] = 0.0

        eligible_mask = valid_mask & (fitness_arr > thresholds)
        if active_cell_mask is not None:
            eligible_mask &= active_cell_mask[flat_indices]
        eligible_indices = np.flatnonzero(eligible_mask)
        if eligible_indices.size == 0:
            empty_i64 = np.zeros((0,), dtype=np.int64)
            empty_f64 = np.zeros((0,), dtype=np.float64)
            empty_u8 = np.zeros((0,), dtype=np.uint8)
            return GridArchiveBatchDecision(
                improvements=improvements,
                statuses=statuses,
                accepted_indices=empty_i64,
                accepted_flat_indices=empty_i64,
                accepted_fitnesses=empty_f64,
                accepted_statuses=empty_u8,
                accepted_thresholds=empty_f64,
                accepted_pre_fitnesses=empty_f64,
                descriptor_finite_mask=descriptor_finite_mask,
                invalid_descriptor_count=invalid_descriptor_count,
            )

        eligible_flat = flat_indices[eligible_indices]
        eligible_fitness = fitness_arr[eligible_indices]
        order = np.argsort(eligible_flat, kind="stable")
        eligible_indices = eligible_indices[order]
        eligible_flat = eligible_flat[order]
        eligible_fitness = eligible_fitness[order]

        unique_flats, group_starts = np.unique(eligible_flat, return_index=True)
        accepted_indices: list[int] = []
        accepted_flat_indices: list[int] = []
        accepted_fitnesses: list[float] = []
        accepted_statuses: list[int] = []
        accepted_thresholds: list[float] = []
        accepted_pre_fitnesses: list[float] = []

        for group_idx, flat_idx in enumerate(unique_flats.tolist()):
            start = int(group_starts[group_idx])
            end = (
                int(group_starts[group_idx + 1])
                if group_idx + 1 < len(group_starts)
                else int(len(eligible_indices))
            )
            group_indices = eligible_indices[start:end]
            group_fitness = eligible_fitness[start:end]
            winner_pos = int(np.argmax(group_fitness))
            winner = int(group_indices[winner_pos])
            winner_fitness = float(fitness_arr[winner])
            incumbent = float(self._fitnesses[flat_idx])
            status = self.NEW if np.isneginf(incumbent) else self.IMPROVE

            accepted_indices.append(winner)
            accepted_flat_indices.append(int(flat_idx))
            accepted_fitnesses.append(winner_fitness)
            accepted_statuses.append(int(status))
            accepted_thresholds.append(
                (1.0 - self.learning_rate) * float(self._thresholds[flat_idx])
                + self.learning_rate * winner_fitness
            )
            accepted_pre_fitnesses.append(incumbent)
            statuses[winner] = status

        return GridArchiveBatchDecision(
            improvements=improvements,
            statuses=statuses,
            accepted_indices=np.asarray(accepted_indices, dtype=np.int64),
            accepted_flat_indices=np.asarray(accepted_flat_indices, dtype=np.int64),
            accepted_fitnesses=np.asarray(accepted_fitnesses, dtype=np.float64),
            accepted_statuses=np.asarray(accepted_statuses, dtype=np.uint8),
            accepted_thresholds=np.asarray(accepted_thresholds, dtype=np.float64),
            accepted_pre_fitnesses=np.asarray(accepted_pre_fitnesses, dtype=np.float64),
            descriptor_finite_mask=descriptor_finite_mask,
            invalid_descriptor_count=invalid_descriptor_count,
        )

    def score_batch(
        self,
        fitnesses: np.ndarray,
        descriptors: np.ndarray,
        active_cell_mask: np.ndarray | None = None,
    ) -> GridArchiveBatchDecision:
        """Score a batch and return archive acceptance decisions."""
        return self._score_batch(fitnesses, descriptors, active_cell_mask=active_cell_mask)

    def commit_scored_batch(
        self,
        decision: GridArchiveBatchDecision,
        genotypes: np.ndarray,
        descriptors: np.ndarray,
    ) -> None:
        """Commit previously scored winners to archive storage."""
        genotypes_arr = np.asarray(genotypes)
        descriptors_arr = np.asarray(descriptors, dtype=np.float64)
        n_accept = int(decision.accepted_indices.shape[0])
        if genotypes_arr.shape[0] != n_accept:
            raise ValueError(
                f"genotypes must have {n_accept} rows, got {genotypes_arr.shape[0]}"
            )
        if descriptors_arr.ndim != 2 or descriptors_arr.shape[0] != n_accept:
            raise ValueError(
                "descriptors must align with accepted winners; "
                f"got {descriptors_arr.shape}"
            )
        if n_accept == 0:
            return

        self._thresholds[decision.accepted_flat_indices] = decision.accepted_thresholds
        for row_idx, flat_idx in enumerate(decision.accepted_flat_indices.tolist()):
            winner_fitness = float(decision.accepted_fitnesses[row_idx])
            incumbent = float(decision.accepted_pre_fitnesses[row_idx])
            if winner_fitness > incumbent:
                self._fitnesses[flat_idx] = winner_fitness
                self._genotypes[flat_idx] = np.array(genotypes_arr[row_idx])
                self._descriptors[flat_idx] = np.array(descriptors_arr[row_idx])

    def add(
        self,
        genotypes: np.ndarray,
        fitnesses: np.ndarray,
        descriptors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        decision = self._score_batch(fitnesses, descriptors)
        if decision.accepted_indices.size > 0:
            genotypes_arr = np.asarray(genotypes)
            descriptors_arr = np.asarray(descriptors, dtype=np.float64)
            winner_rows = decision.accepted_indices
            self.commit_scored_batch(
                decision,
                genotypes_arr[winner_rows],
                descriptors_arr[winner_rows],
            )
        return decision.improvements, decision.statuses

    def _resolve_occupied_indices(
        self,
        *,
        cell_mask: np.ndarray | None = None,
    ) -> list[int]:
        occupied = list(self._genotypes.keys())
        if cell_mask is not None:
            cell_mask = np.asarray(cell_mask, dtype=bool).reshape(-1)
            if cell_mask.shape != (self._total_cells,):
                raise ValueError(
                    f"cell_mask must have shape ({self._total_cells},), got {cell_mask.shape}"
                )
            occupied = [i for i in occupied if cell_mask[i]]
        if not occupied:
            raise ValueError("Cannot sample from empty archive")
        return occupied

    def sample_elites(
        self, key: jax.Array, n: int, cell_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        occupied = self._resolve_occupied_indices(cell_mask=cell_mask)
        idx = jax.random.choice(key, len(occupied), shape=(n,), replace=True)
        idx_np = np.array(idx)
        return np.stack([self._genotypes[occupied[i]] for i in idx_np])

    def sample_frontier_elites(
        self,
        key: jax.Array,
        n: int,
        *,
        axis_directions: tuple[tuple[int, str], ...] | list[tuple[int, str]],
        cell_mask: np.ndarray | None = None,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Sample elites weighted toward emptier target-axis neighborhoods."""
        occupied = self._resolve_occupied_indices(cell_mask=cell_mask)
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        normalized_axis_directions: list[tuple[int, str]] = []
        for axis, direction in axis_directions:
            axis_int = int(axis)
            if axis_int < 0 or axis_int >= self._n_dims:
                raise ValueError(
                    f"axis {axis_int} out of range for archive with {self._n_dims} dimensions"
                )
            normalized_direction = str(direction).strip().lower()
            if normalized_direction not in {"min", "max"}:
                raise ValueError(
                    f"direction must be 'min' or 'max', got {direction!r}"
                )
            normalized_axis_directions.append((axis_int, normalized_direction))
        if not normalized_axis_directions:
            return self.sample_elites(key, n, cell_mask=cell_mask)

        target_axes = tuple(axis for axis, _ in normalized_axis_directions)
        projected_by_flat = {
            int(flat_idx): tuple(int(np.unravel_index(flat_idx, self.dims)[axis]) for axis in target_axes)
            for flat_idx in occupied
        }
        occupied_projected = set(projected_by_flat.values())

        active_projected: set[tuple[int, ...]] | None = None
        if cell_mask is not None:
            mask_arr = np.asarray(cell_mask, dtype=bool).reshape(-1)
            active_projected = {
                tuple(int(np.unravel_index(flat_idx, self.dims)[axis]) for axis in target_axes)
                for flat_idx in np.flatnonzero(mask_arr)
            }

        projected_scores: dict[tuple[int, ...], float] = {}
        for projected_cell in occupied_projected:
            empty_neighbors = 0
            total_neighbors = 0
            for delta in product((-1, 0, 1), repeat=len(target_axes)):
                if all(step == 0 for step in delta):
                    continue
                neighbor = tuple(coord + step for coord, step in zip(projected_cell, delta))
                valid = True
                for neighbor_coord, axis in zip(neighbor, target_axes):
                    if neighbor_coord < 0 or neighbor_coord >= int(self.dims[axis]):
                        valid = False
                        break
                if not valid:
                    continue
                if active_projected is not None and neighbor not in active_projected:
                    continue
                total_neighbors += 1
                if neighbor not in occupied_projected:
                    empty_neighbors += 1

            frontier_score = (
                float(empty_neighbors) / float(total_neighbors)
                if total_neighbors > 0
                else 0.0
            )
            corner_bonus_terms = []
            for coord, (axis, direction) in zip(projected_cell, normalized_axis_directions):
                dim = max(int(self.dims[axis]) - 1, 1)
                if direction == "max":
                    corner_bonus_terms.append(float(coord) / float(dim))
                else:
                    corner_bonus_terms.append(float(dim - coord) / float(dim))
            corner_bonus = (
                float(np.mean(corner_bonus_terms)) if corner_bonus_terms else 0.0
            )
            projected_scores[projected_cell] = frontier_score + 1.0e-3 * corner_bonus

        logits = np.array(
            [projected_scores[projected_by_flat[flat_idx]] for flat_idx in occupied],
            dtype=np.float64,
        ) / float(temperature)
        logits -= float(np.max(logits))
        probs = np.exp(logits)
        if not np.all(np.isfinite(probs)) or float(np.sum(probs)) <= 0.0:
            probs = np.ones_like(probs)
        probs /= float(np.sum(probs))

        idx = jax.random.choice(
            key,
            len(occupied),
            shape=(n,),
            replace=True,
            p=jnp.asarray(probs, dtype=jnp.float32),
        )
        idx_np = np.array(idx)
        return np.stack([self._genotypes[occupied[i]] for i in idx_np])

    def top_elites_by_fitness(self, n: int) -> list[dict[str, np.ndarray | float | int]]:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        ranked = sorted(
            self._genotypes.keys(),
            key=lambda flat_idx: (-float(self._fitnesses[flat_idx]), int(flat_idx)),
        )
        elites = []
        for flat_idx in ranked[:n]:
            elites.append(
                {
                    "flat_index": int(flat_idx),
                    "fitness": float(self._fitnesses[flat_idx]),
                    "genotype": np.array(self._genotypes[flat_idx]),
                    "descriptor": np.array(self._descriptors[flat_idx]),
                }
            )
        return elites

    def top_elites_in_projection(
        self,
        *,
        axis_cells: dict[int, int],
        n: int,
    ) -> list[dict[str, np.ndarray | float | int]]:
        """Return the top occupied elites inside one projected archive cell.

        ``axis_cells`` pins a subset of descriptor axes to explicit cell indices.
        Occupied elites that match all pinned coordinates are sorted by fitness,
        then by flat cell index for determinism.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        normalized_axis_cells: dict[int, int] = {}
        for axis, cell_idx in axis_cells.items():
            axis_int = int(axis)
            cell_int = int(cell_idx)
            if axis_int < 0 or axis_int >= self._n_dims:
                raise ValueError(
                    f"axis {axis_int} out of range for archive with {self._n_dims} dimensions"
                )
            if cell_int < 0 or cell_int >= int(self.dims[axis_int]):
                raise ValueError(
                    f"cell index {cell_int} out of range for axis {axis_int} with dim {self.dims[axis_int]}"
                )
            normalized_axis_cells[axis_int] = cell_int

        ranked = []
        for flat_idx in self._genotypes:
            cell = np.unravel_index(flat_idx, self.dims)
            if all(int(cell[axis]) == cell_idx for axis, cell_idx in normalized_axis_cells.items()):
                ranked.append(int(flat_idx))
        ranked.sort(key=lambda flat_idx: (-float(self._fitnesses[flat_idx]), int(flat_idx)))

        elites = []
        for flat_idx in ranked[:n]:
            elites.append(
                {
                    "flat_index": int(flat_idx),
                    "fitness": float(self._fitnesses[flat_idx]),
                    "genotype": np.array(self._genotypes[flat_idx]),
                    "descriptor": np.array(self._descriptors[flat_idx]),
                }
            )
        return elites

    def top_elites_by_descriptor_priority(
        self,
        *,
        axis_directions: tuple[tuple[int, str], ...] | list[tuple[int, str]],
        n: int,
    ) -> list[dict[str, np.ndarray | float | int]]:
        """Return occupied elites ranked lexicographically by descriptor priorities.

        ``axis_directions`` is an ordered sequence of ``(axis, direction)`` pairs
        where direction is ``"min"`` or ``"max"``. Earlier pairs dominate later
        ones. Fitness is used only as a tie-breaker after descriptor priorities.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        normalized_axis_directions: list[tuple[int, str]] = []
        for axis, direction in axis_directions:
            axis_int = int(axis)
            if axis_int < 0 or axis_int >= self._n_dims:
                raise ValueError(
                    f"axis {axis_int} out of range for archive with {self._n_dims} dimensions"
                )
            normalized_direction = str(direction).strip().lower()
            if normalized_direction not in {"min", "max"}:
                raise ValueError(
                    f"direction must be 'min' or 'max', got {direction!r}"
                )
            normalized_axis_directions.append((axis_int, normalized_direction))

        def _priority_key(flat_idx: int) -> tuple[float | int, ...]:
            descriptor = self._descriptors[flat_idx]
            key_parts: list[float | int] = []
            for axis_int, direction in normalized_axis_directions:
                value = float(descriptor[axis_int])
                if not np.isfinite(value):
                    key_parts.append(float("inf"))
                elif direction == "max":
                    key_parts.append(-value)
                else:
                    key_parts.append(value)
            key_parts.extend((-float(self._fitnesses[flat_idx]), int(flat_idx)))
            return tuple(key_parts)

        ranked = sorted(self._genotypes.keys(), key=_priority_key)

        elites = []
        for flat_idx in ranked[:n]:
            elites.append(
                {
                    "flat_index": int(flat_idx),
                    "fitness": float(self._fitnesses[flat_idx]),
                    "genotype": np.array(self._genotypes[flat_idx]),
                    "descriptor": np.array(self._descriptors[flat_idx]),
                }
            )
        return elites

    def state_dict(self) -> dict:
        occupied = sorted(self._genotypes.keys())
        return {
            "dims": self.dims,
            "ranges": self.ranges,
            "learning_rate": self.learning_rate,
            "threshold_min": self.threshold_min,
            "descriptor_names": self.descriptor_names,
            "descriptor_labels": self.descriptor_labels,
            "thresholds": np.array(self._thresholds, dtype=np.float64),
            "occupied_indices": np.array(occupied, dtype=np.int64),
            "fitnesses": np.array([self._fitnesses[i] for i in occupied], dtype=np.float64),
            "genotypes": np.stack([self._genotypes[i] for i in occupied]) if occupied else None,
            "descriptors": np.stack([self._descriptors[i] for i in occupied]) if occupied else None,
        }

    @classmethod
    def from_state_dict(cls, data: dict) -> "GridArchive":
        archive = cls(
            dims=tuple(data["dims"]),
            ranges=data["ranges"],
            learning_rate=float(data.get("learning_rate", 0.01)),
            threshold_min=float(data.get("threshold_min", 0.0)),
            descriptor_names=tuple(data["descriptor_names"]) if data.get("descriptor_names") else None,
            descriptor_labels=tuple(data["descriptor_labels"]) if data.get("descriptor_labels") else None,
        )
        thresholds = data.get("thresholds")
        if thresholds is not None:
            threshold_arr = np.array(thresholds, dtype=np.float64).reshape(-1)
            if threshold_arr.shape[0] != archive._total_cells:
                raise ValueError(
                    f"Invalid thresholds length {threshold_arr.shape[0]}, "
                    f"expected {archive._total_cells}"
                )
            archive._thresholds = threshold_arr

        if data["genotypes"] is not None:
            indices = data["occupied_indices"]
            for i, flat_idx in enumerate(indices):
                flat_idx = int(flat_idx)
                archive._fitnesses[flat_idx] = float(data["fitnesses"][i])
                archive._genotypes[flat_idx] = np.array(data["genotypes"][i])
                archive._descriptors[flat_idx] = np.array(data["descriptors"][i])
                if thresholds is None:
                    archive._thresholds[flat_idx] = float(data["fitnesses"][i])

        return archive

    @property
    def coverage(self) -> float:
        return len(self._genotypes) / self._total_cells

    @property
    def qd_score(self) -> float:
        if not self._genotypes:
            return 0.0
        return sum(self._fitnesses[i] for i in self._genotypes)

    @property
    def best_fitness(self) -> float:
        if not self._genotypes:
            return self.threshold_min
        return max(self._fitnesses[i] for i in self._genotypes)

    @property
    def size(self) -> int:
        return len(self._genotypes)

    @property
    def total_cells(self) -> int:
        return self._total_cells

    @property
    def occupied_flat_indices(self) -> set[int]:
        return set(self._genotypes.keys())

    @property
    def axis_occupied_bins(self) -> dict[str, int]:
        if not self._genotypes:
            return {name: 0 for name in self.descriptor_names}
        flat_indices = np.asarray(
            sorted(int(index) for index in self._genotypes),
            dtype=np.int64,
        )
        cells = np.stack(np.unravel_index(flat_indices, self.dims), axis=1)
        return {
            name: int(np.unique(cells[:, axis]).size)
            for axis, name in enumerate(self.descriptor_names)
        }

    @property
    def axis_coverage(self) -> dict[str, float]:
        occupied_bins = self.axis_occupied_bins
        return {
            name: float(occupied_bins[name]) / max(int(self.dims[axis]), 1)
            for axis, name in enumerate(self.descriptor_names)
        }

    @property
    def occupied_threshold_mean(self) -> float:
        if not self._genotypes:
            return self.threshold_min
        indices = np.fromiter(self._genotypes.keys(), dtype=np.int64, count=len(self._genotypes))
        return float(np.mean(self._thresholds[indices]))


def _format_descriptor_label(name: str) -> str:
    return str(name).replace("_", " ").title()
