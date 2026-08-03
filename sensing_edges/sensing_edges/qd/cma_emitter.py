"""CMA-ES emitter for MAP-Elites quality-diversity evolution.

Wraps EvolutionEngine with CMA-MAE-compatible restart logic.

Provenance: copied from experiments/gradient_ascent/gradient_ascent/qd/cma_emitter.py
  at commit 93baec2 on 2026-03-27. No modifications.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .evolution_engine import EvolutionEngine


class CMAEmitter:
    """CMA-ES emitter with archive-aware staleness restart criteria."""

    def __init__(
        self,
        genotype_length: int,
        population_size: int,
        strategy: str = "sep_cmaes",
        flat_generations: int | None = None,
        flat_tolerance: float | None = None,
        stale_generations: int = 5,
        max_age: int = 50,
        meaningful_improvement_eps: float = 0.5,
        std_epsilon: float = 1e-12,
        condition_cov_limit: float = 1.0e14,
        param_overrides: dict | None = None,
    ):
        self.engine = EvolutionEngine(
            genotype_length=genotype_length,
            population_size=population_size,
            strategy=strategy,
            maximize=True,
            param_overrides=param_overrides,
        )
        self.flat_generations = (
            None if flat_generations is None else int(flat_generations)
        )
        self.flat_tolerance = (
            None if flat_tolerance is None else float(flat_tolerance)
        )
        self.stale_generations = int(stale_generations)
        self.max_age = int(max_age)
        self.meaningful_improvement_eps = float(meaningful_improvement_eps)
        self.std_epsilon = float(std_epsilon)
        self.condition_cov_limit = float(condition_cov_limit)
        self.param_overrides = dict(param_overrides) if param_overrides else None
        self.stale_counter = 0
        self.age_since_restart = 0
        self._last_improvements: np.ndarray | None = None
        self._last_statuses: np.ndarray | None = None
        self.total_restarts = 0
        self.total_restarts_by_reason = {
            "stale": 0,
            "es_stop": 0,
            "age_cap": 0,
        }
        self.last_restart_reason: str | None = None
        self.es_state = None

    def init(self, key: jax.Array, init_mean: jax.Array | None = None):
        self.es_state = self.engine.init(key, mean_init=init_mean)
        self.stale_counter = 0
        self.age_since_restart = 0
        self._last_improvements = None
        self._last_statuses = None
        self.last_restart_reason = None

    def ask(self, key: jax.Array) -> jax.Array:
        population, self.es_state = self.engine.ask(key, self.es_state)
        return population

    def tell(
        self,
        key: jax.Array,
        population: jax.Array,
        improvements: np.ndarray,
        statuses: np.ndarray | None = None,
    ):
        fitness = jnp.array(improvements, dtype=jnp.float32)
        self.es_state, _ = self.engine.tell(key, population, fitness, self.es_state)

        imp = np.array(improvements, dtype=np.float64)
        self._last_improvements = imp
        status_arr = None
        if statuses is not None:
            status_arr = np.asarray(statuses, dtype=np.uint8).reshape(-1)
            if status_arr.shape != imp.shape:
                raise ValueError(
                    f"statuses must match improvements shape {imp.shape}, got {status_arr.shape}"
                )
        self._last_statuses = status_arr

        accepted_mask = imp > self.meaningful_improvement_eps
        if status_arr is not None:
            accepted_mask &= status_arr > 0

        if np.any(accepted_mask):
            self.stale_counter = 0
        else:
            self.stale_counter += 1
        self.age_since_restart += 1

    def _sigma_collapse(self) -> bool:
        sigma = getattr(self.es_state, "std", None)
        if sigma is None:
            return False
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        if sigma_arr.size == 0:
            return False
        finite_sigma = sigma_arr[np.isfinite(sigma_arr)]
        if finite_sigma.size == 0:
            return False
        return float(np.max(finite_sigma)) <= self.std_epsilon

    def _condition_cov_exceeded(self) -> bool:
        covariance = getattr(self.es_state, "C", None)
        if covariance is None:
            return False
        cov_arr = np.asarray(covariance, dtype=np.float64).reshape(-1)
        if cov_arr.size == 0:
            return False
        if not np.all(np.isfinite(cov_arr)):
            return True
        positive = cov_arr[cov_arr > 0.0]
        if positive.size == 0:
            return False
        return float(np.max(positive) / np.min(positive)) >= self.condition_cov_limit

    @property
    def es_should_stop(self) -> bool:
        if self.es_state is None:
            return False
        return self._sigma_collapse() or self._condition_cov_exceeded()

    @property
    def restart_reason(self) -> str | None:
        if self.es_should_stop:
            return "es_stop"
        if self.stale_counter >= self.stale_generations:
            return "stale"
        if self.age_since_restart >= self.max_age:
            return "age_cap"
        return None

    @property
    def should_restart(self) -> bool:
        return self.restart_reason is not None

    @property
    def flat_counter(self) -> int:
        """Backward-compatible alias for legacy checkpoint/test code."""
        return self.stale_counter

    @flat_counter.setter
    def flat_counter(self, value: int) -> None:
        self.stale_counter = int(value)

    def restart(
        self,
        key: jax.Array,
        new_mean: jax.Array | None = None,
        *,
        reason: str | None = None,
    ):
        resolved_reason = self.restart_reason if reason is None else str(reason)
        self.es_state = self.engine.init(key, mean_init=new_mean)
        self.stale_counter = 0
        self.age_since_restart = 0
        self._last_improvements = None
        self._last_statuses = None
        self.total_restarts += 1
        self.last_restart_reason = resolved_reason
        if resolved_reason in self.total_restarts_by_reason:
            self.total_restarts_by_reason[resolved_reason] += 1
