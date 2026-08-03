"""Multi-strategy evolution engine wrapping evosax v0.2.0.

Provenance: copied from experiments/gradient_ascent/gradient_ascent/qd/evolution_engine.py
  at commit 93baec2 on 2026-03-27. No modifications.
"""

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from evosax.algorithms import CMA_ES, CR_FM_NES, PGPE, Open_ES, SNES, Sep_CMA_ES
from evosax.core.optimizer import clipup

STRATEGIES = {
    "pgpe": PGPE,
    "cmaes": CMA_ES,
    "sep_cmaes": Sep_CMA_ES,
    "snes": SNES,
    "open_es": Open_ES,
    "cr_fm_nes": CR_FM_NES,
}

_FITNESS_METRIC_KEYS = ("best_fitness", "best_fitness_in_generation")


class EvolutionEngine:
    """Multi-strategy evolutionary search via evosax v0.2.0 ask/tell API."""

    def __init__(
        self,
        genotype_length: int,
        population_size: int,
        strategy: str = "pgpe",
        maximize: bool = True,
        clipup_lr: float = 0.2,
        clipup_max_velocity: float = 0.4,
        param_overrides: Mapping[str, Any] | None = None,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}, expected one of {list(STRATEGIES)}"
            )
        if genotype_length < 1:
            raise ValueError(f"genotype_length must be >= 1, got {genotype_length}")
        if population_size < 2:
            raise ValueError(f"population_size must be >= 2, got {population_size}")
        if strategy in ("pgpe", "open_es") and population_size % 2 != 0:
            raise ValueError(
                f"{strategy.upper()} requires even population_size "
                f"(antithetic sampling), got {population_size}"
            )

        self.genotype_length = genotype_length
        self.population_size = population_size
        self.strategy = strategy
        self.maximize = maximize

        solution = jnp.zeros(genotype_length)
        es_cls = STRATEGIES[strategy]

        if strategy == "pgpe":
            self.es = es_cls(
                population_size=population_size,
                solution=solution,
                optimizer=clipup(
                    learning_rate=clipup_lr,
                    max_velocity=clipup_max_velocity,
                ),
            )
        elif strategy == "open_es":
            import optax

            lr = 0.01
            std_init = 1.0
            if param_overrides:
                param_overrides = dict(param_overrides)
                lr = param_overrides.pop("learning_rate", lr)
                std_init = param_overrides.pop("std_init", std_init)
                if param_overrides:
                    raise ValueError(
                        f"Unknown Open_ES param overrides: "
                        f"{list(param_overrides)}. "
                        f"Valid keys: ['learning_rate', 'std_init']"
                    )
            param_overrides = None
            self.es = es_cls(
                population_size=population_size,
                solution=solution,
                optimizer=optax.adam(learning_rate=lr),
                std_schedule=optax.constant_schedule(std_init),
            )
        else:
            self.es = es_cls(
                population_size=population_size,
                solution=solution,
            )

        self.params = self.es.default_params
        if param_overrides:
            try:
                self.params = self.params.replace(**param_overrides)
            except TypeError as e:
                valid = [f.name for f in self.params.__dataclass_fields__.values()]
                raise ValueError(
                    f"Invalid param override for {strategy}: {e}. "
                    f"Valid params: {valid}"
                ) from e

    def init(self, key: jax.Array, mean_init: jax.Array | None = None):
        if mean_init is None:
            mean_init = jnp.zeros(self.genotype_length)
        return self.es.init(key, mean_init, self.params)

    def ask(self, key: jax.Array, state):
        return self.es.ask(key, state, self.params)

    def tell(self, key: jax.Array, population: jax.Array, fitness: jax.Array, state):
        es_fitness = -fitness if self.maximize else fitness
        state, metrics = self.es.tell(key, population, es_fitness, state, self.params)
        if self.maximize:
            metrics = {
                k: (-v if k in _FITNESS_METRIC_KEYS else v)
                for k, v in metrics.items()
            }
        return state, metrics

    @property
    def strategy_name(self) -> str:
        return self.strategy

    def get_best(self, state) -> tuple[jax.Array, jax.Array]:
        best_fitness = state.best_fitness
        if self.maximize:
            best_fitness = -best_fitness
        return state.best_solution, best_fitness
