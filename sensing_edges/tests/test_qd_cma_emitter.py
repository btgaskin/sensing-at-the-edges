from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sensing_edges.qd.cma_emitter import CMAEmitter


def _make_emitter(**overrides) -> CMAEmitter:
    kwargs = {
        "genotype_length": 8,
        "population_size": 6,
        "strategy": "sep_cmaes",
        "stale_generations": 2,
        "max_age": 10,
        "meaningful_improvement_eps": 0.5,
        "std_epsilon": 1.0e-6,
    }
    kwargs.update(overrides)
    emitter = CMAEmitter(**kwargs)
    emitter.init(jax.random.PRNGKey(0))
    return emitter


def _tell_once(
    emitter: CMAEmitter,
    *,
    key_seed: int,
    improvements: np.ndarray,
    statuses: np.ndarray,
) -> None:
    key = jax.random.PRNGKey(key_seed)
    ask_key, tell_key = jax.random.split(key)
    population = emitter.ask(ask_key)
    emitter.tell(tell_key, population, improvements, statuses)


def test_stale_counter_uses_accepted_post_mask_improvements():
    emitter = _make_emitter()

    _tell_once(
        emitter,
        key_seed=1,
        improvements=np.full(6, 2.0, dtype=np.float64),
        statuses=np.zeros(6, dtype=np.uint8),
    )
    assert emitter.stale_counter == 1

    statuses = np.zeros(6, dtype=np.uint8)
    statuses[0] = 1
    improvements = np.full(6, 0.1, dtype=np.float64)
    improvements[0] = 0.75
    _tell_once(
        emitter,
        key_seed=2,
        improvements=improvements,
        statuses=statuses,
    )
    assert emitter.stale_counter == 0
    assert emitter.age_since_restart == 2


def test_restart_reason_can_be_stale_or_age_cap():
    stale_emitter = _make_emitter(stale_generations=2, max_age=10)
    _tell_once(
        stale_emitter,
        key_seed=3,
        improvements=np.zeros(6, dtype=np.float64),
        statuses=np.zeros(6, dtype=np.uint8),
    )
    _tell_once(
        stale_emitter,
        key_seed=4,
        improvements=np.zeros(6, dtype=np.float64),
        statuses=np.zeros(6, dtype=np.uint8),
    )
    assert stale_emitter.restart_reason == "stale"

    age_emitter = _make_emitter(stale_generations=10, max_age=1)
    _tell_once(
        age_emitter,
        key_seed=5,
        improvements=np.zeros(6, dtype=np.float64),
        statuses=np.zeros(6, dtype=np.uint8),
    )
    assert age_emitter.restart_reason == "age_cap"


def test_restart_reason_uses_es_stop_proxy():
    emitter = _make_emitter(std_epsilon=1.0e-6)
    fields = {
        name: getattr(emitter.es_state, name)
        for name in emitter.es_state.__dataclass_fields__
    }
    fields["std"] = jnp.zeros_like(jnp.asarray(fields["std"]))
    emitter.es_state = type(emitter.es_state)(**fields)

    assert emitter.es_should_stop is True
    assert emitter.restart_reason == "es_stop"
