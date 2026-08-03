"""Lightweight helpers for signed raw deformation range summaries."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


DEFORMATION_CHANNEL_KEYS = (
    "abs_h_delta",
    "abs_v_delta",
    "comp_h_delta",
    "comp_v_delta",
)


def zero_deformation_channel_max() -> jnp.ndarray:
    """Return the zero-initialized signed raw deformation-channel abs maxima."""
    return jnp.zeros((len(DEFORMATION_CHANNEL_KEYS),), dtype=jnp.float32)


def signed_deformation_deltas(deformation: jnp.ndarray) -> jnp.ndarray:
    """Convert dual deformation ratios into 4 signed deltas centered at zero."""
    deformation = jnp.asarray(deformation, dtype=jnp.float32)
    return deformation - jnp.float32(1.0)


def update_deformation_channel_max(
    current_max: jnp.ndarray,
    deformation: jnp.ndarray,
) -> jnp.ndarray:
    """Fold one signed raw deformation observation into a running abs-max."""
    signed_delta = signed_deformation_deltas(deformation)
    reduction_axes = tuple(range(max(signed_delta.ndim - 1, 0)))
    channel_max = jnp.max(jnp.abs(signed_delta), axis=reduction_axes)
    return jnp.maximum(jnp.asarray(current_max, dtype=jnp.float32), channel_max)


def deformation_channel_max_dict(values: np.ndarray | jnp.ndarray) -> dict[str, float]:
    """Format a channel abs-max vector for JSON/W&B logging."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape != (len(DEFORMATION_CHANNEL_KEYS),):
        raise ValueError(
            "Expected deformation channel maxima with shape "
            f"({len(DEFORMATION_CHANNEL_KEYS)},); got {arr.shape}."
        )
    return {
        key: float(arr[idx])
        for idx, key in enumerate(DEFORMATION_CHANNEL_KEYS)
    }
