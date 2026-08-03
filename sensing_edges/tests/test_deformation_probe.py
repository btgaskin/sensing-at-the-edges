"""Tests for signed raw deformation probe summaries."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from sensing_edges.deformation_probe import update_deformation_channel_max, zero_deformation_channel_max


def test_update_deformation_channel_max_tracks_signed_channel_absmax():
    deformation = jnp.asarray(
        [
            [1.20, 0.90, 1.10, 0.95],
            [0.80, 1.05, 0.90, 1.20],
        ],
        dtype=jnp.float32,
    )

    observed = update_deformation_channel_max(
        zero_deformation_channel_max(),
        deformation,
    )

    assert tuple(float(value) for value in observed) == pytest.approx(
        (
            0.20,
            0.10,
            0.10,
            0.20,
        )
    )
