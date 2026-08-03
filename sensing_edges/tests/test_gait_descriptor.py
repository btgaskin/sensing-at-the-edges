"""Tests for gait spectral purity helpers."""

from __future__ import annotations

import numpy as np
import pytest

from sensing_edges.qd.gait_descriptor import (
    compute_candidate_gait_spectral_purity,
    compute_gait_spectral_purity,
)


def test_pure_sine_wave_has_high_spectral_purity():
    steps = np.arange(256, dtype=np.float32)
    trace = np.sin(2.0 * np.pi * 0.05 * steps)

    purity = compute_gait_spectral_purity(trace)

    assert purity > 0.6


def test_white_noise_has_low_spectral_purity():
    rng = np.random.default_rng(0)
    trace = rng.normal(size=256).astype(np.float32)

    purity = compute_gait_spectral_purity(trace)

    assert purity < 0.2


def test_stationary_trace_returns_zero():
    purity = compute_gait_spectral_purity(np.ones(256, dtype=np.float32))

    assert purity == pytest.approx(0.0)


def test_short_trace_returns_zero():
    purity = compute_gait_spectral_purity(np.linspace(0.0, 1.0, 10, dtype=np.float32))

    assert purity == pytest.approx(0.0)


def test_high_frequency_trace_is_filtered_out():
    steps = np.arange(256, dtype=np.float32)
    trace = np.sin(2.0 * np.pi * 0.2 * steps)

    purity = compute_gait_spectral_purity(trace)

    assert purity < 0.2


def test_candidate_gait_spectral_purity_averages_episode_scores():
    steps = np.arange(256, dtype=np.float32)
    clean = np.sin(2.0 * np.pi * 0.05 * steps).astype(np.float32)
    noisy = np.random.default_rng(1).normal(size=256).astype(np.float32)
    traces = np.stack(
        [
            np.stack([clean, clean], axis=0),
            np.stack([clean, noisy], axis=0),
        ],
        axis=0,
    )
    steps_alive = np.full((2, 2), 256, dtype=np.int32)

    purities = compute_candidate_gait_spectral_purity(traces, steps_alive)

    assert purities.shape == (2,)
    assert purities[0] > purities[1]
