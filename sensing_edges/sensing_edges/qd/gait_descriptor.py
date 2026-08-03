"""Host-side gait spectral purity helpers."""

from __future__ import annotations

import numpy as np


_MIN_TRACE_STEPS = 20
_F_MAX = 0.1


def compute_gait_spectral_purity(com_x_trace: np.ndarray, dt: float = 1.0) -> float:
    """Return the dominant-frequency energy fraction of a COM-x trace."""
    del dt  # Frequency is measured in cycles per env step for this task.
    com_x = np.asarray(com_x_trace, dtype=np.float32).reshape(-1)
    if com_x.size < _MIN_TRACE_STEPS:
        return 0.0

    velocity = np.diff(com_x)
    if velocity.size < max(_MIN_TRACE_STEPS - 1, 2):
        return 0.0
    if not np.any(np.abs(velocity) > 0.0):
        return 0.0

    window = np.hanning(velocity.size).astype(np.float32)
    windowed = velocity * window
    magnitudes = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(windowed.size, d=1.0)
    f_min = 2.0 / max(windowed.size, 1)
    band = magnitudes[(freqs >= f_min) & (freqs <= _F_MAX)]
    if band.size == 0:
        return 0.0

    energies = np.square(band.astype(np.float64))
    total_energy = float(np.sum(energies))
    if total_energy <= 0.0:
        return 0.0
    return float(np.max(energies) / total_energy)


def compute_candidate_gait_spectral_purity(
    com_x_traces: np.ndarray,
    steps_alive: np.ndarray,
) -> np.ndarray:
    """Return mean per-candidate gait purity across episodes."""
    traces = np.asarray(com_x_traces, dtype=np.float32)
    steps = np.asarray(steps_alive, dtype=np.int32)
    if traces.ndim != 3:
        raise ValueError(
            "com_x_traces must have shape (n_candidates, n_episodes, max_steps), "
            f"got {traces.shape}"
        )
    if steps.shape != traces.shape[:2]:
        raise ValueError(
            "steps_alive must have shape (n_candidates, n_episodes), "
            f"got {steps.shape} for traces {traces.shape}"
        )

    purities = np.zeros(steps.shape, dtype=np.float32)
    for candidate_idx in range(traces.shape[0]):
        for episode_idx in range(traces.shape[1]):
            valid_steps = int(np.clip(steps[candidate_idx, episode_idx], 0, traces.shape[2]))
            purities[candidate_idx, episode_idx] = compute_gait_spectral_purity(
                traces[candidate_idx, episode_idx, :valid_steps]
            )
    return np.mean(purities, axis=1, dtype=np.float32)
