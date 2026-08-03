from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf


SCRIPT_DIR = Path(__file__).resolve().parent
S4_ROOT = SCRIPT_DIR.parent
REPO_ROOT = S4_ROOT.parent
SENSING_EDGES_ROOT = REPO_ROOT / "sensing_edges"
DEFAULT_BUNDLE_ROOT = S4_ROOT / "results" / "bin_03_rank_00_cell_6-0-3_flat_0423"
DEFAULT_MAX_STEPS = 1500


def _ensure_import_path() -> None:
    path = str(SENSING_EDGES_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _save_lag_scan(
    lag_values: np.ndarray,
    scores: np.ndarray,
    *,
    best_lag: int,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(lag_values, scores, linewidth=2.0)
    ax.axvline(float(best_lag), color="#d95f02", linestyle="--", linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("phase lag")
    ax.set_ylabel("median normalized distance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_recurrence_heatmap(
    matrix: np.ndarray,
    *,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    image = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("phase index")
    ax.set_ylabel("phase index")
    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_convergence_plot(
    records: list[dict[str, Any]],
    *,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid_records = [
        record
        for record in records
        if record.get("early_rms") is not None and record.get("late_rms") is not None
    ]
    labels = [f"p{int(record['trial_index']):02d}" for record in valid_records]
    early = np.asarray([float(record["early_rms"]) for record in valid_records], dtype=np.float32)
    late = np.asarray([float(record["late_rms"]) for record in valid_records], dtype=np.float32)
    x = np.arange(len(labels), dtype=np.float32)
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(6.2, 0.8 * len(labels) + 2.0), 4.2))
    if valid_records:
        ax.bar(x - width / 2.0, early, width=width, label="early", color="#9ecae1")
        ax.bar(x + width / 2.0, late, width=width, label="late", color="#3182bd")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "no valid convergence windows", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_ylabel("best phase-aligned RMS")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _load_controller_params(path: Path):
    _ensure_import_path()
    from sensing_edges.nca import CTRNNParams

    with np.load(path, allow_pickle=False) as data:
        return CTRNNParams(
            w_input=jnp.asarray(np.asarray(data["w_input"], dtype=np.float32)),
            w_rec=jnp.asarray(np.asarray(data["w_rec"], dtype=np.float32)),
            w_output=jnp.asarray(np.asarray(data["w_output"], dtype=np.float32)),
            bias=jnp.asarray(np.asarray(data["bias"], dtype=np.float32)),
            log_tau=jnp.asarray(np.asarray(data["log_tau"], dtype=np.float32)),
        )


def _default_width(cfg) -> int:
    widths = [int(width) for width in list(cfg.eval.post_run.double_cliff.widths)]
    if not widths:
        raise ValueError("eval.post_run.double_cliff.widths must be non-empty")
    return max(widths)


def _result_payload(result) -> dict[str, Any]:
    payload = {
        "goal_touched": bool(np.asarray(result.goal_touched).item()),
        "cliff_fell": bool(np.asarray(result.cliff_fell).item()),
        "steps_alive": int(np.asarray(result.steps_alive).item()),
        "turn_count": int(np.asarray(result.turn_count).item()),
        "stable_turn_count": int(np.asarray(result.stable_turn_count).item()),
        "mean_activation": float(np.asarray(result.mean_activation).item()),
        "mean_speed": float(np.asarray(result.mean_speed).item()),
        "goal_progress": float(np.asarray(result.goal_progress).item()),
        "incorrect_progress": float(np.asarray(result.incorrect_progress).item()),
        "time_to_edge": float(np.asarray(result.time_to_edge).item()),
        "time_to_goal": float(np.asarray(result.time_to_goal).item()),
        "controller_hidden_output_max_abs": float(
            np.asarray(result.controller_hidden_output_max_abs).item()
        ),
        "action_output_max_abs": float(np.asarray(result.action_output_max_abs).item()),
    }
    deformation_max = np.asarray(result.deformation_channel_max, dtype=np.float32).tolist()
    payload["deformation_channel_max"] = [float(value) for value in deformation_max]
    return payload


def _phase_reshape(x: np.ndarray, nca_steps_per_phase: int) -> np.ndarray:
    trim = (x.shape[0] // int(nca_steps_per_phase)) * int(nca_steps_per_phase)
    if trim <= 0:
        raise ValueError("Trace ended before a full phase was captured.")
    phase_count = trim // int(nca_steps_per_phase)
    return x[:trim].reshape(phase_count, int(nca_steps_per_phase), *x.shape[1:])


def _reduce_trace(
    trace,
    *,
    nca_steps_per_phase: int,
    ctrnn_grid_hidden: int,
    ou_enabled: bool,
) -> dict[str, np.ndarray]:
    _ensure_import_path()
    from sensing_edges.nca import base_channel_count

    active = np.asarray(trace.active, dtype=bool)
    if not np.any(active):
        raise ValueError("Controller trace contains no active steps.")

    nca_micro = np.asarray(trace.nca_state[active], dtype=np.float32)
    ctrnn_micro = np.asarray(trace.ctrnn_carry[active], dtype=np.float32)
    action_micro = np.asarray(trace.action_outputs[active], dtype=np.float32)
    com_x_micro = np.asarray(trace.com_x[active], dtype=np.float32)
    back_x_micro = np.asarray(trace.back_x[active], dtype=np.float32)
    front_x_micro = np.asarray(trace.front_x[active], dtype=np.float32)

    action_phase = _phase_reshape(action_micro, nca_steps_per_phase).mean(axis=1)
    ctrnn_phase = _phase_reshape(ctrnn_micro, nca_steps_per_phase)[:, -1]
    nca_phase = _phase_reshape(nca_micro, nca_steps_per_phase)[:, -1]
    com_x_phase = _phase_reshape(com_x_micro[:, None], nca_steps_per_phase)[:, -1, 0]
    back_x_phase = _phase_reshape(back_x_micro[:, None], nca_steps_per_phase)[:, -1, 0]
    front_x_phase = _phase_reshape(front_x_micro[:, None], nca_steps_per_phase)[:, -1, 0]

    comm_start = base_channel_count(ou_enabled=ou_enabled)
    comm_stop = comm_start + int(ctrnn_grid_hidden)
    comm_phase = nca_phase[..., comm_start:comm_stop]

    action_flat = action_phase.reshape(action_phase.shape[0], -1)
    comm_flat = comm_phase.reshape(comm_phase.shape[0], -1)
    ctrnn_flat = ctrnn_phase.reshape(ctrnn_phase.shape[0], -1)
    internal_flat = np.concatenate([comm_flat, ctrnn_flat], axis=1)
    joint_flat = np.concatenate([action_flat, internal_flat], axis=1)

    return {
        "action_phase": action_phase,
        "action_flat": action_flat,
        "comm_phase": comm_phase,
        "comm_flat": comm_flat,
        "ctrnn_phase": ctrnn_phase,
        "ctrnn_flat": ctrnn_flat,
        "internal_flat": internal_flat,
        "joint_flat": joint_flat,
        "com_x_phase": com_x_phase,
        "back_x_phase": back_x_phase,
        "front_x_phase": front_x_phase,
        "controller_step_count": np.asarray([action_micro.shape[0]], dtype=np.int32),
        "phase_count": np.asarray([action_phase.shape[0]], dtype=np.int32),
    }


def _trace_scale(x: np.ndarray) -> float:
    norms = np.linalg.norm(x, axis=1)
    scale = float(np.median(norms))
    if scale <= 1.0e-8:
        scale = float(np.sqrt(np.mean(np.square(x)))) if x.size else 1.0
    return max(scale, 1.0e-8)


def _estimate_period(
    x: np.ndarray,
    *,
    warmup_phases: int,
    min_lag: int,
    max_lag: int,
    recurrence_threshold_scale: float = 0.15,
) -> dict[str, Any]:
    if x.shape[0] <= int(warmup_phases) + int(min_lag) + 2:
        raise ValueError("Not enough phases to estimate a recurrence lag.")
    late = x[int(warmup_phases):]
    usable_max_lag = min(int(max_lag), late.shape[0] - 2)
    if usable_max_lag < int(min_lag):
        raise ValueError("usable max lag fell below min lag")

    lag_values = np.arange(int(min_lag), usable_max_lag + 1, dtype=np.int32)
    scale = _trace_scale(late)
    median_scores = np.zeros((lag_values.shape[0],), dtype=np.float32)
    p90_scores = np.zeros_like(median_scores)
    recurrent_fraction = np.zeros_like(median_scores)
    threshold = float(recurrence_threshold_scale) * scale

    for index, lag in enumerate(lag_values):
        diff = np.linalg.norm(late[lag:] - late[:-lag], axis=1)
        median_scores[index] = float(np.median(diff) / scale)
        p90_scores[index] = float(np.quantile(diff, 0.90) / scale)
        recurrent_fraction[index] = float(np.mean(diff <= threshold))

    best_index = int(np.argmin(median_scores))
    best_lag = int(lag_values[best_index])
    return {
        "lag_values": lag_values,
        "median_scores": median_scores,
        "p90_scores": p90_scores,
        "recurrent_fraction": recurrent_fraction,
        "best_lag": best_lag,
        "best_median_score": float(median_scores[best_index]),
        "best_p90_score": float(p90_scores[best_index]),
        "best_recurrent_fraction": float(recurrent_fraction[best_index]),
        "late_phase_count": int(late.shape[0]),
        "scale": scale,
    }


def _pairwise_distance_matrix(x: np.ndarray, *, max_points: int = 240) -> np.ndarray:
    if x.shape[0] > max_points:
        sample_idx = np.linspace(0, x.shape[0] - 1, num=max_points, dtype=np.int32)
        x = x[sample_idx]
    norms = np.sum(np.square(x), axis=1, keepdims=True)
    sq_dist = np.maximum(norms + norms.T - 2.0 * (x @ x.T), 0.0)
    return np.sqrt(sq_dist).astype(np.float32, copy=False)


def _best_phase_aligned_rms(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    window: int,
    max_shift: int,
) -> tuple[float, int]:
    usable = min(int(window), reference.shape[0], candidate.shape[0])
    if usable <= 1:
        raise ValueError("Need at least two phases for phase-aligned RMS.")
    ref_window = reference[-usable:]
    cand_window = candidate[-usable:]
    usable_shift = max(1, min(int(max_shift), usable))

    best_rms = float("inf")
    best_shift = 0
    for shift in range(usable_shift):
        shifted = np.roll(cand_window, shift=shift, axis=0)
        rms = float(np.sqrt(np.mean(np.square(ref_window - shifted))))
        if rms < best_rms:
            best_rms = rms
            best_shift = shift
    return best_rms, best_shift


def _early_window(x: np.ndarray, *, warmup_phases: int, window: int) -> np.ndarray:
    start = min(max(int(warmup_phases), 0), max(x.shape[0] - 1, 0))
    stop = min(start + int(window), x.shape[0])
    return x[start:stop]


def _late_window(x: np.ndarray, *, window: int) -> np.ndarray:
    usable = min(int(window), x.shape[0])
    return x[-usable:]


def _convergence_records(
    reference: np.ndarray,
    candidates: list[np.ndarray],
    *,
    warmup_phases: int,
    window: int,
    period: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    ref_early = _early_window(reference, warmup_phases=warmup_phases, window=window)
    ref_late = _late_window(reference, window=window)
    usable_period = max(1, int(period))

    for trial_index, candidate in enumerate(candidates, start=1):
        cand_early = _early_window(candidate, warmup_phases=warmup_phases, window=window)
        cand_late = _late_window(candidate, window=window)
        try:
            early_rms, early_shift = _best_phase_aligned_rms(
                ref_early,
                cand_early,
                window=min(ref_early.shape[0], cand_early.shape[0]),
                max_shift=usable_period,
            )
            late_rms, late_shift = _best_phase_aligned_rms(
                ref_late,
                cand_late,
                window=min(ref_late.shape[0], cand_late.shape[0]),
                max_shift=usable_period,
            )
            ratio = float(late_rms / max(early_rms, 1.0e-8))
            status = "ok"
        except ValueError:
            early_rms = None
            late_rms = None
            early_shift = None
            late_shift = None
            ratio = None
            status = "too_short"
        records.append(
            {
                "trial_index": int(trial_index),
                "status": status,
                "early_rms": None if early_rms is None else float(early_rms),
                "late_rms": None if late_rms is None else float(late_rms),
                "late_over_early": ratio,
                "best_early_shift": None if early_shift is None else int(early_shift),
                "best_late_shift": None if late_shift is None else int(late_shift),
            }
        )
    return records


def _summarize_convergence(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid_records = [
        record
        for record in records
        if record.get("early_rms") is not None and record.get("late_rms") is not None
    ]
    if not valid_records:
        return {
            "trial_count": 0,
            "valid_trial_count": 0,
            "mean_early_rms": None,
            "mean_late_rms": None,
            "mean_late_over_early": None,
            "late_better_fraction": None,
        }
    early = np.asarray([float(record["early_rms"]) for record in valid_records], dtype=np.float32)
    late = np.asarray([float(record["late_rms"]) for record in valid_records], dtype=np.float32)
    ratio = np.asarray([float(record["late_over_early"]) for record in valid_records], dtype=np.float32)
    return {
        "trial_count": int(len(records)),
        "valid_trial_count": int(len(valid_records)),
        "mean_early_rms": float(np.mean(early)),
        "mean_late_rms": float(np.mean(late)),
        "mean_late_over_early": float(np.mean(ratio)),
        "late_better_fraction": float(np.mean(late < early)),
    }


def _capture_run(
    *,
    env,
    params,
    alive_mask: jnp.ndarray,
    phase_cfg,
    phase_resolution,
    seed: int,
    initial_nca_state: jnp.ndarray | None = None,
    initial_ctrnn_carry: jnp.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    _ensure_import_path()
    from sensing_edges.env import run_episode_with_controller_trace
    from sensing_edges.runtime_config import resolved_noise_settings, rollout_kwargs_from_cfg

    rollout_kwargs = rollout_kwargs_from_cfg(
        phase_cfg,
        collision_tuning=phase_resolution.resolved_collision,
    )
    result, trace = run_episode_with_controller_trace(
        env,
        params,
        alive_mask,
        jax.random.PRNGKey(int(seed)),
        initial_nca_state=initial_nca_state,
        initial_ctrnn_carry=initial_ctrnn_carry,
        **rollout_kwargs,
    )
    trace_payload = _reduce_trace(
        trace,
        nca_steps_per_phase=int(phase_cfg.nca.nca_steps_per_phase),
        ctrnn_grid_hidden=int(phase_cfg.nca.ctrnn_grid_hidden),
        ou_enabled=bool(resolved_noise_settings(phase_cfg)["ou_enabled"]),
    )
    return _result_payload(result), trace_payload


def _save_trace_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _initial_state_payload(
    *,
    rng: np.random.Generator,
    robot_h: int,
    robot_w: int,
    gc: int,
    ctrnn_hidden: int,
    ctrnn_grid_hidden: int,
    ou_enabled: bool,
    nca_init_scale: float,
    ctrnn_init_scale: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    _ensure_import_path()
    from sensing_edges.nca import base_channel_count

    nca_state = np.zeros((robot_h, robot_w, gc), dtype=np.float32)
    comm_start = base_channel_count(ou_enabled=ou_enabled)
    if int(ctrnn_grid_hidden) > 0 and float(nca_init_scale) > 0.0:
        nca_state[:, :, comm_start:comm_start + int(ctrnn_grid_hidden)] = rng.uniform(
            low=-float(nca_init_scale),
            high=float(nca_init_scale),
            size=(robot_h, robot_w, int(ctrnn_grid_hidden)),
        ).astype(np.float32)
    ctrnn_carry = np.zeros((robot_h, robot_w, ctrnn_hidden), dtype=np.float32)
    if float(ctrnn_init_scale) > 0.0:
        ctrnn_carry = rng.uniform(
            low=-float(ctrnn_init_scale),
            high=float(ctrnn_init_scale),
            size=(robot_h, robot_w, ctrnn_hidden),
        ).astype(np.float32)
    return jnp.asarray(nca_state), jnp.asarray(ctrnn_carry)


def run_wave_probe(
    *,
    bundle_root: Path,
    controller_params_path: Path,
    config_snapshot_path: Path,
    output_dir: Path,
    width: int | None,
    plateau_width: int | None,
    max_steps: int | None,
    deformation_condition: str,
    signaling_condition: str,
    perturbation_trials: int,
    nca_init_scale: float,
    ctrnn_init_scale: float,
    warmup_phases: int,
    min_lag: int,
    max_lag: int,
    comparison_window: int,
    seed: int,
) -> dict[str, Any]:
    _ensure_import_path()
    from sensing_edges.evaluate import _build_phase_assets
    from sensing_edges.nca import grid_channel_count
    from sensing_edges.post_run import _build_double_cliff_entry, _double_cliff_phase_cfg
    from sensing_edges.runtime_config import (
        compose_experiment_config,
        default_config_dir,
        resolve_runtime_config,
        resolved_noise_settings,
    )

    snapshot_cfg = OmegaConf.load(config_snapshot_path)
    preset_name = (
        str(snapshot_cfg.get("preset_id")).strip()
        if snapshot_cfg.get("preset_id") is not None
        else None
    )
    base_cfg = compose_experiment_config(
        preset=preset_name,
        config_dir=default_config_dir(),
    )
    cfg = OmegaConf.merge(
        OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False)),
        snapshot_cfg,
    )
    phase_cfg = _double_cliff_phase_cfg(
        cfg,
        active_deformation_condition=deformation_condition,
        active_signaling_condition=signaling_condition,
    )
    if max_steps is not None:
        phase_cfg.train.max_steps = int(max_steps)

    resolved_width = int(width) if width is not None else _default_width(cfg)
    resolved_plateau_width = (
        int(plateau_width)
        if plateau_width is not None
        else int(cfg.eval.post_run.double_cliff.plateau_width)
    )
    runtime_resolution = resolve_runtime_config(
        phase_cfg,
        preset=(str(phase_cfg.get("preset_id")) if phase_cfg.get("preset_id") is not None else None),
    )
    entry = _build_double_cliff_entry(
        phase_cfg,
        width=resolved_width,
        plateau_width=resolved_plateau_width,
        extra_metadata={"analysis_mode": "deformation_off_wave_probe"},
    )
    phase_resolution, envs, alive_mask, _ = _build_phase_assets(
        phase_cfg,
        runtime_resolution,
        [entry],
    )
    env = envs[0]
    params = _load_controller_params(controller_params_path)

    trace_dir = output_dir / "traces"
    figures_dir = output_dir / "figures"
    trace_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    baseline_result, baseline_trace = _capture_run(
        env=env,
        params=params,
        alive_mask=alive_mask,
        phase_cfg=phase_cfg,
        phase_resolution=phase_resolution,
        seed=int(seed),
    )
    _save_trace_npz(trace_dir / "baseline.npz", baseline_trace)

    rng = np.random.default_rng(int(seed) + 1)
    noise_settings = resolved_noise_settings(phase_cfg)
    robot_h, robot_w = alive_mask.shape
    gc = grid_channel_count(
        int(phase_cfg.nca.ctrnn_grid_hidden),
        str(phase_cfg.nca.get("gasnet_mode", "off")),
        bool(noise_settings["ou_enabled"]),
        growth_enabled=False,
    )

    perturbation_results: list[dict[str, Any]] = []
    perturbation_traces: list[dict[str, np.ndarray]] = []
    for trial_index in range(int(perturbation_trials)):
        initial_nca_state, initial_ctrnn_carry = _initial_state_payload(
            rng=rng,
            robot_h=int(robot_h),
            robot_w=int(robot_w),
            gc=int(gc),
            ctrnn_hidden=int(phase_cfg.nca.ctrnn_hidden),
            ctrnn_grid_hidden=int(phase_cfg.nca.ctrnn_grid_hidden),
            ou_enabled=bool(noise_settings["ou_enabled"]),
            nca_init_scale=float(nca_init_scale),
            ctrnn_init_scale=float(ctrnn_init_scale),
        )
        run_result, run_trace = _capture_run(
            env=env,
            params=params,
            alive_mask=alive_mask,
            phase_cfg=phase_cfg,
            phase_resolution=phase_resolution,
            seed=int(seed),
            initial_nca_state=initial_nca_state,
            initial_ctrnn_carry=initial_ctrnn_carry,
        )
        run_result["trial_index"] = int(trial_index + 1)
        perturbation_results.append(run_result)
        perturbation_traces.append(run_trace)
        _save_trace_npz(trace_dir / f"perturb_{trial_index + 1:02d}.npz", run_trace)

    action_period = _estimate_period(
        baseline_trace["action_flat"],
        warmup_phases=int(warmup_phases),
        min_lag=int(min_lag),
        max_lag=int(max_lag),
    )
    internal_period = _estimate_period(
        baseline_trace["internal_flat"],
        warmup_phases=int(warmup_phases),
        min_lag=int(min_lag),
        max_lag=int(max_lag),
    )
    joint_period = _estimate_period(
        baseline_trace["joint_flat"],
        warmup_phases=int(warmup_phases),
        min_lag=int(min_lag),
        max_lag=int(max_lag),
    )

    action_convergence = _convergence_records(
        baseline_trace["action_flat"],
        [trace["action_flat"] for trace in perturbation_traces],
        warmup_phases=int(warmup_phases),
        window=int(comparison_window),
        period=int(action_period["best_lag"]),
    )
    internal_convergence = _convergence_records(
        baseline_trace["internal_flat"],
        [trace["internal_flat"] for trace in perturbation_traces],
        warmup_phases=int(warmup_phases),
        window=int(comparison_window),
        period=int(internal_period["best_lag"]),
    )

    late_action = baseline_trace["action_flat"][int(warmup_phases):]
    late_internal = baseline_trace["internal_flat"][int(warmup_phases):]
    action_recurrence = _pairwise_distance_matrix(late_action)
    internal_recurrence = _pairwise_distance_matrix(late_internal)

    _save_lag_scan(
        action_period["lag_values"],
        action_period["median_scores"],
        best_lag=int(action_period["best_lag"]),
        title="Action-Space Recurrence Scan",
        output_path=figures_dir / "action_lag_scan.png",
    )
    _save_lag_scan(
        internal_period["lag_values"],
        internal_period["median_scores"],
        best_lag=int(internal_period["best_lag"]),
        title="Internal-State Recurrence Scan",
        output_path=figures_dir / "internal_lag_scan.png",
    )
    _save_recurrence_heatmap(
        action_recurrence,
        title="Late Action-Space Distance Matrix",
        output_path=figures_dir / "action_recurrence_heatmap.png",
    )
    _save_recurrence_heatmap(
        internal_recurrence,
        title="Late Internal-State Distance Matrix",
        output_path=figures_dir / "internal_recurrence_heatmap.png",
    )
    _save_convergence_plot(
        action_convergence,
        title="Perturbation Convergence in Action Space",
        output_path=figures_dir / "action_convergence.png",
    )
    _save_convergence_plot(
        internal_convergence,
        title="Perturbation Convergence in Internal State",
        output_path=figures_dir / "internal_convergence.png",
    )

    np.savez_compressed(
        output_dir / "metrics.npz",
        action_lag_values=action_period["lag_values"],
        action_median_scores=action_period["median_scores"],
        action_p90_scores=action_period["p90_scores"],
        action_recurrent_fraction=action_period["recurrent_fraction"],
        internal_lag_values=internal_period["lag_values"],
        internal_median_scores=internal_period["median_scores"],
        internal_p90_scores=internal_period["p90_scores"],
        internal_recurrent_fraction=internal_period["recurrent_fraction"],
        joint_lag_values=joint_period["lag_values"],
        joint_median_scores=joint_period["median_scores"],
        joint_p90_scores=joint_period["p90_scores"],
        joint_recurrent_fraction=joint_period["recurrent_fraction"],
        action_recurrence_matrix=action_recurrence,
        internal_recurrence_matrix=internal_recurrence,
    )

    summary = {
        "bundle_root": str(bundle_root),
        "controller_params_path": str(controller_params_path),
        "config_snapshot_path": str(config_snapshot_path),
        "terrain": {
            "width": int(resolved_width),
            "plateau_width": int(resolved_plateau_width),
            "grid_h": int(phase_cfg.terrain.grid_h),
            "grid_w": int(phase_cfg.terrain.grid_w),
            "platform_y": int(phase_cfg.terrain.get("platform_y", 3)),
            "entry_metadata": {
                key: int(value) if isinstance(value, (int, np.integer)) else value
                for key, value in dict(entry.metadata or {}).items()
            },
        },
        "capture": {
            "deformation_condition": str(deformation_condition),
            "signaling_condition": str(signaling_condition),
            "max_steps": int(phase_cfg.train.max_steps),
            "nca_steps_per_phase": int(phase_cfg.nca.nca_steps_per_phase),
            "sense_act_phases": int(phase_cfg.sim.sense_act_phases_per_env_step),
            "physics_substeps": int(phase_cfg.sim.physics_substeps),
            "ou_enabled": bool(noise_settings["ou_enabled"]),
            "perturbation_trials": int(perturbation_trials),
            "nca_init_scale": float(nca_init_scale),
            "ctrnn_init_scale": float(ctrnn_init_scale),
            "warmup_phases": int(warmup_phases),
            "comparison_window": int(comparison_window),
            "seed": int(seed),
        },
        "baseline": {
            "result": baseline_result,
            "phase_count": int(baseline_trace["phase_count"][0]),
            "controller_step_count": int(baseline_trace["controller_step_count"][0]),
        },
        "period_estimates": {
            "action": {
                "best_lag": int(action_period["best_lag"]),
                "best_median_score": float(action_period["best_median_score"]),
                "best_p90_score": float(action_period["best_p90_score"]),
                "best_recurrent_fraction": float(action_period["best_recurrent_fraction"]),
                "late_phase_count": int(action_period["late_phase_count"]),
            },
            "internal": {
                "best_lag": int(internal_period["best_lag"]),
                "best_median_score": float(internal_period["best_median_score"]),
                "best_p90_score": float(internal_period["best_p90_score"]),
                "best_recurrent_fraction": float(internal_period["best_recurrent_fraction"]),
                "late_phase_count": int(internal_period["late_phase_count"]),
            },
            "joint": {
                "best_lag": int(joint_period["best_lag"]),
                "best_median_score": float(joint_period["best_median_score"]),
                "best_p90_score": float(joint_period["best_p90_score"]),
                "best_recurrent_fraction": float(joint_period["best_recurrent_fraction"]),
                "late_phase_count": int(joint_period["late_phase_count"]),
            },
        },
        "perturbation_runs": perturbation_results,
        "convergence": {
            "action": {
                "records": action_convergence,
                "summary": _summarize_convergence(action_convergence),
            },
            "internal": {
                "records": internal_convergence,
                "summary": _summarize_convergence(internal_convergence),
            },
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture a deformation-off controller trace on the widest double-cliff setup, "
            "then test recurrence and convergence under perturbed initial internal states."
        )
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT,
        help="Specimen result bundle root under s4/results.",
    )
    parser.add_argument(
        "--controller-params",
        type=Path,
        default=None,
        help="Override path to controller_params.npz.",
    )
    parser.add_argument(
        "--config-snapshot",
        type=Path,
        default=None,
        help="Override path to config_snapshot.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for reduced traces, figures, and summary JSON.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Double-cliff width to analyze. Defaults to the widest configured width.",
    )
    parser.add_argument(
        "--plateau-width",
        type=int,
        default=None,
        help="Override plateau width. Defaults to eval.post_run.double_cliff.plateau_width.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Max env steps for the probe run. Defaults to a lightweight local budget.",
    )
    parser.add_argument(
        "--deformation-condition",
        type=str,
        default="deformation_off",
        help="Active deformation condition for the probe run.",
    )
    parser.add_argument(
        "--signaling-condition",
        type=str,
        default="intact",
        help="Active signaling condition for the probe run.",
    )
    parser.add_argument(
        "--perturbation-trials",
        type=int,
        default=4,
        help="Number of perturbed initial-state reruns.",
    )
    parser.add_argument(
        "--nca-init-scale",
        type=float,
        default=0.20,
        help="Uniform perturbation scale for the grid-hidden channels at t=0.",
    )
    parser.add_argument(
        "--ctrnn-init-scale",
        type=float,
        default=0.20,
        help="Uniform perturbation scale for the CTRNN carry at t=0.",
    )
    parser.add_argument(
        "--warmup-phases",
        type=int,
        default=180,
        help="Discard this many initial phases before recurrence analysis.",
    )
    parser.add_argument(
        "--min-lag",
        type=int,
        default=3,
        help="Minimum phase lag to scan when estimating recurrence.",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=240,
        help="Maximum phase lag to scan when estimating recurrence.",
    )
    parser.add_argument(
        "--comparison-window",
        type=int,
        default=240,
        help="Window size, in phases, for early/late perturbation alignment.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for trace capture and initial-state perturbations.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    bundle_root = args.bundle_root.resolve()
    controller_params_path = (
        args.controller_params.resolve()
        if args.controller_params is not None
        else (bundle_root / "controller_params.npz").resolve()
    )
    config_snapshot_path = (
        args.config_snapshot.resolve()
        if args.config_snapshot is not None
        else (bundle_root / "config_snapshot.yaml").resolve()
    )

    resolved_width_for_output: int
    if args.width is not None:
        resolved_width_for_output = int(args.width)
    else:
        cfg = OmegaConf.load(config_snapshot_path)
        resolved_width_for_output = _default_width(cfg)

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (
            bundle_root
            / "analysis_runs"
            / f"deformation_off_wave_probe_width_{resolved_width_for_output}"
        ).resolve()
    )

    summary = run_wave_probe(
        bundle_root=bundle_root,
        controller_params_path=controller_params_path,
        config_snapshot_path=config_snapshot_path,
        output_dir=output_dir,
        width=args.width,
        plateau_width=args.plateau_width,
        max_steps=args.max_steps,
        deformation_condition=args.deformation_condition,
        signaling_condition=args.signaling_condition,
        perturbation_trials=args.perturbation_trials,
        nca_init_scale=args.nca_init_scale,
        ctrnn_init_scale=args.ctrnn_init_scale,
        warmup_phases=args.warmup_phases,
        min_lag=args.min_lag,
        max_lag=args.max_lag,
        comparison_window=args.comparison_window,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
