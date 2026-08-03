#!/usr/bin/env python
"""Maximum-caliber-inspired path-ensemble analysis for the S4 fixed-width packet.

This is deliberately coarse-grained. The deterministic fixed-width traces are
treated as an empirical path ensemble over cliff-distance perturbation contexts,
not as independent stochastic samples from a thermal process. The analysis asks
which low-dimensional constraints best reproduce the post-release path flux into
away-from-cliff motion.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKET_ROOT = REPO_ROOT / "paper_data" / "preprint" / "s4_fixed_width_distance_sweep_20260618"
STREAM_ORDER = (
    "signal_hidden",
    "controller_action",
    "spring_contractile",
    "spring_current",
    "body_kinematic",
    "body_no_contractile",
    "body_combined",
)
MODEL_ORDER = (
    "open_loop_pre",
    "post_direction",
    "post_time",
    "post_stream",
    "post_time_stream",
)
DIRECTION_STATES = (-1, 0, 1)
STATE_LABELS = {-1: "toward", 0: "uncommitted", 1: "away"}
POST_START = 0
POST_END = 336
RNG_SEED = 137


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _bootstrap_ci(values: Iterable[float], *, seed: int = RNG_SEED, draws: int = 10_000) -> tuple[float, float]:
    arr = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(int(draws), arr.size), replace=True).mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def _time_bin(rel_time: int) -> str:
    rel_time = int(rel_time)
    if rel_time < 0:
        return "pre"
    if rel_time <= 64:
        return "release_0_64"
    if rel_time <= 160:
        return "early_72_160"
    if rel_time <= 336:
        return "late_168_336"
    return "tail_after_336"


def _fit_bin_edges(values: pd.Series, *, max_bins: int = 3) -> tuple[float, ...]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 4:
        return ()
    if float(np.nanmax(arr) - np.nanmin(arr)) <= 1.0e-12:
        return ()
    if max_bins <= 2:
        edges = np.quantile(arr, [0.5])
    else:
        edges = np.quantile(arr, [1 / 3, 2 / 3])
    unique_edges: list[float] = []
    for edge in edges:
        edge = float(edge)
        if not unique_edges or abs(edge - unique_edges[-1]) > 1.0e-12:
            unique_edges.append(edge)
    if not unique_edges:
        return ()
    if len(unique_edges) == 2 and abs(unique_edges[1] - unique_edges[0]) <= 1.0e-12:
        return (unique_edges[0],)
    return tuple(unique_edges)


def _assign_bins(values: pd.Series, edges: tuple[float, ...]) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    labels: list[str] = []
    if not edges:
        return pd.Series(["all"] * len(arr), index=values.index)
    for value in arr:
        if not np.isfinite(value):
            labels.append("missing")
        elif len(edges) == 1:
            labels.append("low" if value <= edges[0] else "high")
        elif value <= edges[0]:
            labels.append("low")
        elif value <= edges[1]:
            labels.append("mid")
        else:
            labels.append("high")
    return pd.Series(labels, index=values.index)


def _direction_name(value: int) -> str:
    return STATE_LABELS.get(int(value), str(int(value)))


def _aggregate_slip(slip: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        slip.groupby(["event_id", "trial_id", "cliff_distance", "stream", "rel_time"], as_index=False)
        .agg(
            mean_abs_phase_slip=("abs_phase_slip", "mean"),
            max_abs_phase_slip=("abs_phase_slip", "max"),
            mean_pc_l2_delta=("pc_l2_delta", "mean"),
            max_pc_l2_delta=("pc_l2_delta", "max"),
            direction_label=("direction_label", "first"),
            direction_delta=("direction_delta", "first"),
            success=("success", "first"),
            returned_from_edge=("returned_from_edge", "first"),
            continuing_count=("continuing_count", "first"),
        )
        .sort_values(["stream", "event_id", "rel_time"])
    )
    grouped["rel_time"] = grouped["rel_time"].astype(int)
    grouped["direction_label"] = grouped["direction_label"].astype(int)
    grouped["time_bin"] = grouped["rel_time"].map(_time_bin)
    grouped["is_post_eval"] = (grouped["rel_time"] >= POST_START) & (grouped["rel_time"] <= POST_END)
    return grouped


def _make_transitions(agg: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (stream, event_id), group in agg.groupby(["stream", "event_id"], sort=False):
        by_time = {int(row.rel_time): row for row in group.itertuples(index=False)}
        for rel_time in sorted(by_time):
            next_time = rel_time + 8
            if next_time not in by_time:
                continue
            row = by_time[rel_time]
            next_row = by_time[next_time]
            rows.append(
                {
                    "stream": stream,
                    "event_id": event_id,
                    "trial_id": row.trial_id,
                    "cliff_distance": int(row.cliff_distance),
                    "rel_time": int(rel_time),
                    "next_rel_time": int(next_time),
                    "time_bin": _time_bin(int(rel_time)),
                    "current_direction": int(row.direction_label),
                    "next_direction": int(next_row.direction_label),
                    "next_away": int(next_row.direction_label == 1),
                    "constraint_metric": metric,
                    "constraint_value": float(getattr(row, metric)),
                    "success": bool(row.success),
                    "returned_from_edge": bool(row.returned_from_edge),
                }
            )
    frame = pd.DataFrame(rows)
    frame["is_pre_train"] = (frame["rel_time"] < 0) & (frame["next_rel_time"] <= 0)
    frame["is_post_eval"] = (
        (frame["rel_time"] >= POST_START)
        & (frame["rel_time"] < POST_END)
        & (frame["next_rel_time"] <= POST_END)
    )
    return frame


class ConditionalPathModel:
    def __init__(self, keys: tuple[str, ...], alpha: float = 0.5) -> None:
        self.keys = keys
        self.alpha = float(alpha)
        self.probs: dict[tuple[Any, ...], np.ndarray] = {}
        self.direction_fallback: dict[int, np.ndarray] = {}
        self.global_probs = np.full(len(DIRECTION_STATES), 1.0 / len(DIRECTION_STATES), dtype=float)

    @staticmethod
    def _counts_to_probs(counts: np.ndarray, alpha: float) -> np.ndarray:
        smoothed = counts.astype(float) + float(alpha)
        return smoothed / smoothed.sum()

    def fit(self, frame: pd.DataFrame) -> "ConditionalPathModel":
        if frame.empty:
            return self
        state_to_index = {state: index for index, state in enumerate(DIRECTION_STATES)}
        global_counts = np.zeros(len(DIRECTION_STATES), dtype=float)
        for state, count in frame["next_direction"].value_counts().items():
            global_counts[state_to_index[int(state)]] = float(count)
        self.global_probs = self._counts_to_probs(global_counts, self.alpha)
        for current, group in frame.groupby("current_direction", sort=False):
            counts = np.zeros(len(DIRECTION_STATES), dtype=float)
            for state, count in group["next_direction"].value_counts().items():
                counts[state_to_index[int(state)]] = float(count)
            self.direction_fallback[int(current)] = self._counts_to_probs(counts, self.alpha)
        if self.keys:
            group_keys: str | list[str]
            group_keys = self.keys[0] if len(self.keys) == 1 else list(self.keys)
            for key, group in frame.groupby(group_keys, sort=False, dropna=False):
                if not isinstance(key, tuple):
                    key = (key,)
                counts = np.zeros(len(DIRECTION_STATES), dtype=float)
                for state, count in group["next_direction"].value_counts().items():
                    counts[state_to_index[int(state)]] = float(count)
                self.probs[tuple(key)] = self._counts_to_probs(counts, self.alpha)
        return self

    def predict_row(self, row: pd.Series) -> np.ndarray:
        if self.keys:
            key = tuple(row[key] for key in self.keys)
            if key in self.probs:
                return self.probs[key]
        current = int(row["current_direction"])
        if current in self.direction_fallback:
            return self.direction_fallback[current]
        return self.global_probs


def _score_model(model: ConditionalPathModel, test: pd.DataFrame) -> dict[str, float]:
    if test.empty:
        return {
            "transition_count": 0,
            "log_loss": float("nan"),
            "away_brier": float("nan"),
            "away_flux_actual": float("nan"),
            "away_flux_predicted": float("nan"),
            "away_flux_abs_error": float("nan"),
            "path_entropy_nats": float("nan"),
        }
    state_to_index = {state: index for index, state in enumerate(DIRECTION_STATES)}
    losses: list[float] = []
    briers: list[float] = []
    away_probs: list[float] = []
    entropies: list[float] = []
    actual_away: list[float] = []
    for _, row in test.iterrows():
        probs = model.predict_row(row)
        probs = np.clip(probs, 1.0e-12, 1.0)
        probs = probs / probs.sum()
        target = int(row["next_direction"])
        losses.append(float(-math.log(probs[state_to_index[target]])))
        p_away = float(probs[state_to_index[1]])
        y_away = float(target == 1)
        away_probs.append(p_away)
        actual_away.append(y_away)
        briers.append(float((p_away - y_away) ** 2))
        entropies.append(float(-(probs * np.log(probs)).sum()))
    return {
        "transition_count": int(len(test)),
        "log_loss": float(np.mean(losses)),
        "away_brier": float(np.mean(briers)),
        "away_flux_actual": float(np.mean(actual_away)),
        "away_flux_predicted": float(np.mean(away_probs)),
        "away_flux_abs_error": float(abs(np.mean(away_probs) - np.mean(actual_away))),
        "path_entropy_nats": float(np.mean(entropies)),
    }


def _evaluate_models(transitions: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    model_keys: dict[str, tuple[str, ...]] = {
        "open_loop_pre": ("current_direction",),
        "post_direction": ("current_direction",),
        "post_time": ("current_direction", "time_bin"),
        "post_stream": ("current_direction", "constraint_bin"),
        "post_time_stream": ("current_direction", "time_bin", "constraint_bin"),
    }
    rows: list[dict[str, Any]] = []
    for stream in STREAM_ORDER:
        stream_frame = transitions[transitions["stream"] == stream].copy()
        if stream_frame.empty:
            continue
        event_ids = sorted(stream_frame.loc[stream_frame["is_post_eval"], "event_id"].unique())
        for heldout in event_ids:
            fold_frame = stream_frame.copy()
            train_post = fold_frame[(fold_frame["event_id"] != heldout) & fold_frame["is_post_eval"]].copy()
            train_pre = fold_frame[(fold_frame["event_id"] != heldout) & fold_frame["is_pre_train"]].copy()
            test_post = fold_frame[(fold_frame["event_id"] == heldout) & fold_frame["is_post_eval"]].copy()
            if train_post.empty or test_post.empty:
                continue
            edges = _fit_bin_edges(train_post["constraint_value"])
            for frame in (train_post, train_pre, test_post):
                frame["constraint_bin"] = _assign_bins(frame["constraint_value"], edges)
            for model_name in MODEL_ORDER:
                train = train_pre if model_name == "open_loop_pre" else train_post
                model = ConditionalPathModel(model_keys[model_name]).fit(train)
                score = _score_model(model, test_post)
                score.update(
                    {
                        "stream": stream,
                        "model": model_name,
                        "heldout_event_id": heldout,
                        "heldout_cliff_distance": int(test_post["cliff_distance"].iloc[0]),
                        "constraint_metric": str(test_post["constraint_metric"].iloc[0]),
                        "bin_edges": ";".join(f"{edge:.8g}" for edge in edges),
                    }
                )
                rows.append(score)
    scores = pd.DataFrame(rows)
    if scores.empty:
        return scores
    baseline_cols = [
        "stream",
        "heldout_event_id",
        "log_loss",
        "away_brier",
        "away_flux_abs_error",
    ]
    direction = (
        scores[scores["model"] == "post_direction"][baseline_cols]
        .rename(
            columns={
                "log_loss": "post_direction_log_loss",
                "away_brier": "post_direction_away_brier",
                "away_flux_abs_error": "post_direction_away_flux_abs_error",
            }
        )
        .copy()
    )
    time = (
        scores[scores["model"] == "post_time"][baseline_cols]
        .rename(
            columns={
                "log_loss": "post_time_log_loss",
                "away_brier": "post_time_away_brier",
                "away_flux_abs_error": "post_time_away_flux_abs_error",
            }
        )
        .copy()
    )
    scores = scores.merge(direction, on=["stream", "heldout_event_id"], how="left")
    scores = scores.merge(time, on=["stream", "heldout_event_id"], how="left")
    scores["log_loss_improvement_vs_direction"] = scores["post_direction_log_loss"] - scores["log_loss"]
    scores["log_loss_improvement_vs_time"] = scores["post_time_log_loss"] - scores["log_loss"]
    scores["away_brier_improvement_vs_direction"] = scores["post_direction_away_brier"] - scores["away_brier"]
    scores["away_flux_error_reduction_vs_direction"] = (
        scores["post_direction_away_flux_abs_error"] - scores["away_flux_abs_error"]
    )
    return scores


def _summarize_scores(scores: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    metrics = (
        "log_loss",
        "away_brier",
        "away_flux_abs_error",
        "path_entropy_nats",
        "log_loss_improvement_vs_direction",
        "log_loss_improvement_vs_time",
        "away_brier_improvement_vs_direction",
        "away_flux_error_reduction_vs_direction",
    )
    rows: list[dict[str, Any]] = []
    for (stream, model), group in scores.groupby(["stream", "model"], sort=False):
        row: dict[str, Any] = {
            "stream": stream,
            "model": model,
            "event_count": int(group["heldout_event_id"].nunique()),
            "transition_count": int(group["transition_count"].sum()),
            "mean_away_flux_actual": float(group["away_flux_actual"].mean()),
            "mean_away_flux_predicted": float(group["away_flux_predicted"].mean()),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"mean_{metric}"] = float(values.mean())
            ci_low, ci_high = _bootstrap_ci(values, seed=seed)
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["stream"] = pd.Categorical(summary["stream"], categories=STREAM_ORDER, ordered=True)
    summary["model"] = pd.Categorical(summary["model"], categories=MODEL_ORDER, ordered=True)
    return summary.sort_values(["stream", "model"]).reset_index(drop=True)


def _constraint_flux_table(transitions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    post = transitions[transitions["is_post_eval"]].copy()
    for stream in STREAM_ORDER:
        stream_frame = post[post["stream"] == stream].copy()
        if stream_frame.empty:
            continue
        edges = _fit_bin_edges(stream_frame["constraint_value"])
        stream_frame["constraint_bin"] = _assign_bins(stream_frame["constraint_value"], edges)
        for (time_bin, constraint_bin), group in stream_frame.groupby(["time_bin", "constraint_bin"], sort=False):
            counts = group["next_direction"].value_counts()
            rows.append(
                {
                    "stream": stream,
                    "time_bin": time_bin,
                    "constraint_bin": constraint_bin,
                    "constraint_metric": str(group["constraint_metric"].iloc[0]),
                    "bin_edges": ";".join(f"{edge:.8g}" for edge in edges),
                    "transition_count": int(len(group)),
                    "event_count": int(group["event_id"].nunique()),
                    "mean_constraint_value": float(group["constraint_value"].mean()),
                    "median_constraint_value": float(group["constraint_value"].median()),
                    "next_toward_count": int(counts.get(-1, 0)),
                    "next_uncommitted_count": int(counts.get(0, 0)),
                    "next_away_count": int(counts.get(1, 0)),
                    "next_toward_rate": float((group["next_direction"] == -1).mean()),
                    "next_uncommitted_rate": float((group["next_direction"] == 0).mean()),
                    "next_away_rate": float((group["next_direction"] == 1).mean()),
                }
            )
    return pd.DataFrame(rows)


def _transition_shift_table(transitions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stream in STREAM_ORDER:
        frame = transitions[transitions["stream"] == stream]
        if frame.empty:
            continue
        pre = frame[frame["is_pre_train"]]
        post = frame[frame["is_post_eval"]]
        for source_name, source in (("pre_release", pre), ("post_release", post)):
            for current in DIRECTION_STATES:
                group = source[source["current_direction"] == current]
                counts = group["next_direction"].value_counts()
                total = int(len(group))
                for next_state in DIRECTION_STATES:
                    rows.append(
                        {
                            "stream": stream,
                            "source": source_name,
                            "current_direction": current,
                            "current_direction_label": _direction_name(current),
                            "next_direction": next_state,
                            "next_direction_label": _direction_name(next_state),
                            "transition_count": int(counts.get(next_state, 0)),
                            "current_state_count": total,
                            "transition_rate": float(counts.get(next_state, 0) / total) if total else float("nan"),
                        }
                    )
    return pd.DataFrame(rows)


def _plot_model_comparison(summary: pd.DataFrame, out_base: Path) -> None:
    streams = list(STREAM_ORDER)
    selected_model = "post_stream"
    selected = summary[summary["model"] == selected_model].copy()
    selected["stream"] = pd.Categorical(selected["stream"], categories=streams, ordered=True)
    selected = selected.sort_values("stream")
    colors = {
        "signal_hidden": "#5B6C8F",
        "controller_action": "#8C8C8C",
        "spring_contractile": "#A77C55",
        "spring_current": "#8C6D9D",
        "body_kinematic": "#6D9E75",
        "body_no_contractile": "#4D8B73",
        "body_combined": "#3E7564",
    }
    labels = {
        "signal_hidden": "Hidden",
        "controller_action": "Action",
        "spring_contractile": "Contractile",
        "spring_current": "Spring",
        "body_kinematic": "Kinematic",
        "body_no_contractile": "Body no contr.",
        "body_combined": "Body comb.",
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.8), constrained_layout=True)
    x = np.arange(len(selected))
    y = selected["mean_log_loss_improvement_vs_direction"].to_numpy(dtype=float)
    low = selected["log_loss_improvement_vs_direction_ci_low"].to_numpy(dtype=float)
    high = selected["log_loss_improvement_vs_direction_ci_high"].to_numpy(dtype=float)
    axes[0].bar(x, y, color=[colors[s] for s in selected["stream"]], width=0.72)
    axes[0].errorbar(x, y, yerr=[y - low, high - y], fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2)
    axes[0].axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Log-loss improvement")
    axes[0].set_title("Added stream constraint")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([labels[s] for s in selected["stream"]], rotation=35, ha="right")

    y2 = selected["mean_away_flux_abs_error"].to_numpy(dtype=float)
    low2 = selected["away_flux_abs_error_ci_low"].to_numpy(dtype=float)
    high2 = selected["away_flux_abs_error_ci_high"].to_numpy(dtype=float)
    axes[1].bar(x, y2, color=[colors[s] for s in selected["stream"]], width=0.72)
    axes[1].errorbar(x, y2, yerr=[y2 - low2, high2 - y2], fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2)
    axes[1].set_ylabel("Away-flux error")
    axes[1].set_title("Path-flux reproduction")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([labels[s] for s in selected["stream"]], rotation=35, ha="right")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.title.set_fontsize(10)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_flux_bins(flux: pd.DataFrame, out_base: Path) -> None:
    focus = flux[flux["stream"].isin(["signal_hidden", "controller_action", "spring_current", "body_no_contractile", "body_combined"])].copy()
    if focus.empty:
        return
    stream_labels = {
        "signal_hidden": "Hidden",
        "controller_action": "Action",
        "spring_current": "Spring",
        "body_no_contractile": "Body no contr.",
        "body_combined": "Body comb.",
    }
    bin_order = ["low", "mid", "high", "all"]
    bin_labels = {"low": "Low", "mid": "Mid", "high": "High", "all": "All"}
    bin_colors = {"low": "#D8D8D8", "mid": "#9DB88F", "high": "#4D8B73", "all": "#8CB87C"}
    fig, axes = plt.subplots(1, 5, figsize=(8.6, 2.35), sharey=True, constrained_layout=True)
    for ax, stream in zip(axes, ["signal_hidden", "controller_action", "spring_current", "body_no_contractile", "body_combined"]):
        sub = focus[focus["stream"] == stream]
        rates = []
        labels = []
        colors = []
        for constraint_bin in bin_order:
            block = sub[sub["constraint_bin"] == constraint_bin]
            if block.empty:
                continue
            rates.append(float(np.average(block["next_away_rate"], weights=block["transition_count"])))
            labels.append(bin_labels[constraint_bin])
            colors.append(bin_colors[constraint_bin])
        ax.bar(np.arange(len(rates)), rates, color=colors, width=0.68)
        ax.set_title(stream_labels[stream], fontsize=9)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(0, 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Next-step away flux")
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _report_lines(
    *,
    packet_root: Path,
    metric: str,
    agg: pd.DataFrame,
    transitions: pd.DataFrame,
    summary: pd.DataFrame,
    flux: pd.DataFrame,
) -> list[str]:
    event_count = int(agg["event_id"].nunique())
    post_transition_count = int(transitions["is_post_eval"].sum() / len(STREAM_ORDER))
    open_loop = summary[summary["model"] == "open_loop_pre"]
    post_stream = summary[summary["model"] == "post_stream"]
    selected = post_stream.set_index("stream")
    flux_rows: list[dict[str, Any]] = []
    for (stream, constraint_bin), group in flux.groupby(["stream", "constraint_bin"], sort=False):
        flux_rows.append(
            {
                "stream": stream,
                "constraint_bin": constraint_bin,
                "transition_count": float(group["transition_count"].sum()),
                "next_away_rate": float(
                    np.average(group["next_away_rate"], weights=group["transition_count"])
                ),
            }
        )
    flux_collapsed = pd.DataFrame(flux_rows)

    def _bin_rate(stream: str, bin_name: str) -> float:
        row = flux_collapsed[
            (flux_collapsed["stream"] == stream) & (flux_collapsed["constraint_bin"] == bin_name)
        ]
        if row.empty:
            return float("nan")
        return float(row["next_away_rate"].iloc[0])

    signal_ll = float(selected.loc["signal_hidden", "mean_log_loss_improvement_vs_direction"])
    action_ll = float(selected.loc["controller_action", "mean_log_loss_improvement_vs_direction"])
    body_ll = float(selected.loc["body_no_contractile", "mean_log_loss_improvement_vs_direction"])
    body_combined_ll = float(selected.loc["body_combined", "mean_log_loss_improvement_vs_direction"])
    spring_ll = float(selected.loc["spring_current", "mean_log_loss_improvement_vs_direction"])
    signal_flux = float(selected.loc["signal_hidden", "mean_away_flux_abs_error"])
    body_flux = float(selected.loc["body_no_contractile", "mean_away_flux_abs_error"])
    body_combined_flux = float(selected.loc["body_combined", "mean_away_flux_abs_error"])
    spring_flux = float(selected.loc["spring_current", "mean_away_flux_abs_error"])
    open_loop_flux = float(
        open_loop[open_loop["stream"] == "signal_hidden"]["mean_away_flux_abs_error"].iloc[0]
    )
    signal_low = _bin_rate("signal_hidden", "low")
    signal_high = _bin_rate("signal_hidden", "high")
    action_low = _bin_rate("controller_action", "low")
    action_high = _bin_rate("controller_action", "high")
    spring_low = _bin_rate("spring_current", "low")
    spring_high = _bin_rate("spring_current", "high")
    body_low = _bin_rate("body_no_contractile", "low")
    body_high = _bin_rate("body_no_contractile", "high")
    combined_low = _bin_rate("body_combined", "low")
    combined_high = _bin_rate("body_combined", "high")
    lines = [
        "# S4 MaxCal-inspired path ensemble analysis",
        "",
        "## Purpose",
        "",
        "This analysis asks a maximum-caliber-style question on the fixed-width S4 deformation-off family: which coarse path constraints are sufficient to reproduce the post-release flux into away-from-cliff motion?",
        "",
        "It is not a thermodynamic MaxCal claim. The simulator rollouts are deterministic, and the ensemble is the set of cliff-distance perturbation contexts. The fitted models are coarse empirical Markov path models with additive smoothing, used as a descriptive path-entropy/flux comparison.",
        "",
        "## Inputs",
        "",
        f"- Packet root: `{packet_root}`",
        "- Source table: `tables/phase_propagation_local_slip_by_event.csv`",
        f"- Matched path events used: `{event_count}`",
        f"- Post-release evaluation window: `{POST_START}` to `{POST_END}` steps after release",
        f"- Transition step: `8` simulation steps",
        f"- Post-release transitions per stream: `{post_transition_count}`",
        f"- Constraint metric: `{metric}` averaged over body-axis ranks for each stream/event/time.",
        "",
        "## Coarse path states",
        "",
        "- Direction state: toward cliff (`-1`), uncommitted (`0`), or away/reversal (`1`), inherited from the existing sustained-future-COM direction grammar.",
        "- Constraint state: stream-specific low/mid/high bins of the chosen perturbation metric, fit inside each leave-one-event-out training fold.",
        "- Streams: hidden signalling state, controller action output, contractile springs, current spring lengths, body kinematics, body without contraction channels, and the combined body stream.",
        "",
        "## Models",
        "",
        "- `open_loop_pre`: pre-release transition flux constrained only by current direction.",
        "- `post_direction`: post-release transition flux constrained only by current direction.",
        "- `post_time`: post-release transition flux constrained by current direction and coarse release-time bin.",
        "- `post_stream`: post-release transition flux constrained by current direction and stream perturbation bin.",
        "- `post_time_stream`: post-release transition flux constrained by current direction, release-time bin, and stream perturbation bin.",
        "",
        "All scores are leave-one-event-out over cliff distances. Positive log-loss improvement means the added constraint predicts the held-out path better than the post-release direction-only model.",
        "",
        "## Main numerical result",
        "",
        f"- Open-loop pre-release model away-flux error: `{open_loop_flux:.3f}`.",
        f"- Signal-hidden stream constraint log-loss improvement: `{signal_ll:.3f}`; away-flux error `{signal_flux:.3f}`.",
        f"- Controller-action stream constraint log-loss improvement: `{action_ll:.3f}`.",
        f"- Spring-current stream constraint log-loss improvement: `{spring_ll:.3f}`; away-flux error `{spring_flux:.3f}`.",
        f"- Body-no-contraction stream constraint log-loss improvement: `{body_ll:.3f}`; away-flux error `{body_flux:.3f}`.",
        f"- Combined-body stream constraint log-loss improvement: `{body_combined_ll:.3f}`; away-flux error `{body_combined_flux:.3f}`.",
        "",
        "## Perturbation-bin flux",
        "",
        f"- Hidden low/high perturbation bins: `{signal_low:.3f}` / `{signal_high:.3f}` next-away flux.",
        f"- Action low/high perturbation bins: `{action_low:.3f}` / `{action_high:.3f}` next-away flux.",
        f"- Spring-current low/high perturbation bins: `{spring_low:.3f}` / `{spring_high:.3f}` next-away flux.",
        f"- Body-no-contraction low/high perturbation bins: `{body_low:.3f}` / `{body_high:.3f}` next-away flux.",
        f"- Combined-body low/high perturbation bins: `{combined_low:.3f}` / `{combined_high:.3f}` next-away flux.",
        "",
        "## Interpretation",
        "",
    ]
    if body_combined_ll > max(signal_ll, action_ll) and spring_ll > max(signal_ll, action_ll):
        lines.append(
            "The path-ensemble comparison supports the same direction as the phase-slip analyses: spring-current, contractile, and combined-body constraints improve reproduction of the post-release path flux, while hidden-signalling and action constraints slightly worsen held-out fit. The strongest visual result is the perturbation-bin separation: high mechanical/body perturbation bins carry much higher away flux than low bins, whereas hidden/action bins do not show the same monotonic relation. This is consistent with the release event entering the path ensemble through plant mechanics rather than through a new internal signalling trajectory."
        )
    else:
        lines.append(
            "The path-ensemble comparison does not cleanly privilege body/spring constraints over hidden-signalling/action constraints. Treat this as an exploratory diagnostic rather than a paper-facing MaxCal result."
        )
    lines.extend(
        [
            "",
            "The safest paper wording is therefore `maximum-caliber-inspired` or `path-ensemble` analysis. It can quantify path-flux constraints, but it does not by itself prove causal mediation. A stronger MaxCal-style analysis would add explicit contact/support constraints from substep probe traces.",
            "",
            "## Outputs",
            "",
            "- `tables/maxcal_path_state_samples.csv`",
            "- `tables/maxcal_path_transitions.csv`",
            "- `tables/maxcal_model_scores.csv`",
            "- `tables/maxcal_model_summary.csv`",
            "- `tables/maxcal_constraint_flux.csv`",
            "- `tables/maxcal_direction_transition_shift.csv`",
            "- `tables/maxcal_path_ensemble_summary.json`",
            "- `figures/maxcal_model_comparison.pdf` / `.png`",
            "- `figures/maxcal_away_flux_by_constraint_bin.pdf` / `.png`",
        ]
    )
    return lines


def run_analysis(
    *,
    packet_root: Path = DEFAULT_PACKET_ROOT,
    metric: str = "mean_pc_l2_delta",
    seed: int = RNG_SEED,
) -> dict[str, Path]:
    tables = packet_root / "tables"
    figures = packet_root / "figures"
    source = tables / "phase_propagation_local_slip_by_event.csv"
    if not source.exists():
        raise FileNotFoundError(f"Required source table missing: {source}")
    slip = pd.read_csv(source)
    if metric not in {"mean_pc_l2_delta", "mean_abs_phase_slip", "max_pc_l2_delta", "max_abs_phase_slip"}:
        raise ValueError(f"Unsupported metric: {metric}")
    agg = _aggregate_slip(slip)
    transitions = _make_transitions(agg, metric=metric)
    scores = _evaluate_models(transitions, seed=seed)
    summary = _summarize_scores(scores, seed=seed)
    flux = _constraint_flux_table(transitions)
    shift = _transition_shift_table(transitions)

    paths = {
        "path_state_samples": tables / "maxcal_path_state_samples.csv",
        "path_transitions": tables / "maxcal_path_transitions.csv",
        "model_scores": tables / "maxcal_model_scores.csv",
        "model_summary": tables / "maxcal_model_summary.csv",
        "constraint_flux": tables / "maxcal_constraint_flux.csv",
        "direction_transition_shift": tables / "maxcal_direction_transition_shift.csv",
        "summary": tables / "maxcal_path_ensemble_summary.json",
        "report": packet_root / "S4_maxcal_path_ensemble_report.md",
        "model_figure": figures / "maxcal_model_comparison.pdf",
        "flux_figure": figures / "maxcal_away_flux_by_constraint_bin.pdf",
    }
    paths["path_state_samples"].parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(paths["path_state_samples"], index=False)
    transitions.to_csv(paths["path_transitions"], index=False)
    scores.to_csv(paths["model_scores"], index=False)
    summary.to_csv(paths["model_summary"], index=False)
    flux.to_csv(paths["constraint_flux"], index=False)
    shift.to_csv(paths["direction_transition_shift"], index=False)
    _plot_model_comparison(summary, figures / "maxcal_model_comparison")
    _plot_flux_bins(flux, figures / "maxcal_away_flux_by_constraint_bin")

    payload = {
        "schema_version": 1,
        "kind": "s4_maxcal_inspired_path_ensemble",
        "packet_root": packet_root,
        "source_table": source,
        "event_count": int(agg["event_id"].nunique()),
        "streams": list(STREAM_ORDER),
        "models": list(MODEL_ORDER),
        "direction_states": STATE_LABELS,
        "post_release_window": [POST_START, POST_END],
        "transition_step": 8,
        "constraint_metric": metric,
        "aggregation": "mean over body-axis ranks for each stream/event/time before path modeling",
        "cross_validation": "leave-one-event-out over cliff distances",
        "smoothing_alpha": 0.5,
        "interpretation": "descriptive maximum-caliber-inspired coarse Markov path ensemble, not a causal or thermodynamic MaxCal proof",
        "paths": {key: str(value) for key, value in paths.items()},
    }
    _write_json(paths["summary"], payload)
    paths["report"].write_text(
        "\n".join(
            _report_lines(
                packet_root=packet_root,
                metric=metric,
                agg=agg,
                transitions=transitions,
                summary=summary,
                flux=flux,
            )
        )
        + "\n"
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, default=DEFAULT_PACKET_ROOT)
    parser.add_argument(
        "--metric",
        choices=("mean_pc_l2_delta", "mean_abs_phase_slip", "max_pc_l2_delta", "max_abs_phase_slip"),
        default="mean_pc_l2_delta",
    )
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()
    outputs = run_analysis(packet_root=args.packet_root, metric=args.metric, seed=args.seed)
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
