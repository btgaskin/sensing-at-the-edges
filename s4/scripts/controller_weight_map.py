from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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


def _ensure_import_path() -> None:
    path = str(SENSING_EDGES_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _save_heatmap(
    matrix: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    xlabels: list[str],
    ylabels: list[str],
    output_path: Path,
    cmap: str = "coolwarm",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_h = max(4.0, 0.28 * len(ylabels) + 1.8)
    fig_w = max(6.0, 0.36 * len(xlabels) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = float(np.max(np.abs(matrix))) if matrix.size else 1.0
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_bar(values: np.ndarray, *, labels: list[str], title: str, ylabel: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_w = max(6.0, 0.36 * len(labels) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4.0))
    ax.bar(range(len(labels)), values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _input_labels(cfg, w_input_rows: int) -> list[str]:
    _ensure_import_path()
    from sensing_edges.nca import normalize_active_comm_filters

    private = [
        "abs_h_delta",
        "abs_v_delta",
        "comp_h_delta",
        "comp_v_delta",
        "h_contraction",
        "v_contraction",
    ]
    labels = list(private)
    extra_private = int(w_input_rows) - len(private)

    filters = list(normalize_active_comm_filters(cfg.nca.get("active_comm_filters")))
    grid_hidden = int(cfg.nca.ctrnn_grid_hidden)
    comm_labels = [f"{filter_name}:gh{index}" for filter_name in filters for index in range(grid_hidden)]
    if len(labels) + len(comm_labels) == w_input_rows:
        return labels + comm_labels
    if len(labels) + 1 + len(comm_labels) == w_input_rows:
        return labels + ["ou_noise"] + comm_labels
    if extra_private > 0:
        labels.extend(f"private_{index}" for index in range(extra_private))
    remaining = w_input_rows - len(labels)
    labels.extend(f"comm_{index}" for index in range(remaining))
    return labels


def _output_labels(cfg, w_output_cols: int) -> list[str]:
    labels = ["act_h", "act_v"]
    grid_hidden = int(cfg.nca.ctrnn_grid_hidden)
    labels.extend(f"gh{index}" for index in range(grid_hidden))
    if len(labels) < w_output_cols:
        labels.extend(f"out_{index}" for index in range(len(labels), w_output_cols))
    return labels[:w_output_cols]


def run_weight_map(
    *,
    controller_params_path: Path,
    config_snapshot_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    cfg = OmegaConf.load(config_snapshot_path)
    with np.load(controller_params_path, allow_pickle=False) as data:
        w_input = np.asarray(data["w_input"], dtype=np.float32)
        w_rec = np.asarray(data["w_rec"], dtype=np.float32)
        w_output = np.asarray(data["w_output"], dtype=np.float32)
        bias = np.asarray(data["bias"], dtype=np.float32)
        log_tau = np.asarray(data["log_tau"], dtype=np.float32)

    hidden_labels = [f"h{index}" for index in range(w_rec.shape[0])]
    input_labels = _input_labels(cfg, w_input.shape[0])
    output_labels = _output_labels(cfg, w_output.shape[1])
    tau_min = float(cfg.nca.ctrnn_tau_min)
    tau_max = float(cfg.nca.ctrnn_tau_max)
    tau = tau_min + (tau_max - tau_min) * _sigmoid(log_tau)
    eigvals = np.linalg.eigvals(w_rec.astype(np.float64))
    spectral_radius = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0

    figures_dir = output_dir / "figures"
    _save_heatmap(
        w_input,
        title="CTRNN Input Weights",
        xlabel="Hidden Unit",
        ylabel="Perception Feature",
        xlabels=hidden_labels,
        ylabels=input_labels,
        output_path=figures_dir / "w_input.png",
    )
    _save_heatmap(
        w_rec,
        title="CTRNN Recurrent Weights",
        xlabel="Target Hidden Unit",
        ylabel="Source Hidden Unit",
        xlabels=hidden_labels,
        ylabels=hidden_labels,
        output_path=figures_dir / "w_rec.png",
    )
    _save_heatmap(
        w_output.T,
        title="CTRNN Output Weights",
        xlabel="Hidden Unit",
        ylabel="Controller Output",
        xlabels=hidden_labels,
        ylabels=output_labels,
        output_path=figures_dir / "w_output.png",
    )
    _save_bar(
        np.asarray(bias, dtype=np.float32),
        labels=hidden_labels,
        title="CTRNN Bias",
        ylabel="bias",
        output_path=figures_dir / "bias.png",
    )
    _save_bar(
        np.asarray(tau, dtype=np.float32),
        labels=hidden_labels,
        title="CTRNN Time Constants",
        ylabel="tau",
        output_path=figures_dir / "tau.png",
    )
    _save_bar(
        np.linalg.norm(w_rec, axis=1),
        labels=hidden_labels,
        title="Recurrent Row Norm",
        ylabel="L2 norm",
        output_path=figures_dir / "w_rec_row_norms.png",
    )

    np.savez_compressed(
        output_dir / "weight_matrices.npz",
        w_input=w_input,
        w_rec=w_rec,
        w_output=w_output,
        bias=bias,
        log_tau=log_tau,
        tau=np.asarray(tau, dtype=np.float32),
    )

    summary = {
        "controller_params_path": str(controller_params_path),
        "config_snapshot_path": str(config_snapshot_path),
        "hidden_count": int(w_rec.shape[0]),
        "input_dim": int(w_input.shape[0]),
        "output_dim": int(w_output.shape[1]),
        "tau": {
            "min": float(np.min(tau)),
            "max": float(np.max(tau)),
            "mean": float(np.mean(tau)),
            "median": float(np.median(tau)),
        },
        "spectral_radius": spectral_radius,
        "matrix_stats": {
            "w_input": {
                "shape": list(w_input.shape),
                "min": float(np.min(w_input)),
                "max": float(np.max(w_input)),
                "mean_abs": float(np.mean(np.abs(w_input))),
            },
            "w_rec": {
                "shape": list(w_rec.shape),
                "min": float(np.min(w_rec)),
                "max": float(np.max(w_rec)),
                "mean_abs": float(np.mean(np.abs(w_rec))),
            },
            "w_output": {
                "shape": list(w_output.shape),
                "min": float(np.min(w_output)),
                "max": float(np.max(w_output)),
                "mean_abs": float(np.mean(np.abs(w_output))),
            },
        },
        "labels": {
            "inputs": input_labels,
            "hidden": hidden_labels,
            "outputs": output_labels,
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Map the saved CTRNN weight matrices for the canonical s4 specimen.")
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
        help="Output directory for summaries and figures.",
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
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (bundle_root / "analysis_runs" / "controller_weight_map").resolve()
    )
    summary = run_weight_map(
        controller_params_path=controller_params_path,
        config_snapshot_path=config_snapshot_path,
        output_dir=output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
