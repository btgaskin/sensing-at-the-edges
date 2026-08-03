from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
S4_ROOT = SCRIPT_DIR.parent
DEFAULT_BUNDLE_ROOT = S4_ROOT / "results" / "bin_03_rank_00_cell_6-0-3_flat_0423" / "source_runs" / "current_core_full" / "pca_bundle"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_states(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as data:
        key = data.files[0]
        return np.asarray(data[key], dtype=np.float32)


def _fit_pca(x: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0, dtype=np.float64)
    centered = x.astype(np.float64) - mean
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:n_components].astype(np.float32)
    total_var = np.sum(singular_values ** 2)
    explained = (
        (singular_values[:n_components] ** 2) / total_var if total_var > 0 else np.zeros((n_components,), dtype=np.float64)
    )
    return mean.astype(np.float32), components, explained.astype(np.float32)


def _project(x: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    centered = x.astype(np.float32) - mean[None, :]
    return centered @ components.T


def _rfft_summary(x: np.ndarray, *, sample_spacing_env: float, top_k: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    demeaned = x - np.mean(x)
    if demeaned.size < 2:
        freqs = np.zeros((0,), dtype=np.float32)
        amps = np.zeros((0,), dtype=np.float32)
        return freqs, amps, []
    window = np.hanning(demeaned.size).astype(np.float32)
    windowed = demeaned.astype(np.float32) * window
    amps = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(windowed.size, d=sample_spacing_env).astype(np.float32)
    if amps.size > 0:
        amps[0] = 0.0
    ranked = np.argsort(amps)[::-1][:top_k]
    peaks = [
        {
            "frequency_cycles_per_env_step": float(freqs[index]),
            "amplitude": float(amps[index]),
        }
        for index in ranked
    ]
    return freqs, amps.astype(np.float32), peaks


def _save_overlay_spectra(
    *,
    intact_freqs: np.ndarray,
    intact_amps: np.ndarray,
    ablated_freqs: np.ndarray,
    ablated_amps: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.plot(intact_freqs, intact_amps, label="intact", linewidth=2.0)
    ax.plot(ablated_freqs, ablated_amps, label="deformation_off", linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("cycles / env step")
    ax.set_ylabel("amplitude")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_frequency_analysis(
    *,
    bundle_dir: Path,
    output_dir: Path,
    modality: str = "signaling",
    n_components: int = 3,
    top_k_peaks: int = 5,
) -> dict[str, object]:
    intact_dir = bundle_dir
    ablated_dir = bundle_dir / "cohorts" / "deformation_off_native_core"

    intact_summary = _load_json(intact_dir / "cohort_summary.json")
    ablated_summary = _load_json(ablated_dir / "cohort_summary.json")
    intact_manifest = _load_json(intact_dir / "trace_manifest.json")
    ablated_manifest = _load_json(ablated_dir / "trace_manifest.json")
    intact_labels = pd.read_csv(intact_dir / "trace_labels.csv")
    ablated_labels = pd.read_csv(ablated_dir / "trace_labels.csv")

    intact_trial_id = str(intact_summary["selected_trial_ids"][0])
    ablated_trial_id = str(ablated_summary["selected_trial_ids"][0])

    if modality == "signaling":
        intact_states = _load_states(intact_dir / "nca_states.npz")[..., 6:12]
        ablated_states = _load_states(ablated_dir / "nca_states.npz")[..., 6:12]
    elif modality == "ctrnn":
        intact_states = _load_states(intact_dir / "ctrnn_states.npz")
        ablated_states = _load_states(ablated_dir / "ctrnn_states.npz")
    elif modality == "actions":
        intact_states = _load_states(intact_dir / "actions.npz")
        ablated_states = _load_states(ablated_dir / "actions.npz")
    else:
        raise ValueError(f"Unsupported modality={modality!r}")

    intact_mask = (intact_labels["trial_id"] == intact_trial_id) & (intact_labels["micro_index"] == 0)
    ablated_mask = (ablated_labels["trial_id"] == ablated_trial_id) & (ablated_labels["micro_index"] == 0)
    intact_x = intact_states[np.asarray(intact_mask, dtype=bool)].reshape(int(np.sum(intact_mask)), -1)
    ablated_x = ablated_states[np.asarray(ablated_mask, dtype=bool)].reshape(int(np.sum(ablated_mask)), -1)

    mean, components, explained = _fit_pca(intact_x, n_components=n_components)
    intact_proj = _project(intact_x, mean, components)
    ablated_proj = _project(ablated_x, mean, components)

    sense_act_phases = int(intact_manifest["sense_act_phases"])
    sample_spacing_env = 1.0 / float(sense_act_phases)

    component_summaries: list[dict[str, object]] = []
    aggregate_intact = None
    aggregate_ablated = None
    aggregate_intact_freqs = None
    aggregate_ablated_freqs = None
    figures_dir = output_dir / "figures"
    for index in range(n_components):
        freqs_i, amps_intact, peaks_intact = _rfft_summary(
            intact_proj[:, index],
            sample_spacing_env=sample_spacing_env,
            top_k=top_k_peaks,
        )
        freqs_a, amps_ablated, peaks_ablated = _rfft_summary(
            ablated_proj[:, index],
            sample_spacing_env=sample_spacing_env,
            top_k=top_k_peaks,
        )
        if aggregate_intact is None:
            aggregate_intact_freqs = freqs_i
            aggregate_ablated_freqs = freqs_a
            aggregate_intact = amps_intact ** 2
            aggregate_ablated = amps_ablated ** 2
        else:
            aggregate_intact = aggregate_intact + amps_intact ** 2
            aggregate_ablated = aggregate_ablated + amps_ablated ** 2
        _save_overlay_spectra(
            intact_freqs=freqs_i,
            intact_amps=amps_intact,
            ablated_freqs=freqs_a,
            ablated_amps=amps_ablated,
            title=f"PCA Frequency Spectrum: PC{index + 1}",
            output_path=figures_dir / f"pc{index + 1}_spectrum.png",
        )
        component_summaries.append(
            {
                "component": index + 1,
                "explained_variance_ratio": float(explained[index]),
                "intact_peaks": peaks_intact,
                "deformation_off_peaks": peaks_ablated,
            }
        )

    if aggregate_intact_freqs is None:
        aggregate_intact_freqs = np.zeros((0,), dtype=np.float32)
        aggregate_ablated_freqs = np.zeros((0,), dtype=np.float32)
        aggregate_intact = np.zeros((0,), dtype=np.float32)
        aggregate_ablated = np.zeros((0,), dtype=np.float32)

    _save_overlay_spectra(
        intact_freqs=aggregate_intact_freqs,
        intact_amps=np.sqrt(aggregate_intact),
        ablated_freqs=aggregate_ablated_freqs,
        ablated_amps=np.sqrt(aggregate_ablated),
        title="Aggregate PCA Power Spectrum",
        output_path=figures_dir / "aggregate_spectrum.png",
    )

    np.savez_compressed(
        output_dir / "projections_and_spectra.npz",
        mean=mean,
        components=components,
        explained_variance_ratio=explained,
        intact_projection=intact_proj,
        deformation_off_projection=ablated_proj,
        aggregate_intact_freqs=aggregate_intact_freqs,
        aggregate_deformation_off_freqs=aggregate_ablated_freqs,
        aggregate_intact_power=aggregate_intact,
        aggregate_deformation_off_power=aggregate_ablated,
    )

    summary = {
        "bundle_dir": str(bundle_dir),
        "modality": modality,
        "distance_pair": str(intact_manifest["distance_pair"]),
        "phase_unique_sample_spacing_env_steps": sample_spacing_env,
        "intact_trial_id": intact_trial_id,
        "deformation_off_trial_id": ablated_trial_id,
        "intact_phase_unique_samples": int(intact_proj.shape[0]),
        "deformation_off_phase_unique_samples": int(ablated_proj.shape[0]),
        "n_components": int(n_components),
        "explained_variance_ratio": [float(x) for x in explained],
        "components": component_summaries,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare intact vs deformation-off PCA frequency content for the copied s4 PCA bundle.")
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT,
        help="Path to the copied pca_bundle directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for spectra and summaries.",
    )
    parser.add_argument(
        "--modality",
        choices=["signaling", "ctrnn", "actions"],
        default="signaling",
        help="Which trace tensor to analyze.",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=3,
        help="Number of PCA components to retain.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    bundle_dir = args.bundle_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (bundle_dir.parent.parent / "analysis_runs" / f"pca_frequency_{args.modality}").resolve()
    )
    summary = run_frequency_analysis(
        bundle_dir=bundle_dir,
        output_dir=output_dir,
        modality=str(args.modality),
        n_components=int(args.components),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
