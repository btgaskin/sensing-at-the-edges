"""Modal entrypoint for S4 fixed-width distance-sweep analysis.

This runs the decoder/propagation analysis next to the volume-resident traces
and writes only derived paper-facing outputs back to the shared outputs volume.
"""

from __future__ import annotations

import sys
from pathlib import Path

LOCAL_SENSING_MODAL_DIR = Path(__file__).resolve().parent.parent.parent / "sensing_edges" / "scripts"
SENSING_MODAL_DIR = Path("/root")
if not (SENSING_MODAL_DIR / "_modal_common.py").exists():
    SENSING_MODAL_DIR = LOCAL_SENSING_MODAL_DIR
if str(SENSING_MODAL_DIR) not in sys.path:
    sys.path.insert(0, str(SENSING_MODAL_DIR))

from _modal_common import VOLUMES, app, commit_volumes, se_image


PREPRINT_SCRIPTS_DIR = Path(__file__).resolve().parent
analysis_image = se_image.add_local_dir(str(PREPRINT_SCRIPTS_DIR), "/root/preprint_scripts")


@app.function(
    image=analysis_image,
    timeout=8 * 3600,
    volumes=VOLUMES,
)
def analyze_s4_fixed_width_remote(
    *,
    package_root: str = "/outputs/preprint_s4_fixed_width_distance_sweep_20260618",
    random_seed: int = 137,
) -> dict[str, str]:
    """Run S4 fixed-width analysis inside Modal against /outputs traces."""
    import os
    import sys

    os.environ["MPLCONFIGDIR"] = "/tmp/mpl-s4-fixed"
    os.environ["S4_FIXED_WIDTH_PACKAGE_ROOT"] = str(package_root)
    os.environ["S4_FIXED_WIDTH_DATA_ROOT"] = str(package_root)
    sys.path.insert(0, "/root/preprint_scripts")

    import s4_fixed_width_distance_sweep as sweep

    try:
        outputs = sweep.analyze(random_seed=int(random_seed), init_package=False)
        return {key: str(value) for key, value in outputs.items()}
    finally:
        commit_volumes()


@app.function(
    image=analysis_image,
    timeout=4 * 3600,
    volumes=VOLUMES,
)
def analyze_s4_phase_reset_remote(
    *,
    package_root: str = "/outputs/preprint_s4_fixed_width_distance_sweep_20260618",
    random_seed: int = 137,
) -> dict[str, str]:
    """Run only the S4 phase-reset analysis inside Modal against /outputs traces."""
    import os
    import sys

    os.environ["MPLCONFIGDIR"] = "/tmp/mpl-s4-fixed"
    os.environ["S4_FIXED_WIDTH_PACKAGE_ROOT"] = str(package_root)
    os.environ["S4_FIXED_WIDTH_DATA_ROOT"] = str(package_root)
    sys.path.insert(0, "/root/preprint_scripts")

    import s4_fixed_width_distance_sweep as sweep

    try:
        outputs = sweep.analyze_phase_reset_only(random_seed=int(random_seed), init_package=False)
        return {key: str(value) for key, value in outputs.items()}
    finally:
        commit_volumes()


@app.function(
    image=analysis_image,
    timeout=6 * 3600,
    volumes=VOLUMES,
)
def analyze_s4_phase_propagation_remote(
    *,
    package_root: str = "/outputs/preprint_s4_fixed_width_distance_sweep_20260618",
    random_seed: int = 137,
) -> dict[str, str]:
    """Run only the S4 local phase-propagation analysis inside Modal against /outputs traces."""
    import os
    import sys

    os.environ["MPLCONFIGDIR"] = "/tmp/mpl-s4-fixed"
    os.environ["S4_FIXED_WIDTH_PACKAGE_ROOT"] = str(package_root)
    os.environ["S4_FIXED_WIDTH_DATA_ROOT"] = str(package_root)
    sys.path.insert(0, "/root/preprint_scripts")

    import s4_fixed_width_distance_sweep as sweep

    try:
        outputs = sweep.analyze_phase_propagation_only(random_seed=int(random_seed), init_package=False)
        return {key: str(value) for key, value in outputs.items()}
    finally:
        commit_volumes()


@app.function(
    image=analysis_image,
    timeout=2 * 3600,
    volumes=VOLUMES,
)
def audit_s4_fixed_width_remote(
    *,
    package_root: str = "/outputs/preprint_s4_fixed_width_distance_sweep_20260618",
) -> dict[str, str]:
    """Audit fixed-width raw traces and write derived audit outputs."""
    import os
    import sys

    os.environ["MPLCONFIGDIR"] = "/tmp/mpl-s4-fixed"
    os.environ["S4_FIXED_WIDTH_PACKAGE_ROOT"] = str(package_root)
    os.environ["S4_FIXED_WIDTH_DATA_ROOT"] = str(package_root)
    sys.path.insert(0, "/root/preprint_scripts")

    import s4_fixed_width_distance_sweep as sweep

    try:
        outputs = sweep.audit_remote_data(init_package=False)
        return {key: str(value) for key, value in outputs.items()}
    finally:
        commit_volumes()


@app.function(
    image=analysis_image,
    timeout=8 * 3600,
    volumes=VOLUMES,
)
def analyze_s4_graph_propagation_remote(
    *,
    package_root: str = "/outputs/preprint_s4_fixed_width_distance_sweep_20260618",
    random_seed: int = 137,
) -> dict[str, str]:
    """Run the graph-distance mechanical propagation analysis on Modal."""
    import os
    import sys

    os.environ["MPLCONFIGDIR"] = "/tmp/mpl-s4-fixed"
    os.environ["S4_FIXED_WIDTH_PACKAGE_ROOT"] = str(package_root)
    os.environ["S4_FIXED_WIDTH_DATA_ROOT"] = str(package_root)
    sys.path.insert(0, "/root/preprint_scripts")

    import s4_fixed_width_distance_sweep as sweep

    try:
        outputs = sweep.analyze_graph_propagation_only(random_seed=int(random_seed), init_package=False)
        return {key: str(value) for key, value in outputs.items()}
    finally:
        commit_volumes()


@app.function(
    image=analysis_image,
    timeout=8 * 3600,
    volumes=VOLUMES,
)
def analyze_s4_pca_regime_remote(
    *,
    package_root: str = "/outputs/preprint_s4_fixed_width_distance_sweep_20260618",
    random_seed: int = 137,
) -> dict[str, str]:
    """Run the PCA regime-entry analysis on Modal."""
    import os
    import sys

    os.environ["MPLCONFIGDIR"] = "/tmp/mpl-s4-fixed"
    os.environ["S4_FIXED_WIDTH_PACKAGE_ROOT"] = str(package_root)
    os.environ["S4_FIXED_WIDTH_DATA_ROOT"] = str(package_root)
    sys.path.insert(0, "/root/preprint_scripts")

    import s4_fixed_width_distance_sweep as sweep

    try:
        outputs = sweep.analyze_pca_regime_entry_only(random_seed=int(random_seed), init_package=False)
        return {key: str(value) for key, value in outputs.items()}
    finally:
        commit_volumes()


@app.function(
    image=analysis_image,
    timeout=12 * 3600,
    volumes=VOLUMES,
)
def analyze_s4_remaining_mechanism_remote(
    *,
    package_root: str = "/outputs/preprint_s4_fixed_width_distance_sweep_20260618",
    random_seed: int = 137,
) -> dict[str, str]:
    """Run audit, graph propagation, and PCA regime-entry analyses on Modal."""
    import os
    import sys

    os.environ["MPLCONFIGDIR"] = "/tmp/mpl-s4-fixed"
    os.environ["S4_FIXED_WIDTH_PACKAGE_ROOT"] = str(package_root)
    os.environ["S4_FIXED_WIDTH_DATA_ROOT"] = str(package_root)
    sys.path.insert(0, "/root/preprint_scripts")

    import s4_fixed_width_distance_sweep as sweep

    try:
        outputs = sweep.analyze_remaining_mechanism(random_seed=int(random_seed), init_package=False)
        return {key: str(value) for key, value in outputs.items()}
    finally:
        commit_volumes()


@app.local_entrypoint()
def main(
    package_root: str = "/outputs/preprint_s4_fixed_width_distance_sweep_20260618",
    random_seed: int = 137,
    phase_only: bool = False,
    phase_propagation: bool = False,
    audit_only: bool = False,
    graph_propagation: bool = False,
    pca_regime_entry: bool = False,
    all_remaining: bool = False,
):
    if all_remaining:
        result = analyze_s4_remaining_mechanism_remote.remote(
            package_root=package_root,
            random_seed=random_seed,
        )
    elif audit_only:
        result = audit_s4_fixed_width_remote.remote(
            package_root=package_root,
        )
    elif graph_propagation:
        result = analyze_s4_graph_propagation_remote.remote(
            package_root=package_root,
            random_seed=random_seed,
        )
    elif pca_regime_entry:
        result = analyze_s4_pca_regime_remote.remote(
            package_root=package_root,
            random_seed=random_seed,
        )
    elif phase_propagation:
        result = analyze_s4_phase_propagation_remote.remote(
            package_root=package_root,
            random_seed=random_seed,
        )
    elif phase_only:
        result = analyze_s4_phase_reset_remote.remote(
            package_root=package_root,
            random_seed=random_seed,
        )
    else:
        result = analyze_s4_fixed_width_remote.remote(
            package_root=package_root,
            random_seed=random_seed,
        )
    print(result)
