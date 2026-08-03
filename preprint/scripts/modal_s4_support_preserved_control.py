"""Modal entrypoint for the S4 support-preserved control analysis."""

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
    timeout=4 * 3600,
    volumes=VOLUMES,
)
def analyze_s4_support_preserved_remote(
    *,
    package_root: str = "/outputs/preprint_s4_support_preserved_control_20260620",
    fixed_width_root: str = "/outputs/preprint_s4_fixed_width_distance_sweep_20260618",
) -> dict[str, str]:
    """Run support-preserved analysis inside Modal next to volume-resident traces."""
    import os
    import sys

    os.environ["MPLCONFIGDIR"] = "/tmp/mpl-s4-support-control"
    os.environ["S4_SUPPORT_CONTROL_PACKAGE_ROOT"] = str(package_root)
    os.environ["S4_SUPPORT_CONTROL_DATA_ROOT"] = str(package_root)
    os.environ["S4_FIXED_WIDTH_RELEASE_MANIFEST"] = str(
        Path(fixed_width_root) / "tables" / "release_event_manifest.csv"
    )
    sys.path.insert(0, "/root/preprint_scripts")

    import s4_support_preserved_control as support

    try:
        outputs = support.analyze(init_package=False)
        return {key: str(value) for key, value in outputs.items()}
    finally:
        commit_volumes()


@app.local_entrypoint()
def main(
    package_root: str = "/outputs/preprint_s4_support_preserved_control_20260620",
    fixed_width_root: str = "/outputs/preprint_s4_fixed_width_distance_sweep_20260618",
    forced_command: bool = False,
    config_path: str = "/outputs/rerun_local_runs/paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed137_e4fcc2fb/config_snapshot.yaml",
):
    if forced_command:
        result = analyze_s4_forced_command_support_remote.remote(
            package_root=package_root,
            fixed_width_root=fixed_width_root,
            config_path=config_path,
        )
    else:
        result = analyze_s4_support_preserved_remote.remote(
            package_root=package_root,
            fixed_width_root=fixed_width_root,
        )
    print(result)


@app.function(
    image=analysis_image,
    gpu="A100-40GB",
    timeout=8 * 3600,
    volumes=VOLUMES,
)
def analyze_s4_forced_command_support_remote(
    *,
    package_root: str = "/outputs/preprint_s4_support_preserved_control_20260620",
    fixed_width_root: str = "/outputs/preprint_s4_fixed_width_distance_sweep_20260618",
    config_path: str = "/outputs/rerun_local_runs/paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed137_e4fcc2fb/config_snapshot.yaml",
) -> dict[str, str]:
    """Replay fixed-width command streams on a support-preserved A100 terrain."""
    import os
    import sys
    from pathlib import Path

    os.environ["MPLCONFIGDIR"] = "/tmp/mpl-s4-forced-support-control"
    os.environ["S4_SUPPORT_CONTROL_PACKAGE_ROOT"] = str(package_root)
    os.environ["S4_SUPPORT_CONTROL_DATA_ROOT"] = str(package_root)
    os.environ["S4_FIXED_WIDTH_RELEASE_MANIFEST"] = str(
        Path(fixed_width_root) / "tables" / "release_event_manifest.csv"
    )
    sys.path.insert(0, "/root/preprint_scripts")

    import s4_support_preserved_control as support

    try:
        outputs = support.analyze_forced_command_replay(
            config_path=str(config_path),
            init_package=False,
        )
        return {key: str(value) for key, value in outputs.items()}
    finally:
        commit_volumes()
