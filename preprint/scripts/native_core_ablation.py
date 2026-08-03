from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import shutil
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH_DIR = REPO_ROOT / "sensing_edges" / "tmp" / "preprint_native_core_ablation_v1"
DEFAULT_ANALYSIS_ROOT = "/outputs/preprint_native_core_ablation_v1"
DEFAULT_BODY_TRANSFER_ANALYSIS_ROOT = "/outputs/preprint_native_core_body_transfer_v1"
DEFAULT_STATE_SPACE_ANALYSIS_ROOT = "/outputs/preprint_native_state_space_v1"
DEFAULT_STATE_SPACE_DEFORMATION_OFF_ANALYSIS_ROOT = "/outputs/preprint_native_state_space_deformation_off_v1"
DEFAULT_STATE_SPACE_EDGES_DEFORMATION_OFF_ANALYSIS_ROOT = "/outputs/preprint_edge_state_space_deformation_off_v1"
DEFAULT_LOW_COST_NATIVE_ANALYSIS_ROOT = "/outputs/preprint_perfect_native_l4_v1"
DEFAULT_LOCAL_PACKAGE_ROOT = REPO_ROOT / "paper_data" / "preprint" / "native_core_ablation_v1"

NATIVE_PAIRS = ((2, 1), (4, 2), (6, 3))
LONG_DISTANCE_PAIRS = ((8, 4), (10, 5))
DOUBLE_CLIFF_WIDTHS = (2, 4, 6)
SIDES = ("left", "right")
SURFACE_PRESETS = {
    "native_core": ("native_core",),
    "extension": ("long_distance_generalization", "double_cliff"),
    "all": ("native_core", "long_distance_generalization", "double_cliff"),
}
SURFACE_LABELS = {
    "native_core": "native_core",
    "long_distance_generalization": "extended",
    "double_cliff": "symmetric",
}
SIGNALING_CONDITIONS = (
    "intact",
    "sobel_x_off",
    "sobel_y_off",
    "laplacian_off",
    "identity_comm_off",
    "all_comm_off",
)
DEFORMATION_CONDITIONS = (
    "intact",
    "absolute_only",
    "compliance_only",
    "deformation_off",
)
BODY_TRANSFER_BODIES = (
    "rect_7x2",
    "rect_6x2",
    "rect_8x2",
    "rect_7x3",
)
STATE_SPACE_SPECIMEN_ID = "bin_03_rank_00_cell_6-0-3_flat_0423"
STATE_SPACE_RUN_ID = "paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed137_e4fcc2fb"

STEP_BUDGETS = {
    "native_core": 4000,
    "long_distance_8_4": 5000,
    "long_distance_10_5": 7500,
    "double_cliff": 7500,
}


SPECIMENS: tuple[dict[str, str], ...] = (
    {
        "group_id": "seed42_t2000",
        "batch_label": "preprint_native_core_ablation_v1_seed42_t2000",
        "run_id": "iter_run_cliff_rect_7x2_sep_cmaes_g100_p150_15sub_t2000_e2000_ref003_seed42_d4611d8e",
        "specimen_id": "bin_04_rank_00_cell_6-0-4_flat_0424",
        "remote_run_dir": "/outputs/rerun_local_runs/iter_run_cliff_rect_7x2_sep_cmaes_g100_p150_15sub_t2000_e2000_ref003_seed42_d4611d8e",
        "remote_config_path": "/outputs/rerun_local_runs/iter_run_cliff_rect_7x2_sep_cmaes_g100_p150_15sub_t2000_e2000_ref003_seed42_d4611d8e/config_snapshot.yaml",
        "specimens_output_root": "/outputs/rerun_local_specimens",
        "local_config_path": "run_specimens/iter_run_cliff_rect_7x2_sep_cmaes_g100_p150_15sub_t2000_e2000_ref003_seed42_d4611d8e/specimen_bin_04_rank_00_cell_6-0-4_flat_0424/config_snapshot.yaml",
    },
    {
        "group_id": "seed42_t2000",
        "batch_label": "preprint_native_core_ablation_v1_seed42_t2000",
        "run_id": "iter_run_cliff_rect_7x2_sep_cmaes_g100_p150_15sub_t2000_e2000_ref003_seed42_d4611d8e",
        "specimen_id": "bin_06_rank_01_cell_5-0-6_flat_0356",
        "remote_run_dir": "/outputs/rerun_local_runs/iter_run_cliff_rect_7x2_sep_cmaes_g100_p150_15sub_t2000_e2000_ref003_seed42_d4611d8e",
        "remote_config_path": "/outputs/rerun_local_runs/iter_run_cliff_rect_7x2_sep_cmaes_g100_p150_15sub_t2000_e2000_ref003_seed42_d4611d8e/config_snapshot.yaml",
        "specimens_output_root": "/outputs/rerun_local_specimens",
        "local_config_path": "run_specimens/iter_run_cliff_rect_7x2_sep_cmaes_g100_p150_15sub_t2000_e2000_ref003_seed42_d4611d8e/specimen_bin_06_rank_01_cell_5-0-6_flat_0356/config_snapshot.yaml",
    },
    {
        "group_id": "seed137_t4000",
        "batch_label": "preprint_native_core_ablation_v1_seed137_t4000",
        "run_id": "paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed137_e4fcc2fb",
        "specimen_id": "bin_03_rank_00_cell_6-0-3_flat_0423",
        "remote_run_dir": "/outputs/rerun_local_runs/paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed137_e4fcc2fb",
        "remote_config_path": "/outputs/rerun_local_runs/paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed137_e4fcc2fb/config_snapshot.yaml",
        "specimens_output_root": "/outputs/rerun_local_specimens",
        "local_config_path": "run_specimens/paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed137_e4fcc2fb/stages/gen_0099/specimen_bin_03_rank_00_cell_6-0-3_flat_0423/config_snapshot.yaml",
    },
    {
        "group_id": "seed137_t4000",
        "batch_label": "preprint_native_core_ablation_v1_seed137_t4000",
        "run_id": "paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed137_e4fcc2fb",
        "specimen_id": "bin_04_rank_01_cell_6-0-4_flat_0424",
        "remote_run_dir": "/outputs/rerun_local_runs/paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed137_e4fcc2fb",
        "remote_config_path": "/outputs/rerun_local_runs/paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed137_e4fcc2fb/config_snapshot.yaml",
        "specimens_output_root": "/outputs/rerun_local_specimens",
        "local_config_path": "run_specimens/paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed137_e4fcc2fb/stages/gen_0099/specimen_bin_04_rank_01_cell_6-0-4_flat_0424/config_snapshot.yaml",
    },
    {
        "group_id": "seed137_t4000",
        "batch_label": "preprint_native_core_ablation_v1_seed137_t4000",
        "run_id": "paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed137_e4fcc2fb",
        "specimen_id": "bin_05_rank_02_cell_5-0-5_flat_0355",
        "remote_run_dir": "/outputs/rerun_local_runs/paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed137_e4fcc2fb",
        "remote_config_path": "/outputs/rerun_local_runs/paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed137_e4fcc2fb/config_snapshot.yaml",
        "specimens_output_root": "/outputs/rerun_local_specimens",
        "local_config_path": "run_specimens/paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed137_e4fcc2fb/stages/gen_0099/specimen_bin_05_rank_02_cell_5-0-5_flat_0355/config_snapshot.yaml",
    },
    {
        "group_id": "seed1564_t4000",
        "batch_label": "preprint_native_core_ablation_v1_seed1564_t4000",
        "run_id": "paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed1564_a792f078",
        "specimen_id": "bin_04_rank_00_cell_5-0-4_flat_0354",
        "remote_run_dir": "/outputs/rerun_local_runs/paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed1564_a792f078",
        "remote_config_path": "/outputs/rerun_local_runs/paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed1564_a792f078/config_snapshot.yaml",
        "specimens_output_root": "/outputs/rerun_local_specimens",
        "local_config_path": "run_specimens/paper_run_cliff_rect_7x2_sep_cmaes_g150_p300_15sub_t4000_e4000_ref003_seed1564_a792f078/stages/gen_0099/specimen_bin_04_rank_00_cell_5-0-4_flat_0354/config_snapshot.yaml",
    },
    {
        "group_id": "seed1632_t3000",
        "batch_label": "preprint_native_core_ablation_v1_seed1632_t3000",
        "run_id": "paper_run_cliff_rect_7x2_sep_cmaes_g150_p150_15sub_t3000_e3000_ref003_seed1632_27f7e19b",
        "specimen_id": "bin_03_rank_00_cell_5-0-3_flat_0353",
        "remote_run_dir": "/outputs/rerun_local_runs/paper_run_cliff_rect_7x2_sep_cmaes_g150_p150_15sub_t3000_e3000_ref003_seed1632_27f7e19b",
        "remote_config_path": "/outputs/rerun_local_runs/paper_run_cliff_rect_7x2_sep_cmaes_g150_p150_15sub_t3000_e3000_ref003_seed1632_27f7e19b/config_snapshot.yaml",
        "specimens_output_root": "/outputs/rerun_local_specimens",
        "local_config_path": "run_specimens/paper_run_cliff_rect_7x2_sep_cmaes_g150_p150_15sub_t3000_e3000_ref003_seed1632_27f7e19b/stages/gen_0099/specimen_bin_03_rank_00_cell_5-0-3_flat_0353/config_snapshot.yaml",
    },
    {
        "group_id": "seed42_t3000",
        "batch_label": "preprint_native_core_ablation_v1_seed42_t3000",
        "run_id": "iter_run_cliff_rect_7x2_sep_cmaes_g100_p150_15sub_t3000_e3000_ref003_seed42_be529c33",
        "specimen_id": "bin_06_rank_00_cell_4-0-6_flat_0286",
        "remote_run_dir": "/outputs/iter_run_cliff_rect_7x2_sep_cmaes_g100_p150_15sub_t3000_e3000_ref003_seed42_be529c33",
        "remote_config_path": "/outputs/iter_run_cliff_rect_7x2_sep_cmaes_g100_p150_15sub_t3000_e3000_ref003_seed42_be529c33/resolved_config.yaml",
        "specimens_output_root": "/outputs/specimens",
        "local_config_path": "run_specimens/iter_run_cliff_rect_7x2_sep_cmaes_g100_p150_15sub_t3000_e3000_ref003_seed42_be529c33/specimen_bin_06_rank_00_cell_4-0-6_flat_0286/config_snapshot.yaml",
    },
)


def group_ids() -> list[str]:
    return sorted({item["group_id"] for item in SPECIMENS})


def specimens_for_group(group_id: str) -> list[dict[str, str]]:
    return [dict(item) for item in SPECIMENS if item["group_id"] == group_id]


def surface_ids(surface_preset: str) -> tuple[str, ...]:
    try:
        return SURFACE_PRESETS[str(surface_preset)]
    except KeyError as exc:
        raise ValueError(f"Unknown surface preset: {surface_preset}") from exc


def _omega_list(values: tuple[str, ...]) -> str:
    return "[" + ",".join(values) + "]"


def _shell_join(argv: list[str]) -> str:
    return shlex.join(argv)


def expected_render_names_for_surface(surface_id: str) -> list[str]:
    if surface_id == "native_core":
        return [f"pair_{goal}_{cliff}_{side}.gif" for goal, cliff in NATIVE_PAIRS for side in SIDES]
    if surface_id == "long_distance_generalization":
        return [f"pair_{goal}_{cliff}_{side}.gif" for goal, cliff in LONG_DISTANCE_PAIRS for side in SIDES]
    if surface_id == "double_cliff":
        return [f"width_{width}.gif" for width in DOUBLE_CLIFF_WIDTHS]
    raise ValueError(f"Unsupported surface_id={surface_id!r}")


def expected_rows_per_specimen(*, surfaces: tuple[str, ...], assay: str) -> int:
    return len(_assay_conditions(assay)) * sum(
        len(expected_render_names_for_surface(surface_id))
        for surface_id in surfaces
    )


def expected_body_transfer_rows_per_specimen(*, surfaces: tuple[str, ...]) -> int:
    return len(BODY_TRANSFER_BODIES) * sum(
        len(expected_render_names_for_surface(surface_id))
        for surface_id in surfaces
    )


def expected_totals(*, surfaces: tuple[str, ...]) -> dict[str, Any]:
    signaling_rows = expected_rows_per_specimen(surfaces=surfaces, assay="signaling_ablation")
    deformation_rows = expected_rows_per_specimen(surfaces=surfaces, assay="deformation_ablation")
    rows_per_specimen = signaling_rows + deformation_rows
    return {
        "specimen_count": len(SPECIMENS),
        "signaling_rows_per_specimen": signaling_rows,
        "deformation_rows_per_specimen": deformation_rows,
        "rows_per_specimen": rows_per_specimen,
        "renders_per_specimen": rows_per_specimen,
        "total_rows": rows_per_specimen * len(SPECIMENS),
        "total_renders": rows_per_specimen * len(SPECIMENS),
    }


def expected_totals_with_body_transfer(*, surfaces: tuple[str, ...], include_body_transfer: bool) -> dict[str, Any]:
    totals = expected_totals(surfaces=surfaces)
    if include_body_transfer:
        body_rows = expected_body_transfer_rows_per_specimen(surfaces=("native_core",))
        totals["body_transfer_rows_per_specimen"] = body_rows
        totals["rows_per_specimen"] += body_rows
        totals["renders_per_specimen"] += body_rows
        totals["total_rows"] += body_rows * len(SPECIMENS)
        totals["total_renders"] += body_rows * len(SPECIMENS)
    return totals


def override_tokens(
    *,
    specimens_output_root: str,
    analysis_root: str = DEFAULT_ANALYSIS_ROOT,
    surfaces: tuple[str, ...] = SURFACE_PRESETS["native_core"],
) -> list[str]:
    return [
        f"eval.post_run.analysis_root={analysis_root}",
        f"eval.post_run.signaling_surfaces={_omega_list(surfaces)}",
        f"eval.post_run.deformation_surfaces={_omega_list(surfaces)}",
        "eval.post_run.signaling_render_conditions=[intact,sobel_x_off,sobel_y_off,laplacian_off,identity_comm_off,all_comm_off]",
        "eval.post_run.deformation_conditions=[intact,absolute_only,compliance_only,deformation_off]",
        "eval.post_run.deformation_render_conditions=[intact,absolute_only,compliance_only,deformation_off]",
        f"eval.post_run.native_max_steps={STEP_BUDGETS['native_core']}",
        f"eval.post_run.max_steps={STEP_BUDGETS['long_distance_10_5']}",
        "eval.post_run.long_distance_pair_max_steps=[{\"goal_distance\":8,\"cliff_distance\":4,\"max_steps\":5000},{\"goal_distance\":10,\"cliff_distance\":5,\"max_steps\":7500}]",
        f"eval.post_run.double_cliff_max_steps={STEP_BUDGETS['double_cliff']}",
        "eval.post_run.fast_ablation_eval=true",
        f"specimens.output_root={specimens_output_root}",
    ]


def override_csv(
    *,
    specimens_output_root: str,
    analysis_root: str = DEFAULT_ANALYSIS_ROOT,
    surfaces: tuple[str, ...] = SURFACE_PRESETS["native_core"],
) -> str:
    return ",".join(
        override_tokens(
            specimens_output_root=specimens_output_root,
            analysis_root=analysis_root,
            surfaces=surfaces,
        )
    )


def body_transfer_override_tokens(
    *,
    specimens_output_root: str,
    analysis_root: str = DEFAULT_BODY_TRANSFER_ANALYSIS_ROOT,
) -> list[str]:
    return [
        f"eval.post_run.analysis_root={analysis_root}",
        "eval.post_run.signaling_surfaces=[native_core]",
        f"eval.post_run.native_max_steps={STEP_BUDGETS['native_core']}",
        "eval.post_run.body_transfer_render_bodies=[reference,rect_6x2,rect_8x2,rect_7x3]",
        "eval.post_run.fast_ablation_eval=true",
        f"specimens.output_root={specimens_output_root}",
    ]


def body_transfer_override_csv(
    *,
    specimens_output_root: str,
    analysis_root: str = DEFAULT_BODY_TRANSFER_ANALYSIS_ROOT,
) -> str:
    return ",".join(
        body_transfer_override_tokens(
            specimens_output_root=specimens_output_root,
            analysis_root=analysis_root,
        )
    )


def state_space_override_tokens(
    *,
    specimens_output_root: str,
    analysis_root: str = DEFAULT_STATE_SPACE_ANALYSIS_ROOT,
    deformation_condition: str = "intact",
    signaling_condition: str = "intact",
    state_space_surfaces: tuple[str, ...] = SURFACE_PRESETS["native_core"],
) -> list[str]:
    return [
        f"eval.post_run.analysis_root={analysis_root}",
        f"eval.post_run.native_max_steps={STEP_BUDGETS['native_core']}",
        f"eval.post_run.max_steps={STEP_BUDGETS['long_distance_10_5']}",
        "eval.post_run.long_distance_pair_max_steps=[{\"goal_distance\":8,\"cliff_distance\":4,\"max_steps\":5000},{\"goal_distance\":10,\"cliff_distance\":5,\"max_steps\":7500}]",
        f"eval.post_run.double_cliff_max_steps={STEP_BUDGETS['double_cliff']}",
        f"eval.post_run.state_space_surfaces={_omega_list(state_space_surfaces)}",
        f"eval.post_run.state_space_deformation_condition={deformation_condition}",
        f"eval.post_run.state_space_signaling_condition={signaling_condition}",
        "eval.post_run.fast_ablation_eval=false",
        f"specimens.output_root={specimens_output_root}",
    ]


def state_space_override_csv(
    *,
    specimens_output_root: str,
    analysis_root: str = DEFAULT_STATE_SPACE_ANALYSIS_ROOT,
    deformation_condition: str = "intact",
    signaling_condition: str = "intact",
    state_space_surfaces: tuple[str, ...] = SURFACE_PRESETS["native_core"],
) -> str:
    return ",".join(
        state_space_override_tokens(
            specimens_output_root=specimens_output_root,
            analysis_root=analysis_root,
            deformation_condition=deformation_condition,
            signaling_condition=signaling_condition,
            state_space_surfaces=state_space_surfaces,
        )
    )


def batch_jobs_for_group(group_id: str) -> list[dict[str, str]]:
    return [
        {
            "run_dir": item["remote_run_dir"],
            "specimen_id": item["specimen_id"],
            "config_path": item["remote_config_path"],
        }
        for item in specimens_for_group(group_id)
    ]


def launch_argv(
    *,
    batch_path: Path,
    batch_label: str,
    specimens_output_root: str,
    surfaces: tuple[str, ...] = SURFACE_PRESETS["native_core"],
) -> list[str]:
    return [
        "uv",
        "run",
        "--python",
        ".venv/bin/python",
        "modal",
        "run",
        "--detach",
        "sensing_edges/scripts/modal_post_run.py",
        "--run-dir",
        "/outputs",
        "--batch-jobs-path",
        str(batch_path),
        "--assays",
        "signaling_ablation,deformation_ablation",
        "--override",
        override_csv(specimens_output_root=specimens_output_root, surfaces=surfaces),
        "--batch-label",
        batch_label,
        "--detach-launch",
    ]


def body_transfer_launch_argv(
    *,
    item: dict[str, str],
    analysis_root: str = DEFAULT_BODY_TRANSFER_ANALYSIS_ROOT,
) -> list[str]:
    return [
        "uv",
        "run",
        "--python",
        ".venv/bin/python",
        "modal",
        "run",
        "--detach",
        "sensing_edges/scripts/modal_post_run.py",
        "--run-dir",
        item["remote_run_dir"],
        "--specimen-id",
        item["specimen_id"],
        "--config-path",
        item["remote_config_path"],
        "--assays",
        "native_core,body_transfer",
        "--override",
        body_transfer_override_csv(
            specimens_output_root=item["specimens_output_root"],
            analysis_root=analysis_root,
        ),
        "--detach-launch",
    ]


def state_space_specimen() -> dict[str, str]:
    for item in SPECIMENS:
        if item["run_id"] == STATE_SPACE_RUN_ID and item["specimen_id"] == STATE_SPACE_SPECIMEN_ID:
            return dict(item)
    raise ValueError(f"Configured state-space specimen not found: {STATE_SPACE_RUN_ID}/{STATE_SPACE_SPECIMEN_ID}")


def state_space_launch_argv(
    *,
    analysis_root: str = DEFAULT_STATE_SPACE_ANALYSIS_ROOT,
    assay: str = "state_space_trace",
    deformation_condition: str = "intact",
    signaling_condition: str = "intact",
    state_space_surfaces: tuple[str, ...] = SURFACE_PRESETS["native_core"],
) -> list[str]:
    item = state_space_specimen()
    return [
        "uv",
        "run",
        "--python",
        ".venv/bin/python",
        "modal",
        "run",
        "--detach",
        "sensing_edges/scripts/modal_post_run.py",
        "--run-dir",
        item["remote_run_dir"],
        "--specimen-id",
        item["specimen_id"],
        "--config-path",
        item["remote_config_path"],
        "--assays",
        f"native_core,{assay}",
        "--override",
        state_space_override_csv(
            specimens_output_root=item["specimens_output_root"],
            analysis_root=analysis_root,
            deformation_condition=deformation_condition,
            signaling_condition=signaling_condition,
            state_space_surfaces=state_space_surfaces,
        ),
        "--detach-launch",
    ]


def state_space_deformation_off_launch_argv(
    *,
    analysis_root: str = DEFAULT_STATE_SPACE_DEFORMATION_OFF_ANALYSIS_ROOT,
) -> list[str]:
    return state_space_launch_argv(
        analysis_root=analysis_root,
        assay="state_space_trace_deformation_off",
        deformation_condition="deformation_off",
        signaling_condition="intact",
    )


def state_space_edges_deformation_off_launch_argv(
    *,
    analysis_root: str = DEFAULT_STATE_SPACE_EDGES_DEFORMATION_OFF_ANALYSIS_ROOT,
) -> list[str]:
    return state_space_launch_argv(
        analysis_root=analysis_root,
        assay="state_space_trace_edges_deformation_off",
        deformation_condition="deformation_off",
        signaling_condition="intact",
        state_space_surfaces=SURFACE_PRESETS["all"],
    )


def low_cost_native_launch_argv(
    *,
    analysis_root: str = DEFAULT_LOW_COST_NATIVE_ANALYSIS_ROOT,
    gpu_profile: str = "l4",
) -> list[str]:
    item = state_space_specimen()
    return [
        "uv",
        "run",
        "--python",
        ".venv/bin/python",
        "modal",
        "run",
        "--detach",
        "sensing_edges/scripts/modal_post_run.py",
        "--run-dir",
        item["remote_run_dir"],
        "--specimen-id",
        item["specimen_id"],
        "--config-path",
        item["remote_config_path"],
        "--assays",
        "native_core",
        "--override",
        ",".join(
            [
                f"eval.post_run.analysis_root={analysis_root}",
                f"eval.post_run.native_max_steps={STEP_BUDGETS['native_core']}",
                f"specimens.output_root={item['specimens_output_root']}",
            ]
        ),
        "--gpu-profile",
        str(gpu_profile),
        "--detach-launch",
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def command_write_batches(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    surfaces = surface_ids(args.surface_preset)
    batch_prefix = "" if args.surface_preset == "native_core" else f"{args.surface_preset}_"
    manifest_groups: list[dict[str, Any]] = []
    for group_id in group_ids():
        group_specimens = specimens_for_group(group_id)
        first = group_specimens[0]
        batch_path = output_dir / f"batch_jobs_{batch_prefix}{group_id}.json"
        batch_label = first["batch_label"] if args.surface_preset == "native_core" else f"{first['batch_label']}_{args.surface_preset}"
        write_json(batch_path, batch_jobs_for_group(group_id))
        argv = launch_argv(
            batch_path=batch_path,
            batch_label=batch_label,
            specimens_output_root=first["specimens_output_root"],
            surfaces=surfaces,
        )
        manifest_groups.append(
            {
                "group_id": group_id,
                "batch_label": batch_label,
                "specimen_count": len(group_specimens),
                "specimens_output_root": first["specimens_output_root"],
                "batch_jobs_path": str(batch_path),
                "launch_argv": argv,
                "launch_command": _shell_join(argv),
            }
        )
    write_json(
        output_dir / ("launch_manifest.json" if args.surface_preset == "native_core" else f"launch_manifest_{args.surface_preset}.json"),
        {
            "schema_version": 1,
            "kind": "preprint_native_core_ablation_launch_manifest",
            "analysis_root": DEFAULT_ANALYSIS_ROOT,
            "assays": ["signaling_ablation", "deformation_ablation"],
            "surface_preset": args.surface_preset,
            "surfaces": list(surfaces),
            "surface_labels": {surface_id: SURFACE_LABELS[surface_id] for surface_id in surfaces},
            "step_budgets": STEP_BUDGETS,
            "expected": {
                **expected_totals(surfaces=surfaces),
                "signaling_conditions": list(SIGNALING_CONDITIONS),
                "deformation_conditions": list(DEFORMATION_CONDITIONS),
                "native_pairs": [
                    {"goal_distance": goal, "cliff_distance": cliff}
                    for goal, cliff in NATIVE_PAIRS
                ],
                "long_distance_pairs": [
                    {"goal_distance": goal, "cliff_distance": cliff}
                    for goal, cliff in LONG_DISTANCE_PAIRS
                ],
                "double_cliff_widths": list(DOUBLE_CLIFF_WIDTHS),
            },
            "groups": manifest_groups,
            "specimens": list(SPECIMENS),
        },
    )
    print(output_dir)
    return 0


def command_print_launch_commands(args: argparse.Namespace) -> int:
    batch_dir = Path(args.batch_dir).resolve()
    surfaces = surface_ids(args.surface_preset)
    batch_prefix = "" if args.surface_preset == "native_core" else f"{args.surface_preset}_"
    for group_id in group_ids():
        first = specimens_for_group(group_id)[0]
        batch_path = batch_dir / f"batch_jobs_{batch_prefix}{group_id}.json"
        batch_label = first["batch_label"] if args.surface_preset == "native_core" else f"{first['batch_label']}_{args.surface_preset}"
        print(_shell_join(launch_argv(
            batch_path=batch_path,
            batch_label=batch_label,
            specimens_output_root=first["specimens_output_root"],
            surfaces=surfaces,
        )))
    return 0


def command_write_body_transfer_launch_manifest(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    for item in SPECIMENS:
        argv = body_transfer_launch_argv(item=item, analysis_root=str(args.analysis_root))
        jobs.append(
            {
                "group_id": item["group_id"],
                "run_id": item["run_id"],
                "specimen_id": item["specimen_id"],
                "analysis_root": str(args.analysis_root),
                "assays": ["native_core", "body_transfer"],
                "launch_argv": argv,
                "launch_command": _shell_join(argv),
            }
        )
    write_json(
        output_dir / "launch_manifest_body_transfer.json",
        {
            "schema_version": 1,
            "kind": "preprint_native_body_transfer_launch_manifest",
            "analysis_root": str(args.analysis_root),
            "assays": ["native_core", "body_transfer"],
            "surfaces": ["native_core"],
            "bodies": list(BODY_TRANSFER_BODIES),
            "step_budgets": {"native_core": STEP_BUDGETS["native_core"]},
            "expected": {
                "specimen_count": len(SPECIMENS),
                "rows_per_specimen": expected_body_transfer_rows_per_specimen(surfaces=("native_core",)),
                "total_rows": expected_body_transfer_rows_per_specimen(surfaces=("native_core",)) * len(SPECIMENS),
            },
            "jobs": jobs,
        },
    )
    print(output_dir)
    return 0


def command_print_body_transfer_launch_commands(args: argparse.Namespace) -> int:
    for item in SPECIMENS:
        print(_shell_join(body_transfer_launch_argv(item=item, analysis_root=str(args.analysis_root))))
    return 0


def command_write_state_space_launch_manifest(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    item = state_space_specimen()
    argv = state_space_launch_argv(analysis_root=str(args.analysis_root))
    write_json(
        output_dir / "launch_manifest_state_space.json",
        {
            "schema_version": 1,
            "kind": "preprint_perfect_native_state_space_launch_manifest",
            "analysis_root": str(args.analysis_root),
            "assays": ["native_core", "state_space_trace"],
            "surfaces": ["native_core"],
            "step_budgets": {"native_core": STEP_BUDGETS["native_core"]},
            "state_space_specimen": item,
            "expected": {
                "trial_count": len(NATIVE_PAIRS) * len(SIDES),
                "native_pairs": [
                    {"goal_distance": goal, "cliff_distance": cliff}
                    for goal, cliff in NATIVE_PAIRS
                ],
                "goal_sides": list(SIDES),
            },
            "launch_argv": argv,
            "launch_command": _shell_join(argv),
        },
    )
    print(output_dir)
    return 0


def command_print_state_space_launch_command(args: argparse.Namespace) -> int:
    print(_shell_join(state_space_launch_argv(analysis_root=str(args.analysis_root))))
    return 0


def command_write_state_space_deformation_off_launch_manifest(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    item = state_space_specimen()
    argv = state_space_deformation_off_launch_argv(analysis_root=str(args.analysis_root))
    write_json(
        output_dir / "launch_manifest_state_space_deformation_off.json",
        {
            "schema_version": 1,
            "kind": "preprint_perfect_native_state_space_deformation_off_launch_manifest",
            "analysis_root": str(args.analysis_root),
            "assays": ["native_core", "state_space_trace_deformation_off"],
            "surfaces": ["native_core"],
            "deformation_condition": "deformation_off",
            "signaling_condition": "intact",
            "step_budgets": {"native_core": STEP_BUDGETS["native_core"]},
            "state_space_specimen": item,
            "expected": {
                "trial_count": len(NATIVE_PAIRS) * len(SIDES),
                "native_pairs": [
                    {"goal_distance": goal, "cliff_distance": cliff}
                    for goal, cliff in NATIVE_PAIRS
                ],
                "goal_sides": list(SIDES),
            },
            "launch_argv": argv,
            "launch_command": _shell_join(argv),
        },
    )
    print(output_dir)
    return 0


def command_print_state_space_deformation_off_launch_command(args: argparse.Namespace) -> int:
    print(_shell_join(state_space_deformation_off_launch_argv(analysis_root=str(args.analysis_root))))
    return 0


def command_write_state_space_edges_deformation_off_launch_manifest(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    item = state_space_specimen()
    surfaces = SURFACE_PRESETS["all"]
    argv = state_space_edges_deformation_off_launch_argv(analysis_root=str(args.analysis_root))
    write_json(
        output_dir / "launch_manifest_state_space_edges_deformation_off.json",
        {
            "schema_version": 1,
            "kind": "preprint_perfect_edge_state_space_deformation_off_launch_manifest",
            "analysis_root": str(args.analysis_root),
            "assays": ["native_core", "state_space_trace_edges_deformation_off"],
            "surfaces": list(surfaces),
            "deformation_condition": "deformation_off",
            "signaling_condition": "intact",
            "step_budgets": {
                "native_core": STEP_BUDGETS["native_core"],
                "long_distance_8_4": STEP_BUDGETS["long_distance_8_4"],
                "long_distance_10_5": STEP_BUDGETS["long_distance_10_5"],
                "double_cliff": STEP_BUDGETS["double_cliff"],
            },
            "state_space_specimen": item,
            "expected": {
                "trial_count": len(NATIVE_PAIRS) * len(SIDES)
                + len(LONG_DISTANCE_PAIRS) * len(SIDES)
                + len(DOUBLE_CLIFF_WIDTHS),
                "native_pairs": [
                    {"goal_distance": goal, "cliff_distance": cliff}
                    for goal, cliff in NATIVE_PAIRS
                ],
                "long_distance_pairs": [
                    {"goal_distance": goal, "cliff_distance": cliff}
                    for goal, cliff in LONG_DISTANCE_PAIRS
                ],
                "double_cliff_widths": list(DOUBLE_CLIFF_WIDTHS),
                "goal_sides": list(SIDES),
            },
            "launch_argv": argv,
            "launch_command": _shell_join(argv),
        },
    )
    print(output_dir)
    return 0


def command_print_state_space_edges_deformation_off_launch_command(args: argparse.Namespace) -> int:
    print(_shell_join(state_space_edges_deformation_off_launch_argv(analysis_root=str(args.analysis_root))))
    return 0


def command_write_low_cost_native_launch_manifest(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    item = state_space_specimen()
    argv = low_cost_native_launch_argv(
        analysis_root=str(args.analysis_root),
        gpu_profile=str(args.gpu_profile),
    )
    write_json(
        output_dir / "launch_manifest_low_cost_native.json",
        {
            "schema_version": 1,
            "kind": "preprint_perfect_native_low_cost_launch_manifest",
            "analysis_root": str(args.analysis_root),
            "gpu_profile": str(args.gpu_profile),
            "assays": ["native_core"],
            "surfaces": ["native_core"],
            "step_budgets": {"native_core": STEP_BUDGETS["native_core"]},
            "specimen": item,
            "launch_argv": argv,
            "launch_command": _shell_join(argv),
        },
    )
    print(output_dir)
    return 0


def command_print_low_cost_native_launch_command(args: argparse.Namespace) -> int:
    print(_shell_join(low_cost_native_launch_argv(analysis_root=str(args.analysis_root), gpu_profile=str(args.gpu_profile))))
    return 0


def command_validate_body_transfer_configs(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(REPO_ROOT / "sensing_edges"))
    sys.path.insert(0, str(REPO_ROOT / "src"))

    from omegaconf import OmegaConf
    from sensing_edges.runtime_config import compose_experiment_config, logical_config_sha256, merge_stored_config

    config_dir = REPO_ROOT / "sensing_edges" / "config"
    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    for item in SPECIMENS:
        local_config_path = REPO_ROOT / item["local_config_path"]
        if not local_config_path.exists():
            failures.append(f"Missing local config: {local_config_path}")
            continue
        stored_cfg = OmegaConf.load(local_config_path)
        preset = str(stored_cfg.get("preset_id") or "config")
        base_cfg = compose_experiment_config(preset=preset, config_dir=config_dir)
        cfg = merge_stored_config(base_cfg, stored_cfg, prefer_current_post_run=True)
        cfg = OmegaConf.merge(
            cfg,
            OmegaConf.from_dotlist(
                body_transfer_override_tokens(
                    specimens_output_root=item["specimens_output_root"],
                    analysis_root=str(args.analysis_root),
                )
            ),
        )
        post = cfg.eval.post_run
        body_specs = [dict(body) for body in list(post.body_transfer_bodies)]
        transfer_labels = [str(body["label"]) for body in body_specs]
        checks = {
            "native_max_steps": int(post.native_max_steps) == STEP_BUDGETS["native_core"],
            "signaling_surfaces_native_only": list(post.signaling_surfaces) == ["native_core"],
            "render_bodies": list(post.body_transfer_render_bodies) == ["reference", "rect_6x2", "rect_8x2", "rect_7x3"],
            "transfer_body_labels": transfer_labels == ["rect_6x2", "rect_8x2", "rect_7x3"],
        }
        for check_name, passed in checks.items():
            if not passed:
                failures.append(f"{item['specimen_id']} failed {check_name}")
        rows.append(
            {
                "group_id": item["group_id"],
                "run_id": item["run_id"],
                "specimen_id": item["specimen_id"],
                "logical_config_sha256": logical_config_sha256(cfg),
                "native_max_steps": int(post.native_max_steps),
                "body_transfer_bodies": transfer_labels,
                "body_transfer_render_bodies": list(post.body_transfer_render_bodies),
            }
        )
    payload = {
        "valid": not failures,
        "analysis_root": str(args.analysis_root),
        "failures": failures,
        "rows": rows,
    }
    print(json.dumps(payload, indent=2))
    return 1 if failures else 0


def command_validate_state_space_config(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(REPO_ROOT / "sensing_edges"))
    sys.path.insert(0, str(REPO_ROOT / "src"))

    from omegaconf import OmegaConf
    from sensing_edges.runtime_config import compose_experiment_config, logical_config_sha256, merge_stored_config

    config_dir = REPO_ROOT / "sensing_edges" / "config"
    item = state_space_specimen()
    local_config_path = REPO_ROOT / item["local_config_path"]
    failures: list[str] = []
    stored_cfg = OmegaConf.load(local_config_path)
    preset = str(stored_cfg.get("preset_id") or "config")
    base_cfg = compose_experiment_config(preset=preset, config_dir=config_dir)
    cfg = merge_stored_config(base_cfg, stored_cfg, prefer_current_post_run=True)
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.from_dotlist(
            state_space_override_tokens(
                specimens_output_root=item["specimens_output_root"],
                analysis_root=str(args.analysis_root),
                deformation_condition=str(args.deformation_condition),
                signaling_condition=str(args.signaling_condition),
                state_space_surfaces=surface_ids(str(args.surface_preset)),
            )
        ),
    )
    post = cfg.eval.post_run
    if int(post.native_max_steps) != STEP_BUDGETS["native_core"]:
        failures.append("native_max_steps mismatch")
    if int(post.double_cliff_max_steps) != STEP_BUDGETS["double_cliff"]:
        failures.append("double_cliff_max_steps mismatch")
    pair_budgets = {
        (int(row["goal_distance"]), int(row["cliff_distance"])): int(row["max_steps"])
        for row in OmegaConf.to_container(post.long_distance_pair_max_steps, resolve=True)
    }
    if pair_budgets.get((8, 4)) != STEP_BUDGETS["long_distance_8_4"]:
        failures.append("long_distance_8_4 max_steps mismatch")
    if pair_budgets.get((10, 5)) != STEP_BUDGETS["long_distance_10_5"]:
        failures.append("long_distance_10_5 max_steps mismatch")
    configured_surfaces = tuple(str(value) for value in OmegaConf.to_container(post.state_space_surfaces, resolve=True))
    expected_surfaces = surface_ids(str(args.surface_preset))
    if configured_surfaces != expected_surfaces:
        failures.append("state_space_surfaces mismatch")
    if str(post.state_space_deformation_condition) != str(args.deformation_condition):
        failures.append("state_space_deformation_condition mismatch")
    if str(post.state_space_signaling_condition) != str(args.signaling_condition):
        failures.append("state_space_signaling_condition mismatch")
    if int(cfg.sim.physics_substeps) % int(cfg.sim.sense_act_phases_per_env_step) != 0:
        failures.append("physics_substeps is not divisible by sense_act_phases_per_env_step")
    payload = {
        "valid": not failures,
        "analysis_root": str(args.analysis_root),
        "failures": failures,
        "state_space_specimen": item,
        "logical_config_sha256": logical_config_sha256(cfg),
        "native_max_steps": int(post.native_max_steps),
        "double_cliff_max_steps": int(post.double_cliff_max_steps),
        "long_distance_pair_max_steps": OmegaConf.to_container(post.long_distance_pair_max_steps, resolve=True),
        "state_space_surfaces": list(configured_surfaces),
        "state_space_deformation_condition": str(post.state_space_deformation_condition),
        "state_space_signaling_condition": str(post.state_space_signaling_condition),
        "native_pairs": OmegaConf.to_container(post.native_pairs, resolve=True),
        "sense_act_phases_per_env_step": int(cfg.sim.sense_act_phases_per_env_step),
        "physics_substeps": int(cfg.sim.physics_substeps),
        "nca_steps_per_phase": int(cfg.nca.nca_steps_per_phase),
    }
    print(json.dumps(payload, indent=2))
    return 1 if failures else 0


def command_validate_configs(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(REPO_ROOT / "sensing_edges"))
    sys.path.insert(0, str(REPO_ROOT / "src"))

    from omegaconf import OmegaConf
    from sensing_edges.post_run import compatible_post_run_signaling_conditions
    from sensing_edges.runtime_config import compose_experiment_config, logical_config_sha256, merge_stored_config

    config_dir = REPO_ROOT / "sensing_edges" / "config"
    surfaces = surface_ids(args.surface_preset)
    failures: list[str] = []
    group_signatures: dict[str, set[str]] = {group_id: set() for group_id in group_ids()}
    rows: list[dict[str, Any]] = []
    for item in SPECIMENS:
        local_config_path = REPO_ROOT / item["local_config_path"]
        if not local_config_path.exists():
            failures.append(f"Missing local config: {local_config_path}")
            continue
        stored_cfg = OmegaConf.load(local_config_path)
        preset = str(stored_cfg.get("preset_id") or "config")
        base_cfg = compose_experiment_config(preset=preset, config_dir=config_dir)
        cfg = merge_stored_config(base_cfg, stored_cfg, prefer_current_post_run=True)
        cfg = OmegaConf.merge(
            cfg,
            OmegaConf.from_dotlist(
                override_tokens(
                    specimens_output_root=item["specimens_output_root"],
                    surfaces=surfaces,
                )
            ),
        )
        post = cfg.eval.post_run
        signaling_conditions = list(compatible_post_run_signaling_conditions(cfg))
        deformation_conditions = [str(value) for value in list(post.deformation_conditions)]
        long_distance_budget = {
            (int(entry.get("goal_distance", 0)), int(entry.get("cliff_distance", 0))): int(entry.get("max_steps", 0))
            for entry in list(post.long_distance_pair_max_steps)
        }
        signature = logical_config_sha256(cfg)
        group_signatures[item["group_id"]].add(signature)
        checks = {
            "native_max_steps": int(post.native_max_steps) == STEP_BUDGETS["native_core"],
            "max_steps": int(post.max_steps) == STEP_BUDGETS["long_distance_10_5"],
            "double_cliff_max_steps": int(post.double_cliff_max_steps) == STEP_BUDGETS["double_cliff"],
            "long_distance_8_4_steps": long_distance_budget.get((8, 4)) == STEP_BUDGETS["long_distance_8_4"],
            "long_distance_10_5_steps": long_distance_budget.get((10, 5)) == STEP_BUDGETS["long_distance_10_5"],
            "signaling_surfaces": list(post.signaling_surfaces) == list(surfaces),
            "deformation_surfaces": list(post.deformation_surfaces) == list(surfaces),
            "signaling_conditions": signaling_conditions == list(SIGNALING_CONDITIONS),
            "signaling_render_conditions": list(post.signaling_render_conditions) == list(SIGNALING_CONDITIONS),
            "deformation_conditions": deformation_conditions == list(DEFORMATION_CONDITIONS),
            "deformation_render_conditions": list(post.deformation_render_conditions) == list(DEFORMATION_CONDITIONS),
        }
        for check_name, passed in checks.items():
            if not passed:
                failures.append(f"{item['specimen_id']} failed {check_name}")
        rows.append(
            {
                "group_id": item["group_id"],
                "run_id": item["run_id"],
                "specimen_id": item["specimen_id"],
                "preset": preset,
                "logical_config_sha256": signature,
                "train_max_steps": int(cfg.train.max_steps),
                "native_max_steps": int(post.native_max_steps),
                "expected_signaling_rows": expected_rows_per_specimen(surfaces=surfaces, assay="signaling_ablation"),
                "expected_deformation_rows": expected_rows_per_specimen(surfaces=surfaces, assay="deformation_ablation"),
            }
        )
    for group_id, signatures in group_signatures.items():
        if len(signatures) != 1:
            failures.append(f"{group_id} has {len(signatures)} logical signatures")

    payload = {
        "valid": not failures,
        "surface_preset": args.surface_preset,
        "surfaces": list(surfaces),
        "failures": failures,
        "rows": rows,
        "group_signature_prefixes": {
            group_id: sorted(signature[:12] for signature in signatures)
            for group_id, signatures in group_signatures.items()
        },
    }
    print(json.dumps(payload, indent=2))
    return 1 if failures else 0


def _assay_conditions(assay: str) -> tuple[str, ...]:
    if assay == "signaling_ablation":
        return SIGNALING_CONDITIONS
    if assay == "deformation_ablation":
        return DEFORMATION_CONDITIONS
    if assay == "body_transfer":
        return BODY_TRANSFER_BODIES
    raise ValueError(f"Unsupported assay: {assay}")


def _condition_column(assay: str) -> str:
    return "signaling_condition" if assay == "signaling_ablation" else "deformation_condition"


def _copy_render(
    src: Path,
    by_specimen_root: Path,
    by_condition_root: Path,
    *,
    specimen_id: str,
    assay: str,
    surface_id: str,
    condition: str,
) -> tuple[str, str]:
    surface_label = SURFACE_LABELS.get(surface_id, surface_id)
    specimen_dest = by_specimen_root / specimen_id / assay / surface_label / condition / src.name
    condition_dest = by_condition_root / assay / surface_label / condition / f"{specimen_id}__{src.name}"
    specimen_dest.parent.mkdir(parents=True, exist_ok=True)
    condition_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, specimen_dest)
    shutil.copy2(src, condition_dest)
    return str(specimen_dest), str(condition_dest)


def command_consolidate(args: argparse.Namespace) -> int:
    import pandas as pd

    analysis_roots = [Path(path).resolve() for path in args.analysis_root]
    primary_analysis_root = analysis_roots[0]
    output_root = Path(args.output_root).resolve()
    tables_dir = output_root / "tables"
    renders_by_specimen = output_root / "renders" / "by_specimen"
    renders_by_condition = output_root / "renders" / "by_condition"
    surfaces = surface_ids(args.surface_preset)
    if tables_dir.exists():
        shutil.rmtree(tables_dir)
    renders_root = output_root / "renders"
    if renders_root.exists():
        shutil.rmtree(renders_root)
    tables_dir.mkdir(parents=True, exist_ok=True)

    episode_frames: list[pd.DataFrame] = []
    condition_frames: list[pd.DataFrame] = []
    provenance_rows: list[dict[str, Any]] = []
    completeness_rows: list[dict[str, Any]] = []
    render_rows: list[dict[str, Any]] = []

    def surface_root_for(item: dict[str, str], assay: str, surface_id: str) -> Path:
        for root in analysis_roots:
            candidate = root / item["run_id"] / item["specimen_id"] / assay / "surfaces" / surface_id
            if candidate.exists():
                return candidate
        return primary_analysis_root / item["run_id"] / item["specimen_id"] / assay / "surfaces" / surface_id

    for item in SPECIMENS:
        specimen_root = primary_analysis_root / item["run_id"] / item["specimen_id"]
        provenance_rows.append(
            {
                "group_id": item["group_id"],
                "run_id": item["run_id"],
                "specimen_id": item["specimen_id"],
                "analysis_root": str(specimen_root),
                "analysis_roots": json.dumps([str(root / item["run_id"] / item["specimen_id"]) for root in analysis_roots]),
                "remote_run_dir": item["remote_run_dir"],
                "remote_config_path": item["remote_config_path"],
                "specimens_output_root": item["specimens_output_root"],
                "local_config_path": str(REPO_ROOT / item["local_config_path"]),
            }
        )
        assays = ["signaling_ablation", "deformation_ablation"]
        if bool(args.include_body_transfer):
            assays.append("body_transfer")
        for assay in assays:
            assay_surfaces = ("native_core",) if assay == "body_transfer" else surfaces
            for surface_id in assay_surfaces:
                surface_root = surface_root_for(item, assay, surface_id)
                per_episode_path = surface_root / "per_episode.csv"
                expected_names = expected_render_names_for_surface(surface_id)
                expected_rows = len(_assay_conditions(assay)) * len(expected_names)
                actual_rows = 0
                if per_episode_path.exists():
                    df = pd.read_csv(per_episode_path)
                    actual_rows = int(len(df))
                    df["group_id"] = item["group_id"]
                    df["source_run_id"] = item["run_id"]
                    df["source_specimen_id"] = item["specimen_id"]
                    df["assay_family"] = assay
                    df["assay"] = assay
                    df["surface_id"] = surface_id
                    df["surface_package_label"] = SURFACE_LABELS.get(surface_id, surface_id)
                    episode_frames.append(df)
                condition_summary_path = surface_root / "stats" / "condition_summary.csv"
                if condition_summary_path.exists():
                    cdf = pd.read_csv(condition_summary_path)
                    cdf["group_id"] = item["group_id"]
                    cdf["source_run_id"] = item["run_id"]
                    cdf["source_specimen_id"] = item["specimen_id"]
                    cdf["assay_family"] = assay
                    cdf["assay"] = assay
                    cdf["surface_id"] = surface_id
                    cdf["surface_package_label"] = SURFACE_LABELS.get(surface_id, surface_id)
                    condition_frames.append(cdf)

                expected_render_count = 0
                actual_render_count = 0
                for condition in _assay_conditions(assay):
                    render_dir = (
                        surface_root / "bodies" / condition / "renders"
                        if assay == "body_transfer"
                        else surface_root / "conditions" / condition / "renders"
                    )
                    for render_name in expected_names:
                        expected_render_count += 1
                        src = render_dir / render_name
                        exists = src.exists()
                        if exists:
                            actual_render_count += 1
                            by_specimen_path, by_condition_path = _copy_render(
                                src,
                                renders_by_specimen,
                                renders_by_condition,
                                specimen_id=item["specimen_id"],
                                assay=assay,
                                surface_id=surface_id,
                                condition=condition,
                            )
                        else:
                            by_specimen_path = ""
                            by_condition_path = ""
                        render_rows.append(
                            {
                                "group_id": item["group_id"],
                                "run_id": item["run_id"],
                                "specimen_id": item["specimen_id"],
                                "assay": assay,
                                "surface_id": surface_id,
                                "surface_package_label": SURFACE_LABELS.get(surface_id, surface_id),
                                "condition": condition,
                                "render_name": render_name,
                                "source_path": str(src),
                                "exists": bool(exists),
                                "by_specimen_path": by_specimen_path,
                                "by_condition_path": by_condition_path,
                            }
                        )
                completeness_rows.append(
                    {
                        "group_id": item["group_id"],
                        "run_id": item["run_id"],
                        "specimen_id": item["specimen_id"],
                        "assay": assay,
                        "surface_id": surface_id,
                        "surface_package_label": SURFACE_LABELS.get(surface_id, surface_id),
                        "expected_rows": expected_rows,
                        "actual_rows": actual_rows,
                        "expected_renders": expected_render_count,
                        "actual_renders": actual_render_count,
                        "complete": bool(actual_rows == expected_rows and actual_render_count == expected_render_count),
                    }
                )

    episode_df = pd.concat(episode_frames, ignore_index=True) if episode_frames else pd.DataFrame()
    condition_df = pd.concat(condition_frames, ignore_index=True) if condition_frames else pd.DataFrame()
    provenance_df = pd.DataFrame(provenance_rows)
    completeness_df = pd.DataFrame(completeness_rows)
    render_df = pd.DataFrame(render_rows)

    episode_df.to_csv(tables_dir / "episode_manifest.csv", index=False)
    condition_df.to_csv(tables_dir / "suite_summary.csv", index=False)
    provenance_df.to_csv(tables_dir / "provenance.csv", index=False)
    completeness_df.to_csv(tables_dir / "completeness.csv", index=False)
    render_df.to_csv(tables_dir / "render_manifest.csv", index=False)

    manifest = {
        "schema_version": 1,
        "kind": "preprint_native_core_ablation_package",
        "analysis_root": str(primary_analysis_root),
        "analysis_roots": [str(root) for root in analysis_roots],
        "output_root": str(output_root),
        "surface_preset": args.surface_preset,
        "surfaces": list(surfaces),
        "surface_labels": {surface_id: SURFACE_LABELS[surface_id] for surface_id in surfaces},
        "step_budgets": STEP_BUDGETS,
        "include_body_transfer": bool(args.include_body_transfer),
        "expected": {
            **expected_totals_with_body_transfer(
                surfaces=surfaces,
                include_body_transfer=bool(args.include_body_transfer),
            ),
        },
        "actual": {
            "total_rows": int(len(episode_df)),
            "total_renders": int(render_df["exists"].sum()) if not render_df.empty else 0,
            "complete": bool(not completeness_df.empty and completeness_df["complete"].all()),
        },
        "tables": {
            "episode_manifest": str(tables_dir / "episode_manifest.csv"),
            "suite_summary": str(tables_dir / "suite_summary.csv"),
            "provenance": str(tables_dir / "provenance.csv"),
            "completeness": str(tables_dir / "completeness.csv"),
            "render_manifest": str(tables_dir / "render_manifest.csv"),
        },
    }
    write_json(output_root / "run_manifest.json", manifest)
    print(json.dumps(manifest["actual"], indent=2))
    return 0 if manifest["actual"]["complete"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and package the preprint native-core ablation rerun.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_batches = subparsers.add_parser("write-batches")
    write_batches.add_argument("--output-dir", type=Path, default=DEFAULT_BATCH_DIR)
    write_batches.add_argument("--surface-preset", choices=sorted(SURFACE_PRESETS), default="native_core")
    write_batches.set_defaults(func=command_write_batches)

    body_manifest = subparsers.add_parser("write-body-transfer-launch-manifest")
    body_manifest.add_argument("--output-dir", type=Path, default=DEFAULT_BATCH_DIR)
    body_manifest.add_argument("--analysis-root", default=DEFAULT_BODY_TRANSFER_ANALYSIS_ROOT)
    body_manifest.set_defaults(func=command_write_body_transfer_launch_manifest)

    body_launch = subparsers.add_parser("print-body-transfer-launch-commands")
    body_launch.add_argument("--analysis-root", default=DEFAULT_BODY_TRANSFER_ANALYSIS_ROOT)
    body_launch.set_defaults(func=command_print_body_transfer_launch_commands)

    state_manifest = subparsers.add_parser("write-state-space-launch-manifest")
    state_manifest.add_argument("--output-dir", type=Path, default=DEFAULT_BATCH_DIR)
    state_manifest.add_argument("--analysis-root", default=DEFAULT_STATE_SPACE_ANALYSIS_ROOT)
    state_manifest.set_defaults(func=command_write_state_space_launch_manifest)

    state_launch = subparsers.add_parser("print-state-space-launch-command")
    state_launch.add_argument("--analysis-root", default=DEFAULT_STATE_SPACE_ANALYSIS_ROOT)
    state_launch.set_defaults(func=command_print_state_space_launch_command)

    state_def_off_manifest = subparsers.add_parser("write-state-space-deformation-off-launch-manifest")
    state_def_off_manifest.add_argument("--output-dir", type=Path, default=DEFAULT_BATCH_DIR)
    state_def_off_manifest.add_argument(
        "--analysis-root",
        default=DEFAULT_STATE_SPACE_DEFORMATION_OFF_ANALYSIS_ROOT,
    )
    state_def_off_manifest.set_defaults(func=command_write_state_space_deformation_off_launch_manifest)

    state_def_off_launch = subparsers.add_parser("print-state-space-deformation-off-launch-command")
    state_def_off_launch.add_argument(
        "--analysis-root",
        default=DEFAULT_STATE_SPACE_DEFORMATION_OFF_ANALYSIS_ROOT,
    )
    state_def_off_launch.set_defaults(func=command_print_state_space_deformation_off_launch_command)

    state_edges_def_off_manifest = subparsers.add_parser("write-state-space-edges-deformation-off-launch-manifest")
    state_edges_def_off_manifest.add_argument("--output-dir", type=Path, default=DEFAULT_BATCH_DIR)
    state_edges_def_off_manifest.add_argument(
        "--analysis-root",
        default=DEFAULT_STATE_SPACE_EDGES_DEFORMATION_OFF_ANALYSIS_ROOT,
    )
    state_edges_def_off_manifest.set_defaults(func=command_write_state_space_edges_deformation_off_launch_manifest)

    state_edges_def_off_launch = subparsers.add_parser("print-state-space-edges-deformation-off-launch-command")
    state_edges_def_off_launch.add_argument(
        "--analysis-root",
        default=DEFAULT_STATE_SPACE_EDGES_DEFORMATION_OFF_ANALYSIS_ROOT,
    )
    state_edges_def_off_launch.set_defaults(func=command_print_state_space_edges_deformation_off_launch_command)

    low_cost_manifest = subparsers.add_parser("write-low-cost-native-launch-manifest")
    low_cost_manifest.add_argument("--output-dir", type=Path, default=DEFAULT_BATCH_DIR)
    low_cost_manifest.add_argument("--analysis-root", default=DEFAULT_LOW_COST_NATIVE_ANALYSIS_ROOT)
    low_cost_manifest.add_argument("--gpu-profile", default="l4", choices=("l4", "t4", "a100"))
    low_cost_manifest.set_defaults(func=command_write_low_cost_native_launch_manifest)

    low_cost_launch = subparsers.add_parser("print-low-cost-native-launch-command")
    low_cost_launch.add_argument("--analysis-root", default=DEFAULT_LOW_COST_NATIVE_ANALYSIS_ROOT)
    low_cost_launch.add_argument("--gpu-profile", default="l4", choices=("l4", "t4", "a100"))
    low_cost_launch.set_defaults(func=command_print_low_cost_native_launch_command)

    validate = subparsers.add_parser("validate-configs")
    validate.add_argument("--surface-preset", choices=sorted(SURFACE_PRESETS), default="native_core")
    validate.set_defaults(func=command_validate_configs)

    validate_body = subparsers.add_parser("validate-body-transfer-configs")
    validate_body.add_argument("--analysis-root", default=DEFAULT_BODY_TRANSFER_ANALYSIS_ROOT)
    validate_body.set_defaults(func=command_validate_body_transfer_configs)

    validate_state = subparsers.add_parser("validate-state-space-config")
    validate_state.add_argument("--analysis-root", default=DEFAULT_STATE_SPACE_ANALYSIS_ROOT)
    validate_state.add_argument("--deformation-condition", default="intact")
    validate_state.add_argument("--signaling-condition", default="intact")
    validate_state.add_argument("--surface-preset", choices=sorted(SURFACE_PRESETS), default="native_core")
    validate_state.set_defaults(func=command_validate_state_space_config)

    launch = subparsers.add_parser("print-launch-commands")
    launch.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    launch.add_argument("--surface-preset", choices=sorted(SURFACE_PRESETS), default="native_core")
    launch.set_defaults(func=command_print_launch_commands)

    consolidate = subparsers.add_parser("consolidate")
    consolidate.add_argument("--analysis-root", type=Path, nargs="+", required=True)
    consolidate.add_argument("--output-root", type=Path, default=DEFAULT_LOCAL_PACKAGE_ROOT)
    consolidate.add_argument("--surface-preset", choices=sorted(SURFACE_PRESETS), default="native_core")
    consolidate.add_argument("--include-body-transfer", action="store_true")
    consolidate.set_defaults(func=command_consolidate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
