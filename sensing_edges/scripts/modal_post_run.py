"""Modal entrypoint for specimen-centric post-run assays."""

from __future__ import annotations

import json
from pathlib import Path
import modal

from _modal_common import VOLUMES, app, commit_volumes, load_config, parse_override_csv, se_image

_DEFAULT_POST_RUN_GPU = "A100-40GB"
_A100_80GB_POST_RUN_GPU = "A100-80GB"
_H100_POST_RUN_GPU = "H100!"
_LOW_COST_POST_RUN_GPU = "L4"
_T4_POST_RUN_GPU = "T4"
_ONE_SPECIMEN_POST_RUN_TIMEOUT = 8 * 3600


def _compose_post_run_cfg(*, preset: str, resolved_config_path: str, overrides: list[str] | None = None):
    from omegaconf import OmegaConf
    from sensing_edges.runtime_config import merge_stored_config

    stored_cfg = OmegaConf.load(resolved_config_path)
    base_preset = str(stored_cfg.get("preset_id") or preset)
    base_cfg = OmegaConf.create(OmegaConf.to_container(load_config(config_name=base_preset), resolve=False))
    cfg = merge_stored_config(base_cfg, stored_cfg, prefer_current_post_run=True)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def _compose_batched_post_run_cfg(
    *,
    run_specs: list[dict[str, str]],
    preset: str,
    overrides: list[str] | None = None,
):
    from sensing_edges.runtime_config import logical_config_sha256

    composed: list[tuple[str, str, object]] = []
    for index, run_spec in enumerate(run_specs):
        config_path = str(run_spec.get("config_path") or "")
        if not config_path:
            raise ValueError(f"Each batched run spec requires config_path; spec {index} was missing it")
        cfg = _compose_post_run_cfg(
            preset=preset,
            resolved_config_path=config_path,
            overrides=overrides,
        )
        composed.append((config_path, logical_config_sha256(cfg), cfg))
    reference_path, reference_signature, reference_cfg = composed[0]
    mismatched = [
        f"{config_path} ({signature})"
        for config_path, signature, _cfg in composed[1:]
        if signature != reference_signature
    ]
    if mismatched:
        raise ValueError(
            "Batched post-run requires logically compatible configs. "
            f"Reference: {reference_path} ({reference_signature}); mismatched: "
            + ", ".join(mismatched)
        )
    return reference_cfg


def _post_run_remote_impl(
    *,
    run_dir: str,
    specimen_id: str | None = None,
    specimen_rank: int | None = None,
    preset: str = "config",
    overrides: list[str] | None = None,
    config_path: str | None = None,
    assays: list[str] | None = None,
):
    import os
    from pathlib import Path

    os.chdir("/root/experiment")
    from sensing_edges.logging_utils import configure_logging
    from sensing_edges.post_run import resolve_stored_specimen_root, run_post_run_for_stored_specimen
    from sensing_edges.runtime_config import resolve_runtime_config

    configure_logging("INFO")
    resolved_config_path = config_path or str(Path(run_dir) / "resolved_config.yaml")
    cfg = _compose_post_run_cfg(
        preset=preset,
        resolved_config_path=resolved_config_path,
        overrides=overrides,
    )
    runtime_resolution = resolve_runtime_config(cfg)
    specimen_root = resolve_stored_specimen_root(
        run_dir,
        output_root=str(cfg.specimens.get("output_root", "specimens")),
    )
    try:
        results = run_post_run_for_stored_specimen(
            cfg=cfg,
            runtime_resolution=runtime_resolution,
            run_dir=run_dir,
            specimen_root=specimen_root,
            specimen_id=specimen_id,
            specimen_rank=specimen_rank,
            assays=(None if not assays else set(assays)),
            on_progress=commit_volumes,
        )
        return {
            "specimen_id": results.specimen_id,
            "specimen_dir": str(results.specimen_dir),
            "index_path": str(results.index_path),
            "assays": dict(results.assay_dirs),
        }
    finally:
        commit_volumes()


@app.function(
    image=se_image,
    gpu=_DEFAULT_POST_RUN_GPU,
    timeout=_ONE_SPECIMEN_POST_RUN_TIMEOUT,
    volumes=VOLUMES,
)
def post_run_remote(
    run_dir: str,
    specimen_id: str | None = None,
    specimen_rank: int | None = None,
    preset: str = "config",
    overrides: list[str] | None = None,
    config_path: str | None = None,
    assays: list[str] | None = None,
):
    """Run the fixed post-run assay battery for one stored specimen on Modal."""
    return _post_run_remote_impl(
        run_dir=run_dir,
        specimen_id=specimen_id,
        specimen_rank=specimen_rank,
        preset=preset,
        overrides=overrides,
        config_path=config_path,
        assays=assays,
    )


@app.function(
    image=se_image,
    gpu=_LOW_COST_POST_RUN_GPU,
    timeout=_ONE_SPECIMEN_POST_RUN_TIMEOUT,
    volumes=VOLUMES,
)
def post_run_remote_l4(
    run_dir: str,
    specimen_id: str | None = None,
    specimen_rank: int | None = None,
    preset: str = "config",
    overrides: list[str] | None = None,
    config_path: str | None = None,
    assays: list[str] | None = None,
):
    """Run a one-specimen post-run assay battery on a lower-cost L4 GPU."""
    return _post_run_remote_impl(
        run_dir=run_dir,
        specimen_id=specimen_id,
        specimen_rank=specimen_rank,
        preset=preset,
        overrides=overrides,
        config_path=config_path,
        assays=assays,
    )


@app.function(
    image=se_image,
    gpu=_A100_80GB_POST_RUN_GPU,
    timeout=_ONE_SPECIMEN_POST_RUN_TIMEOUT,
    volumes=VOLUMES,
)
def post_run_remote_a100_80gb(
    run_dir: str,
    specimen_id: str | None = None,
    specimen_rank: int | None = None,
    preset: str = "config",
    overrides: list[str] | None = None,
    config_path: str | None = None,
    assays: list[str] | None = None,
):
    """Run a one-specimen post-run assay battery on an A100-80GB GPU."""
    return _post_run_remote_impl(
        run_dir=run_dir,
        specimen_id=specimen_id,
        specimen_rank=specimen_rank,
        preset=preset,
        overrides=overrides,
        config_path=config_path,
        assays=assays,
    )


@app.function(
    image=se_image,
    gpu=_H100_POST_RUN_GPU,
    timeout=_ONE_SPECIMEN_POST_RUN_TIMEOUT,
    volumes=VOLUMES,
)
def post_run_remote_h100(
    run_dir: str,
    specimen_id: str | None = None,
    specimen_rank: int | None = None,
    preset: str = "config",
    overrides: list[str] | None = None,
    config_path: str | None = None,
    assays: list[str] | None = None,
):
    """Run a one-specimen post-run assay battery on an H100 GPU."""
    return _post_run_remote_impl(
        run_dir=run_dir,
        specimen_id=specimen_id,
        specimen_rank=specimen_rank,
        preset=preset,
        overrides=overrides,
        config_path=config_path,
        assays=assays,
    )


@app.function(
    image=se_image,
    gpu=_T4_POST_RUN_GPU,
    timeout=_ONE_SPECIMEN_POST_RUN_TIMEOUT,
    volumes=VOLUMES,
)
def post_run_remote_t4(
    run_dir: str,
    specimen_id: str | None = None,
    specimen_rank: int | None = None,
    preset: str = "config",
    overrides: list[str] | None = None,
    config_path: str | None = None,
    assays: list[str] | None = None,
):
    """Run a one-specimen post-run assay battery on a T4 GPU."""
    return _post_run_remote_impl(
        run_dir=run_dir,
        specimen_id=specimen_id,
        specimen_rank=specimen_rank,
        preset=preset,
        overrides=overrides,
        config_path=config_path,
        assays=assays,
    )


@app.function(
    image=se_image,
    gpu=_DEFAULT_POST_RUN_GPU,
    timeout=4 * 3600,
    volumes=VOLUMES,
)
def post_run_batched_remote(
    run_specs: list[dict[str, str]],
    preset: str = "config",
    overrides: list[str] | None = None,
    assays: list[str] | None = None,
    batch_label: str | None = None,
):
    """Run batched ablation assays for multiple stored specimens on one Modal GPU."""
    import os
    from pathlib import Path

    if not run_specs:
        raise ValueError("post_run_batched_remote requires at least one run spec")

    os.chdir("/root/experiment")
    from sensing_edges.logging_utils import configure_logging
    from sensing_edges.post_run import run_batched_ablation_post_run_for_stored_specimens
    from sensing_edges.runtime_config import resolve_runtime_config

    configure_logging("INFO")
    cfg = _compose_batched_post_run_cfg(
        run_specs=run_specs,
        preset=preset,
        overrides=overrides,
    )
    runtime_resolution = resolve_runtime_config(cfg)
    try:
        results = run_batched_ablation_post_run_for_stored_specimens(
            cfg=cfg,
            runtime_resolution=runtime_resolution,
            run_specs=run_specs,
            assays=(None if not assays else set(assays)),
            batch_label=batch_label,
        )
        return results
    finally:
        commit_volumes()


@app.local_entrypoint()
def main(
    run_dir: str,
    specimen_id: str | None = None,
    specimen_rank: int | None = None,
    preset: str = "config",
    config_path: str | None = None,
    override: str | None = None,
    assays: str | None = None,
    detach_launch: bool = False,
    batch_jobs_path: str | None = None,
    batch_label: str | None = None,
    gpu_profile: str = "a100",
):
    normalized_gpu_profile = str(gpu_profile).strip().lower()
    aliases = {
        "a100": "a100-40gb",
        "a100-40gb": "a100-40gb",
        "a100_40gb": "a100-40gb",
        "a100-80gb": "a100-80gb",
        "a100_80gb": "a100-80gb",
        "h100": "h100",
        "l4": "l4",
        "t4": "t4",
    }
    normalized_gpu_profile = aliases.get(normalized_gpu_profile, normalized_gpu_profile)
    if normalized_gpu_profile not in {"a100-40gb", "a100-80gb", "h100", "l4", "t4"}:
        raise ValueError("gpu_profile must be one of: a100-40gb, a100-80gb, h100, l4, t4")
    if batch_jobs_path:
        if normalized_gpu_profile != "a100-40gb":
            raise ValueError("gpu_profile is only supported for non-batched post-run calls")
        batch_jobs = json.loads(Path(batch_jobs_path).read_text())
        remote_kwargs = {
            "run_specs": batch_jobs,
            "preset": preset,
            "overrides": parse_override_csv(override) or None,
            "assays": [token.strip() for token in assays.split(",") if token.strip()] if assays else None,
            "batch_label": batch_label,
        }
        if detach_launch:
            function_call = post_run_batched_remote.spawn(**remote_kwargs)
            print(f"\nSpawned batched post-run function call: {function_call.object_id}")
            return
        result = post_run_batched_remote.remote(**remote_kwargs)
        print(f"\nBatched post-run result: {result}")
        return

    remote_kwargs = {
        "run_dir": run_dir,
        "specimen_id": specimen_id,
        "specimen_rank": specimen_rank,
        "preset": preset,
        "config_path": config_path,
        "overrides": parse_override_csv(override) or None,
        "assays": [token.strip() for token in assays.split(",") if token.strip()] if assays else None,
    }
    remote_function = {
        "a100-40gb": post_run_remote,
        "a100-80gb": post_run_remote_a100_80gb,
        "h100": post_run_remote_h100,
        "l4": post_run_remote_l4,
        "t4": post_run_remote_t4,
    }[normalized_gpu_profile]
    if detach_launch:
        function_call = remote_function.spawn(**remote_kwargs)
        print(f"\nSpawned post-run function call ({normalized_gpu_profile}): {function_call.object_id}")
        return

    result = remote_function.remote(**remote_kwargs)
    print(f"\nPost-run result ({normalized_gpu_profile}): {result}")
